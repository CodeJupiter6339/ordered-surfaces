"""Balayage 3 : robustesse.
(1) Validation croisee : reglages choisis sur la moitie des cubes, mesures sur l'autre moitie.
(2) Morceaux 'sales' : bouts de feuilles qui s'arretent dans le morceau (distracteurs), score sur les feuilles completes.
"""
import os, sys, json, time, zlib, itertools, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_bench as fb
DIRTY = os.path.expanduser("~/vesuvius/spike/bench_cache_dirty.npz")


def build_dirty():
    from fast_ordered_notrich import prep, CUBES
    with Pool(14) as pool: preps = pool.map(prep, CUBES)
    Ls, Vs, names = [], [], []
    for f, p in zip(CUBES, preps):
        if p is None: continue
        lab, vol, _ = p
        for x0 in range(0, 128 - 32 + 1, 16):
            for y0 in range(0, 128 - 32 + 1, 16):
                L = lab[x0:x0 + 32, y0:y0 + 32]
                ids = [i for i in np.unique(L) if i]
                pres = {i: (L == i).any(axis=2).mean() for i in ids}
                full = [i for i in ids if pres[i] >= 0.98]
                part = [i for i in ids if 0.01 < pres[i] < 0.98]
                if len(full) >= 2 and part:
                    Ls.append(L); Vs.append(vol[x0:x0 + 32, y0:y0 + 32]); names.append(f"{os.path.basename(f)[:17]}_{x0}_{y0}")
    np.savez_compressed(DIRTY, L=np.stack(Ls), V=np.stack(Vs), names=np.array(names))
    print("morceaux sales:", len(Ls), flush=True)


def evaluate(L, V, names, cfg, K_mode):
    rows = []
    for i in range(len(L)):
        keep, valid, contact = fb.truth(L[i])
        P = fb.signal(V[i], cfg.get("sig_z", 2), "ct")
        K = len(keep) if K_mode == "oracle" else fb.est_K(P, 9, cfg.get("prom", 1.0), "mode")
        S = fb.smooth_dp(P, K, cfg["dmin"], 60, cfg.get("iters", 0), cfg.get("lam", 0.02), cfg.get("size", 7))
        rows.append(fb.score(S, L[i], keep, valid, contact))
    return {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 2) for k in ("purete", "purete_contact", "couverture", "ordre")} | {
        "K_exact_%": round(100 * np.mean([r["dK"] == 0 for r in rows]), 1), "n": len(rows)}


def log(d):
    open(fb.LOG, "a").write(json.dumps(d) + "\n"); print(json.dumps(d), flush=True)


L, V, names = fb.load()
half = np.array([zlib.crc32(n[:17].encode()) % 2 for n in names])
grid = [{"dmin": d, "iters": it, "size": s, "prom": p} for d, it, s, p in itertools.product((9, 12), (4, 8), (7, 11), (0.6, 1.0, 1.4))]
for tune, test in ((0, 1), (1, 0)):
    a, b = half == tune, half == test
    scores = [(evaluate(L[a], V[a], [n for n, m in zip(names, a) if m], g, "est")["purete"], g) for g in grid]
    best = max(scores, key=lambda t: t[0])
    res = evaluate(L[b], V[b], [n for n, m in zip(names, b) if m], best[1], "est")
    log({"exp": f"G_crossval_tune{tune}_test{test}", "cfg": best[1], "tune_purete": best[0], **res})
if not os.path.exists(DIRTY): build_dirty()
d = np.load(DIRTY)
for cfg, km in (({"dmin": 9, "iters": 8, "size": 11}, "oracle"), ({"dmin": 9, "iters": 8, "size": 11, "prom": 1.0}, "est"),
                ({"dmin": 9, "iters": 0}, "oracle")):
    log({"exp": f"H_sales_{km}_it{cfg['iters']}", "cfg": cfg, **evaluate(d["L"], d["V"], list(d["names"]), cfg, km)})
# reference locale sur morceaux sales
from scipy import ndimage as ndi
rows = []
for i in range(len(d["L"])):
    Li = d["L"][i]; keep, valid, contact = fb.truth(Li); K = len(keep); m = Li > 0
    mc = ndi.binary_closing(m, structure=np.ones((1, 1, 5), bool)) | m
    S = np.zeros((K,) + m.shape[:2], int)
    for x in range(32):
        for y in range(32):
            r, n = ndi.label(mc[x, y]); cs = [int(np.mean(np.nonzero(r == j + 1)[0])) for j in range(n)]
            for k in range(K): S[k, x, y] = cs[k] if k < n else 0
    rows.append(fb.score(S, Li, keep, valid, contact))
log({"exp": "H_sales_reference_comptage", **{k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 2) for k in ("purete", "purete_contact", "couverture", "ordre")}, "n": len(rows)})
