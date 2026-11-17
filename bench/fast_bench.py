"""Banc d'essai rapide pour les surfaces ordonnees. Usage : python fast_bench.py [nom_experience]
1er lancement : prepare et met en cache les morceaux propres (cubes annotes du Scroll 1, 128^3, morceaux 32x32).
Ensuite chaque variante tourne en quelques secondes.

Mesures (honnetes) :
- purete   : pour chaque surface estimee, % des colonnes ou elle est dans sa feuille dominante (100 = jamais d'echange)
- couverture : % des vraies feuilles qui sont la feuille dominante d'au moins une surface
- doublons : surfaces estimees en trop qui partagent la meme feuille dominante (dedoublement)
- ordre    : % de paires de surfaces consecutives dont les feuilles dominantes sont dans le bon ordre
- K        : ecart entre K estime et K vrai
"""
import os, sys, json, time, glob, numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.expanduser("~/vesuvius/spike/bench_cache.npz")
LOG = os.path.expanduser("~/vesuvius/spike/bench_log.jsonl")
C = 32


def build_cache():
    import nrrd
    from fast_ordered_notrich import prep, CUBES
    with Pool(14) as pool:
        preps = pool.map(prep, CUBES)
    Ls, Vs, names = [], [], []
    for f, p in zip(CUBES, preps):
        if p is None: continue
        lab, vol, _ = p
        for x0 in range(0, 128 - C + 1, C // 2):
            for y0 in range(0, 128 - C + 1, C // 2):
                L = lab[x0:x0 + C, y0:y0 + C]
                ids = [i for i in np.unique(L) if i]
                pres = {i: (L == i).any(axis=2).mean() for i in ids}
                keep = [i for i in ids if pres[i] >= 0.98]
                if len(keep) < 2 or any(0.01 < pres[i] < 0.98 for i in ids): continue
                Ls.append(L); Vs.append(vol[x0:x0 + C, y0:y0 + C]); names.append(f"{os.path.basename(f)[:17]}_{x0}_{y0}")
    np.savez_compressed(CACHE, L=np.stack(Ls), V=np.stack(Vs), names=np.array(names))
    print("cache:", len(Ls), "morceaux")


def load():
    if not os.path.exists(CACHE): build_cache()
    d = np.load(CACHE)
    return d["L"], d["V"], list(d["names"])


def truth(L):
    ids = [i for i in np.unique(L) if i]
    pres = {i: (L == i).any(axis=2).mean() for i in ids}
    keep = [i for i in ids if pres[i] >= 0.98]
    zc = {i: np.mean(np.nonzero(L == i)[2]) for i in keep}
    keep.sort(key=lambda i: zc[i])
    valid = np.all([(L == i).any(axis=2) for i in keep], axis=0)
    Z = L.shape[2]; zz = np.arange(Z)
    contact = np.zeros(valid.shape, bool)
    for a, b in zip(keep, keep[1:]):
        contact |= (np.where(L == b, zz, Z).min(2) - np.where(L == a, zz, -1).max(2)) <= 1
    return keep, valid, contact


def score(S, L, keep, valid, contact):
    K_est, Z = S.shape[0], L.shape[2]
    rank = {i: k for k, i in enumerate(keep)}
    hits = np.full(S.shape, -1)
    for kk in range(K_est):
        for d in (0, -1, 1):
            lab = np.take_along_axis(L, np.clip(S[kk] + d, 0, Z - 1)[..., None], 2)[..., 0]
            r = np.vectorize(lambda v: rank.get(v, -1))(lab)
            hits[kk] = np.where(hits[kk] < 0, r, hits[kk])
    dom, pur, pur_c = [], [], []
    for kk in range(K_est):
        h = hits[kk][valid]; hc = hits[kk][valid & contact]
        hv = h[h >= 0]
        dk = int(np.bincount(hv).argmax()) if hv.size else -1
        dom.append(dk)
        pur.append(100 * (h == dk).mean())
        if hc.size: pur_c.append(100 * (hc == dk).mean())
    cov = 100 * len({d for d in dom if d >= 0}) / len(keep)
    dup = sum(1 for k in range(1, K_est) if dom[k] == dom[k - 1] and dom[k] >= 0)
    order = [dom[k + 1] > dom[k] for k in range(K_est - 1) if dom[k] >= 0 and dom[k + 1] >= 0]
    return {"purete": float(np.mean(pur)), "purete_contact": float(np.mean(pur_c)) if pur_c else None,
            "couverture": cov, "doublons": dup, "ordre": 100 * float(np.mean(order)) if order else None,
            "dK": K_est - len(keep)}


# ---------------- methodes ----------------
from msurf import solve_dp


def signal(V, sig_z=2, mode="ct"):
    P = ndi.gaussian_filter(V, (1, 1, sig_z))
    if mode == "ridge":  # crete le long de la colonne : -d2/dz2
        P = -ndi.gaussian_filter(V, (1, 1, sig_z), order=(0, 0, 2))
    return (P - P.mean()) / (P.std() + 1e-6)


def est_K(P, dmin, prom, how="median"):
    n = [len(find_peaks(P[x, y], distance=dmin, prominence=prom)[0]) for x in range(P.shape[0]) for y in range(P.shape[1])]
    return max(2, int({"median": np.median, "p25": lambda a: np.percentile(a, 25), "mode": lambda a: np.bincount(a).argmax()}[how](n)))


def smooth_dp(P, K, dmin, dmax, iters=0, lam=0.05, size=5):
    """DP par colonne, puis iterations : penalite quadratique vers la mediane des voisins (lissage approche)."""
    c = np.repeat(-P[None], K, axis=0)
    S = solve_dp(c, dmin=dmin, dmax=dmax)
    zz = np.arange(P.shape[2])[None, None, None, :]
    for _ in range(iters):
        ref = np.stack([ndi.median_filter(S[k].astype(float), size=size) for k in range(K)])
        S = solve_dp(c + lam * (zz - ref[..., None]) ** 2, dmin=dmin, dmax=dmax)
    return S


def run(name, cfg, oracle_K=False):
    L, V, names = load()
    t = time.time(); rows = []
    for i in range(len(L)):
        keep, valid, contact = truth(L[i])
        P = signal(V[i], cfg.get("sig_z", 2), cfg.get("mode", "ct"))
        K = len(keep) if oracle_K else est_K(P, cfg.get("dmin_k", cfg["dmin"]), cfg.get("prom", 0.3), cfg.get("how", "median"))
        S = smooth_dp(P, K, cfg["dmin"], cfg.get("dmax", 60), cfg.get("iters", 0), cfg.get("lam", 0.05), cfg.get("size", 5))
        rows.append(score(S, L[i], keep, valid, contact))
    agg = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 2) for k in rows[0]}
    agg["K_exact_%"] = round(100 * np.mean([r["dK"] == 0 for r in rows]), 1)
    agg["dK_abs"] = round(float(np.mean([abs(r["dK"]) for r in rows])), 2)
    out = {"exp": name, "oracle_K": oracle_K, "cfg": cfg, "n": len(rows), "sec": round(time.time() - t, 1), **agg}
    open(LOG, "a").write(json.dumps(out) + "\n")
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    base = {"dmin": 9}
    run("base_oracleK", base, oracle_K=True)
    run("base_Kest", base)
