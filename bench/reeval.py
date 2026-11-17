"""Reevaluation complete du Scroll 1 avec la mesure corrigee (metric2), sur TOUS les morceaux ayant >= 2 feuilles completes.
Deux reductions de resolution : 'nearest' ([::2], comme avant) et 'mean' (moyenne 2x2x2 du scan, labels au plus proche).
Usage : reeval.py [build]"""
import os, sys, json, glob, zlib, time, numpy as np, nrrd
from multiprocessing import Pool
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_bench as fb
from sweep5 import sig
from fast_ordered_notrich import axis_from_ct, order_sheets
from metric2 import score2, aggregate
D = os.path.expanduser("~/vesuvius/spike/")
CACHE = D + "bench_reeval.npz"; LOG = D + "reeval_log.jsonl"
NPROC = max(1, (os.cpu_count() or 8) - 4)


def prep(f):
    lab, _ = nrrd.read(f); vol, _ = nrrd.read(f.replace("_mask", "_volume"))
    vol = vol.astype(np.float32)
    ln = lab[::2, ::2, ::2].astype(np.int32)
    vn = vol[::2, ::2, ::2]
    vm = vol.reshape(128, 2, 128, 2, 128, 2).mean((1, 3, 5))
    if order_sheets(ln) is None: return []
    a = axis_from_ct(vn)
    ln, vn, vm = (np.moveaxis(x, a, 2) for x in (ln, vn, vm))
    out = []
    for x0 in range(0, 97, 16):
        for y0 in range(0, 97, 16):
            L = ln[x0:x0 + 32, y0:y0 + 32]
            ids = [i for i in np.unique(L) if i]
            if sum((L == i).any(2).mean() >= 0.98 for i in ids) >= 2:
                out.append((L.astype(np.int16), vn[x0:x0 + 32, y0:y0 + 32], vm[x0:x0 + 32, y0:y0 + 32], os.path.basename(f)[:17]))
    return out


def peaks_baseline(P, K, dmin):
    S = np.zeros((K,) + P.shape[:2], int)
    for x in range(P.shape[0]):
        for y in range(P.shape[1]):
            pk = find_peaks(P[x, y], distance=dmin, prominence=1.0)[0][:K]
            S[:len(pk), x, y] = pk
            if 0 < len(pk) < K: S[len(pk):, x, y] = pk[-1]
    return S


def one(args):
    L, V, cfg = args
    P = sig(V, cfg.get("a", 1), cfg.get("s", 2))
    dm = cfg.get("dmin", 9)
    npk = [len(find_peaks(P[x, y], distance=dm, prominence=1.0)[0]) for x in range(32) for y in range(32)]
    K = min(max(2, int(np.percentile(npk, cfg.get("pct", 90)))), (L.shape[2] - 1) // dm + 1)
    if cfg.get("method") == "peaks":
        S = peaks_baseline(P, K, dm)
    else:
        S = fb.smooth_dp(P, K, dm, 60, cfg.get("iters", 8), 0.02, 11)
        if cfg.get("method") == "anchor":
            from anchor import anchor_bridge
            S, _ = anchor_bridge(P, S, dm, 60, cfg.get("thr", 0.5), cfg.get("lam", 0.5), cfg.get("rounds", 1))
    return score2(S, L, cfg.get("tol", 1))


if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1] == "build") or not os.path.exists(CACHE):
        with Pool(NPROC) as pool:
            r = [c for x in pool.map(prep, sorted(glob.glob(D + "cubes/*_mask.nrrd"))) for c in x]
        np.savez_compressed(CACHE, L=np.stack([c[0] for c in r]), Vn=np.stack([c[1] for c in r]), Vm=np.stack([c[2] for c in r]), cube=np.array([c[3] for c in r]))
        print("morceaux:", len(r), "cubes:", len({c[3] for c in r}), flush=True)
    d = np.load(CACHE); L, cube = d["L"], list(d["cube"])
    half = np.array([zlib.crc32(c.encode()) % 2 for c in cube])
    cfgs = [("pics_ordre_sans_DP", "Vn", {"method": "peaks"}),
            ("DP_sans_lissage", "Vn", {"iters": 0}),
            ("chaine_publiee", "Vn", {}),
            ("chaine_scan_moyenne", "Vm", {}),
            ("chaine_dmin6", "Vn", {"dmin": 6}),
            ("chaine_dmin6_moyenne", "Vm", {"dmin": 6})]
    done = {json.loads(l)["exp"] for l in open(LOG)} if os.path.exists(LOG) else set()
    for nm, vk, cfg in cfgs:
        if nm in done: continue
        t = time.time(); V = d[vk]
        with Pool(NPROC) as pool:
            rows = pool.map(one, [(L[j], V[j], cfg) for j in range(len(L))], chunksize=4)
        out = {"exp": nm, "vol": vk, "cfg": cfg, "n": len(L), "sec": round(time.time() - t)}
        for k in ("suivi", "suivi_contact"):
            out[k] = aggregate(rows, cube, k)
            for h in (0, 1):
                sel = [j for j in range(len(L)) if half[j] == h]
                out[f"{k}_moitie{h}"] = (aggregate([rows[j] for j in sel], [cube[j] for j in sel], k) or {}).get("moy")
        open(LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out), flush=True)
