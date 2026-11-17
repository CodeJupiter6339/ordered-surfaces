"""reeval.py, mais la DP + lissage tourne sur GPU (gpu_dp, resultats identiques au CPU).
Signal et K sur CPU (Pool), DP en lots par K sur GPU, score sur CPU. Usage : reeval_gpu.py [noms de configs...]"""
import os, sys, json, time, zlib, numpy as np, torch
from multiprocessing import Pool
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sweep5 import sig
from metric2 import score2, aggregate
from gpu_dp import smooth_dp_t
D = os.path.expanduser("~/vesuvius/spike/")
CACHE = D + "bench_reeval.npz"; LOG = D + "reeval_log.jsonl"
CFGS = {"chaine_publiee": ("Vn", {}), "chaine_scan_moyenne": ("Vm", {}), "chaine_dmin6": ("Vn", {"dmin": 6}),
        "chaine_dmin6_moyenne": ("Vm", {"dmin": 6}), "DP_sans_lissage": ("Vn", {"iters": 0})}


def prep(args):
    V, cfg = args
    P = sig(V, cfg.get("a", 1), cfg.get("s", 2)); dm = cfg.get("dmin", 9)
    npk = [len(find_peaks(P[x, y], distance=dm, prominence=1.0)[0]) for x in range(P.shape[0]) for y in range(P.shape[1])]
    return P, min(max(2, int(np.percentile(npk, cfg.get("pct", 90)))), (P.shape[2] - 1) // dm + 1)


def sc(args): return score2(*args)


def run_cfg(L, V, cfg, pool, bs=96):
    pk = pool.map(prep, [(V[j], cfg) for j in range(len(L))], chunksize=8)
    S = [None] * len(L); dm = cfg.get("dmin", 9)
    for K in sorted({k for _, k in pk}):
        ids = [j for j in range(len(L)) if pk[j][1] == K]
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs]
            P = torch.from_numpy(np.stack([pk[j][0] for j in b]).astype(np.float64)).cuda()
            out = smooth_dp_t(P, K, dm, 60, cfg.get("iters", 8), 0.02, 11).cpu().numpy()
            for j, s in zip(b, out): S[j] = s
    return pool.map(sc, [(S[j], L[j], cfg.get("tol", 1)) for j in range(len(L))], chunksize=8)


if __name__ == "__main__":
    d = np.load(CACHE); L, cube = d["L"], list(d["cube"])
    half = np.array([zlib.crc32(c.encode()) % 2 for c in cube])
    names = sys.argv[1:] or list(CFGS)
    with Pool(20) as pool:
        for nm in names:
            vk, cfg = CFGS[nm]; t = time.time()
            rows = run_cfg(L, d[vk], cfg, pool)
            out = {"exp": nm, "vol": vk, "cfg": cfg, "n": len(L), "sec": round(time.time() - t), "gpu": True}
            for k in ("suivi", "suivi_contact"):
                out[k] = aggregate(rows, cube, k)
                for h in (0, 1):
                    sel = [j for j in range(len(L)) if half[j] == h]
                    out[f"{k}_moitie{h}"] = (aggregate([rows[j] for j in sel], [cube[j] for j in sel], k) or {}).get("moy")
            open(LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out), flush=True)
