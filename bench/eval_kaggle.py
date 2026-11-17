"""Banc 'autres rouleaux' : cubes Kaggle surface-detection (320^3, labels recto : 0 fond, 1 surface, 2 non annote).
Une feuille = une composante connexe de surface annotee. Morceaux 32x32 sans zone non annotee, >= 2 surfaces completes.
Meme chaine que le Scroll 1 (axe calcule, K au 90e centile, DP + lissage). Usage : eval_kaggle.py [build|run] [pred_dir]"""
import os, sys, json, glob, numpy as np, tifffile
from multiprocessing import Pool
from scipy import ndimage as ndi
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_bench as fb
from sweep5 import sig
from fast_ordered_notrich import axis_from_ct
D = os.path.expanduser("~/vesuvius/data/kaggle_surf/")
CACHE = os.path.expanduser("~/vesuvius/spike/bench_kaggle.npz")
NPROC = max(1, (os.cpu_count() or 8) - 4)


def prep(f):
    v = tifffile.imread(f.replace("labels", "images"))[::2, ::2, ::2].astype(np.float32)
    l = tifffile.imread(f)[::2, ::2, ::2]
    a = axis_from_ct(v)
    v, l = np.moveaxis(v, a, 2), np.moveaxis(l, a, 2)
    out = []
    for x0 in range(0, v.shape[0] - 32 + 1, 16):
        for y0 in range(0, v.shape[1] - 32 + 1, 16):
            lc = l[x0:x0 + 32, y0:y0 + 32]
            if (lc == 2).mean() > 0.6: continue  # trop peu annote
            L, n = ndi.label(lc == 1)
            full = [i for i in range(1, n + 1) if (L == i).any(2).mean() >= 0.98]
            if len(full) >= 2:
                out.append((L.astype(np.int16), v[x0:x0 + 32, y0:y0 + 32], f"{os.path.basename(f)[:12]}_{x0}_{y0}"))
    return out


def one(args):
    L, V, w, C, cfg = args
    keep, valid, contact = fb.truth(L)
    P = sig(V, cfg.get("a", 1), cfg.get("s", 2))
    if w and C is not None:
        z = lambda a: (a - a.mean()) / (a.std() + 1e-6)
        P = z((1 - w) * P + w * z(C))
    dm = cfg.get("dmin", 9)
    npk = [len(find_peaks(P[x, y], distance=dm, prominence=1.0)[0]) for x in range(32) for y in range(32)]
    K = min(max(2, int(np.percentile(npk, cfg.get("pct", 90)))), (L.shape[2] - 1) // dm + 1)
    S = fb.smooth_dp(P, K, dm, 80, cfg.get("iters", 8), 0.02, 11)
    # tolerance +-3 : la surface annotee est le recto, notre surface vise le centre de la feuille
    Z = L.shape[2]; res = []
    for sub in (valid, valid & contact):
        if not sub.any(): res.append(np.nan); continue
        sc = []
        for i in keep:
            best = 0
            for k in range(S.shape[0]):
                ins = np.zeros(valid.shape, bool)
                for dz in range(-3, 4):
                    ins |= np.take_along_axis(L == i, np.clip(S[k] + dz, 0, Z - 1)[..., None], 2)[..., 0]
                best = max(best, ins[sub].mean())
            sc.append(best)
        res.append(100 * np.mean(sc))
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "build" or not os.path.exists(CACHE):
        with Pool(14) as pool:
            r = [c for x in pool.map(prep, sorted(glob.glob(D + "labels/*.tif"))) for c in x]
        np.savez_compressed(CACHE, L=np.stack([c[0] for c in r]), V=np.stack([c[1] for c in r]), names=np.array([c[2] for c in r]))
        print("morceaux:", len(r), "| cubes:", len({c[2][:12] for c in r}), flush=True)
    d = np.load(CACHE); L, V, names = d["L"], d["V"], list(d["names"])
    idx = np.random.default_rng(0).choice(len(L), min(400, len(L)), replace=False)
    cfgs = [("ref_sans_lissage", {"iters": 0}), ("chaine", {})] + [(f"dmin{d}_s{s}", {"dmin": d, "s": s}) for d in (3, 4, 5, 6, 12, 15, 20) for s in (2, 3)]
    done = {json.loads(l)["exp"] for l in open(fb.LOG) if "M_kaggle_" in l} if os.path.exists(fb.LOG) else set()
    cfgs = [c for c in cfgs if f"M_kaggle_{c[0]}" not in done]
    for nm, cfg in cfgs:
        with Pool(NPROC) as pool:
            r = np.array(pool.map(one, [(L[j], V[j], 0, None, cfg) for j in idx], chunksize=2))
        out = {"exp": f"M_kaggle_{nm}", "cfg": cfg, "n": len(idx), "cubes": len({names[j][:12] for j in idx}),
               "suivi": round(float(np.nanmean(r[:, 0])), 2)}
        open(fb.LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out), flush=True)
