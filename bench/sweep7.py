"""Test 7 (evaluation) : le 'centre de feuille' predit remplace / complete l'intensite, sur les cubes NON vus.
Usage : sweep7.py TRAIN  (lit center_pred_train{TRAIN}, evalue les morceaux sales des cubes de l'autre moitie)."""
import os, sys, json, zlib, numpy as np
from functools import lru_cache
from multiprocessing import Pool
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_bench as fb
import sweep4 as s4
from sweep5 import sig
from sweep6 import names
import nrrd, glob
from fast_ordered_notrich import axis_from_ct
TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PRED = sys.argv[2] if len(sys.argv) > 2 else f"center_pred_train{TRAIN}"
D = os.path.expanduser("~/vesuvius/spike/")
AXES = json.load(open(D + "cube_axes.json")) if os.path.exists(D + "cube_axes.json") else {}


@lru_cache(maxsize=8)
def pred_cube(cube):
    if cube in AXES: a = AXES[cube]
    else:
        v, _ = nrrd.read(glob.glob(D + f"cubes/{cube}*_volume.nrrd")[0])
        a = axis_from_ct(v[::2, ::2, ::2].astype(np.float32))
    return np.moveaxis(np.load(D + f"{PRED}/{cube}.npy").astype(np.float32), a, 2)


def z(a): return (a - a.mean()) / (a.std() + 1e-6)


def one(args):
    j, w = args
    cube, x0, y0 = names[j][:17], *map(int, names[j].split("_")[-2:])
    Li, Vi = s4.L[j], s4.V[j]
    keep, valid, contact = fb.truth(Li)
    P = sig(Vi, 1, 2)
    if w:
        C = pred_cube(cube)[x0:x0 + 32, y0:y0 + 32]
        P = z((1 - w) * P + w * z(C))
    npk = [len(find_peaks(P[x, y], distance=9, prominence=1.0)[0]) for x in range(32) for y in range(32)]
    K = min(max(2, int(np.percentile(npk, 90))), (Li.shape[2] - 1) // 9 + 1)
    S = fb.smooth_dp(P, K, 9, 60, 8, 0.02, 11)
    return s4.suivi(S, Li, keep, valid, contact)


if __name__ == "__main__":
    test = [j for j in range(len(names)) if zlib.crc32(names[j][:17].encode()) % 2 != TRAIN]
    idx = sorted(np.random.default_rng(1).choice(test, min(400, len(test)), replace=False), key=lambda j: names[j])
    with Pool(20) as pool:
        for w in ([float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else (0, 0.5, 0.75, 1.0)):
            r = np.array(pool.map(one, [(j, w) for j in idx], chunksize=2))
            out = {"exp": f"L_{PRED}_w{w}", "n": len(idx), "suivi": round(float(r[:, 0].mean()), 2),
                   "suivi_contact": round(float(r[:, 1].mean()), 2)}
            open(fb.LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out), flush=True)
