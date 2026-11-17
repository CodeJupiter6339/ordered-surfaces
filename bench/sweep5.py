"""Test 5 : contacts entre feuilles. Accentuer le creux entre deux feuilles collees.
P = lissage - a * derivee seconde en z (rehausse les cretes fines, creuse les jonctions). K = 90e centile, morceaux sales."""
import os, sys, json, numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_bench as fb
import sweep4 as s4


def sig(V, a, s):
    g = ndi.gaussian_filter(V, (1, 1, s)).astype(np.float32)
    d2 = ndi.gaussian_filter(V, (1, 1, s), order=(0, 0, 2)).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6); d2 = (d2 - d2.mean()) / (d2.std() + 1e-6)
    P = g - a * d2
    return (P - P.mean()) / (P.std() + 1e-6)


def one(args):
    j, a, s = args
    Li, Vi = s4.L[j], s4.V[j]
    keep, valid, contact = fb.truth(Li)
    P = sig(Vi, a, s)
    npk = [len(find_peaks(P[x, y], distance=9, prominence=1.0)[0]) for x in range(32) for y in range(32)]
    K = min(max(2, int(np.percentile(npk, 90))), (Li.shape[2] - 1) // 9 + 1)
    S = fb.smooth_dp(P, K, 9, 60, 8, 0.02, 11)
    return s4.suivi(S, Li, keep, valid, contact)


if __name__ == "__main__":
    with Pool(20) as pool:
        for a, s in ((0, 2), (0.5, 2), (1, 2), (2, 2), (1, 1), (1, 1.5)):
            r = np.array(pool.map(one, [(j, a, s) for j in s4.idx]))
            out = {"exp": f"J_a{a}_s{s}", "n": len(s4.idx), "suivi": round(float(r[:, 0].mean()), 2), "suivi_contact": round(float(r[:, 1].mean()), 2)}
            open(fb.LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out), flush=True)
