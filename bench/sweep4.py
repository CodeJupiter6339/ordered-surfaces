"""Test 4 : feuilles qui s'arretent dans la zone.
Idee : prendre K = toutes les feuilles presentes quelque part (pas seulement les completes) et plafonner le cout
(une surface sans feuille sous elle paie un cout fixe tau au lieu de tirer vers un bout de feuille voisin).
Score 'suivi' : pour chaque feuille complete, meilleure surface = % des colonnes ou elle est dans cette feuille.
(Ne penalise pas les surfaces en plus qui suivent des bouts de feuilles.)"""
import os, sys, json, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_bench as fb
from scipy.signal import find_peaks
d = np.load(os.path.expanduser("~/vesuvius/spike/bench_cache_dirty.npz"))
L, V = d["L"], d["V"]
idx = np.random.default_rng(0).choice(len(L), 400, replace=False)


def suivi(S, Li, keep, valid, contact):
    Z = Li.shape[2]; out, outc = [], []
    for i in keep:
        best = bestc = 0
        for k in range(S.shape[0]):
            ins = np.zeros(valid.shape, bool)
            for dz in (-1, 0, 1):
                ins |= np.take_along_axis(Li == i, np.clip(S[k] + dz, 0, Z - 1)[..., None], 2)[..., 0]
            best = max(best, ins[valid].mean())
            if (valid & contact).any(): bestc = max(bestc, ins[valid & contact].mean())
        out.append(best); outc.append(bestc)
    return 100 * np.mean(out), 100 * np.mean(outc)


def one(args):
    j, cfg = args
    Li, Vi = L[j], V[j]
    keep, valid, contact = fb.truth(Li)
    P = fb.signal(Vi, 2, "ct")
    npk = [len(find_peaks(P[x, y], distance=9, prominence=cfg["prom"])[0]) for x in range(32) for y in range(32)]
    if cfg["K"] == "oracle_full": K = len(keep)
    elif cfg["K"] == "oracle_all": K = len([i for i in np.unique(Li) if i and (Li == i).any(2).mean() > 0.01])
    else: K = max(2, int(np.percentile(npk, cfg["K"])))
    K = min(K, (Li.shape[2] - 1) // 9 + 1)
    Pc = P if cfg["tau"] is None else np.maximum(P, cfg["tau"])  # cout = -P plafonne a -tau
    S = fb.smooth_dp(Pc, K, 9, 60, 8, 0.02, 11)
    return suivi(S, Li, keep, valid, contact)


if __name__ == "__main__":
    cfgs = [{"K": "oracle_full", "tau": None, "prom": 1.0}, {"K": 50, "tau": None, "prom": 1.0}]
    cfgs += [{"K": k, "tau": t, "prom": 1.0} for k in ("oracle_all", 75, 90) for t in (None, -0.5, 0.0, 0.5)]
    done = {json.loads(l)["exp"] for l in open(fb.LOG) if '"I_K' in l}
    cfgs = [c for c in cfgs if f"I_K{c['K']}_tau{c['tau']}" not in done]
    with Pool(20) as pool:
        for cfg in cfgs:
            r = np.array(pool.map(one, [(j, cfg) for j in idx]))
            out = {"exp": f"I_K{cfg['K']}_tau{cfg['tau']}", "cfg": cfg, "n": len(idx),
                   "suivi": round(float(r[:, 0].mean()), 2), "suivi_contact": round(float(r[:, 1].mean()), 2)}
            open(fb.LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out), flush=True)
