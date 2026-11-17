"""Balayage rapide de variantes sur le banc fast_bench (une variante = quelques secondes)."""
import os, sys, json, itertools, numpy as np
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_bench as fb

# reference locale : k-ieme bande claire de la colonne (petits vides fermes)
def run_count(gap=4):
    L, V, names = fb.load(); rows = []
    for i in range(len(L)):
        keep, valid, contact = fb.truth(L[i])
        K = len(keep); m = L[i] > 0
        mc = ndi.binary_closing(m, structure=np.ones((1, 1, gap + 1), bool)) | m
        S = np.zeros((K,) + m.shape[:2], int)
        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                r, n = ndi.label(mc[x, y])
                cs = [int(np.mean(np.nonzero(r == j + 1)[0])) for j in range(n)]
                for k in range(K): S[k, x, y] = cs[k] if k < n else 0
        rows.append(fb.score(S, L[i], keep, valid, contact))
    agg = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 2) for k in rows[0]}
    out = {"exp": f"reference_comptage_local_gap{gap}", "oracle_K": True, "n": len(rows), **agg}
    open(fb.LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out))

run_count(4)
# 1) parametres de base, K connu (isole l'effet de la methode)
for dmin, sig, mode in itertools.product((6, 9, 12, 15), (1, 2, 3), ("ct", "ridge")):
    fb.run(f"A_dmin{dmin}_sig{sig}_{mode}", {"dmin": dmin, "sig_z": sig, "mode": mode}, oracle_K=True)
# 2) lissage entre colonnes, K connu
for it, lam, size in itertools.product((2, 4), (0.02, 0.05, 0.2), (3, 7)):
    fb.run(f"B_it{it}_lam{lam}_s{size}", {"dmin": 9, "iters": it, "lam": lam, "size": size}, oracle_K=True)
# 3) estimation de K
for prom, how, dk in itertools.product((0.3, 0.6, 1.0), ("median", "p25", "mode"), (9, 12, 15)):
    fb.run(f"C_prom{prom}_{how}_dk{dk}", {"dmin": 9, "prom": prom, "how": how, "dmin_k": dk})
