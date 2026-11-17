"""Reglage honnete de l'ancrage : grille evaluee sur la moitie A des cubes, meilleur reglage (suivi aux contacts)
mesure sur la moitie B jamais vue, a cote de la chaine publiee sur la meme moitie B."""
import os, sys, json, zlib, itertools, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from reeval import one, CACHE, NPROC
from metric2 import aggregate
LOG = os.path.expanduser("~/vesuvius/spike/anchor_log.jsonl")

d = np.load(CACHE); L, V, cube = d["L"], d["Vn"], list(d["cube"])
half = np.array([zlib.crc32(c.encode()) % 2 for c in cube])
A = [j for j in range(len(L)) if half[j] == 0]; B = [j for j in range(len(L)) if half[j] == 1]


def ev(idx, cfg):
    with Pool(NPROC) as pool:
        rows = pool.map(one, [(L[j], V[j], cfg) for j in idx], chunksize=4)
    cs = [cube[j] for j in idx]
    return aggregate(rows, cs, "suivi"), aggregate(rows, cs, "suivi_contact")


def log(o): open(LOG, "a").write(json.dumps(o) + "\n"); print(json.dumps(o), flush=True)


base_A = ev(A, {})
log({"exp": "A_chaine", "suivi": base_A[0], "contact": base_A[1]})
best = None
for thr, lam in itertools.product((0.25, 0.5, 1.0), (0.002, 0.01, 0.05)):
    cfg = {"method": "anchor", "thr": thr, "lam": lam}
    s, c = ev(A, cfg)
    log({"exp": f"A_anchor_thr{thr}_lam{lam}", "cfg": cfg, "suivi": s, "contact": c})
    if best is None or c["moy"] + s["moy"] > best[0]: best = (c["moy"] + s["moy"], cfg)
for nm, cfg in (("B_chaine", {}), ("B_anchor_meilleur", best[1])):
    s, c = ev(B, cfg)
    log({"exp": nm, "cfg": cfg, "suivi": s, "contact": c})
