"""Autres rouleaux (Kaggle) avec la mesure corrigee (metric2, appariement un a un, agregation par cube, IC bootstrap).
Jeu 'reglage' = cubes 1-30 (bench_kaggle.npz) ; jeu 'validation' = cubes 31-90 jamais utilises (bench_kaggle_val.npz).
Une feuille = composante connexe du recto annote ; tolerance +-3 (notre surface vise le centre, pas le recto)."""
import os, sys, json, glob, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eval_kaggle as ek
from reeval import one as one_s1
from metric2 import aggregate
D = os.path.expanduser("~/vesuvius/spike/")
LOG = D + "reeval_kaggle_log.jsonl"
NPROC = max(1, (os.cpu_count() or 8) - 4)


def build_val():
    p = D + "bench_kaggle_val.npz"
    if os.path.exists(p): return p
    with Pool(NPROC) as pool:
        r = [c for x in pool.map(ek.prep, sorted(glob.glob(os.path.expanduser("~/vesuvius/data/kaggle_val/labels/*.tif")))) for c in x]
    np.savez_compressed(p, L=np.stack([c[0] for c in r]), V=np.stack([c[1] for c in r]), names=np.array([c[2] for c in r]))
    print("validation :", len(r), "morceaux,", len({c[2][:12] for c in r}), "cubes", flush=True)
    return p


if __name__ == "__main__":
    sets = {"reglage": D + "bench_kaggle.npz", "validation": build_val()}
    cfgs = [("pics_ordre_sans_DP", {"method": "peaks", "dmin": 6, "tol": 3}),
            ("DP_sans_lissage_dmin6", {"iters": 0, "dmin": 6, "tol": 3}),
            ("chaine_dmin9", {"tol": 3}),
            ("chaine_dmin6", {"dmin": 6, "tol": 3}),
            ("chaine_dmin5", {"dmin": 5, "tol": 3})]
    done = {json.loads(l)["exp"] for l in open(LOG)} if os.path.exists(LOG) else set()
    for sname, path in sets.items():
        d = np.load(path); L, V, names = d["L"], d["V"], list(d["names"])
        idx = np.random.default_rng(0).choice(len(L), min(600, len(L)), replace=False)
        cubes = [names[j][:12] for j in idx]
        for nm, cfg in cfgs:
            exp = f"{sname}_{nm}"
            if exp in done: continue
            with Pool(NPROC) as pool:
                rows = pool.map(one_s1, [(L[j].astype(np.int32), V[j], cfg) for j in idx], chunksize=2)
            out = {"exp": exp, "cfg": cfg, "n": len(idx), "suivi": aggregate(rows, cubes, "suivi")}
            open(LOG, "a").write(json.dumps(out) + "\n"); print(json.dumps(out), flush=True)
