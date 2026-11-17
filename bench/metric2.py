"""Mesure corrigee (suite a la revue) :
- contact PAR FEUILLE : colonnes ou CETTE feuille touche une autre etiquette (ecart <= 1 voxel en z)
- appariement surface <-> feuille UN A UN (hongrois), fixe sur les colonnes hors contact
- agregation par cube + intervalle de confiance bootstrap sur les cubes
Le suivi d'une feuille = % des colonnes valides ou sa surface attribuee est dans la feuille (tolerance +-tol)."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


def truth2(L, full=0.98):
    ids = [i for i in np.unique(L) if i]
    keep = [i for i in ids if (L == i).any(2).mean() >= full]
    zz = np.arange(L.shape[2])
    keep.sort(key=lambda i: np.mean(np.nonzero(L == i)[2]))
    valid = np.all([(L == i).any(2) for i in keep], axis=0) if keep else np.zeros(L.shape[:2], bool)
    contact = {}
    for i in keep:
        m = L == i
        near = ndi.binary_dilation(m, structure=np.ones((1, 1, 5), bool))  # +-2 voxels en z : ecart <= 1
        other = (L > 0) & ~m
        contact[i] = (near & other).any(2)
    return keep, valid, contact


def score2(S, L, tol=1):
    keep, valid, contact = truth2(L)
    if len(keep) < 1 or not valid.any(): return None
    Z = L.shape[2]; K = S.shape[0]
    inside = np.zeros((K, len(keep)) + L.shape[:2], bool)
    for k in range(K):
        for dz in range(-tol, tol + 1):
            lab = np.take_along_axis(L, np.clip(S[k] + dz, 0, Z - 1)[..., None], 2)[..., 0]
            for j, i in enumerate(keep): inside[k, j] |= lab == i
    W = np.zeros((K, len(keep)))
    for j, i in enumerate(keep):
        ref = valid & ~contact[i]
        if not ref.any(): ref = valid
        W[:, j] = inside[:, j][:, ref].mean(1)
    r, c = linear_sum_assignment(-W)
    tr, trc = np.zeros(len(keep)), []
    for k, j in zip(r, c):
        tr[j] = inside[k, j][valid].mean()
    for j, i in enumerate(keep):
        cc = valid & contact[i]
        if cc.sum() >= 5:
            k = r[list(c).index(j)] if j in c else None
            trc.append(inside[k, j][cc].mean() if k is not None else 0.0)
    return {"suivi": 100 * tr.mean(), "suivi_contact": 100 * np.mean(trc) if trc else np.nan,
            "n_feuilles": len(keep), "K": K}


def aggregate(rows, cubes, key, nboot=2000, seed=0):
    """Moyenne par cube puis moyenne des cubes ; IC 95 % par bootstrap sur les cubes."""
    per = {}
    for r, c in zip(rows, cubes):
        if r is None or np.isnan(r[key]): continue
        per.setdefault(c, []).append(r[key])
    m = np.array([np.mean(v) for v in per.values()])
    if not len(m): return None
    rng = np.random.default_rng(seed)
    bs = [rng.choice(m, len(m)).mean() for _ in range(nboot)]
    return {"moy": round(float(m.mean()), 2), "ic95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)], "cubes": len(m)}
