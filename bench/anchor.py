"""Ancrer puis relier (suite a la revue Codex).
1. Chaine habituelle -> surfaces S (K, X, Y).
2. Colonnes AMBIGUES pour la surface k : pas de vrai creux d'intensite entre k et ses voisines
   (creux = P(surface) - min(P entre les deux surfaces) < thr), ou ecart a la voisine <= dmin + 1.
3. Sur les colonnes ambigues, on remplace la surface par une interpolation harmonique depuis les colonnes sures
   (on interpole la position ET on respecte l'ordre), puis DP final avec forte attraction vers cette reference
   sur les colonnes ambigues seulement."""
import numpy as np
from msurf import solve_dp


def ambiguous(P, S, dmin, thr):
    K, X, Y = S.shape; Z = P.shape[2]
    amb = np.zeros(S.shape, bool)
    for k in range(K):
        v = np.take_along_axis(P, S[k][..., None], 2)[..., 0]
        for nb in (k - 1, k + 1):
            if nb < 0 or nb >= K: continue
            lo, hi = np.minimum(S[k], S[nb]), np.maximum(S[k], S[nb])
            zz = np.arange(Z)[None, None, :]
            between = (zz > lo[..., None]) & (zz < hi[..., None])
            mn = np.where(between, P, np.inf).min(2)
            mn = np.where(np.isfinite(mn), mn, v)
            amb[k] |= (v - mn < thr) | (hi - lo <= dmin + 1)
    return amb


def harmonic_fill(S, amb, iters=300):
    X = S.astype(np.float64).copy()
    if amb.all(): return X
    for _ in range(iters):
        p = np.pad(X, 1, mode="edge")
        avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
        X = np.where(amb, avg, X)
    return X


def anchor_bridge(P, S, dmin, dmax=60, thr=0.5, lam=0.5, rounds=1, lam_smooth=0.02, size=11):
    """Garde le lissage habituel (attraction lam_smooth vers la mediane des voisines) PARTOUT,
    et ajoute une attraction lam vers la reference interpolee depuis les colonnes sures, sur les colonnes ambigues."""
    from scipy import ndimage as ndi
    K = S.shape[0]; zz = np.arange(P.shape[2])[None, None, None, :]
    for _ in range(rounds):
        amb = ambiguous(P, S, dmin, thr)
        ref = np.stack([harmonic_fill(S[k], amb[k]) for k in range(K)])
        for k in range(1, K): ref[k] = np.maximum(ref[k], ref[k - 1] + dmin)
        med = np.stack([ndi.median_filter(S[k].astype(float), size=size) for k in range(K)])
        w = np.where(amb, lam, 0.0)[..., None]
        c = np.repeat(-P[None], K, axis=0) + lam_smooth * (zz - med[..., None]) ** 2 + w * (zz - ref[..., None]) ** 2
        S = solve_dp(c, dmin=dmin, dmax=dmax)
    return S, float(amb.mean())
