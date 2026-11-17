"""Multi-surface optimal graph search (Li, Wu, Chen, Sonka, TPAMI 2006) for papyrus sheets.

K ordered surfaces z = S_k(x, y) along columns, found jointly by one min-cut with:
  smoothness   |S_k(p) - S_k(q)| <= dz       (4-neighbour columns)
  ordering     dmin <= S_{k+1} - S_k <= dmax (sheets may touch, never swap or cross)
Cost of surface k at z: c[k, x, y, z]. Graph = minimum-weight closure, w(z) = c(z) - c(z-1).
"""
import numpy as np
import maxflow

INF = 1e9


def solve_dp(c, dmin=3, dmax=40):
    """Ordered surfaces per column, exact DP, no inter-column smoothness.
    c: (K, X, Y, Z). S_{k+1} - S_k in [dmin, dmax]. Vectorised over columns."""
    K, X, Y, Z = c.shape
    D = c[0].astype(np.float64).copy()
    back = np.zeros((K, X, Y, Z), np.int32)
    for k in range(1, K):
        best = np.full((X, Y, Z), np.inf); arg = np.zeros((X, Y, Z), np.int32)
        for d in range(dmin, dmax + 1):
            cand = np.full((X, Y, Z), np.inf); cand[..., d:] = D[..., :-d]
            better = cand < best
            best = np.where(better, cand, best); arg = np.where(better, d, arg)
        D = c[k] + best; back[k] = arg
    S = np.zeros((K, X, Y), np.int64)
    S[-1] = D.argmin(-1)
    for k in range(K - 1, 0, -1):
        S[k - 1] = S[k] - np.take_along_axis(back[k], S[k][..., None], -1)[..., 0]
    return S


def solve(c, dz=2, dmin=3, dmax=40):
    K, X, Y, Z = c.shape
    g = maxflow.Graph[float]()
    ids = g.add_grid_nodes((K, X, Y, Z))

    def arcs(off):
        """INF arc from every node v to v+off (4D offset), skipping out-of-range."""
        src = [slice(max(0, -o), n - max(0, o)) for o, n in zip(off, (K, X, Y, Z))]
        dst = [slice(max(0, o), n - max(0, -o)) for o, n in zip(off, (K, X, Y, Z))]
        a, b = ids[tuple(src)].ravel(), ids[tuple(dst)].ravel()
        if a.size:
            g.add_edges(a, b, np.full(a.size, INF), np.zeros(a.size))

    arcs((0, 0, 0, -1))                       # closure along the column
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        arcs((0, dx, dy, -dz))                # S(q) >= S(p) - dz
    if K > 1:
        # S_{k+1} >= S_k + dmin : V_k(z) -> V_{k+1}(z + dmin)
        sub_a = ids[:-1, :, :, : Z - dmin].ravel(); sub_b = ids[1:, :, :, dmin:].ravel()
        g.add_edges(sub_a, sub_b, np.full(sub_a.size, INF), np.zeros(sub_a.size))
        # V_k(z) with z + dmin > Z-1 can never be in the closure
        bad = ids[:-1, :, :, Z - dmin:].ravel()
        g.add_grid_tedges(bad, 0, INF)
        # S_{k+1} <= S_k + dmax : V_{k+1}(z) -> V_k(z - dmax)
        if dmax < Z:
            sub_a = ids[1:, :, :, dmax:].ravel(); sub_b = ids[:-1, :, :, : Z - dmax].ravel()
            g.add_edges(sub_a, sub_b, np.full(sub_a.size, INF), np.zeros(sub_a.size))
    w = np.diff(c, axis=3, prepend=0.0)
    w[..., 0] = -INF / 10                     # bottom node always in the closure
    g.add_grid_tedges(ids, np.maximum(-w, 0), np.maximum(w, 0))
    g.maxflow()
    inside = ~g.get_grid_segments(ids)        # True = source side = closure
    return inside.sum(axis=3) - 1             # S_k(x, y) = top z in closure


if __name__ == "__main__":
    # Synthetic check: 3 wavy sheets, sheets 1 and 2 touch (merged blob) in the middle.
    X = Y = 24; Z = 60
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    centers = [15 + 3 * np.sin(xx / 5), 30 + 2 * np.cos(yy / 6), 45 + 0 * xx]
    blend = np.exp(-((xx - 12) / 4.0) ** 2)                      # smooth approach, touch at x~12
    centers[2] = blend * (centers[1] + 5) + (1 - blend) * 45     # contact zone: sheets touch, no gap
    zz = np.arange(Z)[None, None, :]
    mask = np.zeros((X, Y, Z), bool)
    for cz in centers:
        mask |= np.abs(zz - cz[..., None]) <= 2.5          # thickness ~6
    from scipy import ndimage as ndi
    # sheet-agnostic signal: distance to background along the column (merged blob = one peak)
    P = ndi.distance_transform_edt(mask)
    c = np.repeat(-P[None], 3, axis=0)
    S = solve(c, dz=2, dmin=4, dmax=30)
    for k, cz in enumerate(centers):
        err = np.abs(S[k] - cz)
        print(f"surface {k}: erreur moyenne {err.mean():.2f} vox, max {err.max():.1f}, "
              f"zone de contact {err[(xx > 8) & (xx < 16)].mean():.2f}, "
              f"% colonnes dans la bonne feuille {100 * (err <= 2.5).mean():.1f}")
    # naive per-column counting of runs (local baseline)
    runs_ok = 0
    for x in range(X):
        for y in range(Y):
            lab, n = ndi.label(mask[x, y]); runs_ok += (n == 3)
    print(f"baseline comptage local: {100 * runs_ok / (X * Y):.0f}% des colonnes voient bien 3 feuilles")
