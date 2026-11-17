"""Spike: does global phase unwrapping recover winding numbers better than local propagation?

Ground truth: Scroll 1 cubes with one instance ID per papyrus sheet (volumetric-instance-labels).
1. Order the sheets of a cube (adjacency graph of their Voronoi regions must be a simple path).
2. Build a continuous winding coordinate W (sheet rank + fraction toward the neighbouring sheet).
3. Keep only the wrapped phase psi = 2*pi*W mod 2*pi, as a fractional-winding predictor would give,
   then corrupt it: gaussian noise + "fog" blobs where the phase is pure noise (merged/compressed regions).
4. Recover W with: naive path integration (local propagation), quality-guided unwrapping
   (skimage, Herraez 3D), unweighted least squares (Ghiglia-Romero DCT).
5. Score: % of papyrus voxels OUTSIDE the fog blobs whose winding number is wrong.
Throwaway code: results only say whether the idea deserves a real test on real model outputs.
"""
import glob, os, sys, json, time
import numpy as np, nrrd
from scipy import ndimage as ndi
from scipy.fft import dctn, idctn
from skimage.restoration import unwrap_phase

RNG = np.random.default_rng(0)
CUBES = sorted(glob.glob(os.path.expanduser("~/vesuvius/spike/cubes/*_mask.nrrd")))


def order_sheets(lab):
    ids = [i for i in np.unique(lab) if i]
    _, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
    vor = lab[tuple(idx)]  # nearest-sheet label everywhere
    pairs = {}
    for ax in range(3):
        a = np.moveaxis(vor, ax, 0)
        x, y = a[:-1].ravel(), a[1:].ravel()
        m = x != y
        u, c = np.unique(np.sort(np.stack([x[m], y[m]]), 0), axis=1, return_counts=True)
        for (i, j), n in zip(u.T, c):
            pairs[(i, j)] = pairs.get((i, j), 0) + n
    big = max(pairs.values())
    adj = {i: set() for i in ids}
    for (i, j), n in pairs.items():
        if n > 0.05 * big:
            adj[i].add(j); adj[j].add(i)
    ends = [i for i in ids if len(adj[i]) == 1]
    if any(len(adj[i]) > 2 for i in ids) or len(ends) != 2:
        return None
    order, prev, cur = [ends[0]], None, ends[0]
    while True:
        nxt = [k for k in adj[cur] if k != prev]
        if not nxt: break
        prev, cur = cur, nxt[0]; order.append(cur)
    return order if len(order) == len(ids) else None


def winding_field(lab, order):
    D = np.stack([ndi.distance_transform_edt(lab != i) for i in order])  # (K, ...)
    r1 = D.argmin(0)
    K = len(order)
    lo = np.take_along_axis(D, np.clip(r1 - 1, 0, K - 1)[None], 0)[0]
    hi = np.take_along_axis(D, np.clip(r1 + 1, 0, K - 1)[None], 0)[0]
    d1 = np.take_along_axis(D, r1[None], 0)[0]
    use_hi = (hi <= lo) & (r1 < K - 1) | (r1 == 0)
    r2 = np.where(use_hi, r1 + 1, r1 - 1)
    d2 = np.where(use_hi, hi, lo)
    W = r1 + (r2 - r1) * d1 / (d1 + d2 + 1e-9)
    # Relaxation de Laplace (feuilles fixees a leur rang) : enleve les sauts d'un tour
    # crees la ou une feuille est trouee ou s'arrete dans le cube.
    W = W.astype(np.float64)
    fixed = lab > 0
    rank = np.zeros(lab.max() + 1); rank[order] = np.arange(K)
    W[fixed] = rank[lab[fixed]]
    for _ in range(400):
        P = np.pad(W, 1, mode="edge")
        W_new = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] + P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1]
                 + P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2]) / 6
        W = np.where(fixed, W, W_new)
    return W


def corrupt(psi, lab, sigma, n_blobs, radius):
    out = psi + RNG.normal(0, sigma, psi.shape)
    fog = np.zeros(psi.shape, bool)
    pts = np.argwhere(lab > 0)
    zz, yy, xx = np.ogrid[: psi.shape[0], : psi.shape[1], : psi.shape[2]]
    for c in pts[RNG.choice(len(pts), n_blobs, replace=False)]:
        fog |= (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= radius ** 2
    out[fog] = RNG.uniform(-np.pi, np.pi, fog.sum())
    return np.angle(np.exp(1j * out)), fog


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def naive_path(psi):
    u = np.unwrap(psi, axis=2)                       # along each row
    col = np.unwrap(u[:, :, 0], axis=1)              # fix rows using first column
    u += (col - u[:, :, 0])[:, :, None]
    pil = np.unwrap(u[:, 0, 0])                      # fix slices using first pillar
    u += (pil - u[:, 0, 0])[:, None, None]
    return u


def least_squares(psi):
    g = [wrap(np.diff(psi, axis=a)) for a in range(3)]
    rho = np.zeros_like(psi)
    for a in range(3):
        pad = [(0, 0)] * 3; pad[a] = (1, 1)
        gp = np.pad(g[a], pad)
        rho += np.diff(gp, axis=a)
    R = dctn(rho, norm="ortho")
    k = [np.cos(np.pi * np.arange(n) / n) for n in psi.shape]
    den = 2 * (k[0][:, None, None] + k[1][None, :, None] + k[2][None, None, :] - 3)
    den[0, 0, 0] = 1
    R /= den; R[0, 0, 0] = 0
    u = idctn(R, norm="ortho")
    return psi + 2 * np.pi * np.round((u - psi) / (2 * np.pi))  # congruent with data


def score(U, W, lab, fog):
    """% de voxels de chaque feuille (hors brouillard) dont le numero de tour differe
    du numero majoritaire de cette feuille = proxy des sauts de feuille. Plus
    % de paires de feuilles voisines mal ordonnees (ecart != 1)."""
    est = np.round(U / (2 * np.pi) - (W - np.round(W)))  # numero entier estime
    ok = (lab > 0) & ~fog
    ids = np.unique(lab[ok])
    wrong, total, modes = 0, 0, {}
    for i in ids:
        v = est[(lab == i) & ok].astype(np.int64)
        vals, c = np.unique(v, return_counts=True)
        modes[i] = (vals[c.argmax()], np.round(np.median(W[lab == i])))
        wrong += v.size - c.max(); total += v.size
    ranks = sorted(modes.values(), key=lambda t: t[1])
    bad = sum(1 for a, b in zip(ranks, ranks[1:]) if b[0] - a[0] != b[1] - a[1])
    return round(100 * wrong / total, 2), bad


METHODS = {"local (path)": naive_path, "quality-guided": lambda p: unwrap_phase(p), "least-squares": least_squares}
SETTINGS = [(0.0, 0, 0), (0.3, 0, 0), (0.6, 0, 0), (0.6, 4, 6), (0.6, 10, 6), (0.9, 10, 8)]
rows = []
for f in CUBES:
    lab, _ = nrrd.read(f)
    lab = lab[::2, ::2, ::2].astype(np.int32)        # 128^3, keeps the spike fast
    order = order_sheets(lab)
    name = os.path.basename(f)[:-10]
    if order is None:
        print(name, "skip: sheets not a simple stack"); continue
    W = winding_field(lab, order)
    psi0 = wrap(2 * np.pi * W)
    print(name, len(order), "sheets", flush=True)
    for sigma, nb, rad in SETTINGS:
        psi, fog = corrupt(psi0, lab, sigma, nb, rad)
        r = {"cube": name, "sigma": sigma, "blobs": nb, "radius": rad}
        for m, fn in METHODS.items():
            t = time.time(); r[m] = score(fn(psi), W, lab, fog); r[m + " s"] = round(time.time() - t, 1)
        rows.append(r); print("  ", r, flush=True)
json.dump(rows, open(os.path.expanduser("~/vesuvius/spike/results.json"), "w"), indent=1)
print("\nPar reglage, moyenne sur les cubes : % voxels hors du numero majoritaire de leur feuille / nb de paires de feuilles voisines mal ordonnees")
for s in SETTINGS:
    sel = [r for r in rows if (r["sigma"], r["blobs"], r["radius"]) == s]
    if sel:
        print(f"bruit={s[0]} blobs={s[1]} r={s[2]}: " + " | ".join(
            f"{m}: {np.mean([r[m][0] for r in sel]):.2f}% / {np.mean([r[m][1] for r in sel]):.1f}" for m in METHODS))
