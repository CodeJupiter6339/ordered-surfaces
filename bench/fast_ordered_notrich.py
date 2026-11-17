"""Test rapide : surfaces ordonnees SANS triche.
- direction des colonnes : tenseur de structure du CT (pas les annotations)
- nombre de feuilles K : mediane du nombre de pics par colonne (pics espaces d'au moins dmin)
Meme banc que spike_msurf (morceaux 32x32 propres), meme score."""
import os, sys, json, numpy as np, nrrd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.expanduser("~/vesuvius/spike"))
from msurf import solve_dp
src = open(os.path.join(HERE, "spike_unwrap.py")).read()
exec(src.split("def corrupt")[0])  # order_sheets, winding_field, CUBES
C, DMIN = 32, 9


def axis_from_ct(vol):
    g = [ndi.sobel(ndi.gaussian_filter(vol, 2), axis=a) for a in range(3)]
    T = np.array([[np.mean(g[i] * g[j]) for j in range(3)] for i in range(3)])
    w, v = np.linalg.eigh(T)
    return int(np.argmax(np.abs(v[:, -1])))  # axe le plus proche de la normale moyenne des feuilles


def prep(f):
    lab, _ = nrrd.read(f); vol, _ = nrrd.read(f.replace("_mask", "_volume"))
    lab = lab[::2, ::2, ::2].astype(np.int32); vol = vol[::2, ::2, ::2].astype(np.float32)
    order = order_sheets(lab)
    if order is None: return None
    W = winding_field(lab, order)
    a_true = int(np.argmax([np.abs(np.diff(W, axis=a).mean()) for a in range(3)]))
    a_ct = axis_from_ct(vol)
    lab, vol = np.moveaxis(lab, a_ct, 2).copy(), np.moveaxis(vol, a_ct, 2).copy()
    return lab, vol, a_true == a_ct


def crop_job(args):
    name, lab, vol, x0, y0 = args
    L = lab[x0:x0 + C, y0:y0 + C]; V = vol[x0:x0 + C, y0:y0 + C]
    ids = [i for i in np.unique(L) if i]
    pres = {i: (L == i).any(axis=2) for i in ids}
    keep = [i for i in ids if pres[i].mean() >= 0.98]
    if len(keep) < 2 or any(0.01 < pres[i].mean() < 0.98 for i in ids): return None
    zc = {i: np.mean(np.nonzero(L == i)[2]) for i in keep}
    keep.sort(key=lambda i: zc[i]); K_true = len(keep)
    valid = np.all([pres[i] for i in keep], axis=0)
    Z = L.shape[2]; zz = np.arange(Z)
    contact = np.zeros(valid.shape, bool)
    for a, b in zip(keep, keep[1:]):
        contact |= (np.where(L == b, zz, Z).min(2) - np.where(L == a, zz, -1).max(2)) <= 1
    P = ndi.gaussian_filter(V, (1, 1, 2)); P = (P - P.mean()) / (P.std() + 1e-6)
    # K estime : mediane du nombre de pics par colonne
    npk = [len(find_peaks(P[x, y], distance=DMIN, prominence=0.3)[0]) for x in range(C) for y in range(C)]
    K_est = max(2, int(np.median(npk)))
    S = solve_dp(np.repeat(-P[None], K_est, axis=0), dmin=DMIN, dmax=60)
    # score : pour chaque feuille vraie, la surface estimee la plus proche (en rang) tombe-t-elle dedans ?
    ok = np.zeros((K_true, C, C), bool)
    for k, i in enumerate(keep):
        m = L == i
        for kk in range(K_est):
            for d in (-1, 0, 1):
                z = np.clip(S[kk] + d, 0, Z - 1)
                ok[k] |= np.take_along_axis(m, z[..., None], 2)[..., 0]
    # ordre : surfaces estimees qui tombent dans 2 feuilles differentes selon la colonne = echange
    hit = np.full((K_est, C, C), -1)
    for k, i in enumerate(keep):
        for kk in range(K_est):
            inside = np.take_along_axis(L == i, np.clip(S[kk], 0, Z - 1)[..., None], 2)[..., 0]
            hit[kk][inside] = k
    swaps = 0
    for kk in range(K_est):
        h = hit[kk][valid & (hit[kk] >= 0)]
        if h.size: swaps += h.size - np.bincount(h).max()
    return {"cube": name, "K_true": K_true, "K_est": K_est, "cols": int(valid.sum()),
            "found": 100 * ok[:, valid].mean(),
            "found_contact": 100 * ok[:, valid & contact].mean() if (valid & contact).any() else None,
            "contact_cols": int((valid & contact).sum()),
            "swap_pct": 100 * swaps / max(1, sum((hit[kk][valid] >= 0).sum() for kk in range(K_est)))}


if __name__ == "__main__":
    with Pool(14) as pool:
        preps = pool.map(prep, CUBES)
    jobs, axis_ok = [], []
    for f, p in zip(CUBES, preps):
        if p is None: continue
        lab, vol, good = p; axis_ok.append(good)
        for x0 in range(0, 128 - C + 1, C // 2):
            for y0 in range(0, 128 - C + 1, C // 2):
                jobs.append((os.path.basename(f)[:17], lab, vol, x0, y0))
    with Pool(14) as pool:
        res = [r for r in pool.map(crop_job, jobs) if r]
    json.dump(res, open(os.path.expanduser("~/vesuvius/spike/results_notrich.json"), "w"), indent=1, default=float)
    print(f"axe CT = axe vrai sur {sum(axis_ok)}/{len(axis_ok)} cubes ; {len(res)} morceaux propres")
    print("K exact sur", sum(r["K_est"] == r["K_true"] for r in res), "/", len(res),
          "| ecart moyen", round(np.mean([abs(r["K_est"] - r["K_true"]) for r in res]), 2))
    w = [r["cols"] for r in res]
    print(f"feuilles retrouvees : {np.average([r['found'] for r in res], weights=w):.1f}% (toutes colonnes)")
    cc = [(r["found_contact"], r["contact_cols"]) for r in res if r["found_contact"] is not None]
    print(f"feuilles retrouvees aux contacts : {np.average([a for a,_ in cc], weights=[b for _,b in cc]):.1f}%")
    print(f"echanges de feuille (surface qui passe d'une feuille a l'autre) : {np.average([r['swap_pct'] for r in res], weights=w):.1f}%")
