import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric2 import score2, truth2
d = np.load(os.path.expanduser("~/vesuvius/spike/bench_cache_dirty.npz"))
res_ok, res_sw = [], []
for j in range(0, 1481, 37):
    L = d["L"][j].astype(np.int32); keep, valid, contact = truth2(L)
    zz = np.arange(L.shape[2])
    S = np.stack([np.where((L == i).any(2), (np.where(L == i, zz, 0).sum(2) / np.maximum((L == i).sum(2), 1)).round().astype(int), 0) for i in keep])
    # recentrer sur un voxel reellement dans la feuille
    for k, i in enumerate(keep):
        m = L == i
        ok = np.take_along_axis(m, S[k][..., None], 2)[..., 0]
        first = m.argmax(2); S[k] = np.where(ok, S[k], first)
    res_ok.append(score2(S, L)["suivi"])
    Ssw = S.copy(); h = L.shape[1] // 2
    Ssw[0, :, h:], Ssw[1, :, h:] = S[1, :, h:], S[0, :, h:]
    res_sw.append(score2(Ssw, L)["suivi"])
print(f"parfait : {np.mean(res_ok):.1f} % (attendu ~100) | echange des 2 premieres feuilles a mi-morceau : {np.mean(res_sw):.1f} %")
