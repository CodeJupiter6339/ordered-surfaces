"""Verifie que gpu_dp donne EXACTEMENT les memes surfaces que la version CPU, et mesure le temps."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_bench as fb
from sweep5 import sig
from gpu_dp import smooth_dp_t, median2d
from scipy import ndimage as ndi
d = np.load(os.path.expanduser("~/vesuvius/spike/bench_cache_dirty.npz"))
idx = list(range(0, 1481, 25))[:48]
P = np.stack([sig(d["V"][j], 1, 2) for j in idx]).astype(np.float64)
# median seule
S0 = np.random.default_rng(0).integers(0, 128, (3, 32, 32)).astype(np.float64)
m_cpu = np.stack([ndi.median_filter(S0[k], size=11) for k in range(3)])
m_gpu = median2d(torch.from_numpy(S0).cuda(), 11).cpu().numpy()
print("mediane identique :", np.array_equal(m_cpu, m_gpu))
for K in (3, 5):
    t = time.time(); Sc = np.stack([fb.smooth_dp(P[i], K, 9, 60, 8, 0.02, 11) for i in range(len(idx))]); tc = time.time() - t
    torch.cuda.synchronize(); t = time.time()
    Sg = smooth_dp_t(torch.from_numpy(P).cuda(), K, 9, 60, 8, 0.02, 11).cpu().numpy(); torch.cuda.synchronize(); tg = time.time() - t
    print(f"K={K} : {len(idx)} morceaux | CPU 1 coeur {tc:.1f}s | GPU {tg:.1f}s | identiques : {np.array_equal(Sc, Sg)} "
          f"(colonnes differentes : {100 * (Sc != Sg).mean():.3f} %)")
