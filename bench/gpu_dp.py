"""DP ordonnee + lissage sur GPU (PyTorch), en lot sur plusieurs morceaux.
Reproduit exactement msurf.solve_dp et fast_bench.smooth_dp (meme ordre de parcours, egalites tranchees pareil,
float64, median 11x11 avec bord 'reflect' de scipy = 'symmetric' de numpy)."""
import torch


def solve_dp_t(c, dmin, dmax):
    """c : (B, K, X, Y, Z) float64 -> S : (B, K, X, Y) int64."""
    B, K, X, Y, Z = c.shape
    D = c[:, 0].clone()
    backs = []
    inf = torch.tensor(float("inf"), dtype=c.dtype, device=c.device)
    for k in range(1, K):
        best = torch.full_like(D, float("inf")); arg = torch.zeros(D.shape, dtype=torch.int32, device=c.device)
        for d in range(dmin, min(dmax, Z - 1) + 1):
            cand = torch.full_like(D, float("inf")); cand[..., d:] = D[..., :-d]
            better = cand < best
            best = torch.where(better, cand, best); arg = torch.where(better, torch.tensor(d, dtype=torch.int32, device=c.device), arg)
        D = c[:, k] + best; backs.append(arg)
    S = torch.zeros((B, K, X, Y), dtype=torch.int64, device=c.device)
    S[:, -1] = D.argmin(-1)
    for k in range(K - 1, 0, -1):
        S[:, k - 1] = S[:, k] - torch.gather(backs[k - 1], -1, S[:, k][..., None]).squeeze(-1).long()
    return S


def _sym_index(n, pad, device):
    i = torch.arange(-pad, n + pad, device=device)
    i = torch.where(i < 0, -i - 1, i)
    return torch.where(i >= n, 2 * n - 1 - i, i)


def median2d(S, size):
    """S : (..., X, Y) -> mediane glissante size x size, bord symetrique (scipy mode='reflect')."""
    p = size // 2; X, Y = S.shape[-2:]
    Sp = S[..., _sym_index(X, p, S.device), :][..., _sym_index(Y, p, S.device)]
    win = Sp.unfold(-2, size, 1).unfold(-2, size, 1)  # (..., X, Y, size, size)
    return win.reshape(*win.shape[:-2], size * size).median(-1).values


def smooth_dp_t(P, K, dmin, dmax, iters=8, lam=0.02, size=11):
    """P : (B, X, Y, Z) float64 ; meme K pour tout le lot."""
    c = (-P)[:, None].expand(-1, K, -1, -1, -1).contiguous()
    S = solve_dp_t(c, dmin, dmax)
    zz = torch.arange(P.shape[-1], device=P.device, dtype=P.dtype)
    for _ in range(iters):
        ref = median2d(S.double(), size)
        S = solve_dp_t(c + lam * (zz - ref[..., None]) ** 2, dmin, dmax)
    return S
