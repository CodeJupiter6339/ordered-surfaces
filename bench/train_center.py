"""Test 7 : petit 3D U-Net qui apprend le 'centre de feuille' (distance au bord de SA feuille).
Deux feuilles collees -> creux a leur jonction, meme sans creux d'intensite.
Entraine sur une moitie des cubes (crc32 % 2 == TRAIN), predit tous les cubes. Usage : train_center.py [TRAIN] [minutes]"""
import os, sys, glob, zlib, time, numpy as np, nrrd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 12
D = os.path.expanduser("~/vesuvius/spike/")
OUT = D + f"center_pred_train{TRAIN}/"; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); rng = np.random.default_rng(0)

cubes = []
for f in sorted(glob.glob(D + "cubes/*_mask.nrrd")):
    name = os.path.basename(f)[:17]
    lab, _ = nrrd.read(f); vol, _ = nrrd.read(f.replace("_mask", "_volume"))
    lab = lab[::2, ::2, ::2].astype(np.int32); vol = vol[::2, ::2, ::2].astype(np.float32)
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    tgt = np.zeros(lab.shape, np.float32)
    for i in np.unique(lab):
        if i: tgt = np.maximum(tgt, np.minimum(ndi.distance_transform_edt(lab == i), 3) / 3)
    cubes.append((name, vol, tgt, zlib.crc32(name.encode()) % 2))
tr = [c for c in cubes if c[3] == TRAIN]
print(f"{len(cubes)} cubes, {len(tr)} pour l'entrainement", flush=True)


def block(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o), nn.LeakyReLU(0.1),
                         nn.Conv3d(o, o, 3, padding=1), nn.InstanceNorm3d(o), nn.LeakyReLU(0.1))


class UNet(nn.Module):
    def __init__(s, c=(16, 32, 64)):
        super().__init__()
        s.e1, s.e2, s.e3 = block(1, c[0]), block(c[0], c[1]), block(c[1], c[2])
        s.u2, s.d2 = nn.ConvTranspose3d(c[2], c[1], 2, 2), block(2 * c[1], c[1])
        s.u1, s.d1 = nn.ConvTranspose3d(c[1], c[0], 2, 2), block(2 * c[0], c[0])
        s.out = nn.Conv3d(c[0], 1, 1)

    def forward(s, x):
        a = s.e1(x); b = s.e2(F.max_pool3d(a, 2)); c = s.e3(F.max_pool3d(b, 2))
        b = s.d2(torch.cat([s.u2(c), b], 1)); a = s.d1(torch.cat([s.u1(b), a], 1))
        return s.out(a)


def batch(n=4, p=64):
    xs, ys = [], []
    for _ in range(n):
        _, v, t, _ = tr[rng.integers(len(tr))]
        o = rng.integers(0, 128 - p + 1, 3)
        x = v[o[0]:o[0] + p, o[1]:o[1] + p, o[2]:o[2] + p]; y = t[o[0]:o[0] + p, o[1]:o[1] + p, o[2]:o[2] + p]
        for ax in range(3):
            if rng.random() < 0.5: x, y = np.flip(x, ax), np.flip(y, ax)
        perm = rng.permutation(3); x, y = x.transpose(perm), y.transpose(perm)
        xs.append(x.copy()); ys.append(y.copy())
    return torch.from_numpy(np.stack(xs)[:, None]).cuda(), torch.from_numpy(np.stack(ys)[:, None]).cuda()


net = UNet().cuda(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
t0, step = time.time(), 0
while time.time() - t0 < MINUTES * 60:
    x, y = batch()
    with torch.autocast("cuda", torch.bfloat16):
        loss = F.mse_loss(torch.sigmoid(net(x).float()), y)
    opt.zero_grad(); loss.backward(); opt.step(); step += 1
    if step % 200 == 0: print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

net.eval()
torch.save(net.state_dict(), OUT + "center_unet.pt")
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    for name, v, t, h in cubes:
        p = torch.sigmoid(net(torch.from_numpy(v[None, None]).cuda()).float())[0, 0].cpu().numpy()
        np.save(OUT + name + ".npy", p.astype(np.float16))
        if h != TRAIN: pass
mse = {h: np.mean([np.mean((np.load(OUT + n + ".npy").astype(np.float32) - t) ** 2) for n, _, t, hh in cubes if hh == h]) for h in (0, 1)}
print(f"fini : {step} pas | MSE train {mse[TRAIN]:.4f} test {mse[1 - TRAIN]:.4f}", flush=True)
