"""Test 8 : meme modele 'centre de feuille' mais entraine et predit a PLEINE resolution (256^3, pas de x2).
Hypothese : la jonction entre feuilles collees disparait au sous-echantillonnage. La prediction est ensuite
moyennee x2 pour rentrer dans le banc existant (sweep7 lit center_pred_full{TRAIN}). Usage : train_center_full.py TRAIN minutes"""
import os, sys, glob, zlib, time, numpy as np, nrrd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 15
D = os.path.expanduser("~/vesuvius/spike/")
OUT = D + f"center_pred_full{TRAIN}/"; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); rng = np.random.default_rng(0)
import importlib.util
spec = importlib.util.spec_from_file_location("tc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_center.py"))
src = open(spec.origin).read()
exec(src[src.index("def block"):src.index("def batch")])  # block, UNet

cubes = []
for f in sorted(glob.glob(D + "cubes/*_mask.nrrd")):
    name = os.path.basename(f)[:17]
    lab, _ = nrrd.read(f); vol, _ = nrrd.read(f.replace("_mask", "_volume"))
    lab = lab.astype(np.int32); vol = vol.astype(np.float32); vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    tgt = np.zeros(lab.shape, np.float32)
    for i in np.unique(lab):
        if i: tgt = np.maximum(tgt, np.minimum(ndi.distance_transform_edt(lab == i), 6) / 6)
    cubes.append((name, vol.astype(np.float16), tgt.astype(np.float16), zlib.crc32(name.encode()) % 2))
tr = [c for c in cubes if c[3] == TRAIN]
print(f"{len(cubes)} cubes pleine resolution, {len(tr)} pour l'entrainement", flush=True)


def batch(n=4, p=96):
    xs, ys = [], []
    for _ in range(n):
        _, v, t, _ = tr[rng.integers(len(tr))]
        o = rng.integers(0, 256 - p + 1, 3)
        x = v[o[0]:o[0] + p, o[1]:o[1] + p, o[2]:o[2] + p].astype(np.float32); y = t[o[0]:o[0] + p, o[1]:o[1] + p, o[2]:o[2] + p].astype(np.float32)
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
    if step % 500 == 0: print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
net.eval(); torch.save(net.state_dict(), OUT + "center_unet_full.pt")
P, ST = 128, 64
w1 = np.hanning(P + 2)[1:-1]; W = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).astype(np.float32)
mse = {0: [], 1: []}
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    for name, v, t, h in cubes:
        acc = np.zeros(v.shape, np.float32); ws = np.zeros(v.shape, np.float32)
        for a in range(0, 256 - P + 1, ST):
            for b in range(0, 256 - P + 1, ST):
                for c in range(0, 256 - P + 1, ST):
                    x = torch.from_numpy(v[a:a + P, b:b + P, c:c + P].astype(np.float32)[None, None]).cuda()
                    acc[a:a + P, b:b + P, c:c + P] += torch.sigmoid(net(x).float())[0, 0].cpu().numpy() * W
                    ws[a:a + P, b:b + P, c:c + P] += W
        p = acc / np.maximum(ws, 1e-6)
        mse[h].append(np.mean((p - t.astype(np.float32)) ** 2))
        np.save(OUT + name + ".npy", p.reshape(128, 2, 128, 2, 128, 2).mean((1, 3, 5)).astype(np.float16))
print(f"fini : {step} pas | MSE train {np.mean(mse[TRAIN]):.4f} test {np.mean(mse[1 - TRAIN]):.4f}", flush=True)
