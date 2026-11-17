"""Predit la surface recto (modele surface_recto_3dunet de l'equipe) sur les cubes annotes du Scroll 1.
Le modele n'a PAS ete entraine sur le Scroll 1 (PHerc 0139, 1667, 0343P, 0500P2, MAN Bp) : test hors distribution honnete.
Sortie : ~/vesuvius/spike/surf_pred/<cube>.npy (128^3, float16, proba de surface, meme sous-echantillonnage x2)."""
import os, glob, sys, json, time, numpy as np, torch, nrrd
from huggingface_hub import hf_hub_download
from vesuvius.models.build.build_network_from_config import NetworkFromConfig

ck_path = hf_hub_download("scrollprize/surface_recto_3dunet", "checkpoint_inference_ready.pth")
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
cfg = ck["model_config"]


class W:
    def __init__(self, d):
        self.__dict__.update(d)
        self.model_config = d
        self.enable_deep_supervision = False
        ps = d.get("train_patch_size") or d.get("patch_size") or [128, 128, 128]
        self.train_patch_size = tuple(ps)
        self.spacing = d.get("spacing") or [1] * len(ps)
        self.in_channels = d.get("in_channels", 1)
        self.train_batch_size = d.get("batch_size", 2)
        self.model_name = d.get("model_name", "surface")
        self.autoconfigure = d.get("autoconfigure", True)
    def __getattr__(self, k):  # attributs absents : None
        return None


print({k: cfg[k] for k in cfg if k in ("train_patch_size", "patch_size", "targets", "in_channels", "autoconfigure", "spacing")})

net = NetworkFromConfig(W(cfg))
missing = net.load_state_dict(ck["model"], strict=False)
print("missing", len(missing.missing_keys), "unexpected", len(missing.unexpected_keys))
net = net.cuda().eval().half()
OUT = os.path.expanduser("~/vesuvius/spike/surf_pred"); os.makedirs(OUT, exist_ok=True)
P, STEP = 128, 96
cubes = sorted(glob.glob(os.path.expanduser("~/vesuvius/spike/cubes/*_volume.nrrd")))
for f in cubes:
    name = os.path.basename(f)[:17]; out = f"{OUT}/{name}.npy"
    if os.path.exists(out): continue
    t = time.time()
    v, _ = nrrd.read(f); v = v.astype(np.float32); v = (v - v.mean()) / (v.std() + 1e-6)
    acc = np.zeros(v.shape, np.float32); wsum = np.zeros(v.shape, np.float32)
    w1 = np.hanning(P + 2)[1:-1]; wpatch = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).astype(np.float32)
    starts = list(range(0, 256 - P + 1, STEP)) + [256 - P]
    starts = sorted(set(starts))
    with torch.no_grad():
        for z in starts:
            for y in starts:
                for x in starts:
                    patch = torch.from_numpy(v[z:z + P, y:y + P, x:x + P][None, None]).cuda().half()
                    o = net(patch)
                    if isinstance(o, dict): o = list(o.values())[0]
                    if isinstance(o, (list, tuple)): o = o[0]
                    prob = torch.softmax(o.float(), 1)[0, 1].cpu().numpy()
                    acc[z:z + P, y:y + P, x:x + P] += prob * wpatch; wsum[z:z + P, y:y + P, x:x + P] += wpatch
    pred = (acc / np.maximum(wsum, 1e-6))[::2, ::2, ::2].astype(np.float16)
    np.save(out, pred)
    print(name, round(time.time() - t, 1), "s", "moy", float(pred.mean()), flush=True)
