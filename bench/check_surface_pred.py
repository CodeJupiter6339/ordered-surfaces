"""Le modele de surface voit-il les feuilles du Scroll 1 ? Compare proba de surface vs masque annote."""
import os, glob, numpy as np, nrrd
D = os.path.expanduser("~/vesuvius/spike/")
fs = sorted(glob.glob(D + "surf_pred/*.npy"))
from sklearn.metrics import roc_auc_score
aucs, dices = [], []
for f in fs:
    name = os.path.basename(f)[:-4]
    m = glob.glob(D + f"cubes/{name}*_mask.nrrd")
    if not m: continue
    lab, _ = nrrd.read(m[0]); lab = lab[::2, ::2, ::2] > 0
    p = np.load(f).astype(np.float32)
    v, _ = nrrd.read(m[0].replace("_mask", "_volume")); v = v[::2, ::2, ::2].astype(np.float32)
    aucs.append((roc_auc_score(lab.ravel(), p.ravel()), roc_auc_score(lab.ravel(), v.ravel())))
    b = p > 0.5; dices.append(2 * (b & lab).sum() / (b.sum() + lab.sum()))
a = np.array(aucs)
print(f"{len(a)} cubes | AUC modele {a[:,0].mean():.3f} vs AUC intensite brute {a[:,1].mean():.3f} | Dice@0.5 {np.mean(dices):.3f}")
