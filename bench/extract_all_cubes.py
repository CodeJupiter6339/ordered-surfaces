import zipfile, os
z = zipfile.ZipFile(os.path.expanduser("~/vesuvius/data/instance-labels/labels.zip"))
d = os.path.expanduser("~/vesuvius/spike/cubes/"); os.makedirs(d, exist_ok=True)
n = 0
for name in z.namelist():
    if "harmonized" in name and name.endswith(".nrrd"):
        p = d + os.path.basename(name)
        if not os.path.exists(p): open(p, "wb").write(z.read(name)); n += 1
print("nouveaux fichiers:", n, "| cubes:", len([f for f in os.listdir(d) if f.endswith("_mask.nrrd")]))
