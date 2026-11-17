# ordered-surfaces

Sheet separation in small CT crops by fitting K **ordered** surfaces per column (exact DP, min gap between sheets), then smoothing across neighbouring columns. Nothing from the labels is used at run time: the column axis comes from the CT structure tensor and K from per-column peak counts.

## Data
Scroll 1 volumetric instance labels (80 harmonized 256³ cubes, `dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels`), downsampled ×2 to 128³ and cut into 32×32 column crops (stride 16). The benchmark keeps every crop with at least two sheets that cross the whole crop (1502 crops from 32 cubes); partial sheets may start or end inside the crop.

## Method
1. Axis: the cartesian axis closest to the dominant eigenvector of the gradient structure tensor (one axis per cube; oblique sheets are not resampled yet).
2. Signal: Gaussian-smoothed intensity minus a·∂²/∂z² (a=1, σ=2), z-scored.
3. K: 90th percentile of per-column peak counts (min distance = dmin, prominence 1.0).
4. Per-column DP for K ordered surfaces, gap dmin–60 voxels (dmin 9 at ×2 on Scroll 1; 6 also works).
5. 8 rounds: quadratic penalty toward the 11×11 median of each surface (λ=0.02), DP again.

`bench/gpu_dp.py` runs steps 4–5 in PyTorch on batches of crops and returns the same surfaces as the CPU version (checked column by column).

## Metric (`bench/metric2.py`)
- A sheet is a labelled instance that crosses ≥98% of the crop's columns.
- Each surface is matched to at most one sheet (Hungarian assignment), with the matching computed on columns where that sheet touches nothing.
- **Tracking** of a sheet = % of columns where its matched surface lies inside it (±1 voxel).
- **Contact tracking** = the same, restricted to columns where *that* sheet is within 1 voxel of another label.
- Crop scores are averaged per cube, then over cubes; 95% intervals are bootstrapped over cubes.
- Sanity check: perfect surfaces score 100%; swapping two sheets halfway through a crop scores 74%.

## Results (Scroll 1, 1502 crops, 32 cubes)

| Method | Tracking | 95% CI | At contacts | Cube half A / B |
|---|---|---|---|---|
| Intensity peaks taken in order (no DP) | 62.3% | 60.3–64.3 | 52.6% | 63.8 / 61.1 |
| Ordered DP, no smoothing | 70.5% | 68.7–72.4 | 66.9% | 70.9 / 70.2 |
| **Ordered DP + smoothing** | **80.6%** | 78.9–82.2 | **77.1%** | 80.6 / 80.5 |
| same, dmin 6 | 81.2% | 79.6–82.8 | 78.6% | 80.5 / 81.8 |

Ablations, all under this metric or the earlier one:
- 2×2×2 mean downsampling instead of `[::2]`: no change (80.6%).
- K = median instead of p90: −5.
- Letting surfaces sit on air (capped cost): 0 to −5.
- ∂² sharpening: +2.5.
- Subtracting the `scrollprize/surface_recto_3dunet` prediction as a boundary term (not trained on Scroll 1): −1 to −16; adding it: −0.3.
- A small 3D U-Net trained on half the cubes to predict sheet centerness, at ×2 or at full resolution, as a signal: no gain (±0.6).

## History
An earlier version of this README reported higher numbers (90–94% purity on 21 "clean" crops). That metric had flaws, pointed out in review and fixed here:
- contacts were pooled over all sheet pairs instead of the tracked sheet;
- matching between surfaces and sheets was not one-to-one;
- purity could be inflated by several surfaces sitting on one easy sheet;
- some settings were chosen on the crops used for scoring.

## Other scrolls
`bench/eval_kaggle.py` and `bench/reeval_kaggle.py` run the same chain on the Kaggle surface-detection cubes (recto labels, scrolls other than Scroll 1), with cubes 1–30 for tuning and 31–90 held out. Results under the corrected metric are pending. Under the earlier metric, smoothing gave +9 points there and a smaller dmin (5–6) worked best.

## Limits
- Labels exist only where annotators labelled, and crops are small (32×32 columns).
- One column axis per cube, so oblique or curved sheets are cut at an angle.
- Touching sheets are still the main failure mode (77% vs 81% overall).
- Not yet integrated with any tracer (VC3D GrowPatch / lasagna).

## Reproduce
Needs Python 3.12 with numpy, scipy, scikit-image, pynrrd (PyTorch for `gpu_dp.py`). Scripts expect the labels zip at `~/vesuvius/data/instance-labels/labels.zip` and write caches and logs to `~/vesuvius/spike/`.
```
python bench/extract_all_cubes.py   # unzip labels
python bench/reeval.py build        # builds the 1502-crop cache, runs the CPU table
python bench/reeval_gpu.py          # same table with the GPU DP
python bench/check_metric2.py       # metric sanity check
```
