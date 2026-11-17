# ordered-surfaces

Sheet separation in small CT crops by fitting K **ordered** surfaces per column (exact DP, min gap between sheets), then smoothing across neighbouring columns. Nothing from the labels is used at run time: the sheet-normal axis comes from the CT structure tensor and K from per-column peak counts.

## Data
Scroll 1 volumetric instance labels (80 harmonized 256³ cubes, `dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels`), downsampled ×2 to 128³ and cut into 32×32 column crops.
- clean crops: every sheet crosses the whole crop (21 crops)
- dirty crops: some sheets start or end inside the crop (1481 crops; 400 sampled for some tests)

## Method
1. Axis: largest eigenvector of the gradient structure tensor (correct axis on 14/14 cubes checked).
2. Signal: Gaussian-smoothed intensity minus a·∂²/∂z² (a=1, σ=2), z-scored.
3. K: 90th percentile of per-column peak counts (min distance 9, prominence 1.0).
4. Per-column DP for K ordered surfaces, gap 9–60 voxels.
5. 8 rounds: quadratic penalty toward the 11×11 median of each surface (λ=0.02), DP again.

## Results
Baseline: per-column run counting on the *label mask* with K given (an upper bound for naive local counting).

| Bench | Baseline | This |
|---|---|---|
| Clean crops, purity, settings cross-validated across cube halves | 62% | 90–94% |
| Dirty crops (1481), purity | 47% | 69% |
| Dirty crops (400), full-sheet tracking | — | 81% |
| … at sheet contacts | — | 62% |

Purity: % of columns where a surface stays in its dominant sheet. Tracking: for each full sheet, % of columns where the best surface stays inside it.

Ablations (dirty crops, tracking): no smoothing −20 pts; K = median instead of p90 −5; letting surfaces sit on air (capped cost) 0 to −5; ∂² sharpening +2.5; subtracting the `scrollprize/surface_recto_3dunet` prediction as a boundary term (not trained on Scroll 1) −1 to −16, adding it −0.3.

## Limits
- One scroll (Scroll 1), small crops, labels only where annotators labelled.
- Touching sheets remain the failure mode (~62%): intensity alone has no boundary there.
- Not yet integrated with any tracer (VC3D / lasagna).

## Reproduce
Needs Python 3.12 with numpy, scipy, scikit-image, pynrrd. Scripts expect the labels zip at `~/vesuvius/data/instance-labels/labels.zip` and write caches and `bench_log.jsonl` to `~/vesuvius/spike/`.
```
python bench/extract_all_cubes.py   # unzip labels
python bench/fast_bench.py          # builds cache, ~2 s per variant afterwards
python bench/sweep3.py; python bench/sweep4.py; python bench/sweep5.py
```
