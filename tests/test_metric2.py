import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "metric2", Path(__file__).parents[1] / "bench" / "metric2.py"
)
assert _spec is not None and _spec.loader is not None
_metric2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metric2)
aggregate, score2, truth2 = _metric2.aggregate, _metric2.score2, _metric2.truth2


def parallel_sheets(height=12, width=14, depth=32, zs=(8, 20)):
    labels = np.zeros((height, width, depth), dtype=np.int32)
    for sheet_id, z in enumerate(zs, start=1):
        labels[:, :, z] = sheet_id
    return labels


def surfaces(zs, shape=(12, 14)):
    return np.stack([np.full(shape, z, dtype=np.int32) for z in zs])


def test_perfect_surfaces_score_100_percent():
    labels = parallel_sheets()
    result = score2(surfaces((8, 20)), labels)
    assert result is not None
    assert result["suivi"] == 100.0
    assert result["n_feuilles"] == 2


def test_one_surface_cannot_count_for_two_sheets():
    labels = parallel_sheets()
    result = score2(surfaces((8, 8)), labels)
    assert result is not None
    assert result["suivi"] == 50.0


def test_halfway_sheet_swap_is_penalized():
    labels = parallel_sheets()
    pred = surfaces((8, 20))
    midpoint = pred.shape[2] // 2
    pred[:, :, midpoint:] = pred[::-1, :, midpoint:]
    result = score2(pred, labels)
    assert result is not None
    assert result["suivi"] == 50.0


def test_truth_requires_nearly_full_sheet_coverage():
    labels = parallel_sheets()
    labels[:, :2, 20] = 0
    keep, valid, _ = truth2(labels)
    assert keep == [1]
    assert valid.all()


def test_aggregate_bootstraps_over_cubes():
    rows = [{"suivi": 80.0}, {"suivi": 100.0}, {"suivi": 60.0}]
    result = aggregate(rows, ["a", "a", "b"], "suivi", nboot=200, seed=7)
    assert result is not None
    assert result["moy"] == 75.0
    assert result["cubes"] == 2
    assert result["ic95"][0] <= 60.0 <= result["ic95"][1]
    assert result["ic95"][0] <= 90.0 <= result["ic95"][1]
