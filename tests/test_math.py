import numpy as np
import pytest

from visualguard.calibration import calibrate
from visualguard.errors import VisualGuardError
from visualguard.metrics import aupro, image_metrics, latency_summary
from visualguard.models import nearest_distances


def test_knn_hand_calculation():
    np.testing.assert_allclose(
        nearest_distances(np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 0.0], [6.0, 8.0]])),
        [0, 5],
    )


def test_known_auroc_and_frozen_decision():
    result = image_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5, bootstrap=20)
    assert result["image_auroc"] == 0.75
    assert result["tn"] == 2 and result["fn"] == 1 and result["tp"] == 1
    assert image_metrics([0, 0], [0.1, 0.2], 0.2)["image_auroc"] is None


def test_calibration_only_normal_val(dataset):
    _, samples = dataset
    validation = [s for s in samples if s.split == "val"]
    scores = np.arange(len(validation), dtype=float)
    result = calibrate(validation, scores)
    assert result.image_threshold == scores[-1]
    assert result.empirical_false_alarm_rate == 0
    with pytest.raises(VisualGuardError):
        calibrate([s for s in samples if s.split == "test"], [0] * 10)
    with pytest.raises(VisualGuardError):
        calibrate(validation, [float("nan")] * len(validation))


def test_aupro_perfect_and_all_tied():
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 1:3, 1:3] = True
    assert aupro(masks, masks.astype(float)) == pytest.approx(1)
    assert aupro(masks, np.ones_like(masks, dtype=float)) == pytest.approx(0.15)
    assert aupro(np.zeros_like(masks), np.zeros_like(masks, dtype=float)) is None


def test_aupro_equal_region_weight():
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 1:4, 1:4] = True
    scores = np.zeros_like(masks, dtype=float)
    scores[0, 1, 1] = 2
    scores[~masks] = 1
    # The tiny detected component contributes half the PRO, not 1/10.
    assert aupro(masks, scores) == pytest.approx(0.5)


def test_latency_units():
    result = latency_summary([10, 20, 30, 40])
    assert result["median_ms"] == 25
    assert result["p95_ms"] == pytest.approx(38.5)
    assert result["sequential_throughput_fps"] == 40
    with pytest.raises(VisualGuardError):
        latency_summary([0])
