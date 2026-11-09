from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import Sample, require_normal, split_fingerprint
from .errors import VisualGuardError


@dataclass(frozen=True)
class Calibration:
    image_threshold: float
    pixel_threshold: float | None
    quantile: float
    validation_count: int
    validation_fingerprint: str
    image_scores: list[float]
    empirical_false_alarm_rate: float
    method: str = "normal-only-quantile-higher-v1"
    decision_rule: str = "score > threshold"

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(samples: list[Sample], scores, maps=None, *, quantile: float = 0.99) -> Calibration:
    require_normal(samples, "val")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(samples),) or not np.isfinite(scores).all() or (scores < 0).any():
        raise VisualGuardError(
            "Calibration scores must be finite nonnegative values, one per validation image."
        )
    if not 0 < quantile < 1:
        raise VisualGuardError("Calibration quantile must lie strictly between zero and one.")
    threshold = float(np.quantile(scores, quantile, method="higher"))
    pixel_threshold = None
    if maps is not None:
        maps = np.asarray(maps)
        if (
            maps.ndim != 3
            or len(maps) != len(samples)
            or not np.isfinite(maps).all()
            or (maps < 0).any()
        ):
            raise VisualGuardError("Invalid normal validation anomaly maps.")
        pixel_threshold = float(np.quantile(maps.reshape(-1), quantile, method="higher"))
    return Calibration(
        threshold,
        pixel_threshold,
        quantile,
        len(samples),
        split_fingerprint(samples),
        scores.tolist(),
        float(np.mean(scores > threshold)),
    )
