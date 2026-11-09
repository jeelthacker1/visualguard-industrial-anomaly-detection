from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter

import numpy as np

from .errors import VisualGuardError
from .images import PREPROCESSING_VERSION, decode_image, image_quality, pixel_hash
from .utils import utc_now


def validate_capture_timestamp(
    value: str | None, *, now: datetime | None = None, max_age_seconds: float = 2.0
) -> None:
    if value is None:
        return
    try:
        captured = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError("timezone required")
        age = ((now or datetime.now(UTC)) - captured).total_seconds()
    except (ValueError, TypeError) as exc:
        raise VisualGuardError("Capture timestamp must be timezone-aware ISO-8601.") from exc
    if age > max_age_seconds or age < -0.5:
        raise VisualGuardError("Frame timestamp is stale or more than 500 ms in the future.")


def inspect_bytes(
    data: bytes,
    detector,
    metadata: dict,
    *,
    captured_at: str | None = None,
    deadline_ms: float = 1000,
) -> tuple[dict, object, np.ndarray | None]:
    if not np.isfinite(deadline_ms) or deadline_ms <= 0:
        raise VisualGuardError("Inspection deadline must be positive and finite.")
    validate_capture_timestamp(captured_at)
    start = perf_counter()
    image = decode_image(data)
    quality = asdict(image_quality(image))
    model_start = perf_counter()
    scores, maps = detector.predict([image])
    model_ms = (perf_counter() - model_start) * 1000
    warnings = []
    for name, value in quality.items():
        bounds = metadata.get("provenance", {}).get("quality_envelope", {}).get(name)
        if bounds and not bounds[0] <= value <= bounds[1]:
            warnings.append(f"{name} is outside the training acquisition range")
    if image.width / image.height < 0.5 or image.width / image.height > 2:
        warnings.append("Extreme aspect ratio is distorted by square preprocessing")
    if not metadata["config"]["pretrained"]:
        warnings.append("Synthetic smoke-test model with random weights; no industrial validity")
    elapsed = (perf_counter() - start) * 1000
    deadline_missed = elapsed > deadline_ms
    if deadline_missed:
        warnings.append("Inspection deadline exceeded; do not use for automatic acceptance")
    # The acquisition clock is rechecked after inference, not only when it arrives.
    try:
        validate_capture_timestamp(captured_at)
    except VisualGuardError:
        warnings.append("Frame became stale during inspection")
    score, threshold = float(scores[0]), float(metadata["calibration"]["image_threshold"])
    decision = "review_required" if score > threshold or warnings else "normal_appearance"
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "captured_at": captured_at,
        "category": metadata["provenance"]["category"],
        "model_id": metadata["model_id"],
        "method": metadata["config"]["method"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "image_sha256": pixel_hash(image),
        "anomaly_score": score,
        "threshold": threshold,
        "pixel_threshold": metadata["calibration"]["pixel_threshold"],
        "decision": decision,
        "anomaly_above_threshold": score > threshold,
        "warnings": warnings,
        "latency_ms": elapsed,
        "model_and_preprocess_ms": model_ms,
        "deadline_ms": deadline_ms,
        "deadline_missed": deadline_missed,
        "quality": quality,
        "score_is_probability": False,
        "input_dimensions": [image.width, image.height],
        "map_dimensions": list(maps[0].shape) if maps is not None else None,
        "map_coordinate_system": "square resized image; overlay resizes back to original dimensions"
        if maps is not None
        else None,
    }
    return result, image, maps[0] if maps is not None else None
