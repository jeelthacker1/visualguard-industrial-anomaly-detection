from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .errors import VisualGuardError
from .images import PREPROCESSING_VERSION
from .models import Detector, ModelConfig
from .utils import canonical_hash, sha256_file, utc_now, write_json

SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 512 * 1024**2


def save_artifact(
    path: Path, detector: Detector, calibration: Calibration, provenance: dict
) -> dict:
    from safetensors.torch import save_file

    if path.exists():
        raise VisualGuardError("Artifact path already exists; model versions are immutable.")
    path.mkdir(parents=True)
    state_path = path / "model.safetensors"
    save_file(detector.state_dict(), str(state_path))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "config": detector.config.to_dict(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "calibration": calibration.to_dict(),
        "fit": detector.fit_details,
        "provenance": provenance,
        "state_sha256": sha256_file(state_path),
    }
    metadata["model_id"] = canonical_hash(metadata)[:20]
    write_json(path / "metadata.json", metadata)
    write_json(path / "integrity.json", {"metadata_sha256": sha256_file(path / "metadata.json")})
    return metadata


def load_artifact(path: Path, *, category: str | None = None) -> tuple[Detector, dict]:
    from safetensors.torch import load_file

    try:
        state_path, meta_path = path / "model.safetensors", path / "metadata.json"
        if (
            state_path.is_symlink()
            or meta_path.is_symlink()
            or meta_path.stat().st_size > 2 * 1024**2
        ):
            raise VisualGuardError("Unsafe artifact paths or metadata size.")
        integrity = json.loads((path / "integrity.json").read_text(encoding="utf-8"))
        if sha256_file(meta_path) != integrity["metadata_sha256"]:
            raise VisualGuardError("Artifact metadata integrity check failed.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta["schema_version"] != SCHEMA_VERSION
            or meta["preprocessing_version"] != PREPROCESSING_VERSION
        ):
            raise VisualGuardError("Artifact schema/preprocessing version is incompatible.")
        if (
            state_path.stat().st_size > MAX_ARTIFACT_BYTES
            or sha256_file(state_path) != meta["state_sha256"]
        ):
            raise VisualGuardError("Artifact weight integrity or size check failed.")
        if category is not None and category != meta["provenance"]["category"]:
            raise VisualGuardError(
                "Selected category does not match the model's training category."
            )
        calibration = Calibration(**meta["calibration"])
        if not np.isfinite(calibration.image_threshold) or calibration.image_threshold < 0:
            raise VisualGuardError("Invalid artifact image threshold.")
        if calibration.pixel_threshold is not None and (
            not np.isfinite(calibration.pixel_threshold) or calibration.pixel_threshold < 0
        ):
            raise VisualGuardError("Invalid artifact pixel threshold.")
        detector = Detector(ModelConfig(**meta["config"]), initialize_pretrained=False)
        detector.load_state_dict(load_file(str(state_path), device="cpu"))
        return detector, meta
    except FileNotFoundError as exc:
        raise VisualGuardError(
            "Model artifact is missing. Run the training command for this category first."
        ) from exc
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise VisualGuardError("Malformed artifact metadata.") from exc
