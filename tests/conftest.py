import io

import numpy as np
import pytest
from PIL import Image

from visualguard.data import audit_dataset, make_splits
from visualguard.synthetic import generate


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    generate(root)
    samples, report = audit_dataset(root, "synthetic_part")
    assert report["ok"]
    return root, make_splits(samples)


@pytest.fixture
def png():
    buffer = io.BytesIO()
    image = Image.fromarray(
        np.random.default_rng(42).integers(40, 200, (64, 64, 3), dtype=np.uint8)
    )
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fake_model():
    class FakeDetector:
        def predict(self, images):
            return np.full(len(images), 0.2), np.full((len(images), 32, 32), 0.1)

    meta = {
        "model_id": "test-model",
        "config": {"pretrained": True, "method": "patchcore"},
        "provenance": {"category": "bottle", "quality_envelope": {}},
        "calibration": {"image_threshold": 0.5, "pixel_threshold": 0.3},
    }
    return FakeDetector(), meta
