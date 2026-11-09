import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from visualguard.artifacts import load_artifact, save_artifact
from visualguard.calibration import calibrate
from visualguard.errors import VisualGuardError
from visualguard.images import read_image
from visualguard.models import Detector, ModelConfig, configure_torch
from visualguard.pipeline import score_samples

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        importlib.util.find_spec("anomalib") is None, reason="Optional model stack not installed"
    ),
]


@pytest.mark.parametrize("method", ["global_knn", "patchcore"])
def test_real_model_roundtrip_without_network(dataset, tmp_path, method):
    root, samples = dataset
    config = ModelConfig(
        method=method,
        input_size=32,
        coreset_ratio=0.3,
        num_neighbors=3,
        pretrained=False,
        threads=2,
    )
    configure_torch(config)
    model = Detector(config)
    training = [s for s in samples if s.split == "train"]
    validation = [s for s in samples if s.split == "val"]
    model.fit(training, root)
    scores, maps = score_samples(model, validation, root)
    assert np.isfinite(scores).all()
    if method == "global_knn":
        assert maps is None and model.bank.shape[1] == 512
    else:
        assert maps.shape == (len(validation), 32, 32)
    assert all(not parameter.requires_grad for parameter in model.model.parameters())
    calibration = calibrate(validation, scores, maps)
    artifact = tmp_path / "model"
    save_artifact(artifact, model, calibration, {"category": "synthetic_part"})
    restored, _ = load_artifact(artifact, category="synthetic_part")
    image = read_image(root / validation[0].path)
    np.testing.assert_allclose(model.predict([image])[0], restored.predict([image])[0], rtol=1e-5)
    with pytest.raises(VisualGuardError, match="category"):
        load_artifact(artifact, category="cable")
    state = artifact / "model.safetensors"
    state.write_bytes(state.read_bytes() + b"tampering")
    with pytest.raises(VisualGuardError, match="integrity"):
        load_artifact(artifact)


def test_training_rejects_test_labels(dataset):
    root, samples = dataset
    model = Detector(ModelConfig(method="global_knn", pretrained=False))
    with pytest.raises(VisualGuardError):
        model.fit([replace(samples[0], split="test")], root)
