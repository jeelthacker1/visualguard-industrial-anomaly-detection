from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from .artifacts import load_artifact, save_artifact
from .calibration import calibrate
from .data import load_manifest, split_fingerprint
from .errors import VisualGuardError
from .images import image_quality, mask_array, read_image
from .metrics import image_metrics, latency_summary, pixel_metrics
from .models import Detector, ModelConfig, configure_torch
from .utils import environment, resolve_inside, sha256_file, utc_now, write_json


def score_samples(detector, samples, root: Path):
    scores, maps = [], []
    for start in range(0, len(samples), detector.config.batch_size):
        images = [
            read_image(resolve_inside(root, s.path))
            for s in samples[start : start + detector.config.batch_size]
        ]
        batch_scores, batch_maps = detector.predict(images)
        scores.extend(batch_scores.tolist())
        if batch_maps is not None:
            maps.extend(batch_maps)
    return np.asarray(scores), np.stack(maps) if maps else None


def train(root: Path, manifest: Path, output: Path, config: ModelConfig) -> dict:
    if output.exists():
        raise VisualGuardError("Training output already exists; use a new immutable model version.")
    samples = load_manifest(manifest, root)
    training = [s for s in samples if s.split == "train"]
    validation = [s for s in samples if s.split == "val"]
    synthetic = (root / "synthetic_provenance.json").is_file()
    if not config.pretrained and not synthetic:
        raise VisualGuardError(
            "Random-weight models are restricted to the clearly labeled synthetic smoke dataset."
        )
    configure_torch(config)
    detector = Detector(config, initialize_pretrained=config.pretrained)
    start = perf_counter()
    detector.fit(training, root)
    fit_seconds = perf_counter() - start
    scores, maps = score_samples(detector, validation, root)
    calibration = calibrate(validation, scores, maps, quantile=config.quantile)
    quality = [asdict(image_quality(read_image(resolve_inside(root, s.path)))) for s in training]
    envelope = {}
    for name in quality[0]:
        values = [q[name] for q in quality]
        margin = max((max(values) - min(values)) * 0.2, 0.001)
        envelope[name] = [max(0.0, min(values) - margin), max(values) + margin]
    provenance = {
        "category": samples[0].category,
        "dataset": "synthetic-smoke" if synthetic else "MVTec AD",
        "manifest_sha256": sha256_file(manifest),
        "train_fingerprint": split_fingerprint(training),
        "test_fingerprint": split_fingerprint([s for s in samples if s.split == "test"]),
        "normal_validation_count": len(validation),
        "train_count": len(training),
        "fit_seconds": fit_seconds,
        "quality_envelope": envelope,
        "hardware": environment(),
        "device": "cpu",
        "threshold_frozen_utc": utc_now(),
    }
    return save_artifact(output, detector, calibration, provenance)


def evaluate(
    root: Path,
    manifest: Path,
    artifact: Path,
    output: Path,
    *,
    repetitions: int = 100,
    allow_repeat: bool = False,
    corruption: str = "none",
) -> dict:
    if repetitions < 1 or repetitions > 10000:
        raise VisualGuardError("Timing repetitions must lie between 1 and 10000.")
    if output.exists():
        raise VisualGuardError("Evaluation output already exists; never replace previous evidence.")
    if corruption not in {"none", "blur", "noise", "brightness"}:
        raise VisualGuardError("Unsupported preregistered corruption.")
    samples = load_manifest(manifest, root)
    detector, meta = load_artifact(artifact, category=samples[0].category)
    configure_torch(detector.config)
    if sha256_file(manifest) != meta["provenance"]["manifest_sha256"]:
        raise VisualGuardError("Evaluation manifest differs from the one locked during training.")
    tests = [s for s in samples if s.split == "test"]
    if split_fingerprint(tests) != meta["provenance"]["test_fingerprint"]:
        raise VisualGuardError("Official test membership changed after fitting.")
    marker = artifact / "evaluation_started.json"
    previously_evaluated = marker.exists()
    if previously_evaluated and not allow_repeat:
        raise VisualGuardError(
            "This artifact has already entered evaluation. Use --allow-repeat to label a diagnostic repeat explicitly."
        )
    if not previously_evaluated:
        write_json(
            marker,
            {
                "started_utc": utc_now(),
                "model_id": meta["model_id"],
                "test_fingerprint": split_fingerprint(tests),
            },
        )
    output.mkdir(parents=True)
    scores, maps, masks, loaded_images = [], [], [], []
    rng = np.random.default_rng(detector.config.seed)
    from PIL import Image, ImageEnhance, ImageFilter

    for sample in tests:
        image = read_image(resolve_inside(root, sample.path))
        if corruption == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        elif corruption == "brightness":
            image = ImageEnhance.Brightness(image).enhance(0.6)
        elif corruption == "noise":
            values = np.asarray(image, dtype=np.float32) + rng.normal(
                0, 10, (image.height, image.width, 3)
            )
            image = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))
        prediction, anomaly_map = detector.predict([image])
        scores.append(float(prediction[0]))
        if len(loaded_images) < 20:
            loaded_images.append(image)
        if anomaly_map is not None:
            maps.append(anomaly_map[0])
            mask = (
                mask_array(
                    resolve_inside(root, sample.mask),
                    original_size=(sample.width, sample.height),
                    size=detector.config.input_size,
                )
                if sample.mask
                else np.zeros_like(anomaly_map[0], dtype=bool)
            )
            masks.append(mask)
    threshold = meta["calibration"]["image_threshold"]
    image_result = image_metrics([s.label for s in tests], scores, threshold)
    pixel_result = (
        pixel_metrics(np.stack(masks), np.stack(maps), meta["calibration"]["pixel_threshold"])
        if maps
        else None
    )
    for index in range(10):
        detector.predict([loaded_images[index % len(loaded_images)]])
    timings, rss_samples = [], []
    process = psutil.Process()
    for index in range(repetitions):
        start = perf_counter()
        detector.predict([loaded_images[index % len(loaded_images)]])
        timings.append((perf_counter() - start) * 1000)
        rss_samples.append(process.memory_info().rss)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "category": samples[0].category,
        "method": detector.config.method,
        "model_id": meta["model_id"],
        "dataset": meta["provenance"]["dataset"],
        "manifest_sha256": sha256_file(manifest),
        "config": detector.config.to_dict(),
        "image": image_result,
        "pixel": pixel_result,
        "timing": {
            **latency_summary(timings),
            "warmup": 10,
            "scope": "batch=1 CPU preprocessing + feature extraction + scoring; preloaded image; excludes decode, network and UI",
            "sampled_process_rss_max_bytes": max(rss_samples),
            "peak_memory_measured": False,
        },
        "bank_bytes": meta["fit"]["bank_bytes"],
        "fit_seconds": meta["provenance"]["fit_seconds"],
        "hardware": environment(),
        "exploratory_repeat": previously_evaluated,
        "corruption": corruption,
        "threshold_fixed_before_test": True,
        "notes": [
            "Normal-only threshold does not guarantee a future false-alarm rate.",
            "Published benchmark accuracy does not establish factory or camera reliability.",
        ],
    }
    write_json(output / "metrics.json", result)
    with (output / "predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["path", "defect", "label", "score", "threshold", "predicted", "outcome"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample, score in zip(tests, scores, strict=True):
            predicted = int(score > threshold)
            outcome = (
                ("TP" if predicted else "FN") if sample.label else ("FP" if predicted else "TN")
            )
            writer.writerow(
                {
                    "path": sample.path,
                    "defect": sample.defect,
                    "label": sample.label,
                    "score": score,
                    "threshold": threshold,
                    "predicted": predicted,
                    "outcome": outcome,
                }
            )
    write_json(output / "latencies.json", {"milliseconds": timings})
    return result
