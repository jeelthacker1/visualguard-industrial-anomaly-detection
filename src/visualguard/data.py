from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .errors import VisualGuardError
from .images import difference_hash, mask_array, pixel_hash, read_image
from .utils import canonical_hash, resolve_inside, safe_name, sha256_file, utc_now, write_json


@dataclass(frozen=True)
class Sample:
    path: str
    category: str
    source: str
    defect: str
    label: int
    mask: str
    width: int
    height: int
    sha256: str
    pixels_sha256: str
    dhash: str
    mask_sha256: str = ""
    group_id: str = ""
    split: str = ""


def audit_dataset(
    root: Path, category: str, *, groups: dict[str, str] | None = None
) -> tuple[list[Sample], dict]:
    root = root.resolve()
    safe_name(category)
    folder = resolve_inside(root, category)
    if not folder.is_dir():
        raise VisualGuardError(f"Category folder missing: {folder}")
    samples, errors, referenced_masks = [], [], set()
    for source in ("train", "test"):
        for path in sorted((folder / source).glob("*/*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                resolve_inside(root, relative)
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    raise VisualGuardError("Unexpected file in image directory.")
                defect = path.parent.name
                if source == "train" and defect != "good":
                    raise VisualGuardError(
                        "Training directory contains anomalous or unknown labels."
                    )
                image = read_image(path)
                mask, mask_digest = "", ""
                if defect != "good":
                    mask_path = folder / "ground_truth" / defect / f"{path.stem}_mask.png"
                    mask = mask_path.relative_to(root).as_posix()
                    resolve_inside(root, mask)
                    if not mask_path.is_file():
                        raise VisualGuardError("Defective test image has no matching mask.")
                    values = mask_array(mask_path, original_size=image.size)
                    if not values.any():
                        raise VisualGuardError("Defect mask contains no positive pixels.")
                    mask_digest = sha256_file(mask_path)
                    referenced_masks.add(mask)
                digest = pixel_hash(image)
                samples.append(
                    Sample(
                        relative,
                        category,
                        source,
                        defect,
                        int(defect != "good"),
                        mask,
                        image.width,
                        image.height,
                        sha256_file(path),
                        digest,
                        difference_hash(image),
                        mask_digest,
                        (groups or {}).get(relative, digest),
                    )
                )
            except (VisualGuardError, OSError) as exc:
                errors.append({"path": relative, "error": str(exc)})
    for path in sorted((folder / "ground_truth").glob("*/*")):
        if path.is_file() and path.relative_to(root).as_posix() not in referenced_masks:
            errors.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "error": "Orphan or invalid ground-truth mask.",
                }
            )
    train = [s for s in samples if s.source == "train"]
    test = [s for s in samples if s.source == "test"]
    if len(train) < 5 or not test or {s.label for s in test} != {0, 1}:
        errors.append(
            {
                "path": category,
                "error": "Need at least five normal training images and both test classes.",
            }
        )
    duplicates = defaultdict(list)
    for sample in samples:
        duplicates[sample.pixels_sha256].append(sample)
    exact = [[s.path for s in group] for group in duplicates.values() if len(group) > 1]
    for group in duplicates.values():
        if len({s.source for s in group}) > 1 or len({s.label for s in group}) > 1:
            errors.append(
                {
                    "path": group[0].path,
                    "error": "Exact duplicate crosses official splits or labels.",
                }
            )
    near_pairs, near_count = [], 0
    # Perceptual hashes flag candidates, not proof of identity. Never silently deduplicate test data.
    for index, left in enumerate(samples):
        for right in samples[index + 1 :]:
            distance = (int(left.dhash, 16) ^ int(right.dhash, 16)).bit_count()
            if distance <= 2 and left.pixels_sha256 != right.pixels_sha256:
                near_count += 1
                if len(near_pairs) < 1000:
                    near_pairs.append({"a": left.path, "b": right.path, "hamming": distance})
    report = {
        "category": category,
        "created_utc": utc_now(),
        "images": len(samples),
        "train_images": len(train),
        "test_images": len(test),
        "errors": errors,
        "exact_duplicate_groups": exact,
        "near_duplicate_candidate_count": near_count,
        "near_duplicate_candidates_first_1000": near_pairs,
        "near_duplicate_policy": "Review candidates; supply physical-item group IDs where known. dHash alone does not establish identity.",
        "ok": not errors,
    }
    return samples, report


def make_splits(
    samples: list[Sample], *, seed: int = 42, validation_fraction: float = 0.2
) -> list[Sample]:
    if not 0 < validation_fraction < 1:
        raise VisualGuardError("Validation fraction must be between zero and one.")
    groups = sorted({s.group_id for s in samples if s.source == "train"})
    if len(groups) < 5:
        raise VisualGuardError("At least five independent normal training groups are required.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation_count = max(1, int(np.ceil(len(groups) * validation_fraction)))
    validation = set(groups[:validation_count])
    rows = [
        replace(
            s,
            split="test" if s.source == "test" else "val" if s.group_id in validation else "train",
        )
        for s in samples
    ]
    validate_splits(rows)
    return sorted(rows, key=lambda s: s.path)


def validate_splits(samples: list[Sample]) -> None:
    if not samples or len({s.category for s in samples}) != 1:
        raise VisualGuardError("A split must contain exactly one category.")
    if len({s.path for s in samples}) != len(samples):
        raise VisualGuardError("Repeated paths in manifest.")
    if {s.split for s in samples} != {"train", "val", "test"}:
        raise VisualGuardError("All train/val/test splits must be nonempty.")
    hashes, groups = defaultdict(set), defaultdict(set)
    for row in samples:
        parts = row.path.split("/")
        if len(parts) != 4 or parts[:3] != [row.category, row.source, row.defect]:
            raise VisualGuardError("Path and sample provenance disagree.")
        if (
            row.source not in {"train", "test"}
            or row.label not in {0, 1}
            or row.label != int(row.defect != "good")
        ):
            raise VisualGuardError("Invalid source or label.")
        if (row.split == "test") != (row.source == "test"):
            raise VisualGuardError(
                "Official test samples cannot be used for fitting or calibration."
            )
        if row.split in {"train", "val"} and (row.source != "train" or row.label != 0):
            raise VisualGuardError("Fitting and calibration require normal training provenance.")
        hashes[row.pixels_sha256].add(row.split)
        if row.group_id:
            groups[row.group_id].add(row.split)
        if row.label and (not row.mask or not row.mask_sha256):
            raise VisualGuardError("Anomalous sample lacks a mask fingerprint.")
    if any(len(v) > 1 for v in hashes.values()) or any(len(v) > 1 for v in groups.values()):
        raise VisualGuardError("Content duplicate or physical group leakage across splits.")


def save_manifest(path: Path, samples: list[Sample], *, seed: int) -> str:
    validate_splits(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".meta.json").exists():
        raise VisualGuardError("Manifest already exists; use a new path to preserve provenance.")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in samples)
    digest = sha256_file(path)
    write_json(
        path.with_suffix(".meta.json"),
        {
            "sha256": digest,
            "seed": seed,
            "created_utc": utc_now(),
            "protocol": "normal-group-holdout-v1",
            "rows": len(samples),
        },
    )
    return digest


def load_manifest(path: Path, root: Path, *, verify_files: bool = True) -> list[Sample]:
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if metadata["sha256"] != sha256_file(path):
        raise VisualGuardError("Manifest was modified after locking.")
    with path.open(newline="", encoding="utf-8") as handle:
        samples = [
            Sample(
                **{
                    **r,
                    "label": int(r["label"]),
                    "width": int(r["width"]),
                    "height": int(r["height"]),
                }
            )
            for r in csv.DictReader(handle)
        ]
    validate_splits(samples)
    for sample in samples:
        path_in_root = resolve_inside(root, sample.path)
        mask_in_root = resolve_inside(root, sample.mask) if sample.mask else None
        if verify_files:
            if sha256_file(path_in_root) != sample.sha256:
                raise VisualGuardError(f"Image changed after audit: {sample.path}")
            if mask_in_root and sha256_file(mask_in_root) != sample.mask_sha256:
                raise VisualGuardError(f"Mask changed after audit: {sample.mask}")
    return samples


def require_normal(samples: list[Sample], split: str) -> None:
    if not samples or any(s.label or s.source != "train" or s.split != split for s in samples):
        raise VisualGuardError(
            f"Expected exclusively normal {split} samples from official training data."
        )


def split_fingerprint(samples: list[Sample]) -> str:
    return canonical_hash([asdict(s) for s in sorted(samples, key=lambda row: row.path)])
