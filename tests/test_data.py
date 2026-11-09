from dataclasses import replace

import pytest

from visualguard.data import (
    audit_dataset,
    load_manifest,
    make_splits,
    save_manifest,
    validate_splits,
)
from visualguard.errors import VisualGuardError
from visualguard.utils import resolve_inside


def test_splits_deterministic_and_normal(dataset):
    _, samples = dataset
    assert make_splits(samples) == make_splits(samples)
    assert make_splits(samples, seed=7) != make_splits(samples)
    validate_splits(samples)
    assert all(s.label == 0 and s.source == "train" for s in samples if s.split != "test")


def test_manifest_tamper_detected(dataset, tmp_path):
    root, samples = dataset
    path = tmp_path / "split.csv"
    save_manifest(path, samples, seed=42)
    assert load_manifest(path, root) == samples
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(VisualGuardError, match="modified"):
        load_manifest(path, root)


def test_image_change_detected(dataset, tmp_path):
    root, samples = dataset
    manifest = tmp_path / "split.csv"
    save_manifest(manifest, samples, seed=42)
    (root / samples[0].path).write_bytes(b"changed")
    with pytest.raises(VisualGuardError, match="changed"):
        load_manifest(manifest, root)


def test_exact_duplicate_cross_test_fails(dataset):
    root, _ = dataset
    training = root / "synthetic_part/train/good/000.png"
    testing = root / "synthetic_part/test/good/000.png"
    testing.write_bytes(training.read_bytes())
    _, report = audit_dataset(root, "synthetic_part")
    assert not report["ok"]
    assert report["exact_duplicate_groups"]


def test_duplicate_train_stays_same_split(dataset):
    _, samples = dataset
    normal = next(s for s in samples if s.source == "train")
    duplicate = replace(normal, path="synthetic_part/train/good/copy.png")
    rows = make_splits([*samples, duplicate])
    assert len({s.split for s in rows if s.pixels_sha256 == normal.pixels_sha256}) == 1


def test_explicit_physical_groups_do_not_leak(dataset):
    _, samples = dataset
    train = [s for s in samples if s.source == "train"]
    group_ids = {s.path: f"part-{i // 2}" for i, s in enumerate(train)}
    rows = make_splits([replace(s, group_id=group_ids.get(s.path, s.group_id)) for s in samples])
    for group in set(group_ids.values()):
        assert len({s.split for s in rows if s.group_id == group}) == 1


def test_wrong_provenance_rejected(dataset):
    _, samples = dataset
    row = next(s for s in samples if s.split == "test")
    invalid = [replace(s, split="val") if s.path == row.path else s for s in samples]
    with pytest.raises(VisualGuardError, match="Official test"):
        validate_splits(invalid)


def test_missing_mask_rejected(dataset):
    root, samples = dataset
    masked = next(s for s in samples if s.mask)
    (root / masked.mask).rename(root / "missing-mask.png")
    _, report = audit_dataset(root, "synthetic_part")
    assert not report["ok"]


@pytest.mark.parametrize(
    "path", ["../escape.png", "/etc/passwd", "C:/file.png", "foo\\bar.png", "", "a/../../b"]
)
def test_path_escape_rejected(tmp_path, path):
    with pytest.raises(VisualGuardError):
        resolve_inside(tmp_path, path)


def test_manifest_no_overwrite(dataset, tmp_path):
    _, samples = dataset
    path = tmp_path / "split.csv"
    save_manifest(path, samples, seed=42)
    with pytest.raises(VisualGuardError, match="already exists"):
        save_manifest(path, samples, seed=7)
