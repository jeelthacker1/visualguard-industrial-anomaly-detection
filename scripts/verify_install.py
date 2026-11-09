"""Verify an installed wheel in isolated import mode against own synthetic fixtures."""

import json
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))

import visualguard  # noqa: E402
from visualguard.data import audit_dataset, load_manifest, make_splits, save_manifest  # noqa: E402
from visualguard.models import ModelConfig  # noqa: E402
from visualguard.pipeline import evaluate, train  # noqa: E402
from visualguard.synthetic import generate  # noqa: E402

if not Path(visualguard.__file__).is_relative_to(target):
    raise RuntimeError("Verification imported the workspace instead of the installed wheel.")
work = Path("runs/installed-wheel-smoke")
root = work / "data"
generate(root)
samples, audit = audit_dataset(root, "synthetic_part")
assert audit["ok"]
manifest = work / "split.csv"
save_manifest(manifest, make_splits(samples), seed=42)
assert load_manifest(manifest, root) == make_splits(samples)
config = ModelConfig(method="global_knn", input_size=32, pretrained=False, threads=2)
train(root, manifest, work / "artifact", config)
result = evaluate(root, manifest, work / "artifact", work / "evaluation", repetitions=10)
print(
    json.dumps(
        {
            "installed_wheel_import": True,
            "dataset": result["dataset"],
            "test_images": result["image"]["count"],
            "status": "passed",
        }
    )
)
