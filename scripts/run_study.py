"""Execute the preregistered CPU study, never silently repeat a final evaluation."""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

from visualguard.utils import canonical_hash, sha256_file, utc_now, write_json

RUNS = [
    ("E00", "bottle", "baseline"),
    ("E01", "bottle", "patchcore_resnet18"),
    ("E02", "bottle", "patchcore_10pct"),
    ("E03", "bottle", "patchcore_224"),
    ("E04", "metal_nut", "baseline"),
    ("E05", "metal_nut", "patchcore_resnet18"),
    ("E06", "cable", "baseline"),
    ("E07", "cable", "patchcore_resnet18"),
]


def run(root: Path, output_root: Path = Path(".")):
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    plan = {
        "config_sha256": {
            name: sha256_file(Path(f"configs/{name}.yaml"))
            for name in sorted({row[2] for row in RUNS})
        },
        "runs": RUNS,
        "seed": 42,
        "timing_repetitions": 100,
        "model_selection": "All fixed configurations reported; no test-driven selection.",
        "candidate_cap": 20000,
        "normal_validation_fraction": 0.2,
        "corruptions": {
            "E08": "bottle E01, Gaussian blur radius 2",
            "E09": "bottle E01, additive noise std 10/255 seed 42",
        },
    }
    plan_hash = canonical_hash(plan)
    reference = Path(__file__).resolve().parents[1] / "experiments/preregistered_plan.json"
    if reference.exists():
        frozen = json.loads(reference.read_text())
        if (
            frozen["plan_sha256"] != canonical_hash(frozen["plan"])
            or plan_hash != frozen["plan_sha256"]
        ):
            raise RuntimeError("The configuration or run list differs from the published study.")
    plan_file = output_root / "experiments/preregistered_plan.json"
    if plan_file.exists():
        if json.loads(plan_file.read_text())["plan_sha256"] != plan_hash:
            raise RuntimeError("Experiment plan changed after freezing.")
    else:
        write_json(plan_file, {"created_utc": utc_now(), "plan_sha256": plan_hash, "plan": plan})
    for category in ("bottle", "metal_nut", "cable"):
        manifest = output_root / f"splits/{category}_seed42.csv"
        if not manifest.exists():
            command(
                [
                    "prepare",
                    "--root",
                    str(root),
                    "--category",
                    category,
                    "--manifest",
                    str(manifest),
                ],
                f"prepare_{category}",
            )
    for run_id, category, config in RUNS:
        artifact = output_root / "artifacts" / run_id
        result = output_root / "reports" / run_id
        manifest = output_root / f"splits/{category}_seed42.csv"
        if not artifact.exists():
            command(
                [
                    "train",
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--config",
                    f"configs/{config}.yaml",
                    "--output",
                    str(artifact),
                ],
                f"train_{run_id}",
            )
        if result.exists() and not all(
            (result / f).is_file() for f in ("metrics.json", "predictions.csv", "latencies.json")
        ):
            raise RuntimeError(
                f"Incomplete evaluation in {result}. Preserve it and follow the recovery guide."
            )
        if not result.exists():
            command(
                [
                    "evaluate",
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(result),
                ],
                f"evaluate_{run_id}",
            )
        print(f"Completed {run_id}: {category} {config}", flush=True)
        gc.collect()
    for run_id, corruption in [("E08", "blur"), ("E09", "noise")]:
        result = output_root / "reports" / run_id
        if result.exists() and not all(
            (result / f).is_file() for f in ("metrics.json", "predictions.csv", "latencies.json")
        ):
            raise RuntimeError(
                f"Incomplete evaluation in {result}. Preserve it and follow the recovery guide."
            )
        if not result.exists():
            command(
                [
                    "evaluate",
                    "--root",
                    str(root),
                    "--manifest",
                    str(output_root / "splits/bottle_seed42.csv"),
                    "--artifact",
                    str(output_root / "artifacts/E01"),
                    "--output",
                    str(result),
                    "--allow-repeat",
                    "--corruption",
                    corruption,
                ],
                f"evaluate_{run_id}",
            )


def command(arguments, name):
    folder = Path("runs/logs")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Starting {name}", flush=True)
    with (folder / f"{name}.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(  # noqa: S603 -- fixed CLI arguments, no shell
            [sys.executable, "-m", "visualguard.cli", *arguments],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )  # noqa: S603 -- fixed CLI, no shell
    if result.returncode:
        raise RuntimeError(
            f"{name} failed; inspect {folder / f'{name}.log'}. Existing evidence is preserved."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("local_data/MVTecAD"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Use runs/reproduction for a new independent reproduction without overwriting published evidence",
    )
    arguments = parser.parse_args()
    run(arguments.root, arguments.output_root)
