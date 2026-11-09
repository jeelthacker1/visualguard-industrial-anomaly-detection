from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from .errors import VisualGuardError


def main():
    parser = argparse.ArgumentParser(
        prog="visualguard", description="Audited industrial anomaly detection"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report environment and installed model support")
    synthetic = commands.add_parser(
        "synthetic", help="Generate clearly labeled software test images"
    )
    synthetic.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser(
        "prepare", help="Audit data and create immutable normal-only splits"
    )
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--category", required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument(
        "--groups", type=Path, help="Optional path,group_id CSV for physical-item grouping"
    )
    for name in ("train", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "train":
            sub.add_argument("--config", type=Path, required=True)
        else:
            sub.add_argument("--artifact", type=Path, required=True)
            sub.add_argument("--repetitions", type=int, default=100)
            sub.add_argument("--allow-repeat", action="store_true")
            sub.add_argument(
                "--corruption", choices=["none", "blur", "noise", "brightness"], default="none"
            )
    predict = commands.add_parser("predict")
    predict.add_argument("--artifact", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--category", required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from .utils import environment, write_json

        if args.command == "doctor":
            result = environment()
        elif args.command == "synthetic":
            from .synthetic import generate

            generate(args.output)
            result = {"dataset": "synthetic-smoke", "output": str(args.output)}
        elif args.command == "prepare":
            from .data import audit_dataset, make_splits, save_manifest

            groups = None
            if args.groups:
                with args.groups.open(encoding="utf-8", newline="") as handle:
                    groups = {row["path"]: row["group_id"] for row in csv.DictReader(handle)}
            samples, report = audit_dataset(args.root, args.category, groups=groups)
            write_json(args.manifest.with_suffix(".audit.json"), report)
            if not report["ok"]:
                raise VisualGuardError(f"Dataset audit failed: {report['errors']}")
            digest = save_manifest(
                args.manifest, make_splits(samples, seed=args.seed), seed=args.seed
            )
            result = {
                "manifest_sha256": digest,
                "images": len(samples),
                "near_duplicate_candidates": report["near_duplicate_candidate_count"],
            }
        elif args.command == "train":
            from .models import ModelConfig
            from .pipeline import train

            config = ModelConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
            result = train(args.root, args.manifest, args.output, config)
        elif args.command == "evaluate":
            from .pipeline import evaluate

            result = evaluate(
                args.root,
                args.manifest,
                args.artifact,
                args.output,
                repetitions=args.repetitions,
                allow_repeat=args.allow_repeat,
                corruption=args.corruption,
            )
        else:
            from .artifacts import load_artifact
            from .images import MAX_UPLOAD_BYTES
            from .inference import inspect_bytes
            from .models import configure_torch

            if args.image.stat().st_size > MAX_UPLOAD_BYTES:
                raise VisualGuardError("Input exceeds upload limit.")
            detector, metadata = load_artifact(args.artifact, category=args.category)
            configure_torch(detector.config)
            result, _, _ = inspect_bytes(args.image.read_bytes(), detector, metadata)
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (VisualGuardError, OSError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"VisualGuard: {exc}\n")


if __name__ == "__main__":
    main()
