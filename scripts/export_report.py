"""Build public numerical evidence and figures without publishing dataset images."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def export(root=Path("reports")):
    expected = {f"E{index:02d}" for index in range(10)}
    complete = {
        p.parent.name
        for p in root.glob("E*/metrics.json")
        if all((p.parent / f).is_file() for f in ("predictions.csv", "latencies.json"))
    }
    if complete != expected:
        raise RuntimeError(f"Expected the ten complete study runs. Found {sorted(complete)}.")
    rows = []
    for path in sorted(root.glob("E*/metrics.json")):
        result = json.loads(path.read_text())
        if result["dataset"] != "MVTec AD":
            continue
        artifact = root.parent / "artifacts" / path.parent.name
        if artifact.is_dir():
            for source, target in (
                ("metadata.json", "training.json"),
                ("evaluation_started.json", "first_evaluation.json"),
            ):
                shutil.copyfile(artifact / source, path.parent / target)
        pixel = result["pixel"] or {}
        rows.append(
            {
                "run": path.parent.name,
                "category": result["category"],
                "method": result["method"],
                "input_size": result["config"]["input_size"],
                "coreset_ratio": result["config"]["coreset_ratio"] if pixel else None,
                "corruption": result["corruption"],
                "image_auroc": result["image"]["image_auroc"],
                "average_precision": result["image"]["average_precision"],
                "pixel_auroc": pixel.get("pixel_auroc"),
                "aupro": pixel.get("aupro"),
                "false_positives": result["image"]["fp"],
                "missed_defects": result["image"]["fn"],
                "p95_ms": result["timing"]["p95_ms"],
                "bank_mib": result["bank_bytes"] / 1024**2,
            }
        )
    if not rows:
        raise RuntimeError("No measured real dataset results found.")
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [row for row in rows if row["run"] in {"E00", "E01", "E04", "E05", "E06", "E07"}]
    summaries = {}
    for method in ("global_knn", "patchcore"):
        selected = [row for row in primary if row["method"] == method]
        summaries[method] = {
            name: float(np.mean([r[name] for r in selected]))
            for name in ("image_auroc", "average_precision", "p95_ms")
        }
    (root / "macro_average.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    lines = [
        "# Measured benchmark",
        "",
        "Raw per-image predictions, timing samples, configuration and hardware are preserved in each run directory.",
        "",
        "| Run | Category | Method | AUROC | AP | Pixel AUROC | AUPRO@0.30 | FP | FN | p95 ms | Bank MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            row[k]
            for k in (
                "run",
                "category",
                "method",
                "image_auroc",
                "average_precision",
                "pixel_auroc",
                "aupro",
                "false_positives",
                "missed_defects",
                "p95_ms",
                "bank_mib",
            )
        ]
        lines.append(
            "| "
            + " | ".join(
                "N/A" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in values
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "E02/E03 change coreset fraction/resolution; E08/E09 reuse E01 on preregistered corruptions and are explicitly marked repeated robustness evaluations. No configuration was selected using test results.",
            "",
            "## Failure identifiers",
            "",
            "Examples below are the first sorted false alarms and misses, not curated to hide errors. Raw images remain local.",
            "",
        ]
    )
    for run in ("E00", "E01", "E04", "E05", "E06", "E07"):
        path = root / run / "predictions.csv"
        if path.exists():
            with path.open(newline="") as handle:
                predictions = list(csv.DictReader(handle))
            for outcome in ("FP", "FN"):
                examples = [p["path"] for p in predictions if p["outcome"] == outcome][:3]
                lines.append(
                    f"- {run} {outcome}: "
                    + (", ".join(f"`{p}`" for p in examples) or "none at the frozen threshold")
                )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    names = [f"{row['category']}\n{row['method']}" for row in primary]
    colors = ["#087f72" if row["method"] == "patchcore" else "#718c9c" for row in primary]
    axes[0].bar(names, [row["image_auroc"] for row in primary], color=colors)
    axes[0].set(
        ylim=(0, 1), ylabel="Image AUROC", title="Ranking on untouched official test images"
    )
    axes[1].bar(names, [row["p95_ms"] for row in primary], color=colors)
    axes[1].set(ylabel="p95 milliseconds", title="100 warmed CPU inferences, batch 1")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    figure_root = Path("figures") if root == Path("reports") else root / "figures"
    figure_root.mkdir(exist_ok=True)
    figure.savefig(figure_root / "benchmark.png", dpi=160)
    plt.close(figure)
    print(f"Exported {len(rows)} measured runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports"))
    export(parser.parse_args().root)
