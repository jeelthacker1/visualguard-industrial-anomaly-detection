from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .errors import VisualGuardError


def _binary(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or not labels.size or not np.isfinite(scores).all():
        raise VisualGuardError("Labels/scores must be nonempty, aligned and finite.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise VisualGuardError("Only binary labels are supported.")
    return labels.astype(bool), scores


def image_metrics(
    labels, scores, threshold: float, *, bootstrap: int = 500, seed: int = 42
) -> dict:
    labels, scores = _binary(labels, scores)
    if labels.ndim != 1 or not np.isfinite(threshold):
        raise VisualGuardError("Expected one score per image and a finite fixed threshold.")
    predicted = scores > threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    both = len(np.unique(labels)) == 2
    output = {
        "count": len(labels),
        "anomaly_prevalence": float(labels.mean()),
        "image_auroc": float(roc_auc_score(labels, scores)) if both else None,
        "average_precision": float(average_precision_score(labels, scores)) if both else None,
        "undefined_reason": None if both else "Only one test class; ranking metrics are undefined.",
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "accuracy": (tp + tn) / len(labels),
        "auroc_ci95": None,
        "ci_method": "stratified image bootstrap; correlated physical items may invalidate independence",
    }
    if both and bootstrap > 0:
        rng = np.random.default_rng(seed)
        negatives, positives = np.flatnonzero(~labels), np.flatnonzero(labels)
        values = []
        for _ in range(bootstrap):
            indices = np.concatenate(
                [rng.choice(negatives, len(negatives)), rng.choice(positives, len(positives))]
            )
            values.append(roc_auc_score(labels[indices], scores[indices]))
        output["auroc_ci95"] = np.quantile(values, [0.025, 0.975]).tolist()
    return output


def aupro(masks: np.ndarray, scores: np.ndarray, *, max_fpr: float = 0.3) -> float | None:
    """Exact tied-score PRO integration; equal weight per 8-connected defect region.

    Normalized by max_fpr, including a linearly interpolated boundary point.
    No threshold binning, pixel subsampling, or substitution of IoU for PRO.
    """
    masks, scores = _binary(masks, scores)
    if masks.ndim != 3 or not 0 < max_fpr <= 1:
        raise VisualGuardError("AUPRO expects N×H×W maps and 0 < max_fpr <= 1.")
    negative_count = int((~masks).sum())
    weights = np.zeros(masks.shape, dtype=np.float64)
    regions = 0
    for index, mask in enumerate(masks):
        components, count = connected_components(mask, structure=np.ones((3, 3), dtype=int))
        sizes = np.bincount(components.ravel())
        inverse_sizes = np.zeros(len(sizes), dtype=np.float64)
        inverse_sizes[1:] = 1.0 / sizes[1:]
        weights[index] = inverse_sizes[components]
        regions += count
    if not regions or not negative_count:
        return None
    order = np.argsort(-scores.ravel(), kind="stable")
    ranked_scores = scores.ravel()[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(order) - 1]
    pro = np.r_[0.0, np.cumsum(weights.ravel()[order])[ends] / regions]
    fpr = np.r_[0.0, np.cumsum((~masks).ravel()[order])[ends] / negative_count]
    inside = fpr <= max_fpr
    x, y = fpr[inside], pro[inside]
    if x[-1] < max_fpr:
        high = int(np.flatnonzero(fpr > max_fpr)[0])
        boundary = np.interp(max_fpr, [fpr[high - 1], fpr[high]], [pro[high - 1], pro[high]])
        x, y = np.r_[x, max_fpr], np.r_[y, boundary]
    return float(np.clip(np.trapezoid(y, x) / max_fpr, 0, 1))


def pixel_metrics(masks, maps, threshold: float) -> dict:
    masks, maps = _binary(masks, maps)
    if masks.ndim != 3 or not np.isfinite(threshold):
        raise VisualGuardError("Expected N×H×W maps and a finite pixel threshold.")
    if masks.size > 25_000_000:
        raise VisualGuardError(
            "Exact pixel evaluation limited to 25 million pixels; use smaller fixed model resolution."
        )
    predictions = maps > threshold
    intersection = int((masks & predictions).sum())
    union = int((masks | predictions).sum())
    both_sum = int(masks.sum() + predictions.sum())
    return {
        "pixel_auroc": float(roc_auc_score(masks.ravel(), maps.ravel()))
        if len(np.unique(masks)) == 2
        else None,
        "aupro": aupro(masks, maps),
        "aupro_max_fpr": 0.3,
        "aupro_connectivity": 8,
        "iou": intersection / union if union else None,
        "dice": 2 * intersection / both_sum if both_sum else None,
        "pixel_threshold": threshold,
        "pixel_count": masks.size,
        "aggregation": "all pixels pooled at model resolution; ground truth nearest-neighbor resize; no subsampling",
    }


def latency_summary(milliseconds) -> dict:
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise VisualGuardError("Latency measurements must be finite and positive.")
    return {
        "repetitions": len(values),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "mean_ms": float(values.mean()),
        "sequential_throughput_fps": 1000 / float(values.mean()),
    }
