"""Compare PRO integration to Anomalib on perfect, overlapping and tied scores."""

import json

import numpy as np
import torch
from anomalib.metrics.aupro import _AUPRO

from visualguard.metrics import aupro

torch.set_num_threads(2)
rng = np.random.default_rng(7)
masks = np.zeros((4, 16, 16), dtype=bool)
masks[0, 2:4, 2:5] = True
masks[1, 8:12, 8:13] = True
masks[2, 1:3, 10:13] = True
cases = {
    "perfect": rng.random(masks.shape).astype(np.float32) * 0.4 + masks * 0.5,
    "overlapping": rng.random(masks.shape).astype(np.float32) + masks * 0.2,
    "tied": np.round(rng.random(masks.shape).astype(np.float32), 1),
    "all_tied": np.ones(masks.shape, dtype=np.float32),
}
results = []
for name, scores in cases.items():
    reference = float(_AUPRO(fpr_limit=0.3)(torch.tensor(scores), torch.tensor(masks)).item())
    ours = aupro(masks, scores)
    assert abs(reference - ours) < 1e-5, name
    results.append(
        {
            "case": name,
            "reference_value": reference,
            "visualguard_value": ours,
            "absolute_difference": abs(reference - ours),
        }
    )
print(json.dumps({"reference": "anomalib.metrics.aupro._AUPRO", "cases": results}, indent=2))
