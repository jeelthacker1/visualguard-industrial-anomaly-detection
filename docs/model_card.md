# Model card

**Intended use:** non-commercial research and educational inspection of MVTec AD bottle,
metal_nut and cable images under controlled acquisition. Each artifact has one category.
**Excluded uses:** product safety certification, autonomous factory acceptance/rejection,
medical interpretation, human surveillance, or arbitrary-category inspection.

**Architecture:** frozen ImageNet ResNet-18 global 1-NN or Anomalib PatchCore layer2/layer3.
No neural fine-tuning. Normal-only feature bank and calibration; see [methodology](methodology.md).
PatchCore uses a 20,000-candidate cap, a documented departure from full-candidate sampling.

**Evidence:** see [measured results](../reports/RESULTS.md), immutable category splits,
configuration and hardware in every run JSON, and per-image failure identifiers. The
synthetic smoke model is labeled separately and has no industrial performance validity.

**Outputs:** raw distance-based anomaly score; frozen threshold; normal_appearance or
review_required; optional continuous map and binary mask; model ID; UTC times; latency;
acquisition warnings. No score is represented as a probability. Heatmaps show unusual
features and are not causal explanations or calibrated defect boundaries.

**Limits:** reflections, pose, background, illumination, blur, tiny defects and structural
changes can cause errors. A wrong product may look normal to the bank. The quality envelope
checks brightness, contrast and Laplacian variance; it is a heuristic, not a validated OOD
classifier. It can also flag real defects. No industrial camera or live line has been tested.
No demographic fairness claim is relevant to this non-human dataset, and no generalization
claim is made for other products, sites or populations of parts.

**Data/rights:** MVTec AD is CC BY-NC-SA 4.0. Source code MIT does not relicense the dataset,
model weights or dataset-derived artifacts. Raw examples and trained artifacts are kept local
by default. Confirm downstream use and distribution terms separately with the relevant owners.

**Maintenance:** retrain and recalibrate as a new version after camera/process changes using
fresh independent validation and test data. Never adapt thresholds online using production
decisions without a reviewed evaluation protocol. Archive old metadata for rollback.
