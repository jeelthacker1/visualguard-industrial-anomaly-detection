# Experimental protocol

The question is whether local feature matching improves anomaly ranking and localization
relative to a global representation, and what it costs in CPU time and memory. This is
normal-only anomaly detection: defect labels do not train the feature extractor or select
the detection threshold. Both methods use frozen ImageNet ResNet-18 `resnet18.tv_in1k`
weights from timm. Their square RGB resize, ImageNet normalization, and training membership
are identical. The global method retains one 512-dimensional pooled vector per unique
normal training image and scores minimum Euclidean distance. It cannot provide a faithful
pixel anomaly map, so its pixel metrics are absent rather than zero.

The advanced detector is an adapter around Anomalib's PatchcoreModel. Layer2 and layer3
features are locally pooled and concatenated; the library performs k-center coreset selection,
nearest-neighbor patch matching, weighted image scoring and smoothed map generation. We do
not reimplement those steps or claim the simplified nearest-patch formula is the complete
published algorithm. To keep the study usable on an 8 GB CPU laptop, a seeded uniform cap of
20,000 candidate patches precedes the library's 5% or 10% coreset. The actual candidate and
bank counts are stored. This is a declared computational approximation to the full-data
PatchCore setting; comparisons to published leaderboard results would be inappropriate.

## Data governance

MVTec AD's official full archive is downloaded only for local non-commercial research.
Its SHA-256 is verified against the hash published in Anomalib's data module. Only bottle,
metal_nut and cable are extracted. Neither raw images nor masks are committed. Manifests
contain relative file identifiers, labels, dimensions, byte and canonical RGB hashes,
mask hashes, perceptual hashes, physical group identifiers and immutable split assignments.
The validation pass checks decoding, format, dimensions, binary masks, orphan masks,
train/good provenance and the existence of both official test classes.

Exact content duplicated across official train/test or conflicting labels fails the audit.
Within training, exact duplicates share a split and are represented once in fitting.
Optional `--groups path.csv` accepts `path,group_id` physical-item grouping. Without supplied
item identifiers, we cannot prove that two views depict different physical objects. A 64-bit
dHash distance of at most two flags near-duplicate candidates for human review. This is
deliberately not an identity classifier; bottle silhouettes make its false-candidate rate high.
The study retains all official test images and documents that physical-item independence
is unknown. No test image is silently removed to make a score look better.

For each category, sorted normal training groups are shuffled with NumPy seed 42; the first
ceil(20%) are validation, the rest fit the bank. This differs from a naive file split only when
duplicates/groups occur. `validate_splits` independently checks path/content/group separation.
CSV contents are hashed and locked with a sidecar. Files are rehashed before training and
evaluation to detect changes after auditing.

## Calibration and evaluation

The image threshold is the 99th percentile of held-out normal scores with NumPy's `higher`
quantile method. The pixel threshold is independently the 99th percentile of all normal
validation map pixels. A strict `score > threshold` rule handles ties. Small validation sets
often put the image threshold at the largest observed score; this does not prove a 1% future
false-alarm rate. No test label or anomaly mask enters calibration. We bypass Anomalib's
datamodule and automatic postprocessor, avoiding its test-as-validation default entirely.

Before test inference, the artifact stores its calibration, data fingerprints, preprocessing,
configuration, bank dimensions, library versions and hardware. `preregistered_plan.json`
records each configuration hash before the study. An exclusive evaluation marker prevents
silent reruns. Explicit repeats are labeled. The clean test comparison includes both methods
on all three categories plus two bottle ablations. Two fixed corruptions (Gaussian blur radius
2; Gaussian RGB noise standard deviation 10/255 with seed 42) reuse the original bottle
model without refitting. They are robustness probes, not independent datasets or tuning data.

AUROC and average precision use continuous raw image scores. One-class ranking metrics
return null with an explanation. Confusion counts, precision/recall/F1 and false positives
are computed at the already frozen threshold. AUROC uncertainty uses 500 stratified bootstrap
samples of test images; correlated physical items may violate its independence assumption.
Primary summary values are macro-averages of the three categories, not pooled examples.
The initial study uses a single seed; it makes no multi-seed stability claim.

Pixel metrics pool every pixel at the declared model resolution. Masks are resized only
with nearest-neighbor interpolation. Exact pixel AUROC does not subsample. AUPRO uses
8-connected components, equal component weights, tied-score threshold events and exact
trapezoidal integration to false-positive rate 0.30 with linear endpoint interpolation,
normalized by 0.30. Tests cover perfect ranking, tied scores and unequal region sizes.
IoU and Dice are supplementary pooled metrics at the validation-derived pixel threshold.
Resizing can erase tiny defects; these scores are not original-resolution evaluations.

## Runtime protocol

Inference is CPU-only, batch one, with four Torch threads. Ten warmups precede 100 timed
calls. `perf_counter` measures preprocessing, feature extraction and scoring on preloaded
images. Disk decoding, UI, network transport and cold artifact loading are excluded and
explicitly labeled. We retain every timing sample and report median, p95, p99 and sequential
throughput. Sampled process RSS is not peak memory; bank size is exact tensor bytes. Actual
API deadlines include decoding and quality checks; a missed budget changes the decision to
review rather than claiming a hard real-time guarantee.

References: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
[MVTec AD paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html),
[PatchCore paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html),
[Anomalib PatchCore](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html).
