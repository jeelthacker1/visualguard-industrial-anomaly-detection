# VisualGuard technical report

## Abstract

VisualGuard compares frozen ImageNet ResNet-18 global nearest-neighbor features with
Anomalib PatchCore on bottle, metal_nut and cable from MVTec AD. The three-category
macro image AUROC was 0.8975 for global features and 0.9908 for PatchCore. At thresholds
fixed using held-out normal images, PatchCore missed 21 of 248 defects across the three
categories and generated three false alarms among 100 normal test images.
This difference between ranking quality and operational recall motivates explicit
failure reporting, acquisition checks and manual review.

## Data and experimental design

The verified official archive had SHA-256
`cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`.
The selected subset contains 1,001 images. Exact duplicate checks found no duplicate
groups. Near-duplicate dHash flags produced 34,970 bottle pairs, 389 metal-nut pairs
and two cable pairs. Similar bottle silhouettes make these flags an unreliable identity
test. Physical-item independence remains unresolved.

| Category | Normal fit | Normal calibration | Test normal | Test defect |
|---|---:|---:|---:|---:|
| bottle | 167 | 42 | 20 | 63 |
| metal_nut | 176 | 44 | 22 | 93 |
| cable | 179 | 45 | 58 | 92 |

Training groups were shuffled with seed 42. The official test membership was preserved.
The experiment plan was recorded before evaluation. Image and pixel thresholds use
the 99th percentile of their respective normal calibration scores with higher
interpolation. The decision rule is strictly greater than the frozen threshold.
Small calibration sets place the image threshold near the largest normal score.
Future false-alarm rates depend on the deployment distribution.

## Methods and reproducibility

Both methods use the same frozen timm ResNet-18 ImageNet weights, RGB square resize and
ImageNet normalization. The global baseline retains one 512-dimensional embedding per
normal image and uses minimum Euclidean distance. PatchCore uses Anomalib's layer2/layer3
features, coreset selection, weighted image scoring and anomaly maps.

The CPU implementation samples at most 20,000 candidate patches before coreset selection.
The main 5% coreset contains 1,000 vectors of dimension 384, occupying 1.465 MiB.
This computational approximation and the ResNet-18 backbone must be considered when
comparing against published PatchCore results with other backbones and candidate counts.
All eight fitted configurations and two preregistered corruption evaluations are reported.
The experiment identifiers and complete numerical records are in [the results table](../reports/RESULTS.md).

The measured environment used Windows 11, Python 3.13.3, eight logical Intel CPUs,
8.30 GB RAM, Torch 2.15.0 and Anomalib 2.7.0. Inference used four Torch threads.
The delivered runtime follows the available Python 3.13 environment. Package versions
are locked. CLI training, calibration, artifact loading and evaluation run independently
of the dashboard. Safetensors artifacts include the backbone and reference bank for
offline inference.

## Results

| Category | Global AUROC | PatchCore AUROC | PatchCore pixel AUROC | AUPRO@0.30 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| bottle | 0.9937 | 1.0000 | 0.9820 | 0.9392 | 1 | 0 |
| metal_nut | 0.8250 | 0.9902 | 0.9815 | 0.9353 | 0 | 7 |
| cable | 0.8739 | 0.9822 | 0.9795 | 0.9169 | 2 | 14 |

PatchCore's image bootstrap 95% AUROC intervals were [1.0000, 1.0000] for bottle,
[0.9753, 1.0000] for metal_nut and [0.9648, 0.9954] for cable.
These intervals condition on this test sample and assume independent images. They do
not quantify training seed or factory variation. The bottle interval is degenerate
because every resampled positive/negative pair remained correctly ranked.

PatchCore's defect recall was 100% for bottle, 92.47% for metal_nut and 84.78% for cable.
The aggregate descriptive recall was 91.53%. Category macro averaging remains the
primary ranking summary. Normal test sample sizes are small, especially bottle and
metal_nut, which limits conclusions about rare false alarms.

### Ablations and corruptions

Increasing the bottle coreset from 5% to 10% doubled bank storage to 2.930 MiB.
Image AUROC remained 1.0 and AUPRO changed from 0.9392 to 0.9401.
Measured p95 inference rose from 84.08 ms to 99.74 ms. At input size 224,
AUROC remained 1.0, AUPRO was 0.9291 and p95 was 72.43 ms.
These measured differences characterize the fixed runs and do not establish an optimum.

Gaussian blur radius 2 and additive RGB noise with standard deviation 10/255 preserved
bottle AUROC at 1.0 with the same one false alarm and zero misses.
AUPRO was 0.9398 under blur and 0.9340 under noise. These probes reuse the same test
images and cover only the stated corruption settings.

### Failure analysis

Global features missed 39 metal-nut defects, including 19 bent, 12 color and eight scratch
examples. PatchCore reduced this to two color and five scratch misses.
On cable, PatchCore missed ten cable_swap, one cut_inner_insulation and three
poke_insulation examples. Cable swaps can preserve local appearance while changing the
global arrangement, which is a plausible explanation for their concentration among misses.
That explanation is an interpretation of the error distribution.

The first sorted cable-swap miss is `cable/test/cable_swap/000.png`.
The bottle false alarm is `bottle/test/good/019.png`.
All individual predictions and failure identifiers are published for audit.
Frozen pixel-threshold IoU was 0.2955 for bottle, 0.6818 for metal_nut and 0.3902 for cable.
High ranking metrics therefore coexist with imperfect binary region boundaries.

## Timing and operating behavior

The main PatchCore p95 CPU times were 84.08 ms, 114.59 ms and 190.78 ms for bottle,
metal_nut and cable. Each run contains 100 timed calls after ten warmups.
Timing includes preprocessing and model inference on preloaded images. File decoding,
transport and UI costs are outside this measurement. The desktop was shared with other
processes, so background load and thermal variation can affect timing comparisons.
Recorded RSS is sampled process memory. Peak memory was unmeasured.

A separate loopback HTTP probe processed 30 timed requests with the trained bottle model
after three warmups. Median round trip was 125.92 ms and p95 was 193.33 ms.
There were zero inspection deadline misses against the 1,000 ms budget.
The probe verified replay handling and HTTP rejection of stale frames, invalid images and
missing credentials. Its repeated preloaded image isolates software transport behavior.
Camera capture, plant network and actuator timing require independent measurement.
The default API rate cap is 60 inspection requests per minute, which limits sustained
operation despite faster short bursts.

## Engineering controls and remaining validation

Dataset and mask hashes detect changes after the audit. Group checks prevent declared
physical items crossing fitting, calibration and evaluation. Model hashes detect accidental
artifact changes. Trusted artifact provisioning remains an operator responsibility.
Uploads have byte/pixel limits and strict content decoding. The API requires authentication,
fresh timezone-aware capture timestamps and bounded request concurrency.
Quality changes and deadline overruns route results for review.

The software tests exercise model serialization, upload rejection, duplicate leakage,
timestamp replay, rate limits and concurrent requests. Reference AUPRO comparisons cover
perfect, overlapping, tied and all-tied scores, with differences below 0.00001 against
the installed Anomalib metric. The machine-readable verification record documents
actual checks and unverified deployment conditions.

The next research stage requires independent camera data, physical-item identifiers,
multiple seeds and category-specific acceptance criteria based on the costs of missed
defects and false alarms. Factory integration and safety acceptance remain open.

## Attribution

Implementation and writing were developed with AI assistance. The project contribution
is the audited workflow, comparison, operational controls and evidence reporting.
The detector families and datasets are credited to their authors.

- Bergmann et al., MVTec AD, CVPR 2019. [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)
- Roth et al., Towards Total Recall in Industrial Anomaly Detection, CVPR 2022. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)
- Anomalib implementation and metric reference. [Source](https://github.com/open-edge-platform/anomalib)
- timm pretrained backbone. [Source](https://github.com/huggingface/pytorch-image-models)
