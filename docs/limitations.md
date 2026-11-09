# Limits and unverified claims

- Benchmark images are not a live factory. There is no physical camera, conveyor, PLC,
  robot, line-speed acceptance test, or safety certification in this study.
- No algorithm can guarantee every defect is caught. False negatives at the frozen
  normal-only threshold are explicitly reported even when AUROC is high.
- Wrong categories and novel objects are not reliably rejected. Quality checks are
  heuristic and cannot replace domain validation or supervised acquisition control.
- Near-duplicate dHash is a review aid. For bottle silhouettes it has many false candidates.
  Exact duplicates are checked, but physical-item independence is unknown in supplied data.
- The 20,000-patch candidate cap, small ResNet-18 backbone and resized square maps trade
  representation detail for CPU feasibility. Small/elongated defects may be lost on resize.
- AUPRO and pixel AUROC are measured at model resolution, not original image dimensions.
  Regions are 8-connected. These details must match before comparing with another study.
- The first study uses one split/coreset seed. Image-bootstrap intervals do not measure
  seed, site, acquisition or physical-item variability. No multi-seed standard deviation is claimed.
- Lighting/noise/blur tests are synthetic perturbations of the same official test set.
  They do not constitute an independent distribution-shift benchmark such as MVTec AD 2.
- Timing is hardware- and load-dependent. Median/p95 from 100 warmed calls is not a hard
  deadline or throughput guarantee. Cold start, network, transport and UI have separate costs.
- API rate limits and replay cache are process-local and ephemeral. One-worker operation
  and an authenticated reverse proxy are required for the documented deployment assumptions.
- Safetensors and hashes reduce attack surface but do not authenticate a malicious artifact
  owner. Signed artifacts, SBOM provenance and managed key storage are future deployment work.
- Windows Python 3.13 was tested locally. CI results establish any additional platform evidence.
  The optional Dockerfile must be built/tested before claiming a container deployment.
- Raw dataset images and trained artifacts are not publicly redistributed. A GitHub clone
  requires data access and local training to run genuine category inference.
- No admissions outcome, research novelty, commercial readiness or industrial safety is promised.
