# Measured benchmark

Raw per-image predictions, timing samples, configuration and hardware are preserved in each run directory.

| Run | Category | Method | AUROC | AP | Pixel AUROC | AUPRO@0.30 | FP | FN | p95 ms | Bank MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E00 | bottle | global_knn | 0.9937 | 0.9979 | N/A | N/A | 1 | 4 | 95.9737 | 0.3262 |
| E01 | bottle | patchcore | 1.0000 | 1.0000 | 0.9820 | 0.9392 | 1 | 0 | 84.0769 | 1.4648 |
| E02 | bottle | patchcore | 1.0000 | 1.0000 | 0.9822 | 0.9401 | 1 | 0 | 99.7359 | 2.9297 |
| E03 | bottle | patchcore | 1.0000 | 1.0000 | 0.9812 | 0.9291 | 1 | 0 | 72.4325 | 1.4648 |
| E04 | metal_nut | global_knn | 0.8250 | 0.9559 | N/A | N/A | 1 | 39 | 184.0535 | 0.3438 |
| E05 | metal_nut | patchcore | 0.9902 | 0.9978 | 0.9815 | 0.9353 | 0 | 7 | 114.5901 | 1.4648 |
| E06 | cable | global_knn | 0.8739 | 0.9204 | N/A | N/A | 4 | 44 | 81.9351 | 0.3496 |
| E07 | cable | patchcore | 0.9822 | 0.9896 | 0.9795 | 0.9169 | 2 | 14 | 190.7754 | 1.4648 |
| E08 | bottle | patchcore | 1.0000 | 1.0000 | 0.9820 | 0.9398 | 1 | 0 | 95.3394 | 1.4648 |
| E09 | bottle | patchcore | 1.0000 | 1.0000 | 0.9810 | 0.9340 | 1 | 0 | 116.2576 | 1.4648 |

E02/E03 change coreset fraction/resolution; E08/E09 reuse E01 on preregistered corruptions and are explicitly marked repeated robustness evaluations. No configuration was selected using test results.

## Failure identifiers

Examples below are the first sorted false alarms and misses, not curated to hide errors. Raw images remain local.

- E00 FP: `bottle/test/good/006.png`
- E00 FN: `bottle/test/broken_small/000.png`, `bottle/test/broken_small/007.png`, `bottle/test/contamination/004.png`
- E01 FP: `bottle/test/good/019.png`
- E01 FN: none at the frozen threshold
- E04 FP: `metal_nut/test/good/016.png`
- E04 FN: `metal_nut/test/bent/000.png`, `metal_nut/test/bent/001.png`, `metal_nut/test/bent/002.png`
- E05 FP: none at the frozen threshold
- E05 FN: `metal_nut/test/color/004.png`, `metal_nut/test/color/018.png`, `metal_nut/test/scratch/000.png`
- E06 FP: `cable/test/good/007.png`, `cable/test/good/040.png`, `cable/test/good/041.png`
- E06 FN: `cable/test/bent_wire/000.png`, `cable/test/bent_wire/002.png`, `cable/test/bent_wire/008.png`
- E07 FP: `cable/test/good/041.png`, `cable/test/good/057.png`
- E07 FN: `cable/test/cable_swap/000.png`, `cable/test/cable_swap/001.png`, `cable/test/cable_swap/003.png`
