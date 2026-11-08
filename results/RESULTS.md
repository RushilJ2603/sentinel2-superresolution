# Mid-sem results

Train sites **ESTUAMAR, SUDOUE-4**, 1661 patches. Held-out test site **FGMANAUS**, 129 patches. 12000 steps, batch 16, crop 64, 16 residual blocks, 64 channels, lam = 0.1. Metrics on reflectance in [0,1], averaged per band. R1 and R2 are mean ± population std over seeds [0, 1, 2].

## 1. In-domain (validation split of the training sites)

This split also selects the early-stopping checkpoint, so it is not a clean held-out set. It answers *does the model work at all*, not *does it generalise*.

| Run | Method | PSNR (dB) ^ | SSIM ^ | SAM (deg) v | NDVI MAE v | M5 consist. v |
|---|---|---|---|---|---|---|
| R0 | Bicubic x2 | 42.159 | 0.9378 | 1.582 | 0.02810 | 0.00078 |
| R1 | CNN, L1 | 43.774 ± 0.038 | 0.9506 ± 0.0003 | 1.295 ± 0.008 | 0.02415 ± 0.00013 | 0.00464 ± 0.00002 |
| R2 | CNN, L1 + lam*SAM | 43.731 ± 0.050 | 0.9500 ± 0.0005 | 1.292 ± 0.009 | 0.02413 ± 0.00018 | 0.00460 ± 0.00008 |

## 2. Out-of-domain (held-out site FGMANAUS)

| Run | Method | PSNR (dB) ^ | SSIM ^ | SAM (deg) v | NDVI MAE v | M5 consist. v |
|---|---|---|---|---|---|---|
| R0 | Bicubic x2 | 48.013 | 0.9583 | 0.837 | 0.01256 | 0.00074 |
| R1 | CNN, L1 | 44.087 ± 0.133 | 0.9145 ± 0.0025 | 1.564 ± 0.042 | 0.02562 ± 0.00048 | 0.00521 ± 0.00012 |
| R2 | CNN, L1 + lam*SAM | 43.956 ± 0.098 | 0.9118 ± 0.0023 | 1.568 ± 0.026 | 0.02565 ± 0.00083 | 0.00529 ± 0.00009 |

## Findings

**The network works.** In-domain, R1 beats bicubic by +1.61 dB PSNR and +0.0127 SSIM, and improves both spectral metrics (SAM -0.287 deg, NDVI MAE -0.00396). The pipeline is sound.

**It does not transfer across biomes.** On FGMANAUS it loses to bicubic by 3.93 dB. The training sites are European coast and farmland; the test site is Amazon rainforest, whose blue and red bands are 4-8x darker than anything in training. The model brightens the visible bands by about +0.003 reflectance, which against a true red near 0.004 is a ~75% relative error, and it propagates straight into NDVI. Bicubic has no learned prior, so it has no bias to carry across.

**The M5 trap.** Bicubic scores 0.00074 on consistency error against 0.00521 for R1. Bicubic invents no detail, so shrinking it back down reproduces the input almost exactly. M5 is only meaningful read next to PSNR and SSIM.

**R2 vs R1, the overlap question, in-domain.**

| Metric | R1 | R2 | change | vs seed spread |
|---|---|---|---|---|
| PSNR (dB) | 43.774 | 43.731 | -0.043 | within |
| SSIM | 0.9506 | 0.9500 | -0.0006 | **exceeds** |
| SAM (deg) | 1.295 | 1.292 | -0.003 | within |
| NDVI MAE | 0.02415 | 0.02413 | -0.00002 | within |
| M5 consist. | 0.00464 | 0.00460 | -0.00004 | within |

**R2 vs R1, out-of-domain.**

| Metric | R1 | R2 | change | vs seed spread |
|---|---|---|---|---|
| PSNR (dB) | 44.087 | 43.956 | -0.131 | within |
| SSIM | 0.9145 | 0.9118 | -0.0027 | **exceeds** |
| SAM (deg) | 1.564 | 1.568 | +0.004 | within |
| NDVI MAE | 0.02562 | 0.02565 | +0.00003 | within |
| M5 consist. | 0.00521 | 0.00529 | +0.00007 | within |

Two things the tables do not say on their own.

First, where a change is flagged as exceeding the seed spread, check the magnitude before believing it means anything. The SSIM spread across seeds is around 0.0005, so a 0.0006 change clears it while being practically nil. Statistical detectability is not practical significance, and with three seeds the spread estimate is itself rough.

Second, and more pointed: training on SAM did not improve the SAM metric. In-domain it moves 1.295 -> 1.292, out-of-domain 1.564 -> 1.568. A loss term that cannot reliably improve the very quantity it optimises is the clearest possible demonstration that it had nothing left to contribute once L1 was already in the objective.

SAM is scale-invariant, so it attains its minimum exactly where L1 does and cannot introduce a new optimum. That is settled analytically, not by this table. What the table shows is whether the reweighting changes the solution the optimiser lands on. Figure A1 measures how far the gradient actually turns.

**A1, gradient alignment.** Mean cos(grad L1, grad SAM) over the second half of training, across 3 seeds: **0.813** (180 probes). 1.0 would mean SAM adds nothing beyond L1's direction; 0.0 would mean it carries independent information; below zero would mean the two terms conflict.
