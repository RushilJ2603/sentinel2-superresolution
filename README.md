# Super-resolution of Sentinel-2 imagery with a calibrated confidence map

Course project, Neural Networks and Deep Learning.

Sentinel-2 gives away imagery at 10 m per pixel. Commercial imagery at 1 to 2 m
costs money. This project trains a small CNN that reconstructs 5 m detail from the
free 10 m product across four bands, and treats the output as a measurement rather
than a picture: the band values have to survive, because NDVI and every other index
an analyst computes is a ratio of exactly those values.

Training pairs come from SEN2VENµS, where Sentinel-2 and VENµS imaged the same
ground on the same date. Both sides are real captures, so the network learns the
relationship between two real sensors instead of learning to invert a blur we
applied ourselves.

## Result

On a site held out of training entirely, the final model is ahead of bicubic
interpolation on the two standard reconstruction metrics.

| held-out site (FGMANAUS) | bicubic | ours |
|---|---|---|
| PSNR (dB) | 48.013 | **48.187 ± 0.009** |
| SSIM | 0.9583 | **0.9594** |
| SAM (deg) | **0.837** | 0.855 |
| NDVI MAE | **0.01256** | 0.01410 |

Getting there took three steps, each measured separately. At the midpoint of the
project the same architecture was 3.93 dB *behind* bicubic on this site.

| | held-out PSNR | gap to bicubic |
|---|---|---|
| two European training sites | 44.087 | -3.93 |
| plus paired-gain augmentation | 45.857 | -2.16 |
| plus four more biomes | 47.609 | -0.40 |
| plus confidence-weighted fallback | **48.187** | **+0.17** |

Bicubic keeps the two spectral metrics. We report that rather than selecting the
columns we win.

## What the project actually argues

### A spectral loss term that cannot help, and the measurement that shows it

The original plan added a Spectral Angle Mapper term to the loss to protect band
ratios. SAM is scale-invariant, so it attains its minimum exactly where L1 does and
introduces no new optimum. That part is settled by algebra and we concede it.

What algebra does not settle is whether the term still bends the optimisation path.
We measured that by differentiating the two loss terms separately every 100 steps
and taking the angle between the gradients. Over the second half of training across
three seeds, `cos(grad L1, grad SAM)` averages **0.813**, so the term mostly pushes
where L1 already pushes. No metric moved outside the seed spread, including SAM
itself.

The same probe then earned its keep. A back-projection term, which compares the
output against the *input* rather than the target, measures **0.114** on the same
scale and improves every out-of-domain metric. The diagnostic separated the two
terms correctly before the metrics did, which is what makes it useful rather than
decorative.

### Why the model failed across biomes, and what fixed it

Trained on two European sites, the network learned a brightness prior worth about
+0.003 reflectance on the visible bands and applied it to Amazon rainforest, where
true red sits near 0.004. The arithmetic runs through to NDVI: 0.972 true against
0.951 predicted, a shift of 0.021, matching the measured degradation.

Two fixes followed from that diagnosis and both worked in the predicted direction.
Augmenting with a per-band gain applied identically to the low and high resolution
patch removes absolute brightness as something the model can lean on, and it halved
the bias on the bands it targeted. Widening the training set from two sites to six
covered the ratio shift the augmentation could not.

### The decimation operator, and a number that moved with it

The consistency metric compares a decimated prediction against the input, so it is
defined through a degradation operator D. Decimating by two keeps only frequencies
below π/2 rad/sample, which means D needs an anti-alias filter cut there. The 2×2
box average we started with is −3.01 dB at the cutoff and only −8.3 dB at 0.75π,
leaving **18.1%** of surviving energy to fold back as aliasing.

`src/degrade.py` implements ten filters all cut at π/2, designed on a half-integer
grid so their phase matches `avg_pool2d` exactly (the 2-tap case reproduces it to
6e-8, which is the unit test). Applying each to the real VENµS capture and comparing
against the real Sentinel-2 capture, a sensor-matched Gaussian models the instrument
best on both biomes tested, ahead of the near-ideal windowed-sinc and Butterworth
designs. Modelling the sensor beats sharpening the cutoff.

### Using the confidence map instead of only reporting it

The network predicts a per-pixel Laplace scale alongside the mean. Laplace rather
than Gaussian because training with L1 already *is* maximum likelihood under a
Laplace with fixed scale, so letting the scale vary changes one assumption and
nothing else. The head costs 9,232 parameters against the backbone's 1,371,012.

In-domain it is calibrated: ENCE 0.041, and the stated 90% interval covers 91.7%.

The useful part is what it enables at inference:

```
out = w · SR + (1 − w) · bicubic,     w = exp(−σ / τ)
```

Keep the super-resolved pixel where the model is confident, fall back to
interpolation where it is not. τ is fixed to the model's own mean σ on its training
distribution, so no test data influences it, and the three seeds independently
produced 0.00899, 0.00921 and 0.00890. This is what takes the held-out score past
bicubic. It costs 0.19 dB in-domain and buys 0.71 dB out of domain.

A sweep of τ on the test site, run afterwards as a sensitivity check, puts every
value between 0.001 and 0.02 ahead of bicubic, so the result sits on a plateau.

## Things that did not work

Reporting these because they cost real GPU time and the reasons are informative.

**Global residual learning.** Predicting a delta on top of bicubic, zero-initialised
so the untrained network is bicubic exactly, should in principle floor the model at
interpolation. It does not survive training: the delta drifts toward the training
biomes and is then just as wrong on unseen data. Held-out PSNR 47.307 against 47.609
for the plain model, while in-domain it improved to 43.646 from 43.520.

**A three-seed deep ensemble.** The obvious free alternative to a learned confidence
head. Its disagreement does not rise on the unseen biome at all, and AUROC for
detecting the shift is 0.354, below chance, against 0.76 for the head. All three
seeds learned the same brightness prior, so they are confidently and identically
wrong in the same places, and disagreement can only see what members disagree about.

**β = 1 in β-NLL.** Chosen from the gradient algebra to protect the mean. The
ablation disagreed: β = 0 and β = 0.5 both give better PSNR, so gradient clipping
absorbs the scaling we were guarding against. The reported numbers use β = 1 and are
mildly conservative as a result.

## Data

SEN2VENµS v2.0.0, Zenodo record 14603764, non-commercial licence. Seven sites,
7,801 patch pairs. Each pair is 4×128×128 at 10 m and 4×256×256 at 5 m, stored as
int16 reflectance scaled by 10000.

| site | patches | B4 red | B8 NIR | NDVI | role |
|---|---|---|---|---|---|
| ESTUAMAR | 911 | 0.0178 | 0.2821 | 0.864 | train |
| SUDOUE-4 | 935 | 0.0340 | 0.3207 | 0.801 | train |
| K34-AMAZ | 1384 | 0.0075 | 0.2459 | 0.944 | train |
| MAD-AMBO | 1442 | 0.0644 | 0.2230 | 0.547 | train |
| ANJI | 1500 | 0.0395 | 0.2338 | 0.696 | train |
| NARYN | 1500 | 0.0930 | 0.2533 | 0.464 | train |
| FGMANAUS | 129 | 0.0041 | 0.2790 | 0.974 | held out |

Splits are by site. Patches inside one site are neighbouring crops of the same
fields, so a random patch split would put near-duplicates on both sides. Ten percent
of the training sites is held back for early stopping, and FGMANAUS is scored once
per run and never used for any decision.

The held-out site stays outside the training range after the additions: its red
reflectance is 0.0041 against a training minimum of 0.0075, so the score remains an
extrapolation test.

## Protocol

Every learned row uses the same 12,000 steps, Adam at 1e-4 with cosine decay to
1e-6, batch 16, 64 px low-resolution crops, bfloat16 autocast, and the same three
seeds. Only the component under test changes. Rows that carry a claim report the
mean and population standard deviation over three seeds, and a difference smaller
than the seed spread is not treated as a difference.

The comparison includes three other architectures trained under that identical
protocol, because an under-tuned baseline is not a baseline.

## Layout

```
src/
  data.py            patch loading, site-level splits, augmentation
  model.py           EDSR-style backbone, optional confidence head
  baselines.py       SRCNN, RCAN-lite, EDSR-large for comparison
  losses.py          L1, SAM, back-projection, Laplace NLL
  degrade.py         ten pi/2 anti-alias filters for the decimation operator
  uncertainty.py     Laplace head loss, AUSE, AURG, ENCE, Cv, PICP, AUROC
  metrics.py         PSNR, SSIM, SAM, NDVI MAE, consistency
  train.py           training loop and the gradient-alignment probe
  run_night.py       the experiment queue, resumable
  sigma_blend.py     confidence-weighted fallback to bicubic
  diagnose_ood.py    why the held-out gap exists
  figures.py         mid-sem figures
  figures2.py        every figure in the final deck
  sanity.py          pre-flight checks, run this before training
  fetch_data.py      download sites from Zenodo
  prepare_data.py    archives to cached arrays
  finalize.py        rebuild figures, report and slides
results/
  RESULTS_FINAL.md   full tables
  figures/           all figures
  *.json             raw measurements
```

## Reproducing

```bash
python src/fetch_data.py ESTUAMAR SUDOUE-4 K34-AMAZ MAD-AMBO ANJI NARYN FGMANAUS
python src/prepare_data.py --limit 1500
python src/sanity.py
python src/run_night.py --runs R1 R2 R3 R4 R1a R1m R1ma R4m --seeds 0 1 2
python src/sigma_blend.py
python src/finalize.py
```

`sanity.py` is worth running first. It checks shapes and ranges end to end, verifies
the filter bank preserves DC gain and phase, confirms the augmentation leaves the
per-band ratio between the two resolutions unchanged, and overfits the model on
eight patches. A network that cannot memorise eight patches will not learn seven
thousand.

Training was done on a single RTX 4060 laptop GPU, roughly 20 minutes per run.
Inference runs at 8.3 Mpix/s, which is about 207 km² per second at 5 m ground
sample distance.

## References

- Michel et al., SEN2VENµS v2.0.0, Zenodo 14603764
- Lim et al., Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 2017
- Dong et al., Learning a Deep Convolutional Network for Image Super-Resolution, ECCV 2014
- Zhang et al., Image Super-Resolution Using Very Deep Residual Channel Attention Networks, ECCV 2018
- Kim et al., Accurate Image Super-Resolution Using Very Deep Convolutional Networks, CVPR 2016
- Irani and Peleg, Improving Resolution by Image Registration, CVGIP 1991
- Wald, Ranchin and Mangolini, Fusion of satellite images of different spatial resolutions, PE&RS 1997
- Aiazzi et al., MTF-tailored multiscale fusion of high-resolution MS and Pan imagery, PE&RS 2006
- Kendall and Gal, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, NeurIPS 2017
- Lakshminarayanan et al., Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, NeurIPS 2017
- Ilg et al., Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow, ECCV 2018
- Levi et al., Evaluating and Calibrating Uncertainty Prediction in Regression Tasks, 2020
- Seitzer et al., On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks, ICLR 2022
- Ovadia et al., Can You Trust Your Model's Uncertainty?, NeurIPS 2019
- Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020
