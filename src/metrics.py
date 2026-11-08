# The metrics that go in the results tables. Everything here expects
# reflectance already scaled into [0, 1] -- NOT the 8-bit version you'd make
# for display. Measuring on a contrast-stretched image makes every number
# meaningless and it is an easy mistake to make.
#
#   M1 PSNR (dB)     higher   plain reconstruction error
#   M2 SSIM          higher   edges and texture
#   M3 SAM (deg)     lower    how far the 4-band direction drifted
#   M4 NDVI MAE      lower    does the vegetation index still come out right
#   M5 |D(SR) - LR|  lower    does it still agree with the real measurement
#
# M5 needs a warning. Bicubic invents nothing, so shrinking it back down
# reproduces the input almost exactly and it scores beautifully while being
# obviously blurry. M5 only means something next to M1 and M2.
#
# M5 ALSO DEPENDS ON D, WHICH NOBODY HAD JUSTIFIED
# D used to be hard-wired to a 2x2 box average. Decimating by 2 discards
# everything above pi/2 rad/sample, so D needs an anti-alias filter cut
# there, and the box is a bad one: 18% of the energy surviving it sits above
# pi/2 and folds back as aliasing. Since M5 is defined THROUGH D, the choice
# of filter was silently shaping a reported column. `consistency_error` now
# takes a filter name so the whole bank can be swept, and the default stays
# box2 so the midsem numbers remain exactly reproducible.

import math

import torch
import torch.nn.functional as F

from degrade import FILTERS, degrade_with
from losses import DEFAULT_FILTER, ndvi
from uncertainty import laplace_sigma, uncertainty_report


def psnr(pred, target, data_range=1.0):
    # per band, then averaged -- not one PSNR over the stacked tensor
    mse = ((pred - target) ** 2).mean(dim=(2, 3)).clamp_min(1e-12)
    return (10.0 * torch.log10(data_range ** 2 / mse)).mean().item()


def ssim(pred, target, data_range=1.0, ws=11, sigma=1.5):
    c = pred.shape[1]
    g = torch.arange(ws, device=pred.device, dtype=pred.dtype) - (ws - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(c, 1, ws, ws).contiguous()
    pad = ws // 2
    mu1 = F.conv2d(pred, w, padding=pad, groups=c)
    mu2 = F.conv2d(target, w, padding=pad, groups=c)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, w, padding=pad, groups=c) - mu1s
    s2 = F.conv2d(target * target, w, padding=pad, groups=c) - mu2s
    s12 = F.conv2d(pred * target, w, padding=pad, groups=c) - mu12
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.mean().item()


def sam_degrees(pred, target, eps=1e-8):
    # same maths as the loss, reported in degrees instead of radians.
    # floor is ~0.026 deg because of the clamp, so don't expect a clean 0.
    dot = (pred * target).sum(dim=1)
    pn = pred.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    tn = target.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    cos = (dot / (pn * tn)).clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.acos(cos).mean() * 180.0 / math.pi).item()


def ndvi_mae(pred, target):
    return (ndvi(pred) - ndvi(target)).abs().mean().item()


def consistency_error(sr, lr, name=DEFAULT_FILTER):
    """M5 under a named pi/2 anti-alias filter."""
    return (degrade_with(sr, name) - lr).abs().mean().item()


def band_bias(pred, target):
    """Mean SIGNED error per band.

    Signed, not absolute, on purpose. This is the statistic that diagnosed
    the midsem failure: on FGMANAUS the model was +0.0025/+0.0050/+0.0030
    on the visible bands and ~0 on NIR, i.e. a learned brightness prior
    rather than random error. An absolute error would have hidden the sign
    and with it the entire explanation.
    """
    return (pred - target).mean(dim=(0, 2, 3)).tolist()


def bicubic_predict(lr, scale=2):
    return F.interpolate(lr, scale_factor=scale, mode="bicubic",
                         align_corners=False)


@torch.no_grad()
def evaluate(predict, loader, device, filters=(DEFAULT_FILTER,),
             with_uncertainty=False, unc_cap=4_000_000,
             unc_max_patches=150):
    """predict(lr) -> sr, or (mu, log_b) when with_uncertainty.

    Returns M1-M5 weighted by batch size, the per-band signed bias, M5 under
    every filter in `filters`, and -- if the model has an uncertainty head --
    the full calibration suite plus a per-patch mean sigma for OOD scoring.

    `unc_max_patches` bounds how many patches are retained for the
    calibration suite. The suite needs the raw pixels (it sorts them), and
    one patch is 4x256x256 floats across three tensors, so the multi-biome
    validation split would otherwise hold about 2 GB in CPU memory. 150
    patches is ~39 M pixels, far more than the metrics need to be stable.
    M1-M5 are still computed over the FULL loader; only the uncertainty
    suite is subsampled, and the count it used is reported as unc_n_patches.

    The per-patch mean sigma used for the OOD AUROC is kept for every patch
    regardless, since it is one float per patch.
    """
    acc = {k: 0.0 for k in ("psnr", "ssim", "sam", "ndvi_mae", "consist")}
    acc.update({"consist_" + f: 0.0 for f in filters})
    n = 0
    bias = None
    mus, lbs, hrs, patch_sigma = [], [], [], []

    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        out = predict(lr)
        if with_uncertainty:
            sr, log_b = out
            sr = sr.clamp(0, 1)
            # per-patch mean sigma: the scalar score the OOD AUROC uses.
            # One float per patch, so always kept in full.
            s = laplace_sigma(log_b)
            patch_sigma += s.mean(dim=(1, 2, 3)).cpu().tolist()
            if sum(m.shape[0] for m in mus) < unc_max_patches:
                mus.append(sr.cpu())
                lbs.append(log_b.cpu())
                hrs.append(hr.cpu())
        else:
            sr = out.clamp(0, 1)

        b = lr.shape[0]
        acc["psnr"] += psnr(sr, hr) * b
        acc["ssim"] += ssim(sr, hr) * b
        acc["sam"] += sam_degrees(sr, hr) * b
        acc["ndvi_mae"] += ndvi_mae(sr, hr) * b
        acc["consist"] += consistency_error(sr, lr) * b
        for f in filters:
            acc["consist_" + f] += consistency_error(sr, lr, f) * b
        bb = torch.tensor(band_bias(sr, hr)) * b
        bias = bb if bias is None else bias + bb
        n += b

    out = {k: v / max(n, 1) for k, v in acc.items()}
    out["band_bias"] = (bias / max(n, 1)).tolist()
    out["n_patches"] = n

    if with_uncertainty and mus:
        mu_c, lb_c, hr_c = torch.cat(mus), torch.cat(lbs), torch.cat(hrs)
        rep = uncertainty_report(mu_c, lb_c, hr_c, cap=unc_cap)
        out["unc_calib_bins"] = rep.pop("calib_bins", None)
        out.update({"unc_" + k: v for k, v in rep.items()})
        out["unc_n_patches"] = int(mu_c.shape[0])
        out["patch_sigma"] = patch_sigma
    return out


@torch.no_grad()
def filter_sweep(predict, loader, device, names=None):
    """M5 for every filter in the bank, on one model and one regime.

    The point of the sweep is not to pick a winner -- M5 is a metric, not a
    score to optimise -- but to show how much a number already on the
    results slide moves when an unjustified implementation detail changes.
    """
    names = list(names or FILTERS)
    tot = {f: 0.0 for f in names}
    n = 0
    for lr, hr in loader:
        lr = lr.to(device)
        out = predict(lr)
        sr = (out[0] if isinstance(out, (tuple, list)) else out).clamp(0, 1)
        b = lr.shape[0]
        for f in names:
            tot[f] += consistency_error(sr, lr, f) * b
        n += b
    return {f: v / max(n, 1) for f, v in tot.items()}
