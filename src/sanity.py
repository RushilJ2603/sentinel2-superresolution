"""Pre-flight checks. Run this before any real training.

1. shapes and value ranges through the whole pipeline
2. the degradation operator D lines up with the input
3. metrics behave sensibly on trivial inputs
4. the gradient probe returns 1.0 when a term is compared against itself
5. OVERFIT-8: the model must drive training loss near zero on 8 patches.
   A model that cannot memorise 8 examples will never learn 1800, and this
   catches most pipeline bugs in under a minute.
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import AugConfig, SRPatches, available_sites
from degrade import FILTERS, degrade_with, kernel_1d
from losses import LossSpec, degrade
from uncertainty import ause, laplace_nll
from metrics import (bicubic_predict, consistency_error, ndvi_mae, psnr,
                     sam_degrees, ssim)
from model import SRNet, count_params
from train import grad_cosine, set_seed

fails = []


def check(name, cond, detail=""):
    print(("  [ok] " if cond else "  [FAIL] ") + name +
          ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sites = available_sites()
    print("device:", dev, "| cached sites:", sites)
    if not sites:
        sys.exit("no cached sites; run prepare_data.py first")
    site = sites[0]

    print("\n[1] data pipeline")
    ds = SRPatches([site], crop_lr=64, train=True)
    lo, hi = ds[0]
    check("train LR is (4,64,64)", tuple(lo.shape) == (4, 64, 64))
    check("train HR is (4,128,128)", tuple(hi.shape) == (4, 128, 128))
    check("values in [0,1]", lo.min() >= 0 and lo.max() <= 1,
          "lr [%.3f, %.3f]" % (lo.min(), lo.max()))
    full = SRPatches([site], crop_lr=0, train=False)
    flo, fhi = full[0]
    check("eval patches are full 128 / 256",
          tuple(flo.shape) == (4, 128, 128) and tuple(fhi.shape) == (4, 256, 256))

    print("\n[2] degradation operator D")
    x, y = flo[None].to(dev), fhi[None].to(dev)
    check("D(HR) has the same shape as LR", degrade(y).shape == x.shape)
    err = (degrade(y) - x).abs().mean().item()
    check("D(HR) is close to LR but not equal", 0.0 < err < 0.15,
          "mean |D(HR)-LR| = %.5f" % err)
    print("        (target is VENuS, input is Sentinel-2: two sensors, so")
    print("         they agree closely without being identical)")

    print("\n[3] metrics on trivial inputs")
    check("PSNR(x,x) is huge", psnr(y, y) > 80, "%.1f dB" % psnr(y, y))
    check("SSIM(x,x) == 1", abs(ssim(y, y) - 1.0) < 1e-4, "%.6f" % ssim(y, y))
    # floor of ~0.026 deg comes from clamping cos to 1-1e-7 to keep acos off
    # NaN; that clamp is deliberate, so the tolerance allows for it
    check("SAM(x,x) ~ 0 (clamp floor)", sam_degrees(y, y) < 0.05,
          "%.4f deg" % sam_degrees(y, y))
    check("NDVI MAE(x,x) == 0", ndvi_mae(y, y) < 1e-6)
    bic = bicubic_predict(x)
    check("bicubic upsample has HR shape", bic.shape == y.shape)
    ce = consistency_error(bic, x)
    check("bicubic scores LOW on M5 (it invents nothing)", ce < 0.01,
          "%.5f  <- this is why M5 cannot be read alone" % ce)

    print("\n[4] gradient probe (analysis A1)")
    set_seed(0)
    m = SRNet(n_blocks=4, c=32).to(dev)
    xb = torch.stack([ds[i][0] for i in range(4)]).to(dev)
    yb = torch.stack([ds[i][1] for i in range(4)]).to(dev)

    class SelfSpec(LossSpec):
        """Compare L1 against itself: the strongest possible redundancy."""
        def parts(self, out, hr, lr=None):
            l = F.l1_loss(out, hr)
            return l, l

    c_self = grad_cosine(m, SelfSpec("l1_sam"), xb, yb)
    check("cos(grad L1, grad L1) == 1 (upper bound)", abs(c_self - 1) < 1e-3,
          "%.6f" % c_self)
    c_sam = grad_cosine(m, LossSpec("l1_sam"), xb, yb)
    check("cos(grad L1, grad SAM) is in [-1,1]", -1 <= c_sam <= 1,
          "%.4f at init (untrained, so not yet meaningful)" % c_sam)
    c_con = grad_cosine(m, LossSpec("l1_consist"), xb, yb)
    check("cos(grad L1, grad consistency) is in [-1,1]", -1 <= c_con <= 1,
          "%.4f at init" % c_con)

    print("\n[4b] pi/2 filter bank")
    for nm in FILTERS:
        k = kernel_1d(nm)
        if abs(k.sum() - 1.0) > 1e-10:
            check("%s has DC gain 1" % nm, False)
            break
    else:
        check("every kernel has DC gain exactly 1 (reflectance safe)", True)
    d_box = degrade_with(y, "box2")
    check("box2 reproduces avg_pool2d (this is what pins the phase)",
          torch.allclose(d_box, F.avg_pool2d(y, 2), atol=1e-5),
          "max diff %.2e" % (d_box - F.avg_pool2d(y, 2)).abs().max().item())
    shapes_ok = all(degrade_with(y, nm).shape == x.shape for nm in FILTERS)
    check("every filter maps HR to the LR grid", shapes_ok)
    # a constant image must come through untouched, or D shifts reflectance
    const = torch.full_like(y, 0.3137)
    worst = max((degrade_with(const, nm) - 0.3137).abs().max().item()
                for nm in FILTERS)
    check("constant image preserved by every filter", worst < 1e-5,
          "worst %.2e" % worst)

    print("\n[4c] uncertainty head and its metrics")
    mu_ = SRNet(n_blocks=2, c=16, predict_sigma=True).to(dev)
    out = mu_(x)
    check("sigma model returns (mu, log_b)",
          isinstance(out, tuple) and len(out) == 2 and
          out[0].shape == out[1].shape == y.shape)
    plain = SRNet(n_blocks=2, c=16).to(dev)
    miss, unexp = mu_.load_state_dict(plain.state_dict(), strict=False)
    check("an R1 checkpoint still loads into the sigma model",
          not unexp and all(k.startswith("tail_logb") for k in miss))
    # NLL must reduce to L1 + log 2 when b = 1
    pa, ta = torch.rand(2, 4, 8, 8), torch.rand(2, 4, 8, 8)
    nll1 = laplace_nll(pa, torch.zeros_like(pa), ta).item()
    check("Laplace NLL reduces to L1 at b=1",
          abs(nll1 - (F.l1_loss(pa, ta).item() + math.log(2))) < 1e-6)
    # an oracle sigma must rank perfectly; a constant one must not rank at all
    err = torch.rand(20000).numpy()
    check("oracle sigma gives AUSE ~ 0", ause(err, err) < 1e-6)
    flat = torch.ones(20000).numpy()
    check("constant sigma is caught by AUSE", ause(err, flat) > 0.3,
          "%.3f (and Cv would be 0)" % ause(err, flat))

    print("\n[4d] augmentation keeps the pair consistent")
    # The whole defence of paired gain is that it scales BOTH sides equally.
    # Verify it rather than asserting it: the ratio HR/LR must be unchanged.
    base = SRPatches([site], crop_lr=64, train=True, aug=AugConfig())
    gain = SRPatches([site], crop_lr=64, train=True,
                     aug=AugConfig(gain_band=1.6))
    np.random.seed(0)
    a0, b0 = base[0]
    np.random.seed(0)
    a1, b1 = gain[0]
    # same geometric draw, so any difference is purely the gain
    r0 = (b0.mean(dim=(1, 2)) / a0.mean(dim=(1, 2)).clamp_min(1e-8))
    r1 = (b1.mean(dim=(1, 2)) / a1.mean(dim=(1, 2)).clamp_min(1e-8))
    check("per-band HR/LR ratio survives the gain augmentation",
          torch.allclose(r0, r1, rtol=2e-2),
          "max rel change %.4f" % ((r1 - r0).abs() / r0.clamp_min(1e-8)).max())
    check("augmentation actually changed the values",
          not torch.allclose(a0, a1))
    check("augmented values stay in [0,1]",
          a1.min() >= 0 and a1.max() <= 1 and b1.min() >= 0 and b1.max() <= 1)

    print("\n[5] OVERFIT-8")
    set_seed(0)
    m = SRNet(n_blocks=8, c=64).to(dev)
    print("        params: %s" % format(count_params(m), ","))
    sub = DataLoader(Subset(SRPatches([site], crop_lr=64, train=False),
                            list(range(8))), batch_size=4, shuffle=True)
    batches = [(a.to(dev), b.to(dev)) for a, b in sub]
    opt = torch.optim.Adam(m.parameters(), lr=2e-4)
    first = None
    for step in range(400):
        for a, b in batches:
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(m(a), b)
            loss.backward()
            opt.step()
        if first is None:
            first = loss.item()
        if (step + 1) % 100 == 0:
            print("        step %3d  L1 %.6f" % (step + 1, loss.item()))
    check("loss dropped more than 10x on 8 patches", loss.item() < first / 10,
          "%.6f -> %.6f" % (first, loss.item()))

    print("\n" + ("ALL SANITY CHECKS PASSED" if not fails
                  else "FAILURES: " + ", ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
