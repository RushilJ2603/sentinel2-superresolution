"""Is the remaining out-of-domain gap to bicubic closable, or structural?

Bicubic scores 48.0 dB on FGMANAUS and 42.2 dB in-domain. A fixed algorithm
with no parameters doing 5.9 dB BETTER on the held-out site says something
about the site, not about our model. Before spending GPU hours chasing the
last 0.4 dB, find out what.

Four diagnostics, no training:

  1. HIGH-FREQUENCY CONTENT. If FGMANAUS carries less detail above the
     10 m Nyquist than the training sites, there is simply less for
     super-resolution to recover and bicubic gives away less by blurring.

  2. DOES BLUR HELP? Convolve our SR output with a small Gaussian and
     re-score. If PSNR goes UP, the site rewards blur, and bicubic's
     advantage is its blur rather than its fidelity. This is the
     perception-distortion tradeoff showing up as a metric artefact.

  3. SUB-PIXEL REGISTRATION. Shift the prediction against the target on a
     sub-pixel grid. If the best score is at a non-zero shift, LR and HR
     are misaligned on this site, which penalises a sharp prediction and
     forgives a blurry one.

  4. WHERE does bicubic win? Split patches by local gradient energy. If we
     win on textured patches and lose on smooth ones, the aggregate is
     being decided by the smooth majority.

Writes results/ood_diagnosis.json.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import SRPatches, loaders
from metrics import bicubic_predict, psnr, ssim
from model import SRNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(RESULTS, "ckpt")


def hf_energy(site, n=60):
    """Fraction of HR spectral energy above the 10 m Nyquist (pi/2).

    That band is exactly what super-resolution has to invent: it is absent
    from the LR input by construction. More of it means more to win.
    """
    ds = SRPatches([site], crop_lr=0, train=False)
    k = min(n, len(ds))
    tot = hi = 0.0
    for i in np.linspace(0, len(ds) - 1, k).astype(int):
        x = ds[int(i)][1].numpy()
        for c in range(x.shape[0]):
            F2 = np.abs(np.fft.fftshift(np.fft.fft2(x[c]))) ** 2
            H, W = F2.shape
            fy = np.abs(np.fft.fftshift(np.fft.fftfreq(H))) * 2.0   # 0..1
            fx = np.abs(np.fft.fftshift(np.fft.fftfreq(W))) * 2.0
            r = np.maximum(fy[:, None], fx[None, :])
            tot += F2.sum()
            hi += F2[r > 0.5].sum()        # above half the HR Nyquist
    return float(hi / max(tot, 1e-12))


def _gauss(sigma, dev, ch=4):
    r = max(1, int(round(3 * sigma)))
    g = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, 1, -1).repeat(ch, 1, 1, 1), r


@torch.no_grad()
def blur_sweep(predict, loader, dev, sigmas=(0.0, 0.3, 0.5, 0.7, 1.0, 1.4)):
    """PSNR of the prediction after blurring it by sigma pixels."""
    out = {}
    for s in sigmas:
        tot, n = 0.0, 0
        for lr, hr in loader:
            lr, hr = lr.to(dev), hr.to(dev)
            p = predict(lr).clamp(0, 1)
            if s > 0:
                k, r = _gauss(s, dev, p.shape[1])
                p = F.conv2d(F.pad(p, (r, r, 0, 0), mode="reflect"), k,
                             groups=p.shape[1])
                p = F.conv2d(F.pad(p, (0, 0, r, r), mode="reflect"),
                             k.transpose(2, 3), groups=p.shape[1])
            tot += psnr(p.clamp(0, 1), hr) * lr.shape[0]
            n += lr.shape[0]
        out[s] = tot / max(n, 1)
    return out


@torch.no_grad()
def shift_sweep(predict, loader, dev, shifts=(-1.0, -0.5, -0.25, 0.0,
                                              0.25, 0.5, 1.0)):
    """PSNR after shifting the prediction by a sub-pixel amount.

    Uses bilinear grid_sample. A peak away from 0.0 means the pair is
    misregistered on this site.
    """
    out = {}
    for s in shifts:
        tot, n = 0.0, 0
        for lr, hr in loader:
            lr, hr = lr.to(dev), hr.to(dev)
            p = predict(lr).clamp(0, 1)
            if s != 0.0:
                B, C, H, W = p.shape
                ys, xs = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=dev),
                    torch.linspace(-1, 1, W, device=dev), indexing="ij")
                grid = torch.stack([xs + 2 * s / W, ys], -1)[None].repeat(B, 1, 1, 1)
                p = F.grid_sample(p, grid, mode="bilinear",
                                  padding_mode="reflection", align_corners=True)
            tot += psnr(p.clamp(0, 1), hr) * lr.shape[0]
            n += lr.shape[0]
        out[s] = tot / max(n, 1)
    return out


@torch.no_grad()
def by_texture(predict, loader, dev, n_bins=4):
    """Win/lose against bicubic as a function of local detail."""
    rows = []
    for lr, hr in loader:
        lr, hr = lr.to(dev), hr.to(dev)
        sr = predict(lr).clamp(0, 1)
        bic = bicubic_predict(lr).clamp(0, 1)
        for i in range(lr.shape[0]):
            h = hr[i:i + 1]
            g = (h.diff(dim=2).abs().mean() + h.diff(dim=3).abs().mean()).item()
            rows.append((g, psnr(sr[i:i + 1], h), psnr(bic[i:i + 1], h)))
    rows.sort(key=lambda t: t[0])
    out = []
    edges = np.linspace(0, len(rows), n_bins + 1).astype(int)
    for b in range(n_bins):
        chunk = rows[edges[b]:edges[b + 1]]
        if not chunk:
            continue
        out.append({
            "bin": b + 1,
            "n": len(chunk),
            "grad_mean": float(np.mean([c[0] for c in chunk])),
            "sr_psnr": float(np.mean([c[1] for c in chunk])),
            "bic_psnr": float(np.mean([c[2] for c in chunk])),
        })
        out[-1]["delta"] = out[-1]["sr_psnr"] - out[-1]["bic_psnr"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R1ma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-sites", nargs="+",
                    default=["ESTUAMAR", "SUDOUE-4", "K34-AMAZ", "MAD-AMBO",
                             "ANJI", "NARYN"])
    ap.add_argument("--test-sites", nargs="+", default=["FGMANAUS"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"run": a.run}

    print("1. high-frequency content above half the HR Nyquist")
    for s in a.train_sites + a.test_sites:
        try:
            e = hf_energy(s)
        except Exception as ex:
            print("   %-10s skipped (%s)" % (s, ex))
            continue
        out.setdefault("hf_energy", {})[s] = e
        tag = "   <- TEST" if s in a.test_sites else ""
        print("   %-10s %.5f%s" % (s, e, tag))

    p = os.path.join(CKPT, "%s_s%d.pt" % (a.run, a.seed))
    if not os.path.exists(p):
        print("\nno checkpoint %s; stopping after diagnostic 1" % p)
        with open(os.path.join(RESULTS, "ood_diagnosis.json"), "w") as f:
            json.dump(out, f, indent=2)
        return
    model = SRNet().to(dev)
    model.load_state_dict(torch.load(p, map_location=dev))
    model.eval()
    _, va, te = loaders(a.train_sites, a.test_sites, batch=8, crop_lr=64, seed=0)

    print("\n2. does blurring our output IMPROVE it? (FGMANAUS)")
    bs = blur_sweep(model, te, dev)
    out["blur_sweep_ood"] = {str(k): v for k, v in bs.items()}
    best = max(bs, key=bs.get)
    for s, v in bs.items():
        print("   sigma %.1f px  PSNR %.3f%s" % (s, v, "   <- best" if s == best else ""))
    print("   verdict: %s" % ("blur HELPS, the site rewards smoothing"
                              if best > 0 else "blur hurts, our sharpness is earned"))

    print("\n3. sub-pixel registration (FGMANAUS)")
    ss = shift_sweep(model, te, dev)
    out["shift_sweep_ood"] = {str(k): v for k, v in ss.items()}
    bshift = max(ss, key=ss.get)
    for s, v in ss.items():
        print("   shift %+.2f px  PSNR %.3f%s" % (s, v, "   <- best" if s == bshift else ""))
    print("   verdict: %s" % ("aligned" if bshift == 0.0 else
                              "MISALIGNED by about %+.2f px" % bshift))

    print("\n4. where does bicubic win? (FGMANAUS, patches by detail)")
    bt = by_texture(model, te, dev)
    out["by_texture_ood"] = bt
    print("   bin  n   grad      ours     bicubic   delta")
    for r in bt:
        print("   %d   %3d  %.5f  %7.3f  %7.3f  %+6.3f"
              % (r["bin"], r["n"], r["grad_mean"], r["sr_psnr"],
                 r["bic_psnr"], r["delta"]))

    with open(os.path.join(RESULTS, "ood_diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/ood_diagnosis.json")


if __name__ == "__main__":
    main()
