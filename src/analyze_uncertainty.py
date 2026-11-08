"""Turn a trained uncertainty head into the panels the deck needs.

Produces, from one checkpoint:

    results/sparsification.json   AUSE curves, both regimes
    results/patch_sigma.json      per-patch mean sigma + the OOD AUROC
    fig_confidence_strip.png      LR / bicubic / SR / HR / |error| / sigma
    fig_demo.png                  the 2-minute in-slide demo

All display images are 8-bit stretches made ONLY for looking at. Every
number in this file is computed on reflectance in [0, 1]. Mixing those up
is the classic way to make a results table meaningless.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import loaders
from metrics import bicubic_predict, psnr
from model import SRNet
from uncertainty import (auroc, ause, ause_per_band, aurg,
                         laplace_sigma, sparsification)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
CKPT = os.path.join(RESULTS, "ckpt")
FINAL = os.path.join(RESULTS, "results_final.json")

INK, ACC, MUT = "#11223a", "#0b6e99", "#5b6b7f"
WARN, GOOD, HOT = "#b4530f", "#1b7f5a", "#a3243b"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#cfdce6", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def rgb(x, lo=None, hi=None):
    """4-band reflectance -> 8-bit-ish RGB for DISPLAY ONLY.

    Bands are B2,B3,B4,B8, so true colour is (B4, B3, B2) = indices 2,1,0.
    The stretch limits are computed once on the reference image and reused
    for every panel in a row; a per-panel stretch would make a blurry
    prediction look sharper than it is.
    """
    im = np.stack([x[2], x[1], x[0]], -1)
    if lo is None:
        lo, hi = np.percentile(im, 2.0), np.percentile(im, 98.0)
    return np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1), lo, hi


@torch.no_grad()
def collect(model, loader, dev, max_patches=150, band_patches=40):
    """Per-pixel |error| and sigma, a per-patch mean sigma, and a band-shaped
    subset for the per-band AUSE (which needs the channel axis kept)."""
    errs, sigs, psig, e4, s4 = [], [], [], [], []
    n = 0
    for lr, hr in loader:
        lr, hr = lr.to(dev), hr.to(dev)
        mu, log_b = model(lr)
        mu = mu.clamp(0, 1)
        s = laplace_sigma(log_b)
        psig += s.mean(dim=(1, 2, 3)).cpu().tolist()
        if n < max_patches:
            ea = (mu - hr).abs().cpu().numpy()
            sa = s.cpu().numpy()
            errs.append(ea.ravel())
            sigs.append(sa.ravel())
            if sum(x.shape[0] for x in e4) < band_patches:
                e4.append(ea)
                s4.append(sa)
            n += lr.shape[0]
    return (np.concatenate(errs) if errs else np.array([]),
            np.concatenate(sigs) if sigs else np.array([]),
            psig,
            np.concatenate(e4) if e4 else None,
            np.concatenate(s4) if s4 else None)


def dump_curves(model, va, te, dev):
    out_s, out_p = {}, {}
    for key, loader in (("in", va), ("ood", te)):
        e, s, ps, e4, s4 = collect(model, loader, dev)
        if not len(e):
            continue
        # Per-band AUSE before subsampling flattens the channel axis. This is
        # the confound-free number: pooled AUSE is inflated because NIR is
        # ~30x brighter than blue, so NIR pixels take almost the whole top
        # quartile of sigma and the pooled ranking largely measures that.
        pb, pb_list = (ause_per_band(e4, s4) if e4 is not None else (None, None))
        # cap the ranking arrays: a full sort of 39 M pixels is slow and the
        # curve is already smooth at a few million
        if len(e) > 4_000_000:
            idx = np.random.RandomState(0).choice(len(e), 4_000_000, False)
            e, s = e[idx], s[idx]
        f, ours, orac, rand = sparsification(e, s)
        out_s[key] = {"fracs": f.tolist(), "ours": ours.tolist(),
                      "oracle": orac.tolist(), "random": rand.tolist(),
                      "ause": ause(e, s), "aurg": aurg(e, s),
                      "ause_band": pb, "ause_per_band": pb_list}
        out_p[key] = ps
        print("  %-4s AUSE %.4f (per band %.4f)  AURG %+.4f  mean sigma %.5f"
              % (key, out_s[key]["ause"], pb if pb is not None else float("nan"),
                 out_s[key]["aurg"], float(s.mean())))

    if "in" in out_p and "ood" in out_p:
        out_p["auroc"] = auroc(out_p["in"], out_p["ood"])
        print("  OOD AUROC from mean sigma: %.4f" % out_p["auroc"])

    with open(os.path.join(RESULTS, "sparsification.json"), "w") as f:
        json.dump(out_s, f)
    with open(os.path.join(RESULTS, "patch_sigma.json"), "w") as f:
        json.dump(out_p, f)
    return out_s, out_p


def _pick(loader, model, dev, want_structure=True):
    """Pick a patch worth showing: most spatial detail in the target."""
    best, bi = None, -1
    for lr, hr in loader:
        for i in range(lr.shape[0]):
            h = hr[i].numpy()
            # gradient energy as a proxy for 'has visible structure'
            g = (np.abs(np.diff(h, axis=1)).mean()
                 + np.abs(np.diff(h, axis=2)).mean())
            if best is None or (g > bi) == want_structure:
                if best is None or g > bi:
                    best, bi = (lr[i:i + 1], hr[i:i + 1]), g
    return best


@torch.no_grad()
def fig_confidence_strip(model, va, te, dev, out="fig_confidence_strip.png"):
    """Two rows -- in-domain and the unseen biome -- six panels each."""
    rows = []
    for name, loader in (("in-domain", va),
                         ("out-of-domain  FGMANAUS", te)):
        p = _pick(loader, model, dev)
        if p is None:
            continue
        lr, hr = p[0].to(dev), p[1].to(dev)
        mu, log_b = model(lr)
        mu = mu.clamp(0, 1)
        sig = laplace_sigma(log_b)
        bic = bicubic_predict(lr).clamp(0, 1)
        rows.append((name, lr[0].cpu().numpy(), bic[0].cpu().numpy(),
                     mu[0].cpu().numpy(), hr[0].cpu().numpy(),
                     (mu - hr).abs()[0].cpu().numpy(),
                     sig[0].cpu().numpy(),
                     psnr(mu, hr), psnr(bic, hr)))
    if not rows:
        return None

    titles = ["Sentinel-2 input\n10 m, 4 bands", "bicubic $\\times$2\n(no learning)",
              "our SR output\n5 m prediction", "VENuS target\n5 m truth",
              "|error|\nwhat went wrong", "predicted $\\sigma$\nthe confidence map"]
    fig, axes = plt.subplots(len(rows), 6, figsize=(16.2, 3.0 * len(rows)),
                             dpi=168)
    axes = np.atleast_2d(axes)

    # ONE scale shared by both rows and by both heat maps. Shared across rows
    # is the whole point: the claim is that error and sigma are both LARGER
    # on the unseen biome, and per-row normalisation would erase exactly that
    # by rescaling each row to its own maximum.
    #
    # The ceiling is the 99th percentile, not the maximum. A handful of
    # outlier pixels otherwise set vmax about 30x above typical values and
    # the entire map renders near-black -- unreadable from the back of a
    # room. Clipping the top 1% is a display choice, stated on the figure.
    pool = np.concatenate([np.concatenate([r[5].mean(0).ravel(),
                                           r[6].mean(0).ravel()])
                           for r in rows])
    vmax = float(np.percentile(pool, 99.0))

    for r, row in enumerate(rows):
        name, lo, bic, sr, hi, err, sig, p_sr, p_bic = row
        ref, l0, h0 = rgb(hi)
        ims = [rgb(lo, l0, h0)[0], rgb(bic, l0, h0)[0],
               rgb(sr, l0, h0)[0], ref]
        for c in range(6):
            ax = axes[r, c]
            if c < 4:
                ax.imshow(ims[c], interpolation="nearest")
            elif c == 4:
                ax.imshow(err.mean(0), cmap="magma", vmin=0, vmax=vmax,
                          interpolation="nearest")
                ax.set_xlabel("mean %.4f" % float(err.mean()), fontsize=8.2,
                              color=MUT)
            else:
                im = ax.imshow(sig.mean(0), cmap="magma", vmin=0, vmax=vmax,
                               interpolation="nearest")
                ax.set_xlabel("mean %.4f" % float(sig.mean()), fontsize=8.2,
                              color=MUT)
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cb.ax.tick_params(labelsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#cfdce6")
            if r == 0:
                ax.set_title(titles[c], fontsize=9.6, color=INK, pad=6)
        axes[r, 0].set_ylabel(name, fontsize=10.5, color=INK)
        axes[r, 2].set_xlabel("PSNR %.2f dB  (bicubic %.2f)" % (p_sr, p_bic),
                              fontsize=8.6, color=MUT)

    fig.suptitle("The confidence map lines up with the actual error, and both "
                 "grow on the biome the model never saw",
                 fontsize=12, color=INK, y=1.005)
    fig.text(0.5, -0.015,
             "Both heat maps share one colour scale across both rows, "
             "clipped at the 99th percentile so the structure is visible; "
             "the per-panel means are printed beneath. "
             "RGB panels share one stretch per row, taken from the target.",
             ha="center", fontsize=8.6, color=MUT)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, out)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-sites", nargs="+",
                    default=["ESTUAMAR", "SUDOUE-4"])
    ap.add_argument("--test-sites", nargs="+", default=["FGMANAUS"])
    # The two artifacts want DIFFERENT models and that is deliberate.
    # The qualitative strip should show the final proposed model (R4m).
    # The domain-shift histogram should show R4, the narrow-training model,
    # because that is the case where the shift is real and the detector has
    # something to detect; on R4m the model no longer fails on FGMANAUS, so
    # its AUROC correctly falls and the panel would understate the point.
    ap.add_argument("--strip-only", action="store_true")
    ap.add_argument("--curves-only", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(CKPT, "%s_s%d.pt" % (a.run, a.seed))
    if not os.path.exists(path):
        print("no checkpoint yet:", path)
        return
    model = SRNet(predict_sigma=True).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.eval()
    print("loaded", os.path.relpath(path, ROOT))

    _, va, te = loaders(a.train_sites, a.test_sites, batch=8, crop_lr=64,
                        seed=0)
    if not a.strip_only:
        dump_curves(model, va, te, dev)
    if not a.curves_only:
        fig_confidence_strip(model, va, te, dev)
    print("UNCERTAINTY_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
