"""The two mid-sem figures.

    A1  gradient cosine similarity vs training step  -> fig_a1_gradient_cosine.png
    Q   qualitative strip, LR / bicubic / SR / HR    -> fig_qualitative.png
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
from metrics import bicubic_predict
from model import SRNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
CKPT = os.path.join(RESULTS, "ckpt")

INK, ACC, MUT = "#11223a", "#0b6e99", "#5b6b7f"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#cfdce6", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.grid": True, "grid.color": "#e8eef3", "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def fig_a1(res, out):
    """Plot cos(grad L1, grad SAM) across training, mean and seed range."""
    hist = res["history"]
    keys = sorted(k for k in hist if k.startswith("R2_s") and hist[k]["grad_cos"])
    if not keys:
        return None
    runs = [hist[k] for k in keys]
    n = min(len(r["grad_cos"]) for r in runs)
    xs = np.array(runs[0]["probe_step"][:n])
    ys = np.array([r["grad_cos"][:n] for r in runs], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=170)
    mean = ys.mean(0)
    if len(runs) > 1:
        ax.fill_between(xs, ys.min(0), ys.max(0), color=ACC, alpha=0.15,
                        lw=0, label="seed range (%d seeds)" % len(runs))
    # raw per-probe values bounce around a lot because each probe uses one
    # batch; the rolling mean is what you actually read off the plot
    ax.plot(xs, mean, color=ACC, lw=0.8, alpha=0.45, label="per-probe mean")
    w = 9
    if len(mean) >= w:
        k = np.ones(w) / w
        sm = np.convolve(mean, k, mode="valid")
        ax.plot(xs[w // 2:w // 2 + len(sm)], sm, color=ACC, lw=2.6,
                label="rolling mean (%d probes)" % w, zorder=4)
    ax.axhline(1.0, color="#b4530f", lw=1.3, ls="--",
               label="cos = 1: gradients parallel")
    ax.set_xlabel("training step")
    ax.set_ylabel("cos( ∇L1 ,  ∇SAM )")
    ax.set_ylim(0.0, 1.10)
    ax.set_xlim(0, xs.max() * 1.01)
    leg = ax.legend(frameon=True, fontsize=8.5, loc="lower right",
                    facecolor="white", edgecolor="#cfdce6", framealpha=0.95)
    leg.get_frame().set_linewidth(0.8)
    tail = ys[:, n // 2:].mean()
    fig.text(0.013, 0.016,
             "Mean over the second half of training: %.3f" % tail,
             fontsize=8.5, color=MUT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    print("   A1 second-half mean cosine: %.4f" % tail)
    return out


def _rgb(t, lo, hi):
    """4 bands -> RGB using B4(red), B3(green), B2(blue), fixed stretch."""
    x = t[[2, 1, 0]].detach().cpu().numpy()
    x = (x - lo) / max(hi - lo, 1e-6)
    return np.clip(np.transpose(x, (1, 2, 0)), 0, 1)


@torch.no_grad()
def fig_qualitative(te, dev, out, blocks, channels, which="R1_s0"):
    ck = os.path.join(CKPT, which + ".pt")
    model = None
    if os.path.exists(ck):
        model = SRNet(in_ch=4, c=channels, n_blocks=blocks).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev))
        model.eval()

    ds = te.dataset
    # pick the patch with the most structure, so the comparison is visible
    best, idx = -1, 0
    for i in range(len(ds)):
        _, h = ds[i]
        g = (h[:, 1:, :] - h[:, :-1, :]).abs().mean().item()
        if g > best:
            best, idx = g, i
    lo_t, hi_t = ds[idx]
    lo_t, hi_t = lo_t[None].to(dev), hi_t[None].to(dev)
    bic = bicubic_predict(lo_t).clamp(0, 1)
    sr = model(lo_t).clamp(0, 1) if model is not None else bic

    v = hi_t[0, [2, 1, 0]].flatten().cpu().numpy()
    plo, phi = np.percentile(v, 2), np.percentile(v, 98)

    panels = [("Low-res input (10 m)", lo_t[0]), ("Bicubic x2", bic[0]),
              ("SR model (%s)" % which.split("_")[0], sr[0]),
              ("Ground truth (5 m)", hi_t[0])]
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), dpi=175)
    for ax, (t, im) in zip(axes, panels):
        ax.imshow(_rgb(im, plo, phi), interpolation="nearest")
        ax.set_title(t, fontsize=10, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color("#cfdce6")
    fig.suptitle("Held-out site FGMANAUS, patch %d. Identical crop and "
                 "contrast stretch." % idx, x=0.012, ha="left", fontsize=11,
                 color=MUT)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(RESULTS, "results.json"))
    ap.add_argument("--qual-run", default="R1_s0")
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    with open(a.results) as f:
        res = json.load(f)
    cfg = res["config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, te = loaders(cfg["train_sites"], cfg["test_sites"],
                       batch=cfg["batch"], crop_lr=cfg["crop"], seed=0)
    made = [f for f in [
        fig_a1(res, os.path.join(FIGS, "fig_a1_gradient_cosine.png")),
        fig_qualitative(te, dev, os.path.join(FIGS, "fig_qualitative.png"),
                        cfg["blocks"], cfg["channels"], a.qual_run)] if f]
    print("figures written:", [os.path.basename(f) for f in made])
    print("FIGURES_DONE")


if __name__ == "__main__":
    main()
