"""Post-midsem figures. Every panel the 8-slide deck needs.

    fig_filter_response   pi/2 anti-alias bank, |H(w)| and alias energy
    fig_site_spread       per-band reflectance per site: the biome argument
    fig_sparsification    AUSE: ours vs oracle vs random
    fig_calibration       reliability (RMV vs RMSE) and PICP
    fig_ood_sigma         does predicted sigma rise on the held-out biome
    fig_band_bias         signed per-band error: the brightness-prior diagnosis
    fig_grad_cosine       A1, now for SAM and for the consistency term
    fig_m5_sweep          how much M5 moves with the choice of D
    fig_confidence_strip  LR / bicubic / SR / HR / |error| / sigma
    fig_demo              the 2-minute in-slide demo sequence

`main` generates whatever the current results support and skips the rest, so
it is safe to run while the training queue is still going.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from degrade import FILTERS, W_C, freq_response, kernel_1d, response_table
from uncertainty import sparsification

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
    "axes.grid": True, "grid.color": "#e8eef3", "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
BANDS = ["B2 blue", "B3 green", "B4 red", "B8 NIR"]


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT), flush=True)
    return p


def _load():
    if not os.path.exists(FINAL):
        return None
    with open(FINAL) as f:
        return json.load(f)
# 1. the filter bank  (slide 7, needs no training)
def fig_filter_response(out="fig_filter_response.png"):
    """Why the 2x2 box was the wrong D, quantified.

    Left: magnitude responses, all nominally cut at pi/2. Right: the fraction
    of surviving energy that sits ABOVE pi/2 and therefore aliases when the
    image is decimated. The box leaves 18%; a proper filter leaves ~1%.
    """
    show = ["box2", "gauss3db", "gauss6db", "bicubic", "butter8",
            "lanczos3", "sinc_hamming", "mtf30", "mtf20"]
    rows = {r["filter"]: r for r in response_table()}

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12.4, 4.3), dpi=170,
        gridspec_kw={"width_ratios": [1.55, 1.0]})

    cmap = plt.get_cmap("viridis")
    for i, name in enumerate(show):
        h = kernel_1d(name)
        w, H = freq_response(h)
        lw = 3.0 if name == "box2" else 1.7
        col = HOT if name == "box2" else cmap(0.08 + 0.78 * i / len(show))
        ax.plot(w / math.pi, 20 * np.log10(np.maximum(H, 1e-6)),
                color=col, lw=lw, label="%s (%d tap)" % (name, len(h)),
                zorder=5 if name == "box2" else 3)

    ax.axvline(0.5, color=WARN, lw=1.6, ls="--", zorder=6)
    ax.axvspan(0.5, 1.0, color=WARN, alpha=0.06, lw=0)
    # Labels go in the corners the curves do not reach: every response is
    # monotone down across the stopband, so the bottom-right is always clear
    # and the strip above 0 dB is always clear.
    ax.annotate(r"cutoff $\omega_c=\pi/2$", xy=(0.5, 5), xytext=(0.52, 3.2),
                color=WARN, fontsize=9.5, va="top", ha="left")
    ax.annotate("stopband\neverything surviving here\nfolds back as aliasing",
                xy=(0.985, -57.5), ha="right", va="bottom", fontsize=8.8,
                color=WARN,
                bbox=dict(boxstyle="round,pad=0.34", fc="white",
                          ec="#f0d9c8", lw=0.9, alpha=0.93))
    ax.set_xlabel(r"normalised frequency  $\omega/\pi$")
    ax.set_ylabel(r"$|H(\omega)|$   (dB)")
    ax.set_title("Anti-alias filters for $\\times$2 decimation, all cut at $\\pi/2$",
                 fontsize=11, color=INK, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-60, 8)
    ax.legend(fontsize=7.8, loc="lower left", framealpha=0.96, ncol=2,
              bbox_to_anchor=(0.0, 0.0))

    names = sorted(show, key=lambda n: rows[n]["alias_energy_frac"])
    vals = [100 * rows[n]["alias_energy_frac"] for n in names]
    cols = [HOT if n == "box2" else ACC for n in names]
    bx.barh(range(len(names)), vals, color=cols, height=0.68)
    for i, v in enumerate(vals):
        bx.text(v + 0.3, i, "%.1f%%" % v, va="center", fontsize=9,
                color=HOT if names[i] == "box2" else INK,
                fontweight="bold" if names[i] == "box2" else "normal")
    bx.set_yticks(range(len(names)))
    bx.set_yticklabels(names, fontsize=9)
    bx.set_xlabel("energy above $\\pi/2$  (% of surviving energy)")
    bx.set_title("Aliasing left behind\n(box2 is the operator M5 used at mid-sem)",
                 fontsize=10.5, color=INK, loc="left")
    bx.set_xlim(0, max(vals) * 1.22)
    bx.grid(axis="y", visible=False)
    return _save(fig, out)
# 2. the biome argument  (slide 2)
def fig_site_spread(res, out="fig_site_spread.png"):
    """Per-band reflectance per site, with the training range shaded.

    This single panel carries the whole generalisation story: at mid-sem the
    training range was a narrow European band and the test site sat far
    outside it, so the model was extrapolating. It also shows that adding
    biomes widens the range WITHOUT swallowing the test site, so FGMANAUS
    stays genuinely out-of-domain rather than becoming an interpolation.
    """
    meta = res.get("meta", {})
    sites = meta.get("sites", {})
    if not sites:
        return None
    base = meta.get("train_base", [])
    multi = meta.get("train_multi", [])
    test = meta.get("test", ["FGMANAUS"])
    extra = [s for s in multi if s not in base]

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9), dpi=170, sharex=False)
    order = base + extra + [t for t in test if t in sites]

    for c, (ax, bname) in enumerate(zip(axes, BANDS)):
        vals = [sites[s]["mean"][c] for s in order if s in sites]
        labs = [s for s in order if s in sites]
        cols = []
        for s in labs:
            cols.append(ACC if s in base else
                        (GOOD if s in extra else HOT))
        # the mid-sem training range, which is what the model extrapolated from
        bv = [sites[s]["mean"][c] for s in base if s in sites]
        if bv:
            ax.axhspan(min(bv), max(bv), color=ACC, alpha=0.10, lw=0)
        ax.bar(range(len(vals)), vals, color=cols, width=0.68)
        for i, v in enumerate(vals):
            ax.text(i, v, " %.4f" % v, ha="center", va="bottom",
                    fontsize=7.6, rotation=90, color=INK)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=38, ha="right", fontsize=7.8)
        ax.set_title(bname, fontsize=10.5, color=INK)
        ax.set_ylim(0, max(vals) * 1.42)
        ax.grid(axis="x", visible=False)
        if c == 0:
            ax.set_ylabel("mean reflectance")

    h = [plt.Rectangle((0, 0), 1, 1, color=ACC),
         plt.Rectangle((0, 0), 1, 1, color=GOOD),
         plt.Rectangle((0, 0), 1, 1, color=HOT),
         plt.Rectangle((0, 0), 1, 1, color=ACC, alpha=0.10)]
    fig.legend(h, ["mid-sem training sites", "biomes added after mid-sem",
                   "held-out test site (sealed)", "mid-sem training range"],
               loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Why the model failed out of domain: the test site lies "
                 "outside the training range on every visible band",
                 fontsize=11.5, color=INK, y=1.03)
    return _save(fig, out)
# 3-5. uncertainty panels  (slide 6)
def _unc_runs(res):
    return [r for r in res.get("runs", {})
            if any("unc_ause" in m for m in res["runs"][r].get("ood", {}).values())]


def fig_sparsification(res, run=None, out="fig_sparsification.png"):
    """The AUSE curve: remove the most-uncertain pixels, watch error fall.

    Needs raw per-pixel values, so it is rebuilt from a checkpoint by
    `rebuild_sparsification` rather than from the summary JSON. If that has
    not been run, falls back to drawing the AUSE/AURG scalars as a bar.
    """
    cache = os.path.join(RESULTS, "sparsification.json")
    if not os.path.exists(cache):
        return None
    with open(cache) as f:
        d = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=170)
    for ax, regime in zip(axes, ["in", "ood"]):
        if regime not in d:
            continue
        c = d[regime]
        f_ = np.array(c["fracs"])
        ax.plot(f_, c["random"], color=MUT, lw=1.6, ls=":",
                label="random ranking")
        ax.fill_between(f_, c["oracle"], c["ours"], color=ACC, alpha=0.16,
                        lw=0, label="sparsification error (AUSE)")
        ax.plot(f_, c["ours"], color=ACC, lw=2.6,
                label="ranked by predicted $\\sigma$")
        ax.plot(f_, c["oracle"], color=GOOD, lw=2.0, ls="--",
                label="oracle (ranked by true error)")
        ax.set_xlabel("fraction of most-uncertain pixels removed")
        ax.set_ylabel("remaining MAE  (normalised)")
        ax.set_title("%s   AUSE %.4f   AURG %+.4f"
                     % ("in-domain" if regime == "in" else "out-of-domain (FGMANAUS)",
                        c["ause"], c["aurg"]),
                     fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8.4, loc="lower left", framealpha=0.95)
    fig.suptitle("Does the confidence map point at the errors? "
                 "Lower curve = better ranking; the gap to the oracle is AUSE",
                 fontsize=11.5, color=INK, y=1.02)
    return _save(fig, out)


def fig_calibration(res, out="fig_calibration.png"):
    """Reliability (predicted spread vs observed) and interval coverage."""
    runs = _unc_runs(res)
    if not runs:
        return None
    rid = runs[0]
    rec = res["runs"][rid]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=170)

    for regime, col, lab in (("in", ACC, "in-domain"),
                             ("ood", HOT, "out-of-domain")):
        seeds = rec.get(regime, {})
        bins = None
        for m in seeds.values():
            if m.get("unc_calib_bins"):
                bins = m["unc_calib_bins"]
                break
        if not bins:
            continue
        rmv = [b[0] for b in bins]
        rmse = [b[1] for b in bins]
        ax.plot(rmv, rmse, "o-", color=col, lw=1.8, ms=4.5, label=lab)
    lim = ax.get_xlim()
    hi = max(lim[1], ax.get_ylim()[1])
    ax.plot([0, hi], [0, hi], color=MUT, lw=1.3, ls="--",
            label="perfect calibration")
    ax.set_xlabel("predicted spread  RMV $=\\sqrt{\\langle\\sigma^2\\rangle}$")
    ax.set_ylabel("observed error  RMSE")
    ax.set_title("Reliability: is the stated uncertainty the real one?",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.6, loc="upper left", framealpha=0.95)

    levels = [50, 90, 95]
    width = 0.36
    for j, (regime, col, lab) in enumerate((("in", ACC, "in-domain"),
                                            ("ood", HOT, "out-of-domain"))):
        mean = rec.get(regime + "_mean", {})
        got = [100 * mean.get("unc_picp%d" % L, float("nan")) for L in levels]
        bx.bar(np.arange(len(levels)) + (j - 0.5) * width, got, width,
               color=col, label=lab)
    for i, L in enumerate(levels):
        bx.plot([i - 0.55, i + 0.55], [L, L], color=MUT, lw=1.8, ls="--",
                zorder=5)
    bx.set_xticks(range(len(levels)))
    bx.set_xticklabels(["%d%%" % L for L in levels])
    bx.set_xlabel("nominal Laplace interval")
    bx.set_ylabel("measured coverage  (%)")
    bx.set_title("PICP: do the stated intervals actually cover?\n"
                 "dashed = nominal (Laplace quantile, not Gaussian $z$)",
                 fontsize=10.5, color=INK, loc="left")
    bx.legend(fontsize=8.6, loc="lower right", framealpha=0.95)
    bx.grid(axis="x", visible=False)
    return _save(fig, out)


def fig_ood_sigma(res, out="fig_ood_sigma.png"):
    """The headline: does the confidence head notice the unseen biome?"""
    cache = os.path.join(RESULTS, "patch_sigma.json")
    if not os.path.exists(cache):
        return None
    with open(cache) as f:
        d = json.load(f)

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.6, 4.1), dpi=170,
        gridspec_kw={"width_ratios": [1.35, 1.0]})

    si = np.array(d["in"], dtype=float)
    so = np.array(d["ood"], dtype=float)
    lo, hi = min(si.min(), so.min()), max(si.max(), so.max())
    edges = np.linspace(lo, hi, 36)
    ax.hist(si, bins=edges, color=ACC, alpha=0.78, label="in-domain patches")
    ax.hist(so, bins=edges, color=HOT, alpha=0.78,
            label="FGMANAUS (never trained on)")
    ax.axvline(si.mean(), color=ACC, lw=2.0, ls="--")
    ax.axvline(so.mean(), color=HOT, lw=2.0, ls="--")
    ax.set_xlabel("mean predicted $\\sigma$ per patch  (reflectance)")
    ax.set_ylabel("patches")
    ax.set_title("Predicted confidence separates the unseen biome\n"
                 "AUROC %.3f   (0.5 = no signal)" % d["auroc"],
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.8, framealpha=0.95)

    bx.bar([0, 1], [si.mean(), so.mean()],
           yerr=[si.std(), so.std()], capsize=5,
           color=[ACC, HOT], width=0.56)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["in-domain", "out-of-domain"])
    bx.set_ylabel("mean predicted $\\sigma$")
    r = so.mean() / max(si.mean(), 1e-12)
    bx.set_title("Uncertainty rises $\\times$%.2f on the held-out biome" % r,
                 fontsize=10.5, color=INK, loc="left")
    bx.grid(axis="x", visible=False)
    return _save(fig, out)
# 6-8. analysis panels  (slides 7-8)
def fig_band_bias(res, out="fig_band_bias.png"):
    """Signed per-band error: the brightness prior, made visible.

    Signed is the whole point. An absolute error would show "the model is
    wrong on the test site"; the sign shows it is wrong in one DIRECTION on
    exactly the three visible bands, which is what makes it a learned prior
    rather than noise.
    """
    runs = res.get("runs", {})
    want = [r for r in ["R1", "R1a", "R1m", "R1ma", "R4", "R4m"] if r in runs]
    if not want:
        return None

    def bias_of(rid, regime):
        """Mean signed per-band bias for a run.

        Not read from `*_mean`: the aggregator only averages SCALAR metrics,
        so this list-valued one never lands there. Average the per-seed
        records instead, which also keeps it working for runs saved before
        the metric existed.
        """
        seeds = (runs[rid].get(regime) or {}).values()
        vals = [m["band_bias"] for m in seeds if m.get("band_bias")]
        if not vals:
            return None
        return [sum(v[c] for v in vals) / len(vals) for c in range(len(vals[0]))]

    if not any(bias_of(r, "ood") for r in want):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.1), dpi=170, sharey=True)
    x = np.arange(len(BANDS))
    for ax, regime, title in zip(
            axes, ["in", "ood"],
            ["in-domain", "out-of-domain (FGMANAUS)"]):
        w = 0.8 / max(len(want), 1)
        cmap = plt.get_cmap("viridis")
        for j, rid in enumerate(want):
            bb = bias_of(rid, regime)
            if not bb:
                continue
            ax.bar(x + (j - (len(want) - 1) / 2) * w,
                   [1000 * v for v in bb], w * 0.92,
                   color=cmap(0.08 + 0.8 * j / max(len(want), 1)),
                   label=rid)        # run id only; the full label overflows
        ax.axhline(0, color=INK, lw=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(BANDS)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
        if regime == "in":
            ax.set_ylabel("mean signed error  ($\\times 10^{-3}$ reflectance)")
    axes[1].legend(fontsize=7.6, ncol=2, framealpha=0.95)
    fig.suptitle("A learned brightness prior, not random error: the bias is "
                 "positive on all three visible bands",
                 fontsize=11.5, color=INK, y=1.03)
    return _save(fig, out)


def fig_ood_progress(res, out="fig_ood_progress.png"):
    """The project arc on the sealed site: diagnosis, then its two cures.

    Only the out-of-domain column is used, because FGMANAUS is the same 129
    patches for every row. The in-domain split differs between the two-site
    and six-site runs and is not comparable.
    """
    runs = res.get("runs", {})
    mid = os.path.join(RESULTS, "results.json")
    r1 = r0 = None
    if os.path.exists(mid):
        with open(mid) as f:
            mm = json.load(f).get("runs", {})
        r1 = (mm.get("R1") or {}).get("mean")
        r0 = (mm.get("R0") or {}).get("mean")
    if not r1 or not r0:
        return None

    # Short labels: the full descriptions collide at four bars.
    order = [("R1", "mid-sem\n2 sites", r1),
             ("R1a", "+ aug", None),
             ("R1m", "+ 6 biomes", None),
             ("R1ma", "+ both", None)]
    rows = []
    for rid, lab, pre in order:
        m = pre if pre is not None else (runs.get(rid) or {}).get("ood_mean")
        if m:
            rows.append((rid, lab, m))
    if len(rows) < 2:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), dpi=170)
    panels = [("psnr", "PSNR (dB)   higher better", False),
              ("sam", "SAM (deg)   lower better", True),
              ("ndvi_mae", "NDVI MAE   lower better", True)]
    for ax, (k, title, lower) in zip(axes, panels):
        ys = [m[k] for _, _, m in rows]
        xs = np.arange(len(rows))
        cmap = plt.get_cmap("viridis")
        cols = [HOT] + [cmap(0.15 + 0.6 * i / max(len(rows) - 1, 1))
                        for i in range(1, len(rows))]
        ax.bar(xs, ys, color=cols, width=0.62)
        for i, v in enumerate(ys):
            ax.text(i, v, "\n%.3f" % v if k != "ndvi_mae" else "\n%.4f" % v,
                    ha="center", va="top", fontsize=8.6, color="white",
                    fontweight="bold")
        # bicubic is the bar everything is trying to reach
        ax.axhline(r0[k], color=INK, lw=1.8, ls="--", zorder=5)
        # Anchor the label in AXES coordinates. Placing it at a data x of
        # len(rows)-0.45 pushed it past the last bar and outside the axis,
        # so the reference line rendered unlabelled.
        # Sit the label on the side the bars leave free: PSNR rises toward
        # the line from the left, SAM and NDVI fall toward it from the left,
        # so the free corner differs. Opaque bbox either way.
        at_left = lower
        ax.annotate(("bicubic %.4f" if k == "ndvi_mae" else "bicubic %.3f")
                    % r0[k],
                    xy=(0.03 if at_left else 0.97, r0[k]),
                    xycoords=("axes fraction", "data"),
                    ha="left" if at_left else "right",
                    va="bottom", fontsize=8.6, color=INK,
                    fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#cfdce6", lw=0.7, alpha=0.95))
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l, _ in rows], fontsize=8.6)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
        lo, hi = min(ys + [r0[k]]), max(ys + [r0[k]])
        pad = (hi - lo) * 0.22 + 1e-9
        ax.set_ylim(max(0, lo - pad), hi + pad)

    gap0 = r0["psnr"] - rows[0][2]["psnr"]
    gap1 = r0["psnr"] - rows[-1][2]["psnr"]
    fig.suptitle("Closing the out-of-domain gap on the sealed site: the "
                 "diagnosis predicted both cures  "
                 "(deficit vs bicubic %.2f dB $\\rightarrow$ %.2f dB)"
                 % (gap0, gap1), fontsize=11.5, color=INK, y=1.03)
    return _save(fig, out)


def fig_sigma_blend(res, out="fig_sigma_blend.png"):
    """The confidence head doing work, not just reporting.

    out = w*SR + (1-w)*bicubic with w = exp(-sigma/tau). Left panel is the
    held-out comparison on the two metrics where the blend wins; right
    panel is the in-domain cost, because a safety mechanism that were free
    everywhere would not be a trade and should be distrusted.
    """
    cache = os.path.join(RESULTS, "sigma_blend.json")
    if not os.path.exists(cache):
        return None
    with open(cache) as f:
        d = json.load(f)
    if not d.get("mean"):
        return None
    bi = d["bicubic"]["ood"]
    mp, mb, sb = d["mean"]["ood_pure"], d["mean"]["ood_blend"], d["std"]["ood_blend"]
    seeds = d["seeds"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=170,
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})

    for ax, k, lab, fmt in ((axes[0], "psnr", "held-out PSNR (dB)", "%.3f"),
                            (axes[1], "ssim", "held-out SSIM", "%.4f")):
        vals = [mp[k], mb[k], bi[k]]
        errs = [d["std"]["ood_pure"][k], sb[k], 0]
        cols = [MUT, GOOD, HOT]
        ax.bar(range(3), vals, yerr=errs, capsize=4, color=cols, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, "\n" + fmt % v, ha="center", va="top",
                    fontsize=9, color="white", fontweight="bold")
        ax.axhline(bi[k], color=HOT, lw=1.6, ls="--", zorder=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["pure SR", "$\\sigma$-blend", "bicubic"],
                           fontsize=9.5)
        ax.set_title(lab, fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - (hi - lo) * 0.45, hi + (hi - lo) * 0.28)

    # the trade: what it costs in-domain against what it buys out of domain
    ax = axes[2]
    sds = sorted(seeds)
    cost = [seeds[s]["val_blend"] - seeds[s]["val_pure"] for s in sds]
    gain = [seeds[s]["ood_blend"]["psnr"] - seeds[s]["ood_pure"]["psnr"]
            for s in sds]
    x = np.arange(len(sds))
    ax.bar(x - 0.18, cost, 0.34, color=WARN, label="in-domain cost")
    ax.bar(x + 0.18, gain, 0.34, color=GOOD, label="out-of-domain gain")
    for i in range(len(sds)):
        ax.text(x[i] - 0.18, cost[i], "%.2f" % cost[i], ha="center",
                va="top", fontsize=8.4, color=WARN)
        ax.text(x[i] + 0.18, gain[i], "%.2f" % gain[i], ha="center",
                va="bottom", fontsize=8.4, color=GOOD)
    ax.axhline(0, color=INK, lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed " + s for s in sds], fontsize=9)
    ax.set_ylabel("change in PSNR (dB)")
    ax.set_title("The trade, per seed", fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.6, framealpha=0.95, loc="upper left")
    ax.grid(axis="x", visible=False)

    taus = [seeds[s]["tau"] for s in sds]
    fig.suptitle("Using the confidence map at inference: fall back to bicubic "
                 "where the model is unsure  "
                 "($\\tau$ = mean in-domain $\\sigma$ = %.4f to %.4f, no test "
                 "data)" % (min(taus), max(taus)),
                 fontsize=11.2, color=INK, y=1.03)
    return _save(fig, out)


def fig_grad_cosine(res, out="fig_grad_cosine2.png"):
    """A1 extended: SAM was redundant with L1; is the consistency term?

    Histories come from BOTH result files. R2 was trained at mid-sem and
    lives in results.json; R3 is post-mid-sem and lives in
    results_final.json. Reading only the latter silently drops the SAM
    curve, which is the whole comparison.
    """
    hist = dict(res.get("history", {}))
    mid = os.path.join(RESULTS, "results.json")
    if os.path.exists(mid):
        with open(mid) as f:
            for k, v in (json.load(f).get("history") or {}).items():
                hist.setdefault(k, v)
    series = []
    for prefix, lab, col in (("R2_s", "L1 vs $\\lambda\\cdot$SAM", ACC),
                             ("R3_s", "L1 vs $\\mu\\cdot$consistency", GOOD)):
        ks = sorted(k for k in hist
                    if k.startswith(prefix) and hist[k].get("grad_cos"))
        if not ks:
            continue
        runs = [hist[k] for k in ks]
        n = min(len(r["grad_cos"]) for r in runs)
        xs = np.array(runs[0]["probe_step"][:n])
        ys = np.array([r["grad_cos"][:n] for r in runs], dtype=float)
        series.append((xs, ys, lab, col, len(runs)))
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=170)
    for xs, ys, lab, col, nseed in series:
        mean = ys.mean(0)
        if ys.shape[0] > 1:
            ax.fill_between(xs, ys.min(0), ys.max(0), color=col, alpha=0.13, lw=0)
        ax.plot(xs, mean, color=col, lw=0.8, alpha=0.4)
        w = 9
        if len(mean) >= w:
            sm = np.convolve(mean, np.ones(w) / w, mode="valid")
            ax.plot(xs[w // 2:w // 2 + len(sm)], sm, color=col, lw=2.8,
                    label="%s  (%d seeds, mean %.3f)"
                          % (lab, nseed, float(np.mean(mean[len(mean) // 2:]))),
                    zorder=4)
    ax.axhline(1.0, color=WARN, lw=1.3, ls="--")
    ax.annotate("cos = 1: redundant, adds no new direction", xy=(0.015, 0.985),
                xycoords=("axes fraction", "data"), fontsize=8.8, color=WARN,
                va="top")
    ax.axhline(0.0, color=MUT, lw=1.1, ls=":")
    ax.annotate("cos = 0: carries information L1 does not have",
                xy=(0.015, -0.03), xycoords=("axes fraction", "data"),
                fontsize=8.8, color=MUT, va="top")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\cos(\nabla_\theta L_1,\ \nabla_\theta L_{aux})$")
    ax.set_title("A1: how much does each auxiliary term turn the gradient?",
                 fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8.8, loc="lower left", framealpha=0.95)
    lo = min(float(np.nanmin(y)) for _, y, _, _, _ in series)
    ax.set_ylim(min(-0.35, lo - 0.05), 1.10)
    return _save(fig, out)


def fig_filter_match(res, out="fig_filter_match.png"):
    """Which pi/2 filter actually matches the real sensor pair?

    This is the empirical half of the filter study and it needs no trained
    model at all: apply each D to the real VENuS capture and compare with
    the real Sentinel-2 capture of the same ground. The filter that
    minimises |D(HR) - LR| is the best available model of what the Sentinel-2
    instrument does to the scene.

    The answer is not the filter with the best stopband. It is the
    sensor-matched Gaussian, specified by its magnitude at Nyquist, from the
    pan-sharpening literature (Aiazzi et al. 2006). The near-ideal filters
    -- windowed sinc, Lanczos, Butterworth -- have far sharper cutoffs and
    do WORSE, because a real instrument does not have a brick-wall MTF.
    """
    cache = os.path.join(RESULTS, "filter_match.json")
    if not os.path.exists(cache):
        return None
    with open(cache) as f:
        d = json.load(f)
    sites = [s for s in d if isinstance(d[s], dict)]
    if not sites:
        return None

    order = sorted(FILTERS, key=lambda f: d[sites[0]].get(f, 9e9))
    fig, axes = plt.subplots(1, len(sites), figsize=(6.2 * len(sites), 4.2),
                             dpi=170)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sites):
        vals = [1000 * d[s][f] for f in order]
        best = min(range(len(vals)), key=lambda i: vals[i])
        cols = [GOOD if i == best else (HOT if order[i] == "box2" else ACC)
                for i in range(len(order))]
        ax.barh(range(len(order)), vals, color=cols, height=0.68)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.012, i, "%.4f" % v, va="center",
                    fontsize=8.4,
                    fontweight="bold" if i in (best,) else "normal")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("$|D(HR)-LR|$   ($\\times 10^{-3}$ reflectance)")
        ax.set_title("%s   (%d patches)" % (s, d.get("n_patches", 0)),
                     fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(min(vals) * 0.96, max(vals) * 1.06)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Which $\\pi/2$ filter matches the real sensor? The "
                 "sensor-matched Gaussian wins on both biomes "
                 "— the near-ideal filters do worse",
                 fontsize=11.5, color=INK, y=1.03)
    # Say the two caveats on the figure itself rather than hoping they get
    # said out loud: the axis is truncated, and the bulk of the residual is
    # two different satellites disagreeing, not filter error.
    fig.text(0.5, -0.045,
             "Axis truncated to show the ordering. LR and HR come from "
             "different instruments, so most of the residual is cross-sensor "
             "disagreement, not filter error — the ranking is "
             "meaningful, the absolute level is not.",
             ha="center", fontsize=8.8, color=MUT)
    return _save(fig, out)


def fig_m5_sweep(res, out="fig_m5_sweep.png"):
    """M5 under all ten filters: a reported number that moves with D."""
    sw = res.get("sweeps", {})
    if not sw:
        return None
    rids = [r for r in ["R1", "R3", "R4", "R1m", "R4m"] if r in sw]
    if not rids:
        rids = sorted(sw)[:4]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), dpi=170, sharey=True)
    names = [f for f in FILTERS]
    x = np.arange(len(names))
    cmap = plt.get_cmap("viridis")
    for ax, regime, title in zip(axes, ["in", "ood"],
                                 ["in-domain", "out-of-domain (FGMANAUS)"]):
        for j, rid in enumerate(rids):
            d = sw[rid].get(regime, {})
            ys = [1000 * d.get(f, float("nan")) for f in names]
            ax.plot(x, ys, "o-", lw=1.9, ms=4.6,
                    color=cmap(0.08 + 0.78 * j / max(len(rids), 1)), label=rid)
        ax.axvline(0, color=HOT, lw=1.4, ls="--")
        ax.annotate("box2:\nthe mid-sem D", xy=(0.12, 0.96),
                    xycoords="axes fraction", fontsize=8.6, color=HOT,
                    va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8.4)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("M5  $|D(SR)-LR|$   ($\\times 10^{-3}$)")
    axes[1].legend(fontsize=8.6, framealpha=0.95)
    fig.suptitle("M5 is defined through D, so it moves with a choice nobody "
                 "had justified", fontsize=11.5, color=INK, y=1.03)
    return _save(fig, out)
# main
BUILDERS = [
    ("filter_response", lambda r: fig_filter_response(), False),
    ("filter_match", fig_filter_match, False),
    ("site_spread", fig_site_spread, True),
    ("ood_progress", fig_ood_progress, True),
    ("sigma_blend", fig_sigma_blend, False),
    ("grad_cosine", fig_grad_cosine, True),
    ("band_bias", fig_band_bias, True),
    ("m5_sweep", fig_m5_sweep, True),
    ("sparsification", fig_sparsification, True),
    ("calibration", fig_calibration, True),
    ("ood_sigma", fig_ood_sigma, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    res = _load()
    made, skipped = [], []
    for name, fn, needs in BUILDERS:
        if a.only and name not in a.only:
            continue
        if needs and res is None:
            skipped.append(name)
            continue
        try:
            p = fn(res)
        except Exception as e:                      # keep going; a missing
            print("  [skip] %s: %s" % (name, e))    # run shouldn't kill the lot
            skipped.append(name)
            continue
        (made if p else skipped).append(name)
    print("\nmade    %s" % (", ".join(made) or "-"))
    print("pending %s" % (", ".join(skipped) or "-"))


if __name__ == "__main__":
    main()
