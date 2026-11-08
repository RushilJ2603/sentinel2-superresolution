"""The full post-midsem experiment queue, resumable.

    R0    Bicubic x2                          the non-learned floor
    R1    CNN, L1                             baseline               (midsem)
    R2    CNN, L1 + lam*SAM                   spectral term          (midsem)
    R3    CNN, L1 + mu*|D(SR)-LR|             observation constraint
    R4    CNN, Laplace NLL + sigma head       predicted confidence
    R1a   R1 + paired-gain augmentation       does aug fix the bias
    R1m   R1 on the multi-biome training set  does more biome fix it
    R1ma  R1 multi-biome + augmentation       both together
    R4m   R4 on multi-biome                   the final proposed model

RESUMABILITY is the whole point of this file. Every run writes its metrics
and checkpoint as soon as it finishes, and a relaunch skips anything already
complete. So an interrupted session -- a dropped connection, a rate limit, a
reboot -- loses at most the run in flight, and simply running this script
again continues the queue.

PROTOCOL, unchanged from the midsem and not negotiable:
  - the test site is scored once per run and never used for any decision
  - early stopping uses the in-domain validation split of the TRAINING sites
  - validation and test are never augmented; they are measurements
  - 3 seeds on every row that carries a claim, reported mean +- population std
  - R1 gets the same architecture, schedule, step count and effort as the
    rows being compared against it
"""
import argparse
import json
import os
import statistics
import time
import traceback

import torch

from data import AugConfig, available_sites, loaders, site_stats
from degrade import FILTERS, response_table
from losses import LossSpec
from metrics import bicubic_predict, evaluate, filter_sweep
from train import train_one
from uncertainty import auroc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(RESULTS, "ckpt")
OUT_JSON = os.path.join(RESULTS, "results_final.json")

# Midsem training pair, and the two biomes added after it. Chosen from the
# per-band statistics rather than from guessed geography: K34-AMAZ is dark
# tropical canopy (red 0.0075) and MAD-AMBO is the brightest red of all
# (0.0644), so together they widen the training range instead of merely
# shifting it toward the test site.
BASE_TRAIN = ["ESTUAMAR", "SUDOUE-4"]
EXTRA_TRAIN = ["K34-AMAZ", "MAD-AMBO", "ANJI", "NARYN"]
TEST_SITE = ["FGMANAUS"]

AUG_OFF = dict(geometric=True)
AUG_ON = dict(geometric=True, gain_band=1.6, mixup=0.2, cutmix=0.2)

# id -> (loss kind, multi-biome?, augmentation kwargs, beta override)
# Ordered by priority: uncertainty head first, then
# augmentation, then the multi-biome work. The filter study needs no
# training at all and rides along inside every evaluation.
QUEUE = [
    ("R4",    "nll",        False, AUG_OFF, None),
    ("R3",    "l1_consist", False, AUG_OFF, None),
    ("R1a",   "l1",         False, AUG_ON,  None),
    ("R1m",   "l1",         True,  AUG_OFF, None),
    ("R1ma",  "l1",         True,  AUG_ON,  None),
    ("R4m",   "nll",        True,  AUG_ON,  None),
    # beta-NLL ablation: shows the heteroscedastic pathology empirically
    # rather than only citing it. One seed each is enough to expose a
    # effect this large, and they run last so they never block the main rows.
    ("R4b0",  "nll",        False, AUG_OFF, 0.0),
    ("R4b05", "nll",        False, AUG_OFF, 0.5),
]

# ---- round 2 -------------------------------------------------------------
# Added after the OOD diagnosis (results/ood_diagnosis.json). That analysis
# ruled out the comfortable explanations: FGMANAUS has the MOST
# high-frequency content of all seven sites, blurring our output makes it
# worse, and the pair is registered to within a sub-pixel. What it found
# instead is that we beat bicubic on the smoothest quarter of patches
# (+0.24 dB) and lose on the three more textured quarters (-0.63 to -0.72).
# The model is inventing high-frequency detail that is wrong for rainforest
# canopy.
#
# That is precisely what the back-projection term punishes: detail that
# does not survive being decimated back to the measurement. R3 gained
# +0.61 dB on two sites and has never been combined with multi-biome data
# and augmentation, so R3ma is the targeted fix and runs first.
#
# id -> (loss, multi?, aug, beta, arch, mu, global-residual)
#
# RES* are the strongest candidates for actually beating bicubic out of
# domain. Predicting a delta ON TOP of bicubic means the absolute level
# comes from the measurement rather than from a learned prior, and an
# unsure network can emit ~0 and fall back to bicubic instead of inventing
# texture that is wrong for the biome. Both are the exact failures the
# diagnosis found. The delta is zero-initialised, so training starts AT
# bicubic.
QUEUE2 = [
    # the three candidates for actually beating bicubic, first
    ("RESma",   "l1",         True, "ON",  None, None, None, True),
    ("RESma_c", "l1_consist", True, "ON",  None, None, None, True),
    ("R3ma",    "l1_consist", True, "ON",  None, None, None, False),
    # cheap learned baselines, to stop the comparison being bicubic-only
    ("B_srcnn", "l1",         True, "ON",  None, "srcnn", None, False),
    ("B_rcan",  "l1",         True, "ON",  None, "rcan", None, False),
    # the final model, if the residual trick works
    ("RESma_u", "nll",        True, "ON",  0.5,  None, None, True),
    # secondary: capacity check and two small sweeps. Expensive or
    # low-stakes, so they sit at the tail where they can be cut.
    ("B_edsrL", "l1",         True, "ON",  None, "edsr_large", None, False),
    ("R3ma_m3", "l1_consist", True, "ON",  None, None, 0.3, False),
    ("R4m_b05", "nll",        True, "ON",  0.5,  None, None, False),
]

SINGLE_SEED = {"R4b0", "R4b05", "R3ma_m3"}

# The two rows the whole night exists to produce, in case the queue has to
# be cut short. Everything after these is additive.
MUST_HAVE = ["R4", "R1m"]


def log(*a):
    print(*a, flush=True)


def load_state():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {"runs": {}, "history": {}, "sweeps": {}, "meta": {}}


def save_state(st):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, OUT_JSON)      # atomic: a crash mid-write can't corrupt it


def fmt(m):
    return ("PSNR %.3f  SSIM %.4f  SAM %.3f  NDVI %.5f  M5 %.5f"
            % (m["psnr"], m["ssim"], m["sam"], m["ndvi_mae"], m["consist"]))


def fmt_unc(m):
    if "unc_ause" not in m:
        return ""
    return ("      NLL %.4f  AUSE %.4f  AURG %.4f  ENCE %.4f  Cv %.3f  "
            "rho %.3f  PICP90 %.3f"
            % (m["unc_nll"], m["unc_ause"], m["unc_aurg"], m["unc_ence"],
               m["unc_cv"], m["unc_spearman"], m["unc_picp90"]))


def agg(per):
    """Mean and population std across seeds, over the scalar keys.

    Uses the UNION of keys and averages over whichever seeds actually have
    each one. Taking the key list from the first seed instead would silently
    drop any metric added partway through a run -- the earlier seeds lack it,
    so it would never appear even though later seeds measured it.
    """
    keys = []
    for m in per.values():
        for k, v in m.items():
            if isinstance(v, (int, float)) and k not in keys:
                keys.append(k)
    mean, std = {}, {}
    for k in keys:
        vals = [per[s][k] for s in per
                if isinstance(per[s].get(k), (int, float))]
        if not vals:
            continue
        mean[k] = statistics.fmean(vals)
        std[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[q[0] for q in QUEUE])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--mu", type=float, default=0.1)
    # beta = 1.0, not Seitzer's suggested 0.5, and the reason is specific to
    # what this run has to prove. beta-NLL scales the per-pixel loss by a
    # detached b^beta. Working the derivatives at beta = 1:
    #
    #   loss      = b_det * (|e| e^-s + s + log 2)
    #   d/d mu    = sign(e)                      <- EXACTLY the L1 gradient
    #   d/d s     = b - |e|                      <- zero when b = |e|
    #
    # So the mean path trains identically to R1 while the scale head still
    # converges to the local error magnitude. R4 then differs from R1 by the
    # head and nothing else, which is what makes it an ablation.
    #
    # At beta = 0.5 the mean gradient is sign(e) * b^-0.5, about 14x L1's at
    # b ~ 0.005, and it varies spatially. Against the 1.0 gradient clip that
    # lowers the effective learning rate unevenly, so a worse PSNR could not
    # be attributed to the uncertainty head rather than to the optimisation.
    # beta = 0 (plain NLL) is worse still: a ~200x mean gradient. Both are
    # run as an explicit ablation (R4b0, R4b05) to show the pathology.
    ap.add_argument("--beta-nll", type=float, default=1.0)
    ap.add_argument("--d-filter", default="lanczos3")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=100)
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    have = set(available_sites())

    st = load_state()
    st["meta"]["config"] = vars(a)
    st["meta"]["filters"] = response_table()
    st["meta"]["sites"] = {s: site_stats(s) for s in sorted(have)}

    multi = [s for s in BASE_TRAIN + EXTRA_TRAIN if s in have]
    base = [s for s in BASE_TRAIN if s in have]
    st["meta"]["train_base"] = base
    st["meta"]["train_multi"] = multi
    st["meta"]["test"] = TEST_SITE
    save_state(st)

    log("device %s" % dev)
    log("base train  %s" % base)
    log("multi train %s" % multi)
    log("test        %s  (sealed)" % TEST_SITE)

    # Loader cache: building a SRPatches memmaps every site, so don't redo it
    # per run. Keyed by (sites, augmentation).
    lcache = {}

    def get_loaders(sites, aug_kw):
        key = (tuple(sites), tuple(sorted(aug_kw.items())))
        if key not in lcache:
            lcache[key] = loaders(list(sites), TEST_SITE, batch=a.batch,
                                  crop_lr=a.crop, seed=0,
                                  aug=AugConfig(**aug_kw))
        return lcache[key]

    # ---- R0, the non-learned floor. No training, so always cheap. --------
    if "R0" not in st["runs"]:
        log("\n=== R0  Bicubic x2 (no training) ===")
        tr, va, te = get_loaders(base, AUG_OFF)
        m_in = evaluate(bicubic_predict, va, dev, filters=FILTERS)
        m_oo = evaluate(bicubic_predict, te, dev, filters=FILTERS)
        log("   in-domain  " + fmt(m_in))
        log("   ood        " + fmt(m_oo))
        st["runs"]["R0"] = {"label": "Bicubic x2", "multi": False,
                            "aug": AUG_OFF,
                            "in": {"-": m_in}, "ood": {"-": m_oo},
                            "in_mean": m_in, "ood_mean": m_oo,
                            "in_std": {k: 0.0 for k in m_in
                                       if isinstance(m_in[k], (int, float))},
                            "ood_std": {k: 0.0 for k in m_oo
                                        if isinstance(m_oo[k], (int, float))}}
        save_state(st)

    # ---- the learned queue ----------------------------------------------
    full = [(r, k, m, aw, b, None, None, False) for r, k, m, aw, b in QUEUE] + [
        (r, k, m, (AUG_ON if aw == "ON" else AUG_OFF), b, ar, mu, res)
        for r, k, m, aw, b, ar, mu, res in QUEUE2]

    for (rid, kind, use_multi, aug_kw, beta_override, arch, mu_over,
         residual) in full:
        if rid not in a.runs:
            continue
        sites = multi if use_multi else base
        if use_multi and len(multi) <= len(base):
            log("\n[skip] %s needs extra biome sites; only have %s"
                % (rid, base))
            continue

        beta = a.beta_nll if beta_override is None else beta_override
        mu_w = a.mu if mu_over is None else mu_over
        spec = LossSpec(kind, lam=a.lam, mu=mu_w, d_filter=a.d_filter,
                        beta_nll=beta)
        rec = st["runs"].get(rid) or {
            "label": spec.label() + (" [%s]" % arch if arch else ""),
            "multi": use_multi, "aug": aug_kw,
            "arch": arch or ("srnet_res" if residual else "srnet"),
            "train_sites": sites, "beta_nll": beta, "mu": mu_w,
            "in": {}, "ood": {}}
        # keep whatever seeds already finished
        done = set(rec["ood"].keys())
        # The beta ablation needs one seed: the effect it demonstrates is a
        # large systematic one, and spending 3 seeds there would delay rows
        # that actually carry claims. Flagged as 1 seed in the write-up.
        want = a.seeds[:1] if rid in SINGLE_SEED else a.seeds
        todo = [s for s in want if str(s) not in done]
        log("\n=== %s  %s%s ===" % (rid, spec.label(),
                                    ("  arch=" + arch) if arch else ""))
        log("    sites %s | aug %s | seeds done %s todo %s"
            % (sites, AugConfig(**aug_kw).describe(), sorted(done), todo))
        if not todo:
            log("    already complete, skipping")
            continue

        tr, va, te = get_loaders(sites, aug_kw)
        log("    train %d  val %d  test %d patches"
            % (len(tr.dataset), len(va.dataset), len(te.dataset)))

        for sd in todo:
            t0 = time.time()
            log("  seed %d" % sd)
            try:
                model, hist = train_one(
                    spec, tr, va, dev, steps=a.steps, seed=sd,
                    blocks=a.blocks, channels=a.channels,
                    probe_every=a.probe_every, val_every=a.val_every,
                    arch=arch, residual=residual)
            except Exception:
                log("  !! seed %d FAILED\n%s" % (sd, traceback.format_exc()))
                continue
            model.eval()

            def predict(x, m=model):
                return m(x)

            unc = spec.needs_sigma
            m_in = evaluate(predict, va, dev, filters=FILTERS,
                            with_uncertainty=unc)
            m_oo = evaluate(predict, te, dev, filters=FILTERS,
                            with_uncertainty=unc)

            # Does predicted sigma separate in-domain from the held-out
            # biome? This is the headline: a confidence head that fires on
            # the site where the model is known to fail turns the midsem
            # negative result into a usable safeguard.
            if unc:
                m_oo["ood_auroc"] = auroc(m_in.pop("patch_sigma"),
                                          m_oo.pop("patch_sigma"))
                m_in.pop("patch_sigma", None)
                log("   OOD AUROC from mean sigma: %.4f" % m_oo["ood_auroc"])

            log("   IN   " + fmt(m_in) + ("\n" + fmt_unc(m_in) if unc else ""))
            log("   OOD  " + fmt(m_oo) + ("\n" + fmt_unc(m_oo) if unc else ""))
            log("   (%.0fs)" % (time.time() - t0))

            rec["in"][str(sd)] = m_in
            rec["ood"][str(sd)] = m_oo
            st["history"]["%s_s%d" % (rid, sd)] = hist
            torch.save(model.state_dict(),
                       os.path.join(CKPT, "%s_s%d.pt" % (rid, sd)))

            # filter sweep on seed 0 only: it is a property of the metric,
            # not of the seed, so three copies would just cost time
            if sd == a.seeds[0]:
                st["sweeps"][rid] = {
                    "in": filter_sweep(predict, va, dev),
                    "ood": filter_sweep(predict, te, dev)}

            rec["in_mean"], rec["in_std"] = agg(rec["in"])
            rec["ood_mean"], rec["ood_std"] = agg(rec["ood"])
            st["runs"][rid] = rec
            save_state(st)

        log("   MEAN IN   " + fmt(rec["in_mean"]))
        log("   MEAN OOD  " + fmt(rec["ood_mean"]))

    save_state(st)
    log("\nwrote %s" % OUT_JSON)
    log("QUEUE_DONE")


if __name__ == "__main__":
    main()
