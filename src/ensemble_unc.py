r"""Epistemic uncertainty from the seeds we already trained, for free.

The R4 head predicts ALEATORIC uncertainty: noise inherent in the data,
which more training data would not remove. A deep ensemble measures
EPISTEMIC uncertainty instead: where the members disagree, the model does
not know, and more data WOULD help.

We already have three independently seeded R1 models, so the ensemble costs
nothing beyond inference. That makes this a genuinely free comparison, and
it sharpens the domain-shift claim: on an unseen biome the problem has not
become noisier, the model has left the region it knows, so it is the
EPISTEMIC term that should rise.

    Var_total = mean_k( 2 b_k^2 )  +  Var_k( mu_k )
                \___aleatoric___/     \__epistemic__/

MEASURED RESULT: it does not work here, and why is the interesting part.

Three R1 seeds disagree by sigma ~0.00053, about 25x SMALLER than the head's
predicted sigma, and that disagreement does not rise on the held-out biome
at all (ratio 1.00). Its AUROC for domain shift is 0.35, i.e. below chance,
against 0.75 for the learned head.

The explanation: all three seeds trained on the same data with the same
architecture, so they learned the SAME brightness prior. On rainforest they
are confidently and identically wrong. Ensemble disagreement can only see
what the members disagree ABOUT, and a shared systematic bias is invisible
to it by construction. The below-chance AUROC follows too -- disagreement
tracks scene complexity, and dense rainforest canopy is more homogeneous
than European farmland, so the members agree MORE there.

The honest conclusion is that a 3-member ensemble is not a substitute for a
calibrated head on this problem, which is worth saying because "just use an
ensemble" is the obvious free alternative an examiner will raise.

Writes results/ensemble_unc.json.
"""
import argparse
import json
import os

import torch

from data import loaders
from metrics import psnr, ssim
from model import SRNet
from uncertainty import auroc, ause, ause_per_band, aurg, spearman

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(RESULTS, "ckpt")


@torch.no_grad()
def run(models, loader, dev, cap=60):
    """Ensemble mean, epistemic spread, and how well that spread ranks error."""
    errs, epis, patch, e4, s4 = [], [], [], [], []
    n = 0
    tot_p = tot_s = 0.0
    seen = 0
    for lr, hr in loader:
        lr, hr = lr.to(dev), hr.to(dev)
        mus = torch.stack([m(lr).clamp(0, 1) for m in models], 0)
        mean = mus.mean(0)
        epi = mus.var(0, unbiased=False)            # disagreement
        sig = epi.clamp_min(0).sqrt()
        b = lr.shape[0]
        tot_p += psnr(mean, hr) * b
        tot_s += ssim(mean, hr) * b
        seen += b
        patch += sig.mean(dim=(1, 2, 3)).cpu().tolist()
        if n < cap:
            ea = (mean - hr).abs().cpu().numpy()
            sa = sig.cpu().numpy()
            errs.append(ea.ravel())
            epis.append(sa.ravel())
            if len(e4) < 5:                 # keep the band axis for per-band
                e4.append(ea)
                s4.append(sa)
            n += b
    import numpy as np
    e = np.concatenate(errs)
    s = np.concatenate(epis)
    # Same per-band treatment the head gets, so the comparison is like for
    # like: the pooled number is inflated by between-band scale for the
    # ensemble exactly as it is for the head.
    pb, pb_list = ause_per_band(np.concatenate(e4), np.concatenate(s4))
    return {"psnr": tot_p / max(seen, 1), "ssim": tot_s / max(seen, 1),
            "ause": ause(e, s), "ause_band": pb, "ause_per_band": pb_list,
            "aurg": aurg(e, s),
            "spearman": spearman(s, e),
            "mean_epistemic_sigma": float(s.mean()),
            "patch_sigma": patch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="R1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train-sites", nargs="+",
                    default=["ESTUAMAR", "SUDOUE-4"])
    ap.add_argument("--test-sites", nargs="+", default=["FGMANAUS"])
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for sd in a.seeds:
        p = os.path.join(CKPT, "%s_s%d.pt" % (a.run, sd))
        if not os.path.exists(p):
            continue
        m = SRNet().to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
        m.eval()
        models.append(m)
    if len(models) < 2:
        print("need at least 2 seeds, found %d" % len(models))
        return
    print("ensemble of %d %s models" % (len(models), a.run))

    _, va, te = loaders(a.train_sites, a.test_sites, batch=8, crop_lr=64,
                        seed=0)
    out = {"run": a.run, "n_members": len(models)}
    for key, loader in (("in", va), ("ood", te)):
        r = run(models, loader, dev)
        out[key] = r
        print("  %-4s PSNR %.3f  AUSE %.4f  AURG %+.4f  rho %.3f  "
              "mean epistemic sigma %.6f"
              % (key, r["psnr"], r["ause"], r["aurg"], r["spearman"],
                 r["mean_epistemic_sigma"]))

    if "in" in out and "ood" in out:
        out["ood_auroc"] = auroc(out["in"].pop("patch_sigma"),
                                 out["ood"].pop("patch_sigma"))
        ri = out["ood"]["mean_epistemic_sigma"] / \
            max(out["in"]["mean_epistemic_sigma"], 1e-12)
        out["ood_sigma_ratio"] = ri
        print("  OOD AUROC from ensemble disagreement: %.4f" % out["ood_auroc"])
        print("  epistemic sigma rises x%.2f on the held-out biome" % ri)

    p = os.path.join(RESULTS, "ensemble_unc.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
