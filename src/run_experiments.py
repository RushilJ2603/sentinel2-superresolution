"""Run the mid-sem experiments and write results/results.json.

    R0   Bicubic x2        no training, the floor
    R1   CNN, L1           baseline
    R2   CNN, L1 + lam*SAM tests what the spectral term does

Protocol: the test site is never seen in training. A random 10% of the
training patches is the validation split used for early stopping. R1 and R2
run across several seeds, because that comparison carries the whole claim
and a difference smaller than the seed spread is not a difference.
"""
import argparse
import json
import os
import statistics
import time

import torch

from data import available_sites, loaders
from losses import LossSpec
from metrics import bicubic_predict, evaluate
from train import train_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(RESULTS, "ckpt")

RUNS = {"R1": ("l1", "CNN, L1"),
        "R2": ("l1_sam", "CNN, L1 + lam*SAM")}


def fmt(m):
    return ("PSNR %.3f  SSIM %.4f  SAM %.3f  NDVI %.5f  CONSIST %.5f"
            % (m["psnr"], m["ssim"], m["sam"], m["ndvi_mae"], m["consist"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sites", nargs="+", required=True)
    ap.add_argument("--test-sites", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", default=["R1", "R2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(RESULTS, "results.json"))
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    have = available_sites()
    for s in a.train_sites + a.test_sites:
        assert s in have, "%s not cached (have %s)" % (s, have)
    assert not (set(a.train_sites) & set(a.test_sites)), \
        "train and test sites must be disjoint"

    tr, va, te = loaders(a.train_sites, a.test_sites, batch=a.batch,
                         crop_lr=a.crop, seed=0)
    print("device %s | train %d  val %d  test %d patches"
          % (dev, len(tr.dataset), len(va.dataset), len(te.dataset)))
    print("train %s -> test %s" % (a.train_sites, a.test_sites), flush=True)

    out = {"config": vars(a), "device": dev, "n_train": len(tr.dataset),
           "n_val": len(va.dataset), "n_test": len(te.dataset),
           "runs": {}, "history": {}}

    print("\n=== R0  Bicubic x2 (no training) ===", flush=True)
    m = evaluate(bicubic_predict, te, dev)
    print("   " + fmt(m), flush=True)
    out["runs"]["R0"] = {"label": "Bicubic x2", "seeds": {"-": m},
                         "mean": m, "std": {k: 0.0 for k in m}}

    for rid in a.runs:
        kind, label = RUNS[rid]
        spec = LossSpec(kind, lam=a.lam)
        per = {}
        print("\n=== %s  %s  (seeds %s) ===" % (rid, label, a.seeds), flush=True)
        for sd in a.seeds:
            t0 = time.time()
            print(" seed %d" % sd, flush=True)
            model, hist = train_one(
                spec, tr, va, dev, steps=a.steps, seed=sd,
                blocks=a.blocks, channels=a.channels,
                probe_every=a.probe_every, val_every=a.val_every)
            model.eval()
            m = evaluate(model, te, dev)
            print("   TEST  " + fmt(m) + "   (%.0fs)" % (time.time() - t0),
                  flush=True)
            per[str(sd)] = m
            out["history"]["%s_s%d" % (rid, sd)] = hist
            torch.save(model.state_dict(),
                       os.path.join(CKPT, "%s_s%d.pt" % (rid, sd)))
            with open(a.out, "w") as f:
                json.dump(out, f, indent=2)
        keys = list(next(iter(per.values())).keys())
        mean = {k: statistics.fmean(per[s][k] for s in per) for k in keys}
        std = {k: (statistics.pstdev([per[s][k] for s in per])
                   if len(per) > 1 else 0.0) for k in keys}
        out["runs"][rid] = {"label": label, "seeds": per,
                            "mean": mean, "std": std}
        print("   MEAN  " + fmt(mean), flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", a.out)
    print("EXPERIMENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
