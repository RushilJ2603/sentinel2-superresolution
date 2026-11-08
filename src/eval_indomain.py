# Re-score every saved checkpoint on the in-domain validation split as well
# as the held-out test site, and fold both into results.json.
#
# Why bother: FGMANAUS turned out to be a different biome from the two
# training sites (rainforest vs European coast and farmland), so the test
# numbers measure transfer across land cover, not super-resolution quality.
# Both questions are worth answering, and the same six runs answer them.
#
# Honest labelling matters here. The validation split also picks the
# early-stopping checkpoint, so it is NOT a clean held-out set. It says
# "does the model work at all", not "how well does it generalise".

import argparse
import json
import os
import statistics

import torch

from data import loaders
from metrics import bicubic_predict, evaluate
from model import SRNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(RESULTS, "ckpt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(RESULTS, "results.json"))
    a = ap.parse_args()
    with open(a.results) as f:
        res = json.load(f)
    cfg = res["config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, va, _ = loaders(cfg["train_sites"], cfg["test_sites"],
                       batch=cfg["batch"], crop_lr=cfg["crop"], seed=0)

    m_bic = evaluate(bicubic_predict, va, dev)
    res["runs"]["R0"]["mean_val"] = m_bic
    res["runs"]["R0"]["std_val"] = {k: 0.0 for k in m_bic}
    res["runs"]["R0"]["seeds_val"] = {"-": m_bic}
    print("R0 bicubic  (val) PSNR %.3f SSIM %.4f SAM %.3f NDVI %.5f CONSIST %.5f"
          % tuple(m_bic[k] for k in ("psnr", "ssim", "sam", "ndvi_mae", "consist")))

    for rid in ("R1", "R2"):
        if rid not in res["runs"]:
            continue
        per = {}
        for sd in res["runs"][rid]["seeds"]:
            p = os.path.join(CKPT, "%s_s%s.pt" % (rid, sd))
            if not os.path.exists(p):
                continue
            net = SRNet(in_ch=4, c=cfg["channels"], n_blocks=cfg["blocks"]).to(dev)
            net.load_state_dict(torch.load(p, map_location=dev))
            net.eval()
            per[sd] = evaluate(net, va, dev)
            print("%s seed %s (val) PSNR %.3f SSIM %.4f SAM %.3f NDVI %.5f "
                  "CONSIST %.5f" % (rid, sd, per[sd]["psnr"], per[sd]["ssim"],
                                    per[sd]["sam"], per[sd]["ndvi_mae"],
                                    per[sd]["consist"]))
        if not per:
            continue
        keys = list(next(iter(per.values())).keys())
        res["runs"][rid]["seeds_val"] = per
        res["runs"][rid]["mean_val"] = {
            k: statistics.fmean(per[s][k] for s in per) for k in keys}
        res["runs"][rid]["std_val"] = {
            k: (statistics.pstdev([per[s][k] for s in per])
                if len(per) > 1 else 0.0) for k in keys}

    with open(a.results, "w") as f:
        json.dump(res, f, indent=2)
    print("\nupdated", a.results)


if __name__ == "__main__":
    main()
