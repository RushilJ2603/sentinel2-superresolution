"""Which pi/2 anti-alias filter best models the real Sentinel-2 instrument?

This is the empirical half of the filter study, and it needs no trained
model: apply each candidate D to the real VENuS capture and compare against
the real Sentinel-2 capture of the same ground on the same date.

    best D  =  argmin  |D(HR) - LR|

One honest caveat, stated before anyone raises it: the two sides come from
DIFFERENT instruments, so D(HR) can never equal LR exactly. The residual
floor measured here (~0.0066 on ESTUAMAR) is cross-sensor disagreement, not
filter error. What the ranking is entitled to claim is which filter gets
CLOSEST, not that any of them is correct in absolute terms.

Writes results/filter_match.json.
"""
import argparse
import json
import os

import torch

from data import SRPatches, available_sites
from degrade import FILTERS, verify_phase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=["ESTUAMAR", "FGMANAUS"])
    ap.add_argument("--patches", type=int, default=60)
    a = ap.parse_args()

    have = available_sites()
    out = {"n_patches": a.patches}
    for site in a.sites:
        if site not in have:
            print("[skip] %s not cached" % site)
            continue
        ds = SRPatches([site], crop_lr=0, train=False)
        n = min(a.patches, len(ds))
        lo = torch.stack([ds[i][0] for i in range(n)])
        hi = torch.stack([ds[i][1] for i in range(n)])
        r = verify_phase(hi, lo)
        out[site] = r
        print("=== %s  (%d patches) ===" % (site, n))
        for k, v in sorted(r.items(), key=lambda t: t[1]):
            tag = ""
            if k == "box2":
                tag = "   <- the mid-sem operator"
            print("   %-14s |D(HR)-LR| = %.6f%s" % (k, v, tag))
        best = min(r, key=r.get)
        print("   best: %s, %.1f%% better than box2\n"
              % (best, 100 * (r["box2"] - r[best]) / r["box2"]))

    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "filter_match.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
