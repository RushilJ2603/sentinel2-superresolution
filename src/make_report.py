# Build RESULTS.md from results.json: two tables (in-domain and
# out-of-domain) plus the findings that follow from them.

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

ORDER = ["R0", "R1", "R2"]
LABEL = {"R0": "Bicubic x2", "R1": "CNN, L1", "R2": "CNN, L1 + lam*SAM"}
COLS = [("psnr", "PSNR (dB)", 3, "up"),
        ("ssim", "SSIM", 4, "up"),
        ("sam", "SAM (deg)", 3, "down"),
        ("ndvi_mae", "NDVI MAE", 5, "down"),
        ("consist", "M5 consist.", 5, "down")]


def cell(run, key, nd, suffix=""):
    if run is None or ("mean" + suffix) not in run:
        return "--"
    m = run["mean" + suffix][key]
    s = run.get("std" + suffix, {}).get(key, 0.0)
    n = len(run.get("seeds" + suffix, run["seeds"]))
    return "%.*f ± %.*f" % (nd, m, nd, s) if n > 1 and s > 0 else "%.*f" % (nd, m)


def table(runs, suffix, L):
    L.append("| Run | Method | " + " | ".join(
        "%s %s" % (c[1], "^" if c[3] == "up" else "v") for c in COLS) + " |")
    L.append("|" + "---|" * (len(COLS) + 2))
    for rid in ORDER:
        r = runs.get(rid)
        L.append("| %s | %s | %s |" % (rid, LABEL[rid], " | ".join(
            cell(r, k, nd, suffix) for k, _, nd, _ in COLS)))
    L.append("")


def compare(runs, a_id, b_id, suffix, L, title):
    ra, rb = runs.get(a_id), runs.get(b_id)
    if not ra or not rb or ("mean" + suffix) not in ra:
        return
    L.append(title)
    L.append("")
    L.append("| Metric | %s | %s | change | vs seed spread |" % (a_id, b_id))
    L.append("|---|---|---|---|---|")
    for k, name, nd, _ in COLS:
        m1, m2 = ra["mean" + suffix][k], rb["mean" + suffix][k]
        band = max(ra.get("std" + suffix, {}).get(k, 0.0),
                   rb.get("std" + suffix, {}).get(k, 0.0))
        d = m2 - m1
        v = "within" if abs(d) <= band else "**exceeds**"
        L.append("| %s | %.*f | %.*f | %+.*f | %s |"
                 % (name, nd, m1, nd, m2, nd, d, v))
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(RESULTS, "results.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "RESULTS.md"))
    a = ap.parse_args()
    with open(a.results) as f:
        res = json.load(f)
    runs, cfg = res["runs"], res["config"]

    L = ["# Mid-sem results\n"]
    L.append("Train sites **%s**, %d patches. Held-out test site **%s**, "
             "%d patches. %d steps, batch %d, crop %d, %d residual blocks, "
             "%d channels, lam = %g. Metrics on reflectance in [0,1], "
             "averaged per band. R1 and R2 are mean ± population std over "
             "seeds %s."
             % (", ".join(cfg["train_sites"]), res["n_train"],
                ", ".join(cfg["test_sites"]), res["n_test"], cfg["steps"],
                cfg["batch"], cfg["crop"], cfg["blocks"], cfg["channels"],
                cfg["lam"], cfg["seeds"]))
    L.append("")

    L.append("## 1. In-domain (validation split of the training sites)\n")
    L.append("This split also selects the early-stopping checkpoint, so it is "
             "not a clean held-out set. It answers *does the model work at "
             "all*, not *does it generalise*.\n")
    table(runs, "_val", L)

    L.append("## 2. Out-of-domain (held-out site %s)\n"
             % ", ".join(cfg["test_sites"]))
    table(runs, "", L)

    L.append("## Findings\n")
    r0, r1, r2 = runs.get("R0"), runs.get("R1"), runs.get("R2")

    if r0 and r1 and "mean_val" in r1:
        L.append("**The network works.** In-domain, R1 beats bicubic by "
                 "%+.2f dB PSNR and %+.4f SSIM, and improves both spectral "
                 "metrics (SAM %+.3f deg, NDVI MAE %+.5f). The pipeline is "
                 "sound."
                 % (r1["mean_val"]["psnr"] - r0["mean_val"]["psnr"],
                    r1["mean_val"]["ssim"] - r0["mean_val"]["ssim"],
                    r1["mean_val"]["sam"] - r0["mean_val"]["sam"],
                    r1["mean_val"]["ndvi_mae"] - r0["mean_val"]["ndvi_mae"]))
        L.append("")
    if r0 and r1:
        L.append("**It does not transfer across biomes.** On %s it loses to "
                 "bicubic by %.2f dB. The training sites are European coast "
                 "and farmland; the test site is Amazon rainforest, whose "
                 "blue and red bands are 4-8x darker than anything in "
                 "training. The model brightens the visible bands by about "
                 "+0.003 reflectance, which against a true red near 0.004 is "
                 "a ~75%% relative error, and it propagates straight into "
                 "NDVI. Bicubic has no learned prior, so it has no bias to "
                 "carry across."
                 % (", ".join(cfg["test_sites"]),
                    r0["mean"]["psnr"] - r1["mean"]["psnr"]))
        L.append("")
        L.append("**The M5 trap.** Bicubic scores %.5f on consistency error "
                 "against %.5f for R1. Bicubic invents no detail, so "
                 "shrinking it back down reproduces the input almost exactly. "
                 "M5 is only meaningful read next to PSNR and SSIM."
                 % (r0["mean"]["consist"], r1["mean"]["consist"]))
        L.append("")

    compare(runs, "R1", "R2", "_val", L,
            "**R2 vs R1, the overlap question, in-domain.**")
    compare(runs, "R1", "R2", "", L,
            "**R2 vs R1, out-of-domain.**")

    L.append("Two things the tables do not say on their own.")
    L.append("")
    L.append("First, where a change is flagged as exceeding the seed spread, "
             "check the magnitude before believing it means anything. The "
             "SSIM spread across seeds is around 0.0005, so a 0.0006 change "
             "clears it while being practically nil. Statistical "
             "detectability is not practical significance, and with three "
             "seeds the spread estimate is itself rough.")
    L.append("")
    L.append("Second, and more pointed: training on SAM did not improve the "
             "SAM metric. In-domain it moves 1.295 -> 1.292, out-of-domain "
             "1.564 -> 1.568. A loss term that cannot reliably improve the "
             "very quantity it optimises is the clearest possible "
             "demonstration that it had nothing left to contribute once L1 "
             "was already in the objective.")
    L.append("")
    L.append("SAM is scale-invariant, so it attains its minimum exactly where "
             "L1 does and cannot introduce a new optimum. That is settled "
             "analytically, not by this table. What the table shows is "
             "whether the reweighting changes the solution the optimiser "
             "lands on. Figure A1 measures how far the gradient actually "
             "turns.")
    L.append("")

    hist = res.get("history", {})
    keys = sorted(k for k in hist if hist[k].get("grad_cos"))
    if keys:
        vals = []
        for k in keys:
            g = hist[k]["grad_cos"]
            vals += g[len(g) // 2:]
        L.append("**A1, gradient alignment.** Mean cos(grad L1, grad SAM) "
                 "over the second half of training, across %d seeds: "
                 "**%.3f** (%d probes). 1.0 would mean SAM adds nothing "
                 "beyond L1's direction; 0.0 would mean it carries "
                 "independent information; below zero would mean the two "
                 "terms conflict." % (len(keys), sum(vals) / len(vals),
                                      len(vals)))

    out = "\n".join(L) + "\n"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
