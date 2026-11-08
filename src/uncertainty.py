"""Predictive uncertainty: the Laplace head, and the metrics that score it.

Why a Laplace head and not a Gaussian one

This is the single most defensible design choice in the file, so it is worth
stating precisely.

Training with L1 loss is exactly maximum likelihood under a Laplace
observation model with a FIXED scale:

    Laplace(y; mu, b)  =  (1 / 2b) * exp(-|y - mu| / b)
    -log p             =  |y - mu| / b  +  log 2b

Hold b constant and minimising -log p is minimising |y - mu|, i.e. L1. So
letting the network predict b per pixel is not a new objective bolted on
beside L1 -- it is the SAME objective with one assumption relaxed. R4 differs
from R1 by exactly "b is predicted instead of constant", which is what makes
it a clean ablation rather than a different system.

A Gaussian head would generalise L2 instead, and L2 was rejected earlier in
this project for producing blurry means. Choosing Gaussian here would
quietly reintroduce the loss already argued against.

The pathology, and the fix

Heteroscedastic NLL has a known failure mode. The gradient of the NLL with
respect to mu carries a factor 1/b, so pixels the model has decided are
uncertain receive a down-weighted mean gradient. Hard pixels get labelled
uncertain, then stop being learned, which makes them stay hard. The mean
prediction can end up worse than a plain L1 model.

Seitzer et al. (ICLR 2022), "On the Pitfalls of Heteroscedastic Uncertainty
Estimation with Probabilistic Neural Networks", diagnose this and propose
beta-NLL: multiply the per-pixel loss by a DETACHED b^beta, which restores
the mean gradient scale while leaving the b gradient intact. beta = 0 is
plain NLL, beta = 1 fully restores L1-like mean gradients, beta = 0.5 is the
recommended middle. We expose beta and report it, because an uncertainty
head that silently damages PSNR would invalidate the comparison against R1.

What "metricize" means here

A confidence map is only worth having if it is both CALIBRATED (the stated
uncertainty matches the observed error) and INFORMATIVE (it varies, and the
variation tracks where the error actually is). Those are different
properties and need different numbers, so:

  NLL     proper scoring rule; jointly scores mean and spread
  AUSE    does ranking pixels by predicted sigma match ranking by true error
  AURG    how much better than a random ranking (sanity companion to AUSE)
  ENCE    binned calibration: is predicted spread the observed spread
  Cv      coefficient of variation of sigma -- is the map informative at all
  PICP    do the stated 50/90/95% intervals actually cover that often
  rho     Spearman correlation between sigma and |error|

The ENCE/Cv pairing is the trap worth naming out loud: a model that predicts
one constant sigma everywhere can score a near-perfect ENCE while being
completely useless, because it tells you nothing about WHERE the error is.
Cv catches exactly that, which is why Levi et al. (2020) report the two
together and why we do too.

References

Kendall & Gal (NeurIPS 2017), aleatoric/epistemic uncertainty in vision.
Lakshminarayanan et al. (NeurIPS 2017), deep ensembles.
Ilg et al. (ECCV 2018), sparsification error / AUSE for dense regression.
Levi et al. (2020), ENCE and the coefficient of variation.
Seitzer et al. (ICLR 2022), beta-NLL.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

LOG_B_MIN, LOG_B_MAX = -9.0, 2.0     # b in [1.2e-4, 7.4]; reflectance is [0,1]
SQRT2 = math.sqrt(2.0)
# the loss
def laplace_nll(pred, log_b, target, beta=0.0, reduce=True):
    """Negative log-likelihood of `target` under Laplace(pred, exp(log_b)).

        -log p = |y - mu| * exp(-s) + s + log 2,      s = log b

    `beta` applies the beta-NLL correction of Seitzer et al. (2022): the
    per-pixel loss is scaled by a DETACHED b^beta, which cancels the 1/b
    factor in the mean gradient without touching the scale gradient.
    beta=0 is plain NLL, beta=1 restores L1-scale mean gradients.

    The log 2 is kept so the number is a true NLL and comparable across
    papers; it is a constant and does not affect gradients.
    """
    s = log_b.clamp(LOG_B_MIN, LOG_B_MAX)
    nll = (target - pred).abs() * torch.exp(-s) + s + math.log(2.0)
    if beta:
        nll = nll * torch.exp(s * beta).detach()
    return nll.mean() if reduce else nll


def gaussian_nll(pred, log_var, target, beta=0.0, reduce=True):
    """Gaussian NLL, kept only as the ablation that shows why we chose Laplace."""
    v = log_var.clamp(2 * LOG_B_MIN, 2 * LOG_B_MAX)
    nll = 0.5 * ((target - pred) ** 2 * torch.exp(-v) + v + math.log(2 * math.pi))
    if beta:
        nll = nll * torch.exp(v * beta).detach()
    return nll.mean() if reduce else nll


def laplace_sigma(log_b):
    """Standard deviation of Laplace(mu, b) is sqrt(2) * b.

    Needed because ENCE compares a predicted spread against an observed
    RMSE, and both have to be second-moment quantities. Reporting b where
    sigma is meant understates the spread by 41% and breaks calibration.
    """
    return SQRT2 * torch.exp(log_b.clamp(LOG_B_MIN, LOG_B_MAX))


def laplace_interval_halfwidth(log_b, q):
    """Half-width of the central q-mass interval of a Laplace.

    P(|y - mu| <= t) = 1 - exp(-t / b)  =>  t = -b * ln(1 - q)

    So the 90% interval is mu +- 2.3026 b, NOT mu +- 1.645 sigma. Using the
    Gaussian z-score on a Laplace is a real and easy mistake: it would make
    a well-calibrated model look over-confident at 90% and under-confident
    at 50%.
    """
    b = torch.exp(log_b.clamp(LOG_B_MIN, LOG_B_MAX))
    return -b * math.log(1.0 - q)
# metrics
def _flat(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _subsample(n, cap, seed=0):
    """Pixel indices, capped. AUSE needs a full sort; a held-out site is ~34M
    pixels, so we subsample deterministically rather than silently using
    whatever fits. Reported alongside the metric."""
    if cap is None or n <= cap:
        return None
    return np.random.RandomState(seed).choice(n, cap, replace=False)


def sparsification(err, unc, n_steps=100):
    """Sparsification, oracle and random curves, all normalised to 1 at p=0.

    Remove the most-uncertain fraction p of pixels and report the mean error
    of what remains. If sigma ranks pixels the way true error does, our curve
    sits on the oracle. A flat curve at 1.0 means sigma carries no ranking
    information at all.
    """
    err = _flat(err)
    unc = _flat(unc)
    o_unc = np.argsort(-unc, kind="stable")      # most uncertain first
    o_err = np.argsort(-err, kind="stable")
    e_by_unc = err[o_unc]
    e_by_err = err[o_err]
    # mean of the retained tail, for each removal fraction
    csum_u = np.cumsum(e_by_unc)
    csum_o = np.cumsum(e_by_err)
    total = csum_u[-1]
    n = len(err)
    fracs = np.linspace(0.0, 0.99, n_steps)
    cut = (fracs * n).astype(int)
    keep = n - cut
    ours = (total - np.where(cut > 0, csum_u[cut - 1], 0.0)) / keep
    orac = (total - np.where(cut > 0, csum_o[cut - 1], 0.0)) / keep
    base = ours[0]
    if base <= 0:
        base = 1.0
    return fracs, ours / base, orac / base, np.ones_like(fracs)


def ause(err, unc, n_steps=100):
    """Area under the sparsification ERROR curve (ours - oracle). Lower better,
    0 means sigma ranks pixels exactly as well as true error does."""
    f, ours, orac, _ = sparsification(err, unc, n_steps)
    return float(np.trapezoid(ours - orac, f)) if hasattr(np, "trapezoid") \
        else float(np.trapz(ours - orac, f))


def aurg(err, unc, n_steps=100):
    """Area under the random GAIN curve (random - ours). Higher is better.

    Companion to AUSE. AUSE alone can look good simply because the oracle
    curve is shallow on easy data; AURG says whether we beat doing nothing.
    A negative AURG means the uncertainty map is worse than random ranking.
    """
    f, ours, _, rand = sparsification(err, unc, n_steps)
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(tz(rand - ours, f))


def ause_per_band(err, unc, n_steps=100):
    """AUSE computed inside each band, then averaged. Report this WITH the
    pooled number, not instead of it.

    Pooling all four bands into one ranking measures something largely
    trivial. The bands have very different signal levels (NIR reflectance
    ~0.28 against blue ~0.009), so NIR errors are an order of magnitude
    larger and the model correctly assigns them larger sigma. Measured on
    our own head: 96.9% of NIR pixels land in the global top-25% of sigma,
    while the three visible bands contribute about 1% between them. So the
    first quarter of the pooled sparsification curve is essentially
    "remove all the NIR", which is why that curve has a visible kink at
    0.25, and the pooled AUSE is dominated by between-band scale rather
    than by whether sigma ranks pixels usefully WITHIN a band.

    Ranking inside each band separately removes that confound and asks the
    question we actually care about.

    `err` and `unc` must be (N, C, H, W) or (C, ...) with the band axis
    second-to-outermost; both are flattened per band.
    """
    e = np.asarray(err, dtype=np.float64)
    u = np.asarray(unc, dtype=np.float64)
    if e.ndim < 2:
        raise ValueError("need a band axis for a per-band AUSE")
    c_ax = 1 if e.ndim >= 4 else 0
    n_c = e.shape[c_ax]
    out = []
    for c in range(n_c):
        ec = np.take(e, c, axis=c_ax).reshape(-1)
        uc = np.take(u, c, axis=c_ax).reshape(-1)
        out.append(ause(ec, uc, n_steps))
    return float(np.mean(out)), out


def ence(err_sq, sigma, n_bins=15):
    """Expected Normalised Calibration Error (Levi et al. 2020).

    Bin pixels by predicted sigma into equal-count bins. In each bin compare
    the root-mean-variance RMV = sqrt(mean sigma^2) against the observed
    RMSE, and average the relative gap:

        ENCE = (1/K) sum_k |RMV_k - RMSE_k| / RMV_k

    Equal-count (not equal-width) bins, because sigma is heavily skewed and
    equal-width bins would leave most bins nearly empty.
    """
    e = _flat(err_sq)
    s = _flat(sigma)
    order = np.argsort(s, kind="stable")
    e, s = e[order], s[order]
    edges = np.linspace(0, len(s), n_bins + 1).astype(int)
    tot, used = 0.0, 0
    rows = []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b - a < 2:
            continue
        rmv = math.sqrt(float((s[a:b] ** 2).mean()))
        rmse = math.sqrt(float(e[a:b].mean()))
        if rmv <= 0:
            continue
        tot += abs(rmv - rmse) / rmv
        used += 1
        # plain floats/ints: these rows get written to results JSON, and a
        # numpy int64 from the bin edges is not JSON serialisable
        rows.append((float(rmv), float(rmse), int(b - a)))
    return (tot / max(used, 1)), rows


def coeff_variation(sigma):
    """Cv = std(sigma) / mean(sigma).

    Measures whether the confidence map says anything at all. A constant map
    has Cv = 0 and can still score a perfect ENCE, which is why this is
    always reported next to it.
    """
    s = _flat(sigma)
    m = s.mean()
    return float(s.std() / m) if m > 0 else 0.0


def picp(err_abs, log_b, levels=(0.5, 0.9, 0.95)):
    """Prediction Interval Coverage Probability at nominal Laplace levels.

    Uses the Laplace quantile t = -b ln(1-q), not a Gaussian z-score.
    Perfect calibration puts the measured coverage on the nominal level.
    """
    e = _flat(err_abs)
    b = np.exp(np.clip(_flat(log_b), LOG_B_MIN, LOG_B_MAX))
    out = {}
    for q in levels:
        t = -b * math.log(1.0 - q)
        out[q] = float((e <= t).mean())
    return out


def spearman(a, b):
    """Spearman rank correlation between predicted sigma and |error|.

    The most intuitive number on the slide: does the map point at the errors.
    Computed as Pearson on ranks; ties get average ranks.
    """
    x, y = _flat(a), _flat(b)

    def rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=np.float64)
        r[o] = np.arange(len(v), dtype=np.float64)
        # average ranks within tie groups
        sv = v[o]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def auroc(scores_in, scores_out):
    """AUROC for separating in-domain from out-of-domain using a scalar score.

    Computed via the Mann-Whitney U identity, so it needs no threshold sweep:

        AUROC = P(score_out > score_in)

    This is the number behind the claim "the confidence head detects the
    biome shift". 0.5 is chance; 1.0 is perfect separation.
    """
    a, b = _flat(scores_in), _flat(scores_out)
    allv = np.concatenate([a, b])
    o = np.argsort(allv, kind="stable")
    r = np.empty(len(allv), dtype=np.float64)
    r[o] = np.arange(1, len(allv) + 1, dtype=np.float64)
    sv = allv[o]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (r[o[i]] + r[o[j]]) / 2.0
        i = j + 1
    n_in, n_out = len(a), len(b)
    if n_in == 0 or n_out == 0:
        return float("nan")
    u = r[n_in:].sum() - n_out * (n_out + 1) / 2.0
    return float(u / (n_in * n_out))
# the whole suite in one call
def uncertainty_report(pred, log_b, target, n_bins=15, cap=4_000_000, seed=0):
    """Every uncertainty metric, from tensors on any device.

    `cap` bounds the pixel count used by the rank-based metrics (AUSE, AURG,
    Spearman) because they need a full sort. The cap and the effective pixel
    count are returned so the report can state them.
    """
    with torch.no_grad():
        p = pred.detach().float().cpu()
        s = log_b.detach().float().cpu().clamp(LOG_B_MIN, LOG_B_MAX)
        t = target.detach().float().cpu()
        nll = laplace_nll(p, s, t).item()
        e_abs = (p - t).abs().reshape(-1).numpy()
        sig = laplace_sigma(s).reshape(-1).numpy()
        lb = s.reshape(-1).numpy()

    n = len(e_abs)
    idx = _subsample(n, cap, seed)
    ea, sg = (e_abs[idx], sig[idx]) if idx is not None else (e_abs, sig)

    # Per-band AUSE, on a patch subset so the sort stays cheap. This is the
    # confound-free version; see ause_per_band for why the pooled number
    # mostly reflects NIR being brighter than the visible bands.
    pb = pb_list = None
    if p.dim() == 4:
        k = max(1, min(p.shape[0], 40))
        with torch.no_grad():
            e4 = (p[:k] - t[:k]).abs().numpy()
            s4 = laplace_sigma(s[:k]).numpy()
        pb, pb_list = ause_per_band(e4, s4)

    en, rows = ence(e_abs ** 2, sig, n_bins=n_bins)
    cov = picp(e_abs, lb)
    return {
        "nll": nll,
        "ause": ause(ea, sg),
        "ause_band": pb,
        "ause_per_band": pb_list,
        "aurg": aurg(ea, sg),
        "ence": en,
        "cv": coeff_variation(sig),
        "spearman": spearman(sg, ea),
        "picp50": cov[0.5],
        "picp90": cov[0.9],
        "picp95": cov[0.95],
        "mean_sigma": float(sig.mean()),
        "mean_abs_err": float(e_abs.mean()),
        "n_pixels": int(n),
        "n_ranked": int(len(ea)),
        "calib_bins": rows,
    }


def ensemble_moments(mus, log_bs=None):
    r"""Combine K member predictions into mean, aleatoric and epistemic spread.

    For a uniform mixture of Laplace(mu_k, b_k) the law of total variance gives

        Var_total = mean_k(2 b_k^2)  +  Var_k(mu_k)
                    \_aleatoric_/      \_epistemic_/

    Aleatoric is noise the data itself carries and more data will not remove.
    Epistemic is the members disagreeing, and it is the part that should rise
    on a biome the model has never seen. Separating them is what makes the
    domain-shift claim mechanistic rather than just "the number went up".
    """
    mu = torch.stack(mus, 0)
    mean = mu.mean(0)
    epi = mu.var(0, unbiased=False)
    if log_bs is None:
        return mean, torch.zeros_like(epi), epi
    b = torch.stack([torch.exp(s.clamp(LOG_B_MIN, LOG_B_MAX)) for s in log_bs], 0)
    ale = (2.0 * b ** 2).mean(0)
    return mean, ale, epi
# self-test
def _self_test():
    torch.manual_seed(0)
    np.random.seed(0)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  [ok] " if cond else "  [FAIL] ") + name +
              ("   " + detail if detail else ""))

    print("Laplace NLL")
    # With b fixed, NLL must reduce to L1 up to the constant log 2b.
    p = torch.rand(2, 4, 16, 16)
    t = torch.rand(2, 4, 16, 16)
    s = torch.zeros_like(p)                       # b = 1
    chk("reduces to L1 when b=1",
        abs(laplace_nll(p, s, t).item()
            - (F.l1_loss(p, t).item() + math.log(2.0))) < 1e-6)

    # The NLL is minimised, for fixed residuals, at b = mean|y-mu|.
    res = 0.1
    p2 = torch.zeros(1, 1, 64, 64)
    t2 = torch.full_like(p2, res)
    grid = torch.linspace(-4, 1, 201)
    vals = [laplace_nll(p2, torch.full_like(p2, float(g)), t2).item() for g in grid]
    best = float(grid[int(np.argmin(vals))])
    chk("optimal log b equals log(mean|err|)",
        abs(best - math.log(res)) < 0.05,
        "got %.3f want %.3f" % (best, math.log(res)))

    # sigma and interval relations
    chk("sigma = sqrt(2) b",
        abs(laplace_sigma(torch.zeros(1)).item() - SQRT2) < 1e-6)
    chk("90%% halfwidth = 2.3026 b",
        abs(laplace_interval_halfwidth(torch.zeros(1), 0.9).item() - 2.302585) < 1e-5)

    # beta-NLL must not change the loss VALUE at beta=0 and must rescale at beta>0
    v0 = laplace_nll(p, s, t, beta=0.0).item()
    v1 = laplace_nll(p, s, t, beta=1.0).item()
    chk("beta=0 is plain NLL", abs(v0 - v1) < 1e-6)   # b=1 so b^beta=1

    print("\nCalibration metrics on synthetic data")
    # Draw genuinely Laplace-distributed errors with a KNOWN varying scale.
    n = 400_000
    b_true = np.exp(np.random.uniform(-5, -2, n))
    u = np.random.uniform(-0.5, 0.5, n)
    err = -b_true * np.sign(u) * np.log(1 - 2 * np.abs(u))      # Laplace sample
    lb = torch.tensor(np.log(b_true))
    mu = torch.zeros(n)
    tgt = torch.tensor(err)

    r = uncertainty_report(mu, lb, tgt, cap=None)
    print("    AUSE %.4f  AURG %.4f  ENCE %.4f  Cv %.3f  rho %.3f"
          % (r["ause"], r["aurg"], r["ence"], r["cv"], r["spearman"]))
    print("    PICP  50%% %.3f   90%% %.3f   95%% %.3f"
          % (r["picp50"], r["picp90"], r["picp95"]))
    chk("PICP 50 near nominal", abs(r["picp50"] - 0.50) < 0.02)
    chk("PICP 90 near nominal", abs(r["picp90"] - 0.90) < 0.02)
    chk("PICP 95 near nominal", abs(r["picp95"] - 0.95) < 0.02)
    chk("ENCE small for a correctly specified model", r["ence"] < 0.10,
        "%.4f" % r["ence"])
    chk("Cv > 0 (map is informative)", r["cv"] > 0.3, "%.3f" % r["cv"])
    # NOTE: AUSE does NOT go to zero for a correctly specified model, and the
    # threshold here used to assume it did. The oracle curve sorts by the
    # REALISED error of each pixel; sigma can only ever predict the EXPECTED
    # error. The realised error is a random draw from Laplace(0, b), so even
    # with the true b known the ranking cannot match the oracle. That gap is
    # the aleatoric noise floor, and it is a property of the data, not a
    # defect of the model. Same reason Spearman rho tops out near 0.6 here
    # rather than at 1. So the meaningful test is against the uninformative
    # baseline, which is what `aurg` and the constant-sigma case below check.
    chk("AURG > 0 (beats random ranking)", r["aurg"] > 0.3, "%.4f" % r["aurg"])
    chk("AUSE well below the constant-sigma baseline", r["ause"] < 0.3,
        "%.4f (floor is set by aleatoric noise, not by the model)" % r["ause"])

    # A constant sigma: the ENCE/Cv trap. Must stay calibrated-ish but be
    # exposed as uninformative by Cv and by AUSE/AURG.
    lb_const = torch.full((n,), float(np.log(b_true.mean())))
    rc = uncertainty_report(mu, lb_const, tgt, cap=None)
    print("    constant-sigma model:  Cv %.3f  AUSE %.4f  AURG %.4f  rho %.3f"
          % (rc["cv"], rc["ause"], rc["aurg"], rc["spearman"]))
    chk("constant sigma has Cv = 0", rc["cv"] < 1e-9)
    chk("constant sigma is caught by AUSE", rc["ause"] > r["ause"])
    chk("constant sigma has ~zero rank correlation", abs(rc["spearman"]) < 0.02)

    # Oracle uncertainty (sigma = true error) must drive AUSE to ~0.
    lb_or = torch.tensor(np.log(np.abs(err) + 1e-12))
    ro = uncertainty_report(mu, lb_or, tgt, cap=None)
    chk("oracle sigma gives AUSE ~ 0", ro["ause"] < 1e-6, "%.2e" % ro["ause"])
    chk("oracle sigma gives rho ~ 1", ro["spearman"] > 0.999,
        "%.4f" % ro["spearman"])

    print("\nAUROC and ensemble")
    chk("AUROC identical distributions ~ 0.5",
        abs(auroc(np.random.randn(5000), np.random.randn(5000)) - 0.5) < 0.03)
    chk("AUROC separated distributions ~ 1.0",
        auroc(np.random.randn(2000), np.random.randn(2000) + 12) > 0.99)

    mus = [torch.full((1, 1, 4, 4), v) for v in (0.1, 0.2, 0.3)]
    lbs = [torch.zeros(1, 1, 4, 4) for _ in range(3)]
    m, ale, epi = ensemble_moments(mus, lbs)
    chk("ensemble mean", abs(m.mean().item() - 0.2) < 1e-6)
    chk("aleatoric = 2b^2 = 2", abs(ale.mean().item() - 2.0) < 1e-6)
    chk("epistemic = var of members",
        abs(epi.mean().item() - float(np.var([0.1, 0.2, 0.3]))) < 1e-6)

    print("\n" + ("ALL OK" if ok else "FAILURES PRESENT"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
