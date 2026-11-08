# Losses for every run. R1 is plain L1; the rest add exactly one term.
#
#   R1  L1
#   R2  L1 + lam * SAM            spectral angle        (midsem ablation)
#   R3  L1 + mu  * consistency    |D(SR) - LR|          back-projection
#   R4  Laplace NLL               predicted scale       uncertainty head
#
# THE MIDSEM ARGUMENT, KEPT HERE BECAUSE IT IS STILL THE POINT
# SAM is scale-invariant -- SAM(a*p, t) == SAM(p, t) for any a > 0 -- so it
# only sees the DIRECTION of a pixel's 4-band vector and ignores brightness.
# It therefore bottoms out wherever L1 bottoms out and cannot pull the
# network toward a solution L1 would have rejected. That objection is
# correct. What it does NOT settle is whether SAM bends
# the optimisation PATH on the way there, which is a separate, measurable
# question: train.py probes it and analysis A1 reports the answer (cos 0.81).
#
# WHY R3 IS DIFFERENT IN KIND
# L1(SR, HR) compares the prediction with the TARGET. It never looks at the
# input. So a term built on the INPUT cannot be implied by it:
#
#     L = L1(SR, HR) + mu * |D(SR) - LR|
#
# D degrades the prediction back to 10 m and the term demands that doing so
# reproduces the Sentinel-2 image we started from. Invent detail that was
# never measured and this term notices; L1 has no way to. That is iterative
# back-projection (Irani & Peleg 1991) and the consistency property of Wald's
# protocol (Wald, Ranchin & Mangolini 1997) -- old and standard, which is a
# strength, not a weakness. Cite it rather than claiming it as new.
#
# Honest caveat to state before anyone asks: the input is Sentinel-2 and the
# target is VENuS, two different sensors, so D(HR) only approximately equals
# LR. The two terms are therefore in mild tension and R3 will probably COST
# PSNR. The defensible claim is radiometric fidelity to the measurement that
# downstream analytics actually consume.
#
# WHY D NOW COMES FROM degrade.py
# D used to be a 2x2 box average. Decimating by 2 keeps only frequencies
# below pi/2, and the box is a poor anti-alias filter: it leaves 18% of
# surviving energy above pi/2 to fold back as aliasing. Since D sits inside
# both metric M5 and this loss, that choice was silently shaping a reported
# number. degrade.py supplies a bank of filters all cut at pi/2 so the
# filter SHAPE becomes a controlled variable.

import torch
import torch.nn.functional as F

from degrade import degrade_with
from uncertainty import laplace_nll

RED, NIR = 2, 3          # bands are ordered B2, B3, B4(red), B8(nir)

DEFAULT_FILTER = "box2"  # the midsem operator; kept as the default so old
                         # numbers stay reproducible. Studies pass a name.


def degrade(y, name=DEFAULT_FILTER):
    """5 m -> 10 m. Kept at module level because metrics.py and sanity.py
    both import it by this name; `name` selects the anti-alias filter."""
    return degrade_with(y, name)


def ndvi(x, eps=1e-6):
    r, n = x[:, RED], x[:, NIR]
    return (n - r) / (n + r + eps)


def sam_loss(pred, target, eps=1e-8):
    dot = (pred * target).sum(dim=1)
    pn = pred.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    tn = target.pow(2).sum(dim=1).clamp_min(eps).sqrt()
    # Leave this clamp alone. cos can land on 1.0000001 in float32 and
    # acos returns NaN, which silently poisons the whole run.
    cos = (dot / (pn * tn)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos).mean()


def consistency_loss(sr, lr, name=DEFAULT_FILTER):
    """|D(SR) - LR|, the back-projection / Wald consistency term."""
    return (degrade_with(sr, name) - lr).abs().mean()


class LossSpec:
    """Recipe for one run, with the terms kept separate.

    Separate is the point: analysis A1 differentiates the main and auxiliary
    terms on their own to compare their gradients, which is impossible once
    they have been summed into a single scalar.

    `parts(sr, hr, lr)` returns (main, aux). `lr` is only needed by R3, but
    every caller passes it so the signature stays uniform.
    """

    KINDS = ("l1", "l1_sam", "l1_consist", "nll")

    def __init__(self, kind, lam=0.1, mu=0.1, d_filter=DEFAULT_FILTER,
                 beta_nll=0.5):
        assert kind in self.KINDS, "unknown loss kind %r" % kind
        self.kind = kind
        self.lam = lam                  # SAM weight            (R2)
        self.mu = mu                    # consistency weight    (R3)
        self.d_filter = d_filter        # anti-alias filter in D (R3)
        self.beta_nll = beta_nll        # beta-NLL exponent     (R4)

    # -- what the run needs from the model -------------------------------
    @property
    def needs_sigma(self):
        """R4 needs the two-output model; the others use the plain one."""
        return self.kind == "nll"

    @property
    def aux_name(self):
        return {"l1": None, "nll": None,
                "l1_sam": "sam", "l1_consist": "consist"}[self.kind]

    @property
    def aux_weight(self):
        return {"l1": 0.0, "nll": 0.0,
                "l1_sam": self.lam, "l1_consist": self.mu}[self.kind]

    def label(self):
        return {"l1": "CNN, L1",
                "l1_sam": "CNN, L1 + lam*SAM",
                "l1_consist": "CNN, L1 + mu*consistency [D=%s]" % self.d_filter,
                "nll": "CNN, Laplace NLL (uncertainty head)"}[self.kind]

    # -- the loss itself -------------------------------------------------
    def parts(self, out, hr, lr=None):
        """(main, aux). `out` is a tensor, or (mu, log_b) for R4."""
        if self.kind == "nll":
            mu_, log_b = out
            return laplace_nll(mu_, log_b, hr, beta=self.beta_nll), None

        sr = out
        main = F.l1_loss(sr, hr)
        if self.kind == "l1":
            aux = None
        elif self.kind == "l1_sam":
            aux = sam_loss(sr, hr)
        else:                                   # l1_consist
            assert lr is not None, "R3 needs the LR input for consistency"
            aux = consistency_loss(sr, lr, self.d_filter)
        return main, aux

    def total(self, out, hr, lr=None):
        main, aux = self.parts(out, hr, lr)
        return main if aux is None else main + self.aux_weight * aux

    def mean_of(self, out):
        """The SR estimate, whichever head shape the run uses."""
        return out[0] if self.needs_sigma else out
