"""Learned baselines, so the comparison is not only against bicubic.

Why this file exists

Up to now every learned row in the results table was the same architecture
with a different loss, and the only non-learned reference was bicubic. That
leaves an obvious question unanswered: is the EDSR-style backbone itself a
reasonable choice, or would any other network do better? "We beat bicubic"
is a floor, not a comparison.

Three baselines, all trained under the IDENTICAL protocol used for every
other row (12000 steps, Adam 1e-4 cosine to 1e-6, batch 16, crop 64, bf16
autocast, same seeds, same data, same augmentation flags). The project's own
rule applies: a baseline you under-tune is not a baseline.

    SRCNN       shallow, ~57 k params      the learned FLOOR
    RCAN-lite   channel attention, ~1.4 M  same budget, different inductive bias
    EDSR-large  32 blocks, 128 ch, ~11 M   is 1.37 M simply too small?

The three answer three different questions:

  SRCNN       how much of our gain comes from depth at all
  RCAN-lite   does attention beat plain residual blocks at MATCHED capacity,
              which isolates architecture from parameter count
  EDSR-large  is the backbone capacity-limited, i.e. would the honest fix to
              the out-of-domain gap have been "just make it bigger"

None of them is the contribution. They exist so the contribution can be
stated against something other than interpolation.

References

Dong et al. (ECCV 2014), SRCNN.
Lim et al. (CVPRW 2017), EDSR -- the backbone used throughout this project.
Zhang et al. (ECCV 2018), RCAN -- residual channel attention.
"""

import torch
import torch.nn as nn

from model import ResBlock


class SRCNN(nn.Module):
    """Dong et al. 2014, adapted to x2 and 4 bands.

    The original takes an already-upsampled image and refines it, so we
    keep that structure: bicubic to the output grid, then three convs.
    Deliberately tiny. If this matches the deep model, depth was not
    buying anything and the whole project rests on a weaker footing.
    """

    def __init__(self, in_ch=4, scale=2, c1=64, c2=32):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, in_ch, 5, padding=2),
        )

    def forward(self, x):
        up = nn.functional.interpolate(x, scale_factor=self.scale,
                                       mode="bicubic", align_corners=False)
        # predict the residual on top of bicubic, which is what SRCNN
        # effectively learns and what keeps it stable on reflectance
        return up + self.net(up)


class CALayer(nn.Module):
    """Channel attention: squeeze each channel to one number, learn a gate.

    One caution specific to this project. Global average pooling collapses
    the spatial map to a per-channel mean, which is close to the per-band
    brightness, and the gate is then a learned function of it. That is
    exactly the kind of absolute-brightness dependence behind the
    out-of-domain failure, so RCAN-lite is a genuinely interesting row
    rather than a formality: it may transfer WORSE than the plain residual
    stack even if it fits in-domain better.
    """

    def __init__(self, c, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(c, max(c // reduction, 4), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(c // reduction, 4), c, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class RCAB(nn.Module):
    """Residual channel attention block."""

    def __init__(self, c, reduction=16, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), CALayer(c, reduction),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class ResGroup(nn.Module):
    """A run of RCABs with its own skip: the residual-in-residual of RCAN."""

    def __init__(self, c, n_blocks, reduction=16):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(c, reduction) for _ in range(n_blocks)],
                                  nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class RCANLite(nn.Module):
    """RCAN at roughly our parameter count, so capacity is held fixed.

    4 groups of 4 blocks at 64 channels lands near 1.4 M against the
    backbone's 1.37 M. Same head, same PixelShuffle tail, no BatchNorm, so
    the only difference from SRNet is the channel attention inside each
    block.
    """

    def __init__(self, in_ch=4, c=64, n_groups=4, n_blocks=4, scale=2,
                 reduction=16):
        super().__init__()
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(
            *[ResGroup(c, n_blocks, reduction) for _ in range(n_groups)],
            nn.Conv2d(c, c, 3, padding=1))
        self.tail = nn.Sequential(
            nn.Conv2d(c, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, in_ch, 3, padding=1))

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)
        return self.tail(f)


class EDSRLarge(nn.Module):
    """Our backbone scaled up: 32 blocks at 128 channels, about 11 M.

    Answers the capacity question directly. If this closes the
    out-of-domain gap on its own, the interesting part of the project was
    never the loss design; if it does not, capacity was not the limit and
    the data and objective arguments stand.
    """

    def __init__(self, in_ch=4, c=128, n_blocks=32, scale=2):
        super().__init__()
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(n_blocks)],
                                  nn.Conv2d(c, c, 3, padding=1))
        self.tail = nn.Sequential(
            nn.Conv2d(c, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, in_ch, 3, padding=1))

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)
        return self.tail(f)


ARCHS = {"srcnn": SRCNN, "rcan": RCANLite, "edsr_large": EDSRLarge}


def build(name, **kw):
    if name in ARCHS:
        return ARCHS[name](**kw)
    raise KeyError("unknown architecture %r; have %s"
                   % (name, sorted(ARCHS)))


if __name__ == "__main__":
    from model import SRNet, count_params
    x = torch.rand(2, 4, 64, 64)
    rows = [("SRNet (ours)", SRNet()), ("SRCNN", SRCNN()),
            ("RCAN-lite", RCANLite()), ("EDSR-large", EDSRLarge())]
    print("%-14s %12s   output" % ("model", "params"))
    for nm, m in rows:
        y = m(x)
        assert y.shape == (2, 4, 128, 128), (nm, y.shape)
        print("%-14s %12s   %s" % (nm, format(count_params(m), ","),
                                   tuple(y.shape)))
    print("\nRCAN-lite vs ours: %+.1f%% parameters"
          % (100 * (count_params(RCANLite()) / count_params(SRNet()) - 1)))
