# EDSR-style network, x2 upscaling, 4 bands in and 4 bands out.
#
# Shapes for one patch:
#   (4, 64, 64)  --head-->  (64, 64, 64)  --body-->  (64, 64, 64)
#                --tail-->  (4, 128, 128)
#
# No BatchNorm anywhere. EDSR dropped it and explained why: normalising
# feature statistics throws away absolute intensity, which for us IS the
# signal, since we are trying to keep reflectance values intact.

import torch.nn as nn


class ResBlock(nn.Module):
    # output = input + small correction. The '+' is what lets gradients
    # reach the early layers, so a deep stack stays trainable.
    # res_scale shrinks each correction; without it, 16 blocks at 64
    # channels tends to blow up early in training.
    def __init__(self, c, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class SRNet(nn.Module):
    """x2 SR backbone. With predict_sigma=True it also emits a log-scale map.

    The uncertainty head is deliberately TINY: one conv straight to
    in_ch*scale^2 channels plus a PixelShuffle, about 9.2 k parameters
    against the backbone's 1.37 M, so +0.7%. That matters for the ablation.
    Mirroring the full `tail` would have cost ~150 k (+11%) and an examiner
    could then fairly ask whether R4 moved because of the uncertainty head
    or merely because it had more capacity than R1. At +0.7% that objection
    has no room to stand, and the parameter counts are reported either way.

    Existing R1/R2 checkpoints still load: predict_sigma only ADDS the
    `tail_logb` keys and never renames anything on the mean path.
    """

    def __init__(self, in_ch=4, c=64, n_blocks=16, scale=2,
                 predict_sigma=False, init_log_b=-5.3,
                 global_residual=False):
        super().__init__()
        self.predict_sigma = predict_sigma
        # global_residual: output = bicubic(LR) + learned delta, instead of
        # predicting the HR image outright. Kim et al., VDSR, CVPR 2016.
        #
        # Worth the flag because of the diagnosed failure. Predicting
        # the image outright lets the network choose the absolute level,
        # which is how it acquired a +0.003 brightness prior from European
        # farmland and carried it to rainforest. With a bicubic base the
        # level comes from the measurement itself, so that drift is
        # structurally much harder, and a network that is unsure can emit a
        # near-zero delta and fall back to bicubic rather than inventing
        # detail that is wrong for the biome.
        #
        # It also gives the metric a floor: performance degrades toward
        # bicubic instead of below it.
        self.global_residual = global_residual
        self.scale = scale
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(c) for _ in range(n_blocks)],
            nn.Conv2d(c, c, 3, padding=1),
        )
        # Upsampling happens once, at the end, so the 16 blocks all run at
        # low resolution and stay cheap. PixelShuffle makes scale^2 extra
        # channels then interleaves them into a 2x bigger grid -- unlike
        # ConvTranspose2d it doesn't leave checkerboard artifacts.
        self.tail = nn.Sequential(
            nn.Conv2d(c, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, in_ch, 3, padding=1),
        )
        if global_residual:
            # Start the delta at exactly zero, so the untrained network IS
            # bicubic and training departs from it rather than fighting
            # back toward it. Without this the random tail emits noise over
            # the bicubic base and the first few hundred steps are spent
            # cancelling it.
            nn.init.zeros_(self.tail[2].weight)
            nn.init.zeros_(self.tail[2].bias)

        if predict_sigma:
            conv = nn.Conv2d(c, in_ch * scale ** 2, 3, padding=1)
            # Start the scale map flat and near the error magnitude we expect
            # (|err| ~ 5e-3 reflectance, so log b ~ -5.3). Zero weights make
            # the first forward pass output exactly that bias everywhere.
            # Without this the NLL starts with exp(-s) wildly misscaled and
            # the first few hundred steps are spent rescuing it -- a known
            # and avoidable way to lose a run.
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, init_log_b)
            self.tail_logb = nn.Sequential(conv, nn.PixelShuffle(scale))

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)      # long skip: body only learns what to ADD
        mu = self.tail(f)
        if self.global_residual:
            # the base carries the measured level; the net supplies detail
            mu = mu + nn.functional.interpolate(
                x, scale_factor=self.scale, mode="bicubic",
                align_corners=False)
        if not self.predict_sigma:
            return mu
        return mu, self.tail_logb(f)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
