# Patch loading, augmentation, and the train/val/test split.
#
# The split is by SITE, never by patch, and that matters more than it looks.
# Patches inside one site are neighbouring crops of the same fields and the
# same coastline. Shuffle them randomly and a test patch ends up next door
# to a training patch, the model gets credit for recognising terrain it has
# already memorised, and the score means nothing. Training sites train, a
# held-out site is kept sealed.
#
# Reflectance comes out of the archives as int16 scaled by 10000. It can go
# slightly negative where atmospheric correction overshoots, hence the clip.
#
# AUGMENTATION, AND THE RULE WE ARE DELIBERATELY DEPARTING FROM
# The usual rule for this task is flips and 90-degree rotations only, with
# colour jitter and brightness changes ruled out because they corrupt the
# reflectance values the whole project is about preserving. That rule is
# right about what it targets, and it is worth being precise about why it
# does not forbid what is added here.
#
# The rule protects the LR -> HR RELATION. Jittering the brightness of the
# input while leaving the target alone, or jittering the target
# independently, teaches the network a mapping that is not the sensor's and
# destroys radiometric fidelity.
#
# Multiplying LR and HR by the SAME factor is a different operation. The pair
# stays exactly as physically consistent as it was: for a linear sensor,
# scaling the at-sensor radiance scales both captures identically, so
# (a*LR, a*HR) is as valid a training pair as (LR, HR). Nothing about the
# relation is corrupted. What changes is that the network can no longer rely
# on the ABSOLUTE brightness of the training distribution, because the same
# scene content now appears at many brightness levels.
#
# That is precisely the failure measured at the midpoint of the project:
# the model learned a +0.003 reflectance brightness prior from European
# coast and farmland and applied it to Amazon rainforest, where true red is
# about 0.004, making it a 75% relative error that propagates straight into
# NDVI. A model trained to be equivariant to brightness has no absolute
# brightness prior to carry across biomes.
#
# Per-band gain goes one step further and is the more targeted fix. FGMANAUS
# does not merely differ in overall brightness; it differs in the RATIOS
# between bands, with visible bands 4-8x darker relative to NIR than
# anything in training. Drawing an independent gain per band, applied
# identically to LR and HR, covers that part of the shift too. The target's
# NDVI moves when we do this, but it moves CONSISTENTLY with the input, so
# what the network is taught is "preserve whatever spectral ratio arrives"
# rather than "reproduce the spectral ratios of southern France".
#
# Defaults are all OFF, so every midsem number stays reproducible.

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
SCALE_REFL = 10000.0


class AugConfig:
    """Augmentation switches. Everything defaults to the midsem behaviour.

    geometric     flips and 90-degree rotations (the midsem setting)
    gain_band     per-band multiplicative gain, applied to LR and HR alike.
                  Range is log-uniform in [1/r, r] so it is symmetric in
                  log space; a plain uniform range would brighten on average.
    gain_global   one gain for all four bands. Preserves band ratios, so it
                  teaches brightness-equivariance WITHOUT touching NDVI.
                  Weaker than gain_band but the easier sell if anyone
                  objects to spectral augmentation at all.
    mixup         convex blend of two patches with the same weight in LR and
                  HR. Linear in reflectance, so a blend is itself a
                  physically meaningful mixed pixel.
    cutmix        paste a rectangle from another patch, in LR and HR at
                  matching coordinates.
    """

    def __init__(self, geometric=True, gain_band=0.0, gain_global=0.0,
                 mixup=0.0, cutmix=0.0, seed_offset=0):
        self.geometric = geometric
        self.gain_band = gain_band        # 0 disables; else max log-ratio r
        self.gain_global = gain_global
        self.mixup = mixup                # probability
        self.cutmix = cutmix              # probability
        self.seed_offset = seed_offset

    @property
    def any_photometric(self):
        return bool(self.gain_band or self.gain_global)

    def describe(self):
        bits = ["geom" if self.geometric else "no-geom"]
        if self.gain_band:
            bits.append("gain_band=%.2f" % self.gain_band)
        if self.gain_global:
            bits.append("gain_global=%.2f" % self.gain_global)
        if self.mixup:
            bits.append("mixup=%.2f" % self.mixup)
        if self.cutmix:
            bits.append("cutmix=%.2f" % self.cutmix)
        return " ".join(bits)

    def to_dict(self):
        return {"geometric": self.geometric, "gain_band": self.gain_band,
                "gain_global": self.gain_global, "mixup": self.mixup,
                "cutmix": self.cutmix}


AUG_NONE = AugConfig(geometric=True)                      # midsem setting
AUG_FULL = AugConfig(geometric=True, gain_band=1.6, mixup=0.2, cutmix=0.2)
AUG_GAIN = AugConfig(geometric=True, gain_band=1.6)       # the targeted fix


class SRPatches(Dataset):
    def __init__(self, sites, crop_lr=64, train=True, cache=CACHE,
                 indices=None, aug=None):
        # memmap rather than load: the HR arrays are ~0.5 GB per site and
        # we only ever touch one patch at a time
        self.lr, self.hr = [], []
        self.sites = list(sites)
        for s in sites:
            a = np.load(os.path.join(cache, s + "_lr.npy"), mmap_mode="r")
            b = np.load(os.path.join(cache, s + "_hr.npy"), mmap_mode="r")
            assert len(a) == len(b), "%s: %d LR vs %d HR" % (s, len(a), len(b))
            self.lr.append(a)
            self.hr.append(b)
        self.offsets = np.cumsum([0] + [len(a) for a in self.lr])
        self.n_total = int(self.offsets[-1])
        self.indices = (np.arange(self.n_total) if indices is None
                        else np.asarray(indices))
        self.crop_lr = crop_lr
        self.train = train
        self.aug = aug if aug is not None else AUG_NONE

    def __len__(self):
        return len(self.indices)

    def _locate(self, gi):
        """Global patch number -> (which site, which patch in that site)."""
        k = int(np.searchsorted(self.offsets, gi, side="right") - 1)
        return k, gi - int(self.offsets[k])

    def _raw(self, i):
        """One pair as float reflectance, cropped and geometrically augmented."""
        k, j = self._locate(int(self.indices[i]))
        lo = np.asarray(self.lr[k][j], dtype=np.float32) / SCALE_REFL
        hi = np.asarray(self.hr[k][j], dtype=np.float32) / SCALE_REFL

        if self.train and self.crop_lr:
            _, H, W = lo.shape
            c = self.crop_lr
            if H > c and W > c:
                y = np.random.randint(0, H - c + 1)
                x = np.random.randint(0, W - c + 1)
                lo = lo[:, y:y + c, x:x + c]
                hi = hi[:, y * 2:(y + c) * 2, x * 2:(x + c) * 2]
            # Flips and 90-degree turns. Satellite imagery is top-down so
            # there is no "up" to get wrong.
            if self.aug.geometric:
                r = np.random.randint(4)
                if r:
                    lo = np.rot90(lo, r, (1, 2))
                    hi = np.rot90(hi, r, (1, 2))
                if np.random.rand() < 0.5:
                    lo, hi = lo[:, :, ::-1], hi[:, :, ::-1]
                if np.random.rand() < 0.5:
                    lo, hi = lo[:, ::-1], hi[:, ::-1]
        return np.ascontiguousarray(lo), np.ascontiguousarray(hi)

    def __getitem__(self, i):
        lo, hi = self._raw(i)
        a = self.aug

        if self.train:
            # ---- photometric: the SAME factor on both sides ----
            # Drawn log-uniformly so the expected log-gain is 0; a uniform
            # draw on [1/r, r] would brighten on average and bias the
            # reflectance distribution we are trying to stay honest about.
            if a.gain_band:
                g = np.exp(np.random.uniform(-np.log(a.gain_band),
                                             np.log(a.gain_band),
                                             size=(lo.shape[0], 1, 1)))
                lo = lo * g
                hi = hi * g.astype(np.float32)
            if a.gain_global:
                g = float(np.exp(np.random.uniform(-np.log(a.gain_global),
                                                   np.log(a.gain_global))))
                lo, hi = lo * g, hi * g

            # ---- patch mixing: same weight / same box on both sides ----
            if a.mixup and np.random.rand() < a.mixup:
                lo2, hi2 = self._raw(np.random.randint(len(self)))
                if lo2.shape == lo.shape and hi2.shape == hi.shape:
                    w = float(np.random.beta(0.2, 0.2))
                    lo = lo * w + lo2 * (1.0 - w)
                    hi = hi * w + hi2 * (1.0 - w)

            if a.cutmix and np.random.rand() < a.cutmix:
                lo2, hi2 = self._raw(np.random.randint(len(self)))
                if lo2.shape == lo.shape and hi2.shape == hi.shape:
                    _, H, W = lo.shape
                    ch, cw = np.random.randint(H // 8, H // 2 + 1), \
                        np.random.randint(W // 8, W // 2 + 1)
                    y = np.random.randint(0, H - ch + 1)
                    x = np.random.randint(0, W - cw + 1)
                    lo[:, y:y + ch, x:x + cw] = lo2[:, y:y + ch, x:x + cw]
                    # x2 scale: the HR box is exactly the LR box doubled, so
                    # the pasted region stays registered across resolutions
                    hi[:, 2 * y:2 * (y + ch), 2 * x:2 * (x + cw)] = \
                        hi2[:, 2 * y:2 * (y + ch), 2 * x:2 * (x + cw)]

        lo = np.clip(np.ascontiguousarray(lo), 0.0, 1.0).astype(np.float32)
        hi = np.clip(np.ascontiguousarray(hi), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(lo), torch.from_numpy(hi)


def loaders(train_sites, test_sites, batch=16, crop_lr=64, val_frac=0.1,
            seed=0, workers=0, aug=None):
    # Validation is a random slice of the TRAINING sites. It picks the
    # early-stopping checkpoint, so it has to come from data the model is
    # allowed to see. The test site stays sealed until the final numbers.
    #
    # Note validation and test NEVER get augmentation: they are measurements,
    # not training data, and augmenting them would change what is measured.
    n = SRPatches(train_sites, crop_lr=crop_lr).n_total
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(1, int(round(n * val_frac)))

    tr = SRPatches(train_sites, crop_lr=crop_lr, train=True,
                   indices=perm[n_val:], aug=aug)
    va = SRPatches(train_sites, crop_lr=0, train=False, indices=perm[:n_val])
    te = SRPatches(test_sites, crop_lr=0, train=False)

    def mk(d, shuffle, b):
        return DataLoader(d, batch_size=b, shuffle=shuffle,
                          num_workers=workers, drop_last=shuffle,
                          pin_memory=True)
    return mk(tr, True, batch), mk(va, False, 8), mk(te, False, 8)


def available_sites(cache=CACHE):
    if not os.path.isdir(cache):
        return []
    return sorted({f[:-7] for f in os.listdir(cache) if f.endswith("_lr.npy")})


def site_stats(site, cache=CACHE, max_patches=400):
    """Per-band reflectance statistics for one site.

    This is the table that diagnosed the midsem generalisation failure, and
    it is how biome groups get chosen -- empirically, from what the bands
    actually look like, rather than from guessing at each site's geography.
    """
    lr = np.load(os.path.join(cache, site + "_lr.npy"), mmap_mode="r")
    n = len(lr)
    idx = (np.linspace(0, n - 1, min(n, max_patches)).astype(int)
           if n else np.array([], dtype=int))
    x = np.asarray(lr[idx], dtype=np.float32) / SCALE_REFL
    x = np.clip(x, 0.0, 1.0)
    r, nir = x[:, 2], x[:, 3]
    nd = (nir - r) / (nir + r + 1e-6)
    return {
        "site": site,
        "n_patches": int(n),
        "mean": [float(x[:, c].mean()) for c in range(x.shape[1])],
        "std": [float(x[:, c].std()) for c in range(x.shape[1])],
        "ndvi_mean": float(nd.mean()),
        "ndvi_std": float(nd.std()),
    }
