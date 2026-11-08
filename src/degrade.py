"""Anti-alias low-pass filters for x2 decimation, all cut at omega_c = pi/2.

Why this file exists

The degradation operator D maps a 5 m prediction back down to the 10 m grid.
It appears in two places that matter:

    metric M5   |D(SR) - LR|          radiometric agreement with the input
    run R3      L1 + mu*|D(SR) - LR|  back-projection / Wald consistency

Until now D was `avg_pool2d(y, 2)`, a 2x2 box. Decimating by 2 keeps only
frequencies below pi/2 rad/sample; everything above that folds back as
aliasing, so D needs a genuine anti-alias filter cut at pi/2. The box is a
poor one. Its response is

    H(w) = cos(w/2)

which is -3.01 dB at pi/2 (nominally correct) but only -8.3 dB at 0.75*pi,
so the half-band that should be rejected is barely attenuated. Every filter
here targets the same pi/2 cutoff and differs only in how cleanly it gets
there, which isolates "shape of the anti-alias filter" as a single variable.

Phase, and why the grid is half-integer

`avg_pool2d(y, 2)` averages input pixels (0,1), then (2,3), ... so output
sample k sits at input coordinate 2k + 0.5 -- a HALF-PIXEL offset from any
integer input pixel. A symmetric odd-length FIR centred on an integer pixel
would sample at 2k instead, putting every filter in this file half a pixel
out of registration with the LR image and inflating M5 for reasons that have
nothing to do with frequency response.

So every kernel here is sampled on the HALF-INTEGER grid n in {..., -1.5,
-0.5, +0.5, +1.5, ...}, giving even-length symmetric kernels whose phase
matches the box exactly. The 2-tap case IS the box: weights 0.5 at +-0.5.
`verify_phase()` checks this empirically on real data.

DC gain

Every kernel is normalised to sum 1. This project is about preserving
reflectance, so D must not change the mean of an image: a filter with DC
gain 0.99 would shift every reflectance by 1% and silently corrupt M5 and
NDVI. `kernel_1d` enforces it and the self-test asserts it.

References

Wald, Ranchin & Mangolini (1997), consistency property / Wald's protocol.
Irani & Peleg (1991), iterative back-projection.
Aiazzi et al. (2006), MTF-matched filters for resolution reduction in
remote sensing -- the source of the `mtf*` filters, whose Gaussian is
specified by its magnitude AT Nyquist rather than by a bandwidth.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

W_C = math.pi / 2.0          # the only cutoff used anywhere in this file
# kernel design
def _half_grid(taps):
    """Half-integer positions, even count: taps=4 -> [-1.5, -0.5, 0.5, 1.5]."""
    assert taps % 2 == 0, "need an even tap count to land on the half grid"
    k = taps // 2
    return np.arange(-k, k) + 0.5


def _sinc(x):
    return np.sinc(x)                      # np.sinc(x) = sin(pi x)/(pi x)


def _trapz(y, x, axis=-1):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x, axis=axis)


def _fir_from_response(h_of_w, taps, window=None, quad=4096):
    """Design a zero-phase FIR on the half-integer grid from a magnitude response.

    For a symmetric kernel sampled at half-integer n, the inverse DTFT
    reduces to a cosine integral:

        h(n) = (1/pi) * integral_0^pi H(w) cos(w n) dw

    evaluated numerically. This handles any target magnitude -- an ideal
    brick wall, a Butterworth roll-off -- without needing a closed form.
    """
    n = _half_grid(taps)
    w = np.linspace(0.0, math.pi, quad)
    H = h_of_w(w)
    integrand = H[None, :] * np.cos(np.outer(n, w))
    h = _trapz(integrand, w, axis=1) / math.pi
    if window is not None:
        h = h * window(n, taps)
    return h


def _hamming(n, taps):
    """Hamming over the kernel support, evaluated on the half grid."""
    half = taps / 2.0
    return 0.54 + 0.46 * np.cos(math.pi * n / half)


def _lanczos_window(n, a):
    return np.where(np.abs(n) < a, _sinc(n / a), 0.0)


def kernel_1d(name, taps=None):
    """A 1-D anti-alias kernel on the half-integer grid, DC gain exactly 1.

    Every entry targets omega_c = pi/2. Names:

      box2          2x2 average -- the original D, kept as the baseline
      gauss3db      Gaussian, -3 dB at pi/2
      gauss6db      Gaussian, -6 dB at pi/2
      mtf30         Gaussian with magnitude 0.30 at Nyquist (sensor-like)
      mtf20         Gaussian with magnitude 0.20 at Nyquist (sensor-like)
      sinc_hamming  Hamming-windowed ideal brick wall at pi/2
      lanczos3      Lanczos-3 windowed sinc at pi/2
      bicubic       Keys bicubic (a = -0.5) resampling kernel
      butter4       4th-order Butterworth magnitude, FIR-designed
      butter8       8th-order Butterworth magnitude, FIR-designed
    """
    if name == "box2":
        h = np.array([0.5, 0.5])

    elif name in ("gauss3db", "gauss6db", "mtf30", "mtf20"):
        # Gaussian H(w) = exp(-sigma^2 w^2 / 2); pick sigma from the
        # magnitude required at the cutoff.
        target = {"gauss3db": 1.0 / math.sqrt(2.0),
                  "gauss6db": 0.5,
                  "mtf30": 0.30,
                  "mtf20": 0.20}[name]
        sigma = math.sqrt(-2.0 * math.log(target) / (W_C ** 2))
        t = taps or max(4, 2 * int(math.ceil(3.0 * sigma)))
        t += t % 2
        n = _half_grid(t)
        h = np.exp(-(n ** 2) / (2.0 * sigma ** 2))

    elif name == "sinc_hamming":
        h = _fir_from_response(lambda w: (w <= W_C).astype(float),
                               taps or 12, window=_hamming)

    elif name == "lanczos3":
        n = _half_grid(taps or 12)
        x = n / 2.0                         # sinc scaled to a pi/2 cutoff
        h = 0.5 * _sinc(x) * _lanczos_window(x, 3.0)

    elif name == "bicubic":
        n = _half_grid(taps or 8)
        x = np.abs(n / 2.0)                 # kernel for a x2 rate change
        a = -0.5                            # Keys cubic
        h = np.where(
            x <= 1.0,
            (a + 2.0) * x ** 3 - (a + 3.0) * x ** 2 + 1.0,
            np.where(x < 2.0,
                     a * x ** 3 - 5.0 * a * x ** 2 + 8.0 * a * x - 4.0 * a,
                     0.0))

    elif name.startswith("butter"):
        order = int(name[len("butter"):])
        h = _fir_from_response(
            lambda w: 1.0 / np.sqrt(1.0 + (np.maximum(w, 1e-12) / W_C)
                                    ** (2 * order)),
            taps or 16, window=_hamming)

    else:
        raise KeyError("unknown filter %r; see FILTERS" % name)

    h = np.asarray(h, dtype=np.float64)
    s = h.sum()
    assert abs(s) > 1e-12, "%s: degenerate kernel" % name
    return h / s                            # DC gain exactly 1


FILTERS = ["box2", "gauss3db", "gauss6db", "mtf30", "mtf20",
           "sinc_hamming", "lanczos3", "bicubic", "butter4", "butter8"]
# frequency response, for the analysis figure
def freq_response(h, n_w=512):
    """|H(w)| on [0, pi] for a half-integer-grid symmetric kernel."""
    n = _half_grid(len(h))
    w = np.linspace(0.0, math.pi, n_w)
    H = (h[None, :] * np.cos(np.outer(w, n))).sum(axis=1)
    return w, np.abs(H)


def response_table():
    """Per-filter summary: gain at pi/2, worst stopband leak, aliasing proxy.

    `alias_energy_frac` is the one that matters -- the fraction of energy
    surviving the filter that sits above pi/2, i.e. what will fold back when
    we decimate.
    """
    rows = []
    for name in FILTERS:
        h = kernel_1d(name)
        w, H = freq_response(h)
        stop = w >= W_C
        at_c = float(np.interp(W_C, w, H))
        mx = float(H[stop].max())
        rows.append({
            "filter": name,
            "taps": len(h),
            "gain_at_cutoff": at_c,
            "db_at_cutoff": 20.0 * math.log10(max(at_c, 1e-12)),
            "max_stopband": mx,
            "db_max_stopband": 20.0 * math.log10(max(mx, 1e-12)),
            "alias_energy_frac": float((H[stop] ** 2).sum() / (H ** 2).sum()),
        })
    return rows
# the operator itself
class Degrader(torch.nn.Module):
    """Separable anti-alias filter at pi/2, then decimate by `scale`.

    Phase matches avg_pool2d(x, 2) exactly, so outputs stay registered with
    the LR image. Padding is 'reflect' because satellite patches are crops
    out of a larger scene -- zero padding would darken the border and bias
    M5 at the edges.
    """

    def __init__(self, name="box2", scale=2, channels=4, taps=None,
                 padding_mode="reflect"):
        super().__init__()
        assert scale == 2, "kernels here are designed for x2 only"
        h = kernel_1d(name, taps=taps)
        self.name = name
        self.scale = scale
        self.channels = channels
        self.padding_mode = padding_mode
        self.taps = len(h)
        k = torch.tensor(h, dtype=torch.float32)
        # separable: (1, taps) horizontally then (taps, 1) vertically
        self.register_buffer("kh", k.view(1, 1, 1, -1).repeat(channels, 1, 1, 1))
        self.register_buffer("kv", k.view(1, 1, -1, 1).repeat(channels, 1, 1, 1))

    def forward(self, x):
        c = x.shape[1]
        assert c == self.channels, \
            "Degrader built for %d channels, got %d" % (self.channels, c)
        m = self.taps // 2
        # Output sample k sits at input coordinate 2k + 0.5, so with taps on
        # the half grid it reads x[2k+1-m .. 2k+m]. Under cross-correlation
        # that needs m-1 before and m after -- NOT the other way round. Get
        # this backwards and every filter lands a pixel out of registration
        # with the LR image; the box2 == avg_pool2d assertion is what pins it.
        x = F.pad(x, (m - 1, m, m - 1, m), mode=self.padding_mode)
        x = F.conv2d(x, self.kh, groups=c)
        x = F.conv2d(x, self.kv, groups=c)
        return x[:, :, ::self.scale, ::self.scale]

    def extra_repr(self):
        return "name=%s taps=%d scale=%d" % (self.name, self.taps, self.scale)


_CACHE = {}


def get_degrader(name="box2", channels=4, device=None):
    """Cached Degrader. Metrics call this per batch, so don't rebuild kernels."""
    key = (name, channels, str(device))
    if key not in _CACHE:
        d = Degrader(name, channels=channels).eval()
        if device is not None:
            d = d.to(device)
        for p in d.parameters():
            p.requires_grad_(False)
        _CACHE[key] = d
    return _CACHE[key]


def degrade_with(y, name="box2"):
    """Functional form: D_name(y). Differentiable, so R3 can train through it."""
    return get_degrader(name, channels=y.shape[1], device=y.device)(y)
# checks
def verify_phase(hr, lr, names=None):
    """Empirical registration check on real data.

    Applies each D to a real HR patch and reports |D(HR) - LR|. A half-pixel
    registration error shows up here as a large, filter-independent penalty,
    so if a filter is wildly worse than box2 the phase is wrong rather than
    the frequency response.
    """
    out = {}
    for name in (names or FILTERS):
        d = get_degrader(name, channels=hr.shape[1], device=hr.device)
        out[name] = (d(hr) - lr).abs().mean().item()
    return out


def _self_test():
    print("filter        taps   |H| @ pi/2     dB     max stop   alias frac")
    for r in response_table():
        print("%-13s %4d   %8.4f  %8.2f   %8.4f   %9.5f"
              % (r["filter"], r["taps"], r["gain_at_cutoff"],
                 r["db_at_cutoff"], r["max_stopband"],
                 r["alias_energy_frac"]))

    print("\nchecks")
    ok = True
    for name in FILTERS:
        h = kernel_1d(name)
        assert abs(h.sum() - 1.0) < 1e-10, name
        assert np.allclose(h, h[::-1], atol=1e-10), "%s not symmetric" % name
    print("  [ok] all kernels: DC gain 1, symmetric")

    # box2 must reproduce avg_pool2d exactly -- that is what pins the phase
    x = torch.rand(2, 4, 32, 32)
    d = (degrade_with(x, "box2") - F.avg_pool2d(x, 2)).abs().max().item()
    ok &= d < 1e-6
    print("  [%s] box2 == avg_pool2d   max diff %.2e"
          % ("ok" if d < 1e-6 else "FAIL", d))

    # a constant image must survive every filter untouched (DC gain)
    for name in FILTERS:
        c = torch.full((1, 4, 32, 32), 0.3137)
        e = (degrade_with(c, name) - 0.3137).abs().max().item()
        assert e < 1e-5, (name, e)
    print("  [ok] constant image preserved by every filter (reflectance safe)")

    for name in FILTERS:
        o = degrade_with(torch.rand(3, 4, 128, 128), name)
        assert o.shape == (3, 4, 64, 64), (name, o.shape)
    print("  [ok] every filter maps 128 -> 64")

    # gradients must flow: R3 trains through D
    y = torch.rand(1, 4, 16, 16, requires_grad=True)
    degrade_with(y, "lanczos3").sum().backward()
    g = y.grad is not None and y.grad.abs().sum().item() > 0
    ok &= g
    print("  [%s] gradient flows through D" % ("ok" if g else "FAIL"))

    print("\n" + ("ALL OK" if ok else "FAILURES PRESENT"))


if __name__ == "__main__":
    _self_test()
