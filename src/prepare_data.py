"""Turn SEN2VENuS site archives into cached numpy arrays.

Each site ships as a zip of per-date zips of GeoTIFF patches. For the x2
4-band task only two band groups matter:

    b2b3b4b8_10m  ->  Sentinel-2 input   (LR)
    b2b3b4b8_05m  ->  VENuS target       (HR)

Patches are paired by the integer index in the filename, e.g.
    FGMANAUS_17_2019-12-04_21MTT_b2b3b4b8_10m.tif
    FGMANAUS_17_2019-12-04_21MTT_b2b3b4b8_05m.tif

Output per site:  <cache>/<SITE>_lr.npy, <SITE>_hr.npy  (int16, N,C,H,W)
"""
import argparse
import io
import os
import re
import sys
import zipfile

import numpy as np
import tifffile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")

LR_TAG = "b2b3b4b8_10m"
HR_TAG = "b2b3b4b8_05m"
KEY_RE = re.compile(r"^(?P<site>.+?)_(?P<idx>\d+)_(?P<date>[\d-]+)_"
                    r"(?P<tile>[^_]+)_b2b3b4b8_(?P<res>\d+m)\.tif$")


def _read_tif(blob):
    """Return (C, H, W) array from a GeoTIFF byte blob."""
    a = tifffile.imread(io.BytesIO(blob))
    if a.ndim == 2:
        a = a[None]
    elif a.shape[-1] <= 8 and a.shape[0] > 8:     # H, W, C -> C, H, W
        a = np.transpose(a, (2, 0, 1))
    return a


def _index(site_zip, tag):
    """Map (date, tile, idx) -> (inner zip name, entry name).

    References, NOT bytes. The original version read every TIF of both band
    groups into two dicts before pairing them, which for a 4 GB site needs
    several GB of RAM on top of the output arrays. On a 16 GB machine
    already running a training job that is an out-of-memory crash, and it
    would take the training down with it. Indexing first and reading each
    patch only when it is written costs one extra zip open per inner
    archive and bounds memory to one TIF at a time.
    """
    out = {}
    with zipfile.ZipFile(site_zip) as z:
        inners = [n for n in z.namelist() if n.endswith(".zip") and tag in n]
        for inner in inners:
            with zipfile.ZipFile(io.BytesIO(z.read(inner))) as iz:
                for n in iz.namelist():
                    if not n.endswith(".tif"):
                        continue
                    m = KEY_RE.match(os.path.basename(n))
                    if not m:
                        continue
                    k = (m.group("date"), m.group("tile"), int(m.group("idx")))
                    out[k] = (inner, n)
    return out


class _Reader:
    """Reads a referenced TIF, caching the currently open inner archive.

    Entries for one inner archive are contiguous after sorting, so a cache
    of size one turns what would be a reopen per patch into a reopen per
    date.
    """

    def __init__(self, site_zip):
        self.z = zipfile.ZipFile(site_zip)
        self._name = None
        self._iz = None

    def read(self, ref):
        inner, entry = ref
        if inner != self._name:
            self._iz = zipfile.ZipFile(io.BytesIO(self.z.read(inner)))
            self._name = inner
        return self._iz.read(entry)

    def close(self):
        self.z.close()


def prepare(site_zip, cache=CACHE, limit=None):
    site = os.path.basename(site_zip).replace(".zip", "")
    os.makedirs(cache, exist_ok=True)
    lr_path = os.path.join(cache, site + "_lr.npy")
    hr_path = os.path.join(cache, site + "_hr.npy")
    if os.path.exists(lr_path) and os.path.exists(hr_path):
        a = np.load(lr_path, mmap_mode="r")
        print("[skip] %s already cached: %d pairs %s" % (site, len(a), a.shape[1:]))
        return lr_path, hr_path

    print("[%s] scanning archives ..." % site, flush=True)
    lr = _index(site_zip, LR_TAG)
    hr = _index(site_zip, HR_TAG)
    keys = sorted(set(lr) & set(hr))
    n_all = len(keys)
    if limit and n_all > limit:
        # Evenly spaced, not the first N. Keys sort by date, so taking a
        # prefix would keep only the earliest acquisitions and silently
        # drop the seasonal variation that makes the site worth adding.
        sel = np.linspace(0, n_all - 1, limit).astype(int)
        keys = [keys[i] for i in sel]
    print("[%s] %d LR, %d HR, %d paired%s"
          % (site, len(lr), len(hr), len(keys),
             (" (capped from %d, evenly spaced)" % n_all) if limit
             and n_all > limit else ""), flush=True)
    if not keys:
        raise SystemExit("no paired patches found in " + site_zip)

    rd_lr, rd_hr = _Reader(site_zip), _Reader(site_zip)
    a0, b0 = _read_tif(rd_lr.read(lr[keys[0]])), _read_tif(rd_hr.read(hr[keys[0]]))
    print("[%s] LR %s %s   HR %s %s" % (site, a0.shape, a0.dtype,
                                        b0.shape, b0.dtype), flush=True)
    assert b0.shape[1] == a0.shape[1] * 2, "expected exactly x2 scale"

    A = np.zeros((len(keys),) + a0.shape, dtype=np.int16)
    B = np.zeros((len(keys),) + b0.shape, dtype=np.int16)
    for i, k in enumerate(keys):
        A[i] = _read_tif(rd_lr.read(lr[k]))
        B[i] = _read_tif(rd_hr.read(hr[k]))
        if i % 200 == 0:
            print("   %d/%d" % (i, len(keys)), flush=True)
    rd_lr.close()
    rd_hr.close()
    np.save(lr_path, A)
    np.save(hr_path, B)
    print("[%s] cached %d pairs  LR%s HR%s" % (site, len(keys), A.shape[1:],
                                               B.shape[1:]), flush=True)
    print("[%s] LR range [%d, %d]  HR range [%d, %d]"
          % (site, A.min(), A.max(), B.min(), B.max()), flush=True)
    return lr_path, hr_path


def main():
    # --limit N caps patches per site. Two reasons, both worth stating:
    # it bounds the output array (a 4 GB site is ~4000 patches, so 2.1 GB
    # of HR alone), and it keeps the training mix balanced -- an uncapped
    # NARYN would be a larger share of the data than all the other sites
    # together and would quietly reweight the biome distribution.
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    zips = ([os.path.join(RAW, s if s.endswith(".zip") else s + ".zip")
             for s in a.sites] if a.sites else
            sorted(os.path.join(RAW, f) for f in os.listdir(RAW)
                   if f.endswith(".zip")))
    for z in zips:
        if os.path.exists(z):
            prepare(z, limit=a.limit)
        else:
            print("missing:", z, file=sys.stderr)
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
