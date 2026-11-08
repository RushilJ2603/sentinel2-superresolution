"""Download selected SEN2VENuS sites from Zenodo record 14603764.

Dataset: Michel et al., SEN2VENuS v2.0.0, DOI 10.5281/zenodo.14603764
Licence: non-commercial (other-nc). Used here for coursework only.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RECORD = "https://zenodo.org/api/records/14603764"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw")

# Two agricultural / coastal European sites for training, one tropical site
# held out for test. Chosen as the smallest available so the download is
# tractable; every site in the record has the same internal structure.
SITES = ["ESTUAMAR.zip", "SUDOUE-4.zip"]


def wanted():
    """Site list from argv, else the default pair."""
    a = sys.argv[1:]
    if not a:
        return SITES
    return [x if x.endswith(".zip") else x + ".zip" for x in a]


def links():
    with urllib.request.urlopen(RECORD, timeout=60) as r:
        d = json.load(r)
    return {f["key"]: (f["links"]["self"], f["size"]) for f in d["files"]}


def fetch(key, url, size):
    dst = os.path.join(OUT, key)
    if os.path.exists(dst) and abs(os.path.getsize(dst) - size) < 1024:
        print("[skip] %s already complete" % key, flush=True)
        return dst
    t0 = time.time()
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    mb = os.path.getsize(dst) / 1e6
    print("[done] %s  %.0f MB in %.0fs (%.2f MB/s)"
          % (key, mb, time.time() - t0, mb / max(time.time() - t0, 1)),
          flush=True)
    return dst


def main():
    os.makedirs(OUT, exist_ok=True)
    lk = links()
    want = wanted()
    jobs = [(s, lk[s][0], lk[s][1]) for s in want if s in lk]
    missing = [s for s in want if s not in lk]
    if missing:
        print("not in record:", missing, file=sys.stderr)
    total = sum(j[2] for j in jobs) / 1e6
    print("fetching %d sites, %.0f MB total" % (len(jobs), total), flush=True)
    with ThreadPoolExecutor(max_workers=min(2, max(len(jobs), 1))) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))
    print("ALL_DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
