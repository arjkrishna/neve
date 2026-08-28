#!/usr/bin/env python3
"""Check the grafted anatomies against the unmodified host branch.

The host is the reference, not an abstract ideal: if a grafted route bends no
harder than the shipped RCCA already bends, the graft has not introduced
anything the simulator was not already asked to handle.

Reads the .mrk.json files directly, so it needs only numpy and runs outside
the container.

    python topbrain_tools/validate_anatomies.py
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graft_siphon import GRAFT_MM, RCCA_FILE, arclength, kink_deg, read_curve

HOST = "eve_bench/data/dualdevicenav/Centrelines_comb"


def stats(p, r, junction_mm=None):
    s = arclength(p)
    kk = kink_deg(p)
    out = {"len": float(s[-1]), "maxkink": float(kk.max()),
           "rise": float(p[-1][2] - p[0][2]),
           "dmin": float(2 * r.min()), "dmax": float(2 * r.max()),
           "dterm": float(2 * r[-1])}
    if junction_mm is not None:
        j = int(np.argmin(np.abs(s - junction_mm)))
        lo, hi = max(0, j - 4), min(len(kk), j + 4)
        out["junction"] = float(kk[lo:hi].max())
    else:
        out["junction"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", default="topbrain_data/anatomies")
    ap.add_argument("--host", default=HOST)
    a = ap.parse_args()

    _, hp, hr = read_curve(os.path.join(a.host, RCCA_FILE))
    h = stats(hp, hr)

    cols = ("anatomy", "len_mm", "rise", "maxkink", "junction", "d_min", "d_term")
    print("%-16s %8s %7s %8s %9s %7s %7s" % cols)
    print("%-16s %8.1f %7.1f %8.1f %9s %7.2f %7.2f"
          % ("HOST(shipped)", h["len"], h["rise"], h["maxkink"], "-",
             h["dmin"], h["dterm"]))

    rows = []
    for f in sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb",
                                           RCCA_FILE))):
        stem = f.split(os.sep)[-3]
        _, p, r = read_curve(f)
        st = stats(p, r, GRAFT_MM)
        rows.append(st)
        print("%-16s %8.1f %7.1f %8.1f %9.1f %7.2f %7.2f"
              % (stem, st["len"], st["rise"], st["maxkink"], st["junction"],
                 st["dmin"], st["dterm"]))

    if not rows:
        print("\nno anatomies found under %s" % a.anatomies)
        return 1

    def col(k):
        return np.array([x[k] for x in rows])

    print("\n%d anatomies" % len(rows))
    print("  route length   %.0f - %.0f mm   (host %.0f)"
          % (col("len").min(), col("len").max(), h["len"]))
    print("  rise           %.0f - %.0f mm   (host %.0f over its whole length)"
          % (col("rise").min(), col("rise").max(), h["rise"]))
    print("  worst bend     %.0f - %.0f deg  (host %.0f)"
          % (col("maxkink").min(), col("maxkink").max(), h["maxkink"]))
    print("  junction bend  %.0f - %.0f deg  (host worst bend %.0f)"
          % (col("junction").min(), col("junction").max(), h["maxkink"]))
    print("  min diameter   %.2f - %.2f mm   (host %.2f)"
          % (col("dmin").min(), col("dmin").max(), h["dmin"]))

    bad = [i for i, x in enumerate(rows) if x["rise"] <= 0]
    print("\n  descending routes: %s" % (len(bad) or "none"))
    worse = int((col("junction") > h["maxkink"]).sum())
    print("  junctions bending harder than the host's own worst bend: %d/%d"
          % (worse, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
