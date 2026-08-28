"""Trunk IDENTITY check, done properly: point-to-polyline (not index/arclength pairing)
for the RCCA centerline, and declared-radius identity at matched arclength, host vs all 22.
Also checks whether the 15 non-RCCA centerlines are byte-identical to the host's.
"""
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
EXC = {"topcow_mr_013", "topcow_mr_014", "topcow_mr_015"}
NAMES = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))
         if os.path.isdir(d) and os.path.basename(d) not in EXC]


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def dense(C, step=0.1):
    S = arclen(C)
    q = np.arange(0, S[-1], step)
    return np.stack([np.interp(q, S, C[:, k]) for k in range(3)], 1)


def rcca(brs):
    b = next(x for x in brs if "RCCA" in str(x.name).upper())
    return np.asarray(b.coordinates, float), np.asarray(b.radii, float)


hb = list(DualDeviceNav().vessel_tree.branches)
Ch, Rh = rcca(hb)
Sh = arclen(Ch)
HD = dense(Ch)
TH = cKDTree(HD)
hostmap = {str(b.name): (np.asarray(b.coordinates, float), np.asarray(b.radii, float)) for b in hb}
GRID = np.array([10, 20, 40, 60, 80, 90, 100, 103, 105, 110, 115, 120, 125, 129], float)

print("%-15s %s" % ("anatomy", "  ".join("d@%d/dr@%d" % (g, g) for g in [40, 80, 100, 110, 120, 129])))
D = {}
for n in NAMES:
    brs = load_branches(os.path.join(ROOT, n, "Centrelines_comb"))
    C, R = rcca(brs)
    S = arclen(C)
    m = S < 132
    d, _ = TH.query(C[m])
    Sm = S[m]
    dev = np.array([d[np.argmin(abs(Sm - g))] for g in GRID])
    dr = np.array([np.interp(g, S, R) - np.interp(g, Sh, Rh) for g in GRID])
    D[n] = (dev, dr)
    sel = [list(GRID).index(g) for g in [40, 80, 100, 110, 120, 129]]
    print("%-15s %s" % (n, "  ".join("%.3f/%+.3f" % (dev[i], dr[i]) for i in sel)))

DEV = np.array([D[n][0] for n in NAMES])
DR = np.array([D[n][1] for n in NAMES])
print("\ns(mm)   dev_med dev_max | dradius_med dradius_max_abs  (22 anatomies vs host)")
for i, g in enumerate(GRID):
    print("%5.0f   %7.4f %7.4f | %11.4f %15.4f"
          % (g, np.median(DEV[:, i]), DEV[:, i].max(), np.median(DR[:, i]), np.abs(DR[:, i]).max()))

# non-RCCA centerline identity
print("\nnon-RCCA centerline identity vs host (max abs coord diff, mm):")
brs = load_branches(os.path.join(ROOT, NAMES[0], "Centrelines_comb"))
for b in brs:
    nm = str(b.name)
    if "RCCA" in nm.upper() or nm not in hostmap:
        continue
    worst = 0.0
    worstr = 0.0
    nsh = None
    for n in NAMES:
        bb = next(x for x in load_branches(os.path.join(ROOT, n, "Centrelines_comb"))
                  if str(x.name) == nm)
        A = np.asarray(bb.coordinates, float)
        B = hostmap[nm][0]
        nsh = (A.shape, B.shape)
        if A.shape != B.shape:
            worst = np.inf
            continue
        worst = max(worst, float(np.abs(A - B).max()))
        worstr = max(worstr, float(np.abs(np.asarray(bb.radii, float) - hostmap[nm][1]).max()))
    print("  %-32s shape %s vs %s  maxdiff=%.6g  maxdr=%.6g" % (nm, nsh[0], nsh[1], worst, worstr))
