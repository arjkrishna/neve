"""CHECK 11 side-check -- which branches actually vary, by content hash.

Frame sanity: the composites' non-grafted branches should be the shipped host's
own files. If they are, host and composite RCCAs are in the same coordinate
system by construction, which is what licenses the seam comparison.
"""
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

ANAT = "carotid_data/anatomies"
HOSTD = "eve_bench/data/dualdevicenav/Centrelines_comb"


def h(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


names = sorted(n for n in os.listdir(ANAT)
               if os.path.isdir(os.path.join(ANAT, n)))
files = sorted(os.listdir(os.path.join(ANAT, names[0], "Centrelines_comb")))
host_files = set(os.listdir(HOSTD))
print("branches per anatomy: %d" % len(files))
for f in files:
    hs = set()
    for n in names:
        hs.add(h(os.path.join(ANAT, n, "Centrelines_comb", f)))
    same = "-"
    if f in host_files:
        same = "IDENTICAL-TO-HOST" if h(os.path.join(HOSTD, f)) in hs and len(hs) == 1 else "differs"
    print("  %-34s distinct=%3d  %s" % (f, len(hs), same))

# RECA within shared-lower groups
idx = defaultdict(list)
for n in names:
    idx[n.split("__")[0]].append(n)
var = 0
for k, mem in idx.items():
    hs = set(h(os.path.join(ANAT, n, "Centrelines_comb",
                            "Centerline curve - RECA.mrk.json")) for n in mem)
    if len(hs) > 1:
        var += 1
print("lower-donor groups whose RECA is NOT constant within the group: %d of %d"
      % (var, len(idx)))
