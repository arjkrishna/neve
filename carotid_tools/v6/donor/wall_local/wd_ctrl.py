#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\wd_ctrl.py (= /sp/v6/wall_local/wd_ctrl.py in the build containers)
"""wd_ctrl.py (s2_harden_wall_bake, D15 control): the resampling noise of the bake-like union measure.

wd_proxy (imported unchanged) on v5's OWN surface and v5's OWN line with only the line's far end trimmed by TRIM_MM
(0.3 mm, beyond every used stretch: those end 25 mm before the splice). The geometry over the used stretches is
identical to v5's; only prep_siphon's sample spacing changes (it spaces its 1 mm samples by the line's total length).
The per-VA difference of the outside-zone wall share between this control and v5 is the measure's own noise floor,
to compare with local vs v5.
    python3 wd_ctrl.py [unit,...]  -> /v6w/wd/v5trim/<unit>_<side>.json      (neve-build-stagea)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/wall")
import wd_proxy as WP  # noqa: E402

TRIM_MM = 0.3


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def trimmed(P, R, trim=TRIM_MM):
    s = arclen(P)
    keep = s <= s[-1] - trim
    k = int(keep.sum())
    end = np.array([np.interp(s[-1] - trim, s, P[:, i]) for i in range(3)])
    rend = float(np.interp(s[-1] - trim, s, R))
    return np.vstack([P[:k], end]), np.r_[R[:k], rend]


def load(variant, unit, side):
    rec = json.load(open(os.path.join(WP.R5, unit, "%s.json" % side)))
    P = np.asarray(rec["centerline"]["points"], float)
    R = np.asarray(rec["centerline"]["radii_misr"], float)
    P, R = trimmed(P, R)
    return rec, P, R, rec["surface"]["path"]


WP.load = load
if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    for unit in sorted(os.listdir(WP.R5)):
        if only and unit not in only:
            continue
        for sd in ("R", "L"):
            if not os.path.exists(os.path.join(WP.R5, unit, "%s.json" % sd)):
                continue
            sys.argv = ["wd_proxy.py", "v5trim", unit, sd, "--tag", "v5trim"]
            try:
                WP.main()
            except Exception as e:  # noqa: BLE001
                print("FAILED", unit, sd, repr(e)[:200], flush=True)
