#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\wd_sub_d17.py (= /sp/v6/wall_d17/wd_sub_d17.py in the build containers)
"""wd_sub_d17.py (s2_harden_wall_bake, D17): wd_proxy.py (imported unchanged) on a D17 subclavian stretch with the
SURFACE from one folder and the LINE (VMTK MISR json) from another -- so the local wall can be measured along the
v5-rule run's own line (tag d17_local_v5line): local and v5-rule then differ only where their surfaces differ (the D15
zones, the D16 tube), free of the VMTK re-trace noise of a second line.
    python3 wd_sub_d17.py <tag> <surface root> <line root> [unit ...]   -> /v6w/wd/<tag>/<unit>_<side>.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/wall")
import wd_proxy as WP  # noqa: E402

SURF, LINE = None, None


def load(variant, unit, side):
    rec = json.load(open(os.path.join(WP.R5, unit, "%s.json" % side)))
    v = "rsub" if side == "R" else "lsub"
    vj = json.load(open(os.path.join(LINE, unit, "%s_stageA_vmtk.json" % v)))
    return rec, np.asarray(vj["points"], float), np.asarray(vj["radii"], float), os.path.join(SURF, unit, "%s_stageA.vtp" % v)


WP.load = load
if __name__ == "__main__":
    tag, SURF, LINE = sys.argv[1:4]
    units = sys.argv[4:] or sorted(u for u in os.listdir(SURF) if os.path.isfile(os.path.join(SURF, u, ".seg_done")))
    for u in units:
        for sd, v in (("R", "rsub"), ("L", "lsub")):
            if not (os.path.exists(os.path.join(SURF, u, "%s_stageA.vtp" % v)) and os.path.exists(os.path.join(LINE, u, "%s_stageA_vmtk.json" % v))
                    and os.path.exists(os.path.join(WP.R5, u, "%s.json" % sd))):
                continue
            sys.argv = ["wd_proxy.py", "subclavian", u, sd, "--tag", tag]
            try:
                WP.main()
            except Exception as e:  # noqa: BLE001
                print("FAILED", u, sd, repr(e)[:200], flush=True)
    print("WD_SUB_D17_DONE", tag, flush=True)
