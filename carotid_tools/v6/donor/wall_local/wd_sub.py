#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\wd_sub.py (= /sp/v6/wall_local/wd_sub.py in the build containers)
"""wd_sub.py (s2_harden_wall_bake): wd_proxy.py (imported unchanged) on a subclavian stretch from another folder:
  python3 wd_sub.py <tag> <subclavian root> <unit> <R|L>   -> /v6w/wd/<tag>/<unit>_<side>.json
(stage 1's 'subclavian' variant reads /v6w/subclavian only). Batch: wd_sub.py --all <tag> <root>"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/sp/v6/wall")
import wd_proxy as WP  # noqa: E402

ROOT = None


def load(variant, unit, side):
    rec = json.load(open(os.path.join(WP.R5, unit, "%s.json" % side)))
    v = "rsub" if side == "R" else "lsub"
    d = os.path.join(ROOT, unit)
    vj = json.load(open(os.path.join(d, "%s_stageA_vmtk.json" % v)))
    return rec, np.asarray(vj["points"], float), np.asarray(vj["radii"], float), os.path.join(d, "%s_stageA.vtp" % v)


WP.load = load
if __name__ == "__main__":
    if sys.argv[1] == "--all":
        tag, ROOT = sys.argv[2], sys.argv[3]
        jobs = []
        for u in sorted(os.listdir(ROOT)):
            for sd, v in (("R", "rsub"), ("L", "lsub")):
                if os.path.exists(os.path.join(ROOT, u, "%s_stageA_vmtk.json" % v)) and os.path.exists(os.path.join(WP.R5, u, "%s.json" % sd)):
                    jobs.append((u, sd))
        for u, sd in jobs:
            sys.argv = ["wd_proxy.py", "subclavian", u, sd, "--tag", tag]
            try:
                WP.main()
            except Exception as e:  # noqa: BLE001
                print("FAILED", u, sd, repr(e)[:200], flush=True)
    else:
        tag, ROOT, u, sd = sys.argv[1:5]
        sys.argv = ["wd_proxy.py", "subclavian", u, sd, "--tag", tag]
        WP.main()
