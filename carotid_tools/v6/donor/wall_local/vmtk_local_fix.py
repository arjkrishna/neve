#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\vmtk_local_fix.py (= /sp/v6/wall_local/vmtk_local_fix.py in the build containers)
"""vmtk_local_fix.py (s2_harden_wall_bake, D15): a degenerate VMTK line on a LOCAL surface (vmr_vmtk.py returned
< 90 % of v5's line length; stage 1 met the same on plateau ia172_015 L / ia172_069 L). The surface is NOT changed
(it is v5's outside the zones); tried in order:
  B  seeds 2 mm inward along v5's station path (vmtk_fix.py's repair B, without its largest-component rewrite);
  D  seeds 5 mm inward;
  P  the raw line inside the zones = stage 1's plateau VMTK line (/v6w/seg/plateau/<unit>/<v>_stageA_vmtk.json, itself
     repaired there): inside the core the local surface IS plateau's surface, so its centreline there is the plateau
     line; flagged.
The first that gives >= 90 % of v5's length is written as <v>_stageA_vmtk.json (the failed one kept as
<v>_stageA_vmtk_failed.json); record <v>_vmtk_local_fix.json.
    python3 vmtk_local_fix.py [unit,...]     (neve-build-vmtk, -u 0; scans every side when no unit is given)"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

SEG = "/v6w/seg/local"
D = json.load(open("/sp/v6/wall/donors.json"))
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
VP = {"R": "rva", "L": "lva"}


def length(p):
    try:
        return float(json.load(open(p))["length_mm"])
    except Exception:  # noqa: BLE001
        return 0.0


def run(surface, seeds_p, out):
    subprocess.call(["timeout", "900", "python3", "/sp/v5_va/vmr_vmtk.py", surface, seeds_p, out],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return length(out)


def fix(unit, sd):
    v = VP[sd]
    d = os.path.join(SEG, unit)
    vj = os.path.join(d, "%s_stageA_vmtk.json" % v)
    if not os.path.exists(os.path.join(d, "%s_stageA.vtp" % v)) or not os.path.exists(os.path.join(R5, unit, "%s.json" % sd)):
        return None
    L5 = float(json.load(open(json.load(open(os.path.join(R5, unit, "%s.json" % sd)))["centerline"]["src"]))["length_mm"])
    if length(vj) >= 0.9 * L5:
        return None
    rec = {"unit": unit, "side": sd, "v5_length_mm": L5, "first_length_mm": length(vj), "tried": []}
    if os.path.exists(vj) and not os.path.exists(os.path.join(d, "%s_stageA_vmtk_failed.json" % v)):
        shutil.copy2(vj, os.path.join(d, "%s_stageA_vmtk_failed.json" % v))
    st = json.load(open(os.path.join(D[unit]["v5_seg_dir"], "stations_rays.json")))[v]
    Ps, s = np.asarray(st["P"], float), np.asarray(st["s"], float)
    chosen = None
    for tag, inward in (("B", 2.0), ("D", 5.0)):
        a = Ps[int(np.searchsorted(s, inward))]
        b = Ps[int(np.searchsorted(s, s[-1] - inward)) - 1]
        sp = os.path.join(d, "%s_seeds_in%d.json" % (v, int(inward)))
        json.dump({"source": a.tolist(), "target": b.tolist(), "note": "vmtk_local_fix: v5 station path points %g mm from each end" % inward},
                  open(sp, "w"))
        out = os.path.join(d, "%s_stageA_vmtk_%s.json" % (v, tag))
        L = run(os.path.join(d, "%s_stageA.vtp" % v), sp, out)
        rec["tried"].append({"repair": tag, "seeds_inward_mm": inward, "length_mm": L})
        if L >= 0.9 * (L5 - 2 * inward):
            chosen = (tag, out)
            break
    if chosen is None:
        pp = "/v6w/seg/plateau/%s/%s_stageA_vmtk.json" % (unit, v)
        if length(pp) >= 0.9 * L5 * 0.9:
            chosen = ("P", pp)
            rec["tried"].append({"repair": "P", "length_mm": length(pp), "note": "plateau's VMTK line used as the raw line"})
    if chosen:
        r = json.load(open(chosen[1]))
        r["note"] = (r.get("note") or "") + "; vmtk_local_fix repair %s" % chosen[0]
        r["vmtk_local_fix"] = chosen[0]
        json.dump(r, open(vj, "w"))
        rec["chosen"] = chosen[0]
    else:
        rec["chosen"] = None
    json.dump(rec, open(os.path.join(d, "%s_vmtk_local_fix.json" % v), "w"), indent=1)
    return rec


def main():
    units = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(u for u in os.listdir(SEG) if not u.startswith("_"))
    for u in units:
        for sd in ("R", "L"):
            try:
                r = fix(u, sd)
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)}
            if r:
                print(u, sd, json.dumps(r), flush=True)
    print("VMTK_LOCAL_FIX_DONE")


if __name__ == "__main__":
    main()
