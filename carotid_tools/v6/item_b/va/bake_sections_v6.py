#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\va\bake_sections_v6.py (= /sp/v6/va/bake_sections_v6.py in the build containers)
"""bake_sections_v6.py -- the v6 VA-side real-surface sections for the bake (import BEFORE the bake driver runs).

graft_v6_va.py --write records, in provenance.json["xform"], the sections of the VA side with v5's field layout
(R, origin, anchor, mirror, surface, branch, lo_mm, hi_mm, real_surface, why_tube):
    vaC_R / vaC_L   the donor VA (its v5 / v6-wall surface), branch RVA / LVA, [junction path start, splice]
    scC_R           the donor subclavian + junction surface on the RVA, [seam 1 + 8 mm ramp, junction path end + 3]
    scD_R           the same surface on (17), [0 (branch point F), seam-2 start E]
    scC_L / scD_L   the same on (18) [seam 1 + 8, F] / (19) [0, E]
    scJ_L           the same on the LVA start, [0, junction path end + 3]
The sc* sections share one closed surface (the wall agent's <side>junction_stageA.vtp: subclavian label + the VA's
first 25 mm) and list each other's branches as `siblings` (sdf_union_v4: same-lumen sections do not compete for
ownership). real_surface false (no junction wall yet) -> the stretch is baked as tube from its centreline radii.

Use (neve-build-meshlab), exactly as bake_v5b does with bake_v5:
    import sys; sys.path.insert(0, "/sp/v6/va"); import bake_sections_v6   # installs sections_for_v6
    ... then bake_v5b's driver (its is_va() must also treat the sc* sections as VA sections: branch in
        ("RVA", "LVA", "(17)", "(18)", "(19)") -- a one-line change in the pilot's v6 copy of bake_v5b).
"""
import json
import os
import sys

sys.path.insert(0, "/sp/v5_va")
import bake_v5  # noqa: E402  (installs sections_for_v5: v4's sections + vaC_R / vaC_L)

B = bake_v5.B
from sdf_union import load_surface  # noqa: E402
from sdf_union_v4 import Section  # noqa: E402

_V5 = bake_v5.sections_for_v5
SC_KEYS = ("scC_R", "scD_R", "scC_L", "scD_L", "scJ_L")


def sections_for_v6(root, br):
    secs, spans = _V5(root, br)
    x = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))["xform"]
    for key in SC_KEYS:
        xf = x.get(key)
        if not xf:
            continue
        if not xf.get("real_surface") or not xf.get("surface"):
            spans[key] = "tube only (%s)" % xf.get("why_tube", "no junction wall")
            continue
        bname = xf["branch"]
        b = B.branch(br, bname)
        p, r, L = B.slice_b(b, float(xf["lo_mm"]), float(xf["hi_mm"]))
        if len(p) < 4:
            continue
        surf = load_surface(xf["surface"], xf["mirror"], xf["R"], xf["origin"], xf["anchor"])
        secs.append(Section(key, surf, p, r, True, True, bname, float(xf["lo_mm"]), tuple(xf.get("siblings") or ())))
        spans[key] = (bname, float(xf["lo_mm"]), float(min(xf["hi_mm"], L)))
    return secs, spans


B.sections_for = sections_for_v6
