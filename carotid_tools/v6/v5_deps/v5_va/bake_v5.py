#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_v5.py (= /sp/v5_va/bake_v5.py in the build containers)
"""Bake v5 prototype anatomies with topbrain_tools/bake_meshes_v4.py UNCHANGED, plus the donor cervical VA
sections that graft_v5_va.py records in provenance.json["xform"]["vaC_R" / "vaC_L"].

bake_meshes_v4.bake_one looks `sections_for` up in its module at call time, so this wrapper replaces that one
function with v4's own sections_for followed by the two new sections:

    vaC_R / vaC_L   branch RVA / LVA, [lo_mm, hi_mm] = [ostium (+ the left V1 skip), splice]
                    surface: the donor's closed lumen (VMR 0248 ct_seg4 stage-A), rigidly transformed by
                    (mirror, R, origin, anchor) exactly as sdf_union.load_surface does for every v4 section;
                    taper at both seams (the parent subclavian at the start, the TopBrain piece at the end);
                    no siblings (it shares no source surface with any other section).

Everything else -- the owned union (sdf_union_v4), MARGINS, the localise / window / prune / shrink ladder,
whole-section fallback, band-edge clamp, clean decimation to 60k / 23.5k, route and enclosure reports -- is
bake_meshes_v4's own code. The undecimated gate-passing union is dumped (--dump-iso) for the VA wall-detail
measure (va_detail_v5.py).

    python3 /sp/v5_va/bake_v5.py --anatomies /repo/carotid_data/va_sources/v5_proto [--only A,B] [-v]
Runs in neve-build-meshlab.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/opt/eve_training/eve_bench")

import bake_meshes_v4 as B  # noqa: E402
from sdf_union import load_surface  # noqa: E402
from sdf_union_v4 import Section  # noqa: E402

_V4_SECTIONS_FOR = B.sections_for


def sections_for_v5(root, br):
    secs, spans = _V4_SECTIONS_FOR(root, br)
    d = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    x = d["xform"]
    for key, bname in (("vaC_R", "RVA"), ("vaC_L", "LVA")):
        xf = x.get(key)
        if not xf:
            continue
        if xf.get("real_surface") is False:
            spans[key] = "tube only (%s)" % xf.get("why_tube", "")
            continue
        b = B.branch(br, bname)
        p, r, L = B.slice_b(b, xf["lo_mm"], xf["hi_mm"])
        if len(p) < 4:
            continue
        surf = load_surface(xf["surface"], xf["mirror"], xf["R"], xf["origin"], xf["anchor"])
        secs.append(Section(key, surf, p, r, True, True, bname, xf["lo_mm"], ()))
        spans[key] = (bname, float(xf["lo_mm"]), float(min(xf["hi_mm"], L)))
    return secs, spans


B.sections_for = sections_for_v5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", default="/repo/carotid_data/va_sources/v5_proto")
    ap.add_argument("--only", default=None)
    ap.add_argument("--spacing", type=float, default=B.SPACING_MM)
    ap.add_argument("--full-tris", type=int, default=60000)
    ap.add_argument("--obj-tris", type=int, default=23500)
    ap.add_argument("--dump-iso", default=None, help="default: <anatomies>/_iso")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    dump = a.dump_iso or os.path.join(a.anatomies, "_iso")
    folders = sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))
    if a.only:
        want = set(a.only.split(","))
        folders = [f for f in folders if os.path.basename(os.path.dirname(f)) in want]
    bad = 0
    for f in folders:
        root = os.path.dirname(f)
        name = os.path.basename(root)
        t0 = time.time()
        print("== %s" % name, flush=True)
        try:
            rep = B.bake_anatomy(root, a.spacing, a.full_tris, a.obj_tris, a.verbose, dump)
        except Exception as e:                                           # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print("FAILED %s: %s" % (type(e).__name__, e), flush=True)
            continue
        o = rep["obj"]
        g = max(c["genus"] for c in o["genus"]) if o["genus"] else -1
        print("%s: obj %d tris, %d comps, genus %.0f, margin %s, RCCA lumen %.2f, LCCA lumen %.2f, outside %d, pass %s, "
              "tube-only %s, suppressed %s, %.0f s" % (
                  name, o["tris"], o["comps"], g, rep["margin_mm"], rep["routes"]["RCCA"].get("lumen_min_mm", -1),
                  rep["routes"]["LCCA"].get("lumen_min_mm", -1), rep["centerline_pts_outside"], rep["pass"],
                  rep["sections_tube_only"], rep["suppressed_stretches_mm"], time.time() - t0), flush=True)
        print("   sections:", json.dumps({k: v for k, v in rep["sections"].items() if k.startswith("va")}), flush=True)
        print("   right_route_detail:", json.dumps(rep["right_route_detail"]), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
