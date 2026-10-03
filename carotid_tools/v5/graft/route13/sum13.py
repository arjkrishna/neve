#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\sum13.py (= /sp/v5_va/route13/sum13.py in the build containers)
"""v5_route13: compact per-anatomy table from results/v5_route13.json (chosen variant vs v5 as baked; prio where run)."""
import json

R = json.load(open("/sp/v5_va/results/v5_route13.json"))


def rr(s):
    if not s:
        return "-"
    r = s["route_ratio"]
    return "f%.4f/%.4f s%.4f/%.4f" % (r["fork"]["frac_0.3mm"], r["fork"]["frac_0.5mm"], r["siphon"]["frac_0.3mm"],
                                      r["siphon"]["frac_0.5mm"])


def va(s):
    out = []
    for sd in ("right", "left"):
        v = (s.get("va_detail") or {}).get(sd) or {}
        out.append("%s:%s A2=%s tub=%s/%s d03=%s" % (sd[0], v.get("donor"), v.get("A2_detail_kept_in_situ_over_alone"),
                                                   v.get("tubed_mm"), v.get("n_windows"), v.get("donor_real_frac_0.3mm")))
    return " ".join(out)


for n, e in R["anatomies"].items():
    v5 = e["v5_as_baked"]
    ch = e["chosen"]["variant"]
    s = e["variants"].get(ch) if ch else None
    print("=" * 8, n, "->", ch)
    print("  lumen v4 %s v5 %s ch %s | LCCA v5 %s ch %s" % (e["v4_rcca_lumen_min_mm"], v5["rcca_lumen_min_mm"],
                                                           s and s["rcca_lumen_min_mm"], v5["lcca_lumen_min_mm"],
                                                           s and s["lcca_lumen_min_mm"]))
    print("  v5   ", rr(v5), "rung", v5["gates"]["rung"], "flags", v5["gates"]["flags"])
    for k, x in e["variants"].items():
        if not x.get("route_ratio"):
            print("  %-9s (not baked)" % k)
            continue
        g = x["gates"]
        print("  %-9s %s all=%s rung=%s flags=%s tris=%s/%s si=%s ident=%s" % (
            k, rr(x), g["all_pass"], g["rung"], g["flags"], g["obj_tris"], g["full_tris"], g["self_intersecting_faces"],
            x.get("route_identical_v4")))
        print("            VA", va(x))
    print("  v5 VA", va(v5))
    if s and s.get("course"):
        for sd, c in s["course"].items():
            print("  course %s: tgt=%s ident=%s gates=%s pose=%s v5=%s splice=%s(v5 %s) rmsCorr=%s(v5 %s) take=%s(v5 %s, donor %s) "
                  "tip=%s kink=%s clr=%s lost=%s(v5 %s) cand=%s tb=%s other=%s shift=%s" % (
                      sd, c["target"], c["identical_to_v5"], c["pass_all_graft_gates"], c["pose"], c["pose_v5"],
                      c["splice_move_mm"], c["splice_move_mm_v5"], c["course_rms_to_v4_corridor_mm"],
                      c["course_rms_to_v4_corridor_mm_v5"], c["takeoff_10mm_deg"], c["takeoff_10mm_deg_v5"],
                      c["donor_takeoff_10mm_native_deg"], c["tip_moved_vs_v4_mm"], c["join_kink_deg"],
                      c["final_clearance_mm"], c["lost_v4_rcca_voxels_chosen"], c["lost_v4_rcca_voxels_v5"],
                      c["candidates_passing_every_gate"], c["tb_shift_deg_axis"], c["other_va_had_to_move"],
                      json.dumps(c["shift_vs_v5_nearest"])))
        pc = s.get("provenance_check")
        print("  prov", json.dumps(pc))
