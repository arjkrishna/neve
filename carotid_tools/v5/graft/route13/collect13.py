#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\collect13.py (= /sp/v5_va/route13/collect13.py in the build containers)
"""v5_route13: collect the 13 anatomies' variants (v5 as baked, moved_<tag>, prio[_<tag>]) into
/sp/v5_va/results/v5_route13.json and pick one per anatomy:
    moved   if right_route_detail >= 0.99 of v4 on fork AND siphon (> 0.3 mm) and every bake gate passes
    else route-priority (prio, or prio on the moved course 'prio_<tag>' where run), the best route among those
    with every gate; if none passes, keep v5 as baked (flagged).
    python3 collect13.py [--moved-tags ext,tb] [--prio-tags prio,prio_ext]        (pure json; any python3)
"""
import argparse
import json
import os
import time

R13 = "/repo/carotid_data/va_sources/v5_reg/route13"
BF5 = "/repo/carotid_data/va_sources/v5_reg/bake_full/_done"
A4 = "/repo/carotid_data/anatomies_v4"
A5 = "/repo/carotid_data/anatomies_v5"
SP = "/sp/v5_va/route13"
OUT = "/sp/v5_va/results/v5_route13.json"
THR = 0.99


def jl(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def summarise(meas, bake, folder):
    if meas is None:
        return None
    rr = meas["right_route_detail"]
    gt = meas["gates"]
    va = {}
    for sd, v in (meas.get("va_detail") or {}).items():
        va[sd] = {k: v.get(k) for k in ("donor", "real_surface", "whole_tube", "tubed_mm", "tubed_frac", "n_windows",
                                        "donor_real_frac_0.3mm", "donor_real_frac_0.5mm", "topbrain_frac_0.3mm",
                                        "whole_va_conservative_frac_0.3mm", "A2_detail_kept_in_situ_over_alone",
                                        "v4_va_kept")}
    return {"folder": folder,
            "route_ratio": {seg: {t: rr["ratio"][seg][t] for t in rr["ratio"][seg]} for seg in rr["ratio"]},
            "route_v5": rr["v5"], "route_v4": rr["v4"], "route_identical_v4": rr["identical"],
            "route_identical_beyond_mm": rr.get("identical_beyond_mm"),
            "rcca_lumen_min_mm": g(gt, "route_lumen_min_mm", "RCCA"), "lcca_lumen_min_mm": g(gt, "route_lumen_min_mm", "LCCA"),
            "gates": {k: gt.get(k) for k in ("all_pass", "union_clean_tree", "full_clean_tree", "obj_clean_tree",
                                             "self_intersecting_faces", "obj_tris", "full_tris", "obj_budget_ok",
                                             "full_budget_ok", "routes_navigable", "centerline_pts_outside", "bake_pass",
                                             "rung", "flags")},
            "va_detail": va,
            "bake_seconds": (bake or {}).get("seconds"), "measure_seconds": meas.get("seconds")}


def ok_route(s):
    if not s or not s["gates"]["all_pass"]:
        return False
    return all((s["route_ratio"][seg]["frac_0.3mm"] or 0) >= THR for seg in ("fork", "siphon"))


def read_pts(path):
    d = jl(path)
    return [cp["position"] for cp in d["markups"][0]["controlPoints"]] if d else None


def arcl(P):
    s = [0.0]
    for a, b in zip(P[:-1], P[1:]):
        s.append(s[-1] + sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)
    return s


def shift_stats(new, old, lo, hi):
    """Nearest-point distance from the new curve's points with arclength in [lo, hi] to the old curve (mm)."""
    s = arcl(new)
    d = []
    for p, si in zip(new, s):
        if lo <= si <= hi:
            d.append(min(sum((x - y) ** 2 for x, y in zip(p, q)) for q in old) ** 0.5)
    if not d:
        return None
    return {"rms_mm": round((sum(x * x for x in d) / len(d)) ** 0.5, 2), "max_mm": round(max(d), 2),
            "mean_mm": round(sum(d) / len(d), 2)}


def prov_check(name, folder):
    """Same donor pair / assignment / licences as anatomies_v5; every non-VA xform entry unchanged."""
    a = jl(os.path.join(folder, "provenance.json"))
    b = jl(os.path.join(A5, name, "provenance.json"))
    if a is None or b is None:
        return None
    va, vb = a["v5_va"], b["v5_va"]
    same_donor = all((va.get(sd) or {}).get(k) == (vb.get(sd) or {}).get(k) for sd in ("right", "left")
                     for k in ("donor_id", "donor_side", "donor_mirror", "replaced", "v4_va_kept"))
    xa = {k: v for k, v in a["xform"].items() if not k.startswith("va")}
    xb = {k: v for k, v in b["xform"].items() if not k.startswith("va")}
    top = sorted(k for k in set(a) | set(b) if k not in ("v5_va", "xform", "bake_v5") and a.get(k) != b.get(k))
    return {"same_donor_pair_and_sides": same_donor, "licences_same": va.get("licences") == vb.get("licences"),
            "assignment_same": va.get("assignment") == vb.get("assignment"),
            "summary_attributions_same": (va.get("summary") or {}).get("attributions") == (vb.get("summary") or {}).get("attributions"),
            "non_commercial_same": (va.get("summary") or {}).get("non_commercial") == (vb.get("summary") or {}).get("non_commercial"),
            "non_va_xform_same": xa == xb, "other_top_level_keys_differing": top,
            "route13_record": bool(va.get("route13")), "bake_v5_script": (a.get("bake_v5") or {}).get("script")}


def course(name, tag):
    rec = jl(os.path.join(SP, "graft", tag, name + ".json"))
    if rec is None:
        return None
    p5 = jl(os.path.join(A5, name, "provenance.json"))["v5_va"]
    out = {}
    for sd in ("right", "left"):
        s = rec["sides"][sd]
        sep = s.get("separation") or {}
        v = p5.get(sd) or {}
        fn = "Centerline curve - %s.mrk.json" % ("RVA" if sd == "right" else "LVA")
        new = read_pts(os.path.join(R13, "moved_" + tag, name, "Centrelines_comb", fn))
        old = read_pts(os.path.join(A5, name, "Centrelines_comb", fn))
        sh = {}
        if new and old and not sep.get("identical_to_anatomies_v5") and s.get("ostium_at_mm") is not None:
            sh = {"donor_stretch": shift_stats(new, old, s["ostium_at_mm"], s["cut_at_mm"]),
                  "topbrain_piece": shift_stats(new, old, s["cut_at_mm"], 1e9)}
        out[sd] = {"target": sep.get("target"), "identical_to_v5": sep.get("identical_to_anatomies_v5"),
                   "shift_vs_v5_nearest": sh,
                   "pass_all_graft_gates": s.get("pass"),
                   "pose": [s.get("pose_deg"), s.get("pose_axis")], "pose_v5": [v.get("pose_deg"), v.get("pose_axis")],
                   "splice_move_mm": s.get("splice_move_mm"), "splice_move_mm_v5": v.get("splice_move_mm"),
                   "course_shift_vs_v5_rms_mm": sep.get("course_shift_vs_v5_rms_mm"),
                   "course_shift_vs_v5_max_mm": sep.get("course_shift_vs_v5_max_mm"),
                   "course_rms_to_v4_corridor_mm": s.get("course_rms_to_corridor_mm"),
                   "course_rms_to_v4_corridor_mm_v5": v.get("course_rms_to_corridor_mm"),
                   "takeoff_10mm_deg": s.get("takeoff_10mm_deg"), "takeoff_10mm_deg_v5": v.get("takeoff_10mm_deg"),
                   "donor_takeoff_10mm_native_deg": s.get("donor_takeoff_10mm_deg_native"),
                   "tip_moved_vs_v4_mm": s.get("tip_moved_vs_v4_mm"), "join_kink_deg": s.get("join_kink_deg"),
                   "final_clearance_mm": g(s, "final", "clear"), "final_exact_mm": g(s, "final", "exact_all_pairs_from_exit_mm"),
                   "lost_v4_rcca_voxels_chosen": g(sep, "chosen", "lost"),
                   "lost_v4_rcca_voxels_v5": g(sep, "lost_as_in_anatomies_v5", "lost"),
                   "closest_rcca_chosen": sep.get("closest_rcca_chosen"), "closest_rcca_v5": sep.get("closest_rcca_v5"),
                   "candidates_passing_every_gate": sep.get("candidates_passing_every_gate"),
                   "pose_family": sep.get("pose_family"), "tb_shift_deg_axis": sep.get("tb_shift_deg_axis"),
                   "other_va_had_to_move": sep.get("chosen_other_va_must_move")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moved-tags", default="ext")
    ap.add_argument("--prio-tags", default="prio")
    a = ap.parse_args()
    names = [l.strip() for l in open(os.path.join(SP, "names13.txt")) if l.strip()]
    lv = jl(os.path.join(SP, "lost_variants.json")) or {}
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": "moved if fork and siphon >= %.2f of v4 (frac > 0.3 mm) "
           "with every bake gate; else route-priority (best route with every gate); else v5 kept" % THR,
           "anatomies": {}}
    for n in names:
        rep4 = jl(os.path.join(A4, n, "mesh_v4.json"))
        v5 = summarise(jl(os.path.join(BF5, n + ".measure.json")), jl(os.path.join(BF5, n + ".bake.json")), "anatomies_v5")
        diag = jl(os.path.join(SP, "diag", n + ".json"))
        ent = {"v4_rcca_lumen_min_mm": g(rep4, "routes", "RCCA", "lumen_min_mm"), "v5_as_baked": v5,
               "v4_kept_rcca_real_voxels_refused": lv.get(n),
               "diagnosis": diag and {sec: {k: v for k, v in s.items() if k in (
                   "use4", "lost_v4_wall_voxels", "lost_by_ownership", "lost_by_v5_suppression", "gained_voxels",
                   "competitors_v5")} for sec, s in diag["sections"].items()},
               "variants": {}}
        for t in [x for x in a.moved_tags.split(",") if x]:
            v = "moved_" + t
            s = summarise(jl(os.path.join(R13, v + "_bf", "_done", n + ".measure.json")),
                          jl(os.path.join(R13, v + "_bf", "_done", n + ".bake.json")), os.path.join(R13, v, n))
            if s is not None or os.path.isfile(os.path.join(SP, "graft", t, n + ".json")):
                ent["variants"][v] = dict(s or {}, course=course(n, t), kind="moved",
                                          baked=s is not None, provenance_check=prov_check(n, os.path.join(R13, v, n)))
        for v in [x for x in a.prio_tags.split(",") if x]:
            s = summarise(jl(os.path.join(R13, v + "_bf", "_done", n + ".measure.json")),
                          jl(os.path.join(R13, v + "_bf", "_done", n + ".bake.json")), os.path.join(R13, v, n))
            if s is not None:
                ent["variants"][v] = dict(s, kind="route_priority")
        # pick
        moved_ok = [(k, s) for k, s in ent["variants"].items() if s.get("kind") == "moved" and s.get("gates") and ok_route(s)]

        def score(s):
            return min(s["route_ratio"]["fork"]["frac_0.3mm"] or 0, s["route_ratio"]["siphon"]["frac_0.3mm"] or 0)
        if moved_ok:
            def shift(s_):
                c = s_.get("course") or {}
                return sum(((c.get(sd) or {}).get("shift_vs_v5_nearest") or {}).get("donor_stretch", {}).get("rms_mm", 0.0)
                           or 0.0 for sd in ("right", "left"))
            k, s = max(moved_ok, key=lambda t: (round(score(t[1]), 4), -shift(t[1])))
            ent["chosen"] = {"variant": k, "why": "moved: route >= %.2f of v4 on fork and siphon, every gate" % THR}
        else:
            # no moved variant reaches the threshold: the best route among every variant with every gate (route
            # priority included), v5 as baked if nothing beats it
            pool = [(k, s) for k, s in ent["variants"].items() if s.get("gates", {}).get("all_pass")]
            if v5 and v5["gates"]["all_pass"]:
                pool.append(("v5_as_baked", v5))
            if pool:
                k, s = max(pool, key=lambda t: score(t[1]))
                ent["chosen"] = {"variant": k if k != "v5_as_baked" else None,
                                 "why": "no moved variant reached %.2f on both with every gate; best route of the rest "
                                        "(%s, min ratio %.4f); route priority %s" % (
                                            THR, k, score(s), "chosen" if ent["variants"].get(k, {}).get("kind") == "route_priority"
                                            else "does not raise the measure here")}
            else:
                ent["chosen"] = {"variant": None, "why": "no variant passes; keep v5 as baked (flag)"}
        if ent["chosen"]["variant"]:
            ent["chosen"]["folder_ready_to_swap"] = ent["variants"][ent["chosen"]["variant"]]["folder"]
            ent["chosen"]["iso_undecimated"] = os.path.join(R13, ent["chosen"]["variant"] + "_iso", n + ".vtp")
            ent["chosen"]["route_ratio"] = ent["variants"][ent["chosen"]["variant"]]["route_ratio"]
        # route min lumen on the UNDECIMATED unions (lumen_iso.py); mesh_v5.json's lumen is on the decimated OBJ
        li = (jl(os.path.join(SP, "lumen_iso.json")) or {}).get(n)
        if li:
            ent["route_lumen_undecimated_mm"] = li
        res["anatomies"][n] = ent
    ch = [e["chosen"]["variant"] for e in res["anatomies"].values()]
    res["summary"] = {
        "n": len(ch), "chosen_counts": {k: ch.count(k) for k in sorted(set(map(str, ch)))},
        "all_chosen_pass_gates_and_route": all(
            ok_route(e["variants"].get(e["chosen"]["variant"])) for e in res["anatomies"].values() if e["chosen"]["variant"]),
        "route_priority_chosen": sum(1 for e in res["anatomies"].values()
                                     if (e["variants"].get(e["chosen"]["variant"] or "") or {}).get("kind") == "route_priority"),
        "ownership_reach_rule": "a v4-kept RCCA real voxel x stays owned iff f_local(x) - f_VA(x) >= m (sdf_union_v4."
                                "ownership_margin; m = 0.5 mm on all 13; f = r - |x - c| over the VA's 16 nearest 0.25 mm "
                                "samples): the VA tube wall must keep (RCCA real-wall stand-out beyond its tube + m) from x "
                                "= 6-8 mm from the RCCA tube wall at the bulb, 5-6 mm at the siphon",
        "separation_objective": "graft_v5_reg_sep.py S1-S7: among candidates passing EVERY graft_v5_reg gate, the fewest "
                                "v4-kept RCCA real voxels refused (exact rule), ties -> the original order",
        "route_priority": "bake_v5c.py R1-R3 (run on the v5 course; never raises right_route_detail: the measure "
                          "attributes wall cells by the same tube bisector, regained wall counts as VA wall)",
        "tools": {"diagnosis": SP + "/diag13.py", "placement": "/sp/v5_va/graft_v5_reg_sep.py",
                  "route_priority_bake": "/sp/v5_va/bake_v5c.py", "bake": "/sp/v5_va/bake_v5b.py via bake_full/bake_full_one.py",
                  "measure": "/sp/v5_va/bake_full/measure_full_one.py", "lumen_undecimated": SP + "/lumen_iso.py"}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    for n, e in res["anatomies"].items():
        row = [n[5:40]]
        for k, s in [("v5", e["v5_as_baked"])] + list(e["variants"].items()):
            if s and s.get("route_ratio"):
                row.append("%s f%.3f s%.3f %s" % (k, s["route_ratio"]["fork"]["frac_0.3mm"], s["route_ratio"]["siphon"]["frac_0.3mm"],
                                                 "ok" if s["gates"]["all_pass"] else "GATEFAIL"))
        row.append("-> %s" % e["chosen"]["variant"])
        print(" | ".join(row))
    return 0


if __name__ == "__main__":
    main()
