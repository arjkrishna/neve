#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records$1 (= /sp/v6/records/build_v6_record.py in the build containers)
"""s2_records: BUILD_v6.json for carotid_data/anatomies_v6, generated from the FINAL per-anatomy records only.

Every per-anatomy count, list and range is derived from the files in each anatomy folder of anatomies_v6:
provenance.json (the composer / VA tool / bake / checks records, v6_checks stamped by the build), mesh_v6.json (the bake
report) and the LICENSE files. Nothing is copied from assignment-time, dry-run or collection files. The one exception is
the block "checks_final": set-level results that no single anatomy can hold (split, training loader over the folder,
SOFA, D14, the orientation / ICA re-measure on the whole tree), copied from the set-checks result file
_records/results/v6_final_summary.json and labelled as such. check_build_record.py recounts every count from provenance
with separate code and compares.

    python3 build_v6_record.py [--root /repo/carotid_data/anatomies_v6] [--final /sp/v6/results/v6_final_summary.json]
                               [--out /sp/v6/records/out/BUILD_v6.json]
"""
import argparse
import collections
import hashlib
import json
import os
import time

import numpy as np

VERSION = "build_v6_record 1.2 (s4_records, 2026-10-08; 1.1 s3_fix 2026-10-08; 1.0 s2_records 2026-10-07)"
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/repo/carotid_data/anatomies_v6")
ap.add_argument("--v5root", default="/repo/carotid_data/anatomies_v5")
ap.add_argument("--final", default="/sp/v6/results/v6_final_summary.json")
ap.add_argument("--out", default="/sp/v6/records/out/BUILD_v6.json")
# 1.1 (s3_fix): the set-level coverage result (checks_v6 1.6 on all 223) and the per-anatomy checks records of the
# anatomies s3_fix did not re-stamp (for the orient:* derived flags only)
ap.add_argument("--coverage", default="/sp/v6/results/v6_coverage_s3.json")
ap.add_argument("--checks4", default="/repo/carotid_data/anatomies_v6/_records/per_anatomy/checks4")
ap.add_argument("--s3set", default="/sp/v6/results/v6_s3_set.json")   # s3_fix set-level checks after the fix (labelled copy)
a = ap.parse_args()

RZ = np.diag([-1.0, -1.0, 1.0])


def jl(p):
    with open(p) as f:
        return json.load(f)


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def st(v, nd=3):
    v = sorted(float(x) for x in v if num(x))
    if not v:
        return {"n": 0}
    q = lambda f: round(float(np.percentile(v, f)), nd)
    return {"n": len(v), "min": round(v[0], nd), "p10": q(10), "median": q(50), "p90": q(90), "max": round(v[-1], nd)}


def short(n):
    return n.replace("case_", "").replace("__topcow_", " ")


names = sorted(d for d in os.listdir(a.root) if d.startswith("case_") and os.path.isdir(os.path.join(a.root, d)))
names_v5 = sorted(d for d in os.listdir(a.v5root) if d.startswith("case_") and os.path.isdir(os.path.join(a.v5root, d)))

C = collections.Counter()            # every count; keys are documented by their names
L = collections.defaultdict(list)    # lists of anatomies (short names) / details
S = collections.defaultdict(list)    # values for ranges
FILES = collections.Counter()
UNIT = collections.defaultdict(lambda: collections.Counter())
UNIT_PAT = collections.defaultdict(set)
PAT_UNIT = collections.defaultdict(set)
PAT_N = collections.Counter()
PAT_HAND = collections.defaultdict(collections.Counter)
UNIT_SRC = {}
UNIT_LIC = {}
TOOLS = collections.defaultdict(collections.Counter)
PER = {}
digest_lines = []

for n in names:
    d = os.path.join(a.root, n)
    pf = os.path.join(d, "provenance.json")
    p = jl(pf)
    m = jl(os.path.join(d, "mesh_v6.json"))
    files = sorted(os.listdir(d))
    FILES[",".join(files)] += 1
    digest_lines.append("%s %s\n" % (n, sha256f(pf)))
    sn = short(n)
    b = p["v6_build"]
    car = b["carotid"]
    rf = car["right_fork"]
    lf = car["left_fork"]
    vc = p["v6_carotid"]
    va = b["va"]
    bk = p["bake_v6"]
    ck = p["v6_checks"]
    pat = p.get("patient")
    PAT_N[pat] += 1
    PAT_HAND[pat][p.get("handedness")] += 1
    C["anatomies"] += 1
    C["handedness:%s" % p.get("handedness")] += 1
    # ---- toolchain / versions
    TOOLS["compose_manifest_md5"][(b.get("toolchain") or {}).get("manifest_md5")] += 1
    TOOLS["checks_manifest_md5"][ck.get("toolchain_manifest_md5")] += 1
    TOOLS["build_v6"][str(b.get("version", "")).split(" (")[0]] += 1
    for k, v in (b.get("tools") or {}).items():
        TOOLS[k][str(v).split(" (")[0]] += 1
    TOOLS["bake_v6"][str(bk.get("script", "")).split(" (")[0]] += 1
    TOOLS["checks_v6"][ck.get("checks_v6")] += 1
    for part, ver in (ck.get("part_versions") or {}).items():
        TOOLS["checks_part:%s" % part][ver] += 1
    TOOLS["compose_source_stage"][b.get("source")] += 1
    # ---- right fork (D3-fork / D19 / D20 / D24)
    C["right_fork_rung:%s" % rf.get("right_fork_rung")] += 1
    C["right_fork_pose:%s" % rf.get("right_fork_pose")] += 1
    C["right_fork_placement:%s" % rf.get("right_fork_placement")] += 1
    C["right_fork_mirrored_D20"] += bool(rf.get("right_fork_mirrored"))
    C["right_fork_contralateral"] += bool(rf.get("right_fork_contralateral"))
    C["right_host_cut_shifted"] += bool(abs(rf.get("host_cut_shift_mm") or 0.0) > 1e-9)
    S["right_seam1_bend_deg"].append(rf.get("right_seam1_bend_deg"))
    S["right_host_cut_shift_mm"].append(rf.get("host_cut_shift_mm"))
    if (rf.get("right_seam1_bend_deg") or 0) > 45:
        L["right_seam1_bend_gt45"].append([sn, rf.get("right_seam1_bend_deg"), rf.get("right_fork_pose")])
    if rf.get("right_fork_rung") in (3, 4, 5):
        L["right_fork_R%s" % rf.get("right_fork_rung")].append([sn, rf.get("right_fork_pose"), rf.get("right_seam1_bend_deg"),
                                                             rf.get("right_dev_deg")])
    # ---- left fork (D1 / D11 / D20)
    C["left_fork_decision:%s" % lf.get("left_fork_decision")] += 1
    C["left_fork_rung:%s" % lf.get("fork_rung")] += 1
    C["left_fork_pose:%s" % lf.get("left_fork_pose")] += 1
    C["left_fork_mirrored_D20"] += bool(lf.get("left_fork_mirrored"))
    C["left_fork_contralateral"] += bool(lf.get("left_fork_contralateral"))
    C["left_fork_ica_tubed"] += bool((lf.get("tubed_ica_mm") or 0) > 0)
    S["left_seam1_bend_deg"].append(lf.get("left_seam1_bend_deg"))
    if lf.get("left_fork_decision") == "substituted":
        L["left_fork_substituted"].append([sn, (vc.get("left_fork") or {}).get("lower_v5"), lf.get("left_fork_lower"),
                                           (vc.get("left_fork") or {}).get("rung_label"),
                                           (vc.get("left_fork") or {}).get("why_v5_fork_failed")])
    # map deviation of the left fork, from provenance only: xform.lower_L.R against the per-patient Zenodo map
    # (I for an LPS-like frame, Rz(180) for RAS-like) recorded in v6_carotid.left_fork.map.frame
    fr = ((vc.get("left_fork") or {}).get("map") or {}).get("frame")
    xl = (p.get("xform") or {}).get("lower_L") or {}
    ldev = None
    if fr in ("LPS", "RAS") and "R" in xl:
        A0 = np.eye(3) if fr == "LPS" else RZ
        Rm = np.asarray(xl["R"], float)
        ldev = round(float(np.degrees(np.arccos(np.clip((np.trace(Rm @ A0.T) - 1) / 2, -1, 1)))), 3)
    if lf.get("left_fork_pose") == "zenodo_map":
        if ldev is not None and ldev >= 0.01:
            L["left_pose_map_bounded_dev"].append([sn, ldev])
            S["left_pose_map_dev_deg"].append(ldev)
    else:
        L["left_pose_not_map"].append([sn, lf.get("left_fork_pose"),
                                       (vc.get("summary") or {}).get("left_fork_pose_why_not_map")])
    # ---- ICA terminus AP (D11), ECA tips (D4), route baseline (D13)
    ap_ = car.get("ica_ap_post_mm") or {}
    for s_ in ("RICA", "LICA"):
        S["ica_ap_%s_mm" % s_].append(ap_.get(s_))
    if car.get("ica_ap_flag_R") or car.get("ica_ap_flag_L"):
        L["ica_ap_gt40"].append([sn, ap_.get("RICA"), ap_.get("LICA")])
    C["ica_ap_gt40_R"] += bool(car.get("ica_ap_flag_R"))
    C["ica_ap_gt40_L"] += bool(car.get("ica_ap_flag_L"))
    C["eca_right_mode:%s" % car.get("eca_right")] += 1
    for t in car.get("skipped_eca_tips") or []:
        C["eca_tip_skipped:%s" % t.get("side")] += 1
    C["route_baseline:%s" % car.get("route_baseline")] += 1
    if b.get("route13"):
        L["route13_moved"].append([sn, b.get("route13")])
    if b.get("fix_loop"):
        L["fix_loop"].append(sn)
    if (b.get("c4") or {}).get("applied_planned"):
        L["c4_planned"].append(sn)
    if b.get("d23_accepted"):
        L["d23_accepted"].append([sn, b["d23_accepted"].get("pieces"), b["d23_accepted"].get("donor_id")])
    # ---- VA (D2 / D8-D10 / D16-D18 / D21 / D25)
    assigned = (p.get("v5_va") or {}).get("assignment", {}).get("assigned_units") or []
    for u in assigned:
        UNIT_PAT[u].add(pat)
        PAT_UNIT[pat].add(u)
        UNIT[u]["anatomies"] += 1
    for side, key in (("right", "R"), ("left", "L")):
        v = va[side]
        rung = v.get("rung")
        C["va_%s_rung:%s" % (side, rung)] += 1
        C["va_%s_junction:%s" % (side, v.get("junction_type"))] += 1
        did = v.get("donor_id")
        if did:
            C["va_%s_donor" % side] += 1
            UNIT[did]["%s_sides" % ("rva" if side == "right" else "lva")] += 1
            UNIT[did]["rung%s" % rung] += 1
            xc = (p.get("xform") or {}).get("vaC_" + key) or {}
            src = str(xc.get("kind", ""))
            UNIT_SRC.setdefault(did, src.split("(")[1].split(",")[0] if "(" in src else None)
            lic = ((p.get("v5_va") or {}).get("licences") or {}).get(did)
            if lic:
                UNIT_LIC.setdefault(did, lic)
        else:
            L["va_host_no_donor_D21"].append([sn, side])
        if rung == 2:
            L["va_d18_v5_junction"].append([sn, side, did, [f for f in v.get("flags") or [] if f != "d9_rung2_v5_junction"]])
        if rung == 3:
            L["va_d9_rung3_v5_top"].append([sn, side, did, v.get("flags")])
        for f in v.get("flags") or []:
            C["va_%s_flag:%s" % (side, f)] += 1
        C["va_%s_target_band:%s" % (side, v.get("va_target_band"))] += 1
        if did:
            S["va_target_mm"].append(v.get("va_target_mm"))
            S["course_rms_donor_mm"].append(v.get("course_rms_donor_mm"))
            if rung == 1:
                S["takeoff_error_vs_native_deg_abs_rung1"].append(abs(v["takeoff_error_vs_native_deg"])
                                                                  if num(v.get("takeoff_error_vs_native_deg")) else None)
                d17 = v.get("d17") or {}
                S["d17_past_ostium_mm"].append(d17.get("past_ostium_mm"))
                S["d17_host_distal_dropped_mm"].append(d17.get("host_distal_dropped_mm"))
            if v.get("d16_tube_mm"):
                L["va_d16_tube"].append([sn, side, did, v.get("d16_tube_mm")])
            if "d17_tube" in (v.get("flags") or []):
                pos = (((p.get("v6_va") or {}).get("record") or {}).get("sides") or {}).get(side, {}).get("d17", {}).get("positions") or {}
                L["va_d17_tube"].append([sn, side, did, round(pos.get("tube_mm") or 0.0, 3)])
            cd = v.get("corridor_d10") or {}
            if num(cd.get("dev_mm")):
                S["corridor_d10_dev_mm"].append(abs(cd["dev_mm"]))
    # ---- bake (one bake per anatomy, bake_v6)
    C["bake_rung:%s" % bk.get("rung")] += 1
    C["bake_pass"] += bool(bk.get("pass")) and bool(m.get("pass"))
    C["bake_carotid_margin:%s" % bk.get("carotid_margin_mm")] += 1
    C["bake_va_margin:%s" % bk.get("va_margin_mm")] += 1
    C["bake_far_clamp_used"] += bool((bk.get("far_clamp") or {}).get("used"))
    if (bk.get("far_clamp") or {}).get("used"):
        L["bake_far_clamp"].append([sn, (bk.get("far_clamp") or {}).get("why")])
    C["bake_sections_tube_only"] += len(bk.get("sections_tube_only") or [])
    for s_ in bk.get("sections_tube_only") or []:
        L["bake_sections_tube_only"].append([sn, s_])
    if (bk.get("c4") or {}).get("applied"):
        L["bake_c4_applied"].append(sn)
    C["bake_e7:%s" % (bk.get("e7") or {}).get("result")] += 1
    for f in bk.get("flags") or []:
        C["bake_flag:%s" % f] += 1
    S["bake_seconds"].append(bk.get("seconds"))
    S["obj_tris"].append((m.get("obj") or {}).get("tris"))
    S["full_tris"].append((m.get("full") or {}).get("tris"))
    for r_ in ("RCCA", "LCCA"):
        S["route_lumen_min_mm_%s" % r_].append(((m.get("routes") or {}).get(r_) or {}).get("lumen_min_mm"))
        C["route_navigable_%s" % r_] += bool(((m.get("routes") or {}).get(r_) or {}).get("navigable"))
    C["centreline_pts_outside_total"] += int(m.get("centerline_pts_outside") or 0)
    C["obj_genus0_closed"] += all((g.get("genus") == 0 and g.get("boundary_loops") == 0) for g in (m.get("obj") or {}).get("genus") or [{}])
    C["full_genus0_closed"] += all((g.get("genus") == 0 and g.get("boundary_loops") == 0) for g in (m.get("full") or {}).get("genus") or [{}])
    win_secs = {k: v for k, v in (bk.get("va_tubed") or {}).items() if v.get("n_windows")}
    if win_secs:
        C["bake_anatomies_with_tube_windows"] += 1
    for k, v in win_secs.items():
        C["bake_window_section:%s" % k] += 1
        S["bake_window_tubed_mm"].append(v.get("tubed_mm"))
    for k, v in (bk.get("va_tubed") or {}).items():
        if v.get("whole_tube"):
            L["bake_whole_tube_sections"].append([sn, k])
    for k, v in (bk.get("chirality") or {}).items():
        if v.get("mirrored"):
            C["bake_sections_mirrored"] += 1
            C["bake_sections_mirrored_outward"] += bool(v.get("outward"))
    # ---- checks (checks_v6, stamped as v6_checks)
    C["checks_pass"] += bool(ck.get("pass"))
    for g, ok in (ck.get("gates") or {}).items():
        C["gate_pass:%s" % g] += bool(ok)
    C["checks_d22_only_rerun"] += bool(ck.get("d22_only_rerun"))
    C["d22_baseline:%s" % ck.get("d22_baseline")] += 1
    d22 = ck.get("d22") or {}
    for g in ("fork", "siphon"):
        x = d22.get(g) or {}
        C["d22_%s_reference:%s" % (g, x.get("reference"))] += 1
        S["d22_%s_value" % g].append(x.get("value"))
        S["d22_%s_value_v4" % g].append(x.get("value_v4"))
        S["d22_%s_coverage" % g].append(x.get("coverage"))
        if x.get("d26_flag"):
            L["d26_own_surface"].append([sn, g, x.get("value_v4"), x.get("value_own"), x.get("coverage")])
        if x.get("d27_flag"):
            rw = x.get("d22_repair_window") or {}
            L["d27_repair_window"].append([sn, g, x.get("value_before_d27"), x.get("value"), rw.get("cells"), rw.get("area_mm2"),
                                           rw.get("s_range_mm"), rw.get("clipped_in_v4_world_frac")])
        if num(x.get("coverage")) and x["coverage"] < 0.99:
            L["d22_coverage_lt_0.99"].append([sn, g, x.get("reference"), x.get("coverage")])
        C["d22_%s_no_wall_lost" % g] += bool(x.get("no_wall_lost"))
    # ---- flags as recorded
    for f in b.get("flags") or []:
        parts = f.split(":")
        key = ":".join(parts[:2]) if f.startswith(("va_right:", "va_left:")) else f
        C["flag_compose:%s" % key] += 1
    for f in ck.get("flags") or []:
        key = f.split("_repair_window")[0] + "_repair_window" if "_repair_window" in f else f
        C["flag_checks:%s" % key] += 1
    for f in bk.get("flags") or []:
        C["flag_bake:%s" % f] += 1
    if (b.get("flags") or []) or (ck.get("flags") or []) or (bk.get("flags") or []):
        C["anatomies_with_any_flag"] += 1
    # ---- 1.1: derived flags (the flags v6_build_collect derives outside provenance) and the s3_fix record
    if isinstance(ck.get("derived_flags"), list):
        dflags, dsrc = list(ck["derived_flags"]), "provenance v6_checks.derived_flags"
    else:
        dflags, dsrc = [], "provenance (left map deviation) + _records/per_anatomy/checks4 (orient accepted flags)"
        c4p = os.path.join(a.checks4, n + ".json")
        if os.path.exists(c4p):
            dflags += ["orient:%s" % f_ for f_ in ((jl(c4p).get("orient") or {}).get("accepted_flags") or [])]
        if lf.get("left_fork_pose") == "zenodo_map" and ldev is not None and ldev >= 0.01:
            dflags.append("left_pose_map_bounded_dev")
    C["derived_flags_source:%s" % dsrc] += 1
    for f_ in dflags:
        C["flag_derived:%s" % (f_ if not f_.startswith("orient:") else "orient:" + f_.split(":")[1])] += 1
    s3 = ck.get("s3_fix")
    if s3:
        C["s3_fix_kind:%s" % s3.get("kind")] += 1
        sh_ = [float(v) for v in (bk.get("lattice_shift_mm") or [0, 0, 0])]
        rib = bk.get("route_island_bridge") or {}
        L["s3_fix"].append({"anatomy": sn, "kind": s3.get("kind"), "lattice_shift_mm": sh_ if any(sh_) else None,
                            "route_island_bridges": [[b_.get("section"), b_.get("rcca_s"), b_.get("volume_mm3"), b_.get("length_mm")]
                                                     for b_ in (rib.get("bridged") or [])] or None,
                            "obj_fold_fix": (bk.get("obj_fold_fix") or {}).get("result"),
                            "coverage": ck.get("coverage_v4_v5_shipped"), "flags": [f_ for f_ in ck.get("flags") or []
                                                                                  if f_.startswith(("d27_coverage", "d27_bake", "obj_fold_fix"))]})
        C["s3_bake_lattice_shift"] += bool(any(sh_))
        C["s3_bake_route_island_bridge"] += bool(rib.get("bridged"))
        C["s3_route_islands_bridged"] += len(rib.get("bridged") or [])
    # ---- licence (LICENSE files in the folder, cross-read with provenance)
    lic_txt = open(os.path.join(d, "LICENSE_v6_donors.txt")).read() if "LICENSE_v6_donors.txt" in files else ""
    nc_file = "NON-COMMERCIAL:" in lic_txt
    nc_prov = bool(((p.get("v5_va") or {}).get("summary") or {}).get("non_commercial"))
    C["licence_file_v6"] += "LICENSE_v6_donors.txt" in files
    C["licence_file_vmr0248"] += "LICENSE_VMR_0248.txt" in files
    C["licence_non_commercial"] += nc_file
    if nc_file != nc_prov:
        L["licence_mismatch"].append(sn)
    if "LICENSE_VMR_0248.txt" in files:
        L["licence_vmr0248"].append(sn)
    # ---- compact per-anatomy row
    PER[n] = {
        "patient": pat, "handedness": p.get("handedness"),
        "right_fork": [p.get("lower"), rf.get("right_fork_rung"), rf.get("right_fork_pose"), bool(rf.get("right_fork_mirrored"))],
        "left_fork": [lf.get("left_fork_lower"), lf.get("left_fork_decision"), lf.get("left_fork_pose"), bool(lf.get("left_fork_mirrored"))],
        "va": {s_: [va[s_].get("rung"), va[s_].get("donor_id"), va[s_].get("donor_side"), va[s_].get("junction_type")] for s_ in ("right", "left")},
        "bake_rung": bk.get("rung"),
        "d22_reference": [(d22.get("fork") or {}).get("reference"), (d22.get("siphon") or {}).get("reference")],
        "checks_pass": bool(ck.get("pass")),
        "derived_flags": dflags,
        "s3_fix": (s3 or {}).get("kind"),
        "flags": list(b.get("flags") or []) + list(ck.get("flags") or []) + ["bake:%s" % f for f in bk.get("flags") or []],
        "licence": "VMR 0248" if "LICENSE_VMR_0248.txt" in files else ("non-commercial" if nc_file else "other"),
        "provenance_sha256": digest_lines[-1].split()[1],
    }

# ---- donors table (one unit per TopBrain patient)
table = []
for u in sorted(UNIT, key=lambda u: (-UNIT[u]["anatomies"], u)):
    pats = sorted(UNIT_PAT[u])
    pt = pats[0] if pats else None
    table.append({"unit": u, "source": UNIT_SRC.get(u), "topbrain_patient": pt,
                  "anatomies": UNIT[u]["anatomies"],
                  "native_mirrored": [PAT_HAND[pt].get("native", 0), PAT_HAND[pt].get("mirrored", 0)] if pt else None,
                  "rva_donor_sides": UNIT[u]["rva_sides"], "lva_donor_sides": UNIT[u]["lva_sides"],
                  "rung1_own_junction": UNIT[u]["rung1"], "rung2_v5_junction": UNIT[u]["rung2"], "rung3_v5_top": UNIT[u]["rung3"],
                  "non_commercial": ("NON-COMMERCIAL" in (UNIT_LIC.get(u) or "")),
                  "licence": UNIT_LIC.get(u)})
one2one = {"units": len(UNIT_PAT), "patients": len(PAT_UNIT),
           "units_spanning_more_than_one_patient": sorted(u for u, s_ in UNIT_PAT.items() if len(s_) > 1),
           "patients_with_more_than_one_unit": sorted(pt for pt, s_ in PAT_UNIT.items() if len(s_) > 1)}
one2one["ok"] = (one2one["units"] == one2one["patients"] and not one2one["units_spanning_more_than_one_patient"]
                 and not one2one["patients_with_more_than_one_unit"])
by_source = collections.Counter(r["source"] for r in table)

fin = jl(a.final)
cov = jl(a.coverage)
digest = hashlib.sha256("".join(digest_lines).encode()).hexdigest()


def cnt(prefix):
    return {k[len(prefix):]: v for k, v in sorted(C.items()) if k.startswith(prefix)}


out = collections.OrderedDict()
out["set"] = ("B v6: the 223 v5 anatomies with every TopBrain piece posed anatomically (forks, siphons, VA tops), the "
              "donor vertebral artery's own subclavian junction and distal subclavian, the donor-wall bone-side fix, "
              "w_018b_left mr_002's VA moved, and a build record generated from the final state")
out["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
out["generator"] = VERSION
out["derived_from"] = {
    "per_anatomy": "carotid_data/anatomies_v6/<name>/provenance.json (+ mesh_v6.json, LICENSE_*.txt) -- the final state; "
                   "no assignment-time, dry-run or collection copies",
    "checks_final": "carotid_data/anatomies_v6/_records/results/v6_final_summary.json (set-level checks only)",
    "verified_by": "_records/checks/build_record_check.json (check_build_record.py: every count recounted from provenance "
                   "with separate code; also compared with the build collection results/v6_build.json)",
}
out["provenance_digest"] = {"n": len(digest_lines), "sha256": digest,
                            "rule": "sha256 over the lines '<name> <sha256 of provenance.json>\\n', sorted by name"}
out["base"] = {"set": "v5", "tree": "carotid_data/anatomies_v5", "commit": "5c2cdc0 (113c89a)",
               "names_equal_v5": names == names_v5, "dropped": sorted(set(names_v5) - set(names)),
               "added": sorted(set(names) - set(names_v5))}
out["anatomies"] = {"v6": len(names), "v5": len(names_v5), "by_handedness": cnt("handedness:"),
                    "folder_layout": dict(FILES)}
out["what_v6_changes"] = [
    "TopBrain piece orientation (audit 'TopBrain piece orientation audit (v5)'): every piece posed anatomically with "
    "superior and heading pinned and a bounded, blended seam bend; orientation gates in code (orient_gates.py)",
    "right forks (the RCCA training route) re-posed to their per-patient Zenodo map (D19); forks from the patient's other "
    "side x-mirrored (D20); the RCCA prefix (arch -> seam 1) stays host, bit-identical",
    "left forks: own fork, else a split-keeping substitute (D1), re-posed by the Zenodo map (D11); never the host LCCA tube",
    "VA tops posed anatomically toward the basilar target and stopped >= 0.6 mm apart (D2); ECA tips skipped on "
    "down-pointing ECAs (D4)",
    "donor VAs keep the donor's own subclavian -> VA junction (item e) and the donor's own distal subclavian to a cut end "
    "(D17); pinched ostia tube-bridged (D16); sides that fail keep v5's junction (D18)",
    "donor wall bone-side fix only in the artefact zones (D15, plateau rule), v5's coarse surface elsewhere",
    "w_018b_left mr_002: VA moved with the route13 tool (item b)",
    "one bake (bake_v6 1.1 = v5's handle rule + E1-E7, C4, D20) and one frozen toolchain; checks_v6 (D22 / D26 / D27)",
    "s3_fix (final verifier's must-fix list, D27): 8 right-route pieces that lost real scanned wall against v4's / v5's "
    "shipped wall re-baked with bake_v6 1.2 (L1 lattice shift on 4, L2 route-island bridge on 4); 8 pieces with a "
    "0.1-1 % deficit accepted and flagged (record only); one fold-over pair removed from m_022_left mr_021_L's OBJ; "
    "checks_v6 1.6 adds the coverage gate (against the v4 / v5 shipped wall) and the OBJ fold-over gate",
    "s4_records (the second final verifier's must-fix, D27, record only): the 10 pieces with a real-wall deficit > 0 and "
    "< 0.1 % flagged in provenance (v6_checks.flags d27_coverage_*, v6_checks.s4_records), so all 18 pieces with a "
    "measurable deficit carry a flag; no geometry change",
]
out["decisions"] = [
    {"id": "D1", "by": "orchestrator; rule approved and amended by the user 2026-10-03 18:20 / 18:30 / 18:35",
     "text": "left forks: own fork if its siphon passes; else the ladder (re-pose; same-group substitute; any split-keeping "
             "substitute; ICA extension <= 10 mm / anchor trim; last resort a longer ICA extension, then the full bend "
             "allowance); never the host LCCA tube"},
    {"id": "D2", "by": "orchestrator (stage-1 plan, 2026-10-03)", "text": "VA tops anatomical, aimed at the basilar target, >= 0.6 mm apart, no basilar fusion"},
    {"id": "D3", "by": "orchestrator (stage-1 plan)", "text": "anatomical heading (A0) + bounded seam bend <= 45 deg, blended; D3-fork: forks by the per-patient Zenodo map"},
    {"id": "D4", "by": "orchestrator (stage-1 plan)", "text": "skip the ECA tip extension on down-pointing ECAs"},
    {"id": "D5", "by": "orchestrator (stage-1 plan)", "text": "walls stay coarse (real, irregular, non-tube)"},
    {"id": "D6", "by": "orchestrator (stage-1 plan)", "text": "anchor_low blocking on the LICA seam and ECA tips, informational on the RICA"},
    {"id": "D7", "by": "orchestrator (stage-1 plan)", "text": "ICA-termini spacing gate two-sided (within 20 mm of native)"},
    {"id": "D8", "by": "orchestrator 2026-10-04", "text": "va_target <= 20 mm passes; 20-30 mm accepted + flag; > 30 mm -> D9"},
    {"id": "D9", "by": "orchestrator 2026-10-04", "text": "VA ladder: donor junction + anatomical top -> v5 junction + anatomical top -> v5's top as shipped (flag); never drop a donor VA; course RMS <= 20 mm on the donor stretch"},
    {"id": "D10", "by": "orchestrator 2026-10-04", "text": "VA corridor gate over z 440-540: lateral offset within +-8 mm of the donor's native offset (fallback: <= 8 mm medial of the host VA)"},
    {"id": "D11", "by": "orchestrator 2026-10-04 (right-fork part superseded by D19)", "text": "left forks re-posed by the per-patient Zenodo map where bend <= 45 and all gates pass, else v5's pose (flag); ICA-terminus AP offset recorded, flagged > 40 mm"},
    {"id": "D12", "by": "orchestrator 2026-10-04", "text": "ica_spacing like for like (back-mapped piece ends)"},
    {"id": "D13", "by": "orchestrator 2026-10-04", "text": "right-route baseline = v4 without its RECA tip where v4's tip ran along the ICA and v6 re-poses / skips it"},
    {"id": "D14", "by": "orchestrator 2026-10-04", "text": "SOFA VA entry: >= 3 seeds + a geometric entry test, reported against v5, not a gate"},
    {"id": "D15", "by": "orchestrator 2026-10-04", "text": "donor wall: plateau fix only in the artefact zones (bone contact / floored + short blend), v5's coarse surface elsewhere"},
    {"id": "D16", "by": "user 2026-10-04", "text": "pinched donor VA ostia (ia172_164 L, ia172_071 L) bridged with a short tube at VA calibre; flag the tubed mm"},
    {"id": "D17", "by": "user 2026-10-04", "text": "the donor's own distal subclavian continues to the end of its segmentation as a cut end (tube to ~30 mm past the ostium if short, flagged); the host distal subclavian is dropped on those sides"},
    {"id": "D18", "by": "user 2026-10-04", "text": "sides still failing after D16 / D17 keep v5's junction (flag); no other relaxations"},
    {"id": "D19", "by": "user 2026-10-04", "text": "right forks re-posed to their anatomical heading; seam-1 bend <= 45 blended, host-cut search, then up to 60 blended (flag); the RCCA prefix stays host"},
    {"id": "D20", "by": "user 2026-10-04 (left forks: orchestrator's extension)", "text": "forks from the patient's other side x-mirrored before placement (winding flipped, chirality recorded)"},
    {"id": "D21", "by": "user 2026-10-04", "text": "the 17 VA sides with no donor since v5 are not in this build"},
    {"id": "D22", "by": "user 2026-10-06", "text": "right-route check for moved pieces = surface preserved: >= 99 % of the piece's real surface beyond the blend within 0.3 mm of v4's surface carried by the piece's own transform; blend zones and handle windows reported, not gated; RCCA prefix bit-identical"},
    {"id": "D23", "by": "user 2026-10-06", "text": "w_017_left mr_002_L right VA on D9 rung 3 (tier A): keep the donor VA, accept + flag"},
    {"id": "D24", "by": "user 2026-10-06", "text": "right forks on R4 / R5 accepted with flags, re-evaluated under D22"},
    {"id": "D25", "by": "orchestrator 2026-10-06", "text": "D17 direction gate; 1 mm joins at every sampling phase; records fixed; one frozen toolchain (md5 manifest)"},
    {"id": "D26", "by": "user 2026-10-07 15:00", "text": "where v4's mesh hid part of a moved piece's own scanned surface, D22 compares with the piece's own scanned surface; flagged d22_reference = own_surface"},
    {"id": "D27", "by": "user 2026-10-07 16:25 (rule); checks_v6 1.5 is the build agent's implementation",
     "text": "no stopping for choices: residuals get the recommended minimal option + a flag; k_011_left mr_012_L's siphon deficit on the bake's repair capsule = a bake repair window, reported not gated. Applied by s3_fix (2026-10-08) to the final verifier's coverage finding: a real-wall deficit >= 1 % of a piece takes a fix that restores the wall (a targeted re-bake: bake_v6 1.2 lattice shift L1, else route-island bridge L2), else v5's geometry for that piece; < 1 % is accepted with a flag"},
]
out["toolchain"] = {k: dict(v) for k, v in sorted(TOOLS.items())}
out["toolchain"]["manifest_files"] = {
    "compose": "carotid_tools/v6/build/build/TOOLCHAIN_s2build_a80644e0.md5 (59 files; recorded in provenance v6_build.toolchain)",
    "final": "carotid_tools/v6/build/build/TOOLCHAIN_s2build.md5 = d65b7209 (72 files; recorded in provenance v6_checks of the 206 anatomies s3_fix did not change)",
    "s3_fix": "carotid_tools/v6/build/build/TOOLCHAIN_s3fix.md5 (recorded in provenance v6_checks of the 17 anatomies s3_fix changed; = d65b7209's files with bake_v6 1.2 / checks_v6 1.6 + frozen 1.1 / 1.5 + the s3_fix scripts)",
}
out["build_images"] = {"compose_va_checks_loader_sofa": "eve-training-fixed", "bake_checks": "neve-build-meshlab (as v4 / v5)",
                       "donor_walls": "neve-build-stagea / neve-build-vmtk / neve-totalseg-gpu (as v5)"}
out["sources"] = {
    "zenodo_carotid": "CarotidAnalyzer, Zenodo 10695923 v2.0.0 (lumen STL + centerlines) -- forks, as v4 / v5",
    "topbrain": "TopBrain MR labels, Zenodo 16878417 -- siphons, ECA tips, VA tops, as v4 / v5",
    "donors": "the v5 donor CTs (Head-CTA 172-series, Song 2024, VMR 0248); v6 adds the same donors' own subclavian "
              "stretch from the same CT; no new data source",
    "donor_records": "carotid_data/v6_work/wall/donors_local/<unit>/ (D15) and donors_local/_junction_records/ (item e, D16, D17)",
}
out["donors"] = {"units": len(UNIT), "by_source": dict(by_source), "one_to_one": one2one,
                 "va_donor_sides": {"right": C["va_right_donor"], "left": C["va_left_donor"]},
                 "table": table}
out["licences"] = {"anatomies_with_LICENSE_v6_donors_txt": C["licence_file_v6"],
                   "anatomies_with_LICENSE_VMR_0248_txt": C["licence_file_vmr0248"],
                   "non_commercial_anatomies": C["licence_non_commercial"],
                   "non_commercial_rule": "Arterial-derived donor VA (CC BY-NC 4.0 weights, PolyForm-NC code)",
                   "vmr0248_anatomies": L["licence_vmr0248"],
                   "licence_file_vs_provenance_mismatch": L["licence_mismatch"]}
out["carotid"] = {
    "right_fork": {"rung": cnt("right_fork_rung:"), "pose": cnt("right_fork_pose:"), "placement": cnt("right_fork_placement:"),
                   "mirrored_D20": C["right_fork_mirrored_D20"], "contralateral": C["right_fork_contralateral"],
                   "host_cut_shifted": C["right_host_cut_shifted"],
                   "seam1_bend_deg": st(S["right_seam1_bend_deg"]), "host_cut_shift_mm": st(S["right_host_cut_shift_mm"]),
                   "seam1_bend_gt45": L["right_seam1_bend_gt45"],
                   "R3_bend_up_to_60": L["right_fork_R3"], "R4_bounded_deviation": L["right_fork_R4"],
                   "R5_v5_pose": L["right_fork_R5"]},
    "left_fork": {"decision": cnt("left_fork_decision:"), "rung": cnt("left_fork_rung:"), "pose": cnt("left_fork_pose:"),
                  "mirrored_D20": C["left_fork_mirrored_D20"], "contralateral": C["left_fork_contralateral"],
                  "ica_tubed_D1_rung5_6": C["left_fork_ica_tubed"], "seam1_bend_deg": st(S["left_seam1_bend_deg"]),
                  "substituted": L["left_fork_substituted"],
                  "map_bounded_deviation": L["left_pose_map_bounded_dev"], "map_dev_deg": st(S["left_pose_map_dev_deg"]),
                  "not_on_map": L["left_pose_not_map"]},
    "ica_terminus_ap_mm": {"RICA": st(S["ica_ap_RICA_mm"]), "LICA": st(S["ica_ap_LICA_mm"]),
                           "gt40_R": C["ica_ap_gt40_R"], "gt40_L": C["ica_ap_gt40_L"], "gt40": L["ica_ap_gt40"]},
    "eca": {"right_mode": cnt("eca_right_mode:"), "tip_skipped": cnt("eca_tip_skipped:")},
    "route_baseline_D13": cnt("route_baseline:"),
    "route13_moved": L["route13_moved"], "fix_loop": L["fix_loop"], "c4_planned": L["c4_planned"],
}
out["va"] = {
    "rung": {"right": cnt("va_right_rung:"), "left": cnt("va_left_rung:")},
    "junction": {"right": cnt("va_right_junction:"), "left": cnt("va_left_junction:")},
    "flags": {"right": cnt("va_right_flag:"), "left": cnt("va_left_flag:")},
    "va_target_band": {"right": cnt("va_right_target_band:"), "left": cnt("va_left_target_band:")},
    "host_no_donor_D21": L["va_host_no_donor_D21"],
    "d18_v5_junction": L["va_d18_v5_junction"],
    "d9_rung3_v5_top": L["va_d9_rung3_v5_top"],
    "d16_tube": L["va_d16_tube"],
    "d17_tube": L["va_d17_tube"],
    "ranges_donor_sides": {"va_target_mm": st(S["va_target_mm"]), "course_rms_donor_mm": st(S["course_rms_donor_mm"]),
                           "corridor_d10_abs_dev_mm": st(S["corridor_d10_dev_mm"]),
                           "takeoff_error_vs_native_deg_abs_rung1": st(S["takeoff_error_vs_native_deg_abs_rung1"]),
                           "d17_past_ostium_mm_rung1": st(S["d17_past_ostium_mm"]),
                           "d17_host_distal_dropped_mm_rung1": st(S["d17_host_distal_dropped_mm"])},
    "d23_accepted": L["d23_accepted"],
}
out["bake"] = {"pass": C["bake_pass"], "rung": cnt("bake_rung:"), "carotid_margin_mm": cnt("bake_carotid_margin:"),
               "va_margin_mm": cnt("bake_va_margin:"), "far_clamp_used": C["bake_far_clamp_used"],
               "far_clamp": L["bake_far_clamp"],
               "sections_tube_only": C["bake_sections_tube_only"], "sections_tube_only_list": L["bake_sections_tube_only"],
               "whole_tube_sections": L["bake_whole_tube_sections"],
               "c4_mr021_lva_pinch_applied": L["bake_c4_applied"], "e7_va_aware_obj": cnt("bake_e7:"),
               "flags": cnt("bake_flag:"), "anatomies_with_tube_windows": C["bake_anatomies_with_tube_windows"],
               "window_sections": cnt("bake_window_section:"), "window_tubed_mm": st(S["bake_window_tubed_mm"]),
               "mirrored_sections": C["bake_sections_mirrored"], "mirrored_sections_outward": C["bake_sections_mirrored_outward"],
               "seconds": st(S["bake_seconds"], 1), "obj_tris": st(S["obj_tris"], 0), "full_tris": st(S["full_tris"], 0),
               "obj_genus0_closed": C["obj_genus0_closed"], "full_genus0_closed": C["full_genus0_closed"],
               "centreline_pts_outside_total": C["centreline_pts_outside_total"],
               "routes_navigable": {"RCCA": C["route_navigable_RCCA"], "LCCA": C["route_navigable_LCCA"]},
               "route_lumen_min_obj_mm": {"RCCA": st(S["route_lumen_min_mm_RCCA"]), "LCCA": st(S["route_lumen_min_mm_LCCA"])}}
out["checks"] = {"pass": C["checks_pass"], "gates": cnt("gate_pass:"), "d22_only_rerun": C["checks_d22_only_rerun"],
                 "d13_baseline": cnt("d22_baseline:"),
                 "d22_reference": {"fork": cnt("d22_fork_reference:"), "siphon": cnt("d22_siphon_reference:")},
                 "d22_value_under_reference": {"fork": st(S["d22_fork_value"], 4), "siphon": st(S["d22_siphon_value"], 4)},
                 "d22_value_v4_only": {"fork": st(S["d22_fork_value_v4"], 4), "siphon": st(S["d22_siphon_value_v4"], 4)},
                 "d22_coverage_reported": {"fork": st(S["d22_fork_coverage"], 4), "siphon": st(S["d22_siphon_coverage"], 4)},
                 "no_wall_lost": {"fork": C["d22_fork_no_wall_lost"], "siphon": C["d22_siphon_no_wall_lost"]},
                 "d26_own_surface": L["d26_own_surface"], "d27_repair_window": L["d27_repair_window"],
                 "coverage_lt_0.99_reported_not_gated": L["d22_coverage_lt_0.99"],
                 "coverage_v4_v5_shipped": {"file": "_records/results/v6_coverage_s3.json (checks_v6 1.6 parts coverage + folds on all 223; set-level copy)",
                                            "rule": cov.get("rule"), "after": cov.get("after"), "before": cov.get("before")}}
out["flags"] = {"compose_v6_build_flags": cnt("flag_compose:"), "checks_v6_checks_flags": cnt("flag_checks:"),
                "bake_flags": cnt("flag_bake:"), "anatomies_with_any_flag": C["anatomies_with_any_flag"],
                "derived_flags": cnt("flag_derived:"), "derived_flags_source": cnt("derived_flags_source:"),
                "note": "counts of the flags as written in each provenance: v6_build.flags (composer + VA tool; va_right: / "
                        "va_left: grouped by their first two parts), v6_checks.flags (D26 / D27 / coverage / D23; s3_fix: "
                        "d27_coverage_* / d27_bake_* / obj_fold_fix_*; s4_records: d27_coverage_* on 10 more pieces, detail in "
                        "v6_checks.s4_records), bake_v6.flags. derived_flags: the flags the build "
                        "collection derived outside provenance (left_pose_map_bounded_dev from the left fork's map deviation, "
                        "orient:<accepted orientation flag>); in provenance v6_checks.derived_flags on the 17 anatomies s3_fix "
                        "re-stamped, derived here for the others (per_anatomy.<name>.derived_flags)"}
out["s3_fix"] = {"by": "s3_fix (2026-10-08), the final verifier's must-fix list (verify_v6_final NO-GO) under D27",
                 "counts": cnt("s3_fix_kind:"), "bake_lattice_shift": C["s3_bake_lattice_shift"],
                 "bake_route_island_bridge": C["s3_bake_route_island_bridge"], "route_islands_bridged": C["s3_route_islands_bridged"],
                 "changed": L["s3_fix"],
                 "set_checks_after_fix": dict(jl(a.s3set), file="_records/results/s3_fix/ (split_s3.json, loader_s3.json, sofa_rcca_summary.json); set-level copy"),
                 "notes": "_records/notes/s3_fix.md"}
cf = {"file": "_records/results/v6_final_summary.json (s2_checks, collect_s2c 1.0)", "verdict": fin.get("verdict"),
      "invariants": fin.get("invariants")}
for k in ("split", "loader", "sofa_rcca", "d14_va_entry_set25", "va_lumen_polyline", "orientation", "ica_ap_gt40",
          "fork_heading_gt45", "wall_detail_median", "flags_set_checks"):
    cf[k] = fin.get(k)
if isinstance(cf.get("sofa_rcca"), dict):
    cf["sofa_rcca"] = {k: v for k, v in cf["sofa_rcca"].items() if k != "not_exercised_references"}
out["checks_final"] = cf
out["records"] = {
    "per_anatomy": "provenance.json (v4 / v5 keys kept as history; v6: v6_right_fork, v6_carotid, v6_va, v6_build, bake_v6, "
                   "v6_checks), mesh_v6.json (bake report), LICENSE_v6_donors.txt (+ LICENSE_VMR_0248.txt on 7)",
    "build_collection": "_records/results/v6_build.json (s2_build: every compose / bake / check value per anatomy, timings)",
    "set_checks": "_records/results/v6_checks.json, v6_final_summary.json (s2_checks)",
    "undecimated_unions": "carotid_data/v6_work/full/iso/<name>.vtp (git-ignored); for the 8 anatomies s3_fix re-baked "
                          "carotid_data/v6_work/s3fix/iso/final/<name>.vtp (full/iso holds their superseded pre-fix unions; "
                          "s3fix/iso/final also holds the 9 other s3_fix anatomies' unions, identical to full/iso)",
    "toolchain": "carotid_tools/v6/ (MANIFEST.md, FILES.tsv, restore_layout.sh)",
    "doc": "CAROTID_V6.md",
    "s3_fix": "_records/results/v6_coverage_s3.json, _records/results/s3_fix/ (records of the 17 changed anatomies), _records/notes/s3_fix.md",
    "s4_records": "_records/results/s4_records/ (the 10 pieces flagged, the re-stamp, the folder manifest after it, the scripts), "
                  "_records/notes/s4_records.md, _records/notes/verify_v6_final2.md",
}
out["per_anatomy"] = PER
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out + ".tmp", "w") as f:
    json.dump(out, f, indent=1, default=str)
os.replace(a.out + ".tmp", a.out)
print(json.dumps({"n": len(names), "checks_pass": C["checks_pass"], "bake_pass": C["bake_pass"], "digest": digest,
                  "one_to_one": one2one["ok"], "units": len(UNIT)}))
