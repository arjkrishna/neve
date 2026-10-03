#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_summary.py (= /sp/v5_va/pilot_proc_ia172_summary.py in the build containers)
"""pilot_proc_ia172_summary.py -- the pilot's tables from /sp/v5_va/results/pilot_proc_ia172.json (per case and side
records written by pilot_proc_ia172_collect.py), beside the references in the same harness:
VMR 0248 and TaG-Net A (results/tag_ctwall_measure.json, ct_seg4 stage-A, exact ownership; plain-tube bins from
results/pilot_proc_aortaseg_tbbins.json), TopBrain (report: union 0.121 median of 50 pieces, gate-valid use-weighted
0.102; plain-tube bins 0.141 pooled / 0.125 per-piece median). Native bars W1 0.0966 / W2 0.0204 / W3 0.1416 (set medians).
-> results/pilot_proc_ia172_summary.json + .txt. eve-training-fixed.
"""
import json

import numpy as np

RES = "/sp/v5_va/results"
J = json.load(open(RES + "/pilot_proc_ia172.json"))
BARS = {"W1": 0.0966, "W2": 0.0204, "W3": 0.1416}
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)


def med(a):
    a = [x for x in a if x is not None]
    return None if not a else round(float(np.median(a)), 4)


def rng(a, n=3):
    a = [x for x in a if x is not None]
    return None if not a else [round(float(min(a)), n), round(float(np.median(a)), n), round(float(max(a)), n)]


recs = J["records"]
out = {"about": __doc__.split("\n")[0], "n_cases": len(J["cases"]), "sides": {}}
traced = [r for r in recs if r.get("trace_ok")]
donor = [r for r in recs if r.get("passes_as_v5_donor")]
clean = [r for r in donor if r.get("status") == "donor"]
draft = [r for r in recs if r.get("passes_draft_with_additions")]
whole = [r for r in recs if r.get("whole_va_pass")]
cases_with = {}
for r in donor:
    cases_with.setdefault(r["case"], []).append(r["side"])
pairs = [c for c, s in cases_with.items() if len(s) == 2]
n_sides = 2 * len(J["cases"])
out["counts"] = {"cases": len(J["cases"]), "sides": n_sides, "traced": len(traced), "donor_recalibrated": len(donor),
                 "donor_without_review_flags": len(clean), "donor_draft_plus_additions": len(draft),
                 "whole_va_pass": len(whole), "donor_subjects": len(cases_with), "donor_pairs": len(pairs),
                 "pairs": sorted(pairs), "side_pass_rate": round(len(donor) / max(n_sides, 1), 3),
                 "go_no_go_60pct": len(donor) / max(n_sides, 1) >= 0.6}
P("COUNTS", json.dumps(out["counts"]))
# gate sensitivity (verify_tag_calibration: the recalibrated values exc 12 / d 6, V1 <= 3 mm and the IQR drop are tuned
# in-sample; the pilot is the owed out-of-sample check): the donor verdict with the study's V3 gate (15 / 5) instead of
# the robust 12 / 6, with V1 bridges not tolerated (> 0 mm), and the draft + additions verdict.
CERV = ["1_origin_le_2mm", "2_foramina_ge_4", "3a_no_V1_bridge", "3b_V2_bridge_le_16", "3c_V3low_bridge_le_10",
        "3d_total_le_25", "5a_V2_band", "5b_HU_ge_0.6_CCA", "7_v3loop_robust", "8_C1_threaded"]
sens = {"recalibrated (primary)": 0, "with study V3 15/5": 0, "with study V3 15/5 + no V1 bridge": 0,
        "draft + additions (cervical: no rule 4)": 0, "draft + additions (as written, incl. rule 4 and 6)": 0}
per = {}
for r in recs:
    if not r.get("trace_ok"):
        continue
    base = r.get("passes_as_v5_donor")
    v3s = r.get("v3_study") or [None, None]
    v3ok = v3s[0] is not None and v3s[0] >= 15 and (v3s[1] if v3s[1] is not None else 99) <= 5
    a = bool(base)
    b = bool(base and v3ok)
    v1 = [g for g in (r.get("bridges") or []) if g.get("segment") == "V1" and (g.get("length_mm") or 0) > 0]
    c = bool(b and not v1)
    dr = set(r.get("draft_failed") or [])
    d_c = bool(base and v3ok and not (dr - {"4a_landmarks_complete", "4b_origin_to_dura_150_260"}))
    d_w = bool(r.get("passes_draft_with_additions"))
    per["%s %s" % (r["case"], r["side"])] = [a, b, c, d_c, d_w]
    for k, v in zip(sens, (a, b, c, d_c, d_w)):
        sens[k] += int(v)
out["gate_sensitivity_sides"] = sens
out["gate_sensitivity_per_side"] = per
P("GATE SENSITIVITY (donor sides):", json.dumps(sens))
P("")
P("%-10s %-2s %-12s %-5s %-4s %-4s %-3s %-12s %-6s %-6s %-6s %s" % ("case", "sd", "status", "Erec", "ts", "n6", "org",
                                                                       "v3 robust", "HUr", "L", "distal", "fail / review"))
for r in recs:
    v3 = r.get("v3_robust") or [None, None]
    P("%-10s %-2s %-12s %-5s %-4s %-4s %-3s %-12s %-6s %-6s %-6s %s | %s" % (
        r["case"], r["side"], r.get("status"), r.get("entry_Erec"), r.get("entry_ts_qc"), r.get("n_C6_to_C1"),
        r.get("junction_to_TS_SA_mm"), "%s/%s" % (v3[0], v3[1]), (r.get("hu_native_origin_to_C1") or {}).get("ratio_to_CCA"),
        r.get("length_trace_mm"), r.get("distal_ok"), "; ".join(r.get("gate_fail_reasons") or []),
        "; ".join(r.get("review") or [])[:160]))
# ------------------------------------------------------------------ wall
walled = [r for r in recs if r.get("wall")]
P("")
P("WALL (ct_seg4 stage-A, exact ownership, acquisition voxel); union > 0.3 / > 0.5 / > 0.15 r")
W = {}
for r in walled:
    w = r["wall"]
    key = "%s %s" % (r["case"], r["side"])
    row = {"whole": w.get("whole"), "gates_pass": w.get("gates_pass"), "gates": w.get("gates_V1_V6"),
           "genus": w.get("genus"), "offset": w.get("centre_offset_mm_median"), "plain_tube_bins": w.get("plain_tube_bin_fraction_lt_0.01"),
           "top_bins": w.get("top_relief_bins"), "edge": w.get("edge"), "r_misr": w.get("r_misr_median_mm")}
    for lset in ("ts", "ct", "rec"):
        row[lset] = {k: (v or {}).get("0.3mm") for k, v in (w.get("stretches", {}).get(lset) or {}).items()}
        row[lset + "_rel"] = {k: (v or {}).get("rel0.15") for k, v in (w.get("stretches", {}).get(lset) or {}).items()}
    row["pieces"] = {k: (v or {}).get("union") for k, v in (w.get("pieces") or {}).items()}
    row["relief_vs_contacts"] = w.get("relief_vs_contacts")
    row["fitted_lowpass"] = w.get("fitted_abs_0.3_lowpass")
    row["roughness"] = w.get("roughness_rms_mm")
    W[key] = row
    wh = w.get("whole") or {}
    e = w.get("edge") or {}
    P("%-14s whole %s/%s/%s gates %s genus %s off %s | C6-C1 ct %s rec %s ts %s | V1 %s V2 %s C2C1 %s C1du %s V4 %s | seamless %s seam %s | "
      "plain %s | FWHM %s" % (key, wh.get("0.3mm"), wh.get("0.5mm"), wh.get("rel0.15"), w.get("gates_pass"), w.get("genus"),
                              w.get("centre_offset_mm_median"), row["ct"].get("C6_to_C1"), row["rec"].get("C6_to_C1"),
                              row["ts"].get("C6_to_C1"), row["rec"].get("V1"), row["rec"].get("V2"), row["rec"].get("V3_C2_to_C1"),
                              row["rec"].get("V3_C1_to_dura"), row["rec"].get("V4"), row["rec"].get("origin_to_C1_minus_seam"),
                              row["rec"].get("seam_zone_C1m25_to_C1"), w.get("plain_tube_bin_fraction_lt_0.01"),
                              e.get("FWHM_verifier_mm")))
out["wall"] = W
valid = [W[k] for k in W if W[k]["gates_pass"]]
sets = {}
for nm, f in (("whole_0.3", lambda x: (x["whole"] or {}).get("0.3mm")), ("whole_0.5", lambda x: (x["whole"] or {}).get("0.5mm")),
              ("whole_rel", lambda x: (x["whole"] or {}).get("rel0.15"))):
    sets[nm] = med([f(x) for x in valid])
for lset in ("ts", "ct", "rec"):
    for st in ("C6_to_C1", "V1", "V2", "V3_C2_to_C1", "V3_C1_to_dura", "V4", "origin_to_C1", "origin_to_C1_minus_seam",
               "seam_zone_C1m25_to_C1"):
        sets["%s:%s_0.3" % (lset, st)] = med([x[lset].get(st) for x in valid])
        sets["%s:%s_rel" % (lset, st)] = med([x[lset + "_rel"].get(st) for x in valid])
sets["plain_tube_bins"] = med([x["plain_tube_bins"] for x in valid])
sets["FWHM_verifier"] = rng([(x["edge"] or {}).get("FWHM_verifier_mm") for x in valid])
sets["n_valid_walls"] = len(valid)
out["wall_set_medians"] = sets
out["wall_set_vs_bars"] = {"whole_W1": sets["whole_0.3"] is not None and sets["whole_0.3"] >= BARS["W1"],
                           "whole_W2": sets["whole_0.5"] is not None and sets["whole_0.5"] >= BARS["W2"],
                           "whole_W3": sets["whole_rel"] is not None and sets["whole_rel"] >= BARS["W3"]}
for lset in ("ct", "rec", "ts"):
    a, b = sets.get("%s:C6_to_C1_0.3" % lset), sets.get("%s:C6_to_C1_rel" % lset)
    out["wall_set_vs_bars"]["C6_to_C1_%s_W1" % lset] = a is not None and a >= BARS["W1"]
    out["wall_set_vs_bars"]["C6_to_C1_%s_W3" % lset] = b is not None and b >= BARS["W3"]
P("SET MEDIANS (valid walls n=%d):" % len(valid), json.dumps(sets))
P("SET vs native bars:", json.dumps(out["wall_set_vs_bars"]))
# ------------------------------------------------------------------ references in the same harness
ref = {}
try:
    M = json.load(open(RES + "/tag_ctwall_measure.json"))["full"]
    tb = json.load(open(RES + "/pilot_proc_aortaseg_tbbins.json"))
    for src in ("VMR", "TaG-A"):
        for sd in ("right", "left"):
            r = M[src]["ct_seg4_stageA"]["k64"][sd]
            u = r["wall_union"]["core"]
            st = r["stretches"]["union"]
            ref["%s %s" % (src, sd)] = {"whole": [u["frac_0.3mm"], u["frac_0.5mm"], u["frac_rel_0.15"]],
                                        "C6_to_C1_ts": (st.get("C6_to_C1") or {}).get("core", {}).get("frac_0.3mm"),
                                        "C6_to_C1_ct": (st.get("ct:C6_to_C1") or {}).get("core", {}).get("frac_0.3mm"),
                                        "plain_tube_bins": tb["ct_refs"][src][sd]["frac_bins_lt_0.01"]}
    ref["TopBrain"] = {"union_median_50": 0.121, "gate_valid_use_weighted": 0.102, "rel_0.15_median": 0.172,
                       "plain_tube_bins_pooled": tb["topbrain_summary"]["frac_bins_lt_0.01"],
                       "plain_tube_bins_per_piece_median": tb["topbrain_summary"]["per_piece_frac_median"],
                       "bin_0_40": 0.083, "bin_40_80": 0.230}
except Exception as e:  # noqa: BLE001
    ref["error"] = repr(e)
out["references"] = ref
# noise-floor phantoms (pilot_proc_ia172_phantom*)
import glob  # noqa: E402
import os  # noqa: E402
PH = {}
for p in sorted(glob.glob("/repo/carotid_data/va_sources/derived/thin_slice/ia172/*/phantom/phantom_measure*.json")):
    d = json.load(open(p))
    tag = os.path.basename(p).replace("phantom_measure", "").replace(".json", "") or "x1"
    for sd, v in d["sides"].items():
        PH["%s %s %s" % (d["case"], sd, tag)] = {
            "noise_HU": d["phantom"].get("noise_std_HU"), "fwhm": d["phantom"].get("fwhm_mm"),
            "real_union": (v.get("real") or {}).get("union_0.3_0.5_rel"), "phantom_union": (v.get("phantom") or {}).get("union_0.3_0.5_rel"),
            "real_C6C1": (v.get("real") or {}).get("C6_to_C1_ct_0.3_rel"), "phantom_C6C1": (v.get("phantom") or {}).get("C6_to_C1_ct_0.3_rel"),
            "real_plain_bins": (v.get("real") or {}).get("plain_tube_bin_fraction"),
            "phantom_plain_bins": (v.get("phantom") or {}).get("plain_tube_bin_fraction")}
out["noise_floor_phantoms"] = PH
P("NOISE FLOOR", json.dumps(PH))
rc = [W[k]["relief_vs_contacts"] for k in W if W[k].get("relief_vs_contacts") and W[k]["gates_pass"]]
out["relief_vs_contacts_set"] = {"share_in_contact_bins_median": med([x["share_of_detail_area_in_contact_bins"] for x in rc]),
                                 "union_contact_bins_median": med([x["union_0.3_contact_bins"] for x in rc]),
                                 "union_other_bins_median": med([x["union_0.3_other_bins"] for x in rc]), "n": len(rc)}
P("RELIEF vs CONTACTS", json.dumps(out["relief_vs_contacts_set"]))
P("REFERENCES", json.dumps(ref))
# ------------------------------------------------------------------ gross
P("")
P("GROSS (Arterial VMTK line, sigma 3; whole VA where the distal course passed, else origin -> dura)")
G = {}
for r in recs:
    g = r.get("gross")
    if not g:
        continue
    key = "%s %s" % (r["case"], r["side"])
    G[key] = dict(g, c2_bend=r.get("gross_c2_loop_bend_mm"), status=r.get("status"))
    P("%-14s %-6s %-40s L %s DM %s takeoff %s SCAelev %s C2bend %s minbend %s V1-4 %s entry %s r %s" % (
        key, r.get("status"), g.get("extent"), g.get("length_mm"), g.get("DM"), g.get("takeoff_10mm_fwd5_deg"),
        g.get("subclavian_elevation_fwd5_deg"), r.get("gross_c2_loop_bend_mm"), g.get("min_bend_radius_s3_mm"),
        g.get("V1_V4_mm"), g.get("entry_level_Erec"), g.get("r_corrected_x1.10_mm")))
out["gross"] = G
dn = [G[k] for k in G if G[k]["status"] in ("donor", "donor_review")]
whole_g = [x for x in dn if str(x.get("extent", "")).startswith("origin -> confluence")]
out["gross_ranges_donors"] = {"length_whole": rng([x["length_mm"] for x in whole_g], 1), "DM_whole": rng([x["DM"] for x in whole_g]),
                              "takeoff_10mm_fwd5": rng([x["takeoff_10mm_fwd5_deg"] for x in dn], 1),
                              "subclavian_elevation": rng([x["subclavian_elevation_fwd5_deg"] for x in dn], 1),
                              "c2_bend": rng([x["c2_bend"] for x in dn], 2), "r_corrected": rng([x["r_corrected_x1.10_mm"] for x in dn], 2),
                              "entry_levels": {lv: sum(1 for x in dn if x.get("entry_level_Erec") == lv) for lv in ("C5", "C6", "C7", None)}}
out["L_over_R"] = {c: v.get("L_over_R_calibre") for c, v in J["cases"].items()}
P("GROSS RANGES (donor sides):", json.dumps(out["gross_ranges_donors"]))
P("L/R calibre:", json.dumps(out["L_over_R"]))
# ------------------------------------------------------------------ timing
T = {}
for c, v in J["cases"].items():
    T[c] = {}
    for t in v.get("timing") or []:
        T[c][t["stage"]] = T[c].get(t["stage"], 0) + (t["seconds"] or 0)
out["timing_s"] = T
P("TIMING", json.dumps(T))
json.dump(out, open(RES + "/pilot_proc_ia172_summary.json", "w"), indent=1, default=str)
open(RES + "/pilot_proc_ia172_summary.txt", "w").write("\n".join(lines) + "\n")
