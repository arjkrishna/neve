#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_decide_final.py (= /sp/v5_va/pilot_decide_final.py in the build containers)
"""pilot_decide_final.py -- assemble results/pilot_decide.json from pilot_decide_need_rows.json,
pilot_decide_tables.json, pilot_donor_ranking.json and pilot_next_cases.json (read-only; python:3 container)."""
import collections
import json

RES = "/sp/v5_va/results"


def L(p):
    return json.load(open(RES + "/" + p))


need_rows = L("pilot_decide_need_rows.json")["rows"]
T = L("pilot_decide_tables.json")
R = L("pilot_donor_ranking.json")
N = L("pilot_next_cases.json")

# ---- 1. need
pp = collections.defaultdict(lambda: {"anatomies": 0, "native": 0, "mirrored": 0})
for r in need_rows:
    p = pp[r["patient"]]
    p["anatomies"] += 1
    p[r["hand"]] += 1
per_patient = sorted(({"patient": k, **v, "RVA_comes_from_donor_side": ("R" if v["native"] > v["mirrored"] else
                                                                        "L" if v["mirrored"] > v["native"] else "R and L equally")}
                      for k, v in pp.items()), key=lambda x: -x["anatomies"])
need = {"v4_anatomies": len(need_rows), "distinct_TopBrain_patients": len(pp),
        "native_anatomies": sum(r["hand"] == "native" for r in need_rows),
        "mirrored_anatomies": sum(r["hand"] == "mirrored" for r in need_rows),
        "every_anatomy_uses_both_VA_pieces_of_its_patient": all(r["R"] and r["L"] for r in need_rows),
        "donor_pairs_needed_one_to_one": len(pp), "with_5_spares": len(pp) + 5,
        "why_pairs": "native anatomies take the donor's R to the RVA and L to the LVA; mirrored ones (v3 siphon *_L) "
                     "take the donor's L, x-mirrored, to the RVA and its R to the LVA (follow rule). Either way every "
                     "anatomy consumes both sides of the one donor mapped to its TopBrain patient, so each of the 25 "
                     "patients needs a complete pair (also the patients used only native, mr_006, or only mirrored, "
                     "mr_013 / 014 / 015 / 025).",
        "per_patient": per_patient,
        "assignment_hint": "anatomies per patient range 1-16 (mr_005 / 011 / 016 / 020: 16 each; mr_015: 1). Mapping the "
                           "coarsest pairs to the patients with the most anatomies maximises the share of anatomies with "
                           "a coarse lower VA; the RVA (training-reachable) gets the donor's R in native and the donor's L "
                           "in mirrored anatomies.",
        "source": "carotid_data/anatomies_v4/*/provenance.json (patient, handedness, vertebrals.*.patient_side) and "
                  "BUILD_v4.json (graft.mirrored 120 / native 103); also graft_reg state (25 patients)"}

# ---- 2. per source
Y = T["yield"]
fail_causes = {
    "ia172": {
        "V1 / origin break or bridge": ["061 L (5.6 mm at 310 HU: Arterial mask break in an enhanced lumen, rescuable)",
                                        "041 R (4.0 mm at 177 HU)", "041 L (8.0 + 8.3 mm at 102-128 HU)",
                                        "073 L (3.9 mm through -25 HU fat = shortcut; + 6.5 mm V2)"],
        "course leaves the VA above C2 / skips the V3 loop": ["073 R (V3 5.5 / 7.4; 9.9 mm V2 bridge at 47 HU)",
                                                              "090 L (V3 0.5 / 21.1, C1 not threaded; 8.5 mm V2 break at 224 HU)",
                                                              "118 R (V3 1.1 / 21.9, C1 not threaded; 8.2 mm V2 bridge at 94 HU)",
                                                              "140 R (robust V3 9.8 / d 6.3; 8.2 mm bridge above C1; origin -> C1 candidate after review)"],
        "wrong vessel from the start": ["001 L (Arterial labels wrong; mask lost the left VA over > 40 mm)"]},
    "song": {
        "V1 / origin break or bridge": ["044 R (origin 3.25 mm + 9.1 mm V1 bridge at 480 HU: mask break)",
                                        "141 R (4.6 mm V1 bridge at 587 HU: mask break, rescuable)",
                                        "106 R (5.1 + 8.3 mm V1 bridges at 11 / 92 HU: not lumen)"],
        "course leaves the VA above C2 / skips the V3 loop": ["090 R (V3 6.4 / 13.2, origin 6.1 mm, 31 mm V1 bridge)",
                                                              "090 L (V3 0.6 / 20.9, C1 not threaded; head rotated at C1-C2)"]},
    "aortaseg": {"V1 origin bridge + wrong vessel": ["Dissec_01 R (10 mm V1 origin bridge through a 200-250 HU vein-like "
                                                     "structure, then the right ICA from C3: Arterial labelled every class "
                                                     "'L*', so the carotids were never blocked)"]}}
gate_diff = {
    "ia172": {"rule 6 only (label agreement -> review; two-patient-backed)": ["015 R", "058 L", "118 L", "141 R", "190 R"],
              "rule 6 + study V3 floor 15 (13.1 mm)": ["039 L"],
              "V1 bridge <= 3 mm (provisional) + rule 6": ["001 R (2.1 mm at 314 HU)"],
              "V2-band IQR term dropped (provisional)": ["140 L"],
              "checked by eye": "all 6 rule-6 sides are correct VA courses with a V3 loop (pilot_proc_ia172 QC PNGs)"},
    "song": {"rule 6 only": ["008 L", "044 L", "094 L", "100 L", "113 R", "113 L"],
             "rule 6 + study V3 floor 15": ["008 R (12.8 mm)", "106 L (14.3 mm)"],
             "V2-band IQR term (provisional) + rule 6": ["100 R"],
             "V1 bridge <= 3 mm (provisional)": ["109 R"],
             "checked by eye": "pilot_decide viewed the QC overview PNGs of the rule-6 pairs 008, 094, 100, 113: all four "
                               "pairs trace VA courses with V3 loops behind C1 (coarse overview only; 044 L not viewed)"},
    "aortaseg": {"origin read at the ostium (2.23 mm) + rule 6": ["Dissec_01 L"]}}
fixes = [
    {"fix": "re-join Arterial V1 mask breaks that run through enhanced lumen (bridge HU >= 0.6 x CCA) by CT region growing, "
            "then re-trace", "gain": "ia172_061 and AHMU1218141 become pairs; AHMU1218044 R still fails the origin rule; "
                                     "about +1 pair per 10 cases at scale", "cost": "low (existing cases)", "tested": False},
    {"fix": "accept origin -> C1f (or -> splice) donors whose trace deviates only above the splice; CT-guided V3 tracing",
     "gain": "140 R, 090 L, 118 R (ia172) as single sides; 090 L would complete pair ia172_090 only with a splice at or "
             "below C2", "cost": "medium", "tested": False},
    {"fix": "ct_seg4: re-centre stations whose centre HU < T_loc onto the lumen before the ray pass (C2-exit pinch)",
     "gain": "complete walls on 094 R and 106 L (2 of 15 Song walls truncated); recurs at scale", "cost": "low", "tested": False},
    {"fix": "label-independent carotid blocking in the trace", "gain": "AortaSeg-like 'L*' label failures", "cost": "medium",
     "tested": False},
    {"fix": "keep: Song Arterial input = ct_negzero_air1000 re-window; 172-series CT clipped to [-1024, 3071]; "
            "TotalSegmentator --ml + CPU split; watchdog for Arterial's centreline-stage hang; pre-fetch arch check (172)",
     "gain": "already in the pilot pipeline", "cost": "-", "tested": True},
    {"fix": "per-ray / per-triangle contamination exclusion (dim venous plexus merged at half-max is not removed by the "
            "10 mm bin rule)", "gain": "sharper ranking quantity", "cost": "medium", "tested": False}]
per_source = {}
for src in ("ia172", "song", "aortaseg"):
    y = Y[src]
    per_source[src] = {
        "cases": y["cases"], "arterial_ran": y["cases"],
        "arterial_note": {"ia172": "12 runs (native, clipped input); centreline stage hung on 061 and 090 (stopped; masks traced without labels)",
                          "song": "10 runs on the ct_negzero_air1000 re-window (+6 input tests on 100 / 109); the shipped 0-HU-clipped CT loses the neck in Arterial's extracranial model",
                          "aortaseg": "1 run, first attempt"}[src],
        "sides": y["sides"], "sides_traced": y["sides"],
        "sides_pass_recalibrated": y["sides_pass_recalibrated"], "sides_pass_draft": y["sides_pass_draft"],
        "sides_pass_backed_only": y["sides_pass_backed_only"],
        "side_rate": {"recalibrated": y["side_rate_recalibrated"], "draft": y["side_rate_draft"],
                      "backed_only": y["side_rate_backed_only"]},
        "go_60pct": {"recalibrated": y["go_60pct_recalibrated"], "draft": y["go_60pct_draft"],
                     "backed_only": y["go_60pct_backed_only"]},
        "pairs": {"recalibrated": y["pairs_recalibrated"], "draft": y["pairs_draft"], "backed_only": y["pairs_backed_only"]},
        "recal_sides_with_valid_wall": y["recal_sides_with_valid_wall"],
        "subjects_with_a_passing_side": y["subjects_with_a_recal_side"],
        "failure_causes_recalibrated": fail_causes[src], "gates_differ": gate_diff[src]}
per_source["aortaseg"]["note"] = ("one case only (AortaSeg-60 holds no other usable case): L passes QC (recalibrated and "
                                  "backed-only) but its CT wall fails validity gate V1 (offset 0.189 mm), so it is a course / "
                                  "gross donor, not a wall donor; edge FWHM 1.69 mm, late / mixed phase")
per_source["total"] = T["yield_total"]
per_source["fixes"] = fixes

# ---- 3. wall
W = T["wall_summary"]
refs = R["vmr0248_pair_reference"], R["tagnet_A_reference_not_eligible"]
wall = {
    "convention": "ct_seg4 stage-A CT-edge wall, exact ownership k 64, acquisition voxel, native bars W1 0.0966 / W2 0.0204 / "
                  "W3 0.1416 (W3 read with W1), seam zone (C1f - 25 -> C1f) separate, W7 by hand; contact-free = 10 mm bins "
                  "with < 5 % bright-neighbour / dip-before-bright edge rays",
    "per_side_table": "results/pilot_decide_tables.json -> sides[] (whole, C6_to_C1, donor_stretch, plain-tube fractions, FWHM, noise)",
    "summary_by_source": W,
    "references": {
        "host_VA_tube": 0.0007, "dense_host_RVA_mesh": 0.027,
        "VMR0248_same_harness": {"whole_u03_R_L": [0.0446, 0.1383], "C6C1_u03_R_L": [0.0757, 0.1286],
                                 "plain_tube_bin_frac_R_L": [0.545, 0.348],
                                 "donor_contact_free_R_L": refs[0]["R_L_contact_free_donor_u03"],
                                 "C6C1_contact_free_R_L": refs[0]["R_L_C6C1_u03_contact_free"], "FWHM_R_L": refs[0]["edge_FWHM_R_L"]},
        "TaGNet_A_same_harness": {"whole_u03_R_L": [0.0454, 0.1108], "C6C1_u03_R_L": [0.0709, 0.1519],
                                  "plain_tube_bin_frac_R_L": [0.762, 0.231],
                                  "donor_contact_free_R_L": refs[1]["R_L_contact_free_donor_u03"],
                                  "C6C1_contact_free_R_L": refs[1]["R_L_C6C1_u03_contact_free"], "FWHM_R_L": refs[1]["edge_FWHM_R_L"]},
        "TopBrain": {"union_median_50_pieces": 0.121, "gate_valid_use_weighted": 0.102, "rel_0.15_median": 0.172,
                     "plain_tube_bins_pooled": 0.141, "plain_tube_bins_per_piece_median": 0.125, "bin_0_40": 0.083,
                     "bin_40_80": 0.23}},
    "noise_floor_phantoms": {"ia172_061_R": {"noise_x1": 0.0008, "noise_x2": 0.0186, "real": 0.2451, "phantom_plain_tube_bins": [0.952, 0.619], "real_plain_tube_bins": 0.091},
                             "ia172_140_L": {"noise_x1": 0.005, "real": 0.214, "phantom_plain_tube_bins": 0.85, "real_plain_tube_bins": 0.10},
                             "song": "no phantom run; Song noise 12.7-23.8 HU and FWHM 0.95-1.35 mm are at or below the 172-series' 18.3-30.8 HU / 1.12-1.41 mm, so its floor should be no higher (inference, not measured)"},
    "relief_location": {"ia172_share_of_relief_area_in_contact_bins_median": 0.688, "ia172_union_contact_vs_other_bins": [0.237, 0.080],
                        "song_share_in_contact_bins_median": 0.759, "song_union_contact_vs_other_bins": [0.217, 0.046],
                        "low_pass_1.5mm_survival_ia172_coarse_sides": "0.10-0.14 of the wall still > 0.3 mm off a tube fitted after a 1.5 mm low-pass (half or more survives)",
                        "smooth_stretches": "V4 and C1 -> dura smooth on every wall (W7 fails 30 / 30; C1 -> dura median ~0.004-0.03)"},
    "verdict": None}
wall["verdict"] = (
    "Coarse and irregular in the all-bins sense, unevenly. Over the whole VA the 29 valid pilot walls read a median 0.143 "
    "(ia172 0.160, Song 0.086) union > 0.3 mm, x5 the dense host RVA mesh and x200 the host tube, and C6 -> C1 a median "
    "0.19 (ia172 0.22, Song 0.13); only one whole VA (ia172_190 L 0.026) is below the dense host mesh. Plain-tube bins "
    "0.17 (ia172 0.10, Song 0.25) against VMR 0.35-0.55, TaG-Net A 0.23-0.76, TopBrain 0.13-0.14 and the noise phantoms "
    "0.62-0.95: the relief is image structure, not noise (phantoms < 8 % of real). But 69-76 % of it sits in contact bins "
    "(bone of the foramina, enhanced veins), and with those bins excluded (the user's rule) the donor stretch reads a "
    "median 0.072 (ia172 0.092, Song 0.062; range 0.009-0.318), VMR-left / TaG-Net-A-left level and below TopBrain "
    "(0.10-0.12). The 172-series has a coarse tail (090 R 0.318, 058 L 0.276, 118 L 0.249, 061 R 0.244, 140 L 0.209) that "
    "Song lacks (max 0.134); 5 of 30 donor sides are below the dense host mesh on contact-free bins. Evenness is poor along "
    "the VA (V4 / upper V3 smooth; relief in V1-V2 and the C2 loop), between the sides of a pair (058 0.072 / 0.276) and "
    "between cases (x35 range).")

# ---- 4. ranking summary
rank = {"file": "results/pilot_donor_ranking.json",
        "ia172": [(p["case"], p["score_mean_contact_free_donor_u03"], p["score_min_side"]) for p in R["pairs_within_source"]["ia172"]],
        "song": [(p["case"], p["score_mean_contact_free_donor_u03"], p["score_min_side"]) for p in R["pairs_within_source"]["song"]],
        "vmr0248": (R["vmr0248_pair_reference"]["score_mean_contact_free_donor_u03"], R["vmr0248_pair_reference"]["score_min_side"]),
        "tagnet_A_not_eligible": (R["tagnet_A_reference_not_eligible"]["score_mean_contact_free_donor_u03"],),
        "merged_build_order": [(b["order"], b["case"], b["score"]) for b in R["build_order_merged"]],
        "rank_agreement": R["rank_agreement_spearman_vs_other_quantities"],
        "eligible_pairs_now": len(R["build_order_merged"]),
        "coarse_enough_pairs_now (both sides contact-free >= 0.027)": [b["case"] for b in R["build_order_merged"] if b["coarse_enough_both_sides"]]}

# ---- 5. scale-up
sc = N["scenarios"]
scale = {"file": "results/pilot_next_cases.json", "pair_rate_per_case": N["pair_rate_per_case"],
         "eligible_now_incl_VMR": N["need"]["eligible_now_incl_VMR"], "target_pairs": N["need"]["target_pairs"],
         "cases_needed": {k: {kk: (vv if not isinstance(vv, dict) else {"m": vv["m"], "split": vv["split"]})
                              for kk, vv in v["cases_for_30"].items() if kk != "pool_cap"} for k, v in sc.items()},
         "pools_remaining": {k: v for k, v in N["pools_remaining"].items() if k != "ia172_dropped"},
         "selection_rule": N["selection_rule"], "first_10": [(s["order"], s["case"]) for s in N["selection"][:10]],
         "cost_per_case_ts_ml": {k: {kk: v[kk] for kk in ("n_cases", "gpu_s_median", "gpu_s_max", "cpu_s_median", "cpu_s_max",
                                                           "ts_ml_s_range", "arterial_s_range")} for k, v in N["cost_per_case_ts_ml"].items()},
         "throughput_estimate": N["throughput_estimate"], "fetch_and_disk": N["fetch_and_disk"],
         "age_exploratory": {k: {"cases": v["cases"], "pairs": v["pairs_recal"]} for k, v in N["age_exploratory"].items()}}

verdict = (
    "GO for the scale-up, under the recalibrated gate, with both open sources: the 172-series passes 62.5 % of sides "
    "(5 pairs / 12 cases) and Song 75 % (6 pairs / 10 cases); AortaSeg-60 is exhausted (1 case, 0 pairs). Under the "
    "study draft gate both fail (29 % / 25 %, 1 pair in all), and with only the two-patient-backed changes Song passes "
    "(65 %) and the 172-series just misses (58 %). 12 pairs are eligible now (11 thin-slice + VMR 0248) against a need of "
    "25 (30 with spares), so the pool must grow: about 36 more cases (18 Song + 18 172-series) at the central rate, 44 for "
    "80 % and 50 for 90 % confidence, about 12-17 h on this machine with two lanes. If only pairs coarser than the dense "
    "host mesh on both sides (contact-free) are wanted, 8 qualify now and about 70-106 more cases are needed.")

open_issues = [
    "The recalibrated gate's provisional, in-sample limits (V1 <= 3 mm, no IQR term) are the only reason 4 of the 31 "
    "passing sides pass (ia172 001 R, 140 L; Song 100 R, 109 R), which costs Song pairs 100 and 109 under the backed-only "
    "gate; conversely the robust-cord d <= 6 rejects ia172 140 R, which the study cord (12.9 / 3.8) would pass.",
    "Rule 6 -> review is supported out of sample: all 6 ia172 and the Song pairs 008 / 094 / 100 / 113 that differ only "
    "by label agreement trace correct VA courses (by eye, overview PNGs); 044 L was not viewed.",
    "Contamination exclusion is per 10 mm bin: dim venous plexus merged at half-max stays in; bone-contact bins also drop "
    "real foramen wall shape (conservative). Several contact-free scores rest on < 20 % of the stretch (Song 008 R / L, "
    "094 L).",
    "094: R wall truncated at the C2-exit bend (ct_seg4 pinch), L V1 runs at a contrast-filled vein; 140 L has a genus-1 "
    "handle (not in a pair). Fix ct_seg4's pinch before the scale-up.",
    "Rescues not attempted (061 L, 141 R, 044 R, 041 R; 140 R origin -> C1).",
    "The pair rate rests on 22 cases (Wilson 95 % 0.31-0.69); the per-source strata are 3-4 cases each, so the "
    "stratum / age ordering is weak evidence. The 172-series also loses ~1 in 13 fetched series to arch truncation.",
    "Graft compatibility is a further filter: the registry dry run passes 66.5 % of unit-anatomy-sides (graft_reg), so "
    "the +5 spares are needed.",
    "Arterial-derived donors make every v5 anatomy that uses them non-commercial; Song and 172-series are CC BY 4.0.",
    "TaG-Net patient A is used only as a reference (no data licence)."]

out = {"what": "pilot_decide: the thin-slice VA pilot's decision and the donor ranking the v5 build needs",
       "date": "2026-10-02", "agent": "pilot_decide",
       "need": need, "per_source": per_source, "wall": wall, "ranking": rank, "scale_up": scale,
       "verdict": verdict, "open_issues": open_issues,
       "files": {"results": ["results/pilot_decide.json", "results/pilot_donor_ranking.json", "results/pilot_next_cases.json",
                             "results/pilot_decide_tables.json", "results/pilot_decide_need_rows.json"],
                 "notes": "notes/pilot_decide.md", "state": "state/pilot_decide.md",
                 "scripts": ["pilot_decide_need.py", "pilot_decide_collect.py", "pilot_decide_next.py", "pilot_decide_final.py",
                             "pilot_decide_peek.py"]}}
json.dump(out, open(RES + "/pilot_decide.json", "w"), indent=1)
print(json.dumps({"need": {k: need[k] for k in ("distinct_TopBrain_patients", "donor_pairs_needed_one_to_one", "with_5_spares")},
                  "top_patients": per_patient[:6], "rank": rank["merged_build_order"],
                  "coarse": rank["coarse_enough_pairs_now (both sides contact-free >= 0.027)"]}, indent=0)[:3000])
