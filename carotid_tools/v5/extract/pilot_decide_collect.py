#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_decide_collect.py (= /sp/v5_va/pilot_decide_collect.py in the build containers)
"""pilot_decide_collect.py -- the pilot decision's per-side wall table, per-source yield and the DONOR RANKING.

Read-only on every input (python:3 container, no numpy needed). Inputs:
  results/pilot_proc_{ia172,song}_grade.json   (both gates per side, wall numbers, contact-free donor stretch)
  results/pilot_proc_aortaseg.json             (Dissec_01)
  derived/thin_slice/<src>/<case>/ct_seg/ct_seg4/{measure,edge}.json   (10 mm bins + per-10 mm edge contacts)
  results/tag_ctwall_{measure,edge}.json       (VMR 0248 and TaG-Net A in the same harness)
Contact rule = the sibling grades' (pilot_proc_ia172_grade.contact_free): a 10 mm bin is a contact bin when
kind3 (bright neighbour) + kind1 (dip before bright) rays >= 5 %; bins weighted by overlap and wall area.
Outputs: results/pilot_decide_tables.json, results/pilot_donor_ranking.json.
"""
import json
import math
import os
import statistics as stt

SP = "/sp/v5_va"
RES = SP + "/results"
DER = "/repo/carotid_data/va_sources/derived/thin_slice"
BARS = {"W1": 0.0966, "W2": 0.0204, "W3": 0.1416}
HOST = {"host_VA_tube": 0.0007, "dense_host_RVA_mesh": 0.027}
SRCDIR = {"ia172": "ia172", "song": "song"}


def L(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d[k] if isinstance(d, list) and isinstance(k, int) else (d.get(k) if isinstance(d, dict) else None)
    return d


def rr(x, n=4):
    return None if x is None else round(float(x), n)


def binstats(bins, per10, rng):
    """area-weighted union over 10 mm bins overlapping rng: all bins and contact-free bins; plain-tube fractions."""
    if not bins or rng is None:
        return None
    a0, a1 = rng
    cm = {round(x["from_mm"]): x for x in (per10 or [])}
    A = Ac = D = Dc = D5 = DR = 0.0
    nb = nbc = npl = nplc = 0
    for x in bins:
        lo, hi = x["from_mm"], x["to_mm"]
        ov = max(0.0, min(hi, a1) - max(lo, a0)) / max(hi - lo, 1e-9)
        if ov <= 0:
            continue
        e = cm.get(round(lo)) or {}
        k = (e.get("frac_rays_upper_threshold_kind3") or 0) + (e.get("frac_rays_dip_before_bright_kind1") or 0)
        con = k >= 0.05 if per10 else False
        ar = (x.get("area_mm2") or 0.0) * ov
        f3 = x["frac_0.3mm"]
        A += ar
        D += ar * f3
        D5 += ar * x["frac_0.5mm"]
        DR += ar * (x.get("frac_rel_0.15") or 0.0)
        nb += 1
        npl += f3 < 0.01
        if not con:
            Ac += ar
            Dc += ar * f3
            nbc += 1
            nplc += f3 < 0.01
    return {"range_mm": [a0, a1 if a1 < 1e8 else None], "n_bins": nb, "n_contact_free_bins": nbc,
            "u03_all_bins": rr(D / A) if A else None, "u05_all_bins": rr(D5 / A) if A else None,
            "urel_all_bins": rr(DR / A) if A else None,
            "u03_contact_free": rr(Dc / Ac) if Ac else None, "contact_free_area_share": rr(Ac / A, 3) if A else None,
            "plain_tube_bin_frac_all": rr(npl / nb, 3) if nb else None,
            "plain_tube_bin_frac_contact_free": rr(nplc / nbc, 3) if nbc else None}


def pilot_side(src, row):
    case, side = row["case"], row["side"]
    sd = "right" if side == "R" else "left"
    w = row.get("wall") or {}
    W = os.path.join(DER, SRCDIR[src], case, "ct_seg/ct_seg4")
    M, E = L(W + "/measure.json"), L(W + "/edge.json")
    bins = g(M, "full", sd, "bins10")
    per10 = g(E, "sides", side, "per_10mm")
    qr = row.get("qc_recalibrated") or {}
    out = {"source": src, "case": case, "side": side,
           "qc": {"recalibrated": qr.get("status"), "recal_pass": bool(qr.get("passes")),
                  "draft_pass": bool(g(row, "qc_draft_with_additions", "passes")),
                  "backed_only_pass": bool(g(row, "qc_two_patient_backed_only", "passes")),
                  "recal_failed": qr.get("failed"), "review": qr.get("review"),
                  "draft_failed": g(row, "qc_draft_with_additions", "failed"),
                  "backed_failed": g(row, "qc_two_patient_backed_only", "failed"),
                  "distal_ok": qr.get("distal_ok"), "entry_Erec": row.get("entry_Erec"),
                  "v3_robust": row.get("v3_robust"), "trace_set": row.get("trace_set")},
           "has_wall": bool(w)}
    if not w:
        return out
    trunc = bool(g(w, "label_truncation", "truncated")) or bool(g(w, "contact_free", "partial_truncated_wall"))
    don_rng = g(w, "donor_stretch_origin_to_C1m25", "range_mm") or g(w, "contact_free", "range_mm")
    if don_rng and don_rng[1] and don_rng[1] > 1e8:
        don_rng = [don_rng[0], 1e9]
    c6 = g(w, "C6_to_C1_ct")
    c6src = "ct"
    if not c6 or c6.get("range_mm") is None:
        c6, c6src = g(w, "C6_to_C1_rec"), "rec"
    if not c6 or c6.get("range_mm") is None:
        rg = g(M, "full", sd, "stretches", "ranges_mm", "C6_to_C1")
        c6, c6src = ({"0.3mm": None, "range_mm": rg} if rg else None), "measure.json full ranges (ts)"
    c6rng = (c6 or {}).get("range_mm")
    bw = binstats(bins, per10, [0.0, 1e9])
    bd = binstats(bins, per10, don_rng) if don_rng else None
    bc = binstats(bins, per10, c6rng) if c6rng else None
    out.update({
        "wall_valid": bool(w.get("gates_V1_V6_whole")), "genus_whole": w.get("genus_whole"),
        "wall_truncated": trunc,
        "edge_FWHM_mm": w.get("edge_FWHM_mm"), "noise_HU": w.get("noise_HU"), "r_misr_mm": w.get("r_misr_mm"),
        "centre_offset_mm": w.get("centre_offset_mm"),
        "whole": {"u03": g(w, "whole", "0.3mm"), "u05": g(w, "whole", "0.5mm"), "urel": g(w, "whole", "rel0.15"),
                  "u03_contact_free_bins": g(bw, "u03_contact_free"),
                  "contact_free_area_share": g(bw, "contact_free_area_share"),
                  "plain_tube_bin_frac": w.get("plain_tube_bin_fraction"),
                  "plain_tube_bin_frac_recomputed": g(bw, "plain_tube_bin_frac_all"),
                  "plain_tube_bin_frac_contact_free": g(bw, "plain_tube_bin_frac_contact_free")},
        "C6_to_C1": {"landmarks": c6src, "range_mm": c6rng, "u03_exact": (c6 or {}).get("0.3mm"),
                     "u03_all_bins": g(bc, "u03_all_bins"), "u03_contact_free_bins": g(bc, "u03_contact_free"),
                     "contact_free_area_share": g(bc, "contact_free_area_share")},
        "donor_stretch": {"range_mm": don_rng if not don_rng or don_rng[1] < 1e8 else [don_rng[0], None],
                          "source": w.get("donor_stretch_source"),
                          "u03_exact": g(w, "donor_stretch_origin_to_C1m25", "0.3mm"),
                          "u03_all_bins": g(bd, "u03_all_bins"),
                          "u03_contact_free": g(w, "contact_free", "union_0.3_contact_free"),
                          "u03_contact_free_recomputed": g(bd, "u03_contact_free"),
                          "contact_free_area_share": g(w, "contact_free", "contact_free_area_share"),
                          "plain_tube_bin_frac_all": g(bd, "plain_tube_bin_frac_all"),
                          "plain_tube_bin_frac_contact_free": g(bd, "plain_tube_bin_frac_contact_free"),
                          "n_bins": g(bd, "n_bins"), "n_contact_free_bins": g(bd, "n_contact_free_bins")},
        "seam_zone_u03": g(w, "seam_zone_C1m25_to_C1", "0.3mm"),
        "W7_by_hand": g(w, "W7_by_hand", "result"),
        "protrusion_contaminated_share": w.get("protrusion_contaminated_share"),
    })
    return out


def ref_side(P, sd, side, tm, te):
    bins = g(tm, "full", P, "ct_seg4_stageA", "k64", sd, "bins10")
    per10 = g(te, P, "sides", sd, "per_10mm")
    lm = g(te, P, "landmarks_used", sd) or {}
    c1, c6 = lm.get("C1"), lm.get("C6")
    bw = binstats(bins, per10, [0.0, 1e9])
    bd = binstats(bins, per10, [0.0, c1 - 25.0]) if c1 else None
    bc = binstats(bins, per10, [c6, c1]) if (c1 and c6 is not None) else None
    wu = g(tm, "full", P, "ct_seg4_stageA", "k64", sd, "wall_union")
    return {"source": P, "side": side, "landmarks_used": lm,
            "edge_FWHM_mm": g(te, P, "sides", sd, "edge_verifier_method", "all", "FWHM_mm"),
            "whole": {"u03_all_bins": g(bw, "u03_all_bins"), "u03_contact_free_bins": g(bw, "u03_contact_free"),
                      "contact_free_area_share": g(bw, "contact_free_area_share"),
                      "plain_tube_bin_frac": g(bw, "plain_tube_bin_frac_all"),
                      "plain_tube_bin_frac_contact_free": g(bw, "plain_tube_bin_frac_contact_free"),
                      "wall_union_record": wu if not isinstance(wu, dict) else {k: wu.get(k) for k in list(wu)[:6]}},
            "C6_to_C1": {"range_mm": [c6, c1], "u03_all_bins": g(bc, "u03_all_bins"),
                         "u03_contact_free_bins": g(bc, "u03_contact_free"),
                         "contact_free_area_share": g(bc, "contact_free_area_share")},
            "donor_stretch": {"range_mm": [0.0, c1 - 25.0] if c1 else None, "u03_all_bins": g(bd, "u03_all_bins"),
                              "u03_contact_free": g(bd, "u03_contact_free"),
                              "contact_free_area_share": g(bd, "contact_free_area_share"),
                              "plain_tube_bin_frac_all": g(bd, "plain_tube_bin_frac_all"),
                              "plain_tube_bin_frac_contact_free": g(bd, "plain_tube_bin_frac_contact_free")}}


def med(xs):
    xs = [x for x in xs if x is not None]
    return rr(stt.median(xs)) if xs else None


def rng_(xs):
    xs = [x for x in xs if x is not None]
    return [rr(min(xs)), rr(max(xs))] if xs else None


def spearman(a, b):
    n = len(a)
    if n < 3:
        return None

    def rk(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return rr(num / den, 3) if den else None


def main():
    sides = []
    for src in ("ia172", "song"):
        G = L(RES + "/pilot_proc_%s_grade.json" % src)
        for row in G["rows"]:
            sides.append(pilot_side(src, row))
    # AortaSeg Dissec_01 (course-only; the wall fails gate V1)
    A = L(RES + "/pilot_proc_aortaseg.json")
    for r in A["records"]:
        q = r["qc"]
        dr = g(q, "draft_gate", "verdict") == "PASS"
        rules = q.get("rules") or {}
        bk = bool(q.get("passes_as_v5_donor")) and not g(rules, "3a_V1_bridge", "value") and \
            (g(q, "v3loop_study", "posterior_excursion_mm") or 0) >= 12 and (g(q, "v3loop_study", "d_C1_posterior_half_mm") or 99) <= 5
        sides.append({"source": "aortaseg", "case": r["case"], "side": r["side"], "has_wall": r["side"] == "L",
                      "qc": {"recalibrated": "donor_review" if q.get("passes_as_v5_donor") else "fail",
                             "recal_pass": bool(q.get("passes_as_v5_donor")), "draft_pass": dr, "backed_only_pass": bk,
                             "recal_failed": q.get("failed_cervical"), "review": q.get("review_flags"),
                             "draft_failed": g(q, "draft_gate", "failed")},
                      "wall_valid": False if r["side"] == "L" else None,
                      "wall_note": ("not gradeable: gate V1 centreline offset 0.189 mm > 0.15; raw whole 0.387, C6->C1 "
                                    "0.460, plain-tube bins 0/24, FWHM 1.69 mm; course / gross donor only "
                                    "(notes/pilot_proc_aortaseg.md)") if r["side"] == "L" else None})
    # references in the same harness
    tm, te = L(RES + "/tag_ctwall_measure.json"), L(RES + "/tag_ctwall_edge.json")
    refs = {"VMR0248": {s: ref_side("VMR", sd, s, tm, te) for s, sd in (("R", "right"), ("L", "left"))},
            "TaG-Net_A": {s: ref_side("TaG-A", sd, s, tm, te) for s, sd in (("R", "right"), ("L", "left"))}}
    # sanity: recomputed contact-free donor = the grade's
    chk = [(s["case"], s["side"], s["donor_stretch"]["u03_contact_free"], s["donor_stretch"]["u03_contact_free_recomputed"])
           for s in sides if s.get("donor_stretch")]
    mism = [c for c in chk if c[2] is not None and c[3] is not None and abs(c[2] - c[3]) > 0.002]

    # ---- yield per source
    yld = {}
    for src in ("ia172", "song", "aortaseg"):
        ss = [s for s in sides if s["source"] == src]
        cases = sorted(set(s["case"] for s in ss))

        def pairs(key):
            return [c for c in cases if all(s["qc"][key] for s in ss if s["case"] == c) and
                    len([s for s in ss if s["case"] == c]) == 2]
        walls = [s for s in ss if s["qc"]["recal_pass"] and s.get("has_wall") and s.get("wall_valid") is not False]
        yld[src] = {"cases": len(cases), "sides": len(ss),
                    "sides_pass_recalibrated": sum(s["qc"]["recal_pass"] for s in ss),
                    "sides_pass_draft": sum(s["qc"]["draft_pass"] for s in ss),
                    "sides_pass_backed_only": sum(s["qc"]["backed_only_pass"] for s in ss),
                    "pairs_recalibrated": pairs("recal_pass"), "pairs_draft": pairs("draft_pass"),
                    "pairs_backed_only": pairs("backed_only_pass"),
                    "subjects_with_a_recal_side": len(set(s["case"] for s in ss if s["qc"]["recal_pass"])),
                    "recal_sides_with_valid_wall": len(walls)}
        for k in ("recalibrated", "draft", "backed_only"):
            n = yld[src]["sides_pass_" + k]
            yld[src]["side_rate_" + k] = rr(n / len(ss), 3)
            yld[src]["go_60pct_" + k] = n / len(ss) >= 0.6
    tot = {k: sum(yld[s][k] for s in yld) for k in ("cases", "sides", "sides_pass_recalibrated", "sides_pass_draft",
                                                       "sides_pass_backed_only")}
    for k in ("recalibrated", "draft", "backed_only"):
        tot["pairs_" + k] = sum(len(yld[s]["pairs_" + k]) for s in yld)
        tot["side_rate_" + k] = rr(tot["sides_pass_" + k] / tot["sides"], 3)

    # ---- wall summary over recal-passing sides with a valid wall
    W = [s for s in sides if s["qc"]["recal_pass"] and s.get("has_wall") and s.get("whole")]
    wall_sum = {}
    for src in ("ia172", "song", "all"):
        ss = [s for s in W if src == "all" or s["source"] == src]
        val = [s for s in ss if s["wall_valid"]]
        wall_sum[src] = {
            "n_walls": len(ss), "n_valid": len(val),
            "whole_u03_median": med([s["whole"]["u03"] for s in val]), "whole_u03_range": rng_([s["whole"]["u03"] for s in val]),
            "whole_u05_median": med([s["whole"]["u05"] for s in val]), "whole_urel_median": med([s["whole"]["urel"] for s in val]),
            "C6C1_u03_median": med([s["C6_to_C1"]["u03_exact"] or s["C6_to_C1"]["u03_all_bins"] for s in val]),
            "C6C1_u03_range": rng_([s["C6_to_C1"]["u03_exact"] or s["C6_to_C1"]["u03_all_bins"] for s in val]),
            "C6C1_u03_contact_free_median": med([s["C6_to_C1"]["u03_contact_free_bins"] for s in val]),
            "donor_u03_contact_free_median": med([s["donor_stretch"]["u03_contact_free"] for s in val]),
            "donor_u03_contact_free_range": rng_([s["donor_stretch"]["u03_contact_free"] for s in val]),
            "whole_u03_contact_free_median": med([s["whole"]["u03_contact_free_bins"] for s in val]),
            "plain_tube_bin_frac_median": med([s["whole"]["plain_tube_bin_frac"] for s in val]),
            "plain_tube_bin_frac_range": rng_([s["whole"]["plain_tube_bin_frac"] for s in val]),
            "plain_tube_contact_free_median": med([s["whole"]["plain_tube_bin_frac_contact_free"] for s in val]),
            "FWHM_median": med([s["edge_FWHM_mm"] for s in ss]), "FWHM_range": rng_([s["edge_FWHM_mm"] for s in ss]),
            "noise_HU_range": rng_([s["noise_HU"] for s in ss]),
            "n_walls_W1_whole": sum((s["whole"]["u03"] or 0) >= BARS["W1"] for s in val),
            "n_donor_cf_below_dense_host_RVA": sum((s["donor_stretch"]["u03_contact_free"] or 0) < HOST["dense_host_RVA_mesh"] for s in ss),
            "n_whole_below_dense_host_RVA": sum((s["whole"]["u03"] or 0) < HOST["dense_host_RVA_mesh"] for s in ss)}

    # ---- pair ranking within source
    def pair_rows(src):
        ss = [s for s in sides if s["source"] == src]
        out = []
        for c in sorted(set(s["case"] for s in ss)):
            two = {s["side"]: s for s in ss if s["case"] == c}
            if len(two) != 2 or not all(two[k]["qc"]["recal_pass"] and two[k].get("whole") for k in "RL"):
                continue
            R, Lf = two["R"], two["L"]

            def pm(f):
                v = [f(R), f(Lf)]
                return rr(sum(v) / 2) if None not in v else None
            cf = [R["donor_stretch"]["u03_contact_free"], Lf["donor_stretch"]["u03_contact_free"]]
            cav = []
            for s in (R, Lf):
                if s["wall_truncated"]:
                    cav.append("%s wall truncated at the C2-exit bend (scored on the partial wall)" % s["side"])
                if not s["wall_valid"]:
                    cav.append("%s wall gate fails (genus %s)" % (s["side"], s["genus_whole"]))
                if (s["donor_stretch"]["contact_free_area_share"] or 0) < 0.2:
                    cav.append("%s contact-free score rests on %.0f %% of the stretch" % (s["side"], 100 * (s["donor_stretch"]["contact_free_area_share"] or 0)))
                if (s["donor_stretch"]["u03_contact_free"] or 0) < HOST["dense_host_RVA_mesh"]:
                    cav.append("%s contact-free %.3f < dense host RVA 0.027" % (s["side"], s["donor_stretch"]["u03_contact_free"] or 0))
            if c == "AHMU1218094":
                cav.append("L V1 runs at the edge of a contrast-filled vein for ~20 mm (look before use; its bins are excluded)")
            if c == "ia172_015":
                cav.append("R Erec entry C7 (check by eye)")
            out.append({
                "source": src, "case": c,
                "score_mean_contact_free_donor_u03": rr(sum(cf) / 2), "score_min_side": rr(min(cf)),
                "R_L_contact_free_donor_u03": cf,
                "R_L_contact_free_share": [R["donor_stretch"]["contact_free_area_share"], Lf["donor_stretch"]["contact_free_area_share"]],
                "R_L_whole_u03": [R["whole"]["u03"], Lf["whole"]["u03"]],
                "R_L_whole_u03_contact_free": [R["whole"]["u03_contact_free_bins"], Lf["whole"]["u03_contact_free_bins"]],
                "R_L_C6C1_u03": [R["C6_to_C1"]["u03_exact"], Lf["C6_to_C1"]["u03_exact"]],
                "R_L_C6C1_u03_contact_free": [R["C6_to_C1"]["u03_contact_free_bins"], Lf["C6_to_C1"]["u03_contact_free_bins"]],
                "R_L_donor_u03_all": [R["donor_stretch"]["u03_exact"], Lf["donor_stretch"]["u03_exact"]],
                "R_L_plain_tube_bin_frac_whole": [R["whole"]["plain_tube_bin_frac"], Lf["whole"]["plain_tube_bin_frac"]],
                "R_L_plain_tube_frac_donor_contact_free": [R["donor_stretch"]["plain_tube_bin_frac_contact_free"],
                                                           Lf["donor_stretch"]["plain_tube_bin_frac_contact_free"]],
                "mean_whole_u03": pm(lambda s: s["whole"]["u03"]),
                "mean_whole_u03_contact_free": pm(lambda s: s["whole"]["u03_contact_free_bins"]),
                "mean_C6C1_u03_contact_free": pm(lambda s: s["C6_to_C1"]["u03_contact_free_bins"]),
                "mean_C6C1_u03": pm(lambda s: s["C6_to_C1"]["u03_exact"] if s["C6_to_C1"]["u03_exact"] is not None else s["C6_to_C1"]["u03_all_bins"]),
                "mean_plain_tube_frac_donor_cf": pm(lambda s: s["donor_stretch"]["plain_tube_bin_frac_contact_free"]),
                "qc_R_L": [R["qc"]["recalibrated"], Lf["qc"]["recalibrated"]],
                "review_flags": {"R": R["qc"]["review"], "L": Lf["qc"]["review"]},
                "pair_passes_draft": R["qc"]["draft_pass"] and Lf["qc"]["draft_pass"],
                "pair_passes_backed_only": R["qc"]["backed_only_pass"] and Lf["qc"]["backed_only_pass"],
                "coarse_enough_both_sides_vs_dense_host_RVA": min(cf) >= HOST["dense_host_RVA_mesh"],
                "edge_FWHM_R_L": [R["edge_FWHM_mm"], Lf["edge_FWHM_mm"]],
                "trace_sets": [R["qc"].get("trace_set"), Lf["qc"].get("trace_set")],
                "caveats": cav})
        out.sort(key=lambda p: -(p["score_mean_contact_free_donor_u03"] or -1))
        for i, p in enumerate(out):
            p["rank_within_source"] = i + 1
        # rank agreement with the other coarseness quantities
        agree = {}
        if len(out) >= 3:
            base = [p["score_mean_contact_free_donor_u03"] for p in out]
            for k in ("mean_whole_u03_contact_free", "mean_C6C1_u03_contact_free", "mean_whole_u03", "mean_C6C1_u03"):
                v = [p[k] for p in out]
                if None not in v:
                    agree[k] = spearman(base, v)
            v = [-(p["mean_plain_tube_frac_donor_cf"] if p["mean_plain_tube_frac_donor_cf"] is not None else 1) for p in out]
            agree["minus_mean_plain_tube_frac_donor_cf"] = spearman(base, v)
        return out, agree

    ranking = {}
    agreement = {}
    for src in ("ia172", "song"):
        ranking[src], agreement[src] = pair_rows(src)

    # VMR pair (eligible, reference); TaG-Net A (not eligible)
    def refpair(name, eligible, note):
        R, Lf = refs[name]["R"], refs[name]["L"]
        cf = [R["donor_stretch"]["u03_contact_free"], Lf["donor_stretch"]["u03_contact_free"]]
        return {"source": name, "case": name, "eligible": eligible, "note": note,
                "score_mean_contact_free_donor_u03": rr(sum(cf) / 2) if None not in cf else None,
                "score_min_side": rr(min(cf)) if None not in cf else None,
                "R_L_contact_free_donor_u03": cf,
                "R_L_contact_free_share": [R["donor_stretch"]["contact_free_area_share"], Lf["donor_stretch"]["contact_free_area_share"]],
                "R_L_donor_u03_all_bins": [R["donor_stretch"]["u03_all_bins"], Lf["donor_stretch"]["u03_all_bins"]],
                "R_L_whole_u03_all_bins": [R["whole"]["u03_all_bins"], Lf["whole"]["u03_all_bins"]],
                "R_L_whole_u03_contact_free": [R["whole"]["u03_contact_free_bins"], Lf["whole"]["u03_contact_free_bins"]],
                "R_L_C6C1_u03_all_bins": [R["C6_to_C1"]["u03_all_bins"], Lf["C6_to_C1"]["u03_all_bins"]],
                "R_L_C6C1_u03_contact_free": [R["C6_to_C1"]["u03_contact_free_bins"], Lf["C6_to_C1"]["u03_contact_free_bins"]],
                "R_L_plain_tube_bin_frac_whole": [R["whole"]["plain_tube_bin_frac"], Lf["whole"]["plain_tube_bin_frac"]],
                "R_L_plain_tube_frac_donor_contact_free": [R["donor_stretch"]["plain_tube_bin_frac_contact_free"], Lf["donor_stretch"]["plain_tube_bin_frac_contact_free"]],
                "edge_FWHM_R_L": [R["edge_FWHM_mm"], Lf["edge_FWHM_mm"]],
                "coarse_enough_both_sides_vs_dense_host_RVA": (min(cf) >= HOST["dense_host_RVA_mesh"]) if None not in cf else None,
                "landmarks": {"R": R["landmarks_used"], "L": Lf["landmarks_used"]}}
    vmr = refpair("VMR0248", True, "eligible; QC by expert paths + CT (feasibility report); prototype donor; left V1 "
                  "(first 40 mm) baked as tube in the prototype because of the vein contact; research use, cite Wilson 2013")
    tag = refpair("TaG-Net_A", False, "NOT eligible (no data licence; internal calibration only)")

    # merged build order: round-robin over thin-slice sources by within-source rank (the coarser candidate of each
    # round first); VMR placed by its score among them.
    merged = []
    rounds = max(len(ranking["ia172"]), len(ranking["song"]))
    for i in range(rounds):
        cand = [ranking[s][i] for s in ("ia172", "song") if i < len(ranking[s])]
        cand.sort(key=lambda p: -(p["score_mean_contact_free_donor_u03"] or -1))
        merged += cand
    if vmr["score_mean_contact_free_donor_u03"] is not None:
        pos = next((j for j, p in enumerate(merged)
                    if (p["score_mean_contact_free_donor_u03"] or 0) < vmr["score_mean_contact_free_donor_u03"]), len(merged))
        merged.insert(pos, vmr)
    build_order = [{"order": j + 1, "source": p["source"], "case": p["case"],
                    "rank_within_source": p.get("rank_within_source", 1),
                    "score": p["score_mean_contact_free_donor_u03"], "min_side": p["score_min_side"],
                    "coarse_enough_both_sides": p["coarse_enough_both_sides_vs_dense_host_RVA"],
                    "caveats": p.get("caveats", [])} for j, p in enumerate(merged)]

    # single passing sides (recalibrated gate), ranked within source
    singles = {}
    for src in ("ia172", "song", "aortaseg"):
        ss = [s for s in sides if s["source"] == src and s["qc"]["recal_pass"]]
        paired = set(p["case"] for p in ranking.get(src, []))
        rows = []
        for s in ss:
            d = s.get("donor_stretch") or {}
            rows.append({"source": src, "case": s["case"], "side": s["side"], "qc": s["qc"]["recalibrated"],
                         "in_a_pair": s["case"] in paired, "draft_pass": s["qc"]["draft_pass"],
                         "backed_only_pass": s["qc"]["backed_only_pass"],
                         "donor_u03_contact_free": d.get("u03_contact_free"),
                         "contact_free_share": d.get("contact_free_area_share"),
                         "donor_u03_all": d.get("u03_exact"),
                         "whole_u03": g(s, "whole", "u03"), "C6C1_u03": g(s, "C6_to_C1", "u03_exact"),
                         "plain_tube_bin_frac_whole": g(s, "whole", "plain_tube_bin_frac"),
                         "wall_valid": s.get("wall_valid"), "wall_truncated": s.get("wall_truncated"),
                         "note": s.get("wall_note")})
        rows.sort(key=lambda r: -(r["donor_u03_contact_free"] if r["donor_u03_contact_free"] is not None else -1))
        for i, r in enumerate(rows):
            r["rank_within_source"] = i + 1
        singles[src] = rows

    tables = {"what": "pilot_decide per-side table (recomputed contact-free whole / C6->C1 / donor stretch, plain-tube "
                      "fractions) + yield + wall summary", "host": HOST, "bars_native": BARS,
              "sides": sides, "references_same_harness": refs, "yield": yld, "yield_total": tot,
              "wall_summary": wall_sum, "sanity_contact_free_recompute_mismatches": mism,
              "n_checked": len(chk)}
    json.dump(tables, open(RES + "/pilot_decide_tables.json", "w"), indent=1)

    rank = {
        "what": "v5 lower-VA DONOR RANKING for the build (pilot_decide). User's rule: QC-passing PAIRS only, ranked by "
                "coarseness with contamination-driven relief excluded, ranked within source, sources kept represented. "
                "TaG-Net is never a shipped donor.",
        "ranking_quantity": "pair score = mean over R and L of the donor-stretch (origin -> C1f - 25 mm, the piece strategy B "
                            "grafts; seam zone separate) union > 0.3 mm on the ct_seg4 CT-edge wall (stage-A, exact ownership "
                            "k 64, acquisition voxel) over CONTACT-FREE 10 mm bins (bins with >= 5 % bright-neighbour / "
                            "dip-before-bright edge rays = bone in the foramina, enhanced veins, other arteries are excluded). "
                            "Reported beside it: whole-VA and C6->C1 union > 0.3 mm (exact, all bins, and contact-free bins), "
                            "plain-tube bin fraction (10 mm bins < 0.01; whole VA and contact-free donor bins).",
        "qc_gate": "recalibrated gate (primary; notes/tag_calibration.md values); draft and backed-only verdicts per pair",
        "contact_rule_limits": "10 mm bins; dim venous plexus merged at half-max is not removed; TotalSegmentator vertebra "
                               "masks fill the foramina so class labels of bone contacts come from the ray rule only",
        "host_reference": HOST,
        "pairs_within_source": ranking, "rank_agreement_spearman_vs_other_quantities": agreement,
        "build_order_merged": build_order,
        "build_order_rule": "round-robin over the two thin-slice sources by within-source rank (the coarser of each round "
                            "first), VMR 0248 inserted by its score. With 12 eligible pairs against a need of 25, every "
                            "eligible pair is used now; the order sets assignment priority and which pairs are displaced "
                            "once more than need + 5 exist.",
        "vmr0248_pair_reference": vmr, "tagnet_A_reference_not_eligible": tag,
        "single_sides_passing": singles,
        "aortaseg": "Dissec_01: L passes QC (recalibrated and backed-only; draft fails on ostium 2.23 mm + label) but its "
                    "wall fails validity gate V1 (offset 0.189 mm): course / gross donor only; R fails. 0 pairs. "
                    "AortaSeg-60 holds no other usable case.",
    }
    json.dump(rank, open(RES + "/pilot_donor_ranking.json", "w"), indent=1)
    # console summary
    print("YIELD", json.dumps(yld))
    print("TOTAL", json.dumps(tot))
    print("SANITY mismatches", mism, "of", len(chk))
    for src in ("ia172", "song", "all"):
        print("WALL", src, json.dumps(wall_sum[src]))
    for src in ("ia172", "song"):
        print("RANK", src, "agreement", agreement[src])
        for p in ranking[src]:
            print("  #%d %s score %s min %s cf %s share %s | whole %s wholecf %s | C6C1 %s C6C1cf %s | plain %s cfplain %s | %s" % (
                p["rank_within_source"], p["case"], p["score_mean_contact_free_donor_u03"], p["score_min_side"],
                p["R_L_contact_free_donor_u03"], p["R_L_contact_free_share"], p["R_L_whole_u03"], p["R_L_whole_u03_contact_free"],
                p["R_L_C6C1_u03"], p["R_L_C6C1_u03_contact_free"], p["R_L_plain_tube_bin_frac_whole"],
                p["R_L_plain_tube_frac_donor_contact_free"], "; ".join(p["caveats"])))
    for p in (vmr, tag):
        print("REF", json.dumps(p))
    print("ORDER", [(b["order"], b["case"], b["score"]) for b in build_order])


if __name__ == "__main__":
    main()
