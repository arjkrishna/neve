#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\va_standard.py (= /sp/v5_va/va_standard.py in the build containers)
"""Grade va_metrics.measure_va results against the natural-anatomy standard
(results/standard.json, built by ref_summarize.py from the references).

    import sys; sys.path.insert(0, "/sp/v5_va")
    from va_standard import grade, grade_set
    g = grade(result)                 # one VA: per-rule pass / FAIL / n.a. + values
    s = grade_set([r1, r2, ...])      # a source: per-VA grades + the set-median rules

Per-VA rules: V1-V6, G1-G3 (G4 when starts_at_origin), the W per-VA minima and
W6. Set rules: W1-W5 set medians.

* Resolution (V5): voxel_max <= 0.65 mm -> native thresholds; (0.65, 0.85] and
  (0.85, 1.15] -> thresholds x standard.json W_resolution_factors (calibrated by
  re-imaging TopBrain); > 1.15 mm -> W 'n.a. (resolution)'.
* Radius protocol (V6): the union rules W1-W3 need VMTK MISR radii
  (provenance.radius_kind contains 'VMTK MISR'); otherwise they are 'n.a. (radius)'
  and only the radius-free W4 / W5 / W6 are graded; set rules W1-W3 use the
  VMTK-radius VAs of the set only, and every set median is taken over the VAs
  whose V1-V4 gates held.
* A failed V1-V4 gate makes every W rule 'n.a. (gate)': the wall numbers are
  not valid then.
Paths into results use '/' (keys such as 'frac_0.3mm' contain dots).
"""
import json
import os

import numpy as np

STD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "standard.json")


def _get(d, path):
    for k in (path.split("/") if isinstance(path, str) else path):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _missed(res):
    ws = res.get("wall_sections") or {}
    n, m = ws.get("n_sections"), ws.get("n_missed")
    return None if n is None else m / max(n + m, 1)


def _min_bin(res):
    vals = []
    for tag in ("bins_from_origin", "bins_from_distal"):
        for b in (res.get("wall_union") or {}).get(tag) or []:
            vals.append(b["frac_0.3mm"])
    return min(vals) if vals else None


def factors(vox, std):
    """Threshold factors for a source's voxel size: {} = native, None = not gradeable."""
    if vox is None or vox <= std["V_validity_gates"]["V5_resolution"]["native_thresholds_max"]:
        return {}
    rf = std.get("W_resolution_factors") or {}
    for L, (lo, hi) in (rf.get("bins_mm") or {}).items():
        if lo < vox <= hi:
            return {k: v for k, v in rf["factors"][L].items() if k != "n_gate_valid" and v is not None}
    return None


def _vmtk(res):
    return "VMTK MISR" in str(_get(res, "provenance/radius_kind") or "")


def grade(res, std=None):
    std = std or json.load(open(STD))
    V, G, W = std["V_validity_gates"], std["G_gross_anatomy"], std["W_natural_wall"]
    out = {}

    def rule(name, val, lo=None, hi=None, note=None):
        if val is None:
            out[name] = {"value": None, "result": "n.a. (not measured)"}
            return None
        ok = (lo is None or val >= lo) and (hi is None or val <= hi)
        out[name] = {"value": round(float(val), 4), "min": None if lo is None else round(lo, 4),
                     "max": None if hi is None else round(hi, 4), "result": "pass" if ok else "FAIL"}
        if note:
            out[name]["note"] = note
        return ok
    has_wall = res.get("wall_native") is not None
    gates = []
    if has_wall:
        gates.append(rule("V1_medial_centerline", _get(res, "wall_sections/centre_offset_mm_median"),
                          hi=V["V1_medial_centerline"]["max"]))
        gates.append(rule("V2_centerline_inside_wall", _get(res, "centerline_qc/outside_frac"),
                          hi=V["V2_centerline_inside_wall"]["max"]))
        gates.append(rule("V3_closed_sections", _missed(res), hi=V["V3_closed_sections"]["max"]))
    gates.append(rule("V4_union_topology", _get(res, "wall_union/genus"), hi=V["V4_union_topology"]["max"]))
    vox = _get(res, "provenance/voxel_max_mm")
    fac = factors(vox, std)
    if vox is not None:
        out["V5_resolution"] = {"value": vox, "result": "pass (native thresholds)" if fac == {} else (
            "pass (thresholds x resolution factors)" if fac else "n.a. for W (voxel > %.2f mm)"
            % V["V5_resolution"]["full_standard_max"])}
    vmtk = _vmtk(res)
    out["V6_radius_protocol"] = {"value": _get(res, "provenance/radius_kind"),
                                 "result": "pass" if vmtk else "n.a. for W1-W3 (radius not VMTK MISR)"}
    rule("G1_calibre", _get(res, "calibre/r_median_mm"), G["G1_calibre"]["min"], G["G1_calibre"]["max"])
    rule("G2_no_kink", _get(res, "gross/shape@3.0/min_bend_radius_mm"), lo=G["G2_no_kink"]["min"])
    rule("G3_no_unexplained_narrowing", _get(res, "calibre/q_min_W10"), lo=G["G3_no_unexplained_narrowing"]["min"])
    if _get(res, "gross/starts_at_origin"):
        rule("G4_length", _get(res, "gross/length_mm"), *G["G4_full_length"]["length_range"])
        rule("G4_DM", _get(res, "gross/DM"), *G["G4_full_length"]["DM_range"])
    wnames = ("W1_union_0.3", "W3_union_relative", "W4_sections", "W4_fitted", "W6_no_dead_stretch")
    if not has_wall:
        for k in wnames:
            out[k] = {"result": "FAIL (no wall surface: a plain tube carries no natural detail)"}
        return out
    why = None
    if fac is None:
        why = "n.a. (resolution: voxel %.2f > %.2f mm)" % (vox, V["V5_resolution"]["full_standard_max"])
    elif any(g is False for g in gates):
        why = "n.a. (gate: a V rule failed, the wall numbers are not valid)"
    if why:
        for k in wnames:
            out[k] = {"result": why}
        return out
    note = "x%s" % json.dumps(fac) if fac else None
    if vmtk:
        rule("W1_union_0.3", _get(res, "wall_union/core/frac_0.3mm"),
             lo=W["W1_union_0.3"]["per_VA_min"] * fac.get("W1", 1.0), note=note)
        rule("W3_union_relative", _get(res, "wall_union/core/frac_rel_0.15"),
             lo=W["W3_union_relative"]["per_VA_min"] * fac.get("W3", 1.0), note=note)
    else:
        for k in ("W1_union_0.3", "W3_union_relative"):
            out[k] = {"value": _get(res, "wall_union/core/frac_0.3mm" if k.startswith("W1") else
                                    "wall_union/core/frac_rel_0.15"),
                      "result": "n.a. (radius not VMTK MISR: union fractions read ~1.4x high)"}
    rule("W4_sections", _get(res, "wall_sections/in_out_0.3_mean"),
         lo=W["W4_shape_only_guard"]["per_VA_min"]["sec_in_out_0.3_mean"] * fac.get("W4_sections", 1.0), note=note)
    rule("W4_fitted", _get(res, "wall_native/fitted_frac_abs_0.3"),
         lo=W["W4_shape_only_guard"]["per_VA_min"]["fitted_abs_0.3"] * fac.get("W4_fitted", 1.0), note=note)
    rule("W6_no_dead_stretch", _min_bin(res), lo=0.01)
    return out


def grade_set(results, std=None):
    std = std or json.load(open(STD))
    W = std["W_natural_wall"]
    per = [grade(r, std) for r in results]
    vox = [_get(r, "provenance/voxel_max_mm") for r in results]
    vox = max([v for v in vox if v is not None], default=None)
    fac = factors(vox, std)
    # set medians over the VAs whose validity gates V1-V4 held: a failed gate
    # makes that VA's wall numbers meaningless (an eccentric path, a broken label)
    gate_keys = ("V1_medial_centerline", "V2_centerline_inside_wall", "V3_closed_sections", "V4_union_topology")
    valid = [not any(str(g.get(k, {}).get("result", "")).startswith("FAIL") for k in gate_keys) for g in per]
    n_vmtk = sum(_vmtk(r) for r, v in zip(results, valid) if v)
    vmtk = n_vmtk > 0

    def med(path, only_vmtk=False):
        a = [_get(r, path) for r, v in zip(results, valid) if v and (_vmtk(r) or not only_vmtk)]
        a = [x for x in a if x is not None]
        return float(np.median(a)) if a else None
    set_rules = {}
    for name, fk, path, lo, needs_vmtk in (
            ("W1_set", "W1", "wall_union/core/frac_0.3mm", W["W1_union_0.3"]["set_median_min"], True),
            ("W2_set", "W2", "wall_union/core/frac_0.5mm", W["W2_union_0.5"]["set_median_min"], True),
            ("W3_set", "W3", "wall_union/core/frac_rel_0.15", W["W3_union_relative"]["set_median_min"], True),
            ("W4_set_sections", "W4_sections", "wall_sections/in_out_0.3_mean",
             W["W4_shape_only_guard"]["set_median_min"]["sec_in_out_0.3_mean"], False),
            ("W4_set_fitted", "W4_fitted", "wall_native/fitted_frac_abs_0.3",
             W["W4_shape_only_guard"]["set_median_min"]["fitted_abs_0.3"], False),
            ("W5_set_0.75-1.5", "W5_0.75-1.5", ["wall_native", "roughness_bands", "0.75-1.5", "rms_mm"],
             W["W5_resolved_roughness"]["set_median_min"]["0.75-1.5"], False),
            ("W5_set_1.5-3", "W5_1.5-3", ["wall_native", "roughness_bands", "1.5-3", "rms_mm"],
             W["W5_resolved_roughness"]["set_median_min"]["1.5-3"], False)):
        v = med(path, only_vmtk=needs_vmtk)
        if fac is None:
            set_rules[name] = {"median": None if v is None else round(v, 4), "result": "n.a. (resolution)"}
            continue
        thr = lo * fac.get(fk, 1.0)
        if needs_vmtk and not vmtk:
            set_rules[name] = {"median": None if v is None else round(v, 4), "min": round(thr, 4),
                               "result": "n.a. (radius not VMTK MISR)"}
            continue
        set_rules[name] = {"median": None if v is None else round(v, 4), "min": round(thr, 4),
                           "result": "n.a." if v is None else ("pass" if v >= thr else "FAIL")}
    n_fail = sum(any(str(x.get("result", "")).startswith("FAIL") for x in g.values()) for g in per)
    return {"n": len(results), "n_gate_valid": int(sum(valid)), "voxel_max_mm": vox, "threshold_factors": fac,
            "n_radius_vmtk_gate_valid": n_vmtk,
            "n_with_a_failed_rule": n_fail, "set_rules": set_rules, "per_va": per}
