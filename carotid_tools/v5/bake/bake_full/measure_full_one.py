#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\measure_full_one.py (= /sp/v5_va/bake_full/measure_full_one.py in the build containers)
"""v5_bake_full: gates + VA wall detail of ONE baked v5 anatomy (after bake_full_one.py), recorded in the anatomy's
mesh_v5.json ("v5_gates", "va_detail") and provenance.json ("bake_v5" -> "gates", "va_detail"), full record in
<BF>/_done/<name>.measure.json (the done-marker, written last).

  topology   the undecimated union (iso_v5/<name>.vtp), collision_full.vtp and the OBJ: components / genus /
             boundary loops (sdf_union_v4.genus), non-manifold edges / pinch vertices / open edges
             (bake_meshes_v4.manifold_defects), self-intersecting faces (bake_meshes_v4.self_intersections)
  budgets    OBJ <= 23500 tris at the 23.5k budget, collision_full at its budget (60k)
  routes     RCCA / LCCA navigable, 0 route points outside, 0 centreline points outside (the bake's own report)
  route      right_route_detail v5 vs v4 (mesh_v4.json): identical / ratio; and, as bake_probe/diag_route.py, the
             route wall farther than 4 / 6 / 8 mm from every v4 and v5 VA centreline sample compared with v4's exact
             union (rebuilt once, by ref_v4_union.rebuild) -> "identical_beyond_mm"
  VA detail  va_detail_v5.one's measure on the UNDECIMATED union (va_metrics.measure_union_branch unchanged):
             donor_real / topbrain / whole VA, A2 donor in situ vs alone, A3 whole VA vs v4 (the rebuilt v4 union).
             Stated diffs against va_detail_v5.one (copied below as va_detail): the union and the report are passed in
             (iso_v5/<name>.vtp, mesh_v5.json) instead of <anatomies>/_iso and mesh_v4.json; a side without a donor
             (v4 VA kept, 15 one-sided anatomies) is recorded as such instead of raising KeyError; A2 errors are caught.
    python3 measure_full_one.py <name>        (neve-build-meshlab)
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
import va_metrics as vm  # noqa: E402
import va_detail_v5 as VD  # noqa: E402
import ref_v4_union as RV  # noqa: E402
from sdf_union_v4 import Samples, genus  # noqa: E402

A5 = os.environ.get("A5_DIR", "/repo/carotid_data/anatomies_v5")
V4 = "/repo/carotid_data/anatomies_v4"
BF = os.environ.get("BF_DIR", "/repo/carotid_data/va_sources/v5_reg/bake_full")
ISO = os.environ.get("ISO_DIR", "/repo/carotid_data/va_sources/v5_reg/iso_v5")


def vm_hwm_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(tmp, path)


def topo(mesh):
    g = genus(mesh)
    nme, pinch, opn = B.manifold_defects(mesh)
    si = B.self_intersections(mesh)
    return {"tris": int(mesh.n_cells), "components": len(g), "genus": [c["genus"] for c in g][:10],
            "boundary_loops": [c["boundary_loops"] for c in g][:10], "nonmanifold_edges": nme,
            "pinch_vertices": pinch, "open_edges": opn, "self_intersecting_faces": si,
            "two_manifold": nme == 0 and pinch == 0 and opn == 0,
            "clean_tree": len(g) == 1 and g[0]["genus"] == 0 and g[0]["boundary_loops"] == 0 and nme == 0
            and pinch == 0 and opn == 0 and si == 0}


def v4_rebuild(name):
    """ref_v4_union.rebuild (v4's exact union + its VA metrics), keeping the union it builds."""
    cap = {}
    orig = B.keep_largest

    def kl(x):
        r = orig(x)
        cap["iso"] = r[0]
        return r
    B.keep_largest = kl
    try:
        rb = RV.rebuild(name)
    finally:
        B.keep_largest = orig
    return rb, cap.get("iso")


def route_cells(mesh, samp, cut):
    """bake_probe/diag_route.cells: RCCA-owned wall cells (right_route_detail's ownership and depth)."""
    rc = samp.index_of("RCCA")
    cen = np.asarray(mesh.cell_centers().points)
    A = np.asarray(mesh.compute_cell_sizes(length=False, volume=False)["Area"])
    dd, ii = cKDTree(samp.p).query(cen, k=24)
    best = ii[np.arange(len(ii)), np.argmax(samp.r[ii] - dd, axis=1)]
    own = (samp.b[best] == rc) & (samp.s[best] >= cut)
    dep = np.abs(np.linalg.norm(cen - samp.p[best], axis=1) - samp.r[best])
    return cen, A, best, own, dep


def route_away(iso5, samp5, iso4, samp4, cut):
    """diag_route's 'away_from_vas' comparison: identical beyond D mm of every v4 / v5 VA sample."""
    vap = np.vstack([s_.p[(s_.b == s_.index_of("RVA")) | (s_.b == s_.index_of("LVA"))] for s_ in (samp4, samp5)])
    vtree = cKDTree(vap)
    C = {"v4": route_cells(iso4, samp4, cut) + (samp4,), "v5": route_cells(iso5, samp5, cut) + (samp5,)}
    dist = {t: vtree.query(C[t][0])[0] for t in C}
    out = {}
    for D in (4.0, 6.0, 8.0):
        row = {}
        for tag in ("v4", "v5"):
            cen, A, best, own, dep, samp = C[tag]
            far = dist[tag] > D
            for seg, m in (("fork", own & far & (samp.s[best] < 130.0)), ("siphon", own & far & (samp.s[best] >= 130.0))):
                tot = float(A[m].sum()) or 1.0
                row.setdefault(seg, {})[tag] = {"area_mm2": round(float(A[m].sum()), 3),
                                                "frac_0.3mm": round(float(A[m & (dep > 0.3)].sum()) / tot, 6),
                                                "frac_0.5mm": round(float(A[m & (dep > 0.5)].sum()) / tot, 6),
                                                "cells": int(m.sum())}
            row["excluded_area_frac_" + tag] = round(float(A[own & ~far].sum()) / max(float(A[own].sum()), 1e-9), 4)
        for seg in ("fork", "siphon"):
            row[seg]["identical"] = row[seg]["v4"] == row[seg]["v5"]
        row["identical"] = row["fork"]["identical"] and row["siphon"]["identical"]
        out["%.0fmm" % D] = row
    beyond = next((D for D in (4, 6, 8) if out["%dmm" % D]["identical"]), None)
    return out, beyond


def va_detail(root, mesh, rep, rep4, refc, std, do_alone=True):
    """va_detail_v5.one (copied; diffs in the module docstring)."""
    from eve_bench.dualdevicenav import load_branches
    name = os.path.basename(root.rstrip("/"))
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    prov = json.load(open(os.path.join(root, "provenance.json")))
    x, v5 = prov["xform"], prov["v5_va"]
    out = {"name": name, "handedness": prov["handedness"], "patient": prov["patient"], "union_tris": int(mesh.n_cells)}
    rc = {a["name"]: a for a in refc["anatomies"]}.get(name)
    W = std["W_natural_wall"]
    for side, tag, ck, tk in (("right", "RVA", "vaC_R", "va_R"), ("left", "LVA", "vaC_L", "va_L")):
        if ck not in x:                                                   # diff: one-sided anatomy
            out[side] = {"donor": None, "v4_va_kept": True, "why": (v5.get(side) or {}).get("why")}
            if rc:
                v4s = rc[tag]
                out[side]["v4"] = {"host_core": v4s["host"]["core"], "donor_core": v4s["donor"]["core"],
                                   "whole_va_core": v4s["whole_va"]["core"]}
            continue
        bi = samp.index_of(tag)
        p, r = VD.branch_pr(br, tag)
        L = float(vm.arclen(p)[-1])
        sd = v5[side]
        o_h = 0.0 if side == "left" else float(vm.arclen(p)[int(np.argmin(np.linalg.norm(p - np.asarray(x[ck]["anchor"]), axis=1)))])
        ost, cut = float(sd["ostium_at_mm"]), float(x[tk]["cut_mm"])
        lo, hi = float(x[ck]["lo_mm"]), float(x[ck]["hi_mm"])
        ramp = sd["ramp"]
        rd = {"donor": "%s %s" % (x[ck]["donor_id"], x[ck]["donor_side"]),
              "donor_surface_genus": sd.get("donor_surface_genus"),
              "branch_point_mm": o_h, "ostium_mm": ost, "real_lo_mm": lo, "cut_mm": cut, "end_mm": L,
              "ramp_side": ramp["side"], "ramp_r": [ramp["from_r"], ramp["to_r"]]}
        _, (cen, A, best, dep, own) = vm.measure_union_branch(mesh, samp, bi, -1.0, L + 1.0, return_cells=True)
        sc, rr = samp.s[best], samp.r[best]
        s_lo, s_hi = (cut - 25.0, cut) if ramp["side"] == "donor" else (cut, min(cut + 25.0, L))
        seam = own & (sc >= s_lo) & (sc <= s_hi)

        def fr(lo_, hi_, mode="raw", core=True):
            e = vm.END_EXCL_MM if core else 0.0
            m = own & (sc >= lo_ + e) & (sc <= hi_ - e)
            if mode == "real_only":
                m = m & ~seam
            d = np.where(seam, 0.0, dep) if mode == "conservative" else dep
            if not m.any():
                return None
            a_, x_, r_ = A[m], d[m], rr[m]
            tot = float(a_.sum()) or 1.0
            out_ = {"range_mm": [round(lo_, 1), round(hi_, 1)], "mode": mode, "area_mm2": round(float(a_.sum()), 1)}
            for t in vm.ABS_THR_MM:
                out_["frac_%.1fmm" % t] = round(float(a_[x_ > t].sum()) / tot, 4)
            for q_ in vm.REL_THR:
                out_["frac_rel_%.2f" % q_] = round(float(a_[x_ > q_ * r_].sum()) / tot, 4)
            return out_
        segs = {"junction_stub_in_parent": fr(o_h, ost, core=False),
                "donor_tube_V1": fr(ost, lo) if lo > ost + 5.0 else None,
                "donor_real": fr(lo, cut, "real_only"),
                "seam_zone": fr(s_lo, s_hi, core=False),
                "topbrain": fr(cut, L, "real_only"),
                "whole_va_raw": fr(ost, L),
                "whole_va_conservative": fr(ost, L, "conservative"),
                "whole_va_real_only": fr(ost, L, "real_only")}
        rd["segments"] = segs
        rd["seam_zone_mm"] = [s_lo, s_hi]
        bins_o, bins_d = [], []
        k = 0
        while ost + k * 40.0 < L - 10.0:
            m = fr(ost + k * 40.0, min(ost + (k + 1) * 40.0, L), "conservative")
            if m:
                bins_o.append({"from_ostium_mm": k * 40, "frac_0.3mm": m["frac_0.3mm"], "frac_rel_0.15": m["frac_rel_0.15"]})
            k += 1
        for k in range(6):
            a_, b_ = max(L - (k + 1) * 40.0, ost), L - k * 40.0
            if b_ - a_ >= 10.0:
                m = fr(a_, b_, "conservative")
                if m:
                    bins_d.append({"from_distal_mm": k * 40, "frac_0.3mm": m["frac_0.3mm"]})
        rd["bins_from_ostium_40mm"] = bins_o
        rd["bins_from_distal_40mm"] = bins_d
        prof = []
        for a_ in np.arange(o_h, L - 5.0, 10.0):
            m = fr(a_, min(a_ + 10.0, L), core=False)
            if m:
                prof.append([round(a_, 1), m["frac_0.3mm"], m["frac_rel_0.15"]])
        rd["profile_10mm_all"] = prof
        wv = segs["whole_va_conservative"]
        rd["grade_in_situ_conservative"] = {
            "W1_per_VA(>=%.4f)" % W["W1_union_0.3"]["per_VA_min"]: wv["frac_0.3mm"],
            "W1_set_level(>=%.4f)" % W["W1_union_0.3"]["set_median_min"]: wv["frac_0.3mm"],
            "W2_set_level(>=%.4f)" % W["W2_union_0.5"]["set_median_min"]: wv["frac_0.5mm"],
            "W3_per_VA(>=%.4f)" % W["W3_union_relative"]["per_VA_min"]: wv["frac_rel_0.15"],
            "W3_set_level(>=%.4f)" % W["W3_union_relative"]["set_median_min"]: wv["frac_rel_0.15"],
            "W6_min_40mm_bin(>=0.01)": min([b["frac_0.3mm"] for b in bins_o] or [None]),
            "W7_distal_0-40(>=%.4f)" % W["W7_distal_overlap"]["min"]["0-40"]: bins_d[0]["frac_0.3mm"] if bins_d else None,
            "W7_distal_40-80(>=%.4f)" % W["W7_distal_overlap"]["min"]["40-80"]: bins_d[1]["frac_0.3mm"] if len(bins_d) > 1 else None}
        if do_alone:
            try:                                                          # diff: errors caught
                dp, dr = VD.slice_pr(p, r, lo, cut)
                tp, tr = VD.slice_pr(p, r, cut, L)
                a_d = VD.alone(dp, dr, x[ck], True, True)
                a_t = VD.alone(tp, tr, x[tk], True, False)
                ins_d = fr(lo, cut)
                ins_t = fr(cut, L)
                rd["A2"] = {"donor_alone": a_d, "donor_in_situ": ins_d,
                            "donor_ratio_0.3": round(ins_d["frac_0.3mm"] / a_d["core"]["frac_0.3mm"], 3) if a_d["core"]["frac_0.3mm"] else None,
                            "topbrain_alone": a_t, "topbrain_in_situ": ins_t,
                            "topbrain_ratio_0.3": round(ins_t["frac_0.3mm"] / a_t["core"]["frac_0.3mm"], 3) if a_t["core"]["frac_0.3mm"] else None}
            except Exception as e:                                       # noqa: BLE001
                traceback.print_exc()
                rd["A2"] = {"error": "%s: %s" % (type(e).__name__, e)}
        if rc:
            v4s = rc[tag]
            rd["v4"] = {"cut_mm": v4s["cut_mm"], "donor_len_mm": v4s["donor_len_mm"],
                        "host_core": v4s["host"]["core"], "donor_core": v4s["donor"]["core"],
                        "whole_va_core": v4s["whole_va"]["core"]}
            w5 = segs["whole_va_conservative"]
            rd["A3_whole_va_v5_vs_v4"] = {t: [w5[t], v4s["whole_va"]["core"][t]] for t in
                                          ("frac_0.3mm", "frac_0.5mm", "frac_rel_0.15", "frac_rel_0.25", "area_mm2")}
        out[side] = rd
    return out


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def main():
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    name = sys.argv[1]
    root = os.path.join(A5, name)
    marker = os.path.join(BF, "_done", name + ".measure.json")
    if os.path.exists(marker):
        print("done already:", marker, flush=True)
        return 0
    bake = json.load(open(os.path.join(BF, "_done", name + ".bake.json")))
    T0 = time.time()
    rep_path = os.path.join(root, V.REPORT_V5)
    rep = json.load(open(rep_path))
    rep4 = json.load(open(os.path.join(V4, name, "mesh_v4.json")))
    prov = json.load(open(os.path.join(root, "provenance.json")))
    out = {"name": name}
    iso = pv.read(os.path.join(ISO, name + ".vtp"))
    out["union"] = topo(iso)
    out["full"] = topo(pv.read(os.path.join(root, B.FULL_NAME)))
    out["obj"] = topo(pv.read(os.path.join(root, B.MESH_NAME)))
    print("union", out["union"], "\nfull", out["full"], "\nobj", out["obj"], flush=True)
    # right route detail vs v4
    d5, d4 = rep["right_route_detail"], rep4["right_route_detail"]
    rr = {"v5": d5, "v4": d4, "identical": d5 == d4,
          "ratio": {k: {t: (round(d5[k][t] / d4[k][t], 4) if d4[k][t] else None) for t in d4[k]} for k in d4},
          "max_abs_diff": max(abs(d5[k][t] - d4[k][t]) for k in d4 for t in d4[k])}
    t = time.time()
    rb, iso4 = v4_rebuild(name)
    out["v4_rebuild_seconds"] = round(time.time() - t, 1)
    out["v4_rebuild_max_abs_diff"] = rb.get("rebuild_max_abs_diff")
    samp5 = Samples(load_branches(os.path.join(root, "Centrelines_comb")))
    samp4 = Samples(load_branches(os.path.join(V4, name, "Centrelines_comb")))
    away, beyond = route_away(iso, samp5, iso4, samp4, float(prov.get("host_cut_mm", 0.0)))
    del iso4
    rr["away_from_vas"] = away
    rr["identical_beyond_mm"] = 0 if rr["identical"] else beyond
    out["right_route_detail"] = rr
    print("route identical %s, beyond %s mm, ratio %s" % (rr["identical"], rr["identical_beyond_mm"], rr["ratio"]),
          flush=True)
    std = json.load(open(VD.STD))
    vd = va_detail(root, iso, rep, rep4, {"anatomies": [rb]}, std, do_alone=True)
    out["va_detail_full"] = vd
    out["v4_va"] = {tag: {k: rb[tag][k] for k in ("origin_s_mm", "cut_mm", "length_mm", "donor_len_mm", "host",
                                                   "ramp", "donor", "whole_va", "donor_real_surface",
                                                   "donor_suppressed")} for tag in ("RVA", "LVA")}
    # compact records for mesh_v5.json / provenance
    lad = rep["ladder_v5b"]
    tub = lad["va_tubed"]
    vds = {}
    for side, ck in (("right", "vaC_R"), ("left", "vaC_L")):
        s = vd[side]
        if s.get("donor") is None:
            vds[side] = {"donor": None, "v4_va_kept": True, "why": s.get("why"),
                         "v4_host_core_frac_0.3mm": g(s, "v4", "host_core", "frac_0.3mm")}
            continue
        tb = tub.get(ck, {})
        ost = s["ostium_mm"]
        vds[side] = {
            "donor": s["donor"], "donor_surface_genus": s.get("donor_surface_genus"),
            "real_surface": bool(prov["xform"][ck].get("real_surface", True) is not False and not tb.get("whole_tube")),
            "whole_tube": bool(tb.get("whole_tube")),
            "donor_stretch_mm": tb.get("stretch_mm"), "n_windows": tb.get("n_windows"),
            "windows_from_ostium_mm": [[round(a - ost, 1), round(b - ost, 1)] for a, b in tb.get("windows_mm", [])],
            "tubed_mm": tb.get("tubed_mm"), "tubed_frac": tb.get("tubed_frac"),
            "donor_real_frac_0.3mm": g(s, "segments", "donor_real", "frac_0.3mm"),
            "donor_real_frac_0.5mm": g(s, "segments", "donor_real", "frac_0.5mm"),
            "donor_real_frac_rel_0.15": g(s, "segments", "donor_real", "frac_rel_0.15"),
            "topbrain_frac_0.3mm": g(s, "segments", "topbrain", "frac_0.3mm"),
            "whole_va_conservative_frac_0.3mm": g(s, "segments", "whole_va_conservative", "frac_0.3mm"),
            "A2_donor_alone_frac_0.3mm": g(s, "A2", "donor_alone", "core", "frac_0.3mm"),
            "A2_donor_in_situ_frac_0.3mm": g(s, "A2", "donor_in_situ", "frac_0.3mm"),
            "A2_detail_kept_in_situ_over_alone": g(s, "A2", "donor_ratio_0.3"),
            "A2_error": g(s, "A2", "error"),
            "v4_host_core_frac_0.3mm": g(s, "v4", "host_core", "frac_0.3mm"),
            "A3_whole_va_frac_0.3mm_v5_v4": g(s, "A3_whole_va_v5_vs_v4", "frac_0.3mm")}
    out["va_detail"] = vds
    rt = rep["routes"]
    routes_ok = all(rt.get(k, {}).get("navigable") and rt.get(k, {}).get("route_pts_outside", 1) == 0
                    for k in ("RCCA", "LCCA"))
    budget_ok = out["obj"]["tris"] <= 23500 and rep["obj_tris_budget"] == 23500
    full_ok = rep["decimation"].get("full_budget") == 60000 and out["full"]["tris"] <= 60000
    gates = {
        "union_clean_tree": out["union"]["clean_tree"], "full_clean_tree": out["full"]["clean_tree"],
        "obj_clean_tree": out["obj"]["clean_tree"],
        "self_intersecting_faces": {k: out[k]["self_intersecting_faces"] for k in ("union", "full", "obj")},
        "union_tris": out["union"]["tris"], "full_tris": out["full"]["tris"], "obj_tris": out["obj"]["tris"],
        "obj_budget": rep["obj_tris_budget"], "obj_budget_ok": budget_ok,
        "full_budget": rep["decimation"].get("full_budget"), "full_budget_ok": full_ok,
        "routes_navigable": routes_ok,
        "route_lumen_min_mm": {k: rt.get(k, {}).get("lumen_min_mm") for k in ("RCCA", "LCCA")},
        "centerline_pts_outside": rep["centerline_pts_outside"], "bake_pass": rep["pass"],
        "va_real_wall_owned": {side: v.get("real_surface") for side, v in vds.items() if v.get("donor")},
        "right_route_detail_identical_v4": rr["identical"],
        "right_route_detail_ratio_0.3mm": {k: rr["ratio"][k].get("frac_0.3mm") for k in rr["ratio"]},
        "right_route_identical_beyond_mm": rr["identical_beyond_mm"],
        "rung": lad["rung"], "flags": lad["flags"]}
    gates["all_pass"] = bool(gates["union_clean_tree"] and gates["full_clean_tree"] and gates["obj_clean_tree"]
                             and budget_ok and full_ok and routes_ok and rep["centerline_pts_outside"] == 0
                             and rep["pass"])
    out["gates"] = gates
    # write into the anatomy's records (atomic; every other key kept)
    rep["v5_gates"] = gates
    rep["va_detail"] = vds
    atomic_json(rep_path, rep)
    pp = os.path.join(root, "provenance.json")
    prov = json.load(open(pp, encoding="utf-8"))
    prov.setdefault("bake_v5", {})
    prov["bake_v5"]["gates"] = gates
    prov["bake_v5"]["va_detail"] = vds
    tmp = pp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    os.replace(tmp, pp)
    out["seconds"] = round(time.time() - T0, 1)
    out["peak_rss_gb"] = round(vm_hwm_kb() / 1048576.0, 3)
    atomic_json(marker, out)
    for side, v in vds.items():
        print("MEASURE %s %s: %s" % (name, side, json.dumps(v)), flush=True)
    print("MEASURE %s: all_pass %s, union/full/obj clean %s/%s/%s, SI %s, route identical %s (beyond %s mm), "
          "%.0f s, peak RSS %.2f GB" % (name, gates["all_pass"], gates["union_clean_tree"], gates["full_clean_tree"],
                                       gates["obj_clean_tree"], gates["self_intersecting_faces"], rr["identical"],
                                       rr["identical_beyond_mm"], out["seconds"], out["peak_rss_gb"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
