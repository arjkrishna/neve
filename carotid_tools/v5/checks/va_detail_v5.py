#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\va_detail_v5.py (= /sp/v5_va/va_detail_v5.py in the build containers)
"""A1-A3 of the natural-anatomy standard (notes/ref.md) on baked v5 prototypes: VA wall detail measured on the
UNDECIMATED gate-passing union (bake_v5.py --dump-iso), with va_metrics.measure_union_branch unchanged
(right_route_detail's measure: cells owned by the branch stretch, dep = | |x-c| - r |, frac > 0.3 / 0.5 mm and
> 0.15 / 0.25 r; 'core' drops 2 mm at each end).

Per VA, on the branch arclength of the anatomy as written:
    junction      host branch point O_h -> donor ostium (the stub inside the parent subclavian)
    donor_tube    ostium -> ostium + 40 mm (left only: VMR's venous-contact V1, baked as tube)
    donor_real    the donor real section [lo, hi] (minus a donor-side calibre ramp)
    ramp          the calibre ramp (25 mm, on whichever side was eased down)
    topbrain      the TopBrain distal piece (minus a TopBrain-side ramp)
    whole_va      O_h -> end (ref c's 'whole_va' range for v4) and ostium -> end
    bins          40 mm from the branch point and from the distal end (W6 / W7)
A2: each real piece measured ALONE (va_metrics.union_detail on the same centerline, radii and transformed
surface, tapered at its seams) against in situ. A3: whole VA against v4 as shipped (ref_c_v4_union.json,
the exact v4 union rebuilt by the ref agent) and against W1-W3. Also right_route_detail v5 vs v4.

    python3 /sp/v5_va/va_detail_v5.py [--anatomies /repo/carotid_data/va_sources/v5_proto] [--only ...]
eve-training-fixed.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v5_va")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/opt/eve_training/eve_bench")
sys.path.insert(0, "/repo")

import va_metrics as vm  # noqa: E402
from sdf_union import load_surface  # noqa: E402
from sdf_union_v4 import Samples  # noqa: E402

V4 = "/repo/carotid_data/anatomies_v4"
REF_C = "/sp/v5_va/results/ref_c_v4_union.json"
STD = "/sp/v5_va/results/standard.json"


def seg(mesh, samp, bi, lo, hi):
    if hi - lo < 5.0:
        return None
    m = vm.measure_union_branch(mesh, samp, bi, lo, hi)
    m["range_mm"] = [round(lo, 1), round(hi, 1)]
    return m


def branch_pr(br, tag):
    for b in br:
        if str(b.name).endswith(tag + ".mrk") or str(b.name).endswith(tag):
            return np.asarray(b.coordinates, float), np.asarray(b.radii, float)
    raise KeyError(tag)


def slice_pr(p, r, lo, hi):
    s = vm.arclen(p)
    m = (s >= lo - 1e-6) & (s <= hi + 1e-6)
    return p[m], r[m]


def alone(p, r, xf, t0, t1):
    surf = load_surface(xf["surface"], xf["mirror"], xf["R"], xf["origin"], xf["anchor"])
    u = vm.union_detail(p, r, surf, taper_start=t0, taper_end=t1)
    return {"core": u["core"], "genus": u["genus"], "components_before_keep": u["components_before_keep"]}


def one(root, refc, std, do_alone=True):
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    name = os.path.basename(root.rstrip("/"))
    iso_path = os.path.join(os.path.dirname(root.rstrip("/")), "_iso", name + ".vtp")
    mesh = pv.read(iso_path)
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    prov = json.load(open(os.path.join(root, "provenance.json")))
    rep = json.load(open(os.path.join(root, "mesh_v4.json")))
    rep4 = json.load(open(os.path.join(V4, name, "mesh_v4.json")))
    x, v5 = prov["xform"], prov["v5_va"]
    out = {"name": name, "handedness": prov["handedness"], "patient": prov["patient"],
           "union_tris": int(mesh.n_cells), "bake": {k: rep[k] for k in ("union", "margin_mm", "sections_tube_only",
                                                                          "suppressed_stretches_mm", "sections_tube_in_effect",
                                                                          "far_clamp", "decimation", "obj_tris_budget",
                                                                          "routes", "centerline_pts_outside", "pass")},
           "right_route_detail": {"v5": rep["right_route_detail"], "v4": rep4["right_route_detail"]}}
    out["right_route_detail"]["v5_over_v4"] = {
        k: {t: (round(rep["right_route_detail"][k][t] / rep4["right_route_detail"][k][t], 4)
                if rep4["right_route_detail"][k][t] else None) for t in ("frac_0.3mm", "frac_0.5mm")}
        for k in ("fork", "siphon")}
    out["full"] = {"genus": rep["full"]["genus"], "tris": rep["full"]["tris"]}
    out["obj"] = {"genus": rep["obj"]["genus"], "tris": rep["obj"]["tris"]}
    rc = {a["name"]: a for a in refc["anatomies"]}.get(name)
    W = std["W_natural_wall"]
    for side, tag, ck, tk in (("right", "RVA", "vaC_R", "va_R"), ("left", "LVA", "vaC_L", "va_L")):
        bi = samp.index_of(tag)
        p, r = branch_pr(br, tag)
        L = float(vm.arclen(p)[-1])
        sd = v5[side]
        o_h = 0.0 if side == "left" else float(vm.arclen(p)[int(np.argmin(np.linalg.norm(p - np.asarray(x[ck]["anchor"]), axis=1)))])
        ost, cut = float(sd["ostium_at_mm"]), float(x[tk]["cut_mm"])
        lo, hi = float(x[ck]["lo_mm"]), float(x[ck]["hi_mm"])
        ramp = sd["ramp"]
        rd = {"branch_point_mm": o_h, "ostium_mm": ost, "real_lo_mm": lo, "cut_mm": cut, "end_mm": L,
              "ramp_side": ramp["side"], "ramp_r": [ramp["from_r"], ramp["to_r"]]}
        # every VA cell once (measure_union_branch's own ownership and dep), then any stretch by its owner's s
        _, (cen, A, best, dep, own) = vm.measure_union_branch(mesh, samp, bi, -1.0, L + 1.0, return_cells=True)
        sc, rr = samp.s[best], samp.r[best]
        # the seam zone: the 25 mm calibre ramp. Its declared tube is eased DOWN under a real wall (so that no
        # real wall is buried), and right_route_detail's measure reads that calibre offset (~0.3-0.9 mm) as
        # wall detail. It is reported apart and counted as 0 in the 'conservative' figures, as v4's widened
        # ramp (0.000) is counted in ref c's whole_va.
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
        # 40 mm bins (W6) from the ostium and from the distal end (W7), conservative
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
        # 10 mm profile (raw, all cells), for the figure; seam zone marked by rd["seam_zone_mm"]
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
        # A2: each real piece alone vs in situ
        if do_alone:
            dp, dr = slice_pr(p, r, lo, cut)
            tp, tr = slice_pr(p, r, cut, L)
            a_d = alone(dp, dr, x[ck], True, True)
            a_t = alone(tp, tr, x[tk], True, False)
            ins_d = fr(lo, cut)
            ins_t = fr(cut, L)
            rd["A2"] = {"donor_alone": a_d, "donor_in_situ": ins_d,
                        "donor_ratio_0.3": round(ins_d["frac_0.3mm"] / a_d["core"]["frac_0.3mm"], 3) if a_d["core"]["frac_0.3mm"] else None,
                        "topbrain_alone": a_t, "topbrain_in_situ": ins_t,
                        "topbrain_ratio_0.3": round(ins_t["frac_0.3mm"] / a_t["core"]["frac_0.3mm"], 3) if a_t["core"]["frac_0.3mm"] else None}
        # A3: against v4 as shipped (same anatomy, exact v4 union rebuilt by ref_v4_union.py)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", default="/repo/carotid_data/va_sources/v5_proto")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-alone", action="store_true")
    ap.add_argument("--out", default="/sp/v5_va/results/graft_proto_detail.json")
    a = ap.parse_args()
    refc = json.load(open(REF_C))
    std = json.load(open(STD))
    roots = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(a.anatomies, "*", "mesh_v4.json")))
    if a.only:
        roots = [r for r in roots if os.path.basename(r) in a.only.split(",")]
    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out))
    for root in roots:
        n = os.path.basename(root)
        if not os.path.exists(os.path.join(a.anatomies, "_iso", n + ".vtp")):
            print("no iso for", n)
            continue
        r = one(root, refc, std, not a.no_alone)
        res[n] = r
        for side in ("right", "left"):
            sg = r[side]["segments"]
            g_ = lambda k, t="frac_0.3mm": (sg[k][t] if sg.get(k) else float("nan"))
            print("%s %s: whole cons 0.3 %.4f 0.5 %.4f rel %.4f | real-only %.4f | raw %.4f | donor_real %.4f | donor_tube %.4f | seam %.4f | topbrain %.4f | junction %.4f" % (
                n[:34], side, g_("whole_va_conservative"), g_("whole_va_conservative", "frac_0.5mm"), g_("whole_va_conservative", "frac_rel_0.15"),
                g_("whole_va_real_only"), g_("whole_va_raw"), g_("donor_real"), g_("donor_tube_V1"), g_("seam_zone"), g_("topbrain"), g_("junction_stub_in_parent")), flush=True)
            if "A3_whole_va_v5_vs_v4" in r[side]:
                print("     A3 whole VA v5 (conservative, ostium->end) vs v4 (branch point->end):", json.dumps(r[side]["A3_whole_va_v5_vs_v4"]))
            if "A2" in r[side]:
                print("     A2 donor in situ / alone %.4f / %.4f (x%s); topbrain %.4f / %.4f (x%s)" % (
                    r[side]["A2"]["donor_in_situ"]["frac_0.3mm"], r[side]["A2"]["donor_alone"]["core"]["frac_0.3mm"], r[side]["A2"]["donor_ratio_0.3"],
                    r[side]["A2"]["topbrain_in_situ"]["frac_0.3mm"], r[side]["A2"]["topbrain_alone"]["core"]["frac_0.3mm"], r[side]["A2"]["topbrain_ratio_0.3"]))
            print("     grade (conservative):", json.dumps(r[side]["grade_in_situ_conservative"]))
            print("     bins from ostium:", [b["frac_0.3mm"] for b in r[side]["bins_from_ostium_40mm"]], "| from distal:", [b["frac_0.3mm"] for b in r[side]["bins_from_distal_40mm"]])
        print("   right_route_detail v5/v4:", json.dumps(r["right_route_detail"]["v5_over_v4"]))
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
