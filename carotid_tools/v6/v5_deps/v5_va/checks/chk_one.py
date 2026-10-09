#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\chk_one.py (= /sp/v5_va/checks/chk_one.py in the build containers)
"""v5_checks items 1 (strict mesh audit) and 2 (route + VA wall detail) for ONE baked v5 anatomy.

    python3 /sp/v5_va/checks/chk_one.py <name> [--force]          (neve-build-meshlab; /repo mounted READ-ONLY)
Writes /sp/v5_va/checks/out/<name>.json (the done-marker, written last); skips the anatomy if it exists.

Item 1 -- v4's own audit code (/sp/v4_build/audit/topo_audit.py, imported unchanged; independent of the baker's
genus()/is_clean()): pymeshlab topological measures raw and after welding coincident vertices, pymeshlab
self-intersecting faces, numpy Euler / orientation / duplicate-face cross-check, triangle geometry, on the OBJ,
collision_full.vtp and the undecimated union (iso_v5/<name>.vtp); budgets (OBJ <= 23 500, full 60k); v4's route
check on the OBJ and the full mesh (RCCA and LCCA: every route point inside by winding number, lumen_min - 0.3 mm
contact >= 0.35 mm wire radius, exact point-triangle distances); every centreline point inside by winding number
(ALL points incl. the end caps, and v4's caps-excluded count); the pymeshlab OBJ loader's view; and the distance of
the full mesh's vertices to the union (the 60k mesh is a decimation of THIS union).

Item 2 -- not the bake's measure code:
  * right_route_detail recomputed on the undecimated union with v4's function (bake_meshes_v4.right_route_detail),
    compared with mesh_v5.json (the bake's record) and mesh_v4.json (v4 as shipped): identical / ratio.
  * v4's exact union rebuilt from mesh_v4.json (the union part of ref_v4_union.rebuild, copied; check: its
    right_route_detail equals mesh_v4.json) for the v4 side of every comparison.
  * ownership as right_route_detail / va_metrics.measure_union_branch (24 nearest samples of ALL branches, the largest
    r - |x - c| wins), dep = | |x - c| - r |, area fractions > 0.3 / 0.5 mm and > 0.15 / 0.25 r; 'core' drops 2 mm at
    both ends.  Per VA:
      v5  donor stretch [lo, cut] as baked; donor real = minus the 25 mm seam zone minus the tube windows; the
          windows alone; the seam zone (own row); TopBrain [cut, L]; TopBrain real (minus seam, minus windows);
          whole VA from the VA origin (v4's reference range) and from the ostium.
      v4  host tube [origin, cut4 - 25], ramp, TopBrain [cut4, L4], whole VA; and the TopBrain piece over the SAME
          distal range as v5's TopBrain real (aligned from the distal end).
  * TopBrain piece unchanged: surface file + mirror; centreline piece v5 vs v4 (length, rigid Kabsch RMS, radii).
  * RCCA fork / siphon (radius-relative too) and "identical beyond D mm of every VA" (D = 4, 6, 8).
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v4_build/audit")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import topo_audit as TA  # noqa: E402  (v4's audit, unchanged)

A5 = "/repo/carotid_data/anatomies_v5"
V4 = "/repo/carotid_data/anatomies_v4"
ISO = "/repo/carotid_data/va_sources/v5_reg/iso_v5"
OUT = "/sp/v5_va/checks/out"
OBJ_BUDGET, FULL_BUDGET = 23500, 60000
SEAM_MM, END_EXCL = 25.0, 2.0


def vm_hwm_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 1048576.0, 3)
    return -1


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


# ------------------------------------------------------------------ item 1
def enclosure_all(V, F, br):
    """Every centreline point (end caps included) by winding number; and v4's caps-excluded count."""
    P = np.vstack([p for _, p, _ in br])
    W = TA.winding(P, V, F)
    out = np.abs(W) <= 0.5
    per = {}
    k = 0
    for tag, p, _ in br:
        o = int(out[k:k + len(p)].sum())
        if o:
            per[tag] = o
        k += len(p)
    o2, n2, where, wmed = TA.enclosure(V, F, br)
    return {"all_points": int(len(P)), "all_outside": int(out.sum()), "outside_by_branch": per,
            "outside_where": P[out][:10].round(2).tolist(), "median_winding": float(np.median(W)),
            "caps_excluded_checked": n2, "caps_excluded_outside": o2}


def audit_mesh(V, F, loc, routes=True, br=None):
    r = TA.check_mesh(V, F, loc)
    r["clean_pml_raw"] = TA.is_clean_pml(r["pml_raw"])
    r["clean_pml_welded"] = TA.is_clean_pml(r["pml_merged"])
    if routes:
        rr, _ = TA.route_check(V, F, br)
        r["routes"] = rr
        r["enclosure"] = enclosure_all(V, F, br)
    return r


def item1(name, root, rep):
    import pymeshlab
    import pyvista as pv
    out = {}
    br = TA.load_branches(os.path.join(root, "Centrelines_comb"))
    loc = TA.Locator(br, rep.get("spans"))
    t = time.time()
    V, F, extra = TA.read_obj(os.path.join(root, "vessel_architecture_collision.obj"))
    out["obj_file"] = dict(extra, v=int(len(V)), f=int(len(F)))
    out["obj"] = audit_mesh(V, F, loc, True, br)
    msl = pymeshlab.MeshSet()
    msl.load_new_mesh(os.path.join(root, "vessel_architecture_collision.obj"))
    out["obj_pml_loader"] = TA.pml_topo(msl)
    out["obj_pml_loader_clean"] = TA.is_clean_pml(out["obj_pml_loader"])
    del msl
    out["t_obj_s"] = round(time.time() - t, 1)
    t = time.time()
    Vf, Ff, ex = TA.read_vtp(os.path.join(root, "collision_full.vtp"))
    out["full_file"] = dict(ex, v=int(len(Vf)), f=int(len(Ff)))
    out["full"] = audit_mesh(Vf, Ff, loc, True, br)
    out["t_full_s"] = round(time.time() - t, 1)
    t = time.time()
    iso = pv.read(os.path.join(ISO, name + ".vtp"))
    fa = np.asarray(iso.faces)
    Fi = fa.reshape(-1, 4)[:, 1:] if (len(fa) % 4 == 0 and (fa.reshape(-1, 4)[:, 0] == 3).all()) \
        else np.asarray(iso.triangulate().faces).reshape(-1, 4)[:, 1:]
    Vi = np.asarray(iso.points, float)
    out["union_file"] = {"v": int(len(Vi)), "f": int(len(Fi))}
    out["union"] = audit_mesh(Vi, np.asarray(Fi, np.int64), loc, False)
    # the 60k mesh is a decimation of this union: distance of its vertices to the union surface
    _, cp = iso.find_closest_cell(Vf, return_closest_point=True)
    d = np.linalg.norm(Vf - cp, axis=1)
    out["full_to_union_mm"] = {"p50": round(float(np.percentile(d, 50)), 4), "p99": round(float(np.percentile(d, 99)), 4),
                               "max": round(float(d.max()), 4)}
    _, cp = iso.find_closest_cell(V, return_closest_point=True)
    d = np.linalg.norm(V - cp, axis=1)
    out["obj_to_union_mm"] = {"p50": round(float(np.percentile(d, 50)), 4), "p99": round(float(np.percentile(d, 99)), 4),
                              "max": round(float(d.max()), 4)}
    out["t_union_s"] = round(time.time() - t, 1)
    o, f, u = out["obj"], out["full"], out["union"]
    V4o, F4o, _ = TA.read_obj(os.path.join(V4, name, "vessel_architecture_collision.obj"))
    v4_vol = TA.geom_numpy(V4o, F4o)["signed_volume_mm3"]
    out["v4_obj_tris"] = int(len(F4o))
    del V4o, F4o
    gates = {
        "obj_clean_raw": o["clean_pml_raw"], "obj_clean_welded": o["clean_pml_welded"],
        "full_clean_raw": f["clean_pml_raw"], "full_clean_welded": f["clean_pml_welded"],
        "union_clean_welded": u["clean_pml_welded"], "obj_pml_loader_clean": out["obj_pml_loader_clean"],
        "numpy_obj_ok": o["numpy"]["components"] == 1 and o["numpy"]["chi"] == 2 and o["numpy"]["boundary_edges"] == 0
        and o["numpy"]["nonmanifold_edges"] == 0 and o["numpy"]["misoriented_edges"] == 0 and o["numpy"]["duplicate_faces"] == 0,
        "numpy_full_ok": f["numpy"]["components"] == 1 and f["numpy"]["chi"] == 2 and f["numpy"]["boundary_edges"] == 0
        and f["numpy"]["nonmanifold_edges"] == 0 and f["numpy"]["misoriented_edges"] == 0 and f["numpy"]["duplicate_faces"] == 0,
        "selfint": {"obj": o["selfint_faces"], "full": f["selfint_faces"], "union": u["selfint_faces"],
                    "obj_welded": o.get("selfint_faces_merged"), "full_welded": f.get("selfint_faces_merged"),
                    "union_welded": u.get("selfint_faces_merged")},
        "obj_tris": int(len(F)), "full_tris": int(len(Ff)), "union_tris": int(len(Fi)),
        "obj_budget_ok": int(len(F)) <= OBJ_BUDGET, "full_budget_ok": 0.999 * FULL_BUDGET <= int(len(Ff)) <= FULL_BUDGET,
        "obj_file_clean": not any(k in out["obj_file"] for k in ("non_tri_faces",)) and out["obj_file"]["negative_indices"] == 0,
        "full_file_all_tri": bool(out["full_file"]["all_tri"]) and out["full_file"]["n_lines"] == 0
        and out["full_file"]["n_verts"] == 0 and out["full_file"]["n_strips"] == 0,
        "routes_navigable_obj": all(o["routes"][k].get("navigable") and o["routes"][k].get("route_pts_outside", 1) == 0
                                    for k in ("RCCA", "LCCA")),
        "routes_navigable_full": all(f["routes"][k].get("navigable") and f["routes"][k].get("route_pts_outside", 1) == 0
                                     for k in ("RCCA", "LCCA")),
        "route_lumen_min_obj": {k: o["routes"][k].get("lumen_min_mm") for k in ("RCCA", "LCCA")},
        "centreline_outside_obj_all": o["enclosure"]["all_outside"],
        "centreline_outside_full_all": f["enclosure"]["all_outside"],
        "centreline_outside_obj_caps_excl": o["enclosure"]["caps_excluded_outside"],
        "centreline_points": o["enclosure"]["all_points"],
        # orientation: coherent (numpy misoriented_edges 0, above) and the same sign convention as v4's shipped OBJ
        "signed_volume_mm3": {"obj": round(o["geom"]["signed_volume_mm3"], 1), "full": round(f["geom"]["signed_volume_mm3"], 1),
                              "union": round(u["geom"]["signed_volume_mm3"], 1), "v4_obj": round(v4_vol, 1)},
        "orientation_as_v4": bool(np.sign(o["geom"]["signed_volume_mm3"]) == np.sign(v4_vol)
                                  == np.sign(f["geom"]["signed_volume_mm3"])),
    }
    si0 = all(v in (0, None) for v in gates["selfint"].values())
    gates["strict_pass"] = bool(gates["obj_clean_raw"] and gates["obj_clean_welded"] and gates["full_clean_raw"]
                                and gates["full_clean_welded"] and gates["union_clean_welded"] and gates["obj_pml_loader_clean"]
                                and gates["numpy_obj_ok"] and gates["numpy_full_ok"] and si0 and gates["obj_budget_ok"]
                                and gates["full_budget_ok"] and gates["obj_file_clean"] and gates["full_file_all_tri"]
                                and gates["routes_navigable_obj"] and gates["centreline_outside_obj_all"] == 0
                                and gates["centreline_outside_full_all"] == 0 and gates["orientation_as_v4"])
    out["gates"] = gates
    return out, iso


# ------------------------------------------------------------------ item 2
class Own:
    """right_route_detail's ownership on a union, computed once."""

    def __init__(self, mesh, samp):
        self.cen = np.asarray(mesh.cell_centers().points)
        self.A = np.asarray(mesh.compute_cell_sizes(length=False, volume=False)["Area"])
        dd, ii = cKDTree(samp.p).query(self.cen, k=24)
        best = ii[np.arange(len(ii)), np.argmax(samp.r[ii] - dd, axis=1)]
        self.b, self.s, self.r = samp.b[best], samp.s[best], samp.r[best]
        self.dep = np.abs(np.linalg.norm(self.cen - samp.p[best], axis=1) - samp.r[best])
        self.samp = samp

    def mask(self, bidx, lo, hi, core=True, excl=()):
        Lb = float(self.samp.s[self.samp.b == bidx].max())
        e = END_EXCL if core else 0.0
        m = (self.b == bidx) & (self.s >= lo + e) & (self.s <= min(hi, Lb) - e)
        for a, b in excl:
            m &= ~((self.s >= a) & (self.s <= b))
        return m

    def frac(self, m):
        if not m.any():
            return None
        a, x, r = self.A[m], self.dep[m], self.r[m]
        tot = float(a.sum()) or 1.0
        d = {"area_mm2": round(float(a.sum()), 1)}
        for t in (0.3, 0.5):
            d["frac_%.1fmm" % t] = round(float(a[x > t].sum()) / tot, 4)
        for q in (0.15, 0.25):
            d["frac_rel_%.2f" % q] = round(float(a[x > q * r].sum()) / tot, 4)
        return d

    def seg(self, bidx, lo, hi, core=True, excl=()):
        if hi - lo < (2 * END_EXCL + 1.0 if core else 0.5):
            return None
        r = self.frac(self.mask(bidx, lo, hi, core, excl))
        if r:
            r["range_mm"] = [round(lo, 2), round(hi, 2)]
        return r


def rebuild_v4(name):
    """The union part of ref_v4_union.rebuild (copied): v4's accepted union from mesh_v4.json."""
    import bake_meshes_v4 as B
    from eve_bench.dualdevicenav import load_branches
    from sdf_mesher import SPACING_MM, iso_surface, tube_field
    from sdf_union_v4 import Samples, add_real_sections_owned
    root = os.path.join(V4, name)
    rep = json.load(open(os.path.join(root, "mesh_v4.json"), encoding="utf-8"))
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs, spans = B.sections_for(root, br)
    tube_only = rep.get("sections_tube_only") or []
    secs = [s for s in secs if s.name not in tube_only]
    spacing = rep.get("spacing_mm", SPACING_MM)
    margin = rep.get("margin_mm")
    sup = {k: [tuple(x) for x in v] for k, v in (rep.get("suppressed_stretches_mm") or {}).items()}
    field = tube_field(br, spacing=spacing)
    if margin is not None:
        add_real_sections_owned(field, secs, samp, spacing, margin, suppress=sup)
    if (rep.get("far_clamp") or {}).get("used"):
        far = field.values < B.FAR_BELOW
        field.values[far] = -2.0 * spacing
    iso, _, _ = B.keep_largest(iso_surface(field))
    del field
    detail = B.right_route_detail(iso, samp, root)
    rec = rep["right_route_detail"]
    diff = max(abs(detail[g][t] - rec[g][t]) for g in detail for t in detail[g])
    return iso, samp, br, rep, detail, float(diff)


def bpr(br, tag):
    for b in br:
        if str(b.name).endswith(tag + ".mrk") or str(b.name).endswith(tag):
            return np.asarray(b.coordinates, float), np.asarray(b.radii, float)
    raise KeyError(tag)


def origin_s(br, tag, sub):
    P, _ = bpr(br, tag)
    sc, _ = bpr(br, sub)
    s = arclen(P)
    return float(s[int(np.argmin(np.linalg.norm(P - sc[0], axis=1)))])


def kabsch_rms(A, B_):
    a, b = A - A.mean(0), B_ - B_.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    return float(np.sqrt(np.mean(np.sum((a @ R - b) ** 2, axis=1))))


def distal_resample(p, r, s0, step=0.5):
    """Points of the piece s >= s0, resampled by distance from the distal end."""
    s = arclen(p)
    m = s >= s0 - 1e-9
    q, rr, ss = p[m], r[m], s[m]
    dist = ss[-1] - ss
    return q, rr, dist


def tb_geometry(p5, r5, cut5, p4, r4, cut4):
    q5, rr5, d5 = distal_resample(p5, r5, cut5)
    q4, rr4, d4 = distal_resample(p4, r4, cut4)
    Lc = min(d5.max(), d4.max())
    g = np.arange(0.0, Lc, 0.5)
    A = np.stack([np.interp(g, d5[::-1], q5[::-1, k]) for k in range(3)], 1)
    Bq = np.stack([np.interp(g, d4[::-1], q4[::-1, k]) for k in range(3)], 1)
    ra = np.interp(g, d5[::-1], rr5[::-1])
    rb = np.interp(g, d4[::-1], rr4[::-1])
    return {"len_v5_mm": round(float(d5.max()), 2), "len_v4_mm": round(float(d4.max()), 2),
            "common_mm": round(float(Lc), 2), "kabsch_rms_mm": round(kabsch_rms(A, Bq), 4),
            "radius_max_abs_diff_mm": round(float(np.abs(ra - rb).max()), 4)}


def intervals(x):
    return [(float(a), float(b)) for a, b in (x or [])]


def route_cells(O, rc, cut):
    return (O.b == rc) & (O.s >= cut)


def item2(name, root, rep, iso5):
    import bake_meshes_v4 as B
    from eve_bench.dualdevicenav import load_branches
    from sdf_union_v4 import Samples
    out = {}
    br5 = load_branches(os.path.join(root, "Centrelines_comb"))
    samp5 = Samples(br5)
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    prov4 = json.load(open(os.path.join(V4, name, "provenance.json"), encoding="utf-8"))
    rep4 = json.load(open(os.path.join(V4, name, "mesh_v4.json"), encoding="utf-8"))
    t = time.time()
    d5 = B.right_route_detail(iso5, samp5, root)
    rec5, rec4 = rep["right_route_detail"], rep4["right_route_detail"]
    out["right_route_detail"] = {
        "recomputed_v5": d5, "mesh_v5_record": rec5, "mesh_v4_record": rec4,
        "recomputed_equals_v5_record": d5 == rec5,
        "identical_v4": d5 == rec4,
        "ratio_v5_v4": {g: {k: (round(d5[g][k] / rec4[g][k], 4) if rec4[g][k] else None) for k in rec4[g]} for g in rec4},
        "max_abs_diff_v4": round(max(abs(d5[g][k] - rec4[g][k]) for g in rec4 for k in rec4[g]), 4)}
    O5 = Own(iso5, samp5)
    out["t_rrd_own5_s"] = round(time.time() - t, 1)
    t = time.time()
    iso4, samp4, br4, _, d4, diff4 = rebuild_v4(name)
    out["v4_rebuild"] = {"right_route_detail": d4, "max_abs_diff_vs_mesh_v4": round(diff4, 5), "union_tris": int(iso4.n_cells)}
    O4 = Own(iso4, samp4)
    del iso4
    out["t_v4_rebuild_s"] = round(time.time() - t, 1)
    # RCCA (v3 standard, radius-relative too)
    cut5, cut4 = float(prov.get("host_cut_mm", 0.0)), float(prov4.get("host_cut_mm", 0.0))
    rc5, rc4 = samp5.index_of("RCCA"), samp4.index_of("RCCA")
    out["RCCA"] = {"v5": {"fork": O5.frac(route_cells(O5, rc5, cut5) & (O5.s < 130.0)),
                          "siphon": O5.frac(route_cells(O5, rc5, cut5) & (O5.s >= 130.0))},
                   "v4": {"fork": O4.frac(route_cells(O4, rc4, cut4) & (O4.s < 130.0)),
                          "siphon": O4.frac(route_cells(O4, rc4, cut4) & (O4.s >= 130.0))}}
    # identical beyond D mm of every v4 / v5 VA sample
    vap = np.vstack([s_.p[(s_.b == s_.index_of("RVA")) | (s_.b == s_.index_of("LVA"))] for s_ in (samp4, samp5)])
    vt = cKDTree(vap)
    m5, m4 = route_cells(O5, rc5, cut5), route_cells(O4, rc4, cut4)
    dist5, dist4 = vt.query(O5.cen[m5])[0], vt.query(O4.cen[m4])[0]
    away = {}
    for D in (4.0, 6.0, 8.0):
        row = {}
        for seg, f5, f4 in (("fork", O5.s[m5] < 130.0, O4.s[m4] < 130.0), ("siphon", O5.s[m5] >= 130.0, O4.s[m4] >= 130.0)):
            k5, k4 = np.zeros(len(O5.s), bool), np.zeros(len(O4.s), bool)
            k5[np.nonzero(m5)[0][(dist5 > D) & f5]] = True
            k4[np.nonzero(m4)[0][(dist4 > D) & f4]] = True
            a, b = O5.frac(k5), O4.frac(k4)
            row[seg] = {"v5": a, "v4": b, "identical": a == b}
        row["identical"] = row["fork"]["identical"] and row["siphon"]["identical"]
        away["%dmm" % D] = row
    out["route_away_from_vas"] = away
    out["identical_beyond_mm"] = 0 if out["right_route_detail"]["identical_v4"] else \
        next((D for D in (4, 6, 8) if away["%dmm" % D]["identical"]), None)
    # VAs
    x5, x4 = prov["xform"], prov4["xform"]
    sup5 = rep.get("suppressed_stretches_mm") or {}
    sup4 = rep4.get("suppressed_stretches_mm") or {}
    v5va = prov.get("v5_va") or {}
    tub = (rep.get("ladder_v5b") or {}).get("va_tubed") or {}
    vas = {}
    for side, tag, sub, ck, tk in (("right", "RVA", "(17).mrk", "vaC_R", "va_R"), ("left", "LVA", "(19).mrk", "vaC_L", "va_L")):
        d = {}
        b5, b4 = samp5.index_of(tag), samp4.index_of(tag)
        p5, r5 = bpr(br5, tag)
        p4, r4 = bpr(br4, tag)
        L5, L4 = float(arclen(p5)[-1]), float(arclen(p4)[-1])
        s05, s04 = origin_s(br5, tag, sub), origin_s(br4, tag, sub)
        tcut5, tcut4 = float(x5[tk]["cut_mm"]), float(x4[tk]["cut_mm"])
        w5_tb, w4_tb = intervals(sup5.get(tk)), intervals(sup4.get(tk))
        # v4 side (rebuilt union)
        d["v4"] = {"origin_s_mm": round(s04, 2), "cut_mm": round(tcut4, 2), "length_mm": round(L4, 2),
                   "host_tube": O4.seg(b4, s04, tcut4 - SEAM_MM), "ramp": O4.seg(b4, tcut4 - SEAM_MM, tcut4),
                   "topbrain": O4.seg(b4, tcut4, L4), "whole_va": O4.seg(b4, s04, L4),
                   "topbrain_windows": w4_tb, "topbrain_real_surface": x4[tk].get("real_surface", True) is not False
                   and tk not in (rep4.get("sections_tube_only") or [])}
        donor = ck in x5
        d["donor"] = donor
        d["v4_va_kept"] = not donor
        if not donor:
            d["why"] = (v5va.get(side) or {}).get("why")
            d["v5"] = {"origin_s_mm": round(s05, 2), "length_mm": round(L5, 2), "host_tube": O5.seg(b5, s05, tcut5 - SEAM_MM),
                       "topbrain": O5.seg(b5, tcut5, L5), "whole_va": O5.seg(b5, s05, L5),
                       "centreline_identical_v4": bool(p5.shape == p4.shape and np.array_equal(p5, p4) and np.array_equal(r5, r4))}
            vas[side] = d
            continue
        sd = v5va.get(side) or {}
        xf = x5[ck]
        lo, hi = float(xf["lo_mm"]), float(xf["hi_mm"])
        ost = float(sd.get("ostium_at_mm", lo))
        ramp = (sd.get("ramp") or {}).get("side", "donor")
        seam = (tcut5 - SEAM_MM, tcut5) if ramp == "donor" else (tcut5, min(tcut5 + SEAM_MM, L5))
        win = intervals(sup5.get(ck))
        tb_lo = seam[1] if ramp != "donor" else tcut5
        tb_excl = list(w5_tb)
        v5 = {"donor": "%s %s" % (xf.get("donor_id"), xf.get("donor_side")), "origin_s_mm": round(s05, 2),
              "ostium_mm": round(ost, 2), "lo_mm": round(lo, 2), "cut_mm": round(tcut5, 2), "hi_mm": round(hi, 2),
              "length_mm": round(L5, 2), "ramp_side": ramp, "seam_zone_mm": [round(seam[0], 2), round(seam[1], 2)],
              "windows_mm": win, "tubed_mm": round(sum(b - a for a, b in win), 2),
              "donor_stretch_mm": round(hi - lo, 2),
              "tubed_frac": round(sum(b - a for a, b in win) / max(hi - lo, 1e-9), 4),
              "real_surface_flag": xf.get("real_surface", True) is not False,
              "tube_only": ck in (rep.get("sections_tube_only") or []),
              "ladder_record": tub.get(ck),
              "topbrain_windows": w5_tb}
        v5["junction_stub"] = O5.seg(b5, s05, ost, core=False)
        v5["v1_tube_before_real"] = O5.seg(b5, ost, lo) if lo > ost + 5.0 else None
        v5["donor_stretch_as_baked"] = O5.seg(b5, lo, tcut5)
        v5["donor_real"] = O5.seg(b5, lo, tcut5, excl=[seam] + win)
        v5["windows_only"] = O5.frac(np.logical_or.reduce([O5.mask(b5, a, b, core=False) for a, b in win])) if win else None
        v5["seam_zone"] = O5.seg(b5, seam[0], seam[1], core=False)
        v5["topbrain"] = O5.seg(b5, tcut5, L5)
        v5["topbrain_real"] = O5.seg(b5, tb_lo, L5, excl=tb_excl)
        v5["whole_va_from_origin"] = O5.seg(b5, s05, L5)
        v5["whole_va_from_ostium"] = O5.seg(b5, ost, L5)
        d["v5"] = v5
        # TopBrain piece unchanged
        tg = tb_geometry(p5, r5, tcut5, p4, r4, tcut4)
        tg["surface_same"] = os.path.basename(str(x5[tk].get("surface"))) == os.path.basename(str(x4[tk].get("surface")))
        tg["mirror_same"] = list(x5[tk].get("mirror", [])) == list(x4[tk].get("mirror", []))
        tg["R_det"] = round(float(np.linalg.det(np.asarray(x5[tk]["R"], float))), 6)
        a5 = L5 - tb_lo            # aligned from the distal end
        lo4 = L4 - a5
        tb4 = O4.seg(b4, max(lo4, tcut4), L4, excl=w4_tb)
        tb5 = v5["topbrain_real"]
        tg["detail_aligned"] = {"v5_range_mm": [round(tb_lo, 2), round(L5, 2)], "v4_range_mm": [round(max(lo4, tcut4), 2), round(L4, 2)],
                                "v5": tb5, "v4": tb4,
                                "ratio_0.3mm": round(tb5["frac_0.3mm"] / tb4["frac_0.3mm"], 4) if (tb5 and tb4 and tb4["frac_0.3mm"]) else None,
                                "ratio_rel_0.15": round(tb5["frac_rel_0.15"] / tb4["frac_rel_0.15"], 4) if (tb5 and tb4 and tb4["frac_rel_0.15"]) else None}
        d["topbrain_unchanged"] = tg
        vas[side] = d
    out["va"] = vas
    # sanity: Own reproduces va_metrics.measure_union_branch on one stretch
    try:
        sys.path.insert(0, "/sp/v5_va")
        import va_metrics as vm
        bi = samp5.index_of("RVA")
        L = float(samp5.s[samp5.b == bi].max())
        ref = vm.measure_union_branch(iso5, samp5, bi, 10.0, L)["core"]
        mine = O5.seg(bi, 10.0, L)
        out["own_vs_measure_union_branch"] = {"ref": ref, "mine": mine,
                                              "equal": all(abs(ref[k] - mine[k]) < 1e-3 for k in ("frac_0.3mm", "frac_0.5mm", "frac_rel_0.15"))}
    except Exception as e:  # noqa: BLE001
        out["own_vs_measure_union_branch"] = {"error": "%s: %s" % (type(e).__name__, e)}
    return out


def main():
    name = sys.argv[1]
    force = "--force" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, name + ".json")
    if os.path.exists(dst) and not force:
        print("done already", dst)
        return 0
    T0 = time.time()
    root = os.path.join(A5, name)
    rep = json.load(open(os.path.join(root, "mesh_v5.json"), encoding="utf-8"))
    rec = {"name": name, "started": time.strftime("%F %T")}
    try:
        t = time.time()
        rec["item1"], iso = item1(name, root, rep)
        rec["t_item1_s"] = round(time.time() - t, 1)
        print("item1", json.dumps(rec["item1"]["gates"]), flush=True)
        t = time.time()
        rec["item2"] = item2(name, root, rep, iso)
        rec["t_item2_s"] = round(time.time() - t, 1)
        print("item2 rrd", json.dumps(rec["item2"]["right_route_detail"]["ratio_v5_v4"]),
              "identical", rec["item2"]["right_route_detail"]["identical_v4"], flush=True)
    except Exception as e:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        rec["tb"] = traceback.format_exc()
        print(rec["tb"], flush=True)
    rec["seconds"] = round(time.time() - T0, 1)
    rec["peak_rss_gb"] = vm_hwm_gb()
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (
            bool(o) if isinstance(o, np.bool_) else float(o)))
    if "error" in rec:
        os.replace(tmp, os.path.join(OUT, name + ".error.json"))
        return 1
    os.replace(tmp, dst)
    print("DONE %s %.0f s peak %.2f GB" % (name, rec["seconds"], rec["peak_rss_gb"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
