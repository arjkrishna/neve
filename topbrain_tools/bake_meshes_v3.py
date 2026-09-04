#!/usr/bin/env python3
"""Bake v3 collision meshes: v2 tubes + the patients' real surfaces (change 8).

Same outputs as bake_meshes_v2.py (collision_full.vtp, vessel_architecture_
collision.obj at --obj-tris, a report), with the real TopBrain siphon surface
and, for set B, the real Zenodo carotid lumen unioned in where the graft kept
them. Reads the section transforms the grafters wrote:

    set A   <anatomy>/graft_xform.json      sections: siphon
    set B   <anatomy>/provenance.json       xform: lower (CCA/ICA/ECA), siphon

The report (mesh_v3.json) splits the route deficit by section kind, because
the two are supposed to differ: tube sections should read ~0.1 mm as in v2,
real sections should show the spread of a real lumen round its inscribed
radius, as the shipped test anatomy does.

    python3 topbrain_tools/bake_meshes_v3.py --anatomies <dir> [--shard i/n]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/eve_training/eve_bench")
from sdf_mesher import SPACING_MM, decimate_to, iso_surface, mesh_stats, route_lumen, tube_field
from sdf_union import Section, add_real_sections, load_surface

MESH_NAME = "vessel_architecture_collision.obj"
FULL_NAME = "collision_full.vtp"
REPORT = "mesh_v3.json"
CATH_R, CONTACT = 0.35, 0.3
ECA_NAME = "Centerline curve - RECA.mrk"


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def slice_route(route, rad, s, lo, hi):
    m = (s >= lo - 1e-6) & (s <= hi + 1e-6)
    return route[m], rad[m]


def sections_for(root, br, route, rad):
    """Build the real-surface sections from what the grafter recorded."""
    s = arclen(route)
    out, kinds = [], {}
    xf_a = os.path.join(root, "graft_xform.json")
    prov = os.path.join(root, "provenance.json")
    if os.path.exists(xf_a):                                     # set A
        x = json.load(open(xf_a, encoding="utf-8"))["sections"][0]
        surf = load_surface(x["surface"], x["mirror"], x["R"], x["origin"], x["anchor"])
        p, r = slice_route(route, rad, s, x["route_from_mm"], s[-1])
        out.append(Section("siphon", surf, p, r, taper_start=True, taper_end=False))
        kinds["siphon"] = (x["route_from_mm"], s[-1])
    elif os.path.exists(prov):                                    # set B
        d = json.load(open(prov, encoding="utf-8"))
        x = d["xform"]
        lo = d["host_cut_mm"]
        seam2 = lo + d["cca_mm"] + d["ica_mm"]
        lx = x["lower"]
        ica_real_end = lo + d["cca_mm"] + min(d["ica_mm"], lx.get("ica_real_mm", d["ica_mm"]))
        lsurf = load_surface(lx["surface"], lx["mirror"], lx["R"], lx["origin"], lx["anchor"])
        p, r = slice_route(route, rad, s, lo, ica_real_end)
        out.append(Section("lower", lsurf, p, r, taper_start=True, taper_end=True))
        kinds["lower"] = (lo, ica_real_end)
        eca = [b for b in br if ECA_NAME in str(b.name)]
        if eca:
            out.append(Section("eca", lsurf, np.asarray(eca[0].coordinates, float),
                               np.asarray(eca[0].radii, float), taper_start=False, taper_end=True))
        sx = x["siphon"]
        ssurf = load_surface(sx["surface"], sx["mirror"], sx["R"], sx["origin"], sx["anchor"])
        p, r = slice_route(route, rad, s, seam2, s[-1])
        out.append(Section("siphon", ssurf, p, r, taper_start=True, taper_end=False))
        kinds["siphon"] = (seam2, s[-1])
    return out, kinds


def section_shape(mesh, route, rad, s, ranges, step_mm=4.0, n_dir=16, max_r=12.0):
    """How non-circular is the meshed lumen? At stations along the route, cast
    rays in n_dir directions in the plane normal to the centerline and read
    the wall distance. For a tube every ray reads the radius; for a real lumen
    the far wall of a bulb or an elliptical section reads more. Reports the
    ratio of the longest to the inscribed (MISR) radius and of the area-
    equivalent radius to MISR, per section kind."""
    out = {}
    for kind, lo, hi in ranges:
        rows = []
        for st in np.arange(lo + 2.0, hi - 2.0, step_mm):
            i = int(np.searchsorted(s, st))
            if i < 1 or i >= len(route) - 1:
                continue
            t = route[i + 1] - route[i - 1]
            t /= max(np.linalg.norm(t), 1e-9)
            a = np.cross(t, [1.0, 0, 0])
            if np.linalg.norm(a) < 0.2:
                a = np.cross(t, [0, 1.0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(t, a)
            rr = []
            for th in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
                u = np.cos(th) * a + np.sin(th) * b
                pts, _ = mesh.ray_trace(route[i], route[i] + u * max_r, first_point=True)
                if len(pts):
                    rr.append(float(np.linalg.norm(np.asarray(pts).reshape(-1)[:3] - route[i])))
            if len(rr) >= n_dir - 2:
                rr = np.asarray(rr)
                area_r = float(np.sqrt(np.mean(rr ** 2)))          # area-equivalent radius of the polygon of rays
                rows.append((rr.max() / max(rad[i], 1e-6), area_r / max(rad[i], 1e-6), rr.max() / max(rr.min(), 1e-6)))
        if rows:
            rows = np.asarray(rows)
            out[kind] = {"stations": int(len(rows)),
                         "max_over_misr": {"median": round(float(np.median(rows[:, 0])), 3), "p90": round(float(np.percentile(rows[:, 0], 90)), 3)},
                         "area_r_over_misr": {"median": round(float(np.median(rows[:, 1])), 3), "p90": round(float(np.percentile(rows[:, 1], 90)), 3)},
                         "max_over_min": {"median": round(float(np.median(rows[:, 2])), 3), "p90": round(float(np.percentile(rows[:, 2], 90)), 3)}}
    return out


def bake_one(root, spacing, full_tris, obj_tris, verbose):
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    t0 = time.time()
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
    route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)

    field = tube_field(br, spacing=spacing, verbose=verbose)
    secs, kinds = sections_for(root, br, route, rad)
    sec_stats = add_real_sections(field, secs, spacing, verbose=verbose)
    full = iso_surface(field)
    st_iso = mesh_stats(full)
    # The tree is one connected solid by construction (every branch overlaps
    # its parent), so a second component can only be a fragment: a side vessel
    # in the label clipped off by the capsule, or a sliver at the band edge.
    # Keep the largest and record what was dropped.
    if st_iso["comps"] > 1:
        conn = full.connectivity()
        rid = np.asarray(conn.cell_data["RegionId"])
        sizes = np.bincount(rid)
        keep = int(np.argmax(sizes))
        st_iso["dropped_fragments"] = sorted((sizes[rid != keep] if False else sizes[np.arange(len(sizes)) != keep]).tolist(), reverse=True)
        full = conn.extract_cells(np.nonzero(rid == keep)[0]).extract_surface().triangulate()
    full = decimate_to(full, full_tris)
    full.save(os.path.join(root, FULL_NAME))
    # Adaptive budget. Quadric decimation spends its error where it likes, and
    # on one anatomy it took a floored 1.0 mm neck from 0.89 mm (60 k) to
    # 0.63 mm (20 k) -- below what the catheter needs -- where v2's tube-only
    # mesh of the same centerlines had kept 0.84. If the budget cut is not
    # navigable, step the budget up until it is; the report records the count.
    obj_used = obj_tris
    for cand in (obj_tris, int(obj_tris * 1.5), obj_tris * 2, obj_tris * 3, full_tris):
        obj = decimate_to(full, cand)
        d0, ins0, body0, _ = route_lumen(obj, route)
        ok0 = ins0 & body0
        obj_used = cand
        if ok0.any() and d0[ok0].min() - CONTACT >= CATH_R:
            break
    pv.save_meshio(os.path.join(root, MESH_NAME), obj)

    rep = {"spacing_mm": spacing, "seconds": round(time.time() - t0, 1), "iso_surface": st_iso,
           "sections": sec_stats, "full": mesh_stats(full), "obj": mesh_stats(obj), "obj_tris_budget": obj_used}
    s = arclen(route)
    for tag, m in (("full", full), ("obj", obj)):
        d, ins, body, _ = route_lumen(m, route)
        ok = ins & body
        if ok.any():
            i = int(np.argmin(np.where(ok, d, np.inf)))
            rep[tag].update(lumen_min_mm=round(float(d[i]), 3), lumen_min_at_mm=round(float(s[i]), 1),
                            declared_there_mm=round(float(rad[i]), 3),
                            median_deficit_mm=round(float(np.median(rad[ok] - d[ok])), 3),
                            route_pts_outside=int((~ins & body).sum()),
                            navigable=bool(d[ok].min() - CONTACT >= CATH_R))
            real = np.zeros(len(s), bool)
            for lo, hi in kinds.values():
                real |= (s >= lo) & (s <= hi)
            for kind, msk in (("tube", ~real), ("real", real)):
                q = ok & msk
                if q.any():
                    dd = rad[q] - d[q]
                    rep[tag]["deficit_%s" % kind] = {"median": round(float(np.median(dd)), 3),
                                                     "p10": round(float(np.percentile(dd, 10)), 3),
                                                     "p90": round(float(np.percentile(dd, 90)), 3),
                                                     "n": int(q.sum())}
        else:
            rep[tag].update(route_pts_outside=int(body.sum()), navigable=False)
    # shape: real sections against the tube part of the same route
    real_ranges = [(k, lo, hi) for k, (lo, hi) in kinds.items() if k != "eca"]
    lo_real = min([lo for _, lo, _ in real_ranges]) if real_ranges else s[-1]
    ranges = [("tube", 0.0, lo_real)] + real_ranges
    rep["shape"] = section_shape(obj, route, rad, s, ranges)
    with open(os.path.join(root, REPORT), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", required=True)
    ap.add_argument("--spacing", type=float, default=SPACING_MM)
    ap.add_argument("--full-tris", type=int, default=60000)
    ap.add_argument("--obj-tris", type=int, default=20000)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    folders = sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))
    if a.only:
        want = set(a.only.split(","))
        folders = [f for f in folders if os.path.basename(os.path.dirname(f)) in want]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        folders = folders[i::n]
    if not folders:
        print("no anatomies under %s" % a.anatomies)
        return 1
    print("baking %d v3 meshes at %.2f mm (full %d / obj %d tris)\n" % (len(folders), a.spacing, a.full_tris, a.obj_tris))
    print("%-44s %6s %5s %5s %9s %8s %8s %8s %8s %7s %5s" % ("anatomy", "tris", "comps", "open", "lumen_min", "def_tube", "def_real", "shp_tube", "shp_real", "navig", "sec"))
    bad = 0
    for f in folders:
        root = os.path.dirname(f)
        name = os.path.basename(root)
        if os.path.exists(os.path.join(root, REPORT)) and not a.force:
            print("%-44s kept" % name[:44]); continue
        try:
            rep = bake_one(root, a.spacing, a.full_tris, a.obj_tris, a.verbose)
        except Exception as e:                                       # noqa: BLE001
            bad += 1
            import traceback
            print("%-44s FAILED %s: %s" % (name[:44], type(e).__name__, str(e)[:70]))
            if a.verbose:
                traceback.print_exc()
            continue
        o = rep["obj"]
        sh = rep.get("shape", {})
        def shp(k):
            return sh.get(k, {}).get("max_over_misr", {}).get("median", -1)
        print("%-44s %6d %5d %5d %9.2f %8.2f %8.2f %8.2f %8.2f %7s %5.0f" % (
            name[:44], o["tris"], o["comps"], o["open_edges"], o.get("lumen_min_mm", -1),
            o.get("deficit_tube", {}).get("median", -9), o.get("deficit_real", {}).get("median", -9),
            shp("tube"), shp("siphon") if "siphon" in sh else shp("lower"),
            "yes" if o.get("navigable") else "NO", rep["seconds"]), flush=True)
    print("\ndone, %d failed" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
