#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\obj_fix.py (= /sp/v5_va/bake_full/obj_fix.py in the build containers)
"""v5_bake_full: an anatomy whose 60k mesh has NO clean decimation to the 23.5k collision budget (v4's obj_ladder
then steps the budget up to 35250 -- outside the v4/v5 standard of a 23.5k OBJ). Stated diff: other decimation
SOURCES / targets at or under the 23.5k budget, v4's own clean_decimate on each, accepted only with v4's own
acceptance (is_clean: one closed genus-0 two-manifold, 0 self-intersecting faces; both routes navigable
(route_report); every centreline point inside (enclosure)). Tried in this order, first accepted wins:
  1. the anatomy's collision_full.vtp (60k), targets 23450, 23400, 23300, 23200, 23000 (vcg's collapse order
     depends on the target; VTK's does not)
  2. the undecimated union (iso_v5/<name>.vtp), raw_iso=True (vcg marching-cubes collapse first), target 23500,
     then 23400
  3. a fresh 80k decimation of the union (clean_decimate raw_iso=True), then 23500 / 23400 from it
The union, the gates and collision_full.vtp are unchanged; only the OBJ and its report entries are rewritten:
mesh_v5.json obj / obj_tris_budget / routes / centerline_pts_* / decimation.obj* / pass + "obj_fix" (what was tried),
and the provenance bake_v5 obj fields. The measure is re-run afterwards (measure_full_one.py, marker removed).
    python3 obj_fix.py <name>        (neve-build-meshlab)
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402
B = V.B
from sdf_mesher import mesh_stats  # noqa: E402
from sdf_union_v4 import genus  # noqa: E402

A5 = "/repo/carotid_data/anatomies_v5"
ISO = "/repo/carotid_data/va_sources/v5_reg/iso_v5"
BF = "/repo/carotid_data/va_sources/v5_reg/bake_full"
BUDGET = 23500


def main():
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    name = sys.argv[1]
    root = os.path.join(A5, name)
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    rp = os.path.join(root, V.REPORT_V5)
    rep = json.load(open(rp))
    if rep["obj_tris_budget"] <= BUDGET:
        print("budget already met", rep["obj_tris_budget"])
        return 0
    keep = os.path.join(BF, "_objfix", name)
    os.makedirs(keep, exist_ok=True)
    for f in (B.MESH_NAME, V.REPORT_V5, "provenance.json"):
        if not os.path.exists(os.path.join(keep, f + ".before")):
            shutil.copyfile(os.path.join(root, f), os.path.join(keep, f + ".before"))
    full = pv.read(os.path.join(root, B.FULL_NAME))
    iso = None
    tried = []

    def accept(m, src, target, path):
        ok_clean = B.is_clean(m)
        rr, ok_routes = B.route_report(m, br) if ok_clean else ({}, False)
        out_n, tot_n = B.enclosure(m, br) if ok_clean and ok_routes else (-1, -1)
        rec = {"source": src, "target": target, "path": path, "tris": int(m.n_cells), "clean": ok_clean,
               "routes_ok": ok_routes, "centerline_pts_outside": out_n}
        tried.append(rec)
        print(rec, flush=True)
        return ok_clean and ok_routes and out_n == 0 and m.n_cells <= BUDGET, rr, out_n, tot_n

    t0 = time.time()
    found = None
    plan = [("collision_full 60k", t) for t in (23450, 23400, 23300, 23200, 23000)] + \
           [("undecimated union", t) for t in (23500, 23400)] + [("fresh 80k from the union", t) for t in (23500, 23400)]
    full80 = None
    for src, t in plan:
        if src == "collision_full 60k":
            m, path = B.clean_decimate(full, t, raw_iso=False)
        elif src == "undecimated union":
            if iso is None:
                iso = pv.read(os.path.join(ISO, name + ".vtp"))
            m, path = B.clean_decimate(iso, t, raw_iso=True)
        else:
            if iso is None:
                iso = pv.read(os.path.join(ISO, name + ".vtp"))
            if full80 is None:
                full80, p80 = B.clean_decimate(iso, 80000, raw_iso=True)
                if full80 is None:
                    tried.append({"source": src, "why": "no clean 80k"})
                    break
            m, path = B.clean_decimate(full80, t, raw_iso=False)
            path = None if path is None else "80k (%s) -> %s" % (p80, path)
        if m is None:
            tried.append({"source": src, "target": t, "why": "no clean decimation"})
            print(tried[-1], flush=True)
            continue
        ok, rr, out_n, tot_n = accept(m, src, t, path)
        if ok:
            found = (m, src, t, path, rr, out_n, tot_n)
            break
    if found is None:
        rep["obj_fix"] = {"result": "none accepted", "tried": tried}
        json.dump(rep, open(rp, "w"), indent=1)
        print("OBJFIX %s: NONE accepted" % name)
        return 1
    m, src, t, path, rr, out_n, tot_n = found
    pv.save_meshio(os.path.join(root, B.MESH_NAME), m)
    old = {"obj_tris_budget": rep["obj_tris_budget"], "obj_tris": rep["obj"]["tris"],
           "decimation_obj": rep["decimation"].get("obj")}
    rep["obj"] = dict(mesh_stats(m), genus=genus(m))
    rep["obj_tris_budget"] = BUDGET
    rep["routes"] = rr
    rep["centerline_pts_outside"], rep["centerline_pts_checked"] = out_n, tot_n
    rep["decimation"] = dict(rep["decimation"], obj="%s, from %s at target %d" % (path, src, t))
    rep["pass"] = bool(B.is_clean(m) and out_n == 0 and all(v.get("navigable") for v in rr.values()))
    rep["obj_fix"] = {"why": "v4's obj_ladder found no clean decimation of the 60k mesh at 23500 and stepped up to %d"
                             % old["obj_tris_budget"], "before": old, "accepted": {"source": src, "target": t,
                                                                                   "path": path, "tris": int(m.n_cells)},
                      "tried": tried, "script": "v5_va/bake_full/obj_fix.py", "seconds": round(time.time() - t0, 1)}
    tmp = rp + ".tmp"
    json.dump(rep, open(tmp, "w"), indent=1)
    os.replace(tmp, rp)
    pp = os.path.join(root, "provenance.json")
    prov = json.load(open(pp, encoding="utf-8"))
    bv = prov.setdefault("bake_v5", {})
    bv.update(obj_tris=rep["obj"]["tris"], obj_tris_budget=BUDGET, obj_fix=rep["obj_fix"]["accepted"])
    tmp = pp + ".tmp"
    json.dump(prov, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, pp)
    mk = os.path.join(BF, "_done", name + ".measure.json")
    if os.path.exists(mk):
        os.replace(mk, os.path.join(keep, name + ".measure.json.before"))
    print("OBJFIX %s: accepted %s target %d (%s), %d tris, %.0f s" % (name, src, t, path, m.n_cells, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
