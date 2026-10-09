#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\bake\bake_v6_1p2_dev.py (= /sp/v6/bake/bake_v6_1p2_dev.py in the build containers)
"""bake_v6.py -- the versioned v6 bake (stage 2; s2_harden_wall_bake, 2026-10-04).

= bake_v5b (bake_meshes_v4 unchanged + the user handle rule D1-D8; /sp/v5_va/bake_v5b.py, imported UNCHANGED)
  + the v6 VA-side real sections + the pilot's bake fixes, folded into ONE versioned module (the pilot ran them as
  bake_v6p 1.2 + bake_sections_v6 + c4_apply.py + obj_fix_va.py). Stated diffs against bake_v5b:

 E1  SECTIONS: v5's sections (bake_v5.sections_for_v5: v4's + vaC_R / vaC_L) + every VA-side real section the VA tool
     records in provenance xform with a "branch" and "lo_mm" / "hi_mm" (sc* keys: scC_R, scD_R, scC_L, scD_L, scJ_L,
     and any new sc* key): its real surface (load_surface with the xform) over [lo, hi] of its branch, siblings from
     the record; "real_surface": false -> tube only. (= v6_va's bake_sections_v6, generalised from a fixed key list.)
 E2  BRANCH LOOKUP: bake_meshes_v4.branch(br, "(18)") falls back to the ".mrk" / ".mrk.json" name (the loader names a
     branch "Centerline curve (18).mrk"); without it every sc* section on (17) / (18) / (19) raised KeyError.
 E3  VA POOL: the sc* sections (branches (17), (18), (19)) and the CHANGED carotid sections are searched sections
     (VA margin ladder, localisation / bisection, prune / shrink); unchanged carotid sections stay pinned to v4 (D1).
 E3b MARGIN BY BRANCH: the VA margin ladder (0.5 -> 1.0) applies to the VA-branch sections only; a changed carotid
     section searched with the VAs keeps v4's carotid margin (pilot k_011: a VA handle at margin 0.5 moved its siphons
     to margin 1.0 and the exact-ownership fork fell to 0.933 of v4).
 E4  CHANGED CAROTID SECTIONS (per anatomy, against v4): v4's pinned state rewritten in a per-anatomy pin report
     (same surface: v4's suppressed stretches shifted with the section start, v4's tube decision kept; different
     surface: dropped). Unchanged sections: v4's state as in bake_v5b.
 E5  RECORDS: mesh_v6.json (bake_v5b's schema + v6 fields); provenance "bake_v6"; the undecimated union to ISO_DIR.
 E6  BLEND MORPH: a re-posed piece whose first blend_mm was replaced by a cubic Hermite (compose_v6) gets its real
     surface displaced by the blend's own displacement D(i) = P_blended(i) - P_rigid(i) (nearest rigid point; 0 at the
     seam and from blend_mm on). P_rigid from v5's piece back-mapped (the carotid tool's own back-map), checked
     <= 0.01 mm beyond the blend; else the section stays rigid (recorded), as in the pilot. (A fallback from the
     piece's own centerline_src exists but is OFF: tested in 1.0.1, the written pieces sit 0.05-0.22 mm off their raw
     source polyline and the fallback morph came within 0.70 mm of the exact back-map only; env BAKE_V6_E6_SRC=1.)
 E7  VA-AWARE OBJ RE-CUT (after the bake): the VA lumen on the centreline polyline (wall/va_lumen_poly.py, 0.05 mm,
     3 mm caps out) must be >= 0.65 mm on BOTH VAs of the OBJ; if not, the OBJ is re-cut (bake_meshes_v4's
     clean_decimate + its own acceptance: clean genus-0 two-manifold, 0 self-intersections, both routes navigable,
     every centreline point inside) from collision_full at 23500, 23450, 23400, 23300, 23200, 23000, then from the union
     at 23500, 23400, the first passing candidate wins; none -> the OBJ stays and the anatomy is flagged
     (va_lumen_obj_fail). The union and the 60k mesh are never changed.
 C4  topcow_mr_021 LEFT-VA PINCH (pre-bake; V6_ITEMS c4, v6_wall decision): on the VA whose TopBrain top is
     topcow_mr_021_lVA.vtp (8 anatomies; c4_tube_override.json), the centreline radius over TopBrain-native arclength
     [33.5, 53.0] = max(own, linear 1.84 -> 1.64 mm), gated by clearance >= 0.6 mm of the changed points against every
     other branch; written into Centrelines_comb before the bake (backup kept in BAKE_DIR/_c4/<name>/). If the bake
     then fails with it, the original radii are restored and the anatomy is re-baked once without it (v4-heritage
     pinch kept, recorded). Idempotent: points already at the override change nothing. --no-c4 skips it.
 D20 (1.1, user decisions D19 / D20, 2026-10-04): RE-POSED AND X-MIRRORED FORKS.
     * A mirrored piece is xform.<key>.mirror = [-1, 1, 1] on the piece's own surface file (the carotid tool's
       convention, = the TopBrain siphons' since v3): sdf_union.load_surface applies the mirror before the rigid map and
       re-derives the normals on the closed surface (auto-orient + consistency = the winding flip), so the implicit
       distance has the right sign. A pre-mirrored surface FILE (any winding) is also accepted: the same orientation
       pass fixes its winding. 1.1 records per section the mirror determinant (chirality) and the orientation check
       (signed volume of the loaded surface > 0) in mesh_v6.json / provenance bake_v6.chirality.
     * E4: a mirror-only change of a carotid section (same surface file, mirror flipped) counts as the SAME surface
       (a reflection keeps every arclength window of the piece): v4's suppressed stretches shift with the section
       start and v4's tube decision is kept, exactly as for a re-posed unmirrored piece; each window end is also mapped
       through the piece's own frame (v4 slice -> source -> v6 slice) and the mapped window is used where it differs
       from the plain shift by > MAP_TOL_MM = 1 mm (a host-cut shift / trimmed piece). Changed sections (re-posed /
       mirrored right or left forks, re-posed siphons) are searched with the VA pool at v4's carotid margin (E3/E3b),
       i.e. the RCCA sections are no longer pinned to v4's state when the fork moved.
     * E6: forks too (lower_R on RCCA, lower_L on LCCA: the fork-side blend at seam 1). P_rigid from (1) v5's piece
       back-mapped (only when v5's piece is the same surface with the SAME mirror and has no blend of its own; checked
       <= 0.01 mm beyond the blend), else (2) the composer's own record xform.<key>.blend_disp = [s, dx, dy, dz] rows
       (blended - rigid over the piece's first points, s = rigid-piece arclength): aligned on the written branch by
       arclength (|arclen(P_written - D) - s| <= 1e-3 mm), then the same displacement morph. Neither -> rigid, recorded.
     The route invariant for a moved right fork (each real section's own v4 wall detail on the moved surface) is
     measured by checks_v6 1.1 (part route, per section).

 L1  (1.2, s3_fix 2026-10-08; final verifier's must-fix 1, USER rule D27): LATTICE SHIFT, opt-in per anatomy
     (--lattice-shift sx,sy,sz mm; default 0 = 1.1 bit for bit). Diagnosis (s3_fix diag_frag / diag_ray): on 8
     re-posed right-route pieces the scanned lumen carries a thin sliver (~0.1-0.5 mm thick, 1-3 mm off the main lumen,
     joined to it by a neck thinner than the 0.45 mm lattice); at the v6 pose the lattice resolves it as islands (or the
     handle window cuts its neck) and keep_largest drops them -- real scanned wall v4's / v5's bakes showed. The lattice
     origin of the field AND of every section capsule is moved by the same sub-voxel shift (sdf_mesher._grid: lo =
     floor((lo - shift) / h) h - PAD h + shift), so the same surfaces, centrelines and ownership are sampled on a shifted
     lattice; nothing else changes. Recorded in provenance bake_v6.lattice_shift_mm and the done marker; flagged
     bake_lattice_shift_D27 by the caller.
 L2  (1.2, opt-in --bridge-route-islands): ROUTE-ISLAND BRIDGE. Where the scanned sliver is SEPARATE from the main lumen in
     the scan itself (k_011_left mr_012_L fork: 0.62 mm gap at 0.15 mm resolution; v4 / v5 showed it only because their
     RECA TopBrain tip tube ran through the gap, D13), no lattice reaches it: on the ladder's final state, each island of
     lower_R / siphon_R's own scanned lumen that keep_largest would drop is joined to the main body by ONE short tube
     (radius 0.5 mm, along the shortest island -> main segment + 0.3 mm each end); kept only while the union stays genus 0
     (route_island_hook). Recorded in mesh_v6 ladder_v5b.route_island_bridge and provenance bake_v6.route_island_bridge;
     flagged bake_route_island_bridge_D27 by the caller.

    python3 bake_v6.py <name> [--tree DIR] [--bake DIR] [--iso DIR] [--pin DIR] [--no-c4] [--no-e7] [--force]
                       [--lattice-shift sx,sy,sz] [--bridge-route-islands]
        (neve-build-meshlab; /repo read-only). Done marker BAKE_DIR/_done/<name>.bake.json (written last), failure
        BAKE_DIR/_done/<name>.bake_failed.json; a rerun without the marker removes partial outputs first.
Library: setup(), bake_anatomy_v6(), apply_c4(), obj_recut_va(), VERSION.
"""
import argparse
import hashlib
import inspect as _inspect
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/sp/v5_va")
import bake_v5  # noqa: E402  (installs sections_for_v5: v4's sections + vaC_R / vaC_L)
import bake_v5b as V  # noqa: E402  (installs bake_one_v5b / bake_anatomy_v5b into bake_meshes_v4)

B = V.B
from sdf_union import load_surface  # noqa: E402
from sdf_union_v4 import Section  # noqa: E402

VERSION = "bake_v6 1.2 (2026-10-08; s3_fix: 1.2 = L1 opt-in lattice shift + L2 opt-in route-island bridge (defaults off = 1.1); 1.1 2026-10-04; s2_harden_wall_bake: bake_v5b + E1-E7 + C4 + D20; = pilot bake_v6p 1.2 + bake_sections_v6 + obj_fix_va + c4_apply, generalised; 1.0.1: the E6 source-centreline fallback tested and switched off by default; 1.1: D20 mirrored / re-posed forks -- E4 mirror-only = same surface, E6 forks + mirror-aware back-map + the composer's blend_disp, chirality record)"

# ------------------------------------------------------------------ L1 (1.2): lattice shift
import sdf_mesher as _SM  # noqa: E402

_GRID_V11 = _SM._grid
LATTICE_SHIFT = np.zeros(3)


def _grid_shifted(branches, spacing):
    """sdf_mesher._grid with the lattice origin moved by LATTICE_SHIFT (mod spacing); identical to it at 0."""
    if not LATTICE_SHIFT.any():
        return _GRID_V11(branches, spacing)
    lo = np.min([np.min(np.asarray(b.coordinates) - np.asarray(b.radii)[:, None], 0) for b in branches], 0)
    hi = np.max([np.max(np.asarray(b.coordinates) + np.asarray(b.radii)[:, None], 0) for b in branches], 0)
    lo = np.floor((lo - LATTICE_SHIFT) / spacing) * spacing - _SM.PAD_VOX * spacing + LATTICE_SHIFT
    shape = np.ceil((hi - lo) / spacing).astype(int) + 2 * _SM.PAD_VOX
    return lo, shape


_SM._grid = _grid_shifted

# ------------------------------------------------------------------ L2 (1.2): route-island bridge
ROUTE_BRIDGE = {"on": False, "radius_mm": 0.5, "extend_mm": 0.3, "min_cells": 30, "on_surface_frac": 0.5,
                "max_tube_dist_mm": 6.0, "sections": ("lower_R", "siphon_R")}


def _seg_box_values(field, p, q, rb, ext):
    """tube values rb - dist(x, segment p'q') on the voxels of the segment's box (p', q' = p, q extended by ext)."""
    u = q - p
    L = float(np.linalg.norm(u))
    u = u / L if L > 1e-9 else np.array([1.0, 0.0, 0.0])
    a, b = p - ext * u, q + ext * u
    sp = field.spacing
    lo = np.floor((np.minimum(a, b) - rb - 2 * sp - field.offset) / sp).astype(int)
    hi = np.ceil((np.maximum(a, b) + rb + 2 * sp - field.offset) / sp).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(field.values.shape))
    ix = [np.arange(lo[k], hi[k]) for k in range(3)]
    G = np.stack(np.meshgrid(*ix, indexing="ij"), -1)
    X = G * sp + field.offset
    ab = b - a
    t = np.clip(((X - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    d = np.linalg.norm(X - (a + t[..., None] * ab), axis=-1)
    return (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2])), (rb - d).astype(np.float32)


def route_island_hook(cache, full, secs_final, va_margin, v4_margin, suppress, far_clamp, br, root):
    """L2 (1.2, opt-in --bridge-route-islands; s3_fix, USER rule D27): islands of a RIGHT-ROUTE section's own scanned
    lumen that the union holds but keep_largest would drop (components other than the largest whose points lie on
    lower_R's / siphon_R's loaded surface (>= on_surface_frac within 0.3 mm) and near the RCCA (median tube distance <=
    max_tube_dist_mm)) are joined to the main body by ONE short bridge each: a tube of radius_mm along the shortest
    island -> main segment (extended extend_mm into both), max-unioned into the field. One bridge per island cannot make a
    handle by itself; the bridged union must be genus 0 with no edge defect or the bridges are tried one at a time
    (greedy, in island-size order) and only the ones that keep genus 0 stay. Runs on the ladder's final state (the cache
    rebuilt to it, checked against the ladder's own union by cell count). Returns (full, record)."""
    if not ROUTE_BRIDGE["on"]:
        return full, None
    from sdf_mesher import iso_surface
    from sdf_union_v4 import genus
    import pyvista as pv
    t0 = time.time()
    P_ = ROUTE_BRIDGE
    rec = {"rule": "bake_v6 1.2 L2 route-island bridge (s3_fix, D27)", "params": {k: (list(v) if isinstance(v, tuple) else v)
                                                                                  for k, v in P_.items()},
           "islands": [], "bridged": []}
    seq = [(s.name, (va_margin if s.branch in VA_BRANCHES else v4_margin), tuple(tuple(x) for x in suppress.get(s.name, [])))
           for s in secs_final]
    cache.build(seq)
    f = cache.field

    def raw_iso():
        far = None
        if far_clamp:
            far = f.values < B.FAR_BELOW
            f.values[far] = -2.0 * cache.spacing
        try:
            return iso_surface(f)
        finally:
            if far is not None:
                f.values[far] = V.FAR

    raw = raw_iso()
    k0, _, _ = B.keep_largest(raw)
    rec["final_state_rebuilt_cells"], rec["ladder_full_cells"] = int(k0.n_cells), int(full.n_cells)
    if k0.n_cells != full.n_cells:
        rec["result"] = "the ladder's final union was not reproduced from the cache: no bridge"
        return full, rec
    conn = raw.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"])
    sizes = np.bincount(rid)
    keep = int(np.argmax(sizes))
    Vm = np.asarray(conn.extract_cells(np.nonzero(rid == keep)[0]).extract_surface().points, float)
    from scipy.spatial import cKDTree
    tm = cKDTree(Vm)
    rc = B.branch(br, "RCCA")
    P, R = np.asarray(rc.coordinates, float), np.asarray(rc.radii, float)
    S = B.arclen(P)
    tP = cKDTree(P)
    surfs = {s.name: s.surface for s in secs_final if s.name in P_["sections"]}
    isl = []
    for k in [int(x) for x in np.argsort(-sizes) if x != keep]:
        if sizes[k] < P_["min_cells"]:
            break
        pc = conn.extract_cells(np.nonzero(rid == k)[0]).extract_surface()
        X = np.asarray(pc.points, float)
        d, i = tP.query(X)
        td = d - R[i]
        if float(np.median(td)) > P_["max_tube_dist_mm"]:
            continue
        onf = {}
        for nm, sf in surfs.items():
            ds = np.abs(np.asarray(pv.PolyData(X).compute_implicit_distance(sf)["implicit_distance"]))
            onf[nm] = float((ds <= 0.3).mean())
        if not onf:
            continue
        best = max(onf, key=onf.get)
        if onf[best] < P_["on_surface_frac"]:
            continue
        dd, jj = tm.query(X)
        a = int(np.argmin(dd))
        isl.append({"cells": int(sizes[k]), "volume_mm3": round(abs(float(pc.volume)), 2), "section": best,
                    "on_surface_frac": round(onf[best], 3), "rcca_s": [round(float(S[i].min()), 2), round(float(S[i].max()), 2)],
                    "tube_dist_median_mm": round(float(np.median(td)), 2), "gap_mm": round(float(dd[a]), 3),
                    "p": X[a].tolist(), "q": Vm[jj[a]].tolist()})
    rec["islands"] = [dict(x, p=np.round(x["p"], 3).tolist(), q=np.round(x["q"], 3).tolist()) for x in isl]
    if not isl:
        rec["result"] = "no route island"
        return full, rec

    def apply(x):
        sl, val = _seg_box_values(f, np.asarray(x["p"]), np.asarray(x["q"]), P_["radius_mm"], P_["extend_mm"])
        old = (f.values[sl].copy(), f.region[sl].copy())
        f.values[sl] = np.maximum(f.values[sl], val)
        f.region[sl] = True
        return sl, old

    def revert(sl, old):
        f.values[sl], f.region[sl] = old

    def judge():
        m, st, dr = B.keep_largest(raw_iso())
        g = genus(m)
        gmax = max(c["genus"] for c in g) if g else -1
        nme = B.edge_defects(m)
        return m, gmax, nme, dr

    undo = [apply(x) for x in isl]
    m, gmax, nme, dr = judge()
    tries = [{"bridges": len(isl), "genus": gmax, "nonmanifold_edges": int(nme), "dropped_volume_mm3": round(dr["volume_mm3"], 2)}]
    if gmax == 0 and not nme:
        used = list(range(len(isl)))
    else:
        for sl, old in reversed(undo):
            revert(sl, old)
        used, m = [], None
        for j, x in enumerate(isl):
            sl, old = apply(x)
            m2, g2, n2, dr2 = judge()
            tries.append({"bridge": j, "genus": g2, "nonmanifold_edges": int(n2), "dropped_volume_mm3": round(dr2["volume_mm3"], 2)})
            if g2 == 0 and not n2:
                used.append(j)
                m, dr = m2, dr2
            else:
                revert(sl, old)
    rec["tries"] = tries
    rec["seconds"] = round(time.time() - t0, 1)
    if not used or m is None:
        rec["result"] = "no bridge keeps the union genus 0: union unchanged"
        return full, rec
    rec["bridged"] = [{"island": j, "section": isl[j]["section"], "rcca_s": isl[j]["rcca_s"], "volume_mm3": isl[j]["volume_mm3"],
                       "length_mm": round(float(np.linalg.norm(np.subtract(isl[j]["q"], isl[j]["p"]))) + 2 * P_["extend_mm"], 3),
                       "radius_mm": P_["radius_mm"]} for j in used]
    rec["result"] = "bridged %d of %d route islands" % (len(used), len(isl))
    rec["union_cells_before_after"] = [int(full.n_cells), int(m.n_cells)]
    rec["dropped_volume_after_mm3"] = round(dr["volume_mm3"], 2)
    return m, rec


V.ROUTE_ISLAND_HOOK = route_island_hook
VA_BRANCHES = ("RVA", "LVA", "(17)", "(18)", "(19)")
CAROTID = V.CAROTID
STATE = {"changed": set()}
MORPH = {"rec": {}}
A5_ROOT = os.environ.get("BAKE_V6_A5", "/repo/carotid_data/anatomies_v5")
TB_ROOT = "/repo"
C4_SPEC = "/repo/carotid_data/v6_work/wall/c4/c4_tube_override.json"
LUMEN_NEED, LUMEN_STEP = 0.65, 0.05


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ E2
_BRANCH = B.branch


def branch_v6(br, tag):
    try:
        return _BRANCH(br, tag)
    except KeyError:
        for b in br:
            if str(b.name).endswith(tag + ".mrk") or str(b.name).endswith(tag + ".mrk.json"):
                return b
        raise


B.branch = branch_v6

# ------------------------------------------------------------------ E1
_V5_SECTIONS = bake_v5.sections_for_v5


SC_ORDER = ("scC_R", "scD_R", "scC_L", "scD_L", "scJ_L")   # the pilot's order (section order is part of the bake)


def sc_keys(x):
    ks = [k for k, v in x.items() if k.startswith("sc") and isinstance(v, dict) and "branch" in v and "lo_mm" in v]
    return [k for k in SC_ORDER if k in ks] + sorted(k for k in ks if k not in SC_ORDER)


def sections_for_v6(root, br):
    secs, spans = _V5_SECTIONS(root, br)
    x = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))["xform"]
    for key in sc_keys(x):
        xf = x[key]
        if not xf.get("real_surface") or not xf.get("surface"):
            spans[key] = "tube only (%s)" % xf.get("why_tube", "no junction wall")
            continue
        bname = xf["branch"]
        b = B.branch(br, bname)
        p, r, L = B.slice_b(b, float(xf["lo_mm"]), float(xf["hi_mm"]))
        if len(p) < 4:
            continue
        surf = load_surface(xf["surface"], xf["mirror"], xf["R"], xf["origin"], xf["anchor"])
        secs.append(Section(key, surf, p, r, True, True, bname, float(xf["lo_mm"]), tuple(xf.get("siblings") or ())))
        spans[key] = (bname, float(xf["lo_mm"]), float(min(xf["hi_mm"], L)))
    return secs, spans


# ------------------------------------------------------------------ E3
def is_va_v6(sec):
    return sec.branch in VA_BRANCHES or sec.name in STATE["changed"]


V.is_va = is_va_v6

# ------------------------------------------------------------------ E3b (source patch of bake_v5b.bake_one_v5b, asserted)
_src = _inspect.getsource(V.bake_one_v5b)
_a = "        seq = [(s.name, (va_margin if is_va(s) else v4_margin), tuple(tuple(x) for x in sup.get(s.name, [])))\n"
assert _src.count(_a) == 1, "bake_v5b.bake_one_v5b changed: E3b patch point not found"
_src = _src.replace(_a, "        seq = [(s.name, (va_margin if s.branch in VA_BRANCHES_V6 else v4_margin), tuple(tuple(x) for x in sup.get(s.name, [])))\n")
V.VA_BRANCHES_V6 = VA_BRANCHES
# 1.2 L2: the route-island bridge hook, called on the ladder's FINAL state just before the cache is dropped (a no-op that
# leaves the report untouched unless --bridge-route-islands)
_b = "    del cache\n"
assert _src.count(_b) == 1, "bake_v5b.bake_one_v5b changed: L2 patch point not found"
_src = _src.replace(_b, "    if full is not None and mode == \"owned\" and ROUTE_ISLAND_HOOK is not None:\n"
                        "        full, _l2 = ROUTE_ISLAND_HOOK(cache, full, [s for s in secs if s.name not in tube_only], va_margin,\n"
                        "                                      v4_margin, suppress, far_clamp, br, root)\n"
                        "        if _l2 is not None:\n"
                        "            lad[\"route_island_bridge\"] = _l2\n"
                        "    del cache\n")
if not hasattr(V, "ROUTE_ISLAND_HOOK"):
    V.ROUTE_ISLAND_HOOK = None
exec(compile(_src, "<bake_v6 E3b bake_one_v5b>", "exec"), V.__dict__)
B.bake_one = V.bake_one_v5b

# ------------------------------------------------------------------ E6
SRC_TOL_MM = 0.10
# The source-centreline fallback is OFF: e6_test (v6_work/bake_v6/reg/logs/e6_test3.log) found the written pieces 0.05-0.22 mm off
# their raw TopBrain source polyline and the fallback morph up to 0.70 mm from the exact v5 back-map (w_034 siphon_L), so a
# piece without a v5 counterpart stays rigid (recorded), as in the pilot. Switch: env BAKE_V6_E6_SRC=1.
E6_SRC_FALLBACK = os.environ.get("BAKE_V6_E6_SRC", "0") == "1"
MORPH_KEYS = {"siphon_R": ("siphon", "RCCA"), "siphon_L": ("siphon_L", "LCCA"), "ecaext_R": ("ecaext_R", "RECA"),
              "ecaext_L": ("ecaext_L", "LECA"),
              # 1.1 (D19 / D20): the forks' seam-1 blend (fork side, start of the RCCA / LCCA slice)
              "lower_R": ("lower", "RCCA"), "lower_L": ("lower_L", "LCCA")}
DISP_TOL_MM = 1.0e-3
FORK_SECTIONS = ("lower_R", "lower_L")


def _fork_len(root, sname):
    """the fork piece's slice length on its branch (bake_meshes_v4.sections_for: cca + min(ica, ica_real))."""
    d = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    if sname == "lower_R":
        cca, ica, xf = float(d["cca_mm"]), float(d["ica_mm"]), d["xform"]["lower"]
    else:
        cca, ica, xf = float(d["left"]["cca_mm"]), float(d["left"]["ica_mm"]), d["xform"]["lower_L"]
    return cca + min(ica, float(xf.get("ica_real_mm", ica)))


def _mirror_of(xf):
    return [float(v) for v in (xf.get("mirror") or [1.0, 1.0, 1.0])]


def _rigid_v5(root, name, key, bname, x6):
    p5f = os.path.join(A5_ROOT, name, "provenance.json")
    if not os.path.exists(p5f):
        return None, "no v5 anatomy"
    x5 = json.load(open(p5f, encoding="utf-8"))["xform"].get(key)
    if not x5:
        return None, "no v5 piece to back-map (new piece)"
    if os.path.basename(str(x5.get("surface"))) != os.path.basename(str(x6.get("surface"))):
        return None, "v5 piece is a different surface"
    if _mirror_of(x5) != _mirror_of(x6):
        return None, "v5 piece mirrored differently (D20)"
    if float(x5.get("blend_mm") or 0.0) > 0.0:
        return None, "v5 piece has a blend of its own"
    from eve_bench.dualdevicenav import load_branches
    b5 = B.branch(load_branches(os.path.join(A5_ROOT, name, "Centrelines_comb")), bname)
    P5 = np.asarray(b5.coordinates, float)
    i5 = int(np.argmin(np.linalg.norm(P5 - np.asarray(x5["anchor"], float), axis=1)))
    Q = (P5[i5:] - np.asarray(x5["anchor"], float)) @ np.asarray(x5["R"], float) + np.asarray(x5["origin"], float)
    k0 = int(np.argmin(np.linalg.norm(Q - np.asarray(x6["origin"], float), axis=1)))
    Q = Q[k0:]
    return (Q - np.asarray(x6["origin"], float)) @ np.asarray(x6["R"], float).T + np.asarray(x6["anchor"], float), "v5 back-map"


def _rigid_src(x6, Pb, h):
    """P_rigid from the piece's own source centreline: the written points beyond the blend lie on the transformed
    source polyline at arclengths sigma(i) = sigma0 + ds * i (one resampling); fit it there, evaluate in the blend."""
    src = x6.get("centerline_src")
    if not src:
        return None, "no centerline_src"
    f = src if os.path.isabs(src) else os.path.join(TB_ROOT, src)
    if not os.path.exists(f):
        return None, "centerline_src missing"
    S = np.asarray(json.load(open(f))["points"], float) * np.asarray(x6.get("mirror", [1, 1, 1]), float)
    T = (S - np.asarray(x6["origin"], float)) @ np.asarray(x6["R"], float).T + np.asarray(x6["anchor"], float)
    sT = B.arclen(T)
    sb = B.arclen(Pb)
    beyond = np.nonzero(sb > h + 1.0)[0]
    if len(beyond) < 6:
        return None, "piece too short beyond the blend"
    # project the points beyond the blend on T
    sig = np.zeros(len(beyond))
    for n_, i in enumerate(beyond):
        X = Pb[i]
        a, b = T[:-1], T[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((X - a) * ab).sum(1) / L2, 0, 1)
        d = np.linalg.norm(a + t[:, None] * ab - X, axis=1)
        j = int(np.argmin(d))
        if d[j] > SRC_TOL_MM:
            return None, "written piece not on its source centreline beyond the blend (%.3f mm)" % d[j]
        sig[n_] = sT[j] + t[j] * np.sqrt(L2[j])
    A = np.c_[np.ones(len(beyond)), beyond.astype(float)]
    coef, *_ = np.linalg.lstsq(A, sig, rcond=None)
    if np.abs(A @ coef - sig).max() > SRC_TOL_MM:
        return None, "no single resampling fits beyond the blend"
    sall = coef[0] + coef[1] * np.arange(len(Pb))
    return np.stack([np.interp(sall, sT, T[:, k]) for k in range(3)], 1), "source centreline fit"


def _rigid_disp(x6, P6, i6):
    """1.1: P_rigid from the composer's own record xform.<key>.blend_disp ([s, dx, dy, dz] = blended - rigid over the
    piece's first points, s = arclength of the RIGID piece). Rows are aligned on the written branch by trying start
    offsets around the anchor index: the right one makes arclen(P_written - D) equal the recorded s (<= DISP_TOL_MM).
    Returns (Pb, Pr, why) with Pb = the written piece from the aligned start, Pr = Pb - D (D = 0 past the rows)."""
    rows = x6.get("blend_disp") or []
    if not rows:
        return None, None, "no blend_disp in provenance"
    Rw = np.asarray(rows, float)
    s_rec, D = Rw[:, 0] - Rw[0, 0], Rw[:, 1:4]
    best = None
    for o in (i6, i6 + 1, i6 - 1, i6 + 2):
        if o < 0 or o + len(Rw) > len(P6):
            continue
        Pr_ = P6[o:o + len(Rw)] - D
        err = float(np.abs(B.arclen(Pr_) - s_rec).max())
        if best is None or err < best[0]:
            best = (err, o)
    if best is None:
        return None, None, "blend_disp longer than the written piece"
    if best[0] > DISP_TOL_MM:
        return None, None, "blend_disp does not align with the written piece (arclength mismatch %.4f mm)" % best[0]
    o = best[1]
    Pb = P6[o:].copy()
    Pr = Pb.copy()
    Pr[:len(Rw)] -= D
    return Pb, Pr, "composer blend_disp (aligned at offset %+d, arclength err %.1e mm)" % (o - i6, best[0])


def _morph_section(root, sec, br):
    key, bname = MORPH_KEYS[sec.name]
    name = os.path.basename(os.path.normpath(root))
    x6 = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))["xform"].get(key) or {}
    h = float(x6.get("blend_mm") or 0)
    if h <= 0:
        return {"morphed": False, "why": "no blend"}
    b6 = B.branch(br, bname)
    P6 = np.asarray(b6.coordinates, float)
    i6 = int(np.argmin(np.linalg.norm(P6 - np.asarray(x6["anchor"], float), axis=1)))
    Pb0 = P6[i6:]
    tried = []
    for how in (("v5", "disp", "src") if E6_SRC_FALLBACK else ("v5", "disp")):
        Pb = Pb0
        if how == "v5":
            Pr, why = _rigid_v5(root, name, key, bname, x6)
        elif how == "disp":
            Pb, Pr, why = _rigid_disp(x6, P6, i6)
            if Pb is None:
                Pb = Pb0
        else:
            Pr, why = _rigid_src(x6, Pb, h)
        if Pr is None:
            tried.append(why)
            continue
        n = min(len(Pb), len(Pr))
        if sec.name in FORK_SECTIONS:
            # a fork piece is followed on its branch by other pieces (ICA extension, siphon): compare / morph over the
            # fork piece only (its slice length from provenance, as sections_for slices it)
            n = min(n, int(np.searchsorted(B.arclen(Pb[:n]), _fork_len(root, sec.name) - 0.5)))
        if n < 8:
            tried.append("piece too short")
            continue
        s = B.arclen(Pr[:n])
        beyond = s > h + 1.0
        err = float(np.linalg.norm(Pb[:n][beyond] - Pr[:n][beyond], axis=1).max()) if beyond.any() else np.inf
        if not err <= (SRC_TOL_MM if how == "src" else 0.01):
            tried.append("%s: rigid rebuild does not match the written piece beyond the blend (%.3f mm)" % (why, err))
            continue
        D = Pb[:n] - Pr[:n]
        D[beyond] = 0.0
        m = s <= h + 4.0
        sd = np.arange(0.0, float(s[m][-1]) + 1e-9, 0.1)
        Pd = np.stack([np.interp(sd, s[m], Pr[:n][m][:, k]) for k in range(3)], 1)
        Dd = np.stack([np.interp(sd, s[m], D[m][:, k]) for k in range(3)], 1)
        Vx = np.asarray(sec.surface.points, float)
        from scipy.spatial import cKDTree
        d_all, _ = cKDTree(Pr[:n]).query(Vx)
        d_bl, j = cKDTree(Pd).query(Vx)
        mv = d_bl <= d_all + 1e-9
        V2 = Vx.copy()
        V2[mv] += Dd[j[mv]]
        import pyvista as pv
        m2 = pv.PolyData(V2, sec.surface.faces)
        m2 = m2.compute_normals(auto_orient_normals=True, consistent_normals=True, point_normals=True, cell_normals=True,
                                inplace=False)
        sec.surface = m2
        return {"morphed": True, "blend_mm": h, "rigid_from": why, "max_disp_mm": round(float(np.linalg.norm(D, axis=1).max()), 3),
                "vertices_moved": int(mv.sum()), "rigid_rebuild_err_mm": round(err, 5)}
    return {"morphed": False, "why": "; ".join(tried)}


def _signed_volume(m):
    """divergence-theorem volume of a triangulated surface with its faces as stored (> 0 = outward winding)."""
    V = np.asarray(m.points, float)
    F = np.asarray(m.faces).reshape(-1, 4)[:, 1:]
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def sections_for_morph(root, br):
    secs, spans = sections_for_v6(root, br)
    rec = {}
    for sec in secs:
        if sec.name in MORPH_KEYS:
            try:
                rec[sec.name] = _morph_section(root, sec, br)
            except Exception as e:  # noqa: BLE001
                rec[sec.name] = {"morphed": False, "why": "%s: %s" % (type(e).__name__, e)}
    MORPH["rec"] = rec
    # 1.1 (D20): chirality per section (mirror determinant of its xform) + the loaded surface's winding check
    x = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))["xform"]
    ch = {}
    for sec in secs:
        xf = x.get(XF_OF.get(sec.name, sec.name)) or {}
        try:
            sv = _signed_volume(sec.surface)
        except Exception as e:  # noqa: BLE001
            sv = "%s: %s" % (type(e).__name__, e)
        ch[sec.name] = {"mirror": _mirror_of(xf), "mirrored": bool(np.prod(_mirror_of(xf)) < 0),
                        "signed_volume_mm3": round(sv, 2) if isinstance(sv, float) else sv,
                        "outward": bool(isinstance(sv, float) and sv > 0)}
    MORPH["chirality"] = ch
    return secs, spans


B.sections_for = sections_for_morph

# ------------------------------------------------------------------ E5
V.REPORT_V5 = "mesh_v6.json"
B.REPORT_V5 = "mesh_v6.json"

# ------------------------------------------------------------------ E4
XF_OF = {"lower_R": "lower", "eca_R": "lower", "ecaext_R": "ecaext_R", "siphon_R": "siphon",
         "lower_L": "lower_L", "eca_L": "lower_L", "ecaext_L": "ecaext_L", "siphon_L": "siphon_L"}


def carotid_slices(root):
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    orig = B.load_surface
    B.load_surface = lambda *a, **k: None
    try:
        secs, spans = bake_v5._V4_SECTIONS_FOR(root, br)
    finally:
        B.load_surface = orig
    x = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))["xform"]
    out = {}
    for s in secs:
        if s.branch in CAROTID:
            out[s.name] = (s.branch, float(s.s0), s.points, s.radii, x.get(XF_OF.get(s.name, ""), {}))
    return out, spans


def _same_xf(a, b):
    for k in ("R", "origin", "anchor", "mirror"):
        va_, vb_ = np.asarray(a.get(k, [0.0]), float), np.asarray(b.get(k, [0.0]), float)
        if va_.shape != vb_.shape or not np.allclose(va_, vb_, atol=1e-9):
            return False
    return os.path.basename(str(a.get("surface"))) == os.path.basename(str(b.get("surface")))


MAP_TOL_MM = 1.0       # a plain re-pose with a blend changes the arclength by < ~0.5 mm (keep 1.0's shift there); a host-cut
                       # shift (trimmed piece) moves it by its own length


def _map_s(s4, b, a):
    """branch arclength s4 on v4's slice b -> the same SOURCE point -> branch arclength on v6's slice a.
    b / a = (branch, s0, points, radii, xform) from carotid_slices; world = (src * mirror - origin) R^T + anchor."""
    P4, s04, x4 = np.asarray(b[2], float), float(b[1]), b[4]
    P6, s06, x6 = np.asarray(a[2], float), float(a[1]), a[4]
    l4 = B.arclen(P4)
    q = np.array([np.interp(s4 - s04, l4, P4[:, k]) for k in range(3)])
    src = ((q - np.asarray(x4["anchor"], float)) @ np.asarray(x4["R"], float) + np.asarray(x4["origin"], float)) * \
        np.asarray(_mirror_of(x4))
    w6 = (src * np.asarray(_mirror_of(x6)) - np.asarray(x6["origin"], float)) @ np.asarray(x6["R"], float).T + \
        np.asarray(x6["anchor"], float)
    l6 = B.arclen(P6)
    best = (np.inf, 0.0)
    for i in range(len(P6) - 1):
        ab = P6[i + 1] - P6[i]
        t = float(np.clip((w6 - P6[i]) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
        d = float(np.linalg.norm(P6[i] + t * ab - w6))
        if d < best[0]:
            best = (d, l6[i] + t * (l6[i + 1] - l6[i]))
    return s06 + best[1]


def pin_report(name, root, v4_root, pin_dir):
    c6, sp6 = carotid_slices(root)
    c4, sp4 = carotid_slices(os.path.join(v4_root, name))
    rep4 = json.load(open(os.path.join(v4_root, name, B.REPORT), encoding="utf-8"))
    changed, rec = set(), {}
    sup = {k: [list(x) for x in v] for k, v in (rep4.get("suppressed_stretches_mm") or {}).items()}
    tube = list(rep4.get("sections_tube_only") or [])
    for nm in sorted(set(c6) | set(c4)):
        a, b = c6.get(nm), c4.get(nm)
        if a is None or b is None:
            rec[nm] = {"changed": True, "why": "absent in v6" if a is None else "new in v6"}
            if a is not None:
                changed.add(nm)
            sup.pop(nm, None)
            if nm in tube:
                tube.remove(nm)
            continue
        same_pts = a[2].shape == b[2].shape and np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
        same_x = _same_xf(a[4], b[4])
        if same_pts and same_x:
            rec[nm] = {"changed": False}
            continue
        changed.add(nm)
        # 1.1 (D20): a mirror-only difference is the SAME surface (a reflection keeps every arclength window of the
        # piece); 1.0 required the mirror to match too
        same_surf = os.path.basename(str(a[4].get("surface"))) == os.path.basename(str(b[4].get("surface")))
        mir6, mir4 = _mirror_of(a[4]), _mirror_of(b[4])
        r = {"changed": True, "points_same": bool(same_pts), "xform_same": bool(same_x), "same_surface": bool(same_surf),
             "s0_v6": a[1], "s0_v4": b[1], "mirror_v6": mir6, "mirror_v4": mir4,
             "mirrored_vs_v4": bool(np.prod(mir6) * np.prod(mir4) < 0)}
        if same_surf:
            sh = a[1] - b[1]
            if nm in sup:
                r["v4_stretches"] = sup[nm]
                shifted = [[lo + sh, hi + sh] for lo, hi in sup[nm]]
                # 1.1: map each window end through the piece's own frame (v4 slice -> source -> v6 slice); this is exact
                # for a trimmed (host-cut shift), re-posed or mirrored piece. Used only where it differs from the plain
                # shift by > MAP_TOL_MM (1 mm: a host-cut shift; a blend alone stays within it, so every piece the plain
                # shift already places keeps 1.0's windows bit for bit)
                try:
                    mapped = [sorted([_map_s(lo, b, a), _map_s(hi, b, a)]) for lo, hi in sup[nm]]
                    dev = max(max(abs(m[0] - s_[0]), abs(m[1] - s_[1])) for m, s_ in zip(mapped, shifted))
                except Exception as e:  # noqa: BLE001
                    mapped, dev = None, None
                    r["window_map_error"] = "%s: %s" % (type(e).__name__, e)
                r["window_map_dev_from_shift_mm"] = dev
                if mapped is not None and dev is not None and dev > MAP_TOL_MM:
                    sup[nm] = mapped
                    r["pinned_stretches_mapped_through_piece"] = mapped
                else:
                    sup[nm] = shifted
                    r["pinned_stretches_shifted_by_mm"] = sh
            r["v4_tube_only_kept"] = nm in tube
        else:
            if nm in sup:
                r["v4_stretches_dropped"] = sup.pop(nm)
            if nm in tube:
                tube.remove(nm)
                r["v4_tube_only_dropped"] = True
        rec[nm] = r
    pin = dict(rep4)
    pin["suppressed_stretches_mm"] = sup
    pin["sections_tube_only"] = tube
    pin["v6_pin"] = {"source": os.path.join(v4_root, name, B.REPORT), "sections": rec, "changed": sorted(changed)}
    os.makedirs(os.path.join(pin_dir, name), exist_ok=True)
    with open(os.path.join(pin_dir, name, B.REPORT), "w", encoding="utf-8") as f:
        json.dump(pin, f, indent=1)
    return changed, rec


def setup(name, root, v4_root, pin_dir):
    changed, rec = pin_report(name, root, v4_root, pin_dir)
    STATE["changed"] = set(changed)
    V.V4_DIR = pin_dir
    return changed, rec


def record_provenance(root, rep, extra):
    p = os.path.join(root, "provenance.json")
    with open(p, encoding="utf-8") as f:
        prov = json.load(f)
    lad = rep["ladder_v5b"]
    prov["bake_v6"] = {
        "script": VERSION, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "report": V.REPORT_V5, "pass": rep["pass"],
        "files": [B.MESH_NAME, B.FULL_NAME, V.REPORT_V5], "spacing_mm": rep["spacing_mm"],
        "obj_tris": rep["obj"]["tris"], "obj_tris_budget": rep["obj_tris_budget"], "full_tris": rep["full"]["tris"],
        "union": rep["union"], "rung": lad["rung"], "flags": lad["flags"], "carotid_margin_mm": rep["margin_mm"],
        "va_margin_mm": lad["va_margin_mm"], "far_clamp": rep["far_clamp"],
        "sections_tube_only": rep["sections_tube_only"], "suppressed_stretches_mm": rep["suppressed_stretches_mm"],
        "va_tubed": lad["va_tubed"], "right_route_detail": rep["right_route_detail"], "seconds": rep["seconds"]}
    prov["bake_v6"].update(extra)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    os.replace(tmp, p)


# ------------------------------------------------------------------ C4
def _arclength(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def c4_target(root):
    """(tag, key) of the VA whose TopBrain top is the c4 piece, from provenance; None if none."""
    spec = json.load(open(C4_SPEC))
    x = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8")).get("xform", {})
    piece = os.path.basename(spec["piece"])
    for tag, key in (("RVA", "va_R"), ("LVA", "va_L")):
        xf = x.get(key) or {}
        if os.path.basename(str(xf.get("surface"))) == piece:
            return tag, key, spec
    return None


def apply_c4(root, backup_dir=None, write=True):
    """C4: the tube override on the topcow_mr_021 left-VA TopBrain stretch (c4_apply.py's rule, same gate).
    Returns the record (also stored in provenance "c4_v6" when write)."""
    sys.path.insert(0, "/opt/eve_training/topbrain_tools")
    from graft_siphon import pair_gaps, read_curve, write_curve
    name = os.path.basename(os.path.normpath(root))
    t = c4_target(root)
    res = {"script": VERSION, "spec": C4_SPEC, "applied": False}
    if t is None:
        res["why_not"] = "no topcow_mr_021_lVA top in this anatomy"
        return res
    tag, key, spec = t
    listed = any(a == "%s:%s" % (name, tag) for a in spec["anatomies"])
    res["listed_in_spec"] = listed
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    xf = prov["xform"][key]
    fname = "Centerline curve - %s.mrk.json" % tag
    cdir = os.path.join(root, "Centrelines_comb")
    tmpl, p, r = read_curve(os.path.join(cdir, fname))
    src = json.load(open(spec["centreline"], encoding="utf-8"))
    pn = np.asarray(src["points"], float) * np.asarray(xf["mirror"], float)
    sn = _arclength(pn)
    R, origin, anchor = (np.asarray(xf[k], float) for k in ("R", "origin", "anchor"))
    s = _arclength(p)
    k0 = int(np.argmin(np.linalg.norm(p - anchor, axis=1)))
    Q = (p[k0:] - anchor) @ R + origin
    d = np.linalg.norm(Q[:, None, :] - pn[None, :, :], axis=2)
    j = np.argmin(d, axis=1)
    s_tb = sn[j]
    in_ = (s_tb >= spec["s_from_tb"]) & (s_tb <= spec["s_to_tb"])
    ramp = spec["radius_from"] + (spec["radius_to"] - spec["radius_from"]) * (s_tb - spec["s_from_tb"]) / (spec["s_to_tb"] - spec["s_from_tb"])
    r2 = r.copy()
    idx = np.nonzero(in_)[0] + k0
    r2[idx] = np.maximum(r[idx], ramp[in_])
    res.update(anatomy_branch=tag, points_changed=int((r2 != r).sum()), points_in_stretch=int(in_.sum()),
               map_dist_max_mm=float(d[np.arange(len(j)), j][in_].max()) if in_.any() else None,
               s_branch_mm=[float(s[idx].min()), float(s[idx].max())] if len(idx) else None,
               r_min_before=float(r[idx].min()) if len(idx) else None, r_min_after=float(r2[idx].min()) if len(idx) else None)
    others = []
    for f in sorted(os.listdir(cdir)):
        if f == fname or not f.endswith(".json"):
            continue
        _, q, rq = read_curve(os.path.join(cdir, f))
        if rq is not None:
            others.append((f, q, rq))
    ch = idx[r2[idx] != r[idx]] if len(idx) else idx
    if len(ch):
        gaps = {f: float(pair_gaps(p[ch], r2[ch], q, rq).min()) for f, q, rq in others}
        far = np.nonzero(np.abs(s[:, None] - s[ch][None, :]).min(axis=1) > 12.0)[0]
        gaps["self"] = float(pair_gaps(p[ch], r2[ch], p[far], r2[far]).min()) if len(far) else np.inf
        who = min(gaps, key=gaps.get)
        res.update(clearance_min_mm=gaps[who], clearance_who=who)
        ok = gaps[who] >= 0.6
    else:
        ok = True
    res["gate_ok"] = bool(ok)
    if ok and len(ch):
        if write:
            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(os.path.join(cdir, fname), os.path.join(backup_dir, fname))
                res["backup"] = os.path.join(backup_dir, fname)
            write_curve(tmpl, os.path.join(cdir, fname), p, r2)
        res["applied"] = True
    elif not len(ch):
        res["why_not"] = "nothing to change (radii already at or above the override)"
        res["already_at_override"] = bool(in_.any())
    else:
        res["why_not"] = "clearance below 0.6 mm -> the v4-heritage pinch is kept"
    if write:
        prov.setdefault("c4_v6", {}).update(res)
        tmp = os.path.join(root, "provenance.json.tmp")
        json.dump(prov, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, os.path.join(root, "provenance.json"))
    return res


def revert_c4(root, rec):
    b = rec.get("backup")
    if not b or not os.path.exists(b):
        return False
    shutil.copy2(b, os.path.join(root, "Centrelines_comb", os.path.basename(b)))
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    prov.setdefault("c4_v6", {}).update(applied=False, reverted=True,
                                        why_not="bake failed with the override -> original radii restored, re-baked")
    json.dump(prov, open(os.path.join(root, "provenance.json"), "w", encoding="utf-8"), indent=1)
    return True


# ------------------------------------------------------------------ E7
def va_lumen_obj(root):
    sys.path.insert(0, "/sp/v6/wall")
    import argparse as _ap
    import va_lumen_poly as LP
    LP.ARGS = _ap.Namespace(root=os.path.dirname(os.path.normpath(root)), step=LUMEN_STEP, need=LUMEN_NEED,
                            meshes="obj", union_dir=None)
    r = LP.one(os.path.basename(os.path.normpath(root)))
    return {t: r[t]["obj"]["lumen_min_mm"] for t in ("RVA", "LVA") if t in r and "obj" in r[t]}


def obj_recut_va(root, iso_path):
    """E7: returns the record; rewrites the OBJ (and mesh_v6.json / provenance obj fields) only when re-cut."""
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    obj_path = os.path.join(root, B.MESH_NAME)
    keep = open(obj_path, "rb").read()
    before = va_lumen_obj(root)
    rec = {"script": VERSION, "need_mm": LUMEN_NEED, "step_mm": LUMEN_STEP, "before": before, "tried": [],
           "date": time.strftime("%F %T")}
    if before and min(before.values()) >= LUMEN_NEED:
        rec["result"] = "ok as baked"
        return rec
    full = pv.read(os.path.join(root, B.FULL_NAME))
    iso = pv.read(iso_path)
    chosen = None
    for src, mesh, raw, targets in (("collision_full", full, False, (23500, 23450, 23400, 23300, 23200, 23000)),
                                    ("union", iso, True, (23500, 23400))):
        for tg in targets:
            m, path = B.clean_decimate(mesh, tg, raw_iso=raw)
            row = {"source": src, "target": tg, "path": path}
            if m is None:
                row["why"] = "no clean decimation"
                rec["tried"].append(row)
                continue
            rr, ok = B.route_report(m, br)
            out_n, tot_n = B.enclosure(m, br)
            row.update(clean=bool(B.is_clean(m)), routes_ok=bool(ok), centreline_outside=int(out_n), tris=int(m.n_cells))
            if not (row["clean"] and ok and out_n == 0):
                rec["tried"].append(row)
                continue
            pv.save_meshio(obj_path, m)
            lum = va_lumen_obj(root)
            row["va_lumen"] = lum
            rec["tried"].append(row)
            if lum and min(lum.values()) >= LUMEN_NEED:
                chosen = (row, m, rr, out_n)
                break
        if chosen:
            break
    if not chosen:
        with open(obj_path, "wb") as fh:
            fh.write(keep)
        rec["result"] = "no candidate keeps the VA lumen >= %.2f; OBJ unchanged (flag va_lumen_obj_fail)" % LUMEN_NEED
        rec["flag"] = "va_lumen_obj_fail"
        return rec
    row, m, rr, out_n = chosen
    rec["result"] = "re-cut from %s at %d tris" % (row["source"], row["target"])
    rec["after"] = row["va_lumen"]
    rp = os.path.join(root, V.REPORT_V5)
    rep = json.load(open(rp))
    import sdf_union_v4
    rep["obj"] = dict(B.mesh_stats(m), genus=sdf_union_v4.genus(m))
    rep["obj_tris_budget"] = row["target"]
    rep["routes"] = rr
    rep["centerline_pts_outside"] = out_n
    rep["decimation"]["obj"] = row["path"]
    rep["obj_fix_va"] = rec
    json.dump(rep, open(rp, "w"), indent=1, default=str)
    pp = os.path.join(root, "provenance.json")
    prov = json.load(open(pp))
    prov["bake_v6"].update(obj_tris=int(m.n_cells), obj_tris_budget=row["target"], obj_fix_va=rec)
    json.dump(prov, open(pp, "w"), indent=1)
    return rec


# ------------------------------------------------------------------ driver
def vm_hwm_kb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _bake_once(name, root, v4_root, pin_dir, iso_dir, T0, calls):
    changed, pinrec = setup(name, root, v4_root, pin_dir)
    print("changed carotid sections:", sorted(changed), flush=True)
    rep = B.bake_anatomy(root, B.SPACING_MM, 60000, 23500, True, iso_dir)
    return rep, changed, pinrec


def bake_anatomy_v6(name, tree, bake_dir, iso_dir, pin_dir, v4_root="/repo/carotid_data/anatomies_v4",
                    do_c4=True, do_e7=True, force=False, lattice_shift=(0.0, 0.0, 0.0)):
    root = os.path.join(tree, name)
    os.makedirs(os.path.join(bake_dir, "_done"), exist_ok=True)
    marker = os.path.join(bake_dir, "_done", name + ".bake.json")
    failed = os.path.join(bake_dir, "_done", name + ".bake_failed.json")
    if os.path.exists(marker) and not force:
        print("done already:", marker, flush=True)
        return 0
    for f in (V.REPORT_V5, B.FULL_NAME, B.MESH_NAME):
        p = os.path.join(root, f)
        if os.path.exists(p):
            print("removing partial output", p, flush=True)
            os.remove(p)
    for p in (os.path.join(iso_dir, name + ".vtp"), os.path.join(iso_dir, "_flip", name + ".vtp"), marker, failed):
        if os.path.exists(p):
            os.remove(p)
    T0 = time.time()
    calls = []
    orig = B.bake_one

    def bake_one_logged(root_, spacing, full_tris, obj_tris, verbose, dump_iso=None, far_clamp=False, far_clamp_why=None):
        rec = {"call": len(calls), "far_clamp": bool(far_clamp), "why": far_clamp_why, "started_s": round(time.time() - T0, 1)}
        calls.append(rec)
        t = time.time()
        try:
            rep_ = orig(root_, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp, far_clamp_why)
            rec.update(ok=True, passed=bool(rep_["pass"]), rung=rep_["ladder_v5b"]["rung"], flags=rep_["ladder_v5b"]["flags"],
                       attempts=len(rep_["attempts"]))
            return rep_
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error="%s: %s" % (type(e).__name__, e))
            raise
        finally:
            rec["seconds"] = round(time.time() - t, 1)
            rec["vmhwm_gb_after"] = round(vm_hwm_kb() / 1048576.0, 3)
            print("   [v6] bake_one call %d far_clamp=%s: %s in %.0f s, VmHWM %.2f GB" % (
                rec["call"], far_clamp, "ok" if rec.get("ok") else rec.get("error"), rec["seconds"], rec["vmhwm_gb_after"]),
                flush=True)

    B.bake_one = bake_one_logged
    LATTICE_SHIFT[:] = np.asarray(lattice_shift, float)
    out = {"name": name, "script": VERSION, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tree": tree,
           "lattice_shift_mm": [float(v) for v in LATTICE_SHIFT], "route_island_bridge_on": bool(ROUTE_BRIDGE["on"])}
    c4rec = None
    if do_c4:
        c4rec = apply_c4(root, backup_dir=os.path.join(bake_dir, "_c4", name))
        print("C4:", json.dumps({k: c4rec.get(k) for k in ("applied", "points_changed", "clearance_min_mm", "why_not")}), flush=True)
    out["c4"] = c4rec
    rep, err, changed, pinrec = None, None, set(), None
    print("== %s" % name, flush=True)
    for attempt in (1, 2):
        try:
            rep, changed, pinrec = _bake_once(name, root, v4_root, pin_dir, iso_dir, T0, calls)
            err = None
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rep, err = None, "%s: %s" % (type(e).__name__, e)
        ok = rep is not None and bool(rep.get("pass"))
        if ok or attempt == 2 or not (c4rec and c4rec.get("applied")):
            break
        print("C4: bake failed with the override -> restoring the original radii and re-baking", flush=True)
        if revert_c4(root, c4rec):
            c4rec = dict(c4rec, applied=False, reverted=True)
            out["c4"] = c4rec
            for f in (V.REPORT_V5, B.FULL_NAME, B.MESH_NAME):
                p = os.path.join(root, f)
                if os.path.exists(p):
                    os.remove(p)
        else:
            break
    out["changed_carotid"] = sorted(changed)
    out["pin"] = pinrec
    if rep is not None:
        record_provenance(root, rep, {"changed_carotid": sorted(changed), "lattice_shift_mm": [float(v) for v in LATTICE_SHIFT],
                                      "route_island_bridge": rep["ladder_v5b"].get("route_island_bridge"), "pin_report": os.path.join(pin_dir, name, B.REPORT),
                                      "blend_morph": MORPH.get("rec"), "chirality": MORPH.get("chirality"), "c4": c4rec})
    e7 = None
    if rep is not None and rep.get("pass") and do_e7:
        try:
            e7 = obj_recut_va(root, os.path.join(iso_dir, name + ".vtp"))
            print("E7:", e7.get("result"), e7.get("before"), e7.get("after"), flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            e7 = {"error": "%s: %s" % (type(e).__name__, e), "flag": "va_lumen_obj_check_error"}
        pp = os.path.join(root, "provenance.json")
        prov = json.load(open(pp))
        prov["bake_v6"]["e7"] = e7
        json.dump(prov, open(pp, "w"), indent=1)
    out["seconds"] = round(time.time() - T0, 1)
    out["peak_rss_gb"] = round(vm_hwm_kb() / 1048576.0, 3)
    out["calls"] = calls
    out["error"] = err
    if rep is None:
        out["passed"] = False
        atomic_json(failed, out)
        print("V6BAKE %s: FAILED %s (%.0f s)" % (name, err, out["seconds"]), flush=True)
        return 1
    lad = rep["ladder_v5b"]
    rep6 = json.load(open(os.path.join(root, V.REPORT_V5)))
    flags = list(lad["flags"]) + ([e7["flag"]] if e7 and e7.get("flag") else [])
    out.update(passed=bool(rep["pass"]), rung=lad["rung"], flags=flags, union=rep["union"],
               margin_mm=rep["margin_mm"], va_margin_mm=lad["va_margin_mm"], far_clamp=rep["far_clamp"],
               sections_tube_only=rep["sections_tube_only"], suppressed_stretches_mm=rep["suppressed_stretches_mm"],
               va_tubed=lad["va_tubed"], attempts=len(rep["attempts"]), bake_one_seconds=rep["seconds"],
               obj_tris=rep6["obj"]["tris"], obj_tris_budget=rep6["obj_tris_budget"], full_tris=rep["full"]["tris"],
               routes=rep6["routes"], centerline_pts_outside=rep6["centerline_pts_outside"],
               right_route_detail=rep["right_route_detail"], decimation=rep6["decimation"],
               iso=os.path.join(iso_dir, name + ".vtp"), blend_morph=MORPH.get("rec"), chirality=MORPH.get("chirality"), e7=e7, bake_version=VERSION,
               md5={f: _md5(os.path.join(root, f)) for f in (B.MESH_NAME, B.FULL_NAME)})
    if not rep["pass"]:
        atomic_json(failed, out)
        print("V6BAKE %s: bake did not pass (rung %s, flags %s)" % (name, lad["rung"], lad["flags"]), flush=True)
        return 1
    atomic_json(marker, out)
    print("V6BAKE %s: pass %s, rung %s, flags %s, tubed %s, %.0f s, peak RSS %.2f GB" % (
        name, rep["pass"], lad["rung"], flags,
        {k: (v["tubed_mm"], v["tubed_frac"]) for k, v in lad["va_tubed"].items()}, out["seconds"], out["peak_rss_gb"]),
        flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--tree", default=os.environ.get("A6_DIR", "/bw/tree"))
    ap.add_argument("--bake", default=os.environ.get("BF_DIR", "/bw/bake"))
    ap.add_argument("--iso", default=os.environ.get("ISO_DIR", "/bw/iso"))
    ap.add_argument("--pin", default=os.environ.get("PIN_DIR", "/bw/v4pin"))
    ap.add_argument("--v4", default="/repo/carotid_data/anatomies_v4")
    ap.add_argument("--no-c4", action="store_true")
    ap.add_argument("--no-e7", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lattice-shift", default=os.environ.get("BAKE_V6_LATTICE_SHIFT", "0,0,0"))
    ap.add_argument("--bridge-route-islands", action="store_true")
    a = ap.parse_args()
    ROUTE_BRIDGE["on"] = bool(a.bridge_route_islands)
    return bake_anatomy_v6(a.name, a.tree, a.bake, a.iso, a.pin, a.v4, not a.no_c4, not a.no_e7, a.force,
                           tuple(float(v) for v in a.lattice_shift.split(",")))


if __name__ == "__main__":
    sys.exit(main())
