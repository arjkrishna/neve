#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\pilot_build\bake_v6p.py (= /sp/v6/pilot_build/bake_v6p.py in the build containers)
"""bake_v6p: the v6 pilot bake = bake_v5b (bake_meshes_v4 unchanged + the user handle rule, D1-D8; imported
UNCHANGED) + the v6 VA-side sections (v6_va's bake_sections_v6, imported unchanged), with these stated diffs:

 E1  SECTIONS: bake_sections_v6.sections_for_v6 (v4's sections + vaC_R / vaC_L + the donor subclavian / junction
     sections scC_R, scD_R, scC_L, scD_L, scJ_L), installed into bake_meshes_v4.
 E2  BRANCH LOOKUP: bake_meshes_v4.branch(br, "(18)") raises KeyError (the loader names a branch
     "Centerline curve (18).mrk", which neither ends with "(18)" nor contains "- (18)"), so every sc* section on
     (17) / (18) / (19) would crash sections_for_v6 and bake_v5b's reach(). B.branch falls back to the ".mrk" name.
     Ownership (sdf_union_v4: samp.index_of(name) is a substring test) was already right.
 E3  VA POOL: bake_v5b.is_va also counts the sc* sections (branches (17), (18), (19)) and the CHANGED carotid
     sections (below) as searched sections: they get the VA margin ladder, VA localisation / bisection, prune /
     shrink. The UNCHANGED carotid sections stay pinned to v4 (D1) and alone make the "carotid check" union.
 E4  CHANGED CAROTID SECTIONS (set per anatomy by the driver, `changed_carotid`): a carotid section (lower_R, eca_R,
     ecaext_R, siphon_R and the left ones) whose xform (R, origin, anchor, mirror, surface) or centreline slice differs
     from v4's -- v6's re-posed siphons, substituted left forks, re-posed / skipped ECA tips. v4's pinned state for it
     is rewritten in a per-anatomy pin report (V4_DIR -> /pw/v4pin/<name>/mesh_v4.json, written by the driver):
       same source surface (re-posed piece): v4's suppressed stretches kept (handles inside a TopBrain label move
       with the piece), shifted by the section's start change on its branch; v4's whole-section tube kept;
       different surface (substituted fork): v4's stretches and tube decision for it dropped.
     Margin, far-clamp choice and every unchanged section's state are v4's as in bake_v5b.
 E5  RECORDS: report mesh_v6.json (bake_v5b's schema + "v6": changed sections, pin edits); provenance key
     "bake_v6"; the undecimated union dumped to the driver's ISO dir.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/sp/v6/va")
import bake_sections_v6  # noqa: E402,F401  (installs sections_for_v6; imports bake_v5)

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402  (installs bake_one_v5b / bake_anatomy_v5b into bake_meshes_v4)

B = V.B
VERSION = "bake_v6p 1.0 (2026-10-04; v6_pilot: bake_v5b + bake_sections_v6, E1-E5)"
VA_BRANCHES = ("RVA", "LVA", "(17)", "(18)", "(19)")
CAROTID = V.CAROTID
STATE = {"changed": set()}

# ---- E2
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


# ---- E3
def is_va_v6(sec):
    return sec.branch in VA_BRANCHES or sec.name in STATE["changed"]


V.is_va = is_va_v6

# ---- E3b (version 1.1): the VA margin ladder (0.5 -> 1.0) applies to the VA-branch sections only; a changed carotid
# section searched with the VAs keeps v4's carotid margin (k_011_left mr_012_L, bake 1.0: its handle at VA margin 0.5
# moved every changed carotid section -- the siphons, the fork ECA -- to margin 1.0 and the exact-ownership fork fell
# to 0.933 of v4). The margin is chosen by branch; the search pool stays is_va_v6.
import inspect as _inspect  # noqa: E402
_src = _inspect.getsource(V.bake_one_v5b)
_a = "        seq = [(s.name, (va_margin if is_va(s) else v4_margin), tuple(tuple(x) for x in sup.get(s.name, [])))\n"
assert _src.count(_a) == 1, "bake_v5b.attempt changed"
_src = _src.replace(_a, "        seq = [(s.name, (va_margin if s.branch in VA_BRANCHES_V6 else v4_margin), tuple(tuple(x) for x in sup.get(s.name, [])))\n")
V.VA_BRANCHES_V6 = VA_BRANCHES
exec(compile(_src, "<bake_v6p bake_one_v5b>", "exec"), V.__dict__)
B.bake_one = V.bake_one_v5b
VERSION = "bake_v6p 1.1 (2026-10-04; v6_pilot: bake_v5b + bake_sections_v6, E1-E5 + E3b margin by branch)"

# ---- E6 (version 1.2): MORPH the real surface of a blended piece. compose_v6 replaces the first blend_mm of each
# re-posed siphon (and ECA tip) by a cubic Hermite (same point count) but the bake carried the piece's real surface
# with the RIGID xform, so in that stretch the surface sat up to 1.7 mm off its centreline and the capsule clipped it:
# w_034_right mr_017_L lost the siphon relief at RCCA s 130-134 (exact-ownership siphon 0.986 of v4). Each vertex of
# the section surface is displaced by the blend's own displacement D(i) = P_blended(i) - P_rigid(i) at its nearest
# rigid-piece point (linear in the index between points; D = 0 at the seam and from blend_mm on), so the real surface
# follows the blended centreline exactly as the rigid part follows the rigid one. P_rigid is rebuilt from the v5
# piece (the carotid tool's back-map: Q = (P5 - anchor5) R5 + origin5, then (Q - origin6) R6^T + anchor6) and checked
# against the written piece beyond the blend (<= 0.01 mm), else the section is left rigid (recorded).
MORPH = {"rec": {}}
A5_ROOT = "/repo/carotid_data/anatomies_v5"
MORPH_KEYS = {"siphon_R": ("siphon", "RCCA"), "siphon_L": ("siphon_L", "LCCA"), "ecaext_R": ("ecaext_R", "RECA"),
              "ecaext_L": ("ecaext_L", "LECA")}


def _morph_section(root, sec, br):
    key, bname = MORPH_KEYS[sec.name]
    name = os.path.basename(os.path.normpath(root))
    x6 = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))["xform"].get(key) or {}
    h = float(x6.get("blend_mm") or 0)
    if h <= 0:
        return {"morphed": False, "why": "no blend"}
    p5f = os.path.join(A5_ROOT, name, "provenance.json")
    x5 = json.load(open(p5f, encoding="utf-8"))["xform"].get(key)
    if not x5:
        return {"morphed": False, "why": "no v5 piece to back-map (new piece)"}
    from eve_bench.dualdevicenav import load_branches
    b5 = B.branch(load_branches(os.path.join(A5_ROOT, name, "Centrelines_comb")), bname)
    b6 = B.branch(br, bname)
    P5, P6 = np.asarray(b5.coordinates, float), np.asarray(b6.coordinates, float)
    # NB: eve_bench's loader frame == the frame of the xforms (the bake transforms surfaces into it)
    i5 = int(np.argmin(np.linalg.norm(P5 - np.asarray(x5["anchor"], float), axis=1)))
    Q = (P5[i5:] - np.asarray(x5["anchor"], float)) @ np.asarray(x5["R"], float) + np.asarray(x5["origin"], float)
    k0 = int(np.argmin(np.linalg.norm(Q - np.asarray(x6["origin"], float), axis=1)))
    Q = Q[k0:]
    Pr = (Q - np.asarray(x6["origin"], float)) @ np.asarray(x6["R"], float).T + np.asarray(x6["anchor"], float)
    i6 = int(np.argmin(np.linalg.norm(P6 - np.asarray(x6["anchor"], float), axis=1)))
    Pb = P6[i6:]
    n = min(len(Pb), len(Pr))
    if n < 8:
        return {"morphed": False, "why": "piece too short"}
    s = B.arclen(Pr[:n])
    beyond = s > h + 1.0
    err = float(np.linalg.norm(Pb[:n][beyond] - Pr[:n][beyond], axis=1).max()) if beyond.any() else np.inf
    if not err <= 0.01:
        return {"morphed": False, "why": "rigid rebuild does not match the written piece beyond the blend (%.3f mm)" % err}
    D = Pb[:n] - Pr[:n]
    D[beyond] = 0.0
    # dense (0.1 mm) rigid curve and displacement over the blend + 4 mm
    m = s <= h + 4.0
    sd = np.arange(0.0, float(s[m][-1]) + 1e-9, 0.1)
    Pd = np.stack([np.interp(sd, s[m], Pr[:n][m][:, k]) for k in range(3)], 1)
    Dd = np.stack([np.interp(sd, s[m], D[m][:, k]) for k in range(3)], 1)
    surf = sec.surface
    V = np.asarray(surf.points, float)
    from scipy.spatial import cKDTree
    # only vertices nearer to the blend stretch than to the rest of the piece move
    d_all, _ = cKDTree(Pr[:n]).query(V)
    d_bl, j = cKDTree(Pd).query(V)
    mv = d_bl <= d_all + 1e-9
    V2 = V.copy()
    V2[mv] += Dd[j[mv]]
    import pyvista as pv
    m2 = pv.PolyData(V2, surf.faces)
    m2 = m2.compute_normals(auto_orient_normals=True, consistent_normals=True, point_normals=True, cell_normals=True,
                            inplace=False)
    sec.surface = m2
    return {"morphed": True, "blend_mm": h, "max_disp_mm": round(float(np.linalg.norm(D, axis=1).max()), 3),
            "vertices_moved": int(mv.sum()), "rigid_rebuild_err_mm": round(err, 5)}


_SECTIONS_V6 = B.sections_for


def sections_for_morph(root, br):
    secs, spans = _SECTIONS_V6(root, br)
    rec = {}
    for sec in secs:
        if sec.name in MORPH_KEYS:
            try:
                rec[sec.name] = _morph_section(root, sec, br)
            except Exception as e:  # noqa: BLE001
                rec[sec.name] = {"morphed": False, "why": "%s: %s" % (type(e).__name__, e)}
    MORPH["rec"] = rec
    return secs, spans


B.sections_for = sections_for_morph
VERSION = "bake_v6p 1.2 (2026-10-04; v6_pilot: bake_v5b + bake_sections_v6, E1-E5, E3b margin by branch, E6 blend morph)"

# ---- E5
V.REPORT_V5 = "mesh_v6.json"
B.REPORT_V5 = "mesh_v6.json"

XF_OF = {"lower_R": "lower", "eca_R": "lower", "ecaext_R": "ecaext_R", "siphon_R": "siphon",
         "lower_L": "lower_L", "eca_L": "lower_L", "ecaext_L": "ecaext_L", "siphon_L": "siphon_L"}


def carotid_slices(root):
    """{section: (branch, s0, points, radii, xform)} of the carotid sections, without loading surfaces."""
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    orig = B.load_surface
    B.load_surface = lambda *a, **k: None
    try:
        secs, spans = bake_sections_v6.bake_v5._V4_SECTIONS_FOR(root, br)
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


def pin_report(name, root, v4_root, pin_dir):
    """E4: compare the carotid sections with v4's; write the per-anatomy pin report; return (changed, record)."""
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
        same_surf = os.path.basename(str(a[4].get("surface"))) == os.path.basename(str(b[4].get("surface"))) and \
            list(a[4].get("mirror", [])) == list(b[4].get("mirror", []))
        r = {"changed": True, "points_same": bool(same_pts), "xform_same": bool(same_x), "same_surface": bool(same_surf),
             "s0_v6": a[1], "s0_v4": b[1]}
        if same_surf:
            sh = a[1] - b[1]
            if nm in sup:
                r["v4_stretches"] = sup[nm]
                sup[nm] = [[lo + sh, hi + sh] for lo, hi in sup[nm]]
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
