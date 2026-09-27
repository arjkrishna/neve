#!/usr/bin/env python3
"""Bake v4 collision meshes: tubes + every grafted real surface, merge-free.

v4 anatomies (carotid_tools/graft_v4.py) carry up to ten real-surface sections,
all recorded in provenance.json["xform"]:

    lower   (right fork: CCA/ICA on the RCCA, ECA on the RECA)   Zenodo lumen
    siphon  (right siphon on the RCCA)                           TopBrain ICA
    lower_L / siphon_L on the LCCA and LECA                      Zenodo / TopBrain
    va_R / va_L on the RVA / LVA distal segments                 TopBrain VA
    ecaext_R / ecaext_L on the RECA / LECA extensions            TopBrain ECA

The union is sdf_union_v4's OWNED union (see its docstring): real surfaces
add volume only where they own the voxel, so they cannot fuse two vessels.
Then a topology gate: every v4 anatomy is a TREE (the donor vertebrals end
blind, so the host's vertebrobasilar ring is gone), so one component of genus
0 is required -- any merge would show up as a handle. On failure:

  1. the ownership margin widens (MARGINS) -- contacts between vessels;
     tubes alone must then be a tree (else a graft gate missed a contact);
  2. the handle is localised (localise_handles: genus of the mesh patch in
     10 mm bins along each real section, pinch vertices split) and the real
     surface is suppressed in just that stretch, which is then tube -- handles
     that live INSIDE one source surface (most TopBrain ICA labels carry
     them); repeated while new stretches turn up, at the margin with fewer
     handles;
  3. if that stalls: the section whose own real surface makes a handle is
     found, and a 12 mm window slid along it with the anatomy's genus as the
     oracle;
  4. every suppressed stretch the gate does not need is restored (pruning),
     and each remaining one shrunk to the minimum the gate needs (bisection);
  5. one whole section as tube; 6. tubes only.

Then decimation to 60k (collision_full.vtp) and to the collision mesh (20k,
larger only if a route closes): VTK quadric as in v3 wherever its result is
still a clean tree, else a topology-preserving edge collapse (clean_decimate;
see the decimation notes below). Whatever was needed is recorded in the
anatomy's mesh_v4.json.

Runs in neve-build-meshlab (eve-training-fixed + pymeshlab==2023.12.post2).

    python3 topbrain_tools/bake_meshes_v4.py --anatomies carotid_data/anatomies_v4 [--shard i/n]
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
from sdf_union import load_surface
from sdf_union_v4 import WINDOW_MM, Samples, Section, add_real_sections_owned, genus

MESH_NAME = "vessel_architecture_collision.obj"
FULL_NAME = "collision_full.vtp"
REPORT = "mesh_v4.json"
CATH_R, CONTACT = 0.35, 0.3
MARGINS = (0.5, 1.0)
FAR_BELOW = -100.0          # field values below this are sdf_mesher's FAR placeholder
ROUTES = ("RCCA", "LCCA")


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def branch(br, tag):
    for b in br:
        if str(b.name).upper().endswith(tag.upper()) or ("- %s" % tag) in str(b.name):
            return b
    raise KeyError(tag)


def slice_b(b, lo, hi):
    p, r = np.asarray(b.coordinates, float), np.asarray(b.radii, float)
    s = arclen(p)
    m = (s >= lo - 1e-6) & (s <= hi + 1e-6)
    return p[m], r[m], float(s[-1])


def sections_for(root, br):
    d = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
    x = d["xform"]
    secs, spans = [], {}

    def add(name, xf, bname, lo, hi, t0, t1, siblings=()):
        if xf.get("real_surface") is False:
            spans[name] = "tube only (%s)" % xf.get("why_tube", "repair moved the centerline")
            return
        b = branch(br, bname)
        p, r, L = slice_b(b, lo, hi if hi is not None else 1e9)
        if len(p) < 4:
            return
        surf = load_surface(xf["surface"], xf["mirror"], xf["R"], xf["origin"], xf["anchor"])
        secs.append(Section(name, surf, p, r, t0, t1, bname, lo, siblings))
        spans[name] = (bname, float(lo), float(min(hi if hi is not None else L, L)))

    # right: lower (route slice + its ECA), siphon, optional ECA extension
    lo, cca, ica = d["host_cut_mm"], d["cca_mm"], d["ica_mm"]
    lx = x["lower"]
    real_end = lo + cca + min(ica, lx.get("ica_real_mm", ica))
    add("lower_R", lx, "RCCA", lo, real_end, True, True, siblings=("RECA",))
    ext = x.get("ecaext_R")
    add("eca_R", lx, "RECA", 0.0, ext["join_at_mm"] if ext else None, False, True, siblings=("RCCA",))
    if ext:
        add("ecaext_R", ext, "RECA", ext["join_at_mm"], None, True, False)
    add("siphon_R", x["siphon"], "RCCA", lo + cca + ica, None, True, False)
    # left
    L = d["left"]
    lo_l, cca_l, ica_l = L["host_cut_mm"], L["cca_mm"], L["ica_mm"]
    lxl = x["lower_L"]
    add("lower_L", lxl, "LCCA", lo_l, lo_l + cca_l + min(ica_l, lxl.get("ica_real_mm", ica_l)), True, True,
        siblings=("LECA",))
    ext = x.get("ecaext_L")
    add("eca_L", lxl, "LECA", 0.0, ext["join_at_mm"] if ext else None, False, True, siblings=("LCCA",))
    if ext:
        add("ecaext_L", ext, "LECA", ext["join_at_mm"], None, True, False)
    add("siphon_L", x["siphon_L"], "LCCA", lo_l + cca_l + ica_l, None, True, False)
    # vertebrals
    for key, bname in (("va_R", "RVA"), ("va_L", "LVA")):
        add(key, x[key], bname, x[key]["cut_mm"], None, True, False)
    return secs, spans


def enclosure(mesh, br, cap_mm=3.0):
    """Centerline points (all branches, end caps excluded) outside the mesh."""
    import pyvista as pv
    pts = []
    for b in br:
        p = np.asarray(b.coordinates, float)
        s = arclen(p)
        pts.append(p[(s > cap_mm) & (s < s[-1] - cap_mm)])
    P = np.vstack([p for p in pts if len(p)])
    ins = np.asarray(pv.PolyData(P).select_enclosed_points(mesh, tolerance=0.0, check_surface=False)["SelectedPoints"], bool)
    return int((~ins).sum()), int(len(P))


def route_report(mesh, br):
    out, ok_all = {}, True
    for tag in ROUTES:
        b = branch(br, tag)
        route, rad = np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        d, ins, body, s = route_lumen(mesh, route)
        ok = ins & body
        if ok.any():
            i = int(np.argmin(np.where(ok, d, np.inf)))
            nav = bool(d[ok].min() - CONTACT >= CATH_R)
            out[tag] = {"lumen_min_mm": round(float(d[i]), 3), "lumen_min_at_mm": round(float(s[i]), 1),
                        "declared_there_mm": round(float(rad[i]), 3),
                        "median_deficit_mm": round(float(np.median(rad[ok] - d[ok])), 3),
                        "route_pts_outside": int((~ins & body).sum()), "navigable": nav}
        else:
            nav = False
            out[tag] = {"route_pts_outside": int(body.sum()), "navigable": False}
        ok_all &= nav and out[tag]["route_pts_outside"] == 0
    return out, ok_all


LOCALISE_ROUNDS = 5
BIN_MM, BIN_STRIDE_MM, BIN_PAD_MM = 10.0, 5.0, 3.0
WINDOW_SEARCH_MM, WINDOW_STRIDE_MM = 12.0, 6.0
SHRINK_TOL_MM = 2.0
MAX_WINDOW_SEARCHES = 4
TUBE_IN_EFFECT_FRAC = 0.9


def piece_genus(faces):
    """Genus of each component of an OPEN triangle patch, pinch vertices split.

    A patch clipped out of a closed manifold keeps a vertex whose incident
    faces were kept in two separate fans; counted as one vertex, with its two
    boundary loops merged into one, such a pinch reads as genus +0.5 (the v4
    audit: 147 of the first bake's localiser hits were exactly that, and about
    half of all suppressed stretches held no handle). Each fan gets its own
    vertex here, so the count is that of the manifold patch. Returns
    [(genus, boundary_loops, n_faces)] per component of 50+ faces."""
    f = np.asarray(faces, np.int64).copy()
    inc = {}
    for fi, t in enumerate(f):
        for k in range(3):
            inc.setdefault(int(t[k]), []).append((fi, int(t[(k + 1) % 3]), int(t[(k + 2) % 3])))
    nxt = int(f.max()) + 1 if len(f) else 0
    for v, ed in inc.items():
        par = {}

        def find(x):
            while par.get(x, x) != x:
                x = par[x]
            return x
        for _, a, b in ed:
            par.setdefault(a, a); par.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
        fans = {}
        for fi, a, _ in ed:
            fans.setdefault(find(a), []).append(fi)
        for fan in list(fans.values())[1:]:
            for fi in fan:
                f[fi][f[fi] == v] = nxt
            nxt += 1
    # components by shared vertex (after the split)
    par = list(range(len(f)))

    def fnd(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    first = {}
    for fi, t in enumerate(f):
        for v in t:
            v = int(v)
            if v in first:
                a, b = fnd(fi), fnd(first[v])
                if a != b:
                    par[a] = b
            else:
                first[v] = fi
    comp = np.array([fnd(i) for i in range(len(f))])
    out = []
    for c in np.unique(comp):
        fc = f[comp == c]
        if len(fc) < 50:
            continue
        e = np.sort(np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]]), axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        V, E, F = len(np.unique(fc)), len(ue), len(fc)
        bnd = ue[cnt == 1]
        # boundary loops: components of the boundary-edge graph (degree 2 after the split)
        b = 0
        if len(bnd):
            bp = {}

            def bf(x):
                while bp.get(x, x) != x:
                    x = bp[x]
                return x
            for a_, b_ in bnd:
                bp.setdefault(int(a_), int(a_)); bp.setdefault(int(b_), int(b_))
                ra, rb = bf(int(a_)), bf(int(b_))
                if ra != rb:
                    bp[ra] = rb
            b = len({bf(x) for x in bp})
        out.append(((2 - (V - E + F) - b) / 2.0, b, F))
    return out


def localise_handles(mesh, secs):
    """Stretches of each real section whose mesh neighbourhood carries a handle.

    The mesh is clipped to the neighbourhood of a section (within 1.8 r + 2 mm
    of its centerline) in overlapping 10 mm bins; a bin whose patch has genus
    >= 1 with pinch vertices split (piece_genus) holds a handle. The quick
    pinch-blind count screens first. Returns {section: [(lo, hi), ...]} in
    branch arclength, padded by BIN_PAD_MM."""
    from scipy.spatial import cKDTree
    cen = np.asarray(mesh.cell_centers().points)
    out = {}
    for sec in secs:
        p, r = sec.points, sec.radii
        if len(p) < 3:
            continue
        s = arclen(p)
        d, i = cKDTree(p).query(cen)
        near = d < (1.8 * r[i] + 2.0)
        si = s[i]
        for lo in np.arange(0.0, max(s[-1] - 1e-6, 0.0), BIN_STRIDE_MM):
            m = near & (si >= lo) & (si < lo + BIN_MM)
            if m.sum() < 50:
                continue
            piece = mesh.extract_cells(np.nonzero(m)[0]).extract_surface().clean().triangulate()
            if not any(c["genus"] > 0.25 for c in genus(piece) if c["cells"] > 50):
                continue
            faces = np.asarray(piece.faces).reshape(-1, 4)[:, 1:]
            if any(g >= 0.75 for g, _, _ in piece_genus(faces)):
                out.setdefault(sec.name, []).append(
                    (sec.s0 + max(lo - BIN_PAD_MM, 0.0), sec.s0 + min(lo + BIN_MM + BIN_PAD_MM, s[-1])))
    return out


# ---- decimation ---------------------------------------------------------------
# The union passes the topology gate BEFORE decimation, and VTK's quadric edge
# collapse (the v3 protocol, sdf_mesher.decimate_to) is kept wherever its result
# is still a clean tree. It has no topology guard, though: where the union holds
# material narrower than the local edge length (0.8 mm at 200k tris, 1.6 mm at
# 60k, 2.8 mm at 20k) collapsing that material's rim glues its two walls. On the
# first full v4 bake that broke 30 of 223 unions -- a 3-face fin with free edges,
# a 4-face edge, or a small lobe pinched onto the tree at one vertex -- at 10
# physical sites: blades 0.02-0.3 mm thick where an ownership, capsule or
# suppression cut slices a real surface (the carina of case_w_016_left, a side
# branch of case_w_033_right's ECA), clefts in two TopBrain labels (mr_021 lVA,
# mr_007 siphon genu), an ECA side-branch stub of case_w_007_left, and one fold
# with no thin material at all. The collapse ORDER does not depend on the target
# (the defect edge sits at identical coordinates at 59k, 60k and 61k), so
# retrying nearby targets cannot help; nor can splitting pinch vertices, which
# on one anatomy dropped a real 36 mm3 label lobe as a "bubble".
#
# The fallback is an edge collapse that enforces the link condition, so it
# cannot glue walls whatever the source of the thin material: vcg's quadric
# collapse with preservetopology (pymeshlab). For the raw iso it is preceded by
# one pass of vcg's marching-cubes edge collapse (better worst-case error); for
# the obj step, whose input is the adaptive 60k mesh, vertices are weighted by
# their inverse mean incident face area, because vcg's area-weighted quadrics
# otherwise strip small vessels harder than VTK does. Measured on 30 cached
# unions (21 failing, 9 controls): 30/30 clean with both routes navigable at
# the 20k budget, obj fidelity to the undecimated union p99 0.135 mm against
# VTK's 0.134; deterministic across processes and containers.
# pymeshlab remembers the last value of any filter parameter not passed, so
# every parameter is passed explicitly. Needs pymeshlab (neve-build-meshlab:
# eve-training-fixed + pymeshlab==2023.12.post2); imported only on fallback.

def manifold_defects(mesh):
    """(non-manifold edges, pinch vertices, open edges) of a triangle mesh."""
    m = mesh.clean().triangulate()
    nme = m.extract_feature_edges(boundary_edges=False, non_manifold_edges=True,
                                  feature_edges=False, manifold_edges=False).n_cells
    f = np.asarray(m.faces).reshape(-1, 4)[:, 1:]
    inc = {}
    for t in f:
        for k in range(3):
            inc.setdefault(int(t[k]), []).append((int(t[(k + 1) % 3]), int(t[(k + 2) % 3])))
    pinch = 0
    for ed in inc.values():
        par = {}

        def find(x):
            while par.get(x, x) != x:
                x = par[x]
            return x
        for a, b in ed:
            par.setdefault(a, a); par.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
        if len({find(x) for x in par}) > 1:
            pinch += 1
    return int(nme), int(pinch), int(m.n_open_edges)


def edge_defects(mesh):
    """Edges shared by more than two faces, or by one, after coincident points
    are merged (as genus() merges them). Fast; no pinch test -- a pinch vertex
    already moves the Euler count off 0."""
    f = np.asarray(mesh.clean().triangulate().faces).reshape(-1, 4)[:, 1:]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt != 2).sum())


def self_intersections(mesh):
    """Faces that intersect another face of the same mesh (pymeshlab)."""
    ms = _to_meshset(mesh)
    ms.compute_selection_by_self_intersections_per_face()
    return int(ms.current_mesh().selected_face_number())


def is_clean(mesh):
    """One closed component of genus 0 with no non-manifold edge or vertex,
    and no face crossing another. Topology alone is not enough: the v4 audit
    found vcg's topology-preserving collapse folding faces back onto their
    neighbours (normals reversed) in 15 of the 19 collision meshes it made,
    13 of them on the catheter's path, all topologically clean."""
    if mesh is None or mesh.n_cells == 0:
        return False
    g = genus(mesh)
    if not (len(g) == 1 and g[0]["genus"] == 0 and g[0]["boundary_loops"] == 0):
        return False
    if manifold_defects(mesh) != (0, 0, 0):
        return False
    return self_intersections(mesh) == 0


def _to_meshset(mesh, quality=None):
    import pymeshlab
    m = mesh.triangulate()
    F = np.asarray(m.faces).reshape(-1, 4)[:, 1:].astype(np.int32)
    kw = {} if quality is None else {"v_scalar_array": np.asarray(quality, np.float64)}
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(m.points, np.float64), face_matrix=F, **kw))
    return ms


def _from_meshset(ms):
    import pyvista as pv
    cm = ms.current_mesh()
    F = np.asarray(cm.face_matrix(), np.int64)
    return pv.PolyData(np.asarray(cm.vertex_matrix(), np.float64), np.c_[np.full(len(F), 3), F].ravel())


def _inverse_area_quality(mesh, k=1.5, clip=(1.0, 99.0)):
    """Per-vertex weight (mean incident face area) ** -k, percentile-clipped; None if flat."""
    m = mesh.triangulate()
    V, F = np.asarray(m.points, float), np.asarray(m.faces).reshape(-1, 4)[:, 1:]
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    s = np.bincount(F.ravel(), weights=np.repeat(A, 3), minlength=len(V))
    c = np.bincount(F.ravel(), minlength=len(V))
    q = np.maximum(s / np.maximum(c, 1), 1e-12) ** (-k)
    lo, hi = np.percentile(q, clip)
    return np.clip(q, lo, hi) if hi - lo > 1e-9 * abs(hi) else None


def _vcg_quadric(mesh, n_tris, qualitythr=0.3, quality=None, preservenormal=True):
    ms = _to_meshset(mesh, quality)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(n_tris), targetperc=0.0, qualitythr=float(qualitythr),
        preserveboundary=False, boundaryweight=1.0, preservenormal=bool(preservenormal), preservetopology=True,
        optimalplacement=True, planarquadric=False, planarweight=0.001,
        qualityweight=quality is not None, autoclean=True, selected=False)
    return _from_meshset(ms)


def _unfold(mesh):
    """vcg's folded-face removal (edge flips): a fold-over becomes flat again."""
    ms = _to_meshset(mesh)
    ms.meshing_remove_folded_faces()
    return _from_meshset(ms)


def _vcg_mc_collapse(mesh):
    """One pass of vcg's marching-cubes edge collapse. Returned through numpy:
    the quadric filter is a silent no-op on the MeshSet this filter leaves."""
    ms = _to_meshset(mesh)
    ms.meshing_decimation_edge_collapse_for_marching_cube_meshes()
    return _from_meshset(ms)


def clean_decimate(mesh, n_tris, raw_iso):
    """Decimate to n_tris keeping a clean tree. Returns (mesh, path) or (None, None).

    raw_iso: the input is the marching-cubes union itself (the full step), not
    an already decimated mesh (the obj step). VTK first (the v3 protocol); then
    vcg's topology-preserving collapse, with face flips refused
    (preservenormal) first and allowed second, each followed by vcg's
    folded-face removal if the result still crosses itself."""
    out = decimate_to(mesh, n_tris)
    if is_clean(out):
        return out, "vtk_quadric"
    cands = []
    for pn in (True, False):
        tag = "" if pn else ", flips allowed"
        if raw_iso:
            cands.append(("vcg_mc_collapse+quadric_preserve_topology" + tag,
                          lambda pn=pn: _vcg_quadric(_vcg_mc_collapse(mesh), n_tris, preservenormal=pn)))
        q = _inverse_area_quality(mesh)
        if q is not None:
            cands.append(("vcg_quadric_preserve_topology_area_weighted" + tag,
                          lambda pn=pn, q=q: _vcg_quadric(mesh, n_tris, qualitythr=0.1, quality=q, preservenormal=pn)))
        cands.append(("vcg_quadric_preserve_topology" + tag,
                      lambda pn=pn: _vcg_quadric(mesh, n_tris, preservenormal=pn)))
    for name, make in cands:
        out = make()
        if out.n_cells <= 1.02 * n_tris:          # the budget met (a stalled collapse is refused)
            if is_clean(out):
                return out, name
            out = _unfold(out)
            if is_clean(out):
                return out, name + "+unfold"
    return None, None


def right_route_detail(mesh, samp, root):
    """Wall detail of the union on the right route (the training route): area
    fraction of RCCA-owned wall past the host cut standing > 0.3 / > 0.5 mm off
    the plain tube (|x - c| - r), on the fork (< 130 mm) and the siphon.
    Measured on the undecimated union, so it compares with v3's union on the
    same grid, free of triangle-size effects."""
    from scipy.spatial import cKDTree
    with open(os.path.join(root, "provenance.json"), encoding="utf-8") as f:
        cut = float(json.load(f).get("host_cut_mm", 0.0))
    rc = samp.index_of("RCCA")
    cen = np.asarray(mesh.cell_centers().points)
    A = np.asarray(mesh.compute_cell_sizes(length=False, volume=False)["Area"])
    dd, ii = cKDTree(samp.p).query(cen, k=24)
    best = ii[np.arange(len(ii)), np.argmax(samp.r[ii] - dd, axis=1)]
    own = (samp.b[best] == rc) & (samp.s[best] >= cut)
    dep = np.abs(np.linalg.norm(cen - samp.p[best], axis=1) - samp.r[best])
    out = {}
    for seg, m in (("fork", own & (samp.s[best] < 130.0)), ("siphon", own & (samp.s[best] >= 130.0))):
        a, x = A[m], dep[m]
        tot = float(a.sum()) or 1.0
        out[seg] = {"frac_0.3mm": round(float(a[x > 0.3].sum()) / tot, 4),
                    "frac_0.5mm": round(float(a[x > 0.5].sum()) / tot, 4)}
    return out


def tube_in_effect(secs, suppress):
    """Sections whose suppressed stretches cover TUBE_IN_EFFECT_FRAC of them."""
    out = []
    for sec in secs:
        L = float(arclen(sec.points)[-1])
        if L <= 0 or sec.name not in suppress:
            continue
        cov = sum(max(0.0, min(b, sec.s0 + L) - max(a, sec.s0)) for a, b in suppress[sec.name])
        if cov >= TUBE_IN_EFFECT_FRAC * L:
            out.append(sec.name)
    return out


def merge_spans(spans):
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def keep_largest(full):
    """The largest component, and what was dropped: {"cells": sizes, "volume_mm3"}.

    The pieces dropped are real lumen the union cut loose -- ECA beyond the
    modelled ECA tip where the route's capsule reaches it, and the like: no
    centerline runs through them (the enclosure check would say so)."""
    st = mesh_stats(full)
    dropped = {"cells": [], "volume_mm3": 0.0}
    if st["comps"] > 1:
        conn = full.connectivity()
        rid = np.asarray(conn.cell_data["RegionId"])
        sizes = np.bincount(rid)
        keep = int(np.argmax(sizes))
        others = [int(k) for k in np.argsort(-sizes) if k != keep]
        dropped["cells"] = [int(sizes[k]) for k in others]
        vol = 0.0
        for k in others[:20]:
            piece = conn.extract_cells(np.nonzero(rid == k)[0]).extract_surface()
            vol += abs(float(piece.volume)) if piece.n_cells else 0.0
        dropped["volume_mm3"] = vol
        full = conn.extract_cells(np.nonzero(rid == keep)[0]).extract_surface().triangulate()
    return full, st, dropped


class NoCleanDecimation(RuntimeError):
    """The gate-passing union decimates to no clean tree at any full budget."""


def obj_ladder(full, br, obj_tris):
    """The collision mesh: the smallest budget that decimates clean with both
    routes navigable; the full mesh itself only if none does.
    Returns (obj, budget, path, rejected)."""
    tried = []
    for cand in (obj_tris, int(obj_tris * 1.5), obj_tris * 2, obj_tris * 3):
        if cand >= full.n_cells:
            break
        m, path = clean_decimate(full, cand, raw_iso=False)
        if m is None:
            tried.append({"budget": cand, "why": "no clean decimation"})
            continue
        rr, ok = route_report(m, br)
        if ok:
            return m, cand, path, tried
        tried.append({"budget": cand, "why": "route not navigable"})
    return full, int(full.n_cells), "full mesh (no smaller budget passed)", tried


def bake_anatomy(root, spacing, full_tris, obj_tris, verbose, dump_iso=None):
    """bake_one as proven (no FAR clamp); with the clamp only if that union
    cannot be decimated into a clean tree.

    The clamp (see attempt()) changes marching cubes wherever an inside voxel
    meets the band edge, and so could change unions that already bake clean
    -- 222 of 223 in bake 9, every final mesh verified. As with decimation,
    the proven path stays first and the fix is the fallback: it is needed
    where the unclamped union hides a handle behind one duplicated vertex,
    which no decimator can then turn into a clean tree."""
    try:
        first = bake_one(root, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp=False)
    except NoCleanDecimation as e:
        return bake_one(root, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp=True,
                        far_clamp_why=str(e))
    lost = len(first["sections_tube_only"]) + (100 if first["union"].startswith("TUBES") else 0)
    if not lost:
        return first
    # the unclamped union needed a whole section as tube: a seam the clamp
    # would have exposed as a handle can stall localisation (on
    # case_w_008_right__topcow_mr_001 it cost the right siphon, where the
    # clamped union has a 12 mm fix). Keep whichever loses less real surface.
    keep = {f: open(os.path.join(root, f), "rb").read() for f in (FULL_NAME, MESH_NAME, REPORT)}
    try:
        second = bake_one(root, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp=True,
                          far_clamp_why="the unclamped union needed %s as tube" % (first["sections_tube_only"] or "tubes only"))
        lost2 = len(second["sections_tube_only"]) + (100 if second["union"].startswith("TUBES") else 0)
        if second["pass"] and lost2 < lost:
            return second
    except NoCleanDecimation:
        pass
    for f, b in keep.items():
        with open(os.path.join(root, f), "wb") as fh:
            fh.write(b)
    return first


def recut_obj(root, obj_tris):
    """Re-cut only the collision mesh from the anatomy's existing full mesh
    (what bake_one does after the union), and update its report."""
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    full = pv.read(os.path.join(root, FULL_NAME))
    with open(os.path.join(root, REPORT), encoding="utf-8") as f:
        rep = json.load(f)
    obj, obj_used, obj_path, tried = obj_ladder(full, br, obj_tris)
    pv.save_meshio(os.path.join(root, MESH_NAME), obj)
    rr_obj, ok_obj = route_report(obj, br)
    out_n, tot_n = enclosure(obj, br)
    rep.update(obj=dict(mesh_stats(obj), genus=genus(obj)), obj_tris_budget=obj_used, routes=rr_obj,
               centerline_pts_outside=out_n, centerline_pts_checked=tot_n)
    rep["decimation"] = dict(rep.get("decimation") or {}, obj=obj_path, obj_budgets_rejected=tried)
    rep["recut"] = {"obj_tris": obj_tris, "from": FULL_NAME,
                    "note": "collision mesh re-cut from the unchanged full mesh by obj_ladder, as bake_one does"}
    rep["pass"] = bool(ok_obj and is_clean(obj) and out_n == 0)
    with open(os.path.join(root, REPORT), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def bake_one(root, spacing, full_tris, obj_tris, verbose, dump_iso=None, far_clamp=False, far_clamp_why=None):
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    t0 = time.time()
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs, spans = sections_for(root, br)
    attempts = []
    full, mode, used_margin, tube_only = None, None, None, []

    suppress = {}

    def fmt(sup):
        return {k: [[round(a, 2), round(b, 2)] for a, b in v] for k, v in sup.items()}

    def attempt(sections, margin, tag, sup=None):
        field = tube_field(br, spacing=spacing, verbose=verbose)
        st = (add_real_sections_owned(field, sections, samp, spacing, margin, verbose, suppress=sup)
              if sections else {})
        # voxels never computed hold the FAR placeholder (-1000). Next to a
        # barely-positive voxel at a band edge, marching cubes puts the crossing
        # at t ~ 1e-5 -- on the grid node, where another edge's crossing lands
        # too: two sheets meeting at one duplicated vertex, a seam the Euler
        # count reads as genus 0 while the surface is really a handle
        # (case_w_008_right__topcow_mr_001). A signed-distance field never
        # drops more than one grid step between neighbours, so FAR is clamped
        # to two steps below zero: no real value moves, every crossing is
        # well defined, and the gate sees the true genus. Used only as the
        # fallback (bake_anatomy).
        if far_clamp:
            far = field.values < FAR_BELOW
            field.values[far] = -2.0 * spacing
            del far
        iso, st_iso, dropped = keep_largest(iso_surface(field))
        g = genus(iso)
        gmax = max(c["genus"] for c in g) if g else -1
        nme = edge_defects(iso)
        if gmax == 0 and nme:
            # marching cubes let two sheets touch along an edge seam: the
            # Euler count reads 0, but split there the surface is a handle,
            # and every decimator (VTK, vcg topology-preserving) gives genus 1
            # (case_w_008_right__topcow_mr_001, 3 such edges). Not a tree yet.
            gmax = 0.5
        attempts.append(dict(tag, genus=gmax, nonmanifold_edges=nme, components_before_keep=st_iso["comps"],
                             dropped_fragments=dropped["cells"][:10],
                             dropped_volume_mm3=round(dropped["volume_mm3"], 2)))
        if verbose:
            print("   %s -> genus %s, %d comps" % (tag, gmax, st_iso["comps"]), flush=True)
        return iso, st, gmax

    def with_span(sup, name, span):
        out = {k: list(v) for k, v in sup.items()}
        out[name] = merge_spans(out.get(name, []) + [span])
        return out

    def replace_span(sup, name, old, new):
        out = {k: list(v) for k, v in sup.items()}
        out[name] = [new if x == old else x for x in out[name]]
        return out

    def without_span(sup, name, span):
        out = {k: [s for s in v if s != span] for k, v in sup.items()}
        return {k: v for k, v in out.items() if v}

    # 1. every real surface, widening the ownership margin (inter-vessel contacts)
    isos, gen = {}, {}
    for margin in MARGINS:
        iso, st, g = attempt(secs, margin, {"margin_mm": margin})
        isos[margin], gen[margin] = (iso, st), g
        if g == 0:
            full, sec_stats, mode, used_margin = iso, st, "owned", margin
            break
    tubes = None
    m_loc = None
    if full is None:
        # tubes alone must be a tree, or nothing below can converge: a handle
        # here is a centerline-level fusion (a graft gate missed a contact)
        tubes = attempt([], None, {"margin_mm": None, "tubes_check": True})
        if tubes[2] != 0:
            raise RuntimeError("even the tube-only mesh has a handle: a centerline-level fusion")
        tubes = tubes[:2]
    if full is None and secs:
        # 2. handles that live inside one real surface (label / lumen
        #    segmentation artefacts: 16 of 25 right and 20 of 25 left TopBrain
        #    ICA labels carry them): find the stretch, make just that stretch
        #    tube, re-bake; everything else stays real. Localise at the margin
        #    with fewer handles, 0.5 on a tie: it keeps more real surface
        #    (margin 1.0 had MORE handles in 3 of 64 localised anatomies).
        m_loc = min(MARGINS, key=lambda x: (gen[x], x))
        (iso, st), g = isos[m_loc], gen[m_loc]
        for rnd in range(1, LOCALISE_ROUNDS + 1):
            found = localise_handles(iso, secs)
            new = False
            for name, spans_ in found.items():
                cur = suppress.get(name, [])
                for lo, hi in spans_:
                    if not any(a <= lo and hi <= b for a, b in cur):
                        cur = merge_spans(cur + [(lo, hi)]); new = True
                suppress[name] = cur
            if not new:
                break
            iso, st, g = attempt(secs, m_loc, {"margin_mm": m_loc, "round": rnd, "suppressed": fmt(suppress)},
                                 sup=suppress)
            if g == 0:
                full, sec_stats, mode, used_margin = iso, st, "owned", m_loc
                break
        # 3. localisation stalled -- the handle is larger than a bin, or its
        #    material belongs to a sibling section cut from the same source
        #    surface (a lower and its own ECA). Find the section whose own real
        #    surface makes a handle with the tubes, and slide a window along it
        #    with the whole anatomy's genus as the oracle; keep the window that
        #    lowers it most. The first bake dropped 12 whole sections here, two
        #    of them route siphons, where a 10 mm window sufficed in every case.
        searches = 0
        while full is None and searches < MAX_WINDOW_SEARCHES:
            searches += 1
            off = None
            for sec in secs:
                _, _, gs = attempt([sec], m_loc, {"margin_mm": m_loc, "probe": sec.name}, sup=suppress)
                if gs > 0:
                    off = sec
                    break
            if off is None:
                break
            L = float(arclen(off.points)[-1])
            inside = [(off.s0 + lo, off.s0 + min(lo + WINDOW_SEARCH_MM, L))
                      for lo in np.arange(0.0, max(L - WINDOW_SEARCH_MM, 0.0) + 1e-6, WINDOW_STRIDE_MM)]
            # a section's real surface owns voxels up to WINDOW_MM of arclength
            # past its own ends (ownership's self window), so a handle can sit
            # there, out of reach of the windows above (ecaext_L of
            # case_w_051_right__topcow_mr_012); tried only if none inside helped
            beyond = [(off.s0 - WINDOW_MM, off.s0 + min(WINDOW_SEARCH_MM / 2, L)),
                      (off.s0 + max(L - WINDOW_SEARCH_MM / 2, 0.0), off.s0 + L + WINDOW_MM)]
            best = None
            for batch in (inside, beyond):
                for span in batch:
                    if any(a <= span[0] and span[1] <= b for a, b in suppress.get(off.name, [])):
                        continue
                    trial = with_span(suppress, off.name, span)
                    iso_w, st_w, g_w = attempt(secs, m_loc, {"margin_mm": m_loc,
                                                             "window": [off.name, round(span[0], 2), round(span[1], 2)]},
                                               sup=trial)
                    if g_w < g and (best is None or g_w < best[0]):
                        best = (g_w, trial, iso_w, st_w)
                    if g_w == 0:
                        break
                if best is not None:
                    break
            if best is None:
                break
            g, suppress, iso, st = best
            if g == 0:
                full, sec_stats, mode, used_margin = iso, st, "owned", m_loc
        # 4. prune: a stretch the gate does not need goes back to real surface
        #    (the first bake's localiser suppressed 161 stretches, 82 of them
        #    unnecessary)
        if full is not None:
            for name in list(suppress):
                for span in list(suppress.get(name, [])):
                    trial = without_span(suppress, name, span)
                    iso_p, st_p, g_p = attempt(secs, m_loc, {"margin_mm": m_loc,
                                                             "prune": [name, round(span[0], 2), round(span[1], 2)]},
                                               sup=trial)
                    if g_p == 0:
                        suppress, full, sec_stats = trial, iso_p, st_p
        # 4b. shrink: each remaining stretch loses real surface over its whole
        #     length, but the handle it removes is usually a few mm; move each
        #     end inward by bisection while the anatomy stays genus 0
        #     (SHRINK_TOL_MM). The bins and windows are 10-16 mm, and the TopBrain
        #     loops sit in stretches of that size: without this, the anatomies
        #     whose siphon label carries loops kept ~40% less siphon wall detail
        #     than v3, which kept the loops.
        if full is not None:
            for name in list(suppress):
                for span in list(suppress.get(name, [])):
                    a, b = span
                    cur = span
                    for end in ("lo", "hi"):
                        ok_, bad_ = (a, b) if end == "lo" else (b, a)
                        while abs(bad_ - ok_) > SHRINK_TOL_MM:
                            mid = 0.5 * (ok_ + bad_)
                            new = (mid, b) if end == "lo" else (a, mid)
                            trial = replace_span(suppress, name, cur, new)
                            iso_s, st_s, g_s = attempt(secs, m_loc, {"margin_mm": m_loc, "shrink": [
                                name, round(new[0], 2), round(new[1], 2)]}, sup=trial)
                            if g_s == 0:
                                ok_, suppress, full, sec_stats, cur = mid, trial, iso_s, st_s, new
                            else:
                                bad_ = mid
                        if end == "lo":
                            a = ok_
                        else:
                            b = ok_
    if full is None and secs:
        # 5. nothing local worked: bake one whole section as a tube, the one
        #    with the least real surface first, keeping every other section
        m5 = m_loc if m_loc is not None else MARGINS[-1]
        order = sorted(secs, key=lambda s_: (isos[m5][1].get(s_.name, {}).get("voxels_widened", 0), s_.name))
        for sec in order:
            iso, st, g = attempt([s for s in secs if s is not sec], m5,
                                 {"margin_mm": m5, "without": sec.name}, sup=suppress)
            if g == 0:
                full, sec_stats, mode, used_margin, tube_only = iso, st, "owned", m5, [sec.name]
                spans[sec.name] = "tube only (no local fix found)"
                break
    if full is None:
        # 6. last resort: tubes only (v2 geometry), reported as such
        full, sec_stats = tubes
        mode, used_margin = "tubes", None
    iso_full = full
    detail = right_route_detail(iso_full, samp, root)
    if dump_iso:
        # the gate-passing union before decimation, for decimation studies
        os.makedirs(dump_iso, exist_ok=True)
        iso_full.save(os.path.join(dump_iso, os.path.basename(os.path.normpath(root)) + ".vtp"))
    # the full mesh is the gitignored reference and the source of the
    # collision mesh; if no decimation to its budget is clean, a larger one is
    full, full_path, full_used = None, None, None
    for fb in (full_tris, full_tris * 4 // 3, full_tris * 5 // 3, full_tris * 2, full_tris * 3):
        full, full_path = clean_decimate(iso_full, fb, raw_iso=True)
        if full is not None:
            full_used = fb
            break
    if full is None:
        raise NoCleanDecimation("no decimation of the genus-0 union to %d-%d tris is a clean tree"
                                % (full_tris, full_tris * 3))
    full.save(os.path.join(root, FULL_NAME))
    obj, obj_used, obj_path, tried = obj_ladder(full, br, obj_tris)
    pv.save_meshio(os.path.join(root, MESH_NAME), obj)
    rr_obj, ok_obj = route_report(obj, br)
    go = genus(obj)
    out_n, tot_n = enclosure(obj, br)
    rep = {"spacing_mm": spacing, "seconds": round(time.time() - t0, 1),
           "union": {"owned": "owned (sdf_union_v4)" if not tube_only else "owned, %s as tube" % ",".join(tube_only),
                     "tubes": "TUBES ONLY (no owned union passed the topology gate)"}[mode],
           "sections_tube_only": tube_only,
           "suppressed_stretches_mm": fmt(suppress),
           "sections_tube_in_effect": tube_in_effect(secs, suppress),
           "localise_margin_mm": m_loc,
           "far_clamp": {"used": bool(far_clamp), "why": far_clamp_why},
           "right_route_detail": detail,
           "margin_mm": used_margin, "attempts": attempts, "sections": sec_stats, "spans": spans,
           "full": dict(mesh_stats(full), genus=genus(full)), "obj": dict(mesh_stats(obj), genus=go),
           "obj_tris_budget": obj_used, "routes": rr_obj,
           "decimation": {"full": full_path, "full_budget": full_used, "obj": obj_path, "obj_budgets_rejected": tried},
           "centerline_pts_outside": out_n, "centerline_pts_checked": tot_n,
           "pass": bool(ok_obj and is_clean(obj) and out_n == 0)}
    with open(os.path.join(root, REPORT), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", required=True)
    ap.add_argument("--spacing", type=float, default=SPACING_MM)
    ap.add_argument("--full-tris", type=int, default=60000)
    # v2/v3 used 20000. v4 grafts real surface onto all six neck vessels, and
    # at a fixed budget that detail takes triangles from the siphon (median 15%
    # fewer than v3 on the same, byte-identical right route). 23500 restores
    # v3's siphon triangle count: siphon-wall p99 deviation from the 60k mesh
    # 0.14-0.16 mm against 0.16-0.18 at 20000, with no measurable SOFA
    # step-time cost (5 anatomies, 150 steps each).
    ap.add_argument("--obj-tris", type=int, default=23500, help="SOFA collision mesh budget")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--dump-iso", default=None, help="also save the undecimated union here")
    ap.add_argument("--recut-obj", action="store_true",
                    help="only re-cut the collision mesh from each anatomy's existing full mesh")
    a = ap.parse_args()
    folders = sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))
    if a.only:
        want = set(a.only.split(","))
        folders = [f for f in folders if os.path.basename(os.path.dirname(f)) in want]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        folders = folders[i::n]
    print("baking %d v4 meshes at %.2f mm (full %d / obj %d tris)\n" % (len(folders), a.spacing, a.full_tris, a.obj_tris))
    print("%-44s %6s %5s %5s %6s %8s %8s %6s %5s %5s" % ("anatomy", "tris", "comps", "genus", "margin",
                                                       "RCCA_lum", "LCCA_lum", "outside", "pass", "sec"))
    bad = 0
    for f in folders:
        root = os.path.dirname(f)
        name = os.path.basename(root)
        if os.path.exists(os.path.join(root, REPORT)) and not a.force and not a.recut_obj:
            print("%-44s kept" % name[:44]); continue
        try:
            if a.recut_obj:
                rep = recut_obj(root, a.obj_tris)
            else:
                rep = bake_anatomy(root, a.spacing, a.full_tris, a.obj_tris, a.verbose, a.dump_iso)
        except Exception as e:                                           # noqa: BLE001
            bad += 1
            import traceback
            print("%-44s FAILED %s: %s" % (name[:44], type(e).__name__, str(e)[:90]), flush=True)
            traceback.print_exc()
            continue
        o = rep["obj"]
        g = max(c["genus"] for c in o["genus"]) if o["genus"] else -1
        print("%-44s %6d %5d %5.0f %6s %8.2f %8.2f %6d %5s %5.0f" % (
            name[:44], o["tris"], o["comps"], g, rep["margin_mm"],
            rep["routes"]["RCCA"].get("lumen_min_mm", -1), rep["routes"]["LCCA"].get("lumen_min_mm", -1),
            rep["centerline_pts_outside"], "yes" if rep["pass"] else "NO", rep["seconds"]), flush=True)
    print("\ndone, %d failed" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
