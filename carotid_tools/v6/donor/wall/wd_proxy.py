#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\wd_proxy.py (= /sp/v6/wall/wd_proxy.py in the build containers)
"""wd_proxy.py (v6_wall gate 3): the donor VA's WALL DETAIL as the bake would make it, per donor side, before the bake.

The bake unions the donor's real surface with the centreline tube (topbrain_tools: tube_field + add_real_sections:
f = max(tube, min(real, capsule)), capsule 1.8 r + 1 mm tapered over 8 mm at both seam ends) on a 0.45 mm grid, the tube
radius being the donor MISR floored at 1.0 mm (graft VA_MIN_R). Where the MISR is below the floor the tube swallows the
real wall: a plain tube, no relief. Here the same union is built for the donor side ALONE in its native frame (the
centreline prepped exactly as graft_v5_reg.load_side: prep_siphon at 1 mm), then measured as v5's wall measure:
dep = | |x - c*| - r* | with c* the exact owner (argmax r - |x - c| over all 0.25 mm samples), share of wall area with
dep > 0.3 mm, in 1 mm bins of the owner's arclength. The per-VA donor stretch [ostium advance + 2, splice - 25] (v5's
donor_real, seam zone out) is aggregated later (wd_collect.py), so before / after compare on the same used ranges.

Inputs per variant:
  v5        donors_v5/<unit>/<side>.json centreline + MISR, its surface;
  <variant> /v6w/seg/<variant>/<unit>/<rva|lva>_stageA.vtp + _stageA_vmtk.json (oriented ostium -> distal);
  floorX    (option --floor-r) a different floor radius, e.g. 1.5, or "ct" = the bone_scan CT read along the line
            (variant F: floor the graft radius at the CT calibre instead of re-segmenting).
    python3 wd_proxy.py <variant> <unit> <side> [--floor 1.0|ct] [--tag NAME]   (neve-build-meshlab)
        -> /v6w/wd/<tag or variant>/<unit>_<side>.json
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import vtk
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from graft_siphon import prep_siphon  # noqa: E402
from sdf_mesher import SPACING_MM, iso_surface, tube_field  # noqa: E402
from sdf_union import Section, add_real_sections, load_surface  # noqa: E402

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
VA_MIN_R = 1.0
SAMPLE = 0.25


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def load(variant, unit, side):
    rec = json.load(open(os.path.join(R5, unit, "%s.json" % side)))
    if variant == "subclavian":
        # v6 (e): the requested subclavian stretch (its own surface + VMTK line), same measure
        v = "rsub" if side == "R" else "lsub"
        d = "/v6w/subclavian/%s" % unit
        vj = json.load(open(os.path.join(d, "%s_stageA_vmtk.json" % v)))
        return rec, np.asarray(vj["points"], float), np.asarray(vj["radii"], float), os.path.join(d, "%s_stageA.vtp" % v)
    if variant == "v5":
        P = np.asarray(rec["centerline"]["points"], float)
        R = np.asarray(rec["centerline"]["radii_misr"], float)
        surf = rec["surface"]["path"]
    else:
        v = "rva" if side == "R" else "lva"
        d = "/v6w/seg/%s/%s" % (variant, unit)
        vj = json.load(open(os.path.join(d, "%s_stageA_vmtk.json" % v)))
        P, R = np.asarray(vj["points"], float), np.asarray(vj["radii"], float)
        surf = os.path.join(d, "%s_stageA.vtp" % v)
        # ostium -> distal: the end nearer v5's ostium first
        o5 = np.asarray(rec["centerline"]["points"][0], float)
        if np.linalg.norm(P[-1] - o5) < np.linalg.norm(P[0] - o5):
            P, R = P[::-1].copy(), R[::-1].copy()
    return rec, P, R, surf


def ct_floor(rec, p):
    """bone_scan.py's CT read every 1 mm along the prepped line (variant F)."""
    import nibabel as nib
    sys.path.insert(0, "/sp/v5_va/tbo/va_narrow")
    import bone_scan as BS
    import ct_check as C
    ctp = BS.unit_ct(rec)
    im = nib.load(ctp)
    ct = C.Vol(im.dataobj, im.affine, p.min(0), p.max(0), pad=10.0)
    out = np.zeros(len(p))
    for j in range(len(p)):
        a, b = max(j - 2, 0), min(j + 2, len(p) - 1)
        t = C.unit(p[b] - p[a])
        out[j] = BS.ct_req(ct, p[j], t)["ct_req"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("unit")
    ap.add_argument("side")
    ap.add_argument("--floor", default="1.0")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or a.variant
    od = "/v6w/wd/%s" % tag
    os.makedirs(od, exist_ok=True)
    of = os.path.join(od, "%s_%s.json" % (a.unit, a.side))
    if os.path.exists(of):
        print("exists", of)
        return
    rec, P, R, surf_p = load(a.variant, a.unit, a.side)
    p, r = prep_siphon(P, R, 1.0)
    r_misr = r.copy()
    floor = np.full(len(p), VA_MIN_R)
    if a.floor == "ct":
        rc = ct_floor(rec, p)
        # the CT calibre replaces the 1.0 mm floor where the MISR is below 1.0 (the floored stretches only)
        floor = np.where(r_misr < VA_MIN_R, np.maximum(VA_MIN_R, rc), VA_MIN_R)
    else:
        floor[:] = float(a.floor)
    r = np.maximum(r_misr, floor)
    s = arclen(p)
    surf = load_surface(surf_p, [1.0, 1.0, 1.0], np.eye(3), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    br = SimpleNamespace(coordinates=p, radii=r)
    field = tube_field([br], spacing=SPACING_MM)
    add_real_sections(field, [Section("donor", surf, p, r, True, True)], SPACING_MM)
    m = iso_surface(field).triangulate().clean()
    m = m.connectivity("largest").extract_surface().triangulate()
    # exact ownership on the 0.25 mm resample of the tube
    n = max(int(np.ceil(s[-1] / SAMPLE)), 1)
    q = np.linspace(0.0, s[-1], n + 1)
    c = np.stack([np.interp(q, s, p[:, k]) for k in range(3)], 1)
    rq = np.interp(q, s, r)
    F = np.asarray(m.faces).reshape(-1, 4)[:, 1:]
    V = np.asarray(m.points, float)
    X = V[F].mean(1)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    tree = cKDTree(c)
    best = np.full(len(X), -np.inf)
    own = np.zeros(len(X), int)
    # exact: candidates within (max r + max depth) of every centroid -- query the k nearest in growing k until the
    # best cannot change (r - |x - c| <= rmax - d_k)
    rmax = float(rq.max())
    k = 32
    while True:
        dd, ii = tree.query(X, k=min(k, len(c)))
        val = rq[ii] - dd
        j = np.argmax(val, axis=1)
        best = val[np.arange(len(X)), j]
        own = ii[np.arange(len(X)), j]
        ok = (best >= rmax - dd[:, -1]) | (k >= len(c))
        if ok.all():
            break
        k *= 2
    dep = np.abs(np.linalg.norm(X - c[own], axis=1) - rq[own])
    so = q[own]
    nb = int(np.ceil(s[-1])) + 1
    bi = np.minimum(so.astype(int), nb - 1)
    area = np.bincount(bi, weights=A, minlength=nb)
    a03 = np.bincount(bi, weights=A * (dep > 0.3), minlength=nb)
    a05 = np.bincount(bi, weights=A * (dep > 0.5), minlength=nb)
    fl = np.bincount(np.minimum(s.astype(int), nb - 1), weights=(r_misr < floor).astype(float), minlength=nb)
    cnt = np.bincount(np.minimum(s.astype(int), nb - 1), minlength=nb)
    # lumen on the polyline (0.05 mm) to this undecimated union: exact point-to-triangle distance, min per 1 mm bin
    loc = vtk.vtkStaticCellLocator()
    loc.SetDataSet(m)
    loc.BuildLocator()
    tq = np.arange(0.0, s[-1], 0.05)
    Pq = np.stack([np.interp(tq, s, p[:, k]) for k in range(3)], 1)
    cp_, cid_, sid_, d2_ = [0.0, 0.0, 0.0], vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    dq = np.empty(len(Pq))
    for i_, x_ in enumerate(Pq):
        loc.FindClosestPoint([float(x_[0]), float(x_[1]), float(x_[2])], cp_, cid_, sid_, d2_)
        dq[i_] = float(d2_) ** 0.5
    lum = np.full(nb, np.nan)
    bq = np.minimum(tq.astype(int), nb - 1)
    for b_ in np.unique(bq):
        lum[b_] = dq[bq == b_].min()
    out = {"variant": a.variant, "tag": tag, "unit": a.unit, "side": a.side, "floor": a.floor, "surface": surf_p,
           "length_mm": float(s[-1]), "union_tris": int(len(F)), "union_genus_euler": None,
           "frac_0.3_whole": float(a03.sum() / area.sum()),
           "bins_1mm": {"area": np.round(area, 3).tolist(), "a03": np.round(a03, 3).tolist(),
                        "a05": np.round(a05, 3).tolist(),
                        "floored_frac": np.round(fl / np.maximum(cnt, 1), 3).tolist(),
                        "lumen_min_union": [None if not np.isfinite(v_) else round(float(v_), 3) for v_ in lum],
                        "r_used_min": [round(float(r[(s >= b_) & (s < b_ + 1)].min()), 3) if ((s >= b_) & (s < b_ + 1)).any()
                                       else None for b_ in range(nb)]},
           "misr_median": float(np.median(r_misr)), "floored_mm": float((r_misr < floor).sum()),
           "r_used_median": float(np.median(r))}
    json.dump(out, open(of, "w"))
    print("%s %s %s %s: whole %.3f, floored %.0f mm, misr med %.2f, tris %d" % (
        tag, a.unit, a.side, a.floor, out["frac_0.3_whole"], out["floored_mm"], out["misr_median"], len(F)), flush=True)


if __name__ == "__main__":
    main()
