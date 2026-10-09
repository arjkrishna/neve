#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\local_sub.py (= /sp/v6/wall_local/local_sub.py in the build containers)
"""local_sub.py (s2_harden_wall_bake, D15 for item e): the LOCAL donor subclavian + junction walls.

Stage 1 made the subclavian stretches and junctions with the plateau rule only (/v6w/subclavian_j3/<unit>, published
to /v6w/subclavian/<unit>). Here the same request run is repeated with V6_RULE=v5 (/v6w/subclavian_v5rule/<unit>,
jfix/sub_batch2.py v5, every other switch identical) and the two final masks are merged exactly as local_seg.py does
for the VAs: w = 1 in the artefact zones, linear blend 2 mm, 0 elsewhere (w = 0 -> the v5-rule mask exactly).
  zones: VA part     = the VA's D15 zones of local_seg (/v6w/seg/local/<unit>/zones.json, v5 VA stations);
         subclavian  = >= 2 bright rays with T_loc > I_c on the v5-rule subclavian stations, or the plateau mask adding
                       lumen >= 0.6 mm deep (runs >= 0.75 mm); no floor (the graft floor is a VA rule).
  masks merged: <x>sub_clean and <x>junction (junction_labels_kji.npz); surfaces = largest component, marching cubes +
  write_vtp's smoothing (ct_seg4), then every vertex outside the zones takes the v5-rule surface's own position
  (same marching-cubes vertex) -> outside the zones the surfaces ARE the v5-rule surfaces.
Outputs /v6w/subclavian_local/<unit>/: junction_labels_kji.npz, <x>sub_stageA.vtp, <x>junction_stageA.vtp,
<x>sub_seeds.json, seg_qc.json (the plateau run's, + "local"), sub_zones.json, local_qc.json, .junction_v3 (the
marker sc_post3 needs: content says local), .seg_done.
    python3 local_sub.py <unit,...>|all|rev        (neve-build-stagea)
"""
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v6/wall_local")
import local_seg as LS  # noqa: E402

VERSION = "local_sub 1.1 (2026-10-04, s2_harden_wall_bake: D15 zone-local subclavian / junction; 1.1: run roots by env (D17 re-run), D16 ostial tube bridge)"
PL = os.environ.get("LOCAL_SUB_PL", "/v6w/subclavian_j3")
V5R = os.environ.get("LOCAL_SUB_V5R", "/v6w/subclavian_v5rule")
VALOC = "/v6w/seg/local"
OUT = os.environ.get("LOCAL_SUB_OUT", "/v6w/subclavian_local")
SIDE = {"R": ("r", "rsub", "rva", 3), "L": ("l", "lsub", "lva", 4)}   # x, sub name, VA name, sub label index in ALL
# D16 (user, 2026-10-04): the two pinched donor VA ostia (bottleneck < 0.5 of the VA's inscribed radius on the local
# junction: ia172_071 L 0.29, ia172_164 L 0.46) get a short tube at the VA calibre; empty -> off
D16_SIDES = set(x for x in os.environ.get("LOCAL_SUB_D16", "ia172_164:L,ia172_071:L").split(",") if x)
D16_TARGET = 0.7        # stage 1's "good" bottleneck level (45 / 54 sides); the VA tool fails a junction below 0.5
D16_S0, D16_STEP, D16_SMAX = 6.0, 2.0, 16.0     # tube along the VA from s 0 to S (junction6 BRIDGE_S first)
D16_BLEND = 2.0         # the tube region + 2 mm counts as a zone for the surface splice
MERGE_PAD_MM = 12.0     # 1.1: the D15 signed-distance merge is computed in the box of w > 0 grown by this much
MERGE_FULL_CHECK = int(os.environ.get("LOCAL_SUB_FULL_CHECK", "40000000"))   # grids up to this size: also merged in full + compared
MERGE_LOG = []


def _seg_dist(X, A):
    best = np.full(len(X), np.inf)
    for a, b in zip(A[:-1], A[1:]):
        ab = b - a
        t = np.clip(((X - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(X - (a + t[:, None] * ab), axis=1))
    return best


def d16_tube(mj, W, jb5, Pva, sva, org, sp_, shape):
    """D16: union a tube at the VA calibre r_t = max(VA edge radius, station radius) over its first 6 mm (= junction6's
    fallback-tube radius, rb - pad) around the axis [Q, P_va(s 0..S)] into the junction mask; S grows from 6 mm (the
    fallback's own extent) in 2 mm steps until the bottleneck ratio (junction6's measure, Q -> VA s 10) reaches 0.7,
    at most 16 mm. Returns (mask, W with the tube zone, record)."""
    sys.path.insert(0, "/sp/v6/wall/jfix")
    import junction6 as J6
    spk = (sp_[2], sp_[1], sp_[0])
    Q = np.asarray(jb5["Q_xyz"], float)
    r_t = float(jb5["rb_mm"]) - float(jb5["pad_mm"])
    j10 = int(min(np.searchsorted(sva, 10.0), len(sva) - 1))

    def kji(p):
        q = np.clip(np.round((p - org) / sp_).astype(int), 0, np.array(shape[::-1]) - 1)
        return np.array([q[2], q[1], q[0]])
    qa, qb = kji(Q), kji(Pva[j10])

    def ratio(m):
        pts = np.vstack([Q, Pva[:j10 + 1]])
        lo_c = np.maximum(np.floor((pts.min(0) - 12.0 - org) / sp_).astype(int), 0)
        hi_c = np.minimum(np.ceil((pts.max(0) + 12.0 - org) / sp_).astype(int), np.array(shape[::-1]) - 1)
        slc = (slice(lo_c[2], hi_c[2] + 1), slice(lo_c[1], hi_c[1] + 1), slice(lo_c[0], hi_c[0] + 1))
        off = np.array([lo_c[2], lo_c[1], lo_c[0]])
        mc = largest(m)[0][slc]
        rb_ = J6._bottleneck(mc, spk, qa - off, qb - off, 10.0)
        rv = float(ndimage.distance_transform_edt(mc, sampling=spk)[tuple(qb - off)])
        return (rb_ / rv if rv > 0 else 0.0), rb_, rv
    r0 = ratio(mj)
    out = {"rule": "D16 tube at the VA calibre along [Q, VA s 0..S] (junction6 fallback-tube region, lengthened)",
           "r_mm": round(r_t, 3), "ratio_before": round(r0[0], 3), "bottleneck_before_mm": round(r0[1], 3), "steps": []}
    S = D16_S0
    while True:
        jv = int(max(1, np.searchsorted(sva, S)))
        A = np.vstack([Q, Pva[:jv + 1]])
        lo_x, hi_x = A.min(0) - r_t - D16_BLEND - 1.0, A.max(0) + r_t + D16_BLEND + 1.0
        i0 = np.maximum(np.floor((lo_x - org) / sp_).astype(int), 0)
        i1 = np.minimum(np.ceil((hi_x - org) / sp_).astype(int), np.array(shape[::-1]) - 1)
        kk, jj, ii = np.meshgrid(np.arange(i0[2], i1[2] + 1), np.arange(i0[1], i1[1] + 1), np.arange(i0[0], i1[0] + 1), indexing="ij")
        d = _seg_dist(org + np.stack([ii, jj, kk], -1).reshape(-1, 3) * sp_, A).reshape(kk.shape)
        sl = (slice(i0[2], i1[2] + 1), slice(i0[1], i1[1] + 1), slice(i0[0], i1[0] + 1))
        tube = np.zeros(shape, bool)
        tube[sl] = d <= r_t
        m = ndimage.binary_fill_holes(largest(mj | tube)[0])
        rr = ratio(m)
        out["steps"].append({"S_mm": S, "ratio": round(rr[0], 3)})
        if rr[0] >= D16_TARGET or S >= D16_SMAX:
            break
        S += D16_STEP
    wt = np.zeros(shape, np.float32)
    wt[sl] = np.clip(1.0 - (d - r_t) / D16_BLEND, 0.0, 1.0)
    W2 = np.maximum(W, wt)
    out.update(from_xyz=[round(float(v), 3) for v in Q], to_xyz=[round(float(v), 3) for v in Pva[min(jv, len(sva) - 1)]],
               frame="donor RAS mm (the side record's frame, unmirrored)",
               S_mm=S, tubed_mm=round(float(np.linalg.norm(Q - Pva[0]) + sva[min(jv, len(sva) - 1)]), 2),
               ratio_after=round(rr[0], 3), bottleneck_after_mm=round(rr[1], 3), va_inscribed_r_s10_mm=round(rr[2], 3),
               voxels_added=int((m & ~mj).sum()), passes_0_5=bool(rr[0] >= 0.5))
    return m, W2, out


def common(a_org, b_org, sp, a_shape, b_shape):
    """integer offsets of grids a, b (kji shapes, xyz origins) inside their union grid."""
    oa = np.round((a_org - b_org) / sp).astype(int)   # a relative to b, xyz voxels
    assert np.allclose(a_org, b_org + oa * sp, atol=1e-3), "grids not commensurate"
    lo = np.minimum(oa, 0)
    hi = np.maximum(oa + np.array(a_shape[::-1]), np.array(b_shape[::-1]))
    shape = tuple((hi - lo)[::-1])
    org = b_org + lo * sp
    return org, shape, (oa - lo)[::-1], (-lo)[::-1]


def place(arr, shape, off):
    out = np.zeros(shape, arr.dtype)
    out[off[0]:off[0] + arr.shape[0], off[1]:off[1] + arr.shape[1], off[2]:off[2] + arr.shape[2]] = arr
    return out


def sub_zone_weights(st5r, mp_sub, m5_sub, sp_, org):
    P = np.asarray(st5r["P"], float)
    s = np.asarray(st5r["s"], float)
    Tl, Ic, kind = np.asarray(st5r["T_loc"], float), np.asarray(st5r["Ic"], float), np.asarray(st5r["kind"])
    bright_cut = ((kind == 3) & (Tl > Ic[:, None])).sum(1) >= 2
    spk = (sp_[2], sp_[1], sp_[0])
    # 1.1: the depth EDT in the box of the two masks grown by 15 mm (1.0: the full grid; same booleans -- a crop can only
    # lengthen distances beyond 15 mm, and only depth >= DEPTH_MM (0.6 mm) is used)
    nzb = np.nonzero(mp_sub | m5_sub)
    add = mp_sub & ~m5_sub
    kk, jj, ii = np.nonzero(add)
    dep = np.zeros(len(P))
    if len(kk):
        pad = np.ceil(15.0 / np.array(spk)).astype(int)
        lo_b = np.maximum(np.array([a.min() for a in nzb]) - pad, 0)
        hi_b = np.minimum(np.array([a.max() for a in nzb]) + pad + 1, np.array(m5_sub.shape))
        bx = tuple(slice(int(a), int(b)) for a, b in zip(lo_b, hi_b))
        d_out5 = ndimage.distance_transform_edt(~m5_sub[bx], sampling=spk)
        X = org + np.c_[ii, jj, kk] * sp_
        dd, ix = cKDTree(P).query(X, distance_upper_bound=12.0)
        ok = np.isfinite(dd)
        np.maximum.at(dep, ix[ok], d_out5[kk[ok] - lo_b[0], jj[ok] - lo_b[1], ii[ok] - lo_b[2]])
    deep = dep >= LS.DEPTH_MM
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 0.25
    nrun = max(1, int(round(LS.RUN_MM / ds)))
    deep_r = np.zeros_like(deep)
    for a, b in LS.runs(deep, np.arange(len(deep))):
        if b - a + 1 >= nrun:
            deep_r[int(a):int(b) + 1] = True
    core = bright_cut | deep_r
    g = int(round(LS.GAP_MM / ds))
    rr = LS.runs(core, np.arange(len(core)))
    for (a0, b0), (a1, b1) in zip(rr[:-1], rr[1:]):
        if a1 - b0 - 1 < g:
            core[int(b0):int(a1) + 1] = True
    w = LS.station_weights(s, core)
    return w, {"core_runs_s_mm": LS.runs(core, s), "core_mm": float(core.sum() * ds), "length_mm": float(s[-1]),
               "w_gt0_mm": float((w > 0).sum() * ds),
               "detectors_mm": {"bright_cut": float(bright_cut.sum() * ds), "plateau_adds_ge_0.6mm": float(deep_r.sum() * ds)}}


def largest(m):
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1)
    return m, int(n)


def surf(m_loc, m_v5r, W, sp_, org, file_v5r, fo):
    """write the local surface: LS.mc_smooth + outside-zone vertex positions from the v5-rule recompute."""
    import pyvista as pv
    from vtk.util import numpy_support as ns
    P5f = np.asarray(pv.read(file_v5r).points, float)
    pre5, sm5 = LS.mc_smooth(m_v5r, sp_, org)
    P5s, F5s = LS.poly_arrays(sm5)
    repro = bool(len(P5s) == len(P5f) and np.allclose(P5s, P5f, atol=1e-9))
    v5map = dict(zip(LS.keys_of(pre5), range(len(pre5))))
    prel, sml = LS.mc_smooth(m_loc, sp_, org)
    Pls, Fl = LS.poly_arrays(sml)
    ijk = (prel - org) / sp_
    lo = np.floor(ijk + 1e-6).astype(int)
    wmax = np.zeros(len(prel))
    shp = W.shape
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                wmax = np.maximum(wmax, W[np.clip(lo[:, 2] + dk, 0, shp[0] - 1), np.clip(lo[:, 1] + dj, 0, shp[1] - 1),
                                          np.clip(lo[:, 0] + di, 0, shp[2] - 1)])
    j5 = np.array([v5map.get(x, -1) for x in LS.keys_of(prel)])
    outside = wmax == 0
    take = outside & (j5 >= 0)
    newP = Pls.copy()
    newP[take] = P5s[j5[take]]
    sml.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(newP), deep=1))
    LS.write_poly(sml, fo)
    id5 = {x: i for i, x in enumerate(LS.keys_of(P5f))}
    out_ok = int(sum(1 for x in LS.keys_of(newP[outside]) if x in id5))
    T5 = set(tuple(sorted(t)) for t in F5s.tolist())
    KP = np.array([id5.get(x, -1) for x in LS.keys_of(newP)])
    tri_out = outside[Fl].all(1)
    tri_same = int(sum(1 for t in KP[Fl[tri_out]].tolist() if min(t) >= 0 and tuple(sorted(t)) in T5))
    disp = np.linalg.norm(newP - Pls, axis=1)
    return {"v5rule_recompute_equals_file": repro, "vertices": int(len(newP)), "vertices_outside_zones": int(outside.sum()),
            "outside_vertices_equal_v5rule_file": out_ok, "triangles_outside": int(tri_out.sum()),
            "triangles_outside_equal_v5rule": tri_same,
            "spill_undone_max_mm": float(disp[take].max()) if take.any() else 0.0,
            "topology": LS.topo(newP, Fl), "md5": LS.md5(fo)}


def unit_run(unit):
    od = os.path.join(OUT, unit)
    if os.path.exists(os.path.join(od, ".seg_done")):
        return
    pdir, vdir = os.path.join(PL, unit), os.path.join(V5R, unit)
    for d in (pdir, vdir):
        if not os.path.exists(os.path.join(d, ".seg_done")):
            print("SKIP %s: no run in %s" % (unit, d), flush=True)
            return
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    jp = np.load(os.path.join(pdir, "junction_labels_kji.npz"), allow_pickle=True)
    jv = np.load(os.path.join(vdir, "junction_labels_kji.npz"), allow_pickle=True)
    sp_ = np.asarray(jv["spacing_mm"], float)
    assert np.allclose(jp["spacing_mm"], sp_)
    org_p, org_v = np.asarray(jp["origin_mm"], float), np.asarray(jv["origin_mm"], float)
    keys = [k for k in jv.files if k.endswith("junction") or k.endswith("_clean")]
    shp_p = jp[keys[0]].shape
    shp_v = jv[keys[0]].shape
    org, shape, off_p, off_v = common(org_p, org_v, sp_, shp_p, shp_v)
    st_v = json.load(open(os.path.join(vdir, "stations_rays.json")))
    st_p = json.load(open(os.path.join(pdir, "stations_rays.json")))
    va_st = json.load(open(os.path.join(LS.D[unit]["v5_seg_dir"], "stations_rays.json")))
    va_z = json.load(open(os.path.join(VALOC, unit, "zones.json")))
    qc = {"script": VERSION, "unit": unit, "plateau_run": pdir, "v5rule_run": vdir, "grid": {
        "origin_mm": org, "shape_kji": shape, "offset_plateau_kji": off_p, "offset_v5rule_kji": off_v}, "sides": {}}
    zones_out = {}
    masks_out = {}
    for sd in ("R", "L"):
        x, sn, vn, _ = SIDE[sd]
        if "%sjunction" % x not in jv.files or "%sjunction" % x not in jp.files:
            continue
        msub_v = place(jv["%s_clean" % sn].astype(bool), shape, off_v)
        msub_p = place(jp["%s_clean" % sn].astype(bool), shape, off_p)
        mj_v = place(jv["%sjunction" % x].astype(bool), shape, off_v)
        mj_p = place(jp["%sjunction" % x].astype(bool), shape, off_p)
        # station check: the request runs' VA stations equal the VA-only v5 stations (same seeds, same rule)
        dva = float(np.abs(np.asarray(st_v[vn]["P"], float) - np.asarray(va_st[vn]["P"], float)).max()) \
            if len(st_v[vn]["P"]) == len(va_st[vn]["P"]) else None
        w_sub, zsub = sub_zone_weights(st_v[sn], msub_p, msub_v, sp_, org)
        zv = va_z.get(sd)
        Pva = np.asarray(va_st[vn]["P"], float)
        sva = np.asarray(va_st[vn]["s"], float)
        w_va = np.asarray(zv["w_station"], float) if zv else np.zeros(len(Pva))
        keep = sva <= 40.0
        Pall = np.vstack([np.asarray(st_v[sn]["P"], float), Pva[keep]])
        wall = np.r_[w_sub, w_va[keep]]
        region = ndimage.binary_dilation(mj_v | mj_p | msub_v | msub_p, iterations=3)
        kk, jj, ii = np.nonzero(region)
        X = org + np.c_[ii, jj, kk] * sp_
        dd, ix = cKDTree(Pall).query(X, distance_upper_bound=15.0)
        ok = np.isfinite(dd)
        W = np.zeros(shape, np.float32)
        W[kk[ok], jj[ok], ii[ok]] = wall[ix[ok]]
        spk = (sp_[2], sp_[1], sp_[0])

        def merge_full(m5, mp):
            phi = (1.0 - W) * (ndimage.distance_transform_edt(~m5, sampling=spk) - ndimage.distance_transform_edt(m5, sampling=spk)) + \
                W * (ndimage.distance_transform_edt(~mp, sampling=spk) - ndimage.distance_transform_edt(mp, sampling=spk))
            return phi < 0

        def merge(m5, mp):
            if not (W > 0).any():
                return m5.copy()
            # 1.1: the signed-distance blend only inside the box of w > 0 grown by MERGE_PAD_MM (outside it w = 0 and the
            # result is m5 by construction); inside, phi's sign can only differ from m5's / mp's common sign between the
            # two surfaces, where both distances are short -> the crop is exact (D17's longer stretches made the full-grid
            # EDTs too large for 8 GB). Grids <= MERGE_FULL_CHECK voxels are also merged in full and compared.
            nz = np.nonzero(W > 0)
            pad = np.ceil(MERGE_PAD_MM / np.array(spk)).astype(int)
            lo = np.maximum(np.array([a.min() for a in nz]) - pad, 0)
            hi = np.minimum(np.array([a.max() for a in nz]) + pad + 1, np.array(shape))
            bx = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            Wb, m5b, mpb = W[bx], m5[bx], mp[bx]
            phi = (1.0 - Wb) * (ndimage.distance_transform_edt(~m5b, sampling=spk) - ndimage.distance_transform_edt(m5b, sampling=spk)) + \
                Wb * (ndimage.distance_transform_edt(~mpb, sampling=spk) - ndimage.distance_transform_edt(mpb, sampling=spk))
            ml = m5.copy()
            ml[bx] = phi < 0
            assert np.array_equal(ml[W == 0], m5[W == 0])
            if int(np.prod(shape)) <= MERGE_FULL_CHECK:
                assert np.array_equal(ml, merge_full(m5, mp)), "cropped merge differs from the full-grid merge"
                MERGE_LOG.append("full-grid check equal")
            else:
                MERGE_LOG.append("box only (grid %d voxels)" % int(np.prod(shape)))
            return ml
        msub_l = merge(msub_v, msub_p)
        mj_l = merge(mj_v, mj_p)
        masks_out["%s_clean" % sn] = msub_l
        masks_out["%sjunction" % x] = mj_l
        rec = {"va_stations_equal_va_only_run_max_mm": dva, "sub_zones": zsub,
               "va_zone_used": bool(zv is not None and np.asarray(zv["w_station"]).max() > 0),
               "voxels": {"junction_v5rule": int(mj_v.sum()), "junction_plateau": int(mj_p.sum()), "junction_local": int(mj_l.sum()),
                          "junction_local_ne_v5rule": int((mj_l != mj_v).sum()),
                          "sub_v5rule": int(msub_v.sum()), "sub_plateau": int(msub_p.sum()), "sub_local": int(msub_l.sum())}}
        # 1.1 (user decision D16): a pinched donor VA ostium is bridged by a short tube at the VA calibre (the
        # junction6 fallback-tube region, as on ia172_136 L / ia172_172 L), lengthened along the VA until the
        # bottleneck clears (see d16_tube); the tube region counts as a zone for the surface splice (W_t)
        W_j = W
        if "%s:%s" % (unit, sd) in D16_SIDES:
            jb5_ = (json.load(open(os.path.join(vdir, "seg_qc.json"))).get("v6", {}).get("junction_bridge") or {}).get("%sjunction" % x) or {}
            mj_l, W_j, rec["d16"] = d16_tube(mj_l, W, jb5_, Pva, sva, org, sp_, shape)
            masks_out["%sjunction" % x] = mj_l
            print(unit, sd, "D16", json.dumps(rec["d16"]), flush=True)
        # the junction QC the VA tool reads (sc_post3: bottleneck ratio < 0.5 -> junction_failed), re-measured on the
        # LOCAL mask with junction6's own measure: largest ball radius from the subclavian point Q to the VA at s 10
        try:
            sys.path.insert(0, "/sp/v6/wall/jfix")
            import junction6 as J6
            jb5 = (json.load(open(os.path.join(vdir, "seg_qc.json"))).get("v6", {}).get("junction_bridge") or {}).get("%sjunction" % x) or {}
            Q = np.asarray(jb5["Q_xyz"], float)
            j10 = int(min(np.searchsorted(sva, 10.0), len(sva) - 1))

            def kji(p):
                q = np.clip(np.round((p - org) / sp_).astype(int), 0, np.array(shape[::-1]) - 1)
                return np.array([q[2], q[1], q[0]])
            qa, qb = kji(Q), kji(Pva[j10])
            pts = np.vstack([Q, Pva[:j10 + 1]])
            lo_c = np.maximum(np.floor((pts.min(0) - 12.0 - org) / sp_).astype(int), 0)
            hi_c = np.minimum(np.ceil((pts.max(0) + 12.0 - org) / sp_).astype(int), np.array(shape[::-1]) - 1)
            slc = (slice(lo_c[2], hi_c[2] + 1), slice(lo_c[1], hi_c[1] + 1), slice(lo_c[0], hi_c[0] + 1))
            off = np.array([lo_c[2], lo_c[1], lo_c[0]])
            mc = largest(mj_l)[0][slc]
            rb_ = J6._bottleneck(mc, spk, qa - off, qb - off, 10.0)
            rv = float(ndimage.distance_transform_edt(mc, sampling=spk)[tuple(qb - off)])
            jl_ = dict(jb5)
            jl_["after"] = dict(jb5.get("after") or {}, bottleneck_r_mm=round(rb_, 3), va_inscribed_r_s10_mm=round(rv, 3),
                                bottleneck_ratio=round(rb_ / rv, 3) if rv > 0 else None)
            jl_["local"] = {"measured_on": "local junction mask (largest component)", "v5rule_after": jb5.get("after"),
                            "w_at_Q": float(W[tuple(qa)])}
            if rec.get("d16"):
                # D16: a tube fill was used (same meaning as junction6's fallback_tube; the VA tool flags it)
                jl_["after"].update(fallback_tube=True, d16_tube=True, d16_tubed_mm=rec["d16"]["tubed_mm"],
                                    d16_tube_r_mm=rec["d16"]["r_mm"])
                jl_["d16"] = rec["d16"]
            rec["junction_bridge_local"] = jl_
        except Exception as e:  # noqa: BLE001
            rec["junction_bridge_local"] = {"error": "%s: %s" % (type(e).__name__, e)}
        for tag, ml, m5, f5, Wt in (("%s" % sn, msub_l, msub_v, "%s_stageA.vtp" % sn, W),
                                    ("%sjunction" % x, mj_l, mj_v, "%sjunction_stageA.vtp" % x, W_j)):
            ml_c, ncl = largest(ml)
            m5_c, nc5 = largest(m5)
            Wc = Wt.copy()
            # a component dropped / kept differently near a zone stays inside the zone's own surface: if the
            # largest-component choice changed something where w == 0, the outside is not v5's -> record it
            diff_out = int(((ml_c != m5_c) & (Wt == 0)).sum())
            rec[tag] = surf(ml_c, m5_c, Wc, sp_, org, os.path.join(vdir, f5), os.path.join(od, f5))
            rec[tag].update(components_before_largest=ncl, components_v5rule=nc5, largest_choice_differs_outside_vox=diff_out)
        # seeds of the subclavian: the v5-rule run's (label ends) when the label ends are unchanged
        shutil.copy2(os.path.join(vdir, "%s_seeds.json" % sn), os.path.join(od, "%s_seeds.json" % sn))
        rec["seeds"] = "v5-rule run's %s_seeds.json" % sn
        zones_out[sd] = {"sub": zsub, "w_sub_station": np.round(w_sub, 3)}
        qc["sides"][sd] = rec
        print(unit, sd, json.dumps({t: {k: rec[t][k] for k in ("v5rule_recompute_equals_file", "vertices_outside_zones",
                                                                "outside_vertices_equal_v5rule_file", "triangles_outside",
                                                                "triangles_outside_equal_v5rule", "spill_undone_max_mm")}
                                    for t in (sn, "%sjunction" % x)}), "sub core %.1f mm" % zsub["core_mm"], flush=True)
    np.savez_compressed(os.path.join(od, "junction_labels_kji.npz"), origin_mm=org, spacing_mm=sp_,
                        note="local_sub (D15): v5-rule masks with the plateau rule in the artefact zones; bool [k,j,i]",
                        **masks_out)
    sq = json.load(open(os.path.join(pdir, "seg_qc.json")))
    sq.setdefault("v6", {})["local"] = {"script": VERSION, "v5rule_run": vdir, "plateau_run": pdir,
                                        "note": "v6.junction_bridge.<x>junction = the v5-rule run's bridge record with after.bottleneck_* re-measured on the LOCAL junction mask; the two runs' own records in v6_v5rule_junction_bridge / v6_plateau_junction_bridge"}
    sq["v6_v5rule_junction_bridge"] = json.load(open(os.path.join(vdir, "seg_qc.json"))).get("v6", {}).get("junction_bridge")
    sq["v6_plateau_junction_bridge"] = (sq.get("v6") or {}).get("junction_bridge")
    jbl = {("%sjunction" % SIDE[sd][0]): r["junction_bridge_local"] for sd, r in qc["sides"].items()
           if "error" not in (r.get("junction_bridge_local") or {"error": 1})}
    sq.setdefault("v6", {})["junction_bridge"] = dict((sq["v6"].get("junction_bridge") or {}), **jbl)
    LS.jdump(sq, os.path.join(od, "seg_qc.json"))
    for f in ("subclavian_stations.json",):
        if os.path.exists(os.path.join(vdir, f)):
            shutil.copy2(os.path.join(vdir, f), os.path.join(od, f))
    LS.jdump(zones_out, os.path.join(od, "sub_zones.json"))
    qc["merge"] = list(MERGE_LOG)
    del MERGE_LOG[:]
    qc["seconds"] = round(time.time() - t0, 1)
    LS.jdump(qc, os.path.join(od, "local_qc.json"))
    open(os.path.join(od, ".junction_v3"), "w").write("local (D15) junction: %s from v5-rule %s + plateau %s, %s\n" % (
        VERSION, vdir, pdir, time.strftime("%F %T")))
    open(os.path.join(od, ".seg_done"), "w").write("%s %s\n" % (VERSION, time.strftime("%F %T")))
    print("DONE", unit, "%.0f s" % qc["seconds"], flush=True)


FAILS = []


def main():
    arg = sys.argv[1]
    units = sorted(os.listdir(V5R)) if arg in ("all", "rev") else arg.split(",")
    if arg == "rev":
        units = units[::-1]
    os.makedirs(os.path.join(OUT, "_claim"), exist_ok=True)
    for u in units:
        if not os.path.isdir(os.path.join(V5R, u)) or os.path.exists(os.path.join(OUT, u, ".seg_done")):
            continue
        claim = os.path.join(OUT, "_claim", u)
        try:
            os.mkdir(claim)
        except FileExistsError:
            continue
        try:
            unit_run(u)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", u, repr(e), flush=True)
            FAILS.append(u)
        finally:
            if not os.path.exists(os.path.join(OUT, u, ".seg_done")):
                os.rmdir(claim)


if __name__ == "__main__":
    main()
    if FAILS:
        print('UNITS FAILED', FAILS, flush=True)
        sys.exit(1)
