#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\local_seg.py (= /sp/v6/wall_local/local_seg.py in the build containers)
"""local_seg.py (s2_harden_wall_bake, D15): the LOCAL donor VA wall = v5's ct_seg4 wall everywhere EXCEPT in the
artefact zones, where stage 1's plateau rule (V6_RULE=plateau, /v6w/seg/plateau) is taken, with a short blend.

Inputs per donor unit (read-only): v5's seg folder (wall/donors.json v5_seg_dir: labels_kji.npz, stations_rays.json,
<v>_stageA.vtp, <v>_seeds.json), stage 1's plateau seg folder /v6w/seg/plateau/<unit>, v5's donor record line
(donors_v5/<unit>/<R|L>.json: VMTK MISR line).

1. ZONES on v5's VA stations (0.25 mm; arclength s from the VA origin):
   bone  = (a) v5's own bone-side mechanism: >= 2 bright-neighbour rays (kind 3) whose lower threshold T_loc exceeds
               the lumen core I_c (the ray ends at its first sample: v6_wall section 1), OR
           (b) the effect: the plateau label adds lumen >= 0.6 mm deep outside v5's label (EDT), in runs >= 0.75 mm;
   floor = v5's record line MISR < 1.0 (the graft floor acts: floor_scan's definition), mapped to the nearest station;
   core  = bone | floor, gaps < GAP_MM (2 mm) closed; weight w = 1 on the core, linear to 0 over BLEND_MM (2 mm) on
   both sides, 0 elsewhere.
2. LABEL MERGE per VA: signed distance (EDT, mm, anisotropic) of v5's and plateau's final VA labels, phi =
   (1 - w) phi_v5 + w phi_plateau, label = phi < 0, w per voxel from its nearest v5 station. w = 0 gives v5's label
   EXACTLY, w = 1 plateau's. Other labels (BA, subclavians) = v5's.
3. SURFACE: marching cubes + mask_to_surface.write_vtp's clean + windowed sinc (20 it, pass band 0.1) exactly as
   ct_seg4 (checked: the same recompute on v5's labels reproduces v5's stage-A surface point for point); then the
   smoothing spill (20 rings) is undone outside the zones: every vertex whose marching-cubes position lies in the
   w = 0 region (max w over the 2x2x2 voxels around it) takes v5's smoothed position of the SAME marching-cubes vertex
   -> outside the zones the surface is v5's own (same vertices, same triangles), inside it is the re-smoothed merge.
4. SEEDS: v5's seed file where the label's first / last 1.5 mm is unchanged, else ct_seg4's rule (diff 7 centroids).
Outputs /v6w/seg/local/<unit>/: labels_kji.npz, <v>_stageA.vtp, <v>_seeds.json, zones.json, local_qc.json,
seg_qc.json ({"v6": ...} for build_donors6), copies of v5's edge.json / measure.json, .seg_done.
    python3 local_seg.py <unit>[,unit...] | all          (neve-build-stagea)
"""
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

VERSION = "local_seg 1.1 (2026-10-04, s2_harden_wall_bake: D15 zone-local plateau; 1.1 union grid when the runs crop differently, v5 file positions)"
D = json.load(open("/sp/v6/wall/donors.json"))
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
PLAT = "/v6w/seg/plateau"
OUT = os.environ.get("LOCAL_OUT", "/v6w/seg/local")
GAP_MM, BLEND_MM = 2.0, 2.0
DEPTH_MM, RUN_MM = 0.6, 0.75
VK = {"R": (1, "rva"), "L": (2, "lva")}


def jdump(o, p):
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(type(x))
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(o, f, default=conv)
    os.replace(tmp, p)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def runs(m, s):
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            out.append([float(s[i]), float(s[j])])
            i = j + 1
        else:
            i += 1
    return out


def mc_smooth(m, sp, corg):
    """ct_seg4: marching cubes of the label + mask_to_surface.write_vtp (clean, windowed sinc 20 / 0.1, normalize).
    Returns (pre-smoothing cleaned points, smoothed vtkPolyData)."""
    import vtk
    from skimage import measure
    from vtk.util import numpy_support as ns
    vol = np.pad(m.transpose(2, 1, 0).astype(np.uint8), 1)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=tuple(sp))
    V = corg + verts - sp
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(V, dtype=np.float64), deep=1))
    cells = vtk.vtkCellArray()
    tri = np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel()
    cells.SetCells(len(faces), ns.numpy_to_vtkIdTypeArray(tri, deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()
    pre = ns.vtk_to_numpy(clean.GetOutput().GetPoints().GetData()).astype(np.float64).copy()
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(clean.GetOutput())
    smooth.SetNumberOfIterations(20)
    smooth.SetPassBand(0.1)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(smooth.GetOutput())
    return pre, out


def write_poly(poly, path):
    import vtk
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(poly)
    w.Write()


def poly_arrays(poly):
    from vtk.util import numpy_support as ns
    P = ns.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    F = ns.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return P, F


def keys_of(P):
    return [tuple(x) for x in np.round(P, 6)]


def topo(P, F):
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    nV = len(np.unique(F))
    chi = nV - len(cnt) + len(F)
    # components
    import scipy.sparse as sps
    n = int(F.max()) + 1
    A = sps.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    nc, lab = sps.csgraph.connected_components(A, directed=False)
    used = np.unique(F)
    ncomp = len(np.unique(lab[used]))
    return {"euler_chi": int(chi), "components": int(ncomp), "open_or_nonmanifold_edges": int((cnt != 2).sum()),
            "genus_if_one_component": (2 - chi) / 2.0 if ncomp == 1 else None}


def station_weights(s, core):
    """w per station: 1 on the core, linear to 0 over BLEND_MM."""
    if not core.any():
        return np.zeros(len(s))
    sc = s[core]
    # distance (arclength) to the nearest core station
    idx = np.searchsorted(sc, s)
    dl = np.abs(s - sc[np.clip(idx - 1, 0, len(sc) - 1)])
    dr = np.abs(sc[np.clip(idx, 0, len(sc) - 1)] - s)
    d = np.minimum(dl, dr)
    d[core] = 0.0
    return np.clip(1.0 - d / BLEND_MM, 0.0, 1.0)


def one_side(unit, sd, z5, zp, st5, sp_, corg, out_dir, qc):
    k, v = VK[sd]
    st = st5[v]
    P = np.asarray(st["P"], float)
    s = np.asarray(st["s"], float)
    m5 = z5 == k
    mp = zp == k
    spk = (sp_[2], sp_[1], sp_[0])
    # ---- zones
    Tl, Ic, kind = np.asarray(st["T_loc"], float), np.asarray(st["Ic"], float), np.asarray(st["kind"])
    bright_cut = ((kind == 3) & (Tl > Ic[:, None])).sum(1) >= 2
    d_out5 = ndimage.distance_transform_edt(~m5, sampling=spk)
    add = mp & ~m5
    kk, jj, ii = np.nonzero(add)
    tree = cKDTree(P)
    dep = np.zeros(len(P))
    if len(kk):
        X = corg + np.c_[ii, jj, kk] * sp_
        dd, ix = tree.query(X, distance_upper_bound=8.0)
        ok = np.isfinite(dd)
        np.maximum.at(dep, ix[ok], d_out5[kk[ok], jj[ok], ii[ok]])
    deep = dep >= DEPTH_MM
    nrun = int(round(RUN_MM / 0.25))
    deep_r = np.zeros_like(deep)
    for a, b in runs(deep, np.arange(len(deep))):
        if b - a + 1 >= nrun:
            deep_r[int(a):int(b) + 1] = True
    rec = json.load(open(os.path.join(R5, unit, "%s.json" % sd)))
    L = np.asarray(rec["centerline"]["points"], float)
    M = np.asarray(rec["centerline"]["radii_misr"], float)
    _, jl = cKDTree(L).query(P)
    floor = M[jl] < 1.0
    bone = bright_cut | deep_r
    core = bone | floor
    # close gaps < GAP_MM
    g = int(round(GAP_MM / 0.25))
    core_c = core.copy()
    rr = runs(core, np.arange(len(core)))
    for (a0, b0), (a1, b1) in zip(rr[:-1], rr[1:]):
        if a1 - b0 - 1 < g:
            core_c[int(b0):int(a1) + 1] = True
    w_st = station_weights(s, core_c)
    zones = {"core_runs_s_mm": runs(core_c, s), "blend_mm": BLEND_MM, "gap_closed_mm": GAP_MM,
             "core_mm": float(core_c.sum() * 0.25), "weighted_mm": float((w_st > 0).sum() * 0.25),
             "length_mm": float(s[-1]), "core_frac": float(core_c.mean()), "w_gt0_frac": float((w_st > 0).mean()),
             "detectors_mm": {"bright_cut": float(bright_cut.sum() * 0.25), "plateau_adds_ge_0.6mm": float(deep_r.sum() * 0.25),
                              "floor_v5_misr_lt_1": float(floor.sum() * 0.25), "bone_any": float(bone.sum() * 0.25),
                              "floor_without_bone": float((floor & ~bone).sum() * 0.25)},
             "station_s0": float(s[0]), "station_ds": 0.25,
             "w_station": np.round(w_st, 3)}
    # ---- per-voxel weight (nearest v5 station) and label merge
    reach = 8.0
    region = ndimage.binary_dilation(m5 | mp, iterations=3)
    kk, jj, ii = np.nonzero(region)
    X = corg + np.c_[ii, jj, kk] * sp_
    dd, ix = tree.query(X, distance_upper_bound=reach)
    ok = np.isfinite(dd)
    W = np.zeros(m5.shape, np.float32)
    W[kk[ok], jj[ok], ii[ok]] = w_st[ix[ok]]
    if (W > 0).any():
        phi5 = ndimage.distance_transform_edt(~m5, sampling=spk) - ndimage.distance_transform_edt(m5, sampling=spk)
        phip = ndimage.distance_transform_edt(~mp, sampling=spk) - ndimage.distance_transform_edt(mp, sampling=spk)
        phi = (1.0 - W) * phi5 + W * phip
        ml = phi < 0
        del phi5, phip, phi
    else:
        ml = m5.copy()
    assert np.array_equal(ml[W == 0], m5[W == 0]), "w=0 region must be v5's label"
    qc_side = {"zones": {k_: v_ for k_, v_ in zones.items() if k_ != "w_station"},
               "voxels": {"v5": int(m5.sum()), "plateau": int(mp.sum()), "local": int(ml.sum()),
                          "local_ne_v5": int((ml != m5).sum()), "local_ne_plateau_in_core": int(((ml != mp) & (W >= 1)).sum()),
                          "plateau_ne_v5_outside": int(((mp != m5) & (W == 0)).sum())}}
    return ml, W, zones, qc_side


def surface_side(unit, sd, ml, m5, W, sp_, corg, v5dir, out_dir):
    k, v = VK[sd]
    f5 = os.path.join(v5dir, "%s_stageA.vtp" % v)
    import pyvista as pv
    P5f = np.asarray(pv.read(f5).points, float)
    pre5, sm5 = mc_smooth(m5, sp_, corg)
    P5s, F5s = poly_arrays(sm5)
    repro = bool(len(P5s) == len(P5f) and np.allclose(P5s, P5f, rtol=0, atol=1e-9))
    repro_err = float(np.abs(P5s - P5f).max()) if len(P5s) == len(P5f) else None
    v5map = dict(zip(keys_of(pre5), range(len(pre5))))
    prel, sml = mc_smooth(ml, sp_, corg)
    Pls, Fl = poly_arrays(sml)
    # w at each marching-cubes vertex: max over the 2x2x2 voxels around it
    ijk = (prel - corg) / sp_
    lo = np.floor(ijk + 1e-6).astype(int)
    wmax = np.zeros(len(prel))
    shp = W.shape
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                ii_ = np.clip(lo[:, 0] + di, 0, shp[2] - 1)
                jj_ = np.clip(lo[:, 1] + dj, 0, shp[1] - 1)
                kk_ = np.clip(lo[:, 2] + dk, 0, shp[0] - 1)
                wmax = np.maximum(wmax, W[kk_, jj_, ii_])
    kl = keys_of(prel)
    j5 = np.array([v5map.get(x, -1) for x in kl])
    outside = (wmax == 0)
    take = outside & (j5 >= 0)
    newP = Pls.copy()
    newP[take] = (P5f if repro else P5s)[j5[take]]   # 1.1: the v5 FILE positions (bit-identical outside)
    from vtk.util import numpy_support as ns
    sml.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(newP), deep=1))
    fo = os.path.join(out_dir, "%s_stageA.vtp" % v)
    write_poly(sml, fo)
    # checks: every outside vertex is a v5 file vertex; every all-outside triangle is a v5 triangle
    id5 = {x: i for i, x in enumerate(keys_of(P5f))}
    out_ok = int(sum(1 for x in keys_of(newP[outside]) if x in id5))
    tri_out = outside[Fl].all(1)
    T5 = set(tuple(sorted(t)) for t in F5s.tolist())
    KP = np.array([id5.get(x, -1) for x in keys_of(newP)])
    tri_same = int(sum(1 for t in KP[Fl[tri_out]].tolist() if min(t) >= 0 and tuple(sorted(t)) in T5))
    disp = np.linalg.norm(newP - Pls, axis=1)
    rec = {"v5_recompute_equals_v5_file": repro, "v5_recompute_max_err_mm": repro_err,
           "vertices": int(len(newP)), "vertices_outside_zones": int(outside.sum()),
           "outside_unmatched_in_v5_mc": int((outside & (j5 < 0)).sum()),
           "outside_vertices_equal_v5_file": out_ok, "triangles": int(len(Fl)), "triangles_outside": int(tri_out.sum()),
           "triangles_outside_equal_v5": tri_same,
           "spill_undone_max_mm": float(disp[take].max()) if take.any() else 0.0,
           "spill_undone_p99_mm": float(np.percentile(disp[take], 99)) if take.any() else 0.0,
           "topology": topo(newP, Fl), "topology_v5": topo(P5f, F5s), "md5": md5(fo)}
    return rec


def seeds_side(unit, sd, ml, m5, st5, v5dir, platdir, sp_, corg, out_dir):
    k, v = VK[sd]
    st = st5[v]
    P = np.asarray(st["P"], float)
    s = np.asarray(st["s"], float)
    tree = cKDTree(P)

    def ends(m):
        kk, jj, ii = np.nonzero(m)
        X = corg + np.c_[ii, jj, kk] * sp_
        _, ix = tree.query(X)
        sv = s[ix]
        a = sv <= max(1.5, sv.min() + 1.5)
        b = sv >= min(s[-1] - 1.5, sv.max() - 1.5)
        return (kk[a], jj[a], ii[a]), (kk[b], jj[b], ii[b])
    (a5, b5), (al, bl) = ends(m5), ends(ml)
    same = all(np.array_equal(x, y) for x, y in zip(a5 + b5, al + bl))
    src = os.path.join(v5dir, "%s_seeds.json" % v)
    if same:
        shutil.copy2(src, os.path.join(out_dir, "%s_seeds.json" % v))
        return {"seeds": "v5's (label ends unchanged)"}
    X = lambda t: (corg + np.c_[t[2], t[1], t[0]] * sp_).mean(0)  # noqa: E731
    sj = json.load(open(src))
    sj = {"source": X(al).tolist(), "target": X(bl).tolist(),
          "note": "local_seg: centroids of the local label's first / last 1.5 mm (ct_seg4 diff 7 rule); v5's: %s" % src}
    jdump(sj, os.path.join(out_dir, "%s_seeds.json" % v))
    return {"seeds": "recomputed (a zone reaches a label end)"}


def unit_run(unit):
    cfg = D[unit]
    v5dir = cfg["v5_seg_dir"]
    pdir = os.path.join(PLAT, unit)
    od = os.path.join(OUT, unit)
    if os.path.exists(os.path.join(od, ".seg_done")):
        print("done already", unit, flush=True)
        return
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    z5 = np.load(os.path.join(v5dir, "labels_kji.npz"), allow_pickle=True)
    zp = np.load(os.path.join(pdir, "labels_kji.npz"), allow_pickle=True)
    sp_ = np.asarray(z5["spacing_mm"], float)
    corg = np.asarray(z5["origin_mm"], float)
    assert np.allclose(zp["spacing_mm"], sp_), "v5 / plateau spacings differ"
    L5, Lp = z5["labels"], zp["labels"]
    grid = {"aligned": True}
    if not (np.allclose(zp["origin_mm"], corg) and Lp.shape == L5.shape):
        # 1.1: the two runs cropped the CT differently (stations recentred differently): both labels placed in their
        # union grid (same CT voxels, integer offsets); the outputs live on that grid
        o5, op = np.asarray(z5["origin_mm"], float), np.asarray(zp["origin_mm"], float)
        oa = np.round((op - o5) / sp_).astype(int)
        assert np.allclose(op, o5 + oa * sp_, atol=1e-3), "grids not commensurate"
        lo = np.minimum(oa, 0)
        hi = np.maximum(oa + np.array(Lp.shape[::-1]), np.array(L5.shape[::-1]))
        shape = tuple(int(v) for v in (hi - lo)[::-1])
        corg = o5 + lo * sp_
        def _pl(a, off):
            out = np.zeros(shape, a.dtype)
            out[off[0]:off[0] + a.shape[0], off[1]:off[1] + a.shape[1], off[2]:off[2] + a.shape[2]] = a
            return out
        L5, Lp = _pl(L5, (-lo)[::-1]), _pl(Lp, (oa - lo)[::-1])
        grid = {"aligned": False, "union_origin_mm": corg.tolist(), "union_shape_kji": list(shape),
                "plateau_offset_xyz_vox": oa.tolist()}
    st5 = json.load(open(os.path.join(v5dir, "stations_rays.json")))
    Ll = L5.copy()
    qc = {"script": VERSION, "unit": unit, "v5_seg_dir": v5dir, "plateau_seg_dir": pdir, "grid": grid, "sides": {}}
    zones_all = {}
    merged = {}
    for sd in ("R", "L"):
        k, v = VK[sd]
        if v not in st5 or not os.path.exists(os.path.join(R5, unit, "%s.json" % sd)):
            continue
        if not os.path.exists(os.path.join(pdir, "%s_stageA.vtp" % v)):
            continue
        ml, W, zones, qside = one_side(unit, sd, L5, Lp, st5, sp_, corg, od, qc)
        merged[sd] = (ml, W)
        zones_all[sd] = zones
        qc["sides"][sd] = qside
    for sd, (ml, W) in merged.items():
        k, v = VK[sd]
        Ll[Ll == k] = 0
        Ll[ml & (Ll == 0)] = k
    np.savez_compressed(os.path.join(od, "labels_kji.npz"), labels=Ll, origin_mm=corg, spacing_mm=sp_,
                        names=z5["names"] if "names" in z5 else np.array([]),
                        note="local_seg: v5 labels with the VA labels merged with plateau's in the D15 zones")
    for sd, (ml, W) in merged.items():
        k, v = VK[sd]
        mk = Ll == k
        qc["sides"][sd]["surface"] = surface_side(unit, sd, mk, L5 == k, W, sp_, corg, v5dir, od)
        qc["sides"][sd].update(seeds_side(unit, sd, mk, L5 == k, st5, v5dir, pdir, sp_, corg, od))
        print(unit, sd, json.dumps({x: qc["sides"][sd]["surface"][x] for x in (
            "v5_recompute_equals_v5_file", "vertices", "vertices_outside_zones", "outside_vertices_equal_v5_file",
            "triangles_outside", "triangles_outside_equal_v5", "spill_undone_max_mm")}),
            "core %.1f / %.1f mm" % (zones_all[sd]["core_mm"], zones_all[sd]["length_mm"]), flush=True)
    for f in ("edge.json", "measure.json", "landmarks.json"):
        if os.path.exists(os.path.join(v5dir, f)):
            shutil.copy2(os.path.join(v5dir, f), os.path.join(od, f))
    jdump(zones_all, os.path.join(od, "zones.json"))
    qc["seconds"] = round(time.time() - t0, 1)
    jdump(qc, os.path.join(od, "local_qc.json"))
    jdump({"v6": {"variant": "local", "rule": "D15: v5 ct_seg4 outside the artefact zones, plateau inside (blend %.1f mm)" % BLEND_MM,
                  "script": VERSION, "zones_file": os.path.join(od, "zones.json")}}, os.path.join(od, "seg_qc.json"))
    open(os.path.join(od, ".seg_done"), "w").write("%s %s\n" % (VERSION, time.strftime("%F %T")))
    print("DONE", unit, "%.0f s" % qc["seconds"], flush=True)


FAILS = []


def main():
    arg = sys.argv[1]
    units = [u for u in D] if arg == "all" else arg.split(",")
    if arg == "rev":
        units = [u for u in D][::-1]
    for u in units:
        claim = os.path.join(OUT, "_claim", u)
        os.makedirs(os.path.join(OUT, "_claim"), exist_ok=True)
        if os.path.exists(os.path.join(OUT, u, ".seg_done")):
            continue
        try:
            os.mkdir(claim)
        except FileExistsError:
            print("claimed elsewhere", u, flush=True)
            continue
        try:
            unit_run(u)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", u, repr(e), flush=True)
            FAILS.append(u)
            os.rmdir(claim)


if __name__ == "__main__":
    main()
    if FAILS:
        print('UNITS FAILED', FAILS, flush=True)
        sys.exit(1)
