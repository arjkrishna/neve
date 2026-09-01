"""CHECK 5 -- MESH INTEGRITY, all 216 three-source carotid anatomies + the 49-set control.

Per anatomy: open boundary edges, non-manifold edges, duplicate faces, degenerate faces,
connected-component count, single-connected-lumen test, cells/points, edge length
median/p90/max, and edge length relative to LOCAL VESSEL DIAMETER (both declared-radius
and measured-bore variants).

Traps guarded (HANDOFF 11):
  - SIGN: vtkImplicitPolyDataDistance is POSITIVE INSIDE on this build. A known-good
    control (topbrain topcow_mr_001) is run in the SAME script every time; if the control
    does not read positive-inside, the whole run is voided.
  - FRAME: centerlines come through the same LPS->branch map load_branches uses
    ((y,-z,-x)); a mismatch reads in the hundreds of mm and is flagged.
  - ESTIMATOR: exact vtkImplicitPolyDataDistance only. No nearest-vertex, no surface
    sampling. (Bore radii here are at native station spacing -- they are used as a LOCAL
    SCALE for the edge-length ratio, not as a minimum-clearance claim.)

Topology counts are taken on the point-merged mesh (clean); unmerged coincident vertices
manufacture fake boundary edges. Raw vs merged point counts are both reported.
"""
import os, sys, json, glob
import numpy as np
import pyvista as pv
import vtk

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")

CAR = "/opt/eve_training/carotid/anatomies"
TB = "/opt/eve_training/results_topbrain/anatomies"


# ---------------------------------------------------------------- centerlines
def load_cl(folder):
    """Return list of (name, coords Nx3 branch frame, radii N) using load_branches' map."""
    out = []
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith("Centerline curve ") and fn.endswith(".json")):
            continue
        j = json.load(open(os.path.join(folder, fn)))
        pts, rad = [], []
        for m in j["markups"]:
            if m["type"] != "Curve":
                continue
            for cp in m["controlPoints"]:
                x, y, z = [float(v) for v in cp["position"]]
                pts.append((y, -z, -x))          # exactly load_points_from_json
            for meas in m.get("measurements", []):
                if meas["name"] == "Radius" and meas.get("controlPointValues"):
                    rad.extend(meas["controlPointValues"])
        p = np.array(pts, float)
        r = np.array(rad, float) if len(rad) == len(pts) else np.full(len(pts), np.nan)
        out.append((os.path.splitext(os.path.splitext(fn)[0])[0], p, r))
    return out


def sdist(mesh, pts):
    f = vtk.vtkImplicitPolyDataDistance()
    f.SetInput(mesh)
    return np.array([f.EvaluateFunction(p) for p in pts])


def nearest_idx(q, ref, chunk=4000):
    """index into ref of nearest point, brute force, chunked."""
    out = np.empty(len(q), np.int64)
    for i in range(0, len(q), chunk):
        d = ((q[i:i + chunk, None, :] - ref[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def unique_edges(faces):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def feat(mesh, **kw):
    kw2 = dict(boundary_edges=False, feature_edges=False,
               non_manifold_edges=False, manifold_edges=False)
    kw2.update(kw)
    return mesh.extract_feature_edges(**kw2).n_cells


def analyse(d):
    name = os.path.basename(d)
    raw = pv.read(os.path.join(d, "vessel_architecture_collision.obj"))
    n_raw_pts, n_raw_cells = raw.n_points, raw.n_cells
    tri = raw.triangulate()
    n_tri_cells = tri.n_cells
    m = tri.clean(tolerance=1e-6, absolute=False)
    F = m.faces.reshape(-1, 4)[:, 1:]

    # duplicate faces (same vertex set, merged geometry)
    key = np.sort(F, axis=1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    dup_faces = int((cnt - 1).sum())

    # degenerate faces: repeated index, or vanishing area
    rep = int((( key[:, 0] == key[:, 1]) | (key[:, 1] == key[:, 2])).sum())
    P = m.points
    a = P[F[:, 1]] - P[F[:, 0]]
    b = P[F[:, 2]] - P[F[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    tiny = int((area < 1e-8).sum())
    degen = int(((area < 1e-8) | (key[:, 0] == key[:, 1]) | (key[:, 1] == key[:, 2])).sum())

    ob = feat(m, boundary_edges=True)
    nm = feat(m, non_manifold_edges=True)

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(m)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    ncomp = int(conn.GetNumberOfExtractedRegions())

    E = unique_edges(F)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)

    # centerlines + measured bore at every station (exact signed distance)
    cls = load_cl(os.path.join(d, "Centrelines_comb"))
    allp = np.vstack([c[1] for c in cls])
    allr = np.concatenate([c[2] for c in cls])
    sd = sdist(m, allp)                       # POSITIVE = INSIDE on this build
    frac_in = float((sd > 0).mean())
    med_sd = float(np.median(sd))

    mid = 0.5 * (P[E[:, 0]] + P[E[:, 1]])
    ni = nearest_idx(mid, allp)
    r_dec = allr[ni]
    r_mes = sd[ni]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_dec = L / (2.0 * r_dec)
        rm = np.where(r_mes > 0.05, r_mes, np.nan)
        ratio_mes = L / (2.0 * rm)

    # single connected lumen
    lumen = "single"
    comp_note = ""
    if ncomp > 1:
        bodies = []
        for i in range(ncomp):
            c2 = vtk.vtkPolyDataConnectivityFilter()
            c2.SetInputData(m)
            c2.SetExtractionModeToSpecifiedRegions()
            c2.AddSpecifiedRegion(i)
            c2.Update()
            g = vtk.vtkCleanPolyData(); g.SetInputData(c2.GetOutput()); g.Update()
            comp = pv.wrap(g.GetOutput())
            if comp.n_cells == 0:
                continue
            s = sdist(comp, allp)
            bodies.append((i, comp.n_cells, float(comp.area), int((s > 0).sum())))
        holding = [b for b in bodies if b[3] > 0]
        lumen = "single" if len(holding) == 1 else "SPLIT/%d-holding" % len(holding)
        comp_note = ";".join("c%d:cells=%d,area=%.1f,cl_in=%d" % b for b in bodies)

    def q(x, p):
        x = x[np.isfinite(x)]
        return float(np.percentile(x, p)) if len(x) else float("nan")

    return dict(
        name=name, pts_raw=int(n_raw_pts), pts=int(m.n_points),
        cells_raw=int(n_raw_cells), cells_tri=int(n_tri_cells), cells=int(m.n_cells),
        merged_pts=int(n_raw_pts - m.n_points),
        open_b=int(ob), nonman=int(nm), dup=dup_faces, degen=degen,
        rep_idx=rep, tiny_area=tiny, min_area=float(area.min()),
        ncomp=ncomp, lumen=lumen, comps=comp_note,
        n_edges=int(len(E)),
        e_med=float(np.median(L)), e_p90=q(L, 90), e_max=float(L.max()),
        e_min=float(L.min()),
        rd_med=q(ratio_dec, 50), rd_p90=q(ratio_dec, 90), rd_max=q(ratio_dec, 100),
        rm_med=q(ratio_mes, 50), rm_p90=q(ratio_mes, 90), rm_max=q(ratio_mes, 100),
        n_cl=int(len(allp)), n_br=len(cls), frac_cl_inside=frac_in, med_sd=med_sd,
        area=float(m.area),
    )


# ------------------------------------------------------------------- CONTROL
ctrl_dir = os.path.join(TB, "topcow_mr_001")
c = analyse(ctrl_dir)
print("CONTROL topcow_mr_001: frac centerline pts with sd>0 = %.4f  med sd = %+.3f mm"
      % (c["frac_cl_inside"], c["med_sd"]))
if c["frac_cl_inside"] < 0.90 or not (0.2 < c["med_sd"] < 20):
    print("CONTROL FAILED -- sign/frame convention wrong, results VOID")
    sys.exit(1)
print("CONTROL OK: positive==inside confirmed, magnitude at lumen scale")
print("CTRLROW " + json.dumps(c))

for tag, root, pat in (("CAR", CAR, "*"), ("TB", TB, "*")):
    for d in sorted(glob.glob(os.path.join(root, pat))):
        if not os.path.isdir(d):
            continue
        try:
            r = analyse(d)
            r["set"] = tag
            print("ROW " + json.dumps(r))
        except Exception as ex:
            print("ERR %s %s %s" % (tag, os.path.basename(d), repr(ex)[:200]))
        sys.stdout.flush()
print("DONE")
