"""CHECK 5, pass 2 -- what the extra connected components ARE, and edge/diameter ratios.

Pass 1 found 1-4 surface components per anatomy in BOTH the 216-set and the 49-set. A
1-cell fragment cannot be tested with a naive sd>0 containment test (a lone triangle makes
half of space "inside"), so this pass characterises each component directly:

  - per component: cells, area, bbox diagonal, open-boundary edges, non-manifold edges
  - is the fragment INSIDE the main lumen (a floating obstacle a device can hit) or
    OUTSIDE the wall (harmless debris)? -- exact signed distance of the fragment's own
    vertices against the MAIN component, positive == inside on this build
  - centerline attribution: every branch station assigned to the component whose wall is
    nearest (min |sd|). If all 17 branches attribute to one component the navigable lumen
    is single.
  - edge length vs local vessel diameter: fraction of edges longer than 1x and 2x the
    local diameter, overall and restricted to the RCCA route.

SIGN control (HANDOFF 11.2) runs first on topbrain topcow_mr_001.
"""
import os, sys, json, glob
import numpy as np
import pyvista as pv
import vtk

CAR = "/opt/eve_training/carotid/anatomies"
TB = "/opt/eve_training/results_topbrain/anatomies"


def load_cl(folder):
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
                pts.append((y, -z, -x))
            for meas in m.get("measurements", []):
                if meas["name"] == "Radius" and meas.get("controlPointValues"):
                    rad.extend(meas["controlPointValues"])
        p = np.array(pts, float)
        r = np.array(rad, float) if len(rad) == len(pts) else np.full(len(pts), np.nan)
        nm = os.path.splitext(os.path.splitext(fn)[0])[0].replace("Centerline curve ", "")
        out.append((nm, p, r))
    return out


def sdist(mesh, pts):
    f = vtk.vtkImplicitPolyDataDistance()
    f.SetInput(mesh)
    return np.array([f.EvaluateFunction(p) for p in pts])


def nearest_idx(q, ref, chunk=4000):
    out = np.empty(len(q), np.int64)
    for i in range(0, len(q), chunk):
        d = ((q[i:i + chunk, None, :] - ref[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def feat(mesh, **kw):
    kw2 = dict(boundary_edges=False, feature_edges=False,
               non_manifold_edges=False, manifold_edges=False)
    kw2.update(kw)
    return mesh.extract_feature_edges(**kw2).n_cells


def analyse(d):
    name = os.path.basename(d)
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate()
    m = m.clean(tolerance=1e-6, absolute=False)

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(m); conn.SetExtractionModeToAllRegions(); conn.Update()
    ncomp = int(conn.GetNumberOfExtractedRegions())

    comps = []
    for i in range(ncomp):
        c2 = vtk.vtkPolyDataConnectivityFilter()
        c2.SetInputData(m); c2.SetExtractionModeToSpecifiedRegions()
        c2.AddSpecifiedRegion(i); c2.Update()
        g = vtk.vtkCleanPolyData(); g.SetInputData(c2.GetOutput()); g.Update()
        comp = pv.wrap(g.GetOutput())
        if comp.n_cells == 0:
            continue
        b = comp.bounds
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
        comps.append(dict(i=i, cells=int(comp.n_cells), pts=int(comp.n_points),
                          area=float(comp.area), diag=diag,
                          ob=int(feat(comp, boundary_edges=True)),
                          nm=int(feat(comp, non_manifold_edges=True)),
                          ctr=[float(x) for x in comp.center], _m=comp))
    comps.sort(key=lambda c: -c["cells"])
    main = comps[0]["_m"]

    # fragment position relative to the MAIN lumen wall
    for c in comps[1:]:
        s = sdist(main, np.asarray(c["_m"].points))
        c["frag_sd_med"] = float(np.median(s))
        c["frag_frac_inside"] = float((s > 0).mean())

    cls = load_cl(os.path.join(d, "Centrelines_comb"))
    allp = np.vstack([c[1] for c in cls])
    allr = np.concatenate([c[2] for c in cls])

    # attribute every station to the component whose wall is nearest
    D = np.stack([np.abs(sdist(c["_m"], allp)) for c in comps], 1)
    own = D.argmin(1)
    br_owner = {}
    k = 0
    for nm, p, _r in cls:
        o = own[k:k + len(p)]
        vals, cnts = np.unique(o, return_counts=True)
        br_owner[nm] = [int(vals[cnts.argmax()]), float(cnts.max() / len(p)), len(p)]
        k += len(p)
    n_owning = len(set(v[0] for v in br_owner.values()))

    # measured bore at every station, against the MAIN component
    sd_main = sdist(main, allp)

    P = main.points
    F = main.faces.reshape(-1, 4)[:, 1:]
    E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    mid = 0.5 * (P[E[:, 0]] + P[E[:, 1]])
    ni = nearest_idx(mid, allp)
    rd = allr[ni]
    rm = sd_main[ni]
    ok = rm > 0.05
    ratio_d = L / (2 * rd)
    ratio_m = np.where(ok, L / (2 * np.where(ok, rm, 1)), np.nan)

    # RCCA-route subset
    starts = np.cumsum([0] + [len(c[1]) for c in cls])
    rcca_mask = np.zeros(len(allp), bool)
    reca_mask = np.zeros(len(allp), bool)
    for j, (nm, p, _r) in enumerate(cls):
        if nm.strip().endswith("RCCA"):
            rcca_mask[starts[j]:starts[j + 1]] = True
        if nm.strip().endswith("RECA"):
            reca_mask[starts[j]:starts[j + 1]] = True
    er = rcca_mask[ni]

    def q(x, p):
        x = np.asarray(x); x = x[np.isfinite(x)]
        return float(np.percentile(x, p)) if len(x) else float("nan")

    out = dict(
        name=name, ncomp=len(comps),
        comps=[{k2: v for k2, v in c.items() if k2 != "_m"} for c in comps],
        n_owning=n_owning,
        br_owner=br_owner,
        frac_edge_gt_1d=float((ratio_d > 1).mean()), frac_edge_gt_2d=float((ratio_d > 2).mean()),
        frac_edge_gt_1m=float(np.nanmean(ratio_m > 1)), frac_edge_gt_2m=float(np.nanmean(ratio_m > 2)),
        rcca_edges=int(er.sum()),
        rcca_ratio_d_med=q(ratio_d[er], 50), rcca_ratio_d_p90=q(ratio_d[er], 90),
        rcca_ratio_m_med=q(ratio_m[er], 50), rcca_ratio_m_p90=q(ratio_m[er], 90),
        rcca_e_med=float(np.median(L[er])), rcca_e_p90=q(L[er], 90), rcca_e_max=float(L[er].max()),
        bore_rcca_med=float(np.median(sd_main[rcca_mask])),
        bore_reca_med=float(np.median(sd_main[reca_mask])) if reca_mask.any() else float("nan"),
        main_cells=int(main.n_cells), main_ob=comps[0]["ob"], main_nm=comps[0]["nm"],
    )
    return out


c = analyse(os.path.join(TB, "topcow_mr_001"))
print("CONTROL topcow_mr_001 bore_rcca_med=%+.3f mm" % c["bore_rcca_med"])
if not (0.2 < c["bore_rcca_med"] < 20):
    print("CONTROL FAILED -- VOID"); sys.exit(1)
print("CONTROL OK")

for tag, root in (("CAR", CAR), ("TB", TB)):
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        try:
            r = analyse(d); r["set"] = tag
            print("ROW " + json.dumps(r))
        except Exception as ex:
            print("ERR %s %s %s" % (tag, os.path.basename(d), repr(ex)[:200]))
        sys.stdout.flush()
print("DONE")
