import sys, os, json
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/results_topbrain/anatomies"
M2N = {"topcowmr004": "topcow_mr_004", "topcowmr008": "topcow_mr_008",
       "topcowmr017": "topcow_mr_017", "topcowmr023": "topcow_mr_023"}
OFF = 33.314

def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))

def signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance(); imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])

def enclosed(mesh, pts):
    ps = pv.PolyData(np.asarray(pts, float))
    sel = vtk.vtkSelectEnclosedPoints(); sel.SetInputData(ps); sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6); sel.CheckSurfaceOff(); sel.Update()
    o = sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return np.array([o.GetTuple1(i) for i in range(len(pts))]) > 0.5

PROF = {}
for fp, name in M2N.items():
    d0 = os.path.join(ROOT, name)
    mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float); S = arclen(C)
    g = np.arange(0.0, S[-1], 0.25); g = np.append(g, S[-1])
    P = np.stack([np.interp(g, S, C[:, i]) for i in range(3)], 1)
    sd = signed(mesh, P)
    if (sd < 0).mean() < 0.5: sd = -sd
    enc = enclosed(mesh, P)
    d_eff = np.where(enc, np.abs(sd), 0.0)
    PROF[fp] = (g, d_eff, float(S[-1]))
    # radii from the branch, for lumen-radius context
    try:
        R = np.asarray(rc.radii, float)
        rr = np.interp(g, S, R)
    except Exception:
        rr = np.full_like(g, np.nan)
    PROF[fp] = (g, d_eff, float(S[-1]), rr)
    print("[prof] %s L=%.1f  clearance: min=%.3f p01=%.3f p05=%.3f med=%.3f | radius med=%.2f min=%.2f"
          % (fp, S[-1], d_eff.min(), np.percentile(d_eff, 1), np.percentile(d_eff, 5),
             np.median(d_eff), np.nanmedian(rr), np.nanmin(rr)), flush=True)

def at(fp, s):
    g, d, L, rr = PROF[fp]
    return float(np.interp(s, g, d)), float(np.interp(s, g, rr))

# clearance at every episode target station, both arms
data = json.loads(DATA)
print("\n=== clearance at TARGET station, all episodes ===")
for arm in ("H0", "TEACHER"):
    vals = []
    for e in data[arm]:
        c, r = at(e["mesh"], e["path_len"] - OFF)
        vals.append(c)
    v = np.array(vals)
    print("  %-8s n=%d  min=%.3f  p01=%.3f  p05=%.3f  med=%.3f   n<0.18=%d  n<0.30=%d  n<0.35=%d"
          % (arm, len(v), v.min(), np.percentile(v, 1), np.percentile(v, 5), np.median(v),
             int((v < 0.18).sum()), int((v < 0.30).sum()), int((v < 0.35).sum())))

print("\n=== clearance along the corridor UP TO the target (min over 0..s_tgt) ===")
for arm in ("H0", "TEACHER"):
    mins = []
    for e in data[arm]:
        g, d, L, rr = PROF[e["mesh"]]
        s = e["path_len"] - OFF
        m = d[g <= s]
        mins.append(float(m.min()) if len(m) else float("nan"))
    v = np.array(mins)
    print("  %-8s min=%.3f  n_corr<0.18=%d  n_corr<0.30=%d  n_corr<0.35=%d"
          % (arm, v.min(), int((v < 0.18).sum()), int((v < 0.30).sum()), int((v < 0.35).sum())))

print("\n=== clearance at the ARREST depth of each H0 failure ===")
for f in data["ARREST"]:
    c, r = at(f["mesh"], f["s_max"])
    print("  %-12s ep%-3d pid%-4d mode=%-10s s_max=%6.1f  clearance=%6.3f mm  lumen_r=%5.2f mm  (target s=%6.1f)"
          % (f["mesh"], f["ep"], f["pid"], f["mode"], f["s_max"], c, r, f["tgt_s"]))
