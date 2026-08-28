"""NAVIGABILITY QUANTIFICATION: retained 22 cohort vs HOST(collision) vs HOST(visual).
Exact estimators only: vtkImplicitPolyDataDistance (signed) + exact planar cross-sections.
Read-only. Writes JSON to /tmp/out/nav_quality.json
"""
import glob
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches  # noqa: E402

ROOT = "/opt/eve_training/results_topbrain/anatomies"
WIRE_R, CATH_R = 0.18, 0.35
TRUNK_MAX = 130.0
EXCLUDE = {"topcow_mr_013", "topcow_mr_014", "topcow_mr_015"}
RETAINED = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(ROOT, "*")))
            if os.path.isdir(d) and os.path.basename(d) not in EXCLUDE]


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def exact_signed(mesh, pts):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])


def tangents(C):
    T = np.gradient(C, axis=0)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    n[n == 0] = 1
    return T / n


def curv_radius(C, S, win=4.0):
    R = np.full(len(C), np.inf)
    for i in range(len(C)):
        lo = int(np.searchsorted(S, S[i] - win))
        hi = int(np.searchsorted(S, S[i] + win)) - 1
        if lo < 0 or hi >= len(C) or hi - lo < 2:
            continue
        A, B, Cc = C[lo], C[i], C[hi]
        a = np.linalg.norm(B - Cc)
        b = np.linalg.norm(A - Cc)
        e = np.linalg.norm(A - B)
        ar = 0.5 * np.linalg.norm(np.cross(B - A, Cc - A))
        R[i] = np.inf if ar < 1e-12 else (a * b * e) / (4 * ar)
    return R


def bend_deg(C, S, win=4.0):
    out = np.zeros(len(C))
    for i in range(len(C)):
        lo = int(np.searchsorted(S, S[i] - win))
        hi = int(np.searchsorted(S, S[i] + win)) - 1
        if lo < 0 or hi >= len(C) or hi - lo < 2:
            continue
        u = C[i] - C[lo]
        v = C[hi] - C[i]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        out[i] = np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1)))
    return out


def resample(C, step=0.5):
    S = arclen(C)
    q = np.arange(0, S[-1], step)
    return np.stack([np.interp(q, S, C[:, k]) for k in range(3)], 1), q


def total_turning(C, step=0.5):
    Cr, _ = resample(C, step)
    d = np.diff(Cr, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n == 0] = 1
    d = d / n
    cs = np.clip((d[:-1] * d[1:]).sum(1), -1, 1)
    return float(np.degrees(np.arccos(cs)).sum())


def loop_area(pts3, normal):
    n = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    x = pts3 @ u
    y = pts3 @ v
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def xsec_reff(mesh_vtk, origin, normal):
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
    cut = vtk.vtkCutter()
    cut.SetInputData(mesh_vtk)
    cut.SetCutFunction(plane)
    cut.Update()
    o = cut.GetOutput()
    if o.GetNumberOfPoints() == 0:
        return np.nan
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(o)
    conn.SetExtractionModeToClosestPointRegion()
    conn.SetClosestPoint(float(origin[0]), float(origin[1]), float(origin[2]))
    conn.Update()
    st = vtk.vtkStripper()
    st.SetInputData(conn.GetOutput())
    st.JoinContiguousSegmentsOn()
    st.Update()
    sp = st.GetOutput()
    if sp.GetNumberOfPoints() == 0:
        return np.nan
    P = vtk_to_numpy(sp.GetPoints().GetData())
    lines = sp.GetLines()
    lines.InitTraversal()
    ida = vtk.vtkIdList()
    best = 0.0
    while lines.GetNextCell(ida):
        idx = [ida.GetId(k) for k in range(ida.GetNumberOfIds())]
        if len(idx) < 4:
            continue
        if idx[0] == idx[-1]:
            idx = idx[:-1]
        best = max(best, loop_area(P[idx], np.asarray(normal, float)))
    if best <= 0:
        return np.nan
    return float(np.sqrt(best / np.pi))


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {"n": int(x.size), "min": float(x.min()), "p01": float(np.percentile(x, 1)),
            "p05": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)),
            "med": float(np.median(x)), "mean": float(x.mean()), "max": float(x.max())}


def load(tag):
    if tag == "HOST_COLLISION":
        vt = DualDeviceNav().vessel_tree
        return pv.read(vt.mesh_path).triangulate().clean(), list(vt.branches)
    if tag == "HOST_VISUAL":
        vt = DualDeviceNav().vessel_tree
        return pv.read(vt.visu_mesh_path).triangulate().clean(), list(vt.branches)
    d0 = os.path.join(ROOT, tag)
    m = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    return m, load_branches(os.path.join(d0, "Centrelines_comb"))


def analyze(tag):
    mesh, brs = load(tag)
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float)
    R = np.asarray(rc.radii, float)
    S = arclen(C)
    sd = exact_signed(mesh, C)
    inside = sd < 0
    if inside.mean() < 0.5:
        inside = ~inside
        sd = -sd
    d_eff = np.where(inside, np.abs(sd), 0.0)
    T = tangents(C)
    reff = np.array([xsec_reff(mesh, C[i], T[i]) for i in range(len(C))])
    Rc = curv_radius(C, S, 4.0)
    bd = bend_deg(C, S, 4.0)
    chord = float(np.linalg.norm(C[-1] - C[0]))

    def seg(m, name):
        if m.sum() == 0:
            return {"seg": name, "n": 0}
        dd = d_eff[m]
        rr = reff[m]
        rcm = Rc[m]
        fin = np.isfinite(rcm)
        tb = (rcm < 8.0) & (dd < 0.6)
        return {"seg": name, "n": int(m.sum()),
                "clear": stats(dd),
                "n_blk_wire": int((dd < WIRE_R).sum()),
                "n_blk_cath": int((dd < CATH_R).sum()),
                "n_outside": int((~inside[m]).sum()),
                "reff": stats(rr),
                "reff_over_stated": float(np.nanmedian(rr / R[m])),
                "stated_r": stats(R[m]),
                "Rc_min": float(np.nanmin(rcm[fin])) if fin.any() else None,
                "Rc_p05": float(np.percentile(rcm[fin], 5)) if fin.any() else None,
                "Rc_med": float(np.median(rcm[fin])) if fin.any() else None,
                "bend_max": float(bd[m].max()), "bend_p95": float(np.percentile(bd[m], 95)),
                "n_tight_bent": int(tb.sum()),
                "arclen_span": [float(S[m].min()), float(S[m].max())]}

    all_m = np.ones(len(C), bool)
    core_m = np.arange(len(C)) < (len(C) - 2)
    trunk_m = S < TRUNK_MAX
    distal_m = S >= (S[-1] / 2.0)
    graft_m = S >= TRUNK_MAX
    return {
        "tag": tag, "n_stations": len(C), "route_len": float(S[-1]),
        "chord": chord, "tortuosity": float(S[-1] / chord),
        "total_turning_deg": total_turning(C),
        "station_spacing_med": float(np.median(np.diff(S))),
        "segments": {"ALL": seg(all_m, "ALL"), "CORE_no_last2": seg(core_m, "CORE"),
                     "TRUNK_s_lt_130": seg(trunk_m, "TRUNK"),
                     "DISTAL_HALF": seg(distal_m, "DISTAL_HALF"),
                     "GRAFT_s_ge_130": seg(graft_m, "GRAFT"),
                     "GRAFT_core": seg(graft_m & core_m, "GRAFT_core")},
        "C_head": C[:3].tolist(),
        "profile": {"s": S.tolist(), "clear": d_eff.tolist(), "reff": reff.tolist(),
                    "Rc": np.where(np.isfinite(Rc), Rc, -1.0).tolist(),
                    "stated_r": R.tolist(), "bend": bd.tolist(),
                    "C": C.tolist()},
    }


TAGS = ["HOST_COLLISION", "HOST_VISUAL"] + RETAINED
res = {}
for t in TAGS:
    try:
        res[t] = analyze(t)
        a = res[t]["segments"]["ALL"]
        tr = res[t]["segments"]["TRUNK_s_lt_130"]
        print("%-16s n=%3d len=%6.1f | ALL med=%.3f min=%.3f blkW=%d | TRUNK med=%.3f p05=%.3f "
              "min=%.3f reff/r=%.3f" % (t, res[t]["n_stations"], res[t]["route_len"],
                                        a["clear"]["med"], a["clear"]["min"], a["n_blk_wire"],
                                        tr["clear"]["med"], tr["clear"]["p05"],
                                        tr["clear"]["min"], tr["reff_over_stated"]), flush=True)
    except Exception as e:
        print("FAIL", t, repr(e), flush=True)

os.makedirs("/tmp/out", exist_ok=True)
with open("/tmp/out/nav_quality.json", "w") as f:
    json.dump(res, f)
print("WROTE /tmp/out/nav_quality.json")

ref = None
for t in TAGS:
    if t not in res:
        continue
    Cc = np.array(res[t]["profile"]["C"])
    S = np.array(res[t]["profile"]["s"])
    m = S < TRUNK_MAX
    key = (int(m.sum()), np.round(Cc[m], 9).tobytes())
    if ref is None:
        ref = (t, key)
    print("  trunk-identity %-16s n_trunk=%d s_last=%.6f identical_to_%s=%s"
          % (t, int(m.sum()), S[m][-1], ref[0], key == ref[1]))
