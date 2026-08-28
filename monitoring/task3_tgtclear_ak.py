"""Exact clearance AT each of the 220 eval targets, and the min along its route."""
import sys, os, csv, json
sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT = "/opt/eve_training/results_topbrain/anatomies"
RUN = ("/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/"
       "2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
WIRE, CONTACT, CATH = 0.18, 0.30, 0.35
OFFSET = 33.314


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


rows = []
with open(os.path.join(RUN, "episodes.csv")) as f:
    for r in csv.DictReader(f):
        r["plen"] = float(r["path_len_mm"]); r["succ"] = int(r["success"])
        r["steps"] = int(r["steps"])
        rows.append(r)

prof = {}
for a in sorted(set(r["anatomy"] for r in rows)):
    nm = "topcow_mr_" + a.replace("topcowmr", "")
    d0 = os.path.join(ROOT, nm)
    mesh = pv.read(os.path.join(d0, "vessel_architecture_collision.obj")).triangulate().clean()
    brs = load_branches(os.path.join(d0, "Centrelines_comb"))
    rc = next(b for b in brs if "RCCA" in str(b.name).upper())
    C = np.asarray(rc.coordinates, float)
    S = arclen(C)
    g = np.append(np.arange(0.0, S[-1], 0.25), S[-1])
    P = np.stack([np.interp(g, S, C[:, i]) for i in range(3)], 1)
    imp = vtk.vtkImplicitPolyDataDistance(); imp.SetInput(mesh)
    sd = np.array([imp.EvaluateFunction(p) for p in P])
    if (sd < 0).mean() < 0.5:
        sd = -sd
    ps = pv.PolyData(P)
    sel = vtk.vtkSelectEnclosedPoints(); sel.SetInputData(ps); sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6); sel.CheckSurfaceOff(); sel.Update()
    o = sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    ins = np.array([o.GetTuple1(i) for i in range(len(P))]) > 0.5
    prof[a] = (g, np.where(ins, np.abs(sd), 0.0))

print("{:>12} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>6} {:>5}".format(
    "anat", "plen", "s_tgt", "L", "d@tgt", "min[0,s]", "s@min", "steps", "succ"))
recs = []
for r in sorted(rows, key=lambda r: -r["plen"]):
    g, d = prof[r["anatomy"]]
    st = r["plen"] - OFFSET
    i = int(np.argmin(np.abs(g - st)))
    seg = d[:i + 1]
    j = int(np.argmin(seg))
    recs.append(dict(r, dtgt=float(d[i]), dmin=float(seg.min()), smin=float(g[j]), L=float(g[-1])))
    if r["plen"] >= 200 or d[i] < CATH:
        print("{:>12} {:7.1f} {:7.1f} {:7.1f} {:8.3f} {:8.3f} {:8.1f} {:6d} {:5d}".format(
            r["anatomy"], r["plen"], st, g[-1], d[i], seg.min(), g[j], r["steps"], r["succ"]))

print()
for thr, nm in [(WIRE, "wire 0.18"), (CONTACT, "contact 0.30"), (CATH, "cath 0.35")]:
    nt = sum(1 for r in recs if r["dtgt"] < thr)
    nr = sum(1 for r in recs if r["dmin"] < thr)
    print("episodes with clearance AT TARGET < {:>13}: {:3d}/220 | anywhere on route < thr: {:3d}/220"
          .format(nm, nt, nr))
    sub = [r for r in recs if r["dmin"] < thr]
    if sub:
        print("     of those route-blocked: success {}/{}   bands: {}".format(
            sum(r["succ"] for r in sub), len(sub),
            ", ".join(sorted(set("{}:{:.0f}".format(r["anatomy"].replace("topcowmr", ""), r["plen"])
                                 for r in sub)))))

print()
print(">=240 band: clearance at target")
sub = [r for r in recs if r["plen"] >= 240]
print("  n={}  d@tgt min {:.3f} median {:.3f} max {:.3f} ; route-min min {:.3f} median {:.3f}"
      .format(len(sub), min(r["dtgt"] for r in sub), float(np.median([r["dtgt"] for r in sub])),
              max(r["dtgt"] for r in sub), min(r["dmin"] for r in sub),
              float(np.median([r["dmin"] for r in sub]))))
print("  n below wire radius at target: {} ; below catheter radius: {}".format(
    sum(1 for r in sub if r["dtgt"] < WIRE), sum(1 for r in sub if r["dtgt"] < CATH)))
