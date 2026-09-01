"""CHECK 8b, pass 4 -- is the fork OPEN in the collision mesh, or two lumens with a
membrane between them? A positive signed distance along each centerline separately does
not settle this. Two tests:
  (1) mesh connectivity -- number of connected surface regions
  (2) a straight line probe from the RCCA centerline at the fork to the RECA centerline
      just past it, densified to 0.25 mm: any negative reading is a wall in the way.
Control: the same line probe on a segment known to be inside one continuous lumen.
"""
import glob, os, sys
import numpy as np, pyvista as pv, vtk
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

DS = 0.25


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def densify(p, step=DS):
    s = arclen(p)
    g = np.arange(0.0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(g, s, p[:, k]) for k in range(3)]), g


def probe_line(f, a, b, step=DS):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    t = np.linspace(0, 1, n)
    return np.array([f.EvaluateFunction(a + tt * (b - a)) for tt in t])


rows = []
for d in sorted(glob.glob("/opt/eve_training/carotid/anatomies/*")):
    if not os.path.isdir(d):
        continue
    brs = {str(b.name).upper(): b for b in load_branches(os.path.join(d, "Centrelines_comb"))}
    P = np.asarray(brs[[k for k in brs if "RCCA" in k][0]].coordinates, float)
    E = np.asarray(brs[[k for k in brs if "RECA" in k][0]].coordinates, float)
    Pd, Ps = densify(P)
    Ed, Es = densify(E)
    s_fork = float(Ps[int(np.argmin(np.linalg.norm(Pd - E[0], axis=1)))])
    m = pv.read(os.path.join(d, "vessel_architecture_collision.obj")).triangulate().clean()
    reg = m.connectivity("all")
    nreg = int(reg["RegionId"].max()) + 1
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)

    def rp(t):
        return np.array([np.interp(s_fork + t, Ps, Pd[:, k]) for k in range(3)])

    def ep(t):
        return np.array([np.interp(t, Es, Ed[:, k]) for k in range(3)])

    worst = {}
    for t in (2.0, 3.0, 5.0, 8.0):
        if t > Es[-1]:
            continue
        v = probe_line(f, rp(t), ep(t))
        worst[t] = float(v.min())
    ctrl = probe_line(f, rp(2.0), rp(6.0))          # inside one lumen, must stay positive
    rows.append(dict(name=os.path.basename(d), nreg=nreg,
                     w2=worst.get(2.0), w3=worst.get(3.0), w5=worst.get(5.0),
                     w8=worst.get(8.0), ctrl=float(ctrl.min())))

n = len(rows)
print("measured %d" % n)
print("mesh connected surface regions: %s" % dict(
    zip(*np.unique([r["nreg"] for r in rows], return_counts=True))))
print("CONTROL: straight probe RCCA(fork+2) -> RCCA(fork+6), inside one lumen")
c = np.array([r["ctrl"] for r in rows])
print("   min %.3f  p10 %.3f  med %.3f   negative in %d/%d anatomies"
      % (c.min(), np.percentile(c, 10), np.median(c), (c < 0).sum(), n))
for k, t in (("w2", 2.0), ("w3", 3.0), ("w5", 5.0), ("w8", 8.0)):
    v = np.array([r[k] for r in rows if r[k] is not None], float)
    print("probe RCCA(fork+%.0f) -> RECA(t=%.0f): min %.3f p10 %.3f med %.3f max %.3f  "
          "BLOCKED (any negative) in %d/%d"
          % (t, t, v.min(), np.percentile(v, 10), np.median(v), v.max(), (v < 0).sum(), len(v)))
bad = [r["name"] for r in rows if r["w2"] is not None and r["w2"] < 0]
print("blocked at t=2 mm:", bad[:20])
bad8 = [r["name"] for r in rows if r["w8"] is not None and r["w8"] < 0]
print("blocked at t=8 mm: %d  %s" % (len(bad8), bad8[:10]))
