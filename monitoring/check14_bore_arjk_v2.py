"""CHECK 14 addendum -- reconciliation and sharpening.

(a) estimator sensitivity: native station spacing vs 0.25 mm densified, so the
    difference from HANDOFF 11.5's published -0.75 / +0.38 can be attributed.
(b) is ANY densified point pessimistic (declared r < exact bore)?
(c) are set-B's host-segment DECLARED radii identical to the shipped host's below
    the ramp window? If yes, the host-segment delta gap is purely mesh erosion
    introduced by bake_meshes, on geometry that is nominally shared.
(d) delta as a function of arclength, pooled -- shape of the optimism profile.
"""
import glob, json, os, sys
import numpy as np, pyvista as pv, vtk

sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches, DualDeviceNav

SEAM = 130.0


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def densify(p, r, step):
    s = arclen(p)
    n = int(np.ceil(s[-1] / step)) + 1
    t = np.linspace(0.0, s[-1], max(n, 2))
    qq = np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)
    return qq, np.interp(t, s, r), t


def load(mesh_path, cl_dir, want):
    m = pv.read(mesh_path).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    brs = {str(b.name): b for b in load_branches(cl_dir)}
    k = [x for x in brs if want in x.upper()][0]
    b = brs[k]
    return f, np.asarray(b.coordinates, float), np.asarray(b.radii, float)


def sd(f, pts):
    return np.array([f.EvaluateFunction(p) for p in pts])


print("=== (a) ESTIMATOR SENSITIVITY: native station spacing vs 0.25 mm densified ===")
print("%-40s %10s %10s %10s" % ("scope", "native", "dens0.25", "shift"))

dv = DualDeviceNav()
mh = pv.read(dv.vessel_tree.mesh_path).triangulate().clean()
fh = vtk.vtkImplicitPolyDataDistance(); fh.SetInput(mh)
hb = [b for b in dv.vessel_tree.branches if "RCCA" in str(b.name).upper()][0]
hp = np.asarray(hb.coordinates, float); hr = np.asarray(hb.radii, float)
hs_nat = arclen(hp)
dn = np.median(sd(fh, hp) - hr)
q, rr, t = densify(hp, hr, 0.25)
dd = np.median(sd(fh, q) - rr)
print("%-40s %10.3f %10.3f %10.3f" % ("HOST RCCA (HANDOFF 11.5: +0.38)", dn, dd, dd - dn))
print("   host native station spacing: median %.3f mm, max %.3f mm, n=%d"
      % (np.median(np.diff(hs_nat)), np.diff(hs_nat).max(), len(hp)))

CTRL = "/opt/eve_training/results_topbrain/anatomies/topcow_mr_001"
f, p, r = load(os.path.join(CTRL, "vessel_architecture_collision.obj"),
               os.path.join(CTRL, "Centrelines_comb"), "RCCA")
dn = np.median(sd(f, p) - r)
q, rr, t = densify(p, r, 0.25)
dd = np.median(sd(f, q) - rr)
print("%-40s %10.3f %10.3f %10.3f" % ("49-set topcow_mr_001 (HANDOFF: -0.75)", dn, dd, dd - dn))
print("   49-set native station spacing: median %.3f mm, max %.3f mm, n=%d"
      % (np.median(np.diff(arclen(p))), np.diff(arclen(p)).max(), len(p)))

dirs = sorted(glob.glob("/opt/eve_training/carotid/anatomies/*"))
dirs = [d for d in dirs if os.path.isdir(d)]
sub = dirs[::9]
nat, den, natp, denp = [], [], [], []
maxdelta, spacing = [], []
prof_s, prof_d = [], []
host_r_diff = []
for d in sub:
    prov = json.load(open(os.path.join(d, "provenance.json")))
    f, p, r = load(os.path.join(d, "vessel_architecture_collision.obj"),
                   os.path.join(d, "Centrelines_comb"), "RCCA")
    spacing.append(np.median(np.diff(arclen(p))))
    nat.append(np.median(sd(f, p) - r))
    q, rr, t = densify(p, r, 0.25)
    dl = sd(f, q) - rr
    den.append(np.median(dl))
    maxdelta.append(dl.max())
    m = t >= SEAM
    denp.append(np.median(dl[m])); natp.append(prov["host_cut_mm"])
    prof_s.append(t); prof_d.append(dl)
    # (c) host-segment declared radii vs shipped host, below the ramp window
    hc, bl = prov["host_cut_mm"], prov["blend_mm"]
    lim = max(hc - bl, 0.0)
    mm = t < lim
    if mm.sum() > 5:
        ref = np.interp(t[mm], hs_nat, hr)
        host_r_diff.append(float(np.max(np.abs(rr[mm] - ref))))

nat = np.array(nat); den = np.array(den)
print("%-40s %10.3f %10.3f %10.3f  (n=%d subsample)"
      % ("set-B RCCA whole route", np.median(nat), np.median(den),
         np.median(den - nat), len(sub)))
print("   set-B native station spacing: median %.3f mm, max %.3f mm"
      % (np.median(spacing), np.max(spacing)))
print("   per-anatomy shift native->densified: p5 %.3f med %.3f p95 %.3f"
      % (np.percentile(den - nat, 5), np.median(den - nat), np.percentile(den - nat, 95)))

print()
print("=== (b) is any densified point PESSIMISTIC (declared r < exact bore)? ===")
md = np.array(maxdelta)
print("   max over the route of (clearance - declared r), per anatomy:")
print("   p5 %.3f  p25 %.3f  med %.3f  p75 %.3f  p95 %.3f  MAX %.3f"
      % tuple(list(np.percentile(md, [5, 25, 50, 75, 95])) + [md.max()]))
print("   anatomies with ANY point where declared r understates bore: %d / %d"
      % (int((md > 0).sum()), len(md)))

print()
print("=== (c) set-B host-segment declared radii vs the shipped host (below the ramp) ===")
hrd = np.array(host_r_diff)
print("   max |set-B declared r - shipped host declared r| over s < host_cut-blend:")
print("   p50 %.4f mm  p95 %.4f mm  max %.4f mm   (n=%d)"
      % (np.median(hrd), np.percentile(hrd, 95), hrd.max(), len(hrd)))
print("   -> if ~0, the host segment ships IDENTICAL declared radii and any delta")
print("      difference against the host control is the re-baked mesh, not the radii.")

print()
print("=== (d) POOLED delta profile along composed arclength (all subsample points) ===")
S = np.concatenate(prof_s); D = np.concatenate(prof_d)
print("%10s %8s %9s %9s %9s %9s" % ("s band mm", "n", "p10", "median", "p90", "min"))
edges = [0, 10, 20, 40, 60, 80, 100, 130, 150, 170, 190, 210, 230, 260]
for a, b in zip(edges[:-1], edges[1:]):
    m = (S >= a) & (S < b)
    if m.sum() < 20:
        continue
    v = D[m]
    print("%4d-%-5d %8d %9.3f %9.3f %9.3f %9.3f"
          % (a, b, m.sum(), np.percentile(v, 10), np.median(v),
             np.percentile(v, 90), v.min()))
