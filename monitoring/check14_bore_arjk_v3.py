"""CHECK 14 addendum 2 -- put set A (49 TopBrain) and the host on the SAME estimator
as set B, so the cross-set comparison is not borrowed from an earlier computation.
Also test which host mesh the published +0.38 was measured against.
"""
import glob, os, sys
import numpy as np, pyvista as pv, vtk

sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches, DualDeviceNav


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def densify(p, r, step=0.25):
    s = arclen(p)
    t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / step)) + 1, 2))
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1), np.interp(t, s, r), t


def sdf(mesh):
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(mesh)
    return lambda pts: np.array([f.EvaluateFunction(p) for p in pts])


def qs(a, d=3):
    a = np.asarray(a, float)
    return "  ".join("%7.*f" % (d, v) for v in np.percentile(a, [5, 25, 50, 75, 95]))


print("=== HOST: which mesh does the published +0.38 correspond to? ===")
dv = DualDeviceNav()
hb = [b for b in dv.vessel_tree.branches if "RCCA" in str(b.name).upper()][0]
hp = np.asarray(hb.coordinates, float); hr = np.asarray(hb.radii, float)
hq, hrr, ht = densify(hp, hr)
print("   FromMesh accessor path: %s" % dv.vessel_tree.mesh_path)
for lab, path in [("FromMesh (branch frame)", dv.vessel_tree.mesh_path)]:
    m = pv.read(path).triangulate().clean()
    f = sdf(m)
    c = f(hq)
    print("   %-26s clear med %.3f min %.3f outside %d | delta med %+.3f  p25 %+.3f p75 %+.3f"
          % (lab, np.median(c), c.min(), (c < 0).sum(), np.median(c - hrr),
             np.percentile(c - hrr, 25), np.percentile(c - hrr, 75)))
d = os.path.dirname(dv.vessel_tree.mesh_path)
print("   files beside it: %s" % sorted(os.listdir(d))[:10])
print("   NOTE: raw eve_bench/data .obj is MESH frame (HANDOFF 11.1) -- not comparable.")

print()
print("=== SET A: all TopBrain anatomies, identical estimator to set B ===")
rows = []
for a in sorted(glob.glob("/opt/eve_training/results_topbrain/anatomies/*")):
    if not os.path.isdir(a):
        continue
    try:
        m = pv.read(os.path.join(a, "vessel_architecture_collision.obj")).triangulate().clean()
        f = sdf(m)
        brs = {str(b.name): b for b in load_branches(os.path.join(a, "Centrelines_comb"))}
        k = [x for x in brs if "RCCA" in x.upper()][0]
        b = brs[k]
        p = np.asarray(b.coordinates, float); r = np.asarray(b.radii, float)
        q, rr, t = densify(p, r)
        c = f(q)
        dl = c - rr
        rows.append((os.path.basename(a), float(np.median(dl)),
                     float(np.median(dl[t < 130.0])) if (t < 130.0).sum() else None,
                     float(np.median(dl[t >= 130.0])) if (t >= 130.0).sum() else None,
                     float(dl.min()), float(np.mean(dl < 0)), float(np.median(c))))
    except Exception as e:
        print("  FAIL %s: %s" % (os.path.basename(a), e))

v = np.array([r[1] for r in rows])
print("   n = %d anatomies" % len(rows))
print("   whole-route delta      p5/p25/med/p75/p95: %s" % qs(v))
pre = np.array([r[2] for r in rows if r[2] is not None])
post = np.array([r[3] for r in rows if r[3] is not None])
print("   s < 130 (host stretch) p5/p25/med/p75/p95: %s   (n=%d)" % (qs(pre), len(pre)))
print("   s >= 130 (siphon)      p5/p25/med/p75/p95: %s   (n=%d)" % (qs(post), len(post)))
pair = np.array([r[3] - r[2] for r in rows if r[2] is not None and r[3] is not None])
print("   paired (siphon - host stretch)           : %s   | <0: %d / %d"
      % (qs(pair), int((pair < 0).sum()), len(pair)))
fo = np.array([r[5] for r in rows])
print("   optimism fraction                        : %s" % qs(fo))
mn = np.array([r[4] for r in rows])
print("   worst point delta                        : %s" % qs(mn))
srt = sorted(rows, key=lambda x: x[1])
print("   most optimistic: " + ", ".join("%s %+.3f" % (n, d) for n, d, *_ in srt[:4]))
print("   least optimistic: " + ", ".join("%s %+.3f" % (n, d) for n, d, *_ in srt[-4:][::-1]))
