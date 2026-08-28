"""Follow-up: is the HOST VISUAL mesh a valid calibration of the declared radii?
Sweep all 25 cohort anatomies with the same ray-cast estimator.
"""
import os
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches   # noqa

ROOT = "/opt/eve_training/results_topbrain/anatomies"
NDIR = 36


def tri(mesh):
    m = mesh.triangulate().clean()
    return m.points.astype(np.float64), m.faces.reshape(-1, 4)[:, 1:]


def arclen(c):
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))))


def frames(c):
    t = np.zeros_like(c)
    t[1:-1] = c[2:] - c[:-2]
    t[0] = c[1] - c[0]
    t[-1] = c[-1] - c[-2]
    t /= np.linalg.norm(t, axis=1)[:, None]
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(c), 1))
    bad = np.abs((ref * t).sum(1)) > 0.9
    ref[bad] = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    return t, u, np.cross(t, u)


def cast(O, D, A, E1, E2):
    pvec = np.cross(D[:, None, :], E2[None, :, :])
    det = (E1[None, :, :] * pvec).sum(-1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = O[None, :] - A
    u = (tvec[None, :, :] * pvec).sum(-1) * inv
    q = np.cross(tvec, E1)
    v = (D[:, None, :] * q[None, :, :]).sum(-1) * inv
    t = ((E2 * q).sum(-1))[None, :] * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
    return np.where(hit, t, np.inf).min(axis=1)


def prof(pts, faces, coords, radii, crop=30.0):
    A = pts[faces[:, 0]]
    E1 = pts[faces[:, 1]] - A
    E2 = pts[faces[:, 2]] - A
    ctree = cKDTree(A + (E1 + E2) / 3.0)
    _, u, v = frames(coords)
    ang = np.arange(NDIR) * 2 * np.pi / NDIR
    s = arclen(coords)
    rows = []
    for i in range(len(coords)):
        idx = np.asarray(ctree.query_ball_point(coords[i], crop))
        if len(idx) < 3:
            continue
        D = np.cos(ang)[:, None] * u[i][None, :] + np.sin(ang)[:, None] * v[i][None, :]
        d = cast(coords[i], D, A[idx], E1[idx], E2[idx])
        dd = d[np.isfinite(d)]
        if len(dd) < NDIR // 2:
            continue
        rows.append((s[i], radii[i], dd.min(), np.median(dd), dd.max()))
    return np.array(rows)


def getbr(branches, key="RCCA"):
    b = [x for x in branches if key in str(x.name).upper()][0]
    return np.asarray(b.coordinates, float), np.asarray(b.radii, float)


ddn = DualDeviceNav()
vt = ddn.vessel_tree
hb = vt.branches
hc, hr = getbr(hb)
s = arclen(hc)

pv_, fv_ = tri(pv.read(vt.visu_mesh_path))
rv = prof(pv_, fv_, hc, hr)
pc_, fc_ = tri(pv.read(vt.mesh_path))
rc = prof(pc_, fc_, hc, hr)

ratio_v = rv[:, 3] / rv[:, 1]
ratio_c = rc[:, 3] / rc[:, 1]
print("HOST VISUAL  Rmed/stated_r : median %.4f  mean %.4f  CV %.4f  p10 %.3f p90 %.3f"
      % (np.median(ratio_v), ratio_v.mean(), ratio_v.std() / ratio_v.mean(),
         np.percentile(ratio_v, 10), np.percentile(ratio_v, 90)))
print("HOST COLLIS  Rmed/stated_r : median %.4f  mean %.4f  CV %.4f  p10 %.3f p90 %.3f"
      % (np.median(ratio_c), ratio_c.mean(), ratio_c.std() / ratio_c.mean(),
         np.percentile(ratio_c, 10), np.percentile(ratio_c, 90)))
print("Pearson r(stated_r, visual Rmed)  = %.4f  (n=%d)"
      % (np.corrcoef(rv[:, 1], rv[:, 3])[0, 1], len(rv)))
print("Pearson r(stated_r, collis Rmed)  = %.4f" % np.corrcoef(rc[:, 1], rc[:, 3])[0, 1])
print()
print("arc_mm  stated_r  visual_Rmed  collis_Rmed  vis/r  col/r")
for a in range(0, 240, 20):
    m = (rv[:, 0] >= a) & (rv[:, 0] < a + 20)
    m2 = (rc[:, 0] >= a) & (rc[:, 0] < a + 20)
    if m.sum() < 2:
        continue
    print("%6d  %8.3f  %11.3f  %11.3f  %5.2f  %5.2f"
          % (a, np.median(rv[m, 1]), np.median(rv[m, 3]), np.median(rc[m2, 3]),
             np.median(rv[m, 3] / rv[m, 1]), np.median(rc[m2, 3] / rc[m2, 1])))

print()
print("ALL 25 COHORT (ray-cast, RCCA)")
print("%-15s %6s %6s %6s %6s %7s %7s %7s %7s"
      % ("anatomy", "Rmed/r", "Rmin/r", "Rmax/r", "mn/mx", "r_PROX", "r_DIST",
         "Rmed_D", "minR_mm"))
acc = []
for aid in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, aid)
    if not os.path.isdir(d):
        continue
    br = load_branches(os.path.join(d, "Centrelines_comb"))
    cc, rr = getbr(br)
    p, f = tri(pv.read(os.path.join(d, "vessel_architecture_collision.obj")))
    r = prof(p, f, cc, rr)
    dm = r[:, 0] >= 137.0
    pm = r[:, 0] < 137.0
    row = (np.median(r[:, 3] / r[:, 1]), np.median(r[:, 2] / r[:, 1]),
           np.median(r[:, 4] / r[:, 1]))
    acc.append(row)
    print("%-15s %6.3f %6.3f %6.3f %6.3f %7.3f %7.3f %7.3f %7.3f"
          % (aid, row[0], row[1], row[2], row[1] / row[2],
             np.median(r[pm, 1]), np.median(r[dm, 1]), np.median(r[dm, 3]),
             r[:, 2].min()))
a = np.array(acc)
print("RANGE  Rmed/r %.3f-%.3f   Rmin/r %.3f-%.3f   Rmax/r %.3f-%.3f"
      % (a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max(),
         a[:, 2].min(), a[:, 2].max()))
print()
print("HOST for comparison: stated_r PROX %.3f DIST %.3f | visual Rmed DIST %.3f mm"
      % (np.median(rv[rv[:, 0] < 137, 1]), np.median(rv[rv[:, 0] >= 137, 1]),
         np.median(rv[rv[:, 0] >= 137, 3])))
