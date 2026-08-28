"""ATTACK THE CONTROL.

T1  code-path equivalence: raw host .obj rotated by hand  vs  FromMesh temp file.
    cohort .obj round-tripped through pv.read -> rotate(0) -> save_meshio -> pv.read.
T2  contamination-free wall by RAY CASTING (first hit, Moller-Trumbore) on RCCA.
    host collision, host VISUAL, cohort exemplars.
T3  vertex ring done properly: each vertex assigned to its nearest station over ALL
    branches; keep the RCCA-owned ones; ratio d/stated_r.
"""
import os
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import DualDeviceNav, load_branches   # noqa
from eve.intervention.vesseltree.util.meshing import (             # noqa
    load_mesh, rotate_mesh, scale_mesh, save_mesh, get_temp_mesh_path)

ROOT = "/opt/eve_training/results_topbrain/anatomies"
HOSTDIR = "/opt/eve_training/eve_bench/data/dualdevicenav"
NDIR = 36
np.set_printoptions(suppress=True)


def tri(mesh):
    m = mesh.triangulate().clean()
    f = m.faces.reshape(-1, 4)[:, 1:]
    return m.points.astype(np.float64), f


def stats(m, name):
    m2 = m.triangulate().clean()
    print("  %-26s pts=%6d cells=%6d | tri/clean pts=%6d cells=%6d area=%12.4f vol=%12.4f"
          % (name, m.n_points, m.n_cells, m2.n_points, m2.n_cells, m2.area, m2.volume))
    b = np.array(m2.bounds).reshape(3, 2)
    print("     bounds x[%9.4f,%9.4f] y[%9.4f,%9.4f] z[%9.4f,%9.4f]"
          % (b[0, 0], b[0, 1], b[1, 0], b[1, 1], b[2, 0], b[2, 1]))
    return m2


def hausdorff(a, b):
    d1, _ = cKDTree(b).query(a)
    d2, _ = cKDTree(a).query(b)
    return float(max(d1.max(), d2.max()))


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
    v = np.cross(t, u)
    return t, u, v


def raycast_station(O, D, A, E1, E2):
    pvec = np.cross(D[:, None, :], E2[None, :, :])
    det = (E1[None, :, :] * pvec).sum(-1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = O[None, :] - A
    u = (tvec[None, :, :] * pvec).sum(-1) * inv
    qvec = np.cross(tvec, E1)
    v = (D[:, None, :] * qvec[None, :, :]).sum(-1) * inv
    t = ((E2 * qvec).sum(-1))[None, :] * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
    t = np.where(hit, t, np.inf)
    return t.min(axis=1)


def ray_profile(pts, faces, coords, radii, crop=30.0):
    A = pts[faces[:, 0]]
    B = pts[faces[:, 1]]
    C = pts[faces[:, 2]]
    E1 = B - A
    E2 = C - A
    ctr = (A + B + C) / 3.0
    ctree = cKDTree(ctr)
    t, u, v = frames(coords)
    ang = np.arange(NDIR) * 2 * np.pi / NDIR
    s = arclen(coords)
    rows = []
    for i in range(len(coords)):
        idx = ctree.query_ball_point(coords[i], crop)
        if len(idx) < 3:
            continue
        idx = np.asarray(idx)
        D = np.cos(ang)[:, None] * u[i][None, :] + np.sin(ang)[:, None] * v[i][None, :]
        d = raycast_station(coords[i], D, A[idx], E1[idx], E2[idx])
        nmiss = int(np.isinf(d).sum())
        dd = d[np.isfinite(d)]
        if len(dd) < NDIR // 2:
            continue
        rows.append((s[i], radii[i], dd.min(), np.median(dd), dd.max(), nmiss))
    return np.array(rows)


def report_ray(r, label):
    print("  == %s" % label)
    if len(r) == 0:
        print("     no stations")
        return

    def blk(m, tag):
        if m.sum() < 3:
            print("    %-6s n<3" % tag)
            return
        q = r[m]
        rmin = np.median(q[:, 2] / q[:, 1])
        rmed = np.median(q[:, 3] / q[:, 1])
        rmax = np.median(q[:, 4] / q[:, 1])
        print("    %-6s n=%4d stated_r=%.3f Rmin/r=%.3f Rmed/r=%.3f Rmax/r=%.3f "
              "Rmed_mm=%.3f Rmin_mm=%.3f min/max=%.3f miss/stn=%.2f"
              % (tag, m.sum(), np.median(q[:, 1]), rmin, rmed, rmax,
                 np.median(q[:, 3]), np.median(q[:, 2]), rmin / rmax, q[:, 5].mean()))

    blk(np.ones(len(r), bool), "ALL")
    blk(r[:, 0] < 137.0, "PROX")
    blk(r[:, 0] >= 137.0, "DIST")
    out = []
    for lo in np.arange(0, r[:, 0].max(), 20):
        m = (r[:, 0] >= lo) & (r[:, 0] < lo + 20)
        if m.sum() > 2:
            out.append("%d:%.2f" % (lo, np.median(r[m, 3] / r[m, 1])))
    print("    Rmed/r by 20mm bin: " + " ".join(out))


def vertex_ring(pts, branches, rcca_key="RCCA"):
    allc, allr, allb = [], [], []
    for k, b in enumerate(branches):
        c = np.asarray(b.coordinates, float)
        rr = np.asarray(getattr(b, "radii", np.full(len(c), np.nan)), float)
        allc.append(c)
        allr.append(rr)
        allb.append(np.full(len(c), k))
    allc = np.vstack(allc)
    allr = np.concatenate(allr)
    allb = np.concatenate(allb)
    ki = [i for i, b in enumerate(branches) if rcca_key in str(b.name).upper()][0]
    d, j = cKDTree(allc).query(pts)
    m = allb[j] == ki
    if m.sum() < 10:
        return None
    ratio = d[m] / allr[j[m]]
    return int(m.sum()), np.round(np.percentile(ratio, [5, 25, 50, 75, 95]), 3)


def getbr(branches, key="RCCA"):
    b = [x for x in branches if key in str(x.name).upper()][0]
    return np.asarray(b.coordinates, float), np.asarray(b.radii, float)


print("=" * 100)
print("T1  CODE-PATH EQUIVALENCE")
print("=" * 100)
ddn = DualDeviceNav()
vt = ddn.vessel_tree
rawp = os.path.join(HOSTDIR, "vessel_architecture_collision.obj")
raw = pv.read(rawp)
print(" host raw .obj on disk:")
raw2 = stats(raw, "raw")
manual = load_mesh(rawp)
manual = rotate_mesh(manual, [90, -90, 0])
manual = scale_mesh(manual, [1.0, 1.0, 1.0])
print(" manual rotate (identical funcs):")
man2 = stats(manual, "manual-rot")
tmp = pv.read(vt.mesh_path)
print(" FromMesh temp file (THE CONTROL PATH):")
tmp2 = stats(tmp, "vt.mesh_path")
print("  hausdorff(manual-rot, vt.mesh_path) = %.6g mm" % hausdorff(man2.points, tmp2.points))
print("  area ratio temp/manual = %.9f   vol ratio = %.9f"
      % (tmp2.area / man2.area, tmp2.volume / man2.volume))

cdir = os.path.join(ROOT, "topcow_mr_001")
cobj = os.path.join(cdir, "vessel_architecture_collision.obj")
c0 = pv.read(cobj)
print(" cohort raw .obj:")
c0t = stats(c0, "cohort raw")
c1 = load_mesh(cobj)
c1 = rotate_mesh(c1, [0, 0, 0])
c1 = scale_mesh(c1, [1, 1, 1])
tp = get_temp_mesh_path("rt")
save_mesh(c1, tp)
c2 = pv.read(tp)
print(" cohort after pv.read->rotate0->save_meshio->pv.read:")
c2t = stats(c2, "cohort roundtrip")
print("  hausdorff(cohort raw, roundtrip) = %.6g mm" % hausdorff(c0t.points, c2t.points))

print()
print("=" * 100)
print("T2  RAY-CAST WALL (first hit, %d directions/station)" % NDIR)
print("=" * 100)
hb = vt.branches
hc, hr = getbr(hb)
print(" host RCCA: %d stations, arclen %.1f mm, stated r %.2f-%.2f med %.2f, nbranches=%d"
      % (len(hc), arclen(hc)[-1], hr.min(), hr.max(), np.median(hr), len(hb)))
p, f = tri(pv.read(vt.mesh_path))
report_ray(ray_profile(p, f, hc, hr), "HOST COLLISION (vt.mesh_path)")
p, f = tri(pv.read(vt.visu_mesh_path))
report_ray(ray_profile(p, f, hc, hr), "HOST VISUAL (vt.visu_mesh_path)")

for aid in ["topcow_mr_001", "topcow_mr_005", "topcow_mr_013"]:
    d = os.path.join(ROOT, aid)
    br = load_branches(os.path.join(d, "Centrelines_comb"))
    cc, rr = getbr(br)
    p, f = tri(pv.read(os.path.join(d, "vessel_architecture_collision.obj")))
    print(" %s: %d stations, arclen %.1f, stated r med %.2f, nbranches=%d"
          % (aid, len(cc), arclen(cc)[-1], np.median(rr), len(br)))
    report_ray(ray_profile(p, f, cc, rr), "COHORT " + aid)

print()
print("=" * 100)
print("T3  VERTEX RING (nearest-station ownership over ALL branches)")
print("=" * 100)
p, f = tri(pv.read(vt.mesh_path))
print(" host collision :", vertex_ring(p, hb))
p, f = tri(pv.read(vt.visu_mesh_path))
print(" host visual    :", vertex_ring(p, hb))
for aid in ["topcow_mr_001", "topcow_mr_005", "topcow_mr_013"]:
    d = os.path.join(ROOT, aid)
    br = load_branches(os.path.join(d, "Centrelines_comb"))
    p, f = tri(pv.read(os.path.join(d, "vessel_architecture_collision.obj")))
    print(" %-14s:" % aid, vertex_ring(p, br))
print("(n, percentiles 5/25/50/75/95 of vertex-distance / stated radius)")
