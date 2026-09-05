#!/usr/bin/env python3
"""Analytic swept tubes instead of marching cubes: prototype and measure.

Each branch becomes a polygonal tube -- N-gon rings at 1 mm along the
centerline in a parallel-transport (rotation-minimising) frame, radius exactly
as declared, capped at both ends. Then every triangle whose centroid lies
inside ANOTHER branch's tube is dropped, so no wall is left standing inside a
lumen at a junction. What remains is the union's outer skin with a ring-shaped
seam at each junction where the two trimmed boundaries meet.

Measured against the SDF mesh on the same anatomy: lumen along the route,
triangle count, open (seam) edges, and how wide the seams are.

    python3 /opt/eve_training/saved/mesher_probe/sweep_proto.py
"""
import os
import sys
import time

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve_bench")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
from sdf_mesher import route_lumen, mesh_stats

ANAT = "/opt/eve_training/topbrain_data/anatomies_v2_test/topcow_mr_001"
OUT = "/opt/eve_training/saved/mesher_probe"


def resample(c, r, step):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    q = np.arange(0.0, d[-1] + 1e-9, step)
    if q[-1] < d[-1] - 1e-6:
        q = np.r_[q, d[-1]]
    return np.stack([np.interp(q, d, c[:, i]) for i in range(3)], 1), np.interp(q, d, r)


def pt_frames(p):
    """Parallel-transport normals along a polyline: no roll, no Frenet flips."""
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.zeros_like(p)
    a = np.array([1.0, 0, 0]) if abs(t[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    n[0] = a - t[0] * (a @ t[0]); n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(p)):
        v = n[i - 1] - t[i] * (n[i - 1] @ t[i])
        nv = np.linalg.norm(v)
        n[i] = v / nv if nv > 1e-9 else n[i - 1]
    b = np.cross(t, n)
    return n, b


def sweep(c, r, nsides=8, step=1.0):
    p, rr = resample(c, r, step)
    n, b = pt_frames(p)
    th = np.linspace(0, 2 * np.pi, nsides, endpoint=False)
    rings = p[:, None, :] + rr[:, None, None] * (np.cos(th)[None, :, None] * n[:, None, :]
                                                  + np.sin(th)[None, :, None] * b[:, None, :])
    V = rings.reshape(-1, 3)
    F = []
    for i in range(len(p) - 1):
        for j in range(nsides):
            a0, a1 = i * nsides + j, i * nsides + (j + 1) % nsides
            b0, b1 = a0 + nsides, a1 + nsides
            F.append([a0, b0, b1]); F.append([a0, b1, a1])
    # caps: centre vertex + fan
    c0, c1 = len(V), len(V) + 1
    V = np.vstack([V, p[0], p[-1]])
    last = (len(p) - 1) * nsides
    for j in range(nsides):
        F.append([c0, (j + 1) % nsides, j])
        F.append([c1, last + j, last + (j + 1) % nsides])
    return V, np.array(F)


def inside_others(pts, others, k=8):
    """max over other branches of (r_nearest - dist): >0 means inside one."""
    best = np.full(len(pts), -1e9)
    for c, r in others:
        d, nn = cKDTree(c).query(pts, k=min(k, len(c)))
        if d.ndim == 1:
            d, nn = d[:, None], nn[:, None]
        best = np.maximum(best, (r[nn] - d).max(axis=1))
    return best


def build(branches, nsides=8, step=1.0, trim=True):
    dense = [resample(np.asarray(b.coordinates, float), np.asarray(b.radii, float), 0.25) for b in branches]
    parts = []
    for i, b in enumerate(branches):
        V, F = sweep(np.asarray(b.coordinates, float), np.asarray(b.radii, float), nsides, step)
        if trim:
            others = [dense[j] for j in range(len(branches)) if j != i]
            cen = V[F].mean(axis=1)
            keep = inside_others(cen, others) <= 0.0
            F = F[keep]
        parts.append(pv.PolyData(V, np.c_[np.full(len(F), 3), F]))
    m = parts[0]
    for q in parts[1:]:
        m = m + q
    return m.clean()


def main():
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(ANAT, "Centrelines_comb"))
    rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
    route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
    ref = pv.read(os.path.join(ANAT, "vessel_architecture_collision.obj"))
    print("%-26s %7s %6s %6s %9s %9s %9s %6s" % ("mesh", "tris", "comps", "open", "lumen_min", "deficit", "p95_def", "sec"))
    rows = [("sdf v2 obj 20k", ref, 0.0)]
    for nsides, step in ((8, 1.0), (10, 1.0), (8, 2.0), (12, 0.75)):
        t0 = time.time(); m = build(br, nsides, step); dt = time.time() - t0
        rows.append(("sweep N%d s%.2f trimmed" % (nsides, step), m, dt))
        m.save(os.path.join(OUT, "sweep_N%d_s%.2f.vtp" % (nsides, step)))
    t0 = time.time(); m = build(br, 8, 1.0, trim=False); rows.append(("sweep N8 s1 UNtrimmed", m, time.time() - t0))
    for lab, m, dt in rows:
        d, ins, body, s = route_lumen(m, route)
        # distance only: a trimmed sweep is not a closed solid, enclosure is meaningless
        dd = rad[body] - d[body]
        print("%-26s %7d %6d %6d %9.2f %9.2f %9.2f %6.1f" % (
            lab, m.n_cells, mesh_stats(m)["comps"], m.n_open_edges, d[body].min(),
            np.median(dd), np.percentile(dd, 95), dt), flush=True)
    # seam width: for each boundary edge, distance to the nearest boundary edge NOT sharing its vertices
    m = build(br, 8, 1.0)
    e = m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    if e.n_cells:
        cen = e.cell_centers().points
        d, _ = cKDTree(cen).query(cen, k=4)
        print("\nseam edges: %d, total length %.0f mm, median gap between neighbouring seam edges %.2f mm, p95 %.2f"
              % (e.n_cells, e.length, np.median(d[:, 1]), np.percentile(d[:, 3], 95)))
    pv.save_meshio(os.path.join(OUT, "sweep_N8_s1.obj"), m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
