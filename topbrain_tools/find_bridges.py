#!/usr/bin/env python3
"""Locate the bridges that give a vessel-tree mesh its handles.

A composed anatomy is a tree, so a handle (genus > 0) needs a place where the
surface joins two things that are not meant to touch: two different vessels
away from any junction where one branches off the other, or one vessel to
itself more than SELF_MM of arclength away. Every mesh vertex is labelled
with its owning centerline sample (the radius-aware tube measure the mesher
uses, r - |x - c|); every edge whose two ends carry incompatible labels is a
bridge edge; bridge edges are clustered in space and reported.

    python3 topbrain_tools/find_bridges.py <anatomy_dir> [--mesh file] [--margin M]

With --margin, the owned v4 union is rebuilt at that margin and its iso-surface
examined (to see what the union would produce); with --tubes, the tube-only
iso-surface; otherwise --mesh (default the anatomy's collision_full.vtp, else
its .obj) is read.
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/eve_training/eve_bench")

SELF_MM = 15.0
JUNCTION_MM = 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anatomy")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--tubes", action="store_true", help="examine the tube-only iso-surface")
    a = ap.parse_args()
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    from sdf_union_v4 import Samples, genus

    br = load_branches(os.path.join(a.anatomy, "Centrelines_comb"))
    samp = Samples(br)
    if a.tubes:
        from bake_meshes_v4 import keep_largest
        from sdf_mesher import tube_field, iso_surface, SPACING_MM
        mesh, _, _ = keep_largest(iso_surface(tube_field(br, spacing=SPACING_MM)))
    elif a.margin is not None:
        from bake_meshes_v4 import sections_for, keep_largest
        from sdf_mesher import tube_field, iso_surface, SPACING_MM
        from sdf_union_v4 import add_real_sections_owned
        secs, _ = sections_for(a.anatomy, br)
        field = tube_field(br, spacing=SPACING_MM)
        add_real_sections_owned(field, secs, samp, SPACING_MM, a.margin)
        mesh, _, _ = keep_largest(iso_surface(field))
    else:
        path = a.mesh or os.path.join(a.anatomy, "collision_full.vtp")
        if not os.path.exists(path):
            path = os.path.join(a.anatomy, "vessel_architecture_collision.obj")
        mesh = pv.read(path)
    mesh = mesh.clean().triangulate()
    g = genus(mesh)
    print("genus per component:", [c["genus"] for c in g])

    # label vertices
    tree = cKDTree(samp.p)
    d, nn = tree.query(np.asarray(mesh.points), k=24)
    f = samp.r[nn] - d
    best = nn[np.arange(len(nn)), np.argmax(f, axis=1)]
    lb, ls = samp.b[best], samp.s[best]

    # legit junctions: a branch whose START lies on another branch
    starts = {}
    for i, b in enumerate(br):
        p0 = np.asarray(b.coordinates, float)[0]
        for j, c in enumerate(br):
            if i == j:
                continue
            pc = np.asarray(c.coordinates, float)
            k = int(np.argmin(np.linalg.norm(pc - p0, axis=1)))
            if np.linalg.norm(pc[k] - p0) < 3.0:
                sc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))][k]
                starts.setdefault(i, []).append((j, sc))

    new_from = {}
    import json
    pv4 = os.path.join(a.anatomy, "provenance.json")
    if os.path.exists(pv4):
        P = json.load(open(pv4, encoding="utf-8"))
        X = P.get("xform", {})
        cut = {"RCCA": P.get("host_cut_mm", 0.0), "RECA": 0.0}
        if "left" in P:
            cut.update(LCCA=P["left"]["host_cut_mm"], LECA=0.0)
        for k, bn in (("va_R", "RVA"), ("va_L", "LVA")):
            if k in X:
                cut[bn] = X[k]["cut_mm"]
        for i, n in enumerate(samp.names):
            for bn, c0 in cut.items():
                if n.endswith("- %s" % bn) or n.endswith("- %s.mrk" % bn):
                    new_from[i] = c0

    def is_new(b, s):
        return b in new_from and s >= new_from[b]

    def legit(b1, s1, b2, s2):
        if new_from and not (is_new(b1, s1) or is_new(b2, s2)):
            return True          # host-host contact: the shipped host's own geometry
        if b1 == b2:
            return abs(s1 - s2) <= SELF_MM
        for (x, sx, y, sy) in ((b1, s1, b2, s2), (b2, s2, b1, s1)):
            for (par, spar) in starts.get(x, []):
                if par == y and sx < JUNCTION_MM and abs(sy - spar) < JUNCTION_MM:
                    return True
        # siblings starting at the same point (e.g. RCCA and RVA off the BCT end)
        for (p1, q1) in starts.get(b1, []):
            for (p2, q2) in starts.get(b2, []):
                if p1 == p2 and abs(q1 - q2) < 3.0 and s1 < JUNCTION_MM and s2 < JUNCTION_MM:
                    return True
        return False

    edges = mesh.extract_all_edges()
    lines = np.asarray(edges.lines).reshape(-1, 3)[:, 1:]
    # map edge points back to mesh vertices
    epts = np.asarray(edges.points)
    vid = cKDTree(np.asarray(mesh.points)).query(epts)[1]
    e = vid[lines]
    bad = [k for k, (u, v) in enumerate(e) if not legit(lb[u], ls[u], lb[v], ls[v])]
    print("bridge edges: %d of %d" % (len(bad), len(e)))
    if not bad:
        return 0
    mid = (np.asarray(mesh.points)[e[bad, 0]] + np.asarray(mesh.points)[e[bad, 1]]) / 2
    # cluster
    cl = -np.ones(len(bad), int); c = 0
    t2 = cKDTree(mid)
    for i in range(len(bad)):
        if cl[i] >= 0:
            continue
        stack = [i]; cl[i] = c
        while stack:
            u = stack.pop()
            for w in t2.query_ball_point(mid[u], 3.0):
                if cl[w] < 0:
                    cl[w] = c; stack.append(w)
        c += 1
    for k in range(c):
        idx = np.nonzero(cl == k)[0]
        pairs = {}
        for i in idx:
            u, v = e[bad[i]]
            key = tuple(sorted([(samp.names[lb[u]].split(" - ")[-1].replace("Centerline curve ", ""), int(ls[u] // 5) * 5),
                                (samp.names[lb[v]].split(" - ")[-1].replace("Centerline curve ", ""), int(ls[v] // 5) * 5)]))
            pairs[key] = pairs.get(key, 0) + 1
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:3]
        print("bridge cluster %d: %d edges at %s  %s" % (k, len(idx), np.round(mid[idx].mean(0), 1).tolist(),
                                                        "; ".join("%s@%d <-> %s@%d x%d" % (a_[0], a_[1], b_[0], b_[1], n) for (a_, b_), n in top)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
