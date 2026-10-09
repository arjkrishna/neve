#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\fold_fix.py (= /sp/v6/s3fix/fold_fix.py in the build containers)
"""fold_fix.py 1.0 (s3_fix, 2026-10-08; final verifier's should-fix 3) -- remove fold-over face pairs from an anatomy's
training OBJ (vessel_architecture_collision.obj).

A fold-over = two EDGE-ADJACENT triangles whose normals are opposite within 1 deg (vf_lib's criterion): the two faces
lie on each other. MeshLab's self-intersection filter skips faces that share an edge, so the strict audit cannot see it.
Fix, in order, the first whose WHOLE-OBJ validation passes:
  1. edge flip of the shared edge (the two faces (a,b,c) / (b,a,d) become (a,d,c) / (d,b,c); 2 faces change, no vertex
     moves);
  1b. spike vertex removal: an end of the folded edge with exactly 3 faces (a vertex pushed through its own 1-ring) is
     removed and its 1-ring closed with one face in the removed faces' winding (= a half-edge collapse; 3 faces -> 1);
  2. re-cut of the OBJ from collision_full by bake_meshes_v4.clean_decimate at the bake's budgets (E7's ladder: 23500,
     23450, 23400, 23300, 23200, 23000).
Validation (whole OBJ): bake_meshes_v4.is_clean (closed genus-0 two-manifold, 0 self-intersections), both routes
navigable (route_report), every centreline point inside (enclosure), 0 fold-over pairs, the VA lumen on the polyline
>= 0.65 mm on both VAs (bake_v6.va_lumen_obj). The anatomy folder is changed in place: the OBJ, mesh_v6.json
(obj stats, obj_fold_fix record) and provenance bake_v6.obj_fold_fix.
    python3 fold_fix.py <anatomy folder> <report.json>          (neve-build-meshlab)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/sp/v6/bake")
import bake_v6 as BV  # noqa: E402

B = BV.B
VERSION = "fold_fix 1.0 (2026-10-08; s3_fix)"
COS1 = np.cos(np.radians(1.0))


def faces_of(m):
    return np.asarray(m.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def folds(V, F):
    """edge-adjacent face pairs with normals opposite within 1 deg: [(f1, f2, (a, b))]."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fi = np.tile(np.arange(len(F)), 3)
    key = np.sort(E, 1)
    o = np.lexsort((key[:, 1], key[:, 0]))
    key, fi = key[o], fi[o]
    same = np.all(key[1:] == key[:-1], 1)
    i = np.nonzero(same)[0]
    f1, f2 = fi[i], fi[i + 1]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    c = (n[f1] * n[f2]).sum(1)
    k = c < -COS1
    return [(int(a), int(b), tuple(int(x) for x in key[i][j])) for a, b, j in zip(f1[k], f2[k], np.nonzero(k)[0])]


def flip(F, f1, f2, e):
    a, b = e
    t1, t2 = list(F[f1]), list(F[f2])
    # orient: t1 contains edge a->b in its winding, t2 b->a
    def rot(t, x):
        k = t.index(x)
        return t[k:] + t[:k]
    t1 = rot(t1, a)
    if t1[1] != b:
        a, b = b, a
        t1 = rot(list(F[f1]), a)
    c = t1[2]
    t2 = rot(list(F[f2]), b)
    if t2[1] != a:
        return None, "faces not consistently oriented across the edge"
    d = t2[2]
    if c == d:
        return None, "degenerate (same apex)"
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    if np.any(np.all(E == np.sort([c, d]), 1)):
        return None, "edge (c, d) already exists (flip would make it non-manifold)"
    G = F.copy()
    G[f1] = [a, d, c]
    G[f2] = [d, b, c]
    return G, "flipped edge (%d, %d) -> (%d, %d)" % (a, b, c, d)


def spike(V, F, f1, f2, e):
    """an end of the folded edge with exactly 3 faces (the two folded ones + one): a spike vertex pushed through its
    own 1-ring. Remove it (= half-edge collapse onto the other end) and close its 1-ring (3 vertices) with ONE face
    oriented as the removed faces traversed the ring. Returns (V2, F2, what) or (None, None, why)."""
    for v in e:
        fv = np.nonzero((F == v).any(1))[0]
        if len(fv) != 3 or f1 not in fv or f2 not in fv:
            continue
        ring_edges = []
        for f in fv:
            t = list(F[f])
            k = t.index(v)
            ring_edges.append((t[(k + 1) % 3], t[(k + 2) % 3]))      # the edge opposite v, in the face's winding
        nxt = dict(ring_edges)
        if len(nxt) != 3:
            continue
        x0 = ring_edges[0][0]
        loop = [x0, nxt[x0], nxt[nxt[x0]]]
        if nxt[loop[2]] != x0:
            continue
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        keep = np.ones(len(F), bool)
        keep[fv] = False
        tri = np.array(loop)
        # a face on the same three vertices must not exist already
        if np.any(np.all(np.sort(F[keep], 1) == np.sort(tri), 1)):
            continue
        G = np.vstack([F[keep], tri[None, :]])
        # drop the vertex, reindex
        old = np.arange(len(V))
        newidx = np.where(old > v, old - 1, old)
        V2 = np.delete(V, v, axis=0)
        G2 = newidx[G]
        return V2, G2, "removed spike vertex %d (3 faces %s -> 1 face %s)" % (v, [int(x) for x in fv], [int(x) for x in loop])
    return None, None, "no end of the folded edge has exactly 3 faces"


def validate(m, br, root):
    import pyvista as pv
    V, F = np.asarray(m.points, float), faces_of(m)
    rr, ok = B.route_report(m, br)
    out_n, tot_n = B.enclosure(m, br)
    clean = bool(B.is_clean(m))
    fo = folds(V, F)
    pv.save_meshio(os.path.join(root, B.MESH_NAME), m)
    lum = BV.va_lumen_obj(root)
    res = {"clean": clean, "routes_ok": bool(ok), "centreline_outside": int(out_n), "folds": len(fo), "va_lumen": lum,
           "tris": int(m.n_cells)}
    res["ok"] = bool(clean and ok and out_n == 0 and not fo and lum and min(lum.values()) >= BV.LUMEN_NEED)
    return res, rr


def main():
    import pyvista as pv
    from eve_bench.dualdevicenav import load_branches
    root, rep_out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    obj = os.path.join(root, B.MESH_NAME)
    keep = open(obj, "rb").read()
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    m0 = pv.read(obj)
    V, F = np.asarray(m0.points, float), faces_of(m0)
    fo = folds(V, F)
    rec = {"script": VERSION, "date": time.strftime("%F %T"), "folds_before": [
        {"faces": [a, b], "edge": list(e), "at": np.round(V[list(e)].mean(0), 2).tolist()} for a, b, e in fo], "tried": []}
    if not fo:
        rec["result"] = "no fold-over pair: unchanged"
        json.dump(rec, open(rep_out, "w"), indent=1)
        print(json.dumps(rec))
        return 0
    chosen = None
    G = F.copy()
    why = []
    for a, b, e in fo:
        G2, w = flip(G, a, b, e)
        why.append(w)
        if G2 is None:
            G = None
            break
        G = G2
    if G is not None:
        m1 = pv.PolyData(V, np.c_[np.full(len(G), 3), G].ravel())
        res, rr = validate(m1, br, root)
        rec["tried"].append(dict(method="edge flip", what=why, **res))
        if res["ok"]:
            chosen = ("edge flip", m1, rr, res)
    else:
        rec["tried"].append({"method": "edge flip", "what": why, "ok": False})
    if chosen is None:
        # 1b. spike vertex removal (half-edge collapse of a degree-3 end of the folded edge)
        V2, G2, ok_all, whats = V.copy(), F.copy(), True, []
        for a, b, e in fo:
            fo2 = folds(V2, G2)
            if not fo2:
                break
            a2, b2, e2 = fo2[0]
            V3, G3, w = spike(V2, G2, a2, b2, e2)
            whats.append(w)
            if V3 is None:
                ok_all = False
                break
            V2, G2 = V3, G3
        if ok_all:
            m1 = pv.PolyData(V2, np.c_[np.full(len(G2), 3), G2].ravel())
            res, rr = validate(m1, br, root)
            rec["tried"].append(dict(method="spike vertex removal", what=whats, **res))
            if res["ok"]:
                chosen = ("spike vertex removal", m1, rr, res)
        else:
            rec["tried"].append({"method": "spike vertex removal", "what": whats, "ok": False})
    if chosen is None:
        full = pv.read(os.path.join(root, B.FULL_NAME))
        for tg in (23500, 23450, 23400, 23300, 23200, 23000):
            m, path = B.clean_decimate(full, tg, raw_iso=False)
            if m is None:
                rec["tried"].append({"method": "re-cut", "target": tg, "ok": False, "why": "no clean decimation"})
                continue
            res, rr = validate(m, br, root)
            rec["tried"].append(dict(method="re-cut", target=tg, path=path, **res))
            if res["ok"]:
                chosen = ("re-cut at %d (%s)" % (tg, path), m, rr, res)
                break
    if chosen is None:
        with open(obj, "wb") as fh:
            fh.write(keep)
        rec["result"] = "no fix passed: OBJ unchanged"
        json.dump(rec, open(rep_out, "w"), indent=1)
        print(json.dumps(rec))
        return 1
    how, m, rr, res = chosen
    pv.save_meshio(obj, m)
    rec["result"] = how
    rec["after"] = res
    rec["md5_before"] = BV.hashlib.md5(keep).hexdigest()
    rec["md5_after"] = BV._md5(obj)
    rp = os.path.join(root, BV.V.REPORT_V5)
    rep = json.load(open(rp))
    import sdf_union_v4
    rep["obj"] = dict(B.mesh_stats(m), genus=sdf_union_v4.genus(m))
    rep["routes"] = rr
    rep["obj_fold_fix"] = rec
    json.dump(rep, open(rp, "w"), indent=1, default=str)
    pp = os.path.join(root, "provenance.json")
    prov = json.load(open(pp, encoding="utf-8"))
    prov["bake_v6"]["obj_fold_fix"] = rec
    prov["bake_v6"]["obj_tris"] = int(m.n_cells)
    json.dump(prov, open(pp, "w", encoding="utf-8"), indent=1)
    rec["seconds"] = round(time.time() - t0, 1)
    json.dump(rec, open(rep_out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
