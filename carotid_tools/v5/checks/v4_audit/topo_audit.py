#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v4_build\audit\topo_audit.py (= /sp/v4_build/audit/topo_audit.py in the build containers)
"""Topology / surface-validity audit of the shipped v4 (and v3) meshes.

Independent of the baker's own genus()/is_clean(): pymeshlab (vcg) topological
measures and self-intersection selection, plus a numpy Euler / orientation /
winding-number / exact point-triangle-distance cross-check. Read-only on /repo.

    python3 /sp/audit/topo_audit.py [--only a,b] [--workers 5]
Writes /sp/audit/topo_results.jsonl (one line per anatomy).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

V4 = "/repo/carotid_data/anatomies_v4"
V3 = "/repo/carotid_data/anatomies_v3"
OUT = "/sp/audit/topo_results.jsonl"
CATH_R, CONTACT, CAP_MM = 0.35, 0.3, 3.0
ROUTES = ("RCCA", "LCCA")


# ---------------------------------------------------------------- centerlines
def load_branches(folder):
    """Same frame as eve_bench.dualdevicenav.load_points_from_json: (y, -z, -x)."""
    out = []
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith("Centerline curve ") and fn.endswith(".json")):
            continue
        d = json.load(open(os.path.join(folder, fn), encoding="utf-8"))
        P, R = [], []
        for mk in d["markups"]:
            if mk["type"] != "Curve":
                continue
            for cp in mk["controlPoints"]:
                x, y, z = (float(v) for v in cp["position"])
                P.append((y, -z, -x))
            for me in mk.get("measurements", []):
                if me["name"] == "Radius":
                    R.extend(me["controlPointValues"])
        name = os.path.splitext(os.path.splitext(fn)[0])[0]
        m = re.search(r"- (\w+)$", name)
        tag = m.group(1) if m else re.search(r"\((\d+)\)", name).group(0)
        out.append((tag, np.asarray(P, float), np.asarray(R, float)))
    return out


def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


# ---------------------------------------------------------------- mesh io
def read_obj(path):
    V, F, other = [], [], {}
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                if len(idx) != 3:
                    other["non_tri_faces"] = other.get("non_tri_faces", 0) + 1
                    continue
                F.append(idx)
            else:
                k = line.split()[0] if line.split() else ""
                if k and not k.startswith("#"):
                    other[k] = other.get(k, 0) + 1
    F = np.asarray(F, np.int64)
    neg = int((F < 0).sum())
    F = F - 1
    return np.asarray(V, float), F, dict(other, negative_indices=neg)


def read_vtp(path):
    import pyvista as pv
    m = pv.read(path)
    faces = np.asarray(m.faces)
    # all triangles?
    ok = True
    if len(faces) % 4 == 0 and (faces.reshape(-1, 4)[:, 0] == 3).all():
        F = faces.reshape(-1, 4)[:, 1:]
    else:
        ok = False
        F = np.asarray(m.triangulate().faces).reshape(-1, 4)[:, 1:]
    extra = {"n_lines": int(m.n_lines), "n_verts": int(m.n_verts), "n_strips": int(m.n_strips), "all_tri": ok}
    return np.asarray(m.points, float), np.asarray(F, np.int64), extra


# ---------------------------------------------------------------- pymeshlab
def ms_of(V, F):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(V, np.float64), face_matrix=np.asarray(F, np.int32)))
    return ms


def pml_topo(ms):
    d = ms.get_topological_measures()
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else int(v)) for k, v in d.items()}


def pml_selfint(ms):
    ms.compute_selection_by_self_intersections_per_face()
    sel = np.asarray(ms.current_mesh().face_selection_array(), bool)
    ms.set_selection_none()
    return np.nonzero(sel)[0]


# ---------------------------------------------------------------- numpy cross-checks
def euler_numpy(V, F):
    """Merge coincident vertices, then V-E+F, components, boundary/non-manifold
    edge counts and orientation coherence -- no VTK, no vcg."""
    Vu, inv = np.unique(np.round(V, 9), axis=0, return_inverse=True)
    inv = inv.ravel()
    G = inv[F]
    degenerate = int(((G[:, 0] == G[:, 1]) | (G[:, 1] == G[:, 2]) | (G[:, 0] == G[:, 2])).sum())
    e = np.concatenate([G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]])
    und = np.sort(e, axis=1)
    n = len(Vu)
    key = und[:, 0] * n + und[:, 1]
    uk, cnt = np.unique(key, return_counts=True)
    dkey = e[:, 0] * n + e[:, 1]
    _, dcnt = np.unique(dkey, return_counts=True)
    sorted_faces = np.sort(G, axis=1)
    fk = (sorted_faces[:, 0] * n + sorted_faces[:, 1]) * n + sorted_faces[:, 2]
    dup_faces = int(len(fk) - len(np.unique(fk)))
    # components over shared vertices
    rows = np.repeat(np.arange(len(G)), 3)
    A = coo_matrix((np.ones(len(rows)), (rows, G.ravel())), shape=(len(G), n)).tocsr()
    FF = (A @ A.T)
    ncomp, lab = connected_components(FF, directed=False)
    referenced = np.zeros(n, bool); referenced[G.ravel()] = True
    chi = int(referenced.sum()) - len(uk) + len(G)
    return {"V": int(referenced.sum()), "E": int(len(uk)), "F": int(len(G)), "chi": chi,
            "genus_if_closed_single": (2 - chi) / 2.0,
            "components": int(ncomp), "boundary_edges": int((cnt == 1).sum()),
            "nonmanifold_edges": int((cnt > 2).sum()),
            "misoriented_edges": int((dcnt > 1).sum()),
            "duplicate_faces": dup_faces, "degenerate_index_faces": degenerate,
            "coincident_vertex_merges": int(len(V) - n)}


def geom_numpy(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    la, lb, lc = np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)

    def ang(x, y, z):  # angle opposite side x
        cosv = np.clip((y * y + z * z - x * x) / np.maximum(2 * y * z, 1e-30), -1, 1)
        return np.degrees(np.arccos(cosv))
    amin = np.minimum(np.minimum(ang(la, lb, lc), ang(lb, lc, la)), ang(lc, la, lb))
    emax = np.maximum(np.maximum(la, lb), lc)
    return {"area_mm2": float(area.sum()), "signed_volume_mm3": vol,
            "zero_area_faces": int((area < 1e-10).sum()), "tiny_area_faces_1e-4": int((area < 1e-4).sum()),
            "min_angle_deg": float(amin.min()), "p01_min_angle_deg": float(np.percentile(amin, 1)),
            "faces_minangle_lt1": int((amin < 1.0).sum()), "faces_minangle_lt5": int((amin < 5.0).sum()),
            "min_edge_mm": float(np.minimum(np.minimum(la, lb), lc).min()), "max_edge_mm": float(emax.max()),
            "median_edge_mm": float(np.median(np.r_[la, lb, lc]))}


def winding(P, V, F, chunk=40):
    A0, B0, C0 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    W = np.empty(len(P))
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk, None, :]
        a, b, c = A0[None] - p, B0[None] - p, C0[None] - p
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la \
            + np.einsum("ijk,ijk->ij", c, a) * lb
        W[i:i + chunk] = (2 * np.arctan2(det, den)).sum(1) / (4 * np.pi)
    return W


def pt_tri_dist(p, a, b, c):
    """Exact distance from point p to triangles (a,b,c) (arrays m x 3). Ericson 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    q = a + ab * v[:, None] + ac * w[:, None]          # interior
    # edges / vertices
    def seg(p, x, y):
        d = y - x
        t = np.clip(((p - x) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
        return x + d * t[:, None]
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    cands = [np.where(inside[:, None], q, np.inf), seg(p, a, b), seg(p, b, c), seg(p, c, a)]
    dd = np.min(np.stack([np.linalg.norm(p - x, axis=1) for x in cands]), axis=0)
    return dd


def surface_dist(P, V, F):
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cen = (A + B + C) / 3.0
    rad = np.max(np.stack([np.linalg.norm(x - cen, axis=1) for x in (A, B, C)]), 0)
    rmax = float(rad.max())
    ct = cKDTree(cen)
    vt = cKDTree(V)
    dv, _ = vt.query(P)
    out = np.empty(len(P))
    for i, p in enumerate(P):
        idx = ct.query_ball_point(p, dv[i] + rmax + 1e-6)
        idx = np.asarray(idx, int)
        out[i] = pt_tri_dist(p[None], A[idx], B[idx], C[idx]).min()
    return out


def route_check(V, F, br):
    res = {}
    wsign = []
    for tag in ROUTES:
        hit = [b for b in br if b[0] == tag]
        if not hit:
            res[tag] = {"missing": True}
            continue
        _, p, r = hit[0]
        s = arclen(p)
        body = (s > CAP_MM) & (s < s[-1] - CAP_MM)
        W = winding(p, V, F)
        wsign.append(float(np.median(W)))
        ins = np.abs(W) > 0.5
        d = surface_dist(p, V, F)
        ok = ins & body
        if ok.any():
            i = int(np.argmin(np.where(ok, d, np.inf)))
            res[tag] = {"lumen_min_mm": round(float(d[i]), 3), "lumen_min_at_mm": round(float(s[i]), 1),
                        "route_pts_outside": int((~ins & body).sum()),
                        "navigable": bool(d[ok].min() - CONTACT >= CATH_R),
                        "median_deficit_mm": round(float(np.median(r[ok] - d[ok])), 3) if len(r) == len(p) else None}
        else:
            res[tag] = {"route_pts_outside": int(body.sum()), "navigable": False}
    return res, wsign


def enclosure(V, F, br):
    pts = []
    for _, p, _ in br:
        s = arclen(p)
        pts.append(p[(s > CAP_MM) & (s < s[-1] - CAP_MM)])
    P = np.vstack([p for p in pts if len(p)])
    W = winding(P, V, F)
    out = np.abs(W) <= 0.5
    return int(out.sum()), int(len(P)), P[out][:20].round(2).tolist(), float(np.median(W))


# ---------------------------------------------------------------- locating
class Locator:
    def __init__(self, br, spans=None):
        P, B, S, R = [], [], [], []
        for k, (tag, p, r) in enumerate(br):
            P.append(p); B.append(np.full(len(p), k)); S.append(arclen(p))
            R.append(r if len(r) == len(p) else np.full(len(p), np.nan))
        self.br = br
        self.P, self.B, self.S, self.R = np.vstack(P), np.concatenate(B), np.concatenate(S), np.concatenate(R)
        self.tree = cKDTree(self.P)
        self.route = {}
        for tag in ROUTES:
            h = [x for x in br if x[0] == tag]
            if h:
                self.route[tag] = (cKDTree(h[0][1]), arclen(h[0][1]), h[0][2])
        self.spans = spans or {}

    def section(self, tag, s):
        for name, sp in self.spans.items():
            if isinstance(sp, list) and len(sp) == 3 and sp[0] == tag and sp[1] - 1e-6 <= s <= sp[2] + 1e-6:
                return name
        return "tube"

    def locate(self, X):
        d, i = self.tree.query(X)
        out = []
        for j, x in enumerate(X):
            k = int(self.B[i[j]])
            tag = self.br[k][0]
            s = float(self.S[i[j]])
            rec = {"branch": tag, "s_mm": round(s, 1), "d_cl_mm": round(float(d[j]), 2),
                   "r_cl_mm": round(float(self.R[i[j]]), 2), "section": self.section(tag, s)}
            for rt, (tr, sr, rr) in self.route.items():
                dr, ir = tr.query(x)
                rloc = float(rr[ir]) if len(rr) == len(sr) else float("nan")
                rec["d_%s_mm" % rt] = round(float(dr), 2)
                rec["r_%s_mm" % rt] = round(rloc, 2)
                rec["s_%s_mm" % rt] = round(float(sr[ir]), 1)
            out.append(rec)
        return out


def clusters(V, F, fidx, link_mm=2.0):
    if len(fidx) == 0:
        return []
    c = V[F[fidx]].mean(1)
    t = cKDTree(c)
    pairs = t.query_pairs(link_mm, output_type="ndarray")
    n = len(fidx)
    if len(pairs):
        M = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(M, directed=False)
    else:
        lab = np.arange(n)
    out = []
    for L in np.unique(lab):
        m = lab == L
        out.append({"n_faces": int(m.sum()), "centroid": c[m].mean(0).round(2).tolist(),
                    "extent_mm": round(float(np.ptp(c[m], axis=0).max()) if m.sum() > 1 else 0.0, 2)})
    return sorted(out, key=lambda x: -x["n_faces"])


# ---------------------------------------------------------------- per mesh
def check_mesh(V, F, loc=None, want_self=True):
    r = {}
    ms = ms_of(V, F)
    r["pml_raw"] = pml_topo(ms)
    si = pml_selfint(ms) if want_self else np.array([], int)
    ms2 = ms_of(V, F)
    ms2.meshing_remove_duplicate_vertices()
    r["pml_merged"] = pml_topo(ms2)
    r["merged_differs"] = r["pml_merged"] != r["pml_raw"]
    if r["merged_differs"] and want_self:
        m2 = ms2.current_mesh()
        si2 = pml_selfint(ms2)
        r["selfint_faces_merged"] = int(len(si2))
    ms3 = ms_of(V, F)
    nf = ms3.current_mesh().face_number()
    ms3.meshing_remove_duplicate_faces()
    r["pml_dup_faces_removed"] = int(nf - ms3.current_mesh().face_number())
    ms4 = ms_of(V, F)
    ms4.meshing_remove_null_faces()
    r["pml_null_faces_removed"] = int(nf - ms4.current_mesh().face_number())
    r["numpy"] = euler_numpy(V, F)
    r["geom"] = geom_numpy(V, F)
    r["selfint_faces"] = int(len(si))
    if len(si):
        cl = clusters(V, F, si)
        if loc is not None:
            L = loc.locate(np.asarray([c["centroid"] for c in cl]))
            for c, l in zip(cl, L):
                c.update(l)
        r["selfint_sites"] = cl
    return r


def is_clean_pml(t):
    return (t["connected_components_number"] == 1 and t["genus"] == 0 and t["boundary_edges"] == 0
            and t["is_mesh_two_manifold"] and t["non_two_manifold_edges"] == 0
            and t["non_two_manifold_vertices"] == 0 and t.get("number_holes", 0) == 0)


def one(name):
    t0 = time.time()
    rec = {"name": name}
    try:
        root = os.path.join(V4, name)
        rep = json.load(open(os.path.join(root, "mesh_v4.json"), encoding="utf-8"))
        rec["claim"] = {"pass": rep.get("pass"), "obj": rep.get("obj"), "full": rep.get("full"),
                        "routes": rep.get("routes"), "decimation": rep.get("decimation"),
                        "centerline_pts_outside": rep.get("centerline_pts_outside"),
                        "centerline_pts_checked": rep.get("centerline_pts_checked"),
                        "union": rep.get("union"), "margin_mm": rep.get("margin_mm"),
                        "sections_tube_only": rep.get("sections_tube_only"),
                        "suppressed": rep.get("suppressed_stretches_mm"), "obj_tris_budget": rep.get("obj_tris_budget")}
        br = load_branches(os.path.join(root, "Centrelines_comb"))
        loc = Locator(br, rep.get("spans"))
        V, F, extra = read_obj(os.path.join(root, "vessel_architecture_collision.obj"))
        rec["v4_obj_file"] = dict(extra, v=int(len(V)), f=int(len(F)))
        rec["v4_obj"] = check_mesh(V, F, loc)
        rec["v4_obj"]["clean_pml"] = is_clean_pml(rec["v4_obj"]["pml_raw"])
        rr, ws = route_check(V, F, br)
        rec["v4_obj_routes"] = rr
        o, n, where, wmed = enclosure(V, F, br)
        rec["v4_obj_enclosure"] = {"outside": o, "checked": n, "where": where, "median_winding": wmed}
        # also compare pymeshlab's own loader (what a vcg-based consumer sees)
        import pymeshlab
        msl = pymeshlab.MeshSet()
        msl.load_new_mesh(os.path.join(root, "vessel_architecture_collision.obj"))
        rec["v4_obj_pml_loader"] = pml_topo(msl)
        Vf, Ff, ex = read_vtp(os.path.join(root, "collision_full.vtp"))
        rec["v4_full_file"] = dict(ex, v=int(len(Vf)), f=int(len(Ff)))
        rec["v4_full"] = check_mesh(Vf, Ff, loc)
        rec["v4_full"]["clean_pml"] = is_clean_pml(rec["v4_full"]["pml_raw"])
        # v3 context
        r3 = os.path.join(V3, name)
        br3 = load_branches(os.path.join(r3, "Centrelines_comb"))
        loc3 = Locator(br3, None)
        V3m, F3m, ex3 = read_obj(os.path.join(r3, "vessel_architecture_collision.obj"))
        rec["v3_obj_file"] = dict(ex3, v=int(len(V3m)), f=int(len(F3m)))
        rec["v3_obj"] = check_mesh(V3m, F3m, loc3)
        rec["v3_obj"]["clean_pml"] = is_clean_pml(rec["v3_obj"]["pml_raw"])
        try:
            rec["v3_claim"] = json.load(open(os.path.join(r3, "mesh_v3.json"), encoding="utf-8")).get("obj")
        except Exception:
            pass
    except Exception as e:  # noqa
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        rec["tb"] = traceback.format_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(V4, "*", "mesh_v4.json")))
    if a.only:
        want = a.only.split(",")
        names = [n for n in names if n in want]
    print("auditing %d anatomies" % len(names), flush=True)
    done = set()
    if os.path.exists(a.out) and not a.only:
        for line in open(a.out):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["name"])
            except Exception:
                pass
    todo = [n for n in names if n not in done]
    print("todo %d (resuming past %d)" % (len(todo), len(done)), flush=True)
    from multiprocessing import Pool
    with open(a.out, "a") as fo, Pool(a.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(one, todo)):
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            v = rec.get("v4_obj", {})
            print("%3d %-44s %5.1fs obj_clean=%s selfint obj/full/v3=%s/%s/%s %s" % (
                k, rec["name"][:44], rec["seconds"], v.get("clean_pml"), v.get("selfint_faces"),
                rec.get("v4_full", {}).get("selfint_faces"), rec.get("v3_obj", {}).get("selfint_faces"),
                rec.get("error", "")), flush=True)


if __name__ == "__main__":
    main()
