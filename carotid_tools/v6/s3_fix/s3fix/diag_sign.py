#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\diag_sign.py (= /sp/v6/s3fix/diag_sign.py in the build containers)
"""s3_fix diagnosis 2: is the real-surface SDF sign right where the fragment sits?
For a section (lower_R / siphon_R) of an anatomy, as the bake loads it (bake_v6 sections_for_morph), sample rays
from the RCCA centreline at s in [lo, hi] (12 directions, 0..6 mm) and compare
  * the bake's f_real = -implicit_distance (vtkImplicitPolyDataDistance with the loaded normals) and
  * inside / outside by ray parity (select_enclosed_points) on the same surface.
Also: open / non-manifold edges of the loaded surface, faces whose normal points TOWARD the centreline (inverted
patch) within the stretch, and the same counts on the unmirrored / unmorphed load.
    python3 diag_sign.py <name> <section> <lo> <hi> <out.json> [--tree DIR]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/sp/v6/bake")
import bake_v6 as BV  # noqa: E402

B = BV.B
from scipy.spatial import cKDTree  # noqa: E402
from sdf_union import load_surface  # noqa: E402


def probe(surf, P, S, lo, hi):
    import pyvista as pv
    sel = (S >= lo) & (S <= hi)
    pts, meta = [], []
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    for i in np.nonzero(sel)[0][::4]:
        t = T[i]
        a = np.cross(t, [1, 0, 0]) if abs(t[0]) < 0.9 else np.cross(t, [0, 1, 0])
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        for k in range(12):
            ang = 2 * np.pi * k / 12
            d = np.cos(ang) * a + np.sin(ang) * b
            for rr in np.arange(0.0, 6.01, 0.25):
                pts.append(P[i] + rr * d)
                meta.append((float(S[i]), k, rr))
    X = np.asarray(pts)
    imp = np.asarray(pv.PolyData(X).compute_implicit_distance(surf)["implicit_distance"])
    enc = np.asarray(pv.PolyData(X).select_enclosed_points(surf, tolerance=0.0, check_surface=False)["SelectedPoints"], bool)
    sign_in = imp < 0
    mism = sign_in != enc
    M = np.asarray(meta)
    out = {"samples": int(len(X)), "mismatch": int(mism.sum()),
           "mismatch_inside_by_parity_but_sdf_out": int((enc & ~sign_in).sum()),
           "mismatch_outside_by_parity_but_sdf_in": int((~enc & sign_in).sum())}
    if mism.any():
        out["mismatch_s"] = [round(float(M[mism, 0].min()), 2), round(float(M[mism, 0].max()), 2)]
        out["mismatch_r"] = [round(float(M[mism, 2].min()), 2), round(float(M[mism, 2].max()), 2)]
        # per ray: radius where the parity says we leave the lumen vs the sdf
        rows = []
        for s_ in np.unique(M[:, 0]):
            for k in range(12):
                m = (M[:, 0] == s_) & (M[:, 1] == k)
                if not mism[m].any():
                    continue
                rr = M[m, 2]
                rows.append({"s": round(float(s_), 2), "dir": k,
                             "parity_in": "".join("1" if v else "0" for v in enc[m]),
                             "sdf_in": "".join("1" if v else "0" for v in sign_in[m])})
        out["rays"] = rows[:40]
    return out


def surf_stats(surf, P, S, lo, hi):
    m = surf
    fe = m.extract_feature_edges(boundary_edges=True, non_manifold_edges=True, feature_edges=False, manifold_edges=False)
    C = np.asarray(m.cell_centers().points)
    N = np.asarray(m.cell_data["Normals"]) if "Normals" in m.cell_data else np.asarray(m.compute_normals(cell_normals=True, point_normals=False).cell_data["Normals"])
    d, i = cKDTree(P).query(C)
    sel = (S[i] >= lo) & (S[i] <= hi)
    radial = C - P[i]
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-9)[:, None]
    inward = (np.einsum("ij,ij->i", N, radial) < -0.3) & sel
    return {"n_open_or_nonmanifold_edges": int(fe.n_cells), "faces_in_stretch": int(sel.sum()),
            "faces_normal_toward_centreline": int(inward.sum()),
            "inward_dist_from_centreline_mm": [round(float(np.percentile(d[inward], q)), 2) for q in (5, 50, 95)] if inward.any() else None,
            "inward_s": [round(float(S[i][inward].min()), 2), round(float(S[i][inward].max()), 2)] if inward.any() else None}


def main():
    name, sname, lo, hi, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else "/repo/carotid_data/anatomies_v6"
    root = os.path.join(tree, name)
    from eve_bench.dualdevicenav import load_branches
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    secs, _ = B.sections_for(root, br)
    sec = [s for s in secs if s.name == sname][0]
    b = B.branch(br, sec.branch)
    P = np.asarray(b.coordinates, float)
    S = B.arclen(P)
    rec = {"name": name, "section": sname, "stretch": [lo, hi], "morph": BV.MORPH.get("rec", {}).get(sname)}
    rec["bake_load"] = {"probe": probe(sec.surface, P, S, lo, hi), "surface": surf_stats(sec.surface, P, S, lo, hi)}
    prov = json.load(open(os.path.join(root, "provenance.json")))
    key = BV.XF_OF[sname]
    xf = prov["xform"][key]
    plain = load_surface(xf["surface"], xf["mirror"], xf["R"], xf["origin"], xf["anchor"])
    rec["plain_load"] = {"probe": probe(plain, P, S, lo, hi), "surface": surf_stats(plain, P, S, lo, hi)}
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1)[:5000])


if __name__ == "__main__":
    main()
