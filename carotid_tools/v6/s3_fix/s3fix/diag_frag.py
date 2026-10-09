#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\diag_frag.py (= /sp/v6/s3fix/diag_frag.py in the build containers)
"""s3_fix diagnosis: rebuild an anatomy's FINAL bake union (bake_v6 1.1, the state recorded in its mesh_v6.json:
margins, suppressed stretches, tube-only sections) with bake_v5b's own UnionCache, and list what keep_largest drops.
For every dropped component: cells, volume, centroid, its owning curve by tube distance, RCCA arclength range, how far
it sits off the RCCA tube, and how much of it lies on the right-route piece's own scanned surface. For the fragments on
the RCCA: why the voxels between the fragment and the tube were refused (ownership margin: which competitor; window;
outside the section stretch).
    python3 diag_frag.py <name> <out.json> [--tree DIR]   (neve-build-meshlab)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/sp/v6/bake")
import bake_v6 as BV  # noqa: E402

V = BV.V
B = BV.B
from scipy.spatial import cKDTree  # noqa: E402
from sdf_mesher import iso_surface  # noqa: E402
from sdf_union_v4 import Samples, WINDOW_MM, _tube  # noqa: E402


def main():
    name, out = sys.argv[1], sys.argv[2]
    tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else "/repo/carotid_data/anatomies_v6"
    root = os.path.join(tree, name)
    t0 = time.time()
    rep6 = json.load(open(os.path.join(root, "mesh_v6.json")))
    from eve_bench.dualdevicenav import load_branches
    BV.setup(name, root, "/repo/carotid_data/anatomies_v4", "/s3/pin")
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs_all, spans = B.sections_for(root, br)
    pinrep = json.load(open(os.path.join("/s3/pin", name, B.REPORT)))
    v4_margin = pinrep["margin_mm"]
    tube_only = set(rep6.get("sections_tube_only") or [])
    secs = [s for s in secs_all if s.name not in tube_only]
    va_margin = rep6["ladder_v5b"]["va_margin_mm"]
    sup = {k: [tuple(x) for x in v] for k, v in (rep6.get("suppressed_stretches_mm") or {}).items()}
    cache = V.UnionCache(br, samp, secs, B.SPACING_MM)
    seq = [(s.name, (va_margin if s.branch in BV.VA_BRANCHES else v4_margin), tuple(tuple(x) for x in sup.get(s.name, [])))
           for s in secs]
    st = cache.build(seq)
    iso = iso_surface(cache.field)
    conn = iso.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"])
    sizes = np.bincount(rid)
    keep = int(np.argmax(sizes))
    rcca = B.branch(br, "RCCA")
    P, R = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
    Sr = B.arclen(P)
    trR = cKDTree(P)
    # other curves' samples (for owner by tube distance)
    oth = []
    for b in br:
        if b is rcca:
            continue
        q = np.asarray(b.coordinates, float)
        oth.append((str(b.name).split(" - ")[-1].replace(".mrk.json", "").replace(".mrk", ""), q, np.asarray(b.radii, float)))
    lower = [s for s in secs_all if s.name == "lower_R"]
    siph = [s for s in secs_all if s.name == "siphon_R"]
    surf = {}
    for s in lower + siph:
        surf[s.name] = s.surface
    import pyvista as pv
    rec = {"name": name, "v4_margin": v4_margin, "va_margin": va_margin, "suppressed": rep6.get("suppressed_stretches_mm"),
           "tube_only": sorted(tube_only), "comps": int(len(sizes)), "kept_cells": int(sizes[keep]), "dropped": []}
    order = [int(k) for k in np.argsort(-sizes) if k != keep]
    for k in order[:30]:
        piece = conn.extract_cells(np.nonzero(rid == k)[0]).extract_surface()
        X = np.asarray(piece.points, float)
        c = X.mean(0)
        d, i = trR.query(X)
        tdist_R = d - R[i]
        best_o, best_d = None, np.inf
        for nm, q, rq in oth:
            dd, ii = cKDTree(q).query(c)
            v = dd - rq[ii]
            if v < best_d:
                best_o, best_d = nm, v
        cd, ci = trR.query(c)
        row = {"cells": int(sizes[k]), "volume_mm3": round(abs(float(piece.volume)), 2), "centroid": np.round(c, 2).tolist(),
               "rcca_tube_dist_centroid": round(float(cd - R[ci]), 2), "nearest_other": best_o,
               "other_tube_dist_centroid": round(float(best_d), 2),
               "rcca_s_range": [round(float(Sr[i].min()), 2), round(float(Sr[i].max()), 2)],
               "rcca_tube_dist_pts": [round(float(tdist_R.min()), 2), round(float(np.median(tdist_R)), 2), round(float(tdist_R.max()), 2)]}
        for nm, sf in surf.items():
            ds = np.abs(np.asarray(pv.PolyData(X).compute_implicit_distance(sf)["implicit_distance"]))
            row["on_%s_surface_frac" % nm] = round(float((ds <= 0.3).mean()), 3)
        rec["dropped"].append(row)
    # refusal reasons in the gap between each RCCA fragment and the tube
    for row in rec["dropped"]:
        if row["rcca_tube_dist_centroid"] > 4 or row["other_tube_dist_centroid"] < row["rcca_tube_dist_centroid"]:
            continue
        for sname in ("lower_R", "siphon_R"):
            Pp = cache.pre.get(sname)
            if Pp is None:
                continue
            sec = Pp["sec"]
            coords = Pp["gidx"].astype(np.int64) * cache.spacing + cache.field.offset
            c = np.asarray(row["centroid"])
            # voxels of this section's candidate set near the fragment centroid (6 mm) and inside the real lumen
            near = (np.linalg.norm(coords - c, axis=1) < 6.0) & (Pp["f_sec"] > 0)
            if not near.any():
                continue
            m = Pp["m"][near]
            s_ = Pp["s"][near]
            val = cache.field.values[tuple(Pp["gidx"][near].T)]
            refused = (Pp["f_sec"][near] > val + 1e-6)          # real lumen there but the union did not take it
            r2 = {"section": sname, "voxels_in_lumen_6mm": int(near.sum()), "refused": int(refused.sum())}
            if refused.any():
                X = coords[near][refused]
                # recompute the ownership terms for the refused voxels
                bi = samp.index_of(sec.branch)
                own = samp.b == bi
                f_o, nn = _tube(X, samp.p[own], samp.r[own], 48)
                s_nn = samp.s[own][nn]
                best = np.argmax(f_o, axis=1)
                s_star = s_nn[np.arange(len(X)), best]
                nearw = np.abs(s_nn - s_star[:, None]) <= WINDOW_MM
                f_local = np.where(nearw, f_o, -np.inf).max(1)
                f_far = np.where(~nearw, f_o, -np.inf).max(1)
                comp = {"far(self)": f_far}
                for j in np.unique(samp.b[~own]):
                    if any(sb in samp.names[j] for sb in sec.siblings):
                        continue
                    mm = samp.b == j
                    fx, _ = _tube(X, samp.p[mm], samp.r[mm], 16)
                    comp[samp.names[j].split(" - ")[-1]] = fx.max(1)
                names = list(comp)
                M = np.stack([comp[n] for n in names], 1)
                who = np.array(names)[np.argmax(M, 1)]
                margin = f_local - M.max(1)
                inwin = np.zeros(len(X), bool)
                for lo, hi in sup.get(sname, []):
                    inwin |= (s_star >= lo) & (s_star <= hi)
                insec = (s_star >= sec.s0 - WINDOW_MM) & (s_star <= sec.s0 + float(B.arclen(sec.points)[-1]) + WINDOW_MM)
                why = np.where(~insec, "outside_section", np.where(inwin, "window", np.where(margin < v4_margin, "margin", "other")))
                u, cnt = np.unique(why, return_counts=True)
                r2["why"] = {str(a): int(b) for a, b in zip(u, cnt)}
                mm_ = why == "margin"
                if mm_.any():
                    u, cnt = np.unique(who[mm_], return_counts=True)
                    r2["margin_competitor"] = {str(a): int(b) for a, b in zip(u, cnt)}
                    r2["margin_vals"] = [round(float(np.percentile(margin[mm_], q)), 2) for q in (5, 50, 95)]
                    r2["s_star_margin"] = [round(float(s_star[mm_].min()), 2), round(float(s_star[mm_].max()), 2)]
                r2["s_star_refused"] = [round(float(s_star.min()), 2), round(float(s_star.max()), 2)]
            row.setdefault("gap", []).append(r2)
    rec["seconds"] = round(time.time() - t0, 1)
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1)[:6000])


if __name__ == "__main__":
    main()
