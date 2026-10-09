#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vvw\rrd_v4.py (= /sp/v5_va/vvw/rrd_v4.py in the build containers)
"""verify_v5_wall: right_route_detail v5 vs v4 UNROUNDED, with v4's exact union rebuilt from mesh_v4.json (repo
code, as ref_v4_union / chk_one.rebuild_v4), measured with measure.py's own implementation (k24 and exact).
Also separates union change from metric attribution: each union measured with BOTH centreline sets
(v5 union + v4 samples, v4 union + v5 samples), and the RCCA-owned wall compared beyond D mm of every VA.

    python3 rrd_v4.py <name>      (neve-build-meshlab; writes /sp/v5_va/vvw/rrd_v4/<name>.json)
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/sp/v5_va/vvw")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import measure as M  # noqa: E402

OUT = "/sp/v5_va/vvw/rrd_v4"


def rebuild_v4(name):
    import bake_meshes_v4 as B
    from eve_bench.dualdevicenav import load_branches
    from sdf_mesher import SPACING_MM, iso_surface, tube_field
    from sdf_union_v4 import Samples, add_real_sections_owned
    root = os.path.join(M.V4, name)
    rep = json.load(open(os.path.join(root, "mesh_v4.json"), encoding="utf-8"))
    br = load_branches(os.path.join(root, "Centrelines_comb"))
    samp = Samples(br)
    secs, spans = B.sections_for(root, br)
    tube_only = rep.get("sections_tube_only") or []
    secs = [s for s in secs if s.name not in tube_only]
    spacing = rep.get("spacing_mm", SPACING_MM)
    margin = rep.get("margin_mm")
    sup = {k: [tuple(x) for x in v] for k, v in (rep.get("suppressed_stretches_mm") or {}).items()}
    field = tube_field(br, spacing=spacing)
    if margin is not None:
        add_real_sections_owned(field, secs, samp, spacing, margin, suppress=sup)
    if (rep.get("far_clamp") or {}).get("used"):
        far = field.values < B.FAR_BELOW
        field.values[far] = -2.0 * spacing
    iso, _, _ = B.keep_largest(iso_surface(field))
    return iso


def main():
    name = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, name + ".json")
    if os.path.exists(dst):
        print("done already")
        return 0
    t0 = time.time()
    prov5 = json.load(open(os.path.join(M.A5, name, "provenance.json")))
    prov4 = json.load(open(os.path.join(M.V4, name, "provenance.json")))
    rep4 = json.load(open(os.path.join(M.V4, name, "mesh_v4.json")))
    cut5, cut4 = float(prov5.get("host_cut_mm", 0)), float(prov4.get("host_cut_mm", 0))
    s5 = M.Samp(M.read_branches(os.path.join(M.A5, name, "Centrelines_comb")))
    s4 = M.Samp(M.read_branches(os.path.join(M.V4, name, "Centrelines_comb")))
    V5, F5 = M.read_vtp(os.path.join(M.ISO, name + ".vtp"))
    iso4 = rebuild_v4(name)
    V4_ = np.asarray(iso4.points, float)
    F4_ = np.asarray(iso4.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    del iso4
    C5, A5_ = M.tri_geom(V5, F5)
    C4, A4_ = M.tri_geom(V4_, F4_)
    rec = {"name": name, "cut5": cut5, "cut4": cut4, "tris_v5": int(len(F5)), "tris_v4": int(len(F4_))}
    va = np.vstack([s.p[(s.b == s.idx("RVA")) | (s.b == s.idx("LVA"))] for s in (s4, s5)])
    vt = cKDTree(va)
    for mode in ("k24", "exact"):
        res = {}
        for tag, C, A, samp, cut in (("v5u_v5c", C5, A5_, s5, cut5), ("v4u_v4c", C4, A4_, s4, cut4),
                                     ("v5u_v4c", C5, A5_, s4, cut4), ("v4u_v5c", C4, A4_, s5, cut5)):
            O = M.Own(samp, C, A, mode)
            rr = M.rrd(O, samp, cut)
            rc = samp.idx("RCCA")
            own = (O.b == rc) & (O.s >= cut)
            dist = vt.query(C[own])[0]
            away = {}
            for D in (4.0, 6.0, 8.0):
                sel = np.zeros(len(C), bool)
                sel[np.nonzero(own)[0][dist > D]] = True
                row = {}
                for seg, m in (("fork", sel & (O.s < 130.0)), ("siphon", sel & (O.s >= 130.0))):
                    a, x = A[m], O.dep[m]
                    tot = float(a.sum()) or 1.0
                    row[seg] = {"area": float(a.sum()), "f3": float(a[x > 0.3].sum()) / tot, "f5": float(a[x > 0.5].sum()) / tot,
                                "cells": int(m.sum()),
                                # geometry fingerprint of the owned cells (sorted centroids rounded 1e-4 mm)
                                "fp": float(np.round(np.sort(C[m][:, 0]), 4).sum() + np.round(np.sort(C[m][:, 2]), 4).sum())}
                away["%d" % D] = row
            res[tag] = {"rrd": rr, "away": away}
        rec[mode] = res
        a, b = res["v5u_v5c"]["rrd"], res["v4u_v4c"]["rrd"]
        rec[mode]["delta_v5_v4"] = {g: {t: a[g][t] - b[g][t] for t in ("frac_0.3mm", "frac_0.5mm", "area_mm2")} for g in a}
        a2 = res["v5u_v4c"]["rrd"]
        rec[mode]["delta_v5union_v4centrelines_vs_v4"] = {g: {t: a2[g][t] - b[g][t] for t in ("frac_0.3mm", "frac_0.5mm", "area_mm2")} for g in a}
        rec[mode]["identical_beyond"] = {D: all(res["v5u_v5c"]["away"][D][g][k] == res["v4u_v4c"]["away"][D][g][k]
                                                for g in ("fork", "siphon") for k in ("area", "f3", "f5", "cells", "fp"))
                                         for D in ("4", "6", "8")}
    rec["k24_v4_rebuild_r4"] = M.r4(rec["k24"]["v4u_v4c"]["rrd"])
    rec["k24_v4_rebuild_equals_mesh_v4"] = rec["k24_v4_rebuild_r4"] == rep4["right_route_detail"]
    rec["seconds"] = round(time.time() - t0, 1)
    with open(dst + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, default=float)
    os.replace(dst + ".tmp", dst)
    print("DONE", name, rec["seconds"], "rebuild==mesh_v4", rec["k24_v4_rebuild_equals_mesh_v4"],
          "k24 delta", json.dumps(rec["k24"]["delta_v5_v4"]), "beyond", rec["k24"]["identical_beyond"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
