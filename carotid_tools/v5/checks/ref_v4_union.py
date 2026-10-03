#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\ref_v4_union.py (= /sp/v5_va/ref_v4_union.py in the build containers)
"""Reference part c: the v4 VAs as shipped, on the undecimated union.

For a sample of v4 anatomies, the union bake_meshes_v4 accepted is rebuilt
exactly (its final margin, suppressed stretches, tube-only sections and band
clamp, from mesh_v4.json), and checked: right_route_detail on the rebuild must
equal the value mesh_v4.json recorded. Then, with right_route_detail's own
measure (va_metrics.measure_union_branch), the wall detail of each VA:

    host    VA origin -> cut - 25 mm     host tube (the plain smooth tube)
    ramp    cut - 25 -> cut              host tube blended to the donor calibre
    donor   cut -> end                   TopBrain VA label surface (real wall)

plus the RCCA fork / siphon in radius-relative terms, the same VA segments on
the shipped 60k collision_full.vtp (to show the decimation bias), and the gross
shape of the composite (host + donor) VA centerline.

Read-only on the repo. Writes /sp/v5_va/results/ref_c_v4_union.json.
eve-training-fixed.   WORKERS=3 python3 ref_v4_union.py [--n 10]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/sp/v5_va")
sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/opt/eve_training/eve_bench")
import va_metrics as vm  # noqa: E402

ROOT = "/repo/carotid_data/anatomies_v4"
OUT = "/sp/v5_va/results/ref_c_v4_union.json"
RAMP_MM = 25.0


def pick(n):
    scan = json.load(open("/sp/v5_va/results/v4_va_scan.json"))
    sel = [r["name"] for r in scan if r["va_R"]["suppressed"] or r["va_L"]["suppressed"]]
    sel += [r["name"] for r in scan if r["va_L"]["real_surface"] is False][:1]
    sel += ["case_k_004_left__topcow_mr_010"]
    used = {r["patient"] for r in scan if r["name"] in sel}
    want_h = "mirrored"
    for r in scan[::5]:
        if len(sel) >= n:
            break
        if r["name"] in sel or r["patient"] in used or r["handedness"] != want_h:
            continue
        sel.append(r["name"])
        used.add(r["patient"])
        want_h = "native" if want_h == "mirrored" else "mirrored"
    return sel[:n]


def branch_idx(samp, tag):
    for i, n in enumerate(samp.names):
        if n.endswith(tag) or n.endswith(tag + ".mrk"):
            return i
    return samp.index_of(tag)


def rebuild(name):
    import pyvista as pv
    import bake_meshes_v4 as B
    from eve_bench.dualdevicenav import load_branches
    from sdf_mesher import SPACING_MM, iso_surface, tube_field
    from sdf_union_v4 import Samples, add_real_sections_owned
    t0 = time.time()
    root = os.path.join(ROOT, name)
    rep = json.load(open(os.path.join(root, "mesh_v4.json"), encoding="utf-8"))
    prov = json.load(open(os.path.join(root, "provenance.json"), encoding="utf-8"))
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
    iso, _, dropped = B.keep_largest(iso_surface(field))
    del field
    detail = B.right_route_detail(iso, samp, root)
    rec = rep["right_route_detail"]
    match = max(abs(detail[g][t] - rec[g][t]) for g in detail for t in detail[g])
    out = {"name": name, "handedness": prov.get("handedness"), "patient": prov.get("patient"),
           "union_tris": int(iso.n_cells), "margin_mm": margin, "suppressed": rep.get("suppressed_stretches_mm"),
           "tube_only": tube_only, "spans": {k: v for k, v in spans.items() if k.startswith("va_")},
           "right_route_detail_rebuilt": detail, "right_route_detail_recorded": rec,
           "rebuild_max_abs_diff": round(float(match), 5)}
    full = pv.read(os.path.join(root, "collision_full.vtp"))
    # RCCA in radius-relative terms
    rc = branch_idx(samp, "RCCA")
    cut = float(prov.get("host_cut_mm", 0.0))
    out["RCCA"] = {"fork": vm.measure_union_branch(iso, samp, rc, cut, 130.0 - 1e-6),
                   "siphon": vm.measure_union_branch(iso, samp, rc, 130.0, 1e9)}
    # VAs
    by = {str(b.name): b for b in br}

    def bget(tag):
        for k, b in by.items():
            if k.endswith(tag):
                return np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        raise KeyError(tag)
    for side, tag, sub, key in (("right", "RVA", "(17).mrk", "va_R"), ("left", "LVA", "(19).mrk", "va_L")):
        P, R = bget("- %s.mrk" % tag)
        sc, _ = bget(sub)
        s = vm.arclen(P)
        i0 = int(np.argmin(np.linalg.norm(P - sc[0], axis=1)))
        s0 = float(s[i0])
        vcut = float(prov["vertebrals"][side]["cut_mm"])
        bidx = branch_idx(samp, tag)
        real = prov["xform"][key].get("real_surface", True) is not False and key not in tube_only
        segs = {"host": (s0, vcut - RAMP_MM), "ramp": (vcut - RAMP_MM, vcut), "donor": (vcut, 1e9)}
        d = {"origin_s_mm": round(s0, 1), "cut_mm": round(vcut, 1), "length_mm": round(float(s[-1]), 1),
             "donor_len_mm": round(float(s[-1] - vcut), 1),
             "donor_real_surface": bool(real), "donor_suppressed": (rep.get("suppressed_stretches_mm") or {}).get(key),
             "donor_surface": prov["xform"][key].get("surface"), "mirror": prov["xform"][key].get("mirror")}
        for nm, (lo, hi) in segs.items():
            d[nm] = vm.measure_union_branch(iso, samp, bidx, lo, hi)
            d[nm + "_on_collision_full_60k"] = vm.measure_union_branch(full, samp, bidx, lo, hi)["core"]
        # area-weighted whole VA proper
        d["whole_va"] = vm.measure_union_branch(iso, samp, bidx, s0, 1e9)
        # composite centerline shape (host + donor), from the VA origin
        g = vm.measure_va(P[i0:], R[i0:], None, source_type="v4 composite (host tube + TopBrain donor)",
                          radius_kind="host declared + donor VMTK MISR", starts_at_origin=True,
                          parent_dir=None, up=None, name="%s %s" % (name, tag), do_union=False)
        d["composite_gross"] = g["gross"]
        d["composite_calibre"] = g["calibre"]
        out[tag] = d
    out["seconds"] = round(time.time() - t0, 1)
    print("%-44s rebuild diff %.4f  RVA host/donor 0.3: %.3f/%.3f  LVA host/donor 0.3: %.3f/%.3f  (%.0fs)" % (
        name, out["rebuild_max_abs_diff"], out["RVA"]["host"]["core"]["frac_0.3mm"],
        out["RVA"]["donor"]["core"]["frac_0.3mm"], out["LVA"]["host"]["core"]["frac_0.3mm"],
        out["LVA"]["donor"]["core"]["frac_0.3mm"], out["seconds"]), flush=True)
    return out


def safe(name):
    try:
        return rebuild(name)
    except Exception as e:                                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        return {"name": name, "error": "%s: %s" % (type(e).__name__, e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    names = a.only.split(",") if a.only else pick(a.n)
    print("sample:", names, flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", "3"))) as pool:
        res = pool.map(safe, names, chunksize=1)
    json.dump({"version": vm.VERSION, "sample": names, "ramp_mm": RAMP_MM, "anatomies": res,
               "seconds": round(time.time() - t0, 1)}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
