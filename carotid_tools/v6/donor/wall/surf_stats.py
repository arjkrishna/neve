#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\surf_stats.py (= /sp/v6/wall/surf_stats.py in the build containers)
"""surf_stats.py (v6_wall): topology and size of every donor VA stage-A surface, v5 against each variant, plus the VMTK
MISR floor usage (MISR < 1.0 mm length) when the variant's VMTK line exists.
  genus = (2 - chi) / 2 for one closed component (chi = V - E + F after cleaning); handles drive the bake's local tube
  windows (bake_v5b handle rule), so a fix must not add many.
    python3 surf_stats.py <variant,...>  -> /v6w/surf_stats.json"""
import json
import os
import sys

import numpy as np
import pyvista as pv
import vtk

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
VARS = sys.argv[1].split(",")


def topo(path):
    m = pv.read(path).triangulate().clean()
    ncomp = int(m.connectivity().point_data["RegionId"].max()) + 1 if m.n_points else 0
    edges = m.extract_all_edges()
    chi = m.n_points - edges.n_cells + m.n_cells
    fe = m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=True)
    return {"points": int(m.n_points), "tris": int(m.n_cells), "components": ncomp, "chi": int(chi),
            "genus_if_one": (2 - chi) / 2 if ncomp == 1 else None, "open_or_nonmanifold_edges": int(fe.n_cells),
            "area_mm2": float(m.area)}


def floored(P, R):
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    ds = np.r_[np.diff(s), 0.0]
    return float(ds[R < 1.0].sum()), float(np.median(R)), float(s[-1])


out = {}
for unit in sorted(os.listdir(R5)):
    for sd in ("R", "L"):
        f = os.path.join(R5, unit, "%s.json" % sd)
        if not os.path.exists(f):
            continue
        rec = json.load(open(f))
        row = {"v5": topo(rec["surface"]["path"])}
        fl, med, L = floored(np.asarray(rec["centerline"]["points"], float), np.asarray(rec["centerline"]["radii_misr"], float))
        row["v5"].update(floored_mm=fl, misr_median=med, length_mm=L)
        v = "rva" if sd == "R" else "lva"
        for var in VARS:
            p = "/v6w/seg/%s/%s/%s_stageA.vtp" % (var, unit, v)
            if not os.path.exists(p):
                continue
            row[var] = topo(p)
            vj = "/v6w/seg/%s/%s/%s_stageA_vmtk.json" % (var, unit, v)
            if os.path.exists(vj):
                d = json.load(open(vj))
                fl, med, L = floored(np.asarray(d["points"], float), np.asarray(d["radii"], float))
                row[var].update(floored_mm=fl, misr_median=med, length_mm=L)
        out["%s:%s" % (unit, sd)] = row
        print("%-24s " % ("%s:%s" % (unit, sd)) + "  ".join(
            "%s g%s A%.0f fl%s" % (k, r["genus_if_one"], r["area_mm2"], "%.1f" % r["floored_mm"] if "floored_mm" in r else "-")
            for k, r in row.items()), flush=True)
json.dump(out, open("/v6w/surf_stats.json", "w"), indent=1)
