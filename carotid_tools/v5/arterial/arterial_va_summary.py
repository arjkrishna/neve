#!/usr/bin/env python
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\arterial_va_summary.py (= /sp/v5_va/arterial_va_summary.py in the build containers)
"""Summarise Arterial's labelled centerline graph for the vertebral arteries.

Run inside neve-arterial (python on PATH) after run_arterial.sh, or with any python that has
numpy + networkx + nibabel (the pickles are networkx graphs).

Reads   <case>/extracranial_vessels/segments_graph_pred.pickle   (edges = centerline segments,
                                                                  attr vessel_type_name)
        <case>/extracranial_vessels/segmentation.nii.gz         (binary vessel mask, CTA grid)
        <case>/cta.nii.gz, <case>/prep.json
Writes  <case>/va_summary.json        connectivity + geometry per vessel class, frame check
        <case>/labelled_centerlines.json   every labelled segment: world (RAS mm, Arterial frame)
                                       points + inscribed radius, plus source-frame points

Coordinate note: centerline_segments_array / the graph store points relative to the image's
"LPI corner" (world min corner). world_RAS = p + lpi_corner. This script adds it back and checks
the result against the segmentation mask (fraction of centerline points inside the mask).
"""
import json
import os
import pickle
import sys

import numpy as np
import networkx as nx
import nibabel as nib
from scipy import ndimage

NAMES = ["other", "AA", "BT", "RCCA", "LCCA", "RSA", "LSA", "RVA", "LVA", "RICA", "LICA", "RECA", "LECA", "BA"]


def lpi_corner(affine, shape):
    o = nib.aff2axcodes(affine)
    if o == ("R", "A", "S"):
        v = np.array([0, 0, 0])
    elif o == ("L", "A", "S"):
        v = np.array([shape[0] - 1, 0, 0])
    elif o == ("L", "P", "S"):
        v = np.array([shape[0] - 1, shape[1] - 1, 0])
    else:
        raise ValueError(o)
    return (affine @ np.append(v, 1))[:3]


def seg_len(c):
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0


def main(case):
    md = os.path.join(case, "extracranial_vessels")
    G = pickle.load(open(os.path.join(md, "segments_graph_pred.pickle"), "rb"))
    seg = nib.load(os.path.join(md, "segmentation.nii.gz"))
    m = np.asanyarray(seg.dataobj) > 0
    A = seg.affine
    corner = lpi_corner(A, m.shape)
    prep = json.load(open(os.path.join(case, "prep.json"))) if os.path.isfile(os.path.join(case, "prep.json")) else None
    to_src = np.array(prep["arterial_to_input_world"]) if prep else np.eye(4)

    # --- frame check: centerline points inside the mask (1-voxel tolerance) ---
    md1 = ndimage.binary_dilation(m, iterations=1)
    Ainv = np.linalg.inv(A)
    allp = np.concatenate([d["coordinate_array"] for _, _, d in G.edges(data=True)]) + corner
    ijk = np.rint((Ainv @ np.c_[allp, np.ones(len(allp))].T)[:3].T).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(m.shape)), axis=1)
    inside = np.zeros(len(ijk), bool)
    inside[ok] = md1[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    frame_check = {"n_points": int(len(allp)), "frac_inside_mask_1vox": float(inside.mean())}
    # the same check with the x axis mirrored, to detect a direction-matrix mix-up
    ijk_m = ijk.copy(); ijk_m[:, 0] = m.shape[0] - 1 - ijk_m[:, 0]
    okm = np.all((ijk_m >= 0) & (ijk_m < np.array(m.shape)), axis=1)
    insm = np.zeros(len(ijk_m), bool); insm[okm] = md1[ijk_m[okm, 0], ijk_m[okm, 1], ijk_m[okm, 2]]
    frame_check["frac_inside_if_x_mirrored"] = float(insm.mean())

    # --- per-class geometry ---
    by_cls = {}
    out_segments = []
    for u, v, d in G.edges(data=True):
        name = d.get("vessel_type_name", NAMES[int(d.get("vessel_type", 0))])
        c = np.asarray(d["coordinate_array"], float) + corner
        r = np.asarray(d["radius_array"], float)
        L = seg_len(c)
        by_cls.setdefault(name, []).append((u, v, L, c, r))
        cs = (to_src @ np.c_[c, np.ones(len(c))].T)[:3].T
        out_segments.append({"cell_id": int(d.get("cell_id", -1)), "class": name, "nodes": [int(u), int(v)],
                             "length_mm": round(L, 2), "radius_mean_mm": round(float(r.mean()), 3),
                             "points_ras_mm": np.round(c, 3).tolist(), "points_src_world_mm": np.round(cs, 3).tolist(),
                             "radius_mm": np.round(r, 3).tolist()})
    classes = {}
    for name, lst in sorted(by_cls.items()):
        pts = np.concatenate([x[3] for x in lst])
        rr = np.concatenate([x[4] for x in lst])
        classes[name] = {"n_segments": len(lst), "length_mm": round(sum(x[2] for x in lst), 1),
                         "radius_median_mm": round(float(np.median(rr)), 2),
                         "S_range_mm": [round(float(pts[:, 2].min()), 1), round(float(pts[:, 2].max()), 1)]}

    # --- VA connectivity ---
    def cls_of(u, v):
        d = G[u][v]
        return d.get("vessel_type_name", NAMES[int(d.get("vessel_type", 0))])

    va = {}
    for side, sa, other_sa in (("RVA", "RSA", "LSA"), ("LVA", "LSA", "RSA")):
        E = [(u, v) for u, v in G.edges() if cls_of(u, v) == side]
        H = nx.Graph(); H.add_edges_from(E)
        comps = []
        for nodes in nx.connected_components(H):
            sub = H.subgraph(nodes)
            nbr = {}
            for n in nodes:
                for w in G.neighbors(n):
                    c = cls_of(n, w)
                    if c != side:
                        nbr[c] = nbr.get(c, 0) + 1
            Ls = [seg_len(np.asarray(G[u][v]["coordinate_array"])) for u, v in sub.edges()]
            pts = np.concatenate([np.asarray(G[u][v]["coordinate_array"]) for u, v in sub.edges()]) + corner
            comps.append({"n_segments": sub.number_of_edges(), "length_mm": round(sum(Ls), 1),
                          "S_range_mm": [round(float(pts[:, 2].min()), 1), round(float(pts[:, 2].max()), 1)],
                          "neighbour_classes": nbr,
                          "touches_own_subclavian": sa in nbr, "touches_other_subclavian": other_sa in nbr,
                          "touches_arch_or_BT": ("AA" in nbr) or ("BT" in nbr), "touches_BA": "BA" in nbr})
        comps.sort(key=lambda x: -x["length_mm"])
        # one continuous labelled path SA -> VA -> BA ?
        keep = {sa, side, "BA"}
        K = nx.Graph(); K.add_edges_from([(u, v) for u, v in G.edges() if cls_of(u, v) in keep])
        cont = False
        for nodes in nx.connected_components(K):
            cl = {cls_of(u, v) for u, v in K.subgraph(nodes).edges()}
            if keep <= cl:
                cont = True
        va[side] = {"present": bool(E), "n_components": len(comps), "components": comps,
                    "continuous_SA_VA_BA_labelled_path": cont}

    # whole-graph topology
    topo = {"n_edges": G.number_of_edges(), "n_nodes": G.number_of_nodes(),
            "n_connected_components": nx.number_connected_components(G)}
    summary = {"case": case, "frame_check": frame_check, "lpi_corner_world": corner.tolist(),
               "graph": topo, "classes": classes, "vertebral": va}
    json.dump(summary, open(os.path.join(case, "va_summary.json"), "w"), indent=1)
    json.dump({"frame": "points_ras_mm = nibabel world of <case>/cta.nii.gz; points_src_world_mm = world of the "
                        "original input (prep.json arterial_to_input_world)",
               "segments": out_segments}, open(os.path.join(case, "labelled_centerlines.json"), "w"))
    print(json.dumps({"frame_check": frame_check, "graph": topo, "classes": classes,
                      "vertebral": {k: {kk: vv for kk, vv in v.items() if kk != "components"} | {
                          "components": v["components"][:3]} for k, v in va.items()}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
