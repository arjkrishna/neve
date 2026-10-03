#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_art_extract.py (= /sp/v5_va/pilot_art_extract.py in the build containers)
"""pilot_art_extract.py = art_extract.py (the study tool, copied UNCHANGED) + ONE change, the basilar seed fix
(pilot_proc_aortaseg agent, 2026-10-01). Diff, all inside find_ba():
  * seed window top z_FM + 40 mm instead of z_FM + 55 mm. art_extract's own docstring says "+40"; the code used
    +55. verify_tag_bench (section 4) showed that the +55 window lets an ACA segment (z_FM + 35..+55) out-score the
    real basilar on TaG-Net A, that a wider x band does NOT fix it, and that the +40 top DOES.
  * mask-based plausibility of the seed: a candidate component whose lowest voxel lies more than 20 mm above the
    window bottom (z_FM + 8) cannot be the basilar (the basilar runs down to the VA confluence near z_FM + 8..+25),
    so its score is multiplied by 0.1 (the same down-weight the code already gives to components < 8 mm tall).
  * the x band stays +-5 mm about the midline; with --ts the midline is the TS spinal cord at z_FM (art_extract's own
    TS-cord rule, the tag_arterial fix 1).
Every other line is art_extract.py. Run with the same arguments.
--- original docstring follows ---
art_extract.py -- vertebral arteries (VA) from one Arterial run.

    python3 art_extract.py <case_dir> <out_dir> [--ts <ts_subject_dir>] [--zfm <z_mm>] [--tag <name>]

Arterial gives a BINARY vessel mask plus a labelled centerline TREE. The VA-basilar confluence closes a
loop (SA -> RVA -> BA <- LVA <- SA), and a centerline tree cannot hold a loop, so at most one VA can
reach the basilar as a labelled path; the other is dropped or stops early. This script therefore:

  1. reads the labelled RVA / LVA segments (Arterial "as labelled": chain from the subclavian junction
     to the highest node of the proximal VA component);
  2. finds the basilar (BA) on the MASK: midline (|x - x_mid| < 5 mm, x_mid = skull-mask centroid)
     voxels with EDT >= 0.6 mm between z_FM + 8 and z_FM + 40 mm (z_FM = foramen magnum: TS C1 top
     + 6 mm, or --zfm; window to z_FM + 55), largest component; end = its highest voxel; nothing above
     end + 3 mm is passable (no circle-of-Willis detours). x_mid = TS spinal-cord centroid (TS) or skull centroid.
     A plausible Arterial BA label (lowest point within 8 mm of the midline, z_FM - 5..+60, within 3 mm of the
     mask basilar; r >= 0.8, >= 10 mm)
     overrides it: end = the label's lowest point (the confluence node);
  3. traces each VA on the mask with a minimal-cost path (skimage MCP_Geometric, cost
     1 + (0.5 / (EDT + 0.2))^2 per mm inside the mask, 8 per mm across a gap (2.5 where the TS CT is
     contrast-filled was TRIED and disabled: at 1.5 mm cancellous bone and veins are "vessel-like" and the
     bridges cut through them; the HU fraction along each bridge is still reported), impassable: the
     contralateral half-space below z_FM - 5, the cervical carotids [Arterial RCCA/LCCA/RICA/LICA/RECA/
     LECA segments below z_FM - 10, tube r + 1.5 mm, unless within 5 mm of an own-side VA label] and,
     with --ts, the TS common-carotid masks (+1.5 mm) and anything > 10 mm from the TS screening trace
     below the C1 top). Start = the labelled VA's subclavian junction, else every own-side SA label
     point (the cheapest start is the VA origin); end = the first BA voxel reached (the confluence);
  4. trims the start to where the path leaves the subclavian lumen (distance to the SA centerline >
     the SA's labelled radius), smooths it (Gaussian sigma 1 mm along arclength);
  5. cuts the VA out of the mask (capsule 1.8 r_EDT + 1 mm around the path, planar cuts at the origin
     and the end, largest component, holes filled) and writes a stage-A surface (binary marching cubes
     + windowed sinc 20 it / 0.1 = topbrain_tools/mask_to_surface.write_vtp, unchanged) and VMTK seeds;
  6. QC (with --ts): origin distance to the TS subclavian mask, vertebra enclosure per level C7..C1
     (16 rays in the normal plane, 8 mm), landmarks C6 / C2 / C1 / dura proxy (first s above z_FM after
     C1), distance to the TS screening trace, labelled-vs-traced agreement, gaps, radius.

Runs in neve-build-stagea (nibabel, scipy, skimage 0.21, vtk). Licences: Arterial weights CC BY-NC 4.0,
code PolyForm-NC; TotalSegmentator v3 data CC BY 4.0; VMR 0248 research use. Derived data research-only.
"""
import argparse
import heapq
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from skimage import measure

sys.path.insert(0, "/opt/eve_training/topbrain_tools")
sys.path.insert(0, "/sp/v5_va")
from mask_to_surface import write_vtp  # noqa: E402

CAROTID = {"RCCA", "LCCA", "RICA", "LICA", "RECA", "LECA"}
GAP_COST = 8.0
GAP_COST_HU = 8.0  # CT-guided cheaper bridging (2.5) was tried and cut through cancellous bone / veins: disabled
R_REF = 0.5


# ----------------------------------------------------------------------------- small helpers
def arclen(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def resample(p, step=0.5, r=None):
    s = arclen(p)
    if s[-1] < step:
        return p.copy(), (None if r is None else r.copy()), s
    q = np.arange(0, s[-1] + 1e-9, step)
    P = np.stack([np.interp(q, s, p[:, i]) for i in range(3)], 1)
    return P, (None if r is None else np.interp(q, s, r)), q


def smooth(P, sigma_mm, step):
    n = len(P)
    if n < 5:
        return P.copy()
    sig = sigma_mm / step
    pad = int(np.ceil(4 * sig))
    k = int(max(1, min(n - 1, round(2 * sigma_mm / step))))
    t0 = P[k] - P[0]; t0 /= (np.linalg.norm(t0) or 1)
    t1 = P[-1] - P[-1 - k]; t1 /= (np.linalg.norm(t1) or 1)
    pre = P[0] - t0[None] * (step * np.arange(pad, 0, -1))[:, None]
    post = P[-1] + t1[None] * (step * np.arange(1, pad + 1))[:, None]
    Q = ndimage.gaussian_filter1d(np.vstack([pre, P, post]), sig, axis=0, mode="nearest")
    return Q[pad:pad + n]


def w2i(A, P):
    return (np.linalg.inv(A) @ np.c_[P, np.ones(len(P))].T)[:3].T


def i2w(A, I):
    return (A @ np.c_[I, np.ones(len(I))].T)[:3].T


def tangents(P, sigma_mm=2.0, step=0.5):
    Q = smooth(P, sigma_mm, step)
    t = np.gradient(Q, axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


# ----------------------------------------------------------------------------- Arterial graph
def node_positions(segs):
    inc = {}
    for i, s in enumerate(segs):
        for n in set(s["nodes"]):
            inc.setdefault(n, []).append(i)
    pos = {}
    for n, L in inc.items():
        if len(L) >= 2:
            cands = np.concatenate([[segs[i]["P"][0], segs[i]["P"][-1]] for i in L])
            score = [sum(min(np.linalg.norm(c - segs[i]["P"][0]), np.linalg.norm(c - segs[i]["P"][-1])) for i in L)
                     for c in cands]
            pos[n] = cands[int(np.argmin(score))]
    for n, L in inc.items():
        if n in pos:
            continue
        s = segs[L[0]]
        other = [x for x in s["nodes"] if x != n]
        if other and other[0] in pos:
            a, b = s["P"][0], s["P"][-1]
            pos[n] = a if np.linalg.norm(a - pos[other[0]]) > np.linalg.norm(b - pos[other[0]]) else b
        else:
            pos[n] = s["P"][0] if s["nodes"][0] == n else s["P"][-1]
    return pos, inc


def oriented(seg, from_pos):
    P, R = seg["P"], seg["R"]
    if np.linalg.norm(P[-1] - from_pos) < np.linalg.norm(P[0] - from_pos):
        return P[::-1], R[::-1]
    return P, R


def labelled_va(segs, pos, inc, side):
    VA, SA = side + "VA", side + "SA"
    idx = [i for i, s in enumerate(segs) if s["class"] == VA]
    if not idx:
        return {"present": False}
    # components of VA segments sharing nodes
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for n, L in inc.items():
        L2 = [i for i in L if i in parent]
        for a in L2[1:]:
            parent[find(a)] = find(L2[0])
    comps = {}
    for i in idx:
        comps.setdefault(find(i), []).append(i)
    info = []
    for c in comps.values():
        nodes = set(n for i in c for n in segs[i]["nodes"])
        nbr = {}
        for n in nodes:
            for j in inc[n]:
                if j not in c:
                    nbr[segs[j]["class"]] = nbr.get(segs[j]["class"], 0) + 1
        pts = np.concatenate([segs[i]["P"] for i in c])
        info.append({"segs": c, "nodes": nodes, "nbr": nbr, "zmin": float(pts[:, 2].min()), "zmax": float(pts[:, 2].max()),
                     "length_mm": float(sum(segs[i]["length_mm"] for i in c)), "touches_SA": SA in nbr})
    info.sort(key=lambda x: (not x["touches_SA"], x["zmin"]))
    prox = info[0]
    # origin node
    cand = [n for n in prox["nodes"] if any(segs[j]["class"] == SA for j in inc[n])]
    if not cand:
        cand = [n for n in prox["nodes"] if any(segs[j]["class"] in ("AA", "BT", "RSA", "LSA", "RCCA", "LCCA") for j in inc[n])]
    if not cand:
        deg = {n: sum(1 for j in inc[n] if j in prox["segs"]) for n in prox["nodes"]}
        cand = [n for n in prox["nodes"] if deg[n] == 1] or list(prox["nodes"])
    o = min(cand, key=lambda n: pos[n][2])
    # Dijkstra within the component; target = highest node
    adj = {}
    for i in prox["segs"]:
        u, v = segs[i]["nodes"]
        adj.setdefault(u, []).append((v, i)); adj.setdefault(v, []).append((u, i))
    dist = {o: 0.0}; prev = {}
    h = [(0.0, o)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist.get(u, 1e18):
            continue
        for v, i in adj.get(u, []):
            nd = d + segs[i]["length_mm"]
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = (u, i); heapq.heappush(h, (nd, v))
    t = max(dist, key=lambda n: pos[n][2])
    chain = []
    n = t
    while n != o:
        u, i = prev[n]; chain.append((u, i)); n = u
    chain = chain[::-1]
    Ps, Rs = [], []
    for u, i in chain:
        P, R = oriented(segs[i], pos[u])
        if Ps:
            P, R = P[1:], R[1:]
        Ps.append(P); Rs.append(R)
    if not Ps:
        return {"present": True, "n_components": len(info), "error": "empty chain"}
    P = np.concatenate(Ps); R = np.concatenate(Rs)
    comp_rows = [{"length_mm": round(c["length_mm"], 1), "z_range_mm": [round(c["zmin"], 1), round(c["zmax"], 1)],
                  "neighbour_classes": c["nbr"], "touches_own_SA": c["touches_SA"], "touches_BA": "BA" in c["nbr"]}
                 for c in info]
    return {"present": True, "P": P, "R": R, "origin_node_pos": pos[o], "touches_own_SA": prox["touches_SA"],
            "n_components": len(info), "components": comp_rows,
            "all_points": np.concatenate([segs[i]["P"] for c in info for i in c["segs"]]),
            "length_mm": float(arclen(P)[-1]), "z_range_mm": [float(P[:, 2].min()), float(P[:, 2].max())]}


# ----------------------------------------------------------------------------- main pieces
def find_ba(m, A, sp, x_mid, z_fm):
    """Basilar on the mask: a seed component in the midline band (|x - x_mid| < 5 mm, z_FM + 8..+40 mm,
    EDT >= 0.6 mm), grown within |x - x_mid| < 15 mm and z_FM + 8..+50 mm (EDT >= 0.5 mm) so a tortuous
    basilar stays in (reported); the single end point is the highest voxel of the midline seed (z <= z_FM + 55)."""
    zc = A[2, 3] + A[2, 2] * np.arange(m.shape[2])
    xc = A[0, 3] + A[0, 0] * np.arange(m.shape[0])
    ks = np.where((zc >= z_fm + 8) & (zc <= z_fm + 60))[0]
    is_ = np.where(np.abs(xc - x_mid) < 15.0)[0]
    if len(ks) == 0 or len(is_) == 0:
        return None
    i0, i1, k0, k1 = is_.min(), is_.max() + 1, ks.min(), ks.max() + 1
    pi0, pi1 = max(0, i0 - 10), min(m.shape[0], i1 + 10)
    pk0, pk1 = max(0, k0 - 10), min(m.shape[2], k1 + 10)
    sub = m[pi0:pi1, :, pk0:pk1]
    edt = ndimage.distance_transform_edt(sub, sampling=sp)
    xs = xc[pi0:pi1][:, None, None]
    zs = zc[pk0:pk1][None, None, :]
    band = (np.abs(xs - x_mid) < 5.0) & (zs >= z_fm + 8) & (zs <= z_fm + 40)   # PILOT FIX: +40 (was +55)
    wide = (np.abs(xs - x_mid) < 15.0) & (zs >= z_fm + 8) & (zs <= z_fm + 60)
    cand = sub & band & (edt >= 0.6)
    lab, n = ndimage.label(cand, structure=np.ones((3, 3, 3)))
    if n == 0:
        return None
    best, bscore = None, -1
    for c in range(1, n + 1):
        vox = np.argwhere(lab == c)
        zext = (vox[:, 2].max() - vox[:, 2].min()) * sp[2]
        score = len(vox) * (1.0 if zext >= 8 else 0.1)
        zlow = float(zc[pk0 + vox[:, 2].min()])                                   # PILOT FIX: mask-based plausibility
        score *= 1.0 if zlow <= z_fm + 8 + 20 else 0.1
        if score > bscore:
            best, bscore = c, score
    seed = lab == best
    glab, _ = ndimage.label(sub & wide & (edt >= 0.5), structure=np.ones((3, 3, 3)))
    ids = np.unique(glab[seed & (glab > 0)])
    grown = np.isin(glab, ids) if len(ids) else seed
    vox = np.argwhere(grown)
    # end point = the highest voxel of the midline SEED (the basilar inside the +-5 mm band); the grown part
    # (+-15 mm) climbs into the PCA / SCA / circle-of-Willis region and invites detours, so it is not used
    sv = np.argwhere(seed)
    top = sv[np.argmax(sv[:, 2])]
    off = np.array([pi0, 0, pk0])
    W = i2w(A, (vox + off).astype(float))
    Ws = i2w(A, (np.argwhere(seed) + off).astype(float))
    endw = i2w(A, (top + off)[None].astype(float))[0]
    return {"world": W, "end_world": endw, "n": int(len(vox)), "n_seed": int(len(Ws)),
            "z_range_mm": [float(W[:, 2].min()), float(W[:, 2].max())],
            "seed_z_range_mm": [float(Ws[:, 2].min()), float(Ws[:, 2].max())],
            "seed_x_median_minus_mid_mm": float(np.median(Ws[:, 0]) - x_mid),
            "end_point_mm": endw.round(2).tolist()}


def true_carotids(segs, inc):
    """Carotid-labelled segments reachable from the arch / brachiocephalic trunk through carotid labels only.
    A 'CCA' / 'ICA' hanging off a subclavian or off the basilar is an Arterial mislabel (seen: the proximal
    LVA labelled LCCA, the distal LVA labelled LICA), so it is not made impassable."""
    start_nodes = set(n for i, s in enumerate(segs) if s["class"] in ("AA", "BT") for n in s["nodes"])
    seen, stack = set(), list(start_nodes)
    vis_nodes = set(stack)
    while stack:
        n = stack.pop()
        for j in inc.get(n, []):
            if segs[j]["class"] in CAROTID and j not in seen:
                seen.add(j)
                for nn in segs[j]["nodes"]:
                    if nn not in vis_nodes:
                        vis_nodes.add(nn); stack.append(nn)
    return seen


def trace_side(side, m, A, sp, segs, lab, ba, x_mid, z_fm, ts=None, car_ok=None, mid_fn=None):
    """MCP trace of one VA on the mask. Returns dict with the world path origin -> confluence."""
    sgn = 1.0 if side == "R" else -1.0          # RAS: right = +x
    SA = side + "SA"
    sa_pts = np.concatenate([s["P"] for s in segs if s["class"] == SA]) if any(s["class"] == SA for s in segs) else None
    sa_rad = np.concatenate([s["R"] for s in segs if s["class"] == SA]) if sa_pts is not None else None
    if lab.get("present") and lab.get("touches_own_SA"):
        starts_w = lab["origin_node_pos"][None]
        start_kind = "labelled VA junction on the own SA"
    elif sa_pts is not None:
        starts_w = sa_pts
        start_kind = "every own-SA label point (multi-start)"
    elif ts is not None and ts.get("sa_centre", {}).get(side) is not None:
        starts_w = ts["sa_centre"][side]
        start_kind = "TS subclavian mask skeleton points (no own-SA label)"
    else:
        return {"ok": False, "why": "no start (no own-SA label, no VA label on the SA)"}
    if ba is None:
        return {"ok": False, "why": "no basilar found on the mask"}
    # end = one point, the top of the basilar component: both VAs run through the confluence into
    # the basilar, and the confluence is found afterwards where the two traces merge
    ends_w = ba["end_world"][None]
    # crop box (world): own side of the midline, -15 mm across it
    rel = [starts_w, ends_w] + ([lab["all_points"]] if lab.get("present") and "all_points" in lab else [])
    if ts is not None and ts.get("trace", {}).get(side) is not None:
        rel.append(ts["trace"][side])
    R_all = np.concatenate(rel)
    lo = R_all.min(0) - np.array([25, 30, 10.0]); hi = R_all.max(0) + np.array([25, 30, 10.0])
    if side == "R":
        lo[0] = max(lo[0], x_mid - 15)
    else:
        hi[0] = min(hi[0], x_mid + 15)
    hi[2] = max(hi[2], z_fm + 45)
    I = w2i(A, np.array([lo, hi]))
    i0 = np.clip(np.floor(I.min(0)).astype(int), 0, np.array(m.shape) - 1)
    i1 = np.clip(np.ceil(I.max(0)).astype(int) + 1, 1, np.array(m.shape))
    sub = m[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
    Ac = A.copy(); Ac[:3, 3] = i2w(A, i0[None].astype(float))[0]
    edt = ndimage.distance_transform_edt(sub, sampling=sp).astype(np.float32)
    cost = np.where(sub, 1.0 + (R_REF / (edt + 0.2)) ** 2, GAP_COST).astype(np.float64)
    hu_sub = None
    if ts is not None and ts.get("ct") is not None:
        # CT-guided gap bridging: outside the mask, contrast-filled voxels (HU >= 0.5 x the CCA median,
        # <= 900) cost GAP_COST_HU instead of GAP_COST, so a bridge follows the opacified lumen
        ctA, ctV, cca = ts["ct_affine"], ts["ct"], ts["cca_hu"]
        gi2 = np.indices(sub.shape, dtype=np.float32).reshape(3, -1)
        Wg = Ac @ np.r_[gi2, np.ones((1, gi2.shape[1]), np.float32)]
        Ig = (np.linalg.inv(ctA) @ Wg)[:3]
        hu_sub = ndimage.map_coordinates(ctV, Ig, order=1, mode="nearest").reshape(sub.shape)
        del gi2, Wg, Ig
        vessel_like = (hu_sub >= 0.5 * cca) & (hu_sub <= 900)
        cost[~sub & vessel_like] = GAP_COST_HU
    # impassable: contralateral half-space below z_FM - 5
    gi = np.indices(sub.shape, dtype=np.float32)
    Wx = Ac[0, 3] + Ac[0, 0] * gi[0]
    Wz = Ac[2, 3] + Ac[2, 2] * gi[2]
    Wy = Ac[1, 3] + Ac[1, 1] * gi[1]
    del gi
    z_end_cap = float(ba["end_world"][2]) + 3.0
    xm = mid_fn(Wz) if mid_fn is not None else x_mid
    forbid = ((sgn * (Wx - xm) < -3.0) & (Wz < z_fm - 5)) | (Wz > z_end_cap)
    del xm
    # carotid tubes (Arterial labels), below z_FM - 10, unless close to an own-side VA label
    own_va = lab.get("all_points") if lab.get("present") else None
    kd_va = cKDTree(own_va) if own_va is not None and len(own_va) else None
    car = []
    for si, s in enumerate(segs):
        if s["class"] in CAROTID and (car_ok is None or si in car_ok):
            keep = s["P"][:, 2] < z_fm - 10
            if kd_va is not None:
                d, _ = kd_va.query(s["P"])
                keep &= d > 5.0
            if keep.any():
                car.append(np.c_[s["P"][keep], s["R"][keep] + 1.5])
    n_car = 0
    if car:
        C = np.concatenate(car)
        Ci = np.rint(w2i(Ac, C[:, :3])).astype(int)
        inb = np.all((Ci >= 0) & (Ci < np.array(sub.shape)), axis=1)
        C, Ci = C[inb], Ci[inb]
        if len(C):
            seed = np.zeros(sub.shape, bool)
            ii = Ci
            seed[ii[:, 0], ii[:, 1], ii[:, 2]] = True
            rad = np.zeros(sub.shape, np.float32); rad[ii[:, 0], ii[:, 1], ii[:, 2]] = C[:, 3]
            d, ind = ndimage.distance_transform_edt(~seed, sampling=sp, return_indices=True)
            rr = rad[ind[0], ind[1], ind[2]]
            tube = d <= rr
            n_car = int(tube.sum())
            forbid |= tube
            del d, ind, rr, rad, seed
    # TS masks: common carotids, and the screening-trace corridor below the C1 top
    if ts is not None:
        for key in ("cca_R", "cca_L"):
            if ts.get(key) is not None:
                forbid |= ts[key](Ac, sub.shape)
        if False and ts.get("trace", {}).get(side) is not None:  # corridor disabled: the screening trace is QC only
            T = ts["trace"][side]
            Ti = w2i(Ac, T)
            inb = np.all((Ti >= 0) & (Ti < np.array(sub.shape)), axis=1)
            if inb.sum() > 5:
                # densify the trace
                Td, _, _ = resample(T, 0.5)
                Tdi = np.rint(w2i(Ac, Td)).astype(int)
                ok = np.all((Tdi >= 0) & (Tdi < np.array(sub.shape)), axis=1)
                seed = np.zeros(sub.shape, bool); seed[Tdi[ok, 0], Tdi[ok, 1], Tdi[ok, 2]] = True
                dtr = ndimage.distance_transform_edt(~seed, sampling=sp)
                forbid |= (dtr > 10.0) & (Wz < ts["z_c1_top"])
                del dtr, seed
    # never block the first 10 mm around the start points (the VA origin sits next to the CCA origins)
    Sw = w2i(Ac, starts_w)
    okS = np.all((Sw >= -20) & (Sw < np.array(sub.shape) + 20), axis=1)
    if okS.any():
        seedS = np.zeros(sub.shape, bool)
        Si = np.clip(np.rint(Sw[okS]).astype(int), 0, np.array(sub.shape) - 1)
        seedS[Si[:, 0], Si[:, 1], Si[:, 2]] = True
        near_start = ndimage.distance_transform_edt(~seedS, sampling=sp) <= 10.0
        n_unblocked = int((forbid & near_start).sum())
        forbid &= ~near_start
    else:
        n_unblocked = 0
    cost[forbid] = np.inf
    # starts / ends in the crop, snapped to mask voxels
    msk_idx = np.argwhere(sub)
    kd_m = cKDTree(msk_idx * sp)

    def snap(W, maxd=3.0):
        Ii = w2i(Ac, W)
        d, j = kd_m.query(Ii * sp)
        ok = (d <= maxd)
        pts = msk_idx[j[ok]]
        pts = pts[np.isfinite(cost[pts[:, 0], pts[:, 1], pts[:, 2]])]
        return np.unique(pts, axis=0)
    S = snap(starts_w)
    E = snap(ends_w, 1.0)
    if len(S) == 0 or len(E) == 0:
        return {"ok": False, "why": "start (%d) or end (%d) not in the crop / mask" % (len(S), len(E))}
    mcp = MCP_Geometric(cost, sampling=tuple(sp))
    cum, _ = mcp.find_costs([tuple(x) for x in S], [tuple(x) for x in E], find_all_ends=False)
    ec = cum[E[:, 0], E[:, 1], E[:, 2]]
    if not np.isfinite(ec).any():
        return {"ok": False, "why": "no finite path from the start to the basilar"}
    e = E[int(np.argmin(ec))]
    path = np.array(mcp.traceback(tuple(e)))
    inm = sub[path[:, 0], path[:, 1], path[:, 2]]
    W = i2w(Ac, path.astype(float))
    seglen = np.r_[0, np.linalg.norm(np.diff(W, axis=0), axis=1)]
    gap_vessel_like_frac = None
    if hu_sub is not None and (~inm).any():
        hv = hu_sub[path[:, 0], path[:, 1], path[:, 2]]
        gl = (~inm)
        gap_vessel_like_frac = float(np.mean((hv[gl] >= 0.5 * ts["cca_hu"]) & (hv[gl] <= 900)))
    gaps = []
    run = 0.0
    for k in range(len(W)):
        if not inm[k]:
            run += seglen[k]
        elif run > 0:
            gaps.append(run); run = 0.0
    if run > 0:
        gaps.append(run)
    start_w = W[0].copy()
    # trim inside the subclavian lumen
    s_leave = 0.0
    if sa_pts is not None:
        kd_sa = cKDTree(sa_pts)
        d, j = kd_sa.query(W)
        out = d > sa_rad[j]
        k0 = int(np.argmax(out)) if out.any() else 0
        s_leave = float(arclen(W)[k0])
        W2 = W[k0:]
    else:
        W2 = W
    Pr, _, s = resample(W2, 0.5)
    Ps = smooth(Pr, 1.0, 0.5)
    Ps[0], Ps[-1] = Pr[0], Pr[-1]
    # local EDT radius along the path
    Pi = w2i(Ac, Ps)
    r_edt = ndimage.map_coordinates(edt, Pi.T, order=1, mode="nearest")
    # gap flag per final point: nearest raw path voxel outside the mask
    Wk = W[k0:] if sa_pts is not None else W
    ink = inm[k0:] if sa_pts is not None else inm
    _, jn = cKDTree(Wk).query(Ps)
    gap_flag = ~ink[jn]
    sP = arclen(Ps)
    gap_s = [[round(float(sP[i]), 1)] for i in np.where(np.diff(np.r_[0, gap_flag.astype(int)]) == 1)[0]]
    return {"ok": True, "gap_flag": gap_flag, "gap_at_s_mm": [g[0] for g in gap_s], "start_kind": start_kind, "start_on_sa_world": start_w.tolist(), "P": Ps,
            "r_edt": r_edt, "s_leave_sa_mm": round(s_leave, 2), "length_mm": float(arclen(Ps)[-1]),
            "gaps_mm": [round(g, 2) for g in gaps], "gap_total_mm": round(float(sum(gaps)), 2),
            "gap_max_mm": round(float(max(gaps) if gaps else 0.0), 2), "n_forbidden_carotid_vox": n_car,
            "n_forbidden_total_vox": int(forbid.sum()), "n_unblocked_near_start_vox": n_unblocked,
            "gap_ct_guided": hu_sub is not None, "gap_vessel_like_HU_frac": gap_vessel_like_frac,
            "end_world": W[-1].tolist(), "crop": {"i0": i0.tolist(), "i1": i1.tolist()},
            "_sub": sub, "_Ac": Ac, "_edt": edt}


def _cut(tr, k):
    tr["P"] = tr["P"][:k + 1]
    tr["r_edt"] = tr["r_edt"][:k + 1]
    if tr.get("gap_flag") is not None:
        tr["gap_flag"] = tr["gap_flag"][:k + 1]
    tr["length_mm"] = float(arclen(tr["P"])[-1])
    tr["end_world"] = tr["P"][-1].tolist()


def merge_paths(trs, x_mid, z_fm=None):
    """Both traces run into the basilar; the confluence is where they join for good (< tol to the end).
    Each VA is cut there; the shared stretch is the basilar. With one trace only, the VA is cut where it
    last reaches the midline (|x - x_mid| < 2.5 mm)."""
    R, L = trs["R"], trs["L"]
    info = {"method": None}
    if R.get("ok") and L.get("ok"):
        PR, PL = R["P"], L["P"]
        rb = float(np.median(np.r_[R["r_edt"][-20:], L["r_edt"][-20:]]))
        tol = max(1.0, 0.6 * rb)
        dR, _ = cKDTree(PL).query(PR)
        dL, _ = cKDTree(PR).query(PL)
        badR, badL = np.where(dR > tol)[0], np.where(dL > tol)[0]
        kR = int(badR[-1] + 1) if len(badR) else 0
        kL = int(badL[-1] + 1) if len(badL) else 0
        if kR < len(PR) - 4 and kL < len(PL) - 4:
            info.update({"method": "two traces join", "tol_mm": round(tol, 2),
                         "shared_basilar_length_mm": round(float(arclen(PR[kR:])[-1]), 1),
                         "merge_point_R": PR[kR].round(2).tolist(), "merge_point_L": PL[kL].round(2).tolist(),
                         "merge_points_apart_mm": round(float(np.linalg.norm(PR[kR] - PL[kL])), 2),
                         "merge_x_minus_mid_mm": round(float(0.5 * (PR[kR, 0] + PL[kL, 0]) - x_mid), 2),
                         "merge_z_mm": round(float(0.5 * (PR[kR, 2] + PL[kL, 2])), 2),
                         "basilar_points": PR[kR:].round(2).tolist()})
            if z_fm is not None:
                info["merge_above_zfm_mm"] = round(info["merge_z_mm"] - z_fm, 1)
            _cut(R, kR); _cut(L, kL)
            return info
        info["two_traces_did_not_join"] = True
        if np.linalg.norm(PR[-1] - PL[-1]) < 1.5:
            info.update({"method": "common end point (traces meet only at the basilar end point)",
                         "merge_point_R": PR[-1].round(2).tolist(), "merge_point_L": PL[-1].round(2).tolist(),
                         "merge_z_mm": round(float(0.5 * (PR[-1, 2] + PL[-1, 2])), 2),
                         "merge_x_minus_mid_mm": round(float(0.5 * (PR[-1, 0] + PL[-1, 0]) - x_mid), 2)})
            if z_fm is not None:
                info["merge_above_zfm_mm"] = round(info["merge_z_mm"] - z_fm, 1)
            return info
    for side, tr in (("R", R), ("L", L)):
        if tr.get("ok"):
            P = tr["P"]
            off = np.abs(P[:, 0] - x_mid) >= 2.5
            k = int(np.where(off)[0][-1] + 1) if off.any() else len(P) - 1
            k = min(k, len(P) - 1)
            info["single_" + side] = {"cut_index": k, "cut_point": P[k].round(2).tolist()}
            _cut(tr, k)
    info["method"] = "midline entry (no second trace to join)"
    return info


def cut_surface(tr, sp, out_dir, name):
    sub, Ac, edt = tr["_sub"], tr["_Ac"], tr["_edt"]
    P = tr["P"]
    r = ndimage.median_filter(np.clip(tr["r_edt"], 0.7, 4.0), size=11, mode="nearest")
    rc = 1.8 * r + 1.0
    Pi = w2i(Ac, P)
    ii = np.rint(Pi).astype(int)
    ok = np.all((ii >= 0) & (ii < np.array(sub.shape)), axis=1)
    seed = np.zeros(sub.shape, bool); seed[ii[ok, 0], ii[ok, 1], ii[ok, 2]] = True
    idmap = np.full(sub.shape, -1, np.int32); idmap[ii[ok, 0], ii[ok, 1], ii[ok, 2]] = np.where(ok)[0]
    d, ind = ndimage.distance_transform_edt(~seed, sampling=sp, return_indices=True)
    near = idmap[ind[0], ind[1], ind[2]]
    del ind
    cap = sub & (d <= rc[near])
    # bridged gaps: painted as balls of the neighbouring radius (synthetic; recorded)
    gf = tr.get("gap_flag")
    n_syn = 0
    if gf is not None and gf.any():
        rfill = np.empty(len(P))
        for i in range(len(P)):
            lo_, hi_ = max(0, i - 10), min(len(P), i + 11)
            ok_ = ~gf[lo_:hi_]
            rfill[i] = np.median(r[lo_:hi_][ok_]) if ok_.any() else np.median(r[~gf]) if (~gf).any() else 1.0
        dpt = d <= 0  # placeholder shape
        ball = np.zeros(sub.shape, bool)
        for i in np.where(gf)[0]:
            c = Pi[i]
            rad_v = rfill[i] / sp
            lo_ = np.maximum(np.floor(c - rad_v).astype(int), 0)
            hi_ = np.minimum(np.ceil(c + rad_v).astype(int) + 1, np.array(sub.shape))
            if np.any(hi_ <= lo_):
                continue
            g = np.mgrid[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]].astype(np.float32)
            dd = np.sqrt(sum(((g[a] - c[a]) * sp[a]) ** 2 for a in range(3)))
            ball[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]] |= dd <= rfill[i]
        n_syn = int((ball & ~cap).sum())
        cap |= ball
        del dpt
    gi = np.indices(sub.shape, dtype=np.float32)
    W = np.stack([Ac[a, 3] + Ac[a, a] * gi[a] for a in range(3)], -1)
    del gi
    t = tangents(P)
    n_end = int(round(8.0 / 0.5))
    m0 = near <= n_end
    cap &= ~(m0 & (np.einsum("...k,k->...", W - P[0], t[0]) < 0))
    m1 = near >= len(P) - 1 - n_end
    cap &= ~(m1 & (np.einsum("...k,k->...", W - P[-1], t[-1]) > 0))
    del W
    lab, n = ndimage.label(cap)
    if n == 0:
        return None
    hits = lab[ii[ok, 0], ii[ok, 1], ii[ok, 2]]
    hits = hits[hits > 0]
    keep = np.bincount(hits).argmax() if len(hits) else np.bincount(lab.ravel())[1:].argmax() + 1
    va = ndimage.binary_fill_holes(lab == keep)
    frac_path_in = float(np.mean(va[ii[ok, 0], ii[ok, 1], ii[ok, 2]]))
    pad = np.pad(va, 1).astype(np.uint8)
    verts, faces, _, _ = measure.marching_cubes(pad, level=0.5)
    verts -= 1.0
    world = i2w(Ac, verts)
    if np.linalg.det(Ac[:3, :3]) < 0:
        faces = faces[:, [0, 2, 1]]
    vtp = os.path.join(out_dir, name + "_stageA.vtp")
    npts = write_vtp(world, faces, vtp)
    seeds = {"source": P[0].tolist(), "target": P[-1].tolist(),
             "note": "art_extract: traced VA origin (left the SA lumen) -> confluence / BA entry"}
    json.dump(seeds, open(os.path.join(out_dir, name + "_seeds.json"), "w"))
    # the cut-out mask (crop) for later reuse
    np.savez_compressed(os.path.join(out_dir, name + "_mask.npz"), mask=va, affine=Ac)
    return {"vtp": vtp, "surface_points": int(npts), "mask_voxels": int(va.sum()), "synthetic_gap_voxels": n_syn,
            "synthetic_gap_length_mm": round(float(gf.sum() * 0.5), 1) if gf is not None else 0.0,
            "volume_mm3": float(va.sum() * np.prod(sp)), "frac_path_in_cut": round(frac_path_in, 4)}


# ----------------------------------------------------------------------------- TS helpers
def load_ts(ts_dir, A_case, shape_case):
    seg = os.path.join(ts_dir, "segmentations")
    out = {"dir": ts_dir}

    def ld(nm):
        p = os.path.join(seg, nm + ".nii.gz")
        if not os.path.isfile(p):
            return None, None
        im = nib.load(p)
        return np.asanyarray(im.dataobj) > 0, im.affine
    out["masks"] = {}
    ctp = os.path.join(ts_dir, "ct.nii.gz")
    if os.path.isfile(ctp):
        ci = nib.load(ctp)
        out["ct"] = np.asanyarray(ci.dataobj).astype(np.float32)
        out["ct_affine"] = ci.affine
        sid_ = os.path.basename(ts_dir.rstrip("/"))
        try:
            out["cca_hu"] = float(json.load(open("/sp/v5_va/results/ts_screen/%s.json" % sid_))["cca_med_HU"])
        except Exception:
            out["cca_hu"] = 300.0
    for nm in ["subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right",
               "common_carotid_artery_left", "skull", "spinal_cord"] + ["vertebrae_C%d" % k for k in range(1, 8)]:
        mm, aa = ld(nm)
        if mm is not None:
            out["masks"][nm] = (mm, aa)
    A_ts = next(iter(out["masks"].values()))[1]
    out["A"] = A_ts
    out["sp"] = np.sqrt((A_ts[:3, :3] ** 2).sum(0))
    # C1 top
    if "vertebrae_C1" in out["masks"]:
        mm, aa = out["masks"]["vertebrae_C1"]
        W = i2w(aa, np.argwhere(mm).astype(float))
        out["z_c1_top"] = float(W[:, 2].max())
    # CCA forbid functions (mask dilated by 1.5 mm, resampled onto a crop grid)

    def make_forbid(nm):
        if nm not in out["masks"]:
            return None
        mm, aa = out["masks"][nm]
        dd = ndimage.distance_transform_edt(~mm, sampling=out["sp"]) <= 1.5

        def f(Ac, shape):
            gi = np.indices(shape, dtype=np.float32).reshape(3, -1)
            W = (Ac @ np.r_[gi, np.ones((1, gi.shape[1]), np.float32)])[:3]
            I = np.linalg.inv(aa) @ np.r_[W, np.ones((1, W.shape[1]))]
            v = ndimage.map_coordinates(dd.astype(np.uint8), I[:3], order=0, mode="constant", cval=0)
            return v.reshape(shape).astype(bool)
        return f
    out["cca_R"] = make_forbid("common_carotid_artery_right")
    out["cca_L"] = make_forbid("common_carotid_artery_left")
    # screening trace (ts agent), world mm
    sid = os.path.basename(ts_dir.rstrip("/"))
    tp = "/sp/v5_va/results/ts_screen/%s_paths.npz" % sid
    out["trace"] = {}
    if os.path.isfile(tp):
        d = np.load(tp)
        for side in ("R", "L"):
            if side in d.files and len(d[side]) > 3:
                out["trace"][side] = i2w(d["affine"], d[side].astype(float))
    # subclavian skeleton-ish centre points (fallback starts): mask voxels with max EDT per slab
    out["sa_centre"] = {}
    for side, nm in (("R", "subclavian_artery_right"), ("L", "subclavian_artery_left")):
        if nm in out["masks"]:
            mm, aa = out["masks"][nm]
            out["sa_centre"][side] = i2w(aa, np.argwhere(mm).astype(float))
    return out


def ts_qc(tr, ts, side, z_fm):
    """QC of one traced VA against the TotalSegmentator masks."""
    P = tr["P"]
    q = {}
    nm = "subclavian_artery_right" if side == "R" else "subclavian_artery_left"
    if nm in ts["masks"]:
        mm, aa = ts["masks"][nm]
        d = ndimage.distance_transform_edt(~mm, sampling=ts["sp"])
        for key, pt in (("junction", np.array(tr["start_on_sa_world"])), ("origin", P[0])):
            I = w2i(aa, pt[None])[0]
            q["%s_to_TS_SA_mask_mm" % key] = round(float(ndimage.map_coordinates(d, I[:, None], order=1, mode="nearest")[0]), 2)
    # vertebra enclosure, 16 rays in the normal plane, 8 mm, every 1 mm
    Pr, _, s = resample(P, 1.0)
    T = tangents(Pr, 3.0, 1.0)
    a = np.cross(T, [0, 0, 1.0]); bad = np.linalg.norm(a, axis=1) < 1e-3
    a[bad] = np.cross(T[bad], [1.0, 0, 0]); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(T, a)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    steps = np.arange(0.5, 8.01, 0.5)
    levels = {}
    enc_all = {}
    for k in range(7, 0, -1):
        key = "vertebrae_C%d" % k
        if key not in ts["masks"]:
            continue
        mm, aa = ts["masks"][key]
        Ainv = np.linalg.inv(aa)
        hit = np.zeros((len(Pr), 16), bool)
        for j, th in enumerate(ang):
            dirs = np.cos(th) * a + np.sin(th) * b
            for st in steps:
                X = Pr + st * dirs
                I = (Ainv @ np.c_[X, np.ones(len(X))].T)[:3]
                v = ndimage.map_coordinates(mm.astype(np.uint8), I, order=0, mode="constant", cval=0) > 0
                hit[:, j] |= v
        enc = hit.mean(1)
        enc_all[k] = enc
        Wm = i2w(aa, np.argwhere(mm).astype(float))
        zr = (Wm[:, 2].min(), Wm[:, 2].max())
        inlev = (Pr[:, 2] >= zr[0] - 3) & (Pr[:, 2] <= zr[1] + 3)
        dmask = ndimage.distance_transform_edt(~mm, sampling=ts["sp"])
        dd = ndimage.map_coordinates(dmask, (Ainv @ np.c_[Pr, np.ones(len(Pr))].T)[:3], order=1, mode="nearest")
        levels["C%d" % k] = {"z_range_mm": [round(zr[0], 1), round(zr[1], 1)], "n_points_in_level": int(inlev.sum()),
                             "max_enclosure": round(float(enc[inlev].max()) if inlev.any() else 0.0, 3),
                             "median_dist_to_vertebra_mm": round(float(np.median(dd[inlev])) if inlev.any() else -1, 2),
                             "s_enclosed_mm": [round(float(s[i]), 1) for i in np.where(inlev & (enc >= 0.5))[0][[0, -1]]]
                             if (inlev & (enc >= 0.5)).any() else None}
    q["vertebra_levels"] = levels
    passed = [L for L, v in levels.items() if v["max_enclosure"] >= 0.5]
    q["foramen_levels_passed"] = sorted(passed, key=lambda x: -int(x[1:]))
    q["n_C6_to_C1_passed"] = int(sum(1 for L in ("C6", "C5", "C4", "C3", "C2", "C1") if L in passed))
    # landmarks
    lm = {}
    entry = None
    for k in (7, 6, 5, 4, 3):
        if k in enc_all and "C%d" % k in passed and levels["C%d" % k]["s_enclosed_mm"]:
            entry = k
            break
    if entry is not None:
        lm["C6"] = levels["C%d" % entry]["s_enclosed_mm"][0]
        q["foramen_entry_level"] = "C%d" % entry
    for k, key in ((2, "C2"), (1, "C1")):
        if levels.get("C%d" % k, {}).get("s_enclosed_mm"):
            lm[key] = levels["C%d" % k]["s_enclosed_mm"][1]
    if "C1" in lm:
        after = np.where((s > lm["C1"]) & (Pr[:, 2] >= z_fm))[0]
        if len(after):
            lm["dura"] = round(float(s[after[0]]), 1)
    q["landmarks_mm"] = lm
    q["landmarks_complete"] = all(k in lm for k in ("C6", "C2", "C1", "dura")) and \
        lm.get("C6", 0) < lm.get("C2", 0) < lm.get("C1", 0) < lm.get("dura", 0) < s[-1]
    # screening-trace agreement
    if ts.get("trace", {}).get(side) is not None:
        Tt = ts["trace"][side]
        d, _ = cKDTree(Pr).query(resample(Tt, 1.0)[0])
        q["ts_trace_to_path_mm"] = {"median": round(float(np.median(d)), 2), "p90": round(float(np.percentile(d, 90)), 2),
                                    "frac_within_3mm": round(float(np.mean(d <= 3)), 3)}
    return q


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("out")
    ap.add_argument("--ts", default=None)
    ap.add_argument("--zfm", type=float, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    seg = nib.load(os.path.join(a.case, "extracranial_vessels", "segmentation.nii.gz"))
    m = np.asanyarray(seg.dataobj) > 0
    A = seg.affine
    sp = np.abs(np.diag(A)[:3])
    lcp = os.path.join(a.case, "labelled_centerlines.json")
    # Arterial's centerline stage can fail after a good segmentation (s0960: "Centerline could not be extracted
    # for first model"); the binary mask is still traced, from the TS subclavian mask, with no labels
    segs = json.load(open(lcp))["segments"] if os.path.isfile(lcp) else []
    for s in segs:
        s["P"] = np.asarray(s["points_ras_mm"], float); s["R"] = np.asarray(s["radius_mm"], float)
    pos, inc = node_positions(segs)
    rep = json.load(open(os.path.join(a.case, "run_report.json")))
    no_labels = not os.path.isfile(lcp)
    prep = json.load(open(os.path.join(a.case, "prep.json")))
    out = {"case": a.case, "tag": a.tag or os.path.basename(a.case.rstrip("/")), "grid_mm": sp.tolist(),
           "mask_shape": list(m.shape), "mask_voxels": int(m.sum()), "arterial_times_s": rep.get("times_s"),
           "arterial_peak_ram_gb": rep.get("peak_ram_gb_sampled"), "arterial_tta": rep.get("tta"),
           "source_spacing_mm": prep.get("source_spacing_mm", prep.get("spacing_mm")),
           "mask_components": None, "class_lengths_mm": {},
           "arterial_labels": "none (Arterial centerline stage failed: %s)" % (rep.get("errors") or {}).get("cl", "?")
           if no_labels else "labelled_centerlines.json"}
    lab_all, nlab = ndimage.label(m, structure=np.ones((3, 3, 3)))
    cnt = np.bincount(lab_all.ravel())[1:]
    out["mask_components"] = {"n": int(nlab), "largest_frac": round(float(cnt.max() / cnt.sum()), 4) if nlab else 0}
    del lab_all
    for s in segs:
        out["class_lengths_mm"][s["class"]] = round(out["class_lengths_mm"].get(s["class"], 0) + s["length_mm"], 1)
    ts = load_ts(a.ts, A, m.shape) if a.ts else None
    # midline and foramen magnum
    sk_path = os.path.join(a.case, "skull_mask.nii.gz")
    sk = np.asanyarray(nib.load(sk_path).dataobj) > 0
    x_mid = float(i2w(A, np.argwhere(sk)[::50].astype(float))[:, 0].mean())
    if a.zfm is not None:
        z_fm = a.zfm; zfm_from = "--zfm"
    elif ts is not None and ts.get("z_c1_top") is not None:
        z_fm = ts["z_c1_top"] + 6.0; zfm_from = "TS C1 top + 6 mm"
    else:
        Wsk = i2w(A, np.argwhere(sk)[::20].astype(float))
        z_fm = float(np.percentile(Wsk[np.abs(Wsk[:, 0] - x_mid) < 8, 2], 1)); zfm_from = "skull mask midline p1"
    # neck midline: the TS spinal-cord centroid per axial level (a head turned or tilted moves the skull centroid);
    # the basilar search uses the cord at the foramen magnum
    mid_fn = None
    out["x_mid_skull_mm"] = round(x_mid, 2)
    if ts is not None and "spinal_cord" in ts["masks"]:
        mm_, aa_ = ts["masks"]["spinal_cord"]
        Wc = i2w(aa_, np.argwhere(mm_).astype(float))
        if len(Wc) > 50:
            zq = np.round(Wc[:, 2] / 1.5) * 1.5
            zz = np.unique(zq)
            xx = np.array([Wc[zq == z, 0].mean() for z in zz])
            xx = ndimage.median_filter(xx, size=5, mode="nearest")
            mid_fn = (lambda z, zz=zz, xx=xx: np.interp(z, zz, xx))
            x_mid = float(mid_fn(z_fm))
            out["midline_from"] = "TS spinal cord centroid per level (basilar search: the cord at z_FM)"
            out["cord_x_range_mm"] = [round(float(xx.min()), 1), round(float(xx.max()), 1)]
    if mid_fn is None:
        out["midline_from"] = "skull-mask centroid"
    out["x_mid_mm"] = round(x_mid, 2); out["z_fm_mm"] = round(z_fm, 2); out["z_fm_from"] = zfm_from
    ba = find_ba(m, A, sp, x_mid, z_fm)
    out["basilar_on_mask"] = None if ba is None else {k: v for k, v in ba.items() if k not in ("voxels", "world", "end_world")}
    out["ba_label"] = None
    ba_l = [s for s in segs if s["class"] == "BA"]
    if ba_l:
        Pb = np.concatenate([s["P"] for s in ba_l])
        out["ba_label"] = {"length_mm": round(float(sum(s["length_mm"] for s in ba_l)), 1),
                           "radius_median_mm": round(float(np.median(np.concatenate([s["R"] for s in ba_l]))), 2),
                           "z_range_mm": [round(float(Pb[:, 2].min()), 1), round(float(Pb[:, 2].max()), 1)],
                           "x_median_minus_mid_mm": round(float(np.median(Pb[:, 0]) - x_mid), 1)}
        if ba is not None:
            d, _ = cKDTree(ba["world"]).query(Pb)
            out["ba_label"]["frac_within_2mm_of_mask_BA"] = round(float(np.mean(d <= 2.0)), 3)
        # a plausible Arterial BA label (midline, above the foramen magnum, r >= 0.8, >= 10 mm) gives the
        # confluence directly: its lowest point is the VA end (both traces end there)
        bl = out["ba_label"]
        low = Pb[np.argmin(Pb[:, 2])]
        d_low = float(cKDTree(ba["world"]).query(low)[0]) if ba is not None else None
        bl["lowest_point_mm"] = low.round(2).tolist()
        bl["lowest_x_minus_mid_mm"] = round(float(low[0] - x_mid), 1)
        bl["lowest_to_mask_BA_mm"] = None if d_low is None else round(d_low, 2)
        plaus = (abs(low[0] - x_mid) < 8.0 and z_fm - 5 <= low[2] <= z_fm + 60
                 and bl["radius_median_mm"] >= 0.8 and bl["length_mm"] >= 10 and (d_low is None or d_low <= 3.0))
        bl["plausible"] = bool(plaus)
        if False and plaus:  # tested: the label bottom sat 4 mm below the true merge on VMR 1.5 mm; the join of the two traces is more exact
            if ba is None:
                ba = {"world": Pb}
            ba = dict(ba)
            ba["end_world"] = low
            ba["end_from"] = "Arterial BA label, lowest point (= the confluence node)"
            out["ba_target"] = {"from": ba["end_from"], "point_mm": low.round(2).tolist()}
    if ba is not None and "end_from" not in ba:
        ba["end_from"] = "mask basilar: highest voxel of the midline seed (|x - x_mid| < 5 mm, EDT >= 0.6, z_FM + 8..+55)"
        out["ba_target"] = {"from": ba["end_from"], "point_mm": np.round(ba["end_world"], 2).tolist()}
    out["sides"] = {}
    car_ok = true_carotids(segs, inc)
    out["carotid_labels_made_impassable"] = sorted(set(segs[i]["class"] for i in car_ok))
    out["carotid_labels_suspect_not_blocked"] = [{"class": s["class"], "length_mm": s["length_mm"],
                                                  "z_range_mm": [round(float(s["P"][:, 2].min()), 1), round(float(s["P"][:, 2].max()), 1)]}
                                                 for i, s in enumerate(segs) if s["class"] in CAROTID and i not in car_ok]
    labs, trs = {}, {}
    for side in ("R", "L"):
        labs[side] = labelled_va(segs, pos, inc, side)
        trs[side] = trace_side(side, m, A, sp, segs, labs[side], ba, x_mid, z_fm, ts, car_ok, mid_fn)
    out["confluence"] = merge_paths(trs, x_mid, z_fm)
    for side in ("R", "L"):
        name = side.lower() + "va"
        lab, tr = labs[side], trs[side]
        so = {"labelled": {k: v for k, v in lab.items() if k not in ("P", "R", "all_points", "origin_node_pos")}}
        if lab.get("present") and "P" in lab:
            so["labelled"]["points"] = np.round(lab["P"], 3).tolist()
            so["labelled"]["radius_mm"] = np.round(lab["R"], 3).tolist()
            so["labelled"]["radius_median_mm"] = round(float(np.median(lab["R"])), 3)
            so["labelled"]["origin_node"] = np.round(lab["origin_node_pos"], 3).tolist()
        so["trace"] = {k: v for k, v in tr.items() if not k.startswith("_") and k not in ("P", "r_edt", "gap_flag")}
        if tr.get("ok"):
            so["trace"]["points"] = np.round(tr["P"], 3).tolist()
            so["trace"]["r_edt_mm"] = np.round(tr["r_edt"], 3).tolist()
            so["trace"]["r_edt_median_mm"] = round(float(np.median(tr["r_edt"])), 3)
            if lab.get("present") and "P" in lab:
                d1, _ = cKDTree(tr["P"]).query(resample(lab["P"], 0.5)[0])
                d2, _ = cKDTree(lab["P"]).query(tr["P"])
                so["trace"]["labelled_within_2mm_of_trace"] = round(float(np.mean(d1 <= 2)), 3)
                so["trace"]["trace_within_2mm_of_labelled"] = round(float(np.mean(d2 <= 2)), 3)
            cs = cut_surface(tr, sp, a.out, name)
            so["surface"] = cs
            if ts is not None:
                so["ts_qc"] = ts_qc(tr, ts, side, z_fm)
        out["sides"][side] = so
        print("%s %s: labelled %s %.1f mm | trace %s %s" % (
            out["tag"], side, lab.get("present"), lab.get("length_mm", 0), tr.get("ok"),
            ("%.1f mm gap %.1f/%.1f" % (tr["length_mm"], tr["gap_total_mm"], tr["gap_max_mm"])) if tr.get("ok") else tr.get("why")),
            flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, "va_extract.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"tag": out["tag"], "x_mid": out["x_mid_mm"], "z_fm": out["z_fm_mm"], "ba": out["basilar_on_mask"],
                      "mask_components": out["mask_components"], "seconds": out["seconds"]}))


if __name__ == "__main__":
    main()
