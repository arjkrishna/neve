#!/usr/bin/env python3
"""Fallback centerline from a label mask, for the few surfaces vmtkCenterlines fails on.

vmtkCenterlines returned a 2-4 point line for two right VAs (topcow_mr_004 and
_020) although their stage-A seeds are 36-45 mm apart on one connected label;
the Voronoi path search simply did not find a route. This computes the same
thing without VMTK: the longest path through the label's skeleton between its
two trunk ends (mask_to_surface's own seed rule), with the radius at each point
taken from the Euclidean distance map -- the radius of the largest inscribed
sphere, which is what MaximumInscribedSphereRadius measures. The path is
lightly smoothed (the raw skeleton steps voxel to voxel), resampled at 0.5 mm,
and written in vmtk_centerline.py's format with "method" recording the fallback.

    python3 topbrain_tools/skeleton_centerline.py <mask.nii.gz> --label 23 --out <file.json>
"""
import argparse
import json
import sys

import numpy as np
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask")
    ap.add_argument("--label", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=3, help="moving-average half-window, points")
    a = ap.parse_args()
    import nibabel as nib
    from skimage.morphology import skeletonize

    img = nib.load(a.mask)
    data = np.asarray(img.dataobj)
    aff = img.affine
    sp = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
    m = data == a.label
    lab, k = ndimage.label(m, structure=np.ones((3, 3, 3)))
    if k == 0:
        print("label absent"); return 1
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    edt = ndimage.distance_transform_edt(comp, sampling=sp)
    skel = skeletonize(comp)
    pts = np.argwhere(skel)
    index = {tuple(p): i for i, p in enumerate(pts)}
    offs = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
    nbrs = [[index[t] for t in ((p[0] + o[0], p[1] + o[1], p[2] + o[2]) for o in offs) if t in index] for p in pts]

    def bfs(src):
        prev = [-1] * len(pts); dist = [-1] * len(pts); dist[src] = 0; q = [src]; h = 0
        while h < len(q):
            u = q[h]; h += 1
            for v in nbrs[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1; prev[v] = u; q.append(v)
        return int(np.argmax(dist)), prev

    a0, _ = bfs(0)
    b0, prev = bfs(a0)
    path = [b0]
    while path[-1] != a0:
        path.append(prev[path[-1]])
    ijk = pts[path].astype(float)
    world = (aff @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    rad = edt[tuple(pts[path].T)]
    if world[0, 2] > world[-1, 2]:                  # proximal (inferior) -> distal
        world, rad = world[::-1], rad[::-1]
    w = a.smooth
    if w > 0 and len(world) > 2 * w + 1:
        ker = np.ones(2 * w + 1) / (2 * w + 1)
        sm = np.stack([np.convolve(np.pad(world[:, i], w, mode="edge"), ker, mode="valid") for i in range(3)], 1)
        world = sm
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(world, axis=0), axis=1))]
    t = np.arange(0.0, s[-1] + 1e-9, 0.5)
    p = np.stack([np.interp(t, s, world[:, i]) for i in range(3)], 1)
    r = np.interp(t, s, rad)
    L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    out = {"points": p.tolist(), "radii": r.tolist(), "length_mm": L, "n": int(len(p)),
           "diam_prox_mm": float(2 * r[0]), "diam_dist_mm": float(2 * r[-1]),
           "method": "skeleton-EDT fallback (vmtkCenterlines failed)", "label": a.label}
    json.dump(out, open(a.out, "w"))
    print("wrote %s: %.1f mm, %d pts, diam %.1f -> %.1f" % (a.out, L, len(p), 2 * r[0], 2 * r[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
