#!/usr/bin/env python3
"""STAGE A (run in the BASE env): TopBrain label mask -> right-ICA surface + seeds.

The TopBrain release is voxel label masks, but the project's centerline pipeline
(vmr_processing_tools/extract_centerlines_vmtk.py) starts from a SURFACE and uses
VMTK for maximum-inscribed-sphere radii. So this stage only has to get us to a
surface plus the two endpoint seeds VMTK needs; stage B does the centerline.

Split across two conda envs on purpose: the base env has nibabel/scipy/skimage
but no vmtk, and vmtk_env has vmtk but none of those. Both have vtk, so a .vtp
surface is the handoff format.

  1. read mask, keep ICA (TopBrain label 8)
  2. keep the PATIENT-RIGHT component (LPS: +x runs to the patient's LEFT, so
     the right ICA is the component with the smaller mean x)
  3. marching cubes in WORLD millimetres via the NIfTI affine
  4. seeds: skeletonise and take the two ends of the trunk, ordered
     inferior -> superior (LPS +z is superior), i.e. proximal -> distal

    python mask_to_surface.py <mask_dir> --out <out_dir>
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

# Voxel values from the release's ITK-SNAP labelmap (authoritative; the paper's
# "ICA = 8" is the metro-map figure numbering, NOT the voxel value):
#   1 BA | 2 R-P1P2 | 3 L-P1P2 | 4 R-ICA | 5 R-M1 | 6 L-ICA | 7 L-M1 | ...
ICA_LABEL = 4      # R-ICA. Sides are labelled explicitly, so no side heuristic.


def right_component(mask, side="right"):
    """Largest connected component of the (already side-specific) label."""
    lab, k = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if k == 0:
        return None, "no component"
    sizes = ndimage.sum(mask, lab, range(1, k + 1))
    big = int(np.argmax(sizes)) + 1
    frac = sizes[big - 1] / sizes.sum()
    return lab == big, "largest of %d components (%.0f%% of voxels)" % (k, 100 * frac)


def trunk_ends(binary, affine):
    """World-mm coordinates of the two ends of the skeleton trunk."""
    from skimage.morphology import skeletonize
    skel = skeletonize(binary)
    pts = np.argwhere(skel)
    if len(pts) < 10:
        return None, "skeleton too small (%d)" % len(pts)
    index = {tuple(p): i for i, p in enumerate(pts)}
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
            if (a, b, c) != (0, 0, 0)]
    nbrs = [[] for _ in pts]
    for i, p in enumerate(pts):
        for o in offs:
            j = index.get((p[0] + o[0], p[1] + o[1], p[2] + o[2]))
            if j is not None:
                nbrs[i].append(j)

    def bfs(src):
        dist = [-1] * len(pts); dist[src] = 0; q = [src]; h = 0
        while h < len(q):
            u = q[h]; h += 1
            for v in nbrs[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1; q.append(v)
        return int(np.argmax(dist))

    a = bfs(0)
    b = bfs(a)
    ends = np.array([pts[a], pts[b]], dtype=float)
    world = (affine @ np.c_[ends, np.ones(2)].T).T[:, :3]
    if world[0, 2] > world[1, 2]:          # order inferior -> superior
        world = world[::-1]
    return world, None


def write_vtp(verts, faces, path):
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64), deep=1))
    cells = vtk.vtkCellArray()
    tri = np.c_[np.full(len(faces), 3), faces].astype(np.int64).ravel()
    cells.SetCells(len(faces), ns.numpy_to_vtkIdTypeArray(tri, deep=1))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    clean = vtk.vtkCleanPolyData(); clean.SetInputData(poly); clean.Update()
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(clean.GetOutput())
    smooth.SetNumberOfIterations(20)
    smooth.SetPassBand(0.1)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(path)
    w.SetInputData(smooth.GetOutput()); w.Write()
    return smooth.GetOutput().GetNumberOfPoints()


def process(mask_path, out_dir, label, side):
    import nibabel as nib
    from skimage import measure

    stem = os.path.basename(mask_path).replace(".nii.gz", "").replace(".nii", "")
    img = nib.load(mask_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))

    m = (data == label)
    if m.sum() == 0:
        return None, "label %d absent" % label
    comp, note = right_component(m, side)
    if comp is None:
        return None, note

    verts, faces, _, _ = measure.marching_cubes(comp.astype(np.uint8), level=0.5,
                                                spacing=tuple(spacing))
    # marching_cubes returns mm offsets from the voxel grid origin; map to world
    idx = verts / spacing
    world = (aff @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]

    ends, err = trunk_ends(comp, aff)
    if ends is None:
        return None, err

    vtp = os.path.join(out_dir, stem + "_rICA.vtp")
    n = write_vtp(world, faces, vtp)
    seeds = os.path.join(out_dir, stem + "_rICA_seeds.json")
    with open(seeds, "w") as f:
        json.dump({"source": ends[0].tolist(), "target": ends[1].tolist(),
                   "note": note, "surface_points": int(n)}, f)
    span = float(np.linalg.norm(ends[1] - ends[0]))
    return {"vtp": vtp, "pts": n, "span_mm": span, "note": note}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask_dir")
    ap.add_argument("--out", default="topbrain_data/surfaces")
    ap.add_argument("--label", type=int, default=ICA_LABEL)
    ap.add_argument("--side", default="right", choices=["right", "left"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    masks = []
    for root, _, files in os.walk(a.mask_dir):
        for f in sorted(files):
            if f.endswith((".nii", ".nii.gz")):
                masks.append(os.path.join(root, f))
    print("found %d nifti under %s" % (len(masks), a.mask_dir))

    ok = 0
    for p in masks:
        stem = os.path.basename(p)[:44]
        try:
            res, err = process(p, a.out, a.label, a.side)
        except Exception as e:                      # noqa: BLE001
            res, err = None, "%s: %s" % (type(e).__name__, e)
        if res is None:
            print("  SKIP %-46s %s" % (stem, err))
        else:
            ok += 1
            print("  OK   %-46s %6d surf pts  end-to-end %5.1f mm  [%s]"
                  % (stem, res["pts"], res["span_mm"], res["note"]))
    print("surfaces written: %d / %d" % (ok, len(masks)))


if __name__ == "__main__":
    sys.exit(main())
