#!/usr/bin/env python3
"""Centerline + radii -> collision surface, by signed distance, not by blur.

The mesher in eve/intervention/vesseltree/util/meshing.py marks each
centerline point as a binary sphere, Gaussian-smooths the cube twice and
iso-surfaces at the half level. That shrinks every vessel by 0.3-0.9 mm of
radius, more for vessels running in the axial plane (the blur is set in
voxels on a 0.6/0.6/0.9 grid), and deletes anything under ~1.2 mm. Measured
on the shipped anatomies: a median 0.64 mm radius deficit along the RCCA
route, and a 0.35 mm lumen 12 mm short of the terminus where 1.17 mm is
declared -- narrower than the catheter. See MESHING_PIPELINE_ANALYSIS.md.

This module evaluates the SIGNED DISTANCE to the union of centerline spheres
(the tube function VMTK's vmtkcenterlinemodeller computes) on an isotropic
grid, only in a band around the vessel, and iso-surfaces it at zero. No
smoothing pass, no data-dependent level. Measured deficit: 0.02-0.06 mm.

    field = tube_field(branches, spacing=0.45)
    mesh  = iso_surface(field)                # full resolution, one component
    mesh  = decimate_to(mesh, 60000)          # quadric, volume-preserving

The radius is looked up per BRANCH (nearest samples of each branch, max over
branches), not per global nearest sample: near a junction the nearest sample
by distance can belong to the thin vessel while a farther, fatter one governs
the union, and a global k-NN would carve a false cavity into the fat vessel.
"""
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

SPACING_MM = 0.45        # isotropic; the old mesher's 0.6/0.6/0.9 was 1.7x coarser along z
BAND_VOX = 5             # how far past the marked tube the field is evaluated; 4 left seams on the arch
SAMPLE_MM = 0.25         # centerline resampling for the sphere union
K_NN = 8                 # nearest samples per branch that compete for a voxel
PAD_VOX = 6
FAR = -1e3               # field value outside the band: solidly "outside"


class Field:
    def __init__(self, values, spacing, offset, region):
        self.values, self.spacing, self.offset, self.region = values, spacing, offset, region


def _resample(c, r, step):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    if d[-1] < step:
        return c, r
    q = np.arange(0.0, d[-1] + 1e-9, step)
    return (np.stack([np.interp(q, d, c[:, i]) for i in range(3)], 1), np.interp(q, d, r))


def _grid(branches, spacing):
    lo = np.min([np.min(np.asarray(b.coordinates) - np.asarray(b.radii)[:, None], 0) for b in branches], 0)
    hi = np.max([np.max(np.asarray(b.coordinates) + np.asarray(b.radii)[:, None], 0) for b in branches], 0)
    lo = np.floor(lo / spacing) * spacing - PAD_VOX * spacing
    shape = np.ceil((hi - lo) / spacing).astype(int) + 2 * PAD_VOX
    return lo, shape


def _mark(shape, lo, spacing, c, r):
    """Binary union of spheres on the grid, vectorised per centerline point."""
    m = np.zeros(shape, dtype=bool)
    for p, rad in zip(c, r):
        i0 = np.maximum(np.floor((p - rad - lo) / spacing).astype(int), 0)
        i1 = np.minimum(np.ceil((p + rad - lo) / spacing).astype(int) + 1, shape)
        if np.any(i1 <= i0):
            continue
        ax = [np.arange(i0[k], i1[k]) * spacing + lo[k] for k in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        inside = (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2 < rad * rad
        m[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= inside
    return m


def _coarse(c, r, spacing):
    """Sub-sample the dense line for MARKING only: a sphere every max(spacing,
    r/5) covers the tube to within r/200, and the marking only has to place
    the band. The field itself always uses the dense samples. (r/2 was tried:
    on a 15 mm arch the dips between spheres broke through a 4-voxel band and
    marching cubes left 3,500 open edges.)"""
    keep = [0]; acc = 0.0
    for i in range(1, len(c)):
        acc += float(np.linalg.norm(c[i] - c[i - 1]))
        if acc >= max(spacing, 0.2 * r[i]):
            keep.append(i); acc = 0.0
    if keep[-1] != len(c) - 1:
        keep.append(len(c) - 1)
    return c[keep], r[keep]


def tube_field(branches, spacing=SPACING_MM, band=BAND_VOX, sample_mm=SAMPLE_MM, k=K_NN, verbose=False):
    """Signed distance (positive inside) to the union of all branches' spheres,
    evaluated in a band of `band` voxels around the marked tube.

    Everything per branch happens inside that branch's own bounding box, not
    the whole grid: marking, dilation and the distance query. On a full tree
    the grid is ~200 M voxels and the arch's box is a tenth of it; the thin
    vessels' boxes are far smaller."""
    t0 = time.time()
    lo, shape = _grid(branches, spacing)
    values = np.full(shape, FAR, dtype=np.float32)
    region_all = np.zeros(shape, dtype=bool)
    for b in branches:
        c, r = _resample(np.asarray(b.coordinates, float), np.asarray(b.radii, float), sample_mm)
        if len(c) < 2:
            continue
        rmax = float(r.max())
        bmin = np.maximum(np.floor((c.min(0) - rmax - lo) / spacing).astype(int) - band - 2, 0)
        bmax = np.minimum(np.ceil((c.max(0) + rmax - lo) / spacing).astype(int) + band + 3, shape)
        sub_shape, sub_lo = bmax - bmin, lo + bmin * spacing
        cm, rm = _coarse(c, r, spacing)
        mark = _mark(sub_shape, sub_lo, spacing, cm, rm)
        region = ndimage.binary_dilation(mark, iterations=band)
        idx = np.argwhere(region)
        if not len(idx):
            continue
        coords = idx * spacing + sub_lo
        dist, nn = cKDTree(c).query(coords, k=min(k, len(c)))
        if dist.ndim == 1:
            dist, nn = dist[:, None], nn[:, None]
        f = (r[nn] - dist).max(axis=1).astype(np.float32)
        g = idx + bmin
        cur = values[g[:, 0], g[:, 1], g[:, 2]]
        values[g[:, 0], g[:, 1], g[:, 2]] = np.maximum(cur, f)
        region_all[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] |= region
    if verbose:
        print("   field %s at %.2f mm, %d band voxels, %.1f s"
              % ("x".join(map(str, shape)), spacing, int(region_all.sum()), time.time() - t0), flush=True)
    return Field(values, spacing, lo, region_all)


def iso_surface(field, level=0.0):
    """Marching cubes at ZERO, restricted to the band. Returns a pyvista mesh."""
    import pyvista as pv
    sp = (field.spacing,) * 3
    # marching_cubes only visits cubes whose eight corners are all in the mask.
    # DILATE the band, do not erode it: a FAR corner next to a negative one
    # makes no crossing, so the band edge cannot invent a surface, but an
    # eroded mask can CUT one -- the v3 real-surface union put iso-surface
    # within a voxel of the band edge and lost 49 edges and a stray triangle.
    mask = ndimage.binary_dilation(field.region, iterations=2)
    # allow_degenerate=False: zero-area triangles from corner-exact crossings
    # are what makes vtkQuadricDecimation refuse the mesh downstream.
    verts, faces, _, _ = measure.marching_cubes(field.values, level, spacing=sp,
                                                gradient_direction="descent", mask=mask,
                                                allow_degenerate=False)
    faces = np.c_[np.full(len(faces), 3), faces]
    m = pv.PolyData(verts, faces)
    m.translate(field.offset, inplace=True)
    return m


def decimate_to(mesh, n_tris):
    """Quadric decimation to a triangle count, volume-preserving."""
    if mesh.n_cells <= n_tris:
        return mesh
    return mesh.decimate(1.0 - n_tris / mesh.n_cells, volume_preservation=True)


def route_lumen(mesh, route, cap_mm=3.0):
    """Meshed lumen radius along a centerline route -- the distance from each
    point to the surface -- and whether each point is inside. End caps are
    excluded from the body mask: they are not lumen."""
    import pyvista as pv
    route = np.asarray(route, float)
    cloud = pv.PolyData(route)
    d = np.abs(np.asarray(cloud.compute_implicit_distance(mesh)["implicit_distance"]))
    ins = np.asarray(cloud.select_enclosed_points(mesh, tolerance=0.0, check_surface=False)["SelectedPoints"], bool)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    body = (s > cap_mm) & (s < s[-1] - cap_mm)
    return d, ins, body, s


def mesh_stats(mesh):
    conn = mesh.connectivity()
    return dict(tris=int(mesh.n_cells), comps=int(conn.cell_data["RegionId"].max()) + 1 if mesh.n_cells else 0,
                open_edges=int(mesh.n_open_edges))
