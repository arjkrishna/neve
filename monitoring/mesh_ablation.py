"""Where does the re-mesher lose lumen clearance? Ablation over the pipeline.

HYPOTHESIS (from reading the code):
  generate_mesh marks spheres of radius r, then gaussian_smooth(sigma=1 VOXEL) TWICE
  (effective sigma ~1.41 voxels = 0.85mm in x/y, 1.27mm in z), then calls
  get_surface_mesh with level=None. skimage's marching_cubes with level=None uses
  (min+max)/2 over the WHOLE array. The global max is set by the AORTA (r=5.8mm, ~10
  voxels across) where blurring barely dents the peak. But a distal vessel of r=1.25mm
  is only ~2 voxels in radius, so double-blurring drops its PEAK well below 1.
  Thresholding a low-amplitude thin tube at a level calibrated by a wide vessel
  systematically erodes the thin one. Predicts: proximal clearance fine, distal eroded.

Measures, for each setting: clearance at matched centerline stations vs the ORIGINAL
.obj, blocked-station count, triangle edge stats, voxel-grid size, runtime.
"""
import sys, time
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")

from eve.intervention.vesseltree.util.voxelcube import (
    create_empty_voxel_cube_from_branches)
from eve.intervention.vesseltree.util.meshing import get_surface_mesh
from eve_bench.dualdevicenav import DualDeviceNav
from eve_bench.dualdevicenavrccavaried import DualDeviceNavRCCAVaried

WIRE_R = 0.18


def dense(m, per=26):
    f = m.faces.reshape(-1, 4)[:, 1:]
    v = m.points
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    rg = np.random.default_rng(0)
    u = rg.random((per, len(f), 1)); w = rg.random((per, len(f), 1))
    k = u + w > 1; u = np.where(k, 1 - u, u); w = np.where(k, 1 - w, w)
    return np.vstack([v, (a + u * (b - a) + w * (c - a)).reshape(-1, 3)])


def edges(m):
    ed = m.extract_all_edges(); L = ed.lines.reshape(-1, 3)
    return np.linalg.norm(ed.points[L[:, 1]] - ed.points[L[:, 2]], axis=1)


def profile(mesh, pts, rad, tag, t=None, shape=None):
    d, _ = cKDTree(dense(mesh)).query(pts)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    distal = s > 120
    e = edges(mesh)
    blocked = int((d < WIRE_R).sum())
    print(f"{tag:38s} | cells {mesh.n_cells:6d} edge~{np.median(e):5.2f} "
          f"| clear med {np.median(d):5.2f} p05 {np.percentile(d,5):5.2f} "
          f"min {d.min():5.2f} | DISTAL med {np.median(d[distal]):5.2f} "
          f"(r_true {np.median(rad[distal]):5.2f}, ratio {np.median(d[distal])/np.median(rad[distal]):4.2f}) "
          f"| blocked {blocked:3d}" + (f" | {t:5.1f}s" if t else "")
          + (f" | grid {shape}" if shape is not None else ""))
    return d, s


def build(branches, spacing, n_smooth, sigma, decim, pad, level):
    t0 = time.time()
    cube = create_empty_voxel_cube_from_branches(branches, spacing)
    for _ in range(5):
        cube.add_padding_layer_all_sides()
    shape = tuple(cube.value_array.shape)
    nvox = int(np.prod(shape))
    if nvox > 700e6:
        print(f"  SKIP spacing={spacing}: grid {shape} = {nvox/1e6:.0f}M voxels (too large)")
        return None, None, None
    for b in branches:
        cube.mark_centerline_in_array(np.asarray(b.coordinates),
                                      np.asarray(b.radii), 1, pad)
    for _ in range(n_smooth):
        cube.gaussian_smooth(sigma)
    m = get_surface_mesh(cube, "descent", level)
    if decim and decim > 0:
        m = m.decimate(decim)
    return m.triangulate().clean(), time.time() - t0, shape


# ---- reference: the ORIGINAL segmented surface, and the stated radii ----
orig = DualDeviceNav()
ovt = orig.vessel_tree
obr = next(b for b in ovt.branches if "RCCA" in str(b.name).upper())
PTS = np.asarray(obr.coordinates, float)
RAD = np.asarray(obr.radii, float)
om = pv.read(ovt.mesh_path).triangulate().clean()
print("=" * 150)
profile(om, PTS, RAD, "ORIGINAL .obj  << TARGET >>")
print("=" * 150)

iv = DualDeviceNavRCCAVaried(seed=900000, episodes_between_change=10 ** 9,
                             base_amp_mm=0.0, tortuosity_mean_sigma=(0.0, 0.0),
                             tortuosity_clip=(0.0, 0.0),
                             radius_scale_mean_sigma=(1.0, 0.0))
vt = iv.vessel_tree
vt.rva_amp_mm = 0.0
vt._generate()
BR = list(vt.branches)

# ---- ablation ----
CASES = [
    ("CURRENT  vox.6/.6/.9 sm2 s1 dec.99 pad0", (0.6, 0.6, 0.9), 2, 1.0, 0.99, 0.0, None),
    ("  level=0.5 explicit",                    (0.6, 0.6, 0.9), 2, 1.0, 0.99, 0.0, 0.5),
    ("  smooth x1",                             (0.6, 0.6, 0.9), 1, 1.0, 0.99, 0.0, None),
    ("  smooth x0 (none)",                      (0.6, 0.6, 0.9), 0, 1.0, 0.99, 0.0, None),
    ("  sigma 0.5, x1",                         (0.6, 0.6, 0.9), 1, 0.5, 0.99, 0.0, None),
    ("  no decimate",                           (0.6, 0.6, 0.9), 2, 1.0, 0.0,  0.0, None),
    ("  decimate 0.5",                          (0.6, 0.6, 0.9), 2, 1.0, 0.5,  0.0, None),
    ("  pad +0.3mm",                            (0.6, 0.6, 0.9), 2, 1.0, 0.99, 0.3, None),
    ("  pad +0.5mm",                            (0.6, 0.6, 0.9), 2, 1.0, 0.99, 0.5, None),
    ("ISO 0.4 sm2 dec.99",                      (0.4, 0.4, 0.4), 2, 1.0, 0.99, 0.0, None),
    ("ISO 0.4 sm1 s0.5 dec.9",                  (0.4, 0.4, 0.4), 1, 0.5, 0.9,  0.0, None),
    ("ISO 0.3 sm1 s0.5 dec.9",                  (0.3, 0.3, 0.3), 1, 0.5, 0.9,  0.0, None),
    ("CANDIDATE iso.4 sm1 s0.5 dec.5 pad.15",   (0.4, 0.4, 0.4), 1, 0.5, 0.5,  0.15, 0.5),
]
for tag, sp, ns, sg, dc, pd, lv in CASES:
    try:
        m, t, shape = build(BR, sp, ns, sg, dc, pd, lv)
        if m is None:
            continue
        profile(m, PTS, RAD, tag, t, shape)
    except Exception as ex:
        print(f"{tag:38s} | FAILED: {type(ex).__name__}: {str(ex)[:70]}")
