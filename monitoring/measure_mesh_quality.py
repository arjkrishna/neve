"""Measure how much the re-mesher shrinks the lumen, vs the true centerline radii.

The historical FIXED-mesh run loaded the patient surface directly and traversed the
siphon (46.2%). The procedural pipeline (Gen-4 onward, and my real_patient_anatomy
mode) rebuilds the surface from centerlines via
  voxelize [0.6,0.6,0.9] -> mark radius (radius_padding=0)
  -> gaussian_smooth(1) TWICE -> marching cubes -> decimate(0.99)
This measures the resulting effective lumen radius against the stated radius, by
station along the navigated path.
"""
import sys
import numpy as np

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/training_scripts")

sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenavrccavaried import DualDeviceNavRCCAVaried
import pyvista as pv

# exactly what eval_anatomies.py --real_patient_anatomy builds
interv = DualDeviceNavRCCAVaried(
    seed=900000, episodes_between_change=10 ** 9, base_amp_mm=0.0,
    tortuosity_mean_sigma=(0.0, 0.0), tortuosity_clip=(0.0, 0.0),
    radius_scale_mean_sigma=(1.0, 0.0),
)
vt = interv.vessel_tree
vt.rva_amp_mm = 0.0
vt._generate()

mesh = pv.read(vt.mesh_path)
print(f"mesh: {mesh.n_points} points, {mesh.n_cells} cells")

# the navigated branch: RCCA -> ICA -> siphon
target = None
for b in vt.branches:
    if "rcca" in b.name.lower():
        target = b
        break
if target is None:
    target = max(vt.branches, key=lambda b: len(b.coordinates))
print(f"branch '{target.name}': {len(target.coordinates)} centerline points")

pts = np.asarray(target.coordinates, dtype=float)
rad = np.asarray(target.radii, dtype=float)
seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])

tree = mesh.points  # surface vertices
print(f"\n{'arclen':>8} {'r_true':>8} {'r_mesh':>8} {'shrink':>8}")
print("-" * 36)
rows = []
for i in range(0, len(pts), max(1, len(pts) // 28)):
    d = np.linalg.norm(tree - pts[i], axis=1)
    r_eff = float(d.min())          # nearest surface point ~ local lumen radius
    rows.append((s[i], rad[i], r_eff))
    flag = ""
    if r_eff < 0.36:
        flag = "  <-- 0.36mm WIRE DOES NOT FIT"
    elif r_eff < 0.6:
        flag = "  <-- marginal"
    print(f"{s[i]:8.1f} {rad[i]:8.2f} {r_eff:8.2f} "
          f"{100*(1-r_eff/rad[i]):7.0f}%{flag}")

arr = np.array(rows)
distal = arr[arr[:, 0] > 120]
if len(distal):
    print(f"\nDISTAL (arclen > 120mm, n={len(distal)}):")
    print(f"  r_true  mean {distal[:,1].mean():.2f}  min {distal[:,1].min():.2f}")
    print(f"  r_mesh  mean {distal[:,2].mean():.2f}  min {distal[:,2].min():.2f}")
    print(f"  mean shrink {100*(1-distal[:,2].mean()/distal[:,1].mean()):.0f}%")
    tight = distal[distal[:, 2] < 0.36]
    print(f"  stations where 0.36mm wire does NOT fit: {len(tight)}/{len(distal)}")
    if len(tight):
        print(f"  first such station at arclen {tight[0,0]:.1f} mm "
              f"(r_mesh {tight[0,2]:.2f} vs r_true {tight[0,1]:.2f})")
