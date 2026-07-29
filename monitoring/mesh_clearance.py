"""Geometric passability — NO controller involved.

A controller failing proves nothing about the mesh. But clearance is a property of
the geometry alone: at each centerline station, the distance from the centerline to
the nearest point ON THE VESSEL SURFACE is the local lumen radius. If that is smaller
than the guidewire radius, no controller of any kind can pass — it is a necessary
condition, independent of skill.

Densely samples points ON the triangles (not just vertices — on a 6mm-triangle mesh
the vertices sit at the OUTSIDE of each facet and overestimate clearance badly).
"""
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/eve_training/eve")
sys.path.insert(0, "/opt/eve_training/eve_bench")

WIRE_R = 0.36 / 2.0     # guidewire RADIUS mm (0.36mm diameter device)


def dense_points(mesh, per_tri=28):
    """Sample points across every triangle face."""
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    v = mesh.points
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    rng = np.random.default_rng(0)
    u = rng.random((per_tri, len(f), 1))
    w = rng.random((per_tri, len(f), 1))
    m = u + w > 1
    u = np.where(m, 1 - u, u)
    w = np.where(m, 1 - w, w)
    pts = a + u * (b - a) + w * (c - a)
    return np.vstack([v, pts.reshape(-1, 3)])


def clearance(mesh, coords, radii, label):
    surf = dense_points(mesh)
    tree = cKDTree(surf)
    d, _ = tree.query(coords)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))])
    blocked = d < WIRE_R
    tight = d < WIRE_R * 2
    print(f"\n--- {label} ---")
    print(f"  surface sample pts {len(surf):,}   centerline stations {len(coords)}")
    print(f"  clearance: min {d.min():.2f}  p05 {np.percentile(d,5):.2f}  "
          f"median {np.median(d):.2f} mm")
    print(f"  stations with clearance < wire radius ({WIRE_R:.2f}mm): "
          f"{blocked.sum()}/{len(d)}")
    print(f"  stations with clearance < 2x wire radius: {tight.sum()}/{len(d)}")
    if blocked.any():
        first = s[blocked][0]
        print(f"  *** FIRST BLOCKED STATION at arclength {first:.1f} mm "
              f"(clearance {d[blocked][0]:.2f} mm) ***")
    else:
        print("  PASSABLE end-to-end on clearance grounds")
    # deepest reachable before first block
    if blocked.any():
        print(f"  reachable arclength before first block: {s[blocked][0]:.1f} "
              f"of {s[-1]:.1f} mm total")
    return d, s, blocked


# ---------- 1. the ORIGINAL patient collision mesh ----------
from eve_bench.dualdevicenav import DualDeviceNav
orig = DualDeviceNav()
ovt = orig.vessel_tree
om = pv.read(ovt.mesh_path).triangulate().clean()
obr = next(b for b in ovt.branches if "RCCA" in str(b.name).upper())
clearance(om, np.asarray(obr.coordinates, float), np.asarray(obr.radii, float),
          f"ORIGINAL patient mesh ({om.n_cells} cells)")

# ---------- 2. generated anatomies ----------
from eve_bench.dualdevicenavrccavaried import DualDeviceNavRCCAVaried
nblocked = 0
firsts = []
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for k in range(N):
    iv = DualDeviceNavRCCAVaried(seed=910000 + k * 10000, episodes_between_change=10**9)
    vt = iv.vessel_tree
    m = pv.read(vt.mesh_path).triangulate().clean()
    br = next(b for b in vt.branches if "RCCA" in str(b.name).upper())
    d, s, bl = clearance(m, np.asarray(br.coordinates, float),
                         np.asarray(br.radii, float), f"GENERATED #{k} ({m.n_cells} cells)")
    if bl.any():
        nblocked += 1
        firsts.append(s[bl][0])

print(f"\n{'='*60}\nSUMMARY: {nblocked}/{N} generated anatomies geometrically BLOCKED")
if firsts:
    print(f"  first-block arclengths: {[f'{f:.0f}' for f in firsts]}")
    print(f"  median first-block station: {np.median(firsts):.1f} mm")
