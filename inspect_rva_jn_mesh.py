"""Inspect the vessel mesh at the RVA-junction to validate Phase C design.

Key questions:
  1. Is RCCA's opening at the trifurcation larger than RVA's? (would explain
     why wires preferentially flicker INTO RCCA when J-curl is ambiguous)
  2. What is the lumen geometry at z=414-420 mm vessel-CS — separate lumens
     per daughter, or shared open cavity?
  3. Does RVA narrow before its first internal turn (~RVA[10])?
  4. Is there a flow-divider ridge between RVA and RCCA openings?

Run from worktree root inside or outside docker (only needs pyvista + numpy):
    python inspect_rva_jn_mesh.py
"""
import json
import os
import numpy as np

import pyvista as pv
try:
    pv.start_xvfb(wait=2)
except Exception as e:
    print(f"xvfb start failed (may be ok if --skip-render): {e}")
pv.OFF_SCREEN = True

CENTERLINE_DIR = "eve_bench/data/dualdevicenav/Centrelines_comb"
MESH_PATH = "eve_bench/data/dualdevicenav/vessel_architecture_visual.obj"

# Vessel-CS reference coordinates (from earlier analysis)
RVA_JN_VCS = np.array([-0.35, 24.14, 416.22])  # bridge(11)→RVA → RCCA junction
LCCA_JN_VCS = np.array([23.21, 15.75, 384.70])
TRUNK_TOP_VCS = np.array([46.66, 33.91, 391.95])

# The centerlines are stored in a (y, -z, -x) vessel-CS reframe of the
# raw mesh coordinates. The mesh itself is in raw (x, y, z). The transform
# applied to centerlines is:
#   vessel_x = raw_y
#   vessel_y = -raw_z
#   vessel_z = -raw_x
# So to express vessel-CS targets in mesh space: invert.
def vcs_to_mesh(vcs):
    """vessel-CS (vx, vy, vz) -> mesh raw (mx, my, mz)."""
    vx, vy, vz = vcs
    mx = -vz
    my = vx
    mz = -vy
    return np.array([mx, my, mz])

def mesh_to_vcs(m):
    """mesh raw -> vessel-CS."""
    mx, my, mz = m
    return np.array([my, -mz, -mx])

def load_centerlines():
    out = {}
    for fname in sorted(os.listdir(CENTERLINE_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CENTERLINE_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        pts_raw = []
        for m in data["markups"]:
            if m["type"] == "Curve":
                for cp in m["controlPoints"]:
                    pts_raw.append(cp["position"])
        if pts_raw:
            tag = fname.replace(".mrk.json", "").replace("Centerline curve ", "")
            out[tag] = np.array(pts_raw, dtype=float)
    return out


def cross_section_at_z(mesh, z_vcs, eps=0.5):
    """Slice the mesh at vessel-CS z = z_vcs (= mesh -y plane in raw coords).
    Returns a list of (centroid_vcs, area, n_points) for each connected
    region in the slice."""
    # vcs_z = -raw_x; so a vcs-z plane is a raw-x plane
    raw_x_target = -z_vcs
    plane_origin = (raw_x_target, 0, 0)
    plane_normal = (1, 0, 0)
    sliced = mesh.slice(normal=plane_normal, origin=plane_origin)
    if sliced is None or sliced.n_points == 0:
        return []
    bodies = sliced.connectivity().split_bodies()
    regions = []
    for body in bodies:
        pts = body.points
        if len(pts) < 3:
            continue
        # Convert centroid back to vessel-CS
        centroid_raw = pts.mean(axis=0)
        centroid_vcs = mesh_to_vcs(centroid_raw)
        # Approximate cross-section area via convex hull in (y, z) plane
        try:
            from scipy.spatial import ConvexHull
            hull_pts = pts[:, 1:3]
            if len(hull_pts) >= 3:
                hull = ConvexHull(hull_pts)
                area = float(hull.volume)  # 2D hull area
            else:
                area = 0.0
        except Exception:
            area = 0.0
        # Equivalent radius for circular area
        eq_radius = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
        regions.append({
            "centroid_vcs": centroid_vcs,
            "area_mm2": area,
            "eq_radius_mm": eq_radius,
            "n_points": len(pts),
        })
    return regions


def main():
    print(f"Loading mesh: {MESH_PATH}")
    mesh = pv.read(MESH_PATH)
    print(f"  n_points: {mesh.n_points}, n_cells: {mesh.n_cells}")
    print(f"  bounds (raw mesh coords): {mesh.bounds}")

    cls = load_centerlines()

    # Convert junction coords mesh-space for visualization
    rva_jn_mesh = vcs_to_mesh(RVA_JN_VCS)
    lcca_jn_mesh = vcs_to_mesh(LCCA_JN_VCS)
    trunk_top_mesh = vcs_to_mesh(TRUNK_TOP_VCS)
    print(f"\nRVA-jn vessel-CS: {RVA_JN_VCS} -> mesh: {rva_jn_mesh}")
    print(f"LCCA-jn vessel-CS: {LCCA_JN_VCS} -> mesh: {lcca_jn_mesh}")
    print(f"Trunk-top vessel-CS: {TRUNK_TOP_VCS} -> mesh: {trunk_top_mesh}")

    # ===== Cross-section sweep through RVA-jn region =====
    print("\n" + "="*70)
    print("Cross-section sweep at vessel-CS z = 410..420 mm (around RVA-jn)")
    print("="*70)
    print(f"{'z_vcs':>8} {'#regions':>10} {'centroids (vcs)':>40} {'eq_radius_mm':>14}")
    print("-" * 90)
    for z in [408, 410, 412, 414, 415, 416, 417, 418, 420, 422, 425]:
        regions = cross_section_at_z(mesh, z)
        # Filter to regions near RVA-jn (within 30 mm)
        near = [r for r in regions
                if np.linalg.norm(r["centroid_vcs"][:2] - RVA_JN_VCS[:2]) < 35]
        cents_str = " | ".join(
            f"({r['centroid_vcs'][0]:+.1f}, {r['centroid_vcs'][1]:+.1f})"
            f"={r['eq_radius_mm']:.1f}"
            for r in near
        )
        print(f"{z:>8} {len(near):>10} {cents_str}")

    # ===== Local cross-section at RVA[5] and RCCA[5] (inside daughters) =====
    rva = cls.get("- RVA")
    rcca = cls.get("- RCCA")
    if rva is not None:
        print("\n" + "="*70)
        print("RVA centerline first 12 points (raw mesh coords + computed eq_radius)")
        print("="*70)
        for i in range(min(12, len(rva))):
            p_raw = rva[i]
            p_vcs = mesh_to_vcs(p_raw)
            # Local lumen radius at this z
            regions = cross_section_at_z(mesh, p_vcs[2])
            # Pick region closest to centerline
            if regions:
                closest = min(regions,
                              key=lambda r: np.linalg.norm(
                                  r["centroid_vcs"] - p_vcs))
                print(
                    f"  RVA[{i:2d}] vcs=({p_vcs[0]:+.2f},{p_vcs[1]:+.2f},"
                    f"{p_vcs[2]:+.2f}) -> nearest_region centroid="
                    f"({closest['centroid_vcs'][0]:+.2f},"
                    f"{closest['centroid_vcs'][1]:+.2f},"
                    f"{closest['centroid_vcs'][2]:+.2f}) "
                    f"eq_r={closest['eq_radius_mm']:.2f} mm"
                )
            else:
                print(f"  RVA[{i:2d}] vcs={p_vcs} -- no slice region")
    if rcca is not None:
        print("\n" + "="*70)
        print("RCCA centerline first 8 points")
        print("="*70)
        for i in range(min(8, len(rcca))):
            p_raw = rcca[i]
            p_vcs = mesh_to_vcs(p_raw)
            regions = cross_section_at_z(mesh, p_vcs[2])
            if regions:
                closest = min(regions,
                              key=lambda r: np.linalg.norm(
                                  r["centroid_vcs"] - p_vcs))
                print(
                    f"  RCCA[{i:2d}] vcs=({p_vcs[0]:+.2f},{p_vcs[1]:+.2f},"
                    f"{p_vcs[2]:+.2f}) eq_r={closest['eq_radius_mm']:.2f} mm"
                )

    # ===== Render annotated view =====
    if os.environ.get("SKIP_RENDER"):
        print("\n(rendering skipped — set SKIP_RENDER=0 to enable)")
        return
    print("\nRendering annotated mesh view to /work/bif2_jn_mesh.png ...")
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    plotter.add_mesh(mesh, color="lightgray", opacity=0.5, show_edges=False)
    # Mark junctions
    for name, vcs, color in [
        ("trunk-top", TRUNK_TOP_VCS, "blue"),
        ("LCCA-jn", LCCA_JN_VCS, "red"),
        ("RVA/RCCA-jn", RVA_JN_VCS, "orange"),
    ]:
        m = vcs_to_mesh(vcs)
        plotter.add_mesh(pv.Sphere(radius=2.0, center=m), color=color)
        plotter.add_point_labels(
            np.atleast_2d(m), [name], point_size=0, font_size=18,
            text_color=color, shape_color=None, always_visible=True,
        )
    # Plot centerlines
    if rva is not None:
        plotter.add_mesh(pv.PolyData(rva), color="purple",
                         render_points_as_spheres=True, point_size=4,
                         label="RVA centerline")
    if rcca is not None:
        plotter.add_mesh(pv.PolyData(rcca), color="green",
                         render_points_as_spheres=True, point_size=4,
                         label="RCCA centerline")
    bridge11 = cls.get("(11)")
    if bridge11 is not None:
        plotter.add_mesh(pv.PolyData(bridge11), color="red",
                         render_points_as_spheres=True, point_size=4,
                         label="bridge (11)")
    # Zoom to junction region (mesh coords)
    rva_jn_m = vcs_to_mesh(RVA_JN_VCS)
    plotter.camera.focal_point = rva_jn_m
    plotter.camera.position = rva_jn_m + np.array([60, 60, 60])
    plotter.camera.up = (0, 0, 1)
    plotter.add_legend()
    plotter.screenshot("bif2_jn_mesh.png")
    print("Saved bif2_jn_mesh.png")


if __name__ == "__main__":
    main()
