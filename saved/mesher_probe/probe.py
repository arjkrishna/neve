#!/usr/bin/env python3
"""Push known tubes and real anatomies through the CURRENT mesher and an
SDF-based alternative, and measure what each does to the lumen.

current : mark spheres (binary) -> gaussian(1) x2 -> marching cubes @ (min+max)/2 -> decimate(0.99)
sdf     : signed distance to the union of centerline spheres -> marching cubes @ 0 -> decimate

Runs in the container:  python3 /opt/eve_training/saved/mesher_probe/probe.py
"""
import glob
import os
import sys
import time
import types

import numpy as np
import pyvista as pv
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve.intervention.vesseltree.util.meshing import get_surface_mesh
from eve.intervention.vesseltree.util.voxelcube import create_empty_voxel_cube_from_branches

SPACING = [0.6, 0.6, 0.9]
OUT = "/opt/eve_training/saved/mesher_probe"


def tube(r, axis, L=60.0, step=0.5):
    n = int(L / step) + 1
    p = np.zeros((n, 3))
    p[:, axis] = np.linspace(0, L, n)
    p += 10.0
    return types.SimpleNamespace(coordinates=p, radii=np.full(n, r), low=p.min(0) - r, high=p.max(0) + r)


def cube_for(branches, spacing):
    vc = create_empty_voxel_cube_from_branches(branches, spacing)
    for _ in range(5):
        vc.add_padding_layer_all_sides()
    return vc


def mesh_current(branches, decimate=0.99, volume_preserve=False, spacing=SPACING, passes=2):
    vc = cube_for(branches, spacing)
    for b in branches:
        vc.mark_centerline_in_array(b.coordinates, b.radii, 1, 0)
    for _ in range(passes):
        vc.gaussian_smooth(1)
    m = get_surface_mesh(vc, "descent", level=0.5)   # explicit: a lone thin tube never reaches 1.0
    if decimate:
        m = m.decimate(decimate, volume_preservation=volume_preserve)
    return m


def mesh_sdf(branches, decimate=0.99, spacing=SPACING, band=3, sample_mm=0.25, k=12):
    """Signed distance to the union of spheres, evaluated only in a band around
    the marked tube, iso-surfaced at zero. No smoothing pass at all."""
    vc = cube_for(branches, spacing)
    for b in branches:
        vc.mark_centerline_in_array(b.coordinates, b.radii, 1, 0)
    inside = vc.value_array > 0.5
    region = ndimage.binary_dilation(inside, iterations=band)
    P, R = [], []
    for b in branches:
        c, r = np.asarray(b.coordinates, float), np.asarray(b.radii, float)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        q = np.arange(0, d[-1] + 1e-9, sample_mm)
        P.append(np.stack([np.interp(q, d, c[:, i]) for i in range(3)], 1))
        R.append(np.interp(q, d, r))
    P, R = np.concatenate(P), np.concatenate(R)
    idx = np.argwhere(region)
    coords = idx * vc.spacing + vc.world_offset
    dist, nn = cKDTree(P).query(coords, k=min(k, len(P)))
    f = (R[nn] - dist).max(axis=1)            # positive inside the union of spheres
    field = np.full(vc.value_array.shape, -10.0, np.float32)
    field[idx[:, 0], idx[:, 1], idx[:, 2]] = f
    verts, faces, _, _ = measure.marching_cubes(field, 0.0, spacing=vc.spacing, gradient_direction="descent")
    faces = np.c_[np.full(len(faces), 3), faces]
    m = pv.PolyData(verts, faces)
    m.translate(vc.world_offset, inplace=True)
    if decimate:
        m = m.decimate(decimate, volume_preservation=True)
    return m


def ncomp(m):
    return int(m.connectivity().cell_data["RegionId"].max()) + 1 if m.n_cells else 0


def radial_stats(m, axis, lo=15.0, hi=45.0):
    """Inscribed radius of the meshed tube along its axis (middle section):
    distance from axis points to the surface, i.e. what the device sees."""
    if m.n_cells == 0:
        return None
    t = np.arange(lo, hi, 0.5)
    ax = np.zeros((len(t), 3)) + 10.0
    ax[:, axis] = t + 10.0
    lum, ins = lumen_along(m, ax)
    if not ins.any():
        return None
    rr = lum[ins]
    return dict(mean=float(rr.mean()), min=float(rr.min()), max=float(rr.max()), tris=int(m.n_cells), comps=ncomp(m))


def to_budget(m, n_tris, volume_preserve=True):
    """Decimate to a fixed triangle count, comparable across pipelines."""
    if m.n_cells <= n_tris:
        return m
    return m.decimate(1.0 - n_tris / m.n_cells, volume_preservation=volume_preserve)


def weighted_decimate(m, route, near_mm=8.0, near_keep=0.10, far_keep=0.005):
    """Spend triangles where the device goes: cells within near_mm of the route
    keep near_keep of their triangles, the rest keep far_keep."""
    cen = np.asarray(m.cell_centers().points)
    d, _ = cKDTree(np.asarray(route, float)).query(cen)
    near = np.nonzero(d < near_mm)[0]
    far = np.nonzero(d >= near_mm)[0]
    a = m.extract_cells(near).extract_surface().triangulate()
    b = m.extract_cells(far).extract_surface().triangulate()
    a = a.decimate(1.0 - near_keep, volume_preservation=True)
    b = b.decimate(1.0 - far_keep, volume_preservation=True)
    return (a + b).clean()


def lumen_along(m, route):
    """Distance from each centerline point to the surface (= meshed lumen radius
    there), and whether the point is inside at all."""
    cloud = pv.PolyData(np.asarray(route, float))
    d = np.asarray(cloud.compute_implicit_distance(m)["implicit_distance"])
    sel = cloud.select_enclosed_points(m, tolerance=0.0, check_surface=False)
    inside = np.asarray(sel["SelectedPoints"], bool)
    return np.abs(d), inside


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=== A. straight tubes, 60 mm, radius declared vs meshed (middle 30 mm) ===")
    BUD = 250   # triangles per 60 mm tube: the baked anatomies run ~3.7k tris over ~1 m of vessel
    say("%-5s %-4s | %-25s | %-25s | %-25s | %-25s | %-25s | %-25s" % (
        "r", "ax", "current no-dec", "current @250 vol-pres", "current 0.3mm vox no-dec", "sdf no-dec", "sdf @250 vol-pres", "sdf 0.3mm vox @250"))
    for r in (0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0):
        for axis, nm in ((2, "z"), (0, "x")):
            b = [tube(r, axis)]
            cells = []
            for fn in (lambda: mesh_current(b, 0.0),
                       lambda: to_budget(mesh_current(b, 0.0), BUD, True),
                       lambda: mesh_current(b, 0.0, spacing=[0.3, 0.3, 0.45]),
                       lambda: mesh_sdf(b, 0.0),
                       lambda: to_budget(mesh_sdf(b, 0.0), BUD, True),
                       lambda: to_budget(mesh_sdf(b, 0.0, spacing=[0.3, 0.3, 0.3], band=6), BUD, True)):
                try:
                    st = radial_stats(fn(), axis)
                except Exception:
                    st = None
                cells.append("absent" if st is None else
                             "mean %.2f min %.2f n%d" % (st["mean"], st["min"], st["tris"]))
            say("%-5.1f %-4s | %-25s | %-25s | %-25s | %-25s | %-25s | %-25s" % (r, nm, *cells))

    say()
    say("=== B. real anatomies: declared radius vs meshed lumen along the RCCA route ===")
    from eve_bench.dualdevicenav import load_branches
    cases = ["/opt/eve_training/topbrain_data/anatomies/topcow_mr_001/Centrelines_comb",
             sorted(glob.glob("/opt/eve_training/carotid_data/anatomies/*/Centrelines_comb"))[0]]
    for cl in cases:
        root = os.path.dirname(cl)
        name = os.path.basename(root)
        br = load_branches(cl)
        rcca = [b for b in br if "RCCA" in str(b.name).upper()][0]
        route, rad = np.asarray(rcca.coordinates, float), np.asarray(rcca.radii, float)
        baked = pv.read(os.path.join(root, "vessel_architecture_collision.obj"))
        t0 = time.time(); cur = mesh_current(br, 0.99); t_cur = time.time() - t0
        t0 = time.time(); nod = mesh_current(br, 0.0); t_nod = time.time() - t0
        t0 = time.time(); sd_full = mesh_sdf(br, 0.0); t_sd = time.time() - t0
        sd = sd_full.decimate(0.99, volume_preservation=True)
        sd_eq = sd_full.decimate(1.0 - baked.n_cells / max(sd_full.n_cells, 1), volume_preservation=True)
        s_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        rl = float(s_arc[-1])
        body = (s_arc > 3.0) & (s_arc < rl - 3.0)      # end caps are not lumen
        cur_w = weighted_decimate(nod, route)
        sd_w = weighted_decimate(sd_full, route)
        t0 = time.time(); sd_fine = mesh_sdf(br, 0.0, spacing=[0.45, 0.45, 0.45], band=4); t_fine = time.time() - t0
        sd_fine_w = weighted_decimate(sd_fine, route)
        ext = []
        for tag in ("a", "b", "u"):
            f = os.path.join(OUT, name[:40] + "_remesh_%s.vtp" % tag)
            if os.path.exists(f):
                ext.append(("vmtk remesh %s" % tag, pv.read(f).extract_surface().triangulate(), None))
        say("--- %s   route %.0f mm, declared radius min %.2f / median %.2f mm" % (name[:44], rl, rad[body].min(), np.median(rad)))
        say("%-30s %8s %6s %6s %10s %8s %8s %12s %8s %6s" % ("mesh", "tris", "comps", "open", "lumen_min", "at_mm", "decl@", "med_deficit", "pts_out", "sec"))
        for lab, m, t in (("baked (.obj on disk)", baked, None),
                          ("current dec.99 (rebuilt)", cur, t_cur),
                          ("current no-dec", nod, t_nod),
                          ("current route-weighted", cur_w, None),
                          ("sdf no-dec", sd_full, t_sd),
                          ("sdf dec.99 vol-pres", sd, None),
                          ("sdf @ baked tri count", sd_eq, None),
                          ("sdf route-weighted", sd_w, None),
                          ("sdf 0.45mm iso no-dec", sd_fine, t_fine),
                          ("sdf 0.45mm route-weighted", sd_fine_w, None)) + tuple(ext):
            lum, ins = lumen_along(m, route)
            ok = ins & body
            if ok.any():
                i = int(np.argmin(np.where(ok, lum, np.inf)))
                deficit = float(np.median(rad[ok] - lum[ok]))
                say("%-30s %8d %6d %6d %10.2f %8.0f %8.2f %12.2f %8d %6s" % (
                    lab, m.n_cells, ncomp(m), m.n_open_edges, lum[i], s_arc[i], rad[i], deficit,
                    int((~ins & body).sum()), "" if t is None else "%.1f" % t))
            else:
                say("%-30s %8d %6d %6d   (route not enclosed)" % (lab, m.n_cells, ncomp(m), m.n_open_edges))
        sd_full.save(os.path.join(OUT, name[:40] + "_sdf_full.vtp"))
        sd_w.save(os.path.join(OUT, name[:40] + "_sdf_weighted.obj"))
        cur_w.save(os.path.join(OUT, name[:40] + "_current_weighted.obj"))
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
