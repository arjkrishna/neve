#!/usr/bin/env python3
"""Real segmented surfaces, unioned into the tube field (v3, change 8).

The v2 mesh is a union of circular tubes swept along centerlines. The host is
procedural, so a tube is the best it can be; but the TopBrain siphons and the
Zenodo carotids come with the patient's real lumen surface, and a tube throws
that away -- the bulb, the eccentric sections, the plaque-shaped stenosis, the
wall texture the shipped test anatomy has. Because the v2 field is a signed
distance, adding a real surface is one operation:

    f = max(f_tube, min(f_real, f_capsule))

`f_real`    signed distance to the patient's surface, carried through the SAME
            rotation / origin / anchor / mirror its centerline got in the graft
            (recorded by the grafters in graft_xform.json / provenance.json).
`f_capsule` a generous tube round the KEPT centerline of that section, 1.8 x
            the declared radius (+1 mm), tapered down to the tube radius over
            TAPER_MM at every seam. It clips away whatever part of the source
            surface lies outside the section the graft actually kept -- the
            trimmed inlet, the label below the anchor trim, the far side of a
            cut face -- and makes the handover to the tube continuous.
`max(...)`  union with the floored tubes, so the v2 navigability guarantee
            still holds: wherever the real surface pinches under the floor
            the tube wins; everywhere it is wider, the real shape wins.

Surfaces: TopBrain label surfaces (stage A .vtp, closed but for a small cut at
the mask edge -- holes are filled), Zenodo lumen STLs (closed). Largest
connected component only; normals re-oriented outward after the transform
(a mirror flips winding, and the distance sign follows the normals).
"""
import os

import numpy as np

from sdf_mesher import Field, tube_field

CAPSULE_FACTOR = 1.8     # real lumen half-width can exceed MISR this much at a bulb
CAPSULE_PLUS_MM = 1.0
TAPER_MM = 8.0           # capsule shrinks to the tube radius this far before a seam
BAND_MM = 2.0            # evaluate the real SDF this far around the capsule surface


class Section:
    """One real-surface section: transformed surface + the kept centerline it covers."""

    def __init__(self, name, surface, points, radii, taper_start, taper_end):
        self.name, self.surface = name, surface
        self.points, self.radii = np.asarray(points, float), np.asarray(radii, float)
        self.taper_start, self.taper_end = taper_start, taper_end


def load_surface(path, mirror, R, origin, anchor):
    """Read, mirror, transform, clean, orient. Returns a closed pyvista mesh."""
    import pyvista as pv
    m = pv.read(path)
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    m = m.triangulate().clean()
    m = m.connectivity("largest").extract_surface().triangulate()
    pts = np.asarray(m.points, float) * np.asarray(mirror, float)
    pts = (pts - np.asarray(origin, float)) @ np.asarray(R, float).T + np.asarray(anchor, float)
    m = pv.PolyData(pts, m.faces)
    if m.n_open_edges:
        m = m.fill_holes(1000.0).clean().triangulate()
    # outward normals decide the sign of the implicit distance; a mirror has
    # reversed the winding, so re-derive them from the closed surface
    m = m.compute_normals(auto_orient_normals=True, consistent_normals=True,
                          point_normals=True, cell_normals=True, inplace=False)
    return m


def _capsule_radii(points, radii, taper_start, taper_end):
    """1.8 r + 1 mm, tapered linearly to r over TAPER_MM at the seam ends."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    big = CAPSULE_FACTOR * radii + CAPSULE_PLUS_MM
    w = np.ones(len(s))
    if taper_start:
        w = np.minimum(w, np.clip(s / TAPER_MM, 0.0, 1.0))
    if taper_end:
        w = np.minimum(w, np.clip((s[-1] - s) / TAPER_MM, 0.0, 1.0))
    return radii + (big - radii) * w


def add_real_sections(field, sections, spacing, verbose=False):
    """Union each section's real surface into `field` (in place). Returns stats."""
    import pyvista as pv
    from types import SimpleNamespace
    stats = {}
    for sec in sections:
        cap_r = _capsule_radii(sec.points, sec.radii, sec.taper_start, sec.taper_end)
        cap = tube_field([SimpleNamespace(coordinates=sec.points, radii=cap_r)],
                         spacing=spacing, band=int(np.ceil(BAND_MM / spacing)) + 1)
        # candidate voxels: inside the capsule or within BAND_MM of its wall,
        # and only where the two grids overlap (they share spacing and origin
        # alignment, but the capsule grid is smaller)
        off = np.round((cap.offset - field.offset) / spacing).astype(int)
        idx = np.argwhere(cap.region & (cap.values > -BAND_MM))
        gidx = idx + off
        ok = np.all((gidx >= 0) & (gidx < np.array(field.values.shape)), axis=1)
        idx, gidx = idx[ok], gidx[ok]
        if not len(idx):
            stats[sec.name] = {"voxels": 0}
            continue
        coords = gidx * spacing + field.offset
        d = np.asarray(pv.PolyData(coords).compute_implicit_distance(sec.surface)["implicit_distance"])
        f_real = (-d).astype(np.float32)                     # positive inside
        f_sec = np.minimum(f_real, cap.values[idx[:, 0], idx[:, 1], idx[:, 2]])
        cur = field.values[gidx[:, 0], gidx[:, 1], gidx[:, 2]]
        gained = f_sec > cur
        field.values[gidx[:, 0], gidx[:, 1], gidx[:, 2]] = np.maximum(cur, f_sec)
        field.region[gidx[:, 0], gidx[:, 1], gidx[:, 2]] = True
        stats[sec.name] = {"voxels": int(len(idx)), "voxels_widened": int(gained.sum()),
                           "surface_tris": int(sec.surface.n_cells)}
        if verbose:
            print("   real section %-8s %7d voxels, %6d widened by the surface" % (sec.name, len(idx), gained.sum()), flush=True)
    return stats
