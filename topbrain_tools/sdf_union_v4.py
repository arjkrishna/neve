#!/usr/bin/env python3
"""Real surfaces unioned into the tube field WITHOUT creating merges (v4).

v3's union (sdf_union.py) is

    f = max(f_tube, min(f_real, f_capsule))

with a capsule of 1.8 r + 1 mm round each section's own centerline. The
capsule is what lets a real lumen show its bulb and its eccentric sections --
and it is also the defect. The graft-time clearance gates only ever saw the
DECLARED radii, so wherever another vessel, or another stretch of the same
looping siphon, sat inside that capsule reach, the real surface was free to
grow across the gap the gates had passed. Measured on v3: extra topological
handles in 9 of 49 set-A and 64 of 223 set-B anatomies -- siphon self-contact
at the genu, route or siphon against the RVA (one device-passable, in the
TRAIN anatomy case_w_047_left__topcow_mr_018_L), ECA against host vessels,
ICA against its own ECA past the fork.

OWNERSHIP. v4 keeps the union but only lets a real surface add volume to a
voxel it OWNS: a voxel whose nearest centerline, by the same radius-aware tube
measure the field itself uses (r_j - |x - c_j|), is this section's own vessel
at this stretch of it, by a margin of at least MARGIN_MM over

  * every other branch in the anatomy, and
  * the same branch more than WINDOW_MM of arclength away (self-contact).

Between two vessels the real surfaces can then reach at most to the bisector
less MARGIN_MM / 2 each, leaving an unowned band at least MARGIN_MM wide. At
0.5 mm no grid edge (0.45 mm) crosses it, but a cube's face diagonal (0.64 mm)
or body diagonal (0.78 mm) can, and marching cubes may join the two sides
there; from 1.0 mm no cube spans the band. So the margin makes merges rare,
not impossible: the merge-free guarantee is bake_meshes_v4's topology gate,
which widens the margin or localises what remains. Everywhere else -- the
free lumen, the bulb, the nooks -- the real surface is used exactly as in v3.

A section that shares its source surface with a sibling (the lower's route
slice and its own ECA are one Zenodo lumen) does NOT compete with it: the
lumen's own carina and bulb are real anatomy, not a merge. Making them compete
refused real bulb wall towards the ECA -- the v4 fork carried a median 20%
less wall detail than v3 (area standing > 0.3 mm off the tube), 64% less in
the worst donors; without the competition it matches v3 exactly (24 anatomies,
raw unions compared on the same grid).
"""
import numpy as np
from scipy.spatial import cKDTree

from sdf_mesher import tube_field
from sdf_union import BAND_MM, _capsule_radii

MARGIN_MM = 0.5            # wall left between two real surfaces, >= one grid step
WINDOW_MM = 12.0           # same-branch samples farther than this compete (self-contact)
SAMPLE_MM = 0.25
K_OWN, K_OTHER = 48, 16


class Section:
    """A real-surface section: transformed surface, the kept centerline it
    covers, and which branch (and arclength offset on it) that centerline is."""

    def __init__(self, name, surface, points, radii, taper_start, taper_end, branch, s0, siblings=()):
        self.name, self.surface = name, surface
        self.points, self.radii = np.asarray(points, float), np.asarray(radii, float)
        self.taper_start, self.taper_end = taper_start, taper_end
        self.branch, self.s0 = branch, float(s0)
        # branches cut from the SAME source surface (a lower's route slice and
        # its own ECA are one Zenodo lumen): they do not compete for ownership
        self.siblings = tuple(siblings)


def _dense(points, radii, step=SAMPLE_MM):
    p, r = np.asarray(points, float), np.asarray(radii, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < step:
        return p, r, s
    q = np.arange(0.0, s[-1] + 1e-9, step)
    return (np.stack([np.interp(q, s, p[:, i]) for i in range(3)], 1), np.interp(q, s, r), q)


class Samples:
    """Dense samples of every branch: points, radii, branch id and arclength."""

    def __init__(self, branches):
        P, R, B, S = [], [], [], []
        self.names = []
        for i, b in enumerate(branches):
            p, r, s = _dense(b.coordinates, b.radii)
            P.append(p); R.append(r); B.append(np.full(len(p), i)); S.append(s)
            self.names.append(str(b.name))
        self.p, self.r = np.vstack(P), np.concatenate(R)
        self.b, self.s = np.concatenate(B), np.concatenate(S)

    def index_of(self, name):
        for i, n in enumerate(self.names):
            if name in n:
                return i
        raise KeyError(name)


def _tube(coords, p, r, k):
    k = min(k, len(p))
    tree = cKDTree(p)
    d, nn = tree.query(coords, k=k)
    if d.ndim == 1:
        d, nn = d[:, None], nn[:, None]
    return r[nn] - d, nn


def ownership_margin(coords, sec, samp, return_anchor=False):
    """f_own_local - f_competitor at each voxel (positive = owned). With
    return_anchor, also the owning centerline point (the local sample with the
    largest tube value) and its arclength on the branch."""
    bi = samp.index_of(sec.branch)
    own = samp.b == bi
    op, orad, os_ = samp.p[own], samp.r[own], samp.s[own]
    f_o, nn = _tube(coords, op, orad, K_OWN)
    s_nn = os_[nn]
    best = np.argmax(f_o, axis=1)
    s_star = s_nn[np.arange(len(coords)), best]
    near = np.abs(s_nn - s_star[:, None]) <= WINDOW_MM
    f_loc_all = np.where(near, f_o, -np.inf)
    f_local = f_loc_all.max(axis=1)
    anchor = op[nn[np.arange(len(coords)), np.argmax(f_loc_all, axis=1)]]
    f_far = np.where(~near, f_o, -np.inf).max(axis=1)
    # the competitor per branch, as tube_field takes its max per branch: one
    # pooled k-NN over every other branch lets a near thin vessel crowd a
    # farther fat one (the governing tube) out of the k slots, and over-grants
    # ownership -- measured on v4's first bake against same-lumen siblings
    f_other = np.full(len(coords), -np.inf)
    for j in np.unique(samp.b[~own]):
        if any(sb in samp.names[j] for sb in getattr(sec, "siblings", ())):
            continue
        m = samp.b == j
        f_x, _ = _tube(coords, samp.p[m], samp.r[m], K_OTHER)
        f_other = np.maximum(f_other, f_x.max(axis=1))
    # the owned stretch must also be THIS section's stretch of the branch
    inside_sec = (s_star >= sec.s0 - WINDOW_MM) & (s_star <= sec.s0 + _length(sec.points) + WINDOW_MM)
    margin = np.where(inside_sec, f_local - np.maximum(f_far, f_other), -np.inf)
    return (margin, anchor, s_star) if return_anchor else margin


def _length(p):
    return float(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum())


def add_real_sections_owned(field, sections, samp, spacing, margin_mm=MARGIN_MM, verbose=False, suppress=None):
    """Union each section's real surface into `field` where it owns the voxel."""
    import pyvista as pv
    from types import SimpleNamespace
    stats = {}
    for sec in sections:
        cap_r = _capsule_radii(sec.points, sec.radii, sec.taper_start, sec.taper_end)
        cap = tube_field([SimpleNamespace(coordinates=sec.points, radii=cap_r)],
                         spacing=spacing, band=int(np.ceil(BAND_MM / spacing)) + 1)
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
        f_real = (-d).astype(np.float32)
        f_sec = np.minimum(f_real, cap.values[idx[:, 0], idx[:, 1], idx[:, 2]])
        cur = field.values[gidx[:, 0], gidx[:, 1], gidx[:, 2]]
        would = f_sec > cur
        owned = np.ones(len(idx), bool)
        if would.any():
            m, _, s_st = ownership_margin(coords[would], sec, samp, return_anchor=True)
            w = np.nonzero(would)[0]
            owned[w] = m >= margin_mm
            # stretches of this section whose real surface carries a handle
            # (found by the baker's topology localisation): tube only there
            for lo, hi in (suppress or {}).get(sec.name, []):
                owned[w[(s_st >= lo) & (s_st <= hi)]] = False
        use = would & owned
        vals = cur.copy()
        vals[use] = f_sec[use]
        field.values[gidx[:, 0], gidx[:, 1], gidx[:, 2]] = vals
        field.region[gidx[:, 0], gidx[:, 1], gidx[:, 2]] = True
        stats[sec.name] = {"voxels": int(len(idx)), "voxels_widened": int(use.sum()),
                           "voxels_refused": int((would & ~owned).sum()),
                           "surface_tris": int(sec.surface.n_cells)}
        if verbose:
            print("   real %-9s %7d vox, %6d widened, %5d refused (not owned)"
                  % (sec.name, len(idx), use.sum(), (would & ~owned).sum()), flush=True)
    return stats


def genus(mesh):
    """Genus of each connected component of a triangle mesh, from its Euler
    characteristic and boundary loops: chi = V - E + F = 2 - 2g - b."""
    m = mesh.clean().triangulate()
    conn = m.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"])
    out = []
    for k in range(int(rid.max()) + 1 if len(rid) else 0):
        c = conn.extract_cells(np.nonzero(rid == k)[0]).extract_surface().clean()
        V, F = c.n_points, c.n_cells
        E = c.extract_all_edges().n_cells
        b = 0
        if c.n_open_edges:
            be = c.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                         manifold_edges=False, non_manifold_edges=False)
            b = int(be.connectivity().point_data["RegionId"].max()) + 1 if be.n_points else 0
        chi = V - E + F
        out.append({"cells": int(F), "genus": (2 - chi - b) / 2.0, "boundary_loops": int(b)})
    return out
