# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\jfix\junction6.py (= /sp/v6/wall/jfix/junction6.py in the build containers)
"""v6_wall (e) junction fix, imported by ctseg6.py / vmrseg6.py in the subclavian request mode (V6_JUNC_BRIDGE=1, default).

Bug (v6_va, 2026-10-03 21:19): <side>junction_stageA.vtp = subclavian label + the VA label's first 25 mm, largest
component. ct_seg4 cuts the VA label planar at its first station (voxels behind P[0] -> vid -2) and caps the
subclavian label at r_e + CAP from the subclavian stations, so the ostium crotch between the two labels belongs to
neither: in 23/54 sides the labels do not touch (gap 0.5-4.4 mm, wall/subclavian/jdiag.json) and the junction surface
was the subclavian alone; where they touch it is often by 2-16 voxels (a waist at the ostium).

Fix (CT edge, not a synthetic tube): a bridge region = voxels within rb of the axis polyline [Q, P_va(s 0..BRIDGE_S)]
(Q = the subclavian station-line point nearest the VA's first station; rb = the VA's edge radius over its first
BRIDGE_S mm + PAD), and inside it the lumen by the two vessels' own thresholds: T_lo = mean of the median local
half-max T_loc of the VA's first BRIDGE_S mm and of the subclavian's stations within 3 mm of Q; T_hi = ct_seg4's
bright-neighbour level max(1.3 Ic, Ic + 200) of the brighter core; bone (>= 700 HU) excluded. Junction = the
component of (subclavian | VA first 25 mm | bridge) holding the subclavian, closed by one voxel inside the bridge
region, holes filled. Only if the VA part is still not connected: the bridge region is filled to the VA's own edge
radius regardless of HU (flagged fallback_tube).
Measures (info): connected before / after, bridge voxels, bottleneck radius = the largest ball radius that can travel
inside the mask from the subclavian (Q) to the VA at s = 10 mm (maximin over the EDT), before and after, and its ratio
to the VA's inscribed radius at s 10.
"""
import numpy as np
from scipy import ndimage

BRIDGE_S = 6.0
PAD = 1.0


def _seg_dist(Pts, A):
    best = np.full(len(Pts), np.inf)
    for a, b in zip(A[:-1], A[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((Pts - a) @ ab) / max(L2, 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(Pts - (a + t[:, None] * ab), axis=1))
    return best


def _bottleneck(mask, sp_kji, qa, qb, rmax):
    """largest r such that (EDT >= r) connects voxel qa to voxel qb (kji ints); 0 if not connected at all."""
    lab, _ = ndimage.label(mask)
    if lab[tuple(qa)] == 0 or lab[tuple(qa)] != lab[tuple(qb)]:
        return 0.0
    edt = ndimage.distance_transform_edt(mask, sampling=sp_kji)
    lo, hi = 0.0, float(min(edt[tuple(qa)], edt[tuple(qb)], rmax))
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        lab, _ = ndimage.label(edt >= mid)
        if lab[tuple(qa)] > 0 and lab[tuple(qa)] == lab[tuple(qb)]:
            lo = mid
        else:
            hi = mid
    return lo


def junction_mask(msub, mva, ctsub, corg, sp, dva, dsb, ndil_close=1):
    """msub, mva: bool kji masks (subclavian label, VA label first 25 mm); ctsub: CT crop (kji, HU); corg/sp: xyz;
    dva / dsb: the VA / subclavian station dicts (s, P, R, T_loc, Ic, r_e). Returns (mask, info)."""
    sp = np.asarray(sp, float)
    corg = np.asarray(corg, float)
    shp = msub.shape
    sp_kji = (sp[2], sp[1], sp[0])
    info = {"bridge_s_mm": BRIDGE_S, "pad_mm": PAD}
    s_va = np.asarray(dva["s"], float)
    P_va = np.asarray(dva["P"], float)
    P_sb = np.asarray(dsb["P"], float)
    s_sb = np.asarray(dsb["s"], float)
    jv = int(max(1, np.searchsorted(s_va, BRIDGE_S)))
    ksb = int(np.argmin(np.linalg.norm(P_sb - P_va[0], axis=1)))
    Q = P_sb[ksb]
    A = np.vstack([Q, P_va[:jv + 1]])
    re0 = float(np.median(np.asarray(dva["r_e"], float)[:jv + 1]))
    R0 = float(np.median(np.asarray(dva["R"], float)[:jv + 1]))
    rb = max(re0, R0) + PAD
    near = np.abs(s_sb - s_sb[ksb]) <= 3.0
    Tva = float(np.median(np.asarray(dva["T_loc"], float)[:jv + 1]))
    Tsb = float(np.median(np.asarray(dsb["T_loc"], float)[near]))
    Ic = max(float(np.median(np.asarray(dva["Ic"], float)[:jv + 1])), float(np.median(np.asarray(dsb["Ic"], float)[near])))
    Tlo = 0.5 * (Tva + Tsb)
    Thi = min(max(1.3 * Ic, Ic + 200.0), 700.0)
    info.update({"Q_xyz": Q.tolist(), "va_P0_xyz": P_va[0].tolist(), "gap_Q_P0_mm": float(np.linalg.norm(Q - P_va[0])),
                 "rb_mm": rb, "va_edge_r_mm": re0, "T_lo": Tlo, "T_hi": Thi, "T_va": Tva, "T_sub": Tsb, "Ic": Ic})
    # region: box around A
    lo_x = A.min(0) - rb - 1.0
    hi_x = A.max(0) + rb + 1.0
    i0 = np.maximum(np.floor((lo_x - corg) / sp).astype(int), 0)
    i1 = np.minimum(np.ceil((hi_x - corg) / sp).astype(int), np.array(shp[::-1]) - 1)
    kk, jj, ii = np.meshgrid(np.arange(i0[2], i1[2] + 1), np.arange(i0[1], i1[1] + 1), np.arange(i0[0], i1[0] + 1),
                             indexing="ij")
    Xb = corg + np.stack([ii, jj, kk], -1).reshape(-1, 3) * sp
    dist = _seg_dist(Xb, A).reshape(kk.shape)
    omega = np.zeros(shp, bool)
    dbox = np.full(shp, np.inf, np.float32)
    sl = (slice(i0[2], i1[2] + 1), slice(i0[1], i1[1] + 1), slice(i0[0], i1[0] + 1))
    omega[sl] = dist <= rb
    dbox[sl] = dist
    hu = ctsub
    bright = (hu >= Tlo) & (hu <= Thi)
    bridge = omega & bright
    base = msub | mva

    def comp_with_sub(m):
        lab, n = ndimage.label(m)
        if n == 0:
            return m, 0, 0.0
        cnt = ndimage.sum(msub, lab, range(1, n + 1))
        kbest = int(np.argmax(cnt)) + 1
        keep = lab == kbest
        va_frac = float((keep & mva).sum()) / max(int(mva.sum()), 1)
        return keep, n, va_frac

    keep0, n0, f0 = comp_with_sub(base)
    info["before"] = {"components": int(n0), "va_fraction_connected": f0, "touch6_voxels":
                      int((ndimage.binary_dilation(msub) & mva).sum())}
    U = base | bridge
    keep1, n1, f1 = comp_with_sub(U)
    fallback = False
    if f1 < 0.5:
        U = U | (omega & (dbox <= max(re0, R0)))
        keep1, n1, f1 = comp_with_sub(U)
        fallback = True
    m = keep1
    if ndil_close > 0:
        cl = ndimage.binary_closing(m, iterations=ndil_close)
        m = m | (cl & omega)
    m = ndimage.binary_fill_holes(m)
    m, n2, f2 = comp_with_sub(m)
    info["after"] = {"components_before_pick": int(n1), "va_fraction_connected": f2,
                     "bridge_voxels_added": int((m & ~base).sum()), "fallback_tube": fallback}
    # bottleneck sub (Q) -> VA s 10
    j10 = int(min(np.searchsorted(s_va, 10.0), len(s_va) - 1))

    def kji(p):
        q = np.round((p - corg) / sp).astype(int)
        q = np.clip(q, 0, np.array(shp[::-1]) - 1)
        return np.array([q[2], q[1], q[0]])
    qa, qb = kji(Q), kji(P_va[j10])
    # local crop for speed
    pts = np.vstack([A, P_va[:j10 + 1]])
    lo_c = np.maximum(np.floor((pts.min(0) - 12.0 - corg) / sp).astype(int), 0)
    hi_c = np.minimum(np.ceil((pts.max(0) + 12.0 - corg) / sp).astype(int), np.array(shp[::-1]) - 1)
    slc = (slice(lo_c[2], hi_c[2] + 1), slice(lo_c[1], hi_c[1] + 1), slice(lo_c[0], hi_c[0] + 1))
    off = np.array([lo_c[2], lo_c[1], lo_c[0]])
    rmax = 10.0
    for tag, mm in (("before", base), ("after", m)):
        mc = mm[slc]
        r = _bottleneck(mc, sp_kji, qa - off, qb - off, rmax)
        edt = ndimage.distance_transform_edt(mc, sampling=sp_kji)
        rv = float(edt[tuple(qb - off)])
        info[tag]["bottleneck_r_mm"] = round(r, 3)
        info[tag]["va_inscribed_r_s10_mm"] = round(rv, 3)
        info[tag]["bottleneck_ratio"] = round(r / rv, 3) if rv > 0 else None
    return m, info


def clean_sub(msub):
    """v6 (e) handle cleanup of the SUBCLAVIAN label only (V6_SUB_CLEAN=1): the request runs' subclavian labels carry
    1-voxel bridges / spurs (partial-volume attachments to neighbours) that make handles -- genus 5-57 per side
    (jfix/sub_genus.py: VMR 0248 R 57, L 26; ia172_071 L 21) -- which the bake's handle rule would tube. A 1-voxel
    opening then a 1-voxel closing (6-connected cross) removes them (genus 0 on the 6 sides tested) while changing
    0.7-3.3 % of the label's voxels; the component holding most of the label is kept, holes filled. The VA labels are
    NOT touched (ct_seg4's own open-bright / keep-connected rules already apply to them)."""
    cross = ndimage.generate_binary_structure(3, 1)
    n0 = int(msub.sum())
    m = ndimage.binary_opening(msub, structure=cross)
    m = ndimage.binary_closing(m, structure=cross) & ndimage.binary_dilation(msub, structure=cross)
    lab, n = ndimage.label(m)
    if n > 1:
        cnt = ndimage.sum(msub, lab, range(1, n + 1))
        m = lab == (int(np.argmax(cnt)) + 1)
    m = ndimage.binary_fill_holes(m)
    return m, {"voxels_before": n0, "voxels_after": int(m.sum()), "changed_fraction": round(float((m ^ msub).sum()) / max(n0, 1), 4),
               "components_after_open_close": int(n)}


def genus_of(verts, faces):
    """closed-surface genus by Euler characteristic, per the largest component (pyvista); plus component count."""
    import pyvista as pv
    mesh = pv.PolyData(np.asarray(verts, float), np.c_[np.full(len(faces), 3), faces].ravel()).clean()
    conn = mesh.connectivity(extraction_mode="all")
    ncomp = int(len(np.unique(conn["RegionId"]))) if "RegionId" in conn.array_names else 1
    big = mesh.connectivity(extraction_mode="largest").extract_surface().clean()
    V = big.n_points
    F = big.n_cells
    E = big.extract_all_edges().n_cells
    chi = V - E + F
    return {"components": ncomp, "genus_largest": int(round((2 - chi) / 2)), "chi_largest": int(chi)}
