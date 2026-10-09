#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\va_metrics.py (= /sp/v5_va/va_metrics.py in the build containers)
"""va_metrics.py -- the shared vertebral-artery (VA) metric suite for vetting v5 sources.

Every candidate VA source (the host tube, a TopBrain MR label, a VMR lofted wall,
an SPL atlas model, an Arterial / nnU-Net mask on a TotalSegmentator CTA, ...)
is measured by the SAME functions here, so it can be held against the
references v4 already ships (results/reference_metrics.json, notes/ref.md).

Runs in eve-training-fixed (numpy / scipy / vtk / pyvista / skimage). The one
function that needs the v4 SDF code, union_detail(), imports sdf_mesher and
sdf_union_v4 from topbrain_tools (read-only), so it is the v4 protocol itself.

INPUT (measure_va)
  points      (N,3) mm, ordered ORIGIN (subclavian) -> DISTAL (basilar / tip)
  radii       (N,)  mm, the source's DECLARED radius (VMTK maximal inscribed
              sphere, EDT, lofted contour, host tube). None -> fitted from the
              surface (radii_from_surface, mode "misr" = VMTK's definition)
  surface     optional pyvista PolyData lumen wall in the same frame
  voxel_mm    optional (3,) imaging voxel size of the source (sets which wall
              scales are resolvable)
  source_type "tube" | "lofted" | "label-surface" | "mask-MC" | ... (provenance)
  landmarks   optional {"C6": s, "C2": s, "C1": s, "dura": s}: arclength (mm
              from the origin) where the VA enters the C6 foramen, leaves C2,
              passes C1, pierces the dura. Gives V1..V4 segments.
  starts_at_origin  True if points[0] is the VA's subclavian origin
  parent_dir  optional unit vector: the subclavian's distal continuation at the
              VA origin (take-off angle)
  up          optional unit vector pointing superior

Every metric is defined in METRIC_DEFINITIONS below, and that dict is written
into every result so a JSON reader never has to come back here.

Self test:  python3 va_metrics.py --selftest
"""
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

VERSION = "va_metrics 1.0 (2026-10-01)"
STEP_MM = 0.25                 # centerline resampling (= sdf_union_v4.SAMPLE_MM)
SIGMA_COMMON_MM = 3.0          # curve smoothing comparable across ALL sources (2 x the 1.5 mm CT voxel)
SIGMA_FINE_MM = 1.5            # finer curve smoothing, valid for sources with voxel <= 0.75 mm
SIGMA_TANGENT_MM = 1.0         # tangent used to tell wall beside a sample from an end cap
SIGMA_RADIUS_MM = 1.0          # radius profile smoothing before calibre events
END_EXCL_MM = 2.0              # excluded at both ends from curvature / wall statistics (seed jogs, caps)
CAL_END_EXCL_MM = 3.0          # excluded at both ends from calibre statistics (MISR shrinks at the seeds)
KAPPA_TORSION_MIN = 0.02       # 1/mm: torsion only where the curve actually bends (R < 50 mm)
BIN_MM = 40.0                  # arclength bins from the origin / from the distal end
ABS_THR_MM = (0.3, 0.5)        # v4 right_route_detail thresholds
REL_THR = (0.15, 0.25)         # radius-relative thresholds (x local radius)
CAPSULE_FACTOR, CAPSULE_PLUS_MM = 1.8, 1.0   # sdf_union: real surface beyond this never enters the union
OCTAVES_MM = (0.375, 0.75, 1.5, 3.0)         # band-pass roughness scales (sigma of the lower edge)
LOWPASS_COMMON_MM = (0.75, 1.5)              # wall detail with scales below sigma removed
FIT_BIN_MM = 1.0               # fitted plain tube: area-weighted median distance per 1 mm of centerline
FIT_SMOOTH_MM = 5.0            # ... then smoothed along the centerline at this sigma
TOPBRAIN_TOOLS = os.environ.get("TOPBRAIN_TOOLS", "/opt/eve_training/topbrain_tools")

METRIC_DEFINITIONS = {
    "frame": "All lengths mm. Centerline resampled at %.2f mm by arclength; s = arclength from points[0]." % STEP_MM,
    "length_mm": "Arclength of the centerline, origin to distal end.",
    "chord_mm": "Straight distance first point -> last point.",
    "DM": "Distance metric (tortuosity) = length / chord (Bullitt 2003). 1 = straight.",
    "SOAM_deg_per_cm": "Sum-of-angles metric: total turning of the smoothed tangent / length, deg per 10 mm.",
    "ICM": "Inflection-count metric = DM x (inflections + 1); an inflection is a flip of the principal normal "
           "(N_i . N_i+1 < 0) with the curve bending (curvature > %.2f /mm) within 5 mm on BOTH sides, flips "
           "< 2 mm apart merged." % KAPPA_TORSION_MIN,
    "curvature": "kappa = |c' x c''| / |c'|^3 on the centerline smoothed by a Gaussian of sigma (mm) along arclength "
                 "(ends extended linearly by 4 sigma first, so no end artefact). p50/p90/max over the core "
                 "(%.0f mm off each end, where VMTK seed jogs sit). 1/kappa = radius of curvature." % END_EXCL_MM,
    "sigma": "shape@3.0 (sigma %.1f mm) is comparable across every source incl. 1.5 mm CT; shape@1.5 (sigma %.1f mm) "
             "only between sources with voxel <= 0.75 mm." % (SIGMA_COMMON_MM, SIGMA_FINE_MM),
    "torsion": "|tau| = |(c' x c'') . c'''| / |c' x c''|^2, same smoothing, only where kappa > %.2f /mm; "
               "torsion_frac = share of the core where that holds. |tau| because mirroring flips its sign."
               % KAPPA_TORSION_MIN,
    "turning": "Total turning = integral of kappa ds (deg). turn_max_W = most turning inside any W mm of arclength. "
               "reversal_max_30mm = largest angle between two tangents <= 30 mm apart (180 = a hairpin).",
    "loops": "n_loops_360_60mm: greedy count of non-overlapping 60 mm windows turning >= 360 deg (a full loop, e.g. "
             "V3 around C1 + the V4 bend); n_uturns_150_30mm: non-overlapping 30 mm windows whose tangents reverse "
             ">= 150 deg. Both on sigma %.1f mm." % SIGMA_COMMON_MM,
    "segments": "V1 origin->C6 entry, V2 C6->C2, V3 C2->dura, V4 dura->end, when landmarks are given. Otherwise bins of "
                "%.0f mm from the origin (from_origin, meaningful only if starts_at_origin) and from the distal end "
                "(from_distal), each with length, chord, DM, kappa p50 / p90 (sigma 3), turning, median radius." % BIN_MM,
    "takeoff_deg": "Angle between the VA's first 5 mm (and 10 mm) chord and the parent subclavian's distal tangent at "
                   "the origin: 0 = continues straight on, 90 = perpendicular, > 90 = recurrent.",
    "elevation_deg": "Angle of the VA's first 10 mm chord above the plane normal to `up`.",
    "calibre": "On the declared radius, smoothed sigma %.1f mm, core %.0f mm off each end: median, p10, p90, mean, "
               "CV = std/mean, rCV = IQR/median. Events: q = r / flank median (median r over [s-W, s-1] and "
               "[s+1, s+W] mm, so the event does not set its own reference); q < 0.8 = narrowing, q > 1.2 = "
               "dilatation, runs >= 1 mm; counted for W = 5 and 10 mm, and per 100 mm of core." % (SIGMA_RADIUS_MM, CAL_END_EXCL_MM),
    "dominance": "L/R ratio of median radius (and of area); > 1.2 or < 1/1.2 = one side dominant (diameter rule).",
    "radius_kind": "What `radii` are: misr (VMTK maximal inscribed sphere; the tube v4 builds), edt, lofted contour, "
                   "host tube, or fitted from the surface.",
    "wall_native": "On the source's own wall surface (triangle centres, area-weighted). Each wall cell is assigned to "
                   "the centerline sample with the largest r - |x - c| among its 24 nearest (the v4 union's own "
                   "measure). dep = |x - c| - r. Excluded: cells assigned within %.0f mm of either end, cells ahead of / "
                   "behind their sample (|t.(x-c)| > 0.5 |x-c|, end caps), and cells beyond the union capsule "
                   "(|x-c| > %.1f r + %.1f mm: side branches, label blobs; never in a v4 union). "
                   "Reported against (a) the DECLARED tube (what a v4 union sees; out = dep > thr is what survives "
                   "the union's tube floor) and (b) the FITTED plain tube r_ref(s) = area-weighted median |x-c| per "
                   "%.0f mm, smoothed sigma %.0f mm (shape-only, independent of how a source defines its radius)."
                   % (END_EXCL_MM, CAPSULE_FACTOR, CAPSULE_PLUS_MM, FIT_BIN_MM, FIT_SMOOTH_MM),
    "frac_abs_X": "Area fraction of wall with |dep| > X mm (v4 right_route_detail thresholds 0.3 / 0.5).",
    "frac_out_X": "Area fraction with dep > X mm (outward; the v4 union keeps only outward departures).",
    "frac_rel_Y": "Area fraction with |dep| > Y x local radius (0.15 / 0.25): size-free, for the 1.5-2 mm VA "
                  "against the 3-4 mm carotid.",
    "roughness_bands": "Gaussian (Euclidean, area-weighted) smoothing g_sigma of dep_ref on the wall; band(sigma) = "
                       "RMS(g_sigma - g_2sigma) for sigma in %s mm, below = RMS(dep_ref - g_0.375), "
                       "coarse = RMS(g_6). A band is RESOLVED when sigma >= the source's largest voxel "
                       "dimension: below that, partial-volume blur and marching-cubes staircase dominate."
                       % (list(OCTAVES_MM),),
    "lowpass": "frac_abs at sigma S: the wall-detail fraction (vs the fitted tube) after removing scales below S "
               "(g_S(dep_ref)). Compare two sources only at S >= both sources' largest voxel dimension.",
    "wall_union": "The v4 protocol on this VA alone: sdf_mesher.tube_field of the declared centerline+radii on the "
                  "0.45 mm grid, sdf_union_v4.add_real_sections_owned of the surface (margin 0.5 mm, capsule "
                  "1.8 r + 1 mm tapered over 8 mm at the origin end, blind distal end, self-contact window 12 mm), "
                  "marching cubes, largest component; then right_route_detail's measure: dep = | |x-c| - r | on the "
                  "undecimated union, frac > 0.3 / 0.5 mm and > 0.15 r / 0.25 r. core excludes %.0f mm at each end. "
                  "genus = handles the bake ladder would have to suppress. With no surface it is the measure's floor "
                  "(a pure tube)." % END_EXCL_MM,
    "surface_topology": "Native surface: components, open edges, genus of the largest component (Euler, boundary-aware).",
    "wall_sections": "Cross-sections normal to the centerline every 1 mm over the core: the smallest closed loop "
                     "enclosing the centerline point; r_eq = sqrt(area/pi), r_in = min distance from the loop centroid, "
                     "circ = 4 pi A / perimeter^2; perimeter fractions |rho - r_eq| > 0.3 / 0.5 mm (eq_abs), "
                     "rho - r_in > 0.3 / 0.5 mm (in_out), |rho - r_eq| > 0.15 r_eq (eq_rel). Medians and means over "
                     "sections. Same quantities as for SimVascular expert contours, so they compare directly. "
                     "centre_offset_mm = distance from the centerline point to the section's area centroid: the "
                     "centerline's ECCENTRICITY. Every centerline-referenced wall measure (wall_native, wall_union, "
                     "v4's right_route_detail) reads eccentricity as wall detail, so it is valid only for a medial "
                     "centerline (VMTK on TopBrain: ~0.1 mm); recentre() makes one.",
    "centerline_qc": "outside_frac = share of core centerline samples outside the (hole-filled, largest) wall; "
                     "misr_over_declared = median distance-to-wall / declared r over samples inside (1 = the declared "
                     "radius is the inscribed one).",
}


# ---------------------------------------------------------------------------
# centerline helpers
# ---------------------------------------------------------------------------

def arclen(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def resample(points, radii=None, step=STEP_MM):
    """Uniform arclength resampling. Returns (P, R or None, s)."""
    p = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
    p = p[keep]
    r = None if radii is None else np.asarray(radii, float)[keep]
    s = arclen(p)
    if s[-1] < step:
        return p, r, s
    q = np.arange(0.0, s[-1] + 1e-9, step)
    if s[-1] - q[-1] > 0.5 * step:
        q = np.r_[q, s[-1]]
    P = np.stack([np.interp(q, s, p[:, i]) for i in range(3)], 1)
    R = None if r is None else np.interp(q, s, r)
    return P, R, q


def smooth_curve(P, sigma_mm, step=STEP_MM):
    """Gaussian smoothing along arclength; the ends are extended linearly by 4 sigma
    along their own 2-sigma chord first, so the smoothing does not curl them."""
    n = len(P)
    if n < 4 or sigma_mm <= 0:
        return P.copy()
    sig = sigma_mm / step
    pad = int(np.ceil(4 * sig))
    k = int(max(1, min(n - 1, round(2 * sigma_mm / step))))
    t0 = P[k] - P[0]
    t1 = P[-1] - P[-1 - k]
    t0 = t0 / (np.linalg.norm(t0) or 1.0)
    t1 = t1 / (np.linalg.norm(t1) or 1.0)
    pre = P[0] - t0[None, :] * (step * np.arange(pad, 0, -1))[:, None]
    post = P[-1] + t1[None, :] * (step * np.arange(1, pad + 1))[:, None]
    Q = np.vstack([pre, P, post])
    Qs = ndimage.gaussian_filter1d(Q, sig, axis=0, mode="nearest")
    return Qs[pad:pad + n]


def frenet(P, sigma_mm, step=STEP_MM):
    """Tangent, curvature (1/mm), signed torsion (1/mm), principal normal."""
    Q = smooth_curve(P, sigma_mm, step)
    d1 = np.gradient(Q, step, axis=0)
    d2 = np.gradient(d1, step, axis=0)
    d3 = np.gradient(d2, step, axis=0)
    c = np.cross(d1, d2)
    n1 = np.linalg.norm(d1, axis=1)
    nc = np.linalg.norm(c, axis=1)
    kappa = nc / np.maximum(n1 ** 3, 1e-12)
    tau = np.einsum("ij,ij->i", c, d3) / np.maximum(nc ** 2, 1e-12)
    t = d1 / np.maximum(n1[:, None], 1e-12)
    a = d2 - np.einsum("ij,ij->i", d2, t)[:, None] * t
    N = a / np.maximum(np.linalg.norm(a, axis=1)[:, None], 1e-12)
    return t, kappa, tau, N


def _pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


def _r(x, nd=4):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    return x


def _runs(mask, step, min_len_mm):
    """Number of runs of True at least min_len_mm long."""
    if not mask.any():
        return 0
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    a, b = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return int(((b - a) * step >= min_len_mm - 1e-9).sum())


def _greedy_windows(cum, win, thr):
    """Greedy count of non-overlapping windows (win samples) where cum[i+win]-cum[i] >= thr."""
    n, i, cnt = len(cum), 0, 0
    while i + 1 < n:
        j = min(i + win, n - 1)
        if cum[j] - cum[i] >= thr:
            cnt += 1
            i = j
        else:
            i += 1
    return cnt


# ---------------------------------------------------------------------------
# gross shape
# ---------------------------------------------------------------------------

def shape_at(P, s, sigma_mm, step=STEP_MM):
    """Curvature / torsion / turning / tortuosity at one smoothing scale."""
    t, kappa, tau, N = frenet(P, sigma_mm, step)
    L = float(s[-1])
    core = (s >= END_EXCL_MM) & (s <= L - END_EXCL_MM)
    if core.sum() < 4:
        core = np.ones(len(s), bool)
    dth = np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0))
    cum = np.r_[0.0, np.cumsum(dth)]
    out = {"sigma_mm": sigma_mm,
           "kappa_p50": _r(_pct(kappa[core], 50)), "kappa_p90": _r(_pct(kappa[core], 90)),
           "kappa_max": _r(float(kappa[core].max())),
           "min_bend_radius_mm": _r(1.0 / max(float(kappa[core].max()), 1e-9), 2),
           "turning_total_deg": _r(math.degrees(cum[-1]), 1),
           "SOAM_deg_per_cm": _r(math.degrees(cum[-1]) / max(L, 1e-9) * 10.0, 2)}
    bend = core & (kappa > KAPPA_TORSION_MIN)
    out["torsion_frac"] = _r(float(bend.sum()) / max(int(core.sum()), 1), 3)
    out["abs_tau_p50"] = _r(_pct(np.abs(tau[bend]), 50)) if bend.any() else None
    out["abs_tau_p90"] = _r(_pct(np.abs(tau[bend]), 90)) if bend.any() else None
    for W in (30.0, 60.0):
        w = int(round(W / step))
        if len(cum) > w:
            out["turn_max_%dmm_deg" % W] = _r(math.degrees(float((cum[w:] - cum[:-w]).max())), 1)
        else:
            out["turn_max_%dmm_deg" % W] = _r(math.degrees(float(cum[-1])), 1)
    # tangent reversal within 30 mm
    w = int(round(30.0 / step))
    rev = np.zeros(len(t))
    for k in range(1, min(w, len(t) - 1) + 1):
        c = np.einsum("ij,ij->i", t[:-k], t[k:])
        rev[:-k] = np.maximum(rev[:-k], np.degrees(np.arccos(np.clip(c, -1, 1))))
    out["reversal_max_30mm_deg"] = _r(float(rev.max()), 1)
    out["n_loops_360_60mm"] = _greedy_windows(cum, int(round(60.0 / step)), 2 * math.pi)
    # U-turns: greedy over windows whose end tangents reverse >= 150 deg
    cnt, i = 0, 0
    while i < len(t) - 1:
        j = min(i + w, len(t) - 1)
        c = np.einsum("j,ij->i", t[i], t[i + 1:j + 1])
        hit = np.nonzero(np.degrees(np.arccos(np.clip(c, -1, 1))) >= 150.0)[0]
        if len(hit):
            cnt += 1
            i = i + 1 + int(hit[0])
        else:
            i += 1
    out["n_uturns_150_30mm"] = cnt
    # inflections: principal-normal flips where the curve bends
    # (kappa -> 0 AT an inflection, so the flip itself is not gated on kappa: the
    # curve must bend on both sides of it, within 5 mm, for it to count)
    wn = int(round(5.0 / step))
    kpad = np.r_[np.zeros(wn), kappa, np.zeros(wn)]
    kmax_before = np.array([kpad[i:i + wn + 1].max() for i in range(len(kappa))])
    kmax_after = np.array([kpad[i + wn:i + 2 * wn + 1].max() for i in range(len(kappa))])
    ok = (kmax_before > KAPPA_TORSION_MIN) & (kmax_after > KAPPA_TORSION_MIN)
    flips = np.nonzero((np.einsum("ij,ij->i", N[:-1], N[1:]) < 0) & ok[:-1] & core[:-1])[0]
    if len(flips):
        merged = [flips[0]]
        for f in flips[1:]:
            if (f - merged[-1]) * step >= 2.0:
                merged.append(f)
        n_inf = len(merged)
    else:
        n_inf = 0
    chord = float(np.linalg.norm(P[-1] - P[0]))
    dm = L / max(chord, 1e-9)
    out["inflections"] = n_inf
    out["ICM"] = _r(dm * (n_inf + 1), 3)
    return out, t, kappa, cum


def _segment_stats(P, R, s, lo, hi, kappa, cum):
    m = (s >= lo) & (s <= hi)
    if m.sum() < 8:
        return None
    idx = np.nonzero(m)[0]
    L = float(s[idx[-1]] - s[idx[0]])
    if L < 10.0:
        return None
    ch = float(np.linalg.norm(P[idx[-1]] - P[idx[0]]))
    return {"from_mm": _r(float(lo), 1), "to_mm": _r(float(min(hi, s[-1])), 1), "length_mm": _r(L, 1),
            "chord_mm": _r(ch, 1), "DM": _r(L / max(ch, 1e-9), 3),
            "kappa_p50": _r(_pct(kappa[m], 50)), "kappa_p90": _r(_pct(kappa[m], 90)),
            "turning_deg": _r(math.degrees(float(cum[idx[-1]] - cum[idx[0]])), 1),
            "r_median_mm": _r(_pct(R[m], 50), 3) if R is not None else None}


def gross_metrics(P, R, s, landmarks=None, starts_at_origin=False, parent_dir=None, up=None):
    L = float(s[-1])
    chord = float(np.linalg.norm(P[-1] - P[0]))
    out = {"length_mm": _r(L, 2), "chord_mm": _r(chord, 2), "DM": _r(L / max(chord, 1e-9), 4),
           "starts_at_origin": bool(starts_at_origin)}
    shp3, _, kap3, cum3 = shape_at(P, s, SIGMA_COMMON_MM)
    shp15, _, _, _ = shape_at(P, s, SIGMA_FINE_MM)
    out["shape@3.0"] = shp3
    out["shape@1.5"] = shp15
    seg = {}
    if landmarks:
        order = [("V1", 0.0, landmarks.get("C6")), ("V2", landmarks.get("C6"), landmarks.get("C2")),
                 ("V3", landmarks.get("C2"), landmarks.get("dura")), ("V4", landmarks.get("dura"), L)]
        for nm, a, b in order:
            if a is not None and b is not None:
                seg[nm] = _segment_stats(P, R, s, a, b, kap3, cum3)
    out["segments_landmarks"] = seg or None
    fo = []
    if starts_at_origin:
        for k in range(int(np.ceil(L / BIN_MM))):
            st = _segment_stats(P, R, s, k * BIN_MM, (k + 1) * BIN_MM, kap3, cum3)
            if st:
                fo.append(st)
    out["bins_from_origin"] = fo or None
    fd = []
    for k in range(int(np.ceil(L / BIN_MM))):
        lo, hi = L - (k + 1) * BIN_MM, L - k * BIN_MM
        st = _segment_stats(P, R, s, max(lo, 0.0), hi, kap3, cum3)
        if st:
            st["distal_from_mm"], st["distal_to_mm"] = k * BIN_MM, (k + 1) * BIN_MM
            fd.append(st)
    out["bins_from_distal"] = fd or None
    # take-off
    if starts_at_origin and (parent_dir is not None or up is not None):
        tk = {}
        for d in (5.0, 10.0):
            i = int(np.searchsorted(s, d))
            if i < len(P):
                v = P[i] - P[0]
                v = v / (np.linalg.norm(v) or 1.0)
                if parent_dir is not None:
                    pd = np.asarray(parent_dir, float)
                    pd = pd / (np.linalg.norm(pd) or 1.0)
                    tk["takeoff_%dmm_deg" % d] = _r(math.degrees(math.acos(np.clip(v @ pd, -1, 1))), 1)
                if up is not None and d == 10.0:
                    u = np.asarray(up, float)
                    u = u / (np.linalg.norm(u) or 1.0)
                    tk["elevation_deg"] = _r(90.0 - math.degrees(math.acos(np.clip(v @ u, -1, 1))), 1)
        out["takeoff"] = tk
    return out


# ---------------------------------------------------------------------------
# calibre
# ---------------------------------------------------------------------------

def calibre_metrics(R, s, step=STEP_MM):
    L = float(s[-1])
    core = (s >= CAL_END_EXCL_MM) & (s <= L - CAL_END_EXCL_MM)
    if core.sum() < 8:
        core = np.ones(len(s), bool)
    Rs = ndimage.gaussian_filter1d(np.asarray(R, float), SIGMA_RADIUS_MM / step, mode="nearest")
    rc = Rs[core]
    out = {"r_median_mm": _r(_pct(rc, 50), 3), "r_p10_mm": _r(_pct(rc, 10), 3), "r_p90_mm": _r(_pct(rc, 90), 3),
           "r_mean_mm": _r(float(rc.mean()), 3), "r_min_mm": _r(float(rc.min()), 3), "r_max_mm": _r(float(rc.max()), 3),
           "CV": _r(float(rc.std() / max(rc.mean(), 1e-9)), 4),
           "rCV": _r(float((np.percentile(rc, 75) - np.percentile(rc, 25)) / max(np.median(rc), 1e-9)), 4),
           "core_length_mm": _r(float(core.sum() * step), 1)}
    for W in (5.0, 10.0):
        med = _flank_median(Rs, step, W)
        q = Rs / np.maximum(med, 1e-6)
        nar = _runs((q < 0.8) & core, step, 1.0)
        dil = _runs((q > 1.2) & core, step, 1.0)
        out["narrowings_W%d" % W] = nar
        out["dilatations_W%d" % W] = dil
        out["events_per_100mm_W%d" % W] = _r(100.0 * (nar + dil) / max(core.sum() * step, 1e-9), 3)
        out["q_min_W%d" % W] = _r(float(q[core].min()), 3)
        out["q_max_W%d" % W] = _r(float(q[core].max()), 3)
    return out


def _flank_median(Rs, step, W, gap_mm=1.0):
    """Reference calibre at each sample: median radius over the two flanks
    [s - W, s - gap] and [s + gap, s + W], so an event up to ~W long does not
    set its own reference (a centred running median misses events > W/2)."""
    n = len(Rs)
    w, g = int(round(W / step)), int(round(gap_mm / step))
    out = np.empty(n)
    for i in range(n):
        fl = np.r_[Rs[max(0, i - w):max(0, i - g + 1)], Rs[min(n, i + g):min(n, i + w + 1)]]
        out[i] = np.median(fl) if len(fl) else Rs[i]
    return out


def dominance(r_left_median, r_right_median):
    if not r_left_median or not r_right_median:
        return None
    q = r_left_median / r_right_median
    return {"LR_radius_ratio": _r(q, 3), "LR_area_ratio": _r(q * q, 3),
            "dominant": "left" if q > 1.2 else ("right" if q < 1 / 1.2 else "co-dominant")}


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------

def surface_topology(surface):
    """Components, open edges, genus of the largest component (Euler, boundary-aware)."""
    m = surface.triangulate().clean()
    conn = m.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"]) if m.n_cells else np.zeros(0, int)
    ncomp = int(rid.max()) + 1 if len(rid) else 0
    big = conn.extract_cells(np.nonzero(rid == np.argmax(np.bincount(rid)))[0]).extract_surface().clean() if ncomp else m
    V, F = big.n_points, big.n_cells
    E = big.extract_all_edges().n_cells
    b = 0
    if big.n_open_edges:
        be = big.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,
                                       non_manifold_edges=False)
        b = int(be.connectivity().point_data["RegionId"].max()) + 1 if be.n_points else 0
    A = np.asarray(m.compute_cell_sizes(length=False, volume=False)["Area"]) if m.n_cells else np.zeros(0)
    Abig = float(np.asarray(big.compute_cell_sizes(length=False, volume=False)["Area"]).sum()) if big.n_cells else 0.0
    return {"components": ncomp, "open_edges": int(m.n_open_edges), "largest_genus": (2 - (V - E + F) - b) / 2.0,
            "largest_boundary_loops": b, "largest_area_frac": _r(Abig / max(float(A.sum()), 1e-9), 4),
            "area_mm2": _r(float(A.sum()), 1), "n_cells": int(m.n_cells),
            "mean_edge_mm": _r(float(np.sqrt(4.0 * A.mean() / math.sqrt(3.0))), 3) if len(A) else None}


def prepare_surface(surface, largest=True):
    """As sdf_union.load_surface, without the transform: triangulate, clean, largest
    component, fill holes, outward normals (the implicit distance's sign follows them)."""
    m = surface
    if not hasattr(m, "faces"):
        m = m.extract_surface()
    m = m.triangulate().clean()
    if largest:
        m = m.connectivity("largest").extract_surface().triangulate()
    if m.n_open_edges:
        m = m.fill_holes(1000.0).clean().triangulate()
    m = m.compute_normals(auto_orient_normals=True, consistent_normals=True, point_normals=True,
                          cell_normals=True, inplace=False)
    return m


def radii_from_surface(P, surface, mode="misr", s=None):
    """Radius per centerline sample from a wall: misr = distance to the nearest wall
    point (VMTK's maximal inscribed sphere when the centerline is medial);
    mean = area-weighted median |x - c| of the wall cells assigned to it (+-0.5 mm)."""
    import pyvista as pv
    surf = surface.triangulate().clean()
    if mode == "misr":
        d = np.asarray(pv.PolyData(np.asarray(P, float)).compute_implicit_distance(surf)["implicit_distance"])
        return np.abs(d)
    cen = np.asarray(surf.cell_centers().points)
    A = np.asarray(surf.compute_cell_sizes(length=False, volume=False)["Area"])
    d, i = cKDTree(P).query(cen)
    out = np.full(len(P), np.nan)
    for k in range(len(P)):
        m = i == k
        if m.any():
            out[k] = _wmedian(d[m], A[m])
    good = np.isfinite(out)
    sl = s if s is not None else arclen(P)
    return np.interp(sl, sl[good], out[good])


def _wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def _gauss_field(x, a, v, sigma):
    """Area-weighted Gaussian smoothing of a scalar on surface points (Euclidean
    distance), evaluated on surfels of size sigma/3 and mapped back."""
    g = max(sigma / 3.0, 0.08)
    key = np.floor(x / g).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    A = np.bincount(inv, a)
    C = np.stack([np.bincount(inv, a * x[:, i]) for i in range(3)], 1) / A[:, None]
    V = np.bincount(inv, a * v) / A
    T = cKDTree(C)
    M = T.sparse_distance_matrix(T, 2.5 * sigma, output_type="coo_matrix")
    keep = M.row != M.col
    row, col, dd = M.row[keep], M.col[keep], M.data[keep]
    w = np.exp(-dd ** 2 / (2 * sigma * sigma)) * A[col]
    num = np.bincount(row, w * V[col], minlength=len(C)) + A * V
    den = np.bincount(row, w, minlength=len(C)) + A
    return (num / den)[inv]


def _fracs(dep, A, r, prefix, relative=True, signed=True):
    tot = float(A.sum()) or 1.0
    out = {}
    for t in ABS_THR_MM:
        out["%sfrac_abs_%.1f" % (prefix, t)] = _r(float(A[np.abs(dep) > t].sum()) / tot)
        if signed:
            out["%sfrac_out_%.1f" % (prefix, t)] = _r(float(A[dep > t].sum()) / tot)
    if relative:
        for q in REL_THR:
            out["%sfrac_rel_%.2f" % (prefix, q)] = _r(float(A[np.abs(dep) > q * r].sum()) / tot)
    return out


def wall_native(P, R, s, surface, voxel_mm=None):
    """Wall detail on the source's own surface (see METRIC_DEFINITIONS["wall_native"])."""
    surf = surface.triangulate().clean()
    cen = np.asarray(surf.cell_centers().points)
    A = np.asarray(surf.compute_cell_sizes(length=False, volume=False)["Area"])
    k = min(24, len(P))
    dd, ii = cKDTree(P).query(cen, k=k)
    if dd.ndim == 1:
        dd, ii = dd[:, None], ii[:, None]
    best = ii[np.arange(len(ii)), np.argmax(R[ii] - dd, axis=1)]
    t, _, _, _ = frenet(P, SIGMA_TANGENT_MM)
    v = cen - P[best]
    rho = np.linalg.norm(v, axis=1)
    rb = R[best]
    dep = rho - rb
    axial = np.abs(np.einsum("ij,ij->i", v, t[best])) / np.maximum(rho, 1e-9)
    L = float(s[-1])
    inner = (s[best] >= END_EXCL_MM) & (s[best] <= L - END_EXCL_MM)
    capsule = rho <= CAPSULE_FACTOR * rb + CAPSULE_PLUS_MM
    side = axial <= 0.5
    use = inner & capsule & side
    Atot = float(A.sum()) or 1.0
    out = {"area_mm2": _r(Atot, 1), "used_area_frac": _r(float(A[use].sum()) / Atot),
           "beyond_capsule_area_frac": _r(float(A[inner & ~capsule].sum()) / Atot),
           "end_or_cap_area_frac": _r(float(A[~inner | (capsule & ~side)].sum()) / Atot)}
    if use.sum() < 50:
        out["note"] = "fewer than 50 wall cells beside the centerline"
        return out
    x, a, rr, dp = cen[use], A[use], rb[use], dep[use]
    sb = s[best][use]
    out.update(_fracs(dp, a, rr, "declared_"))
    out["declared_dep_median_mm"] = _r(_wmedian(dp, a), 3)
    # fitted plain tube
    rho_u = rho[use]
    nb = int(np.ceil(L / FIT_BIN_MM)) + 1
    bi = np.clip((sb / FIT_BIN_MM).astype(int), 0, nb - 1)
    rfit = np.full(nb, np.nan)
    for b in np.unique(bi):
        m = bi == b
        rfit[b] = _wmedian(rho_u[m], a[m])
    centres = (np.arange(nb) + 0.5) * FIT_BIN_MM
    good = np.isfinite(rfit)
    rf = np.interp(np.arange(nb), np.nonzero(good)[0], rfit[good])
    rf = ndimage.gaussian_filter1d(rf, FIT_SMOOTH_MM / FIT_BIN_MM, mode="nearest")
    r_ref = np.interp(sb, centres, rf)
    dref = rho_u - r_ref
    out.update(_fracs(dref, a, r_ref, "fitted_"))
    out["fitted_rms_mm"] = _r(float(np.sqrt(np.sum(a * dref ** 2) / a.sum())), 4)
    out["fitted_rms_rel"] = _r(float(np.sqrt(np.sum(a * (dref / r_ref) ** 2) / a.sum())), 4)
    out["declared_over_fitted_radius"] = _r(float(np.median(rr / r_ref)), 3)
    # scale analysis
    hmax = float(max(voxel_mm)) if voxel_mm is not None else None
    sig_all = sorted(set(OCTAVES_MM) | {2 * o for o in OCTAVES_MM} | set(LOWPASS_COMMON_MM))
    g = {sg: _gauss_field(x, a, dref, sg) for sg in sig_all}
    bands = {}
    bands["below_0.375"] = {"rms_mm": _r(float(np.sqrt(np.sum(a * (dref - g[OCTAVES_MM[0]]) ** 2) / a.sum())), 4),
                            "resolved": bool(hmax is not None and hmax <= 0.375 / 2)}
    for sg in OCTAVES_MM:
        b = g[sg] - g[2 * sg]
        bands["%.3g-%.3g" % (sg, 2 * sg)] = {"rms_mm": _r(float(np.sqrt(np.sum(a * b ** 2) / a.sum())), 4),
                                             "resolved": bool(hmax is not None and sg >= hmax)}
    bands["coarse_6"] = {"rms_mm": _r(float(np.sqrt(np.sum(a * g[6.0] ** 2) / a.sum())), 4), "resolved": True}
    out["roughness_bands"] = bands
    lp = {}
    for sg in LOWPASS_COMMON_MM:
        lp["sigma_%.2f" % sg] = dict(_fracs(g[sg], a, r_ref, "fitted_"),
                                     comparable=bool(hmax is None or sg >= hmax))
    out["lowpass"] = lp
    out["voxel_max_mm"] = hmax
    return out


def _polygon_stats(Q2):
    """2-D closed polygon (n,2): area, centroid, perimeter (shoelace)."""
    x, y = Q2[:, 0], Q2[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = 0.5 * cr.sum()
    if abs(A) < 1e-12:
        return 0.0, Q2.mean(0), 0.0
    cx = ((x + xn) * cr).sum() / (6 * A)
    cy = ((y + yn) * cr).sum() / (6 * A)
    per = float(np.hypot(xn - x, yn - y).sum())
    return abs(float(A)), np.array([cx, cy]), per


def _winding(Q2, p):
    """Winding number of polygon Q2 around point p (nonzero = inside)."""
    d = Q2 - p
    a = np.arctan2(d[:, 1], d[:, 0])
    da = np.diff(np.r_[a, a[0]])
    da = (da + np.pi) % (2 * np.pi) - np.pi
    return int(round(da.sum() / (2 * np.pi)))


def _section_loop(surf, c, tn):
    """The smallest closed loop of the plane section (normal tn through c) that
    encloses c. Returns (area, centroid_2d, perimeter, Q2, u, v) or None."""
    sl = surf.slice(normal=tn, origin=c)
    if sl.n_points < 6:
        return None
    u = np.cross(tn, [1.0, 0, 0] if abs(tn[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(tn, u)
    best = None
    try:
        strips = sl.strip()
    except Exception:                                                 # noqa: BLE001
        strips = sl
    lines = np.asarray(strips.lines)
    pts_all = np.asarray(strips.points)
    k = 0
    while k < len(lines):
        n = int(lines[k])
        ids = lines[k + 1:k + 1 + n]
        k += n + 1
        if n < 6:
            continue
        pts = pts_all[ids]
        if np.linalg.norm(pts[0] - pts[-1]) > 0.5:
            continue                          # not a closed loop (cut at a mask edge / an open wall end)
        Q2 = np.c_[(pts - c) @ u, (pts - c) @ v]
        if _winding(Q2, np.zeros(2)) == 0:
            continue
        A, cen, per = _polygon_stats(Q2)
        if best is None or A < best[0]:
            best = (A, cen, per, Q2, u, v)
    return best


def recentre(P, surface, every_mm=0.5, iters=2):
    """A medial centerline for a source whose path is not medial (a hand-drawn
    SimVascular path, a skeleton): each point moves to the area centroid of
    its wall section (where a closed loop encloses it), the result is smoothed
    sigma 1 mm and resampled; twice. Radius = MISR (distance to the wall), as
    VMTK would give. Returns (P, R, info)."""
    surf = surface.triangulate().clean()
    Pc, _, _ = resample(P, None, every_mm)
    moved = []
    for _ in range(iters):
        t, _, _, _ = frenet(Pc, SIGMA_TANGENT_MM, every_mm)
        new = Pc.copy()
        mv = np.zeros(len(Pc))
        for i in range(len(Pc)):
            lp = _section_loop(surf, Pc[i], t[i])
            if lp is None:
                continue
            cen, u, v = lp[1], lp[4], lp[5]
            new[i] = Pc[i] + cen[0] * u + cen[1] * v
            mv[i] = float(np.hypot(*cen))
        moved.append(mv)
        new = smooth_curve(new, 1.0, every_mm)
        Pc, _, _ = resample(new, None, every_mm)
    R = radii_from_surface(Pc, prepare_surface(surface), "misr")
    info = {"iters": iters, "first_pass_offset_median_mm": _r(float(np.median(moved[0])), 3),
            "first_pass_offset_p90_mm": _r(float(np.percentile(moved[0], 90)), 3),
            "last_pass_offset_median_mm": _r(float(np.median(moved[-1])), 3)}
    return Pc, R, info


def section_detail(P, R, s, surface, every_mm=1.0):
    """Cross-sections of the wall normal to the centerline, every_mm apart over
    the core: the closed loop that encloses the centerline point (the smallest,
    if several), its equal-area radius r_eq, inscribed radius r_in = min rho
    about the loop centroid, circularity 4 pi A / perimeter^2, and the perimeter
    fraction standing > 0.3 / 0.5 mm off the equal-area circle (|rho - r_eq|)
    and off the inscribed circle (rho - r_in), and > 0.15 r_eq. The 2-D
    analogue of the wall measures, directly comparable with expert contours
    (SimVascular .ctgr); medians over sections are reported."""
    surf = surface.triangulate().clean()
    t, _, _, _ = frenet(P, SIGMA_TANGENT_MM)
    L = float(s[-1])
    stations = np.arange(END_EXCL_MM, L - END_EXCL_MM + 1e-9, every_mm)
    rows, miss = [], 0
    for st in stations:
        i = int(np.argmin(np.abs(s - st)))
        best = _section_loop(surf, P[i], t[i])
        if best is None:
            miss += 1
            continue
        A, cen, per, Q2, _, _ = best
        # dense perimeter, uniform in arclength
        Pq = np.vstack([Q2, Q2[:1]])
        sg = np.r_[0, np.cumsum(np.hypot(*np.diff(Pq, axis=0).T))]
        q = np.linspace(0, sg[-1], 360, endpoint=False)
        D = np.c_[np.interp(q, sg, Pq[:, 0]), np.interp(q, sg, Pq[:, 1])]
        rho = np.hypot(*(D - cen).T)
        r_eq = math.sqrt(A / math.pi)
        r_in = float(rho.min())
        rows.append({"s": float(st), "r_eq": r_eq, "r_in": r_in, "circ": 4 * math.pi * A / max(per * per, 1e-9),
                     "eq_abs_0.3": float((np.abs(rho - r_eq) > 0.3).mean()),
                     "eq_abs_0.5": float((np.abs(rho - r_eq) > 0.5).mean()),
                     "in_out_0.3": float((rho - r_in > 0.3).mean()),
                     "in_out_0.5": float((rho - r_in > 0.5).mean()),
                     "eq_rel_0.15": float((np.abs(rho - r_eq) > 0.15 * r_eq).mean()),
                     "declared_over_req": float(R[i] / max(r_eq, 1e-9)),
                     "centre_offset_mm": float(np.hypot(*cen))})
    if not rows:
        return {"n_sections": 0, "n_missed": miss}
    keys = [k for k in rows[0] if k != "s"]
    out = {"n_sections": len(rows), "n_missed": miss, "every_mm": every_mm}
    for k in keys:
        a = np.array([r[k] for r in rows])
        out[k + "_median"] = _r(float(np.median(a)), 4)
        out[k + "_mean"] = _r(float(a.mean()), 4)
    return out


def centerline_qc(P, R, s, surface):
    """Is the centerline inside its own wall, and is the declared radius the
    inscribed one? outside_frac = share of core samples outside the closed
    surface (v4's known limitation: 80 TopBrain VA sections > 10%);
    misr_over_declared = median (distance to wall) / declared r."""
    import pyvista as pv
    surf = prepare_surface(surface)
    L = float(s[-1])
    core = (s >= END_EXCL_MM) & (s <= L - END_EXCL_MM)
    pts = pv.PolyData(np.asarray(P, float))
    ins = np.asarray(pts.select_enclosed_points(surf, tolerance=0.0, check_surface=False)["SelectedPoints"], bool)
    d = np.abs(np.asarray(pts.compute_implicit_distance(surf)["implicit_distance"]))
    m = core & ins
    return {"outside_frac": _r(float((~ins[core]).mean()) if core.any() else None, 4),
            "misr_over_declared": _r(float(np.median(d[m] / np.maximum(R[m], 1e-6))) if m.any() else None, 3)}


def union_detail(P, R, surface=None, spacing=None, margin=0.5, taper_start=True, taper_end=False):
    """The v4 protocol on this VA alone (METRIC_DEFINITIONS["wall_union"])."""
    if TOPBRAIN_TOOLS not in sys.path:
        sys.path.insert(0, TOPBRAIN_TOOLS)
    from sdf_mesher import SPACING_MM, iso_surface, tube_field
    from sdf_union_v4 import Samples, Section, add_real_sections_owned, genus
    spacing = spacing or SPACING_MM
    P = np.asarray(P, float)
    R = np.asarray(R, float)
    va = SimpleNamespace(name="VA", coordinates=P, radii=R)
    # a radius-0 branch only widens the grid (it marks no voxel): the real surface
    # may stand out up to 1.8 r + 1 mm, past the tube's own padding
    pad = CAPSULE_FACTOR * float(R.max()) + CAPSULE_PLUS_MM + 3.0
    lo, hi = P.min(0) - pad, P.max(0) + pad
    dummy = SimpleNamespace(name="pad", coordinates=np.array([lo, hi]), radii=np.zeros(2))
    samp = Samples([va])
    field = tube_field([va, dummy], spacing=spacing)
    st = {}
    if surface is not None:
        surf = prepare_surface(surface)
        sec = Section("va", surf, P, R, taper_start, taper_end, "VA", 0.0)
        st = add_real_sections_owned(field, [sec], samp, spacing, margin)
    iso = iso_surface(field)
    conn = iso.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"])
    big = int(np.argmax(np.bincount(rid)))
    ncomp = int(rid.max()) + 1
    iso = conn.extract_cells(np.nonzero(rid == big)[0]).extract_surface().triangulate()
    gen = genus(iso)
    out = {"spacing_mm": spacing, "margin_mm": margin, "components_before_keep": ncomp,
           "genus": max(c["genus"] for c in gen) if gen else None, "union_tris": int(iso.n_cells),
           "section_stats": st.get("va")}
    L = float(samp.s[samp.b == 0].max())
    bi = samp.index_of("VA")
    out.update(measure_union_branch(iso, samp, bi, 0.0, L))
    # the same measure per 40 mm bin, from the distal end and from the origin
    for tag, edges in (("bins_from_distal", [(max(L - (k + 1) * BIN_MM, 0.0), L - k * BIN_MM, k * BIN_MM)
                                             for k in range(int(np.ceil(L / BIN_MM)))]),
                       ("bins_from_origin", [(k * BIN_MM, min((k + 1) * BIN_MM, L), k * BIN_MM)
                                             for k in range(int(np.ceil(L / BIN_MM)))])):
        rows = []
        for lo, hi, d0 in edges:
            if hi - lo < 10.0:
                continue
            m = measure_union_branch(iso, samp, bi, lo, hi)["core"]
            m.update({"from_mm": _r(lo, 1), "to_mm": _r(hi, 1), "bin_start_mm": d0})
            rows.append(m)
        out[tag] = rows
    return out


def measure_union_branch(mesh, samp, bidx, lo_mm, hi_mm, end_excl=END_EXCL_MM, return_cells=False):
    """right_route_detail's measure on a union for one branch's [lo, hi] stretch:
    cells owned by the branch (largest r - |x-c| among the 24 nearest samples of
    ALL branches), dep = | |x-c| - r |, frac > 0.3 / 0.5 mm and > 0.15 / 0.25 r.
    'all' is right_route_detail exactly; 'core' drops end_excl mm at each end."""
    cen = np.asarray(mesh.cell_centers().points)
    A = np.asarray(mesh.compute_cell_sizes(length=False, volume=False)["Area"])
    dd, ii = cKDTree(samp.p).query(cen, k=24)
    best = ii[np.arange(len(ii)), np.argmax(samp.r[ii] - dd, axis=1)]
    sb = samp.s[best]
    own = (samp.b[best] == bidx) & (sb >= lo_mm) & (sb <= hi_mm)
    dep = np.abs(np.linalg.norm(cen - samp.p[best], axis=1) - samp.r[best])
    rr = samp.r[best]
    out = {}
    Lb = float(samp.s[samp.b == bidx].max())
    for tag, m in (("all", own), ("core", own & (sb >= lo_mm + end_excl) & (sb <= min(hi_mm, Lb) - end_excl))):
        a, x, r = A[m], dep[m], rr[m]
        tot = float(a.sum()) or 1.0
        d = {"area_mm2": _r(float(a.sum()), 1)}
        for t in ABS_THR_MM:
            d["frac_%.1fmm" % t] = _r(float(a[x > t].sum()) / tot)
        for q in REL_THR:
            d["frac_rel_%.2f" % q] = _r(float(a[x > q * r].sum()) / tot)
        out[tag] = d
    if return_cells:
        return out, (cen, A, best, dep, own)
    return out


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def measure_va(points, radii=None, surface=None, voxel_mm=None, source_type=None, radius_kind=None,
               landmarks=None, starts_at_origin=False, parent_dir=None, up=None, name=None,
               do_union=True, union_spacing=None, extra=None):
    """All metrics for one VA. See the module docstring for the inputs."""
    P, R, s = resample(points, radii)
    if R is None:
        if surface is None:
            raise ValueError("no radii and no surface")
        R = radii_from_surface(P, surface, "misr")
        radius_kind = (radius_kind or "") + " fitted from the surface (misr)"
    out = {"name": name, "version": VERSION,
           "provenance": {"source_type": source_type, "radius_kind": radius_kind,
                          "voxel_mm": None if voxel_mm is None else [float(v) for v in voxel_mm],
                          "voxel_max_mm": None if voxel_mm is None else float(max(voxel_mm)),
                          "has_surface": surface is not None, "n_points_in": int(len(points))}}
    if extra:
        out["provenance"].update(extra)
    out["gross"] = gross_metrics(P, R, s, landmarks, starts_at_origin, parent_dir, up)
    out["calibre"] = calibre_metrics(R, s)
    if surface is not None:
        out["surface_topology"] = surface_topology(surface)
        out["wall_native"] = wall_native(P, R, s, surface, voxel_mm)
        out["centerline_qc"] = centerline_qc(P, R, s, surface)
        out["wall_sections"] = section_detail(P, R, s, surface)
    if do_union:
        out["wall_union"] = union_detail(P, R, surface, spacing=union_spacing)
    return out


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------

def _tube_surface(P, r, n_sides=96, bumps=None):
    """A closed tube surface along P (constant r), optional radial bumps
    bumps = [(s_lo, s_hi, theta_lo, theta_hi, height_mm)]."""
    import pyvista as pv
    P, _, s = resample(P, None, 0.1)
    t, _, _, _ = frenet(P, 1.0, 0.1)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    U, V = [], []
    u = ref - (ref @ t[0]) * t[0]
    u /= np.linalg.norm(u)
    for i in range(len(P)):
        u = u - (u @ t[i]) * t[i]
        u /= np.linalg.norm(u)
        U.append(u.copy())
        V.append(np.cross(t[i], u))
    th = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    pts = []
    for i in range(len(P)):
        rr = np.full(n_sides, r)
        if bumps:
            for a, b, t0, t1, h in bumps:
                if a <= s[i] <= b:
                    rr[(th >= t0) & (th < t1)] += h
        pts.append(P[i] + rr[:, None] * (np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i]))
    pts = np.vstack(pts)
    faces = []
    for i in range(len(P) - 1):
        for j in range(n_sides):
            a, b = i * n_sides + j, i * n_sides + (j + 1) % n_sides
            c, d = a + n_sides, b + n_sides
            faces += [[3, a, b, d], [3, a, d, c]]
    m = pv.PolyData(pts, np.array(faces).ravel())
    return m.fill_holes(1000.0).triangulate().clean()


def selftest():
    res = {}
    # 1 helix a=10, b=5: kappa = 0.08, tau = 0.04 /mm
    tt = np.linspace(0, 4 * np.pi, 4000)
    H = np.c_[10 * np.cos(tt), 10 * np.sin(tt), 5 * tt]
    P, _, s = resample(H)
    sh, _, kap, _ = shape_at(P, s, 1.5)
    res["helix"] = {"kappa_p50": sh["kappa_p50"], "want": 0.08, "abs_tau_p50": sh["abs_tau_p50"], "want_tau": 0.04,
                    "turn_total": sh["turning_total_deg"], "n_loops": sh["n_loops_360_60mm"]}
    # 1b planar sine y = 5 sin(2 pi x / 40), x in [0, 120]: inflections at x = 20, 40, 60, 80, 100
    xs = np.linspace(0, 120, 2400)
    Sn = np.c_[xs, 5 * np.sin(2 * np.pi * xs / 40), np.zeros_like(xs)]
    Ps, _, ss = resample(Sn)
    shs, _, _, _ = shape_at(Ps, ss, 3.0)
    res["sine"] = {"inflections": shs["inflections"], "want": 5}
    # 2 straight tube with a bump band: 10% of the circumference over the middle 40 of 80 mm, +0.6 mm
    S = np.c_[np.zeros(321), np.zeros(321), np.linspace(0, 80, 321)]
    plain = _tube_surface(S, 2.0)
    bump = _tube_surface(S, 2.0, bumps=[(20, 60, 0, 2 * np.pi * 0.1, 0.6)])
    Pp, Rp, sp = resample(S, np.full(len(S), 2.0))
    w0 = wall_native(Pp, Rp, sp, plain, voxel_mm=(0.3, 0.3, 0.3))
    w1 = wall_native(Pp, Rp, sp, bump, voxel_mm=(0.3, 0.3, 0.3))
    res["tube_plain"] = {k: w0[k] for k in ("declared_frac_abs_0.3", "fitted_frac_abs_0.3", "used_area_frac")}
    res["tube_bump"] = {k: w1[k] for k in ("declared_frac_out_0.5", "declared_frac_abs_0.3", "fitted_frac_abs_0.3")}
    res["tube_bump"]["want_declared_out_0.5"] = round(0.1 * 40 / 76.0 * (2.6 / 2.0), 3)
    # 3 calibre: a 30% narrowing 3 mm long in a 2 mm tube
    Rn = np.full(len(sp), 2.0)
    Rn[(sp > 39) & (sp < 42)] = 1.4
    res["calibre_narrowing"] = {k: v for k, v in calibre_metrics(Rn, sp).items() if k.startswith(("narrow", "dilat"))}
    # 4 union floor (pure tube) and the bump through the v4 union
    try:
        res["union_tube"] = union_detail(Pp, Rp, None)["core"]
        res["union_bump"] = union_detail(Pp, Rp, bump)["core"]
    except Exception as e:                                           # noqa: BLE001
        res["union"] = "skipped: %s" % e
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--definitions" in sys.argv:
        print(json.dumps(METRIC_DEFINITIONS, indent=1))
