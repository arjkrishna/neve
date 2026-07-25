"""Polyline projection and geometry utilities.

Used by ArcLengthProgress reward and LocalGuidance observation to project
device tip positions onto the correct-path polyline.
"""

from typing import NamedTuple, Optional
import numpy as np


class ProjectionResult(NamedTuple):
    """Result of projecting a point onto a polyline."""

    s: float  # arclength coordinate along polyline
    cross_track_dist: float  # perpendicular distance to polyline
    proj_point: np.ndarray  # closest point on polyline (3,)
    segment_idx: int  # index of closest segment
    t: float  # interpolation parameter within segment [0, 1]


def compute_cumulative_arclength(polyline: np.ndarray) -> np.ndarray:
    """Compute cumulative arclength along a polyline.

    Args:
        polyline: (N, 3) array of points.

    Returns:
        (N,) array where cumlen[0] = 0 and cumlen[i] = sum of segment
        lengths from point 0 to point i.
    """
    if len(polyline) < 2:
        return np.zeros(len(polyline))
    diffs = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def compute_segment_tangents(polyline: np.ndarray) -> np.ndarray:
    """Compute unit tangent vectors for each segment.

    Args:
        polyline: (N, 3) array of points.

    Returns:
        (N-1, 3) array of unit tangent vectors. Each tangent[i] points
        from polyline[i] to polyline[i+1].
    """
    diffs = polyline[1:] - polyline[:-1]
    lengths = np.linalg.norm(diffs, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)  # avoid division by zero
    return diffs / lengths


def compute_curvature(tangents: np.ndarray, cumlen: np.ndarray) -> np.ndarray:
    """Compute discrete curvature at interior vertices.

    Curvature is estimated as the angle change between consecutive tangent
    vectors divided by the average segment length.

    Args:
        tangents: (N-1, 3) unit tangent vectors from compute_segment_tangents.
        cumlen: (N,) cumulative arclength from compute_cumulative_arclength.

    Returns:
        (N-2,) array of curvature values at interior vertices (indices 1..N-2).
    """
    if len(tangents) < 2:
        return np.array([])

    # Angle between consecutive tangents
    dots = np.sum(tangents[:-1] * tangents[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)

    # Average segment length at each interior vertex
    seg_lengths = cumlen[1:] - cumlen[:-1]
    avg_lengths = 0.5 * (seg_lengths[:-1] + seg_lengths[1:])
    avg_lengths = np.maximum(avg_lengths, 1e-8)

    return angles / avg_lengths


def _project_onto_segment_range(
    point: np.ndarray,
    polyline: np.ndarray,
    cumlen: np.ndarray,
    i0: int,
    i1: int,
) -> ProjectionResult:
    """Segment-wise closest-point scan over segments [i0, i1).

    Indices stay GLOBAL: the returned segment_idx and s are expressed in
    the full polyline's segment/arclength coordinates regardless of the
    scanned sub-range.
    """
    # Vectorized projection onto the selected segments at once
    p0 = polyline[i0:i1]  # (M, 3)
    p1 = polyline[i0 + 1:i1 + 1]  # (M, 3)
    d = p1 - p0  # segment direction vectors (M, 3)
    seg_len_sq = np.sum(d * d, axis=1)  # (M,)

    # Compute t parameter for each segment
    v = point - p0  # (M, 3)
    t_raw = np.sum(v * d, axis=1) / np.maximum(seg_len_sq, 1e-16)  # (M,)
    t_clamped = np.clip(t_raw, 0.0, 1.0)  # (M,)

    # Projected points on each segment
    proj_points = p0 + t_clamped[:, np.newaxis] * d  # (M, 3)

    # Distances from query point to each projected point
    dists = np.linalg.norm(point - proj_points, axis=1)  # (M,)

    # Find the closest segment (global index = local + range offset)
    best_local = int(np.argmin(dists))
    best_idx = i0 + best_local
    best_t = float(t_clamped[best_local])
    best_proj = proj_points[best_local]
    best_dist = float(dists[best_local])

    # Arclength at the projection
    s = float(cumlen[best_idx] + best_t * (cumlen[best_idx + 1] - cumlen[best_idx]))

    return ProjectionResult(
        s=s,
        cross_track_dist=best_dist,
        proj_point=best_proj,
        segment_idx=best_idx,
        t=best_t,
    )


def project_onto_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
    cumlen: np.ndarray,
    prev_s: Optional[float] = None,
    window_mm: Optional[float] = None,
    fallback_dist_mm: float = 15.0,
) -> ProjectionResult:
    """Project a point onto a polyline via segment-wise closest-point.

    For each segment (p0, p1), computes the closest point on the segment
    to the query point, then returns the overall closest projection.

    Hairpin failure mode: a global argmin over ALL segments is ambiguous
    where the polyline folds back on itself (carotid siphon) — the two
    limbs of the hairpin are spatially adjacent, so sub-mm tip jitter
    flips the winning segment between limbs and the arclength ``s`` jumps
    by the full loop length. When ``prev_s`` and ``window_mm`` are given,
    the search is restricted to segments overlapping the arclength window
    ``[prev_s - window_mm, prev_s + window_mm]``, which keeps consecutive
    projections continuous across the hairpin.

    Fallback rule: if the windowed best cross-track distance exceeds
    ``fallback_dist_mm``, the point has genuinely left the window
    neighborhood (e.g. an elastic snap or a SOFA state restore) and
    continuity must yield to correctness — the full scan runs instead.

    Args:
        point: (3,) query point.
        polyline: (N, 3) polyline vertices.
        cumlen: (N,) cumulative arclength (from compute_cumulative_arclength).
        prev_s: previous projection arclength for windowed continuity.
            None (default) scans all segments.
        window_mm: half-width of the arclength search window around
            ``prev_s``. None (default) scans all segments.
        fallback_dist_mm: max windowed cross-track distance before the
            full scan takes over.

    Returns:
        ProjectionResult with arclength, cross-track distance, projected
        point, segment index, and interpolation parameter.
    """
    point = np.asarray(point, dtype=np.float64).ravel()

    if len(polyline) < 2:
        return ProjectionResult(
            s=0.0,
            cross_track_dist=float(np.linalg.norm(point - polyline[0])),
            proj_point=polyline[0].copy(),
            segment_idx=0,
            t=0.0,
        )

    n_seg = len(polyline) - 1

    if prev_s is not None and window_mm is not None:
        # Windowed scan: segments overlapping [prev_s - w, prev_s + w].
        # Segment i spans arclengths [cumlen[i], cumlen[i+1]].
        i0 = max(
            0, int(np.searchsorted(cumlen, prev_s - window_mm, side="left")) - 1
        )
        i1 = min(
            n_seg, int(np.searchsorted(cumlen, prev_s + window_mm, side="right"))
        )
        if i1 > i0:
            result = _project_onto_segment_range(point, polyline, cumlen, i0, i1)
            if result.cross_track_dist <= fallback_dist_mm:
                return result
        # Window empty or windowed best too far — fall back to full scan.

    return _project_onto_segment_range(point, polyline, cumlen, 0, n_seg)


def point_at_inserted_length(
    device_polyline, inserted_mm: float
):
    """RL_IMPROV_18 v3c (machine-2 reward pair) — 3-D point of a device tip
    that sits ``inserted_mm`` of arclength past the insertion end of a
    DISTAL-FIRST device polyline (tracking3d order: index 0 = leading tip,
    last index = insertion point).

    Undeployed nodes pile up as zero-length segments at the proximal end;
    they are dropped before measuring, so the walk runs over deployed
    geometry only. ``inserted_mm`` is clamped to [0, deployed length]: 0
    returns the insertion point, values past the deployed length return the
    leading tip. Returns None when fewer than 2 deployed points exist.
    """
    pts = np.asarray(device_polyline, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return None
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    keep = seg > 1e-9
    if not keep.any():
        return None
    # Keep point i when either adjacent segment is non-degenerate.
    keep_pts = np.concatenate([[keep[0]], keep[:-1] | keep[1:], [keep[-1]]]) \
        if len(seg) > 1 else np.array([True, True])
    pts = pts[keep_pts]
    if len(pts) < 2:
        return None
    # Arc from the PROXIMAL (insertion) end: reverse, then walk forward.
    rev = pts[::-1]
    cum = compute_cumulative_arclength(rev)
    target = float(np.clip(inserted_mm, 0.0, cum[-1]))
    idx = int(np.searchsorted(cum, target, side="right") - 1)
    if idx >= len(rev) - 1:
        return rev[-1].copy()
    seg_len = cum[idx + 1] - cum[idx]
    t = 0.0 if seg_len <= 1e-9 else (target - cum[idx]) / seg_len
    return rev[idx] + t * (rev[idx + 1] - rev[idx])
