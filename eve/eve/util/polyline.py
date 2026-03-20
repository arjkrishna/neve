"""Polyline projection and geometry utilities.

Used by ArcLengthProgress reward and LocalGuidance observation to project
device tip positions onto the correct-path polyline.
"""

from typing import NamedTuple
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


def project_onto_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
    cumlen: np.ndarray,
) -> ProjectionResult:
    """Project a point onto a polyline via segment-wise closest-point.

    For each segment (p0, p1), computes the closest point on the segment
    to the query point, then returns the overall closest projection.

    Args:
        point: (3,) query point.
        polyline: (N, 3) polyline vertices.
        cumlen: (N,) cumulative arclength (from compute_cumulative_arclength).

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

    # Vectorized projection onto all segments at once
    p0 = polyline[:-1]  # (M, 3)
    p1 = polyline[1:]  # (M, 3)
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

    # Find the closest segment
    best_idx = int(np.argmin(dists))
    best_t = float(t_clamped[best_idx])
    best_proj = proj_points[best_idx]
    best_dist = float(dists[best_idx])

    # Arclength at the projection
    s = float(cumlen[best_idx] + best_t * (cumlen[best_idx + 1] - cumlen[best_idx]))

    return ProjectionResult(
        s=s,
        cross_track_dist=best_dist,
        proj_point=best_proj,
        segment_idx=best_idx,
        t=best_t,
    )
