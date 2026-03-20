"""Compact local guidance observation for path-aware navigation.

Provides an 8-dimensional observation vector encoding the agent's
relationship to the known correct path, replacing the much larger
Centerlines2D observation (154+ dims).

Features:
    0: d_rem_norm       - remaining arclength / total length  [0, 1]
    1: cross_track_dist - distance from tip to polyline (mm, clipped)
    2: tangent_x_2d     - path tangent x-component at projection  [-1, 1]
    3: tangent_z_2d     - path tangent z-component at projection  [-1, 1]
    4: heading_error    - angle between device direction and tangent  [-pi, pi]
    5: curvature_ahead  - max curvature in next 20mm of path
    6: dist_to_bifurc   - arclength to next branching point (mm, clipped)
    7: on_correct_branch - 1 if tip is on a path branch, 0 otherwise
"""

import numpy as np
import gymnasium as gym

from .observation import Observation
from ..intervention import Intervention
from ..intervention.vesseltree import find_nearest_branch_to_point
from ..util.coordtransform import tracking3d_to_vessel_cs
from ..util.polyline import (
    compute_cumulative_arclength,
    compute_segment_tangents,
    compute_curvature,
    project_onto_polyline,
)


# Clipping constants
_MAX_CROSS_TRACK_MM = 50.0
_MAX_BIFURC_DIST_MM = 200.0
_LOOKAHEAD_MM = 20.0


class LocalGuidance(Observation):
    """Compact 8-dim observation encoding the agent's state relative to the path.

    Args:
        intervention: The intervention object.
        pathfinder: A FixedPathfinder with ``path_points_vessel_cs``,
            ``path_branch_set``, and ``path_branching_points3d``.
        name: Name for this observation component.
    """

    def __init__(
        self,
        intervention: Intervention,
        pathfinder,
        name: str = "local_guidance",
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.pathfinder = pathfinder

        # Precomputed path data (set in reset)
        self._polyline: np.ndarray = np.empty((0, 3))
        self._cumlen: np.ndarray = np.empty(0)
        self._tangents: np.ndarray = np.empty((0, 3))
        self._curvature: np.ndarray = np.empty(0)
        self._total_length: float = 0.0
        self._bifurc_arclengths: np.ndarray = np.empty(0)

        self.obs = np.zeros(8, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        low = np.array(
            [0.0, 0.0, -1.0, -1.0, -np.pi, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, _MAX_CROSS_TRACK_MM, 1.0, 1.0, np.pi, 10.0, _MAX_BIFURC_DIST_MM, 1.0],
            dtype=np.float32,
        )
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(max(len(self._polyline), 1))
            self._tangents = np.empty((0, 3))
            self._curvature = np.empty(0)
            self._total_length = 0.0
            self._bifurc_arclengths = np.empty(0)
            self.obs = np.zeros(8, dtype=np.float32)
            return

        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])
        self._tangents = compute_segment_tangents(self._polyline)
        self._curvature = compute_curvature(self._tangents, self._cumlen)

        # Compute arclength positions of branching points along the path
        self._compute_bifurcation_arclengths()

        # Compute initial observation
        self.step()

    def step(self) -> None:
        if self._total_length < 1e-6:
            self.obs = np.zeros(8, dtype=np.float32)
            return

        fluoro = self.intervention.fluoroscopy

        # Tip position in vessel CS
        tip_3d = fluoro.tracking3d[0]
        tip_vessel = tracking3d_to_vessel_cs(
            tip_3d, fluoro.image_rot_zx, fluoro.image_center
        )

        # Project tip onto polyline
        proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)

        # Feature 0: remaining arclength (normalised)
        d_rem_norm = max(0.0, (self._total_length - proj.s) / self._total_length)

        # Feature 1: cross-track distance (clipped)
        cross_track = min(proj.cross_track_dist, _MAX_CROSS_TRACK_MM)

        # Features 2-3: path tangent in 2D at projection point
        seg_idx = min(proj.segment_idx, len(self._tangents) - 1)
        tangent_3d = self._tangents[seg_idx]  # (3,) in vessel CS
        # Convert tangent to 2D by dropping the y-component (same as tracking3d_to_2d)
        tangent_2d = np.array([tangent_3d[0], tangent_3d[2]])
        t2d_norm = np.linalg.norm(tangent_2d)
        if t2d_norm > 1e-8:
            tangent_2d = tangent_2d / t2d_norm
        else:
            tangent_2d = np.array([1.0, 0.0])

        # Feature 4: heading error
        heading_error = self._compute_heading_error(fluoro, tangent_3d)

        # Feature 5: max curvature in the next LOOKAHEAD_MM
        curvature_ahead = self._compute_curvature_ahead(proj.s)

        # Feature 6: distance to next bifurcation along path
        dist_to_bifurc = self._compute_dist_to_bifurcation(proj.s)

        # Feature 7: on correct branch?
        on_path = self._is_on_path_branch(tip_vessel)

        self.obs = np.array(
            [
                d_rem_norm,
                cross_track,
                tangent_2d[0],
                tangent_2d[1],
                heading_error,
                curvature_ahead,
                dist_to_bifurc,
                on_path,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_heading_error(self, fluoro, tangent_3d: np.ndarray) -> float:
        """Angle between device tip direction and path tangent."""
        tracking = fluoro.tracking3d
        if len(tracking) < 2:
            return 0.0

        # Device direction: from second tracked point toward tip
        device_dir = tracking[0] - tracking[1]
        # Convert to vessel CS direction (rotation only, no translation)
        # Since tracking3d_to_vessel_cs applies rotation + translation,
        # for a direction vector we convert both endpoints and subtract
        p0_v = tracking3d_to_vessel_cs(
            tracking[0], fluoro.image_rot_zx, fluoro.image_center
        )
        p1_v = tracking3d_to_vessel_cs(
            tracking[1], fluoro.image_rot_zx, fluoro.image_center
        )
        device_dir_v = p0_v - p1_v
        d_norm = np.linalg.norm(device_dir_v)
        if d_norm < 1e-8:
            return 0.0
        device_dir_v = device_dir_v / d_norm

        # Angle between device direction and path tangent
        dot = float(np.clip(np.dot(device_dir_v, tangent_3d), -1.0, 1.0))
        # Use atan2 of cross-product magnitude for signed angle (projected to 2D)
        cross = np.cross(device_dir_v, tangent_3d)
        # Use y-component of cross product as the "signed" part (looking down)
        sign = 1.0 if cross[1] >= 0 else -1.0
        return float(sign * np.arccos(dot))

    def _compute_curvature_ahead(self, s_current: float) -> float:
        """Max curvature in the next LOOKAHEAD_MM along the path."""
        if len(self._curvature) == 0:
            return 0.0

        s_end = s_current + _LOOKAHEAD_MM
        # Curvature[i] corresponds to interior vertex i+1 (arclength cumlen[i+1])
        # Find curvature values within [s_current, s_end]
        vertex_arclengths = self._cumlen[1:-1]  # interior vertices
        mask = (vertex_arclengths >= s_current) & (vertex_arclengths <= s_end)
        if not np.any(mask):
            return 0.0
        return float(np.max(self._curvature[mask]))

    def _compute_dist_to_bifurcation(self, s_current: float) -> float:
        """Distance along path to the next branching point ahead."""
        if len(self._bifurc_arclengths) == 0:
            return _MAX_BIFURC_DIST_MM

        ahead = self._bifurc_arclengths[self._bifurc_arclengths > s_current]
        if len(ahead) == 0:
            return _MAX_BIFURC_DIST_MM
        return min(float(ahead[0]) - s_current, _MAX_BIFURC_DIST_MM)

    def _is_on_path_branch(self, tip_vessel_cs: np.ndarray) -> float:
        """Check if the tip is on one of the path branches."""
        if not hasattr(self.pathfinder, "path_branch_set"):
            return 1.0
        branch = find_nearest_branch_to_point(
            tip_vessel_cs, self.intervention.vessel_tree
        )
        return 1.0 if branch in self.pathfinder.path_branch_set else 0.0

    def _compute_bifurcation_arclengths(self) -> None:
        """Find arclength positions of branching points along the path."""
        bp_3d = self.pathfinder.path_branching_points3d
        if bp_3d is None or len(bp_3d) == 0:
            self._bifurc_arclengths = np.empty(0)
            return

        fluoro = self.intervention.fluoroscopy
        arclengths = []
        for bp in bp_3d:
            bp_vessel = tracking3d_to_vessel_cs(
                bp, fluoro.image_rot_zx, fluoro.image_center
            )
            proj = project_onto_polyline(bp_vessel, self._polyline, self._cumlen)
            arclengths.append(proj.s)
        self._bifurc_arclengths = np.sort(arclengths)
