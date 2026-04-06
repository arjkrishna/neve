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
_ON_PATH_THRESHOLD_MM = 5.0


class LocalGuidance(Observation):
    """Compact 8-dim observation encoding the agent's state relative to the path.

    Args:
        intervention: The intervention object.
        pathfinder: A FixedPathfinder with ``path_points_vessel_cs``,
            ``path_branch_set``, and ``path_branching_points3d``.
        name: Name for this observation component.
        path_context: Optional PathProjectionCache for sharing projection
            results with ArcLengthProgress reward.  When provided, avoids
            redundant polyline projection and coordinate transforms.
    """

    def __init__(
        self,
        intervention: Intervention,
        pathfinder,
        name: str = "local_guidance",
        path_context=None,
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.pathfinder = pathfinder
        # ConfigHandler expects self.path_context to match __init__ param.
        # Store None for serialization; actual cache is in _path_context.
        self.path_context = None  # Serialized as None by ConfigHandler
        self._path_context = path_context  # Actual runtime cache

        # Precomputed path data (set in reset)
        self._polyline: np.ndarray = np.empty((0, 3))
        self._cumlen: np.ndarray = np.empty(0)
        self._tangents: np.ndarray = np.empty((0, 3))
        self._tangents_2d: np.ndarray = np.empty((0, 2))
        self._curvature: np.ndarray = np.empty(0)
        self._total_length: float = 0.0
        self._bifurc_arclengths: np.ndarray = np.empty(0)

        self.obs = np.zeros(8, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        low = np.array(
            [0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        # Refresh shared cache so it picks up the new path from pathfinder
        # (pathfinder.reset() has already run by this point in eve.Env.reset)
        if self._path_context is not None:
            self._path_context.reset()

        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(max(len(self._polyline), 1))
            self._tangents = np.empty((0, 3))
            self._tangents_2d = np.empty((0, 2))
            self._curvature = np.empty(0)
            self._total_length = 0.0
            self._bifurc_arclengths = np.empty(0)
            self.obs = np.zeros(8, dtype=np.float32)
            return

        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])
        self._tangents = compute_segment_tangents(self._polyline)
        self._curvature = compute_curvature(self._tangents, self._cumlen)

        # Pre-compute 2D tangent projections (drop y-component, normalize)
        t2d = self._tangents[:, [0, 2]]
        norms = np.linalg.norm(t2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._tangents_2d = t2d / norms

        # Compute arclength positions of branching points along the path
        self._compute_bifurcation_arclengths()

        # Compute initial observation
        self.step()

    def step(self) -> None:
        if self._total_length < 1e-6:
            self.obs = np.zeros(8, dtype=np.float32)
            return

        fluoro = self.intervention.fluoroscopy

        # Tip position and projection — use cache if available
        if self._path_context is not None:
            tip_vessel = self._path_context.get_tip_vessel_cs()
            proj = self._path_context.get_projection()
        else:
            tip_3d = fluoro.tracking3d[0]
            tip_vessel = tracking3d_to_vessel_cs(
                tip_3d, fluoro.image_rot_zx, fluoro.image_center
            )
            proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)

        # Feature 0: remaining arclength (normalised)
        d_rem_norm = max(0.0, (self._total_length - proj.s) / self._total_length)

        # Feature 1: cross-track distance (clipped)
        cross_track = min(proj.cross_track_dist, _MAX_CROSS_TRACK_MM)

        # Features 2-3: path tangent in 2D (pre-computed at reset)
        seg_idx = min(proj.segment_idx, len(self._tangents) - 1)
        tangent_3d = self._tangents[seg_idx]
        if len(self._tangents_2d) > 0:
            tangent_2d = self._tangents_2d[seg_idx]
        else:
            tangent_2d = np.array([1.0, 0.0])

        # Feature 4: heading error (reuse cached tip_vessel for coord transform)
        heading_error = self._compute_heading_error(fluoro, tangent_3d, tip_vessel)

        # Feature 5: max curvature in the next LOOKAHEAD_MM
        curvature_ahead = self._compute_curvature_ahead(proj.s)

        # Feature 6: distance to next bifurcation along path
        dist_to_bifurc = self._compute_dist_to_bifurcation(proj.s)

        # Feature 7: on correct branch? (cheap threshold check on cross-track)
        on_path = 1.0 if proj.cross_track_dist < _ON_PATH_THRESHOLD_MM else 0.0

        self.obs = np.array(
            [
                d_rem_norm,                          # already [0, 1]
                cross_track / _MAX_CROSS_TRACK_MM,   # [0, 50] → [0, 1]
                tangent_2d[0],                        # already [-1, 1]
                tangent_2d[1],                        # already [-1, 1]
                heading_error / np.pi,               # [-π, π] → [-1, 1]
                curvature_ahead / 10.0,              # [0, ~10] → [0, ~1]
                dist_to_bifurc / _MAX_BIFURC_DIST_MM, # [0, 200] → [0, 1]
                on_path,                              # already {0, 1}
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_heading_error(
        self, fluoro, tangent_3d: np.ndarray, tip_vessel_cs: np.ndarray
    ) -> float:
        """Angle between device tip direction and path tangent.

        Args:
            fluoro: Fluoroscopy object.
            tangent_3d: Path tangent at projection point (vessel CS).
            tip_vessel_cs: Already-computed tip position in vessel CS
                (avoids redundant coordinate transform).
        """
        tracking = fluoro.tracking3d
        if len(tracking) < 2:
            return 0.0

        # Device direction in vessel CS: tip (already transformed) minus
        # second tracked point (needs one transform)
        p1_v = tracking3d_to_vessel_cs(
            tracking[1], fluoro.image_rot_zx, fluoro.image_center
        )
        device_dir_v = tip_vessel_cs - p1_v
        d_norm = np.linalg.norm(device_dir_v)
        if d_norm < 1e-8:
            return 0.0
        device_dir_v = device_dir_v / d_norm

        # Angle between device direction and path tangent
        dot = float(np.clip(np.dot(device_dir_v, tangent_3d), -1.0, 1.0))
        cross = np.cross(device_dir_v, tangent_3d)
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
