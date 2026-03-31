"""Continuous arclength-based progress reward.

Projects the device tip onto the correct-path polyline and rewards
forward progress along the path. Unlike CenterlineWaypointProgress,
this cannot flip between branches — it only measures projection onto
the known-correct path from insertion to target.

Reward per step:
    r = progress_factor * (d_rem_prev - d_rem_curr)
      - lateral_penalty_factor * cross_track_dist

where d_rem = total_path_length - arclength_of_projection.
"""

import numpy as np

from .reward import Reward
from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_vessel_cs
from ..util.polyline import (
    compute_cumulative_arclength,
    project_onto_polyline,
)


class ArcLengthProgress(Reward):
    """Reward based on continuous progress along the fixed path polyline.

    Args:
        intervention: The intervention object (provides fluoroscopy/tracking).
        pathfinder: A FixedPathfinder instance with ``path_points_vessel_cs``.
        progress_factor: Scale factor for forward-progress reward.
            Each mm of forward progress yields ``progress_factor`` reward.
        lateral_penalty_factor: Scale factor for cross-track penalty.
            Each mm of distance from the polyline costs
            ``lateral_penalty_factor`` per step.
    """

    def __init__(
        self,
        intervention: Intervention,
        pathfinder,
        progress_factor: float = 0.01,
        lateral_penalty_factor: float = 0.001,
        path_context=None,
    ) -> None:
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.progress_factor = progress_factor
        self.lateral_penalty_factor = lateral_penalty_factor
        # ConfigHandler expects self.path_context to match __init__ param.
        # Store None for serialization; actual cache is in _path_context.
        self.path_context = None  # Serialized as None by ConfigHandler
        self._path_context = path_context  # Actual runtime cache

        # Set during reset
        self._polyline: np.ndarray = np.empty((0, 3))
        self._cumlen: np.ndarray = np.empty(0)
        self._total_length: float = 0.0
        self._prev_d_rem: float = 0.0

        self.reward = 0.0

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0

        if self._path_context is not None:
            # Refresh cache (idempotent — may already be reset by LocalGuidance)
            self._path_context.reset()
            self._total_length = self._path_context.total_length
            if self._total_length < 1e-6:
                self._prev_d_rem = 0.0
                return
            result = self._path_context.get_projection()
            self._prev_d_rem = self._total_length - result.s
            return

        # Fallback: compute independently (backward compat with env4)
        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(len(self._polyline))
            self._total_length = 0.0
            self._prev_d_rem = 0.0
            return

        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])

        tip_vessel_cs = self._get_tip_vessel_cs()
        result = project_onto_polyline(tip_vessel_cs, self._polyline, self._cumlen)
        self._prev_d_rem = self._total_length - result.s

    def step(self) -> None:
        if self._total_length < 1e-6:
            self.reward = 0.0
            return

        if self._path_context is not None:
            result = self._path_context.get_projection()
        else:
            tip_vessel_cs = self._get_tip_vessel_cs()
            result = project_onto_polyline(
                tip_vessel_cs, self._polyline, self._cumlen
            )

        d_rem_curr = self._total_length - result.s

        # Progress reward: positive when moving toward target
        r_progress = self.progress_factor * (self._prev_d_rem - d_rem_curr)

        # Lateral penalty: penalise straying from the path centerline
        r_lateral = -self.lateral_penalty_factor * result.cross_track_dist

        self.reward = r_progress + r_lateral
        self._prev_d_rem = d_rem_curr

    def _get_tip_vessel_cs(self) -> np.ndarray:
        """Get the guidewire tip position in vessel coordinate system."""
        fluoro = self.intervention.fluoroscopy
        tip_3d = fluoro.tracking3d[0]  # first tracked point = tip
        return tracking3d_to_vessel_cs(
            tip_3d, fluoro.image_rot_zx, fluoro.image_center
        )
