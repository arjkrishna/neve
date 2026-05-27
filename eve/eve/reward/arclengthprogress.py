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
        # Plan v5 — per-step delta tracking of off-path arc since divergence.
        # Off-path reward = -progress_factor * (off_arc_now - prev_off_arc).
        # Going deeper off-path: Δ > 0 → r < 0 (penalty).
        # Retracting toward divergence point: Δ < 0 → r > 0 (reward,
        #   symmetric with on-path forward motion).
        self._prev_off_arc: float = 0.0

        self.reward = 0.0

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        self._prev_off_arc = 0.0  # Plan v5 — per-episode reset

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
            on_path = self._path_context.is_on_correct_path()
        else:
            tip_vessel_cs = self._get_tip_vessel_cs()
            result = project_onto_polyline(
                tip_vessel_cs, self._polyline, self._cumlen
            )
            on_path = True  # fallback (env4 compat); no on-path check

        d_rem_curr = self._total_length - result.s

        # Plan v5 — progress reward is on-path-aware AND symmetric:
        #   On-path: +progress_factor * forward Δs along planned polyline.
        #     (Backward motion on-path naturally goes negative via Δd_rem.)
        #   Off-path: -progress_factor * Δ(off_path_arc_since_divergence).
        #     Going deeper off-path → negative reward.
        #     Retracting toward divergence point → POSITIVE reward
        #     (symmetric to on-path forward motion; matches the heuristic's
        #     "retract when off-path" implicit objective).
        # _prev_d_rem is updated unconditionally so the next on-path step
        # has a correct delta when wire returns to path.
        if on_path:
            delta_s = self._prev_d_rem - d_rem_curr
            pf = self.progress_factor
            # Plan v9 Change 4b — double the progress reward when the
            # wire is inside the target daughter AND moving forward
            # (delta_s > 0). Replaces the originally-drafted +0.005/step
            # constant RCCA bonus, which would have rewarded mere dwell.
            # Doubling on forward motion only: deeper threading -> more
            # reward; freezing in shallow RCCA -> no reward; backward
            # motion still penalised at the standard 1x rate.
            try:
                pc = self._path_context
                if (delta_s > 0
                        and pc is not None
                        and pc._current_branch_idx is not None
                        and pc._target_daughter_branch_idx is not None
                        and int(pc._current_branch_idx)
                            == int(pc._target_daughter_branch_idx)):
                    pf *= 2.0
            except Exception:
                pass
            r_progress = pf * delta_s
            # Reset off-arc baseline so a subsequent off-path transition
            # captures a fresh baseline from the new divergence point.
            self._prev_off_arc = 0.0
        elif self._path_context is not None:
            off_arc_now = self._path_context.get_off_path_arc_since_divergence()
            r_progress = -self.progress_factor * (off_arc_now - self._prev_off_arc)
            self._prev_off_arc = off_arc_now
        else:
            r_progress = 0.0

        # Lateral penalty: penalise straying from the path centerline.
        # Plan v9 — radius-aware deadband. A few mm of cross-track is
        # geometrically unavoidable in a wide vessel (the wire rides the
        # wall, not the centerline polyline), so penalising raw
        # cross_track adds a large constant drag (~-3 over an episode at
        # ~5 mm mean offset) that makes even clean threads net negative.
        # Instead, only penalise the EXCESS beyond the local radius-aware
        # tolerance (max(MIN_TOLERANCE_MM, K_RADIUS*local_radius)) — the
        # same tolerance the state machine uses for on-path detection.
        # Wide trunk -> large tolerance -> ~0 penalty for normal
        # wall-hugging; narrow daughter -> small tolerance -> genuine
        # divergence still penalised.
        ct = result.cross_track_dist
        if self._path_context is not None:
            try:
                tol = self._path_context.get_local_tolerance()
            except Exception:
                tol = 0.0
            ct = max(0.0, ct - tol)
        r_lateral = -self.lateral_penalty_factor * ct

        self.reward = r_progress + r_lateral
        self._prev_d_rem = d_rem_curr

    def _get_tip_vessel_cs(self) -> np.ndarray:
        """Get the guidewire tip position in vessel coordinate system."""
        fluoro = self.intervention.fluoroscopy
        tip_3d = fluoro.tracking3d[0]  # first tracked point = tip
        return tracking3d_to_vessel_cs(
            tip_3d, fluoro.image_rot_zx, fluoro.image_center
        )
