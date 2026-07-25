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
    point_at_inserted_length,
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
        # RL_IMPROV_18 v3c (machine-2 reward pair) — tip-average progress.
        # "avg" pays progress_factor * delta(w*s_gw + (1-w)*s_cath): a
        # parked guidewire halves the pay rate, advancing the TRAILING
        # device is paid, and the catheter-forward + gw-retract telescoping
        # gait nets ~0. Still a pure potential (s_eff is a state function).
        # Defaults = byte-identical legacy frontier signal.
        tip_mode: str = "frontier",
        avg_gw_weight: float = 0.5,
    ) -> None:
        if tip_mode not in ("frontier", "avg"):
            raise ValueError(f"tip_mode must be 'frontier'|'avg', got {tip_mode}")
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.progress_factor = progress_factor
        self.lateral_penalty_factor = lateral_penalty_factor
        self.tip_mode = tip_mode
        self.avg_gw_weight = float(avg_gw_weight)
        # avg-mode trackers (underscore => not serialized by ConfigHandler)
        self._prev_s_eff = None
        self._trail_prev_s = None
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
        # v3c avg mode — fresh trackers each episode; baseline computed
        # below AT RESET so step-1 motion is priced and a restore into a
        # retracted/coiled state re-baselines there (recovery nets +).
        self._prev_s_eff = None
        self._trail_prev_s = None

        if self._path_context is not None:
            # Refresh cache (idempotent — may already be reset by LocalGuidance)
            self._path_context.reset()
            self._total_length = self._path_context.total_length
            if self._total_length < 1e-6:
                self._prev_d_rem = 0.0
                return
            result = self._path_context.get_projection()
            self._prev_d_rem = self._total_length - result.s
            if self.tip_mode == "avg":
                # avg mode projects the trailing tip itself — needs the
                # polyline even when the path_context serves the frontier.
                self._polyline = self.pathfinder.path_points_vessel_cs
                self._cumlen = (
                    compute_cumulative_arclength(self._polyline)
                    if len(self._polyline) >= 2
                    else np.zeros(len(self._polyline))
                )
                self._prev_s_eff = self._effective_avg_arc(result.s)
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
        if self.tip_mode == "avg":
            self._prev_s_eff = self._effective_avg_arc(result.s)

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
            # RL_IMPROV_15 — 1x SYMMETRIC progress only. The former Plan v9
            # Change 4b "2x forward-only inside the target daughter" doubling
            # was NOT potential-based: forward paid 2x while backward paid 1x,
            # so every oscillation cycle banked +progress_factor*Δs. Under
            # relax_failure_truncations (episode runs the full 600 steps) a
            # wire merely dithering in the RCCA farmed ~3-4 return — ≈ a real
            # success — without ever threading the target. At a flat 1x the
            # per-step reward telescopes to progress_factor*(s_final -
            # s_initial): a round trip nets EXACTLY zero, and the episode sum
            # is bounded by the net arclength actually advanced (~2.0 over the
            # full path), earned only by genuinely progressing.
            r_progress = self.progress_factor * delta_s
            # v3c avg mode — override with the tip-average delta. Any
            # geometry failure (s_eff None) degrades to the legacy frontier
            # delta for that step, never a dropped reward.
            if self.tip_mode == "avg":
                s_eff = self._effective_avg_arc(result.s)
                if s_eff is not None:
                    if self._prev_s_eff is not None:
                        r_progress = self.progress_factor * (
                            s_eff - self._prev_s_eff
                        )
                    else:
                        r_progress = 0.0  # resync after a geometry-failure gap
                    self._prev_s_eff = s_eff
                else:
                    self._prev_s_eff = None  # legacy frontier delta this step
            # Reset off-arc baseline so a subsequent off-path transition
            # captures a fresh baseline from the new divergence point.
            self._prev_off_arc = 0.0
            # v3c BLOCKER NOTE (machine-2 adversarial review): _prev_s_eff
            # is updated ONLY here, inside the on-path branch. During
            # off-path steps it FREEZES — a rebaseline there creates a
            # farmable pump (retract trailing off-path free, re-advance
            # on-path paid: +0.125 per 13-step cycle, ~+5/episode). Frozen,
            # the rejoin step pays s_eff(rejoin) - s_eff(last on-path),
            # netting the excursion's trailing motion exactly once.
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

    def _effective_avg_arc(self, s_frontier: float):
        """Weighted-average planned-path arc of the two device tips.

        The LEADING device (larger inserted length) keeps the frontier
        projection ``s_frontier`` — identical to the legacy signal. The
        TRAILING tip is located on the combined device polyline at its own
        inserted length, projected onto the planned path, and clamped to
        [0, s_frontier] (a trailing tip cannot legitimately grade ahead of
        the frontier; projection noise in tight bends must not pay).
        Returns None when the geometry is unavailable (missing devices,
        degenerate polyline) — callers fall back to the frontier delta.
        """
        try:
            inserted = self.intervention.device_lengths_inserted
            if inserted is None or len(inserted) < 2:
                return None
            ins_gw = float(inserted[0])
            ins_cath = float(inserted[1])
            fluoro = self.intervention.fluoroscopy
            track = np.asarray(fluoro.tracking3d, dtype=float)
            trailing_ins = min(ins_gw, ins_cath)
            tip_3d = point_at_inserted_length(track, trailing_ins)
            if tip_3d is None:
                return None
            tip_vessel = tracking3d_to_vessel_cs(
                tip_3d, fluoro.image_rot_zx, fluoro.image_center
            )
            if len(self._polyline) < 2:
                return None
            proj = project_onto_polyline(
                tip_vessel,
                self._polyline,
                self._cumlen,
                prev_s=self._trail_prev_s,
                window_mm=30.0,
            )
            self._trail_prev_s = float(proj.s)
            s_trailing = float(np.clip(proj.s, 0.0, s_frontier))
            w = self.avg_gw_weight
            if ins_gw <= ins_cath:
                s_gw, s_cath = s_trailing, float(s_frontier)
            else:
                s_gw, s_cath = float(s_frontier), s_trailing
            return w * s_gw + (1.0 - w) * s_cath
        except Exception:
            return None

    def _get_tip_vessel_cs(self) -> np.ndarray:
        """Get the guidewire tip position in vessel coordinate system."""
        fluoro = self.intervention.fluoroscopy
        tip_3d = fluoro.tracking3d[0]  # first tracked point = tip
        return tracking3d_to_vessel_cs(
            tip_3d, fluoro.image_rot_zx, fluoro.image_center
        )
