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
        tip_mode: str = "frontier",
        avg_gw_weight: float = 0.5,
    ) -> None:
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.progress_factor = progress_factor
        self.lateral_penalty_factor = lateral_penalty_factor
        # ConfigHandler expects self.path_context to match __init__ param.
        # Store None for serialization; actual cache is in _path_context.
        self.path_context = None  # Serialized as None by ConfigHandler
        self._path_context = path_context  # Actual runtime cache
        # RL_IMPROV_16 (v3c) — "avg" pays Δ of the WEIGHTED-AVERAGE arc of
        # BOTH device tips instead of the frontier tip alone. A parked
        # guidewire then halves the progress rate, advancing the trailing
        # device is paid, and the measured cath-forward/gw-retract reward
        # tie at the choke breaks toward deploying the wire. Still a pure
        # potential (telescopes; round trips net zero). "frontier" is the
        # byte-identical legacy path.
        if tip_mode not in ("frontier", "avg"):
            raise ValueError(f"tip_mode must be 'frontier' or 'avg', got {tip_mode!r}")
        self.tip_mode = tip_mode
        self.avg_gw_weight = float(avg_gw_weight)
        self._prev_s_eff: float = None
        # Review MEDIUM fix — windowed-projection continuity anchor for the
        # TRAILING tip (the frontier already has one in path_context). A
        # bare full-scan projection is ambiguous at hairpins (siphon): the
        # anchor + 30mm window keeps consecutive trailing projections
        # continuous, with the standard fallback escape on genuine jumps.
        self._trail_prev_s: float = None

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
        self._prev_s_eff = None  # RL_IMPROV_16 (v3c) — avg-mode tracker

        if self._path_context is not None:
            # Refresh cache (idempotent — may already be reset by LocalGuidance)
            self._path_context.reset()
            self._total_length = self._path_context.total_length
            if self._total_length < 1e-6:
                self._prev_d_rem = 0.0
                return
            result = self._path_context.get_projection()
            self._prev_d_rem = self._total_length - result.s
            # RL_IMPROV_16 (v3c) — avg mode projects BOTH device tips itself, so it
            # needs the polyline/cumlen even when path_context serves the
            # frontier projection. The avg baseline is computed AT RESET
            # (review LOW fix) so step 1's motion is priced instead of
            # dropped — and a restore into a retracted/coiled state
            # re-baselines there, netting recovery positive.
            if self.tip_mode == "avg":
                self._polyline = self.pathfinder.path_points_vessel_cs
                self._cumlen = (
                    compute_cumulative_arclength(self._polyline)
                    if len(self._polyline) >= 2
                    else np.zeros(len(self._polyline))
                )
                self._trail_prev_s = None  # windowed-projection anchor
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
        # RL_IMPROV_16 (v3c) — avg-mode baseline at reset (fallback branch; see
        # the path_context branch above for rationale).
        if self.tip_mode == "avg":
            self._trail_prev_s = None
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
            # RL_IMPROV_16 (v3c) — tip-average mode: replace the frontier
            # delta with the delta of the weighted-average arc of BOTH
            # device tips. Falls back to the frontier delta (already in
            # r_progress) whenever the per-device geometry is unavailable,
            # so a transient accessor failure degrades to legacy behavior
            # instead of dropping the step's reward.
            if self.tip_mode == "avg":
                s_eff = self._effective_avg_arc(result.s)
                if s_eff is not None:
                    if self._prev_s_eff is not None:
                        r_progress = self.progress_factor * (
                            s_eff - self._prev_s_eff
                        )
                    else:
                        # First avg-mode step of the episode: no delta yet.
                        r_progress = 0.0
                    self._prev_s_eff = s_eff
                else:
                    self._prev_s_eff = None
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
        # RL_IMPROV_16 (v3c) review fix (BLOCKER) — the tracker FREEZES during
        # off-path steps (no rebaseline). Rebaselining forgave trailing-
        # device retraction while the frontier was off-classified and then
        # paid the re-advance in full: a farmable pump worth ~0.005/mm of
        # catheter cycled (proven by in-container counterexample — the same
        # exploit class as the removed RL_IMPROV_15 2x doubling). Frozen,
        # the rejoin step pays s_eff(rejoin) - s_eff(last on-path), netting
        # the excursion's trailing motion exactly once; the frontier
        # component ~cancels (its projection is pinned near the divergence
        # at exit and rejoin) and its off-path motion is priced by the
        # off-arc channel.

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
