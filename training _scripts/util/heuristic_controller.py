"""Simple centerline-following heuristic controller.

Generates "okay" trajectories by:
  - Translating forward at a modest constant speed
  - Rotating to minimize cross-track error and align with the path tangent

Used to seed the replay buffer before SAC training starts, avoiding
the early "collapse to do-nothing" failure mode.

Usage:
    heuristic = CenterlineFollowerHeuristic(pathfinder, intervention)
    heuristic.reset()  # after env.reset()
    action = heuristic.get_action(rng)
"""

import numpy as np
from typing import Optional

from eve.util.coordtransform import tracking3d_to_vessel_cs
from eve.util.polyline import (
    compute_cumulative_arclength,
    compute_segment_tangents,
    project_onto_polyline,
)


class CenterlineFollowerHeuristic:
    """Heuristic controller that follows the correct-path centerline.

    Args:
        pathfinder: A FixedPathfinder (must already have been reset so that
            ``path_points_vessel_cs`` is populated).
        intervention: The intervention object (provides fluoroscopy).
        max_translation: Maximum forward translation speed (mm/s).
        heading_kp: Proportional gain for heading-error correction (tangent alignment).
        crosstrack_kp: Proportional gain for cross-track error correction
            (steers back toward the centerline when off-path).
        noise_std_frac: Gaussian noise as a fraction of the action magnitude.
        catheter_follow_ratio: Catheter translation = gw * this ratio.
    """

    def __init__(
        self,
        pathfinder,
        intervention,
        max_translation: float = 20.0,
        heading_kp: float = 1.0,
        crosstrack_kp: float = 0.05,
        noise_std_frac: float = 0.1,
        catheter_follow_ratio: float = 0.8,
    ) -> None:
        self.pathfinder = pathfinder
        self.intervention = intervention
        self.max_translation = max_translation
        self.heading_kp = heading_kp
        self.crosstrack_kp = crosstrack_kp
        self.noise_std_frac = noise_std_frac
        self.catheter_follow_ratio = catheter_follow_ratio

        # Precomputed after reset
        self._polyline = np.empty((0, 3))
        self._cumlen = np.empty(0)
        self._tangents = np.empty((0, 3))
        self._total_length = 0.0

        # Last computed diagnostics (for env-level logging)
        self._last_heading_error = 0.0
        self._last_cross_track = 0.0
        self._last_arc_past_junction = 0.0

        # Phase 1 step counter — caps Phase 1 at 30 commands per episode.
        # Previously used a distance-based cap (20 mm retracted), but the
        # wire physically stalls around 19 mm retraction (wedged against a
        # vessel bend), so the cap was never crossed and Phase 1 looped
        # forever. Counting commands decouples us from physical motion.
        self._phase1_steps = 0

    def reset(self) -> None:
        """Recompute path data from the pathfinder (call after env.reset)."""
        self._polyline = self.pathfinder.path_points_vessel_cs
        self._phase1_steps = 0
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(max(1, len(self._polyline)))
            self._tangents = np.empty((0, 3))
            self._total_length = 0.0
            return
        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])
        self._tangents = compute_segment_tangents(self._polyline)

    def get_action(
        self,
        rng: "Optional[np.random.Generator]" = None,
        force_translate: bool = False,
        d_corr_mm: float = float("inf"),
        arc_past_junction_mm: float = 0.0,
    ) -> np.ndarray:
        """Compute a heuristic action for the current state.

        Translation: proportional to remaining distance, capped at max_translation.
        Rotation: combination of (a) tangent alignment and (b) cross-track
        error correction — steers toward the centerline when off-path
        AND aligns with the path direction.

        Returns:
            (4,) action array: [gw_trans, gw_rot, cath_trans, cath_rot].
        """
        if rng is None:
            rng = np.random.default_rng()

        if self._total_length < 1e-6:
            return np.zeros(4, dtype=np.float32)

        fluoro = self.intervention.fluoroscopy
        tip_3d = fluoro.tracking3d[0]
        tip_vessel = tracking3d_to_vessel_cs(
            tip_3d, fluoro.image_rot_zx, fluoro.image_center
        )

        proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)

        d_rem = self._total_length - proj.s

        # ---- Compute geometric quantities ----
        seg_idx = min(proj.segment_idx, len(self._tangents) - 1)
        tangent = self._tangents[seg_idx]

        # Heading error: signed angle between the WIRE'S LOCAL DIRECTION
        # (device_dir) and the path tangent (RL_IMPROV_7 §7 Fix 18). The
        # earlier J-tip-curvature formula returned garbage in the trunk:
        # when the wire is well-aligned with the path, the path tangent's
        # perpendicular component to the wire axis is near zero, and the
        # angle between J-curvature direction and that near-zero vector
        # is essentially random — saturating heading_err near ±π. Empirically
        # 47% of run-26 trunk rotation steps improved heading_err and 52%
        # worsened it (pure noise) while the formula commanded ±1.5 rad/s
        # the whole time, wedging the wire at the arch top.
        # The new formula is `angle_between(device_dir, tangent)` signed
        # about the cross product's z-axis component, mirroring run 17's
        # property that |heading_err| → 0 when wire is aligned with path.
        heading_error = 0.0
        cross_track_signed = 0.0
        tracking = fluoro.tracking3d
        K = 5  # beam-node spacing for the wire's local long axis (~5-10 mm)
        if len(tracking) >= K + 1:
            p0_v = tracking3d_to_vessel_cs(
                tracking[0], fluoro.image_rot_zx, fluoro.image_center
            )
            pk_v = tracking3d_to_vessel_cs(
                tracking[K], fluoro.image_rot_zx, fluoro.image_center
            )
            device_dir_raw = p0_v - pk_v
            d_norm = np.linalg.norm(device_dir_raw)
            if d_norm > 1e-6:
                device_dir = device_dir_raw / d_norm
                cross_v = np.cross(device_dir, tangent)
                cos_a = float(np.dot(device_dir, tangent))
                sin_mag = float(np.linalg.norm(cross_v))
                # Sign convention: use cross_v[2] (world-z component) to
                # determine which way to rotate. Falls back to cross_v[1]
                # if z-component is degenerate (wire purely in horizontal
                # plane, very rare in this anatomy).
                if abs(cross_v[2]) > 1e-3:
                    sign_v = float(np.sign(cross_v[2]))
                else:
                    sign_v = float(np.sign(cross_v[1])) if abs(cross_v[1]) > 1e-3 else 1.0
                heading_error = float(np.arctan2(sin_mag * sign_v, cos_a))

                # Cross-track: signed lateral offset from centerline.
                # Use simple y-component of cross(tangent, offset_vec) — this
                # matches the original heuristic's lateral-correction signal
                # that worked in run 17. Magnitude scales with offset.
                offset_vec = tip_vessel - proj.proj_point
                cross_track_signed = float(np.cross(tangent, offset_vec)[2])

        # ---- Two-phase strategy (RL_IMPROV_7 §7 Fix 15) ----
        #   Phase 1 (alignment): |heading_err| > 1.2 rad — the J-tip is
        #     pointing badly wrong. PURE retract at -3 mm/s (no rotation) to
        #     pull the tip out of whatever vessel segment it is wedged
        #     against, so its torsional state can relax before we try to
        #     advance again. Capped at 20 mm of retraction per episode — if
        #     the tip is still misaligned after 20 mm pullback, we accept
        #     that and fall through to Phase 2 (advance with whatever pose).
        #     Earlier attempts rotated during retraction, but retraction
        #     itself changes the path tangent at the tip's projection and
        #     confounds the heading_err signal (atan2 wraparound, sign
        #     thrashing → net rotation cancels). Pure translation is the
        #     cleanest way to unload torsional stress.
        #   Phase 2 (advance): original continuous-blend heuristic —
        #     forward translation + rotation correction (±1.5 clamp), both
        #     simultaneously.
        # force_translate=True (fold brake >5) skips Phase 1.
        heading_abs = abs(heading_error)
        # Phase 1 threshold raised 1.2 → 2.0 (RL_IMPROV_7 §7 Fix 18b): with
        # the new angle_between(device_dir, tangent) heading_err formula,
        # values cluster between 1.0 and 1.5 in trunk/approach (smaller
        # than the old J-curvature formula which saturated at ±π). At the
        # old 1.2 threshold the wire alternated Phase 1 retract / Phase 2
        # advance every step, wasting motion. 2.0 means Phase 1 only fires
        # when the tip is very badly misaligned (≥115°) — actually pre-bent
        # in the wrong direction at the restore point.
        if (
            heading_abs > 2.0
            and self._phase1_steps < 30
            and not force_translate
        ):
            # Phase 1: pure retract, no rotation
            gw_trans = -3.0
            gw_rot = 0.0
            self._phase1_steps += 1
        else:
            # Phase 2: continuous blend of translation + rotation. THREE
            # regimes. RL_IMPROV_8: d_corr_mm and arc_past_junction_mm now
            # carry NEW semantics — d_corr_mm is 3D Euclidean distance to
            # the next daughter entry coord (not arclength projection),
            # and arc_past_junction_mm is arclength past the most-recent
            # DAUGHTER entry (not every junction). Same threshold values,
            # tighter physical meaning.
            #   NEAR daughter entry (d_corr_3d < 10mm): precision threading.
            #     gw_trans = 2 mm/s, gw_rot ∈ [-0.3, +0.3] with gain 0.6.
            #   INSIDE daughter past entry (arc_past_daughter > 10mm):
            #     gw_trans = min(5, 0.1*d_rem),
            #     gw_rot ∈ [-0.2, +0.2] with HIGHER crosstrack weight (0.2)
            #     so wire re-centers on daughter centerline.
            #   DEFAULT (trunk, between junctions, before bif1): full
            #     rotation authority for big course corrections.
            #     gw_trans = min(5, 0.1*d_rem),
            #     gw_rot ∈ [-1.5, +1.5] with crosstrack weight 0.05.
            if d_corr_mm < 10.0:
                gw_trans = 2.0
                gw_rot_raw = (
                    0.6 * heading_error
                    + self.crosstrack_kp * cross_track_signed
                )
                gw_rot = float(np.clip(gw_rot_raw, -0.3, 0.3))
            elif arc_past_junction_mm > 10.0:
                gw_trans = float(min(5.0, d_rem * 0.1))
                gw_rot_raw = (
                    self.heading_kp * heading_error
                    + 0.2 * cross_track_signed
                )
                gw_rot = float(np.clip(gw_rot_raw, -0.2, 0.2))
            else:
                gw_trans = float(min(5.0, d_rem * 0.1))
                gw_rot_raw = (
                    self.heading_kp * heading_error
                    + self.crosstrack_kp * cross_track_signed
                )
                gw_rot = float(np.clip(gw_rot_raw, -1.5, 1.5))

        # Expose diagnostics for env-level STEP logging
        self._last_heading_error = float(heading_error)
        self._last_cross_track = float(cross_track_signed)
        self._last_arc_past_junction = float(arc_past_junction_mm)

        # ---- Add noise for trajectory diversity ----
        gw_trans += rng.normal(0, abs(gw_trans) * self.noise_std_frac)
        gw_rot += rng.normal(0, max(abs(gw_rot), 0.1) * self.noise_std_frac)
        # No blanket "never retract" clamp — Phase 1 intentionally retracts.
        # Phase 2 produces gw_trans >= 1.0 before noise, so it stays positive
        # under the small noise_std_frac=0.1 perturbation.

        # ---- Catheter follows guidewire ----
        cath_trans = gw_trans * self.catheter_follow_ratio
        cath_rot = 0.0

        return np.array([gw_trans, gw_rot, cath_trans, cath_rot], dtype=np.float32)
