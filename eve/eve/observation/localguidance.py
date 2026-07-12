"""Compact local guidance observation for path-aware navigation.

Provides a 51-dimensional observation vector encoding the agent's
relationship to the known correct path, replacing the much larger
Centerlines2D observation (154+ dims). Gen-4 (mesh-invariance): every
feature is local / path-relative / topological — no absolute position,
no route-fraction index, no mesh-identity channel.

RL_IMPROV_9 cleanup: the old features 8-10 (wrong-branch entry distance
+ direction) were removed — they were permanently zero (wrong daughters
have no arclength on the planned path; the wrong-entry coords degenerated
to the tip). Features 11-30 renumbered down by 3 → 8-27.

Features:
    0:  d_rem_mm_norm       - remaining arclength in mm, clipped 400 /400.
                              (Gen-4: was arclength/total — a per-(mesh,
                              route) progress index; the same value meant
                              different mm on different meshes and
                              supported positional memorization.)  [0, 1]
    1:  cross_track_dist    - distance from tip to polyline (mm, clipped)  [0, 1]
    2:  tangent_x_2d        - path tangent x-component at projection  [-1, 1]
    3:  tangent_z_2d        - path tangent z-component at projection  [-1, 1]
    4:  heading_error       - angle between device direction and tangent  [-1, 1]
    5:  curvature_ahead     - max curvature in next 20mm of path  [0, ~1]
    6:  dist_to_bifurc      - arclength to next branching point (mm, clipped)  [0, 1]
    7:  on_correct_path     - 1 if the state machine classifies the tip as
                              on the planned path (the SAME signal the
                              reward gates on), 0 otherwise  {0, 1}
    8:  dist_correct_entry  - graph-routed distance to next correct daughter entry  [0, 1]
    9:  correct_entry_dir_x - x-component of unit vector toward next correct entry  [-1, 1]
    10: correct_entry_dir_z - z-component of unit vector toward next correct entry  [-1, 1]

    Plan v5 — observation enrichment for RCCA-only RL (§32 audit B1/B2/B3):
    11: arc_to_next_daughter_norm   - arclength to next DAUGHTER fork (mm, clipped 100, /100)  [0, 1]
    12: arc_past_last_daughter_norm - arclength past most-recent daughter fork (mm, clipped 100, /100)  [0, 1]
    Gen-4 — path preview, near pair (replaces the dead heuristic phase
    one-hots 13-16: constant "default" during RL, anatomy-bound tokens
    when a heuristic drove; freed for transferable geometry):
    13: preview_10_x         - (p(s+10mm) - tip) 2D image x, /(10+50)                        [-1, 1]
    14: preview_10_z         - (p(s+10mm) - tip) 2D image z, /(10+50)                        [-1, 1]
    15: preview_20_x         - (p(s+20mm) - tip) 2D image x, /(20+50)                        [-1, 1]
    16: preview_20_z         - (p(s+20mm) - tip) 2D image z, /(20+50)                        [-1, 1]
    17: bend_hat_x_2d        - J-tip bend direction, x-component in tracking3d 2D            [-1, 1]
    18: bend_hat_z_2d        - J-tip bend direction, z-component in tracking3d 2D            [-1, 1]

    Plan v5 (Tier 1) — Markov-completing features for step-wise RL:
    19: off_arc_since_divergence_norm - distance wire has drifted along
                                        wrong branch since divergence
                                        (mm, clipped 50, /50)                                [0, 1]
    20: off_branch_steps_norm         - consecutive off-path steps / 50
                                        (= fraction toward wrong_branch_timeout)             [0, 1]
    21: fold_stall_count_norm         - consecutive fold-stall steps / 20                    [0, 1]
    22: episode_step_norm             - episode_step / max_steps (=age toward MaxSteps)      [0, 1]
    23: forks_correct_norm            - n_correct_daughter_commits / 3                       [0, 1]
    24: forks_wrong_norm              - n_wrong_daughter_commits / 3                         [0, 1]
    25: is_in_trunk                   - 1 if state-machine _current_branch == trunk          {0, 1}
    26: is_on_target_daughter         - 1 if state-machine _current_branch == target daughter {0, 1}
    27: is_in_a_wrong_branch          - 1 if state-machine reports off-path (not in path set) {0, 1}

    Plan v13 — per-daughter anatomy enrichment (target-relative; the SAME
    physical branch means different things to different daughter graders):
    28: is_at_ostium                  - 1 if _current_branch == the LAST on-path bridge
                                        before the target daughter (the commit fork:
                                        (11) RCCA/RVA, (0) LCCA, (18) LVA)                   {0, 1}
    29: wrong_recovery_dist_norm      - OFF-PATH ONLY: routed retrace+reroute distance to
                                        rejoin the planned path toward the next correct
                                        daughter entry, clipped 200 mm / 200. 0 when
                                        on-path; grows with how DEEP the wire has committed
                                        into a wrong branch (small=recoverable, ~1=lost).
                                        Same wrong branch is "small" just after diverging,
                                        "large" once deep — recoverability is depth-aware,
                                        not a fixed per-branch label.                        [0, 1]
    (Categorical 25/28/26/27 = trunk / ostium / target / wrong are mutually
    exclusive; approach bridges that are NOT the ostium — e.g. (0) for an
    RCCA grader — fall through as all-zero.)

    Dual-device + fork-disambiguation features:
    30: cath_offset_x2d     - catheter tip minus guidewire tip, 2D image x (mm, /150)  [-1, 1]
    31: cath_offset_z2d     - catheter tip minus guidewire tip, 2D image z (mm, /150)  [-1, 1]
    32: gw_cath_gap_norm    - guidewire insertion minus catheter insertion
                              (mm, /150) — how far the wire extends beyond
                              the catheter tip                                         [-1, 1]
    33: fork_planned_dir_x  - planned takeoff direction at next junction, 2D x         [-1, 1]
    34: fork_planned_dir_z  - planned takeoff direction at next junction, 2D z         [-1, 1]
    35: fork_sister_dir_x   - sister takeoff direction at next junction, 2D x          [-1, 1]
    36: fork_sister_dir_z   - sister takeoff direction at next junction, 2D z          [-1, 1]
    37: heading_dot_planned - dot(device heading 2D, planned takeoff 2D)               [-1, 1]
    38: heading_dot_sister  - dot(device heading 2D, sister takeoff 2D)                [-1, 1]

    Gen-4 — path preview (far pair), buckle/stall block, corridor calibre:
    39: preview_40_x        - (p(s+40mm) - tip) 2D image x, /(40+50)                   [-1, 1]
    40: preview_40_z        - (p(s+40mm) - tip) 2D image z, /(40+50)                   [-1, 1]
    41: preview_80_x        - (p(s+80mm) - tip) 2D image x, /(80+50)                   [-1, 1]
    42: preview_80_z        - (p(s+80mm) - tip) 2D image z, /(80+50)                   [-1, 1]
    43: gw_slack_norm       - inserted_gw minus tip arclength (mm, /50) — wire
                              length stored in bowing, the integral of every
                              past fold (THE buckle scalar)                            [-1, 1]
    44: slip_norm           - last-step commanded-vs-achieved insertion
                              mismatch (delta_gw - delta_s, mm, /4): the raw,
                              continuous fold-detector signal (feature 21 is
                              its thresholded counter)                                 [-1, 1]
    45: gw_action_masked    - 1 if the guidewire translation command was
                              masked (below_zero / max_length / tree_end)              {0, 1}
    46: cath_action_masked  - 1 if the catheter translation command was masked         {0, 1}
    47: local_radius_norm   - radius-aware vessel calibre at the wire (mm /12)         [0, 1]
    48: radius_ahead_norm   - vessel calibre 20 mm ahead on the planned path
                              (mm /12) — narrowing-corridor preview                    [0, 1]
    49: clearance_norm      - cross_track / local_tolerance / 2 (0.5 = at the
                              radius-aware tolerance edge) — dimensionless,
                              transfers across vessel calibres                         [0, 1]

    Gen-4 #5 — log-scaled depth (recovery audit: the linear feature 0 moves
    0.0025/mm everywhere and rails at 400 mm; this channel is ~11x more
    sensitive per mm at 5 mm from the target — where the siphon endgame
    plays out — and stays unsaturated to 1000 mm of route):
    50: d_rem_log_norm      - log1p(remaining mm) / log1p(1000)                        [0, 1]
"""

import numpy as np
import gymnasium as gym

from .observation import Observation
from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_vessel_cs, vessel_cs_to_tracking3d
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
# Gen-4 mesh-invariance constants
_MAX_D_REM_MM = 400.0            # feature-0 remaining-arclength scale
_MAX_D_REM_LOG_MM = 1000.0       # feature-50 log-depth ceiling (unsaturated
                                 # over any plausible route length)
_PREVIEW_DELTAS_MM = (10.0, 20.0, 40.0, 80.0)  # path-preview offsets
_MAX_SLACK_MM = 50.0             # gw slack scale (feature 43)
_MAX_SLIP_MM = 4.0               # per-step slip scale (feature 44)
_MAX_LOCAL_RADIUS_MM = 12.0      # calibre scale (matches pathcontext
                                 # MAX_RADIUS_CEILING_MM)
_ON_PATH_THRESHOLD_MM = 5.0
# Plan v5 — clip for arc-to/past-daughter features
_MAX_DAUGHTER_ARC_MM = 100.0
# Plan v13 — clip for the off-path recovery-distance feature (routed
# retrace+reroute distance back to the planned path). 200 mm matches the
# bifurcation-distance scale; a wire deep in a wrong daughter routes ~100-150
# mm, so this gives a useful small→large spread without saturating early.
_MAX_RECOVERY_DIST_MM = 200.0
# Plan v5 (Tier 1) — clip for off-arc-since-divergence feature
_MAX_OFF_ARC_MM = 50.0
# Per-daughter env5 maxima for counter normalization (must match env5 constants)
_OFF_BRANCH_TIMEOUT_STEPS = 50.0    # env5.py OFF_BRANCH_GRACE_STEPS
_FOLD_STALL_TIMEOUT_STEPS = 20.0    # env5.py FOLD_STALL_STEPS
# Plan v12 — fork-count normalizer is now PER-GRADER (read at step time from
# the bound path_context's on-path junction count), NOT a fixed RCCA-route 3.0.
# _n_correct_commits counts one per on-path junction crossing, so the route's
# max is len(_path_branch_sequence_with_junctions). Daughters with a different
# on-path junction count (e.g. LVA via (18), RVA via (19)) were mis-scaled by
# the old constant. Fallback _DEFAULT below when the field is unavailable.
_DEFAULT_DAUGHTER_FORKS = 3.0


# NOTE: the heuristic phase one-hot helper that populated features 13-16
# was removed in the Gen-4 obs refactor — those slots are now the tip-
# relative path preview (10/20 mm), see _compute_path_preview.


def _compute_bend_hat_2d(fluoro) -> tuple:
    """J-tip bend direction projected to (x, z) image plane.

    Mirrors the heuristic's closed-loop rotation algorithm
    (_compute_rotation_to_target in heuristic_policy_*.py): uses the
    first 3 tracked points (tip, tip-1, tip-2) and computes the
    second-difference vector ``p0 + p2 - 2*p1`` projected perpendicular
    to the tangent. Returns (bx, bz) unit vector in tracking3d 2D, or
    (0, 0) if undefined.
    """
    tracking = fluoro.tracking3d
    if tracking is None or len(tracking) < 3:
        return 0.0, 0.0
    rzx = fluoro.image_rot_zx
    center = fluoro.image_center
    try:
        p0 = tracking3d_to_vessel_cs(tracking[0], rzx, center)
        p1 = tracking3d_to_vessel_cs(tracking[1], rzx, center)
        p2 = tracking3d_to_vessel_cs(tracking[2], rzx, center)
    except Exception:
        return 0.0, 0.0
    t_vec = p0 - p2
    t_norm = float(np.linalg.norm(t_vec))
    if t_norm < 1e-6:
        return 0.0, 0.0
    t_hat = t_vec / t_norm
    bend_raw = p0 + p2 - 2.0 * p1
    bend = bend_raw - float(np.dot(bend_raw, t_hat)) * t_hat
    b_norm = float(np.linalg.norm(bend))
    if b_norm < 1e-3:
        return 0.0, 0.0
    bend_hat = bend / b_norm
    # Project to tracking3d 2D (x, z) — same convention as path tangent_2d
    bend_t = vessel_cs_to_tracking3d(bend_hat, rzx, (0.0, 0.0, 0.0), None)
    bx, bz = float(bend_t[0]), float(bend_t[2])
    n = (bx * bx + bz * bz) ** 0.5
    if n < 1e-8:
        return 0.0, 0.0
    return bx / n, bz / n


def _entry_direction(
    tip_vessel: np.ndarray,
    entry_coords: np.ndarray,
    dist: float,
    image_rot_zx,
) -> tuple:
    """Return (dir_x, dir_z) unit vector from tip toward entry_coords in 2D.

    Converts the delta vector from vessel-CS to tracking3d (applying the C-arm
    rotation) before projecting to 2D, consistent with tracking3d_to_2d().
    Returns (0.0, 0.0) when dist is effectively infinite (no entry exists).
    """
    if dist >= _MAX_BIFURC_DIST_MM:
        return 0.0, 0.0
    # Rotate delta into tracking3d space; use zero center so translation cancels
    delta_vessel = entry_coords - tip_vessel
    delta_tracking = vessel_cs_to_tracking3d(delta_vessel, image_rot_zx, (0.0, 0.0, 0.0), None)
    dx, dz = float(delta_tracking[0]), float(delta_tracking[2])
    norm = (dx * dx + dz * dz) ** 0.5
    if norm < 1e-8:
        return 0.0, 0.0
    return dx / norm, dz / norm


class LocalGuidance(Observation):
    """Compact 51-dim observation encoding the agent's state relative to the path.

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

        self.obs = np.zeros(51, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        low = np.array(
            # 0    1     2     3     4    5    6    7    8     9     10
            [0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0,
             # 11  12    13    14    15    16   17    18
             0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             # 19  20   21   22   23   24   25   26   27   28   29
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             # 30    31    32    33    34    35    36    37    38
             -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             # 39    40    41    42    43    44   45   46   47   48   49
             -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             # 50 (d_rem_log_norm)
             0.0],
            dtype=np.float32,
        )
        high = np.ones(51, dtype=np.float32)
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
            self.obs = np.zeros(51, dtype=np.float32)
            return

        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])
        self._tangents = compute_segment_tangents(self._polyline)
        self._curvature = compute_curvature(self._tangents, self._cumlen)

        # Pre-compute 2D tangent projections: rotate into tracking3d space then
        # drop Y (axis 1), consistent with tracking3d_to_2d() used everywhere else.
        # Use zero image_center so the translation offset cancels for direction vectors.
        fluoro = self.intervention.fluoroscopy
        t_tracking = vessel_cs_to_tracking3d(
            self._tangents, fluoro.image_rot_zx, (0.0, 0.0, 0.0), None
        )
        t2d = t_tracking[:, [0, 2]]
        norms = np.linalg.norm(t2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self._tangents_2d = t2d / norms

        # Compute arclength positions of branching points along the path
        self._compute_bifurcation_arclengths()

        # Compute initial observation
        self.step()

    def step(self) -> None:
        if self._total_length < 1e-6:
            self.obs = np.zeros(51, dtype=np.float32)
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

        # Feature 0: remaining arclength in mm (clipped 400, /400). Gen-4:
        # honest physical scale — the old route-fraction was a per-(mesh,
        # target) progress index ("at f=0.4 turn left" is memorizable and
        # means different mm on different meshes).
        d_rem_norm = float(np.clip(
            (self._total_length - proj.s) / _MAX_D_REM_MM, 0.0, 1.0
        ))
        # Feature 50 (Gen-4 #5): log-scaled companion — ~11x the linear
        # channel's per-mm sensitivity at 5 mm remaining (the siphon
        # endgame), unsaturated to 1000 mm (feature 0 rails at 400).
        d_rem_log_norm = float(np.clip(
            np.log1p(max(self._total_length - proj.s, 0.0))
            / np.log1p(_MAX_D_REM_LOG_MM), 0.0, 1.0
        ))

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

        # Feature 7: on correct path? (state-machine signal — the SAME
        # classifier the reward gates on). The state machine and the
        # debounced nearest-branch classifier disagree by construction near
        # junctions; the policy must see the signal the reward keys on
        # (feature 27 is its negation for the wrong-branch case — redundancy
        # is acceptable, inconsistency was not).
        if self._path_context is not None:
            on_path = 1.0 if self._path_context.is_on_correct_path() else 0.0
        else:
            from ..intervention.vesseltree import find_nearest_branch_to_point
            nearest = find_nearest_branch_to_point(tip_vessel, self.intervention.vessel_tree)
            on_path = 1.0 if self.pathfinder.is_branch_on_path(nearest) else 0.0

        # Features 8-10: graph-routed distance + direction to the next
        # correct-path daughter entry. RL_IMPROV_9 cleanup removed the old
        # features 8-10 (wrong-branch entry distance + direction) — they
        # were permanently zero. RL_IMPROV_8 v2: routed d_corr accounts for
        # sister-branch detours; reports the real centerline-traversal
        # distance the wire must retrace to reach the next daughter.
        if self._path_context is not None:
            d_correct = self._path_context.get_routed_d_corr_to_next_daughter_entry()
            correct_coords = self._path_context.get_closest_correct_entry_coords()
        else:
            d_correct = _MAX_BIFURC_DIST_MM
            correct_coords = tip_vessel.copy()

        rot_zx = fluoro.image_rot_zx
        correct_dir_x, correct_dir_z = _entry_direction(
            tip_vessel, correct_coords, d_correct, rot_zx
        )

        # ----------------------------------------------------------------
        # Plan v5 — observation enrichment (features 11-18)
        # ----------------------------------------------------------------
        # Features 11-12: arclength to / past daughter forks (clipped, normalized)
        arc_to_next = float("inf")
        arc_past_last = 0.0
        if self._path_context is not None:
            try:
                arc_to_next = self._path_context.get_arclength_to_next_daughter_entry()
            except Exception:
                pass
            try:
                arc_past_last = self._path_context.get_arclength_past_last_daughter_entry()
            except Exception:
                pass
        # ∞ → 1.0; finite → clip to MAX and normalize.
        if arc_to_next == float("inf"):
            arc_to_next_norm = 1.0
        else:
            arc_to_next_norm = float(min(arc_to_next, _MAX_DAUGHTER_ARC_MM) / _MAX_DAUGHTER_ARC_MM)
        arc_past_norm = float(min(arc_past_last, _MAX_DAUGHTER_ARC_MM) / _MAX_DAUGHTER_ARC_MM)

        # Features 13-16 + 39-42: multi-point path preview. Planned-path
        # points at fixed arclength offsets ahead of the projection,
        # expressed as tip-relative deltas in the 2D image frame and
        # divided by (delta + 50). Bounded by construction: chord <= arc
        # and the tip is within cross_track of p(s), so
        # |p(s+d) - tip|/(d+50) stays near [-1,1]; clip defensively.
        # This replaces "I recognize where I am, so I know the bend is
        # coming" (absolute-position memorization) with the transferable
        # "the path bends left in 20 mm". Beyond the path end the preview
        # saturates at the final point.
        preview = self._compute_path_preview(proj.s, tip_vessel, rot_zx)

        # Features 17-18: J-tip bend direction in tracking 2D.
        bend_x, bend_z = _compute_bend_hat_2d(fluoro)

        # ----------------------------------------------------------------
        # Plan v5 (Tier 1) — Markov-completing features (19-27)
        # ----------------------------------------------------------------
        # Feature 19: off-arc since divergence (mm clipped 50, /50).
        off_arc_norm = 0.0
        if self._path_context is not None:
            try:
                off_arc = self._path_context.get_off_path_arc_since_divergence()
                off_arc_norm = float(min(max(off_arc, 0.0), _MAX_OFF_ARC_MM) / _MAX_OFF_ARC_MM)
            except Exception:
                pass

        # Features 20-22: env-level counters mirrored by env5.step.
        off_steps = float(getattr(self.intervention, "_env_off_branch_steps", 0))
        fold_steps = float(getattr(self.intervention, "_env_fold_stall_count", 0))
        ep_step = float(getattr(self.intervention, "_env_episode_step", 0))
        ep_max = float(getattr(self.intervention, "_env_max_steps", 600))
        off_branch_norm = float(min(off_steps / _OFF_BRANCH_TIMEOUT_STEPS, 1.0))
        fold_stall_norm = float(min(fold_steps / _FOLD_STALL_TIMEOUT_STEPS, 1.0))
        episode_step_norm = float(min(ep_step / max(ep_max, 1.0), 1.0))

        # Features 23-24: per-episode daughter-fork commit counters.
        n_correct = 0
        n_wrong = 0
        denom = _DEFAULT_DAUGHTER_FORKS
        if self._path_context is not None:
            n_correct = int(getattr(self._path_context, "_n_correct_commits", 0))
            n_wrong = int(getattr(self._path_context, "_n_wrong_commits", 0))
            # Plan v12 — per-grader denominator = THIS route's on-path junction
            # count (one commit per on-path junction crossing).
            seq = getattr(
                self._path_context, "_path_branch_sequence_with_junctions", None
            )
            if seq:
                denom = float(max(1, len(seq)))
        forks_correct_norm = float(min(n_correct / denom, 1.0))
        forks_wrong_norm = float(min(n_wrong / denom, 1.0))

        # Features 25-28: 4-dim branch categorical (mutually exclusive),
        # all TARGET-RELATIVE (computed from THIS grader's planned path).
        # is_in_trunk: current branch == cached trunk branch idx
        # is_at_ostium: current branch == last on-path bridge before target
        #   (the commit fork: (11) RCCA/RVA, (0) LCCA, (18) LVA)
        # is_on_target_daughter: current branch == cached target daughter idx
        # is_in_a_wrong_branch: state machine reports off-path
        # (Approach bridges that are NOT the ostium — e.g. (0) for an RCCA
        #  grader — fall through as all-zero.)
        # Feature 29 (wrong_recovery_dist_norm): OFF-PATH ONLY routed
        # retrace+reroute distance to rejoin the planned path toward the next
        # correct daughter entry. Reuses the already-computed d_correct
        # (= get_routed_d_corr_to_next_daughter_entry()) so it costs nothing
        # extra; 0 when on-path; grows with how deep the wire has committed
        # into a wrong branch (depth-aware: small=recoverable, ~1=lost).
        is_in_trunk = 0.0
        is_at_ostium = 0.0
        is_on_target_daughter = 0.0
        is_in_a_wrong_branch = 0.0
        wrong_recovery_dist_norm = 0.0
        if self._path_context is not None:
            cur_idx = self._path_context._current_branch_idx
            trunk_idx = getattr(self._path_context, "_trunk_branch_idx", None)
            target_idx = getattr(self._path_context, "_target_daughter_branch_idx", None)
            ostium_idx = getattr(self._path_context, "_ostium_branch_idx", None)
            on_planned = bool(getattr(self._path_context, "_on_planned_path", True))
            if not on_planned:
                is_in_a_wrong_branch = 1.0
                rec = d_correct if np.isfinite(d_correct) else _MAX_RECOVERY_DIST_MM
                wrong_recovery_dist_norm = float(
                    min(rec, _MAX_RECOVERY_DIST_MM) / _MAX_RECOVERY_DIST_MM
                )
            elif cur_idx is not None and trunk_idx is not None and cur_idx == trunk_idx:
                is_in_trunk = 1.0
            elif cur_idx is not None and ostium_idx is not None and cur_idx == ostium_idx:
                is_at_ostium = 1.0
            elif cur_idx is not None and target_idx is not None and cur_idx == target_idx:
                is_on_target_daughter = 1.0

        # ----------------------------------------------------------------
        # Dual-device + fork-disambiguation features (30-38)
        # ----------------------------------------------------------------
        # Features 30-31: catheter-tip minus guidewire-tip in the 2D image
        # frame (mm, /150, clipped). device_trackings3d splits the combined
        # tracking into per-device polylines; device order matches
        # intervention.devices (0 = guidewire, 1 = catheter) and row 0 of
        # each polyline is that device's tip. Trackings are already in
        # tracking3d space, so 2D = drop the Y axis (axis 1), the same
        # tracking3d_to_2d convention as the tangent features.
        cath_off_x = 0.0
        cath_off_z = 0.0
        try:
            dev_tracks = fluoro.device_trackings3d
            if (
                len(dev_tracks) >= 2
                and len(dev_tracks[0]) > 0
                and len(dev_tracks[1]) > 0
            ):
                cath_offset = np.asarray(
                    dev_tracks[1][0], dtype=np.float64
                ) - np.asarray(dev_tracks[0][0], dtype=np.float64)
                cath_off_x = float(np.clip(cath_offset[0] / 150.0, -1.0, 1.0))
                cath_off_z = float(np.clip(cath_offset[2] / 150.0, -1.0, 1.0))
        except Exception:
            cath_off_x = 0.0
            cath_off_z = 0.0

        # Feature 32: guidewire extension beyond the catheter tip (inserted
        # gw minus inserted cath, mm, /150, clipped) — the two-device
        # coordination variable.
        gw_cath_gap_norm = 0.0
        try:
            inserted = self.intervention.device_lengths_inserted
            if inserted is not None and len(inserted) >= 2:
                gw_cath_gap_norm = float(np.clip(
                    (float(inserted[0]) - float(inserted[1])) / 150.0, -1.0, 1.0
                ))
        except Exception:
            gw_cath_gap_norm = 0.0

        # Features 33-38: next-junction fork disambiguation. The planned /
        # sister takeoff directions are 3D vessel-CS unit vectors from the
        # path context; each is projected to the 2D image frame through
        # _entry_direction with a zero tip and dist 0.0 (a pure vessel-CS ->
        # tracking3d rotation + Y-drop + renormalize — the same transform as
        # the path-tangent / entry-direction features). The device heading
        # uses the same two tracking points as the heading-error feature 4
        # (tip and tip-1); the rotation cancels in the difference, so it is
        # computed directly in tracking3d space and Y-dropped. Missing
        # accessor / no junction ahead -> all six are 0.0.
        planned_x = 0.0
        planned_z = 0.0
        sister_x = 0.0
        sister_z = 0.0
        dot_planned = 0.0
        dot_sister = 0.0
        if self._path_context is not None and hasattr(
            self._path_context, "get_next_junction_fork_geometry"
        ):
            try:
                geo = self._path_context.get_next_junction_fork_geometry()
            except Exception:
                geo = None
            if geo is not None:
                try:
                    origin = np.zeros(3)
                    planned_dir = geo.get("planned_dir")
                    if planned_dir is not None:
                        planned_x, planned_z = _entry_direction(
                            origin,
                            np.asarray(planned_dir, dtype=np.float64),
                            0.0,
                            rot_zx,
                        )
                    sister_dir = geo.get("sister_dir")
                    if sister_dir is not None:
                        sister_x, sister_z = _entry_direction(
                            origin,
                            np.asarray(sister_dir, dtype=np.float64),
                            0.0,
                            rot_zx,
                        )
                    tracking = fluoro.tracking3d
                    if tracking is not None and len(tracking) >= 2:
                        heading = np.asarray(
                            tracking[0], dtype=np.float64
                        ) - np.asarray(tracking[1], dtype=np.float64)
                        hx, hz = float(heading[0]), float(heading[2])
                        h_norm = (hx * hx + hz * hz) ** 0.5
                        if h_norm >= 1e-8:
                            hx, hz = hx / h_norm, hz / h_norm
                            dot_planned = hx * planned_x + hz * planned_z
                            dot_sister = hx * sister_x + hz * sister_z
                except Exception:
                    planned_x = planned_z = sister_x = sister_z = 0.0
                    dot_planned = dot_sister = 0.0
        # Unit-vector components and unit-vector dots are in [-1, 1] by
        # construction; clip defensively against float drift.
        planned_x = float(np.clip(planned_x, -1.0, 1.0))
        planned_z = float(np.clip(planned_z, -1.0, 1.0))
        sister_x = float(np.clip(sister_x, -1.0, 1.0))
        sister_z = float(np.clip(sister_z, -1.0, 1.0))
        dot_planned = float(np.clip(dot_planned, -1.0, 1.0))
        dot_sister = float(np.clip(dot_sister, -1.0, 1.0))

        # ----------------------------------------------------------------
        # Gen-4 — buckle/stall block + corridor calibre (features 43-49)
        # ----------------------------------------------------------------
        # Feature 43: guidewire slack = inserted length minus tip
        # arclength along the planned path — wire length stored in bowing.
        # Mesh-relative by construction (both operands live on this
        # episode's path). Slightly negative is possible (the wire can cut
        # corners the centerline doesn't).
        gw_slack_norm = 0.0
        try:
            inserted = self.intervention.device_lengths_inserted
            if inserted is not None and len(inserted) >= 1:
                gw_slack_norm = float(np.clip(
                    (float(inserted[0]) - proj.s) / _MAX_SLACK_MM, -1.0, 1.0
                ))
        except Exception:
            gw_slack_norm = 0.0

        # Feature 44: slip — last step's commanded-vs-achieved insertion
        # mismatch (delta_gw - delta_s, mm), mirrored by env5 alongside
        # the counters. The raw continuous signal behind the fold-stall
        # detector; the policy sees it at step 1, not after the counter
        # has ratcheted.
        slip_norm = float(np.clip(
            float(getattr(self.intervention, "_env_slip_mm", 0.0))
            / _MAX_SLIP_MM, -1.0, 1.0,
        ))

        # Features 45-46: translation-command mask flags. LastAction only
        # carries the pre-mask command — without these the policy cannot
        # tell its push was zeroed at the tree end / insertion floor.
        gw_masked = 0.0
        cath_masked = 0.0
        try:
            cmd = self.intervention.last_cmd_action
            exe = self.intervention.last_exec_action
            if cmd is not None and exe is not None:
                if abs(float(cmd[0][0]) - float(exe[0][0])) > 1e-9:
                    gw_masked = 1.0
                if len(cmd) > 1 and abs(
                    float(cmd[1][0]) - float(exe[1][0])
                ) > 1e-9:
                    cath_masked = 1.0
        except Exception:
            gw_masked = cath_masked = 0.0

        # Features 47-49: vessel calibre now / 20mm ahead + dimensionless
        # clearance margin (cross_track / radius-aware tolerance; 0.5 = at
        # the tolerance edge). Calibre-relative clearance transfers across
        # vessel sizes far better than raw mm cross-track (feature 1).
        radius_now_norm = 0.0
        radius_ahead_norm = 0.0
        clearance_norm = 0.0
        if self._path_context is not None:
            try:
                radius_now_norm = float(np.clip(
                    self._path_context.get_local_radius()
                    / _MAX_LOCAL_RADIUS_MM, 0.0, 1.0,
                ))
                radius_ahead_norm = float(np.clip(
                    self._path_context.get_local_radius_at_arclength(
                        proj.s + _LOOKAHEAD_MM
                    ) / _MAX_LOCAL_RADIUS_MM, 0.0, 1.0,
                ))
                tol = self._path_context.get_local_tolerance()
                if tol > 1e-6:
                    clearance_norm = float(np.clip(
                        proj.cross_track_dist / tol / 2.0, 0.0, 1.0
                    ))
            except Exception:
                radius_now_norm = radius_ahead_norm = clearance_norm = 0.0

        self.obs = np.array(
            [
                d_rem_norm,                                              # 0  [0,1]
                cross_track / _MAX_CROSS_TRACK_MM,                       # 1  [0,1]
                tangent_2d[0],                                           # 2  [-1,1]
                tangent_2d[1],                                           # 3  [-1,1]
                heading_error / np.pi,                                   # 4  [-1,1]
                curvature_ahead / 10.0,                                  # 5  [0,~1]
                dist_to_bifurc / _MAX_BIFURC_DIST_MM,                    # 6  [0,1]
                on_path,                                                 # 7  {0,1}
                min(d_correct, _MAX_BIFURC_DIST_MM) / _MAX_BIFURC_DIST_MM,# 8  [0,1] dist_correct_entry
                correct_dir_x,                                           # 9  [-1,1]
                correct_dir_z,                                           # 10 [-1,1]
                arc_to_next_norm,                                        # 11 [0,1]
                arc_past_norm,                                           # 12 [0,1]
                preview[0],                                              # 13 [-1,1] preview 10mm x
                preview[1],                                              # 14 [-1,1] preview 10mm z
                preview[2],                                              # 15 [-1,1] preview 20mm x
                preview[3],                                              # 16 [-1,1] preview 20mm z
                bend_x,                                                  # 17 [-1,1] bend_hat 2D x
                bend_z,                                                  # 18 [-1,1] bend_hat 2D z
                off_arc_norm,                                            # 19 [0,1] off_arc_since_divergence
                off_branch_norm,                                         # 20 [0,1] off_branch_steps_norm
                fold_stall_norm,                                         # 21 [0,1] fold_stall_count_norm
                episode_step_norm,                                       # 22 [0,1] episode_step_norm
                forks_correct_norm,                                      # 23 [0,1] forks_correct_norm
                forks_wrong_norm,                                        # 24 [0,1] forks_wrong_norm
                is_in_trunk,                                             # 25 {0,1} is_in_trunk
                is_on_target_daughter,                                   # 26 {0,1} is_on_target_daughter
                is_in_a_wrong_branch,                                    # 27 {0,1} is_in_a_wrong_branch
                is_at_ostium,                                            # 28 {0,1} is_at_ostium
                wrong_recovery_dist_norm,                                # 29 [0,1] wrong_recovery_dist_norm
                cath_off_x,                                              # 30 [-1,1] cath_offset_x2d
                cath_off_z,                                              # 31 [-1,1] cath_offset_z2d
                gw_cath_gap_norm,                                        # 32 [-1,1] gw_cath_gap_norm
                planned_x,                                               # 33 [-1,1] fork planned takeoff 2D x
                planned_z,                                               # 34 [-1,1] fork planned takeoff 2D z
                sister_x,                                                # 35 [-1,1] fork sister takeoff 2D x
                sister_z,                                                # 36 [-1,1] fork sister takeoff 2D z
                dot_planned,                                             # 37 [-1,1] heading_dot_planned
                dot_sister,                                              # 38 [-1,1] heading_dot_sister
                preview[4],                                              # 39 [-1,1] preview 40mm x
                preview[5],                                              # 40 [-1,1] preview 40mm z
                preview[6],                                              # 41 [-1,1] preview 80mm x
                preview[7],                                              # 42 [-1,1] preview 80mm z
                gw_slack_norm,                                           # 43 [-1,1] gw_slack_norm
                slip_norm,                                               # 44 [-1,1] slip_norm
                gw_masked,                                               # 45 {0,1} gw_action_masked
                cath_masked,                                             # 46 {0,1} cath_action_masked
                radius_now_norm,                                         # 47 [0,1] local_radius_norm
                radius_ahead_norm,                                       # 48 [0,1] radius_ahead_norm
                clearance_norm,                                          # 49 [0,1] clearance_norm
                d_rem_log_norm,                                          # 50 [0,1] d_rem_log_norm
            ],
            dtype=np.float32,
        )

    def _compute_path_preview(
        self, s: float, tip_vessel: np.ndarray, rot_zx
    ) -> np.ndarray:
        """Tip-relative planned-path preview points (features 13-16, 39-42).

        For each delta in _PREVIEW_DELTAS_MM: interpolate the planned
        polyline at arclength s+delta (saturating at the path end), form
        the vessel-CS delta from the tip, rotate to tracking3d (zero
        image-center so translation cancels), drop Y, and scale by
        1/(delta+50). Returns an 8-vector clipped to [-1, 1]; zeros on a
        degenerate path.
        """
        out = np.zeros(2 * len(_PREVIEW_DELTAS_MM), dtype=np.float32)
        if len(self._polyline) < 2 or len(self._cumlen) < 2:
            return out
        for k, delta in enumerate(_PREVIEW_DELTAS_MM):
            s_query = float(np.clip(s + delta, 0.0, self._total_length))
            idx = int(np.searchsorted(self._cumlen, s_query) - 1)
            idx = max(0, min(idx, len(self._polyline) - 2))
            seg_len = self._cumlen[idx + 1] - self._cumlen[idx]
            t = 0.0 if seg_len < 1e-9 else float(
                (s_query - self._cumlen[idx]) / seg_len
            )
            point = self._polyline[idx] + t * (
                self._polyline[idx + 1] - self._polyline[idx]
            )
            delta_vessel = point - tip_vessel
            delta_tracking = vessel_cs_to_tracking3d(
                delta_vessel, rot_zx, (0.0, 0.0, 0.0), None
            )
            scale = delta + 50.0
            out[2 * k] = np.clip(delta_tracking[0] / scale, -1.0, 1.0)
            out[2 * k + 1] = np.clip(delta_tracking[2] / scale, -1.0, 1.0)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_heading_error(
        self, fluoro, tangent_3d: np.ndarray, tip_vessel_cs: np.ndarray
    ) -> float:
        """Angle between device tip direction and path tangent in tracking3d space.

        Both vectors are rotated into tracking3d space before computing the angle
        so the sign (determined by the Y-axis cross product) corresponds to the
        image-plane up direction, consistent with the 2D tangent projection.

        Args:
            fluoro: Fluoroscopy object.
            tangent_3d: Path tangent at projection point (vessel CS).
            tip_vessel_cs: Already-computed tip position in vessel CS
                (avoids redundant coordinate transform).
        """
        tracking = fluoro.tracking3d
        if len(tracking) < 2:
            return 0.0

        # Device direction in vessel CS: tip minus second tracked point
        p1_v = tracking3d_to_vessel_cs(
            tracking[1], fluoro.image_rot_zx, fluoro.image_center
        )
        device_dir_v = tip_vessel_cs - p1_v
        d_norm = np.linalg.norm(device_dir_v)
        if d_norm < 1e-8:
            return 0.0
        device_dir_v = device_dir_v / d_norm

        # Rotate both vectors into tracking3d space so the sign uses the
        # image-plane Y-axis (axis 1 in tracking3d), not the vessel-CS Y-axis.
        rot_zx = fluoro.image_rot_zx
        device_dir_t = vessel_cs_to_tracking3d(device_dir_v, rot_zx, (0.0, 0.0, 0.0), None)
        tangent_t = vessel_cs_to_tracking3d(tangent_3d, rot_zx, (0.0, 0.0, 0.0), None)

        dot = float(np.clip(np.dot(device_dir_t, tangent_t), -1.0, 1.0))
        cross = np.cross(device_dir_t, tangent_t)
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
