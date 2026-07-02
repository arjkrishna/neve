"""Compact local guidance observation for path-aware navigation.

Provides a 48-dimensional observation vector encoding the agent's
relationship to the known correct path, replacing the much larger
Centerlines2D observation (154+ dims).

RL_IMPROV_9 cleanup: the old features 8-10 (wrong-branch entry distance
+ direction) were removed — they were permanently zero (wrong daughters
have no arclength on the planned path; the wrong-entry coords degenerated
to the tip). Features 11-30 renumbered down by 3 → 8-27.

Features:
    0:  d_rem_norm          - remaining arclength / total length  [0, 1]
    1:  cross_track_dist    - distance from tip to polyline (mm, clipped)  [0, 1]
    2:  tangent_x_2d        - path tangent x-component at projection  [-1, 1]
    3:  tangent_z_2d        - path tangent z-component at projection  [-1, 1]
    4:  heading_error       - angle between device direction and tangent  [-1, 1]
    5:  curvature_ahead     - max curvature in next 20mm of path  [0, ~1]
    6:  dist_to_bifurc      - arclength to next branching point (mm, clipped)  [0, 1]
    7:  on_correct_branch   - 1 if tip is on a path branch, 0 otherwise  {0, 1}
    8:  dist_correct_entry  - graph-routed distance to next correct daughter entry  [0, 1]
    9:  correct_entry_dir_x - x-component of unit vector toward next correct entry  [-1, 1]
    10: correct_entry_dir_z - z-component of unit vector toward next correct entry  [-1, 1]

    Plan v5 — observation enrichment for RCCA-only RL (§32 audit B1/B2/B3):
    11: arc_to_next_daughter_norm   - arclength to next DAUGHTER fork (mm, clipped 100, /100)  [0, 1]
    12: arc_past_last_daughter_norm - arclength past most-recent daughter fork (mm, clipped 100, /100)  [0, 1]
    13: phase_default        - heuristic phase one-hot: default                              {0, 1}
    14: phase_trunk_top      - heuristic phase one-hot: Phase A (trunk-top crossing)         {0, 1}
    15: phase_bridge         - heuristic phase one-hot: Phase B (lcca_jn / bridge entry)     {0, 1}
    16: phase_daughter       - heuristic phase one-hot: Phase C (cavity / daughter in-route) {0, 1}
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

    Plan v13 obs-v3 — PathLookahead3D (the "zoom in on the planned path
    locally" fix). Forensic of lcca_awac_v1 showed the LCCA/LVA fork
    discrimination lives substantially in the tracking-y axis, which every
    2D-projected feature above DROPS (monoplane x,z). The heuristic that
    solved daughter entry used full-3D vessel-CS geometry; these features
    give the policy the same visibility. All in VESSEL CS (one fixed frame):
    30-32: waypoint_1 (dx,dy,dz)  - planned-path point at s+5mm, tip-relative,
                                    /50mm, clipped                              [-1, 1]
    33-35: waypoint_2 (dx,dy,dz)  - same at s+10mm                              [-1, 1]
    36-38: waypoint_3 (dx,dy,dz)  - same at s+20mm                              [-1, 1]
    39-41: waypoint_4 (dx,dy,dz)  - same at s+40mm (clamps to path end =
                                    target, so the 3D target position enters
                                    the obs through this feature)               [-1, 1]
    42-44: bend_hat_3d (x,y,z)    - J-tip bend direction unit vector in vessel
                                    CS (the y-component tells the agent which
                                    way ADVANCING will curve in depth — the
                                    axis the 2D bend_hat cannot see)            [-1, 1]
    45-47: entry_dir_3d (x,y,z)   - unit vector tip -> next correct daughter
                                    entry in vessel CS (3D version of 9-10)     [-1, 1]
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
# Plan v13 obs-v3 — PathLookahead3D: arclengths ahead of the tip projection at
# which planned-path waypoints are sampled, and the tip-relative offset clip.
# 5/10 mm cover the immediate hook curvature at a fork; 20/40 mm preview the
# post-fork daughter direction. 50 mm clip keeps offsets in [-1, 1] while
# leaving resolution at the 5-15 mm scale where the bif2 hook lives.
_WAYPOINT_LOOKAHEADS_MM = (5.0, 10.0, 20.0, 40.0)
_WAYPOINT_CLIP_MM = 50.0
_N_OBS = 48  # 30 (v2 features) + 12 waypoints + 3 bend_hat_3d + 3 entry_dir_3d


def _phase_to_onehot(phase: str) -> tuple:
    """Map a heuristic phase string to a 4-dim one-hot vector
    (default, trunk_top=A, bridge=B, daughter=C). Used for LocalGuidance
    features 13-16.

    Phase strings written by per-daughter policies:
      RVA  : default, trunk_top, lcca_jn (B), rva_cavity / rva_daughter (C)
      RCCA : default, trunk_top, lcca_jn (B), rcca_cavity / rcca_daughter (C)
      LCCA : default, trunk_top, lcca_bridge (B), lcca_daughter (C)
      LVA  : default, trunk_top, lva_daughter (C)  (no Phase B)
    """
    if not phase or phase == "default":
        return (1.0, 0.0, 0.0, 0.0)
    p = phase.lower()
    if p == "trunk_top":
        return (0.0, 1.0, 0.0, 0.0)
    # Phase B markers — bridge / jn-entry tokens
    if ("_jn" in p) or ("bridge" in p):
        return (0.0, 0.0, 1.0, 0.0)
    # Phase C markers — cavity / daughter in-route tokens
    if ("cavity" in p) or ("daughter" in p):
        return (0.0, 0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0, 0.0)


def _compute_bend_hat_vessel3d(fluoro) -> np.ndarray:
    """J-tip bend direction as a unit vector in VESSEL CS (3D).

    Mirrors the heuristic's closed-loop rotation algorithm
    (_compute_rotation_to_target in heuristic_policy_*.py): uses the
    first 3 tracked points (tip, tip-1, tip-2) and computes the
    second-difference vector ``p0 + p2 - 2*p1`` projected perpendicular
    to the tangent. Returns the zero vector if undefined.
    """
    tracking = fluoro.tracking3d
    if tracking is None or len(tracking) < 3:
        return np.zeros(3)
    rzx = fluoro.image_rot_zx
    center = fluoro.image_center
    try:
        p0 = tracking3d_to_vessel_cs(tracking[0], rzx, center)
        p1 = tracking3d_to_vessel_cs(tracking[1], rzx, center)
        p2 = tracking3d_to_vessel_cs(tracking[2], rzx, center)
    except Exception:
        return np.zeros(3)
    t_vec = p0 - p2
    t_norm = float(np.linalg.norm(t_vec))
    if t_norm < 1e-6:
        return np.zeros(3)
    t_hat = t_vec / t_norm
    bend_raw = p0 + p2 - 2.0 * p1
    bend = bend_raw - float(np.dot(bend_raw, t_hat)) * t_hat
    b_norm = float(np.linalg.norm(bend))
    if b_norm < 1e-3:
        return np.zeros(3)
    return bend / b_norm


def _compute_bend_hat_2d(fluoro) -> tuple:
    """J-tip bend direction projected to (x, z) image plane.

    2D projection of :func:`_compute_bend_hat_vessel3d` — kept for
    backward compatibility with features 17-18. Returns (bx, bz) unit
    vector in tracking3d 2D, or (0, 0) if undefined.
    """
    bend_hat = _compute_bend_hat_vessel3d(fluoro)
    if float(np.linalg.norm(bend_hat)) < 1e-8:
        return 0.0, 0.0
    # Project to tracking3d 2D (x, z) — same convention as path tangent_2d
    bend_t = vessel_cs_to_tracking3d(
        bend_hat, fluoro.image_rot_zx, (0.0, 0.0, 0.0), None
    )
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


def _entry_direction_3d(
    tip_vessel: np.ndarray,
    entry_coords: np.ndarray,
    dist: float,
) -> np.ndarray:
    """Unit vector tip -> entry_coords in VESSEL CS (3D).

    Plan v13 obs-v3 — the 3D version of :func:`_entry_direction`; keeps the
    y (out-of-image-plane) component the 2D feature drops. Returns the zero
    vector when dist is effectively infinite (no entry exists).
    """
    if dist >= _MAX_BIFURC_DIST_MM:
        return np.zeros(3)
    delta = entry_coords - tip_vessel
    n = float(np.linalg.norm(delta))
    if n < 1e-8:
        return np.zeros(3)
    return delta / n


class LocalGuidance(Observation):
    """Compact 48-dim observation encoding the agent's state relative to the path.

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

        self.obs = np.zeros(_N_OBS, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        low = np.array(
            # 0    1     2     3     4    5    6    7    8     9     10
            [0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0,
             # 11  12   13   14   15   16   17    18
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0,
             # 19  20   21   22   23   24   25   26   27   28   29
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # 30-47 — Plan v13 obs-v3: 4×3 waypoints + bend_hat_3d + entry_dir_3d
            + [-1.0] * 18,
            dtype=np.float32,
        )
        high = np.ones(_N_OBS, dtype=np.float32)
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
            self.obs = np.zeros(_N_OBS, dtype=np.float32)
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
            self.obs = np.zeros(_N_OBS, dtype=np.float32)
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

        # Feature 7: on correct branch? (true branch membership via nearest-branch lookup)
        if self._path_context is not None:
            on_path = 1.0 if self._path_context.is_on_correct_branch() else 0.0
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

        # Features 13-16: heuristic phase one-hot (default / A / B / C).
        # Env5 mirrors `_heur_rva_phase` from the env to intervention at
        # the start of each step() so LocalGuidance (which holds an
        # intervention reference) can read it. Falls back to "default".
        phase_str = getattr(self.intervention, "_heur_rva_phase", "default")
        ph_def, ph_A, ph_B, ph_C = _phase_to_onehot(str(phase_str))

        # Features 17-18: J-tip bend direction in tracking 2D.
        bend_x, bend_z = _compute_bend_hat_2d(fluoro)

        # ----------------------------------------------------------------
        # Plan v13 obs-v3 — PathLookahead3D (features 30-47), all VESSEL CS.
        # ----------------------------------------------------------------
        # Features 30-41: 4 planned-path waypoints at s+{5,10,20,40} mm,
        # tip-relative, /50 mm, clipped. Gives the local path SHAPE ahead
        # (the fork "hook" curve) including the y/depth component every
        # 2D-projected feature drops. Clamps to path end (= target).
        wp_offsets = []
        for la in _WAYPOINT_LOOKAHEADS_MM:
            wp = self._path_point_at_arclength(
                min(proj.s + la, self._total_length)
            )
            off = np.clip((wp - tip_vessel) / _WAYPOINT_CLIP_MM, -1.0, 1.0)
            wp_offsets.extend((float(off[0]), float(off[1]), float(off[2])))
        # Features 42-44: J-tip bend direction in vessel CS (3D).
        bend_hat_3d = _compute_bend_hat_vessel3d(fluoro)
        # Features 45-47: 3D unit vector toward the next correct entry.
        entry_dir_3d = _entry_direction_3d(tip_vessel, correct_coords, d_correct)

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
        # Plan v13 — normalize by the ACTUAL (env-var-overridable) grace max
        # mirrored by env5; falls back to the historical 50.
        off_max = float(getattr(
            self.intervention, "_env_off_branch_max", _OFF_BRANCH_TIMEOUT_STEPS
        ))
        off_branch_norm = float(min(off_steps / max(off_max, 1.0), 1.0))
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
                ph_def,                                                  # 13 {0,1} phase default
                ph_A,                                                    # 14 {0,1} phase trunk_top (A)
                ph_B,                                                    # 15 {0,1} phase bridge (B)
                ph_C,                                                    # 16 {0,1} phase daughter (C)
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
            ]
            # Plan v13 obs-v3 — PathLookahead3D:
            + wp_offsets                                                 # 30-41 [-1,1] 4×3 waypoints
            + [float(bend_hat_3d[0]),                                    # 42 [-1,1] bend_hat_3d x
               float(bend_hat_3d[1]),                                    # 43 [-1,1] bend_hat_3d y
               float(bend_hat_3d[2]),                                    # 44 [-1,1] bend_hat_3d z
               float(entry_dir_3d[0]),                                   # 45 [-1,1] entry_dir_3d x
               float(entry_dir_3d[1]),                                   # 46 [-1,1] entry_dir_3d y
               float(entry_dir_3d[2])],                                  # 47 [-1,1] entry_dir_3d z
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_point_at_arclength(self, s_q: float) -> np.ndarray:
        """Interpolated planned-path point (vessel CS) at arclength ``s_q``.

        Plan v13 obs-v3 helper for the PathLookahead3D waypoints. Clamps to
        the polyline ends; assumes ``self._polyline``/``self._cumlen`` are
        populated (guarded by the ``_total_length`` check in ``step``).
        """
        if len(self._polyline) < 2:
            return self._polyline[0] if len(self._polyline) else np.zeros(3)
        s_q = float(np.clip(s_q, 0.0, self._total_length))
        idx = int(np.searchsorted(self._cumlen, s_q))
        idx = max(1, min(idx, len(self._polyline) - 1))
        s0 = float(self._cumlen[idx - 1])
        s1 = float(self._cumlen[idx])
        if s1 - s0 < 1e-9:
            return self._polyline[idx]
        t = (s_q - s0) / (s1 - s0)
        return self._polyline[idx - 1] + t * (
            self._polyline[idx] - self._polyline[idx - 1]
        )

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
