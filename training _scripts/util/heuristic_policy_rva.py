"""RVA-specific heuristic policy with junction-orientation control.

Strategy (RL_IMPROV_8 RVA experiment):
  Phase A — Trunk-top crossing: at trunk-top junction (47, 34, 392) the
            three branches {trunk(2), (0), (18)} meet. Branch (18) is the
            LVA bridge that climbs up (+z) into the LVA approach; branch
            (0) is the bridge from trunk-top DOWN to the LCCA-junction at
            (23, 16, 385). For RCCA/RVA targets we need (0), so we want
            the wire's J-tip oriented toward -z when crossing trunk-top
            (so it doesn't get caught against (18)'s wall).

  Phase B — LCCA-junction post-crossing: at LCCA-jn the three branches
            {(0), (11), LCCA} meet. Branch (11) (the bridge to RCCA/RVA
            ostium) climbs UP (+z) — the path's first segment jumps
            Δz=+12.5 mm from i=0 to i=1. We want the J-tip oriented
            toward +z right after crossing the junction so it lodges
            against the (11) wall and threads into the bridge.

  Phase C — In-daughter advance: once the wire's projection arclength
            crosses the bridge → RVA junction (rva_jn_arc), we're past
            the last branching choice and inside the target daughter.
            The daughter's centerline tangent rotates continuously
            (RVA's first 10 mm sweeps from +y to +z); a single fixed
            target like Phase A/B can't track it. Phase C samples the
            local planned-path tangent at (s + 5 mm) every step, so the
            J-curl is always oriented toward where the wire is heading
            next. Slow gw_trans (2 mm/s) gives torsion time to land in
            the narrow daughter lumen. Active all the way to target.

Phase activation by **arclength along the planned path** (not Euclidean
distance to junction nodes). The arclength windows are wide and run
mostly BEFORE each junction to give the wire time for SOFA torsional
propagation to actually rotate the J-tip — single-step rotation
commands don't measurably reorient the tip through ~140 mm of
inserted shaft.

Closed-loop tip-bend feedback:
  Each step reads the wire's first 3 tracking points (tracking[0..2]) to
  estimate:
    - Local long axis at tip:   t_hat = (tracking[0] - tracking[2]) / norm
    - Current bend direction:   discrete 2nd-derivative b = p0 + p2 - 2 p1,
                                projected onto plane perpendicular to t_hat
  Desired bend direction is the target (+z or -z) projected into the same
  perpendicular plane. The signed angle from current bend to desired
  bend, taken about t_hat, gives gw_rot magnitude AND sign.

Translation is kept low (2 mm/s for trunk-top, 1.5 mm/s for LCCA-jn)
during phase mode so the wire body doesn't advance faster than torsion
can propagate to the J-tip — fold-stall is the failure mode when the
body races ahead of an unrotated tip.
"""
import numpy as np

from .heuristic_policy import HeuristicActionFunction


# Arclength windows (mm) relative to each junction's planned-path arclength.
# Negative = before the junction (lead-in for pre-rotation), positive = past.
# Wider lead-in than follow-through: most of the rotation work needs to
# finish BEFORE the wire arrives at the kink.
PHASE_A_ARC_BEFORE = 30.0   # start phase A 30 mm before trunk-top
PHASE_A_ARC_AFTER = 10.0    # extend 10 mm past trunk-top
PHASE_B_ARC_BEFORE = 15.0   # start phase B 15 mm before LCCA-jn
PHASE_B_ARC_AFTER = 30.0    # extend 30 mm past LCCA-jn (longer so tip
                             # has time to commit into bridge (11))

# Phase A — fast confident push past trunk-top, target tip bend toward -z.
# Trunk-top is a gentle transition (trunk(2) → (0) descent), not a sharp
# turn — speed avoids getting caught on (18)'s wall.
PHASE_A_GW_TRANS = 10.0
PHASE_A_TARGET_DIR = np.array([0.0, 0.0, -1.0])

# Phase B — slow deliberate push at/past LCCA-jn, target tip bend toward
# +z. This IS the sharp ~100° turn (Δz=+12.5 mm at i=0→1 of bridge
# (11)); slow enough that torsion propagates and J-tip reorients
# before the wire body arrives at the kink.
PHASE_B_GW_TRANS = 1.5
PHASE_B_TARGET_DIR = np.array([0.0, 0.0, +1.0])

# Phase C — in-daughter advance. Activated once projection arclength is
# past the bridge → daughter junction; runs all the way to target.
# Target direction is DYNAMIC: sampled from the planned-path tangent at
# (s + LOOKAHEAD) so the J-curl is oriented toward where the wire is
# heading next, not where it currently sits. As the daughter centerline
# curves through its initial 60° turn (RVA's tangent rotates from +y to
# +z over ~10 mm of arclength), the target rotates continuously and the
# closed-loop rotation drives the J-curl to follow.
PHASE_C_GW_TRANS = 2.0
PHASE_C_LOOKAHEAD_MM = 5.0

# Rotation control gains
GW_ROT_GAIN = 1.0    # rad-per-radian (proportional)
GW_ROT_MAX = 1.5     # action-space limit for guidewire rotation

CATH_FOLLOW_RATIO = 0.8  # cath_trans = gw_trans * ratio


def _compute_rotation_to_target(tracking_vcs: np.ndarray,
                                target_dir: np.ndarray) -> float:
    """Closed-loop gw_rot to align the J-tip's bend direction with target.

    Uses the first three tracking points (in vessel-CS):
      - t_hat: local long-axis at tip = unit (tracking[0] - tracking[2])
      - bend:  2nd-derivative direction at tracking[1], projected onto
               plane perpendicular to t_hat
      - desired_bend: target_dir projected onto plane perpendicular to t_hat

    Signed angle FROM bend TO desired_bend, measured about t_hat, gives
    a gw_rot proportional to misalignment.
    """
    if tracking_vcs is None or len(tracking_vcs) < 3:
        return 0.0
    p0 = np.asarray(tracking_vcs[0], dtype=float)
    p1 = np.asarray(tracking_vcs[1], dtype=float)
    p2 = np.asarray(tracking_vcs[2], dtype=float)

    t_vec = p0 - p2
    t_norm = float(np.linalg.norm(t_vec))
    if t_norm < 1e-6:
        return 0.0
    t_hat = t_vec / t_norm

    bend_raw = p0 + p2 - 2.0 * p1
    bend = bend_raw - np.dot(bend_raw, t_hat) * t_hat
    bend_norm = float(np.linalg.norm(bend))
    if bend_norm < 1e-3:
        return 0.0
    bend_hat = bend / bend_norm

    target = np.asarray(target_dir, dtype=float)
    target_perp = target - np.dot(target, t_hat) * t_hat
    tp_norm = float(np.linalg.norm(target_perp))
    if tp_norm < 1e-3:
        return 0.0
    target_hat = target_perp / tp_norm

    cos_a = float(np.clip(np.dot(bend_hat, target_hat), -1.0, 1.0))
    cross_v = np.cross(bend_hat, target_hat)
    sign = float(np.sign(np.dot(cross_v, t_hat)))
    if sign == 0.0:
        sign = 1.0
    angle = float(np.arccos(cos_a))
    gw_rot = sign * GW_ROT_GAIN * angle
    return float(np.clip(gw_rot, -GW_ROT_MAX, GW_ROT_MAX))


def _find_junction_arc(junctions, prev_token, next_token):
    """Search ``junctions`` (list of (arc, prev_name, next_name) from
    ``path_context.get_path_junctions()``) for the arclength of the
    junction whose previous-branch name contains ``prev_token`` AND whose
    next-branch name contains ``next_token``. Returns None if no match.
    """
    for arc, prev_name, next_name in junctions:
        if prev_token in prev_name and next_token in next_name:
            return float(arc)
    return None


class RVAHeuristicActionFunction(HeuristicActionFunction):
    """RVA-target heuristic with arclength-based phase override at the
    trunk-top and LCCA-jn crossings, and closed-loop tip-bend rotation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cached at first reset (when planned path is built).
        self._trunk_top_arc = None
        self._lcca_jn_arc = None
        self._rva_jn_arc = None

    def _maybe_cache_junctions(self, base_env):
        if (self._trunk_top_arc is not None
                and self._lcca_jn_arc is not None
                and self._rva_jn_arc is not None):
            return
        try:
            junctions = base_env._path_context.get_path_junctions()
        except Exception:
            return
        # trunk-top: prev=trunk(2), next=(0)
        self._trunk_top_arc = _find_junction_arc(junctions, "(2)", "(0)")
        # LCCA-jn: prev=(0), next=(11)
        self._lcca_jn_arc = _find_junction_arc(junctions, "(0)", "(11)")
        # RVA-jn: prev=(11), next=RVA daughter
        self._rva_jn_arc = _find_junction_arc(junctions, "(11)", "RVA")
        # Publish to env for one-shot diagnostics on the first INFO line.
        try:
            base_env._heur_rva_trunk_top_arc = (
                float(self._trunk_top_arc) if self._trunk_top_arc else -1.0
            )
            base_env._heur_rva_lcca_jn_arc = (
                float(self._lcca_jn_arc) if self._lcca_jn_arc else -1.0
            )
            base_env._heur_rva_rva_jn_arc = (
                float(self._rva_jn_arc) if self._rva_jn_arc else -1.0
            )
        except Exception:
            pass

    def _detect_phase(self, base_env) -> str:
        self._maybe_cache_junctions(base_env)
        try:
            s = float(base_env._path_context.get_projection().s)
        except Exception:
            return "default"
        # Phase C takes priority: once past the bridge → daughter junction,
        # we're inside RVA and want continuous tangent-follow until target.
        if self._rva_jn_arc is not None and s >= self._rva_jn_arc:
            return "in_daughter"
        if (self._trunk_top_arc is not None
                and (self._trunk_top_arc - PHASE_A_ARC_BEFORE)
                <= s
                <= (self._trunk_top_arc + PHASE_A_ARC_AFTER)):
            return "trunk_top"
        if (self._lcca_jn_arc is not None
                and (self._lcca_jn_arc - PHASE_B_ARC_BEFORE)
                <= s
                <= (self._lcca_jn_arc + PHASE_B_ARC_AFTER)):
            return "lcca_jn"
        return "default"

    def _tracking_in_vessel_cs(self, base_env):
        from eve.util.coordtransform import tracking3d_to_vessel_cs
        try:
            fluoro = base_env.intervention.fluoroscopy
            tracking = fluoro.tracking3d
            if tracking is None or len(tracking) < 3:
                return None
            rzx = fluoro.image_rot_zx
            center = fluoro.image_center
            return np.array(
                [tracking3d_to_vessel_cs(p, rzx, center) for p in tracking[:5]],
                dtype=float,
            )
        except Exception:
            return None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        base_env = getattr(self.env, "unwrapped", self.env)

        on_correct_path = True
        try:
            on_correct_path = base_env._path_context.is_on_correct_path()
        except Exception:
            pass

        phase = "default"
        if on_correct_path:
            phase = self._detect_phase(base_env)

        try:
            base_env._heur_rva_phase = phase
        except Exception:
            pass

        if phase == "default":
            return super().__call__(obs)

        if self._needs_reset:
            self.heuristic.reset()
            self._needs_reset = False

        tracking_vcs = self._tracking_in_vessel_cs(base_env)
        if phase == "trunk_top":
            gw_trans = PHASE_A_GW_TRANS
            target = PHASE_A_TARGET_DIR
        elif phase == "lcca_jn":
            gw_trans = PHASE_B_GW_TRANS
            target = PHASE_B_TARGET_DIR
        else:  # phase == "in_daughter"
            gw_trans = PHASE_C_GW_TRANS
            try:
                pc = base_env._path_context
                s = float(pc.get_projection().s)
                target = pc.get_planned_path_tangent_at(s, PHASE_C_LOOKAHEAD_MM)
                if float(np.linalg.norm(target)) < 1e-6:
                    # Tangent unavailable — skip rotation override this step.
                    target = np.zeros(3)
            except Exception:
                target = np.zeros(3)
        gw_rot = _compute_rotation_to_target(tracking_vcs, target)

        try:
            base_env._heur_rva_target_z = float(target[2])
            base_env._heur_rva_target_x = float(target[0])
            base_env._heur_rva_target_y = float(target[1])
            base_env._heur_rva_gw_rot = float(gw_rot)
        except Exception:
            pass

        cath_trans = gw_trans * CATH_FOLLOW_RATIO
        raw_action = np.array([gw_trans, gw_rot, cath_trans, 0.0],
                              dtype=np.float64)

        action = raw_action.flatten().astype(np.float64)
        if self.noise_std > 0:
            noise = self._rng.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, self.action_low, self.action_high)
        if self.normalize_output:
            action = 2 * (action - self.action_low) / self.action_range - 1
            action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32)


class RVAHeuristicActionFunctionFactory:
    """Pickleable factory for RVAHeuristicActionFunction."""

    def __init__(self, noise_std: float = 0.0, normalize_output: bool = True):
        self.noise_std = noise_std
        self.normalize_output = normalize_output

    def create(self, env) -> RVAHeuristicActionFunction:
        return RVAHeuristicActionFunction(
            env=env,
            noise_std=self.noise_std,
            normalize_output=self.normalize_output,
        )
