"""RVA-specific heuristic policy — v2 (Phase A/B) + refined Phase C.

  Phase A — Trunk-top crossing: gw_trans=10 mm/s, window
            [trunk_top_arc - 30, trunk_top_arc + 10], target FIXED -z
            (degenerate, just modulates speed).

  Phase B — LCCA-junction crossing: gw_trans=1.5 mm/s, window
            [lcca_jn_arc - 15, lcca_jn_arc + 30], target FIXED +z
            (degenerate, just modulates speed).

  Phase C — RVA-jn cavity transit + in-daughter advance:
    Sub-regime "rva_cavity"   (s in [rva_jn_arc - 5, rva_jn_arc + 5]):
        gw_trans = 3 mm/s. Target = planned-path tangent at (s + 5),
        which equals RVA's first-segment direction (~(-0.33, +0.87, -0.37))
        at the cavity entry. The +y dominant component discriminates
        RVA from RCCA (RCCA has -y component) — closed-loop biases
        the J-curl toward RVA's opening within the merged cavity.
    Sub-regime "rva_daughter" (s > rva_jn_arc + 5):
        gw_trans = 2 mm/s. Target = local RVA tangent at (s + 5),
        following the daughter's curving centerline.

  Mesh inspection (RL_IMPROV_8 §26) revealed the "RVA-jn" is a 5 mm
  tall, ~12 mm radius MERGED CAVITY where bridge(11), RVA, and RCCA
  share open lumen. Wire's J-curl orientation in the cavity decides
  which daughter it commits to. Phase C's dynamic tangent target
  exploits the +y vs -y discrimination cleanly.
"""
import numpy as np

from .heuristic_policy import HeuristicActionFunction


# Arclength windows (v2 strategy — best result observed: 28/50 reached RVA)
PHASE_A_ARC_BEFORE = 30.0   # start phase A 30 mm before trunk-top
PHASE_A_ARC_AFTER = 10.0    # extend 10 mm past trunk-top
PHASE_B_ARC_BEFORE = 15.0   # start phase B 15 mm before LCCA-jn
PHASE_B_ARC_AFTER = 30.0    # extend 30 mm past LCCA-jn

# Phase A — fast confident push past trunk-top with fixed -z target
# (degenerate vs +z long-axis, so closed-loop usually returns ~0 rotation).
PHASE_A_GW_TRANS = 10.0
PHASE_A_TARGET_DIR = np.array([0.0, 0.0, -1.0])

# Phase B — slow deliberate push at/past LCCA-jn with fixed +z target
# (degenerate vs ascending long-axis at the LCCA-jn region).
PHASE_B_GW_TRANS = 1.5
PHASE_B_TARGET_DIR = np.array([0.0, 0.0, +1.0])

# Phase C — RVA-jn cavity transit + in-daughter advance
# (RL_IMPROV_8 §26 — designed from mesh inspection findings: the "RVA-jn"
# is a 5 mm tall, ~12 mm radius MERGED CAVITY where bridge(11)/RVA/RCCA
# share open lumen. Wire's J-curl orientation in the cavity decides
# RVA vs RCCA. RVA's first-segment tangent has +y dominant component
# while RCCA's tangent has strong -x and -y — they discriminate cleanly
# along the y axis. Dynamic tangent target with original sign exploits
# this discrimination naturally.)
#
# Two sub-regimes:
#   (a) Cavity transit (s in [rva_jn_arc - 5, rva_jn_arc + 5]):
#       Wire is approaching or in the merged cavity. Closed-loop biases
#       J-curl toward RVA's local tangent (+y dominant). gw_trans=3 mm/s
#       — keep wire moving so it doesn't dwell in cavity (heuristic's
#       d_rem-based slowdown otherwise drops gw_trans to 0.5 mm/s here).
#   (b) In-daughter advance (s > rva_jn_arc + 5):
#       Wire is committed to RVA proper. Slower gw_trans=2 mm/s for the
#       narrowing daughter lumen. Closed-loop continues to follow the
#       daughter's curving centerline.
PHASE_C_ARC_BEFORE = 5.0    # start Phase C 5 mm before RVA-jn (cavity entry)
PHASE_C_CAVITY_GW_TRANS = 3.0  # cavity transit speed
PHASE_C_DAUGHTER_GW_TRANS = 2.0  # in-daughter speed
PHASE_C_LOOKAHEAD_MM = 5.0  # tangent at (s + 5) — gives ~RVA[1]'s direction
                             # at cavity entry, RVA[2-3] at cavity center.

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
        # RVA-jn: prev=(11), next=RVA daughter (Phase C trigger anchor)
        self._rva_jn_arc = _find_junction_arc(junctions, "(11)", "RVA")
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
        # Phase C takes priority: once approaching/past the RVA-jn we're
        # navigating the merged cavity + RVA daughter and need dynamic
        # tangent steering all the way through.
        if (self._rva_jn_arc is not None
                and s >= self._rva_jn_arc - PHASE_C_ARC_BEFORE):
            # Sub-regime split: cavity transit vs in-daughter
            if s < self._rva_jn_arc + 5.0:
                return "rva_cavity"
            return "rva_daughter"
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
        # Phase A target = fixed -z, Phase B target = fixed +z (both
        # degenerate vs +z long-axis — closed-loop returns ~0 rotation,
        # phases just modulate gw_trans). Phase C uses dynamic
        # planned-path tangent (RVA's first-segment tangent +y dominant
        # discriminates RVA from RCCA in the merged cavity at the
        # trifurcation; in-daughter target follows RVA's curving
        # centerline).
        if phase == "trunk_top":
            gw_trans = PHASE_A_GW_TRANS
            target = PHASE_A_TARGET_DIR
        elif phase == "lcca_jn":
            gw_trans = PHASE_B_GW_TRANS
            target = PHASE_B_TARGET_DIR
        else:  # phase in {"rva_cavity", "rva_daughter"}
            if phase == "rva_cavity":
                gw_trans = PHASE_C_CAVITY_GW_TRANS
            else:
                gw_trans = PHASE_C_DAUGHTER_GW_TRANS
            try:
                pc = base_env._path_context
                s = float(pc.get_projection().s)
                target = pc.get_planned_path_tangent_at(
                    s, PHASE_C_LOOKAHEAD_MM
                )
                if float(np.linalg.norm(target)) < 1e-6:
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
