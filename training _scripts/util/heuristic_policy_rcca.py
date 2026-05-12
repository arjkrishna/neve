"""RCCA-specific heuristic policy — exact mirror of RVA's policy with
the daughter-junction token swapped from RVA to RCCA. Path topology
is identical to RVA up to the bif2 cavity; only the daughter the
planned path commits to differs.

  Phase A — Trunk-top crossing: gw_trans=10 mm/s, window
            [trunk_top_arc - 30, trunk_top_arc + 10], target FIXED -z
            (degenerate, just modulates speed). Identical to RVA.

  Phase B — LCCA-junction crossing: gw_trans=1.5 mm/s, window
            [lcca_jn_arc - 15, lcca_jn_arc + 30], target FIXED +z
            (degenerate, just modulates speed). Identical to RVA.

  Phase C — RCCA-jn cavity transit + in-daughter advance.
    Variant ID is read from ``base_env._phase_c_variant`` (set in
    env5.reset from options['phase_c_variant']). Defaults to "C2"
    (current production) if not set. The 9 variants for the factorial
    grid test (Plan v3) are defined in PHASE_C_VARIANTS below.
    `dynamic_tangent` target adapts naturally to whichever daughter is
    on the planned path — for an RCCA target, the planned-path tangent
    at s+5 points along RCCA's first-segment direction.
"""
import numpy as np

from .heuristic_policy import HeuristicActionFunction


# ============================================================================
# Phase A / B (unchanged from v8)
# ============================================================================
PHASE_A_ARC_BEFORE = 30.0
PHASE_A_ARC_AFTER = 10.0
PHASE_A_GW_TRANS = 10.0
PHASE_A_TARGET_DIR = np.array([0.0, 0.0, -1.0])

PHASE_B_ARC_BEFORE = 15.0
PHASE_B_ARC_AFTER = 30.0
PHASE_B_GW_TRANS = 1.5
PHASE_B_TARGET_DIR = np.array([0.0, 0.0, +1.0])

# ============================================================================
# Phase C — VARIANT CATALOG (Plan v3, RL_IMPROV_8 §28)
# ============================================================================
# Each variant is a dict with keys:
#   trigger_before:  arclength offset before rva_jn_arc to start firing
#   cavity_gw_trans: gw_trans during cavity transit (s in [-trigger_before, +5])
#   daughter_gw_trans: gw_trans inside RVA daughter (s > rva_jn_arc + 5)
#   target_kind:     "fixed_y" | "dynamic_tangent" | "position_to_target"
#   lookahead_mm:    only used by dynamic_tangent variants
#   stagnation:      None | "retract"  (C8: retract -3 if stuck)
#
# The action computation reads these per-variant config values at runtime.
PHASE_C_VARIANTS = {
    "C0": {  # baseline: no Phase C; fall through to default heuristic
        "trigger_before": None,
        "cavity_gw_trans": None,
        "daughter_gw_trans": None,
        "target_kind": None,
        "lookahead_mm": None,
        "stagnation": None,
    },
    "C1": {  # fixed +y target (mesh-discriminator)
        "trigger_before": 5.0,
        "cavity_gw_trans": 3.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "fixed_y",
        "lookahead_mm": None,
        "stagnation": None,
    },
    "C2": {  # dynamic tangent, lookahead=5 (current production / §27 baseline)
        "trigger_before": 5.0,
        "cavity_gw_trans": 3.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "dynamic_tangent",
        "lookahead_mm": 5.0,
        "stagnation": None,
    },
    "C3": {  # reactive lookahead
        "trigger_before": 5.0,
        "cavity_gw_trans": 3.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "dynamic_tangent",
        "lookahead_mm": 1.0,
        "stagnation": None,
    },
    "C4": {  # anticipatory lookahead
        "trigger_before": 5.0,
        "cavity_gw_trans": 3.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "dynamic_tangent",
        "lookahead_mm": 10.0,
        "stagnation": None,
    },
    "C5": {  # position-vector toward target
        "trigger_before": 5.0,
        "cavity_gw_trans": 3.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "position_to_target",
        "lookahead_mm": None,
        "stagnation": None,
    },
    "C6": {  # uniform slow speed
        "trigger_before": 5.0,
        "cavity_gw_trans": 2.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "dynamic_tangent",
        "lookahead_mm": 5.0,
        "stagnation": None,
    },
    "C7": {  # wider lead-in (start at rva_jn_arc - 15)
        "trigger_before": 15.0,
        "cavity_gw_trans": 3.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "dynamic_tangent",
        "lookahead_mm": 5.0,
        "stagnation": None,
    },
    "C8": {  # dynamic + stagnation retract
        "trigger_before": 5.0,
        "cavity_gw_trans": 3.0,
        "daughter_gw_trans": 2.0,
        "target_kind": "dynamic_tangent",
        "lookahead_mm": 5.0,
        "stagnation": "retract",
    },
}

# Stagnation parameters (used by C8)
STAGNATION_WINDOW_STEPS = 20  # look at last N steps' Δs
STAGNATION_DELTA_MM = 1.0     # if Δs over the window < this, declare stuck
STAGNATION_RETRACT_STEPS = 5  # for this many steps, command gw_trans=-3
STAGNATION_RETRACT_GW = -3.0

# Rotation control gains
GW_ROT_GAIN = 1.0
GW_ROT_MAX = 1.5
CATH_FOLLOW_RATIO = 0.8


# ============================================================================
# Helpers
# ============================================================================
def _compute_rotation_to_target(tracking_vcs: np.ndarray,
                                target_dir: np.ndarray) -> float:
    """Closed-loop gw_rot to align J-tip's bend with target_dir.

    Sign convention: original right-hand rule (validated; do NOT flip;
    flipping caused v6/v7 regressions).
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
    for arc, prev_name, next_name in junctions:
        if prev_token in prev_name and next_token in next_name:
            return float(arc)
    return None


def _phase_c_target(base_env, variant_cfg, s):
    """Compute the target direction for Phase C per variant config.
    Returns a 3-vector in vessel-CS, or zero vector if unavailable.
    """
    kind = variant_cfg["target_kind"]
    if kind == "fixed_y":
        return np.array([0.0, 1.0, 0.0])
    if kind == "dynamic_tangent":
        try:
            pc = base_env._path_context
            t = pc.get_planned_path_tangent_at(s, variant_cfg["lookahead_mm"])
            if float(np.linalg.norm(t)) < 1e-6:
                return np.zeros(3)
            return np.asarray(t, dtype=float)
        except Exception:
            return np.zeros(3)
    if kind == "position_to_target":
        try:
            from eve.util.coordtransform import tracking3d_to_vessel_cs
            tgt = base_env.intervention.target.coordinates3d
            tgt_v = np.asarray(tgt, dtype=float)
            fluoro = base_env.intervention.fluoroscopy
            tip_eve = fluoro.tracking3d[0]
            tip_v = np.asarray(
                tracking3d_to_vessel_cs(
                    tip_eve, fluoro.image_rot_zx, fluoro.image_center
                ),
                dtype=float,
            )
            d = tgt_v - tip_v
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                return np.zeros(3)
            return d / n
        except Exception:
            return np.zeros(3)
    return np.zeros(3)


# ============================================================================
# Action function class
# ============================================================================
class RCCAHeuristicActionFunction(HeuristicActionFunction):
    """RCCA-target heuristic with arclength-based phase override at the
    trunk-top, LCCA-jn, and RCCA-jn crossings. Phase C variant ID is read
    from ``base_env._phase_c_variant`` (default "C2")."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cached at first reset (when planned path is built).
        self._trunk_top_arc = None
        self._lcca_jn_arc = None
        self._rcca_jn_arc = None
        # C8 stagnation tracking
        self._proj_s_history = []  # rolling window of last N proj.s values
        self._stagnation_retract_remaining = 0

    def reset(self):
        super().reset()
        # Reset per-episode state. Junction arcs are re-cached lazily
        # when _maybe_cache_junctions is called next; clear here so a
        # different env-with-different-target reuses the same factory
        # cleanly.
        self._trunk_top_arc = None
        self._lcca_jn_arc = None
        self._rcca_jn_arc = None
        self._proj_s_history = []
        self._stagnation_retract_remaining = 0

    def _maybe_cache_junctions(self, base_env):
        if (self._trunk_top_arc is not None
                and self._lcca_jn_arc is not None
                and self._rcca_jn_arc is not None):
            return
        try:
            junctions = base_env._path_context.get_path_junctions()
        except Exception:
            return
        self._trunk_top_arc = _find_junction_arc(junctions, "(2)", "(0)")
        self._lcca_jn_arc = _find_junction_arc(junctions, "(0)", "(11)")
        self._rcca_jn_arc = _find_junction_arc(junctions, "(11)", "RCCA")
        try:
            base_env._heur_rcca_trunk_top_arc = (
                float(self._trunk_top_arc) if self._trunk_top_arc else -1.0
            )
            base_env._heur_rcca_lcca_jn_arc = (
                float(self._lcca_jn_arc) if self._lcca_jn_arc else -1.0
            )
            base_env._heur_rcca_rcca_jn_arc = (
                float(self._rcca_jn_arc) if self._rcca_jn_arc else -1.0
            )
        except Exception:
            pass

    def _detect_phase(self, base_env, variant_cfg) -> str:
        """Determine which phase fires this step. Phase C only fires if the
        variant's trigger_before is not None (C0 disables Phase C)."""
        self._maybe_cache_junctions(base_env)
        try:
            s = float(base_env._path_context.get_projection().s)
        except Exception:
            return "default"
        # Phase C (variant-gated)
        c_before = variant_cfg.get("trigger_before")
        if (c_before is not None
                and self._rcca_jn_arc is not None
                and s >= self._rcca_jn_arc - c_before):
            if s < self._rcca_jn_arc + 5.0:
                return "rcca_cavity"
            return "rcca_daughter"
        # Phase A
        if (self._trunk_top_arc is not None
                and (self._trunk_top_arc - PHASE_A_ARC_BEFORE)
                <= s
                <= (self._trunk_top_arc + PHASE_A_ARC_AFTER)):
            return "trunk_top"
        # Phase B
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

    def _check_stagnation(self, base_env):
        """For C8: track Δs over a rolling window. Returns True if stuck."""
        try:
            s = float(base_env._path_context.get_projection().s)
        except Exception:
            return False
        self._proj_s_history.append(s)
        if len(self._proj_s_history) > STAGNATION_WINDOW_STEPS:
            self._proj_s_history.pop(0)
        if len(self._proj_s_history) < STAGNATION_WINDOW_STEPS:
            return False
        delta = self._proj_s_history[-1] - self._proj_s_history[0]
        return abs(delta) < STAGNATION_DELTA_MM

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        base_env = getattr(self.env, "unwrapped", self.env)

        # Read Phase C variant ID from env (set in env5.reset()).
        variant_id = getattr(base_env, "_phase_c_variant", "C2") or "C2"
        variant_cfg = PHASE_C_VARIANTS.get(variant_id, PHASE_C_VARIANTS["C2"])

        on_correct_path = True
        try:
            on_correct_path = base_env._path_context.is_on_correct_path()
        except Exception:
            pass

        phase = "default"
        if on_correct_path:
            phase = self._detect_phase(base_env, variant_cfg)

        try:
            base_env._heur_rcca_phase = phase
            base_env._heur_rcca_variant = variant_id
            # env5 STEP-log writer reads `_heur_rva_phase` for the
            # phase= field; mirror RCCA phase there too so logs are
            # informative regardless of which factory is in use.
            base_env._heur_rva_phase = phase
        except Exception:
            pass

        if phase == "default":
            return super().__call__(obs)

        if self._needs_reset:
            self.heuristic.reset()
            self._needs_reset = False

        tracking_vcs = self._tracking_in_vessel_cs(base_env)

        # ---- Action selection by phase ----
        if phase == "trunk_top":
            gw_trans = PHASE_A_GW_TRANS
            target = PHASE_A_TARGET_DIR
        elif phase == "lcca_jn":
            gw_trans = PHASE_B_GW_TRANS
            target = PHASE_B_TARGET_DIR
        else:  # phase in {"rcca_cavity", "rcca_daughter"}
            # gw_trans by sub-regime
            if phase == "rcca_cavity":
                gw_trans = variant_cfg["cavity_gw_trans"]
            else:
                gw_trans = variant_cfg["daughter_gw_trans"]

            # Stagnation retract (C8 only)
            if variant_cfg.get("stagnation") == "retract":
                if self._stagnation_retract_remaining > 0:
                    self._stagnation_retract_remaining -= 1
                    gw_trans = STAGNATION_RETRACT_GW
                elif self._check_stagnation(base_env):
                    self._stagnation_retract_remaining = STAGNATION_RETRACT_STEPS
                    gw_trans = STAGNATION_RETRACT_GW

            # target by variant
            try:
                s_now = float(base_env._path_context.get_projection().s)
            except Exception:
                s_now = 0.0
            target = _phase_c_target(base_env, variant_cfg, s_now)

        gw_rot = _compute_rotation_to_target(tracking_vcs, target)

        try:
            base_env._heur_rcca_target_z = float(target[2])
            base_env._heur_rcca_target_x = float(target[0])
            base_env._heur_rcca_target_y = float(target[1])
            base_env._heur_rcca_gw_rot = float(gw_rot)
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


class RCCAHeuristicActionFunctionFactory:
    """Pickleable factory for RCCAHeuristicActionFunction."""

    def __init__(self, noise_std: float = 0.0, normalize_output: bool = True):
        self.noise_std = noise_std
        self.normalize_output = normalize_output

    def create(self, env) -> RCCAHeuristicActionFunction:
        return RCCAHeuristicActionFunction(
            env=env,
            noise_std=self.noise_std,
            normalize_output=self.normalize_output,
        )
