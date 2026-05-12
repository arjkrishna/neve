"""LVA-specific heuristic policy — Phase C only.

LVA target is the easiest daughter to reach geometrically. Wire's
natural ascent up trunk(2) past trunk-top delivers it into branch (18)
(the LVA bridge) without any active steering — this is the same
"deflection into (18)" failure mode we observed in ~10-30% of
RVA/RCCA/LCCA episodes, used INTENTIONALLY here. So:

  Phase A — DISABLED. The default centerline-follower already drives
            the wire up trunk and into the (18) bridge at trunk-top.
            Adding active Phase A risks fighting that natural ascent.

  Phase B — DOES NOT EXIST. LVA path has no intermediate junction
            analogous to RVA's (0)→(11) bridge crossing. The path is
            simply trunk(2) → (18) → LVA, two junctions total.

  Phase C — LVA daughter entry + in-daughter advance. Variant ID is
            read from ``base_env._phase_c_variant`` (default "C2" =
            dynamic_tangent at s+5). Mirror of RVA's Phase C structure
            with daughter junction token swapped to (18)→LVA.

Junction tokens cached:
  _lva_jn_arc = arc of (18)→LVA   [Phase C anchor]
"""
import numpy as np

from .heuristic_policy import HeuristicActionFunction


# ============================================================================
# Phase A — trunk-to-(18) bridge crossing (= entire window before LVA daughter)
# ============================================================================
# Anchored at lva_jn (the (18)→LVA junction). Window = [lva_jn-30, lva_jn]
# covers trunk-top + (18) bridge transit. Target = +z, degenerate vs wire's
# ascending long axis → closed-loop returns ~0 rotation, phase just keeps
# gw_trans high (10 mm/s) so wire ascends confidently through the bridge.
PHASE_A_ARC_BEFORE = 30.0     # window starts 30 mm before lva_jn
PHASE_A_GW_TRANS = 10.0
PHASE_A_TARGET_DIR = np.array([0.0, 0.0, +1.0])

# Phase B does not exist for LVA.
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
class LVAHeuristicActionFunction(HeuristicActionFunction):
    """LVA-target heuristic — Phase C only. Default centerline-follower
    handles the trunk → (18) crossing (no Phase A needed; natural ascent
    delivers wire into LVA bridge). Phase C variant ID is read from
    ``base_env._phase_c_variant`` (default "C2")."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cached at first reset (when planned path is built).
        self._lva_jn_arc = None
        # C8 stagnation tracking
        self._proj_s_history = []  # rolling window of last N proj.s values
        self._stagnation_retract_remaining = 0

    def reset(self):
        super().reset()
        # Reset per-episode state. Junction arc is re-cached lazily
        # when _maybe_cache_junctions is called next.
        self._lva_jn_arc = None
        self._proj_s_history = []
        self._stagnation_retract_remaining = 0

    def _maybe_cache_junctions(self, base_env):
        if self._lva_jn_arc is not None:
            return
        try:
            junctions = base_env._path_context.get_path_junctions()
        except Exception:
            return
        self._lva_jn_arc = _find_junction_arc(junctions, "(18)", "LVA")
        try:
            base_env._heur_lva_jn_arc = (
                float(self._lva_jn_arc) if self._lva_jn_arc else -1.0
            )
        except Exception:
            pass

    def _detect_phase(self, base_env, variant_cfg) -> str:
        """Determine which phase fires this step.
        Phase A: trunk-top + (18) bridge crossing (window [lva_jn-30, lva_jn])
        Phase C: inside LVA daughter (s > lva_jn) — daughter sub-regime only
        Else: default heuristic (= upstream of Phase A window)."""
        self._maybe_cache_junctions(base_env)
        try:
            s = float(base_env._path_context.get_projection().s)
        except Exception:
            return "default"
        if self._lva_jn_arc is None:
            return "default"
        # Phase C — inside LVA daughter (variant-gated)
        c_before = variant_cfg.get("trigger_before")
        if c_before is not None and s > self._lva_jn_arc:
            return "lva_daughter"
        # Phase A — trunk-to-(18) bridge crossing (window before lva_jn)
        if (self._lva_jn_arc - PHASE_A_ARC_BEFORE) <= s <= self._lva_jn_arc:
            return "trunk_top"
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

        # LVA-specific: keep firing Phase C even when off-path, IF the wire's
        # projection arclength is past the LVA-jn (= inside the daughter).
        # This bypasses the default off-path retract handler that was
        # hammering wires backward every time cur_branch flickered to (19)
        # in the LCCA-LVA convergence region. Wire stays in Phase C with
        # dynamic_tangent target; body-shape physics decides tip motion.
        # Also sets `_heur_suppress_wrong_branch=True` on env so env5's
        # wrong_branch_timeout doesn't fire while the wire is intentionally
        # off-path in this override region.
        phase = "default"
        suppress_wbt = False
        if on_correct_path:
            phase = self._detect_phase(base_env, variant_cfg)
        else:
            try:
                s_now = float(base_env._path_context.get_projection().s)
                if (self._lva_jn_arc is not None
                        and s_now > self._lva_jn_arc):
                    c_before = variant_cfg.get("trigger_before")
                    if c_before is not None:
                        phase = "lva_daughter"
                        suppress_wbt = True
            except Exception:
                pass
        try:
            base_env._heur_suppress_wrong_branch = suppress_wbt
        except Exception:
            pass

        try:
            base_env._heur_lva_phase = phase
            base_env._heur_lva_variant = variant_id
            # env5 STEP-log writer reads `_heur_rva_phase` for the
            # phase= field; mirror LVA phase there too so logs show
            # the active phase regardless of which factory is in use.
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
            # Phase A: degenerate +z target, gw_trans=10. Closed-loop
            # returns ~0 rotation (target parallel to wire's long axis);
            # phase just keeps the wire ascending fast through the
            # trunk → (18) bridge crossing.
            gw_trans = PHASE_A_GW_TRANS
            target = PHASE_A_TARGET_DIR
        else:  # phase == "lva_daughter" (Phase C — only inside LVA daughter)
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
            base_env._heur_lva_target_z = float(target[2])
            base_env._heur_lva_target_x = float(target[0])
            base_env._heur_lva_target_y = float(target[1])
            base_env._heur_lva_gw_rot = float(gw_rot)
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


class LVAHeuristicActionFunctionFactory:
    """Pickleable factory for LVAHeuristicActionFunction."""

    def __init__(self, noise_std: float = 0.0, normalize_output: bool = True):
        self.noise_std = noise_std
        self.normalize_output = normalize_output

    def create(self, env) -> LVAHeuristicActionFunction:
        return LVAHeuristicActionFunction(
            env=env,
            noise_std=self.noise_std,
            normalize_output=self.normalize_output,
        )
