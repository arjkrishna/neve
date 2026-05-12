"""LCCA-specific heuristic policy — Plan v4 (post-smoke fix).

Mirrors heuristic_policy_rva.py's Phase A/B/C structure, but for the
LCCA daughter target. Path topology for an LCCA-target episode
(verified empirically from v1 smoke test trajectories):

  trunk_top:     (2)→(0)        same as RVA
  lcca_jn:       (0)→LCCA       wire transitions DIRECTLY from (0) into
                                LCCA daughter — there is no intermediate
                                bridge branch like RVA's (11). Only ONE
                                post-trunk junction on the LCCA path.

  Phase A — Trunk-top crossing: gw_trans=10, fixed -z, window
            [trunk_top - 30, trunk_top + 10]  (identical to RVA)

  Phase B — LCCA approach + entry turn: gw_trans=1.5, dynamic tangent
            at s+5, window [lcca_jn - 15, lcca_jn + 5]. Anchored at
            the (0)→LCCA junction. This is where the wire transitions
            from (0) into the LCCA daughter; the dynamic tangent +
            original sign drives the J-tip into LCCA at the bifurcation.

  Phase C — LCCA in-daughter advance: gw_trans=2.0, dynamic tangent
            at s+5, window s > lcca_jn + 5. Slower speed to track
            LCCA's curving centerline.

Sign convention: original right-hand rule (same as RVA's working
configuration). The reasoning: closed-loop rotation aligns J-tip's
bend with the *target tangent*. Since the target tangent here is
computed against the LCCA-target planned path, the original sign
drives the J-tip toward LCCA — same logic as RVA's Phase B/C, just
applied to a different daughter.

Fallback: if v1 (this file) reaches LCCA at < 80%, build v2 that
literally replicates v6 — feeds an RVA-target tangent through a
flipped-sign rotation to reproduce the empirical 80% deflection.
"""
import numpy as np

from .heuristic_policy import HeuristicActionFunction


# ============================================================================
# Phase A — Trunk-top crossing (identical to RVA)
# ============================================================================
PHASE_A_ARC_BEFORE = 30.0
PHASE_A_ARC_AFTER = 10.0
PHASE_A_GW_TRANS = 10.0
PHASE_A_TARGET_DIR = np.array([0.0, 0.0, -1.0])

# ============================================================================
# Phase B — LCCA approach + entry turn. Active rotation: target =
# unit vector FROM current tip TO a deep LCCA waypoint (LCCA[~10], =
# lcca_jn_arc + ~25 mm). Wire's J-tip rotates to point into the LCCA
# daughter regardless of wire's position in the bif2 cavity.
#   * window = [lcca_jn - 15, lcca_jn + 5] (lead-in + first 5 mm into entry)
#   * gw_trans = 1.5 (slow deliberate, identical to RVA's Phase B)
# Rationale: bif2 cavity has 12 mm radius and LCCA mouth is at low
# x/y (anti-inertia direction). +z target was degenerate, dynamic
# tangent was sparse. A position-vector target gives a well-defined
# rotation regardless of where the wire is in the cavity.
# ============================================================================
PHASE_B_ARC_BEFORE = 15.0
PHASE_B_ARC_AFTER = 5.0     # Fix #3 attempted (=30 mm) caused wire to
                            #   over-rotate and drift backward past LCCA
                            #   entry (z=374). Reverted; position-vector
                            #   target works for short window only.
PHASE_B_GW_TRANS = 1.5
PHASE_B_WAYPOINT_OFFSET_MM = 25.0   # target = polyline at lcca_jn + 25 mm

# ============================================================================
# Stuck-detection + recovery (active only in Phase C / lcca_daughter)
# Detects wedged wires (tip not moving) and applies one of two recovery
# strategies at random:
#   * "retract" — pull catheter back while pushing wire, exposing more
#     free wire body so the J-tip can swing into the daughter mouth
#   * "burst"   — high gw_trans for several steps; stored elastic energy
#     can buckle the wire body past a wall wedge
# ============================================================================
STUCK_WINDOW_STEPS = 10        # rolling window of tip positions
STUCK_THRESHOLD_MM = 1.0       # stuck if max distance from window mean < this
RECOVERY_STEPS = 5             # apply recovery action for N steps
RECOVERY_RETRACT_GW = 2.0      # gw_trans during retract recovery
RECOVERY_RETRACT_CATH_RATIO = -2.0  # cath_trans = -2 * gw_trans (catheter back)
RECOVERY_BURST_GW = 10.0       # gw_trans during burst recovery
RECOVERY_BURST_CATH_RATIO = 0.8     # cath_trans = 0.8 * gw_trans (default)
# Fix #2 (Phase A taper) was tried and reverted: it gave the J-tip too
# much rotation budget during Phase A's last 10 mm, causing wires to
# over-rotate and slip backward (z=370, below LCCA entry).

# ============================================================================
# Phase C — LCCA in-daughter advance
# ============================================================================
PHASE_C_DAUGHTER_GW_TRANS = 2.0
PHASE_C_LOOKAHEAD_MM = 5.0

# Rotation control gains (same as RVA)
GW_ROT_GAIN = 1.0
GW_ROT_MAX = 1.5
CATH_FOLLOW_RATIO = 0.8


# ============================================================================
# Helpers (verbatim from heuristic_policy_rva.py)
# ============================================================================
def _compute_rotation_to_target(tracking_vcs: np.ndarray,
                                target_dir: np.ndarray) -> float:
    """Closed-loop gw_rot to align J-tip's bend with target_dir.

    Sign convention: original right-hand rule (validated for RVA;
    do NOT flip — flipping was v6/v7's failure mode for RVA-target,
    which is what we're exploiting INTO LCCA via target redirection
    rather than sign flip).
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


def _dynamic_tangent_target(base_env, s, lookahead_mm):
    """Planned-path tangent at (s + lookahead). Returns 3-vector or zeros."""
    try:
        pc = base_env._path_context
        t = pc.get_planned_path_tangent_at(s, lookahead_mm)
        if float(np.linalg.norm(t)) < 1e-6:
            return np.zeros(3)
        return np.asarray(t, dtype=float)
    except Exception:
        return np.zeros(3)


def _path_point_at_arc(pc, target_arc: float):
    """3D position on the planned polyline at arclength ``target_arc``.
    Linear interpolation within the segment containing target_arc.
    Returns None if the polyline is degenerate."""
    if pc._total_length <= 0 or len(pc._polyline) < 2:
        return None
    target_arc = float(np.clip(target_arc, 0.0, pc._total_length))
    idx = int(np.searchsorted(pc._cumlen, target_arc) - 1)
    idx = max(0, min(idx, len(pc._polyline) - 2))
    seg_start = float(pc._cumlen[idx])
    seg_end = float(pc._cumlen[idx + 1])
    seg_len = seg_end - seg_start
    if seg_len < 1e-9:
        return np.asarray(pc._polyline[idx], dtype=float)
    t = (target_arc - seg_start) / seg_len
    p0 = np.asarray(pc._polyline[idx], dtype=float)
    p1 = np.asarray(pc._polyline[idx + 1], dtype=float)
    return p0 + t * (p1 - p0)


def _position_vector_target(base_env, target_pos_v):
    """Unit vector from current tip (vessel-CS) to ``target_pos_v``.
    Returns zero vector if anything goes wrong."""
    try:
        from eve.util.coordtransform import tracking3d_to_vessel_cs
        fluoro = base_env.intervention.fluoroscopy
        tip_eve = fluoro.tracking3d[0]
        tip_v = np.asarray(
            tracking3d_to_vessel_cs(
                tip_eve, fluoro.image_rot_zx, fluoro.image_center
            ),
            dtype=float,
        )
        d = np.asarray(target_pos_v, dtype=float) - tip_v
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return np.zeros(3)
        return d / n
    except Exception:
        return np.zeros(3)


# ============================================================================
# Action function class
# ============================================================================
class LCCAHeuristicActionFunction(HeuristicActionFunction):
    """LCCA-target heuristic with arclength-based phase override at the
    trunk-top, LCCA-bridge entry, and LCCA daughter junctions.

    Junction tokens (cached lazily once the planned path is built):
      _trunk_top_arc = arc of (2)→(0)
      _lcca_jn_arc   = arc of (0)→LCCA   [= Phase B + Phase C anchor]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trunk_top_arc = None
        self._lcca_jn_arc = None
        # Stuck-detection state (per-episode)
        self._tip_history = []           # rolling window of tip3d (vessel-CS)
        self._recovery_remaining = 0     # >0 = recovery action active
        self._recovery_mode = None       # 'retract' | 'burst' | None

    def reset(self):
        super().reset()
        self._trunk_top_arc = None
        self._lcca_jn_arc = None
        self._tip_history = []
        self._recovery_remaining = 0
        self._recovery_mode = None

    def _is_stuck(self, tip_v) -> bool:
        """Track tip3d in vessel-CS. Returns True iff stuck (no motion)."""
        if tip_v is None:
            return False
        self._tip_history.append(np.asarray(tip_v, dtype=float))
        if len(self._tip_history) > STUCK_WINDOW_STEPS:
            self._tip_history.pop(0)
        if len(self._tip_history) < STUCK_WINDOW_STEPS:
            return False
        arr = np.stack(self._tip_history, axis=0)
        center = arr.mean(axis=0)
        max_d = float(np.max(np.linalg.norm(arr - center, axis=1)))
        return max_d < STUCK_THRESHOLD_MM

    def _maybe_cache_junctions(self, base_env):
        if (self._trunk_top_arc is not None
                and self._lcca_jn_arc is not None):
            return
        try:
            junctions = base_env._path_context.get_path_junctions()
        except Exception:
            return
        self._trunk_top_arc = _find_junction_arc(junctions, "(2)", "(0)")
        self._lcca_jn_arc = _find_junction_arc(junctions, "(0)", "LCCA")
        try:
            base_env._heur_lcca_trunk_top_arc = (
                float(self._trunk_top_arc) if self._trunk_top_arc else -1.0
            )
            base_env._heur_lcca_jn_arc = (
                float(self._lcca_jn_arc) if self._lcca_jn_arc else -1.0
            )
        except Exception:
            pass

    def _detect_phase(self, base_env) -> str:
        """Phase priority: Phase C → Phase B → Phase A → default.

        Fix #1: Phase B is checked BEFORE Phase A. When the windows
        overlap (Phase A ends at trunk_top+10, Phase B starts at
        lcca_jn-15, and (0) is short so they often overlap), Phase B
        wins. Empirically gave best target-reach rate (4% vs 0% for
        baseline). Tradeoff: ~30% of wires deflect at trunk-top
        because Phase A's strong -z push gets shortened.
        """
        self._maybe_cache_junctions(base_env)
        try:
            s = float(base_env._path_context.get_projection().s)
        except Exception:
            return "default"
        # Phase C — past LCCA junction by ARC_AFTER mm, in-daughter advance
        if (self._lcca_jn_arc is not None
                and s > self._lcca_jn_arc + PHASE_B_ARC_AFTER):
            return "lcca_daughter"
        # Phase B — LCCA approach + entry turn (CHECKED BEFORE Phase A; wins
        # in the window-overlap region, freeing Phase A to fire only upstream)
        if (self._lcca_jn_arc is not None
                and (self._lcca_jn_arc - PHASE_B_ARC_BEFORE)
                <= s
                <= (self._lcca_jn_arc + PHASE_B_ARC_AFTER)):
            return "lcca_bridge"
        # Phase A — trunk-top window (only fires when not in Phase B/C)
        if (self._trunk_top_arc is not None
                and (self._trunk_top_arc - PHASE_A_ARC_BEFORE)
                <= s
                <= (self._trunk_top_arc + PHASE_A_ARC_AFTER)):
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
            base_env._heur_lcca_phase = phase
            # env5's STEP-log writer reads `_heur_rva_phase` for the
            # phase= field; route the LCCA phase there too so logs are
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
        elif phase == "lcca_bridge":
            # Phase B — position-vector target: aim J-tip from current
            # position toward a fixed LCCA waypoint deep enough to be
            # past the sparse cavity segment. This rotates the J-tip
            # toward LCCA's mouth even though the wire's inertial
            # direction (+x +y +z) is opposite to LCCA's mouth (low x,
            # low y). Adapts as wire moves through the cavity.
            gw_trans = PHASE_B_GW_TRANS
            try:
                pc = base_env._path_context
                target_arc = float(self._lcca_jn_arc) + PHASE_B_WAYPOINT_OFFSET_MM
                waypoint_v = _path_point_at_arc(pc, target_arc)
                if waypoint_v is not None:
                    target = _position_vector_target(base_env, waypoint_v)
                else:
                    target = np.zeros(3)
            except Exception:
                target = np.zeros(3)
        else:  # phase == "lcca_daughter"
            # Phase C target: dynamic tangent at s+5 along the planned-path
            # polyline (RVA-style). Adaptive position-vector and stuck-recovery
            # were both tested and didn't beat this baseline.
            gw_trans = PHASE_C_DAUGHTER_GW_TRANS
            try:
                s_now = float(base_env._path_context.get_projection().s)
            except Exception:
                s_now = 0.0
            target = _dynamic_tangent_target(
                base_env, s_now, PHASE_C_LOOKAHEAD_MM
            )
            try:
                base_env._heur_lcca_recovery = "none"
            except Exception:
                pass

        gw_rot = _compute_rotation_to_target(tracking_vcs, target)

        try:
            base_env._heur_lcca_target_x = float(target[0])
            base_env._heur_lcca_target_y = float(target[1])
            base_env._heur_lcca_target_z = float(target[2])
            base_env._heur_lcca_gw_rot = float(gw_rot)
        except Exception:
            pass

        # Recovery override is disabled in this test — _recovery_remaining
        # stays 0 because stuck-detection no longer initiates recovery.
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


class LCCAHeuristicActionFunctionFactory:
    """Pickleable factory for LCCAHeuristicActionFunction."""

    def __init__(self, noise_std: float = 0.0, normalize_output: bool = True):
        self.noise_std = noise_std
        self.normalize_output = normalize_output

    def create(self, env) -> LCCAHeuristicActionFunction:
        return LCCAHeuristicActionFunction(
            env=env,
            noise_std=self.noise_std,
            normalize_output=self.normalize_output,
        )
