"""Heuristic policy wrapper for parallel heuristic seeding.

This module provides a way to use the CenterlineFollowerHeuristic
with the eve_rl worker infrastructure, enabling parallel heuristic
seeding instead of sequential.

Usage:
    from util.heuristic_policy import create_heuristic_action_function

    # In heatup, use heuristic instead of random:
    action_fn = create_heuristic_action_function(env)
    episode, steps = agent._play_episode(
        env=env,
        action_function=action_fn,
        consecutive_actions=1,
    )
"""

import numpy as np
from typing import Callable, Optional


class HeuristicActionFunction:
    """Wraps CenterlineFollowerHeuristic as an action function for workers.

    This class creates a heuristic controller from an environment and
    provides a callable interface compatible with _play_episode().

    The action function returns normalized actions in [-1, 1] when
    normalize_actions=True is used by the agent.
    """

    def __init__(
        self,
        env,
        noise_std: float = 0.0,
        normalize_output: bool = True,
    ):
        """Initialize heuristic action function.

        Args:
            env: The environment (must have .pathfinder and .intervention)
            noise_std: Standard deviation of Gaussian noise to add (default 0)
            normalize_output: If True, return actions in [-1, 1] range
        """
        from .heuristic_controller import CenterlineFollowerHeuristic

        # Unwrap gymnasium wrappers (e.g. ActionCurriculumWrapper) to access
        # pathfinder/intervention on the base BenchEnv4/5.
        base_env = getattr(env, 'unwrapped', env)

        self.env = env
        self.noise_std = noise_std
        self.normalize_output = normalize_output
        self._rng = np.random.default_rng()
        self._needs_reset = True  # Lazy reset flag

        # Create heuristic controller from the unwrapped environment
        self.heuristic = CenterlineFollowerHeuristic(
            pathfinder=base_env.pathfinder,
            intervention=base_env.intervention,
        )

        # Get action space bounds for normalization (same on wrapper and base)
        self.action_low = env.action_space.low.flatten().astype(np.float64)
        self.action_high = env.action_space.high.flatten().astype(np.float64)
        self.action_range = self.action_high - self.action_low

    def reset(self):
        """Mark for lazy reset — actual heuristic.reset() happens on the
        first __call__ AFTER _play_episode has called env.reset(), so the
        pathfinder already has fresh path data."""
        self._needs_reset = True

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Get action from heuristic controller.

        When on correct branch: use centerline-following heuristic.
        When off branch: use random retraction in [-10, 5] to pull back
        toward the correct branch (mirrors the RL action space lower bound).

        Args:
            obs: Observation (unused — branch state read from path_context directly)

        Returns:
            Action as flat numpy array, normalized to [-1, 1] if normalize_output=True
        """
        # Lazy reset: env.reset() has already been called by _play_episode,
        # so pathfinder.path_points_vessel_cs is current.
        if self._needs_reset:
            self.heuristic.reset()
            self._needs_reset = False

        # Check path membership from the shared path cache. RL_IMPROV_8 v2:
        # use is_on_correct_path() (cross-track + nearest-branch + on-path-mask)
        # so the off-path retract handler also fires when the wire is on the
        # trunk branch but laterally off the planned polyline (the arch
        # wedge case). Was reverted in v3/v4 due to thrashing on in-lumen
        # drift, but restored as the user's original intent.
        base_env = getattr(self.env, 'unwrapped', self.env)
        on_correct_path = True
        try:
            on_correct_path = base_env._path_context.is_on_correct_path()
        except Exception:
            pass

        if not on_correct_path:
            # Always-negative retraction: pull back toward bifurcation
            gw_trans = float(self._rng.uniform(-10.0, -1.0))
            cath_trans = gw_trans * self.heuristic.catheter_follow_ratio
            raw_action = np.array([gw_trans, 0.0, cath_trans, 0.0], dtype=np.float32)
        else:
            # Normal centerline-following (alternating trans/rot, RL_IMPROV_7 §7 Fix 10).
            # Fold brake (Fix 11): when the fold detector has fired >5 times,
            # the wire is coiling — force pure translate so the base stops
            # adding twist. Brake threshold lowered to >2 + d_corr stagnation
            # trigger (Fix 13) were tried in runs 15-16 and made things worse
            # (fold rate 83% -> 96%), so reverted to the run-14 config that
            # produced our only success (pid=350 ep=2).
            fold_count = getattr(base_env, "_fold_stall_count", 0)
            force_translate = fold_count > 5
            # RL_IMPROV_8 v2 (restored): heuristic regime metrics use the
            # graph-routed d_corr (accounts for sister-branch detours) +
            # daughter-only arclength. The routed version reports an
            # honestly large d_corr when the wire is in a wrong sister
            # branch (must retrace through bifurcations to reach the
            # actual ostium); arclength projection would misleadingly
            # report a near-zero value just because the projection lands
            # on a junction's arclength, even when the tip is far away.
            #   d_corr_mm     — graph-routed dist to NEXT daughter entry
            #   arc_past_mm   — arclength past most-recent DAUGHTER entry
            try:
                d_corr_mm = float(
                    base_env._path_context.get_routed_d_corr_to_next_daughter_entry()
                )
                arc_past_mm = float(
                    base_env._path_context.get_arclength_past_last_daughter_entry()
                )
            except Exception:
                d_corr_mm = float("inf")
                arc_past_mm = 0.0
            raw_action = self.heuristic.get_action(
                self._rng,
                force_translate=force_translate,
                d_corr_mm=d_corr_mm,
                arc_past_junction_mm=arc_past_mm,
            )

        # Publish heuristic internals onto the env so env5's STEP logger can
        # include them in INFO lines for offline analysis (Fix 14, 18).
        try:
            base_env._heur_heading_error = float(getattr(self.heuristic, "_last_heading_error", 0.0))
            base_env._heur_cross_track = float(getattr(self.heuristic, "_last_cross_track", 0.0))
            base_env._heur_arc_past_junction = float(getattr(self.heuristic, "_last_arc_past_junction", 0.0))
            # Identify which named daughter the wire is currently nearest to —
            # diagnostic to track whether rotations are steering the tip
            # toward the correct daughter or away from it.
            try:
                idx = base_env._path_context.get_nearest_named_branch_idx()
                if idx >= 0:
                    br = base_env._path_context._branches_tuple[idx]
                    name = getattr(br, "name", "") or ""
                    short = "none"
                    for tag in ("LCCA", "LVA", "RCCA", "RVA"):
                        if tag in name:
                            short = tag
                            break
                else:
                    short = "none"
            except Exception:
                short = "none"
            base_env._heur_nearest_named = short
        except Exception:
            pass

        # Ensure it's a flat numpy array
        action = np.asarray(raw_action).flatten().astype(np.float64)

        # Add noise if requested
        if self.noise_std > 0:
            noise = self._rng.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            # Clip to valid range
            action = np.clip(action, self.action_low, self.action_high)

        # Normalize to [-1, 1] if requested
        if self.normalize_output:
            action = 2 * (action - self.action_low) / self.action_range - 1
            action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)


def create_heuristic_action_function(
    env,
    noise_std: float = 0.0,
    normalize_output: bool = True,
) -> HeuristicActionFunction:
    """Factory function to create a heuristic action function for an environment.

    Args:
        env: The environment (must have .pathfinder and .intervention)
        noise_std: Standard deviation of Gaussian noise to add
        normalize_output: If True, return actions in [-1, 1] range

    Returns:
        HeuristicActionFunction instance
    """
    return HeuristicActionFunction(
        env=env,
        noise_std=noise_std,
        normalize_output=normalize_output,
    )


# For pickling support - workers need to recreate the heuristic
class HeuristicActionFunctionFactory:
    """Pickleable factory that creates HeuristicActionFunction in worker process.

    Since HeuristicActionFunction contains references to the environment's
    pathfinder and intervention, it can't be pickled directly. This factory
    is pickled instead and creates the action function in the worker.
    """

    def __init__(self, noise_std: float = 0.0, normalize_output: bool = True):
        self.noise_std = noise_std
        self.normalize_output = normalize_output

    def create(self, env) -> HeuristicActionFunction:
        """Create HeuristicActionFunction for the given environment."""
        return HeuristicActionFunction(
            env=env,
            noise_std=self.noise_std,
            normalize_output=self.normalize_output,
        )
