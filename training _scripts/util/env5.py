"""BenchEnv5 — Optimized env4 with shared projection cache and logging fixes.

Key differences from env4.py:
  - PathProjectionCache shared between ArcLengthProgress and LocalGuidance
    (eliminates duplicate polyline projection + coordinate transform per step)
  - on_correct_branch uses cross_track_dist threshold instead of O(B*N) branch scan
  - Pre-computed 2D tangents at reset
  - Log handler flush only on INFO-level log steps (every 50 or terminal)
  - Action string formatting only when logging at INFO level
"""

import eve
import eve.visualisation
import time
import logging
import re
import sys
import os
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eve.util.pathcontext import (
    PathProjectionCache,
    make_per_target_caches,
    classify_physical_branch,
)


# Plan v12 — daughter short-tag derivation. Used to populate
# `info["final_branch_short"]` so per-Episode downstream filtering
# (Stage 3 + manual heatup inspection) can identify WHICH daughter the
# wire actually ended in — distinct from `target_branch_short` which
# is WHICH daughter this virtual env was tracking. Allows queries like
# "RCCA-target episodes where wire actually went to LCCA" for failure
# analysis.
DAUGHTER_TAGS: Tuple[str, ...] = ("RCCA", "LCCA", "RVA", "LVA")


def _branch_short_from_name(branch_name: Optional[str]) -> str:
    """Map a branch name string (e.g. "Centerline curve - RCCA.mrk") to a
    short daughter tag. Returns "other" for trunk / bridge / unnamed
    branches, "unknown" when the input is None."""
    if not isinstance(branch_name, str):
        return "unknown"
    for tag in DAUGHTER_TAGS:
        if tag in branch_name:
            return tag
    return "other"


def _final_branch_short(path_context) -> str:
    """Look up the wire's current branch's short daughter tag from the
    PathContext state machine. Used at every-step info population so the
    terminal step's info carries `final_branch_short`."""
    try:
        idx = path_context._current_branch_idx
        if idx is None:
            return "unknown"
        branches_tuple = path_context._branches_tuple
        if idx >= len(branches_tuple):
            return "unknown"
        branch = branches_tuple[idx]
        return _branch_short_from_name(getattr(branch, "name", None))
    except Exception:
        return "unknown"

# ---------------------------------------------------------------------------
# Heuristic-mode detector thresholds
# ---------------------------------------------------------------------------
OFF_BRANCH_GRACE_STEPS = 50  # was 20; bumped in RL_IMPROV_7 §7 Fix 3 — with
                              # is_on_correct_branch() hysteresis (§7 Fix 2)
                              # the counter no longer resets on spurious flips,
                              # so 20 was too short for retract recovery from
                              # bif2 wrong branches (~50 steps of retract needed).
# Plan v13 (obs-v3 / recovery-enablement) — env-var override so per-run
# experiments can extend the recovery horizon WITHOUT code edits. Rationale
# (fable.md concern #1): ArcLengthProgress already pays +reward for off-path
# retraction, but at ~1.3 mm/step retract vs up-to-50 mm off-path depth the
# 50-step grace truncates (-5) before a deep recovery can physically finish,
# so online exploration never completes a recovery for AWAC to clone.
# Launchers set e.g. `-e EVE_OFF_BRANCH_GRACE_STEPS=150`. Default unchanged.
OFF_BRANCH_GRACE_STEPS = int(
    os.environ.get("EVE_OFF_BRANCH_GRACE_STEPS", OFF_BRANCH_GRACE_STEPS)
)
OFF_BRANCH_MIN_INSERTED_MM = 0.0  # was 50.0 workaround; now using true branch membership
FAILURE_TRUNCATION_PENALTY = -5.0
# RL_IMPROV_8 OST — overshoot penalty. When the 50-step off-path timeout fires
# but the wire's TIP is physically still inside the CORRECT target daughter
# (classify_physical_branch == target tag), the wire overshot the target INSIDE
# the right daughter rather than diverging into a wrong branch → soft -1, not -5.
OVERSHOOT_PENALTY = -1.0
# Plan v5 — WRONG_BRANCH_ENTRY_PENALTY and WRONG_BRANCH_STEP_PENALTY removed.
# Off-path signal is now carried by ArcLengthProgress's symmetric per-step
# Δoff_arc shaping (penalizes deeper drift, rewards retract — matches the
# heuristic's implicit retract-when-off-path objective). Discrete signals
# at daughter forks come from state-machine ±1 commit events. The 50-step
# wrong_branch_timeout truncation (still active below) gives -5 via
# FAILURE_TRUNCATION_PENALTY when wire is hopelessly off-path.
# CORRECT_ENTRY_REWARD also removed — daughter commits now drive ±1 via
# the state machine event queue (drained in step()), latched once per fork.
FOLD_STALL_STEPS = 20          # kill stuck wires quickly to speed up cycle
# Plan v13 (fold-recovery enablement) — env-var override, same pattern as
# EVE_OFF_BRANCH_GRACE_STEPS. Rationale: the lcca_awac_obsv3 evals showed the
# post-obs-v3 policy aims 100% at the correct bridge corridor but 85-93% of
# eval episodes die as wire_fold_stall at p50=80 steps — the 20-step fold
# guillotine truncates before a pull-back+re-push fold recovery (the standard
# IR maneuver) can complete, so the policy can never learn it. Launchers set
# e.g. `-e EVE_FOLD_STALL_STEPS=60`. Default unchanged.
FOLD_STALL_STEPS = int(os.environ.get("EVE_FOLD_STALL_STEPS", FOLD_STALL_STEPS))
FOLD_INSERTION_MM = 0.5        # min commanded gw insertion per step to count as inserting
FOLD_ARCLENGTH_MM = 0.5        # min tip arclength progress per step to count as advancing


def setup_step_logger(name="step_logger"):
    """Create a logger that flushes immediately after each log message."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    log_dir = os.environ.get("STEP_LOG_DIR", "/tmp")
    log_file = os.path.join(log_dir, f"worker_{os.getpid()}.log")

    try:
        handler = logging.FileHandler(log_file, mode="a")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s,%(msecs)03d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s,%(msecs)03d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_formatter = logging.Formatter("STEP: %(message)s")
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)

    return logger


# ============================================================================
# Plan v12 redesign — coord-based TargetReached reward + terminal.
# ============================================================================
# eve.reward.TargetReached / eve.terminal.TargetReached read the SHARED
# intervention.target.reached. Under MultiTargetEnv5, the 4 graders share one
# intervention but each grades against its OWN daughter target. These coord
# variants check dist(tip, frozen_target_coord3d) < threshold instead, so a
# grader's success/terminal reflects ITS daughter without mutating the shared
# intervention.target. The frozen coord + threshold are updated each episode
# in BenchEnv5.reset() when target_coord3d mode is active.

class _CoordTargetReachedReward(eve.reward.TargetReached):
    """TargetReached reward against a frozen target_coord3d (not the shared
    intervention.target). factor=3.0 same as the env default."""

    def __init__(self, intervention, factor: float = 3.0):
        super().__init__(intervention, factor=factor,
                         final_only_after_all_interim=False)
        self.target_coord3d = None
        self.threshold = None

    def step(self) -> None:
        if self.target_coord3d is None or self.threshold is None:
            # Fall back to shared-target behavior (single-target / not yet set).
            super().step()
            return
        tip = np.asarray(self.intervention.fluoroscopy.tracking3d[0],
                         dtype=np.float64)
        reached = float(
            np.linalg.norm(tip - self.target_coord3d) < self.threshold
        )
        self.reward = self.factor * reached


class _CoordTargetReachedTerminal(eve.terminal.TargetReached):
    """TargetReached terminal against a frozen target_coord3d."""

    def __init__(self, intervention):
        super().__init__(intervention)
        self.target_coord3d = None
        self.threshold = None

    @property
    def terminal(self) -> bool:
        if self.target_coord3d is None or self.threshold is None:
            return self.intervention.target.reached
        tip = np.asarray(self.intervention.fluoroscopy.tracking3d[0],
                         dtype=np.float64)
        return bool(np.linalg.norm(tip - self.target_coord3d) < self.threshold)


class BenchEnv5(eve.Env):
    def __init__(
        self,
        intervention: eve.intervention.SimulatedIntervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps=600,
        default_target_branch: str = None,
        target_coord3d=None,
    ) -> None:
        self.mode = mode
        self.visualisation = visualisation
        # Plan v12 redesign — when set, this grader grades against a FROZEN
        # target_coord3d (sampled per-episode from its daughter centerline)
        # instead of the shared intervention.target. Flows to Target2D, the
        # FixedPathfinder planned path, and the coord-based TargetReached
        # reward + terminal. None = legacy single-target (reads
        # intervention.target). Used by MultiTargetEnv5 to make all 4 graders
        # run the IDENTICAL pipeline against their own targets.
        self._grader_target_coord3d = (
            np.asarray(target_coord3d, dtype=np.float64)
            if target_coord3d is not None else None
        )
        self._is_grader = target_coord3d is not None
        # Plan v5 — per-daughter RL: when set, every reset() that does not
        # explicitly pass a `target_branch` option falls back to this
        # branch. This scopes heatup / explore / eval (which the eve_rl
        # runner resets with no options) to a single daughter, so RCCA-only
        # training actually trains and evaluates on RCCA. Heuristic seeding
        # passes target_branch explicitly via its schedule, so it is
        # unaffected. None = legacy multi-daughter behavior (random target).
        self._default_target_branch = default_target_branch

        # Step-level logging
        self._step_logger = setup_step_logger(f"step_logger_{mode}_{id(self)}")
        self._step_count = 0
        self._episode_count = 0
        self._episode_step_count = 0
        self._last_step_time = None
        self._episode_start_time = None
        self._episode_total_reward = 0.0
        self._prev_inserted = [0.0, 0.0]
        self._step_logger.info(
            f"=== BenchEnv5 initialized (mode={mode}, visualisation={visualisation}) ==="
        )
        sys.stderr.flush()

        # Start condition
        start = eve.start.InsertionPoint(intervention)

        # Pathfinder — fixed path (computed once per episode at reset)
        pathfinder = eve.pathfinder.FixedPathfinder(intervention=intervention)

        # Shared projection cache — eliminates duplicate work between
        # ArcLengthProgress and LocalGuidance
        self._path_context = PathProjectionCache(pathfinder, intervention)

        # ----------------------------------------------------------------
        # Observation
        # ----------------------------------------------------------------
        # Plan v5 — wire tracking: 10 points at 15 mm spacing = 150 mm of
        # distal wire (covers the full bif2 region, the most physics-rich
        # zone). Frame-stacked 2 deep via the Memory wrapper so the agent
        # can derive the wire's ACTUAL kinematic velocity by position
        # finite-difference — the dynamics state SOFA does not expose.
        # FILL reset mode seeds both frames with the reset state, so
        # velocity reads 0 at episode start (correct for a stationary
        # wire). Memory wraps the per-episode normalizer (each frame
        # normalized, then stacked).
        tracking = eve.observation.Tracking2D(intervention, n_points=10, resolution=15)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking = eve.observation.wrapper.Memory(
            tracking,
            n_steps=2,
            reset_mode=eve.observation.wrapper.MemoryResetMode.FILL,
        )

        # Plan v12 — grader mode: Target2D reads the frozen target_coord3d
        # so obs[40:42] is target-coherent for THIS grader's daughter.
        # Keep a ref to the INNER Target2D so set_grader_target() can update
        # its coord per episode (it's wrapped below).
        _inner_target2d = eve.observation.Target2D(
            intervention,
            target_coord3d=self._grader_target_coord3d,
        )
        self._grader_target2d_inner = _inner_target2d if self._is_grader else None
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            _inner_target2d, intervention
        )

        last_action = eve.observation.LastAction(intervention)
        last_action = eve.observation.wrapper.Normalize(last_action)

        guidance = eve.observation.LocalGuidance(
            intervention=intervention,
            pathfinder=pathfinder,
            path_context=self._path_context,
        )

        # Plan v5 — total inserted length of each device (guidewire +
        # catheter). Removes the integration-history non-Markov gap: two
        # wires with identical tracked-point positions but different total
        # insertion respond differently to a retract command. Single-frame
        # (no Memory) — insertion velocity is already covered by
        # last_action's gw_trans / cath_trans commands.
        inserted_lengths = eve.observation.InsertionLengths(intervention)
        inserted_lengths = eve.observation.wrapper.Normalize(inserted_lengths)

        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
                "guidance": guidance,
                "inserted_lengths": inserted_lengths,
            }
        )

        # ----------------------------------------------------------------
        # Reward
        # ----------------------------------------------------------------
        # Plan v12 — grader mode uses the coord-based TargetReached so the
        # +3.0 success reward fires on THIS grader's daughter, not the shared
        # intervention.target. Single-target keeps the original behavior.
        if self._is_grader:
            target_reward = _CoordTargetReachedReward(intervention, factor=3.0)
            self._grader_target_reward = target_reward
        else:
            target_reward = eve.reward.TargetReached(
                intervention,
                factor=3.0,
                final_only_after_all_interim=False,
            )
            self._grader_target_reward = None
        # Plan v9 Change 4 — the uniform -0.001 step penalty is replaced by
        # an inline path-segment-conditioned per-step term added in
        # BenchEnv5.step() after update_branch_state. The eve.reward.Step
        # primitive is no longer part of the Combination.
        arc_progress = eve.reward.ArcLengthProgress(
            intervention=intervention,
            pathfinder=pathfinder,
            progress_factor=0.01,
            lateral_penalty_factor=0.001,
            path_context=self._path_context,
        )
        reward = eve.reward.Combination([target_reward, arc_progress])

        # ----------------------------------------------------------------
        # Terminal and Truncation
        # ----------------------------------------------------------------
        if self._is_grader:
            terminal = _CoordTargetReachedTerminal(intervention)
            self._grader_target_terminal = terminal
        else:
            terminal = eve.terminal.TargetReached(intervention)
            self._grader_target_terminal = None

        max_steps = eve.truncation.MaxSteps(n_max_steps)
        vessel_end = eve.truncation.VesselEnd(intervention)
        sim_error = eve.truncation.SimError(intervention)

        if mode == "train":
            truncation = eve.truncation.Combination(
                [max_steps, vessel_end, sim_error]
            )
        else:
            truncation = max_steps

        # ----------------------------------------------------------------
        # Info
        # ----------------------------------------------------------------
        target_reached = eve.info.TargetReached(intervention, name="success")
        path_ratio = eve.info.PathRatio(pathfinder)
        steps = eve.info.Steps()
        trans_speed = eve.info.AverageTranslationSpeed(intervention)
        trajectory_length = eve.info.TrajectoryLength(intervention)
        info = eve.info.Combination(
            [target_reached, path_ratio, steps, trans_speed, trajectory_length]
        )

        # ----------------------------------------------------------------
        # Visualisation
        # ----------------------------------------------------------------
        if visualisation:
            intervention.make_non_mp()
            visu = eve.visualisation.SofaPygame(intervention)
        else:
            intervention.make_non_mp()
            visu = None

        # Store component references for post-step reward shaping
        self._target_terminal = terminal
        self._max_steps_trunc = max_steps
        self._vessel_end_trunc = vessel_end
        self._sim_error_trunc = sim_error

        # Detector state (reset each episode in reset())
        self._heuristic_mode = False
        self._heuristic_abort_reason = None
        self._off_branch_steps = 0
        self._overshoot_truncation = False  # RL_IMPROV_8 OST — set at WBT trigger
        self._fold_stall_count = 0
        self._prev_tip_s = 0.0
        self._prev_inserted_gw = 0.0

        super().__init__(
            intervention,
            observation,
            reward,
            terminal,
            truncation=truncation,
            start=start,
            pathfinder=pathfinder,
            visualisation=visu,
            info=info,
            interim_target=None,
        )

    # ------------------------------------------------------------------
    # Plan v12 — grader-mode per-episode target setter.
    # ------------------------------------------------------------------
    def set_grader_target(self, target_coord3d, threshold) -> None:
        """Update this grader's frozen target for the upcoming episode.

        Multi-target heatup: MultiTargetEnv5 samples a fresh point on each
        daughter's centerline per episode and calls this BEFORE the grader's
        reset(), so the pathfinder (planned path), the coord-based
        TargetReached reward + terminal, and the Target2D obs all grade
        against the same daughter point. No-op for non-grader (single-target)
        instances.
        """
        if not self._is_grader:
            return
        coord = np.asarray(target_coord3d, dtype=np.float64)
        thr = float(threshold)
        self._grader_target_coord3d = coord
        # Pathfinder picks this up in reset() (positional eve.Env.reset call).
        self.pathfinder._grader_target_coord3d = coord
        if self._grader_target_reward is not None:
            self._grader_target_reward.target_coord3d = coord
            self._grader_target_reward.threshold = thr
        if self._grader_target_terminal is not None:
            self._grader_target_terminal.target_coord3d = coord
            self._grader_target_terminal.threshold = thr
        if self._grader_target2d_inner is not None:
            self._grader_target2d_inner.target_coord3d = coord

    def __setstate__(self, state):
        """Re-initialize components after unpickling in a worker process."""
        self.__dict__.update(state)
        self._step_logger = setup_step_logger(f"step_logger_{self.mode}_{id(self)}")

        # Recreate PathProjectionCache if it was lost during pickling
        # (PathProjectionCache.__reduce__ returns None to skip config serialization)
        if self._path_context is None:
            self._path_context = PathProjectionCache(self.pathfinder, self.intervention)

            # Update references in observation components
            if hasattr(self.observation, 'observations'):
                for obs in self.observation.observations.values():
                    if hasattr(obs, '_path_context'):
                        obs._path_context = self._path_context

            # Update references in reward components
            if hasattr(self.reward, 'rewards'):
                for rew in self.reward.rewards:
                    if hasattr(rew, '_path_context'):
                        rew._path_context = self._path_context

    def reset(self, seed=None, options=None):
        # Plan v5 — per-daughter RL: inject the default target branch when
        # the caller passes no explicit target. The eve_rl runner resets
        # the env with no options for heatup / explore / eval, so without
        # this the agent would explore + be evaluated across all 4
        # daughters even for RCCA-only training. Heuristic seeding passes
        # target_branch via its schedule, so it is unaffected. No-op when
        # _default_target_branch is None (legacy multi-daughter mode).
        if getattr(self, "_default_target_branch", None):
            if options is None:
                options = {"target_branch": self._default_target_branch}
            elif "target_branch" not in options:
                options = {**options, "target_branch": self._default_target_branch}

        if self._episode_count > 0 and self._episode_start_time is not None:
            episode_duration = time.time() - self._episode_start_time
            heur_end_str = ""
            if self._heuristic_mode:
                _abort = self._heuristic_abort_reason or "none"
                heur_end_str = f" | heur_abort={_abort}"
            self._step_logger.info(
                f"EPISODE_END | ep={self._episode_count} | steps={self._episode_step_count} | "
                f"total_reward={self._episode_total_reward:.4f} | duration={episode_duration:.2f}s | "
                f"avg_step_time={episode_duration/max(1,self._episode_step_count):.3f}s | "
                f"wall_time={time.time():.6f} | pid={os.getpid()}"
                f"{heur_end_str}"
            )
            # Plan v12 harvest — consolidated per-episode OUTCOME line carrying
            # EVERY field needed to filter the heatup npzs for AWAC later. These
            # mirror experience_cache.episode_metadata EXACTLY (same final_branch
            # via classify, same is_clean formula), so filtering from logs and
            # from the npz agree. _last_reason / _last_final_branch_short are
            # snapshotted at the terminal STEP (the shared wire moves on for an
            # early-terminating grader, so they cannot be recomputed here).
            try:
                _tbs = str(getattr(self, "_target_branch_short", "unknown"))
                _fbs = str(getattr(self, "_last_final_branch_short", "unknown"))
                _rcd = bool(getattr(self, "_received_correct_daughter", False))
                _rwd = bool(getattr(self, "_received_wrong_daughter", False))
                _ovs = bool(getattr(self, "_overshoot_truncation", False))
                _gsucc = bool(
                    getattr(getattr(self, "_target_terminal", None), "terminal", False)
                )
                _gtimeout = bool(
                    (self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS and not _ovs)
                    or self._fold_stall_count >= FOLD_STALL_STEPS
                )
                _clean = bool(
                    _fbs == _tbs
                    and _fbs not in ("unknown", "other")
                    and _rcd
                    and not _gtimeout
                )
                _reason = str(getattr(self, "_last_reason", "unknown"))
                # Plan v13 — mode=train|eval tag: train and eval envs of the
                # same worker interleave in one worker_<pid>.log with no
                # discriminator, which forced timestamp-window hacks when
                # isolating eval episodes (the eval-#1 forensic). Now explicit.
                self._step_logger.info(
                    f"EPISODE_OUTCOME | ep={self._episode_count} | "
                    f"mode={self.mode} | "
                    f"target_branch={_tbs} | final_branch={_fbs} | reason={_reason} | "
                    f"is_clean={int(_clean)} | grader_success={int(_gsucc)} | "
                    f"grader_timeout={int(_gtimeout)} | overshoot={int(_ovs)} | "
                    f"received_correct={int(_rcd)} | received_wrong={int(_rwd)} | "
                    f"return={self._episode_total_reward:.4f} | "
                    f"steps={self._episode_step_count} | pid={os.getpid()}"
                )
            except Exception:
                pass
            sys.stderr.flush()

        self._episode_count += 1
        self._episode_step_count = 0
        self._episode_total_reward = 0.0
        self._episode_start_time = time.time()
        self._last_step_time = time.time()
        self._prev_inserted = [0.0, 0.0]

        # Capture seed for reproducibility (used by heuristic_policy to
        # derive its own deterministic per-episode RNG, and logged in
        # EPISODE_START so post-hoc analysis can match log entries to
        # the schedule's (seed, options) tuples).
        self._reset_seed = seed

        # RVA-checkpoint trigger state (RL_IMPROV_8 §25). Reset per
        # episode; flips True the first step where the wire's projection
        # arclength crosses rva_jn_arc, at which point we save a SOFA
        # state checkpoint to RVA_CHECKPOINT_DIR.
        self._rva_jn_arc = None
        self._crossed_rva_jn = False

        # Phase C variant ID (RL_IMPROV_8 §28 / Plan v3). The
        # heuristic_policy_rva.RVAHeuristicActionFunction reads this to
        # dispatch among C0-C8 strategies. Stored on env so the action
        # function can read it without modifying its own factory layer.
        self._phase_c_variant = (
            options.get("phase_c_variant") if options else None
        ) or "C2"  # default = current production (dynamic, lookahead=5)

        # Detector state reset
        self._heuristic_mode = bool(options and options.get("heuristic_mode", False))
        self._heuristic_abort_reason = None

        # RL_IMPROV_8: stash the target branch tag (LCCA/LVA/RCCA/RVA) for
        # snapshot bucketing and STEP-log diagnostics. The schedule passes
        # `target_branch="Centerline curve - LCCA.mrk"` (etc.) via options;
        # extract the short tag. Falls back to "unknown" if not provided
        # (e.g. plain training without an explicit schedule).
        self._target_branch_short = "unknown"
        if options is not None:
            tb_full = options.get("target_branch")
            if isinstance(tb_full, str):
                for tag in ("LCCA", "LVA", "RCCA", "RVA"):
                    if tag in tb_full:
                        self._target_branch_short = tag
                        break
        self._off_branch_steps = 0
        self._overshoot_truncation = False  # RL_IMPROV_8 OST — per-episode reset
        self._fold_stall_count = 0
        self._prev_tip_s = 0.0
        self._prev_inserted_gw = 0.0
        # Fold-detector d_corr bypass: if the tip is closing on the correct
        # entry (arclength d_corr decreasing), don't count this step as
        # folding even if path-projection delta is slow. Switched from
        # Euclidean-to-interior-marker d_corr (Fix 18) — the old metric
        # picked up trunk markers that were almost always nearby.
        self._prev_d_corr_arc = float("inf")
        # +1 reward bookkeeping: junction arclengths already rewarded this
        # episode (set, so each junction is rewarded exactly once). Kept
        # for STEP-log compatibility (_entries field); the actual latching
        # is now done by pathcontext's state machine _committed_forks.
        self._correct_entries_seen = set()
        # Plan v5 — per-episode flags from state-machine daughter-commit
        # events. Used by RCCA seeding filter and exposed in `info` dict.
        self._received_correct_daughter = False
        self._received_wrong_daughter = False
        # Plan v5 — "clean thread" latch: True once the state machine
        # commits the wire onto the TARGET daughter branch itself (not
        # merely past any daughter fork). This is the correct
        # "threaded RCCA" signal — `received_correct_daughter` alone is
        # too lenient (it fires at the trunk-top fork, which any
        # trunk-ascending wire crosses). Drives the clean_thread seeding
        # criterion in DualDeviceNav_train.
        self._reached_target_daughter = False

        # Plan v9 Change 5b — pre-bif(11) checkpoint capture state.
        # Captured into memory the first step the wire is ~5 mm before
        # the (2)->(11) junction; written to disk at terminal if the
        # episode finishes with the wire in the target daughter (clean
        # thread). Gated by env var PRE_BIF11_CHECKPOINT_DIR.
        self._pre_bif11_checkpoint_memo = None
        self._pre_bif11_jn_arc = None
        self._captured_pre_bif11 = False

        # Extract restore_checkpoint BEFORE super().reset(). Reason:
        # eve.Env.reset() calls self.start.reset(), which for InsertionPoint
        # runs intervention.reset_devices() and unconditionally zeros xtip /
        # indexFirstNode / DOFs. If we let simulation.reset(checkpoint=...)
        # restore during intervention.reset(), start.reset() immediately
        # wipes it. So we pop the checkpoint here, run super().reset() to
        # let every component initialise normally (with wire at 0 mm), then
        # apply the restore + re-run observation/reward reset so the obs
        # returned from reset() reflects the restored 380 mm state.
        ckpt = None
        self._restore_ckpt_file = None
        self._restore_ckpt_idx = None
        if options is not None and "restore_checkpoint" in options:
            options = dict(options)
            ckpt = options.pop("restore_checkpoint")
            # Plan v10 — per-state logging: record which of the restore-pool
            # checkpoints this episode used (set by CheckpointRestoreWrapper),
            # so success% per start state is a simple EPISODE_START groupby.
            self._restore_ckpt_file = options.get("_restore_checkpoint_file")
            self._restore_ckpt_idx = options.get("_restore_checkpoint_idx")

        # path_context.reset() is called by LocalGuidance.reset() and
        # ArcLengthProgress.reset() inside super().reset(), AFTER
        # pathfinder.reset() has already computed the new path.
        result = super().reset(seed=seed, options=options)

        if ckpt is not None:
            self.intervention.simulation.restore_checkpoint(ckpt)
            # Plan v10 — SOFA first-restore quirk: the FIRST restore after a
            # worker's scene build sets the controller xtip but does NOT apply
            # dof_positions (the wire ends up un-restored ~z=345, so each
            # worker's/respawn's first episode fails before the trunk top —
            # confirmed: 9/16 ep1 started at z<392, branch (18), vs ep2+ at
            # the fork). Apply the restore a SECOND time on the worker's first
            # restore so the wire geometry actually takes.
            if not getattr(self, "_restore_warmed_up", False):
                self.intervention.simulation.restore_checkpoint(ckpt)
                self._restore_warmed_up = True
            # Re-reset observation and reward so their initial per-step
            # quantities (projection, d_rem_prev, cross-track, guidance
            # features) reflect the restored tip state rather than the
            # zero-insertion state that super().reset() computed.
            self._path_context.invalidate()
            ep_nr = max(0, self._episode_count - 1)
            try:
                self.observation.reset(ep_nr)
            except Exception as e:
                self._step_logger.warning(f"observation re-reset after restore failed: {e}")
            try:
                self.reward.reset(ep_nr)
            except Exception as e:
                self._step_logger.warning(f"reward re-reset after restore failed: {e}")
            # Rebuild the (obs, info) tuple gym expects from reset() using
            # the refreshed observation.
            from copy import deepcopy
            result = (deepcopy(self.observation()), deepcopy(self.info.info))

            # Plan v9 Change 8b — restore-start debug snapshot. Render the
            # wire's post-restore state into a separate folder so we can
            # visually verify the SOFA-restore is landing the wire at the
            # intended pose (just before bif(11)). Gated by env var
            # RESTORE_START_SNAPSHOT_DIR; no-op otherwise. Snapshot mode
            # is forced (overrides global SNAPSHOT_MODE) so this works
            # even when regular snapshots are off.
            rs_dir = os.environ.get("RESTORE_START_SNAPSHOT_DIR")
            if rs_dir:
                try:
                    from util.snapshot import save_snapshot
                    rs_mode = os.environ.get(
                        "RESTORE_START_SNAPSHOT_MODE", "centerlines"
                    )
                    save_snapshot(
                        self,
                        episode=self._episode_count,
                        ep_step=0,
                        reason="start",
                        reward=0.0,
                        phase="restore_start",
                        base_dir_override=rs_dir,
                        mode_override=rs_mode,
                    )
                except Exception as e:
                    self._step_logger.warning(
                        f"restore-start snapshot failed: {e}"
                    )

        # Initialise CORRECT_ENTRY_REWARD bookkeeping from the wire's CURRENT
        # position (post-restore). Junctions that the wire is already past at
        # episode start (typically bif1, since checkpoints place the wire at
        # ~372 mm insertion past bif1) are pre-marked as 'seen' so they do
        # NOT trigger the +1 reward. Only junctions the wire CROSSES during
        # the episode count.
        try:
            self._path_context.invalidate()  # ensure projection is fresh
            proj_s = self._path_context.get_projection().s
            # RL_IMPROV_8: pre-populate from daughter-only arclengths to
            # match the +1 reward gate (legacy; kept for STEP-log _entries).
            for j_arc in self._path_context._path_daughter_arclengths:
                if j_arc <= proj_s - 10.0:
                    # wire is at least 10 mm past this daughter at start
                    self._correct_entries_seen.add(float(j_arc))
            # Plan v5 — also pre-populate state-machine's committed-forks
            # latch so SOFA-restored episodes don't spuriously fire +1 on
            # the first step after restore for junctions the wire is
            # already past. Symmetric with _correct_entries_seen above.
            # Plan v9 Change 1: latch is now set of (round(j_arc,3), +1)
            # tuples (only the correct-commit is latched).
            # Plan v9 — +1 now fires at ALL on-path junctions (top, bridge,
            # daughter), so pre-latch from _path_junction_arclengths (all
            # junctions), not just _path_daughter_arclengths. A restored
            # wire starting just before (11) is past (2)->(0); pre-latching
            # it prevents a spurious +1 if the state machine re-detects
            # that crossing.
            for j_arc in self._path_context._path_junction_arclengths:
                if j_arc <= proj_s - 10.0:
                    self._path_context._committed_forks.add(
                        (round(float(j_arc), 3), +1)
                    )
        except Exception:
            pass

        # Log after super().reset() so target coords are populated for this episode
        target_str = ""
        try:
            # RL_IMPROV_8 — log THIS grader's OWN frozen target (graders) instead
            # of the shared dummy intervention.target, so per-grader analysis can
            # tell the 4 daughters' targets apart (previously all 4 EPISODE_START
            # lines logged the identical shared coord).
            if getattr(self, "_is_grader", False) and self._grader_target_coord3d is not None:
                tc = self._grader_target_coord3d
            else:
                tc = self.intervention.target.coordinates3d
            target_str = f" | target=({tc[0]:.1f},{tc[1]:.1f},{tc[2]:.1f})"
        except Exception:
            pass
        ins_str = ""
        try:
            il = self.intervention.device_lengths_inserted
            if il is not None:
                ins_str = f" | inserted=[{il[0]:.2f},{il[1]:.2f}]"
        except Exception:
            pass
        seed_str = (
            f" | seed={self._reset_seed}" if self._reset_seed is not None else ""
        )
        variant_str = (
            f" | phase_c_variant={self._phase_c_variant}"
            if self._phase_c_variant else ""
        )
        restore_str = (
            f" | restore_ckpt={self._restore_ckpt_file}"
            if getattr(self, "_restore_ckpt_file", None) else ""
        )
        self._step_logger.info(
            f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count} | "
            f"wall_time={time.time():.6f} | pid={os.getpid()}{target_str}{ins_str}"
            f" | target_branch={self._target_branch_short}{seed_str}{variant_str}{restore_str}"
        )
        sys.stderr.flush()

        return result

    # ------------------------------------------------------------------
    # Heuristic abort helper
    # ------------------------------------------------------------------
    def _heuristic_abort(self, reason, info):
        """Mark episode for heuristic truncation with the given reason."""
        self._heuristic_abort_reason = reason
        info["heuristic_abort"] = True
        info["heuristic_abort_reason"] = reason

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        step_start_time = time.time()

        if self._last_step_time is not None:
            time_since_last = step_start_time - self._last_step_time
        else:
            time_since_last = 0.0

        # Invalidate projection cache — forces recomputation on first access
        self._path_context.invalidate()

        # Plan v5 — mirror heuristic phase from env to intervention so the
        # LocalGuidance observation (which holds intervention but not env)
        # can read it for the phase one-hot features (16-19). The
        # heuristic policy sets `self._heur_rva_phase` during action
        # computation (before env.step is called); we mirror now so
        # LocalGuidance sees this step's phase, not the previous step's.
        try:
            self.intervention._heur_rva_phase = getattr(self, "_heur_rva_phase", "default")
        except Exception:
            pass
        # Plan v5 (Tier 1) — mirror per-episode counters so LocalGuidance
        # can expose them as Markov-completing observation features 23-25
        # (off_branch_steps_norm, fold_stall_count_norm, episode_step_norm).
        # Mirror BEFORE super().step() so the obs returned by this step
        # reflects last-step's counters (since the counter increment for
        # THIS step happens in env5.step's post-step logic). Conceptually:
        # obs at step t = "state when wire is about to commit to action a_t".
        try:
            self.intervention._env_off_branch_steps = int(self._off_branch_steps)
            self.intervention._env_fold_stall_count = int(self._fold_stall_count)
            self.intervention._env_episode_step = int(self._episode_step_count)
            # n_max_steps lives on the MaxSteps truncation component
            n_max = getattr(self._max_steps_trunc, 'n_max_steps', 600)
            self.intervention._env_max_steps = int(n_max) if n_max else 600
            # Plan v13 — mirror the (env-var-overridable) grace max so the
            # LocalGuidance off_branch_steps_norm feature normalizes by the
            # ACTUAL timeout, not the hardcoded 50 (else the feature would
            # saturate at 1.0 a third of the way to a 150-step timeout).
            self.intervention._env_off_branch_max = int(OFF_BRANCH_GRACE_STEPS)
            # Same for the fold-stall max (EVE_FOLD_STALL_STEPS knob).
            self.intervention._env_fold_stall_max = int(FOLD_STALL_STEPS)
        except Exception:
            pass

        try:
            obs, reward, terminated, truncated, info = super().step(action)
        except Exception as e:
            self._step_logger.error(
                f"STEP_ERROR | ep={self._episode_count} | step={self._episode_step_count} | error={e}"
            )
            sys.stderr.flush()
            raise

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time

        self._step_count += 1
        self._episode_step_count += 1
        self._last_step_time = step_end_time

        # ---- Per-step insertion delta (fold detector) ----
        inserted_gw = self.intervention.device_lengths_inserted[0]
        delta_gw = inserted_gw - self._prev_inserted_gw
        self._prev_inserted_gw = inserted_gw

        # ---- Detector: wire_fold_stall (both modes) ----
        if not terminated and not truncated:
            tip_s = self._path_context.get_projection().s
            delta_s = tip_s - self._prev_tip_s
            self._prev_tip_s = tip_s

            # d_corr bypass: cancel the fold increment if the tip is
            # closing on the next correct junction (arclength d_corr
            # decreasing). Switched from Euclidean-to-interior-marker
            # d_corr (Fix 18) — the old metric included trunk markers,
            # so any tip-near-trunk-centerline reset the fold counter and
            # let wires ball up indefinitely in the trunk.
            d_corr_improving = False
            try:
                d_corr_now = self._path_context.get_arclength_to_next_correct_entry()
                if d_corr_now < self._prev_d_corr_arc - 0.1:
                    d_corr_improving = True
                self._prev_d_corr_arc = d_corr_now
            except Exception:
                pass

            if (
                delta_gw >= FOLD_INSERTION_MM
                and delta_s < FOLD_ARCLENGTH_MM
                and not d_corr_improving
            ):
                self._fold_stall_count += 1
            else:
                self._fold_stall_count = 0

            if self._fold_stall_count >= FOLD_STALL_STEPS:
                if self._heuristic_mode:
                    self._heuristic_abort("wire_fold_stall", info)
                truncated = True

        # ---- State-machine update (RL_IMPROV_8 v2) ----
        # Drives _current_branch_idx and _on_planned_path from this step's
        # projection. Must run before is_on_correct_path() / is_on_correct_branch()
        # so those queries see the fresh state. The projection cache was
        # populated during super().step() by ArcLengthProgress + LocalGuidance.
        try:
            self._path_context.update_branch_state()
        except Exception as e:
            self._step_logger.warning(f"update_branch_state failed: {e}")

        # ---- Plan v9 Change 4: path-segment-conditioned step reward ----
        # Per-step penalty depends on which segment of the planned path
        # the wire is currently on. Replaces the old uniform -0.001 step
        # penalty (which has been removed from the Combination above).
        try:
            reward += self._compute_path_segment_step_reward()
        except Exception as e:
            self._step_logger.warning(
                f"path-segment step reward failed: {e}"
            )

        # ---- Plan v5: drain state-machine daughter-commit events ----
        # update_branch_state may have emitted (j_arc, +1|-1) events when
        # the wire committed at a daughter fork. +1 = correct-daughter
        # commit; -1 = wrong-daughter commit. Each fork latches once per
        # episode via pathcontext._committed_forks.
        try:
            for j_arc, sign in self._path_context._daughter_commit_events:
                reward += float(sign)
                if sign > 0:
                    self._received_correct_daughter = True
                    self._correct_entries_seen.add(float(j_arc))  # for STEP-log _entries count
                else:
                    self._received_wrong_daughter = True
        except Exception as e:
            self._step_logger.warning(f"daughter commit drain failed: {e}")

        # ---- Plan v5: latch "reached target daughter" ----
        # True once the state machine commits the wire onto the target
        # daughter branch (RCCA for RCCA training). This is the correct
        # "cleanly threaded" signal for the seeding filter — distinct
        # from `_received_correct_daughter`, which fires at the trunk-top
        # fork that any trunk-ascending wire crosses.
        try:
            pc = self._path_context
            if (pc._current_branch_idx is not None
                    and pc._target_daughter_branch_idx is not None
                    and pc._current_branch_idx == pc._target_daughter_branch_idx):
                self._reached_target_daughter = True
        except Exception:
            pass

        # ---- Plan v9 Change 5b: pre-bif(11) checkpoint memo ----
        # Capture an in-memory SOFA snapshot the first step the wire is
        # within 5 mm of crossing the (2)->(11) junction. Written to disk
        # at episode end IF the episode terminates in the target
        # daughter (clean thread). Gated by env var
        # PRE_BIF11_CHECKPOINT_DIR; no-op otherwise.
        pre_bif11_dir = os.environ.get("PRE_BIF11_CHECKPOINT_DIR")
        if pre_bif11_dir and not self._captured_pre_bif11:
            try:
                if self._pre_bif11_jn_arc is None:
                    juncs = self._path_context.get_path_junctions()
                    # RCCA path topology is (2)->(0)->(11)->RCCA, so the
                    # junction "just before (11)" is the (0)->(11) bridge
                    # crossing, NOT a direct (2)->(11) (which does not
                    # exist). Matches heuristic_policy_rcca's
                    # _lcca_jn_arc = _find_junction_arc(junctions,"(0)","(11)").
                    for arc, prev_n, next_n in juncs:
                        if re.search(r"\(0\)", prev_n) and re.search(
                            r"\(11\)", next_n
                        ):
                            self._pre_bif11_jn_arc = float(arc)
                            break
                if self._pre_bif11_jn_arc is not None:
                    proj_s = float(self._path_context.get_projection().s)
                    if proj_s >= self._pre_bif11_jn_arc - 5.0:
                        import numpy as _np
                        sim = self.intervention.simulation
                        memo = sim.save_checkpoint()
                        memo["tracking3d"] = _np.array(
                            self.intervention.fluoroscopy.tracking3d,
                            copy=True,
                        )
                        memo["_proj_s_at_capture"] = float(proj_s)
                        memo["_pre_bif11_jn_arc"] = float(self._pre_bif11_jn_arc)
                        self._pre_bif11_checkpoint_memo = memo
                        self._captured_pre_bif11 = True
            except Exception as e:
                self._step_logger.warning(
                    f"pre_bif11_checkpoint capture failed: {e}"
                )

        # ---- RVA-arclength SOFA checkpoint capture (RL_IMPROV_8 §25) ----
        # Save SOFA state the first step where the wire's projection
        # arclength crosses rva_jn_arc — gives us a fixture set for fast
        # Phase C iteration via SOFA-state restore. Gated by RVA_CHECKPOINT_DIR
        # env var; no-op when unset.
        rva_ckpt_dir = os.environ.get("RVA_CHECKPOINT_DIR")
        if rva_ckpt_dir and not self._crossed_rva_jn:
            try:
                if self._rva_jn_arc is None:
                    junctions = self._path_context.get_path_junctions()
                    for arc, prev_n, next_n in junctions:
                        if "(11)" in prev_n and "RVA" in next_n:
                            self._rva_jn_arc = float(arc)
                            break
                if self._rva_jn_arc is not None:
                    proj_s = float(self._path_context.get_projection().s)
                    if proj_s >= self._rva_jn_arc:
                        self._save_rva_checkpoint(rva_ckpt_dir, proj_s)
                        self._crossed_rva_jn = True
            except Exception as e:
                self._step_logger.warning(f"rva_checkpoint failed: {e}")

        # ---- Detector: wrong_path (RL_IMPROV_8 v2 — state machine) ----
        # is_on_correct_path now reads the state-machine flag _on_planned_path
        # which was set with radius-aware tolerance + 10 mm arclength
        # commit hysteresis on junction crossings — no in-lumen flicker.
        on_correct_path = self._path_context.is_on_correct_path()
        on_correct_branch = self._path_context.is_on_correct_branch()  # legacy log
        # ---- RL_IMPROV_8 Fix 2 (state-machine correction, heatup/AWAC only) ----
        # At the shared (11) fork the per-grader state machine can lock a wire that
        # is GEOMETRICALLY in the CORRECT daughter onto a sister and count it
        # off-path (_pick_off_path_branch can never return the target daughter), so
        # a wire threading the right daughter gets spuriously WBT'd. classify_-
        # physical_branch reads ONLY the tip geometry. If the tip is physically in
        # the target daughter AND has NOT yet passed the target (proj.s < total),
        # treat it as on-path so off_branch_steps does not accumulate while it
        # threads the correct daughter. Overshoot (proj.s past the target) is left
        # alone — it still times out and Fix 1 (OST) relabels it. Heuristic path is
        # untouched via the _heuristic_mode gate (its frozen classifier is unchanged).
        if (not on_correct_path and not self._heuristic_mode
                and self._target_branch_short in ("RCCA", "LCCA", "RVA", "LVA")):
            try:
                _proj = self._path_context.get_projection()
                _tl = float(getattr(self._path_context, "_total_length", 0.0) or 0.0)
                if (_tl > 1e-6 and _proj.s < _tl - 1.0
                        and classify_physical_branch(self.intervention)
                        == self._target_branch_short):
                    on_correct_path = True
            except Exception:
                pass
        if not terminated and not truncated:
            if not on_correct_path:
                self._off_branch_steps += 1
                # Plan v5 — heuristic-aligned reward structure:
                # The per-step off-path arc-shaping penalty in
                # ArcLengthProgress (`-progress_factor * Δoff_arc`) is now
                # the SOLE continuous off-path signal. It's symmetric
                # (rewards retract, penalizes deeper drift) and bounded by
                # physics rather than by classification flicker — matching
                # the heuristic's implicit "retract when off-path" priority.
                #
                # The old WRONG_BRANCH_ENTRY_PENALTY (-1 at off_br==3) and
                # WRONG_BRANCH_STEP_PENALTY (-0.1 per step) duplicated this
                # signal as a discrete tax and a per-step constant, neither
                # of which the heuristic enforces. They've been removed.
                # Daughter-fork ±1 commit events (drained above) remain as
                # the only discrete off-path signals — those fire at
                # TOPOLOGICAL decision points (real daughter forks), not
                # for any lateral wedge.
                #
                # The _off_branch_steps counter is still incremented for
                # the 50-step timeout truncation (-5 FailureTruncationPenalty
                # below), matching the heuristic's wrong_branch_timeout abort.
                # Per-daughter policies (currently LVA only) can set
                # _heur_suppress_wrong_branch=True to bypass the timeout in
                # convergence regions where cur_branch flickers but the wire
                # is genuinely advancing along a valid extension branch.
                suppress = bool(getattr(self, "_heur_suppress_wrong_branch", False))
                if not suppress and self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS:
                    # RL_IMPROV_8 Fix 1 (OST) — if the tip is physically in the
                    # CORRECT target daughter at the off-path timeout, the wire
                    # overshot the target INSIDE the right daughter (not a wrong-
                    # branch divergence) → soft -1 + "overshoot" reason instead of
                    # -5 WBT. Geometry-only, heatup-gated. (Fix 2 above already
                    # keeps a still-threading wire on-path, so this fires mainly
                    # for genuine overshoot past the target.)
                    self._overshoot_truncation = False
                    if (not self._heuristic_mode
                            and self._target_branch_short in ("RCCA", "LCCA", "RVA", "LVA")):
                        try:
                            if (classify_physical_branch(self.intervention)
                                    == self._target_branch_short):
                                self._overshoot_truncation = True
                        except Exception:
                            pass
                    if self._heuristic_mode:
                        self._heuristic_abort("wrong_branch_timeout", info)
                    truncated = True
            else:
                self._off_branch_steps = 0

        # ---- Plan v5: +1 daughter-entry reward replaced by state-machine
        # commit events drained above (after update_branch_state). The old
        # `arc_past >= 10` proximity-based +1 fired preemptively and gave
        # no signal for *wrong*-daughter crossings. The new mechanism emits
        # +1 only on actual state-machine commit and -1 on wrong-daughter
        # commit at the same fork.

        # ---- Failure truncation penalty (both modes) ----
        if truncated and not terminated and (
            self._vessel_end_trunc.truncated
            or self._fold_stall_count >= FOLD_STALL_STEPS
            or self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS
        ):
            # RL_IMPROV_8 OST — overshoot inside the correct daughter gets the
            # soft -1; all other truncations (wrong-branch / vessel-end / fold-
            # stall) keep -5. _overshoot_truncation is set ONLY in the off-branch
            # timeout branch above, so vessel_end / fold_stall truncations
            # correctly take the -5 path. Mutually exclusive.
            if getattr(self, "_overshoot_truncation", False):
                reward += OVERSHOOT_PENALTY
            else:
                reward += FAILURE_TRUNCATION_PENALTY

        self._episode_total_reward += reward

        # ---- Logging ----
        # INFO every step for first 30 steps of each episode (diagnosing
        # post-restore bif2 steering), then every 50 steps afterwards or on
        # terminal/truncation.
        # Full per-step diagnostic logging (RL_IMPROV_8 — debug-level
        # detail for analyzing closed-loop / branch-transition dynamics).
        # Was: every 50 steps. Cost: ~14× more log lines per run, but
        # enables exact step-by-step tracing without missing transitions.
        is_info_step = True

        if is_info_step:
            # SOFA info
            delta_ins = [0.0, 0.0]
            sofa_info = ""
            try:
                if hasattr(self.intervention, "simulation"):
                    sim = self.intervention.simulation
                    if hasattr(sim, "inserted_lengths"):
                        inserted_log = list(sim.inserted_lengths)
                        delta_ins = [
                            inserted_log[0] - self._prev_inserted[0],
                            inserted_log[1] - self._prev_inserted[1],
                        ]
                        self._prev_inserted = inserted_log.copy()
                        sofa_info = f"inserted=[{inserted_log[0]:.2f},{inserted_log[1]:.2f}]"
            except Exception:
                pass

            try:
                if hasattr(action, "tolist"):
                    action_str = f"[{','.join(f'{a:.3f}' for a in action.flatten())}]"
                elif hasattr(action, "__iter__"):
                    action_str = f"[{','.join(f'{a:.3f}' for a in action)}]"
                else:
                    action_str = f"{action:.3f}"
            except Exception:
                action_str = str(action)

            # Shared fields (both modes)
            _obr = f"{int(on_correct_branch)}"
            _opath = f"{int(on_correct_path)}"  # RL_IMPROV_8 — stricter
            _off = self._off_branch_steps
            _fold = f"{self._fold_stall_count}/{FOLD_STALL_STEPS}"
            # Replaced Euclidean d_corr/d_wrong + interior-marker coords
            # (Fix 18) with arclength-based metrics that actually measure
            # progress along the planned path.
            _dca = "?"        # arclength to next correct junction (legacy)
            _arc_past = "?"   # arclength past most-recent junction (legacy)
            _d3d = "?"        # RL_IMPROV_8: 3D Euclidean dist to next daughter
            _drouted = "?"    # RL_IMPROV_8 v2: graph-routed dist (handles sister branches)
            _arc_past_d = "?" # RL_IMPROV_8: arclength past most-recent daughter
            _entries = len(self._correct_entries_seen)
            _daughters_passed = "?"  # alias of _entries under daughter-only gate
            try:
                dca_val = self._path_context.get_arclength_to_next_correct_entry()
                _dca = "inf" if dca_val == float("inf") else f"{dca_val:.1f}"
                _arc_past = f"{self._path_context.get_arclength_past_last_junction():.1f}"
            except Exception:
                pass
            try:
                d3d_val = self._path_context.get_3d_dist_to_next_daughter_entry()
                _d3d = "inf" if d3d_val == float("inf") else f"{d3d_val:.1f}"
                drt_val = self._path_context.get_routed_d_corr_to_next_daughter_entry()
                _drouted = "inf" if drt_val == float("inf") else f"{drt_val:.1f}"
                _arc_past_d = f"{self._path_context.get_arclength_past_last_daughter_entry():.1f}"
                _daughters_passed = str(_entries)
            except Exception:
                pass
            # RL_IMPROV_8 v2 state-machine diagnostics
            _cur_branch = "?"
            _local_r = "?"
            _local_tol = "?"
            try:
                cb_idx = self._path_context._current_branch_idx
                if cb_idx is not None and cb_idx < len(self._path_context._branches_tuple):
                    name = getattr(self._path_context._branches_tuple[cb_idx], "name", "")
                    _cur_branch = str(name)[:40]  # truncate long mrk names
                _local_r = f"{self._path_context.get_local_radius():.1f}"
                _local_tol = f"{self._path_context.get_local_tolerance():.1f}"
            except Exception:
                pass

            heur_str = ""
            if self._heuristic_mode:
                _abort = self._heuristic_abort_reason or "none"
                _mask = "?"
                try:
                    _mask = ",".join(self.intervention.last_mask_reasons)
                except Exception:
                    pass
                _phase = str(getattr(self, "_heur_rva_phase", "default"))
                heur_str = (
                    f" | heur=1 | abort={_abort} | mask={_mask}"
                    f" | phase={_phase}"
                )

            # Extra diagnostics: tip3d position and current rotation_instrument
            _tip = "?"
            _rots = "?"
            try:
                t3 = self.intervention.fluoroscopy.tracking3d
                if t3 is not None and len(t3) > 0:
                    _tip = f"({t3[0][0]:.1f},{t3[0][1]:.1f},{t3[0][2]:.1f})"
            except Exception:
                pass
            try:
                r = self.intervention.device_rotations
                if r is not None:
                    _rots = f"[{float(r[0]):.3f},{float(r[1]):.3f}]"
            except Exception:
                pass

            _nearest = str(getattr(self, "_heur_nearest_named", "?"))
            shared_str = (
                f" | on_br={_obr} | on_path={_opath} | off_br={_off}"
                f" | fold={_fold}"
                f" | d_corr_arc={_dca} | arc_past={_arc_past}"
                f" | d_corr_3d={_d3d} | d_corr_routed={_drouted} | arc_past_d={_arc_past_d}"
                f" | cur_branch={_cur_branch} | local_r={_local_r} | tol={_local_tol}"
                f" | nearest_named={_nearest} | entries_passed={_entries}"
                f" | daughters_passed={_daughters_passed}"
                f" | tip3d={_tip} | rot_inst={_rots}"
            )

            # RL_IMPROV_8 — per-grader diagnostics so the 4 interleaved graders
            # are disambiguable AND the on-the-way-vs-overshoot question is
            # answerable from the log alone:
            #   grader      = THIS grader's daughter tag (RCCA/LCCA/RVA/LVA)
            #   tgt/d_tgt   = the grader's OWN target coord + 3D tip->target dist
            #                 (same point & CS the _CoordTargetReachedTerminal
            #                 thresholds on, so d_tgt < threshold == success)
            #   xt_true     = the REAL planned-path cross-track (the cross_tr
            #                 field above is the heuristic's, =0 in heatup)
            #   proj_s/path_len = projection arclength vs planned-path length
            #                 (proj_s >= path_len  ==> tip is PAST the target
            #                 ==> overshoot; proj_s < path_len ==> on the way)
            #   phys        = geometric physical tag (vs state-machine cur_branch)
            _gtgt = "?"; _dtgt = "?"; _xt = "?"; _projs = "?"; _plen = "?"; _phys = "?"
            try:
                _tc = (self._grader_target_coord3d
                       if getattr(self, "_is_grader", False)
                       and self._grader_target_coord3d is not None
                       else np.asarray(self.intervention.target.coordinates3d,
                                       dtype=np.float64))
                _gtgt = f"({_tc[0]:.1f},{_tc[1]:.1f},{_tc[2]:.1f})"
                _tipv = np.asarray(self.intervention.fluoroscopy.tracking3d[0],
                                   dtype=np.float64)
                _dtgt = f"{float(np.linalg.norm(_tipv - _tc)):.1f}"
            except Exception:
                pass
            try:
                _pr = self._path_context.get_projection()
                _xt = f"{float(_pr.cross_track_dist):.2f}"
                _projs = f"{float(_pr.s):.1f}"
                _plen = f"{float(getattr(self._path_context, '_total_length', 0.0)):.1f}"
            except Exception:
                pass
            try:
                _phys = classify_physical_branch(self.intervention)
            except Exception:
                pass
            grader_str = (
                f" | grader={self._target_branch_short}"
                f" | tgt={_gtgt} | d_tgt={_dtgt}"
                f" | xt_true={_xt} | proj_s={_projs} | path_len={_plen}"
                f" | phys={_phys}"
                f" | overshoot={bool(getattr(self, '_overshoot_truncation', False))}"
            )

            # Heuristic diagnostics published by heuristic_policy wrapper
            _head_err = float(getattr(self, "_heur_heading_error", 0.0))
            _cross_tr = float(getattr(self, "_heur_cross_track", 0.0))
            log_msg = (
                f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
                f"global={self._step_count} | wall_time={time.time():.6f} | pid={os.getpid()} | "
                f"cmd_action={action_str} | "
                f"reward={reward:.4f} | cum_reward={self._episode_total_reward:.4f} | "
                f"step_time={step_duration:.3f}s | gap_time={time_since_last:.3f}s | "
                f"term={terminated} | trunc={truncated} | "
                f"{sofa_info} | delta_ins=[{delta_ins[0]:.2f},{delta_ins[1]:.2f}]"
                f" | heading_err={_head_err:+.3f} | cross_tr={_cross_tr:+.2f}"
                f"{shared_str}{grader_str}{heur_str}"
            )
            self._step_logger.info(log_msg)
            for handler in self._step_logger.handlers:
                handler.flush()
            sys.stderr.flush()
        else:
            # Lightweight debug-only log (no action formatting, no SOFA query)
            self._step_logger.debug(
                f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
                f"global={self._step_count} | reward={reward:.4f} | "
                f"cum_reward={self._episode_total_reward:.4f} | "
                f"step_time={step_duration:.3f}s"
            )

            # Track insertion for delta computation on next INFO step
            try:
                if hasattr(self.intervention, "simulation"):
                    sim = self.intervention.simulation
                    if hasattr(sim, "inserted_lengths"):
                        self._prev_inserted = list(sim.inserted_lengths)
            except Exception:
                pass

        # ---- End-of-episode snapshot (RL_IMPROV_8 / Plan v5) ----
        # Gated by SNAPSHOT_MODE env var (none|mesh|centerlines). Saves a
        # PNG of the vessel + wire state at the moment the episode ends.
        # Plan v5 — snapshots are bucketed by training phase so the
        # post-hoc pruner (prune_training_snapshots.py) can keep ALL
        # seed/eval snapshots and only the 10-best/10-worst per 100
        # explore episodes:
        #   eval    — env constructed with mode="eval"
        #   seed    — heuristic-mode episode (heuristic-seeding phase)
        #   explore — everything else (heatup + SAC exploration)
        # Defensive: never raises; never affects rl flow.
        # Plan v12 — under MultiTargetEnv5, suppress this per-grader internal
        # snapshot. The driver fires ONE snapshot per physical episode,
        # bucketed by the physical final branch (not 4x RCCA snapshots from
        # the primary grader). Set on the primary by MultiTargetEnv5.__init__.
        if (terminated or truncated) and \
                not getattr(self, "_suppress_internal_snapshot", False) and \
                os.environ.get("SNAPSHOT_MODE", "none").lower() not in ("", "none", "off", "false", "0"):
            try:
                from util.snapshot import save_snapshot
                reason = self._resolve_termination_reason(terminated, truncated)
                if self.mode == "eval":
                    phase = "eval"
                elif self._heuristic_mode:
                    phase = "seed"
                else:
                    phase = "explore"
                save_snapshot(
                    self,
                    episode=self._episode_count,
                    ep_step=self._episode_step_count,
                    reason=reason,
                    reward=float(self._episode_total_reward),
                    phase=phase,
                )
            except Exception as e:
                self._step_logger.warning(f"snapshot failed: {e}")

        # Plan v5 — expose per-episode commit flags so seeding-filter
        # overrides (e.g., DualDeviceNav_train_RCCA._is_clean_thread) can
        # define "clean RCCA thread" without needing TargetReached.
        try:
            info["received_correct_daughter"] = bool(self._received_correct_daughter)
            info["received_wrong_daughter"] = bool(self._received_wrong_daughter)
            info["reached_target_daughter"] = bool(self._reached_target_daughter)
            # Plan v12 — PER-GRADER deep-target success. info["success"] (from
            # eve.info.TargetReached) reads the SHARED intervention.target, so
            # under MultiTargetEnv5 a secondary would report RCCA's reach, not
            # its own daughter's. self._target_terminal is this grader's
            # terminal (the coord-based _CoordTargetReachedTerminal for graders,
            # the real RCCA TargetReached for the driver), so its .terminal is
            # the correct per-grader deep-target reach (dist(tip, own_coord) <
            # threshold). Used by runner.py is_clean to exclude wrong-branch
            # timeouts. Also a separate Challenge-2 (deep reach) filter signal.
            info["grader_success"] = bool(
                getattr(self._target_terminal, "terminal", False)
            )
            # The episode's terminal REASON (so the harvester can exclude
            # off-path / fold / vessel-end failures from "clean threads").
            # RL_IMPROV_8 OST — an overshoot inside the correct daughter is NOT a
            # wrong-branch failure-timeout. Excluding it here lets is_clean (which
            # excludes grader_failure_timeout) fold an OST reach into the clean
            # lane; info["overshoot"] keeps it individually traceable.
            info["grader_failure_timeout"] = bool(
                (self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS
                 and not getattr(self, "_overshoot_truncation", False))
                or self._fold_stall_count >= FOLD_STALL_STEPS
            )
            info["overshoot"] = bool(getattr(self, "_overshoot_truncation", False))
        except Exception:
            pass

        # Plan v9 Change 2 — expose final-branch identity at end of each
        # step so the seeding filter can switch from the ever-touched
        # `reached_target_daughter` latch to a strict final-state check
        # (final_branch_idx == target_daughter_branch_idx). Only the
        # value at the TERMINAL step matters; consumers should read
        # these from `info` of the last transition of the episode.
        try:
            pc = self._path_context
            info["final_branch_idx"] = (
                int(pc._current_branch_idx)
                if pc._current_branch_idx is not None else None
            )
            info["target_daughter_branch_idx"] = (
                int(pc._target_daughter_branch_idx)
                if pc._target_daughter_branch_idx is not None else None
            )
            # Plan v12 — short tag for downstream filtering. Tells us WHICH
            # daughter the wire is currently in: RCCA/LCCA/RVA/LVA/other.
            # Combined with target_branch_short and received_correct_daughter
            # downstream filters can express the user's "success = (target
            # in daughter AND got +1 bonus)" definition exactly.
            info["final_branch_short"] = _final_branch_short(pc)
            info["target_branch_short"] = self._target_branch_short
        except Exception:
            pass

        # MDP-grounding fix — a truncated episode (wrong_branch_timeout,
        # wire_fold_stall, vessel_end, sim_error, max_steps) is an absorbing
        # FAILURE state: the task is unrecoverable, so its value IS the
        # accumulated penalty and nothing after. Mark it `terminated` so the
        # replay buffer stores done=True and the SAC/AWAC critic STOPS
        # bootstrapping `+ gamma*Q(s_next)` off it. Without this, ~98% of
        # episode-ends (all failures) leave the value function ungrounded —
        # a self-referential bootstrap with no terminal anchor — and the
        # critic diverges (SAC -> +inf, AWAC -> -inf). Done AFTER the -5
        # FailureTruncationPenalty block (which gates on `not terminated`),
        # so the penalty still applies.
        #
        # Plan v12 harvest — snapshot the terminal outcome BEFORE this fix
        # rewrites `terminated` (else EPISODE_OUTCOME would read "success" for
        # every failure). classify here == the broadcast final_branch_short
        # (same shared intervention + step); an early-terminating grader's wire
        # keeps moving for the rest of the physical episode, so it must be
        # captured NOW, not recomputed in the next reset.
        if terminated or truncated:
            try:
                self._last_reason = self._resolve_termination_reason(
                    terminated, truncated
                )
                self._last_final_branch_short = classify_physical_branch(
                    self.intervention
                )
            except Exception:
                pass
        if truncated:
            terminated = True

        # ---- Plan v9 Change 5b: write pre-bif(11) checkpoint on success ----
        # If we captured a memo earlier in this episode AND the wire
        # actually finished in the target daughter (clean RCCA thread),
        # persist the memo to disk. Otherwise drop it.
        if terminated and self._pre_bif11_checkpoint_memo is not None:
            try:
                final_idx = info.get("final_branch_idx")
                target_idx = info.get("target_daughter_branch_idx")
                clean = (
                    final_idx is not None
                    and target_idx is not None
                    and int(final_idx) == int(target_idx)
                )
                if clean:
                    pre_bif11_dir = os.environ.get("PRE_BIF11_CHECKPOINT_DIR")
                    if pre_bif11_dir:
                        self._save_pre_bif11_checkpoint(pre_bif11_dir)
            except Exception as e:
                self._step_logger.warning(
                    f"pre_bif11_checkpoint write-on-terminal failed: {e}"
                )
            # Whether kept or dropped, the memo's role for this episode
            # is done.
            self._pre_bif11_checkpoint_memo = None

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Plan v9 Change 4 — path-segment-conditioned per-step reward
    # ------------------------------------------------------------------
    # Replaces the uniform -0.001 Step penalty with branch-aware values:
    #   trunk (2): linear interp -0.007 (z=345/start) -> -0.002 (pre-bif)
    #   post-bif (0):   -0.007 (wrong-direction off-path branch)
    #   bridge  (11):    0.0   (on-path, between bifs — don't penalise)
    #   target daughter (RCCA for RCCA training): 0.0 (depth reward via
    #     Change 4b's 2x progress doubling, NOT a per-step bonus)
    #   wrong daughter (RVA/LCCA/LVA when target is the other):  -0.007
    #   anything else (fallback): -0.001 (old uniform value, defensive)
    # ------------------------------------------------------------------

    def _classify_branch_segment(self, branch_idx: int) -> str:
        """Return one of: trunk / bridge / target_daughter / wrong_daughter /
        other for a given branch index.

        Plan v12 (de-RCCA-hardcoded) — classify STRUCTURALLY from THIS grader's
        own planned path, NOT by hardcoded branch numbers. The previous version
        hardcoded the bridge as n==11 (RCCA/LCCA's bif2 ostium); RVA routes
        through (19) and LVA through (18), so their own correct bridge failed
        the n==11 test, fell to "other", and was charged -0.002/step instead of
        the bridge's intended 0.0. The structural rule is anatomy-invariant:
          trunk          = the FIRST on-path branch  (pc._trunk_branch_idx)
          target_daughter= the LAST on-path branch   (pc._target_daughter_branch_idx)
          bridge         = any OTHER on-path branch   (in pc._path_branch_idx_set)
          wrong_daughter = a named daughter NOT on this path
          other          = anything else
        All three index fields are derived per-grader from the Dijkstra path
        (pathcontext _build_path_branch_sequence / _build_branch_index), so
        every daughter — (0)+(11) for RCCA/LCCA, (19) for RVA, (18) for LVA —
        maps its own bridge(s) to "bridge" with NO numbered literal.
        """
        try:
            pc = self._path_context
            if branch_idx is None:
                return "other"
            idx = int(branch_idx)
            target_idx = pc._target_daughter_branch_idx
            if target_idx is not None and idx == int(target_idx):
                return "target_daughter"
            trunk_idx = pc._trunk_branch_idx
            if trunk_idx is not None and idx == int(trunk_idx):
                return "trunk"
            if idx in pc._path_branch_idx_set:
                # On THIS grader's planned path, neither trunk nor daughter →
                # an intermediate bridge segment (RCCA/LCCA (0)+(11),
                # RVA (19), LVA (18)). Cheapest on-path segment → 0.0.
                return "bridge"
            # Off-path: a named daughter that isn't this grader's target.
            name = getattr(pc._branches_tuple[idx], "name", "") or ""
            for tag in ("RCCA", "RVA", "LCCA", "LVA"):
                if tag in name:
                    return "wrong_daughter"
            return "other"
        except Exception:
            return "other"

    def _ensure_path_segment_cache(self) -> None:
        """Lazily build the per-branch segment-class map and the trunk
        end-arclength cache, both keyed off the current planned path.
        Recomputed only when the cached path-context object changes
        (cheap pointer check)."""
        try:
            pc = self._path_context
            if (getattr(self, "_step_reward_pc_id", None) == id(pc)
                    and getattr(self, "_step_reward_branch_class", None) is not None):
                return
            self._step_reward_branch_class = {
                i: self._classify_branch_segment(i)
                for i in range(len(pc._branches_tuple))
            }
            # Trunk end arclength = the FIRST on-path junction (top of trunk /
            # the trunk's exit toward the bridge), generic across daughters.
            # The trunk -0.007 -> -0.002 interpolation spans z=345 (proj_s=0)
            # to this junction. Plan v12: dropped the RCCA-specific (2)->(0)
            # named detector — the first on-path junction IS the trunk exit
            # for every daughter (RCCA/LCCA via (0), RVA via (19), LVA via
            # (18)), so this is anatomy-invariant.
            trunk_end_arc = None
            try:
                juncs = pc.get_path_junctions()
                if juncs:
                    trunk_end_arc = float(juncs[0][0])
            except Exception:
                pass
            self._trunk_end_arc = trunk_end_arc
            self._step_reward_pc_id = id(pc)
        except Exception as e:
            try:
                self._step_logger.warning(
                    f"_ensure_path_segment_cache failed: {e}"
                )
            except Exception:
                pass

    def _compute_path_segment_step_reward(self) -> float:
        """Return the per-step reward for the wire's current path segment.
        Called once per step after pathcontext's update_branch_state.

        Plan v9 (corrected) — keyed on ON-PATH vs OFF-PATH first, then
        segment. (0) is a SHARED branch: its on-path portion (the segment
        from bif1 to the (0)->(11) fork) is part of the route to (11) and
        gets the same low 0.002 as the trunk end; only the OVERSHOOT into
        (0) past the fork is off-path and gets the full 0.007. The
        on-path/off-path classifier (is_on_correct_path) already makes
        that distinction, plus it folds in wrong daughters (RVA/LVA/LCCA)
        and LVA-direction drift at the top junction — all off-path -> 0.007.

        On-path corridor:
          trunk (2)         : interpolate -0.007 (z=345) -> -0.002 (pre-bif)
          (0) on-path part  : -0.002 (flat; still progressing to (11))
          (11) bridge       : 0.0
          target daughter   : 0.0 (depth reward via Change 4b progress 2x)
        Off-path (anything): -0.007.
        """
        try:
            self._ensure_path_segment_cache()
            pc = self._path_context
            # Off-path -> full penalty regardless of branch. Covers:
            # wrong daughters (RVA/LVA/LCCA), LVA-direction drift at the
            # top junction, AND overshoot into (0) past the (0)->(11) fork.
            try:
                on_path = pc.is_on_correct_path()
            except Exception:
                on_path = True
            if not on_path:
                return -0.007
            idx = pc._current_branch_idx
            if idx is None:
                return -0.002
            segment_class = self._step_reward_branch_class.get(
                int(idx), self._classify_branch_segment(int(idx))
            )
            if segment_class == "trunk":
                if self._trunk_end_arc is None or self._trunk_end_arc <= 0:
                    return -0.002
                try:
                    proj_s = float(pc.get_projection().s)
                except Exception:
                    return -0.002
                t = max(0.0, min(1.0, proj_s / self._trunk_end_arc))
                return -0.007 + (0.005 * t)  # -0.007 at start -> -0.002 at end
            if segment_class == "bridge":        # on-path intermediate(s)
                return 0.0                        # cheapest on-path segment
            if segment_class == "target_daughter":
                return 0.0                        # depth via Change 4b 2x progress
            return -0.002  # other on-path (defensive, low corridor value)
        except Exception:
            return -0.002

    def _save_pre_bif11_checkpoint(self, out_dir: str) -> None:
        """Plan v9 Change 5b — write the in-memory pre-bif(11) checkpoint
        memo to disk. Called from step() at episode-terminal IF the
        episode finished cleanly threaded into the target daughter.
        """
        try:
            import numpy as np
            os.makedirs(out_dir, exist_ok=True)
            memo = self._pre_bif11_checkpoint_memo
            if memo is None:
                return
            proj_s = float(memo.get("_proj_s_at_capture", 0.0))
            jn_arc = float(memo.get("_pre_bif11_jn_arc", 0.0))
            # Strip leading-underscore metadata before np.savez (NPZ keys
            # can't start with '_' or they'd be hidden on load).
            state = {k: v for k, v in memo.items() if not k.startswith("_")}
            target_coords = None
            try:
                target_coords = list(
                    np.array(
                        self.intervention.target.coordinates3d, copy=True
                    ).flatten().astype(float)
                )
            except Exception:
                pass
            pid = os.getpid()
            base = (
                f"pre_bif11_pid{pid}_ep{self._episode_count:04d}"
                f"_step{self._episode_step_count:04d}"
            )
            npz_path = os.path.join(out_dir, base + ".npz")
            json_path = os.path.join(out_dir, base + ".json")
            np.savez(npz_path, **state)
            inserted_lengths = None
            try:
                inserted_lengths = [
                    float(x) for x in self.intervention.device_lengths_inserted
                ]
            except Exception:
                pass
            meta = {
                "pid": pid,
                "episode_idx": self._episode_count,
                "step_idx": self._episode_step_count,
                "target_branch": self._target_branch_short,
                "target_coordinates3d": target_coords,
                "inserted_lengths": inserted_lengths,
                "proj_s_at_capture": proj_s,
                "pre_bif11_jn_arc": jn_arc,
                "reset_seed": self._reset_seed,
                "wall_time": time.time(),
            }
            import json as _json
            with open(json_path, "w") as f:
                _json.dump(meta, f, indent=2)
            self._step_logger.info(
                f"PRE_BIF11_CHECKPOINT | ep={self._episode_count} | "
                f"pid={pid} | proj_s={proj_s:.2f} | "
                f"pre_bif11_jn_arc={jn_arc:.2f} | file={base}.npz"
            )
        except Exception as e:
            self._step_logger.warning(
                f"_save_pre_bif11_checkpoint exception: {e}"
            )

    def _save_rva_checkpoint(self, out_dir: str, proj_s: float) -> None:
        """Capture SOFA controller + DOF state at the moment wire's
        projection arclength first crosses rva_jn_arc. Mirrors the
        capture mechanism in collect_sofa_checkpoints.py — saves an
        .npz with SOFA controller state + tracking3d, plus a .json
        sidecar with metadata for restore.
        """
        try:
            import numpy as np
            os.makedirs(out_dir, exist_ok=True)
            sim = self.intervention.simulation
            # Plan v9 Change 5 — call the new save_checkpoint() helper to
            # get the fuller-format dict (4 base fields + up to 6 extra
            # DOF DataFields: velocity / force / externalForce /
            # free_position / free_velocity / derivX). Then enrich with
            # tracking3d for downstream consumers.
            state = sim.save_checkpoint()
            state["tracking3d"] = np.array(
                self.intervention.fluoroscopy.tracking3d, copy=True
            )
            target_coords = None
            try:
                target_coords = list(
                    np.array(
                        self.intervention.target.coordinates3d, copy=True
                    ).flatten().astype(float)
                )
            except Exception:
                pass
            pid = os.getpid()
            base = (
                f"rva_ckpt_pid{pid}_ep{self._episode_count:04d}"
                f"_step{self._episode_step_count:04d}"
            )
            npz_path = os.path.join(out_dir, base + ".npz")
            json_path = os.path.join(out_dir, base + ".json")
            np.savez(npz_path, **state)
            inserted_lengths = None
            try:
                inserted_lengths = [
                    float(x) for x in self.intervention.device_lengths_inserted
                ]
            except Exception:
                pass
            meta = {
                "pid": pid,
                "episode_idx": self._episode_count,
                "step_idx": self._episode_step_count,
                "target_branch": self._target_branch_short,
                "target_coordinates3d": target_coords,
                "inserted_lengths": inserted_lengths,
                "proj_s_at_capture": float(proj_s),
                "rva_jn_arc": float(self._rva_jn_arc),
                "reset_seed": self._reset_seed,
                "wall_time": time.time(),
            }
            import json as _json
            with open(json_path, "w") as f:
                _json.dump(meta, f, indent=2)
            self._step_logger.info(
                f"RVA_CHECKPOINT | ep={self._episode_count} | "
                f"ep_step={self._episode_step_count} | pid={pid} | "
                f"proj_s={proj_s:.2f} | rva_jn_arc={self._rva_jn_arc:.2f} | "
                f"file={base}.npz"
            )
        except Exception as e:
            self._step_logger.warning(f"_save_rva_checkpoint exception: {e}")

    def _resolve_termination_reason(self, terminated: bool, truncated: bool) -> str:
        """Map env state into a single short label for snapshot subdirs.

        Priority order matches the order checks fire in step(). Heuristic
        aborts already carry an explicit reason via _heuristic_abort_reason
        so we honour that first.
        """
        if terminated:
            return "success"
        if self._heuristic_abort_reason:
            return str(self._heuristic_abort_reason)
        try:
            if self._fold_stall_count >= FOLD_STALL_STEPS:
                return "wire_fold_stall"
        except Exception:
            pass
        try:
            # Honor the suppression flag here too — if a per-daughter policy
            # was intentionally driving the wire off-path inside a daughter
            # region (e.g., LVA's Phase C override), don't mis-categorize the
            # episode as wrong_branch_timeout. Treat as max_steps instead.
            if (self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS
                    and not getattr(self, "_heur_suppress_wrong_branch", False)):
                # RL_IMPROV_8 OST — same off-path timeout, but if the tip was
                # physically in the target daughter this is an overshoot. Use the
                # flag cached at step-time so the snapshot reason matches the
                # penalty already applied in step().
                if getattr(self, "_overshoot_truncation", False):
                    return "overshoot"
                return "wrong_branch_timeout"
        except Exception:
            pass
        try:
            if getattr(self._vessel_end_trunc, "truncated", False):
                return "vessel_end"
        except Exception:
            pass
        try:
            if getattr(self._sim_error_trunc, "truncated", False):
                return "sim_error"
        except Exception:
            pass
        try:
            if getattr(self._max_steps_trunc, "truncated", False):
                return "max_steps"
        except Exception:
            pass
        return "unknown_truncation"


# ============================================================================
# Plan v12 redesign — Multi-target heatup wrapper (symmetric full graders).
# ============================================================================
# ONE shared eve.intervention.SimulatedIntervention (one SOFA backend) + 4
# FULL BenchEnv5 graders (RCCA primary + LCCA/RVA/LVA), so EVERY daughter runs
# the IDENTICAL reward + observation pipeline as RCCA (the user's "handle all
# branches the same way" requirement) — uniformity guaranteed by construction
# because each grader IS a BenchEnv5, not a hand-rolled simplified peer.
#
# How the shared SOFA is multiplexed:
#   - graders[0] (RCCA) is the DRIVER. Its step()/reset() run the real
#     intervention.step()/reset() that advance / initialise SOFA.
#   - graders[1..3] (LCCA/RVA/LVA) are coord-based BenchEnv5 instances sharing
#     the intervention. During their step()/reset(), intervention.step /
#     intervention.reset / reset_devices are temporarily neutralised (a thin
#     monkeypatch) so they GRADE the already-advanced shared SOFA state
#     against their own daughter WITHOUT re-advancing the physics. This avoids
#     refactoring BenchEnv5's tested 500-line step() while still giving each
#     grader the full pipeline.
#
# final_branch_short (which daughter the wire PHYSICALLY ended in) is ONE
# target-independent geometric fact (classify_physical_branch), computed once
# per tick and broadcast into all 4 graders' info — so the 4 per-target .npz
# agree by construction (was the v0 bug: 56/92; now 91/91).
#
# Snapshots fire ONCE per physical episode from the driver, bucketed by the
# physical final branch with that branch in the header (after reward).
# ============================================================================


class MultiTargetEnv5(gym.Env):
    """Plan v12 multi-target heatup harvester — 4 full BenchEnv5 graders on
    one shared SOFA intervention.

    Inheritance note: subclasses gym.Env solely so eve_rl's confighandler
    stubs serialization. reset/step return a LIST of per-grader tuples (NOT
    the single gym shape); single.py:_play_episode_multitarget dispatches on
    the duck-typed .secondaries / .primary attributes.
    """

    DAUGHTER_SHORT_NAMES: Tuple[str, ...] = ("RCCA", "LCCA", "RVA", "LVA")
    DAUGHTER_CENTERLINE_TEMPLATES: Dict[str, str] = {
        "RCCA": "Centerline curve - RCCA.mrk",
        "LCCA": "Centerline curve - LCCA.mrk",
        "RVA":  "Centerline curve - RVA.mrk",
        "LVA":  "Centerline curve - LVA.mrk",
    }

    def __init__(
        self,
        intervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps: int = 600,
        primary_target_short: str = "RCCA",
        secondary_target_shorts: Optional[Sequence[str]] = None,
    ) -> None:
        if primary_target_short not in self.DAUGHTER_SHORT_NAMES:
            raise ValueError(
                f"primary_target_short must be one of "
                f"{self.DAUGHTER_SHORT_NAMES}; got {primary_target_short!r}"
            )
        if secondary_target_shorts is None:
            secondary_target_shorts = tuple(
                s for s in self.DAUGHTER_SHORT_NAMES if s != primary_target_short
            )
        for s in secondary_target_shorts:
            if s not in self.DAUGHTER_SHORT_NAMES:
                raise ValueError(f"secondary {s!r} not in {self.DAUGHTER_SHORT_NAMES}")
            if s == primary_target_short:
                raise ValueError(f"secondary {s!r} duplicates primary_target_short")

        self.intervention = intervention
        self.mode = mode
        self.primary_target_short = primary_target_short
        self.secondary_target_shorts = tuple(secondary_target_shorts)
        self.n_max_steps = n_max_steps

        # graders[0] — RCCA driver: a normal BenchEnv5 (target_coord3d=None →
        # grades against the shared intervention.target, which its
        # default_target_branch scopes to RCCA). Its step()/reset() advance
        # the real SOFA.
        self.primary = BenchEnv5(
            intervention=intervention,
            mode=mode,
            visualisation=visualisation,
            n_max_steps=n_max_steps,
            default_target_branch=self.DAUGHTER_CENTERLINE_TEMPLATES[
                primary_target_short
            ],
        )
        self.primary._suppress_internal_snapshot = True

        # graders[1..3] — coord-based BenchEnv5 graders sharing the
        # intervention. target_coord3d is a placeholder set per-episode in
        # reset() via set_grader_target(); _is_grader=True makes them use the
        # coord TargetReached reward/terminal + Target2D-coord obs.
        self.secondaries: List[BenchEnv5] = []
        for s in self.secondary_target_shorts:
            g = BenchEnv5(
                intervention=intervention,
                mode=mode,
                visualisation=False,
                n_max_steps=n_max_steps,
                target_coord3d=np.zeros(3, dtype=np.float64),  # placeholder
            )
            g._suppress_internal_snapshot = True
            g._grader_short = s
            self.secondaries.append(g)

        # Per-daughter CenterlineRandom target probes (lazy-built at first reset).
        self._target_probes: Dict[str, Any] = {}

        # Snapshot cadence (driver-level, per physical episode).
        try:
            self._snapshot_every = int(os.environ.get("SNAPSHOT_EVERY", "10"))
        except Exception:
            self._snapshot_every = 10
        self._physical_episode_count = 0

        # Plan v12 R15 — no shared mutable state across the 4 graders' caches.
        from eve.util.pathcontext import assert_caches_independent
        assert_caches_independent(
            [g._path_context for g in ([self.primary] + self.secondaries)]
        )

    # ------------------------------------------------------------------
    # Gym / pickling passthrough.
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        if name in ("primary", "intervention"):
            raise AttributeError(name)
        primary = self.__dict__.get("primary")
        if primary is None:
            raise AttributeError(name)
        return getattr(primary, name)

    def render(self):
        try:
            return self.primary.render()
        except Exception:
            return None

    def close(self):
        try:
            self.primary.close()
        except Exception:
            pass

    def __setstate__(self, state):
        # Each grader is a BenchEnv5 with its OWN __setstate__ that rebuilds
        # its path_context + re-binds obs/reward refs; pickle invokes those
        # recursively. The shared intervention is preserved as ONE object by
        # pickle's identity tracking. Nothing extra needed here.
        self.__dict__.update(state)

    def _ordered_target_shorts(self) -> Tuple[str, ...]:
        return (self.primary_target_short,) + self.secondary_target_shorts

    # ------------------------------------------------------------------
    # target sampling
    # ------------------------------------------------------------------
    def _sample_target_coord3d(self, target_branch_full: str,
                               seed: Optional[int]) -> np.ndarray:
        """Sample a 3-D point on the named daughter centerline (per-episode
        target for a coord-based grader). Probe built lazily after the first
        reset populates the vessel tree."""
        probe = self._target_probes.get(target_branch_full)
        if probe is None:
            probe = eve.intervention.target.CenterlineRandom(
                vessel_tree=self.intervention.vessel_tree,
                fluoroscopy=self.intervention.fluoroscopy,
                threshold=float(getattr(self.intervention.target, "threshold", 5.0)),
                branches=[target_branch_full],
            )
            self._target_probes[target_branch_full] = probe
        probe.reset(seed=seed, target_branch=target_branch_full)
        return np.array(probe.coordinates3d, dtype=np.float64)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """Reset all 4 graders against ONE shared intervention. Returns
        [(obs, info)] per grader, RCCA first."""
        # graders[0] (RCCA) — real intervention reset (builds SOFA + vessel
        # tree + samples the RCCA target). default_target_branch scopes it.
        primary_result = self.primary.reset(seed=seed, options=options)
        # RL_IMPROV_8 — pin the primary's tag explicitly (robust to options not
        # carrying target_branch), matching the secondary override below.
        self.primary._target_branch_short = self.primary_target_short
        results: List[Tuple[Any, Dict[str, Any]]] = [primary_result]

        threshold = float(getattr(self.intervention.target, "threshold", 5.0))

        # graders[1..3] — grade-only reset: neutralise intervention reset so
        # the shared SOFA state (set by the driver) is NOT re-initialised.
        real_reset = self.intervention.reset
        real_reset_devices = getattr(self.intervention, "reset_devices", None)
        self.intervention.reset = lambda *a, **k: None
        if real_reset_devices is not None:
            self.intervention.reset_devices = lambda *a, **k: None
        try:
            for g, s in zip(self.secondaries, self.secondary_target_shorts):
                coord = self._sample_target_coord3d(
                    self.DAUGHTER_CENTERLINE_TEMPLATES[s], seed
                )
                g.set_grader_target(coord, threshold)
                res = g.reset(seed=seed, options=options)
                # RL_IMPROV_8 — g.reset() set _target_branch_short from the SHARED
                # options (the driver's target / "unknown"); override with THIS
                # grader's true daughter tag so the OST + state-machine-correction
                # predicates (which key on _target_branch_short) apply to the
                # secondaries too, not just the RCCA primary.
                g._target_branch_short = s
                results.append(res)
        finally:
            self.intervention.reset = real_reset
            if real_reset_devices is not None:
                self.intervention.reset_devices = real_reset_devices

        self._physical_episode_count = getattr(self, "_physical_episode_count", 0)
        # Per-grader done tracking. Once a grader terminates/truncates, its
        # step() is short-circuited (single.py:219-222 expects this) so it
        # does NOT over-accumulate reward past its own termination. The
        # primary keeps advancing SOFA even when done (the physical episode
        # continues until ALL graders are done) but its grade is stubbed.
        self._grader_done = [False] * (1 + len(self.secondaries))
        self._last_grader_result = [None] * (1 + len(self.secondaries))
        self._snapshot_fired_this_episode = False
        return results

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action) -> List[Tuple[Any, float, bool, bool, Dict[str, Any]]]:
        """One SOFA tick → 4 per-grader (obs, reward, term, trunc, info)
        tuples. Driver advances SOFA + grades RCCA; the other 3 grade the
        shared advanced state without re-advancing physics. Graders that have
        already terminated are short-circuited (no over-stepping)."""
        from eve.util.pathcontext import classify_physical_branch

        if not hasattr(self, "_grader_done"):
            self._grader_done = [False] * (1 + len(self.secondaries))
            self._last_grader_result = [None] * (1 + len(self.secondaries))

        # Advance the shared SOFA physics EXACTLY ONCE, decoupled from grading.
        # All 4 graders are SYMMETRIC: each has its OWN observation / reward /
        # terminal / planned-path and grades the same advanced physical state;
        # they share only the action+physics. The only reason a single physics
        # advance exists is that SOFA must step once per tick, not four times.
        #
        # Previously graders[0] (RCCA) was the "primary" that drove the advance
        # via its own .step(), which made it asymmetric: after it terminated it
        # still had to be stepped (to keep advancing physics for the others),
        # re-grading itself every tick. That wasted grade's -5/step off-path
        # penalty inflated RCCA's *logged* _episode_total_reward to
        # -2000/+15/-91 (a pure logging artifact — the recorded buffer
        # transition was already clean). Decoupling the advance removes that
        # asymmetry entirely: physics steps here, and a terminated grader is
        # short-circuited identically regardless of index. No grader is special.
        self.intervention.step(action)

        # ONE target-independent physical branch, broadcast to all graders.
        try:
            physical_branch = classify_physical_branch(self.intervention)
        except Exception:
            physical_branch = "unknown"

        # Grade all 4 graders against the already-advanced state (grade-only:
        # neutralise intervention.step so NONE of them re-advances physics).
        graders = [self.primary] + list(self.secondaries)
        shorts = [self.primary_target_short] + list(self.secondary_target_shorts)
        real_step = self.intervention.step
        self.intervention.step = lambda *a, **k: None
        results: List[Tuple[Any, float, bool, bool, Dict[str, Any]]] = []
        try:
            for i, (g, s) in enumerate(zip(graders, shorts)):
                if self._grader_done[i]:
                    # Already terminated → never re-stepped; replay its terminal
                    # stub so its reward (and _episode_total_reward) freezes at
                    # termination, identically for every grader.
                    stub = self._last_grader_result[i]
                    if stub is None:
                        stub = (None, 0.0, True, True,
                                {"final_branch_short": physical_branch,
                                 "target_branch_short": s})
                    results.append(stub)
                    continue
                g_obs, g_r, g_term, g_trunc, g_info = g.step(action)
                g_info = dict(g_info)
                g_info["final_branch_short"] = physical_branch
                g_info["target_branch_short"] = s
                g_result = (g_obs, g_r, g_term, g_trunc, g_info)
                self._last_grader_result[i] = g_result
                if g_term or g_trunc:
                    self._grader_done[i] = True
                results.append(g_result)
        finally:
            self.intervention.step = real_step

        # Driver-level snapshot ONCE per physical episode, fired when the
        # PHYSICAL episode truly ends (ALL graders done — not just RCCA), and
        # bucketed by the physical final branch. VERSION-AWARE: render the
        # grader whose target == physical_final_branch (so an LVA-ending
        # episode shows target=LVA, LVA's planned path, LVA's OWN reward — a
        # POSITIVE success — not RCCA's negative reward). Falls back to the
        # highest-reward grader when the wire ended off any daughter
        # (trunk/other). Optional SNAPSHOT_ALL_VERSIONS=1 renders all 4.
        all_done = all(self._grader_done)
        if all_done and not getattr(self, "_snapshot_fired_this_episode", False) and \
                os.environ.get("SNAPSHOT_MODE", "none").lower() not in (
                    "", "none", "off", "false", "0"):
            self._snapshot_fired_this_episode = True
            try:
                self._physical_episode_count += 1
                reached_named = physical_branch in ("RCCA", "LCCA", "RVA", "LVA")
                if reached_named or (
                    self._physical_episode_count % max(1, self._snapshot_every) == 0
                ):
                    from util.snapshot import save_snapshot
                    graders = [self.primary] + list(self.secondaries)
                    shorts = [self.primary_target_short] + list(
                        self.secondary_target_shorts
                    )
                    if self.primary.mode == "eval":
                        phase = "eval"
                    elif getattr(self.primary, "_heuristic_mode", False):
                        phase = "seed"
                    else:
                        phase = "explore"

                    def _render_grader(idx):
                        g = graders[idx]
                        gs = shorts[idx]
                        g_success = bool(
                            getattr(g._target_terminal, "terminal", False)
                        )
                        last = self._last_grader_result[idx]
                        g_trunc = bool(last[3]) if last is not None else False
                        g_reason = g._resolve_termination_reason(g_success, g_trunc)
                        coord = (
                            None if g is self.primary
                            else getattr(g, "_grader_target_coord3d", None)
                        )
                        save_snapshot(
                            g,
                            episode=g._episode_count,
                            ep_step=g._episode_step_count,
                            reason=g_reason,
                            reward=float(g._episode_total_reward),
                            phase=phase,
                            physical_final_branch=physical_branch,
                            target_short=gs,
                            target_coord3d=coord,
                        )

                    if os.environ.get("SNAPSHOT_ALL_VERSIONS", "0") == "1":
                        for idx in range(len(graders)):
                            _render_grader(idx)
                    else:
                        # Pick the version matching the physical outcome; else
                        # the highest-reward grader (physically closest).
                        match_idx = None
                        for idx, gs in enumerate(shorts):
                            if gs == physical_branch:
                                match_idx = idx
                                break
                        if match_idx is None:
                            match_idx = max(
                                range(len(graders)),
                                key=lambda j: graders[j]._episode_total_reward,
                            )
                        _render_grader(match_idx)
            except Exception:
                pass

        return results
