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
import sys
import os
from eve.util.pathcontext import PathProjectionCache

# ---------------------------------------------------------------------------
# Heuristic-mode detector thresholds
# ---------------------------------------------------------------------------
OFF_BRANCH_GRACE_STEPS = 50  # was 20; bumped in RL_IMPROV_7 §7 Fix 3 — with
                              # is_on_correct_branch() hysteresis (§7 Fix 2)
                              # the counter no longer resets on spurious flips,
                              # so 20 was too short for retract recovery from
                              # bif2 wrong branches (~50 steps of retract needed).
OFF_BRANCH_MIN_INSERTED_MM = 0.0  # was 50.0 workaround; now using true branch membership
FAILURE_TRUNCATION_PENALTY = -5.0
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


class BenchEnv5(eve.Env):
    def __init__(
        self,
        intervention: eve.intervention.SimulatedIntervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps=600,
        default_target_branch: str = None,
    ) -> None:
        self.mode = mode
        self.visualisation = visualisation
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

        target_state = eve.observation.Target2D(intervention)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
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
        target_reward = eve.reward.TargetReached(
            intervention,
            factor=3.0,
            final_only_after_all_interim=False,
        )
        step_reward = eve.reward.Step(factor=-0.001)
        arc_progress = eve.reward.ArcLengthProgress(
            intervention=intervention,
            pathfinder=pathfinder,
            progress_factor=0.01,
            lateral_penalty_factor=0.001,
            path_context=self._path_context,
        )
        reward = eve.reward.Combination([target_reward, arc_progress, step_reward])

        # ----------------------------------------------------------------
        # Terminal and Truncation
        # ----------------------------------------------------------------
        terminal = eve.terminal.TargetReached(intervention)

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
        if options is not None and "restore_checkpoint" in options:
            options = dict(options)
            ckpt = options.pop("restore_checkpoint")

        # path_context.reset() is called by LocalGuidance.reset() and
        # ArcLengthProgress.reset() inside super().reset(), AFTER
        # pathfinder.reset() has already computed the new path.
        result = super().reset(seed=seed, options=options)

        if ckpt is not None:
            self.intervention.simulation.restore_checkpoint(ckpt)
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
            for j_arc in self._path_context._path_daughter_arclengths:
                if j_arc <= proj_s - 10.0:
                    self._path_context._committed_forks.append(float(j_arc))
        except Exception:
            pass

        # Log after super().reset() so target coords are populated for this episode
        target_str = ""
        try:
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
        self._step_logger.info(
            f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count} | "
            f"wall_time={time.time():.6f} | pid={os.getpid()}{target_str}{ins_str}"
            f" | target_branch={self._target_branch_short}{seed_str}{variant_str}"
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
                f"{shared_str}{heur_str}"
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
        if (terminated or truncated) and \
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
        if truncated:
            terminated = True

        return obs, reward, terminated, truncated, info

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
            ic = sim._instruments_combined
            state = {
                "xtip": np.array(ic.m_ircontroller.xtip.value, copy=True),
                "rotation_instrument": np.array(
                    ic.m_ircontroller.rotationInstrument.value, copy=True
                ),
                "index_first_node": np.array(
                    ic.m_ircontroller.indexFirstNode.value, copy=True
                ),
                "dof_positions": np.array(ic.DOFs.position.value, copy=True),
                "tracking3d": np.array(
                    self.intervention.fluoroscopy.tracking3d, copy=True
                ),
            }
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
