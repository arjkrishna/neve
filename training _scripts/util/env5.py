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
OFF_BRANCH_GRACE_STEPS = 20
OFF_BRANCH_MIN_INSERTED_MM = 0.0  # was 50.0 workaround; now using true branch membership
FAILURE_TRUNCATION_PENALTY = -5.0
WRONG_BRANCH_ENTRY_PENALTY = -1.0   # one-time penalty on the step the tip enters a wrong branch
WRONG_BRANCH_STEP_PENALTY = -0.1    # per-step penalty for each step while remaining off-branch
FOLD_STALL_STEPS = 15          # consecutive inserting-without-arclength-progress steps
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
        n_max_steps=1000,
    ) -> None:
        self.mode = mode
        self.visualisation = visualisation

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
        tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
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

        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
                "guidance": guidance,
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

        # Detector state reset
        self._heuristic_mode = bool(options and options.get("heuristic_mode", False))
        self._heuristic_abort_reason = None
        self._off_branch_steps = 0
        self._fold_stall_count = 0
        self._prev_tip_s = 0.0
        self._prev_inserted_gw = 0.0

        # path_context.reset() is called by LocalGuidance.reset() and
        # ArcLengthProgress.reset() inside super().reset(), AFTER
        # pathfinder.reset() has already computed the new path.
        result = super().reset(seed=seed, options=options)

        # Log after super().reset() so target coords are populated for this episode
        target_str = ""
        try:
            tc = self.intervention.target.coordinates3d
            target_str = f" | target=({tc[0]:.1f},{tc[1]:.1f},{tc[2]:.1f})"
        except Exception:
            pass
        self._step_logger.info(
            f"EPISODE_START | ep={self._episode_count} | global_steps={self._step_count} | "
            f"wall_time={time.time():.6f} | pid={os.getpid()}{target_str}"
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

            if delta_gw >= FOLD_INSERTION_MM and delta_s < FOLD_ARCLENGTH_MM:
                self._fold_stall_count += 1
            else:
                self._fold_stall_count = 0

            if self._fold_stall_count >= FOLD_STALL_STEPS:
                if self._heuristic_mode:
                    self._heuristic_abort("wire_fold_stall", info)
                truncated = True

        # ---- Detector: wrong_branch (both modes) ----
        # -1 on entry, -0.1 per step while off-branch.
        # After OFF_BRANCH_GRACE_STEPS → truncate + failure penalty.
        on_correct_branch = self._path_context.is_on_correct_branch()
        if not terminated and not truncated:
            if not on_correct_branch:
                self._off_branch_steps += 1
                if self._off_branch_steps == 1:
                    reward += WRONG_BRANCH_ENTRY_PENALTY
                else:
                    reward += WRONG_BRANCH_STEP_PENALTY
                if self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS:
                    if self._heuristic_mode:
                        self._heuristic_abort("wrong_branch_timeout", info)
                    truncated = True
            else:
                self._off_branch_steps = 0

        # ---- Failure truncation penalty (both modes) ----
        if truncated and not terminated and (
            self._vessel_end_trunc.truncated
            or self._fold_stall_count >= FOLD_STALL_STEPS
            or self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS
        ):
            reward += FAILURE_TRUNCATION_PENALTY

        self._episode_total_reward += reward

        # ---- Logging ----
        is_info_step = (
            self._episode_step_count % 50 == 0 or terminated or truncated
        )

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
            _off = self._off_branch_steps
            _fold = f"{self._fold_stall_count}/{FOLD_STALL_STEPS}"
            _dwe = "?"
            _dce = "?"
            _wcoord = "?"
            _ccoord = "?"
            try:
                _dwe = f"{self._path_context.get_dist_to_closest_wrong_entry():.1f}"
                _dce = f"{self._path_context.get_dist_to_next_correct_entry():.1f}"
                wc = self._path_context.get_closest_wrong_entry_coords()
                cc = self._path_context.get_closest_correct_entry_coords()
                _wcoord = f"({wc[0]:.1f},{wc[1]:.1f},{wc[2]:.1f})"
                _ccoord = f"({cc[0]:.1f},{cc[1]:.1f},{cc[2]:.1f})"
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
                heur_str = f" | heur=1 | abort={_abort} | mask={_mask}"

            shared_str = (
                f" | on_br={_obr} | off_br={_off}"
                f" | fold={_fold}"
                f" | d_wrong={_dwe} | wrong_pt={_wcoord}"
                f" | d_corr={_dce} | corr_pt={_ccoord}"
            )

            log_msg = (
                f"STEP | ep={self._episode_count} | ep_step={self._episode_step_count} | "
                f"global={self._step_count} | wall_time={time.time():.6f} | pid={os.getpid()} | "
                f"cmd_action={action_str} | "
                f"reward={reward:.4f} | cum_reward={self._episode_total_reward:.4f} | "
                f"step_time={step_duration:.3f}s | gap_time={time_since_last:.3f}s | "
                f"term={terminated} | trunc={truncated} | "
                f"{sofa_info} | delta_ins=[{delta_ins[0]:.2f},{delta_ins[1]:.2f}]"
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

        return obs, reward, terminated, truncated, info
