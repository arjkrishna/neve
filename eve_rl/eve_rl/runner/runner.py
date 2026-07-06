from typing import Any, Dict, List, Optional
from math import inf
import csv
import json
import logging
import os
import time
import numpy as np
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        quality_info: Optional[str] = None,
        info_results: Optional[List[str]] = None,
        diagnostics_folder: Optional[str] = None,
        policy_snapshot_every_steps: int = 10000,
        n_probe_episodes: int = 3,
        n_probe_near_start_steps: int = 10,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_for_result_file = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.quality_info = quality_info
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)

        # Diagnostics settings
        self.diagnostics_folder = diagnostics_folder
        # Ensure worker step logs directory exists.
        # NOTE: STEP_LOG_DIR must be set BEFORE agent creation (which spawns workers).
        # The training script should set os.environ["STEP_LOG_DIR"] before creating the agent.
        # This setdefault is a fallback but will NOT help if workers are already spawned.
        if self.diagnostics_folder is not None:
            logs_subprocesses = os.path.join(self.diagnostics_folder, "logs_subprocesses")
            os.makedirs(logs_subprocesses, exist_ok=True)
            os.environ.setdefault("STEP_LOG_DIR", logs_subprocesses)
        self.policy_snapshot_every_steps = policy_snapshot_every_steps
        self.n_probe_episodes = n_probe_episodes
        self.n_probe_near_start_steps = n_probe_near_start_steps
        self._next_snapshot_step = policy_snapshot_every_steps
        self._probe_states_set = False
        self._eval_count = 0
        # Plan v10 — save replay buffer every eval (alongside each .everl
        # checkpoint, ~250k explore steps) so the buffer is recoverable and
        # harvestable. Was 4 (every ~1M); the PER buffer save was also
        # broken (deadlock) until the export_all/import_all + loop handlers
        # were added.
        self._replay_save_interval = 1  # save replay buffer every eval

        # Episode summary logger
        self._episode_summary_file = None
        self._episode_summary_logger = None
        self._episode_summary_counter = 0  # Fix #5: monotonic episode_id
        if self.diagnostics_folder is not None:
            csv_folder = os.path.join(self.diagnostics_folder, "csv")
            os.makedirs(csv_folder, exist_ok=True)
            self._episode_summary_file = os.path.join(csv_folder, "episode_summary.jsonl")
            self.logger.info(f"Episode summary will be logged to {self._episode_summary_file}")

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        self._results["quality"] = 0.0
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best quality"] = 0.0
        self._results["best explore steps"] = 0.0

        file_existed = os.path.isfile(results_file)

        with open(results_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_existed:
                writer.writerow(
                    list(self._results.keys())
                    + [" "]
                    + list(agent_parameter_for_result_file.keys())
                )
                writer.writerow(
                    [" "] * (len(self._results.values()) + 1)
                    + list(agent_parameter_for_result_file.values())
                )

        self.best_eval = {"steps": 0, "quality": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def restore_runner_state(self, checkpoint_path: str, probe_states_dir: Optional[str] = None):
        """Restore Runner-level state from a checkpoint for resume support.

        Restores: best_eval, _episode_summary_counter, _next_snapshot_step,
        and optionally probe states from a saved .npz file.
        """
        import torch
        checkpoint = torch.load(checkpoint_path)
        additional_info = checkpoint.get("additional_info", {})
        runner_state = additional_info.get("runner_state") if additional_info else None

        if runner_state:
            self.best_eval = runner_state["best_eval"]
            self._episode_summary_counter = runner_state["episode_summary_counter"]
            self._next_snapshot_step = runner_state["next_snapshot_step"]
            self._eval_count = runner_state.get("eval_count", 0)
            self.logger.info(
                f"Restored runner state: best_eval={self.best_eval}, "
                f"episode_summary_counter={self._episode_summary_counter}, "
                f"next_snapshot_step={self._next_snapshot_step}, "
                f"eval_count={self._eval_count}"
            )
        else:
            # Checkpoint predates runner_state support — compute safe defaults
            explore_steps = self.step_counter.exploration
            self._next_snapshot_step = (
                (explore_steps // self.policy_snapshot_every_steps) + 1
            ) * self.policy_snapshot_every_steps
            self.logger.warning(
                f"No runner_state in checkpoint. Setting _next_snapshot_step={self._next_snapshot_step}"
            )

        # Restore probe states from saved .npz if available
        if probe_states_dir is not None:
            probe_path = os.path.join(probe_states_dir, "probes", "probe_states.npz")
            if os.path.isfile(probe_path):
                try:
                    data = np.load(probe_path)
                    probe_states = data["probe_states"]
                    if hasattr(self.agent, 'set_probe_states'):
                        self.agent.set_probe_states(probe_states.tolist())
                        self._probe_states_set = True
                        self.logger.info(f"Restored {len(probe_states)} probe states from {probe_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to restore probe states: {e}")

    def load_replay_buffer(self):
        """Auto-load replay buffer from checkpoint folder if file exists."""
        replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
        if os.path.isfile(replay_path):
            try:
                n = self.agent.replay_buffer.load_buffer_from_file(replay_path)
                self.logger.info(f"Loaded replay buffer ({n} episodes) from {replay_path}")
                return n
            except Exception as e:
                self.logger.warning(f"Failed to load replay buffer: {e}")
        return 0

    def heatup(
        self,
        steps=None,
        episodes=None,
        heatup_save_every: int = 0,
        heatup_save_path=None,
    ):
        episodes_r = self.agent.heatup(
            steps=steps,
            episodes=episodes,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
            heatup_save_every=heatup_save_every,
            heatup_save_path=heatup_save_path,
        )
        return episodes_r

    @staticmethod
    def _episode_threaded(e) -> bool:
        """Plan v10 — did this episode thread RCCA (final_branch == target)?
        Mirrors Episode.to_replay(): prefer strict final-branch equality from
        the last step's info; fall back to the reached_target_daughter flag
        (raw Episode carries `infos`; EpisodeReplay carries the flag)."""
        infos = getattr(e, "infos", None)
        if infos:
            info = infos[-1]
            fb = info.get("final_branch_idx")
            tb = info.get("target_daughter_branch_idx")
            if fb is not None and tb is not None:
                return int(fb) == int(tb)
            return bool(info.get("reached_target_daughter", False))
        return bool(getattr(e, "reached_target_daughter", False))

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def save_only(self):
        """Plan v11 — save the checkpoint without running eval.

        Used by offline_train(..., skip_inline_eval=True) for the
        train-then-eval-later workflow: run the full training loop with
        zero SOFA-eval overhead, save checkpoints at every cadence
        boundary, then run --eval_only_checkpoint passes in parallel
        afterward. Sidesteps the Synchron eval-hang that blocks
        offline_train for 30-90 min per eval cycle.

        Naming mirrors eval(): in offline mode uses step_counter.update
        so checkpoints land at checkpoint10000.everl,
        checkpoint20000.everl, ... — no overwrites.
        """
        explore_steps = self.step_counter.exploration
        step_for_naming = (
            self.step_counter.update
            if getattr(self, "_is_offline_mode", False) and explore_steps == 0
            else explore_steps
        )
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{step_for_naming}.everl"
        )
        # Minimal eval_results metadata so downstream tooling that reads
        # the .everl additional_info doesn't crash.
        eval_results = {
            "episodes": [],
            "quality": None,
            "reward": None,
            "save_only": True,
            "runner_state": {
                "best_eval": self.best_eval.copy(),
                "episode_summary_counter": self._episode_summary_counter,
                "next_snapshot_step": self._next_snapshot_step,
                "eval_count": self._eval_count,
            },
        }
        self.agent.save_checkpoint(checkpoint_file, eval_results)
        self.logger.info(
            f"Saved checkpoint (no eval): {checkpoint_file}"
        )

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        # Plan v11 — in offline-train mode, exploration counter never
        # advances (no exploration). Naming every eval checkpoint
        # `checkpoint0.everl` overwrites itself across the 10 evals
        # (parity audit CRITICAL #5). Use update steps instead so each
        # eval lands at a unique filename — checkpoint10000.everl,
        # checkpoint20000.everl, etc. Online callers retain
        # exploration-step naming (their exploration counter is
        # non-zero).
        step_for_naming = (
            self.step_counter.update
            if getattr(self, "_is_offline_mode", False) and explore_steps == 0
            else explore_steps
        )
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{step_for_naming}.everl"
        )
        result_episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)
        # Plan v11 — log per-eval-cycle episode summaries to
        # episode_summary.jsonl. Online flow logs explore episodes via
        # explore_and_update(); eval episodes were never logged anywhere
        # (parity audit CRITICAL #4). Calling it here logs only eval
        # episodes — no double-logging of explore episodes.
        if result_episodes:
            self._log_episode_summaries(result_episodes)
        qualities, rewards = [], []
        results_for_info = {
            info_result_name: [] for info_result_name in self.info_results
        }
        eval_results = {"episodes": []}
        for episode in result_episodes:
            reward = episode.episode_reward
            quality = (
                episode.infos[-1][self.quality_info]
                if self.quality_info is not None
                else reward
            )
            qualities.append(quality)
            rewards.append(reward)
            for info_result_name in self.info_results:
                info = episode.infos[-1][info_result_name]
                results_for_info[info_result_name].append(info)
            eval_results["episodes"].append(
                {
                    "seed": episode.seed,
                    "options": episode.options,
                    "quality": episode.infos[-1][self.quality_info],
                }
            )

        reward = sum(rewards) / len(rewards)
        quality = sum(qualities) / len(qualities)
        for info_result_name, results in results_for_info.items():
            result = sum(results) / len(results)
            self._results[info_result_name] = round(result, 3)
        save_best = False
        if quality > self.best_eval["quality"]:
            save_best = True
            self.best_eval["quality"] = quality
            self.best_eval["steps"] = explore_steps

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        self._results["reward"] = round(reward, 3)
        self._results["quality"] = round(quality, 3)
        self._results["best quality"] = self.best_eval["quality"]
        self._results["best explore steps"] = self.best_eval["steps"]

        eval_results.update(self._results)
        eval_results.pop("best quality")
        eval_results.pop("best explore steps")

        self._eval_count += 1

        # Include runner state for resume support
        eval_results["runner_state"] = {
            "best_eval": self.best_eval.copy(),
            "episode_summary_counter": self._episode_summary_counter,
            "next_snapshot_step": self._next_snapshot_step,
            "eval_count": self._eval_count,
        }

        self.agent.save_checkpoint(checkpoint_file, eval_results)
        if save_best:
            checkpoint_file = os.path.join(
                self.checkpoint_folder, "best_checkpoint.everl"
            )
            self.agent.save_checkpoint(checkpoint_file, eval_results)

        log_info = (
            f"Quality: {quality}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        # Periodic replay buffer save
        if self._eval_count % self._replay_save_interval == 0:
            replay_path = os.path.join(self.checkpoint_folder, "replay_buffer.npz")
            try:
                n = self.agent.replay_buffer.save_buffer_to_file(replay_path)
                self.logger.info(f"Saved replay buffer ({n} episodes) at eval #{self._eval_count}")
            except Exception as e:
                self.logger.warning(f"Failed to save replay buffer: {e}")

        return quality, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            # Update budget tracks EXPLORATION steps only. Heatup and
            # heuristic-seeding steps fill the replay buffer but must NOT
            # obligate catch-up updates. heuristic_seed() increments the
            # `heatup` counter, so folding heatup into this formula handed
            # the very first update() call a ~150k-step backlog (with
            # update_step=0) — run all at once on a static buffer, which
            # diverges the SAC critic. Exploration-driven gives clean ~1:1
            # interleaving: one update per new explore step, fresh data
            # every cycle. The heatup/seed transitions still train — they
            # live in the buffer and PER samples them throughout.
            # Online update budget = exploration steps * ratio, minus the
            # online updates already done. The warm-start pretraining
            # updates are subtracted out (via `_pretrain_update_baseline`)
            # so they don't count against the online explore:update ratio.
            # max(0.0, ...) is a hard guard: a negative update budget
            # deadlocks explore_and_update — it must never be passed down.
            update_steps = max(0.0, (
                (self.step_counter.exploration
                 - getattr(self, "_explore_step_baseline", 0))
                * update_steps_per_explore_step
                - (self.step_counter.update
                   - getattr(self, "_pretrain_update_baseline", 0))
            ))
            result = self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )

            # Log episode summaries if we got results
            if result is not None:
                episodes, _ = result if isinstance(result, tuple) else (result, None)
                if episodes:
                    self._log_episode_summaries(episodes)

            # Check for policy snapshots after each exploration cycle
            # This ensures snapshots are saved at the actual milestone with correct weights
            self._maybe_save_policy_snapshot()

    def offline_train(
        self,
        n_updates: int,
        eval_every_updates: int,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
        log_every_updates: int = 1000,
    ):
        """Plan v11 Stage 1 — pure-offline training loop.

        Assumes the caller has already populated `agent.replay_buffer` from
        saved transition archives (`buffer_filter.py` output). No
        exploration is performed. Every `eval_every_updates`, calls
        `self.eval(...)` which uses the eval-env subprocess to score the
        policy and save a checkpoint; periodic replay-buffer saves are
        suppressed (the offline buffer is fixed; resaving it every eval
        wastes disk).

        The loop exits when `agent.step_counter.update` has advanced by
        `n_updates`.
        """
        # Plan v11 — flag this Runner as offline so eval() picks the
        # update-step-based checkpoint naming (the exploration counter
        # stays at 0; without this flag every eval would overwrite
        # checkpoint0.everl — parity audit CRITICAL #5).
        self._is_offline_mode = True
        update_start = self.step_counter.update
        target_updates = update_start + int(n_updates)
        # Suppress periodic replay-buffer save — the buffer is static in
        # offline mode (a re-save would just re-emit ~280k frozen
        # transitions every eval).
        self._replay_save_interval = 10**12
        next_eval_at = update_start + int(eval_every_updates)
        next_log_at = update_start + int(log_every_updates)
        self.logger.info(
            f"Offline-train: {n_updates} updates, eval every "
            f"{eval_every_updates}; starting at update={update_start}."
        )
        # Drive the agent's update loop in chunks bounded by the next
        # eval milestone. self.agent.update() blocks on the trainer
        # subprocess to flush each chunk.
        while self.step_counter.update < target_updates:
            cur = self.step_counter.update
            chunk = min(next_eval_at, target_updates) - cur
            if chunk <= 0:
                break
            self.agent.update(steps=int(chunk))
            done = self.step_counter.update
            if done >= next_log_at:
                self.logger.info(
                    f"Offline-train: updates={done}/{target_updates}"
                )
                next_log_at = done + int(log_every_updates)
            if done >= next_eval_at:
                # Plan v11 — train-then-eval-later mode. When
                # _skip_inline_eval is set, just save the checkpoint and
                # skip the SOFA-eval entirely. Eval whichever checkpoints
                # you want afterward via --eval_only_checkpoint passes,
                # parallel docker containers, etc.
                if getattr(self, "_skip_inline_eval", False):
                    self.save_only()
                else:
                    self.eval(episodes=eval_episodes, seeds=eval_seeds)
                next_eval_at = done + int(eval_every_updates)

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
        heatup_cache_save_path: Optional[str] = None,
        pretrain_updates: int = 0,
        heatup_until_successes: int = 0,
        heatup_episode_limit: int = 0,
        heatup_only: bool = False,
        heatup_save_every: int = 0,
    ):
        # TODO: Log Training Run Infos
        # Plan v10 — heatup-until-N-threaded: instead of a fixed step budget,
        # run heatup in chunks until N episodes have THREADED RCCA (the seed
        # signal), with a safety cap. The whole heatup run (threaded + fails)
        # is the seed. `heatup_steps` is the per-chunk budget here.
        if heatup_until_successes and heatup_until_successes > 0:
            heatup_episodes = []
            threaded = 0
            cap_episodes = 2000
            chunk = 0
            while threaded < heatup_until_successes and len(heatup_episodes) < cap_episodes:
                chunk += 1
                batch = self.heatup(heatup_steps)
                if not batch:
                    self.logger.warning(
                        "heatup-until-N: empty heatup batch — stopping."
                    )
                    break
                heatup_episodes += batch
                threaded = sum(1 for e in heatup_episodes if self._episode_threaded(e))
                self.logger.info(
                    f"heatup-until-N: chunk {chunk} | total_eps={len(heatup_episodes)} "
                    f"| threaded={threaded}/{heatup_until_successes}"
                )
            if threaded < heatup_until_successes:
                self.logger.warning(
                    f"heatup-until-N: only {threaded} threaded in "
                    f"{len(heatup_episodes)} eps (cap {cap_episodes}) — the 5 "
                    f"states may not be as easy as hoped; proceeding anyway."
                )
        elif heatup_episode_limit and heatup_episode_limit > 0:
            # Plan v10 — fixed N heatup episodes (random, restore-at-fork) for
            # the fail/exploration side of the seed, after heuristic seeding.
            self.logger.info(f"Heatup: {heatup_episode_limit} episodes (fixed).")
            heatup_episodes = self.heatup(episodes=heatup_episode_limit)
        else:
            # Plan v12 Stage 2 — the indefinite --heatup_only path. When
            # heatup_save_every>0 the workers stream per-daughter chunk files
            # (Version A + Version B) every N episodes and return an EMPTY list
            # (memory-safe), so the end-of-heatup save block below is a no-op
            # for rolling runs; all data lives in the worker chunks on disk.
            heatup_episodes = self.heatup(
                heatup_steps,
                heatup_save_every=heatup_save_every,
                heatup_save_path=heatup_cache_save_path,
            )

        # Save heatup cache if requested (before training starts).
        # Plan v12 — multi-target dispatch: when any Episode in
        # `heatup_episodes` carries `target_branch_idx >= 0`, partition the
        # list by target and save 4 per-target .npz files (one per virtual
        # env) so Stage 3 can load whichever target it's training. Single-
        # target legacy (target_branch_idx == -1 on every Episode) writes
        # one .npz file as before.
        if heatup_cache_save_path and heatup_episodes:
            import os
            from ..util.experience_cache import save_episodes_npz

            def _ep_metadata(ep) -> Dict[str, Any]:
                """Plan v12 R17 — build per-Episode metadata from
                `infos[-1]` STRICTLY (not from any runtime latch like
                env5._reached_target_daughter which never resets). All
                fields needed by Stage 3 PER balanced_fraction lane +
                manual heatup inspection are populated here."""
                last_info = ep.infos[-1] if ep.infos else {}
                # Plan v12 redesign — is_clean from the PHYSICAL-tag formula
                # (matches the user's success definition exactly):
                #   success(T) = (wire physically ended in daughter T)
                #               AND (T's +1.0 daughter-commit bonus fired)
                #             = (final_branch_short == target_branch_short)
                #               AND received_correct_daughter
                # The old test (final_branch_idx == target_daughter_branch_idx)
                # compared per-grader state-machine indices into each grader's
                # own _branches_tuple — NOT comparable across graders, and the
                # source of the 56/92 disagreement. final_branch_short is now
                # the ONE target-independent physical branch (classify_physical_branch,
                # broadcast to all 4 graders by MultiTargetEnv5.step).
                fbs = str(last_info.get("final_branch_short", "unknown"))
                tbs = str(last_info.get("target_branch_short", "unknown"))
                rcd = bool(last_info.get("received_correct_daughter", False))
                # Plan v12 — per-grader flags (NOT the shared-intervention
                # success). grader_success = deep TargetReached on THIS
                # daughter; grader_failure_timeout = off-branch/fold timeout.
                grader_success = bool(last_info.get("grader_success", False))
                grader_timeout = bool(last_info.get("grader_failure_timeout", False))
                # is_clean = "threaded THIS daughter and did NOT fail as a
                # wrong-branch/fold timeout". This is the user's "success
                # version" (the version where the wire went, that kept
                # recording, not the ones that timed out). It does NOT
                # require the deep target (that is Challenge-2 = grader_success,
                # exposed separately); random heatup rarely reaches it.
                # Excluding grader_timeout repairs the mislabel where an
                # overshoot/off-path episode that was still physically in the
                # daughter was counted clean.
                is_clean = (
                    fbs == tbs
                    and fbs not in ("unknown", "other")
                    and rcd
                    and not grader_timeout
                )
                return {
                    "episode_return": float(ep.episode_reward),
                    "reached_target_daughter": bool(is_clean),
                    "is_demo": bool(getattr(ep, "is_demo", False)),
                    # Plan v12 R17 — explicit is_clean alongside the
                    # legacy reached_target_daughter alias so consumers
                    # that grew up on either name work.
                    "is_clean": bool(is_clean),
                    "target_branch_idx": int(
                        getattr(ep, "target_branch_idx", -1)
                    ),
                    # Per the user's "make it traceable" requirement —
                    # which daughter the wire actually ended in, plus
                    # this Episode's tracking-target and the +1.0 bonus
                    # signal. Allows downstream filters like
                    #   "RCCA-target episodes where wire actually went LCCA"
                    #   "any episode where wire reached LVA"
                    #   "successes = threaded AND not timeout"
                    "final_branch_short": str(
                        last_info.get("final_branch_short", "unknown")
                    ),
                    "target_branch_short": str(
                        last_info.get("target_branch_short", "unknown")
                    ),
                    "received_correct_daughter": bool(
                        last_info.get("received_correct_daughter", False)
                    ),
                    "received_wrong_daughter": bool(
                        last_info.get("received_wrong_daughter", False)
                    ),
                    # Plan v12 — Challenge-2 deep-reach (per-grader) + failure
                    # reason, for separate filtering of "threaded" vs "reached
                    # deep target" vs "timed out".
                    "grader_success": grader_success,
                    "grader_failure_timeout": grader_timeout,
                    # RL_IMPROV_8 OST — overshoot inside the correct daughter.
                    # Folds into is_clean (grader_timeout excludes it) but stays
                    # individually traceable for downstream filtering.
                    "overshoot": bool(last_info.get("overshoot", False)),
                }

            # Partition by target_branch_idx (single-target = all -1 → one
            # group; multi-target = up to 4 groups keyed 0..3).
            from collections import defaultdict
            buckets = defaultdict(list)
            for ep in heatup_episodes:
                tbi = int(getattr(ep, "target_branch_idx", -1))
                buckets[tbi].append(ep)

            os.makedirs(
                os.path.dirname(heatup_cache_save_path) or ".", exist_ok=True
            )

            if len(buckets) == 1 and -1 in buckets:
                # Single-target legacy path — preserve original filename.
                eps = buckets[-1]
                tuples = [
                    (np.array(e.flat_obs), np.array(e.actions),
                     np.array(e.rewards), np.array(e.terminals))
                    for e in eps
                ]
                metadata = [_ep_metadata(e) for e in eps]
                save_episodes_npz(
                    heatup_cache_save_path, tuples, metadata=metadata,
                )
                self.logger.info(
                    f"Saved heatup cache ({len(tuples)} episodes) to "
                    f"{heatup_cache_save_path}"
                )
            else:
                # Plan v12 multi-target: split into per-target .npz files
                # by injecting `_<target_short>` before the .npz extension.
                # The target_short tag comes from MultiTargetEnv5's ordered
                # short names; fall back to the integer index when not
                # available so the file is always uniquely named.
                base, ext = os.path.splitext(heatup_cache_save_path)
                # Try to read the daughter ordering from the env on the
                # primary agent's main process. Worker-side runs don't
                # have this attribute — use integer suffix as fallback.
                ordered_shorts = None
                try:
                    env_train = getattr(self.agent, "env_train", None)
                    if env_train is not None and hasattr(
                        env_train, "_ordered_target_shorts"
                    ):
                        ordered_shorts = env_train._ordered_target_shorts()
                except Exception:
                    ordered_shorts = None

                for tbi, eps in sorted(buckets.items()):
                    if tbi < 0:
                        suffix = "single"
                    elif (
                        ordered_shorts is not None
                        and 0 <= tbi < len(ordered_shorts)
                    ):
                        suffix = ordered_shorts[tbi]
                    else:
                        suffix = f"t{tbi}"
                    per_target_path = f"{base}_{suffix}{ext}"
                    tuples = [
                        (np.array(e.flat_obs), np.array(e.actions),
                         np.array(e.rewards), np.array(e.terminals))
                        for e in eps
                    ]
                    metadata = [_ep_metadata(e) for e in eps]
                    save_episodes_npz(
                        per_target_path, tuples, metadata=metadata,
                    )
                    n_clean = sum(1 for m in metadata if m["is_clean"])
                    self.logger.info(
                        f"Saved heatup cache target={suffix}: "
                        f"{len(tuples)} eps ({n_clean} clean) to "
                        f"{per_target_path}"
                    )

        # Capture and set probe states after heatup
        self._capture_and_set_probe_states(heatup_episodes)

        # Plan v12 Stage 2 — standalone heatup-only mode. After heatup
        # completes and the per-target .npz files are saved, exit immediately
        # without entering pretrain / explore / training loop. Used by the
        # standalone heatup harvester launcher (--heatup_only flag in
        # DualDeviceNav_train.py). Quality signal returned is (0.0, 0.0)
        # so the caller doesn't trip eval thresholds.
        if heatup_only:
            self.logger.info(
                f"Plan v12 --heatup_only: heatup harvest complete "
                f"({len(heatup_episodes)} episodes); exiting before "
                f"pretrain / training loop."
            )
            return 0.0, 0.0

        # Warm-start — pretrain critic+policy on the seeded (heuristic +
        # heatup) buffer before ANY exploration, so the first explore
        # episodes are driven by a policy that has already learned from the
        # demonstrations rather than a random network. Skipped when 0.
        self._pretrain_update_baseline = 0
        # Plan v13 — baseline the EXPLORATION counter too. With a resumed run
        # (--warm_start_checkpoint restores step counters) exploration starts
        # at e.g. 804k while updates-since-baseline start at 0, so the budget
        # formula `exploration*ratio - updates_since_baseline` handed the
        # first explore_and_update a ~402k-update mega-task — hours of
        # updates on a static buffer (the exact diverge-the-critic regime
        # the comment below warns about). Stalled runs 2d/2e/2f at their
        # first segment boundary. Both sides of the subtraction must count
        # only steps accrued THIS session.
        self._explore_step_baseline = self.step_counter.exploration
        if pretrain_updates > 0:
            self.logger.info(
                f"Warm-start: {pretrain_updates} pretraining updates on the "
                f"seeded buffer before exploration."
            )
            self.agent.update(steps=pretrain_updates)
            # Record the pretraining update count. The explore-driven update
            # budget (explore_and_update) must NOT count these — they are
            # offline pretraining, not part of the online explore:update
            # ratio. Without this, `exploration*ratio - update` goes hugely
            # negative right after pretraining (exploration=0, update=
            # pretrain_updates) and the first explore_and_update hangs on a
            # negative update budget.
            self._pretrain_update_baseline = self.step_counter.update

        next_eval_step_limt = (
            self.agent.step_counter.exploration + explore_steps_between_eval
        )
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )

            # Note: policy snapshots are now saved inside explore_and_update()
            # at the actual milestone steps with correct weights

            quality, reward = self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        return quality, reward

    def _capture_and_set_probe_states(self, episodes: List):
        """Capture probe states from heatup episodes and set them in trainer."""
        if not episodes or self._probe_states_set:
            return

        try:
            probe_states = []
            n_episodes = min(len(episodes), self.n_probe_episodes)

            for ep_idx in range(n_episodes):
                episode = episodes[ep_idx]
                if hasattr(episode, 'flat_obs') and len(episode.flat_obs) > 0:
                    # Collect start state
                    probe_states.append(episode.flat_obs[0])

                    # Collect near-start states
                    n_states = min(self.n_probe_near_start_steps, len(episode.flat_obs) - 1)
                    for step_idx in range(1, n_states + 1):
                        probe_states.append(episode.flat_obs[step_idx])

            if probe_states:
                probe_states_array = np.array(probe_states)

                # Set probe states in agent (if it supports it)
                if hasattr(self.agent, 'set_probe_states'):
                    self.agent.set_probe_states(probe_states_array.tolist())
                    self.logger.info(f"Set {len(probe_states)} probe states for diagnostics")

                # Save probe states to disk
                if self.diagnostics_folder is not None:
                    probes_dir = os.path.join(self.diagnostics_folder, "probes")
                    os.makedirs(probes_dir, exist_ok=True)
                    probe_path = os.path.join(probes_dir, "probe_states.npz")
                    np.savez(probe_path, probe_states=probe_states_array)
                    self.logger.info(f"Saved probe states to {probe_path}")

                self._probe_states_set = True

        except Exception as e:
            self.logger.warning(f"Failed to capture probe states: {e}")

    def _maybe_save_policy_snapshot(self):
        """
        Save policy snapshot if we've reached the next snapshot step.

        FIXED: Uses a while loop to catch up on all missed snapshots when the
        eval cycle spans multiple snapshot intervals.
        """
        if self.diagnostics_folder is None:
            return

        current_step = self.step_counter.exploration

        # Save all missed snapshots (in case eval cycle spans multiple snapshot intervals)
        while current_step >= self._next_snapshot_step:
            try:
                snapshots_dir = os.path.join(self.diagnostics_folder, "policy_snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)

                # Use the scheduled snapshot step, not current step, for consistent naming
                snapshot_step = self._next_snapshot_step
                snapshot_path = os.path.join(snapshots_dir, f"policy_{snapshot_step}.pt")

                if hasattr(self.agent, 'save_policy_snapshot'):
                    self.agent.save_policy_snapshot(snapshot_path)
                    self.logger.info(f"Saved policy snapshot at scheduled step {snapshot_step} (current={current_step})")

            except Exception as e:
                self.logger.warning(f"Failed to save policy snapshot at step {self._next_snapshot_step}: {e}")

            self._next_snapshot_step += self.policy_snapshot_every_steps

        # Flush diagnostics logs after all snapshots are saved
        if hasattr(self.agent, 'flush_diagnostics'):
            try:
                self.agent.flush_diagnostics()
            except Exception as e:
                self.logger.warning(f"Failed to flush diagnostics: {e}")

    def _log_episode_summaries(self, episodes: List):
        """
        Log episode summaries to episode_summary.jsonl with explore_step and update_step.

        This directly solves the timestamp correlation problem by providing exact
        step counters at episode completion time.
        """
        if self._episode_summary_file is None:
            return

        try:
            # Fix #4: Use per-episode step counters stamped by the worker at
            # completion time. Fall back to current shared counter for backward
            # compatibility with Episode objects that don't carry step snapshots.
            fallback_explore = self.step_counter.exploration
            fallback_update = self.step_counter.update

            with open(self._episode_summary_file, "a", encoding="utf-8") as f:
                for i, episode in enumerate(episodes):
                    try:
                        wall_time = time.time()  # Per-episode timestamp
                        # Use per-episode step snapshot if available, else fallback
                        explore_step = getattr(episode, 'explore_step_at_completion', None) or fallback_explore
                        update_step = getattr(episode, 'update_step_at_completion', None) or fallback_update
                        # Extract episode data
                        total_reward = getattr(episode, 'episode_reward', 0.0)
                        steps = len(getattr(episode, 'rewards', []))
                        terminated = getattr(episode, 'terminals', [False])[-1] if getattr(episode, 'terminals', []) else False
                        truncated = getattr(episode, 'truncations', [False])[-1] if getattr(episode, 'truncations', []) else False

                        # Try to get max_insertion from episode info if available
                        max_insertion = None
                        infos = getattr(episode, 'infos', [])
                        if infos:
                            last_info = infos[-1] if infos else {}
                            # Look for insertion depth in common info keys
                            for key in ['max_insertion', 'inserted_length', 'insertion_depth']:
                                if key in last_info:
                                    val = last_info[key]
                                    if hasattr(val, '__iter__') and not isinstance(val, str):
                                        max_insertion = float(max(val))
                                    else:
                                        max_insertion = float(val)
                                    break

                        # Fix #5: Monotonically increasing counter for episode_id.
                        # Replaces retroactive calculation that assumed sequential,
                        # non-interleaved episodes.
                        episode_id = self._episode_summary_counter
                        self._episode_summary_counter += 1

                        summary = {
                            "wall_time": wall_time,
                            "explore_step": explore_step,
                            "update_step": update_step,
                            "episode_id": episode_id,
                            "total_reward": float(total_reward),
                            "steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                        if max_insertion is not None:
                            summary["max_insertion"] = max_insertion

                        f.write(json.dumps(summary) + "\n")

                    except Exception as e:
                        self.logger.debug(f"Failed to log episode {i} summary: {e}")

                f.flush()

        except Exception as e:
            self.logger.warning(f"Failed to log episode summaries: {e}")
