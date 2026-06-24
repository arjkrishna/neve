from copy import deepcopy
from importlib import import_module
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import torch
import numpy as np
import gymnasium as gym

from .agent import Agent, StepCounter, EpisodeCounter, AgentEvalOnly
from ..algo import Algo, AlgoPlayOnly
from ..replaybuffer import ReplayBuffer, Episode
from ..util import ConfigHandler, flatten_obs


class SingleEvalOnly(AgentEvalOnly):
    def __init__(
        self,
        algo: AlgoPlayOnly,
        env_eval: gym.Env,
        device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_eval = env_eval
        self.normalize_actions = normalize_actions

        self.step_counter = StepCounter()
        self.episode_counter = EpisodeCounter()
        self.to(device)
        self._next_batch = None
        self._replay_too_small = True
        self.logger.info("Single agent initialized")

    def evaluate(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_eval(steps, step_limit, episodes, episode_limit, seeds, options)
        step_limit, episode_limit = self._log_and_convert_limits(
            "evaluation", steps, step_limit, episodes, episode_limit, seeds, options
        )
        seeds = deepcopy(seeds)
        options = deepcopy(options)
        episodes_data = []
        n_episodes = 0
        n_steps = 0

        while True:
            with self.episode_counter.lock:
                self.episode_counter.evaluation += 1

            next_seed = seeds.pop(-1) if seeds is not None else None
            next_options = options.pop(-1) if options is not None else None

            episode, n_steps_episode = self._play_episode(
                env=self.env_eval,
                action_function=self.algo.get_eval_action,
                consecutive_actions=1,
                seed=next_seed,
                options=next_options,
            )

            with self.step_counter.lock:
                self.step_counter.evaluation += n_steps_episode

            n_episodes += 1
            n_steps += n_steps_episode
            episodes_data.append(episode)

            if (
                (not seeds and not options)
                or self.step_counter.evaluation > step_limit
                or self.episode_counter.evaluation > episode_limit
            ):
                break

        t_duration = perf_counter() - t_start
        self._log_task_completion("evaluation", n_steps, t_duration, n_episodes)
        return episodes_data

    def _play_episode(
        self,
        env: gym.Env,
        action_function: Callable[[np.ndarray], np.ndarray],
        consecutive_actions: int,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Episode, int]:
        terminal = False
        truncation = False
        step_counter = 0

        self.algo.reset()
        obs, _ = env.reset(seed=seed, options=options)
        flat_obs, flat_obs_to_obs = flatten_obs(obs)
        episode = Episode(obs, flat_obs, flat_obs_to_obs, seed, options)

        while not (terminal or truncation):
            action = action_function(flat_obs)

            for _ in range(consecutive_actions):
                env_action = action.reshape(env.action_space.shape)
                if self.normalize_actions:
                    env_action = (env_action + 1) / 2 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low
                obs, reward, terminal, truncation, info = env.step(env_action)
                flat_obs, _ = flatten_obs(obs)
                step_counter += 1
                env.render()
                episode.add_transition(
                    obs, flat_obs, action, reward, terminal, truncation, info
                )
                if terminal or truncation:
                    break

        return episode, step_counter

    def _play_episode_multitarget(
        self,
        env,
        action_function: Callable[[np.ndarray], np.ndarray],
        consecutive_actions: int,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Episode], int]:
        """Plan v12 Stage 1 multi-target heatup play loop.

        Expects `env` to be a `MultiTargetEnv5` (or duck-typed equivalent):
          - reset() returns a List[(obs, info)] of length N (primary +
            secondaries).
          - step(action) returns a List[(obs, reward, term, trunc, info)]
            of length N.

        Behavior:
          - Sample action ONCE per SOFA step (shared across all virtual envs).
          - Tick SOFA once via env.step(action) — same action applied; the
            wrapper computes per-virtual-env (obs, reward, term, trunc, info).
          - For each virtual env k, append the transition to its Episode
            UNTIL that virtual env terminates/truncates. Other virtual envs
            continue accumulating.
          - SOFA loop exits when ALL virtual envs have terminated OR truncated.

        Returns:
            (List[Episode] of length N, total SOFA step count).
            Each Episode is tagged with target_branch_idx = k (0..N-1),
            matching the order env.reset() returned its tuples. Episode's
            R17-safe to_replay() reads infos[-1] strictly to derive
            reached_target_daughter from this virtual env's per-step info.
        """
        n_targets = len(env.secondaries) + 1
        # Per-virtual-env termination flags. Once flipped True, subsequent
        # SOFA steps don't extend that virtual env's Episode.
        done = [False] * n_targets
        step_counter = 0

        self.algo.reset()
        reset_results = env.reset(seed=seed, options=options)
        if len(reset_results) != n_targets:
            raise RuntimeError(
                f"_play_episode_multitarget: env.reset() returned "
                f"{len(reset_results)} tuples, expected {n_targets}. "
                f"env type {type(env).__name__} not multi-target?"
            )

        episodes: List[Episode] = []
        flat_obs_list: List[np.ndarray] = []
        for k, (obs_k, _info_k) in enumerate(reset_results):
            flat_obs_k, flat_obs_to_obs_k = flatten_obs(obs_k)
            ep_k = Episode(obs_k, flat_obs_k, flat_obs_to_obs_k, seed, options)
            ep_k.target_branch_idx = k
            episodes.append(ep_k)
            flat_obs_list.append(flat_obs_k)

        # The action_function reads flat_obs to potentially condition on
        # state (e.g. heatup random sampling ignores it; a learned policy
        # would not). For multi-target heatup, we use the PRIMARY virtual
        # env's flat_obs as the canonical state input (Plan v12 v0 — v1
        # adds per-secondary obs and the action_function would average or
        # vote across them; for random heatup the obs is unused).
        primary_flat_obs = flat_obs_list[0]

        while not all(done):
            action = action_function(primary_flat_obs)

            for _ in range(consecutive_actions):
                env_action = action.reshape(env.action_space.shape)
                if self.normalize_actions:
                    env_action = (env_action + 1) / 2 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low
                step_results = env.step(env_action)
                if len(step_results) != n_targets:
                    raise RuntimeError(
                        f"_play_episode_multitarget: env.step() returned "
                        f"{len(step_results)} tuples, expected {n_targets}"
                    )
                step_counter += 1
                try:
                    env.render()
                except Exception:
                    # Render is best-effort; the per-virtual-env wrapper
                    # may not implement it.
                    pass

                for k, (obs_k, r_k, term_k, trunc_k, info_k) in enumerate(
                    step_results
                ):
                    if done[k]:
                        # Already terminated; skip recording. The
                        # MultiTargetEnv5 internally short-circuits step
                        # computation for done virtual envs.
                        continue
                    flat_obs_k, _ = flatten_obs(obs_k)
                    flat_obs_list[k] = flat_obs_k
                    episodes[k].add_transition(
                        obs_k, flat_obs_k, action, r_k, bool(term_k),
                        bool(trunc_k), info_k,
                    )
                    if term_k or trunc_k:
                        done[k] = True

                # Refresh primary flat_obs for next action sampling
                if not done[0]:
                    primary_flat_obs = flat_obs_list[0]
                if all(done):
                    break

        return episodes, step_counter

    def to(self, device: torch.device):
        self.device = device
        self.algo.to(device)

    def close(self):
        self.env_eval.close()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
        env_eval: Optional[gym.Env] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo: Algo = confighandler.config_dict_to_object(cp["algo"])
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_eval = env_eval or eve_cfh.config_dict_to_object(cp["env_eval"])
        agent = cls(
            algo.to_play_only(),
            env_eval,
            device,
            normalize_actions,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent


class Single(SingleEvalOnly, Agent):
    def __init__(  # pylint: disable=super-init-not-called
        self,
        algo: Algo,
        env_train: gym.Env,
        env_eval: gym.Env,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_train = env_train
        self.env_eval = env_eval
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions

        self.update_error = False

        self.step_counter = StepCounter()
        self.episode_counter = EpisodeCounter()
        self.to(device)
        self._next_batch = None
        self._replay_too_small = True

        # Diagnostics logger (set externally by subprocess runner)
        self.diagnostics = None

        self.logger.info("Single agent initialized")

    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
        episode_schedule: Optional[List[Tuple[Optional[int], Optional[Dict[str, Any]]]]] = None,
        heatup_save_every: int = 0,
        heatup_save_path: Optional[str] = None,
        heatup_stop=None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_heatup(
            steps,
            step_limit,
            episodes,
            episode_limit,
            custom_action_low=custom_action_low,
            custom_action_high=custom_action_high,
        )
        step_limit, episode_limit = self._log_and_convert_limits(
            "heatup", steps, step_limit, episodes, episode_limit
        )

        episodes_data = []

        # Plan v12 R16 — per-episode action RNG. The previous implementation
        # called `np.random.uniform(...)` against the GLOBAL numpy RNG;
        # under Linux fork (default mp.set_start_method), every worker
        # inherits the parent's identical global state at fork time, so all
        # N workers replay the SAME action sequence episode-by-episode →
        # multi-worker compute spent on 1× yield. The fix derives a fresh
        # Generator per (worker_id, episode_idx) from a base seed; same
        # base_seed reproduces the harvest, distinct (worker_id, episode_idx)
        # pairs produce disjoint streams.
        #
        # Worker spawn-time seeding (mirrors this strategy but seeds the
        # global RNG for any consumer that still touches it) lives in
        # singelagentprocess.py's _subprocess_func.
        worker_id = int(getattr(self, "_worker_id", 0))
        base_seed = int(getattr(self, "_heatup_base_seed", 42))
        ep_counter = {"i": 0}  # closure-captured episode index
        # Plan v13 fix — the per-episode Generator is created ONCE per episode
        # (in the episode loop below, where ep_counter is set) and STORED here
        # so successive random_action() calls DRAW FROM IT, advancing the
        # stream. The prior R16 code re-seeded a fresh Generator from the
        # (episode-constant) seed on EVERY call, so every block returned the
        # IDENTICAL draw → the action was frozen for the whole episode: each
        # device got one constant velocity, and a single negative draw pinned
        # that device at insertion 0 for the entire trajectory (the "only one
        # device moves" pathology). Drawing from a persistent per-episode
        # Generator restores the intended per-`consecutive_action_steps`
        # resampling while keeping the harvest reproducible per
        # (base_seed, worker_id, episode_idx).
        ep_rng = {"gen": np.random.default_rng(base_seed + worker_id * 1_000_003)}

        def random_action(*args, **kwargs):  # pylint: disable=unused-argument
            env_low = self.env_train.action_space.low.reshape(-1)
            env_high = self.env_train.action_space.high.reshape(-1)

            if custom_action_low is not None:
                action_low = np.array(custom_action_low).reshape(-1)
            else:
                action_low = env_low.reshape(-1)
            if custom_action_high is not None:
                action_high = np.array(custom_action_high).reshape(-1)
            else:
                action_high = env_high.reshape(-1)
            # Plan v13 fix — draw from the persistent per-episode Generator
            # (created in the episode loop) so each call ADVANCES the stream and
            # the action actually resamples every consecutive_action_steps,
            # instead of re-seeding to the same episode-constant value on every
            # call (which froze the action for the whole episode). The 1_000_003
            # prime stride in the per-episode seed still guarantees disjoint
            # streams across workers.
            action = ep_rng["gen"].uniform(action_low, action_high)

            if self.normalize_actions:
                action = 2 * (action - env_low) / (env_high - env_low) - 1

            return action

        # Plan v12 Stage 1 — detect multi-target heatup mode by duck-typing
        # the env. MultiTargetEnv5 exposes .secondaries; standard BenchEnv5
        # does not. We dispatch to _play_episode_multitarget() and emit
        # N+1 Episodes per SOFA episode in that case.
        is_multitarget = hasattr(self.env_train, "secondaries") and hasattr(
            self.env_train, "primary"
        )

        n_episodes = 0
        n_steps = 0
        sched_idx = 0

        # Plan v12 Stage 2 — worker-side ROLLING save. For an indefinite
        # --heatup_only multi-target harvest the worker never returns, so the
        # only memory-safe place to persist + free episodes is here. Every
        # `heatup_save_every` SOFA episodes (and once more on ANY loop exit —
        # episode_limit, `heatup_stop`/docker-stop, or exception) we write the
        # accumulated batch to per-daughter Version-A + Version-B chunk files
        # and CLEAR the in-memory list → worker RAM stays O(heatup_save_every).
        chunk_idx = 0
        rolling = bool(
            heatup_save_every and heatup_save_path and is_multitarget
        )
        ordered_shorts = None
        if rolling:
            try:
                ordered_shorts = self.env_train._ordered_target_shorts()
            except Exception:
                ordered_shorts = None

        def _flush_heatup_batch():
            nonlocal chunk_idx
            if not rolling or not episodes_data:
                return
            try:
                from ..util.experience_cache import save_heatup_batch
                save_heatup_batch(
                    list(episodes_data),
                    heatup_save_path,
                    ordered_shorts=ordered_shorts,
                    worker_id=int(getattr(self, "_worker_id", 0)),
                    chunk_idx=chunk_idx,
                    rolling=True,
                )
                chunk_idx += 1
            except Exception as exc:  # never let saving break the harvest
                try:
                    self.logger.warning(f"heatup rolling save failed: {exc}")
                except Exception:
                    pass
            finally:
                # Free the batch regardless so an indefinite run cannot OOM.
                episodes_data.clear()

        try:
            while (
                self.step_counter.heatup < step_limit
                and self.episode_counter.heatup < episode_limit
                and not (heatup_stop is not None and heatup_stop.is_set())
            ):
                with self.episode_counter.lock:
                    self.episode_counter.heatup += 1

                # Consume per-episode seed/options from schedule if available
                ep_seed = None
                ep_options = None
                if episode_schedule is not None and sched_idx < len(episode_schedule):
                    ep_seed, ep_options = episode_schedule[sched_idx]
                    sched_idx += 1

                # Plan v12 R16 / v13 — (re)create the per-episode RNG so this
                # episode's random_action draws form a fresh stream distinct from
                # the prior episode's. Same (base_seed, worker_id, episode_idx)
                # reproduces the exact action SEQUENCE; the stream now ADVANCES per
                # draw (fix above) so the action varies within the episode at the
                # consecutive_action_steps cadence.
                ep_counter["i"] = n_episodes
                ep_rng["gen"] = np.random.default_rng(
                    base_seed + worker_id * 1_000_003 + n_episodes
                )

                if is_multitarget:
                    # Plan v12 multi-target heatup fan-out: one SOFA episode →
                    # N+1 Episode objects (one per virtual env), each tagged
                    # with target_branch_idx and individually pushed to the
                    # buffer. runner.py's heatup-save block partitions on
                    # target_branch_idx to emit 4 per-target .npz files.
                    episode_list, n_steps_episode = self._play_episode_multitarget(
                        env=self.env_train,
                        action_function=random_action,
                        consecutive_actions=self.consecutive_action_steps,
                        seed=ep_seed,
                        options=ep_options,
                    )
                    with self.step_counter.lock:
                        self.step_counter.heatup += n_steps_episode
                    n_steps += n_steps_episode
                    n_episodes += 1
                    for ep_k in episode_list:
                        self.replay_buffer.push(ep_k)
                        episodes_data.append(ep_k)
                    if rolling and (n_episodes % heatup_save_every == 0):
                        _flush_heatup_batch()
                else:
                    episode, n_steps_episode = self._play_episode(
                        env=self.env_train,
                        action_function=random_action,
                        consecutive_actions=self.consecutive_action_steps,
                        seed=ep_seed,
                        options=ep_options,
                    )

                    with self.step_counter.lock:
                        self.step_counter.heatup += n_steps_episode
                    n_steps += n_steps_episode
                    n_episodes += 1
                    self.replay_buffer.push(episode)
                    episodes_data.append(episode)
        finally:
            # Final partial (<heatup_save_every) batch — runs on episode_limit
            # exhaustion, heatup_stop / docker-stop, OR an exception. This is the
            # flush that captures the in-flight remainder on SIGTERM.
            _flush_heatup_batch()

        t_duration = perf_counter() - t_start
        self._log_task_completion("heatup", n_steps, t_duration, n_episodes)
        return episodes_data

    def heuristic_seed(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        heuristic_factory=None,
        episode_schedule: Optional[List[Tuple[Optional[int], Optional[Dict[str, Any]]]]] = None,
        push_to_buffer: bool = True,
    ) -> List[Episode]:
        """Seed replay buffer with heuristic-guided episodes.

        Similar to heatup() but uses a heuristic controller instead of
        random actions. The heuristic follows the centerline path.

        Args:
            steps: Target number of steps (mutually exclusive with episodes)
            episodes: Target number of episodes (mutually exclusive with steps)
            step_limit: Maximum steps per episode
            episode_limit: Maximum number of episodes
            heuristic_factory: Factory object with .create(env) method that
                returns a callable action function
            episode_schedule: List of (seed, options) per episode. Each entry
                provides a unique seed and optional options (e.g. target_branch)
                for that episode. If None, episodes run without explicit seeds.

        Returns:
            List of collected episodes
        """
        t_start = perf_counter()
        self.logger.info(
            f"Heuristic seeding: {steps=}, {episodes=}, {step_limit=}, {episode_limit=}"
        )
        # Use "heatup" counters since heuristic seeding is a pre-training phase
        step_limit, episode_limit = self._log_and_convert_limits(
            "heatup", steps, step_limit, episodes, episode_limit
        )

        episodes_data = []

        # Create heuristic action function for this environment
        if heuristic_factory is not None:
            heuristic_action = heuristic_factory.create(self.env_train)
        else:
            raise ValueError(
                "heuristic_factory is required for heuristic_seed. "
                "Pass a HeuristicActionFunctionFactory instance."
            )

        n_episodes = 0
        n_steps = 0
        sched_idx = 0
        while (
            self.step_counter.heatup < step_limit
            and self.episode_counter.heatup < episode_limit
        ):
            with self.episode_counter.lock:
                self.episode_counter.heatup += 1

            # Reset heuristic at episode start
            heuristic_action.reset()

            # Consume per-episode seed/options from schedule if available
            ep_seed = None
            ep_options = None
            if episode_schedule is not None and sched_idx < len(episode_schedule):
                ep_seed, ep_options = episode_schedule[sched_idx]
                sched_idx += 1

            episode, n_steps_episode = self._play_episode(
                env=self.env_train,
                action_function=heuristic_action,
                consecutive_actions=self.consecutive_action_steps,
                seed=ep_seed,
                options=ep_options,
            )

            with self.step_counter.lock:
                self.step_counter.heatup += n_steps_episode
            n_steps += n_steps_episode
            n_episodes += 1
            if push_to_buffer:
                # Plan v11 — heuristic_seed produces demonstrations; tag the
                # episode so PER's demo_priority_bonus protects them from
                # being statistically forgotten as the buffer fills with
                # explore data. Episode.to_replay() forwards is_demo to
                # EpisodeReplay (replaybuffer.py fix); the buffer's push()
                # then reads it via getattr.
                episode.is_demo = True
                self.replay_buffer.push(episode)
            episodes_data.append(episode)

        t_duration = perf_counter() - t_start
        self._log_task_completion("heatup", n_steps, t_duration, n_episodes)
        return episodes_data

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        self._log_exploration(steps, step_limit, episodes, episode_limit)
        step_limit, episode_limit = self._log_and_convert_limits(
            "exploration", steps, step_limit, episodes, episode_limit
        )

        episodes_data = []
        n_episodes = 0
        n_steps = 0
        while (
            self.step_counter.exploration < step_limit
            and self.episode_counter.exploration < episode_limit
        ):
            with self.episode_counter.lock:
                self.episode_counter.exploration += 1

            episode, n_steps_episode = self._play_episode(
                env=self.env_train,
                action_function=self.algo.get_exploration_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.exploration += n_steps_episode
                # Stamp episode with step counters at completion time (inside lock
                # to avoid race where another worker increments before we read)
                episode.explore_step_at_completion = self.step_counter.exploration
            episode.update_step_at_completion = self.step_counter.update

            n_episodes += 1
            n_steps += n_steps_episode

            self.replay_buffer.push(episode)
            episodes_data.append(episode)

        t_duration = perf_counter() - t_start
        self._log_task_completion("exploration", n_steps, t_duration, n_episodes)
        return episodes_data

    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> List[List[float]]:
        t_start = perf_counter()
        self._log_update(steps, step_limit)
        step_limit, _ = self._log_and_convert_limits("update", steps, step_limit)
        results = []
        if self._replay_too_small:
            replay_len = len(self.replay_buffer)
            batch_size = self.replay_buffer.batch_size
            self._replay_too_small = replay_len <= batch_size
        if self._replay_too_small or step_limit == 0 or steps == 0:
            return []

        n_steps = 0

        while self.step_counter.update < step_limit:
            with self.step_counter.lock:
                self.step_counter.update += 1
            batch = self.replay_buffer.sample()
            result = self.algo.update(batch)
            results.append(result)
            n_steps += 1

            # Plan v7 (PER): feed fresh per-sample |TD| back so the buffer
            # can refresh the sampled transitions' priorities. Guarded —
            # uniform buffers have no `update_priorities` and `batch.indices`
            # is None, so this is a no-op for episode/step-uniform runs.
            if (
                batch.indices is not None
                and getattr(self.algo, "last_td_errors", None) is not None
                and hasattr(self.replay_buffer, "update_priorities")
            ):
                self.replay_buffer.update_priorities(
                    batch.indices, self.algo.last_td_errors
                )

            # Log diagnostics after each gradient step
            if self.diagnostics is not None:
                self.diagnostics.log_losses(
                    update_step=self.step_counter.update,
                    explore_step=self.step_counter.exploration,
                    q1_loss=float(result[0]),
                    q2_loss=float(result[1]),
                    policy_loss=float(result[2]),
                    **self.algo.last_metrics,
                )

                # Log probe values periodically
                if (self.diagnostics.probe_states is not None and
                    self.step_counter.update % self.diagnostics.config.log_probe_values_every_n_steps == 0):
                    probe_values = self.algo.compute_probe_values(
                        self.diagnostics.probe_states
                    )
                    self.diagnostics.log_probe_values(
                        update_step=self.step_counter.update,
                        probe_values=probe_values,
                        explore_step=self.step_counter.exploration,  # NEW: Added for timestamp correlation
                    )

                # Log batch samples periodically
                if self.step_counter.update % self.diagnostics.config.log_batch_samples_every_n_steps == 0:
                    try:
                        self._log_batch_samples(batch)
                    except Exception as e:
                        self.logger.warning(f"Failed to log batch samples at step {self.step_counter.update}: {e}")

        t_duration = perf_counter() - t_start
        self._log_task_completion("update", n_steps, t_duration)
        return results

    def _log_batch_samples(self, batch):
        """Log sampled transitions from the current batch."""
        if self.diagnostics is None:
            self.logger.debug("_log_batch_samples: diagnostics is None, skipping")
            return

        # `batch` is a 7-field NamedTuple (PER added is_weights/indices);
        # access by name rather than a positional 5-unpack.
        all_states = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        dones = batch.terminals
        padding_mask = batch.padding_mask

        # Log batch shape info for debugging (only first time)
        if self.step_counter.update <= 100:
            self.logger.info(
                f"_log_batch_samples: batch shapes - states={all_states.shape}, "
                f"actions={actions.shape}, rewards={rewards.shape}, dones={dones.shape}"
            )

        # Determine number of samples to log
        n_samples = min(
            self.diagnostics.config.n_batch_samples,
            all_states.shape[0]
        )

        # Sample random indices from the batch
        batch_size = all_states.shape[0]
        sample_indices = np.random.choice(batch_size, size=n_samples, replace=False).tolist()

        # Get next states (shifted by 1 in sequence dimension)
        seq_length = actions.shape[1]
        next_states = all_states[:, 1:seq_length+1, :] if all_states.shape[1] > seq_length else None

        # Compute actor/critic values for sampled transitions
        batch_samples = self.algo.compute_batch_sample_values(
            states=all_states[:, :seq_length, :],
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_states=next_states,
            sample_indices=sample_indices,
        )

        # Log info about computed samples
        if self.step_counter.update <= 100:
            self.logger.info(f"_log_batch_samples: computed {len(batch_samples)} samples")

        # Log the samples
        self.diagnostics.log_batch_samples(
            update_step=self.step_counter.update,
            explore_step=self.step_counter.exploration,
            batch_samples=batch_samples,
        )

        if self.step_counter.update <= 100:
            self.logger.info(f"_log_batch_samples: successfully logged at step {self.step_counter.update}")

    def explore_and_update(
        self,
        *,
        explore_steps: Optional[int] = None,
        explore_episodes: Optional[int] = None,
        explore_step_limit: Optional[int] = None,
        explore_episode_limit: Optional[int] = None,
        update_steps: Optional[int] = None,
        update_step_limit: Optional[int] = None,
    ) -> Tuple[List[Episode], List[float]]:
        explore_result = self.explore(
            steps=explore_steps,
            episodes=explore_episodes,
            step_limit=explore_step_limit,
            episode_limit=explore_episode_limit,
        )
        update_result = self.update(steps=update_steps, step_limit=update_step_limit)
        return explore_result, update_result

    def close(self):
        # Close diagnostics logger if present
        if self.diagnostics is not None:
            self.diagnostics.close()

        self.env_train.close()
        if id(self.env_train) != id(self.env_eval):
            self.env_eval.close()
        self.replay_buffer.close()
        del self.algo
        del self.replay_buffer

    @classmethod
    def from_checkpoint(  # pylint: disable=arguments-renamed
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        env_train: Optional[gym.Env] = None,
        env_eval: Optional[gym.Env] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo = confighandler.config_dict_to_object(cp["algo"])
        replay_buffer = replay_buffer or confighandler.config_dict_to_object(
            cp["replay_buffer"]
        )
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_train = env_train or eve_cfh.config_dict_to_object(cp["env_train"])
        env_eval = env_eval or eve_cfh.config_dict_to_object(cp["env_eval"])
        agent = cls(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            device,
            consecutive_action_steps,
            normalize_actions,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent
