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
            action = np.random.uniform(action_low, action_high)

            if self.normalize_actions:
                action = 2 * (action - env_low) / (env_high - env_low) - 1

            return action

        n_episodes = 0
        n_steps = 0
        sched_idx = 0
        while (
            self.step_counter.heatup < step_limit
            and self.episode_counter.heatup < episode_limit
        ):
            with self.episode_counter.lock:
                self.episode_counter.heatup += 1

            # Consume per-episode seed/options from schedule if available
            ep_seed = None
            ep_options = None
            if episode_schedule is not None and sched_idx < len(episode_schedule):
                ep_seed, ep_options = episode_schedule[sched_idx]
                sched_idx += 1

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
