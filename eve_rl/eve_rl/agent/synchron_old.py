from copy import deepcopy
from typing import Any, Dict, List, Optional
import logging
import torch
import gymnasium as gym

from .agent import Agent, StepCounter, EpisodeCounter
from .singelagentprocess import SingleAgentProcess
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, Episode


class Synchron(Agent):
    def __init__(
        self,
        trainer_algo: Algo,
        worker_algo: Algo,
        trainer_env_train: gym.Env,
        worker_env_train: gym.Env,
        env_eval: gym.Env,
        trainer_replay_buffer: ReplayBuffer,
        worker_replay_buffer: ReplayBuffer,
        trainer_device: torch.device = torch.device("cpu"),
        worker_device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        n_worker: int = 1,
        trainer_loglevel: int = logging.DEBUG,
        worker_loglevel: int = logging.DEBUG,
    ):
        self.logger = logging.getLogger(self.__module__)
        self.trainer_algo = trainer_algo
        self.worker_algo = worker_algo
        self.trainer_env_train = trainer_env_train
        self.worker_env_train = worker_env_train
        self.env_eval = env_eval
        self.trainer_replay_buffer = trainer_replay_buffer
        self.worker_replay_buffer = worker_replay_buffer
        self.trainer_device = trainer_device
        self.worker_device = worker_device
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions
        self.n_worker = n_worker
        self.trainer_loglevel = trainer_loglevel
        self.worker_loglevel = worker_loglevel

        self.step_counter = StepCounter()
        self.episode_counter = EpisodeCounter()
        self._trainer_agent = self._create_trainer_agent()
        self._worker_agents: List[SingleAgentProcess] = []
        for i_worker in range(n_worker):
            self._create_worker(i_worker)

    def _create_worker(self, i_worker):
        worker_agent = SingleAgentProcess(
            algo=deepcopy(self.worker_algo),
            env_train=deepcopy(self.worker_env_train),
            env_eval=deepcopy(self.env_eval),
            replay_buffer=deepcopy(self.worker_replay_buffer),
            device=self.worker_device,
            consecutive_action_steps=self.consecutive_action_steps,
            normalize_actions=self.normalize_actions,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
            name=f"worker_{i_worker}",
            loglevel=self.worker_loglevel,
        )
        worker_agent.start()
        self._worker_agents.append(worker_agent)
        return worker_agent

    def _create_trainer_agent(self):
        trainer_agent = SingleAgentProcess(
            algo=deepcopy(self.trainer_algo),
            env_train=deepcopy(self.trainer_env_train),
            env_eval=deepcopy(self.env_eval),
            replay_buffer=deepcopy(self.trainer_replay_buffer),
            device=self.trainer_device,
            consecutive_action_steps=self.consecutive_action_steps,
            normalize_actions=self.normalize_actions,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
            name="trainer",
            loglevel=self.trainer_loglevel,
        )
        trainer_agent.start()
        return trainer_agent

    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
    ) -> List[Episode]:
        step_limit = (
            step_limit if step_limit is not None else self.step_counter.heatup + steps
        )
        episode_limit = (
            episode_limit
            if episode_limit is not None
            else self.episode_counter.heatup + episodes
            if episodes is not None
            else None
        )

        for agent in self._worker_agents:
            agent.heatup(
                step_limit=step_limit,
                episode_limit=episode_limit,
                custom_action_low=custom_action_low,
                custom_action_high=custom_action_high,
            )
        all_episodes = []
        for agent in self._worker_agents:
            episodes_data = agent.get_result()
            all_episodes.extend(episodes_data)

        self._push_episodes_to_trainer(all_episodes)
        return all_episodes

    def _push_episodes_to_trainer(self, episodes_data: List[Episode]):
        replay_episodes = [episode.to_replay() for episode in episodes_data]
        self._trainer_agent.push_episodes(replay_episodes)
        self._trainer_agent.get_result()

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> List[Episode]:
        step_limit = (
            step_limit
            if step_limit is not None
            else self.step_counter.exploration + steps
            if steps is not None
            else None
        )
        episode_limit = (
            episode_limit
            if episode_limit is not None
            else self.episode_counter.exploration + episodes
            if episodes is not None
            else None
        )

        for agent in self._worker_agents:
            agent.explore(
                step_limit=step_limit,
                episode_limit=episode_limit,
            )
        all_episodes = []
        for agent in self._worker_agents:
            episodes_data = agent.get_result()
            all_episodes.extend(episodes_data)

        return all_episodes

    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> List[List[float]]:
        step_limit = (
            step_limit if step_limit is not None else self.step_counter.update + steps
        )

        self._trainer_agent.update(step_limit=step_limit)
        update_data = self._trainer_agent.get_result()
        return update_data

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
        self._trainer_agent.evaluate(
            steps=steps,
            episodes=episodes,
            step_limit=step_limit,
            episode_limit=episode_limit,
            seeds=seeds,
            options=options,
        )
        eval_results = self._trainer_agent.get_result()
        return eval_results

    def explore_and_update(
        self,
        *,
        explore_steps: Optional[int] = None,
        explore_episodes: Optional[int] = None,
        explore_step_limit: Optional[int] = None,
        explore_episode_limit: Optional[int] = None,
        update_steps: Optional[int] = None,
        update_step_limit: Optional[int] = None,
    ):
        explore_step_limit = (
            explore_step_limit
            if explore_step_limit is not None
            else self.step_counter.exploration + explore_steps
            if explore_steps is not None
            else None
        )
        explore_episode_limit = (
            explore_episode_limit
            if explore_episode_limit is not None
            else self.episode_counter.exploration + explore_episodes
            if explore_episodes is not None
            else None
        )
        update_step_limit = (
            update_step_limit
            if update_step_limit is not None
            else self.step_counter.update + update_steps
            if update_steps is not None
            else None
        )

        self._trainer_agent.update(step_limit=update_step_limit)
        for agent in self._worker_agents:
            agent.explore(
                step_limit=explore_step_limit,
                episode_limit=explore_episode_limit,
            )
        update_data = self._trainer_agent.get_result()

        all_episodes = []
        for agent in self._worker_agents:
            episodes_data = agent.get_result()
            all_episodes.extend(episodes_data)

        self._push_episodes_to_trainer(all_episodes)

        self._sync_network_params()

        return all_episodes, update_data

    def _sync_network_params(self):
        self._trainer_agent.get_state_dicts()
        state_dicts = self._trainer_agent.get_result()
        network_state_dicts = state_dicts["network"]
        optimizer_state_dicts = state_dicts["optimizer"]
        scheduler_state_dicts = state_dicts["scheduler"]

        for worker in self._worker_agents:
            worker.set_state_dicts(
                network_state_dicts, optimizer_state_dicts, scheduler_state_dicts
            )
        for worker in self._worker_agents:
            worker.get_result()

    def get_state_dicts(self):
        self._trainer_agent.get_state_dicts()
        state_dicts = self._trainer_agent.get_result()
        return state_dicts

    def set_state_dicts(
        self, network_state_dicts, optimizer_state_dicts, scheduler_state_dicts
    ):
        self._trainer_agent.set_state_dicts(
            network_state_dicts, optimizer_state_dicts, scheduler_state_dicts
        )
        self._trainer_agent.get_result()

        for worker in self._worker_agents:
            worker.set_state_dicts(
                network_state_dicts, optimizer_state_dicts, scheduler_state_dicts
            )
        for worker in self._worker_agents:
            worker.get_result()

    def close(self):
        for worker in self._worker_agents:
            worker.close()
        self._trainer_agent.close()
