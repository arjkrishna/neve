from multiprocessing import Process, Queue
from queue import Empty
from time import perf_counter
from typing import Any, Dict, List, Optional, Union
import logging
import torch
import gymnasium as gym

from .single import Single, SingleEvalOnly
from .agent import StepCounter, EpisodeCounter
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, EpisodeReplay


def run(
    algo_config: dict,
    replay_buffer_config: dict,
    env_train_config: dict,
    env_eval_config: dict,
    device: torch.device,
    consecutive_action_steps: int,
    normalize_actions: bool,
    receive_q: Queue,
    send_q: Queue,
    step_counter: StepCounter,
    episode_counter: EpisodeCounter,
    name: str = "SingeAgentProcess",
    loglevel: int = logging.DEBUG,
):
    logging.basicConfig(level=loglevel)
    from ..util import ConfigHandler
    from importlib import import_module

    eve = import_module("eve.util")
    eve_cfh = eve.ConfigHandler()
    cfh = ConfigHandler()
    algo: Algo = cfh.config_dict_to_object(algo_config)
    replay_buffer: ReplayBuffer = cfh.config_dict_to_object(replay_buffer_config)
    env_train: gym.Env = eve_cfh.config_dict_to_object(env_train_config)
    env_eval: gym.Env = eve_cfh.config_dict_to_object(env_eval_config)

    agent = Single(
        algo=algo,
        env_train=env_train,
        env_eval=env_eval,
        replay_buffer=replay_buffer,
        device=device,
        consecutive_action_steps=consecutive_action_steps,
        normalize_actions=normalize_actions,
    )
    agent.step_counter = step_counter
    agent.episode_counter = episode_counter
    run_loop(receive_q, send_q, agent, name)


def run_loop(receive_q: Queue, send_q: Queue, agent: Single, name: str):
    logger = logging.getLogger(name)
    logger.info("Starting agent process: %s", name)
    while True:
        task = receive_q.get()
        task_name = task["task"]
        task_kwargs = task["kwargs"]
        logger.debug("Processing task: %s with kwargs: %s", task_name, task_kwargs)
        if task_name == "close":
            agent.close()
            logger.debug("Terminating agent process")
            break
        if task_name == "heatup":
            result = agent.heatup(**task_kwargs)
        elif task_name == "explore":
            result = agent.explore(**task_kwargs)
        elif task_name == "update":
            result = agent.update(**task_kwargs)
        elif task_name == "evaluate":
            result = agent.evaluate(**task_kwargs)
        elif task_name == "explore_and_update":
            result = agent.explore_and_update(**task_kwargs)
        elif task_name == "get_state_dicts":
            result = agent.get_state_dicts()
        elif task_name == "set_state_dicts":
            agent.set_state_dicts(**task_kwargs)
            result = []
        elif task_name == "push_episodes":
            replay_episodes = task_kwargs.get("replay_episodes", None)
            if replay_episodes is not None:
                for episode_replay in replay_episodes:
                    agent.replay_buffer.push_episode_replay(episode_replay)
            result = []
        else:
            logger.error("Unknown task: %s", task_name)
            result = None
        send_q.put(result)


class SingleAgentProcess:
    def __init__(
        self,
        algo: Algo,
        env_train: gym.Env,
        env_eval: gym.Env,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        step_counter: StepCounter = None,
        episode_counter: EpisodeCounter = None,
        name: str = "SingleAgentProcess",
        loglevel: int = logging.DEBUG,
    ):
        self.send_q = Queue()
        self.receive_q = Queue()

        self.step_counter = step_counter if step_counter else StepCounter()
        self.episode_counter = episode_counter if episode_counter else EpisodeCounter()

        from ..util import ConfigHandler

        cfh = ConfigHandler()

        algo_config = cfh.object_to_config_dict(algo)
        replay_buffer_config = cfh.object_to_config_dict(replay_buffer)
        env_train_config = cfh.object_to_config_dict(env_train)
        env_eval_config = cfh.object_to_config_dict(env_eval)

        self._process = Process(
            target=run,
            args=(
                algo_config,
                replay_buffer_config,
                env_train_config,
                env_eval_config,
                device,
                consecutive_action_steps,
                normalize_actions,
                self.send_q,
                self.receive_q,
                self.step_counter,
                self.episode_counter,
                name,
                loglevel,
            ),
        )

    def start(self):
        self._process.start()

    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
    ):
        task_dict = {
            "task": "heatup",
            "kwargs": {
                "steps": steps,
                "episodes": episodes,
                "step_limit": step_limit,
                "episode_limit": episode_limit,
                "custom_action_low": custom_action_low,
                "custom_action_high": custom_action_high,
            },
        }
        self.send_q.put(task_dict)

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ):
        task_dict = {
            "task": "explore",
            "kwargs": {
                "steps": steps,
                "episodes": episodes,
                "step_limit": step_limit,
                "episode_limit": episode_limit,
            },
        }
        self.send_q.put(task_dict)

    def update(self, *, steps: Optional[int] = None, step_limit: Optional[int] = None):
        task_dict = {
            "task": "update",
            "kwargs": {"steps": steps, "step_limit": step_limit},
        }
        self.send_q.put(task_dict)

    def evaluate(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ):
        task_dict = {
            "task": "evaluate",
            "kwargs": {
                "steps": steps,
                "episodes": episodes,
                "step_limit": step_limit,
                "episode_limit": episode_limit,
                "seeds": seeds,
                "options": options,
            },
        }
        self.send_q.put(task_dict)

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
        task_dict = {
            "task": "explore_and_update",
            "kwargs": {
                "explore_steps": explore_steps,
                "explore_episodes": explore_episodes,
                "explore_step_limit": explore_step_limit,
                "explore_episode_limit": explore_episode_limit,
                "update_steps": update_steps,
                "update_step_limit": update_step_limit,
            },
        }
        self.send_q.put(task_dict)

    def get_state_dicts(self):
        task_dict = {"task": "get_state_dicts", "kwargs": {}}
        self.send_q.put(task_dict)

    def set_state_dicts(
        self, network_state_dicts, optimizer_state_dicts, scheduler_state_dicts
    ):
        task_dict = {
            "task": "set_state_dicts",
            "kwargs": {
                "network_state_dicts": network_state_dicts,
                "optimizer_state_dicts": optimizer_state_dicts,
                "scheduler_state_dicts": scheduler_state_dicts,
            },
        }
        self.send_q.put(task_dict)

    def push_episodes(self, replay_episodes: List[EpisodeReplay]):
        task_dict = {
            "task": "push_episodes",
            "kwargs": {"replay_episodes": replay_episodes},
        }
        self.send_q.put(task_dict)

    def get_result(self, timeout: Union[int, None] = None):
        try:
            result = self.receive_q.get(timeout=timeout)
        except Empty:
            result = None
        return result

    def close(self):
        task_dict = {"task": "close", "kwargs": {}}
        self.send_q.put(task_dict)
        self._process.join()
