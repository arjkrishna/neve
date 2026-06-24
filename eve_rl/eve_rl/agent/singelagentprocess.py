import platform
from typing import Any, Dict, List, Optional, Tuple
from random import randint
import logging
import logging.config
import os
import traceback
import queue

import numpy as np  # Plan v12 R16 — needed for spawn-time np.random.seed
from torch import multiprocessing as mp
import torch

from .agent import (
    Agent,
    EpisodeCounterShared,
    StepCounterShared,
    StepCounter,
    EpisodeCounter,
)
from .single import Single, Algo, ReplayBuffer, gym
from ..replaybuffer import Episode


def _move_state_dict_to_cpu(state_dict):
    """
    Recursively move all tensors in a state dict to CPU.

    This is necessary because CUDA tensors cannot be sent through
    multiprocessing queues - they cause 'invalid resource handle' errors.

    Args:
        state_dict: A dict, tensor, or other value

    Returns:
        The same structure with all tensors moved to CPU
    """
    if isinstance(state_dict, torch.Tensor):
        return state_dict.cpu()
    elif isinstance(state_dict, dict):
        return {k: _move_state_dict_to_cpu(v) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [_move_state_dict_to_cpu(v) for v in state_dict]
    elif isinstance(state_dict, tuple):
        return tuple(_move_state_dict_to_cpu(v) for v in state_dict)
    else:
        return state_dict


def file_handler_callback(handler: logging.FileHandler):
    handler_dict = {
        handler.name: {
            "level": handler.level,
            "class": "logging.FileHandler",
            "filename": handler.baseFilename,
            "mode": handler.mode,
        }
    }
    if handler.formatter is not None:
        formatter_name = handler.name or randint(1, 99999)
        handler_dict[handler.name]["formatter"] = str(formatter_name)
        # pylint: disable=protected-access
        formatter_dict = {str(formatter_name): {"format": handler.formatter._fmt}}
    else:
        formatter_dict = None
    return handler_dict, formatter_dict, handler.name


handler_callback = {logging.FileHandler: file_handler_callback}


def get_logging_config_dict():
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {},
        "handlers": {},
        "loggers": {
            "": {
                "handlers": [],
                "level": logging.WARNING,
                "propagate": False,
            },  # root logger
        },
    }

    config["loggers"][""]["level"] = logging.root.level
    config["loggers"][""]["propagate"] = logging.root.propagate
    for handler in logging.root.handlers:
        handler_dict, formatter_dict, name = handler_callback[type(handler)](handler)
        if formatter_dict is not None:
            config["formatters"].update(formatter_dict)
        config["handlers"].update(handler_dict)
        config["loggers"][""]["handlers"].append(name)
    return config


def run(
    algo: Algo,
    env_train: gym.Env,
    env_eval: gym.Env,
    replay_buffer: ReplayBuffer,
    device: torch.device,
    consecutive_action_steps: int,
    normalize_actions,
    log_config_dict: Dict,
    task_queue,
    result_queue,
    model_queue,
    step_counter,
    episode_counter,
    shutdown,
    is_shutdown,
    name,
    nice_level: int,
    diagnostics_config: Optional[Dict] = None,
    heatup_stop=None,
):
    if platform.system() != "Windows":
        os.nice(nice_level)

    try:
        torch.set_num_threads(4)

        # Plan v12 R16 — per-worker spawn-time RNG reseed.
        # Under Linux fork, every child inherits the parent's exact
        # global RNG state at fork time → all workers replay identical
        # action sequences episode-by-episode. Reseed the global
        # numpy / random / torch RNGs from a worker-distinct entropy
        # stack (deterministic so the same base_seed reproduces).
        # Per-episode action sampling in single.py:heatup uses its own
        # np.random.default_rng() Generator keyed on (base_seed,
        # worker_id, episode_idx); this spawn-time reseed protects any
        # other consumer of the global state.
        import random as _py_random
        try:
            _worker_id = int(name.split("_")[-1]) if name else 0
        except Exception:
            _worker_id = 0
        _pid = int(os.getpid()) & 0x3FFFFFFF
        _base_seed = 42  # Plan v12 R16 — same base_seed used by single.py
        _worker_seed = (_base_seed * 1_000_003 + _worker_id * 1_000_033 + _pid) & 0x7FFFFFFF
        np.random.seed(_worker_seed)
        _py_random.seed(_worker_seed)
        torch.manual_seed(_worker_seed)
        for handler_name, handler_config in log_config_dict["handlers"].items():
            if "filename" in handler_config.keys():
                filename = handler_config["filename"]
                path, _ = os.path.split(filename)
                path = os.path.join(path, "logs_subprocesses")
                if not os.path.isdir(path):
                    os.mkdir(path)
                filename = os.path.join(path, f"{name}.log")
                log_config_dict["handlers"][handler_name]["filename"] = filename
        logging.config.dictConfig(log_config_dict)
        logger = logging.getLogger(__name__)
        logger.info("logger initialized")
        agent = Single(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            device,
            consecutive_action_steps,
            normalize_actions,
        )
        agent.step_counter = step_counter
        agent.episode_counter = episode_counter

        # Plan v12 R16 FIX — propagate worker identity onto the agent so
        # single.py:heatup's per-(worker, episode) np.random.default_rng()
        # seed actually varies across workers. Without this, single.py
        # reads `self._worker_id` / `self._heatup_base_seed` via getattr
        # defaults (0 and 42), collapsing every worker's RNG to
        # default_rng(42 + 0*1_000_003 + ep_idx) — MD5-identical action
        # streams (confirmed in Phase 7 smoke test). The two values
        # below were already computed above for the spawn-time global
        # RNG seeding (line ~136 _worker_id, line 140 _base_seed); this
        # just makes them visible to the heatup action loop.
        agent._worker_id = _worker_id
        agent._heatup_base_seed = _base_seed
        logger.info(
            f"Plan v12 R16 — worker RNG: _worker_id={_worker_id}, "
            f"_heatup_base_seed={_base_seed}, _worker_seed={_worker_seed}"
        )

        # Initialize diagnostics logger for trainer subprocess
        if diagnostics_config is not None and diagnostics_config.get("enabled", False):
            from ..util.diagnostics_logger import DiagnosticsLogger, DiagnosticsConfig
            config = DiagnosticsConfig.from_dict(diagnostics_config)
            agent.diagnostics = DiagnosticsLogger(
                config=config,
                process_name=name,
            )
            logger.info(f"Diagnostics logging initialized for {name}")
        while not shutdown.is_set():
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            task_name = task[0]

            if task_name in ["load_state_dicts_network", "state_dicts_network"]:
                log_debug = f"Received {task[0]=} with {len(task)=}"
            else:
                log_debug = f"Received {task=}"
            logger.debug(log_debug)
            if task_name == "heatup":
                result = agent.heatup(
                    steps=task[1],
                    episodes=task[2],
                    step_limit=task[3],
                    episode_limit=task[4],
                    custom_action_low=task[5],
                    custom_action_high=task[6],
                    episode_schedule=task[7],
                    heatup_save_every=task[8] if len(task) > 8 else 0,
                    heatup_save_path=task[9] if len(task) > 9 else None,
                    heatup_stop=heatup_stop,
                )
            elif task_name == "heuristic_seed":
                result = agent.heuristic_seed(
                    steps=task[1],
                    episodes=task[2],
                    step_limit=task[3],
                    episode_limit=task[4],
                    heuristic_factory=task[5],
                    episode_schedule=task[6],
                    push_to_buffer=task[7],
                )
            elif task_name == "explore":
                result = agent.explore(
                    steps=task[1],
                    episodes=task[2],
                    step_limit=task[3],
                    episode_limit=task[4],
                )
            elif task_name == "evaluate":
                result = agent.evaluate(
                    steps=task[1],
                    episodes=task[2],
                    step_limit=task[3],
                    episode_limit=task[4],
                    seeds=task[5],
                    options=task[6],
                )
            elif task_name == "update":
                try:
                    result = agent.update(steps=task[1], step_limit=task[2])
                except ValueError as error:
                    log_warning = f"Update Error: {error}"
                    logger.warning(log_warning)
                    shutdown.set()
                    result = error
            elif task_name == "explore_and_update":
                result = agent.explore_and_update(
                    explore_steps=task[1],
                    explore_episodes=task[2],
                    explore_step_limit=task[3],
                    explore_episode_limit=task[4],
                    update_steps=task[5],
                    update_step_limit=task[6],
                )
            elif task_name == "state_dicts_network":
                destination = task[1]
                state_dicts = agent.algo.state_dicts_network(destination)
                # Move all tensors to CPU before sending through queue
                # CUDA tensors cannot be pickled/sent via multiprocessing queues
                cpu_state_dicts = _move_state_dict_to_cpu(state_dicts)
                model_queue.put(cpu_state_dicts)
                del state_dicts, cpu_state_dicts
                continue
            elif task_name == "load_state_dicts_network":
                state_dicts = task[1]
                agent.algo.load_state_dicts_network(state_dicts)
                del state_dicts
                continue
            elif task_name == "state_dicts_optimizer":
                state_dicts = agent.algo.state_dicts_optimizer()
                # Move all tensors to CPU before sending through queue
                # Optimizer state contains momentum buffers on CUDA
                cpu_state_dicts = _move_state_dict_to_cpu(state_dicts)
                model_queue.put(cpu_state_dicts)
                del state_dicts, cpu_state_dicts
                continue
            elif task_name == "load_state_dicts_optimizer":
                state_dicts = task[1]
                agent.algo.load_state_dicts_optimizer(state_dicts)
                del state_dicts
                continue
            elif task_name == "state_dicts_scheduler":
                state_dicts = agent.algo.state_dicts_scheduler()
                # Move to CPU for safety (schedulers typically don't have CUDA tensors)
                cpu_state_dicts = _move_state_dict_to_cpu(state_dicts)
                model_queue.put(cpu_state_dicts)
                del state_dicts, cpu_state_dicts
                continue
            elif task_name == "load_state_dicts_scheduler":
                state_dicts = task[1]
                agent.algo.load_state_dicts_scheduler(state_dicts)
                del state_dicts
                continue
            elif task_name == "set_probe_states":
                # Set probe states for diagnostics evaluation
                probe_states = task[1]
                if agent.diagnostics is not None:
                    # np already imported at module top (line 10) for the
                    # Plan v12 R16 spawn-time seeding. A redundant local
                    # `import numpy as np` HERE was the silent bug that
                    # made np a local variable for the whole run()
                    # function and triggered UnboundLocalError at the
                    # R16 reseed (Phase 7 smoke crash).
                    agent.diagnostics.set_probe_states(
                        np.array(probe_states),
                        device=device
                    )
                    logger.info(f"Set {len(probe_states)} probe states")
                continue
            elif task_name == "save_policy_snapshot":
                # Save policy-only checkpoint
                snapshot_path = task[1]
                policy_state = agent.algo.model.policy.state_dict()
                torch.save({
                    "policy": policy_state,
                    "update_step": agent.step_counter.update,
                    "explore_step": agent.step_counter.exploration,
                }, snapshot_path)
                logger.info(f"Saved policy snapshot to {snapshot_path}")
                continue
            elif task_name == "flush_diagnostics":
                # Flush diagnostics logs to disk
                if agent.diagnostics is not None:
                    agent.diagnostics.flush()
                continue
            elif task_name == "shutdown":
                break
            else:
                continue
            result_queue.put(result)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        exception_traceback = "".join(traceback.format_tb(exception.__traceback__))
        logger.warning("Traceback:\n" + exception_traceback)
        logger.warning(exception)
        result_queue.put(exception)
    agent.close()

    for queue_ in [result_queue, model_queue, task_queue]:
        while True:
            try:
                queue_.get_nowait()
            except queue.Empty:
                queue_.close()
                break
    is_shutdown.set()


class SingleAgentProcess(Agent):
    def __init__(
        self,
        agent_id: int,
        algo: Algo,
        env_train: gym.Env,
        env_eval: gym.Env,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        consecutive_action_steps: int,
        normalize_actions: bool,
        name: str,
        parent_agent: Agent,
        step_counter: StepCounterShared = None,
        episode_counter: EpisodeCounterShared = None,
        nice_level: int = 0,
        diagnostics_config: Optional[Dict] = None,
        heatup_stop=None,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.agent_id = agent_id
        self.name = name
        self._shutdown = mp.Event()
        self._is_shutdown = mp.Event()
        # Plan v12 Stage 2 — shared (across all workers) stop Event for the
        # rolling heatup harvest. The main-process SIGTERM/docker-stop handler
        # sets it so each worker exits its heatup loop and runs its final-batch
        # flush. Synchron passes ONE Event to every worker; None (legacy
        # callers) → a private, never-set Event (no behaviour change).
        self._heatup_stop = heatup_stop if heatup_stop is not None else mp.Event()
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._model_queue = mp.Queue()

        self.device = device
        self.parent_agent = parent_agent
        self.diagnostics_config = diagnostics_config

        self._step_counter = step_counter or StepCounterShared()
        self._episode_counter = episode_counter or EpisodeCounterShared()
        logging_config = get_logging_config_dict()

        for handler_config in logging_config["handlers"].values():
            if "filename" in handler_config.keys():
                filename = handler_config["filename"]
                path, _ = os.path.split(filename)
                path = os.path.join(path, "logs_subprocesses")
                if not os.path.isdir(path):
                    os.mkdir(path)

        self._process = mp.Process(
            target=run,
            args=[
                algo,
                env_train,
                env_eval,
                replay_buffer,
                device,
                consecutive_action_steps,
                normalize_actions,
                logging_config,
                self._task_queue,
                self._result_queue,
                self._model_queue,
                self.step_counter,
                self.episode_counter,
                self._shutdown,
                self._is_shutdown,
                name,
                nice_level,
                diagnostics_config,
                self._heatup_stop,
            ],
            name=name,
        )
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
        episode_schedule=None,
        heatup_save_every: int = 0,
        heatup_save_path: Optional[str] = None,
    ) -> None:
        self._task_queue.put(
            [
                "heatup",
                steps,
                episodes,
                step_limit,
                episode_limit,
                custom_action_low,
                custom_action_high,
                episode_schedule,
                heatup_save_every,
                heatup_save_path,
            ]
        )

    def heuristic_seed(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        heuristic_factory=None,
        episode_schedule=None,
        push_to_buffer=True,
    ) -> None:
        """Send heuristic seeding task to worker."""
        self._task_queue.put(
            [
                "heuristic_seed",
                steps,
                episodes,
                step_limit,
                episode_limit,
                heuristic_factory,
                episode_schedule,
                push_to_buffer,
            ]
        )

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> None:
        try:
            self._task_queue.put(
                ["explore", steps, episodes, step_limit, episode_limit]
            )
        except ValueError:
            self.close()

    def evaluate(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        try:
            self._task_queue.put(
                ["evaluate", steps, episodes, step_limit, episode_limit, seeds, options]
            )
        except ValueError:
            self.close()

    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> None:
        try:
            self._task_queue.put(["update", steps, step_limit])
        except ValueError:
            self.close()

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
        try:
            self._task_queue.put(
                [
                    "explore_and_update",
                    explore_steps,
                    explore_episodes,
                    explore_step_limit,
                    explore_episode_limit,
                    update_steps,
                    update_step_limit,
                ]
            )
        except ValueError:
            self.close()

    def get_result(self, timeout: float) -> List[Any]:
        try:
            result = self._result_queue.get(timeout=timeout)
        except queue.Empty as error:
            result = error
        except ValueError:
            self.close()
            result = []
        return result

    def state_dicts_network(self, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            self._task_queue.put(["state_dicts_network", destination])
            return self._model_queue.get()
        except ValueError:
            self.close()
            return None

    def load_state_dicts_network(self, states_dict: Dict[str, Any]):
        try:
            self._task_queue.put(["load_state_dicts_network", states_dict])
        except ValueError:
            self.close()

    def state_dicts_optimizer(self) -> Dict[str, Any]:
        try:
            self._task_queue.put(["state_dicts_optimizer"])
            return self._model_queue.get()
        except ValueError:
            self.close()
            return None

    def load_state_dicts_optimizer(self, states_dict: Dict[str, Any]):
        try:
            self._task_queue.put(["load_state_dicts_optimizer", states_dict])
        except ValueError:
            self.close()

    def state_dicts_scheduler(self) -> Dict[str, Any]:
        try:
            self._task_queue.put(["state_dicts_scheduler"])
            return self._model_queue.get()
        except ValueError:
            self.close()
            return None

    def load_state_dicts_scheduler(self, states_dict: Dict[str, Any]):
        try:
            self._task_queue.put(["load_state_dicts_scheduler", states_dict])
        except ValueError:
            self.close()

    def set_probe_states(self, probe_states):
        """Set probe states for diagnostics evaluation."""
        try:
            self._task_queue.put(["set_probe_states", probe_states])
        except ValueError:
            self.close()

    def save_policy_snapshot(self, path: str):
        """Save policy-only snapshot to disk."""
        try:
            self._task_queue.put(["save_policy_snapshot", path])
        except ValueError:
            self.close()

    def flush_diagnostics(self):
        """Flush diagnostics logs to disk."""
        try:
            self._task_queue.put(["flush_diagnostics"])
        except ValueError:
            self.close()

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._shutdown.set()
            self._task_queue.put(["shutdown"])
            self._process.join(5)
            exitcode = self._process.exitcode
            if exitcode is None:
                if not self._is_shutdown.is_set():
                    self._clear_queues()
                    self._close_queues()
                self._process.kill()
                self._process.join()
            self._process.close()
            self._process = None

    def is_alive(self) -> bool:
        if self._process is None:
            return False
        return self._process.is_alive()

    def _clear_queues(self):
        for queue_ in [self._result_queue, self._model_queue, self._task_queue]:
            while True:
                try:
                    queue_.get_nowait()
                except (queue.Empty, ValueError):
                    break

    def _close_queues(self):
        for queue_ in [self._result_queue, self._model_queue, self._task_queue]:
            queue_.close()

    @property
    def step_counter(self) -> StepCounterShared:
        return self._step_counter

    @step_counter.setter
    def step_counter(self, new_counter: StepCounter) -> None:
        self._step_counter.heatup = new_counter.heatup
        self._step_counter.exploration = new_counter.exploration
        self._step_counter.evaluation = new_counter.evaluation
        self._step_counter.update = new_counter.update

    @property
    def episode_counter(self) -> EpisodeCounterShared:
        return self._episode_counter

    @episode_counter.setter
    def episode_counter(self, new_counter: EpisodeCounter) -> None:
        self._episode_counter.heatup = new_counter.heatup
        self._episode_counter.exploration = new_counter.exploration
        self._episode_counter.evaluation = new_counter.evaluation
