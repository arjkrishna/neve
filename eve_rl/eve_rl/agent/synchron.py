from copy import deepcopy
from importlib import import_module
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Tuple
from math import inf
import logging
import os
import threading
import torch
import queue
import multiprocessing as mp

from ..util import DummyEnv
from .agent import (
    Agent,
    Episode,
    StepCounterShared,
    EpisodeCounterShared,
)
from .single import Algo, ReplayBuffer, gym
from .singelagentprocess import SingleAgentProcess
from ..util import ConfigHandler


class SynchronEvalOnly(Agent):
    def __init__(
        self,
        algo: Algo,
        env_eval: gym.Env,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
        timeout_worker_after_reaching_limit: float = 90,
    ) -> None:
        self.algo = algo
        self.algo.to(torch.device("cpu"))
        self.env_eval = env_eval
        self.worker_device = worker_device
        self.normalize_actions = normalize_actions
        self.timeout_worker_after_reaching_limit = timeout_worker_after_reaching_limit

        self.logger = logging.getLogger(self.__module__)
        self.n_worker = n_worker
        self.worker: List[SingleAgentProcess] = []
        self.step_counter = StepCounterShared()
        self.episode_counter = EpisodeCounterShared()

        for i in range(n_worker):
            self.worker.append(self._create_worker_agent(i))

        self._eval_seeds = None
        self._eval_options = None

        self.logger.info("Synchron Agent initialized")

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
        steps_start = self.step_counter.evaluation
        episodes_start = self.episode_counter.evaluation
        self._log_eval(steps, step_limit, episodes, episode_limit, seeds, options)
        step_limit, episode_limit = self._log_and_convert_limits(
            "evaluation", steps, step_limit, episodes, episode_limit, seeds, options
        )
        if seeds is not None:
            self._eval_seeds = self._split(seeds, self.n_worker)
        if options is not None:
            self._eval_options = self._split(options, self.n_worker)

        for agent in self.worker:
            i = agent.agent_id
            agent_seeds = self._eval_seeds[i] if seeds is not None else None
            agent_options = self._eval_options[i] if options is not None else None
            agent.evaluate(
                step_limit=step_limit,
                episode_limit=episode_limit,
                seeds=agent_seeds,
                options=agent_options,
            )

        self._eval_seeds = None
        self._eval_options = None
        result = self._get_worker_results(step_limit, episode_limit, "evaluation")

        n_steps = self.step_counter.evaluation - steps_start
        n_episodes = self.episode_counter.evaluation - episodes_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("evaluation", n_steps, t_duration, n_episodes)
        return result

    def close(self):
        for agent in self.worker:
            agent.close()

    def _worker_load_state_dicts_network(self, state_dicts: Dict[str, Any]):
        for agent in self.worker:
            agent.load_state_dicts_network(state_dicts)
            sleep(0.1)

    def _get_worker_results(self, step_limit: int, episode_limit: int, task: str):
        episode_results = []
        results_pending = self.worker.copy()
        t_limit_result = inf
        # Plan v11 — HARD wall-clock cap on the entire result-collection
        # loop. The original code waited on `t_limit_result` (which only
        # starts ticking after step/episode targets are reached) and
        # could hang forever if a worker stopped reporting before the
        # target was hit. We've hit this exact failure on every canonical
        # 54-seed eval (Task C, 1B-IQL/AWAC step-0, resume eval @ u20k).
        # Configurable via env var EVE_RL_EVAL_HARD_TIMEOUT_MIN (default
        # 70 min). Returns partial results rather than blocking training.
        import os as _os
        _hard_timeout_min = float(
            _os.environ.get("EVE_RL_EVAL_HARD_TIMEOUT_MIN", "70")
        )
        t_hard_limit = perf_counter() + _hard_timeout_min * 60.0
        while results_pending and perf_counter() < t_limit_result:
            if perf_counter() > t_hard_limit:
                self.logger.warning(
                    f"Eval HARD TIMEOUT ({_hard_timeout_min:.0f} min) reached with "
                    f"{len(results_pending)} workers still pending; "
                    f"returning {len(episode_results)} partial episodes."
                )
                break
            results_pending, t_limit_result, episode_results = self._worker_result_loop(
                step_limit,
                episode_limit,
                task,
                episode_results,
                results_pending,
                t_limit_result,
            )

        for agent in results_pending:
            log_warn = f"Restaring Agent {agent.name} because of Timeout"
            self.logger.warning(log_warn)
            self._restart_worker_agent(agent)

        return episode_results

    def _worker_result_loop(
        self,
        step_limit,
        episode_limit,
        task,
        episode_results,
        results_pending,
        t_limit_result,
    ):
        remove = []
        add = []
        for agent in results_pending:
            result = agent.get_result(timeout=0.1)
            if isinstance(result, queue.Empty):
                if not agent.is_alive():
                    log_warn = (
                        f"Restaring Agent {agent.name} because it is not alive anymore"
                    )
                    self.logger.warning(log_warn)
                    new = self._restart_worker_agent(
                        agent, task, step_limit, episode_limit
                    )
                    remove.append(agent)
                    add.append(new)

            elif isinstance(result, Exception):
                log_warn = f"Restaring Agent {agent.name} because of Exception {result}"
                self.logger.warning(log_warn)
                new = self._restart_worker_agent(agent, task, step_limit, episode_limit)
                remove.append(agent)
                add.append(new)

            else:
                episode_results += result
                remove.append(agent)

        for agent in remove:
            results_pending.remove(agent)
        for agent in add:
            results_pending.append(agent)

        if t_limit_result == inf:
            steps = getattr(self.step_counter, task)
            episodes = getattr(self.episode_counter, task)
            if steps >= step_limit or episodes >= episode_limit:
                log_debug = (
                    task
                    + f": {steps=} >= {step_limit=} or {episodes=} >= {episode_limit=}. Setting time limit for workers to finish to {self.timeout_worker_after_reaching_limit}s"
                )
                self.logger.debug(log_debug)
                t_limit_result = (
                    perf_counter() + self.timeout_worker_after_reaching_limit
                )

        return results_pending, t_limit_result, episode_results

    def _restart_worker_agent(
        self,
        agent: SingleAgentProcess,
        task: str = None,
        step_limit: int = None,
        episode_limit: int = None,
    ):
        log_debug = (
            f"Restarting Agent with {task=}, {step_limit=} and {episode_limit=}."
        )
        self.logger.debug(log_debug)
        agent.close()
        new_agent = self._create_worker_agent(agent.agent_id)
        new_agent.load_state_dicts_network(self.algo.state_dicts_network())
        self.worker[agent.agent_id] = new_agent
        i = agent.agent_id
        seeds = self._eval_seeds[i] if self._eval_seeds is not None else None
        options = self._eval_options[i] if self._eval_options is not None else None
        new_agent.evaluate(
            step_limit=step_limit,
            episode_limit=episode_limit,
            seeds=seeds,
            options=options,
        )
        return new_agent

    def _create_worker_agent(self, i):
        return SingleAgentProcess(
            i,
            self.algo.to_play_only(),
            DummyEnv(),
            deepcopy(self.env_eval),
            None,
            self.worker_device,
            1,
            self.normalize_actions,
            name="worker_" + str(i),
            parent_agent=self,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
            nice_level=10,
        )

    def load_checkpoint(self, file_path: str) -> None:
        super().load_checkpoint(file_path)
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())

    @staticmethod
    def _split(to_split: List, n_parts: int):
        floor, mod = divmod(len(to_split), n_parts)
        split_lists = [
            to_split[i * floor + min(i, mod) : (i + 1) * floor + min(i + 1, mod)]
            for i in range(n_parts)
        ]
        return split_lists

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
        timeout_worker_after_reaching_limit: float = 90,
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
            n_worker,
            worker_device,
            normalize_actions,
            timeout_worker_after_reaching_limit,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent


class Synchron(SynchronEvalOnly, Agent):
    def __init__(  # pylint: disable=super-init-not-called
        self,
        algo: Algo,
        env_train: gym.Env,
        env_eval: gym.Env,
        replay_buffer: ReplayBuffer,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        trainer_device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        timeout_worker_after_reaching_limit: float = 90,
        diagnostics_config: Optional[Dict] = None,
        env_train_factory=None,
    ) -> None:
        self.algo = algo
        self.algo.to(torch.device("cpu"))
        self.env_train = env_train
        self.env_eval = env_eval
        # Per-worker training-env factory (Gen-4 procedural anatomy). When
        # set, worker i's train env is env_train_factory(i) instead of a
        # deepcopy of the single master env_train — so each worker can run a
        # DIFFERENT vessel (e.g. a distinctly-seeded DualDeviceNavProcedural
        # that re-randomizes every N episodes). env_train is still used for
        # network sizing / config; env_eval stays shared (a fixed held-out
        # anatomy is the honest generalization metric). None = legacy
        # (deepcopy the master) behavior.
        #
        # Held under a PRIVATE name: a factory is a function/closure, and the
        # eve_rl ConfigHandler (runner.save_config) serializes every __init__
        # param via getattr(self, name) — a raw function raises
        # NotImplementedError. Expose the param name as None (the eve
        # placeholder convention) so config save records a placeholder; the
        # real factory lives in self._env_train_factory.
        self._env_train_factory = env_train_factory
        self.env_train_factory = None
        self.worker_device = worker_device
        self.trainer_device = trainer_device
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions
        self.timeout_worker_after_reaching_limit = timeout_worker_after_reaching_limit
        self.diagnostics_config = diagnostics_config

        self.logger = logging.getLogger(self.__module__)
        self.n_worker = n_worker
        self.worker: List[SingleAgentProcess] = []
        self.replay_buffer = replay_buffer
        self.step_counter = StepCounterShared()
        self.episode_counter = EpisodeCounterShared()

        # Sync update_step to replay buffer's shared Value for episode stamping.
        # This writes the integer value to a shared mp.Value that the subprocess
        # inherited at spawn time. Called here and periodically during updates.
        self.replay_buffer.set_step_counter(self.step_counter)

        for i in range(n_worker):
            self.worker.append(self._create_worker_agent(i))

        self.trainer = self._create_trainer_agent()

        self.update_error = False

        self._eval_seeds = None
        self._eval_options = None

        # RL_IMPROV_16 (deadlock hardening) — trainer-result deadline for the
        # explore_and_update collection loop. v1 AND v2 both deadlocked at the
        # SAME seam (right after the 3rd eval): the loop's worker branch has a
        # timeout+restart but the TRAINER branch did not, so when the trainer's
        # update-result was lost in the mp.Queue the loop spun on
        # `get(timeout=0.5)` forever (main thread poll_schedule_timeout, all
        # subprocesses idle). If the trainer result does not arrive within this
        # many seconds of a cycle starting, the trainer is force-restarted
        # (same recovery as the exception path) and the cycle proceeds.
        self._trainer_result_timeout_s = float(
            os.environ.get("EVE_RL_TRAINER_RESULT_TIMEOUT_S", "1800")
        )

        # RL_IMPROV_16 — catch-all progress WATCHDOG. A daemon thread that
        # hard-exits the process if NONE of the shared step counters advance
        # for `EVE_RL_WATCHDOG_STALL_S` seconds — converting any IPC deadlock
        # (not just the one above) into a clean, visible crash a restart policy
        # or operator can act on, instead of a silent all-night hang. Set
        # ABOVE the trainer-result deadline so the graceful trainer-restart
        # always fires first; the watchdog only trips if that fails too. 0=off.
        self._watchdog_stall_s = float(
            os.environ.get("EVE_RL_WATCHDOG_STALL_S", "2400")
        )
        if self._watchdog_stall_s > 0:
            self._watchdog_thread = threading.Thread(
                target=self._progress_watchdog, name="progress_watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

        self.logger.info("Synchron Agent initialized")

    def _progress_watchdog(self) -> None:
        """RL_IMPROV_16 — hard-exit on a total-progress stall (see __init__)."""
        sc = self.step_counter

        def _snapshot():
            # Sum of every phase counter; a genuine hang freezes ALL of them,
            # while eval advances `evaluation`, the first explore cycle
            # advances `exploration`, etc. — so no false trigger per phase.
            return (
                int(getattr(sc, "heatup", 0))
                + int(getattr(sc, "exploration", 0))
                + int(getattr(sc, "update", 0))
                + int(getattr(sc, "evaluation", 0))
            )

        last_val = _snapshot()
        last_move = perf_counter()
        poll = min(60.0, max(5.0, self._watchdog_stall_s / 10.0))
        while True:
            sleep(poll)
            try:
                cur = _snapshot()
            except Exception:
                continue  # counter transiently unreadable — do not act
            now = perf_counter()
            if cur != last_val:
                last_val = cur
                last_move = now
                continue
            stalled = now - last_move
            if stalled >= self._watchdog_stall_s:
                # Best-effort thread-state dump for post-mortem, then die.
                try:
                    tids = os.listdir(f"/proc/{os.getpid()}/task")
                    wch = {}
                    for t in tids:
                        try:
                            with open(f"/proc/{os.getpid()}/task/{t}/wchan") as f:
                                w = f.read().strip()
                            wch[w] = wch.get(w, 0) + 1
                        except Exception:
                            pass
                    self.logger.error(
                        "WATCHDOG: no step-counter progress for %.0fs "
                        "(>= %.0fs) — IPC deadlock. main-proc thread wchans: "
                        "%s. Hard-exiting (os._exit 42).",
                        stalled, self._watchdog_stall_s, wch,
                    )
                except Exception:
                    self.logger.error(
                        "WATCHDOG: stall %.0fs — hard-exiting.", stalled
                    )
                for h in list(self.logger.handlers):
                    try:
                        h.flush()
                    except Exception:
                        pass
                os._exit(42)

    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
        episode_schedule: Optional[List] = None,
        heatup_save_every: int = 0,
        heatup_save_path: Optional[str] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        steps_start = self.step_counter.heatup
        episodes_start = self.episode_counter.heatup
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

        # Split episode_schedule across workers (round-robin)
        worker_schedules = self._split_schedule(episode_schedule)

        for i, agent in enumerate(self.worker):
            agent.heatup(
                step_limit=step_limit,
                episode_limit=episode_limit,
                custom_action_low=custom_action_low,
                custom_action_high=custom_action_high,
                episode_schedule=worker_schedules[i],
                heatup_save_every=heatup_save_every,
                heatup_save_path=heatup_save_path,
            )
        result = self._get_worker_results(step_limit, episode_limit, "heatup")
        n_steps = self.step_counter.heatup - steps_start
        n_episodes = self.episode_counter.heatup - episodes_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("heatup", n_steps, t_duration, n_episodes)
        return result

    def heuristic_seed(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        heuristic_factory=None,
        episode_schedule: Optional[List] = None,
        push_to_buffer: bool = True,
    ) -> List[Episode]:
        """Seed replay buffer using parallel workers with heuristic controller.

        Uses the same counters as heatup since both are pre-training phases.

        Args:
            steps: Target number of total steps across all workers
            episodes: Target number of total episodes across all workers
            step_limit: Maximum steps per episode
            episode_limit: Maximum episodes per worker
            heuristic_factory: Factory with .create(env) method returning action function
            episode_schedule: List of (seed, options) per episode, generated by
                the training script. Split across workers round-robin.
            push_to_buffer: If False, episodes are collected but not pushed to
                replay buffer. Caller is responsible for filtering and pushing.

        Returns:
            List of collected episodes from all workers
        """
        t_start = perf_counter()
        steps_start = self.step_counter.heatup
        episodes_start = self.episode_counter.heatup

        self.logger.info(
            f"Parallel heuristic seeding: {steps=}, {episodes=}, "
            f"{step_limit=}, {episode_limit=}, n_workers={self.n_worker}"
        )

        # Use "heatup" counters since heuristic seeding is a pre-training phase
        step_limit, episode_limit = self._log_and_convert_limits(
            "heatup", steps, step_limit, episodes, episode_limit
        )

        # Split episode_schedule across workers (round-robin)
        worker_schedules = self._split_schedule(episode_schedule)

        # Dispatch heuristic_seed task to all workers
        for i, agent in enumerate(self.worker):
            agent.heuristic_seed(
                step_limit=step_limit,
                episode_limit=episode_limit,
                heuristic_factory=heuristic_factory,
                episode_schedule=worker_schedules[i],
                push_to_buffer=push_to_buffer,
            )

        # Heuristic episodes can take 500-800s (1000 steps × 0.5-0.8s/step).
        # The default 90s timeout kills workers mid-episode, losing their results.
        # Temporarily increase to 1200s so all workers finish their last episode.
        saved_timeout = self.timeout_worker_after_reaching_limit
        self.timeout_worker_after_reaching_limit = max(saved_timeout, 1200)

        # Wait for workers to complete (uses heatup counters)
        result = self._get_worker_results(step_limit, episode_limit, "heatup")

        self.timeout_worker_after_reaching_limit = saved_timeout

        n_steps = self.step_counter.heatup - steps_start
        n_episodes = self.episode_counter.heatup - episodes_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("heatup", n_steps, t_duration, n_episodes)
        return result

    def _split_schedule(self, schedule):
        """Split an episode schedule across workers round-robin.

        Returns a list of per-worker schedules. If schedule is None,
        returns [None] * n_worker (backward compat — no schedule).
        """
        if schedule is None:
            return [None] * self.n_worker
        per_worker = [[] for _ in range(self.n_worker)]
        for i, entry in enumerate(schedule):
            per_worker[i % self.n_worker].append(entry)
        return per_worker

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        steps_start = self.step_counter.exploration
        episodes_start = self.episode_counter.exploration
        self._log_exploration(steps, step_limit, episodes, episode_limit)

        step_limit, episode_limit = self._log_and_convert_limits(
            "exploration", steps, step_limit, episodes, episode_limit
        )

        for agent in self.worker:
            agent.explore(step_limit=step_limit, episode_limit=episode_limit)
        result = self._get_worker_results(step_limit, episode_limit, "exploration")

        n_episodes = self.episode_counter.exploration - episodes_start
        n_steps = self.step_counter.exploration - steps_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("exploration", n_steps, t_duration, n_episodes)
        return result

    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> List[float]:
        t_start = perf_counter()
        steps_start = self.step_counter.update
        self._log_update(steps, step_limit)
        step_limit, _ = self._log_and_convert_limits("update", steps, step_limit)

        self.trainer.update(steps=steps, step_limit=step_limit)
        result = self._get_trainer_results()
        self._update_algo_state_dicts()
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())

        # Sync current update_step to replay buffer for accurate episode stamping
        self.replay_buffer.set_step_counter(self.step_counter)

        n_steps = self.step_counter.update - steps_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("update", n_steps, t_duration)
        return result

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
        t_start = perf_counter()
        update_steps_start = self.step_counter.update
        explore_steps_start = self.step_counter.exploration
        explore_episodes_start = self.episode_counter.exploration
        self._log_explore_and_update(
            explore_steps,
            explore_episodes,
            explore_step_limit,
            explore_episode_limit,
            update_steps,
            update_step_limit,
        )

        update_step_limit, _ = self._log_and_convert_limits(
            "update", update_steps, update_step_limit
        )
        self.trainer.update(step_limit=update_step_limit)

        explore_step_limit, explore_episode_limit = self._log_and_convert_limits(
            "exploration",
            explore_steps,
            explore_step_limit,
            explore_episodes,
            explore_episode_limit,
        )
        for agent in self.worker:
            agent.explore(
                step_limit=explore_step_limit, episode_limit=explore_episode_limit
            )

        got_worker_results = False
        got_trainer_results = False
        explore_results = []
        update_result = None
        results_pending = self.worker.copy()
        t_limit_result = inf
        while True:
            if not got_worker_results:
                (
                    results_pending,
                    t_limit_result,
                    explore_results,
                ) = self._worker_result_loop(
                    explore_step_limit,
                    explore_episode_limit,
                    "exploration",
                    explore_results,
                    results_pending,
                    t_limit_result,
                )
                if not results_pending or perf_counter() > t_limit_result:
                    for agent in results_pending:
                        log_warn = f"Restaring Agent {agent.name} because of Timeout"
                        self.logger.warning(log_warn)
                        self._restart_worker_agent(agent)
                    results_pending = []
                    got_worker_results = True
                    n_steps_explore = (
                        self.step_counter.exploration - explore_steps_start
                    )
                    n_episodes_explore = (
                        self.episode_counter.exploration - explore_episodes_start
                    )
                    t_duration_explore = perf_counter() - t_start

            if not got_trainer_results:
                update_result = self._get_trainer_results(0.5)
                if update_result is not None:
                    got_trainer_results = True
                    n_steps_update = self.step_counter.update - update_steps_start
                    t_duration_update = perf_counter() - t_start
                # RL_IMPROV_16 deadlock fix — bound the trainer wait exactly
                # like the worker branch above. Without this, a lost/never-sent
                # trainer update-result spins this loop forever (the v1+v2
                # post-eval3 deadlock). On deadline, force-restart the trainer
                # (same recovery as an exception) and let the cycle finish; the
                # skipped updates are re-earned by the next cycle's budget.
                # RL_IMPROV_18 R2 — the 2026-07-16 P2a run proved the pure
                # wall-clock deadline is WRONG: explore cycles legitimately
                # run ~40 min > 1800s, so a HEALTHY trainer got restarted
                # into fresh init every cycle (32 wipes; combined with the
                # R1 sync bug ⇒ 24h of init-policy evals). Gate the restart
                # on STALL EVIDENCE: the shared update counter must have
                # stopped advancing for the full timeout, not merely the
                # result being slow to arrive.
                elif (
                    perf_counter() - t_start > self._trainer_result_timeout_s
                ):
                    _u_now = self.step_counter.update
                    if _u_now != getattr(self, "_last_update_progress", (None, 0.0))[0]:
                        self._last_update_progress = (_u_now, perf_counter())
                    elif (
                        perf_counter() - self._last_update_progress[1]
                        > self._trainer_result_timeout_s
                    ):
                        self._restart_trainer(
                            f"update counter STALLED at {_u_now} for "
                            f"{self._trainer_result_timeout_s:.0f}s (deadlock guard)"
                        )
                        got_trainer_results = True
                        n_steps_update = (
                            self.step_counter.update - update_steps_start
                        )
                        t_duration_update = perf_counter() - t_start

            if got_worker_results and got_trainer_results:
                break

        self._log_task_completion_explore_and_update(
            n_steps_update,
            t_duration_update,
            n_steps_explore,
            n_episodes_explore,
            t_duration_explore,
        )
        self._update_algo_state_dicts()
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())

        # Re-stamp episodes with accurate update_step from replay buffer arrival time.
        # The replay buffer subprocess reads step_counter.update when each episode
        # arrives, which is more accurate than the worker's stale read.
        arrival_map = self.replay_buffer.drain_episode_arrivals()
        if arrival_map:
            for episode in explore_results:
                explore_step = getattr(episode, 'explore_step_at_completion', None)
                if explore_step is not None and explore_step in arrival_map:
                    episode.update_step_at_completion = arrival_map[explore_step]

        return explore_results, update_result

    def close(self):
        for agent in self.worker:
            agent.close()
        self.trainer.close()
        self.replay_buffer.close()

        del self.worker
        del self.trainer
        del self.replay_buffer

    def _update_algo_state_dicts(self):
        # RL_IMPROV_18 R1 — the trainer is the AUTHORITY on network weights.
        # The old code passed algo's dict as `destination` and DISCARDED the
        # return: the trainer subprocess fills a PICKLED COPY, so in-place
        # mutation can never reach this process — algo's nets stayed at
        # init forever (workers/evals/checkpoints all sync FROM algo ⇒ the
        # 24h init-policy run of 2026-07-16). Pull and LOAD the return.
        state_dicts = self.trainer.state_dicts_network()
        if state_dicts is not None:
            self.algo.load_state_dicts_network(state_dicts)
            # R3 invariant alarm — weights pulled from the trainer must
            # CHANGE between syncs once training has started. A frozen
            # fingerprint across 3+ syncs = the 2026-07-16 failure class
            # (sync broken or trainer re-initing) — scream, don't crash.
            try:
                import hashlib
                import pickle as _pkl

                fp = hashlib.md5(_pkl.dumps(
                    {k: v for k, v in sorted(state_dicts.items())}
                )).hexdigest()[:12]
                self._net_sync_count = getattr(self, "_net_sync_count", 0) + 1
                if fp == getattr(self, "_last_net_fp", None):
                    self._net_fp_repeats = getattr(self, "_net_fp_repeats", 0) + 1
                    if (
                        self._net_fp_repeats >= 3
                        and self.step_counter.update > 1000
                    ):
                        self.logger.error(
                            "NET_SYNC_ALARM: trainer network fingerprint %s "
                            "UNCHANGED for %d consecutive syncs at update=%d "
                            "— weight sync or trainer state is broken!",
                            fp, self._net_fp_repeats, self.step_counter.update,
                        )
                else:
                    self._net_fp_repeats = 0
                self._last_net_fp = fp
                self.logger.info(
                    "NET_SYNC ok #%d fp=%s update=%d",
                    self._net_sync_count, fp, self.step_counter.update,
                )
            except Exception:
                pass

        state_dicts = self.trainer.state_dicts_optimizer()
        self.algo.load_state_dicts_optimizer(state_dicts)

        state_dicts = self.trainer.state_dicts_scheduler()
        self.algo.load_state_dicts_scheduler(state_dicts)

    def _restart_worker_agent(
        self,
        agent: SingleAgentProcess,
        task: str = None,
        step_limit: int = None,
        episode_limit: int = None,
    ):
        log_debug = (
            f"Restarting Agent with {task=}, {step_limit=} and {episode_limit=}."
        )
        self.logger.debug(log_debug)
        agent.close()
        new_agent = self._create_worker_agent(agent.agent_id)
        new_agent.load_state_dicts_network(self.algo.state_dicts_network())
        self.worker[agent.agent_id] = new_agent
        if task == "heatup":
            new_agent.heatup(step_limit=step_limit, episode_limit=episode_limit)
        elif task == "exploration":
            new_agent.explore(step_limit=step_limit, episode_limit=episode_limit)
        elif task == "evaluation":
            i = agent.agent_id
            seeds = self._eval_seeds[i] if self._eval_seeds is not None else None
            options = self._eval_options[i] if self._eval_options is not None else None
            new_agent.evaluate(
                step_limit=step_limit,
                episode_limit=episode_limit,
                seeds=seeds,
                options=options,
            )
        return new_agent

    def _restart_trainer(self, reason: str) -> None:
        """RL_IMPROV_16 — close + recreate the trainer subprocess and reload
        its weights/optimizer/scheduler from the main-process algo. Used by
        both the exception path and the explore_and_update trainer-result
        deadline (a silently-unresponsive trainer looks identical to a crashed
        one from the main's side, and this is the recovery both need)."""
        self.logger.warning("Restarting Trainer because of %s", reason)
        try:
            self.trainer.close()
        except Exception:
            pass
        self.trainer = self._create_trainer_agent()
        self.trainer.load_state_dicts_network(self.algo.state_dicts_network())
        self.trainer.load_state_dicts_optimizer(self.algo.state_dicts_optimizer())
        self.trainer.load_state_dicts_scheduler(self.algo.state_dicts_scheduler())
        self.update_error = True

    def _get_trainer_results(self, timeout: Optional[float] = None):
        result = self.trainer.get_result(timeout=timeout)
        if isinstance(result, queue.Empty):
            return None
        if isinstance(result, Exception):
            self._restart_trainer(f"Exception {result}")
            return []
        return result

    def _create_worker_agent(self, i):
        # Plan v12 Stage 2 — ONE shared heatup-stop Event for ALL workers, so
        # the main-process SIGTERM/docker-stop handler can set it once and every
        # worker exits its heatup loop + runs its final-batch flush.
        if getattr(self, "_heatup_stop", None) is None:
            self._heatup_stop = mp.Event()
        if self._env_train_factory is not None:
            worker_env_train = self._env_train_factory(i)
        else:
            worker_env_train = deepcopy(self.env_train)
        return SingleAgentProcess(
            i,
            self.algo.to_play_only(),
            worker_env_train,
            deepcopy(self.env_eval),
            self.replay_buffer.copy(),
            self.worker_device,
            self.consecutive_action_steps,
            self.normalize_actions,
            name="worker_" + str(i),
            parent_agent=self,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
            nice_level=10,
            heatup_stop=self._heatup_stop,
        )

    def _create_trainer_agent(self):
        return SingleAgentProcess(
            0,
            deepcopy(self.algo),
            DummyEnv(),
            DummyEnv(),
            self.replay_buffer.copy(),
            self.trainer_device,
            0,
            self.normalize_actions,
            name="trainer_synchron",
            parent_agent=self,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
            nice_level=0,
            diagnostics_config=self.diagnostics_config,
        )

    def set_probe_states(self, probe_states):
        """Set probe states for diagnostics evaluation in trainer subprocess."""
        if self.trainer is not None:
            self.trainer.set_probe_states(probe_states)

    def save_policy_snapshot(self, path: str):
        """Save policy-only snapshot from trainer subprocess."""
        if self.trainer is not None:
            self.trainer.save_policy_snapshot(path)

    def flush_diagnostics(self):
        """Flush diagnostics logs in trainer subprocess."""
        if self.trainer is not None:
            self.trainer.flush_diagnostics()

    def load_checkpoint(self, file_path: str) -> None:
        super().load_checkpoint(file_path)
        self.trainer.load_state_dicts_network(self.algo.state_dicts_network())
        self.trainer.load_state_dicts_optimizer(self.algo.state_dicts_optimizer())
        self.trainer.load_state_dicts_scheduler(self.algo.state_dicts_scheduler())

    @classmethod
    def from_checkpoint(  # pylint: disable=arguments-renamed
        cls,
        checkpoint_path: str,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        trainer_device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        timeout_worker_after_reaching_limit: float = 90,
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
            n_worker,
            worker_device,
            trainer_device,
            consecutive_action_steps,
            normalize_actions,
            timeout_worker_after_reaching_limit,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent
