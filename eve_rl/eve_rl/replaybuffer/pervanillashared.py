"""Shared (multiprocess) Prioritized Experience Replay buffer (Plan v7).

`PERVanillaStepShared` mirrors `VanillaStepShared`: the actual buffer
(`PERVanillaStep`) lives in a subprocess; workers push episodes and the
trainer pulls sampled batches through `mp` queues. PER adds a **third
queue** — `_priority_update_queue` — over which the trainer sends
`(indices, td_errors)` back so the subprocess can refresh the sampled
transitions' priorities.

`vanillashared.py` is intentionally left untouched (it backs the
in-use episode / uniform-step paths); the subprocess `loop` is
re-implemented here with one extra branch that drains the priority queue.
"""
from time import sleep, time
import torch
import torch.multiprocessing as mp
import numpy as np

from .replaybuffer import ReplayBuffer, Episode, Batch
from .vanillashared import VanillaSharedBase, VanillaStepShared, _setup_subprocess_logging
from .pervanillastep import PERVanillaStep


class PERVanillaSharedBase(VanillaSharedBase):
    """Handle/copy object held by workers and the trainer. Adds the
    priority-update queue and `update_priorities()` on top of the base
    shared-buffer handle."""

    def __init__(self, *args, priority_update_queue: mp.SimpleQueue = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._priority_update_queue = priority_update_queue

    def update_priorities(self, indices, td_errors) -> None:
        """Send fresh per-transition TD errors to the buffer subprocess so
        it can recompute priorities. ``indices`` are the leaf indices from
        the sampled `Batch`; ``td_errors`` the per-sample |TD|. Both are
        converted to plain numpy (CPU) for the queue."""
        if self._priority_update_queue is None or self._shutdown_event.is_set():
            return
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        else:
            indices = np.asarray(indices)
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        else:
            td_errors = np.asarray(td_errors)
        indices = np.asarray(indices).reshape(-1)
        td_errors = np.asarray(td_errors).reshape(-1)
        self._priority_update_queue.put((indices, td_errors))


class PERVanillaStepShared(VanillaStepShared):
    """Subprocess-owning shared PER buffer."""

    def __init__(
        self,
        capacity,
        batch_size,
        sample_device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: float = 2e7,
        epsilon: float = 1e-6,
        demo_priority_bonus: float = 0.0,
        priority_mode: str = "td",
        balanced_fraction: float = 0.0,
        # RL_IMPROV_16 E3 — stuck-lane knobs, forwarded verbatim to the
        # internal PERVanillaStep (see its __init__ docnote). Defaults OFF.
        stuck_fraction: float = 0.0,
        stuck_slack_index: int = -1,
        stuck_slack_thresh: float = 0.174,
        stuck_contact_index: int = -1,
        stuck_contact_thresh: float = 0.0026,
        # RL_IMPROV_17 (RLPD) — symmetric offline/online sampling fraction,
        # forwarded verbatim (see PERVanillaStep docnote). 0.0 = OFF.
        offline_fraction: float = 0.0,
    ):
        # NB: deliberately do NOT call VanillaStepShared.__init__ — it would
        # start the subprocess before the priority queue exists. Replicate
        # its body, inserting the extra queue first.
        VanillaSharedBase.__init__(
            self,
            mp.SimpleQueue(),  # push
            mp.SimpleQueue(),  # sample
            mp.SimpleQueue(),  # task
            mp.SimpleQueue(),  # result
            mp.Lock(),
            mp.Event(),
            batch_size,
            episode_arrival_queue=mp.Queue(),
        )
        self.capacity = capacity
        self.sample_device = sample_device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        # Plan v8 — stabilization-suite knobs forwarded to the internal
        # PERVanillaStep created in the subprocess.
        self.demo_priority_bonus = demo_priority_bonus
        self.priority_mode = priority_mode
        self.balanced_fraction = balanced_fraction
        # RL_IMPROV_16 E3 — stuck lane (stored BEFORE the process spawn so
        # the subprocess ctor sees them).
        self.stuck_fraction = stuck_fraction
        self.stuck_slack_index = stuck_slack_index
        self.stuck_slack_thresh = stuck_slack_thresh
        self.stuck_contact_index = stuck_contact_index
        self.stuck_contact_thresh = stuck_contact_thresh
        self.offline_fraction = offline_fraction
        # Third queue — trainer → subprocess priority updates. Created
        # before spawn so the subprocess inherits it.
        self._priority_update_queue = mp.SimpleQueue()
        self._process = mp.Process(
            target=self._run_subprocess,
            args=(self._shared_update_step,),
        )
        self._process.start()

    def save_buffer_incremental(self, dir_path: str) -> int:
        """RL_IMPROV_16 — persist only the transitions pushed since the
        last incremental save (chunk file) plus the full small drift-state
        (atomic replace). Returns the number of newly-saved transitions.
        Same request pattern as save_buffer_to_file."""
        with self._request_lock:
            self._task_queue.put(["save_buffer_incremental", dir_path])
            return self._result_queue.get()

    def load_buffer_incremental(self, dir_path: str) -> int:
        """RL_IMPROV_16 — rebuild the buffer from an incremental-save
        directory (chunks + state). Returns transitions restored."""
        with self._request_lock:
            self._task_queue.put(["load_buffer_incremental", dir_path])
            return self._result_queue.get()

    def _run_subprocess(self, shared_update_step):
        self._shared_update_step = shared_update_step
        internal_replay_buffer = PERVanillaStep(
            self.capacity, self._batch_size,
            alpha=self.alpha, beta_start=self.beta_start,
            beta_steps=self.beta_steps, epsilon=self.epsilon,
            demo_priority_bonus=self.demo_priority_bonus,
            priority_mode=self.priority_mode,
            balanced_fraction=self.balanced_fraction,
            stuck_fraction=self.stuck_fraction,
            stuck_slack_index=self.stuck_slack_index,
            stuck_slack_thresh=self.stuck_slack_thresh,
            stuck_contact_index=self.stuck_contact_index,
            stuck_contact_thresh=self.stuck_contact_thresh,
            offline_fraction=self.offline_fraction,
        )
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: ReplayBuffer):
        """Re-implementation of `VanillaStepShared.loop` with a fourth
        branch draining `_priority_update_queue`."""
        logger = _setup_subprocess_logging()
        logger.info(
            f"PER ReplayBuffer subprocess started. batch_size={self.batch_size}, "
            f"capacity={self.capacity}, alpha={self.alpha}, beta_start={self.beta_start}"
        )
        loop_count = 0
        episodes_received = 0
        batches_produced = 0
        priority_updates = 0
        last_status_time = time()

        try:
            while not self._shutdown_event.is_set():
                loop_count += 1
                current_len = len(internal_replay_buffer)
                sample_queue_empty = self._sample_queue.empty()
                can_sample = sample_queue_empty and current_len > self.batch_size

                if time() - last_status_time > 10:
                    logger.info(
                        f"STATUS: buffer_len={current_len}, can_sample={can_sample}, "
                        f"episodes_received={episodes_received}, "
                        f"batches_produced={batches_produced}, "
                        f"priority_updates={priority_updates}, loops={loop_count}, "
                        f"update_step={self._shared_update_step.value}"
                    )
                    last_status_time = time()

                # Priority updates first — keep priorities fresh and stop
                # the queue backing up. Drain all currently pending.
                if not self._priority_update_queue.empty():
                    while not self._priority_update_queue.empty():
                        indices, td_errors = self._priority_update_queue.get()
                        internal_replay_buffer.update_priorities(indices, td_errors)
                        priority_updates += 1
                elif can_sample:
                    batch = internal_replay_buffer.sample()
                    self._sample_queue.put(batch)
                    batches_produced += 1
                    if batches_produced <= 5 or batches_produced % 100 == 0:
                        logger.info(
                            f"SAMPLED: batch #{batches_produced}, buffer_len={current_len}"
                        )
                elif not self._task_queue.empty():
                    task = self._task_queue.get()
                    if task[0] == "length":
                        self._result_queue.put(current_len)
                    elif task[0] == "shutdown":
                        logger.info("SHUTDOWN: received shutdown signal")
                        break
                    elif task[0] == "save_buffer":
                        # Plan v10 — persist the full PER buffer (transitions +
                        # priorities + is_demo/is_clean). MUST put a result so
                        # the blocking save_buffer_to_file() caller doesn't
                        # deadlock (the base VanillaStepShared.loop has these
                        # branches; this PER override previously omitted them).
                        path = task[1]
                        try:
                            data = internal_replay_buffer.export_all()
                            np.savez(path, **data)
                            n_saved = int(data["n"]) if "n" in data else 0
                            self._result_queue.put(n_saved)
                            logger.info(
                                f"SAVE_BUFFER: saved {n_saved} transitions to {path}"
                            )
                        except Exception as e:
                            logger.error(f"SAVE_BUFFER failed: {e}", exc_info=True)
                            self._result_queue.put(-1)
                    elif task[0] == "load_buffer":
                        path = task[1]
                        try:
                            with np.load(path, allow_pickle=False) as data:
                                n_loaded = internal_replay_buffer.import_all(data)
                            self._result_queue.put(n_loaded)
                            logger.info(
                                f"LOAD_BUFFER: loaded {n_loaded} transitions from {path}"
                            )
                        except Exception as e:
                            logger.error(f"LOAD_BUFFER failed: {e}", exc_info=True)
                            self._result_queue.put(-1)
                    elif task[0] == "save_buffer_incremental":
                        # RL_IMPROV_16 — chunked save: only new transitions
                        # + the small drift-state file (vs the 1-2 GB full
                        # re-serialization that stalled this loop at eval3
                        # scale and triggered the v1/v2 post-eval deadlock).
                        dir_path = task[1]
                        try:
                            n_new = (
                                internal_replay_buffer.save_incremental_to_dir(
                                    dir_path
                                )
                            )
                            self._result_queue.put(n_new)
                            logger.info(
                                f"SAVE_BUFFER_INCR: {n_new} new transitions "
                                f"-> {dir_path} (total "
                                f"{len(internal_replay_buffer)})"
                            )
                        except Exception as e:
                            logger.error(
                                f"SAVE_BUFFER_INCR failed: {e}", exc_info=True
                            )
                            self._result_queue.put(-1)
                    elif task[0] == "load_buffer_incremental":
                        dir_path = task[1]
                        try:
                            n_loaded = (
                                internal_replay_buffer.load_incremental_from_dir(
                                    dir_path
                                )
                            )
                            self._result_queue.put(n_loaded)
                            logger.info(
                                f"LOAD_BUFFER_INCR: {n_loaded} transitions "
                                f"from {dir_path}"
                            )
                        except Exception as e:
                            logger.error(
                                f"LOAD_BUFFER_INCR failed: {e}", exc_info=True
                            )
                            self._result_queue.put(-1)
                elif not self._push_queue.empty():
                    item = self._push_queue.get()
                    if isinstance(item, tuple) and len(item) == 2:
                        batch, explore_step = item
                    else:
                        batch, explore_step = item, None
                    internal_replay_buffer.push(batch)
                    episodes_received += 1
                    current_update_step = self._shared_update_step.value
                    if explore_step is not None and self._episode_arrival_queue is not None:
                        self._episode_arrival_queue.put((explore_step, current_update_step))
                    if episodes_received <= 10 or episodes_received % 50 == 0:
                        logger.info(
                            f"PUSHED: episode #{episodes_received}, "
                            f"new_buffer_len={len(internal_replay_buffer)}"
                        )
                else:
                    sleep(0.0001)
        except Exception as e:
            logger.error(f"EXCEPTION in PER replay buffer subprocess: {e}", exc_info=True)
            raise
        finally:
            logger.info(
                f"EXITING: episodes_received={episodes_received}, "
                f"batches_produced={batches_produced}, priority_updates={priority_updates}"
            )
            internal_replay_buffer.close()

    def copy(self):
        return PERVanillaSharedBase(
            self._push_queue,
            self._sample_queue,
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
            episode_arrival_queue=self._episode_arrival_queue,
            priority_update_queue=self._priority_update_queue,
        )

    def update_priorities(self, indices, td_errors) -> None:
        """Same as the handle's — usable directly on the owner object too."""
        if self._shutdown_event.is_set():
            return
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        indices = np.asarray(indices).reshape(-1)
        td_errors = np.asarray(td_errors).reshape(-1)
        self._priority_update_queue.put((indices, td_errors))
