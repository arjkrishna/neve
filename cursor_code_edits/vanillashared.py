from time import sleep, time
from multiprocessing.synchronize import Lock as mp_lock
from multiprocessing.synchronize import Event as mp_event
import queue
import torch
import torch.multiprocessing as mp
import logging
import sys

from .replaybuffer import ReplayBuffer, Episode, Batch
from .vanillaepisode import VanillaEpisode
from .vanillastep import VanillaStep

# Configure logging for the replay buffer subprocess
def _setup_subprocess_logging():
    """Setup logging for the replay buffer subprocess."""
    logger = logging.getLogger("ReplayBufferSubprocess")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class VanillaSharedBase(ReplayBuffer):
    def __init__(
        self,
        push_queue: mp.SimpleQueue,
        sample_queue: mp.SimpleQueue,
        task_queue: mp.SimpleQueue,
        result_queue: mp.SimpleQueue,
        request_lock: mp_lock,
        shutdown_event: mp_event,
        batch_size: int,
        episode_arrival_queue: mp.Queue = None,
    ):
        self._push_queue = push_queue
        self._task_queue = task_queue
        self._sample_queue = sample_queue
        self._result_queue = result_queue
        self._request_lock = request_lock
        self._shutdown_event = shutdown_event
        self._batch_size = batch_size
        self._episode_arrival_queue = episode_arrival_queue
        # Shared update_step counter - inherited by subprocess, updated by parent
        # Created before subprocess spawn so it's inherited, not pickled
        self._shared_update_step = mp.Value('q', 0)  # 'q' = unsigned long long

    def __getstate__(self):
        """Exclude non-pickleable objects when spawning worker processes."""
        state = self.__dict__.copy()
        # _shared_update_step is a mp.Value that can't be pickled for spawn.
        # Workers don't need it - only the replay buffer subprocess uses it.
        state['_shared_update_step'] = None
        return state

    def __setstate__(self, state):
        """Restore state - workers get None for _shared_update_step which is fine."""
        self.__dict__.update(state)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if not self._shutdown_event.is_set():
            try:
                replay_data = episode.to_replay()
                explore_step = getattr(episode, 'explore_step_at_completion', None)
                self._push_queue.put((replay_data, explore_step))
            except Exception as e:
                logging.getLogger("ReplayBufferSubprocess").error(f"PUSH ERROR: {e}", exc_info=True)
                raise

    def sample(self) -> Batch:

        if self._shutdown_event.is_set():
            return Batch([], [], [], [], [])

        return self._sample_queue.get()

    def __len__(
        self,
    ):
        if self._shutdown_event.is_set():  #
            return 0

        with self._request_lock:
            self._task_queue.put(["length"])
            length = self._result_queue.get()
        return length

    def set_step_counter(self, step_counter):
        """Update the shared update_step value from the step_counter.
        
        This method is called periodically by the parent process to sync the
        current update step. The subprocess reads from _shared_update_step
        which is inherited shared memory (no pickling needed).
        
        Workers receive a copy with _shared_update_step=None; this is fine
        since workers don't need to update the step counter.
        """
        # Only update if we have the shared Value (not in worker copies)
        if self._shared_update_step is not None:
            self._shared_update_step.value = step_counter.update

    def drain_episode_arrivals(self):
        """Drain all episode arrival metadata. Returns dict {explore_step: update_step}."""
        arrival_map = {}
        if self._episode_arrival_queue is None:
            return arrival_map
        while True:
            try:
                explore_step, update_step = self._episode_arrival_queue.get_nowait()
                arrival_map[explore_step] = update_step
            except queue.Empty:
                break
        return arrival_map

    def save_buffer_to_file(self, path):
        """Have subprocess save buffer directly to file. Returns episode count."""
        with self._request_lock:
            self._task_queue.put(["save_buffer", path])
            return self._result_queue.get()

    def load_buffer_from_file(self, path):
        """Have subprocess load buffer directly from file. Returns episode count."""
        with self._request_lock:
            self._task_queue.put(["load_buffer", path])
            return self._result_queue.get()

    def copy(self):
        return self

    def close(self) -> None:
        ...


class VanillaStepShared(VanillaSharedBase):
    def __init__(self, capacity, batch_size, sample_device: torch.device):
        super().__init__(
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.Lock(),
            mp.Event(),
            batch_size,
            episode_arrival_queue=mp.Queue(),
        )
        self.capacity = capacity
        self.sample_device = sample_device
        # Pass shared_update_step explicitly so it's inherited by subprocess
        # (not pickled as part of self which would get None from __getstate__)
        self._process = mp.Process(
            target=self._run_subprocess,
            args=(self._shared_update_step,)
        )
        self._process.start()

    def _run_subprocess(self, shared_update_step):
        """Entry point for subprocess - receives shared_update_step as argument."""
        # Store the shared Value that was passed (inherited, not pickled)
        self._shared_update_step = shared_update_step
        internal_replay_buffer = VanillaStep(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: ReplayBuffer):
        logger = _setup_subprocess_logging()
        logger.info(f"ReplayBuffer subprocess started. batch_size={self.batch_size}, capacity={self.capacity}")

        loop_count = 0
        episodes_received = 0
        batches_produced = 0
        last_status_time = time()
        # Read update_step from shared Value (inherited from parent process)
        # Parent updates self._shared_update_step.value via set_step_counter()

        try:
            while not self._shutdown_event.is_set():
                loop_count += 1
                current_len = len(internal_replay_buffer)
                sample_queue_empty = self._sample_queue.empty()
                can_sample = sample_queue_empty and current_len > self.batch_size

                # Log status every 10 seconds
                if time() - last_status_time > 10:
                    current_update_step = self._shared_update_step.value
                    logger.info(
                        f"STATUS: buffer_len={current_len}, batch_size={self.batch_size}, "
                        f"can_sample={can_sample}, sample_queue_empty={sample_queue_empty}, "
                        f"episodes_received={episodes_received}, batches_produced={batches_produced}, "
                        f"loops={loop_count}, update_step={current_update_step}"
                    )
                    last_status_time = time()

                if can_sample:
                    batch = internal_replay_buffer.sample()
                    # IMPORTANT: Keep batch on CPU for safe inter-process transfer
                    # The trainer will move to CUDA after receiving from the queue
                    self._sample_queue.put(batch)
                    batches_produced += 1
                    if batches_produced <= 5 or batches_produced % 100 == 0:
                        logger.info(f"SAMPLED: batch #{batches_produced}, buffer_len={current_len}")
                elif not self._task_queue.empty():
                    task = self._task_queue.get()
                    if task[0] == "length":
                        self._result_queue.put(current_len)
                        logger.debug(f"LENGTH_QUERY: returned {current_len}")
                    elif task[0] == "shutdown":
                        logger.info("SHUTDOWN: received shutdown signal")
                        break
                    elif task[0] == "save_buffer":
                        path = task[1]
                        from eve_rl.util.experience_cache import save_episodes_npz
                        episodes_list, pos = internal_replay_buffer.export_all()
                        save_episodes_npz(path, episodes_list, pos)
                        self._result_queue.put(len(episodes_list))
                        logger.info(f"SAVE_BUFFER: saved {len(episodes_list)} episodes to {path}")
                    elif task[0] == "load_buffer":
                        path = task[1]
                        from eve_rl.util.experience_cache import load_episodes_npz
                        episodes_list, pos = load_episodes_npz(path)
                        internal_replay_buffer.import_all(episodes_list, pos)
                        self._result_queue.put(len(episodes_list))
                        logger.info(f"LOAD_BUFFER: loaded {len(episodes_list)} episodes from {path}")
                elif not self._push_queue.empty():
                    item = self._push_queue.get()
                    # Unpack (replay_data, explore_step) tuple
                    if isinstance(item, tuple) and len(item) == 2:
                        batch, explore_step = item
                    else:
                        batch, explore_step = item, None
                    internal_replay_buffer.push(batch)
                    episodes_received += 1
                    # Record (explore_step, update_step) for accurate stamping
                    # Read current update_step from shared Value
                    current_update_step = self._shared_update_step.value
                    if explore_step is not None and self._episode_arrival_queue is not None:
                        self._episode_arrival_queue.put((explore_step, current_update_step))
                    if episodes_received <= 10 or episodes_received % 50 == 0:
                        logger.info(f"PUSHED: episode #{episodes_received}, new_buffer_len={len(internal_replay_buffer)}")
                else:
                    sleep(0.0001)

        except Exception as e:
            logger.error(f"EXCEPTION in replay buffer subprocess: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"EXITING: episodes_received={episodes_received}, batches_produced={batches_produced}")
            internal_replay_buffer.close()

    def copy(self):
        return VanillaSharedBase(
            self._push_queue,
            self._sample_queue,
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
            episode_arrival_queue=self._episode_arrival_queue,
        )

    def close(self):
        self._shutdown_event.set()
        self._process.join()
        self._process.close()


class VanillaEpisodeShared(VanillaStepShared):
    def _run_subprocess(self, shared_update_step):
        """Entry point for subprocess - receives shared_update_step as argument."""
        # Store the shared Value that was passed (inherited, not pickled)
        self._shared_update_step = shared_update_step
        # os.nice(15)
        logger = _setup_subprocess_logging()
        logger.info(f"VanillaEpisodeShared.run() starting with capacity={self.capacity}, batch_size={self._batch_size}")
        internal_replay_buffer = VanillaEpisode(self.capacity, self._batch_size)
        logger.info(f"Created VanillaEpisode buffer, starting loop...")
        self.loop(internal_replay_buffer)
