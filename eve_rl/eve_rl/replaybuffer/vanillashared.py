from time import sleep, time
from multiprocessing.synchronize import Lock as mp_lock
from multiprocessing.synchronize import Event as mp_event
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
    ):
        self._push_queue = push_queue
        self._task_queue = task_queue
        self._sample_queue = sample_queue
        self._result_queue = result_queue
        self._request_lock = request_lock
        self._shutdown_event = shutdown_event
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        logger = logging.getLogger("ReplayBufferSubprocess")
        if not self._shutdown_event.is_set():
            try:
                replay_data = episode.to_replay()
                logger.info(f"PUSH CALLED: episode length={len(episode) if hasattr(episode, '__len__') else 'N/A'}, putting in queue...")
                self._push_queue.put(replay_data)
                logger.info(f"PUSH COMPLETE: episode put in queue successfully")
            except Exception as e:
                logger.error(f"PUSH ERROR: {e}", exc_info=True)
                raise
        else:
            logger.warning("PUSH SKIPPED: shutdown event is set")

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
        )
        self.capacity = capacity
        self.sample_device = sample_device
        self._process = mp.Process(target=self.run)
        self._process.start()

    def run(self):
        internal_replay_buffer = VanillaStep(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: ReplayBuffer):
        logger = _setup_subprocess_logging()
        logger.info(f"ReplayBuffer subprocess started. batch_size={self.batch_size}, capacity={self.capacity}")
        
        loop_count = 0
        episodes_received = 0
        batches_produced = 0
        last_status_time = time()
        
        try:
            while not self._shutdown_event.is_set():
                loop_count += 1
                current_len = len(internal_replay_buffer)
                sample_queue_empty = self._sample_queue.empty()
                can_sample = sample_queue_empty and current_len > self.batch_size
                
                # Log status every 10 seconds
                if time() - last_status_time > 10:
                    logger.info(
                        f"STATUS: buffer_len={current_len}, batch_size={self.batch_size}, "
                        f"can_sample={can_sample}, sample_queue_empty={sample_queue_empty}, "
                        f"episodes_received={episodes_received}, batches_produced={batches_produced}, "
                        f"loops={loop_count}"
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
                elif not self._push_queue.empty():
                    batch = self._push_queue.get()
                    internal_replay_buffer.push(batch)
                    episodes_received += 1
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
        )

    def close(self):
        self._shutdown_event.set()
        self._process.join()
        self._process.close()


class VanillaEpisodeShared(VanillaStepShared):
    def run(self):
        # os.nice(15)
        logger = _setup_subprocess_logging()
        logger.info(f"VanillaEpisodeShared.run() starting with capacity={self.capacity}, batch_size={self._batch_size}")
        internal_replay_buffer = VanillaEpisode(self.capacity, self._batch_size)
        logger.info(f"Created VanillaEpisode buffer, starting loop...")
        self.loop(internal_replay_buffer)
