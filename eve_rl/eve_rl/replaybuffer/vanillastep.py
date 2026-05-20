import random
import numpy as np
import torch
from .replaybuffer import ReplayBuffer, Episode, Batch


class VanillaStep(ReplayBuffer):
    def __init__(
        self, capacity: int, batch_size: int, balanced_fraction: float = 0.0
    ):
        self.capacity = int(capacity)
        self._batch_size = batch_size
        self.buffer = []
        self.position = 0
        # Plan v8 — balanced two-stream sampling for the uniform buffer.
        # `is_clean` marks transitions from episodes that reached the target
        # daughter; `sample()` then draws a fixed fraction from that subset.
        self.balanced_fraction = balanced_fraction
        self._balanced = balanced_fraction > 0.0
        self.is_clean = np.zeros(self.capacity, dtype=bool)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        # An episode with N actions has N transitions (i = 0..N-1).
        # flat_obs holds N+1 entries (reset state + one per step), so
        # flat_obs[i:i+2] is valid for the final i = N-1 too. The old
        # `range(len(episode) - 1)` dropped transition N-1 — the terminal
        # one, carrying the done flag and the terminal reward — so iterate
        # the full range.
        reached = bool(getattr(episode, "reached_target_daughter", False))
        for i in range(len(episode)):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            episode_np = (
                np.array(episode.flat_obs[i : i + 2]),  # state + next_state
                np.array(episode.actions[i]),
                np.array(episode.rewards[i]),
                np.array(episode.terminals[i]),
            )
            self.buffer[self.position] = episode_np
            self.is_clean[self.position] = reached
            self.position = int((self.position + 1) % self.capacity)  # ring buffer

    def _stack(self, samples) -> Batch:
        batch = list(map(np.stack, zip(*samples)))  # stack for each element
        batch = [torch.from_numpy(batch_entry) for batch_entry in batch]
        batch[1] = batch[1].unsqueeze(1)
        batch[2] = batch[2].unsqueeze(1).unsqueeze(1)
        batch[3] = batch[3].unsqueeze(1).unsqueeze(1)
        return Batch(*batch)

    def sample(self) -> Batch:
        if self._balanced:
            # Plan v8 — balanced two-stream: a fixed fraction of the batch
            # drawn uniformly from the "clean" subset, the rest from all.
            n = len(self.buffer)
            clean_idx = np.where(self.is_clean[:n])[0]
            n_clean = (
                int(round(self.balanced_fraction * self.batch_size))
                if len(clean_idx) > 0 else 0
            )
            n_general = self.batch_size - n_clean
            samples = random.sample(self.buffer, n_general)
            if n_clean > 0:
                picks = np.random.choice(
                    clean_idx, size=n_clean, replace=len(clean_idx) < n_clean
                )
                samples += [self.buffer[int(i)] for i in picks]
        else:
            samples = random.sample(self.buffer, self.batch_size)
        return self._stack(samples)

    def __len__(
        self,
    ):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(
            self.capacity, self.batch_size, self.balanced_fraction
        )
        for i in range(len(self.buffer)):
            copy.buffer.append(self.buffer[i])
        copy.position = self.position
        copy.is_clean = self.is_clean.copy()
        return copy

    def close(self):
        del self.buffer
