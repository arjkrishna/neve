"""Prioritized Experience Replay step buffer (RL_IMPROV_9 §14, Plan v7).

Transition-level replay (like ``VanillaStep``) but samples each transition
with probability proportional to its priority `p_i^alpha`, where `p_i` is
the last-seen TD-error magnitude. Importance-sampling (IS) weights
`w_i = (N * P(i))^{-beta}` correct the bias the prioritized sampling
introduces; SAC multiplies each per-sample loss by `w_i`.

A binary **sum-tree** backs the priorities: O(log N) sample and O(log N)
priority update — negligible overhead (~1 ms/step) versus the SOFA env
step. A naive parallel priority array with ``np.random.choice(p=...)``
would be O(N) per sample (~ms on a 1e6 buffer) — rejected.
"""
import random
import numpy as np
import torch

from .replaybuffer import ReplayBuffer, Episode, Batch


class SumTree:
    """Fixed-capacity binary sum-tree over per-leaf priorities.

    Internal nodes hold the sum of their subtree; leaves hold individual
    priorities. ``total()`` is the root. ``get_leaf(value)`` descends in
    O(log N) to the leaf whose cumulative-priority interval contains
    ``value`` — the basis of proportional sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # 2*capacity-1 nodes: capacity-1 internal + capacity leaves.
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf_idx: int, priority: float) -> None:
        """Set leaf ``leaf_idx`` (0..capacity-1) to ``priority`` and
        propagate the delta to the root."""
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while True:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, value: float):
        """Return (leaf_idx, priority) for the leaf whose cumulative
        interval contains ``value`` in [0, total())."""
        idx = 0
        while idx < self.capacity - 1:  # while idx is an internal node
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1), float(self.tree[idx])


class PERVanillaStep(ReplayBuffer):
    """Prioritized transition replay buffer.

    Args:
        capacity: max stored transitions (ring buffer).
        batch_size: transitions per ``sample()``.
        alpha: priority exponent (0 = uniform, 1 = full prioritization).
        beta_start: initial IS-correction exponent; annealed → 1.0.
        beta_steps: number of ``sample()`` calls over which beta anneals
            from ``beta_start`` to 1.0.
        epsilon: priority floor so zero-TD transitions still get sampled.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: float = 2e7,
        epsilon: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self._batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = max(1.0, float(beta_steps))
        self.epsilon = epsilon

        self.buffer = []
        self.position = 0
        self.tree = SumTree(self.capacity)
        # Raw max priority |td|+eps ever seen; new transitions enter at
        # this raw value so they are sampled at least once.
        self.max_priority = 1.0
        self._sample_count = 0  # drives beta annealing

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        # An episode with N actions has N transitions (i = 0..N-1);
        # flat_obs has N+1 entries so flat_obs[N-1:N+1] is valid.
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
            # New transitions enter at max priority (standard PER) so the
            # learner sees each fresh transition at least once.
            self.tree.update(self.position, self.max_priority ** self.alpha)
            self.position = int((self.position + 1) % self.capacity)

    def sample(self) -> Batch:
        # Anneal beta from beta_start → 1.0 over beta_steps sample() calls.
        beta = min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * (self._sample_count / self.beta_steps),
        )
        self._sample_count += 1

        total = self.tree.total()
        segment = total / self.batch_size
        indices, samples, priorities = [], [], []
        for k in range(self.batch_size):
            # Stratified sampling: one draw per equal-mass segment.
            value = random.uniform(segment * k, segment * (k + 1))
            leaf_idx, priority = self.tree.get_leaf(value)
            indices.append(leaf_idx)
            priorities.append(priority)
            samples.append(self.buffer[leaf_idx])

        # IS weights: w_i = (N * P(i))^{-beta}, normalized by max.
        n = len(self.buffer)
        probs = np.asarray(priorities, dtype=np.float64) / max(total, 1e-12)
        weights = (n * np.maximum(probs, 1e-12)) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)

        # Stack transition tuples — same layout as VanillaStep.sample().
        batch = list(map(np.stack, zip(*samples)))
        batch = [torch.from_numpy(entry) for entry in batch]
        batch[1] = batch[1].unsqueeze(1)              # actions  → (B,1,act)
        batch[2] = batch[2].unsqueeze(1).unsqueeze(1)  # rewards  → (B,1,1)
        batch[3] = batch[3].unsqueeze(1).unsqueeze(1)  # terminals→ (B,1,1)

        is_weights = torch.from_numpy(weights.astype(np.float32))
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return Batch(
            batch[0], batch[1], batch[2], batch[3],
            None,                # padding_mask — step transitions, no padding
            is_weights,
            idx_tensor,
        )

    def update_priorities(self, indices, td_errors) -> None:
        """Set sampled transitions' priorities from their fresh TD errors.
        ``priority_stored = (|td| + epsilon) ** alpha``."""
        for idx, td in zip(indices, td_errors):
            raw = abs(float(td)) + self.epsilon
            self.tree.update(int(idx), raw ** self.alpha)
            if raw > self.max_priority:
                self.max_priority = raw

    def __len__(self):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(
            self.capacity, self._batch_size,
            self.alpha, self.beta_start, self.beta_steps, self.epsilon,
        )
        for entry in self.buffer:
            copy.buffer.append(entry)
        copy.position = self.position
        copy.tree.tree = self.tree.tree.copy()
        copy.max_priority = self.max_priority
        copy._sample_count = self._sample_count
        return copy

    def close(self):
        del self.buffer
