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
import os
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


# Plan v8 — priority-mode constants for the stabilization suite.
_RETURN_FLOOR = -20.0   # return mode: priority = max(episode_return - floor, eps)
_OUTCOME_HIGH = 10.0    # outcome mode: clean (reached-RCCA) transition priority
_OUTCOME_LOW = 1.0      # outcome mode: non-clean transition priority


class PERVanillaStep(ReplayBuffer):
    """Prioritized transition replay buffer.

    Args:
        capacity: max stored transitions (ring buffer).
        batch_size: transitions per ``sample()``.
        alpha: priority exponent (0 = uniform, 1 = full prioritization).
        beta_start: initial IS-correction exponent; annealed → 1.0.
        beta_steps: number of ``sample()`` calls over which beta anneals.
        epsilon: priority floor so zero-TD transitions still get sampled.
        demo_priority_bonus: Plan v8 (DQfD) — additive priority bonus for
            heuristic-seeded (demo) transitions; keeps them from being
            starved as PER drifts toward explore data. 0 = off.
        priority_mode: Plan v8 — "td" (default, |TD|-error priority),
            "return" (episode-return priority, fixed at push), or "outcome"
            (high/low by reached_target_daughter, fixed at push).
        balanced_fraction: Plan v8 — fraction of each batch drawn from a
            second "clean" stream (transitions from episodes that reached
            the target daughter). 0 = off.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: float = 2e7,
        epsilon: float = 1e-6,
        demo_priority_bonus: float = 0.0,
        priority_mode: str = "td",
        balanced_fraction: float = 0.0,
    ):
        self.capacity = int(capacity)
        self._batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = max(1.0, float(beta_steps))
        self.epsilon = epsilon
        self.demo_priority_bonus = demo_priority_bonus
        self.priority_mode = priority_mode
        self.balanced_fraction = balanced_fraction
        self._balanced = balanced_fraction > 0.0

        self.buffer = []
        self.position = 0
        self.tree = SumTree(self.capacity)
        # Plan v8 — per-slot episode-quality metadata.
        self.is_demo = np.zeros(self.capacity, dtype=bool)
        self.is_clean = np.zeros(self.capacity, dtype=bool)
        self.episode_returns = np.zeros(self.capacity, dtype=np.float64)
        # Second sum-tree over the "clean" subset — only built when balanced
        # sampling is on; non-clean / empty slots hold priority 0.
        self.clean_tree = SumTree(self.capacity) if self._balanced else None
        # Raw max priority |td|+eps ever seen; new td-mode transitions enter
        # here so they are sampled at least once.
        self.max_priority = 1.0
        self._sample_count = 0  # drives beta annealing

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _initial_priority(self, is_demo, reached, ep_return) -> float:
        """Stored priority a freshly-pushed transition enters at, per mode."""
        if self.priority_mode == "return":
            raw = max(ep_return - _RETURN_FLOOR, self.epsilon)
        elif self.priority_mode == "outcome":
            raw = _OUTCOME_HIGH if reached else _OUTCOME_LOW
        else:  # "td" — TD unknown at push → enter at the running max.
            raw = self.max_priority
        if is_demo:
            raw += self.demo_priority_bonus
        return raw ** self.alpha

    def push(self, episode):
        # An episode with N actions has N transitions (i = 0..N-1);
        # flat_obs has N+1 entries so flat_obs[i:i+2] is valid.
        is_demo = bool(getattr(episode, "is_demo", False))
        reached = bool(getattr(episode, "reached_target_daughter", False))
        # Plan v13 — EVE_FREEZE_CLEAN_LANE: block NEW episodes from the
        # balanced clean lane (loaded/restored flags are untouched). Breaks
        # the self-cloning feedback loop found in the 2g forensic: the policy's
        # own noisy successes flooded the 60%-weighted lane (674 new cleans =
        # 50.2% of lane), turning the AWAC loss into BC on the policy's own
        # rail-saturated behavior -> mean-rail saturation (|mu| x2.4) ->
        # deterministic eval collapse 33.7%->6.1% while noisy rollouts rose.
        # With the freeze, the lane stays the diverse pre-resume seed cleans.
        if reached and os.environ.get("EVE_FREEZE_CLEAN_LANE", "") not in ("", "0"):
            reached = False
        ep_return = float(getattr(episode, "episode_return", 0.0))
        priority = self._initial_priority(is_demo, reached, ep_return)
        for i in range(len(episode)):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            episode_np = (
                np.array(episode.flat_obs[i : i + 2]),  # state + next_state
                np.array(episode.actions[i]),
                np.array(episode.rewards[i]),
                np.array(episode.terminals[i]),
            )
            pos = self.position
            self.buffer[pos] = episode_np
            self.is_demo[pos] = is_demo
            self.is_clean[pos] = reached
            self.episode_returns[pos] = ep_return
            self.tree.update(pos, priority)
            if self._balanced:
                self.clean_tree.update(pos, priority if reached else 0.0)
            self.position = int((pos + 1) % self.capacity)

    def export_all(self) -> dict:
        """Plan v10 — serialize the full PER buffer state to a dict of
        numpy arrays (for np.savez). Includes the transitions AND the
        PER-specific state (sum-tree priorities, is_demo, is_clean,
        episode_returns, max_priority, sample_count) so a reload restores
        prioritization faithfully. The clean_tree is rebuilt from
        is_clean + leaf priorities on import (not stored)."""
        n = len(self.buffer)
        if n == 0:
            return {"n": np.array(0, dtype=np.int64)}
        obs_pairs = np.stack([self.buffer[i][0] for i in range(n)])
        actions = np.stack([np.asarray(self.buffer[i][1]) for i in range(n)])
        rewards = np.array([self.buffer[i][2] for i in range(n)], dtype=np.float32)
        terminals = np.array([self.buffer[i][3] for i in range(n)])
        return {
            "n": np.array(n, dtype=np.int64),
            "capacity": np.array(self.capacity, dtype=np.int64),
            "obs_pairs": obs_pairs,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "position": np.array(self.position, dtype=np.int64),
            "tree": self.tree.tree,
            "is_demo": self.is_demo,
            "is_clean": self.is_clean,
            "episode_returns": self.episode_returns,
            "max_priority": np.array(self.max_priority, dtype=np.float64),
            "sample_count": np.array(self._sample_count, dtype=np.int64),
        }

    def import_all(self, data) -> int:
        """Plan v10 — repopulate the buffer from an export_all() npz.
        Returns the number of transitions loaded. Capacity must match
        (ring-buffer geometry); if it differs we skip the load defensively.
        Rebuilds the clean_tree from is_clean + leaf priorities."""
        n = int(data["n"])
        if n == 0:
            return 0
        if "capacity" in data and int(data["capacity"]) != self.capacity:
            # Geometry mismatch — refuse rather than corrupt the sum-tree.
            raise ValueError(
                f"replay capacity mismatch: file={int(data['capacity'])} "
                f"buffer={self.capacity}"
            )
        obs_pairs = data["obs_pairs"]; actions = data["actions"]
        rewards = data["rewards"]; terminals = data["terminals"]
        self.buffer = [
            (obs_pairs[i], actions[i], rewards[i], terminals[i])
            for i in range(n)
        ]
        self.position = int(data["position"])
        self.tree.tree = np.array(data["tree"], dtype=np.float64)
        self.is_demo = np.array(data["is_demo"], dtype=bool)
        self.is_clean = np.array(data["is_clean"], dtype=bool)
        self.episode_returns = np.array(data["episode_returns"], dtype=np.float64)
        self.max_priority = float(data["max_priority"])
        self._sample_count = int(data["sample_count"])
        # Rebuild the clean sub-tree from is_clean + the main tree's leaves.
        if self._balanced:
            self.clean_tree = SumTree(self.capacity)
            for i in range(n):
                if self.is_clean[i]:
                    leaf_pri = float(self.tree.tree[i + self.capacity - 1])
                    self.clean_tree.update(i, leaf_pri)
        return n

    def _draw(self, tree, count, beta, n):
        """Stratified proportional draw of ``count`` leaves from ``tree``.
        Returns (indices, samples, is_weight-list). Per-stream IS weights
        use that stream's total — an accepted approximation when balanced
        sampling mixes two streams."""
        indices, samples, weights = [], [], []
        total = tree.total()
        if count <= 0 or total <= 0:
            return indices, samples, weights
        segment = total / count
        for k in range(count):
            value = random.uniform(segment * k, segment * (k + 1))
            leaf_idx, priority = tree.get_leaf(value)
            prob = priority / max(total, 1e-12)
            indices.append(leaf_idx)
            samples.append(self.buffer[leaf_idx])
            weights.append((n * max(prob, 1e-12)) ** (-beta))
        return indices, samples, weights

    def sample(self) -> Batch:
        # Anneal beta from beta_start → 1.0 over beta_steps sample() calls.
        beta = min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * (self._sample_count / self.beta_steps),
        )
        self._sample_count += 1
        n = len(self.buffer)

        # Plan v8 — balanced two-stream sampling: a fixed fraction of the
        # batch from the "clean" sub-tree, the rest from the full buffer.
        n_clean = 0
        if self._balanced and self.clean_tree.total() > 0:
            n_clean = int(round(self.balanced_fraction * self.batch_size))
        n_general = self.batch_size - n_clean

        idx_g, smp_g, w_g = self._draw(self.tree, n_general, beta, n)
        if n_clean > 0:
            idx_c, smp_c, w_c = self._draw(self.clean_tree, n_clean, beta, n)
        else:
            idx_c, smp_c, w_c = [], [], []
        # Backfill from the general tree if a stream came up short.
        deficit = self.batch_size - len(smp_g) - len(smp_c)
        if deficit > 0:
            idx_d, smp_d, w_d = self._draw(self.tree, deficit, beta, n)
            idx_g += idx_d
            smp_g += smp_d
            w_g += w_d

        indices = idx_g + idx_c
        samples = smp_g + smp_c
        weights = np.asarray(w_g + w_c, dtype=np.float64)
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
        """Refresh sampled transitions' priorities from fresh TD errors.
        Only meaningful in "td" mode — "return"/"outcome" priorities are
        fixed at push, so this is a no-op for them. The DQfD demo bonus is
        re-applied so demo transitions keep a permanent priority floor."""
        if self.priority_mode != "td":
            return
        for idx, td in zip(indices, td_errors):
            idx = int(idx)
            raw = abs(float(td)) + self.epsilon
            if raw > self.max_priority:
                self.max_priority = raw
            if self.is_demo[idx]:
                raw += self.demo_priority_bonus
            p = raw ** self.alpha
            self.tree.update(idx, p)
            if self._balanced and self.is_clean[idx]:
                self.clean_tree.update(idx, p)

    def bulk_import_arrays(self, arrays: dict) -> int:
        """Plan v11 Stage 1 — fast offline-buffer load from
        ``buffer_filter.concat_filtered()`` output. Bypasses the per-
        transition Episode wrapping by writing directly into the ring
        buffer + companion arrays. After loading, callers SHOULD invoke
        ``_recompute_all_priorities()`` to rebuild the sum-tree at a
        uniform initial priority so V2-vs-V3 priority skew (risk audit
        #8) is erased.

        Required keys: obs_pairs, actions, rewards, terminals, is_demo,
        is_clean, episode_returns. Returns the number of transitions
        loaded (clamped at capacity)."""
        n = int(arrays["actions"].shape[0])
        if n == 0:
            return 0
        if n > self.capacity:
            raise ValueError(
                f"Filtered buffer ({n}) exceeds PER capacity ({self.capacity}). "
                f"Increase --replay_buffer_size or tighten the filter."
            )
        obs_pairs = np.asarray(arrays["obs_pairs"])
        actions = np.asarray(arrays["actions"])
        rewards = np.asarray(arrays["rewards"])
        terminals = np.asarray(arrays["terminals"])
        is_demo = np.asarray(arrays["is_demo"], dtype=bool)
        is_clean = np.asarray(arrays["is_clean"], dtype=bool)
        episode_returns = np.asarray(arrays["episode_returns"], dtype=np.float64)
        # Resize the buffer list once and write in order.
        self.buffer = [None] * n
        for i in range(n):
            self.buffer[i] = (
                obs_pairs[i], actions[i], rewards[i], terminals[i],
            )
        self.is_demo[:n] = is_demo
        self.is_clean[:n] = is_clean
        self.episode_returns[:n] = episode_returns
        self.position = int(n % self.capacity)
        return n

    def _recompute_all_priorities(self) -> None:
        """Plan v11 Stage 1 — rebuild every leaf priority from the
        current per-transition flags (`is_demo`, `is_clean`,
        `episode_returns`) using the same `_initial_priority` rule that
        `push()` would apply at the running `max_priority`. After a
        bulk-import this erases any V2/V3 push-order skew in the sum-tree
        (risk audit #8) and gives PER a uniform starting distribution."""
        n = len(self.buffer)
        if n == 0:
            return
        # Reset the sum-tree to zero before rewriting leaves.
        self.tree.tree[:] = 0.0
        if self._balanced:
            self.clean_tree.tree[:] = 0.0
        for i in range(n):
            p = self._initial_priority(
                bool(self.is_demo[i]),
                bool(self.is_clean[i]),
                float(self.episode_returns[i]),
            )
            self.tree.update(i, p)
            if self._balanced and self.is_clean[i]:
                self.clean_tree.update(i, p)

    def __len__(self):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(
            self.capacity, self._batch_size,
            self.alpha, self.beta_start, self.beta_steps, self.epsilon,
            self.demo_priority_bonus, self.priority_mode, self.balanced_fraction,
        )
        for entry in self.buffer:
            copy.buffer.append(entry)
        copy.position = self.position
        copy.tree.tree = self.tree.tree.copy()
        copy.is_demo = self.is_demo.copy()
        copy.is_clean = self.is_clean.copy()
        copy.episode_returns = self.episode_returns.copy()
        if self._balanced:
            copy.clean_tree.tree = self.clean_tree.tree.copy()
        copy.max_priority = self.max_priority
        copy._sample_count = self._sample_count
        return copy

    def close(self):
        del self.buffer
