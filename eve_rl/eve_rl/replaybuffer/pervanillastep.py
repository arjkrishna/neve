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
import logging
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
        # RL_IMPROV_16 E3 — stuckness-balanced sampling lane. Stuck states
        # (buckled slack / high contact) are ~10%/1% of the buffer, so the
        # retract-when-stuck behavior is gradient-starved even with correct
        # AWAC weights. A third lane draws `stuck_fraction` of each batch
        # from transitions whose STATE obs shows stuckness:
        #   flat_obs[stuck_slack_index]   > stuck_slack_thresh   OR
        #   flat_obs[stuck_contact_index] > stuck_contact_thresh
        # The indices are env-layout-specific and are passed in by the
        # training script (env5 Gen-4 121-flat: slack=89, contact_max=103);
        # -1 disables that criterion. stuck_fraction 0.0 = lane OFF
        # (legacy behavior, byte-identical).
        stuck_fraction: float = 0.0,
        stuck_slack_index: int = -1,
        stuck_slack_thresh: float = 0.174,
        stuck_contact_index: int = -1,
        stuck_contact_thresh: float = 0.0026,
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
        # RL_IMPROV_15 anti-rail (ported from the LCCA anti-rail branch,
        # EVE_CLEAN_RAIL_MAX) — clean-lane admission filter. A newly
        # collected success enters the balanced clean lane ONLY if its
        # railed-step fraction (fraction of steps with any action dim
        # |a| > 0.95, normalized units) is <= this threshold. Rejected
        # successes still enter the normal buffer (critic data) with
        # is_clean=False — they are just excluded from the amplified lane
        # the AWAC BC term clones, breaking the self-cloning loop
        # (policy clones its own bang-bang noise-successes -> mean rails).
        # Unset/empty env var = disabled (legacy behavior). Reference
        # calibration: seed-diverse cleans ~0.10 railed, self-cloned
        # poison cohort 0.23-0.24 -> threshold 0.15 separates them.
        _rail_max = os.environ.get("EVE_CLEAN_RAIL_MAX", "").strip()
        self.clean_rail_max = float(_rail_max) if _rail_max else None
        self._clean_rail_rejected = 0  # episodes kept out of the clean lane
        # RL_IMPROV_16 E3 — stuck lane state (see __init__ docnote).
        self.stuck_fraction = float(stuck_fraction)
        self.stuck_slack_index = int(stuck_slack_index)
        self.stuck_slack_thresh = float(stuck_slack_thresh)
        self.stuck_contact_index = int(stuck_contact_index)
        self.stuck_contact_thresh = float(stuck_contact_thresh)
        self._stuck = (
            self.stuck_fraction > 0.0
            and (self.stuck_slack_index >= 0 or self.stuck_contact_index >= 0)
        )
        self.is_stuck = np.zeros(self.capacity, dtype=bool)
        self.stuck_tree = SumTree(self.capacity) if self._stuck else None

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
        # RL_IMPROV_16 (incremental save / resume) — monotonic count of
        # transitions EVER pushed (never wraps; slot = idx % capacity), and
        # the watermark up to which chunks have been persisted. The
        # incremental format = append-only transition chunks keyed by this
        # monotonic index + one small, atomically-replaced state file for
        # everything that DRIFTS after push (sum-tree priorities, lane
        # flags, position, counters). See save_incremental_to_dir().
        self._total_pushed = 0
        self._last_incremental_saved = 0

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
        ep_return = float(getattr(episode, "episode_return", 0.0))
        priority = self._initial_priority(is_demo, reached, ep_return)
        # RL_IMPROV_15 anti-rail — clean-lane admission filter (see
        # __init__). NB: must flip `reached` itself (not just the initial
        # clean_tree write) because update_priorities() re-adds any
        # is_clean slot to the clean_tree on every TD-priority update.
        if reached and self.clean_rail_max is not None and len(episode) > 0:
            try:
                acts = np.asarray(episode.actions, dtype=np.float64)
                railed_frac = float(
                    (np.abs(acts) > 0.95).any(axis=tuple(range(1, acts.ndim)))
                    .mean()
                )
            except Exception:
                railed_frac = 0.0  # malformed actions -> admit (legacy)
            if railed_frac > self.clean_rail_max:
                reached = False
                self._clean_rail_rejected += 1
                logging.getLogger(self.__module__).info(
                    "CLEAN_RAIL_FILTER: success episode rejected from clean "
                    "lane (railed_frac=%.3f > %.3f; return=%.2f; total "
                    "rejected=%d) — kept in general buffer.",
                    railed_frac, self.clean_rail_max, ep_return,
                    self._clean_rail_rejected,
                )
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
            # RL_IMPROV_16 E3 — per-transition stuckness flag from the
            # STATE obs (immutable per slot, like is_clean, so
            # update_priorities() lane maintenance stays consistent).
            if self._stuck:
                st = episode.flat_obs[i]
                stuck = False
                if 0 <= self.stuck_slack_index < len(st):
                    stuck = st[self.stuck_slack_index] > self.stuck_slack_thresh
                if not stuck and 0 <= self.stuck_contact_index < len(st):
                    stuck = (
                        st[self.stuck_contact_index]
                        > self.stuck_contact_thresh
                    )
                self.is_stuck[pos] = bool(stuck)
                self.stuck_tree.update(pos, priority if stuck else 0.0)
            self.tree.update(pos, priority)
            if self._balanced:
                self.clean_tree.update(pos, priority if reached else 0.0)
            self.position = int((pos + 1) % self.capacity)
            # RL_IMPROV_16 — monotonic push counter (incremental save).
            self._total_pushed += 1

    # ------------------------------------------------------------------
    # RL_IMPROV_16 — incremental save / load (the resume-grade format).
    #
    # WHY: the legacy per-eval save re-serializes the ENTIRE buffer
    # (~1-2 GB at Gen-4 scale) inside the sample-serving subprocess —
    # wasteful (the file is only ever read once, on resume) and the
    # direct trigger of the post-eval stall window behind the v1/v2
    # eval3 deadlocks. The buffer splits into two unequal halves:
    #   * transitions (obs/actions/rewards/terminals): ~97% of the bytes,
    #     APPEND-ONLY in monotonic push order (slot = idx % capacity) —
    #     saved once, as chunks, never rewritten;
    #   * everything that drifts (sum-tree priorities, lane flags,
    #     position, counters): ~3% of the bytes — re-dumped in full every
    #     save, atomically (tmp + os.replace).
    # A load replays the chunks in monotonic order (later chunks
    # overwrite wrapped slots exactly as the ring did) then applies the
    # state file — byte-faithful to a legacy full save.
    # ------------------------------------------------------------------

    _STATE_FILE = "replay_state.npz"

    def save_incremental_to_dir(self, dir_path: str) -> int:
        """Persist new transitions since the last call + the full small
        state. Returns the number of newly-persisted transitions."""
        os.makedirs(dir_path, exist_ok=True)
        start = self._last_incremental_saved
        end = self._total_pushed
        if end - start > self.capacity:
            # More unsaved pushes than the ring holds — the oldest unsaved
            # ones were already overwritten in memory; persist what exists.
            start = end - self.capacity
        n_new = end - start
        if n_new > 0:
            slots = [i % self.capacity for i in range(start, end)]
            chunk = {
                "start": np.array(start, dtype=np.int64),
                "end": np.array(end, dtype=np.int64),
                "obs_pairs": np.stack([self.buffer[s][0] for s in slots]),
                "actions": np.stack(
                    [np.asarray(self.buffer[s][1]) for s in slots]
                ),
                "rewards": np.array(
                    [self.buffer[s][2] for s in slots], dtype=np.float32
                ),
                "terminals": np.array([self.buffer[s][3] for s in slots]),
            }
            chunk_path = os.path.join(
                dir_path, f"chunk_{start:012d}_{end:012d}.npz"
            )
            np.savez(chunk_path, **chunk)
        state = {
            "total_pushed": np.array(end, dtype=np.int64),
            "capacity": np.array(self.capacity, dtype=np.int64),
            "position": np.array(self.position, dtype=np.int64),
            "n": np.array(len(self.buffer), dtype=np.int64),
            "tree": self.tree.tree,
            "is_demo": self.is_demo,
            "is_clean": self.is_clean,
            "is_stuck": self.is_stuck,
            "episode_returns": self.episode_returns,
            "max_priority": np.array(self.max_priority, dtype=np.float64),
            "sample_count": np.array(self._sample_count, dtype=np.int64),
        }
        tmp_path = os.path.join(dir_path, "replay_state.tmp.npz")
        np.savez(tmp_path, **state)
        os.replace(tmp_path, os.path.join(dir_path, self._STATE_FILE))
        self._last_incremental_saved = end
        return n_new

    def load_incremental_from_dir(self, dir_path: str) -> int:
        """Rebuild the buffer from an incremental-save directory. Returns
        the number of transitions restored. Raises on geometry mismatch or
        missing chunk coverage (a partial dir must not silently load)."""
        import glob as _glob

        state_path = os.path.join(dir_path, self._STATE_FILE)
        with np.load(state_path, allow_pickle=False) as st:
            if int(st["capacity"]) != self.capacity:
                raise ValueError(
                    f"replay capacity mismatch: dir={int(st['capacity'])} "
                    f"buffer={self.capacity}"
                )
            total = int(st["total_pushed"])
            n = int(st["n"])
            tree = np.array(st["tree"], dtype=np.float64)
            is_demo = np.array(st["is_demo"], dtype=bool)
            is_clean = np.array(st["is_clean"], dtype=bool)
            is_stuck = (
                np.array(st["is_stuck"], dtype=bool)
                if "is_stuck" in st
                else np.zeros(self.capacity, dtype=bool)
            )
            episode_returns = np.array(
                st["episode_returns"], dtype=np.float64
            )
            position = int(st["position"])
            max_priority = float(st["max_priority"])
            sample_count = int(st["sample_count"])

        window_start = total - n  # oldest monotonic index still live
        self.buffer = [None] * n
        chunk_paths = sorted(
            _glob.glob(os.path.join(dir_path, "chunk_*.npz"))
        )
        for cp in chunk_paths:
            with np.load(cp, allow_pickle=False) as d:
                cs, ce = int(d["start"]), int(d["end"])
                if ce <= window_start:
                    continue  # fully overwritten by later pushes
                obs_pairs = d["obs_pairs"]
                actions = d["actions"]
                rewards = d["rewards"]
                terminals = d["terminals"]
                for k, m in enumerate(range(cs, ce)):
                    if m < window_start or m >= total:
                        continue
                    self.buffer[m % self.capacity] = (
                        obs_pairs[k], actions[k], rewards[k], terminals[k],
                    )
        missing = sum(1 for b in self.buffer if b is None)
        if missing:
            raise ValueError(
                f"incremental load incomplete: {missing}/{n} slots have no "
                f"covering chunk in {dir_path} — refusing a partial buffer."
            )
        self.position = position
        self.tree.tree = tree
        self.is_demo = is_demo
        self.is_clean = is_clean
        self.is_stuck = is_stuck
        self.episode_returns = episode_returns
        self.max_priority = max_priority
        self._sample_count = sample_count
        self._total_pushed = total
        self._last_incremental_saved = total
        # Rebuild the lane sub-trees from flags + main-tree leaves.
        if self._balanced:
            self.clean_tree = SumTree(self.capacity)
            for i in range(n):
                if self.is_clean[i]:
                    leaf = float(self.tree.tree[i + self.capacity - 1])
                    self.clean_tree.update(i, leaf)
        if self._stuck:
            self.stuck_tree = SumTree(self.capacity)
            for i in range(n):
                if self.is_stuck[i]:
                    leaf = float(self.tree.tree[i + self.capacity - 1])
                    self.stuck_tree.update(i, leaf)
        return n

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
            # RL_IMPROV_16 E3 — stuck-lane membership (rebuild like clean).
            "is_stuck": self.is_stuck,
            "episode_returns": self.episode_returns,
            "max_priority": np.array(self.max_priority, dtype=np.float64),
            "sample_count": np.array(self._sample_count, dtype=np.int64),
            # RL_IMPROV_16 — monotonic push counter (incremental-save
            # bookkeeping survives a legacy round-trip).
            "total_pushed": np.array(self._total_pushed, dtype=np.int64),
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
        # RL_IMPROV_16 E3 — rebuild the stuck lane. Back-compat: an old
        # export (no is_stuck field) loads with an empty lane, and the
        # flags are recomputed only for transitions pushed AFTER the load
        # — acceptable for a seed-cache (heatup data is rarely stuck).
        if self._stuck:
            if "is_stuck" in data:
                self.is_stuck = np.array(data["is_stuck"], dtype=bool)
            self.stuck_tree = SumTree(self.capacity)
            for i in range(n):
                if self.is_stuck[i]:
                    leaf_pri = float(self.tree.tree[i + self.capacity - 1])
                    self.stuck_tree.update(i, leaf_pri)
        # RL_IMPROV_16 — seed the monotonic counter so incremental saves
        # made AFTER a legacy monolithic load keep the slot = idx % capacity
        # invariant (unwrapped: total = n; wrapped: total ≡ position mod
        # capacity). The whole loaded content counts as unsaved so the
        # first incremental save persists it as one bootstrap chunk.
        if "total_pushed" in data:
            self._total_pushed = int(data["total_pushed"])
        elif n < self.capacity:
            self._total_pushed = n
        else:
            self._total_pushed = self.capacity + self.position
        self._last_incremental_saved = max(
            0, self._total_pushed - min(self._total_pushed, self.capacity)
        )
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
        # RL_IMPROV_16 E3 — optional third "stuck" lane (see __init__).
        n_clean = 0
        if self._balanced and self.clean_tree.total() > 0:
            n_clean = int(round(self.balanced_fraction * self.batch_size))
        n_stuck = 0
        if self._stuck and self.stuck_tree.total() > 0:
            n_stuck = int(round(self.stuck_fraction * self.batch_size))
        n_general = self.batch_size - n_clean - n_stuck

        idx_g, smp_g, w_g = self._draw(self.tree, n_general, beta, n)
        if n_clean > 0:
            idx_c, smp_c, w_c = self._draw(self.clean_tree, n_clean, beta, n)
        else:
            idx_c, smp_c, w_c = [], [], []
        if n_stuck > 0:
            idx_s, smp_s, w_s = self._draw(self.stuck_tree, n_stuck, beta, n)
        else:
            idx_s, smp_s, w_s = [], [], []
        # Backfill from the general tree if a stream came up short.
        deficit = (
            self.batch_size - len(smp_g) - len(smp_c) - len(smp_s)
        )
        if deficit > 0:
            idx_d, smp_d, w_d = self._draw(self.tree, deficit, beta, n)
            idx_g += idx_d
            smp_g += smp_d
            w_g += w_d

        indices = idx_g + idx_c + idx_s
        samples = smp_g + smp_c + smp_s
        weights = np.asarray(w_g + w_c + w_s, dtype=np.float64)
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
            # RL_IMPROV_16 E3 — keep the stuck lane's priorities in sync
            # (is_stuck is immutable per slot, mirroring is_clean).
            if self._stuck and self.is_stuck[idx]:
                self.stuck_tree.update(idx, p)

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
