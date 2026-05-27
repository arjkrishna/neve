from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union
import numpy as np

import torch

from ..util import EveRLObject


class Episode:
    def __init__(
        self,
        reset_obs: Dict[str, np.ndarray],
        reset_flat_obs: np.ndarray,
        flat_obs_to_obs: Optional[Union[List, Dict]] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.obs: List[Dict[str, np.ndarray]] = [reset_obs]
        self.flat_obs: List[np.ndarray] = [reset_flat_obs]
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminals: List[bool] = []
        self.truncations: List[bool] = []
        self.infos: List[Dict[str, np.ndarray]] = []
        self.episode_reward: float = 0.0
        self.flat_state_to_state = flat_obs_to_obs
        self.seed = seed
        self.options = options
        # Step counters at episode completion (stamped by worker)
        self.explore_step_at_completion: Optional[int] = None
        self.update_step_at_completion: Optional[int] = None

    def add_transition(
        self,
        obs: Dict[str, np.ndarray],
        flat_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncation: bool,
        info: Dict[str, np.ndarray],
    ):
        self.obs.append(obs)
        self.flat_obs.append(flat_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.truncations.append(truncation)
        self.infos.append(info)
        self.episode_reward += reward

    def to_replay(self):
        # Plan v8 — carry episode-level quality flags into the buffer form.
        # `infos` itself is dropped (too heavy), but the last step's info
        # holds the per-episode latches the env5 state machine sets.
        # Plan v9 Change 2 — prefer strict final-branch equality over the
        # buggy ever-touched `reached_target_daughter` latch. The latch is
        # kept as a fallback for envs that don't expose final_branch_idx
        # in info (e.g. legacy env4 replays).
        reached = False
        if self.infos:
            info = self.infos[-1]
            final_idx = info.get("final_branch_idx")
            target_idx = info.get("target_daughter_branch_idx")
            if final_idx is not None and target_idx is not None:
                reached = int(final_idx) == int(target_idx)
            else:
                reached = bool(info.get("reached_target_daughter", False))
        return EpisodeReplay(
            self.flat_obs, self.actions, self.rewards, self.terminals,
            episode_return=float(self.episode_reward),
            reached_target_daughter=reached,
        )

    def __len__(self):
        return len(self.actions)


@dataclass
class EpisodeReplay:
    flat_obs: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    terminals: List[bool]
    # Plan v8 — episode-level quality metadata for the stabilization-suite
    # samplers. Trailing defaults → backward-compatible; `to_replay()` on an
    # EpisodeReplay returns self with these preserved.
    #   is_demo                : heuristic-seeded (DQfD priority bonus)
    #   episode_return         : sum of rewards (return-prioritized replay)
    #   reached_target_daughter: clean-thread flag (balanced sampling / outcome)
    is_demo: bool = False
    episode_return: float = 0.0
    reached_target_daughter: bool = False

    def __len__(self):
        return len(self.actions)

    def to_replay(self):
        # Already the replay form — idempotent. The shared-buffer handle's
        # push() calls episode.to_replay() before queueing; this lets a
        # cache-loaded EpisodeReplay be pushed the same as a live Episode.
        return self


class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    padding_mask: torch.Tensor = None
    # Plan v7 — Prioritized Experience Replay. Both default None, so
    # uniform episode/step batches are unaffected (SAC guards on
    # `is_weights is not None`). `is_weights` are per-sample IS-correction
    # weights (multiply the loss); `indices` are the buffer leaf indices
    # of the sampled transitions (the trainer sends them back so the
    # buffer can refresh those transitions' priorities).
    is_weights: torch.Tensor = None
    indices: torch.Tensor = None

    def to(self, device: torch.device, non_blocking=False):
        obs = self.obs.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        actions = self.actions.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        rewards = self.rewards.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        terminals = self.terminals.to(
            device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        ).share_memory_()
        if self.padding_mask is not None:
            padding_mask = self.padding_mask.to(
                device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ).share_memory_()
        else:
            padding_mask = None
        if self.is_weights is not None:
            is_weights = self.is_weights.to(
                device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ).share_memory_()
        else:
            is_weights = None
        # `indices` stay on CPU as a long tensor — they never enter the
        # network; the trainer uses them only to address buffer slots
        # when sending priority updates back.
        return Batch(
            obs, actions, rewards, terminals, padding_mask,
            is_weights, self.indices,
        )


class ReplayBuffer(EveRLObject, ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        ...

    @abstractmethod
    def push(self, episode: Union[Episode, EpisodeReplay]) -> None:
        ...

    @abstractmethod
    def sample(self) -> Batch:
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
