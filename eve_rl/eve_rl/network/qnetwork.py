# pylint: disable=no-member
# pylint: disable=arguments-differ

from typing import Optional, Tuple
import os
import torch

from .component import Component, ComponentDummy
from .network import Network


def _parse_critic_zero_obs() -> Tuple[int, ...]:
    raw = os.environ.get("EVE_RL_CRITIC_ZERO_OBS", "").strip()
    if not raw:
        return ()
    return tuple(sorted({int(t) for t in raw.replace(" ", "").split(",") if t}))


# Flat obs indices zeroed at the CRITIC's input (q1, q2 and their targets --
# targets are copies of this class, and the parse is module-level, so every
# instance in the process agrees). Counterpart of EVE_RL_ACTOR_ZERO_OBS in
# gaussianpolicy.py: set BOTH to the same indices for a full ablation in which
# no network anywhere sees those dims; set only the actor one for the
# asymmetric (teacher-critic) ablation. Unset -> empty -> untouched.
#
# Deliberately duplicated rather than shared with gaussianpolicy.py: that file
# is bind-mounted into live containers, and a new shared import would break
# workers that respawn against an image without it.
_CRITIC_ZERO_OBS = _parse_critic_zero_obs()


class QNetwork(Network):
    def __init__(
        self,
        body: Component,
        n_observations: int,
        n_actions: int,
        head: Optional[Component] = None,
    ):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.body = body
        self.head = head or ComponentDummy()

        self.head.n_inputs = n_observations
        self.body.n_inputs = self.head.n_outputs + n_actions
        self.body.output_layer_size = 1

    @property
    def device(self) -> torch.device:
        return self.body.device

    def _mask_critic_obs(self, obs_batch: torch.Tensor) -> torch.Tensor:
        if not _CRITIC_ZERO_OBS:
            return obs_batch
        width = obs_batch.shape[-1]
        idx = [i for i in _CRITIC_ZERO_OBS if i < width]
        if not idx:
            return obs_batch
        keep = torch.ones(width, dtype=obs_batch.dtype, device=obs_batch.device)
        keep[idx] = 0.0
        # Multiply rather than assign in place: obs_batch is the replay batch
        # shared with the policy update and the other critic.
        return obs_batch * keep

    def forward(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        obs_batch = self._mask_critic_obs(obs_batch)
        head_out = self.head(obs_batch)
        body_in = torch.dstack([head_out, action_batch])
        q_value_batch = self.body(body_in)
        return q_value_batch

    def forward_play(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        obs_batch = self._mask_critic_obs(obs_batch)
        head_out = self.head.forward_play(obs_batch)
        body_in = torch.dstack([head_out, action_batch])
        q_value_batch = self.body.forward_play(body_in)
        return q_value_batch

    def reset(self) -> None:
        ...
