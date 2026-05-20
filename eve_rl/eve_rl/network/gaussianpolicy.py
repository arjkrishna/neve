from typing import Optional, Tuple
import logging
import torch
from torch.distributions import Normal

from .component import Component, ComponentDummy
from .network import Network


class GaussianPolicy(Network):
    def __init__(
        self,
        body: Component,
        n_observations: int,
        n_actions: int,
        head: Optional[Component] = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__module__)

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.body = body
        self.head = head or ComponentDummy()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.head.n_inputs = n_observations
        self.body.n_inputs = self.head.n_outputs
        self.body.output_layer_size = [n_actions, n_actions]

    @property
    def device(self) -> torch.device:
        return self.body.device

    def forward(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_out = self.head(obs_batch)

        mean, log_std = self.body.forward(head_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def forward_play(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_out = self.head.forward_play(obs_batch)
        mean, log_std = self.body.forward_play(head_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def log_prob(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """Plan v8 (AWAC) — log-probability of an ARBITRARY given action
        (the SAC path only log-probs its own fresh sample inline). The
        action is a tanh-squashed value in (-1, 1); invert the squash with
        atanh, score the pre-squash z under N(mean, std), then apply the
        tanh change-of-variables correction. Summed over the action dim,
        keepdim — same convention as SAC's `_get_update_action`."""
        mean, log_std = self.forward(obs_batch)
        std = log_std.exp()
        a = torch.clamp(action_batch, -1.0 + epsilon, 1.0 - epsilon)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)
        log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - a.pow(2) + epsilon)
        # Plan v8 fix — clamp the summed log-prob. An arbitrary buffer action
        # can sit many sigma from a collapsed policy mean → log_prob → -1e18
        # → the AWAC BC loss explodes. The floor bounds the worst-case
        # per-sample loss (SAC never hits this — it only scores its own
        # fresh sample, which is O(1) sigma from the mean by construction).
        return log_prob.sum(-1, keepdim=True).clamp(min=-20.0)

    def reset(self) -> None:
        ...
