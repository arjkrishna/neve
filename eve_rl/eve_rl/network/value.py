# pylint: disable=no-member
# pylint: disable=arguments-differ
"""Plan v11 Stage 1B — V-network for IQL.

Mirrors `QNetwork` but takes only the state (no action concat). Outputs a
single scalar value V(s) per observation. Used by the IQL algorithm to
train an expectile regression of Q(s,a) → V(s) and as the AWR baseline
in the policy update.
"""

from typing import Optional
import torch

from .component import Component, ComponentDummy
from .network import Network


class VNetwork(Network):
    def __init__(
        self,
        body: Component,
        n_observations: int,
        head: Optional[Component] = None,
    ):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = 0
        self.body = body
        self.head = head or ComponentDummy()

        self.head.n_inputs = n_observations
        # No action concat — body sees only the head output.
        self.body.n_inputs = self.head.n_outputs
        self.body.output_layer_size = 1

    @property
    def device(self) -> torch.device:
        return self.body.device

    def forward(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        head_out = self.head(obs_batch)
        v_value_batch = self.body(head_out)
        return v_value_batch

    def forward_play(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> torch.Tensor:
        head_out = self.head.forward_play(obs_batch)
        v_value_batch = self.body.forward_play(head_out)
        return v_value_batch

    def reset(self) -> None:
        ...
