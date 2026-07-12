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
        # Midpoint-0 band: under the soft tanh rescale in forward(), a
        # freshly initialized head (raw log_std ~ 0) maps to the band
        # MIDPOINT, so the legacy (-20, 2) default would initialize
        # collapsed at log_std=-9 (std ~ 1.2e-4). (-2, 2) initializes at
        # log_std=0 (std 1), matching the old hard-clamp init behavior.
        log_std_min: float = -2,
        log_std_max: float = 2,
        # Gen-4 — auxiliary prediction heads (privileged-label supervision).
        # n_aux > 0 adds a third linear head off the last hidden layer whose
        # output is stashed in self._last_aux each forward(); SAC adds an
        # MSE loss against privileged-tail labels (representation shaping:
        # the policy learns to INFER contact/buckle state from deployable
        # signals). 0 = no head, byte-identical behavior.
        n_aux: int = 0,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__module__)

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_aux = int(n_aux)
        self.body = body
        self.head = head or ComponentDummy()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self._last_aux = None

        self.head.n_inputs = n_observations
        self.body.n_inputs = self.head.n_outputs
        if self.n_aux > 0:
            self.body.output_layer_size = [n_actions, n_actions, self.n_aux]
        else:
            self.body.output_layer_size = [n_actions, n_actions]

    @property
    def device(self) -> torch.device:
        return self.body.device

    def forward(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gen-4 asymmetric critic — the flat observation may carry a
        # privileged tail (appended LAST in the env's ObsDict) that only
        # the critics consume. The policy sees the deployable prefix only;
        # slicing here covers every caller (log_prob, SAC updates,
        # exploration/eval, play-only workers) at one chokepoint.
        obs_batch = obs_batch[..., : self.n_observations]
        head_out = self.head(obs_batch)

        body_out = self.body.forward(head_out)
        if self.n_aux > 0:
            mean, log_std, self._last_aux = body_out
        else:
            mean, log_std = body_out
        # Soft log_std bound — a hard clamp has zero gradient outside the
        # band, so a railed head becomes a one-way ratchet (only movable by
        # shared-trunk drift); the tanh rescale stays bounded in
        # (log_std_min, log_std_max) with nonzero gradient everywhere. Note:
        # a raw pre-squash output of 0 maps to the band midpoint (for the
        # operational band (-2, 0) that is log_std=-1, std~0.37).
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (torch.tanh(log_std) + 1.0)

        return mean, log_std

    def forward_play(
        self, obs_batch: torch.Tensor, *args, **kwds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Same privileged-tail slice as forward(); see comment there.
        obs_batch = obs_batch[..., : self.n_observations]
        head_out = self.head.forward_play(obs_batch)
        body_out = self.body.forward_play(head_out)
        if self.n_aux > 0:
            mean, log_std, _ = body_out
        else:
            mean, log_std = body_out
        # Soft log_std bound — same tanh rescale as forward(); see comment there.
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (torch.tanh(log_std) + 1.0)
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
        # Plan v8 fix, leaky variant — bound the summed log-prob. An
        # arbitrary buffer action can sit many sigma from a collapsed policy
        # mean → log_prob → -1e18 → the AWAC BC loss explodes. A HARD floor
        # zeroes the BC gradient on exactly the demos the policy is farthest
        # from; instead, values below -20 keep a 5% gradient (direction
        # preserved, magnitude damped 20x) down to raw -220; below that the
        # hard -30 bound caps the worst-case loss (weight<=20 -> per-sample
        # loss <=600). (SAC never hits this — it only scores its own fresh
        # sample, which is O(1) sigma from the mean by construction.)
        lp = log_prob.sum(-1, keepdim=True)
        floor, leak, hard_min = -20.0, 0.05, -30.0
        # Sanitize NaN/Inf before the floor: torch.where(lp>=floor,...) would
        # take the FALSE branch for NaN and pass it straight through into the
        # BC loss, poisoning the whole run. The soft log_std bound makes a
        # NaN here unlikely, but the floor should actually be a safety net.
        lp = torch.nan_to_num(lp, nan=hard_min, posinf=0.0, neginf=hard_min)
        lp_leaky = floor + leak * (lp - floor)
        return torch.where(lp >= floor, lp, torch.clamp(lp_leaky, min=hard_min))

    def reset(self) -> None:
        ...
