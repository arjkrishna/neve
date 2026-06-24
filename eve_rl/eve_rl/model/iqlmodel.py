"""Plan v11 Stage 1B — IQLModel (twin Q + single V + Gaussian policy).

Mirrors `SACModel` but:
  * Adds a single V-network (`v`) trained by expectile regression.
  * No entropy temperature (`log_alpha` / `alpha_optimizer` removed).
  * No target V (V is updated every step; target is the deepcopy'd twin
    Q nets only, per the IQL paper).
"""

from copy import deepcopy
from typing import Any, Dict, Iterator

import torch

from .model import Model, ModelPlayOnly
from .. import network


class IQLModelPlayOnly(ModelPlayOnly):
    def __init__(self, policy: network.GaussianPolicy) -> None:
        self.policy = policy

    def to(self, device: torch.device):
        super().to(device)
        self.policy.to(device)

    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dicts["policy"])

    def reset(self) -> None:
        self.policy.reset()

    def close(self):
        del self.policy


class IQLModel(Model):
    def __init__(
        self,
        q1: network.QNetwork,
        q2: network.QNetwork,
        v: "network.VNetwork",
        policy: network.GaussianPolicy,
        q1_optimizer: torch.optim.Optimizer,
        q2_optimizer: torch.optim.Optimizer,
        v_optimizer: torch.optim.Optimizer,
        policy_optimizer: torch.optim.Optimizer,
        q1_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        q2_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        v_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        policy_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        self.q1 = q1
        self.q2 = q2
        self.v = v
        self.policy = policy

        self.q1_optimizer = q1_optimizer
        self.q2_optimizer = q2_optimizer
        self.v_optimizer = v_optimizer
        self.policy_optimizer = policy_optimizer

        self.q1_scheduler = q1_scheduler
        self.q2_scheduler = q2_scheduler
        self.v_scheduler = v_scheduler
        self.policy_scheduler = policy_scheduler

        self.target_q1 = deepcopy(self.q1)
        self.target_q1.eval()
        self.target_q2 = deepcopy(self.q2)
        self.target_q2.eval()

    def to(self, device: torch.device):
        super().to(device)
        self.target_q1.to(device)
        self.target_q2.to(device)

        for net, optimizer in zip(
            [self.q1, self.q2, self.v, self.policy],
            [
                self.q1_optimizer,
                self.q2_optimizer,
                self.v_optimizer,
                self.policy_optimizer,
            ],
        ):
            old_params = net.parameters()
            net.to(device)
            new_params = net.parameters()

            for param_group in optimizer.param_groups:
                optim_ids = [id(param) for param in param_group["params"]]
                for old_param, new_param in zip(old_params, new_params):
                    if id(old_param) in optim_ids:
                        idx = optim_ids.index(id(old_param))
                        param_group["params"][idx] = new_param

    def update_target_q(self, tau):
        for target_param, param in zip(
            self.target_q1.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(
            self.target_q2.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def reset(self) -> None:
        for net in self:
            net.reset()

    def __iter__(self) -> Iterator[network.Network]:
        return iter(
            [self.q1, self.q2, self.target_q1, self.target_q2, self.v, self.policy]
        )

    def close(self):
        del self.q1
        del self.q1_optimizer
        del self.q2
        del self.q2_optimizer
        del self.v
        del self.v_optimizer
        del self.policy
        del self.policy_optimizer

    def state_dicts_network(
        self, destination: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        ret = state_dicts = {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "v": self.v.state_dict(),
            "policy": self.policy.state_dict(),
        }

        if destination is not None:
            for net in ["q1", "q2", "target_q1", "target_q2", "v", "policy"]:
                state_dict = state_dicts[net]
                dest = destination[net]

                for tensor, dest_tensor in zip(state_dict.values(), dest.values()):
                    dest_tensor.copy_(tensor)
            ret = destination

        return ret

    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        # Plan v11 Stage 1B — partial-load tolerance for SAC→IQL warm-start.
        # The v3 .everl checkpoint was saved by an SAC trainer; it contains
        # q1/q2/target_q1/target_q2/policy/log_alpha but NO v entry. Load
        # what we can; the v-network stays at its fresh initialization (it
        # has to: there is no pre-trained V to inherit from SAC). log_alpha
        # is silently ignored — IQL has no entropy temperature.
        if "q1" in state_dicts:
            self.q1.load_state_dict(state_dicts["q1"])
        if "q2" in state_dicts:
            self.q2.load_state_dict(state_dicts["q2"])
        if "target_q1" in state_dicts:
            self.target_q1.load_state_dict(state_dicts["target_q1"])
        if "target_q2" in state_dicts:
            self.target_q2.load_state_dict(state_dicts["target_q2"])
        if "v" in state_dicts:
            self.v.load_state_dict(state_dicts["v"])
        if "policy" in state_dicts:
            self.policy.load_state_dict(state_dicts["policy"])

    def to_play_only(self) -> IQLModelPlayOnly:
        policy = deepcopy(self.policy)
        policy.eval()
        return IQLModelPlayOnly(policy)

    def state_dicts_optimizer(self) -> Dict[str, Any]:
        ret = {
            "q1_optim": self.q1_optimizer.state_dict(),
            "q2_optim": self.q2_optimizer.state_dict(),
            "v_optim": self.v_optimizer.state_dict(),
            "policy_optim": self.policy_optimizer.state_dict(),
        }
        return ret

    def load_state_dicts_optimizer(self, state_dicts: Dict[str, Any]) -> None:
        if "q1_optim" in state_dicts:
            self.q1_optimizer.load_state_dict(state_dicts["q1_optim"])
        if "q2_optim" in state_dicts:
            self.q2_optimizer.load_state_dict(state_dicts["q2_optim"])
        if "v_optim" in state_dicts:
            self.v_optimizer.load_state_dict(state_dicts["v_optim"])
        if "policy_optim" in state_dicts:
            self.policy_optimizer.load_state_dict(state_dicts["policy_optim"])

    def state_dicts_scheduler(self) -> Dict[str, Any]:
        if (
            self.q1_scheduler is None
            and self.q2_scheduler is None
            and self.v_scheduler is None
            and self.policy_scheduler is None
        ):
            return {}
        ret = {
            "q1_schedule": self.q1_scheduler.state_dict()
            if self.q1_scheduler is not None
            else {},
            "q2_schedule": self.q2_scheduler.state_dict()
            if self.q2_scheduler is not None
            else {},
            "v_schedule": self.v_scheduler.state_dict()
            if self.v_scheduler is not None
            else {},
            "policy_schedule": self.policy_scheduler.state_dict()
            if self.policy_scheduler is not None
            else {},
        }
        return ret

    def load_state_dicts_scheduler(self, state_dicts: Dict[str, Any]) -> None:
        if self.q1_scheduler is not None and "q1_schedule" in state_dicts:
            self.q1_scheduler.load_state_dict(state_dicts["q1_schedule"])
        if self.q2_scheduler is not None and "q2_schedule" in state_dicts:
            self.q2_scheduler.load_state_dict(state_dicts["q2_schedule"])
        if self.v_scheduler is not None and "v_schedule" in state_dicts:
            self.v_scheduler.load_state_dict(state_dicts["v_schedule"])
        if self.policy_scheduler is not None and "policy_schedule" in state_dicts:
            self.policy_scheduler.load_state_dict(state_dicts["policy_schedule"])
