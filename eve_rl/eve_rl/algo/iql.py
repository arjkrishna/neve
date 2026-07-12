"""Plan v11 Stage 1B — Implicit Q-Learning (Kostrikov et al. 2021).

Pure offline algorithm — no policy sampling in the critic target (so no
overestimation bootstrap), no entropy bonus, and policy improvement via
advantage-weighted regression on the BUFFER actions.

Update order (per step):
    1. _update_v   — expectile regression toward min(target_q1, target_q2)
    2. _update_q   — TD with target V(s') (single V; no min-Q trick here)
    3. _update_policy — AWR: -E[ w * log_pi(a|s) ], w = exp((Q-V)/beta).clamp(max)
    4. Soft-update target_q1/target_q2 with Polyak coefficient self.tau
"""

from typing import Tuple, Dict, List, Optional
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .algo import Algo, AlgoPlayOnly
from ..model import IQLModel, IQLModelPlayOnly
from ..replaybuffer import Batch


def compute_grad_norm(parameters) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def _is_nonfinite(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)


class IQLPlayOnly(AlgoPlayOnly):
    model: IQLModelPlayOnly

    def __init__(
        self,
        model: IQLModelPlayOnly,
        n_actions: int,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
    ):
        self.logger = logging.getLogger(self.__module__)
        self.n_actions = n_actions
        self.exploration_action_noise = exploration_action_noise
        self.model = model

        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval
        self.device = torch.device("cpu")

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            action = torch.tanh(Normal(mean, std).sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            # Per-dim noise (a size-less np.random.normal returns ONE scalar
            # shared by all action dims — perfectly correlated exploration);
            # clip so the buffer never stores actions outside the tanh
            # domain [-1, 1].
            action = action + np.random.normal(
                0.0, self.exploration_action_noise, size=action.shape
            )
            action = np.clip(action, -1.0, 1.0)
        return action

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                action = torch.tanh(Normal(mean, std).sample())
            else:
                action = torch.tanh(mean)
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling

    def to(self, device: torch.device):
        super().to(device)
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()


class IQL(Algo):
    """IQL — Implicit Q-Learning (Kostrikov 2021)."""

    model: IQLModel

    def __init__(
        self,
        model: IQLModel,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scaling: float = 1,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
        grad_clip: float = 0.3,
        iql_expectile_tau: float = 0.7,
        iql_beta: float = 3.0,
        iql_awr_max: float = 5.0,
        offline_mode: bool = True,
        # FIX 1 (RL_IMPROV_8 KL-anchor iteration) — KL-to-warmstart
        # penalty. When alpha_kl > 0 AND kl_warmstart_state_dict is
        # provided, the policy loss includes
        #   loss += alpha_kl * mean_batch( KL( pi_current(s) || pi_warm(s) ) )
        # with the closed-form Gaussian KL between (mean, log_std) tuples.
        # Default alpha_kl=0.0 -> no behavioral change (back-compat).
        alpha_kl: float = 0.0,
        kl_warmstart_state_dict: Optional[Dict[str, "torch.Tensor"]] = None,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.gamma = gamma
        # `self.tau` = Polyak coefficient for target_q soft updates (same
        # semantics as SAC.tau). NOT the expectile parameter — see
        # `self.iql_expectile_tau` below.
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
        self.grad_clip = grad_clip
        # IQL-specific hyperparameters.
        self.iql_expectile_tau = float(iql_expectile_tau)  # expectile level (0.5=mean, 0.7..0.9=upper)
        self.iql_beta = float(iql_beta)                    # AWR temperature
        self.iql_awr_max = float(iql_awr_max)              # AWR weight cap (5.0 per workflow synthesis)
        # Tag for downstream tooling that branches on algo identity.
        self.algo = "iql"
        self.offline_mode = bool(offline_mode)

        # Model
        self.model = model

        self.reward_scaling = reward_scaling
        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval

        self.device = torch.device("cpu")
        self.update_step = 0

        # FIX 1 — KL-to-warmstart anchor state. The frozen reference policy
        # is built lazily on first call to _update_policy from the supplied
        # state_dict (must match the architecture of self.model.policy).
        # Storing the state_dict (not the network) keeps __init__ device-
        # agnostic and survives the deepcopy(self.algo) -> trainer-subprocess
        # handoff cleanly. self._kl_warmstart_policy is materialized once,
        # frozen (requires_grad=False), and put in eval mode.
        self.alpha_kl = float(alpha_kl)
        # Public attribute name (no underscore) so eve_rl's confighandler can
        # introspect it via getattr(self, "kl_warmstart_state_dict") when
        # serializing the algo config to .everl. It is cleared to None
        # IMMEDIATELY below (eager materialization, not lazy) so neither the
        # initial config.yaml save NOR the post-update .everl save trip the
        # confighandler "Handling this class <class 'torch.Tensor'>" error.
        # Lazy materialization was the original design, but the agent is
        # wrapped in BenchAgentSynchron (multiprocess) — the parent-process
        # IQL never executes _update_policy, so its kl_warmstart_state_dict
        # would never get cleared, and save_checkpoint runs in the parent.
        self.kl_warmstart_state_dict = kl_warmstart_state_dict
        self._kl_warmstart_policy = None
        # Eager materialization: deepcopy self.model.policy + load state_dict
        # + freeze, RIGHT NOW. self.model is on CPU at this point; the frozen
        # policy will be moved to the live device by self.to(device) later.
        if self.alpha_kl > 0.0 and self.kl_warmstart_state_dict is not None:
            self._ensure_warmstart_policy()

        # Diagnostics
        self.last_metrics: Dict[str, float] = {}
        self.last_td_errors = None

        # NaN/Inf sentinel counters
        self._nonfinite_q_loss_count = 0
        self._nonfinite_v_loss_count = 0
        self._nonfinite_policy_loss_count = 0
        self._nonfinite_grad_count = 0

        # Per-update diagnostic state (populated by _update_*; surfaced in
        # last_metrics at the end of update()).
        self._grad_norm_q1 = 0.0
        self._grad_norm_q2 = 0.0
        self._grad_norm_v = 0.0
        self._grad_norm_policy = 0.0
        self._q1_mean = torch.zeros(1)
        self._q2_mean = torch.zeros(1)
        self._v_mean = torch.zeros(1)
        self._adv_mean = 0.0
        self._adv_std = 0.0
        self._awr_weight_max = 0.0
        self._awr_weight_mean = 0.0
        self._awr_weight_saturation = 0.0
        # FIX 1 — KL diagnostics surfaced through last_metrics.
        self._kl_to_warmstart = 0.0

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #
    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            action = torch.tanh(Normal(mean, std).sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            # Per-dim noise (a size-less np.random.normal returns ONE scalar
            # shared by all action dims — perfectly correlated exploration);
            # clip so the buffer never stores actions outside the tanh
            # domain [-1, 1].
            action = action + np.random.normal(
                0.0, self.exploration_action_noise, size=action.shape
            )
            action = np.clip(action, -1.0, 1.0)
        return action

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                action = torch.tanh(Normal(mean, std).sample())
            else:
                action = torch.tanh(mean)
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling

    # ------------------------------------------------------------------ #
    # Update                                                             #
    # ------------------------------------------------------------------ #
    def update(self, batch: Batch) -> List[float]:
        all_states = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        dones = batch.terminals
        padding_mask = batch.padding_mask
        is_weights = batch.is_weights

        all_states = all_states.to(dtype=torch.float32, device=self.device)
        actions = actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.float32, device=self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.float32, device=self.device)
        if is_weights is not None:
            is_weights = is_weights.to(
                dtype=torch.float32, device=self.device
            ).view(-1, 1, 1)

        seq_length = actions.shape[1]
        states = torch.narrow(all_states, dim=1, start=0, length=seq_length)
        next_states = torch.narrow(all_states, dim=1, start=1, length=seq_length)

        # (1) Update V — expectile regression toward target Q on buffer actions.
        v_loss = self._update_v(states, actions, padding_mask, is_weights)

        # (2) Update Q1 / Q2 — TD with V(s') as bootstrap.
        expected_q = self._get_expected_q(
            next_states, rewards, dones, padding_mask
        )
        q1_loss, td_errors = self._update_q1(
            actions, padding_mask, states, expected_q, is_weights
        )
        q2_loss = self._update_q2(
            actions, padding_mask, states, expected_q, is_weights
        )

        # (3) Update policy — AWR on buffer actions.
        policy_loss = self._update_policy(
            actions, padding_mask, states, is_weights
        )

        # (4) Soft-update target Q.
        self.model.update_target_q(self.tau)

        self.last_td_errors = td_errors
        self.update_step += 1

        # NaN/Inf sentinels
        v_loss_val = v_loss.detach().cpu().item()
        q1_loss_val = q1_loss.detach().cpu().item()
        q2_loss_val = q2_loss.detach().cpu().item()
        policy_loss_val = policy_loss.detach().cpu().item()
        if _is_nonfinite(q1_loss_val) or _is_nonfinite(q2_loss_val):
            self._nonfinite_q_loss_count += 1
            self.logger.warning(
                f"IQL: NaN/Inf in Q loss at step {self.update_step} "
                f"(count={self._nonfinite_q_loss_count})"
            )
        if _is_nonfinite(v_loss_val):
            self._nonfinite_v_loss_count += 1
            self.logger.warning(
                f"IQL: NaN/Inf in V loss at step {self.update_step} "
                f"(count={self._nonfinite_v_loss_count})"
            )
        if _is_nonfinite(policy_loss_val):
            self._nonfinite_policy_loss_count += 1
            self.logger.warning(
                f"IQL: NaN/Inf in policy loss at step {self.update_step} "
                f"(count={self._nonfinite_policy_loss_count})"
            )
        if (
            _is_nonfinite(self._grad_norm_q1)
            or _is_nonfinite(self._grad_norm_q2)
            or _is_nonfinite(self._grad_norm_v)
            or _is_nonfinite(self._grad_norm_policy)
        ):
            self._nonfinite_grad_count += 1

        lr_policy = 0.0
        if self.model.policy_scheduler is not None:
            try:
                lr_policy = self.model.policy_scheduler.get_last_lr()[0]
            except Exception:
                lr_policy = 0.0

        self.last_metrics = {
            "v_loss": v_loss_val,
            "q1_mean": self._q1_mean.cpu().item(),
            "q2_mean": self._q2_mean.cpu().item(),
            "v_mean": self._v_mean.cpu().item(),
            "target_q_mean": expected_q.mean().detach().cpu().item(),
            "adv_mean": float(self._adv_mean),
            "adv_std": float(self._adv_std),
            "awr_weight_mean": float(self._awr_weight_mean),
            "awr_weight_max": float(self._awr_weight_max),
            "awr_weight_saturation": float(self._awr_weight_saturation),
            # FIX 1 — KL anchor diagnostics (0 if alpha_kl == 0).
            "kl_to_warmstart": float(self._kl_to_warmstart),
            "alpha_kl": float(self.alpha_kl),
            "grad_norm_q1": self._grad_norm_q1,
            "grad_norm_q2": self._grad_norm_q2,
            "grad_norm_v": self._grad_norm_v,
            "grad_norm_policy": self._grad_norm_policy,
            "lr_policy": lr_policy,
            "iql_expectile_tau": self.iql_expectile_tau,
            "iql_beta": self.iql_beta,
            "iql_awr_max": self.iql_awr_max,
            "nonfinite_q_loss_count": self._nonfinite_q_loss_count,
            "nonfinite_v_loss_count": self._nonfinite_v_loss_count,
            "nonfinite_policy_loss_count": self._nonfinite_policy_loss_count,
            "nonfinite_grad_count": self._nonfinite_grad_count,
        }

        return [
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
        ]

    # ------------------------------------------------------------------ #
    # Inner update steps                                                 #
    # ------------------------------------------------------------------ #
    def _update_v(self, states, actions, padding_mask, is_weights):
        """Expectile regression: V(s) toward min(target_q1, target_q2)(s,a)."""
        with torch.no_grad():
            tq1 = self.model.target_q1(states, actions)
            tq2 = self.model.target_q2(states, actions)
            target = torch.min(tq1, tq2)

        v = self.model.v(states)
        diff = target - v  # u = target - V; expectile regression weight asymmetric

        # Expectile weight: |tau - I(diff < 0)|. tau in (0.5, 1.0) penalizes
        # underestimation more than overestimation → V tracks upper expectile.
        weight = torch.where(
            diff < 0,
            torch.full_like(diff, 1.0 - self.iql_expectile_tau),
            torch.full_like(diff, self.iql_expectile_tau),
        )
        per_sample = weight * diff.pow(2)

        if padding_mask is not None:
            per_sample = per_sample * padding_mask
        if is_weights is not None:
            v_loss = (is_weights * per_sample).mean()
        else:
            v_loss = per_sample.mean()

        self.model.v_optimizer.zero_grad()
        v_loss.backward()
        self._grad_norm_v = compute_grad_norm(self.model.v.parameters())
        self._v_mean = v.mean().detach()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.v.parameters(), self.grad_clip
            )
        self.model.v_optimizer.step()
        if self.model.v_scheduler:
            self.model.v_scheduler.step()
        return v_loss

    def _get_expected_q(self, next_states, rewards, dones, padding_mask):
        """IQL critic target: r * reward_scaling + gamma * (1-done) * V(s')."""
        with torch.no_grad():
            next_v = self.model.v(next_states)
        # reward_scaling scales the raw reward in the Bellman target
        # (identity at the default 1.0, which current configs use).
        expected_q = (
            rewards * self.reward_scaling
            + (1 - dones) * self.gamma * next_v
        )
        if padding_mask is not None:
            expected_q = expected_q * padding_mask
        return expected_q

    def _update_q1(self, actions, padding_mask, states, expected_q, is_weights):
        curr_q1 = self.model.q1(states, actions)
        if padding_mask is not None:
            curr_q1 = curr_q1 * padding_mask
        td = curr_q1 - expected_q.detach()
        if is_weights is not None:
            q1_loss = (is_weights * td.pow(2)).mean()
        else:
            q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        td_errors = (
            td.detach().abs().view(curr_q1.shape[0], -1).mean(dim=1).cpu().numpy()
        )

        self.model.q1_optimizer.zero_grad()
        q1_loss.backward()
        self._grad_norm_q1 = compute_grad_norm(self.model.q1.parameters())
        self._q1_mean = curr_q1.mean().detach()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.q1.parameters(), self.grad_clip
            )
        self.model.q1_optimizer.step()
        if self.model.q1_scheduler:
            self.model.q1_scheduler.step()
        return q1_loss, td_errors

    def _update_q2(self, actions, padding_mask, states, expected_q, is_weights):
        curr_q2 = self.model.q2(states, actions)
        if padding_mask is not None:
            curr_q2 = curr_q2 * padding_mask
        if is_weights is not None:
            td = curr_q2 - expected_q.detach()
            q2_loss = (is_weights * td.pow(2)).mean()
        else:
            q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.model.q2_optimizer.zero_grad()
        q2_loss.backward()
        self._grad_norm_q2 = compute_grad_norm(self.model.q2.parameters())
        self._q2_mean = curr_q2.mean().detach()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.q2.parameters(), self.grad_clip
            )
        self.model.q2_optimizer.step()
        if self.model.q2_scheduler:
            self.model.q2_scheduler.step()
        return q2_loss

    def _update_policy(self, actions, padding_mask, states, is_weights):
        """AWR: -E[ exp((Q(s,a) - V(s)) / beta).clamp(max=awr_max) * log_pi(a|s) ].

        FIX 1 (KL anchor) — when alpha_kl > 0, adds
            alpha_kl * mean( KL( N(mu_cur, sd_cur) || N(mu_warm, sd_warm) ) )
        per (state, action-dim) slot. Closed-form Gaussian KL on the PRE-tanh
        (mean, log_std) tuples returned by GaussianPolicy.forward. The KL is
        computed over states drawn from the SAME minibatch already used for
        AWR (no extra buffer sample), keeping per-step cost ~O(batch * dim).
        """
        with torch.no_grad():
            q1_buf = self.model.q1(states, actions)
            q2_buf = self.model.q2(states, actions)
            q_buf = torch.min(q1_buf, q2_buf)
            v_s = self.model.v(states)
            advantage = q_buf - v_s
            weight = torch.exp(advantage / self.iql_beta).clamp(max=self.iql_awr_max)
            # Diagnostics
            self._adv_mean = float(advantage.mean().item())
            self._adv_std = float(advantage.std().item())
            self._awr_weight_mean = float(weight.mean().item())
            self._awr_weight_max = float(weight.max().item())
            # Saturation: fraction of weights at or near the cap (Plan v11 audit #1).
            self._awr_weight_saturation = float(
                (weight >= 0.95 * self.iql_awr_max).float().mean().item()
            )

        logp = self.model.policy.log_prob(states, actions)
        if padding_mask is not None:
            logp = logp * padding_mask
            weight = weight * padding_mask
        per_sample_loss = -(weight * logp)

        if is_weights is not None:
            policy_loss = (is_weights * per_sample_loss).mean()
        else:
            policy_loss = per_sample_loss.mean()

        # FIX 1 — KL-to-warmstart anchor.
        if self.alpha_kl > 0.0 and (
            self.kl_warmstart_state_dict is not None
            or self._kl_warmstart_policy is not None
        ):
            kl_term = self._compute_kl_to_warmstart(states, padding_mask)
            self._kl_to_warmstart = float(kl_term.detach().cpu().item())
            policy_loss = policy_loss + self.alpha_kl * kl_term
        else:
            self._kl_to_warmstart = 0.0

        self.model.policy_optimizer.zero_grad()
        policy_loss.backward()
        self._grad_norm_policy = compute_grad_norm(self.model.policy.parameters())
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.policy.parameters(), self.grad_clip
            )
        self.model.policy_optimizer.step()
        if self.model.policy_scheduler:
            self.model.policy_scheduler.step()
        return policy_loss

    # ------------------------------------------------------------------ #
    # FIX 1 — KL-to-warmstart helpers                                    #
    # ------------------------------------------------------------------ #
    def _ensure_warmstart_policy(self) -> None:
        """Lazily materialize a frozen reference policy on the current device.

        Called on the FIRST training step that needs it. Uses
        copy.deepcopy(self.model.policy) so the architecture, log_std clamp,
        and embedder wiring exactly match the trainable policy. Loads the
        warm-start state_dict and freezes all parameters.
        """
        if self._kl_warmstart_policy is not None:
            return
        if self.kl_warmstart_state_dict is None:
            return
        # Deepcopy gives us a fresh module on the same device as the live
        # policy. Tolerate device-mismatch in the state_dict via strict=False
        # / explicit load — the loaded tensors are mapped via PyTorch's
        # load_state_dict semantics (device of the receiving Parameter is
        # preserved).
        import copy as _copy
        ref = _copy.deepcopy(self.model.policy)
        try:
            ref.load_state_dict(self.kl_warmstart_state_dict)
        except RuntimeError as e:
            self.logger.error(
                "IQL FIX 1 — failed to load kl_warmstart_state_dict into "
                "deepcopied policy (architecture mismatch?): %s. KL anchor "
                "disabled for this run.",
                e,
            )
            self.alpha_kl = 0.0
            self.kl_warmstart_state_dict = None
            return
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        # Move to the live policy's device. GaussianPolicy.device proxies
        # body.device, which is set by the first forward; fall back to the
        # algo's device attribute otherwise.
        try:
            ref.to(self.model.policy.device)
        except Exception:
            ref.to(self.device)
        self._kl_warmstart_policy = ref
        # Free the state_dict reference now that the frozen policy holds the
        # tensors. Future checkpoint serializations capture None, not the
        # multi-MB tensor blob.
        self.kl_warmstart_state_dict = None

    def _compute_kl_to_warmstart(self, states, padding_mask):
        """Closed-form KL( N(mu_cur, sd_cur) || N(mu_warm, sd_warm) ) per dim.

        Returns a scalar tensor with grad attached to mu_cur / log_std_cur
        (the trainable policy parameters). The warm reference is detached.
        For independent Gaussians,
            KL = log(sd_warm/sd_cur) + (sd_cur^2 + (mu_cur - mu_warm)^2)
                                        / (2 * sd_warm^2) - 0.5
        We sum over action dim (consistent with log_prob keepdim convention)
        and mean over batch * (1, seq_len, 1) — masked by padding_mask if
        present.
        """
        self._ensure_warmstart_policy()
        if self._kl_warmstart_policy is None:
            return torch.zeros((), device=states.device)

        mean_cur, log_std_cur = self.model.policy.forward(states)
        with torch.no_grad():
            mean_warm, log_std_warm = self._kl_warmstart_policy.forward(states)

        var_cur = (2.0 * log_std_cur).exp()
        var_warm = (2.0 * log_std_warm).exp()
        kl_per_dim = (
            log_std_warm
            - log_std_cur
            + (var_cur + (mean_cur - mean_warm).pow(2)) / (2.0 * var_warm)
            - 0.5
        )
        # Sum over the action dim; keep batch / seq dims for masking.
        kl_summed = kl_per_dim.sum(dim=-1, keepdim=True)
        if padding_mask is not None:
            kl_summed = kl_summed * padding_mask
            denom = padding_mask.sum().clamp(min=1.0)
            return kl_summed.sum() / denom
        return kl_summed.mean()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def to(self, device: torch.device):
        super().to(device)
        self.model.to(device)
        # FIX 1 — keep the frozen reference policy co-resident with the
        # trainable one whenever the parent algo is moved.
        if self._kl_warmstart_policy is not None:
            try:
                self._kl_warmstart_policy.to(device)
            except Exception:
                pass

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()

    def to_play_only(self):
        return IQLPlayOnly(
            self.model.to_play_only(),
            self.n_actions,
            self.action_scaling,
            self.exploration_action_noise,
        )
