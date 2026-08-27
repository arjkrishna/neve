from typing import Tuple, Dict, Any, List, Optional
import logging
import math
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn.functional as F
from .algo import Algo, AlgoPlayOnly
from ..model import SACModel, SACModelPlayOnly
from ..replaybuffer import Batch


def compute_grad_norm(parameters) -> float:
    """Compute the total gradient norm for a set of parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def _is_nonfinite(x: float) -> bool:
    """Check if a value is NaN or Inf."""
    return math.isnan(x) or math.isinf(x)


def _compute_clamp_fraction(actions: torch.Tensor, threshold: float = 0.99) -> float:
    """
    Compute fraction of actions that are near the clamp boundaries.
    Actions are assumed to be in [-1, 1] after tanh.
    
    Args:
        actions: Tensor of actions (any shape)
        threshold: Actions with |value| > threshold are considered clamped
        
    Returns:
        Fraction of action components that are clamped (0.0 to 1.0)
    """
    if actions.numel() == 0:
        return 0.0
    clamped = (actions.abs() > threshold).float()
    return clamped.mean().item()


class SACPlayOnly(AlgoPlayOnly):
    model: SACModelPlayOnly

    def __init__(
        self,
        model: SACModelPlayOnly,
        n_actions: int,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.exploration_action_noise = exploration_action_noise
        # Model
        self.model = model

        # REST
        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval

        self.device = torch.device("cpu")

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
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

    def get_action_exploration(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            # Per-dim noise + clip to the tanh domain [-1, 1] (see
            # get_exploration_action); only the action is clipped — mean and
            # log_std are returned raw.
            action = action + np.random.normal(
                0.0, self.exploration_action_noise, size=action.shape
            )
            action = np.clip(action, -1.0, 1.0)
        return action, mean, log_std

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling
    
    def get_action_evaluation(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling, mean, log_std

    def to(self, device: torch.device):
        super().to(device)
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()


class SAC(Algo):
    model: SACModel

    def __init__(
        self,
        model: SACModel,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scaling: float = 1,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
        stochastic_eval: bool = False,
        grad_clip: float = 0.0,
        algo: str = "sac",
        awac_lambda: float = 3.0,
        # RL_IMPROV_16 E1b — batch-normalized AWAC advantages. The v2
        # investigation showed adv std ~0.09 with lambda=1.0 gives weights
        # spanning only [0.72, 1.25] ~= uniform BC: the critic's correctly
        # signed retract-when-stuck preference (weight edge ~10% in the
        # buckled-slack tail) cannot transmit to the policy. When tau > 0,
        # weight = exp((adv / adv.std()) / tau).clamp(20) — self-calibrating
        # as the critic sharpens (a fixed lambda drifts as adv std grows).
        # tau=2.0 targets a p99/p1 weight ratio ~e^2.5 ~= 12 (band 5-20x).
        # 0.0 = OFF (legacy exp(adv/lambda) path, byte-identical).
        awac_adv_norm_tau: float = 0.0,
        entropy_beta_per_dim: Optional[List[float]] = None,
        action_mean_penalty: float = 0.0,
        awac_mode_adv_norm: bool = False,
        awac_contact_thresh: float = 0.009,
        awac_contact_idx: int = 103,
        contact_mean_penalty: float = 0.0,
        q_target_floor: float = None,
        offline_mode: bool = False,
        target_entropy: Optional[float] = None,
        # RL_IMPROV_15 collapse forensics — configurable log_alpha rails.
        # The v1 freeze: alpha decayed to the -10 floor over 165k updates
        # (entropy term vanished -> entropy ground down), then whipsawed
        # back to 0.45; with log_std pinned at its CEILING the entropy
        # controller's only remaining lever was crushing the action MEAN
        # toward 0 (the freeze). A higher floor keeps the entropy term
        # alive (no deep decay -> no whipsaw); a low ceiling caps alpha so
        # the entropy term can never dominate the BC/advantage term and
        # mean-crush again. Defaults preserve the old (-10, 2) behavior.
        log_alpha_min: float = -10.0,
        log_alpha_max: float = 2.0,
        # Gen-4 — auxiliary privileged-label supervision on the policy.
        # aux_label_indices are ABSOLUTE flat-obs indices (into the
        # privileged tail beyond the policy's input slice); the policy
        # predicts them from its deployable prefix via its n_aux head and
        # an MSE term (weight aux_coef) is added to the policy loss —
        # representation shaping toward inferring contact/buckle state
        # from deployable signals. aux_coef 0.0 = off (byte-identical).
        aux_coef: float = 0.0,
        aux_label_indices: Optional[List[int]] = None,
        # RL_IMPROV_16 E2.2 — z-score the aux labels AT LOSS TIME (running
        # EMA mean/var per label). The v2 contact labels have std ~1e-3 in
        # normalized units, so aux_coef*MSE ~= 5e-8 — zero shaping pressure
        # on the shared trunk. Loss-time normalization fixes the gradient
        # scale WITHOUT touching the stored obs (cache-compatible; an
        # obs-side rescale would silently mix two obs versions in the
        # buffer). False = OFF (legacy raw-MSE, byte-identical).
        aux_label_znorm: bool = False,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
        # Plan v8 — stabilization knobs.
        #   grad_clip   : max global grad-norm for critic+policy (0 = off).
        #   algo        : "sac" (default) or "awac" (advantage-weighted
        #                 policy update — stable on demo-seeded buffers).
        #   awac_lambda : AWAC advantage temperature.
        self.grad_clip = grad_clip
        self.algo = algo
        self.awac_lambda = awac_lambda
        # RL_IMPROV_16 E1b/E2.2 (see __init__ docnotes).
        self.awac_adv_norm_tau = float(awac_adv_norm_tau)
        self.aux_label_znorm = bool(aux_label_znorm)
        self._aux_znorm_mu = None   # running EMA mean per aux label
        self._aux_znorm_var = None  # running EMA var per aux label
        self._aux_loss_value = 0.0
        self._awac_weight_p99p1 = 0.0
        # Plan v11 Stage 1 — scaffold per-dim entropy bonus + mean-margin
        # penalty (Direction 1, applied in Stage 2 to fight the cath_trans
        # mean-rail collapse). Default all zero → no-op in Stage 1 / SAC /
        # AWAC. When `entropy_beta_per_dim[i] > 0`, _update_policy adds
        # `-beta_i * log_std_i.mean()` to the policy loss (encourages
        # larger std on that action dim). `action_mean_penalty > 0`
        # adds `+coef * mean(|atanh(action_mean.clamp(±0.99))|)` — direct
        # rail-saturation guard (risk audit #3 HIGH).
        self.entropy_beta_per_dim = (
            list(entropy_beta_per_dim) if entropy_beta_per_dim else None
        )
        self.action_mean_penalty = float(action_mean_penalty)
        # RL_IMPROV_16 (v3c) — default-OFF trainer levers, CLI-gated so the
        # reward changes can be measured in isolation first (both attrs
        # must mirror __init__ param names for the ConfigHandler getattr
        # round-trip):
        #   awac_mode_adv_norm — normalize E1b advantages PER CONTACT MODE
        #     (contact = states[..., awac_contact_idx] > awac_contact_thresh,
        #     ground-truth privileged contact, zero inference error). The
        #     v3a forensic measured corr(adv, action) flat (0.01) inside the
        #     contact stratum under GLOBAL normalization — per-mode std
        #     re-sharpens ranking where success/failure actually differ.
        #   contact_mean_penalty — contact-gated anti-rail: contact-state
        #     samples pay this mean-margin penalty instead of the global
        #     action_mean_penalty (v3a measured |tanh(mu)|>0.99 on 86% of
        #     contact states vs 0% free — the rail is mode-specific, so the
        #     penalty should be too). 0.0 keeps the legacy global path.
        self.awac_mode_adv_norm = bool(awac_mode_adv_norm)
        self.awac_contact_thresh = float(awac_contact_thresh)
        self.awac_contact_idx = int(awac_contact_idx)
        self.contact_mean_penalty = float(contact_mean_penalty)
        # RL_IMPROV_16 (v3c3) — Bellman-target floor clip, default OFF (None
        # = byte-identical legacy). v3c2's critic diverged super-linearly to
        # q1 ~ -90 while honest buffer returns bottomed at ~-7 (bulk p1
        # -2.9): min-double-Q bootstrapping on the railed policy's OOD
        # next-actions manufactured ever-lower targets in lockstep
        # (target_q == q1 all the way down). Clamping the TARGET at a floor
        # set BELOW any honest return (e.g. -10) never binds on real data
        # but severs the self-feeding spiral — the same guard family as the
        # AWAC target's removed entropy term (the prior -406k divergence).
        # Complement, not substitute, for contact_mean_penalty: the penalty
        # keeps next-actions in-distribution (steering), the floor bounds
        # the target against any residual runaway (airbag).
        self.q_target_floor = (
            float(q_target_floor) if q_target_floor is not None else None
        )
        # RL_IMPROV_15 — log_alpha clamp rails (see __init__ docnote).
        self.log_alpha_min = float(log_alpha_min)
        self.log_alpha_max = float(log_alpha_max)
        if self.log_alpha_min > self.log_alpha_max:
            raise ValueError(
                f"log_alpha_min ({self.log_alpha_min}) > log_alpha_max "
                f"({self.log_alpha_max})"
            )
        # Plan v11 Stage 1 — offline_mode flag. Surfaces in update() so
        # the trainer can route around any online-only telemetry hooks.
        # No behavior change today; reserved for hooks the IQL/CQL/BC
        # branches will introduce.
        self.offline_mode = bool(offline_mode)
        self.aux_coef = float(aux_coef)
        self.aux_label_indices = (
            list(aux_label_indices) if aux_label_indices else None
        )
        # Model
        self.model = model

        # REST
        self.reward_scaling = reward_scaling
        self.action_scaling = action_scaling
        self.stochastic_eval = stochastic_eval

        self.device = torch.device("cpu")
        self.update_step = 0

        # ENTROPY TEMPERATURE
        self.alpha = torch.ones(1)
        # Entropy setpoint for the auto-alpha regulator. With a tanh-Gaussian
        # whose healthy operating entropy is ~+2.6, the SAC default
        # -n_actions leaves the regulator a huge dead zone: alpha decays to
        # ~0 and only re-engages after the policy is already railed below
        # -4. Exposing the setpoint lets runs hold it near the healthy band
        # (e.g. +1.0 for 4 dims with log_std band (-2, 0)). Stored as a
        # native float — the confighandler getattr's every __init__ param
        # name when serializing to config.yaml/.everl and raises on
        # torch.Tensor; a float broadcasts fine in _update_alpha.
        if target_entropy is not None:
            self.target_entropy = float(target_entropy)
        else:
            self.target_entropy = -float(n_actions)

        # DIAGNOSTICS: Store metrics from last update for logging
        self.last_metrics: Dict[str, float] = {}
        # Plan v7 (PER): per-sample |TD| from the most recent update(), fed
        # back to a PER buffer's update_priorities(). None for uniform runs.
        self.last_td_errors = None
        
        # DIAGNOSTICS: NaN/Inf sentinel counters
        self._nonfinite_q_loss_count = 0
        self._nonfinite_policy_loss_count = 0
        self._nonfinite_grad_count = 0
        self._clamp_fraction = 0.0

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
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
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            if self.stochastic_eval:
                std = log_std.exp()
                normal = Normal(mean, std)
                action = torch.tanh(normal.sample())
            else:
                mean, _ = self.model.policy.forward_play(torch_state)
                action = torch.tanh(mean)

            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling

    def update(self, batch: Batch) -> Tuple[float, float, float]:
        # Plan v7 — `Batch` is a 7-field NamedTuple (PER added `is_weights`
        # and `indices`); access by name, not a positional 5-unpack.
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
        # PER importance-sampling weights — per-sample, broadcast as (B,1,1).
        # None for uniform episode/step batches → all loss paths fall back
        # to the plain `.mean()` and behave exactly as before.
        if is_weights is not None:
            is_weights = is_weights.to(
                dtype=torch.float32, device=self.device
            ).view(-1, 1, 1)

        self.alpha = self.model.log_alpha.exp().detach()  # re-derive from (possibly checkpoint-loaded) log_alpha; init/ctor value may be stale.

        seq_length = actions.shape[1]
        states = torch.narrow(all_states, dim=1, start=0, length=seq_length)

        # use all_states for next_actions and next_log_pi for proper hidden_state initilaization
        expected_q = self._get_expected_q(
            all_states, rewards, dones, padding_mask, seq_length
        )

        # q1 update — also returns per-sample |TD| for PER priority refresh
        q1_loss, td_errors = self._update_q1(
            actions, padding_mask, states, expected_q, is_weights
        )

        # q2 update
        q2_loss = self._update_q2(
            actions, padding_mask, states, expected_q, is_weights
        )

        log_pi, policy_loss = self._update_policy(
            actions, padding_mask, states, is_weights
        )

        # Plan v7 — stash the per-sample |TD| (numpy, CPU) so the agent loop
        # can feed it back to a PER buffer via update_priorities(). Harmless
        # for uniform buffers (they simply never read it).
        self.last_td_errors = td_errors

        self.model.update_target_q(self.tau)

        # Plan v11 anti-rail — auto-tuned entropy is now ALSO active for AWAC
        # (the +alpha*log_pi term in the AWAC policy loss above). alpha adapts to
        # hold the policy at target_entropy (default -n_actions), the principled,
        # data-free replacement for hand-picked per-dim entropy betas. (The old
        # behaviour — AWAC with no entropy term — is what railed both prior runs
        # by update ~2.5k.)
        alpha_loss = self._update_alpha(log_pi, padding_mask)

        self.update_step += 1
        
        # Check for NaN/Inf in losses
        q1_loss_val = q1_loss.detach().cpu().item()
        q2_loss_val = q2_loss.detach().cpu().item()
        policy_loss_val = policy_loss.detach().cpu().item()
        
        nonfinite_q = _is_nonfinite(q1_loss_val) or _is_nonfinite(q2_loss_val)
        nonfinite_pol = _is_nonfinite(policy_loss_val)
        nonfinite_grad = (_is_nonfinite(self._grad_norm_q1) or 
                         _is_nonfinite(self._grad_norm_q2) or 
                         _is_nonfinite(self._grad_norm_policy))
        
        if nonfinite_q:
            self._nonfinite_q_loss_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in Q loss at step {self.update_step} (count={self._nonfinite_q_loss_count})")
        if nonfinite_pol:
            self._nonfinite_policy_loss_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in policy loss at step {self.update_step} (count={self._nonfinite_policy_loss_count})")
        if nonfinite_grad:
            self._nonfinite_grad_count += 1
            self.logger.warning(f"ALERT: NaN/Inf in gradients at step {self.update_step} (count={self._nonfinite_grad_count})")
        
        # Populate diagnostics metrics
        lr_policy = 0.0
        if self.model.policy_scheduler is not None:
            try:
                lr_policy = self.model.policy_scheduler.get_last_lr()[0]
            except Exception:
                lr_policy = 0.0
                
        self.last_metrics = {
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.detach().cpu().item(),
            "log_pi_mean": log_pi.mean().detach().cpu().item(),
            "log_pi_std": log_pi.std().detach().cpu().item(),
            "entropy_proxy": -log_pi.mean().detach().cpu().item(),
            "q1_mean": self._q1_mean.cpu().item(),
            "q2_mean": self._q2_mean.cpu().item(),
            "target_q_mean": expected_q.mean().detach().cpu().item(),
            "min_q_mean": self._min_q_mean.cpu().item(),
            "grad_norm_q1": self._grad_norm_q1,
            "grad_norm_q2": self._grad_norm_q2,
            "grad_norm_policy": self._grad_norm_policy,
            "lr_policy": lr_policy,
            # Fix #6: clamp_fraction for direct saturation detection
            "clamp_fraction": self._clamp_fraction,
            # NEW: Enhanced diagnostics
            "target_entropy": float(self.target_entropy),
            "log_alpha": self.model.log_alpha.item(),
            # NaN/Inf counters
            "nonfinite_q_loss_count": self._nonfinite_q_loss_count,
            "nonfinite_policy_loss_count": self._nonfinite_policy_loss_count,
            "nonfinite_grad_count": self._nonfinite_grad_count,
        }
        # Plan v11 risk audit #1 — surface AWAC weight-saturation
        # diagnostics into the trainer-CSV. Skipped for non-AWAC algos
        # (the instance attrs only exist after the first AWAC update).
        if self.algo == "awac" and hasattr(self, "_awac_weight_saturation"):
            self.last_metrics["awac_weight_saturation"] = (
                self._awac_weight_saturation
            )
            self.last_metrics["awac_weight_max"] = self._awac_weight_max
            self.last_metrics["awac_weight_mean"] = self._awac_weight_mean
            # RL_IMPROV_16 — E1b gate metric + E2.3 aux-loss visibility.
            self.last_metrics["awac_weight_p99p1"] = self._awac_weight_p99p1
            self.last_metrics["aux_loss"] = self._aux_loss_value
        
        return [
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
        ]

    def _update_alpha(self, log_pi, padding_mask=None) -> torch.Tensor:
        delta = (-log_pi - self.target_entropy).detach()
        # log_pi arrives pre-multiplied by padding_mask, so padded entries
        # carry a spurious constant -target_entropy; the masked mean
        # excludes them.
        if padding_mask is not None:
            alpha_loss = (
                self.model.log_alpha * delta * padding_mask
            ).sum() / padding_mask.sum().clamp(min=1.0)
        else:
            alpha_loss = (self.model.log_alpha * delta).mean()
        self.model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.model.alpha_optimizer.step()
        with torch.no_grad():
            # Hard rails; prevents unbounded integration in either direction.
            # RL_IMPROV_15 — bounds are configurable: the v1 (-10, 2) rails
            # allowed the alpha decay->whipsaw->mean-crush freeze cycle.
            self.model.log_alpha.clamp_(self.log_alpha_min, self.log_alpha_max)

        self.alpha = self.model.log_alpha.exp().detach()
        return alpha_loss

    def _update_policy(self, actions, padding_mask, states, is_weights=None):
        new_actions, log_pi = self._get_update_action(states)
        q1 = self.model.q1(states, new_actions)
        q2 = self.model.q2(states, new_actions)
        min_q = torch.min(q1, q2)

        if padding_mask is not None:
            min_q *= padding_mask
            log_pi *= padding_mask

        if self.algo == "awac":
            # Plan v8 — AWAC advantage-weighted policy update. Behavior-clone
            # the BUFFER action, weighted by exp(advantage / lambda). The
            # state-value baseline V(s) is the policy-sample value `min_q`
            # (already computed above); advantage + weight are detached —
            # they only scale the BC loss, no grad flows through them.
            with torch.no_grad():
                q1_buf = self.model.q1(states, actions)
                q2_buf = self.model.q2(states, actions)
                q_buf = torch.min(q1_buf, q2_buf)
                advantage = q_buf - min_q.detach()
                # RL_IMPROV_16 E1b — batch-normalized advantages (see
                # __init__). tau=0 keeps the legacy fixed-lambda path.
                if self.awac_adv_norm_tau > 0.0:
                    if self.awac_mode_adv_norm:
                        # RL_IMPROV_16 (v3c) (v3c, CLI-gated, default off) —
                        # PER-MODE normalization: contact-flagged samples
                        # (ground-truth privileged contact, available in
                        # `states` at update time) and free samples are
                        # centered/scaled within their own stratum, so the
                        # contact stratum's systematically-lower advantages
                        # rank actions against EACH OTHER instead of being
                        # uniformly down-weighted by the global std.
                        # Stratum < 8 samples falls back to the global
                        # scale (uncentered, legacy semantics).
                        contact = (
                            states[..., self.awac_contact_idx]
                            > self.awac_contact_thresh
                        ).unsqueeze(-1)
                        g_std = advantage.std().clamp_min(1e-4)
                        adv_n = advantage / g_std
                        for m in (contact, ~contact):
                            if int(m.sum()) >= 8:
                                sel = advantage[m]
                                adv_n[m] = (sel - sel.mean()) / sel.std().clamp_min(1e-4)
                        weight = torch.exp(
                            adv_n / self.awac_adv_norm_tau
                        ).clamp(max=20.0)
                    else:
                        adv_std = advantage.std().clamp_min(1e-4)
                        weight = torch.exp(
                            (advantage / adv_std) / self.awac_adv_norm_tau
                        ).clamp(max=20.0)
                else:
                    weight = torch.exp(
                        advantage / self.awac_lambda
                    ).clamp(max=20.0)
                # Plan v11 risk audit #1 — log AWAC weight saturation
                # EVERY AWAC update. If >80 % of weights are < 0.05 the
                # advantage signal has collapsed onto a few demos and
                # lambda is too small. Stored as instance attrs so the
                # dict rebuild in update() (which reassigns
                # self.last_metrics) doesn't wipe them.
                self._awac_weight_saturation = (
                    (weight < 0.05).float().mean().item()
                )
                self._awac_weight_max = float(weight.max().item())
                self._awac_weight_mean = float(weight.mean().item())
                # RL_IMPROV_16 E1b gate metric — weight p99/p1 ratio.
                # Target band 5-20x; ~1.7x = BC-degenerate (the v2
                # failure), >>20x = over-sharp (cloning too few demos).
                w_q = torch.quantile(
                    weight, torch.tensor(
                        [0.01, 0.99], device=weight.device,
                        dtype=weight.dtype,
                    )
                )
                self._awac_weight_p99p1 = float(
                    (w_q[1] / w_q[0].clamp_min(1e-6)).item()
                )
            logp = self.model.policy.log_prob(states, actions)
            if padding_mask is not None:
                logp = logp * padding_mask
                weight = weight * padding_mask
            # Plan v11 anti-rail — PRIMARY mechanism: AWAC + auto-tuned max-
            # entropy. Add the SAC entropy term (+alpha * log_pi of the policy's
            # OWN sampled action, incl. the tanh Jacobian) with alpha auto-tuned
            # to target_entropy (default -n_actions) in _update_alpha (now enabled for
            # AWAC). This holds policy entropy at target — fighting BOTH log_std
            # collapse AND mean-rail saturation — with NO hand-picked per-dim
            # coefficients (the heatup data cannot derive them: actions are
            # uniform-random, statistically identical for success and failure).
            # alpha is detached here (its own gradient comes from _update_alpha).
            per_sample_loss = -(weight * logp) + self.alpha.detach() * log_pi
        else:
            # SAC — entropy-regularized policy improvement. alpha is already
            # detached (re-derived at the top of update()); the explicit
            # detach keeps correctness independent of _update_alpha's
            # zero_grad ordering.
            per_sample_loss = self.alpha.detach() * log_pi - min_q

        # PER: per-sample IS weighting of the policy objective. Guarded —
        # uniform batches (is_weights is None) use the plain mean as before.
        if is_weights is not None:
            policy_loss = (is_weights * per_sample_loss).mean()
        else:
            policy_loss = per_sample_loss.mean()

        # Plan v11 anti-rail regularizers — AWAC ONLY (SAC's own alpha*log_pi
        # entropy term already prevents collapse). AWAC's advantage-weighted BC
        # objective has NO entropy term, so without these the policy log_std
        # collapses (entropy -> -12 during pretrain) and the squashed mean
        # saturates at the +/-1 tanh rail (clamp_fraction -> 0.4-0.5: the
        # documented cath-leading death-spiral). Two terms keep it off the rails:
        #   (1) per-dim entropy bonus: subtract beta_i * mean_t(log_std_i) so
        #       minimizing the loss RAISES log_std (per-dim, cath_trans weighted
        #       heaviest — it was the saturating dim);
        #   (2) mean-margin penalty: penalize the squashed mean approaching the
        #       rail (push the pre-tanh mean toward 0). Per-dim log_std alone
        #       does NOT stop mean-rail saturation (Plan v11 risk audit #3).
        # Both are detach-free (grad flows to the policy) and gated to AWAC.
        if self.algo == "awac" and (
            self.entropy_beta_per_dim is not None
            or self.action_mean_penalty > 0.0
            or self.contact_mean_penalty > 0.0
        ):
            mean_b, log_std_b = self.model.policy(states)
            if self.entropy_beta_per_dim is not None:
                beta = torch.as_tensor(
                    self.entropy_beta_per_dim,
                    device=log_std_b.device,
                    dtype=log_std_b.dtype,
                )
                policy_loss = policy_loss - (beta * log_std_b.mean(dim=0)).sum()
            if self.contact_mean_penalty > 0.0:
                # RL_IMPROV_16 (v3c) (v3c, CLI-gated, default off) — contact-gated
                # anti-rail. The measured rail is mode-specific (86% of
                # CONTACT states vs 0% free at u=748k), so the mean-margin
                # penalty is applied per-sample with a heavier coefficient
                # on contact-flagged states; free states keep the global
                # action_mean_penalty. Review LOW fix: penalize the PRE-TANH
                # mean clamped at 6 (tanh(6)≈0.99999) instead of
                # atanh(tanh(mean).clamp(0.99)) — numerically identical in
                # the unrailed band but keeps a LIVE gradient on already-
                # railed samples (|pre-tanh|>2.65), which the legacy form
                # zeroes on exactly the population this penalty targets.
                pen = mean_b.clamp(-6.0, 6.0).abs().mean(dim=-1)
                contact_f = (
                    states[..., self.awac_contact_idx]
                    > self.awac_contact_thresh
                ).to(pen.dtype)
                coef = (
                    contact_f * self.contact_mean_penalty
                    + (1.0 - contact_f) * self.action_mean_penalty
                )
                policy_loss = policy_loss + (coef * pen).mean()
            elif self.action_mean_penalty > 0.0:
                am = torch.tanh(mean_b).clamp(-0.99, 0.99)
                policy_loss = (
                    policy_loss
                    + self.action_mean_penalty * torch.atanh(am).abs().mean()
                )

        # Gen-4 — auxiliary privileged-label supervision. The policy's aux
        # head predicted self.model.policy._last_aux during the
        # _get_update_action(states) forward above (same states; the AWAC
        # log_prob call re-runs the same forward, so the stash is current
        # either way). Labels are the true privileged values already inside
        # the full-width flat obs; the policy itself never SEES them (its
        # forward slices [..., :n_observations]) — it only learns to
        # predict them. In padded episode mode the aux head still predicts
        # non-zero on the zero-padded steps, so mask them out for parity
        # with the other losses (step mode: padding_mask is None -> exact).
        if (
            self.aux_coef > 0.0
            and self.aux_label_indices
            and getattr(self.model.policy, "_last_aux", None) is not None
        ):
            aux_pred = self.model.policy._last_aux
            idx = torch.as_tensor(
                self.aux_label_indices, dtype=torch.long, device=states.device
            )
            aux_labels = states.index_select(-1, idx).detach()
            # RL_IMPROV_16 E2.2 — loss-time z-scoring (see __init__). EMA
            # stats persist across updates (not checkpointed — they
            # re-converge within ~100 batches after a warm start, and the
            # aux term is representation shaping, not a value estimate).
            if self.aux_label_znorm:
                flat_labels = aux_labels.reshape(-1, aux_labels.shape[-1])
                with torch.no_grad():
                    b_mu = flat_labels.mean(dim=0)
                    b_var = flat_labels.var(dim=0, unbiased=False)
                    if self._aux_znorm_mu is None:
                        self._aux_znorm_mu = b_mu
                        self._aux_znorm_var = b_var
                    else:
                        self._aux_znorm_mu = (
                            0.99 * self._aux_znorm_mu + 0.01 * b_mu
                        )
                        self._aux_znorm_var = (
                            0.99 * self._aux_znorm_var + 0.01 * b_var
                        )
                    # Dead labels (e.g. an all-zero channel) keep sd at 1
                    # so they contribute ~0 loss instead of exploding.
                    sd = self._aux_znorm_var.sqrt()
                    sd = torch.where(
                        sd > 1e-6, sd, torch.ones_like(sd)
                    )
                    mu = self._aux_znorm_mu
                aux_pred = (aux_pred - mu) / sd
                aux_labels = (aux_labels - mu) / sd
            aux_se = (aux_pred - aux_labels).pow(2)
            if padding_mask is not None:
                aux_se = aux_se * padding_mask
                aux_mse = aux_se.sum() / padding_mask.sum().clamp(min=1.0) / (
                    aux_se.shape[-1]
                )
            else:
                aux_mse = aux_se.mean()
            policy_loss = policy_loss + self.aux_coef * aux_mse
            # RL_IMPROV_16 E2.3 — the aux loss was invisible for the whole
            # v2 run (silently ~0); surface it in the trainer CSV.
            self._aux_loss_value = float(aux_mse.detach().item())

        self.model.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_policy = compute_grad_norm(self.model.policy.parameters())
        self._min_q_mean = min_q.mean().detach()
        # Fix #6: Compute clamp fraction for direct saturation detection in losses CSV
        self._clamp_fraction = _compute_clamp_fraction(new_actions.detach())
        # Plan v8 — gradient clipping (after the diagnostic grad-norm read,
        # so the logged norm stays the pre-clip value). 0 = off.
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.policy.parameters(), self.grad_clip
            )
        self.model.policy_optimizer.step()
        if self.model.policy_scheduler:
            self.model.policy_scheduler.step()
        return log_pi, policy_loss

    def _update_q2(self, actions, padding_mask, states, expected_q, is_weights=None):
        curr_q2 = self.model.q2(states, actions)
        if padding_mask is not None:
            curr_q2 *= padding_mask
        # PER: IS-weighted MSE — `reduction='none'` then per-sample weight,
        # then mean. Guarded — uniform batches keep the plain mse_loss.
        if is_weights is not None:
            td = curr_q2 - expected_q.detach()
            q2_loss = (is_weights * td.pow(2)).mean()
        else:
            q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.model.q2_optimizer.zero_grad()
        q2_loss.backward()
        # Compute gradient norm before optimizer step
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

    def _update_q1(self, actions, padding_mask, states, expected_q, is_weights=None):
        curr_q1 = self.model.q1(states, actions)
        if padding_mask is not None:
            curr_q1 *= padding_mask
        # PER: IS-weighted MSE + per-sample |TD| extraction. The per-sample
        # TD magnitude is what the buffer needs to refresh priorities.
        td = curr_q1 - expected_q.detach()
        if is_weights is not None:
            q1_loss = (is_weights * td.pow(2)).mean()
        else:
            q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        # |TD| per sample — average over any sequence/extra dims so the
        # result is one scalar per buffer transition. Detached, CPU numpy.
        td_errors = (
            td.detach().abs().view(curr_q1.shape[0], -1).mean(dim=1).cpu().numpy()
        )

        self.model.q1_optimizer.zero_grad()
        q1_loss.backward()
        # Compute gradient norm before optimizer step
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

    def _get_expected_q(self, all_states, rewards, dones, padding_mask, seq_length):
        next_actions, next_log_pi = self._get_update_action(all_states)

        with torch.no_grad():
            next_target_q1 = self.model.target_q1(all_states, next_actions)
            next_target_q2 = self.model.target_q2(all_states, next_actions)

        if self.algo == "awac":
            # Plan v8 fix — AWAC critic target is a PLAIN Bellman backup,
            # NO max-entropy term. SAC's `- alpha*next_log_pi` is
            # incompatible with AWAC's advantage-weighted BC policy: that
            # policy goes peaky, so next_log_pi grows large-positive and the
            # entropy term becomes a huge negative drag that drives the
            # critic to -inf (the -406k divergence in rcca_awac_gradclip10).
            next_target_q = torch.min(next_target_q1, next_target_q2)
        else:
            next_target_q = (
                torch.min(next_target_q1, next_target_q2)
                - self.alpha * next_log_pi
            )
        # only use next_state for next_q_target
        next_target_q = torch.narrow(next_target_q, dim=1, start=1, length=seq_length)
        # reward_scaling scales the raw reward in the Bellman target
        # (identity at the default 1.0, which current configs use).
        expected_q = (
            rewards * self.reward_scaling
            + (1 - dones) * self.gamma * next_target_q
        )
        # RL_IMPROV_16 (v3c3) (v3c3, CLI-gated, default off) — target floor clip:
        # bound the regression target from below BEFORE padding masking
        # (clamp of the padded zeros is a no-op for any negative floor).
        # See __init__ docnote for the divergence forensic this guards.
        if self.q_target_floor is not None:
            expected_q = expected_q.clamp(min=self.q_target_floor)
        if padding_mask is not None:
            expected_q *= padding_mask
        return expected_q

    # epsilon makes sure that log(0) does not occur
    def _get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_batch, log_std = self.model.policy(state_batch)
        std_batch = log_std.exp()

        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)

        # log_pi_batch = torch.sum(normal.log_prob(z), dim=-1, keepdim=True) - torch.sum(
        #        torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True)

        return action_batch, log_pi_batch

    def compute_probe_values(self, probe_states: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute policy and critic values for probe states.
        
        Used for diagnostics to track how the policy and critics evolve.
        
        Args:
            probe_states: Tensor of shape (n_probes, seq_len, obs_dim) or (n_probes, obs_dim)
            
        Returns:
            Dictionary with policy_mean, policy_log_std, action_det, q1, q2 arrays
        """
        with torch.no_grad():
            # Ensure proper shape: (batch, seq, features)
            if probe_states.dim() == 2:
                probe_states = probe_states.unsqueeze(1)
                
            probe_states = probe_states.to(device=self.device, dtype=torch.float32)
            
            # Get policy outputs
            mean, log_std = self.model.policy(probe_states)
            action_det = torch.tanh(mean)
            
            # Get critic values for deterministic action
            q1 = self.model.q1(probe_states, action_det)
            q2 = self.model.q2(probe_states, action_det)
            
        return {
            "policy_mean": mean.cpu().numpy(),
            "policy_log_std": log_std.cpu().numpy(),
            "action_det": action_det.cpu().numpy(),
            "q1": q1.cpu().numpy(),
            "q2": q2.cpu().numpy(),
        }

    def compute_batch_sample_values(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        sample_indices: list,
    ) -> List[Dict[str, Any]]:
        """
        Compute actor/critic values for sampled transitions from the training batch.
        
        Args:
            states: Batch states tensor (batch, seq, obs_dim)
            actions: Batch actions tensor (batch, seq, action_dim)
            rewards: Batch rewards tensor (batch, seq, 1)
            dones: Batch done flags tensor (batch, seq, 1)
            next_states: Next states (states[:, 1:, :] shifted)
            sample_indices: Indices of samples to extract
            
        Returns:
            List of dictionaries, each containing transition info and actor/critic values
        """
        samples = []
        
        # Move tensors to device for model forward pass
        states = states.to(device=self.device, dtype=torch.float32)
        actions = actions.to(device=self.device, dtype=torch.float32)
        if next_states is not None:
            next_states = next_states.to(device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            for idx in sample_indices:
                if idx >= states.shape[0]:
                    continue
                    
                # Extract single transition (take first timestep if sequence)
                s_t = states[idx:idx+1, 0:1, :]  # (1, 1, obs_dim)
                a_t = actions[idx, 0, :]  # (action_dim,)
                r_t = rewards[idx, 0, 0].item() if rewards.dim() > 2 else rewards[idx, 0].item()
                done_t = dones[idx, 0, 0].item() if dones.dim() > 2 else dones[idx, 0].item()
                
                # Get next state if available
                if next_states is not None and idx < next_states.shape[0]:
                    s_next = next_states[idx:idx+1, 0:1, :]
                else:
                    s_next = None
                    
                # Compute actor outputs for s_t
                mean, log_std = self.model.policy(s_t)
                action_det = torch.tanh(mean)
                
                # Compute Q values for (s_t, a_taken)
                a_taken = a_t.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
                q1_taken = self.model.q1(s_t, a_taken).squeeze().item()
                q2_taken = self.model.q2(s_t, a_taken).squeeze().item()
                
                # Compute Q values for (s_t, actor_action)
                q1_actor = self.model.q1(s_t, action_det).squeeze().item()
                q2_actor = self.model.q2(s_t, action_det).squeeze().item()
                
                # Extract action components
                actor_det_np = action_det.squeeze().cpu().numpy()
                action_taken_np = a_t.cpu().numpy()
                mean_np = mean.squeeze().cpu().numpy()
                log_std_np = log_std.squeeze().cpu().numpy()
                
                # Compute rotation stats (action[1] is typically rotation for 2-action envs)
                rotation_taken = action_taken_np[1] if len(action_taken_np) > 1 else 0.0
                rotation_actor = actor_det_np[1] if len(actor_det_np) > 1 else 0.0
                rotation_mean_val = mean_np[1] if len(mean_np) > 1 else 0.0
                rotation_log_std_val = log_std_np[1] if len(log_std_np) > 1 else 0.0
                
                # Compute clamp fraction for this sample's actor action
                clamp_frac = _compute_clamp_fraction(action_det, threshold=0.99)
                
                sample_data = {
                    "batch_idx": idx,
                    "state": s_t.squeeze().cpu().numpy().tolist(),
                    "action_taken": action_taken_np.tolist(),
                    "reward": r_t,
                    "done": done_t,
                    "next_state": s_next.squeeze().cpu().numpy().tolist() if s_next is not None else None,
                    "actor_mean": mean_np.tolist(),
                    "actor_log_std": log_std_np.tolist(),
                    "actor_det_action": actor_det_np.tolist(),
                    "q1_action_taken": q1_taken,
                    "q2_action_taken": q2_taken,
                    "q1_actor_action": q1_actor,
                    "q2_actor_action": q2_actor,
                    "min_q_taken": min(q1_taken, q2_taken),
                    "min_q_actor": min(q1_actor, q2_actor),
                    # NEW: Rotation stats
                    "rotation_taken": float(rotation_taken),
                    "rotation_actor": float(rotation_actor),
                    "rotation_mean": float(rotation_mean_val),
                    "rotation_log_std": float(rotation_log_std_val),
                    # NEW: Clamp fraction
                    "clamp_fraction": clamp_frac,
                }
                samples.append(sample_data)
                
        return samples

    def to(self, device: torch.device):
        super().to(device)
        self.alpha = self.alpha.to(device)
        # target_entropy is a native float (see __init__) — device-free.
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()

    def to_play_only(self):
        return SACPlayOnly(
            self.model.to_play_only(),
            self.n_actions,
            self.action_scaling,
            self.exploration_action_noise,
        )
