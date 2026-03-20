from typing import Tuple, Dict, Any, List
import logging
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
            action += np.random.normal(0, self.exploration_action_noise)
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
            action += np.random.normal(0, self.exploration_action_noise)
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
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
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
        self.target_entropy = -torch.ones(1) * n_actions

        # DIAGNOSTICS: Store metrics from last update for logging
        self.last_metrics: Dict[str, float] = {}

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
            action += np.random.normal(0, self.exploration_action_noise)
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
        (all_states, actions, rewards, dones, padding_mask) = batch
        # actions /= self.action_scaling

        all_states = all_states.to(dtype=torch.float32, device=self.device)
        actions = actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.float32, device=self.device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.float32, device=self.device)

        seq_length = actions.shape[1]
        states = torch.narrow(all_states, dim=1, start=0, length=seq_length)

        # use all_states for next_actions and next_log_pi for proper hidden_state initilaization
        expected_q = self._get_expected_q(
            all_states, rewards, dones, padding_mask, seq_length
        )

        # q1 update
        q1_loss = self._update_q1(actions, padding_mask, states, expected_q)

        # q2 update
        q2_loss = self._update_q2(actions, padding_mask, states, expected_q)

        log_pi, policy_loss = self._update_policy(padding_mask, states)

        self.model.update_target_q(self.tau)

        alpha_loss = self._update_alpha(log_pi)

        self.update_step += 1
        
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
        }
        
        return [
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
        ]

    def _update_alpha(self, log_pi) -> torch.Tensor:
        alpha_loss = (
            self.model.log_alpha * (-log_pi - self.target_entropy).detach()
        ).mean()
        self.model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.model.alpha_optimizer.step()

        self.alpha = self.model.log_alpha.exp()
        return alpha_loss

    def _update_policy(self, padding_mask, states):
        new_actions, log_pi = self._get_update_action(states)
        q1 = self.model.q1(states, new_actions)
        q2 = self.model.q2(states, new_actions)
        min_q = torch.min(q1, q2)

        if padding_mask is not None:
            min_q *= padding_mask
            log_pi *= padding_mask

        policy_loss = (self.alpha * log_pi - min_q).mean()

        self.model.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_policy = compute_grad_norm(self.model.policy.parameters())
        self._min_q_mean = min_q.mean().detach()
        self.model.policy_optimizer.step()
        if self.model.policy_scheduler:
            self.model.policy_scheduler.step()
        return log_pi, policy_loss

    def _update_q2(self, actions, padding_mask, states, expected_q):
        curr_q2 = self.model.q2(states, actions)
        if padding_mask is not None:
            curr_q2 *= padding_mask
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.model.q2_optimizer.zero_grad()
        q2_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_q2 = compute_grad_norm(self.model.q2.parameters())
        self._q2_mean = curr_q2.mean().detach()
        self.model.q2_optimizer.step()
        if self.model.q2_scheduler:
            self.model.q2_scheduler.step()
        return q2_loss

    def _update_q1(self, actions, padding_mask, states, expected_q):
        curr_q1 = self.model.q1(states, actions)
        if padding_mask is not None:
            curr_q1 *= padding_mask
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())

        self.model.q1_optimizer.zero_grad()
        q1_loss.backward()
        # Compute gradient norm before optimizer step
        self._grad_norm_q1 = compute_grad_norm(self.model.q1.parameters())
        self._q1_mean = curr_q1.mean().detach()
        self.model.q1_optimizer.step()
        if self.model.q1_scheduler:
            self.model.q1_scheduler.step()
        return q1_loss

    def _get_expected_q(self, all_states, rewards, dones, padding_mask, seq_length):
        next_actions, next_log_pi = self._get_update_action(all_states)

        with torch.no_grad():
            next_target_q1 = self.model.target_q1(all_states, next_actions)
            next_target_q2 = self.model.target_q2(all_states, next_actions)

        next_target_q = (
            torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_pi
        )
        # only use next_state for next_q_target
        next_target_q = torch.narrow(next_target_q, dim=1, start=1, length=seq_length)
        expected_q = rewards + (1 - dones) * self.gamma * next_target_q
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
                
                sample_data = {
                    "batch_idx": idx,
                    "state": s_t.squeeze().cpu().numpy().tolist(),
                    "action_taken": a_t.cpu().numpy().tolist(),
                    "reward": r_t,
                    "done": done_t,
                    "next_state": s_next.squeeze().cpu().numpy().tolist() if s_next is not None else None,
                    "actor_mean": mean.squeeze().cpu().numpy().tolist(),
                    "actor_log_std": log_std.squeeze().cpu().numpy().tolist(),
                    "actor_det_action": action_det.squeeze().cpu().numpy().tolist(),
                    "q1_action_taken": q1_taken,
                    "q2_action_taken": q2_taken,
                    "q1_actor_action": q1_actor,
                    "q2_actor_action": q2_actor,
                    "min_q_taken": min(q1_taken, q2_taken),
                    "min_q_actor": min(q1_actor, q2_actor),
                }
                samples.append(sample_data)
                
        return samples

    def to(self, device: torch.device):
        super().to(device)
        self.alpha = self.alpha.to(device)
        self.target_entropy = self.target_entropy.to(device)
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
