"""
Probe state evaluator for SAC training diagnostics.

Handles collection and evaluation of fixed probe states (start state + near-start states)
to track policy and value function evolution during training.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import torch
import os


class ProbeEvaluator:
    """
    Manages probe states for consistent policy/value evaluation during training.
    
    Probe states are fixed observations used to track how the policy and critics
    evolve during training. Typically includes:
    - Start state (initial observation after reset)
    - Near-start states (first N steps of fixed-seed episodes)
    """
    
    def __init__(
        self,
        n_near_start_states: int = 10,
        n_episodes_for_probes: int = 3,
    ):
        """
        Initialize the probe evaluator.
        
        Args:
            n_near_start_states: Number of near-start states to collect per episode
            n_episodes_for_probes: Number of episodes to collect probe states from
        """
        self.n_near_start_states = n_near_start_states
        self.n_episodes_for_probes = n_episodes_for_probes
        
        self.probe_states: Optional[np.ndarray] = None
        self.probe_states_torch: Optional[torch.Tensor] = None
        self.probe_metadata: List[Dict[str, Any]] = []
        
    def collect_from_episodes(self, episodes: List[Any]) -> np.ndarray:
        """
        Collect probe states from a list of episodes.
        
        Args:
            episodes: List of Episode objects from replay buffer
            
        Returns:
            numpy array of probe states
        """
        probe_states = []
        self.probe_metadata = []
        
        n_episodes = min(len(episodes), self.n_episodes_for_probes)
        
        for ep_idx in range(n_episodes):
            episode = episodes[ep_idx]
            
            # Get flat observations from the episode
            if hasattr(episode, 'flat_obs'):
                flat_obs = episode.flat_obs
            else:
                # Fallback if episode structure is different
                continue
                
            # Collect start state
            if len(flat_obs) > 0:
                probe_states.append(flat_obs[0])
                self.probe_metadata.append({
                    "type": "start",
                    "episode": ep_idx,
                    "step": 0,
                })
                
            # Collect near-start states
            n_states = min(self.n_near_start_states, len(flat_obs) - 1)
            for step_idx in range(1, n_states + 1):
                probe_states.append(flat_obs[step_idx])
                self.probe_metadata.append({
                    "type": "near_start",
                    "episode": ep_idx,
                    "step": step_idx,
                })
                
        if probe_states:
            self.probe_states = np.array(probe_states)
        else:
            self.probe_states = None
            
        return self.probe_states
        
    def set_probe_states(self, probe_states: np.ndarray):
        """
        Manually set probe states.
        
        Args:
            probe_states: numpy array of shape (n_probes, obs_dim)
        """
        self.probe_states = probe_states
        self.probe_states_torch = None
        self.probe_metadata = [
            {"type": "manual", "index": i} 
            for i in range(len(probe_states))
        ]
        
    def get_torch_states(self, device: torch.device = None) -> Optional[torch.Tensor]:
        """
        Get probe states as a torch tensor.
        
        Args:
            device: Target device for the tensor
            
        Returns:
            torch tensor of probe states or None if not set
        """
        if self.probe_states is None:
            return None
            
        if self.probe_states_torch is None or (
            device is not None and 
            self.probe_states_torch.device != device
        ):
            self.probe_states_torch = torch.as_tensor(
                self.probe_states,
                dtype=torch.float32,
                device=device or torch.device("cpu"),
            )
            # Add batch and sequence dimensions if needed
            if self.probe_states_torch.dim() == 1:
                self.probe_states_torch = self.probe_states_torch.unsqueeze(0).unsqueeze(0)
            elif self.probe_states_torch.dim() == 2:
                self.probe_states_torch = self.probe_states_torch.unsqueeze(1)
                
        return self.probe_states_torch
        
    def save(self, path: str):
        """
        Save probe states and metadata to disk.
        
        Args:
            path: Path to save file (.npz)
        """
        if self.probe_states is not None:
            np.savez(
                path,
                probe_states=self.probe_states,
                metadata=np.array(self.probe_metadata, dtype=object),
            )
            
    def load(self, path: str):
        """
        Load probe states from disk.
        
        Args:
            path: Path to load file (.npz)
        """
        if os.path.isfile(path):
            data = np.load(path, allow_pickle=True)
            self.probe_states = data["probe_states"]
            if "metadata" in data:
                self.probe_metadata = data["metadata"].tolist()
            self.probe_states_torch = None
            
    @property
    def n_probes(self) -> int:
        """Number of probe states."""
        if self.probe_states is None:
            return 0
        return len(self.probe_states)
        
    @property 
    def start_state_indices(self) -> List[int]:
        """Indices of start states in the probe array."""
        return [
            i for i, m in enumerate(self.probe_metadata)
            if m.get("type") == "start"
        ]
        
    @property
    def near_start_indices(self) -> List[int]:
        """Indices of near-start states in the probe array."""
        return [
            i for i, m in enumerate(self.probe_metadata)
            if m.get("type") == "near_start"
        ]


def compute_probe_summary(probe_values: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute summary statistics from probe values.
    
    Args:
        probe_values: Dictionary with keys like 'policy_mean', 'q1', etc.
        
    Returns:
        Dictionary of scalar summary statistics
    """
    summary = {}
    
    for key, values in probe_values.items():
        if isinstance(values, np.ndarray):
            # For the first probe (start state)
            if len(values) > 0:
                first = values[0] if values.ndim > 1 else values
                summary[f"{key}_start_mean"] = float(np.mean(first))
                summary[f"{key}_start_std"] = float(np.std(first))
                
            # Overall statistics
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
            
    return summary
