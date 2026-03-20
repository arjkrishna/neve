"""
normalizecenterlines2depisode.py - Normalize centerlines observation

This wrapper normalizes the centerline points to the same [-1, 1] range
as the tracking/target observations, while keeping the path mask and order
values unchanged.
"""

from typing import Optional
import numpy as np
from ..observation import Observation, gym
from ...intervention import Intervention
from .normalize import Normalize


class NormalizeCenterlines2DEpisode(Normalize):
    """
    Normalize centerline points to [-1, 1] based on episode tracking space.
    
    The observation has structure (per branch):
    - centerline points: (M * 2) values to normalize
    - path mask: 1 value (0 or 1) - keep unchanged
    - path order: 1 value (-1, 0, 1, ...) - keep unchanged
    
    Args:
        wrapped_obs: Centerlines2D observation
        intervention: The intervention object (for tracking space bounds)
        name: Optional override for observation name
    """
    
    def __init__(
        self,
        wrapped_obs: Observation,
        intervention: Intervention,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(wrapped_obs, name)
        self.intervention = intervention
        self._normalization_space = None
        
        # Get structure info from wrapped observation
        self._n_points_per_branch = wrapped_obs.n_points_per_branch
        self._points_per_branch = self._n_points_per_branch * 2  # x, y coords
        self._stride = self._points_per_branch + 2  # +2 for mask and order

    def reset(self, episode_nr: int = 0) -> None:
        """Reset and get normalization bounds for this episode."""
        self._normalization_space = (
            self.intervention.fluoroscopy.tracking2d_space_episode
        )
        return super().reset(episode_nr)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize centerline points while keeping mask/order unchanged.
        
        For each branch:
        - Normalize the 2D centerline points to [-1, 1]
        - Keep path_mask (0 or 1) and path_order (-1, 0, 1, ...) as-is
        """
        if self._normalization_space is None:
            return obs
        
        # 2D normalization bounds
        low = self._normalization_space.low  # (2,) for x, y
        high = self._normalization_space.high  # (2,) for x, y
        
        result = obs.copy()
        
        # Determine number of branches from observation length
        n_branches = len(obs) // self._stride
        
        for i in range(n_branches):
            start = i * self._stride
            points_end = start + self._points_per_branch
            
            # Get centerline points for this branch
            points_flat = obs[start:points_end]
            points = points_flat.reshape(self._n_points_per_branch, 2)
            
            # Normalize each point
            normalized_points = 2 * ((points - low) / (high - low + 1e-8)) - 1
            
            # Put back in result
            result[start:points_end] = normalized_points.flatten()
            
            # Keep mask and order unchanged (already at indices points_end and points_end+1)
        
        return result.astype(np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        """
        Get observation space with normalized bounds.
        
        - Centerline points: [-1, 1]
        - Path mask: [0, 1]
        - Path order: [-1, max_order] but we use [-1, 100] for safety
        """
        wrapped_space = self.wrapped_obs.space
        obs_dim = wrapped_space.shape[0]
        
        low = np.full(obs_dim, -1.0, dtype=np.float32)
        high = np.full(obs_dim, 1.0, dtype=np.float32)
        
        # Adjust bounds for mask and order values
        n_branches = obs_dim // self._stride
        for i in range(n_branches):
            mask_idx = i * self._stride + self._points_per_branch
            order_idx = mask_idx + 1
            
            # Path mask: [0, 1]
            low[mask_idx] = 0.0
            high[mask_idx] = 1.0
            
            # Path order: [-1, 100] (allowing for up to 100 branches in path)
            low[order_idx] = -1.0
            high[order_idx] = 100.0
        
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
