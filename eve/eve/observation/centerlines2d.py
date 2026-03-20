"""
centerlines2d.py - Observation for vessel centerlines in 2D

This observation exposes the vessel centerline geometry to the policy,
along with a mask indicating which branches are on the desired path to target.

The observation consists of:
1. centerline_points: (B, M, 2) - Downsampled centerline points for each branch
2. path_mask: (B, 1) - Binary mask: 1 if branch is on path to target, 0 otherwise
3. path_order: (B, 1) - Order of branch in path (0, 1, 2, ...) or -1 if not on path

The observation is flattened for the policy network.

Usage:
    centerlines = Centerlines2D(
        intervention=intervention,
        pathfinder=pathfinder,  # Must be DijkstraPathfinder
        n_points_per_branch=10,
    )
"""

import numpy as np
import gymnasium as gym
from typing import TYPE_CHECKING

from ..intervention import Intervention
from .observation import Observation
from ..util.coordtransform import vessel_cs_to_tracking3d, tracking3d_to_2d

if TYPE_CHECKING:
    from ..pathfinder.dijkstra2 import DijkstraPathfinder


class Centerlines2D(Observation):
    """
    Observation that provides centerline geometry and path membership to the policy.
    
    For each branch in the vessel tree:
    - Downsamples the centerline to a fixed number of points
    - Converts to 2D (same projection as tracking/target observations)
    - Computes a path membership mask based on the pathfinder's shortest path
    
    Args:
        intervention: The intervention object
        pathfinder: DijkstraPathfinder instance (must have path_branches)
        n_points_per_branch: Number of points to downsample each branch to
        name: Name of this observation
    """
    
    def __init__(
        self,
        intervention: Intervention,
        pathfinder: "DijkstraPathfinder",
        n_points_per_branch: int = 10,
        name: str = "centerlines2d",
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.n_points_per_branch = n_points_per_branch
        
        # Will be set in reset()
        self._n_branches = 0
        self._downsampled_centerlines_vessel_cs = None  # (B, M, 3)
        self.obs = None
        
        # Path mask computed ONCE at reset (fixed for episode)
        self._path_mask = None  # (n_branches,)
        self._path_order = None  # (n_branches,)

    @property
    def space(self) -> gym.spaces.Box:
        """
        Observation space.
        
        Shape: (n_branches * (n_points_per_branch * 2 + 2),)
        - For each branch:
          - n_points_per_branch * 2: x, y coordinates of centerline points
          - 1: path mask (0 or 1)
          - 1: path order (-1, 0, 1, 2, ...)
        """
        n_branches = len(self.intervention.vessel_tree.branches)
        obs_dim = n_branches * (self.n_points_per_branch * 2 + 2)
        
        # Use large bounds - normalization wrapper will handle scaling
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        """
        Reset the observation.
        
        Precomputes:
        - Downsampled centerlines (fixed for vessel tree)
        - Path mask/order (fixed for episode - depends on target)
        """
        branches = self.intervention.vessel_tree.branches
        self._n_branches = len(branches)
        
        # Downsample each branch to fixed number of points (in vessel CS)
        self._downsampled_centerlines_vessel_cs = self._downsample_branches(branches)
        
        # FIXED: Compute path mask ONCE here (not every step)
        # Path is fixed for the episode since target doesn't change
        self._path_mask = np.zeros(self._n_branches, dtype=np.float32)
        self._path_order = np.full(self._n_branches, -1.0, dtype=np.float32)
        
        if hasattr(self.pathfinder, 'path_branches') and self.pathfinder.path_branches:
            for order, path_branch in enumerate(self.pathfinder.path_branches):
                for i, branch in enumerate(branches):
                    if branch == path_branch:
                        self._path_mask[i] = 1.0
                        self._path_order[i] = float(order)
                        break
        
        # Compute initial observation
        self.step()

    def step(self) -> None:
        """
        Update the observation.
        
        The centerline points and path mask are fixed (computed at reset).
        Only the 2D projection needs to be updated (in case camera moves, though
        typically it doesn't).
        """
        fluoro = self.intervention.fluoroscopy
        
        # Convert centerlines from vessel CS to 2D
        centerlines_2d = []
        for i in range(self._n_branches):
            points_vessel_cs = self._downsampled_centerlines_vessel_cs[i]
            
            # Convert to tracking3d space
            points_3d = vessel_cs_to_tracking3d(
                points_vessel_cs,
                fluoro.image_rot_zx,
                fluoro.image_center,
                fluoro.field_of_view,
            )
            
            # Handle case where points are out of field of view
            if len(points_3d) < self.n_points_per_branch:
                # Pad with last valid point or zeros
                if len(points_3d) > 0:
                    pad_point = points_3d[-1]
                else:
                    pad_point = np.zeros(3, dtype=np.float32)
                while len(points_3d) < self.n_points_per_branch:
                    points_3d = np.vstack([points_3d, pad_point])
            
            # Convert to 2D
            points_2d = tracking3d_to_2d(points_3d[:self.n_points_per_branch])
            centerlines_2d.append(points_2d.flatten())
        
        # Build flat observation using PRE-COMPUTED path mask (from reset)
        obs_parts = []
        for i in range(self._n_branches):
            obs_parts.append(centerlines_2d[i])  # (M * 2,)
            obs_parts.append(np.array([self._path_mask[i]], dtype=np.float32))  # (1,)
            obs_parts.append(np.array([self._path_order[i]], dtype=np.float32))  # (1,)
        
        self.obs = np.concatenate(obs_parts).astype(np.float32)

    def _downsample_branches(self, branches) -> np.ndarray:
        """
        Downsample each branch to n_points_per_branch points along arc length.
        
        Returns:
            (n_branches, n_points_per_branch, 3) array of points in vessel CS
        """
        result = np.zeros(
            (len(branches), self.n_points_per_branch, 3), dtype=np.float32
        )
        
        for i, branch in enumerate(branches):
            coords = branch.coordinates
            n_original = len(coords)
            
            if n_original <= 1:
                # Degenerate branch - fill with same point
                result[i] = np.tile(coords[0] if n_original > 0 else np.zeros(3), 
                                   (self.n_points_per_branch, 1))
                continue
            
            # Compute cumulative arc length
            diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            cumlen = np.concatenate([[0], np.cumsum(diffs)])
            total_length = cumlen[-1]
            
            if total_length < 1e-6:
                # Zero-length branch
                result[i] = np.tile(coords[0], (self.n_points_per_branch, 1))
                continue
            
            # Sample at evenly spaced arc lengths
            sample_lengths = np.linspace(0, total_length, self.n_points_per_branch)
            
            for j, target_len in enumerate(sample_lengths):
                # Find segment containing this arc length
                idx = np.searchsorted(cumlen, target_len, side='right') - 1
                idx = np.clip(idx, 0, n_original - 2)
                
                # Interpolate within segment
                seg_start = cumlen[idx]
                seg_len = cumlen[idx + 1] - seg_start
                
                if seg_len > 1e-6:
                    t = (target_len - seg_start) / seg_len
                else:
                    t = 0.0
                
                result[i, j] = (1 - t) * coords[idx] + t * coords[idx + 1]
        
        return result

    def get_centerlines_2d(self) -> np.ndarray:
        """
        Get the raw centerline points in 2D (before flattening).
        
        Returns:
            (n_branches, n_points_per_branch, 2) array
        """
        # Extract from flattened obs
        points_per_branch = self.n_points_per_branch * 2
        stride = points_per_branch + 2  # +2 for mask and order
        
        result = np.zeros((self._n_branches, self.n_points_per_branch, 2), dtype=np.float32)
        for i in range(self._n_branches):
            start = i * stride
            result[i] = self.obs[start:start + points_per_branch].reshape(
                self.n_points_per_branch, 2
            )
        return result

    def get_path_mask(self) -> np.ndarray:
        """
        Get the path mask for each branch.
        
        Returns:
            (n_branches,) array of 0s and 1s
        """
        points_per_branch = self.n_points_per_branch * 2
        stride = points_per_branch + 2
        
        result = np.zeros(self._n_branches, dtype=np.float32)
        for i in range(self._n_branches):
            mask_idx = i * stride + points_per_branch
            result[i] = self.obs[mask_idx]
        return result
