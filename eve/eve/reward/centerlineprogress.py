"""
centerlineprogress.py - Reward for centerline-aware navigation

This reward encourages the agent to:
1. Progress along the main trunk (root branch)
2. Stay close to the desired path centerlines
3. Avoid entering wrong branches

Reward components:
- r_root: Reward for progress along the root/main trunk branch
- r_path: Reward for getting closer to the path centerlines
- r_off: Penalty for being in a wrong (off-path) branch

Formula:
    r(t) = w_root * (s_root(t) - s_root(t-1)) * on_root
         + w_path * (d_path(t-1) - d_path(t))
         - w_off * 1[nearest_branch not in B_path]

Where:
- s_root(t): Arc-length position along root branch
- d_path(t): Distance to nearest point on path centerlines
- B_path: Set of branches on the shortest path to target
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Set

from .reward import Reward
from ..intervention import Intervention
from ..intervention.vesseltree import find_nearest_branch_to_point, Branch
from ..util.coordtransform import tracking3d_to_vessel_cs

if TYPE_CHECKING:
    from ..pathfinder.dijkstra2 import DijkstraPathfinder


class CenterlineProgress(Reward):
    """
    Reward for progress along centerlines with path-awareness.
    
    This reward shapes behavior to:
    1. Move forward along the root branch (main trunk)
    2. Stay close to the desired path when navigating branches
    3. Penalize entering wrong branches
    
    Args:
        intervention: The intervention object
        pathfinder: DijkstraPathfinder with path_branches available
        w_root: Weight for root branch progress reward
        w_path: Weight for path distance delta reward
        w_off: Weight for wrong branch penalty
    """
    
    def __init__(
        self,
        intervention: Intervention,
        pathfinder: "DijkstraPathfinder",
        w_root: float = 0.001,
        w_path: float = 0.001,
        w_off: float = 0.01,
    ) -> None:
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.w_root = w_root
        self.w_path = w_path
        self.w_off = w_off
        
        # State from previous step
        self._last_s_root: float = 0.0  # Arc length along root branch
        self._last_d_path: float = 0.0  # Distance to path centerlines
        
        # Precomputed data
        self._root_branch: Optional[Branch] = None
        self._root_cumlen: Optional[np.ndarray] = None  # Cumulative arc length
        self._path_centerlines: Optional[np.ndarray] = None  # All points on path
        
        self.reward = 0.0

    def reset(self, episode_nr: int = 0) -> None:
        """Reset reward state for new episode."""
        self.reward = 0.0
        
        # Get root branch (first branch = main trunk)
        branches = self.intervention.vessel_tree.branches
        if len(branches) > 0:
            self._root_branch = branches[0]
            # Precompute cumulative arc length for root branch
            coords = self._root_branch.coordinates
            diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            self._root_cumlen = np.concatenate([[0], np.cumsum(diffs)])
        else:
            self._root_branch = None
            self._root_cumlen = None
        
        # Initialize last values
        self._update_cached_values()
        self._last_s_root = self._compute_s_root()
        self._last_d_path = self._compute_d_path()

    def step(self) -> None:
        """Compute reward for this step."""
        # Update cached path centerlines
        self._update_cached_values()
        
        # Current values
        s_root = self._compute_s_root()
        d_path = self._compute_d_path()
        
        # Get current tip position and nearest branch
        tip_vessel_cs = self._get_tip_vessel_cs()
        nearest_branch = find_nearest_branch_to_point(
            tip_vessel_cs, self.intervention.vessel_tree
        )
        
        # Reward components
        r_root = 0.0
        r_path = 0.0
        r_off = 0.0
        
        # Term 1: Root branch progress
        if self._root_branch is not None and nearest_branch == self._root_branch:
            # Only reward progress when still on root branch
            delta_s = s_root - self._last_s_root
            r_root = self.w_root * delta_s
        
        # Term 2: Path distance delta
        if self._last_d_path is not None and d_path is not None:
            delta_d = self._last_d_path - d_path  # Positive when getting closer
            r_path = self.w_path * delta_d
        
        # Term 3: Wrong branch penalty
        if hasattr(self.pathfinder, 'is_branch_on_path'):
            if not self.pathfinder.is_branch_on_path(nearest_branch):
                r_off = -self.w_off
        
        # Total reward
        self.reward = r_root + r_path + r_off
        
        # Update last values
        self._last_s_root = s_root
        self._last_d_path = d_path

    def _get_tip_vessel_cs(self) -> np.ndarray:
        """Get tip position in vessel coordinate system."""
        fluoro = self.intervention.fluoroscopy
        position = fluoro.tracking3d[0]  # Tip position
        position_vessel_cs = tracking3d_to_vessel_cs(
            position, fluoro.image_rot_zx, fluoro.image_center
        )
        return position_vessel_cs

    def _update_cached_values(self) -> None:
        """Update cached path centerlines based on current target."""
        if not hasattr(self.pathfinder, 'path_branches'):
            self._path_centerlines = None
            return
        
        # Collect all centerline points from path branches
        path_branches = self.pathfinder.path_branches
        if not path_branches:
            self._path_centerlines = None
            return
        
        all_points = []
        for branch in path_branches:
            all_points.append(branch.coordinates)
        
        if all_points:
            self._path_centerlines = np.vstack(all_points)
        else:
            self._path_centerlines = None

    def _compute_s_root(self) -> float:
        """
        Compute arc-length position along root branch.
        
        Returns the arc-length coordinate of the projection of the tip
        onto the root branch centerline.
        """
        if self._root_branch is None or self._root_cumlen is None:
            return 0.0
        
        tip_vessel_cs = self._get_tip_vessel_cs()
        coords = self._root_branch.coordinates
        
        # Find nearest point on root branch
        distances = np.linalg.norm(coords - tip_vessel_cs, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Return arc length at that point
        return self._root_cumlen[nearest_idx]

    def _compute_d_path(self) -> float:
        """
        Compute distance from tip to nearest point on path centerlines.
        
        Returns minimum distance to any point on any path branch.
        """
        if self._path_centerlines is None or len(self._path_centerlines) == 0:
            return 0.0
        
        tip_vessel_cs = self._get_tip_vessel_cs()
        
        # Distance to all path centerline points
        distances = np.linalg.norm(self._path_centerlines - tip_vessel_cs, axis=1)
        return float(np.min(distances))
