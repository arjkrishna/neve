"""
waypointprogress.py - Waypoint-based progress reward for centerline navigation

This reward gives positive reward for passing waypoints along the shortest path
to the target, with:
- Fixed 10mm waypoint spacing (configurable)
- Scores starting at 0.0 at insertion point (no reward for just existing)
- Increasing reward within each branch on the path
- Big jump when entering the next branch on the path
- NEGATIVE waypoints on wrong branches (deeper = more negative)
- Pulling out of wrong branch gives POSITIVE reward!

Example reward progression:
- Path Branch 0 (start): 0.00, 0.03, 0.06, 0.09, ...  (starts at 0!)
- Path Branch 1:         0.10, 0.13, 0.16, ...        (jump at branch)
- Path Branch 2:         0.20, 0.24, 0.28, ...
- Wrong Branch:         -0.10, -0.13, -0.16, ... (deeper = more negative)
- Target reached:       +1.0 (from TargetReached reward)

Key insight: waypoints exist on ALL branches, not just path branches.
- Path branches: positive scores (forward = good)
- Wrong branches: negative scores (forward = bad, backward = good)
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

from .reward import Reward
from ..intervention import Intervention
from ..intervention.vesseltree import find_nearest_branch_to_point, Branch
from ..util.coordtransform import tracking3d_to_vessel_cs


@dataclass
class Waypoint:
    """A waypoint along a branch (path or wrong branch)."""
    position: np.ndarray  # (3,) position in vessel CS
    branch: Branch        # The branch this waypoint belongs to
    local_idx: int        # Index within the branch (0 = start, N = end)
    score: float          # Progress score: positive for path, negative for wrong branch
    is_on_path: bool      # True if this waypoint is on the correct path


class CenterlineWaypointProgress(Reward):
    """
    Reward for passing waypoints along the shortest path centerline.
    
    At episode reset:
    1. Gets the shortest path polyline from pathfinder
    2. Resamples it into waypoints at fixed 10mm intervals
    3. Assigns scores: starting at 0.0, increasing with branch jumps
    4. Creates negative-scored waypoints on wrong branches
    
    At each step:
    1. Finds the nearest waypoint to the tip
    2. Reward = score_change (current - last)
    3. Naturally handles: forward (+), backward (-), wrong branch entry (--), wrong branch exit (++)
    
    Args:
        intervention: The intervention object
        pathfinder: Pathfinder with path_points3d (FixedPathfinder or DijkstraPathfinder)
        waypoint_spacing_mm: Distance between waypoints in mm (default: 10.0)
        branch_base_increment: Score jump when entering new branch (default: 0.1)
        wrong_branch_penalty: Not used directly - kept for API compatibility
    """
    
    def __init__(
        self,
        intervention: Intervention,
        pathfinder,
        waypoint_spacing_mm: float = 10.0,  # Fixed spacing in mm
        branch_base_increment: float = 0.1,
        wrong_branch_penalty: float = 0.01,
    ) -> None:
        self.intervention = intervention
        self.pathfinder = pathfinder
        self.waypoint_spacing_mm = waypoint_spacing_mm  # Fixed 10mm spacing
        self.branch_base_increment = branch_base_increment
        self.wrong_branch_penalty = wrong_branch_penalty  # Kept for reference but not used directly
        
        # Computed at reset
        self._waypoints: List[Waypoint] = []       # All waypoints (path + wrong branches)
        self._path_waypoints: List[Waypoint] = []  # Only path waypoints
        self._branch_waypoints: dict = {}          # Dict: branch -> list of waypoints
        self._path_branch_set: set = set()
        self._waypoint_spacing: float = waypoint_spacing_mm  # Use fixed spacing
        
        # State
        self._last_score: float = 0.0
        
        self.reward = 0.0

    def reset(self, episode_nr: int = 0) -> None:
        """Reset for new episode. Build waypoints on all branches."""
        self.reward = 0.0
        self._last_score = 0.0
        
        # IMPORTANT: Get path branches BEFORE building waypoints
        # (needed to know which branches are wrong)
        if hasattr(self.pathfinder, 'path_branches'):
            self._path_branch_set = set(self.pathfinder.path_branches)
        elif hasattr(self.pathfinder, 'path_branch_set'):
            self._path_branch_set = self.pathfinder.path_branch_set
        else:
            self._path_branch_set = set(self.intervention.vessel_tree.branches)
        
        # Build waypoints on ALL branches (path + wrong)
        self._build_waypoints()
        
        # Find initial waypoint and score
        if self._waypoints:
            tip_vessel_cs = self._get_tip_vessel_cs()
            nearest_wp = self._find_nearest_waypoint_obj(tip_vessel_cs)
            if nearest_wp is not None:
                self._last_score = nearest_wp.score

    def _build_waypoints(self) -> None:
        """
        Build waypoints on ALL branches:
        - Path branches: positive scores (forward = reward)
        - Wrong branches: negative scores (deeper = more negative)
        
        This allows the agent to get positive reward for pulling OUT of wrong branches.
        """
        self._waypoints = []
        self._path_waypoints = []   # Waypoints on path (for nearest path waypoint lookup)
        self._branch_waypoints = {} # Dict: branch -> list of waypoints on that branch
        
        # Get path info from pathfinder
        if not hasattr(self.pathfinder, 'path_points3d'):
            return
        
        path_points_3d = self.pathfinder.path_points3d
        if path_points_3d is None or len(path_points_3d) < 2:
            return
        
        # Convert path to vessel CS
        fluoro = self.intervention.fluoroscopy
        path_vessel_cs = tracking3d_to_vessel_cs(
            path_points_3d,
            fluoro.image_rot_zx,
            fluoro.image_center,
        )
        
        # 1. Build waypoints on the PATH (positive scores)
        self._build_path_waypoints(path_vessel_cs)
        
        # 2. Build waypoints on WRONG branches (negative scores)
        self._build_wrong_branch_waypoints()

    def _build_path_waypoints(self, path_vessel_cs: np.ndarray) -> None:
        """
        Build waypoints along the path with positive scores.
        
        Scores increase within each branch, with jumps at branch changes.
        Example: Branch 0: 0.10, 0.13, 0.16 → Branch 1: 0.20, 0.24, 0.28
        """
        # Compute cumulative arc length
        diffs = np.linalg.norm(path_vessel_cs[1:] - path_vessel_cs[:-1], axis=1)
        cumlen = np.concatenate([[0], np.cumsum(diffs)])
        total_length = cumlen[-1]
        
        if total_length < 1e-6:
            return
        
        # Resample at waypoint_spacing intervals
        n_waypoints = max(2, int(np.ceil(total_length / self._waypoint_spacing)) + 1)
        sample_lengths = np.linspace(0, total_length, n_waypoints)
        
        resampled_points = []
        for target_len in sample_lengths:
            idx = np.searchsorted(cumlen, target_len, side='right') - 1
            idx = np.clip(idx, 0, len(path_vessel_cs) - 2)
            
            seg_start = cumlen[idx]
            seg_len = cumlen[idx + 1] - seg_start
            t = (target_len - seg_start) / seg_len if seg_len > 1e-6 else 0.0
            
            point = (1 - t) * path_vessel_cs[idx] + t * path_vessel_cs[idx + 1]
            resampled_points.append(point)
        
        resampled_points = np.array(resampled_points)
        
        # Assign each waypoint to a branch
        branch_assignments = []
        for point in resampled_points:
            branch = find_nearest_branch_to_point(point, self.intervention.vessel_tree)
            branch_assignments.append(branch)
        
        # Identify branch segment boundaries
        segments = []
        current_branch = branch_assignments[0]
        segment_start = 0
        
        for i, branch in enumerate(branch_assignments):
            if branch != current_branch:
                segments.append((segment_start, i - 1, current_branch))
                segment_start = i
                current_branch = branch
        segments.append((segment_start, len(resampled_points) - 1, current_branch))
        
        # Assign positive scores starting from 0.0
        # First waypoint = 0.0, then increase with jumps at branch boundaries
        # Example: Branch 0: 0.0, 0.03, 0.06, 0.09 → Branch 1: 0.10, 0.13, 0.16
        for seg_idx, (start, end, branch) in enumerate(segments):
            n_points_in_segment = end - start + 1
            
            # Base score for this branch segment
            # Branch 0 starts at 0.0, Branch 1 at 0.1, Branch 2 at 0.2, etc.
            base = self.branch_base_increment * seg_idx  # 0.0, 0.1, 0.2...
            next_base = self.branch_base_increment * (seg_idx + 1)
            score_range = next_base - base - 0.01  # Leave small gap before jump
            
            increment = score_range / (n_points_in_segment - 1) if n_points_in_segment > 1 else 0.0
            
            for local_idx, global_idx in enumerate(range(start, end + 1)):
                score = base + increment * local_idx
                
                waypoint = Waypoint(
                    position=resampled_points[global_idx],
                    branch=branch,
                    local_idx=local_idx,
                    score=score,
                    is_on_path=True,
                )
                self._waypoints.append(waypoint)
                self._path_waypoints.append(waypoint)
                
                # Store by branch
                if branch not in self._branch_waypoints:
                    self._branch_waypoints[branch] = []
                self._branch_waypoints[branch].append(waypoint)

    def _build_wrong_branch_waypoints(self) -> None:
        """
        Build waypoints on wrong branches with NEGATIVE scores.
        
        Deeper into wrong branch = more negative score.
        Pulling out of wrong branch = positive reward (score becomes less negative).
        """
        vessel_tree = self.intervention.vessel_tree
        
        for branch in vessel_tree.branches:
            # Skip branches that are on the path
            if branch in self._path_branch_set:
                continue
            
            # Get branch centerline points
            coords = branch.coordinates
            if len(coords) < 2:
                continue
            
            # Compute cumulative arc length along this branch
            diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            cumlen = np.concatenate([[0], np.cumsum(diffs)])
            total_length = cumlen[-1]
            
            if total_length < 1e-6:
                continue
            
            # Find connection point to path (where this branch connects to a path branch)
            # This is typically near the start of the branch
            connection_idx = self._find_path_connection_index(branch, coords)
            
            # Resample at waypoint_spacing intervals
            n_waypoints = max(2, int(np.ceil(total_length / self._waypoint_spacing)) + 1)
            sample_lengths = np.linspace(0, total_length, n_waypoints)
            
            # Store waypoints for this wrong branch
            self._branch_waypoints[branch] = []
            
            for local_idx, target_len in enumerate(sample_lengths):
                idx = np.searchsorted(cumlen, target_len, side='right') - 1
                idx = np.clip(idx, 0, len(coords) - 2)
                
                seg_start = cumlen[idx]
                seg_len = cumlen[idx + 1] - seg_start
                t = (target_len - seg_start) / seg_len if seg_len > 1e-6 else 0.0
                
                point = (1 - t) * coords[idx] + t * coords[idx + 1]
                
                # Distance from connection point (deeper = higher distance)
                if connection_idx == 0:
                    depth = target_len  # Distance from start
                else:
                    depth = total_length - target_len  # Distance from end
                
                # Negative score: deeper = more negative
                # Scale similar to path scores but negative
                score = -self.branch_base_increment * (1 + depth / total_length)
                
                waypoint = Waypoint(
                    position=point,
                    branch=branch,
                    local_idx=local_idx,
                    score=score,
                    is_on_path=False,
                )
                self._waypoints.append(waypoint)
                self._branch_waypoints[branch].append(waypoint)

    def _find_path_connection_index(self, branch: Branch, coords: np.ndarray) -> int:
        """
        Find which end of the branch connects to the path.
        
        Returns 0 if start of branch is closer to path, else returns len(coords)-1.
        """
        if not self._path_waypoints:
            return 0
        
        # Get all path waypoint positions
        path_positions = np.array([wp.position for wp in self._path_waypoints])
        
        # Distance from branch start to nearest path point
        start_dists = np.linalg.norm(path_positions - coords[0], axis=1)
        start_min = np.min(start_dists)
        
        # Distance from branch end to nearest path point
        end_dists = np.linalg.norm(path_positions - coords[-1], axis=1)
        end_min = np.min(end_dists)
        
        # Return the end that's closer to path
        return 0 if start_min < end_min else len(coords) - 1

    def step(self) -> None:
        """
        Compute reward based on waypoint progress.
        
        Reward = change in score
        
        The score system naturally handles all cases:
        - Forward on path: score increases → positive reward
        - Backward on path: score decreases → negative reward  
        - Deeper into wrong branch: score becomes more negative → negative reward
        - Pulling out of wrong branch: score becomes less negative → POSITIVE reward!
        
        No separate wrong_branch_penalty needed - it's built into the negative scores.
        """
        if not self._waypoints:
            self.reward = 0.0
            return
        
        tip_vessel_cs = self._get_tip_vessel_cs()
        
        # Find nearest waypoint (could be on path or wrong branch)
        nearest_wp = self._find_nearest_waypoint_obj(tip_vessel_cs)
        
        if nearest_wp is None:
            self.reward = 0.0
            return
        
        current_score = nearest_wp.score
        
        # Reward = score change
        # This naturally handles all cases (see docstring)
        self.reward = current_score - self._last_score
        
        # Update state
        self._last_score = current_score

    def _get_tip_vessel_cs(self) -> np.ndarray:
        """Get tip position in vessel coordinate system."""
        fluoro = self.intervention.fluoroscopy
        position = fluoro.tracking3d[0]  # Tip position
        position_vessel_cs = tracking3d_to_vessel_cs(
            position, fluoro.image_rot_zx, fluoro.image_center
        )
        return position_vessel_cs

    def _find_nearest_waypoint_obj(self, position: np.ndarray) -> Optional[Waypoint]:
        """Find the nearest waypoint to the given position."""
        if not self._waypoints:
            return None
        
        min_dist = np.inf
        nearest_wp = None
        
        for wp in self._waypoints:
            dist = np.linalg.norm(wp.position - position)
            if dist < min_dist:
                min_dist = dist
                nearest_wp = wp
        
        return nearest_wp
    
    def get_waypoints(self) -> List[Waypoint]:
        """Get the list of waypoints (for debugging/visualization)."""
        return self._waypoints
    
    def get_path_waypoints(self) -> List[Waypoint]:
        """Get only the waypoints on the correct path."""
        return self._path_waypoints
    
    def get_current_progress(self) -> Tuple[int, int, float]:
        """
        Get current progress info.
        
        Returns:
            (n_path_waypoints, n_total_waypoints, current_score)
        """
        return (
            len(self._path_waypoints),
            len(self._waypoints),
            self._last_score,
        )
