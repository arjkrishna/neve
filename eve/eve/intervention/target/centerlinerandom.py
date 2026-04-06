from typing import Optional, List
import random
import numpy as np

from .target import Target
from ..vesseltree import VesselTree
from ..vesseltree.util.branch import BranchWithRadii
from ..fluoroscopy import Fluoroscopy
from ...util.coordtransform import vessel_cs_to_tracking3d, tracking3d_to_2d


class CenterlineRandom(Target):
    def __init__(
        self,
        vessel_tree: VesselTree,
        fluoroscopy: Fluoroscopy,
        threshold: float,
        branches: Optional[List[str]] = None,
        min_distance_between_possible_targets: Optional[float] = None,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.fluoroscopy = fluoroscopy
        self.threshold = threshold
        self.branches = branches
        self.min_distance_between_possible_targets = (
            min_distance_between_possible_targets
        )
        self.reached = False
        self.coordinates3d = np.zeros((3,), dtype=np.float32)

        self._potential_targets = None
        self._branch_targets = {}  # branch_name -> list of valid points
        self._branches_initialized = None
        self._rng = random.Random()

    @property
    def coordinates2d(self) -> np.ndarray:
        return tracking3d_to_2d(self.coordinates3d)

    def reset(self, episode_nr=0, seed=None, target_branch=None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        if self._branches_initialized != self.vessel_tree.branches:
            self._init_centerline_point_cloud()
            self._branches_initialized = self.vessel_tree.branches

        if target_branch is not None and target_branch in self._branch_targets:
            pool = self._branch_targets[target_branch]
        else:
            pool = self._potential_targets
        target_vessel_cs = self._rng.choice(pool)
        self.coordinates3d = vessel_cs_to_tracking3d(
            target_vessel_cs,
            self.fluoroscopy.image_rot_zx,
            self.fluoroscopy.image_center,
            self.fluoroscopy.field_of_view,
        )
        self.reached = False

    @property
    def target_branch_names(self) -> List[str]:
        """Return the list of branch names available for target sampling."""
        if self._branch_targets:
            return list(self._branch_targets.keys())
        if self.branches is not None:
            return list(self.branches)
        return []

    def _init_centerline_point_cloud(self):
        if self.branches is None:
            branch_keys = self.vessel_tree.keys()
            excluded_branches = []
        else:
            branch_keys = set(self.branches) & set(self.vessel_tree.keys())
            excluded_branches = set(self.vessel_tree.keys()) - set(self.branches)
        branch_keys = sorted(branch_keys)
        potential_targets = np.empty((0, 3))
        for branch in branch_keys:
            points = self.vessel_tree[branch].coordinates
            potential_targets = np.vstack((potential_targets, points))

        in_excluded = self._in_excluded_branches(potential_targets, excluded_branches)
        outside_forbidden = np.invert(in_excluded)
        self._potential_targets = potential_targets[outside_forbidden]

        # Build per-branch target pools (filtered by excluded-branch overlap)
        self._branch_targets = {}
        for branch_name in branch_keys:
            points = self.vessel_tree[branch_name].coordinates
            mask = np.invert(self._in_excluded_branches(points, excluded_branches))
            valid = points[mask]
            if len(valid) > 0:
                self._branch_targets[branch_name] = list(valid)

    def _in_excluded_branches(
        self, coordinates: np.ndarray, excluded_branches: List[str]
    ):
        in_branch = [False] * coordinates.shape[0]
        for branch_name in excluded_branches:
            branch = self.vessel_tree[branch_name]
            if isinstance(branch, BranchWithRadii):
                in_branch = branch.in_branch(coordinates) + in_branch
        return in_branch
