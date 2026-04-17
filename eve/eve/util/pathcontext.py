"""Shared per-step projection cache for path-aware components.

Both ArcLengthProgress (reward) and LocalGuidance (observation) need to
project the device tip onto the correct-path polyline every step.  Without
caching, this projection is computed twice.  PathProjectionCache computes
it lazily on first access and returns the cached result on subsequent
accesses within the same step.

Usage in env5.py:
    cache = PathProjectionCache(pathfinder, intervention)
    # pass cache to ArcLengthProgress and LocalGuidance
    # call cache.invalidate() at the start of each env step
    # call cache.reset() after each env reset
"""

import numpy as np

from ..util.coordtransform import tracking3d_to_vessel_cs
from ..util.polyline import (
    compute_cumulative_arclength,
    project_onto_polyline,
    ProjectionResult,
)


def _return_none():
    """Unpickle helper: returns None so PathProjectionCache dissolves on pickle."""
    return None


class PathProjectionCache:
    """Caches polyline projection and tip_vessel_cs once per env step.

    Args:
        pathfinder: A FixedPathfinder with ``path_points_vessel_cs``.
        intervention: The intervention object (provides fluoroscopy).
    """

    # Tell eve ConfigHandler to skip this class (not needed for config saving)
    _eve_skip_config = True

    def __init__(self, pathfinder, intervention) -> None:
        self.pathfinder = pathfinder
        self.intervention = intervention

        # Path geometry (set on reset)
        self._polyline = np.empty((0, 3))
        self._cumlen = np.empty(0)
        self._total_length = 0.0

        # Branch index (built at reset)
        self._all_branch_coords = np.empty((0, 3))
        self._branch_id_array = np.empty(0, dtype=np.int32)
        self._branches_tuple = ()
        self._branch_kdtree = None

        # Branch entry points (built at reset)
        self._wrong_branch_entries = np.empty((0, 3))
        self._correct_branch_entries = np.empty((0, 3))

        # Per-step cache (typing removed for Python 3.8 compat)
        self._tip_vessel_cs = None
        self._projection = None
        self._nearest_branch = None
        self._is_on_correct_branch = None
        self._dist_to_wrong_entry = None
        self._dist_to_correct_entry = None
        self._closest_wrong_entry_coords = None
        self._closest_correct_entry_coords = None

    def __reduce__(self):
        """Return None when pickled - cache is runtime-only, not config."""
        # Use module-level function (lambdas can't be pickled in Python 3.8)
        return (_return_none, ())

    def reset(self) -> None:
        """Recompute path geometry from the pathfinder (call after env reset)."""
        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(max(1, len(self._polyline)))
            self._total_length = 0.0
        else:
            self._cumlen = compute_cumulative_arclength(self._polyline)
            self._total_length = float(self._cumlen[-1])
        self._build_branch_index()
        self._build_entry_points()
        self.invalidate()

    def invalidate(self) -> None:
        """Mark cache as stale.  Call at the start of each env step."""
        self._tip_vessel_cs = None
        self._projection = None
        self._nearest_branch = None
        self._is_on_correct_branch = None
        self._dist_to_wrong_entry = None
        self._dist_to_correct_entry = None
        self._closest_wrong_entry_coords = None
        self._closest_correct_entry_coords = None

    @property
    def polyline(self) -> np.ndarray:
        return self._polyline

    @property
    def cumlen(self) -> np.ndarray:
        return self._cumlen

    @property
    def total_length(self) -> float:
        return self._total_length

    def get_tip_vessel_cs(self) -> np.ndarray:
        """Return tip position in vessel CS, computing once per step."""
        if self._tip_vessel_cs is None:
            fluoro = self.intervention.fluoroscopy
            tip_3d = fluoro.tracking3d[0]
            self._tip_vessel_cs = tracking3d_to_vessel_cs(
                tip_3d, fluoro.image_rot_zx, fluoro.image_center
            )
        return self._tip_vessel_cs

    def get_projection(self) -> ProjectionResult:
        """Return projection result, computing once per step."""
        if self._projection is None:
            if self._total_length < 1e-6:
                self._projection = ProjectionResult(
                    s=0.0,
                    cross_track_dist=0.0,
                    proj_point=np.zeros(3),
                    segment_idx=0,
                    t=0.0,
                )
            else:
                tip = self.get_tip_vessel_cs()
                self._projection = project_onto_polyline(
                    tip, self._polyline, self._cumlen
                )
        return self._projection

    # ------------------------------------------------------------------
    # Branch membership (built at reset, queried per step)
    # ------------------------------------------------------------------

    def _build_branch_index(self) -> None:
        """Precompute branch lookup from all branch coordinates."""
        vessel_tree = self.intervention.vessel_tree
        if vessel_tree.branches is None or len(vessel_tree.branches) == 0:
            self._branch_kdtree = None
            self._all_branch_coords = np.empty((0, 3))
            self._branch_id_array = np.empty(0, dtype=np.int32)
            self._branches_tuple = ()
            return

        all_coords = []
        branch_ids = []
        for i, branch in enumerate(vessel_tree.branches):
            all_coords.append(branch.coordinates)
            branch_ids.extend([i] * len(branch.coordinates))

        self._all_branch_coords = np.concatenate(all_coords, axis=0)
        self._branch_id_array = np.array(branch_ids, dtype=np.int32)
        self._branches_tuple = vessel_tree.branches

        try:
            from scipy.spatial import cKDTree
            self._branch_kdtree = cKDTree(self._all_branch_coords)
        except ImportError:
            self._branch_kdtree = None

    def _build_entry_points(self) -> None:
        """Precompute wrong-branch and correct-branch bifurcation points."""
        vessel_tree = self.intervention.vessel_tree
        path_set = self.pathfinder.path_branch_set

        wrong_entries = []
        correct_entries = []
        for bp in vessel_tree.branching_points:
            has_wrong = any(b not in path_set for b in bp.connections)
            has_correct = any(b in path_set for b in bp.connections)
            if has_wrong:
                wrong_entries.append(bp.coordinates)
            if has_correct:
                correct_entries.append(bp.coordinates)

        self._wrong_branch_entries = (
            np.array(wrong_entries) if wrong_entries else np.empty((0, 3))
        )
        self._correct_branch_entries = (
            np.array(correct_entries) if correct_entries else np.empty((0, 3))
        )

    def get_nearest_branch(self):
        """Return the nearest Branch to the tip, computing once per step."""
        if self._nearest_branch is None:
            tip = self.get_tip_vessel_cs()
            if self._branch_kdtree is not None:
                _, idx = self._branch_kdtree.query(tip)
            elif len(self._all_branch_coords) > 0:
                dists = np.linalg.norm(self._all_branch_coords - tip, axis=1)
                idx = int(np.argmin(dists))
            else:
                return None
            self._nearest_branch = self._branches_tuple[self._branch_id_array[idx]]
        return self._nearest_branch

    def is_on_correct_branch(self) -> bool:
        """Return True if the nearest branch is on the correct path."""
        if self._is_on_correct_branch is None:
            branch = self.get_nearest_branch()
            if branch is not None:
                self._is_on_correct_branch = self.pathfinder.is_branch_on_path(branch)
            else:
                self._is_on_correct_branch = True  # default on-path if no branches
        return self._is_on_correct_branch

    def _compute_wrong_entry(self) -> None:
        """Compute distance and coordinates of nearest wrong-branch bifurcation."""
        if len(self._wrong_branch_entries) == 0:
            self._dist_to_wrong_entry = 999.0
            self._closest_wrong_entry_coords = np.zeros(3)
        else:
            tip = self.get_tip_vessel_cs()
            dists = np.linalg.norm(self._wrong_branch_entries - tip, axis=1)
            idx = int(np.argmin(dists))
            self._dist_to_wrong_entry = float(dists[idx])
            self._closest_wrong_entry_coords = self._wrong_branch_entries[idx].copy()

    def _compute_correct_entry(self) -> None:
        """Compute distance and coordinates of nearest correct-path bifurcation."""
        if len(self._correct_branch_entries) == 0:
            self._dist_to_correct_entry = 999.0
            self._closest_correct_entry_coords = np.zeros(3)
        else:
            tip = self.get_tip_vessel_cs()
            dists = np.linalg.norm(self._correct_branch_entries - tip, axis=1)
            idx = int(np.argmin(dists))
            self._dist_to_correct_entry = float(dists[idx])
            self._closest_correct_entry_coords = self._correct_branch_entries[idx].copy()

    def get_dist_to_closest_wrong_entry(self) -> float:
        """Euclidean distance from tip to nearest wrong-branch bifurcation."""
        if self._dist_to_wrong_entry is None:
            self._compute_wrong_entry()
        return self._dist_to_wrong_entry

    def get_closest_wrong_entry_coords(self) -> np.ndarray:
        """3D vessel-CS coordinates of nearest wrong-branch bifurcation."""
        if self._dist_to_wrong_entry is None:
            self._compute_wrong_entry()
        return self._closest_wrong_entry_coords

    def get_dist_to_next_correct_entry(self) -> float:
        """Euclidean distance from tip to nearest correct-path bifurcation."""
        if self._dist_to_correct_entry is None:
            self._compute_correct_entry()
        return self._dist_to_correct_entry

    def get_closest_correct_entry_coords(self) -> np.ndarray:
        """3D vessel-CS coordinates of nearest correct-path bifurcation."""
        if self._dist_to_correct_entry is None:
            self._compute_correct_entry()
        return self._closest_correct_entry_coords
