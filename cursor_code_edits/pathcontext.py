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

        # Per-step cache (typing removed for Python 3.8 compat)
        self._tip_vessel_cs = None
        self._projection = None

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
        self.invalidate()

    def invalidate(self) -> None:
        """Mark cache as stale.  Call at the start of each env step."""
        self._tip_vessel_cs = None
        self._projection = None

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
