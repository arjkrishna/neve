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


def _branch_interior_point(branch, junction_coord, offset_mm=15.0):
    """Walk ``offset_mm`` along ``branch.coordinates`` starting from the point
    nearest ``junction_coord``, in the direction that moves away from the
    junction. Returns the interpolated 3D point; falls back to the far
    endpoint if the branch is shorter than ``offset_mm``.

    Used to disambiguate wrong-branch vs correct-branch entries at
    bifurcation junctions: storing the shared junction coordinate in both
    lists (old behavior) made direction features 8-13 of LocalGuidance
    degenerate. Storing an interior point per connected branch makes
    ``wrong_entry_dir != correct_entry_dir`` at every bifurcation.
    """
    coords = branch.coordinates
    if len(coords) < 2:
        return coords[0] if len(coords) == 1 else np.asarray(junction_coord)

    dists = np.linalg.norm(coords - junction_coord, axis=1)
    i_start = int(np.argmin(dists))

    # Choose direction that takes the first step further from the junction.
    direction = None
    for cand in (+1, -1):
        i_next = i_start + cand
        if 0 <= i_next < len(coords):
            if np.linalg.norm(coords[i_next] - junction_coord) > dists[i_start]:
                direction = cand
                break
    if direction is None:
        direction = +1 if i_start < len(coords) - 1 else -1

    accumulated = 0.0
    i = i_start
    while 0 <= i + direction < len(coords):
        seg = coords[i + direction] - coords[i]
        seg_len = float(np.linalg.norm(seg))
        if accumulated + seg_len >= offset_mm:
            remaining = offset_mm - accumulated
            t = remaining / seg_len if seg_len > 1e-8 else 0.0
            return coords[i] + t * seg
        accumulated += seg_len
        i += direction
    return coords[i]


class PathProjectionCache:
    """Caches polyline projection and tip_vessel_cs once per env step.

    Args:
        pathfinder: A FixedPathfinder with ``path_points_vessel_cs``.
        intervention: The intervention object (provides fluoroscopy).
    """

    # Tell eve ConfigHandler to skip this class (not needed for config saving)
    _eve_skip_config = True

    def __init__(
        self,
        pathfinder,
        intervention,
        on_branch_flip_threshold: int = 5,
    ) -> None:
        self.pathfinder = pathfinder
        self.intervention = intervention
        self._on_branch_flip_threshold = on_branch_flip_threshold

        # Path geometry (set on reset)
        self._polyline = np.empty((0, 3))
        self._cumlen = np.empty(0)
        self._total_length = 0.0

        # Branch index (built at reset)
        # RL_IMPROV_7 §7 Fix 10: replaced KD-tree over all-branch centerline
        # points with per-branch polyline perpendicular-distance projection.
        # The KD-tree produced noisy classifications near bifurcation
        # junctions (proximal segments of adjacent branches are within mm),
        # flipping the nearest-branch winner with tiny tip jitter. The
        # polyline projection is geometrically stable: even AT a junction,
        # each branch's polyline diverges in a different direction, so
        # perpendicular distance to each is well-defined and continuous.
        self._branches_tuple = ()
        self._branch_polylines = ()
        self._branch_cumlens = ()

        # Branch entry points (built at reset)
        self._wrong_branch_entries = np.empty((0, 3))
        self._correct_branch_entries = np.empty((0, 3))

        # Junction arclengths along the planned path (each junction's nearest
        # path-point arclength). Used by heuristic for arclength-based d_corr.
        # Independent from the inside-point arrays above which are kept as-is
        # for reward computation.
        self._path_junction_arclengths = np.empty(0)

        # Cross-step hysteresis state for is_on_correct_branch() —
        # persists across invalidate() calls; reset only in reset().
        self._stable_on_branch = None       # current debounced classification
        self._pending_flip_count = 0         # consecutive raw disagreements

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
        # Reset hysteresis state per episode so a new vessel/path starts
        # with a fresh branch-membership evaluation.
        self._stable_on_branch = None
        self._pending_flip_count = 0
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
        """Precompute per-branch polylines + cumulative arclengths for
        perpendicular-distance nearest-branch classification.

        Each branch has its own centerline polyline. At query time,
        project the tip onto each polyline separately and pick the branch
        with the smallest perpendicular (cross-track) distance. Unlike
        the previous KD-tree approach, the winner is stable near
        bifurcation junctions because polylines diverge in distinct
        directions from the junction.
        """
        vessel_tree = self.intervention.vessel_tree
        if vessel_tree.branches is None or len(vessel_tree.branches) == 0:
            self._branches_tuple = ()
            self._branch_polylines = ()
            self._branch_cumlens = ()
            return

        polylines = []
        cumlens = []
        for branch in vessel_tree.branches:
            coords = np.asarray(branch.coordinates)
            polylines.append(coords)
            if len(coords) >= 2:
                cumlens.append(compute_cumulative_arclength(coords))
            else:
                cumlens.append(np.zeros(max(1, len(coords))))

        self._branches_tuple = tuple(vessel_tree.branches)
        self._branch_polylines = tuple(polylines)
        self._branch_cumlens = tuple(cumlens)

    def _build_entry_points(self) -> None:
        """Precompute branch-interior points (one per connected branch per
        junction), classified by whether that branch is in the correct path.

        Per RL_IMPROV_7 §4: storing the shared junction coordinate in both
        wrong/correct lists made features 9-10 ≡ 12-13 in LocalGuidance
        (degenerate direction features). Storing one interior point per
        connected branch, 15 mm into the branch from the junction,
        disambiguates wrong vs correct direction at every bifurcation.
        """
        vessel_tree = self.intervention.vessel_tree
        path_set = self.pathfinder.path_branch_set

        wrong_entries = []
        correct_entries = []
        for bp in vessel_tree.branching_points:
            for branch in bp.connections:
                interior = _branch_interior_point(
                    branch, bp.coordinates, offset_mm=15.0
                )
                if branch in path_set:
                    correct_entries.append(interior)
                else:
                    wrong_entries.append(interior)

        self._wrong_branch_entries = (
            np.array(wrong_entries) if wrong_entries else np.empty((0, 3))
        )
        self._correct_branch_entries = (
            np.array(correct_entries) if correct_entries else np.empty((0, 3))
        )

        # Precompute arclength of each branching point ON the planned path.
        # Heuristic uses this for arclength-based d_corr (distance along the
        # path to the next junction the wire must thread). A junction is
        # considered "on path" if there is a path point within 5 mm of it.
        junction_arclengths = []
        if len(self._polyline) > 1 and len(self._cumlen) > 0:
            for bp in vessel_tree.branching_points:
                bp_xyz = np.asarray(bp.coordinates, dtype=float)
                dists = np.linalg.norm(self._polyline - bp_xyz, axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] < 5.0:
                    junction_arclengths.append(float(self._cumlen[idx]))
        self._path_junction_arclengths = (
            np.array(sorted(junction_arclengths))
            if junction_arclengths
            else np.empty(0)
        )

    def get_nearest_branch(self):
        """Return the branch with the smallest perpendicular distance from
        the tip to its centerline polyline. Computed once per step."""
        if self._nearest_branch is None:
            if not self._branches_tuple:
                return None
            tip = self.get_tip_vessel_cs()
            best_dist = float("inf")
            best_idx = 0
            for i, (poly, cumlen) in enumerate(
                zip(self._branch_polylines, self._branch_cumlens)
            ):
                if len(poly) < 2:
                    continue
                r = project_onto_polyline(tip, poly, cumlen)
                if r.cross_track_dist < best_dist:
                    best_dist = r.cross_track_dist
                    best_idx = i
            self._nearest_branch = self._branches_tuple[best_idx]
        return self._nearest_branch

    def is_on_correct_branch(self) -> bool:
        """Return the debounced branch-membership classification.

        The raw KD-tree nearest-branch lookup flips between on/off on every
        step when the tip is within a few mm of a bifurcation junction —
        see RL_IMPROV_7_CHANGES.md §3. To prevent that noise from resetting
        `env5.py`'s off-branch counter and making `wrong_branch_timeout`
        unreachable, this method applies a simple debounce: the reported
        state only flips after ``_on_branch_flip_threshold`` *consecutive*
        raw disagreements; a single agreeing step resets the counter.

        The raw signal is still computed every step (required for the
        projection cache semantics), but the *returned* value is the
        debounced one.
        """
        if self._is_on_correct_branch is None:
            branch = self.get_nearest_branch()
            if branch is not None:
                raw = bool(self.pathfinder.is_branch_on_path(branch))
            else:
                raw = True  # default on-path if no branches

            if self._stable_on_branch is None:
                # First query of the episode — seed the debouncer.
                self._stable_on_branch = raw
                self._pending_flip_count = 0
            elif raw != self._stable_on_branch:
                self._pending_flip_count += 1
                if self._pending_flip_count >= self._on_branch_flip_threshold:
                    self._stable_on_branch = raw
                    self._pending_flip_count = 0
            else:
                self._pending_flip_count = 0

            self._is_on_correct_branch = self._stable_on_branch
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
        """DEPRECATED — Euclidean distance from tip to nearest correct-path
        interior marker (15 mm into a correct daughter at any path junction).
        This metric was misleading because it includes trunk markers, so
        small d_corr could just mean "tip near upper trunk", not "tip near
        a daughter entry". Use ``get_arclength_to_next_correct_entry()`` and
        ``get_arclength_past_last_junction()`` instead. Kept for backward
        compatibility; no caller as of RL_IMPROV_7 §7 Fix 18."""
        if self._dist_to_correct_entry is None:
            self._compute_correct_entry()
        return self._dist_to_correct_entry

    def get_arclength_to_next_correct_entry(self) -> float:
        """Arclength along planned path from tip projection to next forward
        junction the path threads. Used by the heuristic; intentionally
        independent of the Euclidean inside-point d_corr above which is
        consumed by LocalGuidance / reward.

        Returns ``float('inf')`` when there is no junction ahead (tip is past
        the last bifurcation) or no junctions were detected on the path —
        callers should treat this as "no entry zone applies; act normally"
        rather than "we're at the entry". Returning 0.0 here would make a
        ``d_corr < threshold`` check fire for the entire post-junction
        trajectory and freeze the wire in slow-mode all the way to the
        target.
        """
        if len(self._path_junction_arclengths) == 0:
            return float("inf")
        proj = self.get_projection()
        ahead = self._path_junction_arclengths[
            self._path_junction_arclengths > proj.s
        ]
        if len(ahead) == 0:
            return float("inf")
        return float(ahead[0] - proj.s)

    def get_arclength_past_last_junction(self) -> float:
        """Arclength from the most recent junction at or behind the tip's
        projection. Mirror of `get_arclength_to_next_correct_entry()` for
        the backward direction. Returns 0.0 if no junction is yet behind
        the tip (i.e. wire hasn't crossed bif1 yet) or if no junctions are
        detected on the path.

        Used by the heuristic to detect "wire has entered a daughter past
        the second-entry threshold" (arc_past > 10 mm) and by env5 to fire
        the +1 CORRECT_ENTRY_REWARD once per junction crossing.
        """
        if len(self._path_junction_arclengths) == 0:
            return 0.0
        proj = self.get_projection()
        behind = self._path_junction_arclengths[
            self._path_junction_arclengths <= proj.s
        ]
        if len(behind) == 0:
            return 0.0
        return float(proj.s - behind[-1])

    def get_nearest_named_branch_idx(self) -> int:
        """Index 0-3 of the NAMED supra-aortic daughter (LCCA, LVA, RCCA,
        RVA in vessel-tree-branch order if their names appear in branches)
        whose centerline polyline the tip is currently nearest to (perp.
        distance). Returns -1 if no named daughter is identifiable or if
        the nearest branch is some other (trunk/bridge/sub-) curve.

        Used purely for diagnostic logging (which daughter is the wire
        committing to right now).
        """
        if not self._branches_tuple:
            return -1
        # Build name→idx map once per process; cheap.
        if not hasattr(self, "_named_indices"):
            named_targets = ("LCCA", "LVA", "RCCA", "RVA")
            mapping = {}
            for i, br in enumerate(self._branches_tuple):
                name = getattr(br, "name", "") or ""
                for t in named_targets:
                    if t in name:
                        mapping[i] = t
                        break
            self._named_indices = mapping
        if not self._named_indices:
            return -1
        tip = self.get_tip_vessel_cs()
        best_idx, best_dist = -1, float("inf")
        for i, _name in self._named_indices.items():
            poly = self._branch_polylines[i]
            cumlen = self._branch_cumlens[i]
            if len(poly) < 2:
                continue
            r = project_onto_polyline(tip, poly, cumlen)
            if r.cross_track_dist < best_dist:
                best_dist = r.cross_track_dist
                best_idx = i
        return best_idx

    def get_closest_correct_entry_coords(self) -> np.ndarray:
        """3D vessel-CS coordinates of nearest correct-path bifurcation."""
        if self._dist_to_correct_entry is None:
            self._compute_correct_entry()
        return self._closest_correct_entry_coords
