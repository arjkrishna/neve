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

Plan v12 Stage 1 (multi-target heatup) multi-instance support:
    pathfinders = [FixedPathfinder(intervention) for _ in range(4)]
    # ... reset each with its daughter k's target_coord3d
    caches = make_per_target_caches(pathfinders, intervention)
    # caches[k] is bound to pathfinders[k]; no shared mutable state across
    # caches (verified by assert_caches_independent()).
"""

from typing import List, Sequence
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


# ============================================================================
# Plan v12 redesign — target-INDEPENDENT physical-branch classifier.
# ============================================================================
# The MultiTargetEnv5 v0 bug: `final_branch_short` (which daughter the wire
# physically ended in) was computed from each grader's OWN planned-path state
# machine (PathProjectionCache._current_branch_idx). That state machine
# commits to branches via junction-crossing history relative to ITS OWN
# planned path, so the SAME physical tip classified up to 4 different ways
# (data forensics: 4 graders agreed only 56/92 of length-aligned episodes).
#
# The physical branch a wire ends in is ONE geometric fact, independent of
# any target. classify_physical_branch() computes it ONCE per SOFA tick from
# the shared tip position via per-branch perpendicular (cross-track)
# projection over ALL vessel-tree branches (the RL_IMPROV_7 Fix 10 metric,
# geometrically stable at junctions). The driver broadcasts the single result
# into all 4 graders' info['final_branch_short'] so it is identical across
# the 4 per-target .npz files by construction.

_DAUGHTER_TAGS = ("RCCA", "LCCA", "RVA", "LVA")


def _physical_named_tag(name) -> str:
    """Map a branch name to a coarse anatomy tag.

    Named daughters → "RCCA"/"LCCA"/"RVA"/"LVA". Numbered trunk/bridge
    curves: (2) and (0) → "trunk", (11) → "bridge" (the bif2 ostium).
    Everything else → "other".
    """
    if not isinstance(name, str):
        return "other"
    for t in _DAUGHTER_TAGS:
        if t in name:
            return t
    import re as _re
    m = _re.search(r"\((\d+)\)", name)
    if m:
        n = int(m.group(1))
        if n in (2, 0):
            return "trunk"
        if n == 11:
            return "bridge"
    return "other"


def classify_physical_branch(intervention, prefer_named_within_mm: float = 8.0) -> str:
    """Target-INDEPENDENT physical branch the guidewire TIP is currently in.

    Returns one of "RCCA"|"LCCA"|"RVA"|"LVA"|"trunk"|"bridge"|"other".

    Reads ONLY intervention geometry + the physical tip — NO pathfinder,
    NO _current_branch_idx, NO planned-path state. So calling it once and
    broadcasting the result to all 4 graders yields an identical
    final_branch_short across the 4 per-target .npz files for the same
    physical trajectory.

    Method: project the tip (converted to vessel-CS) onto every branch's
    centerline polyline and take the min perpendicular (cross-track)
    distance branch. A "prefer named within N mm" snap: if a NAMED daughter
    is within prefer_named_within_mm of the tip, return that daughter even
    if a trunk/bridge curve is marginally closer (the terminal tip is deep
    inside one daughter, far from any ostium, so this disambiguates cleanly).
    """
    vt = getattr(intervention, "vessel_tree", None)
    if vt is None or not getattr(vt, "branches", None):
        return "other"
    fl = intervention.fluoroscopy
    tip3d = fl.tracking3d[0]
    tip_vessel_cs = tracking3d_to_vessel_cs(tip3d, fl.image_rot_zx, fl.image_center)

    best_dist = float("inf")
    best_name = None
    best_named_dist = float("inf")
    best_named_name = None
    for branch in vt.branches:
        coords = np.asarray(branch.coordinates)
        if len(coords) < 2:
            continue
        cumlen = compute_cumulative_arclength(coords)
        r = project_onto_polyline(tip_vessel_cs, coords, cumlen)
        d = r.cross_track_dist
        name = getattr(branch, "name", "") or ""
        if d < best_dist:
            best_dist = d
            best_name = name
        tag = _physical_named_tag(name)
        if tag in _DAUGHTER_TAGS and d < best_named_dist:
            best_named_dist = d
            best_named_name = name

    if best_named_name is not None and best_named_dist <= prefer_named_within_mm:
        return _physical_named_tag(best_named_name)
    if best_name is not None:
        return _physical_named_tag(best_name)
    return "other"


# RL_IMPROV_8: path-aware on_path classification thresholds.
# A branch centerline point is on-path iff its 3D Euclidean distance to the
# nearest planned-path-polyline point is below ON_PATH_TOLERANCE_MM. Used by
# `_branch_on_path_masks`. 3 mm is tight enough that points past a branch
# exit don't false-positive (typical inter-branch separation > 5 mm at
# junctions) but loose enough to absorb sub-mm centerline densification.
ON_PATH_TOLERANCE_MM = 3.0

# Legacy uniform threshold (kept for backward-compat / fallback).
# Superseded in v2 by radius-aware tolerance derived from local vessel
# radius — see K_RADIUS, MIN_TOLERANCE_MM below.
CROSS_TRACK_TOLERANCE_MM = 10.0

# RL_IMPROV_8 v2 — radius-aware cross-track tolerance:
#   tolerance = max(MIN_TOLERANCE_MM, K_RADIUS * local_vessel_radius)
# Trunk (radius ~12) → tolerance ~18 mm (wide lumen, in-vessel drift OK).
# Bif2 cavity (radius ~7) → tolerance ~10 mm (catches 13 mm wedge).
# Bridge (radius ~4) → tolerance ~6 mm (catches 49 mm RCCA wedge).
# Daughter (radius ~3) → tolerance ~5 mm (tight enough inside daughter).
K_RADIUS = 1.5
MIN_TOLERANCE_MM = 2.0
DEFAULT_RADIUS_MM = 5.0
MIN_RADIUS_FLOOR_MM = 2.0
MAX_RADIUS_CEILING_MM = 12.0

# State-machine hysteresis on junction crossings — wire must commit
# COMMIT_HYSTERESIS_MM of arclength past a junction before the
# _current_branch_idx state flips to the next on-path branch.
COMMIT_HYSTERESIS_MM = 10.0

# Plan v9 (reward-only) — a wrong-daughter commit fires -0.05 only when the
# wire has genuinely gone DEEP into a wrong branch: off-path arclength since
# divergence >= WRONG_COMMIT_ARC_MM. This is a REWARD-side threshold; it does
# NOT alter the on/off-path classifier (which the heuristic reads). A wire
# merely wedged at a fork mouth oscillates on/off-path but its off-path arc
# stays shallow, so it never triggers the -0.05 (no flicker hammering).
WRONG_COMMIT_ARC_MM = 10.0


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

        # RL_IMPROV_8: daughter-only entry markers. Subset of
        # _path_junction_arclengths that are real forks (have at least one
        # off-path connected branch). Used for the +1 reward and the
        # heuristic's "near-junction" regime selector. Coords stored in
        # path-arclength order so get_3d_dist_to_next_daughter_entry()
        # can find the next one.
        self._path_daughter_arclengths = np.empty(0)
        self._path_daughter_coords = np.empty((0, 3))  # 3D vessel-CS coords

        # RL_IMPROV_8: per-branch on-path masks. Maps id(branch) -> bool
        # array over branch.coordinates indices, marking which centerline
        # points are within ON_PATH_TOLERANCE_MM of the planned polyline.
        # Used by is_on_correct_path() check (c) — catches "wire on a shared
        # branch but on the off-path section past the exit point".
        self._branch_on_path_masks = {}

        # RL_IMPROV_8 v2: graph-routed d_corr support. Built at reset.
        #   _branch_to_bp_pair: id(branch) -> (bp_at_first_endpoint, bp_at_last_endpoint)
        #   _bp_adjacency:      bp -> list of (other_bp, branch, branch_length)
        #   _path_bp_set:       set of branching_points within 5 mm of polyline
        #   _bp_to_path_dist:   bp -> graph-distance to nearest path BP
        #   _bp_to_path_arclen: bp -> arclength on _polyline at the rejoin point
        # This precomputes the BP-to-path distances so the per-step routed-d_corr
        # only does an O(1) lookup at the wire's current branch endpoints.
        self._branch_to_bp_pair = {}
        self._bp_adjacency = {}
        self._path_bp_set = set()
        self._bp_to_path_dist = {}
        self._bp_to_path_arclen = {}

        # Cross-step hysteresis state for is_on_correct_branch() —
        # persists across invalidate() calls; reset only in reset().
        self._stable_on_branch = None       # current debounced classification
        self._pending_flip_count = 0         # consecutive raw disagreements

        # RL_IMPROV_8: separate hysteresis state for is_on_correct_path()
        self._stable_on_path = None
        self._pending_path_flip_count = 0

        # RL_IMPROV_8 v2: tracks the wire's most recent on-path projection
        # arclength on _polyline. When the wire goes off-path, the routed
        # d_corr uses this as the rejoin reference point. Persists across
        # invalidate() calls (steps); reset only in reset().
        self._last_on_path_arclen = 0.0

        # RL_IMPROV_8 v2 (state-machine + radius-aware) — episode state.
        #   _current_branch_idx: idx into _branches_tuple. Updated by
        #     update_branch_state() on committed junction crossings.
        #   _on_planned_path: True while wire is on the planned route.
        #     Flips to False when cross-track exceeds local-radius-aware
        #     tolerance; flips back True on cross-track return.
        self._current_branch_idx = None
        self._on_planned_path = True

        # Episode-precomputed path topology:
        #   _branch_radii: tuple of per-branch per-point radii arrays.
        #     Mirrors _branch_polylines / _branch_cumlens.
        #   _path_branch_idx_set: O(1) "is this branch on-path?" lookup.
        #   _path_branch_sequence: ordered list of (start_arc, end_arc,
        #     branch_idx) describing which branch the planned-path
        #     polyline lies on at each arclength range.
        #   _path_branch_sequence_with_junctions: list of (j_arc,
        #     prev_branch_idx, next_branch_idx) for forward/backward
        #     junction-crossing detection.
        #   _junction_off_path_candidates: dict {junction_arclen: list of
        #     branch_idx that connect to that junction but are NOT on the
        #     path}. Used to disambiguate which sister branch the wire
        #     entered when going off-path.
        self._branch_radii = ()
        self._path_branch_idx_set = set()
        self._path_branch_sequence = []
        self._path_branch_sequence_with_junctions = []
        self._junction_off_path_candidates = {}

        # RL_IMPROV_8 §39 (Plan v5) — off-path drift shaping + state-machine
        # daughter-commit events.
        #   _divergence_wrong_branch_arc: wire's projection arclength on the
        #     wrong branch at the moment the state machine flipped off-path.
        #     get_off_path_arc_since_divergence() returns
        #     current_arc - _divergence_wrong_branch_arc.
        #   _divergence_wrong_branch_idx: which wrong branch the divergence
        #     was captured on (so we can detect "wire committed to another
        #     wrong branch" and reset the baseline).
        #   _daughter_commit_events: per-step queue of (j_arc, +1|-1)
        #     emitted by update_branch_state() when the state machine
        #     commits at a daughter fork. Drained by env5 once per step.
        #   _committed_forks: per-episode latched list of junction arcs
        #     that have already emitted a commit event. Prevents re-fires.
        self._divergence_wrong_branch_arc = 0.0
        self._divergence_wrong_branch_idx = None
        self._daughter_commit_events = []
        # Plan v9 Change 1: set of (round(j_arc,3), +1) tuples. Only the +1
        # correct-commit is latched (at most once per fork per episode);
        # the -0.05 wrong-commit reward is never latched, so repeated wrong
        # commits at the same fork keep firing.
        self._committed_forks = set()
        # Plan v9 (reward-only) — per-divergence latch for the -0.05
        # deep-wrong-commit (mirrors reset()).
        self._wrong_commit_fired = False

        # Plan v5 (Tier 1) — Markov-completing state exposed in LocalGuidance.
        #   _trunk_branch_idx: cached at reset from _path_branch_sequence[0].
        #     LocalGuidance feature: is_in_trunk = (current_branch == trunk).
        #   _target_daughter_branch_idx: cached at reset from
        #     _path_branch_sequence[-1]. The path's terminal branch IS the
        #     target daughter. Feature: is_on_target_daughter.
        #   _ostium_branch_idx: cached at reset from _path_branch_sequence[-2]
        #     — the LAST on-path bridge before the target daughter (the
        #     commit fork: (11) for RCCA/RVA, (0) for LCCA, (18) for LVA).
        #     Target-relative: the SAME physical (11) is the ostium for an
        #     RCCA grader but a wrong-branch for an LCCA grader. Feature:
        #     is_at_ostium. None when the path has no distinct bridge
        #     (trunk → daughter directly, < 2 segments).
        #   _n_correct_commits / _n_wrong_commits: counters incremented in
        #     _maybe_emit_junction_commit. Per-episode totals exposed as
        #     normalized features so the agent knows what it has and
        #     hasn't accomplished at each fork.
        self._trunk_branch_idx = None
        self._target_daughter_branch_idx = None
        self._ostium_branch_idx = None
        self._n_correct_commits = 0
        self._n_wrong_commits = 0

        # Per-step cache (typing removed for Python 3.8 compat)
        self._tip_vessel_cs = None
        self._projection = None
        self._nearest_branch = None
        self._is_on_correct_branch = None
        self._is_on_correct_path = None
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
        self._build_branching_graph()  # RL_IMPROV_8 v2: BP graph for routed d_corr
        self._build_path_branch_sequence()  # state-machine: arclength -> branch
        # Reset hysteresis state per episode so a new vessel/path starts
        # with a fresh branch-membership evaluation.
        self._stable_on_branch = None
        self._pending_flip_count = 0
        self._stable_on_path = None
        self._pending_path_flip_count = 0
        self._last_on_path_arclen = 0.0  # wire starts at insertion = arclength 0
        # State machine: wire starts on the first segment of the planned
        # path (typically the trunk for high-insert configs). on_planned_path
        # starts True; the very first update_branch_state() call will flip
        # it False if the actual tip happens to be cross-track-far.
        self._on_planned_path = True
        if self._path_branch_sequence:
            self._current_branch_idx = self._path_branch_sequence[0][2]
        else:
            self._current_branch_idx = None
        # Plan v5 — reset divergence tracking and per-episode commit latch.
        # Plan v9 Change 1 — MUST be a set() (matches __init__): the
        # +1 junction-commit latch does `_committed_forks.add((j_arc,+1))`.
        # A previous bug reset this to [] here, so every +1 commit raised
        # 'list' object has no attribute 'add' (swallowed by env5's
        # try/except around update_branch_state) and NO +1 ever recorded.
        self._divergence_wrong_branch_arc = 0.0
        self._divergence_wrong_branch_idx = None
        self._committed_forks = set()
        # Plan v5 (Tier 1) — reset commit counters per episode; cache the
        # trunk and target-daughter branch indices once the path-branch
        # sequence is built (first segment = trunk, last segment = target).
        self._n_correct_commits = 0
        self._n_wrong_commits = 0
        # Plan v9 (reward-only) — per-divergence latch for the -0.05
        # wrong-commit. Set True when a deep wrong-commit fires; reset on
        # return on-path so a later genuine deep excursion can fire again.
        self._wrong_commit_fired = False
        if self._path_branch_sequence:
            self._trunk_branch_idx = self._path_branch_sequence[0][2]
            self._target_daughter_branch_idx = self._path_branch_sequence[-1][2]
            # Ostium = on-path branch immediately before the target daughter
            # (the final pre-daughter commit fork). None if no distinct bridge.
            if len(self._path_branch_sequence) >= 2:
                self._ostium_branch_idx = self._path_branch_sequence[-2][2]
            else:
                self._ostium_branch_idx = None
        else:
            self._trunk_branch_idx = None
            self._target_daughter_branch_idx = None
            self._ostium_branch_idx = None
        self.invalidate()

    def invalidate(self) -> None:
        """Mark cache as stale.  Call at the start of each env step."""
        self._tip_vessel_cs = None
        self._projection = None
        self._nearest_branch = None
        self._is_on_correct_branch = None
        self._is_on_correct_path = None
        self._dist_to_wrong_entry = None
        self._dist_to_correct_entry = None
        self._closest_wrong_entry_coords = None
        self._closest_correct_entry_coords = None
        # Plan v5 — per-step event queue drained by env5 once per step.
        self._daughter_commit_events = []

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

        # RL_IMPROV_8 v2: per-branch radii + on-path index set.
        #   _branch_radii[i] is the per-point radii array of branches[i],
        #   or None if that branch has no radii data.
        radii_list = []
        for branch in self._branches_tuple:
            r = getattr(branch, "radii", None)
            if r is not None and len(r) > 0:
                radii_list.append(np.asarray(r, dtype=float))
            else:
                radii_list.append(None)
        self._branch_radii = tuple(radii_list)

        path_set = self.pathfinder.path_branch_set if self.pathfinder is not None else set()
        self._path_branch_idx_set = {
            i for i, b in enumerate(self._branches_tuple) if b in path_set
        }

        # RL_IMPROV_8: per-branch on-path masks. For each branch, compute
        # which of its centerline indices are within ON_PATH_TOLERANCE_MM
        # of the planned polyline. Off-path branches get all-False masks.
        # On-path branches get partial masks reflecting which indices the
        # path actually uses (e.g. for the trunk, only indices up to the
        # exit point are True).
        self._branch_on_path_masks = {}
        path_set = self.pathfinder.path_branch_set if self.pathfinder is not None else set()
        if len(self._polyline) > 0:
            for branch, coords in zip(self._branches_tuple, self._branch_polylines):
                mask = np.zeros(len(coords), dtype=bool)
                if branch in path_set:
                    # O(n_branch * n_polyline). n,m < 400 so < 160k ops, fine
                    # at episode reset (called once per episode).
                    for i in range(len(coords)):
                        d = float(np.min(np.linalg.norm(self._polyline - coords[i], axis=1)))
                        mask[i] = d < ON_PATH_TOLERANCE_MM
                self._branch_on_path_masks[id(branch)] = mask
        else:
            for branch in self._branches_tuple:
                self._branch_on_path_masks[id(branch)] = np.zeros(0, dtype=bool)

    def _build_branching_graph(self) -> None:
        """RL_IMPROV_8 v2: build the BP-to-BP graph and precompute graph
        distances from every BP to the nearest BP on the planned path.

        Used by ``get_routed_d_corr_to_next_daughter_entry()`` so a wire
        on a wrong sister branch can compute its actual graph-route
        distance back to the planned path (instead of misleading 3D
        Euclidean which can be small while the wire physically has to
        retrace through bifurcations).

        Cost: O(B^2 + E log B) at reset where B = #branches, E = #edges.
        Both <30 in our dataset → microseconds.
        """
        self._branch_to_bp_pair = {}
        self._bp_adjacency = {}
        self._path_bp_set = set()
        self._bp_to_path_dist = {}
        self._bp_to_path_arclen = {}

        vessel_tree = self.intervention.vessel_tree
        if (vessel_tree.branching_points is None
                or len(vessel_tree.branching_points) == 0
                or len(self._polyline) < 2):
            return

        # Map each branch to the BP at its first endpoint and the BP at its
        # last endpoint (matched by 3D proximity). Some branches may have
        # only one or zero matched endpoints (boundary branches near
        # insertion/leaves).
        bps = list(vessel_tree.branching_points)
        for branch, coords in zip(self._branches_tuple, self._branch_polylines):
            if len(coords) == 0:
                self._branch_to_bp_pair[id(branch)] = (None, None)
                continue
            first_pt = np.asarray(coords[0])
            last_pt = np.asarray(coords[-1])
            bp_first = bp_last = None
            best_first_d = best_last_d = float("inf")
            for bp in bps:
                bp_xyz = np.asarray(bp.coordinates)
                d_first = float(np.linalg.norm(bp_xyz - first_pt))
                d_last = float(np.linalg.norm(bp_xyz - last_pt))
                if d_first < best_first_d and d_first < 2.0:
                    best_first_d = d_first
                    bp_first = bp
                if d_last < best_last_d and d_last < 2.0:
                    best_last_d = d_last
                    bp_last = bp
            self._branch_to_bp_pair[id(branch)] = (bp_first, bp_last)

        # Build BP adjacency: each branch connects its two endpoint BPs
        # with an edge of weight = branch length.
        for i_branch, (branch, cumlen) in enumerate(
            zip(self._branches_tuple, self._branch_cumlens)
        ):
            pair = self._branch_to_bp_pair.get(id(branch), (None, None))
            if pair[0] is None or pair[1] is None or pair[0] is pair[1]:
                continue
            length = float(cumlen[-1]) if len(cumlen) > 0 else 0.0
            self._bp_adjacency.setdefault(pair[0], []).append(
                (pair[1], branch, length)
            )
            self._bp_adjacency.setdefault(pair[1], []).append(
                (pair[0], branch, length)
            )

        # Path-BPs: BPs within 5 mm of any path point. Their graph distance
        # is 0 and they "belong" to the path at their nearest path arclength.
        for bp in bps:
            bp_xyz = np.asarray(bp.coordinates)
            dists = np.linalg.norm(self._polyline - bp_xyz, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < 5.0:
                self._path_bp_set.add(bp)
                self._bp_to_path_dist[bp] = 0.0
                self._bp_to_path_arclen[bp] = float(self._cumlen[idx])

        # Dijkstra from path BPs (sources) outward. Each off-path BP gets
        # the shortest graph distance to the nearest path-BP.
        import heapq
        pq = []
        counter = 0
        for bp in self._path_bp_set:
            heapq.heappush(pq, (0.0, counter, bp))
            counter += 1
        while pq:
            dist, _, bp = heapq.heappop(pq)
            if dist > self._bp_to_path_dist.get(bp, float("inf")):
                continue
            for other_bp, _branch, length in self._bp_adjacency.get(bp, []):
                new_dist = dist + length
                if new_dist < self._bp_to_path_dist.get(other_bp, float("inf")):
                    self._bp_to_path_dist[other_bp] = new_dist
                    # Inherit the rejoin arclength from the source path BP.
                    if bp in self._bp_to_path_arclen:
                        self._bp_to_path_arclen[other_bp] = self._bp_to_path_arclen[bp]
                    heapq.heappush(pq, (new_dist, counter, other_bp))
                    counter += 1

    def _build_path_branch_sequence(self) -> None:
        """RL_IMPROV_8 v2 — for the state machine, precompute which on-path
        branch the planned-path polyline lies on at each arclength range.

        For each polyline point, find the on-path branch whose centerline
        is geometrically closest (min cross-track). Adjacent polyline
        points usually agree; transitions mark on-path junctions where the
        path threads from one branch into the next (e.g. trunk → bridge(11)
        → RVA daughter).

        Output (set on self):
          _path_branch_sequence: list of (start_arc, end_arc, branch_idx).
            Concatenation covers [0, total_length]. branch_idx is in
            _branches_tuple coordinate.
          _path_branch_sequence_with_junctions: list of (j_arc, prev_idx,
            next_idx) describing the transitions between adjacent entries
            of _path_branch_sequence. Used by update_branch_state() to
            detect forward/backward junction crossings.

        Cost: O(P * B) where P = #polyline points (~600) and B = #branches
        (~25). At reset only — < 15k ops, sub-millisecond.
        """
        self._path_branch_sequence = []
        self._path_branch_sequence_with_junctions = []
        if (len(self._polyline) < 2
                or len(self._branches_tuple) == 0
                or len(self._path_branch_idx_set) == 0):
            return

        # For each polyline point, find which on-path branch is closest.
        # We project onto each on-path branch's centerline and pick the
        # branch with min perpendicular distance.
        on_path_idxs = sorted(self._path_branch_idx_set)
        point_branch = []
        for pt in self._polyline:
            best_idx = on_path_idxs[0]
            best_d = float("inf")
            for idx in on_path_idxs:
                poly = self._branch_polylines[idx]
                cum = self._branch_cumlens[idx]
                if len(poly) < 2:
                    continue
                r = project_onto_polyline(pt, poly, cum)
                if r.cross_track_dist < best_d:
                    best_d = r.cross_track_dist
                    best_idx = idx
            point_branch.append(best_idx)

        # Smoothing: if a single point disagrees with both its neighbours,
        # snap it to match. Avoids spurious transitions at near-equal
        # cross-track tiebreaks.
        for i in range(1, len(point_branch) - 1):
            if (point_branch[i - 1] == point_branch[i + 1]
                    and point_branch[i] != point_branch[i - 1]):
                point_branch[i] = point_branch[i - 1]

        # Compress runs into (start_arc, end_arc, branch_idx) entries.
        cur = point_branch[0]
        seg_start = float(self._cumlen[0])
        for i in range(1, len(point_branch)):
            if point_branch[i] != cur:
                seg_end = float(self._cumlen[i])
                self._path_branch_sequence.append((seg_start, seg_end, cur))
                self._path_branch_sequence_with_junctions.append(
                    (seg_end, cur, point_branch[i])
                )
                seg_start = seg_end
                cur = point_branch[i]
        # Final segment to end of polyline
        self._path_branch_sequence.append(
            (seg_start, float(self._cumlen[-1]), cur)
        )

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
        # RL_IMPROV_8: also collect daughter-only entries — junctions that
        # are real forks (have at least one off-path connected branch).
        # Stored together with their 3D vessel-CS coords so the 3D Euclidean
        # d_corr can find the next entry's *physical* position.
        daughter_entries = []  # list of (arclength, coord)
        if len(self._polyline) > 1 and len(self._cumlen) > 0:
            for bp in vessel_tree.branching_points:
                bp_xyz = np.asarray(bp.coordinates, dtype=float)
                dists = np.linalg.norm(self._polyline - bp_xyz, axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] < 5.0:
                    arclen = float(self._cumlen[idx])
                    junction_arclengths.append(arclen)
                    # Daughter-entry filter: at least one connected branch
                    # is on-path AND at least one is off-path (real fork,
                    # not just a continuation).
                    on_path_count = sum(
                        1 for b in bp.connections if b in path_set
                    )
                    off_path_count = len(bp.connections) - on_path_count
                    if on_path_count >= 1 and off_path_count >= 1:
                        daughter_entries.append((arclen, bp_xyz.copy()))
        self._path_junction_arclengths = (
            np.array(sorted(junction_arclengths))
            if junction_arclengths
            else np.empty(0)
        )
        if daughter_entries:
            daughter_entries.sort(key=lambda t: t[0])
            self._path_daughter_arclengths = np.array(
                [a for a, _ in daughter_entries]
            )
            self._path_daughter_coords = np.array(
                [c for _, c in daughter_entries]
            )
        else:
            self._path_daughter_arclengths = np.empty(0)
            self._path_daughter_coords = np.empty((0, 3))

        # RL_IMPROV_8 v2: per-junction off-path candidate branches. When the
        # wire goes laterally off the planned path near a junction, pick the
        # nearest off-path branch from THIS junction's connected sister
        # branches (typically 1-2) instead of running per-branch projection
        # over all 25 branches in the tree.
        self._junction_off_path_candidates = {}
        if len(self._polyline) > 1 and len(self._cumlen) > 0:
            for bp in vessel_tree.branching_points:
                bp_xyz = np.asarray(bp.coordinates, dtype=float)
                dists = np.linalg.norm(self._polyline - bp_xyz, axis=1)
                idx = int(np.argmin(dists))
                if dists[idx] >= 5.0:
                    continue
                arclen = float(self._cumlen[idx])
                # Off-path candidates: connections of this BP whose branch
                # is NOT on the path.
                off_path_branch_idxs = []
                for connected_branch in bp.connections:
                    if connected_branch in path_set:
                        continue
                    # Find this branch's index in _branches_tuple (identity
                    # comparison — branches are reused post-consolidation).
                    for i, b in enumerate(self._branches_tuple):
                        if b is connected_branch:
                            off_path_branch_idxs.append(i)
                            break
                if off_path_branch_idxs:
                    self._junction_off_path_candidates[arclen] = off_path_branch_idxs

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

    def is_on_correct_path(self) -> bool:
        """RL_IMPROV_8 v3 — state-machine flag AND on-path-mask check.

        After ``update_branch_state()`` has run for this step, the
        ``_on_planned_path`` flag IS the primary answer. The state machine
        bakes in 10 mm arclength hysteresis on junction crossings, and the
        radius-aware cross-track tolerance prevents in-lumen-drift
        flickering.

        BUG-A fix (re-introduced from v1 plan): also verify the tip's
        nearest centerline-index on its current branch is inside the path-
        used range of that branch. Reason: branches like ``(0).mrk`` are
        on-path but only a *portion* of their centerline is actually used
        by the planned route. The state machine commits to ``(0).mrk`` via
        the trunk-top junction crossing without checking whether the
        wire's lateral position is on the path-used segment of (0). The
        wedge cluster at vessel-CS (~18, 37, 375) sits on (0)'s OFF-path
        lower indices 0..20 (path uses only indices 22..30), and was
        misclassified as on-path. Now: if the tip's nearest centerline
        index on the committed branch has ``mask[i] = False``, override
        the state-machine flag and return False.

        If ``update_branch_state()`` has NOT yet been called this step
        (e.g. during super().step() before env5's call site), the answer
        reflects the END of the previous step — fine for observation /
        heuristic pre-action lookups.
        """
        if self._is_on_correct_path is not None:
            return self._is_on_correct_path
        result = bool(self._on_planned_path)
        if result:
            result = self._tip_on_path_used_segment()
        self._is_on_correct_path = result
        return result

    def _tip_on_path_used_segment(self) -> bool:
        """Check (c): tip's nearest index on _current_branch_idx must be
        True in that branch's on-path mask. Returns True if mask is empty
        (no centerline data) or branch index is out of range — fails open
        to avoid false negatives when the precompute has nothing to say.
        """
        if self._current_branch_idx is None:
            return True
        idx = self._current_branch_idx
        if idx >= len(self._branches_tuple):
            return True
        branch = self._branches_tuple[idx]
        mask = self._branch_on_path_masks.get(id(branch))
        if mask is None or len(mask) == 0:
            return True
        coords = self._branch_polylines[idx]
        if len(coords) == 0:
            return True
        tip = self.get_tip_vessel_cs()
        nearest_idx = int(np.argmin(np.linalg.norm(coords - tip, axis=1)))
        if nearest_idx >= len(mask):
            return True
        return bool(mask[nearest_idx])

    def get_planned_path_tangent_at(self, s: float,
                                     lookahead_mm: float = 0.0) -> np.ndarray:
        """Unit tangent of the planned polyline at arclength
        ``s + lookahead_mm``.

        Used by per-daughter heuristic policies (Phase C — in-daughter
        advance) for closed-loop J-tip alignment with the local centerline
        direction. Look-ahead lets the policy aim for "where the wire is
        going" so torsional propagation has time to land before the wire
        arrives.

        Returns the zero vector if the polyline is degenerate or
        ``s_query`` is out of range.
        """
        if len(self._polyline) < 2 or len(self._cumlen) < 2:
            return np.zeros(3)
        s_query = float(np.clip(s + lookahead_mm, 0.0, self._total_length))
        # Find segment containing s_query
        idx = int(np.searchsorted(self._cumlen, s_query) - 1)
        idx = max(0, min(idx, len(self._polyline) - 2))
        d = self._polyline[idx + 1] - self._polyline[idx]
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return np.zeros(3)
        return d / n

    def get_path_junctions(self):
        """Ordered list of ``(arclength, prev_branch_name, next_branch_name)``
        triples for each on-path junction the planned polyline threads
        through. Used by per-daughter heuristic policies to schedule
        phase activations along the planned path's arclength axis (more
        robust than Euclidean tip-to-junction-node distance, which
        misfires when the wire's lateral transit is offset from the node).
        """
        out = []
        for j_arc, prev_idx, next_idx in self._path_branch_sequence_with_junctions:
            prev_name = ""
            next_name = ""
            try:
                prev_name = getattr(self._branches_tuple[prev_idx], "name", "") or ""
            except Exception:
                pass
            try:
                next_name = getattr(self._branches_tuple[next_idx], "name", "") or ""
            except Exception:
                pass
            out.append((float(j_arc), prev_name, next_name))
        return out

    def get_local_radius(self) -> float:
        """Vessel radius at the wire's nearest centerline point on the
        current branch (state-machine determined). Falls back to a default
        if the current branch has no radii data.

        Clamped to [MIN_RADIUS_FLOOR_MM, MAX_RADIUS_CEILING_MM] so the
        derived tolerance can't get too small (jitter false-positive) or
        too large (hides actual wedges)."""
        if self._current_branch_idx is None:
            return DEFAULT_RADIUS_MM
        idx = self._current_branch_idx
        if idx >= len(self._branch_polylines):
            return DEFAULT_RADIUS_MM
        coords = self._branch_polylines[idx]
        radii = self._branch_radii[idx]
        if radii is None or len(radii) == 0 or len(coords) == 0:
            return DEFAULT_RADIUS_MM
        tip = self.get_tip_vessel_cs()
        nearest_idx = int(np.argmin(np.linalg.norm(coords - tip, axis=1)))
        if nearest_idx >= len(radii):
            return DEFAULT_RADIUS_MM
        return float(np.clip(
            radii[nearest_idx],
            MIN_RADIUS_FLOOR_MM,
            MAX_RADIUS_CEILING_MM,
        ))

    def get_local_tolerance(self) -> float:
        """Cross-track tolerance derived from local vessel radius:
        ``max(MIN_TOLERANCE_MM, K_RADIUS * local_radius)``."""
        return max(MIN_TOLERANCE_MM, K_RADIUS * self.get_local_radius())

    def _branch_at_arclength(self, s: float) -> int:
        """Look up which on-path branch the planned-path polyline lies on
        at arclength ``s``. Returns a branch index in ``_branches_tuple``,
        or the first segment's branch if no entry contains ``s``."""
        if not self._path_branch_sequence:
            return 0
        for start_arc, end_arc, branch_idx in self._path_branch_sequence:
            if start_arc <= s <= end_arc:
                return branch_idx
        return self._path_branch_sequence[-1][2]

    def _pick_off_path_branch(self, s: float) -> int:
        """When the wire goes laterally off the planned path, identify
        which sister branch it entered. Use the precomputed off-path
        candidates at the closest path-junction, picked by min cross-track
        among the (typically 1-2) candidate branches.

        Falls back to the legacy all-branch ``get_nearest_branch()`` if no
        junction is near or no candidates are listed.
        """
        # Find the path-junction whose arclength is closest to s.
        if self._junction_off_path_candidates:
            closest_arc = min(
                self._junction_off_path_candidates.keys(),
                key=lambda a: abs(a - s),
            )
            candidates = self._junction_off_path_candidates.get(closest_arc, [])
        else:
            candidates = []
        if not candidates:
            # Fallback: legacy all-branch projection.
            nearest = self.get_nearest_branch()
            if nearest is None:
                return self._current_branch_idx if self._current_branch_idx is not None else 0
            for i, b in enumerate(self._branches_tuple):
                if b is nearest:
                    return i
            return 0
        # Pick by min cross-track over the small candidate set.
        tip = self.get_tip_vessel_cs()
        best_idx = candidates[0]
        best_d = float("inf")
        for i in candidates:
            poly = self._branch_polylines[i]
            cum = self._branch_cumlens[i]
            if len(poly) < 2:
                continue
            r = project_onto_polyline(tip, poly, cum)
            if r.cross_track_dist < best_d:
                best_d = r.cross_track_dist
                best_idx = i
        return best_idx

    def update_branch_state(self) -> None:
        """RL_IMPROV_8 v2 — once-per-step state machine update.

        Drives _current_branch_idx and _on_planned_path from this step's
        projection on the planned polyline. Uses committed events:
          - 10 mm arclength past a junction → flip current_branch
          - cross-track > local-radius-aware tolerance → off-path
          - cross-track <= tolerance while off-path → re-derive on-path

        Must be called by env5.step() ONCE per step, after super().step()
        has populated the projection cache (the wrong-branch detector
        check immediately following will then read the fresh state).
        """
        if self._total_length < 1e-6 or not self._path_branch_sequence:
            return
        proj = self.get_projection()
        s = float(proj.s)
        ct = float(proj.cross_track_dist)
        # Tolerance based on local radius of the CURRENT branch (whichever
        # the state machine currently believes the wire is on).
        tol = self.get_local_tolerance()

        if self._on_planned_path:
            # Forward / backward junction crossings — one-sided commitment
            # with 10 mm hysteresis. Bug fix vs RL_IMPROV_8 v3: the previous
            # condition required prev_s < j_arc-10 AND s >= j_arc+10 in a
            # single step (a 20 mm leap), which is unreachable at normal
            # 0.5-1 mm/step advance. Fires now whenever the current
            # projection arclength is committed past the junction by 10 mm,
            # held by hysteresis (a 20 mm dead-band on the other side
            # prevents bouncing back without genuine retraction).
            for j_arc, prev_idx, next_idx in self._path_branch_sequence_with_junctions:
                if (s >= j_arc + COMMIT_HYSTERESIS_MM
                        and self._current_branch_idx == prev_idx):
                    self._current_branch_idx = next_idx
                    # Plan v9 — correct on-path commit. Emit +1 at EVERY
                    # on-path junction crossing (top (2)->(0), bridge
                    # (0)->(11), daughter fork (11)->RCCA), not only at
                    # real daughter forks. Rewards each correct
                    # navigational decision along the path: "commit to
                    # (11), don't dwell in (0)" and "turn into (0) at the
                    # top junction, don't drift toward LVA".
                    self._maybe_emit_junction_commit(j_arc)
                elif (s < j_arc - COMMIT_HYSTERESIS_MM
                        and self._current_branch_idx == next_idx):
                    self._current_branch_idx = prev_idx
            # Off-path detection — cross-track exceeded radius-aware tolerance.
            # NOTE: this branch-classification logic is UNCHANGED from the
            # pre-Plan-v9 baseline. is_on_correct_path() / _on_planned_path /
            # _current_branch_idx must stay byte-for-byte identical to the
            # baseline because the heuristic policy READS is_on_correct_path()
            # to gate its RCCA-specific steering — any change here alters the
            # heuristic's ACTIONS, not just the reward. (A debounce attempt
            # here broke threading to 0% RCCA; reverted. The flicker-driven
            # spurious -0.05 is now handled purely on the REWARD side below,
            # via off-path arc magnitude, without touching this classifier.)
            if ct > tol:
                self._on_planned_path = False
                self._current_branch_idx = self._pick_off_path_branch(s)
                self._capture_divergence_point(s)
                dj_arc = self._nearest_daughter_junction_arc(s)
                if dj_arc is not None:
                    self._last_on_path_arclen = dj_arc
            # BUG-A fix: also flip off-path if tip is on the off-path
            # section of a shared branch (e.g. (0).mrk lower indices for
            # non-LCCA targets). Keeps the state machine consistent with
            # is_on_correct_path()'s mask-aware verdict.
            elif not self._tip_on_path_used_segment():
                self._on_planned_path = False
                self._current_branch_idx = self._pick_off_path_branch(s)
                self._capture_divergence_point(s)
                dj_arc = self._nearest_daughter_junction_arc(s)
                if dj_arc is not None:
                    self._last_on_path_arclen = dj_arc
        else:
            # On-path return (UNCHANGED from baseline): cross-track within
            # tolerance AND tip on the path-used segment of the arclength-
            # implied branch.
            if ct <= tol:
                candidate_idx = self._branch_at_arclength(s)
                prev_idx = self._current_branch_idx
                self._current_branch_idx = candidate_idx
                if self._tip_on_path_used_segment():
                    self._on_planned_path = True
                    self._divergence_wrong_branch_arc = 0.0
                    self._divergence_wrong_branch_idx = None
                else:
                    self._current_branch_idx = prev_idx
            elif (self._divergence_wrong_branch_idx is not None
                    and self._current_branch_idx != self._divergence_wrong_branch_idx):
                self._capture_divergence_point(s)

        # Plan v9 (reward-only) — wrong-commit -0.05 on a GENUINE deep
        # divergence, NOT on the fork-mouth flicker. Decoupled from the
        # is_on_correct_path() classifier above (which the heuristic reads).
        # Fires once per divergence event, when the wire has traveled
        # WRONG_COMMIT_ARC_MM along the wrong branch since diverging — a
        # wedged-at-the-mouth wire (shallow off-arc) never triggers it,
        # so threaded episodes are no longer hammered ~100x. Repeats across
        # SEPARATE deep excursions (the latch resets on return on-path).
        if not self._on_planned_path:
            if not self._wrong_commit_fired:
                try:
                    off_arc = self.get_off_path_arc_since_divergence()
                except Exception:
                    off_arc = 0.0
                if off_arc >= WRONG_COMMIT_ARC_MM:
                    jc_arc = self._nearest_junction_arc(s)
                    if jc_arc is not None:
                        self._maybe_emit_junction_commit(jc_arc, -1)
                    self._wrong_commit_fired = True
        else:
            # Back on-path → re-arm so the next genuine deep wrong-commit
            # can fire again.
            self._wrong_commit_fired = False

    # ------------------------------------------------------------------
    # Plan v5 helpers — daughter-commit events + off-path divergence
    # ------------------------------------------------------------------
    def _maybe_emit_junction_commit(self, j_arc: float, sign: int = +1) -> None:
        """Plan v9 — emit a fork-commit reward at ANY on-path junction
        (top (2)->(0), bridge (0)->(11), daughter fork (11)->RCCA), NOT
        gated by the daughter-fork filter.

        Asymmetric, partly-latched scheme:
          sign=+1 (correct forward commit onto the next on-path branch):
            emit +1.0; latch (round(j_arc,3), +1) so each junction fires
            at most once per episode. On a clean RCCA thread this gives
            +3 (top + bridge + daughter).
          sign=-1 (wrong off-path commit at this junction): emit -0.05;
            NEVER latched so every wrong commit re-fires -0.05. The
            state-machine 10mm commit hysteresis prevents per-step
            thrashing.

        Rationale: dense per-junction positive signal for each correct
        navigational decision; a small repeating -0.05 for wrong turns
        that keeps wrong-try-then-correct episodes net positive (vs the
        old symmetric +-1 which made the policy freeze at forks).
        """
        j_key = round(float(j_arc), 3)
        if sign > 0:
            if (j_key, +1) in self._committed_forks:
                return
            self._committed_forks.add((j_key, +1))
            self._daughter_commit_events.append((float(j_arc), 1.0))
            self._n_correct_commits += 1
        else:
            self._daughter_commit_events.append((float(j_arc), -0.05))
            self._n_wrong_commits += 1

    def _nearest_junction_arc(self, s: float):
        """Return the arclength of the on-path junction (ANY fork: top,
        bridge, or daughter) nearest to s, or None. Used to attribute a
        wrong-commit -0.05 to whichever fork the wire diverged at."""
        if len(self._path_junction_arclengths) == 0:
            return None
        idx = int(np.argmin(np.abs(self._path_junction_arclengths - s)))
        return float(self._path_junction_arclengths[idx])

    def _nearest_daughter_junction_arc(self, s: float):
        """Return the arclength of the daughter junction nearest to s,
        or None if no daughter junctions exist. Used by update_branch_state
        when emitting wrong-commit -1 events on cross-track off-path.
        """
        if len(self._path_daughter_arclengths) == 0:
            return None
        idx = int(np.argmin(np.abs(self._path_daughter_arclengths - s)))
        return float(self._path_daughter_arclengths[idx])

    def _capture_divergence_point(self, s: float) -> None:
        """Record the wire's arclength on its current wrong branch at the
        moment of divergence. The progress reward later reads this via
        get_off_path_arc_since_divergence() to penalize continued drift.
        """
        if (self._current_branch_idx is None
                or self._current_branch_idx >= len(self._branch_polylines)):
            self._divergence_wrong_branch_arc = 0.0
            self._divergence_wrong_branch_idx = None
            return
        coords = self._branch_polylines[self._current_branch_idx]
        cum = self._branch_cumlens[self._current_branch_idx]
        if len(coords) < 2:
            self._divergence_wrong_branch_arc = 0.0
        else:
            tip = self.get_tip_vessel_cs()
            r = project_onto_polyline(tip, coords, cum)
            self._divergence_wrong_branch_arc = float(r.s)
        self._divergence_wrong_branch_idx = self._current_branch_idx

    def get_off_path_arc_since_divergence(self) -> float:
        """Plan v5 — arc distance the wire has traveled along its current
        wrong branch since the state machine flipped off-path. Returns 0.0
        if currently on-path or no divergence baseline was captured. Used
        by ArcLengthProgress as the magnitude of negative off-path shaping.
        """
        if self._on_planned_path:
            return 0.0
        if (self._current_branch_idx is None
                or self._current_branch_idx >= len(self._branch_polylines)
                or self._divergence_wrong_branch_idx is None):
            return 0.0
        coords = self._branch_polylines[self._current_branch_idx]
        cum = self._branch_cumlens[self._current_branch_idx]
        if len(coords) < 2:
            return 0.0
        tip = self.get_tip_vessel_cs()
        r = project_onto_polyline(tip, coords, cum)
        return max(0.0, float(r.s) - self._divergence_wrong_branch_arc)

    def get_routed_d_corr_to_next_daughter_entry(self) -> float:
        """RL_IMPROV_8 v2: graph-route distance from tip to next daughter
        entry, accounting for sister-branch detours.

        - When tip is on-path (cross-track < CROSS_TRACK_TOLERANCE_MM and
          nearest branch in path_branch_set), returns the same value as
          ``get_arclength_to_next_daughter_entry()`` (fast case).
        - When tip is off-path (in a sister branch or wedge), computes:
              detour = (distance from tip along its current branch to the
                        nearest path-connected endpoint BP)
                     + (graph distance from that BP to the planned path)
              + (arclength from rejoin point to next daughter entry)

        The detour reflects the actual centerline traversal the wire would
        need to retrace+reroute to reach the next daughter — so a wire in
        a parallel sister branch reports an honest large d_corr instead of
        a misleadingly small 3D Euclidean.

        Returns ``float('inf')`` if no daughter entry is reachable.
        """
        # Fast path: state machine says we're on-path. Use arclength
        # along the planned path to next daughter. (RL_IMPROV_8 v2 — was
        # cross-track-threshold based; now uses the state-machine flag
        # which itself was set with radius-aware tolerance, removing
        # in-lumen flicker.)
        proj = self.get_projection()
        if self._on_planned_path:
            self._last_on_path_arclen = proj.s
            return self.get_arclength_to_next_daughter_entry()

        # Off-path: graph-routed detour. Use the state-machine's
        # _current_branch_idx (which was picked from the junction-local
        # off-path candidates) instead of per-step nearest-branch.
        if self._current_branch_idx is None or len(self._branches_tuple) == 0:
            return float("inf")
        branch_idx = self._current_branch_idx
        if branch_idx >= len(self._branch_polylines):
            return float("inf")
        coords = self._branch_polylines[branch_idx]
        bcumlen = self._branch_cumlens[branch_idx]
        if len(coords) < 2:
            return float("inf")
        tip = self.get_tip_vessel_cs()
        r = project_onto_polyline(tip, coords, bcumlen)
        dist_to_first = r.s
        dist_to_last = float(bcumlen[-1]) - r.s

        # Each endpoint BP has a precomputed graph distance to the planned
        # path. Detour = endpoint-along-branch + endpoint-to-path graph dist.
        current_branch_obj = self._branches_tuple[branch_idx]
        pair = self._branch_to_bp_pair.get(id(current_branch_obj), (None, None))
        candidates = []  # list of (total_detour, rejoin_arclen)
        if pair[0] is not None and pair[0] in self._bp_to_path_dist:
            total = dist_to_first + self._bp_to_path_dist[pair[0]]
            rejoin_s = self._bp_to_path_arclen.get(pair[0], self._last_on_path_arclen)
            candidates.append((total, rejoin_s))
        if pair[1] is not None and pair[1] in self._bp_to_path_dist:
            total = dist_to_last + self._bp_to_path_dist[pair[1]]
            rejoin_s = self._bp_to_path_arclen.get(pair[1], self._last_on_path_arclen)
            candidates.append((total, rejoin_s))

        if not candidates:
            # No graph route available — fall back to 3D Euclidean.
            return self.get_3d_dist_to_next_daughter_entry()

        detour, rejoin_arclen = min(candidates, key=lambda c: c[0])

        # Distance from rejoin point to next daughter ahead on the path.
        if len(self._path_daughter_arclengths) == 0:
            return detour
        ahead = self._path_daughter_arclengths[
            self._path_daughter_arclengths > rejoin_arclen
        ]
        if len(ahead) == 0:
            return detour
        return detour + (float(ahead[0]) - rejoin_arclen)

    def get_3d_dist_to_next_daughter_entry(self) -> float:
        """RL_IMPROV_8: 3D Euclidean distance from tip to the COORDINATE
        of the next daughter entry along the path.

        Replaces ``get_arclength_to_next_correct_entry()`` for cases where
        physical proximity matters (e.g. heuristic regime selection). The
        arclength version is misleading for laterally-wedged wires whose
        polyline projection lands at a junction's arclength even though
        the physical tip is tens of mm away.

        Returns ``float('inf')`` if no daughter entry is ahead of the tip.
        """
        if len(self._path_daughter_arclengths) == 0:
            return float("inf")
        proj = self.get_projection()
        ahead_mask = self._path_daughter_arclengths > proj.s
        if not np.any(ahead_mask):
            return float("inf")
        next_idx = int(np.argmax(ahead_mask))  # first True index
        next_coord = self._path_daughter_coords[next_idx]
        tip = self.get_tip_vessel_cs()
        return float(np.linalg.norm(tip - next_coord))

    def get_arclength_to_next_daughter_entry(self) -> float:
        """RL_IMPROV_8: Arclength along path to the next DAUGHTER entry
        (not every junction). Mirror of ``get_arclength_to_next_correct_entry()``
        but uses the smaller daughter-only set.

        Returns ``float('inf')`` if no daughter ahead.
        """
        if len(self._path_daughter_arclengths) == 0:
            return float("inf")
        proj = self.get_projection()
        ahead = self._path_daughter_arclengths[
            self._path_daughter_arclengths > proj.s
        ]
        if len(ahead) == 0:
            return float("inf")
        return float(ahead[0] - proj.s)

    def get_arclength_past_last_daughter_entry(self) -> float:
        """RL_IMPROV_8: Arclength past the most-recent DAUGHTER entry
        behind the tip (not every junction). Mirror of
        ``get_arclength_past_last_junction()`` but daughter-only.

        Used by env5 to fire +1 CORRECT_ENTRY_REWARD only when the wire
        commits past a real fork (not on every internal junction).
        """
        if len(self._path_daughter_arclengths) == 0:
            return 0.0
        proj = self.get_projection()
        behind = self._path_daughter_arclengths[
            self._path_daughter_arclengths <= proj.s
        ]
        if len(behind) == 0:
            return 0.0
        return float(proj.s - behind[-1])

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


# ============================================================================
# Plan v12 Stage 1 — multi-target heatup support.
# ============================================================================

def make_per_target_caches(
    pathfinders: Sequence,
    intervention,
    on_branch_flip_threshold: int = 5,
) -> List["PathProjectionCache"]:
    """Construct N independent PathProjectionCache instances — one per
    daughter target in Plan v12 Stage 1's multi-target heatup harvest.

    Each cache is bound to its own FixedPathfinder. After this call, each
    pathfinder's reset(target_coord3d=daughter_k_coord) must have already
    populated its path_points_vessel_cs; downstream env5 calls each cache's
    reset() to populate the per-instance polyline / _path_branch_sequence /
    _path_daughter_arclengths / _committed_forks state machine.

    Returns:
        List[PathProjectionCache] of length len(pathfinders). Every field
        in every cache is freshly allocated by PathProjectionCache.__init__
        — there is NO shared mutable state across instances (verified by
        assert_caches_independent()).

    Why a helper, not direct construction: the per-virtual-env state
    machine carries 8+ mutable fields (_committed_forks set, _wrong_commit_fired
    bool, _divergence_wrong_branch_arc/idx, _daughter_commit_events list,
    etc.). Future maintainers must NOT accidentally share these across the
    4 virtual envs in MultiTargetEnv5 — assert_caches_independent() is the
    paranoid backstop.
    """
    if len(pathfinders) == 0:
        raise ValueError("make_per_target_caches: pathfinders must be non-empty")
    caches = [
        PathProjectionCache(pf, intervention, on_branch_flip_threshold)
        for pf in pathfinders
    ]
    return caches


def assert_caches_independent(caches: Sequence["PathProjectionCache"]) -> None:
    """Smoke-test assertion: no two PathProjectionCache instances share
    mutable list/set/dict objects (would leak Plan v9 fork-reward bookkeeping
    across virtual envs and break the per-daughter `+1`/`-0.05` invariant).

    Raises AssertionError on first violation. Called by Stage 1 smoke test
    immediately after make_per_target_caches() to catch any future regression
    that introduces shared defaults.

    Checks every mutable field that participates in the per-virtual-env
    state machine (Decision #14):
      _committed_forks (set), _daughter_commit_events (list),
      _branch_on_path_masks (dict), _path_branch_idx_set (set),
      _path_branch_sequence (list), _path_branch_sequence_with_junctions (list),
      _junction_off_path_candidates (dict), _path_bp_set (set),
      _bp_adjacency (dict), _branch_to_bp_pair (dict),
      _bp_to_path_dist (dict), _bp_to_path_arclen (dict).
    """
    mutable_fields = [
        "_committed_forks", "_daughter_commit_events",
        "_branch_on_path_masks", "_path_branch_idx_set",
        "_path_branch_sequence", "_path_branch_sequence_with_junctions",
        "_junction_off_path_candidates", "_path_bp_set", "_bp_adjacency",
        "_branch_to_bp_pair", "_bp_to_path_dist", "_bp_to_path_arclen",
    ]
    for i in range(len(caches)):
        for j in range(i + 1, len(caches)):
            for fld in mutable_fields:
                a = getattr(caches[i], fld)
                b = getattr(caches[j], fld)
                assert a is not b, (
                    f"PathProjectionCache field '{fld}' is SHARED between "
                    f"cache[{i}] and cache[{j}] (id={id(a)}) — Plan v12 R15 "
                    f"violation; fork-reward bookkeeping would leak across "
                    f"virtual envs. Check PathProjectionCache.__init__ for "
                    f"a mutable-default kwarg or class-level attribute."
                )
