# RL Improv 8 — Changes Reference

All changes in this branch relative to `rl_improv_7` (which has §§1–13 of
RL_IMPROV_7_CHANGES.md applied, plus the heuristic / SOFA-restore pathway,
RVA/LCCA-side findings, and the §16 open question on moving the insertion
point higher up the trunk). Built against env5 + the path-aware
infrastructure introduced in that branch.

---

## 1. High-Insertion-Point Anchor (RL_IMPROV_7 §16)

### Context

RL_IMPROV_7 §16 proposed (but did not implement) replacing the
SOFA-state-restore mechanism with a simpler scheme: move the wire's
`insertion_position` further up the trunk so episodes start near bif2
without needing to capture/restore SOFA state. The motivating question
was whether the dominant Run-28 failure mode — the (80, 55, 395)
aortic-arch wedge that traps 40/50 LCCA+RCCA+RVA episodes (§15.4) —
is caused by the wire's torsional/curvature history accumulated while
threading 380 mm of trunk + arch, or by the local arch geometry at
that point.

### Reason

A "fresh" wire built directly at that height — like a doctor inserted
from the neck instead of the femoral artery — should, if the
wire-history hypothesis holds, wedge less often than a wire that has
been pushed through the entire arch curvature. The §16 prediction:

> If the wedge rate at (80, 55, 395) drops materially in the first 50
> episodes, the wire-history hypothesis is real. If it stays the same,
> geometry is the dominant factor and you've simplified the env without
> solving the core issue.

### Fix

**File:** [eve_bench/eve_bench/dualdevicenav.py](eve_bench/eve_bench/dualdevicenav.py)

Added `insertion_z: float = None` kwarg to `DualDeviceNav.__init__`. When
set, the wire's anchor and direction are derived from the longest-z-span
branch whose z-range contains the requested value (i.e. the trunk for
typical `insertion_z` values like 345). Two important details surfaced
during implementation:

1. **`branches[0]` is NOT the trunk** — `load_branches()` returns paren-less
   files (LCCA, LVA, RCCA, RVA, …) at "key 0" in unstable filesystem order,
   *not* the main trunk. The actual trunk in this dataset is "Centerline
   curve (2).mrk" at vessel-CS z range [79.4, 392.0] with 363 control
   points. Selecting the longest-span branch whose z-range contains
   `insertion_z` is robust to this load-order surprise. Saved as
   `reference_dualdevicenav_branches.md` in agent memory.

2. **Branch (2) coordinates are stored descending z** (first_pt at z=392,
   last_pt at z=79). The forward-tangent of `coords[i+1] - coords[i]`
   therefore points *toward femoral*. Sign-flip if `tangent[2] < 0` so the
   wire pushes UP toward bif2.

### CLI plumbing

**File:** [training _scripts/DualDeviceNav_train.py](training _scripts/DualDeviceNav_train.py)

Added `--insertion_z` argparse flag, passed through to both training and
eval `DualDeviceNav()` constructors. Mutual-exclusion guard with
`--checkpoint_dir` (the two represent opposing strategies — checkpoint
restore replaces the wire state; anchoring high replaces the start
point).

---

## 2. Heuristic-Only Runner

### Context

The §16 A/B test only needed a heuristic-only run (no SAC training, no
eval, no replay buffer). The existing `DualDeviceNav_train.py` is too
heavy: spins up the full agent, runs heatup + 20M training steps, eval
loops, etc. `collect_sofa_checkpoints.py` had the right shape (heuristic
seeding + early exit) but its action function side-effects SOFA state
captures.

### Fix

**File (new):** [training _scripts/heuristic_only_run.py](training _scripts/heuristic_only_run.py)

Modelled on `collect_sofa_checkpoints.py`. Builds a `BenchAgentSynchron`
(networks allocated but never trained), creates a standard
`HeuristicActionFunctionFactory`, dispatches a branch-balanced schedule
via `agent.heuristic_seed(push_to_buffer=False)`, then exits. Worker
logs land in `<RESULTS_FOLDER>/<name>/diagnostics/logs_subprocesses/`
so `analyze_run28_branches.py` parses them without changes.

Run command (Git Bash):

```bash
python3 /opt/eve_training/training_scripts/heuristic_only_run.py \
    --env_version 5 -n env5_rl8_highinsert_50ep \
    --insertion_z 345 --episodes 50 -nw 16 -d cuda:0
```

---

## 3. High-Insertion A/B Test Results (Run env5_rl8_highinsert_50ep)

### Setup

50 heuristic-only episodes, 16 workers, `--insertion_z 345` (no
checkpoint restore). Trunk centerline at vessel-CS z=345 was located
at densified[119/685]=(61.45, 74.89, 345.23) with tangent
(-0.165, -0.290, +0.943) (positive z, toward bif2).

### Per-target outcomes (50/50 episodes parsed, run-28 categorisation)

| Target | n  | SUCCESS | WEDGE (≥580) | MID (200–580) | FOLD (<200) |
|--------|----|---------|--------------|---------------|-------------|
| LCCA   | 12 | 0       | 4            | 8             | 0           |
| LVA    | 13 | **4**   | 9            | 0             | 0           |
| RCCA   | 13 | 0       | 10           | 3             | 0           |
| RVA    | 12 | 0       | 12           | 0             | 0           |
| Total  | 50 | 4       | 35           | 11            | **0**       |

### Findings

- **4/50 successes (8%) vs baseline 2/174 (1.1%)** — 4x absolute
  improvement. All 4 successes are LVA, classifier-verified
  (distance to LVA centerline 0.03–0.05 mm, distances to other
  daughters ≥ 15 mm). The §15.8 LVA-only success bias is preserved.
- **Trunk-fold mode eliminated** — 0/50 episodes fold under 200
  steps, vs ~67% in baseline.
- **Aortic-arch wedge persists.** (80, 55, 395) ± 8 mm wedge rate:
  baseline ~24% of non-LVA episodes (32/130), this run **21.6%**
  (8/37). Statistically unchanged.
- **`entries_passed` reveals junction crossings:** 36/50 episodes
  gained ≥ 1 path-junction; the 8 arch-wedge episodes all have
  `entries_gained=0` (wire never reached bif2). Path-junction crossings
  cluster around LVA (4 successes have `gained=2`, RCCA/RVA targets
  that crossed bif2 mostly wedge at (60, 35, 390) inside a daughter).
- **Two RVA "near-success" episodes** (pid=312 ep=3, pid=654 ep=3,
  rewards −2.66 / −0.38) reached the LVA-RVA shared distal endpoint
  via the LVA route — the pathfinder routed the planned path through
  LVA because the target sample lay near the basilar-merger point at
  vessel-CS (14.8, 12.5, 572.0) which is shared between LVA and RVA.
  No genuine RCCA or RVA navigation occurred in this run.

### Verdict

**§16 wire-history hypothesis: FAILED.** Geometry is the dominant
factor at the (80, 55, 395) wedge zone. High-insertion is still useful
as a training-time speedup (eliminates trunk traversal), but it does
not crack the arch wedge.

---

## 4. Investigation — `is_on_correct_branch`, `d_corr_arc`, Euclidean d_corr

### Context

Run-28 forensic analysis claimed wedge episodes had `d_corr_arc=0`
and `on_branch=True` while the wire was wedged in the arch. Initial
hypothesis: the wire was 96 mm laterally off-path (a math error in my
manual coordinate transform). Re-parsing logs with the actual
`tracking3d_to_vessel_cs` transform gave the correct figures:

- Wedge tracking3d (76, 60, 394) ↔ vessel-CS centroid **(52.4, 49.0, 399.0)**.
- 13.2 mm laterally from trunk centerline (closest pt: trunk[7] at
  (52.91, 44.65, 386.52)).
- 13.2 mm from branch (18) (the trunk → LVA bridge).
- 7 mm past the trunk's top endpoint at z=391.95.
- ≥ 34 mm from any daughter centerline.

So the wedge tip is in the open lumen of the bif2 cavity, only ~13 mm
laterally off the trunk centerline — not 96 mm.

### Three bugs uncovered

1. **`is_on_correct_branch()` only checks branch-identity.**
   [pathcontext.py:319-354](eve/eve/util/pathcontext.py#L319-L354) calls
   `get_nearest_branch()` (per-branch perpendicular-distance argmin)
   and returns `pathfinder.is_branch_on_path(branch)`, which is just a
   set-membership check. **No position-on-branch check.** A wedge tip
   13 mm off the trunk centerline still has trunk as nearest-branch,
   trunk is in `path_branch_set`, so the method returns True. The
   off-branch counter never increments → `wrong_branch_timeout` never
   fires → episode runs to 600-step truncation.

2. **`d_corr_arc` is projection-arclength, not 3D distance.** Forensic
   trace of pid=274 ep=2 (RVA target, R=−31.83 wedge, min `d_corr_arc=0`):
   ```
   ep_step=350: tip vessel-CS = (47.55, 34.09, 392.56)
                d_to_trunk_top   = 1.10 mm
                d_to_LCCA[0]     = 31.48 mm
                d_to_bif2b (RVA) = 54.34 mm    ← actual ostium
                d_to_LVA[0]      = 37.54 mm
   ```
   The wire was AT the trunk-top junction. `d_corr_arc=0` because the
   wire's *projection arclength* on the planned polyline equals a
   junction's arclength — not because the wire is physically at the
   ostium. The polyline projection collapses 3D to a 1D scalar; cross-track
   distance is ignored.

3. **Multi-junction topology is collapsed.** For non-LVA targets the
   planned path threads multiple junctions:
   - **trunk → bridge(11) bridge entry** at z=384.7 (LCCA[0])
   - **bridge(11) → daughter** at z=416.2 (= RCCA[0] = RVA[0])
   - For LVA: trunk → trunk-top junction → bridge(18) → LVA-junction
     internal at z=430.1
   
   `_correct_branch_entries` and `_path_junction_arclengths`
   ([pathcontext.py:280-296](eve/eve/util/pathcontext.py#L280-L296))
   pick up *every* `vessel_tree.branching_point` within 5 mm of any
   path point, regardless of whether crossing it represents a
   meaningful daughter commit. The +1.0 entry reward fires on each.

---

## 5. Path-Aware On-Path Classification (v1)

### Fix

**File:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py)

New constants:
```python
ON_PATH_TOLERANCE_MM = 3.0
CROSS_TRACK_TOLERANCE_MM = 6.0
```

Added per-branch on-path mask in `_build_branch_index()`. For each
branch, compute a boolean mask over its centerline indices marking
which points are within `ON_PATH_TOLERANCE_MM` of the planned polyline.
Off-path branches get all-False masks. On-path branches get partial
masks reflecting which indices the path uses (e.g. for the trunk, only
the indices the planner threads — past the exit point indices are
False even though the trunk *as a branch* is in `path_branch_set`).

Added `is_on_correct_path()` method — three-check classifier:

```python
def is_on_correct_path(self) -> bool:
    """True iff THREE checks pass:
      (a) cross_track_dist from tip to _polyline < CROSS_TRACK_TOLERANCE_MM
      (b) nearest branch is in path_branch_set
      (c) tip's nearest centerline index on its nearest branch is True
          in that branch's on-path mask
    Hysteresis-debounced separately from is_on_correct_branch.
    """
```

Direction-awareness is already encoded: `project_onto_polyline()` returns
the perpendicular distance to the closest segment of the polyline,
which is equivalent to projecting onto the local path-tangent and
measuring lateral drift.

### Cross-track tolerance choice

The 13 mm wedge cross-track from §4 is roughly 2× the 6 mm threshold,
giving safety margin without false-positiving on normal SOFA jitter or
sub-mm centerline densification.

### Effect

For an arch-wedge wire at vessel-CS (52, 49, 399):
- (a) cross-track to planned-path polyline: 13–49 mm (depending on
  target's planned path) — fails 6 mm threshold → returns False
- (b) nearest branch (trunk) is in `path_branch_set` → True (legacy
  check still passes)
- (c) tip's nearest centerline index on trunk is `trunk[7]` at z=386.5;
  for a non-LVA path that uses bridge(11) starting from the LCCA junction,
  the trunk top indices 0–6 are NOT on-path → False

Off-path classification kicks in. The `wrong_branch_counter` increments
correctly. `wrong_branch_timeout` fires within ~50 steps instead of
the episode running to 600 steps.

---

## 6. Daughter-Only Entry Markers + 3D Euclidean d_corr (v1)

### Fix

**File:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py)

In `_build_entry_points()`, in addition to the existing
`_path_junction_arclengths` (every junction within 5 mm of the path),
build a smaller list `_path_daughter_arclengths` containing only
"daughter entry" junctions — real forks where ≥ 1 connected branch is
on-path AND ≥ 1 is off-path (not just a continuation). Coords stored
parallel as `_path_daughter_coords` so 3D distance can find each
entry's physical position.

For an RCCA path (trunk → bridge(11) → RCCA), the daughter entries are:
- bif2-trunk junction at z=384.7 (trunk on, LCCA off, bridge(11) on → real fork)
- bif2b at z=416.2 (bridge(11) on, RCCA on, RVA off → real fork)
- trunk-top junction at z=391.95 (trunk on, bridge(18) off → real fork)

Three real forks, vs five-or-more raw junctions in
`_path_junction_arclengths`.

New accessors:

- `get_arclength_to_next_daughter_entry()` — mirror of
  `get_arclength_to_next_correct_entry()` but uses the smaller list
- `get_arclength_past_last_daughter_entry()` — same for the past-junction case
- `get_3d_dist_to_next_daughter_entry()` — **3D Euclidean** distance
  from tip to the *coordinate* of the next daughter entry. Replaces
  arclength-projection for cases where physical proximity matters.

### Effect on +1 reward gate

**File:** [training _scripts/util/env5.py](training _scripts/util/env5.py)

Switched `+1.0 CORRECT_ENTRY_REWARD` from `_path_junction_arclengths`
to `_path_daughter_arclengths`. Eliminates spurious +1s for crossing
internal trunk-trunk junctions. Episode-start pre-population of
`_correct_entries_seen` switched too.

---

## 7. Graph-Routed d_corr (v2 — handles sister branches)

### Context

After the v1 metrics were implemented, a deeper issue surfaced: 3D
Euclidean distance to the next daughter entry is misleading when the
wire is in a parallel sister branch.

**Concrete case:** wire in LVA branch at vessel-CS (49.16, 35.43, 441.77),
target = RVA (path goes trunk → bridge(11) → RVA). 3D Euclidean to
bif2b = ~30 mm — would suggest the wire is "close" to the RVA ostium.
But the wire physically must retract ~30 mm out of LVA, traverse the
bridge(18) bridge BACKWARD to trunk-top, switch to the trunk path,
descend through bridge(11), THEN reach bif2b. The actual centerline
traversal distance is ~85 mm.

### Reason

3D Euclidean ignores the centerline graph topology. A wire in a sister
branch can be physically close to the next ostium in straight-line
distance but topologically far away. The heuristic regime selector
(near-junction precision regime when `d_corr < 10 mm`) and the
LocalGuidance feature 11 both consume this metric — feeding them
straight-line distance overstates how close the wire is.

### Fix

**File:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py)

Added `_build_branching_graph()` (called from `reset()`):
1. Match each branch to the BPs at its first and last endpoints (3D
   proximity within 2 mm).
2. Build a BP-to-BP adjacency: each branch contributes an edge between
   its two endpoint BPs of weight = branch arclength.
3. Identify path-BPs (within 5 mm of any path point); they become the
   sources for a Dijkstra sweep.
4. Run Dijkstra outward to compute `_bp_to_path_dist[bp]` (graph
   distance from each BP to the nearest path-BP) and
   `_bp_to_path_arclen[bp]` (the path-arclength of the rejoin point).

Cost: O(B²) at reset. With B < 30 in this dataset, this is microseconds
and only runs once per episode.

Added `get_routed_d_corr_to_next_daughter_entry()`:

```python
def get_routed_d_corr_to_next_daughter_entry(self) -> float:
    """Graph-route distance from tip to next daughter entry,
    accounting for sister-branch detours."""
    proj = self.get_projection()

    # FAST PATH: tip is on the planned-path polyline (cross-track < 6 mm).
    # Don't require nearest-branch to also be on-path because at junctions
    # an off-path branch may come perpendicularly closer while the tip is
    # still geometrically AT the polyline.
    if proj.cross_track_dist < CROSS_TRACK_TOLERANCE_MM:
        self._last_on_path_arclen = proj.s
        return self.get_arclength_to_next_daughter_entry()

    # OFF-PATH: graph-routed detour
    nearest_branch = self.get_nearest_branch()
    # ... project tip onto its current (wrong) branch
    # ... look up precomputed graph distance from each endpoint BP to
    #     the nearest path-BP
    # ... pick the shorter detour
    # ... add arclength from rejoin to next daughter
```

### Recomputation triggers

The user's design: detour is only meaningful when the nearest branch is
off-path; trigger recomputation when that transition happens. In this
implementation, the fast-path check runs every step (O(1) projection
lookup) and only pays the off-path detour computation when the wire is
actually off-path. The graph itself is built once per episode at
`reset()` because `FixedPathfinder`'s planned path is fixed; if a future
change switches to dynamic re-planning, the graph would just need to
be rebuilt then.

### Smoke test verification

`test_routed_dcorr.py` exercises the routed metric at 7 representative tip
positions on the actual centerline data:

| tip position | nearest branch | routed_d_corr |
|--------------|----------------|---------------|
| at trunk-top exactly | trunk | ON_PATH 30.5 mm to next daughter |
| 5 mm off trunk-top (= wedge) | trunk | ON_PATH 4.5 mm to next daughter |
| **inside LVA (sister, 30 mm in) ← key case** | LVA | **OFF_PATH 86.4 mm** (detour 55.9 + remaining 30.5) |
| inside RCCA (wrong sister) | RCCA | OFF_PATH 24.4 mm detour |
| at LCCA junction | b0 | ON_PATH 0.01 mm (cross-track 0 wins fast path) |
| at bif2b | RVA | ON_PATH inf (last junction) |
| inside RVA 50 mm in | RVA | ON_PATH inf |

The LVA-sister case (86.4 mm) is the validation: 3D Euclidean would
have reported ~30 mm, but the routed metric correctly reports the
actual centerline retrace + reroute distance.

---

## 8. Wired Into env5 / Heuristic / Observation

### env5.py

**File:** [training _scripts/util/env5.py](training _scripts/util/env5.py)

- Wrong-branch detector (line ~447) switched from
  `is_on_correct_branch()` to `is_on_correct_path()`. Catches the
  arch-wedge case where the wire is on-trunk but laterally off-path.
  Both metrics still computed so the legacy `on_br=` STEP-log field
  works for backward-compat analysis.
- +1 CORRECT_ENTRY_REWARD gate switched to `_path_daughter_arclengths`
  (real-fork-only).
- Episode-start pre-population of `_correct_entries_seen` switched to
  daughter-only arclengths to match the gate.
- STEP log adds `on_path=`, `d_corr_3d=`, `d_corr_routed=`,
  `arc_past_d=`, `daughters_passed=`. Legacy fields kept.

### heuristic_policy.py

**File:** [training _scripts/util/heuristic_policy.py](training _scripts/util/heuristic_policy.py)

- Off-branch retract handler triggers on `is_on_correct_path()` (was
  `is_on_correct_branch()`).
- Heuristic regime metrics: `d_corr_mm` switched to
  `get_routed_d_corr_to_next_daughter_entry()` and `arc_past_mm`
  switched to `get_arclength_past_last_daughter_entry()`.

### heuristic_controller.py

**File:** [training _scripts/util/heuristic_controller.py](training _scripts/util/heuristic_controller.py)

Comments updated to reflect the new metric semantics. Threshold values
unchanged (`d_corr_mm < 10` is now physically meaningful instead of
arclength-projection-based).

### LocalGuidance feature 11

**File:** [eve/eve/observation/localguidance.py](eve/eve/observation/localguidance.py)

Switched feature 11 source from `get_arclength_to_next_correct_entry()`
to `get_routed_d_corr_to_next_daughter_entry()`. The policy now sees
honest "actual centerline distance to next daughter" instead of
projection-arclength.

---

## 9. Smoke Tests

**Files (new):**
- [test_pathcontext_path_aware.py](test_pathcontext_path_aware.py)
- [test_routed_dcorr.py](test_routed_dcorr.py)
- [analyze_rl8_highinsert.py](analyze_rl8_highinsert.py)
- [analyze_rva_traj.py](analyze_rva_traj.py)

The two `test_*` files run outside Docker (no SOFA, no gymnasium) by
mocking the vessel-tree-like API on top of raw centerline JSONs. They
verify that:

1. `_path_daughter_arclengths` filters internal trunk-trunk junctions
   correctly (3 entries vs 5+ in `_path_junction_arclengths`).
2. `_branch_on_path_masks` correctly flags trunk indices on-path
   for an LVA target and partial-or-zero for off-path branches.
3. `is_on_correct_path()` returns False at the (52, 49, 399) wedge tip
   while `is_on_correct_branch()` (legacy) returns True — confirming
   the divergence catches the wedge.
4. `get_routed_d_corr_to_next_daughter_entry()` reports 86.4 mm for the
   LVA-sister case (vs misleading ~30 mm 3D Euclidean), 0.01 mm at
   the LCCA junction (correctly classified as on-path), and `inf` for
   tips past the last daughter.

The `analyze_*` files re-process existing run logs using the run-28
methodology (SUCCESS = R > 1.0, WEDGE600 = ≥580 steps, FOLD = <200
steps) plus 3D-distance-to-named-ostium classification.

---

## 10. Validation Status

End-to-end Docker A/B run with the new path-aware metrics is **not
yet executed** — Docker Desktop was down at the time of this writing.
All five modified files (`pathcontext.py`, `env5.py`, `heuristic_policy.py`,
`heuristic_controller.py`, `localguidance.py`) pass `py_compile`. The
two smoke tests pass. The expected end-to-end behavior:

- Non-LVA `wrong_branch_timeout` firings should INCREASE (the (52,49,399)
  wedge episodes that previously ran to 600 steps with `on_branch=True`
  now correctly classify as off-path).
- Mean episode length for non-LVA targets should DROP (from ~600
  toward ~100–300).
- `daughters_passed` becomes the cleaner replacement for `entries_passed`
  (LVA successes still show 2; arch-wedges should consistently show 0).
- Successes should not decrease (LVA path threading unaffected; the
  wire is on the path during success runs, so `is_on_correct_path=True`).

Run command (when Docker is up):

```bash
MSYS_NO_PATHCONV=1 docker run --name env5_rl8_pathaware_50ep --gpus all \
    --shm-size=24g --init -d \
    <ALL MOUNTS from rl_improv_8 worktree> \
    eve-training-fixed \
    python3 /opt/eve_training/training_scripts/heuristic_only_run.py \
        --env_version 5 -n env5_rl8_pathaware_50ep \
        --insertion_z 345 --episodes 50 -nw 16 -d cuda:0
```

---

## 11. Pending / Deferred

- **Bif1 region not investigated** — the lower-trunk bifurcation at
  z≈70 has 6 numbered branches converging in a small area. We never
  see episodes wedge there in the high-insertion config, but a
  femoral-entry config might.
- **Branch-radius-aware cross-track tolerance** — currently uniform 6 mm.
  In the trunk where the lumen is wider, this is conservative; in
  small distal branches (e.g. inside RVA at z≈570) it may be loose.
  Defer until 6 mm is shown insufficient.
- **Daughter-direction labelling** — `_path_daughter_arclengths` flags
  every real-fork junction the path threads, but doesn't distinguish
  "the daughter we're heading INTO at this fork" from "the off-path
  daughter we're avoiding". For the (80, 55, 395) wedge episodes for
  RCCA targets, the wire's nearest junction was the trunk-top junction
  whose off-path branch (bridge(18)) goes to LVA — which is the
  *opposite-side* of where the RCCA target is. A directional tag
  would help the heuristic decide whether to commit through the
  upcoming junction or steer past it.
- **Removing legacy methods** — `is_on_correct_branch`,
  `get_arclength_to_next_correct_entry`, `get_dist_to_next_correct_entry`,
  `_path_junction_arclengths` are all retained for backward
  compatibility with old run analyses. Once the new metrics are
  validated end-to-end, these can be removed.
- **Dynamic path replanning** — currently the planned path is fixed
  at episode reset (`FixedPathfinder`). If the wire enters a wrong
  daughter and gets stuck, a `Dijkstra2Pathfinder` (recomputes path
  every step from current tip) might find a shorter recovery route.
  The graph-routed d_corr already captures the retrace cost; switching
  pathfinders would give the heuristic a *different* path to follow,
  not just a more honest distance metric. Defer until pathfinder-fixed
  failure modes are exhausted.

---

## 12. A/B Run #1 — Path-Aware v1 / v2 (Uniform Cross-Track Tolerance)

### Context

Ran `heuristic_only_run.py --insertion_z 345 --episodes 50 -nw 16` against
the v1 implementation (uniform cross-track threshold). Tested two
threshold values:

- **v1**: `CROSS_TRACK_TOLERANCE_MM = 6.0`
- **v2**: bumped to `10.0` after observing v1's behavior

### Reason

The v1 runs uniformly produced episodes ending at -53 reward across all
50 episodes, all targets, all PIDs. v2 (10 mm) reproduced the same
pattern. Inspection of worker logs revealed the cause:

The wire's natural in-lumen drift in the wide aortic trunk is
**~9.5 mm cross-track** to the planned-path polyline (centerline to
wire-tip perpendicular). The trunk's lumen is wide enough that this is
normal navigation, not a wedge. With a uniform threshold:

- 6 mm: wire is constantly off-path during normal trunk traversal.
- 10 mm: wire jitter (sub-mm SOFA solver noise) takes the classification
  across the threshold both directions every few steps, producing
  flicker.

Both regimes triggered the heuristic_policy off-path retract handler
on every flicker → wire retracted, returned to path, pushed forward,
drifted again → infinite ~10-step thrashing cycle.

### Findings

- `_off_branch_steps` never reached `OFF_BRANCH_GRACE_STEPS=50` because
  brief on-path returns reset the counter. So `wrong_branch_timeout`
  never fired; episodes ran to 600 steps.
- Step penalty (`WRONG_BRANCH_STEP_PENALTY=-0.1`) accumulated for the
  ~520 off-path steps per episode → cum_reward ≈ -52 to -54.
- Empirical wedge cross-track magnitudes (re-measured with corrected
  vessel-CS transform):
  - LVA-side wedge centroid (52, 49, 399): ~13 mm cross-track.
  - RCCA/RVA-target wedge (path crosses to negative x): ~49 mm.

A *single* uniform threshold cannot simultaneously be loose enough that
9.5 mm trunk drift is on-path AND tight enough that 13 mm bif2 wedge
is off-path. Local lumen radius varies 4–5× across the route.

---

## 13. A/B Runs #2-3 — Surgical Reverts (v3, v4)

### Context

After v2's uniform-threshold thrashing, I reverted *some* of the v1 changes
to isolate which knob caused the regression vs the highinsert baseline
(env5_rl8_highinsert_50ep with 4 LVA successes).

### Reason

To distinguish: did v1's failure come from the new metric being too
strict, the heuristic now retracting on every minor drift, or both?

### Reverts

**v3** reverted two consumers back to the legacy `is_on_correct_branch()`:

1. `env5.py` wrong-branch detector — uses legacy nearest-branch identity check
2. `heuristic_policy.py` off-path retract trigger — same

Outcome: 1 LVA success early then most ep=1 episodes wedged at -48 to
-49 reward (worse than baseline -35). The heuristic regime selector
still using `get_routed_d_corr_to_next_daughter_entry()` (3D Euclidean
to next daughter coord, strict) was the remaining contamination — its
"near junction" precision regime fires only when the wire is physically
near a daughter, but the legacy arclength version fires when projection
arclength is close (more lax). Wire pushes too hard near junctions.

**v4** added a third revert:

3. heuristic regime selector reverted to `get_arclength_to_next_correct_entry()` and `get_arclength_past_last_junction()` (legacy arclength, all-junctions)

Outcome: 0 / 50 successes (vs baseline 4). Outcome distribution:
29 WEDGE, 20 MID, 1 FOLD<200. No -53 thrashing — episodes vary in
shape — but no LVA successes either.

### What was kept new across v3 and v4 (never reverted)

- `+1` daughter reward gate: still uses `_path_daughter_arclengths`
  (real-fork only, daughter-only filter).
- LocalGuidance feature 11: still uses `get_routed_d_corr_to_next_daughter_entry()`.
- All STEP-log diagnostic fields (`on_path`, `d_corr_3d`,
  `d_corr_routed`, `arc_past_d`, `daughters_passed`).
- All precomputed state at reset (`_branch_on_path_masks`,
  `_path_daughter_arclengths`, `_path_daughter_coords`,
  `_branch_to_bp_pair`, `_bp_adjacency`, `_path_bp_set`,
  `_bp_to_path_dist`, `_bp_to_path_arclen`).
- All new methods on `PathProjectionCache`.

### Findings

The reverts isolated the user's diagnosis: a **uniform** cross-track
tolerance can't simultaneously fit the wide trunk and the narrow
daughters. Need **branch-radius-aware** tolerance that scales with the
local vessel's lumen.

---

## 14. State-Machine `current_branch` + Radius-Aware Tolerance (v5)

### Context

After v3/v4 confirmed the threshold-mismatch problem, the user proposed
an elegant alternative: stop deciding "which branch is the wire on"
by per-step geometric projection, and instead **track it as a state
variable updated by committed events** (forward/backward junction
crossings). Combined with **branch-radius-aware tolerance** (each
centerline point already carries `BranchWithRadii.radii`), this
simultaneously eliminates two failure modes: (a) projection-jitter
flicker at junctions, and (b) tolerance-mismatch across vessel sizes.

### Reason

- Multi-branch projection ambiguity at junctions: `get_nearest_branch()`
  projects the tip onto every branch's centerline and picks the min
  cross-track winner. At bifurcations, multiple centerlines are within
  mm; sub-mm tip jitter flips the winner step-to-step.
- Lumen size varies 4–5× across the route. A single threshold can't fit.
- The +1 daughter-reward already commits on `arc_past >= 10 mm` — the
  same signal can drive a state machine.

### Fix

**Files:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py), [training _scripts/util/env5.py](training _scripts/util/env5.py)

**New constants** in `pathcontext.py`:
```python
K_RADIUS = 1.5
MIN_TOLERANCE_MM = 2.0
DEFAULT_RADIUS_MM = 5.0
MIN_RADIUS_FLOOR_MM = 2.0
MAX_RADIUS_CEILING_MM = 12.0
COMMIT_HYSTERESIS_MM = 10.0
```

Empirical fit:
- Trunk (radius 12) → tolerance 18 mm → 9.5 mm in-lumen drift safely
  on-path (no flickering).
- Bif2 cavity (radius 7) → tolerance ~10 mm → 13 mm wedge fails it.
- Bridge (radius 4) → tolerance 6 mm → 49 mm RCCA wedge fails it.
- Daughter (radius 3) → tolerance ~5 mm.

**Per-episode state in `PathProjectionCache`:**
```python
self._current_branch_idx = None    # idx into _branches_tuple
self._on_planned_path = True       # False while wire is in a sister branch
self._prev_proj_s = 0.0             # for forward/backward crossing detection
```

**Per-episode precompute at reset:**
- `_branch_radii: Tuple[Optional[np.ndarray]]` — per-branch radii arrays
  (None for branches without radii data).
- `_path_branch_idx_set: set[int]` — O(1) "is branch i on the path?".
- `_path_branch_sequence: List[(start_arc, end_arc, branch_idx)]` —
  ordered list of which on-path branch the planned-path polyline lies
  on at each arclength range. Built by per-polyline-point projection
  onto each on-path branch and choosing the min-cross-track winner;
  single-point disagreements smoothed.
- `_path_branch_sequence_with_junctions: List[(j_arc, prev_idx, next_idx)]`
  — junctions between adjacent sequence entries. Used to detect
  forward/backward crossings.
- `_junction_off_path_candidates: Dict[float, List[int]]` — for each
  path-junction, the list of off-path connected branches. Used to
  disambiguate which sister branch the wire entered.

**New methods on `PathProjectionCache`:**

- `update_branch_state()` — once-per-step state update. Detects forward
  / backward path-junction crossings (10 mm arclength hysteresis), and
  flips `_on_planned_path` based on the radius-aware tolerance.
- `get_local_radius()` — vessel radius at the wire's nearest centerline
  point on the current branch (clamped to `[MIN_RADIUS_FLOOR_MM, MAX_RADIUS_CEILING_MM]`).
- `get_local_tolerance()` — `max(MIN_TOLERANCE_MM, K_RADIUS * get_local_radius())`.
- `_branch_at_arclength(s)` — look up which on-path branch the planned-path
  polyline lies on at arclength `s`.
- `_pick_off_path_branch(s)` — when wire goes off-path, identify which
  sister branch it entered using the precomputed junction-local
  off-path candidates (typically 1–2 branches), not all 25.

**Refactored consumers:**

- `is_on_correct_path()` simplifies to `return self._on_planned_path`.
  The state machine is the hysteresis (10 mm of arclength commitment
  on junction crossings + radius-aware tolerance prevents in-lumen
  flicker). No separate debouncer needed.
- `get_routed_d_corr_to_next_daughter_entry()` fast path tests
  `_on_planned_path` directly. Off-path detour uses `_current_branch_idx`
  directly (no per-step `get_nearest_branch()` call).

**Wired into `env5.step()`** at line ~451: `self._path_context.update_branch_state()`
runs after `super().step()` populates the projection cache and before
`is_on_correct_path()` / `is_on_correct_branch()` queries. Defensive
try/except so a state-machine bug can't break the simulation.

**STEP log additions:** `cur_branch=`, `local_r=`, `tol=` so we can
correlate state-machine evolution with wire trajectory offline.

---

## 15. Run env5_rl8_pathaware_v5_50ep — State-Machine Run

### Setup

Same `--insertion_z 345 --episodes 50 -nw 16` config. All v2 design +
v5 state-machine + radius-aware tolerance active.

### Findings

| Metric | env5_rl8_highinsert_50ep (baseline) | v4 (legacy detectors + daughter-only +1) | **v5 (state-machine + radius-aware)** |
|---|---|---|---|
| SUCCESS | 4 | 0 | 0 |
| WEDGE600 | 35 | 29 | **17** |
| MID (200–580) | 11 | 20 | 29 |
| FOLD<200 | 0 | 1 | 4 |

**Key wins**:
- **No more uniform-threshold thrashing** — STEP log shows
  `on_path=1, cur_branch=Centerline curve (2).mrk, local_r=11.8, tol=17.7`
  during normal trunk navigation, with cross-track ~3 mm. State stable
  across the entire trunk traversal.
- **State-machine transitions cleanly at junctions** — example: ep=2 of pid=122
  transitioned `cur_branch` from `(2).mrk` (trunk) to `(0).mrk`
  (upper-trunk bridge) at step 200 when the wire crossed the trunk-top
  junction.
- **Wedge count nearly halved** (35 → 17). Episodes that previously ran
  to 600 steps without commit now correctly classify as off-path,
  accumulate -0.1/step penalties, and end via `wire_fold_stall` at
  ~250–500 steps instead of timing out at 600.

**Remaining regression**: 0 LVA successes vs baseline 4. This is
attributable to either:
- Sample variance (with N=50 and an 8% baseline rate, P(0) ≈ 1.5%).
- The `+1` daughter reward gate change to `_path_daughter_arclengths`
  (kept new through v3-v5) — but reward shape doesn't drive heuristic
  actions in this heuristic-only run, so this is unlikely to be the
  cause.
- Likely cause: the radius-aware threshold exposes new flicker
  pockets in regions where local radius gives `tol ≈ wire drift`
  (e.g. near junctions where lumen narrows). State-machine 10 mm
  arclength commit is on **junction crossings**, not on **cross-track
  threshold transitions** — so single-step on/off-path flips can still
  happen at the threshold edge in narrow regions.

### Open hypothesis

Add **asymmetric (deadzone) hysteresis** on the cross-track threshold:
flip off-path at `ct > tol`, flip back on-path only at `ct < tol * 0.7`.
Or add a per-step debouncer on the cross-track flip (e.g. require K
consecutive over-threshold steps to flip `_on_planned_path` False).
Either keeps the state-machine architecture and adds the smoothing
that's only currently applied to junction commits.

---

## 16. End-of-Episode Snapshot Rendering

### Context

Once the state machine + radius-aware metrics were in place, debugging
shifted from "why is the wire flagged on-path/off-path" to "what does
the wire actually look like at the moment the episode ended?". Static
3D snapshots make wedge-mode classification (arch wall vs LVA-side vs
inside-daughter-wedged-against-wall) immediate at a glance.

### Reason

The STEP log captures rich diagnostic state but only at sparse cadence
(first 30 steps + every 50 + terminal). Reconstructing a 3D mental
picture of the final wire pose from log fields is slow. A PNG per
episode keyed by termination reason makes failure-mode triage a flat
file system.

### Fix

**File (new):** [training _scripts/util/snapshot.py](training _scripts/util/snapshot.py)

Two render backends, selected by `SNAPSHOT_MODE` env var:

- `mesh` — vessel surface mesh (loaded from `vessel_tree.visu_mesh_path`
  via `pyvista`, decimated to ≤6000 faces) + per-device wires + planned
  path polyline + target marker. Heavier; cached per-worker after first
  episode.
- `centerlines` — every branch centerline (off-path = light grey,
  on-path = blue) + per-device wires + planned path + target. Faster;
  no mesh dependency.

Both backends use matplotlib's `Agg` backend (headless) and
`Poly3DCollection` / 3D plot. Outputs land at:
```
${SNAPSHOT_DIR}/<reason>/ep<N>_pid<P>_step<S>_<reason>.png
```

`<reason>` is one of: `success`, `wire_fold_stall`,
`wrong_branch_timeout`, `vessel_end`, `sim_error`, `max_steps`, or
`unknown_truncation` — resolved by the new `_resolve_termination_reason()`
helper on `BenchEnv5` that walks env5's truncation flags in priority
order.

All rendering wrapped in **triple-defensive** try/except:
1. Outer `save_snapshot` try/except — any rendering exception caught,
   logged to stderr, no PNG saved.
2. Inner attribute access (`device_trackings3d`, `target.coordinates3d`,
   `vessel_tree.visu_mesh_path`, `pathfinder.path_branch_set`) all
   wrapped — if any is missing the snapshot degrades gracefully.
3. Mesh-mode fallback to centerlines if the visu mesh path is missing.

**File (modified):** [training _scripts/util/env5.py](training _scripts/util/env5.py)

End-of-episode snapshot block added in `step()` — gated by
`SNAPSHOT_MODE` env var; lazy import of `util.snapshot.save_snapshot`;
runs after the state-machine + STEP log so it captures the final
post-update state. Runs once per terminated/truncated episode.

New `_resolve_termination_reason()` method maps env state to a single
short label (priority: `terminated → success`, then heuristic abort
reason, then fold count, off-branch count, vessel-end, sim-error,
max-steps).

**File (modified):** [training _scripts/heuristic_only_run.py](training _scripts/heuristic_only_run.py)

New `--snapshots {none|mesh|centerlines}` CLI flag (default `none`).
When non-`none`, sets `SNAPSHOT_MODE` and `SNAPSHOT_DIR` environment
variables BEFORE `BenchAgentSynchron(...)` creates worker processes,
so workers inherit them via spawn (mirrors the existing `STEP_LOG_DIR`
inheritance pattern).

### Image dependencies

Verified via `docker run --rm eve-training-fixed python3 -c "import ..."`:
- `pyvista` 0.44.2 ✓ (mesh backend works)
- `matplotlib` 3.7.5 + Agg backend + `Poly3DCollection` ✓ (centerlines + mesh both work)

### Cost

- Render cost: ~0.5–3 s per episode end. 50 episodes × 16 workers ≈
  50 PNGs total → adds ~50–150 s to a ~440 s baseline run.
- Mesh cache: ~6000 faces × ~32 bytes × 16 workers ≈ 3 MB total.
- PNG file size: ~150–300 KB each.

### Required mount for the next docker run

The new `util/snapshot.py` file is added to the worktree but isn't
yet in the docker run mount list. Before launching v6 with snapshots,
add:

```
-v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\snapshot.py:/opt/eve_training/training_scripts/util/snapshot.py"
```

Without this mount the lazy import fails (caught by try/except, run
survives, but no PNGs save). With `--snapshots none`, no mount needed.

---

## File Index

| File | Change Type | Purpose |
|---|---|---|
| `eve_bench/eve_bench/dualdevicenav.py` | Modified | §1 — `insertion_z` kwarg with z-range branch lookup + tangent sign-flip for descending-z trunk |
| `training _scripts/DualDeviceNav_train.py` | Modified | §1 — `--insertion_z` CLI flag, mutual-exclusion guard with `--checkpoint_dir`; passed through to both DualDeviceNav constructors |
| `training _scripts/heuristic_only_run.py` | **New** + Modified | §2 — heuristic-only parallel runner (no SAC training). §16 — `--snapshots` CLI flag; sets `SNAPSHOT_MODE` / `SNAPSHOT_DIR` env vars before worker spawn |
| `eve/eve/util/pathcontext.py` | Modified | §5 — `_branch_on_path_masks`, constants `ON_PATH_TOLERANCE_MM`, `CROSS_TRACK_TOLERANCE_MM`, `is_on_correct_path()` with separate hysteresis state. §6 — `_path_daughter_arclengths`, `_path_daughter_coords`, `get_arclength_to_next_daughter_entry()`, `get_arclength_past_last_daughter_entry()`, `get_3d_dist_to_next_daughter_entry()`. §7 — `_build_branching_graph()`, BP-adjacency Dijkstra, `_bp_to_path_dist`, `_bp_to_path_arclen`, `get_routed_d_corr_to_next_daughter_entry()` with on-path-fast-path + off-path-detour. §14 — `K_RADIUS=1.5`, `MIN_TOLERANCE_MM=2.0`, `COMMIT_HYSTERESIS_MM=10.0`; state-machine fields (`_current_branch_idx`, `_on_planned_path`, `_prev_proj_s`, `_branch_radii`, `_path_branch_idx_set`, `_path_branch_sequence`, `_path_branch_sequence_with_junctions`, `_junction_off_path_candidates`); methods `_build_path_branch_sequence()`, `update_branch_state()`, `get_local_radius()`, `get_local_tolerance()`, `_branch_at_arclength()`, `_pick_off_path_branch()`. `is_on_correct_path()` and `get_routed_d_corr_to_next_daughter_entry()` simplified to read from state-machine flags |
| `training _scripts/util/env5.py` | Modified | §8 — wrong-branch detector switched to `is_on_correct_path()`; +1 reward gate switched to `_path_daughter_arclengths`; reset-time pre-population switched; STEP log adds `on_path=`, `d_corr_3d=`, `d_corr_routed=`, `arc_past_d=`, `daughters_passed=`. §14 — `update_branch_state()` call inserted before wrong-branch detector; STEP log adds `cur_branch=`, `local_r=`, `tol=`. §16 — end-of-episode snapshot block (lazy-imports `util.snapshot.save_snapshot`); new `_resolve_termination_reason()` method |
| `training _scripts/util/heuristic_policy.py` | Modified | §8 — off-branch retract trigger uses `is_on_correct_path()`; heuristic regime `d_corr_mm` uses `get_routed_d_corr_to_next_daughter_entry()`; `arc_past_mm` uses `get_arclength_past_last_daughter_entry()` |
| `training _scripts/util/heuristic_controller.py` | Modified | §8 — comments updated for new metric semantics; thresholds unchanged |
| `eve/eve/observation/localguidance.py` | Modified | §8 — feature 11 source switched to `get_routed_d_corr_to_next_daughter_entry()` |
| `training _scripts/util/snapshot.py` | **New** | §16 — end-of-episode 3D PNG renderer; two backends (`mesh` / `centerlines`); matplotlib Agg headless; mesh decimation via pyvista; triple-defensive against missing attributes / deps. Outputs `<reason>/ep<N>_pid<P>_step<S>_<reason>.png` under `SNAPSHOT_DIR` |
| `test_pathcontext_path_aware.py` | **New** | §9 — smoke test for `is_on_correct_path()`, `_path_daughter_arclengths`, `_branch_on_path_masks` |
| `test_routed_dcorr.py` | **New** | §9 — smoke test for graph-routed d_corr at 7 representative tip positions |
| `analyze_rl8_highinsert.py` | **New** | §3, §4 — re-analysis of env5_rl8_highinsert_50ep using run-28 methodology, with 3D-Euclidean ostium distances and full `entries_gained` crosstabs |
| `analyze_rva_traj.py` | **New** | §3 — trajectory trace for RVA "near-success" episodes; demonstrates the LVA-route shortcut via the basilar-merger shared endpoint |
| `RL_IMPROV_8_CHANGES.md` | **New** | This document |
