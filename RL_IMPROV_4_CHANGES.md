# RL Improv 4 — Changes Reference

All changes in this branch relative to `rl_improv_3`. For each change: what was wrong, why it matters, and what was done.

---

## 1. True Branch Membership Detection

### Context

Every wrong-branch detector and the `on_correct_branch` observation feature were using a cross-track distance proxy: if the tip is within 5mm of the correct-path polyline, it is considered on the correct branch. This is geometrically wrong. Two branches at a bifurcation can be within 5mm of each other near the junction, so the proxy reports the tip as on-path when it has actually entered a wrong branch.

### Reason

- 100% false-negative rate at bifurcations: the heuristic was never penalized for wrong-branch entry because the proxy always said "on-path" near junctions
- The wrong-branch detector (`D3`) never fired
- `guidance[7]` (on_correct_branch obs feature) was permanently 1.0 near all bifurcations, exactly where correct branch selection is needed most

### Fix

**File: `eve/eve/util/pathcontext.py`** (new file)

Added `PathProjectionCache` — a shared per-step lazy cache. At reset it builds:

- `_build_branch_index()`: scipy `cKDTree` over all branch centerline coordinates from the full vessel tree. O(log N) nearest-branch lookup.
- `_build_entry_points()`: classifies every `vessel_tree.branching_point` into `_wrong_branch_entries` (connected to any branch not in `path_branch_set`) and `_correct_branch_entries` (connected to any branch in `path_branch_set`).

`is_on_correct_branch()` now: find nearest branch via KD-tree → ask `pathfinder.is_branch_on_path(branch)`. This is exact membership, not a distance proxy.

**Validation**: `test_on_branch.py` (standalone, no SOFA imports) loaded real DualDeviceNav JSON vessel centerlines, tested all 4 targets (LCCA, LVA, RCCA, RVA) against 1950 points. Result: 99.6–99.8% correct. The 0.2–0.4% mismatches are at bifurcation junction points where geometric ambiguity is inherent (one point is equidistant from two branches).

---

## 2. LocalGuidance: 8 → 14 Dimensions

### Context

The RL agent's guidance observation (`guidance` key in the obs dict) provided 8 features. It had no information about how far the agent is from wrong-branch entries or correct-branch entries. The agent could not learn to anticipate and avoid bifurcations.

### Reason

Without entry-point proximity signals, the agent learns purely reactively — it only notices it's on the wrong branch after the fact. With explicit distance + direction features, the agent can start to steer around wrong entries and aim for correct entries.

### Fix

**File: `eve/eve/observation/localguidance.py`**

Extended observation space from 8 → 14 dims. New dims 8–13:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 8 | `dist_wrong_entry` | [0, 1] | Euclidean dist to nearest wrong-branch bifurcation, clipped at 200mm |
| 9 | `wrong_entry_dir_x` | [-1, 1] | x-component of unit vector toward it (in 2D image plane) |
| 10 | `wrong_entry_dir_z` | [-1, 1] | z-component of unit vector toward it |
| 11 | `dist_correct_entry` | [0, 1] | Euclidean dist to nearest correct-path bifurcation |
| 12 | `correct_entry_dir_x` | [-1, 1] | x-component of unit vector toward it |
| 13 | `correct_entry_dir_z` | [-1, 1] | z-component of unit vector toward it |

New module-level helper `_entry_direction(tip_vessel, entry_coords, dist, image_rot_zx)`: computes delta in vessel-CS, rotates to tracking3d via `vessel_cs_to_tracking3d(..., (0,0,0), None)`, then takes `[0]` (x) and `[2]` (z). Returns `(0.0, 0.0)` when dist ≥ `_MAX_BIFURC_DIST_MM` (no nearby entry).

Distance and coordinate values come from `PathProjectionCache.get_dist_to_closest_wrong_entry()`, etc., which are computed once per step and cached.

---

## 3. 3D→2D Coordinate Consistency

### Context

The observation module had three separate places that project 3D vessel-CS vectors into 2D image-plane coordinates: path tangents, heading error computation, and entry directions. They each used different methods, producing inconsistent features.

### Reason

The canonical 2D projection used everywhere else (e.g., `Tracking2D`, `Target2D`) is: vessel-CS → tracking3d (applying the C-arm rotation `image_rot_zx`) → drop Y (axis 1). Directly dropping Y from vessel-CS ignores the C-arm rotation and produces a different 2D space. Features computed with different projections are not comparable to each other or to the tracking observations.

### Fix

**File: `eve/eve/observation/localguidance.py`**

#### Tangent pre-computation in `reset()`

Before:
```python
t2d = self._tangents[:, [0, 2]]  # naive vessel-CS drop
```

After:
```python
fluoro = self.intervention.fluoroscopy
t_tracking = vessel_cs_to_tracking3d(
    self._tangents, fluoro.image_rot_zx, (0.0, 0.0, 0.0), None
)
t2d = t_tracking[:, [0, 2]]
```

#### Heading error sign in `_compute_heading_error()`

Before: cross product sign used `cross[1]` from vessel-CS Y-axis.

After: both `device_dir` and `tangent` are rotated to tracking3d first, then cross product sign uses tracking3d Y-axis (image-plane "up"):

```python
rot_zx = fluoro.image_rot_zx
device_dir_t = vessel_cs_to_tracking3d(device_dir_v, rot_zx, (0.0, 0.0, 0.0), None)
tangent_t    = vessel_cs_to_tracking3d(tangent_3d,   rot_zx, (0.0, 0.0, 0.0), None)
dot   = float(np.clip(np.dot(device_dir_t, tangent_t), -1.0, 1.0))
cross = np.cross(device_dir_t, tangent_t)
sign  = 1.0 if cross[1] >= 0 else -1.0
```

The `None` fourth argument to `vessel_cs_to_tracking3d` skips field-of-view clipping, which is only relevant for position vectors, not direction vectors.

---

## 4. Wire Fold / Loop Detector (D5)

### Context

During training the guidewire can bend or loop back on itself. When this happens, `device_lengths_inserted[0]` (the commanded insertion length from SOFA IRController) keeps increasing, but the physical tip position stops advancing along the correct-path polyline. The episode then runs for 1000 steps pushing a midsection of the wire with no hope of success.

### Reason

Fold episodes waste compute (1000 steps at ~2–3x the normal step speed due to SOFA contact complexity) and pollute the replay buffer with transitions where insertion commands have no tip-position effect — a confusing signal for the RL critic.

### Fix

**File: `training _scripts/util/env5.py`**

New constants:
```python
FOLD_STALL_STEPS   = 15   # consecutive steps of insertion-without-progress to abort
FOLD_INSERTION_MM  = 0.5  # min commanded gw insertion/step to count as "actively inserting"
FOLD_ARCLENGTH_MM  = 0.5  # min tip arclength progress/step to count as "advancing"
```

New per-episode state: `_fold_stall_count`, `_prev_tip_s`, `_prev_inserted_gw`.

Detector in `step()` (runs in both heuristic and RL modes):
```python
tip_s   = self._path_context.get_projection().s
delta_s  = tip_s - self._prev_tip_s
self._prev_tip_s = tip_s

delta_gw = inserted_gw - self._prev_inserted_gw
self._prev_inserted_gw = inserted_gw

if delta_gw >= FOLD_INSERTION_MM and delta_s < FOLD_ARCLENGTH_MM:
    self._fold_stall_count += 1
else:
    self._fold_stall_count = 0

if self._fold_stall_count >= FOLD_STALL_STEPS:
    if self._heuristic_mode:
        self._heuristic_abort("wire_fold_stall", info)
    truncated = True
```

The −5.0 penalty is applied via the unified failure truncation block (see §6).

The 15-step threshold avoids false positives on rotation-only steps: when the action is pure rotation, `delta_gw < 0.5mm` so the counter stays at 0.

---

## 5. Wrong-Branch Shaped Rewards (Both Modes)

### Context

Previously the wrong-branch detector in heuristic mode just aborted the episode with a −5.0 penalty at step 10. The RL agent had no shaped reward signal for being off-branch at all — it only saw the eventual −5.0 at truncation.

### Reason

Without shaped rewards, the RL agent gets a sparse −5.0 signal 10 steps after entering the wrong branch, making credit assignment difficult. Adding per-step penalties means every step off-branch is negatively reinforced, giving the agent a dense gradient to avoid wrong branches.

### Fix

**File: `training _scripts/util/env5.py`**

```python
WRONG_BRANCH_ENTRY_PENALTY = -1.0   # one-time, on the step the tip enters a wrong branch
WRONG_BRANCH_STEP_PENALTY  = -0.1   # per-step while remaining off-branch (steps 2..9)
```

Detector in `step()` (both modes):
```python
if not on_correct_branch:
    self._off_branch_steps += 1
    if self._off_branch_steps == 1:
        reward += WRONG_BRANCH_ENTRY_PENALTY    # -1.0 on entry
    else:
        reward += WRONG_BRANCH_STEP_PENALTY     # -0.1 per subsequent step
    if self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS:
        if self._heuristic_mode:
            self._heuristic_abort("wrong_branch_timeout", info)
        truncated = True
else:
    self._off_branch_steps = 0
```

Total cost for a full 20-step wrong-branch timeout: −1.0 + (19 × −0.1) + −5.0 = −7.9.

---

## 6. Unified Truncation Penalty

### Context

Previously the failure penalty logic was split: heuristic aborts got −5.0, vessel-end truncations got −5.0 separately, and fold truncations were adding their own penalty directly in the detector block. This led to inconsistency and a potential double-penalty bug when heuristic mode was combined with fold detection.

### Fix

**File: `training _scripts/util/env5.py`**

All failure truncations feed a single penalty block:

```python
if truncated and not terminated and (
    self._vessel_end_trunc.truncated
    or self._fold_stall_count >= FOLD_STALL_STEPS
    or self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS
):
    reward += FAILURE_TRUNCATION_PENALTY   # -5.0
```

Conditions:
- `vessel_end_trunc.truncated`: device went past end of vessel tree (bad in both modes)
- `_fold_stall_count >= FOLD_STALL_STEPS`: wire folded
- `_off_branch_steps >= OFF_BRANCH_GRACE_STEPS`: wrong-branch timeout

Plain `max_steps` truncation is excluded — 1000 steps of `Step(factor=-0.001)` already accumulates −1.0 of step penalty.

---

## 7. Heuristic/RL Parity

### Context

Heuristic seeding and RL training used different reward and truncation logic. Heuristic mode had detectors D1 (both_max_stall), D2 (gw_partial_stall), D3 (wrong_branch), D4 (sat_rot) that fired early aborts with penalties. RL mode saw none of these — it ran to `max_steps` with no early termination except vessel-end. This meant the two datasets had fundamentally different reward distributions, making the heuristic buffer data a poor initialization for the RL critic.

### Reason

If heuristic data contains short episodes with negative rewards and RL data contains long episodes with different reward structure, the critic bootstraps poorly from the heuristic buffer. The policy that generated the data is the only thing that should differ between modes, not the reward function.

### Fix

**File: `training _scripts/util/env5.py`**

**Removed** heuristic-only detectors D1 (both_max_stall), D2 (gw_partial_stall), D4 (sat_rot) entirely. Their corresponding constants and state variables were deleted.

Both remaining detectors (D5: fold, D3: wrong_branch) now run in both heuristic and RL modes. The only difference between modes is:

- Heuristic mode additionally calls `_heuristic_abort(reason, info)` to record the abort reason in `info` dict
- RL mode just gets `truncated = True` and the penalty reward — no abort metadata

This means both modes produce identical reward signals for the same behavioral events. Only the policy changes.

---

## 8. Heuristic Policy: Random Retraction When Off-Branch

### Context

When the heuristic policy entered a wrong branch, it continued running `CenterlineFollowerHeuristic.get_action()`. The heuristic tries to follow the correct-path centerline, but from inside a wrong branch the path projection lands on the nearest correct-path point (which may be behind the tip), causing erratic rotation commands.

### Reason

The heuristic has no recovery behavior for wrong-branch entry. It was generating noisy, unhelpful transitions in wrong-branch states. The RL action space allows guidewire translation in `[-10, 30]`, so the agent can retract. The heuristic should demonstrate retraction as a recovery strategy.

### Fix

**File: `training _scripts/util/heuristic_policy.py`**

```python
on_correct_branch = True
try:
    on_correct_branch = base_env._path_context.is_on_correct_branch()
except Exception:
    pass

if not on_correct_branch:
    # Always-negative retraction: pull back toward the bifurcation
    gw_trans  = float(self._rng.uniform(-10.0, -1.0))
    cath_trans = gw_trans * self.heuristic.catheter_follow_ratio
    raw_action = np.array([gw_trans, 0.0, cath_trans, 0.0], dtype=np.float32)
else:
    raw_action = self.heuristic.get_action(self._rng)
```

Initially `[-10, 5]` (allowing some forward probing). Changed to `[-10, -1]` (always retract) after first training run analysis: the `[0, 5]` forward component caused 59% of heuristic episodes to oscillate endlessly at bifurcation 1, entering the wrong branch, partially retracting, then pushing forward again. With always-negative retraction, wrong-branch early-abort rate dropped from 59% (396/671 episodes) to 0%.

Rotation is set to 0.0 — no need to steer when retracting, and random rotation would generate confusing transitions.

Branch state is read from `_path_context.is_on_correct_branch()` — same per-step cached value that the env's D3 detector uses, so no redundant computation.

---

## 9. PathProjectionCache — Branch Membership & Entry Points

### Context

`eve/eve/util/pathcontext.py` was introduced in RL Improv 2 at 114 lines, providing basic per-step caching of `tip_vessel_cs` and `projection` to eliminate redundant `project_onto_polyline` calls between `ArcLengthProgress` and `LocalGuidance`.

### Reason

The original `is_on_correct_branch` used a `cross_track_dist < 5mm` proxy, which has 100% false-negative rate at bifurcations (both branches are within 5mm of the path near junctions). Sections 1–8 all depend on exact branch membership.

### Fix

**File: `eve/eve/util/pathcontext.py`** (114 → 263 lines)

Added branch membership detection and bifurcation entry-point classification:

- `_build_branch_index()` — scipy `cKDTree` over all vessel branch centerline coordinates, built at reset. O(log N) nearest-branch lookup.
- `_build_entry_points()` — classifies every `vessel_tree.branching_point` into `_wrong_branch_entries` (connected to any branch not in `path_branch_set`) and `_correct_branch_entries` (connected to any branch in `path_branch_set`).
- `get_nearest_branch()` — O(log N) KD-tree lookup per step (cached after first call)
- `is_on_correct_branch()` — exact branch membership via `pathfinder.is_branch_on_path()` on nearest branch
- `get_dist_to_closest_wrong_entry()` / `get_closest_wrong_entry_coords()` — Euclidean distance + 3D coords of nearest wrong-branch bifurcation
- `get_dist_to_next_correct_entry()` / `get_closest_correct_entry_coords()` — same for correct-path bifurcation

All six per-step values are lazily cached and invalidated at the start of each step via `invalidate()`.

---

## 10. Logging

### Context

After removing D1/D2/D4 and adding D5, the log string had stale references to removed detector counters and was only emitted in heuristic mode, losing visibility in RL mode.

### Fix

**File: `training _scripts/util/env5.py`**

#### EPISODE_START target coordinates

`EPISODE_START` log moved to **after** `super().reset()` so target coordinates are populated:
```python
result = super().reset(seed=seed, options=options)
target_str = ""
try:
    tc = self.intervention.target.coordinates3d
    target_str = f" | target=({tc[0]:.1f},{tc[1]:.1f},{tc[2]:.1f})"
except Exception:
    pass
self._step_logger.info(
    f"EPISODE_START | ep={self._episode_count} | ... | pid={os.getpid()}{target_str}"
)
```

Previously `EPISODE_START` was logged before `super().reset()`, so target was always from the prior episode.

#### Step-level log split

Log split into two parts:

**`shared_str`** — emitted in both modes:
```
| on_br=1 | off_br=0 | fold=0/15 | d_wrong=45.2 | wrong_pt=(12.3,4.5,6.7) | d_corr=18.1 | corr_pt=(...)
```

**`heur_str`** — appended in heuristic mode only:
```
| heur=1 | abort=none | mask=...
```

Episode-end log includes `heur_abort=<reason>` when heuristic mode is active.

---

## 11. CenterlineFollowerHeuristic Fixes

### Context

The centerline-following heuristic had three issues causing poor trajectory quality during heuristic seeding:

1. **Zero minimum translation**: `max(0.0, 0.1 * d_rem)` could produce zero translation near the target, stalling the guidewire.
2. **Rotation cap near target**: `d_rem <= 50.0` clamped rotation to `[-0.8, 0.8]`, preventing the large steering corrections needed at tight bifurcations that happen to be near the target.
3. **Device-length-aware cap**: A `try/except` block at the end that reduced translation when approaching maximum device length. This interacted poorly with the RL action space and was unnecessarily conservative.

### Fix

**File: `training _scripts/util/heuristic_controller.py`**

```python
# Before:
gw_trans = min(self.max_translation, max(0.0, 0.1 * d_rem))
# After:
gw_trans = min(self.max_translation, d_rem * 0.1)
gw_trans = max(gw_trans, 5.0)  # minimum forward push
```

Removed rotation cap near target (the `d_rem <= 50.0` clip block).

Removed device-length-aware cap (the `try/except` block reading `device_lengths_inserted` / `device_lengths_maximum`).

---

## 12. ReplayBuffer Import Fix

### Fix

**File: `eve_rl/eve_rl/replaybuffer/__init__.py`**

Added `EpisodeReplay` to the module exports:
```python
from .replaybuffer import ReplayBuffer, Batch, Episode, EpisodeReplay
```

Previously `EpisodeReplay` was defined in `replaybuffer.py` but not exported, causing `ImportError` when training scripts imported it.

---

---

# Proposed Improvements (Not Yet Implemented)

## P1. Wrong-Branch Oscillation Detector

### Problem

At bifurcation 1 (~70mm), the heuristic cycles endlessly: insert → enter wrong branch → retract (clears junction in <20 steps) → back on correct branch → insert again → wrong branch again. The `_off_branch_steps` counter resets to 0 each time the tip returns to the correct branch, so `OFF_BRANCH_GRACE_STEPS` never fires. The `_fold_stall_count` also resets every retraction step (`delta_gw < 0`). Result: 27% of second-run episodes ran 1000 steps with insertion stuck at 57–64mm, reward ≈ −218.

### Proposed Fix

Count total wrong-branch **entry events** per episode. After `WRONG_BRANCH_MAX_ENTRIES` (e.g., 8) entries, truncate.

**File: `training _scripts/util/env5.py`**
```python
WRONG_BRANCH_MAX_ENTRIES = 8

# In __init__ and reset():
self._wrong_branch_entries = 0

# In step(), inside `if self._off_branch_steps == 1:` block:
self._wrong_branch_entries += 1

# New truncation condition:
elif self._wrong_branch_entries >= WRONG_BRANCH_MAX_ENTRIES:
    if self._heuristic_mode:
        self._heuristic_abort("wrong_branch_oscillation", info)
    truncated = True
```

8 cycles × ~10–20 steps = 80–160 steps before abort (vs 1000 currently). Add to failure penalty condition too.

---

## P2. Increase OFF_BRANCH_GRACE_STEPS (20 → 50)

### Problem

At bifurcation 2 (~390mm insertion), catheter friction limits actual retraction to ~0.5–1.5mm/step despite commanding −8mm/step. Over 20 grace steps, only 10–20mm of actual retraction — but the tip needs 10–21mm to clear bif2. Episodes that were within 2–12mm of succeeding get truncated.

### Proposed Fix

Increase `OFF_BRANCH_GRACE_STEPS` from 20 to 50. At 0.5mm/step × 50 steps = 25mm minimum retraction, enough to clear bif2. Combined with P1 (oscillation detector), the longer grace period won't cause 50-step oscillation loops — P1 catches repeated entries before that.

---

## P3. Branch Interior Points (Fix Degenerate Observation Features)

### Problem

At both bifurcations, the same 3D junction point is simultaneously the entry to the correct branch AND the wrong branch. `PathProjectionCache._build_entry_points()` classifies it into both lists. Result: `wrong_dir == correct_dir` (features 9–10 == features 12–13) for 95%+ of steps. The 6 new observation dims (§2) reduce to effectively one scalar (distance to junction). The agent cannot learn which way to turn.

### Proposed Fix

In `_build_entry_points()`, instead of storing the bifurcation junction coordinate, store a point **10–15mm inside each branch** along its centerline. Then `wrong_pt ≠ corr_pt` at the same bifurcation — the direction vectors actually point toward different corridors.

**File: `eve/eve/util/pathcontext.py`** — in `_build_entry_points()`:
```python
for bp in vessel_tree.branching_points:
    for branch in bp.connections:
        # Find point ~15mm along branch centerline from the junction
        interior_pt = _get_interior_point(branch, bp.coordinates, offset_mm=15.0)
        if branch not in path_set:
            wrong_entries.append(interior_pt)
        else:
            correct_entries.append(interior_pt)
```

This makes features 9–13 truly informative at bifurcations — the agent sees distinct direction signals for "avoid this way" vs "go this way."

---

## P4. Wider Tracking Points (More Markov State)

### Problem

Current tracking: `Tracking2D(intervention, n_points=3, resolution=2)` — 3 points spaced 2mm apart, covering only ~4mm near the tip. Combined with `Memory(tracking, 2, FILL)` (2-frame stack), total tracking dims = 12. The agent has no spatial information about the wire shape behind the tip, making the state non-Markov — velocity, curvature, and branch commitment are not observable from a single timestep.

### Proposed Fix

**File: `training _scripts/util/env5.py`**
```python
# Before:
tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(tracking, intervention)
tracking = eve.observation.wrapper.Memory(tracking, 2, FILL)  # 3×2×2 = 12 dims

# After:
tracking = eve.observation.Tracking2D(intervention, n_points=5, resolution=10)
tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(tracking, intervention)
# No Memory wrapper — wire shape is sufficient                  # 5×2 = 10 dims
```

5 points × 10mm spacing = coverage from tip back to ~40mm. This captures:
- **Wire curvature/fold**: mid-body points reveal bending or looping
- **Branch commitment**: shape through a bifurcation shows which branch was taken
- **Implicit velocity**: wire shape encodes recent movement direction

SOFA exposes all ~106 FEM beam DOF positions along the wire (`intervention.fluoroscopy.tracking3d`). `Tracking2D` samples `n_points` evenly-distributed points from these. Increasing `n_points` and `resolution` is a parameter change, not a code change.

Total tracking dims: 10 (down from 12), but far richer spatial information.

---

## P5. Step-Level Replay Buffer

### Problem

Currently using episode-level replay (`VanillaEpisode`). Each training sample is a full 200–1000 step episode. This is sample-inefficient — reward credit must propagate across the entire episode via multi-step returns.

### Reason

With wider tracking points (P4) removing the need for frame stacking, a single observation is approximately Markov. This enables step-level Bellman backup (standard SAC/TD3), where every individual transition `(s, a, r, s')` is a training sample.

### Proposed Fix

Switch from `VanillaEpisode` to `VanillaStep` replay buffer. Every step becomes a training sample, dramatically increasing sample efficiency. Single-step TD backup provides better credit assignment — the critic learns per-step value rather than Monte Carlo episode returns.

---

## P6. SOFA State Checkpointing (Start from Pre-Inserted State)

### Problem

All 4 targets (LCCA, LVA, RCCA, RVA) branch off from two junctions at ~383–390mm insertion depth. Every episode currently starts from 0mm insertion at the femoral entry point, spending 350+ steps navigating trivial trunk before reaching the bifurcation zone. This wastes compute and dilutes the replay buffer with easy transitions.

### Vessel Anatomy

```
Femoral entry (0mm) — coords [65, -5, 35]
  │
  ├── Bif1 (~70mm) — (21.8, -22.5, 79.2)
  │
  ├── ... main trunk (~310mm) ...
  │
  ├── Bif2a (~383mm) — Left junction
  │     ├── LCCA (Left Common Carotid, ~181mm)
  │     └── LVA  (Left Vertebral, ~158mm)
  │
  └── Bif2b (~390mm) — Right junction
        ├── RCCA (Right Common Carotid, ~176mm)
        └── RVA  (Right Vertebral, ~156mm)
```

All 4 targets originate within ~7mm of each other along the main trunk.

### Proposed Fix

**Phase 1: Generate checkpoints**

Run 5–10 heuristic episodes to ~350–380mm insertion. At target depth, save SOFA state as `.npz`:
```python
np.savez(checkpoint_path,
    xtip=m_ircontroller.xtip.value,
    rotation=m_ircontroller.rotationInstrument.value,
    index_first_node=m_ircontroller.indexFirstNode.value,
    dof_positions=DOFs.position.value,
)
```

**Phase 2: Restore at episode reset**

After normal `Sofa.Simulation.init()` (scene rebuild), restore controller state:
```python
m_ircontroller.xtip.value = saved['xtip']
m_ircontroller.rotationInstrument.value = saved['rotation']
m_ircontroller.indexFirstNode.value = saved['index_first_node']
DOFs.position.value = saved['dof_positions']
# 5 settle steps for SOFA to reconverge contact forces
for _ in range(5):
    Sofa.Simulation.animate(self.root, dt)
```

SOFA cannot serialize its scene graph (C++ objects), but the controller parameters + DOF positions are numpy arrays. The BeamAdapter recomputes beam topology from `xtip`, so restoring controller state + DOF positions gives a physically valid starting configuration.

**Impact**: Episodes drop from 400–1000 steps to 50–200 steps, focused entirely on bifurcation navigation. Multiple saved checkpoints provide diversity in starting wire shape.

---

## P7. Pre-Bifurcation Look-Ahead Rotation (Heuristic)

### Problem

The `CenterlineFollowerHeuristic` follows the path tangent at the current projection point. At bifurcations, the tangent is ambiguous — the projected segment may be on either side of the junction. The heuristic has no explicit bifurcation handling, causing it to stochastically enter the wrong branch based on rotation noise alone.

### Proposed Fix

When `d_corr < threshold` (approaching a bifurcation), compute the rotation needed to align the device tip direction with the correct branch's entry tangent (first few segments of the correct branch centerline after the junction). Apply this proactively before reaching the junction.

This is a heuristic-only improvement — it generates better seeding data for the replay buffer. The RL agent would still learn its own bifurcation strategy.

Lower priority than P1–P6 since the always-negative retraction (§8) already eliminated 59% early aborts.

---

## File Index

| File | Change Type |
|------|-------------|
| `eve/eve/util/pathcontext.py` | Extended 114→263 lines: KD-tree branch membership, entry-point classification, per-step lazy cache |
| `eve/eve/observation/localguidance.py` | Extended 8→14 dims, fixed 3D→2D coordinate consistency |
| `training _scripts/util/env5.py` | Unified detectors (fold + wrong-branch), shaped rewards, logging; removed heuristic-only detectors D1/D2/D4 |
| `training _scripts/util/heuristic_policy.py` | Always-negative retraction `[-10, -1]` when off-branch |
| `training _scripts/util/heuristic_controller.py` | Min 5mm forward push, removed rotation cap near target, removed device-length cap |
| `eve_rl/eve_rl/replaybuffer/__init__.py` | Added `EpisodeReplay` to module exports |
