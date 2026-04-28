# RL Improv 7 — Changes Reference

All changes in this branch relative to `rl_improv_4` (the latest baseline that
already has §§1–12 of RL_IMPROV_4_CHANGES.md). Built against env5 + the
PathProjectionCache infrastructure introduced in that branch.

---

## 1. P4 — Wider Tracking Observation

### Context

Original tracking stack in `env5.py`:
```python
tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(tracking, intervention)
tracking = eve.observation.wrapper.Memory(tracking, 2, FILL)
```
3 points × 2D × 2 memory frames = **12 dims**, spanning ~4 mm behind the tip.

### Reason

With only ~4 mm of wire geometry in the observation, the agent cannot see fold
or loop states. During heuristic seeding we need to judge whether a wire that
reached 380 mm insertion got there cleanly — the 3-point tracking is too short
to tell. Wider spacing also removes the need for the `Memory` stack (velocity
information is implicit in the wire-shape gradient).

### Fix

**File:** `training _scripts/util/env5.py:106-109`

```python
tracking = eve.observation.Tracking2D(intervention, n_points=5, resolution=10)
tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(tracking, intervention)
# Memory wrapper removed
```

5 points × 2D = **10 dims**, spanning ~40 mm behind the tip. No architectural
impact — `BenchAgentSynchron` reads `n_observations` dynamically from the
flattened obs vector, and `NormalizeTracking2DEpisode` auto-recomputes its
bounds at every episode reset.

Old replay / heatup caches (with 12-dim layout) become incompatible.

---

## 2. P6 (Collection Half) — SOFA-State Checkpoints at ~380 mm

### Context

Every episode starts the guidewire at 0 mm insertion at the femoral entry and
spends ~150 steps threading the trunk before reaching bif1 (~70 mm); another
~250 steps to bif2 (~383–390 mm). Most of an episode is spent on trivial trunk
progress. If we can warm-start episodes near bif2, SAC training (and future
heuristic replay) focuses 100% on the hard bifurcation-navigation problem.

### Design

**Collection only — restoration is a separate follow-up.** The script drives
the existing parallel heuristic-seeding infrastructure, but replaces
`HeuristicActionFunction` with a subclass that side-effect-writes SOFA state
to disk whenever guidewire insertion crosses into [370, 385] mm (one capture
per episode max).

### Files

#### New: `training _scripts/collect_sofa_checkpoints.py` (~230 LOC)

- `CheckpointCollectingActionFunction(HeuristicActionFunction)` — in
  `__call__`, after delegating to the parent for the heuristic action, reads
  `env.intervention.device_lengths_inserted[0]`; if in the capture window and
  the episode hasn't captured yet, dumps the live SOFA state to `.npz` + a
  metadata `.json` sidecar, then sets `_captured_in_window = True`.
- `CheckpointCollectingActionFunctionFactory` — pickleable; workers get it
  via the existing `agent.heuristic_seed(heuristic_factory=...)` task queue
  and call `.create(env)` in-process.
- `main()` — minimal `BenchAgentSynchron` setup (networks allocated at
  smallest viable size, never trained), branch-balanced schedule from a
  local copy of `build_episode_schedule()`, single call to
  `agent.heuristic_seed(episodes=N, push_to_buffer=False)`.

#### SOFA state captured per `.npz`

All directly read from the bare `SofaBeamAdapter` (valid because BenchEnv5
calls `intervention.make_non_mp()` in `__init__`, so `intervention.simulation`
is not wrapped in a multiprocessing proxy):

| Key | Source | Shape |
|---|---|---|
| `xtip` | `..._instruments_combined.m_ircontroller.xtip.value` | `(2,)` |
| `rotation_instrument` | `..._instruments_combined.m_ircontroller.rotationInstrument.value` | `(2,)` |
| `index_first_node` | `..._instruments_combined.m_ircontroller.indexFirstNode.value` | `()` scalar |
| `dof_positions` | `..._instruments_combined.DOFs.position.value` | `(~1107, 7)` |
| `tracking3d` | `intervention.fluoroscopy.tracking3d` | `(~1107, 3)` |

#### Sidecar `.json` metadata

```json
{
  "pid": 122, "episode_idx": 1, "step_idx": 142,
  "target_coordinates3d": [10.74, 60.12, 442.39],
  "inserted_lengths": [370.99, 296.79],
  "is_on_correct_branch": true,
  "wire_shape_score": 1.12e-05,
  "looks_clean": true,
  "wall_time": 1776471602.19
}
```

`wire_shape_score` = coefficient of variation (`std/mean`) of the five
consecutive node distances in `tracking3d[:5]`. `looks_clean = score < 0.25
and is_on_correct_branch`. **Not used to filter** — recorded only, so user
can sort/filter later.

### First collection run (2026-04-18)

- **200 episodes, 16 workers, 39 min wall time**
- **144 `.npz` captures** (72 % yield)
- **144 / 144 `looks_clean=true`**, all with `wire_shape_score ≈ 0` (near-zero
  spacing variance — pristine wire shape)
- 200 `EPISODE_START` lines, 0 tracebacks, 0 `CHECKPOINT_CAPTURE_ERR`
- Outcomes: 77 `wrong_branch_timeout` (42 %), 39 `wire_fold_stall` (21 %),
  30 `none` (22 × 1000-step truncations + 8 genuine target-reached)
- 4 episodes with positive total_reward (reached target without net-negative
  accumulated penalties) — ~2 % raw success rate for the heuristic

---

## 3. Findings — Branch-Detector Flip-Flop at Bifurcations

### Context

While inspecting why some episodes continue making arclength progress while
`on_br=0`, we walked `pid 578 ep 7` of the collection run — a 1000-step
episode where the wire sat at insertion ~490 mm for ~800 steps and never
fired the wrong-branch timeout despite `off_br` repeatedly climbing.

### Observations

- **242 of 539 off-branch debug steps had `reward > -0.101`** — meaning the
  arclength-progress term (+0.01·Δd_rem) outweighed the combined −0.1
  wrong-branch-step penalty and −0.001 step penalty. Several off-branch
  steps had **net-positive reward**.
- `off_br` counter trajectory across 14 consecutive INFO samples at tip
  insertion ~488–494 mm:
  `4 → 8 → 10 → 10 → 1 → 3 → 8 → 5 → 1 → 2 → 1 → 6 → 5 → 5`
  The counter resets every time `on_br` flips back to 1, so
  `OFF_BRANCH_GRACE_STEPS=20` never triggers.
- `d_corr` throughout: **4.6–7.0 mm** — tip is sitting 5 mm from a
  bifurcation junction and the KD-tree nearest-branch classification toggles
  with micrometer tip jitter.
- `wrong_pt` and `corr_pt` were **literally identical** at every sample:
  both `(18.6, 13.6, 383.2)` early, both `(-0.6, 24.5, 416.1)` later. This
  is the degenerate-junction problem where `_build_entry_points()` puts the
  shared bifurcation coordinate into **both** lists because it connects to
  one correct *and* one wrong branch.

### Conclusion

Two distinct bugs surface at bifurcations, both in
`eve/eve/util/pathcontext.py`:

1. **Degenerate entry-point observation features** (§P3 in
   RL_IMPROV_4_CHANGES.md). `_build_entry_points()` stores the raw
   bifurcation junction coordinate; since a junction connects to correct
   *and* wrong branches, the same point goes into both lists. Result:
   `wrong_entry_dir ≡ correct_entry_dir` at every bifurcation → those 6
   observation dims (features 8–13 of `LocalGuidance`) collapse to one
   scalar.
2. **KD-tree nearest-branch flip-flop**. `is_on_correct_branch()` builds a
   single KD-tree over all branch centerline coordinates. Near junctions,
   proximal segments of adjacent branches are within a few mm; tiny tip
   jitter selects different centerline points as "nearest" on consecutive
   steps, flipping the branch-membership verdict. This is why the
   wrong-branch timeout never fires on deep-stuck episodes, and why the
   arclength-reward pays out during apparent off-branch stretches.

Bug #1 is addressed by §4 below. Bug #2 is noted but not fixed in this
branch (see §6).

---

## 4. Fix P3 — Store Branch Interior Points

### Fix

**File:** `eve/eve/util/pathcontext.py`

Added module-level helper `_branch_interior_point(branch, junction_coord,
offset_mm=15.0)`. It walks along `branch.coordinates` starting from the
index nearest the junction, in the direction that moves *away* from it
(first step whose distance to the junction increases), and returns the
interpolated point at `offset_mm` of accumulated arclength. Falls back to
the branch's far endpoint if the branch is shorter than the offset.

Rewrote `_build_entry_points()` to call this helper once per
`(branching_point, connected_branch)` pair and classify the resulting
interior point by whether that specific branch is in `path_branch_set`:

```python
for bp in vessel_tree.branching_points:
    for branch in bp.connections:
        interior = _branch_interior_point(branch, bp.coordinates, offset_mm=15.0)
        if branch in path_set:
            correct_entries.append(interior)
        else:
            wrong_entries.append(interior)
```

### Effect

At every bifurcation, the correct daughter branch contributes one entry
point 15 mm *into* the correct branch from the junction; each wrong daughter
branch contributes one entry point 15 mm *into* the wrong branch. The
`wrong_pt` and `corr_pt` coordinates now differ by roughly 2 × 15 mm ×
sin(branch angle) instead of being identical. LocalGuidance features 8–13:

- `dist_wrong_entry` / `wrong_entry_dir_{x,z}` now point along the wrong
  daughter branch.
- `dist_correct_entry` / `correct_entry_dir_{x,z}` now point along the
  correct daughter branch.
- The agent gets a distinct "avoid this way" vs "go this way" gradient at
  every bifurcation (previously degenerate).

Does **not** change `is_on_correct_branch()` behaviour — that uses its own
KD-tree over all branch centerline points and still flip-flops at junctions
(bug #2, deferred — see §6).

### Validation

Not yet exercised inside the SOFA container. Smoke test when available:
pick a training step near a bifurcation, assert
`env._path_context.get_closest_wrong_entry_coords()` differs from
`get_closest_correct_entry_coords()` by ≥ 10 mm Euclidean.

---

## 5. Implement P6 Restore Path

### Fix

**File:** `eve/eve/intervention/simulation/sofabeamadapter.py`

Added method `restore_checkpoint(state_dict, settle_steps=5)`:

1. `Sofa.Simulation.reset(self.root)` — clears solver/velocity state.
   Must precede the value assignments, because doing it afterwards would
   wipe the just-assigned DOF positions.
2. Assigns `m_ircontroller.xtip.value`, `rotationInstrument.value`,
   `indexFirstNode.value`, and `DOFs.position.value` from the state dict.
3. Runs `settle_steps` × `Sofa.Simulation.animate(self.root, dt)` so the
   solver reconverges contact forces on the restored beam configuration.
4. Calls `self._update_properties()` to refresh the cached numpy views
   (`_dof_positions`, `_inserted_lengths`, `_rotations`).

Added `checkpoint: Optional[dict] = None` kwarg to `SofaBeamAdapter.reset()`.
When non-None, `restore_checkpoint(checkpoint)` is invoked at the end of
`reset()`, after the scene init / reuse path and after
`_update_properties()`.

**File:** `eve/eve/intervention/monoplanestatic.py` — `reset()`

Extracts `options["restore_checkpoint"]` (if present) and forwards it to
`self.simulation.reset(..., checkpoint=...)` via a `sim_kwargs` dict —
doesn't break existing callers that don't pass that option.

**File (new):** `training _scripts/util/checkpoint_restore.py`

`CheckpointRestoreWrapper(gym.Wrapper)` that:

- Loads and sorts all `*.npz` from `checkpoint_dir` at `__init__`.
- Sanity-checks the first file has the required keys (`xtip`,
  `rotation_instrument`, `index_first_node`, `dof_positions`); raises
  otherwise.
- In `reset(seed=, options=)`, if caller didn't supply its own
  `restore_checkpoint`, picks one `.npz` uniformly at random (seeded RNG
  for reproducibility), loads its arrays into a dict, and injects it into
  `options["restore_checkpoint"]` before delegating to `self.env.reset`.

**File:** `training _scripts/DualDeviceNav_train.py`

Added `--checkpoint_dir` CLI flag. When set, wraps `env_train` with
`CheckpointRestoreWrapper` after the optional `ActionCurriculumWrapper`.
Prints the number of checkpoints discovered at startup.

### Reset flow with `--checkpoint_dir`

```
env.reset(seed, options=None)
  → CheckpointRestoreWrapper.reset
      → options = {"restore_checkpoint": {xtip, rot, idx, dofs}}
      → self.env.reset(seed, options=options)
          → eve.Env.reset
              → MonoPlaneStatic.reset(options=options)
                  → simulation.reset(..., checkpoint=options["restore_checkpoint"])
                      → SofaBeamAdapter.reset builds/reuses scene, _update_properties
                      → SofaBeamAdapter.restore_checkpoint(checkpoint)
                          → Simulation.reset (clear solver state)
                          → assign xtip / rot / idx / DOFs.position
                          → 5 × Simulation.animate (settle contacts)
                          → _update_properties (refresh caches)
              → pathfinder.reset (picks up new dof_positions via fluoroscopy)
```

### Validation

Not yet exercised inside the SOFA container. Smoke test when available:
launch one episode with `--checkpoint_dir saved/sofa_checkpoints/selected`
and confirm the first STEP log line shows `inserted=[~370-385, ~290-300]`
instead of `[0, 0]`. If the wire snaps back to zero insertion during the
settle steps, the cause is likely `Sofa.Simulation.reset` order or missing
state — instrument around the assignments.

---

## 6. Post-Collection Selection — Curate 40 / 144

### Context

After the first collection run produced 144 `.npz` captures (§2), we needed
to pick a balanced subset per target branch for training with
`--checkpoint_dir`. The initial selection attempt (§6a below) used an
overly weak criterion and missed 3 of the 4 successful-episode captures.
The refined script in §6b replaces it.

### 6a. Initial naive selection (replaced)

Inline Python in a Bash command: k-means on `target_coordinates3d` (k=4),
sort within cluster by `(wire_shape_score, npz_filename_string)`, keep top
10. Because `wire_shape_score` is ≈ 0 for all 144 captures (pristine wire
shape is a prerequisite for `looks_clean=true`), every tie broke on
filename alphabetic order — which is *not* a meaningful signal.

By accident this happened to pick the 40 earliest-filename captures, which
also happened to be the fast-through-bif1 captures (step 138–148). So the
selection was reasonable by luck, but:

- Missed pid 198 ep 8, pid 312 ep 8, pid 122 ep 10 — all successful
  episodes that captured cleanly at step 141–145.
- Did not explicitly filter by `cum_reward_at_capture` or `step_idx`, so
  future batches could easily end up with arbitrary picks again.

### 6b. Reusable selection script

**File (new):** `training _scripts/select_sofa_checkpoints.py`

Per-cluster selection algorithm:

```
successes  = {captures whose ending EPISODE_END has heur_abort=none AND total_reward > 0}
primary    = {captures with looks_clean=True AND cum_reward_at_capture ≥ 0
              AND wire_shape_score < quality_threshold, excluding successes}
primary.sort(key=(step_idx ASC, wire_shape_score ASC, -cum_reward_at_capture))
fallback   = {remaining clean captures, sort by (step_idx, score)}

keep = successes + primary[:N-len(successes)]
     + fallback (if still short)
```

To populate `cum_reward_at_capture` and `is_success`, the script parses
each `worker_<pid>.log` for `STEP | ... ep=E | ep_step=K | cum_reward=X`
and `EPISODE_END | ... total_reward=X | heur_abort=...`. `cum_reward_at(K)`
looks up `ep_step = step_idx - 1` (the env state just before the action at
`step_idx` was applied) with nearest-neighbour fallback.

Clustering is plain k-means++ on `target_coordinates3d` (k=4) with
quadrant-sorted cluster labels (`cluster_A..D` by sign of centroid `x` and
`y` relative to overall mean) so labels stay stable across runs.

CLI:
```
python "training _scripts/select_sofa_checkpoints.py" \
    --src saved/sofa_checkpoints \
    --out saved/sofa_checkpoints/selected \
    --per_branch 10 \
    --quality_threshold 0.25
```

### Results on the 144-capture pool

| Cluster | Centroid (x, y, z) | Successes available | Kept | Step range |
|---|---|---|---|---|
| A | (+19.1, +69.8, +522.6) | 0 | 10 | 140–141 |
| B | (+21.1, +49.0, +574.9) | 1 | 10 | 139–145 |
| C | (+47.3, +54.8, +444.2) | 1 | 10 | 140–141 |
| D | (+58.2, +48.8, +534.1) | 2 | 10 | 138–144 |

All 4 / 4 successful-episode captures are now explicitly included.
`cum_reward_at_capture` across the 40 selected: **+0.37 to +1.99**, all
positive. Successful trajectories have noticeably *lower* cum_reward at
capture (+0.37 to +0.61) than fast-bif1 picks (+1.4 to +2.0) — the
successes take a slower, more lateral path through bif1 but eventually
reach the target. Good signal to remember: low cum_reward at capture isn't
automatically bad.

---

## 7. First Test-Run Findings (2026-04-19) — Two Bugs Surfaced

The first training run with `--checkpoint_dir selected/` (container
`env5_rl7_ckpttest`, `--heuristic_seeding 100`) was stopped early after two
problems became obvious within the first ~80 heuristic-seeding episodes:

1. **RNG deepcopy sync** — every worker picked the *same* `.npz` per episode
   index. Evidence: 9 distinct pids in ep 1 all finished at 332–345 steps
   with −40 to −45 reward; 8 pids in ep 2 all finished at 92–95 steps with
   −4.5 reward. Identical numerical outcomes across processes is only
   possible if they restored the same initial state.
2. **KD-tree flip-flop ceiling on success rate** — the pid 388 ep 1 / 426
   ep 5 / 312 ep 5 / … cluster of 1000-step `heur_abort=none` truncations
   (reward ≈ −200 each) matched the pid 578 ep 7 pattern from the
   collection run: tip stuck near bif2, `on_br` toggling, wrong-branch
   timeout never firing. The §4 interior-points fix doesn't touch the
   KD-tree; bug #2 was still live.

2 of 75 ended episodes reached the target (`term=True`): pid 502 ep 1 at
step 414 (cum_reward −25.15) and pid 350 ep 3 at step 283 (cum_reward
+5.82). Restore itself verified working end-to-end (short episodes,
deterministic per checkpoint, actual target-reached events from bif2
start), but effective diversity was tiny — roughly 6 distinct checkpoints
exercised across all workers.

### Fix 1 — Per-worker, per-episode RNG in `CheckpointRestoreWrapper`

**File:** `training _scripts/util/checkpoint_restore.py`

The wrapper is deepcopied into every worker in `BenchAgentSynchron._create_worker_agent`,
so a seeded `np.random.default_rng(rng_seed)` at `__init__` time produces
the same call sequence in every worker process. Replaced with a
two-tier `_pick_index(seed)` helper:

1. If the caller passed `seed=K` to `reset()` (which `heuristic_seed` does
   per episode via `episode_schedule`'s unique seeds), the wrapper derives
   a fresh `np.random.default_rng(seed).integers(...)` from it. Same seed
   → same pick (reproducible), different seeds across episodes and workers
   → diverse picks.
2. Otherwise (exploration episodes during main training, which currently
   call `env.reset()` without a seed), a per-worker RNG is initialised
   lazily on first `reset()` using `(rng_seed or 0) + os.getpid()`. Same
   worker keeps drawing from its own stream across episodes; different
   workers start with different pids and never collide.

### Fix 2 — Hysteresis debounce on `is_on_correct_branch()`

**File:** `eve/eve/util/pathcontext.py`

Added two persistent fields to `PathProjectionCache`, `_stable_on_branch`
and `_pending_flip_count`, reset only in `reset()` (*not* in
`invalidate()` — they must persist across steps). Rewrote
`is_on_correct_branch()` so:

- The raw KD-tree nearest-branch lookup is still computed every step.
- The *returned* value is the debounced `_stable_on_branch`.
- A disagreement between the raw result and `_stable_on_branch` increments
  `_pending_flip_count`; a single agreement resets it to 0.
- Only when `_pending_flip_count` reaches `_on_branch_flip_threshold`
  (default 5) does `_stable_on_branch` flip. The counter resets after.

Effect on `env5.py`'s wrong-branch detector: the `off_br` counter no
longer resets on micro-flips, so once the tip is *stably* off-branch for
~5 + `OFF_BRANCH_GRACE_STEPS` steps, `wrong_branch_timeout` reliably
fires. The 1000-step oscillator pattern disappears.

Lag trade-off: entries and exits of off-branch state are reported ~5
steps late. For the entry-penalty (one-time −1.0) this is fine — tip is
still near the wrong branch 5 steps later. For recovery it means the
off-branch penalty (−0.1/step) keeps accruing for 5 steps after the tip
has truly returned; acceptable cost.

Backward-compatible signature: `PathProjectionCache(..., on_branch_flip_threshold=5)`
with an existing-style default, so all callers (including env4.py /
env5.py instantiations) continue to work unmodified.

### Fix 5 — SOFA checkpoint restore was silently no-op; move restore AFTER `start.reset()`

**Files:** `eve/eve/intervention/simulation/sofabeamadapter.py`,
`eve/eve/intervention/monoplanestatic.py`, `training _scripts/util/env5.py`,
`training _scripts/smoke_test_restore.py` (new).

#### What was broken

`SofaBeamAdapter.restore_checkpoint()` itself worked — the diagnostic
smoke test (`smoke_test_restore.py`) showed that inside the method, all
of `xtip`, `rotationInstrument`, `indexFirstNode`, and `DOFs.position`
were correctly assigned and held through 5 settle animates:

```
RESTORE_DEBUG[post-assign] xtip=[372.6, 298.1] indexFirstNode=869 dof[-1]=[70.3, 92.0, 339.8]
RESTORE_DEBUG[after-settle-5] xtip=[372.6, 298.1] indexFirstNode=869 dof[-1]=[70.4, 92.0, 339.9]
RESTORE_DEBUG[final] inserted_lengths=[372.6, 298.1]
```

But immediately after `env.reset()` returned, `inserted` was `[0.0, 0.0]`.
The entire restored state was being wiped between the end of
`restore_checkpoint` and the return of `env.reset`.

Cause: `eve.Env.reset()` runs component resets in order:

```
self.intervention.reset(...)    # simulation.reset(checkpoint=...) runs here — wire at 372
self.start.reset(...)           # InsertionPoint.reset → intervention.reset_devices → wire at 0
self.pathfinder.reset(...)
self.interim_target.reset(...)
self.observation.reset(...)     # builds obs from zero-insertion state
self.reward.reset(...)          # ditto
...
return (self.observation(), self.info())
```

`InsertionPoint.reset()` at [eve/eve/start/insertionpoint.py:9-10](eve/eve/start/insertionpoint.py#L9-L10)
unconditionally calls `self.intervention.reset_devices()`, which is
defined at [sofabeamadapter.py:99-107](eve/eve/intervention/simulation/sofabeamadapter.py#L99-L107)
and explicitly zeroes `xtip`, `rotationInstrument`, and `indexFirstNode`,
then calls `Sofa.Simulation.reset(self.root)`. So any restore driven
from `intervention.reset` is trampled three lines later. Every run with
`--checkpoint_dir` since collection was functionally equivalent to
zero-insertion training — the "successes" we saw (pid 502 ep 1 at step
414, pid 350 ep 3 at step 283) were normal 2–3 % full-trunk successes,
not bif2 starts.

#### Fix

Drive the restore from `BenchEnv5.reset()` *after* `super().reset()`
finishes, not during `intervention.reset`:

1. `BenchEnv5.reset()` pops `options["restore_checkpoint"]` before
   calling `super().reset(...)` so intervention/start/observation/reward
   all reset normally with the wire at 0 mm.
2. After `super().reset()` returns, invoke
   `self.intervention.simulation.restore_checkpoint(ckpt)` directly —
   `InsertionPoint.start.reset()` has already run and can no longer
   clobber anything.
3. Re-reset `self.observation` and `self.reward` with the freshly
   restored tip pose so the obs returned to the caller (`flat_obs` used
   by the heuristic / RL policy on step 1) reflects 380 mm insertion,
   not 0 mm. `self._path_context.invalidate()` is also called so per-step
   caches (tip_vessel_cs, projection, is_on_correct_branch hysteresis)
   don't carry zero-state stale values.
4. Return a newly-built `(obs, info)` tuple instead of the one
   `super().reset()` built from zero-insertion state.

Also cleaned up the dead plumbing: `MonoPlaneStatic.reset()` no longer
forwards `options["restore_checkpoint"]` to `simulation.reset()`, and
`SofaBeamAdapter.reset()` still accepts the `checkpoint=` kwarg for
backward-compat but logs a debug and does nothing with it.

#### Smoke test

[`training _scripts/smoke_test_restore.py`](training%20_scripts/smoke_test_restore.py) (new):
instantiates one BenchEnv5, loads one `.npz`, calls
`env.reset(options={"restore_checkpoint": state})`, then prints
`device_lengths_inserted` and tip3d position at each of N zero-action
steps. Expected output (and confirmed post-fix):

```
[after-reset] inserted=[372.63, 298.11]   tip3d=[107.6, 89.9, 330.6]
[step-1]      inserted=[372.63, 298.11]   tip3d=[107.2, 88.3, 332.8]
[step-2]      inserted=[372.63, 298.11]   tip3d=[106.9, 86.5, 332.2]
```

Wire sits at the saved 372 mm insertion under zero actions (the small
μm-scale tip jitter is normal SOFA solver settling).

---

### Fix 3 — Bump `OFF_BRANCH_GRACE_STEPS` 20 → 50

**File:** `training _scripts/util/env5.py`

Third test run with Fixes 1 + 2 applied (container `env5_rl7_ckpttest2`)
revealed that hysteresis alone was insufficient — every episode now
terminated reliably at ~48–50 steps with `wrong_branch_timeout`, but the
wire never actually recovered. Forensic trace of pid 198 ep 1 at
ep_step=50: `cmd_action=[-5.31, 0.00, -4.25, 0.00]` with
`inserted=[62.65, 50.12]` — the wire had been in full retract mode from
380 mm down to 63 mm for ~40 steps but was still classified as
off-branch (tip still inside the wrong daughter branch during retract),
and `_off_branch_steps` kept incrementing monotonically and fired the
timeout before recovery completed.

Pre-hysteresis, the counter kept resetting every few steps due to
KD-tree flip-flops, so slow recoveries *could* eventually reach the
bifurcation junction and flip back on-branch before 20 consecutive
off-branch ticks. Fix 2 removed that accidental reprieve — so the grace
period must now be long enough to cover a complete retract-to-bif2
recovery.

Empirically: retract rate is ~5–8 mm/step × ~20 mm of wrong-branch
insertion to clear = ~3–5 steps of motion, plus hysteresis lag ~5 steps,
so **50 total steps** is a safer budget.

```python
OFF_BRANCH_GRACE_STEPS = 50  # was 20
```

If later heuristic improvements enable faster recovery, this can be
tuned down. Combined with Fix 2, the `_off_branch_steps` counter now
climbs monotonically to a meaningful threshold and fires only when the
wire truly cannot recover.

---

## 8. Heuristic Centerline-Following Controller (2026-04-22)

**Files:** `training _scripts/util/heuristic_controller.py` (new),
`training _scripts/util/heuristic_policy.py` (new)

### Context

After §§5–7, restore + heatup landed the wire at the saved 380 mm
checkpoint reliably, but the **first batch of training-mode episodes
collapsed into "do nothing"** — SAC's actor produced near-zero outputs
because random exploration around the restored pose immediately tripped
the wrong-branch detector (∼80% of random actions retract the wire into
a wrong daughter), giving a strong negative reward signal that pushed
the policy toward `gw_trans ≈ 0`. The replay buffer needed *deliberate*
trajectories — translate forward, rotate to align, occasionally retract
— before SAC could learn anything useful.

### Design

A simple centerline-following heuristic that:

1. Projects the tip onto the planned path (`pathfinder.path_points_vessel_cs`).
2. Computes a tangent direction at the projection.
3. Computes a heading-error (initially: J-tip curvature vs path tangent
   perpendicular component — replaced in §10) and a cross-track error
   (lateral offset from centerline).
4. Outputs a 4-dim action `[gw_trans, gw_rot, cath_trans, cath_rot]`
   with translation proportional to remaining path length (clamped to
   ±5 mm/s) and rotation as a `kp * heading_err + ct_kp * cross_track`
   PD-style signal (clamped to ±1.5 rad/s).
5. Catheter follows guidewire at `0.8 * gw_trans`.

The heuristic is wrapped as an `action_function` for the eve_rl worker
infrastructure (`HeuristicActionFunction` + pickleable
`HeuristicActionFunctionFactory`) so it can run in *parallel* across
all 16 workers during heatup, not sequentially. `BenchAgent.heuristic_seed`
calls `_play_episode` with this action function instead of the default
random sampler.

### Off-branch retract handler

Inside `HeuristicActionFunction.__call__()`, before delegating to the
heuristic, the wrapper checks
`base_env._path_context.is_on_correct_branch()`. If `False`, it
overrides the action to `[uniform(-10, -1), 0, 0.8*that, 0]` — pull
back toward the bifurcation with no rotation. This mirrors the
RL action-space lower bound and avoids letting the heuristic try to
"correct" itself further down a wrong daughter.

### Fold brake

Empirically, when the wire physically wedges (e.g. against the arch
wall), `gw_trans` keeps adding insertion length while `delta_s` (the
arclength projected onto the path) stalls. The env-side fold detector
(`_fold_stall_count`, incremented in `env5.step()` when `delta_gw / delta_s > 5`)
fires when this exceeds `FOLD_STALL_STEPS = 20`, truncating the
episode. To prevent the heuristic from continuing to twist into a fold
once the detector has tripped a few times, `force_translate = fold_count > 5`
is passed into `get_action()`, which forces `gw_rot = 0` until the
fold counter resets — pure forward translation, no torque, lets the
wire settle.

### Two-Phase strategy (Fix 15)

After early runs showed the wire often arriving at the restore point
with a **badly mis-oriented J-tip** (heading_err > 1.2 rad — pre-bent
the wrong way), the heuristic was extended with a Phase 1 / Phase 2
split:

- **Phase 1 (alignment):** if `|heading_err| > 1.2` and we haven't
  retracted more than a step-cap allows, output a **pure retract**
  `gw_trans = -3, gw_rot = 0`. The wire is too tangled to fix in
  place — pull it out of the bend so its torsional state can relax.
  Capped at **30 commands** (was originally 20 mm of retraction; the
  wire physically stalls at ~19 mm pullback so the distance cap was
  never crossed and Phase 1 looped forever — a step-count cap is
  decoupled from physical motion, see Fix 15).
- **Phase 2 (advance):** continuous-blend forward + rotation as
  originally designed.

`force_translate=True` (fold brake) skips Phase 1 entirely.

### Why no rotation in Phase 1

Earlier attempts rotated during retraction. Retraction itself changes
the path tangent at the tip's projection (different polyline segment
becomes "current"), which makes `heading_err` thrash via atan2 wraparound
and sign flips. Net rotation cancels out and the wire still ends up
mis-oriented. Pure translation is the cleanest way to unload torsional
stress.

---

## 9. Arclength `d_corr` Migration (2026-04-25, Fix 18a)

**Files:** `eve/eve/util/pathcontext.py`,
`training _scripts/util/env5.py`,
`eve/eve/observation/localguidance.py`

### Context

Through runs 17–26 the env logged `d_corr` and `d_wrong` as **Euclidean
distance** from the tip to the nearest interior point of the
correct/wrong branch. Three independent symptoms converged on the same
diagnosis: this metric is **systematically misleading**.

1. The interior-point set was constructed by `_branch_interior_point()`
   over **all** correct-branch members — including the trunk. Tip
   wedged at `(80, 53, 388)` (upper-trunk choke point) reported
   `d_corr ≈ 3 mm` because that point is itself an interior trunk
   marker. "Near correct entry" episodes were mostly **trunk-wedged**,
   not approaching a daughter ostium.
2. The fold-detector bypass (`d_corr improving → don't fold-truncate`)
   read this metric, so trunk-stuck episodes kept escaping the fold
   timeout because the wedge oscillation moved the tip ±0.1 mm
   relative to the trunk marker.
3. The LocalGuidance observation feature 11 (`dist_correct_entry`)
   surfaced this same misleading number to the policy.

### Design

Replace Euclidean distance with **arclength along the planned path**:

- `get_arclength_to_next_correct_entry()` — distance along the path
  from the tip's projection forward to the next bifurcation /
  junction whose downstream branch is on the planned route. Returns
  `inf` once past the last junction.
- `get_arclength_past_last_junction()` — distance along the path from
  the most recent junction (≤ tip projection) to the tip. Returns 0
  if no junction is yet behind the tip. Used to detect "inside daughter
  past the second entry zone" (≥10 mm = committed into branch).
- `get_nearest_named_branch_idx()` — returns 0/1/2/3 for
  LCCA/LVA/RCCA/RVA whichever named centerline the tip is currently
  closest to (perpendicular distance), or −1.

Both arclength helpers consume `_path_junction_arclengths`, a
precomputed sorted array of bifurcation arclengths along the planned
path, populated in `PathProjectionCache.reset()`.

### Migration in env5

- **STEP log fields:** Removed `d_wrong=`, `wrong_pt=`, `d_corr=`,
  `corr_pt=`. Added `d_corr_arc=`, `arc_past=`, `nearest_named=`,
  `entries_passed=`.
- **Fold-detector bypass:** swapped
  `get_dist_to_next_correct_entry()` → `get_arclength_to_next_correct_entry()`,
  same 0.1 mm-of-improvement threshold.
- **Heuristic policy:** publishes `_heur_arc_past_junction` and
  `_heur_nearest_named` onto the env so STEP logs can include them.

### Migration in LocalGuidance

- **Feature 11** (`dist_correct_entry`) — switched to
  `get_arclength_to_next_correct_entry()`, normalised the same way as
  before (clip at the per-episode max).
- **Feature 8** (`dist_wrong_entry`) — zeroed out. No clean arclength
  version exists for the wrong-branch direction (the path doesn't go
  there). Dimensionality preserved to avoid invalidating the policy
  network shape.
- **Features 9–10, 12–13** (entry direction unit vectors) unchanged —
  the direction toward the next correct entry is still meaningful.

### Deprecated, kept-for-now

`_correct_branch_entries`, `_wrong_branch_entries`, and
`get_dist_to_next_correct_entry()` in `pathcontext.py` are marked
`# DEPRECATED` but retained — no caller after these edits, but
keeping them lets us A/B compare if the arclength version turns out
to mislead too.

---

## 10. `heading_err` Formula Rewrite (2026-04-25, Fix 18)

**File:** `training _scripts/util/heuristic_controller.py`

### Context

Forensic analysis of run 26 revealed the original heading_err formula
returned **garbage in the trunk regime**:

```python
# OLD (run 17–26)
tangent_perp = tangent - dot(tangent, j_tip_dir) * j_tip_dir
heading_err = signed_angle(j_tip_curvature_dir, tangent_perp)
```

When the wire is well-aligned with the path (the entire trunk segment),
`tangent_perp ≈ 0`. The angle between J-tip curvature direction and a
near-zero vector is essentially **random**, saturating at ±π. Empirically:
- 47% of run-26 trunk rotation steps **improved** heading_err
- 52% **worsened** it (pure noise)
- The formula commanded `±1.5 rad/s` (clamp ceiling) the whole time
- → wire wedges at the arch top after over-rotation accumulates twist

### Fix

Compute heading_err as **`angle_between(device_dir, tangent)` signed**
about the cross product's z-component:

```python
device_dir = (tracking[0] - tracking[K]) / norm   # K=5 beam-node spacing (~5–10 mm)
cos_a = dot(device_dir, tangent)
cross_v = cross(device_dir, tangent)
sin_mag = norm(cross_v)
sign_v = sign(cross_v[2]) if abs(cross_v[2]) > 1e-3 else sign(cross_v[1])
heading_err = atan2(sin_mag * sign_v, cos_a)   # signed, [-π, π]
```

### Properties

- `|heading_err|` → 0 when wire is aligned with path (run-17-style behavior).
- Bounded, signed, no atan2 wraparound on near-zero inputs.
- No dependence on J-tip curvature direction — that signal was the
  source of trunk garbage.
- Sign is in the C-arm vertical-plane sense (cross_v[2] is world-z),
  so `gw_rot = +x` always commands rotation in the same anatomical
  direction relative to the wire's long axis.

### Side effect on Phase 1 threshold

The new formula returns **smaller** heading_err in trunk than the old
J-tip formula (which pegged at ±π). Run 27 (first run with the new
formula) showed Phase 1 / Phase 2 alternating every step because
`heading_err` hovered around the original threshold of 1.2 rad. Raised
the threshold from **1.2 → 2.0** (Fix 18b), so Phase 1 now only fires
when the tip is severely mis-oriented (≥115°) — actually pre-bent in
the wrong direction at the restore point.

---

## 11. Three-Regime Phase 2 (2026-04-25, Fix 18)

**File:** `training _scripts/util/heuristic_controller.py`

Replaces the single Phase 2 expression with three regimes selected by
position along the path:

```python
if d_corr_mm < 10.0:                          # NEAR JUNCTION (entry threading)
    gw_trans = 2.0
    gw_rot = clip(0.6*herr + 0.05*ct, ±0.3)
elif arc_past_junction_mm > 10.0:             # INSIDE DAUGHTER past entry zone
    gw_trans = float(min(5.0, d_rem * 0.1))
    gw_rot = clip(heading_kp*herr + 0.2*ct, ±0.2)   # tight clamp + strong crosstrack
else:                                          # DEFAULT (trunk, between junctions)
    gw_trans = float(min(5.0, d_rem * 0.1))
    gw_rot = clip(heading_kp*herr + 0.05*ct, ±1.5)
```

**Rationale per regime:**

- **Near-junction (d_corr_arc < 10 mm):** Slow translation (2 mm/s)
  + tight rotation clamp (±0.3) + smaller heading gain (0.6 instead of
  1.0) for precision threading into the bifurcation ostium. Crosstrack
  weight stays low because the path tangent at the junction is
  geometrically reliable.
- **Inside daughter (arc_past_junction > 10 mm):** Wire has committed
  to a branch but tends to drift laterally (the run-26 "near success
  but slid past target by 5–10 mm" pattern). Tight ±0.2 rotation cap
  + **higher crosstrack weight (0.2 vs 0.05)** re-centers the wire on
  the daughter centerline.
- **Default (trunk):** Full rotation authority (±1.5) for big course
  corrections. Note: with the new heading_err formula (§10), the
  ±1.5 clamp **rarely fires** in the trunk — `|heading_err|` stays
  below ~0.5 when the wire follows the path, vs run 26's J-tip
  formula which pegged at ±1.5 in 70–80% of trunk steps.

`d_corr_mm` and `arc_past_junction_mm` are passed in by
`HeuristicActionFunction.__call__()` from `_path_context`.

---

## 12. +1 Reward on Daughter Entry (2026-04-26, Fix 18, Set 3)

**File:** `training _scripts/util/env5.py`

### Context

Through runs 17–26 the wire **only ever entered LVA** (never LCCA,
RCCA, or RVA) in any successful or near-success episode. 84% of run-26
episodes terminated with `max_z < 380` (trunk-wedged). The reward
function provided no positive signal for entering a daughter — only
for reaching the final target — so the policy treated bif2 as a
"hover-and-wait-for-fold-timeout" zone instead of a commit point.

### Fix

```python
CORRECT_ENTRY_REWARD = +1.0   # at the top of env5.py
```

Per-episode state:
```python
self._correct_entries_seen: set[float] = set()
```

In the step loop, after the wrong-branch detector and before final
return:
```python
if not terminated and not truncated and on_correct_branch:
    arc_past = self._path_context.get_arclength_past_last_junction()
    if arc_past >= 10.0:
        junctions_arr = self._path_context._path_junction_arclengths
        proj_s = self._path_context.get_projection().s
        behind = junctions_arr[junctions_arr <= proj_s]
        if len(behind) > 0:
            current_junction_s = float(behind[-1])
            if current_junction_s not in self._correct_entries_seen:
                self._correct_entries_seen.add(current_junction_s)
                reward += CORRECT_ENTRY_REWARD
```

The 10 mm threshold matches the "inside daughter" regime in §11 — wire
must commit *into* the branch, not just touch the ostium.

### Subtle correctness fix — initialise with already-past junctions

Run 27 (first run with this change) immediately revealed a bug: every
episode logged `entries_passed=1` at step 1 because the wire was
*restored* past bif1 (and an upstream bridge curve junction). The
reward fired on step 1 of every episode, giving +1 / +2 free reward
that polluted the success signal.

Fix: in `BenchEnv5.reset()`, after the path context is invalidated,
populate the seen-set with junctions whose arclength is already
*behind* the tip projection (with the same 10 mm cushion as the
trigger):

```python
try:
    self._path_context.invalidate()
    proj_s = self._path_context.get_projection().s
    for j_arc in self._path_context._path_junction_arclengths:
        if j_arc <= proj_s - 10.0:
            self._correct_entries_seen.add(float(j_arc))
except Exception:
    pass
```

After this fix, `entries_passed=2` at episode start (the two pre-init
junctions) is **bookkeeping only** — no reward is fired. New entries
during the episode (e.g. crossing into a daughter) will increment
this set and trigger the +1.

`entries_passed` in the STEP log shows the *total* set size, including
pre-init — this is slightly misleading (the per-episode "new entries"
count is 0 + however many the wire actually crossed) but downstream
analysis can subtract the step-1 value as the baseline.

---

## 13. Diagnostic Logging — `nearest_named` and friends (2026-04-26, Fix 18, Set 4)

**File:** `training _scripts/util/heuristic_policy.py`,
`training _scripts/util/env5.py`

### Added to STEP log (env5)

- `d_corr_arc=…` — replaces old `d_corr` (arclength to next correct
  entry, `inf` past last junction).
- `arc_past=…` — arclength past most recent junction (≥10 = inside
  daughter regime).
- `nearest_named=LCCA|LVA|RCCA|RVA|none` — which named daughter the
  tip is geometrically closest to (perpendicular distance).
- `entries_passed=N` — `len(self._correct_entries_seen)`, total
  including pre-init.

### Plumbing (heuristic_policy → env5)

Inside `HeuristicActionFunction.__call__()`:

```python
base_env._heur_heading_error = float(heuristic._last_heading_error)
base_env._heur_cross_track   = float(heuristic._last_cross_track)
base_env._heur_arc_past_junction = float(heuristic._last_arc_past_junction)
idx = base_env._path_context.get_nearest_named_branch_idx()
short = "none"
if idx >= 0:
    name = getattr(base_env._path_context._branches_tuple[idx], "name", "") or ""
    for tag in ("LCCA", "LVA", "RCCA", "RVA"):
        if tag in name:
            short = tag; break
base_env._heur_nearest_named = short
```

These fields are read by `env5._step_logger` when emitting STEP INFO
lines.

### STEP log gating (unchanged)

INFO STEP lines fire on the first 30 steps of each episode, then
every 50 steps thereafter, then on terminal/truncation. All other
steps emit only a minimal DEBUG STEP (no detail). For 600-step wedge
episodes this gives ~42 detailed snapshots per episode, sufficient to
reconstruct trajectory at 50-step granularity.

### Use in offline analysis

`analyze_run28_branches.py` (new diagnostic script) parses
`diagnostics/logs_subprocesses/worker_<pid>.log`, groups by
`(pid, ep)`, classifies each target via `tracking3d_to_vessel_cs` +
nearest-named-centerline lookup against `env_train.yml`, and reports
per-branch wedge patterns (final-tip cluster, last-100 nearest_named
distribution, mean off_branch fraction, max entries_passed).

---

## 14. Run 28 Results & Failure-Mode Analysis (2026-04-27)

Container: `env5_rl7_ckpttest28`. First run with **all** of §§9–13
applied. 174 episodes parsed before the run was stopped manually.

### Outcomes by TRUE target branch

Target branch derived by transforming `target=(x,y,z)` from the
fluoroscopy / tracking3d frame back to vessel-CS via
`tracking3d_to_vessel_cs(target, image_rot_zx=(20,5), image_center=(0,0,0))`,
then nearest-centerline lookup against the four named branches in
`env_train.yml`. All 174 targets matched a centerline within
**< 0.1 mm**.

| Target | Eps | Successes | Folds (<200) | Mid (200–579) | Wedges (600) |
|--------|-----|-----------|--------------|---------------|--------------|
| LCCA   | 45  | 0         | 29           | 1             | 15           |
| LVA    | 44  | **2**     | 30           | 2             | 10           |
| RCCA   | 44  | 0         | 28           | 1             | 15           |
| RVA    | 41  | 0         | 30           | 1             | 10           |

Both successes:
- pid=540 ep=3, target_t3d=(71.4, 52.5, 437.0), R=+3.35
- pid=540 ep=6, target_t3d=(69.9, 52.5, 440.5), R=+4.42

### Confirmed via fluoro→vessel transform

Verified via direct distance to all four named centerlines after
transforming both success targets through
`tracking3d_to_vessel_cs(t, image_rot_zx=(20,5), image_center=(0,0,0))`:

| ep         | vessel-CS coord       | LVA dist | LCCA dist | RVA dist | RCCA dist |
|------------|----------------------|----------|-----------|----------|-----------|
| pid=540 ep=3 | (49.14, 35.39, 441.77) | **0.05 mm** | 19.33 mm  | 47.09 mm | 56.82 mm |
| pid=540 ep=6 | (47.73, 34.57, 445.21) | **0.07 mm** | 19.37 mm  | 44.87 mm | 57.10 mm |

Both successes lie on the **LVA centerline** (idx 10/208 and 14/208 —
early portion of LVA, just past the LCCA-LVA bifurcation). Run-26
behavior continues: **LVA remains the only daughter the agent can
reach**, even with the §11 inside-daughter regime tightening rotation
authority.

Lesson: **always run targets through the fluoro→vessel transform
before assigning anatomical labels** and **always print distance to
all 4 named centerlines** to make the choice unambiguous (a single
"nearest" answer can be misleading near a bifurcation where two
centerlines are within a few mm of each other). Tracking3d-frame
intuition is unreliable because the 20° z-rotation swaps apparent
left/right.

### Wedge patterns

For each target branch, the dominant 600-step-wedge final-tip
cluster (5 mm grid) and the **late-episode (ep_step ≥ 100)**
`nearest_named` distribution. Important: filtering for `ep_step ≥ 100`
excludes the first 30 INFO snapshots which are during trunk traversal
(tip at z < 380) where LCCA always wins by default because its
centerline starts at the lowest z (384.7) of the four named daughters.
Without this filter the "nearest_named" reading is dominated by
early-trunk steps and gives a misleading LCCA-heavy picture for LVA
wedges. With the filter:

| Target | n_wedges | Per-ep dominant nearest_named (late) | tip cluster (5 mm grid) | mean max_z | mean max_entries |
|--------|----------|---------------------------------------|--------------------------|------------|------------------|
| LCCA   | 18       | **LCCA × 18** (94% LCCA, 6% LVA)      | (80, 55, 395)            | 395        | 2.00             |
| LVA    | 11       | **LVA × 11** (91% LVA, 9% LCCA)       | (70, 50, 420)            | 423        | 3.00             |
| RCCA   | 17       | **LCCA × 17** (94% LCCA, 6% LVA)      | (80, 55, 395)            | 395        | 2.00             |
| RVA    | 11       | LCCA × 9, LVA × 2 (80% LCCA, 20% LVA) | (80, 55, 395)            | 401        | 2.20             |

### Geometric resolution — where is the tip really?

For each tip cluster, after transforming the centroid tip3d → vessel-CS
and computing distance to all four named centerline polylines:

| Tip cluster (tip3d) | vessel-CS coord    | LVA dist | LCCA dist | RVA dist | RCCA dist |
|---------------------|--------------------|----------|-----------|----------|-----------|
| (80, 53, 393) "arch wedge" | (58.0, 43.3, 399.1) | 33.93 mm | **33.12 mm** | 63.80 mm | 63.81 mm |
| (70, 50, 420) "LVA wedge"  | (48.7, 34.1, 424.6) | **5.66 mm**  | 13.58 mm     | 50.72 mm | 50.72 mm |
| (72, 56, 437) SUCCESS      | (49.0, 39.1, 438.7) | **3.65 mm**  | 21.78 mm     | 47.80 mm | 56.30 mm |

**Reading the table:**

- The **arch wedge (80, 53, 393)** — hit by 46 / 57 wedges (LCCA + RCCA + RVA
  targets) — is **~33 mm from any daughter centerline**. The wire is
  in the aortic arch itself, NOT in any branch. `nearest_named=LCCA`
  is technically true (LCCA wins by 0.8 mm out of 33 mm) but functionally
  the wire is wedged on the upper arch wall *near the LCCA ostium*,
  not *inside* LCCA. RCCA and RVA are 64 mm away — wire never gets
  to that side of the arch.
- The **LVA wedge (70, 50, 420)** — hit by 8 / 11 LVA-target wedges —
  is genuinely near LVA centerline (5.7 mm) and 2.4× farther from
  LCCA (13.6 mm). Wire IS in the LVA branch entry zone (`entries=3`
  confirms one new junction crossing), oscillating ~5 mm laterally
  off the LVA centerline.
- The **success tip** (3.34–3.65 mm from LVA) is firmly inside LVA,
  ~7 mm past bif2.

### Interpretation

**RCCA + RVA (and LCCA) — wedge in the aortic arch itself, NOT a
wrong daughter:**

- Final tip clusters at **(80, 55, 395)** for 32 of 40 such wedges.
- `max_entries = 2.00` for every single one — wire **never enters any
  named daughter** (entries=2 = pre-init bif1 + upstream bridge,
  no new junction crossings during episode).
- Mean off_branch fraction = 39–46% — wire spends ~half the episode
  off the planned path, oscillating.
- Mean tip_max_z = 395–401 — that's the **inner curvature of the
  aortic arch at the LCCA ostium height**, ~10–25 mm short of the
  RCCA opening and ~20 mm laterally off-axis from RVA's origin
  (which is the RCCA-RVA bifurcation, further posterior).
- `nearest_named=LCCA` is misleading — it just means LCCA's
  centerline starts geometrically closest to (80, 55, 395). It does
  NOT mean the wire entered LCCA; entries=2 contradicts that.

**LVA — different failure mode:**

- Final tip cluster at **(70, 50, 420)** — the wire actually crosses
  bif2 (entries=3, 1 new entry during episode), reaches z=423.
- But it then **wedges inside the LCCA branch at (70, 50, 420)**
  rather than committing to LVA. last-100 nearest_named is 75% LCCA,
  25% LVA — LVA centerline starts at (47.5, 34.5, 430.1) but the
  wire is locked against LCCA's nearer wall just above the
  bifurcation.

### Mechanism of wedge oscillation

50-step-snapshot inspection of pid=312 ep=4 (typical RVA wedge):

| step | tip3d         | nearest | on_br | off_br | d_corr_arc | arc_past |
|------|---------------|---------|-------|--------|------------|----------|
| 50   | (95, 69, 377) | LVA     | 1     | 0      | 30.1       | 342.6    |
| 100  | (84, 54, 391) | LCCA    | 1     | 0      | 12.0       | 360.6    |
| 150  | (79, 53, 396) | LCCA    | **0** | **15** | 7.9        | 364.7    |
| 200  | (80, 53, 395) | LCCA    | 1     | 0      | 8.4        | 364.3    |
| 400  | (81, 53, 394) | LCCA    | 0     | 2      | 9.1        | 363.6    |
| 500  | (81, 53, 394) | LCCA    | 0     | 7      | 9.5        | 363.2    |

Pattern: wire advances → folds against arch wall (off_br=1, fold
counter trips) → Phase 1 retracts −3 mm → wire returns → tip presents
in same orientation → re-wedges. ~50% off_br is the smoking gun. No
fold-timeout fires because the wire keeps oscillating *back* into a
correct-branch reading every few steps before
`OFF_BRANCH_GRACE_STEPS=50` accumulates.

---

## 15. Discoveries from Recent Log Analysis (Runs 26–28)

Consolidated findings from offline analysis of `diagnostics/logs_subprocesses/`
for runs 26, 27, and 28. Each item is a concrete claim backed by log
evidence; included here so future iterations don't have to re-derive
them.

### 15.1 INFO STEP gating obscures the wedge AND biases nearest_named

`env5._step_logger` writes a full INFO STEP line only on `ep_step ≤ 30`
or `ep_step % 50 == 0` or terminal/truncation. All other steps emit a
minimal DEBUG STEP with **no `tip3d`, `nearest_named`, `on_br`, etc.**
For a 600-step wedge episode this gives ~42 detailed snapshots,
NOT 600. Two consequences:

1. **Episode-length confusion.** Naive parsers that count
   `len(step_lines)` think all wedges are short fold-stalls. Correct
   true-step count comes from the `EPISODE_END | steps=NNN` line;
   `analyze_run28_branches.py` uses that and grades outcomes on the
   true count.
2. **`nearest_named` early-step bias.** Of the ~42 INFO snapshots in
   a wedge episode, **30 are from `ep_step ≤ 30`** — i.e. trunk
   traversal with tip at z < 380. During trunk traversal, the wire is
   far from any daughter centerline; LCCA always wins the
   `nearest_named` argmin by default because LCCA's polyline starts
   at the lowest z (384.7) of the four named daughters. So a naive
   "what `nearest_named` value dominates the snapshots?" aggregation
   gives a misleading LCCA-heavy answer for wedges that are actually
   near LVA. Initial Run-28 doc said "LVA wedges are 75% LCCA" — wrong;
   filtered for `ep_step ≥ 100`, LVA wedges are 91% LVA per-ep
   dominant in 11/11 episodes.

Lesson: **always filter `ep_step ≥ 100` before aggregating
`nearest_named` over wedges**, and reconcile episode length with
`EPISODE_END`.

### 15.2 Two distinct log directories with subtly different content

Run 28 has STEP/EPISODE logs in **two** sibling directories:
- `logs_subprocesses/worker_<N>.log` — indexed by worker number;
  contains the `heuristic_seed` task line with `(seed, target_branch)`
  pairs in order. Useful for matching ep N → target branch during the
  heuristic-seeding phase only.
- `diagnostics/logs_subprocesses/worker_<pid>.log` — indexed by pid;
  contains the full STEP+EPISODE_START+EPISODE_END stream including
  training-mode episodes after heuristic seeding ends. This is the
  authoritative source for trajectory analysis.

The first format only knows about the heuristic-seeding episodes;
the second covers the entire run. Mismatching them was the cause of
the early "all 116 parsed episodes are <200 steps" red herring.

### 15.3 `nearest_named=LCCA` does not mean the wire is in LCCA

`nearest_named` is the perpendicular-distance argmin over the four
named centerlines. Two flavors of LCCA-reading-but-not-in-LCCA exist:

1. **Arch wedge (LCCA / RCCA / RVA targets).** Tip3d (80, 55, 395) =
   vessel-CS (58.0, 43.3, 399.1). Distances: LCCA 33.12 mm, LVA
   33.93 mm, RVA 63.81 mm, RCCA 63.81 mm. Wire is **~33 mm from any
   daughter** — it is in the aortic arch on the upper wall, near
   where LCCA branches off, but NOT inside LCCA. `nearest_named=LCCA`
   wins by 0.8 mm out of 33 mm. Cross-check: `entries_passed=2`
   (pre-init only) confirms wire never crossed any junction during
   the episode.
2. **Trunk traversal during ep_step 1–30.** During the early-episode
   trunk push, the wire is at z < 380 — below all four daughter
   centerline starts. LCCA wins by default because LCCA starts at
   the lowest z (384.7) and is therefore closest in the +z direction.
   This is why aggregating `nearest_named` over INFO snapshots
   without an `ep_step ≥ 100` filter gives an LCCA-heavy answer
   even for episodes that genuinely commit to LVA later (see §15.1).

When `nearest_named=LVA` reads with `entries=3` and tip3d in the
(70, 50, 420) cluster — that is a real LVA-branch lock (5.7 mm from
LVA centerline), distinct from the arch wedge above.

### 15.4 Per-target-branch failure mode breakdown (run 28, n=174)

| Target | n  | S | F<200 | M | W=600 | Wedge tip cluster (5 mm grid) | Mean max_z | Mean max_entries |
|--------|----|---|-------|---|-------|--------------------------------|------------|------------------|
| LCCA   | 45 | 0 | 29    | 1 | 15    | (80, 55, 395) ×12              | 395        | 2.00             |
| LVA    | 44 | 2 | 30    | 2 | 10    | (70, 50, 420) ×8               | 423        | 3.00             |
| RCCA   | 44 | 0 | 28    | 1 | 15    | (80, 55, 395) ×13              | 395        | 2.00             |
| RVA    | 41 | 0 | 30    | 1 | 10    | (80, 55, 395) ×7               | 401        | 2.20             |

Two **geometrically distinct** wedge zones — verified by transforming
each tip3d cluster to vessel-CS and computing distance to all four
named centerline polylines:

- **(80, 55, 395) — aortic-arch wall.** vessel-CS (58.0, 43.3, 399.1),
  ~33 mm from BOTH LCCA and LVA, ~64 mm from RVA and RCCA. Wire is
  **NOT in any daughter** — it's on the upper arch wall near where
  LCCA branches off. Hit by 32 / 40 wedges of LCCA + RCCA + RVA
  targets. `entries=2` (pre-init only) confirms no junction crossings.
  Mean off-branch fraction 39–46% — wire spends ~half the episode
  oscillating off the planned path before re-snapping back via
  Phase 1 retract. RCCA and RVA sit on the opposite (right /
  posterior) side of the arch — the wire never gets there.
- **(70, 50, 420) — inside LVA branch entry zone.** vessel-CS
  (48.7, 34.1, 424.6), **5.66 mm** from LVA centerline (idx 0 = the
  branch start), 13.58 mm from LCCA. Wire IS in LVA branch — `entries=3`
  confirms the bif2 crossing — but oscillates ~5 mm laterally off the
  LVA centerline, can't thread deep enough to reach the target.
  Hit by 8 / 10 LVA wedges. The §11 "inside daughter" regime
  (±0.2 rotation + 0.2 crosstrack) was designed to fix this — it has
  a real but insufficient effect.

### 15.5 The wedge oscillation mechanism (50-step snapshot trace)

pid=312 ep=4 (RVA target, R=−46.25, 600-step wedge) — INFO snapshots:

| step | tip3d         | nearest | on_br | off_br | d_corr_arc | arc_past |
|------|---------------|---------|-------|--------|------------|----------|
| 50   | (95, 69, 377) | LVA     | 1     | 0      | 30.1       | 342.6    |
| 100  | (84, 54, 391) | LCCA    | 1     | 0      | 12.0       | 360.6    |
| 150  | (79, 53, 396) | LCCA    | **0** | **15** | 7.9        | 364.7    |
| 200  | (80, 53, 395) | LCCA    | 1     | 0      | 8.4        | 364.3    |
| 400  | (81, 53, 394) | LCCA    | 0     | 2      | 9.1        | 363.6    |
| 500  | (81, 53, 394) | LCCA    | 0     | 7      | 9.5        | 363.2    |

Pattern: wire advances → folds against arch wall (`off_br=1`,
`fold_count` trips) → Phase 1 retracts −3 mm → wire returns → tip
presents in same orientation → re-wedges. The hysteresis-debounced
`is_on_correct_branch()` keeps re-snapping back to True every few
steps before `OFF_BRANCH_GRACE_STEPS=50` accumulates, so
`wrong_branch_timeout` never fires for these episodes — they run to
the 600-step truncation.

### 15.6 Phase 1 oscillation diagnosed, threshold raised 1.2 → 2.0

Run 27 (first run with the new `angle_between` heading_err formula)
showed Phase 1 / Phase 2 alternating every step because
`|heading_err|` hovered around 1.2 rad in the trunk. The original
1.2 threshold was set for the OLD J-tip-curvature formula (which
saturated near ±π); with the new formula values cluster between 1.0
and 1.5 in trunk/approach, so 1.2 catches benign near-aligned states
and triggers spurious retracts. Raising to **2.0** (≥115° absolute)
restricts Phase 1 to genuine "pre-bent in the wrong direction"
states — see §10 closing note.

### 15.7 +1 entry-reward initially fired on step 1 (fixed)

First Run 27 episodes logged `entries_passed=1` at step 1 because
the SOFA restore lands the wire past bif1 + an upstream bridge curve
junction, and the `arc_past >= 10` trigger fired immediately. This
gave +1 free reward per episode that polluted the success signal.

Fix in `BenchEnv5.reset()`: after invalidating the path context,
populate `_correct_entries_seen` with every junction whose arclength
is ≥10 mm behind the tip projection at episode start. After this fix,
`entries=2` at episode start is bookkeeping only and the +1 only fires
on **new** crossings during the episode — see §12 closing note.

### 15.8 Always-LVA-only success bias persists across runs

Run 17 (15/380 successes, all LVA), Run 26 (28/302 episodes reached
LVA daughter, 0 reached others), Run 28 (2/174 successes, both LVA).
Across three significantly different reward / heuristic configurations
the agent has **never reached LCCA, RCCA, or RVA**. LVA's bif2 entry
geometry is apparently the most forgiving (largest ostium opening
or most-aligned tangent direction relative to the trunk approach
vector). Possible interventions, ranked by intuition:

1. Curriculum learning — train on LVA-only first to get a working
   policy, then add the other branches with LVA's policy as init.
2. Imbalance the heuristic seeding pool toward LCCA/RCCA/RVA so the
   replay buffer has nonzero positive examples for those branches
   before SAC starts.
3. Re-investigate insertion-point geometry (see §16) — the wire's
   tangent at the restored-bif1 pose may be biased toward LVA.

### 15.9 Successes are nearly always early-bif targets

Both Run 28 successes are at LVA centerline indices 10/208 and 14/208
— the **first ~7%** of the LVA centerline (just past the bifurcation,
short distance into the daughter). No success is at idx > 50 in any
run. Targets sampled deep inside daughters (idx > 100) effectively
never succeed — even when the wire enters the correct daughter, it
folds within the first 5–10 mm of intra-daughter travel.

### 15.10 The fluoro→vessel rotation is anatomically significant

`image_rot_zx=(20°, 5°)` rotates the tracking3d frame 20° around z
and 5° around x relative to vessel-CS. This is not negligible:

- LVA's vessel-CS x-range tops out at 49.2; corresponding tracking3d
  x can reach ~71 (target=(71.4, 52.5, 437) → vcs (49.1, 35.4, 441.8)).
- LCCA's vessel-CS x-range is [19, 46.8]; corresponding tracking3d
  reaches up to ~62 for nearby z values.

A target at tracking3d-x ≈ 70 *looks* like it should be on the
"high-x" branch (LVA), but my own intuition tagged it as LCCA in an
earlier exchange because I conflated tracking3d-x with vessel-CS-x.
**The transform must be applied before any anatomical reasoning.**
The `classify_target_t3d()` helper in `analyze_run28_branches.py`
makes this a one-liner; use it.

---

## 16. Open Question (Not Yet Implemented)

**Move insertion point from femoral entry up to z ≈ 300–350 mm,
remove SOFA state restore entirely.**

Trade-off:

- **Pros:** removes the brittle restore path (which has bitten us
  in §§5, 7); eliminates the pre-bent J-tip "history" the wire
  inherits from being pushed up through the arch (a fresh wire would
  arrive at the arch with rest-state J orientation, not arch-curvature-
  imprinted shape).
- **Cons:** the (80, 55, 395) wedge is fundamentally a wire-shape-
  vs-arch-geometry collision, may still happen with a fresh wire
  with different timing distribution; SOFA does not allow "floating"
  a wire mid-vessel so we'd still need ~30–50 trunk steps before
  reaching bif1; the P6 checkpoint becomes useless and we retrain
  from scratch.

Recommended approach if pursued: A/B test `insertion_position` raised
to z≈320 with ~20–30 mm pre-inserted, against the restored baseline
on the same heuristic + 50 episodes. If wedge rate at (80, 55, 395)
drops materially the wire-history hypothesis is real; otherwise the
geometry is dominant and this simplification doesn't fix the core
issue.

---

## 17. Still Not Fixed in This Branch

- **Wrong-branch oscillation counter** (P1 in RL_IMPROV_4_CHANGES.md).
  Entries instead of consecutive-steps truncation. Deferred — largely
  superseded by §7 Fix 2, which stops the counter-reset loop at its
  source.
- **Aortic-arch wedge at (80, 55, 395)** — the dominant failure mode
  for LCCA/RCCA/RVA targets in run 28. Wire never enters any daughter
  in 40 / 50 wedges of these three branches. Not a wrong-branch problem
  — it's a mechanical wedge in the main arch.
- **LCCA-wall lock at (70, 50, 420) for LVA targets** — wire commits
  past bif2 but settles against LCCA's near wall instead of threading
  into LVA. The "inside daughter" regime (§11) was supposed to fix
  this with stronger crosstrack weighting; effect is real but
  insufficient.

---

## File Index

| File | Change Type | Purpose |
|---|---|---|
| `training _scripts/util/env5.py` | Modified | §1 — P4: 5×10 mm tracking, drop Memory; §7 Fix 3 — `OFF_BRANCH_GRACE_STEPS` 20 → 50; §9 — arclength `d_corr_arc` + `arc_past` + `nearest_named` + `entries_passed` STEP fields, fold-bypass switched to arclength; §12 — `CORRECT_ENTRY_REWARD = +1.0` + `_correct_entries_seen` set + pre-init bookkeeping; §13 — diagnostic plumbing |
| `training _scripts/collect_sofa_checkpoints.py` | **New** | §2 — Parallel SOFA-state checkpoint collection |
| `eve/eve/util/pathcontext.py` | Modified | §4 — `_branch_interior_point()` helper + rewritten `_build_entry_points()`; §7 Fix 2 — hysteresis debounce in `is_on_correct_branch()`; §9 — new `_path_junction_arclengths` array, `get_arclength_to_next_correct_entry()`, `get_arclength_past_last_junction()`, `get_nearest_named_branch_idx()`; deprecated tag on `get_dist_to_next_correct_entry()` |
| `eve/eve/intervention/simulation/sofabeamadapter.py` | Modified | §5 — new `restore_checkpoint()` method; §7 Fix 5 — `checkpoint=` kwarg on `reset()` is now a no-op (actual restore driven from BenchEnv5.reset) |
| `eve/eve/intervention/monoplanestatic.py` | Modified | §5 — previously forwarded `options["restore_checkpoint"]`; §7 Fix 5 — that forwarding removed, restore driven from env-level |
| `eve/eve/observation/localguidance.py` | Modified | §9 — feature 11 (`dist_correct_entry`) switched to arclength; feature 8 (`dist_wrong_entry`) zeroed out (no clean arclength version exists for wrong-branch direction) |
| `training _scripts/smoke_test_restore.py` | **New** | §7 Fix 5 — single-process smoke test that verifies `env.reset(options={"restore_checkpoint": state})` actually lands the wire at saved `xtip` |
| `training _scripts/util/checkpoint_restore.py` | **New** / Modified | §5 — `CheckpointRestoreWrapper`; §7 Fix 1 — per-worker + per-episode-seed RNG |
| `training _scripts/util/heuristic_controller.py` | **New** | §8 — `CenterlineFollowerHeuristic`; Two-phase strategy (Phase 1 retract / Phase 2 advance); §10 — new `angle_between(device_dir, tangent)` heading_err; §11 — three-regime Phase 2; Phase 1 step-cap = 30 commands; threshold raised 1.2 → 2.0 |
| `training _scripts/util/heuristic_policy.py` | **New** | §8 — `HeuristicActionFunction` + pickleable `HeuristicActionFunctionFactory` for parallel heatup; off-branch random-retract handler; fold-brake plumbing; §13 — publishes `_heur_arc_past_junction` and `_heur_nearest_named` onto env |
| `training _scripts/DualDeviceNav_train.py` | Modified | §5 — `--checkpoint_dir` CLI flag + wrapper application; §8 — wires heuristic action function into `heuristic_seed` heatup phase |
| `training _scripts/select_sofa_checkpoints.py` | **New** | §6b — curated per-branch selection with step / cum_reward / success priority |
| `saved/sofa_checkpoints/selected/` | **Generated** | §6 — 40 curated captures (10 per cluster, 4/4 successes included) |
| `analyze_run28_branches.py` | **New** (analysis) | §13–14 — parses `diagnostics/logs_subprocesses/worker_<pid>.log`, transforms targets via `tracking3d_to_vessel_cs`, classifies vs `env_train.yml` named centerlines, reports per-branch wedge patterns |
| `extract_branch_coords.py` | **New** (analysis) | §14 — extracts named-branch centerline coordinates from `env_train.yml` (handles python-tuple yaml tags) |
| `classify_targets.py` | **New** (analysis) | §14 — standalone target → branch classifier (used to verify the rotation-transform mapping reaches sub-mm fit on all 167 targets) |
| `RL_IMPROV_7_CHANGES.md` | **New** | This document |
