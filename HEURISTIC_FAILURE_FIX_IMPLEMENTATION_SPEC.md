# Heuristic Failure Fix Implementation Spec (Final)

## Scope

This spec covers the remaining heuristic-seeding fixes after the codebase updates that already added:

- branch-balanced episode scheduling in `DualDeviceNav_train.py`
- per-episode `(seed, options)` scheduling through the worker stack
- `CenterlineRandom(target_branch=...)`
- deferred push plus minimum-success-rate filtering

The remaining work is to fix the main bad-data modes seen in:

- [heuristic cache](/D:/rl_improv_3/saved/env5_200ep/heuristic_cache.npz)
- worker logs in [logs_subprocesses](/D:/rl_improv_3/saved/env5_200ep/logs_subprocesses)

This spec is for the current `env5` path, but the same logic can be mirrored to `env4` if needed.

## Evidence Summary

From the first 20 heuristic episodes in each of 16 worker logs (`320` episodes total):

- `116/320` were clear `both devices maxed / no insertion progress` stalls.
- These wasted `59,600` steps after the first `both delta_ins ~= 0` near max length.
- That is `28.5%` of all logged steps in this slice.
- Mean wasted tail in those episodes was `513.8` steps.
- Late stuck-step time was about `2.71x` slower than early episode step time.
- `40/320` were `GW maxed first` partial-stall episodes.
- `71/320` were short positive truncations: truncated, not successful, but still positive reward.
- `45/320` were actual successes.

From the saved cache:

- failed episodes with `min d_rem < 0.1` were off-branch at closest approach
- positive-return failures ended off-branch
- a few off-branch-close episodes recovered, usually within `2`, `3`, or `7` steps
- one long `229`-step recovery existed but still had poor total reward

## Design Principles

1. Only heuristic seeding should use the new early-abort detectors. Evaluation and normal RL exploration should not use this heuristic-only shaping unless intentionally enabled.
2. Reward semantics should stay consistent between heuristic data and later RL data where possible.
3. Mechanical dead-episodes should be terminated early.
4. Wrong-branch episodes must be aborted aggressively — not slowed down.
5. Recovery should be bounded and short, not endless meandering.
6. Heuristic aborts get explicit negative penalties (heuristic only). Vessel-end truncations get explicit negative penalties (both heuristic and RL). Plain max-step truncations do not — they already accumulate enough negative step reward.

## Recommended Thresholds

### Heuristic-mode activation

- `heuristic_mode = True` only for heuristic seeding episodes
- pass via `options` on reset

### Detector thresholds

- `NEAR_MAX_MM = 2.0`
- `NO_PROGRESS_MM = 0.10`
- `BOTH_MAX_STALL_STEPS = 8`
- `GW_PARTIAL_STALL_GRACE_STEPS = 25`
- `OFF_BRANCH_GRACE_STEPS = 10`
- `OFF_BRANCH_INSERTION_DIST_MM = 20.0`
- `NEAR_TARGET_DREM_NORM = 0.10`
- `SATURATED_ROT_RAD = 1.30`
- `SATURATED_ROT_STEPS = 6`

### Reward adjustments

- `TargetReached(factor=3.0)` — change from `factor=1.0` to `factor=3.0` in `env5.py` constructor
- `FAILURE_TRUNCATION_PENALTY = -5.0`

## Exact Failure Modes And Fixes

### 1. Both devices maxed and stuck

Observed signature:

- both inserted lengths near max
- both `delta_ins ~= 0`
- forward commands continue
- episode runs hundreds more steps until max-step truncation

Fix:

- in heuristic mode, detect when both devices are near max and both insertion deltas stay below `NO_PROGRESS_MM` for `BOTH_MAX_STALL_STEPS`
- abort immediately
- apply `FAILURE_TRUNCATION_PENALTY`
- record explicit abort reason in `info`

Abort condition (all must hold for `8` consecutive steps):

- `gw_inserted >= gw_max - NEAR_MAX_MM`
- `cath_inserted >= cath_max - NEAR_MAX_MM`
- `abs(delta_gw) <= NO_PROGRESS_MM`
- `abs(delta_cath) <= NO_PROGRESS_MM`

Expected effect:

- remove the largest wall-clock waste source (59,600 steps / 28.5% of all compute)
- stop collecting hundreds of masked/no-op tail transitions

### 2. GW maxed first, long partial stall

Observed signature:

- guidewire insertion is capped first
- catheter may still advance for some time
- some episodes can still succeed here
- many others keep drifting without meaningful improvement

Fix:

- do not terminate immediately when only GW is maxed
- start a grace window when:
  - `gw_inserted >= gw_max - NEAR_MAX_MM`
  - `abs(delta_gw) <= NO_PROGRESS_MM`
  - catheter is not yet both-max stalled
- allow `GW_PARTIAL_STALL_GRACE_STEPS = 25`
- during grace, require at least one of:
  - success (terminated)
  - `d_rem_norm` improves by at least `0.01` from entry value
  - catheter inserted length increases by at least `1.0 mm` cumulatively from entry value
- if none happen by the end of grace, abort with `FAILURE_TRUNCATION_PENALTY`

Rationale:

- keeps legitimate catheter-catch-up cases alive
- cuts off the long partial stalls that never recover

### 3. Wrong-branch episodes (aggressive abort)

Observed signature:

- `on_correct_branch == 0` (cross-track distance >= 5mm from correct path)
- heuristic continues pushing forward into wrong branch
- sometimes low `d_rem_norm` on a parallel branch creates illusion of proximity
- episodes meander for hundreds of steps without recovering

Design philosophy:

**Do not slow down on the wrong branch. Abort fast.** The heuristic should learn that the best thing to do is not go on the wrong branch. If the episode enters the wrong branch for a measurable distance, it should stall and abort rather than meander.

Fix:

Track when the episode first becomes off-branch:

- `self._off_branch_start_step` — episode step when `on_correct_branch` first became `0`
- `self._off_branch_start_inserted_gw` — GW inserted length at that moment

Abort if ANY of these conditions hold while `on_correct_branch == 0`:

**Condition A — general wrong-branch timeout:**
- `off_branch_steps >= OFF_BRANCH_GRACE_STEPS` (10)
- Rationale: data shows real recoveries happen in 2, 3, and 7 steps. 10 covers those.

**Condition B — measurable wrong-branch insertion:**
- `gw_inserted - off_branch_start_inserted_gw >= OFF_BRANCH_INSERTION_DIST_MM` (20.0 mm)
- Directional check only (no `abs()`): the goal is to detect advancing deeper into the wrong branch. Using `abs()` would also count retraction/recovery motion as failure if retract mode is added later.
- Rationale: if the device has advanced 20mm while off-branch, it is actively going deeper into the wrong branch. Abort even before the 10-step timeout.

Recovery success condition (resets all off-branch tracking):

- `on_correct_branch == 1`

If the device returns to the correct branch at any point during the grace window, all off-branch state resets. This preserves the genuine short recoveries (2-7 steps) observed in the cache.

Penalty on abort: `FAILURE_TRUNCATION_PENALTY = -5.0`

### 4. Rotation saturation as pseudo-close acceleration

Observed signature:

- many wrong-branch-close tails had repeated saturated rotation near the cap
- the wrong-branch detector (Section 3) gives a general 10-step grace
- this detector accelerates abort for the specific bad pattern: wrong branch + pseudo-close + spinning

Fix:

- in heuristic mode, track consecutive steps where ALL of:
  - `on_correct_branch == 0`
  - `d_rem_norm < NEAR_TARGET_DREM_NORM` (0.10)
  - `abs(gw_rot_cmd) >= SATURATED_ROT_RAD` (1.30)
- if that holds for `SATURATED_ROT_STEPS = 6`, abort with `FAILURE_TRUNCATION_PENALTY`

Rationale:

- the wrong-branch detector alone allows up to 10 steps for all off-branch episodes
- this detector fires at 6 steps specifically when the heuristic is clearly in the worst pattern: wrong branch, looks close, and spinning at max rotation
- avoids killing a normal wrong-branch episode at 5 steps just because `d_rem_norm` happens to be low — it must also be spinning

### 5. True overshoot on correct branch (Phase 2)

Current evidence:

- not the main failure mode in the analyzed cache
- defer to Phase 2

Fix (when implemented):

- track `best_d_rem_norm` throughout episode
- if `on_correct_branch == 1` and `d_rem_norm > best_d_rem_norm + 0.015` and `best_d_rem_norm < 0.10`:
  - enter recovery for `10` steps
  - if `d_rem_norm` doesn't improve back to within `0.01` of best, abort

### 6. Positive-return truncations that look successful

Observed signature:

- short truncated episodes around `150-300` steps
- still moving forward
- total reward positive despite failure

Fix:

- apply `FAILURE_TRUNCATION_PENALTY = -5.0` in two cases:
  1. heuristic abort triggered (any detector, heuristic only) — because early abort removes hundreds of future negative step penalties, so the episode needs an explicit penalty to stay clearly negative
  2. vessel-end truncation without success (both heuristic and RL) — because these can be short episodes with positive accumulated reward from ArcLengthProgress despite failure. A vessel-end truncation is always bad — the device went past the end of the vessel tree.
- plain max-step truncations do NOT get extra `-5.0` — they already accumulate enough negative reward from 1000 steps of `Step(factor=-0.001)`

```python
# Heuristic abort penalty (heuristic mode only)
if self._heuristic_mode and self._heuristic_abort_reason is not None:
    reward += FAILURE_TRUNCATION_PENALTY

# Vessel-end penalty (always — both heuristic and RL)
elif truncated and not terminated and self._vessel_end_trunc.truncated:
    reward += FAILURE_TRUNCATION_PENALTY
```

Rationale:

- makes short positive-reward failures clearly negative
- vessel-end penalty during RL prevents the critic from learning that going past the vessel end is acceptable
- avoids double-punishing episodes that already ran 1000 steps of step penalty
- improves replay label quality

### 7. Success reward too weak

Observed signature:

- some successful episodes still have negative total reward

Fix:

- change `TargetReached(factor=1.0)` to `TargetReached(factor=3.0)` in the `env5.py` constructor
- no post-step reward shaping needed for success — the reward component handles it directly

Rationale:

- cleaner than adding a post-hoc `+2.0` bonus in `step()`
- makes real success clearly separable from short bad truncations
- successful episodes should always have clearly positive total reward
- applies to both heuristic and RL episodes consistently (Design Principle 2)

## Controller Changes

File:

- `training _scripts/util/heuristic_controller.py`

### Root cause analysis

The current `gw_trans = max(gw_trans, 5.0)` should be removed, but it is not enough by itself to prevent the observed stalls.

Reason:

- if the tip is on a wrong branch and `d_rem` remains large, then `0.1 * d_rem` can still command strong forward motion even without the `5.0` floor
- the max-length stall requires either an env-side abort or a controller cap based on remaining device length

### Required controller changes

#### A. Replace unconditional minimum forward push with simpler baseline

Current code:

```python
gw_trans = max(gw_trans, 5.0)  # line 106
```

Replace with:

```python
gw_trans = min(self.max_translation, max(0.0, 0.1 * d_rem))
```

This removes the `5.0` floor and lets `gw_trans` drop to `0.0` when `d_rem` is small.

#### B. Add device-length-aware cap

The controller must not command forward insertion that the device physically cannot deliver. Apply after the baseline and after catheter translation is computed:

```python
# Baseline
gw_trans = min(self.max_translation, max(0.0, 0.1 * d_rem))
cath_trans = gw_trans * self.catheter_follow_ratio

# Device-length cap (applied after both translations exist)
gw_remaining = gw_max - gw_inserted
cath_remaining = cath_max - cath_inserted
dt = 1.0 / fluoroscopy.image_frequency

gw_trans = min(gw_trans, max(0.0, (gw_remaining - 1.0) / dt))
cath_trans = min(cath_trans, max(0.0, (cath_remaining - 1.0) / dt))
```

This prevents the heuristic from repeatedly asking for impossible forward insertion near the device-length limit. The `- 1.0` margin avoids commanding right up to the limit where SOFA masking kicks in.

The controller needs to read `gw_max`, `gw_inserted`, `cath_max`, `cath_inserted` from the intervention. These are available via `intervention.device_lengths_inserted` and `intervention.device_lengths_maximum`.

#### C. Reduce rotation cap near target

- default cap: the env action-space rotation bound
- if `d_rem <= 50 mm`: clip `gw_rot` to `[-0.8, 0.8]`

Purpose:

- reduces oscillation near the target
- simple, no new state required

#### D. Retract line gating (Phase 2)

Current code:

```python
gw_trans = max(gw_trans, 0.0)  # line 143 — never retract
```

No change needed for Phase 1. The env-side wrong-branch detector aborts the episode before the controller would need to retract.

When recovery mode is implemented in Phase 2, this line must be gated:

- clamp to nonnegative only in normal mode
- allow bounded negative translation in recovery mode

## Env Changes

File:

- `training _scripts/util/env5.py`

### Required state to add on env object

Add heuristic-only episode state in `BenchEnv5.__init__` and reset in `reset()`:

```python
# Heuristic mode state
self._heuristic_mode = False
self._heuristic_abort_reason = None

# Best tracking
self._best_d_rem_norm = float('inf')

# Per-step insertion tracking (separate from INFO-only logging tracker)
# BenchEnv5 currently updates _prev_inserted for logging only every 50 steps
# or at terminal. Detectors need every-step deltas, so use a separate tracker.
self._det_prev_inserted = [0.0, 0.0]

# Both-max stall detector
self._both_max_stall_count = 0

# GW partial stall detector
self._gw_partial_stall_count = 0
self._gw_partial_drem_at_entry = None
self._gw_partial_cath_inserted_at_entry = None

# Wrong-branch detector
self._off_branch_steps = 0
self._off_branch_start_inserted_gw = None

# Rotation saturation detector
self._sat_rot_count = 0
```

### Activate heuristic mode

In `reset(seed=None, options=None)`:

- set `self._heuristic_mode = bool(options and options.get("heuristic_mode", False))`
- reset all counters/state above to initial values

### Read guidance values every step

In `step()` after `super().step(action)`:

- extract `guidance = obs["guidance"]`
- use:
  - `d_rem_norm = guidance[0]`
  - `on_correct_branch = bool(round(guidance[7]))`

### Access intervention device data

Use the existing accessors from `MonoPlaneStatic` (not `simulation.inserted_lengths` directly):

```python
inserted = self.intervention.device_lengths_inserted  # [gw, cath]
max_lens = self.intervention.device_lengths_maximum    # [gw, cath]
```

Compute per-step deltas using the detector-specific tracker:

```python
delta_gw = inserted[0] - self._det_prev_inserted[0]
delta_cath = inserted[1] - self._det_prev_inserted[1]
self._det_prev_inserted = [inserted[0], inserted[1]]
```

Do not reuse the logging `_prev_inserted`.

### Store truncation/terminal component references

In `__init__`, keep references for post-step reward shaping:

```python
self._target_terminal = terminal
self._max_steps_trunc = max_steps
self._vessel_end_trunc = vessel_end
self._sim_error_trunc = sim_error
```

### Heuristic abort logic in `step()`

Only if `self._heuristic_mode` and not already `terminated` or `truncated`:

1. Update `self._best_d_rem_norm = min(self._best_d_rem_norm, d_rem_norm)`
2. Read inserted lengths and compute deltas
3. Evaluate detectors in priority order:

**Detector 1: both_max_stall**

```python
if (inserted[0] >= max_lens[0] - NEAR_MAX_MM and
    inserted[1] >= max_lens[1] - NEAR_MAX_MM and
    abs(delta_gw) <= NO_PROGRESS_MM and
    abs(delta_cath) <= NO_PROGRESS_MM):
    self._both_max_stall_count += 1
else:
    self._both_max_stall_count = 0

if self._both_max_stall_count >= BOTH_MAX_STALL_STEPS:
    abort("both_max_stall")
```

**Detector 2: gw_partial_stall**

```python
if (inserted[0] >= max_lens[0] - NEAR_MAX_MM and
    abs(delta_gw) <= NO_PROGRESS_MM and
    not (inserted[1] >= max_lens[1] - NEAR_MAX_MM)):  # not both-max
    if self._gw_partial_stall_count == 0:
        self._gw_partial_drem_at_entry = d_rem_norm
        self._gw_partial_cath_inserted_at_entry = inserted[1]
    self._gw_partial_stall_count += 1

    drem_improved = d_rem_norm < self._gw_partial_drem_at_entry - 0.01
    cath_advanced = inserted[1] > self._gw_partial_cath_inserted_at_entry + 1.0

    if self._gw_partial_stall_count >= GW_PARTIAL_STALL_GRACE_STEPS:
        if not drem_improved and not cath_advanced:
            abort("gw_partial_stall")
else:
    self._gw_partial_stall_count = 0
    self._gw_partial_drem_at_entry = None
    self._gw_partial_cath_inserted_at_entry = None
```

**Detector 3: wrong_branch (aggressive abort)**

```python
if not on_correct_branch:
    if self._off_branch_steps == 0:
        # First step off-branch: record entry state
        self._off_branch_start_inserted_gw = inserted[0]
    self._off_branch_steps += 1

    # Condition A: general timeout
    if self._off_branch_steps >= OFF_BRANCH_GRACE_STEPS:
        abort("wrong_branch_timeout")

    # Condition B: measurable forward insertion while off-branch (directional, no abs)
    elif (inserted[0] - self._off_branch_start_inserted_gw) >= OFF_BRANCH_INSERTION_DIST_MM:
        abort("wrong_branch_insertion")

else:
    # Back on correct branch: reset all off-branch tracking
    self._off_branch_steps = 0
    self._off_branch_start_inserted_gw = None
```

**Detector 4: sat_rot_bad_recovery**

```python
gw_rot_cmd = action[1]  # raw action before super().step()
if (abs(gw_rot_cmd) >= SATURATED_ROT_RAD and
    d_rem_norm < NEAR_TARGET_DREM_NORM and
    not on_correct_branch):
    self._sat_rot_count += 1
else:
    self._sat_rot_count = 0

if self._sat_rot_count >= SATURATED_ROT_STEPS:
    abort("sat_rot_bad_recovery")
```

### Abort implementation

When a detector triggers:

```python
def _heuristic_abort(self, reason, info):
    self._heuristic_abort_reason = reason
    info["heuristic_abort"] = True
    info["heuristic_abort_reason"] = reason
    return True  # signals truncation
```

### Reward shaping in `step()`

Applied AFTER `super().step()` and AFTER detector evaluation, so final reward is what gets stored.

Success reward is handled by `TargetReached(factor=3.0)` in the reward component directly — no post-step shaping needed.

Failure penalty:

```python
# Heuristic abort penalty (heuristic mode only)
if self._heuristic_mode and self._heuristic_abort_reason is not None:
    reward += FAILURE_TRUNCATION_PENALTY  # -5.0

# Vessel-end penalty (always — both heuristic and RL)
elif truncated and not terminated and self._vessel_end_trunc.truncated:
    reward += FAILURE_TRUNCATION_PENALTY  # -5.0
```

The penalty applies to:
- heuristic abort (any detector, heuristic only) — early abort removes future step penalties, so explicit penalty needed
- vessel-end truncation (both heuristic and RL) — a vessel-end truncation is always a failure; short vessel-end episodes can accumulate positive reward from ArcLengthProgress which sends the wrong signal to the critic

The penalty does NOT apply to:
- plain max-step truncations — already accumulate ~1.0 of negative step reward over 1000 steps
- sim-error truncations — rare, not worth special-casing

### Logging additions

Extend the INFO step log line to include when in heuristic mode:

```
heur=1 | d_rem_n=0.234 | on_br=1 | off_br_steps=0 | stall=0/8 | gw_stall=0/25 | abort=none
```

At episode end, log the abort reason if any:

```
EPISODE_END | ... | heur_abort=wrong_branch_timeout
```

## Seeding Schedule Changes

File:

- `training _scripts/DualDeviceNav_train.py`

### Update `build_episode_schedule` for heuristic seeding

Add a `heuristic_mode` parameter:

```python
def build_episode_schedule(n_episodes, branches, base_seed=42, heuristic_mode=False):
    ...
    options = {"target_branch": branch}
    if heuristic_mode:
        options["heuristic_mode"] = True
    schedule.append((ep_seed, options))
```

In the heuristic seeding block, call with `heuristic_mode=True`:

```python
schedule = build_episode_schedule(
    batch_size, TARGET_BRANCHES, base_seed=42 + seed_offset,
    heuristic_mode=True,
)
```

Do not add `heuristic_mode` to eval episodes. Heatup can optionally use it later, but not in this patch.

## Worker Stack Changes

Files:

- `eve_rl/eve_rl/agent/single.py`
- `eve_rl/eve_rl/agent/singelagentprocess.py`
- `eve_rl/eve_rl/agent/synchron.py`

The scheduling and `push_to_buffer=False` support are already present. No new API changes are required for this patch beyond making sure:

- `episode_schedule` carries `heuristic_mode=True` in options
- options pass through unchanged to `env.reset(options=)`

## Implementation Phases

### Phase 1 — Implement all of these

| # | Change | File | Targets | Expected Impact |
|---|--------|------|---------|-----------------|
| 1 | `heuristic_mode` flag + state init/reset | env5.py | Plumbing | Enables all detectors |
| 2 | Per-step insertion delta tracking (`_det_prev_inserted`) | env5.py | Plumbing | Feeds detectors (separate from logging tracker) |
| 3 | Truncation/terminal component references | env5.py | Plumbing | Enables reward shaping |
| 4 | `both_max_stall` detector (8 steps → abort) | env5.py | 116 episodes | Eliminates 59,600 wasted steps (28.5% compute) |
| 5 | `wrong_branch` detector (2-condition: timeout + insertion) | env5.py | Cache evidence, off-branch episodes | Prevents wrong-branch meandering |
| 6 | `gw_partial_stall` detector (25-step grace) | env5.py | 40 episodes | Cuts GW-maxed-first stalls |
| 7 | `sat_rot` detector (6 steps, pseudo-close acceleration) | env5.py | Supporting | Catches wrong-branch spinning before 10-step timeout |
| 8 | Failure truncation penalty `-5.0` on heuristic aborts + vessel-end truncations | env5.py | 71+ episodes | Fixes replay buffer poisoning |
| 9 | `TargetReached(factor=3.0)` (was 1.0) | env5.py | 45 episodes | Makes success clearly positive |
| 10 | Heuristic logging fields | env5.py | Debuggability | Faster future analysis |
| 11 | Simpler forward baseline (`min(max_trans, max(0, 0.1*d_rem))`) | heuristic_controller.py | Root cause contributor | Removes unconditional 5.0 floor |
| 12 | Device-length-aware cap (`(remaining - 1.0) / dt`) | heuristic_controller.py | Root cause of max-length stalls | Prevents impossible insertion commands |
| 13 | Rotation cap near target (`d_rem <= 50mm → ±0.8`) | heuristic_controller.py | Supporting | Reduces oscillation |
| 14 | `heuristic_mode=True` in episode schedule | DualDeviceNav_train.py | Plumbing | Activates all env detectors |

### Phase 2 — Deferred

| # | Change | Reason to defer |
|---|--------|-----------------|
| 15 | Correct-branch overshoot detector | Not the main failure mode per cache data |
| 16 | Controller recovery mode (retract -3.0mm/s) | Needs Phase 1 data to validate; env abort makes it less critical. When implemented, must gate `max(gw_trans, 0.0)` so recovery can use bounded negative translation. |
| 17 | Intervention accessor helpers | Convenience only; direct property access works |

## Implementation Order (within Phase 1)

1. **env5.py** — all detector + reward + logging changes
   - heuristic-mode state variables
   - per-step insertion delta tracking (`_det_prev_inserted`)
   - truncation/terminal component references
   - all 4 detectors (both_max_stall, wrong_branch, gw_partial_stall, sat_rot)
   - `TargetReached(factor=3.0)` in constructor
   - failure penalty reward shaping
   - logging fields
2. **heuristic_controller.py** — forward baseline + device-length cap + rotation cap
   - replace `max(gw_trans, 5.0)` with `min(max_translation, max(0.0, 0.1 * d_rem))`
   - add device-length-aware cap using `(remaining - 1.0) / dt`
   - rotation cap near target
3. **DualDeviceNav_train.py** — schedule plumbing
   - `build_episode_schedule(heuristic_mode=True)` for heuristic seeding

## Acceptance Criteria

After implementation, rerun heuristic seeding and verify:

### Mechanical criteria

- `both-max-stall` episodes should almost disappear (was 116/320)
- wrong-branch episodes should abort within 10 steps of going off-branch
- mean heuristic episode duration should drop materially
- average late-step time inflation should reduce because long stuck tails are gone

### Reward criteria

- short truncated failures should no longer have positive total reward
- successful episodes should usually have positive total reward (now +3.0 target bonus)
- clear bimodal separation between success and failure reward distributions

### Data-quality criteria

- fewer max-step truncations
- fewer episodes ending after hundreds of zero-progress steps
- no replay episodes with obvious "forward command while both delta_ins = 0 for hundreds of steps"
- no replay episodes with 100+ steps on the wrong branch

### Logging criteria

- logs clearly show `heuristic_abort_reason` with specific detector name
- logs show `d_rem_norm` and `on_branch` at INFO steps in heuristic mode
- episode-end log shows abort reason when applicable

### Cache invalidation and revalidation

Any existing `heuristic_cache.npz` files were generated with the old (broken) logic and must be regenerated after implementation.

After implementing this patch:

- delete or archive old heuristic caches
- regenerate `heuristic_cache.npz`
- rerun the validator and log parser on the new cache/logs
- confirm the original failure modes (116 stalls, 71 positive truncations, off-branch meandering) are resolved
