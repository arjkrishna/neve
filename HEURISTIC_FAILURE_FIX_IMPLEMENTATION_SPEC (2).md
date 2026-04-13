# Heuristic Failure Fix Implementation Spec

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

1. Only heuristic seeding should use the new early-abort detectors.
2. Reward semantics should stay consistent between heuristic data and later RL data where possible.
3. Mechanical dead-episodes should be terminated early.
4. Wrong-branch truncations must be clearly negative.
5. Recovery should be bounded, not endless.

## Recommended Thresholds

These are the concrete values to implement first.

### Heuristic-mode activation

- `heuristic_mode = True` only for heuristic seeding episodes
- pass via `options` on reset

### Detector thresholds

- `NEAR_MAX_MM = 2.0`
- `NO_PROGRESS_MM = 0.10`
- `BOTH_MAX_STALL_STEPS = 8`
- `GW_PARTIAL_STALL_GRACE_STEPS = 25`
- `NEAR_TARGET_DREM_NORM = 0.10`
- `OFF_BRANCH_RECOVERY_STEPS = 10`
- `OVERSHOOT_DREM_WORSEN_NORM = 0.015`
- `SATURATED_ROT_RAD = 1.30`
- `SATURATED_ROT_STEPS = 6`

### Reward adjustments

- `TARGET_REACHED_REWARD = +3.0`
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
- terminate immediately
- apply `FAILURE_TRUNCATION_PENALTY`
- record explicit abort reason in `info`

Abort condition:

- `gw_inserted >= gw_max - NEAR_MAX_MM`
- `cath_inserted >= cath_max - NEAR_MAX_MM`
- `abs(delta_gw) <= NO_PROGRESS_MM`
- `abs(delta_cath) <= NO_PROGRESS_MM`
- true for `8` consecutive steps

Expected effect:

- remove the largest wall-clock waste source
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
  - success
  - `d_rem_norm` improves by at least `0.01`
  - catheter inserted length increases by at least `1.0 mm` cumulatively
- if none happen by the end of grace, abort with `FAILURE_TRUNCATION_PENALTY`

Rationale:

- keeps legitimate catheter-catch-up cases alive
- cuts off the long partial stalls

### 3. Wrong-branch pseudo-close states

Observed signature:

- `d_rem_norm` can look small on a parallel branch
- not actually on the correct branch
- heuristic can then spin or drift for a long time

Fix:

- gate all near-target logic on both:
  - `d_rem_norm < NEAR_TARGET_DREM_NORM`
  - `on_correct_branch`
- if `d_rem_norm < 0.10` but `on_correct_branch == 0`, enter bounded recovery mode
- allow `OFF_BRANCH_RECOVERY_STEPS = 10`
- if branch is not recovered within 10 steps, abort with `FAILURE_TRUNCATION_PENALTY`

Recovery success condition:

- `on_correct_branch == 1`

Optional secondary recovery success:

- `d_rem_norm` improves by at least `0.01` and branch flag returns to `1`

Rationale:

- preserves the short real recoveries seen in the cache
- prevents hundreds of steps of wrong-branch looping

### 4. Rotation saturation in bad recovery loops

Observed signature:

- in the cache, many wrong-branch-close tails had repeated saturated rotation near the cap
- in logs, this is less universal than the max-length stall, so it should be a supporting detector, not the only detector

Fix:

- in heuristic mode, track consecutive steps where:
  - `abs(gw_rot_cmd) >= SATURATED_ROT_RAD`
  - `d_rem_norm < NEAR_TARGET_DREM_NORM`
  - `on_correct_branch == 0`
- if that holds for `SATURATED_ROT_STEPS = 6`, abort with `FAILURE_TRUNCATION_PENALTY`

Rationale:

- catches the spinning/oscillation failure family early
- especially useful when the episode is not yet both-max stalled

### 5. True overshoot on correct branch

Current evidence:

- not the main failure mode in the analyzed cache
- still worth handling conservatively

Fix:

- do not add a hard “correct-branch overshoot = immediate failure” rule yet
- instead:
  - reduce forward aggressiveness near target
  - allow brief recovery
  - only abort if recovery fails

Correct-branch overshoot entry condition:

- `on_correct_branch == 1`
- current `d_rem_norm > best_d_rem_norm + OVERSHOOT_DREM_WORSEN_NORM`
- previous best `best_d_rem_norm < 0.10`

Response:

- enter recovery mode for `10` steps
- reduce translation cap
- allow small retract
- if best `d_rem_norm` is not recovered within window, abort

### 6. Positive-return truncations that look successful

Observed signature:

- short truncated episodes around `150-300` steps
- still moving forward
- total reward positive despite failure

Fix:

- apply `FAILURE_TRUNCATION_PENALTY = -5.0` whenever:
  - heuristic abort happens, or
  - vessel-end truncation happens without target reach

This should be applied in env logic, not only in selection filtering.

Rationale:

- makes obviously bad heuristic episodes clearly negative
- improves replay label quality

### 7. Success reward too weak

Observed signature:

- some successful episodes still have negative total reward

Fix:

- increase target reached reward from `+1.0` to `+3.0`

Rationale:

- makes real success clearly separable from short bad truncations
- better aligns replay quality with task objective

## Controller Changes

File:

- [heuristic_controller.py](/D:/rl_improv_3/training%20_scripts/util/heuristic_controller.py)

Current problem areas:

- unconditional minimum forward push
- no explicit retract mode
- full rotation authority even near target

### Required controller changes

#### A. Replace unconditional minimum forward push

Current code:

- `gw_trans = max(gw_trans, 5.0)`

Replace with distance-aware cap/floor:

- if `d_rem > 50 mm`: `gw_trans = clip(0.1 * d_rem, 5.0, 20.0)`
- if `20 mm < d_rem <= 50 mm`: `gw_trans = clip(0.1 * d_rem, 2.0, 8.0)`
- if `d_rem <= 20 mm`: `gw_trans = clip(0.1 * d_rem, 0.0, 4.0)`

#### B. Add bounded recovery mode

Internal controller state to add:

- `self._best_d_rem`
- `self._recovery_steps_left`
- `self._last_on_correct_branch`

Recovery entry:

- off-branch pseudo-close
- or correct-branch overshoot condition above

Recovery action policy:

- `gw_trans = -3.0 mm/s` for first `3` recovery steps
- then `gw_trans = clip(0.1 * d_rem, 0.0, 2.0)`
- `cath_trans = 0.8 * gw_trans`
- `gw_rot` clipped to `[-0.6, 0.6]`
- `cath_rot = 0`
- disable noise in recovery mode

#### C. Reduce rotation cap near target

- default cap: current env/action-space cap
- if `d_rem <= 50 mm`: clip `gw_rot` to `[-0.8, 0.8]`
- if in recovery mode: clip `gw_rot` to `[-0.6, 0.6]`

## Env Changes

File:

- [env5.py](/D:/rl_improv_3/training%20_scripts/util/env5.py)

### Required state to add on env object

Add heuristic-only episode state in `BenchEnv5.__init__` and reset in `reset()`:

- `self._heuristic_mode = False`
- `self._heuristic_abort_reason = None`
- `self._best_d_rem_norm = inf`
- `self._off_branch_recovery_count = 0`
- `self._both_max_stall_count = 0`
- `self._gw_partial_stall_count = 0`
- `self._sat_rot_count = 0`
- `self._gw_partial_drem_at_entry = None`
- `self._gw_partial_cath_inserted_at_entry = None`

### Activate heuristic mode

In `reset(seed=None, options=None)`:

- set `self._heuristic_mode = bool(options and options.get("heuristic_mode", False))`
- reset all counters/state above

### Read guidance values every step

In `step()` after `super().step(action)`:

- extract `guidance = obs["guidance"]`
- use:
  - `d_rem_norm = guidance[0]`
  - `on_correct_branch = bool(round(guidance[7]))`

### Store truncation components on env

In `__init__`, keep references:

- `self._target_terminal = terminal`
- `self._max_steps_trunc = max_steps`
- `self._vessel_end_trunc = vessel_end`
- `self._sim_error_trunc = sim_error`

This allows clean post-step reward shaping.

### Add heuristic abort logic in `step()`

Only if `self._heuristic_mode`:

1. Update `best_d_rem_norm`
2. Read current inserted lengths and max lengths from intervention
3. Compute per-step insertion deltas directly from current inserted minus previous inserted
4. Evaluate detectors in this order:

- `both_max_stall`
- `gw_partial_stall`
- `off_branch_pseudo_close`
- `sat_rot_bad_recovery`

If one triggers:

- set `truncated = True`
- set `terminated = False`
- set `self._heuristic_abort_reason = <reason>`
- add `FAILURE_TRUNCATION_PENALTY` to reward
- inject into `info`:
  - `heuristic_abort_reason`
  - `heuristic_abort = True`

### Reward shaping in `step()`

Apply after `super().step()` so final reward is what gets stored:

- if `terminated`: add `+2.0` extra bonus so total target bonus becomes `+3.0`
- if `truncated and not terminated` and either:
  - `self._vessel_end_trunc.truncated`
  - `self._heuristic_abort_reason is not None`
  then add `-5.0`

### Logging additions

Extend the INFO step log line to include when in heuristic mode:

- `heur_mode`
- `d_rem_norm`
- `on_branch`
- `abort_reason`
- `mask_reasons`

This makes future debugging much faster.

## Intervention Changes

File:

- [monoplanestatic.py](/D:/rl_improv_3/eve/eve/intervention/monoplanestatic.py)

The target-branch forwarding is already present and should stay as-is.

Recommended small addition:

- add a lightweight per-step accessor or helper for:
  - inserted lengths
  - max lengths
  - mask reasons

Not strictly required, but useful if env5 wants to read these cleanly.

## Seeding Schedule Changes

File:

- [DualDeviceNav_train.py](/D:/rl_improv_3/training%20_scripts/DualDeviceNav_train.py)

### Update schedule entries for heuristic seeding

Current schedule entries are:

- `(seed, {"target_branch": branch})`

Change them for heuristic seeding to:

- `(seed, {"target_branch": branch, "heuristic_mode": True})`

Do not add `heuristic_mode` to eval.

Heatup can optionally use it later, but not in this patch.

## Worker Stack Changes

Files:

- [single.py](/D:/rl_improv_3/eve_rl/eve_rl/agent/single.py)
- [singelagentprocess.py](/D:/rl_improv_3/eve_rl/eve_rl/agent/singelagentprocess.py)
- [synchron.py](/D:/rl_improv_3/eve_rl/eve_rl/agent/synchron.py)

The scheduling and `push_to_buffer=False` support are already present.

No new API changes are required for this patch beyond making sure:

- `episode_schedule` carries `heuristic_mode=True`
- options pass through unchanged

## Acceptance Criteria

After implementation, rerun heuristic seeding and verify:

### Mechanical criteria

- `both-max-stall` episodes should almost disappear
- mean heuristic episode duration should drop materially
- average late-step time inflation should reduce because long stuck tails are gone

### Reward criteria

- short truncated failures should no longer have positive total reward
- successful episodes should usually have positive total reward

### Data-quality criteria

- fewer max-step truncations
- fewer episodes ending after hundreds of zero-progress steps
- no replay episodes with obvious “forward command while both delta_ins = 0 for hundreds of steps”

### Logging criteria

- logs clearly show `heuristic_abort_reason`
- logs show `d_rem_norm` and `on_branch` at INFO steps in heuristic mode

## Recommended Implementation Order

1. `env5.py`
   - heuristic-mode state
   - detector counters
   - reward shaping
   - info/log fields
2. `heuristic_controller.py`
   - remove unconditional minimum forward push
   - add device-length-aware forward cap
   - add distance-aware forward schedule
   - bounded recovery mode
   - reduced near-target rotation caps
3. `DualDeviceNav_train.py`
   - add `heuristic_mode=True` into heuristic episode schedule
4. small cleanup in `monoplanestatic.py` only if env access is awkward

## Minimum Patch Set

If you want the smallest first patch with the biggest payoff, do these three first:

1. `both_max_stall` abort after `8` steps with `-5`
2. vessel-end and heuristic-abort penalty `-5`
3. target reached reward `+3`

That should already improve heuristic data quality and cut wall-clock cost sharply.

## Spec Corrections From Review

### Controller floor is a contributor, not the full stall root cause

The current `gw_trans = max(gw_trans, 5.0)` should be removed in Phase 1, but it is not enough by itself to prevent the observed `899 mm` stalls.

Reason:

- if the tip is on a wrong branch and `d_rem` remains large, then `0.1 * d_rem` can still command strong forward motion even without the `5.0` floor
- the max-length stall requires either an env-side abort or a controller cap based on remaining device length

Use this simpler forward baseline instead of the earlier over-clipped piecewise schedule:

- `gw_trans = min(self.max_translation, max(0.0, 0.1 * d_rem))`

Then apply a device-length-aware cap:

- `gw_remaining = gw_max - gw_inserted`
- `cath_remaining = cath_max - cath_inserted`
- `dt = 1.0 / fluoroscopy.image_frequency`
- `gw_trans = min(gw_trans, max(0.0, (gw_remaining - 1.0) / dt))`
- `cath_trans = min(cath_trans, max(0.0, (cath_remaining - 1.0) / dt))`

This prevents the heuristic from repeatedly asking for impossible forward insertion near the device-length limit.

### Penalize targeted heuristic failure modes, not every truncation

Do not blanket-penalize every heuristic-mode truncation.

Reason:

- max-step failures already accrue substantial negative return from the long episode tail
- early heuristic abort removes many future negative step penalties, so it needs an explicit penalty to preserve the failure signal
- vessel-end failures can be short and positive, so they need an explicit penalty

Use this rule instead:

- if `self._heuristic_mode and self._heuristic_abort_reason is not None`: add `FAILURE_TRUNCATION_PENALTY`
- else if `self._heuristic_mode and truncated and not terminated and self._vessel_end_trunc.truncated`: add `FAILURE_TRUNCATION_PENALTY`

Do not add the penalty to plain max-step truncations by default.

Evaluation and normal RL exploration should not use this heuristic-only shaping unless intentionally enabled.

### Keep detector insertion deltas separate from logging deltas

`BenchEnv5` currently updates `_prev_inserted` for logging, and that logging value is not reliable for per-step detectors because INFO logs are only every 50 steps or terminal.

Add a separate detector state:

- `self._prev_inserted_for_detector`

Update it every `step()` call after reading current inserted lengths.

Do not reuse the logging `_prev_inserted`.

### Use the existing insertion accessors

Use these APIs inside `BenchEnv5.step()`:

- `self.intervention.device_lengths_inserted`
- `self.intervention.device_lengths_maximum`

These come from `MonoPlaneStatic` and are cleaner than reaching into `simulation.inserted_lengths` directly.

### Recovery mode must bypass `never retract`

If recovery mode commands `gw_trans = -3.0`, then this line must not run unconditionally:

- `gw_trans = max(gw_trans, 0.0)`

Instead:

- clamp to nonnegative only in normal mode
- allow bounded negative translation in recovery mode

### Regenerate heuristic caches

Any heuristic cache generated before these fixes contains old failure-mode behavior.

After implementing this patch:

- delete or archive old heuristic caches
- regenerate `heuristic_cache.npz`
- rerun the validator and log parser on the new cache/logs
