# cursor_code_edits vs Original Codebase: Full Diff Analysis

> Generated 2026-03-27. Compares the 11 files in `cursor_code_edits/` against their
> originals in the repo, covering bug fixes, new features, potential concerns, and
> the interaction with the reset-order fix already applied in this branch.

---

## Table of Contents

1. [pathcontext.py](#1-pathcontextpy)
2. [localguidance.py](#2-localguidancepy)
3. [arclengthprogress.py](#3-arclengthprogresspy)
4. [env5.py](#4-env5py)
5. [DualDeviceNav_train_vs.py](#5-dualdevicenav_train_vspy)
6. [heuristic_policy.py](#6-heuristic_policypy-new-file)
7. [vanillashared.py](#7-vanillasharedpy)
8. [singelagentprocess.py](#8-singelagentprocesspy)
9. [synchron.py](#9-synchronpy)
10. [runner.py](#10-runnerpy)
11. [single.py](#11-singlepy)
12. [Summary of All Changes](#summary-of-all-changes)
13. [Changes Already Applied in This Branch](#changes-already-applied-in-this-branch)
14. [Changes Still Missing From This Branch](#changes-still-missing-from-this-branch)

---

## 1. pathcontext.py

**Files:**
- Original: `eve/eve/util/pathcontext.py`
- Cursor: `cursor_code_edits/pathcontext.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Added `_eve_skip_config = True` class attribute | 40 | Bug fix |
| 2 | Added `__reduce__` method + module-level `_return_none()` helper | 26-28, 55-58 | Bug fix |
| 3 | Removed `np.ndarray \| None` type annotations (Python 3.10+ syntax) | 52-53 | Compatibility |

### Why

1. **`_eve_skip_config`**: Without this, `eve.Env.save_config()` attempts to serialize
   `PathProjectionCache` via ConfigHandler. The cache holds live references to
   `pathfinder` and `intervention` and is not a config object. ConfigHandler would
   either fail or produce an invalid config YAML.

2. **`__reduce__`**: When `Synchron._create_worker_agent()` calls `deepcopy(env_train)`,
   every attribute is pickled. `PathProjectionCache` holds references to `pathfinder`
   and `intervention` which are already serialized as part of the env. Returning `None`
   via `__reduce__` lets the cache dissolve cleanly during pickling. The corresponding
   `__setstate__` in env5.py (change #4) recreates it in the worker process.

   Uses a module-level `_return_none()` function instead of a lambda because lambdas
   cannot be pickled in Python 3.8.

3. **Type annotation syntax**: `np.ndarray | None` requires Python 3.10+. Replacing
   with `= None` (no annotation) ensures compatibility with Python 3.8 environments.

### Potential Concerns

- None. All three changes are necessary for multiprocessing correctness.

---

## 2. localguidance.py

**Files:**
- Original: `eve/eve/observation/localguidance.py`
- Cursor: `cursor_code_edits/localguidance.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Added `self.path_context = None` (public attr for ConfigHandler) | 64 | Bug fix |

### Why

ConfigHandler introspects `__init__` parameters to build the config dict for
`save_config()`. It expects `self.path_context` to exist (matching the `path_context`
kwarg). Without it, `save_config()` raises `AttributeError`. The actual runtime cache
lives in `self._path_context` (private), so setting the public one to `None` is safe
and allows serialization.

### Potential Concerns

- None. The pattern of `self.param = None` (serializable) + `self._param = actual`
  (runtime) is clean and idiomatic for this codebase.

---

## 3. arclengthprogress.py

**Files:**
- Original: `eve/eve/reward/arclengthprogress.py`
- Cursor: `cursor_code_edits/arclengthprogress.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Added `self.path_context = None` (public attr for ConfigHandler) | 53 | Bug fix |

### Why

Identical rationale to localguidance.py (change #2 above). Without this,
`save_config()` crashes when `path_context` is passed to the constructor.

### Potential Concerns

- None.

---

## 4. env5.py

**Files:**
- Original: `training _scripts/util/env5.py`
- Cursor: `cursor_code_edits/env5.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Expanded `__setstate__` to rebuild `PathProjectionCache` and re-wire references | 195-215 | Bug fix |

### Why

When `Synchron._create_worker_agent()` calls `deepcopy(env_train)`, the
`PathProjectionCache.__reduce__` returns `None` (see pathcontext.py change #2).
After unpickling in the worker process, `self._path_context` is `None`. Without
the expanded `__setstate__`:

- `LocalGuidance` and `ArcLengthProgress` would have `_path_context = None`
- They would fall back to independent computation (doubling projection work per step)
- The reset-order fix (calling `_path_context.reset()` in component resets) would
  be bypassed since there is no cache to reset

The new `__setstate__` detects `_path_context is None` and:
1. Creates a fresh `PathProjectionCache(self.pathfinder, self.intervention)`
2. Walks `self.observation.observations` (ObsDict) to find components with
   `_path_context` and re-wires them
3. Walks `self.reward.rewards` (Combination) to do the same

### Potential Concerns

- **Fragile attribute walk**: The code assumes `self.observation.observations` is a
  dict (ObsDict) and `self.reward.rewards` is a list (Combination). If these
  structures change, the walk silently fails to re-wire. However, this is practical
  for the current architecture and uses `hasattr` guards.
- **No deep unwrapping**: If observations or rewards are nested wrappers, the walk
  only checks the top level. Currently this is sufficient since LocalGuidance and
  ArcLengthProgress are direct children.

---

## 5. DualDeviceNav_train_vs.py

**Files:**
- Original: `training _scripts/DualDeviceNav_train.py`
- Cursor: `cursor_code_edits/DualDeviceNav_train_vs.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | `HEATUP_STEPS = 2e4` (was `1e4`) | 23 | Config |
| 2 | Sequential heuristic seeding replaced with parallel | 307-360 | Feature |
| 3 | New CLI args: `--heuristic_cache_file`, `--save_heuristic_cache`, `--heatup_cache_file`, `--save_heatup_cache` | 137-158 | Feature |
| 4 | Config save from unwrapped env (`env_train_unwrapped`) | 287-290 | Bug fix |
| 5 | `training_run()` passes `heatup_cache_save_path` | 387-395 | Feature |
| 6 | Heatup cache loading (skip heatup phase) | 362-385 | Feature |

### Why

1. **HEATUP_STEPS**: The original `1e4` was labeled "Reduced from 5e5 for quick
   testing". `2e4` is a better production default while still being faster than the
   original `5e5`.

2. **Parallel heuristic seeding**: The original runs heuristic episodes sequentially
   on the main process, creating a dedicated `seed_env` and `CenterlineFollowerHeuristic`.
   The cursor version uses `HeuristicActionFunctionFactory` + `agent.heuristic_seed()`
   to distribute across N workers. This is dramatically faster for large seeding
   counts (e.g., 500 episodes).

3. **Cache CLI args**: Avoids re-running expensive heuristic/heatup phases on resume
   or repeated experiments. Episodes are saved/loaded via `eve_rl.util.experience_cache`.

4. **Config save bug fix**: When `--curriculum` is used, `env_train` is wrapped in
   `ActionCurriculumWrapper`, which has no `save_config()`. The original would crash.
   The cursor version saves from `env_train_unwrapped`.

5. **Heatup cache save**: Passes `heatup_cache_save_path` to `runner.training_run()`
   so heatup episodes are persisted for future runs.

6. **Heatup cache load**: If `--heatup_cache_file` is provided and exists, loads
   episodes directly into the replay buffer and sets `heatup_steps_effective = 0`,
   skipping the heatup phase entirely.

### Potential Concerns

- **Dependency on `eve_rl.util.experience_cache`**: The `save_episodes_npz` and
  `load_episodes_npz` functions must exist. If this module is missing, the cache
  features crash. (This is a new utility module not present in the diff.)
- **`EpisodeReplay` usage**: The cache loading creates `EpisodeReplay` objects and
  pushes them. This type must be compatible with what the replay buffer expects.

---

## 6. heuristic_policy.py (NEW FILE)

**Files:**
- Original: Does not exist (new file)
- Cursor: `cursor_code_edits/heuristic_policy.py`

### What It Does

Wraps `CenterlineFollowerHeuristic` for use with the parallel worker infrastructure:

- **`HeuristicActionFunction`**: Callable wrapper compatible with `_play_episode()`.
  - Uses lazy reset: `reset()` sets a flag; actual `heuristic.reset()` fires on
    first `__call__` after `_play_episode` has called `env.reset()`
  - Unwraps gymnasium wrappers via `getattr(env, 'unwrapped', env)` to access
    `pathfinder`/`intervention` on the base `BenchEnv4`/`BenchEnv5`
  - Handles action normalization to `[-1, 1]` range
  - Supports optional Gaussian noise injection

- **`HeuristicActionFunctionFactory`**: Pickleable factory sent to workers via
  multiprocessing queue. Workers call `.create(env)` to build a
  `HeuristicActionFunction` with their own env instance.

- **`create_heuristic_action_function()`**: Convenience factory function for
  non-parallel use.

### Why

The original `DualDeviceNav_train.py` created a `CenterlineFollowerHeuristic`
directly and ran episodes sequentially. To distribute across workers, the heuristic
must be:
1. Created per-worker (can't share pathfinder/intervention across processes)
2. Pickleable (factory pattern — the factory is pickleable, the action function is not)
3. Compatible with `_play_episode()` signature (takes `flat_obs`, returns action)

### Potential Concerns

- **Lazy reset timing**: If `__call__` is invoked without a prior `reset()`, it uses
  stale path data. This is safe in the current flow (`_play_episode` always calls
  `env.reset()` first, and `heuristic_action.reset()` is called before
  `_play_episode` in `Single.heuristic_seed()`), but would break if the calling
  convention changes.
- **`getattr(env, 'unwrapped', env)` unwrapping**: Relies on gymnasium's standard
  `.unwrapped` property. If a custom wrapper doesn't implement it, the unwrap fails
  silently and the heuristic gets the wrapper instead of the base env.

---

## 7. vanillashared.py

**Files:**
- Original: `eve_rl/eve_rl/replaybuffer/vanillashared.py`
- Cursor: `cursor_code_edits/vanillashared.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Added `episode_arrival_queue` parameter to `VanillaSharedBase` | 40, 49 | Feature |
| 2 | Added `_shared_update_step = mp.Value('q', 0)` | 52 | Feature |
| 3 | Added `__getstate__`/`__setstate__` for pickle safety | 54-64 | Bug fix |
| 4 | `push()` sends `(replay_data, explore_step)` tuples | 70-77 | Feature |
| 5 | Added `set_step_counter()` method | 98-111 | Feature |
| 6 | Added `drain_episode_arrivals()` method | 112-123 | Feature |
| 7 | Added `save_buffer_to_file()` / `load_buffer_from_file()` | 125-135 | Feature |
| 8 | Subprocess receives `shared_update_step` as argument (not pickled) | 160-163, 166-171 | Bug fix |
| 9 | Loop logs `update_step`, handles `(replay_data, explore_step)` tuples, records arrival metadata | 181-247 | Feature |
| 10 | `copy()` passes `episode_arrival_queue` | 259-268 | Bug fix |
| 11 | Reduced push logging verbosity | — | Cleanup |

### Why

**Episode timestamping problem**: In the original architecture, workers read
`step_counter.update` when an episode completes. But by the time
`Synchron.explore_and_update()` collects the results, many more gradient steps have
happened. The `update_step_at_completion` on the episode is stale.

**Solution**: The replay buffer subprocess — which sits between workers and the
trainer — records the *actual* `update_step` when each episode arrives. This is done
via:
1. `_shared_update_step`: A `mp.Value` shared between the parent and the replay
   subprocess (inherited at spawn, not pickled)
2. `set_step_counter()`: Called by `Synchron.update()` after each update cycle to
   sync the current step
3. `episode_arrival_queue`: The subprocess puts `(explore_step, update_step)` when
   processing a push
4. `drain_episode_arrivals()`: `Synchron.explore_and_update()` reads the arrival map
   and re-stamps episodes

**Buffer persistence**: `save_buffer_to_file()` and `load_buffer_from_file()` send
task commands to the subprocess, which serializes/deserializes the internal buffer
using `experience_cache`. This enables training resume.

### Potential Concerns

- **Push format change**: `push()` now sends `(replay_data, explore_step)` instead of
  just `replay_data`. The subprocess loop handles both formats via
  `isinstance(item, tuple)` check, maintaining backward compatibility.
- **`mp.Queue` overhead**: `episode_arrival_queue` adds one `put`/`get` per episode.
  This is lightweight metadata (two integers) and negligible compared to episode
  replay data.
- **`__getstate__` excludes `_shared_update_step`**: Workers receive `None` for this
  field. This is correct — workers don't need to update the step counter, and
  `mp.Value` cannot be pickled on Windows `spawn`.

---

## 8. singelagentprocess.py

**Files:**
- Original: `eve_rl/eve_rl/agent/singelagentprocess.py`
- Cursor: `cursor_code_edits/singelagentprocess.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Added `heuristic_seed` task handler in `run()` function | 176-183 | Feature |
| 2 | Added `heuristic_seed()` method on `SingleAgentProcess` class | 396-415 | Feature |

### Why

Completes the parallel heuristic seeding dispatch chain:

```
DualDeviceNav_train_vs.py
  -> Synchron.heuristic_seed()       [synchron.py]
    -> SingleAgentProcess.heuristic_seed()  [singelagentprocess.py]  <-- THIS
      -> run() dispatches "heuristic_seed" task
        -> Single.heuristic_seed()   [single.py]
```

The `run()` function adds a new `elif task_name == "heuristic_seed"` branch that
calls `agent.heuristic_seed()` with the factory. The `SingleAgentProcess` method
sends the task tuple to the worker's queue.

### Potential Concerns

- None. Mirrors the existing `heatup` pattern exactly.

---

## 9. synchron.py

**Files:**
- Original: `eve_rl/eve_rl/agent/synchron.py`
- Cursor: `cursor_code_edits/synchron.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | `__init__` calls `self.replay_buffer.set_step_counter()` | 302 | Feature |
| 2 | Added `heuristic_seed()` method | 354-414 | Feature |
| 3 | `update()` calls `self.replay_buffer.set_step_counter()` | 457 | Feature |
| 4 | `explore_and_update()` calls `drain_episode_arrivals()` and re-stamps episodes | 562-568 | Feature |

### Why

1. **Initial sync**: Ensures the replay buffer subprocess has `update_step = 0` from
   the start, before any episodes arrive.

2. **`heuristic_seed()`**: Dispatches heuristic seeding to all workers in parallel.
   Key detail: temporarily increases `timeout_worker_after_reaching_limit` to 1200s
   because heuristic episodes can take 500-800s each (1000 steps x 0.5-0.8s/step).
   The default 90s timeout would kill workers mid-episode.

3. **Step counter sync in `update()`**: After each update cycle, syncs the current
   `step_counter.update` to the replay buffer's `_shared_update_step`. This ensures
   subsequent episode arrivals are stamped with the correct value.

4. **Episode re-stamping**: After `explore_and_update()`, reads the arrival map from
   the replay buffer subprocess and updates `episode.update_step_at_completion` on
   each episode with the value that was current when the episode *actually arrived*
   at the replay buffer (not the stale value from the worker).

### Potential Concerns

- **Timeout restoration**: Uses `saved_timeout` / restore pattern. If
  `heuristic_seed()` raises between the save and restore, the timeout stays at 1200s.
  A `try/finally` would be safer, though the impact is minor (just a longer timeout
  for subsequent operations).

---

## 10. runner.py

**Files:**
- Original: `eve_rl/eve_rl/runner/runner.py`
- Cursor: `cursor_code_edits/runner.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Added `_eval_count` counter | 54 | Feature |
| 2 | Added `_replay_save_interval = 4` | 55 | Feature |
| 3 | Added `restore_runner_state()` method | 103-147 | Feature |
| 4 | Added `load_replay_buffer()` method | 149-159 | Feature |
| 5 | `eval()` increments `_eval_count`, saves `runner_state` in checkpoint | 230-238 | Feature |
| 6 | `eval()` periodically saves replay buffer (every N evals) | 255-262 | Feature |
| 7 | `training_run()` accepts and uses `heatup_cache_save_path` | 318-335 | Feature |

### Why

**Training resume support**: Long RL training runs (20M steps) can take days. If the
process crashes, all progress is lost. These changes enable full resume:

1. **Runner state in checkpoints**: `eval()` saves `best_eval`,
   `_episode_summary_counter`, `_next_snapshot_step`, and `_eval_count` inside
   the checkpoint's `additional_info["runner_state"]`.

2. **`restore_runner_state()`**: Reads the saved state from a checkpoint and restores
   all Runner-level counters. Also restores probe states from a saved `.npz` file.

3. **`load_replay_buffer()`**: Auto-loads the replay buffer from
   `checkpoint_folder/replay_buffer.npz` if it exists.

4. **Periodic replay buffer saves**: Every 4th eval, saves the full replay buffer to
   `checkpoint_folder/replay_buffer.npz` via the subprocess task command.

5. **Heatup cache**: Saves heatup episodes to disk so future runs can skip heatup.

### Potential Concerns

- **`save_buffer_to_file` is blocking**: Sends a task to the replay buffer subprocess
  and waits for the result via `_result_queue.get()`. During eval, this adds latency.
  Acceptable since eval already pauses training.
- **Replay buffer size on disk**: With 10k capacity and large observation spaces, the
  `.npz` file could be significant. No compression is applied.
- **Resume gap**: Between the last periodic save and a crash, up to 3 evals worth of
  replay buffer changes are lost. This is a tradeoff for avoiding save overhead on
  every eval.

---

## 11. single.py

**Files:**
- Original: `eve_rl/eve_rl/agent/single.py`
- Cursor: `cursor_code_edits/single.py`

### Changes

| # | Change | Lines | Category |
|---|--------|-------|----------|
| 1 | Added `heuristic_seed()` method | 263-335 | Feature |

All other code (`SingleEvalOnly`, `Single.__init__`, `heatup`, `explore`, `update`,
`_log_batch_samples`, `explore_and_update`, `close`, `from_checkpoint`) is
**byte-for-byte identical**.

### Why

This is the worker-side implementation of parallel heuristic seeding. It mirrors
`heatup()` with two differences:
1. Uses a heuristic action function (from `heuristic_factory.create(env_train)`)
   instead of `random_action`
2. Calls `heuristic_action.reset()` at the start of each episode

Uses `heatup` step/episode counters since heuristic seeding is a pre-training phase.

### Potential Concerns

- **`heuristic_factory` is required**: Raises `ValueError` if `None`. This is correct
  — there's no sensible default.
- **Counter sharing with heatup**: If both heatup and heuristic seeding are used in
  the same run, their step counts are merged. This could be confusing in logs but is
  functionally correct (both fill the replay buffer pre-training).

---

## Summary of All Changes

### Bug Fixes (5)

| File | Fix | Severity |
|------|-----|----------|
| pathcontext.py | `_eve_skip_config` — prevents ConfigHandler crash on `save_config()` | High |
| pathcontext.py | `__reduce__` — prevents `deepcopy` crash in worker creation | High |
| localguidance.py | `self.path_context = None` — prevents ConfigHandler `AttributeError` | High |
| arclengthprogress.py | `self.path_context = None` — same as above | High |
| env5.py | `__setstate__` rebuilds cache in workers after unpickling | High |
| DualDeviceNav_train_vs.py | Config save from unwrapped env (curriculum wrapper crash) | Medium |

### New Features (4 feature areas)

| Feature Area | Files Changed |
|-------------|---------------|
| Parallel heuristic seeding | heuristic_policy.py (new), single.py, singelagentprocess.py, synchron.py, DualDeviceNav_train_vs.py |
| Accurate episode timestamping | vanillashared.py, synchron.py, single.py (explore) |
| Replay buffer persistence | vanillashared.py, runner.py |
| Training resume support | runner.py, DualDeviceNav_train_vs.py |

### Compatibility (1)

| File | Change |
|------|--------|
| pathcontext.py | Python 3.8 type annotation compatibility |

### Config (1)

| File | Change |
|------|--------|
| DualDeviceNav_train_vs.py | `HEATUP_STEPS` 1e4 -> 2e4 |

---

## Changes Already Applied in This Branch

The following fixes were applied earlier in this conversation and are **consistent**
with the cursor_code_edits versions:

1. **Reset order bug** (`localguidance.py`, `arclengthprogress.py`, `env5.py`):
   Moved `_path_context.reset()` into `LocalGuidance.reset()` and
   `ArcLengthProgress.reset()`, removed the too-late call from `env5.py`.
   The cursor_code_edits versions have this same fix.

2. **Dead `_valid` flag** (`pathcontext.py`): Removed `self._valid` and its usage.
   The cursor_code_edits version also doesn't have `_valid`.

3. **Missing export** (`eve/eve/util/__init__.py`): Added `PathProjectionCache` export.
   The cursor_code_edits don't include `__init__.py` but this change is compatible.

---

## Changes Still Missing From This Branch

The following cursor_code_edits changes have **not yet been applied** to the repo:

### Critical (needed for multiprocessing correctness)

1. **pathcontext.py**: `_eve_skip_config`, `__reduce__`/`_return_none()`, Python 3.8
   type annotation fix
2. **localguidance.py**: `self.path_context = None` (ConfigHandler serialization)
3. **arclengthprogress.py**: `self.path_context = None` (ConfigHandler serialization)
4. **env5.py**: Expanded `__setstate__` to rebuild cache in workers

### Feature additions

5. **heuristic_policy.py**: New file — parallel heuristic seeding wrapper
6. **single.py**: `heuristic_seed()` method
7. **singelagentprocess.py**: `heuristic_seed` task dispatch
8. **synchron.py**: `heuristic_seed()`, step counter sync, episode re-stamping
9. **vanillashared.py**: Episode arrival tracking, shared update step, buffer
   persistence, pickle safety
10. **runner.py**: Resume support, replay buffer persistence, heatup cache
11. **DualDeviceNav_train_vs.py**: Parallel seeding, cache args, unwrapped config
    save, heatup steps update
