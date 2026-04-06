# Heuristic Seeding Target Diversity Fix

## Problem Summary

During `DualDeviceNav_train.py` heuristic seeding with parallel workers, all workers appear to experience very similar episode-end rewards at the same episode index. Example observed behavior:

- one wave of workers ends around `-4.x`
- the next wave ends around `-11/-12`
- another wave ends around `+2.x`

This suggests that workers are not seeing sufficiently independent target sequences during heuristic seeding.

## Root Cause

The root cause is cloned target RNG state across workers during heuristic seeding.

### What the code currently does

1. `DualDeviceNav` uses `CenterlineRandom` as its target sampler.
2. `CenterlineRandom` owns an internal Python RNG:
   - `self._rng = random.Random()`
   - [centerlinerandom.py](D:/rl_improv_3/eve/eve/intervention/target/centerlinerandom.py#L33)
3. On reset, `CenterlineRandom` only reseeds this RNG if an explicit seed is provided:
   - `if seed is not None: self._rng = random.Random(seed)`
   - [centerlinerandom.py](D:/rl_improv_3/eve/eve/intervention/target/centerlinerandom.py#L39)
4. Target choice is sampled from that RNG:
   - `target_vessel_cs = self._rng.choice(self._potential_targets)`
   - [centerlinerandom.py](D:/rl_improv_3/eve/eve/intervention/target/centerlinerandom.py#L45)
5. Worker envs are created by deep-copying the parent env:
   - `deepcopy(self.env_train)`
   - [synchron.py](D:/rl_improv_3/eve_rl/eve_rl/agent/synchron.py#L562)
6. During heuristic seeding, episodes reset with `seed=None`:
   - `obs, _ = env.reset(seed=seed, options=options)`
   - [single.py](D:/rl_improv_3/eve_rl/eve_rl/agent/single.py#L104)
7. `MonoPlaneStatic.reset()` only generates a `target_seed` when an explicit reset seed exists:
   - `if seed is not None: ...`
   - `target_seed = None if seed is None else self._np_random.integers(...)`
   - `self.target.reset(episode_number, target_seed)`
   - [monoplanestatic.py](D:/rl_improv_3/eve/eve/intervention/monoplanestatic.py#L133)
   - [monoplanestatic.py](D:/rl_improv_3/eve/eve/intervention/monoplanestatic.py#L148)
   - [monoplanestatic.py](D:/rl_improv_3/eve/eve/intervention/monoplanestatic.py#L149)

### Consequence

Because workers receive deep-copied envs, they also receive cloned `CenterlineRandom._rng` state. Since heuristic seeding does not pass explicit episode seeds, each worker begins from the same target RNG state and advances it in the same way.

If workers also remain roughly in sync by episode count, they will sample the same target sequence or near-identical target sequence.

This is why different workers can show nearly identical reward bands at the same episode index even before any learning has occurred.

## Important Clarification

This is not caused by `EVAL_SEEDS`.

- `EVAL_SEEDS` in [DualDeviceNav_train.py](D:/rl_improv_3/training%20_scripts/DualDeviceNav_train.py#L19) are only used for evaluation.
- They are passed into `runner.training_run(..., eval_seeds=EVAL_SEEDS)` at [DualDeviceNav_train.py](D:/rl_improv_3/training%20_scripts/DualDeviceNav_train.py#L342).
- They do not affect heuristic seeding.

## Target Space Size

For `DualDeviceNav`, `CenterlineRandom` is restricted to 4 named branches:

- `Centerline curve - LCCA.mrk`
- `Centerline curve - LVA.mrk`
- `Centerline curve - RCCA.mrk`
- `Centerline curve - RVA.mrk`

These come from [dualdevicenav.py](D:/rl_improv_3/eve_bench/eve_bench/dualdevicenav.py#L89).

However, the sampler does not choose among only 4 targets. It chooses among centerline points on those branches.

Observed candidate counts from the current mesh data:

- raw points on allowed branches: `924`
- candidate target points after excluded-branch filtering: `898`

Per-branch raw centerline point counts:

- LCCA: `252`
- LVA: `208`
- RCCA: `235`
- RVA: `229`

So the target generator itself has plenty of diversity. The issue is that workers are sampling the same RNG stream.

## Evidence From Saved Heuristic Cache

From `saved/heuristic_cache.npz`:

- total heuristic episodes: `100`
- exact unique normalized target observations at episode start: `7`

This is far smaller than the expected diversity from an 898-point candidate set.

This strongly supports the conclusion that workers are walking through the same target sequence in parallel.

## Requirements For The Fix

The fix must satisfy all of the following:

1. Heuristic seeding workers must not share the same target RNG stream.
2. Different workers should receive different target sequences even when spawned from the same parent env.
3. The behavior should remain reproducible when desired.
4. Evaluation behavior using `EVAL_SEEDS` must remain unchanged.
5. The fix should not reduce the total target space or change branch eligibility.
6. The fix should work for all parallel worker counts, including `16`.
7. The fix should be explicit enough that future debugging can recover which worker saw which targets.

## Recommended Fix

The safest fix is to introduce explicit per-episode seeding for heuristic seeding and optionally heatup as well.

### Recommendation A: Add explicit seed lists for heuristic seeding

Add a deterministic list of episode seeds for heuristic seeding and pass those seeds into worker episodes.

Desired properties:

- each heuristic episode gets a unique seed
- seeds are reproducible across runs if a base seed is fixed
- workers no longer depend on copied RNG state

Implementation idea:

1. Add a CLI argument such as:
   - `--heuristic_seed_base`
2. Generate `N` episode seeds in the parent process for `N = args.heuristic_seeding`
3. Split that list across workers the same way evaluation seeds are split
4. Extend `heuristic_seed()` worker API to accept `seeds`
5. Inside worker heuristic seeding, use those seeds in `_play_episode(..., seed=next_seed)`

This makes target sampling reproducible and diverse across workers.



These are not enough by themselves:

- relying on `episode_nr` inside `CenterlineRandom`
- relying on Python default randomness after deep copy
- assuming process spawn alone guarantees target diversity
- using only evaluation seed machinery without connecting it to heuristic seeding

## Concrete Code Changes Needed

### 1. Extend heuristic seeding API to accept seeds

Affected areas:

- [single.py](D:/rl_improv_3/eve_rl/eve_rl/agent/single.py)
- [singelagentprocess.py](D:/rl_improv_3/eve_rl/eve_rl/agent/singelagentprocess.py)
- [synchron.py](D:/rl_improv_3/eve_rl/eve_rl/agent/synchron.py)

Needed behavior:

- parent generates a full heuristic seed list
- parent splits seeds across workers
- each worker consumes its own seeds for heuristic episodes
- `_play_episode(..., seed=seed)` is used for heuristic seeding too

### 2. Add CLI/config support in training script

Affected area:

- [DualDeviceNav_train.py](D:/rl_improv_3/training%20_scripts/DualDeviceNav_train.py)

Needed behavior:

- add a base seed or explicit heuristic seed list option
- generate one seed per heuristic episode
- pass seeds into `agent.heuristic_seed(...)`




## Bottom Line

The problem is not a lack of available targets. The task has about `898` candidate target points across the 4 allowed branches.

The real issue is that parallel heuristic-seeding workers are inheriting the same `CenterlineRandom` RNG state and then running in sync, which causes them to sample the same target sequence.

The fix is to make heuristic seeding use explicit per-episode seeds and, ideally, also reseed worker-local target RNGs defensively.
