# RL Improvement 3 — Changes Summary

## Overview

This session implemented four categories of changes:

1. **Observation normalization** — eliminated 200:1 scale disparity in LocalGuidance features
2. **Action space safety** — asymmetric bounds, reduced velocity limits, fixed heatup ranges
3. **Target diversity fix** — per-episode seeding with branch-balanced scheduling for heuristic seeding
4. **Minimum success rate for heuristic seeding** — batched collection with 30% success floor + 70% cap

All changes target env4/env5 and the DualDeviceNav training pipeline.

---

## 1. Normalize LocalGuidance Features

### Context

The `LocalGuidance` observation provides 8 features encoding the agent's relationship to the known correct path. These are concatenated with tracking observations (normalized to [-1, 1]) into a single flat vector fed to the MLP.

### Problem

Five of the 8 features had wildly different scales:

| Feature | Before range | Scale |
|---------|-------------|-------|
| `d_rem_norm` | [0, 1] | 1 |
| `cross_track_dist` | [0, 50] mm | 50 |
| `tangent_x_2d` | [-1, 1] | 2 |
| `tangent_z_2d` | [-1, 1] | 2 |
| `heading_error` | [-π, π] | ~6.28 |
| `curvature_ahead` | [0, ~10] | 10 |
| `dist_to_bifurcation` | [0, 200] mm | 200 |
| `on_correct_branch` | {0, 1} | 1 |

With a 200:1 scale disparity, features like `dist_to_bifurcation` dominated gradient updates over critical steering signals like `d_rem_norm` and `tangent_2d`.

### Fix

Divided each unnormalized feature by its maximum clip value in `LocalGuidance.step()`:

- `cross_track_dist / 50` → [0, 1]
- `heading_error / π` → [-1, 1]
- `curvature_ahead / 10` → [0, ~1]
- `dist_to_bifurcation / 200` → [0, 1]

Updated `space` property to match the new normalized bounds.

### Files Changed

- `eve/eve/observation/localguidance.py` — `step()` output normalization + `space` property

---

## 2. Action Space Safety — Asymmetric Bounds

### Context

DualDeviceNav uses two J-shaped devices (guidewire + catheter), each with a `velocity_limit` defining the env action space. The action space was symmetric: `[-velocity_limit, +velocity_limit]`.

### Problem (Velocity Limits)

The original `velocity_limit=(35, 3.14)` meant:

- **35 mm/step** translation at 7.5 Hz = **262 mm/s** — causes SOFA numerical solver failures
- **3.14 rad/step** rotation = **180°/step** = 1350°/s — physically absurd, stresses solver

Analysis of the saved heatup cache (`saved/heatup_cache.npz`, 51 episodes) confirmed:

- **44.2%** of all steps had `|gw_trans| > 15mm` (above the safe threshold from SOFA_TIMEOUT_FIX.md)
- **52.3%** of all steps had `|gw_rot| > 1.0 rad`
- **91.7%** of all steps had at least one action dimension exceeding safe thresholds

### Problem (Symmetric Bounds)

The action space was symmetric `[-35, +35]` mm for translation, but retraction beyond ~10mm/step is rarely useful. A symmetric space wastes half the policy's output range on a direction it almost never needs, making learning harder.

### Fix

**A) Added asymmetric action space support to `MonoPlaneStatic`:**

Added a `velocity_limit_low` parameter (defaults to `-velocity_limits` for backward compat). The `action_space` property and `step()` denormalization now use `[velocity_limits_low, velocity_limits]` instead of the symmetric `[-velocity_limits, velocity_limits]`.

**B) Updated DualDeviceNav device and action bounds:**

| Dimension | Before | After |
|-----------|--------|-------|
| gw_trans | [-35, +35] mm | **[-10, +30] mm** |
| gw_rot | [-3.14, +3.14] rad | **[-1.5, +1.5] rad** |
| cath_trans | [-35, +35] mm | **[-10, +30] mm** |
| cath_rot | [-3.14, +3.14] rad | **[-1.5, +1.5] rad** |

Translation is asymmetric — 3× more range forward than retraction. Rotation is symmetric (no anatomical reason to prefer a direction).

**C) Updated heatup sampling bounds:**

| | Before | After |
|--|--------|-------|
| heatup_low | `[[-10, -1], [-11, -1]]` | `[[-10, -1.5], [-10, -1.5]]` |
| heatup_high | `[[35, 3.14], [30, 3.14]]` | `[[30, 1.5], [30, 1.5]]` |

### Files Changed

- `eve/eve/intervention/monoplanestatic.py` — `velocity_limit_low` parameter, asymmetric `action_space` and `step()` denormalization
- `eve_bench/eve_bench/dualdevicenav.py` — `velocity_limit=(30, 1.5)` on both devices, `velocity_limit_low=[[-10, -1.5], [-10, -1.5]]` on both `DualDeviceNav` and `DualDeviceNavCustom`
- `training _scripts/DualDeviceNav_train.py` — updated heatup bounds

### Note: Heuristic Controller Not Changed

The heuristic controller (`heuristic_controller.py`) uses `max_translation=20` and rotation typically < 0.5 rad. Although 20mm exceeds the 15mm safe threshold, it only occurs when `d_rem > 200mm` (early in episode) and drops proportionally. Left unchanged per user decision.

---

## 3. Target Diversity Fix — Branch-Balanced Episode Scheduling

### Context

During parallel heuristic seeding and heatup, multiple workers collect episodes simultaneously. Each worker gets a deep-copied environment from the parent process.

### Problem

`CenterlineRandom` (the target sampler) uses an internal Python `random.Random()` RNG. When workers are spawned via `deepcopy(env)`, they all inherit the same RNG state. Since heuristic seeding calls `env.reset(seed=None)`, the RNG is never reseeded — all workers walk through the **same target sequence**.

Evidence from saved heatup cache:
- 51 episodes across workers → only **6 unique targets** from an 898-point candidate set
- Workers showed nearly identical reward bands at the same episode index

Full analysis documented in `HEURISTIC_SEEDING_TARGET_DIVERSITY_FIX.md`.

### Fix

Implemented a 6-layer pipeline: training script generates the schedule, framework dispatches and consumes it.

**A) `CenterlineRandom.reset()` — forced branch selection:**

Added `target_branch` parameter. When provided, samples only from that branch's valid centerline points (stored in `_branch_targets` dict, built at initialization). When not provided, behavior is unchanged (samples from all branches).

Added `target_branch_names` property to expose available branches.

**B) `MonoPlaneStatic.reset()` — options forwarding:**

Extracts `target_branch` from the `options` dict and forwards to `target.reset()` via `**kwargs`. Backward compatible — if `target_branch` is not in options, `target.reset()` gets called with just `(episode_number, target_seed)`.

**C) `single.py` — per-episode schedule consumption:**

Both `heatup()` and `heuristic_seed()` accept an `episode_schedule` parameter: a list of `(seed, options)` tuples. Each episode consumes the next entry, passing `seed` and `options` to `_play_episode()` which forwards them to `env.reset(seed=seed, options=options)`.

When `episode_schedule` is None (backward compat), episodes run without explicit seeds (original behavior).

**D) `singelagentprocess.py` — queue forwarding:**

Both `heatup` and `heuristic_seed` task messages now include `episode_schedule` as an additional field. The `run()` function dispatcher forwards it to the `Single` agent methods.

**E) `synchron.py` — round-robin splitting:**

Added `_split_schedule()` method that distributes the full schedule across N workers round-robin. Worker 0 gets entries 0, N, 2N, ...; Worker 1 gets entries 1, N+1, 2N+1, ...; etc.

Both `heatup()` and `heuristic_seed()` accept `episode_schedule`, split it, and pass per-worker slices to each worker.

**F) `DualDeviceNav_train.py` — schedule generation:**

Added `build_episode_schedule(n_episodes, branches, base_seed=42)` function:

1. Generates one unique seed per episode from a deterministic RNG
2. Assigns branches round-robin (episode 0 → branch 0, episode 1 → branch 1, ...) ensuring exact balance (e.g., 100 episodes → 25 per branch)
3. Shuffles the schedule deterministically so workers get diverse branches (not all the same branch)

The heuristic seeding section now builds this schedule and passes it to `agent.heuristic_seed()`. Prints the branch distribution for verification.

For heatup, branch balancing is not enabled (no schedule passed — left as original behavior per user request). For evaluation, existing `EVAL_SEEDS` behavior is unchanged.

### Data Flow

```
DualDeviceNav_train.py        builds: [(seed₁, {"target_branch": "LCCA"}), (seed₂, {"target_branch": "RVA"}), ...]
       ↓
synchron.py                   splits round-robin across N workers
       ↓
singelagentprocess.py         passes via task queue to worker process
       ↓
single.py                     consumes (seed, options) per episode → _play_episode(seed=, options=)
       ↓
eve.Env.reset(seed, options)  forwards to intervention.reset(episode_nr, seed, options)
       ↓
MonoPlaneStatic.reset()       extracts target_branch from options → target.reset(**target_kwargs)
       ↓
CenterlineRandom.reset()      reseeds RNG with seed, samples from target_branch's point pool
```

### Verification

Tested with 100 episodes across 16 workers:

- **Branch balance**: exactly 25/25/25/25 globally
- **Seed uniqueness**: 100/100 unique seeds
- **Reproducibility**: identical schedule from same `base_seed`
- **Worker diversity**: each worker's first episode gets a different seed and diverse branches across workers

### Files Changed

- `eve/eve/intervention/target/centerlinerandom.py` — `target_branch` param, `_branch_targets` dict, `target_branch_names` property
- `eve/eve/intervention/monoplanestatic.py` — `options` forwarding to `target.reset()`
- `eve_rl/eve_rl/agent/single.py` — `episode_schedule` param on `heatup()` and `heuristic_seed()`
- `eve_rl/eve_rl/agent/singelagentprocess.py` — `episode_schedule` forwarding through task queue
- `eve_rl/eve_rl/agent/synchron.py` — `episode_schedule` param, `_split_schedule()` splitter
- `training _scripts/DualDeviceNav_train.py` — `build_episode_schedule()`, schedule generation for heuristic seeding

---

## 4. Minimum Success Rate for Heuristic Seeding

### Context

Heuristic seeding runs N episodes to fill the replay buffer with demonstration data before SAC training. However, the heuristic controller doesn't always succeed — wrong branches, overshoot near target, sim errors. If most seeded episodes are failures, SAC starts with very few positive-reward demonstrations, weakening the purpose of seeding.

### Problem

With `--heuristic_seeding 100`, all 100 episodes were pushed to the replay buffer regardless of outcome. If only 10% were successes, SAC would see mostly failure trajectories in its initial training batches.

### Fix

Implemented a two-phase seeding system with deferred buffer push:

**Phase 1 — Batched collection with deficit-based retry:**

1. First batch runs N episodes with `push_to_buffer=False` (episodes collected but not pushed)
2. Counts successes via `episode.infos[-1]["success"]` (clearer contract than `terminals[-1]`)
3. If successes < `ceil(0.3 * N)`, estimates deficit and runs more episodes:
   - `needed = min_successes - n_success_so_far`
   - `observed_rate = max(n_success / n_total, 0.05)` (floor at 5%)
   - `batch_size = ceil(needed / observed_rate * 1.5)` (1.5× safety margin)
4. Each retry batch gets a new branch-balanced schedule
5. Repeats until ≥ 30% successes or hits safety cap (5× N total episodes)

**Phase 2 — Filtering and push:**

1. Keeps **all** successes (even if > N)
2. Ensures success ratio ≤ 70%: if too many successes, pads with failures so SAC also learns from failure transitions
   - `min_failures = ceil(n_success / 0.7) - n_success`
3. Fills remaining slots up to at least N with randomly sampled failures
4. Uses a dedicated RNG (`np.random.default_rng(42 + 999)`) for failure sampling — fully reproducible
5. Pushes the final selected set to replay buffer
6. Cache saves only the selected set, not all attempts

### Example: --heuristic_seeding 100 --min_success_rate 0.3

```
Batch 1: 100 episodes, 18 successes | Total: 100 episodes, 18 successes (18.0%)
Batch 2: 100 episodes, 20 successes | Total: 200 episodes, 38 successes (19.0%)
Heuristic seeding complete: pushing 38 successes + 62 failures = 100 episodes (38.0% success rate)
```

### CLI Arguments Added

- `--min_success_rate` (float, default 0.3) — minimum fraction of successful episodes
- `--max_seeding_multiplier` (int, default 5) — safety cap: max total = N × multiplier

### Files Changed

- `eve_rl/eve_rl/agent/single.py` — `push_to_buffer` param on `heuristic_seed()`; conditional push
- `eve_rl/eve_rl/agent/singelagentprocess.py` — forward `push_to_buffer` through task queue
- `eve_rl/eve_rl/agent/synchron.py` — forward `push_to_buffer` to workers
- `training _scripts/DualDeviceNav_train.py` — batched collection loop, filtering logic, CLI args

---

## Complete File Change Summary

| File | Changes |
|------|---------|
| `eve/eve/observation/localguidance.py` | Normalize 4 features to [0,1] / [-1,1]; update `space` bounds |
| `eve/eve/intervention/monoplanestatic.py` | `velocity_limit_low` param for asymmetric action space; forward `options` to `target.reset()` |
| `eve/eve/intervention/target/centerlinerandom.py` | `target_branch` param; per-branch target pools; `target_branch_names` property |
| `eve_bench/eve_bench/dualdevicenav.py` | `velocity_limit=(30, 1.5)`; `velocity_limit_low=[[-10, -1.5], [-10, -1.5]]` |
| `eve_rl/eve_rl/agent/single.py` | `episode_schedule` + `push_to_buffer` on `heatup()` and `heuristic_seed()` |
| `eve_rl/eve_rl/agent/singelagentprocess.py` | Forward `episode_schedule` + `push_to_buffer` through task queue |
| `eve_rl/eve_rl/agent/synchron.py` | `episode_schedule` + `push_to_buffer` params; `_split_schedule()` round-robin |
| `training _scripts/DualDeviceNav_train.py` | `build_episode_schedule()`; heatup bounds; balanced heuristic schedule; batched collection with min success rate; `--min_success_rate` and `--max_seeding_multiplier` CLI args |

---
---

# Pending Improvements (Not Yet Implemented)

The following improvements are planned but not yet coded. They are ordered by dependency chain and expected impact.

---

## 5. [PENDING] Reduce Network Size — CLI-Only Change

### Problem

Default architecture is **4×900 MLP + 1-layer 500-node LSTM ≈ 11M parameters** for a 26-dim input / 4-dim output problem. Standard continuous control benchmarks with 100+ dim observations use 2×256 MLPs (~200k params). This is 50× overparameterized, causing slower convergence and overfitting to replay buffer contents.

### Fix

Change CLI launch parameters: `--hidden 256 256` or `--hidden 512 512`. The `--hidden` flag already exists — zero code changes required.

### Impact: HIGH | Effort: Trivial

---

## 6. [PENDING] Increase Update Ratio

### Problem

`UPDATE_PER_EXPLORE_STEP = 1/20` — only 1 gradient step per 20 environment steps. SOFA simulation is expensive (~50-100ms/step) but gradient computation is cheap (~1ms on GPU). The collected data is severely underutilized.

### Fix

Change constant in `DualDeviceNav_train.py` to `1/4` or `1/2`. Single-line change.

### Risk

Too-high ratio with small replay buffer can cause critic overfitting. Monitor Q-values for divergence.

### Impact: MEDIUM | Effort: Trivial

---

## 7. [PENDING] Drop LSTM — Switch to MLP-Only

### Problem

The LSTM adds ~1M params and forces episode-based replay (padded sequences). The observation already encodes temporal context: `Memory(2, FILL)` provides velocity-like info, `LastAction` provides previous command, `LocalGuidance` provides 8-dim spatial state. These approximate a Markov state — the LSTM is redundant for local navigation along a known path.

### Fix

Set `ff_only=True` in `BenchAgentSynchron`. The flag already exists in `training _scripts/util/agent.py`.

### Unlocks

Dropping LSTM is a **prerequisite** for:
- Step-based replay (§8)
- Prioritized Experience Replay (§9)
- Hindsight Experience Replay (§10)

### Validation

Run `ff_only=True` vs `ff_only=False` with identical env and hyperparameters. If success rate is comparable, switch permanently.

### Impact: MEDIUM (HIGH as enabler) | Effort: Low (flag exists)

---

## 8. [PENDING] Switch to Step-Based Replay

### Problem

Episode-based replay (`VanillaEpisodeShared`) pads variable-length episodes to the longest in the batch. This wastes memory, limits batch diversity (32 episodes vs 256+ transitions), and prevents transition-level techniques.

### Fix

Switch from `VanillaEpisodeShared` to `VanillaStepShared` in `training _scripts/util/agent.py`. Increase batch size to 256. Requires LSTM removal first (§7).

| Aspect | Episode-Based (current) | Step-Based (proposed) |
|--------|------------------------|----------------------|
| Batch size | 32 episodes (~32k transitions padded) | 256 individual transitions |
| Sample diversity | Low — 32 trajectories per batch | High — transitions from hundreds of episodes |
| Memory efficiency | Wastes memory on padding | No padding overhead |
| PER/HER compat | No | Yes |

### Impact: HIGH (enables §9 and §10) | Effort: Medium

---

## 9. [PENDING] Prioritized Experience Replay (PER)

### Problem

With 10-20% success rate, ~80% of the replay buffer is failure episodes. Uniform sampling wastes gradient updates on redundant failure patterns. Rare successes and surprising transitions are undersampled.

### Design

Implement PER (Schaul et al., 2016) on step-based replay. Requires §7 (drop LSTM) and §8 (step-based replay).

**Architecture:**

```
PrioritizedStepShared (new, extends VanillaSharedBase)
  └─ subprocess with PrioritizedStep
       ├─ ring buffer of transitions
       ├─ SumTree for O(log n) proportional sampling
       └─ priority array
```

**Algorithm (Proportional Variant):**
- Sampling probability: `P(i) = p_i^α / Σ p_j^α` where `p_i = |TD_error_i| + ε`
- Importance sampling weights: `w_i = (N · P(i))^(-β) / max(w)` (corrects bias)
- `α = 0.6` (prioritization exponent), `β` annealed from 0.4 → 1.0
- New transitions get max priority (sampled at least once)

**Core data structure:** SumTree — binary tree with O(log n) sample + update.

**Files to create:**
- `eve_rl/eve_rl/replaybuffer/sumtree.py` — SumTree data structure
- `eve_rl/eve_rl/replaybuffer/prioritizedstep.py` — PrioritizedStep buffer
- `eve_rl/eve_rl/replaybuffer/prioritizedshared.py` — multiprocessing wrapper

**Files to modify:**
- `eve_rl/eve_rl/algo/sac.py` — extract per-sample TD-errors, apply IS weights to loss
- `eve_rl/eve_rl/agent/single.py` — priority update loop after `algo.update()`
- `training _scripts/util/agent.py` — PER buffer option
- `training _scripts/DualDeviceNav_train.py` — `--per` CLI flag

**Key integration point — SAC update:**
```python
# In sac.py update(), when PER batch detected:
td_errors = |curr_q1 - expected_q|          # per-sample TD errors
weighted_loss = (is_weights * td_errors²).mean()  # IS-weighted MSE
# Return td_errors for priority update
```

### Impact: MEDIUM | Effort: Medium | Depends on: §7, §8

---

## 10. [PENDING] Hindsight Experience Replay (HER)

### Problem

Most episodes fail — the guidewire reaches maybe 60% of the path before truncation. That 60% traversal contains useful navigation data (steering through the aortic arch, taking correct branches), but it's labeled with low cumulative reward because the terminal bonus (+1.0) was never received. This data is wasted.

### Design

Implement HER (Andrychowicz et al., 2017) on step-based replay. When an episode fails, relabel it with a "virtual goal" at the position actually reached. The failed episode becomes a "success" for the virtual target.

**Requirements:**
- Goal-conditioned policy: observation must include target position (already present as `target` in ObsDict)
- Step-based replay (§8) — easier to relabel individual transitions than padded episodes
- Reward recomputation: `ArcLengthProgress` needs to accept a target arclength parameter to recompute rewards for virtual targets

**Relabeling strategy:** "future" — for each transition at step t, sample a virtual goal from a future state in the same episode (step t+k, k random). Recompute:
- Target observation → virtual target position
- `d_rem` → recalculated for virtual target arclength
- `TargetReached` reward → fires if tip was within threshold of virtual target
- Store the relabeled transition alongside the original

**Impact on sample efficiency:** In standard robotics benchmarks, HER improves sample efficiency by 2-10× in sparse-reward settings. Here, with 10-20% success rate, most of the buffer would gain virtual success labels, dramatically increasing the effective success rate.

**Files to modify:**
- `eve_rl/eve_rl/replaybuffer/` — HER-aware buffer that relabels on sampling
- `eve/eve/reward/arclengthprogress.py` — accept target arclength for reward recomputation
- `eve/eve/observation/localguidance.py` — recompute guidance features for virtual target
- `training _scripts/DualDeviceNav_train.py` — `--her` CLI flag

### Impact: HIGH | Effort: High | Depends on: §7, §8

---

## 11. [PENDING] Fix Curriculum Per-Worker Step Counter

### Problem

`ActionCurriculumWrapper._total_steps` in `action_curriculum.py` is a local instance variable. Each of the N workers has its own copy. Workers transition between curriculum stages at different times, mixing data from different stages in the replay buffer.

### Fix

Use `multiprocessing.Value('i', 0)` shared counter, or inject the global explore step count from the runner.

### Impact: LOW-MEDIUM | Effort: Low

---

## 12. [PENDING] Add Catheter Reward Signal

### Problem

`ArcLengthProgress` only tracks guidewire tip (`tracking3d[0]`). The catheter has zero direct reward signal — it only affects reward indirectly through SOFA's mechanical coupling. In curriculum Stages 2-3, catheter learning is extremely slow.

### Fix

Add a catheter-specific reward component:
- Penalize excessive catheter lead (tip ahead of guidewire)
- Reward catheter staying in a "support zone" (a few cm behind guidewire tip)
- Penalize catheter cross-track distance

### Impact: MEDIUM | Effort: Medium

---

## 13. [PENDING] Potential-Based Reward Shaping (PBRS)

### Problem

The lateral penalty `-0.001 * cross_track_dist` in `ArcLengthProgress` is not potential-based — it can change the optimal policy. The agent may avoid paths where cross-track distance is temporarily high (e.g., during bifurcation maneuvers) even if that path reaches the target faster.

### Fix

Reformulate as pure PBRS:
```
Φ(s) = -0.01 * d_rem - 0.001 * cross_track_dist
r_shaping = γ * Φ(s') - Φ(s)
```
This preserves the dense signal while guaranteeing policy equivalence with sparse target-reached reward.

### Impact: MEDIUM | Effort: Medium

---

## 14. [PENDING] Feature Ablation Study

### Problem

`on_correct_branch` (feature 7) is derived from `cross_track_dist` (feature 1) via a 5mm threshold — adds zero information. `heading_error` (feature 4) may be inferrable from tracking + tangent.

### Fix

Add a `features` constructor parameter to `LocalGuidance` to select feature subsets:
- **Full-8**: all features (baseline)
- **Core-5**: features 0-4 (d_rem, cross_track, tangent_x, tangent_z, heading_error)
- **Core-4**: features 0-3 (drop heading_error too)

Run each with identical seeds, compare success rates at 100k/250k/500k updates.

### Impact: MEDIUM | Effort: Low

---

## Dependency Graph

```
§5 Reduce network ──────────────────────────────────── (independent, do first)
§6 Update ratio ────────────────────────────────────── (independent, do first)
§7 Drop LSTM ───────┬──→ §8 Step replay ──┬──→ §9 PER
                    │                      └──→ §10 HER
                    │
§11 Curriculum fix ─────────────────────────────────── (independent)
§12 Catheter reward ────────────────────────────────── (independent)
§13 PBRS reward ────────────────────────────────────── (independent)
§14 Feature ablation ───────────────────────────────── (independent)
```

## Priority Order

| Priority | Item | Effort | Impact | Depends on |
|----------|------|--------|--------|------------|
| 1 | §5 Reduce network (CLI) | Trivial | HIGH | — |
| 2 | §6 Increase update ratio | Trivial | MEDIUM | — |
| 3 | §7 Drop LSTM (ff_only) | Low | HIGH (enabler) | — |
| 4 | §8 Step-based replay | Medium | HIGH (enabler) | §7 |
| 5 | §9 PER | Medium | MEDIUM | §7, §8 |
| 6 | §10 HER | High | HIGH | §7, §8 |
| 7 | §11 Curriculum fix | Low | LOW-MEDIUM | — |
| 8 | §12 Catheter reward | Medium | MEDIUM | — |
| 9 | §13 PBRS reward | Medium | MEDIUM | — |
| 10 | §14 Feature ablation | Low | MEDIUM | — |
