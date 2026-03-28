# RL Improvement Analysis — Design, Algorithmic & Strategic Improvements

## Context & Background

This document merges findings from three independent analyses of the env4/env5 RL training system for endovascular navigation:

1. **This session's deep audit** — systematic code review of env4.py, DualDeviceNav_train.py, and all supporting modules
2. **env5.md (Cursor analysis)** — env4 code analysis covering design rationale, observation feature redundancy, heuristic assessment, and ablation recommendations
3. **chat7.md (Cursor session)** — implementation plan for env5 optimizations with file-level detail

**What this document covers**: Everything *beyond* code-level inefficiencies (which have already been implemented in env5.py). This includes design improvements, algorithmic changes, alternative approaches, ablation studies, and heuristic controller assessment.

**Current state**: The RL system trains SAC agents to navigate endovascular vessels using SOFA simulation. env.py (v1) achieves 10-20% success after 80k-200k updates. env2/env3 failed (policy collapse — guidewire stuck at start). env4 redesigned reward (continuous arclength) and observation (8-dim local guidance). env5 optimized env4's computational overhead. None have been validated in training yet.

---

## 1. Design & Algorithmic Improvements

### 1.1 LocalGuidance Features Are Unnormalized — HIGH Impact

**Problem**: The 8 guidance features have wildly different scales, creating a 200:1 disparity when fed alongside normalized tracking/target observations:

| Feature | Range | Scale |
|---------|-------|-------|
| `d_rem_norm` | [0, 1] | 1 |
| `cross_track_dist` | [0, 50] mm | 50 |
| `tangent_x_2d`, `tangent_z_2d` | [-1, 1] | 2 |
| `heading_error` | [-π, π] | ~6.28 |
| `curvature_ahead` | [0, 10] | 10 |
| `dist_to_bifurcation` | [0, 200] mm | 200 |
| `on_correct_branch` | {0, 1} | 1 |

**Why it matters**: The tracking and target observations are normalized to [-1, 1] via `NormalizeTracking2DEpisode`. When the flat observation vector is fed to the MLP, features like `dist_to_bifurcation` (0-200) dominate gradient updates over `d_rem_norm` (0-1) or `tangent` (-1 to 1). This biases the network toward large-scale features and slows learning of critical steering signals.

**File**: `eve/eve/observation/localguidance.py` — observation space definition at lines 77-85

**Recommendation**: Normalize each feature to [0, 1] or [-1, 1] internally within `LocalGuidance.step()`:
- `cross_track_dist` → divide by `_MAX_CROSS_TRACK_MM` (50)
- `heading_error` → divide by π
- `curvature_ahead` → divide by 10 (max clip)
- `dist_to_bifurcation` → divide by `_MAX_BIFURC_DIST_MM` (200)
- Others are already in [0,1] or [-1,1]

**Effort**: Low — add 4 division operations in `step()`.

---

### 1.2 Network Massively Overparameterized — HIGH Impact

**Problem**: Default architecture is **4×900 MLP + 1-layer 500-node LSTM**, totaling ~11M parameters for a problem with ~26-dim input and 4-dim output.

**Why it matters**: Standard continuous control benchmarks (MuJoCo Ant, Humanoid) with 100+ dim observations use 2×256 MLPs (~200k params). This system has 50× more parameters for a simpler observation space. Overparameterization causes:
- Slower convergence (more parameters to optimize)
- Overfitting to replay buffer contents (especially with only 10k episodes)
- Longer gradient computation per update step

**File**: `training _scripts/DualDeviceNav_train.py` — `--hidden` default at line 88, `--embedder_nodes` at line 94

**Recommendation**: Try `--hidden 256 256` or `--hidden 512 512`. The `--hidden` flag already exists as a CLI argument, so this requires no code changes — just different launch parameters.

**Effort**: Trivial — CLI flag change only.

---

### 1.3 Update Ratio Too Low — MEDIUM Impact

**Problem**: `UPDATE_PER_EXPLORE_STEP = 1/20` — only 1 gradient step per 20 environment steps. Standard SAC uses 1:1.

**Why it matters**: SOFA simulation is expensive (~50-100ms per step). The gradient computation is cheap (~1ms on GPU). With a 1:20 ratio, the collected data is severely underutilized. Each transition is seen very few times before new data displaces it. Increasing the ratio extracts more learning from each expensive simulation step.

**File**: `training _scripts/DualDeviceNav_train.py` line 33

**Recommendation**: Increase to 1/4 or 1/2. Monitor critic loss for overfitting — if Q-values diverge, reduce. The update ratio is a constant, so this is a single-line change.

**Risk**: Too-high ratio with a small replay buffer can cause critic overfitting. Mitigate by also increasing replay buffer size (currently 10k episodes).

**Effort**: Trivial — change one constant.

---

### 1.4 LSTM May Be Unnecessary — MEDIUM Impact

**Problem**: The LSTM embedder adds ~1M parameters and forces episode-based replay (padded sequences with variable-length batching), but the observation already encodes temporal context.

**Why it matters**: The current observation includes:
- 2-frame `Memory` wrapper on tracking — provides velocity-like information (position at t and t-1)
- `LastAction` — tells the agent what it just did
- `LocalGuidance` — 8-dim spatial state relative to path

These features approximate a Markov state. The LSTM is mainly useful if the agent needs to remember events from many steps ago (e.g., "I was on a wrong branch 50 steps ago"). For local navigation along a known path, this is unlikely to be necessary.

**Consequences of removing LSTM**:
- Can switch from episode-based replay (`VanillaEpisodeShared`) to step-based replay (`VanillaStepShared`)
- Enables batch sizes of 128-512 individual transitions (vs 32 padded episodes)
- Enables transition-level prioritization (PER)
- Enables Hindsight Experience Replay (HER)
- Simpler weight synchronization between workers

**File**: `training _scripts/util/agent.py` — the `ff_only` flag already exists in `BenchAgentSynchron`

**Recommendation**: Test with `ff_only=True`. If success rate is comparable, switch permanently and unlock step-based replay benefits.

**Effort**: Low — flag already exists. Step-based replay switch is medium effort.

---

### 1.5 Heatup Action Ranges Not Updated — MEDIUM Impact

**Problem**: The training script uses old aggressive heatup ranges:
```python
heatup_action_low=[[-10.0, -1.0], [-11.0, -1.0]],
heatup_action_high=[[35, 3.14], [30, 3.14]],
```

**Why it matters**: Per `SOFA_TIMEOUT_FIX.md`, these aggressive actions (35mm/step at 7.5Hz = 262mm/s) were already identified as causing SOFA numerical errors ("Case 1" precision failures). The fix document reduced ranges to `[[15.0, 1.0], [12.0, 1.0]]`, but the training script was not updated.

**File**: `training _scripts/DualDeviceNav_train.py` lines 264-265

**Recommendation**: Update to match the safe ranges:
```python
heatup_action_low=[[-10.0, -1.0], [-11.0, -1.0]],
heatup_action_high=[[15.0, 1.0], [12.0, 1.0]],
```

**Effort**: Trivial — change two numbers.

---

### 1.6 Curriculum Step Counter Is Per-Worker — LOW-MEDIUM Impact

**Problem**: `ActionCurriculumWrapper._total_steps` is a local instance variable. Each of the 4 workers has its own copy. Workers transition between stages at different times.

**Why it matters**: Worker 0 might be generating Stage 2 data (catheter scaled ×0.1) while Worker 3 is still in Stage 1 (catheter auto-follows). The replay buffer mixes data from different curriculum stages, which can confuse the policy — especially the catheter policy which sees inconsistent action mappings.

**File**: `training _scripts/util/action_curriculum.py` line 47

**Recommendation**: Use `multiprocessing.Value('i', 0)` shared counter, or inject the global explore step count from the runner/agent. The runner already tracks global steps via `StepCounterShared`.

**Effort**: Low — requires threading a shared counter through the env wrapper.

---

### 1.7 No Reward Signal for Catheter Positioning — MEDIUM Impact

**Problem**: `ArcLengthProgress` only tracks guidewire tip position (`tracking3d[0]`). The catheter has zero direct reward signal — it only affects reward indirectly through SOFA's mechanical coupling to the guidewire.

**Why it matters**: In curriculum Stage 2 and Stage 3, the agent must learn catheter control. Without a reward signal, the catheter policy learns only from indirect effects (e.g., pushing the catheter too far causes the guidewire to buckle, reducing progress). This is extremely slow and noisy.

**File**: `eve/eve/reward/arclengthprogress.py` — only uses `tracking3d[0]` (guidewire tip)

**Recommendation**: Add a catheter-specific reward component:
- Penalize excessive catheter lead (catheter tip ahead of guidewire)
- Reward catheter tip staying within a "support zone" (a few cm behind guidewire tip)
- Project catheter tip onto path polyline and penalize large cross-track distance

**Effort**: Medium — new reward class + wiring into env.

---

### 1.8 Heuristic Controller Minimum Push Floor — LOW Impact

**Problem**: `gw_trans = max(gw_trans, 5.0)` ensures at least 5mm/step even when very close to the target (d_rem < 10mm).

**Why it matters**: Near the target, the heuristic overshoots and may trigger truncation or oscillation. Seeded episodes with noisy terminal transitions give the SAC policy incorrect examples of "how to reach the target."

**File**: `training _scripts/util/heuristic_controller.py` line 106

**Recommendation**: Scale the floor with remaining distance: `max(gw_trans, min(5.0, d_rem * 0.5))`

**Effort**: Trivial — one line change.

---

## 2. Observation Feature Analysis & Ablation Study

### 2.1 Feature Redundancy Analysis

From the env5.md analysis, the 8 LocalGuidance features can be categorized:

| Feature | Essentiality | Reasoning |
|---------|-------------|-----------|
| `d_rem_norm` [0,1] | **Essential** | Tells agent "how far to go" — no other source for this |
| `cross_track_dist` [0,50mm] | **Essential** | Tells agent "how far off-path" — critical for corrective steering |
| `tangent_x_2d` [-1,1] | **Essential** | Path direction in fluoroscopy plane — critical for steering |
| `tangent_z_2d` [-1,1] | **Essential** | Path direction in fluoroscopy plane — paired with tangent_x |
| `heading_error` [-π,π] | **Somewhat redundant** | Can be inferred from tracking (device direction) + tangent. However, computing it explicitly saves the network from learning this relationship. Likely beneficial but not strictly necessary. |
| `curvature_ahead` [0,10] | **Useful** | Anticipation signal — alerts the agent to upcoming sharp turns (aortic arch). Could improve performance at difficult bends. |
| `dist_to_bifurcation` [0,200mm] | **Useful but correlated** | Correlates with curvature (bifurcations often coincide with high curvature). Useful as a distinct "fork warning" signal. |
| `on_correct_branch` {0,1} | **Possibly redundant** | With the env5 threshold-based implementation (`cross_track_dist < 5mm`), this is literally derived from `cross_track_dist`. Adds no new information beyond what `cross_track_dist` already provides. |

### 2.2 Ablation Study Plan

**Goal**: Determine the minimal feature set that maintains navigation performance.

**Configurations to test**:

| Config | Features | Dims | Rationale |
|--------|----------|------|-----------|
| **Full** | All 8 | 8 | Baseline — current implementation |
| **Core-5** | 0-4 (d_rem, cross_track, tangent_x, tangent_z, heading_error) | 5 | Essential features only. Tests whether anticipation features add value. |
| **Core-4** | 0-3 (d_rem, cross_track, tangent_x, tangent_z) | 4 | Removes heading_error (redundant with tangent + tracking). Tests whether explicit heading error computation helps. |
| **Full-minus-branch** | 0-6 (all except on_correct_branch) | 7 | Removes the feature that's now derived from cross_track_dist anyway |

**Implementation**: Add a constructor parameter `features: str = "full"` to `LocalGuidance` that selects the feature set. Modify `space` property and `step()` accordingly.

**Metrics**: Success rate, convergence speed (episodes to first success), average insertion depth at convergence.

**Why this matters**: Fewer features = smaller observation space = faster learning (less for the network to disentangle). If Core-5 matches Full, the anticipation features add complexity without benefit.

---

## 3. Heuristic Controller Assessment

### 3.1 Current Design Strengths

The `CenterlineFollowerHeuristic` (in `training _scripts/util/heuristic_controller.py`) is well-designed:

- **Combined heading + cross-track correction** (line 137): `gw_rot = -heading_kp * heading_error - crosstrack_kp * cross_track_signed` — steers toward the centerline AND aligns with the path direction simultaneously
- **Noise injection** (lines 141-142): 10% Gaussian noise on both translation and rotation — provides trajectory diversity in seeded data, preventing the SAC policy from memorizing a single trajectory
- **Never-retract safeguard** (line 143): `gw_trans = max(gw_trans, 0.0)` — ensures seeded data always shows forward progress, which is the behavior we want the policy to learn
- **Proportional translation** (line 105): `min(max_translation, d_rem * 0.1)` — slows down near the target, which is realistic and teaches the policy to decelerate

### 3.2 Identified Issues

1. **Minimum push floor too high** (covered in §1.8): 5mm/step near target causes overshoot
2. **No catheter rotation**: `cath_rot = 0.0` always. If catheter rotation matters in later training stages, the seeded data provides no signal for this action dimension
3. **Heading vs cross-track gain balance**: `heading_kp=1.0` vs `crosstrack_kp=0.05` — the 20:1 ratio strongly favors alignment over centering. If vessels are wide relative to cross-track deviation, this is fine. If vessels are narrow, the controller may not correct lateral errors fast enough.

### 3.3 Seeded Data Quality

The heuristic generates "okay" trajectories, not optimal ones. Key characteristics:
- Forward-only (never retracts) — good for learning approach behavior
- Proportional speed (slows near target) — teaches deceleration
- Noisy — prevents policy memorization
- Action-normalized to [-1, 1] — matches SAC replay buffer format

**Potential improvement**: Run heuristic with different gain settings to generate diverse trajectory styles (aggressive vs cautious). This could be done by randomizing `heading_kp` and `crosstrack_kp` per episode during seeding.

---

## 4. Alternative Approaches Worth Considering

### 4.1 Potential-Based Reward Shaping (PBRS)

**Background**: The PBRS theorem (Ng et al., 1999) guarantees that adding a shaping reward of the form `F(s, s') = γ·Φ(s') - Φ(s)` preserves the optimal policy — it can only speed up learning, not change what the agent converges to.

**Current reward**: `r = 0.01 * (d_rem_prev - d_rem_curr) - 0.001 * cross_track_dist`

The progress term `0.01 * (d_rem_prev - d_rem_curr)` is valid PBRS with potential `Φ(s) = -0.01 * d_rem`. However, the lateral penalty `-0.001 * cross_track_dist` falls **outside** the potential framework — it's a per-step penalty that depends only on the current state, not on a state transition.

**Why it matters**: A non-PBRS shaping reward can change the optimal policy. The lateral penalty might cause the optimal policy to avoid paths where cross-track distance is temporarily high (e.g., during a bifurcation maneuver) even if that path leads to the target faster.

**Recommendation**: Reformulate as pure PBRS:
```
Φ(s) = -0.01 * d_rem - 0.001 * cross_track_dist
F(s, s') = γ * Φ(s') - Φ(s)
```
This preserves the dense shaping signal while guaranteeing policy equivalence with the sparse target-reached reward.

**Effort**: Medium — modify `ArcLengthProgress` to implement the PBRS formula instead of the current difference + penalty approach.

---

### 4.2 Hindsight Experience Replay (HER)

**Background**: HER (Andrychowicz et al., 2017) addresses sparse-reward goal-conditioned tasks. When an episode fails to reach the goal, HER creates additional training examples by relabeling the episode with a "virtual goal" at the position actually reached.

**Why it's relevant here**: Most episodes fail — the guidewire reaches maybe 60% of the path before truncation. That 60% traversal contains useful data about navigating the first part of the vessel. With HER, this failed episode becomes a "success" for a virtual target at the 60% point.

**Impact**: Dramatically increases the effective success rate in the replay buffer without additional simulation. In standard robotics benchmarks, HER improves sample efficiency by 2-10×.

**Requirements**:
- Goal-conditioned policy: observation must include the target position (already present as `target` in ObsDict)
- Ability to recompute rewards for alternative targets: requires modifying `ArcLengthProgress` to accept a target arclength parameter
- Step-based replay preferred (easier to relabel individual transitions)

**Effort**: High — requires modifications to replay buffer, reward computation, and episode handling. Best combined with step-based replay (§4.4).

---

### 4.3 Prioritized Experience Replay (PER)

**Background**: PER (Schaul et al., 2016) samples transitions with probability proportional to their TD-error, focusing learning on the most "surprising" experiences.

**Current state**: `VanillaEpisodeShared` uses uniform random sampling. An episode where the agent reaches the target (rare, informative) is sampled at the same rate as a trivial do-nothing episode.

**Why it's relevant**: With 10-20% success rates, ~80% of replay buffer episodes are failures. Uniform sampling wastes most gradient updates on redundant failure patterns. PER would automatically upweight:
- Episodes with target reached (rare successes)
- Episodes with wrong-branch recovery (informative corrections)
- Episodes with novel states (high TD-error)

**Requirements**: TD-error tracking per episode or transition. The SAC trainer already computes Q-values and targets — the TD-error is available.

**Effort**: Medium — requires modifying `VanillaEpisodeShared` to track priorities and implement proportional sampling. Alternatively, use an existing library implementation.

---

### 4.4 Step-Based Replay (Section 3.5 from new_rl_envs.md — Deferred)

**Background**: The current system uses episode-based replay with padded sequences. Each batch contains 32 full episodes, padded to the longest episode length. This is required by the LSTM head.

**Why switch**: Dropping the LSTM (§1.4) unlocks step-based replay, which offers:

| Aspect | Episode-Based (current) | Step-Based (proposed) |
|--------|------------------------|----------------------|
| Batch size | 32 episodes (~32k transitions padded) | 128-512 individual transitions |
| Sample diversity | Low — 32 trajectories per batch | High — transitions from hundreds of episodes |
| Memory efficiency | Wastes memory on padding (padding_value=inf) | No padding overhead |
| Prioritization | Episode-level only | Transition-level (enables PER) |
| HER compatibility | Awkward (must relabel whole episodes) | Natural (relabel individual transitions) |
| Update ratio | Limited by batch padding overhead | Can increase to 1:1 cheaply |

**Implementation**: Switch from `VanillaEpisodeShared` to `VanillaStepShared` in `training _scripts/util/agent.py`. Increase batch size to 256. Increase update ratio to 1/4 or higher.

**Effort**: Medium — replay buffer swap + hyperparameter tuning. Must first validate that MLP-only (no LSTM) works comparably.

---

### 4.5 Bifurcation Point Coordinate System — VERIFIED CORRECT

**Concern**: `LocalGuidance._compute_bifurcation_arclengths()` applies `tracking3d_to_vessel_cs()` to `path_branching_points3d`. Is this correct, or are the points already in vessel CS?

**Verification**: Traced through source code. In `FixedPathfinder.reset()` at `eve/eve/pathfinder/fixedpath.py:148-158`:
- `path_points_vessel_cs` stores points in **vessel CS** (line 148)
- `path_branching_points3d` stores points in **tracking3d space** — they are explicitly converted via `vessel_cs_to_tracking3d()` at line 153

The same convention is used in all three pathfinders (fixedpath:153, dijkstra2:136, bruteforcebfs:74). The naming follows the codebase convention: `*3d` = tracking3d space, `*vessel_cs` = vessel coordinate system.

Therefore, `localguidance.py:244` correctly converts from tracking3d → vessel CS before projecting onto the polyline (which is in vessel CS). **No bug — no action needed.**

---

## 5. Priority Ranking — Full Improvement Roadmap

### Tier 1: Quick Wins (implement before next training run)

| # | Improvement | Section | Effort | Expected Impact |
|---|------------|---------|--------|-----------------|
| 1 | Normalize guidance features | §1.1 | Low | **High** — learning speed |
| 2 | Reduce network to 2×256-512 | §1.2 | Trivial (CLI) | **High** — sample efficiency |
| 3 | Fix heatup action ranges | §1.5 | Trivial | **Medium** — stability |
| 4 | Increase update ratio to 1/4 | §1.3 | Trivial | **Medium** — sample efficiency |
| 5 | Fix heuristic min push floor | §1.8 | Trivial | **Low** — data quality |

### Tier 2: Moderate Effort (implement after validating Tier 1)

| # | Improvement | Section | Effort | Expected Impact |
|---|------------|---------|--------|-----------------|
| 6 | Test MLP-only (drop LSTM) | §1.4 | Low | **Medium** — unlocks Tier 3 |
| 7 | Run ablation study (8 vs 5 dims) | §2.2 | Low | **Medium** — simplification |
| 8 | Fix curriculum per-worker counter | §1.6 | Low | **Low-Medium** — consistency |
| 9 | ~~Verify bifurcation coord system~~ | §4.5 | — | **Verified correct** — no action needed |

### Tier 3: Strategic Changes (implement after MLP validation)

| # | Improvement | Section | Effort | Expected Impact |
|---|------------|---------|--------|-----------------|
| 10 | Switch to step-based replay | §4.4 | Medium | **High** — enables 11-13 |
| 11 | Add catheter reward signal | §1.7 | Medium | **Medium** — catheter learning |
| 12 | Implement PBRS reward | §4.1 | Medium | **Medium** — policy optimality |
| 13 | Implement HER | §4.2 | High | **High** — sample efficiency |
| 14 | Implement PER | §4.3 | Medium | **Medium** — sample focus |

---

## 6. Verification Plan

### For Tier 1 changes
- Run `--env_version 5 --hidden 256 256` with updated heatup ranges
- Compare learning curves (success rate over updates) against env4 baseline
- Monitor critic loss, alpha, and entropy for signs of collapse

### For ablation study
- Run 4 configurations (Full-8, Core-5, Core-4, Full-minus-branch) with identical seeds and hyperparameters
- Compare success rates at 100k, 250k, 500k, 1M updates
- Statistical significance: run each config with 3 random seeds

### For LSTM vs MLP comparison
- Run `ff_only=True` vs `ff_only=False` with identical env and hyperparameters
- Compare convergence speed and final success rate
- If MLP matches LSTM, proceed to step-based replay

### For HER/PER
- Implement on step-based replay first
- Compare sample efficiency (updates to first success) against vanilla replay
