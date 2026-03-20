# RL Training Lifecycle — Multi-Worker SAC with Episode Replay

Detailed description of how the eve_rl training pipeline works, from initialization to convergence. Based on the `DualDeviceNav_train.py` training script and the `eve_rl` framework.

---

## Architecture Overview

There are **3 types of processes** running simultaneously:

| Process | Count | Role | Device |
|---------|-------|------|--------|
| **Main** | 1 | Orchestrates phases (heatup → explore/update → eval), logs results, saves checkpoints | CPU |
| **Workers** (`SingleAgentProcess`) | 16 | Each has its own SOFA env + play-only policy copy. Runs episodes, pushes completed episodes to shared replay buffer | CPU |
| **Trainer** (`SingleAgentProcess`) | 1 | Has the full SAC algorithm. Samples batches from shared replay buffer, runs gradient updates | GPU (`cuda:0`) |

### Shared State

All processes share:
- **`StepCounterShared`** / **`EpisodeCounterShared`** — multiprocessing shared counters (with locks) tracking heatup steps, explore steps, update steps, and episode counts
- **`VanillaEpisodeShared`** — replay buffer running in its own subprocess, mediating between workers (push) and trainer (sample) via `mp.SimpleQueue`s

### Replay Buffer Details

- **Type**: `VanillaEpisodeShared` → internally uses `VanillaEpisode` (ring buffer of episodes)
- **Capacity**: `REPLAY_BUFFER_SIZE = 10,000` **episodes** (not steps)
- **Batch size**: `BATCH_SIZE = 32` episodes per gradient update
- **Sampling**: `random.sample(self.buffer, 32)` — **uniform random**, no prioritization. Every episode in the buffer is equally likely to be sampled regardless of age.
- **Minimum to sample**: Buffer needs **> 32 episodes** before any batch can be produced
- **Storage**: Each episode stored as `(flat_obs[], actions[], rewards[], terminals[])` numpy arrays
- **Padding**: Episodes of different lengths are padded with `pad_sequence` (padding value = `inf` for rewards, masked out during training)

### Episode Replay (not Step Replay)

This is a critical architectural choice. The replay buffer stores and samples **full episodes**, not individual transitions:

- Each worker builds a complete `Episode` object locally, step by step
- Only when the episode ends (terminal or truncation) does it get pushed to the buffer as a single unit
- The trainer samples 32 **entire episodes** per gradient step, feeds them through the LSTM sequentially
- **i.i.d. violation**: Steps within each episode are correlated. This is accepted as a tradeoff for LSTM temporal memory.
- The 32 sampled episodes are independent of each other (inter-episode i.i.d.), but intra-episode steps are sequential (not i.i.d.)

---

## Training Constants

From `DualDeviceNav_train.py`:

```python
HEATUP_STEPS = 1e4              # 10,000 steps total across all workers
TRAINING_STEPS = 2e7            # 20,000,000 explore steps until done
CONSECUTIVE_EXPLORE_EPISODES = 100  # Episodes per explore round (shared across workers)
EXPLORE_STEPS_BTW_EVAL = 2.5e5  # 250,000 explore steps between evaluations
REPLAY_BUFFER_SIZE = 1e4        # 10,000 episodes
BATCH_SIZE = 32                 # 32 full episodes per gradient update
UPDATE_PER_EXPLORE_STEP = 1/20  # 1 gradient update per 20 explore steps
GAMMA = 0.99
REWARD_SCALING = 1
CONSECUTIVE_ACTION_STEPS = 1    # No action repeat
LR = 0.00022                    # ~2.2e-4 (from Optuna search)
LR_END_FACTOR = 0.15            # LR decays to 15% of initial
LR_LINEAR_END_STEPS = 6e6       # LR decay over 6M update steps
EVAL_SEEDS = [1,2,3,...,175]    # 98 fixed seeds for evaluation
```

---

## Phase 1: Heatup

**Code**: `runner.training_run()` → `self.heatup(HEATUP_STEPS)`

**What happens**:
1. All 16 workers start simultaneously
2. Each worker runs episodes with **random actions** (uniform between `heatup_action_low` and `heatup_action_high`)
3. Each completed episode is pushed to the shared replay buffer
4. Workers keep going until the **shared** `step_counter.heatup` reaches 10,000 steps total (across all workers)
5. **No gradient updates happen** — the trainer process is idle

**Result**: ~10–60 random episodes in the replay buffer (depending on episode length, ~200–1000 steps each)

**Critical insight — why 500k heatup failed**: The `UPDATE_PER_EXPLORE_STEP` formula counts heatup steps:
```python
total_experience = step_counter.heatup + step_counter.exploration
update_steps = total_experience * (1/20) - step_counter.update
```
With 500k heatup, the first explore_and_update cycle demanded `550k / 20 = 27,500` gradient updates — all on a buffer filled with ~1000 random episodes. The trainer had to grind through all 27,500 steps before the cycle could advance (workers idle, waiting). This caused:
- Networks overfitting to random data (each episode sampled ~880 times)
- Critic and policy hardened into a local minimum
- New policy episodes couldn't shift the learned Q-values

With 10k heatup, only ~3,000 gradient updates were needed, and the buffer quickly became dominated by policy episodes.

---

## Phase 2: Optional Heuristic Seeding

**Code**: Runs after agent/runner creation, before `training_run()`

**What happens**:
1. Creates a separate env + `CenterlineFollowerHeuristic` controller
2. Runs N episodes using the heuristic (not the policy)
3. Each episode is packaged as an `Episode` object with **normalized actions** (mapped to [-1, 1] to match what SAC expects with `normalize_actions=True`)
4. Episodes pushed to the shared replay buffer via `agent.replay_buffer.push(episode)`

**Why it matters**: Replaces random heatup junk with trajectories showing forward progress. Those early gradient updates (which must happen before anything else) now learn from useful data instead of noise.

**Recommended**: ~100 episodes with 10k heatup, or ~5–10k episodes with minimal heatup (to fill buffer with useful data)

---

## Phase 3: Training Loop (Explore & Update)

**Code**: `runner.training_run()` → outer while loop

```
while exploration_steps < 20M:
    explore_and_update(until next_eval_limit)    # inner loop
    eval(98 seeds)
    next_eval_limit += 250,000
```

### Inner Loop: Explore & Update Cycles

**Code**: `runner.explore_and_update()` → calls `agent.explore_and_update()` repeatedly

Each cycle:

```
while exploration_steps < next_eval_limit:
    1. Compute: update_steps = (heatup + explore) * (1/20) - updates_done
    2. Launch trainer.update(update_steps)     ──┐
    3. Launch workers.explore(100 episodes)    ──┤  IN PARALLEL
    4. Wait for BOTH to finish                 ──┘
    5. Sync weights: trainer → all 16 workers
```

#### Step 1: Compute update budget

```python
total_experience_steps = step_counter.heatup + step_counter.exploration
update_steps = total_experience_steps * (1/20) - step_counter.update
```

This maintains a ratio: for every 20 explore steps accumulated (including heatup), there should be 1 gradient update. The formula computes how many updates are "owed" and schedules them.

#### Steps 2–3: Parallel execution

**Trainer** (GPU): Loops `update_steps` times, each iteration:
```python
batch = replay_buffer.sample()    # 32 random full episodes
result = algo.update(batch)        # one SAC gradient step (Q1, Q2, policy, alpha)
```

**Workers** (CPU, 16 processes): Each runs episodes using the current policy copy. After each episode completes, it's pushed to the shared replay buffer. Workers continue until the shared `episode_counter.exploration` reaches the target (previous count + 100 episodes).

**During this time**: Workers are pushing fresh episodes into the buffer while the trainer is simultaneously sampling from it. New episodes become available for sampling immediately after being pushed.

#### Step 4: Wait for both

```python
while True:
    poll workers for results...
    poll trainer for results...
    if got_worker_results and got_trainer_results:
        break
```

**Both must finish.** If workers finish first (typical when update backlog is large), they sit idle. If the trainer finishes first, it waits for workers. The cycle cannot advance until both are done.

#### Step 5: Weight sync

```python
self._update_algo_state_dicts()                              # trainer → main (optimizer, scheduler)
self._worker_load_state_dicts_network(algo.state_dicts_network())  # main → workers (policy, Q-nets)
```

Workers receive updated network weights. This is the **only point** where workers get new policy weights — they run 100 episodes with stale weights during each cycle.

### Typical cycle numbers

With ~500 steps/episode and 100 episodes/cycle:
- ~50,000 explore steps per cycle
- ~2,500 gradient updates per cycle (50k / 20)
- ~5 cycles between evaluations (250k / 50k)
- Each gradient update samples 32 episodes from the buffer

---

## Phase 4: Evaluation

**Code**: `runner.eval(seeds=EVAL_SEEDS)` — runs every 250,000 explore steps

**What happens**:
1. 98 fixed seeds split across 16 workers (~6 episodes each)
2. Workers run episodes using the **current policy** (deterministic or stochastic depending on `stochastic_eval`)
3. Episodes are **not** added to the replay buffer
4. Results computed: success rate, path ratio, average reward, translation speed, trajectory length
5. Checkpoint saved; best checkpoint tracked separately by quality (success rate)

**Eval is blocking**: No exploration or training happens during evaluation. After eval completes, the next explore_and_update cycle begins.

---

## Full Timeline Visualization

```
HEATUP (10k steps)
│  16 workers run random actions
│  ~10-60 episodes → buffer
│  No gradient updates
│
[Optional: Heuristic seeding → buffer]
│
├── CYCLE 1  ─────────────────────────────────
│   Trainer: ~2,750 gradient updates (GPU)  ←── parallel
│   Workers: 100 episodes (16×SOFA, CPU)    ←── parallel
│   Buffer: receives ~100 new episodes
│   Weight sync → workers
│
├── CYCLE 2  ─────────────────────────────────
│   Trainer: ~2,500 gradient updates
│   Workers: 100 episodes
│   Weight sync → workers
│
├── ... (3 more cycles to reach 250k explore steps)
│
EVAL #1 (98 episodes on fixed seeds, not stored)
│   Log: success rate, reward, path ratio
│   Save checkpoint
│
├── CYCLE 6-10  ──────────────────────────────
│   ...
│
EVAL #2 (98 episodes)
│
├── CYCLE 11-15  ─────────────────────────────
│   ...
│
EVAL #3 ...
│
... continues until 20M explore steps (~80 evals total)
```

---

## Key Implications for Training

### Buffer dynamics
- With 10k episode capacity and 100 episodes/cycle, the buffer fills in ~100 cycles
- Once full, old episodes get overwritten (ring buffer)
- All episodes in the buffer have equal sampling probability — old heatup episodes persist until overwritten
- No prioritized experience replay — rare successes are sampled at the same rate as failures

### Stale weights
- Workers use the same policy weights for an entire cycle (100 episodes)
- This means the data being pushed to the buffer was generated by an older policy
- Acceptable for SAC (off-policy), but means the buffer always lags behind the current policy

### Update-to-data ratio
- `1/20` = one gradient step per 20 env steps
- This is conservative. Higher ratios (1/5, 1/1) would train faster per env step but risk instability
- With episode replay + LSTM, each "gradient step" processes 32 full episodes (potentially 32 × 1000 = 32,000 transitions), so the effective data usage per gradient step is much higher than it appears

### Learning rate schedule
- Starts at ~2.2e-4
- Linearly decays to 15% of initial (3.3e-5) over 6M update steps
- At 1/20 ratio, 6M update steps ≈ 120M explore steps — beyond the 20M training budget
- So in practice, the LR only decays to about `2.2e-4 * (1 - (1M/6M) * (1 - 0.15))` ≈ 1.9e-4 by end of training. Minimal decay.

---

## Optional: Action-Space Curriculum

If `--curriculum` is enabled, an `ActionCurriculumWrapper` wraps the env:

- **Stage 1** (0–200k steps): Policy outputs 4D, but catheter dims are overwritten: `cath_trans = gw_trans * 0.8`, `cath_rot = 0`. Effectively 2D control.
- **Stage 2** (200k–500k): Catheter actions scaled × 0.1
- **Stage 3** (500k+): Full 4D control

**Note**: The wrapper modifies actions *after* the policy outputs them but *before* the env executes them. The replay buffer stores the **original policy actions** (pre-modification), not the modified ones. Since the reward only depends on the guidewire tip position (not catheter actions directly), the action-reward mismatch is minimal — the catheter only affects reward indirectly through SOFA physics (mechanical support changing guidewire behavior).

Step counting is per-worker instance, not globally shared — each worker tracks its own curriculum stage independently.
