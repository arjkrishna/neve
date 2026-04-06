# Episode-Level Prioritized Experience Replay — Implementation Plan

## 1. Project Context

### What This Project Is

This is an RL training system for **endovascular navigation** — guiding a catheter and guidewire through blood vessels to reach a target in the aortic arch. The simulation runs in **SOFA** (a soft-body physics engine), and the RL algorithm is **SAC (Soft Actor-Critic)** with an **LSTM** head trained via **episode-based replay**.

The task is a DualDeviceNav benchmark: two J-shaped devices (guidewire + catheter), 4D continuous action space `[gw_translation, gw_rotation, cath_translation, cath_rotation]`, targeting one of 4 supra-aortic branches. The agent observes tracking positions (2D projections of device shape), target position, last action, and 8-dim local guidance features (remaining distance, cross-track error, path tangent, curvature, etc.).

### Current Performance

- **env1** (original): 10-20% success rate after 80k-200k SAC updates
- **env2/env3** (waypoint rewards): failed — policy collapsed to do-nothing
- **env4/env5** (arclength progress + local guidance): redesigned reward and observation, not yet validated in training

### Why Episode PER

The replay buffer uses **uniform random sampling** across episodes. With 10-20% success rate, ~80% of buffer content is failure episodes. Uniform sampling wastes gradient updates on redundant failure patterns. Rare successes and episodes with surprising critic errors are undersampled.

Standard PER operates on individual transitions and requires step-based replay (no LSTM). **Episode PER** adapts the same algorithm to entire episodes, working directly with the existing LSTM + episode-based architecture. This avoids the prerequisite chain of dropping LSTM → switching to step replay → implementing step PER.

---

## 2. Key Files to Read Before Implementing

### RL Framework Core

| File | What It Contains | Why You Need It |
|------|-----------------|-----------------|
| `eve_rl/eve_rl/replaybuffer/replaybuffer.py` | `Episode`, `EpisodeReplay`, `Batch` NamedTuple, `ReplayBuffer` ABC | Base classes — your new buffer must match these interfaces |
| `eve_rl/eve_rl/replaybuffer/vanillaepisode.py` | `VanillaEpisode` — the current episode buffer with `push()`, `sample()`, ring buffer storage | **Template** — PrioritizedEpisode mirrors this but replaces `random.sample()` with SumTree |
| `eve_rl/eve_rl/replaybuffer/vanillashared.py` | `VanillaSharedBase` — multiprocessing wrapper with push/sample/task queues, subprocess `loop()`. Also `VanillaEpisodeShared` subclass | **Must modify** — add priority_queue and PrioritizedEpisodeShared subclass |
| `eve_rl/eve_rl/replaybuffer/__init__.py` | Exports | Add new class exports |

### SAC Algorithm

| File | What It Contains | Why You Need It |
|------|-----------------|-----------------|
| `eve_rl/eve_rl/algo/sac.py` | `SAC.update(batch)` returns `[q1_loss, q2_loss, policy_loss]`. Key methods: `_update_q1()`, `_update_q2()`, `_get_expected_q()` | **Must modify** — extract per-episode TD-errors, apply IS weights, return indices+td_errors for PER |

**Critical code path in `sac.py`:**
- `update()` (~line 222): unpacks Batch, calls `_get_expected_q()`, then `_update_q1()`, `_update_q2()`, `_update_policy()`
- `_get_expected_q()` (~line 383): computes `expected_q = rewards + γ(1-done) * min(Q_target1, Q_target2) - α*log_π`
- `_update_q1()` (~line 367): computes `curr_q1 = model.q1(states, actions)`, then `F.mse_loss(curr_q1, expected_q)`. **This is where TD-error lives**: `|curr_q1 - expected_q|`
- `update()` returns `[q1_loss_np, q2_loss_np, policy_loss_np]` — scalar values only. For PER, must also return per-episode TD-errors and batch indices.

**Batch shape (episode-based):**
- `obs`: `(batch_size, seq_len+1, obs_dim)` — includes next state
- `actions`: `(batch_size, seq_len, action_dim)`
- `rewards`: `(batch_size, seq_len, 1)`
- `terminals`: `(batch_size, seq_len, 1)`
- `padding_mask`: `(batch_size, seq_len, 1)` — 1.0 for valid, 0.0 for padded timesteps

### Agent Layer

| File | What It Contains | Why You Need It |
|------|-----------------|-----------------|
| `eve_rl/eve_rl/agent/single.py` | `Single.update()` (~line 411): calls `batch = replay_buffer.sample()` then `result = algo.update(batch)` | **Must modify** — after update, send TD-errors back to buffer via `replay_buffer.update_priorities()` |
| `eve_rl/eve_rl/agent/singelagentprocess.py` | Subprocess dispatcher: `run()` function handles task queue messages | No changes needed — priority updates go through `VanillaSharedBase` queues, not the agent task queue |
| `eve_rl/eve_rl/agent/synchron.py` | `Synchron` multi-worker agent | No changes needed — replay buffer is shared, priorities are updated by the trainer agent |

### Training Script

| File | What It Contains | Why You Need It |
|------|-----------------|-----------------|
| `training _scripts/DualDeviceNav_train.py` | CLI args, env creation, agent creation, `runner.training_run()` | **Must modify** — add `--per`, `--per_mode`, `--per_success_bonus` CLI flags |
| `training _scripts/util/agent.py` | `BenchAgentSynchron` — creates replay buffer, SAC algo, workers | **Must modify** — add `use_per` param, create `PrioritizedEpisodeShared` when enabled |

### Environment (for understanding success detection)

| File | What It Contains | Why You Need It |
|------|-----------------|-----------------|
| `training _scripts/util/env5.py` | `BenchEnv5` — observation, reward, terminal/truncation setup | Context: `terminal=True` means target reached (success). `info["success"]` also available. |
| `eve/eve/reward/arclengthprogress.py` | `ArcLengthProgress` reward — `progress_factor * (d_rem_prev - d_rem_curr) - lateral_penalty * cross_track` | Context: reward structure for understanding TD-error behavior |

---

## 3. Current Replay Buffer Architecture

### Storage: Ring Buffer of Episodes

`VanillaEpisode` stores episodes as numpy tuples in a list:
```python
buffer[i] = (flat_obs, actions, rewards, terminals)  # all np.ndarray
```
- `capacity`: max episodes (default 10,000)
- `position`: write pointer, wraps circularly
- `push()`: stores at `buffer[position]`, increments position
- `sample()`: `random.sample(buffer, batch_size)` — **uniform random, no indices returned**

### Sampling and Padding

Episodes have variable length. `sample()` pads to the longest in the batch:
```python
episodes = random.sample(self.buffer, self.batch_size)
state_batch = pad_sequence([...], batch_first=True)
reward_batch = pad_sequence([...], padding_value=inf)
padding_mask = (reward_batch != inf).float()
```

### Multiprocessing

`VanillaSharedBase` wraps the buffer in a subprocess:
- **Parent → subprocess**: `_push_queue` (episodes), `_task_queue` (commands)
- **Subprocess → parent**: `_sample_queue` (pre-made batches)
- Subprocess proactively generates batches when `_sample_queue` is empty and buffer has enough data
- `_shared_update_step`: `mp.Value` for sharing the current update step count

### What's Missing for PER

1. `sample()` returns no indices — can't update priorities for specific episodes
2. No priority storage — all episodes have equal sampling probability
3. No feedback channel — main process can't send TD-errors back to subprocess
4. Loss function uses `F.mse_loss()` which averages uniformly — needs IS weighting

---

## 4. Algorithm: Episode PER

### Proportional Priority Sampling (Schaul et al., 2016, adapted to episodes)

**Two priority modes** (selected via `--per_mode`):

**Mode `td` (pure TD-error):**
```
p_i = (max_t |TD_error_t| over valid timesteps)^α + ε
```

**Mode `composite` (TD-error + success bonus):**
```
p_i = (max_t |TD_error_t| + λ * success_bonus_i)^α + ε
```
- `success_bonus_i = 1.0` if episode reached target (`terminals[-1] == True`), else `0.0`
- `λ` tunable via `--per_success_bonus` (default 1.0)
- **Why composite**: Pure TD-error upweights successes early (critic is surprised), but once the critic learns to predict them, their TD-error drops. With only 10-20% success rate, you still want them oversampled. The bonus ensures successes keep a priority floor throughout training.

**Sampling**: `P(i) = p_i / Σ p_j` — probability proportional to priority

**IS weights** (importance sampling correction):
```
w_i = (N * P(i))^(-β) / max_j(w_j)
```
- β anneals from 0.4 → 1.0 over training (full correction at convergence)
- Applied per-episode, broadcast across all timesteps in that episode

**New episodes**: assigned `max_priority` so they're guaranteed to be sampled at least once

**Core data structure**: SumTree — binary tree with O(log n) proportional sampling and priority update. With 10k episode capacity, tree has 20k nodes — trivially small.

---

## 5. Implementation Plan

### Files to Create

#### A. `eve_rl/eve_rl/replaybuffer/sumtree.py`

SumTree for O(log n) proportional sampling.

```python
class SumTree:
    """Binary tree: leaves = priorities, internal nodes = sums.
    Storage: flat numpy array, size 2*capacity - 1.
    Leaves at indices [capacity-1, 2*capacity-2].
    """
    def __init__(self, capacity: int)
    
    def update(self, leaf_idx: int, priority: float) -> None
        """Set leaf priority, propagate to root. O(log n)."""
    
    def sample(self, value: float) -> Tuple[int, float]
        """Walk tree to find leaf containing cumulative value.
        Returns (leaf_idx, priority). O(log n)."""
    
    @property
    def total(self) -> float   # root value, O(1)
    
    @property  
    def max(self) -> float     # max leaf, tracked incrementally, O(1)
```

#### B. `eve_rl/eve_rl/replaybuffer/prioritizedepisode.py`

Drop-in replacement for `VanillaEpisode`.

```python
class PrioritizedEpisode:
    def __init__(self, capacity, batch_size, device,
                 alpha=0.6, beta_start=0.4, beta_end=1.0,
                 beta_episodes=100_000, epsilon=1e-6,
                 mode="td", success_bonus=1.0)
    
    def push(self, episode) -> None
        # Same numpy storage as VanillaEpisode
        # Store success flag: episode.terminals[-1] (for composite mode)
        # Assign max_priority to new episode via tree.update()
        # Circular position increment
    
    def sample(self) -> PrioritizedBatch
        # Stratified sampling: divide [0, tree.total] into batch_size segments
        # Sample one uniform value per segment → tree.sample() → (leaf_idx, priority)
        # Retrieve episodes from buffer[leaf_idx]
        # Compute IS weights: w_i = (N * P(i))^(-beta) / max(w)
        # Pad sequences (same as VanillaEpisode)
        # Return PrioritizedBatch(obs, actions, rewards, terminals, mask, indices, is_weights)
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None
        # In 'td' mode:    p_i = (|td_error_i| + epsilon) ** alpha
        # In 'composite':  p_i = (|td_error_i| + λ * success_flag_i + epsilon) ** alpha
        # tree.update(idx, p_i) for each
        # Track _max_priority
    
    def anneal_beta(self) -> None
        # beta = min(beta_end, beta_start + n_anneals * (beta_end - beta_start) / beta_episodes)
        # n_anneals += 1
```

**PrioritizedBatch** — extends `Batch`:
```python
PrioritizedBatch = namedtuple('PrioritizedBatch',
    ['obs', 'actions', 'rewards', 'terminals', 'padding_mask',
     'indices', 'is_weights'])
```
- `indices`: `List[int]` — leaf indices for priority feedback
- `is_weights`: `torch.Tensor` shape `(batch_size, 1, 1)` — broadcast across timesteps

### Files to Modify

#### C. `eve_rl/eve_rl/replaybuffer/vanillashared.py`

**Add to `VanillaSharedBase.__init__()` (~line 31):**
```python
self._priority_queue = mp.Queue()
```

**Add method to `VanillaSharedBase`:**
```python
def update_priorities(self, indices, td_errors):
    self._priority_queue.put((indices, td_errors))
```

**Add to subprocess `loop()` (~line 185), after push_queue handling, before sleep:**
```python
while not self._priority_queue.empty():
    indices, td_errors = self._priority_queue.get_nowait()
    if hasattr(internal_replay_buffer, 'update_priorities'):
        internal_replay_buffer.update_priorities(indices, td_errors)
```

**Add beta annealing** in push handling section:
```python
if hasattr(internal_replay_buffer, 'anneal_beta'):
    internal_replay_buffer.anneal_beta()
```

**Add new subclass `PrioritizedEpisodeShared`:**
Same as `VanillaEpisodeShared` but creates `PrioritizedEpisode` in `_run_subprocess()`. Constructor accepts alpha/beta/epsilon/mode/success_bonus and passes them through.

**IMPORTANT**: The `_priority_queue` must be passed to the subprocess as an argument (same pattern as `_push_queue`, `_sample_queue`). The subprocess `loop()` must receive and check it.

#### D. `eve_rl/eve_rl/algo/sac.py`

**Changes to `update()` (~line 222):**

Detect PER batch (7 fields vs 5) and unpack IS weights + indices:
```python
def update(self, batch):
    if len(batch) == 7:
        (all_states, actions, rewards, dones, padding_mask, indices, is_weights) = batch
        is_weights = is_weights.to(dtype=torch.float32, device=self.device)
    else:
        (all_states, actions, rewards, dones, padding_mask) = batch
        indices = None
        is_weights = None
    
    # ... existing expected_q computation unchanged ...
    
    q1_loss = self._update_q1(actions, padding_mask, states, expected_q, is_weights)
    q2_loss = self._update_q2(actions, padding_mask, states, expected_q, is_weights)
    
    # ... existing policy/alpha updates unchanged ...
    
    if indices is not None:
        td_errors_np = self._per_episode_td_errors.cpu().numpy()
        return [q1_loss_np, q2_loss_np, policy_loss_np], indices, td_errors_np
    return [q1_loss_np, q2_loss_np, policy_loss_np]
```

**Changes to `_update_q1()` (~line 367):**

Extract per-episode TD-errors and apply IS-weighted loss:
```python
def _update_q1(self, actions, padding_mask, states, expected_q, is_weights=None):
    curr_q1 = self.model.q1(states, actions)
    if padding_mask is not None:
        curr_q1 *= padding_mask
    
    # Store per-episode TD-errors for PER priority update
    with torch.no_grad():
        td_errors = torch.abs(curr_q1 - expected_q.detach())
        if padding_mask is not None:
            td_errors *= padding_mask
        self._per_episode_td_errors = td_errors.max(dim=1).values.squeeze(-1)  # (batch,)
    
    if is_weights is not None:
        # IS-weighted MSE: weight each episode's loss by its importance weight
        per_sample_loss = (curr_q1 - expected_q.detach()) ** 2
        if padding_mask is not None:
            per_sample_loss *= padding_mask
        per_episode_loss = per_sample_loss.sum(dim=1)  # (batch, 1)
        q1_loss = (is_weights.squeeze(-1) * per_episode_loss).mean()
    else:
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
    
    # ... existing optimizer step unchanged ...
```

Same pattern for `_update_q2()`.

#### E. `eve_rl/eve_rl/agent/single.py`

**Changes to `update()` (~line 427):**

After `algo.update(batch)`, send priorities back if PER:
```python
result = self.algo.update(batch)

if isinstance(result, tuple) and len(result) == 3:
    losses, indices, td_errors = result
    self.replay_buffer.update_priorities(indices, td_errors)
    result = losses
results.append(result)
```

#### F. `training _scripts/util/agent.py`

**Changes to `BenchAgentSynchron.__init__()`:**

Add `use_per`, `per_mode`, `per_success_bonus` parameters:
```python
if use_per:
    replay_buffer = eve_rl.replaybuffer.PrioritizedEpisodeShared(
        replay_buffer_size, batch_size, device,
        alpha=0.6, beta_start=0.4,
        mode=per_mode,
        success_bonus=per_success_bonus,
    )
else:
    replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
        replay_buffer_size, batch_size, device,
    )
```

#### G. `training _scripts/DualDeviceNav_train.py`

Add CLI flags:
```python
parser.add_argument("--per", action="store_true",
    help="Enable Episode PER")
parser.add_argument("--per_mode", type=str, default="composite",
    choices=["td", "composite"],
    help="PER priority: 'td' = pure TD-error, 'composite' = TD + success bonus")
parser.add_argument("--per_success_bonus", type=float, default=1.0,
    help="Success bonus weight in composite mode (default: 1.0)")
```

Pass to agent constructor.

#### H. `eve_rl/eve_rl/replaybuffer/__init__.py`

Add exports:
```python
from .prioritizedepisode import PrioritizedEpisode
from .vanillashared import PrioritizedEpisodeShared
```

---

## 6. Data Flow

```
                    ┌──────────────────────────────────┐
                    │     Subprocess (loop)             │
                    │                                   │
push_queue ──────►  │  PrioritizedEpisode.push()        │
                    │   → store episode + success flag   │
                    │   → assign max_priority            │
                    │   → anneal_beta()                  │
                    │                                   │
                    │  PrioritizedEpisode.sample()       │  ──────► sample_queue
                    │   → SumTree proportional sampling  │         (PrioritizedBatch)
                    │   → compute IS weights             │
                    │   → pad sequences                  │
                    │                                   │
priority_queue ──►  │  .update_priorities(idx, td)       │
                    │   → recompute p_i with td + bonus  │
                    │   → SumTree.update()               │
                    └──────────────────────────────────┘

Main process (single.py update loop):
  batch = replay_buffer.sample()                        ← PrioritizedBatch with indices + is_weights
  losses, indices, td_errors = sac.update(batch)        ← IS-weighted Q-loss + per-episode TD-errors
  replay_buffer.update_priorities(indices, td_errors)   → send back to subprocess
```

---

## 7. Hyperparameters

| Param | Default | CLI flag | Description |
|-------|---------|----------|-------------|
| `alpha` | 0.6 | — | Prioritization exponent (0=uniform, 1=full priority) |
| `beta_start` | 0.4 | — | Initial IS correction strength |
| `beta_end` | 1.0 | — | Final IS correction (full debiasing) |
| `beta_episodes` | 100,000 | — | Episodes over which beta anneals |
| `epsilon` | 1e-6 | — | Small constant to prevent zero priority |
| `mode` | `"composite"` | `--per_mode` | `"td"` or `"composite"` |
| `success_bonus` | 1.0 | `--per_success_bonus` | λ weight in composite mode |

---

## 8. Backward Compatibility

- `--per` off (default): everything works exactly as before. No code path changes.
- `VanillaEpisode` is untouched — `PrioritizedEpisode` is a new class.
- SAC detects PER via batch length: `len(batch) == 7` → PER, `len(batch) == 5` → vanilla.
- `update_priorities()` on `VanillaSharedBase` is safe to call even with non-PER buffer (the priority queue just won't be drained if `hasattr` check fails).
- `single.py` checks `algo.update()` return type: `tuple` of length 3 = PER with td_errors, `list` = vanilla losses only.

---

## 9. Verification

1. **Unit test SumTree**: insert priorities, verify `total`, verify sampling distribution over 10k draws matches expected probabilities within statistical tolerance
2. **Unit test PrioritizedEpisode**: push 100 episodes, manually set priorities (some high, some low), verify high-priority episodes appear ~proportionally more in 1000 samples
3. **Integration test**: run short training with `--per --per_mode composite`, verify:
   - Priority queue doesn't grow unbounded (subprocess drains it)
   - IS weights are in reasonable range (0.1 to 10)
   - Beta anneals from 0.4 toward 1.0
   - No NaN/Inf in losses
4. **Training comparison**: `--per` vs baseline, identical hyperparameters, compare success rate curves
5. **Diagnostics**: log `per_max_priority`, `per_beta`, `per_is_weight_mean`, `per_is_weight_max` to tensorboard
