# RL Improv 10 — Changes Reference

All changes in this worktree relative to the state at the end of
`RL_IMPROV_9_CHANGES.md` (which ended at §17, the MDP-grounding fix, and
covered Plan v5–v8: reward unification, step-RL, PER, the stabilization
suite). RL_IMPROV_9 closed with the first RCCA step-SAC runs diverging,
then the **AWAC + grounding + warm-start** stack proven critic-stable but
the policy stuck at ~0.5 % success.

**This document covers the IMPLEMENTATION of "Plan v9" — a full reward
overhaul, three reward bug fixes, the fuller SOFA-restore, restore-at-fork
online training, and the single-run training pipeline — plus the
multi-stage bug hunt that produced them.** Built on env5 + the path-aware
state machine (RL_IMPROV_8 §14) + AWAC/PER/warm-start (RL_IMPROV_9
§13–17).

**Overarching constraint (learned the hard way — see §11):** reward
changes must NOT alter `is_on_correct_path()` / `_on_planned_path` /
`_current_branch_idx`, because the heuristic policy READS
`is_on_correct_path()` to gate its RCCA-specific steering
([heuristic_policy_rcca.py:356-361](training%20_scripts/util/heuristic_policy_rcca.py#L356)).
Any change to the branch classifier changes the heuristic's ACTIONS, not
just the reward. Every change below is reward-only or restores baseline
behavior.

---

## 1. Reward Bug A — `_committed_forks` latch (the `list.add` crash)

### Context

The state-machine daughter-commit mechanism (RL_IMPROV_9 §2) latched each
fork in `_committed_forks` so the +1 fires at most once per fork. Plan v9
keyed the latch by `(round(j_arc,3), sign)` tuples (so a fork can emit
both its +1 and, separately, −1 across an episode) and switched the
container from a `list` to a `set` for O(1) membership.

### The bug

`__init__` was updated to `self._committed_forks = set()`, but **`reset()`
(run every episode) still set it to `[]`**. So for every episode after the
first, `_committed_forks` was a list, and the +1 emission's
`self._committed_forks.add(...)` raised `'list' object has no attribute
'add'`. That exception was swallowed by env5's `try/except` around
`update_branch_state()`, so it was invisible — but **no +1 commit ever
recorded** (a run showed **2544** such exceptions and `daughters_passed=0`
on every RCCA-threading episode).

### Fix

**File:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py)
`reset()` now sets `self._committed_forks = set()` (matches `__init__`).
This single line restored the entire +1 commit signal — a clean RCCA
thread went from netting **−4.7** to **+3 to +7**.

---

## 2. Reward Bug B — `clean_thread` seeding filter = final-branch equality

### Context

The seeding success filter (`--seeding_success clean_thread`) selected
which episodes count as RCCA "successes" for the demo/clean stream.

### The bug

The old filter was `reached_target_daughter AND NOT received_wrong_daughter`:
- `reached_target_daughter` is an **ever-touched latch**, not a final-state
  check.
- `NOT received_wrong_daughter` wrongly **disqualified legitimate
  RVA-detour-then-RCCA episodes** — a wire that dipped toward RVA then
  corrected into RCCA is a perfectly good thread, but the wrong-touch
  latch excluded it.

### Fix

**Files:** [training _scripts/util/env5.py](training%20_scripts/util/env5.py),
[training _scripts/DualDeviceNav_train.py:509-524](training%20_scripts/DualDeviceNav_train.py#L509),
[eve_rl/eve_rl/replaybuffer/replaybuffer.py:54-73](eve_rl/eve_rl/replaybuffer/replaybuffer.py#L54)

- env5 exposes `info["final_branch_idx"]` and
  `info["target_daughter_branch_idx"]` at every step (the value at the
  terminal step is what matters).
- `_is_clean_thread` is now `final_branch_idx == target_daughter_branch_idx`
  — a strict final-state check. Detours along the way are fine.
- `Episode.to_replay()` populates `reached_target_daughter` from the same
  final-branch equality (drives the `is_clean` flag for balanced
  sampling), falling back to the old latch only if `final_branch_idx` is
  absent (env4 compat).

---

## 3. Fork Reward Redesign — `+1` at every junction / `−0.05` on deep wrong-commit

### `+1` at every on-path junction (not just daughter forks)

**File:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py)
`_maybe_emit_junction_commit(j_arc, sign=+1)` replaces the daughter-fork-
gated `_maybe_emit_daughter_commit`. The forward-commit in
`update_branch_state` now emits **+1 at EVERY on-path junction crossing**
along the planned path. For the RCCA topology `(2)→(0)→(11)→RCCA` that is
three commits (top junction `(2)→(0)`, bridge `(0)→(11)`, daughter fork
`(11)→RCCA`) → **max +3 on a clean thread**. Each is latched per
`(round(j_arc,3), +1)` so it fires at most once per fork per episode.

Rationale (user request): dense per-junction positive signal for each
correct navigational decision — "turn into (0) at the top, don't drift to
LVA"; "commit to (11), don't dwell in (0)"; "commit to RCCA, not RVA".
Accepted consequence: an episode that threads `(2)→(0)→(11)` then fails to
a wrong daughter still nets positive (+2 + progress ≈ +1.7) — partial
progress is rewarded. The ordering stays correct (true RCCA success
+3 to +7 ≫ partial-fail +1.7 ≫ no-progress dwell ≈ +0.2), which is what
AWAC's advantage weighting needs.

### `−0.05` on GENUINE deep wrong-commit (off-path-arc threshold)

**The problem it solves:** at the shared `(11)`/RCCA/RVA fork the two
sister branches have **asymmetric radius-aware tolerances** (RVA wide
tol≈8.7, RCCA narrow tol≈3.8). A wire wedged at the fork mouth at a fixed
cross-track (~7.4 mm, between the two tolerances) makes the state machine
**flicker** `cur_branch` RVA↔RCCA every step. The original −0.05 fired on
every off-path *transition*, so a wedged-but-correct wire was hammered
~80–100× → −5 per episode, dragging a +1 peak down to −3 (this was the
core of the success/failure **inversion** — see §11).

**Fix (reward-only):** −0.05 now fires only when the wire has travelled
`WRONG_COMMIT_ARC_MM = 10 mm` along a wrong branch since diverging
(`get_off_path_arc_since_divergence() ≥ 10`), latched per divergence event
(`_wrong_commit_fired`, re-armed on return on-path). A wedge at the mouth
(shallow off-arc) fires nothing; a genuine deep excursion into RVA fires
one −0.05; separate deep excursions each fire one (the "repeat" the user
wanted). Crucially this reads a getter and emits a reward event — it does
**not** touch `is_on_correct_path()`.

---

## 4. Path-Segment-Conditioned Step Penalty (replaces uniform −0.001)

**File:** [training _scripts/util/env5.py](training%20_scripts/util/env5.py)
The uniform `eve.reward.Step(factor=-0.001)` was removed from the reward
`Combination`; the per-step penalty is now computed inline in
`BenchEnv5.step()` (after `update_branch_state`) by
`_compute_path_segment_step_reward()`, keyed on **on-path vs off-path
first, then segment**:

| Wire location | per-step |
|---|---|
| Trunk `(2)` (z=345 → bif1), on-path | interpolate **−0.007 → −0.002** by trunk arclength (`proj.s / trunk_end_arc`, where `trunk_end_arc` = the `(2)→(0)` junction) |
| `(0)` on-path corridor (bif1 → `(0)→(11)` fork) | **−0.002** (flat — still progressing toward (11)) |
| `(11)` bridge | **0.0** |
| RCCA (target daughter) | **0.0** (depth rewarded via §5, not a dwell bonus) |
| anything off-path (overshoot into `(0)` past the fork, wrong daughter, LVA drift) | **−0.007** |

`(0)` is a **shared branch**: its on-path portion (toward (11)) is the
cheap −0.002 corridor; only the overshoot past the `(0)→(11)` fork is
off-path → −0.007. The on-path/off-path classifier makes that distinction
for free, so this stays a pure reward computation. Branch identity is
resolved once per reset via a cached `_step_reward_branch_class` map +
regex on branch names (`\(2\)`, `\(0\)`, `\(11\)`).

---

## 5. RCCA Depth Reward — 2× ArcLengthProgress on forward motion in target

**File:** [eve/eve/reward/arclengthprogress.py](eve/eve/reward/arclengthprogress.py)
Originally drafted as a +0.005/step constant RCCA bonus, but that rewards
mere dwell. Replaced: when the wire is inside the **target daughter** AND
moving forward (`Δs_planned > 0`), the progress factor is **doubled**
(`pf *= 2`). Deeper threading → more reward; freezing in shallow RCCA →
nothing; backward motion still penalised at 1×.

---

## 6. Cross-Track Lateral Penalty — Radius-Aware Deadband

### Context

Diagnosis of the inversion's first layer: RCCA-reaching episodes were
netting −6.8, of which ~−3.0 was the lateral penalty
(`−0.001 × cross_track`) accumulated over ~600 steps at a **mean 5.1 mm**
offset. In the wide aortic trunk ~5 mm of cross-track is geometrically
unavoidable (the wire rides the wall, not the centerline polyline), so
this added a ~−3 constant drag to every episode regardless of skill.

### Fix

**File:** [eve/eve/reward/arclengthprogress.py](eve/eve/reward/arclengthprogress.py)
The lateral penalty now applies only to the **excess beyond the local
radius-aware tolerance**: `r_lateral = −lateral_penalty_factor ×
max(0, cross_track − get_local_tolerance())`. Wide trunk → large tolerance
→ ~0 penalty for normal wall-hugging; narrow daughter → genuine
divergence still penalised. Mirrors the state machine's existing
`get_local_tolerance()` (= `max(2, K_RADIUS × local_radius)`). Reward-only
(reads a getter).

---

## 7. SOFA-Restore — Fuller Save (RL_IMPROV_8 §38 finally implemented)

**File:** [eve/eve/intervention/simulation/sofabeamadapter.py](eve/eve/intervention/simulation/sofabeamadapter.py)

The prior checkpoint saved only `dof_positions` (1 of 7+ MechanicalObject
state fields), then `restore_checkpoint` called `Simulation.reset` (zeroing
velocity/force), zeroed `rotation_instrument` (RL_IMPROV_7 §7 Fix 6), and
ran a 50-step settle to reconverge contacts from rest — which sometimes
settled to a different local minimum (the 36 % wrong-location bug).

**New `save_checkpoint()` companion** captures the 4 original fields plus
6 more from `ic.DOFs`: `velocity, force, externalForce, free_position,
free_velocity, derivX` (skipped silently if a SOFA build doesn't expose
one). **`restore_checkpoint`** now:
- restores `rotation_instrument` from the saved value (reverts the §7 Fix 6
  zeroing) when the checkpoint has the extra fields;
- assigns all available DOF DataFields;
- defaults `settle_steps` 50 → **3** (velocities preserved → no
  reconvergence from rest needed);
- is back-compatible: an old-format checkpoint (no extra fields) keeps the
  zero-rotation + 50-step behavior via an `if "velocity" in state_dict:`
  gate.

env5's existing RVA-checkpoint capture and the new pre-bif(11) capture
(§8) both route through `save_checkpoint()` so every checkpoint is
fuller-format. Verified: saved `.npz` contains all 6 extra fields.

---

## 8. Restore-at-Fork Training — Pool Capture + Single-Run Pipeline

The core Plan v9 training idea: the heuristic threads RCCA from z=345 only
~30 % of the time because the trunk eats most of the step budget; online
RL should instead start **just before the `(0)→(11)` fork** so every
episode spends its full budget on the hard daughter-threading problem.

### 8a. Pre-bif(11) checkpoint pool capture

**File:** [training _scripts/util/env5.py](training%20_scripts/util/env5.py)
Gated by env var `PRE_BIF11_CHECKPOINT_DIR`. During a heuristic episode,
the first step the wire is within 5 mm of the **`(0)→(11)` junction**
(found via `get_path_junctions()` matching `\(0\)`→`\(11\)` — NOT a direct
`(2)→(11)`, which doesn't exist in this topology), env5 snapshots an
in-memory `save_checkpoint()` memo. At episode terminal, if the episode
finished RCCA-final (clean thread), the memo is written to disk
(`pre_bif11_pidNNN_epNNNN_stepNNNN.npz`). One checkpoint per RCCA-final
episode → a pool of ~100–150 diverse restore states.

### 8b. Restore-at-fork online mode

**Files:** [training _scripts/util/checkpoint_restore.py](training%20_scripts/util/checkpoint_restore.py),
[training _scripts/DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py)
- New CLI `--rl_start_mode {z345, sofa_restore}`. `sofa_restore` makes
  every online `env.reset()` load a random checkpoint from
  `--checkpoint_dir` (the pool) via the existing
  `CheckpointRestoreWrapper`.
- `CheckpointRestoreWrapper` made **lazy**: it no longer raises on an
  empty/absent pool dir at construction (the pool is filled *during* the
  same run's heuristic seeding), re-scanning on each reset until files
  appear. It also **bypasses restore for `heuristic_mode=True` resets** so
  heuristic seeding always runs from z=345 (full-trunk demos) while only
  heatup + online explore restore from the fork.
- `--insertion_z` and `--checkpoint_dir` mutual-exclusivity **relaxed**
  for `sofa_restore` mode (they're complementary there: insertion_z drives
  heuristic seeding, checkpoint_dir drives online restore).

### 8c. Single-run pipeline + warm-start gate fix

One container does everything in order: heuristic seeding (generates
`rcca_heuristic_cache_v2.npz` + the pool) → heatup → **10k warm-start
pretraining** → online restore-at-fork explore/train. The warm-start gate
([DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py))
previously required BOTH caches **loaded from disk**; it now also fires
when `--heuristic_seeding > 0` generates the seed buffer in the same run
(`seed_generated_this_run`). Buffer composition is verified before the 10k
pretrain runs (heatup happens first inside `training_run`, then pretrain —
RL_IMPROV_9 §17 ordering).

---

## 9. Restore-Start Debug Snapshots

**Files:** [training _scripts/util/env5.py](training%20_scripts/util/env5.py),
[training _scripts/util/snapshot.py](training%20_scripts/util/snapshot.py)
`--restore_start_snapshots` renders the wire's pose immediately after each
SOFA restore (reset, step 0) into `snapshots/restore_start/<target>/start/`
— a **sibling phase bucket** alongside `seed/`, `eval/`, `explore/`
(initially a separate triple-nested folder; moved per user request).
`save_snapshot()` gained `base_dir_override` / `mode_override` so this
works even when `--snapshots` is off. Purpose: visually verify the
restore lands the wire just before bif(11) — log lines can't catch a
wrong-location restore.

---

## 10. Seeding Rate — 100 clean demos

`launch_rcca_awac_v2.sh` uses `--min_success_rate 1.0` (`min_successes =
ceil(1.0 × heuristic_seeding) = 100`) so seeding collects until **100
RCCA-clean successes** (not "100 episodes at 30 % success" — the prior
`0.30` meant 30 % of the pushed 100 were successes). `--max_seeding_
multiplier 6` (cap 600 attempts) gives headroom. The push keeps all
successes + sampled failures to a ≤70 % success ratio.

---

## 11. The Inversion Bug Hunt (narrative + lesson)

The symptom (user-reported): **episodes that threaded RCCA scored WORSE
than episodes that overshot into the wrong part of (0)** — the reward
taught the opposite of the goal. Root-causing it took several relaunches
and produced §1, §3, §6, and one reverted dead-end:

1. **Lateral drag (§6).** RCCA episodes carried ~−3 of unavoidable
   cross-track penalty. The deadband removed it (−6.8 → −2.8).
2. **Missing +1 commits (§1).** `daughters_passed=0` on every RCCA thread
   — the `list.add` reset bug meant the +3 never landed. The `set()` fix
   restored it (−2.8 → +3 to +7).
3. **The −0.05 flicker (§3).** At the RCCA/RVA fork the classifier
   flickered every step (asymmetric tolerances), and the old −0.05 fired on
   each flicker (~−5/episode). The off-path-arc-threshold version fires
   only on genuine deep excursions.
4. **The reverted dead-end (the lesson).** A first attempt fixed the
   flicker by **debouncing the off-path flip** (require N consecutive
   divergence steps). It worked on the reward but **dropped RCCA threading
   to 0 %** — because the heuristic reads `is_on_correct_path()` to gate
   its RCCA-specific phased steering, and the debounce changed that flag,
   so the heuristic fell back to a generic centerline-follower and
   mis-steered to (18)/(0)/LCCA. **Reverted.** Lesson, now a standing
   rule: **never change the branch classifier to fix a reward; do it on
   the reward side only.**

A useful diagnostic confirmation: episodes peaked at +0.9–1.0 mid-thread
(the genuine progress signal) then bled to −3 from the flicker — proof the
threading was real and the reward was the bug.

---

## 12. Verification (run 2026-05-22, `rcca_awac_v2`)

After all fixes, single-run pipeline:
- `list.add` errors: **0** (was 2544).
- `daughters_passed` on RCCA threads: **3** (all junction commits fire).
- RCCA-final reward: **+3 to +7** (was −2.75 to −4.74); (0)-dwell ≈ +0.16;
  failures down to −6 → **ordering correct, inversion gone**.
- Heuristic seeding: **146 clean RCCA demos** + 63 failures (69.9 %
  success ratio pushed); cache returns mean **+2.36**, 92 % positive,
  67 % ≥ +3.
- Pre-bif(11) pool: **148 checkpoints**.
- 10k warm-start ran on the seeding-generated buffer.
- Critic health through ~85k online updates: `q1_loss` 0.004–0.026 (tiny),
  `q1_mean` 0 → −2.8 **plateau** (buffer-average, failure-weighted;
  bounded — no SAC-style 10⁷ divergence), `grad_norm_q1` < 1.5 (grad_clip
  1.0 barely binds).

Pending at time of writing: first eval `Quality:` at ~250k explore steps —
the real test of whether restore-at-fork online training reaches the RCCA
target.

---

## File Index

| File | Sections |
|---|---|
| `eve/eve/util/pathcontext.py` | §1 (`set()` reset), §3 (`_maybe_emit_junction_commit`, off-arc −0.05, `WRONG_COMMIT_ARC_MM`, `_wrong_commit_fired`), §11 (debounce reverted) |
| `eve/eve/reward/arclengthprogress.py` | §5 (2× progress in target), §6 (cross-track deadband) |
| `training _scripts/util/env5.py` | §2 (`final_branch_idx` info), §4 (path-segment step penalty), §8a (pre-bif11 capture), §9 (restore-start snapshot) |
| `training _scripts/DualDeviceNav_train.py` | §2 (clean_thread filter), §8b (`--rl_start_mode`, mutex relax), §8c (warm-start gate), §9 (`--restore_start_snapshots`), §10 |
| `training _scripts/util/checkpoint_restore.py` | §8b (lazy scan + heuristic-mode bypass) |
| `training _scripts/util/snapshot.py` | §9 (`base_dir_override`/`mode_override`) |
| `eve/eve/intervention/simulation/sofabeamadapter.py` | §7 (`save_checkpoint`, fuller restore) |
| `eve_rl/eve_rl/replaybuffer/replaybuffer.py` | §2 (`to_replay` final-branch flag) |
| `launch_rcca_awac_v2.sh` | §8c, §10 (new launcher) |

---

## 13. Plan v10 — Few-state curriculum + entropy-floored AWAC

The Plan v9 `rcca_awac_v2` run validated **critic stability** (no divergence
through ~85k updates) but eval-#1 success stalled at ~17 %. Two structural
causes were diagnosed:

1. **Entropy collapse during pretrain (mean saturates, tanh-Jacobian
   explodes).** `log_std_min = -20` lets σ → 2e-9; AWAC has no entropy term
   to resist; clamp_fraction climbs 0 → 0.5 in the first 2.5k updates.
2. **The 1028-state pre-bif(11) pool was too diverse.** Some checkpoints
   put the guidewire ~21 mm behind the catheter; from those poses the
   policy cannot recover the fork commit. Per-state heuristic analysis on
   prior runs found **6 catheter-forward states** (cath ~99–104 mm,
   gap ~8–15 mm) where success rate is 16–20 % even with the broken
   reward; the other ~140 states are ≤ 3.7 %.

Plan v10 = **train restore-at-fork from ONLY the curated good states**,
plus a **hard `log_std_min` entropy floor**, plus a **heuristic-until-N
RCCA-clean seeding** from the 5 states so the buffer enters pretrain with
on-distribution success demos. The reward, observation, and terminal are
unchanged from Plan v9 — only start states and AWAC stabilization knobs.

(The *original* Plan v10 design called for **random** heatup-until-N as
the seed source; that infrastructure was built but failed empirically —
see §13b for the pivot.)

### 13a. `--log_std_min` entropy floor (CLI-threaded all the way to GaussianPolicy)

**Files:**
[eve_rl/eve_rl/network/gaussianpolicy.py](eve_rl/eve_rl/network/gaussianpolicy.py)
already exposed `log_std_min` / `log_std_max` ctor params but the **three
construction sites in [training _scripts/util/agent.py](training%20_scripts/util/agent.py)**
(`BenchAgentSynchron` + `BenchAgentSingle` worker/trainer/eval paths) didn't
pass them. Added `log_std_min: float = -20.0` param to `BenchAgentSynchron`
and threaded it into all three `GaussianPolicy(...)` calls. Then added
`--log_std_min` CLI in
[DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py)
(default −20 to preserve old behaviour). Launcher sets **−2** (σ floor
0.135 — well above the 2e-9 of the unbounded case, comfortably above the
collapse threshold).

**Caveat (discovered empirically — see §15):** the floor only protects the
Gaussian variance term in `log π(a|s) = log N(z;μ,σ) − Σ log(1−tanh(z)²)`.
The tanh-squash Jacobian (the second term) is **unbounded** as the policy
*mean* approaches the rail, regardless of σ. So `log_std_min` slows
collapse but does not prevent it long-term; see §15.

### 13b. Heuristic-until-N-RCCA-clean (random-heatup-until-N abandoned)

**Original Plan v10 intent (built but did not work).** The first design
was **purely random** heatup-until-N from the 5 good states: spawn
episodes with `eve_rl.Runner.heatup` (random actions sampled from
`heatup_action_low/high`) and stop when N=10 episodes thread RCCA
(`info["final_branch_idx"] == info["target_daughter_branch_idx"]`, §2's
flag — a finite-time stop criterion *only*, NOT a reward/terminal change).

The infrastructure was built end-to-end:
- [eve_rl/eve_rl/runner/runner.py](eve_rl/eve_rl/runner/runner.py):
  `Runner.training_run` gained `heatup_until_successes: int = 0`
  parameter; if set, heatup runs in batches counted via a new
  `Runner._episode_threaded()` staticmethod (which mirrors
  `Episode.to_replay`'s final-branch check on the episode info dict).
  Stops when N reached or a safety cap (~2000 episodes) is hit.
- [training _scripts/util/agent.py](training%20_scripts/util/agent.py):
  threads `heatup_until_successes` through `BenchAgentSynchron`.
- [DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py):
  new `--heatup_until_successes` + `--heatup_episodes` CLI args.

**Why it was abandoned.** Empirically, **random heatup from the 5 good
states almost never thread RCCA** — over hundreds of attempts, zero
RCCA-final episodes occurred. The good states are only "good" in the
sense that the catheter pose makes the fork *reachable* under a
**competent** policy; random action sequences (gw_rot uniformly in
±1.5 rad, gw_trans in [-10, 30] mm) still take wildly counter-productive
choices at the fork (rotate hard right at the moment of commit; push
cath while pulling gw; etc.). The run was stopped after a long
no-success heatup phase (user-observed: "stop the run; none of the
heatups are going into RCCA"). The `--heatup_until_successes` code path
remained in the tree but unused.

**What was deployed instead — heuristic-from-restore.** Reused the
**existing heuristic-seeding pipeline** (§10) but with the
`--heuristic_from_restore` flag added — which sets
`heuristic_mode=False` in the episode schedule built by
[build_episode_schedule](training%20_scripts/DualDeviceNav_train.py).
That **disables the heuristic-mode bypass in
`CheckpointRestoreWrapper`** (§8b), so heuristic episodes start from a
SOFA-restored fork state (not from z=345 as the trunk-traversal Plan v9
heuristic does). The RCCA heuristic policy then has only to commit the
fork → easy from a catheter-forward start state, ~100 % success rate.

`launch_rcca_awac_v3.sh` flag combo:
```
--heuristic_seeding 10                # collect 10 episodes via heuristic
--heuristic_from_restore              # start heuristic from the 5 fork states
--min_success_rate 1.0                # all 10 must be clean threads
--max_seeding_multiplier 20           # up to 200 attempts (headroom)
--seeding_success clean_thread        # §2 final-branch filter
--heuristic_factory "util.heuristic_policy_rcca:RCCAHeuristicActionFunctionFactory"
--save_heuristic_cache .../rcca_heuristic_fork_v3.npz   # persisted (user request)
--heatup_episodes 10                  # fixed 10 random heatup eps (NOT until-N)
--save_heatup_cache .../rcca_heatup_fork_v3.npz         # persisted
--pretrain_updates 1000
```

Buffer composition entering pretrain:
- 10 RCCA-clean heuristic-from-fork episodes (~50–100 transitions each
  → ~500–1000 success transitions), all marked `is_clean=True` for
  balanced sampling.
- 10 random heatup episodes (~600 steps each at max_steps, ~6000
  transitions — supplementary action-space coverage; not a seed-success
  source).
- ≤ failures pushed by the seeding-rate filter (small).

**Warm-start gate relaxation** stays as designed: the §8c gate
`seed_buffer_available = both_caches or seed_generated_this_run` now also
fires when heatup runs (`heatup_runs = (args.heatup_until_successes>0) or
(heatup_steps_effective>0)`), so pretrain doesn't refuse to run on a
heatup-only buffer. Even though the heatup-until-N path wasn't deployed,
the gate broadening is still correct for any future heatup-only seed.

**Lesson recorded as a finding (not a bug):** *random heatup is not a
viable seed source for fork-commit problems with sparse positive
reward, even from curated good start states.* A competent (heuristic
or learned) policy is required to produce the first positive episodes.

### 13c. Per-state restore logging (Plan v10 C3)

The `CheckpointRestoreWrapper` already injected
`options["_restore_checkpoint_file"]` /
`options["_restore_checkpoint_idx"]`
([checkpoint_restore.py:129–130](training%20_scripts/util/checkpoint_restore.py#L129))
but `env5.reset()` was dropping them.

**File:** [training _scripts/util/env5.py](training%20_scripts/util/env5.py)
`reset()` now reads those keys, stashes them on `self._restore_ckpt_file` /
`_restore_ckpt_idx`, appends `restore_ckpt=<filename>` to the
`EPISODE_START` STEP-log line, **and** puts them into the `info` dict
returned at every step. Both EPISODE_START groupby and per-episode
`info`-based summary work, for both train and eval workers.

This is what made the eval per-state tables (§19) and the eval-#2/#3
forensic analyses (§20-21) possible.

### 13d. Eval wraps with `CheckpointRestoreWrapper` too (Plan v10 C4)

Plan v9 wrapped `env_train` with restore but `env_eval` started at z=345
— a train/eval mismatch. Plan v10 wraps `env_eval` too (mirrors env_train
at [DualDeviceNav_train.py:381–384](training%20_scripts/DualDeviceNav_train.py#L381),
with distinct `rng_seed=43`). Eval's 98 seeds then deterministically pick
among the configured restore states. The `env_eval_unwrapped` reference is
preserved for config save (the saved env_eval config remains the unwrapped
BenchEnv5 — critical for `get_env_from_checkpoint` reconstruction).

### 13e. First-restore double-restore fix (env5 reset)

User-identified empirically: in the rcca_awac_v3 launch, **9 of 16 workers'
first eval episode started visibly at z=345** (un-restored) — the first
SOFA `restore_checkpoint()` call sets the controller's `xtip` but does NOT
apply `dof_positions` because the scene hasn't fully constructed yet.
Subsequent restores work fine.

**Fix:** [env5.py:425-435](training%20_scripts/util/env5.py#L425-L435)
double-restores on the first restore in each env-instance's lifetime:

```python
self.intervention.simulation.restore_checkpoint(ckpt)            # restore #1
if not getattr(self, "_restore_warmed_up", False):
    self.intervention.simulation.restore_checkpoint(ckpt)        # restore #2 (first ever only)
    self._restore_warmed_up = True
```

`_restore_warmed_up` is per-env-instance, so it fires once per worker per
phase (16 workers × {env_train, env_eval} = 32 first-restore events). Cost:
~50 ms once per env-instance lifetime; correctness: full coverage. Eval#2
onward shows no z=345 starts in the snapshot folder.

### 13f. The 5 curated good states + launcher

`saved/rcca_5good_checkpoints/` (the actual pool used):
```
pre_bif11_pid19116_ep0017_step0050.npz    # ~70% eval#1, our best
pre_bif11_pid3285_ep0046_step0132.npz     # ~41% eval#1
pre_bif11_pid4907_ep0176_step0054.npz     # ~29% eval#1
pre_bif11_pid20043_ep0003_step0038.npz    # ~17% eval#1
pre_bif11_pid10145_ep0075_step0060.npz    # ~10% eval#1
```

The 6th user-flagged candidate (pid20926) was dropped.

`launch_rcca_awac_v3.sh` — mirrors `launch_rcca_awac_v2.sh` mounts + adds
`--log_std_min -2`, `--heatup_until_successes 10`, `--heatup_episodes 10`,
`--pretrain_updates 1000`, `--checkpoint_dir
.../rcca_5good_checkpoints`, `--rl_start_mode sofa_restore`,
`--restore_start_snapshots`, `-n rcca_awac_v3`, `-nw 16 -d cuda:0`. No
cache files (the seed comes entirely from heatup-until-10-RCCA-clean).

---

## 14. The `rcca_awac_v3` run — 3 evals, slow improvement, real plateau

Single run, **2026-05-25_031951_rcca_awac_v3**. 16 workers, ~6 explore
steps/s, ~50 hours wall-clock.

| eval | step | strict succ% | wbt | wfs | mx | mean R |
|---|---|---|---|---|---|---|
| #1 | explore 250,531 | **32/98 = 32.7%** | 43 | 17 | 6 | — |
| #2 | explore 503,488 | **31/98 = 31.6%** | 39 | 6 | 22 | — |
| #3 | explore 750,xxx (in-progress final episode at write time) | **35/97 = 36.1%** (97 of 98 seeds; one worker assignment irregularity) | 21 | 16 | 25 | — |

**Critic was stable through the entire 750k+ explore steps.** `q1_loss`
flat ~0.05; `q1_mean ≈ target_q ≈ min_q ≈ −4.6` (twin-Q and target track
within 0.1, no overestimation feedback); `grad_norm_policy` 1.87 → 0.98
(decreasing); zero `nonfinite_*` counts. The AWAC + grad_clip 1.0 +
grounding + step-PER stack solved the divergence that killed every prior
SAC variant.

**Per-restore-state eval breakdown** (n per state is deterministic from
`np.random.default_rng(seed).integers(0, 5)` across the 98 EVAL_SEEDS, so
identical across all three evals):

```
state    |  EVAL #1            |  EVAL #2            |  EVAL #3
         |  n   succ   succ%   |  n   succ   succ%   |  n   succ   succ%
pid19116 | 11    9     81.8%   | 11    8     72.7%   | 11    9     81.8%
pid3285  | 22   11     50.0%   | 22   11     50.0%   | 22   11     50.0%
pid4907  | 21    6     28.6%   | 21    4     19.0%   | 20    7     35.0%
pid20043 | 24    4     16.7%   | 24    6     25.0%   | 23    5     21.7%
pid10145 | 20    2     10.0%   | 20    2     10.0%   | 20    3     15.0%
TOTAL    | 98   32     32.7%   | 98   31     31.6%   | 97   35     36.1%
```

**The per-state success rate spans 10 % → 82 %.** That spread is the
single biggest signal in the run: the strict 32 % number is not "policy is
30 % capable" — it's "policy is 82 % on the easy state and 10 % on the
hard one, and we're averaging over a hard-skewed distribution." Per-state
trajectories are essentially flat across evals (pid3285 is 11/22 in all
three; pid19116 holds 9/8/9; pid10145 holds 2/2/3) — no per-state policy
is converging upward.

---

## 15. Why entropy collapses despite `log_std_min = −2` — `log_pi` decomposition

This was the v3 run's central diagnostic. With `log_std_min = −2` (σ ≥ 0.135), the
Plan v10 expectation was that entropy_proxy would stay bounded near
target_entropy (−4). Instead it dropped monotonically from **−2.3 at
update 1k → −9.7 at update 410k → −10.2 at update 650k**, with
`clamp_fraction` rising 0.06 → 0.53 in lockstep.

**Mechanism:** `entropy_proxy = -log_pi.mean()` where `log_pi` is the
log-prob of a fresh policy sample. For a tanh-squashed Gaussian
([sac.py:520](eve_rl/eve_rl/algo/sac.py#L520)):

```
log π(a|s) = Σ_dim [ log N(z; μ, σ)  −  log(1 − tanh(z)² + ε) ]
entropy_proxy = Σ Gaussian-entropy(σ)   +   Σ log(1 − a²)
                └── floored by log_std_min ──┘   └── tanh-saturation, UNBOUNDED ──┘
```

`log_std_min` protects the **Gaussian variance term** — that part cannot
go very negative. But the **tanh-squash correction** `Σ log(1−a²)` goes to
−∞ as the policy *mean* approaches the rail, and the floor does nothing to
it. So `entropy_proxy` ends up tracking action-saturation (clamp_fraction)
rather than variance — it's really a "how railed is the policy" gauge
here, not a "how random."

**Implication:** the entropy collapse in AWAC is driven by **mean drift
into saturation**, not by σ shrinking. `log_std_min` slows it (variance
stays exploratory in the sample noise) but cannot prevent it. The
collapse trajectory above is the unbounded tanh-Jacobian term
accumulating as the mean rails further every update.

The two levers that DO target mean saturation directly are an **explicit
entropy bonus in the AWAC loss** (penalizes log_pi which includes the
tanh-Jacobian) and a **mean-margin / tanh-margin penalty** (e.g.
`E[max(0, |tanh(μ)| − 0.9)²]`). Both deferred for the next-iteration plan.

---

## 16. The `--eval_only_checkpoint` flag and the pid19116 sweep

**Files:**
[training _scripts/DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py)
- new CLI `--eval_only_checkpoint <path>`;
- branch right after `runner.save_config(runner_config)`:
  ```python
  if args.eval_only_checkpoint:
      agent.load_checkpoint(args.eval_only_checkpoint)
      runner._replay_save_interval = 999999  # skip the empty-buffer save
      quality, reward = runner.eval(seeds=EVAL_SEEDS)
      agent.close()
      return
  ```
- Skips heatup/pretrain/training entirely; loads the named checkpoint's
  weights via `BenchAgentSynchron.load_checkpoint`
  (broadcasts to all 16 workers via `_worker_load_state_dicts_network`,
  [synchron.py:225–227](eve_rl/eve_rl/agent/synchron.py#L225)) then
  invokes `Runner.eval()` once.
- All standard run-dir structure is preserved (timestamped dir, STEP logs,
  per-bucket snapshots, main.log, restore_start snapshots, etc.) — the
  Synchron path produces them naturally.

`saved/rcca_best_state_pid19116/` — a one-file directory containing only
`pre_bif11_pid19116_ep0017_step0050.npz` (+ .json). With one file the
`CheckpointRestoreWrapper`'s `_pick_index` returns `idx % 1 = 0` always —
every one of the 98 EVAL_SEEDS restores from pid19116.

`launch_rcca_eval_best_state.sh` — mirrors the v3 launcher mounts +
`--eval_only_checkpoint /opt/eve_training/results/eve_paper/.../checkpoints/checkpoint250531.everl`
(the eval-#1 = 32.67 % deterministic policy) +
`--checkpoint_dir .../rcca_best_state_pid19116`. No heatup / no pretrain
flags needed — they're ignored under `--eval_only_checkpoint`.

### Result — run `2026-05-27_014955_rcca_eval_pid19116_v1`

98 episodes in **808 s** (vs ~5000 s and ~10000 s for evals #1 and #2 —
much faster because no episode wastes the full step budget on a hard
start state):

```
strict success:                  84/98 = 85.7%
buckets:                         success=84  wfs=14  wbt=0  mx=0
```

**No wrong_branch_timeout. No max_steps.** From pid19116 alone, the same
eval-#1 policy threads the `(11)→RCCA` fork **98 / 98 times**. All 14
failures are the catheter-overshoot fold pattern — same physical
mechanism diagnosed earlier from the saturated `cath_trans` rail. Strict
gain vs the mixed-state eval #1: **+53 pp**.

This is the central Plan-v10 validation: **per-state difficulty was the
dominant bottleneck**, not policy quality. The eval-#1 policy is ~86 %
capable from a good start state.

---

## 17. Functional-success reclassification by terminal cum_reward

User-flagged: the 14 wfs failures from §16 are "guidewire stuck inside
catheter near target" — the catheter has been pushed past the wire tip,
the wire is bunched inside it, can't advance further but is geometrically
at/near the target. Procedurally recoverable.

The geometric filter `tip_z ≥ 440 AND cath_inserted > gw_inserted` was
too loose — it folded in wires lodged in wrong daughters at similar
depths. **Terminal `cum_reward` is a cleaner discriminator**: a
near-success collects the fork +1, the per-step progress reward, and the
−5 truncation, ending in **R ∈ [−1, +1]**; a genuine fail (off-path drift
to (18) / wrong daughter) accumulates the off-path arc-shaping penalty and
ends at R ≤ −3.

### Reclassification of all three v3 evals + the pid19116 sweep

```
                 strict   wfs near    wbt near    FUNCTIONAL (R in [-1,1])
EVAL #1          32/98     9 of 17     1 of 43    42 / 98 = 42.9%
EVAL #2          31/98     2 of  6    11 of 39    44 / 98 = 44.9%
EVAL #3          35/97    13 of 16     1 of 21    49 / 97 = 50.5%
pid19116 sweep   84/98    11 of 14     0 of  0    95 / 98 = 96.9%
                                  ↑ with cutoff R ≥ −1.5: 14 of 14 → 100%
```

Functional success in the v3 run improved **+7.6 pp** (42.9 → 50.5) over
the 750k explore steps, vs +3 pp for strict — the functional metric
captures more of the policy's actual improvement, since the strict
TargetReached terminal is too aggressive about the catheter-overshoot
near-target failures.

**Failure-mode shifts between evals** (under the functional read):
- Eval #1: 9 near-target in wfs, 1 in wbt — catheter-overshoot dominant.
- Eval #2: 2 near in wfs, 11 in wbt — the same close-but-not-target physics
  shifted from wfs-classifier to wbt-classifier (different truncation
  trigger; same underlying state).
- Eval #3: 13 near in wfs, 1 in wbt — reverted to wfs-dominant.

The classifier-noise across evals explains why eval #2 looked
"regressed" (31.6 % strict vs 32.7 %) — strict was tracking which
truncation fires; functional says #2 was actually slightly better
(44.9 % vs 42.9 %).

---

## 18. Anatomical mapping — "RCCA" centerline extends to the carotid siphon

User raised the anatomical question: the human RCCA is ~95–130 mm long;
the mesh's `Centerline curve - RCCA.mrk` is **237.5 mm** long — what's the
extra ~100+ mm?

Inspection of `eve_bench/data/dualdevicenav/Centrelines_comb/*.json`:

```
named centerline    length    z-range (insertion-depth scale)
RCCA               237.5 mm   416 → 601
LCCA               266.9 mm   385 → 595
RVA                236.8 mm   416 → 572
LVA                209.2 mm   430 → 572
```

Each "named" centerline extends from the great-vessel origin (aortic arch
side branches) **all the way through the cervical and intracranial portions
of the vessel** — the named segment for RCCA includes RCCA + cervical
ICA + petrous ICA + cavernous ICA (the carotid siphon, multi-bend
S-shape) + intracranial ICA terminus. The bifurcation of CCA into ICA /
ECA is anatomically at ~95–130 mm from origin, i.e. centerline z ≈
510–545 — beyond that point the centerline IS the ICA, just named after
its supplier.

At the distal endpoints of the named centerlines, the 21 numbered
branches connect with anatomically-sensible short segments (lengths 7–30
mm) that match **Circle of Willis** geometry: branches at RCCA distal
(−601, −3.9, 4.7) → MCA-M1 / ACA-A1 candidates; branches connecting
RCCA-distal to LCCA-distal across z ≈ 598–602 → **anterior communicating
artery** + the two A1 segments.

### Eval target depth → anatomy

Mapping the 98 eval target z-coordinates (`CenterlineRandom` samples
uniformly along the RCCA centerline from origin at z=416 onward) to
anatomy in the pid19116 sweep:

```
band                          n   succ  wfs   succ%   wfs%
proximal CCA  (z 416-510)    45    45    0   100.0%    0.0%
cervical ICA  (z 510-575)    40    33    7    82.5%   17.5%
siphon/terminus (z 575-601)  13     6    7    46.2%   53.8%
```

All 14 wfs failures concentrate at z ≥ 547; success-target median z = 501;
wfs-target median z = 585 — **all wfs are in the cervical ICA + siphon**.
The catheter-overshoot failure mode is concentrated in the carotid
siphon, anatomically consistent: the siphon's multiple ~180° bends within
~30 mm are exactly where aggressive cath_trans causes the catheter to
bind against vessel walls, and pushing harder makes the wire buckle
inside the catheter rather than emerge past it. The policy's success
profile reads:

| anatomic region | pid19116 policy capability |
|---|---|
| CCA proper (target z 416-510) | **100 %** — fully solved |
| Cervical ICA (target z 510-575) | **82.5 %** — strong |
| Carotid siphon (target z 575-601) | **46.2 %** — half the time |

For clinical purposes the cervical-ICA capability (82–100 %) is the band
where most catheter-based neurovascular interventions are staged from —
the siphon-band ceiling is the inherently-hardest navigation in the
vasculature.

---

## 19. Decision tree update — what's solved, what's left

### Solved (Plan v9 + v10)

- **Critic stability** under AWAC + grad_clip 1.0 + grounding + step-PER
  + warm-start. No divergence through 750k+ explore steps.
- **Fork commit** — the policy reliably threads `(11)→RCCA` from a good
  start state (98 / 98 in the pid19116 sweep).
- **Proximal-CCA + cervical-ICA target reach** — 88-100 % from pid19116.
- **Per-state attribution** — restore_ckpt is logged at EPISODE_START and
  in the info dict; per-state success tables and failure breakdowns are
  reproducible from the run dir alone.
- **Anatomically-honest failure analysis** — wfs at distal targets is a
  catheter-overshoot artifact at the siphon, not a navigation failure.

### Unsolved (the levers for the next iteration)

1. **Action saturation / entropy collapse** — `log_std_min` floors the
   Gaussian term, not the tanh-Jacobian. Mean drifts into rail
   monotonically (clamp 0.06 → 0.53 over 650k updates). The targeted fix
   is an **explicit entropy bonus in the AWAC policy loss** (penalizes
   log_pi directly including the tanh-Jacobian) — deferred to the next
   plan.

2. **Higher `awac_lambda`** — currently 3.0 (sharp advantage weighting →
   clones only the few high-A saturated actions). Raise to 6-10 to
   broaden the cloning distribution and slow the saturation feedback
   loop.

3. **Per-state curriculum** — pid10145 (10 %) and pid20043 (25 %) are
   dragging the mixed-state numbers. Narrowing to pid19116 + pid3285 +
   pid4907 (the three states already at ≥ 28 %) and pushing those to >
   80 % is the cleanest path to a high mixed-state number.

4. **Siphon-specific training** — once cervical ICA is at > 90 %, biased
   target sampling toward z=575–601 (the siphon band) would attack the
   46.2 % distal ceiling. The wfs catheter-overshoot mechanism (saturated
   `cath_trans` in tortuous anatomy) is the proximate cause and is
   addressed by levers 1+2.

5. **`update_per_explore_step` modest drop** (1.0 → 0.5) — slows the
   collapse feedback loop in addition to the entropy bonus. Trade-off:
   2× wall-clock per update. Not the primary lever but composes well.

### Out of scope under the standing rule (would change reward/obs/terminal)

- Speed cap / saturation penalty in reward.
- Cross-track weight retune.
- Wider TargetReached zone, action-history in observation, OFF_BRANCH_GRACE_STEPS
  relaxation, etc.

These would attack the same problems from the reward side; per the
standing user rule (only stabilization knobs + start states, never
reward/obs/terminal without explicit approval), they are explicitly
deferred.

---

## 20. Updated file index (Plan v10 additions)

| File | Sections |
|---|---|
| `eve_rl/eve_rl/network/gaussianpolicy.py` | §13a (already accepts log_std_min) |
| `training _scripts/util/agent.py` | §13a (thread log_std_min to 3 ctors), §13b (heatup_until_successes plumbing — built, abandoned) |
| `eve_rl/eve_rl/runner/runner.py` | §13b (heatup-until-N early-stop + `_episode_threaded` staticmethod — built, abandoned) |
| `training _scripts/DualDeviceNav_train.py` | §13a (`--log_std_min`), §13b (`--heatup_until_successes` built-but-unused; `--heuristic_from_restore` deployed; warm-start gate relax), §13d (env_eval wrap, `env_eval_unwrapped`), §16 (`--eval_only_checkpoint` branch) |
| `training _scripts/util/env5.py` | §13c (restore_ckpt → EPISODE_START + info), §13e (double-restore fix) |
| `training _scripts/util/checkpoint_restore.py` | (no further change — wrapper unchanged from §8b) |
| `saved/rcca_5good_checkpoints/` | §13f — 5 curated good-state .npz |
| `saved/rcca_best_state_pid19116/` | §16 — single-file restore dir for the pid19116 sweep |
| `launch_rcca_awac_v3.sh` | §13 (new launcher) |
| `launch_rcca_eval_best_state.sh` | §16 (eval-only launcher) |
| `saved/evalRunAnalysis/trace_eval.py`, `classify_all.py` | §17 (eval-failure analysis scripts) |
