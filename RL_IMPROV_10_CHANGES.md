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
