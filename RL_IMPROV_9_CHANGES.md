# RL Improv 9 — Changes Reference

All changes in this worktree relative to the state at the end of
`RL_IMPROV_8_CHANGES.md`. RL_IMPROV_8 ended with a **deferred/pending**
section (§31–38): a discrepancy audit, a unification plan, an RL
architecture analysis (per-daughter vs shared), an HER/PER discussion,
and a deferred SOFA-restore design. None of §32–38 was implemented in
RL_IMPROV_8 — they were analysis only.

**This document covers the IMPLEMENTATION of the RCCA-scoped subset of
that plan (RL_IMPROV_8 §32–37) — the first per-daughter RL training
cycle. Internally tracked as "Plan v5".**

Scope: make the RCCA heuristic, the reward function, and the observation
space mutually consistent; complete the MDP's Markov property so the
env is sound for step-wise (TD) SAC; and stand up an RCCA-only SAC
training entry point. Built against env5 + the path-aware state machine
from RL_IMPROV_8 §14.

Out of scope (deliberately deferred — see §11): LVA's `path_extension_set`,
LCCA Phase-C variant dispatch, PER/HER/per-target buffers, the fuller
SOFA checkpoint restore (RL_IMPROV_8 §38).

---

## 1. Reward Unification — gate `ArcLengthProgress` on `is_on_correct_path()`

### Context

RL_IMPROV_8 §32 audit finding **A4**: the dense `ArcLengthProgress`
shaping reward projected the tip onto the planned-path polyline
unconditionally (`get_projection()`), so a wire wedged 30–50 mm
laterally off-path still earned `+0.01/mm` shaping whenever its
*projection* drifted forward. The heuristic (via `is_on_correct_path()`)
treated such a wire as off-path; the reward did not. Heuristic and
reward disagreed on "is the wire making valid progress".

### Fix

**File:** [eve/eve/reward/arclengthprogress.py](eve/eve/reward/arclengthprogress.py)

`step()` now branches on `_path_context.is_on_correct_path()`:

- **On-path:** `r_progress = +progress_factor × (prev_d_rem − d_rem_curr)`
  — forward arclength motion on the planned polyline, as before.
- **Off-path:** `r_progress = −progress_factor × Δ(off_arc_since_divergence)`
  — a **symmetric per-step delta**. Drifting deeper into a wrong branch
  → negative; retracting toward the divergence point → **positive**
  (matching the heuristic's implicit "retract when off-path" objective).
- Lateral penalty `−lateral_penalty_factor × cross_track` unchanged.

New field `_prev_off_arc` (reset to 0 per episode; also reset to 0 on
any on-path step so a fresh off-path excursion starts from a clean
baseline).

### Why delta, not cumulative

The first implementation used `−progress_factor × off_arc_total` paid
*every step* — a wire 30 mm off-path for 400 steps lost ~120 reward,
a 35:1 success/failure asymmetry that would prevent SAC critic
convergence. The per-step delta bounds total off-path cost to
`progress_factor × max_drift` (≈ −0.3 to −1), symmetric with on-path
progress. A round-trip drift (out and back) nets ~0.

---

## 2. State-Machine Daughter-Commit Events (replace the +1 reward)

### Context

RL_IMPROV_8 §32 audit finding **A2**: the old `+1 CORRECT_ENTRY_REWARD`
fired when `arc_past_last_daughter ≥ 10 mm` — a proximity threshold
that triggered preemptively (before the wire had actually committed)
and gave **no signal at all for wrong-daughter crossings**.

### Fix

**Files:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py),
[training _scripts/util/env5.py](training _scripts/util/env5.py)

The old `arc_past ≥ 10` block in env5 (was ~lines 562-575) is **deleted**.
Replaced with state-machine-driven discrete events:

- `update_branch_state()` emits a `(junction_arc, ±1)` event into a
  per-step queue `_daughter_commit_events` whenever `_current_branch_idx`
  commits at a real daughter fork (a junction in `_path_daughter_arclengths`):
  - **+1** — committed to the on-path branch at that fork.
  - **−1** — committed to an off-path branch at that fork.
- Each fork latches once per episode via `_committed_forks` — re-crossings
  emit nothing.
- env5 drains the queue right after `update_branch_state()`, adds the
  signed rewards, and records per-episode flags `_received_correct_daughter`
  / `_received_wrong_daughter` (exposed in the `info` dict).

For the RCCA route (3 daughter forks: trunk-top, bridge entry,
`(11)→RCCA`): max +3 (correct at all three), worst −1 to −3.

### New pathcontext helpers

- `_maybe_emit_daughter_commit(j_arc, sign)` — emits if `j_arc` is a real
  daughter fork and not already latched.
- `_nearest_daughter_junction_arc(s)` — nearest daughter junction to
  arclength `s` (used for wrong-commit attribution).
- `_capture_divergence_point(s)` — records the wrong-branch arclength
  baseline at the moment of off-path divergence.
- `get_off_path_arc_since_divergence()` — arc the wire has traveled along
  its current wrong branch since divergence (feeds §1's off-path shaping).

New episode-state fields: `_divergence_wrong_branch_arc`,
`_divergence_wrong_branch_idx`, `_daughter_commit_events` (per-step
queue, cleared in `invalidate()`), `_committed_forks` (per-episode latch).

---

## 3. Removal of the Wrong-Branch Penalty Cascade

### Context

env5 carried `WRONG_BRANCH_ENTRY_PENALTY = −1.0` (one-time at
`off_branch_steps == 3`) and `WRONG_BRANCH_STEP_PENALTY = −0.1` (per
step thereafter), plus a 3-step grace to dampen classification flicker.
These were heuristic-era hacks pre-dating the path-aware state machine.

### Fix

**File:** [training _scripts/util/env5.py](training _scripts/util/env5.py)

Both penalties and the 3-step grace are **removed**. Rationale:

- They duplicated the §1 off-path arc shaping (which already penalizes
  off-path motion, proportional to drift severity, symmetric).
- They fired for any lateral wedge, not just topologically-meaningful
  daughter deflections — gradient noise.
- The heuristic has no analog of an "entry tax"; it just retracts.

What is **kept**: the `_off_branch_steps` counter still increments, and
the 50-step `wrong_branch_timeout` truncation still fires (→ −5
`FAILURE_TRUNCATION_PENALTY`). This matches the heuristic's
`wrong_branch_timeout` abort exactly. The `_heur_suppress_wrong_branch`
flag (LVA-only) still bypasses the timeout.

Constants `WRONG_BRANCH_ENTRY_PENALTY`, `WRONG_BRANCH_STEP_PENALTY`,
`CORRECT_ENTRY_REWARD` deleted from env5 (kept only as comments
documenting the removal).

### Resulting RCCA reward structure (heuristic-aligned)

| signal | trigger | magnitude |
|---|---|---|
| Step base | every step | −0.001 |
| On-path progress | on-path step | +0.01 × Δs_planned |
| Off-path drift / retract | off-path step | −0.01 × Δoff_arc (signed) |
| Lateral penalty | every step | −0.001 × cross_track_mm |
| Daughter commit | state-machine commit at real fork | ±1, latched per fork |
| TargetReached | terminal | +3 |
| FailureTruncationPenalty | wbt / fold-stall / vessel-end | −5 |

Clean RCCA success ≈ +5 to +6; worst trunk-wedge ≈ −6 (was ≈ −160 to
−200 before the §1 delta fix). Asymmetry 35:1 → ~1.5:1 — SAC-trainable.

---

## 4. Observation Enrichment — `LocalGuidance` 14 → 28 features

### Context

RL_IMPROV_8 §32 audit findings **B1/B2/B3**: the agent's observation
lacked the state variables the heuristic routes through — absolute
arclength, junction proximity, heuristic phase, and the J-tip bend
vector. Even perfect imitation learning would plateau because the
policy class isn't expressive enough given the input.

### Fix

**File:** [eve/eve/observation/localguidance.py](eve/eve/observation/localguidance.py)

The enrichment was done in two passes that grew the vector 14 → 31,
then a cleanup removed 3 permanently-zero legacy features (old 8-10,
wrong-branch entry distance + direction) and renumbered, landing at
**28**. Final layout — features 0–7 unchanged; 8–10 = correct-entry
distance + direction (was 11-13); then:

**Plan v5 base (11–18):**

| # | feature | source |
|---|---|---|
| 11 | `arc_to_next_daughter_norm` | `get_arclength_to_next_daughter_entry()`, clip 100 mm |
| 12 | `arc_past_last_daughter_norm` | `get_arclength_past_last_daughter_entry()`, clip 100 mm |
| 13-16 | phase one-hot (default / A / B / C) | `_heur_rva_phase` via `_phase_to_onehot()` |
| 17-18 | `bend_hat_x/z_2d` | J-tip bend `p0+p2−2p1` from `tracking3d[0:3]`, projected 2D |

**Tier 1 — Markov-completing features (19–24):**

| # | feature | source |
|---|---|---|
| 19 | `off_arc_since_divergence_norm` | `get_off_path_arc_since_divergence()`, clip 50 mm |
| 20 | `off_branch_steps_norm` | `_off_branch_steps / 50` |
| 21 | `fold_stall_count_norm` | `_fold_stall_count / 20` |
| 22 | `episode_step_norm` | `episode_step / max_steps` |
| 23 | `forks_correct_norm` | `_n_correct_commits / 3` |
| 24 | `forks_wrong_norm` | `_n_wrong_commits / 3` |

**Branch categorical (25–27):**

| # | feature | meaning |
|---|---|---|
| 25 | `is_in_trunk` | `_current_branch_idx == _trunk_branch_idx AND on_planned_path` |
| 26 | `is_on_target_daughter` | `_current_branch_idx == _target_daughter_branch_idx AND on_planned_path` |
| 27 | `is_in_a_wrong_branch` | `not _on_planned_path` |

(Bridge / intermediate path segments → all three = 0.)

**Cleanup (RL_IMPROV_9):** the old features 8-10 (`dist_wrong_entry`,
`wrong_entry_dir_x/z`) were permanently zero — wrong daughters have no
arclength on the planned path, and the wrong-entry coords degenerated
to the tip. Removed; features 11-30 renumbered down by 3 → 8-27. The
dead write-only field `_prev_proj_s` in pathcontext was also deleted
(the state machine's hysteresis is carried by `_current_branch_idx`).

### Supporting plumbing

- **pathcontext.py:** caches `_trunk_branch_idx` and
  `_target_daughter_branch_idx` at reset (first / last entries of
  `_path_branch_sequence`); counts `_n_correct_commits` /
  `_n_wrong_commits` in `_maybe_emit_daughter_commit`.
- **env5.py:** mirrors `_heur_rva_phase`, `_off_branch_steps`,
  `_fold_stall_count`, `_episode_step_count`, `_env_max_steps` onto the
  `intervention` object at the start of `step()` — LocalGuidance holds
  an `intervention` reference but not the env, so this mirror is how it
  reads env-level counters. All reads use `getattr(..., default)` so
  non-env5 callers don't break.
- `LocalGuidance` helpers `_phase_to_onehot()` and
  `_compute_bend_hat_2d()` added; degenerate-path fallback `np.zeros`
  bumped 14 → 28; `space` low/high arrays rebuilt for 28 dims.

---

## 5. Wire Tracking — 150 mm Coverage, Frame-Stacking, Inserted Length

### Context

The `tracking` observation gave 5 points over 40 mm (positions only,
no velocity). Two physics-level partial-observability gaps remained:
(a) wire pose beyond 40 mm — the full bif2 region — was unobserved;
(b) the wire's kinematic velocity is genuine hidden state (the SOFA
beam has mass); `last_action` only gives *commanded* velocity, not
actual (the two diverge exactly in fold-stall / friction-lock).

### Fix

**File:** [training _scripts/util/env5.py](training _scripts/util/env5.py) — observation block.

1. **`Tracking2D(n_points=10, resolution=15)`** — 10 points at 15 mm
   spacing = 150 mm of distal wire, covering the full bif2 region.
2. **`Memory(n_steps=2, reset_mode=FILL)`** wrapping the normalized
   tracking — frame-stacks 2 deep. The agent derives actual wire
   velocity by position finite-difference. `FILL` seeds both frames
   with the reset state → velocity reads 0 at episode start (correct
   for a stationary wire). `Memory` wraps the per-episode normalizer,
   whose scale is fixed at reset, so stacked frames share one scale.
3. **`InsertionLengths`** added as a 5th obs-dict key (`inserted_lengths`,
   normalized) — total inserted length of guidewire + catheter. Removes
   the integration-history gap (two wires with identical tracked-point
   positions but different total insertion respond differently to a
   retract command). Single-frame — insertion velocity is already in
   `last_action`.

`Memory` and `InsertionLengths` are existing eve classes
([eve/eve/observation/wrapper/memory.py](eve/eve/observation/wrapper/memory.py),
[eve/eve/observation/insertionlengths.py](eve/eve/observation/insertionlengths.py))
— no new observation code.

### Resulting observation

| key | shape | flattened |
|---|---|---|
| `tracking` | (2, 10, 2) | 40 |
| `target` | (2,) | 2 |
| `last_action` | (4,) | 4 |
| `guidance` | (28,) | 28 |
| `inserted_lengths` | (2,) | 2 |
| **total** | | **76** (was 30 pre-Plan-v5) |

SAC auto-discovers the 76-dim input from `env.observation_space`; no
network or architecture change required.

---

## 6. Non-Markov Audit and Resolutions

A full audit identified 18 sources of hidden state that influence the
next observation or reward without being visible to the policy —
relevant because the next planned step is moving from episode-wise
(Monte-Carlo) RL to step-wise (TD) RL, which requires the Markov
property.

| # | hidden state | status after Plan v5 |
|---|---|---|
| 1 | `_committed_forks` latch | resolved — feats 26-27 |
| 2 | `_received_correct/wrong_daughter` flags | resolved — derived from feats 26-27 |
| 3 | `_committed_forks` reset pre-population | not a violation (observation-determined) |
| 4 | `_prev_d_rem` | not a violation (standard reward-shaping potential) |
| 5-6 | `_prev_off_arc` / divergence baseline | resolved — feat 22 |
| 7 | `_prev_proj_s` | not a violation — verified write-only dead code; **deleted** in the RL_IMPROV_9 cleanup |
| 8 | fold-stall delta baselines | minor residual — `delta_s` recoverable from frame-stacked tracking + `_fold_stall_count` exposed (feat 21); only the `d_corr_improving` bypass's `_prev_d_corr_arc` is genuinely hidden |
| 9 | `_off_branch_steps` | resolved — feat 23 |
| 10 | `_fold_stall_count` | resolved — feat 24 |
| 11 | `_episode_step_count` | resolved — feat 25 |
| 12-13 | `_current_branch_idx` / `_on_planned_path` | resolved — feats 28-30 |
| 14 | legacy hysteresis (`_stable_on_*`) | not a violation (log-only / dead code) |
| 15 | `_last_on_path_arclen` | resolved — Option C (§7 below) |
| 16 | wire velocity | resolved — frame-stacking (§5) |
| 17 | BeamAdapter strain | resolved implicitly (corotational — derivable from positions) |
| 18 | contact-force magnitude / LCP warm-start | residual — would need SOFA-side plumbing |

### Item 15 — `_last_on_path_arclen` (Option C)

**File:** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py) — `update_branch_state()`

`_last_on_path_arclen` is a per-episode-history value used as a fallback
rejoin anchor in `get_routed_d_corr_to_next_daughter_entry()`. Two
off-path states with the same tip position but different
`_last_on_path_arclen` would compute different `routed_d_corr`
(guidance feature 11) — non-Markov.

**Option C (chosen over "expose it as an obs feature"):** at the moment
the state machine flips off-path, set
`_last_on_path_arclen = _nearest_daughter_junction_arc(s)` — a
deterministic function of the current projection arclength. This
**removes the history-dependence at the root** rather than observing
it: feature 11 itself becomes a clean function of current geometry, no
new obs dimension, and the nearest-junction rejoin anchor is more
geometrically principled than "wherever the wire last happened to
touch the path". Applied in both off-path transition branches
(cross-track and on-path-mask).

### Residual non-Markov (accepted)

Items 7, 8 (minor delta baselines) and item 18 (contact-force /
solver warm-start) remain. Item 18 is the only fundamental one and
would require exposing SOFA's `velocity` / `force` DataFields — same
plumbing as the deferred fuller-checkpoint-restore (RL_IMPROV_8 §38).

---

## 7. `DualDeviceNav_train.py` — `main(args)` Refactor + Per-Daughter Flags

### Context

The trainer's entire body lived inside `if __name__ == "__main__":`
with no callable entry point — a per-daughter wrapper couldn't reuse it.

### Fix

**File:** [training _scripts/DualDeviceNav_train.py](training _scripts/DualDeviceNav_train.py)

- **Refactor:** the post-argparse body is moved into a module-level
  `def main(args):`. The `if __name__ == "__main__":` block keeps
  `mp.set_start_method("spawn", ...)`, the argparse builder, and a
  final `main(args)` call. Pure structural change — no behavior
  difference for `python DualDeviceNav_train.py ...`. Other entry
  points (per-daughter wrappers, future step-RL variants, eval-only
  invocations) can now `from DualDeviceNav_train import main`.
- **Three backward-compatible CLI flags:**
  - `--target_branches` — comma-separated override of the default
    4-daughter `TARGET_BRANCHES`. Bound to a function-local
    `target_branches` (NOT a rebind of the module global — that would
    raise `UnboundLocalError` when the flag is omitted).
  - `--heuristic_factory` — `module:ClassName` of a custom heuristic
    factory (e.g. `util.heuristic_policy_rcca:RCCAHeuristicActionFunctionFactory`).
  - `--seeding_success {term, clean_thread}` — definition of "success"
    for the heuristic-seeding filter. `term` = TargetReached terminal
    (default, all daughters). `clean_thread` =
    `info["received_correct_daughter"] AND NOT info["received_wrong_daughter"]`
    — the looser per-daughter criterion.

### Seeding-success rationale

RCCA's `term=True` rate is 2.9% — too sparse to seed a replay buffer
without hitting the episode cap. The `clean_thread` criterion
("committed to RCCA at the final fork, not to RVA") raises the
effective rate to ~15-20%, giving ~75 demo successes in ~500 episodes.

---

## 8. Compatibility Audit and Fixes

A three-agent audit checked heuristic↔reward↔observation consistency,
the SAC training pipeline, and reward-magnitude balance for the RCCA
run. The pipeline was found structurally sound (obs-shape change
auto-discovers via `env.observation_space`; replay buffer is
shape-agnostic; factory + schedule flow correct; pickling unchanged).
Five issues were found and fixed:

1. **Off-path arc penalty compounding** — fixed (§1, cumulative →
   per-step delta).
2. **`WRONG_BRANCH_ENTRY/STEP` redundancy** — fixed (§3, removed).
3. **`TARGET_BRANCHES` `UnboundLocalError`** — the `main()` refactor
   plus a conditional rebind made `TARGET_BRANCHES` function-local;
   fixed by using a separate local `target_branches` (§7).
4. **`np.zeros(14)` leftover** in `LocalGuidance.reset()`'s
   degenerate-path branch — bumped to `np.zeros(28)` (after the §4
   cleanup renumber).
5. **Reward magnitude balance** — addressed by 1+2; clean-success vs
   worst-failure asymmetry now ~1.5:1.

Confirmed working without change: RCCA phase-string → one-hot mapping;
per-episode reset of all new fields; `insertion_z=345` leaves all 3
RCCA forks commit-eligible; `mp.set_start_method` ordering across the
refactor; action-range match between heuristic and SAC tanh output.

---

## 9. New File — `launch_rcca_train.sh`

**File:** [launch_rcca_train.sh](launch_rcca_train.sh)

Docker launcher for the RCCA-only SAC run. Mirrors `launch_rcca_v1.sh`'s
mount set (which already includes the eve_rl SAC + replaybuffer
modules) and invokes `DualDeviceNav_train.py` with:

```
--env_version 5 -n rcca_sac_v1 --insertion_z 345
--heuristic_seeding 500 --min_success_rate 0.15 --max_seeding_multiplier 5
--target_branches "Centerline curve - RCCA.mrk"
--heuristic_factory "util.heuristic_policy_rcca:RCCAHeuristicActionFunctionFactory"
--seeding_success clean_thread
-nw 16 -d cuda:0
```

The reward + observation changes mean any pre-existing SAC checkpoint
is incompatible — this run trains from scratch.

---

## 10. Verification Plan

- **Phase 0 — static** (done): all 6 modified files parse; LocalGuidance
  reports `space.shape` consistent with 28-dim guidance; obs dict has
  5 keys; pathcontext exposes the new methods/fields.
- **Phase 1 — RCCA heuristic 50-ep regression** (pending docker): run
  the modified env5 + reward + obs through `launch_rcca_v1.sh` for 50
  episodes. Targets: `term=True` rate ≥ 2% (baseline 2.9%); clean-thread
  RCCA rate ≥ 15%; reach % ≥ 30%; abort distribution within 10% of the
  500-ep baseline ([2026-05-10_033729_env5_rl8_RCCA_v1_500ep](saved/eve_paper/neurovascular/full/mesh_ben/2026-05-10_033729_env5_rl8_RCCA_v1_500ep)).
- **Phase 2 — RCCA SAC training** (pending docker): `launch_rcca_train.sh`.
  Heuristic seeding (~30 min) → SAC training (~8-16 hr). Monitor critic
  loss convergence, eval reward trend, end-of-training RCCA success ≥ 5%.

---

## 11. Single-Daughter Scoping — `default_target_branch`

### Context

A pre-launch review of the trainer's CLI options found that only the
**heuristic-seeding** phase is RCCA-scoped (its `build_episode_schedule`
passes `target_branch` per episode). The other three episode-generating
phases — **heatup**, **explore** (the 20M-step SAC data collection), and
**eval** — are driven by the eve_rl runner, which resets the env with no
`options`. With no `target_branch` option, `CenterlineRandom.reset()`
([centerlinerandom.py:40-50](eve/eve/intervention/target/centerlinerandom.py#L40-L50))
samples a target from **all four daughters**.

Left unfixed, "RCCA-only training" would seed an RCCA buffer but then
explore + be evaluated across all 4 daughters — i.e. the shared-model
path, not per-daughter, and the eval metric would be averaged over 3
daughters the policy never trained on.

### Fix

**Files:** [training _scripts/util/env5.py](training _scripts/util/env5.py),
[training _scripts/DualDeviceNav_train.py](training _scripts/DualDeviceNav_train.py)

- `BenchEnv5.__init__` gains a `default_target_branch` parameter. In
  `reset()`, when `options` lacks `target_branch`, the default is
  injected (a fresh dict — the caller's options are not mutated). No-op
  when `default_target_branch is None` (legacy multi-daughter mode).
  Heuristic seeding passes `target_branch` explicitly, so it is
  unaffected; heatup / explore / eval now resolve to RCCA.
- `DualDeviceNav_train.main()` computes `default_tb = target_branches[0]`
  when exactly one target branch is configured, and passes it as
  `default_target_branch` to both `env_train` and `env_eval`
  construction (env5 only — other env versions don't accept the kwarg).

### Curriculum decision

`--curriculum` is left **OFF** for the RCCA first cycle. The
`ActionCurriculumWrapper`'s Stage 1 (catheter forced to `gw_trans × 0.8`)
already matches the RCCA heuristic's `CATH_FOLLOW_RATIO = 0.8`, so the
seeded demos encode it; Stage 2's `×0.1` catheter scaling is a
mid-training non-stationarity with no clear benefit for guidewire-
dominated RCCA navigation. Curriculum can be revisited if traces show
catheter thrashing.

### Launcher

`launch_rcca_train.sh` also gains `--save_heuristic_cache` so the ~500
RCCA heuristic demos are written to `results/rcca_heuristic_cache.npz`
and reusable by future RCCA runs (skips the ~30-min re-seed).

---

## 12. `clean_thread` Definition Fix + Training Snapshots

### `clean_thread` was too lenient — `reached_target_daughter`

The §7 `clean_thread` seeding criterion was `received_correct_daughter
AND NOT received_wrong_daughter`. But `received_correct_daughter` is set
on the **first +1 commit**, and the RCCA path's first daughter fork is
trunk-top `(2)→(0)` — which qualifies as a `_path_daughter_arclengths`
fork (off-path branch (18)→LVA). So any wire that merely ascends the
trunk past trunk-top and then wedges, never reaching RCCA, satisfied
`clean_thread`. With `min_success_rate` that would seed the buffer with
trunk-wedge episodes.

**Fix:** env5 adds a per-episode latch `_reached_target_daughter`, set
True once the state machine commits `_current_branch_idx` onto
`_target_daughter_branch_idx` (the RCCA daughter itself). Exposed as
`info["reached_target_daughter"]`. The trainer's `clean_thread` filter
is now `reached_target_daughter AND NOT received_wrong_daughter` —
"committed onto the target daughter, never wrong-deflected".

`launch_rcca_train.sh` `--min_success_rate` raised `0.15 → 0.40`. With
the corrected (stricter) definition the RCCA clean-thread rate ≈ 33.7%
(the §31 reach rate), so 0.40 targets ~200 genuine RCCA-thread demos,
collected in ~600 seeding episodes (within the 2500 cap).

### Training snapshots — phase-bucketed, Option A

A `--snapshots {none,mesh,centerlines}` flag is added to the trainer.
When set, `SNAPSHOT_MODE`/`SNAPSHOT_DIR` are exported **before worker
spawn** (same constraint as `STEP_LOG_DIR`), so every worker renders an
end-of-episode PNG. env5 buckets each snapshot by training phase:

    <snapshot_dir>/seed/<target>/<reason>/...
    <snapshot_dir>/eval/<target>/<reason>/...
    <snapshot_dir>/explore/<target>/<reason>/...

(`eval` ← `mode=="eval"`; `seed` ← `_heuristic_mode`; else `explore`.)
The episode reward is encoded in the PNG filename (`_R<+/-><value>_`)
so the keep-policy can rank without opening the images.

**Keep-policy** ([prune_training_snapshots.py](prune_training_snapshots.py),
new standalone host script): keep ALL `seed/` + ALL `eval/`; for
`explore/`, keep only the 10 best + 10 worst (by reward) per 100
consecutive episodes. Idempotent / incremental: `explore/` is a staging
area, finalized blocks' 20 keepers are moved to
`explore_kept/block_<N>/` and the other 80 deleted — so mtime-grouping
stays stable and the script is safe to run periodically (cron) or once
post-hoc. `--dry-run` previews.

Snapshots are rendered for every episode (Option A — "snapshot-all,
then prune"); per-episode render overhead is ~1-2% of episode wall time.

---

## 13. Episode-RL → Step-RL Migration (Plan v6)

### Context

The trainer's SAC was **episode-based**: `VanillaEpisodeShared` stores
whole episodes, `sample()` returns padded episode sequences, batches are
32 episodes. This pairs with an **LSTM embedder** — [agent.py:152-155](training%20_scripts/util/agent.py#L152-L155)
builds `eve_rl.network.component.LSTM` when `embedder_layers and
embedder_nodes and not ff_only`, and `BenchAgentSynchron` is constructed
`ff_only=False`. The LSTM is what makes episode/sequence batching
load-bearing. Every Plan v5 Markov-completion change was the prerequisite
for the canonical **step-based** (transition) SAC, which uses a
feedforward policy. This section migrates RCCA training to step-RL.

Exploration found `VanillaStep` / `VanillaStepShared` transition buffers
already exist and `SAC.update()` is buffer-agnostic (every `padding_mask`
use guarded; `seq_length = actions.shape[1]` resolves to 1 for transition
batches; `Batch.padding_mask` defaults to `None`). So the migration is a
toggle, not a rewrite.

### Fixes

**Issue 1 — `VanillaStep.push` dropped every episode's terminal transition.**
[vanillastep.py:19](eve_rl/eve_rl/replaybuffer/vanillastep.py#L19) iterated `range(len(episode) - 1)` — an episode
with N actions has N transitions (i=0..N-1; `flat_obs` has N+1 entries),
so i=N-1 (the terminal transition, carrying the `done` flag and the +3 /
−5 terminal reward) was never stored. Fixed to `range(len(episode))`.
`VanillaStepShared` ([vanillashared.py:170](eve_rl/eve_rl/replaybuffer/vanillashared.py#L170)) wraps an internal
`VanillaStep`, so the fix propagates with no further change.

**Issue 2 — episode-scaled hyperparameters.** The episode constants
(`REPLAY_BUFFER_SIZE=1e4` episodes, `BATCH_SIZE=32` episodes,
`UPDATE_PER_EXPLORE_STEP=1/20`) are wrong as transition values. Step mode
uses canonical step-SAC values: **1e6**-transition buffer, **256** batch,
**1.0** update/explore-step.

**Issue 3 — recurrent network.** Step-RL feeds `seq_length=1` batches; an
LSTM over a 1-step sequence carries no memory. The feedforward path is
needed. Rather than the `ff_only` MLP-embedder branch (which has a latent
bug — it clobbers `hidden_layers`, shrinking the base nets), step mode
uses the **no-embedder path**: passing `--embedder_layers 0` selects
`ComponentDummy` ([agent.py:159-160](training%20_scripts/util/agent.py#L159-L160)) and leaves `--hidden` intact.
Step-mode network = `ComponentDummy` + `MLP([256, 256])` — the canonical
SAC default, ample for the 76-dim feature-engineered observation. (The
episode `[900,900,900,900]` was sized for the old 154-dim `Centerlines2D`
obs + LSTM + 4-daughter multitask.)

### Mechanism — `--replay_mode {episode,step}` toggle

`DualDeviceNav_train.py` gains `--replay_mode` (default `episode` — the
multi-daughter LSTM path is byte-for-byte unchanged). Step mode sets the
buffer/batch/update hyperparameters via NEW local variables
(`replay_buffer_size` / `batch_size` / `update_per_explore_step` — not a
module-global rebind, avoiding the `UnboundLocalError` trap). `replay_mode`
is passed to `BenchAgentSynchron`, which gains a `replay_mode` param and
selects `VanillaStepShared` vs `VanillaEpisodeShared`. The network arch
stays controlled by the existing `--hidden` / `--embedder_layers` args.

`launch_rcca_train.sh` adds `--replay_mode step --hidden 256 256
--embedder_layers 0 --learning_rate 0.0003` — step buffer + feedforward
`MLP([256,256])` + the canonical step-SAC learning rate.

### Learning rate

Step mode uses **`3e-4`** (`--learning_rate 0.0003`). The trainer's default
`0.00021989…` is a stale Optuna value tuned for the old LSTM / episode-batch
/ `900×4` / 4-daughter regime — not transferable to a feedforward MLP on
transition batches. `3e-4` is the canonical SAC learning rate for exactly
this setup (feedforward `MLP([256,256])`, transition replay, batch 256, one
update per explore step) and is robust across tasks; the existing `LinearLR`
schedule (`LR_END_FACTOR=0.15` over `6e6` steps) decays it regardless. Drop
to `1e-4` only if step-SAC shows instability.

### What needs NO change

SAC algo (buffer-agnostic, `padding_mask`-guarded). agent.py network
construction (the `ComponentDummy` path is reached via `--embedder_layers
0`; the buggy `ff_only` branch is never entered). Heuristic seeding /
cache (`VanillaStep.push` accepts `EpisodeReplay`; episodes explode into
transitions on push). env5 / observation / reward.

---

## 14. Prioritized Experience Replay (Plan v7)

### Context

Plan v5 §36 and Plan v6 deferred PER and HER. Plan v7 pulled them into
scope before the first RCCA step-SAC run. **HER was ruled out** — the
reward is path-structured, not goal-structured: `ArcLengthProgress`
rewards motion along a *planned-path polyline* recomputed per-episode by
the pathfinder from the target; it also depends on per-episode mutable
state (`_prev_d_rem`, the pathcontext state machine — `_current_branch_idx`
hysteresis, `_committed_forks` latches, `_divergence_wrong_branch_arc`)
that cannot be reconstructed from a stored transition; and the goal is
baked into the `LocalGuidance` observation. A post-hoc goal relabel would
require re-running the SOFA simulation. `CenterlineRandom` already samples
a random RCCA goal each episode, so HER's main benefit is largely already
obtained. **PER alone** addresses the rare-success / class-imbalance
problem.

### PER — on/off toggle (`--per`)

PER samples transitions ∝ TD-error magnitude (sum-tree), applies
importance-sampling (IS) weights to correct the bias, and refreshes each
sampled transition's priority with its fresh TD error after the update.
It is an **orthogonal on/off switch on step mode**, not a third replay
mode. Because SAC's IS-weight handling is fully guarded by `if
batch.is_weights is not None`, the uniform path is byte-for-byte unchanged
when `--per` is off — so a PER run (`--replay_mode step --per`) and a
baseline run (`--replay_mode step`) with all other flags identical cleanly
attribute any difference to PER.

### New buffer — `PERVanillaStep` + `PERVanillaStepShared`

[pervanillastep.py](eve_rl/eve_rl/replaybuffer/pervanillastep.py) — **new.** `PERVanillaStep`: `VanillaStep`-style
transition storage backed by a binary **`SumTree`** over per-transition
priorities (O(log N) sample + update — ~1 ms/step, negligible vs the
50-300 ms SOFA env step; the naive O(N) `np.random.choice(p=...)` array
was rejected for being ~3-7 ms/call on a 1e6 buffer). `push()` enters new
transitions at the current **max priority** (standard PER — each fresh
transition sampled ≥ once); it iterates `range(len(episode))` (the same
off-by-one fix as §13). `sample()` does stratified proportional sampling,
returns the `Batch` plus the sampled tree **`indices`** and IS **weights**
`w_i = (N·P(i))^{-β} / max_j w_j` with β annealed `beta_start → 1.0` over
`beta_steps` calls. `update_priorities(indices, td_errors)` sets
`priority = (|td| + ε)^α`.

[pervanillashared.py](eve_rl/eve_rl/replaybuffer/pervanillashared.py) — **new.** `PERVanillaStepShared` wraps
`PERVanillaStep` in the buffer subprocess (mirrors `VanillaStepShared`).
Its `__init__` deliberately calls `VanillaSharedBase.__init__` directly
(not `VanillaStepShared.__init__`) so the **third queue**
`_priority_update_queue` is created *before* the subprocess spawns. The
trainer sends `(indices, td_errors)` over it; the subprocess `loop()`
drains it (before sampling — keeps priorities fresh, stops the queue
backing up) and calls `update_priorities`. `PERVanillaSharedBase` is the
worker/trainer handle, adding `update_priorities()` (numpy-converts and
enqueues). Accepted approximation: a priority update for a ring-buffer
slot overwritten since sampling is applied anyway (bounded, self-healing —
standard distributed-PER behavior, à la Ape-X).

### `Batch` extension

[replaybuffer.py](eve_rl/eve_rl/replaybuffer/replaybuffer.py) — `Batch` NamedTuple gains two **trailing
optional** fields `is_weights` and `indices` (both default `None`, so
episode / step-uniform batches are structurally unaffected). `Batch.to()`
moves `is_weights` to device float32; `indices` stay a CPU long tensor
(they never enter the network — only address buffer slots).

### SAC — IS-weighted losses + TD-error feedback

[sac.py](eve_rl/eve_rl/algo/sac.py) — `update()` now accesses `Batch` fields **by name**
(not a positional 5-unpack, which would break on the 7-field tuple) and
moves `is_weights` to device as `(B,1,1)`. `_update_q1` / `_update_q2`:
when `is_weights` is present, loss is `(is_weights · td²).mean()` instead
of `F.mse_loss`; otherwise the plain `mse_loss` path is unchanged.
`_update_q1` additionally extracts per-sample `|TD|` (averaged over any
sequence dims → one scalar per transition, detached CPU numpy) and returns
it. `_update_policy`: the per-sample `(α·log_pi − min_q)` is IS-weighted
before `.mean()` when weights are present. `update()` stashes the per-
sample `|TD|` in `self.last_td_errors`. All IS-weight logic is
`is_weights is not None`-guarded → uniform batches behave exactly as
before. `single.py` `_log_batch_samples` also switched to name-based
`Batch` access.

### Priority feedback loop

[single.py](eve_rl/eve_rl/agent/single.py) — after `algo.update(batch)`, when the buffer
exposes `update_priorities` and `batch.indices` is not `None`, the trainer
calls `replay_buffer.update_priorities(batch.indices,
algo.last_td_errors)` — routed to the subprocess via the priority queue.
Guarded → a no-op for uniform buffers.

### Wiring

[agent.py](training%20_scripts/util/agent.py) — `BenchAgentSynchron` gains `per` /
`per_alpha` / `per_beta_start` / `per_beta_steps` params; buffer selection
becomes `step and per` → `PERVanillaStepShared`, `step` → `VanillaStepShared`,
else → `VanillaEpisodeShared`.

[DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py) — adds `--per` (`store_true`,
default OFF), `--per_alpha` (0.6), `--per_beta_start` (0.4). Using `--per`
with `--replay_mode episode` raises a `ValueError` (PER is step-only).
`per_beta_steps` is set to `TRAINING_STEPS` so β anneals over the whole
run. The flags are threaded to `BenchAgentSynchron` and recorded in
`custom_parameters`.

[launch_rcca_train.sh](launch_rcca_train.sh) — adds `--per`. The uniform baseline
run is the same launcher with `--per` removed and a distinct `-n` name.

### PER hyperparameters

`alpha = 0.6` (priority exponent), `beta` annealed `0.4 → 1.0` over
`TRAINING_STEPS`, `epsilon = 1e-6` (priority floor). Canonical PER values.

### What needs NO change

`VanillaStep` / `VanillaStepShared` (uniform step mode) — untouched.
SAC behavior for uniform batches — fully `is_weights is not None`-guarded,
so `--per` OFF is byte-for-byte the uniform step-SAC. env5 / observation /
reward / heuristic seeding — PER is purely a buffer-sampling concern;
heuristic-seeded transitions enter the sum-tree at max priority like any
other.

---

## 15. Update-Budget Fix — Exclude Heatup/Seeding from the SAC Update Schedule

### Symptom

The first RCCA step-SAC + PER run (`2026-05-17_215725_rcca_sac_v1`) trained
cleanly through seeding → heatup → explore, then the **critic diverged**:
`q1_loss` 0.1 → 1832, `q1_mean` 21 → 1287 → −504 (sign flip), `grad_norm_q1`
0.3 → ~2900 over update steps 5k–72k. No NaN yet, but the Q-function was
meaningless.

### Root cause — [runner.py:286-298](eve_rl/eve_rl/runner/runner.py#L286)

`explore_and_update()` sized each update call as:

```python
total_experience_steps = step_counter.heatup + step_counter.exploration
update_steps = total_experience_steps * update_steps_per_explore_step
             - step_counter.update
```

`heuristic_seed()` increments the **`heatup`** counter (synchron.py:373 —
"both are pre-training phases"). So after seeding (131k steps) + random
heatup (21k steps), `step_counter.heatup ≈ 152,678`. With
`update_steps_per_explore_step = 1.0` (step mode) and `step_counter.update
= 0`, the **first** `update()` call inherited a budget of
`(152,678 + 10,472) × 1.0 ≈ 163,150` gradient steps — all run at once on
the frozen 87,544-transition buffer (~477× data reuse, zero fresh data).
The critic bootstrapped off its own targets with no corrective signal and
diverged; PER accelerated it by repeatedly resampling the most
TD-inflated transitions.

The formula's "catch-up to target ratio" intent is fine for online RL
where heatup is small (~5-20k) and exploration grows gradually. It breaks
here because (a) heuristic seeding folds 131k steps into the `heatup`
counter and (b) step mode's ratio is `1.0` (episode mode uses `1/20`) —
together handing the first call a 150k-update backlog.

### Fix

Budget the update schedule from **exploration steps only**:

```python
update_steps = step_counter.exploration * update_steps_per_explore_step
             - step_counter.update
```

First update ≈ one explore batch (~10k), every later call ≈ one explore
batch → clean ~1:1 interleaving with fresh data each cycle. The heatup +
heuristic-seeded transitions still train — they live in the replay buffer
and PER samples them throughout; they just no longer mandate a front-loaded
catch-up block. Over a full run the heatup term is negligible, so total
update count is essentially unchanged — only the upfront spike is removed.

### Counter conflation — assessed, left as-is

`heuristic_seed()` deliberately shares the `heatup` step/episode counters.
`step_counter.heatup` is consumed in 4 places: the budget formula (fixed
above), checkpoint save/load, the `heatup()` loop limit (relative —
unaffected), and logging. Only the budget formula was harmed. Adding a
dedicated `heuristic_seed` counter would touch 4 counter classes
(`StepCounter`/`StepCounterShared`/`EpisodeCounter`/`EpisodeCounterShared`)
+ `__iadd__` + the checkpoint schema (breaking checkpoint compat) for zero
functional gain once the formula is fixed. The shared counter is kept.

---

## File Index

| File | Change |
|---|---|
| [eve/eve/reward/arclengthprogress.py](eve/eve/reward/arclengthprogress.py) | §1 — `r_progress` gated on `is_on_correct_path()`; off-path = symmetric per-step `Δoff_arc` shaping; new `_prev_off_arc` field |
| [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py) | §2 — `_daughter_commit_events`, `_committed_forks`, `_divergence_wrong_branch_arc/idx`; methods `_maybe_emit_daughter_commit`, `_nearest_daughter_junction_arc`, `_capture_divergence_point`, `get_off_path_arc_since_divergence`. §4 — `_trunk_branch_idx`, `_target_daughter_branch_idx`, `_n_correct_commits`, `_n_wrong_commits`. §6 — Option C `_last_on_path_arclen` deterministic pinning in `update_branch_state()`; deleted dead write-only field `_prev_proj_s` |
| [training _scripts/util/env5.py](training _scripts/util/env5.py) | §2 — drain `_daughter_commit_events`, per-episode flags into `info`. §3 — delete wrong-branch penalty cascade + grace; delete unused constants. §4 — mirror `_heur_rva_phase` + counters onto `intervention`. §5 — `Tracking2D(10, 15)`, `Memory` 2-frame stack, `InsertionLengths` 5th obs key. §11 — `default_target_branch` ctor param + reset() injection. §12 — `_reached_target_daughter` latch + `info`; phase-bucketed snapshot call |
| [eve/eve/observation/localguidance.py](eve/eve/observation/localguidance.py) | §4 — obs 14 → 28 (enriched to 31, then dead features 8-10 removed + renumbered); features 8-27; helpers `_phase_to_onehot`, `_compute_bend_hat_2d`; `space` bounds + degenerate-path `np.zeros` updated |
| [training _scripts/util/snapshot.py](training _scripts/util/snapshot.py) | §12 — `save_snapshot` gains `phase` param (seed/eval/explore sub-bucket); reward encoded in PNG filename |
| [training _scripts/DualDeviceNav_train.py](training _scripts/DualDeviceNav_train.py) | §7 — body refactored into `main(args)`; CLI flags `--target_branches`, `--heuristic_factory`, `--seeding_success`; per-daughter seeding filter. §11 — `default_tb` / `env_kwargs` → `default_target_branch` passed to env5 `env_train` + `env_eval`. §12 — `clean_thread` filter uses `reached_target_daughter`; `--snapshots` flag + `SNAPSHOT_MODE/DIR` pre-spawn export. §13 — `--replay_mode` flag; mode-conditional `replay_buffer_size`/`batch_size`/`update_per_explore_step` locals; `replay_mode` passed to `BenchAgentSynchron`. §14 — `--per` / `--per_alpha` / `--per_beta_start` flags; PER-with-episode `ValueError` guard; `per_beta_steps = TRAINING_STEPS`; flags threaded to agent + `custom_parameters` |
| [training _scripts/util/agent.py](training _scripts/util/agent.py) | §13 — `BenchAgentSynchron` gains `replay_mode` param; selects `VanillaStepShared` vs `VanillaEpisodeShared`. §14 — `per`/`per_alpha`/`per_beta_start`/`per_beta_steps` params; `PERVanillaStepShared` selected when `step and per` |
| [eve_rl/eve_rl/replaybuffer/vanillastep.py](eve_rl/eve_rl/replaybuffer/vanillastep.py) | §13 — `push` off-by-one fixed (`range(len(episode))`) — captures the terminal transition |
| [eve_rl/eve_rl/replaybuffer/pervanillastep.py](eve_rl/eve_rl/replaybuffer/pervanillastep.py) | §14 — **New.** `SumTree` + `PERVanillaStep` — proportional priority sampling, IS weights, `update_priorities` |
| [eve_rl/eve_rl/replaybuffer/pervanillashared.py](eve_rl/eve_rl/replaybuffer/pervanillashared.py) | §14 — **New.** `PERVanillaStepShared` + `PERVanillaSharedBase` — subprocess wrapper + `_priority_update_queue` |
| [eve_rl/eve_rl/replaybuffer/replaybuffer.py](eve_rl/eve_rl/replaybuffer/replaybuffer.py) | §14 — `Batch` += trailing optional `is_weights` / `indices`; `Batch.to()` moves them |
| [eve_rl/eve_rl/replaybuffer/__init__.py](eve_rl/eve_rl/replaybuffer/__init__.py) | §14 — export `PERVanillaStep`, `SumTree`, `PERVanillaStepShared`, `PERVanillaSharedBase` |
| [eve_rl/eve_rl/algo/sac.py](eve_rl/eve_rl/algo/sac.py) | §14 — name-based `Batch` access; IS-weighted Q/policy losses (guarded); per-sample `|TD|` extracted + stashed in `last_td_errors` |
| [eve_rl/eve_rl/agent/single.py](eve_rl/eve_rl/agent/single.py) | §14 — post-update `update_priorities(batch.indices, algo.last_td_errors)` (guarded); `_log_batch_samples` name-based `Batch` access |
| [eve_rl/eve_rl/runner/runner.py](eve_rl/eve_rl/runner/runner.py) | §15 — update budget driven by `exploration` steps only (was `heatup + exploration`); heatup/seeding no longer front-loads a ~150k-update catch-up block that diverged the critic |
| [launch_rcca_train.sh](launch_rcca_train.sh) | §9 — **New.** RCCA-only SAC docker launcher. §11 — `--save_heuristic_cache` added. §12 — `--min_success_rate 0.40`, `--snapshots mesh`. §13 — `--replay_mode step --hidden 256 256 --embedder_layers 0`. §14 — `--per` |
| [prune_training_snapshots.py](prune_training_snapshots.py) | §12 — **New.** Snapshot keep-policy pruner (all seed+eval, 10-best/10-worst per 100 explore) |
| `RL_IMPROV_9_CHANGES.md` | This document |
