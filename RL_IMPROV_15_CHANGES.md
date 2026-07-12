# RL Improv 15 — Changes Reference (Gen-3 → Gen-4 → RL_IMPROV_15)

All changes in this worktree **relative to the state documented in
[RL_IMPROV_10_CHANGES.md](RL_IMPROV_10_CHANGES.md)** (Plan v9 + v10: reward
overhaul, SOFA-restore-at-fork, per-state curriculum, the pid19116 sweep,
the carotid-siphon anatomy analysis). Everything here is **uncommitted work
on branch `rl_improv_15`** (HEAD = `c810959`); the last committed milestone
was Plan v12 (multi-target heatup harvest, commit `662d5d4`).

RL_IMPROV_10 ended with the diagnosis that killed every prior ceiling:
**per-state difficulty dominated policy quality** (pid19116 = 86% from a
good start; siphon band = 46%), and **entropy/action-mean saturation** was
the unfixed stabilization gap (`log_std_min` floors the Gaussian term, not
the tanh-Jacobian). This document covers the three iterations that followed,
each building on the last:

| Gen | Theme | One line |
|---|---|---|
| **Gen-3** | Observation layout expansion | guidance 30→39, flat obs 78→87 — the intermediate widening that invalidated the 78-dim seed caches (§below). A stepping stone, not the redesign. |
| **Gen-4** | **Mesh generalization** | the centerpiece: mesh-invariant observation (no absolute-coordinate/mesh-identity leak), asymmetric actor-critic (policy 97 / privileged critic 121), aux privileged distillation, per-worker **procedural anatomy** (each of 16 workers sees a distinctly-varied RCCA/siphon), per-worker target RNG, and **recovery training** (relaxed failure truncations + anti-buckle potential shaping) so the policy learns to retract/unbuckle/re-approach instead of being killed for entering a bad state. |
| **RL_IMPROV_15** | **Freeze-collapse fixes** (recent) | the Gen-4 run `rcca_procedural_v1` learned to 34% explore / 13.3% eval, then **froze** (deterministic policy → 0.95 mm/s) and IPC-deadlocked. A 5-agent forensic established the exact causal chain; the fix package (F1–F8 below) closes every link, plus a post-pretrain baseline eval. The fixed run `rcca_procedural_v2` reached **30.6% held-out at its first eval** (2.3× v1's all-time peak). |

**Provenance.** The Gen-4 and RL_IMPROV_15 work implements two external
review documents, mapped finding-by-finding in **Part A**:
`eve_rl_deep_review_combined.pdf` (algo/infra: AWAC/SAC/alpha/log_std/
exploration-noise/optimizer/replay/worker-sync) and
`eve_rl_multimesh_generalization_combined.pdf` (mesh-coupling audit, obs
mesh-invariance audit, asymmetric-critic + recovery design, the
path-cued-navigation redesign).

**Standing constraint (unchanged from RL_IMPROV_9 §17 / RL_IMPROV_10):**
reward / observation / terminal surfaces are **frozen** — changes need
explicit approval, and **nothing may alter `is_on_correct_path()` /
`_on_planned_path` / `_current_branch_idx`**, because the heuristic policy
READS `is_on_correct_path()` to gate its RCCA steering. Every reward/obs
change below was explicitly approved and either *reads* the branch
classifier (never writes it) or is a pure additive channel. The Gen-4 obs
redesign and the recovery-reward shaping are the approved exceptions;
they are called out where they touch a frozen surface.

---

## 0. Gen-3 — the observation-layout precursor (context, mostly pre-this-worktree)

Before the Gen-4 mesh-invariant redesign, an intermediate **Gen-3** obs
expansion widened `LocalGuidance` (guidance **30 → 39** features) and the
flat observation (**78 → 87** dims). Its main live consequence in this
worktree is a **cache-compatibility guard**: the old 78-dim seed archives
(e.g. `lcca_awac_seed_v1.npz`) are stale and must be rejected/upgraded, and
the offline **buffer-filter obs-layout map was corrected** from the Gen-3
layout to Gen-4:

- [training _scripts/DualDeviceNav_train.py:737](training%20_scripts/DualDeviceNav_train.py#L737) — a cache built under the Gen-3 obs change (guidance 30→39, flat 78→87) is layout-incompatible; the loader guards against a stale 78-dim seed ([:993](training%20_scripts/DualDeviceNav_train.py#L993)).
- [training _scripts/util/buffer_filter.py:94](training%20_scripts/util/buffer_filter.py#L94) — the obs-layout map was stale since Gen-3; corrected here for Gen-4 offline archives.
- [eve_bench/eve_bench/dualdevicenav.py:165](eve_bench/eve_bench/dualdevicenav.py#L165) — a Gen-3 re-harvest note (invalid old layout).

Gen-4 supersedes the Gen-3 layout entirely (flat obs → **121**); Gen-3 is
documented only so the cache-version guards and the "78/87 vs 121" numbers
in the code are legible.

---

# Part A — Provenance: the two source reviews → implementation

This maps every concrete finding/recommendation in the two source review
documents to what was implemented on branch `rl_improv_15`. **Status was
verified by reading the working-tree diffs and current code**, not the
change summary. Line anchors point at the implementing code. Deferred and
Partial rows are as load-bearing as Done ones — they record what was
consciously *not* taken.

Source documents:
- `eve_rl_deep_review_combined.pdf` — algo/infra deep review (findings B/C/D/E/F/G/H/I/J/K + harness Part 3 + env/reward Part 4).
- `eve_rl_multimesh_generalization_combined.pdf` — multi-mesh generalization report (Part 1 mesh-coupling audit, Part 2 obs-invariance audit, Part 3 asymmetric-critic + recovery design, Part 4 path-cued redesign).

---

## A.1 Deep Review — core algorithm/infra findings

| ID | Finding (1 line) | Severity/type | Implemented? | Where (file:concept) |
|----|------------------|---------------|--------------|----------------------|
| A2 | `reward_scaling` stored but never applied to the Bellman target (dead knob) | BUG | **Done** | [sac.py](eve_rl/eve_rl/algo/sac.py#L699) `expected_q = rewards*reward_scaling + …`; same in [iql.py](eve_rl/eve_rl/algo/iql.py#L420) |
| A1 | `max_steps`/`sim_error` grounded as absorbing → late healthy states get pessimistic Q | CONCEPTUAL | **Deferred** (design chose penalty over bootstrap; recovery episodes now *intentionally* run to max_steps) | grounding kept in [env5.py](training%20_scripts/util/env5.py) |
| B1 | `log_prob.clamp(min=-20)` zeroes BC gradient on far demos (self-reinforcing freeze) | CONCEPTUAL/BUG-risk | **Done** | [gaussianpolicy.py](eve_rl/eve_rl/network/gaussianpolicy.py#L112) leaky floor (5% grad −20→−220, hard −30) + `nan_to_num` |
| C1 | hard `torch.clamp` on log_std → sticky rails, dead entropy gradient (**the exact v1 mechanism**) | CONCEPTUAL (High) | **Done** | [gaussianpolicy.py](eve_rl/eve_rl/network/gaussianpolicy.py#L66) tanh rescale `log_std_min + 0.5·(max−min)·(tanh(x)+1)` |
| F1 | `log_alpha` unbounded → alpha runaway under AWAC + capped entropy ceiling | BUG-risk (High) | **Done** | [sac.py](eve_rl/eve_rl/algo/sac.py#L453) configurable `log_alpha_min/max` clamp |
| F2 | masked `log_pi` biases the alpha loss (episode-mode) | BUG | **Done** | [sac.py](eve_rl/eve_rl/algo/sac.py#L456) masked mean |
| F3 | SAC branch doesn't detach alpha in policy loss | NIT | **Done** | [sac.py](eve_rl/eve_rl/algo/sac.py#L524) `self.alpha.detach()` |
| F4 | stale `self.alpha` on first update after warm start | BUG (minor) | **Done** | [sac.py](eve_rl/eve_rl/algo/sac.py#L346) re-derive from `log_alpha.exp()` |
| F5 | alpha-loss form + `target_entropy=-n_actions` correct | Checked-correct | N-A (enhanced: `target_entropy` now configurable native float, [sac.py](eve_rl/eve_rl/algo/sac.py#L253)) | — |
| H4 | exploration noise a single scalar shared across all 4 dims, unclipped | BUG | **Done** | [sac.py](eve_rl/eve_rl/algo/sac.py#L290) per-dim `normal(0,σ,size=action.shape)` + `clip(±1)`; also [iql.py](eve_rl/eve_rl/algo/iql.py#L224) |
| J2 | MLP has two stacked Linears with no ReLU between → 1 hidden layer, not 2 | INEFFICIENCY (every run) | **Done** | [mlp.py](eve_rl/eve_rl/network/component/mlp.py#L83) `F.relu(self._input_layer(obs))` |
| J1 | q2/policy optimizers exclude the shared `q1_embedder` | BUG (latent) | **Partial** — policy got its OWN embedder (needed for asymmetric critic); q2 still shares q1's. No-op at `embedder_layers=0` (current runs) | [agent.py](training%20_scripts/util/agent.py#L281) |
| B2 | padding-mask bias pollutes `_awac_weight_*` (episode-mode only) | BUG (minor) | **Deferred** (runs are step-mode, `padding_mask=None`) | — |
| B4 | 3 redundant policy forwards / update | INEFFICIENCY | **Deferred** | — |
| D1–D5 | PER stale-index / β-anneal unit / IS-norm / leaf-boundary correctness tier | BUG (rare) / CONCEPTUAL / NIT | **Deferred** (don't bite step-mode AWAC) | — |
| E2/E3 | autograd graph built outside `no_grad` (tiny) | INEFFICIENCY | **Deferred** | — |
| G1 | `balanced_fraction` proportions correct | Checked-correct | N-A (bonus: new `EVE_CLEAN_RAIL_MAX` clean-lane anti-rail filter) | [pervanillastep.py](eve_rl/eve_rl/replaybuffer/pervanillastep.py#L117) |
| H1 | workers run ~100 eps on frozen weights | CONCEPTUAL | **Deferred** | [synchron.py](eve_rl/eve_rl/agent/synchron.py) |
| H3 | weight-distribution redundant serialization + fixed sleeps | INEFFICIENCY | **Deferred** | — |
| I1 | `env_train_config` saved from `env_eval` | BUG | **Deferred** | — |
| I2 | buffer save deadlocks non-PER runs at first eval | BUG (latent) | **Deferred** (a generic IPC-timeout guard was added, [singelagentprocess.py](eve_rl/eve_rl/agent/singelagentprocess.py#L616), mitigating a related hang class) | — |
| I3 | resume never restores replay buffer / runner state | GAP | **Deferred** | — |
| I5/J3/K2 | `stochastic_eval` dropped in eval / `lstm.py` broken / double `forward_play` | NIT / dead / — | **Deferred** | — |
| J4–J7 | diagnostics `.item()` sync storm / dead code / comment nits | INEFFICIENCY/NIT | **Deferred** | — |

### A.1b Deep Review — harness (Part 3) + env/reward/geometry (Part 4) items acted on

| ID | Finding (1 line) | Type | Implemented? | Where |
|----|------------------|------|--------------|-------|
| P3-B2 | All 16 workers draw the IDENTICAL explore-target sequence (deep-copied `_rng`) | BUG (High) | **Done** | [singelagentprocess.py](eve_rl/eve_rl/agent/singelagentprocess.py#L50) `_reseed_env_target_rng()` from `(base, worker_id, pid)` |
| P3-B4 | `log_std_min/max` threaded into only 1 of 3 `GaussianPolicy` sites | BUG (latent) | **Done** | [agent.py](training%20_scripts/util/agent.py#L88) + [:517](training%20_scripts/util/agent.py#L517) |
| P3-B1 | Eval-seed loss on worker restart corrupts 98-seed Quality denominator | BUG | **Deferred** | — |
| P4-R1 | Target-daughter 2× forward-only progress doubling is jitter-farmable (not potential-based) | BUG | **Done** | [arclengthprogress.py](eve/eve/reward/arclengthprogress.py#L126) reverted to symmetric 1× |
| P4-R2 | `max_steps` timeout carries no penalty → stall-with-banked-reward locally optimal | BUG | **Done** | [env5.py](training%20_scripts/util/env5.py#L99) `MAX_STEPS_PENALTY=-3.0` |
| P4-G1 | Global nearest-point projection not hairpin-safe → siphon arclength discontinuity | BUG | **Done** | [polyline.py](eve/eve/util/polyline.py#L94) local-window `project_onto_polyline` |
| P4-T1 | Success reachable near-ostium without deep daughter entry | BUG | **Done** (mechanism) | [centerlinerandom.py](eve/eve/intervention/target/centerlinerandom.py#L79) `min_arclength_from_start` |
| P4-O3 | No catheter-tip observability | CONCEPTUAL | **Partial** — catheter tip in privileged critic tail + gw-cath gap in guidance; policy still lacks explicit catheter cross-track | [meshinvariant.py](eve/eve/observation/meshinvariant.py#L242) |
| P4-R3/A1 | State-machine one step behind / asymmetric translation-limit neutral bias | BUG/CONCEPTUAL | **Deferred** | — |

**What we took vs. left (Deep Review).** We took the whole entropy/saturation
cluster the review flagged as highest-bite — the sticky `log_std` clamp (C1),
unbounded `log_alpha` (F1), the BC-killing `log_prob` floor (B1), the
alpha-loss masking/staleness/detach bugs (F2/F3/F4), correlated exploration
noise (H4), the capacity-halving MLP ReLU (J2) — plus the reward/geometry
traps that cap success (R1 jitter-farm, R2 unpunished timeout, G1 hairpin
projection) and the two harness data-diversity bugs (explore-target RNG
P3-B2, log_std threading P3-B4). We deliberately left the entire
PER-correctness tier (D1–D5), the throughput inefficiencies (B4, E2/E3,
H1/H3, J4), and the checkpoint/resume gaps (I1–I3, I5) — none bite the
current step-mode AWAC runs. J1 is half-fixed (policy got its own embedder;
q2 still shares q1's — harmless at `embedder_layers=0`).

---

## A.2 Multi-Mesh Generalization Report

| Part/Item | Recommendation | Implemented? | Where |
|-----------|----------------|--------------|-------|
| Part 1 — mesh-coupling audit | Reward/state-machine already de-hardcoded; couplings are naming + constants + heuristics | N-A (verified portable) | [pathcontext.py](eve/eve/util/pathcontext.py) |
| Part 1 — per-worker mesh mechanism | Replace `deepcopy(env_train)` with `env_factory(i)`; forward through agent | **Done** | [synchron.py](eve_rl/eve_rl/agent/synchron.py#L714) `_env_train_factory`; [agent.py](training%20_scripts/util/agent.py#L217); `procedural_env_factory` [DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py#L408) |
| Part 1 — "3 train + 1 held-out" authored VMR meshes | Author static meshes; per-worker round-robin | **Re-scoped** → per-worker **procedural RCCA re-meshing** of the loaded arch (obs-compatible, warm-startable); `env_train_factory` keeps authored meshes a drop-in later | [rccavariedfrommesh.py](eve/eve/intervention/vesseltree/rccavariedfrommesh.py); [dualdevicenavrccavaried.py](eve_bench/eve_bench/dualdevicenavrccavaried.py) |
| Part 2 A.1 — tracking (dominant leak): absolute bbox-normalized wire shape | Tip-relative + fixed-mm-scale | **Done** | [meshinvariant.py](eve/eve/observation/meshinvariant.py#L31) `TipRelativeTracking2D` |
| Part 2 A.2 — target absolute bbox coord (mesh fingerprint) | Delete; clipped tip→target offset | **Done** | [meshinvariant.py](eve/eve/observation/meshinvariant.py#L97) `TargetTipOffset2D` |
| Part 2 A.3 — `inserted_lengths ÷ 900` per-mesh odometer | Replace with gw-slack + gw−cath gap | **Done** — component removed, moved into guidance (slack 43, gap 32) | [env5.py](training%20_scripts/util/env5.py#L383) |
| Part 2 A.4 — `d_rem_norm` route-fraction index | Remaining arclength in mm (log / ÷400 honest scale) | **Done** | [localguidance.py](eve/eve/observation/localguidance.py#L139) log-depth d_rem |
| Part 2 — dead phase one-hots 13–16 | Delete, reclaim dims | **Done** — replaced by path-preview | [localguidance.py](eve/eve/observation/localguidance.py#L39) |
| Part 2 B1 — multi-point path preview (highest value) | K points Δ∈{10,20,40,80}, `(p(s+Δ)−tip)/(Δ+50)` image-2D | **Done** | [localguidance.py](eve/eve/observation/localguidance.py#L770) `_compute_path_preview` (feats 13–16 + 39–42) |
| Part 2 B2 — junction take-off geometry | Per-junction take-off vectors/angles | **Partial** — straight-line entry dir retained; explicit take-off angle not added | [localguidance.py](eve/eve/observation/localguidance.py) `_entry_direction` |
| Part 2 B3 — local vessel radius now/ahead + clearance margin | radius now + s+Δ; `cross_track/tolerance` | **Done** | [localguidance.py](eve/eve/observation/localguidance.py#L107) radius `/12` |
| Part 2 C — buckle/stall awareness | 5–6 mesh-invariant buckle feats | **Done** — gw_slack (43) + gap (32) in guidance; contact/rotations/catheter in privileged tail (action-mask flags not surfaced to policy) | [meshinvariant.py](eve/eve/observation/meshinvariant.py#L192) |
| Part 2 D — wire-shape tip-relative representation | `Tracking2D→RelativeToFirstRow→Normalize(±135)→Memory(2)` | **Done** (inside `TipRelativeTracking2D`) | [meshinvariant.py](eve/eve/observation/meshinvariant.py#L31) |
| Part 3 §1–3 — privileged-critic plumbing (Option A: append `privileged` key last) | env5 tail; policy slices `[…,:n_obs_policy]`; agent sizes policy = total − priv | **Done** | `PrivilegedState` N_DIMS=24; [gaussianpolicy.py](eve_rl/eve_rl/network/gaussianpolicy.py#L60) slice; [agent.py](training%20_scripts/util/agent.py#L222) (**policy 97 / critic 121**) |
| Part 3 §1 — privileged vector contents | forces/velocities/contact proxy/rotations/catheter tip/branch one-hot/counters | **Done** | [meshinvariant.py](eve/eve/observation/meshinvariant.py#L148) node vel/force/argmax, `position−free_position` contact proxy, branch one-hot |
| Part 3 §3 — J1 (policy own embedder) | Give policy its own embedder | **Done** | [agent.py](training%20_scripts/util/agent.py#L262) |
| Part 3 §4 — recovery: stop truncating fold/off-path; run to `max_steps` | Delete those truncations, keep counters as features | **Done** | [env5.py](training%20_scripts/util/env5.py#L238) `relax_failure_truncations` |
| Part 3 §4b — stuck-state restore pool | Buckle-harvester + restore wrapper | **Done** | `_save_stuck_checkpoint` [env5.py](training%20_scripts/util/env5.py); [screen_stuck_pool.py](training%20_scripts/screen_stuck_pool.py) + [escapability.py](training%20_scripts/util/escapability.py); mesh-pinned restore [DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py#L411) |
| Part 3 §5 — auxiliary heads (predict privileged from deployable obs) | `n_aux` MSE head, no separate optimizer | **Done** | [gaussianpolicy.py](eve_rl/eve_rl/network/gaussianpolicy.py#L22) `n_aux`; [sac.py](eve_rl/eve_rl/algo/sac.py#L569) `aux_coef` MSE (labels 2,3,5,6 = force mean/max, contact mean/max) |
| Part 4 pillars 1–3 | obs redesign + learned recovery + privileged critic | **Done** | (as Part 2/3 rows) |
| Part 4 §3 — drop the non-telescoping −0.007/step off-path tax | Remove so recovery isn't net-negative | **Substituted** — tax kept; potential-based **anti-buckle** shaping added instead (telescopes, positive on unbuckle) | [buckle_reward.py](training%20_scripts/util/buckle_reward.py); `buckle_reward_coef` [env5.py](training%20_scripts/util/env5.py#L263) |
| Part 4 §5/§6 — bundle obs+reward breaks + reward-version stamping | Guard caches | **Done** | [experience_cache.py](eve_rl/eve_rl/util/experience_cache.py#L34) `meta_buckle_coef` stamp; `EVE_RL_BUCKLE_COEF` guard [DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py#L759) |

**What we took vs. left (Multi-Mesh).** All three redesign pillars were taken
essentially intact: the full observation de-leaking (tip-relative tracking,
tip→target offset, odometer removal, log-depth d_rem, dead-one-hot removal,
K-point path preview, radius/clearance), the entire privileged-critic
Option-A plumbing (privileged tail + policy slice + asymmetric embedder + aux
heads, at policy 97 / critic 121), and the recovery machinery (truncation
relaxation, stuck-pool harvest with an escapability screen, anti-buckle
potential shaping). Re-scoped: the multi-mesh mechanism was built as
**per-worker procedural RCCA re-meshing** of the loaded arch instead of "3
authored VMR meshes + 1 held-out" — the `env_train_factory` hook is generic,
so authored meshes remain a drop-in. Declined: the explicit junction
take-off-angle feature (B2, straight-line entry retained), and dropping the
−0.007 off-path tax (kept, with the loop-neutral buckle potential added to
price recovery instead). One addition present in **neither** review: the
`EVE_CLEAN_RAIL_MAX` clean-lane anti-rail admission filter (keeps bang-bang
"railed" successes out of the amplified AWAC clone lane) — added from the
RL_IMPROV_15 forensic (§below), not from either PDF.

**Load-bearing honesty caveats:** J1 is Partial (q2 still shares q1's
embedder); A1 was declined in favor of the max_steps penalty; the entire
PER-correctness tier (D1–D5) is Deferred; the authored 3+1 mesh plan was
replaced by procedural RCCA variation.

---

# Part B — Gen-4 mesh-invariant observation & asymmetric actor-critic

## 1. Gen-4 mesh-invariance — remove every absolute-coordinate / mesh-identity channel from the obs

**Context.** The prior obs stack leaked mesh identity three ways: (a) `NormalizeTracking2DEpisode` normalized the wire polyline by the **mesh's own bounding box** (size/aspect/absolute placement — anisotropic per mesh, the dominant memorization channel); (b) the absolute bbox-normalized target coordinate is a mesh fingerprint (on one mesh the 4 daughters form 4 recognizable clusters = a route key); (c) LocalGuidance feature 0 was `remaining_arclength / total_length`, a per-(mesh,route) **progress index** — the same fraction meant different mm on different meshes and let the policy learn "at f=0.4 turn left". Gen-4 replaces all three with local / path-relative / topological features so nothing in the deployable obs encodes *which* mesh the wire is in — a prerequisite for the per-worker procedural-anatomy training (multi-mesh generalization workstream).

**The change (scope).** New module [eve/eve/observation/meshinvariant.py](eve/eve/observation/meshinvariant.py) (`TipRelativeTracking2D`, `TargetTipOffset2D`, `PrivilegedState`), exported from [eve/eve/observation/__init__.py](eve/eve/observation/__init__.py#L17); a 30→51-dim LocalGuidance rewrite; a new `PrivilegedState` tail for an asymmetric critic; and an aux distillation head. The obs stack is rewired in env5 ([training _scripts/util/env5.py](training%20_scripts/util/env5.py#L345-L400)).

**Standing-rule note.** This is an **obs change** (frozen surface — done under the approved multi-mesh design). Every new feature *reads* the state-machine getters (`is_on_correct_path()`, `get_local_tolerance()`, `get_local_radius()`, `classify_physical_branch()`) but **none alter them**; the branch classifier the heuristic gates on is untouched. Where a feature's semantics changed (feature 0, feature 7) it is called out below.

---

## 2. LocalGuidance: 30 → 51 dims, mesh-invariant feature set

**File:** [eve/eve/observation/localguidance.py](eve/eve/observation/localguidance.py) (409-line diff). `self.obs` widened `np.zeros(30)` → `np.zeros(51)`; `space()` low/high vectors extended to 51 ([localguidance.py](eve/eve/observation/localguidance.py#L278)). Docstring header "28-dimensional" → "51-dimensional".

**Two re-semanticized features (indices unchanged):**

| idx | was | now | why |
|---|---|---|---|
| 0 `d_rem…` | `remaining/total` route-fraction [0,1] | `d_rem_mm_norm` = remaining mm, clipped 400, /400 ([L360](eve/eve/observation/localguidance.py#L360)) | honest physical scale; route-fraction was a per-(mesh,route) progress index |
| 7 `on_correct_…` | `on_correct_branch` (debounced nearest-branch classifier) | `on_correct_path` = `path_context.is_on_correct_path()` — the **same signal the reward gates on** ([L397](eve/eve/observation/localguidance.py#L397)) | policy must see the reward's classifier, not a disagreeing one; reads the getter, does not modify it |

**Removed:** `_phase_to_onehot()` helper (mapped heuristic phase strings → one-hot at features 13-16). During RL those slots were a constant "default"; when a heuristic drove they became **anatomy-bound tokens** (a memorization channel). Freed for transferable geometry.

**21 new / repurposed features (all path-relative or dimensionless):**

| idx | feature | definition | mesh-invariant because |
|---|---|---|---|
| 13-16 | `preview_10/20_{x,z}` | planned point at s+10, s+20 mm, tip-relative in 2D image frame, /(δ+50) | replaces "I recognize where I am so I know the bend" with "the path bends left in 20 mm"; chord ≤ arc so bounded |
| 30-31 | `cath_offset_{x,z}2d` | (catheter tip − guidewire tip) 2D, /150 | device-to-device relative |
| 32 | `gw_cath_gap_norm` | (inserted_gw − inserted_cath), /150 | the two-device coordination scalar |
| 33-36 | `fork_planned_dir_{x,z}`, `fork_sister_dir_{x,z}` | planned vs sister takeoff unit dirs at next junction, 2D via `_entry_direction(origin, dir, 0.0, rot_zx)` | unit vectors in image frame; from `path_context.get_next_junction_fork_geometry()` ([L596](eve/eve/observation/localguidance.py#L596)) |
| 37-38 | `heading_dot_{planned,sister}` | dot(device heading 2D, takeoff 2D) | scale-free fork-commit disambiguator |
| 39-42 | `preview_40/80_{x,z}` | far path-preview pair, /(δ+50) | as 13-16 |
| 43 | `gw_slack_norm` | (inserted_gw − proj.s), /50 — wire length stored in bowing, **the buckle scalar** | both operands on this episode's path |
| 44 | `slip_norm` | last-step (Δgw − Δs), /4 — raw continuous fold-detector | env5 mirrors `_env_slip_mm` |
| 45-46 | `gw/cath_action_masked` | 1 if translation cmd was masked (`last_cmd_action`≠`last_exec_action`) | LastAction carries only pre-mask cmd; policy couldn't tell its push was zeroed at tree-end/floor |
| 47-48 | `local_radius_norm`, `radius_ahead_norm` | vessel calibre now / 20 mm ahead, /12 (`_MAX_LOCAL_RADIUS_MM`) | calibre preview |
| 49 | `clearance_norm` | cross_track / local_tolerance / 2 (0.5 = at the radius-aware tol edge) | dimensionless; transfers across vessel sizes far better than raw-mm feature 1 |
| 50 | `d_rem_log_norm` | log1p(remaining mm)/log1p(1000) ([L365](eve/eve/observation/localguidance.py#L365)) | log-scaled companion to feature 0: ~11× the per-mm sensitivity at 5 mm remaining (the siphon endgame), unsaturated to 1000 mm (feature 0 rails at 400) |

Features 1-6, 8-12, 17-29 carry over unchanged (tangents, heading, curvature, correct-entry dir, arc-to/past-daughter, bend_hat, off-arc, wrong-branch/ostium/recovery block).

**New helper `_compute_path_preview(s, tip_vessel, rot_zx)`** ([L770](eve/eve/observation/localguidance.py#L770)): for each δ in `_PREVIEW_DELTAS_MM = (10,20,40,80)` ([L141](eve/eve/observation/localguidance.py#L141)) interpolates the planned polyline at s+δ (saturating at path end), forms the vessel-CS tip delta, rotates to tracking3d with **zero image-center so translation cancels**, drops Y, scales 1/(δ+50), clips [-1,1]. Returns an 8-vector feeding 13-16 and 39-42. New scale constants: `_MAX_D_REM_MM=400`, `_MAX_D_REM_LOG_MM=1000`, `_MAX_SLACK_MM=50`, `_MAX_SLIP_MM=4`, `_MAX_LOCAL_RADIUS_MM=12` (matches pathcontext `MAX_RADIUS_CEILING_MM`).

**Rationale.** Every added feature is either device-relative, path-relative, or a dimensionless ratio; the mesh bounding box, absolute position, and route-fraction index no longer enter the policy's obs. Fork-disambiguation (33-38) and buckle/slip (43-46) also close two Markov gaps that were previously only observable through the counters or not at all.

---

## 3. `TipRelativeTracking2D` — wire shape at a fixed mm scale (replaces `NormalizeTracking2DEpisode`)

**File:** [eve/eve/observation/meshinvariant.py](eve/eve/observation/meshinvariant.py#L30). Wraps a `Tracking2D` and re-expresses the wire polyline relative to the tip at a **fixed mm scale**: row 0 = per-step tip displacement `(tip − prev_tip)/delta_scale_mm` (default 4.0 — restores the rigid-translation velocity that a tip-relative frame otherwise loses under Memory stacking); rows 1..N-1 = `(p_i − tip)/offset_scale_mm` (default 135.0). Offsets are hard-bounded by the tracking resolution (`|p_i − p_0| ≤ resolution·i`), so the fixed scale is exact and **identical for every mesh** and **isotropic** (no per-mesh angle/curvature distortion). `reset()` seeds `_prev_tip` with the reset tip so row 0 reads 0 at episode start.

**Wiring (env5):** `Tracking2D(n_points=10, resolution=15)` → `TipRelativeTracking2D` → `Memory(n_steps=2, FILL)` ([env5.py](training%20_scripts/util/env5.py#L345-L351)) = **40 dims** (10 pts × 2D × 2 frames). Replaces the mesh-bbox episode normalizer whose reference quantities *were* the mesh's size/aspect/placement.

---

## 4. `TargetTipOffset2D` — clipped tip→target offset (replaces absolute bbox target coord)

**File:** [eve/eve/observation/meshinvariant.py](eve/eve/observation/meshinvariant.py#L88). Emits `clip((target2d − tip2d)/clip_mm, ±1)`, `clip_mm=50`. Far targets saturate at ±1 (routing is the guidance block's job); the offset only sharpens near-target homing. Preserves the grader-mode `target_coord3d` override (env5 updates `.target_coord3d` per grader episode via the same attribute contract as the old `Target2D`) — [env5.py](training%20_scripts/util/env5.py#L360). **2 dims.** Replaces the absolute bbox-normalized target coordinate (a mesh-identity fingerprint). Note env5 also **drops** the absolute `InsertionLengths` pair (a per-mesh odometer); its physical content now lives in guidance features 43 (gw slack) and 32 (gw-cath gap).

---

## 5. `PrivilegedState` — 24-dim privileged tail for the asymmetric critic

**File:** [eve/eve/observation/meshinvariant.py](eve/eve/observation/meshinvariant.py#L138) (`N_DIMS=24`, `_TIP_NODES=3`). Sim-side state a deployed system could not have, so **only the critics may consume it**. Every accessor is exception-guarded (any failure → 0 for that block, never a crash inside `env.step`). Appended **LAST** in the ObsDict so it rides inside `flat_obs` through buffer/PER/caches with zero plumbing changes; the policy slices it off internally (§7).

| idx | signal | scale |
|---|---|---|
| 0 | mean \|node velocity\| over K=3 tip-most nodes | /50 mm/s |
| 1 | max \|node velocity\| (all nodes) | /50 |
| **2** | log1p(mean \|node force\|)/10 | log |
| **3** | log1p(max \|node force\|)/10 | log |
| 4 | arg-max-force position along wire (0=proximal…1=tip) | — |
| **5** | mean \|position − free_position\| (contact-impulse proxy) | /2 mm |
| **6** | max \|position − free_position\| | /10 mm |
| 7-10 | sin/cos of accumulated device rotations (gw, cath) | — |
| 11-13 | (catheter tip − guidewire tip) 3-D offset | /150 mm |
| 14-18 | physical-branch one-hot: [target daughter, other daughter, trunk, bridge/on-path-other, off/unknown] via `classify_physical_branch()` + `_env_target_branch_short` | — |
| 19 | off-branch step counter | /50 |
| 20 | fold-stall counter | /20 |
| 21 | slip (`_env_slip_mm`) | /4 mm |
| 22 | cross-track excess beyond radius-aware tolerance | /10 mm |
| 23 | guidewire slack (inserted_gw − proj.s) | /50 mm |

Reads SOFA `DOFs` (`position/velocity/force/free_position`) off `simulation._instruments_combined`, plus env5-mirrored counters and the shared `path_context`. **Bold rows (2,3,5,6)** = the force/contact block the aux head distills (§9). **Wiring:** `PrivilegedState(intervention, path_context=self._path_context)` as the last ObsDict key ([env5.py](training%20_scripts/util/env5.py#L388-L400)). Serialization follows the LocalGuidance pattern: ctor stores `self.path_context=None`, caches the live object in `_path_context` (ConfigHandler getattr's every `__init__` param name).

---

## 6. The observation stack — policy 97 vs critic 121 (the 24-dim asymmetry)

**File:** [training _scripts/util/env5.py](training%20_scripts/util/env5.py#L392). The env's `ObsDict` (order matters — privileged is LAST):

| block | component chain | dims |
|---|---|---|
| `tracking` | Tracking2D(10, 15) → TipRelativeTracking2D → Memory(FILL, ×2) | 40 |
| `target` | TargetTipOffset2D | 2 |
| `last_action` | LastAction → Normalize | 4 |
| `guidance` | LocalGuidance | 51 |
| `privileged` | PrivilegedState (LAST) | 24 |
| **flat_obs (critic input)** | | **121** |
| **policy input = 121 − 24** | | **97** |

The **critic (twin-Q + target, and IQL V) consumes the full 121-wide flat obs**; the **policy is built 97 wide and slices the privileged tail off internally**. Because the tail is the flat-obs suffix, the buffer, PER, episode caches, and heuristic seeding all store/replay full width unchanged — the split is enforced at exactly two code points: the agent constructs the policy at `n_obs_policy` (§7) and `GaussianPolicy.forward` slices `[..., :n_observations]` (§8).

---

## 7. `env.py` post-step hook + asymmetric-width policy construction

**`_on_intervention_stepped` hook.** [eve/eve/env.py](eve/eve/env.py#L70) inserts a `self._on_intervention_stepped()` call in `Env.step()` **after** intervention/pathfinder/interim-target advance and **before** observation/reward/terminal compute (default no-op). env5 overrides it ([env5.py](training%20_scripts/util/env5.py#L923)) to `invalidate()` + `update_branch_state()` the path context, so same-step consumers — LocalGuidance feature 7 (`is_on_correct_path`), features 47-49 (calibre/clearance), ArcLengthProgress, the path-segment step reward — see **this** step's classification. **Standing-rule note:** this changes *when* the machine advances (now inside `super().step()`), but it still advances exactly once per step and the heuristic still reads it between steps, so the classifier's behavior is unchanged.

**Asymmetric policy construction.** [training _scripts/util/agent.py](training%20_scripts/util/agent.py#L43) computes `n_obs_policy = n_observations − privileged_obs_dim` in all three factories (`BenchAgentSingle`, `BenchAgentSynchron`, `create_bench_agent`) and builds `GaussianPolicy(policy_base, n_obs_policy, …)` ([L91](training%20_scripts/util/agent.py#L91), [L294](training%20_scripts/util/agent.py#L294), [L520](training%20_scripts/util/agent.py#L520)) while the critics keep full `n_observations`. `BenchAgentSynchron` raises if `n_obs_policy <= 0` ([L226](training%20_scripts/util/agent.py#L226)). Each factory now also builds a **separate `policy_embedder` instance** (LSTM/MLP/Dummy) instead of sharing `q1_embedder` — required because (a) with asymmetric widths a shared embedder would raise on the `n_inputs` re-set, and (b) the old sharing meant only q1's optimizer ever stepped it while all three losses fed it gradients. **Wiring:** `privileged_obs_dim = PrivilegedState.N_DIMS if env_version==5 else 0` ([DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py#L636)) → 24 for env5, 0 (symmetric/legacy) otherwise.

---

## 8. `GaussianPolicy.forward` — the privileged-tail slice + `n_aux` head

**File:** [eve_rl/eve_rl/network/gaussianpolicy.py](eve_rl/eve_rl/network/gaussianpolicy.py). Both `forward` and `forward_play` first execute `obs_batch = obs_batch[..., : self.n_observations]` ([L63](eve_rl/eve_rl/network/gaussianpolicy.py#L63), [L87](eve_rl/eve_rl/network/gaussianpolicy.py#L87)) — one chokepoint covering every caller (log_prob, SAC updates, exploration/eval, play-only workers), so no caller has to know the obs is wider than the policy. With `n_observations=97` the 24-dim privileged tail is dropped; the critics never call this path so they keep full width.

**`n_aux` head.** New ctor param `n_aux` ([L30](eve_rl/eve_rl/network/gaussianpolicy.py#L30)): when >0, the body's `output_layer_size` becomes `[n_actions, n_actions, n_aux]` ([L47](eve_rl/eve_rl/network/gaussianpolicy.py#L47)) — a third linear head off the last hidden layer whose output is stashed in `self._last_aux` every forward ([L68](eve_rl/eve_rl/network/gaussianpolicy.py#L68)). `n_aux=0` = byte-identical to before. Agent passes `n_aux=len(aux_label_abs)` ([agent.py](training%20_scripts/util/agent.py#L297)). (The same file's soft tanh `log_std` bound and leaky `log_prob` floor are AWAC-stabilization changes adjacent to this subsystem.)

---

## 9. Auxiliary privileged-label distillation — policy predicts labels {2,3,5,6} via MSE

**Context.** The privileged force/contact signals help the critic, but a deployed policy can't see them. The aux head trains the policy to **infer** contact/buckle state from its deployable prefix — representation shaping, not a leak.

**The change.** [eve_rl/eve_rl/algo/sac.py](eve_rl/eve_rl/algo/sac.py#L197) adds `aux_coef` and `aux_label_indices` (**absolute** flat-obs indices). In `_update_policy`, after the primary loss ([L582](eve_rl/eve_rl/algo/sac.py#L582)):
```
aux_pred   = self.model.policy._last_aux                 # stashed in the forward above
aux_labels = states.index_select(-1, idx).detach()       # true privileged values from full-width states
aux_se     = (aux_pred - aux_labels).pow(2)
policy_loss += self.aux_coef * aux_mse                   # padding-masked in episode mode
```
The labels are pulled from the **full-width `states`** (which still carry the privileged tail); the policy's own forward slices them off its input, so it only ever learns to predict them — it never receives them as input. `aux_coef=0` = off.

**Index plumbing + validation.** [training _scripts/util/agent.py](training%20_scripts/util/agent.py#L242): `aux_label_rel_indices` are relative to the privileged tail; converted to absolute `aux_label_abs = [n_obs_policy + i for i in rel]` after **construction-time validation** `0 <= i < privileged_obs_dim` ([L237](training%20_scripts/util/agent.py#L237)) — an out-of-range index would crash `index_select` hours into training and a negative index would silently supervise against a deployable column. **CLI:** `--aux_coef` / `--aux_labels` ([DualDeviceNav_train.py](training%20_scripts/DualDeviceNav_train.py#L1631)), parsed by `_parse_aux_labels` ([L16](training%20_scripts/DualDeviceNav_train.py#L16)). Launchers set `--aux_coef 0.05 --aux_labels "2,3,5,6"` — i.e. relative {2,3,5,6} → absolute {99,100,102,103} = mean/max node force + mean/max contact-impulse proxy (the force/contact block from §5).

---

## 10. `MLP.forward` — ReLU after the input layer

**File:** [eve_rl/eve_rl/network/component/mlp.py](eve_rl/eve_rl/network/component/mlp.py#L83). Resolves a long-standing `# TODO: Add F.relu after input layer`: `state = F.relu(self._input_layer(obs_batch))` (was a bare linear projection). Affects every MLP body/embedder — including the new FF-mode `policy_embedder` and the Q/policy bases — giving the input projection a nonlinearity instead of collapsing two adjacent linear maps. Behavioral change for any run using MLP components (not the default LSTM embedder path).


---

# Part C — Gen-4 procedural anatomy & per-worker mesh variation

## 1. `RCCAVariedFromMesh` — per-worker RCCA-only perturbation (the active Gen-4 tree)

**Context.** Gen-4's anti-memorization goal is to give each of the 16 workers a *different* RCCA→siphon anatomy that re-randomizes every N episodes, so the policy cannot overfit a single fixed mesh. The design constraint (learned from the prior fully-synthetic attempt, §5) is that the wire never navigates the arch/trunk — it starts at the RCCA fork — so ONLY the RCCA needs to vary; everything else can stay byte-for-byte the loaded `DualDeviceNav` geometry, keeping obs semantics and warm-startability intact.

**The change.** New [eve/eve/intervention/vesseltree/rccavariedfrommesh.py](eve/eve/intervention/vesseltree/rccavariedfrommesh.py) defines `RCCAVariedFromMesh(VesselTree)` ([rccavariedfrommesh.py#L133](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L133)). It takes the loaded branch list, keeps every non-RCCA `BranchWithRadii` verbatim, and replaces only the RCCA centerline with a per-generation perturbed copy from the module-level helper `perturb_rcca()` ([#L50](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L50)):

- **Bell/envelope displacement** ([#L86-L118](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L86)): a `_smoothstep` window `w = min(w_up, w_down)` that is **0 over the first `anchor_mm`** (pins the ostium + the (11)/RVA bridge junction) **and 0 over the last `distal_anchor_mm`** (pins the terminus + the distal Circle-of-Willis (13)/(24) junction), ramping to **full through the middle** — so only the cavernous-siphon course varies while both ends stay connected to the loaded tree. Displacement is a normalized sum of low-frequency sinusoids (`freqs=(0.7,1.3,2.1)`) projected onto a stable perpendicular basis (`_perpendicular_basis`, [#L39](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L39)), so the result is a plausibly tortuous vessel, not a kink. Amplitude = `base_amp_mm × tortuosity`; radii bell-blend toward `radius_scale` in mid-vessel (both ends' calibre preserved so junction radius-detection is undisturbed).
- **Per-generation sampling** in `_generate()` ([#L357](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L357)): draws `tortuosity` (clipped `tortuosity_clip`) and `radius_scale`, perturbs the RCCA with a **small `anchor_mm=self.fork_anchor_mm`** (so the takeoff/deflection at the fork itself varies, not just the mid-siphon) and `ramp_mm=15`.
- **RVA proximal co-perturbation** ([#L379-L390](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L379)): the sister RVA (shared (11) ostium) also gets `perturb_rcca` applied but with `distal_anchor_mm = max(5, rva_len − rva_proximal_mm)` so ONLY its proximal `rva_proximal_mm` takeoff varies (amplitude `rva_amp_mm`), while its shared s=0 ostium point stays pinned (`RCCA[0] == RVA[0]`). This makes the RCCA-vs-RVA fork deflection non-memorizable.

**Re-mesh in the same frame + temp cleanup.** After perturbation `_generate` rebuilds `branching_points` (`calc_branching_with_radii`), `centerline_coordinates`, `coordinate_space`, holds `insertion` fixed, and **deletes the prior generation's temp `.obj`** before nulling `_mesh_path` ([#L406-L413](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L406)) so a long 16-worker run doesn't leak one mesh file per regen into `gettempdir()`. The mesh itself is lazily rebuilt on `mesh_path` access via `generate_temp_mesh(...,"rcca_varied",decimate_factor)` (voxel→marching-cubes, watertight by construction — no surface boolean). Because the whole tree is re-meshed from the SAME loaded centerlines, it stays in the identical vessel-CS frame as `DualDeviceNav`.

**Insertion is at the (11) bridge, not literally the ostium** ([#L241-L274](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L241)). The docstring says "FIXED at the RCCA ostium," but the code inserts `start_point_offset` (default 2) points **into the (11) bridge** — the shared parent of RCCA and RVA — oriented from the far entry toward the fork. Rationale: the very first (11) point sits in the (0)/(11)/LCCA junction slightly outside the (11) lumen (inserting there put the wire outside the vessel and hit vessel-end immediately); stepping a couple points in clears the junction. Starting inside the bridge forces the wire to traverse it and DEFLECT into RCCA at the fork — re-enabling the +1/−1 daughter-commit signal. Falls back to the RCCA ostium if (11) is absent. The (11) bridge is not perturbed, so the start is identical across workers/generations.

**Rationale.** Isolates exactly the siphon-navigation problem under per-worker anatomy variation while preserving frame/obs compatibility with the fixed-mesh runs, so a fixed-mesh RCCA policy can warm-start this run.

## 2. Mesh identity — `s{seed}g{gen}` fingerprints + reproducible regeneration (recovery-restore matching)

**Context.** SOFA-restore checkpoints are mesh-bound state (`dof_positions` are only valid in the exact geometry they were captured on). To reuse the restore-curriculum crutch on a *regenerating* mesh, a checkpoint must be restorable ONLY into the identical siphon it came from — which requires the tree to reproduce any past geometry on demand.

**The change.** `RCCAVariedFromMesh` makes its current geometry uniquely reproducible from `(seed, generation)`:

- `_generation` ([#L223](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L223)) counts `_generate()` calls since the RNG was last (re)seeded; `self._rng` is used ONLY inside `_generate()`, so re-seeding and replaying `generation` calls recreates the identical geometry.
- `mesh_fingerprint` property ([#L299](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L299)) returns `f"s{self._seed}g{self._generation}"`.
- `parse_fingerprint()` ([#L304](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L304)) regexes `s(-?\d+)g(\d+)` → `(seed, gen)` or `None` for a non-RCCAVaried tag (e.g. a fixed-mesh checkpoint).
- `regenerate_to_fingerprint()` ([#L316](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L316)) re-seeds and replays `generation` `_generate()` calls; raises on an unparseable fingerprint.
- `pin_next()` ([#L333](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L333)) stores `_pending_fingerprint`; the next `reset()` ([#L338](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L338)) treats a pin as **authoritative** — regenerates to it and returns, ignoring both the per-episode seed and the `episodes_between_change` schedule for that episode.

**Rationale.** `CheckpointRestoreWrapper` reads the checkpoint's stored fingerprint and calls `tree.pin_next(fp)` → `regenerate_to_fingerprint(fp)` **before** the SOFA restore, so `dof_positions` land in the geometry they came from (see [checkpoint_restore.py#L205](training%20_scripts/util/checkpoint_restore.py#L205), env5's stuck-capture tags the mesh at [env5.py#L2083](training%20_scripts/util/env5.py#L2083)). Untagged/fixed-mesh checkpoints correctly fail to parse and are ineligible on a procedural tree, so the episode falls through to the ostium start. This is the mesh-SAFE version of Gen-4 (#3) restore.

## 3. `RCCAVariedFromMesh` ConfigHandler compliance

**Context.** This class lives in the `eve.*` namespace, so eve's `ConfigHandler.save_config` reflects over its OWN `__init__` signature and reads every parameter name back via `getattr(self, name)` (confighandler.py:146). A missing attribute would crash config serialization.

**The change.** The ctor stores a `self.<name>` for every signature arg, including params otherwise kept under private names ([#L189-L207](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L189)): `self.fork_anchor_mm`, `self.rva_amp_mm`, `self.rva_proximal_mm`, `self.start_point_offset`, plus mirrors `self.seed = self._seed`, `self.rcca_name`. The heavy `branch_list` is stored as a **`None` placeholder** ([#L237](eve/eve/intervention/vesseltree/rccavariedfrommesh.py#L237)) (the LocalGuidance `path_context` convention) so config records a placeholder instead of serializing every centerline — the real data stays in `self._loaded`. VesselTree interface attributes (`branches`, `branching_points`, `centerline_coordinates`, `insertion`, `visu_mesh_path`) are also initialized before `_generate()`.

**Rationale.** Keeps `save_config` working (needed for `get_env_from_checkpoint` reconstruction) without bloating the config with megabytes of centerline arrays.

## 4. `CenterlineRandom` near-start exclusion (`min_arclength_from_start`)

**Context.** With the varied-RCCA wire starting at the (11) bridge, a target sampled `< N` mm into the RCCA would be a trivial "deflect and stop" — it must instead sit in the siphon so the wire actually navigates the tortuous course.

**The change.** [eve/eve/intervention/target/centerlinerandom.py](eve/eve/intervention/target/centerlinerandom.py) adds a new optional ctor arg `min_arclength_from_start` ([#L20](eve/eve/intervention/target/centerlinerandom.py#L20), stored [#L34](eve/eve/intervention/target/centerlinerandom.py#L34)) and a helper `_arclength_from_start_mask()` ([#L76](eve/eve/intervention/target/centerlinerandom.py#L76)) that computes cumulative arclength along each branch's ordered points and keeps only points `≥ min_arclength_from_start` mm from that branch's first point (all-True when disabled — backward-compatible). The mask is applied in `_init_centerline_point_cloud` to BOTH the flat `potential_targets` pool ([#L99](eve/eve/intervention/target/centerlinerandom.py#L99)) and each per-branch pool ([#L111](eve/eve/intervention/target/centerlinerandom.py#L111), AND-combined with the excluded-branch mask).

**Wiring.** `DualDeviceNavRCCAVaried` passes `min_arclength_from_start=target_min_arclength_mm` (default **40.0** mm) into its `CenterlineRandom(branches=[_RCCA_NAME], ...)` ([dualdevicenavrccavaried.py#L92](eve_bench/eve_bench/dualdevicenavrccavaried.py#L92)).

**Rationale (frozen-rule note).** This is a target-sampling filter (which centerline points are *eligible*), not a reward/observation/terminal change and not a change to `is_on_correct_path()` — it only narrows the target pool so every target lands in the siphon band.

## 5. Fully-synthetic CHS siphon path (`carotidsiphon.py` + `RCCAProcedural`) — built but NOT wired to the active run

**Context.** The *original* Gen-4 design synthesized the WHOLE arch procedurally (aorta+BCT+arch-sisters + an RCCA extended through the carotid siphon), varying everything per worker. This path exists in the tree and is import/test-reachable, but the active launchers use the RCCA-only design (§1) instead — because a fully-regenerating arch invalidates every restore checkpoint and needlessly re-randomizes geometry the wire never touches.

**What was built.**
- New [eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py](eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py): `right_carotid_siphon()` ([#L131](eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py#L131)) generates the RCCA→RICA→cavernous-siphon→terminus chain as one `BranchWithRadii` from 9 cubic-Hermite control points (CP0 arch take-off … CP8 C7 terminus) encoding Bouthillier ICA segments C1–C7, including the two ~180° cavernous genua (CP5 posterior, CP6 anterior) that are the catheter-overshoot region. `SiphonParams` ([#L50](eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py#L50)) holds per-worker sampling ranges; `SampledSiphon` ([#L115](eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py#L115)) is one concrete draw via `SiphonParams.sample()`. A **meshing constraint** ([#L69-L79](eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py#L69)) clips `tortuosity` to `(0.6, 1.25)`: beyond that the voxel→marching-cubes mesher self-intersects the inner wall where centerline radius-of-curvature drops below the vessel radius (~2 mm at the genua). Exported from [aorticarcharteries/__init__.py](eve/eve/intervention/vesseltree/aorticarcharteries/__init__.py) (`right_carotid_siphon`, `SiphonParams`, `SampledSiphon`).
- New [eve/eve/intervention/vesseltree/rccaprocedural.py](eve/eve/intervention/vesseltree/rccaprocedural.py): `RCCAProcedural(VesselTree)` builds a Type-I arch (reusing `aorta_generator`, `brachiocephalic_trunk_static`, `right/left_subclavian`, `left_common_carotid`) and **swaps the common carotid for `right_carotid_siphon(..., name="RCCA")`** ([#L136](eve/eve/intervention/vesseltree/rccaprocedural.py#L136)), regenerating every `episodes_between_change` episodes with a constant registration `rotation_yzx_deg` (held fixed across generations for obs-invariance). Its docstring explicitly notes restore checkpoints CANNOT be reused here ([#L21-L26](eve/eve/intervention/vesseltree/rccaprocedural.py#L21)).

**Not active.** The `--procedural_rcca` code path in [DualDeviceNav_train.py#L437-L468](training%20_scripts/DualDeviceNav_train.py#L437) imports and instantiates `DualDeviceNavRCCAVaried`, NOT `DualDeviceNavProcedural` — even though the surrounding comment ([#L402](training%20_scripts/DualDeviceNav_train.py#L402)) still names "DualDeviceNavProcedural" (stale). No launcher wires the synthetic classes; `DualDeviceNavProcedural` is referenced only in an `agent.py` comment. Treat `carotidsiphon.py` / `rccaprocedural.py` / `DualDeviceNavProcedural` as the archived alternative, retained for future dataset-fit siphon work.

## 6. eve_bench wiring + symmetric velocity limits

**The change.**
- New [eve_bench/eve_bench/dualdevicenavrccavaried.py](eve_bench/eve_bench/dualdevicenavrccavaried.py) (`DualDeviceNavRCCAVaried`, [#L35](eve_bench/eve_bench/dualdevicenavrccavaried.py#L35)) — **the active Gen-4 bench**: loads the exact `DualDeviceNav` centerlines via the reused `load_branches`/`DATA_DIR` (same `(y,−z,−x)` frame), constructs `RCCAVariedFromMesh(branch_list=branches, rcca_name=_RCCA_NAME, seed=seed, ...)` ([#L52](eve_bench/eve_bench/dualdevicenavrccavaried.py#L52)), identical devices/sim/fluoroscopy to `DualDeviceNav`, and the near-ostium-excluded target (§4). Per worker: distinct `seed` → distinct RCCA anatomy sequence.
- New [eve_bench/eve_bench/dualdevicenavprocedural.py](eve_bench/eve_bench/dualdevicenavprocedural.py) (`DualDeviceNavProcedural`, [#L26](eve_bench/eve_bench/dualdevicenavprocedural.py#L26)) — wraps `RCCAProcedural`, full-trunk from the aortic root, no `insertion_z`/no restore. Built but unused (§5).
- [eve_bench/eve_bench/__init__.py](eve_bench/eve_bench/__init__.py) exports both new benches.
- **Symmetric velocity limits (review finding 2.4)** in [eve_bench/eve_bench/dualdevicenav.py](eve_bench/eve_bench/dualdevicenav.py): both `DualDeviceNav` ([#L156-L174](eve_bench/eve_bench/dualdevicenav.py#L156)) and `DualDeviceNavCustom` ([#L408-L418](eve_bench/eve_bench/dualdevicenav.py#L408)) now **omit `velocity_limit_low`**, so `MonoPlaneStatic` defaults it to `-velocity_limits` (`[[-30,-1.5],[-30,-1.5]]`). The old asymmetric `[-10,+30]` translation band put "policy output 0" at +10 mm/s forward on both devices (hold required `a=-0.5`), a built-in advance bias that suppressed retraction. Both new benches inherit this (they also omit `velocity_limit_low`). NOTE flagged in-code: this changes the `[-1,1]→mm/s` mapping, so buffers/demos recorded under the old space are invalid (Gen-3 re-harvest required).

**Rationale.** Makes `a=0` = hold and retraction as reachable as advance — a precondition for the harvest/exploration that feeds the Gen-4 runs.

## 7. Hairpin-safe windowed polyline projection (siphon support)

**Context.** The carotid siphon folds the planned-path polyline back on itself; a global `argmin` over all segments is ambiguous where the two hairpin limbs are spatially adjacent, so sub-mm tip jitter flips the winning segment between limbs and the projected arclength `s` jumps by a full loop length — corrupting every arclength-derived signal.

**The change.** [eve/eve/util/polyline.py](eve/eve/util/polyline.py) refactors the vectorized closest-point scan into `_project_onto_segment_range(point, polyline, cumlen, i0, i1)` ([#L82](eve/eve/util/polyline.py#L82)) that scans only segments `[i0, i1)` but returns GLOBAL `segment_idx`/`s`. `project_onto_polyline()` ([#L131](eve/eve/util/polyline.py#L131)) gains optional `prev_s`, `window_mm`, `fallback_dist_mm=15.0`: when `prev_s`/`window_mm` are given ([#L186](eve/eve/util/polyline.py#L186)) it restricts the scan to segments overlapping `[prev_s−window_mm, prev_s+window_mm]` (via `searchsorted`), keeping consecutive projections continuous across the hairpin. **Fallback rule**: if the windowed best cross-track exceeds `fallback_dist_mm`, the point has genuinely left the neighborhood (elastic snap / SOFA restore) and continuity yields to correctness — the full scan runs. Both defaults `None` preserve exact legacy full-scan behavior for all other callers.

**Rationale (frozen-rule note).** Engineered as a no-op outside hairpins: the 30 mm window (§8) far exceeds per-step tip travel (~4 mm) and the 15 mm fallback is below the true-nearest cross-track in normal geometry, so the windowed argmin equals the pre-change global argmin everywhere the polyline doesn't fold. It only diverges from the old result inside the tight siphon (where the global scan was the bug), not at the RCCA fork the heuristic's `is_on_correct_path()` reads.

## 8. `PathProjectionCache` re-mesh / continuity support

**Context.** The windowed projection (§7) needs a per-step arclength anchor that survives the every-step `invalidate()` but resets on episode boundaries and mid-episode SOFA teleports.

**The change.** [eve/eve/util/pathcontext.py](eve/eve/util/pathcontext.py):

- New constants `PROJECTION_WINDOW_MM = 30.0` / `PROJECTION_FALLBACK_DIST_MM = 15.0` ([#L173](eve/eve/util/pathcontext.py#L173)).
- `_last_planned_s` continuity anchor ([init #L330](eve/eve/util/pathcontext.py#L330)): `get_projection()` now passes `prev_s=self._last_planned_s, window_mm=..., fallback_dist_mm=...` to `project_onto_polyline` and stores the resulting `s` back — applied ONLY to the planned-path projection (per-branch projections keep full-scan). Cleared in `reset()` ([#L455](eve/eve/util/pathcontext.py#L455)) so a new episode starts with a full scan.
- New `reset_projection_continuity()` ([#L565](eve/eve/util/pathcontext.py#L565)) drops the anchor; env5 calls it after a mid-episode SOFA state restore ([env5.py#L790](training%20_scripts/util/env5.py#L790)) so the wire teleport doesn't pin the projection to the stale pre-restore arclength.
- `assert_caches_independent()` adds `_fork_geometry` to the checked fields but with a `None`-skip ([#L1889](eve/eve/util/pathcontext.py#L1889)), since it is lazily built per instance — preserving the multi-worker guarantee that no cache field is shared across the 16 procedural workers.

**Rationale.** These are the projection-continuity primitives that make windowed projection correct under both per-episode mesh regeneration and mesh-pinned restore.

**Adjacent additions (observation-support; detailed in the observation section).** The same file also gained `get_next_junction_fork_geometry()`/`_build_fork_geometry()` ([#L1183](eve/eve/util/pathcontext.py#L1183)), `get_local_radius_at_arclength()` (radius preview, [#L1244](eve/eve/util/pathcontext.py#L1244)), and a memoized-verdict invalidation `self._is_on_correct_path = None` at the end of `update_branch_state()` ([#L1455](eve/eve/util/pathcontext.py#L1455)). These are consumed by `LocalGuidance` and do not change the LOGIC of `is_on_correct_path()` — the invalidation only forces same-step readers to recompute from the just-updated state using the identical classifier — so the heuristic's gating flag is preserved per the standing rule.


---

# Part D — Gen-4 reward shaping & recovery training

> **Frozen-reward note.** Every reward/terminal change in this subsystem was **explicitly approved** by the user (the recovery-training + anti-buckle plan). None alters `is_on_correct_path()` / `_on_planned_path` / `_current_branch_idx`: the classifier still computes byte-identically; only the *consequences* it feeds (truncation, per-step tax, potential shaping) changed, and the anti-buckle term is deliberately classifier-blind (see §1).

## 1. Anti-buckle potential shaping — φ(slack, contact) ∈ [−1, 0], delta form (not farmable)

### Context
The audit found no incentive to recover from a buckle/fold: the reward priced tip arclength progress but nothing about *stored slack*. A wire bowing inside the catheter (tip stationary, length still being fed) paid no explicit cost, and a wire that STARTED buckled (stuck-pool restore, §7) had no gradient toward unbuckling.

### The change
New module [training _scripts/util/buckle_reward.py](training%20_scripts/util/buckle_reward.py) — pure math, no env deps. `buckle_potential(slack_mm, contact_mm)` returns

```
phi = -( W_SLACK * clip(slack_mm - SLACK_DEADBAND_MM, 0, SLACK_CAP_MM)/SLACK_CAP_MM
       + W_CONTACT * clip(contact_mm, 0, CONTACT_CAP_MM)/CONTACT_CAP_MM )
```

| const | value | role |
|---|---|---|
| `SLACK_DEADBAND_MM` | 5.0 | slack ≤ this priced 0 — normal advance chord-cuts (slack ≤ 0); absorbs projection jitter |
| `SLACK_CAP_MM` | 40.0 | slack saturates (fully-developed fold); zero gradient beyond → fold-trunc/MaxSteps price the rest |
| `CONTACT_CAP_MM` | 2.0 | contact saturates; matches `PrivilegedState` dim-5 normalizer |
| `W_SLACK` / `W_CONTACT` | 0.5 / 0.5 | equal-weight channels → φ ∈ [−1, 0] |

env5 adds `coef * (phi_t − phi_{t-1})` each step (γ=1 potential/delta form) in [BenchEnv5.step()](training%20_scripts/util/env5.py#L1079), computed by [`_compute_buckle_potential`](training%20_scripts/util/env5.py#L1701). Two input signals:
- **`slack_mm = inserted_gw − proj.s`** (fed gw length minus tip arclength on this grader's planned path) — **deployable** (computable from tracking alone; same quantity as obs feature 43).
- **`contact_mm = mean |position − free_position|`** over all SOFA beam nodes — the collision-constraint correction, a **privileged** contact-impulse proxy (same accessor as `PrivilegedState` dims 5–6). Chosen over `dofs.force` deliberately: `force.value` includes the internal elastic *bending* force, and high bending is *required* to conform to the siphon — penalizing it would invert into "don't follow tortuosity". `|pos − free_pos|` is nonzero only where the wall pushes back.

Each channel falls back to its previous raw value (`_buckle_prev_raw`) on a read failure, so a transient accessor glitch contributes delta = 0 for that channel rather than a fake jump.

### Rationale
By construction: (a) any closed loop in (slack, contact) nets **exactly zero** — the term cannot be oscillation-farmed (the same lesson as the arclength 2× fix, §3); (b) the episode sum **telescopes to φ_end − φ_start**, so forming-and-recovering a buckle is net zero while STARTING buckled and unbuckling is net **positive** — the recovery incentive that was missing; (c) φ is bounded so a single-step delta is bounded and a SOFA glitch cannot dominate the return. **Caps are applied to the INPUTS, never to the delta** — clipping the delta would break loop-neutrality. There is deliberately **no on-path/off-path gating**: a classifier gate would make φ jump on classifier flicker (farmable at the boundary), so φ stays a pure function of physical state (respects the frozen-classifier rule).

### First-step baseline & restore safety
`self._buckle_phi_prev = None` in `__init__` and re-nulled every [`reset()`](training%20_scripts/util/env5.py#L677) → the first step of each episode yields delta = 0. This is what makes SOFA-restored episodes safe: a state restored **already buckled** (stuck-pool / bif11 checkpoint) re-baselines φ on step 1 instead of being spiked with the full negative potential, and recovering from it then nets positive.

### Wiring & reward-parity guard
- Ctor param `buckle_reward_coef: float = 0.0` (0 = off / frozen legacy reward). Guarded import of `buckle_potential`; if `coef != 0` but the module is unmounted, [`__init__` raises `ImportError`](training%20_scripts/util/env5.py#L264) (fail-fast — silently skipping the term would score transitions under a different MDP than intended).
- Applied in **both RL and heuristic modes** — the heuristic's actions never read reward, so its behavior is unchanged, but its demo transitions are scored under the same MDP as explore transitions.
- Propagated into `MultiTargetEnv5` (`primary` + every grader `g`) so harvest transitions match training reward.
- `EVE_RL_BUCKLE_COEF` is exported by `DualDeviceNav_train` (`os.environ["EVE_RL_BUCKLE_COEF"] = repr(coef)`) and stamped/checked in `eve_rl/util/experience_cache.py`, so a demo cache harvested at `coef=c` fails-fast if loaded into a run with a different coef — enforcing seed/train reward parity (the exact mismatch this whole design avoids).
- STEP-log gains a `buckle_phi=<±.3f>` diagnostic field ([env5.py:1507](training%20_scripts/util/env5.py#L1507)).
- Launchers `launch_rcca_harvest.sh` / `launch_rcca_procedural_v1.sh` run `--buckle_reward_coef 0.5` (half-weight, conservative first run).

---

## 2. `relax_failure_truncations` — recovery-training terminal relaxation

### Context
You cannot learn recovery (retract, unbuckle, re-approach) from states the env *kills you for entering*. The old fold-stall (≥ `FOLD_STALL_STEPS`=20) and off-path (≥ `OFF_BRANCH_GRACE_STEPS`=50) truncations fired `truncated=True` + the −5 `FAILURE_TRUNCATION_PENALTY`, teaching the critic **V(buckled) = penalty-and-nothing-after**.

### The change
New ctor flag `relax_failure_truncations: bool = False` ([env5.py:249](training%20_scripts/util/env5.py#L249)). When True, the fold/off-path/vessel-end detectors **keep counting** (they remain obs features + stuck-pool triggers, §7) but no longer truncate RL episodes:

| terminal source | default | under relax |
|---|---|---|
| MaxSteps | ends episode | ends episode |
| SimError | ends episode | ends episode |
| VesselEnd | truncates | **not** in the `Combination` — `trunc_components=[max_steps, sim_error]`; `vessel_end` inserted only when not relax ([env5.py:452](training%20_scripts/util/env5.py#L452)) |
| fold-stall ≥ 20 | truncates | `truncated=True` only in heuristic mode; `elif not relax` otherwise ([env5.py:1052](training%20_scripts/util/env5.py#L1052)) — counter ratchets, episode continues |
| off-path ≥ 50 (WBT) | truncates | same gating ([env5.py:1261](training%20_scripts/util/env5.py#L1261)) |

VesselEnd relaxes because a wire overshooting a branch terminus is capped by `stop_device_at_tree_end` and can **retract back onto the path** — truncating denies exactly the recovery relax exists to allow. Heuristic-mode aborts are unaffected (demo harvesting keeps its own timeouts).

### Rationale
Only MaxSteps/SimError end an RL episode, so the off-path shaping (retract toward the divergence point earns reward back symmetrically) and the anti-buckle potential (§1) get the full 600 steps to price recovery, and MaxSteps + its −3 (§4) end genuinely hopeless episodes. Wired via `--relax_failure_truncations` in `DualDeviceNav_train` ([L382](training%20_scripts/DualDeviceNav_train.py#L382)); set by `launch_rcca_harvest.sh` and the procedural launchers. Approved terminal change; the counters/classifier compute unchanged — only the truncation consequence is removed.

---

## 3. Reward-farm fix — ArcLengthProgress 2×-forward doubling → 1× symmetric

### The bug
Plan v9 Change 4b doubled `progress_factor` when the wire was inside the target daughter **and** `delta_s > 0`. That was **not** potential-based: forward motion paid 2× while backward paid 1×, so **every oscillation cycle banked `+progress_factor·Δs`**. Harmless when a wrong-daughter dwell truncated quickly — but under `relax_failure_truncations` (§2) the episode runs the full 600 steps, and a wire merely **dithering in the RCCA** farmed ≈ 3–4 return (≈ a genuine success) **without ever threading the target**.

### Fix
[eve/eve/reward/arclengthprogress.py:140](eve/eve/reward/arclengthprogress.py#L140) — the entire branch-lookup + `pf *= 2.0` block is deleted; `r_progress = self.progress_factor * delta_s` (flat 1×, symmetric). The per-step reward now **telescopes** to `progress_factor·(s_final − s_initial)`: a round trip nets exactly zero, and the episode sum is bounded by the net arclength actually advanced (~2.0 over the full path), earned only by genuine progress. Reads getters only — no branch-classifier change. This fix is what makes §2's full-length episodes safe from the oscillation farm.

---

## 4. Reward-farm fix — `MAX_STEPS_PENALTY = -3`, checked first

### The bug
A max_steps truncation carried **no penalty**, so a wire that loitered to the horizon kept all accumulated shaping with nothing to offset it — the "max_steps farm". Acute under §2, where fold/off no longer end the episode, so **max_steps becomes the default terminal for a stuck wire**.

### Fix
`MAX_STEPS_PENALTY = -3.0` ([env5.py:99](training%20_scripts/util/env5.py#L99)). The truncation-penalty block ([env5.py:1305](training%20_scripts/util/env5.py#L1305)) now checks `self._max_steps_trunc.truncated` **first** → `reward += -3`; the `elif` for vessel-end/fold/off applies the −5 hard-failure penalty only when it is the *actual* trigger. Under relax the fold/off counters keep climbing without truncating, so a pure timeout must be priced as a timeout (−3), **not** mis-charged the −5 just because the wire folded earlier. Ordering: −3 makes a timeout unambiguously worse than a success (+3) and a clean early exit, while staying softer than a hard failure (−5).

---

## 5. Off-path retraction discount — recovery priced below persistence

### Context
The uniform −0.007/step off-path tax (RL_IMPROV_10 §4) also taxed the recovery the policy must learn: backing OUT of a wrong branch paid the same per-step price as pushing DEEPER.

### The change
In [`_compute_path_segment_step_reward`](training%20_scripts/util/env5.py#L1885), while the wire is off-path **and** genuinely retracting, the tax drops to `OFF_PATH_RETRACT_TAX = -0.002` (from −0.007). Gated two ways:
- `_off_branch_steps >= OFF_PATH_RETRACT_MIN_OFF_STEPS` (=3) — the wire must be genuinely **inside** the wrong path; boundary flicker keeps paying full tax, so there's no incentive to dance on the classification edge.
- `_last_delta_gw <= -OFF_PATH_RETRACT_MIN_MM` (=−0.1 mm) — **executed** gw insertion delta (stashed at [env5.py:1023](training%20_scripts/util/env5.py#L1013)), not commanded: a masked retract command that moved nothing, or holding still, pays full tax (no camping discount).

### Rationale
Still negative → no closed loop profits: a wrong-branch round trip nets at most `-0.007·k_in − 0.002·k_out`. Entering / holding / flickering always pay full 0.007; only sustained, executed withdrawal is discounted. Reward-only — reads the off-path counter and executed delta, never the classifier.

---

## 6. WBT / grader_failure_timeout label-leak fixes (3 sites, gated on relax)

### The bug
Under §2 the fold/off counters keep climbing but no longer truncate. Three label sites still read those counters raw and mislabeled **recovery episodes** as fold/WBT failures. That corrupted `is_clean` (defined as `... and not grader_failure_timeout`), wrongly dropping clean RCCA threads from the seed's demo lane, and produced the spurious "wire_fold_stall / wrong_branch_timeout in snapshots" the user saw on max_steps recovery episodes.

### Fix
All three now short-circuit under `not self.relax_failure_truncations`:
1. Shared-grader `_gtimeout` flag ([env5.py:604](training%20_scripts/util/env5.py#L604)).
2. `info["grader_failure_timeout"]` ([env5.py:1601](training%20_scripts/util/env5.py#L1601)).
3. `_resolve_termination_reason` — the `"wire_fold_stall"` label ([env5.py:2155](training%20_scripts/util/env5.py#L2155)) and the off-path→WBT/overshoot label ([env5.py:2165](training%20_scripts/util/env5.py#L2165)) both fall through under relax to the **actual** truncation source (max_steps / vessel_end / sim_error).

Snapshot buckets and `is_clean` are now correct for recovery episodes.

---

## 7. Stuck-state checkpoint pool — recovery-curriculum harvest

### The change
Env-var-gated (`STUCK_CHECKPOINT_DIR`). The **first** time an episode crosses a stuck threshold — `STUCK_FOLD_TRIGGER = 10` (of the 20 fold kill) or `STUCK_OFF_BRANCH_TRIGGER = 25` (of the 50 off kill) ([env5.py:1283](training%20_scripts/util/env5.py#L1283)) — env5 writes a full SOFA snapshot via [`_save_stuck_checkpoint`](training%20_scripts/util/env5.py#L2046). Triggers fire **before** the kill thresholds so the captured state is stuck-but-recoverable — the whole point of the curriculum. Latched per-episode (`_stuck_ckpt_saved_this_ep`), RL-mode only, **unconditional on outcome** (failures ARE the product), bounded per worker by `STUCK_CHECKPOINT_MAX_PER_WORKER` (default 200).

Each snapshot is the fuller-format `save_checkpoint()` + a JSON sidecar carrying `reason`, `fold_stall_count`, `off_branch_steps`, `target_branch`, `inserted_lengths`, `proj_s_at_capture`, and two Gen-4 additions: `mesh_fingerprint` (a stuck checkpoint is mesh-bound SOFA state — under `--procedural_rcca` each worker/generation has a different siphon, so the tag lets `CheckpointRestoreWrapper` restore each state **only** into its exact capture mesh; fixed-mesh → `"fixed"`, a no-op) and `slack_at_capture` (the buckle bow = `inserted_gw − proj_s`, so the escapability screener can gate on restore fidelity — a sprung/popped restore won't reproduce the slack). A guard skips `tracking3d` if it came back object-dtype (ragged), which `np.savez` stores but the `allow_pickle=False` restore load would reject.

### Rationale
Restored later via `CheckpointRestoreWrapper`, these states train dedicated recovery episodes that **start buckled**: short episodes, dense recovery signal, no wall-clock spent walking into trouble (the restore-at-fork mechanism re-aimed at recovery skills).

---

## 8. Restore latch re-arm on mesh change + projection-continuity reset

Two changes in the SOFA-restore path of [`reset()`](training%20_scripts/util/env5.py#L749):
- **Latch re-arm ([env5.py:771](training%20_scripts/util/env5.py#L771)).** RL_IMPROV_10 §13e's double-restore warm-up (`_restore_warmed_up`) was per-**worker** — set once, ever. But the first-restore quirk (restore #1 sets `xtip` but doesn't apply `dof_positions` into a freshly-built scene) is per-**scene-build**. Under `--procedural_rcca` and the escapability screener the mesh regenerates and SOFA rebuilds on many resets, so from restore #2 onward a single-apply into a fresh scene re-fires the quirk. Now `_restore_warmed_up` is re-armed to False whenever `vessel_tree.mesh_path` changed since the last restore (`RCCAVariedFromMesh` mints a new temp `mesh_path` each `_generate`, so a rebuild always flips it). Fixed-mesh runs never change `mesh_path` → byte-identical to the prior warmup-once behavior.
- **`reset_projection_continuity()` ([env5.py:790](training%20_scripts/util/env5.py#L790)).** Explicit defense against the restore teleport pinning the windowed projection to the pre-restore arclength — clears the continuity anchor even if the wrapped observation/reward re-resets (which normally reach `PathProjectionCache.reset()`) throw.

---

## 9. `_on_intervention_stepped` hook — same-step branch state for reward consumers

### The bug
Reward/obs consumers that gate on branch classification — `ArcLengthProgress` on-path gate, the path-segment step reward, `LocalGuidance` branch features, the off-path detector — must see **this** step's classification. Previously `BenchEnv5.step()` ran `update_branch_state()` **after** `super().step()` had already computed observation + reward, so those consumers read the **prior** step's branch state (one-step-stale, non-Markov).

### Fix
[eve/eve/env.py:70](eve/eve/env.py#L70) — `Env.step()` now calls `self._on_intervention_stepped()` after the intervention / pathfinder / interim-target advance but **before** `observation/reward/terminal/truncation`. Base `Env` defines it as a no-op ([env.py:84](eve/eve/env.py#L84)); [`BenchEnv5._on_intervention_stepped`](training%20_scripts/util/env5.py#L923) overrides it to invalidate the projection cache and run `update_branch_state()` there. The old post-`super()` `update_branch_state()` call in `step()` is removed (now a comment). The heuristic's view is unchanged: it reads the state machine *between* steps, and the machine still advances exactly once per step.

### Rationale
A timing/Markov-freshness correction — the classifier logic is untouched (`is_on_correct_path()` returns the same value for a given physical state), only *when* consumers read it changes. Approved as part of the reward-shaping work.


---

# Part E — RL_IMPROV_15 freeze-collapse: forensic & fix package

The Gen-4 stack above (mesh-invariant obs + asymmetric critic + procedural
anatomy + recovery) was validated on the `rcca_procedural_v1` run. It
**worked, then failed in a specific and instructive way** — and the
forensic that dissected the failure is the reason the fix package exists.
The code changes are documented in the next sections (§ALGO); this section
is the **mechanism and evidence** that motivate them.

## The v1 freeze-collapse — what happened

`rcca_procedural_v1` (16 procedural-mesh workers, AWAC, 10k pretrain +
online) learned productively for ~8 hours — explore success climbed to
**34%**, held-out eval to **13.3%**, recovery behaviors appeared (off-path
→ re-approach → success) — then the **deterministic policy collapsed to a
freeze** and the run **IPC-deadlocked** after the 3rd eval. Eval
trajectory:

| eval | explore step | held-out quality | det. speed | trajectory |
|---|---|---|---|---|
| 1 | 278,345 | 8.2% | 4.78 mm/s | 326 mm |
| 2 | 505,588 | **13.3%** (peak; = `best_checkpoint`) | 4.28 mm/s | 220 mm |
| 3 | 771,040 | **3.1%** | **0.95 mm/s** | **59 mm** |

## The forensic (5 parallel investigations)

A 5-agent forensic over the full 950k-step log + losses CSV + replay
buffer + 77 policy snapshots established the causal chain — each link
verified by a different agent, several of my initial hypotheses **refuted**:

1. **`log_std` was pinned at its CEILING (std=1.0) on 100% of states from
   update ≈39k onward** — all four action dims, every probe state, to the
   end (floor −2 never touched). *The entropy controller had no variance
   headroom, ever.*
2. **α (SAC temperature) decayed to the −10 `log_alpha` floor and sat there
   for 165k updates** while `entropy_proxy` ground down 2.63 → 0.14, then
   **whipsawed back to 0.45**. Rails (12%→23% of translations at exactly
   ±30.000) tracked this decay and *receded* after — a **symptom**, not the
   cause; the freeze is the opposite of a rail mode.
3. **α's recovery crushed the action MEAN.** With std capped, the only
   lever left to raise entropy is shrinking |μ| (less tanh-saturation =
   higher log-prob entropy). Snapshot probing: episode-start-state
   |tanh(μ₀)| peaked 0.255 (explore ~440k), fell **−33% in the single
   interval containing the α spike** (→505k = eval2), then monotonically to
   0.089 by eval3. Collapse is **regional** — start states −65%,
   buffer-wide only −17% — concentrated exactly where episodes begin.
4. **AWAC was inert the whole run** (`awac_weight_mean` = 1.00,
   saturation 0.0 always): λ=3.0 against a flat critic ⇒ `exp(adv/λ) ≈ 1`,
   so the policy loss degenerated to **pure behavior-cloning + α·entropy**.
   Nothing opposed the mean-crush.
5. **Amplifiers.** (a) Critic pessimism: `q1_mean` −0.97 → −4.71 monotone,
   converging toward the freeze-return level (−6.6); with `min(Q1,Q2)`
   penalizing the high-variance "attempt" branch, an EV gap of only ~1.2
   return-units separated attempting from freezing. (b) The buffer stores
   the **raw** policy action while the env executes a **clamped** one — at
   the insertion floor, large retraction commands execute as zero motion,
   teaching Q that railing at the limit is consequence-free, right in the
   freeze region.

**Buffer was NOT poisoned** (stored micro-actions 0.02% of 788k rows; PER
neutral; clean-lane worked). **Freezing was never strictly reward-rational**
(EV(attempt) − EV(freeze) never fell below +1.15) — but sits ~1 unit above
freeze's zero-variance −6.6, which a value-pessimistic learner tips into
once success probability dips.

## The two lessons that shaped the fixes

- **Explore returns hide a dead mean.** With std=1.0, sampled explore
  actions average |a|≈17 mm/s regardless of μ, so explore success was still
  39% in the last "healthy-looking" hour. **Only the deterministic policy
  reveals a freeze.** → the monitor now runs a deterministic start-state
  probe (§F8) instead of trusting explore returns.
- **The freeze basin IS the pretrain-BC attractor.** The v2 post-pretrain
  baseline eval (§F7) scored **6.1% at 0.54 mm/s** — kinematically
  identical to v1's *collapsed* eval3. So the "freeze" was a **regression
  to the pretrained mean**, not a new pathology. Online learning grows μ
  away from it; the entropy whipsaw shoved it back.

## The fix package (F1–F8) — one lever per verified link

| Fix | Change | Link it severs |
|---|---|---|
| **F1** | configurable `log_alpha` clamp; run uses **−5.0 / −2.3** (α ∈ [0.0067, 0.100]) | decay-to-floor → whipsaw (link 2/3) |
| **F2** | `awac_lambda` **3.0 → 1.0** | inert BC weights (link 4) |
| **F3** | `action_mean_penalty` **0.005** (`\|atanh(μ)\|` in the policy loss) | mean-inflation rails (LCCA branch #4) |
| **F4** | `EVE_CLEAN_RAIL_MAX=0.15` clean-lane rail filter | bang-bang self-cloning (LCCA branch #2) |
| **F5** | *dropped after review* — commanded action is correct for off-policy Q; env clamps at actuator limits are true dynamics, not a mislabel | (link 5b — deliberately not "fixed") |
| **F6** | `_model_queue.get(timeout)` via `EVE_RL_MODEL_QUEUE_TIMEOUT_S=900` → loud RuntimeError | the no-timeout IPC deadlock (`anon_pipe_read`) |
| **F7** | `--eval_after_pretrain` — one held-out eval + `checkpoint0` before any exploration | banks a pretrain-only baseline (v1 had no reference; all its checkpoints were mid/post-collapse) |
| **F8** | deterministic start-state probe in the monitor | v1's freeze was invisible in explore returns |

The **implementation** of F1–F7 is documented in §ALGO (code) and §HARVEST
(launcher `launch_rcca_procedural_v2.sh`). F5's non-fix is deliberate and
recorded so it is not "re-fixed" later.

---


## Fix package — implementation (code)

The `lcca_awac_v1` run "froze": the policy stopped moving and every eval flat-lined at ~8.2%. Forensics traced a four-link chain, and this package adds one configurable knob per link (all defaults preserve legacy behavior) plus the surrounding infra fixes. None of these touch `is_on_correct_path()`, the env reward computation, the observation layout, or the terminal conditions — they are stabilization knobs, buffer-admission filters, and algo/IPC plumbing. The reward-shaping and obs-tail definitions themselves (`buckle_reward.py`, `PrivilegedState`, `DualDeviceNavRCCAVaried`) live in the env/obs subsystem; here we cover only the algo/network/agent plumbing that consumes them, all gated OFF by default.

**Freeze chain → fix map**

| v1 freeze link | mechanism | fix section(s) |
|---|---|---|
| std ceiling-pin | `log_std` ran to +2 on all 4 dims, σ pinned high (1→2), actions saturate from noise | §4 (soft tanh band + `log_std_max` cap; launchers pass 0.0) |
| alpha decay→floor→whipsaw | α decayed to the −10 floor over 165k updates (entropy term vanished), then whipsawed to 0.45 | §1 (`log_alpha_min` raises the floor), §3 (α re-derived from `log_alpha`) |
| entropy crushes the MEAN | with σ ceiling-pinned, α's only entropy lever was `|mean|→0` (the freeze) | §1 (`log_alpha_max` caps α), §4 (σ cap restores the variance lever), §5 (`action_mean_penalty`) |
| inert AWAC weights ≈ 1 | advantages collapse → `exp(A/λ)≈1` → AWAC degrades to BC and clones the policy's own railed successes | §7 (clean-lane rail filter breaks self-cloning), §6 (leaky BC floor keeps advantage-weighted gradient alive) |

## 1. Configurable `log_alpha` clamp rails (SAC/AWAC auto-alpha)

**Context.** The auto-alpha regulator integrates `log_alpha` with hard `.clamp_()` rails that were hardcoded `(-10, 2)`. In v1 those rails permitted the decay→floor→whipsaw cycle: α ran down to the −10 floor (α≈4.5e-5, entropy term effectively off), entropy ground to 0.14, then α violently corrected up to 0.45 and — with σ ceiling-pinned — spent that budget crushing the action mean.

**The change.** [eve_rl/eve_rl/algo/sac.py](eve_rl/eve_rl/algo/sac.py#L188) adds ctor params `log_alpha_min: float = -10.0` / `log_alpha_max: float = 2.0`, validated (`min > max` raises) at [L227](eve_rl/eve_rl/algo/sac.py#L227) and applied in `_update_alpha` via `self.model.log_alpha.clamp_(self.log_alpha_min, self.log_alpha_max)` at [L474](eve_rl/eve_rl/algo/sac.py#L474). Threaded through [BenchAgentSynchron](training%20_scripts/util/agent.py#L182) → [SAC ctor](training%20_scripts/util/agent.py#L381) and exposed as CLI `--log_alpha_min` / `--log_alpha_max` ([DualDeviceNav_train.py:1538](training%20_scripts/DualDeviceNav_train.py#L1538), [:1553](training%20_scripts/DualDeviceNav_train.py#L1553); wired at [:630](training%20_scripts/DualDeviceNav_train.py#L630)).

**Rationale.** A higher floor (e.g. −5 → α_min 0.0067) keeps the entropy term alive so entropy never craters and α never needs a violent correction; a low ceiling (e.g. −2.3 → α_max 0.100) caps how hard the entropy term can push, so with σ saturated the term can no longer dominate the BC/advantage objective and mean-crush. Defaults `(-10, 2)` reproduce legacy behavior exactly.

## 2. `target_entropy` setpoint override + native-float serialization fix

**Context.** The auto-alpha setpoint was hardcoded `self.target_entropy = -torch.ones(1) * n_actions`. For a 4-dim tanh-Gaussian in the healthy `(-2, 0)` σ-band the operating entropy is ~+2.5, so the −4 default leaves the regulator a huge dead zone — α decays to ~0 and only re-engages after the policy is already railed below −4.

**The change.** [sac.py:178](eve_rl/eve_rl/algo/sac.py#L178) adds `target_entropy: Optional[float] = None`; when provided it is stored as a **native float** (`self.target_entropy = float(...)`, [L265](eve_rl/eve_rl/algo/sac.py#L265)), else falls back to `-float(n_actions)`. The `.item()` reads become `float(self.target_entropy)` ([L433](eve_rl/eve_rl/algo/sac.py#L433)) and `to()` no longer moves it to device ([L870](eve_rl/eve_rl/algo/sac.py#L870)). CLI `--target_entropy` (default `None`) at [DualDeviceNav_train.py:1488](training%20_scripts/DualDeviceNav_train.py#L1488), threaded through all three agent ctors ([agent.py:186](training%20_scripts/util/agent.py#L186), [:383](training%20_scripts/util/agent.py#L383), [:463](training%20_scripts/util/agent.py#L463)).

**Rationale.** Native-float (not `torch.Tensor`) is required because the eve_rl `ConfigHandler` `getattr`'s every `__init__` param name when serializing to `config.yaml`/`.everl` and raises on a Tensor; a float also broadcasts fine in `_update_alpha`. Setting e.g. `+1.0` holds the regulator inside the healthy band instead of inert.

## 3. `alpha` re-derived from `log_alpha` each update + masked alpha loss + explicit detach

**Context.** `self.alpha` (a cached tensor) could go stale relative to `self.model.log_alpha` after a checkpoint load, and `_update_alpha`'s mean over `log_pi` mixed in padded episode steps.

**The change.** [sac.py:346](eve_rl/eve_rl/algo/sac.py#L346) re-derives `self.alpha = self.model.log_alpha.exp().detach()` at the top of every `update()` (comment: init/ctor value may be stale after a checkpoint-loaded `log_alpha`). `_update_alpha` now takes `padding_mask` and, when present, computes a masked mean `(log_alpha * delta * mask).sum() / mask.sum().clamp(min=1.0)` ([L456](eve_rl/eve_rl/algo/sac.py#L456)) — `log_pi` arrives pre-multiplied by the mask, so padded entries otherwise carry a spurious constant `-target_entropy`. Post-step `self.alpha` is `.detach()`ed ([L476](eve_rl/eve_rl/algo/sac.py#L476)), and both AWAC and SAC policy-loss branches use `self.alpha.detach()` explicitly ([L525](eve_rl/eve_rl/algo/sac.py#L525), [L531](eve_rl/eve_rl/algo/sac.py#L531)) so correctness is independent of `_update_alpha`'s `zero_grad` ordering.

**Rationale.** Keeps the entropy temperature and its target consistent across checkpoint restore and padded batches; the detach hygiene prevents α's gradient from leaking into the policy update.

## 4. Soft tanh-rescale `log_std` band (replaces the hard clamp)

**Context.** `GaussianPolicy.forward` bounded `log_std` with `torch.clamp(log_std, min, max)`, which has **zero gradient outside the band** — a railed head becomes a one-way ratchet, movable only by shared-trunk drift, and cannot be pulled back off the σ ceiling.

**The change.** [gaussianpolicy.py:77](eve_rl/eve_rl/network/gaussianpolicy.py#L77) (and the `forward_play` mirror at [L95](eve_rl/eve_rl/network/gaussianpolicy.py#L95)) replaces the clamp with a tanh rescale `log_std = min + 0.5*(max-min)*(tanh(raw)+1)` — bounded in `(min, max)` with nonzero gradient everywhere; a raw pre-squash 0 maps to the band **midpoint**. Because of that remap the ctor default `log_std_min` moves `-20 → -2` ([L22](eve_rl/eve_rl/network/gaussianpolicy.py#L22)) so a fresh head (raw≈0) initializes at the `(-2,2)` midpoint `log_std=0` (σ≈1), matching the old hard-clamp init; the legacy `(-20,2)` would have initialized collapsed at `log_std=-9` (σ≈1.2e-4). CLI `--log_std_min`/`--log_std_max` help text rewritten to describe the soft band; launchers pass `--log_std_max 0.0` for the validated `(-2, 0)` operational band that caps the v1 ceiling-explosion.

**Rationale.** This is a deliberate parameterization change (NOT byte-identical to the old clamp) — it is the mechanism that lets a ceiling-pinned σ recover. Defaults are chosen so init σ is unchanged.

## 5. `action_mean_penalty` |atanh| loss + per-dim entropy bonus (Plan v11 anti-rail scaffold)

**Context.** AWAC's advantage-weighted BC objective has no entropy term, so the squashed mean saturates at the ±1 tanh rail (clamp_fraction → 0.4–0.5). Per-dim `log_std` alone does not stop mean-rail saturation (Plan v11 risk audit #3).

**The change (pre-existing scaffold, now the deliberate mean lever).** [sac.py:553](eve_rl/eve_rl/algo/sac.py#L553), AWAC-only: (1) a per-dim entropy bonus subtracts `Σ beta_i * mean_t(log_std_i)` (minimizing the loss RAISES log_std, cath_trans weighted heaviest), and (2) an `action_mean_penalty` term adds `coef * |atanh(clamp(tanh(mean), ±0.99))|.abs().mean()` ([L565](eve_rl/eve_rl/algo/sac.py#L565)), pushing the pre-tanh mean toward 0. AWAC weight saturation is logged every update (`_awac_weight_saturation`, `_awac_weight_max/_mean`, [L507](eve_rl/eve_rl/algo/sac.py#L507)); `weight = exp(advantage/awac_lambda).clamp(max=20.0)` ([L500](eve_rl/eve_rl/algo/sac.py#L500)). CLI `--action_mean_penalty` and `--entropy_beta_per_dim` remain the tuning surface.

**Rationale.** This is the *controlled, bounded* way to keep the mean off the rail — the counterpart to §1's cap on the *uncontrolled* alpha mean-crush. Both are `coef=0`/`None`-gated (`action_mean_penalty 0.0` = off).

## 6. Leaky log-prob floor + NaN sanitize in `GaussianPolicy.log_prob`

**The bug.** The Plan v8 hard floor `log_prob.sum(-1).clamp(min=-20.0)` zeroes the BC gradient on exactly the demos the policy is farthest from — precisely the high-advantage demos AWAC most needs to clone. And `torch.where(lp>=floor, …)` would take the FALSE branch for a NaN `lp` and pass it straight into the BC loss, poisoning the run.

**Fix.** [gaussianpolicy.py:125](eve_rl/eve_rl/network/gaussianpolicy.py#L125) makes the floor **leaky**: below `floor=-20` the value keeps a 5% gradient (`lp_leaky = floor + 0.05*(lp-floor)`, direction preserved, magnitude damped 20×) down to a `hard_min=-30` bound (so worst-case per-sample loss ≤ 600 at weight ≤ 20). NaN/±Inf are sanitized first via `torch.nan_to_num(lp, nan=-30, posinf=0, neginf=-30)` ([L130](eve_rl/eve_rl/network/gaussianpolicy.py#L130)) so the floor is a real safety net, not a NaN passthrough.

**Rationale.** Restores an advantage-weighted BC gradient on the far demos (attacking the "AWAC degrades to inert BC" link) while still bounding the loss; the NaN guard closes the poisoning path the soft σ-bound makes unlikely but not impossible.

## 7. Clean-lane rail admission filter (`EVE_CLEAN_RAIL_MAX`)

**Context.** The AWAC BC term clones the balanced "clean lane" of successes. When the policy starts producing its own bang-bang (railed) noise-successes and those re-enter the clean lane, the policy clones its own saturation → the mean rails further (the self-cloning loop behind the "inert AWAC weights" link).

**The change.** [pervanillastep.py:118](eve_rl/eve_rl/replaybuffer/pervanillastep.py#L118) reads env var `EVE_CLEAN_RAIL_MAX` (unset/empty → disabled, legacy) into `self.clean_rail_max`. In `push()` ([L175](eve_rl/eve_rl/replaybuffer/pervanillastep.py#L175)) a newly-collected success is admitted to the clean lane only if its railed-step fraction (fraction of steps with any dim `|a| > 0.95`) is `<= clean_rail_max`; otherwise it is kept in the general buffer (critic data) but excluded from the amplified BC lane, logged as `CLEAN_RAIL_FILTER` with a running `_clean_rail_rejected` count.

**Rationale / correctness note.** The filter must flip **`reached` itself** ([L185](eve_rl/eve_rl/replaybuffer/pervanillastep.py#L185)), not just skip the initial clean-tree write — because `update_priorities()` re-adds any `is_clean` slot to the clean_tree on every TD-priority update, so a tree-only exclusion would be silently re-admitted. Reference calibration: seed-diverse cleans ~0.10 railed vs a self-cloned poison cohort at 0.23–0.24, so a 0.15 threshold separates them. Malformed actions → `railed_frac 0.0` → admit (legacy-safe).

## 8. Per-dim exploration noise + tanh-domain clip (SAC + IQL, all sites)

**The bug.** `action += np.random.normal(0, self.exploration_action_noise)` — a size-less `np.random.normal` returns ONE scalar shared by every action dim (perfectly correlated exploration), and the sum could push the stored action outside the tanh domain `[-1, 1]`.

**Fix.** All four SAC exploration sites ([sac.py:290](eve_rl/eve_rl/algo/sac.py#L290) plus `SACPlayOnly.get_exploration_action`/`get_action_exploration`) and both IQL sites ([iql.py:75](eve_rl/eve_rl/algo/iql.py#L75), [:232](eve_rl/eve_rl/algo/iql.py#L232)) now add `np.random.normal(0.0, noise, size=action.shape)` then `np.clip(action, -1.0, 1.0)`. In `get_action_exploration` only the action is clipped; `mean`/`log_std` are returned raw.

**Rationale.** Independent per-dim exploration; the buffer never stores an out-of-domain action that `log_prob`'s `atanh` would blow up on.

## 9. Wire up the previously-dead `reward_scaling` in the Bellman target (SAC + IQL)

**The bug.** Both critics accepted a `reward_scaling` ctor param but never applied it: `expected_q = rewards + (1-dones)*gamma*next_q` used the raw reward. The param was inert.

**Fix.** [sac.py:699](eve_rl/eve_rl/algo/sac.py#L699) and [iql.py:417](eve_rl/eve_rl/algo/iql.py#L417) now compute `expected_q = rewards * self.reward_scaling + (1-dones)*gamma*next_q`. Identity at the default `1.0` (which all current configs use), so no behavior change — but the knob is now functional.

## 10. MLP input-layer ReLU

**The change.** [mlp.py:83](eve_rl/eve_rl/network/component/mlp.py#L83) resolves the long-standing `# TODO: Add F.relu after input layer` — `state = F.relu(self._input_layer(obs_batch))`. Previously the input layer was linear and only the hidden layers had ReLU, so the input projection was wasted (composed linearly with the first hidden layer).

**Rationale.** A genuine nonlinearity at the input restores the first layer's representational capacity. This affects every MLP body/critic; it is a network-capacity change, not a stabilization knob — flagged for the record.

## 11. Asymmetric (privileged) critic + auxiliary privileged-label head

**Context.** Gen-4 appends a privileged tail (`PrivilegedState`, defined in the obs subsystem) as the LAST key of env5's flat ObsDict. The critics may consume it, but a deployable policy must not depend on sim-only state.

**The change.** (a) [gaussianpolicy.py:63](eve_rl/eve_rl/network/gaussianpolicy.py#L63) — `forward`/`forward_play` slice `obs_batch[..., :self.n_observations]` at one chokepoint, so the policy is built `n_obs_total - privileged_obs_dim` wide and never sees the tail. (b) [gaussianpolicy.py:30](eve_rl/eve_rl/network/gaussianpolicy.py#L30) — `n_aux > 0` adds a third body output head whose forward output is stashed in `self._last_aux`; [sac.py:582](eve_rl/eve_rl/algo/sac.py#L582) adds `aux_coef * MSE(_last_aux, states.index_select(-1, aux_label_indices))` to the policy loss (padding-masked), teaching the policy to *predict* contact/buckle labels from its deployable prefix without seeing them. (c) [agent.py](training%20_scripts/util/agent.py#L206) adds `privileged_obs_dim` (policy width = flat − tail; policy gets its OWN embedder instance since the asymmetric widths break head-sharing), validates `aux_label_rel_indices` against the tail bounds at construction, and converts them to absolute flat-obs indices ([L242](training%20_scripts/util/agent.py#L242)). (d) CLI `--aux_coef` / `--aux_labels` ([DualDeviceNav_train.py:1631](training%20_scripts/DualDeviceNav_train.py#L1631), [:1643](training%20_scripts/DualDeviceNav_train.py#L1643)); `privileged_obs_dim = PrivilegedState.N_DIMS` only for env v5 ([:628](training%20_scripts/DualDeviceNav_train.py#L628)); cache-load obs-dim guards compare against the CRITIC width `agent.algo.model.q1.n_observations` so the sliced policy width doesn't false-fail a valid cache.

**Rationale.** Asymmetric critic + representation shaping without breaking deployability. `aux_coef 0.0` / empty `--aux_labels` / `n_aux 0` = byte-identical legacy behavior; the policy always slices to the deployable prefix, so the observation the policy consumes is unchanged.

## 12. IPC deadlock guard on `_model_queue.get()`

**The bug.** The v1 run hung FOREVER after eval3: the main thread blocked in a no-timeout `self._model_queue.get()` (stuck in `anon_pipe_read`) when a subprocess went unresponsive (host suspend / clock-jump mid-IPC) — a silent all-night hang with no error.

**Fix.** [singelagentprocess.py:616](eve_rl/eve_rl/agent/singelagentprocess.py#L616) adds `_model_queue_get(what)` — `self._model_queue.get(timeout=...)` with `EVE_RL_MODEL_QUEUE_TIMEOUT_S` (default 900s); on `queue.Empty` it logs `IPC TIMEOUT: … did not answer '<what>'` and raises `RuntimeError` instead of hanging. The three `state_dicts_network`/`_optimizer`/`_scheduler` getters route through it ([L638](eve_rl/eve_rl/agent/singelagentprocess.py#L638), [:652](eve_rl/eve_rl/agent/singelagentprocess.py#L652), [:666](eve_rl/eve_rl/agent/singelagentprocess.py#L666)).

**Rationale.** Converts a silent deadlock into a loud, actionable crash that an operator or docker restart-policy can act on.

## 13. `eval_after_pretrain` — pretrain-only baseline checkpoint (`checkpoint0`)

**Context.** v1 had no reference for its eval1 (8.2%) and its only checkpoints were mid/post-collapse — there was no clean pretrained snapshot to fall back to.

**The change.** [runner.py:501](eve_rl/eve_rl/runner/runner.py#L501) adds `eval_after_pretrain: bool = False`; when set, immediately after the warm-start pretrain (before any exploration) it logs "Post-pretrain BASELINE eval" and runs `self.eval(...)` ([L757](eve_rl/eve_rl/runner/runner.py#L757)). Because the explore counter is still 0, the banked checkpoint is named `checkpoint0*`. CLI `--eval_after_pretrain` ([DualDeviceNav_train.py:1527](training%20_scripts/DualDeviceNav_train.py#L1527)), wired at [:1154](training%20_scripts/DualDeviceNav_train.py#L1154).

**Rationale.** Establishes the held-out quality of the pretrained policy as the reference for every later online eval, and banks a clean pre-collapse checkpoint. Costs ~1 eval (~30 min) of wall-clock; default off.

## 14. `env_train_factory` — per-worker training-env factory (Gen-4 procedural anatomy)

**Context.** `Synchron` gives every worker a `deepcopy(self.env_train)` — identical anatomy. Gen-4 procedural training needs worker *i* to run a distinctly-seeded vessel.

**The change.** [synchron.py:301](eve_rl/eve_rl/agent/synchron.py#L301) adds `env_train_factory=None`; worker creation uses `worker_env_train = self._env_train_factory(i)` when set, else the legacy deepcopy ([L717](eve_rl/eve_rl/agent/synchron.py#L717)). Because a factory is an unserializable closure, it is stashed under the PRIVATE `self._env_train_factory` while the public `self.env_train_factory = None` placeholder ([L322](eve_rl/eve_rl/agent/synchron.py#L322)) satisfies `ConfigHandler`'s `getattr`-based config save (the eve None-placeholder convention). `env_eval` stays the shared held-out anatomy. Threaded through [agent.py:433](training%20_scripts/util/agent.py#L433); the `procedural_env_factory` closure and the `DualDeviceNavRCCAVaried` it builds live in `DualDeviceNav_train.py`/the env subsystem, gated by `--procedural_rcca` (default off → `env_train_factory=None` → legacy).

**Rationale.** Isolated, serialization-safe hook for per-worker anatomy variation with no change to the legacy single-master path.

## 15. Reward-version cache stamp (`meta_buckle_coef`) + fail-fast load guards

**Context.** Cached episode REWARDS are baked at harvest time. A cache scored with the anti-buckle potential (`buckle_reward_coef != 0`) has an identical obs layout to a coef=0 cache, so the existing obs-dim guard cannot catch a mismatch — the buffer would silently mix two reward MDPs, biasing the critic/advantages.

**The change.** [experience_cache.py:34](eve_rl/eve_rl/util/experience_cache.py#L34) reads the run's coefficient from env var `EVE_RL_BUCKLE_COEF` and every `save_episodes_npz` stamps `meta_buckle_coef` into the archive ([L94](eve_rl/eve_rl/util/experience_cache.py#L94)); `cache_buckle_coef(path)` reads it back ([L304](eve_rl/eve_rl/util/experience_cache.py#L304), absent field → 0.0). The training script exports `os.environ["EVE_RL_BUCKLE_COEF"]` before any worker spawns (so rolling-flush workers inherit it) and the heuristic- and heatup-cache load paths ([DualDeviceNav_train.py:293](training%20_scripts/DualDeviceNav_train.py#L293), [:352](training%20_scripts/DualDeviceNav_train.py#L352)) raise if `|cache_coef - run_coef| > 1e-9`.

**Rationale.** Prevents a silent reward-MDP mix. Absent env var and absent field both mean 0.0, so pre-Gen-4 caches stay valid for coef=0 runs (`--buckle_reward_coef` default 0.0 = OFF = frozen legacy reward).

## 16. Per-worker explore-target RNG reseed (RL_IMPROV_10 B1)

**The bug.** The intervention's target sampler creates `_rng = random.Random()` once in the parent; the deepcopy that ships the master env to each worker clones that exact RNG state, and unseeded explore resets never reseed it — so all workers sampled the IDENTICAL explore-target sequence.

**Fix.** [singelagentprocess.py:50](eve_rl/eve_rl/agent/singelagentprocess.py#L50) adds `_reseed_env_target_rng(env, seed_int)` (unwraps to `.intervention.target`, replaces `_rng` with a fresh `random.Random(seed_int)`); `run()` calls it on `env_train` only with a per-worker seed `hash((base_seed, worker_id, pid, "target_rng")) & 0xFFFFFFFF` ([L235](eve_rl/eve_rl/agent/singelagentprocess.py#L235)). `env_eval` is deliberately left alone — eval resets pass explicit seeds and reseeding would blur eval provenance. Logs success/skip.

**Rationale.** Decorrelates explore targets across workers without disturbing seeded eval determinism.


---

# Part F — harvest / stuck-screening / launchers

## 1. Reward-version guard — `meta_buckle_coef` cache stamp

### Context

Gen-4 adds an anti-buckle potential shaping term to env5 (gw slack + SOFA
contact proxy, delta form) gated by `--buckle_reward_coef`. A cached
episode's **rewards are baked at harvest time**, so a cache scored under
one coefficient silently mixes two reward MDPs if consumed by a run using
a different coefficient. The obs-dim guard **cannot catch this** — the obs
layout is identical; only the scalar reward differs.

### The change

**File:** [eve_rl/eve_rl/util/experience_cache.py](eve_rl/eve_rl/util/experience_cache.py#L34)

- New env var `_BUCKLE_COEF_ENV = "EVE_RL_BUCKLE_COEF"`; helper
  `_current_buckle_coef()` reads it (absent/malformed → 0.0).
- `save_episodes_npz()` stamps every archive with
  `meta_buckle_coef=np.float64(_current_buckle_coef())`
  ([experience_cache.py:94](eve_rl/eve_rl/util/experience_cache.py#L94)) — including
  the worker-side rolling heatup flushes, which **inherit the env var**
  from the parent process.
- New reader `cache_buckle_coef(path)`
  ([experience_cache.py:304](eve_rl/eve_rl/util/experience_cache.py#L304)) returns the
  stamp, `0.0` for pre-Gen-4 archives that predate the field.

The train script sets the env var **before any worker is created**
(`os.environ["EVE_RL_BUCKLE_COEF"] = repr(_buckle_coef)`,
[DualDeviceNav_train.py:399](training%20_scripts/DualDeviceNav_train.py#L399)) and
both cache-load sites fail fast on a mismatch: the heuristic-cache guard
([DualDeviceNav_train.py:764](training%20_scripts/DualDeviceNav_train.py#L764)) and
the heatup-cache guard
([DualDeviceNav_train.py:1014](training%20_scripts/DualDeviceNav_train.py#L1014)) both
`cache_buckle_coef(...)` the archive and `raise ValueError` unless it
matches the run's `--buckle_reward_coef` within 1e-9, with a message
telling the operator to pass the matching coef or re-harvest.

### Rationale

Pre-buckle caches (coef 0.0) stay valid for coef=0 runs. This is the
mechanism that lets `launch_rcca_harvest.sh` (coef 0.5) hand its
`seed.npz` to `launch_rcca_procedural_v1.sh` (coef 0.5) safely, while a
coef-mismatched reuse crashes at load instead of poisoning the buffer.

---

## 2. Matched procedural-ostium harvest → AWAC-from-ostium train flow

### Context

The procedural-siphon training run (`launch_rcca_procedural_v1.sh`) trains
from the **RCCA ostium with NO restore** — insertion comes from the tree
ostium, not an `--insertion_z`. It needs a seed buffer whose transitions
come from the **same MDP** it will train on (same varied meshes, same
relax semantics, same reward version), or the seed is off-distribution.

### The change

**New launcher:** [launch_rcca_harvest.sh](launch_rcca_harvest.sh) — a bounded
random-action harvester matched flag-for-flag to the training MDP:

| flag | value | why it must match the trainer |
|---|---|---|
| `--procedural_rcca` `--procedural_seed 12345` `--procedural_change_every 10` | same per-worker varied-siphon distribution | seed meshes overlap training meshes (seed 12345 shared) |
| `--relax_failure_truncations` | fold/off-path do **not** truncate | buckled transitions carry `done=False` (not a grounded terminal), matching training; long episodes capture live retract/unbuckle attempts |
| `--buckle_reward_coef 0.5` | anti-buckle potential ON | seed reward == training reward; stamped via §1 |
| `--heatup_only` `--heatup_episodes 480` | bounded, single-target RCCA | writes ONE consolidated `seed.npz` at `--save_heatup_cache` (runner single-target legacy save — no rolling chunks / concat) |

Output: `saved/rcca_proc_heatup/seed.npz`. The trainer loads it via
`--heatup_cache_file /opt/eve_training/results/rcca_proc_heatup/seed.npz`,
**guarded by `os.path.isfile`** so harvesting is OPTIONAL — an absent seed
makes v1 self-harvest inline. `--procedural_change_every 10` over 480
eps / 16 workers ≈ 30 ep/worker ≈ 3 meshes/worker.

**Trainer** [launch_rcca_procedural_v1.sh](launch_rcca_procedural_v1.sh): varies
ONLY the RCCA→RICA→siphon per worker (loaded arch fixed), starts at the
ostium, single-target `Centerline curve - RCCA.mrk`, `--algo awac --per`
`--replay_mode step`, `--pretrain_updates 10000`,
`--update_per_explore_step 0.5`, `--aux_coef 0.05 --aux_labels "2,3,5,6"`,
`--target_entropy 1.0`, `--log_std_min -2 --log_std_max 0.0`.

### Rationale

Random heatup **rarely threads the tortuous siphon** — the seed is
action-space coverage + rare reaches + buckle/recovery transitions, **not
clean demos**. It bootstraps the AWAC buffer; the recovery *skill* is
learned **online during explore via live in-situ buckling**
(`--relax_failure_truncations`), not from restore. This is the deliberate
alternative to the (unreliable, see §4) restore-based recovery curriculum.

---

## 3. Gen-4 #3 — mesh-matched stuck-state restore (no cross-mesh teleport)

### Context

Plan v9/v10's `CheckpointRestoreWrapper` picked a random checkpoint from
the pool and restored its SOFA `dof_positions` into whatever mesh the env
currently held. On a **fixed** mesh that's fine. On a **procedural** env,
where each worker's siphon is a different marching-cubes mesh, restoring a
state captured on mesh A into mesh B **teleports the wire into the wrong
geometry** — the mesh-bound SOFA state is meaningless in a different
vessel.

### The change

**File:** [training _scripts/util/checkpoint_restore.py](training%20_scripts/util/checkpoint_restore.py#L36)

- Stuck checkpoints embed their capture mesh as `_mesh-<fp>_` in the
  filename (`fp` = `RCCAVariedFromMesh.mesh_fingerprint`, e.g. `s12345g3`).
  `_mesh_fp_of_file()`
  ([checkpoint_restore.py:40](training%20_scripts/util/checkpoint_restore.py#L40))
  parses it via `_MESH_TAG_RE`; untagged legacy checkpoints (fixed
  DualDeviceNav: RVA, pre-bif11) resolve to the sentinel `"fixed"`.
- `_find_vessel_tree()`
  ([checkpoint_restore.py:47](training%20_scripts/util/checkpoint_restore.py#L47)) does
  a bounded (≤16) `gym.Wrapper` unwrap to the intervention's vessel tree.
- New ctor param `mesh_match: bool = True`
  ([checkpoint_restore.py:86](training%20_scripts/util/checkpoint_restore.py#L86)).
- `_eligible_checkpoints()`
  ([checkpoint_restore.py:189](training%20_scripts/util/checkpoint_restore.py#L189))
  computes the mesh-consistent subset: `mesh_match=False` → whole pool
  (legacy); a **pinnable** tree (`hasattr(tree,
  "regenerate_to_fingerprint")`) → only fingerprinted checkpoints; a
  fixed tree → only checkpoints whose fp == the tree's current
  `mesh_fingerprint`. Untagged states are **excluded** from a procedural
  tree — restoring one is exactly the invalid teleport this prevents.
- `reset()` picks from `candidates` (not the full pool), so `_pick_index`
  gained an `n=` param sizing the eligible subset
  ([checkpoint_restore.py:169](training%20_scripts/util/checkpoint_restore.py#L169)).
  Before the restore is injected, `_pin_mesh_for(path)`
  ([checkpoint_restore.py:211](training%20_scripts/util/checkpoint_restore.py#L211))
  calls `tree.pin_next(fp)` so the next `reset()` **regenerates that exact
  mesh** before SOFA rebuilds the scene — the restored `dof_positions`
  land in the geometry they were captured on.

If no mesh-eligible checkpoint exists, `reset()` falls through to the
normal ostium insertion — **defensive: never crash, and never restore a
mesh-mismatched state.**

### Rationale

For a fixed-mesh env every checkpoint parses to `"fixed"` → this is a pure
no-op (identical to legacy). Only the procedural path activates pinning.
This is what lets `--checkpoint_dir <stuck pool>` become compatible with
`--procedural_rcca` (v1's header documents the now-lifted mutual
exclusion): a recovery-curriculum run can restore mesh-varied stuck states
without teleport.

---

## 4. Gen-4 #6 — escapability + restore-fidelity screener for the stuck pool

### Context

Stuck checkpoints are captured at the `fold==10` / `off-path==25` triggers
with **no guarantee the state is recoverable**. A genuinely wedged state
in a restore pool teaches the critic *"recovery is impossible"*
(V(stuck) = failure-and-nothing-after) — poisoning the exact skill the
recovery curriculum is meant to build. Separately (the user's point), SOFA
restore of a **high-energy buckled state is imperfect**: the bow can spring
during settle, contact isn't fully re-captured, the wire can pop free or
penetrate — so a restored state may not even *be* the captured state.

### The change

**New pure module:** [training _scripts/util/escapability.py](training%20_scripts/util/escapability.py) —
verdict logic only, unit-testable without SOFA.

- `restore_faithful(captured_inserted, restored_inserted, captured_slack,
  restored_slack)`
  ([escapability.py:64](training%20_scripts/util/escapability.py#L64)): the restore
  must reproduce both the fed length (tol `RESTORE_INSERTED_TOL_MM=10`, a
  loose bound catching only gross "restore never landed") **and** the
  buckle bow / slack (tol `RESTORE_SLACK_TOL_MM=8`, catching a sprung
  buckle). Checked FIRST — an unfaithful restore is unusable regardless of
  retract behavior.
- `is_escapable(reason, slack_start, slack_end, retract_mm, on_path_end)`
  ([escapability.py:84](training%20_scripts/util/escapability.py#L84)): a state the
  scripted retract cannot MOVE (`retract_mm < MIN_RETRACT_MM=2.0`) is
  mechanically wedged → NOT escapable, whatever the reason. `fold`
  additionally requires the bow to relax (`slack` drops by
  `FOLD_SLACK_DROP_MM=5` **or** ends below `FOLD_SLACK_ABS_MM=8`);
  `offpath` treats retractability as sufficient (pure retract can't
  re-steer, so `on_path_end` is not required).
- `escape_metrics(...)` returns the verdict + the numbers behind it for
  per-checkpoint logging.

**New SOFA orchestrator:** [training _scripts/screen_stuck_pool.py](training%20_scripts/screen_stuck_pool.py) —
runs IN the training container. For each `stuck_*.npz`: `#3`-pins the tree
to the checkpoint's mesh fingerprint (procedural pools), restores via the
**same `BenchEnv5.reset(options={"restore_checkpoint": ...})` path
training uses**, checks `restore_faithful` against the `.json` sidecar's
capture refs, and — only if faithful — runs a scripted **pure retract**
(`action = [[-retract,0],[-retract,0]]`, no rotation) for `--steps` (40)
at `--retract_mm_s` (20). Escapable checkpoints (+ sidecars) are copied to
`--out_dir`; a `screen_report.json` records **restore-infidelity as its
own first-class bucket** and warns if >50% of the pool fails to survive
restore (in which case restore-based recovery is unreliable → prefer live
in-situ buckling via `--relax_failure_truncations`).

**New launcher:** [launch_screen_stuck.sh](launch_screen_stuck.sh) — one-shot
`--rm` container, `--procedural --procedural_seed 12345 --steps 40
--retract_mm_s 20`; drop `--procedural` (+ add `--insertion_z`) for a
fixed-mesh pool; `--dry_run` reports without copying.

### Respects the frozen heuristic

The screener **reads** `env._path_context.is_on_correct_path()` (via
`_on_path()`, [screen_stuck_pool.py:124](training%20_scripts/screen_stuck_pool.py#L124))
to report `on_path_end`, but does **not** modify the classifier — the
standing rule that `is_on_correct_path()` drives the heuristic's actions is
honored (read-only). The retract uses the env's own below-zero mask + per-
device velocity clip, so an over-retract cannot drive the wire past the
ostium; no reward/terminal is touched.

---

## 5. Buffer-filter `late_task_obs_index` — stale-since-Gen-3 constant corrected

### The bug

`FilterConfig.late_task_obs_index` (the flat obs index of the
`arc_past_last_daughter` guidance feature, used by the `*_strict` offline
filters) was **56**, computed for the old
`last_action=2 / guidance-start=44` layout. Gen-4's mesh-invariant obs
grew `last_action` to 4 and moved guidance to start at flat index 46, so
56 pointed at the wrong feature on any Gen-4-format PER archive.

### Fix

**File:** [training _scripts/util/buffer_filter.py](training%20_scripts/util/buffer_filter.py#L94)

`late_task_obs_index` → **58**. Gen-4 layout: `tracking[0..40)`,
`target[40..42)`, `last_action[42..46)`, `guidance[46..97)`,
`privileged[97..121)`; `arc_past_last_daughter` is guidance feature index
12 → flat `46+12 = 58`. The comment documents that the #5 log-depth
channel appended at guidance **END**, so indices 46..95 are stable across
the 50→51 guidance growth — the index is unaffected by that addition. Pure
index correction on an offline filter (reads obs, does not alter obs
layout); frozen-obs rule respected.

---

## 6. Docker individual-file-mount pattern (Gen-4 launchers)

### Context

The Gen-4 launchers mount **each modified/new source file individually**
over the baked `eve-training-fixed` image rather than rebuilding — new
modules `meshinvariant.py`, `rccavariedfrommesh.py`, `rccaprocedural.py`,
`carotidsiphon.py`, `dualdevicenavrccavaried.py`, `dualdevicenavprocedural.py`,
`checkpoint_restore.py`, `buffer_filter.py`, `buckle_reward.py`, plus the
edited `experience_cache.py`, `sac.py`, `gaussianpolicy.py`, `env5.py`, etc.

### The change (a mount asymmetry worth noting)

Most `eve`/`eve_rl` files are mounted to their `dist-packages` install
path. But `eve_bench/__init__.py` is mounted **only to
`/opt/eve_training/eve_bench/eve_bench/__init__.py`, NOT to
dist-packages** — while the concrete bench modules
(`dualdevicenav.py`, `dualdevicenavrccavaried.py`,
`dualdevicenavprocedural.py`, `archvariety.py`, `basicwirenav.py`) are
mounted to **both** locations. The dist-packages `__init__.py` is left
untouched **to preserve its DATA path** (the packaged `eve_bench` resolves
its centerline/mesh data dir relative to the installed `__init__`);
overmounting it would break data resolution. The `/opt` copy is the one
Python imports for the run (it's on `sys.path` first), so it picks up the
new `DualDeviceNavRCCAVaried` / `DualDeviceNavProcedural` exports while the
installed package keeps serving data.

### Rationale

Fast iteration without image rebuilds; the `__init__`-to-`/opt`-only rule
is the one subtlety that keeps procedural exports available while the
DATA-path-bearing installed `__init__` stays intact.

---

## 7. v1 → v2 launcher delta — the freeze-collapse fix flags

### Context

**New launcher:** [launch_rcca_procedural_v2.sh](launch_rcca_procedural_v2.sh) is
`launch_rcca_procedural_v1.sh` **+ the RL_IMPROV_15 freeze-collapse fix
package**. v1's deterministic-eval quality froze (8.2% → 13.3% → 3.1%; eval
speed 4.8 → 0.95 mm/s) then IPC-deadlocked after eval-3; a 5-agent forensic
traced it to `log_std` pinned at its ceiling + `alpha` decaying to the
`log_alpha` floor for 165k updates + AWAC weights ≈1.0 (no advantage
discrimination) → the action mean crushed toward 0.

### The change (launcher-level only)

v2 adds these flags/env-vars over v1 (the algo agent documents the
mechanisms themselves):

| added flag / env | value | one-line purpose |
|---|---|---|
| `--log_alpha_min` | `-5.0` | alpha floor so the entropy term never vanishes |
| `--log_alpha_max` | `-2.3` | alpha ceiling so entropy can't dominate BC/advantage and mean-crush |
| `--awac_lambda` | `1.0` (was 3.0) | restores advantage discrimination (AWAC stops degenerating to BC) |
| `--action_mean_penalty` | `0.005` | anti-rail `|atanh(mu)|` penalty |
| `--eval_after_pretrain` | — | run a deterministic eval right after the 10k pretrain |
| `-e EVE_CLEAN_RAIL_MAX` | `0.15` | keep railed "successes" out of the amplified clean lane |
| `-e EVE_RL_MODEL_QUEUE_TIMEOUT_S` | `900` | convert the no-timeout `_model_queue.get()` deadlock into a loud `RuntimeError` |

Everything else is unchanged from v1 (procedural RCCA/RVA variation, relax
recovery, buckle 0.5, aux 0.05 on `2,3,5,6`, `target_entropy 1.0`, PER +
`balanced_fraction 0.3`, `grad_clip 1.0`, soft `log_std (-2,0)`, seed.npz
+ 10k pretrain, 16 workers, `cuda:0`). The v2 header also documents that
the deterministic probe (start-state `|tanh(mu0)| > 0.10`, eval speed
≥ 3 mm/s) is the decisive monitor — explore looked healthy all through v1's
freeze because `std=1.0` noise masks a dead mean.


---

# Part G — Verification & current state

## Gen-4 first validation (`rcca_procedural_v1`, 2026-07-12)

The full Gen-4 stack ran end-to-end: 16 procedural-mesh workers, mesh-invariant
obs, asymmetric critic + aux, recovery relax + buckle shaping, AWAC on a
matched procedural-ostium seed (480 eps) + 10k pretrain. It **learned**:
explore success 1%→**34%**, held-out eval to **13.3%**, recovery successes
(off-path→re-approach→target) present, per-worker mesh diversity confirmed
(16 distinct path_lens), reward ranking correct, zero WBT/vessel_end
truncation leaks. Then it froze and deadlocked (Part E). Even in failure it
proved the generalization machinery: train↔eval gap tracked, no memorization
signature.

## RL_IMPROV_15 fix validation (`rcca_procedural_v2`, same day)

Restarted with the F1–F8 package. Every link of the v1 collapse chain was
re-exercised and held:

| Signal | v1 (collapsed) | v2 (fixed) |
|---|---|---|
| pretrain-only baseline eval (F7) | — (never measured) | 6.1% / 0.54 mm/s (the BC/freeze attractor, now a reference) |
| **eval1 held-out quality** | 8.2% @ 278k | **30.6% @ 287k** (2.3× v1's all-time peak of 13.3%) |
| eval1 det. speed | 4.78 mm/s | 4.45 mm/s |
| α at online equilibrium | 4.5e-5 (floored) → whipsaw 0.45 | **0.0067→0.052**, smooth, 0 band violations (F1) |
| entropy_proxy | crashed to 0.14 | glides to target ~1.0, no crater |
| `awac_weight_mean` | 1.00 (inert BC) | 1.007±0.010, max→2.8 (F2 discrimination alive) |
| `q1_mean` trend | −0.97→−4.71 sinking | flat ≈ −1.0 to −1.3 |
| clean-lane rail rejections (F4) | n/a | 0 of 211 online successes (successes genuinely non-railed) |
| deterministic start-state mean\|a0\| (F8) | 0.255→0.089 (regressing to attractor) | 0.032→0.116→0.083, **growing away** (2.6× baseline) |
| train↔held-out gap | 34% vs 13.3% | 31% explore vs 30.6% eval (≈0 → generalizes across procedural meshes) |
| IPC deadlock (F6) | hung forever (`anon_pipe_read`) | 0 timeouts; guard armed |

**The α lift-off (v1's kill shot) executed cleanly in v2:** entropy dipped
below target, `log_alpha` rose −5.0→−3.20, α settled at 0.052 — no whipsaw,
no mean-crush. The single most important structural result: **explore↔eval
gap ≈ 0** — the mesh-invariant obs + procedural anatomy generalize to the
held-out RCCA rather than memorizing the 16 training siphons.

## Monitoring instrumentation (standing)

`saved/monitor_rcca_procedural.md` holds the full v1 forensic + the running
per-pass verification log. A 2-hour cron re-runs
`scratchpad/monitor_pass.sh` (deterministic freeze-probe, α-band,
AWAC-discrimination, rail-filter count, eval trajectory, recovery relax,
IPC-guard). Forensic extracts preserved in `saved/v1_collapse_forensics/`.

## Open levers (next iteration)

- **J1 full fix** — give q2 its own embedder (currently shares q1's; no-op
  only while `embedder_layers=0`).
- **PER-correctness tier (D1–D5)** — deferred; revisit if switching off
  step-mode or if priority staleness is implicated.
- **Authored VMR meshes** — the `env_train_factory` hook is generic;
  swapping procedural RCCA for the report's "3 train + 1 held-out" authored
  meshes is a drop-in when true anatomical diversity (arch variants, not
  just siphon) is wanted.
- **Junction take-off-angle feature (Part 2 B2)** — declined so far; a
  candidate if fork-commit remains a failure mode.
- **Siphon-band curriculum** — v1/v2 both weakest at target z≥575 (the
  multi-bend siphon); biased target sampling there is the RL_IMPROV_10 §19
  lever, still open.


---

# Part H — File index

| File | Part / topics |
|---|---|
| [`eve/eve/observation/meshinvariant.py`](eve/eve/observation/meshinvariant.py) (NEW) | §3 TipRelativeTracking2D, §4 TargetTipOffset2D, §5 PrivilegedState (24-dim tail, aux-label block 2/3/5/6) |
| [`eve/eve/observation/localguidance.py`](eve/eve/observation/localguidance.py) | §2 30→51 dims: feature-0 mm-scale, feature-7 on_correct_path, path-preview 13-16/39-42, dual-device+fork 30-38, buckle/slip/calibre 43-49, log-depth 50, `_compute_path_preview`, removed `_phase_to_onehot` |
| [`eve/eve/observation/__init__.py`](eve/eve/observation/__init__.py) | §1 export TipRelativeTracking2D/TargetTipOffset2D/PrivilegedState |
| [`eve/eve/env.py`](eve/eve/env.py) | §7 `_on_intervention_stepped` post-step hook |
| [`training _scripts/util/env5.py`](training%20_scripts/util/env5.py) | §3-6 obs-stack wiring (TipRelative/TargetOffset/Privileged, ObsDict order, dropped InsertionLengths), §7 `_on_intervention_stepped` override + slip/branch mirrors |
| [`training _scripts/util/agent.py`](training%20_scripts/util/agent.py) | §7 `privileged_obs_dim`/`n_obs_policy`, separate `policy_embedder`; §9 `aux_coef`/`aux_label_rel_indices` validation → `aux_label_abs`, `n_aux` |
| [`eve_rl/eve_rl/network/gaussianpolicy.py`](eve_rl/eve_rl/network/gaussianpolicy.py) | §8 privileged-tail slice `[...,:n_observations]`, `n_aux` head + `_last_aux` |
| [`eve_rl/eve_rl/algo/sac.py`](eve_rl/eve_rl/algo/sac.py) | §9 aux MSE loss (`aux_coef`, `aux_label_indices`, padding-masked, labels from full-width states) |
| [`eve_rl/eve_rl/network/component/mlp.py`](eve_rl/eve_rl/network/component/mlp.py) | §10 ReLU after input layer |
| [`training _scripts/DualDeviceNav_train.py`](training%20_scripts/DualDeviceNav_train.py) | §7 `privileged_obs_dim` wiring, §9 `--aux_coef`/`--aux_labels`/`_parse_aux_labels` |
| [`eve/eve/intervention/vesseltree/rccavariedfrommesh.py`](eve/eve/intervention/vesseltree/rccavariedfrommesh.py) | §1 (RCCA-only bell-envelope perturbation, RVA proximal co-perturb, (11)-bridge insertion, re-mesh+temp cleanup), §2 (`mesh_fingerprint` s{seed}g{gen}, `_generation`, `regenerate_to_fingerprint`, `pin_next`, `parse_fingerprint`), §3 (ConfigHandler `self.<param>`, `branch_list=None`) |
| [`eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py`](eve/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py) | §5 (`right_carotid_siphon`, `SiphonParams`/`SampledSiphon`, Bouthillier C1–C7 CHS points, tortuosity meshing clip) — built, not wired to active run |
| [`eve/eve/intervention/vesseltree/rccaprocedural.py`](eve/eve/intervention/vesseltree/rccaprocedural.py) | §5 (`RCCAProcedural` synthetic Type-I arch with siphon RCCA swap) — built, not wired |
| [`eve/eve/intervention/vesseltree/__init__.py`](eve/eve/intervention/vesseltree/__init__.py) | §1/§5 (exports `RCCAProcedural`, `RCCAVariedFromMesh`, `perturb_rcca`) |
| [`eve/eve/intervention/vesseltree/aorticarcharteries/__init__.py`](eve/eve/intervention/vesseltree/aorticarcharteries/__init__.py) | §5 (exports `right_carotid_siphon`, `SiphonParams`, `SampledSiphon`) |
| [`eve/eve/intervention/target/centerlinerandom.py`](eve/eve/intervention/target/centerlinerandom.py) | §4 (`min_arclength_from_start`, `_arclength_from_start_mask`, per-branch+flat pool filtering) |
| [`eve/eve/util/polyline.py`](eve/eve/util/polyline.py) | §7 (`_project_onto_segment_range`, windowed `project_onto_polyline` with `prev_s`/`window_mm`/`fallback_dist_mm`) |
| [`eve/eve/util/pathcontext.py`](eve/eve/util/pathcontext.py) | §8 (`PROJECTION_WINDOW_MM`/`PROJECTION_FALLBACK_DIST_MM`, `_last_planned_s`, `reset_projection_continuity`, `_fork_geometry` independence guard); adjacent obs-support additions noted |
| [`eve_bench/eve_bench/dualdevicenavrccavaried.py`](eve_bench/eve_bench/dualdevicenavrccavaried.py) | §6 (active Gen-4 bench wiring `RCCAVariedFromMesh` + `target_min_arclength_mm`) |
| [`eve_bench/eve_bench/dualdevicenavprocedural.py`](eve_bench/eve_bench/dualdevicenavprocedural.py) | §6 (`DualDeviceNavProcedural` bench for `RCCAProcedural`) — built, not wired |
| [`eve_bench/eve_bench/__init__.py`](eve_bench/eve_bench/__init__.py) | §6 (exports both new benches) |
| [`eve_bench/eve_bench/dualdevicenav.py`](eve_bench/eve_bench/dualdevicenav.py) | §6 (symmetric velocity limits — drop `velocity_limit_low`, review 2.4) |
| [training _scripts/util/buckle_reward.py](training%20_scripts/util/buckle_reward.py) | §1 (NEW — `buckle_potential`, φ∈[−1,0], `SLACK_DEADBAND_MM`/`SLACK_CAP_MM`/`CONTACT_CAP_MM`/`W_SLACK`/`W_CONTACT`, loop-neutral delta form) |
| [training _scripts/util/env5.py](training%20_scripts/util/env5.py) | §1 (`buckle_reward_coef`, `_compute_buckle_potential`, slack vs privileged-contact proxy, None baseline, STEP-log `buckle_phi`, MultiTargetEnv5 propagation), §2 (`relax_failure_truncations`, vessel_end/fold/WBT gating), §4 (`MAX_STEPS_PENALTY=-3`, max_steps-first), §5 (`OFF_PATH_RETRACT_TAX`/`_MIN_OFF_STEPS`/`_MIN_MM`, `_last_delta_gw`), §6 (3 label-leak gates), §7 (`STUCK_CHECKPOINT_DIR`, `_save_stuck_checkpoint`, `STUCK_FOLD_TRIGGER`/`_OFF_BRANCH_TRIGGER`, mesh_fingerprint/slack_at_capture), §8 (`_last_restore_scene_mesh` re-arm, `reset_projection_continuity`), §9 (`_on_intervention_stepped` override) |
| [eve/eve/reward/arclengthprogress.py](eve/eve/reward/arclengthprogress.py) | §3 (2×-forward doubling killed → 1× symmetric, telescopes to net progress) |
| [eve/eve/env.py](eve/eve/env.py) | §9 (`_on_intervention_stepped` hook: state machine runs before obs/reward) |
| [`eve_rl/eve_rl/replaybuffer/pervanillastep.py`](eve_rl/eve_rl/replaybuffer/pervanillastep.py) | EVE_CLEAN_RAIL_MAX clean-lane rail filter, flips `reached` in push() (§7) |
| [`eve_rl/eve_rl/agent/singelagentprocess.py`](eve_rl/eve_rl/agent/singelagentprocess.py) | _model_queue_get IPC deadlock guard / EVE_RL_MODEL_QUEUE_TIMEOUT_S (§12), per-worker target-RNG reseed B1 (§16) |
| [`eve_rl/eve_rl/runner/runner.py`](eve_rl/eve_rl/runner/runner.py) | eval_after_pretrain baseline eval → checkpoint0 (§13) |
| [`eve_rl/eve_rl/agent/synchron.py`](eve_rl/eve_rl/agent/synchron.py) | env_train_factory per-worker env + serialization-safe private-name plumbing (§14) |
| [`eve_rl/eve_rl/algo/iql.py`](eve_rl/eve_rl/algo/iql.py) | per-dim exploration noise+clip (§8), reward_scaling in Bellman target (§9) |
| [`eve_rl/eve_rl/util/experience_cache.py`](eve_rl/eve_rl/util/experience_cache.py) | meta_buckle_coef reward-version stamp + cache_buckle_coef reader (§15) |
| [`launch_rcca_harvest.sh`](launch_rcca_harvest.sh) | §2 (NEW — matched procedural-ostium bounded harvester → seed.npz), §6 (mount pattern) |
| [`launch_rcca_procedural_v1.sh`](launch_rcca_procedural_v1.sh) | §2 (NEW — AWAC-from-ostium trainer, no restore, relax recovery), §6 |
| [`training _scripts/util/checkpoint_restore.py`](training%20_scripts/util/checkpoint_restore.py) | §3 (mesh-fp parse, `_find_vessel_tree`, `mesh_match`, `_eligible_checkpoints`, `_pin_mesh_for`, `_pick_index` n=) |
| [`training _scripts/util/escapability.py`](training%20_scripts/util/escapability.py) | §4 (NEW — `restore_faithful`, `is_escapable`, `escape_metrics`, thresholds) |
| [`training _scripts/screen_stuck_pool.py`](training%20_scripts/screen_stuck_pool.py) | §4 (NEW — SOFA restore+retract+fidelity screener, #3 pin) |
| [`launch_screen_stuck.sh`](launch_screen_stuck.sh) | §4 (NEW — one-shot screener launcher), §6 |
| [`training _scripts/util/buffer_filter.py`](training%20_scripts/util/buffer_filter.py) | §5 (`late_task_obs_index` 56→58 for Gen-4 obs layout) |
| [`launch_rcca_procedural_v2.sh`](launch_rcca_procedural_v2.sh) | §7 (NEW — v1 + freeze-collapse fix flags), §6 |
