# HANDOFF — endovascular navigation RL

Written 2026-08-27 for a machine move. This is the operational and factual context that
otherwise lives only in a chat transcript. Read §0 and §1 before running anything.

---

## 0. MOVING MACHINES — READ FIRST

**Every launcher hardcodes the absolute repo path in its Docker mounts.**
`D:\Arjun\workspace\neve` appears on **68 lines** in the training launcher, **69** in the
eval launcher, across **15 launcher files** and 18 files repo-wide. None parameterise it.

If the repo is not at that exact path on the new machine, **every launcher fails**. Two
options:

1. **Put the repo at the same absolute path.** Zero work, and it also makes the Claude Code
   session transcript resumable without renaming anything.
2. **Rewrite the paths**, e.g.
   `sed -i 's#D:\\Arjun\\workspace\\neve#<NEW>#g' launch_*.sh` — then verify one mount by
   hand, because a wrong mount fails as a confusing in-container ImportError rather than a
   missing-file error.

**Not in git** (too large; move by drive or cloud):
`saved/lcca_awac_seed_v1.npz` (157 MB — above GitHub's 100 MB hard limit),
`saved/rcca_proc_heatup/` (65 MB), `saved/steplog_rcca_proc_harvest_1813.txt` (53 MB),
`saved/v1_collapse_forensics/` (30 MB), `saved/eval_smoke/` (5.5 MB). None are needed to
reproduce current results.

**Docker image** `eve-training-fixed` must exist on the new machine. It is not in the repo.

---

## 1. WHAT THE SYSTEM IS, IN ONE PAGE

SOFA finite-element simulation of a **guidewire + microcatheter** advanced coaxially through
a patient-derived carotid tree. Four continuous commands: translation rate (±30 mm/s) and
twist rate (±1.5 rad/s) per device, at 7.5 Hz, 600-step episode cap. Success = guidewire tip
within 5 mm of a target sampled ≥40 mm into the RCCA.

**The policy is a RESIDUAL on a scripted controller — this is the single most misunderstood
thing about the setup.** The executed action is

```
a_total = clip(a_heuristic + residual_scale * a_policy)      residual_scale = 1.0
```

where `a_heuristic` is `CenterlineFollowerHeuristic`: a parameterless proportional
controller on heading error and cross-track offset that retracts when off-route and advances
otherwise, with the catheter following at 0.8× the guidewire rate.

Consequences that matter:
- **At initialisation the system *is* the centerline follower.** Training starts at heuristic
  competence, not at zero.
- **`checkpoint0` is therefore the PURE centerline follower** — `--pretrain_updates 0
  --eval_after_pretrain` evaluates it before any learning. That is the run's baseline, H0.
  Every later number must beat it.
- `residual_scale = 1.0` means the network commands the **full** actuation range and can
  override the controller completely. "Residual" understates its authority.
- There is **no demonstration data and no offline seeding.** The buffer fills purely online
  after a 20k-step warm-up at `--heatup_action_scale 0.3` (= heuristic + small noise).

**Observation**: 125 dims = a **101-dim deployable prefix** (projected device tracking,
target offset in the image plane, previous command, planned-route guidance, the heuristic's
own action) + a **24-dim privileged tail** (nodal forces/velocities, contact-impulse proxy,
accumulated twist, ground-truth branch identity).

⚠ **`--privileged_actor` sets `privileged_obs_dim = 0`, which makes the policy's input slice
a no-op — so the POLICY READS THE PRIVILEGED TAIL AT TRAINING AND TEST TIME.** The shipped
models are **teachers, not deployable controllers**. No student was ever trained. Every
headline number is a teacher number.

**Algorithm is SAC**, not AWAC. Both shipped launchers pass `--algo sac`; the AWAC code path
is gated behind `self.algo == 'awac'` and is inert (`awac_lambda: 3.0` still appears in
saved configs but is dead).

---

## 2. OPERATIONS

### 2.1 Train

```bash
bash launch_rcca_p2_teacher_v1bp.sh          # or _v1b.sh
docker logs -f rcca_p2_teacher_v1bp          # monitor
docker stop -t 60 rcca_p2_teacher_v1bp       # stop
```

Entry point inside the container:
`python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py`.
Runs detached (`-d`), `--gpus all --shm-size=30g --init`, 16 CPU workers + 1 GPU trainer.

**Flags that define the experiment** (v1bp; v1b is identical minus the three marked ⋆):

| flag | meaning |
|---|---|
| `--env_version 5` | env5.py — the residual/observation/reward stack |
| `--procedural_rcca --procedural_seed 12345 --procedural_change_every 10` | per-worker generated anatomies, redrawn every 10 episodes |
| `--residual_heuristic --residual_scale 1.0` | the residual composition above |
| `--heur_action_obs` | puts the heuristic's 4-dim intent in the deployable prefix (obs 121→125) |
| `--privileged_actor` | policy consumes the privileged tail — makes it a TEACHER (requires `--aux_coef 0`) |
| `--heatup_action_scale 0.3` | warm-up = heuristic + small residual noise |
| `--pretrain_updates 0 --eval_after_pretrain` | **checkpoint0 = pure heuristic = H0 baseline** |
| `--algo sac --critic_layernorm --no_entropy_backup` | SAC, LayerNorm in critics only, entropy omitted from the critic backup |
| `--target_entropy 1.0 --log_alpha_min -5.0 --log_alpha_max 0.0` | raised target entropy (not −n_actions), α railed |
| `--log_std_min -2 --log_std_max 0.0` | soft-bounded σ ∈ (0.135, 1.0) |
| `--hidden 256 256 --embedder_layers 0` | two-layer MLPs, no recurrence |
| `--per --balanced_fraction 0.3` | prioritised replay + 30% of each batch from episodes reaching the correct daughter |
| `--replay_mode step --replay_buffer_size 2000000` | 2M transitions, batch 256 |
| `--update_per_explore_step 1.0` | UTD 1.0 nominal; **measured 0.99** |
| `--relax_failure_truncations` | vessel-end truncation OFF so off-path excursions stay recoverable |
| `--buckle_reward_coef 0.5` | potential-based anti-buckling term |
| ⋆ `--cath_slack_coef 0.5` | v1bp only — catheter-slack potential |
| ⋆ `--progress_tip_mode avg --avg_gw_weight 0.5` | v1bp only — tip-average progress |
| `-nw 16 -d cuda:0` | 16 workers, GPU trainer |

**Measured throughput: 10–11 environment steps/s** (v1b 10.0, v1bp 11.4), UTD 0.99. Ignore
the ~47 steps/s and UTD 0.25 figures in `PAPER_PLAN` — those are from a different era.

**Startup checks** (from the launcher header, still valid):
- The baseline eval before explore = pure heuristic quality. Everything after must beat it.
- Explore success should **start near the baseline**. If it starts near 0, the residual
  composition is broken — kill and debug.
- `q1_mean` sustained below −2.5 = critic diverging.

### 2.2 Evaluate

```bash
bash launch_eval_anatomies.sh <ckpt-path-inside-container> [n_episodes]
```

Defaults: 98 episodes, 16 workers, `SEED_BASE=900000`, `CHANGE_EVERY=2`, `MAX_STEPS=600`,
deterministic policy, snapshots on. Override via env vars (`NAME`, `EXTRA_FLAGS`,
`ARCH_FLAGS`, `ENV_FLAGS`, `N_WORKER`, …).

`ARCH_FLAGS` **must match the checkpoint** or the strict state-dict load fails loudly:
`--residual_heuristic --heur_action_obs --privileged_actor --critic_layernorm`.

Key eval flags:

| flag | meaning |
|---|---|
| `--real_patient_anatomy` | pin the ORIGINAL segmented surface. **Do not remove** — see trap 1 |
| `--require_passable` | reject/regenerate until the anatomy admits the guidewire AND median clearance ≥ `--passable_min_median_mm` (2.00) |
| `--radius_scale 1.6` | compensates the mesher's ~37% radius erosion; reproduces patient clearance |
| `--stochastic_eval` | sample instead of tanh(mean). Default off. Result was a clean null |
| `--insert_inside_branch {none,RCCA,LCCA}` `--insert_point_idx` | LCCA transfer experiment (§5) |

Outputs land in `<ckpt-dir>/eval_anatomies_<ckpt>/`: `episodes_official_<ts>.jsonl` (the
authoritative per-episode record), `episodes.csv`, `logs/<ts>/worker_*.log`,
`snapshots/<ts>/eval/<BRANCH>/<outcome>/*.png`.

### 2.3 Analysis tooling (`monitoring/`)

| script | does |
|---|---|
| `mesh_clearance.py` | passability gate — clearance vs wire radius, **no controller needed** |
| `mesh_ablation.py` | mesher parameter ablation — **WRITTEN, NEVER RUN** |
| `extract_stuck.py` | per-episode stall/recovery records from worker logs |
| `analyze_stuck.py` / `verify_stuck.py` | the tables and the verification pass |
| `report_stuck.py` / `report_single.py` | the PDF reports |
| `lcca_preflight.py` | LCCA experiment V0+V1 geometry/clearance preflight |
| `verify_wall.py` / `verify_walled.py` | arrest-station forensics from eval logs |

---

## 3. RESULTS LEDGER

### 3.1 Current, verified

| what | number | protocol |
|---|---|---|
| Held-out synthetic, best model | **84.7%** (83/98) | v1bp ckpt2002292, 50 calibrated + gated anatomies |
| — by depth | CCA 100% (26/26), ICA-mid 100% (40/40), siphon 53.1% (17/32) | |
| Held-out synthetic, earlier ckpt | 83.7% (82/98) | v1bp ckpt514264 |
| **Real patient, best model** | **75.5%** (74/98) | v1bp ckpt2002292, ORIGINAL surface |
| — by depth | CCA 100%, ICA-mid 90.2%, siphon 33.3% | |
| Real patient, earlier ckpt | 72.4% (71/98) — siphon **70.0%** | v1bp ckpt514264 |
| Real patient, v1b | 63.3% (ckpt3259127) / 52.0% (ckpt757854) | |
| **LCCA transfer, best model** | **0/98 = 0.0%** | different vessel, same patient |
| LCCA transfer, earlier ckpt | 13/98 = 13.3% | |
| LCCA control (RCCA-internal) | 68/98 = 69.4% | same one-branch topology, trained vessel |

**Best checkpoints are NOT the ones the in-run eval crowned.** True best by explore success:
v1b **3259127** (63.2% local explore), v1bp **2002292** (59.2%). The eval-crowned 757854 and
514264 sit at ~1/5 of training and were selected by a single-anatomy signal.

### 3.2 RETRACTED — do not quote these

| number | why it is wrong |
|---|---|
| real patient 35.7% | measured on a re-meshed reconstruction, not the patient surface |
| 50-generated 57.1% / 55.1% | measured on anatomies ~2/3 of which are geometrically impassable |
| "siphon is an absolute wall, 0/30" | the arrest was at proj_s 153.4 mm — **57 mm BEFORE** the siphon band |
| "the reward pair is a null result" | on the corrected mesh it is +20.4 pp overall, +63.3 pp siphon |
| "the generator makes EASIER vessels than reality" | inverted — it makes *impassable* ones at half the calibre |
| "both runs ended by OOM" | IPC-deadlock watchdog (`os._exit 42`); `grep OOM` = 0 hits in both logs |
| throughput ~47 steps/s, UTD ~0.25 | measured 10–11 steps/s, UTD 0.99 |
| "recovery never emerged, soft 2–6%" | detector-dependent; canonical detector gives 22.6% / 29.5% |
| r = −0.82 avoidance-not-escape | **PROVISIONAL** — computed on walled anatomies, never re-stratified |

---

## 4. RULED OUT — do not re-try

1. **RLPD / from-scratch SAC** — P1 closed negative, evals 0.0/0.0. Compute-starved at
   realised UTD 0.39 against the paper's 20.
2. **Stochastic evaluation as a lever** — clean null, strict superset, +1 episode of 98. (It
   was measured on the walled surface, so it is cheap to redo, but the prior is weak.)
3. **Longer step budget** — a 1000-step rerun converted **zero** extra successes. Eval
   failures are jammed, not slow.
4. **Lumen erosion "refuted"** — that refutation was itself wrong (nearest-vertex artifact).
   Erosion is real, ~45%.
5. **Mesh coarseness as the wall's cause** — the original collision mesh is equally coarse
   (6.28 vs 6.46 mm median edge) and IS passable.
6. **Collision-chord discretisation as the cause** — the historical fixed-mesh run used the
   same devices and traversed the siphon.
7. **Depth-stratified checkpoint selection** — the real-patient siphon split (70.0 vs 33.3)
   did not reproduce across 50 anatomies (53.1 vs 50.0, one episode). Anatomy-specific.
8. **AWAC** — abandoned; its advantage weights collapsed toward uniform behaviour cloning of
   the buffer. Phrase as "in our setting", never as a property of AWAC.

---

## 5. TRAPS THAT COST CYCLES HERE

1. **`--real_patient_anatomy` used to evaluate a RECONSTRUCTION, not the patient.** Zeroing
   the perturbation amplitudes reproduces the patient *centerlines* to zero float error —
   which is why it passed every check — but `RCCAVariedFromMesh` always re-meshes the
   *surface*, and the wire collides with the surface. Fixed by pinning the original `.obj`.
   35.7% → 75.5%.
2. **Never measure lumen clearance from mesh VERTICES.** On a 6 mm-triangle mesh the vertices
   sit outside the facets; it reported 1% erosion where the truth is ~45%. Sample the
   triangle *surfaces* (`mesh_clearance.py` does).
3. **`vt.insertion` alone is silently reverted** — `_generate()` re-asserts
   `self.insertion = self._insertion`. Set the **private** `_insertion`.
4. **Workers are SPAWNED, so anything reachable from the env must pickle.** A closure bound
   to an instance, or a class defined inside a function, both fail with
   `Can't pickle local object`. Patch the CLASS method and leave only plain strings/arrays on
   the instance (`_install_pinned_surface_patch` is the working pattern).
5. **`CenterlineRandom` silently falls back to the RCCA pool** if the requested branch is not
   in `_branch_targets`, and it only rebuilds its pool when
   `_branches_initialized != vessel_tree.branches`. Set `branches`, `min_arclength_from_start`
   AND `_branches_initialized = None`, or the run is logged as one vessel and navigates
   another.
6. **Explore/eval separation**: use the `seed=` field in `EPISODE_START` (eval resets carry a
   seed, explore resets do not). main.log time windows leak — they caught 567 of 980.
7. **`episode_summary.jsonl` logs only ~36% of explore episodes.** Rates only, never counts
   or timing. Worker-log `EPISODE_OUTCOME` is the complete ledger.
8. **Three different "successes" at different units** — episode success (per episode), stall
   resolution (per *event*), and conversion (per episode). Conflating the first two produced
   a confidently wrong reading.
9. **Stall-detector thresholds move absolute rates by ~20 pp.** Report deltas and trends,
   never levels.
10. **`--require_passable` hardcodes `'RCCA' in name` behind a bare `except: return True`** —
    so it gates on the wrong vessel, without erroring, for any non-RCCA experiment.
11. **Always smoke-test 4 episodes and check `path_len` before a 98-episode run.** A start-
    point change once produced path_len 594–752 mm instead of 103–156 mm — a different task
    entirely — and only the smoke test caught it.

---

## 6. OPEN ITEMS

**Substantive**
1. **The mesh generator still produces uncalibrated, largely impassable training anatomies.**
   The *eval* path is worked around by `--radius_scale`, but training is not. Diagnosis is
   measured; `monitoring/mesh_ablation.py` is written and **never run**.
   See `MESH_GENERATOR_FIX_PLAN.md` §5.
2. **LCCA transfer is 0%.** The policy has learned the RCCA course, not CCA→ICA navigation.
   Genuine anatomy generalization needs training anatomies differing in *course and
   topology*, not only tortuosity of one fixed course. See `LCCA_TRANSFER_RESULT.md`.
3. **Continued training made transfer worse** — earlier checkpoint 13.3% on LCCA vs 0% for
   the later, better-on-RCCA one (Fisher p ≈ 0.0003).

**Owed re-analysis — CPU only, data in hand**
4. Re-stratify r = −0.82 on audited anatomies (blocks a paper figure).
5. v1b calibrated-synthetic eval (~1.5 h) — cancelled mid-queue, leaves a hole in the table.
6. LCCA arm-3 control (RCCA-internal with ckpt514264, ~1 h) — needed to state the
   checkpoint-ordering claim rigorously rather than by inference.
7. Bucket LCCA arrests against `saved/lcca_clearance_profile.npy` (tight bands at arclength
   130 / 223 / 266 mm; arrests occur at ~20 mm so this is unlikely to change the conclusion).

**Built, never launched**
8. v1c crunchpass lane — ⚠ **premise contested**: grind-only episodes beat soft in *both*
   runs (99%/95% vs 95%/76%); recovery type appears to proxy stall *severity*, not skill.
9. v1d (v1c + reward pair); E3 stuck lane.

**Never built**
10. P2b student distillation — the gate ("worth doing once the real-anatomy number is worth
    inheriting") is now passed at 75.5%.
11. A genuinely asymmetric run (`privileged_obs_dim > 0`). Believed running on machine 2,
    unverified from here.
12. The planned-path ↔ force correspondence probe.

**Infrastructure debt**
13. Trainer restart must RESTORE state, not re-init.
14. Re-arm probe logging after trainer restart.
15. Remaining invariant alarms (snapshot-hash-unchanged, init-wipe signature).
16. Item formerly "memory cap / replay trim" → **re-scope to IPC-deadlock robustness**; OOM
    was a misdiagnosis.
17. Reproducible init dump (`policy_0.pt`) and launch seeding.

---

## 7. PAPER STATUS

SPIE 4-page paper drafted. Two Methods variants exist: `spie_methods.tex` (teacher–student)
and `spie_methods_asymmetric.tex` (asymmetric actor-critic); `spie_methods_v2.tex` is the
restructured version that follows the order of discovery. The compiled draft
(`saved/p2a_deep_dive/SPIE_2027_paper.pdf`) uses the asymmetric framing.

`spie_refs.bib` holds 82 verified references; `SPIE_CITATIONS.md` carries the placement
table plus **12 gaps and 12 risks**. Highest-priority items from that audit:

- **The 2.5–3.4 mm lumen figure in Methods is wrong** (real: CCA 6.1–6.5, cervical ICA
  4.7–5.1, cavernous ICA 4.3 mm). Highest-probability reviewer catch.
- **The radiation-harm sentence contradicts its own citation** (specialty-matched source says
  operator dose is "generally low"). Pivot to dose *scaling* with attempts.
- **HERMES reports an ordinal mRS shift, not a functional-independence rate**, and it is five
  trials not seven.
- **No "first dual-device siphon" or "highest success" claim survives** — Robertshaw 2025
  reaches M1 through the siphon at 96% on 12 anatomies.
- **Moosa 2025 benchmarks TD3 68% > SAC 58%** on the same stEVE task; preempt "why not TD3".
- ⚠ **The generalization claim needs re-scoping given §6 item 2** — held-out synthetic
  anatomies are deformations of the RCCA, so 84.7% measures robustness to tortuosity of one
  course, not transferable carotid navigation.

---

## 8. FILE MAP

| path | what |
|---|---|
| `RL_IMPROV_18_P2_DESIGN.md` | design doc. **§1–8 are 2026-07-28 and partly superseded; §9 has the corrections** |
| `HANDOFF.md` | this file |
| `STATE_REPORT_2026-08-24.md` | pre-check-in audit: committed vs not, stale doc claims |
| `MESH_GENERATOR_FIX_PLAN.md` | parked mesh fix — diagnosis, hypothesis, resume point |
| `LCCA_TRANSFER_RESULT.md` | the different-vessel transfer result |
| `ANATOMY_GENERATION_RATIONALE.md` | why the generator's parameters are what they are |
| `saved/p2a_deep_dive/GEOMETRIC_WALL_VERIFIED.md` | the full mesh investigation, in order |
| `saved/p2a_deep_dive/STUCK_RECOVERY_ANALYSIS.md` | stall/recovery analysis + verification |
| `saved/monitor_rcca_procedural.md` | running monitor log, all passes |
| `spie_*.tex`, `SPIE_*.md`, `spie_refs.bib` | the paper |
| `training _scripts/DualDeviceNav_train.py` | training entry point |
| `training _scripts/util/env5.py` | env, observations, reward, residual composition |
| `training _scripts/eval_anatomies.py` | standalone evaluator |
| `eve/eve/intervention/vesseltree/rccavariedfrommesh.py` | procedural anatomy generation |
| `eve/eve/intervention/vesseltree/util/meshing.py` | the re-mesher (the defect lives here) |
| `monitoring/` | all analysis tooling (§2.3) |

---

# 9. TOPBRAIN ANATOMY SET — operations and findings (2026-08-28)

Appended, not a rewrite. Sections 0–8 above are unchanged and still accurate except where
this section explicitly supersedes them.

## 9.1 What the set is

25 anatomies at `topbrain_data/anatomies/`, built on branch `rl_improv_16_resume` (build
record: `TOPBRAIN_PIPELINE.md`, pulled into this branch). They are **the shipped host
patient's own tree** — arch, brachiocephalic trunk, cervical carotid — with a different
real patient's **right ICA siphon** grafted onto the distal third, sourced from the
TopBrain 2025 MICCAI release (Zenodo 16878417; 25 MR/CT pairs; open Swiss licence,
non-commercial, **attribution mandatory**, data owner University Hospital Zurich).

Measured, not assumed:

- 15 of 16 centerlines are byte-identical across all 25 **and to the host's**.
- The RCCA courses coincide to rms < 0.01 mm out to **s = 133.6 mm** (the graft seam).
- Declared radii diverge 31 mm earlier, at **s = 102.5 mm** — the pipeline overwrote the
  host's own mid-ICA narrowing (host r 1.27–1.60 mm) with a smoothstep ramp UP to
  2.25–2.90 mm. The "shared" band is therefore **wider** than the host's, not identical.
- Coordinate mapping, derived from 9,133 STEP samples: **s_RCCA = path_len − 33.31 mm**
  (sd 0.042). The seam is path_len **166.91 mm**.
- Only `mr_025` has a mid-vessel blockage touching any evaluated target (s = 166.8–168.3 mm,
  min clearance 0.218 mm). At the 0.18 mm guidewire radius there are **zero** mid-vessel
  blockages anywhere in the cohort. Cohort reachability ceiling 97.5–100%.

## 9.2 Running TRAINING on the set

```bash
bash launch_rcca_topbrain_v1.sh          # container rcca_topbrain_v1
docker logs -f rcca_topbrain_v1
docker stop -t 60 rcca_topbrain_v1
```

The launcher is `launch_rcca_p2_teacher_v1bp.sh` with the anatomy source swapped and four
extra mounts. Every algorithm flag is byte-identical to v1bp. New flags, all default-OFF —
a run without `--topbrain` is byte-identical to legacy (proven: 277 insertions, 0 deletions
in `DualDeviceNav_train.py`; paired before/after config dumps differ only in `_id` memory
addresses):

| flag | meaning |
|---|---|
| `--topbrain` | use `DualDeviceNavTopBrain` instead of `--procedural_rcca`. Mutually exclusive; raises on both. Requires `--env_version 5`. |
| `--topbrain_dir` | container path to `anatomies/` |
| `--topbrain_seed` | base seed; worker *i* gets base+*i* (as `--procedural_seed` does) |
| `--topbrain_change_every` | episodes between anatomy switches (10) |
| `--topbrain_exclude` | anatomies dropped from BOTH train and eval |
| `--topbrain_holdout` | reserved for eval; added to the train exclude so no worker sees them |
| `--topbrain_target_min_arclength` | **the important one.** Minimum target arclength from the RCCA ostium. Default 40.0 = legacy. **Use 133.0** — see 9.4. |

Extra mounts beyond the v1bp set (Fix D5: do NOT add the loader to
`vesseltree/__init__.py` — that file is bind-mounted by 10 launchers that lack the module):
`topbrainanatomyset.py` → the INSTALLED eve package path; `dualdevicenavtopbrain.py` →
BOTH eve_bench locations; `topbrain_data` → `/opt/eve_training/topbrain_data:ro`.

## 9.3 Running EVALUATION on the set

```bash
TOPBRAIN_ONLY="topcow_mr_007 topcow_mr_008 topcow_mr_017 topcow_mr_022" \
NAME=eval_tb CHANGE_EVERY=1 \
  bash launch_eval_topbrain.sh <checkpoint-path-inside-container> 220
```

`launch_eval_topbrain.sh` is `launch_eval_anatomies.sh` plus the four mounts, defaulting
`EXTRA_FLAGS` to `--topbrain` and the three v1bp reward-pair flags that are NOT in the
inherited defaults (`--cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight 0.5`).
Omitting those three changes the observation width and the strict load fails loudly.
Omit `TOPBRAIN_ONLY` to evaluate all 22 retained anatomies.

Output lands in `<ckpt-dir>/eval_anatomies_<ckpt-name>/`. **`episodes.csv` and
`anatomy_success.csv` are OVERWRITTEN by each run**; the per-run authoritative records are
the timestamped `episodes_official_*.jsonl` and the `logs/<ts>/`, `snapshots/<ts>/` dirs.

**Anatomy assignment is a pure function of the eval seed.** Two runs with the same
`SEED_BASE` and episode count are matched pair-for-pair — verified: 220/220 shared seeds
drew the identical anatomy AND the identical target. Use paired statistics.

## 9.4 The finding that dictates the target-depth flag

At the legacy 40 mm, **47–51% of targets land proximal to the seam**, on geometry
byte-identical across every patient. Measured on `ckpt2002292` (the v1bp teacher, trained
on the procedural set, so all 22 are unseen to it), 220 episodes, all 22 anatomies:

| targets | teacher | H0 (checkpoint0) | delta | p |
|---|---|---|---|---|
| shared host course | **96/96 = 100%** | 66/96 = 68.8% | +31.2 pp | <0.0001 |
| **grafted (unseen) anatomy** | 69/124 = **55.6%** | 71/124 = **57.3%** | **−1.6 pp** | **0.90** |
| all | 165/220 = 75.0% | 137/220 = 62.3% | +12.7 pp | 0.0054 |

**Training on the procedural set bought a large, highly significant gain on the anatomy it
trained around, and nothing measurable on unseen anatomy.** By section both controllers
degrade monotonically with depth (H0 73.2 → 61.8 → 52.1%; teacher 100 → 77.6 → 47.9%), and
the teacher's advantage is proximal only. This is the LCCA 0/98 result measured a second,
independent way.

Hence `--topbrain_target_min_arclength 133.0`: it deletes the free episodes so the run
trains and is scored only where the headroom actually is.

## 9.5 RETRACTED — do not cite these

- **"91.8% on unseen patients."** Inflated three ways: a holdout of four anatomies picked
  for being defect-free; 44% of episodes on shared host course (100% of which succeed); and
  a shallower target mix. All-22 is 75.0%, and 55.6% on grafted targets.
- **"Siphon 84.6% vs the host's 33.3%."** H0 scores ~78.6% on the same anatomies. The
  teacher added nothing in the siphon.
- **"The cohort meshes are deflated and therefore harder."** Refuted by re-baking the HOST
  through the cohort's own mesher: the cohort trunk is then +1.0% WIDER, 20/22 wider.
  96–99% of the apparent gap is the shared mesh generator, not the patients.
- **Any section-level number from a 4-anatomy subset.** Intervals there are ±25 points.
  Only the 220-episode all-22 runs support a trend.

## 9.6 Anatomy exclusions, with the measurement behind each

| anatomy | why |
|---|---|
| `mr_015` | surface ends ~20 mm short; 23 centerline points outside it, up to 7.19 mm |
| `mr_013`, `mr_014` | centerline outside mesh (5 pts / 1.37 mm; 3 pts / 2.11 mm) |
| `mr_025` | mid-siphon pinch at s = 166.8–168.3 mm that no distal trim removes. Its 9 eval episodes split perfectly on it: 4 targets proximal ALL succeeded (30–75 steps), 5 distal ALL failed by timeout. Excluding it takes 44.4% → 100% (Fisher p = 0.0079). Unreachable targets would feed unearnable negative reward into training. |

Retained: 21. The current run holds out `mr_007 / mr_008 / mr_017 / mr_022`, stratified
across the observed difficulty range — **not** the earlier `004/008/017/023`, which were
picked for cleanliness and produced the inflated 91.8%.

## 9.7 Open on this set

- The host collision surface reads **1.65× its own declared radii** (1.94–2.27× over the
  siphon) while the host VISUAL mesh reads 1.0019. If confirmed, every host number — siphon
  ceiling, 75.5%, `--real_patient_anatomy` — was obtained inside a vessel roughly twice as
  wide as the patient's. Half a day of signed-distance work to settle. **Highest-stakes
  open item in the project.**
- The mesher applies a uniform **−0.31 mm** inward offset (half its 0.6 mm voxel) to every
  surface it makes, host included. Fixable by pre-dilating radii or gridding at 0.3 mm,
  into a NEW directory — never overwrite.
- The **167–200 mm dip** (59.5%, below the 82.6% of the band beyond it) survives every
  reachability correction with zero episodes excluded. Pure policy defect, spread over 12
  anatomies, and the most informative anomaly in the data.
- `--topbrain` + `--checkpoint_dir` **crashes every worker** (`checkpoint_restore.py` admits
  any non-`fixed` tag without checking it belongs to this anatomy set). Neither launcher
  passes `--checkpoint_dir`, so it is latent, but it needs a guard.

## 9.8 First TopBrain training run — `rcca_topbrain_v1`, 2026-08-28

Launched at 133 mm target depth, 17 training anatomies, 4 held out, all other flags
byte-identical to v1bp. Eval history:

| eval | explore eps | steps | success | steps-to-success | translation speed |
|---|---|---|---|---|---|
| 1 (H0 baseline) | 0 | 0 | **73.5%** | 497 | 3.64 mm/s |
| 2 | 2,200 | 256,370 | **99.0%** | 68 | 25.7 mm/s |
| 3 | 5,300 | 505,230 | **99.0%** | 71 | — |

Saturated by the second eval. **Unverified at time of writing**: the policy drives at
~86% of the 30 mm/s insertion limit, a 7× speed increase over the heuristic. Before this
number is used anywhere, check `buckle_phi` and the `fold=n/20` counter — a 99% obtained by
ramming rather than tracking is a simulator artifact, not a policy. Checkpoints stored:
`checkpoint256370`, `checkpoint505230`.
