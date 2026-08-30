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

---

# 10. EVALUATION PROTOCOL — validation vs test (2026-08-30)

Appended. Supersedes nothing in §9 except where marked.

## 10.1 The two evaluations are different instruments and must not be conflated

**VALIDATION — runs automatically during training.** `env_eval` inside the training
process, 98 episodes on the 4 held-out TopBrain anatomies, fired every
`EXPLORE_STEPS_BTW_EVAL` = 250,000 explore steps. It writes the `quality` / `success`
columns of `<run>.csv`. Its job is liveness and a sanity floor.

**TEST — run separately, after the fact, on the HOST patient.** `eval_anatomies.py` with
`--real_patient_anatomy`, 98 episodes, on a checkpoint file. Nothing about it happens
during training. **This is the only instrument in the project that has demonstrated
resolution, and it is what checkpoint selection must use.**

## 10.2 Why validation cannot be used for checkpoint selection — measured

Run `2026-08-28_075919_rcca_topbrain_v1`:

| checkpoint | VALIDATION (4 held-out TopBrain) | TEST (host patient) |
|---|---|---|
| `checkpoint256370` | **99.0%** | **44.9%** |
| `checkpoint505230` | **99.0%** | **64.3%** |

**Identical on validation, 19.4 points apart on test.** The validation eval saturates:
97/98 twice gives a Wilson interval of [94.4, 99.8], so it cannot separate 99.0% from
99.5%, let alone rank two checkpoints. Stopping that run on "validation is flat" was a
mistake — the policy was still improving substantially and the instrument could not see it.

Root cause: the held-out TopBrain anatomies are too easy. Explore success reaches 98%+
within ~500 episodes and the last 3,000+ episodes of a run buy nothing the validation eval
can register.

## 10.3 Running the TEST evaluation

```bash
EXTRA_FLAGS="--real_patient_anatomy --cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight 0.5" \
NAME=eval_host_ckXXXX CHANGE_EVERY=1 \
  bash launch_eval_anatomies.sh <ckpt-path-inside-container> 98
```

The three flags after `--real_patient_anatomy` are **not** in `launch_eval_anatomies.sh`'s
inherited defaults and are required for the v1b/v1bp/TopBrain P2-teacher family. Omitting
them changes the observation width and the strict checkpoint load fails loudly — which is
the intended behaviour, not a bug.

`--real_patient_anatomy` pins the ORIGINAL segmented collision surface. Do not try to
reach the host by constructing `DualDeviceNav` directly: that also moves the wire's start
to the femoral entry and swaps the target sampler, giving `path_len` 594-752 mm against
the usual 74-269 mm. It is a different task. See §5 traps.

**Output caution.** `<ckpt-dir>/eval_anatomies_<ckpt>/episodes.csv` and
`anatomy_success.csv` are OVERWRITTEN by every run in that directory. The per-run
authoritative records are the timestamped `episodes_official_<ts>.jsonl` and the
`logs/<ts>/` and `snapshots/<ts>/` subdirectories. An agent misidentified one of these
files as the host run during this session and drew two wrong conclusions from it.

## 10.4 TEST baselines on the host — the ledger to beat

| model | host | CCA / ICA-mid / siphon |
|---|---|---|
| `ckpt2002292` (v1bp, procedural-trained) | **75.5%** | 100 / 90.2 / 33.3 |
| `ckpt514264` (v1bp) | 72.4% | — / — / **70.0** |
| `ck505230` (TopBrain-trained) | 64.3% | 100 / 63.4 / 33.3 |
| `ck3259127` (v1b) | 63.3% | — |
| `ck757854` (v1b) | 52.0% | — |
| `ck256370` (TopBrain-trained) | 44.9% | 92.6 / 36.6 / 13.3 |
| `checkpoint0` (scripted heuristic) | **25.5%** | — |

`ckpt514264` scoring 70.0% on the host siphon against `ckpt2002292`'s 33.3% is not a
typo: the earlier checkpoint is twice as good there. Continued training traded siphon
depth for shallower gains — the same direction as the LCCA result (13/98 -> 0/98).

## 10.5 Checkpoints are now in git

Seven models force-added past the `saved/eve_paper/...` gitignore (commit `37d85f9`),
4.3 MB each, 28 MB total. The multi-GB `replay_incremental` buffers are NOT included and
are not needed to evaluate. Everything else required is already tracked: host anatomy
(`eve_bench/data/dualdevicenav`, 28 files), `topbrain_data` (425 files), both eval
launchers, `eval_anatomies.py`. A second machine needs only `git pull`.

## 10.6 CORRECTIONS to §9 and to earlier analysis

- **The recovery taxonomy has been over-read throughout this project.** Only **18.1%** of
  soft/hard "recoveries" genuinely clear a buckle (soft 9.1%, hard 28.9%); against the
  full stall ledger it is **6.3%**. `extract_stuck.py:91` closes a stall as resolved on
  `pass_eps = 1.0 mm` — a retraction plus one millimetre. Over half of soft events have
  **no buckle present at all**. Indicated repairs: raise `pass_eps` to ~15 mm, move the
  soft/hard boundary from 8 mm to ~20 mm.
- **`slack_gw` does NOT detect buckling in this build.** Against an 18,541-window null of
  clean-advance segments, clean-navigation slack rise (p50 3.97 mm) EXCEEDS stall-window
  slack rise (p50 2.80 mm), lift ~1.0. Only the >=10 mm tail is usable. **`fold` is the
  instrument that works** (32x lift at fold >= 4). `cath_slack` is identically +0.0 in
  every July-2026 log set — emitted but never populated in that build.
- **The per-anatomy success spread across the 22 is NOT statistically real** for the
  policy: Pearson X2 = 28.47, df = 21, permutation p = 0.105. The 0-100% range is
  consistent with binomial noise on n = 1-10 per anatomy. The matched HEURISTIC does show
  heterogeneity (p = 0.041) — a real anatomy-difficulty axis exists and the trained policy
  does not track it. Most of the apparent spread is target-depth sampling.
- **The H0 baseline has ~18 points of run-to-run variance.** Two runs with identical
  launcher, holdout and `EVAL_SEEDS` gave H0 = 73.5% and 55.1%. Under residual-on-heuristic
  the untrained policy is not exactly zero, so network init shifts `a_heur + a_policy`.
  "Beat H0" is a weaker criterion than it looks.
- **`extract_stuck.py` uses `abs(cmd_action[0])`** in the stall predicate, so retraction
  commands count as pushing. A hand-rolled version using the signed value finds ~20x fewer
  stalls. Always run the shipped script.

## 10.7 Machine resources — measured, and what actually binds

| | measured |
|---|---|
| logical CPUs | 12 |
| GPU | RTX 4500 Ada, 24,570 MiB, **1,496 MiB used, 2% util** |
| training container | **360% CPU** (~3.6 cores), **20.4 / 23.47 GiB RAM** |
| throughput | **16.07 env-steps/s** total, 1.00 per worker |
| SOFA cost per step | 0.503 s median |

**GPU is never the constraint** in this configuration: the network is a 256x256 MLP
(~98k params), observations are 125-dim vectors under `TrackingOnly` (no images
rendered), and the 2M replay buffer lives in host shared memory, not on the device. It
would only become a constraint if observations went image-based (`MonoPlaneStatic` +
a CNN encoder) or the buffer moved to GPU.

**Host RAM binds at 87%.** That is why a concurrent eval risks OOM-killing the training
run rather than merely slowing it — the v1bp header records "v1b's day-4 host-mem OOM
horizon" as a known hazard. A concurrent eval at `N_WORKER=4` instead of 16 is the safe
option and uses cores that are otherwise idle.

**Scaling headroom is ~1.5x, not 2x.** At 0.503 s of CPU per step and 12 cores the
machine's ceiling is ~24 env-steps/s; we run at 16. Going 16 -> 24 workers (~30 GB RAM)
should approach it. Going to 32 workers (~40 GB) oversubscribes 12 cores 2.67x and yields
the same ~24 steps/s with more memory and more scheduler thrash. Past that it is cores,
not RAM. Step 16 -> 24 and watch for queue stalls — `EVE_RL_MODEL_QUEUE_TIMEOUT_S`,
`TRAINER_RESULT_TIMEOUT_S` and `WATCHDOG_STALL_S` exist because that has bitten before.

## 10.8 Current run

`2026-08-29_235446_rcca_topbrain_v1`, relaunched from step 0 with config identical to the
stopped run (17 train / 4 holdout / 133 mm). Validation history:

| eval | steps | success | steps/ep | speed |
|---|---|---|---|---|
| 1 (H0) | 0 | 55.1% | 502 | 4.24 |
| 2 | 253,934 | 100.0% | 63 | 25.5 |
| 3 | 506,469 | 98.0% | 96 | 21.7 |
| 4 | 756,995 | 96.9% | 88 | 21.7 |

Peaked at eval 2 and drifting down since — but per §10.2 that is inside the validation
instrument's noise and means nothing without host TEST evals on
`checkpoint253934` / `checkpoint506469` / `checkpoint756995`.

---

# 11. MEASURING GEOMETRY — the three traps, and the procedure that avoids them

Every one of these has produced a confident wrong answer in this project, twice each.
None of them announces itself as an error; each returns plausible-looking numbers.
This section is the procedure, not a warning.

## 11.1 Trap 1 — the COORDINATE FRAME. Signature: hundreds of millimetres

`load_branches()` returns centerlines in the **branch frame**. The host's
`vessel_architecture_collision.obj` on disk is in the **mesh frame**. `FromMesh` applies
`rotation_yzx_deg=[90, -90, 0]` and writes a temp copy already rotated into the branch
frame. Comparing the raw file against branch-frame centerlines is comparing two different
coordinate systems.

```python
# WRONG -- mesh frame vs branch frame
mesh = pv.read("eve_bench/data/dualdevicenav/vessel_architecture_collision.obj")

# RIGHT -- FromMesh output, already rotated into the branch frame
from eve_bench.dualdevicenav import DualDeviceNav
mesh_path = DualDeviceNav().vessel_tree.mesh_path
```

**The cohort `.obj` files ARE frame-consistent with their own centerlines** — verified for
all 25 right-ICA and all 24 mirrored-left anatomies. Only the HOST needs the accessor.

DETECTION. A frame mismatch reads in the **hundreds of millimetres**. The original
occurrence reported host clearance of 494.75 mm on a ~2 mm vessel. If a clearance number
is not within a small multiple of the lumen radius, stop and check the frame before
believing anything downstream.

CONFIRMATION that it is the frame and nothing else: every frame-INVARIANT quantity is
unchanged across the fix (cell count, edge median/p90/max) while every frame-DEPENDENT one
moves. That test is cheap and it is conclusive.

## 11.2 Trap 2 — the SIGN of signed distance. Signature: everything inside-out

`vtkImplicitPolyDataDistance` derives its sign from surface normals. **All 49 cohort meshes
are non-watertight** (2-5 open boundary edges, 1-3 non-manifold edges), so the sign is not
something to assume in either direction. On this build's meshes the filter returns
**positive INSIDE**, which is the opposite of the usual convention.

Getting it backwards reports every centerline point as outside its own lumen — 219 of 219,
232 of 232 — which reads as catastrophic mesh corruption and is entirely an artifact.

**PROCEDURE: always run a known-good control in the same script.** When the 24 mirrored-left
anatomies were checked, the sign came out inverted and the natural conclusion was that the
mirror had corrupted them. Running the already-validated RIGHT anatomies through the
identical code showed them reading identically — which located the fault in the convention,
not the data. One extra loop, and it is the difference between a correct result and a
fabricated crisis.

Good controls: any right-ICA cohort anatomy (known enclosed), or the host visual mesh.
A centerline point is inside its own lumen by construction, so a control that reports
otherwise is measuring the code.

## 11.3 Trap 3 — the ESTIMATOR. Signature: plausible but 5-25x optimistic

Ranked, best first:

1. **`vtkImplicitPolyDataDistance`, signed, densified to <=0.25 mm along the centerline.**
   Use this. Densification matters: short pinches are stepped over at native station
   spacing, which is how `mr_025`'s 1.25 mm sub-contact run was nearly missed.
2. **Exact planar cross-sections** (`vtkCutter` + connectivity, keeping only the loop that
   encircles the station). Use when the shape of the lumen matters, not just the minimum.
3. **Dense triangle-SURFACE sampling** (>=20 barycentric points per triangle). Acceptable
   for medians, **systematically optimistic at the minimum by 5-25x**: host 0.31 mm sampled
   against 0.041 mm exact; `mr_024` 0.188 against 0.013. It missed `mr_014` entirely.
4. **Nearest VERTEX — never.** On ~6.5 mm triangles the vertices sit outside the facets.
   This reported 1% lumen erosion where the true figure was ~45%, and that single bad
   number sent three separate wrong theories into circulation before it was caught.

## 11.4 The standing recipe

```python
import vtk, numpy as np, pyvista as pv
f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(mesh)
sd = np.array([f.EvaluateFunction(p) for p in densified_centerline])   # <=0.25 mm spacing
# sign: POSITIVE = inside on this build's meshes. VERIFY with a control every time.
```

Then, before reporting anything:

1. Is the magnitude within a small multiple of the lumen radius? If not -> frame.
2. Does a known-good control give the same sign? If not -> normals.
3. Was the centerline densified? If not -> short defects are invisible.
4. Are frame-invariant quantities (cells, edge lengths) unchanged from the raw file? If
   not, something other than a frame transform happened.

## 11.5 One further caution — DECLARED RADIUS is not lumen

`branch.radii` is what obs 47/48/49 and `get_local_tolerance()` consume, and it does not
agree with the mesh. Median (exact clearance - declared radius):

| | |
|---|---|
| host | **+0.38 mm** (accurate, slightly pessimistic) |
| every one of the 22 cohort anatomies | **-0.75 mm** (optimistic by ~1.1 mm relative to host) |

Cause: the graft pipeline's smoothstep overwrote the host's own mid-ICA narrowing (host
r 1.27-1.60 mm) with a ramp up to 2.25-2.90 mm, compounding with the mesher's uniform
-0.31 mm inward offset. So on the cohort the declared radius **overstates true bore by
roughly a millimetre**, and any analysis keyed to `local_r` is optimistic there while being
accurate on the host. Measure the mesh when the question is "will the device fit".

## 11.6 Result of running this on the 24 mirrored-LEFT anatomies (2026-08-30)

Checked because the mirror (`sp * [-1, 1, 1]` in RAS) is exactly the kind of operation that
can be applied to the centerlines and not the mesh, or vice versa. Script:
`monitoring/check_left_frames.py`; control: `monitoring/check_frames_control.py`.

**FRAME: clean, 0 of 24 mismatched.** Median clearance 1.440-1.941 mm (median 1.759),
against 1.51-1.96 mm for the already-validated 25 right-ICA anatomies. Indistinguishable,
and both at lumen scale rather than the hundreds of mm a frame error produces. The mirror
was applied consistently to mesh and centerlines.

**SIGN: the first pass was wrong and the control caught it.** The initial run reported
every centerline point of every left anatomy as outside its own wall (219/219, 232/232).
Running the known-good RIGHT anatomies through identical code produced identical readings,
locating the fault in the sign convention rather than the data. This is 11.2 in practice.

**ENCLOSURE: only 4 of 24 have any point outside the wall, and only one matters.**

| anatomy | points outside | worst |
|---|---|---|
| `topcow_mr_003_L` | **6** | 1.95 mm |
| `topcow_mr_002_L` | 2 | 1.20 mm |
| `mr_010_L`, `mr_016_L`, `mr_023_L` | 1 each | < 0.2 mm (grazing) |

The other 19 are entirely enclosed. `mr_003_L`'s 6 points at 211-216 mm independently
reproduce what `TOPBRAIN_PIPELINE.md` reports for it (6 consecutive points up to 2.12 mm),
which is why upstream excludes it. **The upstream exclusion of `mr_003_L` is correct and
sufficient — nothing else in the left set needs dropping on frame or enclosure grounds.**

---

# 12. NEW ANATOMY SET — intake checklist

Run this before any new anatomy set is trained on or reported against. Every item is here
because it was got wrong at least once, and none of them announced itself: each returned a
plausible number that was believed for a while. The check is cheap; the re-analysis is not.

## 12.1 Geometry validity — before any conclusion at all

| # | check | guards against | pass criterion |
|---|---|---|---|
| 1 | **Frame consistency** — clearance from centerline to its own mesh | comparing mesh-frame `.obj` against branch-frame centerlines. Reported host clearance of **494.75 mm** on a ~2 mm vessel | magnitude within a small multiple of the lumen radius. See 11.1 |
| 2 | **Signed-distance sign**, with a known-good control in the same script | inverted normals on non-watertight meshes. Reported **every** centerline point of **every** left anatomy as outside its own wall — pure artifact | control reads the same sign. See 11.2 |
| 3 | **Estimator** — exact `vtkImplicitPolyDataDistance`, densified to <=0.25 mm | nearest-vertex gave **1% erosion where the truth was ~45%**; dense sampling is **5-25x optimistic at the minimum** and missed `mr_014` entirely | exact only. See 11.3 |
| 4 | **Enclosure** — count centerline points with negative signed distance | `mr_015` (23 points, 7.19 mm outside), `mr_013`, `mr_014`, `mr_003_L` | 0 points outside, or a documented exclusion |
| 5 | **Watertightness** — open boundary and non-manifold edges | silently invalidates the sign in check 2 | record it; all 49 current meshes have 2-5 open edges, so the sign must always be controlled |

## 12.2 Navigability — is the device physically able to get there

| # | check | guards against | pass criterion |
|---|---|---|---|
| 6 | **Mid-vessel vs terminal blockages**, separately | `mr_025`'s pinch at s=166.8-168.3 mm is **95 mm proximal to its terminus**, so no distal trim removes it — 5 of its 9 targets were unreachable and scored as policy failures | mid-vessel blockage = exclude; terminal-only = trimmable |
| 7 | **Reachability ceiling** — fraction of the admissible target pool distal to any blockage | quoting a success rate against a denominator containing impossible targets | report the ceiling alongside the rate. Current cohort: 97.5-100% |
| 8 | **Inter-vessel fusion** — centre distance minus both radii, all nearby pairs | 9 of 25 grafted siphons initially interpenetrated a neighbour; 4 needed RVA repair | no negative clearance anywhere |
| 9 | **SOFA load / reset / step** on every anatomy | a mesh that loads fine and diverges in simulation | all load and step without divergence |

## 12.3 Provenance — what is actually varying

| # | check | guards against | pass criterion |
|---|---|---|---|
| 10 | **How many centerlines actually differ across the set** | assuming N anatomies means N independent samples. 15 of 16 centerlines are byte-identical across all 25 **and to the host** — only the RCCA varies, and only distal to s=133.6 mm | know which branch varies and from where |
| 11 | **Where the seam is**, measured not nominal | the nominal cut is 130 mm; the measured first >1 mm departure is **133.6 mm**, and the declared radii diverge **31 mm earlier at 102.5 mm** | derive both from the data |
| 12 | **Shared-course fraction of the target pool** | at the default 40 mm, **47-51% of targets** sit on geometry identical across every patient — a policy scores 100% there and it carries zero generalization signal | set `target_min_arclength` past the measured seam for any generalization claim |
| 13 | **Handedness**, if donors come from the contralateral side | a proper rotation preserves handedness; only reflection changes it. Integrated torsion: right +1.95 rad, left -2.00, left mirrored +2.00 | mirror (`sp * [-1,1,1]` in RAS), do not rotate |
| 14 | **Declared radius vs measured bore** | declared radius **overstates true bore by ~1.1 mm** on the cohort while being accurate on the host, so obs 47/48/49 are optimistic there | report median (clearance - declared r) per anatomy |

## 12.4 Experimental hygiene — before training on it

| # | check | guards against | pass criterion |
|---|---|---|---|
| 15 | **Split leakage** — both ICAs of one patient on the same side of the split | the left and right siphon of one person are not independent samples | patient-level split, never anatomy-level |
| 16 | **Do not select the held-out set for cleanliness** | choosing 4 defect-free anatomies produced **91.8%**, against **75.0%** on all 22 and **55.6%** on grafted targets | stratify the holdout across the observed difficulty range |
| 17 | **H0 baseline on the new set** — `checkpoint0`, the scripted follower | not knowing whether the set is harder or easier than the host. Host H0 = **25.5%**; TopBrain H0 = 62-73% | measure it, and expect ~18 points of run-to-run variance from network init |
| 18 | **Per-anatomy heterogeneity test before any per-anatomy story** | a 0-100% spread across 22 anatomies at n~5 each is **consistent with binomial noise** (perm p = 0.105) | test it; if not rejected, do not explain the spread |
| 19 | **Depth-stratified target sampling** | depth and anatomy are entangled — the four 100% anatomies all drew shallow targets | stratify, or depth confounds every per-anatomy claim |

## 12.5 The one-line version

Measure the mesh, not the declared radius; use exact signed distance with a control;
separate mid-vessel defects from terminal artifacts; find out how much of the set is
actually varying before calling it N anatomies; and never pick the holdout for being clean.
