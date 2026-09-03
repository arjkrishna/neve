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
accumulated twist, ground-truth branch identity) -- NOTE the nodal-FORCE dims are identically zero in every buffer ever collected; only velocity and the contact proxy carry signal (see 14.3).

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
| `--relax_failure_truncations` | **fold-stall and off-path** truncation OFF so off-path excursions stay recoverable. MaxSteps / VesselEnd / SimError still end episodes (see 14.2) |
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
| 13b | **Before mirroring anything, run the same-patient control** | assuming mirror symmetry without testing it. Siphons: 19/25 patients show opposite sign between their own two sides, so the mirror is right. Carotid FORKS: only 3/8, below chance -- bifurcation handedness is patient-specific, and mirroring the lower donor would rewrite the trained CCA->ICA route to match neither side. See 12.6 | mirror only where the same-patient control supports it |
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

---

# 13. OUTSTANDING — host TEST queue for run `2026-08-29_235446_rcca_topbrain_v1`

This is the next action. Nothing in the run's own validation can settle it.

## 13.1 Validation history — and why it decides nothing

| eval | explore steps | success | steps/ep | speed mm/s |
|---|---|---|---|---|
| 1 | 0 (H0) | **55.1%** | 502 | 4.24 |
| 2 | 253,934 | 100.0% | 63 | 25.5 |
| 3 | 506,469 | 98.0% | 96 | 21.7 |
| 4 | 756,995 | 96.9% | 88 | 21.7 |
| 5 | 1,010,053 | **95.9%** | 100 | 20.7 |
| 6 | 1,251,464 | 100.0% | 73 | 22.3 |
| 7 | 1,507,175 | 99.0% | 82 | 22.5 |

Six evals spanning 254k to 1.5 M steps all sit in a **4-point band, 95.9-100%**. Over a
comparable span the PREVIOUS run's host performance moved **19.4 points**. The validation
eval is not ranking anything — see 10.2. Do not read the 100 -> 95.9 -> 100 wander as
signal; 94/98 is Wilson [89.7, 98.2] and 98/98 is [96.2, 100].

Note also that **`best_checkpoint.everl` is selected by this metric** and therefore is not
the best model. Ignore it and select on the TEST results below.

## 13.2 The queue, in priority order

All eight checkpoints are committed (`5ebea7d`), so a second machine needs only `git pull`.

| priority | checkpoint | why |
|---|---|---|
| 1 | `checkpoint1507175` | deepest ever trained -- 3x further than anything host-tested |
| 2 | `checkpoint756995` | mid-point; with 1 and 3 it gives the shape of the curve |
| 3 | `checkpoint253934` | the earliest post-heatup checkpoint; pairs with the previous run's `ck256370` at 44.9% |
| 4 | `checkpoint1010053` / `checkpoint1251464` | fill in only if 1-3 show a trend worth resolving |
| 5 | `checkpoint0` | this run's H0 on the HOST. Expected ~25.5%, and confirming it validates the whole comparison |

## 13.3 Command

```bash
EXTRA_FLAGS="--real_patient_anatomy --cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight 0.5" \
NAME=eval_host_ckXXXX CHANGE_EVERY=1 \
  bash launch_eval_anatomies.sh <ckpt-path-inside-container> 98
```

98 episodes, ~35 min each. The three flags after `--real_patient_anatomy` are not in the
launcher defaults and are required -- see 10.3. Read results from the timestamped
`episodes_official_<ts>.jsonl`, never from `episodes.csv`, which is overwritten per run.

## 13.4 What each outcome would mean

Baselines: `ckpt2002292` (v1bp, procedural-trained) **75.5%**; previous TopBrain run's
`ck505230` **64.3%** at 505k and still climbing when stopped; scripted heuristic **25.5%**.

- **> 75.5%** — the TopBrain-trained policy overtakes the procedurally-trained one on
  anatomy neither trained on. That is the headline result this line of work has been
  aiming at, and it would justify the 49-anatomy and synthetic-variant follow-ups.
- **64-75%** — still improving past 505k but not yet past the procedural teacher. Tells us
  where the curve is heading and whether more steps are worth it.
- **~64%, flat against `ck505230`** — the extra 1 M steps bought nothing on the host, and
  the run should be stopped. This is the outcome that would retire the "it was still
  climbing" argument for the previous run's premature stop.
- **< 64%** — over-training, and the earlier checkpoint is the one to keep.

Also worth recording per checkpoint: the CCA / ICA-mid / siphon split. The previous
TopBrain run lost entirely in ICA-mid (90.2% -> 63.4% against the v1bp teacher) because it
trained only on targets past 133 mm, while host ICA-mid targets span s_RCCA 114-176 mm.
If that gap persists at 1.5 M steps it is structural to the 133 mm choice, not a matter of
training longer.

## 12.6 Correction — the mirror rule applies to siphons, not to bifurcations

Row 13 as originally written ("mirror, do not rotate") generalised a siphon result to all
contralateral donors. That is wrong, and it was caught by the other machine re-running the
control rather than inheriting the conclusion.

**The observation is correct**: in the three-source set only the siphon is mirrored, and the
shipped composites are 109 left-donor / 106 right-donor. Fork handedness genuinely differs
between the halves -- left donors 77/109 (71%) positive, median +0.049; right donors 16/106
(15%), median -0.096.

**The inference from it was wrong.** The mirror argument requires that left and right are
mirror images of each other. Same-patient control, on the 8 patients contributing both
carotids:

| same patient, opposite handedness | |
|---|---|
| siphons (mirror applied, and justified) | **19 / 25 = 76%** |
| carotid forks | **3 / 8 = 37.5%** |

Mirror symmetry predicts ~100%; chance is ~50%. Siphons clear it, forks fall below it. A
siphon is a long coiled curve with consistent handedness; a carotid bifurcation is short and
variable, and its handedness is largely patient-specific. Note n = 8 -- the control rejects
mirror symmetry decisively but establishes little beyond that.

Beware a trap in the supporting numbers: native torsion DOES show a population-level
difference (left 14/24 positive against right 7/24) while per-patient signs agree as often as
they disagree. Citing only the population figure looks like evidence of handedness. The
same-patient control is what separates the two.

**And for this task the question is moot.** The route is CCA -> ICA; the ECA is never a target
(`CenterlineRandom(branches=[RCCA])`). Its take-off side is variation in where the DISTRACTOR
sits, not an error in the path being learned. Mirroring the lower donors would reduce
decoy-side diversity, rewrite the trained CCA->ICA geometry of half the set -- the donor
supplies the CCA and cervical ICA too, not just the fork -- and, because the sides are not
mirror pairs, yield a fork matching neither the left nor the right patient.

**It becomes a real defect only if** a left-sided host arch (LCCA route) is added and donors
must match that side, or if a side-specific anatomical claim is made. Neither applies to a
right-sided CCA -> ICA task.

---

# 14. OPERATIONS AND FINDINGS NOT PREVIOUSLY RECORDED (June – September 2026)

Assembled 2026-09-01 by sweeping the session transcript, the ~30 project documents and the
`monitoring/` tree against sections 0-13. Everything here was done, measured, and either
validated, ruled out, or left open — and none of it was in this file. Status tags:
**VALIDATED** / **RULED OUT** / **OPEN** / **BUILT, NEVER RUN**. Sources are the doc or
transcript line range; scripts are named where one exists.

## 14.1 Recovery metrics — four instruments, and what each measured

**Four non-interchangeable metric families exist.** Conflating them produced wrong readings
more than once.

| family | what it counts | where |
|---|---|---|
| event taxonomy | grind / soft / hard / unrecovered by **actually withdrawn** guidewire length, per stall event | `monitoring/extract_stuck.py` |
| intent-based | % of stalls with negative commanded intent; post-stall 20-step intent in mm/s | v2 investigation |
| full-cycle prevalence | stuck → retract → re-advance cycles inside successes, per 400-episode window | `saved/v2_micro_investigation/` |
| eval probe | P(retract \| stalled) on the deterministic policy at eval time | `monitoring/probe_policy_v3.py` |

**The three detector configs, verbatim.** §5.9 says thresholds move rates ~20 pp but never
records them. The shipped extractor scores every episode under all three simultaneously:

```
canon:  stall_eps=0.3 push_min=2.0 stuck_steps=12 retract_min=1.0 soft_max=8.0 pass_eps=1.0
sens:   stall_eps=0.3 push_min=1.0 stuck_steps=8  retract_min=0.5 soft_max=8.0 pass_eps=1.0
strict: stall_eps=0.5 push_min=4.0 stuck_steps=16 retract_min=1.0 soft_max=8.0 pass_eps=1.0
```

A stall opens when `proj_s < running_max + stall_eps` AND `abs(cmd_action[gw_trans]) >
push_min` for `stuck_steps` consecutive steps (counter decays -2 per non-stalled step);
closes when `proj_s > onset + pass_eps`. Recovery-success ranges 42% (strict) to 65% (sens)
on v1b; **every ordering between runs survives all three.** VALIDATED. Transcript 8394-8477.

**Retraction depth is a continuum massed at zero — soft/hard/grind is a cut, not a mode.**
Histogram of escaped-event retraction depth:

| run | <0.5 mm | 0.5-1 | 1-2 | 2-4 | 4-8 | 8-16 | >16 |
|---|---|---|---|---|---|---|---|
| v1b | 32% | 5% | 9% | 17% | 17% | 12% | 8% |
| v1bp | 25% | 5% | 9% | 16% | 19% | 15% | 10% |

No bimodality. The 1 mm and 8 mm boundaries slice a smooth curve. This is why the July "soft
2-6%" and the canonical "22.6%" are both "right" — and why no design should treat soft and
grind as distinct behaviours. The July one-off detector was never preserved, so that audit
is unreproducible. VALIDATED. Transcript 8501-8522.

**Recovery type proxies stall SEVERITY, not skill.** Grind-only episodes convert to success
at **99% / 95%** (v1b / v1bp) against soft's **95% / 76%**, and the raw crunch signature
carries **4-8x more steps in failures than successes** in every era. This is why the v1c
crunchpass lane had to be success-conditioned, and why it was never launched. OPEN — the
confound must be broken within matched depth strata first. `RL_IMPROV_18_P2_DESIGN.md` §5.1.

**The T1-T7 cross-run ledger (v1b vs v1bp, 22,731 explore episodes).** Four rows not
elsewhere in this file:

- **T4, depth-stratified:** stalled-episode share is **5-7% at the CCA vs 90-97% at the
  siphon.** At the siphon v1bp escapes **68%** of stalls yet only **14%** of episodes
  succeed. **The siphon is a stall-DENSITY problem, not an escape-skill problem** — escape
  one jam, meet the next.
- **T5:** recovered-episode conversion **95.2% (v1b) vs 77.0% (v1bp)**; mean catheter-lead
  fraction 0.34 -> 0.04; v1bp jams *more* (stalled-ep 51.0 -> 57.9%).
- **T6:** episode-success <-> recovery-success correlation is **r = -0.82 (v1b)** but
  **-0.21 (v1bp)** — the reward pair *decoupled* them.
- **Recovered-but-failed episodes die on the clock:** of v1bp's 246, **91% hit the 600-step
  cap**, median **61 mm short**, last recovery at 97% of the episode. The conversion gap
  holds in every depth band and is largely **transient** (v1bp bins 3/5 convert at 56%/54%,
  bins 6/7 at 94%/95%). The durable difference: v1bp never learned stall *avoidance*
  (stalled-ep flat 53-57% vs v1b's 64 -> 42%).

Refines §3.2's "reward pair is a null" to: **the two runs reached the same ~61% summit by
opposite routes — avoid-and-run-clean vs stall-and-fight-through.** VALIDATED.
Transcript 8110-8296, 8524-8551. `extract_stuck.py`, `analyze_stuck.py`, `verify_stuck.py`.

**The v2-era micro-recovery finding.** 53% of explore successes contained a full
stall -> retract -> resume cycle, but prevalence **faded 86% -> 58% -> 28%** across
training thirds while post-stall push intent hardened +1.8 -> +7.2 mm/s, and **failures
retract-and-resume MORE than successes (3.8 vs 1.4 cycles/ep).** Micro-recoveries were
distilled from exploration noise, not learned. VALIDATED. Transcript 4034-4041.

**Recovery between TopBrain checkpoints — checked, NEGATIVE.** Whether better recovery
explained the host 44.9 -> 64.3% between `ck256370` and `ck505230`: heatup stalls in 52%
of episodes at 5.49/1k; by bucket 2 that is 7.2% at 0.98/1k — a 5.6x collapse inside 500
episodes — and across the two checkpoints stalls drift 0.63 -> 0.50/1k with resolution flat
within noise. **Stall avoidance was learned before the first checkpoint was written;
whatever bought the host improvement is invisible to every recovery instrument.**
RULED OUT. Transcript 11415-11424.

**The buckle-clearing adjudicator — the definition, since §10.6 quotes only its headline.**
CLEARING := `fold_load >= 4` (buckle present; calibrated against an **18,541-window null**
of clean-advance segments, where P(null >= 4) = 1.21% vs P(event >= 4) = 39.2%, lift 32x)
AND `fold_close == 0` (reduced) AND `adv25 >= 15 mm` (passed; the 15 mm cut sits inside an
**11-mm-wide empty gap** in the data, so any threshold in 14-25 mm gives identical labels)
AND no re-stall within 20 steps at |dp0| <= 2 mm. Each stall window is split at the
guidewire peak into a LOADING phase (`first..peak`) and RELEASE phase (`peak..close`), and
"buckle present" is measured on loading only. Negative control: `checkpoint0` yields zero
clearing events. `slack_gw` was REJECTED as the buckle instrument because clean-advance
slack rise (p50 3.97 mm) exceeds stall-window rise (p50 2.80 mm). VALIDATED.
`monitoring/buckle_clear_final_v1.py` (docstring carries the definition);
`buckle_clear_adjudicate_v4.py` is buckle-load-normalised and runs the home/foreign cells.

**Functional-success reclassification.** Strict `TargetReached` mis-scores catheter-overshoot
near-successes (wire bunched inside the catheter at the target). Terminal
`cum_reward in [-1, +1]` is the clean discriminator (genuine failures land <= -3). Strict
32.7 / 31.6 / 36.1% -> functional **42.9 / 44.9 / 50.5%**; functional improved +7.6 pp over
750k steps vs +3 pp strict. Also showed an "eval regression" was classifier noise — the same
physics moved between truncation buckets. VALIDATED. `RL_IMPROV_10_CHANGES.md` §17.
Matters because strict success is the only metric quoted anywhere, and it tracks *which
truncation fires*.

## 14.2 Planned-path classifiers — evaluation history and measured accuracy

**A uniform cross-track tolerance cannot work — measured.** Natural in-lumen drift in the
aortic trunk is **~9.5 mm**; the bif2 wedge is **13 mm**; the RCCA-target wedge **49 mm**.
6 mm classified normal trunk navigation as off-path; 10 mm flickered on SOFA jitter. Both
produced 50/50 episodes ending at **-53 reward** and an infinite ~10-step retract/re-advance
thrash. RULED OUT. `RL_IMPROV_8_CHANGES.md` §4-5, §12.

**The surviving design: radius-aware tolerance + state-machine `current_branch`.**
`tol = max(2.0, 1.5 * clip(local_radius, 2, 12))` — trunk r=12 -> 18 mm (9.5 drift safe),
bif2 r=7 -> 10 mm (13 wedge fails), bridge r=4 -> 6 mm (49 wedge fails). On 50 heuristic
episodes: WEDGE600 35 -> 17, but successes 4 -> 0. **VALIDATED for classification, OPEN for
task success** — attributed to flicker where `tol ~ drift`; the proposed asymmetric
hysteresis (flip off at `ct > tol`, back on only at `ct < 0.7*tol`) was never built.
`RL_IMPROV_8_CHANGES.md` §14-15; `eve/eve/util/pathcontext.py`.

**Five classifier defects, each of which silently disabled or corrupted off-path detection.**
Every one presented as a policy failure — episodes burning 600 steps at a wall.

| defect | mechanism | fix | status |
|---|---|---|---|
| Bug-A | state machine commits by arclength without checking the tip is on the *path-used segment*; a wedge on `(0).mrk` off-path indices 0-20 read as on-path | on-path-mask check re-layered on the state flag | VALIDATED |
| junction hysteresis | forward-commit required `proj_s` to leap 20 mm in one step against ~0.5 mm/step advance; branch transitions fell **90%** (LCCA 7 vs 45) | one-sided commit, 10 mm dead-band | VALIDATED |
| one-step lag | `update_branch_state()` ran AFTER reward/obs with the `is_on_correct_path` cache uninvalidated — the tax, off-branch counter and obs 19/25-28 all used the **previous** step's classification | `Env._on_intervention_stepped` hook; memo self-clears | VALIDATED |
| hairpin-blind projection | global nearest-segment argmin with no continuity; at the siphon adjacent limbs flip the projection and spike ds by the loop length | **+-30 mm window, 15 mm fallback**; bit-identical to full scan over 200 trials | VALIDATED |
| KD-tree flip-flop | `off_br` counter reset on every micro-flip (`4->8->10->10->1->3...`) so the grace threshold never fired; **242 of 539 off-branch steps had reward > -0.101** — progress out-paid the penalty | `_stable_on_branch` + 5-step `_pending_flip_count` | VALIDATED, entry/exit reported ~5 steps late |

Plus: the WBT label leaked under `--relax_failure_truncations` (counters kept climbing after
truncation stopped; three sites still labelled `wrong_branch_timeout`) — fixed, 0 leaked.
Transcript 70-71, 384, 2277-2334; `RL_IMPROV_7_CHANGES.md` §3, §7; `RL_IMPROV_8_CHANGES.md` §18-19.

**Consumers of the classifier disagree with each other — OPEN.** `is_on_correct_path()`
applies the on-path-mask override; `arclengthprogress.get_projection()` does not — same wire
state, two verdicts. The `+1` junction reward uses `_path_daughter_arclengths` (real forks)
while the heuristic's milestone is a junction with no off-path branch; the reward gate needs
`arc_past >= 10 mm`, the heuristic fires at `s >= jn_arc - 5` — a **10 mm dead zone**. And
**`in_wrong_branch` (obs 73) is an exact -1.0 duplicate of `on_path`** — zero extra
information. The `path_extension_set` unification was never built. `RL_IMPROV_8_CHANGES.md` §32-33.

**Degenerate entry-point features.** `_build_entry_points()` stored the raw bifurcation
coordinate, so `wrong_pt` and `corr_pt` were **literally identical at every sample** —
LocalGuidance features 8-13 ("avoid this way" vs "go this way") collapsed to one scalar at
every fork. Fixed by storing a point 15 mm into each branch. Code VALIDATED; the in-container
assertion (`|corr - wrong| >= 10 mm`) was never run. `RL_IMPROV_7_CHANGES.md` §4.

**The crunch classifier — the only planned-path classifier with a measured accuracy.** Five
path-geometry features (`gw_slack` 89, `cath_offset_z` 77, `gw_cath_gap` 78,
`local_radius` 93, `arc_past_daughter` 58) classify crunch-signature steps at **AUC
0.92-0.95** on v1b chunks; restricted to **deployable** features, **AUC 0.65-0.77** with 9
(machine-2: 0.788 with top-15). Three load-bearing refinements: (a) the raw signature marks
GRINDING — failures carry **4-8x** more crunch steps (eval3 era: 193/failure vs 41/success),
so the lane must be success-conditioned; (b) v1b's 81.6% crest *was* the crunch-passage era
(41 crunch steps/success, 50% of successes with >= 3-step passages, vs 7-14 elsewhere);
eval5's crest was a different mechanism (cleaner avoidance, 14%); (c) physics check: crunch
steps have **LOWER** contact than free steps — close catheter support prevents buckling.
Caveat: normalized `gw_cath_gap` rails in late eras — never use that dim alone. VALIDATED.
Transcript 6805-6817; `saved/p2a_deep_dive/CRUNCH_POOL_STRATEGY.md`.

**The classifier that shipped is NOT the AUC-0.94 one.** `launch_rcca_p2_teacher_v1c.sh` /
`v1d.sh` implement `is_crunchpass = crunch_sig(obs) AND episode.success` as a **hard
threshold rule on three obs dims (42/44/93)**: both device translations > 0.25 normalized
AND local radius <= 0.175. Six container tests pass; default-off. BUILT, NEVER RUN.
Anyone resuming must know the shipped membership function is a 3-dim box, not the 5-feature
classifier the AUC was measured on. Transcript 6888-6892.

**One-segment planned route collapses 14 of 51 guidance dims — cost measured at ~6 points.**
Starting inside a branch makes `FixedPathfinder` take its same-branch shortcut; feature 25
`is_in_trunk` pins at 1.0 and 26 `is_on_target_daughter` at 0.0 for the whole episode while
the privileged bit correctly says "in target daughter" — a contradiction never seen in
training. Nothing errors. The matched RCCA-internal control (69.4% vs 75.5%) priced it at
~6 points, far short of the ~69 needed to explain LCCA 0%. VALIDATED. `LCCA_TRANSFER_RESULT.md`.
**Reusable rule: any start-state or topology change must be checked for silently-constant
observation dimensions before its result is read.**

**Correction to §2.1.** The flag table says `--relax_failure_truncations` turns "vessel-end
truncation OFF". The code (`env5.py:1251`, `:1042`) disables **fold-stall and off-path**
truncation; **MaxSteps, VesselEnd and SimError still end episodes.**

## 14.3 Observation values used as force-field proxies

**Exact inventory.** Deployable, guidance block: `gw_slack` = feature **43** =
`clip((inserted_gw - proj_s)/50, 0, 1)` — wire stored in bowing, the single most informative
buckle scalar, mesh-relative by construction; `slip` = **44** = raw fold-detector input
`delta_gw - delta_s`; command-mask flags 45/46; radius now/ahead **47/48**; `clearance =
cross_track/tol/2` = **49**; gw-cath gap 32; log-depth 50. Privileged tail (24 dims,
`eve/eve/observation/meshinvariant.py`): 0/1 mean/max node velocity; **2/3 log1p mean/max
|node force|; 4 argmax-force position; 5/6 mean/max |position - free_position| (the
contact-impulse proxy)**; 7-10 sin/cos accumulated rotations (windup); 11-13 cath-gw tip
offset; 14-18 branch one-hot; 19 off-branch counter; 20 fold counter; 21 slip; 22
cross-track excess; **23 guidewire slack**.

**The SOFA force channel is DEAD.** Privileged dims **2/3/4 are identically zero across all
568,874 buffer states.** `MechanicalObject.force` / `dofs.force` is a per-solve scratch
buffer cleared before any post-step read; verified on a live-wire checkpoint (force = 0,
velocity alive, `externalForce` shape `(0,6)`). The "argmax-force position" feature has been
argmax-over-zeros noise for the entire program. **RULED OUT: post-step force reads can never
work.** The only route to true force is `GenericConstraintSolver.constraintForces` /
`computeConstraintForces=True` — a scene change, deferred as E6, never done. §1 describes
the tail as "nodal forces/velocities, contact-impulse proxy" without recording that the
force half is zeros. `RL_IMPROV_15_CHANGES.md` Part B §5/§9; transcript 4048, 4163, 5172.

**`|position - free_position|` was chosen over `force` deliberately.** `force.value`
includes internal elastic bending, and high bending is *required* to conform to the siphon —
penalising it inverts into "do not follow tortuosity". `|pos - free_pos|` is nonzero only
where the wall pushes back. It is the only force-field proxy actually wired into reward.
VALIDATED. `training _scripts/util/buckle_reward.py`.

**`gw_slack` is deployable AND is a numerical duplicate of privileged dim 23.** Both are
`(inserted_gw - proj_s)` from the same projection. Consequence: **the paper's "planned-path
<-> force correspondence" claim is an identity, not a correspondence** — its dominant
predictor is the same number on both sides. §6.12 lists the probe as "never built" without
saying the paper's asymmetry pillar rests on it or that the naive version is circular. The
specified fix (`SPIE_METHODS.md` §2.4, pending): predict contact-impulse / distal nodal
force from deployable features with **every deployable feature numerically identical to a
privileged dim excluded**, fit only on audited anatomies, with a shuffled-label control.
Flagged as likely to return a negative — run it early. OPEN. Transcript 9937-9967.

**Contact labels are scale-starved by four orders of magnitude.** Normalised contact std
~1e-3 -> `aux_coef * MSE` gradient ~**5e-8** — the aux term was silently ~0 for the whole
v2 run and was not logged. Aux contact R2 peaked **0.554** at u~131k then regressed to 0.486,
against a linear ceiling of 0.779. Fix specified as E2: repoint labels `2,3,5,6 -> 0,1,5,6`
(velocities are alive), loss-time EMA z-scoring behind `--aux_label_znorm`, log the loss.
BUILT (machine 2), never run on the P2 line — P2 sets `--aux_coef 0` because with a
privileged actor the labels are inputs. `RL_IMPROV_16_EXPERIMENTS.md` E2.

**Distillation-efficacy probe: the teacher can FEEL contact but does not ACT on it.**
(A) The aux head infers contact from deployable obs at **r ~ 0.75** against 0.002 for slack
alone — the trunk carries the knowledge. (B) Behaviourally: **P(retract) is FLAT across
contact quintiles**, and the policy's correlation with its own contact inference is
**POSITIVE — it pushes harder when it senses contact.** Diagnosed as credit assignment: the
critic prefers retract in the buckled tail (AWAC weight 1.042 vs 0.947) but with advantage
std 0.092 and lambda = 1.0 the weights span **[0.72, 1.25] ~ uniform BC**. VALIDATED as a
negative; the measured precondition for any P2b student claim. Transcript 4022-4049.

**Gradient-saliency probes — method and both results.** `|d mu / d obs|` over 1,500 buffer
states, all dims ranked, across snapshots (~10 min `docker exec`). v2 era: **`ep_step` is
the #1 saliency input for all four action means** plus aux and log_std; the four
`last_action` dims rank 2-5 (time + momentum = 17% of saliency). ~21 prunable dims: frame
t-1 body offsets are r = 0.99 duplicates; `in_wrong_branch` is `-on_path`; `d_rem_log`
ignored; `at_ostium` dead at source on procedural meshes; `curv_ahead` variance-crushed by
its /10 scaling. v1b (P2 residual, init / 288k / 543k): **ep-counters demoted to rank 21 of
125; privileged tail ranks 2-3; heuristic-intent dims fall 11 -> 54 -> 78** (the policy
weans off the script). **The residual design defused the ep_step time-hack** — a real
architectural result; dim 68 climbing back toward top-5 is the pre-registered early warning.
VALIDATED (first-order saliency, not causal ablation). `monitoring/probe_policy_v3.py`.

**Deployable features CAN recover the contact signal** — the crunch classifier above
(AUC 0.65-0.77 deployable-only) is the empirical basis for the P2b student's aux head and
the strongest evidence the privileged tail is inferable. VALIDATED.

**`obs47` is degenerate exactly where it matters.** `clip(stated_r, 2.0, 12.0)/12` reports
the floor for every station under 2.0 mm — constant through the siphon. On the host **81.7%**
of distal stations are clamped, 20 distinct values across ~105 stations, while true radius
swings 1.40-2.25 mm. On the cohort 12.6% median. Distinct from §12 row 14 (declared radius
overstates bore): this is a clamp degeneracy in the observation itself. The 2.0 mm floor is
frozen logic (feeds obs 47/48/49, the off-path classifier, `at_tree_end`).
`monitoring/obs47_degeneracy.py`. VALIDATED.

**Observation OOD-by-zone instrument exists with outputs on disk and no narrative result.**
`monitoring/obs_ood_arjun_*.py` -> `ood_out.json`, `ood_clean.json`, `ood2-4.json`: per-zone
(`Z1_shared_0_103`, `Z2_ramp_103_136`, `Z3_graft_136_end`) distributions of obs47, obs48,
`tol`, `d_rem_norm` with `frac_in / frac_above / frac_below / shift_iqr`. The only
measurement of whether the policy's *inputs* are in-distribution on the cohort. OPEN.

**The stuck-lane thresholds are defined on these proxies.** E3's third SumTree lane marks a
transition stuck if stored `gw_slack` (flat 89) > **0.174** or `contact_max` (flat 103) in
its top decile (~2.6e-3); `--stuck_fraction 0.15`; composition 30% clean / 15% stuck / 55%
general. Under `--heur_action_obs` the contact index shifts **103 -> 107**. Flag must be
immutable per slot so `update_priorities` cannot evict lane membership. BUILT, fully
tested (400/400 flags, round-trip, back-compat), NEVER RUN on the P2 line.
`RL_IMPROV_16_EXPERIMENTS.md` E3.

## 14.4 Snapshot / restore / stuck-pool machinery and curricula

**Escapability + restore-fidelity screener — BUILT, TESTED, NEVER RUN.** A wedged state in a
restore pool teaches the critic `V(stuck) = failure-and-nothing-after`, poisoning the skill
the curriculum builds. `training _scripts/util/escapability.py` (pure, 5 unit tests) +
`screen_stuck_pool.py` + `launch_screen_stuck.sh` (41 mounts verified). Verdict logic:
`restore_faithful` FIRST — restored insertion within 10 mm and slack within 8 mm of capture
(a sprung buckle fails); then `is_escapable` — a state a scripted pure retract
`[[-20,0],[-20,0]]` cannot move >= 2 mm in ~40 steps is mechanically wedged; a fold must
additionally drop slack >= 5 mm or end < 8 mm; off-path needs only retractability.
`screen_report.json` warns if > 50% of the pool fails restore. Read-only w.r.t.
`is_on_correct_path()`. `RL_IMPROV_15_CHANGES.md` Part F §4; transcript 1589-1626.

**Stuck-restore as designed FAILS — harvest crunch-ENTRY, not stuck states.** Machine 2:
*"by the time the wire is truly stuck it is knotted and coiled — you'd have to retract
hundreds of steps."* This inverts E4. Harvest the last moment before the knot. Companion
change never made: `CheckpointRestoreWrapper` restores on **every** reset — needs
`--restore_prob 0.3` so the run keeps learning full navigation. RULED OUT as designed.
§6 items 8/9 keep the lane alive without recording that its state source was proved wrong.
`RL_IMPROV_18_P2_DESIGN.md` §5.1; transcript 6797-6815.

**Stuck-pool harvest parameters and the mesh hazard.** `STUCK_CHECKPOINT_DIR` triggers a full
SOFA `save_checkpoint()` once per episode at `STUCK_FOLD_TRIGGER=10` or
`STUCK_OFF_BRANCH_TRIGGER=25`, capped 200/worker. **Hazard:** the wrapper picked from the
directory at random with no mesh matching, so under `--procedural_rcca` a mesh-B worker could
restore a mesh-A snapshot — wire teleported through a wall. Mitigated by fingerprint filter +
pinning; the mesh still regenerates every 10 episodes so checkpoints go stale within a
worker. Rule: stuck-pool restore is fixed-mesh-only unless fingerprinted. Transcript 1202-1475.

**Per-state difficulty dominated policy quality — the single-restore-state sweep.** All 98
eval seeds from one curated restore state (`pid19116`) with the eval-#1 policy (32.67% on
mixed states): **84/98 = 85.7%, +53 pp**, in 808 s vs ~5000 s, zero wrong-branch timeouts.
All 14 failures were the catheter-overshoot fold. VALIDATED. `RL_IMPROV_10_CHANGES.md` §16;
`training _scripts/eval_policy_from_state.py`, `--eval_only_checkpoint`.

**Target depth -> capability mapping, pre-TopBrain.** The RCCA centerline is 237.5 mm and
runs CCA + cervical ICA + petrous + cavernous + terminus. In the pid19116 sweep: proximal CCA
(z 416-510) **100% (45/45)**, cervical ICA (z 510-575) **82.5%**, siphon/terminus
(z 575-601) **46.2%**. All 14 catheter-overshoot failures at z >= 547. Mechanism: siphon 180
degree bends + aggressive `cath_trans` -> wire buckles inside the catheter instead of
emerging. VALIDATED — the mechanism behind §9.4's target-depth finding.

**Action-space curriculum — built, two gotchas.** `ActionCurriculumWrapper`: stage 1
(0-200k) `cath_trans = gw_trans * 0.8, cath_rot = 0`; stage 2 (200k-500k) catheter x 0.1;
stage 3 full 4D. **Gotcha 1:** the replay buffer stores the **original** pre-modification
actions — an action/reward mismatch. **Gotcha 2:** step counting is **per-worker**, so each
worker advances its own stage. Stage 2 also scales heuristic-mode demos, distorting the
seeding distribution. OPEN. `RL_TRAINING_LIFECYCLE.md`; `RL_IMPROV_8_CHANGES.md` §32.

**Checkpoint collection / curation pipeline — reusable, idle.** `collect_sofa_checkpoints.py`
(parallel harvest at 370-385 mm insertion), `select_sofa_checkpoints.py` (all success-ending
captures are gold; else `looks_clean AND cum_reward >= 0 AND wire_shape_score < threshold`,
ranked by `step_idx` ASC — shorter = less hidden kinking outside the 5-point tracking
window; k-means k=4 on target coords), `snapshot_restore_states.py`, `smoke_test_restore.py`
(the historical bug: restore fired *before* `start.reset()` and was a silent no-op). VALIDATED.

**Cache reward-version guard.** `experience_cache.py` stamps `meta_buckle_coef` from
`EVE_RL_BUCKLE_COEF` (exported before workers spawn); both load sites fail fast on mismatch;
upgraded to a 4-field compare when the reward pair landed. The obs-dim guard cannot catch
this class — same layout, different scoring. VALIDATED. Transcript 1584, 6916-6934.

**Sibling levers on machine-2's Tier-A commit `0cd073c`, all BUILT.** E1b
`--awac_adv_norm_tau 2.0`: `w = exp((adv/sigma)/tau)`, measured weight span p99/p1
**1.54 -> 10.2** — the direct fix for the [0.72, 1.25] collapse. E2 `--aux_label_znorm`.
E8 `monitoring/monitor_pass_v3a.sh` + `probe_policy_v3.py`. CSV columns added:
`awac_weight_p99p1` = 29, `aux_loss` = 30.

**`CLEAN_RAIL_FILTER` — armed, silent, aimed at the wrong failure.** `EVE_CLEAN_RAIL_MAX=0.15`
gates clean-lane admission: **0 rejections** across all v2 online successes and 0/480 seed
episodes. But its criterion is |a| > 0.95 bang-bang, while the actual failure was sub-rail
mean growth (0.4-0.7) — "0 rejections" is not evidence of health. Also measured:
`balanced_fraction 0.3` already yields ~67% effective clean-batch composition; 0.6 buys ~78%
at the cost of halving failure/recovery draws. Transcript 3553-3561, 4835-4846.

## 14.5 Reward-term audits — the mechanisms behind numbers §3 quotes

**The anti-buckle potential, and why it is safe.**
`phi = -(0.5 * clip(slack - 5, 0, 40)/40 + 0.5 * clip(contact, 0, 2)/2) in [-1, 0]`, added as
`coef * (phi_t - phi_{t-1})`, `coef = 0.5`. Three deliberate properties: (a) any closed loop in
(slack, contact) nets exactly zero — **not oscillation-farmable**; (b) the episode sum
telescopes to `phi_end - phi_start`, so form-then-recover is neutral while **starting
buckled and unbuckling is net positive** — what makes stuck-pool restores trainable; (c) caps
are on the **inputs, never the delta**. `_buckle_phi_prev = None` at every reset so a
restored-buckled state re-baselines. **Deliberately not gated on the classifier** — a gate
would make phi jump on flicker. Measured in the TopBrain run: the contact channel never
exceeded 0.02 mm against its 2 mm cap in ~130,000 steps — phi is a pure slack signal there,
capped at 0.25 reward units, ~4% of a 6.0 return. VALIDATED. `RL_IMPROV_15_CHANGES.md` Part D §1.

**Two reward-farm fixes that gate `relax_failure_truncations`.** (i) `ArcLengthProgress`
paid **2x forward, 1x backward** inside the target daughter — every oscillation banked
`+pf * ds`; under relax a wire merely dithering in the RCCA farmed ~3-4 return, equal to a
success. Fixed to flat 1x symmetric. (ii) `max_steps` carried **no penalty** — loitering to
the horizon kept all shaping. `MAX_STEPS_PENALTY = -3.0`, checked first so a pure timeout is
priced -3 not -5. VALIDATED. Part D §3-4.

**Off-path retract tax discount.** The uniform -0.007/step off-path tax also taxed backing
out of a wrong branch. Dropped to **-0.002** while off-path AND genuinely retracting,
double-gated: `_off_branch_steps >= 3` (flicker pays full tax) and **executed**
`delta_gw <= -0.1 mm` (a masked retract that moved nothing pays full tax). VALIDATED. Part D §5.

**The full reward-farm audit (all fixed in Gen-3/4).** In-daughter progress doubling;
timeout pays nothing while ~+7 is banked (enter-dither-coast is EV-optimal vs a +3 bonus
risking -5); overshooting the correct daughter trips `VesselEnd` -5 same as a wrong one
(*approach is paid, stopping is free, going deep is the only penalised action*); translation
limits were `[-10, +30]` so **neutral policy output = +10 mm/s forward on both devices** —
now symmetric +-30; the catheter was controlled semi-blind (no tip position, heading or
cross-track feature); and **all 16 workers explored an identical target sequence**
(`random.Random()` deepcopied, unseeded explore resets never reseeded) — effective diversity
1/16 of nominal and PER received 16-way-correlated data. Transcript 44-54.

**Every "256x256" network before Gen-3 was effectively ONE hidden layer.** `mlp.py:83-87`
had no activation between input and first hidden layer (`# TODO: Add F.relu` left in place);
`Linear o Linear` is one affine map. Everything in the MLP era, including the 86% RCCA stack,
ran at half the intended depth. Fixed in Gen-3; old checkpoints are not warm-startable.
Transcript 64, 382.

**The catheter-shove exploit.** In v1b's late logs the catheter **leads the guidewire by
> 50 mm on 69% of steps**, max insertion 820 mm (machine 2: coil failures at the 898 mm cap).
Caused by frontier-only progress pay and unpriced catheter slack. The eval2-3 crest ran at
**6.2% cath-lead**; the decline at **58-62%**. This is the mechanism the v3c reward pair
(`--cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight 0.5`) targets. VALIDATED.
Transcript 6930-6984.

**Waypoint-density reward — both directions RULED OUT.** ENV2 (10 mm, 0.1): net +0.034 at
400 mm, step penalty wins early -> policy stops at ~47 mm. ENV3 (5 mm, 1.0): +0.14 at 30 mm
vs +1.83 at 400 mm risky -> locks into the shallow optimum at 7-30 mm. All three trained
policies insert LESS after training (-95/-96/-98%). Origin of the "no exploitable local
optimum before the target" constraint. `WAYPOINT_REWARDS_ANALYSIS.md`.

**MaxSteps-grounded-as-terminal — UNFIXED, and it taxes recovery.** With relaxed
truncations, buckled episodes run to 600, but `if truncated: terminated = True` bootstraps a
wire mid-recovery at the horizon as "nothing after" — systematically under-valuing
recoveries that pay off past 600. Left open because it touches the terminal semantics that
stabilised the RCCA critic. Combined with 14.1's "91% of recovered-but-failed die at the
cap", this is a concrete, cheap candidate for the conversion deficit. OPEN. Transcript 993-1739.

## 14.6 Evaluation integrity

**The eval ORBIT bug — un-reseeded rotation RNG. SPECIFIED, NEVER FIXED.** `reset_devices`
draws the initial device rotation from `self._rng`, which advances and never resets, so the
starting twist of eval episode N depends on **its position in the sequence, not its seed**.
The identical untrained policy scored **46.9% and 36.7% in two launches**; at position 2 one
run sent 16/16 into the RVA where the other succeeded 13/16 — that alone was the whole gap.
Anatomies and targets were byte-identical. Consequences: cross-launch noise floor **~ +-10
episodes, bimodal**; the same heuristic spans **37-47%**; and within-run deltas are clean
only while the policy is unchanged, because the orbit is created by the policy's own
actions. **This is a second, independent mechanism behind §10.6's ~18-point H0 variance,
and it is a property of the EVALUATOR** — it applies to every eval number in this file,
including the §13 host queue. "Deterministic eval" has never been true in the sense assumed.
Fix (specified): make initial rotation a deterministic function of the episode seed; a
verification test is written. Downgraded because multi-eval averaging is a free workaround.
OPEN. Transcript 6127-6150, 6582-6586, 6740.

**`global_steps` is a per-worker counter.** Ordering explore episodes by it produced "95% on
the first 300 -> solved at initialisation". Ordered by `wall_time` the run starts at
**69.0%** on the first 100 (pure heatup), 84.1% through heatup, 98.7% once updates begin —
reconciling with H0's 73.5%. VALIDATED as a correction. Transcript 11400-11408.

**`geometry_hash` must hash `vessel_tree.branches`, never the planned path.**
`pathfinder.path_points_vessel_cs` moves with the target, so an identity hash built on it
makes every preflight pass meaninglessly. Fixed. Transcript 8406.

**Worker logs drop each worker's final episode** (82 of 98 captured). Use
`episode_summary.jsonl` for eval accounting — but it is *incomplete for explore* (§5.7).
The two sources have opposite reliability. Transcript 6134, 6742.

**The generalised lesson from the two retracted nulls.** *An intervention targeting a
specific capability looks worthless when the evaluation cannot exercise that capability.*
Both "clean negatives" — the reward pair and stochastic eval — were measured against the
walled mesh where nothing ever reached a tight curve. Transcript ~9530.

## 14.7 Diagnostic instruments — and their own defects

**`diagnose_collapse.py` — and the baseline bug that inverted its verdict.** Joins the losses
CSV, probe JSONL, batch-sample JSONL, policy snapshots and worker logs into an event
timeline, a collapse onset, and a mechanism label. **The bug:** baseline was the median of
the first 200 updates, which spans startup (loss 6.0 -> 0.06), so update-1 read as a
"69-126x spike" and both test runs were classified as critic instability. Fixed by taking
the baseline from updates 100-1100. After the fix both reclassify to
`policy_collapse_or_suboptimal_attractor`, matching the manual analysis the tool had
contradicted. Pre-fix conclusions in `DIAG_TEST5_ANALYSIS.md` and `CRITIC_SPIKE_COMPARISON.md`
are RETRACTED by `CORRECTED_ANALYSIS.md`. **An automated detector's baseline window is a
domain assumption.**

**The deterministic freeze probe — and why absolute thresholds do not transfer.** `mean|a0|`
on fixed start states through every snapshot. v1's frozen policy measured 0.086; v1 grew
0.053 -> 0.255 then whipsawed back to 0.086 ~ its pretrain origin — **the freeze basin IS
the pretrain-BC attractor.** v2 tripped FREEZE-ALERT at 0.083 while scoring 30.6%, so the
probe was recalibrated to a **ratio against the run's own pretrain baseline** (OK >= 2x,
WATCH >= 1.25x; v2 reached 4.7x). Standing lesson: **probe the deterministic policy, not
explore returns** — v1 held 39% explore success with a dead mean because sigma = 1 sampling
masked it. VALIDATED. Transcript 3455-4145.

**Stuck detection is still a narrow proxy — spec never implemented.** The shipped detector
flags "freeze" as action mean ~0 and std < 0.1, which cannot see stuck-in-anatomy where
actions vary but insertion is ~0. Specified replacement: `stuck_step = (cmd_translation >
0.05) AND (|delta_ins| < 0.2 mm)`, >= 25 consecutive, with a 25 mm depth-bin mode. Also
flagged: `monoplanestatic` masks translation internally, so **commanded-vs-executed must be
logged** to separate "policy wants to push" from "env zeroed it" — the same defect class as
`extract_stuck.py`'s `abs(cmd_action[0])` (§10.6). OPEN. `reward_analysis_solution.md`.

**Nested multiprocessing silently voids SOFA queue timeouts.** 16 workers each spawning a
SOFA subprocess via `make_mp()` -> 9+ processes on Docker `/dev/shm`; `queue.get(timeout=60)`
ignored its timeout — measured gaps of **26 min, 1h36m, 4h18m and 5h48m**, 0 steps after
9 h. Fixed by `intervention.make_non_mp()` in `util/env.py`. Increasing `--shm-size` only
delays it. Companion: SOFA "Case 1 should never happen" traced to a 0.01 mm threshold under
aggressive heatup — `heatup_action_high` reduced to `[[15,1.0],[12,1.0]]`. VALIDATED.
`SOFA_TIMEOUT_FIX.md`.

**The update-budget formula counts heatup — the 500k-heatup trap.** `update_steps =
(heatup + explore)/20 - done`, so a 500k heatup demanded **27,500 updates in the first cycle**
against ~1000 random episodes, each sampled ~880 times — networks overfit to random data.
This is why `HEATUP_STEPS` is 1e4. Workers also run an entire 100-episode cycle on stale
weights; sync happens once per cycle. VALIDATED. `RL_TRAINING_LIFECYCLE.md`.

**Heuristic-seeding workers sampled identical targets.** Workers are `deepcopy(env_train)`,
cloning `CenterlineRandom._rng`; seeding resets with `seed=None`. **100 heuristic episodes
produced 7 unique targets** against a pool of 898 across 4 branches. Fix specified
(`--heuristic_seed_base`); later work added branch-balanced scheduling. OPEN in its doc.
`HEURISTIC_SEEDING_TARGET_DIVERSITY_FIX.md`.

**Heuristic-seeding waste census.** 320 episodes: **116 both-devices-maxed stalls wasting
59,600 steps = 28.5% of all logged steps**, mean wasted tail 514 steps; 71 short positive
truncations (truncated, not successful, positive reward — the seed lane's "good" episodes
are partly mislabelled); 45 actual successes. Detector spec with thresholds
(`BOTH_MAX_STALL_STEPS 8`, `OFF_BRANCH_GRACE_STEPS 10`, ...) scoped to heuristic mode only.
OPEN. `HEURISTIC_FAILURE_FIX_IMPLEMENTATION_SPEC.md`.

## 14.8 Algorithm arms — the detail behind §4

**v2b penalty-bracketing failed in the OPPOSITE direction — over-braked.** Two knobs on a
byte-identical v2 (`action_mean_penalty` 0.005 -> 0.02, `log_alpha_max` -2.3 -> -2.0):
baseline 2.0% @ 0.36 mm/s, eval1 **0.0%**, eval2 0.0%; deterministic `mean|a0|` frozen at
1.0x baseline across 180k updates; alpha sat at its floor. The 4x penalty neutralised the
O(0.01-0.05) early advantage tilt — "the engine never started". RULED OUT; this closes the
AWAC penalty-bracketing line behind §4's one-liner. `AWAC_STABILITY_EVOLUTION.md`.

**The AWAC pathology chain, named — for anything that reuses a BC term.** Every generation's
instability was **saturation of the squashed Gaussian** escaping through whichever bound was
loose: lambda = 3 with adv std 0.092 -> weights [0.72, 1.25] ~ uniform BC of the buffer
including the policy's own saturated successes; the hard `log_prob` floor at -20 zeroed the
BC gradient on exactly the far/high-advantage demos; entropy collapsed via **mean-rail**
even with sigma floored (tanh-Jacobian half is unbounded in the mean: -2.3 -> -10.2 over
650k updates); a hard clamp on `log_std` has zero gradient outside the band — a one-way
ratchet. Fixes: soft tanh-rescale `log_std` (-2, 0), `log_alpha` rails (-5, -2.3), leaky
log-prob floor, `action_mean_penalty`, per-dim clipped noise (was one scalar across 4 dims),
`EVE_CLEAN_RAIL_MAX` filter, `target_entropy` +1.0. v2 eval **6.1 -> 30.6 -> 49.0%** vs v1's
13.3% then freeze; train <-> held-out gap ~0. VALIDATED. `RL_IMPROV_15_CHANGES.md` Part E/G.

**RLPD's machinery was validated even though the run closed 0.0/0.0.** LayerNorm held Q
bounded for 226k updates with no BC anchor — the *stability* claim confirmed. The failure was
**signal density** (realised UTD ~0.25-0.27 vs the paper's 20), not instability.
`--critic_layernorm` and `--no_entropy_backup` were kept into P2. **§4 item 1 is
architectural (throughput-bound UTD) and does not generalise to a faster simulator.**

## 14.9 Specified with concrete designs, never built

- **E6 observation surgery**: prune ~21 dims (t-1 body offsets, `in_wrong_branch`,
  `d_rem_log`); add catheter **along-path** projection gap `s_gw - s_cath`, catheter
  cross-track, sin/cos cumulative commanded rotation as a deployable windup proxy, a
  stuck-duration integrator; fix `curv_ahead` scaling, radius /12 -> /6, target-dz (86%
  saturated at 50 mm), `at_ostium` dead-at-source, `br_trunk` never fires; **remove
  `ep_step` from the policy prefix**. Counter-evidence: the P2 residual already demoted
  ep-counters to rank 21. `RL_IMPROV_16_EXPERIMENTS.md` E6.
- **E7 escape bonus**: one-shot +0.3 the first time `proj_s` exceeds its pre-stall max after
  a stall that contained executed retraction (net di0 <= -1.5 mm), latched per event, cap
  3/episode. Paying on **escape** not retraction is what makes it unfarmable. Approval-gated.
- **Mode-conditioning** as the cheapest Axis-1 lever: feed a path-derived mode signal
  (wrong-branch flag, stuck-duration EMA, arclength-to-junction, curvature-ahead) so one net
  realises two behaviours. Ranked above a GMM head and far above two-actor MoE, which is
  deferred-last-resort (weakest at the mode boundary, splits data, touches the worker/sync
  plumbing where a one-line bug already cost 24 h). The root problem is data distribution:
  as path-following improves, stuck states vanish from the buffer.
- **The planned-path <-> force correspondence probe** with the circularity exclusion (14.3).
- **The eval orbit fix** (14.6).
- **`success@900`** secondary metric — mentioned, never built.
- **`recovery_tracker.py`** was never committed; exists only as a paste block in
  `user copy.md`. The committed canonical extractor is `extract_stuck.py`.

## 14.10 Inventory gap — `monitoring/` holds ~350 scripts; §2.3 lists 7

Families now on disk with committed outputs, each of which a future engineer will otherwise
rewrite: `buckle_clear_*` (~30), `cell1-4_*` (cross-matrix), `attack1/2/3_*` and `refute_*`
(the mesh/wall refutation suite), `audit22_*`, `check6a/8/10/11/14_*`, `t2/t3/t4_*`
(TopBrain adjudication), `nav_quality_nq*.py` + `out_nq/` (exact-estimator navigability),
`obs_ood_arjun_*`, `obs47_degeneracy.py`, `measure_mesh_quality.py`,
`target_dist_arjun_task*.py`, `split_*_check15_akr.py` (patient-level split optimiser),
`figure_carotid_anatomies.py` / `figure_topbrain_pairs.py` / `figure_vmr_0248.py`,
`smoke_topbrain_loader.py`, `topbrain_eval_flag_equiv.py` (BEFORE/AFTER equivalence of
`eval_anatomies.make_env` — the regression harness for evaluator edits),
`task2_arrest_colocation.py` (failure-mode classifier on the last-100-step window),
`probe_policy_v3.py`, `monitor_pass_v3a.sh`, `extract_chat.py`, and
`training _scripts/validate_experiment.py` (validates log/diagnostic/probe completeness for
a run, including in-progress ones). The rest is one-off scratch.

**Coordinate-system constants that must match across preprocessing, bench, human-play and
training:** `rotation_yzx_deg = [90, -90, 0]` and `fluoroscopy_rot_zx = [20, 5]`, used
identically in `vmr_processing_tools/create_dualdevicenav_format.py`,
`eve_bench/dualdevicenav.py`, the human-play scripts and the training scripts. The
anti-pattern is per-model rotations. `COORDINATE_SYSTEM_CONSISTENCY.md` — the intake step
upstream of §11/§12.
