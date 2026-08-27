# RL_IMPROV_18 — P2: Privileged-Actor Teacher, Residual on the Scripted Heuristic

> ⚠ **SUPERSEDED IN PART — see §9 (added 2026-08-24).** Sections 3, 4 and 7 were written
> 2026-07-28. Six of their conclusions were overturned by the mesh investigation of
> 07-29 → 08-03, including the §7 open question, which is answered and dissolved. The
> original text is left unedited as the record of what we believed at the time; §9
> states what is now true and why each earlier reading was wrong.

Branch `rl_improv_18_p2` (from `rl_improv_17_rlpd` @ 7fc1efb). Roadmap: RL_PARADIGM_ROADMAP.md P2.
Trigger: P1 RLPD closed negative 2026-07-16 (evals 0.0/0.0; start-state deterministic
mean never left random-init across 226k updates at realized UTD 0.39 — from-scratch
SAC cannot bootstrap this task at our sim throughput).

**STATUS 2026-07-28 — P2a is IMPLEMENTED, RUN TWICE, and MEASURED HONESTLY.**
The design works (RL transfers to real anatomy: +10.2 pp over the heuristic and ~10× fewer
steps), but it plateaus, and the **carotid siphon is an absolute wall: 0/30 on the real
patient vessel for every model tested, trained or not.** All headline numbers in this
program before 2026-07-27 were measured on ONE procedurally generated vessel — see §3.

---

## 1. Idea (unchanged, validated)

Stop bootstrapping from zero. The scripted `CenterlineFollowerHeuristic` already navigates
(P-control on heading + cross-track along the planned centerline, off-path retract phase).
P2a trains a **residual** on top of it with a **privileged actor** (Scaffolder,
arXiv:2405.14853: actor-side privileged information recovers most of the teacher-student
gap — critic-only left it on the table, which is what Gen-4 did). P2b (later) distills the
teacher to a deployable obs-only student via DAgger.

At init the residual mean ≈ 0 ⇒ behavior = pure heuristic ⇒ the run *starts* at heuristic
competence and RL only has to learn corrections.

**Validated:** the cold-start problem that killed P1 is gone. Every P2 run started at
heuristic level and never collapsed below it.

## 2. Mechanics (all default-off; legacy byte-identical) — IMPLEMENTED

| Piece | Where | What |
|---|---|---|
| Residual composition | `env5.BenchEnv5.step` | `a_total = clip(a_heur + residual_scale·a_policy)` in RAW units (worker de-normalizes before env.step; action shape (2,2)) |
| Heuristic ownership | `env5._heur_next_action` | Lazy `HeuristicActionFunction(self, noise_std=0, normalize_output=False)`; **once-per-state cache** (controller has phase counters — obs component and step() composition share one call); invalidated in `_on_intervention_stepped`, stale-marked before `super().reset()` |
| Heuristic intent obs | `env5.HeurActionObs` | 4-dim raw heuristic action, Normalize-wrapped, **before** the privileged tail (deployable prefix — the student sees it too). Obs 121→125, policy prefix 97→101 |
| Privileged actor | `DualDeviceNav_train` → `privileged_obs_dim=0` | `GaussianPolicy`'s `[..., :n]` slice becomes a no-op → policy consumes the full obs incl. the 24-dim tail. Guard: requires `--aux_coef 0` |
| Heatup band | `--heatup_action_scale 0.3` | Heatup = heuristic + small residual noise → buffer seeds at ~heuristic quality with diversity |
| Baseline eval | runner.py hoist | `--eval_after_pretrain` fires with `pretrain_updates 0` → measures the **pure heuristic**: the run's null hypothesis H₀ |
| Stuck lane indices | train script | slack 89 unchanged; contact 103→107 under `--heur_action_obs` |

Kept from P1: `--critic_layernorm` (Q bounded for 226k updates with no BC anchor),
`--no_entropy_backup`, sac alpha rails [−5, 0], target_entropy 1.0. Restored from v2:
PER + `--balanced_fraction 0.3`.

**Why the critic stays full-width and actions stay residual:** Q(s, a_res) is
Markov-consistent (a_heur is a deterministic function of sim state, and the critic sees the
privileged tail + heur_action dims); the buffer stores the policy's residual exactly as the
worker emitted it — no change to storage, PER, caches, or diagnostics.

---

## 3. ⚠ THE MEASUREMENT CRISIS (2026-07-27) — read before trusting any older number

**Every eval in this program, in every run, navigated ONE vessel tree.**

`DualDeviceNav_train` builds `env_eval` as
`DualDeviceNavRCCAVaried(seed=procedural_seed-1, episodes_between_change=10**9)`, and
`RCCAVariedFromMesh.reset()` regenerates only when
`episode_nr % episodes_between_change == 0` — a per-episode seed merely **re-seeds the RNG**.
Verified by hashing `vessel_tree.branches` coordinates: byte-identical across episodes. The
98 eval seeds varied only the **target** and the **device start rotation**.

Worse: **`RCCAVariedFromMesh.__init__` calls `_generate()` (line 282)**, so even that frozen
tree was a *procedurally generated variant*, never the patient vessel. Three separate
identity traps hid this:
1. `pathfinder.path_points_vessel_cs` moves with the target → looks like it varies.
2. `vessel_tree.mesh_fingerprint` is `s{seed}g{gen}` and the seed is reassigned each reset →
   the string changes while geometry is frozen; `g0` never incremented.
3. Only hashing the actual branch coordinates is trustworthy.

**Fixes shipped** (`f639fe1`, `61edfb6`):
- `synchron.py` + `agent.py`: **`env_eval_factory`** (mirrors `env_train_factory`,
  default None) → per-worker eval anatomy streams.
- `env5.py`: `EPISODE_START` now logs `anatomy=<branch-coord hash>` and `mesh_fp=` →
  diversity is verifiable in training too.
- `training _scripts/eval_anatomies.py` + `launch_eval_anatomies.sh`: standalone evaluator —
  per-worker anatomy streams, `--real_patient_anatomy` (zeroes `base_amp_mm`, `tortuosity`,
  `radius_scale`, and `rva_amp_mm`; self-verifying because zero perturbation ⇒
  seed-independent geometry), `--max_steps`, snapshots for every episode (successes AND
  failures), Wilson CIs, depth split, `--verify_variation`/`--frozen_anatomy` A/B.
- Audit-driven hardening: headline from the `Episode` objects (`infos[-1]["success"]`, the
  runner's own metric) + **seed reconciliation** (a worker restart silently drops that
  worker's remaining seeds), `spawn` start method (fork made all 16 workers share pid-1's
  log FD), per-invocation log/snapshot dirs (append-mode reuse merged runs), parse audit,
  600-min eval timeout (the 70-min default returns PARTIAL results silently).

**Also note:** `episode_summary.jsonl` logs only ~36% of explore episodes (14.4k ran, ~5.2k
logged) and its `explore_step` is the global 16-worker counter. Use it for *rates* only —
never for counts or timing. Worker-log `EPISODE_OUTCOME` lines are the complete ledger.

---

## 4. RESULTS

### 4.1 Runs

| run | what | outcome |
|---|---|---|
| `rcca_p2_teacher_v1` | first P2a launch | **INVALID** — trainer→worker weight sync never fired (all checkpoints byte-identical to init) and the deadlock guard restarted the trainer into fresh init 32×. Six evals of an *untrained* policy. Fixed in `c29af94` (R1–R3). |
| `rcca_p2_teacher_v1b` | first real training | H₀ 36.7 → evals 68.4 / 58.2 / **81.6** / 51.0 / 71.4 / 61.2 / 54.1 / 54.1 / 64.3. Explore plateaued ~61–62% from ~886k steps. Ended by OOM at 3.52M/6M (watchdog exit 42, as designed). |
| `rcca_p2_teacher_v1bp` | v1b + v3c reward pair ONLY (single variable) | H₀ 45.9 → 33.7 / 66.3 / 43.9 / 61.2 / 52.0 / 65.3 / 53.1 / 54.1 / 64.3. Plateau ~54%. OOM at 2.26M. |

### 4.2 The honest evaluation matrix (98 targets, 600 steps, deterministic; all integrity checks passed)

| checkpoint | 1 generated variant (old protocol) | 50 generated variants | **REAL PATIENT vessel** |
|---|---|---|---|
| v1b peak (`ckpt757854`) | 81.6% | 57.1% (CI 47–67) | **35.7%** (CI 27–46) |
| v1bp peak (`ckpt514264`) | 66.3% | 55.1% (CI 45–65) | **35.7%** |
| H₀ = `ckpt0` (untrained ⇒ heuristic) | 45.9% | — | **25.5%** (CI 18–35) |

Real patient, section-resolved (n = CCA 27 / ICA-mid 41 / siphon 30) + efficiency:

| model | CCA | ICA-mid | siphon | median steps-to-success |
|---|---|---|---|---|
| H₀ heuristic | 70.4% | 14.6% | **0/30** | **306** |
| v1b | **100%** | 19.5% | **0/30** | **32** |
| v1bp | **100%** | 19.5% | **0/30** | **30** |

### 4.3 Verified facts (measured, not inferred)

1. **RL transfers to real anatomy**: +10.2 pp over the heuristic, and v1b's 35 solved seeds
   are a **strict superset** of H₀'s 25 (0 solved-by-H₀-but-not-v1b, 10 the reverse).
2. **The biggest win is efficiency, not success**: ~10× fewer steps (306 → 30) at equal or
   better success. This is the most clinically meaningful result we have.
3. **CCA is solved** (70.4% → 100%); the loss is entirely ICA-mid and deeper.
4. **The siphon is an absolute wall**: 0/30 for heuristic, v1b, and v1bp alike. It coincides
   with the ~180° cavernous genua (R≈4–5 mm within ~15 mm) that `carotidsiphon.py` itself
   flags as "the catheter-overshoot region".
5. **The reward pair is a null result**: v1b and v1bp solve the **identical 35-seed set** on
   the real vessel, and differ only within noise across 50 variants. Their 50-variant results
   *do* differ (55.1 vs 57.1; ICA-mid 54.8 vs 61.9), proving the checkpoints are genuinely
   different models — so the equality on the real vessel is a property of the anatomy. The
   old 66.3-vs-81.6 "deficit" was single-anatomy luck.
6. **The generator makes EASIER vessels than reality** (57.1% vs 35.7%) — the train/eval
   distribution gap runs the wrong way; procedural variation is not a superset of the real
   anatomy.
7. **Step budget is not binding**: successes finish at median 30–45 steps (p90 105) of 600.
8. **Anatomy scope**: the chain ends at the **C7 terminus (MCA/ACA bifurcation, ~265 mm)**.
   **M1/MCA is NOT modelled.** Claims must say "to the ICA terminus".
9. **Recovery never emerged**: soft-recovery share stayed flat at 2–6% across 5,200 explore
   episodes in v1b. The 36%→60% explore gain came from *avoiding* stuck states (better clean
   navigation), not from escaping them. Two independent failure channels exist —
   permanent-stall (recovery-visible) and smooth-wrong-turn/timeout (recovery-invisible) —
   and they decouple success from recovery% (e.g. one eval scored 51% with only 10% recovery
   because 48 successes never stalled at all).

---

## 5. TWO-MODE ("stuck" vs "non-stuck") STRATEGY LADDER

The core problem: path-following and recovery interfere through the **data distribution** —
as path-following improves, stuck states vanish from the buffer, so the recovery skill loses
its gradient signal and decays. This is why soft recovery never emerged. The family of fixes
splits along **two independent axes**; a robust solution needs one lever from each.

### Axis 1 — let the model REPRESENT two modes

| rung | option | status |
|---|---|---|
| 1 | **Mode-conditioning via observation** — one policy fed a path-derived mode signal (wrong-branch flag, stuck-duration EMA, arclength-to-junction, curvature-ahead). A single net conditioned on an explicit mode bit can realize two different behaviors. **Cheapest, dominates the alternatives, try first.** | designed (§6 item A), not built |
| 2 | **Multimodal action head (GMM)** — one net outputs a mixture so commit-vs-retract can coexist *at a single state* instead of averaging. Only earns its cost if conditioning provably saturates. | parked (roadmap P3) |
| 3 | **Two actors + mode-classified dual buffers** — separate actors for stuck/non-stuck, path-classified, one run. | **deferred last-resort, NOT rejected.** Weakest exactly at the mode boundary where the gate is hardest; splits data; heavy surgery in the same worker/sync plumbing that already cost 24 h to a one-line bug. Use only if conditioning AND GMM both fail. |

### Axis 2 — TRAIN the stuck mode densely

| rung | option | status |
|---|---|---|
| 1 | **Stuck-lane sampling (E3)** — third SumTree lane oversampling stuck-state transitions (env5 flat idx 89/103→107) | **built**, never run on the P2 line |
| 2 | **Crunchpass lane (v1c)** — success-conditioned crunch resampling: `is_crunchpass = crunch_signature(obs) ∧ episode.success`. See §5.1. | **built, tested (6/6), pushed `07aecf4`, NOT launched** |
| 3 | **Stuck-restore curriculum (E4)** — start episodes *from* harvested stuck states so recovery is the on-policy path to reward | blocked on machine-2 pool; **and see the v3b lesson below** |

### 5.1 Machine-2 learnings that reshape Axis 2

- **v3b stuck-restore as designed FAILS**: by the time the wire is "truly stuck" it is knotted
  and coiled — unrecoverable without hundreds of retraction steps. **Harvest crunch-ENTRY
  states (the last moment before the knot), not stuck states.**
- **A crunch classifier from planned-path geometry works**: machine-2 reported ~90%;
  reproduced here at **AUC 0.92–0.95** on v1b chunks using 5 features (local_radius 93,
  arc_past_daughter 58, gw_slack 89, cath_offset_z 77, gw_cath_gap 78).
- **Privilege can be recovered for the student**: deployable features → high-contact at
  **AUC 0.65–0.77** with 9 features (machine-2 got 0.788 with top-15). This is the P2b
  student's aux head.
- **The raw crunch signature marks GRINDING, not skill**: failures carry **4–8× more**
  signature steps than successes in every era. Hence the lane must be **success-conditioned**
  — which is exactly what v1c implements.
- **Era evidence**: v1b's 81.6% crest era ran at **6.2% catheter-lead** with 41 crunch
  steps/success-episode; its decline ran at **58–62% catheter-lead**. Coordinated two-device
  crunch work coincides with the peak.

### 5.2 Recommended composition

**One Axis-1 lever + one Axis-2 lever.** Cheapest robust bundle = **path-context/mode
observation (A) + crunchpass lane (v1c)**. Escalate the architecture (GMM) only on evidence
that conditioning saturated; two-actor MoE is the last rung.

⚠ **But see §7** — post-2026-07-28, the siphon being 0/30 for *every* model (including the
heuristic) means the binding constraint may not be a learning problem at all.

---

## 6. PARKED / PENDING CHANGES (complete inventory)

### 6.1 Built but never launched
| # | item | state |
|---|---|---|
| 1 | **v1c crunchpass lane** — success-conditioned crunch resampling; `--crunchpass_fraction/_engage_thresh/_radius_thresh`; 4th SumTree lane; 6/6 container tests | pushed `07aecf4`, launcher `launch_rcca_p2_teacher_v1c.sh` ready |
| 2 | **v1d** = v1c + v3c reward pair | launcher ready; ⚠ now low priority (pair = null result) |
| 3 | **E3 stuck lane** | built, unused on this line |

### 6.2 Infrastructure debt (ship with whatever launches next)
| # | item | why |
|---|---|---|
| 4 | **Trainer restart must RESTORE state, not re-init** (deep-dive R1) | I stall-gated the deadline so it can't fire spuriously, but a *genuine* deadlock still respawns from init — latent landmine |
| 5 | **Re-arm probe logging after trainer restart** (R3) | probe JSONL covered 1 of 32 segments in the v1 run |
| 6 | **Remaining invariant alarms** (R4) | NET_SYNC fingerprint alarm shipped; still missing snapshot-hash-unchanged and the init-wipe signature (α==1.000 ∧ q1≈0 after a CSV gap) |
| 7 | **Memory cap / replay trim** | OOM killed BOTH v1b (3.52M) and v1bp (2.26M) at ~day 4 |
| 8 | **Launch seeding + `policy_0.pt` dump** | reproducible inits |
| 9 | **Eval accounting from worker logs** | `episode_summary.jsonl` drops ~64% of explore episodes |
| 10 | **P0 eval-state reset** (un-reseeded rotation RNG in `reset_devices` ⇒ start twist depends on episode *position*) | downgraded: only affects cross-run/best-checkpoint claims; multi-eval averaging is the free workaround |

### 6.3 Frozen — needs explicit approval (reward changes)
| # | item |
|---|---|
| 11 | **"reduce-slack" potential reward** — telescoping/non-farmable, same family as `buckle_reward`. NB the policy discovered *some* soft recovery without it, and the v3c pair (same family) was a null result — evidence for it is now weaker |

### 6.4 Roadmap
| # | item |
|---|---|
| 12 | **P2b — DAgger student distillation**: obs-only student (deployable prefix + short history replacing ep_step) + **aux head predicting the privileged tail from history** (RMA-style; AUC 0.65–0.77 already demonstrated). The deployable number — and the highest-uncertainty one |
| 13 | **P3 — GMM residual head**; then QC-FQL / Q-chunking |
| 14 | **Machine-2 v3a Tier-A run** (E1b adv-norm + E2 aux z-norm + E3 stuck lane) — handoff ready since day 1, never launched; its harvest feeds E4 |
| 15 | **v2a/c/d AWAC penalty variants** on `rl_improv_15` |
| 16 | **Multi-mesh generalization** (per-worker mesh families beyond procedural RCCA) |
| 17 | **Obs pruning** (~21 dims flagged) + ep_step removal. NB the v1b saliency probe showed ep-counters at rank ~21/125 (NOT the #1 input it was in v2) and privileged dims at rank 2–3 — the residual design already defused the time-hack |

### 6.5 Tooling / docs
| # | item |
|---|---|
| 18 | **`recovery_tracker.py` → commit to `monitoring/`** (currently only a paste-block in `user copy.md` + scratchpad) |
| 19 | **PAPER_PLAN_NEUROVASC_RL.md corrections**: eval-protocol section (single-anatomy bug), every absolute number, the M1/terminus scope, and the C3 "diversity vs competence" figures (it's 45.9% vs ~54%, not ≈0% vs 54%) |
| 20 | **Extend the vessel tree past C7 to M1/MCA** if the clinical story needs thrombectomy-relevant targets — a vessel-tree change, not an RL change |

---

## 7. OPEN QUESTION THAT OUTRANKS EVERYTHING ELSE

**The siphon is 0/30 on the real patient vessel for the heuristic, for v1b, and for v1bp.**

No training, no reward variant, and no amount of procedural experience moved it. Before
spending another multi-day run on Axis-1/Axis-2 levers, we should determine *why*:

- Is it **kinematically reachable at all** with this device pair and action space? (A
  scripted open-loop probe or a human-in-the-loop attempt on the real vessel would answer
  this in hours, not days.)
- Is it the **catheter-overshoot geometry** at the cavernous genua (the file's own warning),
  i.e. a control-bandwidth/physics problem rather than a policy problem?
- Does the **procedural generator's easier siphon** mean we have simply never trained on
  anything as hard as the real one? (57.1% vs 35.7% says the distribution is softer.)

If the siphon is not reachable by *any* controller under the current action space/step
duration, then crunchpass, GMM heads, and MoE are all aimed at the wrong target, and the
honest paper claim is: **cervical-to-petrous ICA navigation with ~10× efficiency over a
scripted controller, with the cavernous siphon as an open problem.**

---

## 8. P2b — student distillation (unchanged plan)

DAgger: run the teacher, log (deployable-prefix obs → teacher action) pairs on-policy, train
an obs-only student; student evaluated WITHOUT the privileged tail. Add an aux head
predicting the privileged tail from history. Optionally warm-start from the teacher's prefix
weights. **Gate:** only worth doing once the teacher's real-anatomy number is worth
inheriting — currently 35.7%, of which the heuristic supplies 25.5%.

---

# 9. CORRECTIONS — 2026-07-29 → 08-03

**Read this before §3, §4 or §7.** Those sections were written 07-28. Everything below
landed afterwards and overturns six of their conclusions, including the §7 open question,
which is now answered and dissolved. Nothing above this line has been edited; the record of
what we believed on 07-28 is left intact deliberately.

## 9.1 The real-patient evaluation never evaluated the real patient

`--real_patient_anatomy` built `DualDeviceNavRCCAVaried` with every perturbation amplitude
zeroed. That reproduces the patient **centerlines** to zero floating-point error — verified,
`max|diff| = 0.000000 mm` — which is exactly why it passed every check we had: path lengths,
target sampling, branch identity and the geometry hash all matched.

But `RCCAVariedFromMesh` **always re-meshes the tree from those centerlines**
(`rccavariedfrommesh.py:10`; `mesh_path → generate_temp_mesh`). The wire collides with the
*surface*, not the centerline, and the regenerated surface is not the patient's:

| | `--real_patient_anatomy` (old) | true original `.obj` |
|---|---|---|
| class | `DualDeviceNavRCCAVaried` (amps 0) | `DualDeviceNav` |
| mesh cells | 3,721 | 3,584 |
| median clearance | 1.23 mm | **2.14 mm** |
| p05 / min clearance | 0.40 / 0.06 mm | 1.20 / 0.22 mm |
| blocked stations (wire radius 0.18 mm) | **2 / 235** | **0 / 235 — passable** |
| first block | raw 120.4 mm → **proj_s ≈ 154.0** | none |
| centerlines | **identical, max\|diff\| = 0.000000 mm** | |

**Behavioural confirmation, measured independently.** On the re-meshed surface, 91% of
real-patient failures arrest in a 2 mm window: v1b 63/69 (modal 153.4, 51×), v1bp 63/69
(modal 153.3, 61×), and the **parameterless heuristic** 41/74. `max solved path_len = 156.2`
for all three. Geometry predicted 154.0; three controllers were measured at 153.4 — **0.6 mm
agreement, nothing fitted to anything.**

A station that arrests a hand-written centerline follower *and* two independently trained
policies at the same tenth of a millimetre cannot be a policy defect.

**And it is not the siphon.** 153.4 mm sits 7 mm past the ICA-mid cut and **57 mm before the
siphon band begins (210 mm)**. §4.3 #4's "absolute wall" was an arrest well proximal to the
anatomy it was attributed to.

### The fix, and the wrong fix that preceded it

Constructing `DualDeviceNav` directly is **not** the fix: it also moves the insertion point to
the femoral entry and swaps the target sampler to four branches with no minimum arclength. A
4-episode smoke test returned `path_len` 594–752 mm against 103–156 mm for every prior run —
a different task, not a different mesh. Had that gone straight to 98 episodes it would have
produced a confident, catastrophic, and entirely spurious result.

The correct fix keeps `DualDeviceNavRCCAVaried` — insertion at the (11) bridge, RCCA-only
targets, same devices and frames — and swaps **only the collision surface** for
`DualDeviceNav`'s `FromMesh` output, which is the original `.obj` already rotated into the
branch frame. One variable. (`eb753d9`)

Implementation note worth keeping: `_generate()` nulls `_mesh_path` and `reset()` calls it, so
the re-meshed surface returns on the first reset. The pin must be a **class-method patch** —
a closure bound to the instance, or a class defined inside a function, both fail to pickle to
spawned workers.

## 9.2 The synthetic anatomies were largely impassable

The same defect at scale. Measured with `monitoring/mesh_clearance.py` (**no controller
involved** — clearance is a property of the geometry):

| mesh | median | p05 | blocked stations |
|---|---|---|---|
| patient-derived reference | **2.14 mm** | 1.20 | 0 / 235 |
| generated, `radius_scale = 1.0` (as trained) | 1.09–1.33 mm | 0.22–0.63 | **4 of 6 anatomies blocked** |

Generated vessels carry roughly **half** the clearance of the source anatomy, and two thirds
of a six-anatomy sample contain stations where the 0.36 mm guidewire physically cannot fit.
Blocked stations at raw arclength 116–134 mm map to proj_s ≈ 150–168, bracketing the measured
arrest.

**Per-anatomy, model-independent walls.** Pooled arrest across the 50-anatomy eval scatters
(SD 34 mm), which reads as ordinary navigation failure. Grouping by anatomy hash shows each
anatomy has its own fixed station: **80% (v1b) / 69% (v1bp)** of anatomies with ≥2 deep
failures have within-anatomy arrest SD < 2 mm, carrying **45% / 41% of all failure mass**. The
clincher is cross-model agreement on the same mesh — 11d068 @152.1, bfadcd @166.3, 8ba8a3
@157.2, dd7d9c @160.8, all ±0.00 for **both** models.

### The fix: gate and calibrate

`--require_passable` rejects and regenerates until an anatomy satisfies **two** conditions:
min clearance ≥ wire radius (the device fits at all) **and** median clearance ≥
`--passable_min_median_mm` (default 2.00). The second matters because fitting is not
sufficient — a raw generated mesh at 1.26 mm median is navigable in principle yet far tighter
than any real patient.

`--radius_scale` compensates the mesher's erosion. Measured calibration:

| radius_scale | median | p05 | min | blocked / 3 |
|---|---|---|---|---|
| ORIGINAL `.obj` | 2.14 | 1.20 | 0.22 | 0 |
| 1.0 (as trained) | 1.26 | 0.46 | 0.24 | 1 |
| 1.4 | 1.80 | 1.12 | 0.68 | 0 |
| **1.6** | **2.14** | **1.13** | 0.75 | **0** |
| 1.8 | 2.49 | 1.16 | 0.75 | 0 |

**1.6 reproduces the patient almost exactly** — equal median, marginally *tighter* p05 — and
pins the mesher's erosion at ~37% of radius. Gating at generation time keeps all 98 episodes
on qualifying anatomies; post-filtering would have left ~33 with a ±17 pp CI.

## 9.3 Checkpoint selection was corrupted by the same bug

The single-anatomy eval crowned **early** checkpoints. Scoring every checkpoint against
explore success within ±150k steps:

| run | eval-picked | its local explore succ | **true best** | its local explore succ |
|---|---|---|---|---|
| v1b | 757854 | 56.1% | **3259127** | **63.2%** |
| v1bp | 514264 | 53.3% | **2002292** | **59.2%** |

Both crowned checkpoints sit at roughly a fifth of training. Selecting on a high-variance
n=1-anatomy signal does not merely add noise — it **systematically favours whichever
checkpoint overfits that particular tree**.

## 9.4 Corrected results

**Real patient, original segmented surface** (98 eps, deterministic, 600 steps):

| model | overall | CCA | ICA-mid | siphon |
|---|---|---|---|---|
| v1b ckpt757854 (eval-picked) | 52.0% | 92.6% | 58.5% | 6.7% (2/30) |
| v1b ckpt3259127 (true best) | 63.3% | 100% | 65.9% | 26.7% (8/30) |
| v1bp ckpt514264 (eval-picked) | 72.4% | 88.9% | 63.4% | **70.0% (21/30)** |
| **v1bp ckpt2002292 (true best)** | **75.5%** | 100% | **90.2%** | 33.3% (10/30) |

**Calibrated, verified-passable synthetic** (50 anatomies, `--require_passable`,
`--radius_scale 1.6`):

| model | overall | CCA | ICA-mid | siphon |
|---|---|---|---|---|
| **v1bp ckpt2002292** | **84.7%** (83/98) | 100% (26/26) | 100% (40/40) | 53.1% (17/32) |
| v1bp ckpt514264 | 83.7% (82/98) | 100% (26/26) | 100% (40/40) | 50.0% (16/32) |
| v1b ckpt3259127 | **NOT RUN** — cancelled mid-queue; ~1.5 h to fill | | | |

**CCA and ICA-mid are solved on synthetic anatomy — 66/66 across both models, zero failures.
Every remaining failure in that evaluation is a siphon target.**

## 9.5 What this overturns in §3, §4, §7

| where | said | now |
|---|---|---|
| §4.1 | both runs "ended by OOM" | **IPC-deadlock watchdog** (`os._exit 42`); `grep OOM` on both `main.log` = 0 hits. §6.2 item 7 is misdiagnosed and should be re-scoped to deadlock robustness |
| §4.2 | real patient 35.7% / 35.7% / H₀ 25.5% | measured on a reconstruction. True surface: 52.0 → 63.3 (v1b), **72.4 → 75.5** (v1bp) |
| §4.2 | 50 generated 57.1 / 55.1 | measured on largely impassable anatomies. Calibrated: **84.7 / 83.7** |
| §4.3 #4 | "the siphon is an absolute wall, 0/30" | **retracted** — an arrest 57 mm proximal to the siphon. v1bp reaches 70.0% siphon on the true surface |
| §4.3 #5 | "the reward pair is a null result" | **retracted** — on the corrected mesh, **+20.4 pp overall and +63.3 pp at the siphon**. It improves recovery and removes the catheter shove; on a walled mesh nothing reached a tight curve, so the benefit was unmeasurable, not absent |
| §4.3 #6 | "the generator makes EASIER vessels than reality" | **inverted** — it makes *impassable* ones, at half the calibre |
| §4.3 #9 | "recovery never emerged; soft share 2–6%" | detector-dependent (canonical: 22.6% / 29.5%). Its successor finding (r = −0.82) is **provisional** — computed on walled anatomies, never re-stratified |
| §6.4 #12 | P2b gate "currently 35.7%" | **75.5%** — gate passed |
| §7 | "why is the siphon 0/30?" | **dissolved.** It was never the siphon |

## 9.6 Measurement errors made *during* this investigation

Recorded because each was asserted before being measured, and each cost a cycle:

1. **"Lumen erosion refuted — 1% distal shrink."** Wrong: measured with nearest-**vertex**
   distance. On a 6 mm-triangle mesh the vertices sit outside the facets and overstate
   clearance. Surface-sampled measurement gives ~45%. **Never measure lumen clearance from
   mesh vertices.**
2. **"The re-mesh is catastrophically under-resolved."** Wrong: the original collision mesh
   is equally coarse (6.28 vs 6.46 mm median edge) and *is* passable. Coarseness is a
   long-standing property of the benchmark, not a Gen-4 regression. Note the *visual* mesh is
   12× finer than the one physics collides against.
3. **"Collision-chord discretization is the cause."** Eliminated by the historical fixed-mesh
   run: same devices, same settings, traversed the siphon.
4. **"Depth-stratified checkpoint selection is required."** Based on the real-patient siphon
   split (70.0% vs 33.3%, ~3 SE). Across 50 anatomies the two are indistinguishable — 53.1%
   vs 50.0%, one episode. The split was **anatomy-specific**; selecting on it would overfit a
   single vessel. n=30 on one anatomy cannot support a claim about a policy.
5. **"Success% tracks recovery%."** Mixed a per-**event** rate against a per-**episode** rate.
   Separated, v1b's correlation is r = −0.82.

## 9.7 Code-state findings that contradict §2

From an audit of the shipped `runner.yml` / `env_train.yml` / launchers:

1. **The algorithm is SAC**, not AWAC — both launchers pass `--algo sac`; AWAC is gated behind
   `self.algo == 'awac'` and inert. `awac_lambda: 3.0` still appears in `runner.yml` but is dead.
2. **The asymmetric actor-critic is OFF in v1b and v1bp.** `--privileged_actor` sets
   `privileged_obs_dim = 0`, making `GaussianPolicy`'s input slice a no-op; `runner.yml`
   records policy `n_observations: 125`, identical to the critics. **The shipped policy reads
   SOFA nodal forces at training *and* test time** — it is a teacher, not a deployable
   controller. Both headline numbers are teacher numbers.
3. **No student was ever trained.** `dagger|distill|student` returns only future-tense
   comments. No observation-only number exists anywhere in the repo.
4. **The planned-path ↔ force correspondence is asserted, not demonstrated.** Its dominant
   predictor (`gw_slack`) is *numerically identical* to privileged dim 23, computed from the
   same projection — an identity, not a proxy.
5. **UTD is 0.99** (measured), not the 0.25 in `PAPER_PLAN`; **throughput 10–11 env-steps/s**,
   not 47.
6. **Eval anatomy seeding is misdescribed** in §3: the per-worker factory seed builds only the
   first tree; the *episode* seed drives every subsequent regeneration.

## 9.8 State after these corrections

**Implemented and verified**
- `--real_patient_anatomy` — pins the original surface (`eb753d9`)
- `--require_passable`, `--passable_min_median_mm`, `--radius_scale` (`cb6bceb`)
- `--stochastic_eval` (default off) — result was a clean null: strict superset, +1 episode.
  Note it was measured on the **walled** surface, so it tested nothing meaningful; cheap to redo
- `monitoring/mesh_clearance.py` — passability gate, no controller required
- Stuck/recovery toolchain: `extract_stuck`, `analyze_stuck`, `verify_stuck`, `report_stuck`,
  `report_single`; explore/eval separated exactly by the `seed=` field in `EPISODE_START`
  (main.log time windows leak — they caught 567 of 980)

**Parked mid-investigation**
- Mesh-generator fix (`MESH_GENERATOR_FIX_PLAN.md`): diagnosis measured; **ablation written and
  never run** (`monitoring/mesh_ablation.py`); mechanism hypothesis unconfirmed. Note the eval
  path is now *worked around* by `--radius_scale`, but **training still generates uncalibrated,
  largely impassable anatomies** — this is the substantive open item

**Owed re-analysis — CPU only, data in hand**
- Re-stratify r = −0.82 on audited anatomies
- v1b calibrated-synthetic eval (~1.5 h) to complete §9.4

**Never built**
- P2b student (gate now passed at 75.5%)
- A genuinely asymmetric run (`privileged_obs_dim > 0`)
- The correspondence probe

**Re-prioritised by these results**
- **v1c crunchpass premise is contested.** It amplifies soft recoveries; grind-only episodes
  outperform soft in *both* runs (99%/95% vs 95%/76%), and recovery type appears to proxy stall
  *severity* rather than skill. Break that confound within matched depth strata before launching
- The distribution gap runs the **opposite** way to §4.3 #6: training anatomies are harder than
  reality in the sense of being impassable, not harder in the sense of being more tortuous
