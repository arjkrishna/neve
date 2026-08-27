# DEEP DIVE — rcca_p2_teacher_v1 (P2a): Why Nothing Compounded, What the Data Actually Says, and the v2 Plan

**Run:** `d:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-16_235715_rcca_p2_teacher_v1`
**Sources (only):** Eval Forensics report (checkpoint-joined, n=98/eval, reconciles reported SR exactly), Training Autopsy report (weight-hash + CSV forensics), Bimodality Evidence report (1.86M STEP lines, mode-binned counterfactual sweep), Design Survey report (candidates A–G).
**Convention:** claims not directly established by those four reports are marked **UNVERIFIED**.

---

## Executive summary — the causal story in 10 lines

1. **The run never deployed a trained weight.** All 142 `policy_*.pt` snapshots and all 7 `checkpoint*.everl` hash byte-identical to initialization (policy md5 `86c50d9fff7a`, critic q1 `ed65a4ccb3da`); the deployed residual was ~zero all run (|tanh(mu)| ≈ 0.001–0.004 per dim ≈ 0.08 mm/s on a 30 mm/s scale).
2. **Two independent infra breaks:** (a) trainer→agent/worker/eval **weight sync never fired once** in 1.45M explore steps — checkpoints saved at update 117k+ are still init; (b) the **deadlock guard restarted the trainer 32 times** (~every 23.4k updates, ~40 min) into a *fresh init* — alpha snaps to exactly 1.000 and q1_mean to ~−0.007 at each of 31 CSV wall-time gaps. The e3bf215 post-eval-deadlock fix did not hold.
3. Therefore "H0 strong but nothing compounded" is **not an RL pathology**: every eval executed the identical init policy; the 45.9→48.0→49.0→46.9→45.9→45.9 movement is episode-mix noise (same 98 seeds each eval: 44 always fail, 41 always succeed, 13 mixed → repeat-eval SE ≈ ±2pp, matching the observed 3.1pp range; the naive binomial ±5pp overstates it).
4. The recurring "alpha limit-cycle ~24k period" **is the restart sawtooth** (init 1.0 → anneal 0.01 → wipe → repeat, ×32); the "monotone reward decline 0.887→0.556" is an aggregation artifact — per-episode eval-window means are non-monotone.
5. Silver lining: six evals = ~590 episodes of clean **H0 characterization**. Failure mass (all max_steps): **58.3% deep-RVA capture** — enter RVA early, spend median 505–515/600 steps inside, **retraction ≈ 0.0 mm** — vs ~34% near-target grind (ratio>0.7); truly-at-the-door (<10 mm) is only ~4% by the checkpoint-joined method.
6. **U-turns are a red herring:** 16.2% of failures, median 4 RVA steps and 0.0 mm retraction — a cheap fork-brush that successes also make. RVA-before-RCCA never happens (0/259). The step waste belongs to the 148 never-exiting RVA captures.
7. **Straight speed is frozen at H0's compromise** (0.53 mm/step; ~5 mm/s commanded = 17% of max, p10–p90 = 4.0–6.0 in every eval) because no learning was ever delivered; the explore-noise counterfactual sweep shows free-mode optimum ≈ **25–30 mm/s** and crunch optimum ≈ **0 mm/s** — the bimodal *target* is confirmed; unimodal-averaging by the *learner* is untested (no learner was deployed).
8. Speed alone converts only ~20–22% of timeouts (mean recoverable ≈ 18 mm vs median deficit 63 mm); the dominant lever is the **commit-without-retract RVA mode** — exactly the target of E4/v3b stuck-restore + wrong-branch observability, and the same untransmitted-retract-preference pathology from the v2 findings.
9. **P2a-v2 = deliverability first** (deadlock-restore + weight-sync fix + invariant alarms), then F (α ceiling, config-only insurance), A (path-context/wrong-branch obs, ~6 dims), D (stuck-restore `--restore_prob 0.3` if pool ready at launch), G-lite (eval-side success@900) + E8 probes, plus a scripted H0 wrong-branch-escape ablation as the no-RL cost floor. **Ceiling arithmetic: ~62–73% by the Q6 table (~60% floor after cohort-overlap haircut) vs H0 ≈ 47%.**
10. **v3 backup:** GMM residual head (C) if v2 ≤ H0+3 at eval3 or the retract probe stays flat. Two-actor MoE (B) rejected at any stage.

---

## Q1 — Why was H0 strong but nothing compounded?

**Answer: nothing was ever delivered to compound.** The Training Autopsy is unambiguous:

- **Smoking gun:** every saved policy (142 snapshots spanning 10k→1,420k explore; metadata `update_step` 0→724,491) and every checkpoint (checkpoint0 → checkpoint1287640 + best) hashes to the same init weights, critics included. Forward passes of 6 snapshots on 3k early + 3k late buffer states give identical outputs.
- **Mechanism 1 — restart wipe:** 32 "Restarting Trainer because of no update-result within 1800s (deadlock guard)" events; 31 CSV wall-time gaps at ~23.4k-update spacing; at every gap alpha 0.01–0.02 → **1.000** and q1_mean ~1.5–2.4 → **−0.007** (freshly initialized nets). The restarted trainer does not restore trained state.
- **Mechanism 2 — sync never fired:** even during healthy segments, workers never received trainer weights. During segment 1's live window (explore 44.5k–71k), when the trainer's mu carried a +0.6–0.8 gw_t bias, buffer actions show signed gw_t mean +0.004 — indistinguishable from frozen mu≈0 + std≈0.37 sampling. |a| mean 0.332 constant across 1.3M transitions.
- **Consequences:** eval1 (48.0) ran before the first gradient update even landed; the eval SR spread 45.9–49.0 is pure noise around the pooled H0 estimate ≈46.9% (45.9 is merely the eval1/5/6 draw); the "alpha limit-cycle" and "reward decline" diagnoses dissolve as artifacts (Finding 3 / Task 1). Prior runs' "P1+P2 alpha limit-cycle" conclusions should be **re-audited for the restart-wipe signature** (alpha==1.000 ∧ q1_mean≈0 after a CSV gap) before being trusted.
- **Learning did occur trainer-side, within segments:** probe JSONL (segment 1 only — probing was never re-armed after restarts, a secondary bug) shows |tanh(mu)| gw_trans 0.023→0.81 (a +24 mm/s standing forward bias) and probe q 0→~2.9 in 23k updates. Whether that runaway residual was *good* is **UNVERIFIED** — it was never evaluated or deployed. Notably its direction (strong forward push) is at least sign-consistent with the free-mode speed deficit in Q3.

**Implication:** every conclusion about "what SAC learned on top of H0" from this run is void. The run's value is (a) the infra bug discovery and (b) a large, clean behavioral census of H0 itself.

---

## Q2 — Verified failure taxonomy (timeout-near vs junction-stuck; U-turn quantified)

Checkpoint-labeled failures, E1–E5, aggregate n=259; **every failure is max_steps** — zero other termination reasons. (Forensics report; join = 98/98 eps per eval, success counts match reported SR exactly.)

| Mode | Share of failures | Signature |
|---|---|---|
| **Deep-RVA capture** | **151/259 = 58.3%** | ratio≤0.7 (median 0.2); 27–33/eval end inside RVA (an end-position superset, ≈162–198 total — a few end-in-RVA failures fall outside the 151 ratio-classified captures); median **505–515 of 600 steps in RVA (~85%)** — entered early, never recovered; retraction ≈ 0 mm |
| **Timeout-near-target** (ratio>0.7) | 88/259 ≈ 34% (39.2→30.2% across evals — composition noise under a constant policy, not a trend) | but ratio>0.9 only 4–5/eval; final d_tgt<10 mm only 2–3/eval (**4.2% aggregate**); failure median d_tgt 51–62 mm |
| "Stuck at the RCCA/RVA fork" | **0/259** | refuted as stated — failures end deep INSIDE RVA, not at the fork (d_corr_arc<15 & arc_past=0 never observed at episode end) |
| RVA entered before RCCA | **0/259** | physical sequence is always bridge→RCCA→(RVA); the diversion happens off the RCCA path |

**The U-turn, quantified:** 42/259 failures (16.2%) show RVA entry followed by RCCA re-engagement with proj_s advancing >5 mm — but median RVA dwell of U-turners is **4 steps** and median retraction **0.0 mm**. It is a cheap fork-brush (slip into the RVA ostium, slide back onto RCCA), not a costly re-thread. Successes brush the same way (3–10 per eval touch RVA, all "U-turn", median 3–4.5 RVA steps). **The step waste belongs to the 148 non-recovering RVA captures, not the U-turners.** 73.4% of failures enter RVA at all.

**Label subtlety worth keeping:** Quality's `success` counts episodes reaching d_tgt≈5.0 *at* the 600-step cap as SUCCESS (7/9/4/6/6/5 eps per eval, 13% of all successes); worker `grader_success` is stricter — the 37 at-cap successes explain most (~6.3pp) of the ~8pp gap between raw-log SR ≈39% and reported ≈47%. The genuinely-at-the-door cases mostly ARE these at-cap successes.

**Reconciliation with the Bimodality report** (window-heuristic method, n=318 window timeouts, deficit = path_len − proj_s): it reports <10 mm = 20%, <50 mm = 40%, U-turn 35%, RVA-entered 57%, ≥100 steps near RVA 44% (p75 = 507 steps). The two methods agree on the structure (dominant junction-trap mode with enormous dwell; near-miss cohort real but minority; "half were close" holds only at a generous ≤50 mm cutoff) and disagree on cohort sizes because of different labels, denominators, and distance definitions. **The checkpoint-joined Forensics numbers are authoritative** (they reconcile with reported SR exactly); treat the Bimodality cohort sizes as method-variant bounds.

**Seed determinism:** 44/98 seeds always fail, 41 always succeed, 13 mixed — under a byte-identical policy, so the mixed 13 measure env/episode stochasticity, not learning (repeat-eval SE ≤ √(13·0.25)/98 ≈ ±1.8pp). **Any real gain must flip whole seeds out of the 44 always-fail set (~45pp of headroom); shaving steps moves nothing.**

---

## Q3 — Why did straight-section speed never improve?

Three stacked answers:

1. **Proximate (sufficient by itself):** no trained weights were ever deployed (Q1). Bridge speed 0.536→0.531 mm/step across six evals (±1% over 1.29M explore steps) is simply H0 replayed six times. Failures' bridge speed 0.497–0.502 every eval — ~6% below successes' and equally frozen. The residual modulated speed not at all, in either cohort, because it was ~0.
2. **Structural (what H0 does):** the deployed controller emits a narrow ~5 mm/s spike in free lumen (free-mode a0 mean 5.0–5.2, p10–p90 = 4.0–6.0, identical eval1→eval5) = **17% of the 30 mm/s max**, and still pushes ~+3–5 in crunch with only a heuristic retract tail (p10 ≈ −7). The C-vs-F gap (1.9 vs 5.0) is the scripted junction logic. H0's single compromise value sits near the crunch-safe end: it bought fold-avoidance by permanently sacrificing free-mode speed.
3. **Counterfactual headroom (from explore noise, std≈12, which sweeps the action space):** free-mode dps rises monotonically with a0 — 0.61 mm/step @ [1,5) → **1.18 @ [25,30)**, reward peak at [20,25); so full throttle would roughly double free-mode progress (0.58→1.10 mm/step). **BUT** timeout episodes contain only ~13 free steps of ~600 (median 90% of steps in crunch): a full free+mixed speedup recovers ~18 mm mean per timeout vs a **median deficit of 63 mm** — covering only the ~20–22% of timeouts with deficit <20 mm. **Speed cannot fix the median timeout; it converts the close-call cohort only.**

Also noted (Forensics): in-daughter (RCCA) speed declined slightly 0.386→0.374 across evals — with a constant policy this must be episode-mix/eval-window composition, not learning; the earlier reading of it as "mirroring the reward slide" is superseded by the Autopsy's artifact finding.

---

## Q4 — The bimodality verdict, with numbers

**The bimodal *target* is CONFIRMED; the unimodal-averaging *failure of the learner* is UNTESTED this run.**

- **Target bimodality (real, quantified):** binning gw-translate a0 vs Δproj_s and reward over 1.58M mode-classified steps: FREE optimum ≈ **+25–30 mm/s** (dps 1.18 @ [25,30), reward peak [20,25), fold-increment stays 10–15%); CRUNCH optimum ≈ **0 mm/s or gentle retract** (pushes barely move the tip but **fold-increment fires on 61–73% of steps for a0>5**, 65–80% in eval; mean reward negative in every push bin, positive only at [−1,1)). Same action, opposite consequences; the optimal-|a| gap is ~25 mm/s — a single Gaussian head cannot place its mean at both. Composition: explore ≈ 80% crunch / 17% mixed / 2.6% free — the task is crunch-dominated, but the modes are sharply distinct.
- **Caveat (from the report itself):** the bin sweep is correlational (explore-noise randomization mitigates but doesn't eliminate state-selection confounds); the cleanest causal signature is the fold asymmetry — 0% on all retract bins vs 60–80% on crunch pushes.
- **What was NOT shown:** "the policy averages the modes into ~5 mm/s" describes **H0**, not a trained SAC policy — the deployed net was init all run. Whether a trained unimodal tanh-Gaussian residual would collapse onto a compromise is plausible from theory and from prior-run history, but it is **UNVERIFIED by this run**. Do not buy a GMM head on this run's eval evidence (Autopsy's explicit warning).
- **Failure-mode bimodality (behavioral):** supported, but not as the crunch/free split originally stated: mode A = a discrete routing commitment at the RVA ostium that, once wrong, is never retracted (58% of failures, retraction ≈ 0 mm — exactly the critic's untransmitted retract preference from the v2 findings); mode B = near-target grind out of steps (~34%).
- **Cheapest falsification path:** candidate A — if the mode is *observable* (wrong-branch flag, junction context), the per-mode conditional action distribution can be unimodal; only if a well-conditioned, correctly-delivered policy still can't hold both regimes does C (GMM) earn its cost.

---

## Q5 — Does v3b/E4 stuck-restore training help, and where does it fit?

**Yes — it is the best-matched lever to the dominant failure mode, and it slots in at launch, not mid-run.**

- **Match to the data:** 58.3% of failures are RVA captures with median ~505/600 steps inside RVA and **0 mm retraction** — precisely the fading-micro-recovery pathology (v2 fade 86→58→28%; noise-distilled demos; λ=1 unable to transmit the critic's retract preference). E4 restores episodes *into screened stuck states*, making retraction the on-policy path to reward instead of relying on dying demo signal. In the residual regime it is even better targeted: stuck states are exactly where the heuristic grinds, so the residual gets dense practice precisely where it must override.
- **Also the U-turn/routing lever:** the Bimodality report's verdict places the junction trap (44% of timeouts burning ≥100–500 steps) under "E4 stuck-restore + mode-classified conditioning", not speed.
- **Mechanics & cost:** machinery exists (STUCK_CHECKPOINT_DIR capture, screener, mesh-fingerprint restore); the one addition is `--restore_prob 0.3` in `training _scripts/util/checkpoint_restore.py` (directory name contains a space) + launcher (~half-day). Gates: pool ≥300 screened states, ≥8 mesh fingerprints; evals stay ostium-only for comparability; abort rule if ostium-start success drops −5pt.
- **Where it fits:** enable **at relaunch only** — flipping restore on mid-run confounds the start-state distribution. If the machine-2 pool isn't ready (**UNVERIFIED** — pool status not established by these reports), launch F+A with `STUCK_CHECKPOINT_DIR` set (free harvest) and relaunch with D as the immediate follow-up.
- **Success metric:** restore-episode escape rate, and P(retract | slack-tail) finally developing a non-flat gradient (E8 probe). Task-3 caveat from the Autopsy: whether the critic already prefers retraction at high-slack states is unanswerable from this run (no trained critic saved) — D is justified by the v2 findings + this run's behavioral census, not by a critic readout.

---

## Q6 — THE RECOMMENDATION

### Precondition zero: deliverability (without this, every lever is a no-op)

The Autopsy's repair list, promoted to launch-blocking:

| # | Repair | Detail |
|---|---|---|
| R1 | **Deadlock-guard must restore, not re-init** | On trainer restart, restore nets, optimizers, log_alpha, update count from live state (or fix the root deadlock — it recurs every ~40 min; e3bf215 is insufficient; 214 worker timeout restarts besides) |
| R2 | **Fix trainer→agent/worker/eval weight sync** | It never fired once, even in healthy segments (`synchron.py`/`agent.py` `state_dicts_network()` path); checkpoints at update 117k+ are init |
| R3 | **Re-arm probe logging after trainer restarts** | Probe JSONL covered only segment 1 of 32 |
| R4 | **Runner invariant alarms** | Hash policy state_dict at each snapshot, alarm if unchanged N times; alarm on (alpha==1.000 ∧ q1_mean≈0 after a CSV wall-time gap) = init-wipe signature |
| R5 | **Re-audit prior "alpha limit-cycle" runs** | Check for the restart-wipe signature before trusting their conclusions |

**Launch gate:** a 1-hour smoke run must show ≥2 distinct policy hashes across snapshots and worker-buffer action stats moving with trainer mu.

### Design decision matrix (candidates A–G, post-autopsy)

| Cand | What it targets | Cost | Risk | Verdict |
|---|---|---|---|---|
| **F** — α ceiling `--log_alpha_max −1.6` (α≤0.2; −2.3 for exact v2 regime) | Consolidation insurance. *Note: the observed limit-cycle was the restart sawtooth (Autopsy F3), so F's original rationale is weakened — keep it as zero-cost insurance with v2-rails precedent, not as the compounding fix* | config-only | ~0 | **IN** |
| **A** — path-context obs (~6 dims: arclength-to-next-planned-junction, curvature-ahead w/ E6 scaling fix, wrong-branch flag + signed retract-vector to last on-plan point, stuck-duration EMA) | Both failure halves; direct counter to RVA commitment (policy currently has no "you are off-plan; reward is backward" feature; env has `cur_branch`/`nearest_named` internally, none in obs); cheapest bimodality falsifier | ~1 day (obs class mirroring `HeurActionObs`, deployable prefix 125→~131, `DualDeviceNav_train.py` plumbing, stuck-lane flat-index shifts 89/107, monitor). **No harvest cost now** (P2a regenerates heatup; no seed cache) | low | **IN** |
| **D** — E4/v3b stuck-restore `--restore_prob 0.3` | The 58% deep-RVA capture mode head-on (see Q5) | ~half-day (machinery built) | low-med (screener calibration; −5pt ostium abort rule) | **IN if pool ready at launch; else harvest now, enable at next relaunch** |
| **H (added post-audit)** — scripted wrong-branch escape in H0: retract to last on-plan point after N consecutive off-plan steps (env already tracks `cur_branch`/`nearest_named`) | The 58% RVA-capture mode directly, with zero RL | hours (eval-side ablation) | low | **IN as baseline ablation only** — the no-RL cost floor A+D must beat; H0 stays frozen in the training run for residual comparability |
| **G-lite** — eval-side success@900 secondary metric + E8 probes (P(retract\|slack), residual magnitude conditioned on new mode features, per-mode stuck composition per eval) | Diagnosis only; 600 stays the headline cap (comparability with v2/H0) | hours | ~0 | **IN** |
| **E** — reduce-slack potential reward | Same retraction pressure as D but via shaping; reward changes frozen/approval-gated; confounds | half-day | med | **HOLD** (E7-style fallback if D leaves P(retract\|slack) flat) |
| **C** — GMM residual head (K=2–3, argmax-component deploy; then QC-FQL) | Representational bimodality (residual ≈0 "trust H0" vs strongly-negative "override-retract") | 2–3 days (mixture log-prob/entropy in `sac.py`, new α semantics) | med | **v3 BACKUP** — premature before A+D+F are tested on a *delivered* policy |
| **B** — two actors + mode-split buffers | Junction half, if the hard gate is right | 4–5 days (`sac.py`/`GaussianPolicy` single-policy assumption, `synchron.py`/`agent.py` single-state-dict sync, dual α, mode-conditional Q-backup, halved samples/actor) | high | **REJECTED at any stage** — its conditioning benefit is A, its multimodality benefit is C, its cost/risk exceeds both combined |
| **G-full** — raise training cap / speed shaping | Mis-targeted: timeout half is routing waste, not straight-line speed; breaks comparability | — | — | **OUT** |

### The P2a-v2 bundle (one relaunch, ~2 days prep)

1. **R1–R4** infra repairs + smoke-run launch gate (blocking).
2. **F**: launcher flag `--log_alpha_max -1.6`.
3. **A**: new path-context observation class in `eve/eve/observation/` + env5 wiring; deployable prefix (student inherits it); monitor updates.
4. **D**: `--restore_prob 0.3` via `training _scripts/util/checkpoint_restore.py` (directory name contains a space) if machine-2 screened pool ≥300 states / ≥8 fingerprints at launch (**UNVERIFIED** — confirm pool first); in all cases set `STUCK_CHECKPOINT_DIR` in the relaunch for free harvest. Never enable mid-run.
5. **G-lite + E8**: eval-side success@900 secondary metric; probes for P(retract|slack-tail), mode-conditional residual saliency, per-eval U-turn/grind counts. Re-arm probes on trainer restart (R3).
6. **H — H0-escape baseline ablation** (eval-side, hours): scripted retract-to-last-on-plan-point after N wrong-branch steps, using env's internal `cur_branch`. Attacks the same 58% RVA-capture mode with no RL; its converted-seed count is the cost floor the A+D gains must beat. H0 remains frozen inside the training run.

Confounding across F/A/D is acceptable: the run sits at a null result (H0), and each lever has an independent gate metric — alpha trace (F); mode-conditional residual saliency + RVA-capture/U-turn counts (A); restore-episode escape rate (D).

### Expected-gain arithmetic (per 98-episode eval; H0 ≈ 46–48, SE ~5pp)

Baseline accounting: ~46 successes, ~52 failures. Of the failures: ~30 deep-RVA captures (58.3%), ~17 near-target grinds (~34%), remainder mid-course.

| Lever | Cohort | Conversion assumption | Gain (abs. pp) |
|---|---|---|---|
| D + A (retract/routing) | ~30 RVA captures/eval | convert 1/3 → 1/2 (these episodes waste ~505 steps in RVA; escaping early leaves a full budget) | **+10 to +15** |
| Learned free/mixed throttle (5→~25 mm/s, enabled by A's mode observability + delivered learning) | close-call cohort: deficit <20 mm ≈ 20–22% of timeouts (Bimodality: full speedup recovers ~18 mm mean vs median deficit 63 mm) | convert half → all of the cohort | **+5 to +11** |
| U-turn elimination per se | 42 fork-brushes | median cost 4 steps / 0 mm — **worth ≈ 0**; do not budget for it | +0 |
| **Ceiling estimate** | | | **≈ 62–73%** by this table's own rows (46+10+5 = 61/98 = 62.2%; 46+15+11 = 72/98 = 73.5%); quote **60%** only as a floor after a ~2pp cohort-overlap haircut |

Upper end is **UNVERIFIED optimism**: the two cohorts overlap (some RVA captures also carry large deficits), the counterfactual sweep is correlational, and seed determinism means gains must flip whole seeds out of the 44 always-fail set — partial improvements inside a failing seed score zero. The repeat-eval noise floor from the 13 mixed seeds is SE ≈ ±1.8pp (~±4pp at 95%) — much tighter than the naive binomial ±5pp, so the +5pp go-gate sits ≈2.8σ above it.

**Go/no-go gates:** eval ≥ H0+5 (≈52%, ≈2.8σ above the mixed-seed noise floor) by eval2; RVA-capture count and grind fraction falling; P(retract|slack) gradient non-flat. Plus the standing invariant alarms (R4) — an eval that repeats H0 numbers must trigger a hash check before any RL interpretation.

### v3 backup plan

**Trigger:** P2a-v2 ≤ H0+3 at eval3, OR retract probe still flat with delivery verified (hashes distinct, sync confirmed).
**Action:** **C** — GMM residual head (K=2–3, argmax-component-mean deterministic deploy), keeping the entire P2a-v2 stack (F/A/D/G-lite). Mixture log-prob/entropy in `eve_rl/eve_rl/algo/sac.py` + network module; target-entropy semantics change — budget 2–3 days, medium risk. QC-FQL stays P3b behind it (JAX port). **B stays rejected.** If D ran but P(retract|slack) is flat while everything else moved, add **E** (reduce-slack potential) as the shaping fallback — approval-gated.

### Implementation checklist

- [ ] R1: deadlock-guard restart restores trainer state (nets/optimizers/log_alpha/update count) — or root-cause the 40-min deadlock
- [ ] R2: fix trainer→agent/worker/eval weight sync (`synchron.py` / `agent.py` `state_dicts_network()` path); verify a worker-side hash changes after first sync
- [ ] R3: re-arm probe logging after every trainer restart
- [ ] R4: snapshot-hash invariant + init-wipe (α==1.000 ∧ q1≈0 post-gap) alarms in runner/monitor
- [ ] Smoke run (1h): ≥2 distinct policy hashes; buffer action stats track trainer mu — **launch gate**
- [ ] F: `--log_alpha_max -1.6` in launcher
- [ ] A: obs class in `eve/eve/observation/` + env5 (deployable prefix, 125→~131 dims); `DualDeviceNav_train.py` plumbing; stuck-lane flat-index shifts (89/107); monitor
- [ ] D: confirm machine-2 pool (≥300 states, ≥8 fingerprints) — **UNVERIFIED**; `--restore_prob 0.3`; set `STUCK_CHECKPOINT_DIR` regardless; −5pt ostium-start abort rule
- [ ] H: eval-side H0-escape scripted ablation (wrong-branch retract reflex) — record converted-seed count as the no-RL baseline
- [ ] G-lite: eval-side success@900; E8 probes (P(retract|slack), mode-conditional residual saliency, per-eval RVA-capture/U-turn/grind counts)
- [ ] R5: re-audit prior "alpha limit-cycle" runs for the restart-wipe signature
- [ ] Reference docs: `RL_PARADIGM_ROADMAP.md`, `RL_IMPROV_16_EXPERIMENTS.md` (E4/E7/E8), `RL_IMPROV_18_P2_DESIGN.md`, `saved/rl_paradigm_research/*.md`

---

## Appendix — cross-report discrepancies (kept honest)

| Quantity | Forensics (checkpoint-joined, authoritative) | Bimodality (window heuristic) | Resolution |
|---|---|---|---|
| Failures "close" | ratio>0.7 ≈ 34%; d_tgt<10 mm ≈ 4.2% | deficit<10 mm = 20%; <50 mm = 40% | different labels + distance definitions (d_tgt vs path_len−proj_s) and denominators (259 checkpoint failures vs 318 window timeouts); Forensics reconciles with reported SR exactly |
| U-turn share | 16.2% of failures (median 4 steps, 0 mm retraction) | 35% of timeouts | method variance; both agree it is not the step-waster |
| Junction-trap share | 58.3% end deep in RVA, ~505 steps inside | 57% entered RVA; 44% ≥100 steps near RVA (p75=507) | consistent structure |
| Reward decline 0.887→0.556 | (reported per-eval) | — | Autopsy: aggregation artifact; per-episode window means non-monotone (0.64, 0.82, 0.87, 0.59) |
| Alpha limit-cycle | — | — | Autopsy: restart sawtooth, not SAC pathology |
| Eval SR movement 45.9–49.0 | 13/98 marginal seeds | — | Autopsy: episode-mix noise around a constant init policy |

**Bottom line:** P2a-v1 was a null experiment on the RL axis and a definitive census of H0. Fix delivery, make the wrong-branch state observable, train retraction where it matters, and only then judge whether one Gaussian is enough.
