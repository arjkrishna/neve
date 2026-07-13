# RL_IMPROV_16 — Experiment Plan (machine-2 iterations)

Derived from the v2 live-run investigation (`saved/v2_micro_investigation/report_{micro-recoveries,distillation-efficacy,obs-usage-audit}.md`), the v1 collapse forensics (`saved/monitor_rcca_procedural.md`, RL_IMPROV_15_CHANGES.md Part E), and the running v2 monitor passes. v2 (`rcca_procedural_v2`) stays untouched on machine 1 as the control: baseline 6.1% → eval1 30.6% → eval2 **49.0%**, ~44–47% explore, α ceiling-pinned, entropy negative, micro-recovery fading.

**The three findings every experiment traces back to:**

| # | Finding | Evidence anchor |
|---|---|---|
| F-A | **Credit assignment too weak to transmit the critic's retract-when-stuck preference.** Critic has it (AWAC weight 1.042 retract vs 0.947 advance in the buckled-slack tail) but adv std=0.092 with λ=1.0 → weights span [0.72, 1.25] ≈ uniform BC. | distillation report §(C), §implications P3 |
| F-B | **Micro-recovery is noise-distilled and fading** (86%→58%→28% across thirds; post-stall intent hardens +1.8→+7.2 mm/s) as entropy collapses; the held-out gap is a recovery gap (eval failures = stuck-grinders, 69% stall, +13 mm net forward post-stall). Stuck states are gradient-starved (slack tail ~10%, contact tail ~1% of buffer). | micro-recoveries report |
| F-C | **The aux/privileged channel is half-dead and the obs tuple is mis-weighted.** Force labels (rel 2/3/4) identically zero — SOFA `MechanicalObject.force` is a per-solve scratch buffer, cleared before any post-step read (verified: live-wire checkpoint force=0 while velocity alive); contact labels scale-starved (std ~1e-3 → aux grad ≈5e-8); `ep_step` is the **#1 saliency input** for all four actions; ~21 prunable dims; missing catheter-path state, windup, stuck-duration. | distillation + obs-usage reports; force root-cause verified 2026-07-12 |

**Standing rules:** reward/obs/terminal changes are frozen without explicit approval (E7 and E6 are flagged accordingly). Nothing touches `is_on_correct_path()`. Every experiment keeps the v2 fix package (alpha rails, action_mean_penalty, EVE_CLEAN_RAIL_MAX, IPC guard, eval_after_pretrain) unless stated.

**Cache-compatibility legend:** ✅ = reuses `saved/rcca_proc_heatup/seed.npz` (282k transitions) unchanged; ❌ = changes the stored obs/reward → fresh harvest required (`launch_rcca_harvest.sh` variant).

---

## Tier A — v3a launch set (bundle together; all cache-✅; launch first on machine 2)

### E1 — Restore AWAC advantage discrimination *(F-A — the single biggest lever)*

**Change (two arms, pick one as primary):**
- **E1a (flag-only):** `--awac_lambda 1.0 → 0.3`. With adv std 0.092, weights span ≈ exp(±2.5σ/0.3) → p99/p1 ≈ 4.6 — near the low end of the target band.
- **E1b (small code change, preferred):** batch-advantage normalization — in [sac.py](eve_rl/eve_rl/algo/sac.py) AWAC branch, `weight = exp((adv / adv.std().clamp_min(1e-4)) / tau).clamp(max=20)` with new flag `--awac_adv_norm_tau 1.0` (0 = off/legacy). Self-calibrating as the critic sharpens (λ-only arms drift as adv std grows); default off preserves legacy.

**Settings:** τ=1.0 → weight p99/p1 ≈ e^5 ≈ … too hot; use **τ=2.0** (p99/p1 ≈ e^2.5 ≈ 12, inside the 5–20× target band). Monitor `awac_weight_max` (expect 3–8) and a new p99/p1 metric.
**Gates:** GO = weight p99/p1 ∈ [5,20] within 20k updates AND explore success ≥ v2's at matched explore steps. **Success signal = the distillation probe: contact-bin P(retract) gradient reappears and steepens** (was flat; v2 late: Q1 0.408 vs Q5 0.406). ABORT = success −10pts vs v2 at matched steps for 2 consecutive checks (over-sharp weights → cloning too few episodes).
**Risk:** weight concentration → effective batch shrinks; watch `awac_weight_saturation` from above (frac at 20-cap < 1%).

### E2 — Aux distillation repair *(F-C, cache-✅ variant)*

**Changes (all algo-side; the stored obs stay byte-identical):**
1. **Repoint labels off the dead force dims:** `--aux_labels "2,3,5,6" → "0,1,5,6"` (rel 0/1 = tip/max node velocity — alive, std healthy; rel 5/6 = contact proxy — alive but tiny). Dead rel 2/3/4 stop wasting head capacity. *(rel map verified in [meshinvariant.py:145-160](eve/eve/observation/meshinvariant.py#L148))*
2. **Z-score the aux targets at LOSS time** (not in the obs — that would break cache compat): in the sac.py aux block, maintain running mean/std per label over batches; `aux_loss = MSE((pred−μ)/σ, (label−μ)/σ)`. New flag `--aux_label_znorm` (default off). This turns aux_coef·MSE from ~5e-8 into a real ~1–10%-of-policy-loss term at `--aux_coef 0.05`.
3. Log the aux loss into the trainer CSV (currently invisible — it was silently ~0 for the entire v2 run).

**Gates:** aux R² (probe script exists) on contact ≥ 0.55 and **rising** past u≈130k (v2 peaked 0.554 then regressed to 0.486); velocity-label R² ≥ 0.5. NO-GO on interference: policy_loss noise ↑ >2× or explore success dip → halve aux_coef.
**Deferred to E6:** real contact forces via `GenericConstraintSolver.constraintForces` / `computeConstraintForces=True` (scene change; the only way to get true force labels — post-step `dofs.force` reads can never work).

### E3 — Stuckness-balanced sampling lane *(F-B: stuck states gradient-starved)*

**Change:** third sampling lane in [pervanillastep.py](eve_rl/eve_rl/replaybuffer/pervanillastep.py), cloning the clean-lane pattern (SumTree over a flag, drawn fraction of each batch): transition is "stuck" if its stored obs has `gw_slack` (flat 89) > 0.174 (the buckled tail ≈ top 10%) OR `contact_max` (flat 103) in its top decile (threshold from buffer stats ≈ 2.6e-3). Flag `--stuck_fraction 0.15` (0 = off). Compose with `balanced_fraction 0.3`: batch = 30% clean-lane + 15% stuck-lane + 55% general. Mirror the clean-lane correctness note: the stuck flag must be immutable per slot so `update_priorities` can't evict lane membership.
**Gates:** GO = stuck-tail transitions reach ~15% of batches (log lane hit-rate); success = same as E1's behavioral probe (they compound — E1 gives the weights, E3 gives the exposure). ABORT if success drops (over-sampling failure states can bias the critic pessimistic — watch q1_mean slope vs v2's −1.9).

### E4-prep — Stuck-pool harvest during v3a *(F-B; zero-risk data collection)*

**Change:** launcher env `-e STUCK_CHECKPOINT_DIR=/opt/eve_training/results/rcca_v3_stuck` — the capture machinery is already wired in env5 (fires before the 20/50 kill thresholds, per-worker lifetime cap, mesh-fingerprint-tagged filenames). Costs nothing; produces the E4 curriculum pool as a side effect.
**After ~12h:** run `launch_screen_stuck.sh` (escapability + restore-fidelity screener, already built) → screened pool for E4.

### E8 — Monitoring upgrades (ship with v3a)

Add to `monitor_pass.sh` / probe scripts: (1) **P(retract | slack-tail bin)** from the latest snapshot on fixed buffer states — the direct micro-recovery-learning metric (v2 reference: 0.448 tail vs 0.416 base and *shrinking*); (2) aux R² per label; (3) awac weight p99/p1; (4) stuck-lane hit-rate. Keep the freeze probe + alpha band + rail-filter checks as-is.

**v3a launcher delta vs `launch_rcca_procedural_v2.sh`:**
```
--awac_adv_norm_tau 2.0            # E1b (or: --awac_lambda 0.3 for E1a)
--aux_labels "0,1,5,6"             # E2.1
--aux_label_znorm                  # E2.2
--stuck_fraction 0.15              # E3
-e STUCK_CHECKPOINT_DIR=...        # E4-prep
-n rcca_procedural_v3a
```
Everything else identical (seed.npz + 10k pretrain + eval_after_pretrain baseline). Files touched: sac.py (E1b, E2.2, E2.3), pervanillastep.py (E3), DualDeviceNav_train.py + agent.py (flags), launcher. All default-off → v2 behavior reproducible byte-for-byte with flags absent.

---

## Tier B — v3b: recovery curriculum (after v3a produces the screened pool)

### E4 — Stuck-restore recovery curriculum *(F-B, the direct attack)*

**Change:** relaunch v3a settings + `--checkpoint_dir <screened stuck pool> --rl_start_mode sofa_restore`. Mesh-matched restore is built and safe (fingerprint pin → `regenerate_to_fingerprint` before SOFA restore; untagged checkpoints fall through to ostium start). **One required addition:** the wrapper currently restores on **every** reset ([checkpoint_restore.py:3](training%20_scripts/util/checkpoint_restore.py#L3)) — add `--restore_prob 0.3` (Bernoulli per reset; else ostium start) so the run keeps learning full navigation while drilling escapes.
**Why this beats waiting for noise:** starting *in* screened-escapable stuck states makes retraction the on-policy path to reward — micro-recovery gets direct credit instead of relying on the dying exploration-noise demo source.
**Gates:** SUCCESS = micro-recovery cycle prevalence in successes stops fading (flat or rising across run thirds; v2: 86→58→28%), post-stall retract latency < 10 steps (v2 eval: ~20), eval stuck-grinder fraction falls, held-out eval ≥ v3a. ABORT = restore-start episodes < 50% escapable in practice (screener miscalibrated) or ostium-start success regresses >5pts.
**Settings:** pool ≥ 300 screened states across ≥ 8 mesh fingerprints; `--restore_prob 0.3`; keep `--eval_after_pretrain` and evals on ostium starts only (eval comparability).

### E7 — Escape bonus (approval-gated fallback; only if E1+E3+E4 leave micro-recovery fading)

**Change (REWARD — needs explicit approval):** one-shot **pay-on-escape** bonus: +0.3 the first time `proj_s` exceeds its pre-stall max after a stall (proj_s flat ≥10 steps AND buckle ≤ −0.05) *that contained executed retraction* (net di0 ≤ −1.5mm). Latched per stall event, cap 3/episode. Paying on escape (not on retraction) makes it unfarmable — jitter can't collect it without real net progress past the stall point; the stall precondition keeps it off free-flowing segments.
**Gates:** reward-audit first (steplog scan: bonus fires on <5% of steps, only within stall contexts); then same success metrics as E4.

---

## Tier C — v4: observation surgery *(F-C; cache-❌ — fresh harvest; biggest scope)*

### E6 — Obs v5.1: prune / add / de-crutch

**Prune (~21 dims):** frame t-1 body offsets (r=0.98–0.99 dups of frame t; keep the two t-1 tip rows or replace with a t-5 delta for real shape-velocity), `in_wrong_branch` (73, exact −1.0 dup of on_path), `d_rem_log` (96, ignored; r=0.92 with d_rem).
**Add (~8 dims):** catheter along-path projection gap (s_gw − s_cath along the planned path — not the insertion gap) + catheter cross-track/clearance *(P4-O3 — closes "catheter steered semi-blind")*; sin/cos of **cumulative commanded rotation** per device (deployable windup proxy — the privileged windup dims carry std 0.71 the policy can't see, with two rotation actions); stuck-duration integrator (EMA of slip over ~10 steps + steps-since-tip-progress — slip is 1-step, fold_stall counter is thresholded/variance-starved); signed cross-track (path-normal side).
**Fix scalings/bugs:** `curv_ahead` /10 → clip 0.5 (std currently 0.011); radius /12 → /6 (range used 0.17–0.55); target-dz clip (86% saturated at 50mm); `at_ostium` dead-at-source on procedural meshes; `br_trunk` never fires (`classify_physical_branch` mis-bins trunk as bridge — also fixes the critic's branch one-hot).
**De-crutch (highest-leverage behavioral change):** remove `ep_step` from the policy prefix (keep it critic-side in the privileged tail) — it is the #1 saliency input for every action head and feeds the determinism drift; add **executed** last-action alongside commanded (the mask flags 91/92 are alive but only cover masking, not magnitude).
**Optionally with scene change:** `computeConstraintForces=True` → real contact-force labels for aux (supersedes E2's repoint).

**Consequences:** obs width changes → new harvest (~3h), new pretrain, all cache guards will correctly fail-fast on old seeds. Bundle as ONE breaking change-set (the multimesh report's §5 rule). Carry every Tier-A/B win into the v4 launcher.
**Gates:** saliency re-audit at ~150k updates — `ep_step` gone from top-10 (it's absent from the prefix), no new dim dead; eval trajectory ≥ v3's at matched explore.

---

## Tier D — held / conditional

### E5 — Entropy-regime A/B *(the current v2 WATCH; adjudicated by v2's eval3)*
If v2's eval3 (~explore 760k) **regresses** (deterministic-aggressive regime hits the wall): arm E5 in v3a — either `--log_alpha_max −2.3 → −1.6` (α_max 0.20) or per-dim entropy floor via the existing `--entropy_beta_per_dim` scaffold on a0 only (e.g. `0.01,0,0,0`), keeping exploration alive specifically on gw translation (the retract dimension). If eval3 holds ≥ 49%: leave the regime alone — aggression is paying and E4 supplies recovery pressure instead.

### E9 — Authored-mesh drop-in (later)
`env_train_factory` is generic: swap procedural RCCA variation for 3 authored VMR meshes + 1 held-out when true arch-level diversity is wanted. No code change beyond a factory; defer until RCCA-line experiments conclude.

---

## Sequencing & allocation

| Order | Run | Machine | Experiments | Cache | Decision point |
|---|---|---|---|---|---|
| 0 | rcca_procedural_v2 (control) | 1 | — | — | eval3 @760k adjudicates E5 |
| 1 | **rcca_procedural_v3a** | 2 | E1b + E2 + E3 + E4-prep + E8 (+E5 iff v2-eval3 regressed) | ✅ seed reuse | eval2 (~510k): ≥49%? behavioral probe moving? |
| 2 | screen pool | 2 (offline) | E4-prep screening | — | ≥300 escapable states? |
| 3 | **rcca_procedural_v3b** | 2 | v3a + E4 (`--restore_prob 0.3`) | ✅ | micro-recovery fade stopped? if not → request E7 approval |
| 4 | **rcca_procedural_v4** | 2 | E6 (+ constraintForces) + all wins | ❌ fresh harvest | saliency re-audit + eval ≥ v3 |

**Comparability rules:** identical eval protocol everywhere (98 fixed seeds, held-out RCCA, ostium starts); `--eval_after_pretrain` baseline in every run; one Tier per launch (attribution); v2's per-eval numbers at matched explore steps are the control curve.

**Effort estimate:** v3a = ~1 day code (E1b/E2/E3 are each <50-line diffs in files already mounted) + tests + launch. v3b = wrapper `--restore_prob` + screening (half-day). v4 = the big one (~2–3 days obs work + re-harvest).
