# Forensic: rcca_procedural_v3a (run 2) — why pretrain worked, online plateaued, and where the gains actually are

**Date:** 2026-07-18 · **Run:** `2026-07-16_230731_rcca_procedural_v3a` (still running at analysis time, explore ~1.5M, u~750k)
**Method:** 7 parallel read-only investigators over losses CSV (657k rows), 1.4 GB per-step worker logs (686 eval episodes + explore), replay chunks on disk (1.54M transitions), checkpoint critics loaded in-container, and the code. Findings cross-validated where 2-3 agents derived the same quantity independently, then an adversarial verification pass (11 CONFIRMED / 3 CORRECTED) plus a direct in-container policy-snapshot measurement to settle the one load-bearing correction.

> ### ⚠️ Verification corrections (read first — one reverses a mechanism)
> 1. **7 eval passes, not 6** (686 episodes). The investigators cut at 588 = 6×98 and missed eval #7. Per-pass success = **49/49/49/50/50/51/51**. Conclusions unchanged (plateau extends by one flat point; the 3 flappers are non-durable: seed156 solved eval#6 then lost at #7).
> 2. **Exploration did NOT collapse — this reverses the "re-open entropy" advice below.** Directly measured on 5 checkpoints: the policy's `log_std` sits at its **ceiling** (σ_pretanh ≈ 0.96-1.0, 85-89% of states; 0% at the floor) the entire run. The negative `entropy_proxy` (→ −7.4) is a tanh-Jacobian artifact of action saturation, not a narrow Gaussian — two investigators misread it. **The real driver is state-specific MEAN-rail:** at the last checkpoint `|tanh(μ)|>0.99` on **86% of contact states (catheter 96%, guidewire 79%) but 0% of free/seed states**, growing from 0% everywhere at u=10k. The policy learned a saturated bang-bang telescoping gait *as its contact-mode response*. **Implication: E5 entropy/log-std-floor flags will do nothing** (stochasticity is already maxed); the lever is an anti-mean-rail regularizer that bites on contact states + a directed update so the critic's retract preference can move the contact-state mean (see revised Q7).
> 3. **The "full-forward vs full-retract action split" was mischaracterized.** It is one saturated *telescoping* gait: catheter pinned full-forward (97-100% of online-eval steps) + guidewire full-retract (rising 3%→90%), `delta_ins=[0,0]` (total slip) → deadlock, never backing the catheter out. Not two competing modes. `clamp_fraction` is a trainer replay-batch metric (97%-stuck buffer), not an eval action stat — don't cite it for eval behavior.

---

## TL;DR

1. **Pretrain worked because E1b (advantage-normalization) gave AWAC a _state-conditional, mode-split_ credit signal** — it up-weights *retract* 1.40× over push in high-slack states and *push* in free lumen — distilling the 5.6%-success seed buffer into a policy that solves a **fixed 49/98-seed subset**. v2's degenerate weights (p99/p1 = 1.8, uniform BC) could not do this; E1b's are p99/p1 = 21.9 on the *same* critic/data.
2. **Online learning added essentially nothing (49→51 seeds over 1.26M steps) because it is a reachability ceiling, not a learning failure.** All 46 never-solved seeds wedge at the **same fixed arc position** (~158 mm, `local_r = 2.0 mm`, the tightest point on the RCCA centerline). Success is a **razor-sharp function of planned-path length**: 100% solved for path ≤170 mm, 0% for >180 mm, *zero overlap*. The "+2" gains are 3 boundary seeds (path 170.6-171.7 mm) popping through stochastically.
3. **No soft-recovery behavior was ever learned by the deterministic policy** — soft-recovery events across the 6 evals were `0,0,0,1,0,1` (the two "softs" are threshold artifacts). The explore-side fade (6.9%→0%) was never a *loss*; eval recovery was at zero from the start. Noise-generated recoveries never transferred into the policy.
4. **The two-modality hypothesis is half right, and the correction matters.** The environment *is* mechanically bimodal (guidewire transmission is binary: 97.5% full / 2.5% blocked, nothing between) and the policy *did* collapse to one gait **on contact states specifically** — measured mean-rail 0%→86% on contact states vs 0% on free states over training; the fine contact response pretrain had was **actively unlearned into a saturated telescoping deadlock**. **But the binding constraint is not "one net can't hold two modes"** — the same net is *still* well-behaved (unsaturated) in free lumen, mode is already linearly readable from the existing 97 policy obs dims, and **there is zero successful training data anywhere past the choke.** The problem is **data frontier + a directed-update/anti-rail training pathology**, not representation capacity or too little exploration noise (σ is at its ceiling).
5. **E-attribution:** E1b = the real driver (confirmed mechanism). E2 (aux) = real but small (~3% of policy loss; was exactly zero in v2). **E3 (stuck lane) = functional no-op** — its stuck definition flags 75-98% of the buffer, so reserving 15% for it changes nothing. E4 (harvest) = a good, uncontaminated pool (79 fingerprints, 0 eval-mesh leakage).
6. **Correction to the handoff:** the seed buffer contains **zero demos** (`meta_is_demo` all-false, 0/480), not "heuristic demos." Pretrain distilled 5.6%-success *exploration* data, not demonstrations. This reframes the whole warm-start story and matters for v3b.

---

## Q1 — Why was pretrain so effective and online learning nearly inert?

**Pretrain (6.1% seed-success → 36.7% run1 / 50.0% run2 held-out):** AWAC is advantage-weighted regression toward buffer actions. The mechanism that makes it work is **E1b's weight dispersion** (Q7-F1): on 20,480 seed transitions scored by run 2's post-pretrain critic, the τ=2 batch-normalized weights span p1=0.147 to p99=3.22 (**p99/p1 = 21.9**, ESS 66%). The v2-equivalent λ=1 raw weighting on the *same* critic reproduces v2's degenerate band `[0.69, 1.24]`, p99/p1 = 1.8 (uniform BC). The selectivity is **not** episode-level "favor winners" (corr(w, success) = 0.046, near zero) — it is **per-state, mode-split action ranking** (Q7-F3):

| state class (of seed buffer) | corr(advantage, gw-translation) | E1b W(retract)/W(push) | v2-equiv |
|---|---|---|---|
| free lumen (n=5149) | **+0.118** (favors push) | — | — |
| high-slack >0.174 (n=1955, 9.5%) | **−0.533** (favors retract) | **1.40×** | 1.07× |
| contact-flagged mass (n=15331, 75%) | +0.011 (no signal) | ~1.0 | ~1.0 |

That −0.533 in high-slack states is the concrete source of the pretrain policy's +74pp retract-vs-slack coupling and the 6.1%→50% jump. **Note the third row:** in the 75%+ of the buffer that is contact-flagged, the advantage has *no* learnable action ranking — foreshadowing why online learning stalls.

**Online (49→51 solved over 1.26M steps) is inert because of two compounding facts:**

- **The deterministic policy railed its mean on contact states** (corrected — see banner): σ is at its *ceiling* (~1.0), so exploration is wide, but `|tanh(μ)|>0.99` on 86% of contact states means the *deployed* (deterministic) action is bang-bang there. AWAC is advantage-weighted BC of buffer actions with **no Q-ascent term**, so it can only *sharpen in-support behavior*; it cannot synthesize out-of-support solutions, and its near-uniform weights (mean 1.05) can't pull the railed contact-mean toward the critic's retract preference.
- **There are zero successful trajectories past the choke to regress toward** (Q5, Q6-F8). Online explore data is 92-97% failed 600-step contact-grinding episodes; the buffer's terminal-success fraction is high only on the *already-solved* anatomies.

The policy **moved continuously in parameter space** (rel L2 vs pretrain 1.1×→6.0×, still climbing) but its **behavior on fixed states saturated at u~400k** (mean|ΔA| plateaus at 0.13) (Q4-F2). Movement without gain — the entire marginal gradient went into *speeding up the solved 49* (mean steps-to-success 123→33) plus behavior-cloning onto retract-grinding on the unsolved 46.

---

## Q2 — The plateau is a reachability ceiling at one fixed choke (per-seed map)

From 588 eval episodes (6 windows × 98 seeds), success counts `49/49/49/50/50/51` match the run CSV exactly. The structure (Q5, Q6, independently in Q3/Q4):

- **49 always-solved, 46 never-solved, 3 flappers** (seeds 131/141/156). "Solved sets are perfectly nested" — nothing is ever *lost*, only 3 boundary seeds ever *flicker in*.
- **Success = f(path length), a clean cliff:** P(ever-solved) = 1.00 for path ≤170 mm, 0.75 in (170,180], **0.00 for >180 mm**. Always-solved median path 124 mm; never-solved median 230 mm. *Zero overlap.* The flappers sit exactly on the boundary (170.6-171.7 mm).
- **One fixed anatomical choke:** every never-solved seed wedges at **max-penetration 158.0 ± 0.4 mm arc** (per-seed sd across 6 evals 0.1-0.4 mm), where `local_r = 2.0 mm` (route minimum: 6.0→2.7→2.5→2.3→2.0), tip z≈510-514, on `Centerline curve - RCCA.mrk`. Logs never name the feature (`nearest_named=?`), so "RCCA/RVA intersection" can't be confirmed *by name* — but it is unambiguously **a single shared choke**. (Q6 places it at 55-90% of arc in the narrow distal RCCA segment; RVA wrong-branch events sit earlier at frac 0.15-0.41, so it is not a branch mis-selection.)
- **Failure phenotype is uniformly fast-then-wedged**, never slow-crawl, never wrong-way: reaches the choke by step ~48/600 at the physics cap, then grinds 550 steps. Navigation is always correct (`on_path=1`, `entries_passed=1`, `overshoot=never`).

---

## Q3 — Was any soft-recovery behavior learned? No.

Measured on the **deterministic eval** episodes (ground truth, not noise-contaminated explore):

| window | succ% | stuck events | soft | hard | grind | unrec% | rail-step% | fine-step% |
|---|---|---|---|---|---|---|---|---|
| W0 (pretrain) | 50.0 | 35 | 0 | 0 | 35 | 0.0 | 0 | 87 |
| W1 | 50.0 | 8 | 0 | 0 | 0 | 100.0 | 4 | 6 |
| W2 | 50.0 | 43 | 0 | 1 | 0 | 97.7 | 10 | 3 |
| W3 | 51.0 | 32 | 1 | 1 | 2 | 87.5 | 71 | 1 |
| W4 | 51.0 | 20 | 0 | 2 | 1 | 85.0 | 86 | 1 |
| W5 | 52.0 | 26 | 1 | 2 | 1 | 84.6 | 86 | 0 |

**Soft-recovery trend `0,0,0,1,0,1` — flat zero** (both "softs" are pump-amplitude threshold artifacts). Key mechanistic findings (Q3):
- Retracts are **not error-triggered recoveries** — onsets follow *positive* progress (+0.80 mm median 3-step Δproj) with ~zero preceding slip. It's **periodic pumping, not reactive recovery**.
- The policy underwent a **complete regime change**: fine steps (|a0|<2) 87%→0%, rail steps (|a0|>28) 0%→86%. The fine/contact mode wasn't just never learned — **it was actively lost.**
- The +2 newly-solved seeds succeed via a **brute-force "pressure-cooker" pop**: pump both devices at rail until a buckled wire stores enough slack to pop the tip +8 mm through the bend (fold counter to 48; inserted [345 mm gw, 560 mm cath] on a 170 mm path), then a couple catheter retracts finish. After the pop, control authority is gone (huge slack), so targets >8 mm past the choke stay unreachable. This is *why only boundary seeds ever succeed.*
- **Blind spot in the current tooling & proposals: propulsion is catheter-led.** In successes the guidewire is parked (<2 mm) ~90% of steps at pretrain (45-54% online) while the catheter carries progress. Failures feed the catheter to its **hard 898 mm insertion cap** (~5× the path length, pure slack) while the guidewire never deploys.

---

## Q4 — What did each E contribute?

| E | Verdict | Evidence |
|---|---|---|
| **E1b** adv-norm | **The real driver.** Restored weight dispersion (p99/p1 1.8→21.9) and mode-split action ranking (retract 1.40× in slack, push in free, flat in contact). Best-supported cause of the pretrain jump. Signal decays online (high-slack corr −0.533→−0.376). | Q7-F1/F3 |
| **E2** aux (znorm, repointed) | **Real but small.** ~3.3% of pretrain policy-loss magnitude (~0.8% online); heads did learn (aux 5.4→0.34, final R² 0.92/0.82/0.85/0.50). v2's aux was *exactly zero* gradient, so any shaping is new — but inseparable from E1b for the jump. | Q7-F4 |
| **E3** stuck lane | **Functional no-op.** The stuck definition (`slack>0.174 OR contact>0.0026`) flags **74.7% of the seed buffer and 92-98% of every online chunk** — driven by the contact threshold (0.0026 < seed contact *mean* 0.0065). Reserving 15% of a batch for a pool that's already 75-98% of the buffer changes composition by ~nothing. Handoff expected ~10% stuck share. | Q7-F5 |
| **E4** harvest | **Good pool.** 354 pairs (run2) + 107 (run1); 79 unique mesh fingerprints; **zero eval-mesh (s12344) contamination.** ⚠️ 84% of captures have *negative* `slack_at_capture` (median −63.7 mm) — a convention mismatch vs obs dim 89; verify before building v3b restore logic on slack assumptions. | Q7-F7 |

Also settled: the **q1_mean drift (−0.24→−1.64) is not E3 pessimism** — it's non-monotone (rose to +0.26 at u=25k), co-moves with policy peaking (q1m ≈ 0.23·entropy − 0.28), target tracks it pointwise, and it's *shallower* than v2's −1.9 reference. And **it is policy-irrelevant anyway**: under E1b the AWAC weight `exp(((q_buf−min_q)/adv_std)/τ)` is invariant to per-batch affine Q drift (Q6-F5) — only within-batch action *ranking* reaches the policy.

---

## Q5 — Two-modality hypothesis: half right; the fix it implies is different

**FOR the hypothesis:**
- The environment is **mechanically binary**: commanded-vs-achieved guidewire ratio is 0.9-1.1 (97.5% of pushes) or ~0 (2.5%), with **literally zero mass between** (Q6-F1).
- The policy collapsed to **one state-independent bang-bang mode**: by W4-5 it commands |cmd|>29 in 99.4% of moving steps in *every* vessel-radius bin (Q6-F2). Pretrain modulated by tightness (0.33 tight / 1.70 mid); online commands 29.9 everywhere, *even while wedged*.
- Online training **traded penetration depth for transit speed** (Q5-F6): the pretrain policy ran at 1/3 amplitude, used 13.5 mm of guidewire in failures, and **penetrated 17 mm deeper** (175 mm) than the online policy ever reaches (158 mm). Same 49-seed solved set, but the fine mode was there and was lost.

**AGAINST "one net can't encode both modes" (why architecture is not the first lever):**
- The **same network** expressed graded, radius-modulated control at pretrain with equal 50% success (Q6-F9-i).
- **Mode is already linearly readable** from the 97 policy obs dims: slip (90), slack (89), gw/cath action-masked flags (91/92 — literally the no-op bits the failing policy sits in), local_r (93), radius_ahead (94), clearance (95), path previews at 10/20/40/80 mm. Proposal (a)'s explicit mode flag adds "at most denoising over ~4 existing dims" (Q6-F7, Q1-F5).
- **The deep-arc "slow fine mode" has zero training data anywhere.** Seed buffer: 0 demos, 5.6% success, no successful traversal past arc ~170 mm. A second "crunch actor" trained on this data inherits the same zero (Q6-F8).
- The failure boundary is a **data/skill frontier at fixed arc depth**, not interleaved mode interference (Q5, Q6-F5).

**Verdict (Q6-F9):** the evidence favors **training-dynamics + data-frontier fixes** over a two-actor architecture. The refinement to the user's model: the missing "mode 2" is not merely *slow fine control* — it's a **device-strategy switch** (stop feeding catheter slack, deploy/rotate the guidewire *through* the r=2 mm bend). The catheter-slack-feeding to the 898 mm cap is the learned anti-pattern.

---

## Q6 — Health & what continuing the run buys

- **Stable degenerate fixed point**, not a divergence path: 0 nonfinite in 657k updates, twin critics + target in lockstep (mean|q1−q2|=0.014), q-losses flat 0.004-0.006, no ALERT/NaN/Traceback in main.log. The AWAC critic target is plain Bellman (no entropy term), so entropy collapse *structurally cannot* feed the critic — the historical −406k divergence mechanism is removed.
- **`clamp_fraction` is a saturation monitor, not a clamp** (fraction of the policy's *own* fresh tanh samples with |a|>0.99). No dead-zone exists; log_std uses a soft tanh rescale with nonzero gradient everywhere.
- **Continuing +1M steps predicts +1-2 seeds (~53-54%)** via the same speed-polish mechanism — no mechanism to cross the choke (explore recoveries already ~0).
- **⚠️ Imminent structural risk — demo/seed eviction:** the replay buffer is a plain FIFO ring with **no demo protection**. The 282k seed transitions occupy slots 0-282309 and are overwritten *first* once the ring wraps at the 2M cap (explore ~1.72M — roughly one more eval). By explore ~2.0M all seed data is gone, and AWAC then behavior-clones only 45%-success self-data with ~zero recoveries → **plateau hardens with real regression risk.** This is the strongest argument for stopping the run rather than letting it drift.

---

## Q7 — Paths forward, ranked

Ranked by (leverage on the never-solved set) × (1/risk): **A > D > C > E > B.**

| Opt | What | Size | Reuse | Verdict |
|---|---|---|---|---|
| **A. v3b restore curriculum** | Start explore episodes *in* screened stuck states so escape gets on-policy credit. Needs the `--restore_prob` Bernoulli gate (~12 LoC in `checkpoint_restore.py`) + arg wiring, **and gating OFF an eval-wrap trap** (`DualDeviceNav_train.py:597-603` wraps `env_eval` unconditionally when `--checkpoint_dir` is set → would start evals from stuck states *and re-mesh the held-out anatomy*; ~4-8 lines). **~40 LoC total.** | small | seed + 1.54M buffer + obs layout unchanged; evals stay ostium-only | **Do first.** Cheapest, directly attacks the escape-data starvation. |
| **D. A + E7 reward** | Pay-on-escape reward term. Adds credit where v3b adds exposure. | +~60 LoC | as A | **Human-approval-gated** (reward change). Hold until v3b gates fail. |
| **C. Two actors** | Mode-classify from existing obs dims; dispatch in `sac.py` action fns; update = mode-masked per-sample AWAC loss ×2 optimizers. **Implementable with NO obs/reward/terminal change** (frozen-rule compliant) and full buffer reuse — two buffers *not* needed. | ~350-500 LoC, 5-7 files | full | The explicit modality test — but forensic says data, not architecture, is binding. **Hold** as the architecture experiment *if* v3b(+E7) leaves the choke untouched. |
| **E. Hierarchical option** | Recovery sub-policy on stuck trigger. | ≥C | — | **Reject** — no HRL machinery exists; its cheap approximations *are* A and C. |
| **B. Obs mode-flag** | Explicit mode dim(s) from planned path. | ~60-100 + cache break | none (obs width changes) | **Dominated** — info already in obs; cache-breaking + approval-gated. Fold into v4. |

### The critical caveat for v3b (from Q6-F9 + the mean-rail measurement)

**v3b restore alone may not be sufficient, for the reason E3 already failed:** E3's stuck lane trained on choke states for 1.26M steps and produced *zero* recovery — "starting states without a reachable success signal don't help." At the choke, `d_tgt` median is 62.8 mm — the reward gradient is too sparse. And the mechanism measurement shows *why exposure alone won't fix it*: the contact-state **mean is railed** (86%) and AWAC's near-uniform weights (mean 1.05) with **no Q-ascent term** (sac.py:575 vs SAC's :581) cannot pull that mean off the rail toward the critic's known retract preference. So v3b should ship **with** at least one directed lever:
1. **Anti-mean-rail regularization on contact states** — the two anti-rail regularizers are effectively off (`entropy_beta_per_dim = null`; `action_mean_penalty = 0.005`, too weak). Raising the mean-margin penalty (it bites hardest exactly where the mean is large = contact states, and the free-mode fast commands are unsaturated so they're barely touched) or re-enabling the per-dim regularizer directly attacks the measured 86% rail. **Do NOT reach for E5 entropy/log-std flags — σ is already at its ceiling; more noise does nothing.**
2. **A directed policy-improvement signal on the hard class** — either seed retract-recovery demos into the buffer (there are currently *zero* demos, correcting the handoff), or add a Q-ascent term so the update can move the mean toward the critic's retract preference instead of pure advantage-weighted BC of the railed buffer actions.
3. **Intermediate targets** on the restore curriculum (a waypoint a few mm past the choke, not the full 62 mm) so escape earns dense reward — *reward/target changes are frozen and need approval.*
4. The **pop-through insight**: past the choke the only observed success is slack-buckle pop, after which control is lost — so the deep targets (>180 mm) likely need a *fundamentally different* guidewire-lead strategy that no restart-at-choke will surface without demos of that strategy or a reward that favors guidewire penetration over catheter slack-feeding.

This measurement also **strengthens the two-actor case (option C) relative to the original ranking**: the single shared mean-head had to represent both "full-speed free lumen" and "gentle contact," and under BC pressure from 97%-contact-grind data it collapsed the *contact* response to a rail while keeping free clean — a representational interference a mode-split actor (each with its own mean target and lane weighting) would isolate. It remains gated on the same data problem (a contact actor still needs good recoveries to regress toward), so the sequence below is unchanged, but C moves from "only if v3b fails" toward "the principled fix if the anti-rail + directed-update levers on a single actor don't hold."

### Recommended sequence

1. **Stop this run** (plateaued; seed-eviction risk imminent at explore ~1.72M). Everything valuable is checkpointed.
2. **Screen the 686+72 stuck pool** (`launch_screen_stuck.sh`; gate ≥300 escapable across ≥8 fingerprints) — and first **resolve the negative-`slack_at_capture` convention** (Q7-F7) before trusting restore-fidelity filters.
3. **Implement `--restore_prob` + the eval-wrap gate** (~40 LoC) + a unit test of the Bernoulli/eval bypass.
4. **Launch v3b fresh** (seed + 10k pretrain + `--eval_after_pretrain`, evals ostium-only) — *paired with an anti-mean-rail lever* (raise `action_mean_penalty` and/or re-enable the per-dim mean-margin regularizer), **not** entropy flags. Fresh beats `--resume` for comparability. One-knob-per-distribution per handoff §6, so if both the restore curriculum and the anti-rail penalty are new, run them as a deliberate 2-change bundle with per-component metrics (rail% on contact states from the snapshot probe; the never-solved seed set).
5. **Gate v3b at eval2** on the micro-recovery fade metric + the fixed never-solved set. If the choke still doesn't move → request **E7** approval (option D). If it moves for near-choke seeds but the deep (>180 mm) targets stay dead → that's the genuine two-mode/device-strategy wall, and **option C (two actors) + a guidewire-penetration reward** becomes the justified experiment.

**Bottom line for the RCCA/RVA goal:** the prize (46 unreachable seeds, all beyond one r=2 mm choke) is real and large, but it is gated by **missing deep-arc success data and a directed-update/anti-rail training pathology** — *not* by single-policy capacity and *not* by too little exploration noise (σ is at its ceiling; the mean is railed on contact states). Fix the data (restore curriculum + demos/denser near-choke reward) and the update (anti-mean-rail penalty + a directed/Q-ascent or mode-split signal so the critic's retract preference can move the contact-state mean) together; reach for two actors when a single actor with those levers still can't hold the fine guidewire-lead strategy through the bend — a case the mean-rail-on-contact measurement makes more likely than the original ranking implied.
