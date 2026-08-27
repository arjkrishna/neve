# PAPER_PLAN_NEUROVASC_RL.md

**Working title (candidates):**
- *Learning to Recover: Privileged-Actor RL for Dual-Device Endovascular Navigation to the Carotid Siphon under Procedural Anatomy*
- *Diversity, Not Competence: Residual-on-Heuristic RL for Buckling-Aware Dual-Device Navigation*

**Status:** war-room planning doc. Premise (hypothetical): the experiment line reaches **>85%** success under the harder-than-published protocol. All hypothetical result numbers are marked **[X%]** / **[X]**. All *observed* numbers below are from in-session verified reports and are safe to cite internally.

**Submission gate (do not violate):** (i) no headline ships until every **[X%]** is a *measured* number carrying a **95% CI**; (ii) the headline success must exceed both our own observed **49%** and the positioning bars we cite against (stEVE 40%, HM-MARL 56–80%) — if it lands between them, re-scope the claim, don't submit it as-is; (iii) the ~0%-deterministic / ~54%-noise heuristic must be defended as a *fair* (non-strawman) baseline — it is the same scripted controller the policy rides as a residual, evaluated identically, so it is the honest null, not a weakened foil.

**Target venue reality check (from lit scan §7):** ICRA/IROS = 6 pp, RA-L = 8 pp, IJCARS/CBM = full length. A *true* 4-page limit fits only workshop / extended-abstract tracks (Hamlyn, ICRA workshops). **Recommendation: write to a 4-page workshop/extended-abstract skeleton but budget content so it upgrades cleanly to a 6-page RA-L/IROS submission.** This doc plans the superset.

---

## TABLE OF CONTENTS

1. [POSITIONING — what to impress](#1-positioning)
2. [THE CHAIN OF TECHNICAL CRUXES](#2-chain-of-cruxes)
3. [EXPERIMENT DESIGN STORY](#3-experiment-design-story)
4. [IMPLEMENTATION CHALLENGES (the 7.5 Hz wall)](#4-implementation-challenges)
5. [FIGURES (6–10 specs)](#5-figures)
6. [SAMPLE ABSTRACT](#6-sample-abstract)
7. [SAMPLE INTRODUCTION](#7-sample-introduction)
8. [METHODS SECTION SKELETON](#8-methods-skeleton)
9. [RESULTS-SECTION PLAN](#9-results-plan)
10. [RELATED-WORK MAP](#10-related-work-map)
11. [RISKS / REVIEWER ATTACKS](#11-risks)

---

<a name="1-positioning"></a>
## 1. POSITIONING — WHAT TO IMPRESS

### 1.1 The honest landscape (SOTA has moved — do not use the stale "we're at frontier" framing)

The in-session premise ("we're at domain SOTA, 49% vs field 40–59%; nobody does siphon dual-device") is **outdated and partly contestable**. Verified current bars:

| System | Task | Success | Anatomy | Device | Notes |
|---|---|---|---|---|---|
| stEVE **DualDeviceNav** (Karstensen et al., arXiv:2410.01956, CBM 2025) | arch→supra-aortic | **40/100** | 1 fixed route | dual (4-dim) | the exact framework/benchmark we run; 2D obs, no privileged critic, no recovery |
| stEVE ArchVariety | arch | 90/100 sim, 84 real | *generated* type-I arches | single | the anatomy-generalization precedent |
| SAC-EIL-GAIL (Jianu et al., arXiv:2512.18081) | **single** renal bifurcation phantom | **59%** | 1 phantom | single | shallow, single-bifurcation — NOT deep/dual |
| SplineFormer (arXiv:2501.04515, IROS'25) | brachiocephalic cannulation | 50% | phantom | single | **VERIFY before citing:** a GAIL-PPO baseline is reported at 69.4% (i.e. above 50%) — confirm task/metric parity or drop; likely a different sub-metric, not the same success criterion |
| Barnes et al. soft-robot IL (arXiv:2510.09497) | aneurysm, unseen geom | **83%** | 36 geometries | single soft | **only prior work with explicit recovery** — but IL demo-injected, single-device |
| **Robertshaw IRL dual** (arXiv:2406.12499 → **IJCARS 2025**, PMC12167253) | ICA→**M1 through siphon** | **96%** | **12 fixed patient** | **dual** | SAC+LSTM+IRL; 0.24 N mean force; the SOTA to beat; no procedural anatomy, no learned recovery. *This is a single paper — the arXiv preprint (2024) and the IJCARS version (2025) are the same 96% result; §10 must not list them as two.* |
| **HM-MARL** (Robertshaw, RA-L 2026, 10.1109/LRA.2026.3664661) | femoral→ICA | 92–100% single-vasc / **56–80% multi-vasc** | multi (fixed set) | dual | **the honest multi-anatomy bar is 56–80%** |
| World-Models TD-MPC2 (Robertshaw, IROS 2026, arXiv:2604.20151) | MT benchmark | 58% sim / 68% in-vitro (SAC 36/60) | — | single | model-based beats SAC — a methodological threat |
| CASOG (arXiv:2304.09632, TMRB 2023) | delivery | 94% | — | — | explicitly measures retraction (14.07 backward steps) |
| **Ours (v2 eval2, observed)** | RCCA→RICA→siphon | **49%** | **per-worker procedural, resampled/10 eps** | dual (4-dim) | under *harder* protocol; first *positive* eval reward +1.01; train↔eval gap ≈0 |

**Consequences for the paper:**
- **Every cross-system number in this table is NOT directly comparable to ours** — different reward MDPs, truncation rules, and success criteria (reach threshold, step cap). We use 5 mm / 600 steps; most cited works do not state theirs. Wherever these numbers appear (table, Fig. 3 bands, Table C) they must be labeled *reference / not a head-to-head*, never presented as a matched benchmark we beat or lose.
- **Do NOT claim "highest %"** — the Robertshaw IRL result is 96% on 12 anatomies. A >85% claim *must be scoped* to *procedural-anatomy + siphon-depth + recovery-enabled* and contrasted head-on with HM-MARL's **56–80% multi-vasc**, or reviewers cite the 96%.
- **Do NOT claim "first siphon dual-device."** The Robertshaw IRL work traverses the siphon to M1 with two devices. That flag is dead — frame the wedge as *a combination not previously assembled*, never as a "first."
- **The defensible wedge is the combination no one holds together:** SOFA dual-device at siphon depth **+ per-worker procedural anatomy resampled every 10 eps + RL-native (not IL-injected) buckling/slack recovery + privileged-ACTOR teacher (vs KCL's privileged-critic-only).**

### 1.2 Ranked claims (strong / defensible / weak)

**C1 — Freeze-collapse mechanism forensics + fix package. [STRONG]**
The full causal chain — log_std ceiling pin → α decay/whipsaw → action-**MEAN** crush while entropy metrics still look alive → AWAC weights collapse to uniform BC — is, *to our knowledge, not published as a mechanism study* (method scan §4: the symptom class is known [2112.02852], this fine-grained chain is not documented at this granularity). We have the quantitative chain (§2.4) AND a fix package (F1–F8) AND the v2b/v3a ablations. This is the paper's most original ML contribution.

**C2 — "Explore returns hide a dead mean" + in-training deterministic-probe detector. [DEFENSIBLE (small)]**
Deterministic *evaluation* is standard SAC practice — the novelty is **not** the probe itself but using it *in-training as a freeze detector* against explore return, and naming the failure mode it exposes; we have found no direct prior art (method scan §5) but claim it only "to our knowledge." Nearest neighbors: RAGEN reward-std collapse indicator (arXiv:2504.20073), action-noise off-policy paper (openreview NljBlZ6hmG). We operationalize it: σ=1.0 gives |a|≈17 mm/s so explore stayed 39% while the deterministic policy was dead. Cheap, useful diagnostic — pitch as a practitioner's instrument, not a new algorithm.

**C3 — Residual-on-scripted-heuristic as endovascular cold-start at low FEM throughput + the noise-diversity quantification. [STRONG for domain / DEFENSIBLE as mechanism]**
Residual RL on controllers is established (Silver 2018; Johannink ICRA'19) but, *to our knowledge, no published endovascular instance exists* (method scan §2). The sharp sub-claim is the **diagnosis**: deterministic heuristic ≈ **0 successes** (correlated deterministic failures), heuristic+noise heatup ≈ **54%** — a striking RL-era replication of DART (Laskey CoRL'17, arXiv:1703.09327). Headline insight: **"diversity, not competence, is the resource a scripted demonstrator provides."**

**C4 — Loop-neutral telescoping buckle-potential shaping + RL-native recovery taxonomy for intraluminal devices. [DEFENSIBLE]**
PBRS itself is textbook (Ng 1999) and relaxed-truncation has prior art (DeepMimic; Termination Curriculum arXiv:1907.11842; Stubborn arXiv:2606.12814) — **claim the physical quantity (buckle/slack potential) and the dual-device intraluminal recovery taxonomy (soft vs hard recovery, defined in Crux 7), NOT the technique.** Say the shaping is **loop-neutral by construction** (telescoping delta form, φ_end−φ_start), *not* "provably non-farmable": Ng-1999 gives optimal-policy invariance, which is a weaker guarantee than non-gameability during learning. The clean contrast is Barnes et al. (IL, demo-injected, single-device) vs ours (RL-native, online-curriculum, dual-device).

**C5 — AWAC uniform-weight-collapse micro-finding. [DEFENSIBLE]**
The specific "advantage std collapses → weights → uniform → inert BC + entropy" failure is not documented as such (method scan §3; adjacent = adaptive-BC-weight motivation arXiv:2210.13846). Publishable as a micro-result with the E1b adv-normalization fix.

**C6 — Two mapped negative results: AWAC penalty-bracket over-braking + RLPD at realized UTD ≈0.25. [DEFENSIBLE — "map of the space"]**
v2b over-braked (4× penalty neutralized the O(0.01–0.05) early advantage tilt → engine never started). RLPD closed 0/0 with **LayerNorm holding Q bounded 226k updates (stability confirmed) — the failure was signal density, not divergence**. Frames cleanly onto "Three Regimes of Offline-to-Online RL" (arXiv:2510.01460). Reviewers *will* note UTD ≈0.25 ≪ RLPD's validated 20; own it explicitly.

**C7 — Privileged-ACTOR teacher → DAgger student for dual-device endovascular. [DEFENSIBLE — application, not mechanism]**
Asymmetric AC (Pinto 2017), privileged teacher-student (Lee/Hwangbo 2020; CTS arXiv:2405.10830; Scaffolder arXiv:2405.14853), DAgger — all established. **Position as "we adopt X (cite)"**; the novelty is *first application to dual-device FEM endovascular* + *which* privileged-tail components matter (true velocities/contact/branch) under procedural anatomy. Scaffolder's "actor-side privileged recovers ~79% of the gap at 250K–5M samples" is the theoretical backing.

**C8 — Mesh-invariant observation design killing three memorization channels. [DEFENSIBLE]**
Concrete engineering: bbox-normalization leak, absolute-target-coord fingerprint, progress-index leak → fixed-mm isotropic tip-relative encoding. Backed by train↔held-out gap ≈0 (31% explore vs 30.6% eval). Modest but real; strengthens the generalization story.

**Weak / do-not-claim:** asymmetric critic per se, PBRS technique, relaxed-truncation principle, symmetric sampling + LayerNorm (RLPD), noise-injected demonstrators as a *concept* (DART owns it). Cite all as adopted prior art.

### 1.3 The one-sentence elevator claim

> **We present a reinforcement-learning system that reaches [X%] autonomous dual-device navigation at carotid-siphon depth under per-worker procedurally-generated anatomy resampled every ten episodes, by pairing a privileged-*actor* teacher (adapting actor-side privilege from Scaffolder, arXiv:2405.14853, to dual-device FEM) with an RL-native, loop-neutral (telescoping) buckling/slack-recovery curriculum — a capability combination (procedural anatomy + online recovery learning + privileged actor) that, to our knowledge, no prior endovascular system holds together — and we back it with a mechanistic account of the entropy-collapse "policy freeze" that defeats naive SAC/AWAC on this task.** *(Frame the wedge as a combination not previously assembled — never as a "first.")*

---

<a name="2-chain-of-cruxes"></a>
## 2. THE CHAIN OF TECHNICAL CRUXES (causal narrative)

Ordering principle: each crux is *(failure observed → mechanism built → evidence it worked)*, and each failure is the thing the previous fix exposed. This is the spine of the Methods + Results narrative.

### Crux 0 — The task is a cold-start desert at 7.5 Hz.
**Failure:** SOFA FEM caps each of 16 workers at 7.5 Hz → ~47 env-steps/s aggregate → ~1–2M steps/day. PPO+domain-randomization regimes (Rudin CoRL'22, 4096 Isaac envs) need 2–3 orders more throughput — off the table. From-scratch exploration essentially never finds the siphon.
**Mechanism:** offline-seeded off-policy RL (seed buffer 282k transitions / 480 eps) + a scripted heuristic to bootstrap coverage.
**Evidence:** heuristic+noise heatup reaches ~54% *coverage-diverse* successes vs ~0 for the deterministic heuristic — motivating everything downstream. (Feeds Crux 5.)

### Crux 1 — Old observations memorized the mesh (anti-generalization).
**Failure:** three mesh-identity leaks — (a) `NormalizeTracking2DEpisode` normalized the wire by mesh bbox (dominant channel); (b) absolute bbox target coord = mesh fingerprint; (c) LocalGuidance feat0 = remaining/total = per-(mesh,route) progress index.
**Mechanism:** `TipRelativeTracking2D` (fixed mm scale 4.0/135.0, isotropic, 40 dims), `TargetTipOffset2D` (clip ±1 @ 50 mm), feat0→`d_rem_mm/400`, feat7→`is_on_correct_path()`, removed anatomy one-hot tokens, added K-point path preview (δ∈{10,20,40,80} mm) + radius/clearance. LocalGuidance 30→**51** dims.
**Evidence:** train↔held-out gap ≈0 (explore 31% vs eval 30.6% in v2) — the policy generalizes across procedurally-resampled anatomy rather than memorizing. **Two caveats to state in the paper:** (1) the eval set is **98 fixed held-out seeds**, each generating a *distinct* anatomy from a reserved seed band disjoint from every training-worker seed (12345+i) and never regenerated during training — the "held-out seed" is a reservation *rule*, not a single anatomy; report the count as 98 anatomies everywhere. (2) This ≈0 gap was measured at the **30.6% checkpoint**, not the 49% peak — report the train↔eval gap **at the reported-best checkpoint too** (mark **[X pp]** until measured), or the ≈0 reads as cherry-picked.

### Crux 2 — A deployable actor can't see sim state, but the critic can.
**Failure:** contact/buckle/true-velocity are unobservable at deployment, yet essential for credit assignment.
**Mechanism:** asymmetric actor-critic — 24-dim `PrivilegedState` tail appended **last** in the ObsDict (rides through buffer/PER/caches unchanged); critic sees 121, policy slices `[...,:97]` at one chokepoint. (Pinto 2017 lineage — cite, don't claim.)
**Evidence:** critic exploits privileged signal; sets up the later privileged-*actor* teacher (Crux 8). Aux distillation attempted here (Crux 6 exposes it as half-dead).

### Crux 3 — You cannot learn recovery from states the env kills you for entering.
**Failure:** fold-stall / off-path / vessel-end truncations delete exactly the failure states recovery must be learned from.
**Mechanism:** `relax_failure_truncations` (only MaxSteps/SimError end RL episodes; fold/off-path/vessel-end counters keep ratcheting as *features*). Companion: off-path retract tax −0.007→−0.002 (gated ≥3 off-steps + executed −0.1 mm gw); stuck-checkpoint pool (fires at fold=10/off=25, before the 20/50 kill thresholds); escapability+restore-fidelity screener (RESTORE_INSERTED_TOL 10 mm, RESTORE_SLACK_TOL 8 mm, MIN_RETRACT 2 mm).
**Evidence:** recovery behaviors *appear* (v1/v2) — micro-recovery cycles (stuck→retract→re-advance), eval retract-given-stall up to 90.5% early, net +13.2 mm inserted post-stall. (Then Crux 7 shows they fade.)

### Crux 4 — Reward farming beats real progress.
**Failure:** 2×-forward ArcLengthProgress let dithering-in-RCCA farm ~3–4 return ≈ a real success once truncations were relaxed; unpunished timeout farm; global-argmin polyline projection jumped a full loop length in the siphon.
**Mechanism:** ArcLengthProgress 2×→**1× symmetric** (telescopes, round-trip = 0); `MAX_STEPS_PENALTY=−3` checked first; hairpin-safe **windowed** polyline projection (30 mm window, 15 mm fallback). **Anti-buckle potential** φ(slack,contact)∈[−1,0] in **delta form** → telescopes to φ_end−φ_start (**loop-neutral by construction**; any closed loop nets 0, and starting-buckled-then-recovering nets positive). W_SLACK/W_CONTACT 0.5/0.5, SLACK_DEADBAND 5 mm, SLACK_CAP 40 mm, CONTACT_CAP 2 mm.
**Evidence:** first *positive* eval reward (+1.01 at v2 eval2) coincides with real-progress dominance, not farming. (The shaping is **loop-neutral by construction** via the telescoping delta form; Ng-1999 PBRS additionally guarantees the *optimal policy* is unchanged. State it as loop-neutrality + policy-invariance, not as a proof of "non-farmability during learning.")

### Crux 5 — Deterministic heuristic fails identically everywhere; noise makes it a demonstrator.
**Failure:** the scripted `CenterlineFollowerHeuristic` alone yields ≈0 successes — its failures are correlated and deterministic (same wall, same buckle, every episode).
**Mechanism:** noise-injected heatup (physical bands ±30 mm/s, ±1.5 rad/s × heatup_scale) widens state coverage → seed buffer of *recoverable* diverse states; later formalized as **residual-on-heuristic (P2)**: `a_total = clip(a_heur + residual_scale·a_policy)` in raw units, with `a_heur` also exposed as a 4-dim obs.
**Evidence:** ≈0 → **~54%** heatup coverage. This is the DART replication (arXiv:1703.09327) and the quantified insight **"diversity, not competence."**

### Crux 6 — Half of the privileged supervision was dead.
**Failure:** aux force labels (rel 2/3/4) were *identically zero* across all 568,874 buffer states. Root cause (verified 2026-07-12): `MechanicalObject.force`/`dofs.force` is a **per-solve scratch buffer, cleared before any post-step read** — post-step reads can never work. Contact labels scale-starved (std ~1e-3 → aux_coef·MSE ~5e-8, vanishing gradient); aux contact R² peaked 0.554@131k then regressed to 0.486 (linear-97 ceiling 0.779).
**Mechanism:** **E2** repoint labels 2,3,5,6→**0,1,5,6** (node velocities are alive) + **E2.2** `aux_label_znorm` (loss-time EMA z-scoring). True forces need `GenericConstraintSolver.constraintForces` / `computeConstraintForces=True` (scene change, deferred E6).
**Evidence:** planned R² recovery on live-velocity labels; documented as an implementation lesson (§4) regardless of headline.

### Crux 7 — Recovery behaviors were noise-distilled and dried up as entropy collapsed.

**Recovery taxonomy (the operational definition C4 promises — define it here, use it everywhere).** We report two disjoint classes, both on consistent **per-400-episode windows** (training-thirds are the coarse roll-up of these windows):
- **Soft recovery** = an in-place slack/micro-retract correction: the agent senses buckle/slack and applies a small retract-then-re-advance *without withdrawing the device from its current branch*. Measured by *%stalls-with-negative-intent* and *post-stall 20-step intent (mm/s)*.
- **Hard recovery** = a full **stuck→retract→re-advance cycle**: a substantial withdrawal (fold-escape or backing out of a wrong branch) followed by re-advance. Measured by *full-cycle prevalence* and *eval deterministic retract-given-stall*.

**Failure (v2 micro-recovery forensics):** the **86%→58%→28%** trajectory is **hard-recovery full-cycle prevalence** inside successes (per-400-ep windows, rolled to thirds); soft-recovery *%stalls-with-negative-intent* fell **30%→19%→13%**; eval deterministic retract-given-stall (hard) **T1 90.5% → T2 47.6% → T3 7.1%**. The policy learned stall *avoidance* (success stall rate 1.22→0.64/100 steps) while trading away both soft and hard stall *response*. 82% of stall onsets are in the RCCA; failures retract *more* than successes (3.8 vs 1.4 cycles/ep) — the discriminator is not-stalling + stalling-without-slack.
**Mechanism:** **E3** `stuck_fraction=0.15` third sampling lane (transitions with gw_slack>~8.7 mm OR contact_max>0.0026) to keep gradient on the ~10%-slack / ~1%-contact tail; stuck-checkpoint harvest (E4-prep). **Clean-lane rail filter (F4, `EVE_CLEAN_RAIL_MAX=0.15`):** the "clean" BC/AWAC lane *excludes* transitions whose actions are saturated/on-rail (|a| within 0.15 of the limit). Rationale: without it, the AWAC/BC term clones the policy's *own* saturated, on-rail clean-lane actions — reinforcing the non-recovery (drive-forward-into-the-wall) behavior it should be unlearning. The filter stops this self-cloning so the BC term pulls toward genuinely clean demonstrations, not the policy's own dead-mean rails.
**Evidence gate:** slack-tail **P(retract) minus base, positive AND growing** (v2 was flat/decaying: +13pp→+8pp). This is the headline recovery metric (a soft-recovery credit-assignment gate).

### Crux 8 — The stall trigger is unobservable to the actor → make a teacher that can see it.
**Failure:** the deployable prefix cannot see contact/slack, so the actor can't reliably time retraction; asymmetric *critic* alone doesn't fix the actor's blindness.
**Mechanism:** **P2 privileged-ACTOR teacher** — policy consumes full **125-dim** obs (121 + 4-dim heur-intent), `privileged_actor` widens the slice; residual-on-heuristic; baseline eval = **pure heuristic H₀** (the null hypothesis); `aux_coef 0` (labels would become inputs). Then **P2b DAgger student**: obs-only (short history replacing ep_step), evaluated without the privileged tail.
**Evidence (Scaffolder-backed):** actor-side privileged recovers ~79% of the teacher-student gap at 250K–5M samples (arXiv:2405.14853). Target: teacher [X%], student [X%], gap [X pp].

### Crux 9 — Entropy collapse "policy freeze" defeats naive SAC/AWAC (the mechanism paper-within-the-paper).
**Failure (v1):** after learning to eval 13.3%, the policy *froze* (deterministic → 0.95 mm/s) and eval3 collapsed to 3.1%. See §2.4 for the five-link chain.
**Mechanism (F1–F8 fix package):** F1 configurable log_alpha clamp (run −5.0/−2.3, α∈[0.0067,0.100]); F2 awac_lambda 3.0→1.0; F3 action_mean_penalty 0.005 `|atanh(μ)|`; F4 `EVE_CLEAN_RAIL_MAX=0.15` clean-lane rail filter; F5 dropped; F6 IPC timeout guard 900 s; F7 eval_after_pretrain (checkpoint0 baseline); F8 deterministic start-state probe. Plus: soft tanh-rescale log_std band (−2,0) replacing hard clamp (a hard rail = zero-gradient one-way ratchet — the exact v1 mechanism); leaky log-prob floor; per-dim clipped exploration noise (was 1 scalar shared across 4 dims); target_entropy +1.0; ReLU input layer.
**Evidence:** v2 broke the freeze — eval2 **49%** vs **v1's 13.3% peak eval** = **3.7× peak-to-peak** (49/13.3; state the basis explicitly — it is peak-vs-peak, and v1's peak *then collapsed to 3.1%*, so against v1's post-collapse floor the ratio is far larger — quote the peak-to-peak 3.7× to be conservative). Deterministic probe stayed 4.5–6.7× baseline (aggressive, not frozen). v3a/E1b then attacks the *residual* uniform-BC issue.

### Crux 10 — The two mapped dead-ends (negative results = map of the space).
**v2b (AWAC penalty bracket):** hypothesis "2 knobs (mean_penalty 0.005→0.02, log_alpha_max −2.3→−2.0) stabilize eval3." **Failed OPPOSITE — over-braked:** baseline 2.0%@0.36 mm/s, eval1 **0.0%@0.256 mm/s**, eval2 **0.0%**; mean|a0| frozen at 1.0× baseline across 180k updates; α sat at *floor* the whole run; the 4× penalty neutralized the O(0.01–0.05) early advantage tilt → "engine never started." **Ended the AWAC penalty-bracketing line.**
**RLPD P1 (data-centric pivot):** SAC + LayerNorm critics + no-entropy-backup + 50/50 offline/online + UTD 1.0 + pretrain 0 + alpha rails [−5,0]. **Closed 0.0/0.0**, ~6h, u≈226k. Start-state deterministic mean *never left random-init 0.001* across 226k updates while weights demonstrably updated (grad_norm 0.04–0.13, distinct md5s) — "gradient exists, too dilute to build behavior." **Realized UTD ≈0.25 (226k updates / ~823k transitions ≈ 0.27) vs paper's 20**, no ensemble, no pretrain. **Machinery validated: LayerNorm held Q bounded (no divergence, 226k updates) — RLPD's stability claim confirmed; failure = signal density.** Maps onto "Three Regimes" (arXiv:2510.01460): data-centric wins when pretrained-policy ≤ buffer quality *given enough signal density*, which throughput denied.

---

### 2.4 The freeze-collapse chain, quantitatively (v1) — the core of Fig. 4 and the mechanism claim

Five links, each isolated by a different forensic agent:
1. **log_std pinned at CEILING** (σ=1.0) for 100% of states from u≈39k, all 4 dims; floor −2 never touched.
2. **α floor-pinned 165k updates** (log_alpha=−10, u33k–200k) while entropy_proxy fell 2.63→0.14, then **whipsawed** to 0.45; action rails 12%→23% at ±30 (symptom, later receded).
3. **α recovery crushed the action MEAN** (the only entropy lever left once σ was capped): |tanh(μ₀)| peaked 0.255 (explore 422–468k) → **−33% in the single α-spike interval** (→ eval2) → 0.089 by eval3. Regional: start-states **−65%**, buffer-wide −17%.
4. **AWAC inert all run**: awac_weight_mean 0.98–1.04, saturation 0.0 (λ=3 vs a nearly flat critic ⇒ exp(A/λ)≈1 = pure BC + entropy).
5. **Amplifiers**: q1_mean −0.97→−4.71 monotone toward the freeze-return −6.6; EV(attempt)−EV(freeze) never dropped below +1.15 (freezing was never strictly rational — it was a gradient artifact, not a value decision). Buffer NOT poisoned (micro-actions 0.02% of 788k rows).
**Punchline:** *explore returns hid a dead mean* — σ=1 → |a|≈17 mm/s kept explore at 39% while the deterministic policy was frozen. Hence the deterministic-probe detector (C2).

---

<a name="3-experiment-design-story"></a>
## 3. EXPERIMENT DESIGN STORY

### 3.1 Framing: bundles + peel-off, not a random hyperparameter walk
Present the lineage as **controlled experimentation on a throughput budget**. At ~1–2M steps/day, exhaustive one-factor-at-a-time ablation is infeasible, so we ran **fix bundles** (v1 Gen-4 stack; v2 = +F1–F8; v3a = +E1b+E2+E3+E4prep+E8) and then **peel off / bracket** individual knobs where a bundle regressed. This is honest and reviewer-legible: state the budget constraint up front, then show the bracketing (v2b) and the paradigm pivots (RLPD, P2) as deliberate map-of-the-space moves.

### 3.2 The controls that make it experimentation (not anecdote)
- **Deterministic start-state probe (F8):** every eval logs the mean-action deterministic policy at fixed start states — *the* instrument that distinguishes freeze (v1: 0.95 mm/s) from aggressive (v2: 4.5–6.7× baseline) from over-braked (v2b: 0.256 mm/s). Report it alongside every success number.
- **Baseline-eval null hypothesis H₀:** `pretrain 0 + eval_after_pretrain` → checkpoint-0 eval = the run's null. In P2, **H₀ = pure heuristic** (the residual is exactly the learned improvement over the script). Every success curve is read as lift over its own H₀.
- **Per-window recovery tracking:** all recovery metrics are computed on **per-400-episode windows** (training-thirds are the coarse roll-up), split by the **soft/hard taxonomy** of Crux 7: hard = full-cycle prevalence + eval retract-given-stall; soft = %stalls-with-negative-intent + post-stall 20-step intent (mm/s); plus slack-tail P(retract)−base. This turns "recovery faded" into a measured trajectory (hard full-cycle 86→58→28%).
- **Held-out anatomy set:** eval = **98 fixed held-out seeds**, each a distinct anatomy drawn from a reserved seed band disjoint from all training-worker seeds and **never regenerated during training**; train↔eval gap ≈0 certifies generalization, not memorization. *Report the gap at the reported-best (49%) checkpoint, not only the 30.6% checkpoint where the observed ≈0 was measured.*
- **Cache-stamp fail-fast:** `EVE_RL_BUCKLE_COEF` / seed md5 guards prevent silently mixing reward-MDP variants across runs — a reproducibility control worth one sentence.

### 3.3 The ablation table the paper needs

**Table A — Main ablation (all at the harder protocol: procedural anatomy, siphon depth, relaxed truncations; eval = 98 held-out seeds; report mean ± 95% CI over N seeds).**

| # | Config | Δ from previous | Success (best eval) | Det. probe (mm/s) | 1st positive eval reward? | Recovery: slack-tail P(retract)−base | Status |
|---|---|---|---|---|---|---|---|
| 0 | Heuristic only (H₀) | — | ~0% (observed) | n/a | no | n/a | observed |
| 1 | Heuristic + noise heatup | +exploration noise | ~54% coverage (observed) | n/a | n/a | n/a | observed |
| 2 | v1: Gen-4 + AWAC λ3 | full stack | 13.3% then froze→3.1% (observed) | 0.95 (frozen) | no | +13pp→+8pp decaying | observed |
| 3 | v2: +F1–F8 | freeze-collapse fixes | **49%** (observed) | 4.5–6.7× base | **yes (+1.01)** | flat | observed |
| 4 | v2b: +4× mean_penalty, +raised α ceiling | penalty bracket | **0.0%** over-braked (observed) | 0.256 (frozen) | no | n/a | observed (neg) |
| 5 | RLPD: data-centric, UTD ≈0.25 | paradigm swap | **0.0%** (observed) | 0.001 (random init) | no | n/a | observed (neg) |
| 6 | v3a: v2 + E1b+E2+E3 | adv-norm + aux repoint + stuck-lane | **[X%]** | [X] | [X] | **[+X pp, growing]** | hypothetical |
| 7 | P2 teacher: privileged actor + residual | actor-side privilege | **[X%]** (>85 premise) | [X] | [X] | [+X pp] | hypothetical |
| 8 | P2b student: DAgger, obs-only | drop privilege at test | **[X%]** | [X] | [X] | [+X pp] | hypothetical |

**Table B — Component ablation off the best config (leave-one-out, hypothetical [X%]):** remove privileged actor→critic-only; remove residual (end-to-end); remove buckle-potential shaping; remove relaxed truncations; remove stuck-lane; remove clean-rail filter; remove mesh-invariant obs (revert leaks). Each row isolates one contribution's marginal success.

**Table C — Cross-protocol comparison (positioning; every cell labeled "reference, not a head-to-head").** Our best vs stEVE DualDeviceNav 40%, HM-MARL multi-vasc 56–80%, Robertshaw IRL 96%/12-anatomy. Build it *now* with **seven** difference columns, not the loose four: **anatomy count · procedural? · target depth · dual? · recovery-enabled? · termination policy · success threshold (reach mm / step cap)**. The last two are load-bearing because they cut *against* us: our `relax_failure_truncations` removes fold / off-path / vessel-end terminations, which makes per-episode *survival* **easier** than a stEVE-style hard-termination protocol and partially **offsets** the harder procedural-anatomy axis. State this offset explicitly — the net-difficulty claim must be *argued from the column deltas*, not asserted. Because success criteria differ (and most prior works do not publish theirs), label the whole table "not directly comparable / reference bars," never "we beat/lose to X."

**Footnote for Tables A & C:** eval = 98 fixed held-out seeds (distinct anatomies, reserved band, never regenerated); report the train↔eval gap at the *reported-best* checkpoint (the observed ≈0 gap is from the 30.6% checkpoint — mark best-checkpoint gap **[X pp]** until measured).

---

<a name="4-implementation-challenges"></a>
## 4. IMPLEMENTATION CHALLENGES — THE 7.5 Hz FEM WALL AND WHAT IT FORCED

Frame every item as a **reusable lesson** for anyone doing RL on a slow high-fidelity physics sim.

### 4.1 The throughput wall (the root constraint)
SOFA BeamAdapter FEM at nominal `image_frequency=7.5` Hz → 16 CPU workers (1 SOFA sim each) measure **~47 env-steps/s aggregate** = **~2.95 env-steps/s per worker** — well *below* the 7.5 Hz image cadence, because each RL env-step incurs multiple sim substeps, reset/render, and IPC serialization. **Do not compute 7.5 Hz × 16 = 120/s; the realized per-worker rate is ~2.95/s, not 7.5.** GPU trainer ~23% util / 1.5 GB (CPU-sim-bound; 16×~73% CPU). **UTD analysis (self-consistent arithmetic):** a single trainer update costs ~**86 ms** (≈11.6 updates/s) against ~47 env-steps/s (a step every ~21 ms aggregate) → realized **UTD ≈ 0.25** update/env-step (11.6 / 47). The RLPD run independently logged **226k updates over ~823k transitions ≈ 0.27**, consistent with ~0.25. **Lesson:** on FEM-class sims, UTD is *bounded by the update/step latency ratio*, not by choice — algorithms that need high UTD (RLPD's validated 20) are structurally out of reach; this reframes the RLPD negative result as an *architectural* rather than *tuning* outcome.

### 4.2 IPC deadlock forensics (systematic, cost two runs)
Both v1 and v2 deadlocked post-eval3 at explore ~770k. Root cause: `explore_and_update`'s result-loop **TRAINER branch had no deadline** (the worker branch did) → a lost trainer update-result (mp.Queue race, reliably after the 3rd eval) spun `get(timeout=0.5)` forever. v1 main thread wchan = `anon_pipe_read` (no-timeout `_model_queue.get()`); v2 = `poll_schedule_timeout` (the result-collection path F6 didn't cover). **Fix (commit e3bf215, synchron.py, default-on):** (1) trainer-result deadline `EVE_RL_TRAINER_RESULT_TIMEOUT_S=1800` → restart trainer + continue; (2) progress watchdog `EVE_RL_WATCHDOG_STALL_S=2400` → `os._exit(42)` + thread-wchan dump. **Lesson:** in async worker/trainer RL, *every* blocking `queue.get` on the critical path needs a deadline; a partial guard (workers only) hides the bug until the rarer path fires.

### 4.3 Incremental replay save/resume
Per-eval full 1–2 GB buffer re-serialization was the *trigger surface* for the deadlock and a throughput tax. Replaced with **append-only monotonic transition chunks** (~97% of bytes, written once) + an atomically-replaced small state file (sum-tree, lane flags, counters). RLPD run saved **822,973 transitions in 2 chunks + replay_state.npz**, resumable via `--resume`. Seed buffer: `rcca_proc_heatup/seed.npz`, **282,310 transitions / 480 eps**, 67,776,360 bytes, md5 54fe108c…, meta_buckle_coef=0.5, with `EVE_RL_BUCKLE_COEF` cache-stamp fail-fast. **Lesson:** separate the immutable bulk (transitions) from the mutable index (priorities/flags); serialize each on its own cadence.

### 4.4 Dead SOFA force labels (a physics-engine gotcha)
Aux force labels (rel 2/3/4) were identically zero across all 568,874 states because `MechanicalObject.force` is a **per-solve scratch buffer cleared before any post-step read** — a live-wire checkpoint reads force=0 while velocity is alive. **Post-step reads can never work.** True labels require `GenericConstraintSolver.constraintForces` / `computeConstraintForces=True` (a scene change, deferred). Contact labels were scale-starved (std ~1e-3 → aux gradient ~5e-8). **Lesson:** validate that every supervision label has nonzero variance *before* trusting an aux loss; a silently-zero label costs a whole design assumption (E2 repoints to live node velocities).

### 4.5 Observation audits (redundancy + dead dims)
Verified layout: tracking[0:40], target[40:42], last_action[42:46], guidance[46:97], privileged[97:121]. Findings: **~21 prunable dims** (frame t-1 body offsets r=0.99, 18 dup dims; on_path vs in_wrong_branch r=−1.000; d_rem vs d_rem_log r=0.917); **dead dims** at_ostium (74, constant-0 procedural-mesh bug), curv_ahead (/10 scale bug, std 0.011), br_trunk (113, never fires — mis-binned as bridge). **ep_step (flat 68) is the #1 saliency input for all four action heads, the aux head, and log_std** — a determinism-drift crutch (time+momentum = 17% of saliency); mitigation = move ep_step to a critic-only tail + add short obs history (the P2b student change). **Lesson:** run a saliency + pairwise-correlation obs audit; time-index features become brittle crutches.

---

<a name="5-figures"></a>
## 5. FIGURES (6–10 specs)

> Data sources present in repo: `saved/rcca_proc_heatup/`, `saved/v1_collapse_forensics/`, `saved/v2_micro_investigation/`, `saved/monitor_rcca_procedural.md`, `saved/steplog_rcca_proc_harvest_1813.txt`, per-run monitor CSVs (rcca_procedural_v1/v2/v2b, rcca_rlpd_v1, rcca_p2_teacher_v1). Held-out eval = 98 fixed seeds.

**Fig. 1 — System / architecture diagram.** SOFA BeamAdapter FEM (dual J-devices, 7.5 Hz monoplane) → ObsDict (tracking 40 | target 2 | last_action 4 | guidance 51 | heur_action 4 | privileged tail 24) → asymmetric AC (actor sees prefix 97/101; critic + privileged-actor teacher see 125; slice chokepoint marked) → residual composition `a_total = clip(a_heur + scale·a_policy)`. Inset: 16 workers + 1 trainer, three-lane PER (clean 0.3 / stuck 0.15 / general). *Argues: the deployable/privileged split and residual composition in one glance.*

**Fig. 2 — Procedural-anatomy gallery.** 8–12 rendered RCCA→RICA→siphon trees from `RCCAVariedFromMesh` (worker seeds 12345+i) + samples from the **98 held-out anatomies** (reserved seed band, never regenerated during training), each annotated with tortuosity ~ N(1.0,0.3). *Argues: per-worker variation resampled every 10 eps, evaluated on 98 distinct held-out anatomies — the generalization axis no KCL paper has.* Source: regenerate deterministically from seeds.

**Fig. 3 — Main results curve.** X = explore steps (0–800k); Y = held-out success (98 seeds), mean ± 95% CI. Curves: v1 (freezes), v2 (peaks 49%), v2b (flat 0), RLPD (flat 0), v3a **[X%]**, P2 teacher **[X%]**. Horizontal dashed **H₀ = pure heuristic (~0%)**; horizontal reference bands for stEVE 40% and HM-MARL 56–80% **explicitly labeled "reference — different protocol, not a head-to-head"** (different reward MDP / truncations / success criterion; §1.1). Second panel: eval reward with the +1.01 first-positive marker at v2 eval2. *Argues: the >85% claim in full context — and only H₀ and our own ablations are true same-protocol comparisons.*

**Fig. 4 — Freeze-collapse forensics timeline (v1).** Shared x = update step (0–800k), stacked panels: (a) log_std vs its ceiling; (b) log_alpha with floor-pin + whipsaw; (c) entropy_proxy 2.63→0.14; (d) **|tanh(μ₀)| with the −33% α-spike crush** and eval markers; (e) awac_weight_mean flat at ~1.0; (f) q1_mean → −6.6. *Argues: the five-link causal chain — the strongest ML contribution.* Source: `saved/v1_collapse_forensics/`.

**Fig. 5 — Recovery-rate evolution (v2), split by soft/hard taxonomy (Crux 7), per-400-episode windows rolled to thirds.** **Hard recovery:** full-cycle prevalence 86→58→28%, eval retract-given-stall 90.5→47.6→7.1%. **Soft recovery:** %stalls-with-negative-intent 30→19→13%, post-stall 20-step intent +1.84→+4.78→+7.20 mm/s. Twin metric: success stall-rate 1.22→0.64/100 (avoidance up as both soft and hard response decay). *Argues: recovery is noise-distilled and fades — the motivation for stuck-lane + privileged actor.* Source: `saved/v2_micro_investigation/`.

**Fig. 6 — Deterministic-probe vs explore-return divergence.** Two lines over training: explore success (stays ~39% in v1) vs deterministic-probe speed/success (collapses to 0.95 mm/s). Annotate σ=1 → |a|≈17 mm/s. *Argues: "explore returns hide a dead mean" (C2) — the detector's raison d'être.*

**Fig. 7 — Slack-tail P(retract) − base, over training (the headline recovery gate).** For the best config vs v2: P(retract | slack-bin) minus base-positive rate, per window; v2 flat/decaying (+13→+8 pp), target config **[+X pp, growing]**. *Argues: the stuck-lane + adv-norm actually restore recovery credit assignment.*

**Fig. 8 — Ablation bars (Table B as a figure).** Leave-one-out success off best config: −privileged-actor, −residual, −buckle-potential, −relaxed-truncations, −stuck-lane, −clean-rail, −mesh-invariant-obs. Each bar Δsuccess with CI. *Argues: marginal contribution of each component.*

**Fig. 9 — Teacher vs student gap.** P2 teacher (privileged, 125-dim) vs P2b DAgger student (obs-only, no privileged tail) on the 98 held-out seeds; annotate Scaffolder's ~79%-gap-recovery expectation. *Argues: the deployable policy retains most of the teacher's skill.*

**Fig. 10 (optional) — Loop-neutrality illustration.** Schematic + measured return for three trajectories on the same anatomy: (a) real progress, (b) dither-in-RCCA loop, (c) start-buckled-then-recover. Show telescoping φ nets 0 on the loop, positive on genuine recovery. *Argues: the buckle-potential shaping is loop-neutral by construction (C4) — closed loops net zero.*

---

<a name="6-sample-abstract"></a>
## 6. SAMPLE ABSTRACT (~200 words)

Autonomous dual-device (guidewire + catheter) endovascular navigation is the open sub-problem of the field's standard benchmark: on stEVE DualDeviceNav, SAC reaches only 40/100, and the shorter device's under-observed dynamics produce chaotic motion. Prior dual-device successes at carotid-siphon depth (96%) rely on small fixed patient anatomy sets and demonstration-injected recovery; none learns recovery online under procedurally varied anatomy. We present a reinforcement-learning system that navigates two independently actuated devices to siphon-depth targets under **per-worker procedurally generated vessel trees resampled every ten episodes**, reaching **[X%]** success (95% CI) on 98 held-out anatomies — versus a **[X%]** pure-heuristic null and our own **49%** freeze-prone SAC/AWAC baseline. Three ingredients drive this: (i) a **privileged-actor teacher** (adapting actor-side privilege from robotics) trained as a residual on a scripted heuristic, where we show a deterministic controller yields ~0 successes but noise injection yields ~54% — *diversity, not competence, is what a script provides*; (ii) an **RL-native, loop-neutral buckling/slack-recovery curriculum** (telescoping-potential shaping + relaxed failure truncations + stuck-state sampling), with a soft/hard recovery taxonomy; and (iii) a **mechanistic account of "policy freeze"** — an entropy-collapse chain (log-std ceiling → temperature whipsaw → action-mean crush) that we detect with an in-training deterministic probe and defeat with a targeted fix package. We also report two mapped negative results (AWAC penalty over-braking; RLPD at throughput-limited UTD). Code and procedural-anatomy generator will be released.

---

<a name="7-sample-introduction"></a>
## 7. SAMPLE INTRODUCTION (~800 words; [ref] placeholders)

Endovascular interventions — mechanical thrombectomy, cerebral aneurysm treatment, carotid stenting — require an operator to thread a thin guidewire and a coaxial catheter through tortuous, patient-specific vasculature under fluoroscopic imaging. Autonomy promises to reduce radiation exposure, standardize technique, and extend specialist reach, and simulation-based reinforcement learning has become the dominant route toward it [stEVE, arXiv:2410.01956; Robertshaw IJCARS 2025]. Yet a sharp gap divides the easy from the hard case. On the community benchmark stEVE, single-guidewire tasks are effectively solved — BasicWireNav 98/100 in simulation and 97/100 on a physical bench, ArchVariety 90/84 [arXiv:2410.01956] — while the two-device task, **DualDeviceNav, sits at 40/100**. The benchmark's authors attribute this to the observation neglecting the shorter device, which induces "chaotic motion" of the outer catheter. Dual-device navigation is where autonomy stalls.

Three difficulties compound in the two-device, deep-target regime. First, **coordination**: two coaxial elastic instruments with four continuous action dimensions interact through contact and slack; advancing one can buckle or drag the other. Second, **anatomy generalization**: the carotid siphon is a sequence of near-180° bends whose geometry varies sharply across patients, and methods tuned to a fixed mesh memorize rather than generalize. Third, **recovery**: a device that stalls, buckles, or enters the wrong branch must retract and re-advance — a skill that episodic failure-termination actively prevents an agent from acquiring, because it deletes the very states recovery must be learned from. The strongest prior dual-device results address the first two on **small fixed patient sets**: Robertshaw et al. reach 96% into the middle cerebral artery through the siphon, but on only twelve patient anatomies with no procedural variation and no explicit recovery learning [IJCARS 2025, PMC12167253]; the hierarchical multi-agent successor drops to **56–80% on multi-vasculature** [HM-MARL, RA-L 2026]. Recovery has been demonstrated only via **imitation learning with recovery demonstrations injected offline**, single-device [Barnes et al., arXiv:2510.09497]. No system combines procedurally varied anatomy, siphon-depth dual-device control, and recovery learned online.

A further obstacle is unglamorous but decisive: **throughput**. High-fidelity FEM simulation (SOFA + BeamAdapter) runs at ~7.5 Hz per worker; sixteen parallel workers yield only ~47 environment-steps per second — one to two million steps per day, two to three orders of magnitude below the massively parallel regimes that make from-scratch deep-RL exploration routine [Rudin, CoRL 2022]. At this budget, exploration essentially never discovers the siphon on its own, and update-to-data ratios are bounded by the trainer's per-update latency rather than chosen freely — a constraint that, as we show, silently defeats otherwise-appropriate algorithms.

We address these jointly. Our agent controls two devices toward siphon-depth targets under **per-worker procedurally generated vessel trees, resampled every ten episodes**, with a held-out anatomy never used in training. To overcome cold-start, we learn a **residual on a scripted centerline-following heuristic**, and we report a diagnostic we believe is broadly useful: the *deterministic* heuristic yields essentially zero successes because its failures are correlated across episodes, whereas injecting exploration noise yields ~54% — **diversity, not competence, is the resource a scripted demonstrator provides** [cf. DART, arXiv:1703.09327]. We make the recovery skill learnable with an **RL-native curriculum**: relaxed failure truncations so failure states persist as features [cf. Termination Curriculum, arXiv:1907.11842], a **loop-neutral telescoping buckle-potential shaping** term (loop-neutral by construction; PBRS additionally leaves the optimal policy invariant [Ng et al. 1999]), and a dedicated stuck-state sampling lane, with recovery reported under an explicit soft/hard taxonomy. Because the stall trigger (contact, slack) is unobservable to a deployable policy, we train a **privileged-actor teacher** [Scaffolder, arXiv:2405.14853; asymmetric AC, Pinto et al. 2017] and distill it to an observation-only student.

Along the way, naive SAC/AWAC froze: after learning to navigate, the deterministic policy collapsed to near-zero motion while exploration returns still looked healthy. We trace this to a five-link **entropy-collapse chain** — log-std ceiling pin, temperature whipsaw, action-mean crush, inert advantage weighting — that, to our knowledge, has not been reported at this granularity [cf. Target Entropy Annealing, arXiv:2112.02852], and we defeat it with a fix package validated by ablation. Finally, we map two dead ends — AWAC penalty-bracket over-braking and RLPD at throughput-limited UTD ≈0.25, where LayerNorm demonstrably held Q-values bounded for 226k updates, isolating *signal density*, not stability, as the failure [RLPD, Ball et al. 2023; Three Regimes, arXiv:2510.01460].

**Contributions.**
1. A dual-device endovascular RL system reaching **[X%]** to siphon depth under per-worker procedural anatomy (resampled every ten episodes) — a combination (procedural anatomy + online recovery + privileged actor) that, to our knowledge, no prior system holds together. *(Framed as a not-previously-assembled combination, not a "first.")*
2. A **privileged-actor residual-on-heuristic** cold-start method for low-throughput FEM RL, with the **noise-diversity quantification** (~0% → ~54%).
3. An **RL-native, loop-neutral buckling/slack-recovery curriculum** (with a soft/hard recovery taxonomy) for intraluminal devices, contrasted with IL-injected recovery.
4. A **mechanistic freeze-collapse account + deterministic-probe detector + fix package**, with ablation.
5. Two **mapped negative results** placing AWAC and RLPD in the throughput-constrained regime.

---

<a name="8-methods-skeleton"></a>
## 8. METHODS SECTION SKELETON

**M1. Task and simulator.** Dual J-devices (900 mm; ±30 mm/s, ±1.5 rad/s; gw OD 0.36 mm, cath OD 0.6/0.7 mm; E=1e3; friction 0.001) in SOFA BeamAdapter FEM, monoplane tracking at **7.5 Hz**. Target: random centerline point ≥40 mm past ostium on RCCA→RICA→siphon, reach threshold 5 mm; episode cap **600 steps (80 s)**. *Must state:* 7.5 Hz, 4 action dims, 5 mm reach, 600-step cap.

**M2. Procedural anatomy.** `RCCAVariedFromMesh`: bell-envelope (smoothstep, anchored 15 mm proximal / 25 mm distal) × amp × Σ 3 random-phase sinusoids (0.7/1.3/2.1 cycles), amp = 4 mm × tortuosity, tortuosity ~ N(1.0,0.3)∈[0.4,1.6]; radii ~ N(1.0,0.07); watertight re-mesh, obs-compatible frame. Worker i seed 12345+i, regen every **10 episodes**. **Held-out = 98 fixed seeds** from a reserved band disjoint from all training-worker seeds, each a distinct anatomy, **never regenerated during training**. *Must state:* the regen cadence, that eval spans 98 distinct held-out anatomies (not one), and the reserved-band isolation.

**M3. Observation design (mesh-invariance).** Flat 121 (125 w/ heur_action): tracking 40 (`TipRelativeTracking2D`, fixed mm scale, frame-stack n=2), target 2 (`TargetTipOffset2D`), last_action 4, guidance 51 (`LocalGuidance`), privileged tail 24 (`PrivilegedState`) appended LAST. Three memorization leaks removed (bbox-norm, absolute target, progress index). *Must state:* the three leaks + the 97/101 policy-prefix vs 121/125 critic split.

**M4. Asymmetric actor-critic + privileged-actor teacher.** Critic input = full flat obs + action; actor slices `[...,:97]`; `privileged_actor` widens the slice to 125 (teacher). *Must state:* the single-chokepoint slice; the 24-dim tail contents (node vel/force, contact proxy |pos−free_pos|, rotations, branch one-hot, counters).

**M5. Residual composition on a scripted heuristic.** `a_total = clip(a_heur + residual_scale·a_policy, low, high)` in raw units; `a_heur` cached per sim state and exposed as 4-dim obs; init behavior = pure heuristic. Heuristic = P-control, 3 phases (§10 of framework notes). *Must state:* the residual equation and that H₀ = pure heuristic.

**M6. Reward.** success +3.0·1[d<5]; ArcLengthProgress **1× symmetric** (0.01·Δs, telescoping); daughter-commit ±1; off-path step −0.007→−0.002 (recovery-gated); **anti-buckle potential** r += 0.5·(φ_t−φ_{t−1}), φ(slack,contact) = −[0.5·clip(slack−5,0,40)/40 + 0.5·clip(contact,0,2)/2]∈[−1,0]; terminal −3 (max_steps) / −5 (failure trunc) / −1 (overshoot-in-daughter). **relax_failure_truncations** (only MaxSteps/SimError end RL eps). *Must state:* the potential equation + **loop-neutrality** (telescoping delta form; any closed loop nets 0) and PBRS optimal-policy invariance — *not* a "non-farmable during learning" proof — + relaxed-truncation rationale, **plus** the difficulty-offset admission (relaxing truncations makes per-episode survival easier, partially offsetting the harder procedural-anatomy axis; see Table C).

**M7. Algorithm.** Twin-critic SAC (γ=0.99, τ=0.005, **no entropy backup**). AWAC actor: L = −E[w·log π(a_buf|s)] + α·log π(ã|s), w = exp(A/λ).clamp(20), λ=1.0; **E1b:** w = exp((A/std A)/τ_adv), τ_adv=2.0. SAC actor (P1/P2): L = E[α·log π(ã) − min Q(s,ã)]. Temperature: hard rails **log α∈[−5,0]**, **target_entropy +1.0**. Aux head (coef 0.05, dims 0,1,5,6 + z-norm; off in P2). *Must state:* the AWAC weight eq, λ, τ_adv, no-entropy-backup, rails, target_entropy.

**M8. Freeze-collapse fixes.** Soft tanh log_std band (−2,0) (nonzero gradient — hard clamp = one-way ratchet); leaky log-prob floor; per-dim clipped exploration noise N(0,0.25); ReLU input layer; F1–F8 (log_alpha clamp, awac_lambda 3→1, action_mean_penalty, clean-rail filter, IPC guard, eval_after_pretrain, deterministic probe). *Must state:* the soft-band-vs-hard-clamp gradient argument.

**M9. Networks.** Actor MLP 256-256, input 101/125, heads mean(4)/log_std(4)/[aux]. Critics Q1/Q2 256-256, **LayerNorm after every hidden Linear (critics only)**, input 125+4. Batch 32; Adam 3e-4 (linear decay 0.15 over 6e6); grad clip 1.0. *Must state:* LayerNorm-critics-only + batch 32.

**M10. Replay & sampling.** Transition-level PER (cap 2M, sum-tree, α=0.6, β 0.4→1.0). **Three-lane batch:** clean (balanced_fraction 0.3; **clean-rail admission `EVE_CLEAN_RAIL_MAX=0.15`** *excludes* saturated/on-rail transitions, |a| within 0.15 of the limit, from the BC/AWAC lane — so the BC term cannot clone the policy's own on-rail non-recovery actions and reinforce driving-into-the-wall), stuck (stuck_fraction 0.15; slack>~8.7 mm OR contact_max>0.0026), general. RLPD variant: symmetric offline_fraction 0.5, IS=1. Incremental append-only chunks + atomic state file. Seed 282k transitions. *Must state:* the three fractions + clean-rail threshold **and its self-cloning rationale** + incremental-save design.

**M11. Distributed training & protocol.** 16 CPU workers + 1 GPU trainer (Synchron); deadlock guards (model-queue 900 s, trainer-result 1800 s, watchdog 2400 s→os._exit(42)). Heatup 20k; **250k explore steps between evals**; eval = **98 fixed held-out seeds**. Realized UTD ~0.25 (nominal v2/v3 UTD 0.5; per §4.1 arithmetic, ~11.6 updates/s ÷ ~47 env-steps/s); ~47 env-steps/s aggregate (~2.95/worker). *Must state:* eval protocol (98 seeds), inter-eval interval, realized UTD and how it is bounded.

**M12. Teacher→student distillation (P2b).** DAgger: run teacher, log (deployable-prefix obs → teacher action), train obs-only student (short obs history replacing ep_step), eval without privileged tail; optional warm-start. *Must state:* student sees no privileged tail at test.

---

<a name="9-results-plan"></a>
## 9. RESULTS-SECTION PLAN

- **R1 Main result:** best config **[X%]** on 98 held-out seeds vs H₀ **[~0%]** and v2 49%; Fig. 3 + Table A. Report mean ± **95% CI** over the 98-seed eval, and across **≥3 training seeds** where budget allows (state N explicitly; if single-seed, say so and lean on the 98-seed eval CI).
- **R2 Generalization:** train↔held-out gap on 98 distinct held-out anatomies (v2: explore 31% vs eval 30.6% ≈ 0) → no memorization; Fig. 2. **Caveat to report:** the observed ≈0 gap is at the 30.6% checkpoint — also report the gap **at the reported-best (49%/[X%]) checkpoint** (mark **[X pp]** until measured), else the ≈0 reads as cherry-picked.
- **R3 Recovery:** per-400-episode recovery metrics under the **soft/hard taxonomy** (Crux 7); slack-tail P(retract)−base as the headline gate (Fig. 5, 7). Report the hard-recovery deterministic eval retract-given-stall trajectory and the soft-recovery negative-intent trajectory separately.
- **R4 Freeze forensics + fix:** Fig. 4 + Fig. 6; v1→v2 delta = **3.7× peak-to-peak (49% vs v1's 13.3% peak eval, before v1's collapse to 3.1%)** — state the basis; deterministic-probe values distinguishing freeze/aggressive/over-braked.
- **R5 Ablations:** Table B / Fig. 8 leave-one-out; state each component's Δsuccess ± CI.
- **R6 Teacher vs student:** Fig. 9; gap in pp.
- **R7 Negative results:** v2b (0/0, over-braked, det-probe 0.256) and RLPD (0/0, UTD ≈0.25, Q bounded 226k updates); one compact paragraph + one row each in Table A.
- **R8 Safety-metric limitation (state it, don't hide it):** because the SOFA force labels are dead (per-solve scratch buffer cleared pre-read, §4.4/Crux 6, E6 deferred), we **cannot report a contact-force / safety metric** this iteration — whereas the SOTA we position against does (Robertshaw 0.24 N mean force; CASOG 14.07 retraction steps). Acknowledge this directly, report available proxies (slack/contact-proxy statistics, retraction counts), and point to the `GenericConstraintSolver.constraintForces` / `computeConstraintForces=True` (E6) path as the fix. See risk A11.
- **Stat reporting rules:** always report eval protocol (98 fixed held-out seeds), CI method (bootstrap or Wilson for proportions over 98 trials), and the deterministic-probe number next to every success number. Flag every hypothetical as [X%] until real.

---

<a name="10-related-work-map"></a>
## 10. RELATED-WORK MAP (one paragraph + citation list)

**Plan (one tight paragraph, four moves):** (1) *Benchmark & single-device solved:* stEVE and its 98/40 single/dual split [arXiv:2410.01956]; CathSim/SplineFormer single-device phantom [arXiv:2501.04515; 2512.18081]. (2) *Dual-device SOTA we position against:* the Robertshaw IRL result — 96%/12-anatomy through the siphon — cited **once** as a single paper [arXiv:2406.12499 → IJCARS 2025, PMC12167253] (the arXiv preprint and the IJCARS version are the *same* work; do not list them as two separate 96% results), HM-MARL 56–80% multi-vasc [RA-L 2026], World-Models TD-MPC2 58/68% [arXiv:2604.20151] — establish that our wedge is procedural anatomy + online recovery + privileged actor, not raw %. (3) *Recovery:* Barnes IL-injected recovery [arXiv:2510.09497], buckling failure modes [arXiv:2110.01840], vs our RL-native curriculum. (4) *Methods we adopt (cite, don't claim):* asymmetric AC [Pinto 2017; KiVi 2509.23650], privileged teacher-student [Lee/Hwangbo 2020; CTS 2405.10830; Scaffolder 2405.14853], residual RL [Silver 2018; Johannink ICRA'19], offline-to-online [AWAC 2006.09359; RLPD PMLR v202/ball23a; WSRL 2412.07762; Three Regimes 2510.01460], PBRS [Ng 1999; ADOPS 2505.12611], termination curricula [1907.11842; DeepMimic 1804.02717; Stubborn 2606.12814], DART [1703.09327], entropy/α [2112.02852; plasticity 2403.00514], RAGEN [2504.20073]. **Priority-overlap check before submission:** Chen et al. decoupled procedural execution [arXiv:2607.00066, Jul 2026] and online-expert-correction DAgger [arXiv:2602.20216] overlap our residual/teacher framing — cite and differentiate.

---

<a name="11-risks"></a>
## 11. RISKS / REVIEWER ATTACKS + PREPARED ANSWERS

**A1. "Simulation only — no real robot."** *Answer:* stEVE transfers single-device to physical benches (97/84%) [arXiv:2410.01956] and HM-MARL/World-Models add in-vitro; sim-to-real for *dual-device deep* nav is only partially shown anywhere. We scope the claim to the SOFA benchmark, keep obs deployment-realistic (monoplane tracking, no privileged tail at student test), and cite the transfer precedent. Frame real-robot transfer as declared future work, not a hole.

**A2. "IJCARS-2025 already did dual-device siphon at 96% — you're not first and you're lower."** *Answer:* different protocol — they use 12 fixed patient anatomies with no procedural variation and no learned recovery; the honest multi-anatomy bar is HM-MARL's 56–80%. Our contribution is the *combination* (procedural anatomy + online recovery + privileged actor) and the mechanistic/negative-result findings, not the top-line percentage. Table C makes the protocol difference explicit. **Delete any "first siphon dual-device" phrasing.**

**A3. "Single anatomy family (RCCA→siphon)."** *Answer:* procedural generation spans a continuous tortuosity/radius distribution (N(1.0,0.3) tortuosity, 3-frequency perturbation) with a strictly held-out seed; this is broader within-family variation than the ≤12 fixed meshes of prior dual-device work. Extending the generator to additional arch types is future work.

**A4. "Heuristic dependence — is the policy doing anything?"** *Answer:* H₀ (pure heuristic) ≈ 0% is the null; the residual *is* the learned lift, and Table B's −residual ablation quantifies it. The noise-diversity result (0%→54%) shows the heuristic contributes coverage, not competence.

**A5. "RLPD at UTD ≈0.25 is not a fair test of RLPD (validated at UTD 20)."** *Answer:* agreed and stated explicitly — we present it as a *throughput-regime* result, not a refutation of RLPD. The load-bearing finding is positive: LayerNorm held Q bounded for 226k updates (RLPD's stability mechanism confirmed), isolating signal density as the failure. This is the honest "map of the space" framing [Three Regimes, arXiv:2510.01460].

**A6. "Freeze-collapse is just known entropy collapse."** *Answer:* the *symptom class* is known [2112.02852]; the specific five-link chain (log-std ceiling → α whipsaw → action-mean crush while entropy metrics look alive → inert AWAC BC) and the deterministic-probe detector are not published at this granularity, and we back them with the F1–F8 ablation and the v2/v2b contrast.

**A7. "4 pages is too short / wrong venue."** *Answer:* target a workshop/extended-abstract track for the 4-page version; the content budget upgrades to a 6-page RA-L/IROS or full IJCARS submission. Decide venue before finalizing scope.

**A8. "Model-based (TD-MPC2) beats SAC on this benchmark — why not that?"** *Answer:* cite World-Models IROS-2026 (58/68%) as a complementary direction and, if budget allows, add it as a baseline/ablation row; note our contributions (recovery curriculum, privileged actor, freeze forensics) are algorithm-agnostic and compose with model-based backbones.

**A9. "Reward shaping is farmable / you tuned to the metric."** *Answer:* the buckle-potential is potential-based (telescoping delta form) — any closed loop nets 0 **by construction** (Fig. 10) — and PBRS leaves the optimal policy invariant [Ng 1999]; ArcLengthProgress was de-farmed 2×→1× symmetric; MAX_STEPS checked first. We claim **loop-neutrality + policy-invariance**, *not* a proof of non-gameability during learning — the honest, defensible version of the claim.

**A10. "Results are hypothetical / cherry-picked eval checkpoint."** *Answer:* every success number is paired with its deterministic-probe value and reported on the fixed 98-seed held-out set with CIs; best-checkpoint selection is disclosed and the full eval trajectory (including regressions) is shown, not just the peak. The train↔eval gap is reported **at the reported-best checkpoint**, not only at the 30.6% checkpoint where the observed ≈0 was measured.

**A11. "You report no contact-force / safety metric — clinical relevance?"** *Answer (concede + path):* the SOFA force labels are a per-solve scratch buffer cleared before any post-step read (§4.4), so this iteration cannot report mean contact force — unlike the SOTA we cite (Robertshaw 0.24 N; CASOG 14.07 retraction steps). We report available proxies (slack, contact-proxy |pos−free_pos|, retraction counts) and specify the fix (`GenericConstraintSolver.constraintForces` / `computeConstraintForces=True`, E6) as the immediate next step. We do not claim a safety result we cannot measure.

**A12. "Relaxed truncations make your protocol *easier*, not harder."** *Answer (own the offset):* correct in isolation — removing fold/off-path/vessel-end terminations makes per-episode survival easier than a hard-termination protocol. But it is *load-bearing for the recovery contribution* (you cannot learn recovery from states the env deletes), and it is only one axis. Table C tabulates all seven axes (anatomy count, procedural, depth, dual, recovery, **termination policy**, **success threshold**); the net-difficulty claim is argued from those deltas — the harder procedural-anatomy + siphon-depth + dual axes against the easier termination axis — never asserted as a blanket "harder."

---

*End of plan. Keep this doc in sync as v3a / P2 teacher / P2b student runs land — replace every [X%] with measured numbers + CIs and re-check the priority-overlap papers (arXiv:2607.00066, 2602.20216) before submission.*
