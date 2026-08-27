# v3a → path forward: knot mechanism, stuck-pool verdict, deep-skill absence, and the recommended sequence

**Date:** 2026-07-19 · Companion to `FORENSIC_RCCA_V3A.md`. Sources: a 4-investigation +
synthesis + 9-verification workflow (knot-verify, pool-audit, deep-skill-mining,
mode-classifier), plus direct snapshot inspection and the live eval8 regression. Verification
disposition folded in: 5 CONFIRMED, 4 CORRECTED (corrections applied below).

---

## 0. Live update: the predicted seed-eviction regression has begun

**eval8 (explore 1.76M): Quality 0.500 (49/98) — down from eval7's 0.520.** The replay ring is
now full (`buffer_len=2,000,000`) and the newest chunk (`…1787315_2042024`) has wrapped past
the cap, overwriting the head of the 282,310 seed transitions. First eval after the wrap lost
both flapper seeds. This is exactly the FIFO demo-eviction failure the prior forensic forecast
(cap at ~1.72M; actual ~1.79M). **The run is now degrading, not plateauing — stop it.**

---

## 1. The user's knot mechanism: CONFIRMED at the body, CORRECTED at the tip

The catheter-knot picture is materially right, with one important geometric correction.

- **CONFIRMED — the coiling/telescoping deadlock.** In deep-target eval failures the guidewire
  is never advanced, so the catheter can't push its tip past the last bend; it buckles and
  feeds all commanded forward motion into redundant coil. Trace (eval seed 7, target z=550):
  the tip **freezes at (16.7, 58.6, 511) for 420 steps** while `cath_ins` climbs 162→419 mm
  (**+257 mm of pure slack**), `gw_ins` stays <8 mm, catheter command stays full-forward.
- **CONFIRMED — the coil sits at the RVA ostium and it is a genuine knot.** NPZ geometry: far-arc
  self-contact (nodes >15–20 mm apart along the shaft, <2 mm in space) in **69% of run-2
  captures**, median self-distance **0.2–0.4 mm** (the beam passes *through* itself), multi-zone
  wraps (11–20 contact zones), all centered at z≈408–416 mm. Self-contact loops appear only past
  ~200 mm catheter insertion and intensify sharply with it (0 pairs below 200 mm → hundreds–
  thousands above 600 mm) — a strong but *not strictly monotonic* trend (dip at ~525 mm, outlier
  at 805 mm); exact counts depend on the loop-counting definition. **Independently verified:** the
  RVA ostium was located from `cur_branch=…RVA…` log labels at mean (17, 65, 415), z-band 410–420 mm,
  and the coil centroid coincides with it to within 1.9–2.5 mm at high insertion — the coil sits far
  closer to the RVA shelf (8–17 mm) than to the insertion origin (20–44 mm), so the RVA-shelf
  location is real, not an access-point buckling artifact. (10/10 pivotal claims now verified;
  this one CORRECTED only on the loop-count series, core geometry confirmed.)
- **CORRECTED — in deterministic eval the *tip* does not enter RVA and re-enter RCCA.** The tip
  goes straight up RCCA and freezes at the r=2.0 choke (`cur_branch=RCCA` the whole grind, 0 RVA
  tip-touches); it's the slack **shaft ~100 mm behind the tip** that coils and prolapses toward
  the RVA ostium. So "folds into RVA" is true of the *body*, not the tip. (The tip-into-RVA +
  fold-counter events are real but come from the noisy explore/eval-capture stream — see §2.)
- **CONFIRMED with nuance — "tight radius is fine catheter-only."** Successful seeds traverse down
  to local_r=2.2 mm cleanly; every failure freezes at the r=2.0 mm point (arc 157.8). The choke is
  the last 0.2 mm-tighter spot, and it's impassable *only because no guidewire lead/stiffness is
  provided* — precisely the user's reading.
- **CONFIRMED — learned at pretrain.** Snapshot `ep001_pid11932` (pretrain baseline) already shows
  catheter coiling; the online tangles are bulkier but the strategy is present from pretrain. This
  matches the earlier finding that the deterministic mean rails on contact states (0%→86% over
  training) while free-lumen behavior stays clean.
- **Correction from verification:** the fold-capture phenomenon is **not purely exploration-side** —
  106/423 (25%) of stuck captures come from the deterministic eval stream across all 7 sessions
  (42 distinct mesh/seed pairs, 25 re-captured session after session). That *reinforces* the
  user's "undiverse" concern (same states captured repeatedly), while the specific seed-7 grind
  genuinely never tripped `fold` (fold=0/20) — different eval seeds behave differently.

**Bottom line:** the user's mechanism is correct — a guidewire-starved catheter knotting itself
into stiffness at the RVA shelf to keep shoving — with the refinement that in clean deterministic
eval the tip wedges at the RCCA choke and the knot forms in the trailing slack.

---

## 2. Stuck pool: the user's skepticism is vindicated

All three doubts (recoverability, diversity, restore fidelity) hold for the states that matter.

| Concern | Verdict | Evidence |
|---|---|---|
| **Unrecoverable knots** | Justified for the choke captures | 69% of run-2 captures are self-overlapping folds; tiered by cath/proj_s ratio: T0 simple 42%, T1 moderate 41%, T2 deep-slack 13%, T3 capped(~898mm) 5%. T1–T3 are 98–100% self-contacted. |
| **Undiverse** | Justified where it counts | 57% of the pool sits in the choke zone and is 85% knot-contaminated there. The pessimistically-clean remainder (233 states) is **88% one shallow RVA-wrong-branch wedge phenotype** (proj_s≈34 mm). Only **15–39 clean states exist at the ~150 mm choke** the curriculum must actually teach. |
| **Restore fidelity** | Partly justified | Dynamic pop risk ≈0 (states quiescent: velocity 0, forces 0, no stored spring-back). But 265 states have <1 mm centerline self-overlap that only exists because **SOFA device self-collision is OFF** (`sofabeamadapter.py:526-539`, proximity=0, no selfCollision). They're restorable in the *same* sim but encode an unphysical regime. |
| **Screener would catch it** | **No** | `escapability.py` judges everything on the **guidewire only** (gw retract_mm, gw slack gate); catheter length and 3D shape are never checked. The fold escape gate (`slack ≤ 8 mm`) is *already true at capture* for 273/309 folds → ~90% of confirmed knots would be stamped "escapable." And the retract budget is only ~107 mm (40 steps × 20 mm/s at 7.5 Hz — corrected down from 800 mm), far short of the 400–900 mm coiled. |

**Verdict:** the v3b gate (≥200 states / ≥8 fingerprints) passes on *count* but fails in *spirit*.
The raw pool cannot carry a choke-recovery curriculum, and the existing screener is blind to
every one of the user's concerns. Tooling is also broken (POOL path points at `rcca_proc_stuck`,
not our `rcca_v3_stuck`; ~45 mounts still on the stale `D:\Arjun` prefix). The user was right to
be suspicious.

---

## 3. The deep guidewire-led skill does not exist in our data — it must be generated

The decisive question for whether *any* training on existing data can work:

- **Seed buffer:** all 27 successes are shallow (reach ≤106 mm, far below the 170 mm cliff) and use
  guidewire and catheter symmetrically (random heatup). Zero deep, zero gw-led.
- **Pretrain eval deep probes:** the 5 seeds that reached 175.4 mm did so **catheter-led** (cath
  194–204 mm, gw 1.7–4.2 mm, 0% gw-forward commands). 175.4 mm is a reproducible *catheter-alone
  mechanical wall*, not a skill.
- **Explore stream (5,319 episodes, whole run):** 110 deep episodes (proj_s>170 mm), 6 deep
  successes — **all catheter-led shoves** (≤190.4 mm, mostly on one worker's vessels). Zero gw-led
  successes anywhere.
- **BUT — 7 guidewire-led excursions prove the sim permits the missing skill.** The star is
  `worker_12 pid2586 ep9`: guidewire commanded full-forward 68% of steps, gw led the catheter from
  step 40, reaching **proj_s 234 mm (d_tgt 22 mm on a 263 mm path)** — deeper than any catheter-led
  attempt and past both the choke and the catheter wall — before mirroring the slack-feed pathology
  onto the guidewire and timing out 19.7 mm short. Five of the seven occurred in the first ~6 h
  (the wide pretrain-noise era, before the mean railed).

**Implication:** AWAC is advantage-weighted BC with no Q-ascent and near-uniform weights (mean
1.05). It can only clone what exists, and the deep gw-led skill has **zero success-labeled
examples**. It must be **generated**, not discovered. The good news: an in-repo scripted gw-led
RCCA heuristic already exists (`heuristic_policy_rcca.py`, Phase A/B/C), and `pid2586 ep9` is a
working template — gw-forward transit to ~230 mm in ~160 steps is already demonstrated; only the
last-20 mm delivery and slack-arrest are missing.

---

## 4. The user's mode-observer proposal: land it training-side, not as an observation

The user proposed a planned-path/wire-shape classifier that correlates with the critic's
privileged observations. Tested empirically (trained probes on the real buffer):

- **The mode signal IS extractable in-distribution:** an MLP on the policy-visible dims 0–96
  reaches AUC 0.90 for high-contact and R²0.65 on privileged contact (dim103), beating the E2 aux
  trunk's 0.50. Most of it lives in the LocalGuidance/planned-path block (dims 46–96): a *linear*
  readout there gets AUC 0.78. So the user is right that path/shape attributes carry the mode.
- **A deploy-time observer's transfer depends entirely on the training corpus (corrected).** The
  original agent test trained on the *seed slice only* and tested late — that setup collapses
  (contact regression negative, full-MLP AUC 0.45 below chance), but it is a mis-setup, not the
  intended usage. Re-measured the correct way (train on a broad corpus = seed + 3 early/mid chunks
  that already contain the contact regime, then test on a **chunk never seen in training**): AUC
  **0.72 (MLP) / 0.69 (logistic)** on unseen states — usable and correctly-oriented, vs 0.83/0.80
  in-corpus. So a frozen observer trained offline on a prior run (privileged-info distillation, as
  E2 aux already does implicitly) is a *usable-but-noisy soft signal* (~0.72), **not** the 0.45
  the first pass implied. Two caveats: (i) ~0.10 AUC degradation to unseen states remains — feed
  the logit as a soft feature, don't hard-switch on it; (ii) the deep gw-led transit states v3c is
  designed to create exist in **no** corpus yet, so an observer frozen from v3a-family data
  extrapolates there — plan to refresh it as v3c generates new behavior.
- **The decisive point: the trainer already has ground-truth privileged mode.** At update time the
  full 121-wide `states` (privileged dims included) are in hand (`sac.py:641`). So the mode signal
  needs **no observer at all** if consumed training-side — zero inference error, zero shift
  fragility. (Bonus: privileged dim118 is already byte-identical to visible dim90 — one privileged
  channel is de facto deployed.)

**Verdict — three consumption paths, ranked:**
1. **Training-side per-mode loss shaping (RECOMMENDED, ~18 LoC, zero obs change):** per-mode
   advantage normalization + contact-gated anti-rail penalty in `sac.py::_update_policy`. Directly
   attacks the measured mean-rail-on-contact and re-sharpens the near-uniform AWAC weights *within
   the contact stratum* where success/failure actually differ. Frozen-rule compliant, no cache/
   checkpoint surgery.
2. **Two-actor dispatcher (hold):** the user's stronger proposal. Still viable, ~350–500 LoC, no
   obs change, full buffer reuse — but option 1 delivers the same mode-conditioning at ~1/20 the
   code with no deploy-time inference risk, so it must be falsified first. If it's ever built, the
   dispatcher must use the shift-robust guidance-subset *linear* probe (cross AUC 0.78) or the
   blocked forward-signal (0.72), **never** the full-prefix MLP (0.45).
3. **Explicit obs dim (rejected):** approval-gated, cache-breaking (touches seed + 7 chunks +
   checkpoint input-layer surgery + shifts every privileged index), and dominated — the info is
   already linearly readable from dims the policy sees.

So the user's instinct (map planned-path/shape → privileged contact) is *correct and measured*,
but the highest-leverage place to spend it is inside the AWAC update, not the observation vector.

---

## 5. Recommended sequence (one path, decisively)

**STOP THE RUN NOW** (graceful: confirm the latest checkpoint + incremental chunk are flushed,
archive `logs_subprocesses/` and `diagnostics/snapshots/`, then `docker stop`). Justification:
7 flat evals now regressing on seed eviction; AWAC cannot amplify zero gw-led successes; σ already
maxed so explore yield won't improve; the rare valuable episodes are safe in on-disk chunks; and
the saturated CPU is the direct blocker for demo generation and screening.

### E1 — "v3c": mode-conditioned AWAC + scripted guidewire-led demos (the main shot)
The two defects are attacked together — demos supply the missing behavior, mode-conditioning lets
the trainer actually clone it where it matters.
- **Trainer (~18 LoC, `sac.py::_update_policy`, zero obs/reward/terminal change):** (1) per-mode
  advantage normalization — split the adv-std by `contact = states[...,103] > thresh` before
  `exp()`; (2) contact-gated anti-rail — make `action_mean_penalty` per-sample and weight it
  `where(contact, ~0.05, 0.005)`.
- **Demos (~50–100 LoC):** extend `heuristic_policy_rcca.py` with a Phase D deep-transit + delivery
  (template = `pid2586 ep9`), run on deep targets (path 170–284 mm), ~100–200 eps, `meta_is_demo=1`,
  small noise for diversity.
- **Data:** warm-start from the **pretrain** checkpoint (online ones are rail-locked); buffer =
  seed + demos + the on-disk chunks holding the 6 deep cath-led successes and 7 gw-led episodes;
  **protect demos from FIFO** (partition or periodic re-injection).
- **Gates:** G1 (data) the scripted heuristic itself solves ≥30% of deep targets *before* training
  — iterate the script, not the trainer, until true. G2 (learning, by eval 3) success >51/98 with
  ≥1 never-solved seed flipped AND eval-failure median final gw insertion >20 mm (vs 0.1–1.3). G3
  (pathology) contact-state mean-rail <60% (from 86%) and contact-stratum `awac_weight_p99p1` in
  5–20. **Abort:** eval 4 success ≤51 with gw-deploy unchanged → single actor can't unlearn the
  rail → go to E3. G1 unreachable after 2 script iterations → escalate for E7 (approval-gated
  gw-penetration reward).

### E2 — "v3b-transit": modified restore curriculum (only if E1 passes G2/G3 but deep success stalls)
Do **not** restore the raw stuck pool. Build the restore pool from (i) SOFA states captured every
~20 mm along the E1 demo transits (clean, on-path, depth-diverse by construction) + (ii) the 15–39
audited clean choke states after an offline self-distance/tier pre-screen and a **catheter-centric**
re-screen (fidelity = tracking3d RMSE + cath xtip; escapability = catheter retract_mm, ≥60-step
budget), early-RVA family capped ≤40%. Fix the screener tooling paths/mounts first (only safe once
the training container is stopped).

### E3 — two-actor dispatcher (fallback, only on E1 abort)
Prior option C (~350–500 LoC, no obs change, full buffer reuse). Deploy-time mode = guidance-subset
linear probe (cross AUC 0.78) or blocked forward-signal, never the full MLP. Does **not** graduate
to "now" — option-1 training-side conditioning must be falsified first.

**Total new code before any approval-gated change:** ~18 LoC trainer + ~50–100 LoC heuristic phase
+ a small buffer-protection patch. E7 reward shaping stays behind human approval, invoked only if
the scripted heuristic can't solve deep targets at all.

---

## 5b. LOOP/KNOT REWARD INVESTIGATION (2026-07-21) — is the coil strategy reward-driven?

User's claim: loops earn positive reward (decreasing d_rem), and gw-retract is learned as part
of the loop-navigation strategy. Tested via (a) reward-code read, (b) log-mining of 10 eval coil
windows + 388 RVA transit stretches + ~640k explore push-steps, (c) detector validation.

**Verdict: the strategy IS reward-driven — but via "indifference plus momentum," not positive
reinforcement.** Mechanism, precisely:

1. **Coiling is reward-FREE, not reward-positive.** Progress reward = 0.01×Δ(frontier-tip
   projection), telescoping, tip-only — the catheter body is invisible to the reward. Measured:
   during max-rate coil growth (Δcath 3.96mm/step, 178→898mm), mean reward = **−0.00003/step**;
   feeding 720mm into the coil cost −0.006 total. The grind phase (400+ steps) bleeds only
   −0.01…−0.22; the final −0.5…−1.0 cum is a **single −3.0 truncation spike at step 600** —
   discount-invisible at the decision point and behavior-invariant (it arrives whether or not
   the coil forms). d_rem does NOT decrease during coils (d_tgt oscillates 70-75mm).
2. **RVA wandering is penalized, but lightly** (−0.0155/step mean; 1/388 stretches net-positive;
   on_path=0 on 100% of RVA steps). Projection aliasing is real (+1.65mm mean fake proj_s per
   stretch, max +19.2mm) but never flips reward positive. A 172-step RVA stay cost −2.41. The
   "loops in RVA earn d_rem" part of the claim is refuted; the penalty being same-order as
   progress (and dwarfed by milestone bonuses ~+1.2) keeps it cheap.
3. **The asymmetric slack tax (buckle_reward.py): gw slack is priced (~0.00625/mm, symmetric —
   retract PAYS), catheter slack appears NOWHERE in the reward.** The one strategy that reaches
   deep (gw-lead through the choke, fold counters 15-114) is continuously taxed once slack
   builds; the pathological strategy is free.
4. **gw-retract during coils is actively COMMANDED (89-100% of grind steps, saturated −29.98)
   and EMERGED over training**: w0/seed8 went 82% gw-FORWARD in round 1 → 92-100% retract by
   rounds 2+, coil tripling while return IMPROVED +0.5 (approach compressed 272→43 steps,
   banking +2.5 vs +1.9). Gradient ascent demonstrably favored retract+slam.
5. **But the mechanical-instrumentality reading is REFUTED — and this strengthens the user's
   stiffness thesis**: measured across ~640k explore push-steps in the deep region, the catheter
   advances **2.4-3.1mm/step with gw>50mm inserted vs 1.25-1.46mm/step with gw<5mm** — the wire
   inside makes the catheter feed BETTER, not worse. gw-retract is not helping; it's a pure
   reward/momentum artifact.
6. **SMOKING GUN — the divergence point:** all 7 eval episodes ever to pass the choke (max
   proj_s 165.5-169.7) had **gw 53.6-756.7mm inserted**. Head-to-head same worker/round: seed1
   SUCCESS is step-identical to seed8 FAILURE until proj_s≈155 (both cum +2.50), then success
   pushes gw (+16..+30, gw→53.6mm, choke crossed, +3.0 bonus) while failure retracts gw and
   coils to 898mm. **The instantaneous rewards at the divergence step are IDENTICAL (+0.0305)**
   — no shaping term distinguishes the winning action from the losing one; only the bootstrapped
   +3 does, and AWAC (no Q-ascent, near-uniform weights) cannot propagate it back.

**Fix design (validated):**
- **Detector (zero cost, no geometry): `cath_slack = inserted_cath − proj_s`.** Validated against
  geometric far-arc self-contact ground truth on the stuck pool: **precision 0.97 / recall 0.96
  at cath_slack > 50mm.** Available every step from already-computed quantities.
- **Primary handler — catheter-slack channel in the buckle potential** (~10-20 LoC in
  buckle_reward.py + env5 feed): same potential-based math as the existing gw channel
  (unfarmable, symmetric — un-coiling pays back, restore-and-uncoil nets positive). This is
  the user's "increase the penalty when it loops," landed at the exact asymmetry. **REWARD
  CHANGE — approval-gated; user has proposed it, awaiting explicit go.**
- Secondary options: action-mask cath-forward at extreme slack (precedented by existing masks);
  filter pool captures at cath_slack>50 (fixes v3b pool quality in one stroke); SOFA device
  self-collision enable (root cause, heavy/risky).
- Complementary (from finding 6): the divergence point is reward-TIED — a per-step cath-slack
  price changes the tie (coiling becomes negative while gw-push stays free→positive), and the
  E1 demos give AWAC the winning action to clone. E7 (gw-lead bonus) remains the approval-gated
  escalation if needed.

### 5b-addendum (2026-07-21): tip-led prolapse loops (user's refined definition) — measured

User's refined loop = tip enters RVA, U-TURNS, comes back while the catheter keeps feeding
(prolapse), with the claim that reward gets LESS negative during the comeback (tip distance to
target falling) and that cath_slack cannot detect this (could be drape, not loop). Measured over
92 sustained RVA runs (3,949 steps, 4 workers):

- **The topology is REAL: 81/92 sustained RVA runs contain the deeper-then-back U-turn pattern,
  and 43% of comeback steps show `dins_cath>0` while the tip retreats — feeding-while-returning,
  the prolapse-loop signature.**
- **But the comeback is NOT paid — it is penalized MORE than entry**: going-deeper mean
  −0.0112/step (20% positive) vs coming-back mean **−0.0208/step** (14% positive); comebacks
  cost −37.6 total vs entry's −19.1.
  **DECOMPOSITION BY on_path FLAG (corrects an earlier mis-attribution):** the state-machine
  correction (off-arc accounting when off-path) EXISTS and covers 81% of RVA steps, and it is
  what produces the comeback penalty — NOT a projection-leak: on_path=0 comeback steps average
  **−0.0232 (9% positive)** because `arc_past` (off-path arc since divergence) KEEPS GROWING
  through the U-turn (median 5.2 during comeback vs 3.6 during entry) — the off-arc measure
  tracks off-path COMMITMENT (fed/routed arc to retrace), not the tip's Euclidean position, so
  a U-turning tip does NOT read as recovery and the design is not fooled. The 19% on_path=1
  leak steps (all near-ostium, proj_s 28–37mm, inside the radius-aware tolerance at the
  bifurcation shelf) are the only steps matching the user's "less negative on return" intuition:
  **+0.0014 mean, 60% positive** — real but negligible (~+0.001/step). Net: the U-turn has no
  meaningful reward incentive AND no counter-incentive that bites; it survives via
  indifference+momentum (−0.02/step is noise against the +2.5 banked approach and the invariant
  −3 terminal).
- **Why does the tip turn at all, then? Mechanics, not choice.** During comebacks the catheter
  is still commanded/fed forward on 43% of steps (50% retract — noise-mixed), with the guidewire
  essentially absent (median 15.9mm inserted). A floppy, wire-less catheter pushed into a
  side-branch ostium prolapses: the tip deflects off the RVA bend and folds back — buckling
  physics converts feed into a loop whose head swings around. The policy never "steers into"
  the turn; it slams forward with exploration noise and the geometry does the rest. Turning
  happens by physics; it persists by reward indifference.
- **Detector consequences (user's critique upheld in part):** cath_slack is a LATE/state
  detector and at low values cannot separate a discrete loop from slack draped through wide
  vessels; the tip-reversal signal is the correct ONSET detector for prolapse loops. Two loop
  MODES require two detectors: **mode B (tip-led RVA prolapse)** — onset-detectable per-step
  with zero geometry as `(off-path OR cur_branch≠planned) AND dins_cath>0 AND tip-progress
  trend<0` (43% of comeback steps show it), or deployment-grade from the leading 3-5 tracking
  points (tip tangent reversed vs branch/path direction); **mode A (trailing-shaft coil behind
  a tip FROZEN at the choke — the deterministic-eval deep-failure mode)** — has NO tip
  reversal, so the user's detector alone misses it; cath_slack>50mm (validated 0.97/0.96) or
  offline geometric self-proximity covers it. The handling design (cath-slack potential channel)
  stands, with the onset event as an additional trigger for any masking/penalty variant.

### 5b-addendum-2 (2026-07-21): is RVA entry a LEARNED bracing strategy? No — it is unlearned.

User hypothesis: for deep targets the policy *specifically learns to enter RVA* so the buckle
loop braces the catheter to keep advancing in RCCA (the loop replacing guidewire stiffness).
Tested three ways:
1. **Deterministic-policy RVA entries per eval round: 1,360 steps in the BASELINE eval
   (pretrain policy) → ZERO in every one of the 14 online eval rounds.** Online training
   ELIMINATED RVA entry rather than learning it — the opposite of instrumental acquisition.
2. **The coil does not advance the tip**: proj_s is frozen (Δ −1 to −4.5mm) across entire
   grind phases while the coil grows by hundreds of mm; the only advance mechanism is the
   stored-energy pop (+8mm, boundary flappers only).
3. **The coil is not needed for holding either**: the pretrain policy penetrated 17mm DEEPER
   (175.4mm) with ≤347mm catheter and no cap-level coils than the online policy ever reaches
   (158mm) with 898mm coils.
What survives: the "otherwise" branch of the user's story is exactly the observed eval
behavior (straight up RCCA → stall at choke → grind), with the coil as a passive byproduct
pooling at the RVA shelf, and explore-stream RVA prolapse as noise+physics. Design
implication: since loops are NOT instrumental, pricing cath_slack removes them at zero
performance cost — no ordering dependency between the slack penalty and the demo skill.

## 6. What changed in the user's model vs the data

| User's claim | Data verdict |
|---|---|
| Catheter folds into RVA and knots to gain stiffness | **Body: yes** (coil at RVA ostium, genuine self-knots). **Tip: no** in eval — tip freezes at the RCCA choke; the slack behind it knots. |
| It never uses the guidewire; guidewire should give stiffness | **Confirmed** — gw never deployed in failures; the 7 gw-led excursions prove gw-lead reaches far deeper. |
| Stops due to strategy, not tight radius | **Confirmed** — passes r=2.2 fine; wedges at r=2.0 only for lack of gw lead. |
| Fatal strategy learned at pretrain | **Confirmed** — coiling present at pretrain; mean-rail-on-contact grows from there. |
| Stuck pool is unrecoverable / undiverse / restore-suspicious | **Confirmed on all three** for the choke states; screener catches none of it. |
| Separate the two modes (2 actors, or a planned-path→privileged observer) | **Right instinct, better landing:** mode is real and extractable, but consume it **training-side** (~18 LoC) first; two-actor is the fallback, obs-flag is dominated. |
