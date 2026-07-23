# Crunch-Pool Strategy — classifier-driven two-mode sampling (from machine-2 v3a/v3b learnings)

2026-07-22. Data: v1b's 15 retained buffer chunks (0→3.55M, full 125-dim obs) + 28 era-spread worker logs (~250 eps).

## Machine-2 inputs (user-reported)
- v3b stuck-lane harvest FAILS as designed: truly-stuck states are knotted/looped, unrecoverable without 100s of retraction steps. Target the states BEFORE — crunch-entry / recover-from states, not stuck states.
- Crunch-step classifier from nearby planned-path params: ~90% on machine 2. Top transfer features (permutation importance): gw_slack(89) 0.237 ≫ cath_offset_z(77) 0.123, arc_past_daughter(58), gw_cath_gap(78), local_radius(93). Top-15 AUC 0.788 hard-transfer; same approach can recover privileged force/contact for the STUDENT.
- Crunch signature: both devices driven, tips close.

## v1b verification results
1. **Path-geometry → crunch-signature classifier: AUC 0.92–0.95 on our chunks** (5 features: 93,58,89,77,78). The ~90% claim HOLDS here.
2. **Deployable-features → high-contact (privilege recovery): AUC 0.65–0.77** with only 9 features (machine-2 got 0.788 with top-15) — viable for the P2b student aux head; needs fuller feature set + history. Weights: arc_past(58) dominant, then cath_offset_z(77), local_radius(93).
3. **Crunch steps have LOWER contact than free steps** (6–10% vs 10–11% high-contact) — catheter-close support PREVENTS buckling. The signature marks *controlled* passage mechanics, physically as expected.
4. **Era density (user's eval3/5 hypothesis): CONFIRMED for eval3, not eval5.** Crunch steps/success-episode: early 24.6 → **eval3-era 41.1 (50% of successes contain ≥3-step crunch passages)** → eval4 10.9 → eval5 10.9 (14%) → mid 7.0 → plateau 13.8 (46%). The 81.6% eval3 crest coincided with peak dual-device crunch work; the eval5 crest did not (different mechanism — cleaner avoidance).
5. **Raw signature alone marks GRINDING, not success: failures have 4–8× more crunch-signature steps than successes in EVERY era** (e.g. eval3-era: 193/failure-ep vs 41/success-ep). ⇒ Pool-2 MUST be success-conditioned ("crunch steps that lead to successful passage"), exactly as the user specified — an unconditioned crunch lane would oversample grinding.
6. **Feature-saturation warning**: normalized gw_cath_gap (flat 78) rails/goes constant in later-era chunks (devices consistently close+deep) — percentile logic on it degenerates; classifier should use raw-log or multi-feature redundancy.
7. Dual-device engagement is high and RISING all run (both-engaged 54%→74%) — the policy does use both devices; the deficit is WHERE/WHEN, not whether.

## Enhanced strategy (supersedes E3 thresholds + fixes E4 harvest)
Two axes × two pools, all machinery mostly existing:
- **Pool-2 membership (push-time, per transition)**: `is_crunchpass = crunch_sig(obs) AND episode.success` — computable at push because push() receives the whole episode (mirror of is_clean). crunch_sig = the 5-feature logistic (weights fixed from offline fit; AUC ~0.94) or simple rule. Reuses the E3 third-lane SumTree plumbing (stuck_fraction → crunchpass_fraction); only the membership function changes.
- **Pool-1**: rest (or non-crunch). Sampling: `--crunchpass_fraction 0.15–0.25` of each batch.
- **E4 restore-harvest fix**: capture restore checkpoints at CRUNCH-ENTRY (first crunch-sig step of an episode, pre-knot), NOT at stuck-detector states (knotted, unrecoverable). Re-harvest on machine 2 with this criterion.
- **Mode conditioning / GMM (Axis-1)**: the classifier output IS the mode bit — feed as obs feature first (cheapest); gate GMM components by it only if conditioning saturates. Two-actor MoE re-filed as deferred last-resort (not rejected).
- **Student (P2b)**: aux head predicting privileged contact/force from deployable top-15 + short history (AUC 0.65–0.77 with 9 features already; machine-2 0.788).

## Costs
- crunchpass lane: ~1 day (membership fn + flag plumbing + launcher flag; SumTree lane exists).
- classifier: offline fit on retained chunks (done in prototype here), hardcode weights.
- E4 re-harvest criterion: small change to the machine-2 screener.
- GMM: later, only on evidence conditioning is insufficient.
