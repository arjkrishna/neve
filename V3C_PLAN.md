# v3c — approved plan (2026-07-21)

**Approved by user:** reward pair (tip-average progress + catheter-slack channel) + stop of
v3a run 2 (done: stopped at eval15/explore 3.56M, all artifacts flushed). Trainer changes
ship as **default-off CLI flags** so run 1 measures the reward changes in isolation.

## Diagnosis this plan answers (all measured — see FORENSIC_*.md)
Deep targets (46/98, path >170mm past the r=2.0 choke) fail because the policy never deploys
the guidewire; the reward made that optimal: progress paid the frontier tip only (parked gw
free), gw slack taxed / gw retract paid (buckle channel), catheter slack priced nowhere
(coils free), and at the measured divergence step (gw-push vs gw-retract at the choke) both
actions earned identical reward. AWAC (advantage-weighted BC, no Q-ascent) cannot propagate
the +3 terminal back through a reward tie.

## Changes

### A. Reward — tip-average progress (`eve/eve/reward/arclengthprogress.py`)
- New params: `tip_mode: str = "frontier"` (`"frontier"`|`"avg"`), `avg_gw_weight: float = 0.5`.
  Default = byte-identical legacy behavior.
- `avg` mode: per-device tip arcs. Device tip 3D = combined-polyline point at
  arc==inserted_dev measured from the PROXIMAL end (tracking3d is distal-first). Project each
  onto the planned path; trailing tip arc clamped to [0, s_frontier]. Effective arc =
  w·s_gw + (1−w)·s_cath. On-path: r = pf × Δ(effective arc). Off-path: existing frontier
  off-arc logic unchanged. All _prev trackers updated unconditionally (no re-entry credit).
- Effect: parked gw halves pay; gw-push at the choke earns +pf·w/mm; gw-retract goes negative;
  telescoping preserved (round trips net zero).
- ConfigHandler: params stored as same-named attributes.

### B. Reward — catheter-slack channel (`training _scripts/util/buckle_reward.py` + env5)
- `buckle_potential(slack_mm, contact_mm, cath_slack_mm=0.0, w_cath_slack=0.0)`;
  new consts CATH_SLACK_DEADBAND_MM=15, CATH_SLACK_CAP_MM=150. phi stays bounded, capped
  inputs, delta/potential form (unfarmable; un-coiling pays back; restore-and-uncoil positive).
- env5 computes `cath_slack = inserted_cath − s_cath` (same helper as A) and passes it with
  the new `--cath_slack_coef` (default 0.0 = OFF = byte-identical).
- Detector context: cath_slack>50mm ≈ knot (precision 0.97/recall 0.96 vs geometric truth).

### C. Trainer — default-OFF CLI flags (`eve_rl/eve_rl/algo/sac.py` + arg wiring)
- `--awac_mode_adv_norm` (store_true, default off) + `--awac_contact_thresh` (default 0.009):
  per-mode (contact = states[...,103] > thresh) masked mean/std advantage normalization in the
  E1b block; fallback to global when a stratum has <8 samples.
- `--contact_mean_penalty` (float, default 0.0 = off): action-mean anti-rail penalty becomes
  per-sample and contact-weighted (contact steps pay `contact_mean_penalty`, others keep
  `action_mean_penalty`). 0.0 → existing global code path untouched.
- ConfigHandler trap: every new __init__ param stored as an attribute with the same name.

### D. Cache — reward-recompute-complete harvest (+ guard)
- Fresh harvest REQUIRED: old seed's rewards are stale under A+B and cath-tip arc is not
  recoverable from stored obs (only a chord-biased approximation via privileged 11-13).
- New cache stores per-step: s_gw, s_cath, ins_gw, ins_cath, raw gw_slack, raw contact,
  on_path → all future reward tweaks become offline recomputes (last forced re-harvest).
- Reward-version guard extended: meta_progress_tip_mode, meta_avg_gw_weight,
  meta_cath_slack_coef checked at load, fail-fast on mismatch (same pattern as
  meta_buckle_coef).

### E. Launchers
- `launch_rcca_heatup_v3c.sh`: fresh harvest under the new reward (A+B flags on).
- `launch_rcca_procedural_v3c.sh`: **identical to v3a except** the reward flags
  (`--progress_tip_mode avg`, `--cath_slack_coef 0.5`) and the new heatup cache path —
  v3c vs v3a then differs in reward only (clean attribution). Trainer flags present but OFF
  (commented for run 2). Guard timeouts keep the 5400/7200 values. Mount list updated for the
  new/changed files.

### F. Monitoring & gates
- Monitor pass v3c: adds cath_slack stats (inserted[1]−proj_s from STEP logs), median gw
  insertion in eval failures, gw-led excursion counter, tip-reversal onset counter.
- GATES (eval2–3): eval-failure median gw insertion >20mm (v3a: 0.1–1.3); cath_slack in
  failures collapsing (v3a: ~740mm); ≥1 never-solved seed flipped by eval3; Quality ≥ its own
  pretrain baseline. Escalation order if gw deployment rises but no seed flips by eval4:
  demos (delivery micro-skill) → E7 → two-actor. Trainer flags (C) turn on for run 2 to
  measure their marginal effect.

## Execution status (2026-07-21, post-review)
- [x] v3a run 2 stopped; monitors/cron retired
- [x] A implemented  - [x] B implemented  - [x] C implemented (CLI, default-off)
- [x] D guard (in-cache per-step extras deferred to v4; enriched STEP logs are the
      offline recompute source — carries cath_slack, inserted, proj_s per step)
- [x] E launchers    - [x] unit tests (25/25 in-container)
- [x] 4-reviewer adversarial review + verification — ALL findings dispositioned:
  - BLOCKER (confirmed): off-path avg-tracker rebaseline created a farmable pump
    (+0.125/13-step cycle, proven numerically) → FIXED: tracker freezes off-path,
    rejoin nets the excursion once; reviewer's counterexample now nets 0.0000 and
    is a permanent regression test (suite section 5).
  - BLOCKER (confirmed): BenchAgentSynchron didn't accept the 3 trainer kwargs
    (TypeError at every launch) → FIXED: params added + forwarded; signature smoke
    test added (suite section 6).
  - HIGH (confirmed): --resume unguarded for reward version → FIXED: 4 stamps
    written into replay_state.npz on incremental save + resume guard fails fast on
    mismatch (absent stamps = legacy values, old-run resumes stay valid).
  - MEDIUM fixes: windowed projections (30mm anchor + fallback escape) for BOTH
    new projections (hairpin limb-flip noise); missing-seed preflight in the v3c
    launcher; .gitattributes (*.sh eol=lf); env_version guard for the
    contact-indexed trainer levers; multi_target+reward-pair hard refusal.
  - LOW fixes: reset-time avg baseline (step-1 motion priced); contact-penalty
    switched to pre-tanh clamp(±6) form (live gradient on already-railed samples);
    harvest-launcher stale comments.
  - Accepted (documented, no action): avg fallback re-entry one-step mispricing
    (rare exception path); per-stratum centering counts padded rows in
    episode-mode batches (v3c runs step mode); stamps record CLI values even when
    inert (avg_gw_weight compared under frontier mode — keep flags consistent).
  - Reviewer-verified clean: launcher attribution diffs, all mounts exist,
    env-stamp propagation through spawn, guard coverage, stuck-pool wiring,
    reward pair reaches train+eval+harvest envs, per-step cost ~0.3ms (no caching
    needed), STEP-line insertion parser-safe (+2.5% log volume), frontier mode
    byte-identical, eval/checkpoint selection quality-gated only.
- [ ] **user go** → harvest (~3h) → pretrain-baseline → online run 1 (reward-only)
