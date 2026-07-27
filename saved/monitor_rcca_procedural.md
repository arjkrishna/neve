# RCCA P2 Teacher v1 — Residual-on-Heuristic Monitor (ACTIVE)

**Run:** `2026-07-16_235715_rcca_p2_teacher_v1` (container `rcca_p2_teacher_v1`, launched 2026-07-16 ~17:57 EDT — dir stamp is container TZ +2h... verify with docker inspect StartedAt on first pass; branch `rl_improv_18_p2` @ 3b71383).
**Design:** policy = RESIDUAL on CenterlineFollowerHeuristic (raw-unit composition in env5.step, clip to limits); privileged ACTOR (policy consumes full 125-dim obs incl. 24-dim tail + 4-dim heuristic intent); sac + LayerNorm critics + entropy-free backup (P1-validated); PER + balanced 0.3; heatup 20k @ 0.3 band = heuristic+noise buffer; pretrain 0; **baseline eval = pure heuristic H₀** (runner hoist). shm 30g. See RL_IMPROV_18_P2_DESIGN.md.
**Mandate:** 2-h read-only passes (cron 5b7885f8, :41 odd hours). Gates: (1) explore success STARTS near H₀ (≈0 = composition broken → ALERT); (2) every eval vs H₀ and vs v2 6.1/30.6/49.0/30.6; (3) Q-divergence watch; (4) alpha regulating (P1 limit-cycle watch); (5) mem slope (P1 OOM at 87% of cap).

### Eval5 event — 2026-07-17 17:29 EDT — **EVAL5 = 45.9%** — 🔔 PRE-REGISTERED GATE FIRED

**Curve: H₀ 45.9 → 48.0 → 49.0 → 46.9 → 45.9 → 45.9. Reward: 0.887 → 0.880 → 0.707 → 0.593 → 0.556 (4-eval monotone decline).** Eval5 window: 162 stuck events → soft 11 (6.8%), hard 2, **grind 97 (60%)**, unrec 52. Gate condition (eval5 ≤49% AND grind >50%) is MET. Conclusion of the P2a-v1 bracket: residual-on-heuristic + privileged actor is STABLE at the heuristic floor (no v2-style collapse across 5 evals — the shape claim holds) but does NOT compound above it under the current reward — the policy has no paid incentive to convert grinds into soft recoveries, and the alpha limit cycle (P1+P2 recurring, ~24k period) plausibly erodes what it samples. Run left RUNNING (21% of budget; read-only mandate) pending user decision on the escalation package (P2a-v2: reduce-slack potential + stuck-restore curriculum + tighter alpha ceiling; relaunch required). 

### Eval4 event — 2026-07-17 12:53 EDT — **EVAL4 = 45.9%** (45/98, reward +0.593, explore 1.02M) — ⚠️ drifting back to the floor

**Curve: H₀ 45.9 → 48.0 → 49.0 → 46.9 → 45.9 — peaked at eval2, gentle decline since, now exactly AT H₀. Reward declining monotonically 3 evals: 0.887 → 0.880 → 0.707 → 0.593** (partial progress in failures is shrinking — this is a trend, not noise, even though quality wiggles are ±2 episodes). Composition: eval4 window 160 events → soft 13 (8.1%, RETREATED from eval3's 11.6%), hard 0, grind 94 (59%), unrec 53 (rising). The soft-share climb did not hold. Health: u=520k, Q bounded, nonfinite 0, alpha 0.22 (cycle ongoing), OOM=false, mem 80%.
**Reading:** no collapse (v2's shape still beaten), but the residual is not compounding — post-eval2 the composed policy is drifting mildly WORSE than the script on reward while staying at script-level success. Consistent with the alpha-cycle re-blur hypothesis and/or residual drift without payoff. **The pre-registered eval5 gate (≤49% AND grind >50% → escalate to reduce-slack potential + stuck-restore curriculum, user approval + relaunch) is now very likely to fire (~17:40 EDT).** Candidate mechanism note for the escalation design: consider a tighter alpha ceiling too (the P1/P2-recurring ~24k-update limit cycle as the consolidation-killer).

### Eval3 event — 2026-07-17 08:07 EDT — **EVAL3 = 46.9%** (46/98, reward +0.707, explore 754,564) — NO COLLAPSE ✅, still heuristic-bound ⚠️

**Curve: H₀ 45.9 → 48.0 → 49.0 → 46.9.** The v2 comparison at the same index: v2 collapsed 49.0 → 30.6 here; this run wiggled −2 episodes. **The eval3 stability bar is PASSED — the P2a design beats v2's SHAPE.** But all four evals sit within ±1.5 episodes of H₀: the residual is stable, not yet statistically above the script. Reward dipped 0.880 → 0.707.
**Composition (the leading indicator, still moving right): soft recoveries per eval window: 13 → 13 → 19 (8.6% → 8.2% → 11.6% of stuck events)**; explore's newest window 12.0% soft (highest yet; was ~7%); hard spiked at eval2 (8) then receded (1) — the crude→refined progression. Grind still dominant (59%). Retract-depth succ/fail 0.7/2.2mm.
Health: u=380k, Q bounded, nonfinite 0, alpha mid-cycle 0.27, OOM=false, mem 80%.
**Verdict: OK/WATCH — continue.** Only 12.6% of the 6M explore budget spent; no collapse; the behavioral metric climbs under a flat headline. **Pre-registered decision gate: if eval5 is still ≤49% AND grind share still >50%, escalate to the deferred pieces (reduce-slack potential + stuck-restore curriculum — user approval required, fresh-run relaunch since reward changes invalidate the buffer).**

### Eval2 event — 2026-07-17 04:02 EDT — **EVAL2 = 49.0%** (48/98, reward +0.880, explore 536,204)

Curve: **H₀ 45.9 → 48.0 → 49.0** — monotone but creeping (+1 episode per eval; +3.1pp over H₀ is still ~noise-band edge). Sits exactly AT v2's all-time ceiling at eval2 (v2: 30.6 at the same index). Reward flat. **Behavior composition (the sharper metric):** eval2 window 158 stuck events → soft 13 (8%), **hard 8 (up from 0 at eval1)**, grind 89 (56%, down from 61%), unrec 48. Explore trend: soft flat ~7% (NOT fading — unlike v2's 86→58→28), **hard rising 0.9→2.1→2.2→5.4% across windows** — the residual is learning blunt large retractions before fine soft ones. Retract-depth succ/fail 0.7/2.6mm. Reading: not a breakout, not a stall — recovery repertoire is shifting under a flat headline. Decision point unchanged: eval3 (v2's collapse index) — holding ≥49% beats v2's SHAPE; a soft/hard share climb with success climb = design working; flat 49% + grind-dominance persisting = case for the deferred reward/curriculum pieces.

### Pass 3 addendum — 23:22 EDT — **EVAL1 = 48.0%** (reward +0.887, explore 267,982)

vs the bars: **H₀ 45.9% → +2.1pp** (2 episodes of 98 — inside the ~±5pp binomial CI, so honestly "at the floor, slightly above"); **v2 eval1 6.1% → +42pp at the same eval index**; v2's ALL-TIME ceiling 49.0% → within 1pp already. Reward +0.636 → +0.887 (+0.25: deeper partial progress even in failures). Post-eval incremental save clean (+267,982). The deterministic policy has not (yet) consolidated clearly above the heuristic — consistent with the explore buckets and the alpha-cycle re-blur suspicion — but the run sits at v2's ceiling at ONE eval instead of three, with 5.75M explore steps of headroom. Next bars: eval2 > 49% decisively, and no eval3 regression.

### Pass 3 (p2_v1) — 2026-07-16 23:03 EDT (~5 h in, u=116,981, explore 250,250) — ✅ OK; eval1 IMMINENT

1. **Liveness ✅** — Up 5 h, Restart=0, OOM=false, mem 17.0/23.5 GiB (72%, still easing). Two Tracebacks in window are **benign**: `multiprocessing resource_sharer ConnectionResetError` teardown noise from worker restarts; STEP flow continuous through both, no watchdog/deadline trips.
2. **Q-DIVERGENCE WATCH ✅** — u=117k: losses O(0.002–0.011), q1_mean cycling −0.12 ↔ +1.53 bounded, nonfinite 0/0/0. Alpha limit cycle continues (~24k period: 0.024→0.62→0.027→0.49→0.024), entropy 1.2 ↔ 2.7. Unchanged pattern; consolidation question goes to eval.
3. **Explore success:** buckets 59% → 45% → 44% (partial). Hovering AT-to-above H₀ 45.9%, no collapse; the 59% spike hasn't consolidated into a sustained climb yet — the alpha-cycle re-blur suspicion stays live. Deep-RCCA STEP lines show entries_passed=1/daughters_passed=1 with positive cum_rewards ~2.0 and active buckle_phi recovery — the wire IS working deep territory.
4. **Eval1 IMMINENT:** explore 250,250 has crossed the 250k threshold — eval fires at the end of the current ~40-min explore cycle; result expected ~23:35–23:45 EDT. Bars: H₀ 45.9% (learning proof), v2's 49% (ceiling), then eval3-stability (v2 regressed to 30.6%).
5. GPU 0–24% (cycle-boundary at pass time); realized UTD 0.47.

**Verdict: OK.** Machinery clean; behavior at the heuristic floor with deep excursions; the run's first real judgment lands within the hour.

### Pass 2 (p2_v1) — 2026-07-16 21:03 EDT (~3 h in, u=60,990, explore 150k) — ✅ OK, explore climbing ABOVE H₀

1. **Liveness ✅** — Up 3 h, Restart=0, OOM=false, guards silent. Mem **17.3/23.5 GiB (74%, DOWN from 84%)** — pressure eased; keep watching. GPU 24% (updates running).
2. **Q-DIVERGENCE WATCH ✅** — 61k updates: q-losses O(0.003–0.08), q1_mean bounded −0.17 ↔ +1.96, nonfinite 0/0/0, clamp ≤0.7%, grad norms sane. LayerNorm holding again.
3. **Alpha limit cycle RECURRING (P1 pattern, ~24k-update period):** 1.0 → 0.02 → snap 0.81 → 0.017 → snap 0.68 → 0.025 now; entropy 1.1 ↔ 2.7 around target 1.0. Same controller hunting as P1 — but this time there's a competence floor under it. WATCH whether the snap-backs cap consolidation above H₀ (the P1 concern, now testable against a nonzero baseline).
4. **Explore success — first sign of LEARNING above the heuristic:** post-heatup buckets 47% (incl. the eval block) → 40% → **59%** → 4/13 partial. Bucket 3 clears H₀ 45.9% AND the noisy-heatup 54%. Early but exactly the signature we want: residual gains on top of the floor.
5. **Evals:** none since baseline (explore 150k of 250k; eval1 in ~1–1.5 h). Realized UTD 0.41 (same trainer ceiling as P1 — expected).
6. **Verdict: OK.** Next pass: eval1 vs H₀ 45.9% / v2's 49% — the run's first real bar.

### Pass 1 (p2_v1) — 2026-07-16 19:16 EDT (~1.3 h in, explore just started) — ✅ OK, strong start

**H₀ BANKED: pure-heuristic baseline = 45.9%** (45/98 held-out seeds, reward +0.64, eval at explore=0 — the runner hoist worked exactly as designed). This is the run's permanent null hypothesis; note it is just below v2's best-ever eval (49.0%) BEFORE any learning.

1. **Liveness ✅** — Up ~1.3 h, Restart=0, **OOM=false**, zero guard trips. Mem 19.7/23.5 GiB (84%) — the docker-stats cap is host RAM (the 30g was /dev/shm; the P1 OOM pressure pattern still applies) — WATCH the slope. GPU 9%/1.5 GB (explore phase, no updates yet).
2. **Widths verified ✅** — runner.yml: n_observations **125 on all three nets** (q1/q2/policy) = privileged ACTOR live (policy consumes the full obs incl. 24-dim tail + 4-dim heuristic intent).
3. **Heatup (28 eps, heuristic + 0.3-band noise): ~54% success** — vs H₀ 45.9% deterministic. The diversity gap (noise +8pp) exists but is modest; the CURRENT controller is far stronger than the harvest-era "our only success" history (off-path retract restore + relax truncations + procedural variation). **Honesty flag for the paper doc:** PAPER_PLAN's "deterministic ≈0% vs noise ~54%" contrast describes the OLD controller/conditions; the new measurement is 45.9% vs ~54% — update §1/C3 before drafting.
4. **Explore STARTS AT H₀ ✅ (the composition gate):** ~80 successes in ~181 post-heatup episodes (~44%, mixing the 45.9% eval block + first explore cycle) — the residual policy + exploration noise sits right at heuristic level, NOT at 0. Residual composition confirmed working end-to-end.
5. **Losses CSV:** not yet (first explore cycle has zero updates by design). Q-health checks begin next pass.
6. **Eval protocol ✅** — baseline eval wrote incremental chunk (+26,376 transitions = 20k heatup + eval), no post-eval stall.
7. **Composed actions** (STEP cmd_action): mixed magnitudes ±5–18 mm/s with sign diversity — heuristic-shaped with noise on top, as expected this early.

**Verdict: OK — the P2a premise is validated at the gate: the run starts at 45.9% instead of P1's 0%.** Success bar for the run: beat H₀ 45.9% (null), then v2's 49% ceiling, then the stuck-grinder regression pattern past eval3. Next pass: first update cycle Q-health + whether eval1 > H₀.

### Pass 0 (p2_v1) — 2026-07-16 ~18:0x EDT — LAUNCH ✅
Smoke-validated in-container before launch (obs 125/ordering, heuristic drives wire, once-per-state cache stable, composition clipping, legacy 121 unchanged). Startup echoes confirm `[P2] residual-on-heuristic ON: scale=1.0 heur_action_obs=True` and `[P2] privileged ACTOR`. runner.yml not yet written at pass time — first cron pass must verify policy n_observations=125 and record H₀ from the baseline eval.

---

# RCCA RLPD v1 — Paradigm-Pivot Monitor (CLOSED 2026-07-16 ~17:3x EDT)

### Run closure — 🛑 KILLED by user decision (b)+(d); cron b2730183 deleted

**Final state:** ~6 h, u≈226k+, explore ≈580k+, evals 0.0 / 0.0 (v2 ref 6.1 / 30.6). Stopped gracefully (`docker stop -t 60`), container removed; run dir retained with checkpoints (300988, 540663), incremental buffer (822,973 transitions in 2 chunks + state), full CSVs and step logs — resumable via `--resume` if ever revisited.
**Closure verdict (the clean negative result):** RLPD-at-low-fidelity (twin critics, realized UTD 0.39 vs paper's 20, no ensemble, uniform 50/50 seed sampling, no pretrain) does NOT bootstrap deep-siphon dual-device navigation from scratch at our sim throughput. Machinery was flawless throughout: LayerNorm held Q bounded (no divergence in 226k updates — the recipe's stability claim VALIDATED), incremental saves + deadlock guards clean, self-recovered from one in-container OOM kill. The failure is signal density: start-state deterministic mean never left random-init (0.001) while weights kept updating — gradient exists, too dilute to build behavior. Alpha limit-cycled (floor↔~0.5+, ~24k-update period) around target_entropy 1.0 the entire run.
**Superseded by:** P2 (privileged-actor teacher + residual-on-heuristic + student distillation) per RL_PARADIGm_ROADMAP.md — implementation starting on a fresh branch. RLPD components (LayerNorm critics, symmetric sampler, entropy-free backup) remain in the codebase default-off; LayerNorm critics in particular are worth carrying into P2.

**Run:** `2026-07-16_172751_rcca_rlpd_v1` (container `rcca_rlpd_v1`, launched 2026-07-16 17:27 EDT, branch `rl_improv_17_rlpd` @ 7fc1efb)
**Launcher:** `launch_rcca_rlpd_v1.sh` — `--algo sac` + RLPD recipe: critic LayerNorm (q1/q2 only), no entropy backup, `rlpd_offline_fraction 0.5` (seed = offline lane, is_demo=True), UTD 1.0, **pretrain 0** (no pretrain checkpoint0 — probe baselines against eval1), alpha rails [−5.0, 0.0], target_entropy 1.0, 16 procedural workers.
**Seed:** same `saved/rcca_proc_heatup/seed.npz` (480 ep / 282,310 transitions, md5 54fe108c…, buckle_coef 0.5).
**Mandate:** every 2 h read-only pass (cron b2730183, :23 odd hours), log here, MAKE NO CHANGES.
**Gates:** (1) Q-divergence watch — critic losses O(1), q-means not monotone-up (the BC-free failure LayerNorm must prevent); (2) alpha ∈ [0.0067, 1.0] and MOVING (floor-pinned >50k updates = ALERT); (3) explore success must climb from ~0 by eval1 or "RLPD transfer failing — reassess, don't tune"; (4) beat v2's 6.1 / 30.6 / 49.0 / 30.6, especially past eval3. Inert columns under sac: awac_weight_*, action_mean_penalty (do not alarm).

### Pass 3 (rlpd_v1) — 2026-07-16 16:53 EDT (~5.5 h in, u=226,044, explore 579.7k) — 🔴 REASSESS TRIGGER FIRED: eval2 = 0.0%

**Headline: eval2 quality 0.0%** (reward −7.1434, explore 540,663, 16:32 EDT) — **second consecutive 0.0% eval with flat explore = the pre-registered reassess trigger from Pass 2 has formally fired.** Per mandate no action taken; run continues pending user decision. Curve so far: 0.0 / 0.0 vs v2's 6.1 / 30.6.

1. **Liveness ✅ with one event:** container Up 5 h, RestartCount=0 — but **State.OOMKilled=true**: the cgroup hit its 23.5 GiB cap and the kernel killed an in-container process (mem 20.4→19.0 GiB across the window; no host dmesg access to name the victim). The system self-recovered: trainer updating (CSV mtime current), 16 distinct worker pids stepping (high pids 1478/1195 = normal post-eval restarts), eval2 + checkpoint540663 + incremental chunk (+239,675 → total 822,973) all written cleanly, no watchdog/deadline trips. Mem now 19.0/23.5 GiB — pressure persists; a repeat kill could hit something load-bearing (buffer subprocess) next time.
2. **Q-DIVERGENCE WATCH ✅** — u=226k: losses O(0.004–0.035), q1_mean cycling −0.31 ↔ +1.90 (bounded, alpha-correlated as before), nonfinite 0/0/0, clamp ≤1.2%.
3. **Alpha limit cycle — continues, mild damping at the low end:** floor visits every ~24k updates (0.0067@156k, 0.0084@180k, 0.0093@204k); entropy swing narrowed to 1.15–2.71 (was 0.59–2.75). Still not converged to the target-1.0 band.
4. **Explore success FLAT:** buckets since Pass 2: 0/0/2/1% (last 94-ep bucket at −7.14 = the eval2 block). Cumulative ~10 successes / ~1,000 explore episodes (~1%). No learning signal reaching behavior after 226k updates / 580k explore steps.
5. **Probe:** start-state mean|tanh(μ₀)| **still 0.001 at u=209k** (= random-init level; nothing ever came up). P(retract)=0.000 all slack bins. The deterministic policy is inert at episode starts — all exploration success is σ-noise. (FREEZE-ALERT verdict remains vacuous — nothing to freeze.)
6. **GPU/mem:** GPU 19%/1.5 GB; container 19.0/23.5 GiB post-OOM-kill (see 1).

**Verdict: 🔴 ALERT (status, not machinery).** Infrastructure is doing its job (Q bounded, saves clean, guards silent, self-recovery from OOM); the learning configuration is not converting gradient work into behavior — realized UTD 0.39 vs paper's 20, no pretrain, uniform offline sampling, alpha hunting. **Decision needed from user:** (a) let it run to eval3 (cheap; but two data points say the trajectory is flat), (b) stop and close the RLPD-at-low-UTD bracket, (c) RLPD-fidelity retry (ensemble + real UTD ≥ 4 via trainer-throughput work, mem cap raised), or (d) skip to P2 (privileged-actor teacher) per roadmap. Recommendation: (b)+(d) — the run's own evidence (start-state mean never left 0.001 while weights demonstrably update) says the gradient signal exists but is too dilute at this UTD; P2's teacher-student path does not depend on that scaling.

### Pass 2 (rlpd_v1) — 2026-07-16 14:53 EDT (~3.5 h in, u=142,388, explore 380k) — ⚠️ GATE TRIPPED: eval1 = 0.0%

**Headline: eval1 quality 0.0%** (93 eps, reward −7.14, 19.5 min eval @ 47.2 steps/s, explore 300,988, 14:12 EDT). Per the pre-registered gate: **"RLPD transfer failing — recommend reassess, don't tune."** No action taken (read-only mandate); run continues to eval2.

1. **Liveness ✅** — Up 3 h, RestartCount=0, zero Traceback/watchdog/trainer-deadline in window. **Post-eval NO stall** (the v1/v2 killer): incremental save `+583,298 transitions` (= seed 282,310 + online ✓, chunk 0→583298 + replay_state.npz written 14:12), CSV advancing normally after eval.
2. **Q-DIVERGENCE WATCH ✅** — u=142k: q-losses O(0.001–0.05), q1_mean bounded −0.32 ↔ +1.77 (cycles with alpha, not monotone), nonfinite 0/0/0, clamp_fraction ≤1.3%.
3. **Alpha limit cycle — UNDAMPED, now touching the floor:** period ~24k updates, alpha 0.7 → 0.0067 (floor hit at u≈104k) → snap to 0.2–0.5; entropy 2.7 ↔ 0.6 around target 1.0. Correlated Q-mean swing (high-Q when sharp, negative when re-noised). Suspected mechanism for explore non-consolidation: each alpha snap-back re-blurs the policy. Not the v1 pathology (no long floor-pin, no std saturation) but clearly not converging either.
4. **Explore success FLAT:** per-100-ep buckets 1/0/1/3/2% (bucket 6 = the eval block itself, −6.96 avg reward); post-eval bucket 0/96. No climb across 380k explore steps.
5. **Probe (banked baseline, interpret with caution — script is v3a-calibrated):** start-state mean|tanh(μ₀)| = **0.001 at BOTH** policy_10000 (u=0) and policy_310000 (u=134k) — the deterministic mean at start states never came up from random-init ~0 (nothing to freeze yet; auto FREEZE-ALERT verdict is vacuous here). All 36 policy_*.pt snapshots have DISTINCT md5s and grad_norm_policy is nonzero (0.04–0.13) — weights update, but the start-state function hasn't moved. Eval reward −7.14 (well below the ~−1.2 idle floor) shows the policy DOES move mid-episode and fails — not a global zero-mean freeze. Retract-vs-slack P(ret)=0.000 all bins (valid — generic measurement). **Aux-R² caveat (correction):** the probe scores the v3a label set {97,98,102,103}, but this run trains the LEGACY set {99,100,102,103} — so the −1.7/−2.3 numbers score heads 0/1 against labels they never train on (meaningless here); only the contact pair {102,103} is a real readout (−51/−2.2 = the known contact scale-starvation, kept as control by design). Probe script is the E8/v3a kit reused with only the run-dir pattern patched; freeze-ratio + retract bins carry over, aux gate text does not.
6. **GPU/mem ⚠️ WATCH escalating** — GPU 20%/1.5 GB fine; container **mem 20.4/23.5 GiB (87%)**, +2.2 GiB since Pass 1 (~1.1 GiB/h) with the ring only at 583k/2M. At this slope the cap is ~3 h away → OOM risk mid-run. (Read-only pass: flagging, not acting.)

**Verdict: WATCH→ALERT-adjacent.** Machinery is healthy (guards, Q, saves) — the learning itself isn't biting: eval1 0.0% vs v2's 6.1% (v2 had 10k pretrain; this is from-scratch — eval2 is the fairer read), explore flat, start-state mean unmoved, alpha hunting undamped. Recommend (user decision, not mine): let it reach eval2 for the fair from-scratch comparison, but treat a second 0.x% eval + flat explore as the reassess trigger (per roadmap: next lever is P2 privileged-actor teacher / RLPD UTD+ensemble fidelity, not knob-tuning). Also flag the mem slope.

### Pass 1 (rlpd_v1) — 2026-07-16 12:53 EDT (~86 min in, u=56,371, explore 179.5k) — ✅ OK, alpha limit-cycle on WATCH

**Timebase correction:** the run-dir stamp `172751` is container TZ (UTC+2); actual launch was **11:27 EDT** (StartedAt 15:27:46Z), not 17:27 EDT as Pass 0 said. RestartCount=0 — "Up About an hour" is correct, no restart.

1. **Liveness ✅** — container Up, STEP lines advancing (worker ep≈20, global≈11.5k/worker), 0 Traceback / watchdog / trainer-deadline trips in the last 2 h.
2. **Q-DIVERGENCE WATCH ✅ (the headline gate)** — q1/q2 losses O(0.001–0.04) across all 56k updates (last row 8.6e-4); q1_mean oscillates −0.31 ↔ +1.77 with target_q tracking tightly — bounded, NOT monotone-up. LayerNorm is doing its job so far. nonfinite counts 0/0/0, grad norms 0.12–1.1, clamp_fraction ≤1.4% (**no std ceiling-pin** — v1's precondition absent).
3. **Alpha/entropy ⚠️ WATCH (not ALERT)** — alpha is INSIDE [0.0067, 1.0] and moving, but in a pronounced **limit cycle** (~25–30k updates period): decays 1.0→~0.009 while entropy_proxy falls 2.75→0.83, then snaps back to 0.4–0.96 and entropy recovers. This is the auto-alpha controller hunting around target 1.0 with rails, not the v1 pathology (no floor-pinning >50k, no std saturation), but amplitude is large — track whether the cycle damps or destabilizes policy learning.
4. **Explore success — early, flat:** 2/300 episodes succeeded (0.7%); avg reward −5.21 (first 150) → −5.11 (last 150). Expected ~0 start with pretrain 0; the real gate is eval1, due in ~35 min (explore 179.5k/250k). episode_summary.jsonl carries no reached flag — success proxied as terminated∧¬truncated (steps<600).
5. **Evals/probe** — none yet (no checkpoints); probe skipped gracefully.
6. **GPU/mem ⚠️ minor WATCH** — GPU 23% util, 1.5/24.5 GB VRAM (light — UTD 1.0 trainer + CPU-bound SOFA workers); container CPU 1173% (16 workers ✓) but **mem 18.2/23.5 GiB (77% of shm cap)** already — the 2M-transition ring still has 1.8M slots to fill; watch for OOM pressure at later passes.

**Verdict: OK.** No changes made. Next pass: eval1 result (the RLPD transfer gate), alpha-cycle evolution, mem trajectory.

### Pass 0 (rlpd_v1) — 2026-07-16 ~11:45 EDT (mislabeled 17:45 in original entry) — LAUNCH BASELINE ✅

Pre-flight: GPU free, no containers, seed md5 verified. Launch clean; 16 workers exploring by ep 1–2 (~5 min in; explore phase from step 0 — heatup skipped, pretrain 0). `runner.yml` confirms the full RLPD config took: `algo: sac`, `use_layernorm: true/true/false` (q1/q2/policy — intended asymmetry), `backup_entropy: false`, `offline_fraction: 0.5`, `log_alpha_min/max −5.0/0.0`, `UPDATE_PER_EXPLORE_STEP 1.0`. "Heatup cache loaded" print not yet visible in docker stdout (main-process block buffering; load itself is proven by the config echo + alive process — obs-dim/buckle-coef guards would have raised). No losses CSV yet (first explore cycle has zero updates by design). SOFA "Case 1" [ERROR] noise present as always — benign. Next pass: first update cycle + Q-health.

---

# RCCA Procedural v1 — Fix-Verification Monitor

**Run:** `2026-07-12_042431_rcca_procedural_v1` (container `rcca_procedural_v1`, RestartCount=0)
**Launcher:** `launch_rcca_procedural_v1.sh` (pretrain 10000, buckle_coef 0.5, aux 0.05 on labels 2,3,5,6, relax_failure_truncations, target_entropy 1.0)
**Seed:** `saved/rcca_proc_heatup/seed.npz` (480 ep, 282310 transitions, meta_buckle_coef=0.5)
**Note:** the `041157` dir was the *without-pretrain* first attempt (aborted, no trainer CSV); this run is the deliberate with-pretrain relaunch.

**Mandate:** every 2 hours, scan logs + losses + step-logs to verify the PDF fixes are holding across three axes — **learning stability**, **diverse ways**, **recovery schedules**. Log findings. **Make NO changes. Let the run continue.**

Sources: `eve_rl_deep_review_combined.pdf` (algo/infra) + `eve_rl_multimesh_generalization_combined.pdf` (obs/privileged-critic/recovery).

---

## The fix checklist (what we verify, and how)

### A. LEARNING STABILITY — deep-review algo fixes (mostly in `losses_trainer_synchron.csv`)

| ID | Fix | Observable signal | Healthy range |
|----|-----|-------------------|---------------|
| **B1** | log_prob floor clamp killed BC gradient → leaky floor | `awac_weight_mean`, `awac_weight_saturation` | mean ~1.0, sat ~0 |
| **B2** | padding-mask advantage bias (episode mode) → masked-advantage parity | AWAC weights stable, policy_loss finite | no blow-up |
| **C1** | hard `log_std` clamp (−2,0) made rails sticky → leaky/soft clamp | `clamp_fraction` | < 5% (small) |
| **F1** | unbounded `log_alpha` runaway under AWAC + log_std ceiling → clamp | `log_alpha`, `alpha` | bounded, not → +∞ |
| **F2** | masked `log_pi` biased alpha loss → masked alpha loss | `alpha_loss` finite, alpha stable | finite |
| **F4** | stale `self.alpha` after warm start → re-derive on load | no alpha discontinuity post-warmstart | continuous |
| **H4** | scalar unclipped exploration noise shared across dims → per-dim clipped | STEP `cmd_action` per-dim varied & within bounds | \|a0,a2\|≤30, \|a1,a3\|≤1.5 |
| **J1** | optimizers over wrong module set → per-network embedder opt | `grad_norm_q1/q2/policy` all nonzero | bounded, nonzero |
| **J2** | MLP missing ReLU after input layer → activation added | q-losses actually decrease | converging |
| **gen** | NaN / divergence guards | `nonfinite_q_loss_count`, `nonfinite_policy_loss_count`, `nonfinite_grad_count` | all 0 |

### B. DIVERSE WAYS — anti-memorization / mesh generalization

| ID | Fix | Observable signal | Healthy |
|----|-----|-------------------|---------|
| **O-RNG** | per-worker target RNG (centerlinerandom deepcopy reseed) | distinct `tgt=` sequences per pid | 16/16 distinct |
| **mesh** | per-worker procedural RCCA/RVA variation (fork_anchor, rva_amp) | `path_len` spread across pids | wide spread |
| **obs** | mesh-invariant obs (TipRelativeTracking2D, TargetTipOffset2D, path-relative LocalGuidance); no absolute-coord leak | deployable obs bounded ~[−1,1] | 0 values >\|1.5\| |
| **G1** | balanced-lane sampling (n_clean=0.6·B) | both trees populated; clean/general mix | both present |
| **2.4** | symmetric action space | `cmd_action[0]` sign ~50/50 fwd/back | ~50/50 |
| **tgt-arc** | target ≥40mm arclength from start (deep RCCA) | `d_tgt` large at episode start | large |
| **fork-var** | RCCA takeoff + proximal RVA perturbed (anti-bifurcation-memorization) | takeoff deflection varies across meshes | varies |

### C. RECOVERY SCHEDULES — recovery-capable navigation

| ID | Fix | Observable signal | Healthy |
|----|-----|-------------------|---------|
| **relax** | `relax_failure_truncations`: fold/off-path/vessel_end don't truncate; counters keep climbing | STEP `off_br`/`fold` climb high with `trunc=False` | high counters, no trunc |
| **v-end** | vessel_end relaxed (no truncation label under relax) | no `vessel_end` in outcomes | 0 |
| **WBT** | WBT label leak fixed in 3 places (_resolve_termination_reason, grader_failure_timeout, _gtimeout) | no WBT/grader_failure_timeout in results | 0 |
| **buckle** | anti-buckle privileged shaping φ(slack,contact), delta form, telescoping | STEP `buckle_phi` nonzero on buckled steps, ~0 when slack low | active, bounded [−1,0] |
| **retract** | off-path retract tax −0.002, min 3 off-steps, min 0.1mm gw | off-path steps small-negative; retraction not over-penalized | gentle |
| **max_steps** | timeout penalty −3 (reward-farm kill: timeout ≠ success) | max_steps episodes net negative | negative |
| **arc1x** | arclengthprogress 1× (killed 2× doubling farm) | cum_reward tracks net arclength, not inflated | telescopes |

### Known-benign noise (do NOT flag)
- `[ERROR] InterventionalRadiologyController ... Case 1 should never happen ... totalLengthIsChanging` — SOFA beam-adapter curvature bookkeeping warning, present in every run, not our bug.
- `can_sample=False` toggling — normal; it's `sample_queue_empty AND len>batch_size` ([pervanillashared.py:134](../eve_rl/eve_rl/replaybuffer/pervanillashared.py#L134)). It reads False whenever a prefetched batch is sitting in the queue awaiting the trainer. Flips True/False continuously once online updates flow.
- **`update_step=10000` "frozen" for the first ~30 min after pretrain is EXPECTED, not a stall.** Heatup was cache-loaded, so `step_counter.exploration=0` entering the online loop. The first `explore_and_update` cycle computes `update_steps = max(0, exploration*ratio - ...) = 0` ([runner.py:395](../eve_rl/eve_rl/runner/runner.py#L395)), so it runs `CONSECUTIVE_EXPLORE_EPISODES=100` pure-exploration episodes with **zero updates**. Online updates begin on the *second* cycle (~100 episodes ≈ 31 min at ~3 ep/min × 16 workers), then arrive in ~22k-update bursts per 100-episode cycle at ratio 0.5.
- **Two `update_step` sources disagree by design:** the ReplayBuffer STATUS line's `update_step` lags (registered at cycle boundaries); the **losses CSV `update_step` + `batches_produced`/`priority_updates` are authoritative** for real-time update progress.
- **Trainer process at ~1% CPU is EXPECTED** — NN updates run on the GPU (`-d cuda:0`). Liveness = `batches_produced` climbing, not trainer CPU%.

### Config (from container cmdline)
`--update_per_explore_step 0.5` · `CONSECUTIVE_EXPLORE_EPISODES=100` · `EXPLORE_STEPS_BTW_EVAL=2.5e5` · `TRAINING_STEPS=2e7` · `-nw 16` · `-d cuda:0` · `--algo awac --awac_lambda 3.0 --balanced_fraction 0.3 --per --log_std_min -2 --log_std_max 0.0 --target_entropy 1.0`

---

## Verification passes

### Pass 0 — t=0 (~18 min in, pretrain just completed, explore starting)

**Overall: HEALTHY. All three axes holding. No changes made.**

**A. Learning stability** (losses CSV row 10000):
- Pretrain complete: `update_step=10000`, `priority_updates=10000`, `batches_produced=10001`. ✅
- `alpha=0.0491`, `log_alpha=-3.014` — bounded, F1 holding (no runaway). Dropped from ~0.208→0.049 over pretrain; **benign** because `entropy_proxy=2.73 ≫ target_entropy=1.0` (SAC correctly lowering α; entropy is healthy, not collapsed). **WATCH-ITEM:** flag only if α→~0 *and* entropy_proxy also collapses toward/below 1.0.
- `clamp_fraction=0.0049` (0.49%) — C1 holding. ✅
- `nonfinite_q/policy/grad_count = 0` — no NaNs. ✅
- `awac_weight_mean=1.00`, `awac_weight_saturation=0.0`, `awac_weight_max=1.17` — B1/B2 holding. ✅
- `grad_norm_q1=0.029, q2=0.033, policy=0.169` — J1/J2 holding (nonzero, bounded). ✅
- `q1_loss=0.0010, q2_loss=0.0012` — converged on seed. ✅

**B. Diverse ways** (STEP logs, last ~4000 lines):
- 16 distinct worker pids exploring. ✅
- `path_len` spread 76–265mm across pids (mesh variation live). ✅
- Actions per-dim varied & within bounds (a0,a2∈±30; a1,a3∈±1.5) — H4 holding. ✅
- (obs-range & action-symmetry deep audit deferred to a heavier pass; verified clean pre-compaction.)

**C. Recovery schedules** (STEP logs):
- `off_br` climbs to **285**, `fold` up to 5/20, all with `trunc=False` — relax holding (recovery not truncated). ✅
- Zero WBT / vessel_end / grader_failure_timeout across full log — label leaks fixed. ✅
- `buckle_phi` active and bounded [−1,0]: down to −0.500 on buckled/off-path steps; ~0.000 when slack low. ✅
- Off-path steps carry small negative reward (e.g. −0.055, −0.024); no over-penalized retraction. ✅
- **Episode outcomes (early online explore)** rank correctly: `reason=max_steps` with `is_clean` split 1/0 by final_branch (RCCA=clean vs RVA/bridge=not), and at least one `reason=success, grader_success=1` already appearing. Timeout ≠ success (arc1x + max_steps −3 farm-kill holding). ✅

**Watch-items for next passes:**
1. α trajectory vs entropy_proxy (collapse check).
2. `update_step` climbing past 10000 (online training resumes; `can_sample` flips True).
3. First online episode-outcome distribution (success vs clean-timeout vs off-path) once online updates run.
4. Any drift in nonfinite counts / clamp_fraction / grad norms as online data enters.

### Pass 0b — t≈36 min (online updates now flowing; corrects a false "stall" read)

**Overall: HEALTHY, on track.** A mid-check "update_step frozen at 10000" was investigated and found **expected by design** (see Known-benign notes: first 100-episode cycle runs 0 updates; heatup was cache-loaded so exploration started at 0). No stall.

- **Online updates flowing:** `batches_produced` 10001→14464, `priority_updates`→14463; losses CSV **14555 rows, update_step=14554** (~4.5k online updates done). Second explore cycle underway.
- **Stability:** `alpha=0.0126` (down from 0.049 at pretrain end; **WATCH** — but `entropy_proxy=2.70` ≫ target 1.0, so still benign/stochastic), `clamp_fraction=0.68%`, `nonfinite_q=0`, `q1_loss=0.0037` (small, slight rise as online data enters — normal).
- **Recovery:** `off_br`→495 with `trunc=False`; `buckle_phi` active to −0.50; zero WBT/vessel_end leaks.
- **Diversity:** 15–16 distinct pids; `path_len` spread 108–268mm.
- **Outcomes:** `max_steps` clean/not split by final branch + at least one `success grader_success=1`.
- **Elapsed:** ~36 min container uptime; pretrain ~7 min, online ~29 min (~112 online episodes at ~3 ep/min × 16 workers).

### Pass 1 — 2026-07-12 00:41 EDT (≈2h since pretrain; ~2h15m uptime)

**Overall: HEALTHY.** Strong online progress; policy now solving episodes. One WATCH note (alpha at clamp floor) that does NOT trip the alarm.

- **Progress:** losses CSV **97,558 rows, update_step=97,557** (~87.5k online updates, up from ~14.5k at Pass 0b). `batches_produced=97,287`, `priority_updates=97,285` corroborate. `explore_step=224,976` (first eval fires at 250k — imminent; eval env already initialized). `episodes_received=886` (+288), `buffer_len=507k`.
- **A. Stability:**
  - **α=4.5e-05, log_alpha=-10.0 → alpha has hit its F1 clamp FLOOR** (was 0.049→0.0126→now floor). **WATCH, not tripped:** `entropy_proxy=2.34` still ≫ target 1.0, so the policy remains stochastic and F1 is working as designed (SAC correctly drove α down because policy entropy > target; the clamp caught it at the floor rather than running away). Re-flag only if entropy starts falling toward 1.0 with α pinned.
  - `clamp_fraction≈0.01–0.02` (still small, C1 ✅), `q1_loss≈0.003–0.006` (small/stable), `policy_loss≈1.1–1.9` (rose from 0.64 as AWAC online term develops — bounded, normal), `grad_norm_q≈0.03–0.09`, `grad_norm_policy≈0.6–1.1` (bounded ~grad_clip=1.0), **nonfinite_q/p/g=0**, `awac_weight_mean≈1.00, saturation=0.0, max=1.24` (B1/B2 ✅).
- **B. Diversity:** 16 distinct pids ✅; `path_len` spread 105–273mm ✅.
- **C. Recovery:** `off_br`→122 (window-dependent; earlier passes to 495) with `trunc=False` ✅; `fold`→3/20 no trunc ✅; **zero** WBT/vessel_end/grader_failure_timeout leaks ✅; `buckle_phi` active −0.26→−0.32 ✅.
- **Outcomes (big positive):** recent window (last ~20k lines) = **5 `success grader_success=1`**, 14 `max_steps is_clean=1` (reached RCCA, timed out), 13 `max_steps is_clean=0` (off-path/wrong). Ranking sane (success > clean-timeout > off-path); timeout ≠ success. Success count up 1→5 vs Pass 0b — online learning is working.
- **Deltas from Pass 0b:** +83k online updates; α 0.0126→floor(4.5e-05); entropy 2.70→2.34 (still healthy); successes 1→5; buffer 349k→507k.

### Pass 2 — 2026-07-12 02:40 EDT (≈4h since pretrain)

**Overall: WATCH.** Nothing broken; performance improving. But the α/entropy watch-item is now actively engaging (entropy on a steady decline toward target with α pinned at floor). No changes made.

- **Progress:** losses CSV **174,358 rows, update_step=174,357** (~164k online updates, +77k since Pass 1). `explore_step=371,660`. `episodes_received=1183` (+297), `buffer_len=654k`.
- **Eval/checkpoint HEALTHY:** first eval fired ~explore_step 278k as scheduled — `checkpoints/best_checkpoint.everl` + `checkpoint278345.everl` written; policy snapshots every 10k (policy_10000…160000+.pt). Next eval ~500k. (My grep missed the eval log strings; the checkpoint files are the proof.)
- **⚠ WATCH — entropy declining toward target with α at floor.** Slope sampled every 20k updates: ent **2.49(u60k) → 2.40(u80k) → 2.25(u100k) → 2.10(u120k) → 1.98(u140k) → 1.65(u160k) → ~1.5(now)**. α=4.5e-05 (floor, log_alpha=-10) throughout. Still **above** target 1.0, so not yet a PROBLEM, and it coincides with **rising success** (→ healthy convergence, not collapse). **Expected self-correction:** once entropy crosses <1.0, `alpha_loss` gradient flips sign and α rises off the floor to restore entropy pressure (the F1 clamp floor does NOT prevent α climbing back). **Decision point next pass:** confirm α actually lifts off −10 as entropy nears/crosses 1.0. Failure mode to escalate on: entropy keeps falling well below 1.0 while α stays pinned (→ policy goes near-deterministic, exploration dies).
- **A. Other stability:** `clamp_fraction≈0.03–0.05` (up from 0.01–0.02 but still small, C1 ✅), `q1_loss≈0.005–0.008` (small/stable), `policy_loss≈1.2–1.9` (bounded), `grad_norm_policy≈1.0–1.5` (~grad_clip 1.0), **nonfinite_q/p/g=0**, `awac_weight_mean≈1.00, sat=0.0, max=1.27` (B1/B2 ✅).
- **B. Diversity:** 18 distinct pids (≥16; extras = worker restarts) ✅; `path_len` 76–266mm ✅.
- **C. Recovery:** high `off_br` with `trunc=False` ✅; `buckle_phi` active to −0.507 ✅; **zero** label leaks ✅.
- **Outcomes (excellent, improving):** window = **13 `success is_clean=1`, 3 `success is_clean=0`** (16 successes vs 5 at Pass 1), 18 `max_steps is_clean=1`, 17 `max_steps is_clean=0`. Success rate ~30% (↑ from ~16%). The **3 `success is_clean=0`** = went off-path then RECOVERED to reach the target → direct evidence recovery training is producing the intended behavior. Ranking sane; timeout ≠ success.

### Pass 3 — 2026-07-12 04:40 EDT (≈6h since pretrain)

**Overall: WATCH.** The Pass-2 entropy/alpha watch RESOLVED favorably (self-correction worked) — but it overshot, and the recent success rate dipped in its wake. Confirm recovery next pass. No changes made.

- **Progress:** losses CSV **246,037 rows, update_step=246,100** (~236k online updates, +72k). `explore_step=505,588`. Second eval fired → `checkpoint505588.everl` written (now 278345 + 505588 + best). `episodes_received=1478` (+295), `buffer_len=788k`. **Cumulative grader_success=251** (baseline for next-pass rate diff).
- **✅ α/entropy watch RESOLVED (self-correction confirmed):** as predicted, once entropy crossed target the `alpha_loss` sign flipped and α climbed off the −10 floor: `α 4.5e-05(u180k) → 4.6e-05(u200k, log_alpha lifts) → 0.0023(u220k) → 0.28(now)`; `log_alpha −10 → −1.25`. Entropy now regulated at target ~1.0 (target_entropy=1.0). The F1 clamp floor did NOT trap α. Textbook SAC temperature control.
- **⚠ NEW WATCH — transient entropy OVERSHOOT + success dip.** Entropy over-corrected **down to 0.14 at u≈220k** (brief over-determinism / lost exploration) before α restored it to ~1.0. In its wake, recent success rate dropped: **~30% (Pass 2) → ~15% (40k-line window: 8 succ / 54) / ~9% (25k window: 3 succ / 33)**. Most likely transient (policy briefly over-exploited during the collapse; α=0.28 now re-establishing exploration), possibly compounded by mesh rotation (`procedural_change_every 10`). **Escalate if:** next pass success stays ≤15% or keeps falling (→ lasting damage from the over-collapse). **Healthy if:** recovers toward/past 30%.
- **⚠ Minor — 2 `vessel_end` labels (first appearance), NOT a truncation leak.** Both `EPISODE_OUTCOME reason=vessel_end` ran `steps=600` (= max_steps → recovery preserved, no early cutoff), both from one restarted-looking worker `pid=1751`, wire down the trunk to dead-end, correctly penalized (`return≈−8.3/−8.8`, is_clean=0). Terminal-state label at max_steps, not a relax regression. Watch if the count grows.
- **A. Other stability:** `clamp_fraction≈0.05–0.06` and `grad_norm_policy≈1.1–2.1` crept up (consistent with the peaked low-entropy phase; expect to settle as entropy normalizes), `q1_loss≈0.008–0.014` (small), `policy_loss≈1.1–2.25` (bounded), **nonfinite_q/p/g=0**, `awac_weight_mean≈1.00, sat=0.0` (B1/B2 ✅).
- **B. Diversity:** 16 distinct pids ✅; `path_len` 127–269mm ✅.
- **C. Recovery:** high `off_br` (→184) with `trunc=False` ✅; `buckle_phi` active to −0.31 ✅; label leaks = only the 2 benign vessel_end above.

### Pass 4 — 2026-07-12 06:40 EDT (≈8h since pretrain)

**Overall: HEALTHY.** All three Pass-3 watch-items resolved favorably. Downgraded from WATCH. No changes made.

- **Progress:** losses CSV **334,773 rows, update_step=334,854** (~324k online updates, +88k). `explore_step=690,776` (3rd eval ~750k, imminent). `episodes_received=1893` (+415), `buffer_len=973k`.
- **✅ RESOLVED — success recovered (Pass-3's main watch):** cumulative `grader_success` **251→388 (+137 in 2h)**; comparable 40k-line window = **24 `success is_clean=1` + 2 `success is_clean=0`** = 26 succ / 76 = **~34%** (back above Pass-2's ~30%, up from the ~15% Pass-3 dip). The entropy-overshoot dent was transient, as predicted.
- **✅ RESOLVED — α/entropy at stable equilibrium:** α **0.28→0.41** (holding), `log_alpha≈−0.89`, entropy oscillating around **target ~1.0** (samples 0.61–1.17). Temperature control in healthy steady state.
- **✅ RESOLVED — vessel_end not growing:** 2→**3** (+1 in 2h) → rare benign terminal-state label at max_steps, not a relax regression. Concern closed.
- **A. Other stability:** `q1_loss≈0.013–0.018` (small, crept up as Q-values develop — `q1_mean≈−4.3`; watch but not concerning), `policy_loss≈1.5–1.6` (bounded), `clamp_fraction≈0.06` (stable ~6%, C1 ✅), `grad_norm_policy≈1.3–1.7` (bounded), **nonfinite_q/p/g=0**, `awac_weight_mean≈1.00, sat=0.0, max=1.58` (B1/B2 ✅).
- **B. Diversity:** 16 distinct pids ✅; `path_len` 75–267mm ✅.
- **C. Recovery:** high `off_br` (→230) with `trunc=False` ✅; `buckle_phi` active to −0.28 ✅; zero *new* label leaks.
- **Outcome mix:** clean (success+clean-timeout = 55) vs not-clean (21) → ~72% of episodes reach RCCA; of those ~44% complete to target. Ranking sane; timeout ≠ success.

### Pass 5 — 2026-07-12 08:40 EDT (≈10h wall since pretrain) — 🛑 PROBLEM: RUN HUNG

**Overall: PROBLEM — training is DEADLOCKED (frozen ~56 min, ongoing). No crash. Per mandate I did NOT restart/stop the container.**

- **Hang evidence (three independent confirmations):** total container CPU **0.34%** (healthy ≈ 1170%); losses CSV frozen at **372,036 rows** across two reads 20s apart AND a 3rd read; STATUS frozen (`episodes_received=2074`, `batches_produced=372036` identical across reads); `hung_for_sec=3399` (**56 min**) = container-now epoch minus last STEP wall_time. Processes alive (17.5GiB held, RestartCount=0, Running=true) but idle → **silent deadlock**.
- **When/where:** last STEP ≈13:49 UTC (ep=14 ended normally at ep_step=600, trunc=True, −3 max_steps penalty). `checkpoint771040.everl` written ≈13:50 UTC (3rd eval completed + saved). Freeze began immediately after → **deadlock in the post-eval → explore_and_update resumption path** (most likely a blocking multiprocessing `.get()` / eval-worker that never returned; NB host `date` and container `date` differed ~2h this pass — a possible clock jump / suspend-resume that can wedge mp semaphores).
- **NOT the cause:** 0 "Restarting Agent"/timeout warnings, 0 tracebacks/exceptions, 0 OOM/CUDA errors. Not a worker-timeout cascade, not a crash.
- **Recovery position (intact):** `checkpoint771040.everl` (explore_step 771040) + `best_checkpoint.everl` are saved. Run is resumable from there — needs USER decision (mandate = no changes).
- **Pre-hang health (for context, all was fine until the freeze):** ~324k→**372k online updates**; cumulative `grader_success` **388→443 (+55)**; α/entropy still in healthy equilibrium (α≈0.45, entropy≈1.0); `nonfinite=0`, `awac_mean≈1.0`; recovery relax + buckle holding. The last ~40k-line window skewed off-path (mostly `final_branch=bridge`, wire stuck making micro-moves at the bridge entrance) — but that's the immediate pre-hang eval-adjacent patch, not a policy collapse (cumulative successes were still rising).
- **⚠ ACTION NEEDED:** the run will not self-recover (deadlocks don't clear). Every minute is idle wall-clock. Awaiting user decision: restart from `checkpoint771040` / investigate the post-eval deadlock / leave as-is.

**DIAGNOSIS (user chose "diagnose then restart"; py-spy blocked by missing SYS_PTRACE, used /proc wchan):**
- **Main process (internal PID 7) main thread: `anon_pipe_read`** = blocked on a **no-timeout multiprocessing queue/pipe `.get()`**, waiting for a subprocess message that never came. All 16 workers: `poll_schedule_timeout` = idle, polling their task queue for a command the wedged main never dispatches. → **IPC deadlock**, main is the stuck party.
- **Smoking gun in code:** synchron `_update_algo_state_dicts()` (runs after every explore_and_update cycle AND post-eval) calls `self.trainer.state_dicts_network/optimizer/scheduler(...)` → `singelagentprocess.state_dicts_*()` → `self._model_queue.get()` **with NO timeout** ([singelagentprocess.py:619](../eve_rl/eve_rl/agent/singelagentprocess.py#L619)). If any subprocess drops/never sends its reply, the main blocks forever. (Deep-review H3 already flagged the weight-distribution path as fragile.)
- **Trigger:** host `date` was ~2h off container `date` this pass → a **host suspend/resume / clock jump** around the 3rd eval (~13:50 UTC, `checkpoint771040` write) interrupted an IPC transfer; the no-timeout `.get()` then hung permanently.
- **Recurrence risk:** if the trigger was a one-off host sleep, a restart runs fine until the next sleep. The underlying fragility (no-timeout gets) means any future IPC hiccup can re-hang it → hardening fix = add timeouts + restart-on-timeout to those `_model_queue.get()` calls (separate change, needs approval).

**POST-PASS-5 FINDING (user-prompted): the run was ALREADY collapsing before the deadlock.**
Eval history (main results CSV): eval1 278k = **8.2%** (8/98, speed 4.78mm/s, traj 326mm) → eval2 505k = **13.3%** (13/98, 4.28mm/s, 220mm, = `best_checkpoint`) → eval3 771k = **3.1%** (3/98, **0.95mm/s, 59mm**). Speed/trajectory collapse = policy learned to FREEZE: last 60 explore episodes = 54× `final_branch=bridge/max_steps`, wire at ~4.8mm insertion micro-oscillating (|cmd_action|≈0.08 on ±30 scale) full 600 steps, return −6 to −6.9. Progressive: successes +137/2h → +55/2h → ~0/h. **Hypothesis:** timidity local optimum (sit-still ≈ −6.8 beats risky attempts −6…−16 under buckle/off-path pressure), plausibly seeded by the u≈220k entropy over-collapse, then self-reinforced via PER+AWAC on freeze-dominated buffer data. Alpha rising kept entropy AT target while the action MEAN froze — entropy regulation does not prevent mean-collapse. **Correction to Pass 5:** the off-path-skewed window was NOT "eval-adjacent noise"; it was the collapse in progress.
**Restart implication:** `checkpoint771040` = degraded (freeze-basin) policy — do NOT warm-start from it. `best_checkpoint.everl` (= eval2, 505k) + `replay_buffer.npz` (saved same eval, 10:37 UTC) form a matched pre-collapse weights+buffer pair. Even so, restarting without addressing the freeze mechanism risks re-entering the same basin.

## COLLAPSE FORENSICS (5-agent workflow, 2026-07-12 ~13:00 EDT) — MECHANISM ESTABLISHED

**One-line verdict: the freeze was caused by the entropy controller itself — with std railed at the log_std ceiling all run, alpha's recovery off its floor could only raise entropy by crushing the action MEAN toward zero; AWAC's advantage weights were ≈1.0 the whole run (zero discrimination), so nothing opposed it. The buffer was NOT poisoned; rails were a symptom, not the cause; freezing was never strictly reward-rational (but was within ~1.2 return-units of it).**

Causal chain (each link verified by a different agent):
1. **std pinned at CEILING from u≈39k onward** (snapshot-probe: log_std ceiling-rail fraction = 100% on all 4 dims, all probe states, from first online snapshot to last; floor −2 never touched). std=1.0 everywhere → the entropy controller had NO std headroom, ever.
2. **alpha floor-pinned 165k updates** (losses: log_alpha=−10 from u=33,333 to u=200,447) while entropy_proxy decayed 2.63→0.81 and tanh-rail clamp_fraction rose 0.9%→10.6% (peak u≈220k, entropy min 0.139). Rails (12%→23% of translations at exactly ±30.000) = tanh saturation from growing means — a SYMPTOM of the same decay, and they receded to 18–19% after recovery (rails did NOT cause the freeze; the freeze is the opposite of a rail mode).
3. **alpha recovery = the mean-crush.** With std already at ceiling, the only remaining entropy lever is shrinking |mu| (less tanh saturation = higher log-prob entropy proxy). Snapshot-probe: episode-start-state |tanh(mu0)| peaked 0.255 (explore 422–468k), fell **−33% in the single interval explore 468k→505k** (= exactly the alpha spike), then monotonically to 0.089 by eval3. Collapse is REGIONAL: start states −65%, early states −26%, buffer-wide −17% — concentrated where episodes begin (= the freeze region, wire ~5mm).
4. **AWAC was inert all run** (losses: awac_weight_mean 0.98–1.04, saturation 0.0 ALWAYS, max p99 1.7) → weight=exp((Qbuf−Qπ)/λ) with λ=3 and a flat critic ≈ 1 → the policy loss degenerated to **pure BC + α·entropy**. Once α≈0.45, the entropy term ruled wherever BC was ambivalent (start states have the most mixed buffer actions) → mean→0 there.
5. **Amplifiers:** (a) critic pessimism: q1_mean −0.97→−4.71 monotone, converging toward the freeze-return level (−6.6); with min(Q1,Q2) penalizing the high-variance "attempt" branch (returns −1.9..−74.8) vs variance-free freeze (σ=0.32), a bias of only ~1.2 units flips the ordering (timidity-EV: EV(attempt)−EV(freeze) never went below +1.15 — freezing was never strictly rational, but "reward-adjacent-rational"). (b) **buffer stores RAW policy actions while the env executes CLAMPED ones** (buffer-poisoning agent: at stream-micro steps the buffer holds large retraction commands, p10 −27.5mm/s, at the insertion floor ~4.8mm) → ~1% of rows teach Q that railed retraction at the floor is consequence-free → flattens Q exactly in the freeze region.
6. **Why explore looked healthy while eval collapsed:** with std=1.0, sampled explore actions average |a|≈17mm/s regardless of the mean — explore success was still 39% in the last full hour. The collapse only shows in DETERMINISTIC rollouts: eval speeds 5.2→4.3→0.95mm/s. The "all 16 workers froze at the end" was eval3 being dispatched synchronously to all workers (rails-timeline agent: ep-counter resets on all 16 pids within 2–3 min at each eval; the final-hour "freeze" window IS eval3). **Monitoring lesson: track deterministic probe actions, not explore returns.**

Buffer verdict: NOT poisoned — stored micro-actions 0.02% of 787,898 rows; PER neutral on freeze-band transitions (mass 8.25% vs share 8.38%); clean-lane worked as designed. Buffer/PER cannot explain the collapse; the policy-side mechanism above carries it.

**Checkpoint damage assessment:** eval3/771040 = fully collapsed (start-state mean 0.089, 57% of start states command <1.5mm/s). **best_checkpoint (=eval2/505588) is already −33% into the mean-collapse.** eval1/278345 is pre-peak (mean 0.150, quality 8.2%). The policy's actual peak (explore ~422–468k) was never checkpointed. There is NO clean high-quality checkpoint.

**Fix candidates for any restart (ALL require approval — no changes made):**
1. **std-ceiling headroom / target-entropy recalibration** — the core trap is (log_std_max=0 ceiling) × (target_entropy=1.0) × tanh-entropy-proxy: the controller can only hit the target by shrinking means once std hits the cap. Note the LCCA v2 work already exposed `log_std_max` ceiling-cap + anti-rail knobs — same disease family.
2. **Raise the alpha floor** (log_alpha −10 → ~−5) and/or rate-limit alpha's recovery so the correction can't whipsaw.
3. **Restore AWAC discrimination** — λ=3.0 with this critic ⇒ weights≡1 (BC). Lower λ or rescale advantages; verify weight variance online.
4. **Store the EXECUTED action in the buffer** (or clamp before store) — fixes the consequence-free-rail mislabel at actuator limits.
5. Deadlock hardening (timeouts on `_model_queue.get()`) — from the earlier diagnosis.
Full agent reports: workflow wf_7f357253-974 journal; scripts + tables in scratchpad/collapse/ (incl. losses_sampled_5k.csv, snapshot_probe_out.txt).

## V2 RELAUNCH — 2026-07-12 ~14:00 EDT (fix package applied, user-approved)

v1 stopped+removed (was deadlocked 5+h; artifacts preserved: run dir + `saved/v1_collapse_forensics/` with steps.tsv.gz, outcomes.tsv, losses_sampled_5k.csv, snapshot-probe outputs). **`launch_rcca_procedural_v2.sh`** launched with:

| Fix | Change | Attacks |
|---|---|---|
| F1 alpha rails | `--log_alpha_min -5.0 --log_alpha_max -2.3` (α ∈ [0.0067, 0.100]; new sac.py/agent.py/train-script plumbing, defaults legacy) | decay-to-floor → whipsaw → mean-crush |
| F2 AWAC discrimination | `--awac_lambda 1.0` (was 3.0) | weights≡1 → loss degenerated to BC+entropy |
| F3 anti-rail loss | `--action_mean_penalty 0.005` (pre-wired, now on) | mean-inflation rails (LCCA branch #4) |
| F4 clean-lane rail filter | `EVE_CLEAN_RAIL_MAX=0.15` (ported to pervanillastep.push; rejected successes stay in general buffer, is_clean=False so update_priorities can't re-admit) | bang-bang self-cloning loop (LCCA branch #2) |
| F5 executed-action storage | **DROPPED after code review** — buffer stores the commanded action, which is CORRECT for off-policy Q (env clamps at actuator limits are true dynamics, not mislabels) | — |
| F6 deadlock guard | `_model_queue.get(timeout)` via `EVE_RL_MODEL_QUEUE_TIMEOUT_S=900` → loud RuntimeError instead of silent hang | the v1 post-eval3 IPC deadlock |
| F8 freeze detector | monitor now runs a deterministic start-state probe each pass (cache `probe_start_states_v1.npz`; validated: reads v1's frozen policy at 0.086). Thresholds: OK≥0.15, WATCH≥0.10, ALERT<0.10 | v1's freeze was invisible in explore returns |

Tests: syntax (container py3.8) ✅; SAC signature ✅; rail-filter functional (admit 0.10 / reject 0.25 / legacy-mode) ✅; `--help` fixed (pre-existing `%` bug at line 1415) + new flags parse ✅; probe end-to-end vs v1 ground truth ✅.
**Host-sleep caution:** the v1 deadlock trigger was a host suspend/clock jump — keep the machine awake during v2; the guard only makes hangs visible, it doesn't survive them.

### Pass 0 (v2) — 2026-07-12 ~14:45 EDT — post-pretrain collapse-signature check

**Overall: HEALTHY — every fix signature present, no v1 collapse precursors.** Pretrain 10,000/10,000 done on the seed (282,310 transitions / 480 eps confirmed loaded).

| Signature (v1 collapse link) | v1 @u10k | v2 @u10k | Verdict |
|---|---|---|---|
| α at pretrain end (link 2: decay-to-floor) | 0.049 → later 4.5e-5 | descended 0.073→0.0067 and **stopped exactly at log_alpha=−5.0** (u9k, u10k) | ✅ F1 floor working; band violations = 0 |
| AWAC weight max (link 4: inert BC) | 1.17 (whole-run p99 1.7) | **1.45–2.49** during pretrain | ✅ F2 discrimination ~3× wider (online is the real test) |
| clamp_fraction (rails) | 0.005 | 0.002–0.005 | ✅ same healthy baseline |
| q1_mean (link 5: pessimism) | −0.97 | **−0.40** | ✅ milder — weaker freeze-attractor |
| entropy_proxy | 2.73 | 2.65 | ✅ equivalent |
| nonfinite / awac_sat | 0 / 0.0 | 0 / 0.0 | ✅ |
| CLEAN_RAIL_FILTER on seed load | n/a | **0 rejections** (480 eps pushed; seed cleans ~0.10 railed < 0.15) | ✅ F4 armed, correctly not tripping on diverse seed data |
| Deterministic probe | 0.053 @pretrain-end snapshot | no snapshot yet (first eval cycle pending) | — probe fires from Pass 1 |

**What to watch online (the real F1 test):** α now starts its online phase AT the floor (0.0067) instead of 150× below it — when entropy decays toward target 1.0, the lift-off should be smooth (recovery distance log-alpha −5→−2.3, vs v1's −10→−0.8 whipsaw). The u≈220k-analog window is the checkpoint to scrutinize.

**Addendum — restart with `--eval_after_pretrain` (run dir `2026-07-12_175210`):** v2 was restarted inside zero-update cycle 1 (nothing learned lost) to add a post-pretrain baseline eval (new runner flag). Pretrain values reproduced (α path identical to 4–5 sig figs — near-deterministic given same seed; batch stats within noise). **BASELINE EVAL (explore=0): quality 6.1% (6/98), speed 0.54 mm/s, traj 57.5mm, reward −4.62.** `checkpoint0.everl` banked; explore resumed cleanly through the post-eval seam that deadlocked v1.
**Key reframe:** the pretrained policy is already deterministically timid (0.54mm/s ≈ v1's collapsed eval3 kinematics) → **the freeze basin IS the pretrain-BC attractor**; online learning grows the mean away from it (v1: 0.053→0.255→crushed back to 0.086). The deterministic probe's job = confirm mean|a0| grows and never regresses. Also calibrates v1: 278k steps/109k updates lifted quality only 6.1%→8.2% (peak 13.3%).

### Pass 1 (v2) — 2026-07-12 15:58 EDT (~2h since restart; eval1 just completed)

**Overall: HEALTHY — best held-out result the project has ever produced, and every fix is verifiably doing its job.**

- **🏆 eval1 @ explore 286,564: quality 30.6% (30/98), speed 4.45mm/s, traj 168mm, reward −1.97.** Baseline was 6.1%/0.54mm/s → **+24.5pts in one eval interval**. v1's all-time PEAK was 13.3% (and its eval1 at the same explore count: 8.2%). v2 is at **2.3× v1's best, 3.7× v1's like-for-like**.
- **F1 validated online (better than designed):** α sat at the floor (0.0067) the entire online phase while entropy glided 2.65→**equilibrium at ~1.0–1.2 (target 1.0)** — no crash to 0.14, no whipsaw, no lift-off even needed; the floor value ≈ the equilibrium α. v1 at this point was mid-decay toward the 0.139 entropy collapse. Band violations: 0.
- **F2 (λ=1.0):** awac_weight_mean 1.007±0.010 (range 0.963–1.094), max regularly 1.6–2.8 with 3,738 rows >3 — ~3× v1's weight variance. Discrimination alive (modest); no saturation.
- **F3/F4:** clamp_fraction 1.3–2.8% (v1 same phase: rising through 5% toward the 10.6% peak); CLEAN_RAIL_FILTER **0 rejections across 211 online successes** → new successes are genuinely non-railed (penalty working at the source, filter as untripped backstop).
- **Probe recalibrated — FREEZE-ALERT was a v1-calibration false alarm.** v2 trajectory (dedup snapshots): 0.032(pretrain) → 0.034 → 0.112 → 0.058 → 0.116 → 0.083 — **growing away from the pretrain/freeze attractor**, opposite of v1's 0.255→0.089 decline. With action_mean_penalty, means are systematically smaller yet decisive (eval 4.45mm/s). New verdict bands: ratio to v2-pretrain baseline 0.032 (OK≥2×, WATCH≥1.25×, ALERT≤1.25× = regression to attractor). Current: 2.6× = OK. Watch the u≈88909-style dips (0.058) — oscillation so far, recovered next snapshot.
- **Stability:** q1_mean ≈ **−1.0 flat** (v1 same phase: −1.7→−2.9 and sinking — the pessimism spiral that fed the freeze). q1_loss ~0.003, grad_norm_policy 0.5–1.3, 0 nonfinite, 0 IPC timeouts.
- **Learning/outcomes (677 online eps):** 202 clean success + **9 recovered success (is_clean=0→target)** + 268 clean-timeout + 198 off-path = **31% explore success**, matching the 30.6% eval (train↔held-out gap ≈ 0 → generalization across procedural meshes, no memorization gap).
- **Recovery:** off_br→309 with trunc=False; 609 off-path steps in window; buckle_phi engaged on 4,505/8,000 steps (to −0.505); 0 label leaks; 0 vessel_end.
- **Diversity/throughput:** 16 pids, path_len 149–263mm; explore 291,748 + 128k online updates in ~2h (update ratio on target: 128k done vs 146k budget).

### Pass 1b (v2) — 2026-07-12 16:17 EDT (+19 min; duplicate cron firing — kept brief)

**Overall: HEALTHY. Key event: F1's untested branch executed — α lifted off the floor smoothly.** Entropy dipped below target (0.86–0.88) → log_alpha −5.0 → −4.92 (u140k) → −3.20 (u150k) → **α=0.052**, 0 band violations, no whipsaw, no mean-crush (probe unchanged at 0.083 = 2.6× baseline, verdict OK under recalibrated bands; no new snapshot yet). This was the exact maneuver that killed v1.
- Outcomes (last 500): 183 clean + 6 recovered successes = **~38% explore success** (↑ from 31%); cumulative online successes 242.
- q1_mean −1.0 → **−1.3** (mild drift as the α·entropy term re-engages; WATCH, not concerning — v1 was −2.9 and sinking).
- awac samples 1.013–1.026 / max to 2.6; clamp 2–4%; 0 nonfinite; 0 IPC timeouts; 0 rail-filter rejections; recovery relax + buckle holding; 16 pids, path_len 81–281mm.
- eval2 due ~explore 536k (currently 318k). NOTE: this pass was mislabeled as "eval2 window" in my working notes — corrected: only baseline + eval1 have run.

### Pass 2 (v2) — 2026-07-12 18:17 EDT (~2h since Pass 1b)

**Overall: HEALTHY, one new WATCH (α at ceiling / entropy below target — the designed containment case).**

- **⚠ NEW WATCH — α pinned at the CEILING (0.1003 = F1 cap) since u≈165k; entropy fell through target: 1.00 → 0.64 → 0.33 → 0.14 → −0.02 → −0.29 (u217k).** This is the **mirror image of v1**: there, uncapped α rose to 0.45 and crushed the mean to force entropy up; here the cap holds, so the policy simply converges toward determinism while the entropy term pushes back at bounded strength. **The freeze channel is structurally blocked** — and the probe proves it: mean|a0| = **0.120 (3.7× baseline, strongest yet, all 4 dims balanced 0.105–0.120)**. The risk now is the OPPOSITE wall — mean-inflation rails (the LCCA failure mode): clamp_fraction crept 2%→5–6% (v1's collapse peaked 10.6%); F3 (mean penalty) + F4 (rail filter, still 0 rejections = successes non-railed) are the guards. **Escalate if:** clamp_fraction >10% or CLEAN_RAIL_FILTER rejections spike or probe median goes hard one-sided. **Resolve if:** entropy stabilizes (tanh-saturation equilibrium) and eval2 confirms quality.
- **Learning (best yet):** last-500 explore = 191 clean + **11 recovered** successes = **~40%** (↑ from 38%); cumulative online successes 379. Recovery behavior strengthening (11 recovered vs 6 last pass).
- **Stability:** 0 band violations (α=0.10026 IS the cap value exp(−2.3), not a breach), 0 nonfinite, 0 IPC timeouts, awac_mean 1.01–1.03 / max to 2.8, q1_mean −1.3→−1.5 (mild creep; v1 at same update count: −3.1), grad_norm ≤1.6.
- **Recovery/diversity:** off_br→320 trunc=False; buckle_phi to −0.506; 16 pids; path_len 103–264mm; **first v2 vessel_end (1×, benign terminal label)**.
- **Probe snapshot** = policy_430000 (u=209k, explore 437k). eval2 due ~explore 536k (~1.5h; currently 445k).

### Pass 3 (v2) — 2026-07-12 20:18 EDT (~2h since Pass 2)

**Overall: HEALTHY (WATCH continues, one trigger technically hit and overridden by ground truth).**

- **🏆 EVAL2 @ explore 510,628: quality 49.0% (48/98), speed 11.67mm/s, traj 220mm, mean reward +1.01 (first POSITIVE eval reward ever).** Trajectory: 6.1% → 30.6% → **49.0%** — 3.7× v1's all-time peak, +18.4pts over eval1. v1 at its second eval: 13.3% then collapsed; v2 is accelerating instead.
- **⚠ WATCH update (α-ceiling/entropy):** entropy oscillates negative (min −0.78 @u255k, now −0.16); α still pinned at 0.1003; **clamp_fraction spiked to 16.6% transiently (trigger >10% HIT)**, now back to ~7%. **Not escalating to PROBLEM** — the trigger was set against v1's context (sustained 10.6% during entropy-crash + mean-crush); here every ground-truth check contradicts pathology: eval quality +18pts, eval speed 11.7mm/s (fast, deliberate), CLEAN_RAIL_FILTER **still 0 rejections** (successes non-railed), probe 0.145 = **4.5× baseline** (median +0.102 = decisive forward, all dims balanced 0.109–0.145). **Refined trigger:** escalate only on (sustained clamp >12% over ≥5k updates) AND (rail-filter rejections >0 OR eval speed/quality degrading). The fast-aggressive profile (11.7mm/s) is the thing to keep watching — investigation already showed post-stall push-forward hardening.
- **Learning:** last-400 explore = 170 clean + 7 recovered = **44%**; cumulative successes 518. q1_mean −1.60 (mild creep, stable slope), awac_max spiking to 4.5 (discrimination widening), 0 nonfinite, 0 IPC timeouts, 0 band violations.
- Liveness note: CSV rows static across the 15s probe — normal burst-pattern idle (update budget balanced at ratio 0.5: 265.3k updates ≈ 530.3k explore × 0.5); STATUS loops climbing.
- Recovery/diversity: relax + buckle holding; 16 pids; path_len 109–268mm; vessel_end still 1 total.
- **Context:** the micro-recovery investigation (reports in `saved/v2_micro_investigation/`) explains the eval gap mechanics: eval failures are stuck-grinders; micro-recovery is noise-distilled and fading. eval3 due ~explore 760k.

### Pass 4 (v2) — 2026-07-12 22:18 EDT (~2h since Pass 3)

**Overall: HEALTHY, WATCH intensifying but refined trigger NOT met.**

- **⚠ WATCH (α-ceiling/entropy/rails) — deepening:** entropy now oscillates −0.5…−1.25 with a **min excursion to −3.92**; clamp_fraction hovers ~10% (last 0.1025) with 1,136 scattered rows >0.12 since u=265k (2.1% of rows — NOT the sustained ≥5k the refined trigger requires); `grad_norm_policy` samples 1.3–2.6 (grad_clip 1.0 binding regularly). **Trigger corroborators still absent:** CLEAN_RAIL_FILTER = 0 rejections (successes non-railed), probe **0.152 = 4.7× baseline** (healthy, balanced dims), no eval degradation (none run since eval2). The policy is riding a deterministic-aggressive regime; the guards are holding it off the rails where it counts.
- **Learning still climbing:** last-400 explore = 170 clean + **19 recovered** = **47%** (recovered successes ↑ from 7→19/400 — macro-recovery *growing*, contrary to the micro-recovery fade). Cumulative successes 667.
- Stability: 0 band violations, 0 nonfinite, 0 IPC timeouts, awac_mean ~1.01/max 2–3.2, q1_mean −1.7…−1.95 (slow creep continues).
- Recovery/diversity: relax + buckle holding; 16 pids; path_len 88–261mm; vessel_end still 1 total.
- **eval3 due ~explore 760k** (currently 648k, ~2h) — the decisive gate: v1's eval3 was the 3.1% collapse; v2's will adjudicate the deterministic-aggressive regime.

### Pass 5 (v2) — 2026-07-13 00:18 EDT (~2h since Pass 4)

**Overall: HEALTHY, WATCH deepening. eval3 IMMINENT (explore 750k vs ~760k threshold) — not yet fired; it is the adjudication.**

- **Probe best-ever & decisively forward:** mean|a0| **0.213 = 6.7× baseline** (median +0.18, only 12% of dims near-zero, all dims 0.12–0.21). The freeze channel is emphatically not engaging.
- **Explore learning still climbing:** last-400 = 195 clean + 10 recovered = **51%** (↑ from 47%); cumulative successes 823.
- **⚠ WATCH deepening (deterministic-aggressive regime):** entropy now −1.6…−2.0 (min excursion −2.0), clamp_fraction 0.10–0.13, longest **consecutive** clamp>0.12 run = 643 updates (still < the 5k-sustained escalation threshold). q1_mean creeping −1.9→**−2.27** (v1 at collapse was −4.7). **Refined trigger NOT met** — both corroborators absent: CLEAN_RAIL_FILTER = 0 rejections, no eval degradation (eval2 49% stands). α pinned at ceiling (0.1003), 0 band violations.
- Stability: 0 nonfinite, 0 IPC timeouts, awac_mean ~1.01 / max 2.1–3.9, grad_norm 1.1–2.9 (clip binding).
- Recovery/diversity: relax + buckle holding; 16 pids; path_len 80–265mm; vessel_end still 1 total.
- **eval3 (~760k) fires within minutes** — v1 collapsed to 3.1% here. If v2 holds ≥49%: the aggressive regime is validated, leave E5 unarmed (v3a's recovery levers supply the missing piece instead). If it regresses: the regime has hit the stuck-grinder wall → E5 (entropy floor) becomes relevant. Next pass adjudicates.

### Pass 6 (v2) — 2026-07-13 01:06 EDT — EVAL3 REGRESSED (49% → 30.6%). Keep running (user).

**Overall: WATCH (regime over-hardening now costing performance) — but NOT a collapse. Run healthy, continues per instruction.**

- **EVAL3 @ explore 770,976: quality 30.6% (30/98), speed 8.9mm/s, traj 192mm, reward −1.25.** Full trajectory: 6.1 → 30.6 → **49.0 → 30.6**. This is a real regression, back to the eval1 level — **but decisively NOT v1's freeze** (v1 eval3 = 3.1% at **0.95mm/s** frozen; v2 eval3 = 30.6% at **8.9mm/s**, still fast/deliberate). `checkpoint770976.everl` written; `best_checkpoint` stays eval2 (49%).
- **The WATCH is now materially engaged — confluence, not a single metric:** entropy deepened to **−2.2**, clamp_fraction to **0.138** (highest yet), q1_mean creeping −1.9→**−2.32**, deterministic probe softened **0.213→0.157** (still 4.9× baseline = OK, but down), AND explore success softened **51%→47%** (last-400: 183 clean + 4 recovered). All six move the same direction together → the deterministic-aggressive regime is over-hardening and starting to cost, not eval-seed noise alone (though 3-pt eval 30.6/49/30.6 has variance).
- **Refined trigger: one corroborator now MET** (eval degrading). Not escalating to PROBLEM: no freeze (kinematics healthy), no NaNs, 0 IPC timeouts, 0 rail rejections, 0 band violations, α at ceiling. User directs keep-running.
- **Interpretation (this is the predicted branch):** the aggressive push-forward regime has hit the **stuck-grinder ceiling** the v2 investigation forecast — micro-recovery faded, so on hard held-out meshes the policy grinds instead of retracting. v2 now serves as the **"no recovery fix" control curve**; every further eval documents the plateau/decline that v3a (E1 credit-assignment + E3 stuck-lane) and v3b (E4 restore curriculum) must beat. E5 (entropy floor) is now clearly relevant — but it belongs to the v3a machine-2 experiment, NOT to this control run.
- Recovery/diversity holding: relax + buckle active, 16 pids, path_len 80–265mm, vessel_end 1. Cumulative online successes ~823.

### Pass 7 (v2) — 2026-07-13 02:17 EDT — 🛑 PROBLEM: DEADLOCKED post-eval3 (same milestone as v1). F6 guard did NOT catch it.

**Overall: PROBLEM — v2 IPC-deadlocked ~69 min ago, immediately after eval3 (explore 770,976). NOT restarted (mandate; control run; container Up-but-hung, not exited).**

- **Deadlock confirmed:** CSV frozen at 381,055 rows across 3 reads / 30s; container CPU **0.20%**; last STEP wall_time = 69 min ago; workers alive (16 pids, MEM 15.65GiB) but idle. Container `Running=true ExitCode=0`.
- **🔑 SYSTEMATIC, not a one-off:** v1 AND v2 both deadlocked at the **exact same seam — right after the 3rd eval, explore ~770k**. My original v1 root-cause (one-off host suspend/clock-jump) is **wrong or incomplete**: this is a reproducible race in the post-eval → resume path that manifests at eval3. Last STEP before the freeze was a restarted worker (pid=8239, ep=1, global=600) finishing a max_steps episode — the eval-completion/worker-resync boundary.
- **🔑 F6 guard insufficient (and now understood):** the guard put a 900s timeout on the three weight-sync `_model_queue.get()` calls (`state_dicts_network/optimizer/scheduler`). But v2's main thread wchan = **`poll_schedule_timeout`** (a timed-poll spin), NOT v1's `anon_pipe_read` — the block is in the **result-collection / eval-resume path** (likely `explore_and_update`'s `while True` result loop or the eval() checkpoint/replay-buffer-save IPC), which F6 does not cover. `_model_queue` is correctly `mp.Queue` (timeout valid), so the guard isn't broken — it's just not on the hung path. 0 IPC TIMEOUT after 69 min confirms.
- **⚠️ CRITICAL for v3a (machine 2):** v3a inherits the SAME post-eval-resume path and will **deadlock at eval3 (~10h in)** the same way; the F6 guard will not save it. **Before machine 2 gets far, the deadlock protection must be broadened** — recommend a top-level training WATCHDOG (a thread that hard-exits the process if `update_step`/CSV makes no progress for N minutes), which catches the hang regardless of which `.get()` blocks, rather than chasing the exact call. This is a code change (needs approval) but is directly in the machine-2 path.
- **Nothing lost:** `best_checkpoint.everl` = eval2 (49%) is safe on disk; the eval3=30.6% regression finding stands (recorded Pass 6). v2's usefulness as the control curve ended at the hang.
- **Awaiting user decision:** (a) broaden the guard / add the watchdog for v3a before launching machine 2 (recommended), (b) restart v2 from a checkpoint or leave it dead, (c) root-cause the exact blocking call first (py-spy blocked by ptrace; would need /proc thread-stack spelunking).

### Pass 8 (v2) — 2026-07-13 04:15 EDT — 🛑 PROBLEM unchanged (still deadlocked, 184 min).

Confirmation-only. CSV frozen at 381,055 (unchanged), CPU 0.27%, hung **184 min** (>3h), `Running=true ExitCode=0`, F6 IPC guard still silent (0 — as expected; the hung `.get()` is not on the guarded path, see Pass 7). No self-recovery (deadlocks don't clear; this one has no covering timeout). Not restarted (mandate). Decisions from Pass 7 still pending — the load-bearing one is **the v3a watchdog before machine 2 launches** (v3a will hit this same eval3 deadlock ~10h in). No further per-2h analysis while frozen; will confirm state and note if the container exits.

### Pass 9 (v2) — 2026-07-13 06:15 EDT — 🛑 deadlocked, unchanged (304 min / >5h).
CSV frozen 381,055; CPU 0.24%; `Running=true`; IPC guard silent (0). No change since Pass 7. Not restarted. Pending: the v3a watchdog (before machine 2 launch) + v2 disposition.

### Pass 10 (v2) — 2026-07-13 08:15 EDT — 🛑 deadlocked, unchanged (424 min / >7h).
CSV frozen 381,055; CPU 0.18%; `Running=true`; IPC guard silent. Permanent hang — polling a frozen container yields no new data. Pending decisions unchanged (v3a watchdog is the load-bearing one).

### CLOSED — v2 killed + deadlock FIXED (2026-07-13, commit e3bf215)

User authorized. **v2 stopped + removed** (deadlocked ~9h; best_checkpoint eval2/49% safe on disk). **Root cause pinned on the live hung container before killing:** the `explore_and_update` result loop's TRAINER branch had no deadline (worker branch did) → a lost trainer update-result (mp.Queue race, reliably post-3rd-eval) spun `get(timeout=0.5)` forever. Evidence: main thread `poll_schedule_timeout`, all 17 subprocesses idle `futex_do_wait`, eval3 checkpoint + 1.1GB replay_buffer.npz both fully written (hang = resume path, not eval). Same seam as v1; F6 model-queue guard didn't cover it.
**Fix (synchron.py, both default-on):** (1) trainer-result deadline `EVE_RL_TRAINER_RESULT_TIMEOUT_S=1800` → restart trainer + continue (run self-heals past eval3); (2) progress watchdog `EVE_RL_WATCHDOG_STALL_S=2400` → os._exit(42) + thread-wchan dump on total stall (catch-all). Tested; in v3a launcher; handoff updated.
**v2's contribution stands as the control curve:** 6.1 → 30.6 → 49.0 → 30.6, deterministic-aggressive regime hit the stuck-grinder ceiling — the target v3a (E1/E3) + v3b (E4) must beat.

Monitoring cron `08c22f9c` (v2) is now moot (container gone) — its next firing will report "container gone"; can be left to expire or re-pointed at v3a once machine 2 launches.

## V2B — AWAC-stability run (launched 2026-07-14 ~15:21 EDT; v2 + penalty 0.02 + α-ceiling 0.135; NO Tier-A)

### Pass 1 (v2b) — 2026-07-14 16:48 EDT (~1.5h in; explore 113k, u≈31.8k)

**Overall: HEALTHY — all gates green (early; the decisive brake-vs-BC window is u>130k).**

- **Gates:** entropy recent-min **+2.0** (gate ≥−0.5 ✅; v2 @u30k was ~2.4 — comparable, divergence expected later); clamp max **1.95%** (≤10% ✅); α at floor 0.0067, 0 band violations ✅; rail filter 0 ✅; guards silent (0 IPC / 0 deadlock-restarts / 0 WATCHDOG) ✅; 0 nonfinite; 16 pids.
- **Baseline eval = 2.0% @ 0.36mm/s** (v2: 6.1% @ 0.54) — the 4× mean penalty acts during pretrain too → a slightly more timid pretrained start. Probe baseline (own u10k snapshot) mean|a0|=0.040. **Noted, not alarming** — but if eval1 lands far under v2's 30.6%, the brakes are taxing learning speed (watch item).
- Explore: ~200 online episodes, success ≈13–15% (26–30 succ incl. 2 recovered) — roughly v2's early pace, slightly more off-path wandering (higher retained entropy → more exploration; is_clean=0 share elevated).
- q1_mean −0.31…−0.75 (v2-like), awac_mean 1.01/max≤2.5, policy_loss 0.86–1.5.
- Curio from the probe: the pretrain policy already shows mild retract-coupling (buckled-tail P(ret)=0.158 vs 0.013–0.04 base) — seed-data-derived; track whether v2b *retains* it longer than v2 did (stronger entropy brake = noise-demos live longer).
- Next: eval1 ~explore 287k (~4h). v2 control at that point: 30.6%.

### Pass 2 (v2b) — 2026-07-14 18:50 EDT (explore 287k, u≈101k) — 🛑 GATE FAIL: OVER-BRAKED

**Overall: PROBLEM — the run failed its gates in the OPPOSITE direction. Not collapsed-aggressive; never-launched.**

- **eval1 @ explore 276,754: 0.0% (0/98), speed 0.256mm/s** (v2 control: 30.6% @ 4.45mm/s). Below even the pretrain baseline (2.0%).
- **Freeze-probe GATE FAIL:** latest snapshot (u=96,376) mean|a0| = 0.041 = **1.0× its own baseline** — the deterministic mean is byte-identical-to-3-decimals to pretrain across 96k updates. The policy never grew decisive actions at all.
- **Mechanism (differs from the written peel rule):** α sat at the FLOOR (0.0067) the entire run — the raised ceiling (−2.0) **never engaged** and cannot be the crusher. The culprit is the **4× action_mean_penalty (0.02)**: its constant restoring gradient neutralizes the small early advantage-tilt (AWAC weights ~1.0 ⇒ near-uniform BC over noise-symmetric buffer actions ⇒ the net "grow decisive means" signal is weak, O(0.01–0.05)); at 0.005 (v2) the tilt won, at 0.02 it loses. Entropy pinned at 2.6 all run = the sound of NO mean growth (v2's entropy decline 2.7→1.0 during its healthy phase WAS mean growth).
- Explore success 14% (56/400; cumulative 86) — pure σ=1 noise wins; classic "explore hides a dead mean."
- Housekeeping green (0 nonfinite, 0 IPC/watchdog, 16 pids, liveness OK) — the run is healthy machinery executing a mis-tuned objective.
- **Written peel rule ("probe <1.25× → revert α ceiling") does not fit the observed mechanism** (ceiling never engaged) → deviating from the pre-authorized script requires the user: recommend KILL v2b + relaunch **v2c = penalty 0.01** (split v2's too-weak 0.005 and v2b's too-strong 0.02), keep ceiling −2.0 (moot early, useful late). Bonus: the parked `--resume` branch is NOT needed here (nothing worth resuming). Awaiting user decision; run left running per mandate.

### Pass 3 (v2b) — 2026-07-14 20:47 EDT — 🛑 KILLED (over-braked, confirmed). Cron retired.

eval2 also **0.0%** (@ explore 277k, speed 0.26mm/s); probe at u=160k still **1.0× baseline** (mean byte-frozen at pretrain across 180k updates); entropy pinned 2.48 the whole run = no mean growth. Guards clean (0 IPC/deadlock/watchdog). **User authorized kill; v2b stopped + removed; cron ae62e63a deleted.** Confirmed diagnosis: `action_mean_penalty 0.02` neutralized the weak early advantage-tilt → the engine never started (α floor the whole time; raised ceiling never engaged). **This ends the AWAC penalty-bracketing line** — superseded by RL_PARADIGM_ROADMAP.md (constraint family is the wrong class; migrate to RLPD → privileged-actor teacher-student). Nothing worth resuming (no checkpoint above the 2% baseline). GPU now free for Phase 1 (RLPD).

### Pass 11 (v2) — 2026-07-13 10:24 EDT — container GONE (expected). Cron retired.
v2 confirmed removed (killed last turn; deadlock fixed in e3bf215). Only the old `rcca_proc_harvest` (Exited 39h) remains. **Local v2 monitoring cron `08c22f9c` deleted** — nothing left on machine 1 to watch, and v3a monitoring runs on machine 2 via `monitoring/monitor_pass_v3a.sh`. This closes the v2 monitor log. Next active monitoring begins when v3a launches on machine 2.

### Pass 6 (v1) — 2026-07-12 10:43 EDT

**Overall: PROBLEM (unchanged — still deadlocked, ~3h since freeze onset).** CSV frozen at 372,036 rows, CPU 0.36%, container Up (never restarted). No manual intervention detected. Per mandate, no changes made. **Status: awaiting user decision** — (a) investigate the freeze-collapse mechanism first (timidity-EV check from step logs, buffer-poisoning check), vs (b) restart now from the eval-2 `best_checkpoint` + matched `replay_buffer.npz`, vs (c) fresh restart. Deadlock hardening (timeouts on `_model_queue.get()`) also pending approval. Monitoring continues; passes will stay brief until the run is live again.

### v1b EVAL1 — 2026-07-18 03:42 EDT — **68.4% (67/98, reward +3.64) — FIRST TRAINED EVAL OF THE PROGRAM; R4 GATE PASSED**
Gate: 3 UNIQUE checkpoint md5s (v1: all identical), STALLED restarts = 0 (NET_SYNC lines go to main.log not stdout — verify count next pass). vs bars: heuristic phase-band top ~47% → **+21pp** (clears the band bar by ~4× noise); v2 ceiling 49% → +19pp; field best 59% → +9pp; already inside the deep-dive 62–73% projected ceiling with NONE of the v2 bundle. The residual+privileged-actor design COMPOUNDS once weights actually deploy. Watch next: eval2 trend, Q-health, recovery composition (grind→soft conversion expected to be the mechanism).

### v1b EVAL2 — 2026-07-18 10:54 EDT — 58.2% (57/98, reward +2.01) — dip from 68.4 peak; machinery CLEAN
u=491k (realized UTD ~0.95 — R2 fix doubled trainer throughput), alpha 0.015/H 1.23 (sharp phase), q1m −0.19, nf 0, STALLED 0, NET_SYNC_ALARM 0. Still +11pp above heuristic band top and +9pp over v2 ceiling. The −10-ep drop is REAL training dynamics (first time observable) or policy-induced eval-orbit shift — not infra. Eval3 decides: recovery→trend resumption vs the consolidation pathology finally observed for real. If monotone decline confirms at eval3-4, activate: alpha investigation (now with genuine dynamics) + v2 bundle gate.

### v1b EVAL3 — 2026-07-18 17:55 EDT — **81.6% (80/98, reward +4.48) — PROGRAM RECORD; the eval2 dip was consolidation**
Curve: H0 36.7 → 68.4 → 58.2 → **81.6**. +34pp over heuristic band top; +23pp over field best (59); ABOVE the deep-dive 62–73% projected ceiling with none of the v2 bundle; 3.4pp from the 85% paper target at 12.6% of explore budget. Eval3 window: stuck-eps 39 (from 70 at eval1 — avoidance holding), events 65, unrec only 14, ret-depth succ/fail 2.2/23.3mm — successes now retract MORE than v2-era successes ever did (recovery as strategy), failures show deep retraction attempts (23mm) that don't convert yet. Health: u=736k, alpha 0.011 stable, H 0.994 on-target, q1m −1.05 (Q decline continues — now clearly NOT tracking performance; pessimism watch), q1L 0.035, nf 0, STALLED 0. Next: eval4 confirms ≥80 stability; Q-mean/performance divergence is the one anomaly to explain; early P2b retention probe recommended per user discussion.

### v1b EVAL4+5 — 2026-07-19 00:45 / 07:20 EDT — **51.0% then 71.4% — OSCILLATION + critic value-drift**
Curve: H0 36.7 → 68.4 → 58.2 → 81.6 → **51.0 → 71.4** (±30pp swings). 81.6 was a CREST, not a ceiling — policy is volatile, not converged.
Recovery composition tracks capability (not orbit-only): eval3(81%) unrec 14 / soft 3; **eval4(51%) unrec 45 / soft 0 / grind 11** — recovery collapsed at the low crest; eval5 recovered (comp pending, tracker timed out on log size).
CRITIC VALUE-DRIFT (the real anomaly): q1_mean monotone negative 700k→1.22M: −1.05 → −1.7 → −2.3 (target-q tracks it, so self-consistent, NOT divergence) while policy performs 51-81%. **policy_loss FLIPPED sign +0.3→+1.0 growing** (actor can't find actions the pessimistic critic likes); q1_loss rising 0.02→0.11 noisy. alpha stable 0.012-0.027, entropy on-target ~1.0, nf 0, no OOM/stall. Explore buckets STABLE 55-70% (stochastic average robust; deterministic eval catches the unstable point).
READ: not broken (explore stable, infra clean), but not converged and showing a slow critic-pessimism build. 81.6 is not the number; honest center ~65% ±15 (real volatility + orbit ±10-15). Budget only 21% spent.
IMPLICATION: (a) cannot trust any single eval nor cleanly pick a best checkpoint until P0 eval-state-reset fix; (b) premature to distill P2b from an oscillating teacher; (c) Q-drift is the leading indicator — watch eval6-7 for monotone-worsen (→ stop+fix) vs oscillate-stable (→ continue).

### v1b RECOVERY AUDIT (full single-pass, 5930 eps) — 2026-07-19 — CORRECTS "soft recovery emerging"
Soft% FLAT 2-6% across all 5200 explore eps, NO trend. Eval soft%: H0 0, E1 2, E2 25(ARTIFACT: only 44 events), E3 3, E4 0, E5 3. The eval2 "24% soft = strategy" claim RETRACTED — small-denominator artifact.
REAL mechanism of 36%→60%: BETTER CLEAN NAVIGATION (avoid getting stuck), NOT learned recovery. unrec = largest category every explore window; stuck episodes still mostly grind/fail. E4(clean 98ep, 51%) unrec 45 vs E3 unrec 13 — oscillation = whether the checkpoint handles junction-capture, which is STILL UNSOLVED. Deep-RVA commit-without-retract (deep-dive finding) persists — policy dodges it, doesn't defeat it. → v2 bundle (wrong-branch obs + stuck-restore curriculum) is the correct next lever, aimed exactly here.

### v1b SUCCESS×STALL CROSS-TAB — 2026-07-19 — REFINES the recovery audit (2 failure channels)
Per-eval outcome × stall decomposition (succeed-clean/succeed-via-recovery | fail-stuck-forever/fail-escaped-lost/fail-clean-wrongturn):
H0 36%: 37/9 | 66/4/11 ; E1 62%: 38/36 | 30/8/7 ; E2 54%: 64/4 | 13/16/30 ; E3 74%: 66/23 | 13/1/17 ; E4 51%: 48/2 | 45/3/0 ; E5 71%: 71/24 | 27/5/7.
CORRECTION to prior "recovery flat/irrelevant": recovery-to-SUCCESS grew H0 9 → E1/E3/E5 36/23/24 (25-50% of successes). Only SOFT stayed flat; the growth is GRIND+HARD (brute escapes). 
KEY: TWO independent failure channels — (1) permanent-stall (recovery-visible; crashed E4 = 45/48 fails were stuck-forever), (2) smooth-wrong-turn/timeout (recovery-INVISIBLE, no stall event; dragged E2 = 30 clean-fails). Oscillation = which channel dominates per checkpoint; E3/E5 good = both quiet. Success partially follows recovery (channel 1) but E2 proves a large failure mass is off-radar. v2 bundle wrong-branch obs targets channel 2; stuck-restore targets channel 1 — both needed.

### v1b EVAL6 — 2026-07-19 15:16 EDT — 61.2% (60/98, reward +3.05, expl 1.52M) — Q-DRIFT VERDICT: mean-reverting, CONTINUE
Curve: 36.7→68.4→58.2→81.6→51.0→71.4→**61.2** (still ~51-81 band, center ~63). Q-mean NOT spiraling: dipped -2.7 (u=1.14M) then RECOVERED to -1.0..-1.6 (u=1.5M); policy_loss fell 1.0→0.2-0.5. alpha 0.008 stable, H ~1.0 on-target, q1L O(0.01-0.04). Infra clean: OOM=false, 0 restarts, 0 NET_SYNC_ALARM, mem 79%. Pre-registered gate → oscillate-stable, CONTINUE. No single-eval trust until P0 reset fix. Budget ~25% spent.
NOTE: log parsing now slow (1.4GB+); use tail-based CSV reads.

### v1b EVAL7 — 2026-07-20 (expl 1.76M) — 54.1%; "3-eval decline" is NOISE, explore STABLE
Evals: ...71.4→61.2→54.1 (looks declining) BUT explore buckets last 2400 eps FLAT ~62% (60,61,77,65,54,60,59,61,71,65,59,66,65,54,57,59 — no trend). => deterministic-eval jitter (marginal-episode sensitivity), NOT policy degradation. Confirms: explore-avg is the honest ruler (~62%, +25pp over H0), single evals are noisy on top. Q-mean oscillating -2.8..+0.5 mean-reverting (not diverging); alpha/H stable; mild q1L uptick 0.04→0.10 (watch). OOM=false, 0 restart, mem 81%. CONTINUE. REPORTING RULE: use explore-avg / multi-eval-avg, never cherry-picked single eval (the 81.6 was a crest).

### v1b EVAL9 — 2026-07-20 (expl 2.26M) — 54.1% (53/98) THIRD IDENTICAL eval → CONVERGED, not oscillating
E7/E8/E9 all exactly 53/98 = deterministic policy stabilized on a fixed solvable seed set. Early ±30 swings were training-phase volatility; now flat. Confirms PLATEAU (explore ~62% flat 1.1M+ steps, eval ~54% converged). v1b has given its answer. Awaiting user decision on next step (a: build v2 bundle while v1b runs / b: stop now / c: run to full budget). No action taken — read-only hold.
### v1b EVAL11 — 2026-07-21 (expl 2.77M) — 56.1% (55/98). Plateau continues (eval band 54-67, explore ~61%). No change; holding for user direction.
### v1b EVAL12 — 2026-07-21 (expl 3.01M = 50% budget) — 50.0% (49/98). Plateau band unchanged. Holding for user direction.
### v1b EVAL13 — 2026-07-21 (expl 3.26M) — 51.0% (50/98). Plateau band unchanged. Holding for user direction.
### v1b EVAL14 — 2026-07-22 (expl 3.52M) — 54.1% (53/98). Plateau band unchanged. Holding for user direction.

### v1b ENDED — 2026-07-22 06:07 EDT — watchdog exit(42) after OOM-induced deadlock. RUN CLOSED.
Chain: eval15 done 05:15 (54.1%, expl 3.52M) → buffer saved → explore resumed → progress stalled 2402s → WATCHDOG os._exit(42). OOM=true: 4-day mem creep (79-81%) crossed 23.5GiB cgroup cap → kernel OOM-killed a worker → futex deadlock (22 threads) → watchdog fired. **The R2 watchdog worked AS DESIGNED** (clean exit vs silent hang). Died at 3.52M/6M PLATEAUED — zero science lost. Banked+resumable: checkpoint3520105 (latest), checkpoint757854 (best eval 81.6% crest), incr buffer→3.55M, replay_state.
RESULT (final): residual-on-heuristic privileged-actor teacher = ~61% explore / ~62% eval-avg stable, no collapse, over H0 37%(this launch)/~47%(band) and v2 49%-then-collapse. +13-15pp over best baseline, STABLE. This is the P2a teacher ceiling for the paper.
NEXT (forced): GPU free. Options — resume v1b (pointless, plateaued) / build+launch v2 bundle (stuck-restore + wrong-branch obs; needs user go + obs-change approval) / P2b student distill from banked checkpoint. Monitor cron 74db498c DELETED (container gone). Awaiting user direction.

# RCCA P2 teacher v1bp — v1b + ONLY the v3c reward pair (ACTIVE, launched 2026-07-24)
Single-variable experiment: byte-identical v1b science + --progress_tip_mode avg --avg_gw_weight 0.5 --cath_slack_coef 0.5 (machine-2 pair, port @ b3254d9, 6/6 tests incl. pump regression). NO crunchpass (parked). Startup echoes confirmed all four config lines ([P2] residual + privileged actor; [v3c] cath-slack + tip-avg). Monitors: cron d7f9a0bb (:37 odd hours), persistent watcher bynggknin (evals + sustained q1<-2.5 + container-down).
GATES: (1) H0 SUCCESS in v1b's 37-47% band = port integrity (reward-blind heuristic); (2) eval1-3 vs v1b's 68.4/58.2/81.6 — the user's target window; (3) cath-lead% stays low (v1b: 6% at crest, 58-62% in decline — the pair taxes exactly that drift); (4) q1_mean sustained <-2.5 = handoff §7 critic divergence.
TARGET REVISED (user, 2026-07-24): aim = reach AND HOLD 75-80% — v1b's crest was REAL capability (eval3 81.6 + explore ~77 same era, both rulers agreeing, built on the 6%-cath-lead gait) and was then lost to shove drift. v1bp success = crest becomes steady state; not the 62 average. Cron re-armed as acaa5224 with these bars.

### v1bp EVAL1 — 2026-07-25 03:09 EDT — 33.7% (33/98) — TRANSITION DIP, MECHANISM CONFIRMED WORKING
Eval1 low (v1b: 68.4; own H0: 45.9) BUT the pair's mechanism readout is exactly as designed: recent explore cath-leads+50mm = **1.3%** (v1b same-era 18-33%, decline 58-62%), gw-leads 57%, mean gw 120mm vs cath 75mm — THE SHOVE IS GONE, policy is wire-led. cath_slack quiet (0.7%>15mm). q1m +4.38 (elevated, no §7 divergence — 0 rows <-2.5; watch optimism instead). Recent explore 48.4% > eval1 (recovering since snapshot). Reading: policy abandoned the catheter-shove (which converted shallow seeds) before the wire-led gait matured — machine-2's dip-then-recover arc, one phase later (no pretrain pathway here). HOLD to eval2-3; ALERT if eval2 doesn't move up or cath-lead re-inflates. NOTE self-correction: first parse of inserted[] was field-shifted (reported mean_gw=0/cath-lead 91% — WRONG); corrected parse above.

### v1bp EVAL2 — 2026-07-25 09:44 EDT — **66.3% (65/98, reward +3.24) — RECOVERY CONFIRMED, ABOVE v1b at same index**
Arc: H0 45.9 → eval1 33.7 (transition dip) → **66.3** (+32.6pp; v1b eval2 was 58.2 → v1bp +8pp ahead). Machine-2's dip-then-overshoot pattern REPRODUCED on this line. Mechanism held through recovery: cath-leads+50mm = 11.1% (vs v1b same-era 18-33%; decline 58-62%), mean gw 140 vs cath 135 — balanced/wire-led two-device configuration at DEPTH (both ~140mm = coordinated deep work, the crest signature). q1m +0.23 (normalized from the +4.4 spike — optimism watch closed), alpha 0.008 stable, H 0.93 on-target, nf 0, no OOM, mem 77%. Recent explore 57.9% and climbing. NEXT BAR: eval3 vs v1b's 81.6 crest and entry into the user's 75-80 hold zone; watch cath-lead stays ≤15% and whether the crest PERSISTS this time (the pair's core promise = taxing the drift that took v1b's crest away).

### v1bp EVAL3 — 2026-07-25 14:29 EDT — ⚠️ 43.9% (43/98, reward -0.30) — DOWN-SWING, critic healthy, DECISION GATE at eval4
Arc: H0 45.9 → 33.7 → 66.3 → **43.9** (reward 0.93→0.93→3.24→**-0.30**, first negative). §7 NOT triggered: q1m oscillates bounded +0.3..+1.2 (never toward -90), alpha 0.008-0.019, H ~1.0, nf 0, no OOM/restart — LayerNorm+entropy-free backup holding as hoped. BUT concerning mechanism: current explore near-zero deep deployment (gw>100mm only 2% of steps vs eval2's deep gw~140/cath~135), explore fell ~58→41%, mean reward NEGATIVE = costly shallow failures (not orbit noise). Two hypotheses: (1) oscillation (v1b swung ±30 and settled; healthy critic favors this) vs (2) reward-pair timid optimum (cath_slack over-tax teaching "don't advance"). HOLD, do NOT kill (v1b lesson). **GATE: eval4 (~6h) — bounce toward 60s w/ deep deploy = oscillation, pair OK; stay low + negative + shallow = timid failure, response = lower --cath_slack_coef 0.5->0.25 or revert (single-variable = clean adjust).** Caveat: shallow-deploy read from a possibly eval-contaminated window; robust signals = eval 43.9 + reward -0.30 + explore 41.

### v1bp EVAL4 — 2026-07-26 01:09 EDT — 61.2% (60/98, reward +1.98) — GATE=OSCILLATION (not collapse); pair SAFE but NOT WINNING yet
Arc: H0 45.9 → 33.7 → 66.3 → 43.9 → 61.2. eval3 was a down-swing, eval4 recovered (+reward). Mechanism holding: cath-lead 7.8%, gw>100mm back to 24% (from eval3's 2%). Critic healthy (q1m -0.16 bounded, nf 0), no OOM, mem 82%.
HONEST BROADER READ (matched explore budget): v1bp eval-avg ~51 (57 ex-dip) vs v1b ~65; explore ~48-58 vs v1b ~62. THE PAIR FIXED THE MECHANISM (shove gone, confirmed) BUT NO NET SUCCESS GAIN over v1b through 1M — crests ~66 < v1b's 81.6, oscillating around comparable-to-lower center. Too early (v1bp 1M vs v1b's crest@758k then decay to 62 plateau@3.5M); the pair's real promise = ceiling that HOLDS (taxes drift), untested yet.
DECISION POINT ~1.5-2M: bar = eval center clearly >v1b 62 OR crest >70 held across 2 evals. If still ~55 oscillation w/ no held crest → pair alone insufficient for this line → stack crunchpass (v1d): pair fixed the SHOVE channel, crunchpass targets the SIPHON/junction channel (different failure). Continue meanwhile (healthy/cheap; deep wire-led gait harder to build than the shove, may still mature).

### v1bp EVAL5 — 2026-07-26 07:56 EDT — 52.0% (51/98, reward +1.88) — oscillation continues, trend BELOW v1b
Arc: 45.9→33.7→66.3→43.9→61.2→52.0. Eval-avg(e1-5) v1bp 51.4 vs v1b 66.1; at matched 1.27M budget v1b eval5 was 71.4 vs v1bp 52.0. Center ~51, NO crest above 66 (bar was >70 held). Mechanism SOLID (cath-lead 1%, gw>100mm=37% — deepest yet, wire-led gait working), critic healthy (q1m -0.6 bounded, nf 0), no OOM. So: pair mechanistically successful, but 5 evals confirm it tracks ~15pp BELOW v1b — the shove it removed was capping v1b at a HIGHER number than the wire-led gait currently reaches. Committed decision point = eval6 (~1.5M): if still ~51 center → pair-alone insufficient → pivot to v1d (stack crunchpass for the siphon/junction channel). Holding to that gate, not jumping early.

### v1bp EVAL6 — 2026-07-26 12:47 EDT — 65.3% (64/98, reward +2.57) — DECISION POINT (~1.5M); resolving via SIPHON split
Arc: 45.9→33.7→66.3→43.9→61.2→52.0→65.3. High-swing bounce; last-3 center ~59.5 (up from ~51). At 1.5M v1bp 65.3 > v1b 61.2, but 6-eval avg 53.7 < v1b 65.3 — overall still borderline-below, bar (center>62 or crest>70 held) not strictly met. Deep-deployment TREND rising (gw>100mm 2→24→37%). Resolving the borderline by the ACTUAL goal: siphon success (pooled eval4/5/6) vs v1b baseline (CCA 100/ICA 86/siphon 53). If v1bp siphon > 53 → pair wins where it matters (pivot decision changes). Result pending (heavy parse).

### v1bp SECTION VERDICT (CORRECTED) — 2026-07-26 — pair is siphon-FAVORABLE; my earlier "pair hurt siphon" RETRACTED
FAIR matched-overall comparison (v1b e2+e4 55.9% vs v1bp e4/5/6 59.4%):
  CCA 100/100 | ICA-mid 57/53 | siphon **8.0% (v1b) vs 16.2% (v1bp)** | overall 55.9/59.4
=> At matched phase the pair ~DOUBLES siphon (16 vs 8; wide CIs, favorable direction) and is comparable-to-higher overall. My prior "pair hurt siphon" was an artifact of comparing v1bp's pool to v1b's 81.6 CREST (orbit-lucky outlier, ~18pp above its explore). Also retracts "v1bp 15pp below v1b" (that vs v1b crests).
VERDICT: pair = mechanism-clean + critic-healthy + siphon-favorable; but siphon still LOW absolute (16% vs the ~53% shown reachable). Siphon is the frontier; needs deep-turn/junction navigation = crunchpass.
DECISION: pair earned its place → next = v1d (pair + crunchpass). Recommend kill v1bp (answered its question at 1.5M) + launch v1d. Awaiting user go.
### v1bp EVAL7 — 2026-07-26 (expl 1.76M) — 53.1% (reward +2.30). Oscillation confirmed as PLATEAU (~54% center, 7 evals 34-66, no crest>66). Reinforces verdict: pair siphon-favorable but settled; siphon frontier (16%) needs crunchpass. Decision unchanged: kill v1bp + launch v1d (pair+crunchpass). Awaiting user go.
### v1bp Q1_MEAN ALERT — 2026-07-26 — FALSE-POSITIVE (deep oscillation dip, NOT §7 divergence). q1m oscillates -2.6..+0.8 (bounced +0.79 at u=1841k, now -2.63), target_q tracks it, nf=0, and EXPLORE HEALTHY 65%/reward+3 — policy not degrading (machine-2 divergence = monotone q1->-90 + policy collapse; neither here). Critic swings with the policy's 34-66 eval oscillation; -2.63 = deepest trough yet but within character. No action. Watch only if q1m<-3/-4 WHILE explore drops. Decision unchanged: kill v1bp + launch v1d.
### v1bp EVAL8 — 2026-07-27 (expl 2.00M) — 54.1% (reward +2.20). Reaches the 2M decision mark. Plateau DEFINITIVE: 8 evals ~54% center, siphon-favorable (16% vs v1b 8%) but capped. Decision window fully reached. Recommendation UNCHANGED: kill v1bp + launch v1d (pair+crunchpass). NOTE: ~3 days runtime, 2M/2M replay — approaching v1b's day-4 OOM horizon; killing for v1d also avoids that. Awaiting user go.
### v1bp EVAL9 — 2026-07-27 (expl 2.26M) — 64.3% (reward +2.95). In-band high-swing (plateau 34-66, center ~55 over 9 evals). No change; decision stands (v1d). Awaiting go.

### v1bp ENDED — 2026-07-27 — OOM (exit 42, as predicted at the day-4 horizon). RUN CLOSED.
2.26M explore, ~3.5 days, 2M/2M replay → kernel OOM-kill → watchdog exit(42) (worked as designed, same as v1b). Banked+resumable (11 ckpts, buffer→2.28M). Plateaued ~54% — zero science lost. Cron acaa5224 deleted; watcher stream ended.
FINAL v1bp VERDICT: reward pair = mechanism-clean (shove gone), critic-healthy (no true §7 — the one alert was a deep oscillation dip w/ explore healthy at 65%), siphon-FAVORABLE at matched phase (16% vs v1b 8%), overall comparable. But siphon still LOW absolute (16%) = the frontier; pair alone plateaus ~54%. Pair VALIDATED as a keeper.
NEXT: GPU free (OOM resolved the transition). v1d (pair + crunchpass) ready + recommended — pair deploys deep, crunchpass targets the deep-turn/junction cap on siphon. AWAITING USER GO (will not auto-launch a multi-day run).

## ★ MULTI-ANATOMY EVALUATION — 2026-07-28 — THE 81.6% MODEL IS A 57.1% MODEL
**Bug (verified empirically, not inferred):** DualDeviceNav_train builds env_eval as
DualDeviceNavRCCAVaried(episodes_between_change=10**9); RCCAVariedFromMesh.reset() regenerates only
when episode_nr % episodes_between_change == 0, so a per-episode seed merely RE-SEEDS the RNG.
Hashing vessel_tree.branches coordinates across resets: geometry byte-identical (mesh_fingerprint
string changed while g0 never incremented). => EVERY eval of EVERY run so far ran in ONE vessel tree;
the 98 seeds varied only the TARGET and the device start rotation.

**Fix:** new env_eval_factory (synchron.py/agent.py, mirrors env_train_factory) => per-worker anatomy
streams; env5 EPISODE_START now logs anatomy=<branch-hash> + mesh_fp so diversity is VERIFIED;
standalone training _scripts/eval_anatomies.py + launch_eval_anatomies.sh.

**Result — v1b checkpoint757854 (the 81.6% crest), 98 episodes / 50 distinct anatomies:**
  OFFICIAL 56/98 = **57.1%** (95% CI 47.3-66.5). Integrity: 98 starts / 98 rows / 0 dropped;
  official (infos[-1]['success']) == log-derived == 56; no seed loss.
| section | 1 anatomy (eval3 81.6%) | 50 anatomies | delta |
|---|---|---|---|
| CCA      | 100% | 96.0% (24/25) | -4  |
| ICA-mid  |  86% | 61.9% (26/42) | -24 |
| siphon   |  53% | 19.4% (6/31)  | -34 |
| OVERALL  | 81.6%| **57.1%**     | -24.5 |
Section-wise comparison controls for target mix (my seeds are slightly deeper: median path 185 vs
167mm), so the drops are genuine cross-anatomy generalization loss.

**Anatomy scope (verified, carotidsiphon.py):** chain ends at the C7 terminus (MCA/ACA bifurcation,
~265mm). M1/MCA is NOT modelled — it begins at that bifurcation. Paper must say "to the ICA
terminus", NOT "to M1/MCA". Depth bins map to: CCA(<146) = CCA+proximal C1; ICA-mid(146-210) =
distal C1 + petrous C2; siphon(>=210) = cavernous C4 genua + supraclinoid C6 -> C7.

**Implications:** (1) every absolute eval number in this program is single-anatomy and must be
re-measured before publication; (2) RELATIVE comparisons (v1b vs v1bp vs H0) remain valid — same
protocol throughout; (3) the paper's generalization claim must rest on multi-anatomy numbers only.
