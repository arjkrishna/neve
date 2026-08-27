# rcca_procedural_v3a — machine-2 monitor log

Read-only passes every ~2h (session cron `f023bf28`, at :23). Healthy ranges per
machine-1 instructions (2026-07-16). v2 control curve: 6.1% → 30.6% → 49.0% → 30.6%.

**Run provenance:** run 1 (`2026-07-16_213028`) launched 21:30, pretrain-baseline
Quality **36.7% (36/98)**, then the trainer-result deadline (1800s, counts from cycle
start — synchron.py:715) force-restarted a HEALTHY trainer ~90% through its first
~21.5k-update online budget (~32 min needed on this 12-core host), rolling it back to
the u=10000 pretrain weights; that would have repeated every cycle. Relaunched 23:07 as
run 2 (`2026-07-16_230731`) with `EVE_RL_TRAINER_RESULT_TIMEOUT_S=5400`,
`EVE_RL_WATCHDOG_STALL_S=7200`. Run-1 stuck pool archived at `saved/rcca_v3_stuck_run1/`
(72 files; contains eval-anatomy captures — exclude eval mesh fingerprint before v3b
screening). Guard-timeout change is observation-infrastructure only (v2 has no such
guard); training math untouched.

---

## Pass 1 — 2026-07-16 23:2x (run 2, mid post-pretrain BASELINE eval)

**Phase:** pretrain done (10k updates, ~8 min); baseline eval (~98 seeds) in progress
since 23:16. CSV static at u=10000 is expected during eval — liveness "10001/10001" is
benign here, NOT a stall.

| metric | value | range | verdict |
|---|---|---|---|
| alpha | 0.006738 | [0.0067, 0.100], moving | at floor rail post-pretrain (same as run 1) — carried WATCH: must lift off during online phase; pinned >50k updates = ALERT |
| entropy_proxy | 2.624 | gentle decline expected | baseline value, pre-online — OK |
| q1_mean | −0.239 | bounded | OK (run 1 same point: −0.372) |
| nonfinite counts | 0,0,0 | 0 | OK |
| clamp_fraction | 0.29% | <5% | OK |
| awac_weight_p99p1 (E1b) | 16.36 (recent 8.5–41.0) | 5–20 | **OK — in band** (v2 was ~1.7 BC-degenerate) |
| aux_loss (E2) | 0.336 (recent mean 0.40) | O(0.1–2), falling | **OK — znorm live** (12.7 → 0.34 over pretrain) |
| CLEAN_RAIL_FILTER | 0 rejections | few % of successes fine | OK (also 0 in main.log — check itself dead on stdout, verified against main.log directly) |
| freeze / retract-vs-slack / aux R² probes | no policy snapshots yet | — | pending first explore cycle (probe baseline banks next pass) |
| eval trajectory | mid-eval; worker tally so far 26 success / 56 done | eval1 gate ≥ 30.6% | pending; run-1 baseline was 36.7% (36/98) |
| stuck-pool harvest (E4) | 8 files (fresh pool, run-1's 72 archived) | growing | OK |
| IPC guard | 0 timeouts | 0 | OK |
| RestartCount / OOMKilled | 0 / false | 0 / false | OK |
| container mem | 14.47 GiB / 30.97 GiB (46.7%) | watch toward ~87% (machine-1 OOM point) | OK |
| CPU / GPU | 997% CPU (eval, 16 workers) / GPU 2% (eval is CPU-bound) | — | OK |

**Recovery tracker (verbatim):**
```
RECOVERY TRACKER: run=2026-07-16_230731_rcca_procedural_v3a episodes=84 (window=400)
window       eps  stuck-eps  events  soft%  hard%  grind%  unrec%  succ%  ret-depth succ/fail (mm)
    1-84       84       0         0    0.0    0.0     0.0     0.0   50.0   nan / nan
GOAL: soft% RISING across windows and retract-depth succ < fail; v2's cycles faded ~86->58->28 across run thirds — that fade recurring (3+ windows monotone down + flat evals) = ALERT.
```
Note: 0 stuck events across 84 baseline-eval episodes at default thresholds
(PUSH_MIN=2.0, STUCK_STEPS=12). Machine-1 guidance: if events stay ~0 once explore
episodes accumulate, lower PUSH_MIN to 1.0 and log the change. Deferring any threshold
change until pass 2 — eval episodes of a pretrained policy may legitimately rarely
grind 12 straight steps. Mid-eval success (42/84 = 50%) is running notably above run
1's 36.7% final — pretrain-to-pretrain variance; final Quality line is authoritative.

**VERDICT: OK** — mid-baseline-eval; all live gates green (E1b in band, E2 live,
nonfinite 0, no restarts, memory comfortable). Carried watches: (1) alpha at floor
rail — must move online; (2) recovery-tracker detector sensitivity; (3) first online
cycle under the raised 5400s deadline completes without a trainer restart.

**Addendum 23:46 — run 2 baseline landed: Quality 0.500 (49/98), Reward +1.67,
explore=0.** Pretrain-only policy already equals v2's all-time PEAK (49.0% @ eval2)
and starts with positive eval reward (v2 needed online learning to get there).
Run-to-run pretrain variance is large: run 1 36.7% / run 2 50.0% from identical
seed+config (13-pt spread from pretrain stochasticity) — interpret eval1 against this
spread, not as a point estimate. Formal eval1 gate (≥30.6%) is near-certain; the real
question online learning must answer is whether it adds anything above a 50% baseline.

---

## Pass 2 — 2026-07-16 22:22 EDT (run 2, online explore ~211k steps / u=115,466)

**Phase:** online training, ~105k online updates done. Zero trainer restarts — cycles
run ~2,943s wall (explore-bound), i.e. the OLD 1800s deadline would have fired on
every cycle; the 5400s fix is validated. eval1 (~287k explore) ≈ 2h away.
(Note: the 20:23 scheduled pass didn't fire — cron only runs while the session is
idle; this pass covers it.)

| metric | value (Δ vs pass 1) | verdict |
|---|---|---|
| alpha | 0.1003 — at CEILING rail since u≈90k (was floor 0.0067) | WATCH — controller worked correctly (floor while ent>target, lifted at u≈70k when ent crossed 1.0, ceiling by u≈90k), but ~25k updates ceiling-pinned with entropy below target; >50k pinned = ALERT per ranges (trips ~u≈140k if unchanged) |
| entropy_proxy | 0.436 (from 2.62) — trajectory 2.67→2.28(u30k)→1.44(u60k)→0.85(u80k)→0.44(u100k) | WATCH — sharpening well past v2's healthy floor (~1.0); "cratering toward 0" is the collapse precursor; E1b over-sharp is the named suspect if eval regresses |
| q1_mean | −0.324 (−0.24 → −0.32) | OK — bounded, not diving (v2 slope ref −1.9) |
| nonfinite / clamp | 0,0,0 / 2.6% (<5%) | OK |
| awac_weight_p99p1 (E1b) | 19.84, recent range 4.95–43.7 | OK — in band but at top edge; saturation not scanned high |
| aux_loss (E2) | 0.19–0.22 falling (0.34 → 0.19) | OK |
| freeze probe | baseline(u=10k)=0.049 → latest(u≈100k)=0.083, ratio 1.7× | WATCH (OK bar 2×) — but RISING, which is the healthy direction; v1 freeze = falling toward 1 |
| retract-vs-slack (HEADLINE) | u=10k: tail 0.812 vs base 0.071 = **+74pp** → u≈100k: tail 0.862 vs base 0.398 = **+46pp** | WATCH — coupling hugely present (v2: +13pp) and tail-bin retraction deepened (meanA0 −2.27→−5.00mm), but the DIFFERENTIAL shrank because base-bin P(retract) rose 0.07→0.40: the policy is turning retract-happy everywhere, not just when stuck. Gate wants differential GROWING — next passes decide |
| aux R² (labels 97/98/102/103) | 0.68 / 0.68 / **−0.008** / 0.60 (early: 0.69/0.69/0.50/0.62) | contact pair ≥0.55 ✓; label 102 velocity head DIED (0.50→−0.01) — WATCH |
| buffer stuck-share | 0.752 (design expectation ~0.10) | WATCH — probe sampled newest 282k rows ≈ mostly online explore data: the online policy spends ~75% of steps in stuck-flagged states. E3's 15% lane is moot when the general lane is already stuck-heavy; q1_mean not diving, so no pessimism ALERT yet |
| eval trajectory | baseline only: q=0.500, speed 0.387 mm/s, reward +1.67 | speed well under the 3 mm/s gate figure (slow-but-successful pretrain policy); eval1 pending |
| stuck harvest (E4) | 111 files (was 8) | OK — growing |
| explore outcomes (recent) | ~47% success (28 succ / 31 max_steps / misc) | OK — above v1's 34% all-time explore peak |
| IPC / RestartCount / OOM | 0 / 0 / false | OK |
| mem / CPU / GPU | 15.7 GiB (50.7%, was 46.7%) / 1202% / GPU 3% between update bursts | OK — watch creep toward 87% |

**Recovery tracker — threshold change adopted this pass** (machine-1-sanctioned:
PUSH_MIN 2.0→1.0, STUCK_STEPS 12→8, because defaults yielded ~0 events on this
policy's gentle ~1.5mm/step pushes; original-threshold table also shown; **all future
trend comparisons use the sensitive thresholds**):
```
[defaults PUSH_MIN=2.0/STUCK_STEPS=12 — undercounts, kept for the record]
window       eps  stuck-eps  events  soft%  hard%  grind%  unrec%  succ%  ret-depth succ/fail (mm)
    1-400     400      17        17    0.0   17.6     0.0    82.4   45.8   14.7 / 22.8
  401-760     360      13        13    0.0    0.0     0.0   100.0   43.6   nan / 14.5

[SENSITIVE thresholds PUSH_MIN=1.0/STUCK_STEPS=8 — the standard from now on]
window       eps  stuck-eps  events  soft%  hard%  grind%  unrec%  succ%  ret-depth succ/fail (mm)
    1-400     400     135       173    6.9   12.1    27.2    53.8   45.8   13.8 / 14.9
  401-760     360     119       125    2.4    4.8     2.4    90.4   43.6   7.4 / 22.6
```
Window 1 includes the 84 eval episodes; window 2 is pure procedural explore (harder,
non-comparable populations — treat window 2 as the trend anchor). unrec% 90 in w2:
when truly wedged, the policy rarely passes the stuck point yet — this is the deficit
v3a exists to fix; soft% must rise from here. 2 windows < the 3-window ALERT rule.

**VERDICT: OK, with a coherent WATCH cluster** — entropy sharpening past v2's floor +
alpha ceiling-pinned + base-bin retract probability tripling + slow eval speed all
describe one phenomenon: E1b is making the policy commit hard, currently toward
caution/retraction. Not an ALERT by any single threshold, and explore success (~47%)
is strong. eval1 (~2h) adjudicates: if Quality holds ≥~50% (its own baseline) the
sharpening is benign; regression with weights concentrated names E1b (raise τ toward
peel-off per handoff §6 — but that decision is machine 1's / the human's, not this
monitor's).

---

## eval1 — 2026-07-17 00:45 EDT (explore 275,506; ~120k online updates)

**Quality 0.500 (49/98), Reward +2.25.** Formal gate (≥ v2's 30.6%) **PASSED** with
+19.4pp headroom. Against its own 50.0% baseline: Quality FLAT, reward +0.58 improved
— online learning so far solves the same seed count but earns more reward per episode
(more progress on failures and/or cleaner successes). The pass-2 WATCH cluster
(sharpening/retract-drift) did NOT translate into eval regression — benign so far.

Post-eval checks all clean: checkpoint275506 banked; incremental buffer save wrote
chunk 282310→557816 (the RL_IMPROV_16 save path working); NO trainer restart in the
post-eval window (the v1/v2 race point); CSV resumed at ~11 updates/s immediately;
next cycle launched with a 17,594-update budget. best_checkpoint still the baseline
snapshot (quality-tie doesn't displace it).

Reference curve so far: v2 6.1 → 30.6 → 49.0 | v3a-run2 50.0 → 50.0 → eval2 (~510k,
gate: climbing; v2 hit 49.0 there). The interesting question at eval2: does online
learning ADD anything, or is the Tier-A pretrain already at this policy's ceiling?

---

## Pass 3 — 2026-07-17 08:53 EDT (post-eval2; explore 509,122 / u≈248,767)

**eval2: Quality 0.500 (49/98), Reward +2.28. Flat for the THIRD consecutive eval**
(50.0 → 50.0 → 50.0; reward +1.67 → +2.25 → +2.28 also plateaued). Formal gates still
pass (≥ v2's curve at every point — v2 eval2 was 49.0). Post-eval clean: checkpoint
banked, no trainer restarts (count still 0 all run), CSV advancing (~11 upd/s), IPC 0.
(Scheduled 02:23/04:23/06:23/08:23 passes did not fire — session cron doesn't dispatch
while the session is parked; Monitor events do wake it, so passes now ride on eval
notifications. Cron f023bf28 left in place for whenever the session is active.)

**VERDICT: ALERT (two machine-1 thresholds tripped) — run alive and stable, but the
experiment's headline mechanism has failed and the run has plateaued at its pretrain
ceiling.**

ALERT 1 — **distillation-fade recurring** (the exact v2 failure v3a was built to fix).
Soft-recovery rate across 400-ep windows: **6.9 → 2.2 → 0.0 → 2.5 → 0.0%**, with
unrecovered 54 → 90 → 96 → 85 → 97% and eval flat — machine-1's ALERT template
(soft% fading 3+ windows + eval stagnant) is met. E1b+E3 did NOT preserve
micro-recovery.
```
window       eps  stuck-eps  events  soft%  hard%  grind%  unrec%  succ%  ret-depth succ/fail (mm)
    1-400     400     135       173    6.9   12.1    27.2    53.8   45.8   13.8 / 14.9
  401-800     400     131       137    2.2    5.1     2.2    90.5   45.0   7.4 / 23.4
  801-1200    400      75        76    0.0    1.3     2.6    96.1   43.5   2.5 / 18.0
 1201-1600    400      37        40    2.5    2.5    10.0    85.0   45.8   0.0 / 20.5
 1601-1947    347      61        62    0.0    1.6     1.6    96.8   45.2   nan / 18.7
```

ALERT 2 — **entropy collapsed to the floor; alpha pinned at ceiling ~159k updates**
(>50k = ALERT). entropy_proxy = **−2.24** (per-dim log_std ≈ −2.0 = the log_std_min
rail: exploration noise is at its minimum allowed, σ≈0.135 pre-tanh). alpha maxed
(0.1003) and impotent against it. clamp_fraction 11.2% (>5% threshold). This is NOT
the v1 freeze — the action MEAN is 14.2× its pretrain baseline (freeze probe VERDICT
OK; eval speed 0.39 → 4.9 → 7.3 mm/s, now above the 3mm/s gate) — it is maximum-
determinism sharpening: the noise that used to discover recovery behavior is gone,
which is precisely WHY the soft-recovery fade recurred (v2 investigation's mechanism,
accelerated by E1b's sharper weights).

Headline retract-vs-slack probe — became partially uninterpretable across passes: the
probe samples its evaluation states from the NEWEST buffer rows, and those are now 97%
stuck-pool states (buffer stuck-share 0.752 → **0.970**), so pass-to-pass numbers
compare different state distributions (the u=10000 'early' row changed too — same
checkpoint, different states). Within THIS pass (same states, both policies): latest
policy on deep-stuck states shows tail-bin P(ret)=0.161/meanA0=+15.8mm vs base-bin
P(ret)=0.857/meanA0=−11.3mm — the desired coupling is NOT present on-distribution
(if anything inverted). Either way the gate's 'positive and GROWING' condition is
unambiguously NOT met. **Handoff §5's own decision row for this outcome: 'flat/
decaying = E1+E3 insufficient → escalate to v3b (E4 restore curriculum).'**

Also green/notable: E1b p99p1 11.7 (in band), aux_loss 0.13-0.18 (falling), aux R²
now strong on ALL FOUR heads (0.92/0.82/0.85/0.50 — velocity head 102 recovered),
q1_mean −0.82 (drifting down from −0.32, not yet v2's −1.9 slope — E3 watch), explore
success stable ~45%, stuck harvest 177 files, nonfinite 0.

**WATCH — memory 72.5%** (15.7 → 22.5 GiB in ~10h, buffer 792k/2M transitions and
growing ~29k/cycle): on this slope the ~87% OOM point (machine-1's kill line) arrives
in roughly 10-15h, BEFORE the buffer cap. If the run is to continue past eval3/4,
machine 1 should decide whether to accept the risk or stop earlier.

---

## Pass 4 (compact) — 2026-07-17 17:36 EDT (post-eval3; explore 756,931 / u≈370,151)

**eval3: Quality 0.5102 (50/98), Reward +2.34 — first Quality movement since
baseline (+1 seed), and NO eval3 regression** (v2's eval3: 49.0 → 30.6 crash; v3a:
50.0 → 51.0). The v2 fix package + raised guards held through the historical race
point: trainer restarts still 0 all run, checkpoint756931 banked, CSV advancing
~10 upd/s, IPC 0. Curve: v2 6.1→30.6→49.0→30.6 | v3a 50.0→50.0→50.0→51.0.

Gates: p99p1 12.9 (in band), aux_loss 0.146 (healthy), q1_mean −1.11 (drifting down,
approaching v2's −1.9 reference — E3 pessimism WATCH), entropy −4.91 (still
deepening), clamp 21.1% (rising), alpha ceiling-pinned (~280k updates now). Stuck
harvest 444 files. Buffer 1.04M/2M transitions.

**MEMORY DEADLINE: 82.3%** (72.5% → 82.3% in 8.7h ≈ 1.1%/h). The ~87% OOM line is
**~4-5h away** — BEFORE eval4 (~8-9h). If nobody intervenes the container will
likely OOM-kill (exit 137) mid-cycle; losses are bounded to the current cycle —
checkpoint756931 + the eval3 incremental buffer chunks are already on disk. Under
the standing read-only instruction this monitor will NOT stop the container; flagged
for a deliberate decision. Recommended: stop cleanly at/before ~86% (or after the
next eval-cycle boundary), then proceed to stuck-pool screening + v3b.

---

## Pass 5 (compact) — 2026-07-18 02:16 EDT (post-eval4; explore 1,000,378 / u≈494,368)

**eval4: Quality 0.5102 (50/98), Reward +2.33 — flat. Plateau CONFIRMED over 1.0M
explore steps:** 50.0 → 50.0 → 50.0 → 51.0 → 51.0 (rewards +1.67/+2.25/+2.28/+2.34/
+2.33). Trainer restarts 0 all run; checkpoint1000378 + all 5 buffer chunks (1.28M
transitions) banked; best_checkpoint = eval3's 51.0% policy. Stuck pool 550 files.
Sharpening continues unabated: ent −7.27, clamp 31%, q1 −1.31, alpha still ceiling.

**MEMORY: 86.9% — AT machine-1's OOM line** (growth decelerated 1.1 → ~0.5%/h,
which is why the pass-4 deadline estimate was outlived; buffer still growing toward
its 2M cap). Everything of value is already on disk; an OOM now would cost only the
in-flight cycle. **Standing recommendation upgraded: the run is scientifically
complete (5 evals, hard plateau) and at the OOM line — stop it deliberately, screen
the stuck pool (550 + 72 archived), proceed to v3b.** Read-only instruction still
observed: no action taken by this monitor.

---

## Pass 6 (compact) — 2026-07-18 11:21 EDT (post-eval5; explore 1,260,060 / u≈625,551)

**eval5: Quality 0.5204 (51/98), Reward +2.40.** Crawl continues: 49→49→49→50→50→51
seeds over 1.26M steps (+2 total from baseline). Trainer restarts still 0; eval5
checkpoint + buffer chunk (→1,542,370) banked; stuck pool 686 files.

**MEMORY CORRECTION: 78.7%, DOWN from 86.9%** — the pass-4/5 linear-growth OOM model
was wrong; container memory fluctuates with allocator/page-cache reclaim and is not
monotone. The structural stabilizer is the replay buffer's 2M ring cap (now 1.54M,
~+260k/eval): ~2 evals from now growth per transition stops. OOM urgency downgraded
WATCH — the run can likely continue safely, so the stop decision is purely scientific,
not operational.

Gates: q1_mean −1.64 (approaching v2's −1.9 pessimism reference — E3 WATCH
strengthening), p99p1 23.4 (above band top), ent −7.32/clamp 30% (unchanged
sharpening), aux healthy. Science unchanged: ~+1 seed per ~250k explore steps is a
glacial crawl, not a curve — v3b remains the sanctioned escalation; every additional
eval mostly buys ~130 stuck-pool files.

---

## Pass 7 (compact) — 2026-07-18 20:15 EDT (post-eval6; explore 1,505,005 / u≈747,768)

**eval6: Quality 0.5204 (51/98), Reward +2.41 — flat, matches eval5.** Curve:
49/49/49/50/50/51/51 solved across 7 evals, 1.5M explore steps. q1_mean −1.82 has
essentially reached v2's −1.9 pessimism reference; entropy −9.0, clamp 37%. Buffer
1.79M/2M (cap ~1 eval away → growth halts there). Mem 86.7% (fluctuating band 78-87%,
not monotone — no OOM, restarts still 0). Stuck pool 806. Nothing new; plateau + slow
sharpening continue exactly as the forensic characterized.

---

## Pass 8 (compact) — 2026-07-19 (post-eval8; explore 1,759,714) — PREDICTED REGRESSION BEGINS

**eval8: Quality 0.500 (49/98) — DROPPED from eval7's 0.520.** Curve: 50/50/50/51/51/52/52/**50**.
The forensic's demo-eviction forecast has triggered: `buffer_len=2,000,000` (ring FULL) and
the newest chunk `chunk_1787315_2042024` has wrapped past the 2M cap — now overwriting slots
0-42024, i.e. the head of the 282,310 seed transitions. First eval after the wrap regressed by
the 2 flapper seeds (back to the base 49). Forecast was explore ~1.72M; actual wrap ~1.79M.
Mem 79%, restarts 0, no OOM. **This is the run actively degrading as its seed/pretrain data is
overwritten by 45%-success self-play — the mechanism the forensic predicted. Strengthens
STOP-NOW to a firm recommendation: continuing now trades the pretrain baseline away.**

---

## Pass 9 (compact) — 2026-07-19 (post-eval9; explore 2,004,434)

**eval9: Quality 0.500 (49/98) — holds the eval8 regression.** Curve: …0.52/0.52/**0.50/0.50**.
Explore now past 2.0M, so the ring (buffer_len=2,000,000, cap) has fully cycled — the 282k
seed transitions are entirely evicted and the buffer is 100% self-play (~45% success, near-zero
recoveries). Quality has settled back to the base 49 seeds and is no longer improving; the two
flapper gains are gone for good. Mem 82%, restarts 0, no OOM. This is the post-eviction steady
state the forensic predicted: nothing more to learn here, seed/demo signal gone. **STOP remains
the firm recommendation.** (Also: a fresh direct check of seed.npz reconfirmed zero gw-led and
zero deep (>170mm) content — max achieved tip depth 164mm, all 27 successes ≤138mm, actions
i.i.d. random — so nothing is lost by stopping; the deep skill must be generated regardless.)

## Pass 10 (one-line) — 2026-07-20 (eval10, explore 2.25M): Quality 0.500 (49/98), flat 3rd straight post-eviction eval. Restarts 0, no OOM, mem 84%. No change; STOP recommendation stands. Buffer 100% self-play (seed long evicted). Nothing further to learn.

## Pass 11 (one-line) — 2026-07-20 (eval11, explore 2.53M): Quality 0.500 (49/98), flat 4th straight. Restarts 0, no OOM, mem 81%. No change; STOP stands.

## Pass 12 (one-line) — 2026-07-20 (eval12, explore 2.76M): Quality 0.510 (50/98), reward 2.37 — one-seed flicker within the 49-51 noise band, not a real gain. Restarts 0, no OOM, mem 81%. STOP stands.

## Pass 13 (one-line) — 2026-07-21 (eval13, explore 3.00M): Quality 0.500 (49/98) — eval12 flicker dropped back out (confirmed noise). Restarts 0, no OOM, mem 82%. 6 evals now flat at ~0.50 post-eviction; STOP stands.

## Pass 14 (one-line) — 2026-07-21 (eval14, explore 3.27M): Quality 0.5476 = **46/84, NOT /98** — workers 0+1 hit the benign cycle-boundary timeout at eval close and their ~14 episodes dropped from the tally. Not comparable to the series (full-set equivalent ~53/98 if lost seeds solved at base rate — possible mild uptick, unmeasurable). Container healthy (mem 80%, no OOM). STOP stands.

**Recommendation to machine 1 / human (report-only; no action taken):** the Tier-A
scientific result is effectively in — the bundle transformed PRETRAIN (6.1% → 36.7/
50.0% baselines, and eval efficiency keeps improving: speed 7.3mm/s, reward +2.28)
but did NOT fix the micro-recovery fade, and online Quality is ceiling-locked at
49/98. Per the handoff pipeline: screen the stuck pool (launch_screen_stuck.sh —
run-2 pool at 177 files; note the eval-anatomy contamination exclusion, and
`saved/rcca_v3_stuck_run1/` holds 72 more from run 1) and escalate to v3b
(--rl_start_mode sofa_restore + the NOT-YET-IMPLEMENTED --restore_prob Bernoulli gate
flagged in handoff §7.2). E7 (retract-on-escape reward) remains the human-approval-
gated fallback if v3b also fails to revive micro-recovery.
