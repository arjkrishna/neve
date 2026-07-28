# Stuck & Recovery Analysis — v1b vs v1bp (EXPLORE ONLY)

2026-07-28. Source: worker-log complete ledger (`EPISODE_START` / `STEP` / `EPISODE_OUTCOME`),
**v1b 14,691 explore episodes / 4.06M steps**, **v1bp 8,040 explore episodes / 2.55M steps**.

## 0. Method and why each definition is what it is

**Explore-only isolation is exact.** `runner.eval()` resets with explicit seeds, so eval
`EPISODE_START` lines carry `seed=` and explore resets do not. Filtering on that field yields
v1b 1,470 eval episodes (= 15 evals × 98) and v1bp 980 (= 10 × 98) — exact. *(The obvious
alternative, main.log time windows, leaks: it caught only 567 of v1bp's 980, contaminating
explore with ~5% eval. Any earlier windowed numbers in this program are approximate.)*
`episode_summary.jsonl` is unusable for this — it logs only ~36% of explore episodes.

**Training-time axis = cumulative explore steps** (sum of per-episode steps, chronological).
Exact from the logs and comparable across runs, unlike wall-clock or the global step counter.

**A stall = pushing but not progressing.** Opens when `proj_s < running_max + 0.3 mm` while
`|cmd_action[gw_trans]| > 2 mm/s` for 12 consecutive steps (counter decays −2 on a
non-stalled step, so one noisy sign-flip inside a stuck push doesn't erase it). Requiring
*push* matters: a wire deliberately retracting or rotating is manoeuvring, not stuck.
Closes as a recovery when `proj_s > proj_s_at_onset + 1 mm` — i.e. it got past the frontier
where it jammed.

**Recovery kind = how much wire actually came out** before passing, measured as
`peak_insertion − deepest_insertion` during the stall (executed length, not commanded — a
retract command that moves nothing must not score as a recovery):

| kind | retraction | reading |
|---|---|---|
| **grind** | < 1 mm | forced through without backing off |
| **soft** | 1–8 mm | ease slack → re-advance (the behaviour we set out to induce) |
| **hard** | > 8 mm | escaped only after a large pullback |
| **unrecovered** | — | never passed; the stall was terminal for that episode |

**The three successes are at different units** — conflating them is what made
"success ≈ recovery" look true earlier:
1. **Episode success** (per episode) — reached target.
2. **Recovery success** (per *event*) — escaped a stall, any kind.
3. **Recovery→episode success** (per episode) — stalled, escaped, *and* reached target.

---

## 1. v1b — stuck & recovery vs training time

| bin | explore steps (k) | eps | ep succ% | stalled-ep% | events | evt/stalled-ep | recovery succ% | soft% | hard% | grind% | unrec% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0–603 | 1836 | 52% | 64% | 2441 | 2.1 | 65% | 27% | 11% | 27% | 35% |
| 2 | 604–1148 | 1836 | 56% | 56% | 1926 | 1.9 | 58% | 24% | 14% | 20% | 42% |
| 3 | 1149–1647 | 1836 | 60% | 51% | 1568 | 1.7 | 53% | 22% | 12% | 18% | 47% |
| 4 | 1647–2157 | 1836 | 58% | 55% | 1797 | 1.8 | 58% | 24% | 12% | 21% | 42% |
| 5 | 2157–2629 | 1836 | 61% | 48% | 1323 | 1.5 | 46% | 21% | 8% | 17% | 54% |
| 6 | 2629–3108 | 1836 | 60% | 46% | 1501 | 1.8 | 52% | 22% | 11% | 18% | 48% |
| 7 | 3108–3559 | 1836 | 62% | 42% | 1142 | 1.5 | 39% | 16% | 10% | 13% | 61% |
| 8 | 3559–4053 | 1836 | 59% | 47% | 1308 | 1.5 | 41% | 19% | 6% | 17% | 59% |

**Episode success rises 52→62% while recovery success FALLS 65→41% and unrecovered rises
35→59%.** Stalled-episode share drops 64→47%.

## 2. v1b — the three successes

| bin | eps | ep succ% | clean eps (succ%) | recovered eps (succ%) | unrec eps (succ%) | % of successes via recovery |
|---|---|---|---|---|---|---|
| 1 | 1836 | 52% | 667 (100%) | 305 (91%) | 864 (1%) | 29% |
| 2 | 1836 | 56% | 811 (100%) | 220 (97%) | 805 (0%) | 21% |
| 3 | 1836 | 60% | 901 (100%) | 192 (99%) | 743 (1%) | 17% |
| 4 | 1836 | 58% | 834 (100%) | 248 (94%) | 754 (1%) | 22% |
| 5 | 1836 | 61% | 952 (100%) | 172 (98%) | 712 (1%) | 15% |
| 6 | 1836 | 60% | 986 (100%) | 127 (93%) | 723 (1%) | 11% |
| 7 | 1836 | 62% | 1065 (100%) | 76 (97%) | 695 (1%) | 6% |
| 8 | 1836 | 59% | 980 (100%) | 90 (99%) | 766 (1%) | 8% |

**Recovery's contribution to success collapses 29% → 6–8%.** Clean episodes rise 667→1065.

⚠ **Caveat — this partition is near-definitional.** "Never stalled ⇒ 100% success" and
"unrecovered ⇒ ~1% success" hold in all 16 bins across both runs, because an episode that
pushes for 600 steps without ever jamming arrives, and one that ends jammed does not.
So T2's *rates* are close to identities; the informative quantities are the **counts** —
how many episodes fall in each channel, and the share of successes flowing through recovery.

## 3. v1b — does recovery TYPE predict episode success?

| recovery profile | episodes | episode succ% | median steps | mean retract mm |
|---|---|---|---|---|
| soft (any) | 820 | 95% | 187 | 4.9 |
| hard (no soft) | 248 | 92% | 227 | 13.7 |
| grind only | 362 | **99%** | **107** | 0.1 |
| unrecovered (any) | 6065 | 1% | 600 | 13.5 |
| never stalled | 7196 | 100% | 41 | 0.0 |

## 4. v1b — by depth

| band | eps | ep succ% | stalled-ep% | events | recovery succ% | soft% | hard% | grind% | unrec% |
|---|---|---|---|---|---|---|---|---|---|
| CCA | 5368 | 99% | 5% | 496 | 94% | 43% | 15% | 36% | 6% |
| ICA-mid | 4692 | 51% | 65% | 5063 | 54% | 23% | 14% | 18% | 46% |
| siphon | 4631 | 19% | 90% | 7454 | 50% | 21% | 9% | 20% | 50% |

---

## 5. v1bp — stuck & recovery vs training time

| bin | explore steps (k) | eps | ep succ% | stalled-ep% | events | evt/stalled-ep | recovery succ% | soft% | hard% | grind% | unrec% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0–374 | 1005 | 46% | 68% | 1623 | 2.4 | 68% | 28% | 14% | 26% | 32% |
| 2 | 375–671 | 1005 | 54% | 54% | 825 | 1.5 | 44% | 22% | 11% | 11% | 56% |
| 3 | 671–1060 | 1005 | 40% | 68% | 1947 | 2.9 | 72% | 32% | 22% | 18% | 28% |
| 4 | 1061–1370 | 1005 | 55% | 55% | 1501 | 2.7 | 73% | 34% | 18% | 21% | 27% |
| 5 | 1370–1686 | 1005 | 53% | 55% | 1603 | 2.9 | 75% | 32% | 21% | 22% | 25% |
| 6 | 1686–1979 | 1005 | 57% | 53% | 1181 | 2.2 | 64% | 28% | 17% | 19% | 36% |
| 7 | 1979–2257 | 1005 | 61% | 57% | 1170 | 2.0 | 66% | 31% | 15% | 21% | 34% |
| 8 | 2257–2551 | 1005 | 56% | 54% | 1149 | 2.1 | 63% | 25% | 15% | 22% | 37% |

**Recovery does NOT decay here** — it holds 63–75% after bin 2, versus v1b's decay to 39–41%.

## 6. v1bp — the three successes

| bin | eps | ep succ% | clean eps (succ%) | recovered eps (succ%) | unrec eps (succ%) | % of successes via recovery |
|---|---|---|---|---|---|---|
| 1 | 1005 | 46% | 326 (100%) | 158 (77%) | 521 (2%) | 27% |
| 2 | 1005 | 54% | 467 (100%) | 73 (96%) | 465 (0%) | 13% |
| 3 | 1005 | 40% | 325 (100%) | 140 (56%) | 540 (0%) | 20% |
| 4 | 1005 | 55% | 449 (100%) | 152 (68%) | 404 (0%) | 19% |
| 5 | 1005 | 53% | 456 (100%) | 145 (54%) | 404 (0%) | 15% |
| 6 | 1005 | 57% | 469 (100%) | 109 (94%) | 427 (1%) | 18% |
| 7 | 1005 | 61% | 432 (100%) | 180 (95%) | 393 (3%) | 28% |
| 8 | 1005 | 56% | 464 (100%) | 111 (85%) | 430 (1%) | 17% |

## 7. v1bp — recovery type, and by depth

| recovery profile | episodes | episode succ% | median steps | mean retract mm |
|---|---|---|---|---|
| soft (any) | 714 | 76% | 296 | 6.2 |
| hard (no soft) | 195 | 65% | 370 | 15.2 |
| grind only | 159 | **95%** | **121** | 0.1 |
| unrecovered (any) | 3584 | 1% | 600 | 11.0 |
| never stalled | 3388 | 100% | 43 | 0.0 |

| band | eps | ep succ% | stalled-ep% | events | recovery succ% | soft% | hard% | grind% | unrec% |
|---|---|---|---|---|---|---|---|---|---|
| CCA | 2804 | 98% | 7% | 457 | 86% | 36% | 18% | 32% | 14% |
| ICA-mid | 2690 | 43% | 74% | 4217 | 65% | 30% | 17% | 19% | 35% |
| siphon | 2546 | 14% | 97% | 6325 | 68% | 29% | 18% | 21% | 32% |

---

## 8. HEAD-TO-HEAD (whole run)

| metric | v1b | v1bp | reading |
|---|---|---|---|
| episodes / explore steps | 14,691 / 4.06M | 8,040 / 2.55M | |
| **episode succ%** | **58.6** | **52.8** | v1b ahead (57.4% at matched 2.55M budget) |
| stalled-ep% | 51.0 | 57.9 | v1bp stalls MORE |
| events | 13,013 | 10,999 | |
| **recovery succ%** | **53.4** | **67.4** | **v1bp escapes far better** |
| soft% | 22.6 | 29.5 | v1bp softer |
| hard% | 10.9 | 17.3 | |
| grind% | 19.8 | 20.6 | equal |
| **unrec%** | **46.6** | **32.6** | **v1bp leaves far fewer terminal stalls** |
| clean-ep succ% | 100.0 | 100.0 | (near-definitional) |
| **recovered-ep succ%** | **95.2** | **77.0** | **v1bp's recoveries convert WORSE** |
| % successes via recovery | 15.8 | 19.4 | |
| **mean cath-lead frac** | **0.34** | **0.04** | the shove is gone in v1bp |

## 9. Threshold sensitivity — do the conclusions survive?

v1b:

| cfg | events | stalled-ep% | recovery succ% | soft% | hard% | grind% | unrec% |
|---|---|---|---|---|---|---|---|
| sens | 17,386 | 55% | 65% | 29% | 10% | 26% | 35% |
| canon | 13,013 | 51% | 53% | 23% | 11% | 20% | 47% |
| strict | 10,394 | 48% | 42% | 18% | 11% | 14% | 58% |

v1bp:

| cfg | events | stalled-ep% | recovery succ% | soft% | hard% | grind% | unrec% |
|---|---|---|---|---|---|---|---|
| sens | 14,271 | 61% | 75% | 36% | 16% | 23% | 25% |
| canon | 10,999 | 58% | 67% | 30% | 17% | 21% | 33% |
| strict | 8,701 | 55% | 60% | 26% | 18% | 15% | 40% |

Absolute rates move a lot with thresholds (recovery success 42–65% for v1b alone), **but every
ordering is preserved**: v1bp > v1b on recovery success and soft share, and v1bp < v1b on
unrecovered, under all three detectors. Report deltas, never absolutes.

## 10. Correlations across chronological bins

| run | ep-success vs recovery-success | ep-success vs soft-share |
|---|---|---|
| v1b | **r = −0.82** | r = −0.76 |
| v1bp | r = −0.21 | r = −0.07 |

---

## 11. FINDINGS

**F1 — Improvement comes from stall AVOIDANCE, not stall ESCAPE.** In v1b, episode success
rises 52→62% while recovery success falls 65→41%, unrecovered climbs 35→59%, and recovery's
share of successes collapses 29→6%. The correlation is strongly **negative (r = −0.82)**. The
policy is not learning to get unstuck; it is learning not to get stuck — and the recovery
skill *decays* as it does. This is the interference mechanism predicted in the design doc,
now measured: as path-following improves, stalls leave the data distribution and the recovery
skill loses its gradient. **My earlier read that "success% ≈ recovery%" was wrong** — it was
an artifact of mixing per-event and per-episode units.

**F2 — The reward pair fixed recovery, mechanically and unambiguously.** v1bp escapes 67.4%
of stalls vs v1b's 53.4%, leaves 32.6% unrecovered vs 46.6%, raises soft share 22.6→29.5%,
and — the design's explicit target — cuts catheter-lead from 0.34 to **0.04**. Crucially the
decay is gone: v1bp's recovery holds 63–75% late (r = −0.21) where v1b decayed to ~40%
(r = −0.82). **The pair did exactly what it was designed to do.**

**F3 — …and it still didn't help, because recovery doesn't pay.** v1bp's episode success is
*lower* (52.8 vs 58.6; 57.4 at matched budget). Its recovered episodes convert at only 77%
vs v1b's 95%, and cost far more time (soft-recovery median 296 steps vs 187; hard 370 vs 227,
against a 600 cap). v1bp also stalls more often (57.9% vs 51.0%) — the cautious, non-shoving
gait jams more readily. Net: more stalls, better escapes, worse conversion, slightly worse
outcomes.

**F4 — GRIND beats SOFT on episode success, in both runs.** Grind-only episodes succeed
99% (v1b) / 95% (v1bp) in ~110 steps; soft-recovery episodes 95% / 76% in 187–296 steps;
hard 92% / 65% in 227–370. **This inverts the premise behind the crunchpass lane and the
"we want soft recoveries" framing.** The likely reason is a confound: recovery *type* is
mostly a proxy for stall *severity*, not for policy skill. A stall you can grind through was
mild; one that forces an 8 mm withdrawal was severe, and severe stalls sit in harder (deeper)
places. Conditioning on recovery type therefore conditions on difficulty. **Soft recovery
should not be treated as a training target until this confound is broken** — e.g. by
comparing soft vs grind *within* matched stall severity/depth strata.

**F5 — Depth dominates everything.** Stalling is near-universal deep: 90% (v1b) / 97% (v1bp)
of siphon episodes stall, versus 5–7% at CCA. And at the siphon, escaping doesn't save you —
recovery success is 50–68% but episode success is only 14–19%. Escaping one stall at the
siphon simply leads to the next. This is the same wall the real-patient evaluation found
(0/30), seen from the inside.

**F6 — Detector-robust.** Absolute rates are threshold-sensitive (±20 pp), but every v1b-vs-
v1bp ordering holds under all three configs. Conclusions are stated as deltas for this reason.

## 12. Implications

1. **The crunchpass lane's premise needs revisiting before launch.** It resamples
   success-conditioned crunch passages on the theory that soft recovery is the skill to
   amplify. F4 says grind-through is what actually converts, and F1 says the winning channel
   is avoidance. Amplifying recovery may be optimising a behaviour that doesn't pay.
2. **If recovery is still wanted, it must be made to pay.** Recovered episodes cost 2–7× the
   steps of clean ones. Under a 600-step cap with a sparse target reward, a slow recovery is
   nearly worthless. Either the budget must accommodate recovery, or recovery must get
   cheaper — otherwise the optimiser is correct to prefer avoidance.
3. **The reward pair is validated as a mechanism and rejected as a win.** It is the cleanest
   demonstration we have that you can move the intended internal behaviour decisively
   (cath-lead 0.34→0.04, unrecovered −14 pp) and gain nothing on the outcome.
4. **The siphon wall is not a recovery problem.** Near-100% stall rates with ~65% escape but
   ~15% success means the limiting factor is the sheer density of stalls, not the ability to
   escape any one of them.
