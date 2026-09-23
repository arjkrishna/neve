# Frontier replay campaign — report (2026-09-20 → 21)

Question asked: which validation seeds actually measure performance at the frontier of the carotid v3
models, what the frontier is made of (recovery, coil/buckle avoidance, catheter-led advance), and how
carotid v3 compares with TopBrain v2 on the real patient (coils) and with the procedural policy
(catheter-led advance). Method: replay the best checkpoints many times on hard seeds only, record every
episode with both device polylines, film them, and analyse per-seed success probabilities over the
repetitions (ANIM_PIPELINE tier 2). 27 replay runs, 2 217 filmed episodes.

## In one paragraph

Only 10 of the 47 "hard" carotid validation seeds sit at the frontier of the strong checkpoints; 14 are
solved by every checkpoint including a 40 % one, and 23 separate 75 % from 93 % but not 93 % from 97 %.
On those ten seeds half of the best checkpoint's successes are recoveries (unload-and-repush inside a
stall, with a visible tip retreat in two thirds of them), against one in seven on the easy seeds; every
strong checkpoint shows the same split. The best checkpoint's five failures are doorstep stalls at
90–94 % of the path that repeated withdrawals never convert. Loop (coil/buckle) rate falls from 23 % of
episodes at 0.8 M steps to 2 % from 1.55 M on, and the best checkpoint escaped every loop it formed;
the actor-masked ablation never reaches that (6–15 %, stuck in half) and its decayed last checkpoint
regresses to mid-path stalls. On the real patient the carotid best solves 93 % of the 85 seeds
TopBrain v2 finds hard, looping once in 268 episodes, while TopBrain v2 solves 57 % and loops in 54 %
of its episodes and 69 % of its failures. The procedural policy is a catheter-led sprinter with no
recovery: fast when the catheter carries the wire through, stalled early otherwise (43 % on the carotid
frontier seeds, 47 % on the host, against 97 % and 93 %).

## 1. Protocol

- **Seeds.** Carotid: 47 of the 98 validation seeds chosen from the in-run validation history (failed
  in ≥ 2 of 14 high blocks, in exactly 1, or never failed but recovery-heavy); the 51 seeds that were
  clean and fast in ≥ 9 of 14 blocks were excluded up front. Host: the 85 of 98 seeds TopBrain v2 fails
  in ≥ 50 % or sometimes across its 12 recorded host sessions, plus one run on all 98. Procedural: 98.
- **Repetitions.** Replays are not episode-identical to the in-run evals (the device-twist RNG is
  unseeded), so each checkpoint was replayed 1–4 times on the same seeds and the unit of analysis is the
  per-seed success probability over repetitions. Outcomes flip between repetitions on ~1 of 47 carotid
  seeds for the strong checkpoints; the *way* an episode goes (atlas type / phase) flips on ~40 %.
- **Seed classes** (per family, `strong` = the checkpoints whose frontier is measured):
  *easy* — every checkpoint always solves it; *strength* — all strong checkpoints always solve it but a
  weak one fails it; *frontier* — a strong checkpoint fails it in some repetition; *floor* — no strong
  checkpoint ever solves it. Frontier + strength + floor = the decisive seeds.
- **Labels.** Atlas k-means type (11 types), second-pass phase taxonomy (free-run, cath-led-run,
  idle-creep, light/deep recovery, fight-escaped, doorstep-stall, shove-capped, coiled-stuck,
  stuck-midpath/proximal), polyline loop detector (validated blind on 26 episodes: outcome 26/26, end
  location 20/20, loops 23/26 with the three disagreements being tip J-hooks the rule rightly ignores),
  explicit catheter-lead measurements, insertion-based withdrawal and guidewire-tip retreat.
  "Recovery" = an unload-and-repush inside a stall (insertion withdrawn > 8 mm, then re-advance and
  escape); in ~2/3 of the frontier models' recoveries the tip itself retreats > 8 mm, in the rest the
  withdrawal releases stored slack while the tip stays put.
- **Runs.** Part 1 (carotid, 16 runs), Part 2 (host, 8 runs incl. a third TopBrain v2 replay), Part 3
  (procedural, 4 runs). Manifests and drivers: `monitoring/traj/campaign/`; analysis:
  `monitoring/traj/traj_frontier.py`, pages: `traj_compare_page.py`, films: `traj_render.py`.

## 2. Part 1 — carotid v3 within one run and against the actor-masked ablation

47 hard seeds, 752 episodes.

| checkpoint | reps | success | successes won by recovery | failures with a loop | loop rate (all episodes) | median steps |
|---|---|---|---|---|---|---|
| baseline 0.80 M (`car_early`) | 1 | 74.5 % | 14 % | 92 % | 23.4 % (0 escaped) | 172 |
| baseline 1.34 M divergence (`car_div`) | 1 | 40.4 % | 26 % | 14 % | 8.5 % | 601 |
| baseline 1.55 M (`car_1545`) | 1 | 93.6 % | 23 % | 33 % | 2.1 % | 68 |
| baseline 1.79 M (`car_peak`) | 2 | 93.6 % | 22 % | 33 % | 4.3 % (50 % escaped) | 75 |
| **baseline 2.29 M best (`car_best`)** | 4 | **97.3 %** | 19 % | **0 %** | 2.1 % (all escaped) | 71 |
| ablation 0.76 M co-peak (`abl_copeak`) | 1 | 87.2 % | 24 % | 100 % | 14.9 % | 68 |
| ablation 1.01 M best (`abl_best`) | 4 | 92.6 % | 18 % | 36 % | 5.9 % (55 % escaped) | 71 |
| ablation 2.76 M last (`abl_last`) | 2 | 76.6 % | 17 % | 27 % | 8.5 % | 68 |

Seed-paired (per-seed probabilities, sign test): best vs 0.8 M **12–0** (p = 0.0005); best vs
divergence **28–0**; best vs 1.55 M 3–1 and vs 1.79 M 3–1 (n.s.); best vs ablation best 6–2 (p = 0.29);
ablation best vs ablation last **12–2** (p = 0.013); ablation best vs co-peak 6–4 (n.s.).

### 2.1 Which seeds measure the frontier (strong = 1.55 M, 1.79 M, 2.29 M, ablation 1.01 M)

- **easy (14)**: 18, 44, 50, 63, 73, 80, 89, 97, 110, 120, 122, 138, 155, 171 — solved by every
  checkpoint every time, the 40 % divergence checkpoint included. Their in-run failures were twist noise.
- **strength (23)**: 1, 2, 3, 10, 21, 34, 37, 39, 43, 47, 48, 69, 70, 81, 91, 108, 115, 118, 124, 140,
  148, 161, 162 — always solved by the strong checkpoints, failed by 0.8 M / divergence / decayed ablation.
- **frontier (10)**: **23, 31, 55, 93, 134, 167, 8, 16, 139, 154** — carry the whole 93 → 97 % difference.
- floor: none.

Per-seed success probability on the frontier seeds (best / 1.79 M / 1.55 M / ablation best):
23: 0.75 / 0 / 1 / 1 · 31: 1 / 1 / 1 / 0 · 55: 0.25 / 0.5 / 0 / 0.75 · 93: 1 / 1 / 0 / 1 · 134: 0.75 / 0 / 0 / 0.5 ·
167: 1 / 1 / 1 / 0.25 · 8: 1 / 0.5 / 1 / 1 · 16: 1 / 1 / 1 / 0.5 · 139: 1 / 1 / 1 / 0.75 · 154: 1 / 1 / 1 / 0.75.

### 2.2 What the frontier is made of

**Recovery.** Share of successes won by recovery (light + deep + won fight), by seed class:

| checkpoint | easy seeds | strength seeds | frontier seeds |
|---|---|---|---|
| best 2.29 M | 14 % | 16 % | **51 %** (43 % deep) |
| 1.79 M | 6 % | 33 % | 50 % |
| 1.55 M | 29 % | 17 % | 71 % |
| ablation best | 7 % | 20 % | 62 % |

Seeds 23, 134, 8, 167 and 139 are won only by recovery. The blind check confirmed what these are:
withdrawals of 50–300 mm at the insertion point, the tip retreating 19–35 mm in about two thirds of
them and staying put in the rest while the slack is released.

**Where it is lost.** All five failures of the best checkpoint are doorstep stalls at 90–94 % of the
path (seeds 23, 55 ×3, 134): repeated unload-and-repush, 180–840 mm withdrawn in total, and the last
bend never converts. On the same seeds the 1.79 M checkpoint fails differently: seed 134 twice by
shoving the catheter forward while the wire coils behind it (catheter slack 561 mm), seed 23 twice by
stalling 510 steps at 47 % of the path and escaping too late. On anatomies casem030…mr003L and
casew013…mr013L the planned path curls in its last 10 %, so a doorstep stall is drawn touching the
target marker; only the trace tells it from a success.

**Coil / buckle avoidance.** Loops go from 23 % of episodes at 0.8 M (92 % of its failures) to 2 %
from 1.55 M on; the best checkpoint formed four marginal loops (catheter slack 63–68 mm, 1–3 steps)
and undid every one at once. The masked ablation never gets there: 15 % at its co-peak (every failure a
shove), 6 % at its best (stuck in 5 of 11), 8.5 % at the end. The ablation's 14 failures split 5 loops
(all after a doorstep stall: the catheter is driven 70–650 mm ahead of the stalled wire and the slack
coils, held 140–460 steps) and 9 plain doorstep stalls with 7–19 mm of catheter slack on seeds 16, 31,
167 — seeds the carotid best solves 4/4, seed 167 by deep recovery every time. Seed 55 goes the other
way (ablation 3/4, carotid best 1/4). Page: `analysis_part1/compare_abl_best_vs_car_best.html`.

**The ablation's decay** (2.76 M): 12 of 21 failures are stalls at 37–45 % of the path, a location the
earlier checkpoints pass, with the catheter 5–15 mm ahead of a wire being pulled back.

**A late behaviour that does not help the frontier.** A quarter of the best checkpoint's easy-seed
successes are catheter-led sprints (catheter 20–40 mm ahead, done in 22–37 steps) that no earlier
checkpoint shows; none of its frontier-seed successes is one.

Pages: `analysis_part1/SUMMARY.md`, `decisive.html` (all 528 films of the 33 decisive seeds),
`anim_carotid/by_seed/s0134.html`, `s0023.html`, `s0055.html`.

## 3. Part 2 — real patient: carotid v3 vs TopBrain v2 (coils), with TopBrain v3 and the ablation

85 hard host seeds (plus all 98 once for the carotid best: 94/98 = 95.9 %, reproducing the ~95 %
seen on the other machine). 693 episodes.

| model | reps | success (85 hard) | successes by recovery or won fight | loop rate | failures with a loop | failures ending before ⅔ of the path |
|---|---|---|---|---|---|---|
| **carotid v3 best 2.29 M** | 3 | **92.9 %** | 77 % | 0.4 % (escaped) | 0 % | 0 % |
| masked ablation best 1.01 M | 1 | 95.3 % | 21 % | 10.6 % | 100 % | 0 % |
| TopBrain v3 257 k (reference) | 1 | 92.9 % | 25 % | 9.4 % | 100 % | 17 % |
| TopBrain v2 best | 3 | 56.5 % | 23 % | **54 %** | **69 %** | 31 % |

Seed-paired: carotid best vs TopBrain v2 **59–3** (p < 10⁻⁴); ablation vs TopBrain v2 62–3; TopBrain v3
vs TopBrain v2 58–5. Among the strong models: ablation vs carotid best 14–4 (p = 0.03, one replay),
TopBrain v3 vs carotid best 15–6 (p = 0.08), ablation vs TopBrain v3 5–3 (n.s.). Classes (strong =
carotid best, TopBrain v3, ablation): easy 18, strength 57, frontier 23, floor none.

- **Coils.** TopBrain v2 formed a loop in 138 of 255 episodes, typically by step 64, and its failures
  are 51 shove-capped, 22 coiled-stuck, 33 stuck mid-path of 111. It escaped 44 % of its loops. The
  carotid best looped once in 268 host episodes and escaped it; its 18 failures are 7 doorstep stalls,
  9 mid-path stalls past ⅔ of the path, 1 shove, none with a loop. The ablation and TopBrain v3 loop in
  ~10 % of episodes and every one of their few failures is a loop.
- **How the carotid model wins the host.** 4 % free runs, 38 % recoveries (tip retreat visible in 96 %),
  39 % won fights; median 186 steps against 73 for the ablation and TopBrain v3, which solve the same
  seeds mostly as free runs. The carotid best is the most coil-proof model on the patient and the least
  fluent; the ablation's single host replay is the best score of the campaign and needs a second replay.
- **Caveat.** TopBrain v2's three replays scored 41, 60 and 43 of 85 on the same seeds — far outside
  twist noise (the carotid best moved 77 → 79). The slow, coil-heavy replays are also the slow-stepping
  ones (1.0–1.3 s per simulation step vs 0.77 s), consistent with more coiling but not explaining what
  set a run on that path. Its in-run host history (72–75 % on all 98 ≈ 70 % on this subset) matches the
  60/85 replay. All three are pooled here; the comparison holds under any of them.

Pages: `analysis_part2/SUMMARY.md`, `decisive.html`, `compare_tbv2_best_host_vs_car_best_host.html`,
`anim_host/by_seed/`.

## 4. Part 3 — catheter-led advance: procedural policy vs carotid v3

Procedural v3c (2026-07-22, AWAC, ck252305, its best at 62 % on its own anatomies; different network:
no residual heuristic, no heuristic-action obs, no privileged actor, a 4-unit auxiliary head).

| anatomy set | procedural | carotid v3 best | paired |
|---|---|---|---|
| carotid frontier 47 | 42.6 % — 100 % of successes catheter-led (median lead 55 mm at the end, catheter ahead on 78 % of steps), median 50 steps, 5 % recovery; failures stall at a median 20 % of the path (6 proximal, 3 mid, 11 with no detected stall and no progress), 4 % loops | 97.3 % — 21 % catheter-led (the easy-seed sprints), 19 % recovery, median 71 steps | **27–0** |
| real patient 98 | 46.9 % — 52 % of successes catheter-led, median 84 steps, 13 % recovery; failures stall at a median 57 % (30 mid-path), 4 % loops | 93.9 % — 10 % catheter-led, 38 % recovery + 39 % won fights, median 181 steps | **52–2** |
| procedural anatomies 98 | 50.0 % — 92 % of successes catheter-led, median 43 steps, 20 % recovery; failures: 35 of 49 coiled-stuck, loop rate 49 % of all episodes | 58.2 % — 40 % catheter-led, median 47 steps, 9 % recovery; failures: 23 mid-path stalls, 8 coils, 8 shoves, loop rate 27 % | **8–0** |

The procedural policy advances catheter-first and does not recover: when the catheter carries the wire
through it is the fastest policy in the campaign, and when it stalls it stays stalled early in the
path. The carotid model uses the catheter-led sprint only on seeds where nothing goes wrong, and wins
the frontier by wire-led advance plus recovery. Catheter-led successes are therefore not what separates
the frontier models; on the carotid and host seeds the difference is entirely recovery and avoidance.

### 4.1 On the procedural anatomies

On the procedural policy's own training distribution (98 varied anatomies, one replay each) the carotid
best's success set is a superset of the procedural policy's: 49 seeds both solve, 8 only the carotid
best, 41 neither (the floor of this family), 0 only the procedural policy. Both are far from their
in-distribution form here: the procedural policy fails 35 of 49 by coiling (it loops in half of all its
episodes), the carotid best mostly by mid-path stalls (23 of 41) and loops in 27 % of episodes, ten times
its rate on its own anatomies. Neither wins much by recovery (9 % and 20 % of successes). When the
procedural policy succeeds it is catheter-led 92 % of the time; the carotid best is catheter-led on 40 %
of its successes here and 26 of its 57 wins are outright catheter-led runs — the sprint it learned late
on the carotid easy seeds transfers, the recovery does not.
Pages: `analysis_part3/procedural/decisive.html`, `compare_proc_on_proc_vs_car_best_on_proc.html`.

## 5. What this says about recovery learning and coil avoidance on v3

1. **The last four points of carotid validation are recovery.** Ten seeds decide 93 % vs 97 %; on them
   every strong checkpoint wins half or more of its successes by unload-and-repush, and the best
   checkpoint's remaining failures are recoveries that did not convert at the last bend. Nothing at the
   frontier is won by a faster clean run or by a catheter-led sprint.
2. **Coil avoidance is learned by 1.55 M and is what the ablation lacks.** The baseline stops feeding
   before a loop can form (marginal loops, undone in 1–3 steps); the masked ablation answers the same
   doorstep stall by shoving the catheter and coiling, and its decayed checkpoint regresses to mid-path
   stalls. On the real patient the same property makes the carotid best the only model with zero looping
   failures, where TopBrain v2 loses two thirds of its failures to coils.
3. **Catheter-led advance is a procedural-policy trait, not a frontier trait.** It appears in the carotid
   best only on easy seeds late in training and never on the frontier; the procedural policy that lives
   by it fails 57 % of the carotid frontier seeds and 53 % of the host seeds with no recovery to fall back on.

## 6. Recommended frontier evaluation set

Replay the 33 decisive carotid seeds (10 frontier + 23 strength) four times (~30 min) instead of the
98-seed single shot: the frontier ten decide the last four points, the strength 23 catch regressions to
the 75–90 % regime, the easy 14 and the 51 excluded earlier only add twist noise. For the real patient,
the 23 frontier host seeds (`analysis_part2/seed_table.csv`, class `frontier`) plus the 57 strength seeds.

## 7. Caveats

- Replays vary by twist; 1–4 repetitions per checkpoint; the ablation's host score and the TopBrain v3
  reference are single replays.
- "Recovery" is insertion-based; the tip-retreat kind is a subset (`tip_retreat_max` in `episodes.csv`).
- Rendering: the progress trace and tip inset follow the leading device (in a shove the trace reads
  0.9 while the guidewire tip is proximal); 500+ mm of crammed wire can lie below the drawn window; on
  two anatomies a doorstep stall is drawn touching the target.
- TopBrain v2's replay-to-replay variance (§3) is unexplained.
- The procedural policy is evaluated with this worktree's environment code and the eval flags of its own
  training (`--cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight 0.5`), on our anatomies.

## 8. Files and pages

- `README.md` — protocol, seed sets, checkpoint tables, blind check, launch fixes.
- `part{1,2,3}_results.tsv` (+ `part2_results_pooled.tsv`) — every run with its record dir and score.
- `analysis_part1/`, `analysis_part2/`, `analysis_part3/{carotid,host,procedural}/` — `SUMMARY.md`,
  `per_checkpoint.csv`, `seed_table.csv`, `failures.csv`, `paired.csv`, `recovery_by_class.csv`,
  `episodes.csv`, `decisive.html`, `compare_*.html`.
- `anim_carotid/`, `anim_host/`, `anim_procedural/` — `<run>/<type>/<seed>_…/episode.gif`, `trace.png`,
  `key/`; `by_seed/*.html`, `overview.html`.
- `blind_check/` — the blind visual verification.
