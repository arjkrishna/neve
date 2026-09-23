# Frontier replay campaign (2026-09-20)

Purpose: decide, seed by seed and with repetitions (ANIM_PIPELINE tier 2), which
validation seeds measure performance at the frontier, and film them — for carotid v3
(within one run and across the baseline vs the actor-masked ablation), for carotid v3
vs TopBrain v2 on the real patient (coils), and for carotid v3 vs the procedural policy
(catheter-led advancement). The ablation training run was stopped at 2.93 M steps to
free the machine (last checkpoint 2757954, final validation 84.7 %).

Every episode is recorded (`training _scripts/util/traj_record.py`, one `.npz` per
episode with both device polylines); replays are NOT episode-identical to the in-run
evals (the device-twist RNG is unseeded), which is why each checkpoint is replayed
several times on the same seeds and the unit of analysis is the per-seed success
probability over repetitions.

## Seed sets

**Carotid frontier set — 47 of the 98 in-run validation seeds** (`monitoring/traj/campaign/frontier_seeds.json`).
From the 14 in-run validation blocks with success ≥ 80 % across the carotid runs
(car_v3 blocks 1,3,4,5,6; car_v3a block 1; car_nopriv blocks 3–9,11), per seed:
- A hard (24): failed in ≥ 2 of the 14 blocks — 50,122,80,118,44,139,8,120,134,55,162,108,23,93,154,2,39,140,171,124,155,161,91,148
- B marginal (16): failed in exactly 1 — 16,115,18,37,47,10,31,34,73,43,110,3,70,81,167,1
- C recovery-heavy (7): never failed but ≥ 5 of 14 successes were recovery-type — 48,138,21,69,63,89,97
- EASY (51, excluded): never failed and clean/fast in ≥ 9 of 14 blocks.

**Host set — 85 of the 98 host seeds (900000–900097)**: the seeds TopBrain v2 fails in
≥ 50 % of its 12 recorded host sessions (48) plus the ones it fails sometimes (37);
the 13 it always solves are excluded. The carotid best checkpoint is additionally run
once on all 98 to reproduce the other machine's ~95 %.

## Part 1 — carotid v3 (launcher `launch_eval_carotid_traj.sh`, 16 workers, 47 seeds)

| name | checkpoint | in-run val | reps |
|---|---|---|---|
| car_best | baseline 2026-09-09_213120 ck2289002 | 96.9 % | 4 |
| abl_best | ablation 2026-09-17_180209 ck1005189 (actor mask) | 91.8 % | 4 |
| car_peak | baseline ck1791699 | 96.9 % | 2 |
| abl_last | ablation ck2757954 (actor mask) | 84.7 % | 2 |
| car_1545 | baseline ck1545187 | 94.9 % | 1 |
| car_early | baseline ck800346 | 76.5 % | 1 |
| car_div | baseline ck1336215 (critic divergence) | 38.8 % | 1 |
| abl_copeak | ablation ck760544 (actor mask) | 91.8 % | 1 |

Masked checkpoints run with `EVE_RL_ACTOR_ZERO_OBS=101,102,106,107,113,115,116,118,119,123`.

## Part 2 — real patient (launcher `launch_eval_traj.sh`, `--real_patient_anatomy --cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight 0.5`, the exact host-sweep flags)

| name | checkpoint | seeds | reps |
|---|---|---|---|
| car_best_host98 | carotid ck2289002 | all 98 | 1 |
| car_best_host | carotid ck2289002 | host set 85 | 2 |
| tbv2_best_host | TopBrain v2 best_checkpoint (72–75 % on host) | host set 85 | 2 |
| tbv3_ref_host | TopBrain v3 ck256854 (99–100 % on host) | host set 85 | 1 |
| abl_best_host | ablation ck1005189 (actor mask) | host set 85 | 1 |

## Part 3 — procedural vs carotid (catheter-led)

The procedural v3c policy (2026-07-22_010917, ck252305, its best 62 %) is an AWAC run
with a different observation layout (no residual heuristic, no heuristic-action obs,
no privileged actor: critics 121-wide, policy 97) and a 4-unit auxiliary policy head
(`--aux_labels 0,1,5,6`). It is evaluated with
`ARCH_FLAGS="--algo awac --hidden 256 256 --aux_labels 0,1,5,6 --embedder_layers 0 --log_std_min -2 --log_std_max 0.0"`
(the launchers use `${ARCH_FLAGS:-...}`, so the override replaces the residual/heuristic/privileged defaults
entirely; without `--aux_labels` the state_dict load fails on `body._output_layers.2`). The first Part 3
attempt also failed because the checkpoint lives on the read-only `results16` mount and the evaluator
writes next to the checkpoint, so the run dir (checkpoint + config + yml) was copied into this worktree's
results tree (`2026-07-22_010917_rcca_procedural_v3c/`) and the manifest `part3b_procedural.tsv` points there.
A third TopBrain v2 host replay (`tbv2_best_host3`) was appended to Part 3 because its first two replays
differed by 22 points (41/85 vs 60/85) on the same seeds.

| name | policy | anatomy | seeds |
|---|---|---|---|
| proc_on_carotid | procedural ck252305 | carotid v3 validation (16 anatomies) | frontier 47 |
| proc_on_host | procedural ck252305 | real patient | all 98 |
| proc_on_proc | procedural ck252305 | procedural varied (its own) | 98 |
| car_best_on_proc | carotid ck2289002 | procedural varied | 98 |

## Typing caveat (found on the first run)

Records reproduce the STEP log to the millimetre, so the atlas typing applies unchanged —
but the k-means "catheter-led sprint" type (2) also absorbs merely fast clean successes:
15 of the baseline-best's 47 frontier episodes were typed 2 with a catheter lead of only
8–29 mm (0 % of steps above 50 mm), against 64 mm for the atlas' genuine sprints.
Catheter-led behaviour (Part 3) is therefore scored on explicit measurements —
`lead_end`, fraction of steps with lead > 20 / 50 mm — not on the type id; the same
holds for coils, scored on slack and (where polylines exist) geometry. Type ids remain
the summary of *how* an episode went; per-seed distributions over repetitions are the
unit, because the twist alone changes the type on ~40 % of same-outcome seeds.

## Blind visual check of the automated labels (`blind_check/`)

26 decisive-seed replays (20 failures across the four Part-1 checkpoints replayed so far, 6 deep-recovery
successes of the baseline best) were shown to a separate agent as key frames + trace only, with no labels
(`blind_episodes.csv`); its reading (`blind_answers.csv`) against the automated labels (`blind_key.csv`):

- outcome 26/26; end location matches the phase for every failure (doorstep-stall -> doorstep 9/9,
  stuck-midpath -> middle third 4/4, shove-capped -> guidewire tip mid/proximal while the catheter sits at
  the doorstep, the two "deep-recovery" failures = late escapes truncated mid-advance at 83 % of the path);
- loops 23/26: every loop the slack rule flags is a real coil in the frames (5/5); the three disagreements
  are small tip fold-overs (J-hooks) at the target, which the rule rightly does not count as coils;
- recovery: of six successes labelled deep-recovery, a *visible tip* pull-back precedes the success in two;
  in the other four the wire is withdrawn at the insertion point (51-300 mm in total) while the tip barely
  moves, i.e. the withdrawal releases stored slack (unloads the buckle) rather than retreating the tip.
  "Recovery" in these tables therefore means insertion-based unload-and-re-advance inside a stall; the
  tip-retreat kind is a subset.

Rendering caveats the check surfaced: the progress trace and the tip inset follow the leading device, so in a
shove the trace reads 0.9 while the guidewire tip is proximal; 500+ mm of crammed wire can lie below the
drawn window (the coil of `abl_last` seed 2 is not visible); and on `casem030righttopcowmr003L` and
`casew013righttopcowmr013L` the planned path curls in its last 10 %, so a doorstep stall at 88-94 % of
the path is drawn touching the target marker - only the trace tells it from a success.

## Outputs

- `records`: `<checkpoint dir>/eval_anatomies_<ckpt>/traj_records/<run_tag>/` (one `.npz` per episode); `campaign/*_results.tsv` maps each run to its record dir and official score.
- `analysis_part{1,2,3}/` (from `monitoring/traj/traj_frontier.py <results.tsv> --out <dir> --strong <names> --pairs a:b --anim <anim dir>`):
  - `per_checkpoint.csv`: success, share of successes won by recovery, catheter-led share, failure character
    (median max-progress of failures, mid-path share, loop share), loop rate and loop-escape rate;
  - `seed_table.csv` / `SUMMARY.md`: per seed the success probability under every checkpoint, the dominant
    success phase and failure phase, and the class — **easy** (every checkpoint always solves it), **strength**
    (all strong checkpoints always solve it, a weak one fails it), **frontier** (a strong checkpoint fails it in
    some repetition), **floor** (no strong checkpoint ever solves it); frontier + strength + floor = the decisive seeds;
  - `recovery_by_class.csv`: how successes are won (free-run / cath-led / light / deep recovery / fight) on easy vs frontier seeds;
  - `failures.csv`, `failures_by_phase.csv`: every failure with where it ended (`p_end`, `p_max`), how long it idled, whether a loop formed;
  - `paired.csv`: seed-paired sign / Wilcoxon tests between checkpoints on per-seed success probabilities;
  - `decisive.html`: every replay of every decisive seed with links to its GIF, trace and key frames;
  - `episodes.csv`: one row per replayed episode (atlas type, phase, indicators).
- `anim_carotid/`, `anim_host/`, `anim_procedural/`: rendered key frames, traces and GIFs per run (`<tag>/<type>/<seed>_.../`), `by_seed/*.html` and `overview.html` across all runs (`traj_render.py --reindex`).

- `slides/` — 12 PNG slides (CB3 vs ablation / TopBrain v2 / procedural, zoomed nook comparisons with the local vessel wall), `SLIDES.md` lists them; regenerate with `monitoring/traj/traj_slides.py`.
