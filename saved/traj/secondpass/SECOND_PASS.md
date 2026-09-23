# Second pass on the trajectory atlas: are the 11 k-means types behavioural indicators of success?

Date: 2026-09-20. Data: every complete episode in the atlas feature set (152,112 episodes: 139,496 explore, 8,791 validation, 3,825 host; 138,555 carry a k-means id, the 13,557 `car_nopriv` episodes extracted after the clustering run are used as a hold-out). Code: `monitoring/traj/secondpass/phase_taxonomy.py`; labels: `saved/traj/secondpass/phase_labels.csv`. Only pandas/numpy/scipy were used.

## Verdict

**The 11 k-means types are adequate outcome *predictors* but poor behavioural *indicators*, and the reason is structural: their feature matrix contains the outcome.** Every failure in this data is a 600-step timeout (`max_steps` and the env's `truncated` flag, both always at step 600; 37 `vessel_end` aside), so `steps < 600` is success. `traj_cluster.py` feeds k-means with `steps`, `prog_final`, `prog_max`, `t_25..t_90` and the 20-point progress profile; a 3-bin `steps` band alone has information gain 0.996 for success. The pure types (0, 1, 2, 9, 10) are pure because they encode "short episode at the target" vs "600 steps, no progress", not because of a stall/recovery signature. The mixed types 4/5/6 are exactly where that leakage is weakest, and there the cluster id explains 18 % of the outcome entropy while an explicit phase reading explains 66 %.

**An explicit phase taxonomy does better, with no length or progress input at all.** Twelve categories built from the stall-event process (did the frontier stall; did every stall close; how many, how deep; where was the last stall; catheter-shove / coil regime; idle driving) reach IG 0.881 vs 0.857 for the clusters on the clustered set, with better 5-fold cross-validated Brier (0.014 vs 0.017) and log-loss (0.050 vs 0.060), better explore (IG 0.922 vs 0.878) and host (0.773 vs 0.621) purity, and it transfers unchanged to the hold-out run (Brier 0.010). It is worse on validation evals (IG 0.652 vs 0.807) for one identifiable reason: the canonical stall detector needs a wire push command > 2 to register a stall, and the deterministic eval policy often creeps below that, so 34 % of validation failures have no detected stall at all. Adding a coarse severity level (max-progress band: reached / far / short) gives a 2-level scheme that dominates everywhere (IG 0.954; val 0.915, host 0.877; CV Brier 0.005, log-loss 0.020) at the price of using a near-outcome quantity, which is why it is reported separately.

**For the 4 / 5 / 6 boundary the answer is clean.** Successful type-5 episodes are 76 % closed-stall recoveries (deep-recovery 31 %, fight-escaped-near 29 %, light-recovery 16 %), i.e. what type 4 is meant to be; failed type-4 episodes are 72 % open-stall failures (shove-capped 30 %, coiled-stuck 31 %, stuck-midpath 12 %), i.e. what types 5/6/8 are meant to be. The same phase has the same success rate wherever k-means put it (deep-recovery 1.00 / 0.94 / 0.95 in types 4 / 5 / 6; light-recovery 0.99 / 0.93 / 0.94; fight-escaped-near 0.90 / 0.80 / 0.90; shove-capped 0.05 / 0.03 / 0.06; stuck-midpath 0 / 0 / 0). The k-means boundaries between 4, 5 and 6 cut across behaviourally identical episodes, sorting them by how long the fight lasted. The one genuinely fuzzy cell is `doorstep-stall` (last stall still open inside the last 10 % of the path, 40 % success): its successes are the tip wobbling into the target tolerance while the frontier detector still reads "stuck", which no pre-terminal behaviour predicts. Likewise, once a multi-stall fight has closed its last stall, success is decided by whether the wire then reaches the target zone before step 600 (`fight-escaped-far`: 96 % if it reaches 0.9 of the path, 0 % otherwise). "Deep recovery that escaped later" and "fight that ran out of steps" are the same phase with different remaining budget; that is a step-budget artefact, not a behavioural difference, and no clustering will resolve it.

**Recommendation for outcome attribution.** Replace the cluster id with `phase_type` (+ `severity` when a near-outcome axis is acceptable) as the categorical variable in the atlas tables, the eval-block and host-checkpoint composition reports, and the animation storyboard selection. Keep the k-means types, if at all, as a descriptor of the progress-curve *shape*, never as evidence that a behaviour caused an outcome. Fix the detector blind spot before the next extraction (see Limitations).

## The phase taxonomy (`phase_taxonomy.phase_type`, level 1; `phase_type2` adds the severity)

Reads: `events` (list of `{k, r, close, onset, first, p0}`), `pl`, `ev_n`, `ev_unrec`, `ev_ret_max`, `cath_lead_max`, `slack_max`, `cath_slack_max`, `cmd_push_frac`; level 2 adds `prog_max`. Decision order: no stall -> catheter-led / idle / free; all stalls closed -> count and depth, location for fights; last stall open -> mechanism (shove, coil) then location. Coil uses the project's own definition (cath_slack > 50 mm or wire slack > 100 mm); the 8 mm withdrawal cut is the detector's `soft_max`.

| phase | definition | n | success | explore | val | host |
|---|---|---|---|---|---|---|
| free-run | no stall event; wire-led (catheter never > 30 mm ahead); the policy pushed | 104,962 | 1.000 | 1.000 | 0.999 | 1.000 |
| cath-led-run | no stall event; catheter ran > 30 mm ahead of the wire at some point (procedural-era sprint) | 11,514 | 0.978 | 1.000 | 0.846 | 0.857 |
| idle-creep | no stall event, wire-led, wire push \|cmd\| > 10 on < 10 % of steps: idling or gentle creeping the detector cannot see | 784 | 0.337 | 1.000 (n=2) | 0.314 | 0.839 |
| light-recovery | 1-2 stalls, all closed, max withdrawal inside a stall <= 8 mm (grind or soft) | 8,597 | 0.977 | 0.991 | 0.824 | 0.940 |
| deep-recovery | 1-2 stalls, all closed, at least one withdrawal > 8 mm | 2,795 | 0.928 | 0.938 | 0.813 | 0.890 |
| fight-escaped-near | >= 3 stalls, all closed; last stall beyond 0.66 of the path | 1,893 | 0.859 | 0.864 | 0.885 | 0.765 |
| fight-escaped-far | >= 3 stalls, all closed; last stall at or before 0.66 of the path | 1,889 | 0.430 | 0.424 | 0.586 | 0.227 |
| doorstep-stall | last stall open at the end, in the target zone (p0/pl > 0.9), no shove, no coil | 1,350 | 0.396 | 0.371 | 0.216 | 0.658 |
| shove-capped | last stall open, catheter shoved > 80 mm ahead of the wire (any location) | 9,989 | 0.011 | 0.007 | 0.008 | 0.090 |
| coiled-stuck | last stall open with a coil (cath_slack > 50 or wire slack > 100 mm), lead <= 80 mm | 1,667 | 0.057 | 0.063 | 0.036 | 0.042 |
| stuck-midpath | last stall open between 0.33 and 0.9 of the path, plain mechanism | 2,686 | 0.000 | 0.000 | 0.000 | 0.000 |
| stuck-proximal | last stall open in the first third of the path, plain mechanism | 3,986 | 0.001 | 0.000 | 0.022 | 0.000 |

Severity (level 2): `reached` = prog_max >= 0.9, `far` = 0.5-0.9, `short` = < 0.5. The level-2 cells are nearly pure (T1b): every closed-stall or no-stall phase is 0.94-1.00 when `reached` and 0.00-0.04 otherwise; the open-stall phases are 0 when not `reached` and 7-40 % when `reached` (the doorstep wobble-in).

## Headline numbers (clustered episodes, n = 138,555; base success 0.852 / explore 0.867 / val 0.732 / host 0.621)

In-sample purity = information gain 1 - H(success | category) / H(success). CV = 5-fold, category success rates fitted on four folds with Laplace smoothing and scored on the fifth; logistic regressions are ridge-IRLS in numpy on standardised raw features.

| predictor | cats | IG all | IG explore | IG val | IG host | CV Brier all | CV log-loss all | Brier explore | Brier val | Brier host | AUC all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k-means 11 types | 11 | 0.857 | 0.878 | 0.807 | 0.621 | 0.017 | 0.060 | 0.014 | 0.033 | 0.089 | 0.995 |
| phase L1 (this work) | 12 | 0.881 | 0.922 | 0.652 | 0.773 | 0.014 | 0.050 | 0.009 | 0.069 | 0.050 | 0.995 |
| phase L2 (phase x severity) | 30 | 0.954 | 0.964 | 0.915 | 0.877 | 0.005 | 0.020 | 0.004 | 0.012 | 0.031 | 0.999 |
| k-means x phase L1 | 97 | - | - | - | - | 0.008 | 0.030 | 0.006 | 0.022 | 0.043 | 0.998 |
| last stall open / closed (1 bit) | 2 | 0.752 | 0.813 | 0.406 | 0.623 | 0.021 | 0.104 | 0.014 | 0.112 | 0.067 | 0.939 |
| prog_max band | 3 | 0.751 | 0.756 | 0.758 | 0.647 | 0.022 | 0.104 | 0.019 | 0.033 | 0.070 | 0.924 |
| giveback_mm band | 4 | 0.641 | 0.698 | 0.429 | 0.364 | 0.045 | 0.150 | 0.037 | 0.121 | 0.152 | 0.958 |
| catheter lead band | 3 | 0.278 | 0.317 | 0.105 | 0.109 | 0.089 | 0.302 | 0.078 | 0.181 | 0.245 | 0.809 |
| LR, 12 behaviour-only features | - | - | - | - | - | 0.011 | 0.039 | 0.007 | 0.059 | 0.047 | 0.998 |
| LR + cumulative retract / giveback / flat | - | - | - | - | - | 0.005 | 0.019 | 0.003 | 0.017 | 0.032 | 0.999 |
| LR + prog_max | - | - | - | - | - | 0.002 | 0.009 | 0.001 | 0.008 | 0.015 | 0.999 |
| steps band (oracle: 600 = timeout) | 3 | 0.996 | 0.998 | 0.972 | 1.000 | 0.000 | 0.002 | 0.000 | 0.002 | 0.000 | 1.000 |

Robustness. Leave-runs-out folds (whole run tags held out): k-means Brier 0.019 / log-loss 0.067, phase L1 0.017 / 0.059, phase L2 0.005 / 0.021. Hold-out run never seen by the clustering (`car_nopriv`, 13,557 episodes, rates fitted on the clustered set): phase L1 Brier 0.010, log-loss 0.036, in-sample IG 0.882; phase L2 0.004 / 0.013 / 0.961. The ordering of the single axes is stable across kinds: open/closed last stall (0.75) >> last-stall location (0.58) >> catheter lead (0.28) >> coil (0.11), which is the decision order used in the taxonomy.

Reading the table. (i) One bit of behaviour, "is the last stall still open", already captures 88 % of what the 11 clusters capture. (ii) The clusters add nothing on top of the phase that the severity band does not add better (k-means x phase 0.008 vs phase L2 0.005). (iii) The behaviour-only logistic regression (no cumulative, no length, no progress) at 0.011 shows the phase categories lose little by being discrete. (iv) The oracle rows are the ceiling: length is the outcome.

## Clusters 4 / 5 / 6 in detail (n = 7,543; success 0.470; base entropy 0.997 bits)

Single-feature separation inside each cluster (rank AUC, success positive; appendix T5). The strongest separators everywhere are the outcome itself or fractions of episode length: `steps` and `dtgt_min` (AUC 0.00), `prog_final`, `ev_stalled_steps` (0.02-0.04), `flat_frac`, `t_max_frac`. Among genuinely behavioural features the best are `ev_unrec` (0.09 / 0.09 / 0.12: failures end with an open stall), `withdrawn_mm` (0.10 / 0.13 / 0.17), `ret_bouts`, `reversals`, and for 5 and 6 the catheter regime (`cath_lead_max` 0.24 / 0.29, `cath_slack_max` 0.26). Nothing within a cluster separates success from failure as well as the open/closed reading of the last stall, which is the taxonomy's first split.

| predictor, within 4/5/6 | cats | H (bits) | IG | Gini | CV Brier | CV log-loss | CV AUC |
|---|---|---|---|---|---|---|---|
| cluster id (4 / 5 / 6) | 3 | 0.819 | 0.179 | 0.381 | 0.191 | 0.568 | 0.735 |
| phase L1 | 11 | 0.341 | 0.658 | 0.134 | 0.067 | 0.238 | 0.957 |
| cluster + phase L1 | 31 | 0.314 | 0.685 | 0.124 | 0.063 | 0.223 | 0.965 |
| phase L2 | 22 | 0.235 | 0.764 | 0.094 | 0.047 | 0.166 | 0.976 |
| cluster + phase L2 | 55 | 0.212 | 0.787 | 0.084 | 0.043 | 0.155 | 0.982 |

Where the ambiguous episodes land (% of group):

| group | n | light-rec | deep-rec | fight-near | fight-far | doorstep | shove-capped | coiled | stuck-mid |
|---|---|---|---|---|---|---|---|---|---|
| type-5 SUCCESS | 979 | 15.9 | 30.9 | 29.0 | 13.4 | 5.9 | 4.5 | 0.0 | 0.0 |
| type-4 FAIL | 486 | 1.0 | 0.4 | 7.8 | 1.4 | 17.1 | 29.8 | 30.7 | 11.7 |
| type-4 success | 2,049 | 44.0 | 28.9 | 17.6 | 1.5 | 5.0 | 0.3 | 1.9 | 0.0 |
| type-5 fail | 2,081 | 0.5 | 1.0 | 3.3 | 3.1 | 6.3 | 68.9 | 2.1 | 14.8 |
| type-6 success | 515 | 9.7 | 56.7 | 11.7 | 0.2 | 11.7 | 5.0 | 4.9 | 0.0 |
| type-6 fail | 1,433 | 0.2 | 1.0 | 0.5 | 0.3 | 33.6 | 26.6 | 27.6 | 10.0 |

Consistency across the three clusters (success rate of a phase inside type 4 / 5 / 6, cells with n >= 20): light-recovery 0.99 / 0.93 / 0.94; deep-recovery 1.00 / 0.94 / 0.95; fight-escaped-near 0.90 / 0.80 / 0.90; fight-escaped-far 0.82 / 0.67 / (n=5); doorstep-stall 0.55 / 0.31 / 0.11; shove-capped 0.05 / 0.03 / 0.06; coiled-stuck 0.20 / 0.00 / 0.06; stuck-midpath 0 / 0 / 0. A successful type-5 and a successful type-4 episode read the same (a closed recovery), and a failed type-4 reads like a failed type-6 (open stall at the doorstep, coiled, or shove-capped). The phase where the cluster id still carries information is `doorstep-stall`: k-means saw the step count, and the shorter doorstep episodes (in type 4) are the ones whose tip wobbled into tolerance.

What the k-means boundary actually is: type 4 has median 216 steps, types 5 and 6 have median 600. Given the same phase, the clusters sort episodes by duration and by the catheter-lead / slack profile, which the phase scheme carries explicitly as `shove-capped` / `coiled-stuck` and as the depth of the recovery.

## Blind-label sanity check (66 eye-labelled end-of-episode snapshots)

Consistency = the assigned category could show the picture the labeller described (mapping stated in appendix T9; the phase side also accepts the scheme's own coil and lead >= 30 mm flags for "coil" / "catheter-ahead"). Phase-consistent 56/66 = 85 %, cluster-consistent 50/66 = 76 %. By primary label (phase / cluster): clean 0.96 / 0.92 (n=24), coil 0.89 / 0.67 (18), catheter-ahead 0.91 / 0.64 (11), stopped-short 0.88 / 0.88 (8), near-target 0 / 0 (3), wrong-branch 0 / 1 (2). Axis-level: eye coil vs coil flag 16/18 agree with 1 false positive in 48; eye catheter-ahead vs lead >= 30 mm 25/28 (11/38 flagged episodes the labeller did not call, expected: the flag is a maximum over the episode, the snapshot is the last frame); eye stopped-short vs open last stall 23/26 (the 8/40 open stalls the labeller did not call short were all placed "at-target", mostly as coil / catheter-ahead pictures: seven doorstep, coiled or shove-capped failures whose tip sits at the target, and one success, id 10, whose tip wobbled into tolerance with the stall still open); eye progress band vs severity 59/66. The three "near-target" misses are two `fight-escaped-far/far` episodes whose wire crept to ~20 mm short after the last detected stall and one 25-step catheter-led success; the two wrong-branch labels have no phase category (see Limitations).

## Limitations and what to fix

1. Detector blind spot. The canonical detector (`buckle_clear_dump_v1` thresholds: frontier flat for 12 steps *while the wire push command exceeds 2*) does not fire when the policy creeps or when the catheter, not the wire, is being pushed. 776 failures have no stall event (770 of them validation evals = 34 % of val failures; median wire push fraction 0, median progress 0.02 or 0.92, i.e. never moved or crept to the doorstep) and 1,738 failures (8 %) end after their last stall closed. Level 1 cannot see their terminal stall; level 2 catches them through the progress band. A frontier-flatness detector that ignores the push command (or reads the catheter command for catheter-led episodes) would move most of them into the open-stall phases and remove the val gap. `evS_*` is the same detector on the signed command, not a more sensitive one.
2. Wrong branch. Off-path fields are absent for 56,152 episodes (older logs), so wrong-branch is not a category; where available it could be a modifier on the open-stall phases (1,919 episodes, 16 % success, mostly clusters 7 and 10).
3. Thresholds are hand-set from the detector's own constants and the project's coil definition (30 / 80 mm lead, 8 mm withdrawal, 3 events, 0.33 / 0.66 / 0.9 of the path, 10 % push fraction) and were not tuned on the outcome; the purity gap to the behaviour-only logistic regression (0.014 vs 0.011 Brier) bounds what tuning could buy.
4. `doorstep-stall` (40 %) and `fight-escaped-far` (43 %) are the impure level-1 cells by construction: their outcome is whether the tip arrived before step 600, which is what the atlas is trying to explain, not a behaviour that precedes it.
5. Level 2 uses `prog_max`, which is necessary for success (prog >= 0.9); treat level-2 purity as descriptive, not as behaviour predicting outcome.

## Files

- `monitoring/traj/secondpass/phase_taxonomy.py`: `phase_type(row)`, `phase_type2(row)`, `severity(row)`, `regimes(row)`, `describe()`; works on a dict, a DataFrame row or a raw features.jsonl record.
- `saved/traj/secondpass/phase_labels.csv`: key, cluster (empty for the 13,557 unclustered car_nopriv episodes), phase_type, success, kind, tag, severity, phase_type2 for all 152,112 episodes.
- `saved/traj/secondpass/SECOND_PASS.md`: this file. The appendix below is the full machine-generated table set (T1-T9).

---

# Appendix: full tables

Machine-generated by the analysis script (scratchpad `secondpass_work/analysis.py`); T-numbers are referenced from the text above.

Episodes: 152112 complete (138555 clustered by the atlas; 13557 car_nopriv episodes added after the clustering run are used as a hold-out for the phase scheme).
kind counts: explore=139496, val=8791, host=3825. Outcomes: success=130621, max_steps=19945, truncated=1509, vessel_end=37.
Every non-success is a 600-step timeout (the env's 'truncated' flag is also always at step 600) except 37 vessel_end episodes, so steps<600 <=> success.

## T1. Phase categories: size and success by kind (all 152112 episodes)

| phase | n | succ | steps_med | coil | shove | n_explore | succ_explore | n_val | succ_val | n_host | succ_host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| free-run | 104962 | 1.000 | 79 | 0.001 | 0.000 | 99241 | 1 | 4188 | 0.999 | 1533 | 1 |
| cath-led-run | 11514 | 0.978 | 39 | 0.005 | 0.578 | 9868 | 1 | 1639 | 0.846 | 7 | 0.857 |
| idle-creep | 784 | 0.337 | 600 | 0.000 | 0.000 | 2 | 1 | 751 | 0.314 | 31 | 0.839 |
| light-recovery | 8597 | 0.977 | 130 | 0.076 | 0.104 | 7647 | 0.991 | 567 | 0.824 | 383 | 0.940 |
| deep-recovery | 2795 | 0.928 | 212 | 0.111 | 0.189 | 2433 | 0.938 | 134 | 0.813 | 228 | 0.890 |
| fight-escaped-near | 1893 | 0.859 | 428 | 0.173 | 0.277 | 1765 | 0.864 | 26 | 0.885 | 102 | 0.765 |
| fight-escaped-far | 1889 | 0.430 | 600 | 0.013 | 0.376 | 1717 | 0.424 | 128 | 0.586 | 44 | 0.227 |
| doorstep-stall | 1350 | 0.396 | 600 | 0.000 | 0.067 | 1153 | 0.371 | 51 | 0.216 | 146 | 0.658 |
| shove-capped | 9989 | 0.011 | 600 | 0.309 | 1 | 8533 | 0.007 | 901 | 0.008 | 555 | 0.090 |
| coiled-stuck | 1667 | 0.057 | 600 | 1 | 0.242 | 1247 | 0.063 | 112 | 0.036 | 308 | 0.042 |
| stuck-midpath | 2686 | 0.000 | 600 | 0.000 | 0.321 | 2248 | 0.000 | 114 | 0.000 | 324 | 0.000 |
| stuck-proximal | 3986 | 0.001 | 600 | 0.000 | 0.130 | 3642 | 0.000 | 180 | 0.022 | 164 | 0.000 |

## T1b. Level-2 cells (phase x severity) with n >= 30

| phase | sev | n | succ |
|---|---|---|---|
| free-run | reached | 104942 | 1.000 |
| cath-led-run | far | 89 | 0.000 |
| cath-led-run | reached | 11283 | 0.998 |
| cath-led-run | short | 142 | 0.000 |
| idle-creep | reached | 275 | 0.960 |
| idle-creep | short | 507 | 0.000 |
| light-recovery | far | 55 | 0.036 |
| light-recovery | reached | 8447 | 0.994 |
| light-recovery | short | 95 | 0.000 |
| deep-recovery | far | 79 | 0.000 |
| deep-recovery | reached | 2618 | 0.991 |
| deep-recovery | short | 98 | 0.000 |
| fight-escaped-near | far | 159 | 0.006 |
| fight-escaped-near | reached | 1734 | 0.937 |
| fight-escaped-far | far | 433 | 0.002 |
| fight-escaped-far | reached | 845 | 0.961 |
| fight-escaped-far | short | 611 | 0.000 |
| doorstep-stall | reached | 1350 | 0.396 |
| shove-capped | far | 7936 | 0.000 |
| shove-capped | reached | 1537 | 0.074 |
| shove-capped | short | 516 | 0.000 |
| coiled-stuck | far | 823 | 0.000 |
| coiled-stuck | reached | 773 | 0.123 |
| coiled-stuck | short | 71 | 0.000 |
| stuck-midpath | far | 1184 | 0.000 |
| stuck-midpath | short | 1500 | 0.000 |
| stuck-proximal | short | 3986 | 0.001 |

## T2. How each k-means type decomposes into phases (row %, clustered episodes)

| cname | free-run | cath-led-run | idle-creep | light-recovery | deep-recovery | fight-escaped-near | fight-escaped-far | doorstep-stall | shove-capped | coiled-stuck | stuck-midpath | stuck-proximal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0:clean-success | 96.6 | 3.2 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1:clean-fast-success | 99.1 | 0.2 | 0.7 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2:cathled-sprint-success | 19 | 75.9 | 0.0 | 4.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3:light-recovery-success | 1.2 | 0.2 | 0.1 | 69.1 | 12.1 | 9.4 | 3.6 | 4 | 0.1 | 0.2 | 0.1 | 0.0 |
| 4:deep-recovery-success | 0.4 | 0.2 | 0.0 | 35.7 | 23.5 | 15.7 | 1.5 | 7.3 | 6 | 7.4 | 2.2 | 0.0 |
| 5:shove-fight | 0.1 | 0.1 | 0.0 | 5.5 | 10.6 | 11.5 | 6.4 | 6.2 | 48.3 | 1.4 | 10 | 0.0 |
| 6:near-buckle-fight | 0.1 | 0.0 | 0.1 | 2.7 | 15.8 | 3.4 | 0.3 | 27.8 | 20.9 | 21.6 | 7.4 | 0.0 |
| 7:midpath-thrash | 0.0 | 0.4 | 0.2 | 4.6 | 3.6 | 4.5 | 23.2 | 0.7 | 12.2 | 1.1 | 36 | 13.6 |
| 8:mega-coil | 0.0 | 0.0 | 0.0 | 3.6 | 1.3 | 2.3 | 0.0 | 0.0 | 53.9 | 38.4 | 0.5 | 0.0 |
| 9:shove-cap | 0.0 | 0.9 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 98.4 | 0.0 | 0.0 | 0.0 |
| 10:proximal-thrash | 0.0 | 2.5 | 11.1 | 1.3 | 0.5 | 0.0 | 3.3 | 0.0 | 2.3 | 1.3 | 2.5 | 75.2 |

## T2b. Success rate of each phase inside each k-means type (cells with n >= 20)

| cname | free-run | cath-led-run | idle-creep | light-recovery | deep-recovery | fight-escaped-near | fight-escaped-far | doorstep-stall | shove-capped | coiled-stuck | stuck-midpath | stuck-proximal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0:clean-success | 1 | 1 |  | 1 |  |  |  |  |  |  |  |  |
| 1:clean-fast-success | 1 | 1 | 0.98 | 1 |  |  |  |  |  |  |  |  |
| 2:cathled-sprint-success | 1 | 0.99 |  | 1 | 1 |  |  |  |  |  |  |  |
| 3:light-recovery-success | 1 |  |  | 0.99 | 1 | 0.95 | 0.99 | 0.80 |  |  |  |  |
| 4:deep-recovery-success |  |  |  | 0.99 | 1 | 0.90 | 0.82 | 0.55 | 0.05 | 0.20 | 0.00 |  |
| 5:shove-fight |  |  |  | 0.93 | 0.94 | 0.80 | 0.67 | 0.31 | 0.03 | 0.00 | 0.00 |  |
| 6:near-buckle-fight |  |  |  | 0.94 | 0.95 | 0.90 |  | 0.11 | 0.06 | 0.06 | 0.00 |  |
| 7:midpath-thrash |  |  |  | 0.71 | 0.43 | 0.52 | 0.32 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8:mega-coil |  |  |  | 0.95 | 0.56 | 0.88 |  |  | 0.01 | 0.01 |  |  |
| 9:shove-cap |  | 0.00 |  | 0.81 |  |  |  |  | 0.00 |  |  |  |
| 10:proximal-thrash |  | 0.00 | 0.00 | 0.00 | 0.00 |  | 0.00 |  | 0.00 | 0.00 | 0.00 | 0.00 |

## T3. In-sample success purity (clustered episodes): weighted entropy H(success|category) in bits, information gain, Gini

| predictor | all_cats | all_H | all_IG | all_Gini | IG_explore | IG_val | IG_host |
|---|---|---|---|---|---|---|---|
| k-means 11 types | 11 | 0.086 | 0.857 | 0.035 | 0.878 | 0.807 | 0.621 |
| phase L1 (12) | 12 | 0.072 | 0.881 | 0.027 | 0.922 | 0.652 | 0.773 |
| phase L2 (phase x severity) | 30 | 0.028 | 0.954 | 0.010 | 0.964 | 0.915 | 0.877 |
| steps band (oracle: 600 = timeout) | 3 | 0.003 | 0.996 | 0.000 | 0.998 | 0.972 | 1.000 |
| prog_max band | 3 | 0.150 | 0.751 | 0.043 | 0.756 | 0.758 | 0.647 |
| last stall open/closed | 2 | 0.150 | 0.752 | 0.042 | 0.813 | 0.406 | 0.623 |
| last event kind | 5 | 0.127 | 0.790 | 0.040 | 0.875 | 0.409 | 0.667 |
| last stall location | 5 | 0.252 | 0.582 | 0.109 | 0.658 | 0.254 | 0.449 |
| catheter lead band | 3 | 0.436 | 0.278 | 0.178 | 0.317 | 0.105 | 0.109 |
| coil flag | 2 | 0.539 | 0.108 | 0.217 | 0.100 | 0.068 | 0.192 |
| withdrawn_mm band | 4 | 0.261 | 0.568 | 0.100 | 0.662 | 0.221 | 0.424 |
| giveback_mm band | 4 | 0.217 | 0.641 | 0.090 | 0.698 | 0.429 | 0.364 |
| max withdrawal in stall band | 3 | 0.340 | 0.437 | 0.131 | 0.499 | 0.141 | 0.251 |
| n stall events | 4 | 0.259 | 0.571 | 0.114 | 0.648 | 0.244 | 0.423 |

Base entropy H(success): all=0.604 (succ 0.852), explore=0.565 (succ 0.867), val=0.838 (succ 0.732), host=0.957 (succ 0.621)

## T4. 5-fold cross-validated Brier / log-loss / AUC of success predicted from category rates (Laplace-smoothed, fitted on 4 folds) and numpy ridge-logistic regressions on raw features

| predictor | Brier_all | logloss_all | AUC_all | Brier_explore | logloss_explore | AUC_explore | Brier_val | logloss_val | AUC_val | Brier_host | logloss_host | AUC_host |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k-means 11 types | 0.017 | 0.060 | 0.995 | 0.014 | 0.049 | 0.996 | 0.033 | 0.135 | 0.984 | 0.089 | 0.273 | 0.945 |
| phase L1 | 0.014 | 0.050 | 0.995 | 0.009 | 0.033 | 0.999 | 0.069 | 0.255 | 0.954 | 0.050 | 0.183 | 0.978 |
| phase L2 | 0.005 | 0.020 | 0.999 | 0.004 | 0.015 | 1.000 | 0.012 | 0.062 | 0.994 | 0.031 | 0.101 | 0.994 |
| k-means x phase L1 | 0.008 | 0.030 | 0.998 | 0.006 | 0.022 | 0.999 | 0.022 | 0.099 | 0.987 | 0.043 | 0.151 | 0.986 |
| steps band (oracle) | 0.000 | 0.002 | 1.000 | 0.000 | 0.001 | 1.000 | 0.002 | 0.020 | 0.995 | 0.000 | 0.000 | 1 |
| prog_max band | 0.022 | 0.104 | 0.924 | 0.019 | 0.096 | 0.924 | 0.033 | 0.146 | 0.935 | 0.070 | 0.287 | 0.904 |
| last stall open/closed | 0.021 | 0.104 | 0.939 | 0.014 | 0.075 | 0.958 | 0.112 | 0.474 | 0.786 | 0.067 | 0.278 | 0.924 |
| last stall location | 0.054 | 0.175 | 0.946 | 0.046 | 0.140 | 0.968 | 0.147 | 0.626 | 0.775 | 0.129 | 0.379 | 0.866 |
| catheter lead band | 0.089 | 0.302 | 0.809 | 0.078 | 0.269 | 0.832 | 0.181 | 0.581 | 0.683 | 0.245 | 0.808 | 0.669 |
| withdrawn band | 0.050 | 0.181 | 0.934 | 0.038 | 0.138 | 0.965 | 0.186 | 0.710 | 0.741 | 0.146 | 0.470 | 0.884 |
| giveback band | 0.045 | 0.150 | 0.958 | 0.037 | 0.121 | 0.974 | 0.121 | 0.464 | 0.861 | 0.152 | 0.465 | 0.852 |
| LR behaviour-only (12 regime/event feats) | 0.011 | 0.039 | 0.998 | 0.007 | 0.025 | 0.999 | 0.059 | 0.177 | 0.972 | 0.047 | 0.199 | 0.989 |
| LR + cumulative retract/giveback/flat | 0.005 | 0.019 | 0.999 | 0.003 | 0.012 | 1.000 | 0.017 | 0.086 | 0.994 | 0.032 | 0.117 | 0.994 |
| LR + prog_max | 0.002 | 0.009 | 0.999 | 0.001 | 0.005 | 1.000 | 0.008 | 0.049 | 0.995 | 0.015 | 0.050 | 0.999 |
| LR + prog_max + steps (oracle) | 0.001 | 0.006 | 1.000 | 0.001 | 0.003 | 1.000 | 0.006 | 0.043 | 0.996 | 0.007 | 0.027 | 1.000 |

## T4b. Same, but folds = whole run tags held out (leave-runs-out; 54 tags round-robin into 5 folds)

| predictor | Brier_all | logloss_all | AUC_all | Brier_explore | logloss_explore | AUC_explore | Brier_val | logloss_val | AUC_val | Brier_host | logloss_host | AUC_host |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k-means 11 types | 0.019 | 0.067 | 0.993 | 0.016 | 0.055 | 0.995 | 0.035 | 0.153 | 0.980 | 0.089 | 0.275 | 0.943 |
| phase L1 | 0.017 | 0.059 | 0.994 | 0.010 | 0.036 | 0.998 | 0.107 | 0.359 | 0.938 | 0.048 | 0.179 | 0.979 |
| phase L2 | 0.005 | 0.021 | 0.999 | 0.004 | 0.015 | 1.000 | 0.012 | 0.068 | 0.993 | 0.030 | 0.100 | 0.994 |
| k-means x phase L1 | 0.010 | 0.036 | 0.998 | 0.007 | 0.027 | 0.999 | 0.026 | 0.123 | 0.983 | 0.044 | 0.153 | 0.986 |

## T4c. Hold-out run never seen by the clustering (car_nopriv, n=13557): phase rates fitted on all clustered episodes

| predictor | Brier_all | logloss_all | AUC_all | Brier_explore | logloss_explore | AUC_explore | Brier_val | logloss_val | AUC_val | IG_insample |
|---|---|---|---|---|---|---|---|---|---|---|
| phase L1 | 0.010 | 0.036 | 0.997 | 0.008 | 0.030 | 0.998 | 0.040 | 0.140 | 0.989 | 0.882 |
| phase L2 | 0.004 | 0.013 | 1.000 | 0.003 | 0.012 | 1.000 | 0.009 | 0.035 | 0.998 | 0.961 |
| last stall open/closed | 0.019 | 0.093 | 0.904 | 0.017 | 0.085 | 0.912 | 0.056 | 0.246 | 0.841 | 0.653 |
| prog_max band | 0.009 | 0.057 | 0.938 | 0.008 | 0.051 | 0.946 | 0.038 | 0.167 | 0.881 | 0.804 |

## T5. Clusters 4/5/6: single-feature separation of success vs failure inside each cluster (rank AUC, success = positive class; AUC far from 0.5 either way = strong)

Features marked * are the outcome itself or a fraction of episode length (a timeout has 600 steps by definition).

| cluster | n | succ | top10 | top_behavioural |
|---|---|---|---|---|
| 4:deep-recovery-success | 2535 | 0.81 | steps*=0.00, dtgt_min*=0.00, prog_final*=0.99, ev_stalled_steps*=0.02, prog_max*=0.97, t_max_frac*=0.96, flat_frac*=0.05, flat_run_max*=0.06, ret_bouts=0.06, ev_first_onset_frac*=0.92 | ret_bouts=0.06, first_p0f=0.92, ev_unrec=0.09, withdrawn_mm=0.10, cmd_push_frac=0.90, giveback_mm=0.11, reversals=0.11, tip_tortuosity=0.12 |
| 5:shove-fight | 3060 | 0.32 | dtgt_min*=0.00, steps*=0.00, prog_final*=0.98, prog_max*=0.97, ev_stalled_steps*=0.04, t_max_frac*=0.95, flat_frac*=0.09, ev_unrec=0.09, flat_run_max*=0.10, withdrawn_mm=0.13 | ev_unrec=0.09, withdrawn_mm=0.13, ret_bouts=0.17, reversals=0.20, cath_lead_max=0.24, cmd_retract_frac=0.25, cath_slack_max=0.26, cath_lead_frac50=0.26 |
| 6:near-buckle-fight | 1948 | 0.26 | dtgt_min*=0.00, steps*=0.00, ev_stalled_steps*=0.03, prog_final*=0.96, prog_max*=0.95, t_max_frac*=0.93, flat_run_max*=0.11, ev_last_close_frac*=0.89, ev_unrec=0.12, flat_frac*=0.12 | ev_unrec=0.12, ret_bouts=0.14, withdrawn_mm=0.17, reversals=0.19, cmd_retract_frac=0.21, cath_lead_max=0.29, n_closed=0.70, cath_lead_frac50=0.32 |

## T6. Clusters 4/5/6 x phase: n and success

| phase | n 4:deep-recovery-success | n 5:shove-fight | n 6:near-buckle-fight | succ 4:deep-recovery-success | succ 5:shove-fight | succ 6:near-buckle-fight |
|---|---|---|---|---|---|---|
| free-run | 11 | 2 | 1 | 1 | 1 | 1 |
| cath-led-run | 5 | 2 | 0 | 1 | 0.50 |  |
| idle-creep | 0 | 1 | 1 |  | 0.00 | 0.00 |
| light-recovery | 906 | 167 | 53 | 0.99 | 0.93 | 0.94 |
| deep-recovery | 595 | 323 | 307 | 1 | 0.94 | 0.95 |
| fight-escaped-near | 398 | 353 | 67 | 0.90 | 0.80 | 0.90 |
| fight-escaped-far | 38 | 195 | 5 | 0.82 | 0.67 | 0.20 |
| doorstep-stall | 186 | 189 | 542 | 0.55 | 0.31 | 0.11 |
| shove-capped | 152 | 1478 | 407 | 0.05 | 0.03 | 0.06 |
| coiled-stuck | 187 | 43 | 421 | 0.20 | 0.00 | 0.06 |
| stuck-midpath | 57 | 307 | 144 | 0.00 | 0.00 | 0.00 |
| stuck-proximal | 0 | 0 | 0 |  |  |  |

## T7. Where the ambiguous episodes land: phase distribution (% of group) of successful type-5 vs failed type-4, with the contrasts

| group | n | free-run | cath-led-run | idle-creep | light-recovery | deep-recovery | fight-escaped-near | fight-escaped-far | doorstep-stall | shove-capped | coiled-stuck | stuck-midpath | stuck-proximal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| type-5 SUCCESS | 979 | 0.2 | 0.1 | 0.0 | 15.9 | 30.9 | 29 | 13.4 | 5.9 | 4.5 | 0.0 | 0.0 | 0.0 |
| type-4 FAIL | 486 | 0.0 | 0.0 | 0.0 | 1 | 0.4 | 7.8 | 1.4 | 17.1 | 29.8 | 30.7 | 11.7 | 0.0 |
| type-4 success | 2049 | 0.5 | 0.2 | 0.0 | 44 | 28.9 | 17.6 | 1.5 | 5 | 0.3 | 1.9 | 0.0 | 0.0 |
| type-5 fail | 2081 | 0.0 | 0.0 | 0.0 | 0.5 | 1 | 3.3 | 3.1 | 6.3 | 68.9 | 2.1 | 14.8 | 0.0 |
| type-6 success | 515 | 0.2 | 0.0 | 0.0 | 9.7 | 56.7 | 11.7 | 0.2 | 11.7 | 5 | 4.9 | 0.0 | 0.0 |
| type-6 fail | 1433 | 0.0 | 0.0 | 0.1 | 0.2 | 1 | 0.5 | 0.3 | 33.6 | 26.6 | 27.6 | 10 | 0.0 |

Level-2 view of the same two groups:

| group | n | top_cells |
|---|---|---|
| type-5 SUCCESS | 979 | deep-recovery/reached 31%, fight-escaped-near/reached 29%, light-recovery/reached 16%, fight-escaped-far/reached 13%, doorstep-stall/reached 6%, shove-capped/reached 4%, free-run/reached 0%, fight-escaped-near/far 0% |
| type-4 FAIL | 486 | shove-capped/far 22%, coiled-stuck/far 21%, doorstep-stall/reached 17%, stuck-midpath/far 12%, coiled-stuck/reached 9%, shove-capped/reached 8%, fight-escaped-near/far 4%, fight-escaped-near/reached 3% |

## T8. Within clusters 4/5/6: does the phase explain success beyond the cluster id?

subset n=7543, succ=0.470, H0=0.997 bits

| predictor | cats | H | IG | Gini |
|---|---|---|---|---|
| cluster id (4/5/6) | 3 | 0.819 | 0.179 | 0.381 |
| phase L1 | 11 | 0.341 | 0.658 | 0.134 |
| cluster + phase L1 | 31 | 0.314 | 0.685 | 0.124 |
| phase L2 | 22 | 0.235 | 0.764 | 0.094 |
| cluster + phase L2 | 55 | 0.212 | 0.787 | 0.084 |

5-fold CV inside clusters 4/5/6 (n=7543):

| predictor | Brier | logloss | AUC |
|---|---|---|---|
| cluster id | 0.191 | 0.568 | 0.735 |
| phase L1 | 0.067 | 0.238 | 0.957 |
| phase L2 | 0.047 | 0.166 | 0.976 |
| cluster+phase L1 | 0.063 | 0.223 | 0.965 |
| cluster+phase L2 | 0.043 | 0.155 | 0.982 |

## T9. The 66 blind-labelled snapshots: eye labels vs cluster vs phase

Consistency rule (my definition; eye primary -> categories that would show that picture). Phase side also accepts the scheme's own coil / lead>=30 flags for eye 'coil' / 'catheter-ahead'. Cluster side: clean->{0,1,2,3,4}, coil->{4,6,8}, catheter-ahead->{2,5,8,9}, stopped-short->{5..10}, near-target->{4,6}, wrong-branch->{7,10}.

phase-consistent: 56/66 = 85%; cluster-consistent: 50/66 = 76%; (wrong-branch primaries: 2, no phase category, counted as inconsistent for phase)

| primary | n | phase_ok | cluster_ok |
|---|---|---|---|
| catheter-ahead | 11 | 0.91 | 0.64 |
| clean | 24 | 0.96 | 0.92 |
| coil | 18 | 0.89 | 0.67 |
| near-target | 3 | 0.00 | 0.00 |
| stopped-short | 8 | 0.88 | 0.88 |
| wrong-branch | 2 | 0.00 | 1 |

Attribute-level agreement (eye vs scheme axes):

- eye 'coil' primary or convoluted=y (n=18) vs coil flag: agree 16/18; flag set but eye not convoluted: 1/48
- eye 'catheter-ahead' (primary/secondary, n=28) vs cath_lead_max>=30: agree 25/28; lead>=30 but eye not catheter-ahead: 11/38
- eye 'stopped-short' (n=26) vs last stall open: agree 23/26; open but eye not short: 8/40
- eye progress band vs severity (prog_max band): agree 59/66 (mapping <1/3->short, 1/3-2/3 and >2/3 -> far, at-target -> reached)

Per-episode listing:

| id | tag | cname | phase | sev | success | steps | primary | secondary | progress | convoluted | ok_phase | ok_cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | car_v3a | 10:proximal-thrash | stuck-proximal | short | False | 600 | stopped-short | - | <1/3 | n | True | True |
| 1 | tb_v2 | 1:clean-fast-success | free-run | reached | True | 64 | clean | - | at-target | n | True | True |
| 2 | car_v3 | 4:deep-recovery-success | fight-escaped-near | reached | True | 560 | coil | catheter-ahead | at-target | y | True | True |
| 3 | car_nopriv | 0:clean-success | free-run | reached | True | 75 | clean | - | at-target | n | True | True |
| 4 | v3c3 | 4:deep-recovery-success | fight-escaped-near | reached | False | 600 | coil | catheter-ahead | at-target | y | True | True |
| 5 | v3c3 | 0:clean-success | cath-led-run | reached | True | 73 | clean | - | at-target | n | True | True |
| 6 | car_v3a | 10:proximal-thrash | stuck-proximal | short | False | 600 | stopped-short | - | <1/3 | n | True | True |
| 7 | v3c2 | 10:proximal-thrash | stuck-proximal | short | False | 600 | catheter-ahead | stopped-short | <1/3 | n | True | False |
| 8 | car_v3a | 10:proximal-thrash | stuck-proximal | short | False | 600 | catheter-ahead | stopped-short | <1/3 | n | True | False |
| 9 | car_v3 | 9:shove-cap | shove-capped | far | False | 600 | catheter-ahead | stopped-short | 1/3-2/3 | n | True | True |
| 10 | car_v3 | 5:shove-fight | shove-capped | reached | True | 506 | coil | catheter-ahead | at-target | y | False | False |
| 11 | v3c3 | 2:cathled-sprint-success | cath-led-run | reached | True | 30 | stopped-short | - | >2/3 | n | False | False |
| 12 | v3c2 | 8:mega-coil | shove-capped | far | False | 600 | coil | catheter-ahead,stopped-short | 1/3-2/3 | y | True | True |
| 13 | car_v3a | 6:near-buckle-fight | coiled-stuck | reached | False | 600 | coil | - | at-target | y | True | True |
| 14 | car_v3a | 10:proximal-thrash | stuck-proximal | short | False | 600 | stopped-short | catheter-ahead | <1/3 | n | True | True |
| 15 | car_v3 | 2:cathled-sprint-success | free-run | reached | True | 68 | clean | - | at-target | n | True | True |
| 16 | car_nopriv | 0:clean-success | free-run | reached | True | 92 | clean | - | at-target | n | True | True |
| 17 | v3c3 | 5:shove-fight | shove-capped | far | False | 600 | coil | catheter-ahead,stopped-short | >2/3 | y | True | False |
| 18 | tb_repl | 5:shove-fight | doorstep-stall | reached | False | 600 | clean | - | at-target | n | False | False |
| 19 | car_v3 | 3:light-recovery-success | light-recovery | reached | True | 98 | clean | - | at-target | n | True | True |
| 20 | car_v3a | 1:clean-fast-success | free-run | reached | True | 86 | clean | - | at-target | n | True | True |
| 21 | car_v3 | 5:shove-fight | shove-capped | far | False | 600 | catheter-ahead | stopped-short | 1/3-2/3 | n | True | True |
| 22 | car_nopriv | 10:proximal-thrash | coiled-stuck | far | False | 600 | catheter-ahead | wrong-branch,stopped-short | 1/3-2/3 | n | False | False |
| 23 | v3c2 | 10:proximal-thrash | stuck-proximal | short | False | 600 | stopped-short | - | <1/3 | n | True | True |
| 24 | car_nopriv | 3:light-recovery-success | light-recovery | reached | True | 132 | clean | - | at-target | n | True | True |
| 25 | tb_v2 | 6:near-buckle-fight | shove-capped | reached | False | 600 | coil | stopped-short | 1/3-2/3 | y | True | True |
| 26 | car_nopriv | 1:clean-fast-success | free-run | reached | True | 58 | clean | - | at-target | n | True | True |
| 27 | car_v3a | 9:shove-cap | shove-capped | far | False | 600 | catheter-ahead | stopped-short | 1/3-2/3 | n | True | True |
| 28 | v3c2 | 7:midpath-thrash | stuck-midpath | short | False | 600 | stopped-short | catheter-ahead | <1/3 | n | True | True |
| 29 | v3c2 | 8:mega-coil | shove-capped | far | False | 600 | coil | catheter-ahead,stopped-short | 1/3-2/3 | y | True | True |
| 30 | car_v3 | 8:mega-coil | coiled-stuck | reached | False | 600 | coil | wrong-branch,stopped-short | 1/3-2/3 | y | True | True |
| 31 | car_nopriv | 5:shove-fight | shove-capped | reached | False | 600 | coil | catheter-ahead | at-target | y | True | False |
| 32 | tb_repl | 2:cathled-sprint-success | cath-led-run | reached | True | 68 | catheter-ahead | - | at-target | n | True | True |
| 33 | v3c2 | 6:near-buckle-fight | shove-capped | reached | False | 600 | coil | catheter-ahead | at-target | y | True | True |
| 34 | v3c2 | 1:clean-fast-success | free-run | reached | True | 47 | clean | - | at-target | n | True | True |
| 35 | v3c3 | 7:midpath-thrash | stuck-midpath | far | False | 600 | wrong-branch | catheter-ahead,stopped-short | 1/3-2/3 | n | False | True |
| 36 | v3c3 | 8:mega-coil | coiled-stuck | far | False | 600 | coil | wrong-branch,stopped-short | 1/3-2/3 | y | True | True |
| 37 | tb_v2 | 5:shove-fight | fight-escaped-near | reached | False | 600 | clean | - | at-target | n | True | False |
| 38 | car_nopriv | 0:clean-success | free-run | reached | True | 98 | clean | - | at-target | n | True | True |
| 39 | car_v3 | 7:midpath-thrash | fight-escaped-far | short | False | 600 | stopped-short | - | 1/3-2/3 | n | True | True |
| 40 | car_v3 | 0:clean-success | free-run | reached | True | 81 | clean | - | at-target | n | True | True |
| 41 | car_v3 | 7:midpath-thrash | fight-escaped-far | far | False | 600 | near-target | - | >2/3 | n | False | False |
| 42 | tb_v2 | 0:clean-success | free-run | reached | True | 79 | clean | - | at-target | n | True | True |
| 43 | car_v3a | 3:light-recovery-success | fight-escaped-near | reached | True | 422 | clean | - | at-target | n | True | True |
| 44 | tb_repl | 7:midpath-thrash | fight-escaped-far | far | False | 600 | near-target | - | >2/3 | n | False | False |
| 45 | car_v3 | 2:cathled-sprint-success | cath-led-run | reached | True | 75 | clean | - | at-target | n | True | True |
| 46 | tb_repl | 4:deep-recovery-success | fight-escaped-near | reached | True | 264 | clean | wire-hook | at-target | n | True | True |
| 47 | v3c2 | 2:cathled-sprint-success | cath-led-run | reached | True | 25 | near-target | catheter-ahead | >2/3 | n | False | False |
| 48 | tb_repl | 6:near-buckle-fight | shove-capped | reached | False | 600 | coil | catheter-ahead | at-target | y | True | True |
| 49 | tb_v2 | 4:deep-recovery-success | doorstep-stall | reached | False | 600 | coil | - | at-target | y | False | True |
| 50 | v3c3 | 7:midpath-thrash | cath-led-run | short | False | 600 | catheter-ahead | stopped-short | 1/3-2/3 | n | True | False |
| 51 | v3c2 | 3:light-recovery-success | light-recovery | reached | True | 174 | coil | - | at-target | y | True | False |
| 52 | v3c2 | 5:shove-fight | shove-capped | far | False | 600 | catheter-ahead | stopped-short | >2/3 | n | True | True |
| 53 | car_nopriv | 6:near-buckle-fight | shove-capped | far | False | 600 | coil | catheter-ahead | at-target | y | True | True |
| 54 | v3c3 | 10:proximal-thrash | stuck-proximal | short | False | 600 | stopped-short | catheter-ahead | <1/3 | n | True | True |
| 55 | car_v3a | 9:shove-cap | shove-capped | far | False | 600 | catheter-ahead | stopped-short | 1/3-2/3 | n | True | True |
| 56 | tb_repl | 0:clean-success | free-run | reached | True | 109 | clean | - | at-target | n | True | True |
| 57 | car_v3a | 7:midpath-thrash | stuck-midpath | far | False | 600 | wrong-branch | stopped-short | 1/3-2/3 | n | False | True |
| 58 | v3c3 | 1:clean-fast-success | free-run | reached | True | 86 | clean | - | at-target | n | True | True |
| 59 | tb_repl | 3:light-recovery-success | light-recovery | reached | True | 214 | clean | - | at-target | n | True | True |
| 60 | car_nopriv | 3:light-recovery-success | light-recovery | reached | True | 111 | coil | catheter-ahead | at-target | y | True | False |
| 61 | v3c2 | 0:clean-success | free-run | reached | True | 75 | coil | catheter-ahead | at-target | y | True | False |
| 62 | car_v3a | 1:clean-fast-success | free-run | reached | True | 128 | clean | - | at-target | n | True | True |
| 63 | car_v3 | 9:shove-cap | shove-capped | far | False | 600 | catheter-ahead | stopped-short | 1/3-2/3 | n | True | True |
| 64 | car_v3 | 0:clean-success | cath-led-run | reached | True | 93 | clean | - | at-target | n | True | True |
| 65 | car_nopriv | 4:deep-recovery-success | deep-recovery | reached | True | 173 | clean | - | at-target | n | True | True |
