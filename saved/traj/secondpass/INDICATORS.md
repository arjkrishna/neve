# Indicators for catheter-led advancement and coil / buckle formation

*2026-09-20 -- second pass over the trajectory atlas.  Code:
`monitoring/traj/secondpass/indicators.py` (`catheter_led`, `coil_from_features`,
`coil_from_record`).  Every number below was produced by that module or by the
scripts that calibrated it (scratchpad `secondpass/*.py`).*

Data: 152 112 complete log-derived episodes (61 `*.features.jsonl` + `.series.npz`
files: procedural v3a/v3c/v3c2/v3c3/harvest, TopBrain v1 (3 files) / repl / v2,
carotid v3 / v3a / nopriv, 39 host sessions, tb22), 138 555 of them carrying an
atlas cluster; **218 recorded replay episodes with device polylines** (carotid
v3 checkpoints 1545187 / 2289002 and the no-priv-actor checkpoints 1005189 /
1760615; 209 successes, 9 failures -- the 2289002 campaign is still writing);
66 blind eye-labelled end-of-episode snapshots (18 labelled `coil`, 28 with
`catheter-ahead` as primary or secondary label).

## 0. Headline

| question | answer |
|---|---|
| catheter-led success (catheter tip > 20 mm beyond the wire tip when the target is reached) | procedural 91-98 % of successes; carotid v3 4.4 % (explore) / 4.6 % (val); TopBrain v1 0.3 %, v2 0.8 %; **host (real patient): v1 2.4 %, v2 10.8 %, v3 2.8 %** |
| catheter-led episode (>= half of the forward progress made with the catheter >= 20 mm ahead) | procedural 74-80 % of all episodes; carotid 3.5-8.5 %; TopBrain <= 0.3 %; host <= 1.5 % |
| coil share of failures (project rule gw slack > 100 or cath slack > 50) on the host | **carotid-v3-era TopBrain-v3 checkpoints 78.6 % (n_fail = 103) vs TopBrain-v2 93.7 % (n_fail = 459)**; multi-loop (> 200 mm stored) 38.8 % vs 69.1 %; bow-only (20-50 mm stored, never above the loop proxy) 10.7 % vs 5.4 %; TopBrain-v1 39.4 %; v1bp (wire rule only) 43 % |
| eye-label validation | catheter-led: `lead_final > 20` precision 0.89 / recall 0.86 (28 labels); coil: project rule precision 0.94 / recall 0.89 (18 labels); the k-means type names score 0.69 / 0.39 and 1.00 / 0.22-0.50 respectively |
| polyline validation | on 218 recorded episodes the project rule flags exactly the 7 episodes in which the smoothed polyline closes on itself (precision 1.00, recall 1.00); slack-only grade == polyline grade in 217 / 218 episodes |

The k-means "cathled-sprint" / "shove-cap" / "mega-coil" names were never the
measurement; they are replaced below by explicit per-episode quantities that
are computable for every log-derived episode (host sessions included) and by
polyline geometry where recordings exist.

## 1. What the raw quantities mean (verified in the eve / env5 source)

* Both devices share one SOFA beam.  `fluoroscopy.tracking3d` is that beam,
  tip first; `device_trackings3d[k]` is the proximal `inserted_length[k]` mm
  of it.  Hence **the catheter tip is beyond the guidewire tip exactly when
  `ins_cath > ins_gw`, and `lead = ins_cath - ins_gw` is the arclength between
  the two tips** -- an exact quantity, not a projection proxy.
* `proj_s` (the STEP log's `proj`) is the planned-path arclength of the
  *frontier* tip (`tracking3d[0]`, the tip of whichever device is further in).
  The log's guidewire slack `ins_gw - proj_s` is therefore negative while the
  catheter leads; `cath_slack = ins_cath - proj(catheter tip)` (env5
  `_compute_cath_slack_mm`) is computed separately.
* The features jsonl carries `cath_final` and `gw_final`, so `lead_final =
  cath_final - gw_final` is available for every episode without touching the
  series (checked against the series on 295 episodes: max difference 0.1 mm,
  identical catheter-led verdicts).
* Recorded polylines are 64 nodes per device in float16.  At z = 400-600 mm
  float16 quantises to 0.25-0.5 mm, and the 64 nodes cover whatever is
  inserted (0.2 mm apart at 10 mm insertion), so raw per-node turning angles
  are noise (3000-5000 deg on a clean wire).  Every curvature-type measure in
  the module first resamples at 2 mm and smooths with a 3 mm Gaussian along
  arclength; the self-closure loop test is robust to the quantisation as is.

## 2. Catheter-led advancement

### 2.1 Indicators (per episode)

| name | definition | needs |
|---|---|---|
| `lead_final` | `ins_cath - ins_gw` at the terminal step (= `cath_final - gw_final`) | features |
| `lead_max` | max over steps (= atlas `cath_lead_max`) | features |
| `lead_frac0 / 10 / 20` | fraction of steps with lead > 0 / 10 / 20 mm | series |
| `lead_at50`, `lead_at90` | lead at the first step the frontier reaches 50 % / 90 % of the path | series |
| `adv_led10 / 20` | share of the frontier's forward progress (sum of positive d`proj_s`) gained while lead > 10 / 20 mm | series |
| **catheter-led success** | `success and lead_final > 20 mm` | features |
| **catheter-led episode** | `adv_led20 >= 0.5` (any outcome) | series |

### 2.2 Distribution per atlas type (medians; 95 960 episodes with a catheter series and a cluster)

```
=== A. medians of catheter-lead indicators by atlas cluster (episodes with catheter series) ===
                    n  succ  lead_max  lead_final  lead_frac0  lead_frac10  lead_frac20  lead_at50  lead_at90  lead_at_progmax  adv_led10  adv_led0
cl                                                                                                                                                 
clean           16627  1.00      18.1       -27.7        0.48         0.20         0.00       -1.4     -23.10           -27.70       0.23      0.48
clean-fast      26714  1.00       9.1       -22.0        0.35         0.00         0.00       -5.8     -18.80           -22.00       0.00      0.35
cathled-sprint  12051  0.99      64.8        63.3        1.00         0.90         0.78       40.6      61.10            63.30       0.96      1.00
light-rec        4965  0.98      22.4       -37.1        0.37         0.17         0.02       -8.9     -38.30           -37.20       0.23      0.42
deep-rec         1620  0.73      27.3       -16.5        0.29         0.15         0.04       -9.2     -36.25           -17.25       0.23      0.41
shove-fight      2803  0.30     124.8        90.6        0.95         0.88         0.80       33.4      13.75            53.50       0.81      0.91
near-buckle       982  0.26      60.4         5.2        0.44         0.32         0.22       -2.6     -27.60           -14.15       0.25      0.49
midpath-thrash   4480  0.14      43.2        -7.2        0.52         0.29         0.15        0.4     -30.70           -13.65       0.36      0.53
mega-coil        1923  0.07     142.1        47.0        0.87         0.72         0.60       20.7      24.10            19.90       0.70      0.88
shove-cap        5900  0.01     899.3       899.0        1.00         1.00         0.99      110.1     150.95           176.30       0.98      1.00
prox-thrash      4338  0.00      28.7        -0.4        0.44         0.14         0.03       -2.0        NaN             0.00       0.21      0.48
```

The clean types have the wire 20-30 mm ahead at the end and gain 0-23 % of their
progress with the catheter > 10 mm ahead; the cathled-sprint type has the
catheter 63 mm ahead at the end and gains 96 % of its progress catheter-first;
shove-cap ends 899 mm ahead (the insertion limit).  Between them sit the
mixed failure types (shove-fight 91 mm, mega-coil 47 mm at the end).

### 2.3 Candidate rules against the atlas types

```
=== B. agreement of candidate catheter-led rules with atlas types (all episodes with cluster) ===
                          rule  n_pos  frac_pos  recall_c2  prec_c2  recall_c9  recall_c5  prec_c2or9or5  pos_in_clean01
                 lead_final>20  23625     0.287      0.917    0.468      1.000      0.673          0.797           0.008
                 lead_final>10  26131     0.317      0.975    0.450      1.000      0.706          0.751           0.022
            lead_at_progmax>20  24247     0.294      0.917    0.456      0.997      0.653          0.774           0.008
                adv_led10>=0.5  27318     0.332      0.997    0.440      0.995      0.695          0.726           0.056
                adv_led20>=0.5  22017     0.267      0.921    0.504      0.993      0.589          0.845           0.008
              lead_frac10>=0.5  26552     0.322      0.989    0.449      1.000      0.785          0.754           0.045
              lead_frac20>=0.5  21903     0.266      0.879    0.484      1.000      0.705          0.843           0.006
lead_max>50 (atlas cath_shove)  20340     0.247      0.565    0.335      0.999      0.781          0.732           0.003
                   lead_max>20  40414     0.490      0.975    0.291      1.000      0.920          0.501           0.166
  adv_led10>=0.5 & lead_max>20  26641     0.323      0.973    0.440      0.995      0.695          0.733           0.048
                  lead_at90>10  14899     0.181      0.975    0.789      0.091      0.234          0.869           0.012
```

```
=== rule positives by cluster ===
                     n  adv_led20_ge05  adv_led10_ge05  lead_final_gt20   succ
cl                                                                           
clean           16627           0.022           0.128            0.016  1.000
clean-fast      26714           0.000           0.011            0.002  1.000
cathled-sprint  12051           0.921           0.997            0.917  0.994
light-rec        4965           0.079           0.168            0.102  0.978
deep-rec         1620           0.131           0.212            0.230  0.732
shove-fight      2803           0.589           0.695            0.673  0.298
near-buckle       982           0.193           0.257            0.455  0.259
midpath-thrash   4480           0.206           0.362            0.331  0.145
mega-coil        1923           0.494           0.631            0.541  0.070
shove-cap        5900           0.993           0.995            1.000  0.006
prox-thrash      4338           0.087           0.180            0.141  0.000
```

`lead_final > 20` recovers 92 % of cathled-sprint, 100 % of shove-cap and 67 %
of shove-fight while firing on 0.8 % of clean successes.  `adv_led10 >= 0.5`
has the best cluster-2 recall (99.7 %) but fires on 5.6 % of clean episodes:
2 824 clean / light-recovery successes gain more than half of their progress
with the catheter 10-30 mm ahead *in the first half of the path* (median
`lead_at50` = +19 mm, `lead_at90` = -12 mm, `lead_final` = -17 mm) -- the
catheter goes first through the arch and the wire takes over for the carotid.
That is benign catheter-first tracking in a wide vessel, not the sprint; the
20 mm progress-weighted rule (`adv_led20 >= 0.5`) excludes it (0.8 % of clean)
while keeping 92 % of cathled-sprint and 99 % of shove-cap.

### 2.4 Eye labels (66 snapshots, `catheter-ahead` as primary or secondary label, n = 28)

```
--- catheter-ahead label (primary or secondary) vs rules ---
              rule  n_pred  n_true  tp  fp  fn  precision  recall
     lead_final>20      27      28  24   3   4       0.89    0.86
     lead_final>10      32      28  26   6   2       0.81    0.93
      lead_final>0      35      28  27   8   1       0.77    0.96
       lead_max>50      23      28  20   3   8       0.87    0.71
       lead_max>20      48      28  26  22   2       0.54    0.93
    adv_led10>=0.5      29      28  20   9   8       0.69    0.71
cluster in {2,5,9}      16      28  11   5  17       0.69    0.39
--- catheter-ahead PRIMARY label vs rules ---
              rule  n_pred  n_true  tp  fp  fn  precision  recall
     lead_final>20      27      11  10  17   1       0.37    0.91
     lead_final>10      32      11  10  22   1       0.31    0.91
       lead_max>50      23      11   8  15   3       0.35    0.73
cluster in {2,5,9}      16      11   7   9   4       0.44    0.64
```

Misses and false alarms of `lead_final > 20` (7 of 66): three "red marginally
ahead of green" secondaries at 10-12.5 mm lead (below the threshold by
design), one wrong-branch episode whose catheter ran parallel to the path with
the wire behind (`lead_final` = -98: the label describes geometry the
insertion lengths cannot see), one coil case with lead 29 mm labelled `coil`
first, and two neat successes with the catheter 31 / 42 mm ahead that the
labeller called clean / stopped-short (`lead_max` 38-42, `adv_led10` = 1.0:
they *are* catheter-led by the insertion lengths).

### 2.5 Proposed definitions

* **catheter-led success** := `success` and `lead_final > 20 mm`.  The
  catheter is the leading device at the moment the target is reached.
  Computable from the features jsonl for every episode whose log carries the
  catheter insertion (all runs except the converted TopBrain-v1 files
  tb_v1r1 / tb_v1r2 and the converted host sessions v1bp / tbv1r1 / tb22).
  Eye-label precision 0.89 / recall 0.86; cluster-2 recall 0.92; 0.8 % of
  clean successes.
* **catheter-led episode** := `adv_led20 >= 0.5`: at least half of the
  frontier's forward progress was gained while the catheter tip was more than
  20 mm beyond the guidewire tip.  Needs the step series.  Cluster recall:
  cathled-sprint 0.92, shove-cap 0.99, shove-fight 0.59, mega-coil 0.49;
  clean 2.2 % / clean-fast 0.0 %.
* Sensitivity: the 10 mm variants (`lead_final > 10`, `adv_led10 >= 0.5`)
  add the benign arch-first episodes (carotid explore successes 4.4 % -> 12 %).

### 2.6 Shares per era (successes) and per run

| era | kind | n | succ | cath_ins | cl_succ | cl_ep_of_succ | cl_ep_all | lead_final_med_succ |
|---|---|---|---|---|---|---|---|---|
| carotid_v3 | explore | 29475 | 0.864 | 1 | 0.044 | 0.047 | 0.085 | -41.100 |
| carotid_v3 | val | 2154 | 0.772 | 1 | 0.046 | 0.029 | 0.035 | -34.400 |
| procedural | explore | 23053 | 0.464 | 1 | 0.910 | 0.897 | 0.742 | 61.700 |
| procedural | val | 3133 | 0.472 | 1 | 0.980 | 0.986 | 0.798 | 77.400 |
| topbrain_v1 | explore | 61187 | 0.979 | 0.130 | 0.003 | 0.002 | 0.003 | -24.500 |
| topbrain_v1 | val | 2622 | 0.953 | 0.150 | 0.000 | 0.000 | 0.000 | -19.200 |
| topbrain_v2 | explore | 25781 | 0.997 | 1 | 0.008 | 0.001 | 0.002 | -21.300 |
| topbrain_v2 | val | 882 | 0.983 | 1 | 0.006 | 0.000 | 0.000 | -17.700 |

`cath_ins`: fraction of episodes whose log carries the catheter insertion
(TopBrain-v1 = tb_repl only).  `cl_succ` = catheter-led successes / successes;
`cl_ep_of_succ` = catheter-led episodes / successes; `cl_ep_all` = catheter-led
episodes / all episodes.

| tag | kind | n | succ | cath_ins | cl_succ | cl_ep_of_succ | lead_final_med_succ |
|---|---|---|---|---|---|---|---|
| car_nopriv | explore | 19350 | 0.909 | 1 | 0.026 | 0.013 | -42.500 |
| car_nopriv | val | 1174 | 0.761 | 1 | 0.056 | 0.003 | -35.200 |
| car_v3 | explore | 8628 | 0.824 | 1 | 0.073 | 0.109 | -36.500 |
| car_v3 | val | 686 | 0.835 | 1 | 0.028 | 0.066 | -33.400 |
| car_v3a | explore | 1497 | 0.516 | 1 | 0.181 | 0.245 | -29.600 |
| car_v3a | val | 294 | 0.667 | 1 | 0.051 | 0.041 | -31.100 |
| harvest_v3c | explore | 480 | 0.054 | 1 | 0.615 | 0.423 | 34.800 |
| tb_repl | explore | 7879 | 0.966 | 1 | 0.020 | 0.017 | -24.500 |
| tb_repl | val | 392 | 0.911 | 1 | 0.003 | 0.000 | -19.200 |
| tb_v1r1 | explore | 5822 | 0.987 | 0.000 |  |  |  |
| tb_v1r1 | val | 294 | 0.905 | 0.000 |  |  |  |
| tb_v1r2 | explore | 47486 | 0.981 | 0.000 |  |  |  |
| tb_v1r2 | val | 1936 | 0.970 | 0.000 |  |  |  |
| tb_v2 | explore | 25781 | 0.997 | 1 | 0.008 | 0.001 | -21.300 |
| tb_v2 | val | 882 | 0.983 | 1 | 0.006 | 0.000 | -17.700 |
| v3a | explore | 9975 | 0.471 | 1 | 0.991 | 0.989 | 103.600 |
| v3a | val | 1369 | 0.502 | 1 | 1 | 1 | 118.500 |
| v3c | explore | 1209 | 0.520 | 1 | 0.887 | 0.884 | 55.300 |
| v3c | val | 196 | 0.459 | 1 | 0.944 | 1 | 65.900 |
| v3c2 | explore | 9495 | 0.466 | 1 | 0.820 | 0.803 | 28.800 |
| v3c2 | val | 1274 | 0.448 | 1 | 0.956 | 0.965 | 34.400 |
| v3c3 | explore | 1894 | 0.479 | 1 | 0.950 | 0.904 | 57 |
| v3c3 | val | 294 | 0.442 | 1 | 1 | 1 | 59.600 |

Answer to question (i): the catheter-led sprint is a product of the procedural
anatomies (91-99 % of v3a successes, 82-96 % of v3c2, 89-100 % of v3c/v3c3).
On the realistic anatomies it collapses to a few per cent (carotid v3 2.6-7.3 %
explore, 2.8-5.6 % val; car_v3a 18 % explore in its weak early phase) and to
< 1 % on TopBrain v2.  On the real patient it stays small (v1 2.4 %, v3 2.8 %)
except for the TopBrain-v2 checkpoints, 10.8 % of whose successes end with the
catheter > 20 mm ahead (up to 34 % for ck1751101) -- but almost none of those
are catheter-led *episodes* (0.7 %): the v2 policy finishes with a catheter
shove, it does not navigate catheter-first.

## 3. Coil / buckle formation

### 3.1 The existing slack rule against the eye labels (18 `coil` labels of 66)

```
--- coil label vs slack rules (episode max) ---
                                   rule  n_pred  n_true  tp  fp  fn  precision  recall
   gw_slack_max>100 | cath_slack_max>50      17      18  16   1   2       0.94    0.89
gw_slack_max>100 | cath_slack_max>1e+09      10      18   9   1   9       0.90    0.50
 gw_slack_max>1e+09 | cath_slack_max>50      14      18  14   0   4       1.00    0.78
    gw_slack_max>50 | cath_slack_max>50      17      18  16   1   2       0.94    0.89
    gw_slack_max>50 | cath_slack_max>30      19      18  17   2   1       0.89    0.94
    gw_slack_max>30 | cath_slack_max>30      22      18  17   5   1       0.77    0.94
    gw_slack_max>80 | cath_slack_max>40      18      18  17   1   1       0.94    0.94
  gw_slack_max>100 | cath_slack_max>100      15      18  14   1   4       0.93    0.78
   gw_slack_max>150 | cath_slack_max>75      16      18  15   1   3       0.94    0.83
  gw_slack_max>200 | cath_slack_max>100      14      18  13   1   5       0.93    0.72
                           fold_max>=10       5      18   5   0  13       1.00    0.28
                            fold_max>=4      38      18  17  21   1       0.45    0.94
                             cluster==8       4      18   4   0  14       1.00    0.22
                       cluster in {6,8}       9      18   9   0   9       1.00    0.50
```

```
--- coil label vs FINAL-step slack (the image is the last frame) ---
                                    rule  n_pred  n_true  tp  fp  fn  precision  recall
gw_slack_final>100 | cath_slack_final>50      17      18  16   1   2       0.94    0.89
 gw_slack_final>50 | cath_slack_final>50      17      18  16   1   2       0.94    0.89
 gw_slack_final>50 | cath_slack_final>30      17      18  16   1   2       0.94    0.89
 gw_slack_final>30 | cath_slack_final>30      18      18  16   2   2       0.89    0.89
 gw_slack_final>30 | cath_slack_final>20      19      18  16   3   2       0.84    0.89
 gw_slack_final>20 | cath_slack_final>20      19      18  16   3   2       0.84    0.89
 gw_slack_final>40 | cath_slack_final>25      18      18  16   2   2       0.89    0.89
 gw_slack_final>60 | cath_slack_final>40      17      18  16   1   2       0.94    0.89
                     cath_slack_final>20      15      18  14   1   4       0.93    0.78
                     cath_slack_final>30      14      18  14   0   4       1.00    0.78
                     cath_slack_final>40      14      18  14   0   4       1.00    0.78
                     cath_slack_final>50      14      18  14   0   4       1.00    0.78
                       gw_slack_final>20      11      18   8   3  10       0.73    0.44
                       gw_slack_final>30      10      18   8   2  10       0.80    0.44
                       gw_slack_final>50       7      18   6   1  12       0.86    0.33
                      gw_slack_final>100       7      18   6   1  12       0.86    0.33
   max(slack_final, cath_slack_final)>30      18      18  16   2   2       0.89    0.89
```

The project rule (gw slack > 100 or cath slack > 50, episode max) scores
precision 0.94 / recall 0.89.  Its one false positive is a wrong-branch
episode (id 22: the wire in a side vessel accumulates 250 mm of "slack"
because the frontier projection stalls while insertion grows -- slack
conflates coils with off-path excursions; polylines or `off_final` separate
them).  Its two misses are the two smallest coils: a single wire loop just
below the target with 28 mm of slack (id 49, "wire loops once below target")
and a small catheter curl with 41 mm of catheter slack (id 10).  Lowering the
thresholds to gw > 50 / cath > 30 recovers one of them at the cost of a second
false positive (a 30 mm catheter slack in a wrong-branch episode).  Cluster
membership (mega-coil, or mega-coil + near-buckle) recalls only 22-50 % of the
labelled coils: the labelled coils live in six different clusters.

### 3.2 Polyline geometry (`coil_from_record`)

Per step and per device: resample the 64-node polyline at 2 mm, smooth with a
3 mm Gaussian along arclength, then

* **closure** = min distance between two samples whose arclength separation
  exceeds 25 mm; **loop** := closure < 5 mm.  Control: the planned paths of
  the 15 carotid anatomies never come closer than **10.2 mm** to themselves at
  that separation (the tight siphon of casem030...003L), clean wires sit at
  13-25 mm, bows at 4-16 mm, loops at 0.1-0.8 mm.  Rendered frames of the
  243 mm mega-coil, the 43 mm siphon bow (closure 6.0), the 24 mm near-target
  buckle (12.6) and a 26 mm bow (7.9) agree with the flags.
* **n_loops** = connected components of the closure pairs in the (i, j)
  sample plane: 1 for a single loop, 2-3 for a double winding, 7-9 for the
  243 mm coil.
* **turning excess** = total turning of the smoothed device polyline minus
  that of the equally smoothed planned path up to the frontier projection.
  It is *not* usable as a detector: clean wires cut the centerline's corners,
  so the floor is -205..+22 deg (p10-p90 of "none" steps); bows sit at +81
  (median), single loops +314, multi-loops +501.  Reported for information.
* **min bend radius** of the smoothed curve is biased upward by the smoothing
  and the centerlines themselves have 2-3 mm radii; not used for grading.
* **grade** per step = max over devices of `multi-loop` (n_loops >= 2) >
  `loop` > `bow` (no loop, >= 20 mm stored in the device: wire slack or
  catheter slack) > `none`.

```
=== anatomy control: planned-path self-closure (mm, 2 mm resample, sigma 3) per mesh ===
mesh
casem030righttopcowmr003L    10.2
casew025righttopcowmr016L    11.2
casem030righttopcowmr016     14.8
casew007lefttopcowmr017      15.9
casew013righttopcowmr013L    16.2
casem030righttopcowmr013L    16.4
casew007lefttopcowmr016L     17.4
casew033lefttopcowmr011L     19.3
casew017righttopcowmr011     19.5
casew013righttopcowmr011     22.1
casew007lefttopcowmr013L     22.1
casew013righttopcowmr021L    22.2
casew025righttopcowmr021L    22.3
casem030righttopcowmr016L    22.3
casew007lefttopcowmr021L     22.4
```

```
=== episode grades (polyline) x outcome ===
success     False  True 
grade_max               
bow             5     39
loop            0      5
multi-loop      2      0
none            2    165

=== polyline grade_max vs slack-only grade_max (coil_from_features on the same episode) ===
f_grade_max  bow  loop  multi-loop  none
grade_max                               
bow           44     0           0     0
loop           0     5           0     0
multi-loop     0     1           1     0
none           0     0           0   167

=== polyline grade_final vs slack-only grade_final ===
f_grade_final  bow  loop  multi-loop  none
grade_final                               
bow             28     2           0     0
loop             0     4           0     0
multi-loop       0     0           1     0
none             0     0           0   183
```

### 3.3 Slack against geometry

Per step (22.6k recorded steps), probability that the device polyline holds a
closed loop given the length stored in it:

```
=== step level: P(loop | stored-length bin), gw and cath separately, n steps=23990 ===
                           n  loop_gw  closure_gw_med  n_loops_gw_med
gbin                                                                 
(-1000000000.0, 0.0]   16148    0.000          24.222             0.0
(0.0, 10.0]             4589    0.000          19.176             0.0
(10.0, 20.0]            1736    0.000          16.305             0.0
(20.0, 30.0]             561    0.004          15.922             0.0
(30.0, 40.0]             231    0.017          10.164             0.0
(40.0, 50.0]              85    0.318           6.438             0.0
(50.0, 75.0]             168    0.750           0.782             2.0
(75.0, 100.0]            165    0.988           0.707             2.0
(100.0, 150.0]           172    1.000           0.568             6.0
(150.0, 200.0]           111    1.000           0.516             7.0
(200.0, 1000000000.0]     24    1.000           0.501             7.5
                           n  loop_cath  closure_cath_med  n_loops_cath_med
cbin                                                                       
(-1000000000.0, 0.0]   12926      0.000            24.450               0.0
(0.0, 10.0]             7848      0.000            23.223               0.0
(10.0, 20.0]            2216      0.000            20.111               0.0
(20.0, 30.0]             271      0.000            17.498               0.0
(30.0, 40.0]             104      0.000            15.915               0.0
(40.0, 50.0]              85      0.035             8.678               0.0
(50.0, 75.0]              78      0.359             6.785               0.0
(75.0, 100.0]             50      0.820             2.031               1.0
(100.0, 150.0]            61      1.000             0.547               1.0
(150.0, 200.0]            16      1.000             0.829               3.0
(200.0, 1000000000.0]    335      1.000             0.480               7.0

slack at loop steps: gw [49.6, 74.4, 97.8]  cath [67.3, 130.8, 244.2]
first-loop stored length per episode: gw slack [45.6, 37.0, 66.2, 49.4, 82.2, -239.7, 22.7]  cath [42.3, 63.2, 62.7, 44.1, 22.8, 55.4, 45.4]
```

Per episode (max over steps), slack-only rules against "a loop existed at some step":

```
=== slack-only rules vs polyline LOOP (any step), episode level, n=218 ===
                              rule  n_pred  n_true  tp  fp  fn  precision  recall
 slack_max>100 | cath_slack_max>50       7       7   7   0   0       1.00    1.00
  slack_max>50 | cath_slack_max>50       7       7   7   0   0       1.00    1.00
 slack_max>50 | cath_slack_max>100       6       7   6   0   1       1.00    0.86
  slack_max>50 | cath_slack_max>30      14       7   7   7   0       0.50    1.00
  slack_max>30 | cath_slack_max>30      25       7   7  18   0       0.28    1.00
slack_max>100 | cath_slack_max>100       2       7   2   0   5       1.00    0.29
  slack_max>40 | cath_slack_max>80       8       7   6   2   1       0.75    0.86
 slack_max>60 | cath_slack_max>100       5       7   5   0   2       1.00    0.71
 slack_max>50 | cath_slack_max>150       5       7   5   0   2       1.00    0.71
            slack_max>50 (gw only)       5       6   5   0   1       1.00    0.83
           slack_max>100 (gw only)       1       6   1   0   5       1.00    0.17
     cath_slack_max>50 (cath only)       7       4   4   3   0       0.57    1.00
    cath_slack_max>100 (cath only)       2       4   2   0   2       1.00    0.50
             f_loop_proxy (module)       7       7   7   0   0       1.00    1.00
             f_coil (project rule)       7       7   7   0   0       1.00    1.00
```

```
=== episodes with a closed loop at any step ===
                              file        ck   T  success  prog_max  slack_max  cath_slack_max  loop_any_gw  loop_any_cath  loop_steps  first_loop_step  slack_at_first_loop  cath_slack_at_first_loop  n_loops_max  closure_min_gw  closure_min_cath  turn_excess_max_deg  grade_max grade_final f_grade_max  f_coil  near_target_buckle  near_target_loop_frac
s0039_casew007lefttopcowmr016L_pid ck2289002 109     True       1.0       59.0            63.9         True          False           2               93                 45.6                      42.3            1             3.6               5.4                737.6       loop         bow        loop    True                  34                    0.1
s0047_casew025righttopcowmr016L_pi ck2289002 128     True       1.0       37.0            63.2         True           True           1              127                 37.0                      63.2            1             3.8               3.9                813.2       loop        loop        loop    True                  29                    0.0
s0134_casew007lefttopcowmr013L_pid ck2289002 212     True       1.0       66.8            63.4         True           True           3              209                 66.2                      62.7            1             4.4               4.4                707.6       loop        loop        loop    True                  30                    0.1
s0039_casew007lefttopcowmr016L_pid ck2289002 111     True       1.0       65.8            68.1         True          False           1               94                 49.4                      44.1            1             4.5               5.9                878.8       loop         bow        loop    True                  38                    0.0
s0055_casew013righttopcowmr021L_pi ck1005189 601    False       0.9       98.5            56.7         True          False         174              427                 82.2                      22.8            3             0.1               8.6                793.8 multi-loop        loop        loop    True                 300                    0.6
s0161_casew007lefttopcowmr013L_pid ck1005189 232     True       1.0       21.2           107.6        False           True          32              200               -239.7                      55.4            1            13.8               0.3                 85.2       loop        loop        loop    True                 104                    0.3
s0161_casew007lefttopcowmr013L_pid ck1760615 601    False       1.0      243.3           296.5         True           True         448              138                 22.7                      45.4            9             0.1               0.1                864.4 multi-loop  multi-loop  multi-loop    True                 505                    0.9
```

```
=== turning excess (smoothed, vs planned path) by grade at step level ===
                n  exc_med  exc_p10  exc_p90
grade                                       
bow          1051     83.0    -67.0    246.0
loop          171    314.0   -708.0    553.0
multi-loop    490    501.0    287.0    634.0
none        22278   -100.0   -209.0     19.0
```

Reading: nothing below 30 mm of stored length is a loop; a wire storing 50-75
mm is looped 75 % of the time and 75-100 mm 99 %; the catheter needs more
(50-75 mm 36 %, 75-100 mm 82 %) because its loops are larger (aorta / arch).
Wire loops first close at 37-82 mm of wire slack and catheter loops at 45-63
mm of catheter slack (the seven first-loop steps; the mega-coil closed its
first loop at 23 mm wire / 45 mm catheter).  Above 100 mm every step is a loop, and the loop count
grows with the stored length (median 6-7 components above 100 mm of wire
slack).  At the episode level the project rule and the 50 / 50 proxy both
find exactly the seven looped episodes.

### 3.4 The near-target buckle

Recorded steps at >= 90 % progress with >= 20 mm stored (18 episodes, 1 691
steps): 39 % hold a closed loop, but the loops are concentrated in two
episodes (the buckle fight s0055_...021L, 98 mm slack, loops in 55 % of its
near-target steps with up to three windings, and the 243 mm mega-coil).  The
other 16 episodes store 20-67 mm at the last bend as a **bow** (closure
3.6-16 mm; a loop in at most 4 % of their near-target steps): the "near-target buckle fight" is a bow of 20-50 mm at
the last bend, and it becomes a loop only when the policy keeps feeding past
~40-50 mm.  The eye-labelled "wire loops once below target" (28 mm of slack)
shows that at the target a loop can close at the wire's bending limit, below
the 50 mm proxy; that is the one regime where only polylines settle bow vs
loop.

```
=== near-target (prog>=0.9) steps with >=20 mm stored: loop fraction, closure ===
n steps 1691 episodes 18 loop frac 0.391 stored med 39.6
                                      n  stored_max  loop  closure_min       grade
file                                                                              
s0002_casew025righttopcowmr016L_pi   67       36.36  0.00         7.90         bow
s0008_casew013righttopcowmr013L_pi   69       34.64  0.00        15.18         bow
s0023_casew013righttopcowmr013L_pi   25       31.72  0.00        15.39         bow
s0039_casew007lefttopcowmr016L_pid   93       65.75  0.03         3.61         bow
s0047_casew025righttopcowmr016L_pi   44       37.03  0.02         3.81         bow
s0048_casew025righttopcowmr021L_pi    1       20.11  0.00        16.73         bow
s0055_casem030righttopcowmr003L_pi  223       42.83  0.00         5.46         bow
s0055_casew013righttopcowmr021L_pi  316       98.50  0.55         0.11         bow
s0093_casew007lefttopcowmr017_pid1    5       22.60  0.00        15.52         bow
s0093_casew007lefttopcowmr017_pid2    3       23.19  0.00        16.16         bow
s0115_casew013righttopcowmr021L_pi    4      -37.01  0.00        20.19         bow
s0134_casew007lefttopcowmr013L_pid   75       66.81  0.04         4.42         bow
s0139_casew013righttopcowmr021L_pi   44       27.12  0.00        11.88         bow
s0148_casew025righttopcowmr016L_pi   82       40.08  0.00         8.89         bow
s0154_casew007lefttopcowmr021L_pid   24       29.34  0.00        11.98         bow
s0161_casem030righttopcowmr016_pid    2       22.15  0.00        15.76         bow
s0161_casew007lefttopcowmr013L_pid  609      243.33  0.74         0.09  multi-loop
s0167_casem030righttopcowmr013L_pi    5       23.46  0.00        14.66         bow
```

### 3.5 Graded severity and the slack-only proxy

| grade | polylines (`coil_from_record`) | slack-only proxy (`coil_from_features.grade_*`) |
|---|---|---|
| none | no closure, < 20 mm stored | max(gw slack, cath slack) < 20 mm |
| bow / kink | no closure, >= 20 mm stored (the buckle) | 20-50 mm |
| loop | closure < 5 mm at >= 25 mm separation | gw > 50 mm or cath > 50 mm |
| multi-loop | >= 2 closure components | > 200 mm (eye-labelled double loops carry 230-260 mm, "multiple" > 400) |

Agreement of the slack-only grade with the polyline grade on the recorded
episodes: 217 / 218 (the one disagreement is the buckle fight, three windings
at 98 mm, graded `loop` by slack).  `coil_from_features` also returns the
project rule (`coil`, with `coil_gw` / `coil_cath`), the recall-tuned
`coil_loose` (gw > 50 or cath > 30), the proxy `loop_proxy` (gw > 50 or cath >
50), the last-step verdict `coil_final`, and `near_target_buckle` (failure at
>= 75 % of the path ending with >= 20 mm stored) with its grade.

### 3.6 Shares per era, per run and per host family

Failures only (`*_of_fail`), all episodes for the has-cath columns.

| era | kind | n_fail | coil_of_fail | coil_gw_of_fail | coil_cath_of_fail | loop_proxy_of_fail | multi_of_fail | bow_only_of_fail | ntb_of_fail | shove_of_fail | coil_of_succ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| carotid_v3 | explore | 4004 | 0.182 | 0.094 | 0.160 | 0.202 | 0.108 | 0.151 | 0.146 | 0.489 | 0.023 |
| carotid_v3 | val | 491 | 0.236 | 0.114 | 0.210 | 0.248 | 0.165 | 0.126 | 0.218 | 0.344 | 0.019 |
| procedural | explore | 12364 | 0.182 | 0.107 | 0.174 | 0.190 | 0.161 | 0.112 | 0.104 | 0.703 | 0.021 |
| procedural | val | 1655 | 0.127 | 0.067 | 0.120 | 0.127 | 0.108 | 0.005 | 0.088 | 0.577 | 0.009 |
| topbrain_v1 | explore | 1263 | 0.389 | 0.275 | 0.258 | 0.485 | 0.221 | 0.317 | 0.526 | 0.089 | 0.006 |
| topbrain_v1 | val | 122 | 0.131 | 0.074 | 0.123 | 0.148 | 0.098 | 0.082 | 0.180 | 0.016 | 0.005 |
| topbrain_v2 | explore | 80 | 0.250 | 0.200 | 0.212 | 0.300 | 0.075 | 0.125 | 0.225 | 0.225 | 0.003 |
| topbrain_v2 | val | 15 | 0.467 | 0.267 | 0.467 | 0.467 | 0.267 | 0.200 | 0.600 | 0.333 | 0.006 |

| tag | kind | n_fail | coil_of_fail | coil_gw_of_fail | coil_cath_of_fail | multi_of_fail | bow_only_of_fail | ntb_of_fail | shove_of_fail |
|---|---|---|---|---|---|---|---|---|---|
| car_nopriv | explore | 1761 | 0.211 | 0.104 | 0.186 | 0.119 | 0.096 | 0.174 | 0.672 |
| car_nopriv | val | 280 | 0.232 | 0.089 | 0.214 | 0.143 | 0.107 | 0.204 | 0.429 |
| car_v3 | explore | 1519 | 0.147 | 0.064 | 0.134 | 0.087 | 0.184 | 0.132 | 0.286 |
| car_v3 | val | 113 | 0.221 | 0.150 | 0.195 | 0.177 | 0.239 | 0.204 | 0.195 |
| car_v3a | explore | 724 | 0.186 | 0.133 | 0.151 | 0.127 | 0.215 | 0.105 | 0.471 |
| car_v3a | val | 98 | 0.265 | 0.143 | 0.214 | 0.214 | 0.051 | 0.276 | 0.276 |
| harvest_v3c | explore | 454 | 0.031 | 0.004 | 0.026 | 0.000 | 0.432 | 0.018 | 0.377 |
| tb_repl | explore | 269 | 0.223 | 0.086 | 0.219 | 0.100 | 0.372 | 0.164 | 0.416 |
| tb_repl | val | 35 | 0.114 | 0.086 | 0.114 | 0.086 | 0.057 | 0.114 | 0.057 |
| tb_v1r1 | explore | 75 | 0.253 | 0.120 | 0.227 | 0.067 | 0.267 | 0.360 |  |
| tb_v1r1 | val | 28 | 0.036 | 0.036 | 0.036 | 0.036 | 0.143 | 0.214 |  |
| tb_v1r2 | explore | 919 | 0.448 | 0.343 | 0.272 | 0.269 | 0.306 | 0.645 |  |
| tb_v1r2 | val | 59 | 0.186 | 0.085 | 0.169 | 0.136 | 0.068 | 0.203 |  |
| tb_v2 | explore | 80 | 0.250 | 0.200 | 0.212 | 0.075 | 0.125 | 0.225 | 0.225 |
| tb_v2 | val | 15 | 0.467 | 0.267 | 0.467 | 0.267 | 0.200 | 0.600 | 0.333 |
| v3a | explore | 5276 | 0.014 | 0.014 |  | 0.013 | 0.005 | 0.012 | 0.993 |
| v3a | val | 682 | 0.018 | 0.018 |  | 0.018 | 0.001 | 0.018 | 0.997 |
| v3c | explore | 580 | 0.621 | 0.526 | 0.612 | 0.510 | 0.171 | 0.343 | 0.750 |
| v3c | val | 106 | 0.226 | 0.226 | 0.226 | 0.226 | 0.000 | 0.217 | 0.670 |
| v3c2 | explore | 5068 | 0.199 | 0.169 | 0.195 | 0.177 | 0.191 | 0.121 | 0.382 |
| v3c2 | val | 703 | 0.161 | 0.102 | 0.161 | 0.114 | 0.010 | 0.100 | 0.148 |
| v3c3 | explore | 986 | 0.810 | 0.088 | 0.805 | 0.729 | 0.097 | 0.406 | 0.920 |
| v3c3 | val | 164 | 0.378 | 0.018 | 0.378 | 0.378 | 0.000 | 0.250 | 0.610 |

`coil_of_fail` project rule; `coil_gw` / `coil_cath` the device that trips it;
`multi` > 200 mm stored; `bow_only` 20-50 mm stored and never above the loop
proxy; `ntb` near-target buckle (failure at >= 75 % of the path ending with
>= 20 mm stored); `shove` = atlas `cath_lead_max > 50`.

## 4. The 39 host sessions (real patient)

Host features are log-derived: the guidewire tip only, no device bodies.  What
is computable there:

| indicator | host_topbrain_v1/v2/v3, host_p2_teacher_v1bp (33 + 2 sessions) | host_v1bp_*, tb22 (4 sessions) | host_tbv1r1_* (2 sessions) |
|---|---|---|---|
| `lead_final`, `lead_max`, catheter-led success | yes | no (converted logs, no catheter insertion) | no |
| `adv_led20`, catheter-led episode | yes (series) | no | no |
| wire-slack coil rule, bow / loop / multi-loop proxy (wire) | yes | yes | yes |
| catheter-slack rule | yes | no | yes (catheter slack was logged, insertion was not) |
| polyline grade, n_loops, closure | **no** (needs a recording; none exists for the host) | no | no |

| fam | n | sess | succ | cath_ins | cath_slack | cl_succ | cl_ep_of_succ | cl_ep_all | lead_final_med_succ | n_fail | coil_of_fail | coil_gw_of_fail | coil_cath_of_fail | loop_proxy_of_fail | multi_of_fail | bow_only_of_fail | ntb_of_fail | shove_of_fail | coil_of_succ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| host:p2_teacher_v1bp_ck0 | 196 | 2 | 0.046 | 1 | 1 | 0.000 | 0.000 | 0.000 | -14.500 | 187 | 0.000 | 0.000 | 0.000 | 0.032 | 0.000 | 0.118 | 0.005 | 0.000 | 0.000 |
| host:tb22_v1bp | 204 | 1 | 0.745 | 0.000 | 0.000 |  |  |  |  | 52 | 0.327 | 0.327 |  | 0.365 | 0.269 | 0.000 | 0.365 |  | 0.013 |
| host:tbv1r1 | 164 | 2 | 0.543 | 0.000 | 1 |  |  |  |  | 75 | 0.800 | 0.547 | 0.787 | 0.800 | 0.613 | 0.120 | 0.293 |  | 0.124 |
| host:topbrain_v1 | 1094 | 14 | 0.570 | 1 | 1 | 0.024 | 0.000 | 0.002 | -41.200 | 470 | 0.394 | 0.338 | 0.317 | 0.428 | 0.313 | 0.168 | 0.228 | 0.226 | 0.062 |
| host:topbrain_v2 | 1164 | 12 | 0.606 | 1 | 1 | 0.108 | 0.007 | 0.015 | -9.100 | 459 | 0.937 | 0.617 | 0.935 | 0.943 | 0.691 | 0.054 | 0.512 | 0.891 | 0.193 |
| host:topbrain_v3 | 757 | 8 | 0.864 | 1 | 1 | 0.028 | 0.000 | 0.008 | -14.800 | 103 | 0.786 | 0.408 | 0.767 | 0.864 | 0.388 | 0.107 | 0.524 | 0.660 | 0.076 |
| host:v1bp | 246 | 3 | 0.577 | 0.000 | 0.000 |  |  |  |  | 104 | 0.433 | 0.433 |  | 0.462 | 0.298 | 0.163 | 0.240 |  | 0.007 |

Per checkpoint (the families are the 39 sessions; the two tiny ck1507175 /
ck253934 / ck756995 restarts with 2-13 episodes are listed but carry no
failures):

| fam | tag | n | succ | cl_succ | cl_ep_all | n_fail | coil_fail | coil_gw_fail | loop_proxy_fail | multi_fail | shove_fail | ntb_fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p2_teacher_v1bp_ck0 | p2_teacher_v1bp_ck0_20260901_191552 | 98 | 0.092 | 0.000 | 0.000 | 89 | 0.000 | 0.000 | 0.011 | 0.000 | 0.000 | 0.000 |
| p2_teacher_v1bp_ck0 | p2_teacher_v1bp_ck0_20260901_215303 | 98 | 0.000 |  | 0.000 | 98 | 0.000 | 0.000 | 0.051 | 0.000 | 0.000 | 0.010 |
| tb22_v1bp | tb22_v1bp_ck2002292 | 204 | 0.745 |  |  | 52 | 0.327 | 0.327 | 0.365 | 0.269 |  | 0.365 |
| tbv1r1 | tbv1r1_ck256370 | 82 | 0.451 |  |  | 45 | 0.711 | 0.333 | 0.711 | 0.444 |  | 0.244 |
| tbv1r1 | tbv1r1_ck505230 | 82 | 0.634 |  |  | 30 | 0.933 | 0.867 | 0.933 | 0.867 |  | 0.367 |
| topbrain_v1 | topbrain_v1_best_checkpoint_20260831_152449 | 98 | 0.592 | 0.052 | 0.000 | 40 | 0.275 | 0.150 | 0.300 | 0.100 | 0.100 | 0.100 |
| topbrain_v1 | topbrain_v1_ck0_20260831_160901 | 98 | 0.020 | 0.000 | 0.000 | 96 | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 |
| topbrain_v1 | topbrain_v1_ck0_20260901_203431 | 98 | 0.010 | 0.000 | 0.000 | 97 | 0.000 | 0.000 | 0.041 | 0.000 | 0.000 | 0.000 |
| topbrain_v1 | topbrain_v1_ck1010053_20260831_112349 | 98 | 0.755 | 0.000 | 0.010 | 24 | 0.792 | 0.792 | 0.833 | 0.708 | 0.708 | 0.125 |
| topbrain_v1 | topbrain_v1_ck1251464_20260831_115629 | 98 | 0.357 | 0.029 | 0.010 | 63 | 0.905 | 0.635 | 0.905 | 0.683 | 0.635 | 0.524 |
| topbrain_v1 | topbrain_v1_ck1507175_20260831_041305 | 4 | 1 | 0.250 | 0.000 | 0 |  |  |  |  |  |  |
| topbrain_v1 | topbrain_v1_ck1507175_20260831_041711 | 96 | 0.656 | 0.063 | 0.000 | 33 | 0.970 | 0.970 | 0.970 | 0.879 | 0.545 | 0.636 |
| topbrain_v1 | topbrain_v1_ck1507175_20260831_070513 | 98 | 0.684 | 0.045 | 0.000 | 31 | 0.968 | 0.903 | 0.968 | 0.871 | 0.613 | 0.548 |
| topbrain_v1 | topbrain_v1_ck253934_20260831_070127 | 2 | 1 | 0.000 | 0.000 | 0 |  |  |  |  |  |  |
| topbrain_v1 | topbrain_v1_ck253934_20260831_094716 | 98 | 0.571 | 0.054 | 0.000 | 42 | 0.214 | 0.214 | 0.238 | 0.048 | 0.095 | 0.071 |
| topbrain_v1 | topbrain_v1_ck506469_20260831_105211 | 98 | 0.765 | 0.000 | 0.000 | 23 | 0.348 | 0.348 | 0.652 | 0.261 | 0.043 | 0.348 |
| topbrain_v1 | topbrain_v1_ck756995_20260831_070128 | 13 | 1 | 0.000 | 0.000 | 0 |  |  |  |  |  |  |
| topbrain_v1 | topbrain_v1_ck756995_20260831_081835 | 97 | 0.907 | 0.000 | 0.000 | 9 | 0.889 | 0.778 | 0.889 | 0.889 | 0.111 | 0.889 |
| topbrain_v1 | topbrain_v1_ck756995_20260831_091525 | 98 | 0.878 | 0.000 | 0.000 | 12 | 0.917 | 0.833 | 1 | 0.917 | 0.167 | 0.833 |
| topbrain_v2 | topbrain_v2_best_checkpoint_20260908_073801 | 98 | 0.745 | 0.041 | 0.000 | 25 | 0.840 | 0.040 | 0.840 | 0.280 | 0.560 | 0.600 |
| topbrain_v2 | topbrain_v2_ck1002078_20260907_214914 | 98 | 0.490 | 0.104 | 0.000 | 50 | 0.800 | 0.140 | 0.820 | 0.380 | 0.620 | 0.480 |
| topbrain_v2 | topbrain_v2_ck1255725_20260908_000610 | 98 | 0.673 | 0.152 | 0.051 | 32 | 1 | 0.594 | 1 | 0.656 | 1 | 0.625 |
| topbrain_v2 | topbrain_v2_ck1501133_20260907_080200 | 97 | 0.722 | 0.071 | 0.041 | 27 | 1 | 0.889 | 1 | 0.889 | 1 | 0.481 |
| topbrain_v2 | topbrain_v2_ck1501133_20260907_110849 | 87 | 0.552 | 0.042 | 0.011 | 39 | 1 | 0.718 | 1 | 0.897 | 1 | 0.436 |
| topbrain_v2 | topbrain_v2_ck1501133_20260908_015543 | 98 | 0.367 | 0.194 | 0.020 | 62 | 1 | 0.742 | 1 | 0.855 | 0.984 | 0.484 |
| topbrain_v2 | topbrain_v2_ck1751101_20260908_041631 | 98 | 0.724 | 0.338 | 0.051 | 27 | 0.926 | 0.593 | 0.926 | 0.556 | 0.889 | 0.704 |
| topbrain_v2 | topbrain_v2_ck2001807_20260908_054146 | 98 | 0.622 | 0.049 | 0.000 | 37 | 0.811 | 0.270 | 0.865 | 0.459 | 0.784 | 0.514 |
| topbrain_v2 | topbrain_v2_ck257466_20260907_183918 | 98 | 0.673 | 0.061 | 0.000 | 32 | 0.875 | 0.781 | 0.875 | 0.781 | 0.906 | 0.531 |
| topbrain_v2 | topbrain_v2_ck503474_20260907_193336 | 98 | 0.510 | 0.120 | 0.010 | 48 | 1 | 0.771 | 1 | 0.562 | 0.979 | 0.521 |
| topbrain_v2 | topbrain_v2_ck755233_20260907_053657 | 98 | 0.582 | 0.070 | 0.000 | 41 | 1 | 0.927 | 1 | 0.976 | 0.927 | 0.561 |
| topbrain_v2 | topbrain_v2_ck755233_20260907_205504 | 98 | 0.602 | 0.051 | 0.000 | 39 | 0.949 | 0.821 | 0.949 | 0.872 | 0.974 | 0.333 |
| topbrain_v3 | topbrain_v3_ck1002606_20260908_163751 | 98 | 0.867 | 0.012 | 0.000 | 13 | 0.615 | 0.385 | 0.846 | 0.538 | 0.538 | 0.538 |
| topbrain_v3 | topbrain_v3_ck1256323_20260908_165015 | 98 | 0.918 | 0.000 | 0.000 | 8 | 0.500 | 0.250 | 0.625 | 0.250 | 0.375 | 0.500 |
| topbrain_v3 | topbrain_v3_ck1502006_20260908_173514 | 98 | 0.796 | 0.064 | 0.051 | 20 | 0.850 | 0.700 | 0.850 | 0.250 | 0.600 | 0.550 |
| topbrain_v3 | topbrain_v3_ck1752049_20260908_184204 | 97 | 0.794 | 0.065 | 0.010 | 20 | 0.850 | 0.500 | 0.950 | 0.350 | 0.800 | 0.600 |
| topbrain_v3 | topbrain_v3_ck1752049_20260908_203358 | 72 | 0.750 | 0.037 | 0.000 | 18 | 0.722 | 0.389 | 0.833 | 0.278 | 0.500 | 0.556 |
| topbrain_v3 | topbrain_v3_ck256854_20260908_160020 | 98 | 0.990 | 0.000 | 0.000 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| topbrain_v3 | topbrain_v3_ck504695_20260908_160801 | 98 | 0.949 | 0.022 | 0.000 | 5 | 0.800 | 0.200 | 0.800 | 0.800 | 0.600 | 0.800 |
| topbrain_v3 | topbrain_v3_ck756872_20260908_161717 | 98 | 0.816 | 0.038 | 0.000 | 18 | 1 | 0.167 | 1 | 0.556 | 1 | 0.333 |
| v1bp | v1bp_H0 | 82 | 0.293 |  |  | 58 | 0.603 | 0.603 | 0.638 | 0.466 |  | 0.328 |
| v1bp | v1bp_ck2002292 | 82 | 0.744 |  |  | 21 | 0.190 | 0.190 | 0.190 | 0.095 |  | 0.143 |
| v1bp | v1bp_ck514264 | 82 | 0.695 |  |  | 25 | 0.240 | 0.240 | 0.280 | 0.080 |  | 0.120 |

Answer to question (ii): with the same rule on the same patient, the
carotid-v3-era TopBrain-v3 checkpoints coil in 78.6 % of their failures
(n_fail = 103, 81 coils) against 93.7 % for the TopBrain-v2 checkpoints
(n_fail = 459, 430 coils); the difference is in the severe end -- multi-loop
(> 200 mm) 38.8 % vs 69.1 %, wire-side coils 40.8 % vs 61.7 % -- while the
bow-only share is higher for v3 (10.7 % vs 5.4 %).  Both are far above the
TopBrain-v1 checkpoints (39.4 %, whose failures are mostly the untrained ck0
sessions that never insert enough to coil) and above the v1bp family (43 %,
wire rule only).  Per checkpoint the v3 coil share ranges 50-100 % on 5-20
failures each, so the family-level difference (15 points) rests on ~100 v3
failures; the multi-loop difference (30 points) is the robust part.  The
near-target buckle is where the v3 failures end (52 % of them, 54 of 103,
versus 51 % for v2), and on v3 more of those are bows than loops (16 bow / 20 loop / 18 multi-loop at the last step, table below) whereas v2's
are loops (21 / 73 / 141).

```
=== T5. near-target buckle failures (prog_max>=0.75, >=20 mm stored at the end): grade_final composition ===
near_target_buckle_grade  bow  loop  multi-loop
fam                                            
carotid_v3                149   230         312
host:p2_teacher_v1bp_ck0    1     0           0
host:tb22_v1bp              0     6          13
host:tbv1r1                 2     6          14
host:topbrain_v1            2    18          87
host:topbrain_v2           21    73         141
host:topbrain_v3           16    20          18
host:v1bp                   3    14           8
procedural                 30    86        1310
topbrain_v1               203   228         255
topbrain_v2                 5    15           7
```

## 5. Files

* `monitoring/traj/secondpass/indicators.py` -- the three functions, thresholds
  as module constants, docstrings with the definitions.
* `saved/traj/secondpass/INDICATORS.md` -- this document.
* scratchpad `secondpass/` (session-local, not committed): `episodes_ind.csv`
  (152k episodes x series indicators), `episodes_indicators.csv` (module
  outputs), `rec_eps_v2.csv` / `rec_steps_v2.csv` (polyline geometry per
  recorded episode / step), `blind_merged.csv` (eye labels joined to features),
  `T1..T6*.csv`, the `frames/*.png` check renders and the `*.log` outputs
  quoted above.
