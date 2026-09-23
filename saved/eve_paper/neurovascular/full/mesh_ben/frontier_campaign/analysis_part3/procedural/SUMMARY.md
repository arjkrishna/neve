# Replay campaign summary

## Per checkpoint (all replayed seeds)

| name | reps | episodes | seeds | succ | clean | recovery | fight_won | f_coil | f_buckle | f_thrash | f_shove | med_steps | lead_end_med | lead_frac50 | cathled_succ | rec_share_of_succ | rec_tipretreat_share | tipretreat_med_rec | coil_share_of_fail | buckle_share_of_fail | thrash_share_of_fail | fail_pmax_med | fail_midpath_share | fail_loop_share | loop_any | loop_escaped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| car_best_on_proc | 1 | 98 | 98 | 0.582 | 0.500 | 0.051 | 0.031 | 0.071 | 0.286 | 0.041 | 0.020 | 87.500 | -15.732 | 0.055 | 0.351 | 0.088 | 0.500 | 14.037 | 0.171 | 0.683 | 0.098 | 0.756 | 0.341 | 0.463 | 0.265 | 0.269 |
| proc_on_proc | 1 | 98 | 98 | 0.500 | 0.388 | 0.102 | 0.010 | 0.378 | 0.000 | 0.122 | 0.000 | 417.000 | 18.701 | 0.174 | 0.878 | 0.204 | 0.583 | 11.333 | 0.755 | 0.000 | 0.245 | 0.717 | 0.408 | 0.755 | 0.490 | 0.229 |

## Seed classes over pooled reps: easy 49, floor 41, frontier 8

## Paired comparisons (per-seed success probability)

- **car_best_on_proc vs proc_on_proc** (n=98 common seeds): 0.582 vs 0.500, mean diff +0.082; seeds better under a: 8, under b: 0; sign p=0.0078; wilcoxon p=0.0078
  - a better on seeds [900023, 900031, 900039, 900043, 900045, 900064, 900069, 900086]; b better on []

## Seed classes (strong = car_best_on_proc, proc_on_proc; weak = none)

easy = every checkpoint always solves it; strength = all strong checkpoints always solve it but a weak one fails it; frontier = a strong checkpoint fails it in some repetition; floor = no strong checkpoint ever solves it.

## Decisive seeds (frontier + strength + floor): success probability per checkpoint

| seed | car_best_on_proc | proc_on_proc | succ_phase | fail_phase | rec_share | cathled_share | medT | cls | min_strong | spread_strong |
|---|---|---|---|---|---|---|---|---|---|---|
| 900001 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900004 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900010 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900012 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900014 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900016 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900019 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900020 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900021 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900024 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900027 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900028 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900029 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900030 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900033 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900035 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900037 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900042 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900046 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900047 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900048 | 0.000 | 0.000 | nan | stuck-proximal | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900049 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900051 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900062 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900063 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900065 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900070 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900071 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900072 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900073 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900076 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900079 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900080 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900083 | 0.000 | 0.000 | nan | coiled-stuck | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900087 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900089 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900090 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900092 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900094 | 0.000 | 0.000 | nan | shove-capped | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900096 | 0.000 | 0.000 | nan | stuck-midpath | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900097 | 0.000 | 0.000 | nan | doorstep-stall | nan | nan | 601.000 | floor | 0.000 | 0.000 |
| 900023 | 1.000 | 0.000 | free-run | coiled-stuck | 0.000 | 0.000 | 351.500 | frontier | 0.000 | 1.000 |
| 900031 | 1.000 | 0.000 | free-run | coiled-stuck | 0.000 | 0.000 | 333.000 | frontier | 0.000 | 1.000 |
| 900039 | 1.000 | 0.000 | light-recovery | coiled-stuck | 1.000 | 0.000 | 362.000 | frontier | 0.000 | 1.000 |
| 900043 | 1.000 | 0.000 | deep-recovery | stuck-midpath | 1.000 | 0.000 | 510.000 | frontier | 0.000 | 1.000 |
| 900045 | 1.000 | 0.000 | light-recovery | coiled-stuck | 1.000 | 0.000 | 362.500 | frontier | 0.000 | 1.000 |
| 900064 | 1.000 | 0.000 | cath-led-run | deep-recovery | 0.000 | 1.000 | 330.500 | frontier | 0.000 | 1.000 |
| 900069 | 1.000 | 0.000 | deep-recovery | coiled-stuck | 1.000 | 0.000 | 410.000 | frontier | 0.000 | 1.000 |
| 900086 | 1.000 | 0.000 | cath-led-run | stuck-proximal | 0.000 | 1.000 | 372.000 | frontier | 0.000 | 1.000 |

## How successes are won, by seed class (share of successes per phase group)

| name | seed_cls | cathled | deep-rec | free | light-rec | n_succ |
|---|---|---|---|---|---|---|
| car_best_on_proc | easy | 0.490 | 0.020 | 0.469 | 0.020 | 49 |
| car_best_on_proc | frontier | 0.250 | 0.250 | 0.250 | 0.250 | 8 |
| proc_on_proc | easy | 0.755 | 0.000 | 0.000 | 0.245 | 49 |

## How failures end (phase of the failure)

| name | cath-led-run | coiled-stuck | deep-recovery | doorstep-stall | fight-escaped-far | idle-creep | light-recovery | shove-capped | stuck-midpath | stuck-proximal |
|---|---|---|---|---|---|---|---|---|---|---|
| car_best_on_proc | 0 | 8 | 0 | 1 | 0 | 0 | 0 | 8 | 23 | 1 |
| proc_on_proc | 2 | 35 | 1 | 0 | 1 | 3 | 3 | 2 | 1 | 1 |