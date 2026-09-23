# Replay campaign summary

## Per checkpoint (all replayed seeds)

| name | reps | episodes | seeds | succ | clean | recovery | fight_won | f_coil | f_buckle | f_thrash | f_shove | med_steps | lead_end_med | lead_frac50 | cathled_succ | rec_share_of_succ | rec_tipretreat_share | tipretreat_med_rec | coil_share_of_fail | buckle_share_of_fail | thrash_share_of_fail | fail_pmax_med | fail_midpath_share | fail_loop_share | loop_any | loop_escaped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| car_best | 4 | 188 | 47 | 0.973 | 0.750 | 0.186 | 0.037 | 0.000 | 0.027 | 0.000 | 0.000 | 71.000 | -18.253 | 0.004 | 0.191 | 0.191 | 0.659 | 10.979 | 0.000 | 1.000 | 0.000 | 0.937 | 0.000 | 0.000 | 0.021 | 1.000 |
| proc_on_carotid | 1 | 47 | 47 | 0.426 | 0.404 | 0.021 | 0.000 | 0.000 | 0.000 | 0.511 | 0.064 | 601.000 | 29.491 | 0.206 | 1.000 | 0.050 | 0.500 | 7.019 | 0.000 | 0.000 | 0.889 | 0.204 | 1.000 | 0.037 | 0.043 | 0.500 |

## Seed classes over pooled reps: strength 24, easy 20, frontier 3

## Paired comparisons (per-seed success probability)

- **car_best vs proc_on_carotid** (n=47 common seeds): 0.973 vs 0.426, mean diff +0.548; seeds better under a: 27, under b: 0; sign p=0.0000; wilcoxon p=0.0000
  - a better on seeds [2, 3, 8, 10, 21, 23, 37, 39, 43, 47, 48, 55, 69, 70, 73, 81, 91, 93, 115, 134, 139, 140, 148, 154, 161, 162, 167]; b better on []

## Seed classes (strong = car_best; weak = proc_on_carotid)

easy = every checkpoint always solves it; strength = all strong checkpoints always solve it but a weak one fails it; frontier = a strong checkpoint fails it in some repetition; floor = no strong checkpoint ever solves it.

## Decisive seeds (frontier + strength + floor): success probability per checkpoint

| seed | car_best | proc_on_carotid | succ_phase | fail_phase | rec_share | cathled_share | medT | cls | min_strong | spread_strong |
|---|---|---|---|---|---|---|---|---|---|---|
| 55 | 0.250 | 0.000 | fight-escaped-near | doorstep-stall | 1.000 | 0.000 | 601.000 | frontier | 0.250 | 0.000 |
| 23 | 0.750 | 0.000 | deep-recovery | doorstep-stall | 1.000 | 0.000 | 271.000 | frontier | 0.750 | 0.000 |
| 134 | 0.750 | 0.000 | deep-recovery | doorstep-stall | 1.000 | 0.000 | 235.000 | frontier | 0.750 | 0.000 |
| 2 | 1.000 | 0.000 | free-run | stuck-proximal | 0.000 | 0.000 | 113.000 | strength | 1.000 | 0.000 |
| 3 | 1.000 | 0.000 | free-run | cath-led-run | 0.250 | 0.000 | 63.000 | strength | 1.000 | 0.000 |
| 8 | 1.000 | 0.000 | deep-recovery | stuck-proximal | 1.000 | 0.000 | 239.000 | strength | 1.000 | 0.000 |
| 10 | 1.000 | 0.000 | free-run | cath-led-run | 0.000 | 0.000 | 69.000 | strength | 1.000 | 0.000 |
| 21 | 1.000 | 0.000 | deep-recovery | light-recovery | 0.750 | 0.000 | 345.000 | strength | 1.000 | 0.000 |
| 37 | 1.000 | 0.000 | free-run | stuck-midpath | 0.250 | 0.250 | 65.000 | strength | 1.000 | 0.000 |
| 39 | 1.000 | 0.000 | light-recovery | cath-led-run | 0.500 | 0.000 | 109.000 | strength | 1.000 | 0.000 |
| 43 | 1.000 | 0.000 | free-run | stuck-proximal | 0.000 | 0.000 | 71.000 | strength | 1.000 | 0.000 |
| 47 | 1.000 | 0.000 | free-run | deep-recovery | 0.250 | 0.000 | 114.000 | strength | 1.000 | 0.000 |
| 48 | 1.000 | 0.000 | free-run | cath-led-run | 0.250 | 0.000 | 76.000 | strength | 1.000 | 0.000 |
| 69 | 1.000 | 0.000 | free-run | cath-led-run | 0.000 | 0.000 | 70.000 | strength | 1.000 | 0.000 |
| 70 | 1.000 | 0.000 | free-run | idle-creep | 0.000 | 0.000 | 68.000 | strength | 1.000 | 0.000 |
| 73 | 1.000 | 0.000 | free-run | stuck-midpath | 0.500 | 0.000 | 147.000 | strength | 1.000 | 0.000 |
| 81 | 1.000 | 0.000 | free-run | cath-led-run | 0.000 | 0.000 | 67.000 | strength | 1.000 | 0.000 |
| 91 | 1.000 | 0.000 | free-run | stuck-proximal | 0.000 | 0.000 | 77.000 | strength | 1.000 | 0.000 |
| 93 | 1.000 | 0.000 | free-run | cath-led-run | 0.000 | 0.000 | 88.000 | strength | 1.000 | 0.000 |
| 115 | 1.000 | 0.000 | deep-recovery | stuck-proximal | 0.500 | 0.000 | 152.000 | strength | 1.000 | 0.000 |
| 139 | 1.000 | 0.000 | deep-recovery | cath-led-run | 0.750 | 0.000 | 170.000 | strength | 1.000 | 0.000 |
| 140 | 1.000 | 0.000 | free-run | stuck-proximal | 0.000 | 0.000 | 72.000 | strength | 1.000 | 0.000 |
| 148 | 1.000 | 0.000 | free-run | cath-led-run | 0.500 | 0.000 | 118.000 | strength | 1.000 | 0.000 |
| 154 | 1.000 | 0.000 | free-run | cath-led-run | 0.250 | 0.000 | 106.000 | strength | 1.000 | 0.000 |
| 161 | 1.000 | 0.000 | free-run | idle-creep | 0.000 | 0.000 | 78.000 | strength | 1.000 | 0.000 |
| 162 | 1.000 | 0.000 | free-run | stuck-midpath | 0.000 | 0.250 | 75.000 | strength | 1.000 | 0.000 |
| 167 | 1.000 | 0.000 | deep-recovery | cath-led-run | 0.750 | 0.000 | 111.000 | strength | 1.000 | 0.000 |

## How successes are won, by seed class (share of successes per phase group)

| name | seed_cls | cathled | deep-rec | fight | free | light-rec | n_succ |
|---|---|---|---|---|---|---|---|
| car_best | easy | 0.238 | 0.038 | 0.000 | 0.662 | 0.062 | 80 |
| car_best | frontier | 0.000 | 0.857 | 0.143 | 0.000 | 0.000 | 7 |
| car_best | strength | 0.021 | 0.156 | 0.010 | 0.708 | 0.104 | 96 |
| proc_on_carotid | easy | 0.900 | 0.000 | 0.000 | 0.000 | 0.100 | 20 |

## How failures end (phase of the failure)

| name | cath-led-run | deep-recovery | doorstep-stall | idle-creep | light-recovery | stuck-midpath | stuck-proximal |
|---|---|---|---|---|---|---|---|
| car_best | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| proc_on_carotid | 11 | 1 | 0 | 3 | 3 | 3 | 6 |