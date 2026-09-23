# Replay campaign summary

## Per checkpoint (all replayed seeds)

| name | reps | episodes | seeds | succ | clean | recovery | fight_won | f_coil | f_buckle | f_thrash | f_shove | med_steps | lead_end_med | lead_frac50 | cathled_succ | rec_share_of_succ | rec_tipretreat_share | tipretreat_med_rec | coil_share_of_fail | buckle_share_of_fail | thrash_share_of_fail | fail_pmax_med | fail_midpath_share | fail_loop_share | loop_any | loop_escaped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_best | 4 | 188 | 47 | 0.926 | 0.723 | 0.165 | 0.037 | 0.005 | 0.053 | 0.000 | 0.016 | 70.500 | -32.369 | 0.018 | 0.029 | 0.178 | 0.421 | 5.983 | 0.071 | 0.714 | 0.000 | 0.957 | 0.000 | 0.357 | 0.059 | 0.545 |
| abl_copeak | 1 | 47 | 47 | 0.872 | 0.638 | 0.213 | 0.021 | 0.043 | 0.043 | 0.000 | 0.043 | 68.000 | -17.668 | 0.096 | 0.024 | 0.244 | 0.455 | 6.983 | 0.333 | 0.333 | 0.000 | 0.908 | 0.333 | 1.000 | 0.149 | 0.143 |
| abl_last | 2 | 94 | 47 | 0.766 | 0.617 | 0.128 | 0.021 | 0.021 | 0.074 | 0.128 | 0.011 | 67.500 | -20.751 | 0.013 | 0.000 | 0.167 | 0.643 | 15.436 | 0.091 | 0.318 | 0.545 | 0.457 | 0.591 | 0.273 | 0.085 | 0.250 |
| car_1545 | 1 | 47 | 47 | 0.936 | 0.681 | 0.213 | 0.043 | 0.000 | 0.064 | 0.000 | 0.000 | 68.000 | -20.187 | 0.000 | 0.000 | 0.227 | 0.692 | 9.951 | 0.000 | 1.000 | 0.000 | 0.887 | 0.000 | 0.333 | 0.021 | 0.000 |
| car_best | 4 | 188 | 47 | 0.973 | 0.750 | 0.186 | 0.037 | 0.000 | 0.027 | 0.000 | 0.000 | 71.000 | -18.253 | 0.004 | 0.191 | 0.191 | 0.659 | 10.979 | 0.000 | 1.000 | 0.000 | 0.937 | 0.000 | 0.000 | 0.021 | 1.000 |
| car_div | 1 | 47 | 47 | 0.404 | 0.043 | 0.106 | 0.255 | 0.000 | 0.000 | 0.511 | 0.000 | 601.000 | -38.959 | 0.000 | 0.000 | 0.263 | 0.684 | 10.078 | 0.000 | 0.000 | 0.857 | 0.563 | 0.571 | 0.143 | 0.085 | 0.000 |
| car_early | 1 | 47 | 47 | 0.745 | 0.383 | 0.106 | 0.255 | 0.170 | 0.021 | 0.021 | 0.021 | 172.000 | -27.210 | 0.094 | 0.086 | 0.143 | 0.722 | 13.544 | 0.667 | 0.083 | 0.083 | 0.685 | 0.417 | 0.917 | 0.234 | 0.000 |
| car_peak | 2 | 94 | 47 | 0.936 | 0.713 | 0.202 | 0.021 | 0.011 | 0.021 | 0.000 | 0.032 | 75.000 | -20.956 | 0.017 | 0.000 | 0.216 | 0.545 | 10.469 | 0.167 | 0.333 | 0.000 | 0.927 | 0.000 | 0.333 | 0.043 | 0.500 |

## Seed classes over pooled reps: strength 23, easy 14, frontier 10

## Paired comparisons (per-seed success probability)

- **car_best vs abl_best** (n=47 common seeds): 0.973 vs 0.926, mean diff +0.048; seeds better under a: 6, under b: 2; sign p=0.2891; wilcoxon p=0.2109
  - a better on seeds [16, 31, 134, 139, 154, 167]; b better on [23, 55]
- **car_best vs car_peak** (n=47 common seeds): 0.973 vs 0.936, mean diff +0.037; seeds better under a: 3, under b: 1; sign p=0.6250
  - a better on seeds [8, 23, 134]; b better on [55]
- **car_best vs car_1545** (n=47 common seeds): 0.973 vs 0.936, mean diff +0.037; seeds better under a: 3, under b: 1; sign p=0.6250
  - a better on seeds [55, 93, 134]; b better on [23]
- **car_best vs car_early** (n=47 common seeds): 0.973 vs 0.745, mean diff +0.229; seeds better under a: 12, under b: 0; sign p=0.0005; wilcoxon p=0.0005
  - a better on seeds [2, 8, 23, 34, 37, 47, 55, 118, 134, 148, 154, 161]; b better on []
- **car_best vs car_div** (n=47 common seeds): 0.973 vs 0.404, mean diff +0.569; seeds better under a: 28, under b: 0; sign p=0.0000; wilcoxon p=0.0000
  - a better on seeds [1, 2, 3, 8, 10, 21, 23, 34, 39, 43, 47, 48, 55, 69, 70, 81, 91, 93, 108, 115, 134, 139, 140, 148, 154, 161, 162, 167]; b better on []
- **car_peak vs abl_best** (n=47 common seeds): 0.936 vs 0.926, mean diff +0.011; seeds better under a: 5, under b: 4; sign p=1.0000; wilcoxon p=0.8555
  - a better on seeds [16, 31, 139, 154, 167]; b better on [8, 23, 55, 134]
- **car_1545 vs abl_best** (n=47 common seeds): 0.936 vs 0.926, mean diff +0.011; seeds better under a: 5, under b: 3; sign p=0.7266; wilcoxon p=0.8984
  - a better on seeds [16, 31, 139, 154, 167]; b better on [55, 93, 134]
- **abl_best vs abl_last** (n=47 common seeds): 0.926 vs 0.766, mean diff +0.160; seeds better under a: 12, under b: 2; sign p=0.0129; wilcoxon p=0.0168
  - a better on seeds [2, 8, 23, 47, 55, 93, 124, 134, 139, 154, 161, 167]; b better on [16, 31]
- **abl_best vs abl_copeak** (n=47 common seeds): 0.926 vs 0.872, mean diff +0.053; seeds better under a: 6, under b: 4; sign p=0.7539; wilcoxon p=0.3320
  - a better on seeds [23, 48, 134, 140, 154, 167]; b better on [16, 31, 55, 139]
- **car_early vs abl_copeak** (n=47 common seeds): 0.745 vs 0.872, mean diff -0.128; seeds better under a: 3, under b: 9; sign p=0.1460; wilcoxon p=0.1460
  - a better on seeds [48, 140, 167]; b better on [2, 8, 34, 37, 47, 55, 118, 148, 161]

## Seed classes (strong = car_best, car_peak, car_1545, abl_best; weak = abl_copeak, abl_last, car_div, car_early)

easy = every checkpoint always solves it; strength = all strong checkpoints always solve it but a weak one fails it; frontier = a strong checkpoint fails it in some repetition; floor = no strong checkpoint ever solves it.

## Decisive seeds (frontier + strength + floor): success probability per checkpoint

| seed | abl_best | abl_copeak | abl_last | car_1545 | car_best | car_div | car_early | car_peak | succ_phase | fail_phase | rec_share | cathled_share | medT | cls | min_strong | spread_strong |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 23 | 1.000 | 0.000 | 0.000 | 1.000 | 0.750 | 0.000 | 0.000 | 0.000 | deep-recovery | deep-recovery | 1.000 | 0.000 | 456.500 | frontier | 0.000 | 1.000 |
| 31 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | free-run | doorstep-stall | 0.420 | 0.080 | 92.000 | frontier | 0.000 | 1.000 |
| 55 | 0.750 | 1.000 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 | 0.500 | light-recovery | doorstep-stall | 0.830 | 0.170 | 601.000 | frontier | 0.000 | 0.750 |
| 93 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | stuck-midpath | 0.000 | 0.000 | 95.000 | frontier | 0.000 | 1.000 |
| 134 | 0.500 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | deep-recovery | shove-capped | 1.000 | 0.000 | 601.000 | frontier | 0.000 | 0.750 |
| 167 | 0.250 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | deep-recovery | doorstep-stall | 0.780 | 0.000 | 266.000 | frontier | 0.250 | 0.750 |
| 8 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.500 | deep-recovery | doorstep-stall | 0.820 | 0.000 | 225.000 | frontier | 0.500 | 0.500 |
| 16 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | free-run | doorstep-stall | 0.140 | 0.210 | 34.500 | frontier | 0.500 | 0.500 |
| 139 | 0.750 | 1.000 | 0.500 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | deep-recovery | shove-capped | 0.850 | 0.080 | 146.500 | frontier | 0.750 | 0.250 |
| 154 | 0.750 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | free-run | stuck-midpath | 0.400 | 0.000 | 129.000 | frontier | 0.750 | 0.250 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.070 | 0.070 | 50.000 | strength | 1.000 | 0.000 |
| 2 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | free-run | coiled-stuck | 0.170 | 0.000 | 112.000 | strength | 1.000 | 0.000 |
| 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.400 | 0.000 | 72.500 | strength | 1.000 | 0.000 |
| 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | stuck-midpath | 0.130 | 0.070 | 73.500 | strength | 1.000 | 0.000 |
| 21 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | deep-recovery | fight-escaped-far | 0.530 | 0.000 | 233.500 | strength | 1.000 | 0.000 |
| 34 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | free-run | shove-capped | 0.290 | 0.070 | 74.000 | strength | 1.000 | 0.000 |
| 37 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | light-recovery | deep-recovery | 0.600 | 0.070 | 77.000 | strength | 1.000 | 0.000 |
| 39 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | doorstep-stall | 0.400 | 0.070 | 96.000 | strength | 1.000 | 0.000 |
| 43 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.000 | 0.070 | 69.500 | strength | 1.000 | 0.000 |
| 47 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | free-run | doorstep-stall | 0.170 | 0.000 | 90.500 | strength | 1.000 | 0.000 |
| 48 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | stuck-proximal | 0.290 | 0.000 | 76.000 | strength | 1.000 | 0.000 |
| 69 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.330 | 0.070 | 70.500 | strength | 1.000 | 0.000 |
| 70 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | stuck-proximal | 0.000 | 0.000 | 67.500 | strength | 1.000 | 0.000 |
| 81 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | stuck-proximal | 0.000 | 0.070 | 62.500 | strength | 1.000 | 0.000 |
| 91 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.070 | 0.000 | 70.500 | strength | 1.000 | 0.000 |
| 108 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.000 | 0.070 | 47.000 | strength | 1.000 | 0.000 |
| 115 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.400 | 0.000 | 82.500 | strength | 1.000 | 0.000 |
| 118 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | free-run | coiled-stuck | 0.200 | 0.000 | 61.500 | strength | 1.000 | 0.000 |
| 124 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | free-run | stuck-midpath | 0.400 | 0.130 | 76.000 | strength | 1.000 | 0.000 |
| 140 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | fight-escaped-far | 0.000 | 0.070 | 70.500 | strength | 1.000 | 0.000 |
| 148 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | free-run | coiled-stuck | 0.500 | 0.000 | 101.500 | strength | 1.000 | 0.000 |
| 161 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | free-run | stuck-midpath | 0.000 | 0.000 | 78.000 | strength | 1.000 | 0.000 |
| 162 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | free-run | doorstep-stall | 0.070 | 0.070 | 74.500 | strength | 1.000 | 0.000 |

## How successes are won, by seed class (share of successes per phase group)

| name | seed_cls | cathled | deep-rec | fight | free | light-rec | n_succ |
|---|---|---|---|---|---|---|---|
| abl_best | easy | 0.161 | 0.000 | 0.000 | 0.768 | 0.071 | 56 |
| abl_best | frontier | 0.077 | 0.385 | 0.000 | 0.308 | 0.231 | 26 |
| abl_best | strength | 0.000 | 0.109 | 0.000 | 0.804 | 0.087 | 92 |
| abl_copeak | easy | 0.000 | 0.000 | 0.000 | 0.786 | 0.214 | 14 |
| abl_copeak | frontier | 0.000 | 0.333 | 0.000 | 0.333 | 0.333 | 6 |
| abl_copeak | strength | 0.000 | 0.000 | 0.000 | 0.810 | 0.190 | 21 |
| abl_last | easy | 0.000 | 0.036 | 0.000 | 0.929 | 0.036 | 28 |
| abl_last | frontier | 0.000 | 0.000 | 0.000 | 0.800 | 0.200 | 5 |
| abl_last | strength | 0.000 | 0.077 | 0.000 | 0.718 | 0.205 | 39 |
| car_1545 | easy | 0.000 | 0.000 | 0.000 | 0.714 | 0.286 | 14 |
| car_1545 | frontier | 0.000 | 0.429 | 0.000 | 0.286 | 0.286 | 7 |
| car_1545 | strength | 0.043 | 0.043 | 0.000 | 0.783 | 0.130 | 23 |
| car_best | easy | 0.250 | 0.071 | 0.000 | 0.607 | 0.071 | 56 |
| car_best | frontier | 0.086 | 0.429 | 0.057 | 0.400 | 0.029 | 35 |
| car_best | strength | 0.043 | 0.054 | 0.000 | 0.793 | 0.109 | 92 |
| car_div | easy | 0.000 | 0.000 | 0.786 | 0.000 | 0.214 | 14 |
| car_div | frontier | 0.000 | 0.500 | 0.000 | 0.000 | 0.500 | 2 |
| car_div | strength | 0.000 | 0.000 | 0.667 | 0.000 | 0.333 | 3 |
| car_early | easy | 0.000 | 0.000 | 0.643 | 0.143 | 0.214 | 14 |
| car_early | frontier | 0.200 | 0.000 | 0.200 | 0.200 | 0.400 | 5 |
| car_early | strength | 0.438 | 0.125 | 0.062 | 0.375 | 0.000 | 16 |
| car_peak | easy | 0.071 | 0.000 | 0.000 | 0.929 | 0.000 | 28 |
| car_peak | frontier | 0.000 | 0.286 | 0.000 | 0.500 | 0.214 | 14 |
| car_peak | strength | 0.022 | 0.109 | 0.000 | 0.652 | 0.217 | 46 |

## How failures end (phase of the failure)

| name | coiled-stuck | deep-recovery | doorstep-stall | fight-escaped-far | fight-escaped-near | light-recovery | shove-capped | stuck-midpath | stuck-proximal |
|---|---|---|---|---|---|---|---|---|---|
| abl_best | 1 | 1 | 6 | 0 | 0 | 2 | 4 | 0 | 0 |
| abl_copeak | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| abl_last | 2 | 0 | 3 | 1 | 0 | 0 | 4 | 12 | 0 |
| car_1545 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| car_best | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| car_div | 0 | 2 | 3 | 13 | 1 | 2 | 0 | 2 | 5 |
| car_early | 5 | 1 | 0 | 0 | 1 | 0 | 4 | 1 | 0 |
| car_peak | 0 | 2 | 2 | 0 | 0 | 0 | 2 | 0 | 0 |