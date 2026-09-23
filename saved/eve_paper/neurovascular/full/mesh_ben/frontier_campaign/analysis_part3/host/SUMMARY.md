# Replay campaign summary

## Per checkpoint (all replayed seeds)

| name | reps | episodes | seeds | succ | clean | recovery | fight_won | f_coil | f_buckle | f_thrash | f_shove | med_steps | lead_end_med | lead_frac50 | cathled_succ | rec_share_of_succ | rec_tipretreat_share | tipretreat_med_rec | coil_share_of_fail | buckle_share_of_fail | thrash_share_of_fail | fail_pmax_med | fail_midpath_share | fail_loop_share | loop_any | loop_escaped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| car_best_host | 3 | 268 | 98 | 0.933 | 0.164 | 0.354 | 0.414 | 0.000 | 0.004 | 0.000 | 0.063 | 186.000 | -28.054 | 0.005 | 0.104 | 0.380 | 0.956 | 16.001 | 0.000 | 0.056 | 0.000 | 0.906 | 0.000 | 0.000 | 0.004 | 1.000 |
| proc_on_host | 1 | 98 | 98 | 0.469 | 0.245 | 0.061 | 0.163 | 0.000 | 0.000 | 0.184 | 0.347 | 601.000 | 25.142 | 0.191 | 0.522 | 0.130 | 0.957 | 12.990 | 0.000 | 0.000 | 0.346 | 0.567 | 0.808 | 0.038 | 0.020 | 0.000 |

## Seed classes over pooled reps: easy 44, strength 39, frontier 15

## Paired comparisons (per-seed success probability)

- **car_best_host vs proc_on_host** (n=98 common seeds): 0.939 vs 0.469, mean diff +0.469; seeds better under a: 52, under b: 2; sign p=0.0000; wilcoxon p=0.0000
  - a better on seeds [900001, 900005, 900007, 900009, 900012, 900013, 900014, 900015, 900019, 900020, 900021, 900024, 900027, 900028, 900029, 900030, 900031, 900033, 900035, 900037, 900039, 900042, 900045, 900047, 900048, 900050, 900051, 900053, 900062, 900063, 900065, 900068, 900069, 900071, 900072, 900074, 900076, 900077, 900079, 900080, 900082, 900083, 900085, 900086, 900087, 900089, 900090, 900091, 900092, 900093, 900096, 900097]; b better on [900061, 900094]

## Seed classes (strong = car_best_host; weak = proc_on_host)

easy = every checkpoint always solves it; strength = all strong checkpoints always solve it but a weak one fails it; frontier = a strong checkpoint fails it in some repetition; floor = no strong checkpoint ever solves it.

## Decisive seeds (frontier + strength + floor): success probability per checkpoint

| seed | car_best_host | proc_on_host | succ_phase | fail_phase | rec_share | cathled_share | medT | cls | min_strong | spread_strong |
|---|---|---|---|---|---|---|---|---|---|---|
| 900069 | 0.330 | 0.000 | light-recovery | stuck-midpath | 1.000 | 0.000 | 601.000 | frontier | 0.330 | 0.000 |
| 900077 | 0.330 | 0.000 | deep-recovery | doorstep-stall | 1.000 | 0.000 | 601.000 | frontier | 0.330 | 0.000 |
| 900090 | 0.330 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 601.000 | frontier | 0.330 | 0.000 |
| 900009 | 0.670 | 0.000 | fight-escaped-near | doorstep-stall | 0.500 | 0.000 | 514.500 | frontier | 0.670 | 0.000 |
| 900012 | 0.670 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 403.500 | frontier | 0.670 | 0.000 |
| 900035 | 0.670 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 490.500 | frontier | 0.670 | 0.000 |
| 900039 | 0.670 | 0.000 | cath-led-run | doorstep-stall | 0.500 | 0.500 | 345.500 | frontier | 0.670 | 0.000 |
| 900048 | 0.670 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 493.500 | frontier | 0.670 | 0.000 |
| 900053 | 0.670 | 0.000 | fight-escaped-near | fight-escaped-near | 1.000 | 0.000 | 462.000 | frontier | 0.670 | 0.000 |
| 900061 | 0.670 | 1.000 | deep-recovery | shove-capped | 1.000 | 0.000 | 89.000 | frontier | 0.670 | 0.000 |
| 900068 | 0.670 | 0.000 | free-run | doorstep-stall | 0.500 | 0.000 | 405.000 | frontier | 0.670 | 0.000 |
| 900074 | 0.670 | 0.000 | fight-escaped-near | doorstep-stall | 1.000 | 0.000 | 418.500 | frontier | 0.670 | 0.000 |
| 900076 | 0.670 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 446.000 | frontier | 0.670 | 0.000 |
| 900089 | 0.670 | 0.000 | fight-escaped-near | doorstep-stall | 1.000 | 0.000 | 421.500 | frontier | 0.670 | 0.000 |
| 900094 | 0.670 | 1.000 | fight-escaped-near | stuck-midpath | 0.670 | 0.330 | 188.500 | frontier | 0.670 | 0.000 |
| 900001 | 1.000 | 0.000 | fight-escaped-near | cath-led-run | 1.000 | 0.000 | 290.000 | strength | 1.000 | 0.000 |
| 900005 | 1.000 | 0.000 | deep-recovery | stuck-midpath | 1.000 | 0.000 | 549.000 | strength | 1.000 | 0.000 |
| 900007 | 1.000 | 0.000 | light-recovery | stuck-midpath | 0.670 | 0.000 | 189.500 | strength | 1.000 | 0.000 |
| 900013 | 1.000 | 0.000 | light-recovery | stuck-midpath | 1.000 | 0.000 | 373.500 | strength | 1.000 | 0.000 |
| 900014 | 1.000 | 0.000 | light-recovery | stuck-midpath | 1.000 | 0.000 | 118.000 | strength | 1.000 | 0.000 |
| 900015 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 237.000 | strength | 1.000 | 0.000 |
| 900019 | 1.000 | 0.000 | fight-escaped-near | deep-recovery | 1.000 | 0.000 | 314.000 | strength | 1.000 | 0.000 |
| 900020 | 1.000 | 0.000 | fight-escaped-near | cath-led-run | 1.000 | 0.000 | 265.000 | strength | 1.000 | 0.000 |
| 900021 | 1.000 | 0.000 | light-recovery | stuck-midpath | 0.670 | 0.000 | 172.500 | strength | 1.000 | 0.000 |
| 900024 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 217.500 | strength | 1.000 | 0.000 |
| 900027 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 0.670 | 0.330 | 204.000 | strength | 1.000 | 0.000 |
| 900028 | 1.000 | 0.000 | light-recovery | stuck-midpath | 0.670 | 0.000 | 128.500 | strength | 1.000 | 0.000 |
| 900029 | 1.000 | 0.000 | fight-escaped-near | cath-led-run | 1.000 | 0.000 | 218.000 | strength | 1.000 | 0.000 |
| 900030 | 1.000 | 0.000 | fight-escaped-near | cath-led-run | 1.000 | 0.000 | 269.500 | strength | 1.000 | 0.000 |
| 900031 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 186.500 | strength | 1.000 | 0.000 |
| 900033 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 202.500 | strength | 1.000 | 0.000 |
| 900037 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 223.000 | strength | 1.000 | 0.000 |
| 900042 | 1.000 | 0.000 | fight-escaped-near | cath-led-run | 1.000 | 0.000 | 214.000 | strength | 1.000 | 0.000 |
| 900045 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 322.000 | strength | 1.000 | 0.000 |
| 900047 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 187.000 | strength | 1.000 | 0.000 |
| 900050 | 1.000 | 0.000 | deep-recovery | light-recovery | 1.000 | 0.000 | 173.000 | strength | 1.000 | 0.000 |
| 900051 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 190.000 | strength | 1.000 | 0.000 |
| 900062 | 1.000 | 0.000 | fight-escaped-near | cath-led-run | 1.000 | 0.000 | 322.000 | strength | 1.000 | 0.000 |
| 900063 | 1.000 | 0.000 | deep-recovery | shove-capped | 1.000 | 0.000 | 346.500 | strength | 1.000 | 0.000 |
| 900065 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 234.500 | strength | 1.000 | 0.000 |
| 900071 | 1.000 | 0.000 | deep-recovery | deep-recovery | 1.000 | 0.000 | 157.500 | strength | 1.000 | 0.000 |
| 900072 | 1.000 | 0.000 | deep-recovery | stuck-midpath | 1.000 | 0.000 | 184.000 | strength | 1.000 | 0.000 |
| 900079 | 1.000 | 0.000 | light-recovery | stuck-midpath | 1.000 | 0.000 | 145.000 | strength | 1.000 | 0.000 |
| 900080 | 1.000 | 0.000 | light-recovery | stuck-midpath | 1.000 | 0.000 | 164.000 | strength | 1.000 | 0.000 |
| 900082 | 1.000 | 0.000 | fight-escaped-near | shove-capped | 0.670 | 0.000 | 371.000 | strength | 1.000 | 0.000 |
| 900083 | 1.000 | 0.000 | light-recovery | deep-recovery | 1.000 | 0.000 | 184.000 | strength | 1.000 | 0.000 |
| 900085 | 1.000 | 0.000 | deep-recovery | stuck-midpath | 1.000 | 0.000 | 155.000 | strength | 1.000 | 0.000 |
| 900086 | 1.000 | 0.000 | fight-escaped-near | cath-led-run | 1.000 | 0.000 | 278.500 | strength | 1.000 | 0.000 |
| 900087 | 1.000 | 0.000 | fight-escaped-near | light-recovery | 1.000 | 0.000 | 309.000 | strength | 1.000 | 0.000 |
| 900091 | 1.000 | 0.000 | light-recovery | stuck-midpath | 0.670 | 0.330 | 169.500 | strength | 1.000 | 0.000 |
| 900092 | 1.000 | 0.000 | deep-recovery | stuck-midpath | 1.000 | 0.000 | 188.500 | strength | 1.000 | 0.000 |
| 900093 | 1.000 | 0.000 | fight-escaped-near | stuck-midpath | 1.000 | 0.000 | 223.500 | strength | 1.000 | 0.000 |
| 900096 | 1.000 | 0.000 | light-recovery | cath-led-run | 1.000 | 0.000 | 211.500 | strength | 1.000 | 0.000 |
| 900097 | 1.000 | 0.000 | fight-escaped-near | shove-capped | 1.000 | 0.000 | 259.500 | strength | 1.000 | 0.000 |

## How successes are won, by seed class (share of successes per phase group)

| name | seed_cls | cathled | deep-rec | fight | free | light-rec | n_succ |
|---|---|---|---|---|---|---|---|
| car_best_host | easy | 0.198 | 0.330 | 0.113 | 0.132 | 0.226 | 106 |
| car_best_host | frontier | 0.037 | 0.185 | 0.519 | 0.074 | 0.185 | 27 |
| car_best_host | strength | 0.017 | 0.248 | 0.453 | 0.034 | 0.248 | 117 |
| proc_on_host | easy | 0.500 | 0.136 | 0.045 | 0.000 | 0.318 | 44 |
| proc_on_host | frontier | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 2 |

## How failures end (phase of the failure)

| name | cath-led-run | deep-recovery | doorstep-stall | fight-escaped-near | light-recovery | shove-capped | stuck-midpath | stuck-proximal |
|---|---|---|---|---|---|---|---|---|
| car_best_host | 0 | 0 | 7 | 1 | 0 | 1 | 9 | 0 |
| proc_on_host | 12 | 3 | 0 | 0 | 3 | 3 | 30 | 1 |