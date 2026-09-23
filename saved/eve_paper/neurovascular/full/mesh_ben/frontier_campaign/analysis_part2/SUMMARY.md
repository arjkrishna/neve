# Replay campaign summary

## Per checkpoint (all replayed seeds)

| name | reps | episodes | seeds | succ | clean | recovery | fight_won | f_coil | f_buckle | f_thrash | f_shove | med_steps | lead_end_med | lead_frac50 | cathled_succ | rec_share_of_succ | rec_tipretreat_share | tipretreat_med_rec | coil_share_of_fail | buckle_share_of_fail | thrash_share_of_fail | fail_pmax_med | fail_midpath_share | fail_loop_share | loop_any | loop_escaped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_best_host | 1 | 85 | 85 | 0.953 | 0.706 | 0.200 | 0.047 | 0.012 | 0.035 | 0.000 | 0.000 | 73.000 | -34.832 | 0.029 | 0.012 | 0.210 | 0.381 | 7.991 | 0.250 | 0.750 | 0.000 | 0.949 | 0.000 | 1.000 | 0.106 | 0.556 |
| car_best_host | 3 | 268 | 98 | 0.933 | 0.164 | 0.354 | 0.414 | 0.000 | 0.004 | 0.000 | 0.063 | 186.000 | -28.054 | 0.005 | 0.104 | 0.380 | 0.956 | 16.001 | 0.000 | 0.056 | 0.000 | 0.906 | 0.000 | 0.000 | 0.004 | 1.000 |
| tbv2_best_host | 3 | 255 | 85 | 0.565 | 0.239 | 0.129 | 0.196 | 0.051 | 0.337 | 0.020 | 0.027 | 389.000 | -6.450 | 0.098 | 0.042 | 0.229 | 0.904 | 30.033 | 0.117 | 0.775 | 0.045 | 0.751 | 0.306 | 0.694 | 0.541 | 0.442 |
| tbv3_ref_host | 1 | 85 | 85 | 0.929 | 0.612 | 0.235 | 0.082 | 0.012 | 0.035 | 0.012 | 0.000 | 74.000 | -15.530 | 0.010 | 0.000 | 0.253 | 0.536 | 10.450 | 0.167 | 0.500 | 0.167 | 0.935 | 0.167 | 1.000 | 0.094 | 0.250 |

## Seed classes over pooled reps: strength 57, frontier 23, easy 18

## Paired comparisons (per-seed success probability)

- **car_best_host vs tbv2_best_host** (n=85 common seeds): 0.929 vs 0.565, mean diff +0.365; seeds better under a: 59, under b: 3; sign p=0.0000; wilcoxon p=0.0000
  - a better on seeds [900001, 900003, 900004, 900006, 900007, 900009, 900010, 900014, 900016, 900017, 900018, 900020, 900021, 900022, 900023, 900024, 900027, 900028, 900029, 900030, 900034, 900035, 900036, 900037, 900039, 900042, 900043, 900045, 900046, 900047, 900049, 900050, 900051, 900053, 900054, 900056, 900057, 900059, 900061, 900062, 900063, 900064, 900068, 900071, 900073, 900074, 900076, 900079, 900080, 900082, 900085, 900086, 900087, 900088, 900091, 900092, 900093, 900094, 900097]; b better on [900069, 900077, 900090]
- **car_best_host vs tbv3_ref_host** (n=85 common seeds): 0.929 vs 0.929, mean diff -0.000; seeds better under a: 6, under b: 15; sign p=0.0784; wilcoxon p=0.8726
  - a better on seeds [900018, 900029, 900030, 900042, 900043, 900087]; b better on [900009, 900012, 900035, 900039, 900048, 900053, 900061, 900068, 900069, 900074, 900076, 900077, 900089, 900090, 900094]
- **car_best_host vs abl_best_host** (n=85 common seeds): 0.929 vs 0.953, mean diff -0.024; seeds better under a: 4, under b: 14; sign p=0.0309; wilcoxon p=0.3137
  - a better on seeds [900019, 900029, 900048, 900096]; b better on [900009, 900012, 900035, 900039, 900053, 900061, 900068, 900069, 900074, 900076, 900077, 900089, 900090, 900094]
- **abl_best_host vs tbv2_best_host** (n=85 common seeds): 0.953 vs 0.565, mean diff +0.388; seeds better under a: 62, under b: 3; sign p=0.0000; wilcoxon p=0.0000
  - a better on seeds [900001, 900003, 900004, 900006, 900007, 900009, 900010, 900012, 900014, 900016, 900017, 900018, 900020, 900021, 900022, 900023, 900024, 900027, 900028, 900030, 900034, 900035, 900036, 900037, 900039, 900042, 900043, 900045, 900046, 900047, 900049, 900050, 900051, 900053, 900054, 900056, 900057, 900059, 900061, 900062, 900063, 900064, 900068, 900069, 900071, 900073, 900074, 900076, 900077, 900079, 900080, 900082, 900085, 900086, 900087, 900088, 900089, 900091, 900092, 900093, 900094, 900097]; b better on [900019, 900048, 900096]
- **tbv3_ref_host vs tbv2_best_host** (n=85 common seeds): 0.929 vs 0.565, mean diff +0.365; seeds better under a: 58, under b: 5; sign p=0.0000; wilcoxon p=0.0000
  - a better on seeds [900001, 900003, 900004, 900006, 900007, 900009, 900010, 900012, 900014, 900016, 900017, 900020, 900021, 900022, 900023, 900024, 900027, 900028, 900034, 900035, 900036, 900037, 900039, 900045, 900046, 900047, 900048, 900049, 900050, 900051, 900053, 900054, 900056, 900057, 900059, 900061, 900062, 900063, 900064, 900068, 900069, 900071, 900073, 900074, 900076, 900077, 900079, 900080, 900082, 900085, 900086, 900088, 900089, 900091, 900092, 900093, 900094, 900097]; b better on [900018, 900030, 900042, 900043, 900087]
- **abl_best_host vs tbv3_ref_host** (n=85 common seeds): 0.953 vs 0.929, mean diff +0.024; seeds better under a: 5, under b: 3; sign p=0.7266; wilcoxon p=0.7266
  - a better on seeds [900018, 900030, 900042, 900043, 900087]; b better on [900019, 900048, 900096]

## Seed classes (strong = car_best_host, tbv3_ref_host, abl_best_host; weak = tbv2_best_host)

easy = every checkpoint always solves it; strength = all strong checkpoints always solve it but a weak one fails it; frontier = a strong checkpoint fails it in some repetition; floor = no strong checkpoint ever solves it.

## Decisive seeds (frontier + strength + floor): success probability per checkpoint

| seed | abl_best_host | car_best_host | tbv2_best_host | tbv3_ref_host | succ_phase | fail_phase | rec_share | cathled_share | medT | cls | min_strong | spread_strong |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 900018 | 1.000 | 1.000 | 0.330 | 0.000 | free-run | coiled-stuck | 0.400 | 0.000 | 220.000 | frontier | 0.000 | 1.000 |
| 900019 | 0.000 | 1.000 | 1.000 | 1.000 | fight-escaped-near | shove-capped | 0.710 | 0.000 | 201.000 | frontier | 0.000 | 1.000 |
| 900029 | 0.000 | 1.000 | 0.000 | 0.000 | fight-escaped-near | coiled-stuck | 1.000 | 0.000 | 601.000 | frontier | 0.000 | 1.000 |
| 900030 | 1.000 | 1.000 | 0.670 | 0.000 | deep-recovery | shove-capped | 1.000 | 0.000 | 269.500 | frontier | 0.000 | 1.000 |
| 900042 | 1.000 | 1.000 | 0.330 | 0.000 | free-run | doorstep-stall | 0.600 | 0.000 | 214.000 | frontier | 0.000 | 1.000 |
| 900043 | 1.000 | 1.000 | 0.330 | 0.000 | free-run | shove-capped | 0.600 | 0.000 | 488.500 | frontier | 0.000 | 1.000 |
| 900048 | 0.000 | 0.670 | 0.670 | 1.000 | fight-escaped-near | stuck-midpath | 0.800 | 0.000 | 326.000 | frontier | 0.000 | 1.000 |
| 900087 | 1.000 | 1.000 | 0.330 | 0.000 | deep-recovery | shove-capped | 0.800 | 0.000 | 389.000 | frontier | 0.000 | 1.000 |
| 900096 | 0.000 | 1.000 | 1.000 | 1.000 | light-recovery | shove-capped | 0.860 | 0.000 | 288.500 | frontier | 0.000 | 1.000 |
| 900069 | 1.000 | 0.330 | 0.670 | 1.000 | deep-recovery | stuck-midpath | 1.000 | 0.000 | 260.000 | frontier | 0.330 | 0.670 |
| 900077 | 1.000 | 0.330 | 0.670 | 1.000 | deep-recovery | doorstep-stall | 0.800 | 0.000 | 381.500 | frontier | 0.330 | 0.670 |
| 900090 | 1.000 | 0.330 | 1.000 | 1.000 | free-run | stuck-midpath | 0.500 | 0.000 | 159.000 | frontier | 0.330 | 0.670 |
| 900009 | 1.000 | 0.670 | 0.330 | 1.000 | free-run | stuck-midpath | 0.600 | 0.000 | 426.000 | frontier | 0.670 | 0.330 |
| 900012 | 1.000 | 0.670 | 0.670 | 1.000 | light-recovery | stuck-midpath | 0.830 | 0.000 | 270.000 | frontier | 0.670 | 0.330 |
| 900035 | 1.000 | 0.670 | 0.330 | 1.000 | deep-recovery | stuck-midpath | 1.000 | 0.000 | 389.500 | frontier | 0.670 | 0.330 |
| 900039 | 1.000 | 0.670 | 0.330 | 1.000 | light-recovery | stuck-midpath | 0.400 | 0.200 | 93.500 | frontier | 0.670 | 0.330 |
| 900053 | 1.000 | 0.670 | 0.330 | 1.000 | deep-recovery | stuck-midpath | 0.600 | 0.000 | 356.500 | frontier | 0.670 | 0.330 |
| 900061 | 1.000 | 0.670 | 0.330 | 1.000 | free-run | shove-capped | 0.400 | 0.000 | 78.500 | frontier | 0.670 | 0.330 |
| 900068 | 1.000 | 0.670 | 0.330 | 1.000 | free-run | doorstep-stall | 0.400 | 0.000 | 274.500 | frontier | 0.670 | 0.330 |
| 900074 | 1.000 | 0.670 | 0.000 | 1.000 | fight-escaped-near | stuck-midpath | 0.500 | 0.000 | 418.500 | frontier | 0.670 | 0.330 |
| 900076 | 1.000 | 0.670 | 0.000 | 1.000 | light-recovery | shove-capped | 0.750 | 0.000 | 446.000 | frontier | 0.670 | 0.330 |
| 900089 | 1.000 | 0.670 | 0.670 | 1.000 | fight-escaped-near | doorstep-stall | 0.830 | 0.000 | 295.500 | frontier | 0.670 | 0.330 |
| 900094 | 1.000 | 0.670 | 0.330 | 1.000 | light-recovery | stuck-midpath | 1.000 | 0.000 | 232.000 | frontier | 0.670 | 0.330 |
| 900000 | nan | 1.000 | nan | nan | free-run | nan | 0.000 | 0.000 | 44.000 | strength | 1.000 | 0.000 |
| 900001 | 1.000 | 1.000 | 0.670 | 1.000 | fight-escaped-near | shove-capped | 0.710 | 0.000 | 204.000 | strength | 1.000 | 0.000 |
| 900003 | 1.000 | 1.000 | 0.330 | 1.000 | free-run | shove-capped | 0.500 | 0.000 | 271.000 | strength | 1.000 | 0.000 |
| 900004 | 1.000 | 1.000 | 0.670 | 1.000 | free-run | shove-capped | 0.430 | 0.000 | 127.500 | strength | 1.000 | 0.000 |
| 900006 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | shove-capped | 0.570 | 0.000 | 355.500 | strength | 1.000 | 0.000 |
| 900007 | 1.000 | 1.000 | 0.670 | 1.000 | free-run | coiled-stuck | 0.570 | 0.000 | 119.000 | strength | 1.000 | 0.000 |
| 900008 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 26.000 | strength | 1.000 | 0.000 |
| 900010 | 1.000 | 1.000 | 0.330 | 1.000 | light-recovery | shove-capped | 1.000 | 0.000 | 216.500 | strength | 1.000 | 0.000 |
| 900011 | nan | 1.000 | nan | nan | light-recovery | nan | 1.000 | 0.000 | 66.000 | strength | 1.000 | 0.000 |
| 900014 | 1.000 | 1.000 | 0.330 | 1.000 | free-run | stuck-midpath | 0.500 | 0.000 | 84.500 | strength | 1.000 | 0.000 |
| 900016 | 1.000 | 1.000 | 0.000 | 1.000 | light-recovery | stuck-midpath | 0.800 | 0.000 | 162.500 | strength | 1.000 | 0.000 |
| 900017 | 1.000 | 1.000 | 0.330 | 1.000 | free-run | shove-capped | 0.330 | 0.000 | 227.000 | strength | 1.000 | 0.000 |
| 900020 | 1.000 | 1.000 | 0.330 | 1.000 | fight-escaped-near | coiled-stuck | 0.500 | 0.000 | 222.000 | strength | 1.000 | 0.000 |
| 900021 | 1.000 | 1.000 | 0.330 | 1.000 | light-recovery | coiled-stuck | 0.670 | 0.000 | 161.500 | strength | 1.000 | 0.000 |
| 900022 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | coiled-stuck | 0.860 | 0.000 | 335.500 | strength | 1.000 | 0.000 |
| 900023 | 1.000 | 1.000 | 0.330 | 1.000 | light-recovery | coiled-stuck | 0.670 | 0.000 | 169.000 | strength | 1.000 | 0.000 |
| 900024 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | shove-capped | 0.710 | 0.000 | 217.500 | strength | 1.000 | 0.000 |
| 900026 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 40.000 | strength | 1.000 | 0.000 |
| 900027 | 1.000 | 1.000 | 0.670 | 1.000 | free-run | shove-capped | 0.430 | 0.140 | 130.000 | strength | 1.000 | 0.000 |
| 900028 | 1.000 | 1.000 | 0.670 | 1.000 | free-run | coiled-stuck | 0.430 | 0.000 | 99.500 | strength | 1.000 | 0.000 |
| 900032 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 36.000 | strength | 1.000 | 0.000 |
| 900034 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | doorstep-stall | 0.570 | 0.000 | 324.500 | strength | 1.000 | 0.000 |
| 900036 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | shove-capped | 0.570 | 0.000 | 81.000 | strength | 1.000 | 0.000 |
| 900037 | 1.000 | 1.000 | 0.330 | 1.000 | light-recovery | stuck-midpath | 1.000 | 0.000 | 259.000 | strength | 1.000 | 0.000 |
| 900040 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 26.000 | strength | 1.000 | 0.000 |
| 900045 | 1.000 | 1.000 | 0.000 | 1.000 | fight-escaped-near | stuck-midpath | 0.600 | 0.000 | 322.000 | strength | 1.000 | 0.000 |
| 900046 | 1.000 | 1.000 | 0.670 | 1.000 | free-run | stuck-midpath | 0.430 | 0.000 | 126.500 | strength | 1.000 | 0.000 |
| 900047 | 1.000 | 1.000 | 0.000 | 1.000 | fight-escaped-near | stuck-midpath | 0.800 | 0.000 | 278.500 | strength | 1.000 | 0.000 |
| 900049 | 1.000 | 1.000 | 0.670 | 1.000 | free-run | shove-capped | 0.570 | 0.000 | 175.500 | strength | 1.000 | 0.000 |
| 900050 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | stuck-midpath | 0.570 | 0.000 | 154.500 | strength | 1.000 | 0.000 |
| 900051 | 1.000 | 1.000 | 0.330 | 1.000 | deep-recovery | coiled-stuck | 0.830 | 0.000 | 163.000 | strength | 1.000 | 0.000 |
| 900052 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 29.000 | strength | 1.000 | 0.000 |
| 900054 | 1.000 | 1.000 | 0.000 | 1.000 | deep-recovery | shove-capped | 0.800 | 0.000 | 310.500 | strength | 1.000 | 0.000 |
| 900056 | 1.000 | 1.000 | 0.330 | 1.000 | free-run | shove-capped | 0.330 | 0.000 | 222.000 | strength | 1.000 | 0.000 |
| 900057 | 1.000 | 1.000 | 0.330 | 1.000 | deep-recovery | stuck-midpath | 0.830 | 0.000 | 207.000 | strength | 1.000 | 0.000 |
| 900059 | 1.000 | 1.000 | 0.000 | 1.000 | free-run | stuck-midpath | 0.200 | 0.000 | 170.000 | strength | 1.000 | 0.000 |
| 900062 | 1.000 | 1.000 | 0.330 | 1.000 | fight-escaped-near | shove-capped | 1.000 | 0.000 | 427.500 | strength | 1.000 | 0.000 |
| 900063 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | coiled-stuck | 0.710 | 0.000 | 227.500 | strength | 1.000 | 0.000 |
| 900064 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | coiled-stuck | 0.570 | 0.000 | 165.000 | strength | 1.000 | 0.000 |
| 900066 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 33.000 | strength | 1.000 | 0.000 |
| 900067 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 24.000 | strength | 1.000 | 0.000 |
| 900071 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | shove-capped | 0.710 | 0.000 | 157.500 | strength | 1.000 | 0.000 |
| 900073 | 1.000 | 1.000 | 0.670 | 1.000 | free-run | coiled-stuck | 0.290 | 0.290 | 67.000 | strength | 1.000 | 0.000 |
| 900075 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 30.000 | strength | 1.000 | 0.000 |
| 900079 | 1.000 | 1.000 | 0.330 | 1.000 | light-recovery | coiled-stuck | 0.500 | 0.000 | 76.500 | strength | 1.000 | 0.000 |
| 900080 | 1.000 | 1.000 | 0.000 | 1.000 | light-recovery | stuck-midpath | 0.800 | 0.000 | 164.000 | strength | 1.000 | 0.000 |
| 900081 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 26.000 | strength | 1.000 | 0.000 |
| 900082 | 1.000 | 1.000 | 0.330 | 1.000 | free-run | shove-capped | 0.330 | 0.000 | 209.500 | strength | 1.000 | 0.000 |
| 900084 | nan | 1.000 | nan | nan | cath-led-run | nan | 0.000 | 1.000 | 22.000 | strength | 1.000 | 0.000 |
| 900085 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | stuck-midpath | 0.860 | 0.000 | 155.000 | strength | 1.000 | 0.000 |
| 900086 | 1.000 | 1.000 | 0.670 | 1.000 | fight-escaped-near | shove-capped | 0.710 | 0.000 | 268.500 | strength | 1.000 | 0.000 |
| 900088 | 1.000 | 1.000 | 0.330 | 1.000 | light-recovery | shove-capped | 0.670 | 0.000 | 106.000 | strength | 1.000 | 0.000 |
| 900091 | 1.000 | 1.000 | 0.330 | 1.000 | light-recovery | stuck-midpath | 0.500 | 0.170 | 169.500 | strength | 1.000 | 0.000 |
| 900092 | 1.000 | 1.000 | 0.330 | 1.000 | deep-recovery | coiled-stuck | 0.830 | 0.000 | 142.500 | strength | 1.000 | 0.000 |
| 900093 | 1.000 | 1.000 | 0.670 | 1.000 | deep-recovery | stuck-midpath | 0.860 | 0.000 | 223.500 | strength | 1.000 | 0.000 |
| 900095 | nan | 1.000 | nan | nan | free-run | nan | 0.000 | 0.000 | 44.000 | strength | 1.000 | 0.000 |
| 900097 | 1.000 | 1.000 | 0.330 | 1.000 | fight-escaped-near | shove-capped | 0.830 | 0.000 | 259.500 | strength | 1.000 | 0.000 |

## How successes are won, by seed class (share of successes per phase group)

| name | seed_cls | cathled | deep-rec | fight | free | light-rec | n_succ |
|---|---|---|---|---|---|---|---|
| abl_best_host | easy | 0.000 | 0.000 | 0.000 | 0.722 | 0.278 | 18 |
| abl_best_host | frontier | 0.000 | 0.053 | 0.000 | 0.632 | 0.316 | 19 |
| abl_best_host | strength | 0.000 | 0.045 | 0.023 | 0.795 | 0.136 | 44 |
| car_best_host | easy | 0.167 | 0.278 | 0.296 | 0.037 | 0.222 | 54 |
| car_best_host | frontier | 0.020 | 0.196 | 0.529 | 0.059 | 0.196 | 51 |
| car_best_host | strength | 0.097 | 0.303 | 0.248 | 0.103 | 0.248 | 145 |
| tbv2_best_host | easy | 0.000 | 0.407 | 0.000 | 0.519 | 0.074 | 54 |
| tbv2_best_host | frontier | 0.000 | 0.500 | 0.031 | 0.312 | 0.156 | 32 |
| tbv2_best_host | strength | 0.000 | 0.466 | 0.034 | 0.397 | 0.103 | 58 |
| tbv3_ref_host | easy | 0.000 | 0.056 | 0.000 | 0.889 | 0.056 | 18 |
| tbv3_ref_host | frontier | 0.000 | 0.235 | 0.000 | 0.471 | 0.294 | 17 |
| tbv3_ref_host | strength | 0.000 | 0.182 | 0.023 | 0.614 | 0.182 | 44 |

## How failures end (phase of the failure)

| name | coiled-stuck | deep-recovery | doorstep-stall | fight-escaped-near | shove-capped | stuck-midpath |
|---|---|---|---|---|---|---|
| abl_best_host | 2 | 0 | 0 | 0 | 2 | 0 |
| car_best_host | 0 | 0 | 7 | 1 | 1 | 9 |
| tbv2_best_host | 22 | 2 | 2 | 1 | 51 | 33 |
| tbv3_ref_host | 4 | 0 | 0 | 0 | 2 | 0 |