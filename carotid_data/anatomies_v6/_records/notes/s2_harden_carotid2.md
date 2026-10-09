# s2_harden_carotid2 -- the carotid tool after the user's D19 / D20 (compose_v6 3.0.2)

Agent s2_harden_carotid2, 2026-10-04. State: `state/s2_harden_carotid2.md`. Builds on the first pass
(`notes/s2_harden_carotid.md`, compose_v6 2.0.1, which kept the right fork in v5's pose). Code:
`scratchpad/v6/carotid/` (runner `dr3.sh`, containers `s2car2_*`; test helpers `carotid/s3/`). Outputs:
`D:\Arjun\neve\carotid_data\v6_work\carotid\s3\` (`/out/s3` in the containers). Nothing under the repo's tracked
trees, anatomies_v2..v5, carotid_tools or topbrain_tools was written.

## Answer

1. **The carotid tool is compose_v6 3.0.2** (`carotid/compose_v6.py`, one versioned file; 3.0.1 / 3.0.0 / 2.0.1
   frozen; 3.0.1 = 3.0.0 + the light left ladder on non-final right variants; 3.0.2 = 3.0.1 + the siphon section gated
   by C0 like the fork -- see "3.0.2"), with route_guard 1.2 (C0 estimator; 1.2 = seam-2 skip for the siphon),
   route_baseline 1.1 (D13 library from the pilot's v4_no_tip.py, relative to the fork), orient_gates 1.2 (D12 one
   convention `backmapped_piece_ends`, fork mirror / chirality; self-test PASS) and the file verifier `verify_s3.py`
   1.1. Order: right fork -> right siphon candidates -> left fork (D1 ladder) with the ICA pair -> ECA tips -> every
   final check, each step gated.
2. **D19**: right forks re-posed by the per-patient Zenodo map; rungs R1 map (v5's cut, bend <= 45) -> R2 host-cut
   shift along the RCCA (prefix arch -> v5's seam 1 byte-identical, seam 2 kept at 130 mm, cut <= 72 mm) -> R3 bend
   <= 60 (flag) -> R4 bounded deviation (flag) -> R5 frame match / v5 (flag) -> R6 v5 unmirrored (flag).
3. **D20**: contralateral forks x-mirrored on both sides (own and substitutes), mirror + chirality in the xform; the
   mirrored fork's TopBrain ECA tip is re-posed by the search; D20 binds every left rung (unmirrored only after rung 6).
4. **Guards**: C0 (new; the redefined route invariant estimated pre-bake on the moved surface, seam zones excluded:
   fork >= 0.995 x v4 on every re-posed variant and the final anatomy; siphon >= 0.99 x v4 in the final check, flagged
   < 0.995 -- 3.0.2), C1 / C2 on the moved fork, D1 ladder rungs 1-6 incl. the ICA extensions, D4, D13.
5. **Pilots**: regression vs 2.0.1 identical 14 / 14; final code: dry 14 / 14 ok, written + verified from the files
   14 / 14 (base v5) and 14 / 14 (joint base); every forced right rung (R2-R6) and both left D20 paths written and
   verified; no ICA terminus > 40 mm posterior on the 14 (v5: 8, 2.0.1: 5).
6. **All-223 dry (base v5), as 3.0.2 decides it** (`dry223_v302`: the 3.0.1 run + the 2 anatomies 3.0.2's siphon gate
   re-decides): 223 / 223 ok. Right fork R1 map 121, R2 host-cut shift 83, R3 bend <= 60 13, R4 1, R5 v5 pose 5
   (19 flagged); 113 contralateral right forks, all mirrored. Left rung 1 201, rung 2 22 (no ICA
   extension needed); 212 map-posed; 102 contralateral left forks (v5: 101), all mirrored. **ICA terminus > 40 mm posterior: 3
   anatomies vs 108 in v5.** D13 v4_no_tip 33. Split projection EXIT 0 (none of 45 lowered). Siphon C0 (10 mm form):
   >= 0.990 on all 223 after 3.0.2 (min w_003_left mr_016_L 0.9904), 10 flagged 0.990-0.995. Details below.
7. **All-223 written + verified from the files** (3.0.1, base v5, the dry as plan): **223 / 223 verify ok** (0 blocking
   carotid or fork flags, 0 hard clearance failures, min 0.633 mm, max 1 mm join 52.8 deg, RCCA prefix identical, D20
   mirror == expected, forks on their raw centrelines <= 0.49 mm, E6 / C0 / AP / D13 from the files == the records);
   **write == dry on 223 / 223**. 58 rare-path figures. The 2 anatomies 3.0.2 re-decides: written by 3.0.2 and verified
   by verify_s3 1.1, 2 / 2 ok (`t11/`); verify_s3 1.1 also passes 14 / 14 pilots written by 3.0.2 (T10).
8. **Verdict**: the carotid tool is stage-2 ready for the joint pass; what the other stage-2 steps must cover is in
   "Open issues" (fork-VA contacts for the VA tool, the right-fork E6 blend for the bake, SOFA RCCA on all 223, D13
   rebuilds, the post-bake route invariant on the flagged siphons).

## Tests on the 14 pilots (outputs under carotid_data/v6_work/carotid/s3/)

| test | what | result |
|---|---|---|
| T1 regression | 3.0 with `--right-fork-pose v5 --no-mirror` (dry, base v5, plan s2/dry223_v201) vs 2.0.1's own all-223 records | 14 / 14 identical (left fork, both siphon R, ECA decisions, ok): `t1_reg_v5mode`, `t1_reg_cmp.txt`. (Since the AP preference, the 2.0.1 order also needs `--no-ap-prefer`.) |
| C0 validation | route_guard 1.1 estimate on v4 and v5 trees vs the baked right_route_detail | v5 / v4 ratio 1.0000 for fork and siphon on 14 / 14 (v5 changed only the VAs; the bake also shows ~1.00); absolute level above the bake's (real surface vs smoothed union): ratio-only use. `c0_validate.{log,json}`; the gated form (10 mm seam skip): 1.0 exactly on 14 / 14 (`c0_validate_skip.json`) |
| T2 / T2b / T2c | dry, base v5, defaults (T2 = first code; T2b = D20 binds every left rung; T2c = + AP preference) | T2c: 14 / 14 ok; right fork R1 map 10, R2 host-cut shift 4 (0.98, 0.99, 0.99, 6.97 mm; R1 failed only on the seam-1 bend > 45 at v5's cut: trimming the donor's cut-face inlet straightens its start tangent); every contralateral fork mirrored (10 right, 7 left); left rung 1: 12, rung 2: 2; ICA AP > 40 mm: 0 of 14 (RICA max 27.4, LICA max 35.1; 2.0.1 on the same pilots: RICA 73.7, 68.6, 54.7, 44.0, 43.2); D13 v4_no_tip 4 (2.0.1: 2); right ECA tips: re-posed by the search 2 (both on mirrored forks), skipped 3, none 9. T2 found the bug that a contralateral own left fork could be taken UNMIRRORED at rung 1 (w_003 mr_010) -> fixed (T2b); T2b's substitute left that anatomy's LICA 68 mm posterior -> the AP preference (T2c: 35 mm). |
| T3 | write, base v5 + `verify_s3.py` from the files | **14 / 14 ok**: RCCA prefix byte-identical (points and radii), host-cut extension on the host polyline (0.000 mm), right section arithmetic == v5's own offset, both forks back on their raw Zenodo centrelines through the WRITTEN xform incl. the mirror (median 0.03-0.04 mm), E6 contract on lower / lower_L / siphons / ECA tips, C0 / ICA AP / D13 recomputed from the files == the records, 0 blocking carotid or fork flags, 0 hard contacts; figures `t3_figs/` |
| T5 | write on the joint base (`v6_work/va/pilot`: v5 carotid side + the stage-1 v6 VAs) + verify_s3 | **14 / 14 ok** (same checks as T3); figures `t5_figs/`. Finding: the anatomically posed right fork can overlap a VA (soft): fork-only contacts in T3 w_018b (-3.4 mm at RCCA s 107, donor RVA) and T5 w_027 (-3.3 mm at s 83: the donor's ICA runs posterior into the host VA corridor, side view); the first 'RCCA_fork_soft' check had also spanned the siphon. Fixed in the final code: the fork checks cover seam 1 + 2 mm .. seam 2 only, and strict soft mode prefers a VA-clean R2 shift over a VA-touching R1 (both pure map poses); `right_fork_va_gap_mm` recorded. T3b / T5b re-run with the final code. |
| C0 probe / diag | seam-zone exclusion (`c0_probe_t3b.json`), refused-variant ratios (`t7_c0diag`) | see Decisions (C0) |
| refactor test | `finish` inside the right-variant loop (`t6_finish_test`) | k_008_right mr_027: R2 composed but its final C0 0.9938 < 0.995 -> next variants -> R4 map_dev_bend60 (45.3 deg, final C0 0.9996), ok; k_011 / w_051 unchanged |
| **FINAL (frozen 3.0.0)** | F_t3 write base v5 + verify; F_t5 joint base + verify; F_t4 forced rungs R2-R6 + left frame match + left D20 last resort, each written + verified (`vsum_final.txt`, `t4_summary.txt`) | **F_t3 14 / 14 ok, F_t5 14 / 14 ok** (same fork choices on both bases; write == the dry plan on 14 / 14): right R1 map 9 / R2 host-cut shift 5 (0.98-8.96 mm); left rung 1: 12, rung 2: 2; 16 contralateral forks, 16 mirrored; max 1 mm join 52.8 deg (a left ECA-tip join), min hard clearance 0.633 mm; forks and fork ECAs back on their raw Zenodo centrelines through the written mirror (median <= 0.046 mm, max 0.32); `orient_gates.py tree --fork-gates-placed`: 0 flagged carotid pieces (RICA, LICA, ECA tips) and 0 fork flags on both trees -- on base v5 the remaining tier-A flags are v5's VA tops (the VA tool's), on the joint base 0 tier A and 2 va_target (D8 band) (`og_F_t3`, `og_F_t5`); C0 from files 0.997-1.014 == the records; AP / D13 from files == the records; RCCA prefix identical; right section arithmetic == v5's offset (0.0 mm); VA soft contacts 5 anatomies (base v5) / 3 (joint base), fork-only ones w_018b (-3.4) / w_027 (-3.3). **F_t4 all verify ok**: skip R1 -> R2 (0.97 / 0.98 mm); skip R1-2 or R1-3 -> R4 (k_011 has no bend 45-60 candidate; dev poses with 12 / 16 mm shifts); skip R1-4 -> R5 (k_011 mirrored frame match, bend 0.06, RICA AP 38 mm; w_051 v5's pose); skip R1-5 -> R6 (k_011 v5 unmirrored, d20_applied False); left map poses off -> the own contralateral fork kept as `frame_match_mirrored`; left rungs 2-6 skipped on w_003 mr_010 -> the post-ladder D20 last resort (`zenodo_map_unmirrored`, d20_applied False; its LICA AP would be 6.5 mm vs 35 mm with the D20-compliant substitute) |
| left D20 last resort under 3.0.1 | F_t4_ld20 re-run (`F_t4_ld20_v301`, verify ok) | with left rungs 2-6 forced off, the post-ladder D20 last resort is now reached only on the final right variant: R6 (v5 unmirrored) + the own left fork unmirrored (`zenodo_map_unmirrored`), both d20_applied False -- by design (3.0.1 spends the last resorts on the final right variant); the normal run gives R1 map + a D20-compliant rung-2 substitute |
| light ladder (3.0.1) | w_003_right mr_001 (`t9_light`) | 3 map-posed right variants (R1, R2 1 / 2 mm; each with a single RICA candidate) fail their light left ladders, then the final v5 variant's full ladder passes at rung 1: R5 v5 pose (flagged), RICA AP 43 mm (flagged), 427 s (3.0.0's full ladders: estimated ~2 h) |
| selftest | `orient_gates.py selftest` (1.2) | PASS: v5 / v4 / v3 counts and per-piece values unchanged, D12 identity, new D20 chirality section (`gates/out/selftest_v1_2.txt`) |
| **T10 (3.0.2)** | dev copies (`s3/dev`, `run_dev.py`): dry 14 pilots with the 223 dry as plan vs the 3.0.1 records (`cmp_records` + every summary key, `cmp_summary.py`); write + verify_s3 1.1 on the 14; verify_s3 1.1 on three 3.0.1-written anatomies (old record form) | **PASS**: dry 14 / 14 IDENTICAL to the 3.0.1 records (cmp_records and every summary key; `logs/t10_cmp.log`, `logs/t10_cmps.log`), the no-skip siphon value == 3.0.1's c0_siphon_ratio on all 14, gated siphon 0.9965-1.0027 (no flag); write + verify_s3 1.1: 14 / 14 ok incl. the siphon gate and record match (`t10_302_verify.json`); verify_s3 1.1 on m_024_left mr_006, k_004_left mr_010, w_003_right mr_001 (3.0.1 records): 3 / 3 ok, m_024's siphon 1.0019 (`t10_vold_verify.json`). Installed after this (`install_302.sh`) |

### Final pilots per anatomy (frozen 3.0.0, base v5; AP = ICA-terminus offset posterior of its anatomical target, mm; `s3/pilot_table_final.md`)

v5 shipped: a terminus > 40 mm posterior in 8 / 14 pilots; 2.0.1 (v5 forks, re-posed siphons): 5 / 14; 3.0: 0 / 14.

| anatomy | right fork (rung, pose, mirrored, shift mm, bend deg, C0) | left (rung, decision, pose, mirrored) | ECA R | AP R / L 3.0 | AP R / L 2.0.1 | AP R / L v5 | baseline |
|---|---|---|---|---|---|---|---|
| k_004_left__topcow_mr_010 | R2 map_hostcut, True, 0.993, 40.36, 1.01413 | 1 kept zenodo_map, True | skipped | 21.17 / 26.57 | 43.96 / 26.59 | 29.86 / 13.93 | v4_no_tip |
| k_011_left__topcow_mr_012_L | R1 map, True, 0.0, 41.97, 1.0033 | 2 substituted zenodo_map, True | v6_search | 15.84 / 8.92 | 43.16 / 9.69 | 48.77 / -26.45 | v4_no_tip |
| m_006_left__topcow_mr_025_L | R1 map, True, 0.0, 31.03, 0.99846 | 1 kept zenodo_map, False | none | 2.66 / 17.06 | 15.2 / 6.87 | 33.07 / 13.04 | v4 |
| m_006_left__topcow_mr_027_L | R1 map, True, 0.0, 31.05, 0.99943 | 1 kept zenodo_map, False | none | -6.22 / 17.9 | 1.38 / 17.34 | 1.21 / 29.34 | v4 |
| w_003_left__topcow_mr_007_L | R1 map, True, 0.0, 23.82, 1.00089 | 1 kept zenodo_map, True | skipped | 0.53 / -5.87 | 68.6 / -18.28 | 73.45 / -25.73 | v4_no_tip |
| w_003_left__topcow_mr_010 | R1 map, True, 0.0, 24.0, 0.99784 | 2 substituted compose_frame_match, False | v6_search | -4.57 / 35.08 | 73.71 / 11.52 | 76.06 / -12.28 | v4_no_tip |
| w_018b_left__topcow_mr_002 | R1 map, True, 0.0, 27.12, 0.99929 | 1 kept zenodo_map, False | skipped | 14.94 / 20.73 | 54.7 / 20.73 | 22.0 / 11.93 | v4 |
| w_025_right__topcow_mr_021_L | R1 map, False, 0.0, 38.92, 0.99845 | 1 kept zenodo_map, False | none | 21.21 / 23.45 | 25.01 / 21.69 | 63.59 / -18.44 | v4 |
| w_027_right__topcow_mr_022 | R1 map, False, 0.0, 20.12, 0.99876 | 1 kept zenodo_map, False | none | 27.35 / 10.93 | -7.63 / 16.11 | -9.33 / 56.1 | v4 |
| w_034_right__topcow_mr_017_L | R2 map_hostcut, False, 0.987, 39.08, 1.00849 | 1 kept zenodo_map, False | none | 25.54 / 14.6 | -20.44 / 14.6 | -33.14 / 34.47 | v4 |
| w_035_left__topcow_mr_024_L | R2 map_hostcut, True, 8.959, 32.6, 0.99682 | 1 kept zenodo_map, True | none | -13.26 / -27.35 | 12.5 / -33.14 | 6.49 / -12.59 | v4 |
| w_047_left__topcow_mr_023 | R1 map, True, 0.0, 35.4, 0.99957 | 1 kept zenodo_map, True | none | -1.88 / -1.31 | 15.38 / -1.31 | 42.27 / 56.66 | v4 |
| w_050_left__topcow_mr_004_L | R2 map_hostcut, True, 0.975, 42.16, 1.00971 | 1 kept zenodo_map, True | none | -1.84 / 14.99 | 16.06 / 14.86 | -12.16 / 51.87 | v4 |
| w_051_right__topcow_mr_008 | R1 map, False, 0.0, 38.52, 0.99997 | 1 kept zenodo_map, False | none | 14.73 / -3.72 | 30.57 / -3.72 | 53.13 / 7.38 | v4 |

## Open issues (for the orchestrator / stage 2)

1. **Fork-VA contacts in the joint pass.** An anatomically posed right fork can lie in the host VA corridor (the
   donor's ICA running posterior: T5 w_027 -3.3 mm, T3 w_018b -3.4 mm against the v5 / stage-1 donor RVA). VA contacts
   are soft in the carotid step by design; the carotid tool only prefers a VA-clean R2 shift <= 10 mm. The VA tool's
   second pass (`--carotid-root` the carotid tree) must clear them (hard 0.6 mm at the end) or the joint pass fails
   there. All-223 on base v5: the moved right fork touches a VA in **17 anatomies** (16 under 3.0.2; min gap -3.36 mm,
   w_018b_left mr_002; list in `dry223_v302_report.txt`), any soft VA contact in 20 (19).
2. **Bake E6 for the right fork**: `xform.lower.blend_mm / blend_disp` at the start of lower_R (as lower_L); the
   right fork now moves in (almost) every anatomy. Mirrored forks: `xform.lower(_L).mirror = [-1, 1, 1]`.
3. **Route invariant post-bake** (checks_v6): each real section's own v4 wall detail on the moved surface; C0 is a
   pre-bake estimate only (ratio-calibrated, seam zone excluded). D13 v4_no_tip is now more frequent (every mirrored
   right fork re-poses its RECA tip): **33** v4 rebuilds (~110-145 s each). Watch list for the post-bake check: the
   siphons flagged by 3.0.2 (`c0_siphon_flag`, gated estimate 0.990-0.995; see "3.0.2") and the forks with an 11-12 mm
   host-cut shift (w_018_right x5, w_036_right x4, w_035_left x2: there the pre-bake C0 is not like-for-like).
4. **SOFA RCCA on all 223** (the training route moved).
5. **R3 (bend up to 60 deg, flagged)**: C0 refuses host-cut shifts that lower the fork section's estimated detail, so
   some anatomies that a shift would have kept <= 45 deg take R3 instead (e.g. k_008_right mr_001_L: 27 shifts refused,
   R3 50.2 deg at v5's cut). All-223: R3 13, R4 1, R5 4 (list in the All-223 section).
6. **ICA termini** (recorded, not gated): with anatomical forks the AP flags drop from 108 (v5) to 3; termini often
   sit 20-40 mm below their targets (the host route length to seam 2 is fixed at 130 mm) and, where the host's two
   seam-2 points are far apart, lateral of them (ica_spacing keeps the pair within 20 mm of native).

7. **D20 can cost AP on the left**: where the mirrored own left fork fails, the D20-compliant choice (a substitute) may
   leave the LICA terminus further back than the unmirrored own fork would (w_003 mr_010: 35 vs 6.5 mm). D20 is
   applied as binding; the trade-off is recorded (F_t4_ld20) for the user.
8. **Run time**: median ~15-20 s per anatomy, but an anatomy whose left side fails on the map-posed right variants
   costs several light ladders before the v5 fallback (minutes); the unsharded all-223 dry (split guard) took 4224 s
   of compose (~70 min); the write ran in 2 parallel shards with the dry as plan.
9. **Stage 2 must re-run the dry on the VA tool's base (W1)**: these records are on base v5. VA moves do not enter the
   hard checks or C0 (VAs are soft competitors), but the strict-mode VA-clean preference and the soft-contact lists
   can differ on W1, so the plan for the write must come from a dry on the same base.
10. **Siphon C0 at the pair level (not done)**: 3.0.2 checks the siphon in the final check, so a siphon loss to the
   other ICA (w_017_right mr_011) can only be answered by the next right-fork variant -- there it ends at v5's fork pose.
   Checking the siphon estimate inside `choose_pair` (another ICA pair, same fork) could keep the anatomical heading;
   it changes the pair search for every anatomy (a full re-run), so it is left for the orchestrator to decide.

## Decisions taken here (reversible; how)

- **Host-cut search moves the cut forward only** (d > 0: the donor CCA trimmed proximally, the host's own RCCA
  continued): the prefix arch -> v5's seam 1 stays byte-identical and seam 2 keeps its 130 mm arclength (graft_three's
  pin; the fork / siphon split of right_route_detail). A cut moved back would cut through v5's seam-1 calibre ramp and
  need donor CCA that v5 trimmed away. Undo / extend: `shifts` in `right_variants`.
- **Rungs after D19's two named steps** (R4 bounded deviation inside the fork gates, R5 frame match / v5, R6 v5
  unmirrored): added so every anatomy composes; all flagged. Undo: drop them from `right_variants`.
- **D20 for left forks too and for substitutes** (the orchestrator's extension, as instructed); the rule is the name
  side vs the route side. Undo: `--no-mirror` (both sides).
- **D20 binds every left rung**: a contralateral fork has no unmirrored variant inside the ladder; the own fork
  unmirrored is a post-ladder last resort (flag `d20_not_applied`). Found in T2 (w_003 mr_010 had taken its own
  contralateral fork unmirrored at rung 1). Undo: re-add the unmirrored fallback to `variants`.
- **A mirrored fork never carries its TopBrain ECA tip** (that would mirror the tip): the tip is re-posed by the ECA
  search with C1 / C2. Consequence: more D13 v4_no_tip baselines.
- **C0 (pre-bake estimate of the redefined route invariant), seam zone excluded, gate 0.995**: margin 0.5 % over the
  invariant's 0.99 for estimator noise (~0.3 % on rigid moves). Checked on the refusals (`s3/t7_c0diag`): w_002_right
  mr_013_L's best map variant reads 0.941 (a real 6 % loss to neighbour ownership: refused at any margin -> R5, v5's
  pose); k_008_right mr_001_L's best R2 shift read 0.9949 (just under) -> R3 (50.2 deg at v5's cut, C0 1.001): the user
  allows bends to 60 deg (flagged) while the route invariant is hard, so the margin stays. Undo: `--c0-min`,
  `--c0-skip`.
- **AP preference inside a rung** (never a gate): see the tool section. Undo: `--no-ap-prefer`.
- **VA-clean preference for the right fork limited to R1 vs R2 shifts <= 10 mm**: a soft contact does not justify
  trimming more real donor CCA. Undo: `VA_SHIFT_MAX_MM`.
- **Light left ladder on non-final right variants (3.0.1)**: rungs 1-4 without rung 4's anchor trim, no ICA extension
  (rungs 5 / 6) and no D20 last resort; the full ladder runs only on the final right variant (v5 / R6 pose, which
  composed in 2.0.1 for all 223). Why: where the left side fails on every map-posed right variant (w_003_right
  mr_001, stage 1's map-mode failure) each full ladder costs ~30 min (rungs 5 / 6 try ~25 extension lengths x 8
  siphon trims), ~2 h for the anatomy; the user's last resorts are meant for the left fork, not to rescue one of
  several right variants. Records 1-49 of the 223 dry were made by 3.0.0 and are unaffected (each passed its first
  ladder at rung <= 2). Undo: `light=False` in the right-variant loop.
- **Siphon C0 gated at 0.99, flagged < 0.995 (3.0.2)**: the invariant names every real section, so the siphon is gated
  like the fork, in the gated form (seam-2 zone skipped). The gate sits at the invariant itself, not the fork's 0.995:
  the siphon estimate spreads more on rigid re-poses, and a 0.995 gate would push sound anatomies into a different
  right-fork variant (the only lever the final check has). Gate value and flag are recorded per anatomy; the post-bake
  check decides. Undo: `--c0-siphon-min 0` (or 0.995 for the fork's margin).
- **Fork clearance from seam 1 + 2 mm** (v3's documented scope "route clearance from host_cut + 2"), the fork stretch
  only (the siphon has its own check).

## Stage-2 interface (for s2_integrate and the bake / checks)

- **Run** (eve-training-fixed, /repo read-only, `--cpus 4 --memory 6g`):
  `python3 /sp/v6/carotid/compose_v6.py dry --base <VA tree W1> --out <dry dir> --c0-cache <dir>` (sequential, NOT
  sharded: the split guard commits substitutions in name order), then `compose_v6.py write --base <W1> --plan <that dry
  dir> --out <tree C> --c0-cache <dir>` and `python3 /sp/v6/carotid/verify_s3.py <tree C> <W1> <out.json> --c0-cache
  <dir>`. The base must carry v5's carotid side (the VA tool's W1 does: recipe in notes/s2_harden_va.md); the carotid
  re-check after the VA tool's second pass is `verify_s3.py <W2> <W1> ...` (it re-measures from files only).
  v4's C0 estimates for all 223 are cached in `carotid_data/v6_work/carotid/s3/c0_v4/` (reusable; 3.0.2 adds the
  gated siphon estimate `siphon_s10` lazily, ~10 s per anatomy the first time). Timing on base v5: the dry ~70 min
  unsharded; the write ~25 min in 2 shards; verify_s3 ~3 s per anatomy.
  The write may run in chunks / in parallel containers once the dry is complete (`--plan` replays every earlier
  substitution into the split guard, so each written anatomy equals its dry record); the dry itself must be sequential.
- **Bake (E6)**: the right fork now carries a seam-1 blend like the left: `xform.lower.blend_mm` / `blend_disp`
  ([s, dx, dy, dz] = written - rigid centreline, s from seam 1) at the START of the lower_R section; lower_L as in 2.0.1.
  Rigid = `(src * mirror - origin) R^T + anchor`. A mirrored fork (`xform.lower(_L).mirror = [-1, 1, 1]`) is loaded by
  sdf_union.load_surface with the mirror (it re-derives outward normals; the triangle winding is reversed by the
  mirror). host_cut_mm / cca_mm moved by the host-cut shift and the blend (host_cut + cca + ica keeps v5's offset to
  the seam-2 arclength), so the lower_R / siphon_R / eca_R spans follow from provenance as before.
- **Checks**: the route invariant is each real section's own v4 wall detail on the moved surface (checks_v6); D13
  (`v6_carotid.summary.route_baseline` = v4 / v4_no_tip, route_baseline 1.1 relative to the fork) -- every mirrored
  right fork re-poses its RECA tip, so v4_no_tip rebuilds are more frequent than in 2.0.1 (33 of 223 on base v5);
  `orient_gates.py tree --fork-gates-placed` now gates both forks wherever placement = v6_map. Pre-bake, compose_v6
  3.0.2 gates the fork (C0 >= 0.995) and the siphon (gated form >= 0.99; `c0_siphon_flag` below 0.995) -- the
  post-bake exact-ownership check stays the authority; watch the flagged siphons and the 11-12 mm host-cut shifts.
- **SOFA**: the RCCA training route moved in every anatomy with a re-posed right fork: SOFA RCCA on all 223 (user).

## The tool: compose_v6 3.0.x (what changed from 2.0.1)

Composer order (all gated): right fork (D19 / D20 variants) -> right siphon candidates -> left fork (D1 ladder,
D11 / D20 poses) with the ICA pair -> ECA tips (C1 / C2 on the moved fork) -> assemble + every final check (`finish`).
A right-fork variant whose right siphon has no passing pose gives way to the next variant (up to RIGHT_TRY = 8); so
does a variant on which the whole left ladder fails, or whose composed anatomy fails a final check (e.g. the final C0
with the actual left fork, right siphon and whole RCCA: k_008_right mr_027 went R2 -> R4) -- up to RIGHT_LADDER_TRY = 3
ladder runs. The base's own right fork is always the last variant, so every anatomy composes; if nothing passes, the
first composed attempt is recorded (ok False). The split guard commits only the accepted anatomy's substitution.
3.0.1: the left ladder on a non-final right variant is LIGHT (rungs 1-4 without rung 4's anchor trim; no ICA extension,
no D20 last resort); the full ladder runs on the final right variant only.

- **D19 right fork** (`right_variants`, `place_right`): the base fork part (seam 1 .. seam 2 on the RCCA, its fork ECA,
  the TopBrain RECA tip) is mapped back into the raw Zenodo scanner frame with the base lower xform; the new pose is
  `branch = (src * M - origin) R^T + anchor` (M = the D20 mirror). Rungs:
  - R1 `map`: R = the per-patient map A (I or Rz(180)) exactly, at v5's host cut, seam-1 bend <= 45 blended on the donor
    CCA's first h mm (cubic Hermite, fork side; h = ceil(bend / 7.5), 3-8 mm);
  - R2 `map_hostcut`: the same pose with the host cut moved d = 1, 2, ... mm along the RCCA: the donor CCA trimmed by d,
    the host's own raw RCCA polyline continued by d past v5's seam 1 (radii smoothstep from v5's seam-1 calibre to the
    trimmed donor's), so seam 2 keeps its arclength (130 mm: graft_three's pin; the fork / siphon split of
    right_route_detail) and the cut stays in graft_three's documented [15, 72] mm with >= 5 mm of donor CCA before the
    bifurcation and the ECA branch beyond the blend; shifts up to 30 mm, smallest first;
  - R3 `map_bend60`: bend 45-60 deg blended (v5's cut, then the shifts) -- flagged;
  - R4 `map_dev` / `map_dev_bend60`: a bounded deviation from the map inside the fork gates (heading <= 44, tilt <= 39),
    bend <= 45 then <= 60, cheapest first (dev^2 / 10 + 0.25 bend + 0.05 shift; 400 poses per bound) -- flagged;
  - R5: the frame-matched x-mirrored fork (contralateral) or v5's own pose (same side) -- flagged;
  - R6 (contralateral only): v5's unmirrored pose -- flagged, D20 not applied.
  Fork-level gates (R1-R5): bend bound, ECA branch beyond the blend, seam-1 1 mm join <= 60, route kink <= 60 (from 8
  points before seam 1 to seam 2), hard clearance >= 0.6 of the route from seam 1 + 2 mm (v3's documented scope) and of
  the fork ECA against the host branches and the base left side, self clearance; VA contacts soft (strict mode: ranked
  last inside a rung); C0 >= 0.995.
- **Prefix**: the RCCA arch -> v5's seam 1 is never rewritten (byte-identical points and radii); a host-cut shift only
  appends the host's own polyline. A cut moved back (d < 0) is not used: it would cut through v5's seam-1 calibre ramp
  and need donor CCA that v5 trimmed away.
- **D20** (`d20_mirror`, `chirality_record`): mirror iff the fork's name side differs from the route side, own fork or
  substitute, right and left; xform.lower(_L).mirror = [-1, 1, 1] (sdf_union.load_surface: `src * mirror`, then the
  proper rotation; it re-derives outward normals, so a mirrored surface's reversed winding is handled at load) plus a
  `chirality` record. A mirrored fork cannot carry its TopBrain ECA tip (the composed map would mirror the tip): the tip
  is re-posed by the ECA search (C1 / C2 / D4) -- `tip_research` in the record.
- **D20 binds every left rung**: a contralateral left fork (own or substitute) only has mirrored variants inside the
  ladder (D11 map poses of the mirrored fork, then the frame-matched mirrored fork); the own fork unmirrored (2.0.1's
  variants) is tried only after rungs 1-6 all failed, flagged `d20_not_applied`.
- **C0** (route_guard 1.1 `SectionDetail`; the redefined route invariant estimated before the bake): the fork section's
  real-surface vertices on the MOVED surface (capsule, exact RCCA owner, depth > 0.3 mm off the RCCA tube) as a share of
  the RCCA-owned area of the fork segment [host cut + 10 mm, 130), every non-VA branch competing, the RECA left out (C1
  judges the tip); ratio to the same estimate on v4's tree (same 10 mm skip past v4's cut) >= 0.995. Gates every
  re-posed right variant (route cut at seam 2) and the final anatomy (whole RCCA); the full-segment ratio (from the cut)
  and the siphon section's ratio are recorded. Why the 10 mm skip (`C0_SKIP_MM`; probe `s3/c0_probe_t3b.json`):
  pre-bake, the rigid donor inlet at a bent seam 1 protrudes from the host tube and reads as relief (the bake's E6 morph
  reshapes exactly that zone), so the full-segment ratio is biased up (1.01-1.16 on the pilots, 1.31 with an 18 mm
  shift); with the skip, rigid R1 moves read 0.998-1.003 (estimator noise ~0.3 %), and host-cut shifts show their real
  effect on the section's make-up (w_025 9 mm shift 0.928, w_035 7 mm 0.988 -> such shifts are now refused). Absolute
  values run higher than the baked measure (real surface vs union), so only the ratio is used; v4 vs v5 reads 1.0000 on
  14 / 14 pilots.
- **C0 siphon (3.0.2)**: the siphon section (s >= 130 + 10 mm, its seam-2 zone left out of both estimates) is gated in
  the final check at >= 0.99 x v4 (C0_SIPHON_MIN, the invariant itself) and flagged below 0.995 (`c0_siphon_flag`);
  see "3.0.2" for why the siphon gets no 0.5 % margin.
- **C1 / C2** measured on the moved fork (RouteGuard from the composed provenance and curves; C2 against the composed
  RCCA past the new host cut). In v5 mode they are exactly 2.0.1's.
- **D13** (route_baseline 1.1): the v4 RECA tip is compared with v6's RELATIVE TO ITS FORK (to_fork_frame incl. the
  mirror): carried rigidly by a re-posed fork = kept (baseline v4); re-posed (e.g. on a mirrored fork) or skipped while
  v4's tip ran along the ICA = v4_no_tip.
- **AP preference (new, never a gate)**: inside a ladder rung, a passing left attempt whose pair leaves an ICA terminus
  > 40 mm posterior of its anatomical target is held (like 2.0.1's VA-clean hold) while the rung's other candidates are
  tried; the hold with the smallest (VA touch, AP flag) key is taken if nothing clean passes. Rung order unchanged.
  `--no-ap-prefer` restores 2.0.1's order.
- **Records** (`summary`): right fork pose / rung / flag / placement / mirrored / contralateral / d20_applied /
  seam-1 bend / blend / host-cut shift / host cut / deviation / yaw vs map / C0 ratios (fork hard, fork with VAs,
  siphon); left fork rung / decision / lower / pose / pose flag / mirrored / contralateral / d20_applied / seam-1 bend /
  tubed ICA; ICA AP offsets both sides + flags (R, L, either); skipped ECA tips; ECA right decision (v5 / carried /
  v6_search / skipped / none); route baseline. Provenance: xform.lower {R, origin, anchor, mirror, placement (v6_map /
  v6_frame_mirrored / v5), v6, blend_mm, blend_disp, chirality}; host_cut_mm / cca_mm moved (host_cut + cca + ica keeps
  v5's own offset to the seam-2 arclength; the bake slices lower_R / siphon_R by it), v6_host_cut_shift_mm,
  v6_cca_trim_mm, v6_host_ext, v6_right_fork; xform.lower_L {mirror, placement, chirality}.

## All-223 dry run (3.0.0 for the first 49 names, 3.0.1 for the rest -- identical for those 49; base v5, unsharded; `s3/dry223_v300/`, `s3/dry223_v300_report.txt`, `_summ.json`, `_table.tsv`)

Records 1-49 by 3.0.0, the rest after the 13:18 resume under 3.0.1; ended 14:24; 4224 s of compose in all (median 16 s per anatomy, max 432 s:
w_003_right mr_001, the light-ladder case). Numbers from `dry223_v300_report.txt` / `_summ.json`, `right_stats_223.json`,
`notes_223_lines.txt`.
**Under the installed 3.0.2** the numbers differ in two anatomies only (see "3.0.2"): w_017_right mr_011 R1 -> R5
(v5's pose, flagged), w_050_left mr_005 host-cut shift 0.97 -> 1.97 mm. So: R1 121, R5 5, right flags 19, fork-VA
contacts 16, any soft VA contact 19, flagged anatomies 31 (`dry223_v302_report.txt`); everything else below unchanged.

- **223 / 223 ok, 0 errors, 0 not ok.**
- **Right fork (D19 / D20)**: R1 map (v5's cut, bend <= 45) **122**; R2 map + host-cut shift **83**; R3 bend 45-60
  (flagged) **13**; R4 bounded deviation (flagged) **1** (k_008_right mr_027, 45.3 deg); R5 v5's pose (flagged) **4**;
  R6 0. So 205 / 223 sit at the anatomical heading with a seam-1 bend <= 45, 219 / 223 are map-posed.
  - Bend > 45 at v5's cut (the user's "~55"): **95**. The host-cut search resolved 83 of them; shifted cuts 87 (incl.
    k_005_right x4 at R3 with 3-4 mm), median 2.0 mm, max 11.96 (histogram, mm: 1: 39, 2: 6, 3: 8, 4: 16, 5: 4, 9: 2,
    10: 1, 11: 5, 12: 6).
  - R3 (13): k_005_right x4 (59.5-59.8 deg), k_008_right mr_001_L (50.2), w_002_right mr_006 / mr_016_L / mr_024
    (55.5-55.7), w_022_right x4 (49.7), w_035_left mr_015_L (53.6). Right seam-1 bend over all: median 39.0, max 59.8.
  - R5 (4): w_002_right mr_013_L and mr_022 (every map variant refused by C0 at 0.940-0.946: a real ~6 % loss of the
    fork's relief to neighbour ownership), w_003_right mr_001 (the left side fails on every map-posed right variant),
    w_036_right mr_018_L (R2 shifts leave no passing right siphon; R3 refused by C0 at 0.989-0.991).
  - Mirrored 113 = contralateral 113 (every "_left" fork on the right route); D20 applied on 223 / 223.
  - C0 fork (gated form): min 0.99512, median 1.00028; 280 variants refused by C0 in 28 anatomies; 5 anatomies
    retried after a final-check failure and then passed. Caveat: with a long host-cut shift (w_018_right x5,
    w_036_right x4: 11 mm) the gated C0 reads 1.12-1.15 -- the v6 segment starts 11 mm further down the fork than
    v4's, so the estimate is not like-for-like there (higher = more relief share, not a loss); the post-bake check
    decides.
- **Left fork (D1 / D11 / D20)**: rung 1 (own fork) **201**, rung 2 (substitute keeping the split) **22**; no rung >= 3,
  no ICA extension or tube needed (tubed_ica_mm 0 on all 223). Pose: Zenodo map **212**, own contralateral fork
  frame-matched mirrored 3, substitute frame-matched 4, v5's 4 (11 flagged). Mirrored 102 = contralateral 102; D20
  applied on 223 / 223. Left seam-1 bend median 24.2, max 44.97.
- **ICA-terminus AP offset (D11; flag > 40 mm posterior)**: **3 anatomies** (w_003_right mr_001 RICA 43.0 -- R5, v5's
  fork pose; w_027_left mr_022 RICA 45.4; w_027_right mr_016 RICA 43.1) vs **108 in v5** (same function, same 223:
  RICA 66, LICA 59; > 20 mm: 189). RICA median 7.1 mm (p90 26.0), LICA 4.7 (p90 22.3).
- **ECA tips (D4, C1 / C2)**: right tip carried with its fork 6, re-posed by the search 27, v5's 1, skipped 18, none
  171; 51 anatomies have at least one skipped ECA tip (down-pointing / anchor_low, TopBrain ECA piece < 10 mm, no ECA
  centreline, or no pose passing the route guard / v6 gates).
- **D13 route baseline**: v4 190, v4_no_tip 33 (33 v4 rebuilds for the post-bake check).
- **VA contacts (soft by design; the VA tool's second pass clears them)**: the moved right fork touches a VA in 17
  anatomies (min gap -3.36 mm, w_018b_left mr_002); any soft VA contact in 20.
- **Split projection** (`v5_split.py --baseline split_v4.json` on the projected tree): **EXIT 0**, none of the 45
  quantities lowered (strict k16_div6x6 and k17_div6x6 kept +1 / +2).
- **Flagged anatomies: 30** (right rung >= 3: 18; left pose not the map: 11; AP > 40: 3; overlaps counted once; list
  in `dry223_v300_report.txt`).

### All 223 written + verified from the files (`s3/w223/`, base v5, the dry as plan)

All 223 anatomies written by 3.0.1 with `--plan dry223_v300` (chunk 1 = the first 65 names in one container; chunk 2 =
the other 158 in two parallel shards, `chain_w223b.sh`) and re-measured from the files by `verify_s3.py` (1.0):

- **223 / 223 written, 223 / 223 verify ok** (`w223_verify_chunk1.json`, `w223_verify_chunk2_{0,1}.json`, merged
  `w223_vsum.json`): 0 blocking carotid or fork flags (orient_gates 1.2 tree mode, fork gates on every v6_map fork),
  0 hard clearance failures (min hard clearance 0.633 mm), max 1 mm join 52.8 deg (w_003_left mr_010, a left ECA-tip
  join), RCCA prefix byte-identical on 223, host-cut extensions on the host polyline (0.000 mm), right section
  arithmetic == v5's offset (max 0.0 mm), D20 mirror == expected on both sides of all 223, forks back on their raw
  Zenodo centrelines through the written xform incl. the mirror (max 0.49 mm), 0 E6 contract failures, C0 / ICA AP /
  D13 recomputed from the files == the records on all 223. Soft (VA) contacts in 20 anatomies (the VA tool's).
  Remaining tier-A orientation flags on base v5 are v5's VA tops only (RVA_top 94, LVA_top 94: the VA tool's job).
- **Write == dry on 223 / 223** (`s2/cmp_records.py`: left fork decision / lower / rung, both siphon poses, both ECA
  decisions and poses, ok; log `logs/cmp223.log`).
- Figures of every rare path (right rung >= 3, left pose not the map, AP flag, fork-VA contact: 45) + 10 random + the
  user's named k_011_left mr_013_L / 016_L / 023_L: `rare_figs/` (58 PNGs, front + side vs v5, ICA targets starred;
  names and reasons `rare_names.txt`, `logs/rare_pick.log`).


## 3.0.2: the siphon section gated too (C0 siphon)

Why: the route invariant is "each real section keeps >= 0.99 of its own v4 wall detail"; 3.0.0 / 3.0.1 gated only
the fork section and recorded the siphon's estimate without a seam skip. On the 223 that recorded value read
0.983-1.15; its one value < 0.99 (m_024_left mr_006, 0.983) prompted a probe (`s3/siphon_c0_probe.py`, on the
written trees vs v4, seam-2 zone 0 / 5 / 10 mm left out of both estimates; `siphon_c0_probe.json`):

| seam-2 zone left out | n | min | p05 | median | max | < 0.99 | < 0.995 |
|---|---|---|---|---|---|---|---|
| 0 mm (3.0.1's record) | 223 | 0.983 | 0.998 | 1.027 | 1.221 | 1 (m_024_left mr_006) | 8 |
| 5 mm | 223 | 0.978 | 0.996 | 1.002 | 1.156 | 2 | 7 |
| 10 mm (3.0.2's gate form) | 223 | **0.972** | 0.995 | 0.999 | 1.155 | **2** (w_017_right mr_011 0.972, w_050_left mr_005 0.987) | 12 |

(`siphon_c0_all.json`, merged by `siphon_c0_merge.py`; the probe's 0 mm value equals the 3.0.1 record and its 10 mm value
equals 3.0.2's in-memory estimate wherever both exist.)

- m_024_left mr_006's 0.983 is the seam-2 zone (0.9986 at 5 mm, 1.0019 at 10 mm): no loss of the siphon's own detail.
- The no-skip value hid two values below 0.99: **w_017_right mr_011** (10 mm: 0.972) -- the competitor diagnostic
  (`siphon_c0_diag.py`, `siphon_c0_diag2.json`: each hard competitor left out in turn) shows the **LCCA (the left ICA
  of the chosen pair) taking 14.4 mm2 of the right siphon's wall in v6 (0 in v4)**; without it the ratio is 0.996.
  This is exactly the loss the invariant forbids. **w_050_left mr_005** (0.987): no competitor moves > 0.05 mm2; the
  owned area is +0.3 %, i.e. the segment boundary / blended arclength, still below 0.99.
- The 4 lowest of the rest (w_003_left mr_016_L 0.990, w_008_right mr_020 0.990, m_030_right mr_016_L 0.994,
  w_007_right mr_023 0.995): no competitor moves > 0.05 mm2 either; v6 owns 0.15-0.47 % more area than v4
  (`siphon_c0_diag.json`) -- boundary / arclength effects of ~0.5-1 %, i.e. the estimator's spread.
- The other 10 below 0.995 (0.990-0.995 at 10 mm) pass the 0.99 gate and carry `c0_siphon_flag` (watch list for the
  post-bake check; names in `siphon_c0_all.json` s10.lt_0995).

What 3.0.2 changes (route_guard 1.2: `skip_mm` also applies to the siphon segment; compose_v6 3.0.2; verify_s3 1.1):
the final check gates the siphon's gated-form estimate (s >= 130 + 10 mm) at **C0_SIPHON_MIN = 0.99** (the invariant
itself) and flags it below 0.995 (`c0_siphon_flag`); the fork keeps its 0.995 margin. Why not 0.995 for the siphon:
on rigid re-poses its estimate spreads about 1 % (10 mm form: p05 0.995; 10 anatomies read 0.990-0.995 with no
competitor change -- boundary / arclength effects; the fork's, with its skip, ~0.3 %), and a 0.995 gate would refuse
sound siphons -- worse, a final-check failure moves to the next RIGHT-FORK variant, which is the wrong lever for
a siphon. `c0_siphon_ratio` is now the gated form; the 3.0.1 value is `c0_siphon_ratio_full_segment`. v4's gated
siphon estimate is added to the C0 cache lazily (`siphon_s10`). Undo: `--c0-siphon-min 0`.

Tests (T10, table above): the 3.0.2 dry on the 14 pilots is identical to 3.0.1 (records and every summary key); 3.0.2
write + verify_s3 1.1 pass 14 / 14; verify_s3 1.1 also passes on 3.0.1-written trees (the no-skip record form). The
gated siphon value of the tool equals the probe's 10 mm value from the files (e.g. k_011_left mr_012_L 0.9976, k_004_left
mr_010 0.99848). Installed as `carotid/compose_v6.py` 3.0.2 / `route_guard.py` 1.2 / `verify_s3.py` 1.1 (the replaced
versions frozen; checksums `s3/md5_final.txt`).

**What 3.0.2 changes on the 223 (T11, `s3/t11/`)**: the siphon gate sits in the final check only, so 3.0.2's flow is
3.0.1's up to that check; every anatomy whose accepted 3.0.1 composition reads >= 0.99 (221 by the probe) keeps its
3.0.1 record, and both re-decided anatomies keep their rung-1 own left fork, so the split guard state of later names
is unchanged. The installed 3.0.2 re-run (dry with the 223 plan, write, verify_s3 1.1, figures) on the two:
- **w_017_right mr_011**: R1 map (LICA steals the siphon, 0.972) -> the next map variants fail the same way -> **R5,
  v5's fork pose (flagged)**, siphon 0.99994, fork C0 1.0, RICA AP 15.6 mm; no fork-VA contact any more (R1 had -0.48
  mm). A pair-level siphon check (another ICA pair, keeping the map fork) could save the anatomical heading here; not
  done: it would change the pair search for every anatomy and need a full re-run (open issue 10).
- **w_050_left mr_005**: R2 host-cut shift 0.97 -> **1.97 mm** (still map-posed, bend 42.7), siphon 0.99847, fork C0
  1.023.
- verify_s3 1.1: 2 / 2 ok (`t11/verify.json`); figures `t11/figs/`. `dry223_v302/` = the 3.0.1 records with these two
  replaced; `dry223_v302_report.txt` / `_summ.json` (its "c0 siphon" line still mixes the two record forms -- use
  `siphon_c0_all.json`), split projection `split_check_223_v302.log`: **EXIT 0, every one of the 45 quantities equal to
  the 3.0.1 projection** (none lowered vs split_v4).

### Per-anatomy record (provenance `v6_carotid.summary`; one TSV row each in `s3/dry223_v302_table.tsv` -- 3.0.2 -- and `s3/dry223_v300_table.tsv` -- 3.0.1)

right_fork_pose (map | map_hostcut | map_bend60 | map_dev | map_dev_bend60 | frame_match_mirrored | v5 | v5_unmirrored),
right_fork_rung (1-6), right_fork_flag (rung >= 3), right_fork_mirrored, right_fork_contralateral,
right_fork_d20_applied, host_cut_shift_mm, right_seam1_bend_deg, right_blend_mm, c0_fork_ratio (gated form),
c0_fork_ratio_full_segment, c0_siphon_ratio (3.0.2: gated form; 3.0.1: no skip), c0_siphon_ratio_full_segment +
c0_siphon_flag (3.0.2), right_fork_va_gap_mm, fork_rung (left D1 rung), left_fork_decision,
left_fork_lower, left_fork_pose (zenodo_map | frame_match_mirrored | v5 | compose_frame_match | *_unmirrored),
left_fork_pose_flag, left_fork_mirrored, left_fork_d20_applied, left_seam1_bend_deg, tubed_ica_mm, ica_ap_post_mm
(RICA / LICA, posterior +) + ica_ap_flag (either > 40 mm), eca_right (carried | v5 | v6_search | skipped | none),
skipped_eca_tips, route_baseline (v4 | v4_no_tip).

## Files

- Tool: `carotid/compose_v6.py` **3.0.2** (frozen copy `carotid/s3/compose_v6_frozen_3.0.2.py`); the all-223 dry used
  3.0.0 (records 1-49; `s3/compose_v6_frozen_3.0.0.py`, md5 6bfcac11) and 3.0.1 (the rest and the w223 write;
  `s3/compose_v6_frozen_3.0.1.py`, md5 eeab0fb8); checksums in `carotid/s3/md5_final.txt`. Libraries
  `carotid/route_guard.py` (1.2: siphon skip; 1.1: + SectionDetail / C0), `carotid/route_baseline.py` (1.1: D13
  relative to the fork), `gates/orient_gates.py` (1.2: fork mirror / chirality, self-test D20); file verifier
  `carotid/verify_s3.py` (1.1: + siphon C0); runner `carotid/dr3.sh` (containers `s2car2_*`).
- Frozen previous versions: `carotid/compose_v6_v201.py`, `route_guard_v1_0.py`, `route_guard_v1_1.py`,
  `route_baseline_v1_0.py`, `verify_s2_v201.py`, `verify_s3_v1_0.py`, `gates/orient_gates_v1_1.py`; self-test output
  `gates/out/selftest_v1_2.txt`. The 3.0.2 development copies stay in `carotid/s3/dev/` (`run_dev.py` ran them).
- Test helpers `carotid/s3/`: `chain_final.sh` (final T3 / T5 / T4), `chain_pilot.sh` (earlier T3 / T5), `chain_223.sh`
  (all-223 dry + v5 AP + report + split), `chain_w223b.sh` (all-223 write, chunk 2, sharded) + `chain_w223_post.sh`
  (merged verify summary `vsum223.py`, write == dry `s2/cmp_records.py`, rare names `pick_rare.py`, figures
  `fig_s3.py` with NAMES_FILE, siphon probe), `chain_t10.sh` (3.0.2 tests), `chain_t11.sh` (3.0.2 on the 2 re-decided),
  `chain_v302.sh` (dry223_v302 + report + split), `install_302.sh` (freeze + install), `cmp_summary.py`,
  `siphon_c0_probe.py` (claims-sharded), `siphon_c0_merge.py`, `siphon_c0_diag.py`, `chain_rare.sh` (superseded by the all-223 write), `c0_validate.py`,
  `c0_probe.py`, `c0_cache.py`, `report_s3.py`, `right_stats.py`, `notes_223.py`, `ap_tree.py`, `pilot_table3.py`,
  `t4_summary.py`, `vsum.py`, `show3.py`, `tries3.py`, `peek_prov.py`, `peek2.py`, `probe1.py`.
- Outputs `carotid_data/v6_work/carotid/s3/`: `c0_v4/` (v4's C0 estimates, all 223, reusable; + `siphon_s10`),
  `c0_validate.*`, `c0_probe_t3b.json`, `t1_reg_v5mode` + `t1_reg_cmp.txt`, `t2*_dry_pilot` (T2 .. T2d), `t3*` / `t5*`
  (earlier code), `F_t3_*`, `F_t5_*`, `F_t4_*` (final code), `dry223_v300/` + `dry223_v300_report.txt` + `_summ.json` +
  `_table.tsv`, `right_stats_223.json`, `notes_223_lines.txt`, `ap_v5.json`, `split_tree_223_v300/`,
  `split_check_223_v300.{json,log}`, **`w223/`** (all 223 written, base v5) + `w223_verify_chunk1.json` /
  `w223_verify_chunk2_{0,1}.json` + `w223_vsum.json`, `rare_names.txt` + `rare_figs/`, `siphon_c0_probe{,_b}.json` +
  `siphon_c0_all.json`, `siphon_c0_diag{,2}.json`, `t10_302_*` (3.0.2 tests), `t11/` (the 2 re-decided: dry, write,
  verify, figs), **`dry223_v302/`** + `dry223_v302_report.txt` + `_summ.json` + `_table.tsv`, `split_tree_223_v302/`,
  `split_check_223_v302.{json,log}`, `logs/`, `markers*`.
