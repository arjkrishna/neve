# v5_assign: donor records, dry run, one-to-one assignment, unbaked anatomies_v5

Agent prefix `v5_assign`. State `v5_va/state/v5_assign.md`. One container of my own (`v5_assign_w`,
eve-training-fixed, `--cpus 8`), removed at the end. No git. No tracked repo file and nothing in
`carotid_data/anatomies_v4` was modified. `V5_BUILD_LOG.md` was not edited: the orchestrator adds the milestone line.

## Answer so far (final, 15:16)

1. **Donor records.** All 27 eligible pairs (25 primary and 2 reserve) were built into a new registry folder,
   `carotid_data/va_sources/v5_reg/donors_v5/`. The graft_reg test records in `v5_reg/donors/` are untouched.
   - Each side is built from the wall, QC and trace files the final tally graded (`results/v5_eligible_final.json`).
     That covers the 15 "donor-tool caveat" sides (ct_seg4v2 walls; ts / tsba_v1r traces).
   - 54 sides; chirality +1 on all; the copied surface's md5 equals the eligible list's md5 on all 54.
   - 53 / 54 sides pass every check (`results/v5_assign_records.json`). The one failure is expected: reserve Song 138 R
     has a 132 mm line that ends at C2 (no C1, no dura), so no splice in V3 is possible.
   - VMR 0248, ia172_058 R and ia172_141 L reproduce the earlier test records exactly (surface md5, centreline,
     take-off). ia172_015 L changed, as intended: it now uses the ct_seg4v2 wall.
2. **Dry run.** 27 units × 223 anatomies (`v5_reg/dry_v5/`, 14:41–14:48). Over the 12,042 unit-anatomy-sides,
   **8,688 pass every gate (72.2 %)**, with 0 errors (`results/v5_assign_dry_stats.json`).
   - Base-candidate pass rates per gate: rotation 1.000, origin turn 0.834, course RMS 0.870, take-off band 0.942,
     rise 0.952, splice window 1.000, calibre 0.813, join 0.820, tip 0.996, clearance 0.551.
   - Sides that pass nowhere: ia172_069 L (0 / 223) and reserve Song 138 R (0 / 223, no V3 window).
     - 069 L fails because its V1 runs nearly horizontal for its first ~30 mm: the 10 mm chord elevation is 2°.
       The rise gate (≥ 15°) therefore fails; the base course RMS has a median of 19.9 mm against the 20 mm bar.
   - Weak sides, limited by the take-off band (native take-off 122–149°): ia172_159 L (12), Song 032 L (46),
     ia172_039 L (103). ia172_071 R reaches 59 (course RMS and calibre).
3. **Assignment** (`graft_v5_assign.py` 1.1, exact lexicographic MILP; `v5_reg/plan_v5.json`,
   `results/graft_reg_assign_v5.json`).
   - **25 primary pairs go to 25 TopBrain patients, one to one. No reserve is used.**
   - **RVAs replaced: 212 / 223. LVAs replaced: 219 / 223** (431 of 446 VAs).
   - **Anatomies v5 on both sides: 208. On one side: 15** (4 with only the RVA replaced, 11 with only the LVA replaced). **On neither side: 0.**
   - The 15 one-sided anatomies keep v4's other VA and are flagged.
   - Re-verifying the assembled anatomies changed nothing (1 round, 0 changed).
4. **Written** (not baked): `carotid_data/anatomies_v5/<v4 name>` for all 223. Each has Centrelines_comb,
   provenance.json and LICENSE files. `results/v5_assign_written_check.json` passes 223 / 223:
   - every non-VA centreline is byte-identical to v4, and so is every kept VA;
   - every replaced RVA keeps v4's subclavian prefix exactly;
   - `xform.vaC_*` is present exactly when a side is replaced, and the surface md5 matches;
   - every other provenance and xform key equals v4's;
   - the follow handedness rule holds, and each unit serves exactly one patient.
5. **Split check** (`v5_split.py`, extended; `results/split_v5.json`):
   - **EXIT 0.** All 45 quantities are compared against `results/split_v4.json` and none is lowered.
   - Extra (V:) donors: 25 donors in 25 TopBrain patients, none spanning more than one patient,
     **one_to_one_nested True** under the v3, v4side and strict definitions.
   - v3 roster: 153 / 16 / 54. Maximum split: 183 / 16 and 181 / 17. Strict roster: 127 / 16 / 80; maximum
     174 / 16 and 172 / 17. All are unchanged from v4.

## 1. Patient → donor pair table

Rows are ordered by the patient's number of anatomies. "cf" is the tally's contact-free donor-stretch union
> 0.3 mm (R / L). RVA / LVA are the counts replaced out of the patient's anatomies.
Full table: `results/v5_assignment.json` → `table`.

| TopBrain patient | anat. (nat / mir) | donor pair | source | cf R / L | RVA | LVA | both / one / none |
|---|---|---|---|---|---|---|---|
| topcow_mr_005 | 16 (8 / 8) | song_AHMU1218108 | Song | 0.062 / 0.119 | 16 | 16 | 16 / 0 / 0 |
| topcow_mr_011 | 16 (8 / 8) | ia172_164 | 172 | 0.127 / 0.091 | 16 | 16 | 16 / 0 / 0 |
| topcow_mr_016 | 16 (8 / 8) | ia172_015 | 172 | 0.117 / 0.039 | 16 | 16 | 16 / 0 / 0 |
| topcow_mr_020 | 16 (8 / 8) | song_AHMU1218002 | Song | 0.052 / 0.024 | 16 | 16 | 16 / 0 / 0 |
| topcow_mr_010 | 15 (7 / 8) | ia172_159 | 172 | 0.090 / 0.051 | 10 | 14 | 9 / 6 / 0 |
| topcow_mr_023 | 15 (7 / 8) | ia172_190 | 172 | 0.056 / 0.030 | 15 | 15 | 15 / 0 / 0 |
| topcow_mr_024 | 15 (7 / 8) | song_AHMU1218109 | Song | 0.037 / 0.009 | 15 | 15 | 15 / 0 / 0 |
| topcow_mr_026 | 15 (7 / 8) | ia172_149 | 172 | 0.116 / 0.063 | 14 | 15 | 14 / 1 / 0 |
| topcow_mr_001 | 12 (8 / 4) | ia172_165 | 172 | 0.136 / 0.106 | 11 | 12 | 11 / 1 / 0 |
| topcow_mr_022 | 10 (7 / 3) | ia172_192 | 172 | 0.058 / 0.156 | 10 | 9 | 9 / 1 / 0 |
| topcow_mr_027 | 9 (6 / 3) | ia172_058 | 172 | 0.072 / 0.276 | 9 | 8 | 8 / 1 / 0 |
| topcow_mr_013 | 8 (0 / 8) | song_AHMU1218040 | Song | 0.016 / 0.001 | 8 | 8 | 8 / 0 / 0 |
| topcow_mr_014 | 8 (0 / 8) | song_AHMU1218095 | Song | 0.077 / 0.143 | 8 | 8 | 8 / 0 / 0 |
| topcow_mr_021 | 8 (1 / 7) | song_AHMU1218113 | Song | 0.076 / 0.133 | 8 | 8 | 8 / 0 / 0 |
| topcow_mr_007 | 7 (3 / 4) | vmr0248 | VMR | 0.035 / 0.076 | 7 | 7 | 7 / 0 / 0 |
| topcow_mr_008 | 6 (3 / 3) | song_AHMU1218100 | Song | 0.129 / 0.076 | 6 | 6 | 6 / 0 / 0 |
| topcow_mr_002 | 5 (3 / 2) | ia172_136 | 172 | 0.095 / 0.079 | 5 | 4 | 4 / 1 / 0 |
| topcow_mr_012 | 5 (3 / 2) | song_AHMU1218008 | Song | 0.010 / 0.080 | 5 | 5 | 5 / 0 / 0 |
| topcow_mr_003 | 4 (2 / 2) | song_AHMU1218094 | Song | 0.061 / 0.083 | 4 | 4 | 4 / 0 / 0 |
| topcow_mr_004 | 4 (2 / 2) | ia172_071 | 172 | 0.063 / 0.200 | 2 | 4 | 2 / 2 / 0 |
| topcow_mr_017 | 4 (1 / 3) | song_AHMU1218141 | Song | 0.002 / 0.019 | 4 | 4 | 4 / 0 / 0 |
| topcow_mr_018 | 4 (2 / 2) | ia172_141 | 172 | 0.032 / 0.012 | 4 | 4 | 4 / 0 / 0 |
| topcow_mr_006 | 2 (2 / 0) | song_AHMU1218032 | Song | 0.022 / 0.046 | 2 | 2 | 2 / 0 / 0 |
| topcow_mr_025 | 2 (0 / 2) | ia172_069 | 172 | 0.142 / 0.123 | 0 | 2 | 0 / 2 / 0 |
| topcow_mr_015 | 1 (0 / 1) | ia172_039 | 172 | 0.078 / 0.106 | 1 | 1 | 1 / 0 / 0 |
| **total** | **223** | 25 primary | 13 × 172, 11 × Song, VMR | | **212** | **219** | **208 / 15 / 0** |

Reserves ia172_172 and Song 138 are not used.

**The 15 anatomies that keep one v4 VA (flagged in `provenance.v5_va.summary.flags`):**

| anatomy | v4 VA kept | donor side that failed (first failing gate) |
|---|---|---|
| case_k_008_right__topcow_mr_010_L | RVA | ia172_159 L — course RMS |
| case_k_011_left__topcow_mr_026_L | RVA | ia172_149 L — calibre / splice window |
| case_m_006_left__topcow_mr_025_L | RVA | ia172_069 L — course RMS (fails every host) |
| case_w_003_left__topcow_mr_010 | LVA | ia172_159 L — take-off band |
| case_w_003_right__topcow_mr_001 | RVA | ia172_165 R — calibre / join / splice window |
| case_w_007_right__topcow_mr_010_L | RVA | ia172_159 L — mixed (course, join, rise, calibre) |
| case_w_007_right__topcow_mr_027 | LVA | ia172_058 L — course RMS |
| case_w_018_right__topcow_mr_022 | LVA | ia172_192 L — course RMS |
| case_w_024_right__topcow_mr_002 | LVA | ia172_136 L — course RMS |
| case_w_027_left__topcow_mr_004 | RVA | ia172_071 R — calibre |
| case_w_029_right__topcow_mr_004 | RVA | ia172_071 R — calibre |
| case_w_029_right__topcow_mr_010_L | RVA | ia172_159 L — mixed |
| case_w_037_right__topcow_mr_010_L | RVA | ia172_159 L — mixed |
| case_w_042_left__topcow_mr_010_L | RVA | ia172_159 L — course RMS |
| case_w_051_right__topcow_mr_025_L | RVA | ia172_069 L — course RMS |

11 anatomies keep the v4 RVA (their LVA is replaced) and 4 keep the v4 LVA (their RVA is replaced). On topcow_mr_010,
5 RVAs and 1 LVA stay v4.

## 2. Per-unit dry-run pass counts (donor side → VAs passing / 223 candidates)

Full per-side and per-gate detail is in `results/v5_assign_dry_stats.json`: candidates, passes as RVA / LVA, base
all-pass, dominant failing gate, calibre pass 2, take-off not preserved, ostium advance and stub.
"Both" is the number of anatomies where the unit's own pair passes on both sides.

| unit | R pass | L pass | both | dominant failing gates |
|---|---|---|---|---|
| vmr0248 | 207 | 209 | 194 | R course 10; L origin 5 |
| song_AHMU1218108 | 213 | 198 | 190 | course |
| song_AHMU1218040 | 202 | 208 | 188 | course |
| song_AHMU1218113 | 191 | 219 | 187 | course |
| ia172_149 | 211 | 193 | 183 | course; L join 11 |
| song_AHMU1218095 | 200 | 200 | 178 | calibre |
| song_AHMU1218002 | 194 | 193 | 171 | course, calibre |
| song_AHMU1218109 | 199 | 191 | 170 | course |
| ia172_141 | 193 | 189 | 166 | course |
| ia172_164 | 196 | 173 | 156 | L calibre 45 |
| ia172_015 | 194 | 172 | 151 | course |
| ia172_190 | 186 | 177 | 148 | course |
| song_AHMU1218100 | 166 | 191 | 138 | R origin turn 32 |
| ia172_165 | 139 | 214 | 130 | R origin turn 46, calibre 38 |
| song_AHMU1218094 | 143 | 200 | 126 | R calibre 46; L uses a 1 mm ostium advance |
| song_AHMU1218141 | 175 | 158 | 119 | course |
| ia172_192 | 196 | 125 | 110 | L take-off band 51 |
| song_AHMU1218008 | 160 | 136 | 99 | course, calibre |
| ia172_058 | 164 | 88 | 74 | course (L 82) |
| ia172_136 | 203 | 77 | 69 | L course 132 |
| ia172_039 | 122 | 103 | 40 | R calibre 58; L take-off band 52 |
| song_AHMU1218032 | 168 | 46 | 37 | L take-off band 111 |
| ia172_071 | 59 | 132 | 30 | R course 98, calibre 63 |
| ia172_159 | 198 | 12 | 11 | L take-off band 179 |
| ia172_069 | 174 | **0** | 0 | L course 166, rise 49 |
| ia172_172 (reserve) | 212 | 111 (advance 3 mm) | 102 | L calibre 65, origin 37 |
| song_AHMU1218138 (reserve) | **0** | 108 | 0 | R course RMS 217 (132 mm line ending at C2; no V3 window) |

## 3. The assignment rule and every judgement

**Objective.** The solve is exact and lexicographic (`solve_lex`: a MILP maximises one term, fixes it at its optimum,
then maximises the next). A summed float weight cannot keep this order over 25 pairs. The terms, in order:
0. patients served by a **primary** pair (≥ 1 VA passing);
1. (a) anatomies with both VAs passing every gate;
2. VAs passing;
3. RVAs passing;
4. (b) the sum, over passing VAs, of the donor side's coarseness (tally contact-free union);
5. (c) the sum, over passing VAs, of (2 − flag level);
6. VAs with the take-off preserved.

Term 4 is the "coarseness weighted by the number of anatomies" rule: a pair contributes once per anatomy it serves.
Reserves enter a second solve, and only for patients that term 0 leaves unserved. Here there were none.
The summed-weight Hungarian (reported beside) gives the same assignment as the strict MILP.

Judgements:
1. **Reserve rule: primaries serve every patient first.** The first solve maximised (a) over the primaries alone. That
   left topcow_mr_015 (1 anatomy) to ia172_069, whose left side fails everywhere, so it passed 0 VAs there. The
   reserve stage then put ia172_172 on mr_015 (its R side, mirrored, as the LVA).
   - Totals for that variant A: 209 anatomies on both sides, 432 VAs, 213 RVAs.
   - It used a reserve where a primary can serve: ia172_039 passes both sides at mr_015.
   - The 14:25 decision says "25 primary pairs for 25 patients; reserves only where a patient has no other
     gate-passing donor". I therefore put "patients served by primaries" first.
   - Cost against variant A: 1 anatomy on both sides (208 vs 209), 1 VA and 1 RVA. ia172_069 then serves mr_025 with
     LVAs only.
   - Variant A is kept for comparison: `v5_reg/plan_v5_A_reserve_on_mr015.json`,
     `results/graft_reg_assign_v5_A_reserve_on_mr015.json` and `results/v5_assign_alt_all_served.json`.
2. **(a) before (b).** Coarseness only orders assignments that tie on the gate terms.
   - Putting coarseness before the gates would raise the coarseness sum from 33.28 to 39.37 (+18 %). It would cost
     24 anatomies on both sides (184) and 24 VAs, and move 17 units (`results/v5_assign_alt_orders.json`).
   - The coarsest pairs (ia172_058, 0.174; ia172_069, 0.132) therefore do not go to 16-anatomy patients: on those
     patients they pass fewer VAs.
   - The four 16-anatomy patients get ia172_164 (0.109), Song 108 (0.090), ia172_015 (0.078) and Song 002 (0.038).
     All four pass 32 / 32.
3. **RVA tie-break** (the RVA is the training-reachable side). It made no difference here; variant D gives the same
   assignment.
4. **(c) flag level.** 0 = QC "donor" and nothing else; 1 = review flags; 2 = a provisional-limit pass, truncated wall,
   V1 rescue or Siemens 1.0 mm provenance flag. Putting it before coarseness would move 4 units, lower the coarseness
   sum by 0.39 and raise the unflag points 364 → 399. Following the brief's order, it stays after coarseness.
5. **Coarseness measure.** The coarseness used is the final tally's per-side contact-free donor-stretch union (decision
   03:50 item 4), not the builder's own clean union. The builder's value is kept as `coarseness.builder_score`.
6. **Donor files.** Records come from the files the tally graded (eligible list `wall_files` / `qc_file` /
   `trace_file`), not from the builder's 1.0 defaults. The pilot record's old QC status is kept as
   `qc.pilot_record_status`.
7. **Calibre gate at the cut splice.** This was verify_graft_reg's defect (a), fixed here: the gate is re-applied at
   the 1 mm sample actually cut. This makes the gate slightly stricter than the graft_reg test runs.
8. **Reserve conditions encoded although unused.** ia172_172 L has `min_ostium_advance_mm` 3 (dry run: 111 / 223
   passes, all at a 3 mm advance) plus an arch-origin flag. Song 138 R carries a "real wall to C2 only" flag. 138 R
   cannot be grafted under strategy B without extending its line above C2 (not done; not needed).
9. **New registry folders.** donors_v5 / dry_v5 / plan_v5 were used so that the graft_reg test records stay as they
   were.
10. **Kept sides.** A side whose assigned donor fails keeps v4's VA, xform and curve byte for byte. It is flagged
    with the reason in `provenance.v5_va.summary.flags` and `v5_va.<side>.why`.

## 4. Provenance of each written anatomy (what bake_v5.py and the records need)

`provenance.json` = v4's provenance with `xform.va_R` / `va_L` updated for every replaced side (TopBrain piece, cut,
trim). It also adds:
- `xform.vaC_R` / `vaC_L` for every replaced side: R, origin, anchor, mirror, donor surface path + md5, donor record,
  donor id / side, `lo_mm` (ostium + real_from), `hi_mm` (splice cut), `real_surface` (true for all 25 pairs; VMR L
  is real from 40 mm);
- `v5_va`: the gates; per side the full graft record (pose, splice, calibre ratio and pass, origin turns, join,
  course RMS, tip move, stub, ostium advance, take-off native vs placed and `takeoff_not_preserved`, wall class,
  `donor_eligible`: class, licence, NC flag, attributions, scanner, provenance flags, side flags, flag level, eye
  verdicts);
- `v5_va.assignment`: patient, unit, class, rule, sides served, flags;
- `v5_va.summary`: which VA kept v4, donor per side, NC flag, attributions, flags.

`LICENSE_v5_donors.txt` holds the licence, the attributions and the NC notice. The 7 VMR anatomies (topcow_mr_007)
also carry `LICENSE_VMR_0248.txt`.

Licence counts:
- 216 anatomies are NON-COMMERCIAL (Arterial-derived 172-series / Song donors).
- The 7 VMR anatomies need the VMR acknowledgement (Wilson et al. 2013; NLM R01LM013120 / NHLBI HHSN268201100035C).
- CC BY 4.0 attribution applies for Song / 172-series images and for TotalSegmentator.

Written-set statistics over the 431 replaced VAs:
- Take-off not preserved (|Δ| > 15° at 10 mm): 252, with a median |Δ| of 17.8°.
- Calibre pass 2 (ratio 1.25–1.6, flagged): 52.
- Ostium advance: 0 mm on 421, 1 mm on 8, 2 mm on 2.
- Stub: median 5.28 mm, max 10.49 mm.
  - Stubs > 8 mm: ia172_136 L at 10.5 mm (4 VAs) and ia172_071 L at 9.2 mm (4 VAs). Their ostium sits 9.3 / 7.5 mm
    from their own subclavian axis. The stub is still ungated (verify_graft_reg item 7).

## 5. Split check output (v5_split.py on anatomies_v5, baseline results/split_v4.json)

```
v3     roster_v3: test 16 train 153 excluded 54; components [223]; donors {'T': 25, 'V': 25, 'Z': 48}
v3     max k=16 183/16 (div6x6 181/16), k=17 181/17 (div6x6 180/17)
v4side roster_v3: test 16 train 127 excluded 80; max k=16 174/16 (172/16), k=17 172/17 (170/17)
strict roster_v3: test 16 train 127 excluded 80; max k=16 174/16 (168/16), k=17 172/17 (167/17)
v3     extra (V:) donors 25 in 25 TopBrain patients; spanning > 1 patient: 0; one_to_one_nested True
v4side extra (V:) donors 25 in 25 TopBrain patients; spanning > 1 patient: 0; one_to_one_nested True
strict extra (V:) donors 25 in 25 TopBrain patients; spanning > 1 patient: 0; one_to_one_nested True
not lowered vs baseline (45 quantities compared)
EXIT 0
```
Full log: `carotid_data/va_sources/v5_reg/logs_v5/split_v5.log`.

## Tool extensions (stated diffs; originals in `v5_va/v5_assign_work/orig/`)

- `graft_v5_reg_donor.py` 1.0 → 1.1
  - New options: `--out-root` (default unchanged) and `--eligible FILE [--classes primary,reserve] [--only]`.
  - `build_thin(..., ov)`: with a side of the eligible list, the wall comes from `wall_files`. That means the surface
    and VMTK line, with measure / edge / landmarks / seg_qc taken from the surface's folder (ct_seg4 or ct_seg4v2).
    The QC record comes from `qc_file` and replaces the pilot record. The chirality trace comes from `trace_file`
    first. Without `ov` nothing changes.
  - `attach_eligible`: adds the eligible facts, `flag_level`, and sets `coarseness.score` to the tally's contact-free
    value. The builder's own value is kept as `builder_score`.
  - Reserve conditions are recorded in `RESERVE_CONDITIONS`.
- `graft_v5_reg.py` 1.0 → 1.1
  - The calibre rule is applied at the cut splice, both in the search and in the base record.
  - The `min_ostium_advance_mm` ladder start is honoured.
  - A plan that breaks the follow handedness rule raises an error.
  - Provenance fields are added (section 4).
  - `--compat` regression re-run: 223 anatomies, 9,366 leaves, **0 differences** (`v5_assign_work/regress_compat_v11.json`).
- `graft_v5_reg_dryall.sh`: new env `CONTAINER`, `DONORS`, `DRY`, `WORKERS` (defaults unchanged). A unit whose
  `_done.json` exists is skipped unless `FORCE=1`.
- `graft_v5_assign.py` 1.0 → 1.1
  - New options: `--donor-root`, `--score v5` (`solve_lex`, LEX_TERMS with "served" first) and `--reserve-units`
    (second stage).
  - `patient_key` is enforced (verify_graft_reg item 10).
  - Each plan entry carries a `v5_assign` provenance block.
  - The verification uses the given donor root.
- `v5_split.py`: prints the V: donor nesting for each definition. It exits with 4 if any V: donor spans more than one
  TopBrain patient, even when no number drops (verify_graft_reg item 12). Exits 2 and 3 take precedence.

## Files

| path | what |
|---|---|
| `carotid_data/anatomies_v5/` | 223 unbaked v5 anatomies (Centrelines_comb, provenance.json, LICENSE files) |
| `va_sources/v5_reg/donors_v5/<unit>/` | 27 donor units (record.json, R/L.json, R/L_surface.vtp copies) |
| `va_sources/v5_reg/dry_v5/<unit>/` | per-anatomy dry-run JSONs, `_done.json`, `_dryall.log` |
| `va_sources/v5_reg/plan_v5.json` (+ `_verify_rows.json`) | the plan written to anatomies_v5 |
| `va_sources/v5_reg/plan_v5_A_reserve_on_mr015*.json` | variant A (not used) |
| `va_sources/v5_reg/dry_v5_compat/`, `dry_v5_smoke/` | compat regression run; 2-anatomy smoke test |
| `va_sources/v5_reg/logs_v5/` | records, compat, assign (+ variant A), write, split logs |
| `results/v5_assignment.json` | **the assignment record** (table, totals, flags, dry-run summary, split, alternatives) |
| `results/graft_reg_assign_v5.json` | assignment tool summary (score matrix, optima, verify log) |
| `results/v5_assign_records.json` | donor record checks |
| `results/v5_assign_dry_stats.json` | per unit / side / gate dry-run statistics |
| `results/v5_assign_written_check.json` | written-anatomy check |
| `results/split_v5.json` | split check |
| `results/v5_assign_alt_all_served.json`, `results/v5_assign_alt_orders.json` | variant A; objective-order sensitivity |
| `v5_assign_work/` | `check_records.py`, `dry_stats.py`, `check_written.py`, `alt_all_served.py`, `alt_orders.py`, `v5_assign_report.py`, `regress_compat_v11.py/.json`, `peek_eligible.py/.txt`, `orig/` |

## Open issues (for the bake stage and the verifier)

- **Not baked.** Run `bake_v5.py --anatomies /repo/carotid_data/anatomies_v5` (neve-build-meshlab).
  - Several assigned donor surfaces have high genus: Song 094 L 47, Song 002 R 32, ia172_159 L 30, Song 008 L 26,
    ia172_015 R 22, and 10–12 for 7 more sides. Two earlier bakes with genus ≤ 6 were clean. The bake ladder may
    suppress stretches, so the first bakes should include these donors.
- **Take-off.** 252 / 431 placed VAs do not preserve the donor's native take-off (|Δ| > 15°). They are recorded
  per VA; this is not a gate.
- **Stub ungated.** The max is 10.5 mm (ia172_136 L, 4 VAs on topcow_mr_002).
- **Unusable donor sides.** ia172_069 L, whose V1 runs horizontally for ~30 mm, contributes nothing; ia172_069 serves
  only 2 LVAs. ia172_159 L (native take-off 149°) serves 14 LVAs and leaves 5 RVAs on topcow_mr_010 as v4. A
  better-matched pair for topcow_mr_010 would need a new donor.
- **V5_BUILD_LOG.** Not edited. Suggested line: "v5_assign COMPLETE: 25 primaries → 25 patients (no reserve);
  RVA 212 / LVA 219; anatomies 208 both / 15 one / 0 none; anatomies_v5 written unbaked (223); split EXIT 0, nested."
