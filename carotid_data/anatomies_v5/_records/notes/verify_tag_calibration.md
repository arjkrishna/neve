# verify_tag_calibration: adversarial check of the two-patient recalibration (notes/tag_calibration.md)

Agent prefix `verify_tag_calibration`. Date 2026-10-01, about 20:16–21:50 EDT. No git, no GPU, one CPU container at a
time (neve-build-stagea). Nothing was written outside these files: `/sp/v5_va/verify_tag_calibration_*.py`, this note and
`/sp/v5_va/state/verify_tag_calibration.md`. The script outputs went to stdout only; their key numbers are copied below
and in the state file.

**Licence.** TaG-Net has no data licence, so everything here about patients A and B is internal only.

**Earlier attempts.** A first attempt (18:13–18:17) was cut off. It left the helpers
`verify_tag_calibration_{frames,radius_tab,show,tree}.py` and an empty directory, `derived/tagnet/verify_calibration/`.
The 20:07 relaunch left `_peek.py`. I used `_tree.py` read-only and wrote no new code on top of these helpers.

## Answer so far / verdict: **the numbers reproduce; "right and not circular" does not hold**

Every key number reproduces with my own code: the V3-loop values, the entry calls, the radius ratios, the gate counts
and the gross values. But four parts of the claim fail:

1. **The headline gate result is circular.** "Recalibrated: 15 / 40, 11 subjects, 4 pairs, the corrected yield
   exactly, 0 of 6 wrong" is true by construction. 5 of the 6 sides the gate gains over the draft get in only because a
   limit was moved to admit that same side. The 0 / 6 is also in-sample: d ≤ 6 was set knowing that s0518 L reads 7.5.
2. **The V3 gate and the Erec parameters were tuned on patient A, then "validated" on patient A.**
   - Without A, the data put the excursion floor at 16.6 mm, and the study rule then fails 17 of A's 32 correct rows.
   - Without A, 11 entry-rule settings are perfect on VMR, and A is what picks R(0.5, 5, 2).
   - VMR constrains neither V3 threshold.
   - d ≤ 6 has no GT basis at all: GT d ≤ 1.1, wrong traces ≥ 12.5. It rests on one TS side each way.
3. **"Native: one radius factor works (× 1.10, 1.5 % apart)" depends on which run of A is used.**
   - On the same 515 stations, A's right ratio is 0.769 with the VMR-skull runs and 0.887 with the TS-skull runs. That
     is a 15 % spread within one patient, from the run protocol alone.
   - Matched on protocol (VMR skull on both patients), the native factors are 1.11 and 1.21, 8–9 % apart.
4. **Two gross ranges are misstated.**
   - Real DM is given as 1.40–1.75, but the cited TS whole VAs reach 1.32.
   - The take-off range 42–93° leaves out SPL R (106°) and the TS neck VAs (20–110°). Its top end is B R alone, which
     rests on an assumed subclavian radius.

Several thresholds still rest on one patient or none: the excursion floor (A, one VA), d (TS only), BCV (VMR only), the
C6 landmark (A only; VMR's trusted right side is still −10 mm) and the band / V1 / total / length changes (TS or design).

Two patients do justify some changes:
- rule 6 to review;
- that the ≥ 15 mm excursion floor is falsified (by one real VA);
- that ts_qc over-calls C7 (both patients);
- that there is no single radius factor at 1.5 mm, so calibre should come from the CT half-max.

## Checks

| # | check (own code) | result | reproduced? |
|---|---|---|---|
| 1 | V3 excursion / d_C1post, study cord and robust cord (top-only and all-partial variants), zcap stretch; 28 VMR + 102 A path × mask rows, 40 TS sides (`_own.py`) | within 0.1 mm everywhere except TS s1154 R robust (mine 2.3 / 11.7, claim 1.7 / 12.8) | yes (s1154 R: no gate effect) |
| 2 | partial top cord slices on the traced TS subjects | 16 / 20 by my count; claim 18 / 20. Their robust cord is computed on a **cropped** volume, so "partial" (< 50 % of the median slice count) depends on the crop | no (minor) |
| 3 | course correctness of the GT-patient traces (within 3 mm of GT over V2 / V3, reaching C1) | VMR 16 / 16 correct; A 10 correct / 68 wrong (total_v3) | yes |
| 4 | the claim's V3 value ranges: VMR traces 18.0–20.8 / 0.1–0.4; A correct 14.5–17.2 / 0.1–0.7; A wrong d 12.5–26.9; A left 14.3 ("total") and 14.8–15.0 (total_v3, robust) | mine 18.1–20.8; 14.5–17.2 / 0.1–0.7; 12.5–26.9; 14.3 and 14.5–15.0 | yes |
| 5 | V3 sweep, robust and study cords × exc 8–15 × d 5–8 | **every exc 8–14 with d ≤ 6 gives the same result on all data** (GT 10 / 10 on both mask sets, correct 26 / 26 and 14 / 14, wrong 0 / 68, TS 19 / 40). d ≤ 5 → 18; d ≤ 8 → 20 (adds the wrong s0518 L). The study cord at ≥ 12 / ≤ 6 equals the robust cord on every count | yes. The robust cord changes no count |
| 6 | V3 separation by source | GT and correct traces d ≤ 1.1; wrong A ≥ 12.5 (robust). TS u15 max d 5.1 (s0346 L); TS wrong6 min d 7.5 (s0518 L). Excursion floor: VMR ≥ 18.1, TS u15 ≥ 16.6, A left 14.3–15.0. The only negative with d ≤ 6 is s1823 L (exc 4.8, a no-foramen trace) | — |
| 7 | V3 leave-one-source-out | **without A:** exc floor 16.6, and the study rule passes only 15 / 32 of A's positive rows. **Without TS:** the d window is (1.1, 12.5), so GT cannot place d. **Without VMR:** nothing changes | **refutes** "two patients" |
| 8 | entry rule Erun / Erec, own EDT profiles | the same calls as the claim on all VMR and A rows. TS: 38 / 40 Erun (s0333 R C5 vs C6, s1883 R None vs C5), Erec 40 / 40. TS Erec: C6 33, C7 3, C5 1, none 3. All 15 usable read C6: 13 backed by a C6 run, s0289 L and s0346 L by ts_qc's own C6 call | yes |
| 9 | entry grid per patient (18 settings) | 11 settings are 100 % on VMR. Of those, only R(0.5, 5, 2/3) reach 84 / 84 on A total_v3 (the others 39–66 / 84). "Always C6" also scores every GT row, since no GT VA enters at C7 or C5 | **refutes** out-of-sample validation. C7 sensitivity is untested |
| 10 | C6 landmark offsets (claim json) | ts_qc −7.2…−14.2 mm. Erec −0.2…−1.2 on A; −0.8…+0.2 on VMR L, against the disputed 37 mm; **−9.9…−10.8 on VMR R**, the trusted VMR side | the improvement is shown on one patient |
| 11 | Stage-2 gates re-applied from `va_extract.json` (gaps, labels, ts_qc), `arterial_metrics.json` (usable flags) and my own V3 / band / HU | draft + additions 9 / 40, 7 subjects, pairs s0960 and s1786; hard 11. Recalibrated 15 / 40, 11 subjects, 4 pairs. **Same side lists and same per-rule failure counts** as the claim. The 15 match the yield table (15 / 11 / 4) | yes |
| 12 | which change admits each of the 6 sides gained | s0346 L: d ≤ 6 only (moved for this side). s0610 L: V1 ≤ 3 mm (1.9; moved for it) + rule 6. **s0777 L: rule 6 only** (GT-backed). s1129 L: IQR dropped. s1304 R: V1 2.9 ≤ 3 + IQR dropped. s1823 R: IQR dropped | **circular** for 5 of 6 |
| 13 | rule 6 on the GT-patient native traces | VMR native R 0.00 / L 0.874; A native (VMR skull, oracle, tstot) L 0.00; A TS-skull L label absent | yes. Two-patient-backed |
| 14 | BCV / CCA HU (own erosion and median, 1.5 mm CT) | VMR R 266 / L 2278 (total_v3), R 419 / L 2218 ("total"). A L 50 / R 199 (total_v3), 50 / 210 ("total"). The implemented draft already sends BCV ≥ 200 to review, so the "move" changes no count | yes. One patient (VMR) |
| 15 | review flags on the 15 | 10 bridged > 2 mm, 5 below the splice (s0346 L, s0777 R, s1154 L, s1304 R, s1823 R), 2 label, 1 BCV, 12 flagged in total | yes |
| 16 | radius: VMTK MISR / r_img, origin → C1, every run of both patients (`_radius.py`) | the claim's ratios to 3 decimals. ct_seg4 MISR / r_img 0.921–0.954 (claim 0.93–0.95) | yes |
| 17 | radius factor robustness | native factors: VMR 1.113; A TS-skull 1.097 / 1.108; **A VMR-skull, oracle, nots and tstot 1.208–1.210**. Same stations, A R: 0.769 vs 0.887 (15 %); A L: 0.895 vs 0.921. 1.5 mm: VMR 1.05–1.08, A 1.20–1.21, A TS-skull 1.30 | **refutes** "one factor at native" as a general result |
| 18 | TS radius band | 38 / 40 raw (s1029 L 0.89, s1823 L 0.85); both fail other gates | yes |
| 19 | gross: own σ-3 bend radius in the C2-loop window, B path-only proxy | A 3.15 / 2.41, B 1.38 / 2.60, VMR CT 1.44 / 1.56 (claim 3.14 / 2.40, 1.37 / 2.59, 1.44 / 1.54). B raw 6.4 / 5.3, fit 17.0 / 18.2, slope 0.165 / 0.225. A L G2 fails by 0.09 mm (2.49 at σ 2.5) | yes (A L is borderline) |
| 20 | gross range arithmetic from the claim's own table | DM 1.40–1.75 omits TS 1.32. Take-off "42–93" omits SPL R 106 and TS neck 20–110 (n 13); its top is B R 92.6, with an assumed SCA radius. The host's take-off is inside either way. "Donors near the host's 13–27°: B, SPL": SPL is 33° | **misstated** (minor) |
| 21 | lengths / DM | follow va_metrics (raw arclength): A L 266.2 / 1.747. On a σ-3 line they would be 249 / 1.657, so G4 length would pass and DM fail by 0.007 | convention-consistent |

## Circularity: where each change was tuned and where it was "validated"

| change | tuned on | validated on | independent support |
|---|---|---|---|
| rule 6 → review | VMR native R / L, A native L (correct traces with label 0.00–0.87) | the same, plus TS s0610 L and s0777 L | **two patients**. Fine |
| V3 excursion ≥ 12 | A left (14.3–15.0) | A's GT and correct rows (10 / 10, 26 / 26) | none: VMR ≥ 18.1, TS ≥ 16.6. One real VA falsifies ≥ 15; the value 12 is arbitrary in (4.8, 14.3] |
| V3 d ≤ 6 | TS s0346 L (5.1) and s0518 L (7.5) | the same TS sides (15 / 15, 0 / 6) | **none**. GT d ≤ 1.1; the TS labels came from the yield verifier's own V3-loop test |
| robust cord | A total_v3 partial top slices | — | changes no pass / fail count; crop-dependent |
| Erec R(0.5, 5, 2) | grid scored on VMR + A (A selects it) | the same GT rows | VMR confirms it out of sample, but "always C6" scores the same. C7 sensitivity untested |
| Erec C6 landmark | A (and VMR L, against a disputed 37 mm) | the same | VMR R still −10 mm |
| band IQR dropped | TS s1129 L, s1304 R, s1823 R | the same 15 | none. GT passes with or without |
| V1 bridge ≤ 3 mm | TS s0610 L (1.9) and s1304 R (2.9): a 0.1 mm margin | the same 15 | none. Wrong-vessel s1129 R has a 2.2 mm V1 break, so the limit does not separate |
| total ≤ 25 below the splice | design argument (strategy B) | changes no u15 side | none |
| BCV ≥ 200 → review | VMR (266; 419 on the "total" masks) | — | one patient. Already "review" in the implemented draft |
| length 260–290 → review | none | none | none |
| radius × 1.10 (native) | VMR native vs A native TS-skull | the same pair | protocol-mismatched. 1.11 vs 1.21 when matched |

## Corrected statement

- The recalibration's numbers are reproducible.
- Two patients justify:
  - rule 6 becoming review only;
  - abandoning the ≥ 15 mm V3 excursion floor (one real GT VA reads 14.3–15.0);
  - correcting ts_qc's systematic C7 over-call, using a run rule like Erec, whose specificity holds on 4 / 4 GT VAs;
  - not using a single radius factor at 1.5 mm, and taking calibre from the CT half-max (ct_seg4 MISR / r_img
    0.92–0.95 on both).
- They do not justify:
  - the specific values excursion 12, d 6 and V1 ≤ 3 mm;
  - dropping the IQR term;
  - "native × 1.10 works": the factor is 1.10–1.21 depending on the Arterial run protocol;
  - the claim that the gate reproduces the corrected yield "automatically". That match was engineered on the same 15
    sides, and the 0 / 6 wrong rejection was tuned on s0518 L.
- Out-of-sample checks still owed:
  - the new gate on new TS subjects, or on the pilot's thin-slice cases;
  - a true C7 entry for Erec's sensitivity;
  - a third GT for the C6, C2 and C1 landmarks.

## Scripts (all new, `/sp/v5_va/`)

| script | what |
|---|---|
| `verify_tag_calibration_tables.py` | prints the claim's sweep and per-side tables from `tag_calibration.json` |
| `verify_tag_calibration_own.py` | own V3 (3 cord variants), correctness, Erun / Erec + grid, band, HU, gate re-application, sweep, leave-one-source-out; `VMR A TS` takes about 8 min |
| `verify_tag_calibration_radius.py` | own radius ratios per run and zone, same-station protocol spread, gross bend / length / DM, B proxy, range arithmetic |
| `verify_tag_calibration_misc.py` | rule-6 labels, BCV / CCA HU on the VMR and A masks, landmark offsets from the claim json |

Not done:
- I did not re-run ts_qc. E0 calls and landmarks come from `va_extract.json` (TS) and the claim's harness json (GT).
- I did not re-measure the C2 / C1 landmark offsets independently.
- No visual check of the 15 or of Erec's 3 TS C7 calls.
