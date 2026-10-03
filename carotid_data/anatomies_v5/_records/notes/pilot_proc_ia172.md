# pilot_proc_ia172: the whole v5 donor pipeline on the 172-series pilot cases

Agent prefix `pilot_proc_ia172`. Four instances: the first was cut off at ~18:20 EDT 2026-10-01 by a session limit, the
second stopped at 20:13 while taking stock, the third ran 20:17-23:38 (cut off by the next session limit; its loops kept
running and finished at 00:11), the fourth (01:09-02:15 Oct 2) ran the closing pass and wrote this answer.
State file: `/sp/v5_va/state/pilot_proc_ia172.md`. No git; no tracked file touched. Data under
`D:\Arjun\neve\carotid_data\va_sources\derived\thin_slice\ia172\` (gitignored).
Licence: Head-CTA 172-series CC BY 4.0 (Zenodo 15697196); everything through Arterial (traces, masks, cut-outs, and
the CT walls seeded from them) is non-commercial.

**Diff of this file (fourth instance, 02:15):** the "Answer so far" became the final Answer below; the per case and side
tables, the gate comparison, the ranking, the relief analysis and the failure list were added; the third instance's
interim answers are kept unchanged under "History"; the Method gained items 8-10. The pre-rewrite file is kept as
`state/pilot_proc_ia172_notes_interim_0125.md.bak`.

## Answer

Status: complete (02:15 EDT 2026-10-02).

All 12 cases (001 039 041 058 061 073 090 118 140 141 + alternates 190 015) went through TotalSegmentator `total_v3`,
Arterial native (clipped input), the trace with the basilar fix, VMTK, the QC gates, the ct_seg4 wall of every passing
side with gates / detail / edge FWHM, gross shape, QC PNGs and 3-D renders. All 24 sides traced; all 24 graded.
Results: `results/pilot_proc_ia172.json` (one record per case and side), `results/pilot_proc_ia172_grade.json/.txt`
(both gates, W flags, seam, W7, ranking), `results/pilot_proc_ia172_summary.json/.txt` (sets, references, gross, timing).

1. **Yield: the source passes the plan's 60 % go/no-go under the recalibrated gate, and fails it under the draft.**
   - Recalibrated gate (primary): **15 / 24 sides (62.5 %), 10 subjects, 5 pairs (015, 039, 058, 141, 190)**. Only 5 of the
     15 carry no review flag (058 R, 061 R, 140 L, 141 L, 015 L).
   - Study draft + the report's additions (A1 V3 loop 15 / 5 on the study cord, A2 C1 threaded): **7 / 24 (29 %), 0 pairs**.
   - Only the four two-patient-backed changes (rule 6 to review, V3 floor below 15 mm, Erec entry, calibre from the CT
     half-max; provisional limits reverted to the draft): **14 / 24 (58 %), the same 5 pairs**.
2. **Where the gates differ (8 sides), as the out-of-sample check verify_tag_calibration said was owed.**
   - 6 sides differ only through rule 6 (Arterial label agreement 0.00-0.90 -> review): 015 R, 058 L, 118 L, 141 R, 190 R,
     and 039 L, which also needs the floor below 15 mm (study cord 13.1 / 0.5). All 6 are visually correct VA courses
     with a V3 loop (QC PNGs). This supports the two-patient-backed changes.
   - 2 sides enter only through provisional, in-sample limits: 001 R (a 2.1 mm V1 break at s 0 through 314 HU, i.e. an
     enhanced lumen: V1 <= 3 mm) and 140 L (V2 band, the IQR term dropped). Both look correct by eye.
   - 140 R is the one side where the two cords disagree: robust 9.8 / d 6.3 fails the recalibrated gate (d > 6), the
     study cord reads 12.9 / 3.8 and would pass a 12 / 5 study gate. The trace follows the VA to the C1 foramen and then
     skips the loop behind C1 through an 8.2 mm bridge at 48 HU above C1. So d <= 6 on the robust cord rejected a real
     deviation here, by 0.3 mm. Its origin -> C1 stretch is a candidate donor only after review.
   - No side the recalibrated gate admitted looks like a wrong vessel on its QC PNG (viewed: 001, 015, 039, 058, 061, 090,
     118, 140, 141, 190). Every failing side either breaks in V1 or leaves the VA above C2.
3. **Wall: the 172-series is coarse, much coarser than the host and than VMR / TaG-Net, and the relief is image
   structure, not noise; but most of it sits at contacts.**
   - 14 valid walls (140 L has a genus-1 handle in V2, so its V4 gate fails). Whole-VA union > 0.3 / > 0.5 / > 0.15 r, set
     medians **0.160 / 0.076 / 0.227** against the native bars 0.0966 / 0.0204 / 0.1416 (W1, W2, W3 pass as a set; W3
     read with W1). Per side, W1 is reached by 9 of 14 whole VAs and 10 of 14 donor stretches. W7 (by hand) fails on
     every side: V4 is smooth in CT (0-40 mm <= 0.15), which does not matter for strategy B (TopBrain distal piece).
   - **Donor stretch** (origin -> C1f - 25 mm, exact ownership; the 25 mm seam zone separate): 0.038-0.354, median 0.192
     over the 14 valid walls.
     Every donor stretch exceeds the dense host RVA (0.027) and the host tube (0.0007); the two smooth sides (141 L, 190 L,
     both 0.038) only by x 1.4.
   - Against the references in the same harness: whole VA 0.026-0.262 here vs VMR 0248 0.045 / 0.138 and TaG-Net A
     0.045 / 0.111. Six sides read 0.21-0.26 (058 L, 061 R, 001 R, 118 L, 090 R, 140 L), five are VMR-right-like
     (0.026-0.094: 058 R, 039 R, 141 L, 190 R, 190 L).
   - **Not noise.** Featureless-tube phantoms with the real PSF, noise level and noise correlation, through the identical
     ct_seg4 + VMTK + va_metrics: 061 R 0.0008 (x 2 noise 0.019) and 140 L 0.005, against 0.245 / 0.214 real.
   - **Coarse.** On the six coarse sides 0.10-0.14 of the wall still stands > 0.3 mm off a tube fitted after a 1.5 mm
     low-pass (half or more survives); band roughness 0.75-3 mm reads 0.11-0.15 mm RMS, while the bands below 0.375 mm
     read ~0.04 mm, the floor also seen on VMR / TaG-Net.
   - **At contacts.** A median 69 % of the relief area lies in 10 mm bins where >= 5 % of the edge rays meet a brighter
     neighbour (bone in the foramina; other enhanced vessels or veins); union 0.237 in those bins vs 0.080 elsewhere.
     Label voxels beyond the station median edge are 0-2 % of the label and 70-86 % lumen-bright on the coarse sides
     (protrusion.json); a CPR of 140 L showed part of the V2 bulges is intermediate-HU perivascular tissue (likely the
     venous plexus) merged at half-max. So the user's rule (exclude contamination relief) matters here.
4. **Ranking quantity and the within-source ranking** (grade.py): union > 0.3 mm of the donor stretch over the
   contact-free 10 mm bins only. Single sides: 090 R 0.318, 058 L 0.276, 118 L 0.249, 061 R 0.244, 140 L 0.209 (genus 1),
   001 R 0.153, 015 R 0.131, 039 L 0.106, 039 R 0.078, 058 R 0.072, 190 R 0.056, 015 L 0.036, 141 R 0.032, 190 L 0.030,
   141 L 0.012 (set median 0.092). **Pairs (both sides recalibrated donors with a wall), mean of R and L: 058 0.174,
   039 0.092, 015 0.084, 190 0.043, 141 0.022.** None of the 5 pairs passes the draft gate on both sides. The coarsest
   single sides (090 R, 118 L, 061 R, 001 R, 140 L) have a failed or wall-less other side, so under the one-to-one pair
   mapping the 172-series gives at most 5 pairs, of which 058, 039 and 015 are coarse on both or one side and 190 and 141
   are smooth on at least one side (contact-free <= 0.032).
5. **Gross shape of the donor sides** (Arterial VMTK line, sigma 3): whole-VA length 208-270 mm (090 R 269.7 is in the
   260-290 review band), DM 1.30-1.53, take-off at 10 mm 46-118 deg, subclavian elevation -26 to 44 deg, C2-loop bend
   1.47-5.26 mm (G2 < 2.5 mm on 015 R 1.95, 061 R 2.01, 140 L 1.47), corrected radius 1.32-2.20 mm. Entry: Erec C6 on
   14 of 15 (015 R: C7, a C7-entry candidate to check by eye); ts_qc calls C7 on 14 of 15 (the known over-call).
   L/R calibre 0.70-1.39.
6. **Failures (9 sides) and why.** V1 breaks at the origin: 061 L (5.6 mm at 310 HU: an Arterial mask break in an
   enhanced lumen, rescuable on review), 041 R (4.0 mm, 177 HU), 041 L (8.0 + 8.3 mm, 102-128 HU), 073 L (3.9 mm through
   -25 HU fat = a shortcut; plus a 6.5 mm V2 bridge). Wrong course above C2 / no V3 loop: 073 R (5.5 / 7.4, 9.9 mm V2
   bridge at 47 HU), 090 L (0.5 / 21.1, 8.5 mm V2 break at 224 HU then a lateral path; traced without labels), 118 R
   (1.1 / 21.9, 8.2 mm V2 bridge at 94 HU, straight up without the loop), 140 R (above). Wrong vessel from the start:
   001 L (Arterial labels wrong; the mask has lost the left VA over > 40 mm; the no-label rescue did not help). The
   unchanged art_extract trace (`arterial/ts`) is identical to tsba below the splice on every failing side, so no route
   rescue exists (unlike Song 100 L).
7. **Pipeline problems found and fixed in the pilot copies.** (a) ct_seg diff 8 (connectivity-preserving opening) is
   required here: where it acted, the plain 0.8 mm opening keeps only 77-129 mm of a 200-250 mm VA (015 R / L, 039 L,
   140 L, 141 L, 190 L) or drops the whole V1 (058 R); where it kept few voxels the readings are unchanged (090 R,
   118 L). (b) diff 7: NaN VMTK seeds. (c) Arterial's centreline stage hung on 2 of 12 (061, 090); both were stopped
   (watchdog), their segmentations kept, and traced without labels. (d) TotalSegmentator's per-class writer is
   single-threaded: 695-1112 s per case with the GPU idle most of it; `--ml` + a CPU split: 205-294 s.

## Per case and side

QC verdict columns: recalibrated (primary; tag_calibration.apply_gate "recalibrated" + the brief's HARD checks) and the
study draft + additions; "differ" names what the draft fails where the recalibrated gate passes. The backed-only variant
is in `results/pilot_proc_ia172_grade.json` (`qc_two_patient_backed_only`). Wall flags: W1 / W2 / W3-with-W1, each
against the native SET bar applied to the side alone (Y / n). "dura cut": the distal course failed QC, so the wall stops
at the dura and W7 has no true distal end. 001 R donor stretch from the ts set (ts C1 = rec C1); 140 L from 10 mm bins
(approximate) because walls cut at the dura have no rec / ct stretch set in va_metrics.








### QC per side (recalibrated gate = primary; study draft + A1 V3 15/5 + A2 C1 threaded beside it)

| case | side | recalibrated | draft + additions | differ: what admits it | Erec / ts_qc | V3 robust (exc / d) | V3 study | HU / CCA | trace mm | distal ok | fail reasons (recalibrated) / review |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 058 | R | donor | pass | – | C6 / C7 | 16.9 / 0.1 | 17.8 / 0.1 | 1.03 | 230.9 | True | – |
| 058 | L | donor_review | FAIL: 6_label_agreement_ge_0.9 | 6_label_agreement_ge_0.9 | C6 / C7 | 19.8 / 0.4 | 19.8 / 0.4 | 1.05 | 242.0 | True | Arterial VA label off the trace (0.89 within 2 mm) |
| 061 | R | donor | pass | – | C6 / C7 | 22.0 / 0.2 | 22.0 / 0.2 | 1.00 | 196.9 | True | – |
| 061 | L | fail | FAIL: 3a_no_V1_bridge | – | C6 / C7 | 16.5 / 0.2 | 17.5 / 0.2 | 1.00 | 205.7 | True | 3a_no_V1_bridge / bridged (max 5.6 mm); bridge(s) > 2 mm below the splice: 5.6 mm at s 5 (V1, 310.0 HU) |
| 140 | R | fail | FAIL: A1_v3loop_study_15_5 (12.9/3.8) | – | C6 / C7 | 9.8 / 6.3 | 12.9 / 3.8 | 0.98 | 235.0 | False | 7_v3loop_robust / bridged (max 8.2 mm) |
| 140 | L | donor | FAIL: 5a_V2_band | 5a_V2_band | C6 / C7 | 18.2 / 2.7 | 18.0 / 2.7 | 0.93 | 251.7 | False | – |
| 141 | R | donor_review | FAIL: 6_label_agreement_ge_0.9 | 6_label_agreement_ge_0.9 | C6 / C7 | 20.8 / 0.3 | 21.0 / 0.3 | 0.95 | 208.0 | True | Arterial VA label off the trace (0.90 within 2 mm) |
| 141 | L | donor | pass | – | C6 / C7 | 17.9 / 0.3 | 23.3 / 0.3 | 0.94 | 208.2 | True | – |
| 001 | R | donor_review | FAIL: 3a_no_V1_bridge; 6_label_agreement_ge_0.9 | 3a_no_V1_bridge; 6_label_agreement_ge_0.9 | C6 / C7 | 16.1 / 0.2 | 16.1 / 0.2 | 1.06 | 248.7 | False | Arterial VA label off the trace (0.73 within 2 mm); bridged (max 17.1 mm); bridge(s) > 2 mm below the splice: 2.1 mm at s 0 (V1, 314.0 HU) |
| 001 | L | fail | FAIL: 1_origin_le_2mm; 2_foramina_ge_4; 4a_landmarks_complete; 4b_origin_to_dura_150_260; 5a_V2_band; 5b_HU_ge_0.6_CCA; 6_label_agreement_ge_0.9; A1_v3loop_study_15_5 (0.8/20.2); A2_C1_threaded | – | None / None | 0.8 / 20.2 | 0.8 / 20.2 | 0.92 | 150.5 | False | 1_origin_le_2mm; 2_foramina_ge_4; 5a_V2_band; 5b_HU_ge_0.6_CCA; 7_v3loop_robust; 8_C1_threaded; origin -> C1 None mm (100-230) / Arterial VA label off the trace (0.00 within 2 mm) |
| 039 | R | donor_review | pass | – | C6 / C7 | 21.2 / 2.2 | 21.2 / 1.5 | 1.04 | 217.1 | True | bridged (max 11.1 mm) |
| 039 | L | donor_review | FAIL: 6_label_agreement_ge_0.9; A1_v3loop_study_15_5 (13.1/0.5) | 6_label_agreement_ge_0.9; A1_v3loop_study_15_5 (13.1/0.5) | C6 / C7 | 13.1 / 0.5 | 13.1 / 0.5 | 1.06 | 217.5 | True | Arterial VA label off the trace (0.87 within 2 mm); bridged (max 11.1 mm) |
| 041 | R | fail | FAIL: 3a_no_V1_bridge | – | C6 / C7 | 19.0 / 0.3 | 19.2 / 0.3 | 0.94 | 212.2 | True | 3a_no_V1_bridge / bridged (max 4.0 mm); bridge(s) > 2 mm below the splice: 4.0 mm at s 1 (V1, 177.0 HU) |
| 041 | L | fail | FAIL: 3a_no_V1_bridge; 6_label_agreement_ge_0.9 | – | C6 / C6 | 24.7 / 0.2 | 24.7 / 0.2 | 0.92 | 214.9 | True | 3a_no_V1_bridge / Arterial VA label off the trace (0.00 within 2 mm); bridged (max 8.3 mm); bridge(s) > 2 mm below the splice: 8.0 mm at s 1 (V1, 128.0 HU), 8.3 mm at s 11 (V1, 102.0 HU) |
| 073 | R | fail | FAIL: 4b_origin_to_dura_150_260; 6_label_agreement_ge_0.9; A1_v3loop_study_15_5 (6.9/7.4) | – | C6 / C7 | 5.5 / 7.4 | 6.9 / 7.4 | 0.99 | 167.8 | False | 7_v3loop_robust / Arterial VA label off the trace (0.64 within 2 mm); bridged (max 9.9 mm); bridge(s) > 2 mm below the splice: 9.9 mm at s 106 (V2, 47.0 HU) |
| 073 | L | fail | FAIL: 3a_no_V1_bridge | – | C6 / C7 | 25.7 / 0.3 | 26.3 / 0.3 | 1.00 | 183.2 | False | 3a_no_V1_bridge / bridged (max 6.5 mm); bridge(s) > 2 mm below the splice: 3.9 mm at s 0 (V1, -25.0 HU), 6.5 mm at s 77 (V2, 277.0 HU) |
| 190 | R | donor_review | FAIL: 6_label_agreement_ge_0.9 | 6_label_agreement_ge_0.9 | C6 / C7 | 21.7 / 0.5 | 21.7 / 0.5 | 0.98 | 209.7 | True | Arterial VA label off the trace (0.83 within 2 mm); bridged (max 9.2 mm) |
| 190 | L | donor_review | pass | – | C6 / C6 | 18.8 / 0.3 | 18.9 / 0.3 | 0.99 | 206.0 | True | bridged (max 9.2 mm) |
| 015 | R | donor_review | FAIL: 6_label_agreement_ge_0.9 | 6_label_agreement_ge_0.9 | C7 / C7 | 16.0 / 0.7 | 18.4 / 0.7 | 0.91 | 227.1 | True | Arterial VA label off the trace (0.00 within 2 mm) |
| 015 | L | donor | pass | – | C6 / C7 | 18.8 / 0.1 | 20.9 / 0.1 | 0.96 | 234.3 | True | – |
| 118 | R | fail | FAIL: 4a_landmarks_complete; 4b_origin_to_dura_150_260; 5b_HU_ge_0.6_CCA; A1_v3loop_study_15_5 (1.1/20.5); A2_C1_threaded | – | C6 / C7 | 1.1 / 21.9 | 1.1 / 20.5 | 0.99 | 217.7 | False | 5b_HU_ge_0.6_CCA; 7_v3loop_robust; 8_C1_threaded; origin -> C1 None mm (100-230) / bridged (max 8.2 mm); bridge(s) > 2 mm below the splice: 8.2 mm at s 114 (V2, 94.0 HU) |
| 118 | L | donor_review | FAIL: 6_label_agreement_ge_0.9 | 6_label_agreement_ge_0.9 | C6 / C7 | 16.5 / 0.3 | 16.7 / 0.3 | 1.00 | 250.7 | True | Arterial VA label off the trace (0.86 within 2 mm); bridged (max 6.8 mm) |
| 090 | R | donor_review | pass | – | C6 / C7 | 20.6 / 0.5 | 20.7 / 0.5 | 0.98 | 259.1 | True | bridged (max 9.9 mm) |
| 090 | L | fail | FAIL: 4a_landmarks_complete; 4b_origin_to_dura_150_260; 5b_HU_ge_0.6_CCA; A1_v3loop_study_15_5 (0.5/21.1); A2_C1_threaded | – | C6 / C6 | 0.5 / 21.1 | 0.5 / 21.1 | 0.97 | 223.3 | False | 5b_HU_ge_0.6_CCA; 7_v3loop_robust; 8_C1_threaded; origin -> C1 None mm (100-230) / bridged (max 8.5 mm); bridge(s) > 2 mm below the splice: 8.5 mm at s 129 (V2, 224.0 HU) |

### Wall per side (ct_seg4 stage-A, exact ownership k 64, acquisition voxel, native bars; flags W1 / W2 / W3-with-W1)

| case | side | gates V1–V6 (genus) | whole > 0.3 / 0.5 / 0.15 r | flags | donor stretch origin → C1f − 25 | flags | seam zone C1f − 25 → C1f | contact-free donor (area share) | C6 → C1 ct / rec | W7 0–40 / 40–80 | FWHM mm | noise HU | MISR mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 058 | R | pass (0) | 0.075 / 0.027 / 0.152 | nYn | 0.102 / 0.036 / 0.196 | YYY | 0.054 / 0.150 | 0.072 (0.49) | 0.091 / 0.088 | 0.028 / 0.036 (FAIL) | 1.380 | 27.6 | 1.459 |
| 058 | L | pass (0) | 0.250 / 0.133 / 0.277 | YYY | 0.330 / 0.183 / 0.363 | YYY | 0.074 / 0.062 | 0.276 (0.40) | 0.340 / 0.278 | 0.018 / 0.072 (FAIL) | 1.343 | 24.0 | 1.863 |
| 061 | R | pass (0) | 0.245 / 0.153 / 0.295 | YYY | 0.318 / 0.207 / 0.367 | YYY | 0.203 / 0.275 | 0.244 (0.36) | 0.306 / 0.300 | 0.027 / 0.139 (FAIL) | 1.315 | 19.4 | 1.611 |
| 140 | L | FAIL (1) | 0.214 / 0.145 / 0.301 | YYY | 0.320 / 0.226 / 0.420 | YYY | 0.066 / 0.177 | 0.209 (0.31) | – / – | 0.018 / 0.029 (FAIL, dura cut) | 1.330 | 29.5 | 1.304 |
| 141 | R | pass (0) | 0.125 / 0.068 / 0.157 | YYY | 0.149 / 0.090 / 0.177 | YYY | 0.162 / 0.210 | 0.032 (0.35) | 0.193 / 0.203 | 0.099 / 0.093 (FAIL) | 1.216 | 18.3 | 1.770 |
| 141 | L | pass (0) | 0.037 / 0.013 / 0.096 | nnn | 0.038 / 0.012 / 0.096 | nnn | 0.098 / 0.205 | 0.012 (0.68) | 0.065 / 0.062 | 0.012 / 0.065 (FAIL) | 1.198 | 23.8 | 1.332 |
| 001 | R | pass (0) | 0.227 / 0.147 / 0.260 | YYY | 0.248 / 0.165 / 0.288 | YYY | 0.211 / 0.220 | 0.153 (0.40) | – / – | 0.149 / 0.163 (FAIL, dura cut) | 1.217 | 23.7 | 1.800 |
| 039 | R | pass (0) | 0.094 / 0.029 / 0.185 | nYn | 0.092 / 0.029 / 0.182 | nYn | 0.071 / 0.151 | 0.078 (0.57) | 0.084 / 0.084 | 0.042 / 0.111 (FAIL) | 1.363 | 24.4 | 1.391 |
| 039 | L | pass (0) | 0.148 / 0.068 / 0.201 | YYY | 0.172 / 0.085 / 0.227 | YYY | 0.044 / 0.046 | 0.106 (0.47) | 0.194 / 0.173 | 0.051 / 0.092 (FAIL) | 1.406 | 22.6 | 1.637 |
| 190 | R | pass (0) | 0.055 / 0.014 / 0.108 | nnn | 0.072 / 0.019 / 0.139 | nnn | 0.013 / 0.030 | 0.056 (0.64) | 0.063 / 0.055 | 0.009 / 0.071 (FAIL) | 1.298 | 28.4 | 1.487 |
| 190 | L | pass (0) | 0.026 / 0.004 / 0.087 | nnn | 0.038 / 0.006 / 0.104 | nnn | 0.010 / 0.072 | 0.030 (0.80) | 0.028 / 0.027 | 0.005 / 0.005 (FAIL) | 1.326 | 30.8 | 1.394 |
| 015 | R | pass (0) | 0.171 / 0.083 / 0.274 | YYY | 0.212 / 0.111 / 0.320 | YYY | 0.307 / 0.455 | 0.131 (0.35) | 0.253 / 0.245 | 0.020 / 0.121 (FAIL) | 1.359 | 25.9 | 1.298 |
| 015 | L | pass (0) | 0.171 / 0.093 / 0.253 | YYY | 0.213 / 0.121 / 0.288 | YYY | 0.189 / 0.302 | 0.036 (0.28) | 0.262 / 0.258 | 0.049 / 0.089 (FAIL) | 1.400 | 22.7 | 1.428 |
| 118 | L | pass (0) | 0.247 / 0.126 / 0.296 | YYY | 0.294 / 0.153 / 0.349 | YYY | 0.256 / 0.301 | 0.249 (0.42) | 0.302 / 0.296 | 0.051 / 0.190 (FAIL) | 1.262 | 27.6 | 1.706 |
| 090 | R | pass (0) | 0.262 / 0.152 / 0.309 | YYY | 0.354 / 0.209 / 0.390 | YYY | 0.259 / 0.337 | 0.318 (0.53) | 0.312 / 0.312 | 0.128 / 0.021 (FAIL) | 1.115 | 25.8 | 1.617 |

### Gross shape per traced side (Arterial VMTK line, sigma 3)

| case | side | status | extent | length mm | DM | take-off 10 mm ° | SCA elevation ° | C2-loop bend mm | V1 / V2 / V3 / V4 mm | r × 1.10 mm |
|---|---|---|---|---|---|---|---|---|---|---|
| 058 | R | donor | origin -> confluence (whole VA) | 239.0 | 1.41 | 76.7 | 17.8 | 3.03 | 42.5 / 90.6 / 79.1 / 26.7 | 1.59 |
| 058 | L | donor_review | origin -> confluence (whole VA) | 250.9 | 1.45 | 58.3 | 28.9 | 2.75 | 49.8 / 94.3 / 88.9 / 18.0 | 2.20 |
| 061 | R | donor | origin -> confluence (whole VA) | 208.5 | 1.39 | 87.4 | -5.4 | 2.01 | 30.6 / 78.3 / 74.5 / 25.2 | 1.75 |
| 061 | L | fail | origin -> confluence (whole VA) | 216.0 | 1.41 | 114.7 | -12.3 | 2.58 | 33.2 / 84.2 / 69.9 / 28.7 | 1.91 |
| 140 | R | fail | origin -> dura only (course above C1 failed QC) | 175.7 | 1.26 | 55.1 | 47.3 | 1.52 | 48.0 / 89.2 / 38.5 / – | 1.51 |
| 140 | L | donor | origin -> dura only (course above C1 failed QC) | 202.2 | 1.70 | 79.6 | 21.7 | 1.47 | 39.2 / 85.2 / 77.8 / – | 1.32 |
| 141 | R | donor_review | origin -> confluence (whole VA) | 215.0 | 1.38 | 63.1 | 29.0 | 4.54 | 43.3 / 80.7 / 71.7 / 19.3 | 2.01 |
| 141 | L | donor | origin -> confluence (whole VA) | 214.0 | 1.36 | 56.7 | 37.3 | 4.00 | 44.5 / 78.0 / 70.2 / 21.4 | 1.41 |
| 001 | R | donor_review | origin -> dura only (course above C1 failed QC) | 183.8 | 1.59 | 95.7 | -26.0 | 2.81 | 23.8 / 91.3 / 68.7 / – | 2.05 |
| 001 | L | fail | origin -> confluence (whole VA) | 161.2 | 1.49 | 90.0 | 0.0 | – |  | 2.26 |
| 039 | R | donor_review | origin -> confluence (whole VA) | 223.8 | 1.37 | 73.9 | 43.8 | 3.98 | 45.3 / 85.9 / 76.4 / 16.3 | 1.53 |
| 039 | L | donor_review | origin -> confluence (whole VA) | 230.2 | 1.48 | 118.4 | 20.8 | 3.27 | 45.7 / 93.0 / 80.0 / 11.5 | 1.73 |
| 041 | R | fail | origin -> confluence (whole VA) | 220.0 | 1.48 | 104.1 | 21.2 | 3.68 | 34.6 / 88.1 / 79.7 / 17.8 | 1.38 |
| 041 | L | fail | origin -> confluence (whole VA) | 220.2 | 1.46 | 87.5 | 13.0 | 4.56 | 37.7 / 85.0 / 82.0 / 15.5 | 1.34 |
| 073 | R | fail | origin -> dura only (course above C1 failed QC) | 154.5 | 1.30 | 24.0 | 50.1 | 5.32 | 43.3 / 72.8 / 38.5 / – | 1.82 |
| 073 | L | fail | origin -> dura only (course above C1 failed QC) | 166.8 | 1.50 | 101.2 | 20.0 | 4.72 | 22.7 / 73.0 / 71.2 / – | 1.37 |
| 190 | R | donor_review | origin -> confluence (whole VA) | 215.4 | 1.31 | 67.3 | 38.1 | 4.23 | 43.1 / 82.8 / 76.7 / 12.8 | 1.61 |
| 190 | L | donor_review | origin -> confluence (whole VA) | 210.2 | 1.29 | 69.8 | 36.2 | 5.26 | 44.4 / 81.9 / 71.7 / 12.2 | 1.49 |
| 015 | R | donor_review | origin -> confluence (whole VA) | 235.2 | 1.43 | 68.6 | 31.8 | 1.95 | 30.1 / 99.9 / 80.9 / 24.3 | 1.59 |
| 015 | L | donor | origin -> confluence (whole VA) | 243.0 | 1.48 | 72.8 | 32.3 | 2.57 | 38.1 / 97.2 / 79.3 / 28.4 | 1.76 |
| 118 | R | fail | origin -> confluence (whole VA) | 224.0 | 1.22 | 138.9 | -37.1 | 3.64 |  | 1.99 |
| 118 | L | donor_review | origin -> confluence (whole VA) | 257.5 | 1.33 | 52.2 | 43.5 | 4.32 | 46.3 / 84.1 / 69.1 / 58.0 | 2.00 |
| 090 | R | donor_review | origin -> confluence (whole VA) | 269.7 | 1.53 | 46.4 | 41.1 | 2.68 | 46.6 / 98.4 / 80.0 / 44.8 | 1.89 |
| 090 | L | fail | origin -> confluence (whole VA) | 230.5 | 1.30 | 56.2 | 35.6 | 3.73 |  | 2.13 |

### Relief controls per walled side (from the records)

| case side | fitted > 0.3 mm after low-pass σ 0 / 0.75 / 1.5 mm | roughness 0.75-1.5 / 1.5-3 mm RMS | share of relief in contact bins | union contact / other bins | protrusion share (lumen-bright part) | plain-opening (nofix) whole union |
|---|---|---|---|---|---|---|
| 001 R | 0.214 / 0.182 / 0.140 | 0.129 / 0.148 | 0.75 | 0.287 / 0.145 | 0.018 (0.86) | fix 8 did not act |
| 015 L | 0.204 / 0.137 / 0.040 | 0.097 / 0.073 | 0.87 | 0.251 / 0.057 | 0.002 (0.85) | 0.081 on 83 mm |
| 015 R | 0.228 / 0.170 / 0.078 | 0.105 / 0.089 | 0.78 | 0.256 / 0.081 | 0.004 (0.82) | 0.058 on 77 mm |
| 039 L | 0.127 / 0.083 / 0.044 | 0.095 / 0.101 | 0.73 | 0.223 / 0.079 | 0.003 (0.76) | 0.166 on 129 mm |
| 039 R | 0.077 / 0.040 / 0.015 | 0.081 / 0.060 | 0.36 | 0.105 / 0.090 | 0.002 (0.57) | unchanged |
| 058 L | 0.241 / 0.192 / 0.120 | 0.110 / 0.113 | 0.66 | 0.305 / 0.187 | 0.005 (0.80) | unchanged |
| 058 R | 0.072 / 0.031 / 0.012 | 0.072 / 0.055 | 0.64 | 0.114 / 0.049 | 0.012 (0.70) | 0.054 (V1 lost) |
| 061 R | 0.238 / 0.189 / 0.137 | 0.134 / 0.154 | 0.69 | 0.301 / 0.170 | 0.015 (0.81) | fix 8 did not act |
| 090 R | 0.222 / 0.171 / 0.119 | 0.119 / 0.135 | 0.56 | 0.351 / 0.203 | 0.012 (0.81) | 0.263 (unchanged) |
| 118 L | 0.280 / 0.226 / 0.139 | 0.112 / 0.112 | 0.68 | 0.312 / 0.182 | 0.013 (0.84) | 0.249 (unchanged) |
| 140 L | 0.186 / 0.151 / 0.102 | 0.133 / 0.127 | 0.77 | 0.318 / 0.105 | 0.020 (0.83) | 0.034 on 88 mm |
| 141 L | 0.041 / 0.012 / 0.000 | 0.045 / 0.035 | 0.78 | 0.120 / 0.011 | 0.000 | 0.019 on 116 mm |
| 141 R | 0.136 / 0.100 / 0.052 | 0.085 / 0.088 | 0.77 | 0.204 / 0.059 | 0.001 | unchanged |
| 190 L | 0.028 / 0.007 / 0.000 | 0.044 / 0.034 | 0.37 | 0.062 / 0.019 | 0.000 | 0.029 on 103 mm |
| 190 R | 0.070 / 0.019 / 0.001 | 0.054 / 0.042 | 0.44 | 0.090 / 0.042 | 0.000 | unchanged |

Noise-floor phantoms (featureless tube, real PSF / noise / correlation, identical method): 061 R x1 0.0008 / x2 noise
0.019 (real 0.245); 140 L 0.005 (real 0.214). Real plain-tube bin fraction (10 mm bins < 0.01) 0.00-0.43 vs phantom 0.62-0.95.

## Ranking within the 172-series (the user's rule)

Ranking quantity = union > 0.3 mm of the donor stretch (origin -> C1f - 25 mm on the ct_seg4 centreline, rec landmarks)
over the 10 mm bins with < 5 % bright-neighbour edge rays (contact-free), area-weighted; only recalibrated-gate donors
with a ct_seg4 wall; pair score = mean of R and L (min also reported). The bin rule removes bone / bright-vessel /
bright-vein contacts; it does not remove dim venous plexus merged at half-max (the intermediate-HU 14-24 % of the
protrusion voxels on the coarse sides).

| rank | pair | score (mean) | min | R / L contact-free | R / L all bins | R / L status | draft passes both |
|---|---|---|---|---|---|---|---|
| 1 | 058 | 0.174 | 0.072 | 0.072 / 0.276 | 0.102 / 0.330 | donor / donor_review | no (L rule 6) |
| 2 | 039 | 0.092 | 0.078 | 0.078 / 0.106 | 0.092 / 0.172 | donor_review / donor_review | no (L rule 6 + V3 13.1) |
| 3 | 015 | 0.084 | 0.036 | 0.131 / 0.036 | 0.212 / 0.213 | donor_review / donor | no (R rule 6) |
| 4 | 190 | 0.043 | 0.030 | 0.056 / 0.030 | 0.072 / 0.038 | donor_review / donor_review | no (R rule 6) |
| 5 | 141 | 0.022 | 0.012 | 0.032 / 0.012 | 0.149 / 0.038 | donor_review / donor | no (R rule 6) |

Single sides that cannot form a pair here: 090 R 0.318, 118 L 0.249, 061 R 0.244, 140 L 0.209 (genus 1), 001 R 0.153.

## Method (per case; `pilot_proc_ia172_run.sh` / `run2.sh`, idempotent, per-case outputs and done-markers)

1. **TotalSegmentator 2.18.0 `total_v3`** on the CTA grid (GPU; tag_totalseg's recommendation: C1, CCA and skull closer to
   the v3 dataset than "total", and the V3-loop gate fails real loops on "total" masks), then the 1.5 mm RAS copy
   (`tag_totalseg_make1p5.py`, unchanged) for `art_extract --ts`. Input = raw HU: TotalSegmentator's CTNormalization clips
   to fixed training percentiles (lower bound >= -1024), so the [-1024, 3071] pre-clip is a no-op for it.
2. **Arterial** (`run_arterial.sh` unchanged, GPU, `--no-tta`, core stages) at the NATIVE grid (no `--resample-mm`),
   `--crop-k` from `arterial_ts_args.py` (unchanged: bottom slice a quarter into the two-limb arch range, top = skull top),
   `--fov-mm 220 --fov-center` = trachea centroid, `--skull-mask` = TS total_v3 skull (the tag_arterial best-known native
   protocol). **Input = the CT clipped to [-1024, 3071]** (`pilot_proc_ia172_clip.py`): Arterial's extracranial model uses
   nnU-Net ZScoreNormalization (`plans.json`, `use_mask_for_norm` false = whole-image mean/std) and Arterial has no HU clip,
   so the 172-series padding (-2048 HU, 21-31 % of voxels; 35 % of voxels below -1024 in 058) would have shifted its
   normalisation. One corrected crop if the arch check fails (never needed: 118, the arch-apex case, passed).
3. **Trace**: `pilot_proc_ia172_art_extract.py` = art_extract + the basilar-seed fix (seed window top z_FM + 40 instead of
   + 55, the docstring's value, which verify_tag_bench §4 showed fixes TaG-Net A's ACA seed; a seed component starting
   > 20 mm above the window bottom is down-weighted x 0.1; midline = TS spinal cord at z_FM), `--ts totalseg_1p5` ->
   `arterial/tsba` (primary). The unchanged art_extract is run too (`arterial/ts`). VMTK MISR on the cut-out stage-A
   surfaces (`art_vmtk_one.py` unchanged).
4. **QC** (`pilot_proc_ia172_qc2.py`): tag_calibration's harness and gate imported unchanged (`tag_calibration_paths.one`,
   `tag_calibration.augment / gate_info / apply_gate(..., "recalibrated")`). Values = tag_calibration.md RECOMMENDED:
   V3 loop robust cord >= 12 mm / <= 6 mm, C6 from Erec, C1 = C1f, V1 breaks <= 3 mm, 25 mm total counted below the
   splice, V2 band without IQR, label agreement and BCV >= 200 to review, length 150-260 (260-290 review). Added:
   native-CT HU along origin -> C1 vs the native CCA, >= 700 HU fraction, VMTK ok with MISR x 1.10 in 0.9-2.6 mm.
   **Donor = the cervical stretch origin -> C1f clean** (rules 1, 2, 3a-d, 5a, 5b, 7, 8 + VMTK + origin -> C1f
   100-230 mm); rule 4 and the course above C1 are graded separately (`distal_ok`, `whole_va_pass`). The draft gate
   (`apply_gate(..., "draft")` + A1 / A2) is recorded beside it.
5. **Wall** of every donor side: ct_seg4 (`pilot_proc_ia172_ct_seg.py` = tag_ct_seg.py = vmr_ct_seg3.py, identical
   parameters, `VMR_OPEN_BRIGHT=1`, + diff 7 seed repair and diff 8 connectivity-preserving opening), seeded from the
   tsba VMTK line (cut at the dura when the distal course failed); stage-A surface; VMTK MISR (`vmr_vmtk.py`); landmarks
   on three sets ("ts", "ct" = VMR's bony-ring method, "rec" = Erec / C1f); `va_metrics` unchanged with exact ownership
   (k 64) and the ACQUISITION voxel (in-plane pixel x 0.5 mm thickness) -> native thresholds; edge FWHM / image relief /
   noise / contacts (`pilot_proc_ia172_edge.py`). Where diff 8 acted, the plain method is kept as `ct_seg/ct_seg4_nofix`.
6. **Gross** of every traced side on its Arterial VMTK line (`pilot_proc_ia172_gross.py`), and of every walled side on its
   CT centreline (measure.json).
7. **Figures**: `qc/qc_<case>.png` (mask MIPs + CT slab MIP, traces coloured by QC status), `qc/<case>_wall3d_<side>.png`
   (wall coloured by |x - c| - r, plus the unrolled wall map), `qc/<case>_detail_along_va.png` (10 mm bins beside VMR
   and TaG-Net A).
8. **Controls**: noise-floor phantoms (`pilot_proc_ia172_phantom*.py`), protrusion (`pilot_proc_ia172_protrusion.py`),
   per-slice centroid / CPR looks (`pilot_proc_ia172_look.py`), no-label rescue retrace (`pilot_proc_ia172_rescue.sh`).
9. **Closing pass** (`pilot_proc_ia172_final.sh`, then `pilot_proc_ia172_final2.sh`): 058 wall with diff 8, nofix records
   for the 7 other cases where diff 8 acted, protrusion for all walls, figures again, re-measure of 140 / 001 (their
   measure.json predated the rec landmark set; old files kept as `measure_pre_rec.json`), collect, summary, grade.
10. **Grade and ranking** (`pilot_proc_ia172_grade.py`, new): both gates per side + the backed-only variant; W1 / W2 /
    W3-with-W1 at native bars on the whole VA and the donor stretch; seam zone separate; W7 by hand; the contact-free
    ranking quantity. The va_standard `grade` block inside each record carries the grader's voxel factors (W1 min
    0.0159 at 0.5 mm, wrong-signed per the report) and is NOT used.

Compute: TotalSegmentator 695-1112 s per case with the per-class writer, 205-294 s with `--ml`; make1p5 139-263 s;
Arterial 245-336 s on 9 cases, 954 s on 118, and two centreline-stage hangs stopped after 1293-1605 s; trace 150-440 s
per mode; ct_seg4 + VMTK + landmarks + measure + edge 4-15 min per walled case.

## History (interim answers, kept as written)

Interim 2 (01:25 EDT Oct 2, fourth instance; all 12 cases through QC, wall pending only for the closing pass):
- QC, all 24 sides: recalibrated gate 15 / 24 donors (62.5 %; 058 R L, 061 R, 140 L, 141 R L, 001 R, 039 R L, 190 R L,
  015 R L, 118 L, 090 R), 5 pairs (058, 141, 039, 190, 015). Study draft + additions: 7 / 24 (29 %), 0 pairs. The gates
  differ on 8 sides; 6 of them only through rule 6 (label agreement -> review; two-patient-backed), 001 R through
  V1 <= 3 mm and 140 L through the dropped IQR term (both provisional).

Interim (21:58 EDT Oct 1; 2 of 12 cases through QC and wall):
- 058: both sides pass as v5 donors (R clean; L with a label-agreement review flag). 061: R passes; L fails only on a
  5.6 mm V1 bridge at the origin whose native HU is 310 (an enhanced lumen: an Arterial mask break, rescuable on review).
- Entry level: ts_qc calls C7 on all four sides; the recommended Erec rule calls C6 on all four (the known over-call).
- The ct_seg4 walls of these sides read union > 0.3 mm 0.245-0.250 over the whole VA and 0.31-0.34 over C6 -> C1
  (CT landmarks), about 2x VMR 0248 left (0.138) and TaG-Net A left (0.111). [Note 02:15: the 0.250 is 058 L; 058 R
  with diff 8 reads 0.075.] A featureless-tube phantom of 061 R with the same edge FWHM (1.32 mm), noise level and noise
  correlation reads 0.0008 through the identical method: the relief is image structure, not noise. Half of it survives a
  1.5 mm low-pass (coarse, not pits).
- Two method problems found and fixed in the pilot copies (documented diffs 7 and 8): NaN VMTK seeds, and the ct_seg4
  opening dropping a whole 52 mm V1 (058 R) at a bone contact. Arterial's centreline stage hung on 061 (stopped; the
  binary mask was traced without labels).

What was inherited (third instance): TotalSegmentator total_v3 for 058 061 140 141 001 039 041 073 118 from the first
instance's batch (090 had failed in 2 s on a docker-inspect race in the copied GPU guard; 090, 190, 015 were run with
`--ml`). The first instance's scripts were reused where sound; its QC script (draft thresholds, written before
tag_calibration.md existed) was replaced by `pilot_proc_ia172_qc2.py`.

## Open issues

- **Contamination is only partly excluded.** The contact-free bins remove bone and bright-vessel contacts at 10 mm
  resolution; dim venous plexus merged at half-max is not removed (14-24 % of the protrusion voxels on the coarse sides
  are intermediate-HU). A per-ray or per-triangle exclusion would be sharper. Rankings are within source only.
- **140 L** has a genus-1 handle inside origin -> C1 (V2); it needs a topology repair before any bake, and its donor
  stretch values come from 10 mm bins (approximate). The 001 R and 140 L walls stop at the dura cut.
- **Review flags carried by the 5 pairs**: rule 6 (label agreement) on 058 L, 039 L, 015 R, 190 R, 141 R; bridges above
  the splice on 039 and 190 (both sides); 015 R Erec entry C7 (check by eye).
- **Rescue candidates** (not done): 061 L (5.6 mm origin break at 310 HU) and 041 R (4.0 mm at 177 HU) by re-joining
  the mask; 140 R origin -> C1 if the splice is at or below C1f (V3 loop skipped above C1); 090 L / 118 R origin -> C2 only.
- **Arterial labels**: 061 and 090 have none (centreline stage hung; stopped); rule 6 could not apply there.
- **Gate thresholds**: this pilot is one out-of-sample test of the recalibration. It supports rule 6 -> review and a
  floor below 15 mm; it gives one data point for d (140 R, robust d 6.3 vs 6) and none for the IQR drop or V1 <= 3 mm
  beyond the two sides they admit.
- **W7** fails everywhere (smooth CT V4): irrelevant for strategy B, but no whole-VA donor comes from this source.
- 090 R length 269.7 mm (260-290 review band); G2 < 2.5 mm on 015 R, 061 R, 140 L (C2 loop).

## Files

- Results: `/sp/v5_va/results/pilot_proc_ia172.json` (cases + 24 per-side records), `pilot_proc_ia172_grade.json/.txt`,
  `pilot_proc_ia172_summary.json/.txt`.
- Per case: `D:\Arjun\neve\carotid_data\va_sources\derived\thin_slice\ia172\<case>\` with `totalseg/`, `totalseg_1p5/`,
  `arterial/{runs/native,tsba,ts}`, `qc/qc.json`, `qc/gross.json`, `qc/qc_<case>.png`, `qc/<case>_wall3d_<side>.png`,
  `qc/<case>_detail_along_va.png`, `ct_seg/ct_seg4/{seg_qc,landmarks,measure,edge,protrusion}.json` + surfaces,
  `ct_seg/ct_seg4_nofix/` (plain opening), `phantom/` (061, 140); logs `_logs_proc/` (timing.jsonl, final.log,
  final2.log).
- Scripts: `/sp/v5_va/pilot_proc_ia172_*` (run.sh, run2.sh, final.sh, final2.sh, qc2.py, ct_seg.py, ctseg_common.py,
  measure.py, landmarks.py, edge.py, gross.py, fig.py, collect.py, summary.py, grade.py, mdtable.py, phantom*.py,
  protrusion.py, look.py, rescue.sh, watchdog.sh, clip.py, args.py, tsrun2.sh, tssplit.py).
