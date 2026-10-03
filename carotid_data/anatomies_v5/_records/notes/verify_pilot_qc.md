# verify_pilot_qc: adversarial check of the pilot decision's QC, counts, pairs, donor need and go / no-go

Agent prefix `verify_pilot_qc`, first instance, 02:36–03:35 EDT 2026-10-02. State file `/sp/v5_va/state/verify_pilot_qc.md`.
No git, no GPU, no downloads, no tracked file touched. All the code is new (`/sp/v5_va/verify_pilot_qc_*.py`). It ran in
python:3 or neve-build-stagea, one CPU container at a time, and only read the pilot's files.

**Diff of this file:** the 02:55 "Answer so far" is replaced by the final Answer below. Its findings are kept, and the CT
checks are added.

## Answer

Status: complete (03:35).

**Verdict: the claim holds on every key number. Two disclosure gaps and three minor misstatements need fixing.**

Every count reproduces with my own code:
- the donor need;
- per-side QC under all three gates;
- the pass and pair counts;
- the gates-differ lists;
- the ranking arithmetic and the Wilson intervals.

An independent CT check of all 46 traced sides finds:
- no wrong-vessel trace counted as a VA;
- no venous trace;
- no wrong side or frame;
- no resampled grid presented as the acquisition voxel.

There is no cherry-picking: all 12 processed 172-series cases are in the denominator, and 166 was replaced before
processing because it was truncated.

### What reproduces

1. **Donor need: 25 pairs, 30 with spares** (`verify_pilot_qc_need.py`).
   - The v4 set has 223 anatomies from 25 distinct TopBrain patients: 103 native and 120 mirrored, the same as `BUILD_v4.json`.
   - Both vertebral pieces are used in all 223 anatomies, and the follow rule holds in all 223:
     - native: R comes from R, L from L;
     - mirrored: R comes from L, L from R.
   - The per-patient counts equal the claim's 25 rows. The distribution is 16 ×4, 15 ×4, 12, 10, 9, 8 ×3, 7, 6, 5 ×2,
     4 ×4, 2 ×2 and 1.
   - mr_006 is used only native; mr_013, 014, 015 and 025 only mirrored.
2. **Per-side QC verdicts** (`verify_pilot_qc_recount.py`).
   - I re-implemented the gates from the raw values in every per-case `qc/*.json`, without importing any pilot gate code.
   - My verdict equals the stored one on 44 of 44 thin-slice sides, for both the recalibrated donor gate and the draft
     plus additions.

   | source | recal sides | recal pairs | draft sides | draft pairs | backed sides | backed pairs |
   |---|---|---|---|---|---|---|
   | 172-series | 15 / 24 | 015, 039, 058, 141, 190 | 7 / 24 | none | 14 / 24 | the same 5 |
   | Song | 15 / 20 | 008, 032, 094, 100, 109, 113 | 5 / 20 | 032 | 13 / 20 | 008, 032, 094, 113 |

   - AortaSeg (its own gate re-implementation, from its notes) is 1 / 0 / 1.
   - Totals are 31 / 12 / 28 sides and 11 / 1 / 9 pairs.
   - The gates-differ lists match: 8 sides on the 172-series and 10 on Song.
3. **CT along the trace, all 46 sides.**
   - Scripts: `verify_pilot_qc_ct.py`, `_v3.py` and `_final.py`. Output: `verify_pilot_qc_out/final_sides.json`.
   - My own course test requires all of the following:
     - the first point within 2 mm of the TS subclavian on the same side;
     - at least 4 of C6..C1 threaded (at least 2 mm of path inside the TS vertebra mask, which fills the foramina);
     - a cord-relative V3 posterior excursion of at least 12 mm, and at most 6 mm from the C1 posterior arch;
     - origin → C1 HU at least 0.6 × the eroded TS CCA;
     - no non-lumen run below C1 (HU < max(150, 0.4 × CCA) for more than 1.5 mm);
     - the correct side of the cord;
     - less than 10 % of s 15 → C2 within 1 mm of a TS CCA.
   - **All 31 sides the pilot passes also pass my test.** That includes:
     - all 14 rule-6 sides (the 172-series 6 and Song 8);
     - the 4 sides admitted only by the provisional limits (001 R, 140 L, S100 R, S109 R);
     - the 4 Song sides on the ts trace.
   - Across those 31 sides:
     - origin 0.0–0.7 mm (AortaSeg L 1.47 mm);
     - threaded 6 / 6 (Song 106 L 5);
     - V3 13.3–29.4 mm, at 0.0–4.0 mm from the arch;
     - HU 0.92–1.25 × CCA;
     - V2 within 2 mm of a vertebra on 73–100 % of points;
     - the nearest CCA 4.8–15 mm away.
   - **11 of the 15 failing sides also fail my test, for the stated reasons:**
     - 001 L: origin 61 mm from the subclavian, 0 levels threaded;
     - 073 R, 090 L, 118 R, 140 R, Song 090 R / L and AortaSeg R: V3 off the C1 loop (d 9–34 mm);
     - 041 L, 073 L and Song 106 R: non-lumen V1 bridges at −25 to 128 HU.
   - **The other 4 failing sides pass my test.** They are exactly the V1 mask-break sides, whose bridges run through
     enhanced lumen: Song 044 R (480 HU), Song 141 R (587), 061 L (310) and 041 R (177).
   - **Song 094 L V1** (`verify_pilot_qc_prof.py`): over s 11–38 the trace reads 650–820 HU, with a ≥ 1500–3071 HU neighbour
     within 2–4 mm. It runs at the edge of the dense vein, not inside it. The pilot already flags it for review (10.4 % of
     points ≥ 700 HU).
4. **Ranking arithmetic and supporting checks.**
   - Ranking (`verify_pilot_qc_rank.py`): all 11 pair scores equal the mean of the grade files' per-side contact-free values.
     The order, the merged build order and the "coarse enough on both sides" set (7 thin-slice + VMR = 8) all match.
   - The Wilson intervals reproduce: 5/12 0.19–0.68, 6/10 0.31–0.83, 11/22 0.31–0.69, 9/22 0.23–0.61, 7/22 0.16–0.53.
   - Acquisition voxel (`verify_pilot_qc_vox.py`):
     - The 172-series grids are 0.25–0.4 mm in z (overlapping reconstruction), but the measure uses the coarser 0.5 mm slice
       thickness.
     - On Song, the acquisition voxel equals the grid.
     - Song's QC and wall read the shipped CT, while Arterial ran on air1000 (10 / 10 per `input_used.json`).
5. **Next-case list** (`verify_pilot_qc_next.py`).
   - 197 distinct ids (122 Song + 75 172-series), in contiguous order.
   - No overlap with the 22 processed pilot cases or with the rejected ia172_166.
   - No identical sex / age / grid duplicates.
   - The first ten and the 18 + 18 split of orders 1–36 match the claim.

### What is wrong or missing

1. **Undisclosed: Song's counts rest on a best-of-two trace choice.**
   - `qc/select.json` grades the protocol trace (tsba) and the unchanged tool's trace (ts), and keeps the better of the two
     per side. 4 Song sides use ts: 008 R / L and 100 R / L.
   - On tsba, 008 L and 100 L fail:
     - 008 L: V3 0.7 / 21.0, C1 not threaded;
     - 100 L: V3-low bridge, V3 2.8 / 13.2.
   - **Song on the protocol trace alone:**
     - recalibrated 13 / 20 (65 %, still go), 4 pairs (032 094 109 113);
     - backed-only **11 / 20 (55 %, no-go)**, 3 pairs.

     The claim's "backed: Song go (65 %)" therefore depends on the choice.
   - The chosen ts traces are correct VAs on CT, so this is a legitimate rescue, not wrong-vessel counting.
   - But the choice does not appear anywhere in `pilot_decide` (per_source, ranking, verdict, or the "keep from the pilot"
     list), and 2 of the 6 Song pairs (100 and 008) and 2 of the 12 eligible pairs depend on it.
   - The scale-up must keep both traces and the selection, or the yield estimates do not transfer.
   - It made no difference on the 172-series: ts = tsba below the splice on every failing side.
2. **The 172-series backed count of 14 / 24 includes 140 R.**
   - The recalibrated gate rejects 140 R. `backed_only()` drops rule 7 and re-tests V3 on the study cord (12.9 / 3.8), which
     passes.
   - My cord V3 reads 12.4 / 13.4: the trace leaves the C1 loop, with an 8.2 mm bridge at 48 HU above C1.
   - Without it the count is 13 / 24 (54 %). The "no (just short)" verdict stands either way, but 140 R is not in the
     claim's gates-differ list.
3. **Minor misstatements.**
   - **Batch size.** The claim says "Batch 1 is orders 1–36", but `pilot_next_cases.json → batch_plan.first_batch` is m = 44
     (the P80). The notes say 36. The json and the text disagree.
   - **041 R is borderline, not "not lumen".** Its V1 bridge reads 177 HU = 0.47 × CCA, and the trace passes my course
     test. It is a possible third rescue alongside 061 L and Song 141 R.
   - **Song 090 R has more failures than its category says.** It fails origin (4.5 mm) and a 31 mm non-lumen V1 bridge
     (26 HU) as well as the V3 loop.
4. **Residual risks** (no change to the numbers).
   - **Duplicate patients.** The 172-series dedup is a heuristic: same sex, age ±1, model and table height ±5. Same sex +
     age + model groups with different table heights remain in the list (125 / 127 F 92, 146 / 180 F 85, 064 / 066 F 87,
     156 / 152 F 82).
   - **Sample size.** The report asked for a 25-case pilot; 23 cases were run.
   - **The 172-series "go" has no margin.** It is 15 / 24 = 62.5 %, and dropping either provisional-only side (001 R or
     140 L) gives 58 %.

### Corrected statement

- The donor need (25 pairs, 30 with spares), the per-side QC under all three gates, the pass and pair counts (31 / 12 / 28
  sides, 11 / 1 / 9 pairs), the ranking and the yield intervals are right. They reproduce from the per-case files, and the
  CT confirms that every passing side is a VA course.
- Go under the recalibrated gate is right for both sources.
- Song's 6 pairs and its backed-only "go" use a best-of-two choice between the tsba and ts traces. On the protocol trace
  alone, Song gives 4 pairs and backed-only 55 % (no-go).
- The 172-series backed 14 / 24 includes 140 R, whose course leaves the C1 loop (13 / 24 without it).
- Batch 1 is 36 cases in the text but 44 in the json.

## Checks (all own code, read-only)

| # | check | result |
|---|---|---|
| 1 | donor need from 223 `provenance.json` + `BUILD_v4.json` | reproduced exactly |
| 2 | gate re-application from raw qc values, 44 sides × 3 gates | 44 / 44 equal to stored; counts and pairs reproduced |
| 3 | gates-differ lists | reproduced; backed-only adds 140 R (not listed) |
| 4 | Song trace selection | best-of-two on 4 sides; undisclosed in claim; protocol-only yield lower |
| 5 | CT origin on own subclavian, all sides | passing 0.0–0.7 mm (AoD L 1.47); 001 L 61 mm, Song 090 R 4.5, 118 R 2.6 |
| 6 | CT foramen threading C6..C1 | own = qc on 43 / 44 thin-slice sides (Song 106 L 5 vs 6); AortaSeg own 6 (L) / 4 (R) |
| 7 | CT V3 loop (cord-relative, own) | passing 13.3–29.4 / 0.0–4.0; failing V3 sides d 9–34 |
| 8 | CT HU vs CCA, ≥ 700 HU, non-lumen runs, bridge HU | passing 0.92–1.25 × CCA, no non-lumen run; 094 L vein edge; failure HU match the claim (041 R borderline) |
| 9 | CCA / BCV contact, laterality | 0 % within 1 mm of CCA on s 15 → C2 for all passing sides; all sides on the correct side |
| 10 | length | own arclength = qc length on all 44 thin-slice sides |
| 11 | QC PNGs viewed | 172-series 001, 015, 039, 041, 058, 061, 073, 118, 140, 141, 190; Song 008, 094, 100, 106, 113, 141; AortaSeg: VA courses on passing sides |
| 12 | ranking, coarse-enough set, Wilson intervals | reproduced |
| 13 | next-case list distinct / no overlap | yes; dedup heuristic caveat; batch 36 vs 44 |
| 14 | acquisition voxel vs grid | honest (0.5 mm thickness on the 172-series, grid on Song) |
| 15 | cherry-picking of cases | none: 13 fetched, 1 rejected before processing (arch), all 12 processed counted |

## Files

- Scripts: `/sp/v5_va/verify_pilot_qc_{peek,side,recount,need,next,rank,ct,ctsum,v3,final,vox,prof}.py` and
  `verify_pilot_qc_{ct,v3}_batch.sh`.
- Outputs:
  - `/sp/v5_va/verify_pilot_qc_out_recount.json` (gate recount);
  - `/sp/v5_va/verify_pilot_qc_out/ct_<case>.json`, `v3_<case>.json`, `ct_summary.json`, `final_sides.json`;
  - logs `ct_batch.log`, `v3_batch.log`.
