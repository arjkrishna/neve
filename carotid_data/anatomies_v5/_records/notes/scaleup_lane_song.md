# scaleup_lane_song: the Song lane of the v5 VA donor scale-up

Agent prefix `scaleup_lane_song`, started 03:30 EDT 2026-10-02. State file `/sp/v5_va/state/scaleup_lane_song.md`.
No git; no tracked file touched. Data under `carotid_data/va_sources/{thin_slice,derived/thin_slice}/song/<case>/`
(gitignored), same layout as the pilot. Licences: Song 2024 CC BY 4.0 (figshare 26965450); everything derived through
Arterial is non-commercial.

## Answer so far

**Final (13:55, lane complete).** All 25 accepted Song cases have been graded under pipeline v2, and step 2 has run on
all of them. 010 was rejected at fetch and 013 took its place, so 26 cases were fetched.
- **Lane pairs eligible: 5.** 002, 040, 095, 108 and 138. Every flagged side was checked by eye and kept.
- **Song total in the tally: 12 eligible pairs.** That is the 5 lane pairs plus 7 pilot pairs (008, 032, 094, 100, 109,
  113, 141).
- **Combined tally** (`results/scaleup_tally.json`, snapshot `scaleup_tally_song_final25_1355.json`):
  - 27 eligible pairs: 14 ia172, 12 Song, plus VMR 0248.
  - 26 of them have all-bins donor-stretch union > 0.027 (coarser than the dense host mesh) on both sides. Song 138
    is the one smooth pair.
  - **Stop rule NOT met:** eligible 27 < 30, short by 3. The >= 25 coarse condition is met (26).
  - Both lanes are at the 25-case cap, so by SCALEUP_PIPELINE.md sec 4 the lanes stop and report the shortfall.
- **Why the Song yield is low:** 8 lane cases pass QC on both sides but have a wall that fails gate V4 (genus >= 1):
  003, 004, 030, 065, 111, 117, 120 and 128. The remaining 12 lane cases fail QC on one or both sides.
- **The last four cases:**
  - 128: QC pair (donor / donor, contact-free 0.23 / 0.18), but the R wall has genus 1, so it is not eligible.
  - 108: eligible (R rule-6 review kept by eye; all-bins 0.12 / 0.14).
  - 012: R fails 3c, a V3-low bridge over 10 mm.
  - 013: R fails because Arterial's mask has no R VA below C2.
- **By-eye work this instance:**
  - Fetch review of 012, 108, 128 and 013, all accepted. 012 is low contrast on the 065 precedent: TS-mask medians are
    aorta 221 and CCA 267 / 253 HU, against 217 / 220 / 203 for 065 and 189 / 202 / 208 for the rejected 010.
  - Flagged sides kept: 002 R/L and 108 R, plus the pilot Song pairs the tally listed as pending: 113 R/L, 100 R/L,
    094 L and 008 R/L.
  - No side was rejected by eye.

(03:40) Batch 1 (18 cases) launched: fetch, TotalSegmentator, Arterial and trace chains running detached. Grade
stages (QC onward) wait for pipeline v2 (`notes/SCALEUP_PIPELINE.md`, fixes agent).

(06:30, relaunched agent after the 04:57 session limit; the chains survived.) Fetch finished 05:44: 18/18 fetched with
CRC OK; by-eye review 17 accepted, 0 rejected, 065 pending (its arch lies at the image bottom, so the automatic aorta
track found no limb). Two chains had stalled on the arch-top failures: TS on 065 from 05:29 and Arterial on 030 from
04:59. Both were unblocked at 06:18-06:25 (`scaleup_song_refqc.sh`). Arterial done 5/18 (111 138 096 120 020); TS
done 8/18. Grading = pipeline v2 via `scaleup_song_gradechain.sh` → `scaleup_song_grade.sh` → `scaleup_v2_post.sh song
post2`; it waits for `_logs_scaleup/.grade_go`. The pilot grade group of `scaleup_song_run.sh` is never run on scale-up
cases.

(09:55, third agent instance after the ~09:21 cut; chains survived.) Fetch review 18/18 accepted (065 low contrast,
flagged). Graded under v2 by 09:42: 10 cases; pairs with both sides passing QC: 030, 111, 095, 120 (all-bins > 0.1 on
both sides), 040 (all-bins 0.046 / 0.099, contact-free near 0), 138 (R wall unmeasured; fallback running). Failing:
096 (both sides), 020 (R), 137 (R). All flagged sides of those pairs checked by eye and kept
(`results/scaleup_eye_rejections.json`). Two gaps found and fixed:
- **065 was never graded.** Its Arterial run crashed on a NaN FOV centre: the pilot args fallback takes the trachea
  centroid over the lowest 60 mm, and TotalSegmentator found the trachea only at k 98-121. The fix is
  `scaleup_song_args.py`, which changes the fallback branch only. The case is now being re-run (Arterial, trace, then
  its first real v2 grade under the same rules).
- **Step 2 (the ct_seg4 fallback) had not run on any Song case.** It now runs on every case
  (`scaleup_song_postchain.sh`), starting with 138 R.

Extension cases 19-22 (007 009 011; 010 pending a verdict) have been in TS / Arterial / trace since 09:21. The GPU
would otherwise have sat idle, and the stop rule cannot be met at 18 cases.

(11:15) Graded under v2: 13 of 18 batch-1 cases. Pairs with both sides passing QC and a wall:
- 030, 111, 117, 095, 120, 002, 003, 040 (all-bins > 0.027 on both sides)
- 138 (smooth; its R wall was re-cast by the step-2 fallback, measured but truncated at 131 mm)

No pair: 096, 020, 137, 025. All flagged sides of these pairs were checked by eye and kept. Still grading: 065
(re-run), 004, 005, 125, 130.

- 010 was rejected at the fetch review: the aorta is 189 HU and the CCAs 186 / 226 HU, and dense venous collaterals
  hide the VAs. It is replaced by 013 (#26).
- Cases 012, 108, 128 and 013 are being pre-fetched and run through TotalSegmentator, Arterial and trace. Their Arterial
  runs wait for the by-eye fetch acceptance.
- Grading now runs mostly in two light 2-CPU slots (the 172 lane's design: resources, not rules). Two grade chains
  share the case list through per-case claims.

(13:20, fourth instance; the third stopped at 11:15 on a permission denial, its detached chains finished 12:21.)
21 cases graded under v2 with step 2: batch 1 (18, incl. 065 after its Arterial re-run) + 007 009 011. Cases 23-26 in
selection order (012 108 128 013; 010 rejected) were traced but not fetch-reviewed: reviewed by eye now, all four
accepted (012 low contrast like 065: aorta 221, CCA 267 / 253 HU on TS masks; 108's auto arch top was 44 mm low, crops
re-made). That is 25 accepted cases = the lane cap; nothing more is fetched. The four are being graded one at a time
(`scaleup_song_grade4.sh`, unpatched `scaleup_v2_post.sh song post2` + `scaleup_ctseg_fallback.sh`).
Tally 12:59 (before them): Song lane pairs eligible = 002, 040, 095, 138 only. 003 004 030 065 111 117 120 pass QC on
both sides but a wall fails gate V4 (genus >= 1), so they are not eligible.

## Cases (selection order, Song entries of `results/pilot_next_cases.json -> selection`)

Batch 1 (1-18): AHMU1218111 138 096 120 (Siemens 0.7 mm), 020 030 095 040 065 (GE 0.625), 137 117 (Philips 0.45),
025 002 003 004 005 (GE), 125 130 (Philips). Extension in steps of 4 (19-25): 007 009 010 011 012 108 128; 010 rejected at fetch, so 013 (#26) was added
for 25 accepted cases (the cap).

## Method

Pilot pipeline, unchanged scripts unless named `scaleup_*`:
- **Fetch** (pilot_fetch_song method): RAR5 range read of the stored member + CRC32 (`pilot_fetch_song_get.py`), QC pass
  1 with MIPs and side files (`pilot_fetch_song_qc.py`), column-frame VA check (`_va_check2.py`), zoomed VA crops
  (`_va_zoom.py`), CCA/SCA picks + `meta.json` (`scaleup_song_fetch_picks.py`, `scaleup_song_fetch_meta.py` = the pilot
  scripts with scale-up result paths). By-eye acceptance review of the MIPs and VA crops ->
  `results/scaleup_song_fetch_review.json`.
- **Process**: `scaleup_song_run.sh` = `pilot_proc_song_run5.sh` + fetch stages, own log dir (`_logs_scaleup`), CPU
  containers capped at 6 CPU / 12 GB with a RAM wait, Arterial on `ct_negzero_air1000` (copy deleted after the run), a
  protrusion stage, `scaleup_song_collect.py` -> `results/scaleup_song.json`. Detached per-group loops
  `scaleup_song_chain.sh <fetch|ts|art|trace|grade>`.

## Decisions

- A case rejected at the by-eye fetch review is skipped and the next case in order takes its place; the 18 / 25 counts
  are of accepted, processed cases.
- Pilot-only studies (plain ct_seg4_nofix record, clipped-vs-re-windowed comparison) are not run at scale.
