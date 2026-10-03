# scaleup_lane_ia172: the 172-series lane of the v5 donor scale-up

Agent prefix `scaleup_lane_ia172` (first instance started 03:29 EDT 2026-10-02). State file
`/sp/v5_va/state/scaleup_lane_ia172.md`. No git; no tracked file touched; pilot files are not modified (new scripts are
`scaleup_ia172_*`). Data under `D:\Arjun\neve\carotid_data\va_sources\thin_slice\ia172\` and
`...\derived\thin_slice\ia172\` (gitignored; pilot layout). Logs `derived/thin_slice/ia172/_logs_scaleup/`.
Licence: Head-CTA 172-series CC BY 4.0 (Zenodo 15697196); everything through Arterial is non-commercial.

## Answer (final, 13:05)

The lane is complete: **25 cases processed, the per-lane cap. 9 new eligible donor pairs.**

**Pairs:** 136, 149, 159, 164, 165, 172, 192, 069 and 071. All 9 have an all-bins donor-stretch union above 0.027
on both sides.

| pair | contact-free score: mean (min side) | all-bins R / L |
|---|---|---|
| 069 | 0.132 (0.123) | 0.279 / 0.156 |
| 071 | 0.132 (0.063) | 0.104 / 0.275 |
| 165 | 0.121 (0.106) | 0.226 / 0.150 |
| 164 | 0.109 (0.091) | 0.113 / 0.148 |
| 192 | 0.107 (0.058) | 0.086 / 0.152 |
| 172 | 0.091 (0.079) | 0.169 / 0.128 |
| 149 | 0.090 (0.063) | 0.282 / 0.114 |
| 136 | 0.087 (0.079) | 0.178 / 0.178 |
| 159 | 0.070 (0.051) | 0.168 / 0.191 |

**Results:**
- `results/scaleup_ia172.json`: 50 records = 25 cases x 2 sides (25 donor_review, 12 donor, 13 fail).
- `results/scaleup_tally.json`: the final tally, 13:03. A copy is `results/scaleup_tally_ia172_final25_1303.json`.

**By-eye checks:** every flagged side of the 9 pairs was looked at, and all were kept.
- That includes the tally's "passes only via provisional limits" flags: 136 L, 149 L, 164 L, 165 L, 172 L and 071 L.
- The verdicts are in `results/scaleup_eye_rejections.json`. The batch files are `results/scaleup_ia172_eye/eye_<case>.json`.
- No pair of this lane has a pending eye check.
- **071 L is kept but flagged.** It has a 2.8 mm V1 gap at the origin, at 60 HU (0.16 x CCA). Above that gap the
  trace sits in an oval enhanced lumen at 380-470 HU, and that lumen is the VA (`qc/origin_look_ia172_071_L.png`). The
  gap is either an ostial narrowing or the trace cutting through the SCA wall to a nearby ostium. Rank it below
  unflagged sides of equal score.

**Yield:**

| | cases | pairs | pairs per case |
|---|---|---|---|
| batch 1 | 18 | 2 (136, 149) | 0.11 |
| extension (cases 19-25) | 7 | 7 | 1.0 |
| whole lane | 25 | 9 | 0.36 |
| 172-series pilot | | | 0.42 |

- In batch 1, 11 cases have a side that fails QC: 062 060 160 137 044 174 068 080 138 139 150. Rule 7 (V3 loop) is the
  most common failure, on 8 of them.
- 7 batch-1 cases pass QC on both sides. 5 of those 7 lose the pair to wall gate V4 (genus >= 1): 065 R, 067 L, 124 R,
  155 R + L, 158 L. 068 R and 139 R are also genus failures, but their other side fails QC anyway.
- None of the 14 extension walls fails genus.
- Contrast does not explain the difference. The extension cases have CCA 348-522 HU, and batch-1 cases with equal or
  stronger contrast still failed (062 at 800 HU, 138 at 564, 150 at 511). Seven cases is a small sample.

**Tallies** (both lanes + pilot + VMR 0248):

| when | eligible pairs | all-bins > 0.027 on both sides | contact-free >= 0.027 on both sides | stop rule |
|---|---|---|---|---|
| after 18 (11:30) | 19 | 18 | 11 | not met |
| after 22 (12:30) | 23 | 22 | 15 | not met |
| after 25 (13:03) | 26 (172-series 14, Song 11, VMR 1) | 25 | 18 | not met |

- Copies of the earlier tallies: `results/scaleup_tally_ia172_batch1_1130.json` and `..._after22_1232.json`.
- At 13:03 the stop rule was short by 4 eligible pairs; the coarseness part (>= 25 pairs) was already met.
- This lane hit its cap (`stop_rule.lanes.ia172.cap_reached`). The Song lane was at 21 of 25 cases, so it decides the rest.
- The 26 pairs cover the 25 TopBrain patients, but not the 30 wanted with spares.

**Pipeline health:**
- Arterial ran on 25/25 cases, with no watchdog stop.
- The step-2 VMTK fallback was needed once: on 124 R the centreline was 0.5 mm. The fallback wall then failed V4.
- No Arterial clip or resample copy is left on disk.
- 119 was skipped before fetch because it is non-contrast. Every fetched case was accepted by eye.
- 159 and 069 have one side where Arterial labelled no VA or the wrong vessel. The tsba trace follows the foraminal
  vessel, and the rule-6 flag is review-only.

**Session limits (no computed work lost):**
- 04:57: the first instance was cut. Its chains survived, and instance 2 took over at 06:16.
- About 09:21: instance 2 was cut. Its grade chains kept running and graded 044, 174, 065 and 080 with no agent
  present. Instance 3 took over at 09:48.
- What was lost: about 5 minutes of agent bookkeeping (state lines and the eye checks for those cases), which instance 3
  redid from the files.

### Per-case results (pipeline v2 + step-2 fallback; QC rule numbers as in pilot_proc_*_qc2)

| case | R | L | pair |
|---|---|---|---|
| 062 | donor_review (V1 rescue used) | fail 3a: 4.2 mm V1 bridge at 104 HU | no |
| 060 | fail 5b: HU 0.57 x CCA | donor (whole union 0.25) | no |
| 067 | donor_review | donor_review, but the wall fails V1 + V4 (genus 3) | no |
| 160 | fail 7: V3 loop | donor_review | no |
| 155 | donor, wall fails V4 (genus 5) | donor, wall fails V4 (genus 2); the label is truncated at 132 of 250 mm | no |
| 137 | donor | fail: wrong trace, 93 mm bridged | no |
| 044 | fail 3c + 7: 17.7 mm V3-low bridge at 60 HU | donor_review (bridged 7.6 mm), wall valid, whole union 0.196 | no |
| 174 | donor_review (bridged 9.1 mm), wall valid, whole union 0.129 | fail 3c + 5a + 7 (label off the trace) | no |
| 065 | donor, wall fails V4 (genus 2 in C6 -> C1) | donor, wall valid, whole union 0.199 | no |
| 068 | donor, wall fails V4 (genus 1) | fail 3a: 122.6 mm V1 bridge at 76 HU (weak contrast, CCA 265 HU) | no |
| 080 | fail 5b + 7 + 8 | fail 2 + 5a + 5b + 7 + 8 (label off the trace) | no |
| 124 | donor; VMTK collapsed (0.5 mm) -> step-2 fallback wall, which fails V4 (genus 1 in C6 -> C1) | donor_review (rule 6, 0.90), wall valid, whole union 0.227 | no |
| 136 | donor_review (7.3 mm bridge above the splice), wall valid, whole union 0.133; kept by eye | donor_review (rule 6, 0.85; provisional 3a + 5a), wall valid, whole union 0.120, label from s 6.75 mm; kept by eye | **yes** |
| 138 | fail 7: V3 loop (C2 -> C1 only 4 mm on both traces) | donor_review after the V1 rescue (0.6 mm gap, 356 HU), wall valid, whole union 0.095 | no |
| 139 | donor_review (rule 6), wall fails V4 (genus 3; whole union 0.361) | fail 3c + 5b: 16.5 mm V3-low bridge at 319 HU | no |
| 149 | donor_review (rule 6), wall valid, whole union 0.224 (V2 bins 0.36-0.58); kept by eye | donor_review (rule 6, 0.00: Arterial's label is on a lateral parallel vessel; the trace runs 6 foramina; provisional 5a), wall valid, whole union 0.085; kept by eye | **yes** |
| 150 | fail 3a + 3b + 3d + 5b + 7 (label off the trace; 36.7 mm V2 bridge at 97 HU) | fail 7 (bridges at 4 HU in V2 and 97 HU in V3-low) | no |
| 158 | donor_review (bridged 7.4 mm), wall valid, whole union 0.112 | donor_review, wall fails V4 (genus 1) | no |
| 159 (19) | donor_review (4.4 mm bridge above the splice), wall valid, whole union 0.164; kept by eye | donor_review (same bridge), wall valid, whole union 0.123; kept by eye (Arterial labelled no left VA; the trace is the foraminal vessel from the LSA root) | **yes** |
| 164 (20) | donor, wall valid, whole union 0.077 | donor (provisional 3a in the tally: 1.5 mm origin gap at 254 HU), wall valid, whole union 0.100; kept by eye | **yes** |
| 165 (21) | donor_review (6.4 mm bridge above the splice), wall valid, whole union 0.141; kept by eye | donor_review (rule 6, 0.00; provisional 5a), wall valid, whole union 0.111; kept by eye | **yes** |
| 172 (22) | donor_review (5.3 mm bridge above the splice), wall valid, whole union 0.166; kept by eye | donor_review (rule 6, 0.02; 1.9 mm V1 origin gap = provisional 3a), wall valid, whole union 0.116; kept by eye | **yes** |
| 192 (23) | donor_review (8.1 mm bridge above the splice), wall valid, whole union 0.066; kept by eye | donor_review (rule 6, 0.86; same bridge), wall valid, whole union 0.143; kept by eye | **yes** |
| 069 (24) | donor, wall valid, whole union 0.200 | donor_review (rule 6, 0.00: Arterial's label is on the medial parallel vessel), wall valid, whole union 0.114; kept by eye | **yes** |
| 071 (25) | donor, wall valid, whole union 0.101 | donor_review (2.8 mm V1 origin gap at 60 HU = provisional 3a), wall valid, whole union 0.255; kept FLAGGED by eye | **yes** |

Instance 2 noted, after the first 6 graded cases, that "the scale-up walls are much rougher than the pilot's" (whole
union 0.21-0.39 against a pilot median of 0.16). That does not hold for the lane's valid pair walls: their whole union
is 0.066-0.255, median about 0.12. The rough walls were mostly the genus-failing ones (155, 139, 067).

## Case list (selection order, `results/pilot_next_cases.json` -> selection, source ia172)

| order | case | pre-fetch arch check | fetch + by eye |
|---|---|---|---|
| 2 | 062 | auto FAIL -> PASS by eye (two limbs on the lowest slice; contrast streaks from the SVC broke the partner test; near-apex cut) | accepted |
| 4 | 119 | **FAIL: non-contrast** (no enhanced aorta at 0 to +120 mm; `119_probe.png`) -> skipped | – |
| 6-36 | 060 067 160 155 137 044 174 065 068 080 124 136 138 139 149 150 158 | PASS | all accepted |
| 38-52 | 159 164 165 172 192 069 071 (extension, cases 19-25) | PASS | all accepted (09:55-11:20) |

By-eye fetch verdicts are in `results/scaleup_ia172_visual.json`:
- 062: strong, CCA 800 HU.
- 060: moderate, CCA 330 HU; head turned.
- 067: weak, CCA 240 HU.
- 160: CCA 320 HU.
- 155: CCA 290 HU.
- 137: CCA 387 HU.
- 044: CCA 322 HU.
- 174: strong, CCA 477 HU; near-apex arch cut, so Arterial was cropped from k = 53.
- 065: CCA 332 HU.
- 068: weak, CCA 263 HU.
- 080: CCA 407 HU.
- 124: CCA 372 HU; head turned.
- 136: CCA 400 HU; tortuous, off-centre.
- 138: strong, CCA 564 HU.
- 139: CCA 363 HU.
- 149: strong, CCA 488 HU; inclined spine.
- 150: strong, CCA 511 HU.
- 158: CCA 400 HU; thin VAs.
- 159: CCA 480 HU; possible arch-origin left VA. The mask places it at the LSA root.
- 164: CCA 469 HU.
- 165: CCA 495 HU; tortuous great vessels.
- 172: CCA 348 HU.
- 192: CCA 522 HU; 0.40 mm slice spacing.
- 069: CCA 402 HU; Aquilion PRIME; patient rotated.
- 071: CCA 371 HU; Aquilion PRIME; noisier.

## Method

1. **Pre-fetch arch check** (`scaleup_ia172_prearch.py`, new; the pilot's proposal "range-read the lowest slice, require
   two aortic components above 150 HU"). Three DICOM members per series (lowest slice, +5, +10 mm; ~200 KB each) by
   range read. Mask-free (no TotalSegmentator before the fetch): HU > 150 inside the body, 1.5 mm opening, components
   >= 150 mm^2; descending limb = round (fill >= 0.55), 250-1500 mm^2, homogeneous (min(IQR, std) <= 90 HU), median
   180-800 HU; anterior partner = >= 400 mm^2, (spread <= 90 or fill >= 0.6), median 180-800, >= 25 mm anterior and
   >= 30 mm away. PASS = two limbs on any of the three slices (limbs at +10 mm imply limbs at the bottom; an apex cut
   shows one elongated component and nothing above). A FAIL is probed up to +80 mm and every FAIL PNG read by eye.
   Calibration: the 12 pilot cases on their stored DICOM agree 12/12 with the pilot's post-fetch archcut (090, 118 fail).
   Two earlier versions (a std-only rule, then IQR-only) mis-called 3-5 pilot cases (calcified walls, a limb touching
   the vertebra, a chest wall cut open by the FOV); the final rule is the one recorded in the JSON.
2. **Fetch** = the pilot's method unchanged: `pilot_ia172_fetch.py` (zip byte-span range read, CRC-32 per member;
   `pilot_ia172_convert.py`), quick-look TotalSegmentator `total --fast` (`scaleup_ia172_ts.sh` = pilot copy, 6 CPUs),
   `scaleup_ia172_qc.py` (= `pilot_ia172_qc.py`; phase, subclavian origins, skull base, MIP, meta.json),
   `scaleup_ia172_archcut.py` (= `pilot_ia172_archcut.py`, post-fetch two-limb confirmation), then my visual check of
   `mip_neck.png` (both VAs enhanced at C4 / C6, arterial phase, origins and skull base in view) ->
   `results/scaleup_ia172_visual.json` (`scaleup_ia172_visual.sh`).
3. **Processing** = the pilot driver (`scaleup_ia172_run.sh` = `pilot_proc_ia172_run2.sh` + the diffs D1-D10 listed in
   its header): TotalSegmentator `total_v3 --ml` on the GPU + CPU split + 1.5 mm copy; Arterial args / [-1024, 3071]
   clip / Arterial native (GPU, `--no-tta`, watchdog); trace `tsba` and `ts`; VMTK.
4. **Pipelining**: detached chains (`scaleup_ia172_chain.sh` fetch / ts / tscpu / gpu / post / grade) with per-case
   done-markers; GPU lock shared with the Song lane; one heavy CPU container at a time for this lane (6 CPUs / 12g),
   GPU-feeding stages first.
   - **D8 / D9 (instance 2, 06:30 / 06:35).** By 06:20 the GPU was 7 cases ahead of post. The heavy CPU slot was being
     held by work that is not CPU-heavy: the fetch download (network-bound, 2-25 min per case) and `make1p5`
     (single-threaded, ~1 core / 2 GB in `docker stats`). Both now run as light containers (2 CPUs / 8g, after a 6 GB
     RAM wait, no CPU lock). The command, image and outputs are unchanged.
   - **D10 (07:35).** Post-chain VMTK runs light under the lane's light-slot lock.
   - **08:51.** A second grade chain was added. Both chains share the case list through per-case claim dirs (`grade_claim_<case>`).
     The authoritative markers are `grade_done_<case>` + `fallback_done_<case>`. A chain that skips a claimed case still
     writes `chain_grade_done_<case>`.
   - **Instance 3 (09:51 / 10:58).** The extension cases 19-22 and then 23-25 went through fetch / TS / Arterial / post
     ahead of the tallies, to keep the GPU busy. Each group was graded only after the tally that allowed it (11:29 for
     19-22, 12:28 for 23-25).
5. **Provenance**: `scaleup_ia172_provenance.py` regenerates a scale-up section of `thin_slice/ia172/PROVENANCE.md`
   between markers, re-run after every fetched case. It now lists 25 cases plus 119 as skipped. The pre-scale-up file is
   backed up as `scaleup_work/PROVENANCE_ia172_before_scaleup.md`.
6. **Grading** = pipeline v2 only (`notes/SCALEUP_PIPELINE.md`):
   - `scaleup_ia172_grade.sh <case>` runs step 1, `scaleup_v2_post.sh ia172 post2 <case>`, through the per-call
     resource patch `scaleup_ia172_v2patch.sh`. Step 1 is: QC on both traces -> select -> V1 rescue -> gross -> ct_seg4
     with the pinch fix -> measure -> protrusion -> figures -> collect into `results/scaleup_ia172.json` -> grade.
   - It then runs step 2, `scaleup_ctseg_fallback.sh` (the VMTK-collapse fallback).
   - Every case is graded once, under v2. No case was graded under the pilot rules.
7. **By-eye checks of flagged sides**:
   - What is looked at: `qc_<case>.png`, `<case>_detail_along_va.png`, `wall3d`, the bridge positions in `qc.json` and,
     for the 071 L origin gap, the new read-only `scaleup_ia172_origin_look.py`.
   - Which sides: only sides of pairs that are otherwise eligible. A verdict on a pair whose other side fails cannot
     change the tally.
   - Where they go: `results/scaleup_eye_rejections.json`, through `scaleup_ia172_eye_add.py`. That is a copy of
     `scaleup_song_eye_add.py` with the tmp suffix and print filter changed, using the same `.eye_lock`.

## Files

- Scripts (new, `/sp/v5_va/`):
  - `scaleup_ia172_{prearch,prearch_look,qc,archcut,meta_visual,visual_set,provenance,peek,eye_add,origin_look}.py`
  - `scaleup_ia172_{run,chain,ts,tsrun,grade,v2patch,status,visual,wait}.sh`
- Results:
  - `results/scaleup_ia172.json`: 50 side records.
  - `results/scaleup_ia172_visual.json`: by-eye fetch verdicts.
  - `results/scaleup_ia172_prearch.json` + `results/scaleup_ia172_prearch/`.
  - `results/scaleup_ia172_eye/eye_<case>.json`.
  - `results/scaleup_tally_ia172_{batch1_1130,after22_1232,final25_1303}.json`.
  - Shared: `results/scaleup_eye_rejections.json` and `results/scaleup_tally.json`.
- Data:
  - `thin_slice/ia172/<case>/` (`ct.nii.gz`, `dicom`, `meta.json`, `mip_neck.png`) and `PROVENANCE.md`.
  - `derived/thin_slice/ia172/<case>/` (totalseg, arterial, qc, ct_seg).
  - Logs in `_logs_scaleup/`.
