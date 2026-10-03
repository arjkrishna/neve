# pilot_fetch_ia172: 10 thin-slice CTAs from the Head-CTA 172-series set

Date 2026-10-01. Task: select 10 series from 10 distinct patients of Zenodo 15697196 (CC BY 4.0) and make them ready
for the v5 donor pilot: Head+Neck, z ≥ 300 mm, thinnest reconstructions. Convert DICOM to NIfTI (RAS, raw HU), write
meta.json and a neck MIP for each, and reject and replace any case that is venous, non-contrast, truncated or
motion-corrupted.

## Answer

1. **Ten cases are ready**, all arterial phase, with both VAs enhanced, the skull base in view and the subclavian
   origins in view.
   - Path: `D:\Arjun\neve\carotid_data\va_sources\thin_slice\ia172\ia172_<CaseID>\`, holding `ct.nii.gz`,
     `meta.json`, `mip_neck.png` and `dicom/`.
   - Cases: **001, 090, 118, 061, 141, 039, 041, 140, 073, 058**.
   - 5 women and 5 men, aged 26–87 (5 under 60, 3 aged 60–69, 2 aged 70 or over). 8 controls and 2 with anterior-
     circulation aneurysms. 5 on the Aquilion PRIME and 5 on the Aquilion.
2. **One case was rejected and replaced.** Case **166** is truncated at the bottom: only the top 6 mm of the arch is
   in view, the brachiocephalic-trunk (BCT) origin lies below the field of view, and the left subclavian (LSA) origin
   sits 4.5 mm above the bottom slice. It was replaced by **041** (M/66, z 384 mm).
3. **Two of the ten cut only the arch apex: 090 and 118.**
   - The LSA origin is 13.5 and 10.5 mm above the bottom slice. The bottom slice holds **one** aortic component, the
     apex, not the ascending and descending limbs. I checked this on the native CT (`results/pilot_ia172_archcut.json`).
   - The subclavian origins are in view, so the stated rules pass them.
   - Arterial's start point needs two limbs. It rejected TotalSegmentator head CTAs s1163 and s0152 for this reason
     (donor_pool_plan §1.2), so **Arterial may refuse these two.**
   - **Two alternates with a full arch were fetched and checked for that case: 190 (M/40) and 015 (M/31).** Both are
     young men like 090 and 118.
4. **Resolution.**
   - Every series is a 0.5 mm-thick reconstruction. The acquisition voxel is 0.388–0.507 mm in-plane × 0.5 mm
     (alternate 015: 0.616 mm in-plane).
   - The grid's z spacing is 0.25 mm (090), 0.3 mm (8 cases) or 0.5 mm (001); alternate 190 is at 0.4 mm. These are
     overlapping reconstructions.
   - **Grade with the acquisition voxel** (in-plane × 0.5 mm thickness), never the 0.25–0.3 mm interval
     (GRADING CONVENTIONS).
5. **What comes next:**
   - TotalSegmentator `total_v3` on the CTA grid (`tag_totalseg.md`);
   - Arterial;
   - the trace and the wall work.
   - None of these ran here. The masks in `derived/` are a 3 mm quick look only.

## 1. Selection

**Pool.** From `results/pilot_ia172_headers.json` (first, middle and last DICOM header of every series, read
remotely) and `Metadata.csv`:

| filter | series |
|---|---|
| in the zip | 172 |
| Head+Neck = 1 (Metadata.csv) | 145 |
| and z ≥ 300 mm (IPP first to last) | 120 |
| and SliceThickness ≤ 0.5 mm | 90, all Toshiba: 62 Aquilion FC43, 26 Aquilion PRIME FC43, 2 Aquilion FC04 |
| of those, overlapping reconstruction (interval < 0.45 mm) | 87 |

All 172 series are axial: IOP [1,0,0,0,1,0], gantry tilt 0, head-first supine, JPEG-lossless.

**Distinct patients.**
- PatientID, PatientName, sex, age and dates are blank in every DICOM, so the CaseID is the only identifier.
- The screen in `pilot_ia172_select.py` → `results/pilot_ia172_candidates.json`:
  - no StudyInstanceUID, FrameOfReferenceUID or SeriesInstanceUID is shared between series;
  - no two series share an identical acquisition (model, software, table height, first IPP, file count, kVp, mA);
  - 11 pairs have the same sex, age within 1 year, the same model and table height within 5 mm: 003/148, 005/042,
    023/174, 062/110, 065/098, 097/128, 114/168, 125/147, 154/159, 159/176, 164/173. **None of these series was
    picked**;
  - StackID collides in 4 groups (015/137, 064/097, 078/118/173, 083/174), but every group differs in sex or age. It
    is a Toshiba stack counter, not an identity.
- The 12 fetched cases all differ in sex or age, and in every acquisition key.

**How the ten were chosen.**
- **Thickness.** Prefer 0.5 mm thickness; prefer the finer in-plane pixels (001 0.398, 090 0.388, 061 0.439,
  118 0.456; the rest 0.506–0.507 mm).
- **Balance:** 5 women and 5 men; ages spread across the < 60, 60–69 and ≥ 70 bands; both scanner families,
  which are probably different centres (PRIME V4.73/4.74 against Aquilion V4.62/4.86).
- **Controls first,** plus 2 young men with anterior-circulation aneurysms (090, 118). Young subjects are scarce
  among the controls.
- **Avoided:**
  - posterior-circulation aneurysms (basilar or vertebral, for example 049, 100, 038, 093), which could distort the
    V4 or confluence;
  - series of 1 mm or thicker;
  - non-Toshiba series, which are all 0.6 mm thick or more.
- **001 was reused** from the earlier va_pool sample: 0.5 mm, and viewed before. **003** (1 mm) was not used.

**Reserves**, all 0.5 mm, z ≥ 340, controls, not yet fetched: 062 (F53, 0.47 mm), 155 (M42), 160 (M42), 067 (M49,
369 mm), 060 (M52, 372 mm), 119 (M60, 399 mm), 052 (F78, 372 mm), 131 (F93).

**Lesson for bulk selection.** z ≥ 300–350 mm does not guarantee the arch.
- 166 (351 mm) started above the arch, and 090 and 118 (351–352 mm) start at its apex.
- The scan ranges are planned for the skull and differ by patient.
- **Proposal:** before fetching a whole series, range-read its lowest-z slice (one member, about 200 KB) and require
  two aortic components above 150 HU. That costs one request per series.

## 2. Fetch

- **Script:** `pilot_ia172_zip.py` extends `va_pool/rzip.py`.
  - It reads the central directory once and caches it as JSON on D: (149,452 members).
  - Each series' members sit contiguously in the zip (`other_members_in_span` = 0 for every case). One byte span per
    series is downloaded in 8 MB chunks over 4 threads, then each member is parsed from the local header and
    CRC-32-checked.
- **Speed:** 177–246 MB per series in 31–56 s (3.8–7.0 MB/s). A single stream manages 1.1 MB/s.
- **Total:** 13 series fetched (12 kept, 1 rejected), about 2.9 GB of DICOM. Case 001 was copied from
  `va_pool/data/ia172/001` (0 CRC mismatches).
- **Disk on D:**
  - per case, DICOM 119–267 MB and `ct.nii.gz` 154–344 MB;
  - the 12 kept cases: 2.6 GB of DICOM and 3.3 GB of NIfTI;
  - in total 5.6 GB under `thin_slice/ia172/`;
  - derived data: 256 MB, of which about 160 MB is the fast-model weights in `_ts_home`.

## 3. Conversion (`pilot_ia172_convert.py`, 33–100 s per case)

The checks below are recorded in `derived/.../convert.json` and in `meta.json → geometry_checks`. All pass for all 12
cases.

| check | result |
|---|---|
| single SeriesInstanceUID, no localizers | yes |
| Rows/Columns, PixelSpacing, IOP and SliceThickness each have one value | yes |
| slice positions sorted along the normal | spacing uniform within 0.0001 mm; 0 gaps; 0 duplicates |
| InstanceNumber monotonic with position | yes |
| in-plane IPP drift along the stack (tilt / shear) | 0.000 mm |
| oblique | no, so **no resampling**: voxels copied 1:1 with axis flips to RAS |
| HU | stored × RescaleSlope + RescaleIntercept, per slice. Slope 1; intercept 0 (001, 090, 118, 190, 015) or −32768 (unsigned storage: 061, 141, 039, 041, 140, 073, 058), handled. Stored as int16 |

The output is `ct.nii.gz`, int16, RAS, qform and sform code 1, axcodes RAS.

**Padding and extreme values.**
- **Outside the reconstruction circle the value is −2048 HU** in every case. That is 21–31% of all voxels, so HU p1
  of the whole volume is −2048.
- `meta.json` therefore also gives percentiles over voxels above −500 HU: p1 −441 to −314, p50 32–68, p99.9 1671–2537.
- Some slices 14–45 mm below the C1 top hold values down to −5530 HU. I looked at 3 cases: dental-restoration streaks
  in the oral cavity. They stay anterior and do not reach the VA (`results/pilot_ia172_artefacts.png`).
- The data are **defaced:** voxels in front of the face and mandible are blanked.
- **Advice:** clip to [−1024, 3071] before any tool that normalises by the intensity range. Local-HU methods (edge
  FWHM) are unaffected.

## 4. Quick look and coverage (`pilot_ia172_ts.sh`, `pilot_ia172_qc.py`)

**Method.**
- TotalSegmentator 2.18.0 `total --fast` (3 mm model, CPU, about 25 s) on a 1.5 mm RAS copy. It used a private
  weights home on D: with `send_usage_stats=false`, so the shared cache was not touched.
- Vessel HU is the median of the native CT sampled at the 1-voxel-eroded masks.
- **Injection side** = the brachiocephalic vein (BCV) with the higher HU.
- **Phase** = arterial if the CCA is ≥ 250 HU and the contralateral BCV is < 200 HU. That BCV threshold is the
  polytrauma guard of donor_pool_plan Stage 2.

| case | CCA R/L | SCA R/L | aorta | BCV R/L (inj.) | phase | LSA / RSA origin above bottom slice (mm) | aortic comps at bottom | image top above C1 top (mm) |
|---|---|---|---|---|---|---|---|---|
| 001 | 493 / 511 | 452 / 476 | 451 | 231 / 157 (R) | arterial | 40 / 62 | 2 | 150 |
| 090 | 416 / 439 | 384 / 432 | 366 | 293 / 78 (R) | arterial | **14** / 40 | **1** | 142 |
| 118 | 421 / 426 | 400 / 431 | 375 | 1256 / 85 (R) | arterial | **10** / 34 | **1** | 162 |
| 061 | 411 / 399 | 344 / 336 | 317 | 159 / 261 (L) | arterial | 63 / 81 | 2 | 132 |
| 141 | 446 / 444 | 394 / 421 | 365 | 177 / 415 (L) | arterial | 46 / 68 | 2 | 144 |
| 039 | 359 / 361 | 334 / 334 | 323 | 281 / 87 (R) | arterial (weakest) | 52 / 62 | 2 | 123 |
| 041 | 401 / 398 | 312 / 340 | 367 | 76 / 219 (L) | arterial | 51 / 78 | 2 | 159 |
| 140 | 503 / 473 | 469 / 449 | 435 | 188 / 517 (L) | arterial | 86 / 87 | 2 | 114 |
| 073 | 461 / 438 | 422 / 405 | 377 | 102 / 1380 (L) | arterial | 106 / 104 | 2 | 111 |
| 058 | 486 / 470 | 431 / 414 | 349 | 165 / 120 (R) | arterial | 40 / 50 | 2 | 135 |
| 190 (alt.) | 413 / 420 | 382 / 365 | 352 | 292 / 139 (R) | arterial | 39 / 60 | 2 | 154 |
| 015 (alt.) | 393 / 408 | 326 / 348 | 334 | 218 / 126 (R) | arterial | 75 / 84 | 2 | 147 |
| *166 (rejected)* | 308 / 332 | 285 / 315 | 311 | 194 / 52 (R) | arterial (lowest) | **4.5** / 16.5; BCT origin below the FOV | – | – |

**Coverage.** In all 12, C1–C7 and T1 are present and the skull base and foramen magnum are in view. The vertex is
cut in some cases (140, 039, 058), which does not matter for the neck.

**Motion.**
- Proxy: the largest slice-to-slice jump of the body centroid. It reads 0.6–4.8 mm (p99 0.38–1.25 mm) and is
  dominated by the shoulders and arms entering the field of view, so it does not decide.
- The neighbour-slice HU-difference check (`results/pilot_ia172_slicecheck.json`) finds no step inside the neck
  stretch, apart from 3 slices at the skull base in 061, at the edge of the defacing. The other spikes lie in the head.
- I saw no motion steps on any MIP or axial panel.

## 5. Visual check (`mip_neck.png`, read by me; `results/pilot_ia172_visual.json`)

**Every case is arterial, with dark veins apart from the injected arm.**

**Both VAs are enhanced in the C4 and C6 transverse foramina in every case.**
- The coronal VA-corridor MIP shows the whole course from the subclavian to C1 in 058, 118, 141, 061 and 140.
- In the elderly (001, 039, 041, 073), trabecular bone and tortuous CCAs hide the VA on the coronal MIPs, so the axial
  panels decide there.

**Per-case notes:**
- **073:** marked forward head posture / cervical kyphosis. The VA course reflects it.
- **141:** the left VA reaches the LSA close to its root. Check for an arch origin when tracing.
- **039:** the weakest enhancement (CCA about 360 HU).
- **001:** tortuous CCAs; the left CCA overlies the left VA corridor on the coronal MIP.

**Bone suppression in the MIP is for display only.** Cortex above 600 HU, closed by about 1.5 mm, plus axial holes
over 120 mm² are removed. The transverse foramina are kept.

## 6. Rejections

| case | reason | replaced by |
|---|---|---|
| 166 (M/65, control) | truncated: aorta only in the bottom 6 mm (TS z −113..−107, bottom slice −113), BCT origin below the FOV, LSA origin 4.5 mm above the bottom slice; also the lowest contrast (CCA 320 HU). Phase and VAs otherwise fine | 041 |

- The record is kept: `thin_slice/ia172/_rejected/ia172_166/{meta.json, mip_neck.png}` and
  `derived/.../_rejected/ia172_166/`.
- The volume and DICOM were deleted.
- No case was venous, non-contrast or motion-corrupted.

## 7. For the processing step

- **Masks.** Run TotalSegmentator `total_v3` on the CTA grid for `--skull-mask` and `arterial_ts_args`, and use a
  1.5 mm copy for `art_extract --ts` (tag_totalseg.md §5).
  - These grids are 168–369 M voxels, most about 300 M; 090 is the largest at 369 M. TaG-Net patient A was 208 M voxels and already peaked at 17.7 GB
    in `load_ts` + `ts_qc` at full resolution.
- **Arch start.** 090 and 118 cut only the arch apex, so expect Arterial's arch start to fail there. Use 190 and 015,
  or the start-point patch (donor_pool_plan Stage 0, item 5).
- **Basilar seed.** `art_extract.find_ba` assumes the basilar within ±5 mm of the skull midline. Some heads are rotated or
  tilted (061 at the skull base), so use the cord midline or a mask-based seed (tag_arterial.md).
- **Grading.** Use the acquisition voxel (`meta.json → acquisition_voxel_mm`), not `grid_spacing_mm` and never
  Arterial's 0.5 mm grid.
- **Field of view.** The in-plane FOV is 199 mm (090) to 315 mm (015). Every case holds both subclavian origins and
  the trachea, so Arterial's 220 mm FOV crop is possible except where the FOV is already smaller (090 199 mm,
  001 204 mm).

## Files

**Scripts, new, in `v5_va/`:**
- `pilot_ia172_zip.py`: range-read helpers;
- `pilot_ia172_hdr.py`: headers of all 172 series;
- `pilot_ia172_select.py`: candidate table and duplicate screen;
- `pilot_ia172_fetch.py`: fetch, then convert;
- `pilot_ia172_convert.py`: DICOM to NIfTI with checks, plus the 1.5 mm copy;
- `pilot_ia172_ts.sh`: quick-look TotalSegmentator, CPU;
- `pilot_ia172_qc.py`: HU, coverage, motion proxy, MIP and meta.json;
- `pilot_ia172_meta_visual.py`: merges the visual verdicts;
- `pilot_ia172_archcut.py`: aortic limbs at the bottom slices;
- `pilot_ia172_slicecheck.py`: per-slice padding and steps;
- `pilot_ia172_artefact_look.py`: axial views of flagged slices;
- `pilot_ia172_summary.py`: builds the results JSON.

No study tool was changed or copied.

**Results, in `v5_va/results/`:**
- `pilot_ia172_fetch.json` (main);
- `pilot_ia172_headers.json`, `pilot_ia172_candidates.json`;
- `pilot_ia172_visual.json`, `pilot_ia172_rejections.json`;
- `pilot_ia172_archcut.json`, `pilot_ia172_slicecheck.json`;
- `pilot_ia172_zenodo_record.json`, `pilot_ia172_artefacts.png`;
- logs `pilot_ia172_*.log`, and `pilot_ia172_summary.txt`.

**Data:**
- `D:\Arjun\neve\carotid_data\va_sources\thin_slice\ia172\`, with `PROVENANCE.md`;
- `D:\Arjun\neve\carotid_data\va_sources\derived\thin_slice\ia172\`.
