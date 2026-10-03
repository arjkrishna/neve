# pilot_fetch_song: 10 Song et al. 2024 CTAs for the thin-slice VA pilot

Agent prefix `pilot_fetch_song`. The work ran in three instances:
- **Instance 1** (to about 18:20 EDT) selected, fetched and ran QC pass 1. A session limit cut it off.
- **Instance 2** (20:03-20:13) wrote two scripts and was stopped.
- **Instance 3** (20:17-21:00) finished the QC, gave each case a verdict and wrote the outputs.

State file: `v5_va/state/pilot_fetch_song.md`.

## Answer so far (final, 21:00 EDT)

**All 10 cases pass. No case was rejected and none needed replacing.**
- **Phase:** every case is arterial.
- **VAs:** both VAs are visible in the C6-C1 transverse foramina on every case.
- **Coverage:** every image runs from below the aortic-arch top to the vertex.
- **Artefacts:** no slice step or motion band.
- **Integrity:** RAR5 CRC32 matches on all ten, and the side files check out voxel by voxel.
- **Mix:**
  - 5 aneurysm patients (008, 032, 044, 090, 094) and 5 controls (100, 106, 109, 113, 141).
  - Slices: 0.45 mm ×4 (Philips), 0.625 mm ×3 (GE), 0.7 mm ×3 (Siemens).
  - 5 F / 5 M, ages 27-78, z 328-390 mm, 806 MB on D:.
- **Clipping:** every case is clipped as reported. All voxels at or below 50 HU are stored as 0, and the smallest
  non-zero value is 51. The sign survives: -0.0 marks an originally negative voxel.
  - The images are kept as shipped.
  - Each case has two int16 re-windowed side files, `ct_bg-100` and `ct_negzero-100`, in
    `carotid_data/va_sources/derived/thin_slice/song/<case>/`.
  - Arterial on the shipped image is still untested; that is the process step's job.
- **Flags for the process step** (none is a reason to reject):
  - Dense injection-side venous reflux next to the left V1-V2 in 008 and 094, and in the left IJV in 106.
  - 008 has low-density vertebrae.
  - 090 has its head rotated at C1-C2.
  - 113 is a dual-energy 90 kV image.

## Cases

HU are medians. The CCA value is from arch+60 mm, or arch+100 mm where arch+60 has no clean blob. The coverage
column gives mm below / above the aortic-arch top. "Both" in the VA column means both VAs were seen in the
foramina.

| case | group | sex/age | slice mm | vendor | z mm | aorta | CCA R / L | coverage | VAs | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AHMU1218008 | aneurysm | M 48 | 0.625 | GE | 350.6 | 342 | 320 (418 at +100) / 422 | 29 / 321 | both | accept | dense left IJV and vertebral venous reflux by left V1-V2; vertebrae 300-700 HU |
| AHMU1218032 | aneurysm | F 65 | 0.625 | GE | 375.0 | 334 | 362 / 381 | 73 / 301 | both (R small) | accept | right-arm injection |
| AHMU1218044 | aneurysm | M 76 | 0.45 | Philips iCT 256 | 331.7 | 442 | 484 / 521 | 31 / 301 | both | accept | short margin below the arch |
| AHMU1218090 | aneurysm | F 78 | 0.45 | Philips | 328.5 | 547 | 618 / 622 | 40 / 288 | both | accept | head rotated at C1-C2; neck off-centre |
| AHMU1218094 | aneurysm | F 57 | 0.7 | Siemens | 385.0 | 529 | 599 / 603 | 70 / 314 | both | accept | **dense left venous reflux** at arch+54..+98 beside left V1-V2; FOV 206 mm |
| AHMU1218100 | control | M 78 | 0.45 | Philips | 352.8 | 559 | 625 / 658 | 45 / 307 | both | accept | census sample |
| AHMU1218106 | control | M 51 | 0.625 | GE | 387.5 | 355 | 382 / 312 (377 at +100) | 36 / 351 | both | accept | left IJV reflux; lowest contrast; dental streaks |
| AHMU1218109 | control | F 31 | 0.7 | Siemens | 385.0 | 448 | 542 / 550 | 70 / 314 | both | accept | none |
| AHMU1218113 | control | M 66 | 0.7 | Siemens DE | 387.1 | 406 | 436 / 435 | 69 / 317 | both | accept | 90 kV dual-energy image; arch top set by eye |
| AHMU1218141 | control | F 27 | 0.45 | Philips | 389.7 | 523 | n/a* / 615 | 73 / 316 | both | accept | small vessels |

\* On 141 the right CCA merges with the IJV in the blob mask (blob median 252 HU, p90 619 HU).

**Side files.** `ct_bg-100` sets -100 HU on 58-75% of voxels: every 0 outside the body mask. `ct_negzero-100` sets
-100 HU on 68-81% of voxels: exactly the -0.0 voxels. Both equal the shipped image everywhere else, on the same grid
and affine (`qc_va_check.json` side_check, all ok).

**Subclavian-level HU.** The arch+15/+35 picks (292-617 HU) are in `meta.json` as an arterial reference only. At
those levels a blob's position alone cannot separate the subclavian from the brachiocephalic, CCA or VA origin.

## Selection

Instance 1 chose the cases but was cut off before writing down why. The rule below is reconstructed from
`results/pilot_fetch_song_table.json`:
- **Eligible:** a 3-D header and z ≥ 300 mm. That gives 132 of 143: 41 of 44 controls and 91 of 99 patients.
  `other_ctas.md` said 134.
- **Excluded:** AHMU1218077 and 097, whose 4-D headers are broken.
- **Picks per stratum:**
  - patients: 0.45 mm 2 of 4; 0.625 mm 2 of 85; 0.7 mm 1 of 2
  - controls: 0.45 mm 2 of 34 (one is 100, the census sample); 0.625 mm 1 of 2; 0.7 mm 2 of 5
- **Not recorded:** the tie-break among the 85 GE patients at 0.625 mm.

The pool is skewed. Patients are almost all GE 0.625 mm, and controls are almost all Philips 0.45 mm. The 0.7 mm
(Siemens) and the 0.625 mm controls are scarce: 7 and 2 cases.

## How each case was checked (instance 3)

1. **Inventory.** All files listed in the resume brief exist. `PROVENANCE.md` still had its case-table
   placeholder and pointed to a `meta.json` that did not exist. Instance 2's `pilot_fetch_song_va_check.py` had
   been written but never run.
2. **Overview MIPs** (`mip_neck_*.png`), viewed for all 10.
   - All are arterial: CCAs and ICAs bright, IJVs unopacified apart from injection-side reflux.
   - The sagittal slabs were off-centre on 008, 090, 106 and 109. On 008 qc_pass1's `spine_mid_x` is 29 mm off,
     because it is a bone median pulled by the shoulder.
3. **VA check in the column frame** (`pilot_fetch_song_va_check2.py` → `qc_va_check.png/.json`).
   - Instance 2's canal tracker found no canal on 008. That patient's vertebrae read 300-700 HU, so the ≥ 700 HU
     ring never closes (`results/pilot_fetch_song_dbg/AHMU1218008_canal.png`).
   - Version 2 tracks the vertebral column by mean shift instead: voxels ≥ 200 HU within 22 mm. Above arch+150 it
     is clamped, because at the skull base the track drifts.
   - The same run checks the side files against the shipped image voxel by voxel.
4. **Zoomed axial crops** (`pilot_fetch_song_va_zoom.py` → `qc_va_zoom.png`): 10 levels from arch+25 to
   arch+185, window 0..1.6 × the aorta. All viewed. Both VAs appear as round lumens in the foramina on every case.
5. **Slice step or motion.** The suspect bands were checked on single slices (`pilot_fetch_song_motion_look.py`):
   - 109's dark band at z -750 is not a step.
   - 113's edge at z 550 is the skull base.
6. **CCA and subclavian HU** (`pilot_fetch_song_picks.py` → `results/pilot_fetch_song_picks.json`).
   - These are column-frame rules on the qc_pass1 blobs. All 10 `qc_axial.png` were viewed.
   - qc_pass1's own picks were wrong or missing on several sides. On 032 its left CCA was a 14 mm² blob at 254 HU;
     the real one is 381 HU.
   - One override: on 090 the left CCA is blob 8, which the rule skipped because it touches the bone mask.
7. **Outputs** (`pilot_fetch_song_meta2.py`): instance 1's `pilot_fetch_song_meta.py` extended. It writes 10
   `meta.json` files and `results/pilot_fetch_song.json`. The by-eye review is in
   `results/pilot_fetch_song_review.json`.

Nothing was re-downloaded or re-run from instance 1's work: the fetches, QC pass 1, MIPs and side files were all
intact. Only the never-run VA check and the missing outputs were produced.

## For the process step

- **Run order:** run Arterial on the shipped `ct.nii.gz` first. Use the side files only if it fails.
  `ct_negzero-100` restores the fat/air distinction; `ct_bg-100` only fixes the background.
- **Left side on 008 and 094:**
  - Expect dense veins next to the left V1-V2.
  - Cap the vessel HU at about 1.6 × the aorta median, or check the left trace by eye.
- **Bone thresholds:** do not trust a 700 HU bone cut on 008.
- **Phase gate:** use the CCA values above (0.88-1.23 × the aorta) or TotalSegmentator CCA masks, not the
  subclavian-level picks.
- **Grading:** grade on the acquisition voxel. That is 0.39-0.57 mm in-plane with 0.45 / 0.625 / 0.7 mm slices,
  never Arterial's resampled grid.
- **Licence:** Song is CC BY 4.0. Anything derived through Arterial is non-commercial.

## Files

- Data: `D:\Arjun\neve\carotid_data\va_sources\thin_slice\song\<case>\`
  - fetched: `ct.nii.gz`, `fetch.json`
  - written by instance 3: `meta.json`, `qc_va_check.png/.json`, `qc_va_zoom.png`
  - from instance 1: `qc_pass1.json`, `mip_neck_coronal/sagittal.png`, `qc_axial.png`, `qc_va_axial.png`
    (superseded)
  - `PROVENANCE.md` (case table filled) and `_source_meta/` (sidecars, `clinical_all.csv`)
- Side files: `...\derived\thin_slice\song\<case>\{body_mask,ct_bg-100,ct_negzero-100}.nii.gz`
- Results:
  - `v5_va/results/pilot_fetch_song.json` (summary)
  - `_review.json`, `_picks.json`, `_va_check.log`
  - `_index.json`, `_table.json`, `_get.log`, `_qc.log`
  - `pilot_fetch_song_dbg/`
- Scripts:
  - written by instance 3: `v5_va/pilot_fetch_song_va_check2.py`, `_va_check_all.sh`, `_va_zoom.py`, `_picks.py`,
    `_blobtab.py`, `_meta2.py`, `_motion_look.py`, `_canal_debug*.py`
  - instance 2's `pilot_fetch_song_va_check.py`: instance 3 edited its seed block; `va_check2` imports its
    `side_check()` unchanged
  - earlier: instance 1's scripts and instance 2's `pilot_fetch_song_peek.py`
