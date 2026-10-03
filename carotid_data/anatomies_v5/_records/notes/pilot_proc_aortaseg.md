# pilot_proc_aortaseg: the donor pipeline on AortaSeg-60 Dissec_01

Agent prefix `pilot_proc_aortaseg`. Date 2026-10-01. Three instances worked on this:
- the first, to about 18:20 EDT, was cut off by the session limit. It ran TotalSegmentator, the Arterial arguments and
  Arterial;
- the second, 20:03–20:13, was stopped. It ran the traces, VMTK and the look figures;
- the third, 20:17–21:30, wrote this note. It ran QC, the CT wall, the measures, the controls and the figures.

Nothing finished was redone. One output was re-run because it was wrong: the first CT-wall segmentation started 16 mm
above the ostium. It is kept as a sensitivity.

No git, and no tracked file was modified. Data is in
`D:\Arjun\neve\carotid_data\va_sources\derived\thin_slice\aortaseg\Dissec_01\` (gitignored). The state file is
`state/pilot_proc_aortaseg.md`.

**Licence.** AortaSeg-60 is treated as CC BY 4.0. Everything derived through Arterial (traces, masks, cut-outs, and
the CT wall seeded from them) is non-commercial.

## Answer

| side | QC (recalibrated Stage 2 + V3 loop + C1) | v5 donor? | CT wall (ct_seg4, acquisition voxel) | gross |
|---|---|---|---|---|
| **Left** | **PASS (review).** The only flag is Arterial's label, which is off the trace (review-only). Entry C6; 6 of C6..C1 foramina; V3 loop 20.2 mm and 0.2 mm from the C1 arch; no bridge; HU 1.23 × CCA; origin junction 0.0 mm | **yes**: origin → C1 is clean, and the distal part is clean too | **Not gradeable under the standard.** Gate V1 fails: centreline offset 0.189 mm > 0.15. Raw union > 0.3 / > 0.5 / > 0.15 r is 0.387 / 0.255 / 0.370 for the whole VA and 0.460 / 0.322 / 0.437 for C6 → C1; 0 of 24 10 mm bins are plain tube. Edge FWHM 1.69 mm | length 244.5 mm, DM 1.63, take-off at 10 mm 71.9°, subclavian elevation 21.3°, C2-loop bend 2.58 mm, V1 / V2 / V3 / V4 32 / 96 / 80 / 36.5 mm |
| **Right** | **FAIL**: a 10.0 mm V1 bridge at the origin; V3 loop 4.9 mm and 15.6 mm from C1; C1 not threaded; no dura | **no** | not segmented | entry C6, V1 30 mm; the rest is not measurable |

**Pilot yield: 1 of 2 sides (50%), 0 pairs**, against the plan's Stage 3 go/no-go of at least 60% of sides. That is one
case, so it says almost nothing about the source. AortaSeg-60 holds no other usable case.

**What the case shows:**
1. **Course and gross shape work at 0.78 mm.**
   - The left VA traced from origin to basilar with no bridge and passes every gate.
   - Its gross shape lies inside the real-VA ranges (notes/tag_calibration.md §5).
   - A strongly left-dominant pair: V2 calibre L/R is 1.84.
2. **The right side failed in the pipeline, not in the anatomy.**
   - Arterial put the right V1–V2 in a separate mask piece. It labelled every class "L*", so the right carotids were
     never blocked.
   - The trace left the VA at C3 and went up the right ICA.
   - The right VA and its V3 loop are in the CT.
3. **The wall is the opposite of "smooth" here, but it is not a usable reading.** The surface is lumpy:
   - 4–10× VMR's and TaG-Net's section irregularity;
   - 2–3× their roughness at 0.75–6 mm.
   - The standard's validity gate rejects it.

   Controls:
   - **The voxel staircase is excluded.** A smooth tube with this VA's course and calibre, on the same grid and through
     the same surface and VMTK, reads 0.028.
   - **The irregularity is in the image.** Arterial's own mask, which reads 0.01–0.07 on VMR and TaG-Net, reads 0.18
     here. The image's own half-max edge deviates > 0.3 mm on 35% of rays (VMR 8–15%, TaG-Net 7–11%).
   - **The image is the worse one.** Its edge FWHM is 1.69 mm, against 1.30–1.44 for VMR and TaG-Net. It is a
     late / mixed phase with lumen-bright veins: the left V1 label takes in an adjacent vessel at s 15 and 25.

   Where the relief lies:
   - 31% of the > 0.3 mm area lies where a lumen-bright structure continues outside the edge.
   - 4% lies at bone.
   - 65% lies in "free" zones, as axial-band bulges in V1–V2.

   So this is coarse irregularity of uncertain anatomical truth, not a validated real wall. Treat Dissec_01 as a
   **course / gross-shape donor** (strategy B, origin → C1), not a wall donor.

## 1. Run record (Dissec_01; per-stage wall time in `_logs/timing.jsonl`)

| stage | result | time |
|---|---|---|
| TotalSegmentator `total` (GPU, lock) | 117 masks; `total_v3` came from the fetch agent | 158 s |
| 1.5 mm RAS copies (`totalseg_1p5`, `totalseg_v3_1p5`) | for `art_extract --ts` and the QC harness | 234 + 290 s |
| `arterial_ts_args` + two-limb check | `--crop-k 18:415 --fov-mm 220 --fov-center=14.7,25.5`; the bottom slice cuts the arch into two limbs | 16 s |
| Arterial native (GPU, lock, `--no-tta`, TS `total` skull mask) | rc 0 on the **first attempt**; the aortic-arch check passed, so no corrected crop was needed. `cta.nii.gz` deleted | 89 s (80 s Arterial) |
| traces: `ts` (art_extract unchanged), `tsba` (pilot_art_extract: basilar seed window z_FM + 40, a mask-plausibility down-weight) | both sides traced. `ts` R climbs into intracranial vessels; `ts` L overshoots 21 mm up the basilar. **`tsba` is primary** | 53 + 55 s |
| VMTK on the stage-A cut-outs | all 4 ok, but the L line starts 16 mm above the ostium (see §3) | 13 s |
| QC (`pilot_proc_aortaseg_qc.py`) | `qc/qc.json`, `qc/Dissec_01_qc_tsba.png` | 269 s |
| ct_seg4, left (`pilot_ct_seg.py`) | primary `ct_seg/ct_seg4/` (third run) | 26 + 11 s (VMTK) |
| landmarks / measure / edge | `landmarks.json`, `measure.json`, `edge.json` | 426 / 89 / 87 s |
| figures | `qc/Dissec_01_wall3d_L.png`, `qc/Dissec_01_detail_along_va.png` | 12 s |

The whole case took about 4 min of GPU time and about 26 min of CPU. The landmark harness is the slowest step (6
centrelines × 2 mask sets). The 1.5 mm copies are only needed for the harness.

**Arterial's labels are confused.** Every class is "L*": the left VA is labelled LICA, and the right V1–V2 "LVA". No
right subclavian or RCCA is labelled. 28 mask components, the largest 0.896.

## 2. QC per side (recalibrated gate of notes/tag_calibration.md; the draft gate beside it)

The gate is re-implemented in `pilot_proc_aortaseg_qc.py` (`gate_recal`):
- Erec entry: `tag_calibration.augment`, imported unchanged;
- V1 bridge ≤ 3 mm; each V2 bridge ≤ 16 mm; each C2 → C1 bridge ≤ 10 mm; total below the splice ≤ 25 mm;
- robust-cord V3 loop ≥ 12 mm and ≤ 6 mm from C1; C1 threaded;
- band without the IQR term; HU ≥ 0.6 × CCA;
- BCV and label rules are review-only.

"Passes as a v5 donor" means every cervical rule passes (origin → C1). The distal rules (dura reached, length,
bridges above C1) are graded separately.

**Left: PASS (review)** (`qc.json` → `extractions.tsba.sides.L`)
- **Origin.** Junction 0.0 mm from the TS subclavian. The ostium reads 2.23 mm: the draft as implemented reads the
  ostium and fails it by 0.23 mm; the calibration reads the junction.
- **Entry.** ts_qc calls C7 on a 1 mm C7 run; Erec calls C6 at s 28.
  - Landmarks on the trace: C2 111, C1 foramen exit 137, ts_qc C1 159, dura 182.
  - 6 of C6..C1 foramina.
- **V3 loop.** Robust 20.2 mm and 0.2 mm (study measure 20.4 / 0.2; on the "total" masks 19.4 / 0.0). It passes both
  the study gate (15 / 5) and the recalibrated one (12 / 6).
- **Bridges and band.** No bridge. V2 band |dx| and dy inside, including the IQR term.
- **Phase.**
  - HU origin → C1 is 1.23 × CCA (CCA 264 HU on the native CT).
  - BCV: right 348, left 153, so the minimum is below 200.
  - No bone > 700 HU on the path.
- **Distal: clean.** Dura reached; origin → dura 182 mm on the trace; merge 35 mm beyond; no bridge.
- **Draft gate as written: FAIL** on origin (ostium 2.23 mm) and label agreement (0.00). Both are rules the calibration
  moved, and the verifier confirms that "label → review" is backed by two patients.
- **Visual check.** `qc/Dissec_01_look_axial_tsba_L.png` and `qc/Dissec_01_xsec_L.png` show the VA in the C6..C2
  foramina, the C2 loop, the V3 loop on the C1 posterior arch, and V4 to the basilar.

**Right: FAIL**
- **Automatic rules.** It fails:
  - 3a: a 9.99 mm V1 bridge at the origin;
  - 7: V3 loop 4.9 mm and 15.6 mm from C1 (study measure 10.8 / 15.6);
  - 8: C1 not threaded.

  Also: landmarks incomplete (no C1, no dura), band out, another 8.5 mm bridge at C3/C2, and 12.5 mm above C1.
- **Visual check** (`qc/Dissec_01_rlook_origin.png`, `_rlook_origin_zoom.png`, `_rlook_c2c1.png`, `_rlook_v3.png`,
  `qc/Dissec_01_xsec_R.png`):
  - **Arterial mask.** The right V1–V2 is mask component 8: 2336 voxels, z ≈ 2 → 103.5, labelled "LVA" (cell 25). It
    is cut off from the subclavian and from the main tree.
  - **Origin.** The 10 mm origin bridge runs through a mid-grey (200–250 HU), vein-like structure beside the
    subclavian.
  - **Above C3.** At s ≈ 98 (C3, z 74) the trace jumps 8.5 mm into the **right ICA**, which lies directly above the
    top of the TS right CCA at (39, 69, 33). It climbs the ICA to the skull base and bridges 12.5 mm into the basilar.
  - **The real right V3 loop is in the CT and in the mask.** Its V2 lumen is about 2.6 mm across, in dark fat.
  - **Recoverable?** Only with carotid blocking that does not depend on Arterial's labels. Even then the origin gap
    would fail 3a.

**Sensitivity to the verifier** (notes/verify_tag_calibration.md, 20:50). The verifier finds the specific values
unjustified: excursion 12, d 6, V1 ≤ 3 mm, dropping the IQR term, and the native × 1.10 factor. It gives no
replacement values. Neither verdict depends on them:
- **Left.** It passes the study's 15 / 5 V3 rule, has no bridge, and is in band with the IQR term.
- **Right.** It fails at any V3 threshold tested (4.9–11.3 mm, d 15–18 mm) and at any V1 tolerance.

## 3. CT wall of the left VA (ct_seg4, method and parameters unchanged)

**Method.** `pilot_ct_seg.py` is `tag_ct_seg.py` (that is, `vmr_ct_seg3.py` with `VMR_OPEN_BRIGHT=1`) plus a loader,
`pilot_ctwall_common.py`.
- The seed is the VMTK line of Arterial's cut-out, with the subclavian from Arterial's LSA labels and the basilar from
  the joined traces.
- The opening element on this grid is the 7-voxel ±1 cross, about 0.8 mm, as on VMR.

The third instance fixed two problems in the loader:
1. **The CT is now read once.** Random access into the gzipped CT had left the first run in disk sleep for 12 minutes,
   so it was killed.
2. **The proximal V1 is restored.**
   - Arterial's mask pinches at trace s 16–18: EDT 0.35 mm. The CT lumen there is continuous, at 260–280 HU with a
     1.7–1.9 mm half-max radius.
   - `vmtkCenterlines` therefore returned a line starting 16 mm above the ostium. art_extract's retry only fires below
     50% of the seed distance.
   - The trace's own first 17.9 mm is now prepended (recorded in `seg_qc.json`).
   - The run without it is kept as `ct_seg/ct_seg4_vmtkonly_noV1`.

**Segmentation QC** (`ct_seg/ct_seg4/seg_qc.json`)
- **Rays.** 64 rays, cap 0.5 mm. 87.9% soft-tissue half-max, 5.6% dip-before-bright, 3.3% steepest descent, 3.2%
  bright-neighbour upper threshold.
- **Opening.** 415 stations flagged, 609 voxels removed.
- **Intensities.** I_core 341 HU, background 82, T_loc 223.
- **Contacts.** Label voxels touching ≥ 700 HU: 0.
- **Size.** r_rays 2.43 mm, r_vol 2.38.
- **Surfaces.** Closed, genus 0 (iso: genus 1).
- **CT VMTK line.** 244.5 mm, MISR median 1.93 mm. It runs 0.86 mm median (p95 2.35) from the trace. It is about 14 mm
  longer than the Arterial line in the C2 → C1 region, where the label is wider.

**Sections** (`qc/Dissec_01_segxsec_L_*.png`)
- The label is tight in V2–V4.
- In V1 it takes in an adjacent vessel of equal HU at s 15 and s 25 (the vertebral vein, enhanced in this phase).
- It protrudes at s 35, 55, 110 and 120.

**Standard grade** (`measure.json` grades, `va_standard` unchanged, native thresholds, acquisition voxel 0.779 × 0.779 ×
0.625 mm, exact ownership k 64; k 24 gives identical numbers)

| rule | value | result |
|---|---|---|
| V1 centreline offset | **0.189 mm** (mean 0.417) | **FAIL** (≤ 0.15). W1–W6 are therefore "n.a.": the wall numbers are not valid |
| V2 / V3 / V4 / V5 / V6 | 0.0 / 0.0 / genus 0 / 0.78 mm / VMTK MISR | pass |
| G1 calibre | 1.90 mm | pass |
| G2 bend radius | 2.58 mm (the C2 loop) | pass, narrowly |
| G3 narrowing | q_min 0.544 | FAIL |
| G4 length / DM | 244.5 mm / 1.63 | pass |
| W7 by hand | 0.115 / 0.272 | would pass (≥ 0.0785 / 0.2183) |

**Union detail, raw, for comparison only** (ct_seg4 stage-A, core; W3 always beside W1)

| stretch | > 0.3 mm | > 0.5 mm | > 0.15 r |
|---|---|---|---|
| whole VA | 0.387 | 0.255 | 0.370 |
| V1 (origin → C6, s 0–32) | 0.486 | 0.359 | 0.487 |
| V2 (C6 → C2, s 32–128) | 0.495 | 0.359 | 0.479 |
| C2 → C1, foramen exit on the "total" masks (s 128–141) | 0.212 | 0.064 | 0.137 |
| C1 → dura (s 141–208) | 0.293 | 0.157 | 0.254 |
| V4 (s 208–244.5) | 0.127 | 0.023 | 0.177 |
| **cervical C6 → C1** (C1 = foramen exit, s 141) | **0.460** | 0.322 | **0.437** |
| C6 → C1 (Erec total_v3 C1 = groove end, s 177 / ts_qc C1 s 184) | 0.418 / 0.426 | 0.31 / 0.32 | 0.383 / 0.390 |
| origin → C1, with the last 25 mm as the seam zone counted separately | 0.491 (seam zone 0.358) | | |
| as pieces alone (C6 → C1, origin → C1; Erec) | 0.416, 0.436 | | |
| iso surface (whole) | 0.423 (genus 1) | 0.301 | 0.411 |
| 10 mm bins that are plain tube (< 0.01) | **0 of 24** | | |

**References.**
- VMR 0248 (whole): 0.045 / 0.138; C6 → C1 W1 0.099, W3 0.151.
- TaG-Net A (whole): 0.045 / 0.111; C6 → C1 W1 0.093–0.111.
- TopBrain: pieces 0.121, distal 0.083 / 0.230.
- Plain-tube bins: TopBrain 14% (use-weighted 22%), VMR 55% / 35%, TaG-Net 76% / 23%
  (`results/pilot_proc_aortaseg_tbbins.json`).
- Host tube: 0.0007.

**Where the relief sits** (`attrib.json`; vertex-area approximation)
- **Zones.** Zones are read from the CT along the outward normal at 0.5–2 mm: bone if the maximum is > 600 HU, "bright"
  if the median is ≥ 0.6 × the lumen core. The TotalSegmentator vertebra masks fill the foramina, so they would call
  everything "bone".
- **Whole VA.**
  - Bone: 3% of the area, holding 4% of the > 0.3 mm area.
  - Bright neighbour (vein, plexus, branch or marrow continuing outside): 24% of the area, 31% of the > 0.3 mm area.
  - Free: 73% of the area, 65% of the > 0.3 mm area. Even there, 38% of the free area is > 0.3 mm out.
- **Bulges.** Bulges > 1 mm cover 15% of the wall: 30% in V1 and 23% in V2.
- **V4.** 95% free, and still 0.17 reads > 0.3 mm.
- **Pattern.** The unrolled map (`qc/Dissec_01_wall3d_L.png`) shows the relief as axial-band bulges in V1–V2, not as a
  continuous texture.
- **Slice test.** A per-slice centroid test of the CCAs, trachea and VA (`slices.json`, `qc/Dissec_01_slices.png`)
  found no single-slice jumps. It cannot confirm or exclude a gating stair-step.

**Is it real?** (`controls/floor.json`, `pilot_proc_aortaseg_floor.py`)

| surface (left VA) | union > 0.3 / > 0.5 / > 0.15 r | centre offset (median) |
|---|---|---|
| smooth synthetic tube: this VA's course and ray r_eq, voxelised on the same grid, same stage-A + VMTK | **0.028** / 0.017 / 0.022 (one 40 mm bin 0.13 at the tight C2 loop) | 0.031 mm |
| Arterial's own cut-out surface (from s 18) | 0.180 / 0.072 / 0.165 (VMR / TaG-Net Arterial native: 0.01–0.07) | 0.085 mm |
| ct_seg4 stage-A | 0.387 / 0.255 / 0.370 | 0.189 mm |

Section and roughness statistics against the references (centroid-based, `wall_sections` / `wall_native`):

| | Dissec_01 L | VMR R / L | TaG-Net R / L |
|---|---|---|---|
| perimeter > 0.3 mm off the equal-area circle, mean | 0.235 | 0.020 / 0.056 | 0.024 / 0.034 |
| centre offset, median / mean (mm) | 0.189 / 0.417 | 0.04 / 0.08, 0.07 / 0.14 | 0.03 / 0.08, 0.06 / 0.12 |
| roughness 0.75–1.5 / 1.5–3 / coarse (mm RMS) | 0.162 / 0.201 / 0.213 | 0.08 / 0.07 / 0.06, 0.11 / 0.09 / 0.08 | 0.07 / 0.07 / 0.07, 0.08 / 0.10 / 0.09 |
| image half-max edge rays > 0.3 mm off the station median, whole | 0.354 (V4 0.143) | 0.082 / 0.154 | 0.067 / 0.109 |
| edge FWHM, verifier / vmr method (mm) | **1.69 / 1.65** | 1.39 / 1.44 | 1.30 / 1.29 |
| lumen noise, CNR | 14.2 HU, 18.3 | 28 HU, 16–17 | 15 HU, 35–37 |

**Reading.**
- The staircase floor is about 0.03, so the 0.39 is image content passed through ct_seg4. The edge is blurrier and its
  position less stable than on VMR or TaG-Net, and the lumen touches veins and plexus of equal HU.
- The surface is coarse and irregular, which is what the user asked for. But the share that is true VA anatomy (oval
  sections, contact dents, stubs) rather than vein halo and partial volume cannot be separated at 0.78 mm.
- The standard's own validity gate (V1) rejects it.
- A PILOT_UPPER_HU 350–400 sensitivity, suggested by the fetch agent, was not run. Bone contacts carry only 4% of the
  relief, so it cannot change the conclusion.

## 4. Gross shape per side (va_metrics unchanged, σ 3 mm)

**Left** (the CT VMTK line, which starts at the ostium)

| measure | value | real-VA range (tag_calibration §5) |
|---|---|---|
| length (mm) | 244.5 | 193–266 |
| DM | 1.63 | 1.32–1.75 |
| take-off at 10 mm (fwd5 on Arterial's LSA label) | 71.9° (80.9° at 5 mm) | 42–106° |
| subclavian elevation at the origin (fwd5) | 21.3° | 12–64° |
| VA elevation | 60.9° | |
| C2-loop bend radius | 2.58 mm (s 121.8) | 1.37–3.25 mm |
| loops ≥ 360° in 60 mm | 1 | |
| V1 / V2 / V3 / V4 (Erec on the CT line: C6 32, C2 128, dura 208) | 32 / 96 / 80 / 36.5 mm | |
| entry level | C6 (Erec; ts_qc C7) | |

- The trace length is 217 mm (origin → dura 182). The CT line is about 13% longer because it meanders in the wide
  lumen. The Arterial VMTK line is 212.5 mm, but starts 16 mm above the ostium; its 120° take-off is not an ostium value.

**Right.**
- Entry C6, V1 30 mm.
- V2 is on the VA only up to C3.
- Take-off is not measurable: the origin is bridged and there is no RSA label.

**Calibre** (`calibre.json`, V2 window)
- L MISR 2.44 mm (× 1.10 = 2.68), R 1.32 mm (1.46), so **L/R is 1.84** (by EDT, 1.93).
- The verifier puts the native factor at 1.10–1.21, depending on the protocol; the ratio does not depend on it.
- The left V2 lumen, about 5 mm, fills the foramen. An enhanced plexus could contribute at 0.78 mm. But the right
  foramen shows its VA in dark fat, with no enhanced plexus, so a true large dominant left VA is the likelier reading.

## 5. Files

**Scripts** (`/sp/v5_va/`). New in the third instance:
- `pilot_proc_aortaseg_run.sh` (CPU runner);
- `_rlook.py`, `_rlook2.py`, `_rlook3.py` (right-side checks);
- `_xprof.py`, `_segxsec.py` (cross-sections);
- `_calibre.py`;
- `_floor.py` (controls);
- `_attrib.py`, `_slices.py`;
- `_measure_extra.py`;
- `_collect.py` (results).

**Extended by the third instance** (the originals are kept as `state/*.orig_1st_instance`):
- `pilot_proc_aortaseg_qc.py`: the recalibrated gate and Erec; the draft gate kept;
- `pilot_proc_aortaseg_landmarks.py`: the Erec set;
- `pilot_proc_aortaseg_measure.py`: Erec as the primary set;
- `pilot_proc_aortaseg_edge.py` and `pilot_proc_aortaseg_fig.py`: Erec;
- `pilot_ctwall_common.py`: CT read once, V1 prepend;
- `pilot_ct_seg.py`: the prepend recorded;
- `pilot_proc_aortaseg.sh`: env pass-through, the ct_seg image, line-buffered logs.

**Unchanged:** `art_extract.py`, `tag_calibration*.py`, `va_metrics`, `va_standard`, `vmr_measure`, `vmr_vmtk.py`,
`art_vmtk_one.py`, `tag_ctwall_edge`, `mask_to_surface`.

**Results:**
- `results/pilot_proc_aortaseg.json`: one record per case and side, with QC, trace, landmarks, gross, calibre, wall,
  controls, attribution and the visual checks;
- `results/pilot_proc_aortaseg_tbbins.json`: the TopBrain / VMR / TaG-Net plain-tube bin reference.

**Data** (`derived/thin_slice/aortaseg/Dissec_01/`, 277 MB):
- `totalseg*`;
- `arterial/{runs/native, ts, tsba}`;
- `ct_seg/{ct_seg4, ct_seg4_vmtkonly_noV1}`;
- `controls/`;
- `qc.json` (in `qc/`), `landmarks.json`, `measure.json`, `measure_extra.json`, `edge.json`, `attrib.json`,
  `calibre.json`, `xprof.json`, `slices.json`;
- `qc/*.png`. The QC PNG for the case is `qc/Dissec_01_qc_tsba.png` (coronal MIP with both traces coloured by QC
  status, sagittal slabs, CPRs and HU profiles). The 3-D wall render is `qc/Dissec_01_wall3d_L.png`.

**Logs:** `derived/thin_slice/aortaseg/_logs/`.

## Open issues

1. **One case** cannot rate a source. AortaSeg-60 has no other usable case (notes/pilot_fetch_aortaseg.md).
2. **Carotid blocking depends on Arterial's labels.** When Arterial mislabels a whole side, as here, the ICA is free.
   A TS-independent block is needed: for example, a tube around any vessel continuing above the TS CCA top, or the
   TotalSegmentator `headneck_bones_vessels` task.
3. **VMTK silently drops a proximal V1 at an Arterial mask pinch.** art_vmtk_one's retry only fires below 50% of the
   seed distance. A check "VMTK start within 3 mm of the trace origin" belongs in the batch QC.
4. **The C1 foramen exit is still undefined on TotalSegmentator masks.**
   - On this CT line, total_v3 merges the foramen and the groove (s 128–183); the "total" masks separate them at about
     s 141.
   - C6 → C1 reads 0.46 (exit at 141) or 0.42 (s 177 / 184).
5. **The wall question for thin-slice CTs depends on the scanner and phase.** Dissec_01 is a 0.78 mm, 1.7 mm-FWHM,
   late-phase aortic CTA, not a neck CTA. Its irregularity is mostly image and contact content. Song and 172-series
   cases (about 0.4 mm, neck CTAs) must be judged on their own controls.
6. **The draft gate's "origin" rule** is ambiguous: junction or ostium. Here it reads 0.0 or 2.23 mm.
