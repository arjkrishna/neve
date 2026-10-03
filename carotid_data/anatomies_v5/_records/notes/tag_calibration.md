# tag_calibration: the v5 extraction and graft checks recalibrated on two ground-truth patients

Agent prefix `tag_calibration`. Date 2026-10-01. No git. No tracked file and no existing `/sp/v5_va` script, note or
result was modified: the study tools (`art_extract.ts_qc` / `load_ts`, `tag_totalseg_qc.v3loop`, `va_metrics`,
`tag_labels_gross.takeoffs`, `tag_totalseg_make1p5.py`, `tag_totalseg_run.sh`) are imported or run unchanged.

**Licence.** TaG-Net has no data licence: everything derived from patients A and B is internal benchmarking only.
VMR 0248 is research use (Wilson et al. 2013). Arterial-derived traces are non-commercial. TotalSegmentator v3 data
is CC BY 4.0.

**Ground truth (GT) used.**
- **VMR 0248:** expert SimVascular paths and the ct_seg4 CT centreline, both sides. Only the right side's
  landmarks are trusted; the left entry *level* is C6 by both readings.
- **TaG-Net A:** GT points (ostium → merge), full points, and the ct_seg4 CT centreline. Landmarks are
  semi-manual, ±3 mm.
- **TaG-Net B:** GT path only. It has no image and no vertebrae.
- **Traces:** every Arterial trace and VMTK line of both patients at native, 1.5 mm, 1.5 mm + clean and 1.5 mm +
  sharp TS noise (VMR 8, A 21 extractions × 2 sides).
- **TotalSegmentator donors:** the 21 Arterial runs (24 subjects tried; 42 sides, 40 traced) in
  `derived/arterial/ts_*`. The study's corrected yield is 15 usable neck VAs from 11 subjects, both sides in 4.

**New input made here: VMR 0248 vertebrae.** TotalSegmentator 2.18.0 was run on the VMR native CT with the
unchanged `tag_totalseg_run.sh` (GPU): `total_v3` in 107 s and `total` in 91 s. 1.5 mm RAS copies were made with
`tag_totalseg_make1p5.py`. The C1–C7 masks are ordered, and C6 spans z −245.5…−220.0, against the study's C6 entry
at z −229.6. Data is in `derived/tagnet/calibration/`; see its README. These masks are VMR-derived, not TaG-Net.

## Answer first

| item | GT values | recommended | basis | TotalSegmentator traces (40 traced sides; study: 15 usable / 11 subjects) |
|---|---|---|---|---|
| **1 V3-loop gate** | excursion / distance to the posterior arch of C1 (mm), total_v3 masks: VMR 20.2 / 0.5 (R), 19.4 / 0.4 (L); A 17.8 / 0.3 (R), 18.9 / 0.4 (L) with the study measure, **17.1 and 14.9 with a robust cord reference**; A on "total" masks 15.7 / 14.3. B: path-only proxy only (17.0 / 18.2 tilt-corrected) | **robust-cord excursion ≥ 12 mm and d_C1post ≤ 6 mm** (study: ≥ 15 and ≤ 5) | **two patients** for real loops (4 GT VAs, 26 correct traces) plus 68 wrong A traces | **19 / 40 pass**: all 15 of the corrected usable, 0 of the 6 wrong-vessel sides, plus 4 that fail other gates. The study gate on the same traces passes 18 (it loses s0346 L, d_C1post 5.1 mm) |
| **2 C6 entry** | truth C6 on all 4 GT VAs. ts_qc calls **C7** on 9 of 10 GT paths and 91 of 94 traces (total_v3), 10/10 and 16/16 on "total"; its C6 landmark is 8–14 mm early | **Erec**: keep ts_qc's call, but accept C7 only when the path has its own ≥ 5 mm run within 0.5 mm of the C7 mask, ≥ 2 mm before the C6 run; otherwise call C6 | **two patients**, 130 path × mask-set rows: Erec C6 on 129 | ts_qc: C6 16, **C7 20**, C5 1, none 3; on the 15 usable, 9 are C7. **Erec: C6 33, C7 3, C5 1, none 3; all 15 usable are C6**. The count of foramina (≥ 4) is unchanged: 36 / 40 |
| **3 radius** | VMTK MISR / image half-max radius, origin → C1. Native: VMR 0.87 / 0.93, A 0.89 / 0.94. 1.5 mm: VMR 0.86 / 1.01, A 0.79 / 0.86. Trace EDT 0.70–0.87. ct_seg4 MISR 0.93–0.95 on both | **native: one factor works, MISR × 1.10** (VMR 1.11, A 1.10, 1.5% apart). **1.5 mm: no single factor** (VMR 1.05–1.08, A 1.20–1.21, 11–14% apart). Take calibre from the CT half-max profile, as plan Stage 4 says; never from the trace EDT | **two patients**; consistent at native, inconsistent at 1.5 mm | radius band 0.9–2.6 mm: 38 / 40 raw, 40 / 40 with × 1.10–1.25 (s1029 L 0.89 and s1823 L 0.85 pass). Yield unchanged: both fail other gates |
| **4 draft QC gate** (plan Stage 2 + report additions) | GT paths of both patients pass every draft rule; VMR is flagged for review by the BCV rule (266 HU). Correct traces: VMR native R/L and A native L **fail rule 6** (Arterial's label off the trace: 0.00 / 0.87 / 0.00). A at 1.5 mm: every trace is wrong above C1 and is rightly rejected | **move 6 limits**: rule 6 to review only; drop the band's IQR term; allow V1 mask breaks ≤ 3 mm; total of 25 mm counted below the splice only; V3 loop 12 / 6; BCV ≥ 200 to review. Also fix the C6 landmark (item 2) | **two patients** (rule 6, BCV, V3 loop); band, V1 and total from the TS sides | **draft as written + additions: 9 / 40, 7 subjects, 2 pairs** (loses 6 of the 15). **Recalibrated: 15 / 40, 11 subjects, 4 pairs (s0610, s0777, s0960, s1786)**: the corrected yield exactly, and 0 of 6 wrong-vessel sides. 12 of the 15 still carry a review flag; only 5 have a bridge > 2 mm below the splice |
| **5 gross ranges** | A and B added (table in §5). Real VAs: length 193–266 mm, DM 1.40–1.75, C2-loop bend 1.37–3.25 mm, take-off 42–93°, subclavian elevation 12–64° | — | 5 real patients (VMR, A, B, SPL, TS n = 11) | **The host is the outlier** in length (209/210 mm, below every GT VA) and C2 loop (none; min bend 4.2/5.4 mm). **VMR is the outlier** in right subclavian elevation (64°; the others 12–45°) and right V2 length (67 mm; the others 74–110) |

## 1. V3-loop gate

**Definitions** (`tag_calibration_paths.py`, rows in `tag_calibration_paths_{VMR,A,TS}.json`):
- *study*: `tag_totalseg_qc.v3loop`, unchanged (= `verify_arterial_yield_v3loop`). The measure is dy = y − y_cord(z).
  - excursion = median dy over V2 (C6 mid … C3 top) − min dy over V3 (z ≥ C2 bottom, s ≤ ts_qc's dura);
  - d_C1post = minimum distance from V3 to the posterior half of the C1 mask.
- *zcap*: the same, but V3 ends at C1 top + 10 mm instead of the dura proxy. It equalled *study* within 0.1 mm on every path, so
  the dura proxy is **not** what moved patient A's left value.
- *robust*: zcap with a cord centroid that ignores partial top slices (fewer than 50% of the median slice
  count). **This is what moved it.**
  - The TS cord mask ends at the foramen magnum in 1–11 partial slices, whose centroid jumps by up to about 6 mm
    in y (A, total_v3: −3.6 mm at the last full slice, +2.6 mm at z −528 from 4 voxels). That moved A left's
    excursion by 4.0–4.7 mm.
  - The V3 loop's most posterior point lies exactly there: 1.4–3 mm below the C1 top.
  - 18 of the 20 TS subjects with a trace have such partial top slices.

**GT values** (total_v3 masks; "total" in brackets; excursion / d_C1post, mm):

| path | study | robust |
|---|---|---|
| VMR expert R / L | 20.2 / 0.5, 19.4 / 0.4 (20.2 / 0.7, 19.3 / 0.4) | same |
| VMR CT R / L | 20.3 / 0.2, 18.7 / 0.2 | same |
| A GT R / L | 17.8 / 0.3, **18.9** / 0.4 (15.7, **14.3**) | 17.1, **14.9** (15.7, 14.3) |
| A CT R / L | 17.9 / 0.2, 19.0 / 0.2 (15.9, 14.5) | 17.0, 14.9 |

- **Patient A's left VA is a real V3 loop (GT) that reads 14.3–14.9 mm** whenever the cord reference is stable, or
  the "total" masks are used. The study's ≥ 15 mm rule rejects it. VMR's loops read 18.7–20.4 mm under every
  variant.
- **Traces** (robust, total_v3):
  - VMR, 16 traces / VMTK lines at all four qualities: 18.0–20.8 / 0.1–0.4. All pass.
  - A, 10 correct-course native traces: 14.5–17.2 / 0.1–0.7.
  - A, **68 wrong-vessel traces**: 0.3–3.2 / 12.5–26.9. These are every 1.5 mm variant, and the native VMR call,
    "--ts total" and TS-skull "nots" runs. **0 of 68 pass at any threshold tested.**
- **TotalSegmentator sides** (trace, dataset masks; the verifier's VMTK numbers agree to ±1 mm):
  - With d ≤ 5: excursions 4.6 (s1823 L, a no-foramen trace), then 18.2–31.2.
  - With d > 5: 0.0–13.3 (s0333 R 13.3 / 25.2; s0857 R 11.2 / 9.0), plus **s0346 L 18.0 / 5.1**. That is a real loop
    (verifier: 18.2 / 4.8), so d ≤ 5 is on the edge.
  - The wrong sides' d is ≥ 7.5 (s0518 L 8.5 / 7.5).
- **Threshold choice.** d is the main separator: real loops reach ≤ 5.1, wrong traces ≥ 7.5, so **d ≤ 6**.
  Excursion ≥ 12 leaves 2.3 mm under A left's 14.3; every other trace with d ≤ 6 reads ≤ 4.7 mm (s1823 L, a no-foramen trace).
  The `threshold_sweep` in `results/tag_calibration.json` shows the stable region:

  | gate | GT paths | correct traces | wrong traces |
  |---|---|---|---|
  | **robust, ≥ 12 and ≤ 6** | 10/10 (total_v3), 10/10 ("total") | 26/26, 14/14 | 0/68, 0/2 |
  | study, ≥ 15 and ≤ 5 | 10/10 (total_v3), 7/10 ("total") | 26/26, 8/14 | 0/68, 0/2 |

  The study gate fails A left on "total".
- **TS counts.**
  - Recommended gate: 19 / 40. That is the 15 corrected usable, 0 of the 6 wrong (s0289 R, s0333 L, s0518 L, s1107 R,
    s1129 R, s1154 R), and s1029 R/L, s1075 R and s1192 R, which fail other gates.
  - Study gate (15 / 5) on the same traces: 18 / 40. It loses s0346 L.
  - With d ≤ 8, s0518 L (wrong) passes. Do not loosen d further.
- **B (path only, no vertebrae).** It cannot give d_C1post, a cord-relative excursion or any landmark. What can be
  checked:
  - *Tilt-corrected proxy* (V2 = 0.20–0.45 L, y(z) fitted on V2 and extrapolated; V3 window 0.45–0.90 L):
    **B 17.0 / 18.2 mm**, against A 15.1 / 15.0 and VMR 22.2–23.5. Consistent with real loops.
  - *Raw proxy* (no tilt correction): B 6.4 / 5.3, A 17.2 / 15.1, VMR 18.9–19.8. B's neck is tilted: its V2 slope
    dy/dz is 0.17 / 0.23, against A −0.04 / 0.00 and VMR 0.08–0.09.
  - B's most posterior point sits at 0.76 / 0.77 L, against A 0.74 and VMR 0.73–0.76: the C1-arch position.
  - B has one loop ≥ 360° in 60 mm per side.
  - The proxy is **not** usable as a gate on traces: TS real −12…31 and wrong −19…11 overlap, because trace
    windows include wrong distal stretches.

## 2. Foramen entry level

**Rules tested** (`tag_calibration_paths.py`, `tag_calibration_entry.py`; definitions in
`items.2_entry_level.rule_definitions`):
- E0 = ts_qc;
- own-z, 5 mm-ray and run variants (E0own, E0r5, E1, E1h, E1z, E2, E2h, E3);
- a majority-run rule (E5; failed: adjacent foramina merge into one run);
- the grid R(d_thr, run_min, gap_min);
- **Erun = R(0.5, 5, 2)** and **Erec**.

**Why ts_qc over-calls C7.**
- The TotalSegmentator vertebra masks fill the transverse foramina, and C7's z range overlaps C6's by about 9 mm.
- A VA entering at C6 is enclosed by, or within 1 mm of, the C7 label for 3–9 mm, just below or at the lower rim of
  the C6 foramen.
  - On A's left VA the path is even *inside* the C7 mask for 3–5 mm, contiguous with the C6 run.
  - Median distance to C7 over its z band is 3.4–4.7 mm; at C6 it is 0.0–0.5 mm.
- This holds at native and at 1.5 mm, on GT, CT and trace paths, and with both mask sets. So it is the test, not
  the resolution.

**Calls on the GT patients** (truth C6 on all four VAs):

| rule | GT paths C6 (total_v3 / "total") | traces C6 (total_v3 / "total") |
|---|---|---|
| E0 = ts_qc | **1/10 / 0/10** | **3/94 / 0/16** |
| E1z (≥ 3 mm inside the mask) | 7/10 / 7/10 (A left: C7) | 52/94 / 10/16 |
| E2 (5 mm rays ≥ 0.75) | 4/10 / 4/10 | 16/94 / 3/16 |
| **Erun = Erec** | **10/10 / 10/10** | **94/94 / 15/16** (one A VMTK line on "total" → C7) |

**Recommended rule (Erec).**
- Keep ts_qc's level count (≥ 4 of C6..C1) and its call.
- Accept a C7 call only if Erun also says C7:
  - Erun runs are stretches with z inside the level's own range and the path within 0.5 mm of the level mask,
    at least 5 mm long;
  - the C7 run must end ≥ 2 mm before the first C6 run.
- Otherwise call C6. The C6 landmark is the start of the C6 run.

**Landmarks** (projected on the GT path, + = distal; total_v3):

| landmark | Erec | ts_qc | VMR | A |
|---|---|---|---|---|
| C6 | −0.2…−1.2 mm on A and VMR left; **−10 mm on VMR right** | −7…−14 mm | | |
| C2 (ts_qc's, unchanged) | | | +7.3…+8.9 mm late | +6.7…+11.8 mm late |
| C1 (ts_qc's) | | +26…+40 mm, the posterior-arch groove (as tag_ctwall found) | | |
| C1 (first C1 run exit) | | | within 2 mm | +9.5 mm (R) or +38 mm (L, the run continues along the groove) |

- VMR right's C6 offset: there the TS C6 mask encloses the VA about 10 mm below the CT bony ring at s 44.75.
- The C2 offset is systematic: the TS C2 mask includes the C2 loop. ts_qc's V2 is therefore about 8–11 mm long and
  its V3 short on both patients.
- C1: no mask rule finds the C1 foramen exit reliably at 1.5 mm. That is an open issue for the splice point.

**TotalSegmentator sides (40).**

| rule | C6 | C7 | C5 | none | on the 15 usable |
|---|---|---|---|---|---|
| ts_qc | 16 | **20** | 1 | 3 | 6 C6, **9 C7** |
| **Erec** | **33** | 3 (s1075 R, s1138 R/L) | 1 (s1192 R) | 3 (no-foramen traces) | **15 C6** |

- The verifier's axial test also says C7 first on 19 / 40, so it shares the over-call and is not truth. Erec
  agrees with it on 23 / 40.
- An anatomical C7 entry is rare (about 5%); 20 / 40 is not plausible, 3 / 40 is.
- Plan §3 counted "entry level C7: 4 of 8 sides over-called". This confirms it on 2 GT patients, and Erec removes
  it.

## 3. Radius correction

**Method** (`tag_calibration_radius.py` → `results/tag_calibration_radius.json`).
- The reference is the image half-maximum equal-area radius r_img (vmr_ct_seg3 rays, pass 2, σ 1 mm): VMR from
  ct_seg4's `stations_rays.json`, A from the GT radius. This is the same definition on both patients.
- Candidates per Arterial extraction:
  - **trace** = art_extract's r_edt;
  - **VMTK** MISR;
  - **maskEDT** = the maximum EDT on the normal disc;
  - **maskArea** = the equal-area radius of the cut-out mask's normal section.
- Only stations where the trace is on the VA (within 1.5 mm) and away from bridged gaps count.
- "clean" drops r_img spikes above 1.3 × the running median (vein and bone contacts). The table uses origin → C1,
  the stretch every trace keeps. Whole-VA ratios differ from these by up to 0.05.

**Ratio to r_img** (R / L), origin → C1. A's native row is the production TS-skull run.

| | VMR native | VMR 1.5 | VMR 1.5 + clean | VMR 1.5 + sharp | A native | A 1.5 | A 1.5 + clean | A 1.5 + sharp |
|---|---|---|---|---|---|---|---|---|
| VMTK MISR | 0.87 / 0.93 | 0.86 / 1.01 | 0.89 / 1.00 | 0.90 / 1.02 | 0.89 / 0.94 | 0.79 / 0.86 | 0.81 / 0.86 | 0.82 / 0.84 |
| maskArea | 0.92 / 1.01 | 0.93 / 1.09 | 0.96 / 1.08 | 1.00 / 1.10 | 0.93 / 0.98 | 0.85 / 0.91 | 0.90 / 0.91 | 0.89 / 0.90 |
| maskEDT | 0.82 / 0.89 | 0.82 / 0.97 | 0.85 / 0.96 | 0.85 / 0.98 | 0.84 / 0.90 | 0.77 / 0.83 | 0.78 / 0.82 | 0.78 / 0.80 |
| trace r_edt | 0.76 / 0.81 | 0.76 / 0.87 | 0.78 / 0.87 | 0.79 / 0.88 | 0.71 / 0.71 | 0.74 / 0.73 | 0.73 / 0.72 | 0.75 / 0.70 |
| ct_seg4 MISR | 0.94 / 0.93 | | | | 0.95 / 0.94 | | | |

- r_img medians: VMR 1.39 / 1.63 mm, A 1.25 / 2.04 mm.
- A native with the VMR-skull run instead: MISR 0.77 / 0.90, maskArea 0.85 / 0.96.

**Is one factor consistent?** The factor below is r_img / VMTK MISR (`factor_consistency_origin_to_C1`).
- **Native: yes.**
  - VMR 1.16 / 1.07, A 1.13 / 1.07: mean 1.11 against 1.10, 1.5% apart, with a side spread of 5–8%.
  - **MISR × 1.10 (±0.04)** brings a native Arterial VA to the half-max radius on both patients. maskArea needs
    × 1.04.
- **1.5 mm: no.**
  - VMR 1.16 / 0.99 against A 1.26 / 1.16: means 1.05–1.08 against 1.20–1.21, 11–14% apart, which is more than A's
    side spread.
  - An additive offset is no better: VMR −0.03…+0.19 mm, A +0.17…+0.30 mm.
  - A pooled × 1.14 carries about ±10% (±0.15 mm).
  - The thin VAs (A R 1.25, VMR R 1.39 mm) under-read most, but A's 2.0 mm left also needs 1.16. So the factor depends
    on the patient (A's brighter, sharper lumen; Arterial's surface sits 0.11–0.15 mm inside A's edge), not on
    calibre alone.
- **CT-segmentation MISR is consistent:** 0.93–0.95 × r_img on both patients. That supports plan Stage 4: measure
  calibre from the CT half-max profile along the VMTK normals. The trace EDT runs 13–30% low; do not use it.

**TotalSegmentator effect.**
- Radius band 0.9–2.6 mm (VMTK MISR median):
  - 38 / 40 pass raw (s1029 L 0.89 and s1823 L 0.85 fail);
  - 40 / 40 pass after × 1.10 to × 1.25.
- G1 (1.09–2.25 mm): 34 → 35–37 / 40.
- The yield does not change: s1029 L also fails its 7.8 mm V1 bridge and the 25 mm total, and s1823 L is a
  no-foramen trace.
- The 0.9 mm floor would reject true hypoplastic VAs (diameter < 2 mm, wanted by the diversity plan). Apply it to the
  *corrected* radius and use it only to catch mask fragments.

## 4. The draft QC gate (donor-pool plan Stage 2) and the report's additions

The implementation is in `tag_calibration.py` (`apply_gate`). For GT paths the origin rule reads the ostium. "Draft"
applies the rules as written; "+ additions" adds the report's V3-loop gate (15 / 5) and C1 threading.

**GT patients, with every draft value:**
- origin: 0.0–1.65 mm;
- foramina: 6 of C6..C1;
- origin → dura: VMR 188–195 mm, A 186–219 mm (GT and CT paths);
- V2 band: |dx| 13.9–18.0 mm, dy 10.6–14.3 mm, IQR 1.1–2.4 mm;
- cervical HU / CCA on the 1.5 mm CT: VMR 0.79–0.96, A 0.70–1.02. A's thin right VA is the closest to the 0.6
  limit;
- BCV minimum: A 50 HU, VMR **266 HU** (its injection-side BCV reads 2278 HU).

| path | draft + additions | recalibrated |
|---|---|---|
| VMR GT (expert, CT), R / L | pass, BCV review | pass, BCV review |
| VMR traces, native R / L | **fail rule 6** (label 0.00 / 0.87) | pass (review) |
| VMR traces, 1.5 mm ×3, R / L | pass | pass |
| A GT (points, full, CT), R / L | pass | pass |
| A native, TS skull + `--ts` (production), R / L | pass | pass (R flagged: 7.1 mm V4 bridge) |
| A native oracle end, R / L | R pass; **L fails rule 6** (label 0.00) | pass |
| A native, VMR skull + `--ts`, L (correct) / R (wrong) | **L fails rule 6**; R rejected (3c 11.6 mm + V3 loop) | L pass; R rejected |
| A, every 1.5 mm trace (wrong above C1) | rejected: 3c on R, rule 6 on L; the V3 loop rejects all | rejected by the V3 loop (and 3c / 3b / reach) |

So **both GT VAs pass at native** under the recalibrated gate, and under the draft only with the production protocol.
**At 1.5 mm, VMR's pass and A's are correctly rejected**: their course is wrong above C1. As origin → C2 donors
they would be accurate (V1–V2 VMTK ≤ 0.34 mm, tag_arterial), but no gate admits that class.

**Limits that must move** (evidence → change):

| rule | evidence | change |
|---|---|---|
| 6 label agreement ≥ 0.9 | rejects 3 correct GT-patient traces and 2 of the 15 usable TS sides (s0610 L 0.76, s0777 L 0.00). Arterial's labels are unreliable on both patients | **review only** |
| 5 V2 band IQR ≤ 4 mm | a screening-trace criterion; rejects s1129 L, s1304 R, s1823 R of the 15; every GT passes with or without it | **drop the IQR term**, keep \|dx\| 12–22, dy 7–19 |
| 3a "no V1 bridge" | s0610 L (1.9 mm) and s1304 R (2.9 mm) are mask breaks at the subclavian junction | **tolerate ≤ 3 mm**, with V1 measured to the Erec C6 landmark |
| 3d total ≤ 25 mm | the draft counts distal gaps above the splice, which a strategy-B donor does not take | **count gaps below the splice only** |
| A1 V3 loop ≥ 15 / ≤ 5 | A left 14.3–14.9 (real); s0346 L d 5.1 (real) | **robust cord, ≥ 12 / ≤ 6** (§1) |
| 5c BCV < 200 HU | VMR, a clean arterial CTA, reads 266 (the other arm 2278); s1154 reads 320 | a value ≥ 200 → **visual review, not reject** (the draft's own fallback) |
| 4b length 150–260, > 270 review | GT origin → dura 186–219, TS usable 177–233. Nothing is near the limits, but 260–270 is undefined | pass 150–260, **review 260–290** |
| landmarks for 3a–3c | ts_qc's C6 is 8–14 mm early, its C2 7–12 mm late and its C1 26–40 mm late | **C6 from Erec**; C2 and C1 are open issues |

**Kept and supported by both patients:**
- origin ≤ 2 mm (GT 0–1.65 mm);
- foramina ≥ 4 (GT 6; ts_qc counts match the verifier on 39 / 40);
- C1 threading;
- cervical HU ≥ 0.6 × CCA (GT ≥ 0.71);
- the V2 band without its IQR term.

**TotalSegmentator counts.**

| gate | sides / 40 | subjects | both sides | of the 15 usable | wrong-vessel passing |
|---|---|---|---|---|---|
| draft as written + additions | **9** | 7 | 2 (s0960, s1786) | 9 | 0 |
| draft, hard rules only | 11 | | | | |
| **recalibrated** | **15** | **11** | **4** (s0610, s0777, s0960, s1786) | **15** | **0** |

- The recalibrated gate reproduces the verifier-corrected yield exactly, automatically, with no visual input.
- 12 of its 15 pass with a review flag:
  - 10 are "bridged > 2 mm", but only 5 have such a bridge **below the splice** (s0346 L, s0777 R, s1154 L,
    s1304 R, s1823 R); the rest lie above C1, in what a strategy-B graft discards;
  - 2 are label disagreements (s0610 L, s0777 L);
  - 1 is BCV (s1154 L).

  With the V3-loop gate in place none of the 6 wrong-vessel bridges passes. So "every bridged side to review" can
  shrink to bridges below the splice plus a 10% sample. That is my inference; it is not tested beyond these 40
  sides.
- Failure reasons, recalibrated (counts over the 40; one side can fail several):
  - V3 loop 21;
  - landmarks incomplete 12;
  - HU 11 (all on sides with no dura landmark);
  - C1 not threaded 9, length 9, total bridge 9;
  - V1 bridge 8;
  - V3-low bridge 7;
  - foramina 4, band 4, origin 2, V2 bridge 2.

## 5. Gross reference ranges (σ 3 mm; `results/tag_calibration_gross.json`)

| | length (mm) | DM | take-off 10 mm (fwd5) | C2-loop bend radius (mm) | V1 / V2 / V3 / V4 (mm) | subclavian elevation (fwd5) |
|---|---|---|---|---|---|---|
| host R / L | **210 / 209** | 1.39 / 1.42 | 92–93 / 76–78 (other tangent definitions: 81–93) | **none** (synthetic; min bend 4.24 / 5.39) | synthetic | 26.5 / 13.2 |
| VMR 0248 CT R / L | 219.9 / 225.0 | 1.40 / 1.46 | 53.5 / 58.7 | 1.44 / 1.54 | R 44.8 / **67.2** / 71.2 / 36.7; L 27.5 (31–40) / 87.5 (75–84) / 73.8 / 36.2 | **63.9** / 36.3 |
| SPL R / L | 221.1 / 193.2 | 1.59 / 1.41 | 106 (origin hook; report 113–134) / 43.8 | 3.02 / 3.25 (global minimum 2.19 at the origin hook / 1.71 at the V1 coil) | 44.5 / 109.8 / 53.0 / 39.5; 34.8 / 84.0 / 64.5 / 27.5 | 33.2 / 33.3 (new) |
| **TaG-Net A R / L** | 219.0 / **266.2** | 1.49 / **1.75** | 48.3 / 42.4 | 3.14 / 2.40 | 37.2 / 74.3 / 69.7 / 37.7; 41.7 / 86.5 / 82.3 / 55.7 (±3) | 41.0 / 44.5 |
| **TaG-Net B R / L** | 228.7 / 233.5 | 1.51 / 1.54 | 92.6 / 54.4 (assumed SCA r) | **1.37** / 2.59 (path-only window) | not measurable (no vertebrae) | 12.1 / 26.9 |
| TS whole VAs, corrected (n = 11) | 232.5 (206–258) | 1.48 (1.32–1.59) | 56 (42–86, n = 9) | whole-VA min bend 3.33 (1.92–4.67) | ts_qc landmarks biased (§2) | — |

- **C2-loop window.** For A, VMR and SPL the window is [C2 − 20, C2 + 15] mm. The path-only window (s 0.40–0.65 L)
  gives the same bend on A (both sides), VMR (both) and SPL L, so it was used for B. B's right bend (1.37 mm at s 126,
  z −575) is its C2 loop and fails G2, like VMR's.
- **The host is the outlier:**
  - it is shorter (209–210 mm) than every GT VA (219–266 mm) and sits at the bottom of the TS range (min 206 mm).
    Only SPL's left label (193 mm) is shorter;
  - it has no C2 loop at all, where every real VA here bends at 1.37–3.25 mm.

  Its take-off (76–93°) and subclavian elevation (13–27°) are inside the real ranges (42–93° and 12–64°).
- **VMR is the outlier:**
  - its right subclavian elevation is 63.9°, against 12–45° for the others;
  - its right V2 is the shortest, at 67 mm against 74–110;
  - its DM (1.40 / 1.46) is at the low end but inside the TS range.

  The report's finding that the host's flat subclavian, not the donor, causes the take-off loss was drawn from VMR
  alone. With A and B the real elevation spread is 12–64°, so donors near the host's 13–27° exist (B, SPL); VMR is the
  steep case.
- **A's left VA is the long, tortuous one:** 266 mm and DM 1.75 both exceed G4 (180–260, ≤ 1.65), and its V4 is
  55.7 mm. Its C2-loop bend (2.40 mm) also breaks G2 (≥ 2.5 mm), and **B's right VA breaks G2** (1.37 mm). With VMR
  (1.44 / 1.54), **4 of the 6 GT VAs fail G2 and 1 of 6 fails G4**: the G-gates reject real anatomy.

## Files

**Scripts** (all new, in `/sp/v5_va/`):

| script | image | what it does |
|---|---|---|
| `tag_calibration_run.sh` | | CPU docker runner, study mounts, no output filter |
| `tag_calibration_paths.py` | neve-build-stagea | the QC harness: ts_qc and v3loop unchanged, plus entry rules, V3 variants, band, HU, path-only proxy and stored per-level profiles. TS volumes are cropped around the paths (world-consistent affines; s0289 reproduced exactly; side time 25 s → 5 s), and progress is written incrementally |
| `tag_calibration_patch_crop.py` | | the one-off patch that added the crop and incremental save to my own harness |
| `tag_calibration_entry.py` | eve-training-fixed | entry-rule grid on the stored profiles |
| `tag_calibration_radius.py` | neve-build-stagea | radius |
| `tag_calibration_gross.py` | eve-training-fixed | gross ranges (imports `tag_labels_gross.takeoffs` and `va_metrics`) |
| `tag_calibration.py` | neve-build-stagea | assembly: items 1–5 |
| `tag_calibration_answers.py` | | the answer block |
| `tag_calibration_print.py` | | print helper |

**Results** (`/sp/v5_va/results/`):
- **`tag_calibration.json`**: `items` holds the full evidence and `answers` the per-item summary;
- `tag_calibration_paths_{VMR,A,B,TS}.json`, `tag_calibration_entry.json`, `tag_calibration_radius.json`,
  `tag_calibration_gross.json`;
- logs `tag_calibration*.log` and `tag_calibration_print.txt`.

**Data:** `D:\Arjun\neve\carotid_data\va_sources\derived\tagnet\calibration\` (gitignored; README inside).
- VMR TotalSegmentator masks: `vmr_totalseg_v3`, `vmr_totalseg` (117 masks each on the VMR CT grid; `ct.nii.gz` is a
  hard link) and their 1.5 mm copies.
- The 1.5 mm provenance json carries `tag_totalseg_make1p5.py`'s hard-coded TaG-Net note, which is wrong for these
  VMR copies.

**Diffs from the study tools:** none edited. The re-implemented definitions are:
- the robust cord: `tag_totalseg_qc.cord_fn` minus partial slices;
- `v3_zcap`: v3loop's measure with a different stretch end;
- the 8 mm enclosure loop in `level_profiles`: a verbatim copy of ts_qc's, extended to 5 and 4 mm rays;
- the HU statistics: `ts_screen.hu_stats`'s rule; the TS subjects use the stored ts_screen values;
- the V2 band: `ts_metrics2` / `ts_rank`'s rule.

**Cost:**
- TotalSegmentator on VMR: 107 s + 91 s GPU.
- Harness: VMR 2.5 min, A 8 min, TS 11 min after cropping (one uncropped run was killed at its last subject: OOM
  or contention, with no traceback).
- Radius 10 min.

## Open issues

1. **The C1 and C2 landmarks are still biased** on TotalSegmentator masks.
   - ts_qc's C2 is 7–12 mm late on both patients.
   - Its C1 is 26–40 mm late (the posterior-arch groove).
   - The first-C1-run exit is right on VMR but +9.5 / +38 mm on A.
   - The V3-low bridge rule (3c) and any C1 splice depend on them. A C1-exit rule needs a third GT or a CT bony-ring
     test.
2. **No C7, C5 or C4 entry among the GT VAs.**
   - Erec is validated only for "C6 is not over-called".
   - Its 3 TS C7 calls (s1075 R, s1138 R/L) are plausible but unverified.
   - The verifier's axial test cannot arbitrate: it over-calls too.
3. **The 1.5 mm radius factor is patient-dependent** (VMR × 1.05–1.08, A × 1.20–1.21). Two patients cannot say which
   is typical. Use CT half-max calibre instead.
4. **The V3 thresholds rest on 4 real GT loops and about 20 real TS loops.**
   - The margins are about 2 mm (excursion) and about 1 mm (d: 5.1 against 6).
   - The cord-top artefact is now handled, but the measure still depends on the cord mask.
   - A posterior-arch-only reference (no cord) would be more robust; it was not built.
5. **VMR's left landmarks** remain disputed (C6 s 27.5 vs ≈ 37); only its level (C6) was used.
6. **B is path-only.**
   - Its take-off rests on an assumed SCA radius (±2–4° per 0.5 mm).
   - Its V3 check is a proxy, valid only after tilt correction.
7. **A's 1.5 mm traces fail as whole or origin → dura donors.** An "origin → C2 donor" class with a C2 splice would
   keep them (V1–V2 ≤ 0.34 mm). Untested in the graft.
8. **Licence.** TaG-Net: internal only. Nothing derived from A or B may leave `derived/tagnet/`.
