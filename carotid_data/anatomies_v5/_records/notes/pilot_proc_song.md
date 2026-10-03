# pilot_proc_song: the whole v5 donor pipeline on the 10 Song et al. 2024 pilot cases

Agent prefix `pilot_proc_song`: first instance 21:01-23:38 EDT 2026-10-01 (cut off by a session limit; its detached loops
kept running), second instance 01:09-02:15 Oct 2 (took over post loop 2, ran the closing pass, finished). State file:
`/sp/v5_va/state/pilot_proc_song.md`. No git; no tracked file touched. Data under
`carotid_data/va_sources/derived/thin_slice/song/<case>/` (gitignored). Licence: Song 2024 CC BY 4.0 (figshare 26965450);
everything through Arterial is non-commercial.

## Answer so far

**Final (02:10 EDT Oct 2, second instance).** All 10 Song cases / 20 VAs went through the whole pipeline: TotalSegmentator ->
Arterial on the `air1000` re-window -> trace (tsba + ts) -> QC -> ct_seg4 wall -> gross -> contamination -> grade -> figures.
GPU chain 4 ran all 9 remaining Arterial cases (rc 0, done 00:17); post loop 2 graded every case by 01:12; the closing
pass (`pilot_proc_song_final.sh`) finished 02:00. One record per case and side: `results/pilot_proc_song.json`. Grade and
ranking: `results/pilot_proc_song_grade.json/.txt`. Tables: `results/pilot_proc_song_summary.json/.txt`.

1. **QC under both gates.**
   - Recalibrated gate (primary): **15 / 20 sides (75 %), 6 pairs** (008 032 094 100 109 113).
   - Study draft + additions as written: **5 / 20, 1 pair** (032).
   - Draft with only the two-patient-backed changes (rule 6 -> review, V3 floor 12, Erec): **13 / 20, 4 pairs**
     (008 032 094 113).
   - The gates differ on 10 sides:
     - 6 only through rule 6 (Arterial label agreement): 008 L, 044 L, 094 L, 100 L, 113 R/L;
     - 008 R and 106 L also through the study V3 floor (12.8 and 14.3 mm; both pass the robust 12 / 6);
     - 100 R through the V2-band IQR term;
     - 109 R through a V1 bridge of <= 3 mm.
     The last two are in-sample relaxations.
   - Fails (both gates):
     - 090 R/L: wrong course (V3 loop 6.4 / 13.2 and 0.6 / 20.9; 31 mm bridge).
     - 044 R and 141 R: V1 bridges through enhanced lumen (480 / 587 HU), i.e. Arterial mask breaks. Rescuable on review.
     - 106 R: V1 bridges through 11 / 92 HU (not lumen).
2. **Arterial input = `ct_negzero_air1000`, kept.** The shipped 0-HU-clipped CT loses the neck vessels in Arterial's
   extracranial model on 2 of 2 cases. The CCA is found on 0.065 / 0.000 of C6-C2 slices on 100 and 0.28 / 0.33 on 109;
   `ct_bg-100` and `negzero` fail the same way. Air at -1000 outside the body restores it (0.72 / 0.62). Evidence is in
   the section below.
3. **Clipped vs re-windowed CT, on 4 cases / 7 VAs (100 R, 113 R/L, 008 R/L, 032 R/L).**
   - The per-ray half-max edge moves by a median of 0.0015-0.0076 mm; 0.5-4.2 % of rays move > 0.1 mm (13.9 % on 032 R,
     the smallest lumen, r 1.29 mm).
   - On re-windowing, union whole rises by 0.002-0.016 and the edge FWHM by 0.02-0.08 mm.
   - So the clipped CT is usable for the wall, but it slightly understates the detail of small, smooth VAs (032 L 0.065
     -> 0.081).
4. **Wall coarseness: the set looks coarse, but most of that is contact relief.**
   - Set medians on the 15 walls: whole VA 0.086 / 0.053 / 0.130 (W1 fail, W2 pass, W3 fail); donor stretch (origin ->
     C1f - 25, seam separate) 0.104 / rel 0.156 (W1 and W3 pass); C6 -> C1 (ct) 0.118 / 0.147 (pass).
   - But 76 % of the detail area lies in contact bins (bone of the foramina, abutting veins). Union is 0.217 in contact
     bins against 0.046 in the others.
   - Contact-free donor stretch: median 0.062 (0.009-0.134).
   - All walls beat the host tube (0.0007) and the dense host RVA (0.027) on all bins. On contact-free bins, 4 sides do
     not beat 0.027: 008 R 0.010, 109 L 0.009, 141 L 0.019, 032 R 0.022.
5. **Within-Song pair ranking.** The user's rule: QC-passing pairs only, contact relief excluded, the same definition as
   the 172-series grade. Score = mean over R and L of the contact-free donor-stretch union > 0.3 mm.

   | rank | pair | score (min side) | note |
   |---|---|---|---|
   | 1 | 113 | 0.104 (0.076) | |
   | 2 | 100 | 0.100 (0.072) | ts traces |
   | 3 | 094 | 0.073 (0.062) | R wall truncated at C2, partial; L V1 vein-contaminated, excluded by the rule |
   | 4 | 008 | 0.044 (0.010) | ts traces; 6-15 % contact-free |
   | 5 | 032 | 0.036 (0.022) | the only pair that passes the draft too |
   | 6 | 109 | 0.022 (0.009) | |

   - Single donor sides: 106 L 0.134 (partial wall), 044 L 0.040, 141 L 0.019.
   - For context, the 172-series pairs read 058 0.174, 039 0.092, 015 0.084, 190 0.043, 141 0.022.
6. **Problems found**, all recorded and none silently patched:
   - ct_seg4 truncates 2 of 15 walls at a tight C2-exit bend.
   - TotalSegmentator bone masks fill the transverse foramina, so they cannot flag bone contamination.
   - 094 L's V1 abuts a dense, contrast-filled vein (checked by eye).
   - Diff 8 matters (032 R plain ct_seg4 0.032 vs 0.064).
   - W7 by hand fails on all 15 walls. This does not matter for strategy B, which drops the donor above the V3 splice.
   - The previous instance left `pilot_proc_song_fig.py` with a SyntaxError, so no figure was made 23:32-01:12. Fixed;
     all figures redrawn.
7. **Gross (donor sides).** Length 192-253 mm (median 231), DM 1.31-1.60, C2 bend 1.8-5.5 mm, take-off 26-131 deg,
   subclavian elevation -7-75 deg, entry (Erec) C6 11 / C7 4. L/R calibre 0.81-1.38.

## Method (per case; `pilot_proc_song_run.sh`, idempotent, per-case outputs and done-markers)

The pipeline is the 172-series pilot agent's (`pilot_proc_ia172_run2.sh` and its scripts), copied to `pilot_proc_song_*`
with the Song differences below, so the two sources are processed alike.

1. **TotalSegmentator 2.18.0 `total_v3`** (GPU, `--ml`, split on the CPU) on the CTA grid. Input = `ct_negzero-100`
   (the fetch agent's side file: -100 HU on every originally negative voxel) as an exact int16 copy, because the shipped
   image stores air, fat and soft tissue all as 0 HU. Checked against TotalSegmentator on the shipped image on
   AHMU1218100 (see "TS input"). `total_v3` rather than "total": tag_totalseg (C1 Dice 0.98 vs 0.92) and the
   recalibrated QC gate were set on total_v3 masks.
2. **Arterial** (`run_arterial.sh` unchanged, GPU, `--no-tta`, core stages, native grid), crop / FOV / TS skull from
   `arterial_ts_args.py` (unchanged) + the two-limb check (`pilot_proc_song_args.py`). The brief said: the SHIPPED
   `ct.nii.gz` first, `ct_bg-100` if it fails. Both lose the neck vessels (see "Arterial input" below), so from 22:56 the
   input for every case is **`ct_negzero_air1000`** (decision and evidence below). GPU chain 4 (`pilot_proc_song_gpuchain4.sh`)
   ran all 9 remaining cases on it, rc 0 each, 186-620 s per case, finished 00:17.
3. **Trace**: `pilot_art_extract.py` (art_extract + the basilar-seed fix, shared pilot copy) `--ts totalseg_1p5` ->
   `arterial/tsba` (protocol trace); unchanged art_extract -> `arterial/ts`. VMTK MISR on the cut-out stage-A surfaces.
4. **QC** (`pilot_proc_song_qc2.py` = ia172's qc2): recalibrated gate (tag_calibration.md recommended values) + V3 loop,
   C1 threading, origin, foramina, length, HU/phase; added: dense-venous review (>= 1.6 x aorta along origin -> C1).
   **Both trace sets are graded by the same gate and the better one is kept per side** (`qc/select.json`; rank = donor,
   whole VA, tsba; tsba wins ties). The ts trace was kept on 4 sides ("route rescues"; both verdicts are in each record:
   `tsba_protocol`, `ts_unchanged_tool`): on 100 L and 008 L the tsba trace FAILS (100 L: a 21.8 mm bridge low in V3 and
   the V3 loop 2.8 / 13.2; 008 L: V3 loop, C1 not threaded, HU) and ts passes; on 100 R and 008 R both pass as donors but
   only ts reaches the whole VA (tsba: V4 69 mm, traces meet only at the basilar end point).
5. **Wall**: ct_seg4 (`pilot_proc_song_ct_seg.py` = ia172's, incl. its diffs 7 (NaN seeds) and 8 (connectivity-preserving
   opening)) seeded from the selected trace's VMTK line, on the SHIPPED CT (the re-windowed file is the Arterial input
   only); VMTK MISR on the stage-A surface; `va_metrics` unchanged via `pilot_proc_song_measure.py`: exact ownership
   (k 64), acquisition voxel, native bars, landmark sets ts / ct / rec, seam zone C1 - 25 mm -> C1 separate; edge FWHM
   (`pilot_proc_song_edge.py`). Plain ct_seg4 without diff 8 kept as `ct_seg4_nofix` where diff 8 acted (032 R/L,
   094 R/L, 106 L, 113 R).
6. **Gross** shape on the Arterial VMTK lines (sigma 3; `pilot_proc_song_gross.py`).
7. **Contamination**: (a) the edge rays' contact rule (10 mm bins with >= 5 % of rays meeting a brighter neighbour or a
   dip before one; classes by the TS masks); (b) `pilot_proc_song_protrusion.py` (ia172's protrusion + LAS read +
   classes): label voxels > 0.75 mm beyond their station's median edge, by normalised HU and by TS carotid / vein.
8. **Grade and ranking** (`pilot_proc_song_grade.py` = ia172's grade script, paths only + trace set / truncation /
   protrusion fields) and tables (`pilot_proc_song_summary.py` = ia172's summary + Song sections). Figures
   `pilot_proc_song_fig.py`.

## TS input: negzero, not the shipped image (AHMU1218100, `results/pilot_proc_song_tscmp_AHMU1218100.json`)

TotalSegmentator total_v3 was run on the same case twice: on `ct_negzero-100` (primary) and on the shipped 0-clipped
image (`totalseg_shipped`, comparison only).

| class | Dice | note |
|---|---|---|
| vertebrae C1-C7, T1 | 0.985-0.995 | z ranges equal within 1.4 mm (C4 / C6 bottoms) |
| skull | 0.979 | |
| aorta, BT, subclavians, CCAs | 0.97-0.996 | centroid shifts 0.1-1.7 mm |
| **trachea** | **0.00** | the shipped run finds **no trachea** (air = soft tissue = 0 HU); `arterial_ts_args` centres the FOV on it |
| **spinal cord** | **0.90** | shipped: centroid 7.4 mm off, top 4.5 mm lower; the cord is art_extract's midline and the V3-loop reference |
| left BCV, oesophagus, thyroid | 0.32, 0.39, 0.86 | soft tissue lost |

So bone and contrast-filled vessels survive the clipping, but the soft-tissue classes the tools use (trachea, cord) do
not. The negzero input restores them. Cost: 90 s GPU (negzero) against 255 s (shipped).

## Arterial input: the 0-HU clipping breaks Arterial's NECK model; a re-windowed copy fixes it

Neck coverage of Arterial's binary mask, C6-C2 slab (`pilot_proc_song_artcheck.py`; per slice, 2-D components >= 3 mm^2;
"CCA" = a component within 6 mm of the TS CCA centroid, "VA band" = a component 10-30 mm lateral / -5..+25 mm anterior
of the TS cord). Look PNGs: `derived/thin_slice/song/<case>/qc/look_native*.png`.

| case | Arterial input | CCA slices R / L | VA-band slices R / L | slab mask ml | verdict |
|---|---|---|---|---|---|
| AHMU1218100 | shipped (0-clipped) | 0.065 / 0.000 | 1.00 / 0.76 | 3.5 | **fails**: no CCA, VA fragments only below the mandible; both traces fail QC |
| AHMU1218109 | shipped (0-clipped) | 0.28 / 0.33 | 0.74 / 0.85 | 4.9 | **fails** the same way (Siemens 0.7 mm) |
| AHMU1218100 | `ct_bg-100` (brief's side file) | 0.072 / 0.022 | 1.00 / 0.82 | 3.4 | **fails** |
| AHMU1218100 | `ct_negzero-100` | 0.21 / 0.48 | 1.00 / 0.86 | 3.9 | **fails** |
| AHMU1218100 | `ct_negzero_air1000` (negzero + outside-body -1000) | 0.72 / 0.62 | 1.00 / 1.00 | 6.3 | **works** |
| AHMU1218100 | `ct_tex` (synthetic 40 / -100 HU + noise, air -1000) | 0.72 / 0.62 | 1.00 / 1.00 | 6.1 | works, same as air1000 |

- The head model (above the mandible) segments normally on every input; only the extracranial (neck) model fails.
  It z-scores the whole image (nnU-Net ZScoreNormalization, no mask), and on the clipped image 59-75 % of the voxels
  (the air outside the body) are 0 or -100 instead of -1000.
- **Decision (22:56): Arterial input = `ct_negzero_air1000` for every Song case** (`pilot_proc_song_rewindow.py`: the fetch
  agent's negzero file with every voxel outside its body mask set to -1000). It is a re-windowing of the shipped data, not
  a recovery of the lost HU. It is the ARTERIAL INPUT ONLY: QC HU, ct_seg4 wall and edge FWHM read the shipped image.
- Cost of the tests: 6 extra Arterial runs (~25 min GPU). Kept: `arterial/runs/native_{shipped_failed,bg,negzero,tex}`
  on 100, `native_shipped_failed` on 109.

## What the high relief is made of (contamination)

The user's ranking excludes relief from vein / carotid / bone contamination. Two measures, both on the ct_seg4 wall:

- **Edge-ray contacts** (`edge.json`, the 172-series rule): a 10 mm bin is a contact bin when >= 5 % of its half-max edge
  rays meet a brighter neighbour (kind 3) or a dip before one (kind 1). On Song the contact rays are almost all classed
  "bone (TS mask <= 1 mm)" (V2 in the transverse foramina; peak HU 450-3000) plus "unlabelled". The walls with the highest
  whole-VA relief are mostly contact relief: contact-free area share 008 R 0.06, 008 L 0.15, 094 L 0.12, against 0.85-0.88
  on 100. On contact-free bins 008 R falls from 0.244 to 0.010 and 094 L from 0.396 to 0.083 (donor stretch).
- **094 L V1 is vein contamination, not wall.** Its V1 reads union 0.62 (rec landmarks). The QC flagged 10.4 % of the
  origin -> C1 trace points >= 700 HU, and the contact rays there meet "unlabelled" neighbours of 1023-2932 HU: most likely undiluted
  contrast in an adjacent vein on the injection side (TotalSegmentator total_v3 has no subclavian, jugular or vertebral
  vein class, so the class stays "unlabelled"). CONFIRMED BY EYE (`derived/.../AHMU1218094/qc/lookv1_AHMU1218094_L.png`,
  `pilot_proc_song_lookv1.py`): from s 5 to 25 mm the VA runs against a round ~5 mm structure of >= 1500 HU (arterial
  lumen ~500 HU), behind the CCA and outside every TS class -- most likely the left vertebral vein full of undiluted
  contrast. The contact rule removes those bins (s 0-60 are all contact bins); the V1 course itself (the trace runs at
  the vein's edge) deserves a look before 094 L is used.
- **008 R** (`qc/lookv1_AHMU1218008_R.png`): in V1 (s 5-25 mm) the VA runs beside a second round vessel of similar
  brightness (paired lumina; most likely the enhanced vertebral vein), in V2 / V3 inside the foramen cortex. Its contact
  bins (all but 6 % of the area) are such merged neighbours and bone edges, so the contact rule rightly removes them.
- **TotalSegmentator bone cannot flag bone contamination of a VA.** The total_v3 vertebra masks FILL the transverse
  foramina: 66-71 % of 094's VA label voxels lie inside them and 75-87 % within 1 mm. The protrusion script therefore
  counts bone only through the over-bright share (h > 1.3) and reports the TS-bone shares as context.
- **Protrusion** (label voxels > 0.75 mm beyond their station's median edge) is small on every Song wall (0.2-4 % of the
  label), and most of it is lumen-bright (h >= 0.75: 47-97 %), as on the 172-series. So the union detail is mostly
  shape (ovality, wander, bends), not outward spurs.

## ct_seg4 fails at a tight C2-exit bend (2 of 15 walls truncated)

- **094 R**: the label stops at s 122.75 of 241 mm. The selected trace's VMTK line turns 45-63 deg per 2 mm at the C2 exit
  (gross C2 bend radius 1.87 mm, near the lumen radius 1.7 mm). Over s 122.25-124 the station centre leaves the CT lumen.
  The ray edges then sit at their 0.35 mm floor on all 64 rays (centreline HU still 460-480 nearby), and the label is cut.
  Keep-largest drops the 118 mm above (7635 voxels). Fix 8 cannot help: the cut is in the un-opened label.
- **106 L**: the same at s 164.5-165 (turn 60-70 deg per 2 mm, r_e at the 0.4 mm floor); the 78 mm above is lost.
- Both walls are graded on what exists (origin -> C2 bend, inside the donor stretch) and flagged `label_truncation`
  (collect) / `partial_truncated_wall` (grade). Their donor-stretch numbers are partial. The C2 loop is the tightest bend
  of most real VAs (report: 1.37-3.25 mm), so at scale this will recur. Candidate fix (not applied, needs both sources):
  re-centre stations whose centre HU < T_loc onto the nearest lumen-bright voxel before the ray pass, or bridge the
  pinch through the Arterial mask.

## Per-side results (generated by `pilot_proc_song_tables.py` from the result files, 02:05)

### QC per side (recalibrated = primary; draft = study draft + additions A1/A2; backed = two-patient-backed only)
| case | side | voxel mm | trace set | recalibrated | draft | backed | Erec (ts_qc) | V3 robust exc/d | V3 study | origin mm | HU/CCA | length mm | why (fail / draft-only / review) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 008 | R | 0.45x0.45x0.62 | ts | donor_review | fail | pass | C7 (C7) | 13.4 / 0.6 | 12.8 / 0.6 | 0.0 | 0.992 | 223.9 | draft: 6_label_agreement_ge_0.9; A1_v3loop_study_15_5 (12.8/0.6) |
| 008 | L | 0.45x0.45x0.62 | ts | donor_review | fail | pass | C6 (C6) | 16.5 / 0.4 | 15.9 / 0.4 | 0.0 | 1.052 | 231.7 | draft: 6_label_agreement_ge_0.9 |
| 032 | R | 0.44x0.44x0.62 | tsba | donor | pass | pass | C6 (C7) | 16.3 / 0.4 | 19.5 / 0.4 | 0.0 | 0.999 | 214.3 |  |
| 032 | L | 0.44x0.44x0.62 | tsba | donor | pass | pass | C6 (C6) | 16.5 / 3.8 | 24.1 / 0.4 | 0.0 | 1.046 | 218.1 |  |
| 044 | R | 0.49x0.49x0.45 | tsba | fail | fail | fail | C6 (C7) | 22.1 / 0.1 | 22.0 / 0.1 | 3.25 | 0.927 | 270.1 | FAIL: 1_origin_le_2mm; 3a_no_V1_bridge |
| 044 | L | 0.49x0.49x0.45 | tsba | donor_review | fail | pass | C7 (C7) | 22.2 / 0.1 | 29.1 / 0.1 | 0.0 | 0.969 | 243.1 | draft: 6_label_agreement_ge_0.9 |
| 090 | R | 0.49x0.49x0.45 | tsba | fail | fail | fail | C6 (C6) | 6.4 / 13.2 | 6.4 / 13.2 | 6.12 | 0.953 | 288.9 | FAIL: 1_origin_le_2mm; 3a_no_V1_bridge; 3d_total_le_25; 5b_HU_ge_0.6_CCA |
| 090 | L | 0.49x0.49x0.45 | tsba | fail | fail | fail | C6 (C7) | 0.6 / 20.9 | 0.6 / 20.9 | 0.14 | 1.008 | 230.1 | FAIL: 3a_no_V1_bridge; 5b_HU_ge_0.6_CCA; 7_v3loop_robust; 8_C1_threaded |
| 094 | R | 0.40x0.40x0.70 | tsba | donor | pass | pass | C7 (C7) | 20.5 / 0.4 | 20.5 / 0.4 | 0.0 | 0.933 | 229.4 |  |
| 094 | L | 0.40x0.40x0.70 | tsba | donor_review | fail | pass | C7 (C7) | 19.7 / 0.2 | 19.7 / 0.2 | 0.0 | 0.909 | 214.2 | draft: 6_label_agreement_ge_0.9 / review: bridged (max 2.4 mm); 10.4 % of origin->C1 points >= 700 HU (native); bridge(s) > 2 mm below the splice: 2.4 m |
| 100 | R | 0.49x0.49x0.45 | ts | donor_review | fail | fail | C6 (C7) | 19.6 / 0.3 | 20.5 / 0.3 | 0.0 | 0.983 | 220.6 | draft: 5a_V2_band; 6_label_agreement_ge_0.9 / backed: 5a V2 band with IQR (draft) / review: bridged (max 2.4 mm) |
| 100 | L | 0.49x0.49x0.45 | ts | donor_review | fail | pass | C6 (C7) | 17.2 / 0.2 | 17.3 / 0.2 | 0.0 | 1.007 | 218.0 | draft: 6_label_agreement_ge_0.9 / review: bridged (max 3.5 mm) |
| 106 | R | 0.43x0.43x0.62 | tsba | fail | fail | fail | C6 (C7) | 16.8 / 0.6 | 16.7 / 0.6 | 0.0 | 1.007 | 235.9 | FAIL: 3a_no_V1_bridge |
| 106 | L | 0.43x0.43x0.62 | tsba | donor_review | fail | pass | C6 (C7) | 14.3 / 3.5 | 14.3 / 3.5 | 0.0 | 1.018 | 228.6 | draft: 6_label_agreement_ge_0.9; A1_v3loop_study_15_5 (14.3/3.5) |
| 109 | R | 0.57x0.57x0.70 | tsba | donor | fail | fail | C6 (C6) | 20.2 / 0.4 | 21.9 / 0.4 | 0.0 | 0.973 | 186.7 | draft: 3a_no_V1_bridge / backed: 3a any V1 bridge (draft) |
| 109 | L | 0.57x0.57x0.70 | tsba | donor | pass | pass | C6 (C7) | 19.1 / 0.3 | 21.3 / 0.3 | 0.0 | 0.994 | 194.3 |  |
| 113 | R | 0.49x0.49x0.70 | tsba | donor_review | fail | pass | C6 (C6) | 20.0 / 0.3 | 20.6 / 0.3 | 0.0 | 1.025 | 228.6 | draft: 6_label_agreement_ge_0.9 |
| 113 | L | 0.49x0.49x0.70 | tsba | donor_review | fail | pass | C6 (C7) | 18.0 / 0.6 | 19.1 / 0.6 | 0.0 | 1.042 | 224.4 | draft: 6_label_agreement_ge_0.9 |
| 141 | R | 0.49x0.49x0.45 | tsba | fail | fail | fail | C6 (C7) | 17.4 / 0.4 | 19.6 / 0.4 | 0.0 | 0.985 | 211.7 | FAIL: 3a_no_V1_bridge |
| 141 | L | 0.49x0.49x0.45 | tsba | donor | pass | pass | C6 (C7) | 25.0 / 0.2 | 25.0 / 0.2 | 0.0 | 0.985 | 222.4 |  |
### Wall per side (ct_seg4 on the shipped CT; exact ownership k64; acquisition voxel; union > 0.3 / > 0.5 mm / > 0.15 r; W3 read with W1)
| case | side | whole .3/.5/rel | donor stretch origin->C1f-25 .3/.5/rel | seam C1-25->C1 .3/rel | contact-free .3 (area share) | C6->C1 ct .3 | protrusion % (contaminated) | FWHM mm | r MISR mm | W7 0-40/40-80 | gates | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 008 | R | 0.1762/0.1191/0.2308 | 0.2443/0.1681/0.301 | 0.0924/0.1403 | 0.0097 (0.059) | 0.1981 | 1.0 (0.219) | 1.21 | 1.427 | 0.0002/0.0745 FAIL | pass | ts trace |
| 008 | L | 0.2548/0.1743/0.2556 | 0.2876/0.1973/0.2893 | 0.2191/0.1857 | 0.0791 (0.145) | 0.2881 | 0.3 (0.381) | 1.23 | 2.07 | 0.0117/0.2725 FAIL | pass | ts trace |
| 032 | R | 0.0638/0.038/0.1233 | 0.0458/0.0237/0.1028 | 0.1452/0.2123 | 0.0215 (0.477) | 0.0853 | 0.8 (0.247) | 1.191 | 1.222 | 0.0299/0.0794 FAIL | pass | plain ct_seg4 whole 0.0324/0.0144/0.0802 |
| 032 | L | 0.0652/0.0366/0.1301 | 0.0763/0.0364/0.156 | 0.1644/0.1907 | 0.0499 (0.557) | 0.0745 | 0.6 (0.394) | 1.235 | 1.269 | 0.0121/0.0189 FAIL | pass | plain ct_seg4 whole 0.0515/0.035/0.1 |
| 044 | L | 0.054/0.0184/0.0963 | 0.0592/0.0244/0.0999 | 0.0291/0.0634 | 0.0404 (0.594) | 0.0625 | 1.1 (0.152) | 1.269 | 1.516 | 0.0845/0.0266 FAIL | pass |  |
| 094 | R | 0.143/0.1014/0.1678 | None/None/None | None/None | 0.062 (0.582) | None | 2.7 (0.164) | 0.951 | 1.644 | 0.2752/0.0811 FAIL | pass | label truncated at s 122.75 of 241.0 mm (C2-bend pinch); plain ct_seg4 whole 0.1446/0.1021/0.1698 |
| 094 | L | 0.2575/0.1844/0.3122 | 0.3964/0.3069/0.4642 | 0.2113/0.2712 | 0.0832 (0.117) | 0.3526 | 4.0 (0.228) | 1.037 | 1.518 | 0.0136/0.0184 FAIL | pass | plain ct_seg4 whole 0.3547/0.2678/0.4191 |
| 100 | R | 0.0862/0.0291/0.117 | 0.1146/0.0348/0.158 | 0.1309/0.1576 | 0.1294 (0.845) | 0.1027 | 0.2 (0.529) | 1.192 | 1.719 | 0.0201/0.0272 FAIL | pass | ts trace |
| 100 | L | 0.0607/0.0253/0.0986 | 0.0726/0.0311/0.1183 | 0.0624/0.0976 | 0.0715 (0.878) | 0.0827 | 0.9 (0.363) | 1.204 | 1.579 | 0.0351/0.0369 FAIL | pass | ts trace |
| 106 | L | 0.2199/0.1257/0.2678 | None/None/None | None/None | 0.1339 (0.383) | None | 2.1 (0.069) | 1.196 | 1.719 | 0.3042/0.1965 FAIL | pass | label truncated at s 164.25 of 242.25 mm (C2-bend pinch); plain ct_seg4 whole 0.2199/0.1257/0.268 |
| 109 | R | 0.0447/0.0263/0.0865 | 0.065/0.0437/0.0944 | 0.0076/0.0479 | 0.0349 (0.343) | 0.0475 | 0.3 (0.375) | 1.072 | 1.282 | 0.0148/0.0035 FAIL | pass |  |
| 109 | L | 0.0771/0.0458/0.0908 | 0.1035/0.0643/0.118 | 0.0033/0.002 | 0.0094 (0.278) | 0.1179 | 0.3 (0.258) | 1.026 | 1.86 | 0.0207/0.139 FAIL | pass |  |
| 113 | R | 0.1561/0.076/0.1842 | 0.1944/0.1076/0.2281 | 0.1685/0.196 | 0.0756 (0.433) | 0.2576 | 0.9 (0.284) | 1.347 | 1.791 | 0.057/0.0376 FAIL | pass | plain ct_seg4 whole 0.1353/0.0538/0.1648 |
| 113 | L | 0.162/0.0811/0.1945 | 0.2322/0.1191/0.2683 | 0.1262/0.2017 | 0.1327 (0.309) | 0.2315 | 0.2 (0.308) | 1.311 | 1.777 | 0.0383/0.0444 FAIL | pass | plain ct_seg4 whole 0.162/0.0811/0.1945 |
| 141 | L | 0.0735/0.0525/0.0889 | 0.0952/0.0693/0.1097 | 0.1138/0.159 | 0.0194 (0.605) | 0.1356 | 0.4 (0.516) | 1.155 | 1.74 | 0.0249/0.0082 FAIL | pass |  |
### Within-Song pair ranking (recalibrated donor pairs with walls; contact-free donor stretch, union > 0.3 mm)
| rank | case | mean | min | R / L contact-free | R / L all bins | status R / L | both pass draft | note |
|---|---|---|---|---|---|---|---|---|
| 1 | 113 | 0.1042 | 0.0756 | 0.0756 / 0.1327 | 0.1944 / 0.2322 | donor_review / donor_review | False |  |
| 2 | 100 | 0.1004 | 0.0715 | 0.1294 / 0.0715 | 0.1146 / 0.0726 | donor_review / donor_review | False | ts trace R, L |
| 3 | 094 | 0.0726 | 0.062 | 0.062 / 0.0832 | None / 0.3964 | donor / donor_review | False | truncated wall: R |
| 4 | 008 | 0.0444 | 0.0097 | 0.0097 / 0.0791 | 0.2443 / 0.2876 | donor_review / donor_review | False | ts trace R, L |
| 5 | 032 | 0.0357 | 0.0215 | 0.0215 / 0.0499 | 0.0458 / 0.0763 | donor / donor | True |  |
| 6 | 109 | 0.0221 | 0.0094 | 0.0349 / 0.0094 | 0.065 / 0.1035 | donor / donor | False |  |
### Clipped (shipped) vs re-windowed (negzero) CT: same seeds and code
| case | side | rays | edge shift median mm | p5 / p95 | share abs > 0.1 / > 0.2 mm | I_bg HU | T_loc HU | union whole shipped -> negzero | FWHM verifier mm |
|---|---|---|---|---|---|---|---|---|---|
| 100 | R | 51520 | 0.0029 | -0.0031 / 0.0751 | 0.016 / 0.0 | 9.5 -> 7.4 | 331.7 -> 329.3 | [0.094, 0.0305, 0.1277] -> [0.097, 0.0324, 0.1282] | 1.191 -> 1.266 |
| 113 | R | 61568 | 0.0015 | -0.007 / 0.0371 | 0.005 / 0.002 | 53.1 -> 52.1 | 269.9 -> 269.6 | [0.1561, 0.076, 0.1842] -> [0.1583, 0.0744, 0.1865] | 1.347 -> 1.386 |
| 113 | L | 61120 | 0.002 | -0.0148 / 0.0323 | 0.007 / 0.003 | 75.1 -> 75.1 | 282.8 -> 282.7 | [0.162, 0.0811, 0.1945] -> [0.1646, 0.0843, 0.1986] | 1.311 -> 1.342 |
| 008 | R | 59712 | 0.0015 | -0.0128 / 0.0648 | 0.021 / 0.004 | 64.3 -> 64.2 | 235.6 -> 235.4 | [0.1762, 0.1191, 0.2308] -> [0.1844, 0.1227, 0.2396] | 1.21 -> 1.242 |
| 008 | L | 63616 | 0.0032 | -0.0367 / 0.0678 | 0.042 / 0.019 | 61.8 -> 61.3 | 243.9 -> 243.6 | [0.2548, 0.1743, 0.2556] -> [0.2619, 0.1823, 0.2624] | 1.23 -> 1.25 |
| 032 | R | 56704 | 0.0076 | -0.017 / 0.1684 | 0.139 / 0.028 | 25.0 -> 23.1 | 202.7 -> 201.1 | [0.0638, 0.038, 0.1233] -> [0.0693, 0.0398, 0.125] | 1.191 -> 1.234 |
| 032 | L | 58368 | 0.0059 | -0.0153 / 0.0715 | 0.027 / 0.003 | 31.6 -> 30.9 | 211.0 -> 209.9 | [0.0652, 0.0366, 0.1301] -> [0.0809, 0.0413, 0.1481] | 1.235 -> 1.275 |

Reading the tables:
- "backed" = `pilot_proc_song_grade.py` `qc_two_patient_backed_only` (ia172's definition). It keeps the recalibrated gate's
  backed changes (rule 6 -> review, V3 floor 12 on the study cord with d <= 5, Erec) and reverts the in-sample limits
  (any V1 bridge fails, V2 band with the IQR term, 25 mm total over all bridges).
- Contact-free = the donor stretch's 10 mm bins with < 5 % bright-neighbour rays. The area share in brackets is how much
  of the stretch that is. A low share (008 R 0.06, 094 L 0.12, 008 L 0.15) means the score rests on few bins.
- 094 R and 106 L walls are truncated at the C2 bend. Their contact-free value is for the wall that exists (origin -> C2
  bend, inside the donor stretch) and their W7 is meaningless.
- W7 (by hand) fails everywhere. It applies to whole-VA donors only, and strategy B discards the donor above the V3 splice.
- The plain ct_seg4 (no diff 8) differs from ct_seg4 where diff 8 acted: 032 R 0.032 vs 0.064, 032 L 0.052 vs 0.065,
  113 R 0.135 vs 0.156. On 094 L plain is HIGHER (0.355 vs 0.258) because without diff 8 the label loses s 107-226 and
  keeps only the vein-contaminated lower part.

## Clipped vs re-windowed CT (is the 0-HU clipping a problem for the wall?)

ct_seg4 + VMTK + measure + edge were run twice per case with identical seeds and code: once on the shipped CT (every
voxel <= 50 HU stored as 0) and once on `ct_negzero-100` (-100 HU on every originally negative voxel). The comparison
script is `pilot_proc_song_clipcmp.py`, the results are in `results/pilot_proc_song_clipcmp.json`, and the per-case
directories are `ct_seg/ct_seg4_negzero` (100: `ct_seg4_negzero_tsbaR` vs `ct_seg4_tsbaR_clipref`, made on its first
tsba R trace). Four cases from three scanner families were run: 100 (Philips), 113 (Siemens), 008 and 032 (GE).
- **The half-max edge does not move:** the median ray shift is 0.0015-0.0076 mm and p95 <= 0.075 mm on 6 of 7 VAs.
  The exception is 032 R, the smallest lumen (r 1.29 mm): p95 0.17 mm, 13.9 % of rays > 0.1 mm, median r_e +0.025 mm.
- **Background and threshold barely change.** I_bg drops by 0-2 HU (7.4-75 HU) and T_loc by 0-2.4 HU, because the
  clipping affects only voxels <= 50 HU and the soft-tissue background is mostly above 0.
- **Detail and FWHM rise slightly when re-windowed:** union whole +0.002 to +0.016 (+1 % to +24 %; largest on 032 L,
  0.065 -> 0.081), edge FWHM +0.02 to +0.08 mm. The clipped image gives a marginally sharper-looking edge and marginally
  less relief.
- Verdict: the shipped CT is fine for the wall. Detail on the smallest, smoothest VAs is slightly understated, by less
  than the gaps between ranks except near the bottom of the ranking. QC HU, ct_seg4 and edge all use the shipped CT,
  as decided at 22:56.

## Open issues

1. **ct_seg4 C2-bend pinch** (094 R, 106 L): a method fix is needed before scale-up and should be applied to both
   sources. Until then, flag truncated walls (collect `wall.label_truncation`).
2. **094 L V1**: the trace runs at the edge of a dense, contrast-filled vein for about 20 mm. Look at it before using
   094 L as a donor. Its relief there is excluded from the ranking already.
3. **Rescues not attempted**: 044 R and 141 R fail only on V1 bridges through enhanced lumen (Arterial mask breaks).
   A no-label retrace or a manual check could turn them, and so 141 and 044, into pairs.
4. **Contact rule coverage**: on 008, 094 L and 109 L the contact-free score rests on 6-28 % of the stretch, so those
   values are noisy. The rule also treats all bone contact in V2 as contamination, which is conservative: real foramen
   wall shape is excluded too.
5. **Gate thresholds**: the recalibrated limits are in-sample. Song is a second out-of-sample check, and it agrees with
   the 172-series pattern: rule 6 is the main difference and the rest are single sides.
6. Licence: Song 2024 is CC BY 4.0, but everything through Arterial is non-commercial.

## Files

- Results: `results/pilot_proc_song.json` (one record per case and side: QC under both gates, trace set, wall,
  contamination, gross), `pilot_proc_song_grade.json/.txt` (grade + within-Song ranking), `pilot_proc_song_summary.json/.txt`,
  `pilot_proc_song_clipcmp.json`, `pilot_proc_song_tscmp_AHMU1218100.json`, `pilot_proc_song_artcheck_<case>.json`.
- Per case (`carotid_data/va_sources/derived/thin_slice/song/<case>/`):
  - `qc/qc.json` (tsba) and `qc/qc_ts.json`, `qc/select.json`, `qc/gross.json`;
  - figures `qc/qc_<case>.png`, `qc/<case>_wall3d_<side>.png`, `qc/<case>_detail_along_va.png`, `qc/lookv1_*.png`;
  - `ct_seg/ct_seg4/{seg_qc,measure,landmarks,edge,protrusion}.json`, `ct_seg/ct_seg4_nofix`, `ct_seg/ct_seg4_negzero`;
  - `arterial/runs/native` (air1000) and the failed-input runs.
- Logs: `derived/thin_slice/song/_logs_proc/` (`timing.jsonl`, `driver_*.out`, `final.log`).
- Scripts: `/sp/v5_va/pilot_proc_song_*`. The closing pass is `final.sh`; `grade.py` and `summary.py` are copies of
  ia172's with Song additions; `protrusion.py`, `tables.py` and `lookv1.py` are new.
