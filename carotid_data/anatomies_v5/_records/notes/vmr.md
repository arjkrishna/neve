# vmr: VMR 0248 as the real-anatomy reference and as a v5 VA donor

Agent prefix `vmr`. Scripts are under `/sp/v5_va/vmr_*.py`. Results are in `results/vmr0248_metrics.json`, with working files under `results/vmr/`. Derived meshes and centerlines are in `carotid_data/va_sources/derived/vmr0248/`, with a `README.txt` and the VMR licence copied in. Nothing tracked in the repo was modified, and no git was run.

All metrics come from the shared `va_metrics.py` and `va_standard.py`, unchanged. The one adapter is `vmr_measure.union_segments`. It repeats the calls of `union_detail` so that `measure_union_branch` can run on landmark ranges, and it reproduced measure_va's union core exactly (0.0245 = 0.0245, and so on, on every VA). The voxel is 0.4 × 0.4 × 0.7 mm, so the grader applies the 0.75 mm resolution factors.

![union detail along the VA](vmr_detail_along_va.png)

## Answer first

- **The lofted SimVascular walls cannot meet the natural-anatomy standard.** VMTK confirms what ref found:
  - union > 0.3 mm is 0.025 (R) and 0.035 (L);
  - sections are 0.019 / 0.018;
  - 4 of 6 bins of 40 mm read below 0.01 on each side.
- **A segmentation of the CT does not meet it for a whole VA either.**
  - The validity gates pass.
  - The wall detail it finds is real but **local**: branch ostia, contacts with bone and veins, and the C2 loop.
  - The upper V3 and V4 are smooth tubes at CT resolution: 0.000–0.012 > 0.3 mm, against TopBrain's 0.083 / 0.230 in the distal 0–40 / 40–80 mm.
  - It therefore fails W6 and W7, and W1 and W3 at set level. G2 fails too, on a real, tight C2 loop.
- **The cervical stretch C6 → C1 is borderline.** Union is 0.079–0.152 (R) and 0.122–0.187 (L), against W1 set ≥ 0.1135. The result depends on the segmentation choices: a 0.8 mm opening at bone and vein contacts halves the right V2.
- **As a donor it is one patient.** A rigid whole-VA placement into the 223 v4 anatomies clears 0.6 mm in 66 of them when the anatomy keeps the patient's sides (native) and in none when the sides are swapped and mirrored. A graft would need non-rigid fitting.
- **Recommended role:**
  - the gross-anatomy and landmark reference for V1–V4;
  - the benchmark for Arterial / TotalSegmentator centerlines, which needs the CT segmentation together with the expert paths;
  - not the v5 wall source.

## 1. Lofted walls through VMTK (`vmr_prep_lofted.py`, `vmr_vmtk.py`)

The open wall faces `wall_{right,left}_vertebral` were capped with vmtkSurfaceCapper (centerpoint). vmtkCenterlines was then run with pointlist seeds (from the origin where the expert path leaves the subclavian lumen, to the merge) and AppendEndPoints=1, exactly as in `topbrain_tools/vmtk_centerline.py`. The wall was measured on the open face, as in ref e.

| | R | L | standard |
|---|---|---|---|
| centerline eccentricity | 0.025 | 0.032 | V1 ≤ 0.15. VMTK fixes the drawn path's 0.31 |
| length, DM | 218.6 mm, 1.40 | 226.3 mm, 1.46 | G4 pass |
| r MISR median (p10–p90) | 1.53 (1.22–1.73) | 1.77 (1.41–2.11) | G1 pass |
| q_min at W = 10 mm | 0.78 | 0.79 | G3 pass. ref's 0.47 pinch was an artefact of implicit-distance MISR |
| min bend radius at σ = 3 | **1.72** | **1.70** | G2 FAIL (the C2 loop, see §3) |
| union > 0.3 / 0.5 mm | **0.025 / 0.004** | **0.035 / 0.009** | W1 set ≥ 0.1135 FAIL |
| union > 0.15 r | 0.038 | 0.039 | W3 set ≥ 0.194 FAIL |
| sections > 0.3 mm off r_in; circularity | 0.019; 0.992 | 0.018; 0.994 | W4 set ≥ 0.051 FAIL |
| bins of 40 mm below 0.01 | 4 of 6 | 4 of 6 | W6 FAIL |

The only "detail" sits at the subclavian junction (V1, 0.075 / 0.10) and at the C2 loop (V3 C2→C1, 0.056–0.076). V2, the upper V3 and V4 read 0.000 by construction.

## 2. The VAs from the CT (`vmr_ct_seg3.py` → `ct_seg4`, primary)

### Method

- **Stations** every 0.25 mm along the expert paths: both VAs from origin to merge, the basilar, and the subclavians within ±30 mm of each origin. R is the contour r_eq.
- **Rays.** At each station, 64 rays in the normal plane, sampled every 0.05 mm with trilinear interpolation.
- **Local half-maximum edge** per ray, T = (I_core + I_exterior)/2:
  - I_core is the median HU within 0.4 R of the path;
  - I_exterior is the soft tissue just outside, or the dip before a brighter neighbour.
- **Upper threshold** where a neighbour brighter than max(1.3 I_core, I_core + 200 HU) touches the lumen without a dip: T_hi = (I_core + its peak)/2. This catches dense venous contrast and cortical bone.
- **Smoothing.** T is smoothed over 11° × 0.5 mm. T is a threshold field, not the wall.
- **Recentring.** Two passes: each station moves to the centroid of its edge polygon. Pass 1 moved them 0.33 / 0.35 mm, which is the drawn path's eccentricity (ref measured 0.31). Pass 2 moved them 0.05 mm.
- **3D label.** Each voxel goes to the vessel with the smallest d/R. The label is T_lo ≤ HU ≤ T_hi within the per-ray edge + 0.5 mm, 6-connected to the stations, with holes filled. The VA ends are planar cuts at the origin and at the merge.
- **Local opening.** Within 2 mm of any upper-threshold ray, a 0.8 mm opening removes partial-volume spurs from the neighbour's halo. This is what makes `ct_seg4`; `ct_seg3` is the same without it.
- **Surfaces:**
  - **stage-A:** binary marching cubes + windowed sinc 20 it / 0.1. This is `mask_to_surface.write_vtp`, unchanged, so it matches TopBrain;
  - **iso:** a sub-voxel isosurface of HU = T, unsmoothed.
- **Centerlines:** VMTK on each surface.

Superseded versions:
- `ct_seg` used one threshold per station. 20–24% of the label boundary ran out to the ROI limit, through the venous plexus and bone.
- `ct_seg2` had no upper threshold. Left V1 leaked into dense venous contrast: union 0.60, circularity 0.73.

QC for `ct_seg4`:
- lumen core 479 / 511 HU; soft tissue 40 / 45 HU; T median 282 / 305 HU (p10–p90 215–350 / 247–385);
- ray rules: soft-tissue crossing 96 / 93%, dip rule 4 / 5%, upper threshold 0.1 / 1.7%;
- boundary at the cap: 0.03 / 0.28%;
- label voxels touching ≥ 700 HU: 0 / 0.27%;
- all surfaces closed.

**Known defect.** From s ≈ 10 to 40 mm (V1 into low V2) the left VA runs against a vein full of undiluted contrast, probably reflux into the left vertebral vein. There is no dip between them. The arterial wall there cannot be separated from that vein's partial-volume halo, so left V1 wall numbers (union 0.47, circularity 0.84) are **not** wall detail.

### What the CT can resolve

An erf edge fit on rays with a soft-tissue exterior gives these edge widths:

| vessel | σ (mm) | FWHM (mm) |
|---|---|---|
| VA lumen | 0.58 / 0.59 | 1.36 / 1.39 |
| basilar | 0.61 | 1.44 |
| subclavians | 0.76 / 0.79 | 1.8 |

The effective resolution is therefore about 3.5 × the 0.4 mm voxel, beyond the standard's 1.15 mm full-grading limit if it were judged by PSF rather than voxel. Sub-millimetre wall relief cannot appear in this CT. What it shows is relief at 1 mm and above.

### Results (whole VA, origin → merge)

| | lofted | CT seg4 stage-A (primary) | CT seg4 iso | CT seg3 stage-A | CT seg3 iso |
|---|---|---|---|---|---|
| union > 0.3, R / L | 0.025 / 0.035 | 0.045 / 0.138* | 0.054 / 0.157* | 0.069 / 0.175* | 0.081 / 0.185* |
| union > 0.5, R / L | 0.004 / 0.009 | 0.025 / 0.091* | 0.034 / 0.107* | 0.046 / 0.132* | 0.059 / 0.136* |
| union > 0.15 r, R / L | 0.038 / 0.039 | 0.091 / 0.183* | 0.088 / 0.198* | 0.116 / 0.217* | 0.114 / 0.226* |
| sections > 0.3 off r_in, R / L | 0.019 / 0.018 | 0.057 / 0.178* | 0.072 / 0.210* | 0.081 / 0.208* | 0.084 / 0.223* |
| union genus, R / L | 0 / 0 | 0 / 0 | 0 / **1** | 0 / **1** | 0 / **2** |
| r MISR median, R / L | 1.53 / 1.77 | 1.31 / 1.47 | 1.31 / 1.47 | 1.31 / 1.47 | 1.31 / 1.48 |
| eccentricity, R / L | 0.025 / 0.032 | 0.044 / 0.066 | 0.035 / 0.058 | 0.044 / 0.065 | 0.034 / 0.057 |

\* The left values include the venous-contact V1, which reads 0.47–0.50.

Grades for the primary representation (`grade_set`, with the 0.75 factors):

- **Passes:**
  - V1–V6 on both VAs, including genus 0;
  - G1, G3, G4;
  - W2 set 0.058 (≥ 0.020), W4 set 0.117 / 0.091, W5 set 0.092 / 0.082.
- **Fails:**
  - **W1 set** 0.0915 against ≥ 0.1135;
  - **W3 set** 0.137 against ≥ 0.194;
  - **W6** on both VAs: the dead stretches are the upper V3 and V4;
  - **G2**, the C2 loop.
- **W7** is not automated in the grader. On the distal 0–40 / 40–80 mm the CT reads 0.000 / 0.005 (R) and 0.007 / 0.011 (L), against ≥ 0.078 / ≥ 0.218. That is a clear FAIL.
- **W5 is high for the wrong reason.** The 0.75–1.5 mm roughness band reads 0.08–0.11, above TopBrain's 0.055, because of features of 1 mm and up (branch stubs, contacts), not fine texture.

### Per segment (V1–V4, primary vs lofted)

Each cell gives the lofted value / the CT seg4 stage-A value. The left VA's V1 marked † is the venous-contact stretch.

| R VA | V1 | V2 | V3 | V4 | V3 C2→C1 | V3 C1→dura |
|---|---|---|---|---|---|---|
| range (CT, mm) | 0–44.8 | 44.8–112.0 | 112.0–183.3 | 183.3–219.9 | 112.0–143.5 | 143.5–183.3 |
| length, DM | 44.8, 1.08 | 67.2, 1.15 | 71.2, **2.19** | 36.7, 1.14 | | |
| κ p50 σ3 (/mm); turning | 0.054; 138° | 0.029; 180° | 0.113; 451° | 0.029; 77° | | |
| r MISR | 1.62 / 1.39 | 1.59 / 1.30 | 1.51 / 1.29 | 1.22 / 1.14 | 1.44 / 1.20 | 1.66 / 1.41 |
| union > 0.3 | 0.075 / 0.043 | 0.000 / 0.061 | 0.029 / 0.052 | 0.000 / 0.000 | 0.076 / 0.126 | 0.000 / 0.005 |
| sections > 0.3 | 0.074 / 0.072 | 0.000 / 0.069 | 0.018 / 0.055 | 0.000 / 0.000 | 0.044 / 0.128 | 0.000 / 0.004 |

| L VA | V1 | V2 | V3 | V4 | V3 C2→C1 | V3 C1→dura |
|---|---|---|---|---|---|---|
| range (CT, mm) | 0–27.5 | 27.5–115.0 | 115.0–188.8 | 188.8–225.0 | 115.0–142.5 | 142.5–188.8 |
| length, DM | 27.5, 1.19 | 87.5, 1.23 | 73.8, **2.16** | 36.2, 1.07 | | |
| κ p50 σ3 (/mm); turning | 0.092; 149° | 0.031; 294° | 0.114; 441° | 0.049; 99° | | |
| r MISR | 2.24 / 1.21† | 1.79 / 1.49 | 1.78 / 1.59 | 1.43 / 1.28 | 1.53 / 1.40 | 1.87 / 1.69 |
| union > 0.3 | 0.108 / 0.47† | 0.037 / 0.113 | 0.017 / 0.046 | 0.000 / 0.007 | 0.056 / 0.116 | 0.000 / 0.010 |
| sections > 0.3 | 0.053 / 0.70† | 0.005 / 0.164 | 0.031 / 0.066 | 0.000 / 0.003 | 0.090 / 0.126 | 0.000 / 0.018 |

Per 10 mm (figure above), the CT detail sits in a few places:

| VA | where the CT detail sits (union > 0.3) |
|---|---|
| R | s 70–80 (0.23, a side protrusion: branch stub or venous contact); the C2 loop at 110–120 (0.17); the C1 foramen at 130–140 (0.18) |
| L | the C2 loop at 100–120 (0.35–0.43) |

Other 10 mm bins read 0–0.10 (most below 0.06 on the right), and every bin from s ≈ 150 mm to the merge reads ≤ 0.03.

### CT vs lofted calibre

Expert contour r_eq is 1.23 × (R) and 1.18 × (L) the CT VMTK MISR. Lofted VMTK MISR is 1.17 × and 1.13 ×.

How the profiles correlate with the CT, per 1 mm:

| | Pearson r |
|---|---|
| R, whole VA | 0.70 |
| R, V4 | 0.95 |
| R, V2 | 0.41 |
| L, V3 | 0.95 |
| L, V1 | −0.23 (both representations are contaminated there) |

About 0.1 mm of the gap is the half-maximum bias of a blurred small cylinder (σ²/2r ≈ 0.12 mm). The rest is the expert's threshold. CT stage-A and iso agree to within 1% (r ≥ 0.98).

## 3. Landmarks, take-off and the "hairpin" (`vmr_landmarks.py`, `vmr_ccj.py`, `vmr_fm.py`)

**Bony enclosure.** The enclosure measure is the share of 48 normal-plane rays that meet HU ≥ 400 within r + 0.3–7 mm. Foramen runs ≥ 0.5 sit at:

| VA | z of the foramen runs (mm) |
|---|---|
| R | −228, −216/−213, −204/−200, −188, −173 (C2), −157 (C1) |
| L | −242…−213 (merged C6 + C5), −203, −197, −185, −173 (C2), −154 (C1) |

**Levels.** These were checked against the bodies on the midsagittal CT:

| body | z (mm) |
|---|---|
| C2 incl. dens | −142…−180 |
| C3 | −182…−197 |
| C4 | −200…−212 |
| C5 | −215…−228 |
| C6 | −231…−244 |
| C7 | −247…−260 |

Both VAs enter at **C6**. The left entry is lower (z −242) and less certain, because its venous contrast also reads as "bone" here.

**Dura.** The dural entry is taken where the VA first rises above the foramen-magnum plane after C1. The basion is at (y, z) = (105.5, −140.5) and the opisthion at (76.5, −145.0), picked on `results/vmr/fm_zoom.png`. This is a radiological proxy; the anatomical piercing lies a few mm proximal. The V3 posterior loop's most posterior point is at s 160 (R) and 167 (L).

| (CT s, mm) | C6 entry | C2 exit | C1 exit | dura | merge |
|---|---|---|---|---|---|
| R | 44.8 | 112.0 | 143.5 | 183.2 | 219.9 |
| L | 27.5 | 115.0 | 142.5 | 188.8 | 225.0 |

- **V3 dominates the tortuosity.** V3 has DM 2.2 (two loops: C2 and C1/suboccipital) against V1 1.1–1.2, V2 1.15–1.23 and V4 1.07–1.14. Each whole VA has one ≥ 360° loop within 60 mm.
- **Take-off from the subclavian**, at 5 / 10 mm, with elevation:

  | | R | L |
  |---|---|---|
  | CT | 56° / 55°, elevation 58° | 70° / 62°, elevation 60° |
  | lofted (VMTK) | 64° / 56° | 98° / 73° |
  | host | 85° | 81° |

  ref's 61° / 73° came from the recentred loft.
- **ref's "hairpin" at 80–120 mm is real.** It is the **C2 loop**, just above the C2 transverse foramen:

  | | where | min bend radius σ3 (CT) | r MISR there | lofted |
  |---|---|---|---|---|
  | R | s 116–119, z −171 | 1.44 mm | 1.25 | 1.72 |
  | L | s 102.5–106, z −169 | 1.54 mm | 1.12 | 1.70 |

  So G2 (≥ 2.5 mm) fails on real anatomy. The bend radius is about the vessel radius, so a tube folds there: expect the v4 ladder to work at that spot. There is no narrowing at the loop in the CT (q_min 0.83 R).

## 4. Graft probe (`vmr_graft_probe.py`, not a graft)

The whole CT-segmented VA (seg4 stage-A, VMTK MISR) was placed by a proper weighted Kabsch: origin on the host VA origin, course along the host VA, no scaling. In mirrored anatomies the patient's opposite side was x-mirrored, as v4's handedness rule does. Clearance was taken as centerline distance − r − r_other against every non-VA branch of each v4 anatomy, against the other VMR VA, and against itself.

| | native placement (tested on all 223; v4 has 103 native) | mirrored placement (tested on all 223; v4 has 120 mirrored) |
|---|---|---|
| course RMS to the host VA | 7.9 / 8.0 mm | 9.7 / 8.0 mm |
| end mismatch at the confluence | 8.5 / 11.8 mm | 10.6 / 12.0 mm |
| anatomies passing 0.6 mm everywhere | **66 / 223** | **0 / 223** |
| failures by branch | RVA×RCCA 101; LVA×LCCA 88, ×LECA 12 | RVA×RCCA 134, ×right subclavian (17) 74, ×RECA 15; LVA 80 |

Self-clearance and VA-to-VA clearance (≥ 4.1 mm) are fine.

A rigid single-patient VA fits the corridor the host and v4 grafts left for it in only a minority of anatomies. Grafting it would need the v4 repairs (cut moves, deflection) or a smooth non-rigid warp onto the host corridor. Every v5 anatomy would then carry the same patient's VA pair.

## Verdict for VMR 0248

- **Lofted walls: no.** They are smooth by construction and fail W1–W6. Keep them only as gross references.
- **CT segmentation: not as a natural-anatomy VA meeting the standard.**
  - Its gates are clean.
  - Its V1–V2 wall is real CT relief at ≥ 1 mm scales: borderline against W1 and below W3.
  - Its upper V3 and V4 are smooth (W6 and W7 fail).
  - The left V1 is unusable for wall detail (venous contrast).
- **If v5 still wants one real cervical wall**, the representation to graft is the CT segmentation:
  - `ct_seg4`: stage-A surface for comparability with TopBrain, or iso for sub-voxel fidelity, with VMTK centerlines;
  - right VA origin → C1, and left C6 → C1 (its V1 from a tube or the mirrored right);
  - joined to v4's TopBrain distal pieces above C1.
  - Caveats: one patient for 223 anatomies, a real G2-failing C2 loop, rigid placement failing clearance in most anatomies, a CT effective resolution of about 1.4 mm, and a research-only licence with acknowledgement.
- **Best use:**
  - gross-anatomy and landmark ground truth: V1–V4 lengths, DM per segment, C6 entry, C2 and C1 loops, take-off, calibre;
  - the benchmark for an Arterial run on TotalSegmentator CTAs. VMR's expert paths plus this CT segmentation at 0.4 mm can be resampled to 1.5 mm to test how a 1.5 mm centerline degrades, which ref listed as an open item.

## Files

- Scripts in `/sp/v5_va/`:
  - `vmr_common.py` and `vmr_run.sh` (docker helper; passes `VMR_*` env vars);
  - `vmr_cpr.py`, `vmr_zoom.py`, `vmr_xsec.py`, `vmr_seg_qc.py` (QC views);
  - `vmr_ct_seg.py`, `vmr_ct_seg2.py` (superseded) and `vmr_ct_seg3.py` (v3, and v4 with `VMR_OPEN_BRIGHT=1`);
  - `vmr_vmtk.py` and `vmr_vmtk_all.sh`, in neve-build-vmtk run as `-u 0` (the container user cannot write to the bind mount);
  - `vmr_prep_lofted.py`, `vmr_landmarks.py`, `vmr_spine.py`, `vmr_ccj.py`, `vmr_fm.py`;
  - `vmr_measure.py`, `vmr_compare.py`, `vmr_graft_probe.py`, `vmr_assemble.py`, `vmr_fig.py`.
- Results:
  - `results/vmr0248_metrics.json`: summary, segment tables, grades, 10 mm bins, edge sharpness, bends, radius comparison, cervical pieces, graft probe, and the full measure_va results;
  - `results/vmr/`: per-tag metrics, `vmr_compare.json`, `graft_probe.json`, enclosure profiles and the QC images.
- Data: `carotid_data/va_sources/derived/vmr0248/{lofted,ct_seg4,ct_seg3,ct_seg2,ct_seg}`, `README.txt`, `LICENSE_VMR_0248.txt`.

## Open issues

- **Which modality is right?** TopBrain's distal detail (0.08 / 0.23) comes from 0.3 mm TOF-MRA labels, and VMR's CT has an edge FWHM of about 1.4 mm. Whether the CT's smooth V4 and upper V3 is truth or blur cannot be settled from this patient, and no TopBrain image-domain PSF was measured.
- **Single segmentation rule set.** There is no expert CT label to validate against, only the contours, which are smooth circles about 0.2 mm larger in radius. The sensitivity range (seg3 vs seg4, stage-A vs iso) is reported throughout.
- **Landmarks are semi-manual.** The foramen-magnum points were picked by eye (±2 mm), and the left C6 entry is uncertain because of the venous contrast.
- **The graft probe is centerline-level only.** There is no union, bake or real-surface clearance.
- **Stray empty file.** A zero-byte file was created by mistake at `C:\Program Files\Git\tmp_patch.awk`, by a stray `cat >` in Git Bash. Removing it was blocked by the sandbox; it is harmless but can be deleted.
