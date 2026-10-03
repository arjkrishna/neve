# Cervical vertebral artery grafting feasibility (v5)

Date: 2026-10-01. Scope: whether VMR 0248, the SPL head-and-neck atlas, Arterial run on TotalSegmentator v3 CTAs, and
the open thin-slice CTAs in the donor-pool plan can supply real cervical vertebral arteries (VAs) for v5. Two
questions: do they meet the natural-anatomy wall standard ("nooks and crannies"), and can they be grafted under the
v4 rules. The host VA, VMR 0248, v4's shipped VAs and the 50 TopBrain VA pieces serve as references.

Each number below was measured by a study agent and then checked by an independent verifier with its own code.
Where a verifier overturned a number, this report uses the corrected value or shows both values. The section
[What the verification changed](#what-the-verification-changed) lists every change. Working notes and scripts are in
the session scratchpad under `v5_va/` (paths at the end). Derived data is in `carotid_data/va_sources/`, which is
gitignored. Apart from this report, nothing in the repo was changed.

## Answer

1. **No source meets the VA wall standard for a whole VA. The two sub-millimetre CTs, VMR 0248 and TaG-Net patient A,
   come closest, and only borderline on the neck stretch.**
   - **VMR 0248's lofted walls** are smooth by construction. They read 0.025 / 0.035 (R / L) of wall more than
     0.3 mm off the tube, against a set minimum of 0.0966.
   - **VMR 0248's own CT, re-segmented,** reads 0.045 / 0.138 over the whole VA. Its upper V3 and V4 are smooth even
     in the raw image (distal 0–40 mm: 0.000 / 0.007, against ≥ 0.078). Its neck stretch, C6 to C1, reads
     0.076 / 0.126. That passes the native thresholds by a hair, but the relief sits at bone and vein contacts and
     at the C2 loop.
   - **TaG-Net patient A's CT** (0.4 mm slices, segmented the same way) reads 0.045 / 0.111 over the whole VA (set
     0.078, below VMR's 0.0915). Its neck stretch passes W1 or fails it depending on where C1 is placed (set
     0.093–0.117). Its relief, like VMR's, sits at ≥ 1 mm scales and at bone and branch contacts: the thinner slices
     sharpen the edge only along z.
   - **SPL** (1.4 mm slices) and **Arterial on TotalSegmentator** (1.5 mm) cannot be graded. Their apparent detail
     is label and voxel artefact.
2. **The graft mechanics work.** The v5 prototype grafts VMR's CT-segmented VA from the subclavian origin to a V3
   splice, and keeps v4's TopBrain piece above the splice (strategy B).
   - All 4 prototypes bake to one closed genus-0 tree, decimate cleanly to 60k / 23.5k triangles, pass SOFA, and
     keep the RCCA route's v3 detail exactly (ratio 1.000). A verifier re-bake was bitwise identical.
   - One v4 rule is broken. The new join at the VA origin is never turn-gated, and 22 of 223 anatomies turn more
     than 60° there (up to 98°). With that join gated, 201 of 223 pass.
3. **v5's VA walls beat v4's but miss the standard.**
   - On the 8 prototype VAs, v5 reads a median of 0.093, against v4's 0.041 on the same anatomies. The set minimum
     is 0.0966, so v5 fails W1.
   - Over all 223 anatomies the modelled median is about 0.073–0.077. That is a clear fail, not a borderline one:
     the prototypes happen to carry unusually long TopBrain pieces.
   - The donor's take-off is not kept: a median of 93° / 78°, against VMR's own ≈ 54° / 59°.
4. **What each source can supply:**
   - VMR 0248: 2 VAs from 1 patient, with a real CT wall.
   - SPL: 2 VAs, course only.
   - Arterial on TotalSegmentator: 15 neck VAs from 11 of the 24 subjects tried, course only. Six of the 21
     originally counted had followed the internal carotid above C2. On TaG-Net patient A degraded to 1.5 mm, every
     trace (12 of 12) also left the VA for the internal carotid, and the V3-loop gate rejected all of them.
   - TaG-Net patient A: 2 VAs with a real CT wall at VMR's level, as a second benchmark and calibration case. Its data
     has no licence, so it is for internal use only.
   - Other open thin-slice CTAs: untested.
5. **Recommendation.** Keep strategy B. Scale the cervical stretch with real course from Arterial and TotalSegmentator,
   baked as tube. Pilot thin-slice CTAs as the only route to real cervical wall. Fix the grader and the graft gates
   listed below. Do not build all 223 anatomies from VMR 0248 alone.

## Terms used

- **Union detail** (also "> 0.3 mm"). The share of the VA's wall area that stands more than 0.3 mm (or 0.5 mm, or
  0.15 × radius) off the plain tube.
  - It is measured on the undecimated owned union, with the measure of `bake_meshes_v4.right_route_detail`, applied
    to the VA by `va_metrics.measure_union_branch`.
  - It is the quantity v4 used to show that the RCCA route kept v3's nooks.
- **Gate-valid TopBrain.** The 36 of 50 TopBrain VA pieces whose VMTK centerline passes the validity gates (V1–V4).
  The standard's thresholds are 0.95 × their use-weighted medians.
- **Stage-A surface.** Binary marching cubes followed by windowed-sinc smoothing (20 iterations, pass band 0.1).
  This is TopBrain's surface protocol (`mask_to_surface.write_vtp`).
- **Cervical stretch.** From the VA origin to the dura (V1–V3). **Distal** means dura → confluence (V4).
- **Strategy B.** Donor VA from its origin to a splice in V3, then v4's TopBrain piece to the blind end. There are
  two seams: the origin seam (host subclavian → stub → donor) and the splice.
- **Follow handedness.** In a v4-mirrored anatomy, the donor's opposite VA is used, x-mirrored, so the whole
  posterior circulation is one consistent mirror image.
- **Ownership k.** The number of nearest centerline samples searched when a union triangle is assigned to a vessel.
  The repo measure uses 24; "exact" means k ≥ 64, where results stop changing.
- **Edge FWHM.** The width of a CT lumen edge. It sets the finest wall relief the image can show, whatever the
  voxel size.

## The yardstick: host, v4 and TopBrain

### Wall references

Every value is the union detail on the undecimated union. R / L where two values are given.

| reference | > 0.3 mm | > 0.5 mm | > 0.15 r | note |
|---|---|---|---|---|
| host VA (eve_bench dualdevicenav) | 0.0007 / 0.000 | 0 | 0 | plain tube from centerline + radii |
| v4 VA as shipped, whole VA (20 VAs) | **0.021** | 0.007 | 0.033 | 64.5% host tube (0.0003), 9.3% ramp (0), 26.1% TopBrain (0.125) |
| v4 TopBrain donor stretch, in situ | 0.125 | 0.035 | 0.176 | the only real VA wall in v4 |
| TopBrain pieces alone, median of 50 | 0.121 | 0.026 | 0.172 | use-weighted 0.113 |
| TopBrain gate-valid, use-weighted (the standard's basis) | 0.102 | 0.0215 | 0.149 | 36 pieces, 360 of 446 uses |
| v3 / v4 RCCA route: fork / siphon | 0.196 / 0.322 | 0.097 / 0.156 | 0.130 / 0.257 | the original v3 standard |

- **The TopBrain VA has v3-like density, measured size-free.** At > 0.15 r it reads 0.17, between the fork (0.13)
  and the siphon (0.26). Its absolute numbers are lower only because the vessel is smaller.
- **Detail grows away from the skull.** TopBrain's distal 0–40 mm bin (about V4) reads 0.083; its 40–80 mm bin
  (V3 and the C1 loop) reads 0.230.
- **The reference measure reproduces exactly.** An independent re-implementation matched `mesh_v4.json` route detail
  to 0.0000 on 5 rebuilt v4 unions, the host floor exactly, and the TopBrain pieces to within 0.004.

### The standard (`results/standard.json`, graded by `va_standard.py`)

| rule | what | threshold |
|---|---|---|
| V1–V6 | validity: centerline offset from the lumen centre ≤ 0.15 mm; ≤ 10% of the centerline outside the wall; closed sections; the VA's own union genus 0; voxel ≤ 1.15 mm; VMTK MISR radii | gates; wall rules apply only if these pass |
| G1–G5 | gross anatomy at σ = 3 mm: radius 1.09–2.25 mm; bend radius ≥ 2.5 mm; local calibre ≥ 0.6 × surroundings; length 180–260 mm with DM 1.25–1.65; distal 80 mm shape | per VA |
| W1 / W2 / W3 | union > 0.3 mm / > 0.5 mm / > 0.15 r | set medians ≥ 0.0966 / 0.0204 / 0.1416 |
| W4 | shape-only guards: section perimeter > 0.3 mm off the inscribed circle; native surface vs fitted tube | ≥ 0.1045 / ≥ 0.0627 |
| W5 | roughness, band 0.75–1.5 / 1.5–3 mm RMS | ≥ 0.051 / ≥ 0.045 mm |
| W6 | no dead stretch: every 40 mm bin | ≥ 0.01 |
| W7 | full-length VA, distal 0–40 / 40–80 mm | ≥ 0.0785 / ≥ 0.2183 |
| A1–A4 | grafted anatomy: VA detail recorded beside route detail; in situ ≥ 0.95 × the piece alone; the whole VA beats v4 (0.021); every v4 gate kept | |

The verifier found five problems with this standard. They change how some results must be read:

1. **The bar sits at 0.85 × what v4 ships, not 0.95.** The 14 pieces left out of the gate-valid set are the most
   detailed ones (union 0.222 against 0.1045). Measured against all pieces, use-weighted, the bars would be W1
   0.1074, W2 0.0218, W3 0.1519.
2. **The resolution factors point the wrong way.** For 0.65–1.15 mm sources the standard multiplies W1–W3 by > 1
   (×1.175 at 0.75 mm, ×1.439 at 1.0 mm). Under the radius protocol the standard itself requires (VMTK on the
   source's own surface, rule V6), coarser imaging lowers union detail instead: ×0.43 at 0.75 mm, ×0.30 at 1.0 mm.
   The earlier "coarse voxels inflate detail" result held only when the native centerline was kept fixed. So no
   result in this report relies on the factors; native thresholds are quoted throughout.
3. **The resolution gate fails open.** `va_standard.factors(None)` returns native thresholds. A 1.5 mm label resampled
   onto a 0.5 mm grid and declared 0.5 mm passes W1, W3, W4 and W6 on staircase artefacts. Arterial writes its masks
   on exactly such a 0.5 mm grid.
4. **W7 and G5 are not implemented in the grader.** Applied per VA, W7 would fail 24 of the 50 TopBrain pieces.
5. **A2 holds for v4 only as a median.** Pairing in-situ against alone, v4 itself fails A2 on 7 of 20 VAs: short or
   tube-baked donors read 0.000 in situ. Separately, W3 can be met by centerline eccentricity alone within V1's
   bound: a featureless tube 0.15 mm off-centre reads 0.301.

**What the standard does not cover.** No real-wall reference exists for V1–V2. TopBrain's labels stop near C1–C2, and
VMR's walls are lofted. The cervical stretch is therefore held to bars derived from distal MRA pieces. Some of
TopBrain's V3 detail may itself be TOF-MRA artefact. The VMR CT cannot settle whether a smooth CT V4 is true anatomy
or blur.

### Gross-anatomy references (σ = 3 mm unless noted)

| | host R / L | VMR 0248 CT R / L | SPL R / L | TaG-Net A R / L (GT) | TaG-Net B R / L (centrelines only) | Arterial on TotalSegmentator, whole VAs | TopBrain pieces |
|---|---|---|---|---|---|---|---|
| length (mm) | 210 / 209 | 219.9 / 225.0 | 221.1 / 193.2 | 219.0 / **266.2** (G4 fails) | 228.7 / 233.5 | median 232.5 (206–257), n = 11 | 70 (33–88) |
| DM | 1.39 / 1.42 | 1.40 / 1.46 | 1.59 / 1.41 | 1.49 / **1.75** (G4 fails) | 1.51 / 1.54 | 1.45 (1.33–1.56), n = 12 † | 1.77 |
| radius median, MISR (mm) | 1.67 / 1.75 | 1.31 / 1.47 (lofted 1.53 / 1.77) | label 1.89 / 1.81; image FWHM 1.47 / 1.44 | CT 1.18 / 1.88; image half-max 1.24 / 2.01 | – | 1.59 (1.24–1.69) † | 1.72 |
| minimum bend radius (mm) | 4.24 / 5.39 | **1.44 / 1.54** (real C2 loop; G2 fails) | **1.40–2.19 / 1.71** (G2 fails) | 2.97 (V4) / **2.40** (C2 loop; G2 fails); right C2 loop 3.14 | **1.37** (C2 loop; G2 fails) / 2.59 | 3.5 (2.1–4.3) † | 4.86 |
| loops ≥ 360° in 60 mm | 1 / 1 | 1 / 1 | 2 / 1 | 2 / 1 | 1 / 1 | – | 0 |
| take-off at 10 mm | ≈ 81–93° (depends on the tangent definition) | ≈ 54° / 57–62° | R 113–134° (origin hook), L 40–50° | ≈ 40–49° / 35–43° by tangent definition (CT centreline 46° / 35°) | ≈ 80–93° / 33–55° (assumed subclavian radius) | 56° † | – |
| V1 / V2 / V3 / V4 (mm) | synthetic; V3-like loop 80–200 mm from the origin | R 44.8 / 67.2 / 71.2 / 36.7; L ≈ 37 / ≈ 78 / 73.8 / 36.2 ‡ | R 44.5 / 109.8 / 53.0 / 39.5; L 34.8 / 84.0 / 64.5 / 27.5 | R 37.2 / 74.3 / 69.7 / 37.7; L 41.7 / 86.5 / 82.3 / 55.7 (semi-manual, ±3 mm) | – (no image) | – | – |

† From the 12 VAs before the yield correction, which still include the wrong-vessel s0289 R.
‡ Left C6 entry corrected by the verifier; see the VMR 0248 section.

## Per-source verdicts

| source | resolution | wall standard | measured wall, > 0.3 mm (R / L) | graftable under the v4 gates | VAs available | licence |
|---|---|---|---|---|---|---|
| **VMR 0248, lofted walls** | expert contours, 4.5–5.5 mm apart, circularity 0.99 | **fail**: W1, W3, W4, W6 | 0.025 / 0.035; 4 of 6 40-mm bins < 0.01 | as gross reference only | 2 (1 patient) | research use; cite Wilson et al. 2013 and the NIH grants |
| **VMR 0248, CT segmentation** (`ct_seg4`) | 0.4 × 0.4 × 0.7 mm voxels; edge FWHM 1.39 / 1.44 mm | **whole VA: fail** (W1 0.0915, W3 0.137, W6, W7). **C6 → C1: borderline** (set W1 0.101 ≥ 0.0966, W3 0.155 ≥ 0.1416) | whole 0.045 / 0.138 (left V1 vein-contaminated); C6→C1 0.076 / 0.126; distal 0–40 mm 0.000 / 0.007 | **yes, as strategy B**: 4 of 4 bakes; dry-run gates as implemented 223 / 223; 201 / 223 with the origin join gated | 2 VAs (4 shapes with follow): right origin → C1; left C6 → C1, its V1 baked as tube | as above |
| **SPL atlas** | 0.98 × 0.98 × 1.4 mm | **not gradeable** (V5); also fails V1 (offset 0.226 / 0.158 mm) and V4 (left genus 2) | 0.325 / 0.349 — artefact, 1.4–4.7 × an ideal scan of a real VA on the same grid | course only; untested (needs scale ≈ 1.09, origins moved ≈ 7.7 mm, G2 kinks reviewed) | 2 (1 patient), tube walls | 3D Slicer licence B; the full-resolution MANIX scan is paywalled and not redistributable |
| **Arterial on TotalSegmentator v3** | 1.5 mm isotropic; Arterial resamples to 0.5 mm | **not gradeable** (V5); readings sit at the resolution and noise floor | median 0.088 (0.057–0.132) on 21 VAs; same pipeline on VMR: 0.026 / 0.050 at 0.4 mm → 0.071–0.118 at 1.5 mm | course only, baked as tube; registry grafting not yet built | **15 neck VAs** from 11 of 24 subjects (both sides in 4); 11 whole VAs from 7 subjects (both sides in 4) | non-commercial (Arterial CC BY-NC weights, PolyForm NC code) + TotalSegmentator CC BY 4.0 |
| **Open thin-slice CTAs** (Song 2024, 172-series, AortaSeg-60) | 0.4–0.8 mm | **untested**. Expected to be at best VMR-like: real relief only at ≥ 1 mm scales, and smooth from Arterial's own mask (TaG-Net A bears this out) | none measured | expected yes through the same pipeline | plan estimates ~400 sides (unverified) | CC BY 4.0 / CC0 images, but NC through Arterial |
| **TaG-Net patient A, CT segmentation** (`ct_seg4`; CTA 002) | 0.405 × 0.405 × 0.4 mm; edge FWHM 1.30 / 1.29 mm (along z 1.11, in the axial plane 1.35) | **whole VA: fail** (W1 0.078, W3 0.100, W4, W6, W7). **C6 → C1: borderline**, set by the C1 definition (W1 0.093–0.117, W3 0.117–0.149) | whole 0.045 / 0.111; C6→C1 0.054–0.071 / 0.131–0.164; V3 C2→C1 0.000 / 0.048 | internally only, as a strategy-B donor (gates V1–V6 and both V3 loops pass); not grafted | 2 VAs (1 patient); patient B (case 003): centrelines only | no data licence: internal use only, never shipped |

### VMR 0248

**Frame and source.** One CT patient: SimVascular paths, lofted walls, and a 512 × 512 × 446 contrast CT at 0.4 × 0.4 ×
0.7 mm. Units are centimetres, the axes RAS. Both VAs run from inside the subclavian lumen to the basilar merge, and
both lie fully inside the CT. Frame, units and lumen position were verified independently: centerline HU median
499 / 523, none below 250.

**Lofted walls: no.** The 98 expert contours are circles (circularity 0.99). Their only "detail" sits at the
subclavian junction and the C2 loop.

| measure | R / L | threshold |
|---|---|---|
| union > 0.3 mm | 0.025 / 0.035 | W1 set ≥ 0.0966 |
| sections | 0.019 / 0.018 | W4 ≥ 0.1045 |
| 40 mm bins below 0.01 | 4 of 6 per side | W6 |

**CT segmentation: real wall, but local and coarse.**

The `ct_seg4` method:
- a per-ray local half-maximum edge, recentred twice;
- an upper threshold for dense venous contrast and cortical bone;
- a 0.8 mm opening, only near those bright contacts;
- stage-A surface and VMTK MISR.

The gates pass: offset 0.044 / 0.066 mm, genus 0 / 0.

| stretch | union > 0.3 mm, R / L | TopBrain or the standard |
|---|---|---|
| whole VA | 0.045 / 0.138 (left includes the vein-contaminated V1, 0.47) | W1 set ≥ 0.0966 |
| V2 | 0.061 / 0.113 | – |
| V3, C2 → C1 (the C2 loop) | 0.126 / 0.116 | – |
| V3, C1 → dura | 0.005 / 0.010 | TopBrain 40–80 mm: 0.230 |
| V4 | 0.000 / 0.007 | TopBrain 0–40 mm: 0.083 |
| cervical C6 → C1 | 0.076 / 0.126 (set W1 0.101, W3 0.155) | ≥ 0.0966 / ≥ 0.1416 at native thresholds |

- **Where the relief sits.** A right V2 side protrusion (s 70–80, 0.23–0.24), the C2 loop (0.17 R, 0.35–0.43 L), and
  the C1 foramen. Every 10 mm bin from s ≈ 150 mm to the merge reads ≤ 0.03.
- **The smooth V4 is in the image, not the processing.** Image half-max edges more than 0.3 mm off the station median
  occur on 0.0006 / 0.027 of rays in V4, against 0.15 / 0.20 in V2. So the smooth V4 is not caused by the
  segmentation or the smoothing.
- **The relief is at ≥ 1 mm scales and partly at contacts.** The CT's edge FWHM is about 1.4 mm (3.5 voxels). The
  0.8 mm opening halves the right V2 figure. The left V1 (s ≈ 10–40 mm) runs against a vein full of undiluted
  contrast, so its "detail" is that vein's halo. Strictly, the verifier also rated the cervical CT relief "real" too
  strongly: it sits at bone and vein contacts and at one protrusion of unknown nature.
- **The C2 loop is real and fails G2.** Bend radius 1.44 / 1.54 mm at r 1.25 / 1.12, with tangents turning
  133° / 143° over 10 mm. The CT and the VMTK-lofted centerline agree. Earlier "right VA pinch" reports were radius-
  method artefacts (CT q_min 0.83 / 0.72).
- **Landmarks.** Both VAs enter at C6. Right: V1 / V2 / V3 / V4 = 44.8 / 67.2 / 71.2 / 36.7 mm. **Left entry,
  disputed:** the study put it at s 27.5 mm, but that bony-enclosure run counts the contrast-filled vein as bone. Both
  readings are shown below.

  | left VA | V1 | V2 | V3 | V4 |
  |---|---|---|---|---|
  | study | 27.5 mm | 87.5 mm | 73.8 mm | 36.2 mm |
  | verifier (vein rays excluded; bony ring by eye at s ≈ 37) | 31–40 mm | 75–84 mm | 73.8 mm | 36.2 mm |

  So VMR's left V1 / V2 is not yet clean ground truth.
- **Take-off at 10 mm** is ≈ 54° (R) and 57–62° (L) by tangent definition (53.5° / 58.7° on the graft's definition),
  against the host's ≈ 81–93°. The subclavian elevation at the VA origin is 64° / 36° in VMR, against 26.5° / 13.2°
  on the host (5 mm chord).
- **Rigid whole-VA placement** clears 0.6 mm in **34 of 223** anatomies when the handedness rule is applied: 34 of
  the 103 native anatomies, 0 of the 120 mirrored. The study's 66 of 223 placed 32 mirrored anatomies natively.

**Verdict.** The CT segmentation is the only real cervical VA wall in hand. As a whole VA it fails W1, W3, W6, W7
and G2. Its neck stretch is borderline, and the result depends on segmentation choices. It is one patient. Its best
uses are as the first strategy-B donor, as gross and landmark ground truth (right side), and as the benchmark for
Arterial.

### SPL atlas

- **The labels are complete.** Labels 105 / 106 are one piece each, with no empty slice, from the subclavian to the
  confluence (VMTK ends 2.9 mm apart).
- **They are 2–3 voxels across.** 39–41% of label voxels are below the lumen half-maximum, the label radius is about
  25% too wide (MISR 1.89 / 1.81 mm against image FWHM 1.47 / 1.44 mm), and a quarter of the boundary is drawn
  against bone labels.
- **The wall fails the gates before any wall rule applies.**
  - V5: the 1.4 mm slices are over 1.15 mm.
  - V1: offset 0.226 / 0.158 mm.
  - V4: the left union has genus 2.
- **The apparent detail is artefact.** Union 0.325 / 0.349 and sections 0.497 / 0.485 would pass a naive nooks test
  by 3–5×. But against an ideal re-imaging of the 36 gate-valid TopBrain walls on SPL's own grid, SPL reads ×1.4–1.5
  on the union and ×4.7 on sections. After a 1.5 mm low-pass it still reads 0.21, where a real VA wall reads 0.002.
- **Smoothing cannot rescue it.** A σ 1 mm blur removes 84% of a real wall's section detail and leaves SPL's union
  unchanged. It also splits the thin label.
- **The basilar is labelled as skull.** It is present in the image (370–440 HU, z 47.6 → 67) but carries the skull
  label.
- **The shipped models are unusable.** Each is one closed piece after cleaning, not the 24 / 55 pieces first
  reported, but they are shrunk to r 0.85 mm.
- **Grafting would be course only.**
  - Smooth the centerline at σ 3 mm.
  - Take radii from the image, or scale the label's by 0.8.
  - Similarity-fit to the host: scale 1.087, origins 7.6–7.8 mm off.
  - Review the right origin hook (bend radius 1.40–2.19 mm, by end handling) and the left V1 coil (1.71 mm).
  - The wall would be a plain tube, like the host's. This was not attempted.

**Verdict.** SPL adds one patient's course and nothing toward wall detail.

### Arterial on TotalSegmentator v3

**What TotalSegmentator v3 is.** All 239 fetched CTs are released at 1.5 mm isotropic, even where the source slices
were 0.75 mm, and none has a VA label. 123 are CTAs; 50 show the whole course; 12 were ranked tier A, plus 7 tier A2
from mixed-phase polytrauma and neck CT.

**Arterial's own labels are not usable; its binary mask is.** On VMR 0248 at four image qualities:

| check | result |
|---|---|
| VA labels covering ≥ 90% of the expert VA | 5 of 8 |
| continuous subclavian → VA → basilar labelled paths | 3 of 8; only 1 ends on a mostly correct basilar label |
| the native RVA label | on an intracranial vessel 62 mm away |
| expert VA centre inside the binary mask | 97–100% at every quality |

**Tracing the mask, then VMTK, recovers the course accurately** (verified):

| measure | value |
|---|---|
| centerline distance to the expert path | mean 0.33–0.49 mm, p95 0.61–1.07 mm |
| for comparison: expert path to the independent CT centerline | 0.35 / 0.42 mm (the reference floor) |
| coverage | 0.94–1.00 |
| radius | 0.90–1.08 × the CT-segmentation MISR |
| VMTK origin to the expert VA origin | 0.9–2.1 mm |

One defect: at 1.5 mm the left trace cuts the C2-loop corner. CT → Arterial wall distance p95 reaches 2.9 mm there,
and 11% of the CT wall lies more than 1 mm from Arterial's surface.

**The wall is not gradeable, and the readings are artefact.**
- Same patient, same pipeline: union 0.026 / 0.050 at 0.4 mm becomes 0.080 / 0.111 at 1.5 mm (×3.1 / ×2.2), and
  0.071–0.118 with noise matched to TotalSegmentator.
- The bin-by-bin rank correlation with the real CT wall drops from 0.61 (native) to 0.45, 0.40 and 0.23 as noise
  rises.
- The TotalSegmentator VAs read a median of 0.088 and do not rise above that band once calibre is accounted for.
- **This inflation is consistent with the verifier's resolution finding.** A 1.5 mm image segmented on a 0.5 mm grid
  inflates union detail about ×2.2 even with VMTK. A surface made directly on the 1.5 mm grid deflates it. Arterial
  does the former.
- **Even at 0.4 mm, Arterial's mask is smoother than the CT.** It reads W1 0.038 against the CT segmentation's 0.0915.
  Arterial is a course generator, not a wall source, at any resolution.

**Yield (corrected by the verifier).** 24 subjects were tried. 3 were rejected before Arterial because no axial slice
cuts the arch into two limbs; these include tier-A ranks 1 and 2 (s1163, s0152). 21 runs gave 42 sides, and 40 were
traced.

| | as first reported | corrected |
|---|---|---|
| usable neck VAs (origin → dura) | 21 from 13 subjects; both sides in 8 | **15 from 11 subjects; both sides in 4** (s0610, s0777, s0960, s1786) |
| continuous / bridged | 9 / 12 | 9 / 6 |
| usable whole VAs (origin → confluence) | 12 from 7; both sides in 5 | **11 from 7; both sides in 4** |
| yield by noise (low / high) | 17 of 26 / 4 of 16 sides | 12 of 26 / 3 of 16 sides |

- **Six VAs were dropped: s0289 R, s0333 L, s0518 L, s1107 R, s1129 R, s1154 R.** Each trace bridges a 4.5–11 mm
  mask gap at the C2/C1 foramen, often through soft tissue at 41–65 HU. It lands on an anterior vessel, mostly the one
  Arterial labels ICA, and climbs it to the skull base, never taking the V3 loop around the atlas.
- **A V3-loop test separates them cleanly.** Real loops have a posterior excursion of 18–32 mm and lie ≤ 4.8 mm from
  the posterior arch of C1. The failures read ≤ 9.1 mm and ≥ 7.7 mm.
- **"Bridged" is the risk marker.** 6 of the 12 bridged sides went to the wrong vessel; none of the 9 continuous
  sides did.
- **Other checks came out clean.**
  - No venous or bone contamination was found on the usable sides: centerline HU is 0.74–1.24 × the CCA, and at most
    0.9% of points exceed 700 HU.
  - Noise is close to white, so the noise-matched VMR copies are a fair floor.
  - The ts agent's edge-jitter estimate (0.29–0.96 mm) overstated the noise about 2×.

**Grafting.** These donors would be baked as tube (`real_surface: false`). They add course, take-off, entry level,
V3 loop and calibre variety, but no wall detail. `graft_v5_va.py` handles only the single VMR donor so far, so no
TotalSegmentator VA has been grafted.

### Open thin-slice CTAs (the scale-up path)

The donor-pool plan lists these open sets:
- Song et al. 2024: about 133 arch-to-vertex CTAs, 0.39–0.62 mm in-plane, CC BY 4.0. HU is clipped at 0 in the one
  sample viewed.
- The Head-CTA 172-series set: about 120 series at about 0.4 × 0.4 × 0.5 mm, CC BY 4.0. Patients are not yet
  de-duplicated.
- AortaSeg-60: 28 head-first arterial CTAs, 0.51–0.82 / 0.6 mm.
- RSNA-ICA 2025: gated, non-commercial, no redistribution.

The TaG-Net sample is already on disk at `carotid_data/va_sources/tagnet/`:
- CTA 002 at 0.405 × 0.405 × 0.4 mm, with 18-class labelled centerlines, both VAs running subclavian → basilar;
- case 003 as centerlines only.
- No data licence is stated, so it is for internal benchmarking only.

**Apart from TaG-Net patient A (next section), not one of them has been run or measured in this study.** What the
evidence predicts:
- **Wall.** VMR 0248 at 0.4 × 0.4 × 0.7 mm was the only thin-slice CT measured before TaG-Net. It shows real relief
  only at ≥ 1 mm scales, a smooth upper V3 and V4, and borderline neck numbers. A thin-slice donor processed by a
  CT-edge segmentation like `ct_seg4` should land in the same range, and Arterial's own mask would land lower (W1 0.038
  on VMR). The 0.4 mm slices of TaG-Net and the 172-series set may sharpen the edge beyond VMR's 1.4 mm FWHM. On
  TaG-Net they sharpen it only along z, and no finer relief appears; the 172-series set is untested.
- **Course.** Arterial traced VMR at 0.4 mm with no gap. Thin-slice yields should therefore beat TotalSegmentator's,
  but the plan's 70–90% per-side rate is an assumption.
- **Count.** The plan's open estimate is about 400 thin-slice sides (Song ~190, 172-series ~170, AortaSeg ~40), before
  de-duplication. Treat it as unverified until a pilot runs.

## Second benchmark patient: TaG-Net

TaG-Net's public sample gives a second sub-millimetre CT with ground-truth vessel centrelines, after VMR 0248. It was
run through the same tools as VMR (TotalSegmentator masks, Arterial, the trace, VMTK, `ct_seg4`), and the study's QC
thresholds were then recalibrated on both patients. Three verifiers checked the work. The Arterial benchmark and the
CT wall reproduce. The recalibration's numbers reproduce too, but several of its recommended values were tuned on the
same data that tests them.

### The data and its licence

- **Patient A** is the head-and-neck CTA `SampleData/002/CTA.nii.gz`: 512 × 512 × 796 voxels at 0.405 × 0.405 ×
  0.4 mm, raw HU. Its 18-class, one-voxel centreline ground truth (GT) is filed under `001/`, on the same grid.
- **Patient B** is case 003: GT centrelines only (0.366 × 0.366 × 0.4 mm), with no image. `001/seg.nii.gz` is a mask
  of a third, different scan and was not used.
- **Labels and frame are settled.** The NIfTI value is the txt label + 1: R-VA 8, L-VA 13, BA 9, R-SCA 14, L-SCA 15.
  The code's "BCT(1)" is the arch and its "AO(2)" the brachiocephalic trunk. The world frame is RAS and the voxel axes
  LPS. The header is not mirrored: the image shows a normal left arch.
- **Licence: none.** The code is GPL-3.0, but the data has no stated terms. Everything derived from it is for internal
  benchmarking and calibration only. It stays under the gitignored `carotid_data/va_sources/derived/tagnet/` and must
  not enter a shipped anatomy until the authors grant terms.

### Ground-truth quality

On patient A the GT is a better course reference than VMR's expert paths.

| check, R / L | TaG-Net A, GT | VMR 0248 |
|---|---|---|
| HU at the path, median | 595 / 607; none below 150 | CT centreline 499 / 523 |
| inside the image half-max lumen | 100% / 100% | – |
| offset from the local lumen centre, median (p95) | 0.13 / 0.16 mm (0.35 / 0.54) | expert path 0.33 / 0.35 mm |
| continuity | one 26-connected chain per VA, no gaps, steps ≤ 0.70 mm | 98 lofted contours, 4.5–5.5 mm apart |
| course | threads the transverse foramina C6 → C1 and takes the V3 loop on both sides (checked by eye every 5 mm) | the same, from the CT |

- **What it is.** Most likely the skeleton of an expert-labelled or expert-corrected segmentation.
  - It has skeleton junction clumps, and its labels are inherited from mask classes: the first 4.6–6.2 mm of every VA
    carries the SCA label, and the last 3.1–3.2 mm of each left VA the BA label.
  - It is not TaG-Net's own output: along a thin right V4 it follows the image where TaG-Net's centreline breaks.
  - The paper's annotation protocol was not checked.
- **Its limits.** It is voxel-stepped (smoothed here at σ 0.75 mm), its ends are label boundaries (VMR's ostium and
  merge rules were applied), and it has no radius. Patient A's radius is therefore the image half-max equal-area
  radius, **r_img**. Landmarks were placed semi-manually (±3 mm). Patient B can be checked only from its structure.
- **Anatomy.** Patient A's left VA is dominant (r_img 2.01 against 1.24 mm), long and tortuous: 266 mm, DM 1.75, so it
  fails G4. Its right V4 is thin (r 0.7–0.8 mm). Both VAs enter at C6, like VMR's.
- **Verified.** The bench verifier's own extraction lies within 0.154 mm of the GT file, and the HU, lumen and offset
  checks reproduce.

### Arterial benchmark, side by side with VMR 0248

The protocol copies VMR's: native, 1.5 mm, and 1.5 mm with TotalSegmentator-like clean and sharp noise; the same
Arterial settings, trace, VMTK and scoring code. Re-scoring VMR with this code reproduces its stored benchmark
exactly. Two choices were added, because on patient A they decide the outcome:
- **The skull mask** (Arterial's head/neck split): the VMR rule (HU ≥ 400 above the basion), or the TotalSegmentator
  skull mask that the TotalSegmentator donors used ("TS skull").
- **The masks for the trace QC** (`art_extract --ts`). TotalSegmentator 2.18.0 was run on patient A. Its `total_v3`
  task reproduces the dataset's masks (C1 Dice 0.98); the older `total` task puts C1 1.5–3.2 mm low, which moves the
  V3-loop values. All values below use `total_v3`. On the 0.4 mm grid the QC peaks at 17.7 GB and takes 18 min;
  1.5 mm copies of the masks give the same landmarks within about 1 mm in 47 s and 1.4 GB. These mask figures come
  from the study agent alone; no verifier re-ran TotalSegmentator.

"The exact VMR call" below means the VMR skull rule, art_extract's foramen-magnum height z_FM at the basion, and no
`--ts`.

| | VMR 0248 | TaG-Net A |
|---|---|---|
| Arterial VA labels covering ≥ 90% of the reference VA | 5 of 8 | 2 of 16 |
| continuous subclavian → VA → basilar labelled paths | 3 of 8 | 0 of 16 |
| reference VA centre inside the native binary mask, R / L | 1.00 / 1.00 | 0.94 / 1.00 |
| whole VA at native, VMTK mean / p95 (mm), R; L | 0.38 / 0.62; 0.40 / 0.91 | 0.32 / 0.56; 0.26 / 0.47 (TS skull and `--ts`). The exact VMR call goes to the wrong vessel on both sides |
| the same, scored against each patient's own CT-segmentation centreline, exact mean (mm), R / L | 0.16 / 0.21 | 0.16 / 0.16 |
| ostium → C2 exit, VMR skull rule, all four image qualities, VMTK mean (mm) | 0.36–0.70 | 0.26–0.34 |
| 1.5 mm sides that stay on the VA to the basilar | 6 of 6 | **0 of 12** |
| VMTK MISR / CT-segmentation MISR | 0.89–1.08 | 0.85–1.00 |
| Arterial native wall, union > 0.3 mm, R / L (CT segmentation in brackets) | 0.026 / 0.050 (0.045 / 0.138) | 0.022 / 0.066 (0.045 / 0.111) |
| bin-by-bin rank correlation with the CT wall, native | 0.61 | 0.41 |
| union at 1.5 mm ÷ native, matched stretch ostium → C1, R / L | ×2.5–2.9 / ×1.6–2.0 | ×1.5–1.7 / ×0.6–1.0 |

- **Labels: less reliable than on VMR.** At native the "LVA" label lies on the right MCA (91% of its points), the
  dominant left VA carries no label, and the BA label lies on the MCA, ACA and PCA.
- **Course: as accurate as on VMR wherever the trace stays on the VA.** Against the GT, A's correct native traces read
  0.26–0.32 mm mean, against VMR's 0.38–0.40. That gap is a reference confound: A's GT is medial, while VMR's expert
  path sits 0.33–0.35 mm off-centre. Against each patient's own CT centreline the right sides are equal and the left
  sides differ by 0.03–0.05 mm.
- **At native, the protocol decides.** Both sides are correct only with the TS skull and `--ts`.
  - With the exact VMR call, both traces leave the VA at the C1 foramen and climb the internal carotid (coverage
    0.61–0.63, p95 21 mm).
  - The cause, as verified: `art_extract.find_ba` seeds the basilar on an ACA segment. The basilar lies mostly inside
    find_ba's ±5 mm midline band, but it loses the largest-component contest to an ACA component. The seed window
    admits that component because it reaches z_FM + 55 mm, where the docstring says + 40. A wider band does not fix
    it; the documented + 40 mm does.
  - With `--ts` and the VMR skull rule, the right side still fails, because the mask misses the thin right V4.
- **At 1.5 mm, every trace goes to the wrong vessel: 12 of 12 sides.** This holds at all three noise levels, with both
  skull masks, and even when the trace is given the GT basilar end.
  - Arterial's 1.5 mm mask loses the intradural V4: the right from s ≈ 180–191 mm, the left from s ≈ 234–247 mm.
  - The cheapest path then bridges a 4–12 mm gap at the C1 foramen into the ICA and crosses the PCA to the basilar
    top: 70–133 mm of path off the VA, up to 33 mm from it.
  - This is the TotalSegmentator failure mode, now on a 0.4 mm patient degraded to 1.5 mm. VMR's 1.5 mm masks never
    lost a V4, so VMR was the favourable case.
- **The V3-loop gate catches all of them.** Every correct path passes: GT excursion 17.8 / 18.9 mm, correct traces
  18.0–19.2 mm, all ≤ 0.7 mm from the posterior arch of C1. Every wrong trace fails: 0.2–3.2 mm, and 9.5–24.6 mm from
  the arch. On `total` masks, though, the left GT and the correct left traces read 14.2–14.6 mm and fail the ≥ 15 mm
  floor.
- **The V1–V2 stretch is accurate only with the VMR skull rule.** With it, VMTK reads 0.26–0.34 mm mean at every
  quality. With the TS skull at 1.5 mm, which is how the TotalSegmentator donors were run, the mask breaks inside V2:
  VMTK reads 0.53–2.12 mm mean with p95 up to 12 mm, and fails outright on the right in 2 of 3 runs. So this patient
  does not support an "origin → C2" donor class for TotalSegmentator CTAs. The donors' yield may partly reflect the
  skull mask, not only noise; that is untested.
- **Wall: Arterial is smoother than the CT, as on VMR.** At native, ≤ 0.3% of rays cast onto Arterial's surface lie
  more than 0.3 mm off their station median, against 2.1–3.4% for the image's own edges. At 1.5 mm the inflation
  depends on calibre (thin right VA ×1.5–1.7, dominant left ×0.6–1.0), and the bin pattern no longer tracks the CT
  (ρ −0.05 to 0.22). "The 1.5 mm reading is artefact" still holds. The bench verifier did not re-derive these wall
  numbers (its lens was course, labels and the gate); the CT values they are compared with were verified.

### CT wall: what 0.4 mm slices buy

`ct_seg4` was run unchanged apart from its loader, with stations along the GT path, stage-A surfaces, VMTK MISR, exact
ownership, the acquisition voxel (0.405 mm) and native thresholds. One grid effect: on this grid the 0.8 mm opening
becomes a 3 × 3 × 3-voxel cube (±0.4 mm); a 0.82 mm ball changes the results by ≤ 0.004. Each segmentation took
72 s. The gates V1–V6 pass on both VAs (offset 0.033 / 0.062 mm, genus 0 / 0).

**Edge FWHM** (mm, R / L; the verifier's erf fit, VMR re-measured with the same code):

| rays | TaG-Net A | VMR 0248 |
|---|---|---|
| all | 1.30 / 1.29 | 1.39 / 1.44 |
| along z | **1.11 / 1.12** | 1.46 / 1.45 |
| in the axial plane | 1.35 / 1.34 | 1.41 / 1.45 |
| per stretch, R: V1 / V2 / C2 → C1 / C1 → dura / V4 | 1.53 / 1.33 / 1.25 / 1.25 / 1.19 † | 1.36 / 1.46 / 1.37 / 1.44 / 1.28 |
| per stretch, L | 1.40 / 1.30 / 1.22 / 1.32 / 1.27 | 1.63 / 1.46 / 1.55 / 1.41 / 1.38 |
| noise in the subclavian lumen (HU) | 15 | 28 |

† The erf fit under-reads at r 0.75 mm; the thin right V4's 1.19 is about 1.27.

- **Thinner slices sharpen the edge only along z.** That helps where the VA runs horizontally: the V1 curve, the C2
  loop and V3. In the axial plane, which carries the edge for the mostly vertical V2 and V4, TaG-Net is only 4–7%
  sharper than VMR. Its noise is 0.55–0.63 × VMR's, depending on the method.
- **No finer relief appears.** The roughness bands below 0.75 mm read the same floor as VMR's (about 0.04 mm RMS).
  Lower noise removes edge jitter; it does not add detail. So thin slices alone will not make CT a sub-millimetre wall
  source. A sharper reconstruction kernel might, and TaG-Net's low noise suggests the scanner had headroom. That is
  untested.

**Wall detail per stretch** (union > 0.3 mm, R / L; TaG-Net with landmarks placed on the CT by VMR's method; VMR in
the same harness):

| stretch | TaG-Net A | VMR 0248 | TopBrain or the standard |
|---|---|---|---|
| whole VA | 0.045 / 0.111 (set 0.078) | 0.045 / 0.138 (set 0.0915) | W1 ≥ 0.0966; TopBrain gate-valid 0.102 |
| V1 | 0.009 / 0.088 | 0.043 / 0.468 (vein) | – |
| V2 | 0.093 / 0.203 | 0.061 / 0.113 | – |
| V3, C2 → C1 | **0.000** / 0.048 | 0.126 / 0.116 | – |
| V3, C1 → dura | 0.000 / 0.031 | 0.005 / 0.010 | TopBrain 40–80 mm: 0.230 |
| V4 | 0.066 / 0.071 | 0.000 / 0.007 | TopBrain 0–40 mm: 0.083 |
| distal 0–40 / 40–80 mm | R 0.049 / 0.000; L 0.041–0.044 / 0.041–0.043 | R 0.000 / 0.005; L 0.007 / 0.011 | W7 ≥ 0.0785 / ≥ 0.2183 |
| cervical C6 → C1 | 0.071 / 0.152 (set 0.111, W3 0.143) ‡ | 0.079 / 0.120 (set 0.099, W3 0.151) § | ≥ 0.0966 / ≥ 0.1416 at native thresholds |

‡ art_extract's C1 is the end of the C1-mask enclosure, which runs on along the V3 groove of the posterior arch for
about 30 mm. With it, C6 → C1 reads 0.054 / 0.131 (set 0.093, W3 0.117) and fails. With the left C1 at the foramen
exit by VMR's exact convention (s 161, from the verifier), it reads set 0.117, W3 0.149.
§ Same harness. The VMR section's 0.101 / 0.155 came from the study's own harness.

- **Grades at native thresholds.** The whole VA fails W1 (0.078), W3 (0.100), W4 (sections 0.092, fitted 0.059), W6
  (a 0.000 right bin) and W7 (both sides). It passes W2 (0.044) and W5. W3 is not an eccentricity artefact: the
  offsets are 0.03–0.06 mm.
- **C6 → C1 is borderline, as on VMR, and turns on where C1 is put.**
  - With the foramen-exit C1, W1 passes the native and all-piece bars (0.111–0.117). W3 passes native (0.143–0.149)
    but not all-piece (0.1519).
  - With art_extract's C1, both fail.
  - Shifting C6 and C1 by ±3 mm, the CT landmarks pass W1 in 9 of 9 cases and W3 in 7 of 9; the TotalSegmentator
    landmarks pass W1 in 1 of 9 and W3 in none.
- **Where the relief sits.** Over the whole VA, almost all of it sits in a few places:
  - bone contacts in the transverse foramina;
  - a lumen-bright vessel of about 1 mm attached at the C4/C5 foramen on both sides: a branch stub, or a vein of equal
    enhancement. It carries 71% of the right VA's relief, and VMR's right "side protrusion" sits at the same place
    (s 70–80);
  - the left C2 loop, a right V4 branch origin (possibly PICA) with a calibre step, and an elliptical left V4.

  Outside these zones the union reads 0.002 (R) and 0.008–0.009 (L). In C6 → C1, 91–97% of the relief lies in contact
  and branch zones that cover 18–23% (R) and 46–56% (L) of the wall. The zones were drawn where the relief is, so the
  outside values are upper bounds, not an independent test.
- **The upper V3 is smooth, and the right C2 loop has no relief.** That loop is wide (bend radius 3.2 mm, against
  VMR's 1.44) and reads 0.000. VMR's main relief, its tight C2 loop, has no counterpart on TaG-Net's right side.
- **No venous contamination.** No contact reaches undiluted contrast (≥ 1800 HU; the maximum is 1418–1430 HU), so
  unlike VMR's left V1, nothing needs to be baked as tube.
- **As a donor** (internal only; not grafted or baked):
  - origin → C1 without the 25 mm seam zone reads 0.055–0.070 (R) and 0.138–0.152 (L), close to VMR's 0.056 / 0.225
    (its left inflated by the vein). With the seam zone counted as 0, it reads 0.047–0.058 / 0.121–0.131 (VMR
    0.049 / 0.192);
  - the left C2 loop fails G2 (2.45 mm), and the whole left VA fails G4 (264.5 mm, DM 1.76);
  - the right VA is thin (MISR 1.18 mm against the host's 1.67), so it would need a calibre ramp;
  - the take-off at 10 mm is 35–46° on the CT centreline (34–60° across tangent definitions).
- **Not independently re-derived:** the roughness bands, the low-pass survival, the erf-bias correction, the dura
  landmarks, and the CT gross shape and take-off.

### Calibration on two patients

The calibration re-ran the study's QC (`ts_qc`, the V3-loop test and the donor-pool plan's Stage-2 gate) on both GT
patients, on all their traces and on the 40 traced TotalSegmentator sides. TotalSegmentator was also run on VMR's CT
to give it vertebra masks. Three terms:
- **Excursion** and **d** are the V3-loop test's two numbers: the posterior swing of V3 relative to the spinal cord,
  and the distance from V3 to the posterior half of the C1 mask.
- **The robust cord** is the cord reference without its partial top slices, whose centroid jumps by up to about 6 mm
  at the foramen magnum.
- **Erec** is an entry rule. It keeps ts_qc's call, but accepts C7 only when the path has its own run of ≥ 5 mm within
  0.5 mm of the C7 mask, ending ≥ 2 mm before the C6 run; otherwise it calls C6.

The verifier reproduced every key number with its own code. The claim that the recalibration is "right and not
circular" does not hold. The table shows both.

| item | the calibration recommends | the verifier found | what two patients support |
|---|---|---|---|
| V3-loop gate | robust-cord excursion ≥ 12 mm, d ≤ 6 mm (study: ≥ 15, ≤ 5) | Values reproduce within 0.1 mm. Every floor from 8 to 14 mm with d ≤ 6 gives the same counts (GT 10 / 10, correct traces 26 / 26, wrong 0 / 68, TotalSegmentator 19 / 40), and the robust cord changes no count. The floor was set and checked on A: without A the data put it at 16.6 mm. d ≤ 6 has no GT basis (GT ≤ 1.1, wrong A traces ≥ 12.5); it rests on one TotalSegmentator side each way (s0346 L 5.1, real; s0518 L 7.5, wrong) | **the ≥ 15 mm floor is falsified**: A's left VA, a real loop, reads 14.3–15.0 mm. Any floor in (4.8, 14.3] fits the data; 12 and 6 are provisional |
| entry level | Erec | Calls reproduce: C6 on 129 of the two patients' 130 path × mask-set rows; on the 40 TotalSegmentator sides C6 33, C7 3, C5 1, none 3, with all 15 usable at C6. But A picked the parameters among 11 settings that are perfect on VMR, "always C6" scores every GT row too, and no GT VA enters at C7, so C7 sensitivity is untested. Erec's C6 landmark is within 1.2 mm on A but still 10 mm early on VMR's trusted right side | **ts_qc over-calls C7**: C7 on 9 of 10 GT paths and 91 of 94 traces, though all 4 GT VAs enter at C6; also on 20 of the 40 TotalSegmentator sides (9 of the 15 usable) |
| radius | native: VMTK MISR × 1.10; 1.5 mm: no single factor | Ratios reproduce to 3 decimals. The native factor is 1.10–1.21 depending on the Arterial run: on the same stations A's right MISR reads 0.769 × r_img with the VMR skull and 0.887 with the TS skull. Matched on protocol, the factors are 1.11 (VMR) and 1.21 (A). At 1.5 mm VMR needs × 1.05–1.08, A × 1.20–1.30 | **no single factor works.** Take calibre from the CT half-max profile: ct_seg4 MISR / r_img is 0.92–0.95 on both patients. The trace EDT runs 13–30% low |
| Stage-2 QC gate | rule 6 (Arterial label agreement) to review; drop the V2-band IQR term; allow V1 mask breaks ≤ 3 mm; count the 25 mm gap total only below the splice; V3 loop 12 / 6; BCV ≥ 200 HU to review; review lengths 260–290 mm | Counts reproduce. The draft as written, plus this report's additions, keeps **9 of 40 sides** (7 subjects, 2 pairs) and loses 6 of the 15 usable. Recalibrated: 15 / 40, 11 subjects, 4 pairs, 0 of the 6 wrong-vessel sides, matching the corrected yield. That match is in-sample: 5 of the 6 sides gained enter only through a limit moved for that side (s0346 L by d ≤ 6; s0610 L by V1 ≤ 3 mm; s1129 L, s1304 R and s1823 R by dropping the IQR term), and d ≤ 6 was set knowing s0518 L | **rule 6 to review**: it rejects 3 correct traces on the GT patients (labels 0.00–0.87). The BCV change rests on VMR alone and changes no count. The other limits come from the TotalSegmentator sides or from design. All need an out-of-sample test |
| gross ranges | length 193–266 mm, DM 1.40–1.75, C2-loop bend radius 1.37–3.25 mm, take-off 42–93°, subclavian elevation 12–64° | Values reproduce (bend radii within 0.02 mm). Two ranges are misstated: DM is **1.32–1.75** with the TotalSegmentator VAs; the take-off range leaves out SPL R (106°, origin hook) and the TotalSegmentator neck VAs (20–110°, n = 13), and its top is B R alone, on an assumed subclavian radius | **the G gates reject real anatomy**: 4 of the 6 GT VAs fail G2 (VMR both, A L by 0.1 mm, B R at 1.37 mm) and 1 of 6 fails G4 (A L) |

- **The host is the gross outlier.** It is shorter (209–210 mm) than every GT VA (219–266 mm), and it has no C2 loop
  (minimum bend radius 4.2 / 5.4 mm), where every real VA here bends at 1.37–3.25 mm. Its take-off and its subclavian
  elevation (13–27°) lie inside the real ranges.
- **VMR is the outlier in subclavian elevation:** 64° on the right, against 12–45° on the other patients' sides. The
  take-off analysis below, which traces the take-off loss to the host's flat subclavian, was done on the steepest
  donor. TaG-Net B (12° / 27°) lies within about 15° of the host's 26.5° / 13.2°.
- **Still open.** ts_qc's C2 and C1 are late on both patients (by 7–12 mm and 26–40 mm), and no mask rule finds the
  C1 foramen exit reliably at 1.5 mm. That affects the splice point and the rule for bridges low in V3.

**Verdict.** TaG-Net patient A confirms VMR's picture of the CT wall and sharpens the picture of Arterial.
- Its CT wall is VMR-like, not better: whole VA 0.078 against 0.0915, a borderline neck stretch, and relief only at
  ≥ 1 mm scales and at contacts. Its 0.4 mm slices sharpen the edge only along z. If relief at contacts does not
  count as nooks (decision 1 below), both patients' cervical walls are tube-like.
- Arterial's course is as accurate as on VMR wherever the trace stays on the VA. At 1.5 mm it went to the wrong vessel
  on every side, and the V3-loop gate caught every wrong trace.
- Two patients justify four changes: rule 6 to review, a V3 excursion floor below 15 mm, a fix for ts_qc's C7
  over-call, and calibre from the CT half-max. The specific new thresholds stay provisional until tested out of
  sample.
- It can serve internally as a second real-wall donor and calibration case. It must never ship.

Notes: `v5_va\notes\tag_labels.md`, `tag_totalseg.md`, `tag_arterial.md`, `tag_ctwall.md`, `tag_calibration.md`, and
the verifiers' `verify_tag_bench.md`, `verify_tag_ctwall.md`, `verify_tag_calibration.md`. Results:
`v5_va\results\tag_gt.json`, `tag_totalseg.json`, `tag_bench.json`, `tag_ctwall.json`, `tag_calibration*.json`. Data:
`carotid_data\va_sources\tagnet\` and `derived\tagnet\`.

## Can v5 be grafted under the v4 standards?

The prototype is in `carotid_data/va_sources/v5_proto/`. It uses strategy B with VMR's CT segmentation:
- **Origin seam.** The host subclavian is kept byte for byte. VMR's branch point is pinned on the host's, the VA is
  placed by a weighted Kabsch fit onto the v4 VA corridor, and a stub inside the parent joins them. The VA must
  leave the parent within 12 mm and never re-enter it.
- **Splice.** Length-matched, with v4's repair ladder. The TopBrain piece is Kabsch-fitted onto VMR's distal VA:
  v4's frame match sent tips up to 104 mm off on a real V3. The calibre ramp follows v4's rule.
- **Left V1.** VMR's left V1 (first 40 mm) is baked as tube because of the vein contact.
- **Bake.** `bake_v5.py` calls `bake_meshes_v4` unchanged apart from a run-time swap of `sections_for`.

### Gates (dry run on all 223 anatomies, plus 4 baked prototypes)

| v4 rule | v5 result | verified |
|---|---|---|
| 0.6 mm clearance, including ramps, the parent after exit, self and other branches | 223 / 223; minimum 0.63 / 0.60 mm (exact all-pairs 0.61 to the parent, 0.60 to the LECA) | yes: 24 re-run anatomies identical row for row |
| new joins ≤ 60°, at the splice | maximum 60.0° / 59.8°; median 42° / 36° (v4 frame match about 20°) | yes, to 0.0000 on 446 VAs |
| **new joins ≤ 60°, at the origin seam** | **not gated by `graft_v5_va.py`.** RVA turn at the branch point: median 44°, p90 57°, maximum 98°; **> 60° in 22 / 223** (14 / 223 with follow). v4 at the same point: maximum 40.2° | found by the verifier; 201 / 223 pass if gated. The 4 prototypes are ≤ 44.6° |
| exit from the parent ≤ 12 mm, no re-entry | 8.9–11.9 / 7.0–11.0 mm | yes |
| proper rotations, handedness | all proper (det +1); TopBrain pieces follow v4's rule; VMR native or follow; host and VMR have the same chirality | yes |
| owned union, genus-0 tree, 60k / 23.5k, no self-intersection, centerline inside | 4 / 4 bakes: 1 component, genus 0, 0 boundary or non-manifold edges, 0 self-intersecting faces, 0 centerline points outside | yes; the w_012 re-bake is bitwise identical |
| bake cost | 129–174 s for the clean three (v4 98–125 s); w_039 26 ladder attempts, 2627 s (v4 32, 2741 s), suppressing only v4's own Zenodo lower_L handle | yes |
| right-route detail (the v3 standard) | ratio 1.000 on all 4 | yes |
| SOFA load test | passes on all 4 at 150 steps; 334–426 ms per step (v4 371–404, within run noise) | yes, but the wire went up the right subclavian, so the new VA was never exercised. Under a harsher retract schedule w_012 segfaults in BeamAdapter, as v4 w_012 also does on a repeat run |

The 22 sharp origin turns are in the centerline only: the stub lies inside the parent lumen. But the RVA is
reachable from the training insertion in the brachiocephalic trunk, and v4 gated every new join.

The left (18) → LVA branch angle widens from 64° (host) to a median of 78° (follow: 81°). The host already exceeds
60° there, so this counts as take-off, not as a v4 join.

### VA wall in situ (exact ownership; the 25 mm seam counted as 0)

| anatomy | TopBrain piece R / L (mm) | v5 R / L | v4 R / L |
|---|---|---|---|
| case_k_004_left__topcow_mr_010 | 80 / 82 | 0.106 / 0.114 | 0.071 / 0.059 |
| case_w_007_left__topcow_mr_017 | 50 / 60 | 0.070 / 0.162 | 0.023 / 0.105 |
| case_w_012_right__topcow_mr_024_L (follow) | 34 / 34 | 0.080 / 0.053 | 0.019 / 0.011 |
| case_w_039_right__topcow_mr_021_L | 93 / 31 | 0.165 / 0.072 | 0.121 / 0.0005 |
| **median of 8** | | **0.0933** (0.0970 as first reported) | 0.041 |

**What holds:**
- **v5 beats v4 on 8 of 8 VAs**, by ×1.4 to ×4.9 (w_039 left ×144), under every definition the verifier tried. A3
  (the whole VA beats v4's 0.021) holds.
- **VMR's in-situ wall depends on the side.** VMR's left VA reads 0.10–0.14 in situ and its right only 0.05–0.06. With
  native handedness the RVA, the VA reachable in training, gets VMR's smoother side.

**What fails:**
- **W1.** It fails at 0.0933 against 0.0966. The first-reported 0.0971 came from 24-neighbour ownership assigning
  parent-subclavian wall to w_012 R's first 3 mm. Counting the seam as real instead of 0 gives 0.1015, so the verdict
  also depends on that convention.
- **W2 and W3.** W2 is 0.043. W3 fails at 0.1398 against 0.1416.
- **W6.** It fails on 5 of 8 VAs, not 4: every VA with a V1 tube.
- **W7.** It fails on 2 of 8 at 0–40 mm and on 5 of 8 at 40–80 mm, wherever the TopBrain piece is 60 mm or shorter.
- **A2.** w_039 left's TopBrain piece reads 0.914 of its standalone value, under the 0.95 rule (absolute values
  0.0064 / 0.0070).

**Over all 223 anatomies.** The 4 prototypes carry long TopBrain pieces (right median 65 mm, against 48 mm across the
set). A per-mm additive model was built, with calibration MAE 0.005 on the 8 VAs:

| | modelled value |
|---|---|
| set median over 446 VAs | **0.073** (about 0.077 bias-corrected; follow 0.071) |
| VAs reaching 0.0966 | 26% |
| a random 4-anatomy sample reaching W1 | 3.8% of draws |

VMR's wall covers a median 66.5% (R) and 49.2% (L) of every VA's length in all 223 anatomies. So the set medians
mostly measure which TopBrain pieces were drawn.

**Strategy choice.** B beats A (the whole VMR VA) on wall detail: A reads 0.044 / 0.061, with a distal stretch of
about 0. B was not shown to beat C, a tube bridge to v4's TopBrain piece left where v4 put it; that cheap variant was
never tested.

### Take-off (corrected)

The graft's host parent tangent spanned an 8.7 / 7.1 mm chord, where the donor's was 5 mm. With that fixed (placements
unchanged):

| | first reported | corrected |
|---|---|---|
| grafted take-off at 10 mm, median R / L | 99° / 82° | **93.0° / 78.3°** |
| \|Δ\| against the donor, median | 45.7° / 23.2° | **39.5° / 19.6°** |
| within 15° of the donor | 29 / 41 of 223 | **41 / 63 of 223** |
| host subclavian elevation at the VA origin | 20.4° / 14.1° | **26.5° / 13.2°** |
| prototype R take-offs (k_004, w_007, w_012, w_039) | 70 / 95 / 99 / 96° | 61 / 87 / 95 / 88° (k_004 R is in fact preserved) |

- **The VA keeps its own rise** (elevation at 10 mm 59°, against VMR's 58–63°). The angle changes because the
  host's subclavian runs nearly flat where VMR's rises steeply.
- **The donor-pool plan's Stage-6 recipe should be dropped.** It frame-matches the subclavian tangents and keeps the
  angle exactly (|Δ| 0° / 3.4°), but sends the VA out of the neck (tips 164 mm apart, course 71 / 37 mm RMS off the
  corridor) while passing every clearance gate.
- **Rolling about the VA chord** recovers about half the angle at 1.5–2.2× the course error.
- **No rigid placement keeps both take-off and course** while the host subclavian is kept.

## What the verification changed

| item | first reported | corrected or disputed | effect on the conclusions |
|---|---|---|---|
| resolution calibration (ref) | coarse voxels inflate union detail ×1.18 / 1.44 / 2.15 at 0.75 / 1.0 / 1.5 mm; at 1.5 mm only 3 of 49 pieces pass the gates; radius −22% | true only with the native centerline held fixed. With VMTK on the coarse surface (V6): ×0.43 / 0.30 / 0.28; the gates pass on 5 of 5 traversable pieces; radius −5%. A 1.5 mm image on a 0.5 mm grid inflates ×2.16 | the W1–W3 resolution factors are wrong-signed; native thresholds are used here. 1.5 mm sources stay non-gradeable for a different reason: the measure does not track the wall |
| V5 gate | grades by declared voxel | fails open with voxel None, and on resampled grids | must use the acquisition voxel and fail closed |
| standard's bar | 0.95 × TopBrain | 0.85 × what v4 ships (all-piece bars W1 0.1074, W2 0.0218, W3 0.1519) | any "pass" here is against the softer bar |
| W7, G5 | in the standard | not implemented in the grader; per VA, W7 fails 24 / 50 TopBrain pieces | W7 results here were computed by hand |
| A2 | holds for v4 (0.125 in situ against about 0.12 alone) | holds as a median only; 7 of 20 v4 VAs fail when paired | A2 needs a per-piece decision |
| W3 | size-free wall measure | met by 0.15 mm eccentricity alone (0.301) | always read W3 together with W1 |
| VMR left C6 entry | s 27.5 mm; V1 / V2 27.5 / 87.5 mm | s ≈ 31–40 mm (likely 37); V1 / V2 ≈ 31–40 / 75–84 mm | left VMR landmarks are not ground truth yet |
| VMR rigid whole-VA probe | 66 / 223 pass | 34 / 223 under the handedness rule | a rigid single donor fits even fewer anatomies |
| VMR neck stretch, C6 → C1 | W1 / W3 fail | pass at native thresholds (0.101 / 0.155), fail only with the disputed factors | borderline, not a clean fail |
| VMR left take-off at 10 mm | 62° | 56–57° with a local subclavian tangent | minor |
| SPL right bend radius | 2.19 mm | 1.40 mm, by end handling | G2 fails either way |
| SPL shipped models | 24 / 55 pieces | one piece each after cleaning; still unusable (r 0.85 mm) | none |
| Arterial benchmark | wall distance p95 0.42–0.89 mm | that is Arterial → CT only; CT → Arterial p95 reaches 2.9 mm at the 1.5 mm left C2 loop | discloses a corner cut |
| Arterial benchmark | 3 of 8 continuous labelled paths; 1.5 mm confluence +2.7 mm | only 1 of 8 ends on a mostly correct basilar; the +2.7 mm is set by construction (the traces never joined) | labels are less reliable than stated |
| Arterial on TotalSegmentator, yield | 21 neck VAs / 13 subjects / 8 pairs; 12 whole / 5 pairs | **15 / 11 / 4; 11 whole / 4 pairs** | the scale-up estimate drops (below) |
| Arterial wall vs noise | ρ 0.13 on the usable VAs | ρ 0.13 is over 40 sides; −0.17 on the 21 usable; union also scales with radius (ρ 0.54) | conclusion unchanged |
| ts edge jitter | 0.29–0.96 mm | overstated about 2× (noise is near white) | conclusion unchanged |
| graft gates | 223 / 223 pass every v4 gate | the origin seam is ungated: 22 / 223 > 60° (201 / 223 pass if gated) | a gate must be added |
| graft wall | W1 passes at 0.0971 | 0.0933 with exact ownership: **fail**; about 0.073–0.077 over 223 | v5 does not meet the VA standard |
| graft W6, A2 | W6 fails on 4 / 8; A2 holds | W6 fails on 5 / 8; one piece fails A2 (0.914) | minor |
| graft take-off | 99° / 82°, \|Δ\| 45.7° / 23.2° | 93.0° / 78.3°, \|Δ\| 39.5° / 19.6° (host chord fixed) | still not preserved for most VAs |
| strategy B over C | C needs a non-rigid warp | the tube-bridge variant of C was never tested | B is defensible, not proven best |
| load test | target on the RCCA, about 46–52 mm off | frame mix-up: the target is on the RCCA at s ≈ 181 mm; the wire never entered the new VA | the load test does not test the VA |
| TaG-Net bench: native failure with the exact VMR call | A's basilar lies 3–8 mm off the skull midline, outside find_ba's ±5 mm band; fix with a wider band | the basilar is mostly inside the band (74%). An ACA component wins find_ba's largest-component contest because the seed window reaches z_FM + 55 mm (docstring + 40). A wider band does not fix it; the + 40 mm window does. The midline margin with `--ts` is about 2 mm, not 3 | fix the seed window |
| TaG-Net bench: course accuracy | more accurate than on VMR (0.26–0.32 against 0.38–0.40 mm) | a reference confound: against each patient's own CT centreline the right sides are equal and the left sides differ by 0.03–0.05 mm | as accurate as on VMR, not better |
| TaG-Net bench: V1–V2 course | accurate at every quality, so TotalSegmentator donors could supply the cervical stretch | holds only with the VMR skull rule. With the TS skull at 1.5 mm (the donors' protocol): 0.53–2.12 mm mean, p95 up to 12 mm, and the right VMTK fails in 2 of 3 runs | no "origin → C2" donor class follows |
| TaG-Net bench: minor | off-VA paths up to 31 mm; left V4 recall 0–0.4 at 1.5 mm | up to 32.8 mm at 1.5 mm (about 35–44 mm in wrong native runs); with clean noise the left V4 reads 0.8 over the last 40 mm (an 8 mm gap); the 1.5 mm + clean right junction lies 4.6 mm off, outside the lumen | none |
| TaG-Net CT wall | in-plane edge as blurred as VMR's; noise about half; the right VA a plain tube from C2; left C1 at s 171.75 | in-plane 4–7% sharper; noise 0.55–0.63 × VMR's; the right V4 carries branch-origin relief (one bin 0.19); by VMR's convention the left C1 is at s 161, which raises C6 → C1 to W1 0.117, W3 0.149 | none: the claim holds, and its landmark choice was conservative |
| calibration: V3-loop gate | robust-cord excursion ≥ 12 mm and d ≤ 6 mm, backed by two patients | the floor was tuned and checked on A (16.6 mm without A); d ≤ 6 has no GT basis and rests on two TotalSegmentator sides; any floor in (4.8, 14.3] gives the same counts | only "≥ 15 mm is too high" is two-patient-backed; 12 and 6 are provisional |
| calibration: entry rule (Erec) | removes the C7 over-call; C6 landmark within about 1 mm | A selected the parameters; "always C6" scores the same on the GT rows; no GT VA enters at C7; the C6 landmark is still 10 mm early on VMR's right side | the over-call is real; Erec is a candidate, not validated |
| calibration: Stage-2 gate | recalibrated: 15 / 40, 11 subjects, 4 pairs, 0 of 6 wrong-vessel sides, automatically | reproduces, but in-sample: 5 of the 6 sides gained enter only through limits moved for them, and d ≤ 6 was set knowing s0518 L | needs an out-of-sample test on new TotalSegmentator subjects or pilot cases |
| calibration: radius | native VMTK MISR × 1.10 works on both patients (1.5% apart) | protocol-dependent: 1.10–1.21; matched on protocol, 1.11 against 1.21 | take calibre from the CT half-max at every resolution |
| calibration: gross ranges | DM 1.40–1.75; take-off 42–93°; donors near the host's 13–27° subclavian elevation include SPL | DM 1.32–1.75 with the TotalSegmentator VAs; the take-off range leaves out SPL R (106°) and the TotalSegmentator neck VAs (20–110°); SPL's elevation is 33° | minor; the host's take-off is inside either way |

What reproduced exactly:
- the v4 route-detail measure and every reference number for the host, TopBrain and v4;
- VMR's wall, gross shape, C2 loop and edge width;
- SPL's labels, gates and wall readings;
- every Arterial benchmark number;
- the prototype meshes, bitwise;
- for TaG-Net: the GT paths (within 0.15 mm), the Arterial benchmark (means within 0.04 mm), the CT wall and edge
  width (within 0.01), and the calibration's counts (exactly) and values (V3 values within 0.1 mm on all but one
  TotalSegmentator side, radius ratios to 3 decimals).

## Recommended v5 strategy

1. **Keep strategy B with three donor classes.** Every class takes the host subclavian byte for byte, then donor
   V1–V3 to a splice, then v4's TopBrain distal piece. The TopBrain piece carries the only VA wall that meets the
   standard.
   - **Real-wall donors** are sub-millimetre CTAs, segmented from the image by a CT-edge method like `ct_seg4`, seeded
     from an Arterial trace, with stage-A surfaces and VMTK. Today that means VMR 0248 (2 VAs), plus TaG-Net patient
     A (2 VAs) for internal calibration only, since its data has no licence.
   - **Course-only donors** are Arterial traces of TotalSegmentator CTAs, plus SPL. They are baked as tube along the
     donor centerline (`real_surface: false`). They bring real course, take-off, entry level, V3 loop and calibre, and
     no nooks.
   - **The TopBrain distal piece** stays as in v4, Kabsch-fitted to the donor's own distal course.
2. **Say plainly what v5's VA does and does not have.**
   - v5 keeps v3's nooks on the RCCA route exactly (ratio 1.000).
   - It improves the VA wall over v4 by about ×2–3.5.
   - It does not reach TopBrain parity over the whole VA. Record `va_detail` per anatomy (A1) and report the cervical
     wall as a measured gap.
3. **Raise wall density where it is cheap.** This is my inference, untested. The set median is driven by TopBrain piece
   length: the prototypes with 65 mm median pieces read 0.093, the full set (48 mm) about 0.073. Preferring long
   TopBrain pieces (≥ 60 mm) at the splice, and placing the splice where the donor's V3 is smooth anyway, should move
   W1 toward parity. Test it in the dry run before relying on it.
4. **Do not build all 223 from VMR 0248.** That would put one patient's cervical wall over half or more of every VA.
   Use VMR as the pipeline template and the first registry donor.
5. **Prefer real-wall donors on the RVA.** The RVA is reachable from the training insertion; the LVA is collision
   geometry until a loader inserts from the aorta.

## What v5 at scale needs

### Fixes before scaling (all in the scratch tools; no repo edits needed to test them)

| area | fix | why |
|---|---|---|
| grader (`va_standard.py`) | re-derive the W1–W3 resolution factors under V6, or gate by measured edge blur; fail closed when the voxel is unknown; take the voxel from the acquisition, never from Arterial's 0.5 mm grid | the factors are wrong-signed; the gate fails open |
| grader | implement W7 and G5; decide per-VA or set for W7; use exact ownership (k ≥ 64) in the VA measure; report the seam zone separately; relax G2 and G4, or make them review flags; fix one C1 definition (the exit from the C1 transverse foramen) before grading C6 → C1 | k = 24 put a false pass on W1. G2 and G4 reject real anatomy (4 and 1 of the 6 GT VAs). TaG-Net's C6 → C1 passes or fails on the C1 definition alone |
| grader | settle the bar (gate-valid or all-piece), the A2 per-piece rule, and always pair W3 with W1 | user decisions below |
| graft (`graft_v5_va.py`) | gate the origin-seam turn at ≤ 60° | 22 / 223 fail today |
| graft | fix the host parent-tangent chord to 5 mm and re-record the take-off flags | k_004 R is mislabelled |
| graft | add a plausibility gate: course RMS to the corridor (corridor fit median 9.6 / 9.4 mm, maximum 30; the absurd placements start at 21.5 mm), tip distance, take-off band | clearance gates pass absurd placements |
| graft | add a splice calibre rule (TopBrain / donor r up to 2.3; the ramp widens the donor's last 25 mm up to 2×) | |
| graft | generalise from the single VMR donor to a donor registry; test the tube-bridge strategy C | |
| graft | drop the plan's Stage-6 parent-frame take-off recipe; record each donor's subclavian elevation and prefer donors within about 15° of the host (26.5° R, 13.2° L), which is untested | |
| extraction (`art_extract.py`) | add the V3-loop gate to the cervical gate, on `total_v3` masks (posterior excursion ≥ a floor below 14.3 mm, about 12, and ≤ about 5–6 mm from the posterior arch of C1), require C1 threading, send every bridged side to review | removes the 6 wrong-vessel VAs and all 68 wrong TaG-Net A traces. The ≥ 15 mm floor rejects TaG-Net A's real left loop (14.3–15.0 mm). The values 12 and 6 are provisional: they were tuned on A and on two TotalSegmentator sides |
| extraction (donor-pool plan Stage 2) | make rule 6 (Arterial label agreement) a review flag; replace ts_qc's entry call with a run rule like Erec; take calibre from the CT half-max profile, never from a radius factor or the trace EDT; test the calibration's other limits (V1 breaks ≤ 3 mm, no IQR term, gaps counted below the splice only) out of sample before adopting them | the draft gate keeps only 9 of the 40 sides and loses 6 of the 15 usable; rule 6 rejects 3 correct GT-patient traces; ts_qc calls C7 on 9 of 10 GT paths, though all 4 GT VAs enter at C6; the radius factor varies from 1.05 to 1.30 with patient and protocol |
| extraction | set `find_ba`'s seed window to the documented z_FM + 40 mm; run the `--ts` QC on 1.5 mm copies of thin-slice masks; treat ts_qc's C2 (7–12 mm late) and C1 (26–40 mm late) as uncertain | an ACA seed sent both native TaG-Net traces to the wrong vessel; 0.4 mm masks need 17.7 GB and 18 min, 1.5 mm copies 1.4 GB and 47 s for the same QC within about 1 mm |
| extraction | choose the lowest basilar component, or keep the cervical stretch only | 22 sides fail the V4 ≤ 50 mm gate |
| extraction | patch Arterial's arch check (recovers s1163, s0152, s1113); add variant starts (arch-origin left VA, aberrant right subclavian); exempt or smooth G2 / G3 at 1.5 mm (noise alone dropped VMR's q_min from 0.71 to 0.45) | |
| load test (`load_test_v5.py`) | check the PASSED line, not the exit code (SOFA segfaults at teardown); fix the tracking / vessel frame mix-up; add a schedule that enters the RVA | the current test never exercises the new VA |

### Donor counts

| source | VAs now (verified) | realistic with the fixes | wall class |
|---|---|---|---|
| VMR 0248 | 2 (4 shapes with follow) | 2 | real CT wall, ≥ 1 mm relief; left V1 as tube |
| TaG-Net patient A (+ B, centrelines only) | 2 measured, not grafted | 2, for internal calibration only until the authors grant terms; never in a shipped anatomy | real CT wall, VMR-like (≥ 1 mm relief at contacts); no vein contact, so no V1 tube |
| SPL | 2 (not grafted) | 2 | tube |
| TotalSegmentator via Arterial | 15 neck VAs, 4 pairs, from 24 subjects tried | **≤ about 85 sides and ≤ about 23 pairs** if all 139 full-course subjects yield like the 24 best-ranked (my extrapolation; lower tiers and sharp kernels yield less: 3 of 16 noisy sides). The plan's ~140 sides / ~45 pairs is about 2× too high for the current code | tube |
| open thin-slice sets | 0 | plan: ~400 sides before de-duplication, unverified; needs a 25-case pilot (go if ≥ 60% of sides pass) | real wall only with CT-edge segmentation, at best VMR-like |
| RSNA-ICA (gated) | 0 | plan: ~950 sides | non-commercial, internal only |

For scale: 223 anatomies need 446 VAs. v4 reuses each TopBrain VA about 9×. With at most 2 uses per donor, v5 needs
about 112 pairs; with v4-like reuse, about 25 pairs. Even at the top of its range (about 23 pairs), TotalSegmentator
alone falls short of the second figure.

### Compute on the local RTX A5000 (one Arterial container at a time)

| step | measured per unit | tranche | wall time |
|---|---|---|---|
| Arterial (GPU, TTA off, core stages) | median 168 s (121–300) per subject | ~117 TotalSegmentator subjects not yet run | ~5.5 h GPU |
| trace extraction (CPU) | median 164 s (up to 818 s on 500 mm volumes) | same | ~5.5 h; runs beside the GPU container |
| VMTK + metrics | ~15 s per side + 15 s per VA | same | ~2 h |
| **TotalSegmentator remainder, total** | | | **~13 h serial, ~7 h with extraction overlapped** (plan: ~14 h) |
| thin-slice pilot (10 Song, 10 172-series, 5 AortaSeg) incl. TotalSegmentator `total` masks | ~10 min per case | 25 | ~4 h + ~5 GB download |
| thin-slice bulk | same | ~255 | ~38 h + ~50 GB on D: |
| CT-edge wall segmentation for real-wall donors | not timed for VMR | per donor | to be measured |
| graft feasibility dry run | 23–53 s per donor over 223 anatomies | 200 donors | ~2–3 h CPU |
| bake (`bake_v5.py`) | 2–3 min per clean anatomy; ladder cases up to 44 min | 223 | ~9–10 h on 5 shards, as v4 |
| SOFA load test | ~1 min per anatomy | 223 | ~4 h |

CPU-only Arterial takes about 10 min per case (12 threads), so about 28 h for the TotalSegmentator remainder. C: has
about 41 GB free. Keep every download and output on D: under `carotid_data/va_sources/`, and delete each 136 MB
resampled `cta.nii.gz` after extraction.

## Decisions only you can make

1. **What "keeps v3's nooks" means for the VA.**
   - (a) TopBrain parity over the whole VA, which no available source reaches;
   - (b) parity only where a real-wall reference exists (the distal TopBrain stretch), with the cervical wall reported
     as a measured gap;
   - (c) "beats v4 by a stated factor".

   Also: should cervical relief that exists only at ≥ 1 mm scales and at bone or vein contacts count as nooks?
2. **The bar.** Gate-valid TopBrain (W1 0.0966, equal to 0.85 × what v4 ships) or all pieces (0.1074). Per-VA or set
   for W7. Whether v4's own A2 failures (7 of 20) are acceptable for v5.
3. **Ship v5 now with VMR only, or wait for the registry.** VMR-only v5 gives one patient's cervical VA in every
   anatomy. My recommendation is to wait, and use VMR for a small pilot set.
4. **Wall policy for 1.5 mm donors.** Tube along the real course (recommended), or their stage-A surfaces, which carry
   artefact detail.
5. **Handedness.** Follow (recommended by the graft study: consistent with v4, and twice the shapes) or native.
6. **Take-off.** Accept "not preserved", flagged per VA; select donors by subclavian elevation; or bend V1 non-rigidly
   and bake it as tube.
7. **Licences.**
   - Every Arterial-derived VA is non-commercial, and any v5 anatomy that contains one inherits that.
   - TaG-Net needs the authors' data terms before anything derived from it ships (yaolinlin23@sjtu.edu.cn).
   - RSNA needs your Kaggle account and cannot be redistributed.
8. **Donor reuse.** At most 2 uses per donor (about 112 pairs needed), or v4-like reuse.
9. **Priority side.** Spend real-wall donors on the RVA, the training-reachable side, first.

## Open risks

- **Two benchmark patients, one unlicensed.** VMR 0248 and TaG-Net patient A now both underpin the Arterial accuracy
  and the CT-wall expectations. The graft's take-off analysis still rests on VMR, which has the steepest subclavian of
  the four patients measured. The recalibrated thresholds rest on 4 GT VAs, several of them on one patient or on the
  TotalSegmentator sides they were tuned on, and none has had an out-of-sample test. No GT VA enters at C7, and
  trusted landmarks exist only for VMR's right side and patient A (±3 mm). TaG-Net cannot ship.
- **Modality.** TopBrain's distal detail (0.3 mm TOF-MRA) may partly be MRA artefact, while VMR's CT (edge FWHM
  1.4 mm) may blur real V4 relief. The bar and the best candidate disagree for reasons that one patient cannot settle.
- **Thresholds at noise level.** The thresholds sit at TopBrain parity, so marginal results are within noise:
  TopBrain re-imaged at 1.0 mm reads 0.0585 on W4 sections, against a factor-adjusted bar of 0.053. The v5 W1 verdict
  crossed the line through ownership k and the seam convention alone.
- **Yield.** The corrected TotalSegmentator yield rests on 24 subjects. Thin-slice yields and Song's 0-HU clipping are
  untested. Series and patients may be duplicated across the 172-series, Song, Large IA and RSNA sets.
- **Wrong-vessel traces.** Without the V3-loop gate, a quarter of the bridged TotalSegmentator traces would put an ICA
  segment into a VA.
- **Mixed phase.** Polytrauma donors carry an enhanced vertebral venous plexus. No contamination was found on the
  usable sides, but the phase gate and a visual sample remain necessary.
- **SOFA.** Teardown segfaults, non-deterministic routes, and a BeamAdapter retraction crash shared by v4 and v5.
  None is a mesh defect, but a batch runner must allow for them.
- **The prototype take-off fields were hand-patched** and lack some recorded fields (`course_rms_to_corridor_mm`,
  elevations). Regenerate the provenances from the fixed `graft_v5_va.py`.
- **Housekeeping.**
  - A stray zero-byte file `C:\Program Files\Git\tmp_patch.awk` (from the vmr study) can be deleted.
  - `derived/arterial/runs/vmr_native/cta.nii.gz` (134 MB) can be deleted.

## Files

Scratchpad root: `C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\`.

- Notes (`v5_va\notes\`):
  - studies: `ref.md`, `data.md`, `vmr.md`, `spl.md`, `ts.md`, `arterial_infra.md`, `art.md`, `graft.md`;
  - verifiers: `verify_metrics.md`, `verify_sources.md`, `verify_arterial_bench.md`, `verify_arterial_yield.md`,
    `verify_graft_mesh.md`, `verify_graft_wall.md`.
- Results (`v5_va\results\`):
  - `standard.json`, `reference_metrics.json`, `vmr0248_metrics.json`, `spl_metrics.json`, `arterial_metrics.json`,
    `graft_proto.json`, `graft_proto_detail.json`, `ts_candidates.tsv`;
  - verifier outputs: `verify_*.json`, `verify_sources\`;
  - also `v5_va\vgw\` (graft-wall verifier), `v5_va\verify_graft_mesh_out\` and `v5_va\verify_arterial_yield_out\`.
- Tools (`v5_va\`): `va_metrics.py`, `va_standard.py`, `graft_v5_va.py`, `bake_v5.py`, `va_detail_v5.py`,
  `load_test_v5.py`, `run_arterial.sh`, `arterial_ts_args.py`, `art_extract.py`, `vmr_ct_seg3.py`.
- Donor-pool plan: `va_pool\donor_pool_plan.md`.
- Data (`D:\Arjun\neve\carotid_data\va_sources\`, gitignored): `vmr0248\`, `spl_atlas\`, `totalseg_v3\`,
  `arterial\`, `tagnet\`, `derived\` (vmr0248 `ct_seg4`, spl, arterial traces and QC), `v5_proto\` (the 4 baked
  prototypes, with undecimated unions in `_iso\`).
- Companion reports: `reports\Public cervical vertebral artery datasets.md`,
  `reports\Chinese cervical vertebral artery datasets.md`.

Licence notices to keep with any derived anatomy:
- VMR 0248: Wilson et al. 2013 (doi 10.1115/1.4025983), NLM R01LM013120 / NHLBI HHSN268201100035C.
- SPL atlas: 3D Slicer licence B; Jakab and Kikinis 2015, P41 EB015902.
- TotalSegmentator v3: CC BY 4.0; Wasserthal et al. 2023, doi 10.1148/ryai.230024.
- Arterial: weights CC BY-NC 4.0, code PolyForm Noncommercial.
- TaG-Net: no data licence; internal use only.

Session note: the graft and Arterial studies were cut off by a session limit at 04:29–04:31 and resumed at about
10:15. No file was lost, but unfinished work was: the Arterial study's measurements, notes and verdict had not been
written yet (its measuring script had crashed before saving), so the resumed agent redid them and re-ran every trace
extraction on the final code. The graft study had its bakes and notes on disk; after the resume it added the w_039
load test and the take-off analysis. This report was written from scratch after the reset; no earlier draft existed.
