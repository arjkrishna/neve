# ref: the shared VA metric suite and the reference measurements

Agent prefix `ref`. All paths below are under the scratchpad `v5_va/` (container `/sp/v5_va/`).
Nothing in the git repo was modified; the repo, `carotid_data/anatomies_v4`, `topbrain_data`
and `eve_bench` were read only.

## What was built

| file | what |
|---|---|
| `va_metrics.py` | the metric suite. `measure_va(points, radii, surface, voxel_mm, ...)` returns gross shape, calibre, native wall detail, wall cross-sections, centerline QC and the v4-protocol union detail. Every metric is defined in `METRIC_DEFINITIONS`, which is copied into every result. `python3 va_metrics.py --selftest` checks it on synthetic curves and tubes. |
| `va_standard.py` | `grade(result)` and `grade_set([results])` score a source against `results/standard.json`. |
| `ref_measure.py` | references a (host), b (TopBrain pieces), d (v4 route detail) |
| `ref_v4_union.py` | reference c: rebuilds the exact v4 union of 10 anatomies and measures the VA host / ramp / donor stretches |
| `ref_vmr0248.py` | reference e: VMR 0248 through the suite (uses the shared `sources.load_vmr0248`) |
| `ref_resolution.py` | calibration: the TopBrain VA masks re-imaged at 0.75 / 1.0 / 1.5 mm, then measured the same way |
| `ref_bins.py`, `ref_summarize.py` | per-40 mm-bin detail; pools everything into `results/reference_metrics.json` and derives `results/standard.json` |
| `ref_grade.py` | grades every reference with `va_standard.py` -> `results/reference_grades.json` |

Runs in `eve-training-fixed` (`ref_resolution.py` in `neve-build-stagea` for nibabel). Mounts as in the task;
`union_detail` imports `sdf_mesher` / `sdf_union_v4` from `/opt/eve_training/topbrain_tools`.

Self-test (`results/selftest.json`):
- helix κ 0.0804 / τ 0.0406 (want 0.08 / 0.04);
- sine: 5 of 5 inflections;
- plain tube: 0 wall detail;
- a +0.6 mm bump over 10% of the circumference reads 0.061 native and 0.046 through the union, against about 0.068 geometric;
- a 3 mm 30% narrowing is found at W = 5 and 10 mm;
- a pure tube through the v4 union reads 0.000, so the measure has no floor of its own.

## How the metrics work (short; full text in `METRIC_DEFINITIONS`)

- **Input.** A VA as centerline + radii ordered origin → distal, optionally a wall surface and the voxel size.
  Resampled at 0.25 mm (= `sdf_union_v4.SAMPLE_MM`).
- **Gross shape.** Length, chord, DM = length/chord, and SOAM. ICM uses principal-normal flips with bends on
  both sides. Curvature and |torsion| p50 / p90 / max, the minimum bend radius, and turning (max in any 30 / 60 mm).
  Loops (≥ 360° in 60 mm) and U-turns (tangent reversal ≥ 150° in 30 mm). All at a stated Gaussian smoothing:
  **σ = 3 mm is comparable across every source including 1.5 mm CT**; σ = 1.5 mm only between sources with
  voxel ≤ 0.75 mm. The ends are extended linearly before smoothing, so there is no end artefact.
- **Segments.** V1–V4 when landmarks (C6, C2, dura) are passed. Otherwise 40 mm bins from the origin and from
  the distal end, each with DM, κ, turning and median r.
- **Take-off.** Angle of the first 5 / 10 mm chord against the subclavian's distal tangent, and elevation
  above the horizontal.
- **Calibre.** Median, p10, p90, CV, IQR/median. Events: r over the *flank* median (the median over
  [s−W, s−1] ∪ [s+1, s+W]) below 0.8 or above 1.2 for ≥ 1 mm, at W = 5 and 10 mm. L/R dominance.
- **Wall, v4 protocol (`wall_union`).** This is the operational measure. The VA alone goes through
  `tube_field` + `add_real_sections_owned` on the 0.45 mm grid, as in v4. Then `right_route_detail`'s own
  measure, `| |x−c| − r |`, at > 0.3 / 0.5 mm and > 0.15 r / 0.25 r. It reports the genus, and the same
  measure per 40 mm bin. `measure_union_branch` applies it to any branch stretch of any union.
- **Wall, native.** On the source surface itself:
  - against the declared tube (the radius the union will see);
  - against a fitted plain tube (shape only);
  - a band-pass roughness spectrum (octaves from σ = 0.375 to 3 mm, each flagged resolved when σ ≥ the
    source's largest voxel dimension);
  - fractions after low-passing at σ = 0.75 / 1.5 mm.
- **Wall cross-sections.** Every 1 mm, the loop around the centerline. Reports r_eq, r_in, circularity, and the
  perimeter share > 0.3 / 0.5 mm off the equal-area and the inscribed circle. These are the same quantities as for
  SimVascular expert contours. `centre_offset_mm` is the centerline's **eccentricity**.
- **QC.** Centerline outside the wall, native surface topology.
- **`recentre()`.** Builds a medial centerline from the section centroids, with MISR radii (what VMTK gives).

## Reference results

The full numbers are in `results/reference_metrics.json` (`groups.*`). Medians unless stated; "union" means
`wall_union.core`.

### a. Host VAs: smooth tubes, no wall detail

The VA proper runs from the subclavian origin to the basilar confluence. On the host RVA branch the VA proper
starts at s = 26.8 mm, where branch (17) leaves it; the first 27 mm are the right subclavian.

| | RVA | LVA |
|---|---|---|
| length, DM | 210 mm, 1.39 | 209 mm, 1.42 |
| r median (p10–p90) | 1.67 (1.30–2.75) mm | 1.75 (1.56–2.27) mm |
| take-off at 10 mm / elevation | 85° / 60° | 81° / 81° |
| loops ≥ 360° in 60 mm | 1 | 1 |
| union > 0.3 mm | **0.0007** | **0.000** |

The host is not straight. Bins 80–200 mm from the origin turn 150–237° per 40 mm, with κ p90 up to 0.21 /mm:
it carries a V3-like loop. L/R is co-dominant (radius ratio 1.05).

### b. TopBrain VA donor pieces: what v4 accepts as a real VA wall

These are 50 pieces (25 patients × 2 sides). All are used by v4, 446 uses in total. They are MR label surfaces
(voxel 0.297 × 0.297 × 0.6 mm), made by marching cubes plus windowed-sinc smoothing (20 iterations, pass band 0.1),
with VMTK MISR centerlines.

| metric | p10 | p25 | median | p75 | p90 | use-weighted median |
|---|---|---|---|---|---|---|
| length (mm) | 33 | 39 | 70 | 80 | 88 | 63 |
| r median (mm) | 1.33 | 1.45 | 1.72 | 1.91 | 2.12 | 1.70 |
| DM | 1.16 | 1.22 | 1.77 | 2.01 | 2.14 | 1.72 |
| κ p50 σ3 (/mm) | 0.044 | 0.056 | 0.079 | 0.099 | 0.116 | 0.077 |
| min bend radius (mm) | 2.96 | 3.96 | 4.86 | 5.59 | 6.72 | 4.82 |
| **union > 0.3 mm** | 0.031 | 0.058 | **0.121** | 0.204 | 0.269 | **0.113** |
| **union > 0.5 mm** | 0.000 | 0.006 | **0.026** | 0.078 | 0.137 | **0.023** |
| union > 0.15 r | 0.062 | 0.114 | 0.172 | 0.240 | 0.283 | 0.160 |
| union > 0.25 r | 0.010 | 0.030 | 0.065 | 0.092 | 0.140 | 0.060 |
| sections: perimeter > 0.3 mm off r_in | 0.027 | 0.081 | 0.146 | 0.265 | 0.393 | 0.124 |
| sections: circularity | 0.962 | 0.971 | 0.976 | 0.982 | 0.984 | 0.979 |
| native vs fitted tube > 0.3 mm | 0.020 | 0.041 | 0.091 | 0.159 | 0.216 | 0.069 |
| roughness band 0.75–1.5 mm, RMS | 0.037 | 0.051 | 0.061 | 0.077 | 0.090 | 0.055 |
| roughness band 1.5–3 mm, RMS | 0.029 | 0.038 | 0.053 | 0.075 | 0.110 | 0.051 |
| centerline eccentricity (mm) | 0.048 | 0.058 | 0.072 | 0.091 | 0.109 | 0.068 |
| centerline outside label | 0 | 0 | 0 | 0.076 | 0.174 | 0 |

- **Topology.** The standalone union has genus 0 on 49 of 50 pieces; mr_025 LVA reads 1, a handle the union
  itself makes. The raw label surfaces carry handles on 6 of 50: genus 4 on mr_007 / mr_021 LVA, 1–2 on
  mr_008 / mr_012 / mr_015 LVA and mr_017 RVA. The union closes all six.
- **Pooled.** Area-weighted over all pieces: 0.181 (> 0.3 mm) and 0.073 (> 0.5 mm).
- **Length and radius matter.** Long pieces (≥ 60 mm, r 1.83) read 0.186 / 0.069; short pieces (< 60 mm,
  r 1.49) read 0.054 / 0.004.
- **Detail grows away from the skull.** Median > 0.3 mm per 40 mm bin from the distal end:
  - 0–40 mm (≈ V4): 0.083, n 50;
  - 40–80 mm (≈ V3 and the C1 loop): 0.230, n 28;
  - 80–120 mm: 0.372, n 5.

  Some of the V3 "detail" may be TOF-MRA artefact (loop signal loss, the venous plexus). It is still what v4
  accepted.
- **Dominance.** 9 co-dominant, 9 left, 7 right.

### c. The v4 VAs as shipped

The undecimated union was rebuilt exactly for 10 anatomies. On every one, `right_route_detail` equals
mesh_v4.json to 0.0000. Sample:

| anatomy | note |
|---|---|
| `case_m_022_left__topcow_mr_021_L` | carries VA suppressed stretches |
| `case_w_039_right__topcow_mr_021_L` | carries VA suppressed stretches |
| `case_w_050_left__topcow_mr_021_L` | carries VA suppressed stretches |
| `case_w_036_left__topcow_mr_022` | LVA donor baked as a tube |
| `case_k_004_left__topcow_mr_010`, `case_k_011_left__topcow_mr_013_L`, `case_m_022_left__topcow_mr_020`, `case_w_003_right__topcow_mr_011_L`, `case_w_007_left__topcow_mr_017`, `case_w_012_right__topcow_mr_024_L` | — |

Results on 20 VAs, both sides pooled:

| VA stretch | share of VA wall | union > 0.3 mm | > 0.5 mm | > 0.15 r |
|---|---|---|---|---|
| host, origin → cut − 25 mm (plain tube) | 64.5% | **0.0003** (p90 0.001) | 0 | 0 |
| ramp, cut − 25 → cut | 9.3% | 0 | 0 | 0 |
| TopBrain donor, cut → end | 26.1% | **0.125** (p10–p90 0.006–0.279) | 0.035 | 0.176 |
| whole VA | 100% | **0.021** (p10–p90 0.001–0.121) | 0.007 | 0.033 |

- **Donor in situ ≈ donor alone.** 0.125 in situ against a standalone median of 0.12, so ownership and taper
  cost nothing measurable.
- **Decimation bias.** On the 60k `collision_full.vtp` the donor reads 0.113, about 10% below the union.
- **Composite gross shape** (host + donor): length 205 mm, DM 1.43, κ p50 0.043.

**RCCA in the same unions** (the v3 standard, now also in radius-relative terms):

| | > 0.3 mm | > 0.5 mm | > 0.15 r | > 0.25 r |
|---|---|---|---|---|
| fork | 0.214 | 0.082 | 0.130 | 0.046 |
| siphon | 0.319 | 0.149 | 0.257 | 0.132 |

The TopBrain VA's size-free detail (0.17 > 0.15 r) sits between the fork's and the siphon's. Its lower
absolute numbers come from the smaller vessel.

### d. v4 right_route_detail, all 223 anatomies (the v3 standard)

| | p10 | median | p90 | mean |
|---|---|---|---|---|
| fork > 0.3 mm | 0.091 | 0.196 | 0.338 | 0.212 |
| fork > 0.5 mm | 0.030 | 0.097 | 0.211 | 0.113 |
| siphon > 0.3 mm | 0.250 | 0.322 | 0.426 | 0.327 |
| siphon > 0.5 mm | 0.092 | 0.156 | 0.229 | 0.159 |

### e. VMR 0248 (populated by the vmr agent; loaded with `sources.load_vmr0248`)

The CT is 0.4 × 0.4 × 0.7 mm. The walls are the simulation mesh's wall faces, lofted between expert contours.
Both VAs run from the subclavian to the merge: 218 mm (R) and 224 mm (L).

| | R as drawn (contour r_eq) | R recentred (MISR) | L recentred (MISR) | TopBrain median |
|---|---|---|---|---|
| centerline eccentricity | **0.31 mm** | 0.034 | 0.039 | 0.072 |
| union > 0.3 / 0.5 mm | 0.096 / 0.021 (MISR on the drawn path: **0.49 / 0.29**) | 0.079 / 0.028 | 0.085 / 0.016 | 0.121 / 0.026 |
| sections > 0.3 mm off r_in | 0.021 | 0.018 | 0.016 | 0.146 |
| sections circularity | 0.991 | 0.992 | 0.994 | 0.976 |
| native vs fitted > 0.3 mm | 0.289 (eccentricity) | 0.028 | 0.022 | 0.091 |
| band 0.75–1.5 / 1.5–3 mm RMS | — | 0.047 / 0.052 | 0.036 / 0.043 | 0.061 / 0.053 |
| min bend radius σ3 | 1.29 | **2.21** | **1.67** | 4.86 |
| r median, DM | 1.65, 1.39 | 1.46, 1.38 | 1.72, 1.45 | 1.72, 1.77 |

- **The expert contours are circles.** 47 + 51 "Contour"-type contours, circularity 0.991 / 0.994, no point
  more than 0.08 mm off its equal-area circle.
- **The lofted wall is smooth by construction.** Per 40 mm bin, its union "detail" sits only:
  - at the subclavian junction (0–40 mm: 0.16–0.18);
  - at a hairpin 80–120 mm from the origin on both VAs (0.21–0.27).

  **Every other bin reads 0.000–0.016.**
- **The hairpin.** Bend radius 1.3–1.7 mm, below the vessel radius. There the recentred right-VA MISR drops to
  0.62 mm (q_min 0.47). It is either the C2 knee or a loft fold, and only the CT can say which.
- **Take-off** at 10 mm: 61° (R) and 73° (L). Co-dominant, L/R 1.13.

## Three findings that change how sources must be measured

1. **Eccentricity reads as wall detail.** Every centerline-referenced wall measure counts the centerline's
   offset from the lumen centroid as wall standing off the tube. That includes v4's own `right_route_detail`.
   VMR's hand-drawn path is 0.31 mm off-centre, and with MISR radii it scores 0.49 (> 0.3 mm), four times
   TopBrain, on a wall whose sections are perfect circles. v4 never hit this, because VMTK centerlines are
   medial (0.07 mm). **Gate V1 + `recentre()`.**
2. **The radius method moves the union measure by 40%.** On the same 40 gate-valid TopBrain surfaces and
   centerlines, implicit-distance MISR instead of VMTK's radii reads union > 0.3 mm **x1.40** (0.114 -> 0.166)
   and > 0.5 mm x2.0. Median radii are only 1.6% apart; the difference is local dips at bends, which VMTK's
   interpolation smooths. Calibre events go the same way: 27 of 50 fail G3 with implicit MISR, against 4 with
   VMTK radii. The section and fitted-tube measures don't use the radius and agree (0.126 vs 0.121).
   **Gate V6: union rules need VMTK MISR** (`vmtk_centerline.py`, as in v4). My VMR numbers use implicit MISR,
   so they flatter VMR.
3. **The union measure can be fooled by artefacts.** A MISR deficit at a sharp bend, or a junction cap, scores
   as detail: VMR recentred reads 0.08, inside TopBrain's p25-p50, with every other 40 mm bin at 0.000. Hence
   the shape-only guard (W4) and the no-dead-stretch rule (W6).

## Resolution calibration (`ref_resolution.py`)

Each of the 50 TopBrain VA masks was re-imaged:
- isotropic h = 0.75, 1.0 and 1.5 mm;
- Gaussian PSF with FWHM = h, trilinear sampling, threshold at 0.5;
- surfaces made exactly as in stage A (marching cubes + windowed-sinc 20 / 0.1);
- MISR from that surface, on the piece's own VMTK centerline.

"native" re-runs the native mask through the same code as a control. Ratio = paired median of level / native
over gate-valid pieces.

| metric | native | 0.75 mm | 1.0 mm | 1.5 mm |
|---|---|---|---|---|
| pieces passing V1 + V2 | 36 / 50 | 35 | 33 | **3 / 49** |
| labels broken (centerline > 10% outside) | 11 | 12 | 13 | **20** (mr_013 rVA vanishes) |
| centerline eccentricity, median | 0.074 | 0.094 | 0.124 | **0.195** |
| r median, ratio | - | 0.94 | 0.91 | **0.78** |
| union > 0.3 mm, ratio | - | x1.18 | x1.44 | **x2.15** |
| union > 0.5 mm, ratio | - | x0.96 | x1.24 | x2.20 |
| union > 0.15 r, ratio | - | x1.37 | x1.76 | x2.65 |
| sections > 0.3 mm off r_in, ratio | - | **x0.49** | x0.51 | x0.67 |
| native vs fitted tube > 0.3 mm, ratio | - | x0.93 | x1.32 | x3.78 |
| band 0.75-1.5 / 1.5-3 mm RMS, ratio | - | x1.02 / x1.04 | x1.10 / x1.12 | x1.56 / x1.52 |

What this means:
- **Union fractions rise on coarser grids; they do not fall.** Partial-volume blur shrinks the MISR more than
  the section, so more wall stands off the tube. At 1.5 mm a VA 3-4 mm across is 2-3 voxels wide: its
  sections turn blocky (circularity 0.95) and the corners stand about r(sqrt2 - 1) off.

  **A 1.5 mm source would pass a naive nooks test on voxel artefacts.** Meanwhile the real non-circularity
  (sections) halves already at 0.75 mm.
- **TotalSegmentator v3 (1.5 mm), and Arterial run on it, cannot be graded on wall detail at all.**
  - Only 3 of 49 re-imaged TopBrain VAs keep a medial centerline inside the label.
  - The radius biases -22%.
  - 20 of 49 labels break.

  Such sources can be graded on gross shape (sigma 3 mm) and calibre (with the -22% bias) only. If the gross
  shape passes, their wall must come from elsewhere (e.g. a real wall grafted onto their centerline).
- **0.65-1.15 mm sources** (VMR CT 0.7; a resampled CTA) are graded with the W thresholds multiplied by these
  factors (`standard.json` `W_resolution_factors`). TopBrain re-imaged at 0.75 / 1.0 mm passes its own
  radius-free set rules under them, which self-checks the factors.

## The natural-anatomy standard (`results/standard.json`; graded by `va_standard.py`)

Each new VA is measured alone with `measure_va`. The thresholds are derived by `ref_summarize.build_standard`
from the references, and the "from" field in `standard.json` names the value each one rests on.

The TopBrain reference population is the pieces that pass V1-V4 with VMTK radii: 36 of 50, carrying 360 of
v4's 446 uses. Set medians are graded on gate-valid VAs only, so the standard is defined and graded on the
same population. *Set* rules apply to the median over all VAs a source would supply; *per-VA* rules to each
one.

### V: validity gates

The wall numbers mean nothing unless these pass.

| rule | metric | threshold | source |
|---|---|---|---|
| V1 | centerline eccentricity (sections) | <= 0.15 mm | TopBrain p90 0.109; else `recentre()` |
| V2 | centerline outside its wall | <= 0.10 | v4's known-limitation flag |
| V3 | sections without a closed loop | <= 0.31 | TopBrain p90 |
| V4 | genus of the VA's own union | 0 | 49 of 50 TopBrain pieces (mr_025 LVA: 1); raw labels carry handles on 6 of 50 that the union closes |
| V5 | voxel | <= 0.65 native; 0.65-1.15 x factors; > 1.15 mm wall not gradeable | calibration |
| V6 | radius | VMTK MISR for W1-W3 | finding 2 |

### G: gross anatomy (sigma = 3 mm)

| rule | metric | threshold | source |
|---|---|---|---|
| G1 | r median (MISR) | 1.09-2.25 mm | gate-valid TopBrain p10 / p90 +-10%. Flag a hypoplastic VA rather than reject it |
| G2 | min bend radius | >= 2.5 mm | TopBrain p10 2.96 (all) / 3.38 (valid); host 4.2 / 5.4 |
| G3 | q_min at W = 10 mm | >= 0.6 | TopBrain p10 0.65-0.72. VMTK radii; a documented stenosis is exempt |
| G4 | full-length source | 180-260 mm, DM 1.25-1.65 | host 1.39 / 1.42; VMR 1.38-1.45; v4 composite 1.43 |
| G5 | distal 80 mm | DM 1.14-2.13, kappa p50 0.041-0.117 /mm | TopBrain p10-p90 |

### W: natural wall ("nooks and crannies")

Applies only when V1-V4 pass and the voxel is <= 1.15 mm.

| rule | metric | per VA | set median | source |
|---|---|---|---|---|
| W1 | union > 0.3 mm (VMTK) | >= 0.016 | **>= 0.097** | p10; 0.95 x use-weighted median 0.102 |
| W2 | union > 0.5 mm (VMTK) | - | **>= 0.020** | 0.95 x 0.0215 |
| W3 | union > 0.15 r (VMTK) | >= 0.055 | **>= 0.142** | 0.95 x 0.149. Size-free; RCCA fork 0.130, siphon 0.257 |
| W4 | shape-only guard: sections > 0.3 mm off r_in | >= 0.023 | **>= 0.105** | 0.95 x 0.110. Centerline- and radius-free |
| W4 | shape-only guard: native vs fitted tube > 0.3 mm | >= 0.014 | **>= 0.063** | 0.95 x 0.066 |
| W5 | roughness band 0.75-1.5 mm RMS | - | >= 0.051 mm | 0.95 x 0.054 |
| W5 | roughness band 1.5-3 mm RMS | - | >= 0.045 mm | 0.95 x 0.048 |
| W6 | every 40 mm bin of the union | >= 0.01 | - | 1 of 83 TopBrain bins falls below; VMR lofted 3-4 of 6 bins at 0.000 |
| W7 | full-length source: union > 0.3 mm, distal 0-40 / 40-80 mm | >= 0.078 / >= 0.218 | - | 0.95 x TopBrain bin medians (~ V4 / ~ V3) |

- **V1-V2 have no real-wall reference anywhere.** TopBrain stops near C1-C2, and VMR is lofted, so V1-V2 is held
  to W1-W6 over the whole VA.
- **Per-VA failures are tolerated at the reference's own rate.** TopBrain fails some per-VA rule on 26 of 50
  pieces: V2 10, G1 5, the p10 floors about 5 each.

### A: grafted anatomy (v5 bake; as v4 did for the RCCA route)

- **A1.** Record `va_detail` (host / ramp / new segment, each frac > 0.3 / 0.5 mm and > 0.15 / 0.25 r) on the
  undecimated union, beside `right_route_detail`. Use `va_metrics.measure_union_branch`.
- **A2.** In situ >= 0.95 x the same piece measured alone. v4 TopBrain donors: 0.125 in situ vs about 0.12
  alone, so this holds.
- **A3.** The whole VA must beat v4 as shipped (0.021), and should reach W1's set level once V1-V3 are real.
- **A4.** Every v4 gate stays:
  - 0.6 mm clearance, including ramp-widened stretches and each branch against itself;
  - joins <= 60 deg, calibre ramps, frame matching;
  - handedness from `provenance.json`;
  - owned union, genus-0 tree, clean decimation to 60k / 23.5k.

### How the references grade (`ref_grade.py` -> `results/reference_grades.json`)

| reference | grade |
|---|---|
| host VAs | **fail** every W rule (a plain tube) |
| TopBrain (37 gate-valid) | **passes every set rule**: W1 0.105, W2 0.021, W3 0.159, W4 0.123 / 0.066, W5 0.055 / 0.048 |
| VMR 0248 as drawn | **fails V1** on both VAs (0.31 / 0.37 mm), so every W rule is n.a.; also fails G2 |
| VMR 0248 recentred | x0.75 factors; W1-W3 n.a. (not VMTK radii) |
| TopBrain re-imaged at 0.75 / 1.0 mm | passes W4 / W5 under the factors (self-check) |
| TopBrain re-imaged at 1.5 mm | not gradeable |

VMR 0248 recentred fails:
- **W4 sections** (0.017 vs >= 0.051);
- **W4 fitted** (0.025 vs >= 0.058);
- **W5 0.75-1.5** (0.041 vs >= 0.052);
- **W6** (dead bins);
- **G2** (bend radius 2.2 / 1.7 mm);
- **G3** on the right VA (0.47).

**VMR 0248's lofted walls do not meet the natural-anatomy standard; they have no nooks.** Its CT
(0.4 x 0.4 x 0.7 mm) is within the gradeable range: segmenting the VAs from the CT, with stage-A surfaces and
VMTK centerlines, is the way to test this patient. Its paths and contours are still a ready benchmark for the
gross shape and calibre of an automatic method's VA. Its hairpin, at 80-120 mm from the origin, has to be
checked against that CT.

## Which comparisons are resolution-limited

- **Union and native wall fractions.** Native thresholds apply at <= 0.65 mm; 0.65-1.15 mm needs the
  calibrated factors; above 1.15 mm they are not comparable. They rise with voxel size, and the real
  non-circularity is lost first.
- **Roughness band 0.375-0.75 mm.** Unresolved for every candidate (TopBrain z 0.6, VMR CT z 0.7,
  TotalSegmentator 1.5).
- **Band 0.75-1.5 mm.** TopBrain and VMR CT only. It holds half of TopBrain's resolved wall energy.
- **Band 1.5-3 mm and the sigma = 1.5 low-pass.** In principle the only wall comparison open to a 1.5 mm CT.
  TopBrain's own detail barely survives the low-pass (fitted > 0.3 mm: median 0.010), and the band is inflated
  x1.5 by voxel corners, so it decides nothing.
- **Curvature.** sigma = 3 mm for every source (2 voxels of a 1.5 mm CT); sigma = 1.5 only at <= 0.75 mm. The
  calibration held each centerline fixed, so it does NOT test how a 1.5 mm centerline itself degrades:
  benchmark that against the VMR 0248 paths.
- **Calibre.** Compare MISR with MISR, from VMTK. Equal-area and fitted radii run 6-8% larger (TopBrain
  declared/fitted 0.94; VMR MISR 1.46 vs contour r_eq 1.65). Partial volume shrinks radii by 6% at 0.75 mm,
  9% at 1.0 and 22% at 1.5.

## Open points

- **c's sample.** Three of the 10 sampled anatomies share the mr_021_L donor (the only VA suppressions in v4),
  so per-anatomy spread is under-sampled. The host, ramp and donor numbers agree with b.
- **V1-V2 reference.** No public real-wall reference exists below C2. W1-W6 over the whole VA is the best
  available test.
- **TopBrain V3 detail** may partly be TOF-MRA artefact (loop signal loss, the venous plexus). It is what v4
  accepted, so it is the bar.
- **VMR hairpin**: C2 knee or loft fold? Check against the CT.
- **SPL atlas** was not measured here; another agent owns it. For grading: its 256 x 256 source is coarse, so
  check V5 first.
