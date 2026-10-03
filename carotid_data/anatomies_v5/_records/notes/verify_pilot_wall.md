# verify_pilot_wall: adversarial check of the pilot's wall results and donor ranking

Agent prefix `verify_pilot_wall`, first instance, 03:00–04:25 EDT 2026-10-02. State file
`/sp/v5_va/state/verify_pilot_wall.md`. No git, no GPU, no downloads, and no tracked file touched. Every input was only
read. All code is this verifier's own:
- `verify_pilot_wall_tables.py` (python:3);
- `verify_pilot_wall_union.py`, `verify_pilot_wall_collect.py` and `verify_pilot_wall_zres.py` (neve-build-stagea, CPU);
- `verify_pilot_wall_bins.py` and `verify_pilot_wall_peek.py`, which print per-bin values and input structure.

Outputs are in `/sp/v5_va/verify_pilot_wall_out/`: one JSON per job, plus `tables.json`, `collect.json`, `zres.json`
and the logs.

**Diff of this file:** the 03:30 "Answer so far" (3 lines) is replaced by the final answer below.

## Answer

Status: complete (04:25).

**Verdict:** the claim holds for the numbers, and for the ranking as computed under its stated rule. Every key wall
number reproduces with independent code to within 0.007. The within-source rankings and pair scores reproduce exactly
from the per-case files.

The claim overstates how robust the ranking is, and one convention is violated:
- The Song order changes under a different contamination-exclusion method.
- The bottom contact-free scores sit at the matched-noise floor.
- The "agreement" figures it cites are close to circular.
- The Song Siemens cases' "acquisition voxel" is the reconstruction grid, not the 1.0 mm slice.

### 1. Own union and measure on all 30 recalibrated-gate walls

**The method, written from the v4 definition** (no va_metrics, sdf_* or pilot code imported):
- The tube field is max(r − |x − c|).
- The capsule is 1.8 r + 1 mm, tapered over the first 8 mm.
- The real surface is the signed distance of the stage-A surface. Ownership is the self-contact margin of 0.5 mm with a
  12 mm window.
- Marching cubes at 0.45 mm; the largest component is kept.
- Each triangle is owned by the sample with the largest r − d among its k nearest.
- dep = | |x − c| − r |. "Core" drops 2 mm at each end, as the pilot's numbers do.

**Agreement, own minus claimed, over the 30 sides:**

| quantity | max abs diff | Spearman |
|---|---|---|
| whole u03 | 0.0024 | 1.00 |
| whole u05 | 0.0014 | 1.00 |
| C6 → C1 (ct) | 0.0066 | 1.00 |
| donor stretch | 0.0029 | 1.00 |
| contact-free donor (pilot rule, own bins) | 0.0070 | 1.00 |
| edge FWHM (own erf fit) | 0.036 mm (median −0.004) | 0.98 |
| plain-tube fraction | 0.08 (1–2 bins) | 0.95 |

- k = 24, 64 and 128 give identical values on every side.
- My noise estimator (voxel std within 0.5 r) reads systematically lower than edge.json. It uses a different
  definition, and the pilot's conclusions do not depend on it.

**Set statistics from the per-case files, recomputed:**
- Whole-VA medians: 0.1595 / 0.0862 / 0.143.
- C6 → C1: 0.217 / 0.127 / 0.194.
- Contact-free donor: 0.092 / 0.062 / 0.072.
- Plain-tube fraction: 0.103 / 0.25 / 0.167.
- FWHM: 1.115–1.406 / 0.951–1.347 mm.
- 28 of 29 walls beat 0.027; 190 L reads 0.026.
- 5 sides have a contact-free donor below 0.027.
- Range ×35.
- W1 / W2 / W3 set flags as claimed.

**Small mismatches, none in a ranked pair:**
- 140 L donor: the decide table uses the approximate bins value 0.320, but `measure.json` has the exact rec stretch,
  0.272.
- 001 R and 140 L C6 → C1 come from ts bins (0.239 / 0.237); the exact ct stretch reads 0.26 / 0.31.
- Song 106 L contact-free reads 0.141 (own range) against the claimed 0.134.

**Frames and sides:**
- Centreline HU is ≥ 357 on all 30 walls.
- On all 11 pairs the RVA lies on the subject's right (RAS x of R > L).
- No wrong frame, wrong side or wrong-vessel wall was found.

### 2. Noise-floor phantom (061 R, 140 L)

**Reproduced.** Own union on the pilot's phantom surfaces: 061 R 0.0008, ×2 noise 0.0184, 140 L 0.0052. The real walls
read 0.2437 and 0.2131.

Own segmentation-free image relief (share of clean rays whose half-max edge lies > 0.3 mm off the station median):

| | real | phantom | ×2 phantom |
|---|---|---|---|
| 061 R | 0.104 | 0.000 | 0.004 |
| 140 L | 0.121 | 0.001 | |

**Caveat: the ×1 phantom is under-noised at the scale that matters.**
- Its noise was Gaussian-filtered with σ = 2–3 voxels, to match a lag-1 autocorrelation of 0.94–0.97 measured in the
  subclavian lumen. That autocorrelation is image structure, not noise.
- Local lumen voxel std (0.5 r ball): real 061 R 13.8 HU, phantom ×1 8.5 HU, ×2 14.9 HU.
- The matched-noise floor is therefore about 0.018 (the claim's "0.019 at twice the noise"), not 0.0008.

The relief conclusion still holds for the coarse sides (×13). But the contact-free scores of 141 L (0.012), Song 109 L
(0.009), 008 R (0.010), 141 L (0.019) and 032 R (0.022) sit at that floor, so the order among the bottom pairs is not
meaningful.

### 3. Contamination and the ranking

**What the pilot rule removes:**
- The rule drops every 10 mm bin with ≥ 5 % kind3 + kind1 rays. In practice that removes nearly all V2 bins (bone of the
  foramina).
- The score is then mostly V1, plus a few V2 / V3 bins.
- Song 008 R keeps 6 % of its stretch.

**Own per-cell detector.** HU is sampled 0.75–3 mm radially beyond each union triangle. A triangle is "bright" if the
HU rises above the station half-max, and "dim" if it exceeds I_bg + 0.25 ΔI.
- Relief that remains inside the pilot's "contact-free" bins is still bright-adjacent: median 0.23 on the 172-series
  (0.10–0.45) and 0.16 on Song (0–0.49).
- A further 0.18 / 0.10 is dim-adjacent.
- So contamination is reduced, not excluded.
- Over the whole wall, 54 % / 55 % of the relief area is bright-adjacent; the pilot's per-bin attribution says
  69 % / 76 %.

**Ranking under the stated rule:** it reproduces exactly.
- 172-series: 058, 039, 015, 190, 141.
- Song: 113, 100, 094, 008, 032, 109.

**Ranking under per-cell exclusion** (bright triangles removed everywhere, no bin dropping):
- 172-series: 058, 015, 039, 190, 141 (Spearman 0.9 against the stated rule).
- Song: 008, 113, 100, 109, 032 (Spearman 0.6). 008 moves from 0.047 to 0.187 (min side 0.186), and the dim-excluded
  value is 0.161.
- The median per-cell donor value is about 0.097, against the pilot rule's 0.072.
- The 172-series top (058) is robust. The Song top is not.
- The "coarse enough on both sides" set changes too: 008 and 109 come in, while 141 and 032 stay out.

**The agreement figures cited in the claim:**
- Plain-tube 0.98 / 0.94 is computed on the same contact-free donor bins as the score, so it is near-circular.
- `pilot_donor_ranking.json` also records a C6 → C1 contact-free agreement of only 0.3 for the 172-series. The claim
  does not mention it.

**Scanner / kernel:**
- Every 172-series case is Toshiba Aquilion / PRIME, FC43, 0.5 mm. Within-source order therefore cannot be a kernel
  difference, but the 172-series versus Song coarseness gap cannot separate kernel from anatomy.
- The merged order alternates sources by construction.
- The Song top pair 113 is Siemens DE (Qr40, 90 kV); #2 is 100 (Philips IMR1); #3 is 094 (Siemens Bv40).

### 4. Acquisition voxel (Song)

**The recorded voxel is the grid, not the slice.**
- The Song meta records the grid spacing as the "acquisition voxel".
- The Siemens series 094, 109 and 113 are named "CarotidAngio 1.0 Bv40 3" and "DE_CarotidAngio 1.0 Qr40 3": Siemens
  naming for a 1.0 mm slice thickness, reconstructed on a 0.7 mm grid.
- The Song sidecars carry no SliceThickness. The 172-series correctly uses the DICOM thickness, 0.5 mm.

**Measured z blur (own erf FWHM split by ray inclination):**

| source | z / in-plane FWHM ratio | z FWHM |
|---|---|---|
| 172-series (0.5 mm, calibration) | 0.74–0.82 | |
| Philips 0.45 mm grid | 0.92–1.00 | |
| GE 0.625 mm grid | 0.99–1.08 | |
| Siemens 0.7 mm grid | 1.09–1.25 | 1.20–1.49 mm |

The Siemens z blur is consistent with 1.0 mm slices.

**Effect:**
- No union value changes.
- The "acquisition voxel" convention is not met for 3 of 10 Song cases, including Song's #1 and #3 pairs.
- Even the recorded 0.70 mm exceeds va_standard's 0.65 mm native limit, so native bars apply only by the brief's
  convention.
- The possible resolution effect on 113 / 094 coarseness is untested.

### 5. Song air1000 re-window and clipping (113 R / L, 100 R)

- The ct_seg4 wall uses the shipped 0-clipped CT. air1000 is identical to negzero near the VA.
- On identical rays, the half-max edge shift is 0 at the median; 0.2–3 % of rays move > 0.1 mm, and ≤ 0.06 % move
  > 0.3 mm.
- Own union on `ct_seg4_negzero` against `ct_seg4` is +0.002 (113 R 0.1587 vs 0.1566; L 0.1634 vs 0.1613).
- FWHM is 0.03–0.06 mm larger on negzero, so the clipping slightly understates Song FWHM.
- No material effect on the wall.
- The 172-series clip [−1024, 3071] is applied to the Arterial input only; ct_seg4 reads the raw `ct.nii.gz`.

## Open issues

- The per-cell detector is my own heuristic: radial sampling only, 3 mm reach and a half-max threshold. It is not
  validated. It shows the ranking's sensitivity to the method, not which method is right.
- The resolution effect of 1.0 mm Siemens slices on union detail was not simulated.
- W7 was not re-checked.
