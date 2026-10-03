# verify_v5_wall: adversarial check of the v5 bake, wall-detail lens

Prefix `verify_v5_wall`. State `v5_va/state/verify_v5_wall.md`. Scripts and outputs `v5_va/vvw/`. Containers mount the
repo READ-ONLY (vvw/dr.sh, copied from checks/dr.sh); nothing in anatomies_v5 / anatomies_v4 / the repo was written.

## Answer (final, 01:25)

**holds = false, on one key number and one convention; the substance holds.**
- The universal "right_route_detail identical to v4" is refuted (117 / 223), exactly as the checks say.
- The donor-relief numbers reproduce with my own implementation at exact ownership on all 223 anatomies.
- The split reproduces (exit 0, 45 / 45 equal).
- The key-number class table does NOT reproduce: the strict rebuild gives the bake's 63 / 26 / 9 / 8 (identical
  beyond 4 / 6 / 8 mm / not even 8 mm), not the checks' 66 / 24 / 8 / 8. All 4 disputed anatomies go to the bake's
  class.
- The "whole VA 0.150" holds only under k24 ownership with the seam counted raw. Under the study's convention
  (exact ownership, seam = 0) it is 0.128, which is also the bake's own record. All 429 still beat v4.
- Omitted from the checks' summary: 3 carotid tube windows (one on the RCCA real wall), 2 new TopBrain-piece windows,
  2 assigned VAs kept from v4 by the last rung, and that 4 of the 13 route "losses" are k24 artefacts.

### Method
- `vvw/measure.py` is my own implementation of right_route_detail's definition:
  - own markups reader and own 0.25 mm resampling;
  - owner = argmax_j (r_j - |x - c_j|), dep = ||x - c*| - r*|, area fraction > 0.3 / > 0.5 mm;
  - both v4's k = 24 ownership and EXACT ownership (no k truncation);
  - measured on the UNDECIMATED union (`va_sources/v5_reg/iso_v5`), plus collision_full, the OBJ and v4's OBJ
    (report only).
- It ran on ALL 223 anatomies, the 25-anatomy sample first. 20 anatomies cannot cover 25 donors, because each donor
  serves one TopBrain patient. The sample covers all 25 donors and all 49 donor sides. Outputs:
  `vvw/out/<name>.json`; rows `vvw/va_rows.json`; summaries `vvw/summ.py`, `summ2.py`, `summ3.py`, `summ4.py`,
  `host_cmp.py`.
- `vvw/rrd_v4.py` rebuilds v4's exact union from mesh_v4.json (repo code; it equals the mesh_v4 record on 27 / 27).
  It measures both unions with my implementation (k24 and exact) and compares the RCCA-owned wall beyond 4 / 6 / 8 mm
  of every VA centreline. The comparison is strict: area, fractions, cell count and a centroid fingerprint.
  - 27 anatomies: all 13 below 0.99, all 8 "not even 8 mm", the 4 class-disputed ones, and 3 identical controls.
  - Table in `vvw/rrd_table.py`; results in `vvw/rrd_v4/`.
- `vvw/seam.py` builds the surface-radius profile across the splice. It ran on 20 anatomies (40 VAs): the 10 worst
  seam zones plus 10 random. Results in `vvw/seam/`.
- `vvw/gap.py` measures the gap between the VA tube and the RCCA tube (`vvw/gap.json`).
- `vvw/flags.py` lists the windows outside the donor.

### 1. right_route_detail vs v4
- My k24 implementation equals the mesh_v5.json record on 223 / 223 and the mesh_v4.json record on 117 / 223.
  - Ratios v5 / v4 (frac > 0.3 mm): fork 0.819-2.387, siphon 0.957-1.037.
  - 13 anatomies are below 0.99 and 1 is below 0.95 (w_050_left mr_004_L, fork 0.818).
  - Medians: fork 0.1968 (v4 0.1958), siphon 0.3218 (v4 0.3218).
  - All of these reproduce.
- **Identity classes.** My strict class equals the bake's on 27 / 27, and the checks' on 23 / 27. All 4 disputed
  anatomies take the bake's class:

  | anatomy | bake / strict | checks |
  |---|---|---|
  | w_003_right mr_016 | 6 | 4 |
  | w_017_right mr_018 | 6 | 4 |
  | w_048_left mr_005 | 6 | 4 |
  | w_046_left mr_020_L | 8 | 6 |

  - The correct table is 117 / 63 / 26 / 9 / 8. The checks' 66 / 24 / 8 / 8 is 4 anatomies too favourable.
  - The checks disclosed the rounding but kept their own table in the key numbers.
- **Exact ownership** moves the RRD (4 dp) on 156 / 223 v5 unions (max fork |d| 0.0134). Of the 13 losses, 4 are k24
  artefacts and return to ≥ 0.99 under exact ownership:

  | anatomy | k24 | exact |
  |---|---|---|
  | w_012_left mr_001 fork | 0.979 | 1.002 (k24 hands 62.8 mm2 of RCCA bulb wall to the RVA) |
  | k_005_right mr_024_L fork | 0.966 | 0.992 |
  | w_024_right mr_026_L fork | 0.971 | 1.000 |
  | w_029_right mr_026_L fork | 0.986 | 0.999 |

  - 9 stay below 0.99. The worst are w_050_left mr_004_L (0.818), m_030_right mr_013_L (0.971) and m_022_left
    mr_021_L siphon (0.957 → 0.974).
  - Under exact ownership w_012_left mr_021_L becomes "identical beyond 6 mm" (k24: none).
- **Windows outside the donor, not in the checks' summary** (`vvw/flags.py`):
  - 3 carotid_changed anatomies have a new v4-style carotid tube window:
    - m_030_right mr_013_L: lower_R, the RCCA real wall, s 65.8-66.95 (fork 0.971);
    - w_016_left mr_011_L and w_016_right mr_011_L: siphon_L, 1.3-1.6 mm.
  - 2 anatomies have a new 2.95 mm window on the TopBrain RVA piece: m_022_left mr_021_L and w_039_right mr_021_L.
  - The bake's notes disclose the carotid windows; the checks' notes name m_030 only.
  - The bake also kept 2 assigned VAs from v4 (w_050_left mr_026_L R, w_033_left mr_007 L; ostium-junction fusion,
    flagged), which the checks' summary does not mention.
- **The w_033_left gains** (fork 2.355-2.387, mr_010 / mr_011_L / mr_024) differ even beyond 8 mm of any VA: the
  RCCA-owned area is +51 mm2 beyond 8 mm. That fits the bake's explanation (v4's host RVA had refused the real bulb
  wall).
- **VA-RCCA tube gap:** no overlap in v5 (min 0.59 mm). The gap is under 1 mm on 32 RVAs in v5, against 67 in v4.

### 2. Donor VA relief on the undecimated union (429 donor VAs, all 25 donors, all 49 donor sides)
| measure (exact ownership) | median | p10 | min-max | claim |
|---|---|---|---|---|
| donor_real > 0.3 mm | 0.1343 | 0.0555 | 0.0294-0.3364 | 0.134 (p10 0.056, 0.029-0.337): reproduces |
| donor_real > 0.5 mm | 0.0742 | 0.0190 | 0.0033-0.2482 | not claimed |
| donor_real > 0.15 r | 0.1865 | 0.1037 | 0.053-0.382 | 0.187: reproduces |
| donor_real vs a FITTED tube (> 0.3 mm) | 0.1083 | 0.0460 | 0.023-0.315 | calibre offsets removed |
| seam zone (donor-side 25 mm ramp) | 0.1121 | 0.0001 | 0-0.917 | 0.112: reproduces |
| TopBrain piece | 0.1030 | 0.0164 | 0-0.407 | 0.103: reproduces |
| whole VA, origin → end, seam raw, k24 | 0.1499 | | | 0.150: reproduces with k24 only |
| whole VA, origin → end, seam raw, exact | 0.1451 | | | |
| whole VA, ostium → end, seam = 0, exact (the study's convention) | 0.1280 | | 0.028-0.267 | bake record 0.128 |

- **k24 vs exact on donor_real:** max |d| 0.050 (w_012_left mr_001 R: 0.251 → 0.201); more than 0.005 on 25 VAs.
  The median is unchanged.
  - The whole-VA 0.150 is inflated by k24 in the junction stub, which gives parent-subclavian wall to the VA (up to
    0.62 under k24 against 0.05 exact). The feasibility study already named this artefact.
- **The relief is the donor's own wall.**
  - 99.2% of donor relief (min 91.7%) lies within 0.25 mm of the donor surface (`vvw/extra2`).
  - 4% (max 18%) lies on the 1.8 r + 1 capsule (`vvw/extra`).
  - All of it is outward of the centreline-radius tube (inward > 0.3 mm: 0), since that radius is inscribed.
  - Per donor, the fitted-tube reading is close to the raw one, so the relief is shape, not a calibre offset.
  - Protrusions > 1.5 mm cover at most 7% of the wall (Song 094 L, Song 002 R, the high-genus donors).
- **Against v4:** donor_real beats the recorded v4 host core on 429 / 429 (host median 0.000, max 0.0201; the lowest
  donor is 1.5 × that maximum). The whole VA under the study's convention beats v4's whole VA (median 0.018) on
  429 / 429.
- **Windows:** 25 VAs with one window each. Median 1.31 mm (1.15-9.14 mm), 0.55-6.17% of the donor stretch.
  Reproduces.
- **A2** (donor in situ / alone, exact): median 0.997. Below 0.95 on 17 (record: 16). Minimum 0.756 on w_048_left
  mr_010_L R (ia172_159 L; record 0.901).

### 3. Seams
- **Placement follows the study.** All 429 put the 25 mm calibre ramp on the donor side (v4's ramp rule). The checks
  report the seam zone apart and exclude it from donor_real, as the study asks.
- **Profile across the splice** (20 anatomies, 40 VAs; `vvw/seam/`):
  - The surface radius steps by a median 0.11 mm across the cut (max 0.35).
  - The largest bin-to-bin jump within 5 mm of the cut is a median 0.19 mm. It exceeds the VA's own donor-stretch
    maximum on 3 / 40 VAs, and stays below the largest donor jump overall (0.79 vs 0.84 mm).
  - There is no step at the seam.
- **High seam-zone values are calibre, not relief.** The donor's real wall (r ≈ 2.0 mm) is kept in the seam zone while
  the centreline radius is ramped (≈ 1.5 mm). The wall then tapers onto the TopBrain wall over about 4 mm. Example:
  Song 095 L, seam 0.917.
- **Deviation:** the checks' whole-VA figure counts the seam raw and uses k24. The study uses exact ownership with the
  seam counted as 0, which gives 0.128 (above).

### 4. Relief in the 23.5k OBJ (report only)
- **OBJ** (triangle centroids, exact ownership): donor_real median 0.111, which is 0.81 of the union (p10 0.63, min
  0.29). At > 0.5 mm it is 0.068.
- **60k mesh:** 0.92 of the union.
- 428 / 429 are above twice v4's OBJ host-tube maximum (0.0053).
- **17 VAs read below 0.02 on the OBJ** (0.010-0.019; v4 host 0.000-0.0025). All come from the two smoothest donors,
  ia172_190 L and ia172_141 L (union 0.029-0.037). There the collision mesh is close to a plain tube.
- The vertex-weighted reading is higher (median 0.224), because decimation keeps feature vertices. Not used.

### 5. Split
- `v5_split.py --baseline split_v4.json`, rerun: EXIT 0, 45 / 45 quantities equal the baseline.
- 25 donors in 25 TopBrain patients, one_to_one_nested under all 3 definitions.
- Rosters: v3 153 / 16; v4side k16 174 / 16.
- Files: `vvw/split_v5_recheck.json`, `vvw/logs/split.log`.
