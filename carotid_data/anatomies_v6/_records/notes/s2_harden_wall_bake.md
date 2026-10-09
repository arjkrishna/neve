# s2_harden_wall_bake -- D15 local donor wall, bake_v6, checks_v6

Agent s2_harden_wall_bake (stage 2 harden), 2026-10-04. State: `state/s2_harden_wall_bake.md`. Code:
`scratchpad/v6/wall_local/` (D15), `scratchpad/v6/bake/` (bake_v6.py, runner dr.sh, regression scripts),
`scratchpad/v6/checks/` (checks_v6.py, loader_one.py). Outputs: `carotid_data/v6_work/wall/` (= /v6w: `seg/local/`,
`subclavian_v5rule/`, `subclavian_local/`, `donors_local/`, `wd/local/`, `scan/local/`, `local/`) and
`carotid_data/v6_work/bake_v6/` (= /bw: `reg/` regression). Containers `s2wb_*` (<= 2 at a time), /repo read-only.

## Answer so far

Run 1 (below, final) + run 2 (the user's D16 / D17 / D20, relaunch 11:28; section 8 and the run-2 bullets at the end
of this list).
- **D15 done for the 27 donor units' VAs** -> `carotid_data/v6_work/wall/donors_local/` (54 side records, donors_v5
  schema, v6.d15 block per side; `CHOSEN.json`). Zones = v5's own bone-side cut mechanism | the plateau label adding
  >= 0.6 mm | v5's floored stretches, 2 mm blend: core 13.5 % of the donor VA length (21.6 % with the blends).
- **Outside the zones the wall IS v5's**: on all 54 sides every vertex / triangle outside the zones is v5's own (bit
  for bit, after undoing the ct_seg4 smoothing spill), and the centreline is v5's own VMTK line there; inside the core
  the label is plateau's exactly. On the bake-like union the outside-zone wall detail equals v5's within 0.006 per VA
  (median 0.0007), the measure's own resampling noise (control: v5 against itself with the line 0.3 mm shorter,
  max 0.0025, median 0.0007).
- **Subclavian / junction (item e) local too** -> `wall/subclavian_local/` + VA-tool records
  `donors_local/_junction_records/` (54): artefact zones there are small (0 mm on 43 / 54 sides); outside them the
  surfaces are the v5-rule run's (108 surfaces, all outside vertices; triangles all but 2 + 2 on ia172_071 L); the
  same two junctions fail the bottleneck rule as in stage 1 (ia172_071 L, ia172_164 L -> v5 junction there).
- **Artefact gone in the zones**: CT / surface at the audit's 140 floored points 1.584 (v5) -> 1.155 (= plateau),
  >= 1.5x 83 -> 12, continuity 0.642 -> 0.937; floored mm in the used stretches 4730 -> 421 (0 outside the zones);
  VAs touching the floor 361 -> 84; VAs using a floored stretch (audit's floor_scan) 377 -> 106.
- **Wall detail kept coarse**: donor stretch median 0.140 (v5) / 0.110 (stage-1 plateau) / **0.120 (local)**; outside
  the zones 0.086 = v5; inside 0.347 (artefact) -> 0.212; sides keeping >= 0.8 of v5: 32 / 49 (plateau 28); <= 0.51: 2 (6).
- **bake_v6.py** (1.0.1) = bake_v5b + E1 (generalised sc* sections) + E2 + E3/E3b + E4 + E5 + E6 (blend morph) + E7
  (VA-aware OBJ re-cut) + C4 (mr_021 left-VA pinch, gated, revert on bake failure), one module + CLI with done
  markers. **checks_v6.py** (1.0): strict audit, route vs v4 exact ownership with the D13 baseline (carotid tool's
  provenance key, or v4 rebuilt without its tip), VA polyline lumen, D5 wall detail + D15 zone split, orientation
  gates with D8, loader.
- **Regression: bake_v6 reproduces the stage-1 pilot bakes bit for bit** on k_004 (E7: re-cut from the union at
  23500, LVA 0.561 -> 0.769), w_034 (E6 morph) and w_025 (c4 idempotent, sc* sections): union, 60k mesh, OBJ body
  (only meshio's timestamp line differs), every mesh-record field, morph records, route exact ratios to the last
  digit, donor_real, VA lumen; checks_v6 passes every gate on all three. c4: apply_c4 == the pilot's c4_apply.py byte
  for byte, idempotent, and applicable on all 8 mr_021_L anatomies on v5 geometry (clearance >= 3.27 mm).
- Loader rule: v5's own loader record lists the host's point gaps and duplicate points for every anatomy (k_004:
  (0) 19.1 mm + 1 duplicate, ..., (18) 11.7 mm); loader_one counts a gap / duplicate as the host's when the same
  consecutive point pair is in the host branch (also inside a donor-modified (17) / (18) / (19)).

Run 2 (FINAL, 16:40; details section 8):
- **D17 + D16 records published** (16:14) in `donors_local/_junction_records/` (54 sides; the D15-only ones in
  `_junction_records_d15only/`): each = ONE closed surface (junction + the donor's whole subclavian to the end of its
  segmentation), label end 19.0-121.1 mm past the ostium (median 73.2; < 30 mm on song 040 R / song 109 R -> the VA
  tool's short tube); junction_failed on 0 sides (bottleneck ratio min 0.53, median 0.857).
- **D16**: ia172_164 L 0.459 -> 0.897 (tube 9.71 mm at r 1.517 mm), ia172_071 L 0.291 -> 1.0 (10.01 mm at r 2.077 mm):
  both get the donor's own junction.
- **Local-fix rule on the longer subclavians**: outside the zones every vertex of all 108 surfaces is the v5-rule run's
  (2 triangles re-cut at a zone border); wall detail outside the zones = v5 rule's (4772 / 4808 bins identical, class
  share within 0.003). **Artefact gone in the subclavian zones**: CT / surface median 1.161 (v5 rule; 34 of 185 points
  >= 1.25) -> 1.070 (local; 1 point), = plateau; outside 1.012 for both (surface r_eq identical within 4e-5 mm).
- **bake_v6 1.1 / checks_v6 1.1 (D20)**: mirrored forks (flag or pre-mirrored file with flipped winding) and re-posed
  right forks bake; route gate per real section against its own v4 section + rcca_prefix_unchanged. 1.1 reproduces 1.0
  bit for bit on the 3 regression anatomies; D20 smoke k_011 (mirrored + map-posed right fork) passes every bake / route
  gate; D17 cut-end bake passes every gate.
- **checks_v6 1.2 adds part geom** (`checks/geom_v6.py`, runs pre-bake too): cross-group clearance >= 0.6 mm on new
  points + 1 mm turns <= 60 deg. 0 false fails on v5's own changes (223), the pilot (14), the regression; it explains the
  one smoke failure (w_051: re-posed RICA 3.06 mm into the v5 RVA -> "tube-only mesh has a handle"); 17 / 165 of the
  carotid tool's written trees have such fork-RVA contacts (all donor-RVA sides the VA tool re-places; its unit trees
  have none); on the VA tool's trees it flags a (19) hairpin (k_004 mr_014_L, 152 deg) and two 61 deg LVA seam turns
  (FYI `wall-geom-1` in `wall/requests.json`).

## 1. D15: the local wall

### Method
- Zones (per donor VA side, on v5's own ct_seg4 stations, 0.25 mm): bone = v5's bone-side mechanism (>= 2 bright-
  neighbour rays with lower threshold T_loc above the lumen core I_c: the ray ends at its first sample, v6_wall section 1)
  OR the effect (stage 1's plateau label adds lumen >= 0.6 mm deep outside v5's label, runs >= 0.75 mm); floor = v5's
  record line MISR < 1.0 (floor_scan's definition). Core = bone | floor, gaps < 2 mm closed; weight w = 1 on the core,
  linear to 0 over 2 mm (the blend), 0 elsewhere.
- Label merge: signed distances of v5's and plateau's final VA labels, phi = (1 - w) phi_v5 + w phi_plateau; w = 0
  gives v5's label exactly (asserted), w = 1 plateau's.
- Surface: ct_seg4's own extraction (marching cubes + write_vtp's clean + windowed sinc 20 / 0.1); the smoothing
  spill (20 rings) is undone outside the zones by giving every marching-cubes vertex in the w = 0 region v5's smoothed
  position of the same vertex. Checked per side: the recompute on v5's labels equals v5's file point for point; every
  outside vertex and every all-outside triangle is v5's.
- Line: v5's own VMTK line outside the zones, the VMTK line of the local surface inside (splice_lines.py).
- Subclavian + junction (item e): stage 1 made them with the plateau rule only; the same request runs repeated with the
  v5 rule (subclavian_v5rule/), the two masks merged the same way (VA part: the VA zones; subclavian: its own
  bright-cut / plateau-adds zones), surfaces spliced to the v5-rule surfaces outside the zones (local_sub.py).

What "outside the zones == v5" means at each level:
- surface (record): bit-identical vertices and triangles (checked per side, every outside vertex / all-outside
  triangle found in v5's file);
- line (record): v5's own VMTK points and MISR radii outside the zones (spliced; the raw VMTK line of the local
  surface is kept as <v>_stageA_vmtk_raw.json and its deviation from v5's line outside the zones is recorded);
- union / wall detail (wd_proxy, the bake's union per donor side): equal up to the union's own reach (0.45 mm grid,
  capsule) next to a zone -- hence a 3 mm GUARD class -- and the measure's resampling (prep_siphon spaces its 1 mm
  samples by the line's total length, so any length change moves every tube sample slightly); measured per bin and
  per class in wd_local.json, with a control (v5 itself, line trimmed 0.3 mm at the far end) for the noise floor;
- anatomy: the donor is re-posed in v6 (donor junction, D9), so the anatomy-level comparison with v5 is per class
  (checks_v6 va.d15_split: donor_real cells inside / outside the zones, v6 and v5).

### Results

Surfaces (`seg/local/<unit>/local_qc.json`; `seg/local/_splice_summary.json`):
- 27 / 27 units, 54 / 54 VA sides; 3 units (ia172_164, ia172_172, song 138) were cropped differently by the two
  runs (stations recentred differently) and were merged on the union grid (local_seg 1.1).
- On all 54 sides: the recompute of v5's own surface from v5's labels equals v5's file point for point; every vertex
  in the w = 0 region and every all-outside triangle of the local surface is v5's (0 unmatched); inside the core the
  local label equals plateau's label exactly (0 differing voxels). The smoothing spill undone outside the zones was at
  most 0.08 mm (song 113 R; typically 0.02-0.04).
- Handles: the local surfaces are one component on all 54 sides; genus sum 138 over the 52 sides where v5's surface
  is one component (v5: 309 there, plus 2 multi-component sides; stage-1 plateau 149 over 54) -- most of v5's handles
  sat in the artefact zones, so the bake's handle rule will tube fewer windows than on v5's walls.
- Zones: core 1618 mm of 11942 mm of donor VA (13.5 %), with the 2 mm blends 2580 mm (21.6 %); per side 1-34 % core
  (song 100: ~1 %; ia172_015: 31-34 %). So ~78 % of the donor VA wall is v5's surface exactly.
- Lines: VMTK on the local surfaces 54 / 54; ia172_069 L gave a 0.4 mm line (as plateau did in stage 1) and seeds 2 / 5
  mm inward did not help, so the raw line there is stage 1's plateau line (inside the core the local surface IS
  plateau's). The raw VMTK line deviates from v5's line by 0.05-0.37 mm on identical surface stretches (VMTK's
  backtrace), which is why the line is spliced; splice 1.1 blends by w ((1 - w) v5 projection + w raw), junction
  lateral offset <= 0.24 mm (along-line point gaps up to 1.7 mm are just sampling).

The artefact is gone in the zones (the audit's 140 floored + 136 reference points, `seg/local/gate_ct.json`, same
points / plane / CT read as stage 1's gate 1):

| p10 / median / p90 | v5 | stage-1 plateau | **local** |
|---|---|---|---|
| floored points: CT / surface | 1.027 / 1.584 / 2.442 | 0.803 / 1.155 / 1.487 | 0.803 / 1.155 / 1.487 |
| floored: CT / surface >= 1.25 / >= 1.5 | 119 / 83 | 40 / 12 | 40 / 12 |
| floored: surface r_eq < 1.0 | 62 | 7 | 7 |
| reference points: CT / surface (median) | 1.041 | 1.04 | 1.04 |
| continuity floored / reference (median); < 0.75 | 0.642; 107 | 0.938; 13 | 0.937; 13 |

Floored stretches on the records (`scan/local/floor_scan.json`, the audit's floor_scan on donors_local): donor VAs
using a floored stretch (MISR < 1.0) in their used range 377 (v5) / 139 (plateau) / **106** (local) of 429; donor sides
with a floored stretch 52 / 46 / 41 of 54. (Local is below plateau because outside the zones it keeps v5's line,
whose MISR is >= 1.0 there by construction -- every v5 floor is inside a zone -- while plateau's own re-centred line
had a few short new floors.) The audit's bone_scan on the remaining floored stretches >= 2 mm (`scan/local/bone_scan.json`):
22 (v5 140, plateau 26), their CT r_eq median 1.25 mm (v5 1.78, plateau 1.26), CT / surface >= 1.25 on 11 (v5 119,
plateau 12); 7 of them read < 1.0 on CT with no bright contact (the c3 type: real foramen narrowings / failed reads).

Records (`v6_work/wall/donors_local/`, donors_v5 schema, 27 units / 54 sides, `CHOSEN.json`): built by v6_wall's
build_donors6 (imported) with the local seg folders; each side record has v6.d15 (zones, surface identity check,
splice record). A first pass failed on 10 sides with "no subclavian path": v5's builder falls back to the seg
folder's stations_rays.json for the parent subclavian when Arterial's label is > 6 mm from the VA end; the local seg
folders now carry v5's own stations_rays.json, so every record has v5's own subclavian path (checked on song 113 R,
ia172_015 R, ia172_071 L: the same ct_seg4 station path as v5).

**Wall detail (D5 / D15)** -- `local/wd_local.json` (wd_proxy's bake-like union per donor side, calibrated against v5's
measured donor_real at r 0.991 in stage 1; over every v5 donor VA's used stretch, 429 VAs; classes by the local zones:
ZONE = w > 0, GUARD = within 3 mm of a zone bin, OUTSIDE = the rest):

| share of wall > 0.3 mm off the tube (p10 / median / p90) | v5 | stage-1 plateau | **local** |
|---|---|---|---|
| whole used donor stretch | 0.053 / 0.140 / 0.250 | 0.045 / 0.110 / 0.200 | 0.048 / **0.120** / 0.208 |
| in the zones (the artefact) | 0.129 / 0.347 / 0.536 | 0.036 / 0.207 / 0.408 | 0.060 / 0.212 / 0.411 |
| outside the zones | 0.028 / 0.086 / 0.172 | 0.031 / 0.083 / 0.165 | 0.028 / **0.086** / 0.173 |
| floored mm (MISR < 1.0) in the used stretches, total / outside the zones | 4730 / 0 | 489 / 0 | 421 / 0 |
| VAs whose used stretch touches the floor | 361 | 117 | 84 |
| union lumen on the polyline, median (min) | 0.935 (0.915) | 1.060 (0.923) | 1.068 (0.930) |

- Outside the zones local == v5: per VA |local - v5| of the outside share max 0.006, median 0.0007 (49 donor sides);
  the surfaces and lines there are bit-identical, the residual is the union measure's own resampling: prep_siphon
  spaces its 1 mm samples by the line's TOTAL length, so any change of length anywhere moves every tube sample by a
  fraction of a mm, and the exact-ownership bins shift between neighbours (per bin up to 0.37, only ~2 % of the
  outside bins stay bit-identical, while the class aggregate stays within 0.006). CONTROL (`wall_local/wd_ctrl.py`,
  tag v5trim): v5's own surface and line with only the line's far end trimmed by 0.3 mm (no geometric change over any
  used stretch) gives a per-VA outside difference to v5 of max 0.0025 (median 0.0007; whole stretch max 0.0043, 429 VAs) -- the measure's noise
  floor, against which local's 0.006 / 0.0007 is to be read.
- Per donor side (median over the VAs that use it; v5 measured = vvw exact ownership on iso_v5): local keeps >= 0.8 of
  v5's measured value on 32 of 49 sides (stage-1 plateau 28), <= 0.51 on 2 (plateau 6: ia172_164 L 0.124 -> 0.062,
  song 040 R 0.035 -> 0.023 remain; ia172_141 R 0.143 -> 0.103, song 040 L -> 0.040, song 095 L/R -> 0.119 / 0.087 are no
  longer <= 0.51). The remaining loss is entirely in the zones (the artefact's relief), the outside class is v5's.

Full per-side table (`local/wd_local.json` per_donor_side; printed in `local/chain/wdlocal.log`):

| donor side | VAs | v5 meas | v5 proxy | plateau | **local** | zone v5 / plat / local | outside v5 / local | floored mm v5 / local |
|---|---|---|---|---|---|---|---|---|
| ia172_015 L | 16 | 0.180 | 0.175 | 0.110 | 0.114 | 0.341 / 0.196 / 0.197 | 0.028 / 0.028 | 44 / 0 |
| ia172_015 R | 16 | 0.192 | 0.184 | 0.131 | 0.138 | 0.306 / 0.191 / 0.201 | 0.095 / 0.097 | 52.5 / 2 |
| ia172_039 L | 1 | 0.175 | 0.176 | 0.158 | 0.175 | - | 0.178 / 0.177 | 0 / 0 |
| ia172_039 R | 1 | 0.089 | 0.088 | 0.095 | 0.092 | 0.142 / 0.207 / 0.214 | 0.081 / 0.081 | 1 / 1 |
| ia172_058 L | 8 | 0.336 | 0.327 | 0.231 | 0.257 | 0.554 / 0.302 / 0.339 | 0.209 / 0.211 | 13 / 0 |
| ia172_058 R | 9 | 0.094 | 0.089 | 0.085 | 0.084 | 0.235 / 0.208 / 0.198 | 0.055 / 0.054 | 14 / 3 |
| ia172_069 R | 2 | 0.281 | 0.273 | 0.238 | 0.245 | 0.567 / 0.414 / 0.417 | 0.165 / 0.167 | 9 / 0 |
| ia172_071 L | 4 | 0.293 | 0.280 | 0.243 | 0.250 | 0.409 / 0.297 / 0.307 | 0.237 / 0.241 | 33 / 0 |
| ia172_071 R | 2 | 0.082 | 0.080 | 0.068 | 0.072 | 0.120 / 0.100 / 0.100 | 0.061 / 0.061 | 24 / 15.5 |
| ia172_136 L | 4 | 0.197 | 0.192 | 0.194 | 0.196 | 0.325 / 0.322 / 0.309 | 0.095 / 0.101 | 17 / 0 |
| ia172_136 R | 5 | 0.180 | 0.171 | 0.123 | 0.122 | 0.302 / 0.174 / 0.173 | 0.074 / 0.075 | 19 / 0 |
| ia172_141 L | 4 | 0.029 | 0.030 | 0.018 | 0.021 | 0.154 / 0.061 / 0.075 | 0.013 / 0.013 | 8 / 0 |
| ia172_141 R | 4 | 0.143 | 0.142 | 0.072 | 0.103 | 0.363 / 0.149 / 0.178 | 0.054 / 0.055 | 5 / 0 |
| ia172_149 L | 13 | 0.091 | 0.091 | 0.098 | 0.092 | 0.493 / 0.483 / 0.484 | 0.050 / 0.050 | 1 / 0 |
| ia172_149 R | 15 | 0.308 | 0.306 | 0.245 | 0.254 | 0.536 / 0.367 / 0.378 | 0.172 / 0.173 | 14 / 0 |
| ia172_159 L | 9 | 0.171 | 0.174 | 0.101 | 0.120 | 0.381 / 0.212 / 0.222 | 0.058 / 0.058 | 20 / 0 |
| ia172_159 R | 15 | 0.172 | 0.169 | 0.107 | 0.121 | 0.398 / 0.138 / 0.156 | 0.098 / 0.098 | 9 / 0 |
| ia172_164 L | 16 | 0.124 | 0.124 | 0.046 | 0.062 | 0.250 / 0.072 / 0.086 | 0.072 / 0.072 | 25 / 9 |
| ia172_164 R | 16 | 0.095 | 0.098 | 0.090 | 0.092 | 0.141 / 0.094 / 0.092 | 0.102 / 0.104 | 0 / 0 |
| ia172_165 L | 12 | 0.160 | 0.156 | 0.156 | 0.150 | 0.419 / 0.400 / 0.376 | 0.100 / 0.099 | 5 / 0 |
| ia172_165 R | 11 | 0.201 | 0.197 | 0.198 | 0.198 | 0.254 / 0.254 / 0.250 | 0.199 / 0.201 | 10 / 0 |
| ia172_190 L | 15 | 0.036 | 0.036 | 0.029 | 0.030 | 0.129 / 0.029 / 0.050 | 0.027 / 0.027 | 7 / 5 |
| ia172_190 R | 15 | 0.071 | 0.070 | 0.045 | 0.048 | 0.192 / 0.035 / 0.047 | 0.048 / 0.049 | 10 / 0 |
| ia172_192 L | 9 | 0.178 | 0.171 | 0.154 | 0.156 | 0.354 / 0.095 / 0.086 | 0.166 / 0.166 | 2 / 0 |
| ia172_192 R | 10 | 0.086 | 0.084 | 0.081 | 0.080 | 0.130 / 0.034 / 0.060 | 0.086 / 0.086 | 3 / 0 |
| song 002 L | 16 | 0.110 | 0.107 | 0.132 | 0.131 | 0.431 / 0.537 / 0.542 | 0.052 / 0.051 | 2 / 0 |
| song 002 R | 16 | 0.250 | 0.250 | 0.190 | 0.202 | 0.541 / 0.373 / 0.400 | 0.099 / 0.102 | 8 / 0 |
| song 008 L | 5 | 0.309 | 0.306 | 0.229 | 0.248 | 0.662 / 0.453 / 0.479 | 0.141 / 0.141 | 4 / 2 |
| song 008 R | 5 | 0.244 | 0.240 | 0.210 | 0.223 | 0.296 / 0.218 / 0.242 | 0.230 / 0.226 | 2 / 0 |
| song 032 L | 2 | 0.073 | 0.071 | 0.076 | 0.075 | 0.150 / 0.168 / 0.165 | 0.060 / 0.061 | 13 / 0 |
| song 032 R | 2 | 0.055 | 0.055 | 0.038 | 0.052 | 0.141 / 0.085 / 0.124 | 0.027 / 0.028 | 15 / 3 |
| song 040 L | 8 | 0.089 | 0.086 | 0.032 | 0.040 | 0.263 / 0.087 / 0.090 | 0.002 / 0.001 | 19 / 0 |
| song 040 R | 8 | 0.035 | 0.032 | 0.015 | 0.023 | 0.124 / 0.018 / 0.037 | 0.011 / 0.013 | 12 / 0 |
| song 094 L | 4 | 0.324 | 0.330 | 0.247 | 0.250 | 0.527 / 0.345 / 0.370 | 0.121 / 0.120 | 32 / 4 |
| song 094 R | 4 | 0.145 | 0.144 | 0.150 | 0.148 | 0.331 / 0.363 / 0.368 | 0.100 / 0.097 | 3 / 0 |
| song 095 L | 8 | 0.229 | 0.222 | 0.112 | 0.119 | 0.451 / 0.130 / 0.149 | 0.122 / 0.122 | 13 / 0 |
| song 095 R | 8 | 0.166 | 0.161 | 0.078 | 0.087 | 0.428 / 0.081 / 0.088 | 0.105 / 0.106 | 6 / 0 |
| song 100 L | 6 | 0.075 | 0.074 | 0.071 | 0.074 | - | 0.074 / 0.074 | 0 / 0 |
| song 100 R | 6 | 0.109 | 0.108 | 0.108 | 0.108 | - | 0.108 / 0.108 | 0 / 0 |
| song 108 L | 16 | 0.146 | 0.140 | 0.130 | 0.135 | 0.503 / 0.310 / 0.317 | 0.127 / 0.129 | 0 / 0 |
| song 108 R | 16 | 0.128 | 0.124 | 0.135 | 0.135 | 0.386 / 0.408 / 0.411 | 0.065 / 0.066 | 1 / 0 |
| song 109 L | 15 | 0.102 | 0.101 | 0.090 | 0.102 | 0.347 / 0.320 / 0.336 | 0.075 / 0.076 | 0 / 0 |
| song 109 R | 15 | 0.056 | 0.053 | 0.057 | 0.057 | 0.013 / 0.072 / 0.064 | 0.061 / 0.061 | 3 / 0 |
| song 113 L | 8 | 0.226 | 0.222 | 0.159 | 0.164 | 0.391 / 0.212 / 0.228 | 0.104 / 0.102 | 10 / 0 |
| song 113 R | 8 | 0.188 | 0.184 | 0.115 | 0.135 | 0.408 / 0.207 / 0.233 | 0.102 / 0.102 | 12 / 0 |
| song 141 L | 4 | 0.095 | 0.091 | 0.089 | 0.093 | 0.485 / 0.430 / 0.445 | 0.036 / 0.037 | 0 / 0 |
| song 141 R | 4 | 0.034 | 0.035 | 0.031 | 0.031 | 0.213 / 0.122 / 0.128 | 0.028 / 0.028 | 0 / 0 |
| vmr0248 L | 6 | 0.137 | 0.211 | 0.178 | 0.181 | 0.458 / 0.336 / 0.344 | 0.103 / 0.106 | 16 / 8 |
| vmr0248 R | 7 | 0.059 | 0.060 | 0.079 | 0.077 | 0.032 / 0.225 / 0.225 | 0.058 / 0.058 | 8 / 6 |

(ia172_172 and song 138 are reserves: no v5 VA uses them; their records and zones exist. Zone values "-": no zone bin in
the used stretches.)

### Subclavian stretches and junctions (item e)

- v5-rule request runs: 27 / 27 (`subclavian_v5rule/`, sub_batch2.py with V6_RULE=v5, every other switch as stage 1's
  j3 run: subclavian cleanup + junction bridge).
- local_sub (`subclavian_local/<unit>/local_qc.json`): 27 / 27 units, 54 sides, 0 failures. The subclavian artefact
  zones are small: 0 mm on 43 of 54 sides, 0.2-17 mm on the other 11 (ia172_071 L 11.2). The v5-rule recompute equals
  the v5-rule run's file on all 108 surfaces (54 subclavian + 54 junction); outside the zones every vertex is the
  v5-rule run's on all 108, and every all-outside triangle too except 2 of 34 707 / 40 753 on ia172_071 L's two
  surfaces (re-triangulated in marching-cubes cells that touch the zone border; their vertices are the v5-rule ones). The junction's VA part follows the VA's own zones, so it matches
  the local donor VA record.
- Junction QC re-measured on the LOCAL junction masks (junction6's bottleneck: largest ball from the subclavian to
  the VA at s 10): the VA tool's records (`donors_local/_junction_records/`, sc_post3 imported unchanged) flag the same
  two sides as stage 1 (ia172_071 L 0.29, ia172_164 L 0.46 of the VA's inscribed radius, < 0.5 -> the VA tool keeps
  v5's junction there). Over the 54: bottleneck ratio median 0.854 (stage 1 0.85), < 0.7 on 10 (stage 1 9), tube
  fill (fallback_tube) on 2 (= stage 1's ia172_136 L, ia172_172 L); sc_post3's junction wall-detail proxy median
  0.515 (min 0.30).

Subclavian wall detail (`local/wd_sub_local.json`, 54 sides, wd_proxy on each surface along its OWN VMTK line, core =
without the 8 mm ends and +-6 mm about the ostium): v5 rule p10 / median / p90 0.324 / 0.528 / 0.700, local 0.324 /
0.528 / 0.698 (plateau 0.325 / 0.528 / 0.697); inside the (small) subclavian zones median 0.794 (v5 rule) -> 0.829
(local). Outside the zones the class share differs from the v5-rule run's by up to 0.074 on one side (exactly 0 on
the sides whose surfaces are unchanged, e.g. 14 of the 30 ia172 / VMR sides): the subclavian lines were re-traced by VMTK on each surface (not spliced as the VA lines are), so the bins
do not coincide -- superseded by the D17 run (section 8), which measures the local surface along the v5-rule run's own
line, bin for bin.

## 2. bake_v6.py

`scratchpad/v6/bake/bake_v6.py` (VERSION "bake_v6 1.0.1"; the regression ran 1.0, which differs only in the E6
fallback switch, OFF in 1.0.1), one versioned module + CLI (neve-build-meshlab, /repo read-only):

    python3 /sp/v6/bake/bake_v6.py <name> --tree T --bake BDIR --iso ISO --pin PIN [--no-c4] [--no-e7] [--force]
    library: bake_anatomy_v6(name, tree, bake_dir, iso_dir, pin_dir), apply_c4(root), obj_recut_va(root, iso), setup()

= bake_v5b (imported unchanged) + the pilot's fixes folded in (the pilot ran bake_v6p 1.2 + v6_va's
bake_sections_v6 + obj_fix_va.py + c4_apply.py as four separate pieces):
- E1 sections: v5's (v4's + vaC_R / vaC_L) + every VA-side real section in provenance xform whose key starts with
  "sc" and has branch / lo_mm / hi_mm (the pilot's fixed list scC_R, scD_R, scC_L, scD_L, scJ_L kept in its order --
  section order is part of the bake -- and any new sc* key after it, sorted).
- E2 branch lookup fallback to the ".mrk" name; E3 VA pool (sc* branches + changed carotid sections searched);
  E3b margin by branch (source patch of bake_v5b.bake_one_v5b, asserted); E4 per-anatomy pin report against v4;
  E5 records (mesh_v6.json, provenance "bake_v6", union to ISO).
- E6 blend morph: as the pilot (rigid piece from v5's piece back-mapped, checked <= 0.01 mm beyond the blend); a piece
  without a v5 counterpart stays rigid and is recorded. (A source-centreline fallback is in the code but OFF: tested
  in section 4, it is not exact enough; env BAKE_V6_E6_SRC=1.)
- E7 (in the bake, every anatomy): VA lumen on the OBJ polyline (wall/va_lumen_poly.py, 0.05 mm) >= 0.65 on both VAs,
  else the OBJ is re-cut (full 23500 ... 23000, then union 23500 / 23400; bake_meshes_v4's own acceptance); none ->
  OBJ kept and flag va_lumen_obj_fail in the done marker.
- C4 (pre-bake): the topcow_mr_021 left-VA pinch override on the VA whose top surface is topcow_mr_021_lVA.vtp
  (found from provenance; cross-checked against c4_tube_override.json's list of 8), clearance gate >= 0.6 mm, backup of
  the curve in BDIR/_c4/<name>/; if the bake then fails, the radii are restored and the anatomy re-baked once without it
  (recorded "reverted"). Idempotent (radii already at the override -> nothing changes). Provenance "c4_v6".
- Done marker BDIR/_done/<name>.bake.json (written last; includes md5 of the OBJ and the 60k mesh, morph, c4, e7,
  timing, peak RSS); BDIR/_done/<name>.bake_failed.json on an exception or a non-passing bake; a rerun without the
  marker removes partial outputs first.

## 3. checks_v6.py

`scratchpad/v6/checks/checks_v6.py` (VERSION "checks_v6 1.0") -> OUT/<name>.json with "gates" and "pass":

    python3 /sp/v6/checks/checks_v6.py <name> --tree T --iso ISO --out OUT [--parts item1,route,lumen,va,orient,loader]

| part | what | gate |
|---|---|---|
| item1 | strict mesh audit (chk_one.item1 = v4's topo_audit: topology raw / welded, self-intersections on OBJ / 60k / union, budgets, routes navigable, every centreline point inside) | strict_pass |
| route | right route vs v4, exact ownership (+ k24), v4's union rebuilt from mesh_v4.json; D13: when the v6 RECA tip is re-posed or skipped vs v4, v4 is also rebuilt WITHOUT its tip (RECA cut at v4's join, ecaext_R dropped, v4's far clamp kept) and used as the baseline when it differs from v4 shipped (v4's tip owned RCCA wall = ran along the ICA); provenance override honoured (route_baseline / d13 keys) | fork and siphon >= 0.99 vs the baseline used |
| lumen | VA polyline lumen (va_lumen_poly, 0.05 mm) on OBJ, 60k, union | OBJ >= 0.65 on both VAs |
| va | D5 wall detail (exact ownership, undecimated unions) per VA side: donor stretch, donor_real, subclavian stretch, TopBrain top, whole VA; v5 the same; D15 split of donor_real into zone / outside cells (local zones placed with the section's own xform), v6 and v5 | info |
| orient | orient_gates.measure_tree on the anatomy: tier A; v6-blocking flags with D8 applied (va_target 20-30 accepted + flag; > 30 accepted only on a recorded D9 rung 3) | tier A == 0, blocking after D8 == 0 |
| loader | the pilot's loader checks in a clean interpreter (repo eve / eve_bench; loader_one.py) on this anatomy alone; host point gaps identical to the host reported as known | ok |

## 4. Regression (3 pilot anatomies)

Inputs: the stage-1 pilot anatomies copied from `v6_work/pilot/tree/` WITHOUT their bake outputs into
`v6_work/bake_v6/reg/tree/` (same composed centrelines and provenance, plateau donor walls -- the pilot's inputs);
`bake_v6.py` + `checks_v6.py` run by `bake/slot2.sh` (container s2wb_slot2); comparison `bake/reg_compare.py` ->
`v6_work/bake_v6/reg/reg_compare.json`. Chosen to exercise every fold-in:
- case_k_004_left__topcow_mr_010: E7 (the pilot's OBJ re-cut, LVA 0.561 on the OBJ) and a skipped RECA tip (D13 path);
- case_w_034_right__topcow_mr_017_L: E6 (the blend-morph case, siphon 0.986 -> 0.991);
- case_w_025_right__topcow_mr_021_L: c4 (already applied in the pilot tree -> idempotence), sc* sections (E1/E2) and
  the G4 case, the narrow VAs.
Plus a c4 equivalence test on a v5 copy (`bake/c4_test.py`, `v6_work/bake_v6/c4test/`) and an E6 fallback test
(`bake/e6_test.py`).

(results below)

**Result (`reg_compare.json`): bake_v6 reproduces the stage-1 pilot bakes exactly on all three.**

| anatomy | union / 60k / OBJ body | mesh record (rung, flags, margins, tube windows, routes, right_route_detail) | morph | E7 | route exact fork / siphon (= pilot) | donor_real R / L (= pilot) | VA lumen OBJ R / L | checks_v6 gates |
|---|---|---|---|---|---|---|---|---|
| k_004_left mr_010 | identical | identical | identical | re-cut from the union at 23500, LVA 0.561 -> 0.769 | 0.99997911 / 1.00548114 | 0.1375 / 0.1056 | 1.274 / 0.769 | all pass (loader after the host-pair rule) |
| w_034_right mr_017_L | identical | identical | identical (0.917 / 0.721 mm) | ok as baked | 1.00246763 / 0.99079433 | 0.0529 / 0.0493 | 1.062 / 0.978 | all pass |
| w_025_right mr_021_L | identical | identical | identical | ok as baked | 1.00175350 / 0.99686853 | 0.2067 / 0.1443 | 1.087 / 0.804 | all pass |

(The OBJ files differ only in meshio's write-timestamp header line; their bodies are byte-identical. D13 on k_004: the
RECA tip is skipped in v6; v4 rebuilt without its tip gives the same RCCA relief (ratio 0.99997917 / 1.00548123), so
the v4 baseline is used -- the carotid agent found the same, k_004 v4_no_tip == v4.)

- **c4 equivalence** (`v6_work/bake_v6/c4test/c4_test.json`): on a copy of v5's case_w_050_left__topcow_mr_021_L,
  bake_v6.apply_c4 writes the RVA curve byte-identical to the pilot's c4_apply.py (md5 3765de19...), same numbers
  (19 points, RVA s 211.8-229.6 mm, r_min 1.00 -> 1.643 mm, clearance 7.42 mm to its own VA beyond 12 mm); a second
  call changes 0 points (idempotent).

- **c4 on all 8** (`c4test/c4_all8.json`, apply_c4 write=False on anatomies_v5 and the pilot tree): the override applies
  on all 8 listed anatomies on v5's geometry (19 points each, r_min 1.00 -> 1.643 mm, clearance >= 3.27 mm, w_039 the
  closest); the pilot's w_025 is already at the override (0 points). In stage 2 the gate is re-run on the v6 geometry
  inside the bake.
- **E6 source-centreline fallback** (`bake/e6_test.py`, logs `reg/logs/e6_test{2,3}.log`): with the v5 back-map
  disabled, the written pieces sit 0.05-0.22 mm off their raw TopBrain source polyline, and the fallback morph agreed
  with the exact back-map only within 0.15 mm (k_011 siphon_L) and 0.70 mm (w_034 siphon_L). Not good enough: the
  fallback is OFF by default in bake_v6 1.0.1 (env BAKE_V6_E6_SRC=1 turns it on); a piece without a v5 counterpart
  stays rigid and is recorded, exactly as in the pilot. (1.0.1 changes only that branch; the regression ran 1.0 and
  used the v5 back-map on every blended piece.)
- **k_004 loader** re-run with the host-pair rule: all checks_v6 gates pass on all three regression anatomies.

## 5. Interfaces for s2_integrate / s2_build

- Donor VA records (D15 local): `V6VA_DONOR_ROOT=/repo/carotid_data/v6_work/wall/donors_local` (graft_v6_va's env;
  same schema as donors_v5 / donors/plateau; every side record has v6.d15 = zones, surface identity, splice).
- Donor junction walls (item e, D15 local): `V6VA_SEG=/repo/carotid_data/v6_work/wall/donors_local/_junction_records`
  (sc_post3 records of `wall/subclavian_local/<donor>/`; the junction QC there is re-measured on the local mask).
  Run 2: the same folder now holds the D16 + D17 records (published by `wall_d17/publish_d17.py`; the D15-only ones in
  `donors_local/_junction_records_d15only/`): surfaces `wall/d17/local/<donor>/<x>junction_stageA.vtp` (the junction
  + the donor's whole subclavian to the end of its segmentation, ONE closed surface), `d17` {covered_s_mm on the path
  arclength, label_end_past_ostium_mm, path_file, ...}, `d16_tube` on ia172_164 L / ia172_071 L. Paths:
  `V6VA_PATHS=/repo/carotid_data/v6_work/va_s2/paths_d17` (the VA tool's; identical to `wall/d17/paths` up to the
  shorter end, checked in `_paths_check.json`).
- Bake: `python3 /sp/v6/bake/bake_v6.py <name> --tree <tree> --bake <bake dir> --iso <iso dir> --pin <pin dir>`
  (neve-build-meshlab, /repo read-only, ~2.5-5 min, peak ~1.9 GB). c4 runs inside it (or call bake_v6.apply_c4(root)
  at compose time: idempotent). A worker loop like pilot_build/worker_v6.sh with claim dirs works unchanged.
- Checks: `python3 /sp/v6/checks/checks_v6.py <name> --tree <tree> --iso <iso dir> --out <checks dir>` (~6-8 min;
  "pass" + "gates" in the output). D13: reads provenance v6_carotid.summary.route_baseline.
  Run 2: bake_v6 1.1 / checks_v6 1.1 / loader_one 1.1 (same CLIs): mirrored forks = xform.<key>.mirror on the fork's
  own surface file; E6 uses xform.<key>.blend_disp where v5's back-map does not apply (every moved or mirrored fork);
  the route gates are route_fork_ge_0.99 / route_siphon_ge_0.99 per real section + rcca_prefix_unchanged; D17 cut ends
  need nothing in the bake (requests.json va-D17-bake-1).
  checks_v6 1.2 (same CLI; 1.1 frozen `checks_v6_v1_1.py`) adds part `geom` (gates geom_clearance_new_ge_0.6,
  geom_kink_new_le_60). It needs only the tree, so run it BEFORE the bake as well to catch a cross-tool contact
  early: `python3 /sp/v6/checks/geom_v6.py <tree> <name>|all [--out F.json]` (~1 s / anatomy, any image with numpy +
  scipy and /opt/eve_training/topbrain_tools). Existing checks records get geom with `--parts geom`.

## 6. Decisions (mine, reversible)

- D15 merge at the final-label level of two existing runs (v5's own ct_seg4 labels, stage 1's plateau labels), not a
  per-station rule switch inside ct_seg4: w = 0 gives v5's label exactly by construction, and the 20-ring smoothing
  spill is undone by vertex identity, so "outside the zones == v5" holds bit for bit on the surface (not just within a
  tolerance). Undo: use donors/plateau (stage 1) as the donor root.
- Zone detector = mechanism OR effect OR floor (see 1.); the three agree closely (song 113: 32 / 24-37 / 16 mm). The
  2 mm blend and 2 mm gap closing follow stage 1's NEAR class (+-2 mm of a floor). Undo / tune: local_seg GAP_MM,
  BLEND_MM, DEPTH_MM, RUN_MM.
- Line = v5's VMTK line outside the zones (the local surface IS v5's there), the local surface's VMTK line inside.
- Subclavian: the v5-rule request runs are the "v5 surface" of item (e) (v5 had a host tube there, no donor wall).
- checks_v6 1.2 geom: placement groups (a fork's CCA/ECA, a VA side's VA + subclavian curves) are not gated against
  each other (one real piece; the composers' self / hard checks cover them); only NEW points are gated (v5's own
  contacts are inherited, e.g. w_037_right mr_010_L RECA-RVA 0.475 mm from v4); a join is left out over the child's
  fused run + 15 mm; the kink gate also passes a turn no sharper than the same v5 curve's (rigidly moved pieces keep
  their source's turns). Undo: run checks_v6 without the geom part (`--parts item1,route,lumen,va,orient,loader`).
- D17 subclavian CT gate: bone_scan's CT read with the ring at [R + 1.5, R + 3.5] mm and rays to R + 5 mm (its fixed
  3-5 mm ring would sit inside a 4-5 mm subclavian lumen).
- checks_v6 D13: the carotid tool's declared baseline (provenance v6_carotid.summary.route_baseline) first; without it,
  v4-without-tip is the reference when the v6 RECA tip differs from v4's AND v4's tip owned RCCA wall (exact ownership
  differs > 5e-4) -- the operational form of "ran along the ICA".

## 7. Files

Code (`scratchpad/v6/`, /sp/v6 in containers):
- `wall_local/local_seg.py` (1.1: zones, VA label merge, surface splice), `vmtk_local_fix.py`, `splice_lines.py` (1.1),
  `build_donors_local.py`, `local_sub.py` (subclavian / junction), `junction_records.py` (sc_post3 on the local walls),
  `wd_local.py`, `wd_sub.py`, `wd_sub_local.py`, `d15_report.py`, `chainA.sh`, `chainB.sh`, probes `probe1-3.py`.
- `bake/bake_v6.py` (1.0.1), `bake/dr.sh` (runner, containers s2wb_*), regression `reg_run.sh`, `slot2.sh`,
  `post_slot2.sh`, `reg_compare.py`, `c4_test.py`, `c4_all8.py`, `e6_test.py`, `s2wb_results.py`.
- `checks/checks_v6.py` (1.0), `checks/loader_one.py`.
- Run 2: `wall_d17/` (`paths_d17.py`, `make_req_d17.py`, `sub_batch_d17.py`, `junction_records_d17.py`,
  `wd_sub_d17.py`, `d17_report.py`, `d17_fig.py`, `publish_d17.py`, `ls11_test.py`, `chainD17.sh`, `slotB.sh`,
  `slotC.sh`, `answer_req.py`); `wall_local/local_sub.py` 1.1 (1.0 frozen `local_sub_v1_0.py`); `bake/bake_v6.py` 1.1
  (1.0.1 frozen `bake_v6_v1_0_1.py`), `bake/{d20_test.py, reg11.sh, reg11_compare.py, d20smoke.sh, d17bake.sh,
  slot2b.sh, probe/pair_clear.py}`, `bake/s2wb_results.py` (run 2; run-1 copy `s2wb_results_run1.py`);
  `checks/checks_v6.py` 1.2 (1.1 frozen `checks_v6_v1_1.py`, 1.0 `checks_v6_v1_0.py`), `checks/geom_v6.py` 1.0,
  `checks/geom_summ.py`, `checks/kink_probe.py`, `checks/loader_one.py` 1.1 (1.0 frozen `loader_one_v1_0.py`);
  `wall_d17/{sub_ct_gate.py, genus_cmp.py, answer2.json, fyi_va.py}`.
- Run-2 outputs: `wall/d17/` (`paths/`, `req/`, `v5rule/`, `plateau/`, `local/`, `_junction_records/`, `figs/`,
  `chain/`, `D17_REPORT.json`, `publish.json`, `ls11_test.json`), `wall/wd/{d17_v5rule,d17_local,d17_local_v5line}/`,
  `bake_v6/{reg11,d20,d20smoke,d17bake}/`, `wall/donors_local/_junction_records_d15only/`, `wall/d17/sub_ct_gate.json`,
  `wall/d17/genus_cmp.json`, `bake_v6/geom/` (geom runs + calibration), `scratchpad/v6/results/s2_harden_wall_bake.json`.

Outputs (`D:\Arjun\neve\carotid_data\v6_work\`):
- `wall/donors_local/<unit>/{record.json,R.json,L.json,R_surface.vtp,L_surface.vtp}`, `CHOSEN.json`, `D15_REPORT.json`,
  `_junction_records/<donor>/<R|L>/junction.json` (VA tool, item e).
- `wall/seg/local/<unit>/` (merged labels, local stage-A surfaces, raw / spliced VMTK lines, zones.json, local_qc.json,
  v5's stations_rays.json), `seg/local/gate_ct.json`, `seg/local/_splice_summary.json`.
- `wall/subclavian_v5rule/<unit>/` (request runs with the v5 rule), `wall/subclavian_local/<unit>/` (local subclavian
  and junction surfaces, junction_labels_kji.npz, sub_zones.json, local_qc.json, seg_qc.json with the local junction QC).
- `wall/wd/local/`, `wall/wd/subclavian_local/`, `wall/wd/subclavian_v5rule/` (wd_proxy bins), `wall/scan/local/`
  (floor / bone scans), `wall/local/{wd_local.json,wd_sub_local.json,chain/,logs/}`.
- `bake_v6/reg/` (regression tree, bake markers, iso, pins, checks, reg_compare.json, logs), `bake_v6/c4test/`.
- `scratchpad/v6/results/s2_harden_wall_bake.json` (summary).

## 8. Run 2 (relaunch 11:28): the user's decisions D16 / D17 / D20

### 8.1 D17: the donor's own subclavian to the end of its segmentation

- Paths (`wall_d17/paths_d17.py` 1.1 -> `wall/d17/paths/`, `_d17_summary.json`): v6_va's `va/subclavian_paths.py`
  imported UNCHANGED with its distal cap (KEEP_PAST_MM 45) lifted, so the path follows the Arterial xSA chain (origin
  -> ostium -> largest-radius continuation) to the chain's end (fallback sides: the record's subclavian; VMR 0248: the
  expert SimVascular path), clipped 3 mm inside the CT (no side needed it). Up to the old 45 mm cap every point is
  v6_va's stage-1 path (max diff 0.0 on all 54). One wall-side rule: where Arterial's calibre collapses at the chain end
  (5 mm running median < 0.5 x the subclavian's own calibre about the ostium [O - 5, O + 20], to the end, only beyond
  O + 25) the path stops there -- the lumen fading out / the walk's last step into a small branch (8 sides, 0-12.4 mm
  cut: ia172_015 L 12.4, ia172_159 R 8.4, ia172_172 R 11.4, ia172_192 L 6.0 / R 8.5, ia172_165 L 1.5, song 113 L/R ~0).
  Length past the ostium: 19.0-121 mm (median ~74); < 30 mm on 2 sides (song 040 R 20.6, song 109 R 19.0 -> the VA
  tool's short tube to 30 mm), 30-31 mm on 4 (ia172_015 R, song 002 R, song 094 L, song 095 R 32.7).
  The VA agent built the same paths independently (`v6_work/va_s2/paths_d17`, same code, no cut); the records use
  theirs when identical to mine up to the shorter end (checked per side, `d17/_junction_records/_paths_check.json`).
- Requests (`make_req_d17.py` -> `d17/req/`): stage 1's 54 entries (same s_from, ostium, per-donor env) with the D17
  centreline and s_to = its end; checked to continue stage 1's centreline exactly over stage 1's stretch.
- Request runs (`sub_batch_d17.py` = jfix/sub_batch2 with claim dirs): BOTH rules, v5 (`d17/v5rule/`) and plateau
  (`d17/plateau/`), every other switch as stage 1's j3 runs (V6_JUNC_BRIDGE, V6_SUB_CLEAN). The longer stretches widen
  the CT crop: at --memory 6g the runs thrashed (ia172_039 363 s vs 142 s in stage 1) -> restarted at 8g.
- Local merge (`local_sub.py` 1.1, roots by env): exactly D15's rule (v5-rule mask outside the zones, plateau rule in
  the bright-cut / plateau-adds zones + 2 mm blend; VA part = the VA's own D15 zones), surfaces spliced to the v5-rule
  run's outside the zones -> `d17/local/<unit>/`; VMTK lines (`jfix/vmtk_sub3.sh`) for local and v5 rule.
- Records (`junction_records_d17.py`: sc_post3 imported unchanged, WALL = d17/local, PATHS = the VA tool's paths_d17)
  -> `d17/_junction_records/`, + `d17` {covered_s_mm, label_end_past_ostium_mm, path_file, surface, how,
  tube_extension_needed} and `d16_tube` (below); published over `donors_local/_junction_records/` by `publish_d17.py`
  after its checks (D15-only records kept in `donors_local/_junction_records_d15only/`).
- Wall detail (`wd_sub_d17.py`, wd_proxy unchanged): d17_v5rule (v5-rule surface + its line), d17_local (local surface +
  its line) and d17_local_v5line (local surface along the v5-rule run's OWN line: bin for bin comparable, so outside the
  zones the identity is exact rather than within a re-trace noise floor); `d17_report.py` -> `d17/D17_REPORT.json`.

### 8.2 D16: the two pinched ostia (ia172_164 L, ia172_071 L)

`local_sub.py` 1.1 `d16_tube`: the junction6 fallback-tube region -- radius = max(VA edge radius, station radius) over
VA s 0-6 (= rb - pad, the radius of the ia172_136 L / ia172_172 L tube fills) around the axis [Q, VA stations s 0..S]
-- is unioned into the LOCAL junction mask; S = 6 mm (the fallback's own extent), then +2 mm until junction6's
bottleneck ratio (largest ball Q -> VA s 10, over its inscribed radius) reaches 0.7 (stage 1's "good" level), max 16
mm. The tube + 2 mm counts as a zone for the v5-surface splice. Records: junction_bridge.after.{fallback_tube: true,
d16_tube: true, d16_tubed_mm, d16_tube_r_mm}, d16_tube {tube_mm, tube_r_mm, from_xyz, to_xyz, ratio before / after,
S, steps, how}; junction_failed = null when the ratio reaches 0.5 (sc_post3's rule).

### 8.3 D20 (and D19): mirrored and re-posed forks in the bake and the checks

bake_v6 1.1 (1.0.1 frozen as `bake_v6_v1_0_1.py`):
- Mirrored fork = xform.<key>.mirror [-1, 1, 1] on the fork's own surface file (the carotid tool's convention, = the
  TopBrain siphons' since v3): `sdf_union.load_surface` mirrors the points then re-derives outward normals on the closed
  surface (auto-orient + consistency = the winding flip) -- accepted as is; a pre-mirrored FILE with flipped winding is
  also accepted (same orientation pass). New per-section record `chirality` {mirror, mirrored, signed_volume_mm3,
  outward} in the bake marker and provenance bake_v6.
- E4: a mirror-only change = the SAME surface (a reflection keeps every arclength window): v4's suppressed stretches
  and tube decision carry over; each window end is mapped through the piece's own frame (v4 slice -> source -> v6
  slice: exact for trimmed / host-cut-shifted, re-posed or mirrored pieces) and used where it differs from 1.0's plain
  shift by > 1 mm (a host-cut shift; a blend alone stays within it, so 1.0's pieces keep their windows bit for bit). Changed sections (moved right forks included) are
  searched with the VA pool at v4's carotid margin (E3 / E3b): the RCCA sections are no longer pinned when the fork moves.
- E6 for the forks too (lower_R / lower_L: the fork-side seam-1 blend). P_rigid: (1) v5's piece back-mapped, only for
  the same surface with the SAME mirror and no blend of its own, checked <= 0.01 mm beyond the blend; else (2) the
  composer's own `blend_disp` rows ([s, dx, dy, dz], blended - rigid over the piece's first points): aligned on the
  written branch by arclength (<= 1e-3 mm), D = 0 beyond them. The fork is compared / morphed over its own slice only.
checks_v6 1.1 (1.0 frozen): the route gate is PER REAL SECTION -- fork = lower_R slice [host_cut, host_cut + cca +
min(ica, ica_real)], siphon = siphon_R slice, each tree's own provenance (the slices the bake uses), exact ownership,
2 mm trims, v6 against the same section of the baseline (v4, or v4 without its tip by D13): the moved surface against
itself. 1.0's positional split (RCCA wall < / >= 130 mm) stays as information (`gate_ratio_legacy_130`). New gate
`rcca_prefix_unchanged`: v6's RCCA points and radii up to min(host_cut v6, v4) - 0.5 mm are v4's bit for bit (host as
information). loader_one 1.1: the RCCA is checked against v5 up to seam 1 only (1.0: up to the siphon seam, which a
moved fork breaks by design); 1.0's test is kept as information.

### 8.4 Run-2 results

Chain (`wall/d17/chain/`, all steps rc 0): request runs 27 / 27 units x 2 rules (13:20; ia172_164's v5-rule run was
OOM-killed at 8 GB and re-run alone at 12 GB), local merge 13:52, VMTK 14:02, records 14:09, wall detail 14:24; slot B
(D15 report, ls11 test, D20 smoke, D17 bake, 1.1 regression) 14:10; slot C (same-line wall detail, report, figures,
publish dry run) 14:32. `D17_REPORT.json` (summary), figures `wall/d17/figs/` (`d17_sheet.png`: 8 sides, the two D16
tubes in orange).

D17 (54 sides, `wall/d17/D17_REPORT.json`):
- label end past the ostium p10 / median / p90 35.3 / 73.2 / 93.2 mm (min 19.0, max 121.1); < 30 mm (the VA tool's
  short tube): song 040 R 20.6, song 109 R 19.0; junction_failed on 0 sides; junction bottleneck ratio p10 / median /
  min 0.650 / 0.857 / 0.53 (none < 0.5); every surface closed (0 open / non-manifold edges).
- The local-fix rule held on the longer stretches: the v5-rule recompute equals the v5-rule run's file on all 108
  surfaces; outside the zones every vertex is the v5-rule run's (108 / 108) and every all-outside triangle too except 2
  on one subclavian surface (marching-cubes cells re-triangulated at a zone border, vertices identical). Subclavian
  zones (bright-cut | plateau-adds >= 0.6 mm, + 2 mm blend): core 0 mm on 21 sides, median 3.25, p90 16.4, max 43 mm.
- Handles (`d17/genus_cmp.json`, genus of the junction surfaces): sum 76 (v5 rule) / 78.5 (plateau) / 75 (local); 6
  sides gain, 6 lose -- the handles are the subclavian's own side branches, not the zones.
- Wall detail (wd_proxy, share > 0.3 mm off the tube over the whole subclavian stretch, p10 / median / p90): v5 rule
  0.332 / 0.506 / 0.640; local along the v5-rule run's OWN line 0.333 / 0.506 / 0.645; local along its own VMTK line
  0.332 / 0.496 / 0.635. Outside the zones, local vs v5 rule along the same line: 4772 of 4808 1 mm bins identical, the
  class share differs by at most 0.003 (per bin at most 0.026, the bins next to a zone).
  By class (`results/s2_harden_wall_bake.json` d17_zone_class; 34 sides have zone bins): OUTSIDE v5 rule 0.332 / 0.474 /
  0.602, local 0.332 / 0.474 / 0.602 (max per-side difference 0.003); ZONE v5 rule 0.418 / 0.635 / 0.775, local 0.440 /
  0.654 / 0.796 (the subclavian zones are short bone-contact stretches of a big lumen: the plateau label there follows
  the CT lumen into the contact instead of v5's flattened cut, so the relief stays about the same -- unlike the VA, whose
  zone relief was mostly the artefact).
- CT vs surface in the subclavian zones (`wall_d17/sub_ct_gate.py` -> `d17/sub_ct_gate.json`; bone_scan's CT read with
  the ring / ray length scaled to the station radius; 31 sides with zones, vmr0248 skipped: no thin-slice CT):

  | CT r_eq / surface r_eq (p10 / median / p90) | v5 rule | plateau | **local** |
  |---|---|---|---|
  | 185 points in the zones (every 2 mm) | 1.057 / 1.161 / 1.295; >= 1.25: 34 | 1.022 / 1.069 / 1.155; 1 | 1.022 / **1.070** / 1.156; **1** |
  | 201 reference points outside (every 10 mm, >= 5 mm from a zone) | 0.988 / 1.012 / 1.077; 1 | same | same (surface r_eq = v5 rule's within 4e-5 mm) |

  CT rays ended by a bright neighbour: median 19 % of the rays in the zones, 0 % outside -- the zones are bone /
  dense-neighbour contact, where the v5 rule's surface sat ~16 % inside the CT lumen; the local surface follows the CT
  there as plateau does, and is the v5-rule surface outside.

D16 (in the published records): ia172_164 L bottleneck ratio 0.459 -> 0.897 (tube 9.71 mm, r 1.517 mm, S 6 mm, one
step); ia172_071 L 0.291 -> 1.0 (tube 10.01 mm, r 2.077 mm, S 6 mm); junction_failed = null on both, so both sides get
the donor's own junction (the VA tool flags d16_tube_bridge with the tubed mm). ls11 test (`d17/ls11_test.json`):
local_sub 1.1 with D16 off reproduces 1.0's masks and surface md5 on 015 / 164 / 071, and the cropped-grid merge equals
the full-grid merge.

Published 16:14 host clock (the containers' clock reads 22:14): all 54 records replaced in
`donors_local/_junction_records/` (byte-checked against `wall/d17/_junction_records/`), the D15-only records in
`donors_local/_junction_records_d15only/`, marker `_PUBLISHED_D16_D17.txt`; requests va-D16-1 / va-D17-1 DONE,
va-D17-bake-1 CONFIRMED (below).

D20 / bake_v6 1.1 / checks_v6 1.1:
- Regression 1.1 vs 1.0 (`bake_v6/reg11/reg11_compare.json`): on k_004 / w_034 / w_025 the OBJ, the 60k mesh and the
  union are md5-identical, every mesh-record field equal, morph records equal; checks_v6 1.1 gates all pass (route per
  section fork / siphon: k_004 0.99997 / 1.00455, w_034 1.00000 / 0.99159, w_025 >= 0.99; rcca_prefix identical to v4
  on all three; loader 1.1 ok).
- Unit test (`bake_v6/d20/d20_test.json`): T1 E6 with the composer's blend_disp vs v5's back-map on 8 blended pieces:
  same morph, vertices within 6e-6 mm (w_034 siphon_R 0.024 mm); T2 mirror flag on the fork's own file vs a
  pre-mirrored file with flipped winding: identical points, positive signed volume (outward), centrelines inside 1.0,
  the placement an exact reflection (< 1e-13 mm); T3 arclength windows mapped through identity / rot20 / mirror / 4 mm
  trim exact to 1e-13 mm (the trim's plain shift would be ~5 mm off).
- D20 smoke on the carotid tool's compose_v6 3.0.0 trees (`bake_v6/d20smoke/`): k_011_left mr_012_L (right fork = the
  patient's left carotid, x-mirrored and map-posed) bakes and passes every bake / route / audit / lumen / loader gate;
  its orientation gate fails only on the VA tops (the smoke tree still carries v5's VAs; the VA tool re-poses them).
  w_051_right mr_008 FAILS the bake: "even the tube-only mesh has a handle: a centerline-level fusion" -- the re-posed
  right fork's ICA runs 3.06 mm into the v5 RVA (probe `bake/probe/pair_clear.py`: RCCA s 138 vs RVA s 199; v5 +1.44
  mm; the carotid record has the same soft_clear -3.061 "cc - RVA"). A composition contact, not a bake defect: the
  bake refuses fused centrelines by design. See 8.5.
- D17 cut end (`bake_v6/d17bake/`, a copy of the VA tool's unit11 w_034): bake pass at rung 1, VA whole at va margin
  0.5, scC_R / scD_R / scC_L / scD_L / scJ_L all real (0 mm tubed), E7 ok as baked (VA lumen OBJ 1.367 / 0.918);
  checks_v6 every gate passes. (That tree used the D15 records, cut end ~30-34 mm past the ostium; the D17 records
  change only scD_* hi_mm.)

### 8.5 checks_v6 1.2: whole-tree centreline geometry (part geom)

Why: each composer checks what it places against the neighbours it sees, so a contact made ACROSS tools (a re-posed
fork against a VA still at v5's pose, or the reverse) is caught only by an independent check -- or by the bake failing.
`checks/geom_v6.py` (needs only the tree, so it runs before the bake too) + checks_v6 1.2 part `geom` (1.1 frozen as
`checks_v6_v1_1.py`; the other parts unchanged):
- clearance: lumen gap d - r_a - r_b between curves of different placement groups (CR {RCCA, RECA}, CL {LCCA, LECA},
  VR {RVA, (17)}, VL {LVA, (18), (19)}, every other curve its own), over point pairs with at least one point that is
  not v5's; joins (an end point inside the other tube) left out over the child's fused run + 15 mm (the fused run
  matters for the arch: a 15 mm cut alone left LCCA-arch overlaps of -3 to -4 mm on three left-fork anatomies).
  Gate `geom_clearance_new_ge_0.6`.
- 1 mm turn (graft_siphon.kink_deg on the written curve, the composers' join measure) at every new point: <= 60 deg or
  no sharper than the same v5 curve's largest turn + 1 deg. Gate `geom_kink_new_le_60`.

Calibration / results (`bake_v6/geom/`):

| tree | anatomies | clearance fails | kink fails |
|---|---|---|---|
| v5 against v4 (v5's own changes) | 223 | 0 | 0 |
| stage-1 pilot tree (`v6_work/pilot/tree`) | 14 | 0 | 0 |
| bake_v6 regression / D17 bake trees | 3 + 1 | 0 | 0 |
| D20 smoke (carotid 3.0.0, v5 VAs) | 2 | 1: w_051 RCCA-RVA -3.06 (the bake failure) | 0 |
| carotid tool's written trees `carotid/s3/w223` at 16:20 (v5 VAs still in place) | 165 | 17: RCCA-RVA -4.45 ... -0.05 mm (+ RECA-RVA -0.71 on w_018b mr_007) | 0 |
| VA tool's unit11 / unit13 tree_car3 | 14 + 14 | 0 | 1 + 1: w_018b mr_002 LVA 61.0 deg at s 15.0 |
| VA tool's unit13 tree_j13 | 65 | 0 | 2: k_004 mr_014_L (19) 152.1 deg at s 4.9 (a hairpin); m_006 mr_002 LVA 61.0 deg at s 15.0 |

(v5 with every point counted as new: clearance 1 / 223, w_037_right mr_010_L RECA-RVA 0.475 mm -- a v4 contact; the
kink scan flags the host arch (0) at 90 deg and some v4 ECA tip joins at 65-81 deg on every anatomy, which is why the
kink gate is relative to v5.)
- The 17 fork-VA contacts on the carotid trees all have a donor RVA in v5 (vaC_R), and v4 / v5 had deflected the RVA
  there ("RVA deflected 1.8-7.8 mm" in their repairs): the VA tool's joint pass re-places these RVAs and its trees
  checked here have none (k_004 mr_014_L: 1.387 mm). Any left after the VA stage fails geom (and the bake).
- The VA-tree kink findings went to the VA agent as FYI `wall-geom-1` in `wall/requests.json` (the hairpin breaks D17's
  "no hairpin"; the 61 deg turns are 1 deg over the 60 deg 1 mm join limit at the proximal seam).
- geom was added to the existing reg11 (3), d17bake (1) and d20smoke k_011 check records with `--parts geom`: both geom
  gates pass on all five (their "script" field now reads 1.2; the other parts were computed by 1.1, whose code 1.2
  keeps unchanged). Full gate sets: reg11 3 / 3 PASS, d17bake PASS, d20smoke k_011 FAIL only on
  orient_v6_blocking_after_D8 (v5 VA tops in the carotid-only smoke tree).
