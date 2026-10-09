# v6_wall: the donor wall fix for v6 (V6_ITEMS c1-c4, decision D5)

Task (orchestrator, stage 1): fix the ct_seg4 bone-side artefact and/or the 1.0 mm graft floor for the v6 donors, try
at least two variants, re-extract the donor VA surfaces of the 27 donor units, gate the choice (CT-vs-surface ratio at
the audit's 140 floored + 136 reference points, VA lumen on the polyline, wall detail kept coarse), inspect the 8
CT-narrow points (c3), decide the topcow_mr_021 left-VA TopBrain pinch (c4).

Code: `scratchpad/v6/wall/` (/sp/v6/wall in containers). Large outputs: `D:\Arjun\neve\carotid_data\v6_work\wall\`
(/v6w in containers; git-ignored). State: `scratchpad/v6/state/v6_wall.md`. Read-only on the repo and on v5's data.

## Answer

- **Mechanism (c1), reproduced bit for bit:** the cut is ct_seg4's LOWER threshold on bone-side rays. A ray that runs
  from the lumen straight into bone finds no soft-tissue crossing, so v5 takes its exterior level after the steepest
  descent, inside the bone: T_loc 530-690 HU against a 480-525 HU lumen; the ray ends at 0.3 R and the threshold
  smoothing spreads it. Not the upper (bright-neighbour) threshold. (section 1)
- **Variants:** `half` (cap that lower threshold at the soft-tissue half max: the ray ends half-way to the bone peak),
  `plateau` (half + end the ray at the lumen plateau's end, the valley nearest the bone rise), F (keep the v5 surface,
  floor the graft radius at the CT read), and `plateau_g` (first plateau form, global minimum; 12 units, dropped). At
  the w_025 point: v5 1.13, half 2.22, plateau 1.98 against 1.90-1.96 on both sides. (section 2)
- **Chosen: plateau** (reversible). Records for all 27 donor units: `carotid_data/v6_work/wall/donors/plateau/`
  (donors_v5 schema; `donors/CHOSEN.json`). At the audit's 140 floored points CT / surface goes from a median 1.584 to
  1.155 (>= 1.5: 83 -> 12), the reference points stay at 1.04, and the calibre across a foramen (continuity) goes from
  0.64 to 0.94. Donor VAs using a floored stretch (the audit's floor_scan) 377 -> 139 of 429 (361 -> 117 on the
  proxy's used donor stretch); floored length on the walls 1084 -> 452 mm; the remaining floored stretches >= 2 mm (26,
  v5 140) are small on CT too (median CT read 1.26 mm, v5 1.78; 8 of 26 still >= 1.5x), so the graft's 1.0 mm floor
  stays as the safety net and no CT-calibre floor is used (F was evaluated: it keeps v5's cut walls as plain tubes).
  An Arterial-mask bone side was not built. half reads
  slightly closer to the CT (1.099, >= 1.5: 6) but its surfaces leak into the bone halo (28 points reach > CT + 1.5 mm
  against 15) and carry more handles. (section 3)
- **D5 (coarse wall) kept:** donor-stretch wall detail median 0.134 (v5 measured) -> 0.112 (plateau; proxy calibrated
  at r 0.991), above the plan's 0.10. On the 71 % of the wall v5 segmented correctly it is unchanged (0.078 -> 0.071);
  the drop is where v5's cut acted (29 % of the wall, relief 0.33-0.37 = the artefact), which still keeps 0.20-0.27.
  Per donor side before / after in section 3 (6 of 49 sides keep <= 0.51 of v5's value). (section 3)
- **c2:** `wall/va_lumen_poly.py` (0.05 mm polyline, exact distance, OBJ / full / union) reproduces the audit's minima
  on v5 exactly (4 VAs < 0.65 of 446). On the proxy unions the 4 narrow VAs' donor stretches go from 0.92-0.93 to
  0.99-1.42 mm, all floored mm in their used stretch -> 0; the OBJ check runs in stage 2. (section 3)
- **c3:** 4 of the 8 CT-narrow points are v5's cut (plateau restores Arterial's calibre within 0.1 mm; the v5
  centreline sat at the lumen edge); 4 are mild real narrowings (20-30 % in diameter) inside a transverse foramen,
  unchanged by the fix. None is a trace leaving the lumen. Figures `c3/*.png`. (section 5)
- **c4: tube** the topcow_mr_021 left-VA TopBrain pinch at the neighbouring calibre (s_tb 33.5-53, r 1.84 -> 1.64 mm;
  `c4/c4_tube_override.json`, 8 anatomies), applied by the stage-2 grafter and kept as v4 heritage only where a bake
  gate fails. Reason: the low stretch is exactly the horizontal V3 run (in-plane TOF saturation pattern), calibre
  1.9-2.3 on both sides, and with the c1 fix it becomes the narrowest point of those VAs. (section 6)
- **(e) subclavian stretches:** all 54 requested sides done with plateau, plus a junction fix (the VA and subclavian
  labels did not touch in 23 of 54 sides) and a subclavian handle cleanup (1-voxel spurs gave genus up to 57). Final
  (v3, published for all 27 units in `subclavian/<u>/`, marker `.junction_v3`): every junction connected, bottleneck
  ratio median 0.85 (>= 0.7 on 45 of 54), tube fill on 2 sides, subclavian surface genus 0 on 43 of 54, junction VA
  part identical to the donor record's VA beyond s 5. Flag: ia172_071 L (VA trace leaves the lumen at the ostium ->
  treat as failed); ia172_071 R and song 094 L are CT-consistent ostial narrowings. Subclavian wall detail (core)
  median 0.53: coarse, not a tube (D5). All requested stretches lie inside the donor CTs. (section 7)

## 1. Mechanism (reproduced)

`wall/ctseg6.py` is v5's `scaleup_ct_seg.py` (ct_seg4, pipeline v2) copied with one env switch `V6_RULE`; with
`V6_RULE=v5` it reproduces v5's walls bit for bit (labels_kji.npz, stations_rays.json and both stage-A surfaces equal:
Song AHMU1218113 = a pilot wall, `seg/v5repro/regress.json`; the three pipeline-v2 cases below). The per-donor
configuration that reproduces each v5 wall (pilot vs pipeline v2, select file, pinch fix, pinch-skip file) is read from
each wall's own seg_qc.json (`wall/make_donors.py` -> `wall/donors.json`; 5 pilot walls: ia172_058 / 039 / 190 / 141,
Song 113).

The diagnostic dump (`V6_DIAG=rva:76:96`, read by `wall/diag_view.py`) at the w_025 LVA point (Song 113 R, donor
s 80-91, V2 at about C4) shows exactly one mechanism:
- every ray that ends at the 0.3 R start floor ends because the smoothed LOWER threshold Tp is above the lumen HU at
  0.5 mm (13-25 such rays per station, 0 by the upper threshold);
- every one of those is a bright-neighbour ray (kind 3) whose pre-override rule was "steepest descent" (kind 2): no
  soft-tissue crossing before R + 3 mm, because the ray runs from the lumen straight into bone;
- for those rays v5 takes the exterior level I_ext = min over 0.8 mm after the STEEPEST DESCENT, which lies inside the
  bone (the far flank of the cortical peak): T_loc = (I_c + I_ext)/2 = 530-690 HU against a lumen of 480-525 HU. The
  ray ends at its first sample, and the 2-ray x 2-station smoothing of the threshold field spreads it to neighbours;
- the 3-D label then also inherits the CAP (rho <= r_e + 0.5 mm) of those rays: <= 0.8-0.9 mm on the bone side.

So the cut is the LOWER threshold of bone-side rays (not the upper T_hi rule, which never cut a lumen sample here).

## 2. Variants

| variant | rule (bright rays only; free-side rays unchanged) |
|---|---|
| `half` ("half-way to the bone peak") | lower threshold capped at the soft-tissue half max T_s = (I_c + I_bg)/2; the ray then ends at v5's own upper threshold T_hi = (I_c + bone peak)/2 (min-filtered 5 stations x 3 rays) |
| `plateau` ("end of the lumen plateau") | `half` + the ray ends at the deepest point between 0.3 R and the bright rise when it is >= 10 HU below the plateau before it (the wall / gap between lumen and bone); the 3-D label there is capped at rho <= r_v + 0.15 mm. Straight rises (no gap) keep the half-way rule |
| F (floor at the CT calibre, no re-segmentation) | v5 surface; graft radius floor = the CT read instead of 1.0 mm where MISR < 1.0 (evaluated with the wall-detail proxy only) |

Song 113 R (`prof/song_AHMU1218113_R_70_100.json`): at the w_025 point (s 85.9) the surface r_eq is v5 1.13,
half 2.22, plateau 1.95, against 1.90-1.96 on both sides of the contact; outside the contact all three are identical.
bone_scan's CT read there is 2.92: its bone-side rays end at I_c + 150 HU, inside the bone's partial-volume rise, so at
bone contact it reads wide (the audit itself read the lumen as 1.3 + 2.2 = 3.5 mm wide there).

## 3. Gates

### Gate 1: CT against surface at the audit's 140 floored + 136 reference points (`wall/gate_ct.py`)

Same points (s on the v5 donor line, native frame), same plane (tangent P[j+2] - P[j-2]), same section functions
as bone_scan.py; the CT read is bone_scan's own value (copied). `seg/<variant>/gate_ct.json`.

| quantity (p10 / median / p90) | v5 (reproduces the audit) | half | **plateau (chosen)** |
|---|---|---|---|
| floored points: surface r_eq | 0.734 / 1.041 / 1.47 | 1.234 / 1.572 / 2.083 | 1.178 / 1.482 / 1.977 |
| floored: CT / surface | 1.027 / 1.584 / 2.442 | 0.766 / 1.099 / 1.355 | 0.803 / 1.155 / 1.487 |
| floored: CT / surface >= 1.25 / >= 1.5 | 119 / 83 of 140 | 27 / 6 | 40 / 12 |
| floored: surface wider than CT by 25 % (ratio <= 0.8) | 7 | 15 | 14 |
| floored: surface section reaching > CT r_eq + 1.5 mm (leak indicator) | 4 | **28** | 15 |
| floored: surface r_eq < 1.0 | 62 | 7 | 7 |
| reference points: CT / surface | 1.00 / 1.041 / 1.331 | 0.999 / 1.04 / 1.344 | 1.00 / 1.04 / 1.331 (free side unchanged) |
| continuity: floored point / its reference point (<= 30 mm, same side) | 0.439 / 0.642 / 0.846; < 0.75: 107 of 136 | 0.788 / 0.983 / 1.166; < 0.75: 13; > 1.25: 6 | 0.755 / 0.938 / 1.098; < 0.75: 13; > 1.25: 4 |
| stage-A surfaces: genus sum / multi-component sides | 356 / 2 (54 sides) | 173 / 2 (52 sides) | **149 / 0** (54 sides) |
| floored length (MISR < 1.0) total / median per side | 1084 / 16.7 mm | 404 / 3.7 mm | 452 / 4.3 mm |

(`gate_summary.json`; half has 25 units: no VMR 0248 run, and ia172_136 L's half VMTK line failed.)

The CT read itself is biased wide at bone contact (its bone-side rays end at I_c + 150 HU, in the bone's partial-volume
rise: 2.92 at the w_025 point, where the audit read a 3.5 mm-wide lumen), so a median CT / surface of 1.155 at contact
points with 1.04 at the reference points is close to the attainable; continuity (a VA keeps its calibre across a
foramen) is the cleaner measure, and it went from 0.64 to 0.94.

Leaks: the 14-15 indicator points are mostly where the foraminal plexus is iso-dense with the lumen (5 are c3
points whose CT read failed, so ratio <= 0.8 there is meaningless); real local leaks (section reaching 4-7 mm) at
song 094 L s 106-113 (a vein-contact stretch that already leaks in v5: v5 r_eq 2.4-3.3 there), song 002 L s 105-111,
ia172_149 R s 105-106, ia172_165 R s 38-39 (`prof/*.json`). They add some relief that is not anatomy; they are
local (<= 6 mm) and the bake's handle rule tubes any handle they make.

### The audit's scans re-run on the v6 records (`floor_scan6.py`, `bone_scan6.py` -> `scan/plateau/`)

| | v5 (audit) | v6 plateau |
|---|---|---|
| donor VAs using a floored stretch (MISR < 1.0) in their used range | 377 of 429 | 139 of 429 |
| floored length per VA, median / p90 / max | 11.2 / 35.9 / 72.1 mm | 0 / 7.1 / 24.5 mm |
| floored stretches >= 2 mm read on CT | 140 | 26 |
| their CT r_eq median | 1.78 | 1.26 |
| CT / surface >= 1.25 | 119 of 140 | 12 of 26 |
| CT / surface >= 1.5 | 83 of 140 (489 mm) | 8 of 26 (27 mm) |
| CT r_eq < 1.0 with no bright contact | 8 | 8 (the c3 type: real foramen narrowings and failed reads) |
| reference points CT / surface | 1.04 | 1.047 |

### Gate 3: wall detail kept coarse (D5) -- `wall/wd_proxy.py`, `wd_collect.py` -> `wd/`

No bake exists before stage 2, so the donor union is rebuilt per donor side exactly as the bake builds it
(topbrain_tools tube_field + add_real_sections: max(tube, min(real, capsule)), 0.45 mm grid, tube = MISR floored at
1.0 on the prep_siphon 1 mm line) and measured with v5's definition (share of wall area > 0.3 mm off the tube, exact
ownership over all 0.25 mm samples, undecimated), in 1 mm bins, then aggregated over every v5 donor VA's used donor
stretch [ostium advance + 2, splice - 25] (v5's donor_real). **Calibration on v5's own inputs against v5's measured
donor_real (vvw, the wall verifier's exact-ownership measure on iso_v5): Pearson r 0.991, median |difference| 0.003,
p90 0.008 (proxy median 0.140 against measured 0.134), 429 VAs.**

| per-VA donor stretch, > 0.3 mm (429 VAs) | p10 / median / p90, min / max |
|---|---|
| v5 measured (vvw) | - / 0.134 / - |
| v5 proxy | 0.053 / 0.140 / 0.250, 0.030 / 0.328 |
| **plateau (chosen)** | 0.045 / 0.112 / 0.200, 0.015 / 0.248 |
| half (412 VAs: no VMR 0248, no ia172_136 L) | 0.048 / 0.132 / 0.222, 0.023 / 0.285 |
| F (v5 surface, floor at the CT read) | 0.051 / 0.108 / 0.229, 0.022 / 0.297 |

The fix LOWERS the measured relief, and the 1 mm bins say why (the reproducible class table "Where the relief changes"
below supersedes a first inline estimate): where v5's cut acted the lumen is flattened and off-centre, and its free
side stands 0.3-1 mm off the floored 1.0 mm tube, which the measure counts as wall detail. So part of v5's 0.134 was
the very artefact c1 removes. The plateau median 0.112 is still above the plan's D5 acceptance (>= 0.10) and 6x v4's
whole-VA 0.018; floor-at-CT (F) lands at the same level but with plain tubes at every contact (and a CT-read-inflated
calibre there). Per donor side: `wd/wd_collect.json` -> per_donor_side.

**Per donor side, before / after** (median over the v5 VAs that use the side; share of wall > 0.3 mm on the used donor
stretch; v5 measured = vvw exact ownership on iso_v5, the others the calibrated proxy; `chain/collect.log`):

| donor side | VAs | v5 meas | v5 proxy | F | **plateau** | half |
|---|---|---|---|---|---|---|
| ia172_015 L / R | 16 / 16 | 0.180 / 0.192 | 0.175 / 0.184 | 0.080 / 0.105 | 0.112 / 0.131 | 0.159 / 0.153 |
| ia172_039 L / R | 1 / 1 | 0.175 / 0.089 | 0.175 / 0.088 | 0.175 / 0.088 | 0.157 / 0.095 | 0.168 / 0.102 |
| ia172_058 L / R | 8 / 9 | 0.336 / 0.094 | 0.327 / 0.089 | 0.274 / 0.089 | 0.231 / 0.085 | 0.274 / 0.096 |
| ia172_069 R | 2 | 0.281 | 0.273 | 0.240 | 0.238 | 0.256 |
| ia172_071 L / R | 4 / 2 | 0.293 / 0.082 | 0.280 / 0.080 | 0.206 / 0.063 | 0.248 / 0.067 | 0.281 / 0.077 |
| ia172_136 L / R | 4 / 5 | 0.197 / 0.180 | 0.193 / 0.169 | 0.159 / 0.124 | 0.194 / 0.123 | - / 0.162 |
| ia172_141 L / R | 4 / 4 | 0.029 / 0.143 | 0.030 / 0.142 | 0.023 / 0.121 | 0.018 / **0.072** | 0.023 / 0.100 |
| ia172_149 L / R | 13 / 15 | 0.091 / 0.308 | 0.090 / 0.306 | 0.080 / 0.249 | 0.097 / 0.247 | 0.107 / 0.265 |
| ia172_159 L / R | 9 / 15 | 0.171 / 0.172 | 0.173 / 0.169 | 0.133 / 0.128 | 0.101 / 0.107 | 0.115 / 0.133 |
| ia172_164 L / R | 16 / 16 | 0.124 / 0.095 | 0.123 / 0.098 | 0.080 / 0.098 | **0.049** / 0.091 | 0.061 / 0.098 |
| ia172_165 L / R | 12 / 11 | 0.160 / 0.201 | 0.156 / 0.197 | 0.135 / 0.157 | 0.156 / 0.198 | 0.158 / 0.204 |
| ia172_190 L / R | 15 / 15 | 0.036 / 0.071 | 0.036 / 0.070 | 0.025 / 0.051 | 0.028 / 0.045 | 0.030 / 0.048 |
| ia172_192 L / R | 9 / 10 | 0.178 / 0.086 | 0.170 / 0.084 | 0.159 / 0.079 | 0.153 / 0.080 | 0.162 / 0.084 |
| song 002 L / R | 16 / 16 | 0.110 / 0.250 | 0.106 / 0.250 | 0.104 / 0.229 | 0.130 / 0.191 | 0.135 / 0.197 |
| song 008 L / R | 5 / 5 | 0.309 / 0.244 | 0.304 / 0.240 | 0.297 / 0.234 | 0.230 / 0.210 | 0.241 / 0.222 |
| song 032 L / R | 2 / 2 | 0.073 / 0.055 | 0.071 / 0.055 | 0.058 / 0.045 | 0.076 / 0.038 | 0.085 / 0.048 |
| song 040 L / R | 8 / 8 | 0.089 / 0.035 | 0.086 / 0.032 | 0.040 / 0.024 | **0.032 / 0.015** | 0.062 / 0.027 |
| song 094 L / R | 4 / 4 | 0.324 / 0.145 | 0.328 / 0.143 | 0.264 / 0.139 | 0.244 / 0.150 | 0.285 / 0.155 |
| song 095 L / R | 8 / 8 | 0.229 / 0.166 | 0.222 / 0.161 | 0.158 / 0.130 | **0.113 / 0.078** | 0.158 / 0.132 |
| song 100 L / R | 6 / 6 | 0.075 / 0.109 | 0.074 / 0.108 | 0.074 / 0.108 | 0.071 / 0.108 | 0.070 / 0.110 |
| song 108 L / R | 16 / 16 | 0.146 / 0.128 | 0.140 / 0.124 | 0.140 / 0.114 | 0.130 / 0.134 | 0.132 / 0.139 |
| song 109 L / R | 15 / 15 | 0.102 / 0.056 | 0.100 / 0.053 | 0.100 / 0.051 | 0.090 / 0.056 | 0.102 / 0.058 |
| song 113 L / R | 8 / 8 | 0.226 / 0.188 | 0.222 / 0.184 | 0.177 / 0.156 | 0.159 / 0.115 | 0.201 / 0.160 |
| song 141 L / R | 4 / 4 | 0.095 / 0.034 | 0.091 / 0.035 | 0.091 / 0.035 | 0.089 / 0.031 | 0.100 / 0.034 |
| vmr0248 L / R | 6 / 7 | 0.137 / 0.059 | 0.209 / 0.059 | - | 0.177 / 0.078 | - |

(ia172_172 and song 138 are reserves: no v5 VA uses them.) Plateau keeps >= 0.8 of v5's measured value on 28 of 49
donor sides (counted from the table); the 6 sides that keep <= 0.51 of it (bold: ia172_141 R, ia172_164 L, song 040
L/R, song 095 L/R) are the
ones whose v5 relief sat largely in floored stretches -- F (v5's own surface, only the floored stretches widened to the
CT calibre) already loses about half of it on 015 L and 040 L and a third on 164 L (where the drop happens: see the
bin analysis below).

**Where the relief changes** (`jfix/wd_bins.py`, `jfix/wd_cut.py` -> `wd/wd_bins.json`, `wd/wd_cut.json`): every
used 1 mm donor bin (60 404 VA-bin uses) classed by what v5's ct_seg4 did there: FLOOR (v5 MISR < 1.0 in the bin),
NEAR (within 2 mm of FLOOR), CUT (v5's bone-side cut acted -- >= 2 rays of a station end >= 0.3 mm shorter in v5 than
in plateau -- without reaching the floor), FREE (the rest: v5 and plateau segment the wall alike).

| class (share of bin uses) | v5 proxy | F | **plateau** | half |
|---|---|---|---|---|
| FLOOR (8 %) | 0.335 | 0.132 | 0.218 | 0.303 |
| NEAR (7 %) | 0.375 | 0.220 | 0.196 | 0.244 |
| CUT (14 %) | 0.336 | 0.332 | 0.274 | 0.304 |
| FREE (71 %) | 0.078 | 0.078 | 0.071 | 0.071 |

So 29 % of v5's used donor wall -- exactly where its bone-side cut acted -- carried relief 0.33-0.37, four times the
0.078 of the wall it segmented correctly; that excess is the cut's eccentric, flattened lumen against a tube. On the
correctly segmented wall plateau keeps the relief (0.078 -> 0.071, -9 %, the re-centred line / new MISR), and at the
former cut sites it still has 0.20-0.27 (real bone-contact dents, 3x the free wall). The 6 bold donor sides lose it
in the CUT bins (e.g. song 095 R 0.261 -> 0.107, ia172_141 R 0.275 -> 0.156); two also lose in FREE bins (ia172_141 R
0.072 -> 0.038, song 113 R 0.103 -> 0.069; not traced further -- the FREE class only tests the ray ends, so a
re-centred line or a sub-0.3 mm contour change there is not excluded).

**Why plateau and not half, although half keeps more relief (0.132):** half ends a bright ray half-way up the bone's
partial-volume rise, i.e. beyond the lumen-bone valley wherever one exists, so its extra relief is bone halo, not
lumen: its surfaces reach > CT r_eq + 1.5 mm at 28 of the 140 floored points (plateau 15, v5 4) and carry more handles
(genus sum 173 on 52 sides against 149 on 54). Plateau ends the ray at the valley (the vessel wall / periosteum gap),
which is where the lumen ends; its CT / surface (1.155) is slightly above half's (1.099) only because bone_scan's CT
read itself ends inside the bone rise (I_c + 150 HU). Both pass the plan's D5 acceptance (donor median >= 0.10).
Reversible: `seg/half/` holds half's walls for 25 units; `python3 /sp/v6/wall/build_donors6.py half` builds its records
(fix ia172_136 L's 0.1 mm VMTK line first, as done for plateau's ia172_015 L).

Union lumen on the polyline (same proxy unions, undecimated): median v5 0.935 (the floored tube) -> plateau 1.049
(p10 0.939, max 1.67); no VA under 0.9 in either. VAs whose used stretch touches the floor: 361 -> 117.

### c2: the VA lumen on the polyline (`wall/va_lumen_poly.py`)

New checker for stage 2 (and any tree): every VA polyline resampled at 0.05 mm (plus its control points, 3 mm end caps
out), exact point-to-triangle clearance (vtkStaticCellLocator) to the OBJ and collision_full (and an undecimated union
with `--union-dir`), inside by topo_audit's winding number, evaluated once per crossing-free run (a pair of samples can
only straddle the wall when d_i + d_i+1 <= their spacing) -- exact for a closed mesh. Per VA: minimum, where, which
section (junction / donor / seam zone / TopBrain / host, from provenance), the control-point minimum beside it, the
samples below 0.65 mm and their s-runs.

Validation on v5 (`lumen/test_v5_4.json`): w_025_right mr_021_L RVA 0.547 at s 158.7 (donor; control points 0.632 at
219.7), LVA 0.565 at 92.5, w_027_right mr_022 RVA 0.630 at 93.4, w_047_left mr_023 RVA 0.637 at 152.8 -- the audit's
verifier numbers (0.547 / 0.567 at 0.1 mm / 0.630 / 0.637), control-point values = checks/va_lumen.py's. About 2.5 s
per anatomy (OBJ + full). All 446 v5 VAs (`lumen/va_lumen_poly_v5.json`): OBJ minimum < 0.65 in exactly those 4,
median 0.818. Stage 2 runs it on the v6 tree (no bake exists before then).

Before the bake, the same polyline measure on the per-donor proxy union (wd_proxy: 0.05 mm samples, exact distance,
undecimated) over each v5 VA's used donor stretch: median 0.935 (v5) -> 1.049 (plateau), minimum 0.915 -> 0.922, none
under 0.9 in either; VAs touching the floor 361 -> 117. The audit's 4 narrow VAs (`jfix/narrow4.py` -> `wd/narrow4.json`):

| VA (donor side) | v5 floored mm | plateau floored mm | union lumen min v5 / plateau / half |
|---|---|---|---|
| w_025_right mr_021_L RVA (song 113 L) | 10 | 0 | 0.930 / 1.222 / 1.346 |
| w_025_right mr_021_L LVA (song 113 R) | 12 | 0 | 0.924 / 1.423 / 1.400 |
| w_027_right mr_022 RVA (ia172_192 R) | 3 | 0 | 0.926 / 1.077 / 1.080 |
| w_047_left mr_023 RVA (ia172_190 R) | 10 | 0 | 0.933 / 0.990 / 0.998 |

(The OBJ values 0.55-0.64 come after decimation; the proxy is the undecimated union, so the gate stays the stage-2
polyline check on the OBJ. w_047's donor stretch (ia172_190 R, MISR median 1.51) keeps one spot at about 1.0 mm (its
16 floored mm under plateau lie outside the used stretch); w_025 RVA keeps the c4 TopBrain pinch unless the c4 override is applied.)

## 4. Re-extraction (donor records)

`wall/build_donors6.py <variant>` = v5's builder `graft_v5_reg_donor.py` imported unchanged and called as v5_assign
called it (`build_unit` with the final eligible list's pair), with three inputs swapped per side: the wall files (the
variant's surface + its VMTK MISR line, md5 updated), the seg folder's landmarks.json (v5's landmarks remapped onto the
new line by nearest point; the vertebra-level definitions do not change with a wall fix), and v5's edge.json /
measure.json copied (contamination zones; the coarseness numbers the assignment ranked by -- the assignment is not
re-run, so they stay v5's and are labelled so). VMR 0248 goes through v5's `build_vmr` with a stand-in derived root.
Records: `carotid_data/v6_work/wall/donors/<variant>/<unit>/{record.json, R.json, L.json, R_surface.vtp,
L_surface.vtp}` = donors_v5's layout and schema, every path written as `/repo/carotid_data/v6_work/wall/...` (the
graft containers mount the repo); each side record gains a `v6` block (variant, rule, seg folder, v5 record and
surface md5, floored length MISR < 1.0 before / after, MISR median before / after, landmark remap shifts).

For stage 2 (the grafter): point graft_v5_reg's donor root (`--donor-root`, CFG donor_root) at
`/repo/carotid_data/v6_work/wall/donors/<chosen>`; nothing else changes (same schema). The graft's VA_MIN_R = 1.0 floor
stays as a safety net: after the fix it only acts where the donor is genuinely that small.

## 5. The 8 CT-narrow points (c3)

`wall/c3_inspect.py plateau` -> `c3/c3_<unit>_<side>_<s>.png/.json` (radius profiles of the own half-max CT read,
bone_scan's read, v5 / plateau surfaces, v5 MISR, Arterial's VA mask and full segmentation, core / ring / centreline HU;
5 CT cross-sections with the contours; two straightened views; a wide cross-section).

Why bone_scan's read was under 1.0 with "no bright contact": at all 8 points the 3-5 mm ring is NOT soft tissue -- its
median is 370-960 HU (the transverse foramen's bone and the enhancing foraminal venous plexus). bone_scan excludes ring
values > 500 HU, so its "background" is the plexus at about the lumen's own HU (Ib 302-435 against Ic 303-357 at the
5 failed reads): the half-max is crossed at once (r_eq 0.05). Its bright rule (HU > I_c + 150) never fires because the
plexus is not brighter than the lumen.

| point | verdict | evidence | v5 / plateau / Arterial r_eq |
|---|---|---|---|
| ia172_015 R s 151.1 (C1-C2) | **not a narrowing**: low lumen contrast in the C1-C2 foramen; v5 surface cut | lumen continuous in both straightened views at ~3 mm; v5 MISR 0.56 | 0.86 / 1.46 / 1.52 |
| ia172_039 R s 126.2 (just below C2) | **not a narrowing**: v5 surface cut to a small blob at the lumen edge, v5 centreline at that edge | CT lumen ~3 mm wide and continuous | 0.63 / 1.39 / 1.49 |
| ia172_039 L s 125.5 | **not a narrowing**: v5 cut | CT lumen continuous; plateau has a small lobe at s 128 into iso-dense foraminal tissue | 1.38 (0.9-1.3 at s 126-130) / 1.65 / 1.70 |
| ia172_071 L s 110.3 (V2) | **not a narrowing**: v5 cut over 13 mm, v5 centreline at the lumen edge | CT lumen ~4 mm wide | 1.12 / 1.90 / 1.96 |
| ia172_058 R s 47.7 (C6 foramen entry) | **plausible real mild focal narrowing** | own CT read 0.66-0.75 over ~3 mm, Arterial 1.14 against 1.4-1.6 around, lumen visibly narrower in view 1 | 0.92 / 0.84 / 1.14 |
| ia172_071 R s 87.0 and 89.2 (V2 foramen) | **plausible real mild narrowing** of a small VA in a tight foramen | CT read 0.72-0.85, Arterial 0.9-1.2 against 1.15-1.5 | 0.87-1.01 / 0.87-1.29 / 0.99-1.19 |
| ia172_190 L s 120.2 (C2-C3 foramen) | **plausible real mild narrowing** | lumen ~2 mm wide against ~3 mm at +-6 mm, Arterial 1.34 against 1.5 | 1.00 / 1.00 / 1.34 |

So 4 of the 8 are the same bone / plexus-side cut (v5's ct_seg4), which the plateau rule restores to Arterial's calibre
(within 0.1 mm: 1.39-1.90 against 1.49-1.96); 4 are mild real narrowings (about 20-30 % in diameter) inside a transverse foramen, unchanged by the
fix, with the lumen at about 0.85-1.0 mm radius where the graft's 1.0 mm floor still applies. None is a trace problem
that leaves the lumen (the v5 centreline sits at the lumen edge at the 3 cut points, because it is the MISR line of the
cut surface; the new lines are re-centred by the new surfaces).

## 6. c4: the topcow_mr_021 left-VA TopBrain pinch

Known from the audit (`notes/va_narrow.md` section 3, `tbo/va_narrow/tb_w025_RVA.json`): label 24 (L-VA) of
topcow_mr_021 narrows to r_eq 0.70-1.04 against 2.1-2.4 mm 10-15 mm away, 4.4-5 mm above the label's lower end, on a
horizontal stretch (the V3 loop) -- the classic place for in-plane TOF flow saturation. No MRA is on disk, so the
artery cannot be checked; v4 carries the same point (v4 OBJ 0.850 / full 0.864, v5 OBJ 0.632 / full 0.923).
8 v5 anatomies use this TopBrain piece (xform va_* surface topcow_mr_021_lVA.vtp).

Measured here (`wall/c4_check.py` -> `c4/c4_check.json`, on the TopBrain piece in its own frame and on every v5
anatomy that carries it, with the v5 polyline lumen of `lumen/va_lumen_poly_v5.json`):
- the piece's surface r_eq along its centreline: 2.1-2.3 mm up to s 33, falling to **0.67 at s 40.5** (centreline
  radius 0.44), then a reduced plateau of 1.15-1.55 over s 43-50, back to 1.9-2.1 from s 52. The low stretch
  (< 0.75 x the neighbouring calibre) is s 36.5-50 (13.5 mm), and over s 39-49 the label's z is constant (-82.4 to
  -82.7 mm): the whole low stretch is the horizontal V3 segment. A narrowing that starts where the vessel turns into
  the axial plane and ends where it leaves it is the in-plane saturation pattern of a TOF acquisition, not a focal
  stenosis (no MRA on disk to confirm).
- all 8 anatomies use the pinch (it lies 31.8 mm past each cut, in use): w_025_right mr_021_L RVA (OBJ TopBrain
  minimum 0.632), w_050_left mr_021_L RVA 0.719 (= that VA's narrowest point), w_012_left 0.787, w_007_left 0.792,
  w_013_right 0.792, w_039_right 0.825, m_022_left 0.836, w_040_left mr_021 LVA 0.869; collision_full 0.866-0.900.
  The graft radius there is the 1.0 mm floor (centreline radius 0.40-1.0).

**Decision: tube it** (reversible; recorded for stage 2). Reasons: (1) the evidence points to an acquisition artefact
of the TOF label (the low stretch is exactly the in-plane horizontal run, the calibre is 1.9-2.3 mm on both sides);
(2) with the c1 fix the donor floors no longer set the VA minimum, so this artefact becomes the narrowest point of
those VAs (already in w_050; w_025's 0.632 is below the 0.65 wire criterion); (3) it is trivial: one TopBrain piece,
a radius override in the grafter's input, no new geometry code -- the piece's centreline radius over TopBrain-native
arclength s 33.5-53.0 mm = max(own, a linear ramp between its own centreline radius at the two ends, 1.84 -> 1.64 mm),
i.e. the neighbouring calibre in the graft's own convention (the real surface still adds the rest outside the
stretch; inside it the pinched label lies within the tube). Spec: `c4/c4_tube_override.json`. The stage-2 grafter
applies it to the 8 anatomies; where a bake gate fails (clearance >= 0.6, genus 0, join <= 60) that anatomy keeps the
v4-heritage pinch, recorded per anatomy. Cost: ~20 mm of the piece becomes plain tube in 8 anatomies (D5: the relief
lost there is the artefact's).

## 7. Subclavian stretches for (e)

Coordination by files with v6_va: their request `wall/subclavian_requests.json` (READY 20:41, 54 entries = 27 units x
R/L, VMR 0248 included); my answer `wall/subclavian_answer.json` (what each request produces and where, status, the
junction fix). All 54 sides were run with the plateau rule (`sub_batch.py`, chain3 sub_pilot / sub_rest) and their
subclavian VMTK lines computed (`vmtk_sub.sh`); vmrseg6.py got the same request mode, so VMR 0248 has them too.

### 7a. Junction bug and fix (v6_va issue, 2026-10-03 21:19)

v6_va found `ljunction_stageA.vtp` identical to `lsub_stageA.vtp` in ia172_058. Cause (`jfix/jdiag.py` ->
`subclavian/jdiag.json`): ct_seg4 cuts the VA label planar at its first station (voxels behind P[0] -> vid -2) and caps
the subclavian label at r_e + CAP, so the ostium crotch belongs to neither label. In 23 of 54 sides the two labels do
not touch (gap 0.5-4.4 mm) and "largest component" kept the subclavian alone; where they touch it is often by 2-16
voxels, a waist at the ostium (bottleneck ratio before the fix: median of the connected sides about 0.3).

Fix (`jfix/junction6.py`, imported by ctseg6.py and vmrseg6.py when V6_JUNC_BRIDGE=1, the default; backups
`jfix/*.bak_pre_jfix`): a bridge = the CT lumen inside a capsule around [Q, VA stations s 0-6 mm] (Q = the subclavian
line point nearest the VA's first station; radius = the VA's edge radius + 1 mm), with the two vessels' own
thresholds (T_lo = mean of their median local half-max, T_hi = ct_seg4's bright-neighbour level, bone >= 700 HU out);
junction = the component of (subclavian | VA first 25 mm | bridge) holding the subclavian, closed by one voxel inside
the capsule, holes filled; a VA-calibre tube only if the VA part is still not connected (flagged fallback_tube).
Measure: bottleneck = the largest ball radius that can travel inside the junction mask from Q to the VA at s 10
(maximin over the EDT), and its ratio to the VA's inscribed radius at s 10. Re-run of all 27 units
(`jfix/sub_batch2.py` -> `subclavian_j2/<u>/`), published over `subclavian/<u>/{r,l}junction_stageA.vtp` + seg_qc.json
only where the subclavian / VA surfaces and labels are bit-identical to the first run (`jfix/publish_j2.sh`; old files
in `subclavian/<u>/old_v1/`, marker `.junction_v2`).

The first fix (v2, bridge only; `subclavian_j2/`, 19 units published 00:05-00:20) was superseded by v3 when a second
defect showed: the subclavian labels carry 1-voxel bridges / spurs to neighbours, so the junctions had handles (genus
up to 57: VMR 0248 R; 26 VMR L, 21 ia172_071 L, 14 song 108 R). `jfix/sub_genus.py`: a 1-voxel opening then closing
(6-connected cross) of the SUBCLAVIAN label removes them (genus 0 on all 6 sides tested; the opening alone leaves VMR R 3, L 1) while changing
0.7-3.3 % of its voxels. v3 (`junction6.clean_sub`, V6_SUB_CLEAN=1) = that cleanup + the bridge; the VA labels are not
touched. All 27 units re-run (`subclavian_j3/`), subclavian VMTK lines recomputed (`jfix/vmtk_sub3.sh`, 54/54), and
published over `subclavian/<u>/` (`jfix/publish_j3.sh`: VA surfaces and labels_kji.npz bit-identical, checked; previous
files in `old_pre_j3/`, v1 junctions of the j2-published units in `old_v1/`; marker `.junction_v3`; new
`junction_labels_kji.npz` = the final junction masks + cleaned subclavian labels on the labels' grid).

**Result (`subclavian_j3/jsum.json`, 54 sides):**
- before (labels only, after the cleanup): the VA part was disconnected from the subclavian in 31 of 54 sides (23 of
  54 before the cleanup), and where connected the contact was 1-110 voxels; bottleneck ratio < 0.7 in 49 of 54;
- after: all 54 connected; bottleneck ratio p10 0.62 / median 0.85 / max 1.0 (bottleneck radius 0.45-2.02 mm);
  >= 0.7 in 45 of 54. A tube fill was needed on 2 sides (fallback_tube: ia172_136 L, ia172_172 L -- the CT bridge did
  not connect them); bridge 78-1284 voxels (median 214);
- genus: subclavian surfaces 0 on 43 of 54 (1 on 6, max 7 ia172_172 L), junction surfaces 0 on 29, median 0, max 11
  (song 008 L 11, song 094 L 9, ia172_172 L 8; the rest <= 4) -- the remaining handles sit in the VA label part or the
  bridge; the bake's handle rule treats them as it does the donor VAs';
- the 9 sides with ratio < 0.7: ia172_071 L 0.25 (**trace problem**: the VA station line leaves the lumen at VA s 6-8,
  HU 16-55 on it, `subclavian_j3/jct_ia172_071_L.png/.json` -> treat that junction as failed), ia172_071 R 0.53 and
  song 094 L 0.54 (**CT-consistent ostial narrowing**: the junction r_eq, the own half-max CT read and Arterial's
  segmentation all dip at the take-off, `jct_ia172_071_R.png`, `jct_song_AHMU1218094_L.png`), ia172_164 L 0.46 (the
  junction r_eq 1.5-1.9 follows the CT read 2.2-2.6 where Arterial loses the vessel; the ball metric is conservative
  for a flattened neck, `jct_ia172_164_L.png`), ia172_149 L 0.62, 159 L 0.67, 190 L 0.63, song 032 L 0.60, song 095 L
  0.67 (not inspected one by one; `jct.py` draws any side). v6_va's own gates (exit rule, clearance >= 0.6 mm) decide
  per side; its fallback is v5's stub junction;
- the junction's VA part equals the donor record's VA surface: 0.0 mm beyond VA s 5 on all 54 sides, <= 0.36 mm in
  s 0-5 on 12 (`subclavian_j2/jva_vs_record.json`), so the junction and the donor VA union without a seam;
- field of view: every requested stretch is inside the donor CT (each subclavian label covers its stations from 0 to
  within ~1 mm of the end, `<x>sub_seeds.json` label_s_range_mm), VMR 0248 included.

Figures: `subclavian_j3/fig_<u>_<side>.png` (v1 vs v3 junction, two slabs through the ostium: ia172_015 R / 058 L /
071 L / 071 R, VMR 0248 L / R, song 138 R) and `subclavian_j3/jct_<u>_<side>.png` (CT check along the bridge axis).

### 7b. D5 for the subclavian stretches (`wd_batch.py subclavian`, `jfix/wd_sub_collect.py` -> `wd/wd_sub_collect.json`)

Same measure as the donor VAs (the stretch's own v3 surface unioned with its VMTK-MISR tube exactly as the bake does,
share of wall > 0.3 mm off the tube, exact ownership, undecimated), 'core' = without the 8 mm end tapers and +-6 mm
about the VA ostium (the VA stub counts as relief there): **median 0.53** (p10 0.32, p90 0.70, min 0.086 song 138 L,
max 0.79) over 54 sides; whole stretch 0.55. The subclavian's MISR (median 3.6 mm) sits well inside its flattened,
branching section, so the 0.3 mm threshold is relatively finer than on a VA and the number is not comparable to the
VAs' 0.11-0.13 -- but it is unambiguous that the wall is not a tube. Polyline lumen to that union over the core: min
1.70 mm (song 095 L), median 2.71.

### 7c. What the request mode does

`ctseg6.py V6_SUB_REQ=<entries.json>` re-runs a donor with the requested subclavian centreline as the subclavian
stations (the VAs segmented in the same run, so the ostium split is v5's d/R rule) and writes `<r|l>sub_stageA.vtp`
(subclavian label alone, after the 7a cleanup: closed, with a flat lid at the VA ownership boundary), `<r|l>junction_stageA.vtp` (subclavian
+ the first 25 mm of the VA label + the 7a bridge, one closed surface -- recommended for the bake: a union of two
lidded surfaces meeting at a label interface would leave a waist there), seeds and the VMTK line.

## Files

Code, `scratchpad/v6/wall/` (= /sp/v6/wall):
- `ctseg6.py` (v5's scaleup_ct_seg.py + V6_RULE v5 / half / plateau / plateau_g, V6_OUT, V6_DIAG, V6_SUB_REQ),
  `make_vmrseg6.py` -> `vmrseg6.py` (the same patch on VMR 0248's vmr_ct_seg3.py), `donors.json` / `make_donors.py`
  (per-donor configuration that reproduces each v5 wall), `seg_batch.py`, `vmtk_batch.sh`, `regress.py`,
  `diag_view.py`, `prof.py`;
- gates: `gate_ct.py` (140 + 136 points), `va_lumen_poly.py` (c2), `wd_proxy.py` / `wd_batch.py` / `wd_collect.py`
  (wall detail and union lumen per donor side, before / after), `surf_stats.py`, `floor_scan6.py` / `bone_scan6.py`
  (the audit's scans on the new records), `gate_summary.py`;
- `build_donors6.py` (records), `c3_inspect.py`, `c4_check.py`, `sub_batch.py` / `vmtk_sub.sh` ((e) requests),
  `chain2.sh` / `chain3.sh` (the host-side step chains, <= 2 containers), `post_light.sh`, `dr.sh`;
- `jfix/` ((e) junction and D5 follow-up, 2026-10-04): `junction6.py` (junction bridge, clean_sub, genus), `jdiag.py`
  (label contact diagnosis), `sub_batch2.py` (re-runs into subclavian_j2 / _j3), `vmtk_sub3.sh`, `publish_j2.sh` /
  `publish_j3.sh`, `jsum.py`, `jfig.py`, `jct.py` (CT check of a junction), `sub_genus.py`, `jva_vs_record.py`,
  `wd_sub_collect.py` (subclavian D5), `wd_bins.py` / `wd_cut.py` (where the VA relief changes), `narrow4.py`;
  backups of the seg scripts before each patch (`*.bak_pre_jfix`, `*.bak_j2`).

Outputs, `D:\Arjun\neve\carotid_data\v6_work\wall\` (= /v6w; /repo/carotid_data/v6_work/wall in repo-mounted
containers): `seg/<variant>/<unit>/` (walls), `donors/<variant>/` (records), `wd/`, `lumen/`, `c3/`, `c4/`, `scan/`,
`subclavian/<donor_id>/`, `prof/`, `chain/` (step logs), `gate_summary.json`, `surf_stats.json`.
