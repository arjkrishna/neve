# verify_v5_route13: adversarial check of the route13 fix

Agent prefix `verify_v5_route13`. State `v5_va/state/verify_v5_route13.md`. Own code in `v5_va/vr13/`
(`vlib.py`, `v4iso.py`, `route.py`, `gates.py`, `mesh.py`, `si_incl.py`, `prov.py`, `prio.py`, `priowhere.py`,
`geomdiff.py`, `selftest.py`); outputs `v5_va/vr13/out/`, logs `v5_va/vr13/logs/`. Containers `vr13_*` (all exited),
every repo / carotid_data mount read-only (only /sp writable). No git, nothing written to the repo; no file in the 13
anatomies_v4 / anatomies_v5 folders is newer than 2026-10-02 22:55.

## Answer (complete)

**The fix holds.** Every key number of the route13 result reproduces with independent code.

- **Right route >= 0.99 of v4 on all 13 picks**, with right_route_detail re-implemented and ownership taken EXACTLY
  (argmax over all samples, not the coded 24-nearest cut-off), on the undecimated unions, against v4's union rebuilt
  from the repo baker (13/13 rebuilds reproduce mesh_v4.json's recorded value to 0.0000). Lowest pick: w_007 siphon
  0.9994 (> 0.5 mm: 0.9968). With the coded k=24 attribution my code reproduces the fix's ratios (lowest
  w_012_left mr_021_L fork 0.9926, w_007 siphon 0.9994).
- **7 picks are bit-identical to v4** on the RCCA route cells (k_005, k_008, m_030, w_012_right, w_024, w_029 x2);
  m_022 differs in 12 fork cells by <= 3.5 um (the same 12 cells differ in v5, so not from the move). The fix's "8
  identical" is right within float noise.
- **Every graft gate holds** on all 26 VAs of the 13 picks (own re-implementation from the written curves): exact
  all-pairs clearance min 0.71 mm (m_022 L), joins max 59.3 deg (w_050 R splice), course RMS 14.8-19.9 mm on the
  moved VAs, posed take-off max 118.3 deg on the moved VAs, tip move max 38.0 mm, rise >= 19.4 deg, splice in [C2,
  dura], calibre recomputed = recorded on all 26, det = +1 (donor and TopBrain rotations), follow handedness; the
  written VA between ostium and splice IS the posed donor (<= 0.25 mm).
- **Meshes clean** on all 13 (own code): 1 component, genus 0, 0 open / non-manifold edges, 0 pinch vertices,
  consistent orientation, 0 self-intersections (strict and inclusive tests; positive controls detect), OBJ 23,498-
  23,500 tris, full 60,000, 0 centreline points outside OBJ or full mesh, RCCA / LCCA navigable with the lumen equal
  to the recorded one to 1e-3 mm; undecimated route lumen identical v4 = v5 = pick on all 13.
- **Donor, sides, mirror, plan unit, licences, attributions, non-VA xform and non-VA curves unchanged**; only the
  target VA curve(s) differ from anatomies_v5.

**Corrections** (none changes a pick):

1. *Route priority did NOT change only the contested voxels, and the stated reason for its two drops is wrong.*
   The bake's topology ladder added / widened tube windows under route priority: m_030 lower_R 65.8-66.95 ->
   63.18-67.61 mm (1.15 -> 4.43 mm; the fix's table says "rung C as v5"), w_012_left mr_021_L a new siphon_R window
   224.9-226.54, w_050 a new vaC_R window 1.88 mm. The m_030 fork loss sits entirely at RCCA s 60-75 mm (the widened
   window) and the w_012_left mr_021_L siphon loss at s 205-235, mostly 225-230 (the new window); m_030 also has
   252-318 changed vertices outside the RCCA-VA contested zone. So the drops (0.971 -> 0.851, 0.968 -> 0.933) come
   from RCCA tube windows, not from "the measure attributing regained wall to the VA". Under exact attribution route
   priority also RAISES w_050's fork slightly (0.8185 -> 0.8275); "never raises" holds only for the k=24 measure.
2. *In 4 of the 13 anatomies the deficit was an artefact of the measure's 24-nearest attribution, not lost wall.*
   Under exact ownership v5 already read k_005 fork 0.9915 (k24 0.966), w_024 fork 1.0000 (0.971), w_012_left
   mr_001 fork 1.0024 (0.980), w_029 mr_026_L fork 0.9986 (0.986); w_012_left mr_021_L's fork 0.993 is likewise
   1.0000 exact (its siphon deficit is real). Attribution-free, k_005's v5 RCCA route surface has 0 vertices changed
   vs v4; w_024's only changed vertices within 2 mm of the RCCA reach are new surface a median 1.7 mm from any v4
   vertex that exact ownership gives to the VA (7 RCCA cells differ). w_012_left mr_001 and w_029 mr_026_L do have
   changed surface near the VA (67 and 111 mm2), but it does not lower the exact reading. The moves are harmless
   there, but the "below 0.99 because contested wall fell back to tube" reading does not hold for these four. The real deficits (exact) were w_050 fork
   0.8185, m_030 fork 0.9708, w_007 0.9824 / 0.9874, m_022 siphon 0.9744, w_012_left mr_021_L siphon 0.9775,
   w_029 mr_024_L siphon 0.9845, w_012_right siphon 0.9878, k_008 siphon 0.9889, w_035 siphon 0.9893.
   Consequence: w_012_left mr_021_L's pin variant was chosen over ext because ext read fork 0.9887 at k=24; under
   exact ownership ext reads 1.0000 too.
3. *The summary range "moved VAs keep 3.9-12.7 mm wall-to-wall" omits w_007*: its moved RVA comes within 1.66 mm
   (own: min over the VA past the exit to the RCCA tube wall; the per-anatomy list says 1.66).
4. Minor: v5_va.summary differs from anatomies_v5 only in the per-side take-off record (e.g. w_050 R
   takeoff_not_preserved false -> true); no calibre flag is new (every pick VA with ratio > 1.25 had > 1.25 in v5).

## 1. Right route detail (own code, `vr13/out/route_<name>.json`, cells in `vr13/cells/<name>.npz`)

v4 reference: `vr13/v4iso/<name>.vtp` (13/13: B.right_route_detail == mesh_v4.json, max abs diff 0.0). Own k=24 on
the rebuilt v4 union vs the recorded value: within 0.0005 everywhere.

Ratios vs v4 at > 0.3 mm (fork / siphon). "exact" = own exact ownership; "k24" = the coded attribution. Ident = RCCA
route cells vs v4 (tolerant match: centroid 1e-3 mm, depth 1e-3 mm).

| anatomy | pick | pick exact | pick exact > 0.5 | pick k24 | v5 exact | v5 k24 | ident |
|---|---|---|---|---|---|---|---|
| w_050_left mr_004_L | ext | 1.0000 / 1.0023 | 1.0000 / 1.0000 | 1.0004 / 1.0022 | 0.8185 / 1.0023 | 0.8188 / 1.0022 | no (siphon = v5's) |
| m_022_left mr_021_L | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0002 / 0.9999 | 1.0000 / 0.9744 | 1.0002 / 0.9567 | 12 fork cells <= 3.5 um |
| k_005_right mr_024_L | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9998 / 0.9998 | 0.9915 / 1.0000 | 0.9661 / 0.9998 | yes |
| w_012_left mr_021_L | pin | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9926 / 1.0001 | 1.0000 / 0.9775 | 0.9926 / 0.9677 | 124 fork cells <= 0.12 mm (as v5) |
| m_030_right mr_013_L | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 0.9999 | 0.9708 / 1.0000 | 0.9707 / 0.9999 | yes |
| w_024_right mr_026_L | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0002 / 0.9999 | 1.0000 / 1.0000 | 0.9711 / 0.9999 | yes |
| w_012_left mr_001 | ext | 1.0024 / 1.0000 | 1.0049 / 1.0000 | 1.0175 / 0.9999 | 1.0024 / 1.0000 | 0.9795 / 0.9999 | no (fork reads higher) |
| w_007_right mr_023 | ext | 1.0000 / 0.9994 | 1.0000 / 0.9968 | 1.0000 / 0.9994 | 0.9824 / 0.9874 | 0.9824 / 0.9874 | no (83 siphon cells) |
| w_029_right mr_024_L | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 0.9845 | 1.0000 / 0.9844 | yes |
| w_029_right mr_026_L | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9986 / 1.0000 | 0.9859 / 1.0000 | yes |
| w_012_right mr_024_L | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9999 / 1.0001 | 1.0000 / 0.9878 | 0.9999 / 0.9878 | yes |
| k_008_right mr_027 | ext | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0001 | 1.0000 / 0.9889 | 1.0000 / 0.9890 | yes |
| w_035_left mr_016 | ext | 1.0068 / 1.0000 | 1.0162 / 1.0000 | 1.0068 / 0.9999 | 1.0056 / 0.9893 | 1.0056 / 0.9892 | no (fork reads higher) |

- Exact vs k24 ownership disagree on ~49k cells per union (mostly which RCCA sample), and flip RCCA ownership on
  260-770 cells; that is the k24 artefact in correction 2.
- Attribution-free (`vr13/out/geomdiff.json`): union vertices within 2 mm of the RCCA route's tube reach that are
  not bit-identical to v4: picks 0 on both segments for k_005, k_008, m_030, w_012_right, w_024, w_029 x2; m_022 2
  (fork, 0.01 mm, as v5); w_012_left mr_001 6 + 5; w_035 1; w_050 siphon 80 (<= 0.14 mm); w_012_left mr_021_L fork
  43 (<= 0.22 mm, as v5); w_007 siphon 135 at s 195-198 (where its moved RVA still sits 1.66 mm from the RCCA tube).
- Route min lumen on the undecimated unions (exact point-to-surface): v4 = v5 = pick on all 13 (RCCA 0.959-1.886,
  LCCA 0.972-1.604 mm).
- Route-priority unions, exact (k24) fork / siphon vs v4: w_050 0.8275 (0.8188) / 1.0023; m_022 1.0000 / 0.9720
  (0.9553); k_005 0.9858 (0.9661); w_012_left mr_021_L 1.0000 / 0.9424 (0.9325); m_030 0.8512 (0.8512); w_024
  1.0000 (0.9711); w_012_left mr_001 1.0000 (0.9759). The fix's k24 prio numbers reproduce.

## 2. Gates (`vr13/gates.py`, `vr13/out/gates.json`, log `vr13/logs/prov_gates.log`)

All 26 VAs pass every gate (also anatomies_v5's, as a baseline). Moved VAs (target sides):

| anatomy | VA | clearance mm (who) | joins O_h / ostium / splice deg | RMS mm | take-off posed (written) deg | tip mm | calibre | RCCA w-w v5 -> pick mm |
|---|---|---|---|---|---|---|---|---|
| w_050 | R | 1.41 (self) | 53.1 / 54.7 / 59.3 | 19.2 | 118.3 (115.5) | 16.6 | 1.230 | 1.06 -> 5.96 |
| m_022 | L | 0.71 (parent after exit) | 0 / 24.6 / 42.0 | 19.6 | 81.5 (81.2) | 37.8 | 1.397 | 0.92 -> 12.10 |
| k_005 | R | 0.88 (parent) | 56.6 / 24.1 / 48.1 | 14.8 | 85.6 (85.1) | 18.0 | 1.202 | 2.75 -> 8.16 |
| w_012_left mr_021_L | L | 0.72 (parent) | 0 / 24.4 / 20.5 | 19.7 | 84.8 (84.6) | 37.9 | 1.461 | 2.23 -> 12.35 |
| w_012_left mr_021_L | R (pinned donor, TB -15 x) | 1.37 | 34.7 / 19.0 / 18.1 | 16.3 | 119.99 (119.3) = v5 | 30.9 | 1.223 | 4.36 = v5 |
| m_030 | R | 1.37 | 47.2 / 13.4 / 49.4 | 19.4 | 114.2 (115.3) | 34.7 | 1.347 | 1.02 -> 10.56 |
| w_024 | R | 0.73 (parent; exit 10.9) | 50.3 / 27.3 / 50.7 | 19.1 | 59.4 (57.3) | 38.0 | 1.524 | 1.65 -> 7.99 |
| w_012_left mr_001 | R | 1.05 | 47.0 / 25.0 / 19.3 | 19.9 | 69.8 (69.6) | 24.6 | 1.197 | 1.00 -> 4.96 |
| w_007 | R | 1.11 | 58.8 / 33.1 / 23.1 | 17.1 | 89.5 (89.8) | 26.9 | 1.061 | 0.89 -> 1.66 |
| w_029 mr_024_L | R | 0.88 | 56.6 / 24.1 / 48.1 | 14.8 | 85.6 (85.1) | 18.0 | 1.202 | 0.68 -> 12.71 |
| w_029 mr_026_L | R | 0.73 (exit 10.9) | 57.9 / 27.3 / 55.1 | 16.1 | 66.3 (64.1) | 18.5 | 1.535 | 1.16 -> 3.89 |
| w_012_right | R | 0.83 | 50.2 / 24.2 / 35.2 | 19.3 | 84.2 (83.8) | 12.9 | 1.138 | 0.69 -> 8.17 |
| k_008 | R | 1.47 | 44.8 / 15.0 / 37.8 | 18.9 | 77.7 (77.8) | 30.4 | 1.132 | 1.07 -> 9.79 |
| w_035 | R | 1.23 | 50.4 / 23.4 / 17.9 | 18.0 | 74.3 (75.1) | 30.0 | 1.544 | 1.42 -> 11.44 |

- Clearance is exact all pairs from the VA origin (stricter than the tool's 8-nearest); exit <= 10.9 mm, no
  re-entry. Rise >= 19.4 deg everywhere. Splice within [C2, dura] on all 26. Calibre recomputed equals the recorded
  value on all 26; the pick VAs above 1.25 (pass 2) were above 1.25 in v5 too.
- Non-target VAs: curves identical to anatomies_v5 (and they re-pass every gate against the moved VA).
- The written w_050 RVA take-off is 115.5 deg by my ostium location, 118.4 recorded; the gated (posed) value is
  118.28 in both.

## 3. Meshes (`vr13/mesh.py`, `vr13/si_incl.py`, `vr13/out/mesh_<name>.json`, `si_incl.json`; controls `logs/selftest.log`)

All 13 picks: OBJ / full / undecimated union each 1 component, Euler genus 0, 0 open, 0 non-manifold edges, 0 pinch
vertices, 0 orientation-inconsistent edges, no duplicate vertices or degenerate faces; own self-intersection test 0
on OBJ and full (strict and inclusive); OBJ 23,498-23,500, full 60,000; every centreline control point (3 mm caps
excluded) inside OBJ and full (winding number 1.000); RCCA / LCCA navigable, lumen = recorded (e.g. w_024 RCCA
1.392); OBJ / full vertices within 0.26 / 0.11 mm of the pick's union (they are decimations of it). Signed volume is
negative, the same convention as the shipped v4 / v5 OBJs. Controls: two interpenetrating spheres and an off-axis
pushed vertex are detected; bow-tie -> 1 pinch; torus -> genus 1; exact_best == brute force.

## 4. Provenance (`vr13/prov.py`, `vr13/out/prov.json`, `logs/prov.log`)

For all 13 picks (and the prio / non-pick folders): donor id / side / mirror equal anatomies_v5 and plan_v5.json, and
the donor is the patient's single assigned unit; xform donor record / surface / md5 / kind unchanged; TopBrain piece
source and mirror unchanged; v5_va.licences and .assignment identical; LICENSE_v5_donors.txt byte-identical; file
lists identical; every non-VA xform entry equals v5 and v4; every Centrelines_comb file except the target VA
curve(s) is byte-identical to anatomies_v5 AND anatomies_v4; no TaG-Net donor. Rungs / flags / tubed lengths as the
fix reports (m_022 L 1.15 -> 1.77 mm, m_030 rung C -> 1 without flag).

## 5. Route priority (`vr13/prio.py`, `vr13/priowhere.py`, `vr13/out/prio.json`, `priowhere.json`)

prio unions vs v5 unions (same centrelines, verified): changed vertices 5 (w_024) to 879 (w_012_left mr_021_L), mostly
in the RCCA-VA contested zone, but the ladder changed tube windows (correction 1):

| anatomy | changed vertices (contested / other) | windows v5 -> prio |
|---|---|---|
| w_050 | 496 (484 / 12) | + vaC_R 87.99-89.87 (rung 1 -> 2) |
| m_022 | 542 (542 / 0) | same |
| k_005 | 107 (104 / 3) | same |
| w_012_left mr_021_L | 879 (877 / 2) | + siphon_R 224.9-226.54 (rung 2 -> C) |
| m_030 | 781 (529 / 252) | lower_R 65.8-66.95 -> 63.18-67.61 |
| w_024 | 5 (5 / 0) | same |
| w_012_left mr_001 | 124 (124 / 0) | same |

m_030 fork, cells with depth > 0.3 mm lost / gained (mm2) by RCCA s bin: 60-65 15.4 / 2.7, 65-70 63.6 / 18.8, 70-75
6.4 / 4.7, nothing elsewhere. w_012_left mr_021_L siphon: 225-230 62.3 / 33.4, small elsewhere in 205-235.
