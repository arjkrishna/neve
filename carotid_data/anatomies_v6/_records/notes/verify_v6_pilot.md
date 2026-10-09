# verify_v6_pilot -- adversarial verification of the v6 stage-1 claims

Agent verify_v6_pilot, 2026-10-04. State: `state/verify_v6_pilot.md`. Own code: `scratchpad/v6/verify/` (runner
`dr.sh`, containers `vfy_*`, /repo and /v6w mounted read-only, <= 2 containers, 6 GB each). Large outputs:
`D:\Arjun\neve\carotid_data\v6_work\verify\` (`/vo`). Measurement code is my own (`vlib.py`, `vgeom.py`,
`vmesh.py`, `vown.py`, `vdonor.py`, `vorient223.py`, `vap223.py`, `vva_lat.py`, `ident223.py`, `vtip.py`,
`vprox_rva.py`, `recount_dry.py`). Borrowed only: the repo's v4 bake functions to rebuild v4's union (validated:
my k24 measure on it reproduces mesh_v4.json right_route_detail on 14/14) and `v5_split.py` (the specified check).

## Answer

**The stage-1 numbers reproduce. The pilot's NO-GO stands, but its route to GO is incomplete.**

| # | check | result |
|---|---|---|
| 1 | gates library vs audit | HOLDS. Own independent measure (Kabsch fit to raw TopBrain, no recorded R used) on all 223: v5 tier A 141 (siphon 47, VA 115), primary 218; v4 171 / 189. Per piece vs the library: rise <= 0.05 mm, tilt <= 2.3 deg, yaw <= 3.6 deg. |
| 2 | pilot invariants | HOLD with the known exceptions. Mesh 14/14 (one closed manifold genus-0 component on OBJ / 60k / union, 0 self-intersections by my own edge-triangle test and pymeshlab, every centreline point inside). Route (my exact ownership): k_011 fork 0.932, all others >= 0.991; k_011 is 0.9996 once the RCCA wall within 8 mm of v4's RECA tip is left out (pilot's explanation confirmed). Clearance on new geometry >= 0.69 mm. 1 mm turns <= 59.2 deg. VA lumen (polyline, OBJ) >= 0.769. VA tops gap >= 1.55 mm. va_target fails on 2 tops (21.6, 21.5 mm). Take-off within 6.4 deg of native (own definition; claim 0.04 median). D5 donor 0.156 -> 0.135, subclavian 0.696. |
| 3 | right route untouched | HOLDS. RCCA identical to v5 up to the siphon seam, (11) / (0) identical: pilot 14/14 and the full plan 223/223. Union wall on RCCA < cut, (11), (0): 0.0 mm deviation on 13/14, 0.068 mm on the arch in one. |
| 4 | split | HOLDS for the plan. v5_split.py on the written full_plan_v121 (not the agent's split tree): exit 0, none of 45 lowered; plan == split tree == pilot (43 substitutes). |
| 5 | wall fix | HOLDS mostly. Cut-in wall (inside the tube by > 0.3 mm) -> 0; floored length 6.6 % -> 0.7 % (median); wall detail 0.133 -> 0.110 (54 sides). But away from v5's floored stretches it also drops 0.118 -> 0.090, so the fix smooths somewhat beyond the artefact; 6 sides keep <= 0.51 of v5. |
| 6 | dry runs | Counts reproduce (carotid 223/223, 43 substitutes, rungs 1-2 only; VA 410/446 sides, 195/223 anatomies). 28/223 (12.6 %) anatomies have a failing VA side; with va_target <= 30 mm still 30 sides / 23 anatomies (10.3 %) fail on course_rms (13), va_gap_v6 (6), va_lateral (7), no placement (5). The dry runs predate the pilot's tool fixes. |
| 7 | read-only | HOLDS. No file under anatomies*, carotid_tools, topbrain_tools newer than the v6 start (19:38); repo changes since then are only reports/ (v4-slides workflow); user.md / user copy.md untouched. |

**New findings (not covered by any v6 gate):**
1. ICA termini sit too far posterior. With v5's fork headings (the D3-fork fallback), the anatomically posed siphons
   start where v5's forks put them. In the planned 223, 57 anatomies have an ICA terminus > 40 mm posterior of its
   anatomical target and 64 have one > 20 mm behind the host confluence (v5: 108 / 123). Pilot w_003 mr_010 RICA:
   75.7 mm (the side figure shows it far behind the VA tops). Only lateral order and spacing are gated.
2. Donor right VAs run medially. In 4/14 pilots the v6 donor RVA lies within 10 mm of x = 14.8 over 52-100 mm of the
   neck: k_004, w_003 mr_010 (both donor ia172_159 R), w_018b and w_047. They reach x ~ 18, the host's own neck
   midline (~19). In v5 the same anatomies sat >= 13.6 mm lateral. course RMS <= 20 mm allows this.
3. SOFA entry is noise at this n. On the v5-junction RVAs that lost entry (w_051, w_034), the union at the entry
   region is identical to v5 (0 mm) and the OBJ differs by <= 0.23 mm. So 7/14 vs 5/14 is not evidence either way.
   Acceptance (e) ">= 20/25" is unlikely to be reached as worded.
4. ica_spacing depends on convention. With the raw TopBrain termini as native, m_006 mr_025_L (20.8 mm) and
   mr_027_L (21.6 mm) fail the 20 mm gate. orient_gates' back-mapped piece ends give 19.4 / 18.8 and pass.

**Corrected GO / NO-GO: NO-GO for stage 2 now.** GO once the pilot's three items are done (a D2 rule, the k_011-type
route baseline and the tool fixes with the dry runs re-run), plus the following:
- (a) a rule for the ~23 anatomies whose VA side fails on things other than va_target (course_rms, va_gap_v6,
  va_lateral, or no placement). For example: keep the v5 junction with an anatomical top, keep v5's top flagged, or
  re-reference course_rms;
- (b) a decision on ICA-terminus AP placement: accept and record it, add an AP gate, or re-pose the forks;
- (c) a VA corridor gate (lateral offset from the host neck midline, or not medial of the host VA by more than X mm),
  then the VA dry run re-run;
- (d) one ica_spacing convention;
- (e) a SOFA entry criterion that is robust (several seeds, or a geometric entry test);
- (f) the split re-run after the guarded joint pass on all 223.

## Method and results (files)

- Orientation, pilot: `/vo/geom/<name>.json`, summary `verify/sum_geom.txt`; ICA AP + take-off `verify/sum_ap.txt`;
  ICA AP on whole trees `/vo/ap/ap_{v6plan,v5,v3}.json`.
- Orientation, all 223: `/vo/orient/{v5,v4}__*.json`, `verify/sum_orient_v5.txt`, `sum_orient_v4.txt`.
- Mesh: `/vo/mesh/{v6,v5}/<name>.json`, `verify/sum_mesh.txt`.
- Ownership / route / proximal / D5: `/vo/own/<name>.json`, `verify/sum_own.txt`; v4 unions `/vo/v4union/`;
  k_011 tip `verify/sum_tip_k011.txt`.
- Donor walls: `/vo/donor/donor_wall.json`, `verify/sum_donor.txt`.
- VA lateral: `/vo/va_lat_{pilot,v4pilot,v5all}.json`, `verify/sum_valat.txt` (v5 all: 32/223 RVAs with >= 50 mm
  within 10 mm of x = 14.8; 8 cross it).
- Split: `/vo/split/split_full_plan_v121.{json,log}`.
- Dry recount: `verify/recount_dry.json`.
- SOFA geometry: `verify/sum_prox_rva.txt`.
- Figures viewed: k_011, w_003 mr_010, m_006 mr_027_L, w_025, w_018b, w_034, w_047, w_035, w_051, k_004.
