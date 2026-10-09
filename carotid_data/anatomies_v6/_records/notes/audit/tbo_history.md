# tbo_history: the documented TopBrain orientation rule, and every code path that places a TopBrain piece

Agent prefix `tbo_history` (state: `state/tbo_history.md`). Read-only on repo and anatomies. Paths below: repo-relative
for repo files, `/sp/v5_va/...` for scratch. Measurements: provenance `xform` only (no meshes), scripts and outputs in
`/sp/v5_va/tbo/` (`hist_xform.py` -> `hist_xform_v4.json` / `_v5.json`; `hist_summary.txt`; `hist_lower.txt`;
`hist_anat.txt`; `siphonL_flags.txt`; `tb_frame.txt`; `host_frame.json`). The per-piece geometry audit itself is
`tbo_geom`'s job; the numbers here exist to tie code paths to the four images.

## Answer so far

1. **The rule** (set A Aug 27, fixes C1/C2 in `TOPBRAIN_PIPELINE.md` 212-250; set B a898eb2 Sep 1): place a TopBrain piece by
   a FRAME match, tangent onto tangent and superior onto superior (`graft_siphon.frame_rotation`, UP = +z), after
   `anchor_trim` has removed a doubling-back start (70 deg vs chord, <= 30 mm); for the donor section of set B, a gated
   inlet trim (`graft_three.head_trim`, 27 deg, 25/12 mm), and `extend_ica` with +z as up. The check that caught the defect
   was a MEASUREMENT, rise in z against the donor's native rise (plus donor-section tilt from +z). It was never made a
   code gate. Gates that came out of the audit: rise < 20 mm 11 -> 0, negative 4 -> 0, worst rise error 112 -> 44.8 mm,
   tilt max 52.7 -> 33.2 (none > 40). It was verified on set A v1 (25, then 49) and set B v1 (215). v2/v3 (223) reran the
   same graft code (V2_BUILD_PLAN.md:77-78 required the rise counts again). v4 and v5 did NOT re-run any orientation measure.
2. **Every v4/v5 placement path applies the frame rule or replaces it, and none checks orientation.** The left siphon
   (`G3.compose`), the ECA tips (`extend_eca`) and the v4 VA tops (`graft_va`) all call `graft_three.place` ->
   `frame_rotation`, with `anchor_trim` in `graft_v4.tb_part`. The v5 VA tops replace it with a Kabsch fit onto the donor's
   distal VA (`graft_v5_va.tb_rotation`, TB_PLACE "fit"), which has no superior constraint. route13 adds +-15 deg
   turns on top of that.
3. **Why the rule still fails.** Frame matching gives tilt(R) = |elev(donor start tangent) - elev(ANCHOR tangent)|
   exactly (proved below; checked numerically to <= 10 deg on every placement). So it keeps superior only when the
   anchor tangent is upright. It also sets yaw (rotation about +z) to the azimuth difference of the two tangents, which
   is anatomically arbitrary. The documented rule trims the donor's start but explicitly NOT the anchor
   (`graft_three.py:307`, "there the local tangent IS the direction of travel"). That held on the right route
   (ICA >= 57.8 mm, tip elevation >= 24 deg) and fails on paths v4 added.
   - **Left siphons (the k_011 / w_003 images):** v4's `lower_records` dropped the 58 mm ICA and 2 mm tip rules. 46
     anatomies get a left ICA < 40 mm, 28 of them with a native tip < 30 deg. k_011_right's 19.3 mm ICA tip points
     -72.6 deg natively and -57.4 deg as placed, so all five k_011_left siphon_L descend or stay flat (rise -38.7 / -38.1 /
     -30.2 / -2.1 / -1.1 mm vs native 51-69). In all, 50/223 siphon_L have rise < 20 mm or tilt > 40 deg. w_003_right's tip
     sits at 25-31 deg, giving tilt 26-32 deg, rise -12 / -19 mm, a 61-93 deg horizontal turn and the genu thrown
     22-35 mm forward.
   - **v4 VA tops (mr_025_L RVA, kept from v4):** the anchor is the host VA's tangent near its confluence (median
     elevation 17 / 27 deg, min -19 / -40). mr_025_L: anchor -18.9 deg, tilt 66 deg, rise -6.8 vs native +32.7 mm, medial
     excursion 33 mm. 65 R + 65 L tops run laterally where natively medial.
   - **v5 fit VA tops (mr_027_L):** tilt > 40 in 99/211 R and 113/218 L. mr_027_L: tilt 71 / 49 deg, both tops pushed
     medially and crossing.
   - **Yaw is unpinned in every generation:** 69/140 of v3's right siphons are turned > 90 deg horizontally. The rise
     metric cannot see this.
4. **Mirroring of _L** is x -> -x in the source RAS frame, before prep and trims. It leaves superior (+z) and anterior (+y)
   unchanged, and every placement is a proper rotation, so mirroring is not the cause. Mirrored and native anatomies
   are both affected (w_003_left mr_010 is native).

## 1. The documented rule(s) and checks

| what | where (code) | where (doc / commit) | thresholds |
|---|---|---|---|
| frame match: rows [t, unit(UP - t(UP.t)), t x n], R = frame(t_host)^T frame(t_siph); UP = +z in both frames (TopBrain RAS; host JSON runs superior along -x, `json_to_branch` -> branch +z) | `topbrain_tools/graft_siphon.py:73` (UP), `:157-165` (frame), `:167-169` (frame_rotation), `:257-260` (used in graft) | TOPBRAIN_PIPELINE.md 212-240 "Fix C1"; graft_siphon docstring 23-36; introduced a2a1f26 (2026-08-27) | degeneracy guard 1e-6, deliberately not tightened (CAROTID_THREE_SOURCE.md 198-200) |
| `anchor_trim`: advance the siphon start until its 5-pt tangent is within 70 deg of the chord to the end | `graft_siphon.py:172-184` (ANCHOR_THR 70, ANCHOR_MAX_TRIM 30 mm, > 30 pts left), called `:238` | TOPBRAIN_PIPELINE.md 242-250 "Fix C2" (mr_021, 125 deg, 4 mm) | fires on mr_021 idx 4, mr_024_L idx 2 (GUIDE_stitched_datasets.md:155) |
| set B: `graft_three` never called anchor_trim -> added | `carotid_tools/graft_three.py:650-662` | a898eb2 ORIENTATION; CAROTID_THREE_SOURCE.md 113-131; VERIFICATION_HISTORY.md 50-80 | mr_021 1.1 -> 65.4 mm; mr_024_L 19.5 -> 42.3; rise < 20 mm 11 -> 0; negative 4 -> 0; worst error vs native 112 -> 44.8 mm |
| gated inlet trim of the donor CCA (not span widening) | `graft_three.py:412-436` (head_trim), constants `:314` START_TC_THR 27, `:324-325` INLET 25 / 12 mm; used `:467` | a898eb2; CAROTID_THREE_SOURCE.md 220-224; VERIFICATION_HISTORY.md 165-180 | tilt max 52.7 -> 33.2 deg, none > 40; seam-1 kink max 29.1 |
| `extend_ica`: +z as the frame's up (was the inlet-to-tip chord) | `carotid_tools/extend_ica.py:16-23, 43` | CAROTID_THREE_SOURCE.md 154-162 | |
| `place()`: dead `up` argument removed; siblings ride the same R | `graft_three.py:390-409` | CAROTID_THREE_SOURCE.md 164-176 | |
| minimal rotation rejected | | CAROTID_THREE_SOURCE.md 185-196 | rise error median 2.2 (frame) vs 16.5 mm |
| span widening NOT applied to the siphon, and **NOT applied to ANCHORS (host cut point, ICA tip)**, "there the local tangent IS the direction of travel" | `graft_three.py:301-312` | a898eb2 | the anchor tangent is never trimmed or checked |
| siphon mirror for left ICAs (x -> -x in RAS, before anything) | `graft_siphon.py:485-494`; `graft_three.py:648-649` | a0cae4c (both ICAs, 49), ceadf86 (mirror rule applies to siphons, not forks); VERIFICATION_HISTORY.md 345-386 | torsion R +1.95 / L -2.00 / L mirrored +2.00 |

**The check.** "The metric that catches it is rise in z against the donor's own native rise" (CAROTID_THREE_SOURCE.md
107-110), plus donor-section tilt from +z (v1: median 12.1, p90 22.0, max 33.2, none above 40). These were audit
measurements by the Workflow 1-4 agents (VERIFICATION_HISTORY layers 1-6). **They are not code gates.** No grafter rejects
on rise or tilt. `topbrain_tools/validate_anatomies.py:27-33, 88-89` prints whole-route RCCA rise and "descending routes".
It covers the route only, not per piece, and not LCCA, VA or ECA. `check_anatomies.py` has no orientation test.
VERIFY_f292143_report.md:439, 461 marks the frame-match and unconstrained-DOF fixes "not covered", not re-verified.

**Which builds ran the rule, and where it was verified:**
- set A v1: 25, then 49 anatomies on the host RCCA (graft_siphon). Rise and descending count were measured in
  TOPBRAIN_PIPELINE.md 231-236 and with validate_anatomies.
- set B v1, three-source (graft_three, 215): measured in the a898eb2 audit.
- set B v2/v3 (223): the same graft_three code and flags. V2_BUILD_PLAN.md:77-78 requires "mean rise and descending
  count per set (validate_anatomies.py)" and "rising < 20 mm = 0". The guide reports route rise of 156.4-187.7 mm,
  whole route only. v3's centerlines are identical to v2's.
- v4 / v5: no orientation measure at all. CAROTID_V4.md 350-379 checks names, byte identity, clearance, LECA fusion,
  the ECA join turn, "1 443 transforms, all proper rotations" (det only), records, topology and SOFA. CAROTID_V5.md
  185-196 adds take-off at the donor origin, course RMS, and tip move <= 60 mm.

## 2. Every code path that places a TopBrain piece (v4, v5)

Shared prep for every v4/v5 TopBrain part is `graft_v4.tb_part` (`carotid_tools/graft_v4.py:184-208`): source side
mapped by handedness, mirror x -> -x `:198-199`, `prep_siphon`, `trim_short_seeds` for va/eca `:201-203`, then
`anchor_trim` `:204`. `place()` at `:262-264` is `G3.place`, a frame match that translates the piece onto the anchor.

| piece (xform key) | builder, file:line | rotation | anchor tangent | anchor_trim / frame (tangent AND superior) | gates on the piece | rise / orientation check |
|---|---|---|---|---|---|---|
| right ICA siphon (`siphon`) | v3 `graft_three.compose` seam 2, `graft_three.py:530`; v4 copies RCCA byte for byte `graft_v4.py:835`; v5 unchanged | frame | the right lower's ICA tip (ICA >= 58 mm via manifest/extend_ica, tip >= 2 mm, `match_sections.py:63-97`) | yes / yes (+ head_trim on the lower) | max_kink 60, clearance (band) | audited in v1 only |
| **left ICA siphon** (`siphon_L`) | `graft_v4.py:544-556` (tb_part "ica" left; mr_006 fallback = the right ICA in the opposite handedness) -> `G3.compose(lcca...)` `:577` under `g3_constants` (seam_L 157.6, HOST_CUT 35-110, BLEND 30) -> `place` `graft_three.py:530` | frame | **the LEFT lower's ICA tip**; lowers from `lower_records` `graft_v4.py:268-318`, which **drops the 58 mm ICA rule and the 2 mm tip rule** (`:294-303`; need only CCA+ICA >= 47.6 mm) and prefers the NATIVE, unextended ICA | yes / yes (head_trim on the inlet; the anchor itself untouched, `graft_three.py:307`) | kink <= 60, ECA >= 17, re-entry, LECA fused <= 16, clearance 0.6, seam landing | **none** |
| TopBrain ECA tips (`ecaext_R/L`) | `graft_v4.py:750-800` tb_part "eca" -> `extend_eca` `:388-407` -> `place` `:402` | frame | the fork ECA's tip tangent (right: v3's RECA; left: the new LECA) | yes / yes | calibre <= 1 mm, join kink <= 60 (`join_kink` +-3 mm `:228-237`), clearance, re-entry | none |
| v4 TopBrain VA tops (`va_R/L`) | `graft_v4.py:528-540` (cut plan) and `:652-748` (VA_CUT_STEPS +-30 mm, donor start trim 0-5 mm, then deflect) -> `graft_va` `:366-385` -> `place` `:380` | frame | the host VA's tangent at cut = host length - donor length (clamped 80 .. L-10), moved for clearance only | yes / yes | join kink <= 60, clearance 0.6 | none |
| v5 TopBrain VA tops, re-fit (`va_R/L`, placement "fit": 211 R, 218 L) | `/sp/v5_va/graft_v5_reg.py:464` (same tb_part, same mirror) -> `P.compose` (`graft_v5_va.py:515-535`) -> `graft_splice` `:401-437` -> `tb_rotation` `:445-473` with `TB_PLACE = "fit"` `:105` | **Kabsch** onto the donor's own distal VA past the splice (21 matched normalised-arclength points, start pinned, `:460-465`); `frame_rotation` is computed `:457` and discarded | the donor VA (a different patient, Kabsch-posed onto v4's corridor, `base_pose` `:248-256`) | anchor_trim yes (tb_part) / **no frame, no superior constraint** | GATES `graft_v5_reg.py:91`: join kink <= 60, tip move <= 60 mm vs v4's blind end `:359-363`, clearance. The "rise" gate `:229` is the DONOR's 10 mm chord at the ORIGIN (>= 15 deg), not the TopBrain piece | none |
| route13 turn (13 anatomies) | `/sp/v5_va/graft_v5_reg_sep.py:193-211` (S4 --tb-shift): R = rot(axis, a) @ R_fit, a = +-3..15 deg about branch x/y/z; picked by the RCCA-wall objective (S6) | fit + turn | as above | as above | as above | none |
| v4 VA kept in v5 (17 sides: 12 R, 5 L) | `/sp/v5_va/bake_full/keep_v4_va.py` (= carotid_tools/v5/bake/bake_full/) | frame (v4) | host VA | yes / yes | v4's | none |
| donor CT VA (not TopBrain, context) | `graft_v5_reg.py:118-175` load_side (mirror `:146, 175`, "follow" `:180-185`) | weighted Kabsch onto v4's corridor + pose search | - | - | origin turn, course RMS <= 20, take-off 20-120, rise >= 15 deg at the origin | take-off only |

Rung-C anatomies (w_016_left/right mr_011_L) changed only their mesh windows. The centerlines and xforms of the carotid
side are v4's.

## 3. Which paths are expected to give wrong orientation, and why

**The identity.** `frame()` rows are t, n = unit(z - t(z.t)) (vertical plane of t, pointing up) and the horizontal
b = t x n. R maps the donor frame onto the anchor frame, so R z = sin(e_s) t_h + cos(e_s) n_h, where e is a tangent's
elevation above horizontal:
- **tilt** = angle(R z, z) = |e_s - e_h|;
- **yaw** about z = azimuth(t_h) - azimuth(t_s), because b maps horizontal to horizontal.

Checked on every frame placement: |tilt - |e_s - e_h|| <= 3.7-10.4 deg, the residue being the tangent span
(`hist_summary.txt`, "frame_identity_max_dev"). Frame matching therefore keeps the piece's superior axis only if the
anchor tangent has the same elevation as the donor's start. It never makes the piece's anterior or medial direction
anatomical. The rise metric sees only the first effect.

Frames, measured: TopBrain is RAS (rICA x > lICA x in 25/25; ICA anterior of VA in 25/25, `tb_frame.txt`). The host
branch frame is +x left, +y posterior, +z superior (RCCA x -10 vs LCCA +43.8; VAs y 33-36 vs CCAs 17-22,
`host_frame.json`). An anatomically correct placement is about Rz(180 deg).

| path | expected wrong? | evidence (v5 = v4 except the VA tops) |
|---|---|---|
| **siphon_L** (v4 left carotid) | **YES**, wherever the left lower's ICA tip is not upright. v4 admitted short, unextended ICAs, and the anchor is untrimmed by design | 46 anatomies with left ICA < 40 mm, 28 of them with native tip elevation < 30 deg. Right route: ICA >= 57.8 mm, tip >= 24.2 deg. 29 anatomies (9 lowers) have the left ICA tip below horizontal as placed (17 natively). 50/223 siphon_L have rise < 20 mm or tilt > 40 deg; 10 have negative rise; worst -38.7 vs native +61.4 mm (`siphonL_flags.txt`). By lower: k_011_right 5, w_024_left 5, w_013_left 5, w_008_left 5, w_047_right 4, k_004_right 4, ... |
| v4 VA tops (frame, host anchor) | **YES**: the host VA's tangent near the confluence is often horizontal or descending, and the cut search moves it for clearance only | anchor elevation median 17 (R) / 27 (L), min -19 / -40 deg. Tilt > 40: 49 R / 30 L. Placed rise < 20 mm: 105 / 101 (native 18 / 25). Running laterally where natively medial: 65 R / 65 L (native medial 206 / 190) |
| v5 fit VA tops | **YES**, more tilt: no superior term, and the TopBrain piece takes the donor's distal shape and pose | tilt > 40: 99/211 R, 113/218 L. Lateral 31 / 36. Horizontal chord turned > 90: 29/203, 59/207. Start-vs-donor tangent median 36-38 deg, max 97 |
| ECA tips | yes, short pieces (10-36 mm) | anchor elevation min -79 / -81. Tilt > 40: 16/52 R, 22/53 L. Negative rise where native is positive: 5 / 11 |
| route13 turn | same as fit, +-15 deg more | |
| right siphon (v3 route, every generation) | **pitch: no** (rise error >= -27.7 mm, none < 20 mm, tilt > 40 in 17). **Yaw: yes**, a property of the documented rule | 69/140 pieces with a >= 10 mm horizontal chord turned > 90 deg from their native anatomical direction; yaw error median ~90 deg, i.e. uniform (`hist_anat.txt`). Never measured, because rise is blind to yaw |

**The user's four images.**
- **k_011_left mr_012_L / 013_L / 016_L / 023_L** (plus mr_026_L, not shown): siphon_L sits on the left lower k_011_right
  (ICA 19.3 mm, unextended; host cut clamped at 110). Its ICA tip tangent is -72.6 deg natively and -57.4 deg as placed.
  Tilt 99-136 deg, rise -38.7 / -38.1 / -30.2 / -2.1 (-1.1) mm vs native 51-69 mm. The right siphons are fine (tilt 0-27).
- **w_003_left mr_007_L / mr_010:** the left lower w_003_right (60.4 mm ICA, but the tip leans) has tip elevation
  25 deg native / 31 deg placed. Tilt 32 / 26 deg, rise 56.0 / 51.8 vs 68.0 / 70.8 mm. Horizontal turn 93 / 61 deg; the
  genu is thrown 34.6 / 22.0 mm anterior (native 4.3 / 0.2). Its third anatomy, mr_016_L, has tilt 46 deg.
- **m_006_left mr_025_L:** the RVA is v4's frame placement, kept. Host RVA anchor at the cut (186.9 mm) -18.9 deg vs
  donor start +52.4. Tilt 66 deg, rise -6.8 vs +32.7 mm, horizontal turn -87 deg, medial excursion 33 mm (native 19),
  across the midline.
- **m_006_left mr_027_L:** both VAs are v5 fits. Tilt 71 / 49 deg; medial excursion 29 / 16 mm (native 14 / 10). Tops end
  at x 28.4 (RVA) and 7.2 (LVA) about a midline near x 17, so they cross.

## 4. Mirrored "_L" anatomies

- **Handedness** is decided once, from v3's right siphon id: an `_L` anatomy is "mirrored" (`graft_v4.py:518-520`). Every
  TopBrain part then comes from the patient's opposite side, x-mirrored (`tb_part` `:189`, `:198-199`). The left siphon
  of a mirrored anatomy is the patient's RIGHT ICA mirrored. The v5 donor CT VAs follow the same rule ("follow",
  `graft_v5_reg.py:180-185`, mirror applied to points, O_v and the subclavian tangents, `:146-175`, after prep).
- **Mirror axis:** x in the source frame (TopBrain RAS = the sagittal plane). It is applied before prep_siphon and
  anchor_trim, and recorded as `xform.<key>.mirror = [-1, 1, 1]` so the baker reflects the surface first. Zenodo
  forks are never mirrored (VERIFICATION_HISTORY 345-386).
- **Superior is not flipped:** z (and anterior y) are unchanged by the reflection. frame_rotation uses UP = +z after
  the mirror. Kabsch and frame are proper rotations (det +1 in all recorded R). So mirroring changes only chirality
  and is not the cause.
- The affected pieces include native anatomies (w_003_left mr_010; the yaw spread is the same for native and
  mirrored). The k_011_left cases are all `_L` only because every k_011_left v3 anatomy uses an `_L` siphon. The cause
  there is the unmirrored left lower's ICA tip.

## Pointers for v6 (not decided here)

- Gate every TopBrain piece on every path: rise vs native, tilt = angle(R z, z), and horizontal direction against
  the anatomical rotation (TopBrain RAS -> host LPS-like, about Rz(180 deg)).
- Never let an ANCHOR tangent set the pose unchecked. Either restore a minimum ICA length / tip test for left lowers,
  or trim or validate anchor tangents, or place the piece by anatomical frame (superior AND anterior) and let the
  join absorb the bend under the 60 deg join gate.
- The v5 fit needs a superior term, or a slerp back to an anatomical frame.

Status: complete
