# tbo_cause: root cause of the TopBrain orientation defects, per class

Agent prefix `tbo_cause` (state: `state/tbo_cause.md`). Read-only on the repo and the anatomies. Code and outputs are
in `/sp/v5_va/tbo/repro/` (= `C:\...\scratchpad\v5_va\tbo\repro\`). Nothing was baked; no repo file was written.
This builds on `notes/tbo_history.md` (code paths, documented rule) and `notes/tbo_geom.md` (per-piece metrics,
flags). Repo line numbers are for HEAD 113c89a. The v5 files are cited by their repo copies in
`carotid_tools/v5/graft/`, which have one extra header line, so they read +1 against the scratch originals.

## Answer so far (final)

**One mechanism behind every class.** Every TopBrain piece except the v5 VA tops is placed by the documented fix:
`graft_three.place` -> `graft_siphon.frame_rotation` (tangent onto tangent, superior onto superior, after
`anchor_trim`). Re-running that code on the build's own inputs reproduces the shipped poses. On the 218 flagged
anatomies (batch) the rotation agrees to a median of 1.1-3.8 deg per piece type (p90 <= 11.5). On the hand-picked
cases the siphon_L poses reproduce within 0.15-0.86 deg and 0.6-0.95 mm.

A frame match fixes the roll only. The other two rotational degrees of freedom are copied from the ANCHOR:
- the piece's pitch error equals |elev(anchor tangent) - elev(piece start tangent)|;
- its heading (yaw) equals az(anchor) - az(piece start).

The code takes the anchor tangent as "the direction of travel" and never checks it. The comment at
`graft_three.py:307-310` says so: anchors are deliberately not trimmed.

The only gate at a seam is the join kink (<= 60 deg). The frame match makes the two 5 mm tangents equal by
construction, so that gate cannot see a wrong pose: the shipped joins are 5-46 deg even on the descending siphons.

The documented check, rise against the native rise, was a one-off audit with no script in the repo. It was never a
gate, and v4 and v5 never ran it. The v5 VA tops replaced the frame match with a Kabsch fit that has no superior
term at all. So the user's reading is right: the v1-v3 fix is still in the code, and the CHECK did not carry over.

### Classes (v5 counts; tbo/repro/classes.txt; "primary" = tbo_geom tier A or up_tilt > 40)

| class | pieces / anatomies v5 (tier A) | v4 | v3 | since | code path and the missing step |
|---|---|---|---|---|---|
| **L_PITCH**: left siphon pitched by the left fork's ICA tip (user images 1-2) | 50 / 50 (35 / 35); 30 with the seam tangent below 10 deg | same (LICA identical) | - | **v4** (cb98025, 2026-09-27) | See below. |
| **S_HEAD**: siphon heading not pinned; termini cross or converge (user image 3) | 22 (14 RICA + 8 LICA) / 16, all tier A | same | 8 RICA | RICA: **set B v1** (5335a00 / a898eb2, Aug 30-Sep 1); LICA: v4 | Yaw = az(anchor) - az(piece start): `graft_siphon.py:157-169`, used by `graft_three.place` `:390-409`, seam 2 `:530-531`. Plus C6: the lower's own seam-1 heading `graft_three.py:509-511` puts the RICA seam at the midline in 30 / 223. |
| **R_PITCH**: right siphon tilted back on a vertical ICA tip | 17 / 17 (4 tier A) | same | 17 (3 tier A) | **set B v1** | The siphon's first 5 mm leans (about 30 deg) after a vertical anchor (75-86 deg). The frame match straightens the start and tilts the whole siphon 41-55 deg. This was a documented trade-off: a longer start chord was rejected at `graft_three.py:301-305` because it pushed the seam-2 kink to 65 deg. The v1 audit accepted rise errors up to 44.8 mm. |
| **VA4**: v4 VA top frame-matched on the host VA (mr_025_L RVA, user image 4a) | 14 / 14 (the 17 sides kept from v4) | 269 / 165 (158 tier A) | - | **v4** | `graft_v4.graft_va` `:366-385` (place `:380`); the cut is length-matched, `:538`. The host VA tangent there is below 10 deg in 9 / 14 (all v4 tops: median 17 / 27, min -19 / -40 deg). The cut search `:678-700` moves the cut ±30 mm for join and clearance only. |
| **VA5**: v5 VA top Kabsch-fitted with no superior term (mr_027_L, user image 4b) | 319 / 206 (174 / 109 tier A) | - | - | **v5** (5c2cdc0, 2026-10-03) | `carotid_tools/v5/graft/graft_v5_va.py:446-466` with `TB_PLACE = "fit"` `:106`. It fits the TopBrain V3/V4 piece onto the donor's distal remainder past the splice (a different patient's V3, often a different length: 33 vs 64 mm in m_022 mr_020). `frame_rotation` is computed at `:458` and discarded. route13 adds ±3-15 deg turns (`graft_v5_reg_sep.py:197-212`). |
| **ECA**: TopBrain ECA tip on a fork ECA tip that points down | 39 / 36 (16 / 15 tier A) | same | - | **v4** | `graft_v4.extend_eca` `:388-407` (place `:402`); the anchor is the fork ECA's crop end, below 10 deg in 19 / 39. |
| **C6**: lower-section heading at seam 1 (not a TopBrain piece; feeds S_HEAD) | 30 RICA seams within 10 mm of the midline | same | same | set B v1 (left lowers: v4) | `graft_three.py:509-511`. Same frame-match freedom: in the 168 anatomies whose two forks come from one Zenodo patient (same scanner frame), R_lower and R_lower_L differ by a median 110 deg, > 90 deg in 103 (`lower_yaw.py`). An anatomical placement would give about 0. |
| mirrored "_L" | not a cause | | | | x -> -x before prep (`graft_v4.py:198-199`); +z and +y are unchanged and every R is proper. Repro cases from both groups behave identically (k_011 mirrored, w_024_right mr_002 native; w_003_left mr_010 native, w_037_right mr_010_L mirrored). |

**L_PITCH: code path and missing step.** The anchor is the LEFT fork's ICA tip (seam 2: `graft_three.py:530-531`, via
`G3.compose` at `graft_v4.py:577`). The left fork comes from:
- `graft_v4.lower_records` (`:268-318`), which **drops v3's lower rule**. `match_sections.load_lowers` (`:71-97`)
  required `extend_mm <= 10`, i.e. a native ICA of at least 48 mm extended to 58, and a tip diameter of at least 2 mm.
  v4 needs only CCA + ICA >= the seam distance and prefers the native, unextended ICA.
- `left_lower_candidates` (`:338-354`), which tries the patient's own other side first.

The gates at `:578-592` (kink, ECA length, seam landing, clearance) contain no rise or tilt check.

**Why the gates missed it.**
- The join kink is satisfied by construction (above). Clearance (0.6 mm) passes any pose that touches nothing, and a
  siphon lying sideways in free space touches nothing.
- v4's verification (CAROTID_V4.md 350-379) checks det(R) = +1, not the pose.
- v5's VA gates are referenced to v4's pose:
  - `tip_move <= 60 mm` vs v4's blind end (`graft_v5_reg.py:361`), which was wrong in 158 v4 anatomies;
  - course RMS <= 20 mm to v4's corridor;
  - the "rise" gate (`:230`) tests the DONOR's 10 mm chord at the subclavian origin, not the TopBrain piece.
- The siphons, the ECA tips and the 17 kept VA sides are identical v4 -> v5, so no v5 check re-measured them.
- The front-view figures cannot show heading errors.

**Coverage of the lower rules** (`lower_check.json`):
- v3's rule would refuse 23 left lowers, i.e. 113 anatomies' left forks. That covers 34 of the 35 tier-A L_PITCH
  anatomies; the exception is w_003_left mr_016_L (60.5 mm ICA, tip 19.7 deg).
- A native tip-elevation test (< 30 deg) refuses 63 anatomies and covers 27 / 35.
- Either rule covers 35 / 35, at a cost of 133 anatomies.

## Reproduction (2+ flagged pieces per class; tbo/repro/repro.txt, fig_<anatomy>.png)

"doc" = the documented rule re-run with the build's code (`tb_part` -> `frame_rotation` / `G3.place`). "fit" = the v5
Kabsch re-run from the donor record, the vaC pose and `splice_on_donor_mm`. The candidates (all rigid about the seam):
- **anat**: R = diag(-1, -1, 1). TopBrain RAS to branch LPS, i.e. the piece as it sits in its patient (superior AND
  heading pinned).
- **bendT**: anat plus the minimal turn that brings the 5 mm seam angle to at most T deg.
- **untilt**: doc with the superior axis restored (pitch-only fix).
- **atrim**: doc at an anchor moved back along the kept vessel to the last point whose tangent rises at least
  30 deg (the documented `anchor_trim` idea applied to the anchor).

Rise in mm, native = the same piece in its own scan. "join" = the build's 1 mm join kink; "clear" = the lumen gap to
the shipped neighbours.

| class | piece | cause re-run vs shipped | shipped: rise / native, up_tilt, join | candidate fix (best rigid) | gates |
|---|---|---|---|---|---|
| L_PITCH | k_011_left mr_012_L LICA (anchor -57.4, start +65.5 -> tilt identity 122.9) | doc 0.85 deg, 0.91 mm | -38.9 / 61.6, 123.7, 7.4 | anat: 61.6 / 61.6, 0, join **106**; bend60: 7.0, 85.1; atrim (11 mm back): 61.8, 2.6, join 15.4, clear **-4.2** (LVA) | none passes |
| L_PITCH | w_024_right mr_002 LICA (anchor -53.3, start +73.3) | doc 0.16 deg, 0.61 mm | -26.5 / 61.5, 126.8, 8.7 | anat: 61.5, 0, join **83.5**; bend60: 37.0, 71.2; atrim 10 mm: 60.6, 3.5, join 20.9, clear **-5.6** (LECA) | none passes |
| L_PITCH (extras) | k_011_left mr_013_L / 016_L / 023_L LICA | doc 0.59 / 0.86 / 0.60 deg | -38.2 / 69.0; -30.1 / 51.4; -2.2 / 62.2 | anat restores the rise exactly, join 82 / 163 / 93; atrim: 69.2 / 52.2 / 52.0 but clear -2.4 / -1.5 / -2.0 | none passes |
| S_HEAD | w_003_left mr_010 (RICA yaw -26, LICA yaw 42; termini du **-48** vs native +25.6) | doc 0.48 / 0.23 deg | RICA 66.2 / 72.3, crosses 33.5; LICA 51.8 / 70.8, crosses 21.2 | anat: RICA 72.3, crosses 17.3 (seam at u -1.5 = C6); LICA 70.8, crosses 0, clear **-2.7** (LVA); termini du +4.7 | LICA clearance; RICA residual crossing |
| S_HEAD | w_037_right mr_010_L RICA (yaw -3: the crossing is the seam position, C6) | doc 0.44 deg | 64.0 / 71.0, crosses 31.9; termini du 1.1 vs 24.8 | anat: 71.0, crosses 17.7; termini du 15.2; LICA bend45 clear 0.99 (anat hits RCCA, -4.5) | partial |
| S_HEAD (extra) | w_003_left mr_007_L | doc 3.71 / 0.26 deg | termini du -2.7 vs 36.8 | anat: du 26.6, RICA crosses 11.2; LICA clear 0.53 | borderline |
| R_PITCH | w_035_left mr_024_L RICA (anchor 82.1, start 29.7) | doc 10.0 deg (anchor tangent re-sampled) | 24.7 / 52.8, 54.9, 20.0 | bend45: 50.8, 10.2, join 37.5, clear 0.80 | **pass** |
| R_PITCH | m_022_left mr_024_L RICA (anchor 74.9, start 29.7) | doc 1.85 deg | 31.9 / 52.8, 46.1, 36.1 | bend30: 33.0, 45.0, join 54.9; bend45: 42.6, 30.0, join **64.7** | none passes |
| R_PITCH (extras) | w_035_left mr_011; w_025_right mr_021_L RICA | doc 3.90 / 2.54 deg | 28.3 / 49.1, 40.9; 57.9 / 73.3, 51.8 | anat: 49.1, 0, join 32.0, clear 3.5; bend45: 74.9, 7.8, join 48.6, clear 0.71 | **pass** |
| VA4 | m_006_left mr_025_L RVA (anchor -18.8, start 52.4) | doc 5.09 deg | -6.4 / 32.7, 66.4, 37.9 | atrim (cut 13 mm lower on the host RVA): 31.0, 4.3, join 13.5, clear 12.3; anat join 115 | **atrim passes** (VA ends 57 mm apart, native 5.3) |
| VA4 | w_007_right mr_027 LVA (anchor -38.5, start 11.8) | doc 2.28 deg | 3.1 / 30.8, 51.0, 25.2 | atrim: 36.9, 42.7; untilt: 30.8, 0, join 25.2, clear 4.7 | untilt passes |
| VA5 | m_006_left mr_027_L RVA + LVA (tops cross by 21 mm) | fit **0.00 deg** both (target 41 / 34 mm, piece 41 / 34 mm) | RVA 22.3 / 31.2, 71.4; LVA 21.0 / 28.4, 49.2 | anat: ends meet (du 0.3, sep 3.6 vs native 5.7 / 6.2), RVA join 19; LVA join **74.7**; bend45: du -5.2, sep 5.8, LVA join 62.9 | LVA join; VA-VA contact |
| VA5 | m_022_left mr_020 RVA + LVA (RVA crosses 28 mm) | fit **0.00 deg** both (target 33 mm vs piece **64** mm) | RVA 40.7 / 24.0, 82.9; LVA 31.7 / 33.2, 11.2 | bend45: RVA 31.8, 38.1, join 26.1, clear 5.0; LVA clear **-2.3** (RVA); anat: sep 11.0 vs 6.8, joins 60-69 | partial |
| ECA | k_004_left mr_010 RECA (anchor -74.8, start 33.5) | doc 2.48 deg | -21.2 / 18.6, 107.9 | anat: 18.6, join 127; bend60: -12.3; no anchor rising >= 30 within 30 mm | none: skip the extension |
| ECA | w_008_right mr_001 LECA (anchor -56.4) | doc 4.04 deg | -15.1 / 12.5, 96.7 | atrim 11 mm: 16.8, 22.3, join 30.2, clear **-2.3** (LVA) | none |

**At scale** (`batch.py` over all 218 v5 anatomies with a primary-flagged piece; `batch_summary.txt`). The
re-run reproduces the shipped pose: frame pieces at a median 1.1-3.8 deg; v5 fits exactly (<= 0.1 deg) in 397 /
404. The other 7 are the route13 / pose-search extra turns, 0.9-15 deg. The table counts pieces that pass the
repro gates (no neg_rise, |rise err| <= 20, up_tilt <= 40, join <= 60, clearance >= 0.6) under each rule:

| class (n pieces) | shipped | anat | bend45 | bend60 | untilt | atrim |
|---|---|---|---|---|---|---|
| L_PITCH (50) | 0 | 18 (join > 60: 25, clearance: 18) | 19 | 23 | 20 | 15 (clearance 18, up_tilt 18) |
| R_PITCH (17) | 0 | 8 | 9 | 9 | 7 | 1 |
| S_HEAD (22; heading only) | 22 pass, but crossing > 15 mm in 9 | 12 pass, crossing 4 | 15, 5 | 12, 5 | 19, 3 | 22, 8 |
| VA4 (14) | 7 | 5 | 6 | 8 | 9 | 10 |
| VA5 (319) | 106 (up_tilt 212) | 164 (join 118, clearance 70) | 204 | 212 | 248 | 212 / 286 |
| ECA (39) | 0 | 7 | 13 | 10 | 9 | 11 / 38 |

Pair checks (`batch_summary.txt`):
- **ICA termini, 70 anatomies:** the median lateral-order error is 40.2 mm shipped and 12.2 with anat; the order is
  wrong in 8 shipped and 1 with anat.
- **VA ends, 210 anatomies:** the median error is 17.4 mm shipped and 9.0 with anat; the order is wrong in 95
  shipped and 69 with anat.
- untilt and atrim keep the arbitrary heading and make the pairs worse: ICA error 31.5, VA 25-42.

**What the reproduction shows:**
1. The defect is not a missing piece of the documented fix. The fix, re-run, *produces* these poses. It pins roll
   only, and takes pitch and heading from an unchecked anchor tangent.
2. L_PITCH and ECA cannot be fixed by rotating the TopBrain piece. On a down-pointing anchor, every pose either
   breaks the 60 deg join or keeps the piece pitched. The fix has to move or replace the ANCHOR:
   - L_PITCH: a lower rule, or moving seam 2;
   - ECA: skipping the extension.
3. For k_011_right, moving the seam back (atrim, 11 mm) fails v4's own seam-landing gate. The host cut is already
   clamped at HOST_CUT_MAX_L = 110 and the seam lands 1.3 mm short today; 12.3 mm short would exceed the 5 mm gate
   (`graft_v4.py:585`). It also hits the LECA / LVA. So for this class the left lower must change.
4. R_PITCH and S_HEAD are fixable by pinning both superior and heading (anat) with a bounded seam bend of at most
   45 deg. The residual RICA crossings need the lower's own seam-1 heading pinned too (C6).
5. The VA tops need a placement aimed at the basilar: upright, ends meeting near the midline. That collides with the
   current VA-VA clearance rule (next section).

## Side effects a v6 fix must handle

- **Seam angle and join.** Pinning superior and heading turns the anchor/piece mismatch into a bend at the seam:
  - 94-156 deg on the reproduced L_PITCH, VA4 and ECA pieces;
  - batch: join > 60 for anat in 25 / 50 L_PITCH, 118 / 319 VA5 and 29 / 39 ECA.

  A bounded bend (<= 45 deg) trades part of that back as pitch. v6 needs:
  - a seam blend over several mm (as the radius ramp does), so that a 30-45 deg bend stays under the 1 mm,
    60 deg join gate;
  - or a better anchor (lower rule, cut or splice moved to where the vessel rises, as atrim / VA_CUT_STEPS
    already allow).
- **Clearance with neighbours.** A re-posed piece hits what was placed against the old pose:
  - the opposite VA top: the v5 VAs were posed against the shipped siphons;
  - the LECA and the RCCA.

  Clearance failures under anat: L_PITCH 18, S_HEAD 9, R_PITCH 8, VA5 70, ECA 10. v6 has to re-run the build order
  with every gate: lowers -> siphons -> VA tops (v5 donors re-posed) -> ECA tips.

  Anatomical VA tops MEET at the basilar (mr_027_L gap 0.3 mm; native ends 2.5-8.2 mm apart). Decide whether to:
  - fuse them into a basilar confluence, which changes topology and genus and opens a loop the catheter can
    traverse, so it needs SOFA and the navigation checks;
  - or stop each top short of the other by >= 0.6 mm.
- **Route length and seam landing.** A rigid re-pose keeps every length, but end z moves by up to +100 mm (k_011
  LICA end 476 -> 577 mm), so the head extent and the camera/figure framing change. A lower swap changes CCA + ICA
  and must re-pass the seam landing (left seam 157.6 mm, max 5 mm short, HOST_CUT_L 35-110). An anchor trim
  shortens the ICA, the host VA or the donor V3.
- **Gates to add** (they exist nowhere today):
  - rise against native (|err| <= 20 mm) and negative rise for every piece;
  - up_tilt <= 40 (siphons; informational on short VA tops);
  - heading against the anatomical map, plus the pair checks: ICA termini order and spacing, VA ends near the
    midline in native order;
  - an anchor-tangent elevation check before any frame match (siphon seam >= 30 deg; ECA tips skip below about
    20 deg);
  - the same heading check on the lower's seam 1;
  - v5's `tip_move` / course-RMS references must stop pointing at v4's (wrong) VA tips.
- **Train/test split.**
  - Unaffected by any re-pose: the donors per anatomy (right lower, left lower, TopBrain patient, VA donor) do not
    change.
  - It IS affected if the L_PITCH fix swaps left lowers: 63-133 anatomies, depending on the rule. The left lowers
    are nodes in the v4side and strict split definitions, and the roster already drops 153 -> 127 because of left
    forks. Re-run `/sp/v5_va/v5_split.py`, or restrict substitutes to the anatomy's own partition.
  - VA donors are unchanged unless a basilar-targeted re-fit fails a donor's gates.
- **Re-bake and verification.**
  - Every changed xform (R / anchor) moves a real surface. Re-bake those pieces: v5 bake handle rule, real-surface
    carry. Then rerun genus/topology, the v5 post-bake checks, SOFA loading and the figures.
  - The figure scripts need a side view: front views cannot show heading errors.
  - The v4 -> v5 byte-identity argument no longer holds for the re-posed pieces.

## Files (all in /sp/v5_va/tbo/repro/)

- `classify.py` -> `classes.json`, `classes.txt`: class membership and counts for v5, v4 and v3.
- `repro.py` -> `repro.json`, `repro.txt`, `repro.log`, `fig_<anatomy>.png` (18 anatomies, front and side views; red
  shipped, orange the documented rule re-run, blue anatomical, green anatomical + bend <= 45).
- `batch.py` -> `batch/<anatomy>.json` (218), `batch_summary.txt`; `batch_repro_stats.py`: re-run accuracy.
- `lower_check.py` -> `lower_check.json`: v3 lower rule and tip test vs L_PITCH.
- `lower_yaw.py` -> `lower_yaw.json`: seam-1 heading of the lowers.
- `dr.sh`: container runner, /repo read-only. Also `peek.py`, `peek2.py`, `members.py`.

Status: complete
