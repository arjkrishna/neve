# tbo_geom: orientation of every TopBrain piece, v5 / v4 / v3

Agent prefix tbo_geom. State: ../state/tbo_geom.md. Code and results: ../tbo/ (scratch only; nothing in the repo
was written).

## Answer so far (final)

1. **The user is right, and the cause is wider than the LICA.** The four images show three separate mechanisms.
   Every TopBrain piece except the v5 VA tops is still placed by the documented fix, `graft_three.place` →
   `frame_rotation`, which matches the tangent and then the superior axis. That fix pins **roll only**. The
   *check* that went with it (rise against native, rise < 20 mm, negative rise, tilt ≤ 40°) was a one-off audit
   measurement. It was **never written into any grafter as a gate**: grep finds no rise or tilt gate in
   graft_three, graft_v4 or graft_v5_*. So v4's new seams and v5's new VA placement were never measured against it.
   - **(a) Pitch inherited from the seam (LICA, user images 1–2).** The frame match turns the piece's own first
     5 mm chord onto the host tangent. When the left fork's ICA tip points down, the whole siphon is pitched down
     with it. k_011_right's ICA (19 mm) ends 57–59° *below* horizontal, so the four k_011_left LICAs descend:
     rise −38.9 / −38.2 / −30.1 / −2.2 mm against native +61.6 / +69.0 / +51.4 / +62.2, with the superior axis
     tilted 99–136°. 35 LICAs have a seam tangent below 10° elevation (left lowers k_011_right −58°,
     w_024_left −54°, w_046_right −35°, w_013_left −18°, w_022_left / w_008_left / w_002_left / w_047_right
     about 0°). 26 of those 35 are flagged on rise. On the right route no seam tangent is below 25.8°, which is
     why v1–v3 never showed this.
   - **(b) Yaw is unconstrained in every generation, v1–v3's RICA included.** For every frame-matched piece, yaw
     (the rotation about +z, measured against the anatomical RAS→LPS map) equals the azimuth of the host seam
     tangent minus the azimuth of the piece's own first chord, to within a median of 1.3–3.5°
     (tbo/yawcheck.py). The host azimuth is spread round the full circle. So the heading is chance: |yaw| > 90°
     in 111 / 223 RICA and 107 / 223 LICA, close to uniform. Rise survives a yaw, which is why the documented
     metric could not see it. Siphons then cross the midline or each other. In w_003_left (user image 3) the
     RICA ends 19–33 mm past the midline and the termini swap sides. There is a lower-section contribution too:
     the right fork's ICA (w_003_left) already brings the RICA seam to the midline (u_seam −1.5 mm). 30 RICA
     seams lie within 10 mm of the midline, against native ICA starts 12–29 mm from it.
   - **(c) v5 VA tops have no superior constraint at all.** graft_v5_va.tb_rotation uses TB_PLACE "fit": a
     Kabsch fit of the TopBrain piece onto the donor's own distal remainder. v4 VA tops are frame-matched, so
     (a) and (b) apply to them. m_006_left mr_025_L's RVA was kept from v4 (frame-matched). It ends 6.4 mm
     *below* its seam (native +32.7) and its chord is 98° off vertical. In mr_027_L the two tops cross: the RVA
     end lies 21 mm left of the LVA end, where the native order is 5.7 mm the other way.
2. **Identity.** The v5 RICA, LICA and both ECA tips are identical to v4 (223 / 223 / 52 / 53). The RICA is
   identical v4 = v3 (223). So every siphon and ECA-tip defect is inherited: LICA from v4, RICA from v1–v3. The
   VA tops were re-posed in 211 R / 218 L and are identical in 12 R / 5 L, the 17 kept sides.
3. **Counts (tier A = a visible course defect; rules below).**

   | piece | n | v5 tier A | v4 tier A | v3 tier A | + up_tilt > 40° (v5) |
   |---|---:|---:|---:|---:|---:|
   | RICA | 223 | 18 | 18 | 11 | 31 |
   | LICA | 223 | 43 | 43 | – | 58 |
   | RECA tip | 52 | 5 | 5 | – | 16 |
   | LECA tip | 53 | 11 | 11 | – | 23 |
   | RVA top | 223 | 94 | 128 | – | 156 |
   | LVA top | 223 | 94 | 128 | – | 177 |
   | anatomies with any tier-A piece | 223 | **141** (siphon 47, VA 115) | 171 | 11 | 218 |

   v3's 11 are 10 midline crossings and 3 rise errors (−20.8 to −28.0 mm). The pair rules need the LICA, so they
   are not in v3. The v3 RICA passes all three documented gates: rise min 24.7 mm, no negative rise, worst error
   −28.0 mm against v1's 44.8.
4. **Sensitivity.** My metrics flag every piece the user pointed at, in v5 and identically in v4:
   - the k_011_left LICAs in mr_012_L, 013_L, 016_L and 023_L: rise < 20, negative rise, rise error, droop,
     up_tilt;
   - w_003_left mr_007_L: ica_cross on both siphons and a midline crossing (19.3 mm) on the RICA;
   - w_003_left mr_010: midline crossings of 33.4 mm (RICA) and 21.2 mm (LICA), plus ica_cross;
   - m_006_left mr_025_L RVA top: negative rise and rise error;
   - m_006_left mr_027_L: va_cross on both VA tops.

   One correction to the user's reading of image 3: the *RICA* (the v3-era piece) crosses further than the
   LICA. Its seam already sits at the midline. The metrics also flag pieces the user did not name. In those
   anatomies the VA tops end lateral of the midline, or are up_tilt-only.

   I looked at 20 more anatomies. Tier-A flags matched what the images show in every flagged case:
   - LICA descending: w_024_right mr_002, w_013_right mr_004_L;
   - LICA crossing the midline: w_046_left mr_020_L;
   - siphons converging: w_046_left mr_012 (missed by the first rule set; ica_converge was added);
   - RICA horizontal across the midline: w_035_left mr_024_L, w_037_right mr_010_L;
   - VA tops crossing: m_022_left mr_020, w_013_right mr_004_L;
   - VA top lateral or descending: w_007_left mr_013_L, k_008_right mr_010_L, w_048_left mr_010_L,
     w_037_right mr_010_L (missed at first; va_lateral was added);
   - ECA tips dropping: w_036_left mr_001_L.

   up_tilt alone is right for siphons: w_050_left mr_026_L, a horizontal lateral RICA with rise error only
   −16.7; and w_036_right mr_023, a diagonal LICA. It is not visibly wrong for short v5 VA tops: w_022_right
   mr_026, m_022_left mr_016 and w_012_right mr_026_L look plausible. Treat VA up_tilt as informational.

   Controls with no primary flag: w_007_left mr_005, w_051_right mr_008 and w_034_right mr_017_L look right.
   w_048_left mr_008 shows tangled siphons that no piece rule catches. That one is a lower-placement fault: the
   RICA seam is 4.5 mm from the midline. It is not a TopBrain-piece fault.

## Method

- **Piece extent.** The shipped `Centrelines_comb` curve (json → branch `(y, −z, −x)`), from the point nearest the
  piece's `xform.<key>.anchor` to the curve's end:
  - RICA `siphon` on RCCA;
  - LICA `siphon_L` on LCCA;
  - `ecaext_R` / `ecaext_L` on RECA / LECA;
  - `va_R` / `va_L` on RVA / LVA.

  Anchor distance ≤ 0.50 mm in all versions, except one deflected v4 LVA, w_036_left mr_022, at 5.7 mm.
- **Native.** The same points back-mapped through the inverse xform, `q = (v − anchor) R + origin`, land on the
  raw TopBrain source × mirror at a median distance of 0.05–0.09 mm (max 1.35 mm, tbo/check_map.py). The native
  metrics are therefore exactly the shipped piece in the patient's own RAS frame. This is the documented "rise
  against the donor's own native rise". No script for it exists in the repo or in git history; it was
  re-implemented here.
- **Axes.**
  - Branch frame = LPS: +x is the patient's left (RCCA at −x), +y posterior (the host VAs lie +y of the CCAs),
    +z superior.
  - TopBrain = RAS.
  - A correctly oriented piece has `diag(−1, −1, 1)·R = I`. up_tilt = angle(R ẑ, ẑ). yaw = the twist of that
    product about z.
  - Host midline x = 14.8, the host's vertebrobasilar confluence. Head-level estimates run 8.6–21, so the
    midline is uncertain by ±6 mm and the rules leave at least a 10 mm margin.
- **Metrics per piece** (geom_v{3,4,5}.json; one row per piece in pieces_all.tsv):
  - rise, native rise and rise error;
  - chord tilt and principal-axis tilt, placed and native;
  - up_tilt, yaw and yaw_pred;
  - droop below the seam;
  - medial end and midline crossing (cross_mm);
  - lateral span;
  - seam angle (5 mm chords) and seam-tangent elevation;
  - azimuths and u_seam;
  - pair metrics: ICA termini and VA ends (separation, dz, lateral order).

## Rules (tbo/geom_flags.py; documented thresholds plus native-derived ones)

- **Tier A.** Any one of these flags the piece:
  - rise < 20 mm (siphons);
  - negative rise where the native rise is > 5 mm;
  - |rise error| > 20 mm;
  - droop > 10 mm where native droop is < 5 mm;
  - midline: the piece runs > 15 mm past the midline (native max 6.8);
  - va_lateral: the VA end lies > 25 mm lateral of the midline (native ≤ 12.6 in all 25 patients);
  - ica_cross: the RICA terminus lies left of the LICA terminus (native 24.8–38.2 mm right of it);
  - ica_converge: the termini are > 20 mm closer laterally than natively;
  - va_cross: the RVA end lies > 10 mm left of the LVA end (native ≥ −0.3 in all 25).
- **Primary** = tier A plus up_tilt > 40°. This is the documented "tilt ≤ 40°", applied to the piece's own
  superior axis.
- **Secondary**, reported only:
  - yaw > 90°;
  - va_gap: VA ends > native + 20 mm apart (native 2.5–8.2 mm in 23 / 25; mr_013 and mr_026 have one short VA
    at 34–38 mm);
  - ica_splay: termini > native + 40 mm apart laterally;
  - chord tilt > native + 40°;
  - seam > 60°.
- **VA-top rule, justified from native data** (native_topbrain.json, 25 patients): the two VA ends meet at the
  basilar 2.5–8.2 mm apart, in the anatomical order. Each VA end lies within 12.6 mm of the midline. In v5 the
  tops end a median 26.6 mm apart (v4 34.5) and cross by up to 34.8 mm.

## Breakdowns (counts.json, breakdown_v5.txt, summary_counts.txt)

- **Mirrored vs native.** Tier A is similar in both: RICA 11/120 vs 7/103; LICA 24/120 vs 19/103; VA top
  50 + 48 / 120 vs 44 + 46 / 103. The exception is the LECA tip, 1/21 vs 10/32. Mirroring is not the cause.
- **Left-fork donor (LICA, tier A).**
  - Pitch-driven, with the ICA tip below 10°: k_011_right 5/5, w_024_left 5/5, w_008_left 4/7, w_013_left 4/5,
    w_047_right 3/5, w_022_left 2/4, w_046_right 2/2, w_002_left 1/2.
  - Lateral or yaw-driven, with a vertical ICA tip: k_012_left 5/22, k_004_right 3/5, w_003_right 3/5.
- **v5 VA tops by TopBrain patient** (= one donor each). Tier A clusters by donor: mr_010 / ia172_159 15/15 RVA;
  mr_020 / Song 1218002 14/16 each side; mr_001 / ia172_165 11–12/12; mr_002, 003 and 004 all. Others are low:
  mr_011 1–2/16, mr_014 RVA 0/8. A fit onto a donor's distal remainder puts the top wherever that donor's own
  V3/V4 runs.
- **v5 vs v4 VA tops.** v5 is better on the visible rules: tier-A anatomies VA 115 vs 158, va_cross 47 vs 67,
  rise error 6 vs 45. It is worse on up_tilt (219 vs 79 pieces), as expected for a fit with no superior axis.

## For v6 (proposals; not applied)

1. Encode the documented check as a gate on every TopBrain seam: rise against native, negative rise, and
   up_tilt ≤ 40°. Add the pair checks: ICA termini order and spacing, VA ends meeting near the midline.
2. Pin yaw anatomically rather than from tangent azimuths. All sources are RAS, or LPS for the host, so the
   rotation should stay near `diag(−1, −1, 1)`. Let the tangent decide only a small correction, or replace the
   frame match with the anatomical map plus a bounded bend at the seam.
3. Reject or trim left lowers whose ICA tip points below about 30° elevation. Better, measure the seam-tangent
   elevation and refuse a pitch above 40°.
4. VA tops: target the basilar, with ends meeting near the midline in native order, instead of a
   donor-remainder Kabsch fit.
5. Lowers have the same yaw freedom at seam 1. The right fork's ICA brings 30 RICA seams within 10 mm of the
   midline. Not measured here (out of scope).

## Caveats

- Native medial-end values for ICAs use each patient's VA-end midline. For some patients that sits up to
  13.6 mm from the ICA-termini midpoint. The ICA rules use placed values and pair spacing, so they are not
  affected.
- seam_angle uses 5 mm chords, not the build's 1 mm join kink, so values above 60° (v5 VA, 38 + 20) are not
  build-gate violations.
- The images are frontal 3D projections. Anterior–posterior heading errors (yaw) are mostly invisible in them.

## Files (all in /sp/v5_va/tbo = C:\...\scratchpad\v5_va\tbo)

- `geom_v5.json`, `geom_v4.json`, `geom_v3.json`: per anatomy, per piece, with flags.
- `pieces_all.tsv`: one row per piece.
- `flagged.tsv` (primary) and `flagged_tierA.tsv`.
- `identity_v5_v4.json`, `counts.json`, `summary_counts.txt`, `breakdown_v5.txt`, `peranat_v5.txt`,
  `peranat_v4.txt`, `native_topbrain.json`.
- Scripts: `geom_measure.py`, `geom_flags.py`, `check_map.py`, `yawcheck.py`, `seamstats.py`, `native.py`,
  `show.py`, `dr.sh`.
