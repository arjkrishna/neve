# s2_images: v6 figures (front + side) for all 223 anatomies

## Answer so far
- Script: `monitoring/figure_carotid_anatomies_v6.py` 1.2 (md5 2f1aaf16; 1.0 / 1.1 renders archived in
  v6_work/s2images/v1_0, v1_1). One PNG per anatomy, FRONT | SIDE, one shared
  window for the whole set (x -104..148, y -54..101, z 363..619 mm, branch frame), v5 house colours (donor teal
  unchanged, item f deferred).
- Output: `saved/figs/carotid_v6/carotid_v6_<name>.png` (223). Per-image records (drawn vs provenance, ICA-terminus
  midline offsets, host-VA ostium s): `carotid_data/v6_work/s2images/records/<name>.json`; logs `.../logs/shard_{0,1}.log`.
- Run: render_all.sh (2 containers s2i_render0/1, shards 0/2 + 1/2, --skip-existing + claims, --build-json
  results/v6_build.json for the flags provenance lacks: left_pose_map_bounded_dev x29, orient:* x16).
- Review: see "Inspection" below (2 tests, 8 user-named, 10 random, 5 flagged = 24 distinct anatomies, + k_004_left
  mr_011_L as a drawing test). Short answer: in all 24
  every fork, siphon and VA top RISES (none points down; the v5 left-siphon pitch of k_011_left is gone).
  Carotid heading is right in 15 of 24. The 9 exceptions all pass the build gates; 4 carry a flag (w_047 / w_052 via
  left_pose_map_bounded_dev, w_003_right / w_018_right via ica_ap_gt40_R), the 5 medial right siphons carry none:
  right siphons that run
  medially and end 2-8 mm across the midline (w_003_left mr_010, w_012_left mr_003_L, w_017_right mr_011,
  w_042_left mr_010_L) or loop ~6 mm across it before ending 7 mm lateral (w_036_left mr_022),
  a left fork leaning across the midline (w_047_left mr_027, left_pose_map_bounded_dev), a left
  siphon at 58 deg deviation ending across the midline (w_052_left mr_026), and two posterior right siphons (w_003_right
  mr_001 R5 v5 pose AP +46, w_018_right mr_007_L AP +53, both ica_ap_gt40_R). VA tops converge on the confluence
  except w_017_left mr_002_L RVA (D23, 55 mm, accepted) and the 20-30 mm band RVAs of k_011_left mr_013_L and
  w_002_right mr_013_L (flagged). Set-wide: ICA termini across the midline
  R 24 / L 14 (v5 26 / 18), more than 5 mm R 8 / L 9 (v5 17 / 15); 4 w_047_left left forks cross (v5: 5 right forks).

## What each image shows
- FRONT (from anterior; patient's right on the image left; host midline x 14.8 dotted) and SIDE (from the patient's
  right; anterior on the image right; the patient's left side faded as the far side).
- Carotid routes host grey | CCA blue | ICA cyan | siphon orange, white dots at the 3 seams, orange triangle at the
  ICA terminus; ICA extension (114 inherited right-fork extensions; 0 left) as white dashes over the ICA tip.
- ECA: fork ECA magenta, TopBrain ECA tip purple.
- VA: host subclavian grey | donor subclavian + junction olive (xform scC_*, scJ_*; drawn under the donor VA where the
  junction section overlaps it) | donor VA dark teal | TopBrain VA top brown; host VA (no donor) green from its ostium.
- (17) / (19): on donor-junction sides the donor's own distal subclavian (D17, scD_*) olive to its cut end (x), D17
  tube part dashed; else the host's distal subclavian grey.
- Junction marker at the VA ostium: olive hexagon (donor's own, rung 1), yellow hexagon (own + D16 / tube bridge),
  grey diamond (v5's junction, rung 2 / 3), green triangle (host VA, no donor).
- Bake tube windows (bake_v6.va_tubed windows, any section) yellow squares; D27 repair window yellow square, red edge.
- Mesh: baked OBJ as a faint shadow shaded by nearest piece. v5's six named centrelines thin dotted grey.
- FRONT call-outs per side: siphon (re-posed anatomical, source piece, mirrored, deviation, seam bend, terminus offset
  from the midline), fork (lower, rung, re-posed by map / v5 pose, MIRRORED D20, SUBSTITUTED D1, host-cut shift,
  seam-1 bend, ICA extension), VA (D9 rung, junction type, donor, D16 / D17 tubes).
- SIDE: ICA-terminus AP offsets (v6_build.carotid.ica_ap_post_mm = terminus y - host confluence y; > 40 red).
- Title: D22 reference per piece (v4 / own surface D26 / D27 repair window), fork / siphon / ECA per side, VA per side,
  bake tube windows, every flag of v6_build.flags + v6_checks.flags (va_<side>_rung1 omitted, it is in the VA line).

## Set numbers (termini.py on the centrelines; /w/termini.json)
| | v6 | v5 |
|---|---|---|
| ICA termini spacing, median (p5-p95) | 40.9 (18.4-56.8) mm | 66.0 (29.0-123.4) mm |
| RICA terminus y (posterior +), median (max) | 3.2 (54.5) | 24.6 (84.3) |
| LICA terminus y, median (max) | 0.0 (47.2) | 16.9 (94.9) |
| RVA / LVA top to the host confluence, median | 8.7 / 8.7 mm | 26.9 / 25.8 mm |
| ICA terminus across the midline (x 14.8), R / L (heading_scan) | 24 / 14 | 26 / 18 |
| ... by more than 5 mm, R / L | 8 / 9 | 17 / 15 |
| cervical ICA (fork) across the midline, R / L | 0 / 4 (+1 within 5 mm) | 5 / 0 |
| ... within 5 mm of it, R / L | 48 / 19 | 42 / 27 |

## Inspection
(filled below)

Verdict key: "rise" = each piece goes up from its seam (no downward-pointing ICA / siphon / VA top);
"head" = FRONT: ICA and siphon stay on their own side and end lateral of the midline; SIDE: no strong posterior run
(ICA terminus AP <= 40 mm). Offsets from records/<name>.json (ica_terminus_lat_mm: + = on its own side).

### 2 tests (first renders, 1.0; same geometry on 1.1)
- k_011_left mr_012_L: rise + head OK both sides. L fork substituted (w_051_right, mirrored) -- v5's left siphon
  pointed DOWN (L_PITCH, user's image); v6's left siphon rises to z 595. RICA / LICA AP +16 / +14; termini 20 / 29 mm
  lateral. D27 repair window drawn on the right siphon (s 206.75-207.5). VA tops 6 / 11 mm from the confluence.
- w_017_left mr_002_L (D23): carotids rise; both termini medial (7 / 8 mm lateral), RICA AP +26, LICA -15 (left
  siphon dev 35 deg curls anterior). RVA rung 3 (v5's top as shipped): the brown top runs laterally and stops 55 mm
  from the confluence -- visible and flagged (D23). LVA rung 2 (v5's junction).

### User's 8 named anatomies
- k_011_left mr_012_L: see tests. Fixed vs v5 (left siphon rises; left fork substituted).
- k_011_left mr_013_L: rise + head OK. L fork substituted (m_023_left); left siphon rises to z 599 (v5: pointed down,
  ended z 477). Termini 7 / 20 mm lateral, AP +1 / +9. RVA rung 2 (v5 junction): its brown top swings laterally
  (x -28) before turning in, 17 mm from the confluence (va_target 20-30 flag, 22 mm). LVA D17 tube 10.6 mm.
- k_011_left mr_016_L: rise + head OK. L fork substituted (m_030_right); left siphon up to z 577 (v5 z 485). Termini
  12 / 26 mm lateral, AP -2 / +22. VA tops 8.5 / 7 mm from the confluence.
- k_011_left mr_023_L: rise + head OK. L fork substituted (w_051_right); left siphon to z 591 (v5 z 513). Termini
  9 / 13 mm lateral, AP -1 / +5. VA tops 12 / 11 mm.
- w_003_left mr_007_L (VMR 0248 donor): rise + head OK. Siphons (user's S_HEAD image) now go up and anterior; RICA AP
  +1 (v5's right terminus was 75 mm posterior and 18 mm across the midline); termini 26 / 30 mm lateral. RVA top keeps
  TopBrain's V3 loop at the splice; VA tops 5.5 / 1.5 mm from the confluence.
- w_003_left mr_010: forks rise. Two things to note: (1) the RIGHT siphon's top turns medially and its terminus ends
  4.7 mm ACROSS the midline (x 19.5; v5: 27 mm across and 84 mm posterior); (2) the LEFT siphon (substituted fork
  k_007_left, pose compose_frame_match = flagged, siphon dev 21 deg) rises but tilts posteriorly, LICA AP +35 (below
  the 40 flag). LVA host VA (no donor); both VA tops end ~10 mm left of the midline, 14 / 13 mm from the confluence.
- m_006_left mr_025_L (user image 4a, RVA host VA): rise + head OK. RVA (no donor; 1.1 draws grey to its ostium,
  then green) rises and its top reaches 5 mm from the confluence (v5: 60 mm). LVA rung 2. Termini 22 / 26 mm lateral,
  AP +3 / +17.
- m_006_left mr_027_L (user image 4b, VA tops): rise + head OK. Both VA tops converge on the confluence (3.5 / 8 mm;
  v5 33 / 26 mm), not crossed. Termini 15 / 18 mm lateral, AP -6 / +22.

### 10 random (seed 20261007, excluding the named, tests and flagged picks)
- m_022_left mr_020: rise + head OK. R fork R2 (host cut +4 mm, seam-1 bend 45). Termini 10 / 30 mm lateral, AP -7 / +4.
  RVA D17 tube 0.6 mm; bake tube windows on scC_R / scD_R / vaC_R / lower_L (1-3 mm each).
- w_012_left mr_003_L: forks rise. RIGHT siphon's top turns medially and its terminus sits on the midline (2 mm
  across), just in front of the VA confluence; the bake tubed 16 mm of that siphon (siphon_R window) there. Left OK
  (15 mm lateral). AP +5 / -8. VA tops 1 / 3 mm from target.
- w_014_right mr_027 (R5: v5's right-fork pose, flagged): rise + head OK; ICA extension 5.5 mm visible; RICA AP +24.
  Termini 16 / 29 mm lateral; VA tops 2 / 2 mm.
- w_017_right mr_011: forks rise. RIGHT siphon (dev 25 deg) runs medially and ends 8 mm ACROSS the midline, both
  termini left of the midline ~9 mm apart. Same TopBrain patient in w_034_left mr_011 with dev 5 deg ends 9 mm lateral
  -> the crossing comes with the larger deviation. LVA D16 tube 9.7 mm. AP +10 / +3.
- w_034_left mr_011: rise + head OK (termini 9 / 36 mm lateral, AP +5 / +6). LVA D16 tube.
- w_035_left mr_024_L (V6_ITEMS a3 R_PITCH showcase): rise OK; both siphons run anteriorly (AP -14 / -27), termini
  18 / 14 mm lateral. VA tops 4 / 6 mm.
- w_036_left mr_022: forks rise; the RIGHT siphon (dev 25) loops medially over the VA confluence (~6 mm across the
  midline) before ending 7 mm right of it; left terminus far lateral (47 mm). AP -1 / -18.
- w_042_left mr_010_L: forks rise (right ICA extension 6 mm); the RIGHT siphon (dev only 5) runs medially from a seam
  15 mm from the midline and ends 7 mm ACROSS it (the w_042_left fork places its ICA top medially: 3 of the 4 w_042_left
  anatomies cross, mr_010_L / mr_005_L / mr_023 by 6.5-6.9 mm; mr_026_L ends 5.5 mm lateral). RVA host VA, top 9 mm
  from target. AP +7 / +5.
- w_047_left mr_027: right OK (terminus 3 mm lateral). LEFT fork (map deviation 25 deg = left_pose_map_bounded_dev)
  leans medially: bifurcation 7.7 mm from the midline, cervical ICA 1.9 mm ACROSS it; the two ICAs meet at the midline
  in the front view (15-20 mm apart in depth, side view). The same left fork pose in w_047_left mr_022 / 023 / 026_L
  (1.9 mm across, bifurcation 7.6 mm) and mr_018_L (map dev 20: 1.8 mm short of the midline, bifurcation 10.4 mm); the
  w_047_left RIGHT ICA also bulges to 5.6 mm from the midline in all 5.
  1.2 prints the left-fork warning in the call-out and title.
- w_052_left mr_027_L: rise + head OK; termini 13 / 18 mm lateral, AP +4 / +13; VA tops 17 / 2 mm.

### 5 flagged
- w_003_right mr_001 (right_fork_R5_v5 + ica_ap_gt40_R + RVA host VA): the RIGHT carotid keeps v5's pose and leans
  posteriorly in the side view; the right siphon heads up and back, RICA AP +46 (flagged). Left OK (AP -16). RVA host VA
  (grey to the ostium, green above) rises to 6 mm from target.
- w_018_right mr_007_L (ica_ap_gt40_R, the set's largest, 53 mm): front view fine (terminus 33 mm lateral); side view:
  the right siphon (dev 25, R2 host cut +11 mm) runs posteriorly from its seam, terminus 53 mm behind the confluence.
- w_018b_left mr_002_L (R4 map_dev_bend60 + fix loop; item b VA move): carotids rise and head correctly (termini 18 /
  25 mm lateral, AP +4 / +3; seam-1 bend 45, ICA ext 7 mm). RVA rung 2 (v5's junction) keeps v5's winding course with
  a loop at z 500-520 beside the right ECA tip (course RMS > 20 inherited from v5, flagged); VA tops 12 / 8 mm.
- w_002_right mr_013_L (R3 map_bend60, seam-1 bend 56 > 45): right carotid rises (terminus 12 mm lateral, AP +8);
  left fork (dev 35) leans laterally ~25 deg, left siphon rises (33 mm lateral, AP +27). VA tops 28 / 19 mm from the
  target (RVA va_target 20-30 flag).
- w_052_left mr_026 (d22 fork coverage 0.950, the open issue): right side rises and heads correctly (21 mm lateral);
  the missing 65 mm2 of v4 fork wall is not visible at figure scale. LEFT: fork map deviation 43 deg and siphon
  deviation 58 deg -> the left siphon runs medially at ~45 deg from its seam and ends 2 mm ACROSS the midline
  (left_pose_map_bounded_dev). No provenance flag names the siphon deviation itself; the orientation gates passed.

## Final run (1.2)
- 223/223 PNGs in saved/figs/carotid_v6 (1:1 with anatomies_v6, 129 MB, git-ignored), 0 failures (shard_0 112 +
  shard_1 111 written, both "EXIT 0"); 223 records, all 1.2. Drawn RCCA / LCCA length minus provenance total_mm:
  -0.02..+0.11 / -0.06..+0.27 mm. Fork-near-midline warning printed on exactly the 5 w_047_left anatomies.
- Inspection was done on the 1.1 renders (geometry and drawing identical in 1.2; 1.2 only adds the build-record flags,
  the fork warning line and label spacing); 1.2 spot-checked on w_047_left mr_027, w_035_left mr_024_L,
  k_011_left mr_012_L, w_003_left mr_010.

## Decisions
- One PNG per anatomy (FRONT | SIDE) instead of v5's pairs: each image already has two panels.
- Colours: v5 house colours; the donor VA stays dark teal (item f deferred). New: olive for every donor subclavian
  section (the pilot's colour; the re-pilot's orange for the D17 stretch was dropped, too close to siphon orange).
- Orthographic projections (as the pilot), not mplot3d: exact AP / lateral reading in the side view.
- v5's named centrelines as thin dotted grey lines for comparison (the shared window includes them so they are not
  clipped); the patient's left side faded in the side view to tell the sides apart.
- Flags: provenance v6_build.flags + v6_checks.flags + the build record's per-anatomy flags (--build-json
  results/v6_build.json): 29 left_pose_map_bounded_dev and 16 orient:* flags exist only there.
- 1.0 and 1.1 were rendered in full and archived (v6_work/s2images/v1_0, v1_1) after the inspection found
  improvements (host-VA sides drawn green from the brachiocephalic in 1.0; missing ICA-terminus marker; missing
  build-record flags; overlapping side labels).

## Open issues (reported, not acted on: the build is closed)
1. Medial right siphons carry no flag: 24 right ICA termini end across the host midline (8 by > 5 mm), 14 left
   (9 by > 5 mm); list of > 3 mm in /w/heading_scan.json (heading_scan.py). They track the siphon's deviation from the
   anatomical pose (median right terminus 16 mm lateral at dev <= 5 deg, 4 mm at dev > 30 deg). Better than v5
   (26 / 18; 17 / 15 by > 5 mm), all within the orientation gates (crossing > 15 mm would fail).
2. The 5 w_047_left left forks (map deviation 20-25 deg) put the cervical ICA 1.9 mm across the midline (4) or 1.8 mm
   short of it (mr_018_L), bifurcation 7.6-10.4 mm from the midline. Flagged only as left_pose_map_bounded_dev in
   results/v6_build.json, not in provenance; the records agent should carry it into BUILD_v6.json / CAROTID_V6.md.
3. Two posterior right siphons (ica_ap_gt40_R): w_018_right mr_007_L (+53) and w_003_right mr_001 (+46, R5 v5 pose);
   w_027_right mr_016 (+45) not inspected.
4. The script reads the extra flags from --build-json (default <anatomies>/BUILD_v6.json). If BUILD_v6.json is written
   with anatomies.<name>.flags, a re-render picks them up without the scratch path.
