# v6_plan: stage-1 plan, orientation gates, heading evidence, pilot set

Prefix `v6_plan` (state: `../state/v6_plan.md`). Everything is in the scratchpad `v6\` folder; nothing in the repo
was written; anatomies and code were read-only (containers mount `/repo:ro`).

## Answer so far

### Heading maps (task 3; `heading/heading_evidence.py` -> `heading_evidence.txt` / `.json`, `zenodo_frames.json`)

**TopBrain -> branch frame: settled, A0 = diag(-1, -1, 1).**
- TopBrain sources are RAS: rICA at +x of lICA in 25/25 patients; each ICA at +y (anterior) of its VA in 50/50;
  ICA and VA rise in +z in 100/100. (+x right, +y anterior, +z up) is right-handed, so the sources are not mirrored.
- The host branch frame is LPS and proper, and its labels are anatomical: the RCCA and the RVA start at the same
  point, the end of a 48 mm trunk off the arch (the brachiocephalic); the right subclavian runs to the arm at
  x -87.8, the left at x +140.7; the arch segment runs +42.8 in x and +23.7 in y, i.e. from the ascending aorta
  (anterior, right) to the descending aorta (posterior, left); both VAs lie at +y of both CCAs at z 440/470/500.
- So RAS -> LPS is the proper rotation diag(-1, -1, 1) (180 deg about z). Mirrored `_L` pieces are x-flipped in the
  source first; the map is applied after the flip (as `geom_measure.py` already does).

**Zenodo lower forks -> branch frame: settled per patient (the "y sign" was unsettled because there is no single
scanner frame).**
- The left-right axis differs between patients. In the 61 patients with both forks, the left bifurcation lies at
  +x of the right one in 22 (+x = patient left, LPS-like) and at -x in 39 (RAS-like). By group: m 17/0, w 6/42,
  k 6/6 (LPS-like / RAS-like, incl. one-sided patients).
- One-sided patients (16): the scanner origin sits near the patient midline, so the sign of the bifurcation x and
  the side give the axis. That rule agrees with the bilateral answer on 121/122 forks.
- The frames are related by a 180 deg turn about z, not by a mirror: the distal-CCA y-lean flips sign with the
  x axis in 112/122 forks of bilateral patients (an x-only mirror would keep it).
- Chirality: taking each frame as proper (LPS-like -> I, RAS-like -> Rz(180)), the ECA lies anterior of the ICA
  10-20 mm above the bifurcation in 91/138 forks (66 %; 5-15 mm: 90/138), the textbook relation (ECA anteromedial at
  the origin) and the same sign as TopBrain at its own level (ECA anterior 17/29, lateral 29/29). Mirrored frames
  would make it 47/138. Posture-dependent leans (CCA, distal ICA) are not used: they disagree with TopBrain's
  head-coil posture and are not anatomical landmarks.
- Whatever the chirality, a rotation-only placement can only use I or Rz(180), and the x axis (settled) picks
  which. **The fork sign is settled**: per-patient map in `heading/zenodo_frames.json` (77 patients: 29 LPS-like, 48 RAS-like).

**v5 forks against that map (part D).**
- Right route: |yaw| median 135 deg, > 45 in 213/223, > 90 in 181; same-side right forks median 156 (the frame match
  turns them round: the fork CCAs lean anterior, the host RCCA posterior); placed ECA anterior of ICA 78/223
  (native 66 %).
- Left route: |yaw| median 66, > 45 in 140/223.
- 113 right and 101 left routes use a contralateral fork (forks are never mirrored). No rotation makes those
  anatomical in both A-P and M-L; the map keeps A-P.
- Seam-1 bend an anatomical fork pose needs (angle between the host tangent and the mapped fork inlet tangent):
  right median 38.5 deg, <= 45 in 168/223 (<= 30 in 55); left median 22.4, <= 45 in 216/223.
- C6: of the 30 RICA seams within 10 mm of the midline, 22 sit on a contralateral (left) fork; 7 lowers account for
  all 30 (m_022_left, w_003_left, w_027_right, w_052_left 5 each; w_048_left 4; w_037_right, w_046_left 3).

### Orientation gates (task 2; `gates/orient_gates.py`, self-test `gates/out/selftest.txt`, API `gates/out/test_api.txt`)

- One numpy-only module with two entry points: tree mode on any anatomy tree, and an in-memory API
  (`measure_candidate`, `pair_metrics`, `evaluate`, `anchor_ok`, `fork_metrics`, `va_targets`, plus `bend`,
  `untilt`, `yaw_of`, `anat_place`). The API reproduces tree mode to 4.9e-13 on 19 pieces of 4 anatomies.
- Two gate sets: `audit` (the audit's rules exactly) and `v6` (tier A plus up_tilt on siphons and ECA tips,
  heading within 45 deg on siphons and VA tops, two-sided ICA spacing within 20 mm, VA gap within native + 20,
  VA-VA lumen gap >= 0.6, VA end within 20 mm of its anatomical target, anchor >= 30 deg (LICA) / 20 deg (ECA);
  fork heading / tilt with `--fork-gates`).
- **Self-test PASS:** v5 141 tier-A anatomies (siphon 47, VA 115), 218 including tilt; v4 171 / 189; v3 11 / 25;
  every class count (L_PITCH 50/50, S_HEAD 22/16, R_PITCH 17/17, VA4 14/14, VA5 319/206, ECA 39/36, tier-A splits;
  v4 VA4 269/165); every per-rule count equal to the audit's `counts.json`; all pieces numerically identical to
  tbo_geom's `geom_v*.json` (worst difference 0.0), 0 flag mismatches.
- v5 under the v6 set: all 223 fail (heading 604 pieces, ICA spacing 143 anatomies, VA gap 104, VA target 331/446
  tops (median 27.6 mm), anchor 68 seams). The RICA anchor rule is informational (3 seams < 30 deg, minimum 25.8).
- API cases: k_011 mr_012_L LICA in the anatomical pose clears tier A but stays blocked by its -57 deg anchor
  (the fork must change); mr_027_L VA tops in the anatomical pose touch (gap 0.03 mm) and end 32 mm above the host
  confluence (va_touch, va_target); mr_025_L with the cut 13 mm back clears tier A but is 126 deg off in heading.

### Pilot set (task 4; `pilot_set.json`, `pilot/make_pilot.py`)

14 anatomies, 8 mirrored and 6 native: k_011_left mr_012_L (L_PITCH), w_003_left mr_010 (S_HEAD + C6),
w_003_left mr_007_L (the VMR 0248 donor anatomy; S_HEAD), w_035_left mr_024_L (R_PITCH), m_006_left mr_025_L (VA4)
and mr_027_L (VA5, D2), k_004_left mr_010 (ECA, D4), w_018b_left mr_002 (item b), w_025_right mr_021_L,
w_027_right mr_022 and w_047_left mr_023 (the four narrow VAs, c4 label pinch, two c3 donors), w_050_left
mr_004_L (route13-moved, c3 donor ia172_071), and the controls w_051_right mr_008 (native) and w_034_right
mr_017_L (mirrored), both free of any audit flag. Every class, every item, each donor source (ia172, Song, VMR
0248) and the v4-kept VA sides are covered.

### Plan (task 1; `V6_BUILD_PLAN.md`)

- Stage 1 tools T1-T10 (fork pose, left-fork rule, siphon placer, donor subclavian + wall, VA re-pose, ECA rule,
  w_018b move, narrow VA, bake/checks/figures, pilot review), each with a dry-run mode and pilot acceptance.
- Build order: gates -> seam-1 fork poses -> left-fork rule (+ split dry run) -> siphons -> VA + subclavian junction
  (a4 + e together, with the c1 wall fix) -> ECA tips -> w_018b -> one bake, checks, figures, records.
- Decisions D1-D5 as given, plus D3-fork (the fork sign is settled, so forks are posed by the Zenodo map; fallback
  to v5 headings if the pilot fails), D2-target (va_target 20 mm against the host confluence), D6 (RICA anchor
  informational), D7 (two-sided ICA spacing). Each has its evidence and how to reverse it.
- Acceptance criteria per item, and 9 GO / NO-GO rules. The main risk: at v5's host cuts 55/223 right forks would
  need a seam-1 bend > 45 deg, so the fork re-pose is likely to need a host-cut search, or it falls back.

Status: complete
