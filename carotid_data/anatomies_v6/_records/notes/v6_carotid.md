# v6_carotid: the v6 carotid-side composer (siphons, left forks, ECA tips)

Prefix `v6_carotid` (state: `../state/v6_carotid.md`). Code in `/sp/v6/carotid/` (= scratchpad `v6\carotid\`); outputs
in `carotid_data/v6_work/carotid/` (`/out` in the containers). Read-only on the repo, `anatomies_v2..v5`,
`carotid_tools/`, `topbrain_tools/`; containers mount `/repo:ro`. Items: V6_ITEMS a2, a3 (siphons and forks), a5;
decisions D1, D3, D4 (+ the plan's D3-fork, D6, D7) and the user's addendum ladder (DECISIONS_ADDENDUM.md).

## Answer so far

Final state: `compose_v6.py` 1.2.1, strict soft mode, v5 fork headings. Every number below is from the files named
in the sections further down.

1. **The tool** re-composes the carotid side of any v5 anatomy: both siphons re-posed in the anatomical frame
   (A0 = diag(-1, -1, 1)) with a bounded, blended seam bend; the left fork chosen by D1 + the user's ladder
   (rungs 1-6, never the host LCCA tube); ECA tips by D4; every candidate gated by orient_gates v6 and every v4 gate
   before anything is written.
2. **Dry run on all 223: 223 / 223 pass.** Per class: L_PITCH 50 / 50 (20 keep their fork, 30 substituted),
   S_HEAD 22 / 22, R_PITCH 17 / 17, C6 30 / 30, ECA 39 = 19 skipped (D4) + 20 kept or re-posed and passing.
3. **Ladder, count per rung:** rung 1 (own fork) 180, rung 2 (split-compatible native substitute) 43, rungs 3 / 4 /
   5 / 6: 0. No anatomy loses its left fork or left siphon, no ICA is tubed, no substitute needs the guard-less
   rung 3. 43 substitutes: 35 for a down-pointing anchor, 8 for no passing pose / pair without a start trim.
4. **Split on the planned assignment: `v5_split.py --baseline split_v4.json` EXIT 0, none of the 45 quantities
   lowered** (strict k16_div6x6 / k17_div6x6 raised by 1 / 2).
5. **Pilot written** (`pilot/<name>/`: Centrelines_comb + provenance.json with the siphon / fork xforms, ladder and
   ECA fields + carotid_record.json) and verified from the files: 14 / 14 ok; the full plan for all 223
   (`full_plan_v121/`) verifies 223 / 223 from the files.
6. **Right route** byte-identical to v5 up to seam 2 in 223 / 223; soft (v5 VA) contacts left in 2 anatomies (right
   siphons), for the VA merge.
7. **D3-fork (map mode, evidence only):** the pure Zenodo map needs a seam-1 bend > 45 deg on 31 % of right forks
   (GO rule 4 allows 10 %), so the plan keeps v5 fork headings; the bounded map variant passes 222 / 223 but moves
   the training route in every anatomy.
8. **Rungs 5 / 6 work when forced** (test only): 11 / 12 representative failing forks pass at rung 5 with an ICA
   tube of 5-9.5 mm; the twelfth fails the ICA-spacing gate, which an extension cannot fix (it takes rung 2 in the
   plan).

## The tool: `compose_v6.py` (version 1.2.1)

Starts from a v5 anatomy (provenance + Centrelines_comb) and re-composes its carotid side. Imports `graft_three`,
`graft_v4`, `graft_siphon`, `extend_ica` (repo, unchanged) and `/sp/v6/gates/orient_gates.py`; every candidate is
gated before anything is written. Earlier versions: `compose_v6_v12.py` (1.2), `compose_v6_v11.py` (1.1), 1.0
superseded. 1.2 added the addendum ladder and graft_v4's candidate order; 1.2.1 added the VA-clean preferences of
strict soft mode for substitutes and ECA tips (below).

- **Re-pose of a TopBrain piece (siphons, ECA tips).** The shipped piece is back-mapped into its own mirrored source
  frame, `Q = (P - anchor) R + origin` (orient_gates' tree-mode measure, which lands on the raw TopBrain centreline
  at 0.05-0.09 mm). A new rotation places it at the same seam point `K[-1]`:
  `P' = (Q - Q[0]) R'^T + K[-1]`, and the xform becomes `{R', origin = Q[0], anchor = K[-1]}`. With `R' = R` this
  equals v5's map exactly, so the bake's real-surface carry stays consistent with the centreline; points and radii
  are the shipped ones.
- **Pose search (D3).** `R' = Rz(psi) rot(h(omega), phi) A0`, `A0 = diag(-1, -1, 1)` (TopBrain RAS -> branch LPS,
  settled in `heading/heading_evidence.txt`). Grid: psi -44..44 (heading), phi 0..39 (tilt of the piece's own
  superior axis), i.e. 1 deg inside the heading 45 / tilt 40 gates (1.1: a pose exactly on a gate re-measured
  as 40.0000001 from the written R and failed in 19 anatomies), omega step 30 deg. Candidates must bring the 5 mm
  seam angle to <= 45 deg; cost = dev^2 / 10 + 0.25 x seam angle (dev = rotation angle away from anatomical). The
  anatomical pose wins whenever it passes.
- **Blend (D3).** The remaining seam turn (<= 45) is spread over the piece's first h mm (h = ceil(seam / 7.5), 3-8
  mm; none below 10 deg) by a cubic Hermite that leaves the seam with the kept vessel's own tangent (G1) and meets
  the rigid piece at h with its tangent. Piece side only: the kept vessel is untouched, so the right route is
  byte-identical to v5 up to seam 2. Recorded as `xform.<key>.blend_mm`.
- **Gates per candidate:** orient_gates v6 piece flags (tier A + up_tilt_v6 + heading + anchor_low), the 1 mm join
  (`graft_v4.join_kink` on the composed 1 mm curve, span = blend + 3 mm) <= 60, self clearance (12 mm window) and
  clearance >= 0.6 mm against every carotid / host branch (HARD). Against the v5 VAs (RVA, LVA) clearance is SOFT:
  recorded, preferred, not blocking, because the VA tool (e + a4) re-poses both VAs after this step.
- **Pair (siphons):** every passing right x left candidate pair, vectorised, 0.5 mm inside the pair gates: du >= 0,
  |du - du_native| <= 20 (ica_cross, ica_converge, ica_spacing; D7) and the cross clearances. Cheapest pair wins;
  `pair_ok` re-checks with orient_gates' own `pair_rules`.
- **Left fork: D1 + the addendum ladder (1.2).** No anatomy ever falls back to the host LCCA tube. The first rung
  that passes every gate (piece, pair, join, clearance, graft_v4's left gates) is taken and recorded in
  `left_fork.rung` / provenance `left.v6_ladder_rung`:
  1. the own (v5) left fork, siphon re-posed anatomically, no start trim;
  2. a substitute with its native ICA whose donor node is on the anatomy's split side in every saved v5 split
     solution (SplitGuard over `results/split_v5_final.json`: defs v4side + strict, roster + k16 / k16_div6x6 / k17 /
     k17_div6x6; accepted substitutions are committed so later anatomies see them);
  3. a native-ICA substitute the guard cannot certify (accepted only if the final `v5_split.py --baseline` exits 0);
  4. a substitute with the documented ICA extension <= 10 mm (`graft_v4.lower_records`' extended form), then the
     siphon anchor (start) trim 2-15 mm on the own fork, then on the composed substitutes;
  5. keep the fork (own, then up to 3 composed split-keeping substitutes) and extend its ICA upward
     (`extend_fork`: extend_ica's templates -- the distal 10 mm of the 31 manifest ICAs that need no extension, in
     the frame [tangent, +z orthogonalised, binormal] -- chained in the branch frame, the most-rising template each
     10 mm, E = 5..60 mm, radius = seam calibre; gates: kink <= 60 per 1 mm, hard clearance 0.6 mm incl. the fork's
     own ECA, self clearance); flagged `left.v6_tubed_ica_mm` + `left.v6_tubed_ica` (from_mm, templates, anchor
     before / after);
  6. as 5 (E = 0..60) with the full bend allowance: seam bend <= 90, blend <= 12 mm, the 1 mm join <= 60 binding.
  Inside a rung: `graft_v4.left_lower_candidates`' order (the route patient's own other carotid first, calibre
  |tip_d - siphon proximal d|, left carotids before right, name), never the anatomy's own right lower; a substitute
  already used 5 times as a left fork ranks last in its rung (match_sections' LOWER_CAP). The v3 rule and the tip
  test are recorded as information only. Each substitute goes through `graft_three.compose` with graft_v4's left
  constants and gates (ECA >= 17 mm, seam landing <= 5 mm short, HOST_CUT_L 35-110, kink, LECA trim / re-entry /
  fused run, route clearance + ramp growth).
- **ECA tips (D4).** Kept in the v5 pose if that passes the v6 gates and clears the re-posed siphons; else re-posed by
  the search (psi +-180, no heading gate); else, or if the fork ECA tip rises < 20 deg, SKIPPED
  (`eca_extension.<side>.why`). A skipped right extension restores v3's unextended RECA exactly; a skipped left one
  is truncated at the join. A substituted fork tries `graft_v4`'s ECA extension rules (piece >= 10 mm, calibre
  within 1 mm) then the same gates.
- **Forks (D3-fork):** `--fork-mode v5` (default) keeps v5's fork poses. `--fork-mode map` re-poses the right fork
  about seam 1 by the per-patient Zenodo map with the same bounded search (fork tilt <= 40, heading <= 45, seam-1
  angle <= 45, blended on the donor CCA side); GO rule 4 evidence only.
- **Soft mode:** `strict` (default of every run below), `prefer` (+10 cost), `ignore`. Strict: VA-clean siphon
  poses first (lexicographic); (1.2.1) inside a ladder rung a substitute whose new route, new LECA or left siphon
  touches a v5 VA is held as a fallback while the rung's other candidates are tried for a VA-clean one (taken only
  if none is clean); an ECA tip extension is searched over 60 passing poses and SKIPPED if every one touches a v5 VA.

Files: `compose_v6.py` (tool), `summarize.py` (dry-run aggregate incl. ladder), `extra_stats.py` (V6_ITEMS a2/a3/a5
checks), `make_split_tree.py` (planned split tree), `verify_pilot.py` (re-measures written folders from the files),
`cmp_runs.py`, `fig_pilot.py` (front + side views), `fork_stats.py` (map mode), `show.py`, `chain_v12.sh`,
`chain_map_v12.sh`, `chain_r56fig.sh`, `pilot_names.txt`, `ladder_test_names.txt`, `dr.sh` (container runner,
names `v6car_*`).

## Decisions taken here (reversible)

- **Re-pose = back-map the shipped piece** (keeps points, radii and the real-surface carry); undo: none needed, R' = R
  reproduces v5.
- **Clearance against the v5 VAs is SOFT** (the VA tool re-poses them next); `strict` mode makes VA-clean poses win
  lexicographically. Undo: `--soft-mode ignore|prefer`.
- **Forks keep v5 headings by default** (GO rule 4 not met by the pure map; see the map section). Undo:
  `--fork-mode map`.
- **Ladder reading:** the own fork WITH an extra siphon start trim counts as rung 4 (after the substitutes), because
  rung 1 of the addendum names no trim. Undo: add `START_TRIMS_MM` to the rung-1 `try_left` call.
- **Inside-rung order = graft_v4's** (the user's "v4 candidate order"); 1.1 ranked the v3 rule / tip test before
  calibre. The use cap (5) is kept from match_sections. Undo: `rank_substitutes` key.
- **VA-clean substitutes first (1.2.1):** the v5 VAs are soft for the carotid step, but v4 gated them hard and a new
  fork that already clears them leaves the VA merge less to fix; the choice stays inside the rung and graft_v4's
  order decides among the VA-clean ones. Undo: `--soft-mode prefer` (or drop the fallback block).
- **ECA tips touching a v5 VA are skipped in strict mode (1.2.1):** D4's spirit (the ECA is not navigated). Undo:
  the `soft_clear` test in `decide_eca`.
- **Rung 5 extension in the branch frame with +z** (what the anchor gate measures) rather than in the donor scanner
  frame before placement (v3 set B); greedy most-rising template. Undo: `extend_fork`.

## Full dry run on all 223 (1.2.1: `dry_v121/<name>.json`, `summary_dry_v121.{json,txt}`)

`compose_v6.py dry --soft-mode strict` (v5 fork headings, SplitGuard on, use cap 5); 673 s for 223. Checks:
`extra_stats_v121.txt`; per-substitution table: `ladder_detail_v121.txt`; VA contacts: `soft_contacts_v121.tsv`.

- **223 / 223 anatomies pass every carotid gate** (orient_gates v6 on RICA, LICA and every kept ECA tip; ICA pair;
  1 mm joins; hard clearance 0.6 mm; graft_v4's left-route gates). Both siphons placed in all 223; no anatomy
  loses its left fork or left siphon; 0 blocking orientation flags.
- **Per class (the audit's v5 classes, `gates/out/orient_v5.json`):**
  - L_PITCH 50 pieces: 50 pass. 20 keep their v5 fork (anchor rises >= 30 deg and the anatomical re-pose alone fixes
    them), 30 get a substituted fork.
  - S_HEAD 22: 22 pass. R_PITCH 17: 17 pass. C6 (30 RICA seams < 10 mm from the midline): 30 pass, RICA midline
    crossing <= 14.6 mm.
  - ECA 39 tips: 19 skipped (D4), 20 kept or re-posed and passing (0 ECA tips flagged by a primary or v6 rule; 21
    carry the secondary 'yaw' flag, which ECA tips are not gated on).
- **Ladder (addendum), count per rung:** rung 1 (own fork) **180**, rung 2 (split-compatible native substitute)
  **43**, rungs 3 / 4 / 5 / 6: **0 / 0 / 0 / 0**. No tubed ICA, no LICA anchor trim, no split-incompatible
  substitute, no anatomy without a left fork. Why the 43 v5 forks failed: anchor_low 35 (k_011_right -57 to -59 deg
  x5, w_024_left -53 x5, w_047_right 3-6 x5, w_008_left -3 x7, w_013_left -15 to -21 x5, w_022_left ~0 x4,
  w_002_left x2, w_046_right -33/-38 x2); no passing LICA pose or ICA pair without a start trim 8 (m_006_left
  mr_027_L, w_003_left mr_024, w_012_left mr_021_L, w_016_left mr_024, w_029_right mr_010_L, w_036_right mr_014_L /
  mr_018_L, w_040_left mr_021).
- **Substitutes (43):** 24 left and 19 right carotids; most used 5 times (case_w_052_left, case_w_006_left); seam
  landing error 0.0 mm (host cut 63.7-106.2, inside 35-110); none extended. Filters as information: the v3 rule
  passes 20 / 43 -- 18 fail only on its ICA-ALONE length (manifest extend_mm 10.3-28.1 > 10; under graft_v4's
  composition the native CCA + ICA reaches the seam and the host cut absorbs the rest, no tube), 5 are
  DEFECTIVE_LOWERS (case_w_010_right x3, case_k_012_right x2: v3 collision failures; v6 gates check collisions
  directly); the tip test passes 32 / 43 (the 11 others have native tip elevations 21-29.5 deg; the placed anchor
  rises >= 30 deg in all 43).
- **V6_ITEMS checks:** a3 ICA-termini order error |du - du_native| median 13.9 mm, p90 19.2, max 19.5 (the batch:
  v5 40.2 -> 12.2 anatomical); midline crossing max RICA 14.7 / LICA 12.6 mm (0 > 15); a2 L_PITCH = 0, seam landing
  re-passed; a5 ECA class = 0.
- **Siphon poses:** RICA anatomical exactly in 78 / 223, LICA in 71; deviation from anatomical median 5 deg, p90
  30-32 deg, max 55.7; tilt > 20 deg in 38 RICA / 31 LICA (max 39); |yaw| <= 44 by construction; |rise error| median
  0.7 mm, max 19.4; seam angle median 32.9-33.8, max 45.0; 1 mm join median 18.7-19.8, max 38.1; blend
  0 / 3 / 4 / 5 / 6 mm; the blended tube leaves the rigid piece by median 0.51 mm, p90 1.02, max 1.72 mm. Siphon
  start trim only on two RICAs (w_024_right mr_002 12.9 mm, w_037_right mr_023 3.0 mm; the right fork is never
  swapped, so the trim is its only escape).
- **ICA spacing (D7) drives the deviations:** with v5 fork headings the host's two siphon seams sit ~55 mm apart
  against native termini ~28-36 mm, so many pairs need tilt / yaw to keep |du - du_native| <= 20 (|dd| > 10 in
  139 / 221).
- **ECA tips (D4):** 51 kept in the v5 pose, 18 re-posed (+8 on new forks), 28 skipped for anchor_low (+2 on new
  forks), 1 skipped because every pose touched the v5 LVA (w_024_right mr_002, new fork); new forks otherwise skip
  for a TopBrain ECA piece < 10 mm or none (graft_v4's rule).
- **Right route:** byte-identical to v5 up to seam 2 in 223 / 223.
- **Soft (v5 VA) contacts: 2 anatomies** (1.2: 13), both right siphons against a v5 VA that the VA tool re-poses:
  w_039_right mr_021_L (RICA vs the RVA TopBrain top, -2.85 mm), w_046_left mr_026 (RICA vs the donor LVA, -2.13 mm).
- **Versions:** 1.2.1 vs 1.2 (`dry_v12`, split EXIT 0 too): 15 substitutes changed by the VA-clean preference, soft
  contacts 13 -> 2. 1.2 vs 1.1 (`superseded_v1_1/dry_v11`): the graft_v4 order changed about 20 substitutes, and the literal ladder
  moved 4 anatomies that 1.1 kept with a 3 mm LICA start trim (w_003_left mr_024, w_012_left mr_021_L, w_016_left
  mr_024, w_036_right mr_018_L) to rung-2 substitutes.

## D3-fork evidence: map mode (1.2: `dry_map_v12/`, `summary_dry_map_v12.txt`, `fork_stats_v12.txt`; GO rule 4)

- Right forks re-posed about seam 1 by the per-patient Zenodo map with the bounded search: 223 / 223 posed; 153
  exactly on the map, the rest |yaw vs map| <= 10 and tilt <= 20; chosen seam-1 angle median 40.0, max 45.0; seam-1
  1 mm join median 19.9, max 24.5.
- **The pure map pose needs a seam-1 bend > 45 deg in 70 / 223 right forks (31 %)**; with the bounded deviation,
  0 / 223. GO rule 4's threshold is 10 %, so by the pure map the fork re-pose is NO-GO and the plan keeps v5 fork
  headings (the default of every run above); whether a bounded deviation of <= 10 deg yaw / 20 deg tilt from the
  map counts as "anatomical" is the orchestrator's / user's call.
- The carotid side then passes in 222 / 223: **case_w_003_right__topcow_mr_001 fails on every rung** -- the
  map-posed right fork leaves its siphon an anatomical seam angle of 96 deg, a single RICA pose within the 45 deg
  bound, and no left fork (128 tries over rungs 1-6, incl. the tubed ICA and the full left bend allowance) makes a
  passing ICA pair with it. The ladder is a LEFT-side ladder; in map mode the right siphon would need its own bend
  allowance (the training route) or a joint fork + siphon search. In the default mode this anatomy keeps its own
  left fork (rung 1).
- Other costs: the RCCA training route changes from seam 1 in every anatomy (right route identical to v5 in
  0 / 223); soft (v5 VA) contacts rise from 13 to 45 anatomies under 1.2 (17 right siphons touch the v5 RVA; the
  map run predates 1.2.1, whose VA-clean preferences act on the left side and ECA tips only); ICA pair |dd|
  median 13.6 (v5 headings 13.9).
- Caveat on the map itself: under the plan's "proper" reading of the Zenodo frames the distal CCA leans anterior in
  127 / 138 forks and the distal ICA in 110 / 138 (heading_evidence.txt C6 table), while the host CCAs lean
  posterior and TopBrain's lowest ICA leans posterior in 37 / 50. The plan set these leans aside as posture; the map
  rests on the ECA-anterior landmark (91 / 138). If the frames were improper (y flipped), no rotation could be
  anatomical in both axes.

## Rungs 5 / 6 exercised (TEST ONLY; not the plan, where rungs 3-6 are never reached)

- `--skip-rungs 2 3 4` (and `2 3 4 5` for rung 6) forces the last resorts. Demonstration written and verified from
  the files (`ladder_test_r56_pilot/`, `ladder_test_r56_pilot_verify.json`: 3 / 3 ok, joins <= 26 deg, 0 hard
  contacts; figures `ladder_test_r56_figs/`):
  - k_011_left mr_012_L (own fork k_011_right, anchor -57.3 deg): rung 5, ICA tubed 9.45 mm, anchor -> +33.1 deg;
  - w_024_right mr_002 (w_024_left, -53.3 deg): rung 5, 9.45 mm, -> +35.0 deg;
  - w_047_left mr_018_L (w_047_right, +6.2 deg): rung 5, 9.45 mm, -> +83.8 deg.
  The greedy most-rising choice takes the same template (case_w_008_right's distal ICA, a ~90 deg curl in 10 mm,
  kink <= 22 deg per mm) in all three; the 5 mm step was not enough in each. Rung 6 alone (`2 3 4 5` skipped):
  k_011 mr_012_L the same 9.45 mm with seam 73 deg / blend 10 mm / join 22; m_006_left mr_027_L at E = 0 (own fork
  k_007_left, seam 67.7 deg, join 26.6).
- Dry test over one anatomy per failing v5 left fork (`ladder_test_names.txt`, 12: every fork type of the 43,
  incl. the 8 non-anchor failures' forks; `ladder_test_r56/`, `summary_ladder_test_r56.txt`), rungs 2-4 skipped so
  only the own fork can be kept: **11 / 12 pass at rung 5** with an ICA tube of 4.97 mm (m_006_left mr_027_L,
  w_022_right mr_005_L, w_036_right mr_014_L, w_040_left mr_021) or 9.45 mm (the other 7); none needed rung 6.
  **w_029_right mr_010_L fails on rungs 5 and 6**: with its own fork (case_w_050_left) every siphon pair misses the
  ICA-spacing gate (D7) by >= 26 mm against the 20 mm limit, which an upward extension cannot correct; in the plan it
  takes a rung-2 substitute (case_w_018b_left). In the real ladder rungs 5 / 6 would also try up to 3 composed
  split-keeping substitutes.

## Split check on the planned assignment (1.2.1)

`make_split_tree.py /out/dry_v121 /out/split_tree_v121` writes v5's provenance for all 223 with the 43 planned left
forks patched in (`left.lower`, `left.lower_side`, `xform.lower_L.surface`: what v5_split reads; the route lower, the
TopBrain patient and the VA donors do not change in the carotid step). Then
`v5_split.py /out/split_tree_v121 --baseline results/split_v4.json` (`split_check_v121.{json,log}`):

- **EXIT 0: none of the 45 quantities lowered.**
- v3: roster 153 / 16 (as v4), max k16 183 / 16, k17 181 / 17 (as v4).
- v4side: roster 127 / 16, k16 174, k16_div 172, k17 172, k17_div 170 (all as v4).
- strict: roster 127 / 16, k16 174, k17 172 (as v4); k16_div6x6 168 -> 169 and k17_div6x6 167 -> 169 (raised).
- V: donors one-to-one nested (25 in 25 TopBrain patients, none spanning two).
- 1.2 (`split_check_v12.log`) and 1.1 gave the same 45 numbers.

## Pilot: the composed carotid side written (`carotid_data/v6_work/carotid/pilot/<name>/`)

`compose_v6.py pilot --plan /out/dry_v121 --names-file pilot_names.txt` (the guard replays the dry run's earlier
substitutions, so the pilot equals the dry plan: `pilot_cmp_v121.txt` 14 / 14 identical). Each folder holds
`Centrelines_comb/` (v5's bytes for unchanged branches; the changed RCCA / LCCA / RECA / LECA rewritten),
`provenance.json` (v5's, with `xform.siphon` / `siphon_L` = {R', origin = Q0, anchor, placement v6_anat, v6 pose,
blend_mm}, a substituted fork's `xform.lower_L` + `left.*` section fields, `eca_extension.*` decisions,
`left.v6_ladder_rung` / `v6_ladder_label` / `v6_tubed_ica_mm`, and `v6_carotid` {version, checks, changed files};
the VA keys stay v5's until the VA tool's merge) and `carotid_record.json` (the full decision record).

- **File-based verification (`verify_pilot.py` -> `pilot_verify_v121.json`): 14 / 14 ok** -- orient_gates tree mode on
  the written files: 0 blocking carotid flags (the only tier-A pieces left are v5 VA tops, 13, the VA tool's); 0 hard
  contacts (min 0.91 mm); 0 soft contacts; 1 mm joins <= 30.9 deg; right route identical to v5 up to seam 2 in
  14 / 14; the bake's section arithmetic within 1.38 mm; re-posed pieces sit on their TopBrain source within the
  blend (dev_max <= 1.35 mm).
- Per anatomy (`pilot_table_v121.txt`): 11 keep their left fork (rung 1); 3 substituted (rung 2): k_011_left
  mr_012_L (k_011_right -> case_w_051_right), m_006_left mr_027_L (k_007_left -> case_k_008_right), w_047_left
  mr_023 (w_047_right -> case_w_039_right). The user's images: the k_011_left mr_012_L and w_003_left mr_007_L / mr_010 left siphons now rise
  (m_006_left mr_027_L: both siphons now turn medially instead of laterally); m_006_left mr_025_L / mr_027_L pass
  on the carotid side (their VA tops are the VA tool's). w_003_left mr_007_L also shows the C6 limit of v5 fork
  headings: its right route reaches the midline on a contralateral fork, so the RICA seam sits at x = 15 mm.
- Figures, front + side (`pilot_figs/`): v5 grey / orange, v6 blue, seams x, the ladder rung in the title.

## Full plan written for all 223 (`full_plan_v121/`, for the stage-2 build)

The same writer over all 223 (`full_plan_v121_cmp.txt`: 223 / 223 identical to the dry plan) and
`verify_pilot.py` on the written files (`full_plan_v121_verify.json`): **223 / 223 ok** -- 0 blocking carotid flags
(the tier-A pieces left are 188 v5 VA tops, the VA tool's), hard clearance min 0.613 mm, 1 mm joins max 43.1 deg,
dev_max 1.58 mm, section arithmetic within 1.43 mm, right route identical to seam 2 in 223; soft contacts only the
2 RICA ones listed above.

## Open issues (for the orchestrator / stage 2)

1. **GO rule 4 (D3-fork):** by the pure map 31 % of right forks need a seam-1 bend > 45 deg (threshold 10 %), so
   the plan keeps v5 fork headings; a bounded deviation from the map (<= 10 deg yaw / 20 deg tilt) reaches 0 %, but
   it changes the RCCA training route in 223 / 223 and fails w_003_right mr_001 (right siphon). The chirality of the
   Zenodo frames is supported, not proven (leans).
2. **VA merge:** the carotid side was gated against v5's VAs as SOFT; 2 right siphons touch a v5 VA
   (`soft_contacts_v121.tsv`). The VA tool / pilot merge must re-check clearance after both VAs move.
3. **Substitutes without the old filters:** 5 of the 43 are v3 DEFECTIVE_LOWERS forks (collision failures in v3's
   graft; they pass every v6 gate here) and 11 fail the native tip test (the placed anchor passes). Reversible by
   restoring 1.1's filter-first order in `rank_substitutes`; the bake will show whether the DEFECTIVE ones mesh.
4. **Bake inputs:** the siphon blends make the first <= 6 mm of each re-posed piece leave its rigid real surface by
   <= 1.72 mm (`xform.<key>.blend_mm`); the bake must tube or morph that stretch. Rung 5 / 6 tubes are flagged
   (`left.v6_tubed_ica`) but none occur in the plan.
5. **Not done here:** the bake, wall detail (exact ownership on the RCCA), SOFA, the loader -- stage 2.

## Output layout (`carotid_data/v6_work/carotid/`)

- Final (1.2.1): `dry_v121/`, `summary_dry_v121.*`, `extra_stats_v121.txt`, `ladder_detail_v121.txt`,
  `soft_contacts_v121.tsv`, `split_tree_v121/`, `split_check_v121.*`, `pilot/` (+ `pilot_cmp_v121.txt`,
  `pilot_verify_v121.json`, `pilot_table_v121.txt`, `pilot_figs/`), `full_plan_v121/` (+ cmp, verify),
  `chain_v121.log`, markers `v121_done/`.
- Evidence / tests: `dry_map_v12/` + `summary_dry_map_v12.*` + `fork_stats_v12.txt` + `extra_stats_map_v12.txt`
  (map mode, 1.2); `ladder_test_r56/` + `summary_ladder_test_r56.*` (rungs 5 / 6 forced); `ladder_test_r56_pilot/`,
  `ladder_test_r56_figs/` (rung-5 demonstration); `tests/` (unit tests of 1.2 / 1.2.1).
- Superseded: 1.2 (`dry_v12/`, `summary_dry_v12.*`, `split_*_v12`, `extra_stats_v12.txt`, `ladder_detail_v12.txt`,
  `soft_contacts_v12.tsv`), `superseded_v1_1/`, `superseded_v1_0/`.
