# v6 to-do list

Written 2026-10-03 by tbo_report, after the TopBrain orientation audit. Sources:
- the user's decisions on the open list (`state/V5_BUILD_LOG.md`, 2026-10-03 17:25: 1 yes, 2 check, 3 yes, 4 later,
  5 deleted, 6 yes);
- `CAROTID_V5.md` → Follow-ups;
- the audit report `D:\Arjun\neve\reports\TopBrain piece orientation audit (v5).md`, with agent notes
  `notes/tbo_history.md`, `tbo_geom.md`, `tbo_cause.md`, `va_narrow.md` and `verify_tbo.md`.

Nothing here is applied. The base is v5 (`carotid_data/anatomies_v5`, commit 5c2cdc0 / 113c89a); v6 is a new tree.

## Order

Items interact, so build in this order:
1. (a1) gates and anchor checks in code;
2. (a2) left-fork rule, followed by a split re-run;
3. (a3) heading for forks and siphons;
4. (a4) VA tops;
5. (a5) ECA tips;
6. (e) the donor subclavian junction, which changes the VA lower end and re-poses the donor;
7. (b) the w_018b move, on the new VA;
8. (c) the narrow-VA follow-up, which needs a re-bake;
9. one bake, all checks, then (d) records and figures.

(e) and (a4) both re-pose the v5 donor VA, so plan them together.

## (a) TopBrain piece orientation (from the audit; user: fixes go into v6)

Root cause. The documented frame match (tangent onto tangent and superior onto superior, after `anchor_trim`) pins
only the roll. Tilt and heading are copied from the anchor tangent, which nothing checks (`graft_three.py:307-310`).
The rise-against-native check was never a gate. v5's VA tops use a Kabsch fit with no superior term.

v5 baseline:
- 141 / 223 anatomies with a tier-A piece (siphon 47, VA 115), 218 including tilt > 40°;
- v4: 171 / 189;
- classes (pieces / anatomies): L_PITCH 50 / 50, S_HEAD 22 / 16, R_PITCH 17 / 17, VA4 14 / 14, VA5 319 / 206,
  ECA 39 / 36, and C6 (fork heading at seam 1) with 30 right-siphon seams within 10 mm of the midline.

- [ ] **a1. Gates in code** (`graft_v4`-successor and post-bake checks). Every TopBrain piece:
  - rise against native: no negative rise where native > 5 mm, |error| ≤ 20 mm, siphon rise ≥ 20 mm;
  - tilt of the superior axis ≤ 40° (siphons, ECA tips; informational on VA tops);
  - heading against the anatomical map R ≈ diag(−1, −1, 1), TopBrain RAS → branch frame.

  Pair checks:
  - ICA termini in native order, spacing within 20 mm of native;
  - VA ends within native + 20 mm, not crossed > 10 mm, each ≤ 25 mm from the midline (host x 14.8 ± 6).

  Anchor checks before any frame match:
  - siphon seam tangent rises ≥ ~30°;
  - ECA anchor rises ≥ ~20°, otherwise skip.

  Move `tbo/geom_measure.py` + `geom_flags.py` into the repo as the measuring script.
  *Check:* 0 tier-A pieces on all anatomies; anchor elevation logged per seam.
- [ ] **a2. Left-fork rule (L_PITCH; images k_011_left).** Rotating the siphon cannot fix this class; the fork must
  change. Options:
  - v3 rule (ICA ≥ 58 mm, ≤ 10 mm extension, tip ≥ 2 mm, `DEFECTIVE_LOWERS`): refuses 26 left forks / 125
    anatomies, covers 34/35 tier A (verifier);
  - tip-elevation < 30° test: 63 anatomies, 27/35;
  - both: 35/35, about 133 anatomies (tbo_cause count without `DEFECTIVE_LOWERS`; re-derive).

  *Check:*
  - L_PITCH = 0;
  - seam landing re-passed (left seam 157.6 mm, ≤ 5 mm short, HOST_CUT_L 35–110);
  - `v5_split.py --baseline` re-run. The left forks are split nodes, or keep substitutes within the anatomy's
    partition.

  **Decision for the user:** which rule, given the anatomy count it costs.
- [ ] **a3. Heading pinned anatomically** for the siphons (S_HEAD, R_PITCH) and the lower forks at seam 1 (C6).
  - Anatomical frame plus a bounded seam bend (≤ 45°), blended over several mm so the 1 mm join stays ≤ 60°.
  - The forks need a landmark-based heading: the Zenodo y sign is unsettled.

  *Check:*
  - ICA-termini order error near native (batch median 40.2 → 12.2 mm with anat);
  - no right-siphon midline crossing > 15 mm;
  - clearance ≥ 0.6 after neighbours are re-placed.

  Note: the R_PITCH showcase w_035_left mr_024_L bend45 is borderline on clearance (0.39–0.56 mm).
- [ ] **a4. VA tops.**
  - VA4 (17 v4-kept sides; includes the user's m_006_left mr_025_L): move the cut back to where the host VA rises
    ≥ 30°; 10/14 pass, mr_025_L passes.
  - VA5: add a superior term to the fit or restore superior (248/319 pass), or better, aim at the basilar.
    **Decision for the user:** fuse the two tops into a basilar confluence (topology change, SOFA and navigation
    review), or stop them ≥ 0.6 mm short.
  - Re-reference `tip_move` and course RMS to the basilar point, not v4's tips.

  *Check:* VA ends within native + 20 mm, not crossed, near the midline (user image m_006_left mr_027_L).
- [ ] **a5. ECA tips** on down-pointing fork ECAs: skip the extension (the best rotation passes 13/39).
  *Check:* ECA class = 0.
- [ ] **a6. Rebuild and re-check.**
  - Order: forks → siphons → VA tops → ECA tips, with every gate.
  - Re-bake the changed pieces (handle rule, real-surface carry).
  - Re-run genus/topology, the v5 post-bake checks, SOFA loading, and the figures **with a side view** (front views
    hide heading errors).
  - Join kinks are measured on the real resampled curve: re-posed joins carry 20–45° of sampling uncertainty.
- [ ] **a7. Docs.**
  - Record that the frame match pins roll only, and the new gates.
  - Fix the stale `graft_splice` docstring (`carotid_tools/v5/graft/graft_v5_va.py:403-404`, "frame-matched").
  - Add CAROTID_V5.md's record corrections from the audit.

## (b) case_w_018b_left__topcow_mr_002: move its VA with the route13 tool (user: yes)

- **Why.** It is the one v5 anatomy below 0.99 of v4 under exact ownership (fork 0.956). k24 equals v4, so the 13-move
  missed it.
  - The loss is 121 v4 wall cells (7.85 mm²) at RCCA s 92–103 mm, near the bifurcation.
  - The donor VA centreline (ia172_136) passes 1.6–3.9 mm from that wall; v4's host VAs were 13.5–16 mm away.
  - `carotid_data/anatomies_v5/_records/swap13/diag_case_w_018b_left__topcow_mr_002.json`.
- **Tool.**
  - `carotid_tools/v5/graft/route13/`: `graft13.sh` → `graft_v5_reg_sep.py` (the S4 pose search with the RCCA-wall
    separation objective), then `VARIANT=moved_ext OWNERSHIP=standard worker13.sh` for the scratch bake.
  - Then `swap13/` (backup, swap, verify).
  - Tool version check: `carotid_data/anatomies_v5/_records/checks/route13_tool_v1_0/cmp.txt`.
- **Interaction with (a).** Both VA tops of this anatomy are flagged va_lateral in v5, so (a4) re-poses them. Run the
  move after (a4) and (e), on the v6 VA.
- **Check:**
  - exact-ownership right route ≥ 0.99 of v4 (fork and siphon);
  - all v5 gates (clearance ≥ 0.6, join ≤ 60, take-off, course RMS ≤ 20);
  - strict mesh audit;
  - SOFA loading.

## (c) Narrow VAs (user: check; done in the audit)

**Verdict.** No real narrowing that the CT supports, and no geometric block: each OBJ keeps a wire path with
0.71–0.85 mm of clearance.
- The w_025_right LVA and w_027_right RVA are a ct_seg4 bone-side segmentation artefact, then the 1.0 mm floor, then
  decimation.
- The w_025_right RVA's deepest point is 0.547 mm on the donor stretch. Its 0.632 mm point is a TopBrain label pinch
  (topcow_mr_021 left VA), the same as in v4.
- On the polyline a fourth VA is below 0.65: w_047_left mr_023 RVA, 0.637 mm, donor.

Follow-up:
- [ ] **c1.** Fix the donor segmentation's bone-side ray rule (end at the lumen plateau or half-way to the bone peak,
  or take the bone side from Arterial's mask), or floor the graft radius at the CT calibre instead of 1.0 mm.
  - Systematic: 377/429 donor VAs use floored stretches.
  - CT r_eq median 1.78 against a surface of 1.04; 83/140 are ≥ 1.5× wider on CT.
  - *Check:* re-run `tbo/va_narrow/floor_scan.py` and `bone_scan.py`, the VA lumen minimum (v5 median 0.834 against
    v4 0.978) and the wall-detail figures.
- [ ] **c2.** Measure the VA lumen on the centreline polyline (0.05–0.1 mm), not the control points
  (`carotid_tools/v5/checks/checks/va_lumen.py`).
- [ ] **c3.** Inspect the 8 CT-narrow points without bone contact:
  - ia172_015 R s 151;
  - ia172_039 R 126, L 125;
  - ia172_058 R 48;
  - ia172_071 R 87 and 89, L 110;
  - ia172_190 L 120.
- [ ] **c4.** topcow_mr_021 left VA label pinch: decide whether to leave it (v4 heritage) or tube it at about 2 mm.
- [ ] **c5.** A SOFA passage test needs a run that starts inside the VA, or (e).

## (d) Regenerate the build record (user: yes)

- [ ] `BUILD_v5.json`:
  - `donors.table` still holds assignment-time counts (14 RVAs for ia172_149, 7 LVAs for VMR 0248). Take them from
    each `provenance.json` → `v5_va.summary.replaced` (totals 211 / 218 = `results/v5_final_summary.json` →
    `counts.replaced`).
  - `checks_final.sofa.note` still says "user question pending". The user decided 2026-10-03 14:05: diversity, not a
    defect; then the junction graft for v6.
- [ ] For v6, generate `BUILD_v6.json` from the final state only, with no assignment-time copies.
- **Check:** the record equals the per-anatomy provenance on every count.

## (e) The donor's own subclavian junction (user: yes; CAROTID_V5.md Follow-ups)

- [ ] Graft a short stretch of the donor's own subclavian around the ostium: from about 10 mm after the subclavian
  origin (keeping the brachiocephalic split and the route start untouched) to about 20 mm past the VA ostium.
  - The VA then leaves at its native take-off, and the catheter meets the donor's real subclavian → VA junction.
  - Cost: one more seam on the subclavian, and a per-anatomy subclavian course.
- **Addresses:**
  - the ungated junction stub (median 5.28 mm, max 10.49 mm);
  - take-off not preserved on 251 VAs (median 17.8° off native);
  - the SOFA wire reaching ≥ 20 mm into the VA in 10/25 anatomies against 24/25 in v4. The audit's SOFA runs also
    stopped at the donor ostium.
- **Check:**
  - take-off against native;
  - seam kink and clearance;
  - the right route unchanged (RCCA is the training target);
  - SOFA VA entry on the 25-anatomy set;
  - exact-ownership route against v4.

## (f) Donor VA colour in the figures (user: later)

- [ ] The donor lower VA is drawn dark teal (`DONOR_C = "#00332e"`, `monitoring/figure_carotid_anatomies_v5.py:57`), near-black on the page. Pick a lighter distinct colour in
  `monitoring/figure_carotid_anatomies_v5.py`'s successor. Do this together with the side view from (a6).

## Done (no v6 work)

- (5) Stackdumps: deleted 2026-10-03 17:20 (user).
