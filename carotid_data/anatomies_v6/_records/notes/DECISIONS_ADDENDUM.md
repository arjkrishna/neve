# v6 decisions addendum (orchestrator; feed into stage 2 prompts)

- 2026-10-03 18:20 USER approved D1 as explained: a failing left fork is replaced by another Zenodo CarotidAnalyzer
  fork (v4's candidate order: calibre match at the siphon join, left carotids first, the anatomy's own right fork never),
  which must pass the new orientation gates, every v4 gate and keep the split (substitute from a donor already on the
  same side of the split). No per-anatomy substitute table for the user.
- Fallback where NO substitute passes: v3's left side (host LCCA tube, no left fork, no left siphon from TopBrain).
  Apply without asking; report only the count (and names in the records).
- 2026-10-03 18:30 USER: no more than 3-4 fallback cases (host LCCA tube, no left fork). Before falling back, escalate
  per anatomy: (1) re-pose the left siphon on its own fork (anatomical heading, blended bend <= 45 deg, 1 mm join <= 60);
  (2) substitute from a donor already in the anatomy's split group/side; (3) any substitute for which the full split
  check (v5_split.py --baseline split_v4.json) still exits 0 with none of the 45 quantities lowered; (4) substitute
  allowing the documented ICA extension (<= 10 mm) / anchor trim. Fallback only after (1)-(4). If more than 4
  anatomies would still fall back, STOP that step and report to the user with the options (do not exceed 4).
- 2026-10-03 18:35 USER: "more than 4 -> use your best judgement; just don't do host". SUPERSEDES the host fallback:
  NO anatomy falls back to the host LCCA tube, and nothing stops to ask. After the ladder (1)-(4), the last resort is
  (5) keep the fork (own or the best substitute that keeps the split) and EXTEND its ICA upward with extend_ica
  (up axis +z, the v3 set-B method) beyond 10 mm until the anchor tangent rises enough for the siphon to pass the
  orientation gates; then (6) if still needed, use the full bend allowance (1 mm join <= 60 deg). Flag every (5)/(6)
  anatomy in provenance (tubed ICA mm) and report the count. The split is never lowered.

## Stage-1 verifier NO-GO items -> orchestrator decisions (2026-10-04 07:40; user standing rule: don't wait, best judgement; all reversible)
- D8 va_target: <= 20 mm passes; 20-30 mm accepted with a flag; > 30 mm goes to the D9 ladder.
- D9 VA fallback ladder (per failing side, never drop a donor VA, one-to-one kept): (1) donor junction + anatomical top
  with the full pose search; (2) v5's junction + anatomical top; (3) v5's top as shipped, flagged. Count per rung.
  Course RMS (<= 20 mm vs the v4 corridor) stays for the DONOR stretch; the TopBrain top is referenced to its
  anatomical target (D2-target) instead.
- D10 VA corridor gate (new, in the grafter and post-bake): over z 440-540 the placed donor VA's lateral offset from
  the host neck midline must be within +-8 mm of the donor's NATIVE lateral offset from its own spinal-canal midline
  (measure it in the donor CT; if not measurable, no more than 8 mm medial of the host VA at the same z). Failing sides
  go down the D9 ladder.
- D11 ICA termini front-back: the RIGHT fork (RCCA = training route) keeps v5's pose -- the training route is not
  re-posed; its ICA-terminus AP offset is recorded per anatomy and flagged when > 40 mm. LEFT forks (not the training
  route): re-pose by the per-patient Zenodo map where the seam-1 bend <= 45 deg and every gate passes, else keep v5's
  pose (flag). Report counts.
- D12 ica_spacing convention: like for like -- native = the back-mapped ends of the same pieces (not raw termini).
- D13 right-route baseline: where v4's RECA TopBrain tip ran along the ICA and v6 re-poses or skips that tip, the
  reference is v4 rebuilt without that tip (v4_no_tip.py), recorded per anatomy; run it on every such anatomy.
- D14 SOFA VA entry (item e): >= 3 seeds per anatomy plus a geometric entry test; reported vs v5 with the same seeds;
  not a pass gate (as in v5). Drop the ">= 20/25" target.
- D15 donor wall (D5 priority, the user's handle-rule philosophy "fix only where needed"): apply the plateau fix ONLY in
  the artefact zones (bone-contact / floored stretches + a short blend); keep v5's coarse surface elsewhere. Gate:
  wall detail outside the fixed zones identical to v5; report per donor. (Stage 1's global plateau lowered detail
  0.133 -> 0.110 and 0.118 -> 0.090 away from the artefact; 6 sides <= 0.51 of v5.)
- Plus every stage2_changes item of verify_v6_pilot (tool fixes adopted in the tools, joint pass, guards, bake fixes
  E2/E3/E3b/E6/E7, sofa_rva_v6, dry runs + split re-run on all 223, c2 on every OBJ, c4 on all 8 mr_021_L anatomies,
  BUILD_v6 flags per anatomy).

## USER decisions 2026-10-04 (after the stage-2 limit stop) -- binding, supersede D11's right-fork part
- D16 (user: yes) pinched donor VA ostia (ia172_164 L in 16 anatomies, ia172_071 L in 4; ostium bottleneck < 0.5 of
  the VA radius): bridge the pinch with a short tube at the VA calibre (as already done for ia172_136 L / ia172_172 L)
  so these sides get the donor's own junction. Flag the tubed mm.
- D17 (user: yes) arm side of the subclavian: NO seam-2 bridge back to the host's distal subclavian. The donor's own
  subclavian continues toward the arm as the (17)/(19) branch to the end of the donor's segmentation and ends as a cut
  end (same end treatment as other cut vessels); if the label covers < ~30 mm past the ostium, extend with a short tube
  at donor calibre along its last direction (flag mm). The host's distal subclavian beyond the donor stretch is dropped
  on donor-junction sides. Gates: clearance 0.6 mm, 1 mm joins <= 60, no hairpin; the loader's branch set unchanged.
- D18 (user: yes) the remaining donor-junction failures (course RMS / va_target / clearance) are re-tried after D16/D17;
  sides still failing keep v5's junction (flagged) -- no other relaxations.
- D19 (user: yes) RIGHT forks re-posed to their anatomical heading (per-patient Zenodo map, I or Rz(180), from
  scratchpad/v6/heading/zenodo_frames.json) -- the RCCA training route may move. Seam-1 bend <= 45 deg blended (1 mm
  join <= 60); where > 45 is needed (~55 anatomies): search the host cut along the RCCA within the documented seam
  limits, then allow up to 60 deg blended; flag. The RCCA prefix (arch -> seam 1) stays host.
- D20 (user: 4a) forks taken from the patient's OTHER side (113 right routes; "_left" fork names on the right route)
  are x-MIRRORED before placement so each is a true right carotid -- the same treatment their TopBrain siphons already
  get (mirrored surfaces: flip triangle winding; record chirality in provenance). Orchestrator extends the same rule to
  contralateral LEFT forks (101) for consistency (reversible). Then D11's left re-pose applies.
- Route invariant redefined for re-posed forks: each real section keeps >= 0.99 of its own v4 wall detail under exact
  ownership (same surface, moved); right route navigable; SOFA RCCA pass on ALL 223 (the training route moved).
  ICA-terminus AP offset: report per anatomy, flag > 40 mm (target: far fewer than v5's 108).
- D21 (user: later) the 17 VA sides with no donor since v5: NOT in this build.
- Scope (user): finish the v6 anatomies and images; no further additions.

## USER decisions 2026-10-06 (after the stage-2 gate NO-GO) -- binding
- D22 (user: 1A) RIGHT-ROUTE CHECK for moved pieces = "surface preserved": for the right fork and the right siphon,
  >= 99 % of the piece's real-surface area beyond the seam-1 blend (siphon: beyond the seam-2 zone) lies within 0.3 mm
  of v4's surface carried by that piece's own transform (mirror included). The blend zones and bake handle windows are
  REPORTED (area, max deviation), not gated. Replaces the 0.99 relief-ratio gate post-bake AND the composer's pre-bake
  route guard C0 (which pushed 28 anatomies off their anatomical fork pose on the noisy ratio). The relief ratio is
  still reported for information. The RCCA prefix (arch -> seam 1) stays bit-identical.
- D23 (user: 2A) w_017_left mr_002_L right VA on D9 rung 3 (v5's top, orientation tier A): keep the donor VA, accept
  and flag (1/223). The "0 tier-A" invariant reads "0 tier-A except flagged D23".
- D24 (user: 3 yes) right forks on R4 (bounded deviation) / R5 (v5 pose) are accepted with flags; re-evaluate every
  one under D22 (some should now reach the anatomical pose).
- D25 (orchestrator, verifier's must-fix): D17 direction gate (the donor distal subclavian's cut end lies lateral of
  its ostium and its direction is < 90 deg from the host's distal subclavian; failing sides -> D18 v5 junction; known:
  ia172_069 R in m_006_left mr_025_L and w_051_right mr_025_L); 1 mm join gate evaluated at EVERY 1 mm sampling phase
  on all new seams (10 known exceedances: (19) 61.1 deg on 8 mr_014_L, (18) 64.5 deg on 2 mr_025_L) -> fix by blend
  or fall back; records: fill v6_build.va.<side>.donor_id on rung-2/3 sides, correct the tool-version notes; ONE frozen
  version of every tool for the whole build (re-run the dry run on all 223 with it).

## USER decision 2026-10-07 15:00 -- binding
- D26 (user: "A") D22 reference where v4 itself hid part of a piece's real surface: for the moved right fork / siphon,
  if v4's mesh did not expose part of the piece's own scanned surface (e.g. a mirrored contralateral fork whose own
  lumen bulge sat inside a neighbour in v4), the D22 check compares against the piece's OWN scanned (source) surface
  carried by its transform instead of v4's mesh: >= 99 % of the real surface beyond the blend within 0.3 mm of it.
  Applies to the 9 R-type failures (w_012_left mr_020 / mr_021_L; w_033_left mr_007 / 010 / 011_L / 016 / 024;
  w_040_left mr_021; w_048_left mr_008; value_own >= 0.996) and any other anatomy where the same condition holds.
  Each such anatomy is FLAGGED (d22_reference = own_surface) and reported; no real wall may be lost.

## USER rule 2026-10-07 16:25 -- binding for the rest of the build
- D27 (user: "just do the recommended and then report/flag it"): no stopping for choices. Residual cases get the
  orchestrator's recommended minimal option + a flag. k_011_left mr_012_L siphon (own-surface 0.9898; 17.79 mm2 on the
  bake's repair capsule) = bake repair window -> reported not gated (D22) -> pass, flagged. Other residuals: next ladder
  rung; else accept + flag if deficit < 1 % of the piece and lumen >= 0.65 / clearance >= 0.6; else v5 geometry for that
  piece, flagged. All 223 must end in anatomies_v6.
