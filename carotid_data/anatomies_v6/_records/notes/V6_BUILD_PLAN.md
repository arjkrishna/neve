# v6 build plan

Written 2026-10-03 by `v6_plan` (stage 1 of the v6 workflow). Base: v5 = `carotid_data/anatomies_v5`, commit 5c2cdc0
(113c89a), `CAROTID_V5.md`, build log `v5_va/state/V5_BUILD_LOG.md`, toolchain `carotid_tools/v5` (MANIFEST.md).
Spec: `v5_va/v6/V6_ITEMS.md` and the audit `reports/TopBrain piece orientation audit (v5).md`.

v6 is a **new tree**, `carotid_data/anatomies_v6`. Nothing under `anatomies_v2..v5`, `topbrain_tools/`,
`carotid_tools/` (incl. `carotid_tools/v5`) is edited: v6 code imports or copies v4/v5 functions. New code lives in the
scratchpad `v6\<area>\` (`/sp/v6/<area>` in containers), large outputs in `carotid_data/v6_work/<area>/` (git-ignored).
No git writes; `user.md` / `user copy.md` and the repo docs are never edited.

Paths below: `/sp/v6/...` = `C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\...`.

## 0. Summary

- **Stage 1 (this workflow): tools and a 14-anatomy pilot.** Done by `v6_plan`: this plan, the orientation-gate
  library (self-test passes on v5 / v4 / v3), the heading maps settled with evidence (fork sign included), and the
  pilot set. The other stage-1 agents write the placement tools (T1-T9 below) and run them on the pilot, then a
  GO / NO-GO review ([section 8](#8-go--no-go-for-stage-2)).
- **Stage 2 (a later workflow): the full build** of all 223 anatomies into `anatomies_v6`, one bake, every check,
  the split, SOFA, figures with a side view, `BUILD_v6.json` from the final state only, `CAROTID_V6.md`, the
  toolchain copy `carotid_tools/v6`, and an independent final verifier.
- **What changes:** every TopBrain piece is posed anatomically (superior and heading pinned) with a bounded seam
  bend; left forks that fail the new gates are substituted; ECA tips on down-pointing ECAs are skipped; the VA
  tops are posed toward the basilar target and stopped >= 0.6 mm short of each other; the donor VA brings its own
  subclavian junction; the donor wall floor follows the CT calibre; the w_018b VA is moved with the route13 tool.
  The lower forks' seam-1 heading is now settled per Zenodo patient and is pinned too (decision D3-fork,
  reversible, pilot-gated).
- **v5 against the v6 gates** (`gates/out/selftest.txt`): every anatomy fails somewhere (heading on 604 pieces,
  ICA spacing in 143 anatomies, VA gap 104, VA target 331 of 446 tops, anchor 68 seams); tier A in 141. v6 must
  bring tier A and every v6 gate to 0 on all 223.

## 1. Items and invariants

| item | what | v5 baseline | v6 target |
|---|---|---|---|
| a1 | orientation gates in code (pieces, pairs, anchors, forks) | none; tier A 141 / 223 | `orient_gates.py` at build time (API) and post-bake (tree); 0 tier A, 0 v6-blocking |
| a2 | left-fork rule (L_PITCH) | 50 pieces / 50 anatomies (35 tier A) | L_PITCH 0; substitutes keep the split |
| a3 | heading pinned: siphons (S_HEAD, R_PITCH) and forks (C6) | S_HEAD 22/16, R_PITCH 17/17; 30 RICA seams < 10 mm from the midline | anatomical + bend <= 45 |
| a4 | VA tops (VA4 kept-v4 14/14, VA5 fitted 319/206) | tops 26.6 mm apart (median), crossed up to 34.8 mm | anatomical, at the basilar target, stopped >= 0.6 mm short |
| a5 | ECA tips on down-pointing ECAs | 39 / 36 | skipped, recorded |
| a6/a7 | rebuild, re-check, docs | - | one bake, all checks, side-view figures, CAROTID_V6.md |
| b | w_018b_left mr_002 VA moved with the route13 tool | exact-ownership fork 0.956 of v4 | >= 0.99 |
| c | narrow VAs (ct_seg4 bone-side artefact + 1.0 mm floor) | 377/429 donor VAs floored; 4 VAs < 0.65 mm on the polyline | floor at the CT calibre; polyline lumen >= 0.65 |
| d | BUILD_v6.json from the final state only | BUILD_v5.json carried assignment-time counts | record == provenance on every count |
| e | the donor's own subclavian junction | stub median 5.28 / max 10.49 mm; take-off off native median 17.8 deg; SOFA VA entry 10/25 | donor junction, take-off native, entry >= v4-like |
| f | donor colour in figures | dark teal | LATER (not in v6 unless asked); the side view is in |

**Invariants v6 must keep (all gated, every anatomy):** genus 0; 0 self-intersections; closed; centrelines inside;
right route (RCCA, the training target) >= 0.99 of v4 wall detail under exact ownership (fork and siphon);
clearance >= 0.6 mm; joins <= 60 deg (1 mm, on the resampled curve); course RMS <= 20 mm; one-to-one donor units
onto the 25 TopBrain patients; the split not lowered (`v5_split.py --baseline results/split_v4.json` exit 0, none of
the 45 quantities lower); 223/223 load in the training loader; SOFA passes; the new orientation gates (a1) at 0
tier-A pieces (and 0 v6-blocking flags); coarse walls kept (D5).

## 2. Decisions (all reversible; how to reverse is given)

| id | decision | evidence | how to reverse |
|---|---|---|---|
| **D1** left forks | No blanket rule. Keep each anatomy's left fork if its left siphon passes every new gate under the v6 fork pose (D3-fork) and the v6 siphon pose (D3); otherwise substitute the best passing fork. v3's rule (`match_sections.py:71-97` + `DEFECTIVE_LOWERS`) and the tip test (< 30 deg) are candidate filters, not refusals. Substitutes must keep the split (exit 0, none of the 45 quantities lowered; v3 roster 153/16; best 174/16 or 172/17); prefer forks from the same donor group / partition. | 35 LICA seams rise < 10 deg (all of the anchor_low class); rotation cannot fix them (audit sec. 5); v3 rule would cost 125 anatomies, the tip test 63 | switch the candidate filter to v3's rule or the tip test (`left_fork_rule` option in T2) |
| **D2** VA tops | No basilar fusion (genus 0, topology and navigation unchanged). Pose each top anatomically (superior + heading by A0), aim its end at the anatomical target (the patient's two native VA ends centred on the host confluence C_HOST = (14.8, 12.5, 572.0) through A0), keep the native separation, then stop both tops short until the lumen gap is >= 0.6 mm. Re-reference `tip_move` and course RMS to that target, not v4's tips: gate `va_target` <= 20 mm (D2-target). | anat tops meet at the midline 3.6 mm apart but touch (gap 0.03 mm, `gates/out/test_api.txt`); v5 tops end 27.6 mm (median) from the target, > 20 mm in 331/446 | fuse into a confluence (topology change + SOFA/navigation review); or reference the target to the siphon termini instead of C_HOST; or relax va_target to 30 mm |
| **D3** heading | Anatomical map for siphons and VA tops (A0 = diag(-1,-1,1), settled: `heading/heading_evidence.txt`) with a bounded seam bend <= 45 deg, blended over several mm (>= 5 mm, tuned in T3) so the 1 mm join stays <= 60 deg. The bend direction is chosen to keep up_tilt <= 40 (the tilt gate binds before the 45 deg bound on a vertical bend). | anat restores rise exactly; batch: ICA-termini order error 40.2 -> 12.2 mm; S_HEAD 12/22 pass under anat alone | frame match with untilt (pitch-only fix) |
| **D3-fork** | The fork sign IS settled with evidence (below), so v6 also poses the lower forks at seam 1 by the per-patient Zenodo map (I for LPS-like scanner frames, Rz(180) for RAS-like; table in `orient_gates.ZENODO_FRAMES`) with the same bounded-bend rule. Contralateral forks (113 right, 101 left: forks are never mirrored) take the map, which keeps anterior-posterior and mirrors medial-lateral. Pilot-gated: if the pilot fails the RCCA invariants or > 10 % of right forks need a seam-1 bend > 45 deg that cannot be blended, fall back to v5's fork heading and record it. | per-patient frames (61 bilateral patients; one-sided rule 121/122); x and y flip together (112/122); ECA anterior of ICA 91/138 under the proper reading; seam-1 bend <= 45: right 168/223, left 216/223 | keep v5 fork headings (`--fork-heading v5` in T1); `orient_gates` fork gates then run informational (`fork_gates=False`) |
| **D4** ECA tips | On a fork ECA whose anchor rises < 20 deg and has no passing pose: skip the extension, record it (`eca_extension.<side>.why`). | best rotation passes 13/39; k_004 RECA's rising anchor 12 mm back fails clearance | re-enable the extension with an anchor search (atrim) |
| **D5** walls | The wall stays COARSE: real, irregular, non-tube (wall detail = share of wall > 0.3 mm off the tube; v5 donor stretch median 0.134, whole VA 0.128 vs v4 0.018). Any wall fix (c1, e) keeps it; re-measured on undecimated unions under exact ownership. | user's v5 goal | - |
| D6 (new) | `anchor_low` blocks the LICA seam and the ECA tips, but is informational on the RICA: the right fork is the training route and is never swapped; the right siphon is fixed by D3 (3 RICA seams < 30 deg, min 25.8). | `gates/out/selftest.txt` anchor elevation | make it blocking on the RICA too |
| D7 (new) | ICA-termini spacing gate is two-sided (`ica_spacing`: lateral spacing within 20 mm of native, the V6_ITEMS a1 wording); the audit's one-sided `ica_converge` / `ica_splay` stay as tier A / secondary. | a1 spec | one-sided only |

**Why the fork sign is settled (D3-fork).** The audit could not settle "the Zenodo scanner frame's y sign" because
there is no single frame: the left-right axis differs between patients (77 patients: LPS-like 29, RAS-like 48; group m all
LPS-like, w mostly RAS-like, k half and half). The two sides of one patient fix the x axis (61 patients); the
bifurcation x sign about the scanner origin fixes it for the 16 one-sided patients (agrees 121/122 where both are
known); the CCA's y-lean flips with x in 112/122, so the frames differ by a 180 deg turn about z, not a mirror; and
under the proper reading the ECA sits anterior of the ICA in 91/138 forks (the textbook relation; TopBrain agrees at
its level). A rotation-only placement can only use I or Rz(180), and the settled x axis picks which. Full numbers:
`heading/heading_evidence.txt`, `notes/v6_plan.md`.

## 3. The gates (a1; done)

`/sp/v6/gates/orient_gates.py` (numpy only). Built from the audit's `geom_measure.py`, `geom_flags.py`,
`classify.py`, `repro.py` helpers and `vmeasure.py`.

- **Tree mode** on any anatomy tree (provenance `xform.<key>` {R, origin, anchor, mirror, centerline_src} +
  `Centrelines_comb`): `python3 orient_gates.py tree --root <tree> --out <dir> [--fork-gates]` ->
  `orient_<tag>.json`, `pieces_<tag>.tsv`; exit 1 if any tier-A piece.
- **API** for the grafters, before baking: `measure_candidate(piece, Q, R, anchor, K, ...)`, `pair_metrics`,
  `evaluate(..., gateset="v6")`, `anchor_ok(piece, K)`, `fork_metrics(R_lower, lower, side, ...)`, `va_targets`,
  helpers `yaw_of`, `up_tilt`, `bend`, `untilt`, `anat_place`. Round trip against tree mode: 4.9e-13
  (`gates/out/test_api.txt`).
- **Gate sets.** `audit` = the audit exactly (TIER_A, PRIMARY = + up_tilt > 40, SECONDARY). `v6` = TIER_A +
  up_tilt_v6 (siphons, ECA tips; VA tops reported), heading |yaw| > 45 (siphons, VA tops), ica_spacing (20 mm),
  va_gap_v6 (native + 20), va_touch (< 0.6 mm), va_target (> 20 mm), anchor_low (LICA < 30, ECA < 20); forks
  (`--fork-gates`): fork_heading (> 45 vs the Zenodo map), fork_tilt (> 40). Thresholds in `TH`.
- **Self-test** (`orient_gates.py selftest`, PASS): v5 141 tier A (siphon 47, VA 115) / 218 primary; v4 171 / 189;
  v3 11 / 25; every class count (L_PITCH 50/50 (35 tier A), S_HEAD 22/16, R_PITCH 17/17, VA4 14/14, VA5 319/206
  (174/109), ECA 39/36 (16/15); v4 VA4 269/165); every per-rule piece count equal to the audit's `counts.json`; all
  997 pieces numerically identical to `geom_v5.json` (worst difference 0.0) and 0 flag mismatches; the same for v4
  (997) and v3 (223).
- **Where they run in v6:** inside every placement tool (reject a candidate pose before baking), in the dry runs on
  all 223, and post-bake on `anatomies_v6` (tree mode, with `--fork-gates` if D3-fork holds). The script is copied
  into `carotid_tools/v6/` in stage 2 (V6_ITEMS a1: "move into the repo").

## 4. Build order

From V6_ITEMS (Order), refined by the heading evidence:

1. **a1 gates** (done) and the anchor checks in every tool.
2. **D3-fork: seam-1 fork poses** (right and left). They set every ICA-tip anchor, so they come before the left-fork
   rule and the siphons.
3. **a2 left-fork rule** under the v6 fork pose (anchor >= 30 deg, then the siphon test), substitutions, seam landing,
   then a split dry run (`v5_split.py --baseline`) on the projected provenance.
4. **a3 siphons** (right, then left): anatomical + bounded bend, blended; neighbours re-checked.
5. **a4 + e VA, planned together:** the donor's subclavian stretch and own junction (e) re-pose the donor VA from the
   ostium; the TopBrain top is then posed anatomically at the splice, aimed at the D2 target, stopped short. The 17
   v4-kept sides get the cut moved back to a >= 30 deg rise (atrim) or an anatomical re-pose.
6. **a5 ECA tips:** extend only on a rising anchor with a passing pose; else skip (D4).
7. **b** the w_018b VA move with the route13 tool, on the v6 VA.
8. **c** narrow VAs: the c1 wall / floor fix enters at the donor-wall step of 5 (it changes the donor surfaces and
   radii before the bake); c2-c4 are checks and one decision.
9. **One bake** (v5 handle rule, real-surface carry), every check, figures with a side view, then **d** the records.

Every step re-runs the orientation gates, the clearance gate (0.6 mm against every placed neighbour) and the 1 mm
join gate on the resampled curve, because a re-posed piece moves against neighbours placed for the old pose.

## 5. Stage 1: tools and the pilot

Pilot: `/sp/v6/pilot_set.json`, 14 anatomies (8 mirrored, 6 native) with the reason for each:

| anatomy | why |
|---|---|
| case_k_011_left__topcow_mr_012_L | L_PITCH, user image 1, D1 substitution (left fork k_011_right, seam -57 deg) |
| case_w_003_left__topcow_mr_010 | S_HEAD + C6 (RICA seam at the midline), user image 3; LVA v4-kept |
| case_w_003_left__topcow_mr_007_L | the VMR 0248 donor anatomy; S_HEAD, user image 3 |
| case_w_035_left__topcow_mr_024_L | R_PITCH showcase (bend45 borderline clearance) |
| case_m_006_left__topcow_mr_025_L | VA4, user image 4a; RVA v4-kept (no donor) |
| case_m_006_left__topcow_mr_027_L | VA5, user image 4b; D2 stop-short; c3 donor ia172_058 |
| case_k_004_left__topcow_mr_010 | ECA (RECA -75 deg), D4 skip |
| case_w_018b_left__topcow_mr_002 | item b (route13 move) |
| case_w_025_right__topcow_mr_021_L | narrow LVA 0.567 / RVA 0.547 + TopBrain label pinch 0.632 (c4); L_PITCH, R_PITCH |
| case_w_027_right__topcow_mr_022 | narrow RVA 0.630; ECA (LECA); C6 seam 8.8 mm |
| case_w_047_left__topcow_mr_023 | 4th narrow VA (0.637, polyline); c3 donor ia172_190; L_PITCH -> second D1 case |
| case_w_050_left__topcow_mr_004_L | a route13-moved anatomy; VA5; c3 donor ia172_071 |
| case_w_051_right__topcow_mr_008 | clean control (no audit flag), native |
| case_w_034_right__topcow_mr_017_L | clean control, mirrored |

Tools (each in `/sp/v6/<area>/`, each with a `--dry` mode that runs the placement and every gate on any anatomy
list without writing a tree, per-anatomy done markers and logs on disk; at most 2 containers per agent, `--cpus 4
--memory 6g`; GPU jobs take `/sp/v6/gpu.lock` by atomic mkdir):

| tool | area | does | built from | pilot acceptance |
|---|---|---|---|---|
| T1 fork pose | `forks/` | seam-1 pose = Zenodo map (per patient) + bounded bend (<= 45, blended on the CCA) onto the host CCA tangent; ICA/ECA carried by the same R; `--fork-heading v5` fallback | `graft_three.compose` seam 1, `orient_gates.fork_map` | fork_heading / fork_tilt 0; seam-1 1 mm kink <= 60 (and v3's seam-1 gate); RCCA exact-ownership >= 0.99 of v4; route length change reported |
| T2 left-fork rule | `forks/` | evaluates each anatomy's left fork under T1 + T3 (anchor >= 30, siphon gates, seam landing 157.6 mm <= 5 mm short, HOST_CUT_L 35-110, clearance); ranks substitutes (same donor group / partition first; candidate filters: v3 rule, tip test); projects the provenance and runs `v5_split.py --baseline` | `graft_v4.lower_records`, `left_lower_candidates`, `match_sections` | k_011 mr_012_L and w_047 mr_023 get passing forks; the dry run on all 223 lists every substitution; split exit 0 on the projection |
| T3 siphon placer | `siphons/` | A0 pose at the seam + minimal bend (<= 45, direction chosen for up_tilt <= 40) blended over >= 5 mm; gates via the API; neighbours re-checked | `repro.bend`, `graft_three.place`, `graft_v4.join_kink` | all siphons: 0 tier A, heading <= 45, up_tilt <= 40, 1 mm join <= 60 on the resampled curve, clearance >= 0.6, ICA spacing within 20 mm |
| T4 donor subclavian + wall | `subclavian/` | per donor side: trace the donor's own subclavian from origin + ~10 mm to ostium + ~20 mm (TotalSegmentator subclavian / brachiocephalic masks or the Arterial graph; VMR 0248 from its model), CT-edge wall (ct_seg4 style) with the c1 bone-side fix; FOV check per donor | `pilot_proc_*_tssplit.py`, `scaleup_ct_seg.py`, `vmr_*` | every pilot donor side (ia172, Song, VMR 0248) has a stretch or a recorded shortfall; wall detail median >= 0.10 (D5) |
| T5 VA re-pose | `va/` | donor subclavian stretch + VA posed on the host subclavian corridor: proximal seam ~10 mm after the subclavian origin (brachiocephalic split and route start untouched), distal seam at ostium + ~20 mm back onto the host subclavian (net +1 seam vs v5's junction stub); TopBrain top at the splice by A0, aimed at the D2 target, stopped >= 0.6 mm short; 17 v4-kept sides: atrim or A0 | `graft_v5_reg.py` (pose search, gates), `graft_v5_va.py`, `orient_gates.va_targets` | take-off within ~10 deg of native (median); subclavian seams kink <= 60, clearance >= 0.6; course RMS <= 20; va_target <= 20; VA-VA gap >= 0.6; heading <= 45 |
| T6 ECA rule | `eca/` | extend only if anchor >= 20 deg and a pose passes; else skip and record | `graft_v4.extend_eca` | ECA class 0 on the pilot |
| T7 w_018b move | `route13/` | the route13 tool (`graft13.sh` -> `graft_v5_reg_sep.py`, `VARIANT=moved_ext OWNERSHIP=standard worker13.sh`) on the T5 VA | `carotid_tools/v5/graft/route13/` (version check `route13_tool_v1_0/cmp.txt`) | exact-ownership fork >= 0.99 of v4; v5 gates; strict mesh audit; SOFA load |
| T8 narrow VA | `va_narrow/` | c1: the bone-side ray rule (stop at the lumen plateau or half-way to the bone peak, or take the bone side from Arterial's mask) or a floor at the CT calibre instead of 1.0 mm; c2: polyline lumen (0.05-0.1 mm); c3: the 8 CT-narrow points inspected with figures; c4: the mr_021 label pinch left (v4 heritage) unless tubed trivially at the neighbouring calibre (~2 mm), recorded either way | `tbo/va_narrow/floor_scan.py`, `bone_scan.py`, `checks/va_lumen.py` | the 4 narrow VAs >= 0.65 mm on the polyline (or the c4 pinch recorded); floored stretches wider than 1.5x on CT -> ~0; wall detail kept |
| T9 bake + checks + figures | `bake/`, `checks/`, `figs/` | v5 bake (handle rule, real-surface carry) of the changed pieces; strict audit (genus, SI, closed, centrelines inside), exact-ownership wall detail (right route and VA, undecimated unions), clearance, joins, course RMS, loader, SOFA (RVA entry, `checks/sofa_rva.py`), orientation gates tree mode, figures front + side | `carotid_tools/v5/bake`, `checks`, `monitoring/figure_carotid_anatomies_v5.py` successor | every invariant on all 14; SOFA VA entry >= 20 mm in more pilot donor RVAs than v5 |
| T10 pilot review | `pilot/` | one table per anatomy: v5 vs v6, every gate, every invariant; figures | - | the GO / NO-GO of section 8 |

## 6. Stage 2: the full build

1. **Freeze** the stage-1 tools (version strings in every provenance entry); `orient_gates.py` thresholds frozen.
2. **Dry runs on all 223** for T1-T6 (no tree written): pass counts per gate; every anatomy without a passing
   configuration gets a fallback from the decisions (substitute fork, skip ECA, v5 fork heading) or blocks the build.
3. **Build** `carotid_data/anatomies_v6` in the section-4 order with detached, idempotent per-anatomy workers (done
   markers, logs on disk), <= 2 containers per agent, RAM checked first.
4. **Bake once** (handle rule, real-surface carry), then the checks: strict audit, exact-ownership wall detail
   (right route vs v4, fork and siphon; VA), clearance, joins, course RMS, take-off, VA lumen (polyline), the
   orientation gates (tree, `--fork-gates`), the split (`v5_split.py anatomies_v6 --baseline results/split_v4.json`),
   loader 223/223, SOFA (load on all, RVA entry on the 25-anatomy set), `anatomies_v5` untouched.
5. **Figures** for all 223 with a front and a side view (front views hide heading errors); donor colour unchanged
   (item f is later).
6. **Records:** `BUILD_v6.json` generated from the final per-anatomy provenance only (no assignment-time copies;
   check record == provenance on every count); `CAROTID_V6.md` (what changed, the gates, the frame-match note "pins
   roll only", the v5 record corrections from the audit, the fork-frame table, the stale `graft_splice` docstring
   fixed in the v6 copy); `carotid_tools/v6/` toolchain copy with `MANIFEST.md`, `FILES.tsv`, `restore_layout.sh`
   (as for v5).
7. **Final verifier:** an independent agent re-measures with its own code (as verify_tbo did) every invariant and
   the orientation gates on all 223, and spot-checks 20 figures.

## 7. Acceptance criteria per item

| item | accepted when (all 223 unless stated) |
|---|---|
| a1 | `orient_gates.py tree --fork-gates` on anatomies_v6: 0 tier-A pieces, 0 v6-blocking flags; anchor elevation logged per seam; self-test still PASS on v5/v4/v3 |
| a2 | L_PITCH = 0; every LICA seam rises >= 30 deg; seam landing re-passed (<= 5 mm short of 157.6 mm, HOST_CUT_L 35-110); `v5_split.py --baseline` exit 0 with none of the 45 quantities lowered; substitutions listed per anatomy with the reason |
| a3 | siphons: heading <= 45, up_tilt <= 40, rise gates, no RICA crossing > 15 mm, ICA spacing within 20 mm of native; seam bend <= 45 blended, 1 mm join <= 60 on the resampled curve; clearance >= 0.6; forks (D3-fork): fork heading (yaw against the Zenodo map) within 45 deg, fork tilt <= 40, seam-1 kink gates |
| a4 | VA tops: 0 va_cross / va_lateral / va_gap; va_target <= 20 mm; VA-VA lumen gap >= 0.6; heading <= 45; tilt reported; joins and clearance; course RMS <= 20 |
| a5 | ECA class 0; every skipped extension recorded with the anchor elevation |
| a6 | every invariant of section 1 |
| a7 | CAROTID_V6.md has the gate section, the frame-match note and the record corrections |
| b | w_018b exact-ownership right route >= 0.99 of v4 (fork and siphon); v5 gates; strict audit; SOFA load |
| c | polyline VA lumen >= 0.65 mm in every VA except a recorded c4 pinch; floored stretches re-scanned (`floor_scan`, `bone_scan`); c3 verdicts with figures; c4 decision recorded; donor wall detail median >= 0.10 (D5) |
| d | BUILD_v6.json == provenance on every count (checked by script) |
| e | every donor VA leaves at its own junction (no synthetic stub); take-off vs native median <= 10 deg; subclavian seams kink <= 60, clearance >= 0.6; RCCA exact-ownership >= 0.99 of v4; SOFA wire >= 20 mm into the VA in >= 20/25 (v4 24/25, v5 10/25); subclavian wall coarse (D5) |

## 8. GO / NO-GO for stage 2

**GO** only if all of these hold on the pilot (14 anatomies) and in the all-223 dry runs:

1. **Gates:** every pilot anatomy has 0 tier-A pieces and 0 v6-blocking flags after T1-T9.
2. **Invariants:** every pilot anatomy passes every invariant of section 1 after the bake.
3. **D1:** both L_PITCH pilots get a passing left fork; the T2 dry run on all 223 finds a passing fork for every
   anatomy, and the projected split is not lowered (exit 0).
4. **D3-fork:** the right-route RCCA keeps >= 0.99 of v4 wall detail under exact ownership on every pilot, and the
   T1 dry run leaves <= 10 % of right forks needing a bend > 45 deg. Otherwise NO-GO **for the fork re-pose only**:
   stage 2 proceeds with v5 fork headings (recorded; fork gates informational). Note: at v5's own host cuts,
   55/223 right forks (25 %) and 7/223 left forks would need > 45 deg (`heading/heading_evidence.txt`, part D), so
   T1 must lower this (a host-cut search along the CCA, a longer blend) for the fork re-pose to go ahead.
5. **D2:** every pilot VA pair: va_target <= 20 mm and gap >= 0.6 mm; the T5 dry run reaches va_target <= 20 on
   >= 95 % of the 446 tops. If not, revisit D2-target (siphon-termini reference or 30 mm) before stage 2.
6. **e:** at least one donor of each source (ia172, Song, VMR 0248) grafted with its own junction; every pilot donor
   side either has its subclavian stretch or a recorded, gated fallback; SOFA VA entry on the pilot better than v5.
7. **c:** the 4 narrow VAs pass (or the c4 pinch is recorded); D5 wall detail kept.
8. **Dry runs:** every anatomy of the 223 has a passing configuration in T1-T6 (with the decision fallbacks).
   Any anatomy without one is a NO-GO for the whole build until a rule is decided (223/223 must load).
9. **Resources:** measured per-anatomy time x 223 fits the window with <= 2 containers per agent; disk checked.

## 9. Risks and open points

- **The training route changes.** Pinning the fork heading (D3-fork) and the right siphon re-pose move the RCCA
  geometry in most anatomies (v5 right forks are a median 135 deg off the map). Wall detail survives a rigid move,
  but policies trained on v5 geometry see new routes. GO rule 4 keeps a fallback.
- **Seam bends.** Anatomical poses turn the anchor mismatch into seam bends (right forks median 38.5 deg, siphons
  up to 156 deg on bad anchors). The blend length, the bend direction and the anchor rules decide whether the 1 mm
  join stays <= 60; the audit found 20-45 deg of sampling uncertainty, so the gate runs on the resampled curve.
- **Contralateral forks** cannot be anatomical in both A-P and M-L; the map keeps A-P (D3-fork). 22 of the 30
  near-midline RICA seams sit on contralateral forks (7 lowers); T1 must show whether the anatomical pose clears
  them, else T2-style substitution on the right is NOT available (the right fork is the training route).
- **The VA target reference** (C_HOST) assumes the host head size; mr_027_L's anatomical tops end 32 mm above it.
  The donor pose search must aim the splice (T5); the siphon-termini reference is the fallback.
- **Donor FOV for (e):** head-CTA may not reach 10 mm after the subclavian origin; T4 records a shorter stretch
  where it must, gated on the seam.
- **VMR 0248** has no NIfTI: c1's CT-calibre floor does not apply (its wall is the VMR model's); T4 uses the model.
- **Midline** uncertain by +-6 mm (host x = 14.8); rules keep >= 10 mm margins.
- **Item f** (donor colour) is deferred; the side view is not.

## 10. Files (stage 1, v6_plan)

- `/sp/v6/V6_BUILD_PLAN.md` (this), `/sp/v6/pilot_set.json`, `/sp/v6/dr.sh` (container runner, names `v6plan_*`).
- `/sp/v6/gates/orient_gates.py`, `gates/test_api.py`; outputs `gates/out/selftest.txt`, `orient_v{3,4,5}.json`,
  `pieces_v{3,4,5}.tsv`, `test_api.txt`.
- `/sp/v6/heading/heading_evidence.py` -> `heading_evidence.{txt,json}`, `zenodo_frames.json`; `host_tree.py`,
  `zen_frames.py`, `c6check.py`.
- `/sp/v6/pilot/survey.py`, `make_pilot.py`.
- Notes `/sp/v6/notes/v6_plan.md`; state `/sp/v6/state/v6_plan.md`.
