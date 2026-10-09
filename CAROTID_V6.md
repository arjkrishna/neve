# v6 carotid anatomies: anatomically posed TopBrain pieces and the donor's own subclavian junction

`carotid_data/anatomies_v6/` holds 223 anatomies, one for each v5 anatomy under the same name. v5 gave every VA a real
donor lower vertebral artery, but an audit then showed that many TopBrain pieces were posed wrongly: left ICA siphons
pitched down, siphons and lower forks facing in arbitrary directions, VA tops fitted with no superior term. In v6 every
TopBrain piece and every lower fork is posed anatomically, with a bounded, blended seam bend. Left forks that cannot
carry a rising siphon are substituted. The donor VA now brings the donor's own subclavian junction and its own distal
subclavian. The donor wall's bone-side artefact is fixed where it occurs, and case_w_018b_left__topcow_mr_002's VA is
moved off the right route. The collision mesh is baked once, by v5's baker with the v6 sections, under one frozen
toolchain. The same 25 donor patients serve the same 25 TopBrain patients one to one, so the train/test split is not
lowered.

**The training route moved.** The right fork carries the RCCA, the training target. In v6 it is re-posed to its
anatomical heading (user decision D19), and in 113 anatomies it is x-mirrored as well (D20). The RCCA from the arch to
seam 1 is still the host's, bit for bit. Policies trained on v5 geometry will see new routes.

**After the final verifier (s3_fix, 2026-10-08).** The independent final verifier found real scanned wall that v4's and
v5's meshes show missing from 8 moved right-route pieces (1–5 % of each). The cause was the bake cutting thin slivers of
the scanned lumen loose; the 8 anatomies were re-baked (bake_v6 1.2) and now lose none of it, 8 pieces with a 0.1–1 %
deficit are accepted and flagged (D27), a gate for this direction is in checks_v6 1.6, and one fold-over face pair was
removed from a training OBJ (*The right route*; `notes/s3_fix.md`). A second final verifier reproduced all of s3_fix with
its own code and found no geometry problem, only records: 10 more pieces lose a measurable amount of real wall below
0.1 % and carried no flag. s4_records (record only, D27) flagged them, so all 18 pieces with a real-wall deficit (each
below 1 %) are flagged, and corrected stale numbers in this file (`notes/verify_v6_final2.md`, `notes/s4_records.md`).

    carotid_tools/v6/build/carotid/compose_v6.py   forks, siphons, ECA tips: anatomical pose + bounded seam bend; left-fork ladder
    carotid_tools/v6/build/va/graft_v6_va.py       donor VA with its own junction and distal subclavian; VA tops; D8-D10, D16-D18
    carotid_tools/v6/build/bake/bake_v6.py         v5's baker (handle rule) + the v6 sections, blend morph, mirrored forks; 1.2: lattice shift, route-island bridge
    carotid_tools/v6/build/checks/checks_v6.py     per-anatomy gates: strict audit, right route (D22 / D26 / D27), coverage against the v4 / v5 shipped wall, OBJ fold-over, lumen, VA, orientation, loader, geometry
    carotid_tools/v6/build/gates/orient_gates.py   the orientation gates (rise, tilt, heading, pair checks)
    carotid_tools/v6/build/build/build_v6.py       the build stages, refused unless the frozen toolchain's md5 manifest matches
    carotid_tools/v6/MANIFEST.md                   every copied file, the run that used it, how to re-run
    carotid_data/anatomies_v6/BUILD_v6.json        counts, lists and ranges generated from the final provenance only
    carotid_data/anatomies_v6/_records/            the results and notes this file cites

Every number below comes from a file in `carotid_data/anatomies_v6/_records/` (cited by its path there) or from
`BUILD_v6.json`. Per-anatomy numbers come from the provenance in each anatomy folder; `BUILD_v6.json` is generated from
those files and nothing else, and a separate script recounted its counts and lists from provenance: 123 checks, 0
differences (`checks/build_record_check.json`). Set-level results (split, loader, SOFA, orientation re-measure) are in
`results/v6_final_summary.json` and `results/v6_checks.json`; s3_fix's (coverage on all 223, SOFA on the 9 re-run
anatomies, split and loader on the final folder) in `results/v6_coverage_s3.json` and `results/s3_fix/`, copied into
`BUILD_v6.json` as labelled set-level blocks; s4_records' (the 10 pieces flagged below 0.1 %, the folder manifest after
their re-stamp) in `results/s4_records/`.

## At a glance

| | v5 | v6 |
|---|---|---|
| anatomies | 223 | **223**, same names, none dropped |
| anatomies with a visibly wrong TopBrain piece (audit tier A) | 141 | **0**, except 1 accepted and flagged by the user (D23) |
| right fork (RCCA training route) | v3's pose, byte for byte | **re-posed to the per-patient Zenodo map** (218) or v5's pose (5); 113 x-mirrored; RCCA arch → seam 1 bit-identical to v4 in 223 / 223 |
| right fork more than 45° off its anatomical heading | 213 | **5** (the 5 kept on v5's pose) |
| left fork more than 45° off its anatomical heading | 140 | **7** |
| ICA terminus more than 40 mm posterior of its target | 108 | **3** |
| left forks | v4's | own fork 201, a substitute 22; never the host LCCA tube, no ICA extension |
| right route wall (D22: real surface beyond the blend within 0.3 mm of v4's, carried by the piece's own transform) | — | **≥ 0.99 on 223 / 223**, fork and siphon (19 anatomies against the piece's own scanned surface, D26; 3 pieces with a bake repair window, D27) |
| right route wall the other way (coverage: the real wall v4's / v5's shipped mesh shows, carried, still on v6) | — | **no piece loses ≥ 1 %** (8 did before s3_fix and were re-baked); 18 lose more than 0 and less than 1 % (8 at 0.1–1 %, 10 below 0.1 %), all accepted and flagged |
| lower VA | donor VA in 429 of 446; a synthetic junction stub (median 5.28 mm) | the same 429 donor VAs; **the donor's own subclavian junction on 413**, v5's junction on 15, v5's top on 1; host VA on 17 (no donor, as in v5) |
| donor take-off against its native take-off | 251 VAs more than 15° off (median 17.8°) | own-junction sides: median 0.08°, max 0.73° |
| VA lumen in the OBJ, on the polyline | 4 VAs below 0.65 mm (min 0.547) | **none below 0.65** (min 0.664) |
| VA wall > 0.3 mm off the tube (median, undecimated union) | donor stretch 0.134, whole VA 0.136 (v6's measure; CAROTID_V5.md's 0.128 counts the seam zone as 0) | donor stretch **0.150**, whole VA **0.159**; donor subclavian stretch 0.743; v4 whole VA 0.018 |
| SOFA wire ≥ 20 mm into the RVA (25-anatomy set, same tool and seeds) | 33 / 75 seeds | **48 / 75** (v4 74 / 75) |
| SOFA on the RCCA training route | — | **223 / 223 pass**; tip never more than 0.19 mm outside the wall |
| collision mesh | one closed genus-0 two-manifold, 23,300–23,500 triangles | the same, 223 / 223; 23,498–23,500 triangles |
| overlap-free train/test split | 45 quantities | none lowered, 4 raised |
| licence | 216 non-commercial, 7 VMR 0248 | the same 216 / 7 |

Sources: `results/v6_final_summary.json` (`orientation`, `ica_ap_gt40`, `fork_heading_gt45`, `right_route`,
`va_lumen_polyline`, `wall_detail_median`, `d14_va_entry_set25`, `sofa_rcca`, `split`), `results/v6_checks.json` →
`wall_detail`, and `BUILD_v6.json` (`carotid`, `va`, `bake`, `checks`, `licences`). Where v6's checks re-measured v5 with
the same code (orientation, fork heading, ICA position, VA lumen, wall detail, SOFA), the v5 column gives that value;
the other v5 values are CAROTID_V5.md's.

## What changed from v5, and why

### The orientation audit

The user saw left ICA siphons and VA tops in eight v5 anatomies that went sideways or down instead of rising (k_011_left
mr_012_L / 013_L / 016_L / 023_L, w_003_left mr_007_L / 010, m_006_left mr_025_L / 027_L). The audit (report
`reports/TopBrain piece orientation audit (v5).md`; agent notes `notes/audit/`) found:

- **141 of 223 v5 anatomies** have at least one TopBrain piece whose course is visibly wrong (tier A: siphon rise
  < 20 mm, negative rise where the native one is positive, rise error > 20 mm, droop, or a VA top crossing or lateral);
  v4 171, v3 11.
- **The frame match pins only the roll.** Since set A the documented fix has matched tangent onto tangent and superior
  onto superior. That fixes the roll about the seam tangent. The tilt and the heading of the piece are copied from the
  anchor it joins (the fork's ICA tip, the host VA, the fork's ECA tip), and nothing checks the anchor. v1–v3 got away
  with it because their only anchor was a long, rising right ICA. v4 added left forks with short, down-pointing ICAs.
  v5's VA tops used a Kabsch fit with no superior term.
- The rise check that found the original fault was a one-off audit, never a gate. The 60° join gate cannot see a wrong
  pose, because the frame match makes the two tangents agree by construction.
- Six fault classes, each tied to a code path (anatomies affected in v5): left siphons pitched down by the left fork's
  ICA tip (L_PITCH 50), siphon heading left to chance (S_HEAD 16), right siphons tilted back (R_PITCH 17), VA tops on
  v4's host VA (VA4 14) or from v5's fit (VA5 206), ECA tips on down-pointing ECAs (36).

The audit's to-do list for v6 is `notes/V6_ITEMS.md`, and the plan is `notes/V6_BUILD_PLAN.md`.

### The decisions

The user asked that the v6 build not wait for questions. The orchestrator therefore took decisions D1–D15 and D25
itself (reversible, written down before use); the user took D16–D24, D26 and D27. Full text: `notes/DECISIONS_ADDENDUM.md`
and the plan's section 2; timeline: `notes/V5_BUILD_LOG.md` (which carries the v6 entries from 2026-10-03 17:25).

| id | taken by | decision |
|---|---|---|
| D1 | orchestrator (the user approved it on 2026-10-03 and ruled out the host fallback) | left forks: own fork if its siphon passes; else a ladder of substitutes that keep the split, then ICA extension; **never the host LCCA tube** |
| D2 | orchestrator | VA tops posed anatomically, aimed at the basilar target, stopped ≥ 0.6 mm apart; no basilar fusion |
| D3 | orchestrator | anatomical heading (TopBrain RAS → branch frame, A0 = diag(−1, −1, 1)) plus a seam bend ≤ 45°, blended; forks by the per-patient Zenodo map (D3-fork) |
| D4 | orchestrator | no ECA tip extension on a down-pointing ECA |
| D5 | orchestrator | walls stay coarse (real, irregular, non-tube) |
| D6, D7 | orchestrator | anchor_low blocking on the LICA seam and ECA tips, informational on the RICA; ICA-termini spacing two-sided |
| D8 | orchestrator | va_target ≤ 20 mm passes; 20–30 mm accepted and flagged; > 30 mm → D9 |
| D9 | orchestrator | VA ladder: donor junction + anatomical top → v5's junction + anatomical top → v5's top as shipped (flag); never drop a donor VA |
| D10 | orchestrator | VA corridor gate: over z 440–540 the donor VA's offset from the host neck midline within ±8 mm of its native offset |
| D11 | orchestrator (right-fork part superseded by D19) | left forks re-posed by the Zenodo map where every gate passes, else v5's pose (flag); ICA-terminus AP offset recorded, flagged > 40 mm |
| D12, D13 | orchestrator | ICA spacing like for like; right-route baseline = v4 without its RECA tip where v4's tip ran along the ICA and v6 re-poses or skips it |
| D14 | orchestrator | SOFA VA entry: 3 seeds + a geometric test, reported against v5, not a gate |
| D15 | orchestrator | donor wall: the bone-side fix only in the artefact zones, v5's coarse surface elsewhere |
| D16 | user | pinched donor VA ostia (ia172_164 L, ia172_071 L) bridged with a short tube at VA calibre, flagged |
| D17 | user | the donor's own distal subclavian continues toward the arm to the end of its segmentation as a cut end (a short tube if < ~30 mm past the ostium, flagged); the host's distal subclavian is dropped on those sides |
| D18 | user | sides still failing keep v5's junction, flagged; no other relaxations |
| D19 | user | right forks re-posed to their anatomical heading; the RCCA training route may move; seam-1 bend ≤ 45° blended, then a host-cut search, then up to 60° (flag); the RCCA prefix stays host |
| D20 | user (left forks: orchestrator's extension) | forks from the patient's other side are x-mirrored before placement, as their siphons already were |
| D21 | user | the 17 VA sides with no donor since v5 are not in this build |
| D22 | user | right-route check for moved pieces = "surface preserved": ≥ 99 % of the piece's real surface beyond the blend within 0.3 mm of v4's surface carried by the piece's own transform; blend zones and handle windows reported, not gated |
| D23 | user | w_017_left mr_002_L's right VA on D9 rung 3 (tier A): keep the donor VA, accept and flag |
| D24 | user | right forks on R4 / R5 accepted with flags, re-evaluated under D22 |
| D25 | orchestrator | the D17 direction gate; 1 mm joins gated at every sampling phase; one frozen toolchain (md5 manifest) for the whole build |
| D26 | user | where v4's mesh hid part of a moved piece's own scanned surface, D22 compares with that own surface (flag `d22_reference = own_surface`) |
| D27 | user (the rule); the build agent's implementation is checks_v6 1.5; s3_fix's is bake_v6 1.2 / checks_v6 1.6 | no stopping for choices: residuals get the recommended minimal option and a flag; k_011_left mr_012_L's siphon deficit on the bake's repair capsule is a bake repair window, reported and not gated. Applied after the final verifier (s3_fix, *The right route*): a right-route piece that lost real scanned wall against v4's / v5's shipped wall by ≥ 1 % gets a fix that restores the wall (a targeted re-bake), else v5's geometry for that piece; below 1 % it is accepted and flagged |

### The items

| item (V6_ITEMS) | what v6 does | where below |
|---|---|---|
| a1 orientation gates | `orient_gates.py` (rise against native, tilt, heading against the anatomical map, ICA-termini order and spacing, VA-top gap / target / crossing, anchor elevation); run before every placement and post-bake | *Gates* |
| a2 left-fork rule | D1 ladder; 22 substitutes | *Left forks* |
| a3 heading | siphons and both forks posed anatomically with a bounded, blended bend | *Forks and siphons* |
| a4 VA tops | aimed at the basilar target, ≥ 0.6 mm apart (D2, D8, D9) | *VA* |
| a5 ECA tips | skipped on down-pointing ECAs (D4) | *Forks and siphons* |
| a6 / a7 rebuild, docs | one bake, every check; this file; figures with a side view | *Collision mesh*, *Gates* |
| b w_018b move | the route13 tool on the v6 VA | *VA* |
| c narrow VAs | D15 bone-side fix; polyline lumen gate; the topcow_mr_021 pinch tubed (C4) | *Donor wall* |
| d build record | `BUILD_v6.json` from the final provenance only | *Records* |
| e donor junction | the donor's own subclavian stretch and junction; D16, D17 | *VA* |
| f donor colour in figures | **not done** (user: later) | *Follow-ups* |

## Sources and licences

v6 uses no new data source. The forks are CarotidAnalyzer forks (Zenodo 10695923 v2.0.0), the siphons, ECA tips and
VA tops are TopBrain pieces (Zenodo 16878417), and the donors are v5's 25 thin-slice CT patients. The donor's own
subclavian stretch comes from the same donor CT, segmented with the same CT-edge method. Sources and licences are as in
CAROTID_V5.md (*Sources and licences*):

| source | used for | licence |
|---|---|---|
| Head-CTA 172-series (Blagosklonova et al., Zenodo 15697196; doi 10.3390/data11040074) | 13 donor patients, 124 anatomies | CC BY 4.0, attribution |
| Song et al. 2024 (CMHA, figshare 26965450; doi 10.1038/s41597-024-04056-8) | 11 donor patients, 92 anatomies | CC BY 4.0, attribution |
| VMR 0248_H_AOCERE_CAS (Wilson et al. 2013, doi 10.1115/1.4025983) | 1 donor patient, 7 anatomies | research and development use; keep LICENSE_VMR_0248.txt; acknowledge NLM R01LM013120 / NHLBI HHSN268201100035C |
| Arterial (FLOWCAT-CV/arterial; weights CC BY-NC 4.0, code PolyForm Noncommercial) | every Song and 172-series donor | non-commercial |
| TotalSegmentator (Wasserthal et al. 2023, doi 10.1148/ryai.230024) | landmarks, QC, the spinal-canal midline for D10 | CC BY 4.0, attribution |

Every donor that went through Arterial makes its anatomy non-commercial: 216 anatomies. The 7 VMR 0248 anatomies carry
the VMR terms. `LICENSE_v6_donors.txt` is in all 223 folders (it replaces v5's `LICENSE_v5_donors.txt`, names the
donor, its licence and attributions, and notes the D16 / D17 tubes); the 7 VMR anatomies also keep
`LICENSE_VMR_0248.txt`. The licence files agree with the provenance on all 223 (`BUILD_v6.json` → `licences`;
`results/build/license_check.json`: 0 problems). v6 uses no TaG-Net data. Arterial-derived data is non-commercial.

## Composition

`build_v6.py` composes each anatomy in stages: the carotid side (`compose_v6` 3.2.0: right fork, left fork, siphons,
ECA tips), then the VA side (`graft_v6_va` 1.4.2) with the carotids as fixed neighbours, a joint re-check, a fix loop,
the route13 move, and a final check of every gate on the written tree (`notes/s2_integrate.md`, `notes/s2_integrate2.md`,
`notes/s2_build.md`). The carotid decisions were made once, in the all-223 dry run under a frozen toolchain, and the
final compose replays them (`--plan`); the VA side and the gates are re-run in full.

### The forks and the per-patient Zenodo frames

A rigid placement can only use a proper rotation, so a fork's heading has to come from a frame. The Zenodo scanner
frames are not one frame: the two sides of one patient fix the left-right axis for 61 patients, the bifurcation's x
sign about the scanner origin fixes it for the 16 one-sided ones (it agrees on 121 of 122 forks where both are known),
and x and y flip together in 112 of 122, so the frames differ by a 180° turn about z, not by a mirror. 29 patients are
LPS-like and 48 RAS-like (group m all LPS-like, w mostly RAS-like, k half and half). The map is the identity for an
LPS-like patient and Rz(180°) for a RAS-like one. Read this way, the ECA lies anterior of the ICA in 91 of 138 forks,
the textbook relation (`gates/heading_evidence.txt`, `gates/zenodo_frames.json`).

### Right forks: the training route (D19, D20, D24)

| rung | pose | anatomies |
|---|---|---:|
| R1 | the Zenodo map, seam-1 bend ≤ 45° blended at the host cut | 124 |
| R2 | the map with the host cut moved along the RCCA (within the documented seam limits; up to 15.0 mm) | 88 |
| R3 | the map with a seam-1 bend up to 60°, blended (w_002_right mr_013_L, mr_022, mr_024: 55.6–55.7°) | 3 |
| R4 | a bounded deviation from the map (w_018b_left mr_002_L and mr_007: 20° and a 45.4° bend; w_027_right mr_020: 5°) | 3 |
| R5 | v5's pose (w_003_right mr_001 and w_036_right mr_018_L: no left fork passes with any map-posed right fork; w_014_right mr_010 / mr_011_L / mr_027: every map pose fuses the fork's own ECA and ICA tubes, gate U-TG) | 5 |

Source: `BUILD_v6.json` → `carotid.right_fork`; the R5 reasons from `notes/s2_integrate2.md` and `notes/s2_build.md`
(3.1, 3.3). Seam-1 bend on the 218 map-posed forks: median 39.7°, p90 44.5°, max 55.7°. 113 right forks come from the
patient's other side and are x-mirrored before placement (D20): the bake flips their winding, and every one of the 1,732
mirrored sections in the bake (forks, siphons, VA pieces of the mirrored anatomies) came out with outward normals
(`BUILD_v6.json` → `bake`). Against the per-patient map,
5 right forks are more than 45° off (exactly the 5 R5 forks), against 213 in v5; fork tilt above 40°: 0
(`results/v6_checks.json` → `ica_ap_and_fork_headings`).

The RCCA from the arch to seam 1 is v4's, point for point, on all 223 (gate `rcca_prefix_unchanged`). Beyond seam 1 the
route is the fork's and the siphon's own surfaces in their new pose, and D22 checks that those surfaces survived the
move (*Gates*). The route moved, so SOFA was run on the RCCA of every anatomy (*SOFA*).

### Left forks (D1, D11, D20)

- **201 anatomies keep their own left fork; 22 take a substitute.** Every substitute is a split-compatible fork with
  its native ICA (the composer's rung 2). The v5 fork failed because its LICA seam rises less than 30° (anchor_low, 18)
  or because no pose of its siphon passes (4). The v5 forks replaced: case_k_011_right (5 anatomies),
  case_w_013_left (5), case_w_024_left (5), case_w_046_right (2), and one each of case_w_003_right, w_022_left,
  w_036_left, w_040_right and w_050_right; the full list with each substitute is in `BUILD_v6.json` →
  `carotid.left_fork.substituted`.
- **No anatomy fell back to the host LCCA tube, and no ICA was extended** (D1's last rungs were never needed).
- **Pose.** 215 left forks sit on the Zenodo map, 29 of them at a bounded deviation of 5–43° (median 25°, flagged
  `left_pose_map_bounded_dev`); 8 could not be posed on the map with a passing siphon and keep a frame-matched or v5
  pose (flagged `left_pose_not_map`). 7 of those 8 are more than 45° off the map; in v5, 140 left forks were. Seam-1
  bend: median 24.5°, max 45.0°. 101 left forks are mirrored (D20).
- The split was re-checked on the final tree: not lowered (*Gates*).

### Siphons and ECA tips (D3, D4, D13)

Every siphon is posed by the anatomical map at its seam, with the bend bounded and blended over several millimetres.
The ICA termini now sit where they belong: the terminus lies more than 40 mm posterior of its anatomical target in 3
anatomies (all RICA: w_003_right mr_001 46.0 mm, on v5's fork; w_018_right mr_007_L 53.3 mm; w_027_right mr_016
45.3 mm) against 108 in v5; the median offset is 6.9 mm (RICA) and 4.2 mm (LICA), against 29.5 and 20.8 mm in v5
(`results/v6_checks.json` → `ica_ap_and_fork_headings`).

ECA tip extensions dropped: 20 on the right (16 on a down-pointing fork ECA, D4; 3 where no tip pose keeps the right
route; 1 by the post-bake D22 feedback, w_008_right mr_001) and 36 on the left (19 down-pointing, D4; 1 with no passing
pose; 16 on substituted forks, where the TopBrain ECA piece is shorter than 10 mm or missing). Each is recorded with its
reason (`v6_build.carotid.skipped_eca_tips`). Where v4's RECA tip ran along the ICA and v6 re-poses or skips it, the
right-route reference is v4 rebuilt without that tip (D13): 36 anatomies (`BUILD_v6.json` → `carotid.eca`,
`checks.d13_baseline`).

### VA: the donor's own junction (item e, D2, D8–D10, D16–D18, D25)

v5 joined each donor VA to the host subclavian through a short synthetic stub and posed it on the host's VA corridor,
so the donor's take-off was lost. In v6 the donor brings a stretch of its own subclavian: from about 10 mm after the
subclavian origin (the brachiocephalic split and the route start untouched) through its own VA ostium, and on toward
the arm to the end of its segmentation (D17). The VA leaves at its native take-off, and the catheter meets the donor's
real subclavian → VA junction.

| side | donor's own junction (D9 rung 1) | of which tube-bridged ostium (D16 / stage-1 fill) | v5's junction (rung 2, D18) | v5's top as shipped (rung 3) | host VA, no donor (D21) |
|---|---:|---:|---:|---:|---:|
| RVA | 198 | 8 (ia172_164 L, 9.71 mm) | 12 | 1 (D23) | 12 |
| LVA | 215 | 12 (ia172_164 L 9.71 mm × 8, ia172_071 L 10.01 mm × 2, ia172_136 L × 2) | 3 | 0 | 5 |

Source: `BUILD_v6.json` → `va.rung`, `va.junction`, `va.d16_tube`, `va.d18_v5_junction`. The 15 rung-2 sides
(`va.d18_v5_junction`) lost the donor junction because no rung-1 pose passed the VA-top gates (6), the subclavian's arm
direction (1) or the orientation gates (2, song_040 R: droop), because the final side gates refused it (2, ia172_069 L,
the D17 direction case), or because the joint pass demoted it (4); each side's reason is in its provenance
(`v6_va.record.sides.<side>.why_v5_junction`). The 17 host sides are v5's (`va.host_no_donor_D21`).

- **Take-off.** On the 413 own-junction sides the placed take-off differs from the donor's native one by a median 0.08°
  (max 0.73°), against 17.8° median in v5 (`va.ranges_donor_sides`).
- **The distal subclavian (D17).** On every own-junction side the donor's subclavian continues toward the arm and ends
  as a cut end 29.2–108.2 mm past the ostium (median 69.2 mm). Where the label ends short, a straight tube at donor
  calibre runs to 30 mm past the ostium: 15 RVA and 24 LVA sides (song_109 R 11.9 mm, song_040 R 9.8 mm, song_002 R
  0.57 mm; `va.d17_tube`). The host's distal subclavian beyond the donor stretch is dropped on these sides. Every cut end
  lies lateral of its ostium, within 90° of the host's distal subclavian (D25 direction gate; it demoted ia172_069 L in
  m_006_left mr_025_L and w_051_right mr_025_L to v5's junction).
- **VA tops (D2, D8).** Each TopBrain top is posed anatomically and aimed at the basilar target, and the two tops stop
  at least 0.6 mm apart. The end lies within 20 mm of the target on 435 of 446 tops, 20–30 mm on 10 (accepted, flagged;
  an eleventh flag sits on w_022_right mr_026 at 19.999 mm), and 55.3 mm on 1, the D23 side
  (`va.va_target_band`).
- **Course and corridor.** Course RMS on the donor stretch: median 12.8 mm, max 21.3 mm (the 2 sides above 20 keep v5's
  junction and its course, flagged `course_rms_donor_gt20_inherited_v5`). D10 corridor: |placed − native| offset median
  3.5 mm; 7 RVA and 6 LVA sides use the host-VA fallback (VMR 0248 has no CT for the native offset); the medial tail is
  flagged on 7 sides and the lateral tail on 210 (recorded only: the fixed window also takes in V1 / V3).
- **Fold gate (U3).** The full bake found one RVA whose new seam-1 stretch turned back into its own host prefix
  (m_030_right mr_013_L, gap −0.067 mm). graft_v6_va 1.4.2 gates each side's new curves against their own tube (lumen
  gap ≥ 0.6 mm between points more than r_i + r_j + 6 mm apart), and a side that fails only on that re-runs its pose
  search with folding poses refused (`notes/s2_build.md` 3.1, 3.3; `diag/tube_m030_mr013L.json`).
- **D23.** w_017_left mr_002_L's RVA stays on v5's top as shipped (rung 3) with its donor (ia172_136 L); it is the one
  tier-A piece of the build, accepted by the user and flagged.

**case_w_018b_left__topcow_mr_002 (item b).** v5's only anatomy below 0.99 of v4 under exact ownership (fork 0.956: the
donor RVA passed 1.6–3.9 mm from the RCCA wall). The build's route13 stage re-placed its VA with the route13 tool on the
v6 VA (graft_v5_reg_sep's separation objective; the v4 RCCA wall from `item_b/pilot_build/route_diag_v6.py`), and the
move was adopted (`BUILD_v6.json` → `carotid.route13_moved`). Its right route now passes D22 against v4 (fork and
siphon 1.0).

### Donor wall: the bone-side fix and the narrow VAs (item c, D15)

The audit found that v5's narrow VA points were not narrowings in the CT: where the VA lies against the
transverse-foramen bone, the CT-edge ray ran into the bone and v5's lower threshold cut the wall short, the 1.0 mm
radius floor then held the union near 0.93 mm, and decimation took the OBJ below 0.65 mm. The systematic fix
(`notes/v6_wall.md`): on bone-side rays end the ray at the lumen plateau (the `plateau` rule). Applied everywhere it
also smoothed correct wall, so D15 applies it **only in the artefact zones** (bone contact or floored stretches, plus a
2 mm blend) and keeps v5's coarse surface elsewhere (`wall_local/`, `notes/s2_harden_wall_bake.md`). The donor's
subclavian stretch uses the same local rule.

- Polyline VA lumen in the OBJ (0.05 mm sampling, exact distance): minimum 0.664 mm on 446 VAs, none below 0.65 mm,
  against 4 VAs below in v5 (min 0.547). The four v5 cases are all fixed. collision_full: min 0.826 mm
  (`checks/lumen/va_lumen_poly_v6.json`).
- c3: of the 8 CT-narrow points without bone contact, 4 were v5's cut and 4 are mild real narrowings (20–30 % in
  diameter) inside a foramen, left as they are (`notes/v6_wall.md`, section 5).
- c4: the pinch in the topcow_mr_021 left-VA TopBrain label is tubed at the neighbouring calibre (1.84 → 1.64 mm over
  native s 33.5–53.0) on the 8 anatomies that carry it (`BUILD_v6.json` → `bake.c4_mr021_lva_pinch_applied`).

## Collision mesh

`bake_v6.py` 1.1 is v5's baker (`bake_v5b`: `bake_meshes_v4` unchanged plus the handle rule) with stated additions: the
donor subclavian sections (E1, E2), the changed carotid sections searched with the VA pool at v4's carotid margin (E3,
E3b), v4's pinned windows carried through a moved or mirrored piece's own frame (E4), the blended seam morphed by the
composer's displacement (E6), a VA-aware OBJ re-cut if the VA polyline lumen fell below 0.65 mm (E7; never needed: "ok
as baked" on 223), and the C4 tube (`notes/s2_harden_wall_bake.md`). Each anatomy was baked once; 216 bakes come from
the first full build and 7 from the VA-1.4 round, reused only where every curve and every bake input was identical
(`results/build/tree3_merge.json`, `results/build/verify_tree3.json`). After the final verifier, 8 of those bakes were
replaced by **bake_v6 1.2** (s3_fix), which is 1.1 plus two opt-in options, both off by default: **L1** bakes on a
lattice shifted by a sub-voxel step (the field and every section capsule move together; same surfaces, centrelines and
ownership), and **L2** joins each island of the right-route piece's own scanned lumen that `keep_largest` would drop to the
main body with one 0.5 mm tube, kept only while the union stays genus 0 (*The right route*). L1 on 4 anatomies, L2 on 3,
both on 1; their provenance records `bake_v6.lattice_shift_mm` and `bake_v6.route_island_bridge`
(`per_anatomy/bake_s3/`). With both options off, 1.2 bakes exactly as 1.1, and the final re-bakes reproduce the dev
re-bakes bit for bit.

| rung | anatomies |
|---|---:|
| 1: carotid pinned, VA whole, VA margin 0.5 mm | 129 |
| 2: VA-local tube windows | 94 |
| fallbacks (carotid windows widened, donor tube, v4 VA kept) | 0 |

Source: `BUILD_v6.json` → `bake`. v5 had 196 / 25 / 2. In v6 the donor subclavian sections and the changed carotid
sections join the searched pool (E3), and more anatomies need a local window. s3_fix moved 4 anatomies from rung 2 to
rung 1 (the window-class re-bakes need no handle window: w_052_left mr_026, w_027_left mr_022 / mr_026, w_051_right
mr_008_L); the s2 build had 125 / 98.

- **Tube windows.** 124 anatomies carry a tube window somewhere: 209 sections with windows, tubed length per section
  median 1.63 mm (0.32–21.0 mm). On the VA side: the donor VA (vaC_R 18, vaC_L 14) and the right subclavian stretch
  (scC_R 3, scD_R 8). On the carotid side (v4's windows carried through each piece's frame, E4, and those found where a
  changed section was searched): siphon_L 47, lower_L 32, siphon_R 32, lower_R 22, the ECAs 33. No section became a
  whole tube; 4 ECA sections are tube-only as in v4 / v5, and w_008_right mr_001 uses v4's far clamp, as in v4.
- **Margins.** Carotid 0.5 mm on 201 anatomies and 1.0 mm on 22, as in v4; VA 0.5 mm on all 223.
- **Bake fusions found and fixed.** The first full bake failed on 4 anatomies whose tube-only mesh already had a handle:
  m_030_right mr_013_L (the RVA fold above) and w_014_right mr_010 / mr_011_L / mr_027, whose fork ECA runs 0.37 mm from
  its ICA and fuses at the re-posed pose. compose_v6 3.2.0 now requires the right fork group's tube union to be genus 0
  (U-TG), which put the three w_014_right forks on v5's pose (`notes/s2_build.md` 3.1, 3.3;
  `per_anatomy/tube_genus_prebake/`).
- Bake time 134–848 s per anatomy (median 194 s; provenance `bake_v6.seconds`; the s2 bake markers collected in
  `results/v6_build.json` read 137–851 s; the 8 s3_fix re-bakes 174–645 s, their bake markers in `per_anatomy/bake_s3/`
  read 177–647 s), peak RSS 1.54–2.11 GB (`results/v6_build.json` →
  `stats`; the re-bakes 1.58–1.87 GB, `per_anatomy/bake_s3/`).

## Gates and results (all 223)

Per-anatomy gates are checked by `checks_v6` on every anatomy (records `per_anatomy/checks4/<name>.json`, stamped
into each provenance as `v6_checks`). `anatomies_v6` is the checked tree file for file (sha256 on 4,690 files;
`checks/ident/ident_v6_tree3.json`), and a fresh run of the route check straight from `anatomies_v6` on 4 anatomies gave
identical records (`checks/route_spot/`). The 17 anatomies s3_fix changed were checked again with checks_v6 1.6
(`per_anatomy/checks_s3/<name>.json`, re-stamped): every part on the 9 whose geometry changed, the new coverage and fold
parts on the 8 whose geometry did not; 1.6's coverage and fold parts also ran on all 223 (`results/v6_coverage_s3.json`).
s4_records changed no check result: it added a flag and its detail (`v6_checks.s4_records`) to 10 provenance files.

| gate | rule | result | source |
|---|---|---|---|
| strict mesh audit | union, collision_full and OBJ: one closed genus-0 two-manifold, 0 self-intersecting faces, every centreline point inside the OBJ | 223 / 223 | `results/v6_checks.json` → `per_anatomy_gates` |
| right route, fork (D22 / D26) | ≥ 99 % of the fork's real surface beyond the seam-1 blend within 0.3 mm of the reference carried by its own transform | 223 / 223 (min 0.9915) | `BUILD_v6.json` → `checks.d22_value_under_reference` |
| right route, siphon (D22 / D26 / D27) | the same beyond the seam-2 zone | 223 / 223 (min 0.9923) | the same |
| right route, coverage (checks_v6 1.6, D27) | the real scanned wall v4's and v5's shipped meshes show, carried by the piece's own transforms, lost on v6 for < 1 % of each piece (0.1–1 % flagged by checks_v6 1.6; a measurable loss below 0.1 % flagged in the record) | 223 / 223 after s3_fix (0 pieces ≥ 1 %; 18 flagged: 8 at 0.1–1 %, 10 below 0.1 % flagged by s4_records); before it 8 pieces ≥ 1 % | `results/v6_coverage_s3.json`, `results/s4_records/coverage_sub01_pieces.json` |
| RCCA prefix | arch → seam 1 bit-identical to v4 | 223 / 223 | `results/v6_checks.json` → `right_route` |
| OBJ fold-over (checks_v6 1.6) | no two edge-adjacent OBJ faces folded onto each other (normals opposite within 1°); MeshLab's self-intersection test skips such pairs | 223 / 223 after s3_fix (m_022_left mr_021_L had 1; collision_full of w_017_left mr_014_L has 1, information, as v4 / v5 each have one) | `results/v6_coverage_s3.json` → `after` |
| VA lumen | polyline lumen in the OBJ ≥ 0.65 mm on both VAs | 223 / 223 (min 0.664 mm) | `checks/lumen/va_lumen_poly_v6.json` |
| orientation | 0 tier-A pieces (D23 accepted), 0 v6-blocking flags after D8 / D23 | 223 / 223 | `checks/orient/orient_set_v6.json` |
| loader | the training loader on the anatomy | 223 / 223 | `checks/loader/loader_set_v6.json` |
| clearance | lumen gap ≥ 0.6 mm between placement groups on every new point | 223 / 223 (min 0.614 mm) | `results/v6_checks.json` → `per_anatomy_gates.geom` |
| 1 mm joins | every new 1 mm turn ≤ 60° at every sampling phase, or no sharper than the same v5 curve's own turn | 223 / 223 | the same |
| split | `v5_split.py anatomies_v6 --baseline split_v4.json`: exit 0, none of 45 lowered | exit 0; 4 raised | `checks/split/split_v6.json` |
| donors one to one | 25 donor units onto 25 TopBrain patients, none spanning two | 25 / 25 | `BUILD_v6.json` → `donors.one_to_one` |
| SOFA, RCCA | soundness on the training route (no exception, finite, tip inside) | 223 / 223 | `sofa/rcca_v6/` |

Composition-time gates, enforced before writing and re-checked on the final tree: proper rotations (mirrors only as D20),
course RMS ≤ 20 mm on the donor stretch, va_target (D8), the D10 corridor, the D17 direction gate, the fold gate,
U-TG, the ICA-termini order and spacing, anchor elevation, and the orientation gates on every candidate pose
(`notes/s2_integrate2.md` section 1).

**All 223 anatomies pass every gate** (`BUILD_v6.json` → `checks.pass`, `checks.gates`; `results/v6_final_summary.json`
→ `invariants`, verdict "ALL INVARIANTS PASS").

### The right route (D22, D26, D27)

The relief ratio (v5's ≥ 0.99 of v4's wall detail) cannot gate a moved piece: the share of wall more than 0.3 mm off
the tube changes by 1–10 % under a rigid or mirror move even when the surface is the same, and the stage-2 gate verifier
showed forks identical to v4 within 0.1 mm reading 0.978–0.990 (`notes/verify_s2_gate.md`). The user replaced it by
D22: the piece's real surface must still be there, carried by its own transform.

- **Reference.** Fork: v4 on 205, the piece's own scanned surface on 18 (D26); siphon: v4 on 220, own surface on 3. D26
  applies where v4's mesh never showed part of the piece's own surface, for example a mirrored contralateral fork whose
  bulb sat inside a neighbour in v4: there v6 shows real wall that v4 hid. The 19 D26 anatomies: k_011_left mr_012_L
  (fork and siphon), w_002_right mr_013_L (fork and siphon), w_008_right mr_003, w_012_left mr_003_L / mr_020 / mr_021_L,
  w_033_left mr_007 / 010 / 011_L / 016 / 024, w_040_left mr_021, w_046_left mr_012 / mr_020_L / mr_026, w_047_left
  mr_023 (siphon), w_048_left mr_008 / mr_010_L, w_051_right mr_008_L. Each is flagged
  `d26_d22_<piece>_reference_own_surface` in its provenance. Four of the pieces are new with s3_fix: the bridged islands
  of the tip class are real scan that the D13 baseline (v4 without its tip) never showed, so those forks (and w_002's
  siphon, whose island was bridged too) now read against their own surface.
- **Values** under the reference used: fork min 0.9915, siphon min 0.9923; against v4 alone: fork min 0.922
  (w_008_right mr_003's bridged islands), siphon min 0.9776 (k_011_left mr_012_L).
- **D27, k_011_left mr_012_L's siphon.** Under D26 it read 0.9898. The 244 cells off both references (17.79 mm², s
  206.75–207.5 mm) lie where the bake's own section field is its capsule round the siphon centreline, not the scanned
  surface: the scanned lumen reaches 0.3–2.4 mm past the capsule, and v4's bake clipped the same cells (100 %). checks_v6
  1.5 reports such capsule-clipped cells as a bake repair window and leaves them out of the gated area, like the seam
  zone and the handle windows; the siphon then reads 1.0 against its own surface, and no v4 wall is missing (coverage
  0.9915). It changes nothing for a piece that passes under 1.4 (`diag/d27test/`, `diag/d27eq_cmp.txt`). Flagged
  `d27_d22_siphon_repair_window_17.79_mm2`. The same rule applies to two forks re-baked by s3_fix, where parts of the
  bridged islands reach past the capsule: w_002_right mr_013_L (12.01 mm², s 108.3–116.5; own surface 0.988 → 0.997;
  75 % of these cells also capsule-clipped by v4's own bake) and w_008_right mr_003 (56.57 mm², s 78.8–120.3; 0.968 →
  0.997; 90 % in v4), flagged `d27_d22_fork_repair_window_*`. **w_008_right mr_003's re-baked fork, disclosed:** the
  second final verifier, measuring against v4's shipped mesh and the fork's own scan together, finds 2.0 % of its
  D22-gated surface on neither (39.7 mm², 0.9798): bake capsule (33.6 mm², where the scan reaches past 1.8 r + 1 mm) plus
  the surface next to the 0.5 mm bridge tubes (7.1 mm²). Before the re-bake the fork read 1.0. It passes D22 only
  through the D27 repair window, which leaves the capsule cells out (0.997), and is flagged
  `d27_d22_fork_repair_window_56.57_mm2` (`notes/verify_v6_final2.md` section 2).
- **Reported, not gated.** The seam-1 blend zone (fork area median 161 mm², a median 72 % of it within 0.3 mm, max deviation median
  1.23 mm), the bake's handle windows (on the fork in 25 anatomies, on the siphon in 32 in the final folder, each
  anatomy's `mesh_v6.json` → `suppressed_stretches_mm`; 28 / 31 as first built, before s3_fix) and the relief ratio (fork
  median 1.001, 52 below 0.99; siphon 1.002, 18 below 0.99) (`results/v6_checks.json` → `right_route`).
- **Coverage, the other direction (gate since checks_v6 1.6).** D22 asks that every v6 RCCA cell lie on the reference.
  Coverage asks that the real wall v4 and v5 showed reappear on v6. Per piece and per reference (v4's and v5's SHIPPED
  collision_full, each with its own curves and exact ownership), the reference's RCCA-owned cells of the gated stretch are
  carried by the piece's own transforms; a cell farther than 0.3 mm from the v6 union that lies on the piece's own scanned
  surface and outside the union is real wall lost (a cell inside the union is a neighbour's wall the reference showed,
  D26's case). Gate `route_<piece>_coverage_loss_lt_1pct`: lost area < 1 % of the piece on both references; 0.1–1 % is
  accepted and flagged by the check (D27), and a measurable loss below 0.1 % is flagged in the record (s4_records). The
  bake's handle windows (± 0.5 mm) are left out, as D22 leaves them out.

**What the final verifier found, and the fix (s3_fix, D27).** On the folder as first built, 8 pieces lost ≥ 1 %
(`checks/coverage_v1p6_prefix/`; the final verifier's list exactly):

| piece | lost (v4 / v5 reference) | class | fix (bake_v6 1.2) | after |
|---|---|---|---|---|
| w_052_left mr_026 fork | 4.7 % (60.4 mm²) | window | L1 lattice shift (0.225, 0, 0) | 0 |
| w_027_left mr_022 fork | 2.1 % (27.4 mm²) | window | L1 (0.225, 0, 0) | 0 |
| w_027_left mr_026 fork | 2.1 % (27.6 mm²) | window | L1 (0.225, 0, 0) | 0 |
| k_011_left mr_012_L fork | 2.1 % (25.5 mm²) | tip | L2 route-island bridge | 0 |
| w_012_left mr_003_L fork | 1.5 % (23.4 mm²) | tip | L2 | 0 |
| w_002_right mr_013_L fork | 1.5 % (17.4 mm²) | tip | L1 (0.225, 0, 0) + L2 | 0 |
| w_008_right mr_003 fork | 1.1 % (21.0 mm²) | tip | L2 | 0 |
| w_051_right mr_008_L siphon | 1.0 % (14.6 mm²) | window | L1 (0, 0, 0.225) | 0 |

- **Cause** (`notes/s3_fix.md` section 1, `diag/s3_fix/`). Every lost stretch is a component of the bake's union that
  `keep_largest` dropped: an island of the piece's own scanned lumen, a thin sliver (about 0.1–0.5 mm thick) 1–3 mm off
  the main lumen. *Window class*: at the v6 pose the sliver closes a handle with the main lumen, the bake's handle window
  cuts its neck, and the rest of the sliver becomes an island (the shrink step accepts any genus-0 window). *Tip class*
  (exactly the four D13 "v4 without its tip" anatomies): the sliver is separate from the lumen in the scan itself (0.62 mm
  gap at 0.15 mm resolution on k_011); v4 and v5 showed it only because their RECA TopBrain tip's tube ran through the gap,
  and v6 re-posed that tip anatomically. The capsule, the radii and the SDF sign were ruled out.
- **Fixes, in the order D27 gives** (first a fix that restores the wall, else v5's geometry for the piece; the latter was
  never needed). *L1 lattice shift*: the same surfaces, centrelines and ownership baked on a lattice shifted by a sub-voxel
  step (tried in a fixed order); on the window class it removes the handle, so no window cuts the sliver. *L2 route-island
  bridge*: each island of the route piece's own scan that the union would drop is joined to the main body by one short tube
  (radius 0.5 mm), kept only while the union stays genus 0. Dev runs of every variant tried are in `diag/s3_fix/dev_rebake/`;
  w_002's L2 alone failed (the island holding its lost wall is a ring, genus 1 by itself) and L1 + L2 passed.
- Each re-baked anatomy passes every checks_v6 1.6 gate and SOFA on the RCCA, and carries `d27_bake_lattice_shift_*` or
  `d27_bake_route_island_bridge_<n>` in `v6_checks.flags` (`per_anatomy/checks_s3/`, `results/v6_coverage_s3.json`).
- **After the fix, on all 223**: no piece loses ≥ 1 %; 8 lose 0.1–1 % and are flagged (`results/v6_coverage_s3.json` →
  `after`); 10 lose more than 0 and less than 0.1 % (0.005–0.097 %). s3_fix left these 10 unflagged as discretisation
  noise; the second final verifier showed one of them (w_034_left mr_008_L's fork) is a real patch, and s4_records
  flagged all 10 (`results/s4_records/coverage_sub01_pieces.json`, *Known limitations*). The other 428 pieces lose
  nothing. Two of them show a small loss only against the shipped, decimated collision_full and none against the union
  the rule measures (w_021_right mr_008_L siphon 0.036 %, w_052_left mr_027 fork 0.013 %); they are not flagged.

### Orientation

On the same code (`orient_gates.py` in tree mode on the whole folder): tier A on 1 anatomy, the D23 side, 0 after D23,
against 141 in v5; v6-blocking flags after D8 / D23: 0, against 223 in v5. Accepted flags: va_target 20–30 mm on 11 tops
(D8) and D23's 5 flags. ECA anterior of the ICA at the fork: right 207 and left 191 of 223, against 78 and 122 in v5
(`checks/orient/orient_set_v6.json`, `orient_set_v5.json`).

### Wall detail

The measure is v5's: the share of wall area more than 0.3 mm off the centreline tube, exact ownership, on the undecimated
unions (`results/v6_checks.json` → `wall_detail`).

| stretch | v6 median | v5 median | v4 |
|---|---:|---:|---:|
| donor stretch (seam zone and tube windows excluded), 429 VAs | **0.150** (p10 0.099, min 0.019) | 0.134 (min 0.029) | host tube 0.000 (max 0.020) |
| donor subclavian stretch, 413 sides | **0.743** (0.523–0.875) | host tube | host tube |
| TopBrain piece, 446 | 0.157 | 0.106 | 0.102 |
| whole VA, ostium → end, 446 | **0.159** | 0.136 | 0.018 |

Every donor stretch beats v4's host tube on the same side (429 / 429), and the whole VA beats v4's on 438 of 446; the
other 8 are host sides (D21) whose TopBrain top moved. D5 (donor median ≥ 0.10) holds. Two comparisons are not like for
like: the v6 donor stretch runs from the donor's own junction to a re-posed splice, v5's from the stub; and on
song_AHMU1218141 L (4 anatomies) the v6 stretch is 100.6 mm long against 170.6 mm in v5, so its 0.019 is not v5's 0.094
measured again (its whole VA reads 0.146–0.147 against 0.157–0.159).

### SOFA

`sofa_rcca_v6` 1.0 and `sofa_va_entry` 1.1, seeds 777 / 778 / 779 × 300 steps from the training insertion, through the
training loader (`notes/s2_checks.md` section 6).

**RCCA, all 223.** 223 / 223 pass (0 exceptions, 0 non-finite steps, the wire tip at most 0.193 mm outside the wall).
The 9 anatomies whose geometry s3_fix changed were run again with the same tool and seeds, on their final files: 9 / 9
pass, the tip never outside the wall (`sofa/rcca_v6_s3/`, `results/s3_fix/sofa_rcca_summary.json`). Reach is not a gate
(the controller is a scripted heuristic) and is reported, after the fix (before it in brackets):

| point reached | seeds (of 669) | anatomies (≥ 1 seed) |
|---|---:|---:|
| seam 1 + 20 mm | 664 (664) | 223 (223) |
| past the bifurcation | 637 (638) | 218 (219) |
| siphon | 511 (512) | 204 (205) |
| target | 373 (374) | 182 (183) |

The one-seed differences are w_002_right mr_013_L, whose one seed that reached the target in the first build (seed 777)
no longer passes the bifurcation: it went 9.9 mm into the ICA (to RCCA s 73.2, just short of the 73.3 mm
past-the-bifurcation mark), buckled and ended at RCCA s 51; its other two seeds turn into the ECA, as before. The fork
pose and the bifurcation lumen are unchanged (the OBJ round the bifurcation moved ≤ 0.5 mm with the re-sampled mesh:
the sub-voxel lattice shift and the bridges, which lie past the bifurcation), so this is the scripted controller's
sensitivity to the mesh (`sofa/rcca_v6_s3/`, `notes/verify_v6_final2.md` section 3); w_008_right mr_003 gained a target seed,
w_012_left mr_003_L lost one at the siphon; the other 6 reach as before or further.

**The 25-anatomy set, against v5** (same tool and seeds): no geometry in it changed after the build; 3 of its anatomies
were re-stamped for a flag, provenance only (w_008_right mr_001 and w_047_left mr_027 by s3_fix, m_024_left mr_001 by
s4_records), so the numbers stand:
past the bifurcation 73 vs 72, siphon **52 vs 54**, target **36 vs 46** seeds (anatomies with a target seed 19 vs 21).
The drop comes with the moved training route: the re-posed (D19) and, on 113 anatomies, x-mirrored (D20) right forks
put the scripted push on a different bifurcation and siphon course than v5's. The 4 w_042_left anatomies never pass the
bifurcation: the wire turns into the ECA, which leaves the CCA only 20.6° off axis against the ICA's 45° at equal
calibre; v4 and v5 behave the same (`sofa/bif_angles_w042.json`).

**VA entry (D14), the 25-anatomy set.** Wire ≥ 20 mm into the RVA: v6 48 / 75 seeds (22 / 25 anatomies), v5 33 / 75
(19 / 25), v4 74 / 75 (25 / 25); median entry depth 42.8 mm, v5 5.0 mm, v4 138.5 mm. The geometric probe (a straight
0.65 mm probe into the ostium) enters 204 RVAs and 188 LVAs of 223, against v5 185 / 180 and v4 223 / 223
(`sofa/d14_*`, `checks/entry_geom/`). v5's own record (10 / 25) used a schedule without a device reset per seed and is
not comparable seed for seed.

### Split, loader and the copy

- **Split** on `anatomies_v6`: exit 0, 45 compared, none lowered; strict div6x6 k16 train 168 → 169, kept 184 → 185; k17
  train 167 → 169, kept 184 → 186 (`checks/split/split_v6.json`).
- **Loader.** The training call over the whole folder: 223 / 223, 18 branches each, every OBJ its own, the shared
  insertion inside each OBJ, RCCA 206.2–261.0 mm (`checks/loader/loader_set_v6.json`); and a deterministic load of every
  anatomy (`results/build/copy_verify_final.json` → `load_v6`).
- **The copy.** Staged first, then `anatomies_v6`: 5,136 files, `sha256sum -c` exit 0, manifest sha256 977388c1… equal
  to the staged copy (`results/build/copy_verify_final.json`, `results/build/anatomies_v6.sha256`).
- **After s3_fix** (2026-10-08): the 17 changed folders were staged (same file set), swapped in, and the whole folder
  verified against the updated manifest (sha256 024edcd2…; 223 anatomies, 5,136 files, `sha256sum -c` exit 0;
  `results/s3_fix/copy_verify_s3.json`). Only the intended files differ from the first copy: OBJ, collision_full,
  mesh_v6.json and provenance on the 8 re-baked, OBJ, mesh_v6.json and provenance on m_022_left mr_021_L, provenance on
  the 8 re-stamped; no centreline changed. On the final folder the split is unchanged (exit 0, 45 compared, none lowered,
  the same baseline rows) and the training loader passes 223 / 223 (`results/s3_fix/split_s3.json`, `loader_s3.json`).
- **After s4_records** (2026-10-08): the 10 re-stamped provenance files replaced, no other file; the whole folder
  verified against the new manifest (sha256 6ad80caf…; 223 anatomies, 5,136 files, `sha256sum -c` exit 0; exactly the 10
  provenance lines differ from s3_fix's 024edcd2; `results/s4_records/anatomies_v6.sha256`, `copy_verify_s4.json`).
  s3_fix's manifest stays in `results/s3_fix/` as the record of the folder after s3_fix.

## Flags and their counts

The composer's, the VA tool's, the bake's and the checks' flags are written in the anatomy's provenance
(`v6_build.flags`, `bake_v6.flags`, `v6_checks.flags`) and counted in `BUILD_v6.json` → `flags`. Two flag classes are
derived outside provenance, from values that are in it: `left_pose_map_bounded_dev` (from the left fork's recorded
rotation against its map frame) and `orient:<flag>` (the orientation gates' accepted flags, from the checks record).
They are listed per anatomy in `BUILD_v6.json` → `per_anatomy.<name>.derived_flags` and counted in
`flags.derived_flags`; the 17 anatomies s3_fix re-stamped also carry them in provenance (`v6_checks.derived_flags`).
The first version of this file said every flag was in provenance; that was not true for these two classes (*Record
corrections*).

| flag | count | meaning |
|---|---:|---|
| `right_fork_mirrored_D20` / `left_fork_mirrored_D20` | 113 / 101 | fork from the patient's other side, x-mirrored |
| `right_host_cut_shift` | 89 | R2: host cut moved along the RCCA |
| `right_fork_R3_map_bend60` | 3 | seam-1 bend up to 60° (w_002_right × 3) |
| `right_fork_R4_map_dev` / `_bend60` | 1 / 2 | bounded deviation from the map (D24) |
| `right_fork_R5_v5` | 5 | v5's pose (D24) |
| `right_seam1_bend_gt45` | 5 | seam-1 bend over 45° |
| `ica_ap_gt40_R` | 3 | RICA terminus > 40 mm posterior |
| `left_fork_substituted` | 22 | D1 substitute |
| `left_pose_map_bounded_dev` | 29 | left fork 5–43° off the map (gates pass); derived, in `BUILD_v6.json` → `per_anatomy.<name>.derived_flags` |
| `orient:va_target_20_30` / `orient:D23` | 11 / 5 | the orientation gates' accepted flags (D8 / D23); derived, as above |
| `left_pose_not_map` | 8 | left fork not on the map |
| `eca_tip_skipped:right` / `:left` | 20 / 36 | ECA tip extension dropped (D4 and the other reasons in *Siphons and ECA tips*) |
| `route_baseline_v4_no_tip` | 36 | D13 |
| `route13_moved` | 1 | item b |
| `c4_planned` | 8 | the topcow_mr_021 pinch tube |
| `va_right_rung1/2/3/host`, `va_left_rung1/2/host` | 198 / 12 / 1 / 12, 215 / 3 / 5 | D9 rung per side |
| `va_*:d9_rung2_v5_junction` | 12 R / 3 L | D18 |
| `va_right:d9_rung3_v5_top_as_shipped`, `D23_tierA_accepted:RVA_top` | 1 | D23 |
| `va_*:d16_tube_bridge` / `junction_tube_bridge` | 8 R, 10 L / 8 R, 12 L | D16 / tube-bridged ostium |
| `va_*:d17_tube` | 15 R / 24 L | D17 tube to 30 mm past the ostium |
| `va_*:va_target_20_30` | 9 R / 2 L | D8 |
| `va_*:corridor_fallback_host_va` | 7 R / 6 L | D10 fallback |
| `va_*:corridor_medial_excursion` / `lateral_excursion` | 5 R, 2 L / 117 R, 93 L | D10 tails (recorded) |
| `va_*:corridor_fail_inherited_v5` | 7 R / 1 L | rung-2 / 3 side keeps v5's course |
| `va_right:course_rms_donor_gt20_inherited_v5` | 2 | the same |
| `va_right:route_guard_best_effort` | 3 | VA pose limited by the right-route guard |
| `fix_loop_va_hard` | 3 | the joint fix loop re-placed a fork against the final VAs |
| `d22_coverage_lt_0.99_prebake` | 2 | the composer's coverage estimate, reported |
| `d26_d22_fork_reference_own_surface` / `siphon` | 18 / 3 | D26 (4 of them new with s3_fix's bridged islands) |
| `d27_d22_siphon_repair_window` / `d27_d22_fork_repair_window` | 1 / 2 | D27: capsule-clipped cells reported, not gated (k_011_left mr_012_L siphon; w_002_right mr_013_L and w_008_right mr_003 forks) |
| `d27_coverage_<piece>_real_wall_loss_<p>pct_accepted` | 9 fork / 9 siphon (17 anatomies) | real-wall deficit > 0 and < 1 % against the v4 / v5 shipped wall, accepted (D27): 8 at 0.1–1 % (s3_fix; `<p>` to 2 decimals), 10 below 0.1 % (s4_records; `<p>` to 3 decimals, detail in `v6_checks.s4_records`) |
| `d27_bake_lattice_shift_<sx>_<sy>_<sz>` | 5 | s3_fix: re-baked on a shifted lattice (bake_v6 1.2 L1) |
| `d27_bake_route_island_bridge_<n>` | 4 | s3_fix: route islands bridged (bake_v6 1.2 L2; 8 islands) |
| `obj_fold_fix_spike_vertex_removal` | 1 | s3_fix: the OBJ fold-over pair removed (m_022_left mr_021_L) |
| `d23_tierA_accepted` | 1 | D23 |
| bake flags | 0 | |

Set-level flags (`results/v6_final_summary.json` → `flags_set_checks`): SOFA RCCA not past the bifurcation 4
(w_042_left), 5 after s3_fix (+ w_002_right mr_013_L, *SOFA*; `sofa/rcca_v6_s3/`); no seed reaching the target 40 (41 after s3_fix: w_002_right mr_013_L and w_012_left mr_003_L lost theirs,
w_008_right mr_003 gained one); geometric VA probe not entering 54 sides (v5 81); fewer entering SOFA
seeds than v5 on 4 of the 25.

## How it was built and verified

1. **Stage 1** (2026-10-03 / 04): the plan, the orientation-gate library (its self-test reproduces the audit's counts
   on v5, v4 and v3: `gates/selftest.txt`), the heading evidence, the placement tools, and a 14-anatomy pilot. The
   pilot verifier (`notes/verify_v6_pilot.md`) reproduced the numbers and found what no gate covered: ICA termini far
   posterior (→ D11), donor RVAs running toward the midline (→ D10), a missing VA-top rule (→ D8, D9), the k_011 route baseline (→ D13),
   SOFA entry as noise at that n (→ D14). The orchestrator took D8–D15.
2. **Stage 2 hardening** (2026-10-04): the carotid tool (forks re-posed, D19 / D20 after the user's decisions), the VA
   tool (D9, D10, D16, D17), the wall and bake (D15, the subclavian stretches, bake_v6). Notes: `s2_harden_*.md`.
3. **Integration and the first gate** (2026-10-06): joint pass, re-pilot, a dry run on all 223. The gate verifier said
   NO-GO (`notes/verify_s2_gate.md`): the relief gate fails moved forks whose surface is unchanged, one tier-A conflict,
   the ia172_069 D17 loop, 1 mm joins checked only at written vertices. → D22–D25.
4. **Second integration** (2026-10-06 / 07): one frozen toolchain, the DRY re-run on all 223; the second gate verifier
   said GO, with conditions (`notes/verify_s2_gate2.md`).
5. **The full build** (2026-10-07, `notes/s2_build.md`): compose, one bake, checks. 4 bake fusions and 10 D22 fork
   failures were diagnosed and fixed by versioned tools (compose_v6 3.2.0 U-TG and the RECA-tip feedback; graft_v6_va
   1.4.2 fold gate) or resolved by the user (D26, then D27 for the last residual). Then the copy into `anatomies_v6`
   and a load of every anatomy.
6. **Set checks** (2026-10-07, `notes/s2_checks.md`): split, loader, SOFA on all 223 RCCAs, D14, the polyline lumen and
   the orientation gates re-run on `anatomies_v6` itself.
7. **Records** (this file, `BUILD_v6.json`, `_records/`, `carotid_tools/v6/`; `notes/s2_records.md`).
8. **The final verifier** (2026-10-07, `notes/verify_v6_final.md`) reproduced every gate with its own code and said
   NO-GO on one point: real scanned wall lost on 8 right-route pieces (1.3–4.9 %; v5's mesh has it), plus two record
   errors and one fold-over pair on a training OBJ.
9. **s3_fix** (2026-10-08, `notes/s3_fix.md`) under D27: the coverage gate in checks_v6 1.6, the cause found (thin scan
   slivers the bake cut loose and dropped), 8 anatomies re-baked with bake_v6 1.2, 8 small deficits flagged, the fold
   removed, every changed anatomy re-checked (all parts), re-run in SOFA, re-stamped, copied with hash verification and
   re-drawn; these records regenerated.
10. **The second final verifier** (2026-10-08, `notes/verify_v6_final2.md`) reproduced every s3_fix claim with its own
   code (coverage on all 223 against both shipped references, strict audit, D22, loader, SOFA, split, hashes, the build
   record) and found no geometry problem. It asked for flags on the 10 pieces with a loss below 0.1 % and listed stale
   numbers in this file.
11. **s4_records** (2026-10-08, `notes/s4_records.md`), record only: the 10 flags stamped into provenance,
   `BUILD_v6.json` regenerated and re-checked, the folder manifest and `_records` updated, this file corrected.

## Record corrections

These correct statements in earlier records; CAROTID_V5.md and the v5 copies are not edited.

- **The frame match pins the roll only** (`TOPBRAIN_PIPELINE.md`, `CAROTID_THREE_SOURCE.md` describe it as fixing the
  orientation). Tilt and heading follow the anchor tangent (audit section 2). v6 poses every piece anatomically instead.
- **The `graft_splice` docstring** (`carotid_tools/v5/graft/graft_v5_va.py:403-404`) says the TopBrain VA top is
  "frame-matched (tangent + superior)". v5 placed it with a Kabsch fit onto the donor's distal VA, with no superior
  term, and the "start pinned" option pins the translation only. The copy in `carotid_tools/v6/v5_deps/` is byte-faithful
  and keeps the stale text.
- **Narrow VAs.** CAROTID_V5.md first listed 3 VAs below 0.65 mm on control points; on the polyline there were 4
  (corrected there by the audit). v6 has none.
- **BUILD_v5.json's stale fields** (`donors.table` held assignment-time counts; `checks_final.sofa.note` called the SOFA
  question pending after the user decided it on 2026-10-03). `BUILD_v6.json` is generated from the final provenance
  only and checked count for count.
- **This file's first version (2026-10-07) understated the lost wall.** It listed 4 right-route pieces whose v4 wall did
  not reappear (coverage < 0.99, "reported, not gated"). Measured against v4's and v5's SHIPPED meshes the real scanned
  wall lost was ≥ 1 % on 8 pieces: the 4 listed plus k_011_left mr_012_L, w_002_right mr_013_L, w_012_left mr_003_L and
  w_008_right mr_003 (forks, 1.1–2.1 %), whose D22 coverage had read 1.0 because D13 measured it against v4 rebuilt
  without its RECA tip, which hides the same wall (final verifier, `notes/verify_v6_final.md` section 6; reproduced by
  checks_v6 1.6 on the pre-fix folder, `checks/coverage_v1p6_prefix/`). All 8 are fixed (*The right route*). The old
  D22 `coverage` field stays in the records as it was measured (against the D13 baseline); the gate is the new one.
- **"Every flag is written in the anatomy's provenance"** (first version, *Flags*) was not true for two flag classes the
  build collection derived outside provenance: `left_pose_map_bounded_dev` (29) and `orient:<accepted orientation flag>`
  (16). They are now in `BUILD_v6.json` → `per_anatomy.<name>.derived_flags` for all 223 and in provenance
  (`v6_checks.derived_flags`) on the 17 anatomies s3_fix re-stamped; the other provenance files were not rewritten.
- **This file's s3_fix version (2026-10-08)** had stale numbers, found by the second final verifier and corrected here:
  SOFA RCCA not past the bifurcation is 5 after s3_fix, not 4; w_002_right mr_013_L's seed went into the ICA and
  buckled, it did not turn into the ECA; the handle windows in the final folder are 25 (fork) / 32 (siphon), not 28 /
  31; provenance gives the s3_fix re-bakes 174–645 s, not 177–647 s (the bake markers' times); the undecimated unions of
  the 8 re-baked anatomies are in `v6_work/s3fix/iso/final/`, not `full/iso/` (also `BUILD_v6.json` →
  `records.undecimated_unions`); the bridge tubes are not the only synthetic surface on the moved right route (the D27
  capsule repair windows are bake capsule); 2 of the 25-set anatomies had been re-stamped by s3_fix (provenance only);
  and only 8 of the 18 pieces with a measurable real-wall loss were flagged.

## Known limitations

- **Small real-wall deficits accepted (D27).** 18 right-route pieces (17 anatomies) miss part of the real scanned wall
  that v4's / v5's shipped mesh showed, each less than 1 %. 8 miss 0.14–0.77 % (w_035_left mr_015_L siphon 0.77 %,
  w_012_left mr_001 fork 0.60 %, w_012_left mr_013_L fork 0.56 %, w_025_right mr_021_L fork 0.46 %, w_008_right mr_001
  fork 0.43 %, w_017_right mr_011 siphon 0.42 %, w_047_left mr_027 siphon 0.22 %, w_034_left mr_011 siphon 0.14 %). 10
  miss less than 0.1 % (w_046_left mr_026 fork 0.097 %, w_012_right mr_012 siphon 0.070 %, m_024_left mr_023_L fork
  0.057 %, w_034_left mr_008_L fork 0.056 %, w_042_left mr_010_L siphon 0.045 %, m_024_left mr_001 fork 0.045 %,
  w_051_right mr_025_L fork 0.015 %, w_012_left mr_013_L siphon 0.013 %, m_006_left mr_022_L siphon 0.008 %,
  w_018b_left mr_002 siphon 0.005 %; against the union, as the rule measures; w_046_left mr_026 and w_012_right mr_012
  read 0.124 % and 0.115 % against the shipped, decimated collision_full). Nine of these ten are one or two reference
  cells 0.31–0.59 mm outside the v6 wall, at the level of decimation noise. **w_034_left mr_008_L's fork is a real
  patch**: 0.82 mm² of the carotid bulb at RCCA s 69.0–69.75, just proximal to the external carotid take-off, 1.1–1.4 mm
  outside the v6 wall against both references. Each of the 18 is flagged `d27_coverage_<piece>_real_wall_loss_<p>pct_accepted`
  in its provenance; the 10 also carry the measured values in `v6_checks.s4_records`
  (`results/s4_records/coverage_sub01_pieces.json`). The 8 pieces that lost 1 % or more were re-baked (*The right route*).
- **Synthetic bridges on 4 anatomies.** On the 4 tip-class pieces the scanned sliver is separate from the lumen in the scan
  itself; v4 and v5 showed it only because their RECA tip's tube ran through the gap. bake_v6 1.2 joins each such island
  with one tube of 0.5 mm radius and 1.1–2.7 mm length (`bake_v6.route_island_bridge` in provenance; flagged
  `d27_bake_route_island_bridge_<n>`). These tubes are not the only synthetic surface on the moved right route: the D27
  capsule repair windows (k_011_left mr_012_L siphon 17.79 mm², w_002_right mr_013_L fork 12.01 mm², w_008_right mr_003
  fork 56.57 mm²) are the bake's capsule where the scan reaches past it, not scan, and the bake's tube windows
  (*Collision mesh*) are tube, as in v4 / v5. On w_008_right mr_003's fork, capsule (33.6 mm²) and the surface next to
  the bridges (7.1 mm²) make up 2.0 % of the D22-gated surface; it passes D22 through the D27 repair window and is
  flagged (*The right route*).
- **The training route moved.** RCCA target reach on the 25-anatomy set fell from 46 to 36 seeds (v5 → v6; siphon 54 →
  52), with the re-posed and mirrored right forks; no geometry in those 25 changed after the build (3 were re-stamped
  for a flag, provenance only), so the numbers stand. On all 223, 41
  anatomies have no seed reaching the target after s3_fix (40 before: w_002_right mr_013_L and w_012_left mr_003_L lost
  their one target seed, w_008_right mr_003 gained one). Reach is the scripted controller's, not a gate; the route is
  navigable and sound on all 223.
- **w_042_left** (mr_005_L, mr_010_L, mr_023, mr_026_L): the scripted wire never passes the right bifurcation, as in v4
  and v5. A property of that Zenodo fork.
- **VA entry** improved on v5 (48 against 33 of 75 seeds) but stays below v4's single host junction (74 / 75); 4
  anatomies enter less often than in v5 (m_030_right mr_013_L 2 → 0, w_007_left mr_021_L 1 → 0, w_003_left mr_010 3 → 1,
  w_036_left mr_011 2 → 1). As in v5, these are 23 different real junctions, not one host junction under a schedule
  tuned on it.
- **Bounded and flagged poses.** 5 right forks keep v5's pose, 3 have a seam-1 bend of 55.6–55.7° and 3 a bounded
  deviation; 29 left forks sit 5–43° off the map and 8 are off it; 3 RICA termini are 45–53 mm posterior. All flagged.
- **15 VA sides keep v5's junction and 1 v5's top** (D18, D23), with their v5 course (corridor and course-RMS flags
  inherited); **17 VA sides have no donor** (D21, as in v5).
- **1 mm turns over 60° kept from v5.** 33 curves have a turn over 60° at a new point (up to 82.1°: RECA / LECA tip
  joins and VA tops). Each is a rigidly moved piece keeping a turn the same curve already had in v5 (within 0.6°), so the
  rule passes it; they are listed in `results/v6_checks.json` → `per_anatomy_gates.geom`.
- **Donor-side tools not fingerprinted.** The tools that made the donor walls, junctions and D17 paths ran in stage 1 and the hardening
  stage before the toolchain was frozen. The donor VA surfaces they made are pinned by md5 in each provenance
  (`xform.vaC_*.surface_md5`) and the junction surfaces by path, but the tools themselves
  were not fingerprinted when they ran (`carotid_tools/v6/MANIFEST.md`).
- **A record nit.** On the 16 sides that use ia172_015 R's junction, `v6_build.va.<side>.d17.tube_mm` reads 0.664 mm,
  but no tube was written: the branch ends at the label end, 29.3 mm past the ostium (`v6_va.record.sides.<side>.d17.
  positions.tube_mm` = 0), so the side is not flagged `d17_tube`. `BUILD_v6.json` counts the tubes as written.
- **Undecimated unions** are kept only in the git-ignored work folder: `carotid_data/v6_work/full/iso/<name>.vtp`, and
  for the 8 anatomies s3_fix re-baked `carotid_data/v6_work/s3fix/iso/final/<name>.vtp` (`full/iso/` holds their
  superseded pre-fix unions; `s3fix/iso/final/` also holds the 9 other s3_fix anatomies' unions, identical to `full/iso/`).
- **Git.** `anatomies_v6` falls under the `.gitignore` rule `carotid_data/*` (anatomies_v5 was force-added), so tracking it
  needs `git add -f`.

## Follow-ups

- **The bake's handle rule and thin scan slivers.** The shrink step accepts any genus-0 window, also one that cuts a thin
  real sliver loose (s3_fix's window class), and keep_largest then drops it silently. A bake-time guard (refuse a window
  that drops real route surface; bridge route islands by default) would catch the next case at bake time instead of in
  the coverage check.
- **Donor VA colour in the figures** (item f; user: later). The v6 figures (`monitoring/figure_carotid_anatomies_v6.py`,
  `saved/figs/carotid_v6/`, front and side views) still draw the donor lower VA dark teal (`DONOR_C = "#00332e"`),
  near-black on the page. Pick a lighter, distinct colour.
- **The 17 VA sides with no donor** (D21; user: later).
- **VA entry**: a scripted schedule tuned on real junctions, or a run that starts inside the VA, if VA targets are to be
  trained.

## Records

- `provenance.json` keeps v4's and v5's records as history (`v5_va`, `bake_v5` describe v5's VA and bake) and adds:
  - `xform.lower` / `lower_L` with `placement`, `mirror`, `chirality` (D20), the blend (`blend_mm`, `blend_disp`) and
    `v6` (rung, pose source, bend, shift, deviation); `xform.vaC_*` (donor VA surface and md5), `scC_*` / `scD_*` /
    `scJ_*` (the donor's subclavian and junction; D17 cut end);
  - `v6_right_fork`, `v6_carotid` (every carotid decision, the left-fork ladder, ICA AP offsets, route baseline, checks),
    `v6_va` (the VA tool's full record per side), `v6_build` (the stage summary, `flags`, the toolchain md5s, the D23
    block), `bake_v6` (rung, windows, morph, chirality, C4, E7; on the 8 s3_fix re-bakes also `lattice_shift_mm` and
    `route_island_bridge`; on m_022_left mr_021_L `obj_fold_fix`), and `v6_checks` (the final gate verdict, the D22 record
    per piece, the check-level flags, the final manifest md5; on the 17 anatomies s3_fix re-stamped also
    `coverage_v4_v5_shipped`, `folds`, `derived_flags` and `s3_fix`, what changed and why; on the 10 anatomies
    s4_records re-stamped also `s4_records`: the flagged piece's loss per reference and wall, the lost stretch and depth,
    checks_v6 1.6's value for it, and the flag).
- `mesh_v6.json`: the bake report (v5's schema plus the v6 fields).
- `LICENSE_v6_donors.txt` in every folder; `LICENSE_VMR_0248.txt` in the 7 VMR anatomies.
- `BUILD_v6.json`: generated by `build_v6_record.py` from the final provenance only, with a digest of the 223 provenance
  files; checked by `check_build_record.py` (build_v6_record 1.2 / check_build_record 1.2 after s4_records: 123 checks,
  0 differences; `_records/checks/build_record_check.json`). 1.2 changes only text in both (the s4_records entries and
  `records.undecimated_unions`), no count.
- `_records/`: `results/` (the final summary, set checks and build collection, dry runs, pilot, wall results, the copy
  and bake-reuse records), `checks/` (split, loader, orientation, lumen, entry geometry, identity, route spot checks,
  coverage diagnosis, the record check), `gates/` (gate self-test, heading evidence, Zenodo frames), `per_anatomy/`
  (the final checks record per anatomy, the pre-bake tube-genus scan), `sofa/`, `diag/` (D26, D27 and the bake-fusion
  diagnoses), and `notes/` (plan, decisions, build log, every stage's and verifier's notes, the audit's agent notes).
  s3_fix added `results/v6_coverage_s3.json`, `results/s3_fix/` (the folder manifest after the fix, the copy, the
  re-stamp, SOFA, split, loader), `per_anatomy/checks_s3/`, `bake_s3/`, `fold_fix/`, `sofa/rcca_v6_s3/`,
  `checks/coverage_v1p6_prefix/`, `diag/s3_fix/` and `notes/s3_fix.md`, `notes/verify_v6_final.md`.
  s4_records added `results/s4_records/` (the 10 pieces with the second verifier's values, the re-stamp summary, the
  folder manifest after it and its check, the verifier's coverage summary on all 223, the scripts including
  build_v6_record / check_build_record 1.2), `notes/s4_records.md` and `notes/verify_v6_final2.md`.
  `_records/README.md` lists them.

## Reproduce

The toolchain ran from a session scratchpad and is copied to `carotid_tools/v6/`, one folder per stage (`build/`,
`donor/`, `gates/`, `item_b/`, `set_checks/`, `records/`, `s3_fix/`, `v5_deps/`). `carotid_tools/v6/MANIFEST.md` maps
every file to its stage and run, lists the eight toolchain manifests and the frozen earlier versions, and lists the
hard-coded paths a re-run must change. `carotid_tools/v6/restore_layout.sh <X>` rebuilds the build-time layout byte for
byte (the scripts call each other as `/sp/v6/<path>`); mount `<X>` at `/sp` and the repo at `/repo`. On the restored
layout the final manifest (`TOOLCHAIN_s3fix.md5`, 2288f778) passes `md5sum -c`.

Images: eve-training-fixed (compose, VA tool, loader, SOFA, split), neve-build-meshlab (bake, checks), neve-build-stagea
and neve-build-vmtk (donor walls and centrelines), as in CAROTID_V5.md. The build order is in the MANIFEST
(*The build order, as it ran*). The final manifest is `TOOLCHAIN_s3fix.md5` (2288f778): the s2 manifest's files with
bake_v6 1.2 and checks_v6 1.6, the frozen 1.1 / 1.5 copies and s3_fix's scripts (`carotid_tools/v6/s3_fix/`); the
s2 manifest (d65b7209) is kept and resolves to the frozen copies. s4_records ran no build tool; its scripts (the flag
stamp, the copy, build_v6_record / check_build_record 1.2) are in `_records/results/s4_records/scripts/`, and
`carotid_tools/v6/records/records/` holds 1.2 (1.1 kept as `*_v1_1.py`; added 2026-10-09, restore test 255/255).

## Loading

`DualDeviceNavTopBrain(anatomy_dir="carotid_data/anatomies_v6", ...)` loads v6 anatomies exactly as it loads v4 and v5
ones. Over the whole folder: 223 anatomies, 18 branches each, the shared insertion inside every OBJ, RCCA targets on
every anatomy (`checks/loader/loader_set_v6.json`). Targets are drawn on the RCCA as before; the RCCA's course beyond
seam 1 is new.
