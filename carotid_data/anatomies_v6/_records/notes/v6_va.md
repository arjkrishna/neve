# v6_va notes -- the VA side of v6 (V6_ITEMS e + a4; decisions D2, D3)

Agent v6_va, 2026-10-03. State: `state/v6_va.md`. Code: `scratchpad/v6/va/`. Outputs: `D:\Arjun\neve\carotid_data\v6_work\va\`
(`/work` in the containers). Nothing under `anatomies_*`, `topbrain_tools/`, `carotid_tools/` was written; `/repo` is
mounted read-only.

## Answer so far

Status 2026-10-04 03:25: steps 1-4 done. **Tool**: `va/graft_v6_va.py` v0.8 (frozen as `graft_v6_va_dry10.py`;
writer `graft_v6_va_dry10w.py`), junction records `va/sc_post3.py`, paths `va/subclavian_paths.py`, bake hook
`va/bake_sections_v6.py`, checks `va/pilot_check.py`, summaries `va/dry_summary.py`, `dry_summary2.py`,
`fail_clusters.py`, `cmp57.py`; runners `va/run_dry.sh <tag>`, `va/pilot_write.sh <tag>`.
- (e) the donor's own subclavian junction on **359 / 429 donor sides** (take-off vs native median 0.08 deg; no stub;
  right route point-identical up to seam 1 on all 168 right donor junctions); 67 keep v5's junction (reasons in s3),
  3 have no placement, 17 have no donor (v4 VA kept since v5).
- (a4, D2/D3) anatomical VA tops (A0 + bend <= 45) on 441 sides, D2 target <= 20 mm on 421; no fusion, no crossing,
  no touch (min 0.61 mm); 410 / 446 sides and 195 / 223 anatomies pass every gate; one-to-one assignment unchanged.
- Pilot: 14 anatomies written to `carotid_data\v6_work\va\pilot\` (19 donor junctions / 28 sides, 26 pass; tree-mode
  orient gates 0 tier A on the VA tops).

## 1. Donor subclavian paths (step 1a)

`va/subclavian_paths.py` -> `/work/subclavian/paths/<donor>_<side>.json` (54 sides = 27 units x R/L, incl. the two
reserves ia172_172 and Song 138, and VMR 0248). Per side, in the donor's own RAS frame: the subclavian centreline from
its origin node through the VA ostium foot O_v to ~45 mm past it (0.5 mm), Arterial radius, `s_ostium_mm`, the
TotalSegmentator lumen exit (`ts_origin`: where the subclavian mask touches the brachiocephalic trunk / aorta mask),
what the v5 ct_seg run already held for that subclavian, and the CT field of view.

- Source: Arterial labelled centrelines, the chain of RSA / LSA segments from the segment nearest O_v back to the node
  touching BT / AA (the brachiocephalic split, the arch) and forward along the widest continuation. 52/54 sides.
  - song_AHMU1218002 R: no RSA label -> the record's subclavian (+-40 mm) joined to the TS origin.
  - song_AHMU1218095 R: the medial right subclavian is an unlabelled ('other', r 4.4 mm) Arterial segment and never
    reaches the BT label -> walked through wide 'other' segments; origin not reached (22.4 mm before the ostium; flagged).
  - VMR 0248: the expert SimVascular subclavian paths (origin = path start). The model covers both subclavians from
    the arch / BT; the CT (p12full.vti) covers the left one 99 % and the right one 88 % (the BT end lies ~10 mm below
    the CT), and the wanted stretch is inside the CT on both (want_inside_frac 1.00 L, 0.98 R).
- Origin -> ostium along the path: right 24-72 mm, left 49-92 mm. The TS lumen exit sits 4-48 mm along it (the
  Arterial origin node is on the parent's centreline, inside its lumen).
- Every wanted stretch [exit + 10, ostium + 20] lies inside the donor CT (fov want_inside_frac 1.0 on 53/54).
- **The v5 ct_seg subclavian labels cannot be reused as they are**: they cover +-30 mm about the VA origin on ONE
  Arterial segment, which on the left is often a short stub (ia172_058 L: 7 mm, s 43-50 of a 49 mm origin->ostium).
  So the subclavian wall is re-segmented on the extended path (step 1b).

## 1b. Subclavian + junction wall (requested from v6_wall)

The wall agent's `ctseg6.py` (v5's ct_seg4 + the c1 bone-side fix) has a request mode for this item
(`wall/subclavian_answer.json`). `wall/subclavian_requests.json` (status READY) holds the 54 entries in its schema
(centreline, radius, s_from = exit + 10 - 4, s_to = ostium + 25). It writes per donor
`carotid_data/v6_work/wall/subclavian/<donor>/{r,l}junction_stageA.vtp` (subclavian label + the VA's first 25 mm as
one closed surface, stage-A protocol) and the VMTK MISR line. The wall agent ran all 27 units (both sides, incl.
VMR 0248 through vmrseg6) by 22:01 with its c1 plateau rule.

**Junction surfaces used: the wall agent's v3** (published 2026-10-04 00:56 over `wall/subclavian/<donor>/`, marker
`.junction_v3`; records via `va/sc_post3.py` -> `/work/subclavian/seg/v6va3/<donor>/<R|L>/junction.json`, 54 sides).
- History: their first junction surfaces were subclavian-only on 23/54 sides because the VA label (planar cut at its
  first station) did not touch the subclavian label (gap 0.5-4.4 mm). I reported it (21:19) and meanwhile built my
  own union with an ostial capsule (`va/sc_post2.py`, `/work/subclavian/seg/v6va`; capsule on 24 sides, 4.5-9.7 mm;
  genus median 7.5, > 10 on 20 sides). v6_wall then fixed it in ctseg6 (`jfix/junction6.py`): a CT-lumen bridge in
  the ostium crotch (capsule subclavian line -> VA first 6 mm, both vessels' thresholds) plus a 1-voxel opening +
  closing of the subclavian label (removes the 1-voxel bridges to neighbours). Their v3 supersedes my v6va surfaces.
- v3 on 54 sides: **genus 0 on 29, <= 2 on 49, > 5 on 3 (song_008 L 11, song_094 L 9, ia172_172 L 8)**; wall-detail
  proxy (surface points of the subclavian stretch > 0.3 mm off a local-median tube, ostium +-6 mm excluded) median
  0.515 (0.30-0.87), coarse (D5); ostium bottleneck / VA inscribed radius median 0.85.
- **Junction failed -> v5's junction kept**: ia172_071 L (bottleneck 0.54 mm = 0.25 of the VA radius; the VA station
  line leaves the lumen at s 6-8 mm, wall agent) and ia172_164 L (0.59 mm = 0.46; my rule: < 0.5). Tube bridge (the
  wall agent's flagged fallback, a few mm of tube in the crotch) used and flagged on ia172_136 L and ia172_172 L.
- Coverage: the wanted stretch [TS exit + 10, ostium + 20] lies inside the label on 53/54 sides (song_109 R ends
  1.0 mm short).
- **Calibre**: graft_v6_va declares the stretch radius as the junction mask's inscribed radius on my path, clipped to
  [0.7, 1.15] x Arterial's, and flags `narrow_label` when the label/Arterial median is < 0.7. On v6va (pre-v3) labels
  the narrow sides were song_032 L 0.48 / R 0.57, song_040 R 0.57, vmr0248 L 0.62 / R 0.46, song_008 L 0.67
  (`va/misr_cmp.py`); the dry-run records carry the v3 ratio per side (`junction_wall_info`).
- A 1-voxel closing of my own union (`V6VA_CLOSE=1 sc_post2.py`, 5 donors) only halved the genus (vmr R 57 -> 26):
  the wall agent's label-level clean was the right fix.
- **VMR 0248**: its expert model covers BOTH subclavians from the arch / brachiocephalic trunk; the CT covers L 99 %,
  R 88 % (the wanted stretch fully), and v3 junctions exist for both sides (genus 1). So VMR has a subclavian; its
  anatomies get the donor junction where the gates pass (no blanket "keep v5" for VMR).

## 2. The graft (step 2): `va/graft_v6_va.py`

Built on graft_v5_reg (load_side with the follow handedness rule, splice_window, calibre_ratio, advanced), the
prototype graft_v5_va (Neighbours, gate_va, self_gaps, at_frac, rot, d_chord), graft_v4 (join_kink, tb_part),
graft_three (ramp_radius) and `gates/orient_gates.py` (A0, bend, va_targets, measure_candidate, pair_metrics,
evaluate), all imported unchanged. Per anatomy it reads the v5 anatomy (carotid side, host body, the one-to-one donor
assignment from `provenance.v5_va.<side>`), and per side:

- **Host seams.** Seam 1 at the host subclavian's lumen exit + 10 mm: right RVA s 16.0 (exit 6.0 from the BT/RCCA
  lumen), left (18) s 22.7 (exit 12.7 from the arch). The kept host stretch is byte-identical (checked:
  `final.<side>.rva_prefix_identical`). Seam 2 = a Hermite bridge from the donor stretch's end E (ostium + 20 mm) to the
  host distal subclavian ((17) / (19)), whose remainder is kept byte for byte.
- **Donor stretch.** The donor's own path from its lumen exit + 10 mm (at least 12 mm before its ostium foot) to the
  foot + 20 mm, 1 mm; radius = the junction wall's inscribed radius when present, else Arterial's.
- **One rigid pose for the stretch and the donor VA** (so the VA leaves at its native take-off exactly): pinned
  Procrustes about seam 1. Bases: W_VA 2 / 6 / 15 / 0.5 (stretch vs the host's OWN VA at 21 normalised points), and
  "aim" (D2): + the donor's length-matched splice point onto S* = target - A0 (Q_end - Q_start), weight 20 / 80 / 300;
  each perturbed by +-3..15 deg about x / y / z and rolled +-10..40 deg about the stretch chord.
- **Junction path** (no synthetic stub): along the subclavian axis to the foot, up the untraced ostial VA (foot -> the
  VA trace's first point; 3-11 mm), then the VA trace; corners smoothed (Gaussian sigma 1 mm within +-3 mm, ends
  pinned). Gated: 1 mm turn <= 60 deg, inside the union of the subclavian tube / ostial VA / VA tube.
- **TopBrain top (D2/D3).** R = A0 (anatomical: superior and heading pinned); if the 5 mm seam angle > 45 deg, the
  minimal bend capping it at 45 (bend <= 45, else rejected). Candidates over the splice (the donor's [C2, dura] window,
  extendable 40 mm below C2 / up to the donor end with a ranking penalty) and the TopBrain start trim (v4's 0-5 mm +
  8/11/15 mm, penalised), ranked by the distance of the piece's end from its D2 target (+ for the left side the pair
  error relative to the placed right end), then fully gated (join <= 60 on the 1 mm curve, calibre <= 1.25 / 1.6,
  clearance via gate_va with the junction-aware exit limit).
- **Subclavian placement gates** (patches 23-26): the donor stretch against the host subclavian corridor at the same
  arclength from seam 1: RMS <= 12 mm (`sub_rms`) and the donor stretch past its ostium vs the host distal subclavian
  direction <= 60 deg (`arm_dir`). A "sub" pose family (stretch fitted to the corridor alone, then rolled through the
  full circle about the corridor chord, +-10 deg tilts) was added; poses are tried corridor-direction-first, then by
  the aim error; at most 220 per side.
- **Rise re-referenced to native** (patch 26, reversible): the v5 gate (VA 10 mm chord >= 15 deg) stays, except for
  donor sides whose OWN VA rises < 15 deg in its CT (ia172_069 L 2.0, ia172_159 L -7.4, song_032 L 12.3 deg;
  `va/native_rise.py`): there the floor is native - 10 deg. With a native take-off the first 10 mm of these VAs is
  real anatomy that no rigid pose keeping the subclavian on its corridor can raise.
- **Course RMS re-referenced** (D2, patches 27-28): the reference is the host's own VA (O_h -> confluence) with its
  end carried onto the D2 target (linear in normalised arclength), 21 points; gated on the ABSOLUTE RMS <= 20 mm,
  because the composite now starts at the donor's own ostium (moved along the host corridor by design, median
  ~10 mm), and pinning at the start would add that shift to every point. The pinned value and the value against the
  unmodified host end are recorded too.
- **Donor VA records** (v0.3): the v6_wall c1-fixed walls (`/repo/carotid_data/v6_work/wall/donors/plateau`, same
  schema as donors_v5, wall/donors/CHOSEN.json), so the donor VA centreline, radii and surface in every placement are
  the plateau-rule ones (env `V6VA_DONOR_ROOT` switches back to donors_v5).
- **Junction walls** (v0.4): the v6_wall v3 junction records (`/work/subclavian/seg/v6va3`); a side marked
  junction_failed (ia172_071 L, ia172_164 L: ostium bottleneck < 0.5 of the VA's inscribed radius) keeps v5's
  junction. The VA clearance gate in the search asks 0.65 mm against the OTHER branches (SEARCH_CLEAR_MARGIN, v0.6) so
  that the exact final gate (pair gaps) holds 0.6: in dry5 two right VAs passed the search at 0.60 and missed the
  final gate by 0.01 mm (w_016_right mr_010 vs RECA 0.586, w_022_right mr_011 vs RCCA 0.590). (dry7 applied the margin
  to the VA's own parents after exit too and lost 88 donor junctions -- a native VA runs alongside its own subclavian
  at 0.6-0.65 mm; v0.6 restricts it to the other branches; the 8-anatomy check wt11 recovered all 16 sides.)
- **Seam-2 bridge choice** (v0.5): of the bridge lengths 6-35 mm whose two 1 mm joins are <= 60 deg, the smoothest
  (largest 1 mm turn from 3 mm before E to 3 mm past H, + 0.5 deg per mm of bridge) wins instead of the shortest; the
  shortest made S-hooks on the arm side when the donor's post-ostium direction and the host's distal subclavian
  disagree (figures `/work/pilot_t/_fig/case_m_006_left__topcow_mr_027_L_zoom.png` vs the dry5 pilot figure).
  v0.7 adds a gate on the bridge's total turn (E tangent -> chord E->H -> H tangent) <= 90 deg: a donor subclavian
  arching above the host's (VMR 0248 L on w_003 mr_007_L) otherwise made an arm-side hairpin that still met the
  1 mm join limit (`/work/pilot/_fig` of the dry7 pilot vs `/work/pilot_t2/_fig`).
- **Selection**: every candidate passing all gates is scored (target + pair + 0.5 course RMS + 0.2 bridge + 0.1
  bend + 0.5 sub_rms); the search stops at target <= 10 mm and RMS <= 15 mm.
- **Stop-short (D2)**: after both sides, the two tops' distal ends are trimmed 1 mm at a time until their lumens are
  >= 0.6 mm apart (no fusion). The left VA is gated against the right VA minus its last 15 mm.
- **Fallbacks**: a donor side whose junction fails keeps v5's junction (stub, host subclavian) and only its top is
  re-placed on the v5-posed donor VA (extended past v5's splice); sides with no donor (v5 kept v4's VA) re-place the
  top on the host VA (extended with the host's own VA beyond v4's cut). Both flagged (`junction`). The fallback wins
  only if it passes and the donor junction does not; if neither passes, the donor junction attempt is kept (and
  reported as failing). On a v5-junction side the written vaC section keeps v5's pose but gets the v6 (plateau) wall
  of the same donor side and reaches the new splice (write-only patch, `graft_v6_va_<tag>w.py`).
- **Final gates** on everything as written: VA (gate_va + exact all-pairs from exit), the new subclavian parts (exit
  rule + clearance), orient_gates v6 set on the two tops (`orient.blocking`).
- **Bake records** (`--write`): xform va_R/va_L (placement v6_anat), vaC_R/vaC_L (donor VA, same pose), scC_*/scD_*/scJ_L
  (the junction surface on RVA/(17) or (18)/(19)/LVA, siblings); `va/bake_sections_v6.py` installs them into the bake
  (`sections_for_v6`); bake_v5b's is_va must include the (17)/(18)/(19) sections.
- Checked on one written anatomy: orient_gates tree mode back-maps the tops onto their TopBrain sources at 0.06 /
  0.18 mm median (anchor distance 0.0) and flags nothing on them (`/work/og/pilot_test`).

## 3. Dry run on all 223 (step 3)

Final run **dry10** = `va/graft_v6_va_dry10.py` (v0.8, frozen; = graft_v6_va.py), records only:
`/work/dry/dry10/<name>.json`, summaries `/work/dry/dry10_summary.txt` (dry_summary.py), `dry10_summary2.txt`
(dry_summary2.py: fallbacks, sources, pairs, fail list), `dry10_fail_clusters.txt` (fail_clusters.py),
`dry10_vs_dry5.txt` (cmp57.py). 223/223 ran, 0 errors, ~23 min on 2 x 4 workers.

| | dry10 (final, v0.8) | dry5 (v0.3, for reference) |
|---|---|---|
| donor sides with the **donor's own junction** | **359 / 429 (84 %)** -- ia172 186/232, Song 167/184, VMR 6/13 | 397 / 429 |
| v5's junction kept (fallback) | 67 | 32 |
| no placement | 3 | 0 |
| host VA (no donor, v4 VA kept) | 17 | 17 |
| sides passing every gate | **410 / 446** | 425 / 446 |
| anatomies passing on both sides | **195 / 223** | 206 / 223 |
| one-to-one donor assignment changed | **0** | 0 |

dry5 had my capsule-bridged junction walls, no bridge-shape gate and the 0.6 search clearance; dry10 has the v6_wall v3
junctions (2 donor sides failed at the ostium), the 90 deg bridge-turn gate (no arm-side hairpins) and the 0.65
search clearance against other branches. dry6-dry9 were the intermediate variants (notes s2).

**Why 67 sides keep v5's junction**: junction wall failed 20 (ia172_164 L 16 anatomies, ia172_071 L 4); no bridge
passes the 90 deg turn gate even at ostium + 10 mm 18; the donor junction passes the hard gates but not course RMS /
va_target where v5's junction does 26 (course RMS 10, va_target + course RMS 13, others 3); VA clearance below the
splice on every pose 2; donor VA course pre-filter 1.

**Donor-junction sides (359)**:
- take-off vs NATIVE: median **0.08 deg**, p90 0.29, max 1.09 (2 sides > 1 deg) -- one rigid pose for the stretch
  and its VA (v5: median 17.8 deg off native on 251 VAs, synthetic stub median 5.28 / max 10.49 mm). Native
  take-off median 75 deg (16-149).
- junction: no synthetic stub; donor stretch median 38.7 mm (24-62), of it before the ostium 20 mm (12-42); the
  junction path's largest 1 mm turn median 44.9 deg (max 54.6), inside the junction tubes by >= 0.63 mm.
- seam 1 (host subclavian lumen exit + 10 mm): 1 mm join median 16.9 deg (max 47.9); the right route is untouched up
  to it: **RVA prefix point-identical on 168/168** right donor junctions (RCCA, BT and arch curves never written).
- seam 2: bridge median 26 mm (6-40), E -> host gap median 10.8 mm (0.5-25), joins median 10.5 / 9.8 deg (max 25).
  E at ostium + 20 mm, shortened to 15 / 10 mm where needed (adaptive, recorded `past_ostium_mm`).
- donor ostium vs host ostium O_h: median 12.5 mm (2.7-23.8) along / off the host corridor.
- clearance as written (exact pair gaps from the exit): min **0.61**, median 1.0 mm; course RMS (re-referenced,
  absolute) median 15.9 mm; right VA lumen to the RCCA real stretch median 3.8 mm (min 0.67).
**VA tops (441 placed)**: D2 target <= 20 mm on **421 / 441** (median 7.6 mm); bend median 0.5 deg (max 44.5, all
<= 45); up-tilt median 1.1; **VA ends: 0 crossings > 10 mm, 0 touches < 0.6 mm (min lumen gap 0.61), no fusion**,
end separation minus native median 2.9 mm, va_gap_v6 flagged on 3 anatomies; stop-short trims 0 mm median (max 15).

**The 36 failing sides** (`dry10_fail_clusters.txt`) cluster by TopBrain patient: topcow_mr_013_L 10 (its native VA
ends are 36 mm apart, the targets sit +-18 mm off the confluence), mr_023_L 4, mr_026 4, mr_004 4, others 1-2. By
kind: 6 fail only va_target and end <= 30 mm (pass at the plan's D2 fallback threshold of 30 mm); 6 fail only course
RMS; 5 have no placement (w_003_right mr_011_L R, w_016_left mr_017_L R, w_022_right mr_026 R, w_037_right mr_010_L R,
w_051_right mr_008_L R); 19 fail on several (va_target > 30, va_lateral 7, va_gap_v6 on both sides of 3 anatomies).

## 4. Pilot VA side (step 4)

`bash va/pilot_write.sh dry10` -> **`D:\Arjun\neve\carotid_data\v6_work\va\pilot\<name>\`** for the 14 pilot anatomies
(same frozen code + the write-only vaC wall swap, `graft_v6_va_dry10w.py`; log `logs/pilot_write_dry10.log`):
`Centrelines_comb/` (changed: RVA, (17) on right donor junctions; (18), (19), LVA on left ones; every other curve
copied from v5), `provenance.json` (v5's + `xform.va_R/va_L` placement v6_anat, `vaC_*` donor VA in its pose with the
plateau wall, `scC_*` / `scD_*` / `scJ_L` the v3 junction surface on RVA/(17) or (18)/(19)/LVA with siblings,
`v6_va.record` = the full dry-run record, `v6_va.bake_sections`), LICENSE files; figures front / side / top + a
junction zoom per anatomy in `pilot\_fig\`. The carotid side is v5's.
- 28 sides: **19 donor junctions, 7 v5 junctions, 2 host**; 26 / 28 pass every gate. Failing: k_011_left mr_012_L
  right (v5 junction, va_target 21.6 mm) and w_018b_left mr_002 left (donor junction, va_target 28.3 mm; item b
  moves this VA next).
- Independent check `va/pilot_check.py` (`pilot\_pilot_check.json`): RCCA, (11), (0), (2) byte-identical in 14/14;
  RVA point-identical up to seam 1 on every right donor junction; no unexpected file changes; joins at the recorded
  seams <= 58.2 deg; VA clearance from 15 mm >= 0.65 mm to every other curve.
- orient_gates tree mode on the written pilot (`og\pilot_dry10\`): 28 VA tops, **0 tier A**, 2 v6 flags (the two
  va_target cases above) -- identical to the in-tool evaluation.
## Open issues

1. **36 failing sides** (dry10, s3) cluster by TopBrain patient (mr_013_L 10 -- native VA ends 36 mm apart --,
   mr_023_L 4, mr_026 4, mr_004 4). The plan's D2 fallbacks would help: a 30 mm target threshold passes 6 of them;
   the siphon-termini reference is not implemented. 5 sides have no placement at all (VA clearance on every pose /
   top) and need a decision (keep v5's VA top as it is, flagged).
2. **Junction topology / ostium.** v3 junction surfaces: genus 0 on 29/54, <= 2 on 49; three keep handles (song_008 L
   11, song_094 L 9, ia172_172 L 8) for the bake's handle rule. ia172_071 L (bottleneck 0.25 of the VA radius) and
   ia172_164 L (0.46) keep v5's junction (my rule: < 0.5; 20 sides); a tube bridge from v6_wall for ia172_164 L (as
   for 136 L / 172 L) would give those 16 anatomies the donor junction. Tube-bridged and used: ia172_136 L, 172 L.
3. **Narrow subclavian labels** in use (label / Arterial calibre < 0.7 on the donor stretch, flag `narrow_label`):
   song_040 R 0.69 (7 sides), vmr0248 L 0.66 (6), song_008 L 0.67 (3), song_032 L 0.61 (2), song_032 R 0.67 (1).
   Declared radius clipped; the surface is the label's. Stage 2: tube at Arterial calibre or a v6_wall re-run.
4. **Seam 2 is a synthetic bridge** (median 26 mm, 6-40) from the donor's E to the host's distal subclavian; the arm
   side beyond is the host's. Its shape is gated (two 1 mm joins <= 60, total turn <= 90 deg, smoothest of 6-35 mm,
   E shortened to ostium + 15 / 10 mm when needed). The turn gate costs 18 donor junctions; `V6VA_BRIDGE_TURN=120`
   recovers them (wt13: 16/16) but brings back arm-side hairpins (VMR 0248 L, `/work/pilot_t3/_fig`).
5. **Reversible decisions** (all recorded per side): rise floor = native - 10 deg for the 3 donor sides whose own VA
   rises < 15 deg; course RMS re-referenced (host VA, end carried to the D2 target) and gated on the absolute RMS;
   search clearance 0.65 vs other branches; junction failed when the ostium bottleneck < 0.5 of the VA radius;
   bridge turn <= 90 deg. Undo knobs: RISE_NATIVE_TOL_DEG, course_ok on rms_c, SEARCH_CLEAR_MARGIN, sc_post3
   BOTTLENECK_MIN, V6VA_BRIDGE_TURN.
6. **Bake integration** (pilot stage): `va/bake_sections_v6.py` adds the sc* sections; a v6 copy of bake_v5b must
   count (17)/(18)/(19) as VA sections (is_va). The vaC section ([F, splice]) and the junction surface's VA part
   ([F, fillet end + 3]) overlap on ~10-25 mm; both are from the same CT labels (v6_wall: equal beyond VA s 5,
   <= 0.36 mm in s 0-5) and the union takes the max. Not baked here: genus, self-intersection, wall detail of the
   subclavian stretch (D5) and SOFA (VA entry) are for the pilot stage.
7. **Item (b)**, the w_018b route13 move, runs after this re-pose (not done here); its left VA is the pilot's worst
   va_target case (28.3 mm).
8. **Carotid side** in the pilot output is v5's; the pilot stage merges the carotid tool's output and re-checks VA
   clearance (the VA gates here ran against v5's carotid side). The host's own (18) prefix overlaps the arch (0) for
   its first ~15 mm (host geometry, unchanged, seam-parent rule).
