# s2_harden_carotid -- the carotid tool made stage-2 ready (compose_v6 2.0.1)

Agent s2_harden_carotid, 2026-10-04. State: `state/s2_harden_carotid.md`. Code: `scratchpad/v6/carotid/` (runner
`dr2.sh`, containers `s2car_*`, /repo read-only; /pw = the pilot outputs read-only). Outputs:
`D:\Arjun\neve\carotid_data\v6_work\carotid\s2\` (`/out/s2` in the containers). Nothing in the repo, anatomies_v2..v5,
carotid_tools or topbrain_tools was written.

## Answer (final)

0. **Verdict: the carotid tool is stage-2 ready.** compose_v6 2.0.1 on the 14 pilots: 14 / 14 pass every carotid gate
   from the written files (base v5 and the joint-pass base); with D11 off it reproduces the pilot's guarded carotid pass
   exactly (14 / 14). All-223 dry run on base v5: 223 / 223 ok, 0 errors; ladder rung 1: 201, rung 2: 22 (1.2.1: 180 /
   43), rungs 3-6: 0; left fork pose by the Zenodo map 216 / 223, fallback (flagged) 7; right ECA tips v5 pose 26,
   re-posed 9, skipped 17 (anchor_low), 0 by the route guard; D13 v4_no_tip on 15; RICA terminus > 40 mm posterior on
   37; projected split EXIT 0, none of the 45 quantities lowered, donors one-to-one nested. One bug found and fixed
   (C2 sampling, 2.0.0 -> 2.0.1; latent in the pilot's wrapper too).
1. **`carotid/compose_v6.py` 2.0.1** is the stage-2 carotid tool, one versioned file (no wrappers). 1.2.1 is frozen
   as `compose_v6_v121.py`. New in 2.0 (docstring has the full rules):
   - `--base ROOT` (the joint pass: the VA tool's tree; default anatomies_v5); the writer copies the base.
   - **C1 / C2** (pilot `car_guarded.py`, folded in): the right ECA tip takes the first passing pose whose route excess
     <= 0.3 % of the right fork's relief (route_guard, now `carotid/route_guard.py`) and whose wall clearance to the base
     RCCA past the host cut >= min(3 mm, v4's own tip; 2.0.1: sampled from the same seam point as a candidate, so the
     v5 = v4 pose passes exactly); else skipped (D4 spirit). C3 available, off.
   - **D1** ladder rungs 1-6 unchanged (rung 4 = documented ICA extension <= 10 mm / anchor trim; rung 5 = extend_ica
     upward (+z) beyond 10 mm, tubed; rung 6 = full bend allowance). Never the host LCCA tube.
   - **D11**: right fork = v5 pose, never re-posed; its map metrics and the RICA-terminus AP offset are recorded and
     the offset flagged > 40 mm. Left forks (own and every substitute) are re-posed about seam 1 by the per-patient
     Zenodo map (bounded search: fork heading <= 45 vs the map, tilt <= 40, seam-1 bend <= 45 blended on the donor
     CCA, seam-1 join <= 60, route kink <= 60, hard clearance 0.6, self clearance, LICA seam rising >= 30), then
     every downstream gate; else v5's pose (own) / the composed frame-match (substitute), flagged. Inside each rung
     a fork tries up to 4 map poses (strict soft mode: VA-clean ones of up to 40 collected first), then its fallback.
   - **D13**: `rec.route_baseline` per anatomy (`carotid/route_baseline.py`).
   - **Records** per anatomy (`carotid_record.json` `summary`, provenance `v6_carotid.summary`): fork rung, left-fork
     decision / lower / pose (+ flag, why not map), tubed ICA mm, ICA AP offsets (RICA / LICA, posterior +) + flags,
     skipped ECA tips (side, why), right ECA decision, route baseline (+ why), soft contacts, rung-3 split note.
   - **Bake interface**: every blended piece carries `xform.<key>.blend_disp` ([s, dx, dy, dz] = written - rigid
     centreline over the blend) -- siphons, ECA tips and the map-posed `lower_L` (blend at the START of the lower_L
     section); `left.cca_mm` follows the fork blend's length change (`v6_cca_blend_adjust_mm`) so host_cut + cca + ica =
     the siphon seam stays exact; rung 5 / 6 tubes are written as v4's extended forks (`left.ica_mm` includes the
     tube, `xform.lower_L.ica_real_mm` = the real ICA; 1.2.1 left ica_mm unchanged).
2. **`carotid/route_baseline.py` 1.0** (D13 library = the pilot's v4_no_tip.py as functions + CLI): `decide`,
   `decide_from_tree`, `v4_tip`, `rebuild_v4_no_tip` (= vvw/rrd_v4.rebuild_v4 exactly, RECA truncated at the join,
   ecaext_R dropped), `measure_no_tip`. Rule: baseline "v4_no_tip" iff v4 has a RECA tip that comes <= 8 mm (wall to
   wall) of the RCCA past the host cut AND v6 skips or re-poses it; else "v4". The k_011 rebuild reproduces the pilot
   exactly (fork 0.2034 / siphon 0.3896 frac_0.3mm; 110 s).
3. **`gates/orient_gates.py` 1.1 (D12)**: one ICA-pair native convention, like for like = the back-mapped ends of the
   same pieces (it was already the measure; now stated, recorded as `pairs.ica.native_convention`, self-tested); plus
   `--fork-gates-placed` (fork gates block only forks with placement v6_map = the D11 left forks) and tree mode skips
   non-folder `case_*` entries. **SELFTEST PASS**: v5 / v4 / v3 counts unchanged; like-for-like identity exact on all
   223 x 3; shipped piece ends lie 1.84 mm (median, max 2.38) from the raw termini, and the raw reading would flip 6
   v5 verdicts (4 ica_spacing pass only raw, 1 fail only raw, 1 converge); the verifier's two pilot cases reproduce
   (m_006 mr_025_L |dd| 19.37 like for like vs 20.77 raw; mr_027_L 18.82 vs 21.65; no flag).
4. **Unit tests on the 14 pilots (all dry or written to v6_work only):** see "Tests".

## Tests (files under carotid_data/v6_work/carotid/s2/)

| test | what | result |
|---|---|---|
| T2 regression | 2.0 `--left-fork-pose v5`, base = the pilot's VA tree, `--plan dry_v121` vs the pilot's guarded carotid pass `pilot/car3` | 14 / 14 identical (left fork, both siphon R, ECA decisions and R, ok): `t2b_regress`, `s2/cmp_records.py` |
| T1 | dry, base v5, defaults (MAP_POSES 3, no anchor filter yet) | 14 / 14 ok; map pose 9 / 14; the 5 flagged: w_003 x2 (map poses dropped the LICA anchor to 25-29 deg), k_004 / w_018b / w_027 (map poses touch a v5 VA): led to the fork-level anchor filter and the VA-clean ranking |
| T5b | write, base v5, final defaults + `verify_s2.py` from the files | 14 / 14 ok; 14 / 14 left forks map-posed; 1 substitute (k_011 -> w_051_right, map-posed); w_047 mr_023 and m_006 mr_027_L now keep their own fork (rung 1) |
| T3b | write, base = the pilot's VA tree (the joint pass), + verify | 14 / 14 ok; map 14 / 14; rung 1: 13, rung 2: 1; AP R > 40 mm: 4 (w_003 mr_010 73.7, mr_007_L 68.6, k_004 44.0, w_018b 54.7); D13 v4_no_tip: 2 (k_011 re-posed tip, k_004 skipped tip) |
| T4 | forced ladder on k_011 (`--skip-rungs 2 3 4` / `2 3 4 5`, and rung 5 with `--left-fork-pose v5`) | rung 5 map pose + ICA tube 13.9 mm (anchor -72.6 -> +37.3 deg), rung 6, rung 5 v5 pose: verify ok each (after the ica_mm / ica_real_mm writer fix) |
| og | orient_gates tree on T3b | `--fork-gates-placed`: 0 tier A, 0 fork flags; blocking = va_target on 2 VA tops of the base (the VA tool's); `--fork-gates`: fork_heading on the 14 right forks (v5 pose, informational by D11) |
| 2.0.1 reruns | t3c (joint base write + verify), t5c (base v5 write + verify), t2c (regression vs car3) | 14 / 14 ok, 14 / 14 ok, 14 / 14 identical; t3c == t3b (2.0.0) on 14 / 14 (the C2 fix changes no pilot); summaries `t3c_summ.txt`; figures front + side `figs_t3c/` |
| D13 | `route_baseline.py rebuild` k_011, k_004 | k_011 fork 0.2034 / siphon 0.3896 (= pilot); k_004 0.0945 / 0.2786 = v4 shipped (its tip never counted as relief: the 8 mm rule only costs compute); pilot v6 / v4_no_tip: k_011 fork 0.9995, k_004 1.000 |
| selftest | `orient_gates.py selftest` on the final 1.1 file (md5 999f084c...) | PASS (`s2/logs/og_selftest3.log`, `gates/out/selftest.txt`) |

`verify_s2.py` (new, file-based): orient gates (carotid blocking 0; left fork gates where map-posed), right route
byte-identical to the base up to the siphon seam, clearance (hard 0.6 incl. the moved fork ECA), joins (siphons, ECA
tips, seam 1 of a map-posed fork, route kink), section arithmetic both sides <= 1.5 mm, the E6 contract (blend_disp ==
written - rigid, rigid beyond the blend <= 0.01 mm), summary fields present, D13 from the files == the record.

## Decisions taken here (reversible; how)

- **Tool identity**: compose_v6.py bumped to 2.0.x in place (2.0.1 final) (the name every note uses); 1.2.1 frozen as
  `compose_v6_v121.py`, orient_gates 1.0 as `gates/orient_gates_v1_0.py` (+ `gates/out/selftest_v1_0.txt`).
- **D11 inside D1**: the ladder decides WHICH fork (rung order unchanged: own fork before any substitute); D11
  decides its POSE (map first, fallback pose flagged). So an own fork that passes only in v5's pose stays at rung 1
  (flagged) rather than giving way to a map-posed substitute. Undo: put the map variants of every rung first.
- **Map pose must leave the LICA seam rising >= 30 deg** (the D1 anchor gate) at fork level, so the 4 map slots are not
  spent on poses try_left would refuse; rungs 5 / 6 drop the filter (the extension lifts the anchor).
- **Strict soft mode ranks VA-clean map poses first** (route from seam 1 + fork ECA vs the base VAs; up to 40 passing
  poses collected): the pilot showed tools gated against each other's old geometry collide. A map pose touching a VA
  is still taken if no VA-clean variant of the same fork exists. Undo: `--soft-mode prefer`.
- **C2 against the base RCCA** (exactly the pilot's car_guarded); the v6 siphon is covered by the hard 0.6 mm gate.
- **D13 threshold 8 mm** (wall to wall, v4 tip vs RCCA past the host cut): the verifier's zone; generous on purpose
  (a far tip changes nothing, the cost is one ~110 s rebuild in the checks). A v5-pose tip with only a piece-side
  blend counts as kept. Undo: `route_baseline.D13_NEAR_MM`.
- **ICA AP offset** uses the D12 convention (back-mapped piece end) against C_HOST + A0 (end - mid of the patient's raw
  VA ends): it reproduces the verifier's dyPost (k_011 38.86 vs 38.9, k_004 43.96 vs 43.6, w_018b 54.7 vs 54.4); the
  LEFT offset is recorded as information (D11 flags the right only).
- **Rung 5 / 6 bookkeeping**: ica_mm includes the tube, ica_real_mm keeps the real ICA (the v4 extended-fork
  convention the bake already slices by); 1.2.1 had not updated ica_mm (never written there).
- **cca_mm += the fork blend's length change** (<= ~0.5 mm) so the bake's section arithmetic stays exact.

## Stage-2 interface (what the other stage-2 agents need)

- **Run** (eve-training-fixed, /repo read-only, `--cpus 4 --memory 6g`):
  `python3 /sp/v6/carotid/compose_v6.py dry --base <VA tree> --out <dry dir>` (sequential, NOT sharded: the split
  guard commits substitutions in name order), then
  `python3 /sp/v6/carotid/compose_v6.py write --base <VA tree> --plan <that dry dir> --out <tree>` (the plan replays
  the earlier substitutions, so the write equals the dry). Use a plan made by 2.0 on the SAME base (not dry_v121).
  Then `verify_s2.py <tree> <base> <out.json>`.
- **Bake (E6)**: `xform.lower_L.blend_mm` / `blend_disp` exist on map-posed left forks (placement "v6_map") -- the
  blend is at the START of the lower_L section (seam 1), not at a piece start after a seam 2; for a substituted fork
  there is no v5 piece to back-map, so the morph must use `blend_disp` (written - rigid per centreline point, s from
  seam 1) instead of bake_v6p's v5 rebuild. Siphons / ECA tips carry `blend_disp` too (same numbers bake_v6p 1.2
  rebuilds). Rung 5 / 6 tubes: ica_mm includes the tube; lower_L.ica_real_mm = real ICA.
- **Checks (D13)**: `provenance v6_carotid.summary.route_baseline` = "v4" | "v4_no_tip"; for "v4_no_tip" run
  `python3 /sp/v6/carotid/route_baseline.py rebuild <name> --out <dir>` (neve-build-meshlab, ~110 s; skips if done)
  and divide v6's exact-ownership right_route_detail by its `exact.rrd_v4_no_tip` (fork and siphon), else by v4's.
  `route_baseline.py decide --tree <tree> --out <json>` recomputes the rule from the files.
- **Orientation post-bake**: `orient_gates.py tree --root <tree> --out <dir> --fork-gates-placed` (D11: only the
  map-posed left forks gated; right fork heading informational).
- **Split**: D11 keeps more own left forks, so the left assignment differs from dry_v121's 43 substitutions: re-run
  `v5_split.py --baseline split_v4.json` on the final tree (the projection here is on base v5).

## Open issues (for the orchestrator / stage 2)

1. **Bake E6 must cover `lower_L`** (map-posed left forks: blend <= 8 mm at the START of the section, seam 1) and use
   `blend_disp` where there is no v5 piece (substituted forks); bake_v6p 1.2's MORPH_KEYS cover siphons and ECA tips
   only. Without it the Zenodo surface over the first <= 8 mm of the donor CCA sits up to ~1 mm off the blended
   centreline (the same defect E6 fixed on w_034's siphon). For the s2_harden_wall_bake agent.
2. **The left route changes from seam 1 in almost every anatomy** (D11 map pose; the right route is untouched up to the
   siphon seam). D11 + the anchor filter also keeps more own forks than 1.2.1 (fewer substitutes), so the left
   assignment differs from dry_v121: the split must be re-checked on the final tree (projection on base v5 below).
3. **ICA AP (D11, right, recorded only)**: right termini > 40 mm posterior of the anatomical target remain on a
   quarter of the anatomies (the right fork keeps v5's heading by D11); a gate would need the right fork re-posed.
4. **--plan replay**: the stage-2 write must use a dry plan made by this version on the same base; dry_v121 (used for
   the pilot tests here) predates D11, so the guard state of later anatomies differs slightly from a 2.0 plan.
5. **D13 compute**: one v4 rebuild (~110-145 s, neve-build-meshlab) per v4_no_tip anatomy in the checks.
6. C2 bug shared with the pilot's car_guarded (fixed here in 2.0.1): any pilot-era tree whose right v5-pose tip was
   re-posed only by C2 noise is not affected in the pilot (k_011 re-posed for other reasons), but stage 2 must use 2.0.1.

## All-223 dry run (2.0.1, base v5, unsharded; `s2/dry223_v201/`, `s2/r223_report.txt`, `s2/dry223_v201_summ.txt`)

Not the stage-2 joint pass (no all-223 VA tree exists yet): a smoke test of every code path at scale and the counts.
1070 s total (median 3.4 s per anatomy, max 12.3) in one container.
- **223 / 223 ok, 0 errors.** Rungs: 1 = 201, 2 = 22, 3-6 = 0; no tubed ICA, no host tube.
- **Substitutes 22** (1.2.1: 43): 11 with the same lower as 1.2.1, 11 with another split-compatible lower; 21 anatomies
  that 1.2.1 substituted now keep their own fork in a map pose (e.g. w_047_left mr_023, m_006_left mr_027_L,
  w_008_right x5); no new substitution.
- **D11 left pose**: Zenodo map 216; flagged 7 = 3 own forks in v5's pose (w_018b_left mr_007, w_025_right mr_007,
  w_050_left mr_020_L: every map pose passes but touches a v5 VA; strict mode prefers the VA-clean v5 pose -- in the
  joint pass the v6 VAs decide, as w_018b mr_002 showed) + 4 substitutes case_w_006_left in the composed pose (no LICA
  pose passes on any map pose).
- **Right ECA tips**: v5 pose 26, re-posed 9, skipped 17 (all anchor_low; 0 by the route guard), none 171.
- **D13**: v4_no_tip on 15 (6 skipped + 9 re-posed tips within 8 mm of the RCCA): `s2/r223_d13.txt`.
- **ICA AP (D11 record)**: RICA posterior offset median 18.9 mm, p90 48.1, max 86.0, > 40 mm on 37; LICA median 4.2,
  > 40 on 2 (the left re-pose brings the left termini forward).
- **Soft (VA) contacts**: 1 anatomy (w_046_left mr_026, also in 1.2.1).
- **Projected split** (`make_split_tree.py` + `v5_split.py --baseline split_v4.json`, `s2/split_check_223_v201.log`):
  **EXIT 0, none of the 45 quantities lowered** (strict k16_div6x6 168 -> 169, k17_div6x6 167 -> 169 raised, as 1.2.1);
  V donors 25 in 25 TopBrain patients, one-to-one nested.

## Files

- Tool: `carotid/compose_v6.py` (2.0.1), libraries `carotid/route_guard.py` (1.0, = pilot), `carotid/route_baseline.py`
  (1.0, D13), file verifier `carotid/verify_s2.py`, runner `carotid/dr2.sh`; frozen `carotid/compose_v6_v121.py`.
- Gates: `gates/orient_gates.py` (1.1), frozen `gates/orient_gates_v1_0.py`, `gates/out/selftest.txt` (1.1 run),
  `gates/out/selftest_v1_0.txt`.
- Test helpers `carotid/s2/`: `cmp_records.py`, `summ.py`, `report_223.py`, `tries.py`, `show_survey.py`, `fig_s2.py`,
  `chain_223.sh`, name lists.
- Outputs `carotid_data/v6_work/carotid/s2/`: `t2b_regress`, `t2c_regress` (regression), `t1b_pilot_v5base`,
  `t1c_mp8` (T1), `t5b_write_v5base` / `t5c_write_v5base` + `*_verify.json` (base v5 writes), `t3b_joint_write` /
  `t3c_joint_write` + verify (joint base), `t4b_r5`, `t4b_r6`, `t4b_r5_v5pose` (forced rungs), `figs_t3c/`,
  `og_t3b_placed/`, `og_t3b_all/`, `d13/` (survey + 2 rebuilds), `dry223_v201/` + `r223_report.txt`, `r223_d13.txt`,
  `dry223_v201_summ.*`, `split_tree_223_v201/`, `split_check_223_v201.{json,log}`, `logs/`.
  Superseded (2.0.0 runs before the C2 fix / map-pose refinements): `t1_pilot_v5base`, `t2_regress`, `t3_joint_write`,
  `t5_write_v5base`, `t4_r5`, `t4_r6`, `dry223_v200` (stopped at ~70).
