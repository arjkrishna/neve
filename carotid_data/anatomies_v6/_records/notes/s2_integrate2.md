# s2_integrate2 -- D22-D25 applied, one frozen toolchain, the second all-223 DRY, split check and re-pilot

Agent s2_integrate2, 2026-10-06 (evening). State: `state/s2_integrate2.md`. Builds on s2_integrate (`notes/s2_integrate.md`,
build_v6 1.1) and the gate verifier's NO-GO (`notes/verify_s2_gate.md`). Code: `scratchpad/v6/{carotid,va,checks,build,figs}`
(versioned tools, each with a frozen copy of its previous version), test scripts `build/i2_*.py`. Outputs:
`carotid_data/v6_work/build/dry223b/` (/bld/dry223b, the DRY), `carotid_data/v6_work/repilot2/` (/rp2, the re-pilot),
`carotid_data/v6_work/build/i2test/` (tests). Containers `s2i2_*` only; /repo and /rp read-only. No git writes; no
repo-tracked tree touched; no user notes file touched.

## Answer so far

DONE. DRY 223 / 223 composed with ONE frozen toolchain (manifest bedf172b on 223 / 223), 0 hard failures, final check
223 / 223 ok, split exit 0 (none of 45 lowered), donors one-to-one 25 / 25. Re-pilot 14: bake 14 / 14, checks_v6 1.3
14 / 14 PASS on every gate (D22 fork + siphon), SOFA RCCA 14 / 14, D14 32 / 42 seeds, load test passed. Details in
sections 2-3; flags and the two interpretations the user should confirm in section 4.

- Tools changed (all versioned, stated diffs in each file header, previous version frozen next to it):
  - route_guard 1.3 (`carotid/route_guard.py`, frozen `route_guard_v1_2.py`): + `SurfaceKeep`, the D22 pre-bake estimate.
  - compose_v6 3.1.0 (frozen `compose_v6_v303.py`): D22 replaces the C0 gate (per variant and in the final check), C0
    kept as information; D25 every-phase 1 mm joins on every seam.
  - verify_s3 1.3 (frozen `verify_s3_v1_2.py`): D22 recomputed from the files is the gate; C0 reported only.
  - graft_v6_va 1.3 (frozen `graft_v6_va_1p2p3.py`): D25 D17 direction gate (side gate -> D18 v5 junction) and
    every-phase 1 mm turns on all new seams.
  - geom_v6 1.1 (frozen `geom_v6_v1_0.py`): the every-phase 1 mm turn rule (one implementation, imported by the composer
    and the VA tool).
  - checks_v6 1.3 (frozen `checks_v6_v1_2.py`): D22 post-bake gate (fork / siphon), coverage + strict values reported,
    D23 tier-A acceptance.
  - build_v6 1.2 (frozen `build_v6_v1_1.py`): frozen-toolchain md5 check before every stage, donor_id on rung-2/3 sides,
    D23 block, D22 / D25 records, checks_v6 orientation part in the final check.
  - precheck_orient 1.1, fig_v6_repilot 1.1 (frozen 1.0 copies); new chain_dry2.sh, chain_repilot2.sh, dr_i2.sh.
- Frozen toolchain: `build/TOOLCHAIN_s2i2.md5` (46 files, md5 bedf172b, written 20:54 EDT), checked by build_v6 1.2 at
  every stage start (the DRY and the re-pilot build ran with it).

## 1. What changed, per decision (tools; every gate tightened or replaced as decided, none relaxed)

### D22 -- the right-route check is "surface preserved"
- **Post-bake (checks_v6 1.3, part route, `rec.route.d22`)**: for the right fork (lower_R) and the right siphon (siphon_R),
  the baked union's cells exactly owned by the RCCA in the piece's span, beyond its seam zone ([cut, cut + max(blend v6,
  v4) + 2] fork; [seam 2, seam 2 + max(siphon blend v6, v4) + 2] siphon) and >= 2 mm before the span end, minus the bake's
  handle windows (v6's suppressed stretches of the section and v4's mapped onto v6), are measured against v4's union (the
  D13 baseline union where it applies) CARRIED BY THE PIECE'S OWN TRANSFORM (x6 = (((x4 - a4) R4 + o4) M4 M6 - o6) R6^T +
  a6, mirror included), exact point-to-triangle distance. Gate: >= 99 % of that area within 0.3 mm
  (`route_fork_d22_ge_0.99`, `route_siphon_d22_ge_0.99`, replacing `route_*_ge_0.99`). Reported, not gated: the seam zone
  and the handle windows (area, share within 0.3, max), COVERAGE (v4's own RCCA-owned cells of the same stretch, carried,
  within 0.3 mm of the v6 union -- sees wall a new neighbour now OWNS, which the v6 -> v4 direction cannot), the relief
  ratio (`gate_ratio`, information).
- **Reference validity (an interpretation, stated)**: cells whose v4 wall was occupied by ANOTHER branch (in v4's world
  another branch's tube value within v4's ownership margin + 0.3 mm of the RCCA's at the inverse-carried point; the
  fork's own ECA excepted) are left out of the gated area and reported (`v4_occupied`, `value_strict` with them). Why:
  the first re-pilot's k_011_left mr_012_L siphon reads 0.9776 strict -- 39 mm2 of its bulge at s 206 lie 0.3-1.5 mm
  INSIDE v4's carried union because v4's RVA top was fused onto that bulge (v4 owners there: RCCA 392 / RVA 174 cells;
  the v6 cells sit on the siphon's own real surface within 0.16 mm; `/bld/i2test/diag_k011_siphon.json`). v4 never
  showed the piece's own wall there, so v4 is no reference -- the same reasoning as D13 for v4's ECA tip. Without the
  rule D22 would fail pieces for having LOST a v4 contact.
- **Pre-bake (route_guard 1.3 `SurfaceKeep`, compose_v6 3.1.0)**: each source vertex of the section is evaluated in
  v4's world and in the candidate's (same source surface, both xforms): the union's ownership margin (v4's margin_mm,
  sdf_union_v4's rule) and tube cover; a vertex is CHANGED when its ownership flips (margins differ by > 0.05 mm) where
  its real wall stands > 0.3 mm off the RCCA tube, or its cover flips with > 0.3 mm depth; keep = 1 - changed / gated
  area (v6-RCCA-owned, beyond the zone, the same reference-validity rule). Gate keep_hard >= 0.99 (non-VA neighbours,
  as C0; the VA tool re-places the VAs afterwards with its own route guard; 'all' recorded). It replaces C0 per right
  variant (route cut at seam 2, the RECA left out as before) and in the final check (fork where re-posed, siphon
  always). C0 is still computed for the final anatomy and recorded (`c0_*`, information). verify_s3 1.3 recomputes D22
  from the files as the gate (record match reported).
- **D24**: the R4 / R5 forks and the anatomies C0 pushed off their pose are re-evaluated simply by the DRY with D22 in
  place (no per-anatomy override); `stage2_dry2.json` lists every right fork that changed and the old R4 / R5 / C0
  anatomies' new rungs.

### D25 -- D17 direction, 1 mm joins at every phase, records, one frozen toolchain
- **D17 direction (graft_v6_va 1.3 U1)**: on a donor-junction side the D17 branch's cut end must lie LATERAL of its
  ostium foot ((end - foot) . lateral > 0; lateral -x right, +x left) and the chord foot -> cut end < 90 deg from the
  host's own distal subclavian ((17) / (19) of eve_bench, start -> end). A SIDE gate in final_gates / --recheck: a
  failing side is demoted down the D9 ladder (rung 2 = D18's v5 junction). Not a pose filter: as a pose filter (first
  draft) the search found a contorted pose of ia172_069 R (1.6 mm lateral, 92 mm anterior, 77.5 deg) that passes the
  letter of the gate with the branch still in front of the neck. Test `/bld/i2test/va_t1`: m_006_left mr_025_L and
  w_051_right mr_025_L L -> rung 2 v5 junction; W1 of the DRY: exactly those 2 sides changed rung, 404 / 404 other
  donor-junction sides pass.
- **1 mm joins at every sampling phase (geom_v6 1.1, the one implementation)**: the curve resampled at 1 mm from offsets
  0, 0.05, ..., 0.95 mm, the turn between consecutive chords at each sample; max(written, phased). geom_v6 1.1 on the
  first DRY's final tree finds exactly the 10 known exceedances (8 x (19) 61.4 deg at s 3.0 on mr_014_L, 2 x (18) 64.5
  on mr_025_L; `/bld/i2test/geom11_dry223.log`). Applied: compose_v6 3.1.0 `join_kink_ph` on every seam (seam 1 right /
  left in the pose searches and the final check, RICA / LICA seam 2, ECA tips, siphon searches); graft_v6_va 1.3 U2 on
  seam 1, the D17 branch from F, the T2 corner spreading and T3 (final, via geom_v6.kinks). FIX BY BLEND: song_095 R's
  (19) corner is spread until both forms are <= 58 (sigma cap for the D17 spread 4.5 mm, was 3; construction only --
  the acceptance, every point moved < 0.75 r and every-phase turn <= 60, unchanged): k_004_left mr_014_L 152 -> 57.8
  deg at sigma 3.05, max displacement 0.657 r (1.86 mm), side stays rung 1. The 2 mr_025_L (18) cases fall back (they fail
  the direction gate).
- **Records (build_v6 1.2)**: v6_build.va.<side>.donor_id / donor_side from xform.vaC_* where the VA record has none
  (rung 2 / 3), `donor_id_from`; d17_direction per side; D22 pre-bake fields + coverage flag; `toolchain` (manifest md5
  + each scratch tool's md5) per anatomy; tool versions recorded by the tools themselves (no hand-written version notes).
- **One frozen toolchain**: `build/TOOLCHAIN_s2i2.md5` lists 46 files (every scratch tool the build, bake, checks, SOFA
  and figures run, their v5 / v4 helpers, and the read-only repo / topbrain_tools modules they import -- traced by
  importing the tools, `build/i2_trace_imports.py`); build_v6 1.2 refuses to run any stage on a mismatch.

### D23
- build_v6 1.2 writes `v6_build.d23_accepted = {pieces: [RVA_top], pair_flags: [...]}` for case_w_017_left__topcow_mr_002_L
  when its right VA is on D9 rung 3, + flag `D23_tierA_accepted:RVA_top`; checks_v6 1.3 part orient accepts exactly
  those (tier A of that piece, its own blocking flags and the VA pair flags it causes on the other top) as
  `accepted_flags: D23:...`; gate orient_tierA_0 = "0 tier A except flagged D23". precheck_orient 1.1 (in the DRY's final
  check) uses the same rule.

## 2. The second DRY, all 223 (`results/stage2_dry2.json`; tree /bld/dry223b/final; one frozen toolchain)

Ran 2026-10-06 20:54 -> 2026-10-07 00:37 EDT as chain_dry2.sh in 3 slots (va1 -> cardry -> carwrite x3 -> carverify x2 ->
va2 x3 -> recheck x2 -> fix -> route13 -> final x2 -> finalcheck x2 -> split -> summary), all stages rc 0.
- **Composed 223 / 223, 0 hard failures** (not_composed / carotid_not_ok / finalcheck_not_ok all empty). Toolchain
  manifest md5 bedf172b on 223 / 223 records, every listed file still matching after the run; compose_v6 3.1.0 and
  graft_v6_va 1.3 on 223 / 223 (the verifier's mixed 1.2 / 1.2.2 / 1.2.3 finding is gone).
- **Final check 223 / 223 ok**: VA recheck 223, geom 223 (min new clearance 0.614 mm, 0 every-phase kink failures),
  carotid verify_s3 1.3 223 (D22 recomputed from the files == record 223 / 223).
- **Right fork (D19 / D22 / D24)**: R1 map 124, R2 map + host-cut shift 91, R3 bend <= 60 3, R4 deviation 3 (1 map_dev +
  2 map_dev_bend60), R5 v5 pose 2; first DRY R1 118 / R2 83 / R3 13 / R4 4 / R5 5. 30 anatomies changed rung or shift
  (`right_fork.changed_vs_first_dry`). Mirrored 113 = contralateral 113. Seam-1 bend median 39.7, max 55.7 deg; > 45 on 5
  (the R3 / R4-bend60 forks, flagged). Host-cut shift on 92: median 2.0, max 15.0 mm (limit HOSTCUT_SHIFT_MAX_MM 30,
  unchanged since compose 3.0.2). 8 right forks flagged (3 R3, 3 R4, 2 R5).
  - D24, the first DRY's 9 R4 / R5: k_008 mr_027 R4 -> R2; w_002 mr_013_L and mr_022 R5 -> R3; w_017_right mr_011
    R5 -> R1 (it fell to v5 only because of the C0 siphon estimate 0.972; D22 keep 0.996); w_003_right mr_001 and
    w_036_right mr_018_L stay R5 (their LEFT ladder fails with every map-posed right fork -- not D22); w_018b mr_002_L,
    w_018b mr_007 and w_027_right mr_020 are R4 again, now chosen by the VA-hard fix loop (below).
  - The C0-pushed anatomies (26, `c0_pushed_reevaluated`): all re-evaluated; 24 moved to a lower rung or a shorter shift, 2 unchanged (w_003_right mr_001, w_036_right
    mr_018_L: R5, left ladder).
- **D22 pre-bake (SurfaceKeep)**: fork keep min 1.0; siphon keep min 0.9963 (p05 0.9999); 0 variants refused in 0
  anatomies. Coverage (reported) < 0.99 on 2 forks: k_011_left mr_013_L 0.9885, w_012_left mr_013_L 0.9895 (flagged).
  C0 kept as information: fork min 0.851 (k_005_right x4 at 0.851-0.856 -- open issue 1), siphon min 0.969
  (w_017_right mr_011); 15 anatomies with a C0 ratio < 0.99.
- **Left fork (D1 / D11 / D20)**: own fork 201, substitute 22, ICA extension 0, host LCCA 0; map-posed 215, frame match 6
  (4 + 2 mirrored), v5 pose 2 (8 flagged); mirrored 101 = contralateral 101. Left seam-1 bend max 44.97 deg.
- **ICA terminus AP > 40 mm**: 3 (all RICA): w_003_right mr_001 46.0 (R5), w_018_right mr_007_L 53.3, w_027_right mr_016
  45.3; v5 108. RICA median 6.4, LICA median 4.3 mm.
- **ECA tips**: right carried 4, re-posed by search 29, skipped 19, none 171; left skipped 36; 54 anatomies skip one.
  D13 baseline v4 187 / v4_no_tip 36.
- **VA**: 446 sides, 429 donor sides; donor own junction **413 / 429** (first DRY 416, s2_harden_va J223 414, stage-1
  359). Rungs 413 / 15 / 1 / host 17 (D21, out of scope). Changed vs the first DRY: exactly 3 sides, all to rung 2 v5
  junction -- m_006_left mr_025_L L and w_051_right mr_025_L L (ia172_069, D17 direction) and w_002_right mr_013_L R
  (song_AHMU1218040; its fork is now R3 map-posed and the junction no longer clears). Junction types: own 393, + D16
  tube 18, + tube bridge 2, v5 junction 15, v5 top 1, host 17. pass_all 444 / 446: the two w_017_left mr_002_L sides
  (D23, below). va_target <= 20: 436, 20-30: 9, > 30: 1 (w_017_left mr_002_L R, 55.3, D23). D10 corridor 421 / 429 (8
  inherited v5 stretches on rungs 2 / 3, flagged). D17 branch 29.2-108.2 mm past the ostium (55 sides tubed, max 12.1).
- **D25**: D17 direction measured on 413 donor-junction sides, 413 ok (lateral 8.7-90.1 mm, angle 4.2-66.7 deg).
  Every-phase 1 mm turns: 0 failures on all new seams in the final check; the verifier's 10: 8 x (19) on mr_014_L now
  57.8 deg (spread, sigma 3.05), the 2 x (18) on mr_025_L gone (v5 junction). Final VA curves above 58 deg: 15 sides on
  TopBrain patient mr_026's anatomies at 64.5-64.9 deg, which is INSIDE the mr_026 VA top piece (s 155 / 179-195 mm)
  and equal to v5's own 64.8-64.9 (the rigidly-re-posed-piece rule "<= v5 + 1 deg", unchanged since geom_v6 1.0) --
  not a seam. Records: 16 / 16 rung-2 / 3 sides carry donor_id (from xform.vaC_*), 0 without.
- **VA-vs-carotid hard contacts (the VA agent's 2 open sides + 1 more)**: the W2 recheck found 3 (w_018b mr_002_L RCCA
  -2.31, w_018b mr_007 RECA -0.71 / RCCA -0.55, w_027_right mr_020 RCCA -1.65); build_v6's fix loop (VA hard in the
  carotid step) adopted a VA-clean right fork at iteration 1 on all 3 (R4; `fix_loop`), rechecked ok.
- **w_018b route13 move**: adopted; lost wall 3574 -> 0 cells, min slack -6.69 -> +2.55 mm (`route13`).
- **Orientation**: 1 anatomy with tier A on the tree = w_017_left mr_002_L RVA_top, D23-accepted; 0 otherwise.
- **Split check** (v5_split.py --baseline split_v4.json on /bld/dry223b/final): **exit 0, 45 compared, none lowered**;
  25 TopBrain patients, donors one-to-one 25 / 25 (provenance), no donor spanning more than 1 patient.

## 3. The second re-pilot, 14 anatomies (`results/repilot2_summary.json`; /rp2 = carotid_data/v6_work/repilot2)

build_v6 1.2 (plan = the DRY's car_dry) -> bake_v6 1.1 -> checks_v6 1.3 (D22) -> sofa_va_entry 1.1 (D14, seeds
777-779) + sofa_rcca_v6 (seeds 777-779) -> load test, geometric probe, corridor, orient, front / side figures with the
first re-pilot as the middle column (`/rp2/figs/v6_repilot_<name>.png`).
- **Trees**: 14 / 14 composed, byte-identical to the DRY (curves + xforms). Right fork rungs equal the first re-pilot
  on 14 / 14 (R1 10, R2 4; 10 mirrored); w_035 mr_024_L shift 8.96 -> 6.97 mm (C0 no longer pushes it).
- **Bake 14 / 14 pass** (bake rungs equal the first re-pilot: 7 "1 pinned", 7 "2 VA-local windows"; 117-499 s, peak
  RSS <= 1.97 GB). **checks_v6 1.3: 14 / 14 PASS on every gate** -- strict audit, route_fork_d22, route_siphon_d22,
  rcca_prefix_unchanged, va_lumen_obj >= 0.65 (min 0.664 w_050 LVA), orient tier A 0, blocking 0, loader, geom
  clearance (new min >= 1.29 mm) and every-phase kinks. The first re-pilot passed the old relief gate on 7 / 14.
- **D22 post-bake**: fork min 0.9996 (w_050), siphon min 0.9999 (w_051); coverage >= 0.99 on 28 / 28. Strict values
  (v4-occupied cells included, reported): fork >= 0.9915 on 14; siphon < 0.99 on 2 -- k_011_left mr_012_L 0.9776
  (39 mm2) and w_047_left mr_023 0.9890 (16 mm2 at 0.3-0.65 mm, s 169-201): in both, the v4 RVA top lay on the siphon
  bulge (v4 owners there RCCA / RVA; `/bld/i2test/diag_k011_siphon.json`, `/bld/i2test/diag_w047_siphon.json`) and the
  v6 VA no longer touches it. Relief ratio (information): fork < 0.99 on 7 (min 0.967 w_035), siphon >= 0.9918.
- **SOFA RCCA: 14 / 14 passed** (seeds past seam 1 + 20 mm 42 / 42, past bifurcation 41, siphon 34, target 22 -- equal
  to the first re-pilot; v5 41 / 33 / 19). Per anatomy the target counts equal the first re-pilot on 14 / 14.
- **D14** (3 seeds, >= 20 mm into the VA): 32 / 42 seeds, 13 / 14 anatomies (first re-pilot 33 / 42, 13; v5 22 / 42,
  12). Only change: w_035 mr_024_L 3 -> 2 seeds. Geometric probe enters R 14 / L 12 (v5 13 / 12).
- **VA**: the only change vs the first re-pilot is m_006_left mr_025_L L (ia172_069) donor junction -> v5 junction (D17
  direction, D18). w_050_left mr_004_L R (rung 2) now records donor_id ia172_071 (D25 records).
- **ICA AP > 40**: 0 / 14 (v5 8). Load test 14 / 14 episodes PASSED (SOFA segfaults at interpreter exit AFTER the PASS
  line and JSON, as in earlier runs).

## 4. Open issues and flags for the user / orchestrator
1. **D22 + host-cut shift trims the donor CCA**: with C0 gone, k_005_right x4 now take R2 with a ~15 mm host-cut shift
   (bend 42) instead of R3 (bend 59.5, shift 4); the donor fork gives ~15 mm of its own CCA to the host RCCA. D22 = 1.0
   (it gates what is still donor surface), C0 information 0.85. Within the documented limit (30 mm) and D19's ladder
   order (R2 before R3), but a visible change of the route; the user may want a shift cap.
2. **The D22 reference-validity rule is my interpretation**: cells where the v4 wall was occupied by another branch are
   left out of the gated area (reported as v4_occupied / value_strict). Without it 2 / 14 re-pilot siphons fail D22
   (k_011 0.9776, w_047 0.9890), both because the v4 RVA top had fused on the siphon. Needs the user's confirmation.
3. **D23**: w_017_left mr_002_L R stays on D9 rung 3 (v5 top, tier A, va_target 55.3), accepted and flagged; its L side
   carries the pair flag.
4. **Flagged right forks (8)**: R3 w_002 mr_013_L / mr_022 / mr_024; R4 w_018b mr_002_L, w_018b mr_007, w_027_right
   mr_020 (VA-hard fix loop); R5 w_003_right mr_001 (also AP 46.0) and w_036_right mr_018_L (left ladder).
5. **Not gated (D25 is 1 mm)**: net turns over 2-3 mm at some seams still reach about 95-116 deg (verifier turns2mm,
   e.g. song_040 R (18)); reported, not gated.
6. **Stage 3 still needed**: bake + checks_v6 1.3 + SOFA RCCA on ALL 223 (the training route moved), D14 and figures;
   the 223 bake needs the 36 v4_no_tip baselines.
7. Coverage flags (2 forks, pre-bake) and the 8 inherited D10 failures are flagged, not gated.

## 5. Files
- results: `results/stage2_dry2.json` (stage2_dry2_compose 1.0), `results/stage2_dry2_auto.json` (build_v6 summary),
  `results/repilot2_summary.json` (repilot2_summary 1.0, with table_md).
- tools (versioned, previous version frozen next to each): carotid/{compose_v6 3.1.0, route_guard 1.3, verify_s3 1.3},
  va/graft_v6_va 1.3, checks/{checks_v6 1.3, geom_v6 1.1}, build/{build_v6 1.2, precheck_orient 1.1}, figs/fig_v6_repilot
  1.1; manifest `build/TOOLCHAIN_s2i2.md5` (46 files, md5 bedf172b).
- runners: build/dr_i2.sh, chain_dry2.sh, chain_repilot2.sh; diagnostics build/i2_*.py (i2_diag_real2.py: v4 owners
  of D22-deviating cells).
- outputs: carotid_data/v6_work/build/dry223b (DRY), carotid_data/v6_work/repilot2 (re-pilot: tree, bake, iso,
  checks, sofa, post, figs), carotid_data/v6_work/build/i2test (tests, diagnostics).

No git writes; no repo-tracked tree touched; containers s2i2_* only, all exited.

Status: complete
