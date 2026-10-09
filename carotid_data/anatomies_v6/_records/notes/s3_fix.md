# s3_fix -- the final verifier's must-fix list on anatomies_v6

Agent s3_fix, 2026-10-08. State: `state/s3_fix.md`. Code: `scratchpad/v6/s3fix/` (runner `dr_s3.sh`, containers `s3f_*`).
Work outputs: `carotid_data/v6_work/s3fix/` (/s3). Mounts: /repo read-only, /w (v6_work) read-only, /s3 read-write.
No git writes.

## Answer (final, 2026-10-08 ~07:00 CEST)

**All three must-fix items are applied; all 223 anatomies stay in anatomies_v6 and pass every gate.**
- **(1) Real wall.** Cause found: every lost stretch of the 8 pieces is a component of the bake's union that keep_largest
  dropped, an island of the piece's own scanned lumen (a thin sliver 1-3 mm off the main lumen). Window class (w_052,
  w_027 x2, w_051 siphon): the bake's handle window cut the sliver's neck. Tip class (k_011, w_002, w_012_left mr_003_L,
  w_008_right mr_003 = exactly the D13 v4_no_tip anatomies): the sliver is separate in the scan (0.62 mm gap); v4 / v5
  showed it only through their RECA tip's tube. All 8 fixed by a targeted re-bake (bake_v6 1.2, D27 rung 1 / 2 / 3):
  L1 lattice shift on 4 (w_052, w_027 x2 at (0.225,0,0); w_051 siphon at (0,0,0.225)), L2 route-island bridge on 3
  (k_011, w_012_left mr_003_L, w_008_right mr_003), L1 + L2 on w_002_right mr_013_L. Real-wall deficit after: 0.0 on
  both references for all 8. v5's geometry was never needed. The 8 pieces at 0.1-1 % (the verifier's 5 + w_008_right
  mr_001 fork, w_047_left mr_027 siphon, w_034_left mr_011 siphon) are accepted and flagged in provenance (record-only).
- **The coverage check is a gate**: checks_v6 1.6 parts coverage (against the v4 / v5 SHIPPED wall) + folds, under the
  new frozen manifest TOOLCHAIN_s3fix.md5 = 2288f778 (bake_v6 1.2, checks_v6 1.6; 1.1 / 1.5 frozen). On the pre-fix
  folder it finds exactly the verifier's 8 >= 1 %; on the final folder 0 >= 1 %, 8 flagged, 223 / 223 pass.
- **Every changed anatomy** (8 re-baked + m_022 fold fix): full checks_v6 1.6 (every part) pass; SOFA RCCA 3 seeds pass
  9 / 9; split on the final set exit 0, none of 45 lowered; loader 223 / 223; copied with sha256 verification (manifest
  024edcd2..., 5,136 files); provenance re-stamped (stamp_prov 1.1, 17 anatomies: + coverage, folds, derived_flags,
  s3_fix, the D27 flags); figures re-rendered (17).
- **(3) Fold-over** on m_022_left mr_021_L's OBJ removed (spike vertex 7648, 3 faces -> 1; OBJ clean, 0 folds).
- **(2) Records**: BUILD_v6.json regenerated (build_v6_record 1.1, checker 1.1: 123 checks, 0 differences; s3_fix block,
  coverage block, derived_flags per anatomy); _records (+ s3_fix results, 1,466 files in RECORDS.sha256); carotid_tools/v6
  (make_tools 1.1, 257 files, restore test: 253 / 253 identical, 2288f778 passes md5sum -c); CAROTID_V6.md (coverage
  table, D27 cases, record corrections incl. the 4 -> 8 coverage list and the "every flag in provenance" statement,
  RCCA reach after the fix).
- **Reported, not gated**: SOFA reach on the 9 re-run anatomies moved by one seed (all 223: target 373 vs 374, 182 vs
  183 anatomies; w_002 lost its one target seed); the 25-anatomy set holds none of them (target 36 vs v5 46, siphon 52 vs
  54, unchanged). D22 now reads against the own scanned surface (D26) on 4 more pieces (the bridged islands) and two
  bridged forks pass D22 through D27's capsule repair window (w_002 12.0 mm2, w_008_right mr_003 56.6 mm2).

## 1. Diagnosis: why the 8 right-route pieces lost real scanned wall

- `diag_frag.py` rebuilds an anatomy's FINAL bake union (bake_v6 1.1 state from mesh_v6.json, bake_v5b's own
  UnionCache) and lists what keep_largest drops. w_052_left mr_026 (fork, 4.9 %): a dropped component of 932 cells,
  18.1 mm3, at RCCA s 110-126, 1.0-2.7 mm off the RCCA tube, 98.5 % of its points on lower_R's own scanned surface
  = exactly the verifier's lost stretch (s 111.25-125.5, depth 1.6-2.6 mm). (`/s3/diag/frag_w052.json`)
- `diag_ray.py` along a ray through it (s 118): the scanned lumen ends at 2.3 mm from the centreline (= the tube),
  then is outside to 3.9 mm, then a thin INSIDE sliver (3.9-4.4 mm, f_real <= 0.04 mm), then outside: the Zenodo
  lumen carries a thin sliver ~1.6 mm off the main lumen. `diag_sign.py`: the SDF sign agrees with ray parity
  everywhere (no inverted patch), so the sliver is in the scan itself.
- The bake's attempts: with no window the v6 lattice already leaves parts of the sliver as islands (dropped 400 + 352
  cells); the handle window [108.99, 110.24] (the sliver forms a handle with the main lumen at the v6 pose) cut its neck
  and the islands grew (932 + 308 + 260 + 190 cells dropped). v5's bake (same baker, v4's pose) kept it attached at rung
  1 with no window.
- All 13 listed pieces (`sum_frag.py` over `/s3/diag/frag_*.json`): the lost stretch is a dropped component on the piece's
  own scanned surface (lower_R / siphon_R), 1-3 mm off the RCCA tube. Two classes:
  * WINDOW class (w_052_left mr_026, w_027_left mr_022 / mr_026 forks, w_051_right mr_008_L siphon): at the v6 pose the
    sliver makes a handle with the main lumen; the bake's handle window (lower_R [108.99, 110.24], [60.0, 62.0],
    [59.81, 61.45]; siphon_R [200.45, 205.45]) cuts its neck and the rest of the sliver becomes an island that
    keep_largest drops. The shrink step accepts any genus-0 window, also one that cuts real wall loose.
  * TIP class (k_011_left mr_012_L, w_002_right mr_013_L, w_012_left mr_003_L, w_008_right mr_003 forks = exactly the
    D13 v4_no_tip anatomies): the sliver is SEPARATE from the main lumen in the scan itself (`diag_neck.py`, k_011 at
    0.15 mm: own component, 0.62 mm from the main lumen, also without the capsule). v4 / v5 showed it only because
    their RECA TopBrain tip ran along the ICA and its TUBE bridged the gap; v6 re-posed the tip anatomically, the island
    is dropped at rung 1 (no window involved). No lattice shift reaches it (screen below).
- Lattice-shift screen (`diag_shift.py`, the rung-1 union on 10 sub-voxel shifts; partial, stopped once the fixes were
  clear): shift (0.225, 0, 0) makes the rung-1 union genus 0 on w_052 and w_027 x2 (no handle -> no window); the
  tip-class islands stay islands at every shift tried (k_011 42.3 mm3, w_002 25.4 mm3).

## 2. The coverage check: checks_v6 1.6 (dev `checks/checks_v6_1p6_dev.py`)

- New part `coverage` (gates `route_fork_coverage_loss_lt_1pct`, `route_siphon_coverage_loss_lt_1pct`): per piece and per
  reference (v4's and v5's SHIPPED collision_full, each in its own world, own curves, exact ownership), the reference's
  RCCA-owned cells of the gated stretch (span + max(blend) + 2 .. end - 2), carried by the piece's own transforms
  (piece_map, mirror included), kept where their v6 RCCA arclength is in the v6 gate and outside v6's handle windows
  (+- 0.5 mm). LOST REAL WALL = carried cells farther than 0.3 mm from the v6 union AND on the piece's own scanned
  surface (<= 0.3 mm) AND outside the v6 union. Deficit = lost area / gated area; gate < 1 % on both references; a
  deficit >= 0.1 % (and < 1 %) is accepted and FLAGGED (D27).
- New part `folds` (gate `obj_no_fold_over`): edge-adjacent face pairs with normals opposite within 1 deg on the OBJ
  (gate) and on collision_full (information).
- Validation: w_052_left mr_026 reproduces the verifier (v4: 62.59 mm2 uncovered vs 62.97; lost 60.36 mm2 = 4.68 %).
  Baseline on all 223 anatomies_v6 (`/s3/cov_v6_0`, summary `v6_work/s3fix/cov_v6_0_summary.txt`): deficit >= 1 % on
  exactly the verifier's 8 pieces; 0.1-1 % on 8 pieces (the verifier's 5 + w_008_right mr_001 fork 0.43 %,
  w_047_left mr_027 siphon 0.22 %, w_034_left mr_011 siphon 0.14 %); folds: OBJ only m_022_left mr_021_L,
  collision_full only w_017_left mr_014_L (information, as v4 / v5 each have one).

## 3. The fixes (bake_v6 1.2, dev `bake/bake_v6_1p2_dev.py`; defaults = 1.1)

- L1 LATTICE SHIFT (window class): the bake lattice origin (field and every capsule) moved by a sub-voxel shift; same
  surfaces, centrelines, ownership; flagged `d27_bake_lattice_shift_*`. Shift (0.225, 0, 0): w_052_left mr_026,
  w_027_left mr_022, w_027_left mr_026 bake at rung 1 with NO handle window; coverage deficit 0.0 on v4 and v5
  (`/s3/checks/A`).
- L2 ROUTE-ISLAND BRIDGE (tip class): on the ladder's final state, each island of lower_R / siphon_R's own scanned lumen
  that keep_largest would drop (>= 50 % of its points on the section's surface, median <= 6 mm off the RCCA tube) is
  joined to the main body by ONE tube of radius 0.5 mm along the shortest island -> main segment (+ 0.3 mm each end);
  kept only while the union stays genus 0. One bridge per island cannot form a handle. Flagged
  `d27_bake_route_island_bridge_<n>`. w_012_left mr_003_L: 1 island (980 cells, 24.0 mm3, s 104.8-116.8, gap
  0.84 mm) bridged by a 1.44 mm tube; genus 0; coverage deficit 0.0 (`/s3/checks/B`).
- Fold-over (should-fix 3): `s3fix/fold_fix.py` 1.0 on m_022_left mr_021_L: the edge flip is invalid (edge c-d exists);
  vertex 7648 is a degree-3 spike pushed through its own 1-ring (faces 15297 / 15302 / 15315) -> removed and its ring
  closed by one face [7738, 7647, 7737] (= a half-edge collapse); the OBJ stays clean (is_clean), both routes
  navigable, every centreline point inside, 0 folds, VA lumen unchanged (RVA 1.124 / LVA 0.854 mm); 23500 -> 23498
  triangles. Re-cuts from collision_full at 23500-23000 all reproduce the same fold (tested, rejected).

### 3.1 The fix ladder per piece (D27, applied without stopping; every choice flagged)

1. L1: re-bake on a shifted lattice, shifts tried in a fixed order (0.225, 0, 0), (0, 0.225, 0), (0, 0, 0.225),
   (0.225, 0.225, 0.225) mm; the first whose bake passes and whose coverage deficit is < 1 % on both references wins.
2. L2: the route-island bridge at shift 0.
3. L1 + L2: the bridge on each shift, same order.
4. v5's geometry for the piece (flag `right_fork_v5_geometry_D27`): not needed.
The tip class skipped rung 1: its islands are separate from the main lumen in the scan (0.62 mm gap at 0.15 mm), and
the lattice screen showed them islands on every shift tried.

| piece | class | dev runs (`/s3/checks/<tag>`) | chosen |
|---|---|---|---|
| w_052_left mr_026 fork | window | A: L1 (0.225,0,0) deficit 0 | L1 (0.225, 0, 0) |
| w_027_left mr_022 fork | window | A: L1 (0.225,0,0) deficit 0 | L1 (0.225, 0, 0) |
| w_027_left mr_026 fork | window | A: L1 (0.225,0,0) deficit 0 | L1 (0.225, 0, 0) |
| w_051_right mr_008_L siphon | window | A: L1 (0.225,0,0) 1.41 % (a siphon window again); C1 (0,0.225,0) fails; C2 (0,0,0.225) deficit 0, rung 1; (C3 (0.225,0.225,0.225) also 0; C0 L2 alone fails) | L1 (0, 0, 0.225) |
| k_011_left mr_012_L fork | tip | B: L2, 1 of 2 islands bridged (the other would close a handle), deficit 0 | L2 |
| w_012_left mr_003_L fork | tip | B: L2, 1 island (24 mm3) bridged, deficit 0 | L2 |
| w_008_right mr_003 fork | tip | B: L2, 3 of 4 islands bridged, deficit 0 | L2 |
| w_002_right mr_013_L fork | tip | B: L2 at 0 fails (1.46 %: the island holding the lost wall, s 111-117, is a ring -- genus 1 by itself, `diag_island.py`; a binary closing of its voxels adds nothing, `diag_ringclose.py`); D1 L1+L2 (0.225,0,0): all 3 islands bridged, deficit 0; (D2 (0,0.225,0) deficit 0 but its OBJ got a fold-over pair -- the new fold gate refuses it; D3 (0,0,0.225) ring stays) | L1 + L2 (0.225, 0, 0) |

### 3.2 Tools installed (one frozen toolchain for the final round)

- `install_s3fix_tools.sh` (2026-10-08 06:17 CEST): bake_v6 1.2 (md5 3a291c2a; 1.1 frozen as `bake/bake_v6_v1_1.py`,
  b3731df4), checks_v6 1.6 (883cb77e; 1.5 frozen as `checks/checks_v6_v1_5.py`, b7b6465b), both byte-identical to the
  tested dev copies; NEW manifest `build/TOOLCHAIN_s3fix.md5` = **2288f778** (97 files: d65b7209's 72 with the two new
  versions, + the frozen copies and the s3_fix scripts). The s2 manifest d65b7209 is kept unchanged as the s2 record (its
  two changed entries resolve to the frozen copies). Every final-round worker checks 2288f778 with `md5sum -c` first.
- bake_v6 1.2 with both options off is 1.1: the only code paths it adds are behind `--lattice-shift` (0 -> the original
  `_grid`) and `--bridge-route-islands` (the hook returns at once). Determinism: the final-round bakes reproduce the
  dev-round bakes of the same options bit for bit (collision_full md5 equal; the OBJ equal but for meshio's timestamp
  comment line).

## 4. The final round (manifest 2288f778; `/s3/trees/final`, `/s3/checks/final`, logs `/s3/logs/final*.batch.log`)

- Re-bakes (`batch_rebake.sh final_rebake.txt final`, bake_v6 1.2 installed; each from a fresh copy of `full/tree3/<n>`):

| anatomy | options | bake rung | windows after | bridges (island s, mm3, bridge length) | seconds |
|---|---|---|---|---|---:|
| w_052_left mr_026 | L1 (0.225, 0, 0) | 1 | none (was lower_R [108.99, 110.24]) | -- | 189 |
| w_027_left mr_022 | L1 (0.225, 0, 0) | 1 | none (was lower_R [60.0, 62.0]) | -- | 189 |
| w_027_left mr_026 | L1 (0.225, 0, 0) | 1 | none (was lower_R [59.81, 61.45]) | -- | 216 |
| w_051_right mr_008_L | L1 (0, 0, 0.225) | 1 | siphon_R [201.69, 203.12] = v4's pinned stretch (was [200.45, 205.45]) | -- | 177 |
| k_011_left mr_012_L | L2 | 2 (scD_R VA window, as before) | as before | 1 of 2: s 110.8-115.8, 16.9 mm3, 1.12 mm (the other island would close a handle) | 386 |
| w_012_left mr_003_L | L2 | 1 | as before | 1 of 1: s 104.8-116.8, 24.0 mm3, 1.44 mm | 197 |
| w_008_right mr_003 | L2 | 2 | as before | 3 of 4: s 77.8-84.8 39.0 mm3 1.39 mm; s 113.7-120.7 31.3 mm3 1.36 mm; s 85.8-88.8 0.8 mm3 2.74 mm | 547 |
| w_002_right mr_013_L | L1 (0.225, 0, 0) + L2 | 2 | siphon_L / siphon_R / lower_L VA-pool windows | 3 of 3: s 111.2-117.2 12.8 mm3 1.78 mm; s 108.2-113.2 8.1 mm3 1.65 mm; siphon s 222.0-228.0 4.4 mm3 1.38 mm | 647 |

- m_022_left mr_021_L: `fold_fix.py` on a fresh copy of tree3 (spike vertex removal, as in the dev test), union unchanged.
- The 8 record-only anatomies: fresh copies of tree3, their checks4 record + checks_v6 1.6 parts coverage and folds.
- Full checks_v6 1.6 on the 9 geometry-changed anatomies (`/s3/checks/final/<n>.json`, every part): all pass every gate
  (strict audit, D22 fork / siphon, RCCA prefix, coverage fork / siphon, OBJ fold-over, VA lumen, orientation tier A /
  blocking, loader, clearance, 1 mm joins). Coverage deficit 0.0 on v4 and v5 for every re-baked piece. D22 reference
  after the fix: the bridged islands are real scan the D13 baseline (v4 without its tip) never showed, so the bridged
  forks now read against their own scanned surface (D26, flagged): k_011_left mr_012_L fork (v4 0.965 -> own 0.995; its
  siphon stays own surface + D27 window), w_012_left mr_003_L fork (0.970 -> 0.998), w_008_right mr_003 fork
  (0.922 -> 0.997), w_002_right mr_013_L fork (0.941 -> 0.997) and siphon (0.984 -> 0.994: its siphon island was
  bridged too). w_051_right mr_008_L fork stays own surface (as before). w_052 / w_027 x2 / m_022: v4, 1.0.
- The 8 record-only anatomies: coverage / folds parts pass (deficits 0.14-0.77 %, flagged), the carried parts as before.
- D27 repair windows (checks_v6 1.5's rule, unchanged in 1.6, applied automatically): including the bridged islands put
  some cells of them where the bake's section field is the CAPSULE (the scan reaches beyond 1.8 r + 1 mm): w_002_right
  mr_013_L fork 12.0 mm2 (s 108.3-116.5; own 0.988 -> 0.997; 75 % of the cells also capsule-clipped by v4's bake) and
  w_008_right mr_003 fork 56.6 mm2 (s 78.8-120.3; own 0.968 -> 0.997; 90 % also clipped in v4's world). Both flagged
  `d27_d22_fork_repair_window_*`; their coverage against the v4 / v5 shipped wall is 0.0 lost.
- SOFA RCCA (`batch_sofa.sh`, sofa_rcca_v6 1.0, seeds 777 / 778 / 779 x 300, on the final files; `sofa_summary.py` ->
  `/s3/sofa_summary_final.json`): 9 / 9 PASSED, tip never outside. Reach per anatomy before -> after (target seeds):
  w_002_right mr_013_L 1 -> 0 (its one target seed now turns into the ECA at the bifurcation, as its other two did before;
  fork pose unchanged), w_008_right mr_003 0 -> 1, w_012_left mr_003_L 1 -> 0 (siphon 3 -> 2), k_011 2 -> 2, w_051 3 -> 3,
  w_052 1 -> 1, w_027 mr_022 0 -> 0 (siphon 2 -> 3), w_027 mr_026 3 -> 3, m_022 0 -> 0. All 223 after: seeds past seam 1
  + 20 mm 664, past the bifurcation 637 (638 before), siphon 511 (512), target 373 (374); anatomies 223 / 218 / 204 / 182
  (223 / 219 / 205 / 183). The 25-anatomy set holds none of the 9: target 36 / 75 vs v5 46 / 75, siphon 52 vs 54 (as
  s2_checks measured).
- Stamp (`stamp_prov_s3.py`, stamp_prov 1.1 -> `/s3/prov_final`, summary `stamp_prov_s3.json`): 17 / 17 pass; the
  assertion "source provenance == tree3's outside the keys s3_fix owns" held on all 17.
- Copy (`copy_v6_s3.sh`, copy_v6 1.2, host side): the 17 folders staged (file set equal to the current folder's),
  replaced (old folders kept in `v6_work/s3fix/replaced/`), and the whole folder verified against the new manifest
  `v6_work/s3fix/anatomies_v6.sha256` (sha256 024edcd2...): 223 anatomies, 5,136 files, `sha256sum -c` exit 0, no extra
  or missing file. `diff -rq` against the replaced folders: re-baked -> OBJ, collision_full, mesh_v6.json, provenance;
  m_022 -> OBJ, mesh_v6.json, provenance; record-only -> provenance only; Centrelines_comb unchanged everywhere.
- Figures: `monitoring/figure_carotid_anatomies_v6.py` 1.2 (unchanged) on the 17 changed anatomies of the final folder
  (`--build-json` = their `v6_checks.derived_flags`, so the titles carry the same flag classes as the s2 renders; same
  shared window as the s2 run) -> `saved/figs/carotid_v6/` (17 replaced, the old ones kept in `v6_work/s3fix/figs_replaced/`;
  records `v6_work/s3fix/figs/records/`). Checked w_052 and w_002: checks_v6 1.6, the D22 / D27 title, the new flags.
- Split on the final folder (`final_set.sh`: `v5_split.py anatomies_v6 --baseline split_v4.json`): exit 0, 45 compared,
  none lowered; its baseline rows are identical to s2_checks' run (the curves did not change) (`/s3/set/split_s3.json`).
- Loader on the final folder (`loader_set.py`): 223 / 223 (fingerprints unique, schedule covers all, RCCA 206.2-261.0 mm).

## 5. Records (must-fix item 2)

- **BUILD_v6.json** regenerated from the final provenance by `records/build_v6_record.py` **1.1** (1.0 frozen as
  `build_v6_record_v1_0.py`; patches `s3fix/patch_bvr.py`, `patch_bvr2.py`): per anatomy `derived_flags` (provenance
  `v6_checks.derived_flags` on the 17 re-stamped; else the left map deviation from provenance + orient accepted flags from
  `_records/per_anatomy/checks4`) and `s3_fix` kind; `flags.derived_flags` (left_pose_map_bounded_dev 29, orient:D23 5,
  orient:va_target_20_30 11); `checks.coverage_v4_v5_shipped` (labelled copy of `results/v6_coverage_s3.json`: before /
  after lists); `s3_fix` block (counts 8 / 1 / 8, lattice shift 5, bridge 4, 8 islands, every changed anatomy with its
  fix, coverage and flags, and the set-level checks after the fix); D27 text; toolchain (checks manifest 2288f778 x17 /
  d65b7209 x206; bake_v6 1.2 x8). Checker `check_build_record.py` **1.1** (+ s3_fix recounts, derived-flag recounts from
  provenance / checks4): **123 checks, 0 differences** on the installed copy; the information rows that differ from the
  s2 collection are s3_fix's changes (noted in the file). Installed copy sha256 b9e44e28...; the s2 BUILD kept as
  `v6_work/s3fix/replaced/BUILD_v6_s2.json`.
- **_records** (`s3fix/copy_records_s3.sh`): + `results/v6_coverage_s3.json`, `results/s3_fix/` (folder manifest after
  the fix, copy verification, stamp summary, SOFA summary, split, loader, set summary, item lists, logs, coverage before
  the fix), `per_anatomy/checks_s3/` (17), `bake_s3/` (8), `fold_fix/` (1), `sofa/rcca_v6_s3/` (9),
  `checks/coverage_v1p6_prefix/` (223), `diag/s3_fix/` (diagnoses, shift screen, dev re-bake records), notes
  `s3_fix.md`, `verify_v6_final.md`, `s2_images.md`; README.md updated; `checks/build_record_check.json` replaced (both
  logged in `results/s3_fix/records_replaced.txt`); RECORDS.sha256 rebuilt: 1,466 files, `sha256sum -c` ok.
- **carotid_tools/v6** (`records/finalize_tools.sh` 1.1, `make_tools_v6.py` 1.1; 1.0 frozen): + the s3_fix manifest,
  the `s3_fix/` folder (every file of `scratchpad/v6/s3fix/`), bake_v6 1.2 / checks_v6 1.6 and their frozen
  predecessors, the records 1.1 scripts and frozen 1.0 versions. 253 copied files + 4 = 257. Restore test: 253 / 253
  sha256-identical and byte-identical to the scratchpad; `md5sum -c` of TOOLCHAIN_s3fix.md5 (2288f778) passes on all 88
  /sp entries of the restored layout; 463 manifest rows resolve, 2 not preserved (the known supplement entries); 63 repo
  rows md5-equal. MANIFEST.md updated (s3_fix row, folder, restore paragraph).
- **CAROTID_V6.md**: intro paragraph; tool lines; at a glance (D22 19 D26 anatomies / 3 D27 pieces; the coverage row);
  D27 row; collision mesh (bake_v6 1.2, rung 129 / 94, windows 124 / 209); gates table (coverage gate, OBJ fold gate,
  checks_s3); the right route (references 205 / 18 and 220 / 3, the 19 D26 anatomies, D27 on w_002 / w_008, the coverage
  definition, the before / after table, cause, fixes); SOFA (after / before table, the 25-set unchanged with the reach
  note); split / loader / copy after s3_fix; flags (the corrected statement, derived flags, new rows); how it was built
  (steps 8, 9); record corrections (the 4 -> 8 coverage list, the flags statement); known limitations (the 8 accepted
  deficits, the synthetic bridges, 41 anatomies without a target seed); follow-ups (bake guard instead of the done
  coverage item); records; reproduce (final manifest 2288f778).

## 6. Decisions (D27: the recommended minimal option, flagged)

- Fix ladder per piece: L1 lattice shift in a fixed order -> L2 bridge -> L1 + L2 -> v5 geometry. A lattice shift is a
  pure resampling (no synthetic surface), so it is tried first; the bridge adds 1.1-2.7 mm of 0.5 mm-radius tube per
  island (8 islands on 4 anatomies), the minimal synthetic link, as v4 / v5 had one (their RECA tip tube) for the tip class.
- Flag threshold for accepted deficits 0.1 % (below that: discretisation noise on 10 pieces); gate 1 %.
- Record-only re-stamp for the 8 accepted pieces (their flag travels with the anatomy, as D26 / D27 flags do); the other
  206 provenance files are not rewritten (writes limited to changed anatomies), so the "every flag in provenance"
  statement is corrected in the doc and the derived flags go to BUILD_v6.json per anatomy (+ provenance on the 17).
- D27's capsule repair window (checks_v6 1.5 rule, unchanged) applied automatically to the two bridged forks whose islands
  reach past the capsule; reported and flagged, as the rule says.
- The verifier's percentages (1.3-4.9 %) are "uncovered / area"; checks_v6 1.6 reports "lost real wall / area" (uncovered,
  on the own scan, outside the union): 1.0-4.7 % on the same 8.

## 7. Open issues (none blocking)

1. SOFA reach is a heuristic: w_002_right mr_013_L's one target-reaching seed now turns into the ECA (pose unchanged; the
   mesh moved by the lattice shift / bridges); all 223 still pass soundness. 41 anatomies have no target seed (40 before).
2. Bridged islands are real scan joined by synthetic 0.5 mm tubes; one island of k_011 (s 115.7-122.7, 25 mm3) and one of
   w_008_right mr_003 (s 72.8-75.8, 1.9 mm3) stay dropped because bridging them would close a handle -- they hold no wall
   v4 / v5 showed (coverage deficit 0).
3. collision_full of w_017_left mr_014_L keeps one fold-over pair (information; v4 / v5 have one each on some anatomy);
   the OBJ gate is the training mesh.
4. The bake's shrink step can still cut a sliver loose on a future bake; a bake-time guard is a follow-up (CAROTID_V6.md).

## 8. Files written

- Scratchpad `v6/s3fix/`: runners (`dr_s3.sh`, `p.sh`, `batch_rebake.sh`, `final_other.sh`, `batch_sofa.sh`,
  `batch_cov.sh`, `batch_frag.sh`, `batch_shift.sh`, `final_set.sh`), diagnoses (`diag_frag/sign/ray/neck/shift/island/
  ringclose.py`, `peek_*.py`, `sum_*.py`), `fold_fix.py`, `stamp_prov_s3.py`, `copy_v6_s3.sh`, `copy_records_s3.sh`,
  `cov_results.py`, `sofa_summary.py`, `s3_set.py`, `make_fig_flags.py`, `install_s3fix_tools.sh`, patch scripts, item lists.
- Scratchpad tools: `v6/bake/bake_v6.py` 1.2 (dev `bake_v6_1p2_dev.py`; 1.1 frozen `bake_v6_v1_1.py`), `v6/checks/
  checks_v6.py` 1.6 (dev `checks_v6_1p6_dev.py`; 1.5 frozen `checks_v6_v1_5.py`), `v6/build/TOOLCHAIN_s3fix.md5`;
  records 1.1 (`build_v6_record.py`, `check_build_record.py`, `make_tools_v6.py`, `finalize_tools.sh`, `MANIFEST_head.md`;
  1.0 frozen `*_v1_0.*`); `v6/results/v6_coverage_s3.json`, `v6_s3_set.json`.
- Work `carotid_data/v6_work/s3fix/`: diag, shift, cov_v6_0 (+ summary), trees / bake / iso / checks (A, B, C0-C3, D1-D3,
  fold, final), sofa, prov_final, staged, replaced (the 17 old folders + the s2 BUILD), figs (+ figs_replaced),
  set, logs, anatomies_v6.sha256, copy_verify_s3.json.
- Repo: `carotid_data/anatomies_v6/` (17 anatomy folders, BUILD_v6.json, _records additions), `carotid_tools/v6/`
  (re-staged), `CAROTID_V6.md`, `saved/figs/carotid_v6/` (17 PNGs). No git writes; user.md / "user copy.md" untouched.
