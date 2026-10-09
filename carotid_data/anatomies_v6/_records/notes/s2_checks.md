# s2_checks -- the set-level checks on carotid_data/anatomies_v6

Agent s2_checks, 2026-10-07. State: `state/s2_checks.md`. Code: `scratchpad/v6/s2checks/` (runner `dr_s2c.sh`, containers
`s2c_*`). Work outputs: `carotid_data/v6_work/s2checks/` (/s2c). Mounts: /repo read-only, /full (carotid_data/v6_work/full,
the build's records, tree3 and unions) read-only, /s2c read-write. No git writes; no repo-tracked file touched.
Results: `results/v6_checks.json` (everything, per anatomy) and `results/v6_final_summary.json` (the short form).

## Answer so far

**Complete (2026-10-07 20:40 EDT). Every set-level invariant passes on carotid_data/anatomies_v6** (`results/v6_checks.json`,
`results/v6_final_summary.json`): split exit 0 with none of 45 lowered; training loader 223 / 223; **SOFA on the RCCA
training route 223 / 223 passed** (soundness; tip at most 0.19 mm outside); right route D22 fork + siphon 223 / 223 and
the RCCA prefix bit-identical 223 / 223; orientation tier A 0 except D23 (w_017_left mr_002_L), blocking 0; VA polyline
lumen >= 0.65 on all 446 VAs (min 0.664; v5 had 4 below); genus 0 / closed / 0 SI / centrelines inside 223; clearance
>= 0.6 (min 0.614); 1 mm joins 223 under geom_v6's rule. ICA-terminus AP > 40 mm: 3 (v5 108); fork heading > 45: right 5
(the R5 forks) / left 7 (v5 213 / 140). D14 VA entry on the 25-anatomy set: 48 / 75 seeds (22 / 25 anatomies) vs v5 33 / 75
(19 / 25) vs v4 74 / 75. Reported for attention: the 4 w_042_left anatomies' RCCA is never passed beyond the bifurcation
by the scripted wire (also in v4 and v5: inherited); RCCA target reach on the 25-set is lower than v5 (36 vs 46 seeds);
4 pieces show v4 wall missing from v6 (coverage < 0.99, real scanned wall 2-3 mm outside the v6 mesh next to bake
handle windows; w_052_left mr_026 fork 65 mm2, 5 %) -- not gated, root cause not established.

## 1. Method
- Every measuring tool is a file of the build's frozen manifest `build/TOOLCHAIN_s2build.md5` (d65b7209, `md5sum -c` passes
  before the runs): v5_split.py, sofa_rcca_v6.py 1.0, sofa_va_entry.py 1.1, va_entry_geom.py 1.0, va_lumen_poly.py,
  orient_gates.py 1.2, checks_v6.py 1.5, loader / eve from the repo. New set-level scripts (this agent, scratchpad only):
  `orient_set.py` (orient_gates measure_tree on a whole tree + checks_v6 part orient's D8 / D23 post-processing, ICA AP as
  compose_v6.ica_ap re-implemented on the tree), `loader_set.py` (the training loader on the whole folder), `ident_check.py`
  (anatomies_v6 vs the checked tree), `collect_s2c.py` (the two result files), `sofa_worker.sh` (idempotent SOFA worker:
  claim dirs, done = output exists, slot cap file).
- What is re-measured directly on anatomies_v6 vs carried from the build's per-anatomy records: split, loader, SOFA, D14,
  the geometric probe, the VA polyline lumen and the orientation gates run on anatomies_v6 itself. The right-route D22 /
  D26 / D27 check, the strict mesh audit, the wall detail and the clearance / 1 mm turn gates come from the build's checks
  (`/full/checks4`), which measured /full/tree3 + the undecimated unions /full/iso; ident_check shows anatomies_v6 is that
  tree file for file (sha256), so those records hold for anatomies_v6. A fresh full part-route run straight from
  anatomies_v6 on 4 anatomies confirms it (section 8).

## 2. Identity of anatomies_v6 with the checked tree
- `ident_check.py` -> `/s2c/ident/ident_v6_tree3.json`: 223 = 223 names; 4,690 files compared, every one sha256-equal to
  /full/tree3; each provenance.json = tree3's + exactly one key (`v6_checks`); the only extra files are the licences
  (LICENSE_v6_donors.txt; LICENSE_VMR_0248.txt kept); every anatomy has its undecimated union in /full/iso and a complete,
  passing checks4 record. **ok.**

## 3. Split (`/s2c/split/split_v6.json`, log `split_v6.log`)
- `v5_split.py /repo/carotid_data/anatomies_v6 --baseline /sp/v5_va/results/split_v4.json`: **exit 0, 45 quantities
  compared, none lowered.** Four went up (strict div6x6: k=16 train 168 -> 169, kept 184 -> 185; k=17 train 167 -> 169,
  kept 184 -> 186). V: donors 25 in 25 TopBrain patients, none spanning more than one patient, on every definition
  (one-to-one; the build record's donor check says the same: 25 units onto 25 patients).

## 4. Training loader
- `loader_set.py` (the training call: DualDeviceNavTopBrain(anatomy_dir=anatomies_v6), one TopBrainAnatomySet with every
  anatomy, then each selected through its fingerprint): **223 / 223 pass** -- anatomy_count 223 = the 223 folders, names
  equal, fingerprints unique, the schedule covers all 223 in one pass; every anatomy: 18 branches with the same names,
  finite, radii > 0, mesh_path = its own OBJ, the shared insertion inside its OBJ, RCCA 206.2-261.0 mm (> 138).
- Complemented by the build's deterministic per-anatomy loader_one on the final folder (/full/load_v6: 223 / 223) and the
  checks gate (223 / 223); every SOFA run (section 6) loads one anatomy into SOFA through the same loader.

## 5. Orientation gates, ICA-terminus AP offsets, fork headings (`/s2c/orient/orient_set_v6.json`, `_v5.json`)
- v6: **tier A on 1 anatomy (w_017_left mr_002_L RVA_top, D23 accepted) -> 0 after D23; v6-blocking after D8 / D23: 0.**
  Accepted flags: va_target 20-30 mm on 11 tops (D8), D23 on w_017_left mr_002_L (5 flags: RVA_top va_lateral / heading /
  va_gap_v6 / va_target, LVA_top va_gap_v6). v5 on the same code: tier A 141, blocking 223 (reproduces the selftest).
- ICA-terminus AP offset > 40 mm: **v6 3** (all RICA: w_003_right mr_001 46.0, w_018_right mr_007_L 53.3,
  w_027_right mr_016 45.3) vs **v5 108** (RICA 66, LICA 59 -- exactly the compose reference, so the tree re-implementation
  matches). Medians RICA 6.9 / LICA 4.2 mm (v5 29.5 / 20.8). Tree-measured values equal the compose records on all 446
  ICAs (max |diff| 0.0 mm).
- Fork heading |yaw vs the Zenodo map| > 45: **right 5** (exactly the 5 R5 forks on v5's pose: w_003_right mr_001,
  w_014_right mr_010 / mr_011_L / mr_027, w_036_right mr_018_L) vs v5 213; **left 7** (of the 8 "not on map" left forks)
  vs v5 140. Fork tilt > 40: 0 (v5 0). Mirrored (D20): right 113, left 101 (none contralateral after mirroring; v5 113 /
  101 contralateral). ECA anterior of the ICA: right 207, left 191 (v5 78 / 122).

## 6. SOFA (`/s2c/sofa/<kind>_<tag>/<n>.json`, worker log `/s2c/sofa/logs/worker.log`)
Runs: sofa_rcca_v6 1.0 and sofa_va_entry 1.1 (manifest), seeds 777 / 778 / 779 x 300 steps each, from the training
insertion through the training loader; 4 containers x 3 slots (single-threaded SOFA, ~1 CPU and 0.5 GB per slot,
~6 min per anatomy); 223 + 75 + 25 + 8 runs, 17:38-20:38 EDT.
### 6a. RCCA training route, all 223 (the route moved: D19 / D20)
- **223 / 223 PASSED** (0 exceptions, 0 non-finite steps, 18 branches, the loader's mesh = the anatomy's OBJ, insertion
  inside, tip never > 0.5 mm outside the wall: max 0.193 mm). Device body points outside the wall (reported, as v4 / v5):
  max per anatomy median 1.80 mm (max 4.50); on the 25-set v6 median 1.96 vs v5 2.25.
- Reach (669 seeds): past seam 1 + 20 mm 664; past the bifurcation 638 (219 / 223 anatomies with >= 1 seed); siphon 512
  (205); target 374 (183). 40 anatomies have no seed reaching the target (list in v6_checks.json sofa_rcca).
- **Not past the bifurcation on any seed: the 4 w_042_left anatomies** (mr_005_L, mr_010_L, mr_023, mr_026_L; mirrored
  w_042_left fork on the map pose). The wire turns into the RECA at s ~73 mm on every seed and the hook-back fails. **The
  same in v5 and v4** (run with the same tool and seeds: 0 / 3 seeds past the bifurcation, max s 75.8-82.0) -> inherited,
  not caused by v6. The fork's own take-off (`/s2c/sofa/bif_angles_w042.json`, identical in v4 / v5 / v6): the ICA leaves
  45 deg off the CCA axis, the ECA only 20.6 deg, equal calibre (3.66 / 3.56 mm), so the scripted wire follows the ECA.
  Flagged (sofa_rcca_not_exercised_past_bifurcation 4).
- Against v5 on the 25-anatomy set (same tool, seeds): passed 25 / 25 both; seeds past the bifurcation 73 vs 72, siphon
  52 vs 54, target **36 vs 46** (anatomies with a target seed 19 vs 21). Fewer target seeds in v6 on 10 anatomies, more on 5
  (per anatomy in v6_checks.json). The scripted controller is a heuristic, so reach is reported, not gated (as in v5).
### 6b. D14 VA entry on the v5 25-anatomy set (+ v4 controls), not a gate
| tree | SOFA passed | seeds >= 20 mm into the RVA | anatomies with an entry | donor RVAs: seeds / anatomies | geometric probe R / L (25) | probe R / L (all 223) |
|---|---|---|---|---|---|---|
| v6 | 25 / 25 | **48 / 75** | **22 / 25** | 42 / 69, 20 / 23 | 23 / 20 | 204 / 188 |
| v5 | 25 / 25 | 33 / 75 | 19 / 25 | 27 / 69, 17 / 23 | 19 / 22 | 185 / 180 |
| v4 | 25 / 25 | 74 / 75 | 25 / 25 | (host VAs) | 25 / 25 | 223 / 223 |
- v6 enters more often than v5 on 12 anatomies, less on 4 (m_030_right mr_013_L 2 -> 0, w_007_left mr_021_L 1 -> 0,
  w_003_left mr_010 3 -> 1, w_036_left mr_011 2 -> 1). Median depth of entry (median over anatomies of the per-anatomy
  median) 42.8 mm vs v5 5.0 vs v4 138.5. Still below v4 (the donor junctions' take-off is harder than v4's host stub).
- The v5 build's own record (10 / 25 vs v4 24 / 25) used the older schedule without a device reset per seed; the numbers
  above re-measure v5 and v4 with the same tool and seeds as v6.

## 7. VA polyline lumen (`/s2c/lumen/va_lumen_poly_v6.json`; run directly on every anatomies_v6 OBJ + collision_full)
- **446 VAs on 223 OBJ: minimum 0.664 mm, none below 0.65, 0 samples outside** (lowest: w_050_left mr_004_L LVA 0.664
  donor, k_004_left mr_010 LVA 0.668 TopBrain, w_036_left mr_011 LVA 0.678). collision_full min 0.826. v5 (same tool):
  min 0.547, 4 VAs below 0.65 (w_025_right mr_021_L L 0.565 / R 0.547, w_027_right mr_022 R 0.630, w_047_left mr_023 R
  0.637) -- all four fixed. Equal to the checks' lumen part to 0.0 mm.

## 8. Right route (D22 / D26 / D27, D13 baseline, relief information; checks4 records + a fresh re-run from anatomies_v6)
- Gates: **D22 fork 223 / 223, siphon 223 / 223, RCCA prefix bit-identical 223 / 223.** Reference used: fork v4 209 /
  own surface 14; siphon v4 221 / own surface 2 (D26 on 16 anatomies: the 9 R-type mirrored forks, w_046_left mr_012 /
  mr_020_L / mr_026, w_047_left mr_023 (siphon), w_048_left mr_010_L, w_051_right mr_008_L, k_011_left mr_012_L
  (siphon)). D27 on 1 (k_011_left mr_012_L siphon: repair window 244 cells, 17.79 mm2, s 206.75-207.5, 100 % also clipped
  by v4's bake). Values under the reference used: fork min 0.9915, siphon min 0.9923; v4-only: fork min 0.929 (w_033_left
  x5), siphon min 0.9776 (k_011).
- D13 baseline: v4 187, v4 without its RECA tip 36.
- Reported, not gated: blend zone (fork area median 161 mm2, within 0.3 mm median 0.72, max deviation median 1.23 /
  max 2.60 mm; siphon 115 mm2, 0.78, 0.79 / 2.27 mm); bake handle windows on the fork in 28 anatomies, on the siphon in
  31 (32 counting v4's mapped windows; window cells within 0.3 mm: fork median 1.0 / min 0.91, siphon median 0.91 /
  min 0.22; per anatomy in v6_checks.json); relief ratio (information) fork median 1.001 (min 0.821; 52 below 0.99), siphon
  1.002 (min 0.907; 18 below 0.99).
- Fresh full part route (checks_v6 1.5, `--parts route`, straight from anatomies_v6 + /full/iso; `/s2c/route_spot`) on
  k_011_left mr_012_L (D13 v4_no_tip, D26 + D27), w_033_left mr_007 (D26 fork), w_052_left mr_026 (coverage 0.950),
  w_051_right mr_008 (control): the whole D22 record of every piece is identical to the checks4 record (every key; the
  1.4 records only lack the key d27_flag, which 1.5 writes as false), and the relief ratio and D13 baseline too.

## 9. Coverage below 0.99 (reported, not gated): where v4's wall does not reappear (`s2checks/diag_cov.py` 1.2, `/s2c/diag_cov12`)
D22 gates one direction (every v6 RCCA cell lies on v4's surface); coverage looks the other way (v4's exposed RCCA wall,
carried by the piece's transform, within 0.3 mm of the v6 union). Four pieces are below 0.99 (the build listed them,
undiagnosed). diag_cov re-runs checks_v6 1.5's route_d22 unchanged, recompiled in memory with one statement that keeps the
coverage cells, and then looks at the cells that are off:
| piece | off area | where (v6 RCCA s) | v6 handle window | off v6 by (median / max) | inside v6 union |
|---|---|---|---|---|---|
| w_052_left mr_026 fork | 65.0 mm2 (5.0 %) | 110.8-125.5 | 109.0-110.2 | 1.75 / 2.66 mm | 0 |
| w_027_left mr_022 fork | 26.7 mm2 (2.1 %) | 49.5-59.3 | 60.0-62.0 | 3.02 / 3.74 mm | 0 |
| w_027_left mr_026 fork | 26.9 mm2 (2.1 %) | 49.5-59.3 | 59.8-61.5 | 3.02 / 3.74 mm | 0 |
| w_051_right mr_008_L siphon | 24.1 mm2 (1.7 %) | 206.0-209.5 | 200.5-205.5 (v4's 201.6-203.1) | 2.03 / 2.48 mm | 0 |
- Every off cell lies OUTSIDE the v6 union, and on the piece's own scanned surface (median 0.03-0.10 mm from it): this is
  real scanned wall that v6's mesh does not show; there the v6 wall lies 2-3 mm inside the scan (lumen locally narrower
  than the scan on a strip of the ICA wall). D22 still reads 1.0 because the v6 cells nearby lie on v4's surface (median
  0.003-0.009 mm) and on the own surface.
- The bake's own section field there is the real surface (the capsule clips only 4-11 mm2 of it, and v4's bake clips the
  same amount); no VA or other tube comes within 1 mm (only the RCCA / RECA tubes, 1-3 mm away). Every off stretch sits next
  to a bake handle window of that section (the windows D22 reports and does not gate). w_052's final bake attempt dropped
  18 small fragments (107 mm3; v4's bake dropped 91 mm3 on the same anatomy). The root cause is NOT established here.
- Under D27's rule (deficit share of the piece, < 1 % accept + flag) these would be above the 1 % line if coverage were a
  gate; it is not (D22: reported). They stay flagged (d22_*_coverage_lt_0.99, provenance v6_checks) -- open issue.

## 10. VA wall detail (checks4 part va on the undecimated unions; v4 from v5's checks)
- 446 VA sides: 429 donor, 17 host (no donor since v5, D21).
- **donor real** (donor stretch minus the 25 mm seam zone and tube windows): v6 median **0.150** (p10 0.099, min 0.019) vs
  v5 0.134 (min 0.029); v4 host tube on the same side 0.000 (max 0.020); v6 > v4 host tube on 429 / 429. D5 (median >=
  0.10): yes.
- **subclavian stretch (v6 only, the donor's own junction, D17)**: 413 sides, median 0.743 (0.523-0.875), area median
  2136 mm2; v5 / v4 used the host's subclavian tube there.
- **whole VA** (origin -> end): v6 median **0.159** vs v5 0.136 vs v4 0.018; v6 > v4 on 438 / 446 (the 8 others are host
  sides, D21, where only the TopBrain top moved: 0.039-0.086 vs v4 0.043-0.107).
- **TopBrain piece**: v6 median 0.157, v5 0.106, v4 0.102.
- D15 split (donor_real inside / outside the artefact zones): outside v6 median 0.135 vs v5 0.084 (v6 / v5 median 1.45,
  min 0.31) -- NOT like for like: v6's donor stretch is measured from the donor's own junction to a re-posed splice, v5's
  from the stub; D15's identity gate is the donor-level wd_local check, not this.
- Lowest donor_real: song_AHMU1218141 L in 4 anatomies, 0.019-0.020 (v5 0.094-0.095). On w_007_left mr_017 the v6 donor stretch is 100.6 mm
  long (0-100.6) against v5's 170.6 mm (4.0-174.5): the re-posed VA is spliced to its TopBrain top much lower, so the two
  numbers measure different stretches of the donor. The whole VA is 0.146-0.147 (v5 0.157-0.159, v4 0.100-0.105).

## 11. Mesh and centreline gates (checks4 records, anatomies_v6 identical)
- Strict mesh audit 223 / 223: OBJ genus 0 223, closed 223, 0 self-intersections on OBJ / collision_full / union 223,
  every centreline point inside the OBJ 223.
- Clearance between placement groups on new points: min 0.614 mm (gate 0.6), 0 failures.
- 1 mm turns at every sampling phase (D25): 0 failures under geom_v6 1.1's rule (a new turn fails when > 60 deg AND
  sharper than the same v5 curve's own maximum + 1 deg). 33 curves carry a new-point turn > 60 deg that is no sharper than
  v5's (max 82.1 deg; RECA / LECA tip joins at s 37-47 mm, and VA tops: (RVA s ~180-195, LVA s ~155), the same angle as
  v5 to 0.6 deg) -- listed in v6_checks.json per_anatomy_gates.geom: turns the same curves already had in v5 (a rigidly
  moved piece keeps its own turns), not new seams.

## 12. Flags (counts; per anatomy in results/v6_build.json and v6_checks.json)
- Check flags stamped into provenance (v6_checks): D26 own-surface reference fork 14 / siphon 2; D27 repair window 1
  (17.79 mm2); D23 tier A accepted 1; coverage < 0.99 fork 3 / siphon 1.
- Build record (v6_build.json counts, unchanged): right fork mirrored (D20) 113, host-cut shift 89, R3 bend 60 3, R4 3
  (map dev 1, map dev + bend 60 2), R5 v5 pose 5, seam-1 bend > 45 5; left fork mirrored 101, substituted 22, map bounded
  deviation 29, not on map 8; ECA tip skipped right 20 / left 36; route baseline v4 without tip 36; route13 moved 1;
  ICA AP > 40 (R) 3; VA rungs right 198 / 12 / 1 + 12 host, left 215 / 3 + 5 host; D16 tube bridge R 8 / L 10, D17 tube
  R 15 / L 24, junction tube bridge R 8 / L 12; D9 rung 2 (v5 junction) R 12 / L 3, rung 3 R 1; va_target 20-30 R 9 / L 2;
  corridor: lateral excursion R 117 / L 93, medial R 5 / L 2, fallback host VA R 7 / L 6, inherited fail R 7 / L 1; course
  RMS > 20 inherited R 2; fix loop VA hard 3; orient va_target 20-30 11, D23 5; c4 planned 8; the remaining
  single-anatomy VA flags (gates_fail accepted on the terminal rung, pair flag, route-guard best effort 3) in v6_checks.json.
- New set-level flags (this agent): SOFA RCCA not past the bifurcation 4 (w_042_left x4, inherited from v4 / v5); no seed
  reaching the target 40; geometric VA entry probe not entering 54 sides (v5 81, v4 0); D14 fewer entering seeds than v5
  4 anatomies (set of 25).

## 13. Open issues
1. **Coverage < 0.99 (section 9)**: real scanned wall missing from v6's mesh on 4 pieces (w_052_left mr_026 fork 65 mm2 /
   5.0 %, w_027_left mr_022 and mr_026 forks 27 mm2 / 2.1 %, w_051_right mr_008_L siphon 24 mm2 / 1.7 %), 2-3 mm deep,
   next to bake handle windows. Not a gate (D22 reports it), flagged. Root cause not established; a figure or a targeted
   re-bake test would tell. If the orchestrator applies D27's residual rule to it, all four exceed the 1 % line.
2. **w_042_left x4**: the RCCA beyond the bifurcation is never reached by the scripted wire (ECA straighter than the ICA),
   identical in v4 and v5. A training-route property of that Zenodo fork, not a v6 change.
3. **RCCA target reach on the 25-set** 36 / 75 seeds vs v5 46 / 75 (the route moved, D19 / D20); reach is not a gate.
4. **D14** improves on v5 (48 vs 33 seeds) but stays below v4 (74 / 75); 4 anatomies enter less than in v5.
5. Wall detail: song_AHMU1218141 L's donor_real 0.019-0.020 in 4 anatomies (v5 0.094) measures a shorter stretch (the
   splice moved); whole VA comparable (0.146 vs 0.157). 8 host-VA sides (D21) read slightly below v4 on the whole VA
   (their TopBrain top moved).
6. Records left to later stages: CAROTID_V6.md, figures, the folder-level BUILD record.

## 14. Files
- Results: `results/v6_checks.json` (1.2 MB: invariants, every section, per anatomy), `results/v6_final_summary.json`.
- Scratchpad code `v6/s2checks/`: dr_s2c.sh, sofa_worker.sh, sofa_container.sh, make_items.py, orient_set.py,
  loader_set.py, ident_check.py, diag_cov.py 1.2 (1.0 / 1.1 frozen), bif_angles.py, collect_s2c.py, peek_rec.py.
- Work outputs `carotid_data/v6_work/s2checks/`: split/, loader/, orient/, lumen/, entry_geom/, ident/, route_spot/,
  diag_cov*/, sofa/ (items.txt, set25.json, rcca_v6 223, rcca_v5 29, rcca_v4 4, d14_v6 / v5 / v4 25 each, logs/,
  bif_angles_w042.json), collect_test/ (test runs).
