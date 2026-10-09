# verify_v6_final: independent final verifier of carotid_data/anatomies_v6

Agent verify_v6_final, 2026-10-07 (evening). State file: `state/verify_v6_final.md`.
Own code: `scratchpad/v6/verify_final/` (numpy / scipy; vtk only to read .vtp/.stl; PIL for the PNGs). Library
`vf_lib.py` (topology, self-intersection, winding number, exact point-to-triangle distance, Kabsch) is self-tested
(`test_lib.py`: a clean icosphere reads genus 0 / 0 intersections, two interpenetrating spheres 142 intersecting pairs).
Runner `dr_vf.sh` (containers `vf_*`, /repo read-only). Outputs `carotid_data/v6_work/verify_final/` (= /out below).
No git writes, no repo-tracked file written, no other agent's file edited.

## Answer (final): every gate reproduces; NO-GO on one point, real scanned wall lost on 8 right-route pieces

Reproduced with my own code, all 223 unless stated:
- anatomies_v6 is the built tree. Hashes match the build manifest and the staged copy, and provenance differs from
  tree3 only by the v6_checks key.
- Mesh: OBJ and collision_full are closed, manifold, single-component, genus 0 with 0 crossing self-intersections.
  Every centreline point is inside. VA polyline lumen minimum 0.664 mm. RCCA prefix bit-identical to v4.
- Orientation (validated on v5 against the audit): tier A 0 after D23; 0 blocking after D8 / D23.
- D22 / D26 / D27: fork minimum 0.9944, siphon minimum 0.9898 (k_011's D27 window only); within 0.01 of the build.
- 1 mm joins: every turn over 60° is a rigid copy of v5. Clearance: only same-curve readings at the adjacency limit.
- Split: exit 0, none of 45 lowered.
- Decisions D1-D27 all hold in the geometry: mirroring, map poses, D17 cut ends, D18 / D21 sides as v5, and the
  rest in section 10.
- SOFA RCCA: 223 records pass. Loader: 10/10 with my own call.
- Toolchain restore: byte-identical. Figures place every piece where provenance says.
- Read-only trees: unchanged.

Not holding:
1. **Real scanned wall lost on 8 pieces (1.3-4.9 % each, 22-63 mm2, new in v6: v5's mesh has it).** The build
   reports 4 of them. The other 4 (k_011_left mr_012_L fork, w_002_right mr_013_L fork, w_012_left mr_003_L fork,
   w_008_right mr_003 fork) were measured against "v4 without its tip", which hides the same wall, and recorded 1.0.
   The user's rules: D26 says no real wall may be lost. D27 says a residual deficit at or above 1 % of the piece takes
   the next ladder rung, else v5's geometry for that piece, flagged. That rule was not applied, and is the must-fix.
   See section 6.
2. **Records.** BUILD_v6.json, CAROTID_V6.md and v6_checks understate coverage (4 instead of 8 at or above 1 %). The
   doc's "every flag is written in provenance" is false for left_pose_map_bounded_dev (29) and orient:* (16).
3. **One fold-over on a training OBJ**, m_022_left mr_021_L: two adjacent triangles folded onto each other, at z 590 on
   the left side and away from the RCCA. v4 and v5 OBJs have none. MeshLab's self-intersection test, used by every
   audit, cannot see this class. Should-fix: re-decimate or flip that edge.

## 1. anatomies_v6 is the built tree (/out/hash)
- My own sha256 of every file (5,136, excluding BUILD_v6.json and _records added later) == the build manifest
  `v6_work/full/anatomies_v6.sha256` (sha256 977388c1...) == the staged copy's manifest.
- Against the bake output tree3: only `provenance.json` (223) and the licence file differ. provenance(v6) minus the one
  key `v6_checks` == tree3's on 223/223; tree3's minus the bake keys == the final compose (build2/final) on 223/223.
- OBJ and collision_full md5 == the md5 recorded by the bake that produced them (216 first-build bakes, 7 bake2).
- Curves == the final compose on 215; the 8 topcow_mr_021 anatomies differ ONLY in 19 radii widened by 0.11-0.79 mm
  on the mr_021 VA (the C4 pinch tube; positions identical).
- LICENSE_v6_donors.txt == the generated licence on 223; LICENSE_VMR_0248.txt exactly on the 7 VMR anatomies; no
  LICENSE_v5_donors.txt left. v6_checks gates / pass == the final checks records on 223.

## 2. Read-only trees and user notes
- git status clean on anatomies_v2..v5, topbrain_tools, carotid_tools (only `?? carotid_tools/v6`). No file under
  anatomies_v2 (4,461), v3 (4,461), v4 (4,907, incl. git-ignored collision_full.vtp), v5 (6,027), topbrain_tools (26),
  carotid_tools outside v6 (306) is newer than 2026-10-03 17:33 (the v5 commit); only the carotid_tools directory entry
  (v6 created inside it).
- user.md: mtime 2026-09-11, untouched. "user copy.md": modified 2026-10-07 00:33 (+91 lines); the added text is the
  user's own notes (slide-deck requests in the user's wording, pasted agent summaries) -- not an agent edit.

## 3. Mesh invariants, all 223 (/out/mesh)
- OBJ and collision_full: closed, edge- and vertex-manifold, consistently oriented, 1 component, genus 0, 0 crossing
  self-intersections, 0 degenerate faces: 223/223. Undecimated union on the 81-anatomy sample: the same, 81/81.
- Every centreline control point of the 18 curves inside the OBJ (|winding| > 0.5): 223/223; winding sign = v5's.
- VA lumen on the polyline in the OBJ (0.05 mm, exact distance, 3 mm caps out): min 0.6637 mm
  (w_050_left mr_004_L LVA), none below 0.65.
- RCCA prefix (arch -> min(host cut v6, v4) - 0.5 mm): points and radii bit-identical to v4 on 223/223.
- FINDING (not seen by the build's audit): fold-over pairs of edge-adjacent triangles (normals 179-180 deg apart,
  apexes on the same side, so the two faces overlap). MeshLab's self-intersection filter, used by every audit since v4,
  skips faces that share a vertex or an edge. v6 OBJ: 1 (case_m_022_left__topcow_mr_021_L, faces 15297/15302, 0.35 mm2
  folded onto 0.59 mm2 at (42.7, 19.8, 590.5)); v4 and v5 OBJs: 0. collision_full: v6 1 (w_017_left mr_014_L), v5 1, v4 1.
  Union: w_036_right mr_018_L 1 (v5's union of it also 1). (`mesh/folds.json`, `mesh/foldscan.json`)

## 4. Orientation gates, all 223 (/out/orient; own vf_orient.py)
- Independent of orient_gates.py: the native piece is read from its TopBrain / Zenodo source file (x-mirrored by the
  recorded mirror), the placed piece from Centrelines_comb, the rotation FITTED from geometry (arclength Kabsch + ICP)
  and only compared with the record (siphons / VA tops 0.01-1 deg; forks <= 1.8 deg). Where a short straight piece
  leaves the roll undetermined, the recorded R is used only after it is verified on the geometry (max residual 0.5 mm).
- Self-test on v5 reproduces the audit exactly: 141 tier-A anatomies (siphon 47, VA 115), ICA terminus > 40 mm
  posterior 108, forks > 45 deg off the map right 213 / left 140.
- v6: tier A on 1 anatomy (w_017_left mr_002_L RVA_top: va_lateral, D23) -> 0 after D23; v6-blocking flags after
  D8 / D23: 0 (raw: 11 anatomies, all va_target 20-30 or the D23 piece and its pair flags); va_target <= 20 / 20-30 /
  > 30: 435 / 10 / 1 (55.3 mm, D23); VA tops >= 0.615 mm apart; ICA terminus > 40 mm posterior: 3
  (w_003_right mr_001 45.9, w_018_right mr_007_L, w_027_right mr_016 45.4); forks > 45 deg off the map: right 5 (= the
  5 R5 forks) / left 7; fork tilt > 40: 0. My ICA-AP medians run 0.1-0.6 mm below the build's (method: nearest source
  point vs back-mapped point); the build's figures match its cited files.

## 5. Right route D22 / D26 / D27, all 223 (/out/d22; own vf_d22.py)
- Own ownership rule (tube distance to the RCCA vs every other curve), own references: v4's SHIPPED collision_full
  carried by the piece's own transforms, and the piece's own scanned surface (Zenodo STL / TopBrain vtp) placed by the
  v6 transform; v6 cells from the undecimated union written by the bake.
- Fork: min 0.9944 over 223 (v4 or own surface, handle windows out); siphon: min 0.9898 = k_011_left mr_012_L only,
  whose off-both cells are 17.79 mm2 at s 206.75-207.5 -- exactly the build's D27 repair window. |own - build| <= 0.0052
  (fork) / 0.0102 (siphon). v4-only minima 0.929 (w_033_left forks) / 0.9774 (k_011 siphon) as recorded.
- D22 as the user defined it (v6 cells on the reference) is CONFIRMED on 223/223 (k_011 by D27).

## 6. Real wall lost on the right route: the coverage direction (/out/d22, /out/d22b, /out/covdiag)
- Coverage = v4's RCCA-owned real-wall cells of the same gated stretch (v6 gate: v6 host cut + blend zone .. end,
  v6 handle windows +- 0.5 mm left out), carried by the piece's own transforms, within 0.3 mm of the v6 union.
  Reference = v4's SHIPPED mesh. Each uncovered cell is attributed: inside / outside the v6 union, on the piece's own
  scanned surface or not, covered by v5's union (v5 = same baker, v4's pose) or not.
- The method reproduces the build on the 4 pieces it reports (63.0 / 27.4 / 27.7 / 24.4 mm2 vs 65.0 / 26.7 / 26.9 / 24.1).
- 12 pieces < 0.99, all >= 1 % of the piece:
  * 8 = REAL SCANNED WALL CLIPPED, NEW IN v6 (v6 wall 1.6-3.0 mm inside the scan, 0 % inside the v6 union, the missing
    cells on the piece's own scan, v5's mesh covers ~all of it; radii and centrelines identical to v4 there):
      w_052_left mr_026 fork 4.9 % (63.0 mm2) | w_027_left mr_022 fork 2.2 % (27.4) | w_027_left mr_026 fork 2.2 % (27.7)
      k_011_left mr_012_L fork 2.1 % (25.5; user-named) | w_002_right mr_013_L fork 1.9 % (22.4; R3)
      w_051_right mr_008_L siphon 1.7 % (24.4) | w_012_left mr_003_L fork 1.5 % (24.7) | w_008_right mr_003 fork 1.3 % (24.2)
    The build reports only the first 3 forks + the w_051 siphon. The other 4 are exactly the D13 "v4 without its RECA
    tip" anatomies: the build measured coverage against v4 REBUILT without the tip, which lacks the same bulge, and
    recorded 1.0 -- but v4's and v5's shipped meshes show that wall. Only 4-10 mm2 of each lies next to a handle window,
    so it is not confined to a bake repair window (D27's condition for "reported, not gated").
  * 4 = NOT real-wall loss (the missing v4 cells lie INSIDE the v6 union and not on the piece's own scan: v4 showed a
    neighbour's wall there, D26's situation): m_030_right mr_003_L siphon, k_011_left mr_012_L siphon, w_040_left
    mr_021 fork, w_047_left mr_023 siphon.
- Below 1 % real-wall loss: w_012_left mr_001 fork 0.9 %, w_017_right mr_011 siphon 0.7 %, w_035_left mr_015_L siphon
  0.6 %, w_012_left mr_013_L fork 0.6 %, w_025_right mr_021_L fork 0.5 %.
- Cause not established (capsule size ruled out: v6 radii == v4 radii at the same piece points). Common pattern: the
  piece was re-posed (and on 5 of 8 x-mirrored) and, on 4, v4's RECA tip that ran along the ICA is gone.

## 7. Clearance and 1 mm joins, all 223 (/out/geom; own vf_geom.py)
- Graph of the centreline tree; site = lumen gap < 0.6 mm between samples > r_a + r_b + 6 mm apart along the tree,
  with at least one point moved from v5. 9 sites, all self-approaches of one curve AT the adjacency limit (path
  15.0-16.4 mm vs r_a + r_b + 6 ~ 15-16): (18) on the 4 mr_013_L anatomies 0.48-0.56 mm (host part vs the new donor
  subclavian at a tight bend); m_030_right mr_013_L RVA 0.592 mm (the U3 fold site, fixed from -0.067). Meshes there are
  genus 0. Sibling pairs inside one Zenodo fork are reported separately (inherited source geometry).
- 1 mm turns at 10 phases: 35 sites > 60 deg (max 80.9), every one a rigid copy of v5's same curve (6 mm window RMS
  <= 0.103 mm): RECA / LECA tip ends and the mr_026 VA tops. No new seam turns > 60.

## 8. Split, loader, SOFA
- Split (own run of v5_split.py on anatomies_v6 --baseline split_v4.json): EXIT 0, 45 compared, none lowered; strict
  k16 / k17 div6x6 raised as recorded (`split/split_v6_own.json`).
- Loader (own call of DualDeviceNavTopBrain on the 8 named + 2): 10/10 selects, 18 branches, the six named routes,
  finite, radii > 0, each anatomy's own OBJ.
- SOFA RCCA records (`_records/sofa/rcca_v6`): 223/223 present and passed, all on anatomies_v6's own OBJ
  (loader_mesh_is_this_obj), 3 seeds each, 0 exceptions, tip at most 0.193 mm outside.

## 9. CAROTID_V6.md numbers (/out/doc/doc_check.json)
- 186 numeric claims checked against the files they cite, most also recounted from the 223 provenance files or my own
  measurements: 185 consistent (4 ICA-AP values differ only by my method, 0.1-0.6 mm; 1 phrasing variant).
- Inaccurate: "Every flag is written in the anatomy's provenance" -- left_pose_map_bounded_dev (29) and orient:* (16)
  are only in results/v6_build.json (BUILD_v6.json carries the 29 as carotid.left_fork.map_bounded_deviation).
- Understated: Known limitations / the D22 section list 4 coverage pieces; real-wall loss >= 1 % is on 8 (section 6),
  incl. k_011_left mr_012_L's FORK (the doc only discusses its siphon).

## 10. Decisions D1-D27 against anatomies_v6 (/out/dec/dec_check.json)
- D20: x-mirror == contralateral on both routes (113 / 113, 101 / 101); fitted fork rotation proper, total det -1.
- D19 / D24: right forks R1-R3 on the per-patient map (210/215 within 1 deg by my fit; w_021_right x5 at 1.7 deg =
  fit noise), R4 deviations 20 / 20 / 4.9 deg, R5 rotation == v5's on all 5; RCCA prefix host (bit-identical to v4);
  SOFA RCCA on all 223 (section 8).
- D1 / D11: LECA on 223, no host LCCA tube, no ICA extension; 201 own + 22 substitutes (split not lowered).
- D2: VA tops >= 0.615 mm apart. D8: 435 / 10 / 1. D9: donor ids == v5's on all 429 sides (none dropped).
- D16: ia172_164 L tube-bridged 9.71 mm on all 16; ia172_071 L 10.01 mm on 2, the other 2 on rung 2 (D18).
- D17 / D25 (own geometry, 413 own-junction sides): the donor's distal subclavian ends as a cut end >= 30.2 mm past my
  ostium (the record's ostium sits ~1-5 mm further), lateral of the ostium by >= 8.96 mm, <= 78.5 deg from the host's
  distal subclavian; the host distal stretch is gone (3 left sides run within 1 mm of v5's host course over 6 % of it,
  because the donor's own course parallels it at 10 deg).
- D18 / D21: on the 33 non-rung-1 sides the subclavian is v5's and the VA is v5's below min(cut v6, cut v5); above it the
  same donor stretch (same donor R) is extended to the anatomical top's splice.
- D23: the only tier-A piece. D26 / D27: section 5. D25 one toolchain: section 11.

## 11. carotid_tools/v6 is what ran (/out/restore)
- restore_layout.sh -> 176 files, 0 sha256 mismatches; every restored file byte-identical to the live scratchpad file.
- Final manifest d65b7209: 72/72 (67 restored + the 5 topbrain_tools modules in the repo). All 7 manifests resolve to a
  current or frozen file except the 2 disclosed early versions (copy_v6.sh 2042e459, v6_build_collect.py a1e486d6,
  supplement manifest 13f6ee50); the final copy ran copy_v6 1.1 (preserved), so nothing in the tree depends on them.
- Versions in provenance == the preserved code: compose_v6 3.2.0, graft_v6_va 1.4.2, bake_v6 1.1, build_v6 1.2,
  checks_v6 1.3 (frozen) / 1.4 (frozen) / 1.5, stamp_prov 1.0, orient_gates 1.2. The baker (bake_v6 b3731df4,
  sdf_union_v4 c4ee3cbb) is the same in every manifest the 216 + 7 bakes ran under. No route13-style version gap.

## 12. Figures (/out/img/img_check.json)
- The 8 named: both axes found from their spines; equal aspect (mm/px equal within 0.1 %); every piece's house colour
  present along the anatomy's own centreline as provenance cuts it (front 0.925-1.0, side 0.955-1.0 for right-side
  pieces); the same pieces shifted 12 mm: 0-0.37; off-curve negative control 0. One partial: k_011_left mr_013_L left
  donor distal subclavian 0.65 (overdrawn by neighbours in projection).
