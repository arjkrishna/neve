# verify_v5_mesh: adversarial check of the v5 bake, mesh and simulation lens

Agent prefix `verify_v5_mesh`. State `v5_va/state/verify_v5_mesh.md`. Work dir `v5_va/vmesh/`. No git, nothing
written outside the scratchpad; `/repo` mounted read-only in every container.

## Answer so far

**Final (02:20): holds in the mesh and simulation lens.** Every key number in my lens reproduced with my own code.
The checker's one exception, right-route detail on 106 anatomies, belongs to the wall lens and the checker
already reports it.

- **Sweep of all 223** (`vmesh/vm_all.py` -> `vmesh/all/*.json`, digest `vmesh/all_summary.json`):
  - OBJ and 60k mesh: 223/223 one component, genus 0, closed and two-manifold.
  - OBJ: own SI 0 and pymeshlab 0. The 60k mesh: pymeshlab 0 and own SI 0, except one fold pair (see below).
  - OBJ 23,300-23,500 tris and full 59,998-60,000.
  - 0 of 421,304 centreline points outside on either mesh, end caps included. That is the checker's exact count.
  - RCCA and LCCA navigable 223/223 on both meshes.
  - OBJ route minimum is RCCA 0.701 mm (w_048_left mr_010_L) and LCCA 0.690 mm (k_004_left mr_010), the checker's 0.70 / 0.69.
  - OBJ VA clearance is under 0.65 mm in 3 anatomies: m_022_left mr_021_L 0.551, w_025_right mr_021_L 0.578 and
    w_027_right mr_022 0.646. These are exactly the checker's open item 3. VA clearance is not a v4 gate, and the 60k mesh is >= 0.818 everywhere.
- **Union topology, all 223** (`vmesh/union_topo.json`): raw 223/223 clean, inward and chi 2.
  - Exact welding merges one vertex pair, in w_007_left mr_013_L. It is a zero-length EDGE, not a pinch: a vertex on a grid node, at LCCA s 243 mm and 70 mm from any VA.
  - Welding plus dropping the 2 resulting degenerate faces gives a clean genus-0 mesh (`vmesh/weld_dig.json`). pymeshlab's merge does the same.
  - The checker's own numpy row for this union shows the raw-weld artefact (nonmanifold 2, degenerate 2). Their "clean after welding" therefore depends on pymeshlab dropping null faces. This is benign.
- **v4 baseline** (`vmesh/v4_si.json`): v4 OBJ and 60k meshes have own SI 0 on all 223, with near folds (dot < -0.99) 10 / 18 against v5's 8 / 23.
  - The single v5 fold pair past -0.999999 is in the m_022_left mr_021_L 60k mesh. It is a 0.025 mm sliver folded onto its neighbour at a dihedral of about 0.02 deg, 4.7 mm from LVA s 234 mm (TopBrain region).
  - The exact test finds no crossing, pymeshlab gives 0, and the OBJ that SOFA uses has no fold. v4 has no fold this sharp. This is a note, not a gate failure (`si_dig_full.json`).

- **40-anatomy own audit** (`vmesh/out/*.json`, digest `vmesh/summary.json`): OBJ and 60k mesh 40/40 one face/vertex
  component, chi 2 (genus 0), 0 boundary / non-manifold edges, 0 pinch vertices, 0 orientation conflicts, welding
  (exact and 1e-6 mm) merges nothing; own SI 0 and pymeshlab SI 0 on both; signed volume negative (inward) 40/40.
  OBJ 23,300-23,500 tris, full 60,000; tri counts = mesh_v5.json records 40/40.
  0 of 75,303 centreline points outside (OBJ and full, end caps included; no ambiguous winding);
  RCCA/LCCA navigable 40/40 on both; route lumen = bake record (max |diff| 0.001 mm).
  60k mesh to union p99 <= 0.066 mm; OBJ to union p99 <= 0.175 mm.
- **Union (undecimated)**: one component, genus 0, closed, manifold 40/40. My SI flag fired on 2 unions
  (w_016_left mr_011_L, k_004_left mr_018): all 5 hits are my fold heuristic (edge-sharing pair with normals
  dot < -0.999999), i.e. a ~0.05 deg crease, not a crossing: exact rational test finds no crossing, pymeshlab 0 on the
  pair and on the mesh. They sit on the carotid real wall (4-5 mm from the LCCA / RCCA centreline, >= 10 mm from any
  VA) and do not reach the OBJ / 60k mesh (0 folds there). Not a refutation (`vmesh/si_dig.json`, `fold_loc.json`).
- **Narrow VA**: w_025_right mr_021_L OBJ VA clearance LVA 0.578 / RVA 0.632 mm (< 0.65); the 60k mesh >= 0.854.
  This is the checker's open item 3 (Song 113 at the 1.0 mm radius floor); VAs are not a v4 route gate.
- **Records over all 223** (`vmesh/flags.json`): 429 donor VAs real wall, 0 whole-section tubes; 25 donors with one
  window 1.15-9.14 mm (median 1.31, max 6.17 % of the stretch); 17 v4-kept sides (15 at assignment, 2 at bake) all
  flagged with a reason, centreline + transform exactly v4's; 3 rung-C anatomies flagged carotid_changed;
  4 anatomies with full = 59,998 (v4 has 3 such).
- **Keep-v4 justification** (`vmesh/tubes.json`, own genus on the bake's tube field): both bake keep-v4 sides have a
  handle in the TUBE-ONLY union with the donor VA (genus 1) and none with the v4 VA (genus 0), so the whole-section
  tube rung cannot work: w_050_left mr_026_L RVA (donor runs into the subclavian (17) tube at s 27 mm; genus 0 if cut
  at 30 mm, 1 at 45 mm); w_033_left mr_007 LVA (genus 1 already at 15 mm: ostium handle).
- **SOFA**: original `load_test_v5.py` on the 5: "LOAD TEST PASSED", 5 x 300 steps, tips finite (the loader drew only
  4 distinct anatomies: episodes 2 and 3 are the same one); the process then segfaults at interpreter exit (exit 139),
  the same teardown crash as v4's `v4_logs/final_checks/sim_step.log`. Own `vm_sofa.py` 400 steps each: "VM LOAD
  TEST PASSED" 5/5: no exception, 18 branches, the loader's mesh = this OBJ, insertion inside (5.92-5.97 mm from the
  wall), 0 non-finite steps, tip never outside the OBJ. Non-tip DOFs > 0.3 mm outside: w_048 162 checks (max 2.04 mm),
  w_050 82 (1.53), m_022 3 (0.35), others 0.
  - v4 control: same script and seed on the v4 versions (`vmesh/sofa_v4/`). All 3 PASSED, with non-tip DOFs > 0.3 mm outside v4 150 / 178 / 128 (max 1.63-1.79 mm) against v5 3 / 162 / 82. The DOFs that poke outside are v4's own behaviour.
- **Files**:
  - `anatomies_v5` has 223 folders, each with obj, vtp, mesh_v5.json and provenance.json; provenance keeps its licences.
  - No v5 OBJ is byte-equal to its v4 OBJ.
  - Nothing in anatomies_v5 is newer than 2026-10-02 20:28, so the audit read the final bake.

## Method

- **Own code** (`vmesh/vm_one.py`, numpy/scipy only; no bake, v4-audit or check-stage function):
  - OBJ parser, VTP arrays via vtk's reader;
  - topology raw, after exact-coordinate welding and after 1e-6 mm tolerance welding: unreferenced, degenerate,
    duplicate and zero-area faces; edge face-counts (boundary, non-manifold); directed-edge orientation conflicts;
    face components (edge adjacency) and vertex components; vertex-link fans (pinch vertices); Euler characteristic
    and genus per component; signed volume;
  - self-intersections: kd-tree + AABB broad phase, exact orientation-predicate segment/triangle tests for disjoint
    pairs, opposite-edge tests for vertex-sharing pairs, fold-back test for edge-sharing pairs; pymeshlab's count as a
    second opinion only;
  - generalized winding number for every centreline point (end caps included);
  - exact point-to-triangle distance (exact candidate set) for route clearance, every branch's clearance, and
    OBJ/60k vertices to the undecimated union.
- **Self-tests** (`vmesh/vm_selftest.py`, `vm_selftest2.py`): sphere/cube genus 0 clean, torus genus 1, pinch found;
  own SI = pymeshlab on crossing spheres (244 = 244) and a poked sphere (12 = 12); winding = vtk enclosed points
  (0/1200 mismatches); distance = vtkImplicitPolyDataDistance (max diff 2e-15); closest point vs a dense grid
  (0 violations of 2000).
- **Sample**: `vmesh/hard10.txt` (the 9.14 mm and 4.43 mm windows; the 3 rung-C carotid_changed; the 2 bake keep-v4;
  the 3 obj_fix) + `vmesh/rand30.txt` (30 random of the other 213, seed 20261003).
- **Records over all 223** (`vmesh/vm_flags.py` -> `vmesh/flags.json`).
- **SOFA** (`vmesh/sofa_run.sh`): the original `load_test_v5.py` on 5 anatomies, and an own load test
  `vmesh/vm_sofa.py` (training loader, vessel-frame checks, seeded forward-biased random schedule, 400 steps).
- **Keep-v4 justification** (`vmesh/vm_tubes.py`): tube-only genus with the rejected donor VA vs the v4 VA.
