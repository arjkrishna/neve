# verify_s2_gate2: adversarial check of the second integration pass (D22 to D25), GO or NO-GO for stage 3

Agent verify_s2_gate2, 2026-10-07. State file: `state/verify_s2_gate2.md`.

My own code is in `scratchpad/v6/verify2/`. It uses numpy and scipy only. `vlib.py` and `audit14.py` are copied from the gate-1 verifier. The scripts are:
- dry223.py, agg223.py, turnfix.py
- d22own.py, strict223.py, d15bake.py
- midline.py, distal.py, vatops.py
- headmap.py, forkyaw.py, trace_imports.py

The runner is `dr_v2.sh`. It mounts /repo, /bld, /rp and /rp2 read-only.

Outputs are in `carotid_data/v6_work/verify_s2_gate2/` (called /out below). I ran containers named vg2_* only, and all of them have exited. I made no git writes, touched no repo-tracked tree and edited no other agent's file.

## Answer so far (verdict: GO, with confirmations and non-blocking fixes)

Most of the claims hold up when measured with my own code. A full build can be expected to pass every invariant with the flags as decided. Three conditions apply:

1. The user (or the orchestrator, reversibly) confirms the D22 reference-validity reading.
2. Stage 3 reports both D22 values (value and value_strict) for every anatomy.
3. Stage 3 bakes the 223 anatomies successfully. Only 14 of 223 have been baked so far, and every one of those passed.

### The gate-1 refutations: each is now fixed or covered by D22 to D25
1. **Right-route relief gate → D22.** I measured D22 my own way on the second re-pilot (`/out/d22own`):
   - Method: I took the v6 collision_full vertices owned by the RCCA beyond the seam zone and measured them against v4's collision_full. v4 was carried by a transform I fitted myself: ICP initialised from the records, plus an independent Procrustes fit.
   - The fitted transform equals the recorded one. The centreline RMS is 0 to 0.006 mm, except w_035's fork at 0.14 mm.
   - Chirality: det is -1 on all 10 mirrored forks and +1 on the 4 native ones.
   - Fork, strict (nothing left out but handle windows): 14 of 14 are at 0.99 or above. The minimum is 0.9913 (w_051).
   - Siphon, strict: 12 of 14 are at 0.9986 or above.
   - Siphon failures: k_011_left mr_012_L reads 0.9761 and w_047_left mr_023 reads 0.9868. Both read 1.0 once the v4-occupied cells are left out. Every deviating cell maps to a v4 position where v4's RVA (or another branch) occupied the siphon wall.
   - On w_047, the left-out cells lie on the piece's own TopBrain surface carried by the v6 transform (max 0.036 mm).
   - On k_011, the cells form a bulge of the real surface about 3 mm off the tube. They are 15 mm or more from any v6 VA.
   - My numbers match the integrator's (0.9776 and 0.9890).
2. **Tier A on w_017_left mr_002_L → D23.** The DRY's orientation check finds tier A on that anatomy only, and it is accepted. After D23 there are 0 tier-A anatomies and 0 blocking flags. The acceptance is hard-wired to that one name with rung 3.
3. **ia172_069 D17 direction.** The direction gate now exists as a side gate (graft_v6_va 1.3 U1). Both sides are demoted to the v5 junction.
   - My own direction check finds 413 of 413 donor-junction sides lateral. The smallest lateral offset is 9.97 mm. The result matches the records.
   - The re-pilot figure of m_006_left mr_025_L no longer shows the anterior loop.
4. **1 mm joins at every phase.** I resampled every changed curve at 1 mm from 20 phases (`/out/turnfix.json`).
   - 34 sites exceed 60°. All 34 are rigid copies of v5's own geometry on the same curve (RMS 0.0003 to 0.076 mm over 4 to 6 mm windows), so they are not new seams.
   - The sites are TopBrain ECA-tip distal ends (65 to 82°), the mr_026 VA tops (62 to 65°) and w_018b mr_002_L RVA (73.4° = v5).
   - That leaves 0 new-seam exceedances. song_095's (19) is spread to 57.8°.
5. **Records and a single toolchain.**
   - donor_id in the record equals xform.vaC_* on every donor side, including rung 2 and 3.
   - compose_v6 3.1.0 and graft_v6_va 1.3 ran on 223 of 223.
   - Manifest md5 bedf172b is recorded on 223 of 223, and all 46 of its files match today.

### Re-checks with no refutation
- **Read-only trees.** git status is clean on anatomies_v2 to v5, topbrain_tools, carotid_tools, eve and eve_bench. No file in those trees is newer than 10-03 17:35. None of the repo-tracked v6 targets exists yet.
- **DRY of 223** (`/out/agg223.json`):
  - Right fork rungs: R1 124, R2 91, R3 3, R4 3, R5 2.
  - Contralateral equals mirrored: 113 on the right, 101 on the left.
  - The map-posed right forks (R1 to R3, 218) use exactly the per-patient map rotation.
  - Every fork outside the fork gates (heading over 45 or tilt over 40) is flagged: 2 R5 forks and 7 left forks.
  - Left: 201 kept and 22 substituted. 0 host LCCA, LECA present on 223, 0 ICA extensions.
  - RCCA prefix (arch to v5 seam 1) is bit-identical to v5 on 223 of 223. The host-cut shift region lies on the host RCCA polyline (0.000 mm).
  - Shift: 92 nonzero, maximum 15.0 mm, none negative.
  - 0 new clearances under 0.6 mm. The minimum is 0.614 mm (k_005_right mr_013_L, 17|RCCA).
  - VA tops are at least 0.615 mm apart on 223, so there is no basilar fusion.
  - D10 (my own geometry): 421 of 429, matching the records. The 8 failures are rung 2 or 3 stretches inherited from v5.
  - D17: 413 sides, every cut end at least 10 mm from other curves, and the host distal stretch dropped. The 33 non-D17 sides are identical to v5.
  - Donors are never dropped or changed. 25 donors map one-to-one onto 25 patients. 3 anatomies have AP over 40 mm.
- **Split** (my own rerun of v5_split.py on /bld/dry223b/final): exit 0, 45 quantities compared, none lowered, V donors one-to-one 25 of 25 (`/out/split`).
- **Second re-pilot, 14 anatomies** (`/out/audit14`):
  - OBJ and collision_full: genus 0, closed, consistently oriented, 1 component, 0 self-intersections.
  - Every centreline point is inside. Winding sign matches v5.
  - VA polyline lumen minimum 0.664 mm (w_050). 0 new clearances.
  - The RCCA prefix is bit-identical to v5 and v4. The OBJ near the prefix matches v5 except 3 vertices on w_025 (0.45 mm), where D17's (17) passes 0.64 mm away.
- **D15:** the inputs are unchanged since gate 1. On the bakes, the wall outside the zones (±4 mm station tolerance, the 25 mm VA seam excluded) is within 0.3 mm of v5's bake on at least 0.999 of the area on 25 of 26 sides. The exception is w_027_right mr_022 R at 0.989, where a v5 RCCA contact was released (`/out/d15bake_seam26_tol4.json`).
- **SOFA RCCA re-pilot:** 14 of 14 passed. Note that "passed" means soundness only. Reach (seam 1 / bifurcation / siphon / target) is 42 / 41 / 34 / 22 against v5's 42 / 41 / 33 / 19. Per anatomy, reach fell on 5 and rose on 4.
- **Figures:** I looked at 8 of 14. Nothing new stands out.
  - On w_018b mr_002 the RVA runs medially; this is the approved route13 move, and its D10 deviation is -7.78 (passes).
  - On w_050 mr_004_L the R side was already medial in v5 (rung 2).
- **The re-pilot tree equals the DRY** except w_025 RVA. There the bake's c4 widened the radii (+0.79 mm on 19 points); the positions are identical. This is the documented mr_021_L c4 step.

### Must decide (not a rebuild)
- **The D22 reference-validity rule (left-out cells where v4's wall was occupied by another branch) carries real weight.** My pre-bake estimate on 223 (`/out/strict223`) is the share of gated v4 real-surface area that was occupied in v4 and released in v6:
  - at least 0.4%: 37 pieces
  - at least 0.8%: 13 pieces
  - at least 1%: 9 pieces

  On the re-pilot the estimate understates the strict loss by 1.7 to 3.5 times. So under a literal reading of D22, roughly 10 to 40 anatomies would fail, with no possible remedy.

  The rule matches D22's intent ("same surface, moved") and follows the D13 precedent. It needs the user's confirmation. Stage 3 must report value_strict and the count.

### Should fix (not blocking)
1. **Manifest.** Two imported modules are not listed: carotid_tools/extend_ica.py (git-clean, used only by the unused ICA-extension rungs) and v5_va/sources.py (imported by graft_v5_va, mtime 10-01). Add them, or record their md5s, for stage 3.
2. **Left fork pose labels.** Of the 215 left forks labelled "zenodo_map", 186 sit exactly on the map. The other 29 are bounded deviations of 5 to 43° (within the fork gates) and are not flagged, while the right-side equivalent (R4) is flagged. Report them as 186 + 29 and flag them as D11 implies.
3. **geom_v6 1.1 turn exemption.** A turn over 60° is exempt when it is at most v5's whole-curve maximum + 1°, regardless of where it is. Make the exemption location-specific. On this DRY it changes nothing: all 34 sites are rigid copies.
4. **Load test.** load_test_v5 picks anatomies at random (10 distinct of 14). Stage 3 should load every anatomy deterministically. The loader part and the SOFA runs do cover all 14 today.

### Flags for the user (already recorded)
- k_005_right ×4: the host cut moves 15 mm and trims the donor CCA. This is within D19's limit.
- 8 right forks flagged: R3 ×3, R4 ×3, R5 ×2. The R5 forks are reversed (yaw about 180°).
- 8 left poses not taken from the map.
- 3 anatomies with AP over 40 mm.
- D23.
- 8 inherited D10 failures.
- 17 D21 sides with the host VA.

Status: complete
