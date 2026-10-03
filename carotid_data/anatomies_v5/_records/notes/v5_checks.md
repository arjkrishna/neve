# v5_checks: post-bake checks on the 223 baked v5 anatomies

Agent prefix `v5_checks`. State `v5_va/state/v5_checks.md`. Results `v5_va/results/v5_checks.json` (written by
`checks/collect.py`). No git writes. Every check container mounted the repo READ-ONLY, so no check could change
`anatomies_v4` or `anatomies_v5`. Scripts and raw outputs are in `v5_va/checks/`.

## Answer

| item | check | result |
|---|---|---|
| 1 | strict mesh audit (v4's own `topo_audit.py` functions, pymeshlab) | **pass, 223 / 223** |
| 2a | right_route_detail identical to v4 (ratio 1.000) | **fails as a literal gate: 117 / 223 identical** (same 106 as the bake found) |
| 2b | VA wall detail on the undecimated union | donor stretch median **0.134** vs v4 host tube **0.000**; whole VA **0.150** vs v4 **0.018**; TopBrain piece unchanged |
| 3 | split, `v5_split.py --baseline split_v4.json` | **exit 0**, all 45 quantities equal to v4 |
| 4a | training loader over all 223 | **pass, 223 / 223** |
| 4b | SOFA load test that enters the RVA | **PASSED on 25 / 25** (0 exceptions or non-finite steps; tip at most 0.19 mm outside); **the wire entered the donor RVA in 9 / 23** (v4 control: 22 / 23) |
| 5 | anatomies_v4 untouched | **pass**: 0 of 4,684 tracked files differ from git HEAD |

## Item 1: strict mesh audit (all 223)

Code: `checks/chk_one.py`, which imports the v4 audit's `topo_audit.py` unchanged. That audit is independent of the
baker's `genus()` and `is_clean()`.

- **Topology, with pymeshlab, on the OBJ, collision_full and the undecimated union (669 meshes):** every mesh is one
  closed two-manifold genus-0 component, both raw and after welding coincident vertices. The pymeshlab OBJ loader
  sees the same. The numpy Euler check agrees: chi 2, 0 boundary edges, 0 non-manifold edges, 0 misoriented edges,
  0 duplicate faces.
- **Self-intersecting faces:** 0 on all 669 meshes, raw and welded.
- **Budgets:** OBJ 23,300-23,500 triangles (v4: 23,498-23,500); collision_full 59,998-60,000.
- **Routes (v4's check):** RCCA and LCCA are navigable on the OBJ and on collision_full in all 223. Every route point
  is inside, by winding number. OBJ lumen_min is 0.701-2.094 mm (RCCA) and 0.69-1.822 mm (LCCA), against the 0.65 mm
  wire-plus-contact need.
- **Centreline enclosure:** 0 of 421,304 centreline points are outside the OBJ or collision_full. End caps included.
- **The meshes come from this union:** collision_full vertices lie p99 ≤ 0.069 mm (max 0.26) from the undecimated
  union; OBJ vertices lie p99 ≤ 0.175 mm (max 0.41).
- **Orientation:** coherent, and inward-facing (signed volume < 0) on all 223. v4's shipped OBJs are inward-facing
  too, so this is the convention, not a defect. Collision and the winding-number tests do not depend on it.
- **Agreement with the bake's record** (clean flags and triangle counts): 223 / 223.

## Item 2: route detail and VA wall detail (undecimated union)

### 2a. right_route_detail against v4

Recomputed on `iso_v5/<name>.vtp` with v4's own `right_route_detail`:

- **It reproduces `mesh_v5.json` exactly on 223 / 223.** So iso_v5 is the union the bake accepted.
- **v4's side:** v4's exact union was rebuilt from `mesh_v4.json`. Its right_route_detail equals `mesh_v4.json` to
  0.0000 on all 223.
- **Identical to v4 (ratio 1.000): 117 / 223.** The other 106 are identical only beyond some distance of every VA
  centreline:

| identical beyond every VA centreline | anatomies |
|---|---|
| everywhere | 117 |
| 4 mm | 66 |
| 6 mm | 24 |
| 8 mm | 8 |
| not even beyond 8 mm | 8 |

- **The 8 that differ even beyond 8 mm:** m_030_right mr_013_L (carotid_changed), w_012_left mr_001 / mr_020 /
  mr_021_L, w_033_left mr_010 / mr_011_L / mr_024, w_048_left mr_008.
- **Ratios (frac > 0.3 mm, v5 / v4):** fork 0.819-2.386 (median 1.000), siphon 0.957-1.038 (median 1.000).
- **13 anatomies below 0.99, 1 below 0.95:** case_w_050_left__topcow_mr_004_L, fork 0.819. 28 anatomies are above
  1.01.
- **The RCCA standard holds as a set:** v5 fork median 0.197 against v4's 0.196 and v3's 0.196; siphon 0.322 against
  0.322.
- **Agreement with the bake:** the identical / not identical split agrees on 223 / 223. The "identical beyond" class
  agrees on 219. On the other 4, my comparison rounds more coarsely (4 decimals), so it reads one step (2 mm) closer.

This check **cannot pass as written**, for the reason the bake found: the donor's new course changes which vessel
owns the carotid bulb wall. Every non-VA centreline is byte-identical to v4 on all 223 (item 4a). The differences
sit within 4-8 mm of a VA centreline in 98 of the 106. This needs the user's decision (open issue 1).

### 2b. VA wall detail per VA

Ownership is right_route_detail's (24 nearest samples of all branches); fraction of wall area > 0.3 mm off the tube,
core = 2 mm off each end. My ownership code reproduces `va_metrics.measure_union_branch` exactly. On the 404
window-free donor stretches it also reproduces the bake's recorded donor_real exactly (difference 0.0000).

| stretch (429 donor VAs) | median | p10-p90 | min-max | reference |
|---|---|---|---|---|
| **donor real** (donor stretch minus seam zone minus tube windows) | **0.134** | 0.056-0.254 | 0.029-0.337 | v4 host tube **0.0003** (ref c); this set's v4 host tube **0.000** (max 0.020) |
| donor stretch as baked (windows and seam included) | 0.149 | 0.056-0.266 | 0.027-0.365 | |
| tube windows only (25 VAs) | 0.050 | | 0.000-0.212 | |
| **seam zone** (25 mm calibre ramp, kept separate) | 0.112 | 0.000-0.456 | 0-0.917 | ramp in v4: 0 |
| junction stub in the parent | 0.130 | | 0-0.706 | |
| **TopBrain piece** (cut → end) | **0.103** | 0.017-0.264 | 0-0.407 | TopBrain **0.12** alone, 0.125 in situ (ref c); this set in v4: 0.085 |
| **whole VA** (origin → end) | **0.150** | 0.061-0.270 | 0.034-0.341 | v4 VA **0.021** (ref c); this set in v4: **0.018** |

- **Every donor VA beats v4's host tube on the same side, and every whole VA beats v4's whole VA: 429 / 429.** The
  minimum donor_real, 0.029 (ia172_141 L), is still 100 times the host tube's 0.0003.
- **Radius-relative detail:** donor_real > 0.15 r is a median 0.187, above the RCCA fork's 0.130 (ref c).
- **Against the v3 RCCA route:** in absolute terms the donor (0.134) sits below the RCCA fork (0.197) and siphon
  (0.322). That gap is mostly vessel size: the radius-relative measure above puts the donor above the fork.
- **Real wall:** all 429 donor VAs are owned as real wall. None is a whole tube. 25 have one window: median 1.31 mm
  (1.15-9.14), 0.55-6.2% of the stretch.
- **v4-kept sides (17):** the centreline is byte-identical to v4.
- **TopBrain piece unchanged (geometry, `checks/tb_geom.py`):**
  - same surface file and mirror on all 429, and a proper rotation;
  - centreline shape after rigid alignment: RMS ≤ 0.036 mm;
  - radii: median |Δr| ≤ 0.0025 mm along every piece; beyond 10 mm of the splice, |Δr| ≤ 0.050 mm and RMS ≤ 0.029 mm.
    Local differences up to 0.38 mm sit only in the first ~10 mm past the splice, where the start trim and the
    splice point differ.
- **TopBrain piece unchanged (detail):** measured in situ over the same distal range in both unions, v5 / v4 is a
  median 1.005; 315 of 417 are within ±5%.
  - The outliers are where v4's own value was tiny (0.005-0.05) and v5 recovers the piece's standalone detail. One
    example: k_008_right mr_012_L reads 0.022 in v4, 0.100 in v5 and 0.098 alone.
  - Or they are short pieces where the splice spills into the range (m_022_left mr_021_L: 0.008 alone, 0.145 in situ).
  - The bake's in-situ / alone ratio for the TopBrain piece is a median 1.002 (p10-p90 0.973-1.048).
  - 12 of the 429 have no ratio:
    - 8 short (17 mm) mr_013_L pieces read 0 in both;
    - 4 were whole tubes in v4 (next point).
- **Changed for the better:** 4 TopBrain LVA pieces were whole tubes in v4, because v4's repair deflected the VA. In v5
  they carry their real wall (0.12-0.26): w_036_left mr_022, w_042_left mr_010_L, w_048_left mr_010_L,
  w_051_right mr_025_L.

### Extra: the VA lumen in the OBJ (not a v4 gate)

`checks/va_lumen.py`, using v4's route-check measures:

- 0 VA centreline points lie outside the OBJ, v5 or v4.
- 5 v5 VAs have 1-2 points whose wall distance is below 0.65 mm (v4: 0):
  - m_022_left mr_021_L: RVA 0.551 mm at s 118 and LVA 0.589;
  - w_025_right mr_021_L: RVA 0.632 and LVA 0.578;
  - w_027_right mr_022: RVA 0.646.
- All five sit where the donor's radius is at the 1.0 mm floor (Song 113 L/R, ia172_192 R).
- Median VA lumen_min: 0.84 mm in v5, 0.88 (R) and 1.18 (L) in v4.

## Item 3: split

`python /sp/v5_va/v5_split.py /repo/carotid_data/anatomies_v5 --baseline results/split_v4.json` →
`results/split_v5_baked.json`, log `checks/logs/split.log`.

- **EXIT 0:** "not lowered vs baseline (45 quantities compared)". Every quantity equals the baseline.
- **v3 definition:** roster 153 / 16; max k=16 183 / 16; k=17 181 / 17.
- **v4side and strict definitions:** roster 127 / 16; max k=16 174 / 16.
- **V: donors:** 25, in 25 TopBrain patients, one_to_one_nested True on all three definitions.

## Item 4a: training loader over all 223

`checks/loader_v5.py` is v4's `sim_loader.py`, pointed at anatomies_v5. Stated diffs: the RCCA is also checked
against v3, every non-VA centreline file against v4, and the VA files against the replaced flags. Output
`checks/loader_v5.json`.

- 18 branches on 223 / 223.
- The insertion equals the host's on all 223 and is inside the OBJ, 5.87-6.00 mm from the wall.
- 0 centreline points are outside the OBJ (vtkSelectEnclosedPoints, all points).
- RCCA byte-identical to v3: 223. Every non-VA centreline byte-identical to v4: 223.
- VA files replaced: R 211, L 218. These match the replaced flags and BUILD_v5.json's `replaced_after_bake`.
- RCCA targets beyond 133 mm: at least 74 per anatomy.
- The audit's only "problems" are point gaps and one duplicate point in host-numbered branches. They are identical in
  v4's audit and come from the unchanged host data.

## Item 4b: SOFA load test that enters the RVA

**Set** (`checks/select_sofa.py` → `checks/sofa_selection.json`). The 15 most changed are:
- every flagged anatomy of the bake: carotid_changed ×3, OBJ re-decimated ×3, a VA kept from v4 ×2;
- the largest tube window (w_008_right mr_020);
- the worst right-route ratio (w_050_left mr_004_L);
- the 5 largest RVA course changes against v4 (34-42 mm).

Plus 10 drawn at random (seed 20261002). Each anatomy ran on v5 and, as a control, the same schedule ran on its v4
version.

**Frame fix and test** (`checks/sofa_rva.py`, eve-training-fixed, training loader and training insertion):
- **Frame:** every geometric test uses `simulation.dof_positions`, which is the vessel frame of the centrelines and
  the OBJ. The old load test compared `fluoroscopy.tracking3d` (the tracking frame) with vessel-frame centrelines.
  The target is converted with `tracking3d_to_vessel_cs`: it lies on the RCCA (0.000 mm), at s ≥ 181 mm, in all 25.
- **Schedule:** a closed loop on the tip's position.
  - Fork 1 (RVA against RCCA): the J tip is turned toward the RVA before pushing.
  - Fork 2 is the VA ostium. In v5 it sits 3-10 mm past the subclavian (17) take-off. The tip is aligned there; if it
    runs on into (17), it is hooked back.
  - The schedule also unbuckles a buckled wire, unsticks a stalled one, and runs a back-and-forth exercise in the VA.
- **Steps:** 300 per episode. A second or third episode (reset, new seed) runs only if the tip did not get 20 mm into
  the donor stretch, because SOFA's route choice is stochastic. In all, v5 ran 63 episodes (18,900 steps) and the v4
  control 51 (15,300).
- **Every step:** all DOFs finite, and the tip and every inserted device point tested against the OBJ.
- **Pass rule:** the PASSED line is read, not the exit code.
- **Drafts:** v1-v4 of the schedule are in `checks/sofa/run`, `run2`, `run3` and `dev`, superseded and kept. The
  final run is `checks/sofa/run4`.

**Results:**

| | v5 | v4 control |
|---|---|---|
| SOFA RVA PASSED (18 branches, the loader uses this OBJ, insertion inside, 0 exceptions, 0 non-finite steps, tip never > 0.5 mm outside) | **25 / 25** | 25 / 25 |
| tip outside the wall | 34 of 18,900 steps, all on w_007_left mr_013_L at RVA s 60-70 (donor stretch), max **0.19 mm** (inside SOFA's 0.3 mm contact distance) | 0 |
| device points > 0.3 mm outside | 0.37% of point checks; max 3.5 mm | 0.33%; max 3.5 mm |
| ms / step (median) | 331 | 341 |
| wire entered the new RVA ≥ 20 mm past its ostium | **11 / 25** (9 of the 23 donor RVAs, plus the 2 v4-kept RVAs) | **24 / 25** (22 of the same 23) |
| reached ≥ 20 mm into the real donor wall | 10 / 25; up to 136.6 mm traversed | n.a. (host tube) |

- **The device-point excursions are the same in both.** They are buckled loops pressed into the wall at the arch, the
  (11) bridge and the subclavian, where both versions share host geometry. They come from SOFA's contact, not the v5
  mesh.
- **Entering the donor VA is much harder than entering v4's host VA under this schedule:** 9/23 against 22/23,
  Fisher p = 0.0001.
  - **Where the wire stops:** in every v5 failure the tip got to within 7 mm of the ostium and stalled there (max tip
    position −7 to +18 mm from the ostium) after 0-8 hook attempts.
  - **The ostia differ** (`checks/entry_geom.json`, `checks/ostium_lumen.json`):

    | | v4 host VA | v5 donor VA |
    |---|---|---|
    | where it leaves | at the (17) take-off | 3-10 mm past it (median 5, all 211 donor RVAs), behind the junction stub inside the subclavian |
    | angle to the subclavian continuation | a uniform 88° T-junction | 35-127° |
    | wall clearance at the ostium | 2.2-2.3 mm | 0.95-1.69 mm in the 23 tested |

    Among the 23 v5 ostia tested, the clearance was a median 1.22 mm where the wire failed (14) and 1.51 mm where it
    entered (9).
  - **What this is:** a real-calibre ostium that a scripted J-wire catches less often. It is not a mesh defect: the
    lumen is ≥ 0.95 mm against the 0.65 mm needed, and the wire passed it into 9 donor VAs.
  - **Training is unaffected:** it targets the RCCA, which is unchanged.

## Item 5: anatomies_v4 untouched

Read-only git commands, at 20:36 and again at 23:05, after all checks:

- `git status --porcelain` and `git diff --stat HEAD` on carotid_data/anatomies_v4: empty.
- `git hash-object` of all 4,684 tracked files against the HEAD blobs (29c8c5d): 0 mismatches.
- The 223 untracked files are the gitignored `collision_full.vtp`. Newest mtime in the folder: 2026-09-27 01:46.
  0 files newer than 2026-10-01.

Files: `checks/item5_v4_untouched.json` and `checks/item5_v4_untouched_end.json`.

## Open issues for the user

1. **Right-route gate (decision, as the bake reported).** "Identical to v4" holds on 117 of 223 anatomies. It holds
   beyond 4, 6 or 8 mm of every VA on 98 more. 8 differ even beyond 8 mm, and 13 lose more than 1%; the worst is
   w_050_left mr_004_L, fork 0.819. The options are:
   - (a) gate on "identical beyond 8 mm" plus the reported ratios;
   - (b) stop the VAs from competing for the carotid wall in the union, which changes the v4 union standard;
   - (c) re-assign or re-place the donors that lose the most.
2. **The donor RVA ostium is harder to enter** (item 4b): 9 of 23 against v4's 22 of 23, with the same schedule. The
   cause is the junction stub (3-10 mm, still ungated: verify_graft_reg item 7) and the donor's real calibre at the
   ostium. It does not affect RCCA training. If the RVA is ever a training target, grafting the ostium at the (17)
   take-off, or gating the stub, would make it more like v4's.
3. **Narrow VA points (not a v4 gate).** 5 donor VAs have 1-2 centreline points with less than 0.65 mm of wall
   clearance (0.55-0.65 mm): Song 113 L/R in m_022_left mr_021_L and w_025_right mr_021_L, and ia172_192 R in
   w_027_right mr_022. They sit where the donor radius is at the 1.0 mm floor.
4. **Recorded, not a problem:**
   - 4 TopBrain LVA pieces that v4 baked as tubes now carry real wall in v5.
   - The v5 TopBrain pieces start up to 7 mm shorter than v4's (start trim at the splice); their radii differ only in
     the first 10 mm.
5. **Housekeeping.** The superseded SOFA drafts in `checks/sofa/{out,v4ctl,run,run2,run3,dev}` can be deleted. My
   removal of two of them was denied by the built-in rm safety check; it was not retried and is left to you. The
   bake's scratch folders are listed in notes/v5_bake_full.md.

## Files

- **Scripts** (`v5_va/checks/`):
  - `chk_one.py` (items 1 and 2, per anatomy), `worker.sh`, `launch.sh`, `status.sh`, `names.txt`, `dr.sh` (docker,
    repo read-only);
  - `loader_v5.py`, `select_sofa.py`, `sofa_rva.py`, `sofa_worker.sh`;
  - `va_lumen.py`, `tb_geom.py`, `entry_geom.py`, `item5.sh`, `collect.py`.
- **Outputs:**
  - `checks/out/<name>.json` (223);
  - `checks/loader_v5.json`, `checks/va_lumen.json`, `checks/tb_geom.json`, `checks/entry_geom.json`,
    `checks/sofa_selection.json`, `checks/bake_tb_a2.json`;
  - `checks/sofa/run4/{v5,v4}/<name>.json` and `run4/logs/`;
  - `results/split_v5_baked.json`, `results/v5_checks.json`.
- **Superseded drafts, kept** (a removal of my own test outputs was denied by the safety check; nothing was deleted):
  - `checks/sofa/out`, `v4ctl`, `run`, `run2`, `run3`, `dev`. These are SOFA schedule drafts.
  - They can be deleted once this stage is accepted.
