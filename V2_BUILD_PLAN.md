# v2 anatomy build: plan, guards, and the record of what was learned before

Rebuild of both anatomy sets with the signed-distance mesher and the
re-derived stage-C constants. Nothing in the shipped sets is touched; the
build goes to new folders and new figure folders.

| | shipped (v1) | v2 |
|---|---|---|
| TopBrain set A | `topbrain_data/anatomies/` (49) | `topbrain_data/anatomies_v2/` |
| Carotid set B | `carotid_data/anatomies/` (215) | `carotid_data/anatomies_v2/` |
| figures | `saved/figs/topbrain`, `saved/figs/carotid` | `saved/figs/topbrain_v2/`, `saved/figs/carotid_v2/` |

Same donors, same pairing plan (`pairing.json`, 237 pairs over 48 lowers),
same host, same fixes in the grafters. What changes is the mesher and the four
constants that only existed to compensate it. Keeping the pairing identical is
deliberate: it makes v1 → v2 a controlled comparison.

## Pool, so nothing is left out

- **TopBrain**: 25 MR masks × both ICAs = 50 vessels. `label_necks.py` screened
  all 50: 20 carry necks, **1 rejected** (`mr_006_lICA`, the 19.5 mm fragment
  the graft already refuses). So 49 vessels enter the graft, as before, and
  `mr_015` / `mr_003_L` are back in play — the neck that severed them is now
  repaired by the 1.0 mm floor instead of sealed by the mesher.
- **Carotid**: the manifest already covers the whole Zenodo release, 138
  carotids. 31 reach the seam unaided, 34 with ≤ 10 mm of extension, **73 are
  too short** (ICA < 48 mm past the bifurcation). Those 73 are a design
  decision, not a meshing one — they need either a longer extension budget or
  the seam moved — and v2 does not change it. Flagged, not done.

## What changes, and why each number

| constant | v1 | v2 | basis |
|---|---|---|---|
| mesher | binary spheres, Gaussian ×2, iso (min+max)/2, 0.6/0.6/0.9 mm | SDF, iso 0, **0.45 mm isotropic** | 0.64 → 0.06 mm median deficit; one component |
| `DISTAL_TRIM_MM` | 4.0 | **0** | SDF cap is a full-radius hemisphere; verified by enclosure |
| `ROUTE_MIN_R` (B donor route) | 1.60 | **1.0** | 1.0 mm tube meshes at 0.97; navigable radius 0.97 − 0.3 contact ≥ 0.35 catheter; restores stenosis to ~60 % |
| siphon floor (A, new) | none | **1.0** | label necks (0.3 mm) are segmentation error; ICA is never that narrow |
| `ECA_MESH_R_MM` | 1.6 | **1.0** | same enterability arithmetic; the ECA is a decoy |
| `FUSE_BAND_MM` | 0.35 | **0.35** (measured 0.2) | two 1.6 mm tubes merge below a 0.2 mm gap at 0.45 mm voxels; keep the margin |
| triangle budget | 3.7 k, global quadric | see SOFA timing below | |

## SOFA cost, measured (topcow_mr_001, ms per simulator step)

| triangles | 3.7 k | 16 k | 20 k | 38 k | 387 k |
|---|---|---|---|---|---|
| free advance | 134 | 207 | 202 | 219 | 2281 |
| pushing / twisting at the wall | 150 | 200 | 209 | 234 | 2243 |

The knee, measured: 6 k +5 %, 9 k +25 %, 12 k +55 %, then flat to 38 k.
Lumen deficit on `topcow_mr_001` (v1: 0.64 mm median, 0.35 mm minimum):

| triangles | 6 k | 9 k | 12 k | 20 k | vmtk weighted 12 k |
|---|---|---|---|---|---|
| median deficit (mm) | 0.33 | 0.23 | 0.19 | **0.12** | 0.25 |
| minimum lumen (mm) | 0.36 | 0.59 | 0.62 | **0.75** | 0.75 |

**Decision: 20 k, plain volume-preserving quadric.** Cost is the same as 12 k,
and it dominates every 12 k option; the VMTK route-weighted remesh is not
needed and the host/vmtk_env hop is dropped from the pipeline. The cheaper
alternative is 9 k (+25 % SOFA cost, 0.23 mm) — `recut_obj.py --tris 9000`
re-cuts a whole set from the kept `collision_full.vtp` in seconds.

Also measured and set aside: an analytic swept-tube mesh (exact radius, no
voxels). Untrimmed, the child tubes' walls inside parent lumens choke the route
to 0.33 mm; trimmed, the junction seams are 1.4 mm median, 7 mm at p95. It
needs seam stitching before it can be a collision mesh. Future work.

## Every earlier mistake, and the guard that carries into v2

Grafting bugs found in rounds one and two are all in the code the v2 build
runs; the point of this list is the *verification* each one demands, because
several were invisible to the check in place when they were made.

| what went wrong before | fixed by | v2 must verify |
|---|---|---|
| siphon placed by tangent only; roll free, siphons leaned over | frame match, tangent + superior | mean rise and descending count per set (`validate_anatomies.py`) |
| `anchor_trim` written but never called | called | anatomies rising < 20 mm = 0 |
| clearance measured from 130 mm instead of `host_cut` | fixed | clearance **bucketed by arclength**; the shared ostium is negative by construction and must not be read as a defect |
| 231/231 "negative clearance" near-miss | bucketing | same |
| mesh severing on stenosis donors | radius floor | floor lowered to 1.0 → **route check on the mesh**, not the centerline |
| `select_enclosed_points` blind to severed tubes | component-aware `navigable_route` | run it; enclosure alone is not a pass |
| `head_trim` returning 0 on budget exhaustion; ungated inlets | returns best index | seam-1 kink distribution ≤ v1's |
| ECA floor creating ICA↔ECA rings at +0.06 mm gap | gates on the fusing band | rings = 0 with the band re-measured for the v2 mesher |
| terminal cap erosion; targets outside the lumen | trim, now no trim + SDF cap | every target inside the mesh, including terminal ones |
| `fit` check reads the declared radius | **new: meshed lumen − contact vs catheter** | must pass on every anatomy |
| tortuosity 3.7 for all 25 — a metric too uniform to be real | walk cell ids | distributions, not just pass/fail, for every stat |
| `others` shadowing → 213 silent failures | renamed | build log: expected count built, zero exceptions |
| CRLF in container scripts | LF | scripts written with `\n` |
| left donors: mirrored in A, unmirrored in B | measured, not a defect | unchanged; noted as a known asymmetry between sets |
| figure overwrite | | figures go to `_v2` folders only |
| "checks pass" but SOFA never exercised the graft | rollouts with targets past 130 mm | SOFA sample on both sets, targets forced into the graft |
| **(found during v2)** `ROUTE_MIN_R` scoped to the donor section left set B's *siphons* unfloored; three `_L`-siphon anatomies baked at 0.37–0.57 mm lumen | `--siphon-min-r 1.0` in `graft_three.py`; 22 pairs regrafted | the meshed-lumen check at bake time is what caught it — keep it in the bake, not only in the audit |

## Order of work — all done 2026-09-03 except as noted

1. `label_necks.py` — done, 50 vessels.
2. Regraft A (`graft_siphon.py --distal-trim 0 --route-min-r 1.0`, right then `--mirror` left) → `anatomies_v2`.
3. Regraft B (`graft_three.py --route-min-r 1.0 --eca-mesh-r 1.0 --distal-trim 0 --fuse-band 0.35`, 4 shards) → `anatomies_v2`.
4. `bake_meshes_v2.py` on both, sharded in the container (~2 min per anatomy).
5. Budget decision from the SOFA knee: 20 k plain quadric; the VMTK remesh was not needed.
6. `check_anatomies.py`: A 49/49, B 223/223 pass; SOFA 6/6 and 9/9, targets past the seam.
7. Stats v1 vs v2: rise/kink/junction unchanged; lumen in `saved/mesher_probe/compare_v1_v2.txt` (navigable 45 % → 100 %, 33 % → 100 %).
8. Figures: 25 in `saved/figs/topbrain_v2`, 112 in `saved/figs/carotid_v2`; `BUILD_v2.json` in each set; results in `MESHING_PIPELINE_ANALYSIS.md` §7.

Not in v2: the hybrid real-surface union (change 8) and the single-environment
consolidation (change 9). Both are worth doing; neither is needed to ship a
set that the device can actually navigate.
