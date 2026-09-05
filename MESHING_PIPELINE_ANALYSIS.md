# Label map → SOFA collision mesh: what the pipeline does, what it costs, what to change

Analysis of the mesh-making path used for both anatomy sets (TopBrain siphons,
Zenodo carotid bifurcation database) as of 2026-09-02. Every number below was
measured on this repo's code and data; the scripts and raw outputs are in
`saved/mesher_probe/` (`probe.py` runs in the container, `remesh_vmtk.py` in
`vmtk_env`, `report.txt`, `label_qa.txt`, `vmtk_options.txt`).

---

## 1. What is actually built — and one correction to the premise

VMTK does **not** make the meshes. It is used once, to extract a centerline
with maximum-inscribed-sphere radii (MISR) from a surface. The surface it is
given is thrown away afterwards. The collision mesh SOFA loads is
**re-synthesised from the centerline + radii** by stEVE's voxel mesher
(`eve/intervention/vesseltree/util/meshing.py`) — a union of spheres, blurred,
iso-surfaced, decimated. So the SOFA mesh is an idealised circular tube for
every vessel, and the real lumen cross-section (elliptical, plaque, wall
texture) never reaches the simulator for any of the grafted anatomies.

| stage | tool | what it does | parameters that matter |
|---|---|---|---|
| A. label → surface | `topbrain_tools/mask_to_surface.py` | keep label (4 R-ICA / 6 L-ICA), largest component, `skimage.marching_cubes` at 0.5 in world mm via the NIfTI affine, `vtkWindowedSincPolyDataFilter`, skeleton double-BFS for the two seeds | mask voxels **0.297 × 0.297 × 0.6 mm**; sinc 20 iterations, pass band 0.1 |
| B. surface → centerline | `topbrain_tools/vmtk_centerline.py` | `vmtkCenterlines`, `pointlist` seeds, `AppendEndPoints=1`, walk the longest cell, read `MaximumInscribedSphereRadius` | no resampling, no smoothing; ends trimmed 1–3 mm downstream |
| A'/B'. Zenodo database | shipped as-is | `*_lumen.stl` + `*_lumen_centerlines.vtp` (VMTK root-to-tip paths with MISR) already in the release; `analyze_bifurcations.split_tree` recovers CCA/ICA/ECA | the STL is only probed (`graft_three.py` comment), never meshed |
| C. graft | `graft_siphon.py` / `graft_three.py` | frame-matched placement, radius ramps, floors (`ROUTE_MIN_R 1.60`, `ECA_MESH_R_MM 1.6`), `DISTAL_TRIM_MM 4`, `FUSE_BAND_MM 0.35` | every one of these constants exists to pre-compensate stage D |
| D. centerline → mesh | `bake_meshes.py` → `generate_mesh()` | voxel cube **[0.6, 0.6, 0.9] mm**, 5 padding layers, mark every centerline point as a binary sphere (`dist < r`), `gaussian_smooth(1)` **twice**, `marching_cubes(level=None)`, `decimate(0.99)` | σ = √2 voxels = 0.85 mm x/y, **1.27 mm z**; iso level = (min+max)/2; quadric decimation, no volume preservation |
| E. into SOFA | `sofabeamadapter.py` | `MeshObjLoader` → `TriangleCollisionModel` + `LineCollisionModel`; `LocalMinDistance(contactDistance=0.3, alarmDistance=0.5)` | the device is a zero-proximity line; the wall is effectively **0.3 mm thicker** than the mesh |

The shipped host patient is different again: its `vessel_architecture_collision.obj`
is the real VMR surface, `decimate(0.9)`, written unwelded (3,583 triangles,
10,750 points, every edge open). So the one test anatomy is a real segmented
surface while every training anatomy is a smooth tube.

---

## 2. Where the fidelity goes — measured

### 2.1 Stages A and B are faithful

For all 25 patients, both ICAs, the MISR read off the smoothed surface was
compared with the raw label's own Euclidean distance transform at the same
centerline points (`label_qa.txt`, last column):

| | median (MISR − EDT) | 5th percentile | MISR below EDT |
|---|---|---|---|
| 50 vessels | **+0.00 … +0.03 mm** | −0.07 … −0.11 mm | 31–50 % |

The windowed-sinc smoothing does not shrink the vessel and MISR tracks the
label to within a tenth of a millimetre. Nothing in A/B needs fixing for
radius fidelity. (MISR is still an *inscribed* radius — an elliptical lumen
reads as its semi-minor axis. `vmtkcenterlinesections` would give the area
and shape index; not needed until the mesh stops being a circular tube.)

### 2.2 The label maps are clean, but have necks

Every R-ICA/L-ICA label is a single connected component with zero holes.
What they do have is **necks**: skeleton voxels whose inscribed radius is
under 0.6 mm, i.e. the segmentation is one or two voxels thin there.

| case | necks | 5th-pct radius | note |
|---|---|---|---|
| mr_022 L | 20 | **0.32 mm** | thin over ≥5 % of the vessel; shipped |
| mr_006 L | 21 | 0.35 | the 19.5 mm fragment — already rejected |
| mr_003 L | 8 | 0.73 | **known SEVERED failure** |
| mr_015 R | 4 | 0.84 | **known PINCHED failure** |
| mr_008 L | 9 | 0.84 | shipped |
| mr_013 R, mr_014 R | 5, 2 | 1.07 | the two "borderline, tolerated" cases |
| mr_001 R, 007 R/L, 010 L, 011 R, 012 L, 018 L, 021 L, 023 L | 2–6 | ≥1.2 | isolated pinches |

Both known failures and both borderline cases are in the neck list. A label
neck (`EDT < 0.6 mm` along the skeleton) predicts the mesh severing that the
component-aware check later catches, at a fraction of the cost. `mr_022_L`
and `mr_008_L` were never flagged downstream and are worth a look.

### 2.3 Stage D is where the lumen is lost

Straight 60 mm tubes of known radius through the real `generate_mesh` path
(iso level pinned at 0.5, see §3.4), inscribed radius measured the way the
device sees it — distance from the axis to the nearest triangle:

| declared r | current, along z | current, along x | SDF, 0.6/0.9 voxels | SDF, 0.3 mm voxels |
|---|---|---|---|---|
| 1.0 | absent | absent | 0.92 | 0.97 |
| 1.2 | 0.66 | absent | 1.14 | 1.17 |
| 1.4 | 0.78 | **0.47** | 1.35 | 1.37 |
| 1.6 | 1.11 | **0.72** | 1.54 | 1.57 |
| 2.0 | 1.62 | 1.41 | 1.96 | 1.97 |
| 2.5 | 2.22 | 2.05 | 2.47 | 2.47 |
| 3.0 | 2.83 | 2.66 | 2.97 | 2.96 |
| 4.0 | 3.69 | 3.64 | 3.98 | 3.96 |

Three things are visible: the current mesher removes 0.3–0.9 mm of radius;
it removes **more from vessels running in the axial plane** (the z spacing is
0.9 mm and σ is set in voxels, so the blur is physically anisotropic — this is
why the earlier calibration found tubes first appear at 1.15 mm along z but
1.45 mm across); and the erosion is not the voxel grid's fault, since a
signed-distance field on the same grid is within 0.06 mm from r = 1.2 up.

On two real anatomies, lumen radius along the RCCA route (distance from each
centerline point to the mesh, end caps excluded):

| mesh | triangles | components | open edges | min lumen (mm) | declared there | median deficit | build |
|---|---|---|---|---|---|---|---|
| **`topcow_mr_001`** — declared min 0.84 mm at 226 mm | | | | | | | |
| baked `.obj` (current) | 3,709 | 3 | 3 | **0.35** | 1.17 | **0.64** | 10.8 s |
| current, no decimation | 371,020 | 3 | 0 | 0.31 | 0.84 | 0.34 | 8.0 s |
| SDF, no decimation | 387,140 | 1 | 0 | 0.76 | 0.84 | **0.06** | 8.8 s |
| SDF at the baked triangle count | 3,708 | 1 | 1 | 0.24 | 1.17 | 0.49 | |
| SDF 0.45 mm isotropic, no decimation | 924,328 | 1 | 0 | 0.82 | 0.84 | 0.03 | 29 s |
| SDF → `vmtksurfaceremeshing`, edge 1.4 mm on route → 6 mm far | 19,984 | 1 | 0 | 0.76 | 0.84 | 0.11 | |
| SDF → remesh, 1.8 → 7 mm | 16,190 | 1 | 0 | 0.54 | 0.84 | 0.16 | |
| SDF → remesh, uniform 2.5 mm | 37,628 | 1 | 0 | 0.52 | 0.84 | 0.26 | |
| **`case_k_004_left__topcow_mr_010`** — declared min 1.52 mm at 238 mm | | | | | | | |
| baked `.obj` (current) | 3,756 | 4 | 8 | **0.43** | 1.52 | **0.62** | 10.6 s |
| current, no decimation | 375,704 | 4 | 0 | 0.86 | 1.52 | 0.30 | 8.2 s |
| SDF, no decimation | 392,084 | 1 | 0 | 1.45 | 1.52 | **0.05** | 8.2 s |
| SDF at the baked triangle count | 3,756 | 1 | 0 | 0.96 | 1.52 | 0.48 | |
| SDF 0.45 mm isotropic, no decimation | 936,768 | 1 | 0 | 1.48 | 1.52 | 0.02 | 29 s |
| SDF → remesh, 1.4 → 6 mm | 21,502 | 1 | 0 | 1.39 | 1.52 | 0.10 | |
| SDF → remesh, 1.8 → 7 mm | 17,100 | 1 | 0 | 1.32 | 1.52 | 0.14 | |
| SDF → remesh, uniform 2.5 mm | 38,296 | 1 | 0 | 0.28 | 1.81 | 0.24 | |

Reading it:

- **The shipped meshes carry a median radius deficit of 0.62–0.64 mm along the
  route.** Set-wide (`saved/mesher_probe/lumen_v1.json`, all 264 v1 anatomies):
  median minimum lumen 0.56 mm (A) / 0.53 mm (B), median deficit 0.65 mm in
  both, and only **93 of 264 — 22/49 A, 71/215 B — pass the meshed-lumen test**
  (minimum lumen − 0.3 mm contact ≥ the 0.35 mm catheter radius). At the tightest point 12 mm short of the terminus the lumen is
  0.35 / 0.43 mm radius where 1.17 / 1.52 mm is declared. The catheter is
  0.35 mm radius and SOFA adds 0.3 mm of `contactDistance`: the device cannot
  physically reach the last centimetre of these anatomies, and
  `check_anatomies.py`'s `fit` test cannot see it because it reads the
  *declared* radius.
- The deficit splits roughly in half: **~0.3 mm from the double Gaussian,
  ~0.3–0.45 mm from decimating to 1 %.** (The SDF at the same 3.7 k budget
  still loses 0.48 mm — the budget alone does that.)
- Replacing "binary spheres + Gaussian + iso 0.5" with "signed distance + iso 0"
  removes the smoothing loss entirely (0.06 mm), produces **one** connected
  component instead of 3–4 (the strays are thin vessels the blur severed), and
  costs the same 8–9 s.
- Decimation to the current budget throws the gain away again; spending the
  triangles **where the device goes** keeps it. `vmtksurfaceremeshing` with a
  per-point edge length (fine within a few mm of the route, coarse elsewhere)
  gives 0.10–0.11 mm deficit, one manifold component, zero open edges, at
  ~5× the current triangle count. A uniform 2.5 mm remesh is worse than the
  adaptive one at twice the triangles, and collapsed a thin distal section of
  the carotid case — coarse uniform edges are the wrong tool for a tree whose
  radii span 0.8–12 mm.
- `decimate_pro` (VTK's progressive decimator) destroyed both meshes at 0.99
  (40 route points outside, lumen 0.00). Quadric it must be, and with
  `volume_preservation=True`, which is free.

---

## 3. Defects and faults in what is implemented

1. **Erosion by design (D).** A binary tube blurred at σ = √2 voxels and cut
   at 0.5 shrinks every convex surface by ≈ σ²/(2r) and deletes anything
   thinner than ≈ 1.2 σ. Every radius floor, distal trim and fusing band in
   stage C is a patch over this one choice.
2. **Anisotropic erosion (D).** σ is in voxels on a [0.6, 0.6, 0.9] grid, so
   horizontal vessels — the arch, the RVA take-off, the ECA — lose up to
   0.3 mm more radius than superior-running ones (table §2.3).
3. **Data-dependent iso level (D).** `marching_cubes(level=None)` means
   (min + max)/2. It is 0.5 only because the arch saturates the field to 1.0.
   Any cube without a fat vessel (a test tube, a future thin-vessel-only set)
   silently iso-surfaces at a lower level and comes out *fatter*. Pin it.
4. **Decimation that ignores where the device goes (D).** `decimate(0.99)`
   minimises global quadric error, so it spends the budget on the 12 mm arch
   and starves the 1–2 mm siphon. 3.7 k triangles over roughly a metre of
   vessel is ~4 per millimetre; a 1.5 mm vessel needs ~10 per millimetre to
   hold its inradius within 0.1 mm.
5. **Severed strays.** The baked meshes have 3–4 connected components; the
   extras are thin branches the blur cut into islands. They are garbage in the
   collision set and the reason the component-aware route check had to be
   written.
6. **The fit check tests the wrong radius (E).** `check_anatomies.py` compares
   the catheter OD with the *centerline* radius. The meshed lumen at the same
   point can be a third of that (§2.3), and with `contactDistance = 0.3` the
   navigable radius is smaller still.
7. **Real lumen shape discarded (A→D).** The label surface / Zenodo STL is used
   only to seed VMTK. Stenoses, elliptical sections and the bulb become
   circular tubes. The host test anatomy, by contrast, *is* its real surface —
   a train/test mismatch in wall geometry on top of the anatomy split.
8. **End caps.** Spheres at the terminus blurred to nothing → `DISTAL_TRIM_MM`.
   An SDF cap is a hemisphere of full radius; the trim becomes unnecessary,
   and a 2–3 mm flow-extension-style overshoot would put every terminal target
   in real lumen.
9. **Label necks are not screened (A).** One-voxel necks in the label predict
   both known failures (§2.2); the pipeline discovers them four stages later.
10. **Unwelded host mesh (E).** The shipped `.obj` is a triangle soup with
    10,750 open edges. SOFA copes, but it is not what the baked meshes are,
    and any manifold-dependent check on it is meaningless.

## 4. Inefficiencies

- The whole cube is densified and blurred twice (~40 M voxels) to place a
  surface that occupies a thin shell; a band-limited field (what the SDF probe
  does) costs the same today only because marching cubes still walks the full
  array — `skimage.measure.marching_cubes(mask=…)` restricts it.
- `mark_centerline_in_array` is a Python loop per centerline point, each
  building index grids; vectorisable, and irrelevant once the field is an SDF.
- Two conda environments (base without VMTK, `vmtk_env` without scipy/skimage)
  with a `.vtp` handoff, purely because the `vmtk` package was installed bare.
  One environment with `vmtk` + `scipy` + `scikit-image` pinned removes stage
  A/B's split and made this analysis need a scipy-free workaround.
- Meshing at [0.6, 0.6, 0.9] to "save cost" and then decimating 99 % is
  spending resolution in the wrong place: the voxel size limits the wall
  resolution (fusing band), the decimation limits the lumen accuracy, and
  neither is where SOFA's cost lives. Whether SOFA's BVH narrow phase even
  notices 4 k vs 20 k static triangles has never been measured — the
  "collision cost is the bottleneck" line in `graft_three.py` is an
  assumption carried forward, and it is the one that fixes the budget.
- The radius floors (`ROUTE_MIN_R 1.60`, `ECA_MESH_R_MM 1.6`) erase stenosis
  above 37 % to survive a mesher that would have sealed them. With an SDF mesh
  a 1.0 mm tube comes out at 0.97 mm; the floor could drop to ≈ 1.0 mm
  (catheter 0.35 + contact 0.3 + margin) and the 40–74 % grades come back.

---

## 5. What VMTK provides, per stage (this install: 151 script modules; full option dump in `saved/mesher_probe/vmtk_options.txt`)

### Label-map preprocessing (image stage)

| script | options (default) | use here |
|---|---|---|
| `vmtkimagereader` | Format (nifti via ITK), Flip, DesiredOrientation | read the NIfTI directly, no nibabel round-trip |
| `vmtkimagebinarize` | Threshold, LowerLabel, UpperLabel | isolate label 4 / 6 |
| `vmtkimagemorphology` | Operation ∈ {dilate, erode, open, **close**}, BallRadius [1,1,1] per axis | **close** with radius (1,1,1) bridges one-voxel necks; open removes specks |
| `vmtkimagesmoothing` | Method ∈ {gauss, anisotropic}; StandardDeviation *in real units*; Conductance, NumberOfIterations, TimeStep | smooth a *signed distance* or level-set image, not the binary label |
| `vmtkimageinitialization` | Method ∈ {isosurface, threshold, collidingfronts, fastmarching, seeds}; IsoSurfaceValue; Source/TargetPoints (IJK) | seed a level set from the label surface (`isosurface`) |
| `vmtklevelsetsegmentation` | LevelSetsType ∈ {geodesic, curves, threshold, laplacian}; PropagationScaling, **CurvatureScaling**, AdvectionScaling; NumberOfIterations; SmoothingIterations/TimeStep/Conductance; FeatureImageType | regularise the label surface: curvature term closes gaps and rounds staircase without shrinking a tube the way a Gaussian does |
| `vmtkimagevesselenhancement` | frangi / sato / ved; SigmaMin/Max, steps, Alpha/Beta/Gamma | only relevant with the underlying MRA, which TopBrain does not ship |
| `vmtkimagefeaturecorrection` | Sigma, ScaleValue | CT bone/air correction — not applicable |
| `vmtkmarchingcubes` | Level, **Connectivity** (largest region only) | iso-surface with the largest-component filter built in |
| `vmtksurfacetobinaryimage` | PolyDataToImageDataSpacing, Inside/OutsideValue | voxelise a surface (the STL) onto the mesher grid |

Not in VMTK but the same category: `scipy.ndimage.binary_fill_holes`,
`distance_transform_edt` (used in `label_qa.txt`), `skimage.morphology.skeletonize`.

### Surface stage

| script | options (default) | use here |
|---|---|---|
| `vmtksurfacesmoothing` | Method ∈ {**taubin**, laplace}; NumberOfIterations; PassBand (0.1 typical); RelaxationFactor; BoundarySmoothing; NormalizeCoordinates | Taubin ≈ the windowed sinc already used; volume-preserving. Laplace shrinks — avoid |
| `vmtksurfaceremeshing` | ElementSizeMode ∈ {area, edgelength, areaarray, **edgelengtharray**}; TargetEdgeLength; TargetEdgeLengthArrayName; TargetAreaFactor; TriangleSplitFactor; Min/MaxArea; NumberOfIterations (10); AspectRatioThreshold 1.2; NormalAngleTolerance; CollapseAngleThreshold; PreserveBoundaryEdges | **the importance-weighted remesh of §2.3**: per-point edge length from distance to route (and local radius) |
| `vmtksurfacedecimation` | TargetReduction, BoundaryVertexDeletion | plain decimation; pyvista's quadric with volume preservation is better |
| `vmtksurfacesubdivision` | linear / butterfly / loop | refine before remeshing a coarse input |
| `vmtksurfacecapper` | Method ∈ {simple, centerpoint, smooth, annular}; NumberOfRings; ConstraintFactor | cap open ends before centerline extraction or booleans |
| `vmtksurfaceconnectivity` | Method ∈ {largest, closest, all}; ClosestPoint; GroupIds | drop severed strays; "closest" keeps the component holding the insertion point |
| `vmtksurfaceclipper` / `vmtksurfaceendclipper` | box/sphere or scalar clip; CenterlineNormals | clip ends normal to the centerline (clean cut faces instead of the "drooping" cut faces found in the Zenodo STLs) |
| `vmtkflowextensions` | ExtensionMode ∈ {centerlinedirection, boundarynormal}; ExtensionLength / Ratio / Radius; TransitionRatio; InterpolationMode (thin-plate spline to circular) | extend a terminus by a few mm so terminal targets sit in lumen — removes the need for `DISTAL_TRIM_MM` |
| `vmtksurfacebooleanoperation` | Operation ∈ {union, intersection, difference}; Method ∈ {default, loop}; Tolerance | union real donor surfaces with the tube host (fragile on near-tangent surfaces; the SDF union in §6 is more robust) |
| `vmtksurfacedistance` / `vmtkdistancetocenterlines` | signed distance to a reference surface / tube-function evaluation on a surface | QA: real surface vs tube |
| `vmtksurfacecurvature` | mean / gaussian / min / max, median filtering | QA |

### Centerline stage

| script | options (default) | use here |
|---|---|---|
| `vmtkcenterlines` | SeedSelectorName ∈ {pickpoint, openprofiles, **carotidprofiles**, profileidlist, idlist, **pointlist**}; AppendEndPoints; **Resampling + ResamplingStepLength**; CostFunction (1/R); SimplifyVoronoi; UseTetGen; CapDisplacement; DelaunayTolerance; StopFastMarchingOnReachingTarget; CheckNonManifold; FlipNormals | `carotidprofiles` picks CCA/ICA/ECA open profiles automatically — the Zenodo STLs are exactly that shape; `Resampling` avoids the ad-hoc resample downstream |
| `vmtknetworkextraction` / `vmtkcenterlinesnetwork` | AdvancementRatio | seed-free network for a whole tree; no MISR-quality radius |
| `vmtkcenterlineresampling` | Length (spline) | uniform spacing before grafting |
| `vmtkcenterlinesmoothing` | NumberOfSmoothingIterations (100), SmoothingFactor (0.1) | moving-average; the source of the "resampling jitter" caveat if applied blindly |
| `vmtkcenterlineattributes` | Abscissas, **ParallelTransportNormals** | a roll-free frame along the vessel — the principled version of the "superior" axis in `frame_rotation` |
| `vmtkcenterlinegeometry` | Curvature, Torsion, Tortuosity, Frenet frame; LineSmoothing | the metrics slide 14 computes by hand |
| `vmtkbranchextractor` + `vmtkbifurcationreferencesystems` + `vmtkendpointextractor` | GroupIds, Blanking, TractIds | split a tree at bifurcations — replaces `split_tree`'s common-prefix heuristic |
| `vmtkcenterlinesections` | section area, min/max diameter, **shape index**, closed | real cross-section instead of MISR |
| `vmtkcenterlinemerge`, `vmtkcenterlineoffsetattributes` | | tree bookkeeping |

### Centerline → surface (the alternative to stage D)

| script | options | what it is |
|---|---|---|
| `vmtkcenterlinemodeller` | RadiusArrayName, SampleDimensions, ModelBounds, NegateFunction | evaluates the **tube function** (signed distance to a swept tube with interpolated radius) on an image — exactly the SDF mesher of §2.3, with radius interpolated *along* the line rather than a union of discrete spheres |
| `vmtkpolyballmodeller` | same | union-of-spheres version |
| `vmtkmarchingcubes` | Level = 0, Connectivity | iso-surface it |
| `vmtkmeshgenerator` | TargetEdgeLength(+Array), boundary layers, tetgen | CFD volume meshes — not needed |

---

## 6. What to change, in order of payoff

1. **Mesh from a signed distance field, iso-surface at zero, no Gaussian.**
   `probe.py::mesh_sdf` already does it band-limited on the existing grid;
   `vmtkcenterlinemodeller` is the packaged equivalent. Median deficit
   0.64 → 0.06 mm, one component, same build time. Pin the level explicitly.
   Everything below assumes this.
2. **Spend triangles by importance, not by global error.** Either
   `vmtksurfaceremeshing` with a per-point edge length (`edgelengtharray`:
   ≈ 0.8 × local radius within ~8 mm of the route, 6–7 mm elsewhere), or a
   quadric decimation constrained to weld its region seams. Measured:
   0.10 mm deficit at 20 k triangles, 0.16 mm at 16 k. Then **measure SOFA
   step time at 4 k / 10 k / 20 k / 40 k triangles** before choosing — the
   budget is the one number in this pipeline that was never tested.
3. **Add the meshed-lumen profile to `check_anatomies.py`.** `lumen_along()`
   in the probe: distance from each route point to the mesh, minus
   `contactDistance`, against the catheter radius. It is the check that would
   have caught §2.3 and it costs a second per anatomy.
4. **Screen the labels for necks before anything else** (`label_qa.txt`
   logic: skeleton EDT < 0.6 mm). Repair with `vmtkimagemorphology close`
   (1,1,1) or a skeleton-constrained dilation to a 0.6 mm floor; reject if the
   neck is long. This front-loads what the component check finds last.
5. **Isotropic voxels.** With a band-limited SDF the grid is cheap: 0.45 mm
   isotropic measured 29 s and 0.02–0.03 mm deficit. It also shrinks the
   fusing band toward one voxel — re-derive `FUSE_BAND_MM` on the new mesher
   rather than carrying 0.35 forward.
6. **Lower the floors.** With (1) a 1.0 mm tube meshes at 0.97. `ROUTE_MIN_R`
   1.60 → ≈ 1.0 mm restores stenosis grades to ~60 % (NASCET-significant), and
   `ECA_MESH_R_MM` can follow. Re-run the ring/re-entry gates when it drops.
7. **Drop `DISTAL_TRIM_MM`; extend instead.** SDF caps are full-radius
   hemispheres; a 2–3 mm `vmtkflowextensions`-style overshoot puts terminal
   targets in lumen.
8. **Hybrid real surfaces.** Because everything is an SDF, union is
   `max(f_tube_host, f_donor_surface)`: voxelise the TopBrain label surface /
   Zenodo STL (`vmtksurfacetobinaryimage` → EDT, or `vtkImplicitPolyDataDistance`)
   in the same grid and combine. Real siphon and bulb geometry, circular tube
   only where the host is procedural. Same operation would give the host test
   anatomy the same wall character as the training set.
9. **One environment** with `vmtk` + `scipy` + `scikit-image`; stage A and B
   become one script and `carotidprofiles`/`Resampling` replace the hand-rolled
   seeds and resampler.

Items 1–3 are a day's work and change what every anatomy in both sets looks
like to the device; 4–7 fall out of them; 8 is the one that changes what the
policy is actually trained on.


---

## 7. v2 build — results (2026-09-03)

Both sets rebuilt with changes 1–7 into new folders (`topbrain_data/anatomies_v2`,
`carotid_data/anatomies_v2`; figures in `saved/figs/topbrain_v2`, `saved/figs/carotid_v2`;
plan and bug-guard table in `V2_BUILD_PLAN.md`). Same donors, same pairing, same
host; SDF mesher at 0.45 mm isotropic, iso 0, 20 k triangles; `DISTAL_TRIM 0`,
route floor 1.60 → 1.0, ECA floor 1.6 → 1.0, **siphon floor 1.0 (new, both sets)**,
fuse band kept at 0.35 (measured 0.2 on the new mesher).

Meshed lumen along the RCCA route, every anatomy (`saved/mesher_probe/compare_v1_v2.txt`):

| | A v1 (49) | **A v2 (49)** | B v1 (215) | **B v2 (223)** |
|---|---|---|---|---|
| navigable: min lumen − 0.3 contact ≥ 0.35 catheter | 22 (45 %) | **49 (100 %)** | 71 (33 %) | **223 (100 %)** |
| minimum lumen, median (p10) mm | 0.56 (0.15) | **1.44 (0.93)** | 0.53 (0.17) | **1.38 (0.82)** |
| median radius deficit, mm | 0.65 | **0.12** | 0.65 | **0.12** |
| mesh components, median (max) | 3 (4) | **1 (1)** | 3 (4) | **1 (1)** |
| per-anatomy min-lumen change v2 − v1 | | **+0.74 mm median, none worse** | | **+0.78 mm median, none worse** |

Set B built 223 of 237 pairs against v1's 215: the lower ECA floor removed eight
fusing rejections. Set A's centerlines are identical to v1's on the shared prefix
(49/49), +5 mm at the terminus (no trim), and six siphons have radii lifted to
1.0 mm at their label necks (`mr_001, 003_L, 007_L, 010_L, 013, 015`); rise, kink
and junction statistics are unchanged (`validate_anatomies.py`).

The shipped host tree through the v2 mesher (`topbrain_data/host_v2_control`):
0.99 mm minimum lumen, 0.14 mm deficit, navigable — the test anatomy meshed the
same way as the training set, should that be wanted.

Verification (`check_anatomies.py`, v2 with the meshed-lumen column): set B
**223/223 pass** the static checks — route inside one connected component,
enclosure, targets, declared fit, meshed lumen − contact ≥ catheter — and all
9 SOFA rollouts sampled across the three shards step normally with targets at
140–177 mm along the route, i.e. past the seam. Set A **49/49 pass** — including
`mr_015` and `mr_003_L`, which v1 had to exclude — and 6/6 SOFA rollouts with
targets at 195–244 mm. The two v1-style host controls (shipped tree, old mesher)
now **fail** the meshed-lumen test (0.00 mm at the worst point, 0.76 mm deficit,
5 components) while the same tree through the v2 mesher passes (0.99 mm, 0.14,
1 component): the test separates the meshers, not the anatomies.

Two things the build itself taught:

- **The meshed-lumen check caught a real gap on its first run.** `ROUTE_MIN_R` is
  scoped to the donor section, so set B's TopBrain siphons carried their label
  necks unfloored; 13 anatomies baked at 0.30–0.63 mm. `graft_three.py` gained
  `--siphon-min-r`, the 22 affected pairs were regrafted and re-baked. Keep the
  check in the bake, not only in the audit.
- **Cost.** 20 k triangles is +55 % SOFA step time over v1's 3.7 k (flat from 12 k
  to 38 k; 9 k is +25 % at 0.23 mm deficit). `collision_full.vtp` is kept per
  anatomy so `recut_obj.py --tris N` changes the budget for a whole set in seconds.

Not done in v2: the hybrid real-surface union (change 8) and the single
environment (change 9). The analytic swept-tube mesher was prototyped and set
aside (junction seams up to 7 mm without stitching).


---

## 8. v3 build — real surfaces in the training meshes (change 8)

Same centerlines as v2 (verified identical, 272/272 anatomies); only the
collision mesh changes. `topbrain_tools/sdf_union.py` adds each patient's real
lumen surface to the v2 tube field as a signed-distance union:

    f = max( f_tube ,  min( f_real , f_capsule ) )

- `f_real` — signed distance to the segmented surface (TopBrain label surface
  for the siphon; Zenodo lumen STL for CCA / ICA / ECA in set B), carried
  through the **same rotation, origin, anchor and mirror the graft applied to
  that section's centerline**. The grafters now record these
  (`graft_xform.json` in set A, `provenance.json["xform"]` in set B).
- `f_capsule` — a tube round the kept centerline of the section at 1.8 × MISR
  + 1 mm, tapered to the tube radius over 8 mm before every seam. It clips
  away the parts of the source surface the graft did not keep (trimmed inlet,
  label below the anchor trim, the ICA terminus widening, the far side of a cut
  face) and makes the handover to the tube continuous.
- `max` with the floored tubes — the v2 navigability guarantee survives:
  where the real surface pinches under the floor, the tube wins; everywhere it
  is wider, the real shape wins.

Surfaces are cleaned first (largest component, holes at the mask cut filled,
normals re-oriented outward after the transform — a mirror reverses winding
and the distance sign follows the normals).

Two measurement lessons from building it:

- **Distance from centerline to nearest wall cannot show a real surface.** On
  the centerline that distance *is* the MISR, by construction. The v3 report
  therefore casts 16 rays in the normal plane at 4 mm stations and reports the
  longest wall ray over MISR, the area-equivalent radius over MISR, and the
  max/min ray ratio. A tube reads 1.02 / 0.99 / 1.07; a real siphon reads
  ~1.23 / 1.10 / 1.22 (p90 up to 1.85) — the bulges and eccentric sections
  are in the mesh.
- **The marching-cubes band mask must be dilated, not eroded.** The union put
  iso-surface within a voxel of the band edge; the eroded mask cut 49 edges
  and left a stray triangle. A FAR corner next to a negative one cannot invent
  a surface, so dilating by two is safe and closes everything (`sdf_mesher.py`).

### v3 results (mesh level; `saved/mesher_probe/compare_v2_v3.txt`)

| | A v2 | **A v3** | B v2 | **B v3** |
|---|---|---|---|---|
| navigable | 49/49 | **49/49** | 223/223 | **223/223** |
| minimum lumen, median (p10) mm | 1.44 (0.93) | **1.49 (0.97)** | 1.38 (0.82) | **1.42 (0.86)** |
| deficit on tube sections | 0.12 | 0.12 | 0.12 | 0.12 |
| deficit on real sections | — | **0.04** | — | **0.06** |
| shape, tube sections: longest ray / MISR, area-r / MISR, max/min | 1.02 / 0.99 / 1.06 | same | same | same |
| shape, **siphon** (real) | — | **1.25 / 1.10 / 1.26** | — | **1.24 / 1.09 / 1.25** |
| shape, **CCA–ICA** (real, set B) | — | — | — | **1.13 / 1.04 / 1.14** |
| components / open edges (max) | 1 / 0 | 1 / 5 | 1 / 3 | 1 / 9 |

Reading it: the tube sections are unchanged from v2 (they are the same field);
the real sections now carry the segmentation's own cross-sections — the longest
wall ray runs a quarter beyond the inscribed radius on the siphons, and the
carotid bulb region an eighth — while the v2 navigability guarantee holds on
every anatomy (the floored tube is still inside the union). One set B anatomy
needed 30 k triangles instead of 20 k to keep its floored neck navigable after
decimation; the bake now steps the budget up automatically and records it
(`obj_tris_budget`). Fragments of source surface clipped off by the capsule
(8–2,300 triangles against ~900 k) are dropped and logged
(`iso_surface.dropped_fragments`).

Verification (`check_anatomies.py`, same suite as v2 plus the meshed-lumen
column): **A 49/49 and B 223/223 pass**; SOFA rollouts 6/6 (A) and 10/10 (B)
step normally with targets 140–244 mm along the route, past the seams.

Figures: `saved/figs/topbrain_v3/` (25, drawn from the baked v3 meshes, split
into the house colour groups by nearest branch) and `saved/figs/carotid_v3/`
(112, the existing carotid figure script reading the baked `.obj`).

What v3 still does not do: the host arch, trunk, RVA and the other shipped
branches remain tubes (the host is procedural — there is no segmentation to
use); the ICA extension templates on 34 short carotid donors are synthetic and
stay tubes; and the test anatomy is unchanged. If train and test are to share
construction exactly, the shipped VMR surface can go through the same union
(`host_v2_control` shows the tube half of that already passes).

