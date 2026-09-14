# Hybrid pelvis simulator — Stage 1 build contract

Purpose: predict, from the PRE-brachytherapy (preBT) MRI alone plus a planned applicator, the post-insertion
pelvic anatomy, as a **displacement field** and predicted contours, for pre-planning of MRI-guided cervical
brachytherapy. Not RL. The uterine corpus and tandem are placed **kinematically** by a geometric rule; the
FEM is used only for the compliant tissue around the device.

All agents building Stage 1 read this file. Where a detail is not fixed here, decide it, record the decision
in your module's README section, and keep it parameterised in `cfg`.

## 0. Ground rules
- Data folder `~/Downloads/MRI_GYN` is READ-ONLY. Possibly real patient data: everything stays
  local; never upload, never SendUserFile/Artifact, never put data/meshes/images in the repo.
- Code: `D:/neve/.claude/worktrees/sofa_applicator/applicator_sim/hybrid/` (this folder). Code only (.py .sh .md
  and small parameter-only .json under `defs/`). Do not git add/commit/stash anything.
- Derived data: `~/Downloads/MRI_GYN_sim/hybrid/` (create): `meshes/`, `applicator/`, `runs/`,
  `eval/`, `figs/`, `logs/`.
- Reuse from `applicator_sim/` (parent package, import as `sys.path` parent): `geom.py` (frames, projections),
  `align.py` (preBT<->BT rigid frames: E_app device-matched frame, peri-organ MI "BONE" frame), the label->surface
  code in `prep_inputs.py`, `evaluate.py`'s voxelisation (`vtkPolyDataToImageStencil` on the BT grid) and its
  `dice`/`surf_dists`. Existing measured inputs: `MRI_GYN_sim/inputs/*.json` (L_end, a0, O_pre, canal polyline,
  internal os, BT flange/axis, tandem radius/L_iu), `MRI_GYN_sim/final/bdev.json` (the depth rule).
- SOFA only in docker image `eve-training-fixed` (SOFA v22.12 + SofaPython3, Python 3.8). Run containers via
  `applicator_sim/run_docker.sh <tag> hybrid/<script>.py ...` (mounts `applicator_sim` at `/app`, data at
  `/data:ro`, `MRI_GYN_sim` at `/out`; `--cpus` defaults to 12; one compute container at a time; the user's
  live viewer container `appsim_gui_*` may be open — never stop it). Host python: `python` (3.13: numpy, scipy,
  nibabel, scikit-image, vtk, matplotlib, PIL, imageio) and `py -3.11` (pyvista 0.46, vtk, numpy, PIL — no
  nibabel/scipy). Extra packages only via `pip install --target <scratch>/pk` and `sys.path`.
- Units: mm, kg, s (stress kPa, force mN, stiffness mN/mm, density 1.05e-6 kg/mm^3). Frame: **preBT world
  RAS mm** from the nibabel affine (x=R, y=A, z=S). Every JSON states its frame and units.

## 1. Bodies (from the preBT labels; voxel 1.125x1.125x1.6 mm)
Assign each voxel to ONE body by priority `corpus > cervix > vagina > bladder > rectum > sigmoid`; keep the
largest connected component of each (the rectum has stray voxels). Then:

| body | source | role | notes |
|---|---|---|---|
| corpus | uterus label (37.5 cc) | **rigid, kinematic** | attached to the tandem by the pose rule (§3) |
| cervix | HR-CTV label minus uterus label (~46 cc, includes the tumour) | deformable | the tandem passes through it along the preBT canal (IUcanal label below the internal os) |
| vagina | vagina label (10.8 cc, collapsed tube) | deformable | lower ~15 mm fixed (pelvic floor / out of FOV) |
| bladder | bladder label (251 cc) | deformable, near-incompressible (filled) | soft solid; filling is a scenario parameter later |
| rectum | rectum label (39 cc) | deformable | cut ends fixed |
| sigmoid | sigmoid label (40 cc) | deformable | cut ends fixed |

Surfaces: marching cubes in world mm, windowed-sinc smoothing, decimate to ~3-6k triangles per body, closed
(0 open edges), volume within 2% of the label. Volumetric: tetrahedra, target edge ~3-4 mm (bladder may use
4-5 mm), min dihedral > 10 deg; prefer TetGen on the host (`py -3.11 -m pip install --target ... tetgen`),
fall back to SOFA `MeshTetraStuffing` inside the container (then export once with `VTKExporter`). Output per
body: `meshes/<body>/surface.obj`, `meshes/<body>/tets.vtk` (legacy VTK unstructured grid, points in preBT mm),
`meshes/<body>/meta.json` (volumes, counts, quality, named node sets below) and `meshes/bodies.json` (index).

Named node sets (indices into `tets.vtk` points), computed by the meshing module:
- `cervix.interface_corpus`: cervix surface nodes within 1.5 mm of the corpus surface (the internal-os junction).
- `cervix.canal`: cervix nodes within 3.0 mm of the preBT canal polyline below the internal os.
- `cervix.lateral_os_level`: cervix surface nodes within 6 mm of the internal-os level, |x - x_os| > 10 mm
  (cardinal-ligament attachment ring, left and right).
- `vagina.fixed_inferior`: vagina nodes in the lowest 15 mm of the vagina label along the vaginal axis.
- `vagina.apex`: vagina surface nodes within 3 mm of the cervix surface (fornices).
- `bladder.anterior_support`: bladder surface nodes on the anterior-superior third (facing +y / +z).
- `rectum.posterior_support`, `rectum.fixed_ends`, `sigmoid.fixed_ends`: cut ends = nodes within 4 mm of the
  label's extreme slices along its own axis.
- For every body: `surface_nodes`.

## 2. Applicator (Venezia-type hybrid, parametric, from this case's measurements)
Module `applicator_venezia.py` (host, numpy + pyvista). Applicator frame: origin = flange centre, `z` = intrauterine
tube axis (flange -> tip), `y` = anterior in the sagittal plane, `x = y × z` (patient right).
- Intrauterine tube: straight cylinder, radius 2.18 mm, length L_iu = 61.8 mm (flange -> tip), hemispherical tip.
- Shaft below the flange: an arc of 30 mm, tangent to -z at the flange, bending anteriorly to reach the
  measured 22-25 deg junction angle (`angle_deg` parameter, default 24); it is outside tissue except the vagina.
- Ovoids: two lunar (half-ellipsoid) caps mounted around the tube, assembly extents 41.5 (LR) x 33.7 (AP) x
  22.2 (along z) mm, centred 2.6 mm below the flange along -z (`sph_dz` from the calibration), each ovoid
  centre at x = ±11 mm; parameters `ovoid_diam_mm` (default from the BT ovoid label fit), `ovoid_gap_mm`.
  Fit the two-ovoid model to the BT ovoid label (report Dice of the fitted rigid model vs the label; the
  remainder is packing and is NOT part of the rigid device).
- Needle channels (Stage 2, define geometry now): 5 parallel channels per ovoid on a 17 mm radius from the
  tube axis, plus 2 oblique channels per ovoid at 15 deg; needle radius 1.0 mm. Store axes in `applicator.json`.
- Outputs: `MRI_GYN_sim/hybrid/applicator/{tube.obj, shaft.obj, ovoid_L.obj, ovoid_R.obj}` in the applicator
  frame, `applicator.json` (all parameters, channel axes, units), and a `pose.json` from §3.

## 3. Kinematic pose rule (v2) and insertion path
Everything below is computed from preBT geometry only (plus the chosen device); the real BT pose is used only
to SCORE the rule. **Rule v1 (shaft axis = vaginal chord, flange pinned at the external os) was scored and
rejected on 2026-09-11** (axis 23 deg off, flange 30 mm too inferior: the device + packing push the cervix
cranially; `pose.json[validation][rule_v1]`). Rule v2 replaces steps 1 and 3 as follows.
1. Device shaft axis in the pelvis: the principal axis of the preBT vagina body (oriented superiorly), through the
   external os `O_pre` (or through the vagina axis line: `shaft_line_through`). The flange sits at
   `base + Delta` along the shaft axis, where **Delta = seating shift of the ovoids/packing is an explicit
   SCENARIO parameter** (not predictable from preBT; swept; ~30 mm on this case, in-sample). The scene module
   picks Delta from its cfg; `pose.json` provides the corpus target per Delta (`corpus.by_flange_shift_mm`).
2. Intrauterine tube axis: the shaft axis rotated by `angle_deg` anteriorly in the sagittal plane (about the
   patient LR axis), applied at the flange.
3. Corpus placement: the rigid transform that maps the preBT cervical canal axis `a0` (through `L_end`) onto
   the tube axis with the canal end at the tip: flange depth rule `d_F = L_iu + 0.5 - canal_above_L_end`
   (= 17.5 mm here; `bdev.json`), minimal rotation, roll preserved. The corpus (and the tube, rigidly attached
   to it) moves from its preBT pose to this target pose.
4. Insertion path (parameter `u` in [0,1], N_steps default 80): the device translates along the shaft axis from
   tip 4 mm below the external os until the flange reaches the os; the corpus rigid motion is interpolated
   (screw motion) from identity to the target, starting when the tip passes the internal os and completing when
   the flange seats; then a settle phase until convergence (§5).
5. Score the rule on this case (applicator module): predicted tube pose vs the real BT tandem pose, expressed in
   the peri-organ MI frame from `align.py`: angle (deg) and flange offset (mm). Report it in `pose.json`.

## 4. SOFA scene (Stage 1)
`scene_hybrid.py` with `createScene(rootNode)` (reads `APPSIM_CFG` for the live viewer) and `run_hybrid.py`
(headless driver). Patterns: `applicator_sim/scene.py`, `controller.py`, and the neuro scene
`eve/eve/intervention/simulation/sofabeamadapter.py` (FreeMotionAnimationLoop, GenericConstraintSolver/LCP,
LocalMinDistance, FrictionContactConstraint, LinearSolverConstraintCorrection).
- Per deformable body: `EulerImplicitSolver` (rayleigh 0.1/0.1) + `SparseLDLSolver`, tet topology from
  `tets.vtk`, `TetrahedralCorotationalFEMForceField` (default; `TetrahedronHyperelasticityFEMForceField`
  NeoHookean as option `material=neohookean` for the near-incompressible bodies), `MeshMatrixMass`,
  `Tetra2TriangleTopologicalMapping` surface with `TriangleCollisionModel` (+Line/Point), `OglModel` with a
  distinct colour per body, `LinearSolverConstraintCorrection`.
- Materials (assumed defaults, cfg): cervix 30 kPa nu 0.45; vagina 15 kPa; bladder 8 kPa nu 0.49; rectum,
  sigmoid 8 kPa nu 0.45. Gravity 0 (the supine rest state is the preBT geometry). dt 0.02 s, quasi-static via
  damping and a convergence criterion.
- Corpus: `MechanicalObject template=Rigid3d` driven by the controller (pose rule §3), with a `RigidMapping`
  child carrying the corpus surface (collision + visual) and the tube + ovoid surfaces (collision + visual).
- Cervix coupling: `AttachConstraint` between `cervix.interface_corpus` nodes and their rigid-mapped copies on
  the corpus; the canal nodes (`cervix.canal`) tied laterally to the tube axis (sliding along it) once the tip
  has passed them — reuse the distributed-node-tie idea from `ties.py`/`controller.py`, or
  `SlidingConstraint`.
- Supports (cfg-parameterised, defaults justified in the README): cardinal ligaments = springs from
  `cervix.lateral_os_level` to fixed points 25 mm lateral (k default 20 mN/mm per node); `vagina.fixed_inferior`
  fixed; `bladder.anterior_support` soft springs to rest (k 2 mN/mm per node); rectum posterior soft springs to
  rest (2 mN/mm), `rectum.fixed_ends`, `sigmoid.fixed_ends` fixed.
- Contact: all deformable surfaces vs each other and vs the device surfaces; no self-collision; contactDistance
  0.5 mm, alarmDistance 1.5 mm, friction 0.2. The ovoids seat against the portio/fornices via contact only.
- Logging per step (`log.jsonl`): u, phase, per-body max/mean displacement, contact count, constraint residual,
  step wall time; stop rules: NaN, inverted tets (min volume ratio < 0.2), wall time.
- Outputs `runs/<tag>/`: `cfg.json`, `log.jsonl`, `final/<body>.obj` (deformed surfaces, preBT mm),
  `final/<body>_u.npy` (per-node displacement, preBT mm, same order as `tets.vtk` points), `device_final.json`
  (flange, axis, ovoid poses), `summary.json` (steps, ms/step, convergence, forces).
- Live view: `view_hybrid.sh <tag>` = `runSofa -l SofaPython3 scene_hybrid.py` over X11 (copy `view_gui.sh`).

## 5. Convergence
Settle phase converged when, for 5 consecutive steps, max nodal displacement change < 0.02 mm/step AND the
contact/constraint residual is below tolerance; report the projected remaining drift.

## 6. Evaluation (Stage 1, `eval_hybrid.py`, host)
Frames (from `align.py`): `E_app` (simulated tube superimposed on the real BT tandem; scores the tissue around
the device independently of the pose rule) and the peri-organ MI frame `PELVIS` (scores the pose rule and the
OARs). Structures scored against the BT labels: cervix (BT HR-CTV minus BT uterus), vagina (tissue; the BT
vagina label includes the device, so score `vagina ∪ device` in both), bladder, rectum, sigmoid, and the corpus
(reported, not a target). Metrics: Dice, mean surface distance, HD95, plus device-to-OAR minimum distances
(predicted vs BT: applicator-rectum 2.76 mm, applicator-bladder 6.27 mm), tip-to-serosa margin (BT 5.44 mm),
and the pose-rule error (§3.5).
Comparators, all pre-declared: `NONE` (preBT organs unmoved, PELVIS frame), `K0` (corpus+cervix moved rigidly by
the pose rule, all else unmoved), `SIM` (the hybrid run). Report every SIM-minus-comparator difference with the
frame-perturbation noise floor used in `applicator_sim/final_eval.py` (30 rigid perturbations, ±0.5 mm
equivalence margin).
Displacement field: rasterise the per-body nodal `u` onto the preBT grid (barycentric inside bodies; outside
bodies a smooth extrapolation from the nearest body surfaces, flagged in a mask); write
`eval/<tag>/field/u_preBT.nii.gz` (4-D, 3 components, mm, RAS, preBT affine), a warped preBT image and warped
labels as a sanity check. Provide `--ref-dvf <path>` to compare against a reference field (RayStation export,
DICOM REG or NIfTI; format to be confirmed) with per-structure error statistics; if absent, print the hook.

## 7. Definition of done for Stage 1
1. All six bodies meshed and visible as meshes in the live `runSofa` window (no MRI backdrop needed).
2. One headless run converges, all stop rules pass, with `final/` outputs and the displacement field.
3. `eval/<tag>/metrics.json` with SIM vs NONE vs K0 for cervix, vagina, bladder, rectum, sigmoid, the pose-rule
   error, and device-to-OAR distances; figures in `figs/`.
4. `hybrid/README.md`: components, parameters with units, measured vs assumed, exact commands, results table,
   limitations. No patient-derived coordinates hard-coded in code (read them from `MRI_GYN_sim/inputs`).
