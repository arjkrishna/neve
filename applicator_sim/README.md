# applicator_sim: SOFA prototype of tandem-and-ovoid insertion into pre-brachytherapy anatomy

**What this is.** A working, honest prototype built on **one patient**. A rigid gynecologic tandem-and-ovoid
applicator is driven kinematically into the pre-brachytherapy (preBT) uterus and cervix. The tissue is a
volumetric FEM body in SOFA v22.12 (SofaPython3). The simulated result is compared with the segmentations of the
brachytherapy-session MRI (BT), where the real applicator is in place.

**Bottom line (details in [Results](#results)).** The pipeline runs end to end and converges under strict
numeric gates. On this patient, however, the FEM insertion does **not** beat a simpler comparator: the undeformed
preBT anatomy placed rigidly on the real tandem (`B_dev`), which uses no simulation at all. The pre-registered M1
validation fails. Every calibrated number is a fit, not a prediction.

This folder holds **code only** (`.py`, `.sh`, `.md`, small parameter-only `.json` in `runs_def/`). The images
are read-only. All derived data (meshes, runs, logs, evaluations, figures) stays local in `$APPSIM_OUT`
(default `~/Downloads/MRI_GYN_sim`). The data may be real patient data. Nothing is uploaded,
and no patient identifier is transcribed anywhere. The results table below holds only aggregate agreement metrics
(Dice, distances in mm): no images, coordinates or identifiers.

## Units
The code works in mm, kg and s. It follows that:

| Quantity | Unit |
|---|---|
| Force | mN |
| Stress, Young's modulus | kPa (= mN/mm^2) |
| Spring stiffness | mN/mm |
| Winkler (foundation) modulus | mN/mm^3 |
| Moment | mN*mm |

Every configuration key carries its unit in its name (`config.py`).

## Model
**Body.** The body is the preBT uterus, HR-CTV and IUcanal. Its surface (`pre_body.obj`, 83 cc) is voxelised by
`SparseGridTopology` into 4 mm hexahedra and modelled with `HexahedronFEMForceField` (corotational "large", one
material). It is solved quasi-statically, one `StaticSolver` solve per step, with `CGLinearSolver`. Gravity is 0,
and the preBT state is taken as stress-free.

**Applicator (kinematic, rigid).** The tandem is a straight rod: L_iu 61.8 mm from flange to tip, radius 2.18 mm,
with a 26 mm shaft. The ovoids and cap are a 14-sphere pack. The device follows this schedule:
1. Approach (phase A): a null test in which every tie is engaged with zero force.
2. Insertion (phase T): the flange moves by pivot-lerp and the axis by slerp, at most 1 mm and 0.5 deg per step.
3. Hold (phase H): the device stays put until equilibrium.

**Coupling (all forces act on FEM nodes).**
- **Canal ties.** The canal points are grouped per rest cell into one tie point (their mean position), with the
  members' stiffness summed. Ties act only perpendicular to the rod, so sliding along it is frictionless. They are
  distributed to the cell's 8 nodes with trilinear weights and implemented by `ties.py:NodeTieFF`, a matrix-free
  Python ForceField. A matrix-free force field needs CG.
- **Tip push.** The last 3 canal points are pushed ahead of the rod tip. The push is one-sided: it never pulls.
- **Equality enforcement.** An augmented-Lagrangian correction enforces the tie equality during the hold. It moves
  at most 0.5 mm per step and is clipped at 6 mm.
- **Ovoid contact.** A frictionless penalty acts on the true body-surface vertices along the normal of a smooth
  sphere-union SDF.
- **Supports.** A Winkler foundation acts on all grid boundary nodes (`kappa_f`), plus a paracervical band
  (`kappa_lig`, extent `lig_h_max_mm`).
- **Support anchors.** Anchors sit at the preBT rest positions by default. `found_anchor="device_rigid"` carries
  them instead with the rigid placement that puts the canal on the tandem.

**Hold convergence (rule v2).** The hold converges only when all of these hold for 3 consecutive steps:
- max nodal |dx| < 0.05 mm per step;
- max nodal force residual < 1 mN;
- the multipliers moved < 0.01 mm;
- tie residual mean < 0.3 mm and max < 0.6 mm;
- penetration < 0.5 mm;

and, in addition, the geometric-series drift estimate dx*rho/(1-rho) is < 0.05 mm. The hold runs CG with up to 800
iterations at tolerance 1e-10. The multipliers are frozen after 25 hold steps.

**Parameters** (tag: MEASURED from the images, ASSUMED, NUMERICAL device, or CALIBRATED as a stage-3 fit):

| Parameter | Value | Tag |
|---|---|---|
| Body geometry, canal centre line, internal os, L_end, a0 | preBT labels | MEASURED |
| Tandem L_iu 61.8 mm, r 2.18 mm, shaft 26 mm; 14-sphere ovoid pack | BT applicator/ovoid labels | MEASURED |
| Young's modulus E | 30 kPa | ASSUMED (literature 1-100 kPa); **not identifiable from shape**, forces scale with it |
| Poisson ratio nu | 0.445 (calibrated) / 0.45 (S0) | CALIBRATED/ASSUMED; not a meaningful knob (compressible linear law, locking near 0.5) |
| kappa_f, kappa_lig (mN/mm^3), band height (mm) | 0.0070, 0.0084, 17.9 (S0: 0.01, 0.1, 20) | CALIBRATED on the uterus (weakly identified; only kappa/E matters) |
| Fundal canal extension delta_fund | 7 mm (calibrated tip margin -6.7 mm re-expressed; S0: 0) | CALIBRATED; an anatomical hypothesis (canal under-labelling), not measured |
| Ovoid radius factor / shift along the tandem | 1.018 / -2.6 mm | CALIBRATED device tolerance |
| k_tie 500 mN/mm per canal point, k_s 600 mN/mm, cell 4 mm, CG caps | | NUMERICAL |
| Bladder, rectum, sigmoid, packing, vaginal wall | lumped into kappa_f / not modelled | ASSUMED (limitation) |

**Data assumptions.**
- The two MRI sessions are on different grids and are not co-registered. Registration frames come from the
  earlier workflow (`snapshot_prior.py`):
  - **BONE**: the key name is historical. It is a peri-organ Mattes-MI frame on non-organ pelvic tissue, whose mask
    is built by excluding the BT organ and applicator labels. No organ enters the metric, but it was not checked
    against bony landmarks.
  - **HR**: HR-CTV-only surface ICP.
  - **UH**: uterus+HR-CTV label fit. This is a leaky ceiling and is never used for validation.
- The BT "IUcanal" label is the tandem itself: it equals applicator minus ovoid, voxel for voxel. The preBT canal
  label is the lumen. This explains the 37 vs 55.9 mm canal-length mismatch (`canal_resolution.py`).
- The flange depth in preBT tissue (d_F) is not identifiable from preBT alone.
- The BT vagina label contains the ovoids and the packing.

## Setup
Host: Windows with Git Bash, python 3.13 (`requirements-host.txt`). SOFA runs in the docker image
`eve-training-fixed` (SOFA v22.12 + SofaPython3, python 3.8). That image is built from the repository-root
`dockerfile`; it is not built by this package. Each run records the image id and the sha256 of every code file.

| Environment variable | Meaning | Default |
|---|---|---|
| `APPSIM_DATA` | read-only image and label folder | `~/Downloads/MRI_GYN` |
| `APPSIM_OUT` | derived-data root | `~/Downloads/MRI_GYN_sim` |
| `APPSIM_PRIOR` | snapshot of the earlier workflow's registration, centre lines and sphere pack | `$APPSIM_OUT/prior` |
| `APPSIM_IMAGE` | docker image | `eve-training-fixed` |

Docker discipline: at most one container of our own, `--cpus 2 --memory 8g`, and `timeout 570` per invocation.
`run_docker.sh` waits while any other container of the image is running, and never touches `rcca_carotid_v3`.

## How to run (Git Bash, in this folder)
```
# M0 inputs (host)
python snapshot_prior.py --src <earlier-workflow scratch dir>   # once; then: python snapshot_prior.py --verify
python prep_inputs.py --accept-deviations runs_def/accepted_deviations.json   # surfaces, ROI, named geometry, poses
python canal_resolution.py                    # canal-length mismatch table
python evaluate.py --e0                       # evaluator gate E0 (reproduces the rigid baselines)
python align.py                               # registration frames + BT device mapped into preBT (poses PB, PBg)

# SOFA runs (container)
bash run_docker.sh G0 gate_tests.py                                      # component/solver gate G0 (stage 1)
bash run_docker.sh R1 run_insertion.py --tag R1_P2_S0 --cfg '{"pose":"P2"}'   # one run
bash run_docker.sh final_A run_insertion.py --batch /app/runs_def/final_A.json   # batch (also final_B, final_C)
#   stage 1-3 definitions: runs_def/stage1_m1.json, stage2_V.json, stage2_posthoc_X.json, stage3_calibration.json
#   (run them with the frozen code, or add "preset": "legacy_v1" to reproduce the stage 1-3 model with this code)

# evaluation (host)
python evaluate.py --freeze <tags> ; python evaluate.py --validate      # stage-2 frozen validation
python posthoc.py                                                       # stage-2 post-hoc rigid controls
python calibrate.py plan|run lhs|refine|run refine|freeze|run controls|heldout|analyze   # stage-3 calibration
python final_eval.py bdev                     # run-independent rigid-on-tandem baseline B_dev(d_F)
python final_eval.py rescore_calib            # calibration argmin with sub-voxel landing terms
python final_eval.py score F0_R1_P2_S0_fixed F1_best ...                # all comparators + gates
python final_eval.py boot F1_best ...         # frame-perturbation noise floor
python final_eval.py report                   # final/verdict.json, final/report.md, figs/final_*.png
python render.py --validate BONE:V4_PB_M2_soft HR:R1_P2_S0              # stage-2 slice figures

# optional GUI: runSofa over X11 (VcXsrv/X410 on :0 with access control off; nothing else of ours running)
bash view_gui.sh F1_best                      # DISPLAY=host.docker.internal:0, APPSIM_CFG=/out/runs/F1_best/cfg.json
```
A run writes `$APPSIM_OUT/runs/<tag>/`:
- `cfg.json` (with provenance);
- `log.jsonl`, one JSON line per step: timings, CG iterations, force residual, multipliers, tie residual,
  penetration, forces, element quality;
- `final.npz`;
- the deformed `surf_{body,uterus,hrctv}.obj` (preBT world mm);
- `summary.json`, which includes the numeric gates.

## Results
All numbers are from runs made with the **fixed code**. Frames: BONE (peri-organ MI, primary) and HR (HR-CTV ICP).
Each line gives Dice / MSD mm / HD95 mm, where HD95 is the pooled symmetric 95th percentile of 6-connected surface
voxels, the registration-stage definition; `final/report.md` adds the max-directed HD95. Every model goes through
the identical route: surface, then rigid map into BT, then `vtkPolyDataToImageStencil` on the BT grid, then
`dice`/`surf_dists` (validated by gate E0). The uterus was unblinded in stage 2, so everything after the
pre-registration table is post hoc.

**Pre-registration status** (criteria below; `final/verdict.json`):

| criterion | result | detail |
|---|---|---|
| M1 (R1_P2_S0, frozen stage-1 code) | FAIL | uterus 0.756 / 2.86 / 5.96 vs B_HR 0.816 / 2.07 / 4.66; tip inside: True; volume < 3%: True |
| M1 re-run with the fixed code (F0, post hoc) | FAIL | uterus 0.757 / 2.85 / 5.96; status converged |
| M2-1 P2-S1 beats B_HR on all 3 uterus metrics (S1 = exploratory stage-3 calibration, redefined after unblinding) | FAIL | F2: uterus 0.748 / 2.95 / 6.28 (stage-3 C3_K_HR: 0.752 / 2.91 / 6.28) |
| M2-2 P1-S1 beats its B0 on uterus MSD and HD95 | PASS | F3 (E_APP frame): sim 2.88 / 6.75 vs B0 7.40 / 15.46 |
| M2-3 P1-S1 tip-to-serosa within +/-3 mm of BT | PASS | error +0.68 mm (sub-voxel) |
| M2-4 mesh convergence |dMSD_uterus| 3 vs 4 mm < 0.3 mm | INCONCLUSIVE | F1_c3 - F1: +0.120 mm; statuses abort_wall_time / converged (the criterion needs two converged states; the metric is also insensitive to the force errors, see mesh study) |

**Calibrated model (stage-3 best parameters, fixed code), primary frame** (BONE frame; status converged; pre-registered gates pass, added gates pass, strict physics gate FAIL)

| model | uterus | HR-CTV (held out of the objective, not blind) | HR-CTV minus uterus |
|---|---|---|---|
| rigid, registration frame (BONE) | 0.262 / 10.27 / 21.37 | 0.363 / 9.51 / 23.16 | 0.211 / 10.02 / 23.47 |
| B_dev, d_F 17.5 mm (canal end at tip; **no run, no BT label**) | 0.881 / 1.45 / 3.57 | 0.671 / 4.37 / 14.11 | 0.610 / 4.32 / 14.50 |
| B_dev, d_F 15.0 mm (chosen on BT HR-CTV) | 0.859 / 1.70 / 3.91 | 0.679 / 4.18 / 13.87 | 0.601 / 4.35 / 14.15 |
| rigid canal-on-tandem (fitted to this run's canal) | 0.735 / 3.06 / 7.49 | 0.777 / 2.92 / 11.08 | 0.715 / 3.13 / 11.21 |
| **simulation F1_best** | 0.732 / 3.08 / 7.20 | 0.711 / 3.86 / 14.20 | 0.635 / 4.05 / 14.42 |

Noise floor (30 rigid frame perturbations applied to both; sim minus comparator MSD, median [5-95 %]): vs B_dev(tip) uterus +1.45 [+0.30, +2.23] mm (comparator better), HR-CTV -0.50 [-0.65, -0.10] mm (sim better); vs rigid canal uterus +0.02 [-0.06, +0.09] mm (interval covers 0; within +/-0.5 mm equivalence).

**M1 tandem-only S0 run (R1 settings, fixed code), HR frame** (HR frame; status converged; pre-registered gates pass, added gates pass, strict physics gate FAIL)

| model | uterus | HR-CTV (held out of the objective, not blind) | HR-CTV minus uterus |
|---|---|---|---|
| rigid, registration frame (HR) | 0.817 / 2.08 / 4.77 | 0.785 / 2.73 / 9.15 | 0.740 / 2.77 / 9.28 |
| B_dev, d_F 17.5 mm (canal end at tip; **no run, no BT label**) | 0.881 / 1.45 / 3.57 | 0.671 / 4.37 / 14.11 | 0.610 / 4.32 / 14.50 |
| B_dev, d_F 15.0 mm (chosen on BT HR-CTV) | 0.859 / 1.70 / 3.91 | 0.679 / 4.18 / 13.87 | 0.601 / 4.35 / 14.15 |
| rigid canal-on-tandem (fitted to this run's canal) | 0.752 / 2.92 / 6.47 | 0.749 / 3.36 / 10.49 | 0.708 / 3.25 / 11.17 |
| **simulation F0_R1_P2_S0_fixed** | 0.757 / 2.85 / 5.96 | 0.688 / 4.12 / 12.43 | 0.631 / 4.05 / 12.65 |

Noise floor (30 rigid frame perturbations applied to both; sim minus comparator MSD, median [5-95 %]): vs B_dev(tip) uterus +1.06 [+0.19, +2.12] mm (comparator better), HR-CTV -0.20 [-0.29, -0.14] mm (sim better; within +/-0.5 mm equivalence); vs rigid canal uterus -0.07 [-0.12, -0.01] mm (sim better; within +/-0.5 mm equivalence).

**Calibrated parameters in the HR frame** (HR frame; status converged; pre-registered gates pass, added gates pass, strict physics gate FAIL)

| model | uterus | HR-CTV (held out of the objective, not blind) | HR-CTV minus uterus |
|---|---|---|---|
| rigid, registration frame (HR) | 0.817 / 2.08 / 4.77 | 0.785 / 2.73 / 9.15 | 0.740 / 2.77 / 9.28 |
| B_dev, d_F 17.5 mm (canal end at tip; **no run, no BT label**) | 0.881 / 1.45 / 3.57 | 0.671 / 4.37 / 14.11 | 0.610 / 4.32 / 14.50 |
| B_dev, d_F 15.0 mm (chosen on BT HR-CTV) | 0.859 / 1.70 / 3.91 | 0.679 / 4.18 / 13.87 | 0.601 / 4.35 / 14.15 |
| rigid canal-on-tandem (fitted to this run's canal) | 0.744 / 3.02 / 6.56 | 0.748 / 3.36 / 10.49 | 0.708 / 3.25 / 11.14 |
| **simulation F2_best_HR** | 0.748 / 2.95 / 6.28 | 0.679 / 4.29 / 13.21 | 0.625 / 4.18 / 13.46 |

Noise floor (30 rigid frame perturbations applied to both; sim minus comparator MSD, median [5-95 %]): vs B_dev(tip) uterus +1.18 [+0.30, +2.22] mm (comparator better), HR-CTV -0.07 [-0.18, +0.18] mm (interval covers 0; within +/-0.5 mm equivalence); vs rigid canal uterus -0.07 [-0.12, -0.00] mm (sim better; within +/-0.5 mm equivalence).

**Calibrated + device-rigid support anchors** (BONE frame; status converged; pre-registered gates pass, added gates pass, strict physics gate pass)

| model | uterus | HR-CTV (held out of the objective, not blind) | HR-CTV minus uterus |
|---|---|---|---|
| rigid, registration frame (BONE) | 0.262 / 10.27 / 21.37 | 0.363 / 9.51 / 23.16 | 0.211 / 10.02 / 23.47 |
| B_dev, d_F 17.5 mm (canal end at tip; **no run, no BT label**) | 0.881 / 1.45 / 3.57 | 0.671 / 4.37 / 14.11 | 0.610 / 4.32 / 14.50 |
| B_dev, d_F 15.0 mm (chosen on BT HR-CTV) | 0.859 / 1.70 / 3.91 | 0.679 / 4.18 / 13.87 | 0.601 / 4.35 / 14.15 |
| rigid canal-on-tandem (fitted to this run's canal) | 0.745 / 3.00 / 6.84 | 0.752 / 3.30 / 10.49 | 0.708 / 3.25 / 11.21 |
| **simulation F1a_best_anchor** | 0.754 / 2.88 / 6.27 | 0.684 / 4.26 / 14.33 | 0.628 / 4.19 / 14.62 |

Noise floor (30 rigid frame perturbations applied to both; sim minus comparator MSD, median [5-95 %]): vs B_dev(tip) uterus +1.05 [+0.22, +2.11] mm (comparator better), HR-CTV -0.06 [-0.11, +0.04] mm (interval covers 0; within +/-0.5 mm equivalence); vs rigid canal uterus -0.14 [-0.19, -0.07] mm (sim better; within +/-0.5 mm equivalence).

**What the numbers say.**
- **Against B_dev (no simulation run).** Placing the undeformed preBT anatomy rigidly on the real tandem, with the
  canal end at the rod tip (d_F 17.5 mm), matches the BT uterus better than every simulation. The gap is +1.1 to
  +1.5 mm MSD, and the 5-95 % interval over the frame perturbations excludes 0 in every run.
- **HR-CTV.** The simulations are within the pre-declared +/-0.5 mm equivalence margin of B_dev, or slightly better
  (F1: -0.50 mm). The rigid canal-on-tandem control is better still.
- **Against the rigid canal-on-tandem control.** Each simulation is within 0.15 mm MSD of it on the uterus
  (equivalent). The deformation adds nothing measurable beyond the rigid motion the ties impose.
- **The registration-frame rigid baseline.** In the peri-organ frame it is beaten by a wide margin. This only shows
  that the device position is informative (the organs moved ~24 mm relative to that frame); it is not evidence for
  the mechanics.
- **Two HR-CTV caveats.** HR-CTV is not in the calibration objective, but the analysts saw peri-organ-frame HR-CTV
  numbers in stage 2, and it overlaps the in-sample uterus by 20-30 %. The HR-CTV-minus-uterus column is the less
  coupled check.
- **Which calibrated parameter drives the gap.** The calibrated canal hypothesis (delta_fund 7 mm, equivalent to
  the frozen tip margin of -6.7 mm) places the organ about 7 mm lower on the tandem than B_dev's best. B_dev is
  best at d_F 17-18 mm. The calibration traded uterus shape for the landing term.

**Mesh study (calibrated model at 5 / 4 / 3 mm cells).** Status converged only at 4 mm:
- at 5 mm a slow sliding mode grows (step ratio 1.02) and the hold never converges;
- at 3 mm the drift decays too slowly (ratio 0.96) and the run hits the 540 s budget, with about 0.6 mm estimated
  drift left.

On these states:
- Uterus MSD is 3.14 / 3.08 / 3.20 mm, non-monotone and within 0.12 mm. The pre-registered M2-4 criterion is
  therefore INCONCLUSIVE: it needs two converged states.
- The axial tip force (625 / 753 / 771 mN) and the ovoid force (2.48 / 2.39 / 2.37 N) change little.
- The lateral tie force sum (22.5 / 18.8 / 8.4 N) is not mesh-converged, so it is not a device load.
- The Richardson orders computed on these unconverged states are not meaningful.

The fixed code removes the tie over-constraint seen by the review: the largest AL multiplier is 1.5-3.1 mm, against
6 mm (saturated) before.

**Performance** (docker `--cpus 2` on a host whose CPUs are saturated by another training container; medians).
- Insertion steps take 120-155 ms at 4 mm cells: CG is capped at 100 iterations and always reaches the cap.
- Hold steps take 2.0-2.8 s: CG needs 540-580 iterations to reach 1e-10.
- A full 4 mm run takes 70-100 s (25-40 hold steps) and initialisation 1-2 s.
- At 3 mm, insertion steps take 365 ms and hold steps 7.1 s, which does not fit one 570 s container.
- The earlier "~250 ms/step" figures were bought by an unconverged hold.

**Reproducibility.** The frozen stage 1-3 code re-run on the frozen R1 and V1 configurations reproduces them
bit-identically (`final/A1_repro.json`). The finisher code with `"preset": "legacy_v1"` also reproduces R1
bit-identically (run `D0_R1_legacy`). Every finisher run records the sha256 of all container code and the docker
image id.

Figures (local): `figs/final_bdev_curve.png` (B_dev vs flange depth), `figs/final_summary.png` (uterus/HR-CTV MSD
per comparator with noise-floor brackets), `figs/final_slices.png` (BT MRI planes through the real tandem; BT
labels vs rigid frame, B_dev and simulations). Full tables: `final/report.md`, `final/verdict.json`.

**Status of the review findings.**
- **Fixed:**
  - hold convergence: rule v2 (3 consecutive steps, force residual, drift estimate, frozen multipliers, CG
    iterations logged, hold CG to tolerance);
  - tie over-constraint: one tie per cell, a fillet at the L_end kink, and a multiplier-saturation gate;
  - tip overrun: an explicit delta_fund canal hypothesis replaces the negative margin, plus a rod-in-lumen gate;
  - supports: a device-rigid support-anchor option (F1a);
  - B_dev as the run-independent primary comparator; the frame-perturbation noise floor with an equivalence margin;
  - sub-voxel landing and the HR-CTV-minus-uterus check; RIGID_CANAL roll-prior sensitivity (eps 0, 0.01, 0.1 in
    `final/scores`);
  - labelling: the pre-registration status table with hashed criteria; BONE relabelled as a peri-organ frame;
  - provenance: code and image hashes per run; amendments for post-freeze edits; environment-variable paths;
    snapshotted prior inputs; run definitions in `runs_def/`; a single gate definition (`gates.py`).
- **Not fixed (see Limitations):**
  - the voxel-hull body (the non-uniform FEM failed);
  - the compressible linear material (the strict physics gate fails where the ovoids press);
  - isotropic supports that are not on the true surface;
  - forces are not mesh-converged, and no 3-level convergence was reached;
  - the peri-organ frame was not verified with bony landmarks.


## Limitations
- **One patient.** There is no generalisation claim, and the calibrated values are fits to this patient's uterus,
  not predictions.
- **E cannot be fitted.** Loading is displacement-driven, so only kappa/E, k/(E h) and nu set the shape. Forces
  scale with the assumed E (30 kPa) and are **unvalidated**.
- **Forces are not mesh-converged.** The lateral tie force sum is dominated by penalty over-constraint along the
  canal. It is not a device load. See the mesh study.
- **Voxel-hull body.** `HexahedronFEMForceField` gives every partially filled boundary cell full stiffness. The
  mechanical body is the voxel hull: 119 cc at 4 mm and 111 cc at 3 mm, for an 83 cc tissue surface.
  - The fill-weighted `NonUniformHexahedronFEMForceFieldAndMass` (`fe_impl="nonuniform"`) was built and tested but
    failed: a nearly empty fundal cell collapses because the supports act on grid nodes outside the tissue.
  - It needs supports on the true surface, which is not built.
- **Material law.** The law is linear, compressible and corotational. Where the ovoids press on the portio,
  element volumes drop by up to ~45-50%, which near-incompressible tissue cannot do. The strict physics gate
  (every hexa volume ratio in [0.85, 1.15]) fails there. Hyperelastic nearly-incompressible tets are needed.
- **Not modelled:**
  - vaginal packing, the vaginal wall (the preBT vagina label is a collapsed solid tube), bladder, rectum and
    sigmoid filling;
  - cervical shortening and HR-CTV regression between the sessions (the HR-CTV shrinks 57 to 47 cc). These cap
    every metric.
- **Supports are isotropic.** They are zero-length springs anchored at the rest positions (or at a device-rigid
  placement). A normal-only foundation on the true surface is not built.
- **The tandem-canal interaction is a frictionless tie**, not frictional contact (M3). The canal hypothesis
  (delta_fund) is fitted, and the rod tip is pushed onto the labelled or extended canal end.
- **The HR frame uses the HR-CTV**, so HR-CTV is not held out there.
- **The BONE-frame HR-CTV is not a clean hold-out.** It was not in the objective, but the analysts saw it in
  stage 2, and it overlaps the in-sample uterus by 20-30%.
- **The pre-registered S1 was never run as designed.** The stage-3 calibration is exploratory and was redefined
  after unblinding.
- **Performance.** The insertion runs CG capped at 100 iterations, which it always reaches. Every step re-solves
  the full residual, and the hold then solves to tolerance. The hold costs about 2 s/step on the contended host.
- **Teardown segfault.** SofaPython3 v22.12 segfaults at interpreter exit once a Python ForceField has existed. The
  scripts close all outputs and call `os._exit(0)`.

## Next steps
**(a) Better physics.**
1. Build M3: a rigid-contact applicator. Use a FreeMotionAnimationLoop with GenericConstraintSolver and
   LinearSolverConstraintCorrection, a tandem sphere chain and an ovoid mesh against a tet lumen mesh. Use it to
   validate the frictionless-tie abstraction, including mu = 0 path independence.
2. Switch to a hyperelastic, nearly incompressible tet body (`TetrahedronHyperelasticityFEMForceField` on a
   `MeshTetraStuffing` mesh of the true surface), with supports on the true surface (normal-only foundation).
   Measure the fill/volume of the mechanical body against the label.
3. Build M4: organs at risk (bladder, rectum, sigmoid) as a multi-material block, with filling and packing as
   explicit load cases and HR-CTV regression as an eigenstrain. Without these, the mechanics cannot beat rigid
   placement.
4. A measured insertion force is needed to identify E.

**(b) A cohort.** Everything here is n = 1.
1. Run the pipeline (`prep_inputs` -> `align` -> runs -> `final_eval`) per patient, with `APPSIM_DATA` and
   `APPSIM_OUT` per case.
2. Freeze the model on a training set.
3. Report B_dev and the mechanics on held-out patients, with the frame-perturbation noise floor and a pre-declared
   equivalence margin.
4. The canal hypothesis (d_F, delta_fund) and the support ratios need a population prior.

**(c) RL environment (eve_rl).**
1. `controller.ApplicatorController.step_action(d_depth_mm, d_angles_deg)` already moves the applicator pose per
   step.
2. Wrap `scene.build_scene` + the controller in an `eve_rl` environment. Observations: tip force, tie residuals,
   tip-to-serosa, max stretch, contact penetration. Reward: reach the target pose while penalising stretch,
   penetration and tip force.
3. Train with the repo's SAC/PER stack, following the headless SofaPython3 pattern of
   `eve/eve/intervention/simulation/sofabeamadapter.py`.
4. The per-step cost needs to fall first. Steps take ~0.13 s in insertion and ~2 s in the hold; a direct solver
   needs an assembled tie stiffness (`tie_impl="rssff"`).

## Files
| File | Runs on | Role |
|---|---|---|
| `config.py` | both | defaults with units, presets (`legacy_v1`), paths from environment variables |
| `geom.py` | both | pure numpy geometry: frames, path, projections, trilinear weights, hexa volume and stretch |
| `gates.py` | both | the single definition of the numeric gates (pre-registered, added after review, physics) |
| `snapshot_prior.py` | host | copies the earlier workflow's inputs into `$APPSIM_OUT/prior` with a sha256 manifest |
| `prep_inputs.py`, `canal_resolution.py`, `prep_m2.py` | host | M0 inputs, canal-length table, `pre_bodyvag.obj` |
| `scene.py`, `controller.py`, `ties.py` | container | SOFA graph, applicator controller (schedule, ties, contact, stop rules, `step_action`), tie ForceField |
| `run_insertion.py`, `calib_run.py`, `gate_tests.py`, `probe_pyff.py` | container | runs and batches, calibration alias, G0 gate, ForceField probe |
| `run_docker.sh`, `view_gui.sh` | host | guarded container launch; optional runSofa GUI |
| `align.py`, `evaluate.py`, `posthoc.py`, `calibrate.py` | host | frames, E0 and stage-2 validation, post-hoc controls, stage-3 calibration |
| `final_eval.py`, `final_report.py` | host | finisher comparators (B_dev, RIGID_CANAL), noise floor, report and figures |
| `render.py`, `report.py` | host | stage-2 figures; stage-1 report (superseded) |
| `runs_def/*.json` | - | parameter-only run definitions; accepted named-geometry deviations |

## Pre-registered pass criteria (copied from the design spec before any unsealing)
**Numeric** (every run):
- the Phase A control passes;
- the hold converges;
- the minimum hexa volume ratio is > 0.2;
- there are no NaNs;
- the tie and penetration criteria are met;
- piercing = 0.

**M1 validation** (P2-S0, tandem only):
- the uterus MSD < 2.07 mm, HD95 < 4.66 mm and Dice > 0.816 (beats B_HR);
- tip-to-serosa > 0;
- the uterus volume changes by < 3%.

**M2 validation** (S1 frozen):
- P2-S1 beats B_HR on all 3 uterus metrics;
- P1-S1 beats its B0 on uterus MSD and HD95;
- P1-S1 tip-to-serosa is within +/-3 mm of BT;
- mesh convergence: |ΔMSD_uterus| between 3 and 4 mm cells < 0.3 mm.

The criteria text is hashed into `final/verdict.json`.
