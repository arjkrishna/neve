# RCCA mesh-generator fix — diagnosis, strategy, code pointers

**Status: PARKED mid-investigation (2026-07-30) to prioritise paper evaluation.**
Diagnosis is measured and solid; the mechanism hypothesis is reasoned from code but the
confirming ablation had NOT yet run when this was parked. Resume at §5.

---

## 1. The problem (measured, §7 of `saved/p2a_deep_dive/GEOMETRIC_WALL_VERIFIED.md`)

`RCCAVariedFromMesh` rebuilds the whole tree surface from centerlines every generation.
The rebuilt surface is systematically narrower than the original segmented surface, and
a large fraction of generated anatomies are impassable to the 0.36 mm guidewire.

Clearance = distance from each centerline station to the nearest point **on the triangle
surface** (dense-sampled, NOT nearest-vertex — vertices sit outside facets and overstate
clearance badly on a 6 mm-triangle mesh).

| mesh | median clearance | p05 | stations < wire radius |
|---|---|---|---|
| **ORIGINAL `vessel_architecture_collision.obj`** | **2.11 mm** | 1.15 | **0/235 — passable** |
| zeroed-amplitude regeneration (same centerlines) | 1.23 mm | 0.40 | 2/235, first block raw 120.4 mm |
| 6 sampled varied anatomies | 1.09–1.33 | 0.22–0.63 | **4 of 6 blocked** |

Independent corroboration: geometric first-block at raw 120.4 mm + the raw→proj offset
(~34 mm) = proj_s ≈ 154.0; three controllers were measured arresting at **153.4 mm**.

**Consequence:** every run since Gen-4 trained on anatomies that dead-end before the
siphon, i.e. the hard skill was absent from the curriculum. Fixing the evaluation surface
alone moved the real patient from 35.7% → 75.5%.

## 2. The pipeline, and why it erodes

`eve/eve/intervention/vesseltree/util/meshing.py::generate_mesh`:

```python
cube = create_empty_voxel_cube_from_branches(branches, [0.6, 0.6, 0.9])  # anisotropic
for _ in range(5): cube.add_padding_layer_all_sides()
for b in branches:
    cube.mark_centerline_in_array(b.coordinates, b.radii, 1, radius_padding=0)
cube.gaussian_smooth(1)        # sigma is in VOXELS
cube.gaussian_smooth(1)        # ...applied TWICE -> effective sigma ~1.41 vox
mesh = get_surface_mesh(cube, gradient_direction)   # level=None
mesh = mesh.decimate(0.99)     # keeps 1% of triangles
```

**Primary hypothesis (reasoned from code, NOT yet confirmed by ablation).**
`get_surface_mesh` passes `level=None` to `skimage.measure.marching_cubes`, which then
uses `(min+max)/2` over the **whole array**. The global max is set by the **aorta**
(r ≈ 5.8 mm ≈ 10 voxels across) whose peak survives blurring at ≈1.0. But a distal
vessel at r ≈ 1.25 mm is only ~2 voxels in radius, so double-blurring with σ ≈ 1.41
voxels drops its **peak well below 1**. Thresholding a low-amplitude thin tube at a
level calibrated by a wide vessel erodes the thin one. Predicts exactly the measured
signature: proximal clearance intact, distal eroded.

Contributing factors:
- **σ is in voxel units** and the grid is anisotropic → σ ≈ 0.85 mm in x/y but 1.27 mm
  in z. Erosion is direction-dependent.
- **Curvature shrinkage**: for a cylinder, the 0.5-isosurface moves inward by ≈ σ²/(2r).
  With σ ≈ 0.85 mm, r = 1.25 mm → ≈ 0.29 mm ≈ 23% of radius.
- **`radius_padding=0`** — nothing compensates for either effect.
- **`decimate(0.99)`** leaves ~6.5 mm triangles. NOTE: the original `.obj` is equally
  coarse (6.28 mm) and is passable, so decimation is **not** the primary cause — but it
  adds faceting noise on top (measured excursions −56%…+28%).

**Explicitly falsified** (do not re-propose): systematic 25–43% erosion measured by
nearest-VERTEX (artifact); collision-chord discretisation (`collis_edges_per_mm_straight
=0.1`) — the historical fixed-mesh run used the same devices and traversed the siphon;
mesh coarseness per se (original is equally coarse).

## 3. The three acceptance criteria (user-specified)

1. **Varied** — believed already satisfied by `perturb_rcca`; re-verify after any change.
2. **Passable** — 0 stations with clearance < wire radius, on ~every generated anatomy.
3. **As hard as the real one** — clearance distribution matching the original `.obj`, or
   *slightly tighter*. **Must not be easier.** Concretely: median and p05 clearance ≤
   original (2.11 / 1.15 mm), and curvature not reduced.

Criterion 3 is why "just inflate the radii" is not an acceptable fix on its own — it
would buy passability by making the task easier.

## 4. Candidate fixes to test

| lever | rationale | risk |
|---|---|---|
| `level=0.5` explicit | removes the global-max coupling — the thin-tube killer | may leave staircase artifacts |
| smoothing ×1 instead of ×2 | halves the blur that drops thin-tube peaks | rougher surface |
| σ 1.0 → 0.5 voxels | erosion scales ~σ² | rougher surface |
| isotropic voxels (0.4/0.3 mm) | removes z-anisotropy, thin tubes span more voxels | memory ∝ 1/spacing³ — see §6 |
| `radius_padding` +0.15–0.3 mm | compensates residual erosion | **must not overshoot** (criterion 3) |
| `decimate` 0.99 → 0.5 | reduces faceting noise | more triangles → slower SOFA collision |

Preferred direction: fix the **isolevel** and the **smoothing** first (they are free),
use `radius_padding` only to close a small residual gap, and treat voxel refinement as
the fallback because of memory.

## 5. RESUME HERE — the ablation

`monitoring/mesh_ablation.py` (written, **not yet run**) builds the zero-amplitude tree,
measures the ORIGINAL `.obj` as the target, then rebuilds the mesh under 13 settings and
reports for each: cells, median triangle edge, median/p05/min clearance, distal clearance
vs stated radii, blocked-station count, grid size and runtime. It skips any grid over
700M voxels.

Run it:
```bash
docker run --rm -m 24g \
  -v "D:\Arjun\workspace\neve\eve:/opt/eve_training/eve" \
  -v "D:\Arjun\workspace\neve\eve_bench:/opt/eve_training/eve_bench" \
  -v "D:\Arjun\workspace\neve\monitoring\mesh_ablation.py:/tmp/a.py" \
  eve-training-fixed python3 /tmp/a.py
```

**Read the result as:** which single lever moves DISTAL clearance from ~1.2 mm back to
~2.1 mm without exceeding it. If `level=0.5` alone does it, the hypothesis is confirmed
and the fix is one line.

## 6. Feasibility note on voxel refinement

Memory scales as 1/(dx·dy·dz). Current [0.6,0.6,0.9] over the full arterial tree. Going
isotropic 0.3 mm is ~18× the voxels — likely tens of GB and infeasible for a per-episode
regeneration (the class docstring already flags this: *"a per-generation full-tree
re-mesh is not free … or restrict meshing to the navigated subtree"*). If refinement
proves necessary, mesh **only the RCCA subtree** finely and keep the rest coarse.

## 7. Verification protocol (must pass all three)

1. **Identity check** — zero-amplitude generation must reproduce the ORIGINAL surface's
   clearance profile (median ≈2.11, p05 ≈1.15, 0 blocked). Then re-run the real-patient
   eval *through the generator* (not the pinned `.obj`) and it must reproduce **75.5%**
   for v1bp ckpt2002292 / **63.3%** for v1b ckpt3259127. This is the decisive test: same
   centerlines, same everything, generator instead of original mesh.
2. **Passability** — ≥50 varied anatomies, 0 blocked stations each (`mesh_clearance.py`).
3. **Difficulty** — clearance distribution ≤ original at median and p05; curvature
   statistics unchanged or tighter. Verify variation is retained (`geometry_hash` spread).

Only after all three pass should any training run be launched.

## 8. Code pointers

| file | role |
|---|---|
| `monitoring/mesh_ablation.py` | **the ablation — RUN THIS FIRST** (§5) |
| `monitoring/mesh_clearance.py` | passability gate: clearance vs wire radius, no controller |
| `monitoring/measure_mesh_quality.py` | earlier nearest-vertex probe — **superseded, biased high** |
| `monitoring/verify_wall.py` | per-episode arrest station from eval logs |
| `monitoring/verify_walled.py` | per-anatomy walled/scattered classification |
| `eve/.../util/meshing.py` | `generate_mesh` — the pipeline to change |
| `eve/.../util/voxelcube.py` | `mark_centerline_in_array` (radius_padding), `gaussian_smooth` |
| `eve/.../rccavariedfrommesh.py` | `perturb_rcca` (variation), `_generate`, `mesh_path` |
| `training _scripts/eval_anatomies.py` | `--real_patient_anatomy` now pins the original surface |
