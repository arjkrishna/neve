# Hybrid pelvis simulator, Stage 1

Module sections are appended by the agents that build them (contract: `CONTRACT.md`). Code only lives here; every
derived file is under `~/Downloads/MRI_GYN_sim/hybrid/` (local, possibly patient-derived, never
committed).

## MESHING (`mesh_bodies.py`, CONTRACT section 1)

Six voxel-exclusive bodies from the preBT labels, closed surfaces, TetGen tetrahedra, and the named node sets.
Frame: preBT world RAS mm (nibabel affine, x=R, y=A, z=S). Units mm, volumes cc, angles deg. Every JSON states both.

### Commands (Git Bash, in `applicator_sim/`)
```bash
PK=<scratch>/pk313                                     # any writable dir outside the repo
python -m pip install --target "$PK" tetgen pyacvd     # once: tetgen 0.8.4 + pyacvd (pull pyvista 0.49 + vtk into $PK)
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
python   hybrid/mesh_bodies.py build --pk "$PK"        # py3.13 host: labels -> surfaces -> tets -> node sets (16 s)
py -3.11 hybrid/mesh_bodies.py render                  # pyvista off-screen -> hybrid/figs/mesh_{all_v1,all_v2,nodesets}.png
bash run_docker.sh MESHCHK hybrid/mesh_bodies.py sofacheck   # optional: SOFA v22.12 MeshVTKLoader reads every tets.vtk
```
`--pk` is appended to `sys.path` (the interpreter's own numpy/scipy/vtk stay first); `APPSIM_PK` is the env
equivalent. `--edge body=mm` overrides a target edge, `--bodies` builds a subset. Outputs: `hybrid/meshes/bodies.json`
(index), `hybrid/meshes/<body>/{surface.obj, tets.vtk, meta.json}`, `hybrid/logs/mesh_bodies.json`,
`hybrid/logs/mesh_sofacheck.json`.

### Pipeline and decisions (all parameters in `mesh_bodies.CFG`, copied into `bodies.json["cfg"]`)
1. **Bodies.** One body per voxel by priority `corpus > cervix > vagina > bladder > rectum > sigmoid`
   (corpus = uterus label, cervix = HR-CTV minus uterus), then the largest **6-connected** component (drops the rectum's
   diagonal-only strays, 5 + 1 voxels). The priority rule removes 11.2 cc of HR-CTV under the uterus and **5.3 cc of the
   vagina label under the HR-CTV** (the fornices/upper vagina belong to the cervix body; the vagina body is 5.5 cc, not
   10.8 cc). All labels are hole-free.
2. **Surfaces.** `prep_inputs.surface` (marching cubes in world mm, windowed-sinc 20 it / pass-band 0.1) without
   decimation, then an **isotropic ACVD remesh** (`pyacvd`) at the target edge instead of quadric decimation: TetGen
   with `-Y` keeps the surface triangles as tet faces, so uniform 3.5 mm triangles are what makes 3.5 mm tets with
   bounded dihedrals; quadric decimation leaves long thin triangles that become slivers. Consequence: ~1.2-2.4k
   triangles per body, below the contract's "3-6k" (at 3.5-4.5 mm edges these bodies do not have the area for more;
   the edge target governs the FEM). Outward orientation, then a **uniform normal offset** (capped at 0.3 mm) restores
   the label volume lost to smoothing + remeshing (0.07-0.14 mm; the cap binds only for the vagina).
   **Thin-sheet fallback:** if the raw remesh is never manifold, the body mask is opened (6-connected, radius 1 voxel,
   drops sheets <= 2 voxels thick, largest component) and the remesh repeated; the volume target stays the un-opened
   label. Needed for the cervix (0.50 cc of HR-CTV shell around the corpus), the vagina (0.49 cc of collapsed-lumen
   wings) and the rectum (0.26 cc). Gates: 0 open / non-manifold edges, positive volume within 2 % of the label.
3. **Tetrahedra: TetGen route** (`tetgen` 0.8.4 wheel, python 3.13; the SOFA `MeshTetraStuffing` fallback was not
   needed). Switch equivalent `-pq1.4/10 -Y -a(1.5 V_regular(edge)) -O7/10`, with `fixedvolume=True` (without it the
   wrapper ignores `maxvolume`) and `optmaxdihedral=165`. TetGen's dihedral bound is a request, not a guarantee (it
   delivers 8.2-9.1 deg with 4-8 slivers per body), so `fix_slivers` follows: greedy local relocation of the offending
   tets' nodes (interior first; boundary nodes at most 0.3 mm from their input position) maximising the star's minimum
   dihedral, with boundary wedge/cap deletion as a last resort (never triggered here). Result: every body > 11 deg,
   sliver fraction 0, volume change < 0.01 cc. If `-Y` still failed the bound, the build retries without `-Y`
   (boundary Steiner points) and keeps the better mesh (not needed). `surface.obj` is the boundary of `tets.vtk`
   (`meta.surface_obj_vertex_to_tet_node` maps its vertices to tet points). `tets.vtk` is a legacy ASCII unstructured
   grid (classic `CELLS` layout) read by both VTK and SOFA's `MeshVTKLoader`.
4. **Node sets** (indices into `tets.vtk` points; definitions and sizes in `meta.json`):
   `surface_nodes` (boundary of the tet mesh); `cervix.interface_corpus` / `corpus.interface_cervix` (surface nodes
   within 1.5 mm of the other body's surface, via `vtkImplicitPolyDataDistance`); `cervix.canal` (all nodes within
   3 mm of `inputs/canal.npz` `pts[:i_internal_os+1]`, external to internal os); `cervix.lateral_os_level` (surface
   nodes with |z - z_os| <= 6 mm and |x - x_os| > 10 mm, "level" read as the axial level of the internal os;
   both sides reported); `vagina.fixed_inferior` (all nodes within 15 mm of the inferior end along the vagina body's
   principal axis); `vagina.apex` (surface nodes within 3 mm of the cervix surface); `bladder.anterior_support`
   (surface nodes in the outer third of the projection on (0,1,1)/sqrt2 whose outward normal faces it);
   `rectum.posterior_support` (surface nodes whose outward normal is within 60 deg of -y: the posterior third of the
   circumference along the whole tube, since a global-y third of this oblique tube would only select its superior
   segment); `rectum.fixed_ends`, `sigmoid.fixed_ends` (both cut ends: nodes within 4 mm of the label's extreme slice
   measured along the local tube direction of that end; the rectosigmoid junction is not a slice but an oblique
   tapered contact of the two labels, so the end within 15 mm of the other body is taken as the nodes within 4 mm of
   the other body's surface, also exposed as `rectum.junction_sigmoid` / `sigmoid.junction_rectum` in case the scene
   prefers to attach them to each other rather than fix them).
5. **Figures** (`hybrid/figs/`): `mesh_all_v1.png` (anterior-right-superior oblique), `mesh_all_v2.png` (left lateral),
   `mesh_nodesets.png` (2x3 panels, node sets coloured on each body, canal polyline and internal os on cervix/corpus).
   Viewed: bodies sit as expected (bladder anterior, rectum posterior, sigmoid superior, vagina anterior to the rectum
   below the cervix), all sets are non-empty and on the intended side (both lateral rings, canal nodes along the
   polyline below the os, posterior wall of the whole rectum, both ends of rectum and sigmoid).

### Quality table (label = after priority + largest component)
| body | prio | label raw -> body (cc) | mesh (cc) / vs body | thin fix, offset | tris / verts | nodes / tets | edge target / surf / tet (mm) | min dihedral TetGen -> final | slivers <10 |
|---|---|---|---|---|---|---|---|---|---|
| corpus | 1 | 37.52 -> 37.52 | 37.52 / +0.01 % | raw, +0.065 mm | 1180 / 592 | 1779 / 9001 | 3.5 / 3.50 / 3.46 | 8.17 -> 11.36 | 0 |
| cervix | 2 | 56.99 -> 45.77 (-11.21 under uterus) | 45.76 / -0.00 % | opened -0.50 cc, +0.142 mm | 1580 / 792 | 2188 / 10861 | 3.5 / 3.54 / 3.47 | 8.95 -> 11.60 | 0 |
| vagina | 3 | 10.78 -> 5.49 (-5.29 under HR-CTV) | 5.46 / -0.48 % | opened -0.49 cc, +0.300 mm (cap) | 1416 / 710 | 1503 / 6887 | 2.0 / 1.96 / 2.00 | 8.92 -> 11.37 | 0 |
| bladder | 4 | 251.28 -> 251.28 | 251.28 / -0.00 % | raw, +0.069 mm | 2356 / 1180 | 5234 / 28425 | 4.5 / 4.54 / 4.43 | 9.04 -> 11.65 | 0 |
| rectum | 5 | 38.97 -> 38.96 (-0.01 strays) | 38.96 / -0.01 % | opened -0.26 cc, +0.123 mm | 1616 / 810 | 1968 / 9409 | 3.5 / 3.36 / 3.45 | 8.25 -> 11.14 | 0 |
| sigmoid | 6 | 40.41 -> 40.41 | 40.40 / -0.01 % | raw, +0.123 mm | 1972 / 988 | 2039 / 9186 | 3.5 / 3.54 / 3.53 | 9.12 -> 11.13 | 0 |

Total 14 711 nodes, 73 769 tets. All surfaces: 0 open edges, 0 non-manifold edges/vertices, outward, tet volume =
surface volume. Tets: 0 inverted, every dihedral > 11 deg.

Node-set sizes: corpus surface 592, interface_cervix 129 | cervix surface 792, interface_corpus 127, canal 38,
lateral_os_level 168 (73 left / 95 right of the os) | vagina surface 710, fixed_inferior 435, apex 98 | bladder
surface 1180, anterior_support 377 | rectum surface 810, posterior_support 237, fixed_ends 103 (28 anal slab + 75
junction), junction_sigmoid 75 | sigmoid surface 988, fixed_ends 82 (61 junction + 21 distal slab), junction_rectum 61.

### Caveats (for the scene and evaluation modules)
- **Rest-state overlaps at shared interfaces** (`bodies.json["pairs"]`, signed distance of one body's surface nodes to
  the other): corpus|cervix -0.99 mm (25 nodes deeper than 0.5 mm, 244 within 1 mm), cervix|vagina -0.87 mm (13),
  rectum|sigmoid -0.57 mm (3), cervix|bladder -0.20 mm (0). Independent smoothing of voxel-exclusive labels makes the
  two surfaces cross by up to one voxel; with contactDistance 0.5 mm these pairs would start in contact. Disable
  collision for corpus|cervix (attached through `interface_corpus`) and consider excluding `vagina.apex` and the two
  junction sets from contact, or attach them.
- The vagina body is the least faithful: 5.5 cc after the priority rule, ~3 mm thick, opened by 0.49 cc and inflated
  by the full 0.3 mm cap to recover the label volume; 1-2 tets across the thickness at 2 mm edges.
- The cervix body includes the tumour and the parametrial extension (HR-CTV), so `lateral_os_level` on the
  right side lies on that lateral extension, ~20 mm from the canal.
- `cervix.canal` is sparse (38 nodes over ~34 mm of canal at 3.5 mm edges); a tie needs no more, but a finer
  cervix (`--edge cervix=2.5`) triples it if the scene wants a denser sleeve.
- The volume offset and the sliver fix move surface nodes by <= 0.3 mm; `surface.obj` therefore differs from the
  ACVD remesh by that much and is not the marching-cubes surface of `inputs/pre_*.obj`.
- ACVD clustering is deterministic here (identical counts across runs) but is not guaranteed to be across pyacvd
  versions; the remesh history (cluster counts, manifold checks) is in `meta.json["surface"]["remesh_history"]`.

## APPLICATOR + POSE RULE

Module `applicator_venezia.py` (host, python 3.13; `py -3.11` only for the pyvista render). Implements CONTRACT sections 2 and 3: the parametric Venezia-type applicator (surfaces in the applicator frame, needle-channel axes), the fit of the two-cap ovoid model to the BT ovoid label, the kinematic pose rule v1 with its insertion path, and the rule's score against the real BT tandem.

Frames/units: applicator frame = origin at the flange centre (tube exit from the caps), z = tube axis flange->tip, y = anterior in the sagittal plane, x = y x z (patient right); pose.json in preBT world RAS mm. All JSON state frame and units.

### Commands
```
cd D:/neve/.claude/worktrees/sofa_applicator/applicator_sim/hybrid
set MPLBACKEND=Agg                 (PowerShell: $env:MPLBACKEND='Agg')
set PYTHONDONTWRITEBYTECODE=1      (PowerShell: $env:PYTHONDONTWRITEBYTECODE='1')
python applicator_venezia.py                 # meshes + ovoid fit + pose rule + validation + figures + README section
python applicator_venezia.py --no-fit        # reuse the stored ovoid fit
py -3.11 applicator_venezia.py --render3d    # figs/applicator_3d.png
```

### Outputs
`MRI_GYN_sim/hybrid/applicator/{tube.obj, shaft.obj, ovoid_L.obj, ovoid_R.obj, applicator.json, pose.json}`, `MRI_GYN_sim/hybrid/figs/{ovoid_fit.png, pose_rule_BT.png, pose_rule_preBT.png, pose_rule_path.png, applicator_3d.png}`, `MRI_GYN_sim/hybrid/logs/applicator_venezia.json`. Surfaces: tube 1152 tris, closed=True, vol 901 mm3 (-1.2 % vs reference); shaft 1440 tris, closed=True, vol 443 mm3 (-1.1 % vs reference); ovoid_L 1500 tris, closed=True, vol 10934 mm3 (-0.1 % vs reference); ovoid_R 1500 tris, closed=True, vol 11092 mm3 (-0.1 % vs reference).

### Parameters (applicator.json `params`)

| parameter | value | unit | provenance |
|---|---|---|---|
| `L_iu_mm` | 61.8 | mm | MEASURED (BT applicator label, flange->tip; inputs/applicator.json) |
| `r_tandem_mm` | 2.18 | mm | MEASURED (BT rod FWHM 4.36 mm / 2; inputs/applicator.json) |
| `tip_hemispherical` | True | - | ASSUMED (rounded tandem tip) |
| `shaft_arc_len_mm` | 30 | mm | CONTRACT 2 (30 mm arc below the flange) |
| `angle_deg` | 24 | deg | MEASURED 22-25 deg tube/shaft junction angle (CONTRACT 2), default 24 |
| `r_shaft_mm` | 2.18 | mm | ASSUMED = r_tandem (one continuous tube through the caps) |
| `ovoid_diam_mm` | 39.4 | mm | FITTED to the BT ovoid label (AP = LR diameter of the cap assembly; initial 40) |
| `ovoid_height_mm` | 22.2 | mm | MEASURED assembly extent along z (CONTRACT 2) |
| `ovoid_dome_mm` | 11.4 | mm | FITTED dome height of each cap (= height -> pure half-ellipsoid cap; initial 8) |
| `ovoid_offset_x_mm` | 0 | mm | FITTED lateral offset of each cap's centre from the tube axis (0 = the two halves of one body) |
| `ovoid_dz_mm` | 1.36 | mm | FITTED cap apex level relative to the flange along z |
| `ovoid_tilt_deg` | -4.13 | deg | FITTED AP tilt of the cap assembly about x (positive = apex tilted anteriorly) |
| `ovoid_gap_mm` | 0.5 | mm | ASSUMED slot between the two caps at the mid-sagittal plane (the tube passes through the caps) |
| `channel_radius_mm` | 17 | mm | CONTRACT 2 (parallel needle channels on a 17 mm radius) |
| `n_parallel_per_ovoid` | 5 | - | CONTRACT 2 |
| `n_oblique_per_ovoid` | 2 | - | CONTRACT 2 |
| `oblique_deg` | 15 | deg | CONTRACT 2 (oblique channels at 15 deg to the tube axis, tilted laterally outward) |
| `oblique_radius_mm` | 12 | mm | ASSUMED entry radius of the oblique channels at the cap bottom |
| `needle_r_mm` | 1 | mm | CONTRACT 2 |
| `parallel_azimuth_deg` | [20.0, 55.0, 90.0, 125.0, 160.0] | deg | ASSUMED layout: azimuth from +y (anterior) toward the cap's side |
| `oblique_azimuth_deg` | [45.0, 135.0] | deg | ASSUMED layout |
| `mesh_n_theta` | 24 | - | NUMERICAL (vertices per ring of the tube/shaft sweeps) |
| `mesh_target_tris` | 1500 | - | NUMERICAL (~1-2k triangles per part) |
| `mesh_voxel_mm` | 0.3 | mm | NUMERICAL (implicit-function grid for the cap surfaces) |
| `tip_below_os_mm` | 4 | mm | CONTRACT 3.4 (path start: tube tip 4 mm below the external os) |
| `n_steps` | 80 | - | CONTRACT 3.4 (N_steps default 80) |
| `path_axis` | tube | - | CONTRACT 3.4 (device translates along the shaft axis; alternative 'tube' also written) |
| `corpus_ease` | smoothstep | - | DECISION: screw-motion parameter s(u) = smoothstep((u - u_ios)/(1 - u_ios)); 'linear' available |
| `tilt_axis` | LR | - | CONTRACT 3.2: tube axis = shaft axis rotated anteriorly about the patient LR axis (variant 'sagittal' reported) |
| `d_F_margin_mm` | 0.5 | mm | CONTRACT 3.3 / final/bdev.json: d_F = L_iu + 0.5 - canal_above_L_end |
| `vagina_fixed_inferior_mm` | 15 | mm | CONTRACT 1 (vagina.fixed_inferior = lowest 15 mm along the vaginal axis) |
| `shaft_axis_mode` | pca | - | RULE v2: 'pca' = principal axis of the vagina body (oriented +S); 'chord' = v1 (fixed point -> O_pre) |
| `shaft_line_through` | vagina_axis | - | RULE v2: shaft line through 'O_pre', or 'vagina_axis' (the vagina body's own axis line; O_pre projected onto it) |
| `flange_shift_mm` | 22.5 | mm | RULE v2 SCENARIO parameter Delta = seating push of the ovoids/packing: flange = base + Delta along shift_axis. Not predictable from preBT; the default is the in-sample best of the sweep (pose.json default_flange_shift_mm) |
| `shift_axis` | shaft | - | RULE v2: direction of the seating shift ('shaft' = insertion direction; 'tube' reported as a variant) |
| `flange_shift_sweep_mm` | [0.0, 10.0, 20.0, 30.0, 35.0, 40.0] | mm | RULE v2: Delta values scored against the BT tandem (validation only) |

Ovoid shape (decision): each ovoid is a lunar cap = the half (patient left / right of the mid-sagittal slot `ovoid_gap_mm`) of a dome-on-cylinder body of diameter `ovoid_diam_mm` and height `ovoid_height_mm`, the dome being the upper half of an ellipsoid of height `ovoid_dome_mm` (dome = height recovers a pure half-ellipsoid cap). Its centre may be offset laterally (`ovoid_offset_x_mm`, 0 = one body split in two) and the assembly may be tilted about x (`ovoid_tilt_deg`). The BT ovoid label is a solid ~40 mm disc-like signal void with a domed top at the flange and a flat bottom ~20 mm below, plus a packing tail; this family matches it, the contract's 'two ovoids at x = +/-11 mm' corresponds to the centroids of the two halves (+/-7.4 mm here).

### Ovoid fit (rigid two-cap model vs the BT ovoid label, pose = measured BT tandem)

| quantity | value |
|---|---|
| fitted diameter / lateral offset / dz / AP tilt / dome | 39.4 mm / 0.0 mm / 1.4 mm / -4.1 deg / 11.4 mm |
| assembly extents LR x AP x z | 39.4 x 39.4 x 22.2 mm (contract: 41.5 x 33.7 x 22.2) |
| Dice model vs label (whole label / ovoid zone) | 0.852 / 0.868 |
| volumes: label / model / overlap | 23.45 / 21.86 / 19.31 cc |
| leftover label - model (= packing, not device) | 4.14 cc (1.85 cc below the cap bottom; z_app range [-28.6, 3.1] mm) |
| model outside the label | 2.56 cc |

### Pose rule v2 (pose.json default): vaginal principal axis + seating shift Delta

Shaft axis = principal axis of the preBT vagina body (oriented +S), line through **vagina_axis**; tube axis = shaft axis rotated 24 deg anteriorly about LR; flange = base + Delta along the **shaft** axis. **Delta is the seating shift of the ovoids/packing: a SCENARIO parameter, not a prediction** (nothing in the preBT scan gives it; the in-sample best on this case is 22.5 mm, fine scan step 2.5 mm). Corpus placement is unchanged (a0 -> tube axis, L_end 17.5 mm above the flange). `pose.json` carries the corpus target for every Delta of the sweep (`corpus.by_flange_shift_mm`) so the scene module sweeps Delta from its cfg; the device translates along the tube axis (the tube enters the canal first).

| v2 variant (line through, Delta along) | axis angle to BT (deg): total (sag / cor) | best Delta of the coarse sweep (mm) | flange along / lateral (mm) | tip offset (mm) | K0 corpus: Dice, centroid err (mm) |
|---|---|---|---|---|---|
| pca_O_pre_shaft | 5.2 (-5.0 / -1.6) | 20 | -11.2 / 11.8 | 20.2 | 0.312, 19.9 |
| pca_O_pre_tube | 5.2 (-5.0 / -1.6) | 30 | +1.2 / 9.8 | 12.6 | 0.506, 13.2 |
| pca_vagina_axis_shaft **(chosen)** | 5.2 (-5.0 / -1.6) | 20 | -5.4 / 3.5 | 6.2 | 0.781, 5.5 |
| pca_vagina_axis_tube | 5.2 (-5.0 / -1.6) | 20 | -2.9 / 11.0 | 6.3 | 0.718, 6.9 |

Fine Delta scan of the chosen variant (K0 corpus vs the BT uterus; NONE = unmoved preBT corpus, centroid error 24.3 mm; rule v1 41.6 mm):

| Delta (mm) | K0 Dice | K0 centroid err (mm) | flange along / lateral (mm) | tip offset (mm) |
|---|---|---|---|---|
| 0.0 | 0.241 | 24.1 | -22.9 / 12.8 | 24.2 |
| 2.5 | 0.304 | 21.6 | -20.7 / 11.6 | 21.8 |
| 5.0 | 0.372 | 19.2 | -18.5 / 10.4 | 19.3 |
| 7.5 | 0.443 | 16.7 | -16.3 / 9.2 | 16.9 |
| 10.0 | 0.517 | 14.3 | -14.1 / 8.0 | 14.6 |
| 12.5 | 0.590 | 11.9 | -11.9 / 6.9 | 12.3 |
| 15.0 | 0.661 | 9.6 | -9.7 / 5.7 | 10.0 |
| 17.5 | 0.727 | 7.4 | -7.6 / 4.6 | 7.9 |
| 20.0 | 0.781 | 5.5 | -5.4 / 3.5 | 6.2 |
| 22.5 **best** | 0.809 | 4.1 | -3.2 / 2.6 | 5.1 |
| 25.0 | 0.797 | 4.1 | -1.0 / 2.1 | 5.1 |
| 27.5 | 0.752 | 5.4 | +1.2 / 2.1 | 6.3 |
| 30.0 | 0.686 | 7.4 | +3.4 / 2.8 | 8.0 |
| 32.5 | 0.615 | 9.6 | +5.6 / 3.7 | 10.1 |
| 35.0 | 0.539 | 11.9 | +7.7 / 4.8 | 12.3 |
| 37.5 | 0.465 | 14.3 | +9.9 / 5.9 | 14.7 |
| 40.0 | 0.395 | 16.7 | +12.1 / 7.1 | 17.0 |
| 42.5 | 0.330 | 19.1 | +14.3 / 8.2 | 19.4 |
| 45.0 | 0.272 | 21.6 | +16.5 / 9.4 | 21.9 |

Residual at the best Delta: axis angle 5.2 deg (sagittal -5.0, coronal -1.6), flange lateral offset 2.6 mm (along -3.2), tip offset 5.1 mm; K0 corpus Dice 0.809, centroid error 4.1 mm. What remains is the axis error and the perpendicular offset of the shaft line; the along-axis error is absorbed by Delta by construction (in-sample). Selection rule: IN-SAMPLE: variant and Delta minimising the K0 corpus centroid error on this case (ties: smaller lateral flange error).

### Pose rule v1 (REJECTED on this case; kept as the reference in `validation.rule_v1`)

Shaft axis = chord from the vaginal fixed point (centroid of the lowest 15 mm of the vagina body along its principal axis, computed from the preBT labels exactly as the meshing module defines `vagina.fixed_inferior`) to the external os `O_pre`; flange at `O_pre`; tube axis = shaft axis rotated 24 deg anteriorly about the patient LR axis (the in-plane variant differs by 0.00 deg); corpus: a0 -> tube axis by the minimal rotation with L_end 17.5 mm above the flange (d_F = L_iu + 0.5 - 44.8; bdev.json d_F_tip = 17.5); insertion path: translation along the tube axis over 91.0 mm (tip 4 mm below the os at u = 0), corpus screw motion (smoothstep) from u = 0.390 (tip at the internal-os level) to u = 1, then settle.

| variant | axis angle to the BT tandem (deg): total (sagittal / coronal) | flange offset total / along BT axis / lateral (mm) | tip offset (mm) | K0 corpus: Dice vs BT uterus, centroid err (mm) |
|---|---|---|---|---|
| **rule v1**, angle 24 (BONE frame) | 23.1 (-17.6 / -15.5) | 30.2 / -28.7 / 9.4 | 44.8 | 0.001, 41.6 |
| contract alternative: angle 0 | 43.9 (-41.6 / -15.7) | 30.2 / -28.7 / 9.4 | 64.1 | 0.000, 55.1 |
| contract alternative: angle 30 | 19.1 (-11.5 / -15.8) | 30.2 / -28.7 / 9.4 | 41.9 | 0.011, 39.8 |
| diagnostic: oracle angle 41 (in-sample) | 15.3 (-0.5 / -16.8) | 30.2 / -28.7 / 9.4 | 40.0 | 0.020, 38.9 |
| diagnostic: shaft axis = vagina principal axis (preBT-only) | 5.2 (-5.0 / -1.6) | 30.2 / -28.7 / 9.4 | 30.9 | 0.127, 30.8 |
| diagnostic: tube axis = canal axis a0 (preBT-only) | 14.0 (-8.3 / +11.8) | 30.2 / -28.7 / 9.4 | 31.3 | 0.165, 28.9 |
| rule v1 in the GLOBAL frame | 22.9 (-17.3 / -15.4) | 30.4 / -28.8 / 9.7 | 45.1 | - |
| rule v1 in the HR frame | 18.1 (-16.7 / -7.2) | 15.3 / -6.1 / 14.1 | 27.0 | - |

NONE = preBT anatomy unmoved (centroid error 24.3 mm in the BONE frame). Sign convention: along-axis offset < 0 = predicted flange inferior to the real one along the BT tandem; sagittal component > 0 = predicted axis more anterior, coronal > 0 = more to the patient's right.

Why rule v1 fails on this case (numbers in `pose.json[validation]`): (i) the real flange sits 29.9 mm above the preBT external os along a0 (the cervix was pushed cranially by the device and packing; uterus centroid shift 24.3 mm in the BONE frame), so a flange pinned at O_pre is ~29 mm too inferior whatever the axis; (ii) the vaginal chord from the fixed point to O_pre tilts 18.6 deg away from the vagina body's own principal axis because the external os lies 16.3 mm lateral of that axis line - the chord inherits a 15 deg leftward tilt that the real (centred) tandem does not have; (iii) the preBT vagina is ~17 deg more posterior than the inserted shaft. Replacing the chord by the vaginal principal axis brings the axis error from 23.1 to 5.2 deg and the K0 corpus centroid error from 41.6 to 30.8 mm without using any BT information; the canal-axis variant gives 14.0 deg / 28.9 mm.

Sensitivity to the vaginal fixed point (+/-5 mm along x, y, z; chord length 49.2 mm): tube axis changes by <= 5.78 deg, the corpus centroid moves by <= 4.91 mm, the tip by <= 6.23 mm; the BT-frame axis error ranges 19.9-27.0 deg. The flange is pinned to O_pre, so the fixed point only steers the axes.

Reading: the along-axis flange offset is the between-session cranial shift of the cervix by the applicator + packing that a rule with the flange at the preBT external os cannot produce (the real flange sits 7.4 mm above L_end along a0 in the BONE frame, poses_align.json); the lateral offset and the axis angle are what the vaginal-chord construction gets wrong on its own. NONE -> K0 corpus centroid error: 24.3 -> 41.6 mm.

### Caveats
- Device identity is unconfirmed until the RTPLAN / vendor geometry is available: tube radius, L_iu, junction angle and the cap shape are measured from the BT labels (MRI signal voids include partial volume and susceptibility, the ovoid label includes packing); the 30 mm curved shaft, the channel layout (azimuths, oblique entry radius) and the 0.5 mm slot are assumed.
- The BT ovoid label is not the device: 4.1 cc of it is packing; the fit optimises Dice against the whole label, bounded so the caps stay at the flange (dz in [-5, 3] mm) - the reported Dice is a lower bound on the device fit.
- Insertion path: translation along the tube axis (tip 11.8 mm off the os line at u = 0; the other axis is written as `insertion_path_alt`). The cranial cervix shift that rule v1 could not produce is, in v2, the seating shift Delta: an explicit scenario parameter to sweep, not a prediction; a cohort or an intra-procedural measurement is what would give it a prior.
- Roll about the tube axis is unobservable from the labels (the caps are near-axisymmetric): the applicator x axis is defined by the patient LR direction in both the rule and the BT frame.

## EVALUATION (`eval_hybrid.py`, CONTRACT section 6)

Module `eval_hybrid.py` (host, python 3.13: numpy, scipy, nibabel, vtk, matplotlib). Scores a hybrid run against the BT labels in two frames and against the two pre-declared comparators, with the frame-perturbation noise floor and the equivalence margin taken from `applicator_sim/final_eval.py`; and rasterises the per-node displacement into `u_preBT.nii.gz`, **the project's actual deliverable**. Voxelisation, `dice()` and `surf_dists()` are reused verbatim from `evaluate.py` (which copies them from the registration stage), so the hybrid numbers are on the identical route as the stage-2/3 validation.

Frames/units: everything is scored on the **BT grid** (320x288x104, 1.125x1.125x1.6 mm); model surfaces live in preBT world RAS mm and are carried into BT world by a rigid map. Units mm, volumes cc, angles deg, Dice dimensionless. Every JSON states its frame and units. No patient-derived coordinate is hard-coded: the device, the pose rule, the frames and the landmarks are read from `hybrid/applicator/pose.json`, `inputs/*.json`, `validation/alignment.json` and the labels.

### Commands

```
cd D:/neve/.claude/worktrees/sofa_applicator/applicator_sim
$env:MPLBACKEND='Agg'; $env:PYTHONDONTWRITEBYTECODE='1'     # (bash: export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1)
python hybrid/eval_hybrid.py selftest                  # build runs/SELFTEST_K0, then score + compare + field on it
python hybrid/eval_hybrid.py comparators               # NONE + K0 at every Delta -> eval/comparators/metrics.json
python hybrid/eval_hybrid.py score   --tag H1          # -> eval/<tag>/metrics.json
python hybrid/eval_hybrid.py compare --tag H1          # -> eval/<tag>/compare.json + figs/eval_<tag>_*.png   (~6 min)
python hybrid/eval_hybrid.py field   --tag H1          # -> eval/<tag>/field/                                  (~2 min)
python hybrid/eval_hybrid.py field   --tag H1 --ref-dvf <ref.nii.gz>    # per-structure error vs a reference DVF
python hybrid/eval_hybrid.py figs    --tag H1          # redraw both figures from an existing compare.json (seconds)
```
Options: `--deltas` (restrict the comparator Delta list), `--delta` (selftest Delta), `--n-boot` (noise-floor draws, default 30), `--band-mm` (extrapolation band, default 25).

### Frames (CONTRACT 6)

| frame | definition | what it scores |
|---|---|---|
| `E_app` | Kabsch putting the MODEL's own final tube pose onto the REAL BT tandem (applicator-frame landmarks; residual ~1e-14 mm) | the tissue around the device, independently of the pose rule: the device lands on the real one by construction |
| `PELVIS` | the peri-organ MI frame, `BONE` in `align.py` / `validation/alignment.json` (no organ label enters its fit); `x_BT = R^T (y_pre - t)` | the pose rule and the OARs |

### Structures, references and metrics

| structure | BT reference | model mask |
|---|---|---|
| cervix | BT HR-CTV **minus** BT uterus (38.5 cc) | cervix body |
| vagina | BT vagina label (99.9 cc) | vagina body (tissue only, no device either side) |
| vagina+device | BT vagina **union** BT applicator | vagina body union model device (CONTRACT 6: the BT vagina label contains the device, so the device is added on **both** sides) |
| bladder / rectum / sigmoid | their BT labels (232.4 / 41.4 / 26.6 cc) | their bodies |
| corpus | BT uterus (37.7 cc) | corpus body — **reported, not a target** |

Metrics per structure: **Dice**; **MSD** = symmetric mean surface distance; **HD95** = 95th percentile of the **pooled** symmetric surface distances (the registration-stage definition, `evaluate.surf_dists` verbatim); **HD95max** = max of the two directed 95th percentiles; `vol_cc`. Computed by `final_eval.metrics_ext`.

Also reported, per frame: **device-to-OAR minimum distance** (rectum, bladder, sigmoid; minimum distance between the device mask and the OAR mask on the BT grid, 0 = intersection, with `overlap_cc` saying how deeply); **tip-to-serosa** and **flange-to-corpus** (sub-voxel, `final_eval.exit_sub`/`entry_sub`, along the model tube axis); and the **pose-rule error** (predicted tube vs real BT tandem: axis angle, flange offset split into along-axis and lateral, tip offset).

BT reference values, all recomputed here rather than copied: applicator-rectum **2.76 mm**, applicator-bladder **6.27 mm**, applicator-sigmoid **15.19 mm**, tip-to-serosa **5.44 mm**, flange-to-corpus **22.94 mm**. The first two and the tip-to-serosa reproduce the contract's 2.76 / 6.27 / 5.44 exactly, which is an independent check that this module's route matches the stage-2/3 one.

### Comparators (pre-declared, CONTRACT 6)

- **`NONE`** - preBT bodies unmoved (no insertion). It has no device of its own; the rule's device at the default Delta is attached **only** to define its `E_app` frame and to keep `vagina+device` like-for-like. The plain `vagina` structure is device-free on every model, so the tissue-only number is always available.
- **`K0`** - corpus + cervix moved rigidly by the pose rule's corpus transform (the cervix rides with the corpus), every other body unmoved; one entry per Delta in `pose.json corpus.by_flange_shift_mm` plus the default (22.5 mm).
- **`SIM`** - the hybrid run.

**Cross-check:** this module's K0 corpus Dice reproduces the applicator module's independently computed sweep (`pose.json validation.rule_v2`) to +/-0.002 at every Delta (0.241 / 0.516 / 0.781 / 0.811 / 0.688 / 0.541 / 0.396 here vs 0.241 / 0.517 / 0.781 / 0.809 / 0.686 / 0.539 / 0.395 there), and the pose-rule error at the default Delta (axis 5.204 deg, flange 4.133 mm total / -3.183 along / 2.636 lateral) matches `pose.json validation.default_rule_score` to 3 decimals. The two implementations share no code beyond `geom.py`.

### NONE (Dice / MSD mm / HD95 mm)

| structure | E_app | PELVIS |
|---|---|---|
| cervix | 0.272 / 9.32 / 20.84 | 0.206 / 10.59 / 23.77 |
| vagina | 0.103 / 15.40 / 26.69 | 0.104 / 16.67 / 30.87 |
| vagina+device | 0.402 / 7.61 / 21.56 | 0.391 / 7.93 / 25.88 |
| bladder | 0.766 / 5.22 / 13.51 | 0.791 / 4.64 / 11.98 |
| rectum | 0.400 / 7.80 / 20.27 | 0.371 / 9.05 / 23.88 |
| sigmoid | 0.253 / 18.99 / 75.61 | 0.189 / 19.37 / 73.21 |
| corpus | 0.351 / 8.70 / 18.06 | 0.262 / 10.36 / 21.46 |

NONE tip-to-serosa is **-12.1 mm** (negative = the predicted tip does not reach the unmoved corpus at all).

### K0 per Delta, PELVIS frame (Dice / MSD mm / HD95 mm)

In this frame the four bodies K0 does not move are Delta-invariant: vagina 0.104 / 16.67 / 30.87, bladder 0.791 / 4.64 / 11.98, rectum 0.371 / 9.05 / 23.88, sigmoid 0.189 / 19.37 / 73.21.

| Delta (mm) | cervix | corpus (reported) | vagina+device |
|---|---|---|---|
| 0 | 0.195 / 11.35 / 25.06 | 0.241 / 11.03 / 22.51 | 0.375 / 9.05 / 24.11 |
| 10 | 0.405 / 7.46 / 18.68 | 0.516 / 6.41 / 13.04 | 0.371 / 8.46 / 23.89 |
| 20 | 0.554 / 5.09 / 15.63 | 0.781 / 2.63 / 5.62 | 0.390 / 8.01 / 25.60 |
| **22.5 (default)** | **0.570 / 4.91 / 15.75** | **0.811 / 2.21 / 4.91** | **0.391 / 7.93 / 25.88** |
| 30 | 0.517 / 5.63 / 17.35 | 0.688 / 3.77 / 7.47 | 0.331 / 8.46 / 26.02 |
| 35 | 0.429 / 7.00 / 19.19 | 0.541 / 5.86 / 10.72 | 0.255 / 9.21 / 26.02 |
| 40 | 0.331 / 8.80 / 21.52 | 0.396 / 8.10 / 14.76 | 0.177 / 9.86 / 26.02 |

### K0 per Delta, E_app frame (Dice / MSD mm / HD95 mm)

Here corpus and cervix are **Delta-invariant** (corpus 0.883 / 1.44 / 3.56, cervix 0.613 / 4.38 / 14.46): K0 moves them rigidly with the device, and `E_app` pins the device onto the real tandem, so they land identically whatever Delta is. What Delta changes in this frame are the bodies that do **not** ride with the device, because the frame itself moves with the device pose.

| Delta (mm) | vagina | vagina+device | bladder | rectum | sigmoid |
|---|---|---|---|---|---|
| 0 | 0.102 / 16.36 / 32.66 | 0.380 / 10.38 / 32.02 | 0.504 / 12.04 / 21.78 | 0.246 / 11.58 / 27.21 | 0.035 / 24.48 / 74.32 |
| 10 | 0.103 / 15.09 / 24.55 | 0.394 / 8.74 / 23.53 | 0.643 / 8.32 / 14.80 | 0.330 / 9.30 / 22.81 | 0.177 / 20.29 / 74.52 |
| 20 | 0.104 / 15.13 / 25.02 | 0.403 / 7.77 / 21.70 | 0.752 / 5.56 / 13.37 | 0.394 / 7.95 / 20.53 | 0.269 / 18.89 / 75.37 |
| **22.5 (default)** | 0.103 / 15.40 / 26.69 | 0.402 / 7.61 / 21.56 | 0.766 / 5.22 / 13.51 | 0.400 / 7.80 / 20.27 | 0.253 / 18.99 / 75.61 |
| 30 | 0.103 / 16.64 / 34.26 | 0.403 / 7.20 / 21.21 | 0.753 / 5.71 / 15.35 | 0.392 / 7.73 / 21.18 | 0.136 / 20.68 / 76.91 |
| 35 | 0.102 / 17.81 / 38.99 | 0.402 / 7.06 / 21.24 | 0.709 / 6.93 / 17.40 | 0.369 / 8.05 / 22.67 | 0.065 / 22.67 / 77.76 |
| 40 | 0.093 / 19.39 / 43.73 | 0.395 / 7.05 / 21.35 | 0.649 / 8.55 / 20.27 | 0.341 / 8.61 / 24.29 | 0.016 / 25.28 / 79.04 |

### K0 per Delta: pose rule, landing and device-to-OAR

The predicted axis angle is **5.20 deg at every Delta** (Delta slides the flange along the shaft axis, it does not rotate the tube). Device-to-rectum and device-to-bladder are **0.00 mm at every Delta** because the kinematic comparators do not push the OARs aside: the device simply interpenetrates them, and `overlap_cc` is the honest measure. A converged FEM run should drive these overlaps to ~0 cc and the distances towards the BT reference (2.76 / 6.27 mm).

| Delta (mm) | flange offset total / along / lateral (mm) | tip-to-serosa (mm) | device-sigmoid (mm) | overlap rectum / bladder (cc) |
|---|---|---|---|---|
| 0 | 26.19 / -22.86 / 12.79 | 5.24 | 13.00 | 5.42 / 0.92 |
| 10 | 16.24 / -14.11 / 8.03 | 6.04 | 3.91 | 3.61 / 1.94 |
| 20 | 6.43 / -5.37 / 3.55 | 5.86 | 0.00 | 1.70 / 2.35 |
| **22.5 (default)** | **4.13 / -3.18 / 2.64** | **5.56** | 0.00 | 1.30 / 2.27 |
| 30 | 4.37 / +3.38 / 2.78 | 5.62 | 0.00 | 0.44 / 1.60 |
| 35 | 9.11 / +7.75 / 4.79 | 5.42 | 0.00 | 0.31 / 0.89 |
| 40 | 14.03 / +12.12 / 7.07 | 5.23 | 2.25 | 0.19 / 0.25 |

K0's tip-to-serosa sits at 5.2-6.0 mm against the BT 5.44 mm at every Delta, which is **not** a prediction: the corpus is placed by the `d_F = L_iu + 0.5 - canal_above_L_end` rule, which puts the canal end at the tip by construction. Sign convention (as the applicator module): along < 0 = predicted flange inferior to the real one along the BT tandem.

### SIM minus comparator, with the noise floor

`compare --tag <tag>` reports SIM minus NONE and SIM minus K0 (at the **run's own** Delta, read from `device_final.json` / `summary.json`) for every structure in both frames, as a point estimate and with the noise floor reused from `final_eval.cmd_boot`: **30 draws** of the same random rigid perturbation (rotation sd 2.0 deg/axis about the BT uterus centroid, translation sd 2.34/sqrt(3) = 1.351 mm/axis) applied to SIM **and** to the comparator, reporting the 5 / 50 / 95 percentiles of the difference and the declared **+/-0.5 mm equivalence margin** on MSD. Verdicts: `sim better` (p95 < 0), `comparator better` (p5 > 0), or `interval covers 0`, with `; within +/-0.5 mm equivalence` appended when the whole interval lies inside the margin. Masks and metrics are memoised per draw (the models share the bodies they do not move), which is what makes 30 draws ~5 min rather than ~1 h.

Figures: `figs/eval_<tag>_summary.png` (2 frames x [absolute Dice | dMSD vs NONE | dMSD vs K0] with 5-95% whiskers and the equivalence band) and `figs/eval_<tag>_overlay_{E_app,PELVIS}.png` (BT sagittal/coronal/axial planes through the tandem mid-point, cropped to the structures; BT labels solid, SIM dashed, NONE dotted, K0 dash-dot; device in white).

### Displacement field - `eval/<tag>/field/` (the deliverable)

| file | content |
|---|---|
| `u_preBT.nii.gz` | **4-D (288, 320, 104, 3) float32, preBT affine, RAS.** Component `c = 0,1,2` = displacement along world **+x (R), +y (A), +z (S)** in **millimetres**. **Forward (Lagrangian)** displacement defined at preBT material positions: `x_deformed = x_preBT + u(x_preBT)` |
| `mask.nii.gz` | uint8: `0` = zero field (beyond the band), `1` = extrapolated, `2` = inside a body |
| `warped_preBT_MRI.nii.gz` | the preBT image pulled back through the map |
| `warped_preBT_labels.nii.gz` | uint8, codes `1..6` = uterus, HR-CTV, vagina, bladder, rectum, sigmoid |
| `field_stats.json` | format, method, per-body magnitudes, voxel counts, inverse diagnostics, the warped-label check, the `--ref-dvf` result or hook |

Construction: **inside** a body, barycentric interpolation of the nodal displacement in that body's tetrahedra (`vtkProbeFilter` on `meshes/<body>/tets.vtk`, same point order as `final/<body>_u.npy`); overlaps resolved by the meshing priority `corpus > cervix > vagina > bladder > rectum > sigmoid`. **Outside**, inverse-distance (k=12, 1/d^2) from the nearest body-surface points (surface vertices + triangle centroids, displacement taken from the tet nodes) times a taper `(1 - d/25)^2`, so the field decays continuously to zero at the edge of a **25 mm** band; beyond the band it is exactly zero.

The **backward** map needed for warping is built by scatter, not by a plain fixed point: every supported voxel `x` is pushed forward to `x + u(x)`, the nearest pushed-forward points of a target give `v` by inverse-distance averaging (exact for a rigid motion), and the branch is chosen by the same priority order. A fixed point `v <- u(y - v)` started at `v = 0` does **not** work here - the corpus moves ~30 mm, so the first sample is taken outside the body where the field has tapered away. Warping is applied only where the map is covered; outside the support the material is unchanged, and inside the support but outside the image the material left and nothing arrived (set to background). Without that restriction a moving body is duplicated at its rest position.

`--ref-dvf <path>` accepts a **NIfTI vector field** (4-D `(i,j,k,3)` or 5-D `(i,j,k,1,3)`, mm, forward displacement, RAS components; resampled onto the preBT grid via its own affine) and reports per-structure `|u_pred - u_ref|` (mean, median, p95, max) plus the per-component signed bias. Without it the hook and both formats are printed. **DICOM REG (RayStation) is not implemented**: the export flavour is unknown (rigid spatial-registration object vs deformable grid, the grid frame of reference, and whether the vectors are stored pre- or post-image-orientation). Given one export it is a small addition (`pydicom` -> `DeformableRegistrationSequence` -> `DeformableRegistrationGridSequence` -> `VectorGridData`, then the same comparison code).

### Self-test - `runs/SELFTEST_K0`, PASS

`selftest` builds a synthetic run by applying the K0 rigid motion at the default Delta to the corpus and cervix **nodes** (`u` = the rigid displacement, zero elsewhere), writing `final/<body>_u.npy`, `final/<body>.obj` (= rest surface + `u[surface_obj_vertex_to_tet_node]`, i.e. the same route the scene module uses), a `device_final.json` derived from `pose.json` and a `cfg.json`; then runs `score`, `compare` and `field` end to end. Results (`eval/SELFTEST_K0/selftest.json`):

- **SIM == K0 at the same Delta**: max abs difference over all metrics, structures and both frames **0.0005** (the metrics are stored rounded to 4 dp; this is sub-voxel agreement, not a discrepancy). Every SIM-minus-K0 interval is `[0.000, 0.000]`, i.e. within the equivalence margin, in both frames.
- **SIM minus NONE** behaves as it must: only the two bodies K0 moves differ - cervix dMSD -4.89 mm [-5.73, -3.63] (E_app) and -5.64 mm [-6.32, -4.65] (PELVIS), corpus -7.00 mm [-7.85, -6.13] and -7.94 mm [-8.57, -7.26]; the four unmoved bodies are exactly 0.00 within equivalence.
- **Warped-label check** (each body warped with its own backward map, compared with the voxelisation of `final/<body>.obj`): corpus Dice **0.977**, MSD 0.290 mm, HD95 1.125 mm; cervix **0.976**, 0.267 mm, 1.125 mm; vagina, bladder, rectum, sigmoid all **1.0000 / 0.000 / 0.000**. The residual on the two moving bodies is one voxel of resampling, and their warped volumes match to 0.1 % (37.45 vs 37.51 cc; 45.67 vs 45.77 cc).
- Inverse residual inside the bodies: median **0.000 mm**, p95 **0.310 mm** (see the caveat on the max below).

### Runs scored

| tag | status | result |
|---|---|---|
| `SELFTEST_K0` | synthetic | self-test PASS (above) |
| `HSMOKE` | `abort_inverted_tets`, 2 steps, u = 0.0, min vol ratio -0.57 | scored and flagged; the device never left the start pose, so `E_app` translates the anatomy ~90 mm and its numbers are degenerate (Dice 0, some nan) |
| `H1` | `abort_inverted_tets`, 17 steps, u = 0.604, min vol ratio 0.062 | scored and flagged; partially inserted (flange still 39.1 mm short along the axis), corpus moved 5.9 mm of the 20.8 mm target |

Neither SOFA run has completed, so **no SIM number here is a prediction of the post-insertion anatomy yet**. `metrics.json` carries `run.status` with `usable_as_prediction`, and `score` prints a `*** WARNING` banner whenever a run aborted, did not converge or did not reach the final pose.

`H1` for the record, PELVIS frame, Dice / MSD / HD95: cervix 0.258 / 9.86 / 22.91, vagina 0.104 / 16.73 / 30.88, vagina+device 0.371 / 8.40 / 19.69, bladder 0.796 / 4.51 / 11.63, rectum 0.371 / 9.04 / 24.01, sigmoid 0.188 / 19.40 / 73.24, corpus 0.405 / 7.79 / 16.05. SIM minus NONE (dMSD, median [5-95%]): cervix **-0.72 [-0.77, -0.63]** and corpus **-2.54 [-2.66, -2.47]** (sim better), bladder -0.12 (sim better, within equivalence), vagina / rectum / sigmoid within +/-0.5 mm equivalence. SIM minus K0 at Delta 22.5: cervix +4.96 and corpus +5.39 (comparator better) - as it must be, since H1 stopped at u = 0.604 and moved the corpus 5.9 mm of the 20.8 mm the rule asks for. All `E_app` numbers for `H1` are meaningless (the frame is built from a device 39.1 mm short of its final pose, so it translates the whole anatomy; the overlay figure shows this directly).

`H1`'s displacement field is nonetheless well formed and passes the warped-label check on **all six** bodies (Dice 0.973-0.996, MSD 0.04-0.33 mm), with inverse residual median 0.001 mm / p95 0.080 mm and only 6625 branch-ambiguous targets against the self-test's 54132 - small, physical displacements (max 7.3 mm) fold the extrapolated band far less than K0's rigid 31 mm jump.

### Caveats

- **The comparators are not physical.** K0 and NONE move nothing but the corpus and cervix, so their device interpenetrates the rectum (up to 5.4 cc) and bladder (up to 2.4 cc) and the device-to-OAR distances are 0.00 mm. That is the point of the comparison - it is what the FEM has to fix - but the distances are only informative once a run converges.
- **The `vagina` Dice (~0.10) is not a model failure.** The BT vagina label is 99.9 cc because it contains the device **and the packing**, against a 5.5 cc preBT vagina body (itself reduced from the 10.8 cc label by the meshing priority rule). `vagina+device` (~0.40) is the like-for-like number, and even it cannot account for packing, which is not modelled. The stage-2 evaluation hit the same wall and used an ovoid-zone slab; that slab is **not** reproduced here.
- **Sigmoid agreement is poor for every model** (Dice 0.19 PELVIS, HD95 ~73 mm) because the BT sigmoid label is 26.6 cc against 40.4 cc preBT and its extent differs between sessions; the HD95 is dominated by the ends of the labelled segment, not by a local error. Treat the sigmoid as weakly constrained.
- **Delta is in-sample.** 22.5 mm was chosen by the applicator module to minimise the K0 corpus centroid error on this case. The K0 tables above are therefore an in-sample best, not a held-out result, and the whole Delta sweep is reported so a run can be judged against the right one.
- **The extrapolated band is not invertible.** The taper carries the full body displacement (~30 mm) down to zero over 25 mm, so `|grad u| > 1` there and the map folds; 54132 of 1050701 covered targets (5 %) have a competing source more than 2 mm away in displacement, resolved by the priority order. The max inverse residual (~28 mm) is dominated by the interface between a moving body's band and a static body's interior, where the field is genuinely discontinuous - the median (0.000 mm) and p95 (0.310 mm) are the meaningful numbers, and the warped-label check is the real test. A physically converged run, where contact stops bodies passing through each other, will fold far less.
- The band width (25 mm), the IDW neighbour count (12) and the taper exponent are modelling choices, not measurements; outside the band the field is exactly zero, which is a hard edge at 25 mm from any body surface.
- `E_app` is defined from the model's own device pose, so for a run that never reached its final pose the frame itself is displaced and the organ metrics in that frame are meaningless (see `HSMOKE`). `PELVIS` has no such dependence and stays interpretable for a partial run.
- The noise floor perturbs the **frame**, not the model. It answers "could this difference be produced by registration uncertainty alone", not "is the simulation right".

## SCENE (`scene_hybrid.py`, `run_hybrid.py`, `view_hybrid.sh`, CONTRACT sections 4-5)

The SOFA v22.12 scene: five deformable FEM bodies, the kinematic rigid corpus and the kinematic rigid device, coupled
by Lagrangian contact, an attach constraint and distributed node ties. Units mm / kg / s (force mN, stress kPa,
stiffness mN/mm, density kg/mm^3). Frame: preBT world RAS mm. No patient-derived coordinate is hard-coded: every
number comes from `hybrid/meshes/**` and `hybrid/applicator/{applicator.json, pose.json}`.

### Commands

```bash
cd D:/neve/.claude/worktrees/sofa_applicator/applicator_sim
# the converged run (container, SOFA; ~350 s, one compute container at a time)
bash run_docker.sh H6 hybrid/run_hybrid.py --tag H6 \
  --cfg '{"n_presettle":2,"n_approach":4,"n_insert":24,"n_seat":0,"max_settle_steps":40,"max_steps":75,"wall_limit_s":500}'
# verification figures + interpenetration report (host, pyvista)
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 py -3.11 hybrid/run_hybrid.py render --tag H6
# live viewer over X11 (needs an X server on the host; stops with the window)
bash hybrid/view_hybrid.sh H6
```
`--cfg` takes inline JSON or a path. Every key of `scene_hybrid.CFG` is overridable; `flange_shift_mm` selects the
scenario Delta from `pose.json` (default = `default_flange_shift_mm` = 22.5). Outputs go to `hybrid/runs/<tag>/`
(`cfg.json`, `log.jsonl`, `final/<body>.obj`, `final/<body>_u.npy`, `device_final.json`, `summary.json`), figures to
`hybrid/figs/scene_<tag>_{initial,final,compare}.png` and the interpenetration table to
`hybrid/logs/scene_<tag>_contacts.json`. `final/<body>_u.npy` is (n_nodes, 3) in mm, in the **same point order as
`meshes/<body>/tets.vtk`** (asserted against `meta.json` node counts; the evaluation module depends on it).

### Graph

```
root  gravity 0, dt 0.02 s, FreeMotionAnimationLoop + GenericConstraintSolver (250 it, tol 1e-3)
      CollisionPipeline / BruteForceBroadPhase / BVHNarrowPhase
      LocalMinDistance (alarm 1.5 mm, contact 0.5 mm, angleCone 0.01) + CollisionResponse FrictionContactConstraint mu 0.2
  /<body>   x5 deformable (cervix, vagina, bladder, rectum, sigmoid)
      EulerImplicitSolver (rayleighMass 0.1, rayleighStiffness 0) + SparseLDLSolver (CompressedRowSparseMatrixMat3x3d)
      MeshVTKLoader(tets.vtk) + TetrahedronSetTopologyContainer/Modifier/GeometryAlgorithms + MechanicalObject
      MeshMatrixMass (massDensity 1.05e-6 kg/mm^3, lumped) + TetrahedralCorotationalFEMForceField (method large)
      supports (below) ... then LinearSolverConstraintCorrection LAST in the node
      /surf  TriangleSetTopology* + Tetra2TriangleTopologicalMapping + MechanicalObject + IdentityMapping
             Triangle/Line/PointCollisionModel (group = the contact matrix)   /vis OglModel (bodies.json colour)
  /corpus   Rigid3d MechanicalObject (pose rule) -> RigidMapping: corpus surface.obj (collision+visual)
            /iface  rigid-mapped copy of the 127 cervix `interface_corpus` nodes (AttachConstraint target)
  /tandem   Rigid3d -> RigidMapping: tube.obj + shaft.obj (collision + visual)
  /ovoids   Rigid3d -> RigidMapping: ovoid_L.obj + ovoid_R.obj (collision + visual)
  /targets  solver-less Vec3d states the controller rewrites each step: canal_tgt, apex_tgt, lig_tgt
  /supports solver-less Vec3d state: the static cardinal-ligament anchors (drawn in the viewer)
```
The corpus and the device are **two** rigid bodies, not one: the rule gives them different motions (the device
translates from u = 0, the corpus starts its screw motion only at `u_tip_at_internal_os` = 0.39 and they coincide only
at u = 1). Both are `moving=True, simulated=False`, i.e. kinematic obstacles the controller repositions each step;
contact constraints are therefore one-sided and act only on the tissue.

### Parameters (`scene_hybrid.CFG`, copied verbatim into `runs/<tag>/cfg.json`)

| key | value | unit | provenance |
|---|---|---|---|
| `flange_shift_mm` | None -> 22.5 | mm | SCENARIO Delta, from `pose.json.default_flange_shift_mm` |
| `E_kPa` | cervix 30, vagina 15, bladder 8, rectum 8, sigmoid 8 | kPa | ASSUMED (CONTRACT 4) |
| `nu` | cervix/vagina/rectum/sigmoid 0.45, bladder 0.49 | - | ASSUMED (CONTRACT 4) |
| `density_kg_per_mm3` | 1.05e-6 | kg/mm^3 | ASSUMED (soft tissue ~ water) |
| `material` | `corotational` | - | CONTRACT 4; `neohookean` -> TetrahedronHyperelasticityFEMForceField (mu = E/2(1+nu), k = E/3(1-2nu)) |
| `dt_s` | 0.02 | s | CONTRACT 4 |
| `rayleigh_stiffness` / `rayleigh_mass` | 0.0 / 0.1 | - | DEVIATION, see below |
| `alarm_mm` / `contact_mm` / `friction_mu` | 1.5 / 0.5 / 0.2 | mm, mm, - | CONTRACT 4 |
| `gcs_max_it` / `gcs_tol` | 250 / 1e-3 | -, - | NUMERICAL |
| `k_canal_mN_per_mm` | 200 | mN/mm | NUMERICAL (canal dilation tie) |
| `canal_max_offset_mm` | 0.5 | mm | NUMERICAL (bounds the tie force at k*offset) |
| `k_attach_mN_per_mm` | 2000 | mN/mm | only for `attach_impl="springs"` |
| `k_apex_mN_per_mm` | 20 | mN/mm | ASSUMED (vagina apex follows the cervix) |
| `k_cardinal_mN_per_mm` / `cardinal_len_mm` | 20 / 25 | mN/mm, mm | CONTRACT 4 (per node, anchors 25 mm lateral) |
| `k_bladder_support_mN_per_mm` | 2 | mN/mm | CONTRACT 4 (per node, to rest) |
| `k_rectum_support_mN_per_mm` | 2 | mN/mm | CONTRACT 4 (per node, to rest) |
| `k_fixed_mN_per_mm` | 1e4 | mN/mm | NUMERICAL, replaces FixedConstraint (see below) |
| `n_presettle`/`n_approach`/`n_insert`/`n_seat` | 2 / 4 / 24 / 0 | steps | NUMERICAL (see schedule) |
| `max_settle_steps` / `wall_limit_s` | 40 / 500 | steps, s | NUMERICAL (container `timeout 570`) |
| `conv_dx_mm` / `conv_steps` | 0.02 / 5 | mm, - | CONTRACT 5 |
| `conv_constraint_err_per_contact` | 0.01 | - | CONTRACT 5, normalised (see below) |
| `min_vol_ratio_abort` | 0.2 | - | CONTRACT 4 stop rule |

Node sets used (sizes from `meta.json`): cervix `interface_corpus` 127, `canal` 38, `lateral_os_level` 168; vagina
`fixed_inferior` 435, `apex` 98; bladder `anterior_support` 377; rectum `posterior_support` 237, `fixed_ends` 103;
sigmoid `fixed_ends` 82.

### Contact matrix: what was excluded and why

Two collision models collide iff their `group` sets are **disjoint**, so a shared id encodes an excluded pair. Every
model has a non-empty group, so self-collision is off everywhere (CONTRACT 4).

| id | pair | reason |
|---|---|---|
| 1 | corpus \| cervix | attached through `interface_corpus`; they overlap **-0.99 mm** at rest (CONTRACT 4) |
| 2 | cervix \| vagina | rest overlap **-0.87 mm**; coupled instead by the apex follow springs |
| 3 | rectum \| sigmoid | rest overlap **-0.57 mm**; both `fixed_ends` sets (which include the junction) are pinned, so the junction is already rigid |
| 4 | bladder | its own group (self-collision off) |
| 9 | corpus \| tube, shaft, ovoids | the tandem is the corpus's own device; both are kinematic, so a contact between them could produce nothing |
| 10, 11 | vagina \| tandem, cervix \| tandem | the tube and shaft travel inside the **cervical canal** and the **vaginal lumen**, and neither lumen is meshed |
| 12 | vagina \| ovoids | the ovoids seat in the vaginal **fornices**, i.e. inside the same unmeshed lumen |

Everything else collides: cervix/vagina/bladder/rectum/sigmoid against each other, the corpus against
bladder/rectum/sigmoid, and the **ovoids against the cervix, bladder and rectum** (the portio seating and the OAR
push, which are the clinically meaningful device-tissue interactions).

The lumen exclusions are not cosmetic. The bodies are solid, voxel-exclusive volumes, so where the lumen should be
there is a closed surface, and contact simply forbids the tandem from ever entering: MEASURED (probe 7), the
advancing tube rams the cervix surface and crushes its elements (min volume ratio 0.94 at **0.48 mm** of
displacement, then inversion) while the corpus has not yet moved. The tandem acts on the cervix through the canal
dilation tie instead. `tandem_lumen_contact` / `ovoid_vagina_contact` re-enable the contacts.

The remaining rest-state overlaps (`bodies.json["pairs"]`) are left to contact: `cervix|bladder` -0.20 mm and the
sub-mm gaps `corpus|sigmoid` 0.05, `cervix|rectum` 0.16, `vagina|rectum` 0.19 mm are inside `contact_mm` = 0.5, so
the scene starts slightly out of equilibrium. The **pre-settle phase (P)** exists for exactly this: 2-4 steps with
the device parked, in which the contact offset relaxes (max nodal change decays 0.47 -> 0.02 mm, per-body max
displacement 0.14-0.66 mm) before any device motion. That relaxation is part of the reported displacement field.

### Couplings and supports

- **cervix -> corpus**: `AttachConstraint` (CONTRACT 4) between `/corpus/iface/mo` and the 127 `interface_corpus`
  nodes. The rigid-mapped copies are initialised **at the cervix rest positions** (the corpus rigid starts at
  identity, so its local coords are preBT world), so the constraint is a no-op at rest and introduces no snap.
  `attach_impl="springs"` swaps it for stiff RestShapeSprings to the same copies.
- **cervix canal -> tube** (`canal` 38 nodes): a one-sided **dilation** tie. A node engages while the tube occupies
  its axial level (0 <= s <= L_iu measured along the tube axis from the flange) and is pushed radially **outward**
  onto the tube surface (radius `r_tandem` + slack) - never pulled inward, and with **no axial component**, so it
  slides freely along the axis. The target is at most `canal_max_offset_mm` from the node's current position, which
  bounds the tie force at k*offset; the full dilation is still reached over several steps. This is the
  `ties.py`/`controller.py` distributed-node-tie idea with a RestShapeSpringsForceField to a controller-written
  target state, which is LDL-safe and puts no force on any other body.
- **vagina apex -> cervix** (`apex` 98 nodes): each apex node follows its nearest cervix surface node, offset by
  their rest difference (zero force at rest). **One-way**: no reaction on the cervix, so the two solvers stay
  decoupled and the thin vagina cannot destabilise the cervix.
- **cardinal ligaments** (`lateral_os_level` 168 nodes): a **cable** of natural length 25 mm to a static anchor
  25 mm lateral (sign from the internal-os x), tension-only. Force k*(|A - x| - L) along the ligament, so it resists
  lateral traction but offers almost no resistance to the cranial lift at first - which is the point of a ligament
  model rather than a spring-to-rest.
- **`vagina.fixed_inferior` (435), `rectum.fixed_ends` (103), `sigmoid.fixed_ends` (82)**: pinned.
- **`bladder.anterior_support` (377), `rectum.posterior_support` (237)**: soft springs to rest, k 2 mN/mm per node.

### Schedule

`u` in [0, 1] is the pose-rule path parameter: flange `F(u) = F_final - (1 - u)*travel*axis`, corpus screw motion
`s(u) = smoothstep((u - u_ios)/(1 - u_ios))` from `pose.json`'s screw decomposition. The reconstruction is checked
against the stored keyframes every run: `path_vs_pose_json_max_mm` = **0.000408 mm**. Phases: **P** pre-settle (2) ->
**A** approach, u 0 -> 0.39 (4) -> **T** insertion, u 0.39 -> 1 with the corpus screw motion (24) -> **D** ovoid
seating (0 in the default) -> **H** settle to convergence. Velocities are scaled by `settle_vel_scale` = 0 during H
(quasi-static relaxation) and by `motion_vel_scale` = 1 while moving.

CONTRACT 3.4 asks for N_steps 80 on the device path. The default splits 30 steps over the path instead and spends the
rest of the budget on the settle, because a step costs ~5.8 s (below) and one container invocation is capped at
`timeout 570`. Step size is not what limits fidelity here: 20 vs 24 insertion steps and 8 vs 20 seating steps change
the final interpenetration by < 1 mm (runs H4/H5/H6/H7).

### Convergence and the converged run (`H6`)

CONTRACT 5, made precise: 5 consecutive settle steps with max nodal change < 0.02 mm, **constraint residual per
contact** < 0.01, and the constraint solver inside its iteration cap. The GenericConstraintSolver's `currentError` is
an absolute sum over the active set and scales with the contact count (MEASURED 0.19 at rest with 85 contacts, 0.78
at the seated pose with 280 - 0.0022 and 0.0028 per contact), so the tolerance is normalised; the solver's own
convergence at tolerance 1e-3 is required as well.

| quantity | value |
|---|---|
| status / steps / settle steps | **converged** / 60 / 30 |
| ms per step (median, mean) | 5807 / 5679 ms (P 4749, A 4469, T 5419, H 6147) |
| total wall / scene init | 345.4 s / 3.9 s |
| max nodal change, last step | **0.0170 mm** (5 consecutive steps < 0.02) |
| decay ratio / projected remaining drift | 0.888 / **0.135 mm** |
| constraint residual, per contact, iterations | 0.533 / **0.00248** / 12 of 250 |
| contacts at convergence | 215 |
| min tet volume ratio (run) | **0.459** (abort threshold 0.2) |
| gates: NaN / inverted / wall time | false / false / false |
| reached final pose (u = 1, s = 1) | true |
| constraint solver inside its cap | 68.3 % of steps |

Final displacement, |u| max / mean (mm): corpus **25.98 / 20.93** (rigid, the pose rule's own motion), cervix
**25.38 / 20.74**, vagina **21.34 / 7.96**, bladder **5.53 / 2.47**, rectum **1.30 / 0.28**, sigmoid **9.51 / 3.40**.
The cervix mean tracks the corpus mean to 0.2 mm, i.e. it is carried by the attach rather than sheared off it; the
vagina's mean is much lower than its max because `fixed_inferior` holds its lower 15 mm while the apex follows the
cervix. Device at convergence: flange (-59.356, -41.604, 1.370) mm, tube axis (0.00296, 0.14858, 0.98890), tip
(-59.174, -32.422, 62.484) mm, Delta 22.5 mm, ovoid lag 0.

### Verification figures (viewed)

`figs/scene_H6_{initial,final,compare}.png`, two views each (anterior-right-superior oblique, left lateral).
- **initial**: the six bodies sit as in the meshing figures (corpus superior, cervix below it, vagina descending,
  bladder anterior, rectum posterior, sigmoid superior), and the device is entirely below the introitus with the
  ovoid assembly clear of all tissue - the correct u = 0 state.
- **final / compare** (rest wireframe vs converged solid): the corpus and cervix have risen ~21 mm, the tandem runs
  up the cervical canal into the corpus with its tip inside the corpus, the ovoids sit at the fornices under the
  portio, the vagina is stretched into a long thin tube with its inferior end pinned, the sigmoid is displaced
  superiorly and the rectum has barely moved. No body flies off, nothing is left behind, the corpus reaches the
  rule's target.
- Honest reading of the same figures: the ovoid assembly visibly overlaps the cervix above it and the rectum behind
  it, and the corpus fundus is inside the sigmoid. Those are the interpenetrations quantified next.

### Interpenetration at convergence (`logs/scene_H6_contacts.json`, mm, negative = overlap)

| pair | rest | final | reading |
|---|---|---|---|
| corpus \| cervix | -0.99 | **-0.99** | unchanged: they move together, as intended |
| cervix \| vagina | -0.87 | -0.81 | unchanged (excluded pair, apex follow) |
| rectum \| sigmoid | -0.57 | -0.57 | unchanged (both ends pinned) |
| tube \| cervix | - | -0.55 | BY DESIGN: the tandem is in the canal |
| ovoid \| vagina | - | -4.1 / -7.6 | BY DESIGN: the ovoids are in the vaginal lumen |
| cervix \| bladder | -0.20 | **-2.29** | residual, contact lost |
| vagina \| bladder | +4.52 | **-1.36** | residual, contact lost |
| ovoid \| cervix | - | **-6.97 / -4.58** | structural, see below |
| corpus \| sigmoid | +0.05 | **-11.04** | structural, see below |
| cervix \| rectum, bladder \| rectum, vagina \| rectum, corpus \| bladder, cervix \| sigmoid | | +14.9, +17.4, +2.35, +0.52, +0.50 | separated or in clean contact |

Two of these are **structural, not numerical**, and were diagnosed by their reproducibility: they are identical to
three decimals across runs with 20 vs 24 insertion steps, 8 vs 20 seating steps, alarm 1.5 vs 2.0 mm and a 250 vs
1000 iteration cap (H4, H5, H6, H7).
- **ovoid | cervix ~ -7 mm.** A geometric consequence of the pose rule at this Delta: the rule puts `L_end` only
  `d_F` = 17.5 mm above the flange, and the ovoid top *is* the flange plane, so the ovoid top lands ~4 mm **above**
  where the rigidly carried external os sits. The cervix is pinned to the kinematic corpus at its upper interface
  and cannot retreat, so the overlap is the portio compression the model has no way to represent. It is the model's
  statement that Delta = 22.5 mm seats the ovoids into the portio.
- **corpus | sigmoid ~ -11 mm.** The kinematic corpus rises ~21 mm into the sigmoid, which is pinned at **both** cut
  ends and can only bulge; it yields 9.5 mm and is overrun for the rest. A real sigmoid slides out of the pelvis on
  its mesentery; ours cannot, because CONTRACT 1 fixes both cut ends. `k_fixed_mN_per_mm` is the knob.
- The two **residual** overlaps (cervix|bladder, vagina|bladder, 1-2 mm) are ordinary lost contacts: on the 32 % of
  steps where the constraint solver hits its cap the penetration grows past `alarm_mm`, and proximity detection
  never sees that vertex again. Raising the cap to 1000 moved the converged fraction 0.68 -> 0.76 and the
  penetration by < 0.1 mm, so the cap is not the binding constraint.

### Deviations from CONTRACT 4, each forced by a measurement

1. **`rayleighStiffness` 0.1 -> 0.** With implicit Euler and a negligible tissue mass a step solves
   `(dt*C + dt^2*K) dv = dt*f`, reaching only `dt/(rayleigh + dt)` = **17 %** of the elastic response per step. The
   cervix bulk then chronically lags the rigidly attached corpus interface and the one-element transition layer
   absorbs the whole drag: runs H1/H2 inverted at corpus s = 0.28 with only 4.2 mm of displacement. At 0 each step
   is essentially the static solve, which is what a quasi-static scene wants, and the same path completes with min
   volume ratio 0.46.
2. **"fixed" node sets are stiff springs (1e4 mN/mm), not `FixedConstraint`.** With hard constraints the rest-state
   contacts that land on fixed nodes have zero compliance, so the Gauss-Seidel divides by w = 0: the residual and
   every multiplier came back **NaN from step 0** (run HSMOKE). `fixed_impl="constraint"` restores the contract
   wording.
3. **Cardinal ligaments are a controller-written one-sided cable, not `StiffSpringForceField`.** A stiff spring to a
   massless, solver-less anchor object blew the cervix up by **35 mm in the first rest-state step**; the anchor DOFs
   enter the cervix solver's system with a singular mass matrix. The cable formulation applies the same force and
   puts nothing on the anchors.
4. **`LinearSolverConstraintCorrection` is added last in each body node**, after the supports and couplings. Omitting
   it entirely (the first version did) leaves the constraint solver with no compliance at all: W = 0, NaN residual
   and multipliers, and **contacts produce no force whatsoever** while the scene still looks like it is running
   (probe 5: every body's max displacement exactly 0.0000 with 85-137 contacts detected).
5. **Corpus and device are two rigid bodies** (the rule gives them different motions until u = 1).
6. **Lumen contacts excluded** and **`ovoid_mode="travel"`** (contact matrix section). In the alternative `"seat"`
   mode the ovoids materialise inside the vagina at the start of the seating phase and the vertices they enclose are
   further from the ovoid surface than `alarm_mm`, so they are never detected again: 5-7 mm of permanently trapped
   tissue, identical whether the seating takes 8 or 20 steps, which is how the teleport rather than the step size
   was identified as the cause.
7. **Convergence residual normalised per contact** (CONTRACT 5 section).
8. **Step counts** (schedule section).

### Limitations

- **No lumina.** The cervix, vagina and sigmoid are solid voxel-exclusive volumes, so the canal, the vaginal lumen
  and the bowel lumen do not exist in the mesh. The tandem and the ovoids therefore occupy space that is filled with
  tet material, handled by excluding those contacts and dilating the canal with a tie. Every device-tissue
  interaction except the portio/OAR contact is a modelling convention, not a computed contact.
- **The vagina is the least trustworthy body**, as the meshing module warned (5.5 cc, ~3 mm thick, 1-2 tets across).
  It survives (min volume ratio 0.46 over the run, no inversion) and its motion is plausible, but it is driven
  almost entirely by the one-way apex springs and its own fixed inferior end rather than by device contact.
- **The corpus is kinematic and never yields**, so wherever an OAR cannot get out of its way within its own boundary
  conditions it is simply overrun (sigmoid, -11 mm). This is a property of the hybrid design, not a solver failure.
- **Forces are not validated.** `lambda_abs_sum` / `lambda_max` in `log.jsonl` are the raw Lagrange multipliers of
  the contact constraints in SOFA's FreeMotion formulation; they are logged for monitoring and are not calibrated
  force measurements. The material parameters are ASSUMED throughout.
- **One case, one Delta.** Everything above is a single converged run at the in-sample Delta = 22.5 mm. Delta is a
  scenario parameter: `--cfg '{"flange_shift_mm": 30}'` selects any value present in `pose.json`.
- The settle shows isolated 0.3-0.5 mm cervix/bladder contact slips every ~10 steps between smooth decays; H6
  converged in the window after the last one, so the converged state sits at the end of a decaying sequence
  (ratio 0.89) rather than at a perfectly stationary point. The projected remaining drift, 0.135 mm, is the honest
  bound on that.

## ANIMATION (`run_hybrid.py` frame export + `animate_hybrid.py`, CONTRACT section 4 outputs)

The live `runSofa` viewer repaints once per solve step (~6 s), so it cannot be watched. This module exports the
deformed state of every body at every step and renders it off-screen into an MP4/GIF. Two pieces: a **per-step
export** inside the container (`run_hybrid.write_frame`, cfg `frame_every`) and a **host renderer**
(`animate_hybrid.py`, `py -3.11` for pyvista). Frame preBT world RAS mm throughout; no patient-derived coordinate is
hard-coded (the device, the os landmarks and the colours are read from `pose.json`, `applicator.json` and
`meshes/bodies.json`). The MRI is read read-only and every output stays under `MRI_GYN_sim/hybrid/`.

### Export contract — `runs/<tag>/frames/` (cfg `frame_every`, int, **default 0 = off**)

`frame_every > 0` writes, every `frame_every` steps **and at the final step**:

| file | content |
|---|---|
| `step_XXXX_<body>.obj` | the deformed surface of **every** body (all six, corpus included), preBT world RAS mm, `%.3f` (1 um). Same faces and same vertex order as `meshes/<body>/surface.obj`, so a frame and the rest mesh are vertex-comparable |
| `step_XXXX_device.json` | `phase`/`phase_name`, `u`, `corpus_s`, `flange_mm`, `tube_axis`, `path_axis`, `x_app`/`y_app`, `tip_mm`, `ovoid_origin_mm`, `ovoid_lag_mm`, `ovoid_centres_mm`, inserted depth as `advance_mm` / `remaining_mm` / `tip_beyond_O_pre_mm` / `tip_beyond_internal_os_mm`, per-body `disp{umax_mm, umean_mm}`, `n_contacts`, `n_constraint_rows`, `constraint_err_per_contact`, `n_canal_ties`, `dx_max_mm`, `min_vol_ratio`, `wall_ms` |
| `index.json` | the frame list plus the run constants: `R_rows` (device surfaces: `p_world = origin + p_app @ R_rows`, origin = `flange_mm` for tube/shaft, `ovoid_origin_mm` for the ovoids), `flange_shift_mm`, `travel_mm`, `u_tip_at_internal_os`, `bodies`, `skipped_bodies` |

Read-out only: the mechanical states are copied, never written, so the export cannot change the solve (MEASURED:
`HA1` reproduces `H6` exactly — same 60 steps, same `dx_max_final` 0.01701 mm, same drift 0.135 mm, same min volume
ratio 0.4589). The **corpus** has no mechanical DOFs, so its surface is the rigid pose-rule placement
`X0 @ T_corpus^T`, and its `disp` is computed from the same transform. Faces and the surface-vertex -> tet-node map
are read **once** per run (`frame_cache`); a body whose stored surface does not index its tet mesh is skipped, with
the reason recorded in `index.json[skipped_bodies]` and `summary.json[frames][skipped_bodies]`, so a scene variant
that rebuilds one body (e.g. a hollow vagina wall) still exports the other five instead of aborting the run.

`frame_every` is consumed by `run_hybrid.py`, **not** by `scene_hybrid.CFG`, so `load_cfg` lists it in
`_unknown_keys` in `cfg.json`. That is expected and harmless; the key is echoed in `summary.json[frames][every]`.

### Renderer — `animate_hybrid.py` (host)

Three modes, because the two host interpreters carry different packages (py3.13: nibabel/imageio; py3.11: pyvista):
`prep` (optional MRI backdrop, py3.13), `render` (PNG per frame + GIF, `py -3.11`), `mp4` (py3.13 + imageio-ffmpeg).
Each frame is **two panels**:

- **left** a sagittal-like view along the patient LR axis: camera on the patient's left looking +x, parallel
  projection, so **anterior is on the LEFT and superior is up**. The anatomy is **cut** at the device's own sagittal
  plane (`x_cut` = mid-point of the final intrauterine tube) and the near half removed, which is what makes the
  device inside the tissue visible at all (VERIFIED numerically: the bladder's 1180 surface points, x in
  [-100.0, -9.1] mm, become 777 points in [-59.3, -9.1] mm). `--no-clip` falls back to translucency.
- **right** an oblique 3-D view from the patient's left-anterior-superior, nothing cut, bladder 0.20 / sigmoid 0.28
  opacity so they do not hide the cervix and vagina behind them.

Both carry the six organs in the **scene's own colours** (`meshes/bodies.json`, identical to the `OglModel`
colours), the device in grey (tube/shaft near-black, ovoids light grey at **0.50 opacity in the cut panel** — an
opaque 39 mm ovoid assembly hides exactly the portio/fornix tissue the animation exists to show), and the **rest
state as a faint wireframe** in each body's colour, so the motion is visible. The camera is **fixed for the whole
run** (computed once from the rest + final surfaces and the device path), with 17 % headroom reserved at the top of
both viewports for the overlay — without it the third overlay line is drawn over the corpus and is unreadable.
Overlay per frame: step, phase, inserted depth (mm along the path, `u`, and tip past the preBT external os), max
|u| overall and per body, contact count, dx/step.

| option | meaning |
|---|---|
| `--tag` | run tag; `runs/<tag>/frames/` must exist |
| `--bodies vagina,cervix` | render a subset (device always drawn); anything the run did not export is dropped with a warning |
| `--zoom` | frame the **vagina / ovoid region** (vagina + cervix rest and final surfaces, the seated ovoid assembly +- 24 mm) instead of the whole pelvis |
| `--backdrop` | draw the preBT MRI sagittal plane (needs `prep` first); behind everything on the left, at its true position and 0.55 opacity on the right |
| `--no-clip` | do not cut the anatomy in the left panel |
| `--every N` / `--limit N` | render every N-th frame / only the first N (tests) |
| `--fps` (9) / `--hold` (1.5 s) | GIF and MP4 rate, and how long the final frame is held |
| `--size` (1600x800) | PNG size; multiples of 16 keep libx264 from rescaling |
| `--gif-max-mb` (12) | GIF size cap: shrinks to 80 %/62 % width and then drops to every 2nd/3rd frame until it fits |
| `--name` | output basename (default `<tag>`, `+_zoom`, `+_<bodies>`) |
| `--pk` | `pip install --target` dir holding `imageio-ffmpeg` (mp4 mode); `APPSIM_PK` is the env equivalent |

### Commands

```bash
cd D:/neve/.claude/worktrees/sofa_applicator/applicator_sim
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
# 1. the run, with per-step frames (container; frame_every 1 = every step)
bash run_docker.sh HA1 hybrid/run_hybrid.py --tag HA1 \
  --cfg '{"n_presettle":2,"n_approach":4,"n_insert":24,"n_seat":0,"max_settle_steps":40,"max_steps":75,"wall_limit_s":500,"frame_every":1}'
# 2. optional MRI backdrop plane (host, py3.13: nibabel)
python   hybrid/animate_hybrid.py prep   --tag HA1
# 3. render (host, py -3.11: pyvista off-screen) -> PNG per frame + GIF
py -3.11 hybrid/animate_hybrid.py render --tag HA1
py -3.11 hybrid/animate_hybrid.py render --tag HA1 --zoom --bodies vagina,cervix --name HA1_vagina
py -3.11 hybrid/animate_hybrid.py render --tag HA1 --backdrop --every 29 --limit 3 --name HA1_bd
# 4. MP4 (host, py3.13 + imageio-ffmpeg from a scratch --target install)
python -m pip install --target <scratch>/pk imageio-ffmpeg     # once
python   hybrid/animate_hybrid.py mp4 --tag HA1            --pk <scratch>/pk
python   hybrid/animate_hybrid.py mp4 --tag HA1 --name HA1_vagina --pk <scratch>/pk
```

### Outputs and costs (run `HA1`, converged, 60 steps)

| output | path | size / duration |
|---|---|---|
| exported frames | `runs/HA1/frames/` (421 files: 60 x 6 OBJ + 60 JSON + index) | 17 MB |
| PNG frames | `figs/anim/frames_HA1/step_XXXX.png`, 1600x800 | 60 files, 26.4 MB |
| **MP4 (preferred)** | `figs/anim/insertion_HA1.mp4` | **1.5 MB, 73 frames at 9 fps = 8.1 s** (final frame held 1.5 s) |
| GIF | `figs/anim/insertion_HA1.gif`, 1000x500 | 2.6 MB, 60 frames (under the 12 MB cap at full rate) |
| zoom, vagina + cervix | `figs/anim/insertion_HA1_vagina.{mp4,gif}` + `frames_HA1_vagina/` | 1.2 MB / 2.0 MB, 8.1 s; PNG 15.3 MB |
| backdrop demo | `figs/anim/frames_HA1_bd/`, `insertion_HA1_bd.gif` | 3 frames, 2.0 MB |
| backdrop plane | `figs/anim/prep_HA1/{backdrop.npz, backdrop_preview.png, prep.json}` | 61 KB, 170 x 229 px (190 x 365 mm) |
| render metadata | `figs/anim/frames_<name>/frames.json` (camera box, `x_cut`, gif/mp4 records) | - |

**Cost of the export: 37.3 ms per frame (median), 2.5 s over the whole run** against 5903 ms per solve step, i.e.
**0.7 %** of a 356.8 s run (`H6`, the same configuration without frames, took 345.4 s). Rendering costs ~2.5-3 min
for 60 frames at 1600x800 (dominated by pyvista actor rebuilds, ~3 s/frame); `prep` and each `mp4` take seconds.

### What the animation actually shows (frames viewed: 0, 20, 29, 59, plus the zoom and the backdrop frame)

- **step 0 (pre-settle, u = 0)**: the six bodies in their preBT rest pose, solid surfaces sitting exactly on their
  rest wireframes, and the whole device below the introitus with the ovoid assembly clear of all tissue.
- **step 20 (insertion, u = 0.77)**: the tandem has advanced up the cervical canal into the corpus, which has risen
  17.8 mm; the ovoids are still low in the vagina and the vagina is being drawn upward with the cervix.
- **step 29 (u = 1.00, ovoids seated)**: the corpus has risen 26.0 mm off its wireframe, the tandem runs the full
  length of the canal with its tip inside the corpus, and the ovoids sit at the fornices under the portio. The
  translucent ovoids let the cervix and vagina behind them be seen.
- **step 59 (converged, settle 30)**: visually almost identical to step 29 (dx 0.017 mm/step) — the settle is a
  relaxation, not a further motion, which is exactly what the convergence numbers claim.
- **`--zoom --bodies vagina,cervix`**: this is the honest picture of the Stage-1 vagina. The solid vagina body is
  **dragged up beside the applicator rather than opened around it** — the device never enters it, because the
  collapsed-lumen body has no lumen and the `vagina|tandem` / `vagina|ovoids` contacts are excluded (SCENE section).
  The purple vagina visibly interpenetrates the grey ovoid (the -4.1 / -7.6 mm overlap of the interpenetration
  table), and its motion is the apex springs following the cervix, not device contact. This is the limitation the
  Stage-2a expanding wall (`VAGINA_WALL.md`) exists to remove, and the animation makes it obvious.

### Caveats

- **The left panel is a cut, not a slice.** It is a 3-D surface rendering with the patient-left half removed at the
  device's sagittal plane, so what looks like a "cut face" is the inside of the far half of each surface. It is not
  a tomographic slice and distances read off it are projections.
- **`--backdrop` leaves black bands above and below the image.** The preBT MRI covers 104 x 1.6 mm = 166 mm in z,
  while the animation's view spans ~233 mm because the device path starts ~90 mm below the anatomy, outside the
  scan. The bands are the absence of data, not a rendering fault. The backdrop is also a **single sagittal column**
  (x = -58.85 mm here, the tube leaves that plane by 0.09 mm over its length), so organs off the midline are drawn
  in front of a slice that does not contain them.
- The GIF is 1000x500 and 8-bit palettised; the **MP4 is the deliverable** and the GIF is the fallback for viewers
  that will not play video. At the default cap both come out at full frame rate for a 60-step run; a longer run
  will start dropping frames (recorded in `frames.json[gif][every]`).
- The overlay's `max |u|` is the maximum over the **rendered** bodies only, so it changes with `--bodies`.
- The animation shows the run's own kinematics and displacements; it is **not** an evaluation. Nothing here is
  compared with the BT anatomy — that is `eval_hybrid.py`'s job.

## VAGINA WALL (`vagina_wall.py` + `scene_hybrid.py` cfg `vagina_model`, Stage 2a / `VAGINA_WALL.md`)

Stage 1 meshed the vagina as a SOLID rod from the collapsed preBT label, so the applicator could not pass through it
(ANIMATION section: "dragged up beside the applicator rather than opened around it"). This module replaces that body
by a **hollow annular wall around a lumen**. Frame preBT world RAS mm; units mm / kg / s (kPa, mN). No patient-derived
coordinate is hard-coded: the axis comes from `applicator/pose.json` and every cross-section from the preBT labels.

**Headline, stated up front.** The wall model now runs the **full insertion path** (u = 1.0, corpus screw complete) and
gives the **best vagina surface distance in the project** — PELVIS MSD **11.50 mm** against Stage-1 `H6`'s 15.40 and
`NONE`/`K0`'s 16.67, i.e. **dMSD −5.11 mm [−5.69, −4.64]**. **But the lumen still does not dilate**, and the reason is
now *measured* rather than guessed: two confirmed mechanisms, neither a solver failure. Five separate hypotheses were
tested and refuted along the way; they are tabulated below so nobody retries them.

### Status against the milestones

- **M1 (shaft alone travels the lumen and the wall opens): PARTIAL.** The mesh, node sets, wiring and contact are
  verified, and the shaft does travel *inside* the lumen (`Z7S`: 100 % of shaft surface vertices inside, 0 outside).
  The wall opens only locally and transiently (peak circumferential stretch **1.129** at u = 0.42 in `Y5`); it is not
  an opening sustained to the final pose.
- **M2 (full device with the ovoids, `lumen_r0_mm` swept): NEGATIVE, and the negative is the result.** r0 was swept
  2 / 3.5 / 5 / 7 mm. The ovoids never enter the lumen at any r0 — see Mechanism 2 — so the milestone as written is
  not reachable with `ovoid_mode="travel"`.
- **M3 (scoring): DONE**, including `field`, which previously could not run for a wall run at all.

### Refuted by experiment — do NOT retry these

| # | hypothesis | the test | result |
|---|---|---|---|
| 1 | The wall is too thin (`n_radial = 1`); more radial layers will stop the inversion | `vagina_wall_nr2` / `nr3` builds, run `WA20` | **FALSE.** Volume is conserved but element quality gets *worse* (interior dihedral 25.7 → 19.0 → 12.9 deg) and the wall inverts **earlier** (`WA20` min volume ratio 0.025 at u = 0.239 vs `V1B` 0.19 at u = 0.266) |
| 2 | The lumen is dragged off the device axis, so the shaft presses through the wall sideways | `vagina_wall.py axis`, per step | **FALSE at the moment of failure.** The lumen centre is 0.59–1.6 mm off the device axis against 2.8 mm of clearance. The 9.3 mm migration is real but *late* and at the fornix end |
| 3 | The pinned introitus ring clamps the lumen shut | `WB_i5` (ring 5 mN/mm), `WB_out` (outer layer only), `WB_a5` / `WB_a500` (apex springs) | **FALSE as a fix.** All still abort (steps 22–30, u 0.24–0.33). Softening *delays* failure — `WB_i5` reached u = 0.334, furthest of that series — so the soft ring was kept, but it is not the cause |
| 4 | The bladder drifts into the wall and shears it (the wall is the failing body) | `XM1` wall/bladder contact removed; `XM2` bladder supports 2 → 20 mN/mm; `XM3` both; `XN*` re-test with ovoids | **FALSE, twice.** `XM1` reproduces the baseline to three decimals (min vol ratio 0.0251 vs 0.0253; bladder umax **identical** at 6.94 mm) and `XM2` is slightly *worse* (0.0171). Stiffening with ovoids present made it worse again (`XN5` u 0.489 vs `XO5` 0.560). And once the ovoids load the wall, the aborting body is the **bladder** (0.126–0.162) while the **wall is healthy** (0.62–0.89) |
| 5 | The 30 mm shaft **arc** (sagitta 6.19 mm) cannot lie inside a 4–5 mm lumen, so the device never engages | straight rod built (`vagina_wall.py shaft`) + **`Z5A` arc control**, single variable | **FALSE.** With signed containment the **arc** shaft is 63.4 % of vertices inside the lumen (5.4 % outside); the straight rod 51.9–100 %. `Z5A` and `Z5S` stop at essentially the same place (u 0.941 vs 0.964). The sagitta argument ignores that the lumen bends *with* the wall and that the shaft spans only part of its length. The straight rod is kept (more physical; 100 % containment at r0 = 7) but it was **never the blocker** |

A methodological note on #5, because it nearly became a sixth wrong conclusion: the first containment metric reported
the **minimum** radius over device vertices, which finds the innermost point rather than testing containment — a rod
straddling the wall scores as "inside", which made the arc look *better contained* than the straight rod. The metric
was corrected to count vertices (inside / embedded / outside) before anything was concluded from it.

### Confirmed mechanism 1 — axial Poisson necking (most of the missing dilation)

The cervix lifts the vaginal apex **21.5–25.4 mm** while the introitus ring is held, stretching the tube **axially**
over its 54 mm length: measured per-station axial stretch **λ = 1.18–1.34** (mean ≈ 1.30). An incompressible tube
stretched by λ thins by 1/√λ:

| | predicted | measured |
|---|---|---|
| radius ratio at λ = 1.30 | 1/√1.30 = **0.877** | raw radius 4.31–4.48 mm ÷ 5.00 mm ref = **0.862–0.896** |

So the narrowing is **not** the device crushing the lumen; it is the wall obeying incompressibility while being pulled.
`log.jsonl["wall"]["containment"]` therefore reports the lumen radius **both raw and normalised** as `r·√λ`.
Normalised, `Z5S` sits at **4.95–5.55 mm against a 5.00 mm reference** — roughly neutral, with a modest **+10–16 %
dilation at the fornix stations** where the ovoids bear. Quoting the raw radius alone makes a correctly-behaving
material look like a failure to dilate.

### Confirmed mechanism 2 — the ovoid assembly is ~2.5× the whole tube (the hard blocker)

| quantity | value |
|---|---|
| ovoid assembly diameter (`applicator.json ovoid_diam_mm`) | **39.39 mm** |
| modelled vagina OUTER diameter, r0 = 7 (largest swept) | 2 × (7.0 + 1.05…2.35) = **16–19 mm** |
| ovoid surface vertices OUTSIDE the wall at the final pose (all Z arms) | **98.6–98.9 %** |
| nearest ovoid vertex radius vs wall outer radius, fornix stations | **11–18 mm** vs **6.2–8.0 mm** |

The ovoids do reach the wall and stay in contact from u ≈ 0.06 onward (`ovoid_gap_mm` 0.10–0.50 mm, `in_contact` true),
so this is **not** a "the device never touches it" failure. They **envelop the tube and compress it from outside** —
far too large to thread a 10–16 mm lumen. No stepping, stiffness, mesh or shaft change can alter that; it is
arithmetic. **`ovoid_mode="travel"` cannot seat these ovoids through this lumen.**

### Stage-2a simplification: static obstacle organs (cfg `static_bodies`)

With the ovoids loading the wall, every abort was triggered by the **bladder** (5234 nodes, E 8 kPa, ν 0.49 — linear
corotational tets in the near-incompressible locking regime) drifting 15 mm under its own 2 mN/mm supports with nothing
pushing it. `static_bodies` builds a body as a **non-deformable collision obstacle**: tet topology, the full-node
`dofs` MechanicalObject at rest and the surface collision models are kept (so `run_hybrid.write_outputs`,
`frame_cache` and the per-body log see identical shapes and read u = 0), but there is no ODE solver, mass, FEM,
support or constraint correction.

- **Probed before use** (`YSMOKE`): SOFA v22.12 **does** generate contact forces for `simulated=False` obstacles under
  FreeMotionAnimationLoop + GenericConstraintSolver — 56–75 live contacts, cervix and vagina responding, solver inside
  its cap on 100 % of steps.
- **Cost in wall time: 13 500 ms/step → 554 ms/step (~24×)**, by dropping 9241 of 12 549 deformable nodes.
- **COST IN MEANING, to be quoted wherever those organs appear:** a frozen organ cannot deform out of the device's way,
  so its predicted displacement is **identically zero** and any **device-to-OAR distance measured against it is a
  rest-state distance, not a prediction**. In `Z7S` the bladder / rectum / sigmoid rows equal the `NONE` comparator to
  three decimals **by construction**. Only the **vagina and cervix** rows of a fully-static run are predictions.
  `Z5SB` (bladder frozen only) is the fairer arm for any organ-displacement claim.

### Runs (selected; full set in `runs/<tag>/summary.json`)

| tag | what | steps | ms/step | status | final u | min vol ratio (run) | weakest body at final step |
|---|---|---|---|---|---|---|---|
| `V1` | Stage-1 schedule, shaft only | 64 | 7857 | settle_not_converged | 1.000 | 0.607 | vagina |
| `V1B` | 1.05 mm/step, alarm 3 mm | 23 | 11497 | abort_inverted_tets | 0.266 | 0.1915 | vagina 0.19 |
| `WA20` | `n_radial` 2 baseline | 22 | 7620 | abort_inverted_tets | 0.239 | 0.0253 | vagina 0.025 |
| `WB_i5` | introitus ring 5 mN/mm | 30 | 10255 | abort_inverted_tets | 0.334 | 0.1045 | vagina 0.10 |
| `XM1`/`XM2`/`XM3` | bladder-contact / support tests | 22 | ~9330 | abort_inverted_tets | 0.239 | 0.025 / 0.017 | vagina |
| `XO5` | **ovoids + soft ring**, r0 5 | 49 | 13514 | abort_inverted_tets | 0.560 | 0.151 | **bladder 0.151** (vagina 0.866) |
| `XN5R` | + supports stiffened (fallback) | 43 | 11749 | abort_inverted_tets | 0.489 | 0.178 | bladder 0.178 |
| `Y5` | **static bladder**, r0 5 | 111 | 3697 | settle_not_converged | **1.000** | 0.339 | rectum 0.345 (vagina 0.698) |
| `Y3P5` | static bla+rec+sig, r0 3.5 | 111 | 1367 | settle_not_converged | **1.000** | 0.375 | cervix 0.611 (vagina 0.689) |
| `Z5A` | **arc-shaft control**, r0 5 | 81 | 1034 | abort_inverted_tets | 0.941 | 0.085 | vagina 0.085 |
| `Z5S` | straight shaft, r0 5 | 83 | 946 | abort_inverted_tets | 0.964 | −0.117 | vagina −0.117 |
| `Z5SB` | straight, static bladder only | 111 | 3439 | settle_not_converged | **1.000** | 0.356 | rectum 0.381 (vagina 0.728) |
| **`Z7S`** | **straight, static bla+rec+sig, r0 7 — WINNER** | **111** | **1250** | settle_not_converged | **1.000** | **0.216** | cervix 0.610 (**vagina 0.818**) |

No run has met CONTRACT 5 convergence: the settle is still creeping (`Z7S` dx 0.077 mm/step at step 110, projected
drift infinite). The insertion path completes; the relaxation does not.

### Lumen: raw vs axial-stretch-normalised (the number that matters)

| run | r0 | raw mean r at u = 1 | axial stretch λ | **normalised r·√λ** | local max r | peak stretch anywhere |
|---|---|---|---|---|---|---|
| `Y5` | 5.0 | 4.38 mm | ≈1.30 | ≈4.99 mm | 5.23 mm | **1.129** (u = 0.42, transient) |
| `Z5S` | 5.0 | 4.31–4.48 mm | 1.18–1.34 | **4.95–5.55 mm** | 6.52 mm | 1.071 |
| `Z7S` | 7.0 | 6.2–6.9 mm | ≈1.3 | ≈7.0 mm | 7.26 mm | 0.997 |

Read this as: **the lumen holds its reference calibre once necking is removed, and bulges ~10–16 % locally at the
fornices. It never approaches the ~4× dilation an ovoid would need.**

### M3 — scoring the winner (`Z7S`), Dice / MSD mm / HD95 mm

| structure | **`Z7S` (wall)** | `Z5SB` (wall, bladder-only static) | Stage-1 `H6` (solid) | `NONE` | `K0@22.5` |
|---|---|---|---|---|---|
| vagina (PELVIS) | 0.102 / **11.50** / 25.68 | 0.103 / 12.69 / 26.60 | 0.101 / 15.40 / 31.68 | 0.104 / 16.67 / 30.87 | 0.104 / 16.67 / 30.87 |
| vagina+device (PELVIS) | 0.374 / **7.21** / 23.73 | 0.376 / 7.75 / 24.78 | 0.377 / 8.11 / 26.32 | 0.391 / 7.93 / 25.88 | 0.391 / 7.93 / 25.88 |
| cervix (PELVIS) | 0.626 / 4.25 / 14.49 | 0.613 / 4.42 / 14.80 | 0.641 / 4.12 / 14.62 | 0.206 / 10.59 / 23.77 | 0.570 / 4.91 / 15.75 |
| corpus (PELVIS) | 0.811 / 2.21 / 4.91 | 0.811 / 2.21 / 4.91 | 0.811 / 2.21 / 4.91 | 0.262 / 10.36 / 21.46 | 0.811 / 2.21 / 4.91 |
| vagina (E_app) | 0.104 / **10.90** / 21.14 | 0.103 / 12.10 / 21.84 | 0.103 / 14.22 / 28.15 | 0.103 / 15.40 / 26.69 | 0.103 / 15.40 / 26.69 |

`compare --tag Z7S` (30 frame-perturbation draws, ±0.5 mm equivalence), dMSD median [5–95 %]:

| structure | SIM minus NONE / K0 (PELVIS) |
|---|---|
| **vagina** | **−5.11 [−5.69, −4.64] sim better** |
| vagina+device | **−0.74 [−0.81, −0.52] sim better** |
| cervix | −0.57 [−0.94, −0.39] sim better |
| rectum / corpus | +0.00 — **frozen or kinematic, not a prediction** |

This is the largest vagina improvement in the project (Stage 2a's previous best was −2.62, Stage 1's solid body
−1.40), and `vagina+device` moves off its long-standing tie for the first time. **It is still not evidence that the
lumen opened** — it did not. It is the hollow rest shape plus the wall being carried correctly along the full
insertion. Dice stays pinned at ~0.10 for the reason the EVALUATION section gives: the BT vagina label is 99.9 cc
because it contains the device **and the packing**, against ~5.5 cc of modelled tissue.

`field --tag Z7S` **now runs for a wall run** (it previously exited with "vagina: tets.vtk has 1503 points,
vagina_u.npy has 1120"); `eval_hybrid.run_mesh_dirs` resolves the vagina mesh per run from `cfg.json`. Warped-label
self-check: cervix 0.973, vagina 0.835, frozen organs 1.000, `pass_all = False` (the vagina check is below the 0.9
gate — expected for a body whose rest mesh is an annulus, not the label).

### Animation — and what the frames actually show (viewed)

`figs/anim/insertion_Z7S_vagina.mp4` (1.4 MB, 69 frames at 9 fps, 7.7 s) + `.gif` + `frames_Z7S_vagina/`.
Companion renders exist for `Z5SB`, `Y5`, `XO5` and `XO7B`.

Rendering a wall run honestly needs two substitutions, because `animate_hybrid.py` is owned by the ANIMATION module
and was not modified: it reads the rest state from `meshes/<body>/surface.obj` (still the Stage-1 SOLID vagina) and
the device from `applicator/<part>.obj`. Both are redirected by running it with `APPSIM_OUT` pointed at a shadow tree
whose `meshes` is a junction to `meshes/_scene_<wall_dir>/` and whose `applicator` is a junction to
`applicator_straight/` for runs that used the straight rod. Verified per render by vertex count: rest vagina **1120**
(the wall) not 710 (the solid), shaft **386** (straight) not 722 (arc).

Frames 20, 60 and 110 of `Z7S` viewed directly:

- **step 20 (u = 0.23, approach):** the vagina sits on its rest wireframe (max |u| 0.5 mm) and the grey ovoid dome is
  already **visibly wider than the purple tube**, wrapped around its base.
- **step 60 (u = 0.70, insertion):** the ovoid mass **envelops** the upper tube — the purple wall passes behind and
  through it, not around it — while the cervix rises away from its wireframe.
- **step 110 (u = 1.0, settled 25 steps):** the tube is long and slender and lies **inside its own rest wireframe**,
  i.e. visibly narrower than it began, with the ovoids a broad band around the fornix and the tandem up the canal.

**So the animation shows the lumen necking, not opening, and the ovoids unmistakably outside the tube.** Picture and
numbers agree; nothing here is a rendering artefact.

Two more renders of `Z7S` exist (2026-09-13), made because the zoomed video above hides the other organs (`--bodies
vagina,cervix`) and, with an opaque wall, cannot show the shaft *inside* the tube:

- `figs/anim/insertion_Z7S_full.{mp4,gif}` + `frames_Z7S_full/` — the standard two-panel render with all six bodies,
  through the same shadow tree (straight rod, wall rest state).
- `figs/anim/insertion_Z7S_lumen.{mp4,gif}` + `frames_Z7S_lumen/` — **`animate_lumen.py`**, a second renderer written
  for wall runs only (it needs `meshes/vagina/meta.json["wall"]`, i.e. the shadow `APPSIM_OUT`). Each 1920 x 800 frame
  has: LEFT a see-through side view (wall at opacity 0.30 + its wireframe, tube/shaft opaque black, ovoids grey 0.45,
  cervix/corpus faint, nothing cut) with four station rings highlighted; MIDDLE a straightened coronal silhouette —
  at every one of the 28 stations the left-right extent of the inner and outer ring and of each device part, each
  station about its own lumen centre, rest state dashed; RIGHT four true cross-sections (planes normal to the lumen
  axis at stations 4, 11, 18, 25 = s −18, −4, +10, +24 mm) with the wall annulus, the rest annulus, the device
  section (convex hull of the plane cut) and a verdict **read off those same sections**: `inside` (every device
  polygon within the inner ring), `crossing the wall`, `outside, beside the wall`, or `wall INSIDE the ovoid solid`
  (the union of the device polygons contains the whole outer ring). The run's own logged shaft containment is quoted
  under the panels.
  The verdict rule deliberately differs from `wall_metrics`' nearest-vertex rule: for a closed ovoid surface cut
  at mid-height the nearest *vertex* is on the rim (r ≈ 18 mm), so the log reports the ovoids "outside the wall"
  at levels where the ovoid *solid* in fact contains the entire annulus — the wall passes through the ovoid. The
  section verdict says so; the logged shaft numbers (inside 10/10 at the end) are unaffected because the shaft is
  thin. The tube, the shaft and the ovoids are judged separately, because they behave differently.
  Frames 20, 40, 60, 80 and 110 viewed:
  - **the shaft proper is inside the lumen at every level it reaches** (r_max 3.0–6.2 mm vs lumen 6.0–7.0; step 60
    is the grazing phase the log reports as 1 of 15 stations);
  - **the intrauterine TUBE crosses the anterior wall of the upper vagina during the approach** (step 20, u = 0.23:
    crossing the wall at s = +10 mm, 16–21 mm anterior of the lumen at s = +24 mm). The tube is angled ~24° anterior
    of the vaginal axis (`tube_axis` vs `path_axis`) and is **not** in `wall_contact_parts` (shaft, ovoid_L, ovoid_R
    only), so nothing stops it passing through the fornix wall before the cervix carries the apex up. *This is what
    reads as "the shaft breaking out of the vagina" in the side views.* Not fixed (user instruction 2026-09-13);
    the fix is a tube|vagina_outer contact pair, cheap to add;
  - ovoids crossing the wall or engulfing it from s ≈ −16 mm (step 40) to the apex (step 110);
  - lumen rings inside their dashed rest rings (necking) throughout.

### Commands

```bash
cd D:/neve/.claude/worktrees/sofa_applicator/applicator_sim
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
python   hybrid/vagina_wall.py build --sweep          # r0 2/3.5/5/7 -> meshes/vagina_wall_r*
python   hybrid/vagina_wall.py build --r0 7 --variant xo7b --set n_theta=32
python   hybrid/vagina_wall.py shaft                  # applicator/shaft_straight.obj (straight vaginal rod)
# the winner: full path, static obstacle organs, straight shaft, r0 7
APPSIM_TIMEOUT=1650 bash run_docker_par.sh Z7S hybrid/run_hybrid.py --tag Z7S --cfg /out/hybrid/runs/_cfg/Z7S.json
#   Z7S.json = {"vagina_model":"wall","vagina_wall_dir":"vagina_wall_z7s",
#               "wall_contact_parts":["shaft","ovoid_L","ovoid_R"],"ovoid_mode":"travel",
#               "static_bodies":["bladder","rectum","sigmoid"],
#               "device_part_files":{"shaft":"shaft_straight"},"E_kPa":{"vagina":10},"alarm_mm":3.0,
#               "vagina_inferior":"spring","k_vagina_inferior_mN_per_mm":5,
#               "n_presettle":2,"n_approach":34,"n_insert":50,"n_seat":0,
#               "max_settle_steps":25,"max_steps":115,"wall_limit_s":1500,"frame_every":2}
python   hybrid/vagina_wall.py report --tag Z7S       # lumen radius + circumferential stretch per station
python   hybrid/vagina_wall.py axis   --tag Z7S       # lumen centreline vs the device travel axis
python   hybrid/eval_hybrid.py score  --tag Z7S       # M3; compare and field also work now
# animation: shadow APPSIM_OUT so the WALL is the rest state and the STRAIGHT rod is drawn
APPSIM_OUT=<shadow> py -3.11 hybrid/animate_hybrid.py render --tag Z7S --zoom --bodies vagina,cervix --name Z7S_vagina
APPSIM_OUT=<shadow> python   hybrid/animate_hybrid.py mp4    --tag Z7S --name Z7S_vagina --pk <scratch>/pk
APPSIM_OUT=<shadow> py -3.11 hybrid/animate_hybrid.py render --tag Z7S --name Z7S_full         # all six bodies
APPSIM_OUT=<shadow> python   hybrid/animate_hybrid.py mp4    --tag Z7S --name Z7S_full  --pk <scratch>/pk
APPSIM_OUT=<shadow> py -3.11 hybrid/animate_lumen.py  render --tag Z7S                         # see-through + sections
APPSIM_OUT=<shadow> python   hybrid/animate_hybrid.py mp4    --tag Z7S --name Z7S_lumen --pk <scratch>/pk
```

Parallel arms **must** use distinct `vagina_wall_dir` values: sibling containers share the shadow root
`meshes/_scene_<dir>/` and a concurrency bug there once truncated `bodies.json`. Every batch here pre-built its roots
on the host and gave each arm its own directory.

### Caveats and limitations

- **Nothing is calibrated.** `E_vagina_kPa = 10`, ν 0.45, `lumen_r0_mm`, the apex / introitus spring stiffnesses and
  the static-organ choice are assumed or swept, never fitted. No force is validated.
- **The reference state is a modelling choice** (unfolded rugae), not a measured geometry — see above.
- **Frozen organs are not predictions** (see the `static_bodies` cost paragraph). In `Z7S` the bladder, rectum and
  sigmoid rows equal `NONE` exactly, and its device-to-OAR distances (rectum 0.0, bladder 0.0, sigmoid 0.0 mm) are
  rest-state overlaps, not predicted clearances. `Z5SB` keeps rectum and sigmoid deformable and is the arm to quote
  for those bodies.
- **No run converged per CONTRACT 5.** The insertion completes; the settle creeps at 0.077–0.58 mm/step.
- **Round annulus vs flat slit**: the volume-conserving circular wall leaves the label footprint laterally and starts
  inside the cervix (−5.9 mm) and rectum (−4.0 mm), so those two pairs are excluded from contact
  (`wall_outer_exclude`). The bladder is NOT excluded: it clears the wall by +2.5 mm at rest, and excluding it changed
  nothing (hypothesis 4).
- **`path_vs_pose_json_max_mm` is meaningless for wall runs** (35.6 mm): `run_hybrid.py` self-checks the path against
  `pose.json`'s **tube-axis** keyframes while a wall run travels the **shaft** axis. Not an error; no gate uses it.
- **`alarm_mm` 3.0 is load-bearing and has a side effect.** It is needed for the device contact to fire at ~1 mm/step,
  but it also switches on wall/bladder proximity at rest (contact count 43 → 122). That cost nothing here, but it is
  why the bladder looked implicated for two batches.

### Recommendation for Stage 2b

The travelling-ovoid route is closed by arithmetic, not by numerics. Two honest options:

1. **Seat the ovoids by growth in place** (`ovoid_mode="seat"`, ovoids scaled from ~0 to full size at the fornices over
   a ramp). This avoids threading a 39.4 mm assembly through a 16 mm tube. Note the Stage-1 trapped-vertex failure:
   seating must *grow*, never *teleport*, or enclosed vertices are never seen again by proximity detection.
2. **Build the reference lumen near the DISTENDED state** (r0 ≈ 15–20 mm, the post-insertion calibre) and treat the
   collapsed preBT vagina as unreachable from a continuum mesh. This is the honest continuum analogue of "the rugae
   unfold": the stress-free state becomes the opened tube and the preBT geometry is a *compressed* configuration.

Either way the axial necking is physical and will remain; it should be reported normalised, as here. A third, cheaper
step is to raise `max_settle_steps` and relax `settle_vel_scale` to chase CONTRACT 5 convergence on `Z7S`, which is
the only gate the winner still misses.
