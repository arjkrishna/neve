# Stage 2a — the vagina as an expanding hollow wall

Extends CONTRACT.md (read that first). Everything there still holds: frames (preBT world RAS mm), units
(mm, kg, s -> kPa, mN), read-only patient data, local only, code-only in the repo, no git operations.

## Why
Stage 1 modelled the vagina as a SOLID rod tet-meshed from the collapsed preBT label (5.49 cc, ~3 mm thick,
1503 nodes). The applicator therefore cannot pass through it: it travels beside the body while the ovoids
push the fornices, and the vagina is only dragged along with the cervix (~21 mm). Scored against the real
post-insertion vagina it gets Dice 0.10 / MSD 15 mm — the worst structure in the model, and the one whose
behaviour the user most wants to see. In reality the applicator is inside the vagina and distends it: the BT
vagina label is 99.9 cc including the device against 10.8 cc before.

## The model
Replace the solid body by a **hollow tube of wall tissue around a lumen**, so the device enters the lumen and
the wall parts and stretches around the shaft and ovoids.

### Reference (stress-free) state — the key modelling decision
The real vaginal wall is folded into rugae; opening it is mostly UNFOLDING, which costs almost no stress, not
tissue stretch. A continuum mesh cannot unfold, so a mesh built at the true collapsed radius would have to
stretch ~10x circumferentially at the ovoids and will invert. Therefore:

- The stress-free reference is an **already-unfolded tube**: lumen radius `lumen_r0_mm` (parameter, default
  5.0, sweep 2 / 3.5 / 5 / 7), i.e. larger than the anatomical slit.
- **Wall tissue volume is conserved** from the label, per axial station: with the measured cross-sectional
  area `A(s)` of the vagina body, `r_out(s) = sqrt(lumen_r0^2 + A(s)/pi)`. Report the resulting wall
  thickness profile; it will be ~1-2 mm at the default.
- Record this honestly in the README: the reference state is a modelling choice (unfolded rugae), not a
  measured geometry; `lumen_r0_mm` is a parameter to sweep, not a fitted value.

### Geometry
- Centreline: the vagina body's own principal axis is, by construction, the rule-v2 shaft axis
  (`pose.json.device_final.shaft_axis`), so the device travels straight up the lumen. Build the centreline
  from per-station centroids of the label projected on that axis, smoothed, and clamp its deviation so the
  lumen stays a function of the axis (report the deviation).
- Structured annular mesh (NOT TetGen): rings of `n_theta` (default 20) nodes x `n_radial+1` (default 2 layers)
  x `n_axial` stations (~1.5-2.5 mm spacing), parallel-transported frames along the centreline; each
  hexahedral cell split into 6 tets with a consistent diagonal so faces match. This guarantees quality and a
  clean inner/outer surface. Report min dihedral and the inverted-tet count (must be 0).
- Extent: from the introitus (inferior end of the label, which the FOV truncates at k=0) up to the fornix
  level, where the top ring attaches to the cervix.

### Node sets (same JSON contract as mesh_bodies.py `meta.json`)
`inner_surface` (lumen; contacts the device), `outer_surface` (contacts bladder / rectum / cervix),
`apex` (top ring(s), attached to the cervix as the Stage-1 `vagina.apex` was), `fixed_inferior`
(bottom ring(s), springs as in Stage 1), plus `surface_nodes`.

### Material
Large strain: use `TetrahedronHyperelasticityFEMForceField` (available in the image; probe the exact
`materialName` / `ParameterSet` syntax in v22.12 — NeoHookean or StableNeoHookean) with
`E_vagina_kPa` default 10, nu 0.45 (mu = E/(2(1+nu)), bulk k = E/(3(1-2nu))). Keep the corotational option
behind a cfg flag for comparison, and report which one converged.

## Scene changes (`scene_hybrid.py`)
- cfg `vagina_model`: `"solid"` (Stage 1, default off) | `"wall"` (new default for this work).
- The wall's INNER surface collides with tube, shaft and ovoids; the OUTER surface with bladder, rectum,
  cervix. Keep the apex attached to the cervix and the inferior ring sprung, as in Stage 1.
- The device starts inside the lumen at the introitus (the rule-v2 path already travels along the shaft axis).

## Milestones — report each honestly, partial success is an acceptable outcome
- **M1 (must)**: wall mesh built and verified (volume vs label, thickness profile, quality, node sets), and
  a SOFA run in which the **shaft alone** (r 2.2 mm) travels up the lumen and the wall opens around it,
  converged per CONTRACT §5. This alone demonstrates the mechanism.
- **M2 (target)**: the full device including the ovoids (assembly ~39 mm across) with `lumen_r0_mm` swept.
  Expect this to be hard: circumferential stretch is ~4x at the default. If it will not converge, try the
  gentler `ovoid_mode="seat"` (ovoids grow in place with a ramp) and a larger `lumen_r0_mm`, and REPORT the
  largest stretch and the failure mode rather than tuning until something passes.
- **M3 (if M2 works)**: score the run with `eval_hybrid.py score/compare/field --tag <tag>` and report the
  vagina numbers against Stage 1's Dice 0.10 / MSD 15.4 mm (PELVIS) and against NONE / K0.
