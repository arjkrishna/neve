# TopBrain grafted anatomies: how the 25 meshes were built

Replaces the synthetic sinusoidal RCCA perturbation (`RCCAVariedFromMesh`) with
real patient internal carotid arteries from the TopBrain 2025 release, grafted
onto the shipped host tree.

This document is the build record: what each stage does, and every defect found
and fixed along the way. The fixes are the point. Most of them were invisible to
the check that was in place when they were introduced, and several were only
caught because a later, stricter check was written.

**Result:** 25 anatomies at `topbrain_data/anatomies/`, of which **24 are usable**
and `topcow_mr_015` should be excluded (see [Known failure](#known-failure)).

---

## Why graft at all

No public dataset covers the full route this project navigates, from the CCA
ostium to the carotid siphon. TopBrain's ICA segmentations begin at the
atlantoaxial level (MR) or intradurally (CT), so a TopBrain case cannot be used
whole. The shipped host tree does have the proximal route but only one siphon,
which is why it was being varied procedurally in the first place.

Grafting takes the half each source actually has: the host supplies the arch,
trunk and cervical vessel; TopBrain supplies 25 genuinely different siphons.
The alternative considered and rejected was the VMR case `0248_H_AOCERE_CAS`,
the only anatomy found spanning aortic root to A1/M1/P1, but it is a single
patient with artificial stenosis variants, so it suits a held-out test case
rather than a training set.

---

## Pipeline overview

```
TopBrain_batches1n2.zip (1.9 GB)
   └─ labelsTr_topbrain_mr/  25 MR label masks (.nii.gz)
        │
        │  stage A   mask_to_surface.py        base conda env
        ▼
   surfaces/  *_rICA.vtp + *_rICA_seeds.json
        │
        │  stage B   vmtk_centerline.py        vmtk_env conda env
        ▼
   centerlines/  *_ica.json   points + MISR radii, proximal → distal
        │
        │  stage C   graft_siphon.py           host repo env
        ▼
   anatomies/<case>/Centrelines_comb/*.mrk.json
        │
        │  stage D   bake_meshes.py            inside the container
        ▼
   anatomies/<case>/vessel_architecture_collision.obj
```

Stages A and B are split across two conda environments on purpose: the base env
has nibabel/scipy/skimage but no VMTK, and `vmtk_env` has VMTK but none of
those. Both have VTK, so a `.vtp` surface is the handoff format.

---

## Stage A — label mask to right-ICA surface

`topbrain_tools/mask_to_surface.py`

1. Read the NIfTI mask, keep the right-ICA label.
2. Keep the largest connected component.
3. Marching cubes, mapping to **world millimetres** through the NIfTI affine
   (marching cubes returns offsets from the voxel grid origin, not world
   coordinates).
4. Smooth with `vtkWindowedSincPolyDataFilter`, write `.vtp`.
5. Seeds for VMTK: skeletonise, then a double BFS to find the two ends of the
   trunk, ordered inferior → superior so the centerline comes out proximal →
   distal.

### Fix A1 — wrong label value

Started with `ICA_LABEL = 8`, taken from the numbering in the TopBrain paper's
metro-map figure. That figure is a schematic; the release's ITK-SNAP labelmap is
the authority, and there **R-ICA = 4** (8 is a different vessel).

Fixing this also removed code that had been written to compensate: a
left/right heuristic picking the component with smaller mean x. Sides are
labelled explicitly in the release, so no heuristic is needed.

---

## Stage B — surface to centerline with radii

`topbrain_tools/vmtk_centerline.py`

Mirrors the existing `vmr_processing_tools/extract_centerlines_vmtk.py`:
`vmtkCenterlines` with explicit `SourcePoints`/`TargetPoints`, then the
`MaximumInscribedSphereRadius` point array, so the radii are the same quantity
the project already ships.

### Fix B1 — polyline zig-zag (the worst bug in the pipeline)

The first version read the centerline points straight out of the polydata in
storage order. **A polydata's point order is not the path order.** The resulting
polyline zig-zagged back and forth along the vessel.

It was caught by a number that was too uniform to be real: tortuosity came out
at ≈3.7 for *all 25 patients*. Real ICAs do not agree to two significant figures.
Arclength was inflated roughly 2×.

Fix: always walk the cell's point ids (`GetCell(c).GetPointIds()`), even when
there is only one cell, and take the longest traversable cell.

| | before | after |
|---|---|---|
| tortuosity | 3.7 for all 25 | 1.49 – 2.40 |
| arclength | ~2× inflated | 80 – 144 mm |

### Fix B2 — seed points tacked off-path

`AppendEndPoints=1` appends the source and target seeds to the line, and they sit
slightly off the true path. This showed up as 47–79° turns.

Diagnosis: every large kink sat at arclength fraction < 0.03 or > 0.98, i.e.
only at the two ends, never mid-vessel. So the ends were trimmed adaptively
(1–3 mm), not the whole line smoothed.

Max kink fell to 23–48°, against 36.9° for the host's own worst bend.

### Non-issue B3 — VMTK "Unable to factor linear system"

VMTK emits `Generic Warning: In vmtkMath.cxx line 590` repeatedly. Checked
rather than assumed: no NaNs in any output, all radii positive, radius tapers
distally in all 25, and only 2 cases contained even a single duplicate point.
Harmless.

---

## Stage C — grafting onto the host

`topbrain_tools/graft_siphon.py`

### Coordinate convention

The branch loader does `points.append((y, -z, -x))` on each `.mrk.json`
position, so `branch = (jy, -jz, -jx)` and the inverse is
`json = (-bz, bx, -by)`. Both directions are implemented as `json_to_branch` /
`branch_to_json`; getting this backwards silently mirrors the anatomy.

### Where the cut goes

The shipped RCCA is 237.5 mm from the ostium. The TopBrain ICAs run
atlantoaxial → terminus at a median of 106 mm. Both end at the terminus, so the
matching host cut is 237.5 − 106 ≈ **130 mm**.

Each siphon then contributes its own real length, so total route length varies
across the set. That is genuine inter-patient variation and is deliberately
**not** normalised away.

### Radius repair

The host's own radii over roughly 105–130 mm are not plausible for a cervical
ICA (2.6–3.3 mm where 4–5 mm is expected) and are non-monotonic distally.

Rather than scale the real MISR radii down to meet a suspect number, the real
values are held exactly as measured and the **host** is ramped from its last
trustworthy anchor at 100 mm (4.45 mm) up to each patient's own proximal calibre
at the cut, with a smoothstep so there is no kink at either end.

This overwrites ≈30 mm of measured host radii with an interpolation. It is a
deliberate repair of a suspect segment and should be disclosed in any writeup.

### Fix C1 — siphons leaning over instead of climbing

**Symptom.** Rendering the anatomies (see [Figures](#figures)) showed
`topcow_mr_011`'s siphon leaving the junction almost horizontally, gaining ~10 mm
of height over 92 mm of vessel. A real ICA climbs.

**False lead.** The first hypothesis was a bad proximal tangent estimate, so a
sweep over tangent spans was run. It showed every case losing height, which was
too broad for a per-case estimation problem, and a probe axis that had been
assumed rather than measured turned out to be wrong: superior in the branch
frame is **+z**, not −y. The shipped `.mrk.json` is tagged LPS but actually runs
superiorly along −x, which `json_to_branch` carries to branch +z.

**Root cause.** Placement matched only the junction *tangent*. That leaves one
degree of freedom undetermined — the roll about the tangent — and it was taking
whatever value the accidental alignment of the two coordinate frames produced.

**Fix.** Match a **frame**, not a tangent: tangent onto tangent *and* superior
onto superior. Superior is +z in both frames (nibabel world is RAS; the host is
branch +z as above), so the construction is well defined.

| | tangent-only | frame-matched | real vessels |
|---|---|---|---|
| mean rise | 30 mm | 55 mm | 59 mm |
| descending | 3 / 25 | 1 / 25 | — |
| max junction kink | 46.5° | 45.8° | host's own worst 36.9° |

The one siphon still descending after this is `mr_021`, cleared by Fix C2.

No roll is introduced as a source of variation; it is pinned by the patient's
own superior axis, so the 25 real siphons remain the only thing that differs.

### Fix C2 — one siphon that cannot be anchored

`topcow_mr_021` opens with a cervical coil that doubles back: its proximal
tangent sits **125°** from its own chord. Anchoring on that stands the whole
siphon on its head, and no tangent span fixes it (span 20 gave a 116° junction
bend instead).

Fix: `anchor_trim` advances the proximal end until the local tangent agrees with
the chord to within 70°. It fires on this one case and trims 4 mm. The host
already supplies 130 mm of cervical vessel, so the dropped millimetres cost no
anatomy.

### Fix C3 — orphaned branches at the moved terminus

Two branches, `Centerline curve (13)` and `(24)`, start **exactly** at the
shipped RCCA terminus — they are its cerebral continuation. The graft moves that
terminus by 11–42 mm, leaving them beginning in mid-air.

Fix: drop the short unnamed cranial stubs. The split is unambiguous — all 9 of
them sit above z = 572, while every other unnamed branch tops out at z = 437 in
the arch. Named vessels (RCCA, RVA, LCCA, LVA) and the arch are kept.

### Fix C4 — siphons running through neighbouring vessels

With the stubs still present, **9 of 25** grafted siphons interpenetrated a
neighbour (centre distance minus both radii, negative = fused lumens). The
shipped RCCA keeps +7.5 mm from the RVA.

Two rounds of fixing:

1. **Drop the stubs** (C3), which removed 5 of the 9.
2. **Repair the RVA per anatomy** for the 4 that remained.

The order matters. Recomputing clearance *after* the drop changed which cases
failed: `mr_026` fell below zero only once its nearest stub was gone, while
`mr_011` (+0.48 mm) survives. Judging on the pre-drop list would have excluded
the wrong patients.

A single global RVA deflection was considered and rejected: perturbing all 25
anatomies to fix 4 is backwards, and it would have needed a much larger
displacement to satisfy every patient at once. Per-anatomy repair needs only
4–5 mm and leaves 21 anatomies byte-identical to the shipped tree.

| case | contact position | repair | clearance |
|---|---|---|---|
| `mr_018` | 96% along RVA, plus a second zone mid-vessel | deflected 4.00 mm | +0.91 mm |
| `mr_024` | 96% along RVA | shortened 20 mm | +0.78 mm |
| `mr_025` | 64% along RVA | deflected 5.25 mm | +0.78 mm |
| `mr_026` | 72% along RVA | deflected 4.00 mm | +0.52 mm |

Shortening is preferred where it works because it invents no geometry — the RVA
tip is a free end once the cranial stubs are dropped. It only reaches contacts
near the tip, which is why the two mid-vessel cases are deflected instead.
`mr_018` has a *second* contact at 163–173 mm sitting at +0.74 mm that no cut can
reach, so it takes a deflection despite its worst contact being distal.

Deflections are zero at the proximal end, smoothstepped to full amplitude before
the contact, and held to the tip. Verified afterwards: displacement starts
84–109 mm along the vessel, so the bifurcation where wrong-branch entry happens
never moves, and the worst bend stays at the shipped RVA's own 90°.

### Fix C5 — terminal cap erosion

Found only once an enclosure test existed (see [Stage E](#stage-e--verification)).

The mesher marks centerlines into a 0.6/0.9 mm voxel cube and smooths twice,
which pulls the terminal cap inward. On a thin distal ICA that leaves the final
centerline points **outside** the lumen they define — and since targets are
sampled from centerline points, a target there could never be reached.

The shipped RCCA never shows this because its terminus is 4.3 mm across. The
control confirmed it: `HOST(-stubs)` scores 1.000 enclosed, so this is not a
generic property of the meshing pipeline but specific to thin grafted termini.

Fix: trim the distal end (`DISTAL_TRIM_MM`). A 2 mm trim took the pass rate from
16/25 to 19/25; going to 4 mm, together with the depth-based tolerance described
in [Stage E](#stage-e--verification), reached 24/25.

---

## Stage D — baking the meshes

`topbrain_tools/bake_meshes.py`

Each anatomy's collision mesh is generated once and written into its own folder
as `vessel_architecture_collision.obj` (159 KB each, 4.1 MB total), matching the
layout of the shipped patient in `eve_bench/data/dualdevicenav/`.

### Fix D1 — meshing at runtime instead of baking

The first loader generated meshes at load time. That was wrong for three
reasons, all of which the fixed-mesh approach the shipped patient already uses
would have avoided:

- **Not portable.** The folder held only centerlines, so the geometry was
  re-derived by whatever scikit-image and pyvista happened to be installed on
  the target machine.
- **Slow.** Every anatomy switch paid seconds of marching cubes.
- **Weaker identity.** See D2.

Baked, an anatomy folder is a self-contained artifact that reproduces
byte-identically anywhere.

### Fix D2 — the loader inherited the wrong base class

The first `TopBrainAnatomySet` subclassed `RCCAVariedFromMesh` with its
perturbation amplitudes zeroed out. That class exists to *synthesise* a vessel
per generation; these anatomies are fixed. It was the shortest path to something
that ran, not the right structure.

Rewritten as a standalone `VesselTree` that loads baked meshes — the shipped
patient's `FromMesh` approach extended to hold a set and switch between them.

A concrete benefit beyond tidiness: the checkpoint fingerprint. Under the
procedural class it was `s{seed}g{gen}`, meaning "re-seed an RNG, replay N
generations, and trust it lands on the same geometry". It is now the anatomy
name, so restoring mesh-bound SOFA state is an exact lookup.

### Fix D3 — fingerprint format broke checkpoint matching

`training _scripts/util/checkpoint_restore.py` embeds the fingerprint in a
**filename** and matches it with `_mesh-([A-Za-z0-9]+)_`.

The fingerprint being emitted was `s0g1:topcow_mr_001` — a colon and
underscores. It would never have matched, and mesh-bound restores would have
silently stopped working with no error. Fingerprints are now stripped to
alphanumerics (`topcowmr001`) and the round-trip is tested against that exact
regex.

### Fix D4 — a stateful shuffle that would desynchronise

Anatomy selection was initially a reshuffled permutation held as instance state.
`reset()` and `regenerate_to_fingerprint()` both re-seed the RNG, and the
leftover permutation would survive that, so replaying a fingerprint could land on
a different anatomy than the checkpoint was captured on.

Made a pure function of `(seed, generation)`. (Largely moot after D2 made the
fingerprint name-based, but the selection order is still deterministic and still
covers the set evenly rather than sampling iid.)

### Fix D5 — an import that would have broken 10 launchers

`TopBrainAnatomySet` was first exported from
`eve/eve/intervention/vesseltree/__init__.py`. That file is bind-mounted into
the container by **10 launcher scripts**, none of which mount the new module, so
every existing training run would have failed at import.

The env now imports from the module path directly. Verified afterwards that the
existing mount set still imports `DualDeviceNavRCCAVaried` cleanly.

---

## Stage E — verification

Two harnesses, because centerline geometry says nothing about what the
simulator actually sees.

**`topbrain_tools/validate_anatomies.py`** — centerline geometry against the
unmodified host. Runs outside the container (numpy only).

**`topbrain_tools/check_anatomies.py`** — the real test, run in the container:

| check | what it catches |
|---|---|
| load | branch list parses, RCCA found |
| insertion | comes from branch (11), not the fallback, and is inside the lumen |
| **enclosure** | every RCCA centerline point inside the mesh — the erosion test |
| targets | pool non-empty after the 40 mm near-ostium exclusion, all inside the mesh |
| fit | catheter OD (0.7 mm) fits the narrowest lumen |
| `--sofa` | SOFA loads the mesh and steps without diverging |

Two **controls** are run through the identical code path: the shipped tree as it
ships, and the shipped tree with the same cranial stubs dropped. Both score
1.000 enclosed. Without the second control, Fix C5 would have been
misattributed to dropping the stubs.

The enclosure check has a tolerance, because a point a fraction of a millimetre
proud of a smoothed wall has no consequence — the pathfinder works off
centerlines, not the mesh, and the target threshold is 5 mm. A case fails only
if a point is **> 1.5 mm** beyond the wall **and** the run is **> 3** consecutive
points. Depth is measured, not just membership, precisely so this distinction
can be made.

### Current numbers

```
25 anatomies
  route length   201 - 263 mm   (host 238)
  rise           155 - 183 mm   (host 185 over its whole length)
  worst bend      18 -  45 deg  (host 37)
  junction bend    7 -  38 deg  (host worst bend 37)
  min diameter  1.60 - 4.00 mm  (host 2.43)

  descending routes: none
  junctions bending harder than the host's own worst bend: 1/25
  24/25 pass the static checks
```

SOFA rollouts confirmed on a sample, with targets forced past the 130 mm graft
junction (165–228 mm along the route) so the real siphon is exercised rather
than the shared trunk. Both devices advance 190 mm; tips move 121–163 mm.

### Known failure

**`topcow_mr_015` should be excluded.** Its distal 22 mm pinches shut: 23
centerline points outside the mesh, 22 of them consecutive, up to **7.19 mm**
beyond the wall, and 11% of its targets unreachable. Minimum lumen 1.60 mm,
which is around two voxels at the mesher's 0.6/0.9 mm spacing.

It still *runs* in SOFA without complaint, which is exactly why the enclosure
check was worth writing.

```python
DualDeviceNavTopBrain(exclude=["topcow_mr_015"])
```

Two more are **borderline and tolerated**, not clean: `mr_013` (5 points, up to
1.37 mm) and `mr_014` (3 points, up to 2.11 mm). Drop them too for a strict set
of 22.

---

## Using the anatomies

```python
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain

train = DualDeviceNavTopBrain(anatomy_dir="topbrain_data/anatomies",
                              seed=base_seed + worker_id,
                              episodes_between_change=10,
                              exclude=["topcow_mr_015"] + HELD_OUT)
test  = DualDeviceNavTopBrain(only=HELD_OUT)
```

Devices, simulation, fluoroscopy, insertion and target semantics are identical
to `DualDeviceNavRCCAVaried`, so a policy trained under the procedural variation
can warm-start on this.

### Transporting to another machine

`topbrain_data/anatomies/` is self-contained and sufficient — 30 MB on disk,
1.2 MB per anatomy, holding 16 centerlines plus the baked `.obj`. Verified by
copying two anatomies to an empty directory, mounting only that, and running
SOFA from it.

In git it is far smaller than it looks: the 425 files share only **69 unique
blobs (6.9 MB)**, because the non-RCCA host branches are byte-identical across
anatomies.

`.gitattributes` marks the anatomies binary. `core.autocrlf` is true in this
repo, so without that a Windows checkout would rewrite the `.obj` line endings
and no longer match a Linux one.

Two code files are needed alongside the data:

- `eve/eve/intervention/vesseltree/topbrainanatomyset.py`
- `eve_bench/eve_bench/dualdevicenavtopbrain.py`

A TopBrain run needs two mounts beyond the current launcher set: the module into
`eve/intervention/vesseltree/`, and the env into both `eve_bench` locations.

---

## Figures

`monitoring/figure_topbrain_pairs.py` renders the set two anatomies per image,
13 images, at one shared scale and view so patients compare directly. The RCCA
is drawn in two colours — host portion and real siphon — with the graft junction
marked, which is what made Fix C1 visible in the first place.

---

## Not done

- **No `--topbrain` flag** in `DualDeviceNav_train.py`. Training builds
  `DualDeviceNavRCCAVaried` behind `--procedural_rcca`; a parallel branch is
  needed.
- **`BenchEnv5` untested** on these. The intervention layer is verified; the
  reward/observation wrapper on top is not.
- **Target distribution.** `min_arclength_from_start` is 40 mm, so roughly 45%
  of targets land in the 130 mm of host trunk every anatomy shares rather than
  in the graft. Same as the procedural env, but it dilutes the inter-patient
  signal. Raising it to ~130 mm would force every target into the real siphon.
- **Regeneration needs the raw download.** `centerlines/` and `surfaces/` are
  gitignored intermediates of the 1.9 GB zip, so `graft_siphon.py` alone cannot
  rebuild the anatomies from a fresh clone. The committed anatomies are the
  durable artifact.

---

## Reproducing

```bash
# stage A (base conda env: nibabel, scipy, skimage, vtk)
python topbrain_tools/mask_to_surface.py <mask_dir> --out topbrain_data/surfaces

# stage B (vmtk_env)
conda run -n vmtk_env python topbrain_tools/vmtk_centerline.py \
    topbrain_data/surfaces --out topbrain_data/centerlines

# stage C
python topbrain_tools/graft_siphon.py \
    --centerlines topbrain_data/centerlines \
    --host eve_bench/data/dualdevicenav/Centrelines_comb \
    --out topbrain_data/anatomies

# stage D + E (in the container)
python3 topbrain_tools/bake_meshes.py --anatomies <dir>
python3 topbrain_tools/check_anatomies.py --anatomies <dir> \
    --host <shipped Centrelines_comb> --sofa 4
```

`graft_siphon.py` reports the RVA repairs and any rejections at the end of the
run. **Re-bake after regenerating** — the env raises `FileNotFoundError` naming
`bake_meshes.py` rather than silently falling back to runtime meshing.
