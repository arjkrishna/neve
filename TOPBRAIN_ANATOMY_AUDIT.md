# TopCoW anatomy cohort — quality audit

2026-08-27. 25 anatomies from `topbrain_data/anatomies/` (`topcow_mr_001`…`_027`,
009 and 019 absent), pulled from `origin/rl_improv_16_resume`. Reference is the host
patient, `eve_bench/data/dualdevicenav/`.

Scripts: `monitoring/audit_topbrain.py` (quality + passability),
`monitoring/audit_topbrain_provenance.py`, `_base.py`, `_rcca.py` (provenance).

## 1. Mesh quality — the question asked

Decimation extent is **the same as the host**, to within a few percent.

| | host patient | cohort (25) |
|---|---|---|
| mesh cells | 3584 | 3654 – 3731 |
| edge median | 6.28 mm | 6.48 – 6.51 mm |
| edge p90 | 11.00 mm | 11.80 – 11.91 mm |
| edge max / median | 4.20 | 3.97 – 3.99 |
| edge / vessel diameter | 1.89 | 1.19 – 1.85 |
| branch length | 237.5 mm | 201 – 263 mm |

Triangles are equally coarse in absolute terms and *relatively finer*, because the
cohort's vessels are wider. No anatomy is more crudely decimated than the host.

## 2. SUPERSEDED — the deficit is real, the mechanism and the baseline were wrong

An earlier version of this section reported cohort clearance/stated-radius 0.69-0.79
against a host 1.07, called it "the regeneration signature", and listed 3 blocked /
2 marginal anatomies. A 4-agent diagnosis with 3 adversarial verifiers (workflow
wf_8e9b3126-902) refuted the mechanism, the baseline and the passability numbers.
What replaced them:

**The host reference of 1.07 is not a reference.** `DualDeviceNav` carries TWO host
surfaces through the identical `FromMesh` path (`dualdevicenav.py:23-24`):

| host surface | pts | R_med / stated_r | Pearson | CV |
|---|---|---|---|---|
| `vessel_architecture_visual.obj` | 22437 | **1.0019** | 0.981 | 0.054 |
| `vessel_architecture_collision.obj` — *what the wire hits* | 1786 | **1.647** | 0.472 | 0.252 |

The host's declared radii describe its real segmented lumen to 0.2%. The 1.07 came
from dividing accurate radii by a **dilated low-poly collision proxy** (~3.3 mm
near-constant lumen). Code-path equivalence was proven by measurement — hand-rotated
raw `.obj` vs `vessel_tree.mesh_path`, Hausdorff 0.000000 mm — so this is which
`.obj` is handed to the collision model, not a frame or loader difference.

**The correct calibration target is 1.00, not 1.07.**

**The cohort deficit is real but smaller.** Four independent estimators (exact planar
section, exact ray cast, vertex ring, slab+bin): exact-section R_med/stated_r
**0.84-0.89**, vertex ring 0.95-0.99. The original 0.74 came from a slab+azimuth-bin
estimator biased ~6% low on the cohort but ~18% low on the host — a differential that
inflated the very contrast the analysis was built around.

**The chord-error mechanism is refuted.** Cross-sections have **11-13 sides**, counted
directly (not 4.1). A 12-gon loses cos(pi/12) = 3.4%. A synthetic tube decimated
*coarser* than these meshes reads 0.983-0.987. Faceting explains ~3 of ~13 points.

**Mechanism unsettled**, two live candidates: a near-constant ~0.30-0.45 mm inward wall
displacement from `gaussian_smooth(1)` applied twice plus `marching_cubes(level=None)`
(`meshing.py:44-56`), which is LARGER on smaller vessels — the opposite of chord error;
versus axial under-resolution (facets spanning 8.3-10.6 mm on a vessel with radius of
curvature ~16 mm). Settled by `monitoring/mesh_ablation.py`, which exists and has never
been run.

## 2b. Passability — corrected with exact signed distance

Dense-surface point sampling **overestimates clearance 5-25x at the minimum** (host
0.31 mm sampled vs 0.041 exact; mr_024 0.188 vs 0.013). Corrected with
`vtkImplicitPolyDataDistance` including sign:

| anatomy | blocked | centerline outside mesh | note |
|---|---|---|---|
| topcow_mr_015 | 15 | 15 | surface truncated ~20 mm short; up to 6.87 mm outside — **unusable** |
| topcow_mr_013 | 7 | 5 | runs at s=206-219 mm |
| topcow_mr_014 | 3 | 3 | **missed entirely by the old gate** |
| topcow_mr_024 | 2 | 0 | 7 mm run at s=194-201 mm |
| topcow_mr_027 | 1 | 0 | terminal station only |
| **HOST PATIENT** | **3** | **2** | s=223.9-225.9 mm |
| topcow_mr_006 | 0 | 0 | catheter-marginal only (0.264 mm) — **not blocked** |

20/25 have zero. Dropping the last 2 centerline stations clears mr_027, mr_006 and
mr_014 — the terminal deficit is a universal end-cap artifact affecting all 26 meshes
including the host.

**These are meshing artifacts, not stenoses.** The narrowest stated radius anywhere in
the cohort is 0.800 mm = 4.4x the guidewire radius. No station in any anatomy could
physically block either device. The deficit is a roughly constant ~0.6-0.8 mm absolute
depth independent of calibre; a real stenosis would scale with r.

Also: all 25 cohort meshes are non-watertight (2-5 open boundary edges).

## 3. What varies, and what doesn't

The cohort is **independent subjects** — each anatomy carries a different TopBrain
patient's **right-side CCA→ICA** course. It is grafted onto a common carrier, and the
carrier is our host patient. The geometry says precisely how much is subject-specific:

- **15 of 16 centerlines are byte-identical across all 25, and identical to the host
  patient's** — LCCA, LVA, RVA, and curve (11), the insertion bridge. Only the RCCA
  carries subject anatomy.
- **91% of mesh vertices coincide exactly** (<1e-4 mm) between any two anatomies —
  the shared carrier.
- Along the RCCA, the 25 courses agree to **rms < 0.01 mm for the first 137 mm**, and
  match the host within 0.41 mm over that span. Divergence begins at ~137 mm and
  ramps over ~24 mm to 17 mm rms, with max pairwise separation reaching 58 mm.

So of a 201–263 mm branch, roughly the **distal 30–45% is subject anatomy** and the
proximal 68% is the host's own course, shared by every episode.

That split is favourable rather than not: the divergence point at ~137 mm sits
**proximal to the siphon band (~210 mm)**, so the subject-specific segment contains
the part of the task that actually fails. The cohort supplies genuine inter-subject
variation exactly where the difficulty is — and none in the approach.

## 4. Caveats before using them

**Surface and centerline disagree, systematically.** §2's clearance/stated-radius
deficit (0.69–0.79 against 1.07 for the host) holds for all 25. Whichever of the two
is the more faithful, the wire collides with the **surface** while guidance, targets
and `is_on_correct_path()` are computed from the **centerline** — so the mismatch is
operational regardless of provenance. It worsens distally (0.82 proximal → 0.40–0.61
at the terminus), i.e. it is concentrated in the grafted segment.

**Three are impassable and two are catheter-marginal**, as listed in §2. Any training
or eval run should gate on `monitoring/mesh_clearance.py` rather than assume 25 usable
anatomies; the clean subset is 20, and 22 if only the guidewire matters.

**What this cohort does and does not answer.** It varies the distal CCA→ICA course
across subjects. It does not vary the approach, the arch, or the topology — every
episode still threads the host's own proximal run to reach the graft. Against the LCCA
transfer result (0/98 on a different vessel of the same patient, 69.4% same-vessel
control), that makes this cohort a real but partial test: it will measure whether the
policy generalizes across **distal course**, and will not measure whether it
generalizes across **approach or topology**, which is where the LCCA failure lives.
