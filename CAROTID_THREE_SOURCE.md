# Three-source carotid anatomies: build record and defect analysis

215 navigable anatomies composed from three real donors each, and an account of a
class of bug that produced silently wrong geometry twice in this pipeline.

    host arch        eve_bench/data/dualdevicenav      ostium to the first seam
    lower            CarotidAnalyzer bifurcation DB    real CCA + ICA/ECA fork + cervical ICA
    siphon           TopBrain                          skull base to the ICA terminus

Siphon donors are named `topcow_mr_NNN` because TopBrain's labels were drawn on
the TopCoW MR scans. TopCoW's own labels are not used: its ICA stops about 18 mm
below the circle of Willis (`TOPBRAIN_PIPELINE.md`, *Why TopCoW does NOT extend
this further*).

The point of the third source is the **ECA fork**. The previous 49 anatomies vary
only in the siphon and have no carotid bifurcation at all, because the host's RCCA
is one continuous branch. Entering the external carotid is a real clinical error
that those anatomies cannot represent.

    carotid_tools/analyze_bifurcations.py   parse the VMTK centerline tree, split CCA / ICA / ECA
    carotid_tools/build_manifest.py         measure all 138 carotids
    carotid_tools/extend_ica.py             copy a real distal 10 mm into ICAs that fall short
    carotid_tools/match_sections.py         decide which lower joins which siphon, under usage caps
    carotid_tools/graft_three.py            compose, repair, write centerlines
    topbrain_tools/bake_meshes.py           bake the collision mesh per anatomy (v1)
    topbrain_tools/bake_meshes_v2.py        v2: signed-distance mesher (sdf_mesher.py)
    topbrain_tools/bake_meshes_v3.py        v3: v2 tubes unioned with the real surfaces (sdf_union.py)
    topbrain_tools/check_anatomies.py       load / insert / enclose / ROUTE-CONNECTIVITY / target / fit
    monitoring/figure_carotid_anatomies.py  four-colour QC figures, two per image
    carotid_tools/run_container.sh          run any of the above with the right mounts

---

## Current state (v2 / v3)

This document is the build record of the **v1** set, `carotid_data/anatomies/`
(215). The set has since been rebuilt twice from the same 237-pair plan. Both
rebuilds keep every graft fix recorded below; what changes is the mesher, plus
the constants that existed only to compensate for it.

| | v1 (this document) | v2 | v3 |
|---|---|---|---|
| folder | `carotid_data/anatomies/` | `anatomies_v2/` | `anatomies_v3/` |
| anatomies | 215 | **223** | **223**, same as v2 |
| collision mesh | blurred binary tube, 3.7 k triangles | signed-distance tube, 0.45 mm grid, 20 k triangles | v2 tube ∪ the real Zenodo lumen (CCA/ICA/ECA) and the TopBrain siphon surface |
| `ROUTE_MIN_R` / `ECA_MESH_R_MM` | 1.60 / 1.60 | 1.0 / 1.0 | 1.0 / 1.0 |
| siphon floor (`--siphon-min-r`) | none | 1.0 | 1.0 |
| `FUSE_BAND_MM` | 0.35 | 0.35 | 0.35 |
| navigable (meshed lumen − 0.3 mm contact ≥ 0.35 mm catheter) | 71 / 215 | **223 / 223** | **223 / 223** |
| shipped stenosis grade, max | 36 % | 56 % | 56 % |
| anatomies at or above NASCET 50 % | 0 | 30 | 30 |

v2 gains eight anatomies over v1:

- **five** that v1 excluded as severed at the siphon terminus
  (`excluded_severed.json`). The signed-distance mesher and the 1.0 mm siphon
  floor keep that terminus open.
- **three** `case_w_014_right__*` pairs that v1 rejected for fusing. The lower
  floors put their clearance back above the fusing band.

The 14 pairs still not built are fusing rejections. v3's centerlines are
identical to v2's, so only the mesh differs; each v3 `provenance.json` also
records the transforms the graft applied (`xform`), which the union uses to
carry the source surfaces into place.

The grade is the declared grade on the centerline, after the floor, measured
the way `build_manifest.py` measures it: `1 − min ICA radius / distal ICA radius`.
Constants and guards: `V2_BUILD_PLAN.md`. Results: `MESHING_PIPELINE_ANALYSIS.md`
§7–8, and `BUILD_v2.json` / `BUILD_v3.json` in each folder.

---

## The defect class: an unconstrained degree of freedom, silently chosen

Every bug below is the same shape. A rigid placement has six degrees of freedom.
Matching a tangent fixes two. **The roll about that tangent is the third**, and in
each case something chose it without saying so — a default, an ignored argument, an
ill-conditioned projection, or a skipped normalisation step. Nothing errors. The
geometry is smooth, continuous, plausible, and wrong.

This is why it recurred: a kink would have been caught immediately by `max_kink`,
but a *roll* produces a perfectly smooth vessel pointing the wrong way. Nothing in
the validator looked at where the vessel ended up, only at whether it was
well-formed. The metric that catches it is **rise in z against the donor's own
native rise** — cheap, and now measured.

### 1. `anchor_trim` never called — the one that actually broke the meshes

`graft_siphon.graft()`, which builds the original 49, trims the siphon's first few
points before reading its proximal tangent, with the comment *"trim first:
everything below reads the siphon's proximal point"*. Some siphons open with the
petrous segment doubling back, so point 0's tangent aims away from where the vessel
actually goes.

`graft_three.py` called `prep_siphon` but not `anchor_trim`. Frame-matching that
backwards tangent onto the ICA tip applied a **near-180° rotation** and laid the
whole siphon on its side.

Only **2 of 45** siphons need the trim, and only 2–4 points each. But both are
heavily used:

| | before | after |
|---|---|---|
| `topcow_mr_021` median siphon rise | 1.1 mm | 65.4 mm |
| `topcow_mr_024_L` median siphon rise | 19.5 mm | 42.3 mm |
| anatomies with rise < 20 mm | 11 | **0** |
| anatomies with *negative* rise | 4 | **0** |
| worst rise error vs native | 112 mm | 44.8 mm |

### 2. Clearance measured from the wrong place — the one that mattered most for data quality

`min_clearance(..., from_mm)` is correct: below the graft the route is shipped host
geometry meeting neighbours at real junctions, where touching is right. But
`graft_three` passed `from_mm = 130.0` while its graft actually starts at
`host_cut`, **15–54 mm**. Everything between was unchecked, and the ECA — new
geometry in its entirety — was never checked at all.

Measured on the previous 231: **27 routes and 32 ECAs were interpenetrating a
neighbouring vessel.** The mesher fuses interpenetrating lumens into a shortcut no
patient has, which a policy can then learn to exploit.

Care was needed not to over-report this. The first measurement said "231/231
negative", which is the *shared ostium* — negative by construction, the same trap
that produced a bogus RVA finding earlier in this work. Bucketing the gap by
arclength separated the artifact (0–10 mm band, all 231) from the real defect
(70–130 mm band, 5–18 per band).

### 3. `extend_ica` used the vessel's own chord as its "up"

`frame(t, up)` builds a reference direction as `n = up − t(up·t)`. The manifest's
`superior` entry is **not** an anatomical axis; it is the CCA-inlet-to-ICA-tip
chord — roughly the direction the vessel already runs. Orthogonalising the tangent
against a vector nearly parallel to it leaves almost nothing to normalise, so the
roll of the copied 10 mm was decided by numerical noise. Three models sat under
|n| = 0.15, `case_m_022_left` at **0.047 (2.7°)** — and it duly appeared among the
worst rises. Now +z at both ends: correct anatomically and better conditioned for
every model in the set.

### 4. `place()` took an `up` argument and ignored it

The signature advertised a configurable reference axis; the body called
`frame_rotation`, which uses the module-level `UP`. Harmless in effect — all three
donors do carry superior as +z, so `UP` was right — but it is exactly the kind of
signature that makes a later reader believe a degree of freedom is being managed
when it is not. Removed, with the reasoning written down instead.

### 5. The sibling transform was recomputed rather than shared

`place()` returned only the moved points, so the caller recomputed the same
rotation to carry the ICA and ECA along with their own CCA. Two expressions that
must agree forever, with nothing enforcing it. `place()` now returns `(moved, R,
origin)` and the siblings ride the identical transform.

### 6. `resample(p, None)` raised an opaque numpy error

A centerline without radii is legal upstream; `np.interp` answered with *"object of
too small depth for desired array"*. Latent only — 0 of 34 extended files currently
have null radii — but it would have surfaced as a mystery graft failure.

### One hypothesis that measurement killed

Before finding `anchor_trim`, the obvious-looking fix was to replace frame-matching
with **minimal (geodesic) rotation**, which is well-conditioned exactly where
frame-matching degenerates. It is worse, and by a lot:

| siphon rise error vs native | median | p90 | over 20 mm |
|---|---|---|---|
| frame match (kept) | **2.2 mm** | 16.9 | 20 / 264 |
| minimal rotation (rejected) | 16.5 mm | 52.9 | 113 / 264 |

Frame-matching is right because all three donors share superior = +z, so forcing
that axis to agree *is* the anatomical constraint. The 179° rotations were not
frame-matching failing; they were frame-matching faithfully honouring a backwards
input tangent.

`frame()`'s own degeneracy guard was deliberately left at `1e-6`. Tightening it
would have silently altered the validated 49, and once `anchor_trim` is applied,
conditioning no longer drives the failures.

---

## Consequences for the set

**Radius floors** *(v1; both are 1.0 mm in v2/v3, and the siphon is floored too)*.
`ROUTE_MIN_R = 1.60` on the donor CCA/ICA and `ECA_MESH_R_MM = 1.60` on
the fork, pre-compensating the mesher's erosion *and* its `decimate(0.99)` — roughly half of
what the floor buys is decimation, so re-deriving it against the un-decimated surface yields
~1.35 and the wrong conclusion. Host and siphon are untouched, so the one segment the two
sets share stays identical. Cost: 55 anatomies floored, median 8% of the donor section, with
stenosis grades falling from 59–74% to 20–30%. That is a real loss of pathology, disclosed
per anatomy in `provenance.json` — but at those calibres the mesher produced *no lumen at
all*, so the choice was between a 27% stenosis and a sealed vessel.

**Gated inlet trim.** Where a donor's first millimetres point somewhere the vessel does not
go, the inlet is trimmed until its tangent agrees with the section's course — rather than
measuring the tangent over a longer span, which keeps the bad points and bends the seam.
Fires on 50 of 220; seam-1 kink max stays at 29.1° where span-widening drove it to 40.7°.

**ECA capped at 30 mm**, floor 17 mm. It exists to be a wrong turn, not a complete external
carotid, and its distal third was the largest single source of collisions with the host's
vertebral artery.

**RVA repair.** Deflection now runs before shortening, uses a per-point baseline, and can see
the ECA. Shortenings went 17 → **0**; the minimum deflection went 4.0 → 1.0 mm once the
host's own LVA/RVA confluence stopped setting the floor.

**Exclusions.** Nine donors (eight that collide with the host on every pairing, plus
`case_w_014_left` whose stenosis is too tight to mesh) and six anatomies severed at the
TopBrain siphon terminus — a defect set A shares, so it is excluded rather than repaired in
B alone.

**Mesh fusion.** Positive clearance is not enough. The mesher cannot resolve a wall thinner
than about one voxel-sigma, so two lumens closer than `FUSE_BAND_MM = 0.35` merge even though
their centerlines never touch: `case_m_024_left` sat at a +0.057 mm gap between its route and
its own ECA — clear by a `>0` test — and 4 of its 5 anatomies baked into one continuous lumen
with a **1.60 mm channel against a 0.35 mm catheter**. Every clearance gate is now the fusing
band rather than zero. Verified on the shipped set: route-vs-host min +0.497 mm, ECA-vs-host
+0.352 mm, and zero route↔ECA rings (was 10).

**ECA re-entry.** `ECA_MESH_R_MM` can inflate a thin distal fork back into the vessel it grew
from, closing an ICA–ECA ring the catheter could drive around. A plain route-vs-ECA clearance
test cannot catch it — the fork legitimately shares lumen with the route at its origin, and
that test reads −3.7 to −10.2 mm in *every* anatomy. `eca_reentry()` is topological instead:
the contiguous opening run at the fork's origin is the bifurcation; any overlap resuming after
it has ended is a ring, and the ECA is cut back before it. Fired on 5, rejected 5 more.

v1, as shipped in `carotid_data/anatomies/`:

| | |
|---|---|
| anatomies | **215**, all unique pairs |
| distinct donors | 47 lowers (max 5 uses), 44 siphons (max 8) |
| route length | 201.2 – 256.0 mm, median 227.1 |
| donor-section tilt from +z | median 12.1°, p90 22.0°, max 33.2°, none above 40° |
| z-rise | 132 – 191 mm, median 165 (set A: 154 – 187, median 170) |
| clearance | all positive, min +0.10 |
| max kink | 18.8 – 52.0°, median 28.8 |
| seam-1 kink | 3.6 – 29.1°, median 12.2 |
| ostium radius | 5.8121 mm in all 216, one unique value |
| validation | **215 / 215** component-aware (set A: 47/49); SOFA rollout 8/8 |
| repairs | 77 RVA deflections (1.0–7.5 mm), 9 ECA trims, 5 re-entry cuts |
| inlet trims | 87 anatomies, 1.0 – 24.8 mm |
| on disk | meshes baked, 108 figures |

## Audit outcome

A six-dimension statistical audit against the original 49, each dimension's non-cosmetic
findings handed to an independent agent instructed to refute them:

    navigability          aligned        lumen-and-stenosis    aligned
    curvature-and-seams   aligned        eca-fork              aligned
    route-geometry        minor_drift    clearance-and-host    minor_drift

Two concerns survived refutation and both were fixed here: the inlet-trim fall-through and
the ECA re-entry ring. The remaining drift is structural — B's donor bifurcations consume
arclength against the pinned 130 mm seam, so its z-rise floor sits below A's, and A's range is
not a population range because A holds the host and cervical carotid fixed.

## Known accepted cost *(v1)*

The v1 radius floor caps every shipped ICA stenosis grade at ~37%, erasing the 40–74% population
the donor database was built around. Set B therefore contains **no clinically significant
carotid stenosis** (the NASCET threshold is 50%). This is deliberate and reversible: roughly
half the erosion is `decimate(0.99)` rather than the smoothing, so baking the affected subset
at a finer decimation would let the floor drop and keep more grade, at the cost of SOFA
collision performance. Do not claim lesion realism for this set as it stands.

**In v2/v3 this is largely recovered.** At a 1.0 mm floor the shipped grade
reaches 56 %: 30 of 223 anatomies are at or above the NASCET 50 % threshold,
and 47 are at or above 40 %. Donor grades above 56 % (the database goes to 74 %)
are still capped, so the most severe lesions remain absent. The v3 union does
not change this: where the real lumen pinches below the floor, the floored tube
wins by construction.

## Rebuilding

    bash carotid_tools/run_container.sh python3 carotid_tools/extend_ica.py --manifest carotid_data/lower_manifest.json --out carotid_data/extended
    bash carotid_tools/run_container.sh python3 carotid_tools/match_sections.py --out carotid_data/pairing.json
    # graft in 8 shards (--only LO:HI), then prune folders not in pairing.json
    bash carotid_tools/run_container.sh python3 topbrain_tools/bake_meshes.py --anatomies carotid_data/anatomies --force
    bash carotid_tools/run_container.sh python3 topbrain_tools/check_anatomies.py --selftest
    bash carotid_tools/run_container.sh python3 topbrain_tools/check_anatomies.py --anatomies carotid_data/anatomies
    # figures: monitoring/figure_carotid_anatomies.py carotid_data/anatomies saved/figs/carotid_fixed --shard i/6

    # v2: same pairing, the v2 constants, the signed-distance baker
    bash carotid_tools/run_container.sh python3 carotid_tools/graft_three.py --out carotid_data/anatomies_v2 --route-min-r 1.0 --eca-mesh-r 1.0 --distal-trim 0 --fuse-band 0.35 --siphon-min-r 1.0 --only LO:HI
    bash carotid_tools/run_container.sh python3 topbrain_tools/bake_meshes_v2.py --anatomies carotid_data/anatomies_v2 --shard i/n
    # v3: graft into anatomies_v3 with the same flags (that run records provenance.json["xform"]), then
    bash carotid_tools/run_container.sh python3 topbrain_tools/bake_meshes_v3.py --anatomies carotid_data/anatomies_v3 --shard i/n
    bash carotid_tools/run_container.sh python3 topbrain_tools/check_anatomies.py --anatomies carotid_data/anatomies_v3

`graft_three.py`, `bake_meshes.py`, `check_anatomies.py` and the figure script all take
`--shard i/n`; each writes only into its own anatomy folders, so shards need no coordination.
Docker runs outlive a 10-minute shell timeout — launch with `nohup ... &` and poll.
