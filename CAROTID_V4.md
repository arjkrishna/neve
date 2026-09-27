# v4 carotid anatomies: the whole neck, merge-free

`carotid_data/anatomies_v4/` holds 223 anatomies, one for each v3 anatomy under
the same name. v3 grafted one route, on the right: host arch → a real carotid
fork → a TopBrain siphon. Every other neck vessel was still the shipped
host's. The LCCA ran unbroken to the left intracranial ICA with no fork, and
both vertebrals met at the host's basilar confluence. v4 grafts real patient
geometry onto all six neck vessels. It removes v3's mesh merges (defect 1)
with a union that cannot create them. Its collision meshes are gated to be
closed, manifold, genus-0 trees with no self-intersecting face, which v3's
were not.

    carotid_tools/graft_v4.py             compose the left carotid, both vertebrals, ECA extensions
    topbrain_tools/sdf_union_v4.py        the owned union: real surfaces that cannot fuse two vessels
    topbrain_tools/bake_meshes_v4.py      union ladder, topology gate, clean decimation, report
    topbrain_tools/find_bridges.py        diagnostic: where a mesh's handles are, and between what
    topbrain_tools/section_genus.py       diagnostic: genus of the mesh around each real section
    topbrain_tools/skeleton_centerline.py VMTK-free centerline (skeleton + EDT radii) for 2 VA labels
    carotid_data/anatomies_v4/BUILD_v4.json   flags, sources, counts, fallbacks

## At a glance

| | v3 | v4 |
|---|---|---|
| anatomies | 223 | **223**, same names, none dropped |
| RCCA route (host → fork → siphon) | grafted | **byte-identical to v3** |
| RECA | Zenodo fork ECA | same, plus a TopBrain ECA tip on 52 |
| LCCA / LECA | host tube, no fork | **host LCCA → Zenodo fork → TopBrain siphon**; the fork's own ECA, with a TopBrain tip on 53 |
| RVA / LVA | host, joined at the basilar | **host proximal → TopBrain VA distal**, blind ends |
| clearance gate on new geometry | 0.35 mm | **0.6 mm**, including ramp-widened host stretches and each branch against itself |
| new joins | — | ≤ 60° turn (the left route's own limit); ≤ 58° measured |
| collision mesh | tubes ∪ real surfaces (capsule union) | tubes ∪ real surfaces, **owned** union + topology gate |
| collision mesh topology | not gated: 70 of 223 non-manifold, 26 with extra handles, 207 not genus 0 | **all 223: one closed two-manifold genus-0 component, no self-intersecting face** |
| collision mesh size | 20 k triangles | **23.5 k**, all 223 (same siphon triangle count as v3) |
| right-route wall detail | — | **equal to v3**: median 1.00 on the fork and the siphon; below v3 only where v3's route carried a loop |
| navigable routes | RCCA | **RCCA and LCCA**, all 223 |

## Sources and handedness

- **Zenodo CarotidAnalyzer v2.0.0** (record 10695923): lumen STL and
  centerlines of 138 carotids. These supply the forks, as in v3.
- **TopBrain** (Zenodo 16878417, `labelsTr_topbrain_mr`, 25 MR masks). The
  labels used are 4/6 (ICAs), 23/24 (VAs) and 35/36 (ECAs). Centerlines come
  from VMTK 1.5.0; see *Reproduce*. VMTK fails on two right-VA labels,
  topcow_mr_004 and topcow_mr_020. `skeleton_centerline.py` supplies theirs,
  and the VMTK output is kept as `*.vmtk_failed.json.bak`.

Each v4 anatomy takes every new part from the patients already in its v3 name,
`case_<fork patient>_<side>__topcow_mr_<N>[_L]`:
- the left fork comes from the fork patient's other carotid;
- the left siphon, both VAs and the ECA tips come from TopBrain patient N.

**Handedness** follows v3's siphon. A siphon id ending in `_L` means v3 put the
patient's LEFT ICA, x-mirrored, on the right route. Such an anatomy is
*mirrored* (120 of 223), and every TopBrain part of it comes from the patient's
opposite side, x-mirrored. Its left siphon is the patient's right ICA
mirrored, its right VA is the patient's left VA mirrored, and so on, so the
donor head is one consistent mirror image. A *native* anatomy (103) takes every
part from its own side, unmirrored. Frame matching is a proper rotation and
never changes handedness: all 1 443 recorded rotations are proper. The mirror
flag in `provenance.json` is therefore the only place where chirality is
decided. Zenodo forks are never mirrored, as in v3.

## Composition

### Right side: unchanged

The RCCA centerline file is copied byte for byte from v3. The RECA file is too,
unless a TopBrain tip was added. v3's own right-route gate value is kept in
`provenance.json` as `v3_clearance_mm`. It was measured against v3's
neighbours under v3's 0.35 mm band, so it is not a v4 clearance.

### Left carotid

The left carotid is built by `graft_three.compose()`, the function that built
v3's right route, run with left-side constants through `g3_constants`. Both
sides are therefore composed by one method.

- **Seam.** The right seam is at 130 mm of RCCA, level with z = 530.3 in the
  branch frame. The host LCCA reaches that height at 157.6 mm and has
  108.2 mm left above it, against 105.8 mm on the right. The two ways of
  defining a left seam (same level, or same remaining length) agree. The seam
  is computed from the host at run time, and the left seam lands 0 to 2.1 mm
  short of it.
- **Host cut** is kept between 35 and 110 mm. The LCCA opens inside the aortic
  arch at 14.7 mm radius and reaches CCA calibre only by about 24 mm, so the
  calibre ramp is anchored at 30 mm. The host LCCA holds CCA calibre to about
  120 mm.
- **Donor order.** The fork patient's other carotid comes first (168 of 223).
  Then come substitutes ordered by calibre match at the siphon join, left
  carotids before right ones. The anatomy's own right fork is never a
  substitute. The 55 substitutes break down as:
  - 30: the other side is absent from the database;
  - 15: its CCA+ICA is too short to reach the seam;
  - 10: it failed the gates.
- **The ECA must be a fork.** `graft_three.load_lower` takes the second
  daughter path by calibre as the ECA. In five left lowers that v4 newly
  admitted, that path is an ICA-calibre sibling that shares the ICA for
  23–40 mm, so the "LECA" ran inside the LCCA. Where the calibre choice clears
  its ICA within 16 mm (v3's right forks: at most 15.7 mm), it stands.
  Otherwise the earliest-clearing daughter of adequate length and calibre is
  taken. This happened in 24 anatomies. After composition the LECA must clear
  its route within 16 mm: median 10.0, max 15.9.
- **Gates.** Kink ≤ 60° (max 48.1°), ECA ≥ 17 mm, the ECA re-entry check, and
  clearance ≥ 0.6 mm. Clearance is measured on the route from the cut, on the
  host stretch the CCA ramp widened, on the LECA, and on each against itself.
  - The LECA is trimmed where it closes on a neighbour or on itself (4
    anatomies).
  - Where the host LVA blocks the route, it is bent aside by `rva_deflect`
    (64 anatomies, 0.2–6.5 mm), with the host LCCA kept as a constraint.
- **Left siphon.** topcow_mr_006's left ICA is a 19.5 mm fragment. Its two
  anatomies take the right route's own ICA in the opposite handedness, a
  mirror copy of the right siphon (flagged).

### Vertebrals

A TopBrain VA's distal segment replaces the host VA above a cut. It is
frame-matched onto the host tangent exactly as a siphon is, and its blind end
sits at the level of the basilar junction. There is no basilar and no circle of
Willis: each VA is an independent segment, so the host's vertebrobasilar ring is
gone by design and every v4 anatomy is a tree.

- **Cut.** Placed at host length − donor length, and clamped to at least 80 mm
  and at most host − 10 mm.
- **Join calibre.** The host is ramped over 25 mm to the donor radius 4 mm in,
  capped at 1.3 × the donor median. The donor's first 4 mm are flattened to
  that value. This stops a label blob from setting the calibre: mr_025's left
  VA opens at 5.3 mm against a 1.4 mm median.
- **Moving the cut is the first repair.** It keeps the protocol intact, so it
  is tried first, in steps of ±5 mm up to ±30 mm, nearest first. If a donor
  opens on a turn over 60°, its start is advanced by up to 5 mm at the same
  cut first.
  - 90 of 446 VAs needed a moved cut, and 2 a trimmed donor start.
  - The first run, without the cut search, dropped 23 anatomies. Each was a
    right donor VA crossing the right carotid route at the length-matched cut:
    the same crowding that made v3 bend the host RVA in 100 anatomies.
- **Then deflect, then shorten**, at the length-matched cut. A deflected VA no
  longer sits where its transform put it, so its real surface is marked
  `real_surface: false` and baked as a tube (4 anatomies). Shortening was never
  needed.

### ECA extensions

The rule is "a convenient TopBrain ECA without a long gap". A piece qualifies
when it:
- comes from the same (handedness-mapped) patient side;
- is one connected label piece with at least 10 mm of usable centerline;
- has a calibre within 1 mm of the fork ECA's at the join, both measured 4 mm
  in from their cut faces.

The piece's proximal VMTK seed point is trimmed first: it sits on the wall,
1–2 mm off the axis. The join is held to 60°. 52 right and 53 left tips
qualify, adding 10–36 mm each. The remaining 341 ECA sides have no extension:

| why no extension | ECA sides |
|---|---:|
| no ECA label for that patient side | 190 |
| label piece shorter than 10 mm once its seed point is trimmed | 104 |
| collides with a neighbour or itself within its first 10 mm | 44 |
| calibre mismatch at the join | 3 |

The *distal* seed point of an extension, and of a VA, is not trimmed. It sits at
a blind tip, and trimming it too cost 43 extensions their 10 mm. Some tips
therefore end on a 1–3 mm jog of up to 82° (ECA) or 65° (VA), off every route.

### Clearance: 0.6 mm, the ramps, and each branch against itself

Every gate on new v4 geometry uses 0.6 mm (`NEW_CLEAR_MM`), not v3's 0.35 mm.
`fuse_band_v2.py` measured a 0.35 mm wall fusing in 3 of 6 grid alignments on
the 0.45 mm SDF grid, while 0.45 mm never fused. The first v4 bake confirmed it
on real anatomies: grafts passed at 0.35–0.45 mm came out as handles at the
tube level.

Three more places are measured:

- **The stretch below each cut.** Every graft blends the kept host vessel to
  the donor's calibre just below the cut, and the donor-side gates never looked
  there. On `case_k_011_left__topcow_mr_013_L` the RVA ramp grew into the right
  route behind a recorded clearance of 2.42 mm. `grown_clear` now gates every
  widened host point. Contacts that already overlapped count as host junctions
  only against unchanged host geometry, never against new branches.
- **Each new branch against itself**, for points more than 12 mm apart in
  arclength (the ownership window).
- **The seam itself**, and a VA cut moved upwards from its planned position.

Minimum over all new geometry: 0.604 mm. The records hold the final
clearances, measured on everything as written, next to the gate-time values
(`clearance_at_gate_mm`).

## Collision mesh

### Defect 1 and the owned union

v3 unioned each real surface as `max(f_tube, min(f_real, f_capsule))`, with a
capsule of 1.8 r + 1 mm around the section's own centerline. The graft gates saw
only the declared radii, so wherever another vessel, or another stretch of the
same looping siphon, lay within the capsule, the real surface could grow across
a gap the gates had passed. That caused extra handles in 64 of the 223 v3
anatomies. One of them is device-passable, in the training anatomy
`case_w_047_left__topcow_mr_018_L`.

`sdf_union_v4` keeps the union but lets a real surface add volume only to a
voxel it **owns**. A voxel is owned when its nearest centerline, by the
radius-aware tube measure the field itself uses (r − |x − c|, taken per branch
as the field does), is this section's own vessel at this stretch, by a margin
over:
- every other branch;
- the same branch more than 12 mm of arclength away.

A fork's route slice and its own ECA are cut from **one** Zenodo lumen, so
they do not compete with each other (`Section.siblings`). Its bulb and carina
are real anatomy, not a merge. Making them compete cost the fork a median 20%
of its wall detail against v3, and up to 64% in the worst donors; without that
competition the fork matches v3.

Between two vessels, the real surfaces leave an unowned band at least one
margin wide. That makes merges rare, not impossible: at 0.5 mm a cube diagonal
can still span the band. The guarantee is the topology gate below. Everywhere
else (the free lumen, the bulb, the nooks) the real surface is used as in v3.
`case_w_047_left__topcow_mr_018_L` bakes as a tree.

### Topology gate and ladder

Every v4 anatomy is a tree. So the union must be one closed component of genus 0
(Euler characteristic, boundary-aware) with no non-manifold edge once
coincident points are merged. On failure the baker climbs a ladder:

| step | what | anatomies ending here |
|---|---|---:|
| 1 | ownership margin 0.5 mm, then 1.0 mm | 201 at 0.5, 22 at 1.0 overall |
| — | tubes alone must be a tree (otherwise the graft gates missed a contact: raise) | checked on 83 |
| 2 | localise the handle: genus of the mesh patch around each real section in 10 mm bins (pinch vertices split, so a patch boundary is not read as a handle), then suppress the real surface in just those stretches, which become tube; at the margin with fewer handles | 82 anatomies carry suppressed stretches |
| 3 | if that stalls: find the section whose own real surface makes a handle, and slide a 12 mm window along it (then just past its ends) with the anatomy's genus as the oracle | 30 anatomies |
| 4 | prune: every suppressed stretch the gate does not need goes back to real surface; then shrink each remaining one from both ends by bisection, to 2 mm, while the anatomy stays genus 0 | all localised; shrunk in 79 |
| 5 | bake one whole section as tube | 4, and never a route section: a TopBrain ECA tip (×2) or a left fork ECA (×2) |
| 6 | tubes only | 0 |

Step 2 exists because the handles are mostly **in the source surfaces**. 16 of
25 right and 20 of 25 left TopBrain ICA label surfaces have genus > 0, with the
handles 70–100 mm along the ICA where the label reaches 10–12 mm outside the
inscribed radius. Ownership cannot prevent these, because the surface owns its
own voxels. Morphological opening and closing, admitting the neighbouring
labels, and a star-visibility test all left genus 2–4.

After shrinking, a suppressed stretch averages 3.7 mm on a siphon, 2.1 mm on a
fork and 2.7 mm on an ECA. Before shrinking it was the size of the search bin or
window, 12–16 mm, which cost the anatomies whose siphon label carries loops about
40% of their siphon wall detail against v3.

**Band edges.** The field is computed only in a band around the vessels, and
every other voxel holds a placeholder value (−1000). Where a barely-inside voxel
meets it, marching cubes puts two crossings on one grid node. Two sheets then
share a vertex: a seam the Euler count reads as genus 0 while the surface is
really a handle. The gate checks for these after merging coincident points. If
the union still cannot be baked without dropping a whole section, the baker
retries with the placeholders clamped to two voxels below zero and keeps the
result that loses less real surface. The clamp moves nothing but band-edge
crossings (at most 0.23 mm, on 8–9 of 30 000 vertices where measured), and it
is used in 1 anatomy, `case_w_008_right__topcow_mr_001`, where the unclamped
union would have lost its whole right siphon.

### Decimation

The union is decimated to 60 k triangles (`collision_full.vtp`, gitignored) and
from that to **23.5 k** (`vessel_architecture_collision.obj`, the SOFA mesh).

**Why 23.5 k rather than v2/v3's 20 k.** The v4 tree has the same wall area as
v3's: v3 already had the left carotid and the VAs, as tubes. But at a fixed
budget, v4's added real-surface detail takes triangles from the siphon. On the
byte-identical right route, the siphon had a median 15% fewer triangles than
v3's at 20 k (edges 7.7% longer). 23.5 k restores v3's siphon triangle count.
On five anatomies:
- siphon-wall p99 deviation from the 60 k mesh fell from 0.16–0.18 mm to
  0.14–0.16 mm;
- SOFA step time was unchanged (354 vs 328 ms, within noise).

**Clean** means one closed two-manifold genus-0 component with **no face
crossing another**. VTK's quadric decimation (the v3 protocol) is kept wherever
its result is clean: 160 full meshes and 170 collision meshes. It has no
topology guard. Where the union holds material narrower than the local edge
length (about 1.6 mm at 60 k), collapsing that material's rim glues its two
walls. The first full bake broke 30 unions this way, at 10 physical sites:
- 0.02–0.3 mm blades left where an ownership, capsule or suppression cut slices
  a real surface;
- clefts in two TopBrain labels;
- an ECA side-branch stub;
- one fold with no thin material at all.

VTK's collapse order does not depend on the target, so retrying nearby targets
cannot help. Splitting pinch vertices and dropping the pieces deleted a real
36 mm³ label lobe.

The fallback is vcg's (pymeshlab) quadric edge collapse with
`preservetopology`, which enforces the link condition and cannot glue walls.
Face flips are refused (`preservenormal`) first and allowed second, each
followed by vcg's folded-face removal if the result still crosses itself. For
the raw union it runs after one pass of vcg's marching-cubes collapse (56 full
meshes; 7 more by the area-weighted collapse). For the collision step, vertices are weighted by inverse area, because
vcg's area-weighted quadrics otherwise strip small vessels harder than VTK
does (53 collision meshes). If no decimation to 60 k is clean, the full budget
steps up (80 k, 100 k, 120 k, 180 k).

On 30 test unions it gave clean meshes with fidelity equal to VTK's (p99
0.135 mm vs 0.134). It is deterministic across processes and containers.
Three alternatives were tried and rejected:
- staged VTK: less faithful;
- VTK plus surgical repair: leaves repaired walls touching geometrically;
- topology-preserving collapse alone: worse maximum error.

### Wall detail against v3

The right route is the one training uses, and v4 must keep v3's nooks and
crannies there. Each `mesh_v4.json` records `right_route_detail`, measured on
the undecimated union, and v3's union was rebuilt the same way on the same
grid. On the decimated collision meshes the measure would be biased: coarser
triangles alone read as extra departure.

The measure is the share of the route's wall that stands off the plain tube
(`|x − c| − r`):

| right route | v3 (median) | v4 (median) | v4 ÷ v3, median | mean | anatomies ≥ 0.95 of v3 |
|---|---|---|---|---|---|
| fork/bulb, > 0.3 mm | 0.196 | 0.196 | **1.000** | 0.982 | 207 / 223 |
| fork/bulb, > 0.5 mm | 0.099 | 0.097 | **1.000** | 0.972 | 206 / 223 |
| siphon, > 0.3 mm | 0.322 | 0.322 | **0.999** | 0.988 | 204 / 223 |
| siphon, > 0.5 mm | 0.156 | 0.156 | **0.999** | 0.978 | 180 / 223 |

Every anatomy below parity has a loop in its v3 route union, and the missing
"detail" is that loop's material:
- the five `case_w_033_left` anatomies sit at 0.44 on the fork, where v3's bulb
  merges into a neighbour;
- `case_w_008_right__topcow_mr_001` sits at 0.86 on the siphon, where v3's
  route has genus 5.

89 of the 223 v3 route unions carry such a loop; no v4 union does.

### v3's shipped meshes

v3 had no topology gate. Of its 223 collision meshes:
- 70 have manifold defects: non-manifold edges on 67, pinch vertices on 17,
  holes on 39;
- 26 are closed but carry 2–3 handles, i.e. merges;
- 111 have genus 1, the host's vertebrobasilar ring;
- 16 are clean trees;
- 3 have self-intersecting faces.

The defects sit on the same kinds of site as above. Keep this in mind when
comparing v3 and v4 training results.

## Verification

After the first complete build, an independent audit ran five lenses:
topology (pymeshlab, not the baker's own code), grafts, union fidelity,
simulation, and an adversarial code review with each finding re-checked by a
skeptic. Everything in *Composition* and *Collision mesh* above that differs
from the first build answers one of its findings:
- LECA forks that were not forks;
- 60–98° joins;
- the VA blob;
- self-intersecting folds from the first fallback;
- localiser false hits and whole-section drops;
- a pooled nearest-neighbour search that over-granted ownership;
- stale records.

The final set was re-checked by the audit's own scripts, re-run unchanged:

| check | result |
|---|---|
| names one-to-one with v3; RCCA byte-identical; other host files identical | 223 / 223 |
| new-geometry clearance, pairs and self | 0 violations, min 0.604 mm |
| LECA fused with its route over ≥ 80% of the fork ECA | 0 (was 19) |
| ECA-extension join turn | ≤ 58° (was up to 98°) |
| transforms | 1 443, all proper rotations |
| records against geometry | left ECA length, step-2 deflections, right `extend_mm`: 0 stale |
| topology and self-intersection with pymeshlab (`topo_audit.py`), all 446 meshes | all one closed two-manifold genus-0 component; **0 self-intersecting faces** (was 126 on 16 collision meshes); v3 for comparison: 8 faces on 3 |
| training loader over all 223; SOFA, 300 stress steps on the 10 most changed anatomies | 18 branches, host insertion inside the mesh, every centerline point inside, RCCA byte-identical, targets beyond 133 mm; 0 exceptions, simulation errors or non-finite values; tip outside the wall in 7 of 3 000 steps, at most 0.24 mm (within SOFA's 0.3 mm contact distance, as the audit measured on v3) |

The whole set was baked in one run by the final code, so the committed code
reproduces it. The baker's gate itself checks every union for band-edge seams.

## Known limitations

- **TopBrain VA/ECA centerlines leave their own label surfaces** in 63
  anatomies (80 VA and 19 ECA-extension sections have more than 10% of
  centerline points outside). This is in the source data, not the transforms.
  The tube covers the centerline, so the lumen is continuous, but the real
  surface sits off-axis there.
- **Blind tips.** ECA-extension and VA tips can end on a 1–3 mm seed jog (see
  *ECA extensions*).
- **The left side is not reachable from the training insertion**, which sits
  inside the brachiocephalic trunk. The new LCCA, LECA and LVA are collision
  geometry until a loader inserts from the aorta.
- **Host end caps.** Under a hard stress schedule the device tip can escape
  through the blind end cap of host branch (17), the right subclavian stump,
  in v4 as in v3 at the same rate. This is host geometry, unchanged.
- **Walls between vessels** can be as thin as about 0.26 mm on the collision
  mesh. That is fine for SOFA's triangle contact, but it is below the union's
  0.5 mm margin, because decimation moves the surface.

## Records

- `provenance.json` has v3's record (right-route gate value renamed
  `v3_clearance_mm`) plus:
  - `v4`, `new_graft_clearance_mm`, `handedness`, `patient`;
  - `left`: fork, substitute reason, rejected candidates, ECA re-pick, fused
    run, siphon, seam, final and gate-time clearance, repairs;
  - `vertebrals`: donor/host lengths, cut, repair (incl. step-2 LVA bends),
    final and gate-time clearance, join kink, donor trim;
  - `eca_extension`: decision, reason and join kink per side;
  - `xform` for every real-surface section (`lower`, `siphon`, `lower_L`,
    `siphon_L`, `va_R`, `va_L`, `ecaext_R/L`) with its source surface, mirror
    and rotation.
- `mesh_v4.json`:
  - union mode, margins, every ladder attempt (incl. windows, probes and
    prunes), suppressed stretches;
  - tube-only and tube-in-effect sections, dropped fragment volume, band-edge
    clamp;
  - decimation path and budget per step, rejected budgets;
  - genus, route lumens, enclosure, pass.
- `BUILD_v4.json` holds every flag and the counts quoted here.

## Reproduce

Images:
- `eve-training-fixed` for grafting;
- `neve-build-meshlab` for baking: `FROM eve-training-fixed` plus
  `pip install pymeshlab==2023.12.post2`;
- `neve-build-stagea` for TopBrain surfaces: eve-training-fixed plus nibabel 5.2.1;
- `neve-build-vmtk` for TopBrain centerlines: micromamba, python 3.10,
  `vmtk=1.5.0`, `libitk=5.3.*`, and the libGL apt packages.

Mounts, as for v3:
`-v <repo>:/repo -w /repo -v <repo>/carotid_data:/opt/eve_training/carotid_data -v <repo>/topbrain_tools:/opt/eve_training/topbrain_tools`.

    python3 carotid_tools/graft_v4.py --out /repo/carotid_data/anatomies_v4 --shard i/6        # 6 shards, ~10 min
    python3 topbrain_tools/bake_meshes_v4.py --anatomies /repo/carotid_data/anatomies_v4 --shard i/5   # 5 shards, ~9.5 h
    python3 topbrain_tools/bake_meshes_v4.py --anatomies ... --recut-obj    # only re-cut collision meshes from the 60k meshes

`graft_v4.py` reads:
- v3's anatomies;
- the lower manifest and the extended ICAs;
- the TopBrain surfaces and centerlines under `topbrain_data/`.

The TopBrain files are regenerated from the MR masks in two stages, as in
`TOPBRAIN_PIPELINE.md`, once per label:

    python3 topbrain_tools/mask_to_surface.py <masks> --label L --suffix S --out topbrain_data/<surfaces>      # neve-build-stagea
    python3 topbrain_tools/vmtk_centerline.py topbrain_data/<surfaces> --suffix S --out topbrain_data/<centerlines> --tag T   # neve-build-vmtk

| label L | suffix S | surfaces | centerlines | tag T |
|---|---|---|---|---|
| 4 (R-ICA) | rICA | surfaces | centerlines | ica |
| 6 (L-ICA) | lICA | surfaces_left | centerlines_left | ica |
| 23 (R-VA) | rVA | surfaces_va | centerlines_rva | va |
| 24 (L-VA) | lVA | surfaces_va | centerlines_lva | va |
| 35 (R-ECA) | rECA | surfaces_eca | centerlines_reca | eca |
| 36 (L-ECA) | lECA | surfaces_eca | centerlines_leca | eca |

For the two VA labels VMTK fails on (mr_004 and mr_020):

    python3 topbrain_tools/skeleton_centerline.py <masks>/topcow_mr_004.nii.gz --label 23 --out topbrain_data/centerlines_rva/topcow_mr_004_va.json

## Loading

`DualDeviceNavTopBrain(anatomy_dir="carotid_data/anatomies_v4", ...)` loads v4
anatomies exactly as it loads v3 ones. You get 18 branches, including LCCA,
LECA, RECA, RVA and LVA, and the host's insertion. Targets are drawn on the RCCA
as before.
