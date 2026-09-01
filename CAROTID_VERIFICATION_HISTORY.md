# How the carotid anatomies were checked, what the checks missed, and what fixed it

A record of the verification machinery built around the three-source anatomy set, in the
order it happened, and an honest account of what each layer caught and what it let through.

The short version: the geometry was never the hard part. **Every defect in this set composed
cleanly, passed the validator of its day, and looked plausible in the figures.** None raised an
error. Each was found by measuring the output against what the donors actually are — and
several were found only because a later, harsher check was built specifically to look for them.

---

## The set

Each anatomy composes three real donors:

```
 0 ──────────── 42 mm ─────────── 70 mm ────────── 128 mm ──────── 227 mm
 │   HOST        │   CCA           │   ICA           │   SIPHON        │
 │  shipped arch │◄──── donor section (carotid) ────►│  TopBrain       │
 │  + prox CCA   │        ECA forks off here ↑       │  skull base     │
 │               │        (the carotid bulb)         │  → terminus     │
 ostium       seam 1                              seam 2 (pinned 130)
```

Two sets exist and are compared throughout:

- **Set A** — 49 anatomies, host arch + TopBrain siphon. The reference. Already shipped.
- **Set B** — the three-source set, which adds a real carotid bifurcation and an **ECA fork**.
  The fork is the whole point: entering the external carotid is a real clinical error that
  set A structurally cannot represent.

Set B's size over the work: **231 → 229 → 220 → 216 → 215**, each drop a defect class removed.

---

## Verification, layer by layer, in the order it was built

### Layer 0 — the original static validator

`topbrain_tools/check_anatomies.py`, written with set A. Per anatomy: the branches parse, the
insertion point comes from the `(11)` bridge and lands inside the mesh, every RCCA centerline
point is enclosed by the mesh, the target pool is non-empty and inside the mesh, and the
catheter fits the narrowest lumen.

It reported **208/229 passing**. That number was wrong, and the reason it was wrong took two
workflows to find.

### Layer 1 — a user report, and three hypotheses measurement killed

A figure was flagged by eye: a siphon that did not rise in z
(`case_k_005_right__topcow_mr_024_L`). No check had complained.

What followed is worth recording because most of it was wrong:

1. **"The per-model superior axis is ignored."** `place()` accepted an `up` argument and never
   used it, and `build_manifest` stored a measured `superior` per model. Plausible. Measured:
   correlation between axis tilt and siphon rise was **−0.21**, and the same donor with four
   different siphons gave rises of 63/48/8/22 mm — a per-model constant cannot produce that
   spread. **Refuted.**
2. **"Use minimal rotation instead of frame-matching."** Well-conditioned exactly where
   frame-matching degenerates. Measured over 264 pairs:

   | siphon rise error vs the donor's native rise | median | p90 | over 20 mm |
   |---|---|---|---|
   | frame match (kept) | **2.2 mm** | 16.9 | 20 / 264 |
   | minimal rotation (rejected) | 16.5 mm | 52.9 | 113 / 264 |

   **Refuted, decisively.** Frame-matching is right *because* all three donors carry superior
   as +z, so forcing that axis to agree is the anatomical constraint.
3. **What was actually wrong:** seam-2 rotations reaching **179°**. `graft_siphon.graft()`,
   which builds set A, calls `anchor_trim` to drop a siphon's first few points before reading
   its proximal tangent — some siphons open with the petrous segment doubling back.
   `graft_three` never called it. Only **2 of 45** siphons need it, 2–4 points each, but both
   are heavily used: `mr_021` rose 1.1 mm instead of 65.4. Anatomies rising under 20 mm went
   **11 → 0**, negative rises **4 → 0**.

The lesson that shaped everything after: **a kink is caught immediately by `max_kink`; a bad
roll produces a perfectly smooth vessel pointing the wrong way, and nothing was looking at
where the vessel ended up.**

### Layer 2 — a hand audit of the same defect class

Searching for other unconstrained degrees of freedom found four more, all silent:

- `extend_ica` used each model's inlet-to-tip **chord** as the frame's reference axis. That is
  nearly parallel to the tangent by construction, so the roll of the copied 10 mm was decided
  by numerical noise (three models under |n| = 0.15, one at 0.047 = 2.7°).
- `min_clearance` was called with `from_mm=130` while the graft starts at `host_cut`
  (15–54 mm). The whole new CCA/ICA went unchecked and the ECA was never checked at all:
  **27 routes and 32 ECAs were interpenetrating a neighbour.**
- `resample(p, None)` raised an opaque numpy error; `place()`'s dead `up` argument; the sibling
  transform recomputed rather than shared; `note` printed but never persisted.

One near-miss worth recording: the first clearance measurement said "231/231 negative", which
is the **shared ostium** — negative by construction where branches meet at real junctions.
Bucketing the gap by arclength separated the artifact (0–10 mm band, all 231) from the real
defect (70–130 mm band). Reporting the raw number would have been a false alarm.

### Layer 3 — Workflow 1: the statistical audit (`carotid-vs-topbrain-stats`, 21 agents)

The first multi-agent run. Eight dimensions — route geometry, calibre, curvature, clearance,
mesh integrity, enclosure, ECA fork, donor balance — each measured on **both sets through one
code path**, then every non-cosmetic finding handed to an independent agent instructed to
**refute** it.

This is where the floor fell out:

- **enclosure → misaligned:** *"28 of 229 have the RCCA lumen COMPLETELY SEALED mid-route, at
  30–54% of the navigable path."*
- **mesh-integrity → misaligned:** *"26 of 229 carry the route on a mesh component disconnected
  from the arch."*
- **calibre → fatal:** *"18 have the centerline leaving the baked lumen mid-route."*

Three views of one defect. And the validator had passed them, which meant the validator was
the second problem.

Also surfaced: the RVA deformed 2.4× more often than in set A; `note` never written; the
ostium radius rewritten in ten anatomies; the CCA→ICA seam unramped; ECAs below their own
declared floor.

### Layer 4 — Workflow 2: root cause (`carotid-defect-rootcause`, 13 agents)

Six defects, each agent required to **prototype and test** a fix, not merely propose one, and
each fix then attacked by an independent challenger. Three agents died on a session limit.

**The severing.** The mesher marks a binary tube into 0.6/0.6/0.9 mm voxels, smooths twice at
σ√2 voxels and iso-surfaces at 0.500 — anything thinner than the kernel is simply absent. The
donors are **stenosis patients**: `stenosis_pct` 3–74%, and corr(min diameter, stenosis_pct) =
**−0.92**. Their real lumen sealed shut mid-vessel. The existing guard, `MIN_TIP_D = 2.0`,
tested the ICA **tip** — which the seam-2 ramp overwrites anyway. Fix: `ROUTE_MIN_R = 1.60`
on the donor section only, so host and siphon keep set A's calibre. 27/27 repaired.

**The validator's blindness, proven constructively.** `select_enclosed_points` answers "inside
the union of all components", so a vessel cut into two closed tubes still scores ~100%
enclosed. Demonstrated with two disjoint spheres, then with the real mesher on a tube with a
shrinking neck. No threshold on the old metrics could separate the cases — healthy `encl_frac`
went as low as 0.9774 while severed went as high as 0.9959. It was **the wrong measurement,
not a mis-tuned one**. Truth once measured component-wise: **set A 2/49 broken, set B 32/229**
— severance is a pre-existing mesher property the graft amplifies ~3×, and **11 severed
anatomies had been passing**.

**A defect nobody had flagged.** The ECA — the fork the entire set exists to provide — was
absent from the collision mesh in **103 of 229** anatomies (median fork radius 1.07 mm).

**Three things confirmed as real anatomy and deliberately left alone**, each refuted on
measurement rather than taste:

- the **CCA→ICA calibre step** is the carotid bulb — it matches the donor's own uncut profile
  to 0.087 mm, its sign is patient-specific (26 widen / 23 narrow), and 42 of 49 donors carry
  a larger step elsewhere. Ramping it would erase the one feature these donors were added for.
- the **distal siphon taper** is present in set A identically.
- the **"hole in the wall"** at breaks is a uniform `decimate(0.99)` artifact, uncorrelated
  with severing. **Refuted.**

**The challengers earned their keep by killing two fixes**: the RVA repair as first written
lost an anatomy, and a bundled `ECA_MIN_MM = 20` guard would have deleted 14.

### Layer 5 — Workflow 3: the missing challenges (`carotid-missing-challenges`, 2 agents)

Two challenges had died on the session limit, leaving `ROUTE_MIN_R = 1.60` and the orientation
fix **unchallenged**. Re-run against the rebuilt set rather than resumed, so the brief matched
what was on disk.

- **The floor: sound.** And it corrected me — I had tested 1.45 on a 24-anatomy subset and seen
  24/24 pass. Across the full set **10 anatomies still seal at 1.45**. My subset was
  unrepresentative. It also established that roughly half of what the floor buys is
  compensation for `decimate(0.99)`, not the smoothing.
- **The orientation fix: not sound**, and the challenger implemented a better one. Widening the
  tangent span keeps the bad inlet points *and* measures a tangent that is no longer the
  tangent at the joint, so the joint bends — every one of the 25 re-posed pairs got a worse
  seam-1 kink, worst +28.1°. **Trimming** the inlet instead — what `anchor_trim` already does
  for siphons — gets the same tilt fix with the regression gone:

  | variant | tilt max | >40° | seam-1 max |
  |---|---|---|---|
  | original 5 mm span | 52.7° | 15 | 29.1° |
  | adaptive span | 35.8° | 0 | **40.7°** |
  | **gated inlet trim** | 35.8° | 0 | **29.1°** |

### Layer 6 — Workflow 4: the final audit (`carotid-vs-topbrain-stats-final`, 28 agents)

Six dimensions on the rebuilt 220, same measure → refute → synthesise structure. Result:
**four aligned, two minor_drift**. Unlike every earlier round, **two concerns survived
refutation**, and both were mine:

1. **The inlet trim silently never fired on the case it was written for.** `head_trim()` had a
   bare `return 0` fall-through. `case_w_037_right` needed 21 mm of trim against a 15 mm
   budget; `case_w_047_left` cleared the threshold at 13.9 mm but was refused for leaving
   14.88 mm of CCA against a 15 mm floor — **a 0.12 mm miss**. Scope was worse than claimed:
   **8 donors / 38 anatomies** shipped an ungated inlet, and `provenance.json` honestly
   recorded `inlet_trim_mm 0.00` for every one. My own docstring table claimed a result the
   shipped constants did not produce.
2. **The ECA floor was closing false ICA↔ECA rings** in 8 anatomies.

---

## What was checked in, and when

| commit | contents |
|---|---|
| `5335a00` *(pre-existing)* | titled *"231 validated meshes"* — contained **8 code files and no geometry at all** |
| `a898eb2` | the eight fixes: `anchor_trim`, clearance scope, `extend_ica` axis, `ROUTE_MIN_R`, `ECA_MESH_R_MM`, ramp anchoring, RVA per-point baseline, the component-aware validator |
| `430bc76` | **the 216 anatomies themselves** — 4104 files — plus the extended ICAs and build records |
| `ba3715b` | the fusing-band gate and the first SOFA run (this check-in) |

Shipping the geometry needed two mechanical fixes, both of which fail silently:

- The ignore had to become `carotid_data/*` rather than `carotid_data/`. Excluding a
  **directory** makes git stop descending, and no later `!` can re-include anything beneath it.
  The first staging attempt added **zero files** for exactly this reason.
- `.gitattributes` marks the set `-text`. `core.autocrlf` is true here, so without it a Windows
  checkout rewrites the `.obj` and `.mrk.json` line endings and stops matching a Linux one —
  defeating the entire reason the meshes are baked to disk.

Of the 257 MB on disk, **79% is duplicated host branches** — the same 15 non-route vessels
copied into every folder so each is self-contained. They are only **100 unique blobs**, so git
stores ~59 MB of unique content. The lone branch that genuinely varies is `RVA.mrk.json`, with
86 versions — exactly the deflection log.

---

## What this check-in found and fixed

The trigger was a direct question after the push: *were mid-vessel vs terminal blockages,
reachability ceiling, inter-vessel fusion, and SOFA load all actually checked?*

Honest answer at the time: **three of four, and only one on the final build.**

| check | status when asked | now |
|---|---|---|
| mid-vessel vs terminal blockage | verified | ✅ `gap_mm` 0.00 in most, max 2.25 (tol 3.0) |
| reachability ceiling | measured, misread by me | ✅ the 4 shortfall cases were the already-excluded severed ones |
| inter-vessel fusion | **not verified on the final build** | ✅ **real defect found and fixed** |
| SOFA load | **never run** | ✅ 8/8 load and step cleanly |

### The defect: positive clearance is not enough

`case_m_024_left` had a centerline gap of **+0.057 mm** between its route and its own ECA.
Positive — so every `> 0` test passed it. But the mesher cannot resolve a wall thinner than
about one voxel-sigma, and **4 of its 5 anatomies baked into a single continuous lumen** away
from the fork, with a **1.60 mm channel against a 0.35 mm catheter radius**.

That is a ring connecting the internal and external carotid where no patient has one — and
precisely the geometry a policy learns to exploit instead of learning the task. It was in the
set that had just been pushed.

The root cause was conceptual, not arithmetic: **the code tested for *overlap* when the mesher
fuses at *proximity*.** `eca_reentry` used a `< 0.0` threshold; the fusing band is ~0.35 mm.

### The fix

`FUSE_BAND_MM = 0.35` now gates every clearance decision: the route-vs-neighbour acceptance
test, the ECA clearance trim, the RVA repair trigger, and `eca_reentry`'s definition of
"overlapping".

| measured on the shipped set | before | after |
|---|---|---|
| route vs host, minimum clearance | +0.092 mm (6 inside the band) | **+0.497 mm, none inside** |
| ECA vs host, minimum clearance | +0.352 mm | +0.352 mm, none inside |
| route↔ECA rings away from the fork | **10** | **0** |
| ECA re-entry cuts fired | 5 | 10 |

### SOFA, run for the first time

`check_anatomies --sofa` builds the real simulation environment and steps the guidewire. All
8 anatomies load, the beam adapter initialises, 63.5 mm inserts and the tip advances 40–54 mm
over 40 steps with no divergence.

Geometry checks demonstrably cannot substitute for this, and it should be re-run **whenever
the mesher or the radii change**.

---

## Where it landed

| | |
|---|---|
| anatomies | **215**, all unique pairs |
| donors | 47 lowers (max 5 uses), 44 siphons (max 8) |
| validation | **215 / 215** component-aware (set A: 47/49) |
| SOFA rollout | 8/8 |
| clearance | all above the fusing band, min +0.497 mm |
| donor-section tilt | median 12.1°, max 33.2°, none above 40° |
| ostium radius | 5.8121 mm in all 215, one unique value |
| figures | 108 |

Four workflows, **64 agents**, across the fix cycle.

## What the checks now catch that they did not

- **route connectivity** — the route must lie in ONE mesh component reachable from the
  insertion point, tested per component. Ships with a `--selftest` that reproduces the old
  blindness on a synthetic 0.6 mm neck, so it cannot silently regress.
- **honest reachable targets** — the fraction inside the *hosting* component, not merely
  inside the mesh.
- **the fusing band** — clearance calibrated to what the mesher can represent, not to zero.
- **ECA re-entry** — topological, because a metric test cannot distinguish the legitimate
  bifurcation from a ring.
- **SOFA rollout** — the only check that exercises the simulator rather than the geometry.

## What remains known and open

- **No clinically significant stenosis.** The radius floor caps every shipped grade at ~37%,
  below the 50% NASCET threshold, erasing the 40–74% population the donor database was built
  around. Deliberate and disclosed per anatomy in `provenance.json` — at those calibres the
  mesher produced no lumen at all, so the real choice was a 27% stenosis or a sealed vessel.
  Reversible: about half the erosion is `decimate(0.99)`, so baking the affected subset at a
  finer decimation would recover grade at the cost of SOFA collision speed. **Do not claim
  lesion realism for this set as it stands.**
- **Five anatomies excluded** as severed at the TopBrain siphon terminus — a defect set A
  shares, so it is excluded rather than repaired in B alone.
- **Set B's z-rise floor sits below set A's.** Structural, not a defect: real donor
  bifurcations consume arclength against the pinned 130 mm seam, and set A's range is not a
  population range because A holds the host and cervical carotid fixed.

## The pattern worth carrying forward

Every defect here shared a shape: **something chose an unconstrained degree of freedom, or
applied a threshold calibrated against the wrong quantity, and nothing errored.** A skipped
normalisation step, an ignored argument, a clearance measured from the wrong arclength, a
tolerance of zero where the instrument's resolution is 0.35 mm.

The checks that caught them were not cleverer, they were *aimed differently*: measuring the
output against the donor's own geometry, comparing both sets through one code path, and — most
productively — instructing an independent agent to **refute** each finding rather than confirm
it. Two of the fixes in this history were overturned that way, and both replacements were
better. One of them, the inlet trim, was overturned twice.
