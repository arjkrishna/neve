# v5_bake_full: the full bake of the 223 v5 anatomies

Agent prefix `v5_bake_full`. State `v5_va/state/v5_bake_full.md`. Results `v5_va/results/v5_bake_full.json`
(written by `bake_full/collect.py`). No git. No tracked repo file and nothing in `carotid_data/anatomies_v4` was
changed (checked with find -newermt). Outputs go into `carotid_data/anatomies_v5/<name>/` only, plus
`anatomies_v5/BUILD_v5.json`.

## Answer

**All 223 v5 anatomies are baked and pass the bake gates.** The bake used `/sp/v5_va/bake_v5b.py`, the probe's fix
that applies your handle rule.

- **Gates:** the undecimated union, collision_full and the OBJ are each one closed genus-0 two-manifold, 223 of 223,
  with 0 self-intersecting faces on all three. collision_full is at 60k and the OBJ is within the 23.5k budget
  (23,300-23,500 tris). Both routes are navigable on all 223, and 0 centreline points are outside.
- **Records:** `mesh_v5.json` and `provenance.json` (`bake_v5`) hold the rung, time, gates, right_route_detail, VA
  detail and fallbacks for every anatomy. Licences are untouched.
- **Handle rule:**
  - 429 donor VAs went in. All 429 keep their real wall: none became a whole-donor tube, and none fell to v4's
    rung 5 or to tubes-only.
  - 404 donor VAs needed no window.
  - 25 needed exactly one local window: median 1.31 mm (1.15-9.14 mm), 0.55-6.2% of the donor stretch.
  - Only two windows are over 3 mm. Song 002 R in case_w_008_right__topcow_mr_020 has 9.14 mm (two handles in one
    window; detail kept 0.81). Song 094 L in case_m_030_right__topcow_mr_003_L has 4.43 mm.
- **Detail kept** (measured on the undecimated union, in situ against the donor alone): median 0.999, range
  0.81-1.27. 16 of 429 are below 0.95.
- **Donor-stretch wall** more than 0.3 mm off the tube: median 13.4% (2.9-33.7%). v4's host tube scored a median of
  0.0% (max 2.0%).
- **Failure-rule fallbacks:**
  - In 2 anatomies even the tube-only union had a handle: a centreline-level fusion at the donor's ostium junction
    with the subclavian. So neither local windows nor the whole donor as tube could reach genus 0. Those two sides
    keep v4's VA, flagged: case_w_050_left__topcow_mr_026_L right and case_w_033_left__topcow_mr_007 left.
  - Carotid windows: in 3 anatomies (carotid_changed, flagged) the pinned carotid sections alone were no longer a tree.
    The VA's new course changed which vessel owns the carotid wall. Each got one v4-style carotid window of
    1.15-1.6 mm.
  - OBJ budget: in 3 anatomies v4's obj_ladder found no clean 23.5k OBJ and stepped up to 35,250 tris. A stated
    alternative decimation brought each back within budget, clean (obj_fix).
- **right_route_detail:**
  - Identical to v4 in 117 of 223 anatomies.
  - Identical beyond 4 mm of every VA centreline in 63 more, beyond 6 mm in 26 and beyond 8 mm in 9. 8 differ even
    beyond 8 mm.
  - Fork ratio 0.82-2.39, siphon ratio 0.957-1.038.
  - 25 anatomies are below 1 (13 below 0.99, 4 below 0.97). The worst is case_w_050_left__topcow_mr_004_L, fork
    0.82. 68 are above 1.
  - The cause is ownership competition with the VA's new course, not the bake. In every differing bin examined, an
    RVA (v4 host or v5 donor) runs 2-6 mm from the carotid bulb wall. **This needs your decision** (Open issues).

## How the bake ran

- **Bake code:** `bake_v5b.py`, unchanged. `bake_full/bake_full_one.py <name>` makes the same calls as
  `bake_v5b.py --only <name>`, which are `B.bake_anatomy` then `record_provenance`. It also:
  - writes a done-marker;
  - logs every bake_one call;
  - records peak RSS;
  - removes a half-written anatomy's own outputs before a redo.
- **Queue:** 8 worker containers `v5bake_w*` (neve-build-meshlab, `--cpus 4 --memory 6g`), raised to 10 at 18:34
  once RAM showed 9.2 GB in use and a peak RSS of 2.15 GB. They replace the static `--shard i/8`.
  - Each worker claims the next anatomy from `bake_full/order.txt` (atomic mkdir).
  - The order is hardest first: the 45 anatomies whose donor side needed a window in probe B, then donor genus, then
    v4 bake time (`bake_full/plan.py`).
- **Measure:** `bake_full/measure_full_one.py <name>`, which reuses the probe's measures:
  - topology of the union, collision_full and the OBJ, plus pymeshlab self-intersections;
  - budgets, routes and enclosure;
  - right_route_detail against v4, plus diag_route's "identical beyond 4/6/8 mm of every VA" against v4's exact
    union, rebuilt with `ref_v4_union.rebuild`;
  - VA wall detail as `va_detail_v5.one` measures it on the undecimated union.

  The results go into `mesh_v5.json` (`v5_gates`, `va_detail`) and `provenance.json` (`bake_v5.gates`,
  `bake_v5.va_detail`).
- **Checks on the method:**
  - A smoke test reproduced probe B on case_w_012_left (detail kept 0.988, donor_real 0.329, identical beyond 6 mm).
  - The v4 rebuild matched mesh_v4.json exactly on all 223.
  - `bake_full/check_inputs.py` showed every non-VA input byte-identical to v4 on all 223 (all centrelines except
    RVA/LVA, and every xform except va*/vaC*).
- **Time and memory:** wall clock 17:29-20:21, fixes done by 20:26. Per anatomy the bake took a median 176 s
  (129-522 s) and the measure 199 s. Peak RSS was 1.80-2.13 GB in the bake and up to 2.16 GB in the measure.
  12.1 CPU-hours of bake in total.
- **Bookkeeping:** markers, logs and claims are in `carotid_data/va_sources/v5_reg/bake_full/`. The undecimated
  unions are in `carotid_data/va_sources/v5_reg/iso_v5/`.

## Rungs and flags

| rung | anatomies |
|---|---|
| 1 pinned, VA whole (VA margin 0.5) | 191 |
| 1 pinned, VA whole (VA margin 1.0) | 4 |
| 2 VA-local windows | 25 |
| C local windows, carotid included (flag carotid_changed) | 3 |
| D donor whole as tube / E v4 rung 5 / tubes only | 0 |

- **Far clamp:** case_w_008_right__topcow_mr_001 started from v4's far clamp, as v4 did.
- **Second bake_one call:** no anatomy needed one.
- **carotid_changed** (each has `carotid_check_genus` 1-2, meaning the pinned carotid sections alone had a handle in
  v5):
  - case_m_030_right__topcow_mr_013_L: lower_R window 65.8-66.95 mm. Right fork ratio 0.971.
  - case_w_016_left__topcow_mr_011_L: siphon_L window 236.4-238.0. Right route identical beyond 4 mm.
  - case_w_016_right__topcow_mr_011_L: siphon_L window 238.1-239.4. Right route identical.

  The cause, shown on case_w_012_left mr_020 with `bake_full/diag_bulb2.py`: carotid inputs are identical to v4, but
  v4's host RVA had been the winning ownership competitor for about 2.3k voxels of real carotid bulb wall. The donor
  RVA runs elsewhere, so that real wall now joins the union, and where it touches a neighbour it can close a handle.

## The handle windows (25 donor VAs, one window each)

| donor side (surface genus) | VAs | VAs with a window | window from the ostium | tubed |
|---|---|---|---|---|
| Song 094 L (47) | 4 | 4 | 104.2-108.6 mm | 1.31-4.43 mm |
| Song 002 R (32) | 16 | 6 | 111.4-123.2 | 1.21-9.14 |
| ia172_159 L (30) | 9 | 3 | 94.8-97.3 | 1.15-1.25 |
| ia172_015 R (22) | 16 | 3 | 83.0-124.5 | 1.48-2.00 |
| Song 032 R (12) | 2 | 1 | 111.9-113.1 | 1.15 |
| ia172_164 L (11) | 16 | 1 | 100.5-101.8 | 1.31 |
| Song 113 R (6) | 8 | 6 | 46.8-48.0 | 1.15 |
| ia172_039 L (6) | 1 | 1 | 108.0-110.0 | 2.00 |

Every other donor side baked with no window. That includes Song 008 L (g26), ia172_159 R (g12), ia172_136 R (g12),
Song 008 R (g11), Song 095 R (g10) and ia172_071 R (g10). Per-VA details are in results `anatomies[].va.*`
(`tubed_mm`, `tubed_frac`, `windows_from_ostium_mm`, `A2_detail_kept_in_situ_over_alone`, `donor_real_frac_0.3mm`).

The largest window is case_w_008_right__topcow_mr_020 RVA (Song 002 R), at 144.4-153.5 mm:
- The 10 mm bin localised [133, 159] with genus 2: two handles.
- Restoring the middle half failed, so the window could not be split.
- Each end was then shrunk to v4's SHRINK_TOL of 2 mm.
- The result is 9.14 mm (6.2% of the stretch), with 0.81 of the detail kept.

## Failures and the failure rule

### Tube-only handle: v4's VA kept on that side (2 anatomies)

Running `bake_full/diag_tubes.py` and `diag_tubes2.py` on the two failures gave the same picture:

| anatomy | side (donor) | handle | diagnosis |
|---|---|---|---|
| case_w_050_left__topcow_mr_026_L | RVA (ia172_149 L, mirrored) | RVA s 22-43 mm | without the RVA, genus 0; RVA truncated at 30 mm, genus 0. The junction stub (r 5.9 in the parent subclavian) meets the donor (r 1.4) where the donor leaves the parent with 0.70 mm clearance (assignment gate 0.6 mm) |
| case_w_033_left__topcow_mr_007 | LVA | LVA s 0-18 mm | without the LVA, genus 0. Parent clearance after the exit is 0.83 mm |

- **Why the rule ends at v4's VA:** the handle is in the tube geometry itself, so local windows and a whole donor
  tube both stay at genus 1.
- **Re-assembly** (`bake_full/keep_v4_va.py`): the anatomy was rebuilt the way graft_v5_assign writes a side whose
  donor fails:
  - v4's VA centreline file and xform entry restored, and `vaC_*` removed;
  - `v5_va.<side>` set to `{replaced: false, v4_va_kept: true, why, kept_by, donor_rejected_at_bake}`;
  - `v5_va.summary` and `v5_va.assignment` updated;
  - licences kept, since the other side still uses the donor;
  - originals saved in `bake_full/_reassembled/<name>/`.
- **Result:** both re-baked at rung 1 and pass every gate.
- **Counts after the bake:** RVAs replaced 211 (assignment 212), LVAs 218 (219). 206 anatomies have both sides
  replaced and 17 one; there are 17 v4-kept sides (15 from the assignment, 2 from the bake).

### OBJ budget (3 anatomies)

The topology was clean in all three, but v4's `obj_ladder` stepped up to a 35,250-tri OBJ.

`bake_full/obj_fix.py` makes one stated change: it tries other decimation targets at or under 23.5k, and other
sources, using v4's own `clean_decimate` and v4's own acceptance (is_clean, routes, enclosure). collision_full and
the union are unchanged, the originals are in `bake_full/_objfix/<name>/`, and each anatomy was re-measured.

| anatomy | accepted OBJ |
|---|---|
| case_w_048_left__topcow_mr_008 | from the 60k mesh at a 23,300 target (vcg quadric, topology preserving) |
| case_w_018b_left__topcow_mr_007_L | from the undecimated union at 23,500 (vcg marching-cubes collapse + quadric, as v4's full step) |
| case_m_006_left__topcow_mr_007_L | from the undecimated union at 23,500 (same path) |

## right_route_detail against v4

| beyond every VA centreline | anatomies identical to v4 |
|---|---|
| everywhere | 117 |
| 4 mm | 63 |
| 6 mm | 26 |
| 8 mm | 9 |
| not even beyond 8 mm | 8 |

The 8 that differ even beyond 8 mm are:
- case_m_030_right mr_013_L (carotid_changed)
- case_w_012_left mr_001
- case_w_012_left mr_020
- case_w_012_left mr_021_L
- case_w_033_left mr_010
- case_w_033_left mr_011_L
- case_w_033_left mr_024
- case_w_048_left mr_008

The ratio of the area fraction > 0.3 mm, v5 over v4: fork median 1.000 (0.819-2.386), siphon median 1.000
(0.957-1.038).

Ratios below 0.99:

| anatomy | fork | siphon |
|---|---|---|
| case_w_050_left__topcow_mr_004_L | 0.819 | |
| case_m_022_left__topcow_mr_021_L | | 0.957 |
| case_k_005_right__topcow_mr_024_L | 0.966 | |
| case_w_012_left__topcow_mr_021_L | | 0.968 |
| case_m_030_right__topcow_mr_013_L (carotid_changed) | 0.971 | |
| case_w_024_right__topcow_mr_026_L | 0.971 | |
| case_w_012_left__topcow_mr_001 | 0.980 | |
| case_w_007_right__topcow_mr_023 | 0.982 | 0.988 |
| case_w_029_right__topcow_mr_024_L | | 0.984 |
| case_w_029_right__topcow_mr_026_L | 0.986 | |
| case_w_012_right__topcow_mr_024_L | | 0.988 |
| case_k_008_right__topcow_mr_027 | | 0.989 |
| case_w_035_left__topcow_mr_016 | | 0.989 |

The largest gains are case_w_033_left mr_007/010/011_L/024, fork 2.12-2.39: there v4's host RVA had suppressed the
real bulb wall.

**Mechanism.** diag_route was run on w_012_left mr_020, w_050_left mr_004_L, m_030_right mr_013_L,
w_012_left mr_001 and mr_021_L (`va_sources/v5_reg/bake_full_diag/`). diag_bulb and diag_bulb2 were run on
w_012_left mr_020.

- Every differing 5 mm bin has an RVA, v4's host or v5's donor, 2-6 mm from the RCCA bulb or fork wall.
- The owned union (sdf_union_v4.ownership_margin) refuses real carotid wall where the VA tube is the nearer
  competitor.
- Where the donor now runs closer than v4's host did, real bulb wall is refused and detail drops. On w_050_left
  mr_004 the area > 0.3 mm in s 60-75 fell from 25 mm2 to 0.
- Where v4's host ran closer, the real wall now comes back and detail rises.
- Leaving out any one VA real section changes nothing. v4's own code on the v5 inputs reproduces the bake's union
  exactly. Carotid inputs are byte-identical to v4.

So the cause is the donor's course (the assignment), not the bake.

## Open issues for the user

1. **Route-detail gate.** A literal "ratio 1.000" holds on 117 of 223 anatomies. Losses above 1% are in 13
   anatomies, the largest a 18% fork loss on case_w_050_left__topcow_mr_004_L, where the donor RVA passes 3.5-5.5 mm
   from the RCCA bulb. There are three ways forward:
   - **(a)** accept and report it, with "identical beyond 8 mm of every VA" as the check (fails 8);
   - **(b)** route-first ownership: VA branches would not compete for the RCCA/RECA real wall. That changes the v4
     union standard, and handles at carotid-VA contacts would then need VA-side windows;
   - **(c)** re-assign or re-pose the donor for the anatomies with the larger losses.
2. **Two v4 VAs kept by the bake.** case_w_050_left mr_026_L right and case_w_033_left mr_007 left fail on
   ostium-junction clearance (0.70 and 0.83 mm). A re-graft of those two sides with a larger parent clearance after
   the exit (for example 1.0 mm) could probably keep a donor. That is outside the bake.
3. **9.14 mm window** on case_w_008_right__topcow_mr_020 RVA (two handles, detail kept 0.81). A per-handle split
   with a finer tolerance than v4's 2 mm could shrink it. Only 2 of the 25 windows are over 3 mm.
4. **Three carotid_changed anatomies** have 1.15-1.6 mm carotid windows that v4 did not have. They are flagged.
   Their right route is identical, or identical beyond 4 mm, except m_030_right mr_013_L (fork 0.971).
5. **Three OBJs** came from an alternative decimation target or source (obj_fix). All are clean and within budget.
6. **Housekeeping.** These scratch folders under `carotid_data/va_sources/v5_reg/` can be deleted once the check
   stage has accepted the bake:
   - `bake_full_test/` (smoke-test copies)
   - `bake_full_diag/` (diagnosis copies)
   - `bake_full/_reassembled` and `bake_full/_objfix` (originals)

   Keep `iso_v5/`, the undecimated unions that the VA-detail check needs.

## Files

- **Scripts** (`v5_va/bake_full/`): `plan.py` (order), `bake_full_one.py`, `measure_full_one.py`, `worker.sh`,
  `launch.sh`, `status.sh`, `collect.py` (results + BUILD_v5.json), `check_inputs.py`, `keep_v4_va.py`, `obj_fix.py`,
  `diag_bulb.py`, `diag_bulb2.py`, `diag_tubes.py`, `diag_tubes2.py`.
- **Order and plan:** `bake_full/order.txt`, `bake_full/plan.json`, `bake_full/check_inputs.json`.
- **Records:** `carotid_data/va_sources/v5_reg/bake_full/_done/<name>.{bake,measure}.json` and
  `_logs/<name>.{bake,measure,objfix,keepv4}.log`.
- **Undecimated unions:** `carotid_data/va_sources/v5_reg/iso_v5/<name>.vtp`.
- **Outputs:** `carotid_data/anatomies_v5/<name>/{vessel_architecture_collision.obj, collision_full.vtp, mesh_v5.json,
  provenance.json}` and `carotid_data/anatomies_v5/BUILD_v5.json`.
- **Results:** `v5_va/results/v5_bake_full.json`.
