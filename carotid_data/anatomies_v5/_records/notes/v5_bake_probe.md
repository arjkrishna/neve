# v5_bake_probe: probe bakes of the high-genus donor sides before the full v5 bake

Agent prefix `v5_bake_probe`. State `v5_va/state/v5_bake_probe.md`. Results `v5_va/results/v5_bake_probe.json`
(`probes.<name>.<run>` and `decision`). No git. I modified no tracked repo file and nothing in
`carotid_data/anatomies_v4` or `carotid_data/anatomies_v5`. Every bake ran on scratch copies under
`carotid_data/va_sources/v5_reg/bake_probe*/<name>/`.

## Answer

**GO WITH FIX.** The full bake should use **`/sp/v5_va/bake_v5b.py`** on **8 shards**.

bake_v5b implements the user's handle rule. Only the short stretch of donor wall around each handle becomes tube, and
the rest of the donor keeps its real coarse wall. The probes are the 12 anatomies picked below plus one far-clamp
anatomy.

- **All 13 pass.** Each one passed at rung 1 (VA walls whole) or rung 2 (local VA windows). No donor needed a
  whole-section tube, and no fallback was flagged.
- **Topology:** the union, collision_full (60k) and the OBJ (23.5k budget) are each one closed genus-0 two-manifold
  with 0 self-intersecting faces.
- **Routes and centrelines:** both routes are navigable and every centreline point is inside.
- **Donor wall tubed:** 0 to 2.63 mm per VA, at most 1.3% of the donor stretch. Twelve donor sides have surface genus
  10 to 47. Only 4 of them needed any window, and each needed exactly one window of 1.25 to 2.63 mm, in mid-V2:
  84 to 123 mm from the ostium, the foraminal stretch.
- **Wall detail kept:** measured on the undecimated union, the donor stretch stands > 0.3 mm off the tube on 3-33% of
  its area. v4's host tube there scores 0.0-0.8%. Measured in situ against the donor alone, the detail kept is
  0.90-1.01 (median ~0.99).
- **Time and RAM:** 101-317 s per anatomy, with about 13 s per ladder attempt. Peak RSS is 1.82-2.12 GB.
- **v4's unchanged ladder (run A)** also passed all 12. It cost up to 50 min per anatomy, and on Song 094 L it fell
  to its whole-section rung, which skips shrink: 21 mm of donor wall went to tube and the donor kept 0.82 of its
  detail (B: 0.988).
- **right_route_detail:** identical to v4 everywhere more than 4-8 mm from a VA centreline. Within that band the
  donor's new course changes ownership (see "Gate note"). This is not a bake effect: it is the same with v4's ladder
  and at rung 1 with nothing suppressed.

## Probe set

Every assigned donor side whose stage-A surface has genus >= 10 gets one anatomy (`donors_v5/<unit>/<side>.json`
`surface.genus_measure_json`). Of all v5 anatomies using that side, the probe is the one with the longest donor stretch,
ties broken by the smallest clearance to another vessel (`bake_probe/pick.py`). Two picks cover two sides each, so the
12 sides need 10 anatomies.

| donor side (genus) | probe anatomy |
|---|---|
| Song 094 L (47) | case_w_012_left__topcow_mr_003_L |
| Song 002 R (32) | case_m_022_left__topcow_mr_020 |
| ia172_159 L (30) + 159 R (12) | case_w_048_left__topcow_mr_010_L |
| Song 008 L (26) + 008 R (11) | case_w_051_right__topcow_mr_012 |
| ia172_015 R (22) | case_w_035_left__topcow_mr_016 |
| ia172_136 R (12) | case_w_018b_left__topcow_mr_002 |
| Song 032 R (12) | case_m_024_left__topcow_mr_006 |
| ia172_164 L (11) | case_w_022_right__topcow_mr_011 |
| ia172_071 R (10) | case_w_013_right__topcow_mr_004_L |
| Song 095 R (10) | case_w_017_right__topcow_mr_014_L |

There are two controls, case_w_012_right__topcow_mr_024_L (Song 109) and case_k_011_left__topcow_mr_023_L (ia172_190).
For B only I added one extra anatomy, case_w_008_right__topcow_mr_001. It is the one v4 anatomy baked with the far
clamp, so it exercises bake_v5b's far-clamp-first path.

## Diagnosis (run A: bake_v5.py, v4's ladder unchanged)

- A high-genus donor surface does not by itself make the union handle. In A, Song 032 R (g12), ia172_071 R (g10),
  Song 095 R (g10) and both controls baked at rung 1 with nothing suppressed. Most donor handles lie outside the
  ownership capsule, or ownership cuts them away.
- The one failure of the kind the rule forbids was case_w_012_left (Song 094 L):
  - v4's ladder re-chooses every section's margin. It localised at 1.0 (6 handles against 7 at 0.5), where v4 had
    baked at 0.5.
  - Round 1 suppressed vaC_R, eca_R and siphon_R and left genus 1. Round 2's 10 mm bins found nothing new, and all 11
    single-section window probes were genus 0, so the ladder fell to its whole-section rung.
  - `bake_probe/diag_handle.py` (ball scan plus section-pair probes; `_diag/<name>.handle.json`) showed that the
    remaining handle is **v4's own carotid-fork handle**: an RCCA s~96 ball hit, inside the already-suppressed eca_R
    stretch. No pair involving a donor makes it. v4 itself needed ecaext_R as a whole tube there.
  - A's rung 5 duly tubed ecaext_R (as v4), but rung 5 skips prune and shrink. The donor's 21 mm round-1 window
    stayed: 11.2% of the stretch tubed, and the donor kept 0.82 of its detail in situ against alone.
- A was slow: 60-100 s per attempt with 9-10 bakes running at once, and 1080-2982 s on 7 of 12 anatomies.

## The fix: bake_v5b.py

`bake_v5b.py` adds stated diffs (D1-D8) on top of `bake_v5.py`. `bake_meshes_v4` itself is unchanged; its
`bake_one` and `bake_anatomy` are swapped at run time, and the v4 tail (decimation, obj ladder, routes, enclosure,
right_route_detail) is v4's own code.

- **D1 carotid pinned.** The RCCA, RECA, LCCA and LECA sections are baked exactly as v4 baked them: v4's margin,
  suppressed stretches, whole-section tubes and far clamp. Their centrelines and surfaces are unchanged in v5.
- **D2 VA margin.** The VA sections (va_R and va_L, which are re-registered TopBrain pieces, plus the donors vaC_R and
  vaC_L) get their own margin: 0.5, then 1.0, by v4's rule. They come after every carotid section in the union, so
  they cannot change what the carotid sections add.
- **D3 VA localisation.** Up to 25 rounds of bins along each VA section's ownership reach (+-12 mm). The bins are
  10 mm as in v4, then 20, 40 and 80 mm only when the smaller ones find nothing new.
- **D4 bisection.** When a round finds nothing new, the whole reach of a VA section is suppressed and then halved
  while the genus still drops, down to 12 mm.
- **D5 prune, split and shrink.** These run on the new (VA) stretches only. Windows over 24 mm are split, and each end
  is shrunk to SHRINK_TOL 2 mm.
- **D6 flagged fallbacks.**
  - C: local windows on every section, carotid included ("carotid_changed").
  - D: a donor section as a whole tube ("donor_tube", course kept).
  - E: v4's rung 5/6.
  - Keeping the v4 VA instead needs a re-assembled anatomy, which is outside the bake.
- **D7 cached union.** The tube field and each section's inputs are computed once per bake_one, and an attempt
  re-applies only the sections that changed. The result is bit-identical to `add_real_sections_owned`: `--selftest`
  checked 5 apply/undo sequences on k_011, and the far clamp with its restore on w_008. Each attempt takes ~13 s
  instead of 60-100 s.
- **D8 report.** The report is `mesh_v5.json`: v4's keys plus `ladder_v5b`, which holds the pinned state, VA margin,
  rung, flags, and per VA section the windows and the tubed mm and %. The CLI (`main`) skips anatomies that already
  have `mesh_v5.json`, dumps the undecimated union to `carotid_data/va_sources/v5_reg/iso_v5/` (outside
  anatomies_v5), and adds a `bake_v5` record to provenance.json without touching any other key. I tested this on a
  scratch copy: the provenance gains only `bake_v5`, a rerun skips, and collision_full, the OBJ and the iso match the
  probe bake byte for byte (the OBJ differs only in the meshio timestamp line).

## Results: run B (bake_v5b) against run A

Columns:
- **route ratio:** fork / siphon at 0.3 mm, v5 over v4. "id>Xmm" means identical beyond X mm of every VA.
- **donor tubed:** mm (% of the donor stretch).
- **kept:** donor detail in situ over alone.

| anatomy | B rung | B s | B RSS | B route ratio | B donor tubed | B kept | A rung | A s | A donor tubed | A kept |
|---|---|---|---|---|---|---|---|---|---|---|
| w_012_left mr_003_L (094 L g47) | 2 local | 315 | 2.00 | 0.996/1.000 id>6mm | R 1.31 (0.7%) | R 0.988 | 5 ecaext_R tube | 1888 | R 21.0 (11.2%) | R 0.82 |
| m_022_left mr_020 (002 R g32) | 2 local | 317 | 2.06 | 1/1 identical | R 2.63 (1.3%) | R 0.946 | 2 | 2260 | R 3.28 (1.6%) | R 0.941 |
| w_048_left mr_010_L (159 L g30, R g12) | 2 local | 296 | 2.05 | 1.133/1.000 id>6mm | R 1.25 (0.8%) | R 0.901 | 2 | 1885 | R 2.0 (1.3%) | R 0.821 |
| w_051_right mr_012 (008 L g26, R g11) | 1 | 164 | 1.85 | 1.054/1.001 id>8mm | 0 | 0.996 / 1.0 | 5 eca_L tube (as v4) | 2982 | 0 | 0.996 |
| w_035_left mr_016 (015 R g22) | 2 local | 283 | 1.98 | 1.005/0.989 id>6mm | R 1.47 (0.7%) | R 0.996 | 2 | 1246 | R 1.15 (0.6%) | R 0.996 |
| w_018b_left mr_002 (136 R g12) | 1 | 145 | 1.86 | 1/1 identical | 0 | 1.006 | 3 window | 2366 | 0 | 1.006 |
| m_024_left mr_006 (032 R g12) | 1 | 133 | 1.82 | 1/1.008 id>4mm | 0 | 1.0 | 1 | 142 | 0 | 1.0 |
| w_022_right mr_011 (164 L g11) | 1 | 140 | 1.82 | 1/1.004 id>4mm | 0 | L 0.992 | 2 | 1081 | 0 | L 0.984 |
| w_013_right mr_004_L (071 R g10) | 1 | 144 | 1.88 | 1/1.003 id>4mm | 0 | L 1.0 | 1 | 159 | 0 | L 1.0 |
| w_017_right mr_014_L (095 R g10) | 1 | 129 | 1.84 | 1/1.020 id>4mm | 0 | L 0.986 | 1 | 151 | 0 | L 0.986 |
| control w_012_right mr_024_L | 1 | 141 | 1.82 | 1/0.988 id>4mm | 0 | ~1.0 | 1 | 163 | 0 | ~1.0 |
| control k_011_left mr_023_L | 1 | 124 | 1.89 | 1/1.003 id>4mm | 0 | ~1.0 | 1 | 120 | 0 | ~1.0 |
| w_008_right mr_001 (far clamp, B only) | 1 | 127 | 1.84 | 1/0.999 id>4mm | 0 | 0.997 / 0.981 | - | - | - | - |

More measurements:
- Every B union, collision_full (60000 tris) and OBJ (23498-23500 of 23500) is a clean tree with 0 self-intersecting
  faces.
- Routes: RCCA lumen minimum 0.70-1.77 mm and LCCA 1.15-1.75 mm, both navigable; 0 centreline points outside.
- Donor-stretch detail (fraction of area > 0.3 mm off the tube, undecimated union): 0.03-0.33 against v4's host tube
  0.000-0.008.
- Where the in-situ/alone ratio is below 0.95 (w_048 R 0.90, w_022 R 0.93), the cause is ownership near other
  vessels, not handle windows. w_022 has no window at all.
- Windows, measured from the ostium: w_012_left vaC_R 105.5-106.8 mm, m_022 120.5-123.2, w_035 84.2-85.7, w_048
  96.0-97.3. The carotid stretches stay as v4 had them.
- Full table: `bake_probe/collect_table.md`. Per-probe records: `bake_probe*/_done/<name>.{bake,measure}.json`.

## Validation of the deeper rungs

| test | setting | result |
|---|---|---|
| T1 | `V5B_BINS=""` (no bins, bisection only) | 4/4 pass, one window each of 1.35-1.77 mm at the same places as B (w_012_left 139.1-140.8 against B 139.5-140.8), 358-396 s |
| T2 | `V5B_BINS=40,80` (coarse bins, then split and shrink) | 4/4 pass, windows 1.38-3.87 mm, 343-380 s |
| T3 | `V5B_NO_LOCAL=1` (local search disabled) | rung D works: vaC_R whole tube with the course kept, flagged `donor_tube`; the flip-clamp second call ran as designed and the first result was kept; 416 s; peak 2.94 GB in the far-clamp call, after which I made the FAR check cheaper |
| selftests | cached union | bit-identical in 5 apply/undo configurations plus the far clamp and its restore |
| CLI | `main` | shard, idempotent rerun, provenance record and iso dump all work; output deterministic |

## Gate note: right_route_detail

right_route_detail equals v4 exactly in only 2 of 13 B probes (m_022, w_018b). The other ratios are fork 0.996-1.133
and siphon 0.988-1.020.

`bake_probe/diag_route.py` compares the route wall cell by cell (`bake_probe_b/_diag/<name>.route.json`). The wall more
than 4 mm (7 probes), 6 mm (4) or 8 mm (w_051) from every VA centreline sample, v4's or v5's, is **identical** to v4.
The excluded band is 8-18% of the route wall. Inside it, the VA's course has changed: either the v4 host VA refused
RCCA bulb wall at 5.8 mm (w_051, w_048), or the donor now passes close to the route.

The same numbers appear with v4's unchanged ladder (A) and at rung 1 with nothing suppressed, so the cause is the
assignment, not the bake. bake_v5b keeps the route sections exactly as v4 baked them.

A literal "ratio 1.000" gate fails 11 of 13 anatomies for this reason. I suggest the check stage gates on "identical
beyond 8 mm of every VA" (diag_route) and reports the ratios. The decreases are small, at most 1.2% (siphon w_012_right
0.988, w_035 0.989; fork w_012_left 0.996). That choice is for the user.

## Full-bake recipe

- Command, per shard i in 0..7: `python3 /sp/v5_va/bake_v5b.py --anatomies /repo/carotid_data/anatomies_v5 --shard i/8
  -v`, in neve-build-meshlab with `bake_probe/driver.sh`'s mounts, plus `-v <scratchpad>:/sp`.
- Restart-safe: anatomies that already have `mesh_v5.json` are skipped.
- Expected time: ~1.4 h (223 x ~180 s / 8).
- RAM: 8 x ~2.1 GB = ~17 GB of the 26.4 GiB VM, with at least 6 GB free even if one worker is in a ~3 GB far-clamp
  call. Check `docker stats` free RAM before starting. Do not run the SOFA check at the same time.
- Outputs per anatomy: `vessel_architecture_collision.obj` (23.5k), `collision_full.vtp`, `mesh_v5.json` and
  `provenance.json` (+`bake_v5`; licences untouched).
- Undecimated unions go to `carotid_data/va_sources/v5_reg/iso_v5/`, for the VA wall-detail check.

## Files

- Scripts in `v5_va/`: `bake_v5b.py` (the fix); `bake_probe/`: `probe_bake.py`, `probe_measure.py` (patched to read
  ladder_v5b and mesh_v5.json; the `.bak1` files are instance 1's originals), `driver.sh`, `driver2.sh` (adds
  `$DOCKER_ENV`), `in_container.sh`, `diag_handle.py`, `diag_route.py`, `collect.py`, `decide.py`, `peek_b.py`,
  `pick.py`, and the lists `probe_list*.txt`.
- Scratch bakes under `carotid_data/va_sources/v5_reg/`:
  - `bake_probe/` (A)
  - `bake_probe_b/` (B)
  - `bake_probe_t1|t2|t3/` (tests)
  - `bake_probe_cli/` (CLI test)
  - Each has `_done`, `_logs`, `_iso`, `_iso_calls` and `_diag`.
