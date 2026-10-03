# v5_route13: giving the right route its v4 wall back on 13 anatomies

Agent prefix `v5_route13`. State `v5_va/state/v5_route13.md`. Results `v5_va/results/v5_route13.json`
(`route13/collect13.py`). Scratch only: `carotid_data/va_sources/v5_reg/route13/` (gitignored). Nothing was
written into `anatomies_v4` or `anatomies_v5`. No git.

## Answer (complete, 2026-10-03 01:05)

**Moving the VA fixed all 13. Route priority is not needed anywhere.** Each anatomy has a re-planned, re-baked
scratch copy with right_route_detail at or above 0.99 of v4 on both fork and siphon, and every bake gate and every
graft gate passes. The donor pair, donor sides, assignment, licences and attributions are unchanged, and every
non-VA xform entry matches anatomies_v5.

- Picks: 12 x `moved_ext` and 1 x `moved_pin` (w_012_left mr_021_L).
  - 8 of the 13 now have a right route identical to v4, cell for cell on the undecimated union.
  - w_050, w_012_left mr_001 and w_035 read 1.000 or better on the segment that was low.
  - w_007 had both segments low; it now reads fork 1.000 and siphon 0.9993.
  - The lowest pick is w_012_left mr_021_L. Its fork reads 0.9926 at > 0.3 mm, the same as v5, because its right
    donor was kept pinned and it was never one of the low readings. Its siphon goes from 0.968 to 1.000.
- Route lumen did not change. The RCCA and LCCA min lumens on the undecimated unions are identical, v5 vs chosen, on
  all 13 (`route13/lumen_iso.json`).
  - The OBJ lumen in mesh_v5.json moves by up to 0.07 mm, for example w_024 RCCA 1.458 -> 1.392.
  - That is decimation scatter: in w_024 the untouched LCCA moves by the same amount.
- What it costs the VA:
  - Donor detail kept in situ stays at A2 0.976-1.041 on the moved VAs. Some of v5's higher values (1.05-1.21) were
    bulb wall being credited to a VA that sat inside the bulb.
  - Tubed length changes in 2 VAs only: w_035 RVA 1.48 -> 2.95 mm, and m_022 LVA 1.15 -> 1.77 mm. Each is still a
    single window.
  - m_030 drops from rung C (carotid_changed) to rung 1 with no flag.
- What it costs the course:
  - The moved donor stretch shifts by 2.8-13.9 mm RMS from v5, and the TopBrain distal piece moves with it by 4.8-27
    mm RMS.
  - The tip moves at most 38 mm from v4 (gate 60).
  - Take-off at 10 mm changes by -8 to +12 deg, and every value stays inside the 20-120 band.
  - Course RMS to the v4 corridor rises to 14.8-19.9 mm (gate 20). The separation pushes VAs toward the edge of the
    corridor; that is the price of moving them out of the RCCA's ownership reach.
- Route priority (bake_v5c.py R1-R3) was run on the v5 course for 7 of the 13.
  - It never raises right_route_detail, and it lowers it twice: m_030 fork 0.971 -> 0.851, and w_012_left
    mr_021_L siphon 0.968 -> 0.933.
  - It is reported, not chosen (section 3).

## 1. Diagnosis: which VA piece takes which RCCA wall

Tool: `route13/diag13.py` (read-only on anatomies_v4 / anatomies_v5). It applies v4's carotid sections to v4's
own tube field exactly as the owned union does (v4 margin, v4 suppressed stretches, union order) and records, for
every voxel where an RCCA real section widened the field in v4 ("v4-kept RCCA wall"), every term of
`sdf_union_v4.ownership_margin` with the two VAs kept apart. The ownership decision depends only on the voxel and
the centrelines, so v5's decision at the same voxels is exact. Check: the formula equals the rule to 0.0 and
reproduces v4's widened-voxel count on every anatomy.

The rule, written for one VA: a v4-kept RCCA voxel x stays owned iff `f_local(x) - f_VA(x) >= m` (m = 0.5 mm on all
13), with `f = r - |x - c|` over the VA's 16 nearest 0.25 mm samples. So the VA's tube wall must stay at least
(the RCCA real wall's stand-out beyond its tube + m) from x. At the bulb the real wall stands out up to 5.6-7.4 mm,
so the VA wall has to keep 6-8 mm from the RCCA tube wall on the side facing it; at the siphon 4.6-5.4 mm, so
5-6 mm.

Lost = v4-kept RCCA real voxels v5 refuses (lower_R = Zenodo fork / bulb slice; siphon_R = TopBrain siphon). w-w =
closest VA tube wall to RCCA tube wall over the contested stretch (declared radii), v5 vs v4's VA.

| anatomy | ratio fork / siphon | lost fork | lost siphon | competing piece (voxels; RCCA s mm; v5 w-w vs v4 w-w) |
|---|---|---|---|---|
| w_050_left mr_004_L | 0.819 / 1.002 | 15,362 | 4,987 | RVA donor lower VA (ia172_071 L): fork s 44-77, 1.74 vs 9.26 mm; siphon s 139-189 (donor V3), 1.06 vs 0.85 |
| m_022_left mr_021_L | 1.000 / 0.957 | 289 | 11,649 | LVA TopBrain distal (re-fitted): s 191-232, 0.90 vs 5.83 mm (11,487); RVA pieces 439 |
| k_005_right mr_024_L | 0.966 / 1.000 | 3,025 | 0 | RVA donor lower VA (Song 109 L): s 55-86, 2.75 vs 2.40 mm (net with 2,595 gained) |
| w_012_left mr_021_L | 0.993 / 0.968 | 9,610 | 9,735 | RVA donor lower VA fork s 53-124 (4.45 mm); LVA TopBrain distal s 202-231, 2.25 vs 16.1 mm (6,121); RVA TopBrain distal s 147-223 (3,524) |
| m_030_right mr_013_L | 0.971 / 1.000 | 24,995 | 3,109 | RVA donor lower VA (Song 040 L): s 42-119, 1.02 vs 5.80 mm (23,846) + 1,149 by the rung-C carotid window; RVA TopBrain distal s 131-167 |
| w_024_right mr_026_L | 0.971 / 1.000 | 25,684 | 1,799 | RVA donor lower VA (ia172_149 L): s 48-115, 1.64 vs 9.65 mm; siphon s 130-142 |
| w_012_left mr_001 | 0.980 / 1.000 | 27,371 | 544 | RVA donor lower VA (ia172_165 R): s 56-110, 1.00 vs 4.58 mm (net with 10,083 gained) |
| w_007_right mr_023 | 0.982 / 0.988 | 10,751 | 14,186 | RVA donor lower VA s 50-97, 1.29 vs 3.59 mm; RVA TopBrain distal s 188-231, 0.90 vs 2.02 mm (10,369) + ramp s 171-201 (3,660) |
| w_029_right mr_024_L | 1.000 / 0.984 | 3,238 | 2,697 | RVA ramp (donor side) s 176-195, 0.68 vs 3.07 mm (2,530) |
| w_029_right mr_026_L | 0.986 / 1.000 | 20,256 | 0 | RVA donor lower VA (ia172_149 L): s 46-91, 1.16 vs 6.79 mm |
| w_012_right mr_024_L | 1.000 / 0.988 | 135 | 10,157 | RVA donor upper V3 s 173-193, 0.85 vs 3.95 mm (4,859) + ramp s 166-186, 0.66 mm (3,972); TopBrain pieces 1,326 |
| k_008_right mr_027 | 1.000 / 0.989 | 0 | 7,842 | RVA TopBrain distal s 201-228, 1.06 vs 6.36 mm (4,086) + ramp s 201-209 (3,756) |
| w_035_left mr_016 | 1.005 / 0.989 | 333 | 6,167 | RVA TopBrain distal s 218-222, 1.56 vs 10.16 mm (4,018) + donor V3 s 135-160 (2,149) |

Per anatomy and section: `route13/diag/<name>.json` (competitors, VA arclength and distance from the ostium, the
closest approach in v5 and v4, the stand-out and the clearance the rule needs vs what the VA has, 5 mm bins of
lost / gained voxels); the voxel tables `route13/diag/<name>.rcca_v4.npz`.

Reading the table:
- The fork losses all come from the donor's own lower VA (V2 / low V3) running 0.7-2.8 mm wall-to-wall from the
  RCCA tube at the bulb, inside the bulb's real reach. The siphon losses come from the upper end of the VA: the
  donor's V3, the calibre ramp before the splice, or the re-fitted TopBrain distal piece, which in two mirrored
  anatomies is the LEFT VA's (m_022_left mr_021_L, w_012_left mr_021_L).
- Where the VA tube wall sits inside the bulb's real lumen ("had" clearance < 0: up to 1.7 mm inside), the v4 bulb
  and the v5 VA tube overlap; v4's graft gates only saw declared radii (0.6 mm clearance), not the real bulb.
- Gains (v4's host VA had refused some RCCA wall that v5 lets in) partly offset the losses in k_005, w_012_left
  mr_001 and others; the route measure counts both.

## 2. Moving the VA: the separation objective

Tool: `/sp/v5_va/graft_v5_reg_sep.py`. It is graft_v5_reg 1.1 plus the stated diffs S1-S7, which are listed in its
docstring. Every gate, constant and helper is graft_v5_reg's own:

- 0.6 mm clearance, including ramps, parent and self
- joins <= 60 deg, including the origin seam
- course RMS to the v4 corridor <= 20 mm
- tip move <= 60 mm
- take-off band 20-120 deg
- rise >= 15 deg
- splice in V3
- the calibre rule, proper rotations and follow handedness

**The ownership reach, from the union's own rule.** A v4-kept RCCA real voxel x stays owned iff, for both VAs:

    f_local(x) - f_VA(x) >= m

where m = 0.5 mm on all 13 and f = r - |x - c| over the VA's 16 nearest samples. So the VA tube wall must keep this
clearance from x:

    clearance = (stand-out of the RCCA real wall beyond its tube) + m

That works out to 6-8 mm from the RCCA tube wall at the bulb and 5-6 mm at the siphon.

**The objective.** It does not use a distance proxy. It counts LOST = the number of v4-kept RCCA real voxels (lower_R
plus siphon_R, from the tables in `route13/diag/<name>.rcca_v4.npz`) that the candidate VA would refuse under that
exact rule.

- S1: among the candidates that pass EVERY gate, take the fewest LOST. Ties go to the original order, which is pose
  nearest first, then splice move nearest first.
- S2: only the competing side is re-planned. The other side keeps its anatomies_v5 placement whenever that still
  passes the gates. It did on all 13, except where it was itself a target.
- S3 `--extended`: adds 300 two-axis pose compositions after the 31 standard poses.
- S4 `--tb-shift`: turns the re-fitted TopBrain distal piece about its splice anchor (+-3..15 deg) within its own
  gates.
- S6 `--objective net`: per segment, lost minus gained. Tried on w_012_left mr_021_L. It picked the same pose as
  `ext`, so it was not baked.
- S7 `--pin-pose`: keeps a side's v5 donor placement and varies only the TopBrain fit.

**Runs.** `ext` (`--extended`) ran on all 13 and targeted the side(s) named by the diagnosis. `pin` (`--pin-pose right
--tb-shift --extended`, both sides) ran on w_012_left mr_021_L, because `ext` had moved its right donor and dropped
the fork to 0.9887. The records are in `route13/graft/<tag>/<name>.json` and the anatomies are in
`va_sources/v5_reg/route13/moved_<tag>/<name>/`.

**Bake and measure.** Bakes used `bake_v5b.py` through `bake_full/bake_full_one.py` with standard ownership.
Measurement used `bake_full/measure_full_one.py`, the code bake:full used, with v4's union rebuilt for the route
comparison. The undecimated unions are in `route13/moved_<tag>_iso/<name>.vtp`.

## 3. Route priority (fallback): built, run on 7, never chosen

Tool: `/sp/v5_va/bake_v5c.py`. It wraps bake_v5b at run time; sdf_union_v4, bake_meshes_v4 and bake_v5b are not
edited.

- R1: the RCCA route sections ignore the VAs as competitors, but stay at least m outside the VA tube.
- R2: the VA real wall is refused where the RCCA real wall is nearer. The VA tube floor stays.
- R3: everything else is unchanged.
- Selftest: with priority off, the function equals the rule exactly.

It ran on the v5 course with `route13/run_bake.py OWNERSHIP=prio`, and the outputs are in `route13/prio/<name>/`.
Here is the measured effect (fork / siphon > 0.3 mm vs v4):

| anatomy | v5 | route priority | VA cost of route priority |
|---|---|---|---|
| w_050_left mr_004_L | 0.8191 / 1.0023 | 0.8191 / 1.0023 | RVA 1.88 mm tube window (rung 2) |
| m_022_left mr_021_L | 1.000 / 0.9567 | 1.000 / 0.9553 | none |
| k_005_right mr_024_L | 0.9660 / 1.000 | 0.9660 / 1.000 | none |
| w_012_left mr_021_L | 0.9926 / 0.9677 | 0.9926 / 0.9326 | rung C, carotid_changed |
| m_030_right mr_013_L | 0.9709 / 1.000 | 0.8512 / 1.000 | rung C as v5 |
| w_024_right mr_026_L | 0.9711 / 1.000 | 0.9711 / 1.000 | none |
| w_012_left mr_001 | 0.9796 / 1.000 | 0.9758 / 1.000 | none |

The w_007 route-priority bake was cut off by the session limit. It was not re-run, because the moved variant
already passes.

**Why route priority cannot lift this measure.** It does regain the real wall geometrically; on w_050 it adds
+9.9k real lower_R voxels. But right_route_detail gives each wall cell to the branch with the largest tube value,
r - |x - c|, and that is the same bisector the ownership rule uses. So wall regained on the VA's side counts as VA
wall: `route13/cmp_iso.py` finds every 5 mm bin identical on w_050. Where it lowers the reading, the standard
rule's owned cut face had been counting as RCCA relief, and under R1 that face is replaced by wall continuing into
the VA-attributed zone.

## 4. Pick per anatomy

Rule: take the moved variant if fork and siphon are both >= 0.99 of v4 (> 0.3 mm) with every gate; otherwise take
route priority. Every pick below is a moved variant.

- Ratios are vs v4 at > 0.3 mm / > 0.5 mm. "ident" means the route wall is identical to v4 cell for cell.
- w-w is the closest VA tube wall to the RCCA tube wall over the contested stretch.
- LOST is the number of v4-kept RCCA real voxels refused.
- Take-off is at 10 mm. Corr is the course RMS to the v4 corridor. A2 is donor detail kept in situ over alone.
- Every pick is rung 1 or 2, has 0 self-intersections, OBJ 23,498-23,500 tris, full 60,000 tris, every gate passing
  and every graft gate passing.

| anatomy | v5 fork / siphon | pick | fork | siphon | moved piece: pose (v5) | w-w v5 -> new (mm) | LOST v5 -> new | donor / TB shift RMS (mm) | take-off v5 -> new (deg) | corr v5 -> new (mm) | VA: A2 v5 -> new, tubed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w_050_left mr_004_L | 0.819 / 1.002 | moved_ext | 1.000 / 1.000 | 1.002 / 1.000 | RVA 6 deg y+6z, splice -25 (0, -5) | 1.74 -> 5.99 | 20,349 -> 3,809 | 9.0 / 9.7 | 110.8 -> 118.4 | 16.2 -> 19.2 | 1.001 -> 1.006, 0 |
| m_022_left mr_021_L | 1.000 / 0.957 | moved_ext | ident | ident | LVA 3 deg x+6y, splice 0 (-3 x, -10) | 0.90 -> 12.1 | 11,938 -> 439 | 11.9 / 19.2 | 89.4 -> 81.6 | 16.2 -> 19.6 | 0.991 -> 1.040, 1.15 -> 1.77 mm |
| k_005_right mr_024_L | 0.966 / 1.000 | moved_ext | ident | ident | RVA 9 y (3 y) | 2.75 -> 8.18 | 3,025 -> 0 | 7.1 / 13.5 | 80.3 -> 86.2 | 5.5 -> 14.8 | 1.054 -> 1.000, 0 |
| w_012_left mr_021_L | 0.993 / 0.968 | moved_pin | 0.993 / 0.986 (= v5) | 1.000 / 1.000 | RVA donor pinned, RVA TB turned -15 x; LVA 3 x+3y, splice -10 (3 x, 0) | LVA TB 2.25 -> 12.4 | 19,345 -> 10,144 (fork 9,610 as v5) | RVA 0 / 4.9; LVA 4.3 / 7.1 | LVA 86.0 -> 84.8 | LVA 19.6 -> 19.7 | R 1.026 = ; L 0.998 -> 0.992, 1.15 mm as v5 |
| m_030_right mr_013_L | 0.971 / 1.000 | moved_ext | ident | ident | RVA 9 y+9z, splice 0 (-3 x, -10) | 1.02 -> 10.6 | 27,292 -> 81 | 11.1 / 18.2 | 104.0 -> 113.5 | 10.4 -> 19.4 | 0.943 -> 1.003, 0; rung C -> 1 |
| w_024_right mr_026_L | 0.971 / 1.000 | moved_ext | ident | ident | RVA -9 x+15z (0) | 1.64 -> 7.99 | 27,483 -> 2,258 | 13.9 / 27.0 | 56.7 -> 61.4 | 12.5 -> 19.1 | 1.073 -> 1.041, 0 |
| w_012_left mr_001 | 0.980 / 1.000 | moved_ext | 1.018 / 1.036 | 1.000 / 1.000 | RVA -3 x (0); only 54 passing candidates | 1.00 -> 4.96 | 27,915 -> 17,291 | 2.8 / 4.8 | 69.4 -> 69.2 | 19.1 -> 19.9 | 1.212 -> 0.989, 0 |
| w_007_right mr_023 | 0.982 / 0.988 | moved_ext | 1.000 / 1.000 | 0.999 / 0.996 | RVA 6 y+15z (0) | 0.90 -> 1.66 (siphon) | 24,937 -> 3,877 | 11.0 / 20.2 | 84.1 -> 89.7 | 11.3 -> 17.1 | 1.000 -> 0.998, 0 |
| w_029_right mr_024_L | 1.000 / 0.984 | moved_ext | ident | ident | RVA 9 y (0) | 0.68 -> 12.7 | 5,935 -> 0 | 10.5 / 20.3 | 77.4 -> 86.2 | 3.8 -> 14.8 | 1.005 -> 1.000, 0 |
| w_029_right mr_026_L | 0.986 / 1.000 | moved_ext | ident | ident | RVA 6 y+12z, splice -15 (-3 x, -5) | 1.16 -> 3.88 | 20,256 -> 6,417 | 10.4 / 13.8 | 55.7 -> 68.1 | 12.2 -> 16.1 | 1.107 -> 1.039, 0 |
| w_012_right mr_024_L | 1.000 / 0.988 | moved_ext | ident | ident | RVA -6 x+9y, splice -10 (3 y, 0) | 0.66 -> 8.19 | 10,292 -> 603 | 9.5 / 18.8 | 79.2 -> 84.5 | 7.7 -> 19.3 | 1.002 -> 1.004, 0 |
| k_008_right mr_027 | 1.000 / 0.989 | moved_ext | ident | ident | RVA -6 x (3 x) | 1.06 -> 9.81 | 7,842 -> 0 | 13.2 / 22.7 | 78.8 -> 78.2 | 16.0 -> 18.9 | 1.001 -> 1.038, 0 |
| w_035_left mr_016 | 1.005 / 0.989 | moved_ext | 1.007 / 1.017 | 1.000 / 1.000 | RVA -9 x, splice 0 (-3 x, 5) | 1.56 -> 11.5 | 6,500 -> 0 | 8.8 / 11.1 | 75.9 -> 75.6 | 11.1 -> 18.0 | 0.996 -> 0.976, 1.48 -> 2.95 mm (1 window) |

Notes on the table:

- The "ident" rows read 1.000 / 1.000 at both thresholds.
- Where LOST stays > 0 and the route still reads >= 0.99, the remaining refused voxels are offset by wall gained
  where v4's host VA had refused it, or they lie outside the measured route cells.
- The minimum graft clearance of a moved VA is 0.71 mm (m_022 LVA) against the 0.6 mm gate. The largest join is
  59.3 deg (w_050 RVA) against the 60 deg gate. The largest take-off is 118.4 deg (w_050) against 120.
- Undecimated route min lumen is identical, v5 vs chosen, on all 13 (RCCA 0.959-1.886 mm). The per-anatomy values
  are under `route_lumen_undecimated_mm` in the results.

## 5. Ready to swap (after the bake workflow's checks / verifiers finish)

Each folder below is a complete anatomy. Its file list is identical to anatomies_v5/<name>: Centrelines_comb,
LICENSE_v5_donors.txt, collision_full.vtp, mesh_v5.json, provenance.json (with v5_va.<side>.separation) and
vessel_architecture_collision.obj.

Swapping one in means three replacements:

1. `carotid_data/anatomies_v5/<name>` -> the folder below.
2. `va_sources/v5_reg/iso_v5/<name>.vtp` -> `route13/<variant>_iso/<name>.vtp`.
3. `bake_full/_done/<name>.{bake,measure}.json` -> `route13/<variant>_bf/_done/`.

After that, re-run the targeted checks and refresh BUILD_v5.json.

- `D:\Arjun\neve\carotid_data\va_sources\v5_reg\route13\moved_ext\<name>` for the 12 moved_ext picks
- `D:\Arjun\neve\carotid_data\va_sources\v5_reg\route13\moved_pin\case_w_012_left__topcow_mr_021_L`

The other scratch folders are not for swapping:

- `route13/prio/` (route priority, 7 baked; w_007 incomplete)
- `route13/moved_net/` (not baked)
- `route13/moved_ext/case_w_012_left__topcow_mr_021_L` (superseded by moved_pin)
