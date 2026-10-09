# s2_harden_va notes -- the VA tool made stage-2 ready (graft_v6_va 1.1; D14 tools va_entry_geom 1.0 + sofa_va_entry 1.1)

Agent s2_harden_va, 2026-10-04. State: `state/s2_harden_va.md`. Code: `scratchpad/v6/va/` (runner `dr_s2.sh`, containers
`s2va_*`). Outputs: `D:\Arjun\neve\carotid_data\v6_work\va_s2\` (`/s2` in the containers). Read-only on the repo,
anatomies_v2..v5, carotid_tools, topbrain_tools.

## Answer so far

- **Tool**: `va/graft_v6_va.py` **1.1** (versioned; 1.0 frozen as `graft_v6_va_1p0.py`, 0.8 as `graft_v6_va_0p8.py`;
  every diff stated in the docstring). Folded in: route guard G1/G2/G3, distal-subclavian gate G4, the route13 slot,
  the D8 / D9 ladder with the rung recorded per side, the D10 corridor gate, the carotid side from files
  (`--carotid-root`), and for the user's run-2 decisions: D17 (the donor's own distal subclavian to a cut end, no
  seam-2 bridge), D16 (tube-bridged junction records read and flagged), D18 (= the ladder), D19 / D20 (placed and gated
  against the given moved / mirrored forks; route-guard sanity), plus `--recheck` (the VA gates re-run from files for
  the carotid -> VA -> carotid re-check). Stage-2 sources are the defaults.
- **Regressions**: 1.0 COMPAT == the pilot's guarded tree (12 / 14 byte-identical, the 2 others explained); route13
  w_018b == pilot (LOST 0); 1.1 with D17 off == 1.0 (28 / 28 sides).
- **D10**: native offsets 52 / 54 donor sides (12.9-21.0 mm, median 16.2; VMR 0248 -> fallback rule); gate = median
  within +-8 mm; medial tail a preference + flag (hard tail would block w_018b's route13 move).
- **D14**: geometric probe (v5 / stage-1 pilot, 14 anatomies) RVA 13 / 13, LVA 12 / 11; the baked D17 anatomy
  (w_034) enters on both sides. SOFA (sofa_va_entry 1.1, independent seeds; run 1's 1.0 numbers WITHDRAWN -- no device reset
  between seeds): seeds entering >= 20 mm v5 22 / 42, stage-1 pilot 30 / 42 (12 / 14 anatomies each); the baked D17
  anatomy w_034 3 / 3 (v5 2 / 3).
- **Records**: the wall agent's D16 + D17 junction records (published 16:14; byte-identical to the staged files used).
- **Unit tests (14 pilots)**: U12' joint pass on car3 (written): 28 / 28 sides pass, donor junctions 25 / 26 (the one
  rung 2 = w_050 R, ia172_071 L: its D16 junction now tried and failing the top gate -> v5 junction per D18),
  `--recheck` 14 / 14; U2' on v5: 28 / 28, 22 / 26.
- **Joint pass on the carotid tool's REAL D19 / D20 side, ALL 223** (J223; w223 + the 2 anatomies 3.0.2 re-decided):
  **donor junctions 414 / 429** (rung 2: 11, rung 3: 4; host 17); pass_all 439 / 446 (219 / 223 anatomies); the
  guard reads the moved / mirrored forks (area ratio 0.8-1.4, 0 off); carotid files byte-identical to the carotid
  tool's; RVA prefix identical 199 / 199; `--recheck` 221 / 223. Of the carotid tool's 17 fork-VA soft contacts the VA
  side clears 15; **2 need the carotid re-check (VA hard)**: w_018b_left mr_007 R (-0.71 mm vs RECA) and w_018b_left
  mr_002_L R (-2.30 mm vs RCCA) -- every rung fails there (handoff `va_s2/unit13/j223_carotid_recheck.json`). Watch:
  w_013_right mr_021_L R route excess 0.0091 (best effort). First 65 (J13) checked in detail: 130 / 130, gap >= 0.725.
- **All 223 on v5 (dry13, like stage 1's dry10)**: **donor junctions 406 / 429** (stage 1 359 / 429; dry12a before the
  D16 / D17 records 387 / 429); rung 2: 21, rung 3: 2; 31 sides moved up the ladder, none down; D16 18 / 20 sides on
  rung 1; pass_all 439 / 446 (the 7 = terminal rungs with v5's tops as shipped + pair flags, 4 anatomies).

## 1. Files

| file | what |
|---|---|
| `va/graft_v6_va.py` | the VA tool, version **1.1** (1.0 frozen `graft_v6_va_1p0.py`; 0.8 `graft_v6_va_0p8.py` == `graft_v6_va_dry10.py`) |
| `va/route_guard.py` | the pilot's route guard (copy of `pilot_build/route_guard.py`, imported by the tool) |
| `va/va_corridor.py` | D10 corridor gate: API for the grafter + tree mode for post-bake checks |
| `va/native_offsets.py` | D10 native offset of every donor side from its own spinal-canal midline (donor CT, TotalSegmentator) |
| `va/paths_d17.py` | D17: the donor's full Arterial subclavian path (stage-1 code without the 45 mm cut) -> `va_s2/paths_d17` |
| `va/va_entry_geom.py` | D14 geometric entry test (straight 0.65 mm probe from the subclavian into the VA ostium) |
| `va/sofa_va_entry.py` | D14 SOFA entry **1.1** (device reset per seed; 1.0 frozen `sofa_va_entry_1p0.py`, its multi-seed numbers withdrawn) |
| `va/sofa_worker_s2.sh` | idempotent SOFA worker (claim dirs, per-item outputs; `S2_SOFA_DIR`, `TREE_<tag>`) |
| `va/cmp_trees.py`, `va/summarize_s2.py`, `va/summarize_d14.py` | tree regression compare; record table + tree checks; D14 table |
| `va/fig_d17.py` | D17 centreline figures (front / side) |
| `va/dr_s2.sh` | container runner (`s2va_*` names; /repo read-only, /sp, /s2) |
| `va/run_unit13.sh`, `va/run_j223.sh`, `va/run_j223_sofa.sh`, `va/run_sofa11.sh`, `va/j223_post.sh` | run 3 drivers (idempotent) |
| `va/s2tmp/dry_counts.pl`, `cmp_recs.pl`, `j13_check.py` | record counts, per-side record compare, independent VA-vs-carotid gap check |

## 2. graft_v6_va 1.0 -- what changed against 0.8 (stated diffs; no gate relaxed)

The docstring of `va/graft_v6_va.py` lists every diff. In short:
- **G1-G3 route guard** folded in (was `pilot_build/va_guarded.py` source patches): G1 in `place_top` (first passing
  top with route excess <= 0.3 % of the right-route fork / siphon relief, else the least excess, flagged), G2 in the
  `place_side` pose score and early stop, G3 the guard per anatomy from the base (`build_guard`, v4 shares with the
  same code). `va/route_guard.py` = the pilot's module (imported).
- **G4** (bug fix, pilot): the distal new subclavian stretch is gated against every branch but its own side (seam-1
  parents included) in `place_side` and `final_gates`.
- **R13** the route13 objective slot (`--route13-diag/--route13-side/--route13-other-tree`), replacing
  `pilot_build/route13_v6.py`'s patching (SepObjective imported unchanged from graft_v5_reg_sep 1.1).
- **B, base from files** (joint pass): `--carotid-root <tree>` = the carotid tool's output; the VA side is reset to
  v5's in memory (== `pilot_build/make_base3.py`), unchanged curves are copied from the file they were read from,
  LICENSE files from anatomies_v5, `v6_va.base` records what differs from v5 (carotid files changed, unexpected
  changes, missing / extra curves).
- **D8**: va_target <= 20 passes; 20-30 accepted + flag `va_target_20_30`; > 30 fails the rung. orient_gates' v6
  `va_target` blocking flag is demoted to that flag when <= 30 (`orient_blocking_d8_accepted`).
- **D9 ladder** per side (`rung` recorded: 1, 2, 3, `host`, `host_v5`; the rungs tried with why in `ladder`):
  (1) donor junction + anatomical top (full pose search); (2) v5's junction + anatomical top; (3) v5's top as shipped
  (curves + xforms unchanged), flagged. A side failing the final gates after both sides are placed (exact clearance,
  subclavian stretches, orientation blocking, D8 > 30) is demoted to its next rung and re-placed (a demoted right side
  re-places the left too; up to 4 rounds). Course RMS on the **donor stretch only**: fillet .. splice against the
  host VA corridor (eve_bench host VA = v4's VA below its cut) up to the stretch end's height, 21 normalised points,
  absolute; the top is referenced to its D2 target (D8).
  Reading (reversible, knob `V6VA_D9_INHERIT=0`): on rung 2 the donor stretch is v5's shipped pose, so its D10 /
  course values are v5's and identical on rung 3; they are recorded and flagged (`corridor_fail_inherited_v5`,
  `course_rms_donor_gt20_inherited_v5`) and do not push the side to rung 3 -- rung 3 would keep the same stretch and
  only lose the anatomical top.
- **D10 corridor** (`va/va_corridor.py`): see section 3.
- **Paths** mount-independent (`/repo/carotid_data/v6_work/va/subclavian/...`), `--donor-root` (stage 2: the D15
  local-fix records), `--seg`, `--paths`, `--native`; records `--rec-dir` (default `/s2/dry/<tag>`).
- `V6VA_COMPAT_PILOT=1` switches D8/D9/D10/the donor-stretch course gate/the corridor pose order off -> the pilot's
  guarded tool exactly (regression test, section 5).

Joint pass with the carotid tool, by files (stage-2 recipe, as validated by the pilot):
1. `graft_v6_va.py --write W1 [--donor-root L]` (base anatomies_v5) -> the v6 VA side on v5's carotid side;
2. the carotid tool with W1 as its base (the v6 VAs are its soft neighbours) -> carotid tree C;
3. `graft_v6_va.py --write W2 --carotid-root C [--donor-root L]` -> the composed anatomy (C's carotid side + the VA side
   placed against it; C's own VA keys are reset to v5's in memory first);
4. w_018b: diag13 on W2, then `graft_v6_va.py --write W2 --only case_w_018b_left__topcow_mr_002 --carotid-root C
   --route13-diag <npz> --route13-side right --route13-other-tree W2`;
5. the carotid tool's file check on W2 (carotid re-check), then c4, bake.

## 3. D10 corridor gate (`va/native_offsets.py`, `va/va_corridor.py`)

- **Native offset** (donor CT, V2 = landmark C6 -> C2 on the record centreline): lateral offset of the VA from the
  TotalSegmentator `spinal_cord` centroid per 1 mm z bin (running median, own-side sign from the TS subclavian mask).
  52 / 54 sides measured: **12.9-21.0 mm, median 16.2**; cross-check against the C2-C7 vertebra-centroid midline:
  |difference| median 0.54, max 2.2 mm. VMR 0248 R / L: no NIfTI CT -> D10's fallback (no more than 8 mm medial of
  the host VA at the same z). `v6_work/va_s2/native_offsets/native_offsets.json`.
- **Host neck midline**: the midpoint of the host's own two VAs (eve_bench dualdevicenav) per z: x 17.0-22.5 over
  z 440-540 (median ~19.9); host VA half-separation ~15.5 mm (z 460-530). The confluence (x 14.8) is NOT the midline.
- **Gate** (decision as written): median over z 440-540 of the donor-owned VA points' lateral offset within +-8 mm of
  the native median. Pre-gate on each pose (donor VA up to the splice window, first in the pose order) and final gate
  on the donor part as placed. The **medial tail** (placed p10 > 8 mm medial of the native p10 -- a pose tilting V2
  across the neck) is a preference in the pose score + flag `corridor_medial_excursion`, not a hard gate: as a hard
  gate (unit round 3) it made w_018b's route13 move impossible (LOST 370 v4 fork voxels instead of 0); in the pose
  order (round 4) it crowded the route-protecting pose out of the 220-pose window. A symmetric p90 test (round 2)
  failed sides on the V3 swing inside the fixed host window and was withdrawn. Knob `V6VA_D10_STAT=tail` = hard tail.
- **Tree mode** (post-bake check): `va_corridor.py tree --root T --all`. On anatomies_v5 (429 donor sides): 129 fail
  (mostly too LATERAL; deviation p90 15.5 mm), 35 medial excursions -- v5's VAs were placed without this gate.
  Stage-1 pilot tree: 4 fail / 6 medial; the 1.0 unit tree (car3): 1 fail (k_011 R, rung 2 = v5's stretch inherited).

## 4. D14 VA entry (`va/sofa_va_entry.py`, `va/va_entry_geom.py`; not a pass gate)

- **SOFA**: sofa_rva_v6 made multi-seed (seeds 777, 778, 779; 300 steps; no early stop; entry = the tip >= 20 mm past
  the ostium on the RVA), idempotent worker `va/sofa_worker_s2.sh` (claim dirs). **Geometric**: a straight probe with
  0.65 mm clearance (0.35 wire radius + 0.3 contact) from the subclavian (antegrade starts within 25 mm before the
  take-off) into the VA; ENTERS = clear to >= 5 mm past the ostium exit.
- **WITHDRAWN (run 3, 16:34): the SOFA columns below were measured with sofa_va_entry 1.0, which never reset the
  devices between seeds** (eve's Intervention.reset() reloads the scene only when the mesh changes; eve.Env calls
  reset_devices() for that) -- seeds 778 / 779 continued from seed 777's wire state, and the initial device rotations
  came from an unseeded rng. Fixed in 1.1 (section 8); the geometric columns are unaffected. Same pattern in
  pilot_build/sofa_rva_v6.py and v5's checks/sofa_rva.py (their extra episodes continued from a failed state).
- 14 pilot anatomies, v5 vs the stage-1 pilot tree (same seeds; `v6_work/va_s2/d14/d14_summary.json`):

| | SOFA seeds entering >= 20 mm | anatomies with any entry | geometric RVA enters | geometric LVA enters |
|---|---|---|---|---|
| v5 | 12 / 42 | 6 / 14 | 13 / 14 | 12 / 14 |
| stage-1 pilot (v6, seam-2 bridge) | 22 / 42 | 9 / 14 | 13 / 14 | 11 / 14 |

  k_011 R fails the geometric probe in both (v5's stub kept there). The 1.1 tree (D17 cut-end arm side) needs a bake
  before SOFA; the integrator's re-pilot runs the same two tools on it.

## 5. Unit tests of 1.0 (dry; 14 pilot anatomies; `v6_work/va_s2/unit/`)

- **COMPAT regression** (`V6VA_COMPAT_PILOT=1`, base car3) == the pilot's guarded tree: 12 / 14 anatomies all 18
  curves byte-identical; w_025 RVA differs only by the c4 radii; w_018b = the route13 case (separate test); xforms
  equal except the 'wall' label text. Route13 (w_018b, `--route13-diag`) == pilot: 18 / 18 curves, pose aim2_20:turn+6y,
  LOST 0.
- **U1 joint pass** (base = pilot car3, written): 28 / 28 sides pass every gate; rung 1: 21, rung 2: 5, host: 2;
  D10 ok 25 / 26 (k_011 R inherited from v5 on rung 2); va_target 20-30 mm: 3; donor-stretch course RMS max 19.7;
  route excess max 0.0008 (tol 0.003); non-VA curves == base 14 / 14; RVA prefix identical on 9 / 9 right donor
  junctions. U2 (base v5, dry): 28 / 28 (rung 1: 20, rung 2: 6, host 2). U3 (test knob: accept <= 8 mm): rungs 2 / 3
  and the demotion loop exercised. U6 (D15 local-fix donor records + local junction records): 14 / 14 ran.

## 6. Run 2: the user's decisions D16-D20 -> graft_v6_va 1.1

The relaunch (11:28) brought D16-D21. 1.0 is frozen as `va/graft_v6_va_1p0.py`; the docstring of 1.1 lists the diffs.

**D17 -- the donor's own distal subclavian, cut end, no seam-2 bridge.**
- `va/paths_d17.py`: stage 1's `subclavian_paths.py` imported unchanged and re-run without its 45 mm distal cut; the
  path ends where the Arterial xSA chain ends inside the CT (3 mm FOV margin). 54 / 54 sides: same chain and the
  same arclength origin as the stage-1 path (prefix deviation 0.0 mm on every side); end 19.0-121.1 mm past the
  ostium (median 74.4); every end is the chain end inside the CT; < 30 mm: song_109 R 19.0, song_040 R 20.6.
  `v6_work/va_s2/paths_d17/`.
- Request to the wall agent (`wall/requests.json` va-D17-1, ACK 12:15): label + surface extended along these paths
  to their end with the D15 local-fix rule, records published over `donors_local/_junction_records` (ETA ~14:00);
  their one clip: where Arterial's calibre collapses at the chain end (8 sides, 0-12 mm) the label stops earlier.
- The tool: the stretch and every stretch gate are unchanged (seam 1 .. foot + 20, arm_dir on foot .. foot + 20).
  The (17) / (19) branch = F .. the stretch .. the donor's path to min(label end - 0.5, path end), same rigid pose;
  a straight tube at the median radius of the last 5 mm along the last 5 mm chord to 30 mm past the foot when the
  label ends short (flag `d17_tube`). Gates on the branch as written:
  - exit its parents within 30 mm and never re-enter them -- the parents as `final_gates` has them (right: the whole
    RVA = the proximal new part, the fillet and the posed donor VA; left: (18) + the LVA start). A first version used
    only the proximal part + fillet in the search, so m_022 mr_021_L R passed the search and was demoted by the final
    gates (the (17) re-entering its own RVA); the search now uses the final gates' parents (stricter, not looser);
  - clearance >= 0.6 to every branch but its own side (the G4 rule, the whole branch);
  - 1 mm turn <= 60 on what D17 adds (from foot + 19: the extension's join with the stretch, the tube start, the
    rest). F .. foot + 20 is 1.0's stretch and is not re-gated: song_095 R's own Arterial path has a 108 deg corner at
    its ostium (a segment junction; it was in 1.0 / stage 1 too) -- recorded (`max_turn_branch_deg`, flag
    `d17_branch_kink`), not a reason to lose the junction. Every other path is <= 28 deg per mm past the ostium
    (`s2tmp/kinks.pl`);
  - no hairpin = no FOLD: two abutting 10 mm chords turning > 120 deg (where the second reaches the added part). The
    gradual arch over the first rib (any 10 mm chord vs the first one) is recorded only (`arch_deg`): a real
    subclavian turns 60-115 deg there (the first draft gated that arch at 120 and would have trimmed real anatomy);
  - no self-contact (window 6 x the radius).
  My decision (reversible, `V6VA_D17_TRIM=0`): a failure only beyond 30 mm past the foot moves the cut end back to
  3 mm before it (flag `d17_trimmed`) instead of losing the donor junction -- no gate is relaxed; the end is where
  the written branch passes them. The bridge term leaves the pose cost. The host's distal subclavian is dropped on
  rung-1 sides; rung 2 / 3 / host sides keep v5's (17) / (19) byte for byte.
- Bake records: `xform.scD_*` hi_mm = the label end on the branch, `scD_*.d17` = {cut_end, end_mm, foot_mm,
  past_ostium_mm, label_end_mm, tube_from_mm, tube_mm, trimmed_mm, host_distal_dropped_mm} (request va-D17-bake-1 to
  the bake: cut-end treatment, tube part an ordinary tube).

**D16 -- tube-bridged pinched ostia.** The wall agent builds the tube into the junction records (request va-D16-1,
ACK: local_sub 1.1 d16_tube, 6 mm + 2 mm steps until the bottleneck ratio >= 0.7, max 16 mm; records with
`d16_tube` and junction_failed null, with the D17 records). 1.1 reads them as they come: junction_failed null ->
rung 1 is tried; flags `junction_tube_bridge` (also stage 1's fallback_tube sides 136 L / 172 L) and
`d16_tube_bridge` (+ tube mm in the record).

**D18** = the D9 ladder: rung 1 (with D16 / D17) -> rung 2 (v5's junction + anatomical top, flagged) -> rung 3.

**D19 / D20 -- the right fork moves.** Every placement gate reads the base's curves, so with `--carotid-root C` the VA
side is placed and gated against C's re-posed / mirrored forks. The route guard is built from C's provenance (xform
lower / siphon with their mirror, host_cut_mm): 1.1 adds a sanity record (relief area per segment beside the same
code on v4; flag `route_guard_area_low` when < 0.5 x v4 or 0 -- a fork pose the guard cannot read is never silent).
RCCA / RECA changes are expected carotid changes in `v6_va.base`. New `--recheck TREE`: the VA side's gates re-run
from files on a composed tree (final_gates: exact VA clearance from exit, the new stretch and the D17 branch vs every
other branch; RVA prefix vs v5; D17 end vs record) -- the closing step of carotid -> VA -> carotid re-check when the
carotid pass changed anything after the VA pass.

**Defaults** are the stage-2 sources: donor records `wall/donors_local` (D15), junction records
`wall/donors_local/_junction_records`, paths `va_s2/paths_d17` (COMPAT keeps 1.0's).

**Joint pass for the integrator (1.1; eve-training-fixed; /repo read-only; records with --rec-dir):**
1. `graft_v6_va.py --write W1 --names-file N --workers 4 --rec-dir R1` (base anatomies_v5) -- the v6 VA side for the
   carotid tool to see as soft neighbours (optional; the pilot did it);
2. the carotid tool (D19 / D20 right + left forks) with W1 as its base -> C;
3. `graft_v6_va.py --write W2 --carotid-root C --names-file N --workers 4 --rec-dir R2` -- the VA side placed and
   gated against C's moved / mirrored forks (C's own VA keys are reset to v5's in memory first);
4. w_018b: diag13 on W2, then `--write W2 --only case_w_018b_left__topcow_mr_002 --carotid-root C --route13-diag
   <npz> --route13-side right --route13-other-tree W2`;
5. carotid re-check: the carotid tool's file check on W2, and `graft_v6_va.py --recheck W2 --names-file N --rec-dir
   R3` (exit 0 = every VA-side gate still holds on the files as written; nothing re-placed); then c4, bake_v6;
   (run 3 did steps 3 + 5 for all 223 on the carotid tool's w223 tree: `va/run_j223.sh`, records `va_s2/unit13/rec_j13`,
   tree `va_s2/unit13/tree_j13` -- usable as W2 if the carotid side does not change again; step 4 needs the bake);
6. post-bake: `va_corridor.py tree --root <baked tree> --all` (D10 as built), `va_entry_geom.py` + `sofa_va_entry.py`
   (D14 report) -- sofa_va_entry 1.1 (device reset per seed; `va/run_sofa11.sh` is the worker pattern, set
   TREE_<tag> to the baked tree), D17 tube / trim / kink flags from provenance `v6_va.record.sides.<side>.flags`.

## 7. Unit tests of 1.1 (dry; `v6_work/va_s2/unit11/`, `unit12/`)

- **U7 regression** (`V6VA_D17=0`, same stage-2 sources as 1.0's U6, base car3): 28 / 28 sides identical to 1.0
  (rung, junction, pose, va_target, pass) -- the 1.1 changes are neutral when D17 is off (`s2tmp/cmp_recs.pl`).
- **U1' first draft** (D17 on, base car3, written; before the gate refinements of 12:20): 28 / 28 sides pass every
  gate; donor junctions 25 of 26 donor sides (1.0 with the same sources: 21 -- the seam-2 bridge gates no longer
  reject poses); the one rung-2 side is w_050 R (ia172_071 L, junction_failed: D16 pending). D17 branch 29.8-33.6 mm
  past the ostium (today's labels end ~32 mm past); tube on 2 sides (w_034 R, w_035 L: the < 30 mm donors, up to
  11.9 mm); trimmed 0; 1 mm turn max 23 deg on the branch. Tree checks: every non-VA curve == the base (14 / 14), RVA
  prefix identical on every right donor junction, (17) / (19) end == the recorded cut end, v5's files kept on rung-2 /
  host sides.
- **U8 synthetic moved right fork** (the D19 stand-in until the carotid tool's tree exists; `s2tmp/syn_d19.py`): the
  right fork + siphon of k_004 and w_027 (car3) turned 15 deg about z through the RCCA point at host_cut, curves AND
  xforms (good) / curves only (bad = stale fork pose). Good: the base record lists RCCA / RECA (/ LCCA, car3's own) as
  expected carotid changes, 0 unexpected; both anatomies place rung 1 / rung 1 against the moved fork (w_027 R va_target
  13.3 -> 4.0 mm: the search used the new geometry); independent check (`s2tmp/u8_check.py`, all point pairs): RVA
  beyond 30 mm and (17) vs the moved RCCA / RECA >= 3.04 mm; `--recheck` on the written tree: 2 / 2 ok. Guard relief
  area v6 / v4: good x1.00-1.14; **bad x3.5-4.3 on the fork** (a stale pose INFLATES the area: every vertex stands
  off the tube) -> the sanity flag was made two-sided (outside [0.5, 2.0] x v4 -> `route_guard_area_off`).

## 8. Run 3: final numbers with the wall agent's D16 / D17 records (relaunch 16:10)

**Records used.** The wall agent's D16 + D17 junction records were complete at 14:24 (chain D17 finished; publish
dry run: 54 sides, 0 problems) but not yet published over `donors_local/_junction_records` when this run started. The
tool reads them where they are staged (`--seg /repo/carotid_data/v6_work/wall/d17/_junction_records`). The wall agent published at 16:14
(requests va-D16-1 / va-D17-1 DONE): the 54 published `donors_local/_junction_records/*/*/junction.json` are
byte-identical to the staged files used here (md5 list `va_s2/unit13/staged_records_md5.txt`), so these results are
the tool's DEFAULT inputs. Both D16 sides
carry `d16_tube` and `junction_failed = null` (ia172_164 L: bottleneck ratio 0.459 -> 0.897, tube 9.71 mm r 1.52;
ia172_071 L: 0.291 -> 1.0, tube 10.01 mm r 2.08). Runner `va/run_unit13.sh` (containers s2va_u13a / s2va_u13b).

**U12' -- the 14 pilots, joint pass on car3, written** (`va_s2/unit13/summary_car3.json`, logs `u13_car3.log`,
`u13_recheck.log`): 28 / 28 sides pass every gate; donor junctions 25 of 26 donor sides; rung 2: 1 (w_050 R =
ia172_071 L: with D16 its junction is now TRIED on rung 1, and no pose passes the top gate (best aim6_20:turn+3y fails
'top') -> v5's junction per D18, flagged); host 2. D10 ok 26 / 26. D17 branch 29.8-119.7 mm past the ostium (median
76.1; the labels now run to the end of the donor's segmentation), tube on 1 side (w_035 L, 11.9 mm), trimmed 0, 1 mm
turn on the added part max 25 deg, fold test max 67 deg. `--recheck` on the written tree: 14 / 14 ok (rc 0). Tree
checks: every non-VA curve == the base (14 / 14), RVA prefix identical on every right donor junction, (17) / (19) end
== the recorded cut end, v5's files on rung-2 / host sides.

**U2' -- the 14 pilots on anatomies_v5 (dry)** (`va_s2/unit13/counts_v5.json`): 28 / 28 pass; rung 1: 22, rung 2: 4,
host 2 (run 1's U2 with 1.0 and the stage-1 records: 20 / 6 / 2). The base matters: on car3 / the real v6 carotid side
the same sides take more rung 1 (k_004 mr_010 R is rung 2 on v5's carotid side, rung 1 on both v6 carotid sides).

**J13 -- the joint pass on the carotid tool's REAL D19 / D20 trees** (`carotid/s3/w223`, compose_v6 3.0.x, the 65
chunk-1 anatomies written + verified by s2_harden_carotid2; right forks re-posed, contralateral forks x-mirrored; 62 / 65
with the right fork moved). `graft_v6_va.py --write --carotid-root w223` (records `va_s2/unit13/rec_j13`, tree
`tree_j13`, counts `counts_j13.json`, summary `summary_j13.json`, independent check `j13_check.json`):
- 65 / 65 anatomies, 130 / 130 sides pass every gate; donor junctions 121 of 123 donor sides (rung 2: k_008_right
  mr_001_L R -- the donor stretch fails D10, inherited from v5 on rung 2; k_011_left mr_013_L R -- rung 1 blocked by
  the orientation 'droop' gate); host 7. D16: ia172_164 L junction (tube 9.71 mm) holds on rung 1 on both of its
  chunk-1 sides (k_004_left mr_011_L R, w_003_right mr_011_L R).
- Route guard on the moved / mirrored forks: relief area v6 / v4 fork 1.0-1.3, siphon 1.0-1.2, 0 `route_guard_area_off`
  (the guard reads compose_v6 3.0's `xform.lower.mirror` [-1, 1, 1]); route excess max 0.0008 (tol 0.003).
- D10 121 / 123 donor sides ok (the 2 = v5-inherited stretches on rung 2); donor-stretch course RMS max 19.8; va_target
  median 8.3, max 28.3 (20-30: 6 sides, flagged); take-off vs native |error| median 0.1, max 0.7 deg.
- D17: branch 29.2-119.7 mm past the ostium (median 66.3), tube on 14 sides (<= 11.9 mm: the song_040 R / song_109 R
  donors), trimmed 0, fold max 72 deg, `d17_branch_kink` 1 (song_095 R). Pose-level D17 rejects are mostly the
  branch passing within 0.6 mm of the (moved) RCCA (720 of 768 pose rejects) -- the search moves on to the next pose.
- `--recheck` on the written tree: 65 / 65 ok (rc 0). Tree checks: non-VA curves == the carotid tool's files on 65 / 65
  (byte for byte), RVA prefix identical on 58 / 58 right donor junctions, (17) / (19) end == record.
- Independent check (`s2tmp/j13_check.py`, all point pairs, RVA / LVA from 30 mm and (17) / (19) from 5 mm vs RCCA,
  RECA, LCCA, LECA): smallest lumen gap 0.725 mm, 0 below 0.6.
- D10 tree mode on the written tree (`unit13/corridor_tree_j13.json`): 121 / 123 donor sides ok (the 2 rung-2 sides fail,
  inherited from v5), 1 medial excursion. (U12' tree: 26 / 26 ok, 1 medial -- `corridor_tree_car3.json`.)

**D14 bug found and fixed (sofa_va_entry 1.1).** Running D14 on the one baked D17 anatomy (the wall agent's bake_v6 1.1 +
checks_v6 1.1 of the unit11 tree's w_034: `bake_v6/d17bake/tree`, GATES PASS) showed episode 1 (seed 778) starting
with the wire already 190 mm in: eve's `Intervention.reset()` reloads the SOFA scene only when the mesh / insertion
changes, and eve.Env resets the devices with a separate `reset_devices()`, which 1.0 never called. So in 1.0 seeds
778 / 779 CONTINUED from seed 777's wire state, and the initial device rotations came from the simulation's unseeded
rng. Run 1's multi-seed SOFA numbers (section 4) are withdrawn. 1.1 (S4): every episode = `iv.reset(...)` +
`reset_devices()` with the simulation rng seeded by the episode's seed (the initial device rotations -- the only random
input of the scripted controller -- are a function of the seed, identical on every tree); the inserted lengths and
rotations after each reset are recorded per seed. Checked: episode 1 now starts at the insertion (guidewire 9.5 mm in at
step 9). The same pattern is in `pilot_build/sofa_rva_v6.py` and v5's `checks/sofa_rva.py` (their extra episodes
continued from a failed state rather than retrying). Re-run (`va/run_sofa11.sh`, list `s2tmp/sofa11.txt`: d17b w_034,
v5 x 14, stage-1 pilot x 14 -> `va_s2/sofa11/`).

**Figures** (`va/fig_d17.py`, centrelines front / side, the donor's own distal subclavian in orange to its cut end, the
host's dropped (17) / (19) dashed): `va_s2/figs/unit13_j13/` (k_004_left mr_011_L and w_003_right mr_011_L = the D16
ia172_164 L junction on rung 1 against the moved fork; k_005_right mr_013_L = a D17 tube side) and
`va_s2/figs/unit13_car3/` (w_034 R, m_006 mr_027_L).

**All 223 on anatomies_v5 (dry13; like-for-like with stage 1's dry10)** (`va_s2/dry13/rec`, counts
`va_s2/dry13/counts.json`, logs `dry13_s0/s1.log`): 223 / 223 ran, 0 errors.
- **Donor junctions 406 of 429 donor sides** (stage 1 dry10: 359; dry12a with 1.1 before the D16 / D17 records: 387).
  Rung 1: 406, rung 2 (v5's junction + anatomical top, D18): 21, rung 3 (v5's top as shipped): 2; host sides 15 +
  host_v5 2 (the 17 no-donor sides, D21 out of scope).
- Against dry12a: 31 sides changed, every one UP the ladder (rung 2 -> 1: 15, rung 3 -> 1: 4, host_v5 -> host: 2, the
  rest same rung with another pose); none down.
- D16: the 20 tube-bridged sides -- ia172_164 L 16 / 16 on rung 1; ia172_071 L 2 / 4 (w_013_right mr_004_L R and
  w_050_left mr_004_L R: the junction is tried and no pose passes the TOP gate -> rung 2 per D18).
- Why rung 1 did not hold on the 23 rung-2 / 3 sides (`counts.json` rung1_not_held): the top gate (va_target / D2
  top) 13, course RMS 2, orientation blocking 2 (rise_err, midline), corridor 1, VA pair 1, arm_dir 1, D17 distal
  clearance 1, final exact clearance vs the other VA 2.
- pass_all 439 / 446 sides, 219 / 223 anatomies. The 7 sides (4 anatomies) are terminal rungs with v5's top as
  shipped that fail a v6 orientation gate / D8, plus their partners' pair flags: w_008_right mr_020 R (rung 3,
  va_target 22.9 + heading / midline), w_017_left mr_002_L R (rung 3, va_target 55.3) + L (rung 2, pair), w_018_right
  mr_022 L (host_v5, va_target 51.1) + R (pair), w_037_right mr_010_L R (host_v5: v4's host VA kept since v5, va_target
  63, and RVA-RECA 0.475 mm -- PRE-EXISTING in anatomies_v5, measured there; on the carotid tool's tree its RECA moved
  away: 11.7 mm) + L (rung 2, pair). These are on v5's carotid side; the joint pass on the real v6 carotid side below
  is the build's answer.
- D17: branch 29.2-119.7 mm past the ostium (median 70.1); tube on 38 sides (<= 11.9 mm), trimmed 0, fold max 72 deg,
  `d17_branch_kink` 8 (all song_095 R: its own Arterial path's 108 deg corner at the ostium, inside 1.0's stretch).
  Take-off vs native |error| median 0.1, max 1.1 deg. Guard area ratio 1.0 (v5's own forks).
- One-to-one donor units: the tool never reassigns donors (v5's assignment, `v6_va` records the donor per side);
  25 TopBrain patients <-> 25 donors (checked on the 223 records).

**J223 -- the joint pass on the carotid tool's REAL D19 / D20 side for ALL 223** (`carotid/s3/w223`, compose_v6 3.0.x,
223 / 223 written + verified by s2_harden_carotid2; the 2 anatomies its 3.0.2 re-decided -- w_017_right mr_011,
w_050_left mr_005 -- placed against `carotid/s3/t11/write`). `va/run_j223.sh` + `run_j223_t11.sh` (`--write` into
`va_s2/unit13/tree_j13`, records `unit13/rec_j13`; counts `counts_j223.json`, summary `summary_j223.json`, independent
check `j223_check*.json`, D10 tree `corridor_tree_j223.json`, handoff `j223_carotid_recheck.json`):
- **Donor junctions 414 of 429 donor sides** (v5 base dry13: 406; stage 1: 359). Rung 1: 414, rung 2: 11, rung 3: 4;
  host 17 (every no-donor side got its anatomical top on the host VA; host_v5 0). The real v6 carotid side frees room
  for the VA side: more rung 1 than on v5's carotid side.
- pass_all 439 / 446 sides, 219 / 223 anatomies. `--recheck` 221 / 223 ok; independent gap check: every VA-side curve
  >= 0.646 mm from every carotid curve EXCEPT 2 sides; carotid files == the carotid tool's (223 / 223, byte for byte);
  RVA prefix identical on 199 / 199 right donor junctions; (17) / (19) end == record 223 / 223; D10 as built 419 / 429
  donor sides (the 10 = v5-inherited stretches on rungs 2 / 3), 3 medial excursions.
- **The carotid tool's 17 fork-VA soft contacts** (its open issue 1, `dry223_v302_report.txt`): the VA side clears 15
  (re-placed on rung 1 / 2). **2 remain -- the joint pass fails there and needs the carotid re-check with the VA hard**
  (`unit13/j223_carotid_recheck.json`): w_018b_left mr_007 R (rung 3: v5's RVA vs the moved RECA -0.71 mm; rung 1's
  best pose has donor course RMS 31.9 mm on the VMR 0248 donor, rung 2: all 30 tops fail clearance -- the contact is on
  v5's donor stretch, which no top can move) and w_018b_left mr_002_L R (rung 3: -2.30 mm vs the moved RCCA; rung 1:
  no pose passes the top gate; rung 2: all 30 tops fail clearance). Both are w_018b siblings (the host whose RCCA runs
  against the RVA: the route13 case). No D9 / D18 rung can clear them without relaxing a gate; the fork pose must
  change (the carotid tool's VA-clean R2 shift is capped at 10 mm and treats the VA as soft).
- The other 5 not-pass sides are terminal rungs flagged per D9 (v5's top as shipped failing a v6 orientation gate /
  D8, clearance fine) and their partners' pair flags: w_017_left mr_002_L R (rung 3, va_target 55) + L, w_039_right
  mr_001_L R (rung 3, va_target 45: on the moved fork every rung-2 top fails clearance; on v5's carotid side it was
  rung 2) + L.
- Route guard: relief area v6 / v4 fork 0.8-1.4, siphon 1.0-1.2, 0 `route_guard_area_off`; route excess <= tol on 441
  of 442 placed sides; **w_013_right mr_021_L R: best effort, excess 0.0091** (no passing pose under 0.003 against the
  moved fork; v5's carotid side: 0) -- watch list for the post-bake right-route check (>= 0.99 of v4 detail).
- D17 on rung 1: branch 29.2-119.7 mm past the ostium (median 69.2), tube 39 sides (<= 11.9 mm), trimmed 0, fold max
  72 deg, `d17_branch_kink` 8 (song_095 R). Take-off vs native |error| median 0.1, max 1.2 deg.

## 9. Open issues (for the orchestrator / integrator)

1. **Carotid re-check with the VA hard on 2 anatomies** (`va_s2/unit13/j223_carotid_recheck.json`): w_018b_left mr_007 R
   (v5's RVA vs the moved RECA -0.71 mm) and w_018b_left mr_002_L R (vs the moved RCCA -2.30 mm). The VA side cannot
   clear them on any D9 rung without relaxing a gate (rung 1: course RMS 31.9 on the VMR donor / top gate; rung 2: all
   30 tops fail clearance because the contact is on v5's donor stretch). The carotid tool treats VAs as soft and caps
   its VA-clean R2 shift at 10 mm; these two need a fork pose that respects the VA (or an orchestrator decision).
2. **Terminal rungs (flagged, clearance fine)**: w_017_left mr_002_L R (rung 3, va_target 55), w_039_right mr_001_L R
   (rung 3, va_target 45 -- on the moved fork every rung-2 top fails clearance) and their partners' pair flags.
3. **Route watch**: w_013_right mr_021_L R route excess 0.0091 (> 0.003, best effort) -- check its right-route detail
   post-bake (>= 0.99 of v4).
4. **w_018b_left mr_002 (route13)**: the route13 move (`--route13-diag`) needs diag13 on the BAKED joint tree; J223's
   record for it is the plain joint placement (rung 1). Integrator step 4 of section 6.
5. **D14 SOFA**: `pilot_build/sofa_rva_v6.py` and v5's `checks/sofa_rva.py` share 1.0's missing device reset (their
   extra episodes continued from a failed state); use `va/sofa_va_entry.py` 1.1 for any multi-seed run. D14 on the v6
   anatomies needs the integrator's bake (only w_034's D17 bake exists).
6. **D17 kink flag**: song_095 R's own Arterial path turns 108 deg in 1 mm at its ostium (inside 1.0's stretch, not a
   join); 8 sides flagged `d17_branch_kink`, not gated.
7. **D21** (no donor, 17 sides) out of scope: they keep the host VA; the joint pass placed all 17 tops anatomically
   (rung host).

**D14 with sofa_va_entry 1.1** (independent seeds 777 / 778 / 779, 300 steps; geometric probe 0.65 mm;
`va_s2/sofa11/d14_summary11.json`, per item `va_s2/sofa11/{v5,v6p,d17b}/`; reported, not a gate):

| tree (14 pilot anatomies) | SOFA load pass | seeds >= 20 mm into the VA | anatomies with any entry | geometric probe enters R / L |
|---|---|---|---|---|
| anatomies_v5 | 14 / 14 | 22 / 42 | 12 / 14 | 13 / 14, 12 / 14 |
| stage-1 pilot (donor junctions, seam-2 bridge) | 14 / 14 | 30 / 42 | 12 / 14 | 13 / 14, 11 / 14 |
| baked 1.1 D17 anatomy (w_034 only) | 1 / 1 | 3 / 3 (146, 137, 146 mm) | 1 / 1 | R yes / L yes |

Same anatomy w_034: v5 2 / 3 seeds (2, 115, 115 mm), stage-1 pilot 2 / 3 (135, 120, 3), the D17 cut-end bake 3 / 3.
Not entering in any tree: k_011 R (v5's stub kept in both v5 and the pilot; the geometric probe fails there too) and
w_025 R on the pilot (13-14 mm: the narrow c4 VA). The run-1 (1.0) numbers (12 / 42 vs 22 / 42) understated both
trees because seeds 2-3 continued from seed 1's failed state. A v6 D14 over the 25-anatomy SOFA set needs the
integrator's baked joint tree (the same two tools, `run_sofa11.sh` pattern with `TREE_<tag>`).

Resumed 2026-10-06 after the weekly-limit reset: the D14 rerun had finished (29 / 29 outputs, summary written after the last one); the table above == `d14_summary11.json`. Nothing lost.

Status: complete
