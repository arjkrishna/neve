# verify_v5_final: adversarial check of the finished v5 set, its record, toolchain copy and images

Agent prefix `verify_v5_final`. State `v5_va/state/verify_v5_final.md`. Work folder `v5_va/verify_final/`.
No git writes; `/repo` was mounted read-only in every container. Nothing was written outside the scratchpad, apart from
one stray file that my own crashed shell loop created and I then removed (see the end).

Status: complete (2026-10-03 17:05 EDT).

## Answer so far

**Verdict: holds = false, on one narrow point.** Every data claim reproduces with my own code: the set, the gates,
the split, the record's numbers, the images, and v4 and the tracked code being untouched. Claim (3) fails in one place.
The toolchain copy omits the script version that made 12 of the 13 shipped route moves. A one-file copy fixes it.

The interruptions did no damage. There are no partial files, every JSON parses, every hash chain is intact and no
container is left running.

### (1) anatomies_v5 = verified 210 + the 13 route13 picks; gates; split

- **Hashes** (`verify_final/hash/`, sha256 of every file, my own manifests):
  - 13/13 anatomies_v5 folders equal their route13 pick on all 23 files. The iso_v5 union and both bake_full
    `_done` markers also equal the pick.
  - Each folder differs from its backup in 5 files (6 for the moved_pin anatomy).
  - The variants equal `results/v5_route13.json` → chosen: 12 moved_ext, plus w_012_left mr_021_L as moved_pin.
  - No file under route13/ is newer than its verifier (03:05).
  - The other 210 folders have no file newer than 2026-10-02 20:31, so they are the bake that verify_v5_mesh audited.
  - Since then only the 13 folders, `BUILD_v5.json` and `_records/` have changed.
  - `BUILD_v5.json` sha256 is 907b7717…, equal to swap13/build.
- **Own audit** (`verify_final/own_audit.py`, numpy/scipy only, vtk used only to read files; `own_selftest.py` PASS):
  - Self-tests: sphere genus 0, torus genus 1, crossing spheres flagged SI, pinch found, winding, and exact distance
    checked against brute force.
  - Sample: the 13 swapped anatomies, 10 random others (`awk srand(20261003)`, `rand10.txt`), plus
    w_018b_left mr_002. That is 24 in all.
  - Result (`audit/*.json`, `audit_sum.py`): the OBJ, collision_full and undecimated union pass on 24/24.
    - Each is one face component and closed, with 0 non-manifold edges, 0 orientation conflicts and 0 pinch vertices.
    - χ = 2 (genus 0), and the signed volume is negative (inward normals).
    - My own self-intersection test finds 0 SI faces. 3 near-flat edge folds sit on k_004_left mr_018's union, as
      verify_v5_mesh already noted.
    - 0 centreline points lie outside OBJ or full, with end caps included.
    - RCCA and LCCA are navigable (v4's rule).
    - Triangle counts equal the records, and so does the count of checked centreline points.
    - Route lumen equals the bake record to within 0.0005 mm.
- **Records over all 223** (`scan_prov.py` → `scan_prov.json`):
  - Every JSON parses.
  - mesh_v5 v5_gates all_pass holds on 223.
  - Rungs are 193/3/25/2. Carotid margins are 201/22 and VA margins 218/5. carotid_changed is set on the 2 w_016 anatomies.
  - Handedness is correct on all 429 donor VAs.
  - 25 donor units map one to one onto the 25 TopBrain patients.
  - 17 sides are kept from v4, and 13 anatomies carry v5_va.route13.
  - Licences: 216 NC, 7 VMR. LICENSE_v5_donors is in 223 folders and LICENSE_VMR_0248 in the 7 VMR folders.
- **Split**: the original `v5_split.py` on the final set with `--baseline split_v4.json` exits 0, and none of the 45
  quantities is lowered. The V donors are one-to-one nested (`verify_final/split/`, `logs/split.log`).

### (2) CAROTID_V5.md against its cited files: more than 120 numbers checked

- **At a glance**: every row checked.
- **Yield table**: I recounted it from scaleup_tally (`tally_ne.py`):
  - 172-series: 37 graded, 14 eligible, 17 fail QC, 6 fail V4.
  - Song: 35 graded, 12 eligible, 15 fail QC, 8 fail V4.
  - The 8 provisional pairs, 26 all-bins and 19 contact-free figures also match.
- **Primaries**: 25 above the dense host on all bins and 18 contact-free.
- **Assignment**:
  - Ostium advance 421/8/2; stub median 5.28 and max 10.49.
  - Dry run: 12,042 sides, 8,688 passing (72.2%), 0 errors; ia172_159 L passes 12 times.
  - Alternative order: 33.28 → 39.37, costing 24 and 24.
  - 419/431; 127.5° and 120.8°; 10 splices at C2; Kabsch 0.034.
- **Patient table**: all 25 rows recomputed from provenance (anatomies, native/mirrored, RVA, LVA, both/one). The
  contact-free R/L values equal `v5_assignment.json`.
- **Kept-v4 table**: the 17 rows and their reasons equal BUILD and notes/v5_assign.
- **Bake**:
  - Rungs, margins, windows (25; 1.15–9.14 mm, median 1.31; 0.55–6.2%; 46–125 mm, 19 of them in 83–125).
  - Detail kept 0.999 (0.81–1.27).
  - Timings: 172.7 s (121.6–496.5), 11.9 CPU-h, RSS 1.80–2.16 GB.
  - obj_fix on 3 anatomies.
- **Gates table**: every row.
- **Wall table**: all 10 rows, with medians, p10–p90 and min–max.
- **Right route table**: recomputed by my own aggregation of the 223 `swap13/rrdx` records (`rrdx_agg.py`). This
  covers the identity classes 106/80/23/8/6 and 104/78/26/8/7, the fork and siphon min/median/max, the 222 and 223
  counts, and the medians.
  - The rrdx records themselves are the swap agent's outputs (version 2 on all 223; the 13 were computed after the
    swap). I aggregated them; I did not re-measure them.
- **w_018b_left mr_002 diagnosis**: 121 cells, 7.85 mm², s 92–103, 1.6–3.9 mm against 13.5–16 mm.
- **Route13**:
  - 0.66–4.45 mm (notes table).
  - prio run on 7.
  - Clearance ≥ 0.71, joins ≤ 59.3, RMS 14.8–19.9, tip ≤ 38, shift 2.8–13.9 mm RMS.
  - RCCA lumen 0.959–1.886.
- **SOFA**: 36/36; 10/25 (8/23) against 24/25.
- **Entry geometry**:
  - Clearance 0.95–1.69 mm on the 23 tested.
  - Stub 3–10 mm, median 5.
  - Angle 35–127° (over all 223).
  - v4: 88°, 2.2–2.3 mm.
- **va_entry_geom**: 213/223, 3.3 mm, 0.046; 8.6 / 23.8 mm on 211; 9.3 mm on 219.
- **Lumen**: 3 VAs below 0.65 mm; medians 0.834 / 0.978.
- **Loader**: 5.87–6.00 mm, 74.
- **Split figures**: 153/16, 127/16, 174/16, 172/17.
- **Licences and attributions**: 124/92/7/216.
- **Pilot and scale-up numbers** found in the notes: 205–294 s; 0.065/0.000 → 0.72/0.62; 122.75 → 241.75;
  12/31 against 8/37; 2.9 mm; 14 rule-6 sides; Song 138 0.0154/0.0065.
- **Cited paths**: all 47 cited `_records/…` paths exist. The 868 logged record copies hash-equal their log and
  their scratch sources.
- **Decisions** all match the build log:
  - 03:50: items 1, 2, 3 and 6.
  - 14:00: stop at 27.
  - 14:25: reserves.
  - 16:05: primaries for every patient.
  - 16:02: handle rule.
  - 22:50: move the VA first.
  - 2026-10-03 14:05: no junction fix, framed as diversity.
  - 2026-10-03 14:02: the v5.1 follow-up.
- **Nits** (wording, not substance):
  - a. Gates table "OBJ vertices to the union, p99 ≤ 0.175": the maximum is 0.1753.
  - b. "the case_w_033_left forks at 2.36–2.39": the exact ratios are mr_007 2.30, mr_010 2.36, mr_011_L 2.38,
    mr_024 2.39 and mr_016 1.12. Say 2.30–2.39, or name the three.
  - c. Route13 "take-off ≤ 118.4°" is true of the 14 moved VAs. The unmoved, pinned RVA donor of
    w_012_left mr_021_L records 127.5°. This is the known record artefact the Assignment bullet already names.
    Say "on every moved VA".
  - d. "The finalize workflow's own verifier runs after this record is written": it has now run, so point to this note.
  - e. As already disclosed: BUILD_v5.json `donors.table` still has the assignment-time counts, and the
    `checks_final.sofa.note` still says the question is pending.

### (3) Toolchain copy and MANIFEST

- **Hash chain**:
  - 292/292 copies equal the FILES.tsv sha256.
  - 265/265 scratch originals equal their recorded sha256, so nothing was edited after the copy.
  - Stripping the header line gives back the original on 292/292.
  - The 27 donor records equal va_sources/v5_reg/donors_v5.
- **Coverage**:
  - Import closure is complete (`closure.py`): every local module that a copied .py imports is copied.
  - The only uncopied `/sp/v5_va/*.sh` references are "Run:" usage comments.
  - The key drivers named in CAROTID's Reproduce section are all present.
- **MANIFEST counts**: stage counts 11/161/59/17/44. The scratchpad path appears in 42 shell scripts, the repo path
  in 48 files (the same list as MANIFEST's) and `/sp` in 192.
- **GAP (claim 3 does not fully hold)**: `graft/graft_v5_reg_sep.py` is version **1.1**.
  - It was edited 2026-10-03 00:02–00:06 to add S6 `--objective net`, S7 `--pin-pose` and gain-candidate voxels.
    Only moved_pin (w_012_left mr_021_L) used 1.1.
  - The other **12 shipped anatomies record `v5_va.route13.tool = "graft_v5_reg_sep 1.0 (2026-10-02 …S1-S5…)"`**.
    That version survives only as scratch backups `route13/graft_v5_reg_sep.v1.py` (23:19) and `.v2.py` (23:49,
    VERSION 1.0). Neither is in carotid_tools/v5 or FILES.tsv.
  - By the diff, 1.1's default (`--objective lost`, no `--pin-pose`) ranks candidates by the same key (lost, other
    breaks, order) over the same is4 voxels. So it should choose the same poses, but it is not byte-identical and
    stamps 1.1.
  - The moved_ext provenance files were written 23:35–00:02. v2 (23:49) is the latest surviving 1.0 state, so it
    is not certain to be byte-for-byte the code that wrote all 12.
  - MANIFEST's row ("(v5_route13, 2026-10-02)") and "the copies are what the build ran" do not mention this.
  - **Fix**: copy `v5_va/route13/graft_v5_reg_sep.v2.py` (and optionally `.v1.py`) into
    `carotid_tools/v5/graft/route13/`, with FILES.tsv rows and a MANIFEST note. The note should say: 12 moves made
    with 1.0 (v2 backup, the last 1.0 state), moved_pin with 1.1; 1.1's defaults reproduce 1.0's objective.

### (4) Images

- 112 PNGs cover exactly the 223 names, once each: 111 pairs at 2210×1235 and 1 single panel at 1105×1235.
- All were written 15:29–15:48, after the swap, and no anatomy file has changed since.
- **Random 8** (`awk srand(20261004)`, `img8.txt`), plus 2 kept-v4 controls (`img_pix.py`):
  - Method: I rebuilt each figure only for its camera. I projected **my own** centreline reading and provenance
    `lo_mm`/`hi_mm` onto the **saved** PNG and sampled the pixels there.
  - Donor stretch: dark teal 0.975–1.000 of samples.
  - Host subclavian stretch: grey 1.000.
  - TopBrain piece: brown 1.000.
  - Kept-v4 sides: green 0.995–1.000.
- The title blocks (`img_titles.png`) equal provenance on every sampled panel: donor id, side, mirror, source,
  length (hi − lo), route13 moved_ext, tube windows (1.1 and 2.0 mm), and "KEPT FROM v4 (at bake / at assignment)".

### (5) v4 and tracked code untouched

- `git status` is the session-start snapshot: HEAD 29c8c5d and the same 3 modified files. Those are docs; "user
  copy.md" is the user's own notes, edited 13:54, and no agent mentions it.
- anatomies_v4: all 4,684 tracked blobs equal HEAD (my own `git hash-object`). The 223 ignored collision_full.vtp
  have none newer than 2026-10-01, and v2/v3 have 0 newer files.

### Interruptions (the user's concern)

- No partial files (`*.part`, `*.tmp`) in anatomies_v5, iso_v5, the bake markers, carotid_tools/v5 or the images.
- Every provenance, mesh_v5 and centreline JSON parses.
- The swap is complete on 13/13. BUILD_v5.json's hash equals the swap output.
- _records: 868/868. Toolchain: 292/292. Images: 223/223.
- No container is running. The leftover `claim/` folders are markers of finished jobs.
- Pre-existing debris from other agents, which the user may delete: `D:\Arjun\neve\grep.exe.stackdump` (2026-10-03
  00:49, in the session-start git status) and `carotid_data\va_sources\v5_reg\grep.exe.stackdump` (00:56).
- **My own slip**: a Git Bash grep loop I ran over carotid_tools/v5 crashed grep repeatedly, and Git Bash wrote
  `carotid_tools/v5/grep.exe.stackdump` (16:41–16:47). I stopped the task and removed that file, which I had
  created; the folder is back to 295 files.
