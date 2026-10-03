# carotid_tools/v5: the v5 toolchain, copied from the build session

These are the scripts that built `carotid_data/anatomies_v5` (see `CAROTID_V5.md` at the repo root). They ran from a
session scratchpad, which is temporary, so they are copied here. Only what the final build used is copied, not the
experiments around it. The copies are not edited: each code file gained one header line naming its source, and
nothing else changed.

- Source: `C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\`
  (`/sp/v5_va/` inside the build containers). Three dependencies sat beside it in the scratchpad root:
  `va_pool/rzip.py`, `v4_build/audit/topo_audit.py` and `v4_build/audit/sim_loader.json`.
- Copied 2026-10-03 by the `v5_records` agent of the finalize workflow (wf_fc195e37-2a3).
- 293 files: arterial 11, extract 161, graft 60, bake 17, checks 44 (one added after the final verifier, below).
- Not copied: the experiments and look-only scripts, the verifiers' own code (their notes are in
  `carotid_data/anatomies_v5/_records/notes/`), the per-side donor records and surfaces (51 MB, in
  `carotid_data/va_sources/v5_reg/donors_v5/`), and the workflow scripts (`*.js`).

## Files in this folder

| file | what |
|---|---|
| `MANIFEST.md` | this file |
| `FILES.tsv` | one row per copied file: stage, original path (relative to `v5_va/`), copy path, whether a header line was added, sha256 of the original and of the copy, size |
| `restore_layout.sh` | rebuilds the build-time layout: `bash restore_layout.sh <dir>/v5_va` writes every file back to its original relative path with the header line removed, and checks each against the original sha256 |
| `arterial/` | Arterial runner and the Docker images |
| `extract/` | fetch, TotalSegmentator, trace, QC gate, CT-edge walls, the two pipeline fixes, tally, eligibility; VMR 0248 |
| `graft/` | donor records, registry graft, dry run, assignment, the route13 move; `donor_records/` |
| `bake/` | the v5 bake (handle rule) and the full-bake drivers |
| `checks/` | split checker, measure libraries, post-bake checks, exact-ownership measure, swap and final record, v4 audit |

Inside each stage folder a file keeps its path relative to `v5_va/`: `checks/checks/chk_one.py` was
`v5_va/checks/chk_one.py`, and `checks/swap13/rrdx.py` was `v5_va/swap13/rrdx.py`. The scripts call each other by
those paths (`/sp/v5_va/<path>`), so they do not run from the stage folders directly; restore the layout first.

**What the copies differ in.** Every `.py`, `.sh` and Dockerfile has one extra line, right after the shebang where
there is one:

    # [carotid_tools/v5 copy] source: <scratchpad>\v5_va\<path> (= /sp/v5_va/<path> in the build containers)

Lists, JSON and pip pins are byte copies. One side effect: `run_arterial.sh` prints its usage with
`sed -n '2,34p' "$0"`, so in the copy the usage starts with the header line. `restore_layout.sh` removes the
line, and every restored file then matches the sha256 of the file the build ran (tested on 2026-10-03: 265 of
265 restored files, the 27 donor records being reference copies only; 266 of 266 after the addition below).

## The route13 tool version

`graft/graft_v5_reg_sep.py` is version **1.1** (edited 2026-10-03 00:02–00:06 to add `--objective net` and
`--pin-pose`). Only the moved_pin anatomy (case_w_012_left__topcow_mr_021_L) was made with 1.1. The other 12 moved
anatomies record `graft_v5_reg_sep 1.0` in `provenance.json` → `v5_va.route13.tool`; they were planned 2026-10-02
23:23–23:39. The final verifier found that 1.0 was missing from this copy, so it was added afterwards as
`graft/route13/graft_v5_reg_sep.v2.py`, the 1.0 backup taken at 23:49.

Nothing fingerprinted the bytes that ran, so the backup was checked by re-running it, dry, on all 13 anatomies with
the pre-move `anatomies_v5` folders (`backup_pre_route13`) mounted in place: all 13 decision records are identical to
the ones the move wrote, field for field, except the run time (`carotid_data/anatomies_v5/_records/checks/
route13_tool_v1_0/cmp.txt`). The earlier backup `route13/graft_v5_reg_sep.v1.py` (23:19, before the S2 rewrite) was
not what ran and is not copied. To re-run 1.0, run the restored `route13/graft_v5_reg_sep.v2.py` in place of
`graft_v5_reg_sep.py` (same options, see `route13/graft13.sh`).

## Stages and the runs that used them

Runs are the build workflows named in the build log (`_records/notes/V5_BUILD_LOG.md`):

- **infra**: Arterial set-up (feasibility study, 2026-10-01)
- **study**: feasibility study / TaG-Net calibration (2026-10-01), reused as a library
- **pilot**: thin-slice pilot (wf_54a8a0f4-215, finished by wf_a476693a-699; 2026-10-01/02)
- **scaleup**: donor scale-up (wf_05522008-519; Song lane finished by wf_f370c47e-829; 2026-10-02)
- **graft_tool**: graft tool (wf_6b2e5344-f6d, 2026-10-02 01:16; version 1.0)
- **assign**: assignment (wf_5f2b8dd6-9cc, 2026-10-02 14:27; version 1.1 = the build)
- **bake**: bake (wf_549f9dbd-cfc, 2026-10-02 16:08: probe, full, check, verifiers)
- **route13**: route13 move (wf_f52a4310-8ea, 2026-10-02 22:55)
- **final**: finalize (wf_fc195e37-2a3, 2026-10-03: swap13 and the final record)
- **v4**: v4 build audit (2026-09-25/27)
- **main**: main session check (2026-10-03 14:05)

| stage | group | files | used by |
|---|---|---:|---|
| arterial | images | 7 | infra; image of every Arterial / TotalSegmentator run in pilot and scale-up |
| arterial | Arterial runner | 3 | infra; pilot; scaleup |
| arterial | Arterial input arguments | 1 | infra; imported by the Song args scripts (pilot, scaleup) |
| extract | fetch, 172-series | 12 | pilot (fetch); scaleup (172 lane) |
| extract | fetch, Song | 13 | pilot (fetch); scaleup (Song lane) |
| extract | QC gate (recalibrated) | 2 | pilot; scaleup (run unchanged by pipeline v2) |
| extract | wall measures | 6 | pilot; scaleup (run unchanged by pipeline v2) |
| extract | pilot processing | 39 | pilot (process:ia172 / process:song) |
| extract | trace tools | 3 | study (art_extract); pilot; scaleup |
| extract | pilot decision | 4 | pilot (decide) |
| extract | pipeline v2 post stage | 12 | scaleup (fixes agent wrote it; both lanes ran it) |
| extract | pilot cases under pipeline v2 | 6 | scaleup (fixes agent) |
| extract | Song lane | 18 | scaleup (lane:song, song-lane-finish) |
| extract | 172-series lane | 14 | scaleup (lane:ia172) |
| extract | tally, eligible list, by-eye checks | 7 | scaleup (tally); assign (eye:172, eligible list) |
| extract | VMR 0248 donor and source loaders | 11 | study (VMR 0248 wall and metrics); vmr_vmtk.py is also the VMTK step of every thin-slice wall |
| extract | study modules used as libraries | 14 | study; imported by the pilot / scale-up measures (tag_totalseg_run.sh = the TotalSegmentator runner) |
| graft | prototype graft | 1 | study; imported by graft_v5_reg and bake_full/collect |
| graft | registry graft and assignment | 5 | graft_tool (1.0); assign (1.1, the build) |
| graft | assignment checks and reports | 7 | assign |
| graft | route13 move | 20 | route13 (graft_v5_reg_sep 1.0 made 12 moves, 1.1 the moved_pin one) |
| graft | donor unit records | 27 | assign (graft_v5_reg_donor 1.1) |
| bake | bake wrapper (base) | 1 | study; base of bake_v5b |
| bake | bake (handle rule) | 1 | bake (probe, full); route13 |
| bake | route-priority fallback | 1 | route13 (run on 7 of the 13, never chosen) |
| bake | full-bake drivers and records | 14 | bake (full); collect.py also final |
| checks | split checker | 1 | graft_tool; assign; bake (check); final |
| checks | measure libraries | 4 | study; used by the pilot measures, bake_full/measure_full_one.py and checks |
| checks | post-bake checks | 16 | bake (check stage); re-used by final |
| checks | exact-ownership measure | 2 | bake (verify:wall); final (rrdx, record) |
| checks | swap and final record | 17 | final |
| checks | v4 audit | 2 | v4; imported / read by the checks and final |
| checks | VA-entry geometry check | 2 | main |

Notes on what each group is:

- **arterial / images**: neve-arterial (py3.11, CPU) and neve-arterial:py310 (CPU; the base of the GPU image), neve-arterial-gpu (py3.10 + torch cu124; every Arterial run used it, `ARTERIAL_GPU=1` is the default), neve-totalseg-gpu (TotalSegmentator 2.18.0 `--ml`, on top of neve-arterial-gpu). Pins in `constraints.txt` and the pip freezes. Arterial itself is FLOWCAT-CV/arterial @3ff58ab with the Zenodo 22694951 weights, installed under `carotid_data/va_sources/arterial/` (not copied; weights CC BY-NC 4.0, code PolyForm NC).
- **extract / fetch**: range-reads the cases out of the Zenodo 15697196 ZIPs (172-series) and the figshare 26965450 RAR5 archives (Song), converts DICOM to NIfTI, writes meta.json, quick-look MIPs and the arch checks.
- **extract / pilot processing**: per case: TotalSegmentator split, Arterial arguments and input windowing (172-series clip to [-1024, 3071]; Song `ct_negzero_air1000` re-window), the two traces (tsba = `pilot_art_extract.py`, ts = `art_extract.py`), VMTK, the QC gate, the ct_seg4 CT-edge wall, landmarks, measure, edge, protrusion, figures, collect and grade, and the closing passes (`*_final*.sh`). The pilot cases that stayed in the build without a v2 re-run keep these walls.
- **extract / pipeline v2**: the post stage every scale-up case ran (`scaleup_v2_post.sh <src> post2 <case>`, then `scaleup_ctseg_fallback.sh`): both traces through the unchanged QC gate, trace selection, fix 1 (V1 mask-break rescue), gross shape, fix 2 (ct_seg4 C2-exit pinch fix, `scaleup_ct_seg.py`), the unchanged measures, protrusion, figures, collect, grade. `scaleup_fixes_*` re-ran the pilot cases the fixes touched (walls `ct_seg4v2`).
- **extract / tally**: `scaleup_tally.py` (eligibility, coarseness, stop rule; `results/scaleup_tally.json`) and `v5_eligible_final.py` (`results/v5_eligible_final.json`: 25 primary + 2 reserve pairs, with the wall / QC / trace file of every side). By-eye verdicts go to `results/scaleup_eye_rejections.json` through the `*_eye_add.py` scripts.
- **extract / study modules**: the pilot measures import the TaG-Net calibration and CT-wall study modules (`tag_*`, `verify_tag_ctwall_image.py`) and the VMR modules (`vmr_*`); they are libraries here, run unchanged.
- **graft**: `graft_v5_reg_donor.py` builds the donor records, `graft_v5_reg.py` places each donor (pose search, every gate, compose, write), `graft_v5_reg_dryall.sh` dry-runs every unit against all 223 anatomies, `graft_v5_assign.py` solves the one-to-one assignment (exact lexicographic MILP, `--score v5`). `graft_v5_reg_sep.py` and `route13/` re-placed the VA of the 13 route anatomies. `donor_records/` holds the 27 unit records (`record.json`) as built; the per-side records and surfaces stay in `va_sources/v5_reg/donors_v5/`.
- **bake**: `bake_v5b.py` baked all 223 (the handle rule, stated diffs D1-D8 in its docstring; `topbrain_tools/bake_meshes_v4.py` is used unchanged). `bake_v5c.py` is the route-priority fallback: it was run on 7 of the 13 route anatomies and never chosen, and is kept because `route13/run_bake.py` imports it. `bake_full/` is the full-bake queue (10 worker containers), the per-anatomy bake and measure, the collection into `results/v5_bake_full.json` and `BUILD_v5.json`, the keep-v4-VA re-assembly (2 sides), the OBJ re-decimation (3 anatomies) and the tube-handle diagnostics behind the keep-v4 decision.
- **checks**: `checks/` is the post-bake check stage (strict audit with v4's `topo_audit.py`, right-route and VA detail, loader, SOFA into the RVA, VA lumen, TopBrain geometry, entry geometry, anatomies_v4 untouched). `vvw/` is the wall verifier's exact-ownership implementation, which `swap13/rrdx.py` and the final record use. `swap13/` swapped the 13 route-fixed anatomies in, re-checked them and wrote the final `BUILD_v5.json` and `results/v5_final_summary.json` (`final_collect.py`).

## Re-running

1. **Restore the layout.** `bash carotid_tools/v5/restore_layout.sh <X>/v5_va`. This also writes
   `<X>/va_pool/rzip.py` and `<X>/v4_build/audit/{topo_audit.py,sim_loader.json}`. Copy the results the scripts
   read into `<X>/v5_va/results/`: the final versions are in `carotid_data/anatomies_v5/_records/results/`.
   `va_standard.py` reads `results/standard.json` next to itself.
2. **Mounts.** Every container mounts the repo at `/repo` (read-only for the checks) and `<X>` at `/sp`. The bake and check containers also
   mount `<repo>/carotid_data` at `/opt/eve_training/carotid_data` and `<repo>/topbrain_tools` at
   `/opt/eve_training/topbrain_tools`, as for v4 (`checks/checks/dr.sh`, `bake/bake_full/launch.sh` show the full
   lists). The loader and SOFA scripts put `/repo/eve` and `/repo/eve_bench` first on `sys.path`.
3. **Images.** neve-arterial-gpu (Arterial), neve-totalseg-gpu (TotalSegmentator), neve-build-stagea (QC, walls,
   measures; PYTHONPATH `/sp/va_pool/pylib` is a pip-installed folder with pydicom 2.4.4 and pylibjpeg 2.0.1
   for the 172-series DICOMs, not copied), neve-build-vmtk (VMTK), eve-training-fixed (graft, assignment, tally, loader,
   SOFA), neve-build-meshlab (bake, measures, checks), python:3 (by-eye records). CAROTID_V4.md's Reproduce
   section describes neve-build-stagea, neve-build-vmtk, eve-training-fixed and neve-build-meshlab.
4. **Repo modules imported** (not copied, they are tracked): `topbrain_tools/` bake_meshes_v4, sdf_union_v4,
   sdf_union, sdf_mesher, mask_to_surface, graft_siphon; `carotid_tools/` graft_three, graft_v4; `eve`, `eve_bench`.
5. **Data not in the repo**: the thin-slice cases and everything derived from them are in
   `carotid_data/va_sources/` (gitignored): `thin_slice/`, `derived/thin_slice/`, `derived/vmr0248/`, `v5_reg/`,
   `arterial/` (weights). The 172-series selection reads `va_pool/data/ia172/Metadata.csv` from Zenodo 15697196.

**Hard-coded paths a re-run must change.**

- **The host scratchpad path** `C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad`
  (also as `/c/Users/akrish41/...` and `C:/Users/...`) is written into 42 host-side shell scripts, mostly as the
  `-v <scratchpad>:/sp` mount and the lane lock / log folders. Replace it with `<X>`:

  - `arterial/run_arterial.sh` (3 lines)
  - `extract/pilot_fetch_song_qc_all.sh` (2 lines)
  - `extract/pilot_ia172_ts.sh` (1 line)
  - `extract/pilot_proc_ia172_final.sh` (2 lines)
  - `extract/pilot_proc_ia172_final2.sh` (1 line)
  - `extract/pilot_proc_ia172_rescue.sh` (1 line)
  - `extract/pilot_proc_ia172_run2.sh` (2 lines)
  - `extract/pilot_proc_ia172_tsrun2.sh` (1 line)
  - `extract/pilot_proc_song_final.sh` (2 lines)
  - `extract/pilot_proc_song_run5.sh` (2 lines)
  - `extract/pilot_proc_song_tsrun.sh` (1 line)
  - `extract/scaleup_ctseg_fallback.sh` (2 lines)
  - `extract/scaleup_fixes_pilot_v2.sh` (1 line)
  - `extract/scaleup_fixes_wall2.sh` (1 line)
  - `extract/scaleup_fixes_wall2_finish.sh` (1 line)
  - `extract/scaleup_ia172_chain.sh` (1 line)
  - `extract/scaleup_ia172_grade.sh` (1 line)
  - `extract/scaleup_ia172_run.sh` (2 lines)
  - `extract/scaleup_ia172_ts.sh` (1 line)
  - `extract/scaleup_ia172_tsrun.sh` (1 line)
  - `extract/scaleup_ia172_v2patch.sh` (1 line)
  - `extract/scaleup_ia172_visual.sh` (1 line)
  - `extract/scaleup_song_artredo.sh` (2 lines)
  - `extract/scaleup_song_chain.sh` (1 line)
  - `extract/scaleup_song_grade.sh` (1 line)
  - `extract/scaleup_song_grade4.sh` (1 line)
  - `extract/scaleup_song_gradechain.sh` (1 line)
  - `extract/scaleup_song_postchain.sh` (1 line)
  - `extract/scaleup_song_refqc.sh` (1 line)
  - `extract/scaleup_song_run.sh` (2 lines)
  - `extract/scaleup_song_tsrun.sh` (1 line)
  - `extract/scaleup_song_v2patch.sh` (1 line)
  - `extract/scaleup_v2_post.sh` (1 line)
  - `extract/tag_totalseg_run.sh` (1 line)
  - `extract/vmr_run.sh` (1 line)
  - `graft/route13/dr13.sh` (1 line)
  - `bake/bake_full/launch.sh` (1 line)
  - `checks/checks/dr.sh` (1 line)
  - `checks/checks/item5.sh` (1 line)
  - `checks/checks/launch.sh` (1 line)
  - `checks/checks/status.sh` (1 line)
  - `checks/swap13/swap.sh` (1 line)

- **The host repo path** `D:\Arjun\neve` (or `/d/Arjun/neve`) appears in 48 files, as the `-v <repo>:/repo`
  mount and as host-side data paths. On this machine it is unchanged; elsewhere replace it. Files: `arterial/docker_totalseg/Dockerfile`, `arterial/run_arterial.sh`, `extract/pilot_fetch_song_qc_all.sh`, `extract/pilot_fetch_song_sidecars.sh`, `extract/pilot_fetch_song_va_check_all.sh`, `extract/pilot_ia172_summary.py`, `extract/pilot_ia172_ts.sh`, `extract/pilot_proc_ia172_final.sh`, `extract/pilot_proc_ia172_final2.sh`, `extract/pilot_proc_ia172_rescue.sh`, `extract/pilot_proc_ia172_run2.sh`, `extract/pilot_proc_ia172_tsrun2.sh`, `extract/pilot_proc_ia172_watchdog.sh`, `extract/pilot_proc_song_final.sh`, `extract/pilot_proc_song_run5.sh`, `extract/pilot_proc_song_tsrun.sh`, `extract/pilot_song_rar.py`, `extract/scaleup_ctseg_fallback.sh`, `extract/scaleup_fixes_pilot_v2.sh`, `extract/scaleup_fixes_wall2.sh`, `extract/scaleup_fixes_wall2_finish.sh`, `extract/scaleup_ia172_chain.sh`, `extract/scaleup_ia172_grade.sh`, `extract/scaleup_ia172_run.sh`, `extract/scaleup_ia172_ts.sh`, `extract/scaleup_ia172_tsrun.sh`, `extract/scaleup_ia172_visual.sh`, `extract/scaleup_song_artredo.sh`, `extract/scaleup_song_chain.sh`, `extract/scaleup_song_grade.sh`, `extract/scaleup_song_grade4.sh`, `extract/scaleup_song_gradechain.sh`, `extract/scaleup_song_postchain.sh`, `extract/scaleup_song_refqc.sh`, `extract/scaleup_song_run.sh`, `extract/scaleup_song_tsrun.sh`, `extract/scaleup_v2_post.sh`, `extract/sources.py`, `extract/tag_totalseg_run.sh`, `extract/v5_eligible_final.py`, `extract/vmr_run.sh`, `graft/graft_v5_reg_dryall.sh`, `graft/route13/dr13.sh`, `bake/bake_full/launch.sh`, `bake/bake_full/status.sh`, `checks/checks/dr.sh`, `checks/checks/item5.sh`, `checks/swap13/swap.sh`.
- **Container paths** `/sp/v5_va/...` (192 files) and `/repo/...` need no change when the mounts above are kept.
- **Registry folders** are defaults in the code: `graft_v5_reg_donor.py` OUT_ROOT `v5_reg/donors` (the build passed
  `--out-root .../donors_v5`), `graft_v5_assign.py` ROOT `v5_reg` (`--donor-root`, `--dry` given), and the route13
  and bake_full scripts write under `carotid_data/va_sources/v5_reg/`.
- **Docker** is called as `/c/Program Files/Docker/Docker/resources/bin` on PATH with `MSYS_NO_PATHCONV=1` (Git Bash).

**The build order, as it ran** (details and exact commands in each run's notes under `_records/notes/`):

1. Pilot and scale-up lanes per case: `scaleup_song_chain.sh <fetch|ts|art|trace|grade> <case...>` and
   `scaleup_ia172_chain.sh <fetch|ts|tscpu|gpu|post|grade> <case...>` (Arterial through `run_arterial.sh`), then
   `scaleup_v2_post.sh <src> post2 <case...>` and `scaleup_ctseg_fallback.sh <src> <case...>`.
2. `scaleup_tally.py` (eve-training-fixed), by-eye verdicts, `v5_eligible_final.py`.
3. `graft_v5_reg_donor.py --eligible /sp/v5_va/results/v5_eligible_final.json --out-root /repo/carotid_data/va_sources/v5_reg/donors_v5`;
   `DONORS=donors_v5 DRY=dry_v5 bash graft_v5_reg_dryall.sh`; `graft_v5_assign.py --tag v5 --solver strict --score v5`
   with `--donor-root` / `--dry` on the v5 folders and the two reserves as `--reserve-units`; `graft_v5_reg.py --plan
   /repo/carotid_data/va_sources/v5_reg/plan_v5.json --out /repo/carotid_data/anatomies_v5`; `v5_split.py
   /repo/carotid_data/anatomies_v5 --baseline results/split_v4.json`.
4. `bake_full/plan.py`, `bash bake_full/launch.sh 10` (each worker: `bake_full_one.py` then `measure_full_one.py`),
   `keep_v4_va.py` / `obj_fix.py` where flagged, `bake_full/collect.py --build`.
5. `bash checks/launch.sh 8`, `checks/select_sofa.py`, the SOFA worker, `checks/collect.py`.
6. route13: `route13/diag13.py`, `route13/graft13.sh <jobs>`, `VARIANT=moved_ext OWNERSHIP=standard bash
   route13/worker13.sh <list>`, `route13/collect13.py`.
7. `bash swap13/swap.sh backup|swap|verify`, `bash swap13/run_a.sh 5` (neve-build-meshlab) and
   `swap13/sofa_worker.sh` (eve-training-fixed), `v5_split.py` again, `swap13/final_collect.py`; the host copies
   `swap13/build/BUILD_v5.json` into `carotid_data/anatomies_v5/`.

## Every file

Paths: `S` = holds the host scratchpad path, `R` = holds the host repo path, `/sp` = uses the scratch mount,
`/repo` = uses the repo mount, `D` = starts containers itself. "What" is the first line of the file's own
docstring or header comment (several scale-up copies still carry their pilot original's first line).

### arterial/

| copy | original (under v5_va/) | group | what | paths |
|---|---|---|---|---|
| `arterial_cpu_run.py` | `arterial_cpu_run.py` | Arterial runner | CPU runner for the Arterial pipeline (neve-arterial image, conda base env: py3.10 + vmtk 1.5.0). |  |
| `arterial_ts_args.py` | `arterial_ts_args.py` | Arterial input arguments | Suggest run_arterial.sh arguments for a TotalSegmentator v3 subject so that the CT matches the | /repo |
| `arterial_va_summary.py` | `arterial_va_summary.py` | Arterial runner | Summarise Arterial's labelled centerline graph for the vertebral arteries. |  |
| `docker_arterial/Dockerfile` | `docker_arterial/Dockerfile` | images | neve-arterial (tag neve-arterial:latest, image 57bc8d21fc6b, built 2026-10-01): CPU-only runtime for the | /sp |
| `docker_arterial/Dockerfile.py310` | `docker_arterial/Dockerfile.py310` | images | neve-arterial:py310 (image 66003f1ef503): CPU-only runtime for the Arterial head-and-neck CTA pipeline | /sp |
| `docker_arterial/constraints.txt` | `docker_arterial/constraints.txt` | images |  |  |
| `docker_arterial/pip_freeze_py310.txt` | `docker_arterial/pip_freeze_py310.txt` | images |  |  |
| `docker_arterial/pip_freeze_py311.txt` | `docker_arterial/pip_freeze_py311.txt` | images |  |  |
| `docker_arterial_gpu/Dockerfile` | `docker_arterial_gpu/Dockerfile` | images | neve-arterial-gpu: neve-arterial:py310 (CPU build, conda base env /opt/conda, py3.10 + vmtk) with CUDA | D |
| `docker_totalseg/Dockerfile` | `docker_totalseg/Dockerfile` | images | neve-totalseg-gpu: TotalSegmentator 2.18.0 on top of neve-arterial-gpu | R |
| `run_arterial.sh` | `run_arterial.sh` | Arterial runner | run_arterial.sh <input_ct.nii.gz> <outdir> [options passed to arterial_cpu_run.py] | S R /sp /repo D |

### extract/

| copy | original (under v5_va/) | group | what | paths |
|---|---|---|---|---|
| `art_extract.py` | `art_extract.py` | trace tools | art_extract.py -- vertebral arteries (VA) from one Arterial run. | /sp |
| `art_vmtk_one.py` | `art_vmtk_one.py` | trace tools | art_vmtk_one.py <surface.vtp> <seeds.json> <trace_points.json/-> <out.json> |  |
| `pilot_art_extract.py` | `pilot_art_extract.py` | trace tools | pilot_art_extract.py = art_extract.py (the study tool, copied UNCHANGED) + ONE change, the basilar seed fix | /sp |
| `pilot_decide_collect.py` | `pilot_decide_collect.py` | pilot decision | pilot_decide_collect.py -- the pilot decision's per-side wall table, per-source yield and the DONOR RANKING. | /sp /repo |
| `pilot_decide_final.py` | `pilot_decide_final.py` | pilot decision | pilot_decide_final.py -- assemble results/pilot_decide.json from pilot_decide_need_rows.json, | /sp |
| `pilot_decide_need.py` | `pilot_decide_need.py` | pilot decision | pilot_decide step 1: exact donor need from anatomies_v4 provenance (read-only). | /sp /repo |
| `pilot_decide_next.py` | `pilot_decide_next.py` | pilot decision | pilot_decide_next.py -- scale-up: pair yield per processed case with honest intervals, cases still needed for | /sp |
| `pilot_fetch_song_get.py` | `pilot_fetch_song_get.py` | fetch, Song | pilot_fetch_song_get.py: range-read one Song et al. 2024 CTA member out of its RAR5 archive (stored, method 0, | /sp /repo |
| `pilot_fetch_song_meta.py` | `pilot_fetch_song_meta.py` | fetch, Song | pilot_fetch_song_meta.py (pass 2): assemble <img>/<case>/meta.json for every fetched Song case from | /sp /repo |
| `pilot_fetch_song_meta2.py` | `pilot_fetch_song_meta2.py` | fetch, Song | pilot_fetch_song_meta2.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:45 EDT): instance 1's | /sp /repo |
| `pilot_fetch_song_picks.py` | `pilot_fetch_song_picks.py` | fetch, Song | pilot_fetch_song_picks.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:36 EDT): CCA and subclavian (SCA) picks | /sp /repo |
| `pilot_fetch_song_qc.py` | `pilot_fetch_song_qc.py` | fetch, Song | pilot_fetch_song_qc.py (pass 1): per Song case, read the shipped ct.nii.gz and write | /repo |
| `pilot_fetch_song_qc_all.sh` | `pilot_fetch_song_qc_all.sh` | fetch, Song | pilot_fetch_song_qc_all.sh: run pilot_fetch_song_qc.py on each case, ONE container at a time, once its ct.nii.gz | S R /sp D |
| `pilot_fetch_song_sidecars.sh` | `pilot_fetch_song_sidecars.sh` | fetch, Song | pilot_fetch_song_sidecars.sh (pilot_fetch_song): fetch the dcm2niix BIDS sidecar (cta_images_head_<id>.json, RAR5 | R |
| `pilot_fetch_song_table.py` | `pilot_fetch_song_table.py` | fetch, Song | pilot_fetch_song_table.py: one row per Song case = archive index (header geometry) + clinical_all.csv (sex, age, | /sp /repo |
| `pilot_fetch_song_va_check.py` | `pilot_fetch_song_va_check.py` | fetch, Song | pilot_fetch_song_va_check.py (pilot_fetch_song, resume pass, 2026-10-01 ~20:15 EDT). | /repo |
| `pilot_fetch_song_va_check2.py` | `pilot_fetch_song_va_check2.py` | fetch, Song | pilot_fetch_song_va_check2.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:27 EDT). | /sp /repo |
| `pilot_fetch_song_va_check_all.sh` | `pilot_fetch_song_va_check_all.sh` | fetch, Song | pilot_fetch_song_va_check_all.sh (pilot_fetch_song, instance 3, 2026-10-01 ~20:25 EDT) | R /sp /repo D |
| `pilot_fetch_song_va_zoom.py` | `pilot_fetch_song_va_zoom.py` | fetch, Song | pilot_fetch_song_va_zoom.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:36 EDT): zoomed axial crops for the by-eye | /repo |
| `pilot_ia172_archcut.py` | `pilot_ia172_archcut.py` | fetch, 172-series | pilot_ia172_archcut: does the bottom slice cut the aortic arch into two limbs (ascending + descending)? | /sp /repo |
| `pilot_ia172_convert.py` | `pilot_ia172_convert.py` | fetch, 172-series | pilot_ia172_convert: one DICOM series -> ct.nii.gz (RAS, raw HU, int16), with geometry checks, plus a 1.5 mm RAS | /sp |
| `pilot_ia172_fetch.py` | `pilot_ia172_fetch.py` | fetch, 172-series | pilot_ia172_fetch: download whole series of the 172-series set by one byte-span read each, then convert. | /sp /repo |
| `pilot_ia172_hdr.py` | `pilot_ia172_hdr.py` | fetch, 172-series | pilot_ia172_hdr: full DICOM headers of every series of the 172-series set, read remotely, for selection and | /sp |
| `pilot_ia172_meta_visual.py` | `pilot_ia172_meta_visual.py` | fetch, 172-series | pilot_ia172_meta_visual: merge my visual reading (results/pilot_ia172_visual.json) into each case's meta.json and | /sp /repo |
| `pilot_ia172_qc.py` | `pilot_ia172_qc.py` | fetch, 172-series | pilot_ia172_qc: quick-look QC, neck MIP PNG and meta.json for fetched 172-series cases. | /sp /repo |
| `pilot_ia172_select.py` | `pilot_ia172_select.py` | fetch, 172-series | pilot_ia172_select: candidate table + duplicate screen for the 172-series set, from the remote headers | /sp |
| `pilot_ia172_slicecheck.py` | `pilot_ia172_slicecheck.py` | fetch, 172-series | pilot_ia172_slicecheck: per-slice sanity of the converted volumes. | /sp /repo |
| `pilot_ia172_summary.py` | `pilot_ia172_summary.py` | fetch, 172-series | pilot_ia172_summary: collect the fetch / convert / QC records of the 172-series pilot into | R /sp /repo |
| `pilot_ia172_ts.sh` | `pilot_ia172_ts.sh` | fetch, 172-series | pilot_ia172_ts.sh <case> [<case> ...] | S R /repo D |
| `pilot_ia172_zip.py` | `pilot_ia172_zip.py` | fetch, 172-series | pilot_ia172_zip: range-read helpers for the Head-CTA 172-series set (Zenodo 15697196, CC BY 4.0). | /sp /repo |
| `pilot_proc_ia172_args.py` | `pilot_proc_ia172_args.py` | pilot processing | pilot_proc_ia172_args.py <case> -- Arterial arguments for a 172-series pilot case (neve-build-stagea). | /sp /repo |
| `pilot_proc_ia172_art_extract.py` | `pilot_proc_ia172_art_extract.py` | pilot processing | pilot_proc_ia172_art_extract.py = art_extract.py (the study tool) + ONE change, the basilar-seed fix; a byte copy of the | /sp |
| `pilot_proc_ia172_clip.py` | `pilot_proc_ia172_clip.py` | pilot processing | pilot_proc_ia172_clip.py <case> -- the Arterial input of a 172-series pilot case: the native ct.nii.gz with HU clipped | /repo |
| `pilot_proc_ia172_collect.py` | `pilot_proc_ia172_collect.py` | pilot processing | pilot_proc_ia172_collect.py <case> [<case> ...] -- gather one 172-series pilot case's outputs into ONE record per | /sp /repo |
| `pilot_proc_ia172_ct_seg.py` | `pilot_proc_ia172_ct_seg.py` | pilot processing | pilot_proc_ia172_ct_seg.py = /sp/v5_va/tag_ct_seg.py (tag_ctwall's copy of vmr_ct_seg3.py, the study's ct_seg3 / | /sp |
| `pilot_proc_ia172_ctseg_common.py` | `pilot_proc_ia172_ctseg_common.py` | pilot processing | pilot_proc_ia172_ctseg_common.py -- loader adapter for the ct_seg4 CT-edge wall segmentation of a 172-series pilot | /sp /repo |
| `pilot_proc_ia172_edge.py` | `pilot_proc_ia172_edge.py` | wall measures | pilot_proc_ia172_edge.py = pilot_proc_aortaseg_edge.py (the AortaSeg pilot agent's script, read-only) for a 172-series | /sp /repo |
| `pilot_proc_ia172_extract.sh` | `pilot_proc_ia172_extract.sh` | pilot processing | pilot_proc_ia172_extract.sh <case> <mode>   (inside neve-build-stagea) -- the VA traces of one 172-series pilot case. | /sp /repo |
| `pilot_proc_ia172_fig.py` | `pilot_proc_ia172_fig.py` | pilot processing | pilot_proc_ia172_fig.py <case> -- figures for one 172-series pilot case (neve-build-stagea: nibabel + matplotlib). | /sp /repo |
| `pilot_proc_ia172_final.sh` | `pilot_proc_ia172_final.sh` | pilot processing | pilot_proc_ia172_final.sh -- the closing pass after the post loop (Git Bash on the host). Idempotent; log | S R /sp /repo D |
| `pilot_proc_ia172_final2.sh` | `pilot_proc_ia172_final2.sh` | pilot processing | pilot_proc_ia172_final2.sh -- after pilot_proc_ia172_final.sh (fourth instance, 2026-10-02 01:20). Idempotent; log | S R /sp D |
| `pilot_proc_ia172_grade.py` | `pilot_proc_ia172_grade.py` | pilot processing | pilot_proc_ia172_grade.py -- per case and side GRADE + donor-ranking table for the 172-series pilot (fourth instance of | /sp /repo |
| `pilot_proc_ia172_gross.py` | `pilot_proc_ia172_gross.py` | pilot processing | pilot_proc_ia172_gross.py -- gross shape of every traced VA side of one 172-series pilot case (PILOT_CASE), on the VMTK | /sp /repo |
| `pilot_proc_ia172_landmarks.py` | `pilot_proc_ia172_landmarks.py` | wall measures | pilot_proc_ia172_landmarks.py -- V1..V4 landmarks on the CT-segmented (ct_seg4) VA centrelines of one 172-series pilot | /sp /repo |
| `pilot_proc_ia172_measure.py` | `pilot_proc_ia172_measure.py` | wall measures | pilot_proc_ia172_measure.py -- wall detail, validity gates and gross shape of one 172-series pilot donor's | /sp /repo |
| `pilot_proc_ia172_protrusion.py` | `pilot_proc_ia172_protrusion.py` | pilot processing | pilot_proc_ia172_protrusion.py <case> [...] -- what the ct_seg4 label's outward relief is made of (neve-build-stagea). | /repo |
| `pilot_proc_ia172_qc2.py` | `pilot_proc_ia172_qc2.py` | QC gate (recalibrated) | pilot_proc_ia172_qc2.py <case> -- the RECALIBRATED Stage-2 QC gate (notes/tag_calibration.md) plus the brief's checks, | /sp /repo |
| `pilot_proc_ia172_rescue.sh` | `pilot_proc_ia172_rescue.sh` | pilot processing | pilot_proc_ia172_rescue.sh <case> [...] -- RESCUE retrace of a side whose trace failed because Arterial's labels were | S R /sp /repo D |
| `pilot_proc_ia172_run2.sh` | `pilot_proc_ia172_run2.sh` | pilot processing | pilot_proc_ia172_run2.sh <stage/group> [case ...] = pilot_proc_ia172_run.sh + ONE change (20:50): the light args / clip | S R /sp /repo D |
| `pilot_proc_ia172_summary.py` | `pilot_proc_ia172_summary.py` | pilot processing | pilot_proc_ia172_summary.py -- the pilot's tables from /sp/v5_va/results/pilot_proc_ia172.json (per case and side | /sp /repo |
| `pilot_proc_ia172_tsrun2.sh` | `pilot_proc_ia172_tsrun2.sh` | pilot processing | pilot_proc_ia172_tsrun2.sh = pilot_proc_ia172_tsrun_ml.sh (first instance; itself tag_totalseg_run.sh + 3 changes) + ONE fix: | S R /sp /repo D |
| `pilot_proc_ia172_tssplit.py` | `pilot_proc_ia172_tssplit.py` | pilot processing | pilot_proc_ia172_tssplit.py <case> -- split TotalSegmentator's multilabel output (--ml) into one uint8 | /repo |
| `pilot_proc_ia172_vmtk.sh` | `pilot_proc_ia172_vmtk.sh` | pilot processing | pilot_proc_ia172_vmtk.sh <extract_dir> [<extract_dir> ...]  (inside ONE neve-build-vmtk container) | /sp |
| `pilot_proc_ia172_watchdog.sh` | `pilot_proc_ia172_watchdog.sh` | pilot processing | pilot_proc_ia172_watchdog.sh -- stops an Arterial run of THIS agent (an ia172 case) whose centreline stage has written | R |
| `pilot_proc_song_args.py` | `pilot_proc_song_args.py` | pilot processing | pilot_proc_song_args.py <case> -- Arterial arguments for a Song 2024 pilot case (neve-build-stagea). | /sp /repo |
| `pilot_proc_song_clipcmp.py` | `pilot_proc_song_clipcmp.py` | pilot processing | pilot_proc_song_clipcmp.py <case> [<case> ...] -- does the Song 0-HU clipping move the lumen edge? (eve-training-fixed or | /sp /repo |
| `pilot_proc_song_collect.py` | `pilot_proc_song_collect.py` | pilot processing | pilot_proc_song_collect.py <case> [<case> ...] -- gather one 172-series pilot case's outputs into ONE record per | /sp /repo |
| `pilot_proc_song_ct_seg.py` | `pilot_proc_song_ct_seg.py` | pilot processing | pilot_proc_song_ct_seg.py = /sp/v5_va/tag_ct_seg.py (tag_ctwall's copy of vmr_ct_seg3.py, the study's ct_seg3 / | /sp |
| `pilot_proc_song_ctseg_common.py` | `pilot_proc_song_ctseg_common.py` | pilot processing | pilot_proc_song_ctseg_common.py = pilot_proc_ia172_ctseg_common.py (the 172-series pilot agent's adapter, read-only; | /sp /repo |
| `pilot_proc_song_edge.py` | `pilot_proc_song_edge.py` | wall measures | pilot_proc_song_edge.py = pilot_proc_aortaseg_edge.py (the AortaSeg pilot agent's script, read-only) for a 172-series | /sp /repo |
| `pilot_proc_song_extract.sh` | `pilot_proc_song_extract.sh` | pilot processing | pilot_proc_song_extract.sh <case> <mode>   (inside neve-build-stagea) -- the VA traces of one Song pilot case. | /sp /repo |
| `pilot_proc_song_fig.py` | `pilot_proc_song_fig.py` | pilot processing | pilot_proc_song_fig.py <case> -- figures for one 172-series pilot case (neve-build-stagea: nibabel + matplotlib). | /sp /repo |
| `pilot_proc_song_final.sh` | `pilot_proc_song_final.sh` | pilot processing | pilot_proc_song_final.sh -- the Song closing pass after post loop 2 (2nd instance, 2026-10-02 01:35; Git Bash on the host; | S R /sp /repo D |
| `pilot_proc_song_grade.py` | `pilot_proc_song_grade.py` | pilot processing | pilot_proc_song_grade.py -- per case and side GRADE + donor-ranking table for the Song pilot (SONG COPY of pilot_proc_ia172_grade.py by pilot_proc_son | /sp /repo |
| `pilot_proc_song_gross.py` | `pilot_proc_song_gross.py` | pilot processing | pilot_proc_song_gross.py -- gross shape of every traced VA side of one 172-series pilot case (PILOT_CASE), on the VMTK | /sp /repo |
| `pilot_proc_song_landmarks.py` | `pilot_proc_song_landmarks.py` | wall measures | pilot_proc_song_landmarks.py -- V1..V4 landmarks on the CT-segmented (ct_seg4) VA centrelines of one 172-series pilot | /sp /repo |
| `pilot_proc_song_measure.py` | `pilot_proc_song_measure.py` | wall measures | pilot_proc_song_measure.py -- wall detail, validity gates and gross shape of one 172-series pilot donor's | /sp /repo |
| `pilot_proc_song_protrusion.py` | `pilot_proc_song_protrusion.py` | pilot processing | pilot_proc_song_protrusion.py <case> [...] -- what the ct_seg4 label's outward relief is made of (neve-build-stagea). | /repo |
| `pilot_proc_song_qc2.py` | `pilot_proc_song_qc2.py` | QC gate (recalibrated) | pilot_proc_song_qc2.py <case> = pilot_proc_ia172_qc2.py (the 172-series pilot agent's script, read-only) with: song paths, | /sp /repo |
| `pilot_proc_song_rewindow.py` | `pilot_proc_song_rewindow.py` | pilot processing | pilot_proc_song_rewindow.py <case> -- one more re-windowed Song CT for Arterial (neve-build-stagea; pilot_proc_song). | /repo |
| `pilot_proc_song_rewindow2.py` | `pilot_proc_song_rewindow2.py` | pilot processing | pilot_proc_song_rewindow2.py <case> -- a TEXTURED re-windowing of a Song CT, ONLY as an Arterial input (neve-build-stagea; | /repo |
| `pilot_proc_song_run5.sh` | `pilot_proc_song_run5.sh` | pilot processing | pilot_proc_song_run5.sh = pilot_proc_song_run4.sh + (23:40) ctseg runs when qc/qc.json OR qc/qc_ts.json (the unchanged tool trace, see qc2 select.json | S R /sp /repo D |
| `pilot_proc_song_summary.py` | `pilot_proc_song_summary.py` | pilot processing | pilot_proc_song_summary.py -- the pilot's tables from /sp/v5_va/results/pilot_proc_song.json (per case and side | /sp /repo |
| `pilot_proc_song_tsprep.py` | `pilot_proc_song_tsprep.py` | pilot processing | pilot_proc_song_tsprep.py <case> <input> -- the TotalSegmentator input of a Song pilot case (neve-build-stagea). | /repo |
| `pilot_proc_song_tsrun.sh` | `pilot_proc_song_tsrun.sh` | pilot processing | pilot_proc_song_tsrun.sh = pilot_proc_ia172_tsrun2.sh with ONE change: container / log prefix pilot_proc_song_ (pilot_proc_song, 2026-10-01). | S R /sp /repo D |
| `pilot_proc_song_tssplit.py` | `pilot_proc_song_tssplit.py` | pilot processing | pilot_proc_song_tssplit.py <case> -- split TotalSegmentator's multilabel output (--ml) into one uint8 | /repo |
| `pilot_proc_song_vmtk.sh` | `pilot_proc_song_vmtk.sh` | pilot processing | pilot_proc_song_vmtk.sh = pilot_proc_ia172_vmtk.sh, byte copy apart from this line (pilot_proc_song). <extract_dir> [<extract_dir> ...] inside ONE nev | /sp |
| `pilot_song_rar.py` | `pilot_song_rar.py` | fetch, Song | pilot_song_rar.py (pilot_fetch_song): full RAR5 header walk of the two Song et al. 2024 archives | R /sp D |
| `scaleup_ct_seg.py` | `scaleup_ct_seg.py` | pipeline v2 post stage | pilot_proc_song_ct_seg.py = /sp/v5_va/tag_ct_seg.py (tag_ctwall's copy of vmr_ct_seg3.py, the study's ct_seg3 / | /sp |
| `scaleup_ctseg_common_ia172.py` | `scaleup_ctseg_common_ia172.py` | pipeline v2 post stage | pilot_proc_ia172_ctseg_common.py -- loader adapter for the ct_seg4 CT-edge wall segmentation of a 172-series pilot | /sp /repo |
| `scaleup_ctseg_common_song.py` | `scaleup_ctseg_common_song.py` | pipeline v2 post stage | pilot_proc_song_ctseg_common.py = pilot_proc_ia172_ctseg_common.py (the 172-series pilot agent's adapter, read-only; | /sp /repo |
| `scaleup_ctseg_fallback.sh` | `scaleup_ctseg_fallback.sh` | pipeline v2 post stage | scaleup_ctseg_fallback.sh <source> <case> [case ...]   (Git Bash on the host; scaleup_fixes, 2026-10-02 08:55) -- PIPELINE v2, | S R /repo D |
| `scaleup_fixes_before_after.py` | `scaleup_fixes_before_after.py` | pilot cases under pipeline v2 | scaleup_fixes_before_after.py -- (scaleup_fixes, 2026-10-02) BEFORE (pilot) / AFTER (pipeline v2) record of the two pre-bulk | /sp /repo |
| `scaleup_fixes_ctseg_compare.py` | `scaleup_fixes_ctseg_compare.py` | pilot cases under pipeline v2 | scaleup_fixes_ctseg_compare.py -- (scaleup_fixes, 2026-10-02) compare every pilot ct_seg4 wall with its pipeline-v2 re-run | /sp /repo |
| `scaleup_fixes_ctseg_offside.py` | `scaleup_fixes_ctseg_offside.py` | pilot cases under pipeline v2 | scaleup_fixes_ctseg_offside.py -- (scaleup_fixes, 2026-10-02) where do the few differing voxels of a NON-triggered VA lie | /sp /repo |
| `scaleup_fixes_pilot_v2.sh` | `scaleup_fixes_pilot_v2.sh` | pilot cases under pipeline v2 | scaleup_fixes_pilot_v2.sh <source> <stage/group> <case> [...] -- (scaleup_fixes, 2026-10-02) pipeline v2 on PILOT cases without | S R /sp |
| `scaleup_fixes_wall2.sh` | `scaleup_fixes_wall2.sh` | pilot cases under pipeline v2 | scaleup_fixes_wall2.sh -- (scaleup_fixes, 2nd instance, 2026-10-02 06:30) pipeline v2 POST stages after "qc select rescue" | S R |
| `scaleup_fixes_wall2_finish.sh` | `scaleup_fixes_wall2_finish.sh` | pilot cases under pipeline v2 | scaleup_fixes_wall2_finish.sh <wall2 msys pid> -- (scaleup_fixes, 2026-10-02 07:10) DECISION: group C of scaleup_fixes_wall2.sh | S R |
| `scaleup_ia172_archcut.py` | `scaleup_ia172_archcut.py` | 172-series lane | scaleup_ia172_archcut = pilot_ia172_archcut.py (copied 2026-10-02, scaleup_lane_ia172) with ONE change: output | /sp /repo |
| `scaleup_ia172_chain.sh` | `scaleup_ia172_chain.sh` | 172-series lane | scaleup_ia172_chain.sh <fetch/ts/tscpu/gpu/post/grade> <case> [case ...]   (scaleup_lane_ia172, 2026-10-02; Git Bash) | S R |
| `scaleup_ia172_eye_add.py` | `scaleup_ia172_eye_add.py` | 172-series lane | scaleup_ia172_eye_add.py ENTRIES.json (scaleup_lane_ia172 inst 3, 2026-10-02; = scaleup_song_eye_add.py with tmp suffix + print filter ia172) | /sp |
| `scaleup_ia172_grade.sh` | `scaleup_ia172_grade.sh` | 172-series lane | scaleup_ia172_grade.sh <case> [case ...]   (scaleup_lane_ia172, 2026-10-02; Git Bash) | S R |
| `scaleup_ia172_meta_visual.py` | `scaleup_ia172_meta_visual.py` | 172-series lane | scaleup_ia172_meta_visual = pilot_ia172_meta_visual.py (copied 2026-10-02, scaleup_lane_ia172) reading | /sp /repo |
| `scaleup_ia172_prearch.py` | `scaleup_ia172_prearch.py` | 172-series lane | scaleup_ia172_prearch: PRE-FETCH arch check for the 172-series scale-up (scaleup_lane_ia172). | /sp /repo |
| `scaleup_ia172_provenance.py` | `scaleup_ia172_provenance.py` | 172-series lane | scaleup_ia172_provenance: (re)write the scale-up section of thin_slice/ia172/PROVENANCE.md (scaleup_lane_ia172). | /sp /repo |
| `scaleup_ia172_qc.py` | `scaleup_ia172_qc.py` | 172-series lane | scaleup_ia172_qc = pilot_ia172_qc.py (copied 2026-10-02 for scaleup_lane_ia172) with ONE change: the visual verdicts are read | /sp /repo |
| `scaleup_ia172_run.sh` | `scaleup_ia172_run.sh` | 172-series lane | scaleup_ia172_run.sh <stage/group> <case> [case ...]   (scaleup_lane_ia172, 2026-10-02; Git Bash on the host) | S R /sp /repo D |
| `scaleup_ia172_ts.sh` | `scaleup_ia172_ts.sh` | 172-series lane | scaleup_ia172_ts.sh <case> [<case> ...] = pilot_ia172_ts.sh (copied 2026-10-02, scaleup_lane_ia172) with TWO changes: | S R /repo D |
| `scaleup_ia172_tsrun.sh` | `scaleup_ia172_tsrun.sh` | 172-series lane | scaleup_ia172_tsrun.sh = pilot_proc_ia172_tsrun2.sh (copied 2026-10-02, scaleup_lane_ia172) with ONE change: the container | S R /sp /repo D |
| `scaleup_ia172_v2patch.sh` | `scaleup_ia172_v2patch.sh` | 172-series lane | scaleup_ia172_v2patch.sh <out>  (scaleup_lane_ia172 instance 2, 2026-10-02; v2 of this file 08:45, v1 in scaleup_work/) | S D |
| `scaleup_ia172_visual.sh` | `scaleup_ia172_visual.sh` | 172-series lane | scaleup_ia172_visual.sh <case> <accept/reject> '<json object without verdict>'  -- record my by-eye verdict | S R /sp D |
| `scaleup_ia172_visual_set.py` | `scaleup_ia172_visual_set.py` | 172-series lane | scaleup_ia172_visual_set: merge my by-eye reading of one scale-up case's mip_neck.png into | /sp |
| `scaleup_select.py` | `scaleup_select.py` | pipeline v2 post stage | scaleup_select.py <source> <case> -- PIPELINE v2 trace-set SELECTION per side (scaleup_fixes, 2026-10-02), both sources. | /repo |
| `scaleup_song_archtop.py` | `scaleup_song_archtop.py` | Song lane | scaleup_song_archtop.py CASE [CASE ...] (scaleup_lane_song, 2026-10-02; read-only; neve-build-stagea) | /repo |
| `scaleup_song_args.py` | `scaleup_song_args.py` | Song lane | scaleup_song_args.py <case> (scaleup_lane_song, 2026-10-02 09:55) = pilot_proc_song_args.py + ONE diff, in the | /sp /repo |
| `scaleup_song_artredo.sh` | `scaleup_song_artredo.sh` | Song lane | scaleup_song_artredo.sh <case>   (scaleup_lane_song instance 3, 2026-10-02 09:58; Git Bash on the host; detached via nohup) | S R /sp D |
| `scaleup_song_chain.sh` | `scaleup_song_chain.sh` | Song lane | scaleup_song_chain.sh <fetch/ts/art/trace/grade> <case> [case ...]   (scaleup_lane_song, 2026-10-02; Git Bash on the host) | S R |
| `scaleup_song_collect.py` | `scaleup_song_collect.py` | Song lane | pilot_proc_song_collect.py <case> [<case> ...] -- gather one 172-series pilot case's outputs into ONE record per | /sp /repo |
| `scaleup_song_eye_add.py` | `scaleup_song_eye_add.py` | Song lane | scaleup_song_eye_add.py ENTRIES.json (scaleup_lane_song, 2026-10-02 relaunch; python:3) | /sp |
| `scaleup_song_fetch_meta.py` | `scaleup_song_fetch_meta.py` | Song lane | pilot_fetch_song_meta2.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:45 EDT): instance 1's | /sp /repo |
| `scaleup_song_fetch_picks.py` | `scaleup_song_fetch_picks.py` | Song lane | pilot_fetch_song_picks.py (pilot_fetch_song, instance 3, 2026-10-01 ~20:36 EDT): CCA and subclavian (SCA) picks | /sp /repo |
| `scaleup_song_fetch_qc.py` | `scaleup_song_fetch_qc.py` | Song lane | pilot_fetch_song_qc.py (pass 1): per Song case, read the shipped ct.nii.gz and write | /repo |
| `scaleup_song_grade.sh` | `scaleup_song_grade.sh` | Song lane | scaleup_song_grade.sh <case> [case ...]   v2 (scaleup_lane_song instance 3, 2026-10-02 10:15; v1 in scaleup_work/; Git Bash) | S R |
| `scaleup_song_grade4.sh` | `scaleup_song_grade4.sh` | Song lane | scaleup_song_grade4.sh <case> [case ...]  (scaleup_lane_song instance 4, 2026-10-02 13:15; Git Bash; one background job) | S R |
| `scaleup_song_gradechain.sh` | `scaleup_song_gradechain.sh` | Song lane | scaleup_song_gradechain.sh <case> [case ...]   (scaleup_lane_song, 2026-10-02 relaunch; Git Bash on the host) | S R |
| `scaleup_song_postchain.sh` | `scaleup_song_postchain.sh` | Song lane | scaleup_song_postchain.sh <case> [case ...]   (scaleup_lane_song instance 3, 2026-10-02 09:55; Git Bash; detached via nohup) | S R |
| `scaleup_song_provenance.py` | `scaleup_song_provenance.py` | Song lane | scaleup_song_provenance.py (scaleup_lane_song, 2026-10-02; neve-build-stagea or python:3 with the same mounts) | /sp /repo |
| `scaleup_song_refqc.sh` | `scaleup_song_refqc.sh` | Song lane | scaleup_song_refqc.sh <case> <arch_top_z> [aorta_hu]  (scaleup_lane_song, 2026-10-02; Git Bash on the host) | S R /sp /repo D |
| `scaleup_song_run.sh` | `scaleup_song_run.sh` | Song lane | scaleup_song_run.sh <stage/group> <case> [case ...]   (Git Bash on the host; agent scaleup_lane_song, 2026-10-02) | S R /sp /repo D |
| `scaleup_song_tsrun.sh` | `scaleup_song_tsrun.sh` | Song lane | scaleup_song_tsrun.sh (scaleup_lane_song, 2026-10-02) = pilot_proc_song_tsrun.sh with ONE change: container / log prefix scaleup_song_. | S R /sp /repo D |
| `scaleup_song_v2patch.sh` | `scaleup_song_v2patch.sh` | Song lane | scaleup_song_v2patch.sh <out>  (scaleup_lane_song instance 3, 2026-10-02 10:15) = scaleup_ia172_v2patch.sh (the 172 lane's v2 of 08:45) | S D |
| `scaleup_tally.py` | `scaleup_tally.py` | tally, eligible list, by-eye checks | scaleup_tally.py -- the v5 VA donor TALLY and STOP RULE (scaleup_fixes, 2026-10-02). Run any time; read-only on every input | /sp /repo |
| `scaleup_v1rescue.py` | `scaleup_v1rescue.py` | pipeline v2 post stage | scaleup_v1rescue.py <source> <case> -- PIPELINE v2 fix 1: the V1 MASK-BREAK RESCUE (scaleup_fixes, 2026-10-02). | /repo |
| `scaleup_v2_collect.py` | `scaleup_v2_collect.py` | pipeline v2 post stage | pilot_proc_song_collect.py <case> [<case> ...] -- gather one 172-series pilot case's outputs into ONE record per | /sp /repo |
| `scaleup_v2_fig.py` | `scaleup_v2_fig.py` | pipeline v2 post stage | pilot_proc_song_fig.py <case> -- figures for one 172-series pilot case (neve-build-stagea: nibabel + matplotlib). | /sp /repo |
| `scaleup_v2_grade.py` | `scaleup_v2_grade.py` | pipeline v2 post stage | pilot_proc_ia172_grade.py -- per case and side GRADE + donor-ranking table for the 172-series pilot (fourth instance of | /sp /repo |
| `scaleup_v2_gross.py` | `scaleup_v2_gross.py` | pipeline v2 post stage | pilot_proc_song_gross.py -- gross shape of every traced VA side of one 172-series pilot case (PILOT_CASE), on the VMTK | /sp /repo |
| `scaleup_v2_post.sh` | `scaleup_v2_post.sh` | pipeline v2 post stage | scaleup_v2_post.sh <source> <stage/group> <case> [case ...]   (Git Bash; scaleup_fixes, 2026-10-02) | S R /sp /repo D |
| `scaleup_v2_protrusion.py` | `scaleup_v2_protrusion.py` | pipeline v2 post stage | scaleup_v2_protrusion.py <case> [...] -- (scaleup_fixes, 2026-10-02) PIPELINE v2 protrusion: the source's UNCHANGED pilot | /sp |
| `sources.py` | `sources.py` | VMR 0248 donor and source loaders | Loaders for the two ready-made cervical-VA sources of the v5 work. | R /sp /repo |
| `tag_calibration.py` | `tag_calibration.py` | study modules used as libraries | tag_calibration.py -- assemble the recalibration of the v5 extraction and graft checks on the two ground-truth (GT) | /sp /repo |
| `tag_calibration_entry.py` | `tag_calibration_entry.py` | study modules used as libraries | tag_calibration_entry.py -- offline search for a corrected foramen-entry rule, on the per-level distance profiles | /sp |
| `tag_calibration_gross.py` | `tag_calibration_gross.py` | study modules used as libraries | tag_calibration_gross.py -- gross reference ranges with TaG-Net A and B beside VMR 0248, the host and SPL. | /sp /repo |
| `tag_calibration_paths.py` | `tag_calibration_paths.py` | study modules used as libraries | tag_calibration_paths.py -- ONE QC harness for every vertebral-artery (VA) path the calibration needs: the ground | /sp /repo |
| `tag_ct_seg.py` | `tag_ct_seg.py` | study modules used as libraries | tag_ct_seg.py = /sp/v5_va/vmr_ct_seg3.py (the study's ct_seg3 / ct_seg4 CT segmentation), COPIED for TaG-Net | /sp |
| `tag_ctwall_assemble.py` | `tag_ctwall_assemble.py` | study modules used as libraries | tag_ctwall_assemble.py -- assemble results/tag_ctwall.json (the tag_ctwall deliverable) and the figures, from | /sp |
| `tag_ctwall_common.py` | `tag_ctwall_common.py` | study modules used as libraries | tag_ctwall_common.py -- loader adapter + shared constants for the tag_ctwall agent (the real CT wall of TaG-Net | /sp /repo |
| `tag_ctwall_edge.py` | `tag_ctwall_edge.py` | study modules used as libraries | tag_ctwall_edge.py -- what the TaG-Net A CTA can resolve at the VA lumen edge, and what touches the VA, | /sp /repo |
| `tag_ctwall_seeds.py` | `tag_ctwall_seeds.py` | study modules used as libraries | tag_ctwall_seeds.py <tag> -- VMTK seeds for a tag_ct_seg.py run, repaired where the label does not reach an end. | /sp |
| `tag_labels_gross.py` | `tag_labels_gross.py` | study modules used as libraries | tag_labels_gross.py -- gross shape of the TaG-Net GT VAs with /sp/v5_va/va_metrics.py UNCHANGED (centreline mode). | /sp /repo |
| `tag_totalseg_make1p5.py` | `tag_totalseg_make1p5.py` | study modules used as libraries | tag_totalseg_make1p5.py <fine_ts_dir> <out_ts_dir> -- a 1.5 mm isotropic RAS copy of a 0.4 mm ts dir, laid out like a |  |
| `tag_totalseg_qc.py` | `tag_totalseg_qc.py` | study modules used as libraries | tag_totalseg_qc.py -- does TotalSegmentator 2.18.0 "total" (neve-totalseg-gpu) reproduce the TotalSegmentator v3 | /sp /repo |
| `tag_totalseg_run.sh` | `tag_totalseg_run.sh` | study modules used as libraries | tag_totalseg_run.sh <input_ct> <out_seg_dir> <name> [extra TotalSegmentator args] | S R /sp /repo D |
| `v5_eligible_final.py` | `v5_eligible_final.py` | tally, eligible list, by-eye checks | v5_eligible_final.py -- (v5_eye172, 2026-10-02) the FINAL eligible v5 VA donor list, from the re-run tally. | R /sp |
| `v5_eye172_bridges.py` | `v5_eye172_bridges.py` | tally, eligible list, by-eye checks | v5_eye172_bridges.py -- (v5_eye172) print, per pending side, the qc record items the by-eye verdict needs: bridges | /repo |
| `v5_eye172_eye_add.py` | `v5_eye172_eye_add.py` | tally, eligible list, by-eye checks | v5_eye172_eye_add.py -- (v5_eye172, 2026-10-02) append the by-eye verdicts of the 7 flagged 172-series pilot sides to | /sp |
| `v5_eye172_look.py` | `v5_eye172_look.py` | tally, eligible list, by-eye checks | v5_eye172_look.py <source> <case> <side> [ext] [qc_file] -- (v5_eye172, 2026-10-02) read-only by-eye look at one donor VA | /sp /repo |
| `v5_eye172_look_all.sh` | `v5_eye172_look_all.sh` | tally, eligible list, by-eye checks | v5_eye172_look_all.sh -- idempotent: one look PNG per side (skips existing). Runs INSIDE neve-build-stagea. | /sp |
| `v5_eye172_origin.py` | `v5_eye172_origin.py` | tally, eligible list, by-eye checks | v5_eye172_origin.py <source> <case> <side> [ext] -- (v5_eye172, 2026-10-02) read-only by-eye look at a donor VA ORIGIN: | /sp /repo |
| `verify_tag_ctwall_image.py` | `verify_tag_ctwall_image.py` | study modules used as libraries | verify_tag_ctwall_image.py -- adversarial verifier (prefix verify_tag_ctwall): image-domain checks of the | /sp /repo |
| `vmr_ccj.py` | `vmr_ccj.py` | VMR 0248 donor and source loaders | Cranio-cervical junction views: midsagittal zoom (basion / opisthion), and the V3 course of both VAs |  |
| `vmr_common.py` | `vmr_common.py` | VMR 0248 donor and source loaders | Shared helpers for the vmr agent (VMR 0248 VA from the CT and from the lofted walls). | /sp /repo |
| `vmr_compare.py` | `vmr_compare.py` | VMR 0248 donor and source loaders | VMR 0248, CT-derived vs lofted: (1) CT edge sharpness (erf fit to lumen edges: what wall scales the CT | /sp |
| `vmr_ct_seg3.py` | `vmr_ct_seg3.py` | VMR 0248 donor and source loaders | v3 (this file) = v2 + an UPPER threshold where a BRIGHTER neighbour touches the lumen before it has |  |
| `vmr_fm.py` | `vmr_fm.py` | VMR 0248 donor and source loaders | Zoomed midsagittal slices (3 x positions) around the foramen magnum for picking basion / opisthion. |  |
| `vmr_landmarks.py` | `vmr_landmarks.py` | VMR 0248 donor and source loaders | Bony landmarks along the VMR 0248 VAs (CT): transverse-foramen enclosure profile along the |  |
| `vmr_measure.py` | `vmr_measure.py` | VMR 0248 donor and source loaders | VMR 0248 VAs through the shared suite (va_metrics.measure_va, UNCHANGED) and the grader | /sp |
| `vmr_run.sh` | `vmr_run.sh` | VMR 0248 donor and source loaders | usage: vmr_run.sh <image> <script> [args...]   (Git Bash) | S R /sp D |
| `vmr_vmtk.py` | `vmr_vmtk.py` | VMR 0248 donor and source loaders | VMTK centerlines (MaximumInscribedSphereRadius) for the VMR 0248 VA surfaces. | /sp |
| `vmr_vmtk_all.sh` | `vmr_vmtk_all.sh` | VMR 0248 donor and source loaders | all VMTK centerlines for the vmr agent, inside ONE neve-build-vmtk container | /sp /repo |
| `va_pool/rzip.py` | `../va_pool/rzip.py` | fetch, 172-series | Remote ZIP probe: list entries and read NIfTI headers via HTTP range requests. |  |

### graft/

| copy | original (under v5_va/) | group | what | paths |
|---|---|---|---|---|
| `graft_v5_assign.py` | `graft_v5_assign.py` | registry graft and assignment | graft_v5_assign.py -- ONE-TO-ONE assignment of donor units (donor patients: a VA pair, or one side) to TopBrain | /sp /repo |
| `graft_v5_reg.py` | `graft_v5_reg.py` | registry graft and assignment | graft_v5_reg.py -- v5 REGISTRY graft: strategy B for ANY donor record (graft_v5_reg_donor.py), with the | /sp /repo |
| `graft_v5_reg_donor.py` | `graft_v5_reg_donor.py` | registry graft and assignment | graft_v5_reg_donor.py -- build the v5 REGISTRY donor record of one donor patient (one "unit", 1-2 VA sides). | /sp /repo |
| `graft_v5_reg_dryall.sh` | `graft_v5_reg_dryall.sh` | registry graft and assignment | graft_reg: dry-run EVERY donor unit in v5_reg/donors against all 223 v4 anatomies (graft_v5_reg.py, default mode: | R /sp /repo D |
| `graft_v5_reg_sep.py` | `graft_v5_reg_sep.py` | route13 move | graft_v5_reg_sep.py -- graft_v5_reg 1.1 + a SEPARATION OBJECTIVE for the placement (v5_route13, 2026-10-02). | /sp /repo |
| `graft_v5_va.py` | `graft_v5_va.py` | prototype graft | v5 PROTOTYPE: a real cervical vertebral artery from the subclavian origin, spliced to v4's TopBrain distal VA. | /sp /repo |
| `route13/cmp_iso.py` | `route13/cmp_iso.py` | route13 move | v5_route13: compare the RCCA route wall (right_route_detail's own attribution and depth) of two undecimated unions | /sp |
| `route13/collect13.py` | `route13/collect13.py` | route13 move | v5_route13: collect the 13 anatomies' variants (v5 as baked, moved_<tag>, prio[_<tag>]) into | /sp /repo |
| `route13/diag13.py` | `route13/diag13.py` | route13 move | v5_route13 step 1 (read-only on anatomies_v4 / anatomies_v5): WHICH VA piece takes WHICH RCCA real wall. | /sp /repo |
| `route13/dr13.sh` | `route13/dr13.sh` | route13 move | v5_route13: run a command in a container. /repo READ-ONLY except carotid_data/va_sources/v5_reg/route13 (rw scratch). | S R /repo D |
| `route13/graft13.sh` | `route13/graft13.sh` | route13 move | v5_route13: re-plan anatomies with graft_v5_reg_sep.py (inside eve-training-fixed), one per line of a job file: | /sp /repo |
| `route13/lumen_iso.py` | `route13/lumen_iso.py` | route13 move | v5_route13: RCCA / LCCA route min lumen on the UNDECIMATED unions (v5 as baked: iso_v5/<name>.vtp; chosen variant: | /sp /repo |
| `route13/run_bake.py` | `route13/run_bake.py` | route13 move | v5_route13: bake ONE scratch anatomy with bake_full/bake_full_one.py (unchanged), the ownership chosen by env | /sp |
| `route13/sum13.py` | `route13/sum13.py` | route13 move | v5_route13: compact per-anatomy table from results/v5_route13.json (chosen variant vs v5 as baked; prio where run). | /sp |
| `route13/worker13.sh` | `route13/worker13.sh` | route13 move | v5_route13 worker (inside neve-build-meshlab): bake + measure each name of a list in one VARIANT folder of | /sp /repo |
| `route13/worker13c.sh` | `route13/worker13c.sh` | route13 move | v5_route13 worker with CLAIMS (several workers on one list): claim <V>_bf/_claim/<name> by atomic mkdir, then bake + | /sp /repo |
| `route13/worker13w.sh` | `route13/worker13w.sh` | route13 move | v5_route13: like worker13.sh, but waits (up to 3 h per anatomy) for the graft record /sp/v5_va/route13/graft/<TAG>/<name>.json | /sp |
| `route13/graft_v5_reg_sep.v2.py` | `route13/graft_v5_reg_sep.v2.py` | route13 move | graft_v5_reg_sep **1.0** (backup taken 2026-10-02 23:49): the version that made the 12 moved_ext picks (23:23-23:39); see "The route13 tool version" | /sp /repo |
| `v5_assign_work/alt_all_served.py` | `v5_assign_work/alt_all_served.py` | assignment checks and reports | v5_assign: alternative B -- PRIMARIES ONLY, every TopBrain patient must get a primary with >= 1 passing VA | /sp /repo |
| `v5_assign_work/alt_orders.py` | `v5_assign_work/alt_orders.py` | assignment checks and reports | v5_assign: sensitivity of the one-to-one assignment to the order of the objective (primaries only, every patient | /sp /repo |
| `v5_assign_work/check_records.py` | `v5_assign_work/check_records.py` | assignment checks and reports | v5_assign: check every donor record in v5_reg/donors_v5 against the final eligible list and for internal sanity. | /sp /repo |
| `v5_assign_work/check_written.py` | `v5_assign_work/check_written.py` | assignment checks and reports | v5_assign: check the written (unbaked) anatomies_v5 against anatomies_v4 and the plan. | /sp /repo |
| `v5_assign_work/dry_stats.py` | `v5_assign_work/dry_stats.py` | assignment checks and reports | v5_assign: per-unit / per-side / per-gate statistics of the dry run v5_reg/dry_v5 (27 units x 223 anatomies). | /sp /repo |
| `v5_assign_work/regress_compat_v11.py` | `v5_assign_work/regress_compat_v11.py` | assignment checks and reports | graft_reg: compare graft_v5_reg --compat (VMR unit, native hand) with the prototype's final dry run | /sp /repo |
| `v5_assign_work/v5_assign_report.py` | `v5_assign_work/v5_assign_report.py` | assignment checks and reports | v5_assign: the final assignment record /sp/v5_va/results/v5_assignment.json, built from the plan, the assignment | /sp /repo |
| `verify_graft_mesh_chiral.py` | `verify_graft_mesh_chiral.py` | registry graft and assignment | verify_graft_mesh: chirality of the host frame vs VMR 0248's frame, from anatomy alone (eve-training-fixed). | /sp /repo |
| `route13/names13.txt` | `route13/names13.txt` | route13 move | (byte copy) |  |
| `route13/order_ext.txt` | `route13/order_ext.txt` | route13 move | (byte copy) |  |
| `route13/jobsA0.txt` | `route13/jobsA0.txt` | route13 move | (byte copy) |  |
| `route13/jobsA1.txt` | `route13/jobsA1.txt` | route13 move | (byte copy) |  |
| `route13/jobsB.txt` | `route13/jobsB.txt` | route13 move | (byte copy) |  |
| `route13/jobsC.txt` | `route13/jobsC.txt` | route13 move | (byte copy) |  |
| `route13/one_w012l021.txt` | `route13/one_w012l021.txt` | route13 move | (byte copy) |  |
| `donor_records/ia172_015/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_015/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_039/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_039/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_058/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_058/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_069/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_069/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_071/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_071/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_136/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_136/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_141/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_141/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_149/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_149/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_159/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_159/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_164/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_164/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_165/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_165/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_172/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_172/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_190/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_190/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/ia172_192/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/ia172_192/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218002/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218002/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218008/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218008/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218032/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218032/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218040/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218040/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218094/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218094/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218095/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218095/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218100/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218100/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218108/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218108/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218109/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218109/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218113/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218113/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218138/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218138/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/song_AHMU1218141/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/song_AHMU1218141/record.json` | donor unit records | (byte copy) | /repo |
| `donor_records/vmr0248/record.json` | `carotid_data/va_sources/v5_reg/donors_v5/vmr0248/record.json` | donor unit records | (byte copy) | /repo |

### bake/

| copy | original (under v5_va/) | group | what | paths |
|---|---|---|---|---|
| `bake_full/bake_full_one.py` | `bake_full/bake_full_one.py` | full-bake drivers and records | v5_bake_full: bake ONE anatomy of carotid_data/anatomies_v5 in place with /sp/v5_va/bake_v5b.py (bake_meshes_v4 | /sp /repo |
| `bake_full/check_inputs.py` | `bake_full/check_inputs.py` | full-bake drivers and records | v5_bake_full: are the NON-VA bake inputs of every v5 anatomy byte-identical to v4's? (every centreline file except | /sp /repo |
| `bake_full/collect.py` | `bake_full/collect.py` | full-bake drivers and records | v5_bake_full: collect the 223 bake + measure records into /sp/v5_va/results/v5_bake_full.json, and (--build, only | /sp /repo |
| `bake_full/diag_tubes.py` | `bake_full/diag_tubes.py` | full-bake drivers and records | v5_bake_full diagnosis (read-only): a tube-only mesh with a handle ("centerline-level fusion"). Which VA makes | /sp /repo |
| `bake_full/diag_tubes2.py` | `bake_full/diag_tubes2.py` | full-bake drivers and records | v5_bake_full diagnosis (read-only): localise the handle of a tube-only union along one VA branch (10 mm bins, | /sp /repo |
| `bake_full/keep_v4_va.py` | `bake_full/keep_v4_va.py` | full-bake drivers and records | v5_bake_full: the LAST rung of the bake failure rule -- keep v4's VA on one side of one anatomy (flagged), when | /repo |
| `bake_full/launch.sh` | `bake_full/launch.sh` | full-bake drivers and records | v5_bake_full launcher (host, Git Bash). Starts NW detached worker containers v5bake_w<k> (docker run -d --rm). | S R /sp D |
| `bake_full/measure_full_one.py` | `bake_full/measure_full_one.py` | full-bake drivers and records | v5_bake_full: gates + VA wall detail of ONE baked v5 anatomy (after bake_full_one.py), recorded in the anatomy's | /sp /repo |
| `bake_full/obj_fix.py` | `bake_full/obj_fix.py` | full-bake drivers and records | v5_bake_full: an anatomy whose 60k mesh has NO clean decimation to the 23.5k collision budget (v4's obj_ladder | /sp /repo |
| `bake_full/plan.py` | `bake_full/plan.py` | full-bake drivers and records | v5_bake_full: the bake order -- hardest first, so the ladder cases do not finish last. | /sp /repo |
| `bake_full/status.sh` | `bake_full/status.sh` | full-bake drivers and records | v5_bake_full: one status line set (host).   bash status.sh | R |
| `bake_full/worker.sh` | `bake_full/worker.sh` | full-bake drivers and records | v5_bake_full worker (inside neve-build-meshlab). Walks the ordered list (hardest first), claims an anatomy by an | /sp /repo |
| `bake_v5.py` | `bake_v5.py` | bake wrapper (base) | Bake v5 prototype anatomies with topbrain_tools/bake_meshes_v4.py UNCHANGED, plus the donor cervical VA | /sp /repo |
| `bake_v5b.py` | `bake_v5b.py` | bake (handle rule) | bake_v5b: bake_v5 (bake_meshes_v4 UNCHANGED + the donor vaC_R / vaC_L sections) with the USER HANDLE RULE | /sp /repo |
| `bake_v5c.py` | `bake_v5c.py` | route-priority fallback | bake_v5c: bake_v5b + ROUTE-PRIORITY OWNERSHIP (v5_route13 fallback, 2026-10-02). | /sp |
| `bake_full/order.txt` | `bake_full/order.txt` | full-bake drivers and records | (byte copy) |  |
| `bake_full/plan.json` | `bake_full/plan.json` | full-bake drivers and records | (byte copy) |  |

### checks/

| copy | original (under v5_va/) | group | what | paths |
|---|---|---|---|---|
| `checks/chk_one.py` | `checks/chk_one.py` | post-bake checks | v5_checks items 1 (strict mesh audit) and 2 (route + VA wall detail) for ONE baked v5 anatomy. | /sp /repo |
| `checks/collect.py` | `checks/collect.py` | post-bake checks | v5_checks: collect items 1-5 into /sp/v5_va/results/v5_checks.json (+ a printed summary for the notes). | /sp /repo |
| `checks/dr.sh` | `checks/dr.sh` | post-bake checks | v5_checks: run a command in a container with the v4/v5 mounts; /repo READ-ONLY (checks never write the repo). | S R D |
| `checks/entry_geom.py` | `checks/entry_geom.py` | post-bake checks | v5_checks item 4 diagnostic: the RVA entry geometry the SOFA schedule meets at fork 2 (the VA leaving the | /sp /repo |
| `checks/item5.sh` | `checks/item5.sh` | post-bake checks | v5_checks item 5 (host, read-only git): anatomies_v4 against git HEAD.   bash item5.sh <out.json> | S R |
| `checks/launch.sh` | `checks/launch.sh` | post-bake checks | v5_checks launcher (host). Starts NW detached workers v5chk_w<k>; idempotent (a live worker is not restarted; | S /sp |
| `checks/loader_v5.py` | `checks/loader_v5.py` | post-bake checks | v5_checks item 4a: the training loader over ALL 223 v5 anatomies, no SOFA (v4 audit sim_loader.py, pointed at | /sp /repo |
| `checks/ostium_lumen.py` | `checks/ostium_lumen.py` | post-bake checks | v5_checks item 4 diagnostic: the RVA lumen in the OBJ around the VA ostium (s_o - 3 .. s_o + 10 mm) for the 25 SOFA | /sp /repo |
| `checks/select_sofa.py` | `checks/select_sofa.py` | post-bake checks | v5_checks item 4: pick the SOFA set = the 15 most changed anatomies + 10 random (neve-build-meshlab, read-only). | /sp /repo |
| `checks/sofa_rva.py` | `checks/sofa_rva.py` | post-bake checks | v5_checks item 4b: SOFA load test that EXERCISES THE NEW RVA, from the training insertion (eve-training-fixed). | /sp /repo |
| `checks/sofa_worker.sh` | `checks/sofa_worker.sh` | post-bake checks | v5_checks SOFA worker (eve-training-fixed, /repo read-only). Items "v5:<name>" (anatomies_v5 -> sofa/run4/v5) and | /sp /repo |
| `checks/status.sh` | `checks/status.sh` | post-bake checks | v5_checks: one status block (host). | S |
| `checks/tb_geom.py` | `checks/tb_geom.py` | post-bake checks | v5_checks item 2, TopBrain piece unchanged (geometry): for every grafted VA, the TopBrain distal piece of v5 (cut -> | /sp /repo |
| `checks/va_lumen.py` | `checks/va_lumen.py` | post-bake checks | v5_checks (extra to item 1): the vertebral lumens in the shipped OBJ, v5 against v4, with v4's route-check | /sp /repo |
| `checks/worker.sh` | `checks/worker.sh` | post-bake checks | v5_checks worker (inside neve-build-meshlab, /repo read-only). Claims an anatomy by atomic mkdir of claim/<name>, | /sp |
| `ref_v4_union.py` | `ref_v4_union.py` | measure libraries | Reference part c: the v4 VAs as shipped, on the undecimated union. | /sp /repo |
| `swap13/chk13.py` | `swap13/chk13.py` | swap and final record | v5_swap13: checks/chk_one.py UNCHANGED (strict mesh audit = item 1, route + VA detail k24 = item 2) on one swapped | /sp |
| `swap13/diag_lost.py` | `swap13/diag_lost.py` | swap and final record | v5_swap13 diagnostic: for an anatomy whose RCCA right_route_detail under EXACT ownership is below v4, find the v4 | /sp |
| `swap13/final_collect.py` | `swap13/final_collect.py` | swap and final record | v5_swap13 step 4: the final record of the 223 v5 anatomies after the route13 swap. | /sp /repo |
| `swap13/loader_v5.py` | `swap13/loader_v5.py` | swap and final record | v5_checks item 4a: the training loader over ALL 223 v5 anatomies, no SOFA (v4 audit sim_loader.py, pointed at | /sp /repo |
| `swap13/lumen13.py` | `swap13/lumen13.py` | swap and final record | v5_swap13: RCCA / LCCA route min lumen on the UNDECIMATED unions with bake_meshes_v4.route_report (unchanged), as | /sp /repo |
| `swap13/rerun_v1.sh` | `swap13/rerun_v1.sh` | swap and final record | v5_swap13: re-run rrdx.py --force on every finished output written by version 1 (no "rrdx_version" key), one at a | /sp |
| `swap13/rrdx.py` | `swap13/rrdx.py` | swap and final record | v5_swap13: right_route_detail v5 vs v4 under EXACT ownership (k24 beside it) for ONE anatomy, on the undecimated | /sp |
| `swap13/run_a.sh` | `swap13/run_a.sh` | swap and final record | v5_swap13 container A entry: NW workers over jobs_a.txt, then exit.   bash run_a.sh [NW=5] | /sp |
| `swap13/sofa_worker.sh` | `swap13/sofa_worker.sh` | swap and final record | v5_swap13 SOFA worker (eve-training-fixed, /repo read-only): checks/sofa_rva.py UNCHANGED (final run4 schedule: | /sp /repo |
| `swap13/swap.sh` | `swap13/swap.sh` | swap and final record | v5_swap13 step 1 (host, Git Bash). Idempotent: per-anatomy done-markers in swap13/done/. | S R |
| `swap13/valumen13.py` | `swap13/valumen13.py` | swap and final record | v5_swap13: checks/va_lumen.py's one() UNCHANGED (VA lumen in the shipped OBJ, v4's route-check measures) on the 13 | /sp /repo |
| `swap13/vvw13.py` | `swap13/vvw13.py` | swap and final record | v5_swap13: vvw/measure.py UNCHANGED (verify_v5_wall's own right_route_detail + VA wall detail, k24 AND exact | /sp |
| `swap13/worker_a.sh` | `swap13/worker_a.sh` | swap and final record | v5_swap13 job worker (inside neve-build-meshlab; /repo read-only). Jobs in swap13/jobs_a.txt ("kind name"), claimed by | /sp /repo |
| `v5_split.py` | `v5_split.py` | split checker | v5_split.py -- donor-overlap train/test split checker for the carotid anatomy sets (v3, v4, v5). | /repo |
| `va_detail_v5.py` | `va_detail_v5.py` | measure libraries | A1-A3 of the natural-anatomy standard (notes/ref.md) on baked v5 prototypes: VA wall detail measured on the | /sp /repo |
| `va_entry_check.py` | `va_entry_check.py` | VA-entry geometry check |  | /repo |
| `va_entry_geom.py` | `va_entry_geom.py` | VA-entry geometry check |  | /repo |
| `va_metrics.py` | `va_metrics.py` | measure libraries | va_metrics.py -- the shared vertebral-artery (VA) metric suite for vetting v5 sources. |  |
| `va_standard.py` | `va_standard.py` | measure libraries | Grade va_metrics.measure_va results against the natural-anatomy standard | /sp |
| `vvw/measure.py` | `vvw/measure.py` | exact-ownership measure | verify_v5_wall: OWN implementation of right_route_detail's definition, applied per VA stretch and to the RCCA | /sp /repo |
| `vvw/rrd_v4.py` | `vvw/rrd_v4.py` | exact-ownership measure | verify_v5_wall: right_route_detail v5 vs v4 UNROUNDED, with v4's exact union rebuilt from mesh_v4.json (repo | /sp |
| `checks/names.txt` | `checks/names.txt` | post-bake checks | (byte copy) |  |
| `swap13/picks.tsv` | `swap13/picks.tsv` | swap and final record | (byte copy) |  |
| `swap13/names13.txt` | `swap13/names13.txt` | swap and final record | (byte copy) |  |
| `swap13/jobs_a.txt` | `swap13/jobs_a.txt` | swap and final record | (byte copy) |  |
| `swap13/sofa_list.txt` | `swap13/sofa_list.txt` | swap and final record | (byte copy) |  |
| `v4_audit/topo_audit.py` | `../v4_build/audit/topo_audit.py` | v4 audit | Topology / surface-validity audit of the shipped v4 (and v3) meshes. | /sp /repo |
| `v4_audit/sim_loader.json` | `../v4_build/audit/sim_loader.json` | v4 audit | (byte copy) |  |

