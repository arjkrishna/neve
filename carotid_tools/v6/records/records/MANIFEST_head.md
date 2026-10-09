# carotid_tools/v6: the v6 toolchain, copied from the build session

These are the scripts that built `carotid_data/anatomies_v6` (see `CAROTID_V6.md` at the repo root). They ran from a
session scratchpad, which is temporary, so they are copied here. Only what the build used is copied, not the
experiments, probes and patch scripts around it. The copies are not edited: each code file gained one header line naming
its source, and nothing else changed.

- Source: `C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\`
  (`/sp/` inside the build containers): `v6\` (the v6 tools), plus the `v5_va\` and `v4_build\` files the v6 tools import.
- Copied 2026-10-07 by the `s2_records` agent of the v6 stage-2 workflow (wf_87bbef23-7e4), with
  `records/records/make_tools_v6.py`; re-staged 2026-10-08 by `s3_fix` (the final verifier's must-fix list) with
  make_tools_v6 1.1, which adds the `s3_fix/` folder, the s3_fix manifest and the frozen s2_records versions.
- **172 files**: build 69, donor 59, v5_deps 20, set_checks 10, gates 6, records 5, item_b 3.
- Not copied: the probes, `peek*` / `patch*` / `i2_*` diagnosis scripts and superseded development copies; the stage-1
  pilot tools (`pilot_build/`, except the three item-b files below); the verifiers' own code (their notes are in
  `carotid_data/anatomies_v6/_records/notes/`); the donor records and surfaces (`carotid_data/v6_work/wall/`,
  `carotid_data/v6_work/va_s2/`, git-ignored); the workflow scripts (`*.js`).

## Files in this folder

| file | what |
|---|---|
| `MANIFEST.md` | this file |
| `FILES.tsv` | one row per copied file: stage, original path (relative to the scratchpad root), copy path, whether a header line was added, sha256 of the original and of the copy, size, md5 of the original, why it is here |
| `TOOLCHAINS.tsv` | every entry of the build's eight md5 manifests (seven of s2, s3_fix's one), resolved: the copied file that has that md5 (the current file, or the frozen copy of an earlier version), or the repo module with its md5 checked |
| `restore_layout.sh` | rebuilds the build-time layout: `bash restore_layout.sh <X>` writes every file back to `<X>/<original path>` with the header line removed and checks each against the original sha256 |
| `build/` | the build toolchain proper: every scratchpad file of the toolchain manifests -- `build_v6` (stages), `compose_v6` (forks, siphons, ECA tips), `graft_v6_va` (VA, donor junction, D17 branch), `bake_v6`, `checks_v6`, `geom_v6`, `orient_gates`, `loader_one`, the SOFA scripts, the full-build runners and assembly scripts (`full/`), the manifests themselves, and the frozen earlier versions that made records still in the final tree |
| `donor/` | the stage-1 / stage-2 tools that made the donor-side inputs the build reads: CT-edge walls with the D15 local bone-side fix (`wall/`, `wall_local/`), the donor subclavian paths and junction walls (`va/subclavian_paths.py`, `va/sc_post3.py`, `wall/jfix/`), D16 / D17 (`wall_d17/`), the C4 check |
| `gates/` | the heading evidence behind the per-patient Zenodo frames (`heading/`, with `zenodo_frames.json`) and the gate API test |
| `item_b/` | the route13 diagnosis that writes the v4 RCCA wall the build's route13 stage reads (case_w_018b_left__topcow_mr_002) |
| `set_checks/` | the set-level checks on anatomies_v6 (split, loader, SOFA, orientation, identity, coverage diagnosis) |
| `records/` | this stage: `BUILD_v6.json` generator and checker, the `_records` copy, this copy (`make_tools_v6.py`, `finalize_tools.sh`, `imports_check.py`, the restore script and this file's head); s3_fix's versions 1.1 (`build_v6_record`, `check_build_record`, `make_tools_v6`, `finalize_tools`) with the s2_records 1.0 versions frozen as `*_v1_0.*` |
| `s3_fix/` | the final verifier's must-fix list (2026-10-08): the lost-wall diagnoses (`diag_*.py`), the re-bake / check runners (`batch_rebake.sh`, `final_other.sh`, `batch_sofa.sh`, `batch_cov.sh`, `dr_s3.sh`), the OBJ fold-over fix (`fold_fix.py` 1.0), the provenance stamp (`stamp_prov_s3.py` = stamp_prov 1.1), the copy of the changed anatomies (`copy_v6_s3.sh` = copy_v6 1.2), the coverage / SOFA summaries, the install script and the item lists of the final round. `bake_v6` 1.2 and `checks_v6` 1.6 themselves are in `build/` |
| `v5_deps/` | the `v5_va/` and `v4_build/` files the v6 tools import or list; 16 of the 20 are byte-identical to the originals of `carotid_tools/v5` (the other 4: `v4_build/audit/topo_audit.py`, `v5_va/load_test_v5.py`, `v5_va/tbo/va_narrow/{bone_scan,ct_check}.py`) |

Inside each stage folder a file keeps its path relative to `v6/` (or the scratchpad root for `v5_deps/`):
`build/carotid/compose_v6.py` was `v6/carotid/compose_v6.py`, and `v5_deps/v5_va/v5_split.py` was `v5_va/v5_split.py`.
The scripts call each other by those paths (`/sp/v6/<path>`, `/sp/v5_va/<path>`), so they do not run from the stage
folders directly; restore the layout first.

**What the copies differ in.** Every `.py` and `.sh` has one extra line, right after the shebang where there is one:

    # [carotid_tools/v6 copy] source: <scratchpad>\v6\<path> (= /sp/v6/<path> in the build containers)

JSON files and the md5 manifests are byte copies. `restore_layout.sh` removes the line, and every restored file then
matches the sha256 of the file the build ran. **Restore test** (2026-10-08, s3_fix, `records/records/finalize_tools.sh`
1.1): all 253 restored files sha256-identical to the originals and byte-identical to the scratchpad files; on the
restored layout `md5sum -c` of the final manifest `build/build/TOOLCHAIN_s3fix.md5` (2288f778) passes on all 88
scratchpad entries; across the eight manifests 463 entries resolve to a copied file (current or frozen) or to a repo
module with the same md5, and 2 are not preserved (below); the 63 repo-module rows match the repo's tracked files
(`TOOLCHAINS.tsv`). (s2_records' test of 2026-10-07 ran the same on d65b7209, 63 scratchpad entries.)

## The toolchain versions

The build ran under ONE frozen toolchain at a time (decision D25): `build_v6` 1.2 refuses to run a stage, and every bake /
check worker refuses to start, unless every file of the md5 manifest matches. The manifests, in order:

| manifest (md5) | files | what it froze | records it made that are in anatomies_v6 |
|---|---:|---|---|
| `TOOLCHAIN_s2i2.md5` (bedf172b) | 46 | the stage-2 integration: compose_v6 3.1.0, graft_v6_va 1.3, checks_v6 1.3, bake_v6 1.1 | the DRY on all 223 (`/bld/dry223b`), whose carotid plan the final compose replays (`--plan`); the first full build's bakes (216 anatomies) and their checks parts (carried) |
| `TOOLCHAIN_s2build_80b29bc3.md5` | 57 | + graft_v6_va 1.4 (fold gate, pose filter) | the 7 re-bakes and their checks parts (carried) |
| `TOOLCHAIN_s2build_11875b8d.md5` | 59 | graft_v6_va 1.4.1 | none (superseded) |
| `TOOLCHAIN_s2build_a80644e0.md5` | 59 | **the final compose**: compose_v6 3.2.0, graft_v6_va 1.4.2, + runners | every anatomy's composed curves and provenance (`v6_build.toolchain.manifest_md5`) |
| `TOOLCHAIN_s2build_88ab481d.md5` | 65 | + checks_v6 1.4 (D26), the final checks round | route_d22 on 222 anatomies |
| `TOOLCHAIN_s2build.md5` (d65b7209) | 72 | + checks_v6 1.5 (D27), stamp_prov, copy_v6 1.1, v6_build_collect 1.3 | route_d22 on case_k_011_left__topcow_mr_012_L; the provenance stamp (`v6_checks.toolchain_manifest_md5`) and the copy of the 206 anatomies s3_fix did not change |
| `TOOLCHAIN_s3fix.md5` (**2288f778**, final) | 97 | d65b7209's files with **bake_v6 1.2** (L1 lattice shift, L2 route-island bridge; both opt-in, defaults = 1.1) and **checks_v6 1.6** (parts coverage + folds), the frozen 1.1 / 1.5 copies, and the s3_fix scripts | the 8 re-bakes, the OBJ fold fix, checks_v6 1.6 on the 17 changed anatomies (and its coverage / folds parts on all 223), their provenance stamp and copy |
| `full/TOOLCHAIN_s2build_supplement.md5` | 10 | the first full build's runners and the two modules the gate verifier found imported but unlisted | the first build's bakes |

A manifest entry whose file was later versioned is matched (by md5) to the frozen copy kept beside it, and that copy is
here: `carotid/compose_v6_v310.py` (3.1.0), `va/graft_v6_va_1p3.py`, `_1p4.py`, `_1p4p1.py`, `checks/checks_v6_v1_3.py`,
`_v1_4.py`, `_v1_5.py`, `bake/bake_v6_v1_1.py`, `full/check4_helper_v1_0.py`, `full/copy_v6_v1_0.sh`,
`full/v6_build_collect_v1_2.py`. To re-run an earlier
version, run the frozen file in place of the current name. **Two supplement entries are not preserved**: the first
build's `full/copy_v6.sh` (md5 2042e459) and `full/v6_build_collect.py` (a1e486d6), record-assembly scripts whose outputs
were all replaced (the final copy was made by copy_v6 1.1, the collection by v6_build_collect 1.3; both are here).

Versions recorded in every anatomy's provenance (`BUILD_v6.json` -> `toolchain`): build_v6 1.2, compose_v6 3.2.0,
graft_v6_va 1.4.2, bake_v6 1.1, checks_v6 parts 1.3 with route_d22 1.4 (222) / 1.5 (1); on the 17 anatomies s3_fix
changed: bake_v6 1.2 (the 8 re-baked), checks_v6 1.6 (every part on the 9 whose geometry changed; coverage and folds on
the 8 re-stamped for a flag).

**The donor-side tools (`donor/`, `gates/`, `item_b/`) were not fingerprinted when they ran** (stage 1 and the stage-2
hardening, 2026-10-03/04). The copies are the files in the scratchpad at copy time; each tool's version string and
frozen predecessors (`local_sub_v1_0.py`) are as its notes describe (`_records/notes/v6_wall.md`, `s2_harden_wall_bake.md`,
`s2_harden_va.md`). What the build read from them is pinned by md5 in each anatomy's provenance
(`xform.vaC_*.surface_md5`, the junction and D17 records' paths).

## Stages and the runs that used them

Runs (build log `_records/notes/V5_BUILD_LOG.md`, which carries the v6 entries from 2026-10-03 17:25):

- **audit**: TopBrain piece orientation audit (wf_05a3a826-307, 2026-10-03), report `reports/TopBrain piece orientation audit (v5).md`
- **stage 1**: tools and the 14-anatomy pilot (wf_cdc27985-4f6, 2026-10-03 18:05 -- 10-04 07:40)
- **harden**: stage-2 hardening -- carotid, VA, wall + bake (wf_87bbef23-7e4, 2026-10-04)
- **integrate**: joint pass, re-pilot, DRY on all 223 (2026-10-06 / 07; gate verifiers verify_s2_gate, verify_s2_gate2)
- **build**: the full build, bake, checks, copy into anatomies_v6 (s2_build, 2026-10-07)
- **checks**: the set-level checks on anatomies_v6 (s2_checks, 2026-10-07)
- **records**: BUILD_v6.json, _records, this copy, CAROTID_V6.md (s2_records, 2026-10-07)
- **verify**: the final verifier (verify_v6_final, 2026-10-07): NO-GO on lost real wall (8 pieces), records, one fold-over
- **s3_fix**: the must-fix list under D27 (2026-10-08): coverage check, 8 re-bakes, 8 flags, the fold fix, records 1.1

| stage folder | group | used by |
|---|---|---|
| build | `build/build_v6.py`, `precheck_orient.py`, `route13_table_v6.py`, `eca_c3_feedback.json` | integrate (DRY, re-pilot), build (compose stages va1, carwrite, carverify, va2, recheck, fix, route13, final, finalcheck, split, summary) |
| build | `carotid/` compose_v6 (+ 3.1.0), route_guard, route_baseline (D13), verify_s3 | harden, integrate, build |
| build | `va/` graft_v6_va (+ 1.3 / 1.4 / 1.4.1), va_corridor (D10), native_offsets (D10), route_guard, va_entry_geom, sofa_va_entry (D14) | harden, integrate, build, checks |
| build | `bake/bake_v6.py`, `wall/va_lumen_poly.py` | build (one bake), checks |
| build | `checks/` checks_v6 (+ 1.3 / 1.4), geom_v6, loader_one; `gates/orient_gates.py`; `sofa/sofa_rcca_v6.py` | build, checks |
| build | `figs/` the pilot / re-pilot figure scripts (in the manifests; the v6 figures are `monitoring/figure_carotid_anatomies_v6.py`) | integrate |
| build | `full/` runners (`dr_full.sh`, `chain_compose*.sh`, `chain_bake*.sh`, `chain_check4.sh`, `chain_check5.sh`), bake reuse (`cmp_build2*.py`, `cmp_build4_classify.py`, `merge_tree2/3.py`, `verify_tree3.py`), scans (`scan_tube_genus.py`, `scan_self.py`), assembly (`check4_helper.py`, `stamp_prov.py`, `gen_license.py`, `copy_v6.sh`, `load_v6_all.sh`, `v6_build_collect.py`) | build |
| donor | `wall/` ctseg6 / vmrseg6 (CT-edge wall with the V6_RULE switch), donors.json / make_donors.py, seg / vmtk batches, build_donors6, gate_ct, wd_proxy, floor / bone scans, c4_check, `jfix/` junction6 + sub_batch2 + vmtk_sub3 | stage 1 (v6_wall), harden |
| donor | `wall_local/` local_seg, splice_lines, vmtk_local_fix, build_donors_local (D15 records), local_sub (junction, D16 tube), junction_records, wall detail, D15 report, `chainA.sh`, `chainB.sh` | harden (D15) |
| donor | `wall_d17/` paths_d17, make_req_d17, sub_batch_d17, junction_records_d17, wd_sub_d17, d17_report, sub_ct_gate, genus_cmp, publish_d17, `chainD17.sh` | harden (D16 / D17 run 2) |
| donor | `va/` subclavian_paths (stage-1 path), sc_post / sc_post3 (junction records), paths_d17 (the VA tool's D17 paths) | stage 1 (v6_va), harden |
| gates | `heading/` heading_evidence, zen_frames, host_tree, c6check, zenodo_frames.json; `gates/test_api.py` | stage 1 (v6_plan) |
| item_b | `pilot_build/route_diag_v6.py` (+ bake_v6p, `va/bake_sections_v6.py`) -> `v6_work/pilot/route13/diag/<name>.rcca_v4.npz` | stage 1 (pilot), read by build's route13 stage |
| set_checks | `s2checks/` collect_s2c, orient_set, loader_set, ident_check, diag_cov 1.2, bif_angles, make_items, sofa_worker / sofa_container, dr_s2c | checks |
| records | `records/` build_v6_record, check_build_record, copy_records, make_tools_v6, dr_rec | records, s3_fix (1.1) |
| build | `bake/bake_v6.py` 1.2, `checks/checks_v6.py` 1.6 (+ `bake_v6_v1_1.py`, `checks_v6_v1_5.py`) | s3_fix |
| s3_fix | `s3fix/` diagnoses, runners, fold_fix, stamp_prov_s3, copy_v6_s3, copy_records_s3, cov_results, sofa_summary, install_s3fix_tools | s3_fix |
| v5_deps | `v5_va/` v5_split, graft_v5_reg / _reg_sep / _va / _reg_donor, bake_v5 / v5b, chk_one, vvw/measure, vvw/rrd_v4, swap13/rrdx, route13/diag13, sources, va_metrics, vmr_common, vmr_vmtk, load_test_v5, tbo/va_narrow/{bone_scan,ct_check}; `v4_build/audit/topo_audit.py` | imported by the v6 tools |

## Re-running

1. **Restore the layout.** `bash carotid_tools/v6/restore_layout.sh <X>`. `<X>` plays the scratchpad and is mounted as
   `/sp`. Copy the results the scripts read into `<X>/v6/results/` and `<X>/v5_va/results/` (`split_v4.json`,
   `split_v5_final.json`: `carotid_data/anatomies_v5/_records/results/`).
2. **Mounts** (`build/full/dr_full.sh` shows the full list): the repo at `/repo` (read-only), `<X>` at `/sp`,
   `carotid_data/v6_work/build` at `/bld`, `carotid_data/v6_work/full` at `/full`, `<repo>/carotid_data` at
   `/opt/eve_training/carotid_data` and `<repo>/topbrain_tools` at `/opt/eve_training/topbrain_tools` (read-only), as
   for v4 / v5. The donor-side tools mount `carotid_data/v6_work/wall` at `/v6w` (`donor/wall/dr.sh`). Note: a folder
   mounted inside a read-only parent mount comes up empty on this host; mount from a backup root as the runners do.
3. **Images.** eve-training-fixed (compose, VA tool, loader, SOFA, split), neve-build-meshlab (bake, checks), neve-build-stagea
   (donor walls, wall detail), neve-build-vmtk (VMTK centrelines), as in CAROTID_V5.md.
4. **Repo modules imported** (tracked, not copied; their md5 in the manifests matches the repo, `TOOLCHAINS.tsv`):
   `carotid_tools/` graft_v4, graft_three, analyze_bifurcations, extend_ica; `topbrain_tools/` graft_siphon, sdf_mesher,
   sdf_union, sdf_union_v4, bake_meshes_v4; `eve`, `eve_bench`.
5. **Data not in the repo** (git-ignored): `carotid_data/v6_work/` -- `wall/donors_local/` (D15 donor VA records and
   surfaces), `wall/donors_local/_junction_records/` and `wall/d17/` (item e, D16, D17), `va_s2/paths_d17/`,
   `va_s2/native_offsets/`, `wall/c4/c4_tube_override.json`, `wall/seg/local/`, `build/dry223b/` (the DRY), `full/`
   (the build: trees, bakes, checks, the undecimated unions `full/iso/`); and v5's `carotid_data/va_sources/`.

**Hard-coded paths a re-run must change.** The host scratchpad path
`C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad` is written into
12 host-side files, as the `-v <scratchpad>:/sp` mount: `build/build/dr_i2.sh`, `build/full/dr_full.sh`,
`donor/wall/dr.sh`, `donor/wall/chain2.sh` (2), `donor/wall/chain3.sh` (2), `donor/wall_d17/chainD17.sh`,
`donor/wall_local/chainA.sh`, `donor/wall_local/chainB.sh`, `set_checks/s2checks/dr_s2c.sh`, `records/records/dr_rec.sh`,
`records/records/copy_records.sh`, `records/records/make_tools_v6.py` (the header text only); counted on the restored
files. The host repo path `D:\Arjun\neve` (`/d/Arjun/neve`) appears in 15 scripts, as the `-v <repo>:/repo` mount and
host-side paths: the 11 above other than `make_tools_v6.py`, plus `build/full/copy_v6.sh`, `build/full/copy_v6_v1_0.sh`,
`donor/wall/jfix/publish_j3.sh` and `v5_deps/v5_va/sources.py` (and in a comment of `TOOLCHAIN_s2i2.md5`).
Container paths (`/sp/...`, `/repo/...`, `/full/...`) need no change when the mounts are kept. Docker is called with
`/c/Program Files/Docker/Docker/resources/bin` on PATH and `MSYS_NO_PATHCONV=1` (Git Bash).

**The build order, as it ran** (exact arguments in each run's notes under `_records/notes/`):

    # donor-side inputs (harden; neve-build-stagea / vmtk, Git Bash on the host)
    bash wall_local/chainA.sh; bash wall_local/chainB.sh             # D15 local walls + records, local junctions
    bash wall_d17/chainD17.sh                                         # D16 / D17 junction records, published by publish_d17.py
    python3 va/paths_d17.py; python3 va/native_offsets.py             # D17 paths, D10 native offsets (eve-training-fixed)
    # the DRY on all 223 (integrate2): one frozen toolchain bedf172b
    bash build/chain_dry2.sh A|B|C                                     # build_v6.py <stage> --root /bld/dry223b ...
    # the full build (s2_build), toolchain a80644e0 for the compose
    bash full/chain_compose2.sh A|B|C|D                                # build_v6.py <stage> --root /full/build2 --plan /bld/dry223b/car_dry --manifest ...
    bash full/chain_bake.sh <wid>                                      # bake_v6 1.1 + checks_v6 (first build; chain_bake2.sh: the 7 re-bakes)
    python3 full/cmp_build2.py; python3 full/cmp_build4_classify.py; python3 full/merge_tree3.py; python3 full/verify_tree3.py
    bash full/chain_check4.sh <wid>                                    # checks_v6 1.4 route_d22 on all 223 (D26), toolchain 88ab481d
    bash full/chain_check5.sh <wid> <names>                            # checks_v6 1.5 route_d22 on the failing one (D27), toolchain d65b7209
    python3 full/gen_license.py /full/tree3 /full/license; python3 full/stamp_prov.py
    DST=.../v6_work/full/anatomies_v6_staged bash full/copy_v6.sh; bash full/copy_v6.sh    # staged copy, then anatomies_v6
    bash full/load_v6_all.sh <k> <K>; python3 full/v6_build_collect.py --checks-dir /full/checks4 ...
    # set checks (s2_checks) and records (s2_records)
    python3 /sp/v5_va/v5_split.py /repo/carotid_data/anatomies_v6 --baseline /sp/v5_va/results/split_v4.json
    bash s2checks/sofa_worker.sh ...; python3 s2checks/orient_set.py ...; python3 s2checks/collect_s2c.py
    python3 records/build_v6_record.py; python3 records/check_build_record.py

## Every file

`FILES.tsv` has the hashes; the note after a name says why it is here (the manifest that first lists it, or the file
that imports it).
