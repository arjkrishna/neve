# s2_records -- the v6 records: BUILD_v6.json, _records, carotid_tools/v6, CAROTID_V6.md

Agent s2_records, 2026-10-07. State: `state/s2_records.md`. Code: `scratchpad/v6/records/` (runner `dr_rec.sh`, containers
`s2r_<pid>`, foreground only; /repo, /full and /s2c mounted read-only, writes into the repo by host-side cp). No git
writes. Repo-tracked locations written: `carotid_data/anatomies_v6/BUILD_v6.json`, `carotid_data/anatomies_v6/_records/`,
`carotid_tools/v6/`, `CAROTID_V6.md`. Nothing else touched (section 6).

## Answer so far

**Complete.** All four deliverables are in place and checked.

1. `carotid_data/anatomies_v6/BUILD_v6.json` (build_v6_record 1.0) is generated from the 223 final `provenance.json`
   (+ `mesh_v6.json`, LICENSE files) only, with a sha256 digest of the 223 provenance files (1fce35ed...). The one
   block not derivable per anatomy (`checks_final`: split, folder loader, SOFA, D14, orientation re-measure) is copied
   from `_records/results/v6_final_summary.json` and labelled. `check_build_record.py` recounts it from provenance with
   separate code: **98 checks, 0 differences** (`_records/checks/build_record_check.json`). Against the build's own
   collection (`results/v6_build.json`) every comparable count is equal except one explained boundary case
   (w_022_right mr_026 RVA va_target 19.999 mm: band "<=20" but flagged va_target_20_30).
2. `carotid_data/anatomies_v6/_records/`: 1,154 byte copies (41 MB: results, set checks, gates, per-anatomy checks
   records, SOFA, diagnoses, all notes incl. the audit's) + README.md + checks/build_record_check.json + notes/s2_records.md
   + RECORDS.sha256. Copy script `records/copy_records.sh` (idempotent, never overwrites a differing file).
3. `carotid_tools/v6/`: 176 copied files (build 69, donor 59, v5_deps 20, set_checks 10, records 9, gates 6, item_b 3)
   + MANIFEST.md, FILES.tsv (sha256), TOOLCHAINS.tsv (all 368 entries of the 7 manifests resolved: 300 current files, 12
   frozen copies, 54 repo modules md5-equal, 2 not preserved), restore_layout.sh. Restore test: 176 / 176 sha256-identical
   and byte-identical to the scratchpad; the final manifest d65b7209 passes md5sum -c on all 63 /sp entries of the
   restored layout (`records/out/restore_test.txt`). 16 of the 20 v5 deps are byte-identical to carotid_tools/v5's
   originals.
4. `CAROTID_V6.md`, laid out like CAROTID_V5.md: at a glance, the audit, the decisions (D1–D15 and D25 marked as the
   orchestrator's, D16–D24, D26, D27 the user's), items, sources / licences, composition (forks with the Zenodo frame
   table, left forks, siphons / ECA tips, VA, w_018b, donor wall), the bake, every gate with its result and source, the
   right route (D22 / D26 / D27), orientation, wall detail, SOFA, split / loader / copy, the flag table, the build history,
   record corrections, limitations, follow-ups (donor colour first among the later items), records, reproduce, loading.
   Every cited `_records` path exists.

**Flagged for the orchestrator (not changed by this stage):**
- The 4 coverage pieces (w_052_left mr_026 fork 65.0 mm² / 5.0 %, w_027_left mr_022 and mr_026 forks 26.7 / 26.9 mm² /
  2.1 %, w_051_right mr_008_L siphon 24.1 mm² / 1.7 %): real scanned wall shown by v4 missing from v6, D22 passes (one
  direction only). Under D27's residual rule they would be above the 1 % line (-> v5 geometry for that piece); not a
  gate, so not applied here. Listed in CAROTID_V6.md (limitations, first follow-up).
- Record nit: 16 ia172_015 R sides carry `v6_build.va.<side>.d17.tube_mm` 0.664 mm although no tube was written
  (`positions.tube_mm` 0, branch ends 29.3 mm past the ostium); not flagged d17_tube. Documented.

## 1. BUILD_v6.json
- Generator `records/build_v6_record.py` (eve-training-fixed; numpy for the left-fork map deviation from
  `xform.lower_L.R` and the frame recorded in `v6_carotid.left_fork.map.frame`). Sections: set, derived_from,
  provenance_digest, base (names equal v5, none dropped), anatomies, what_v6_changes, decisions (D1–D27 with who took
  each), toolchain (versions and manifest md5s as recorded in provenance: compose a80644e0 x223, checks d65b7209 x223),
  build_images, sources, donors (25 units, one-to-one ok, per-unit table with rung counts and licence), licences,
  carotid (right fork rungs / poses / lists, left fork decisions / substitutes / map deviation / not-on-map, ICA AP,
  ECA, D13, route13), va (rungs, junction types, flags, va_target bands, D21 / D18 / D9-rung-3 / D16 / D17 lists, ranges),
  bake (rungs, margins, windows, C4, E7, tube-only, far clamp, triangles, lumen), checks (gates, D22 references, values,
  D26 / D27 / coverage lists), flags (as written), checks_final (from v6_final_summary.json), records, per_anatomy
  (compact row + provenance sha256).
- Checker `records/check_build_record.py`: dotted-path recounts, list membership, the digest, the donor table per unit,
  licence files vs provenance, mesh_v6 vs bake_v6 rung. Run on the installed copy.
- Numbers (from BUILD_v6.json): right fork R1 124 / R2 88 / R3 3 / R4 3 / R5 5; left kept 201 / substituted 22 (anchor_low
  18, no passing siphon pose 4); left on map 215 (29 bounded dev 5–43°), not on map 8; VA rung R 198/12/1 + 12 host,
  L 215/3 + 5 host; own junction 413; take-off |error| median 0.08°, max 0.73°; D17 cut end 29.2–108.2 mm past the
  ostium; va_target bands <=20 435, 20-30 10, >30 1; bake rung 1 125 / 2 98; C4 8; E7 ok 223; checks pass 223.

## 2. _records
- `records/copy_records.sh`; layout and contents in `_records/README.md`. Not copied (large, cited by path in the work
  folder): full/diag/c3test, r5test (18 MB trees each), car32*, va14*, sofa logs and claim dirs, the unions full/iso.

## 3. carotid_tools/v6
- `records/make_tools_v6.py` (selection by manifests + explicit donor / gates / item_b / set_checks / records lists +
  import / script closure), `records/finalize_tools.sh` (stage, MANIFEST = MANIFEST_head.md + per-file list, restore test,
  install), `records/imports_check.py` (every import of every copied .py resolves: stdlib, third party, repo, or a copied
  file -> {} unresolved).
- Not preserved: the first build's `full/copy_v6.sh` (md5 2042e459) and `full/v6_build_collect.py` (a1e486d6), listed in
  the supplement manifest; superseded record-assembly versions (stated in MANIFEST.md).
- Hard-coded host paths: the scratchpad path in 12 restored files, the repo path in 15 scripts (listed in MANIFEST.md).

## 4. CAROTID_V6.md
- Facts checked against BUILD_v6.json, v6_final_summary.json, v6_checks.json, the build / checks notes and provenance
  (scripts `records/explore*.py`, `peek*.py`): substitute list and reasons, rung-2 reasons (top gates 6, arm direction
  1, droop 2, final side gates 2 = ia172_069 L, demoted 4), ECA skip reasons (R: 16 D4, 3 route guard, 1 C3 feedback;
  L: 19 D4, 1 no pose, 16 on substitutes), tube-only sections and the far clamp inherited from v5 / v4, vaC surface md5
  on 429 / 429, w_018b D22 1.0 / 1.0.

## 5. Decisions
- BUILD_v6.json counts come only from per-anatomy files; set-level results are a labelled copy (they cannot come from
  provenance). The orient:* counts of the build collection are not provenance fields; BUILD_v6 counts the VA flags
  instead (equal except the 19.999 mm boundary).
- carotid_tools/v6 also copies the upstream donor-side tools (not fingerprinted) and the frozen earlier versions that
  made records still in the tree, so the restored layout covers every manifest entry that still exists.
- v5 deps are copied again (byte-identical to carotid_tools/v5's originals where present) so the v6 restore is
  self-contained.
- The coverage residuals are reported, not acted on: the records stage does not change anatomies.

## 6. Read-only check (2026-10-07 ~21:30 EDT)
- `git status --porcelain` on anatomies_v2..v5, topbrain_tools, carotid_tools, eve, eve_bench, user.md, user copy.md,
  CAROTID_V5.md: only `?? carotid_tools/v6/` (this stage) plus ` M CAROTID_V5.md` and ` M "user copy.md"`, both modified
  before this stage started (mtimes 2026-10-03 19:30 and 2026-10-07 00:33; the session's starting git status already
  listed them). Nothing else. No file under anatomies_v2..v5, topbrain_tools or carotid_tools/v5 is newer than the
  build's final copy record (`v6_work/full/copy_verify_final.json`, 17:30).
- anatomies_v6: `sha256sum -c v6_work/full/anatomies_v6.sha256` exit 0; 223 case folders, 5,136 files outside
  `_records/` and `BUILD_v6.json` = the copy manifest (`records/out/v6_tree_sha256_check.txt`). This stage added only
  `BUILD_v6.json` and `_records/`.
- No containers left running (all `s2r_*` were foreground `--rm`).
