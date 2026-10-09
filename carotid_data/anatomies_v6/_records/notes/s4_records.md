# s4_records -- the second final verifier's record items on anatomies_v6 (record only, D27)

Agent s4_records, 2026-10-08. State: `state/s4_records.md`. Code: `scratchpad/v6/s4rec/` (runner `dr_s4.sh`, containers
`s4r_<pid>` one at a time, /repo read-only; host-side copies in Git Bash). Inputs: `notes/s3_fix.md`,
`notes/verify_v6_final2.md` and the verifier's outputs (`scratchpad/v6/verify_final2/out/`). **No geometry change**: the
only anatomy files written are 10 `provenance.json`. No git writes; user.md / "user copy.md" not touched; carotid_tools/v6
not touched (outside this step's write list).

## Answer (final)

**(a) Done.** The 10 pieces with a real-wall loss > 0 and < 0.1 % carry a D27 flag in provenance; BUILD_v6.json was
regenerated from provenance and recounts with 0 differences; the folder manifest, `_records` and RECORDS.sha256 are updated.
- Flags (`v6_checks.flags`, value = the verifier's `loss_max_union` x 100, 3 decimals) and a detail block
  `v6_checks.s4_records` (losses per reference v4 / v5 and wall union / shipped collision_full, lost area, the lost
  stretch and depth, checks_v6 1.6's value, a detail sentence) on 10 anatomies:

| anatomy | piece | flag value (union) | shipped full | lost mm2 | RCCA s | depth mm |
|---|---|---:|---:|---:|---|---|
| w_046_left mr_026 | fork | 0.097 % | 0.124 % | 1.22 | 121.5-121.75 | 0.34-0.38 |
| w_012_right mr_012 | siphon | 0.070 % | 0.115 % | 1.42 | 202.5-203.25 | 0.32-0.34 |
| m_024_left mr_023_L | fork | 0.057 % | 0.057 % | 0.82 | 112.25 | 0.41-0.59 |
| **w_034_left mr_008_L** | fork | 0.056 % | 0.056 % | 0.82 | **69.0-69.75** | **1.10-1.41** |
| w_042_left mr_010_L | siphon | 0.045 % | 0.045 % | 0.88 | 210.75-211.5 | 0.40-0.51 |
| m_024_left mr_001 | fork | 0.045 % | 0.045 % | 0.65 | 112.25-112.5 | 0.52-0.56 |
| w_051_right mr_025_L | fork | 0.015 % | 0.015 % | 0.19 | 90.5 | 0.31 |
| w_012_left mr_013_L | siphon | 0.013 % | 0.013 % | 0.23 | 221.25 | 0.31 |
| m_006_left mr_022_L | siphon | 0.008 % | 0.008 % | 0.18 | 222.25 | 0.34-0.36 |
| w_018b_left mr_002 | siphon | 0.005 % | 0.005 % | 0.10 | 148.5 | 0.32 |

  The values equal the task's list and checks_v6 1.6's own deficits for the same pieces (`s4rec/cov10.json`). w_034_left
  mr_008_L's flag detail says it is a real 0.82 mm2 patch at the carotid bulb (RCCA s 69.0-69.75, just proximal to the
  RECA take-off at s 71.0, 1.10-1.41 mm outside the union, both references); the other 9 say decimation-noise level.
  w_012_left mr_013_L already had its fork flag (s3_fix) and now has both. Flag names, e.g.
  `d27_coverage_fork_real_wall_loss_0.056pct_accepted`.
- All 18 pieces with a measurable loss are now flagged: 9 fork / 9 siphon on 17 anatomies.
- Two more pieces lose wall only against the shipped (decimated) collision_full, 0 against the union the rule measures:
  w_021_right mr_008_L siphon 0.036 %, w_052_left mr_027 fork 0.013 %. Not flagged (the rule and all 18 flags use the
  union); recorded in `coverage_sub01_pieces.json` and the doc.
- Provenance: `stamp_flag_s4.py` 1.0 asserted per file that the source round-trips byte for byte through json (indent=1)
  and that the stamped dict minus the flag and the `s4_records` block equals the source. Copied with `copy_prov_s4.sh`
  (each replaced file first checked against s3_fix's manifest; originals kept in `s4rec/replaced/<name>/`).
- Folder manifest: `_records/results/s4_records/anatomies_v6.sha256`, **sha256 6ad80caf...**; 223 anatomies, 5,136 files,
  `sha256sum -c` exit 0, file sets equal, exactly the 10 provenance lines differ from s3_fix's 024edcd2
  (`copy_verify_s4.json`). s3_fix's manifest stays in `results/s3_fix/` as the record of that state; the work copy in
  `v6_work/s3fix/` was not touched (outside the write list) and is now stale for these 10 lines.
- **BUILD_v6.json**: `build_v6_record` **1.2** (1.1 frozen as `build_v6_record_v1_1.py`) changes text only: the
  `what_v6_changes` s4 line, `records.undecimated_unions` (now names `v6_work/s3fix/iso/final/` for the 8 re-baked),
  `records.s4_records`, the flags note. Installed sha256 **cd833da2...** (s3_fix's b9e44e28 kept as
  `s4rec/replaced/BUILD_v6_s3.json`). Against it, exactly 38 leaves changed (`build_v6_diff_vs_s3.txt`): the 10 new flag
  counts, the 10 anatomies' per-anatomy flags and provenance hashes, the digest (32c9a554...), generator / generated, the
  text entries, and `s3_fix.changed` for w_012_left mr_013_L (its flag list is read from provenance, so it now also shows
  the s4 siphon flag). `check_build_record` **1.2** (text-only: version, the information note) on the installed copy:
  **123 checks, 0 differences**.
- `_records`: + `results/s4_records/` (coverage_sub01_pieces.json, verify_final2_coverage_all223.json, stamp_flag_s4.json,
  anatomies_v6.sha256, copy_verify_s4.json, build_v6_diff_vs_s3.txt, records_replaced.txt, scripts/ with the 1.2
  generator / checker), `notes/s4_records.md`, `notes/verify_v6_final2.md`; replaced `checks/build_record_check.json` and
  README.md (logged in `results/s4_records/records_replaced.txt`); RECORDS.sha256 rebuilt: **1,485 files** (1,466 + 19),
  `sha256sum -c` ok (`copy_records_s4.sh`; this notes file was copied again after this line was written). The other notes
  in `_records/notes/` are byte-equal to their scratchpad originals (checked).

**(b) Done.** CAROTID_V6.md corrected (s3_fix version kept as `s4rec/replaced/CAROTID_V6_s3.md`):
- SOFA RCCA not past the bifurcation: 5 after s3_fix (+ w_002_right mr_013_L); w_002's seed 777 went 9.9 mm into the ICA
  (RCCA s 73.2, mark 73.3), buckled and ended at RCCA s 51; its other two seeds end in the ECA (`sofa/rcca_v6_s3/`: final
  tips RECA s 8 / 0); fork pose and bifurcation lumen unchanged: controller sensitivity.
- Handle windows on the final folder 25 (fork) / 32 (siphon), from each `mesh_v6.json` `suppressed_stretches_mm`
  (own recount; 28 / 31 as first built).
- s3_fix re-bakes 174-645 s from provenance `bake_v6.seconds` (own recount: 174.1-644.8); the bake markers read 177-647 s.
- Undecimated unions: the 8 re-baked in `v6_work/s3fix/iso/final/` (md5 differ from `full/iso/`, which holds the
  superseded pre-fix unions; the other 9 files in `s3fix/iso/final/` are md5-identical to `full/iso/`); also in BUILD.
- Synthetic surface on the moved right route: the D27 capsule repair windows (17.79 + 12.01 + 56.57 mm2) besides the
  bridges (and the bake's tube windows, as in v4 / v5).
- 25-anatomy set: 3 of its anatomies re-stamped, provenance only (w_008_right mr_001, w_047_left mr_027 by s3_fix;
  **m_024_left mr_001 by s4_records**, one of the 10); geometry unchanged, SOFA numbers stand.
- Flagged coverage pieces 18 (8 at 0.1-1 % + 10 below 0.1 %): intro, at a glance, gate table, coverage paragraph,
  after-fix bullet, flag table (9 fork / 9 siphon, 17 anatomies), known limitations (all 18 listed, w_034 as a real patch).
- w_008_right mr_003's re-baked fork disclosed (right route section and limitations): 2.0 % of its D22-gated surface off
  v4 + own scan (39.7 mm2, 0.9798) = bake capsule 33.6 mm2 + next to the bridge tubes 7.1 mm2; pre-fix 1.0; passes D22
  only via the D27 repair window (0.997), flagged `d27_d22_fork_repair_window_56.57_mm2`.
- Also: build steps 10 (second verifier) and 11 (s4_records), a record-corrections bullet listing these stale items, the
  copy section (new manifest), the records / reproduce sections (s4 block, 1.2 tools in `_records`), the
  check_build_record version.
- Not changed: the three rounding-only items the verifier listed (108.25 -> 108.3, 120.25 -> 120.3, 1.865 -> 1.87).

## Decisions (D27: minimal option, flagged)

- Flag value = the verifier's `loss_max_union` (the union is the rule's wall and the 8 existing flags' wall), printed
  with 3 decimals so that values below 0.1 % do not round to 0.00-0.10 as the 8 s3_fix flags' 2-decimal format would.
- New provenance key `v6_checks.s4_records` (not `s3_fix`), so BUILD's s3_fix counts stay s3_fix's and the checker's
  per-anatomy derived-flag checks stay on the 17 s3_fix anatomies (123 checks as before).
- Generator / checker changed by text only (1.2), so BUILD_v6.json is still generated by the generator (no hand edit).
  The 1.2 scripts and this step's scripts go to `_records/results/s4_records/scripts/`; carotid_tools/v6 keeps 1.1
  (re-staging it is outside this step's write list).
- New folder manifest in `_records/results/s4_records/` rather than overwriting s3_fix's record.

## Open issues

1. `carotid_tools/v6/` (MANIFEST.md, records/records/) does not yet hold build_v6_record / check_build_record 1.2 or the
   s4 scripts; they are in `_records/results/s4_records/scripts/` (the doc says so). A later tools step can re-stage.
2. `v6_work/s3fix/anatomies_v6.sha256` (024edcd2) no longer matches the 10 re-stamped provenance files; the current
   manifest is `_records/results/s4_records/anatomies_v6.sha256`.
3. Judgment item carried from the verifier: w_008_right mr_003's fork passes D22 only through the D27 repair window
   (disclosed and flagged).
