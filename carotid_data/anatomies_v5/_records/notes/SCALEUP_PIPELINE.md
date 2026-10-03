# SCALEUP_PIPELINE.md: pipeline v2 for the POST stage of the v5 VA scale-up lanes

Written by `scaleup_fixes` (state `/sp/v5_va/state/scaleup_fixes.md`, notes `/sp/v5_va/notes/scaleup_fixes.md`).
**READY for the lanes as of 2026-10-02 06:45.** Every v2 stage has run on the pilot cases: qc, select and rescue on
all 22, wall + grade on the rescued cases first. The lanes can rely on it.

**UPDATE 08:45 -- one extra step after post2.** Run `scaleup_ctseg_fallback.sh` on every case (section 2, step 2),
including cases already post-processed. On Song 138 R the pinch fix produced a wall whose VMTK centreline was 0.7 mm, so
nothing was measured. The tally now counts a wall as valid only if it was measured. `/sp` = the scratchpad
`C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad`
(`/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad` in Git Bash).

Pipeline v2 = the thin-slice pilot's pipeline plus three changes:
1. **Trace selection for both sources** (`scaleup_select.py`): the Song pilot's best-of-two rule over tsba / ts. The
   172-series now also grades the ts trace (qc_ts.json). On the pilot it picks tsba for every 172 side, so the result
   matches the pilot there.
2. **Fix 1, the V1 mask-break rescue** (`scaleup_v1rescue.py`): a second selection level.
3. **Fix 2, the ct_seg4 C2-exit pinch fix** (`scaleup_ct_seg.py`, one script for both sources).

Everything else is the pilot's, unchanged: the QC gate `pilot_proc_<src>_qc2.py` (recalibrated, rule 6 review-only),
landmarks, measure, edge, VMTK and protrusion.

## 1. Before the post stage: unchanged, run by the lane's own driver

| | Song (`scaleup_song_run.sh`) | 172-series (`scaleup_ia172_run.sh`) |
|---|---|---|
| fetch | fetch group (by-eye review; reject -> `rejected_<case>`) | pre-fetch arch check (lowest slice shows two aortic limbs) -> fetch -> by-eye review |
| TS | ts group: TotalSegmentator total_v3 `--ml` (GPU) + CPU split on `ct_negzero-100`; make1p5 -> `totalseg_1p5` | tsgpu / tscpu (same, on the CTA); make1p5 |
| Arterial | art group: input **`ct_negzero_air1000`** (the shipped 0-HU-clipped CT loses the neck), native, watchdog; the re-windowed copy and `cta.nii.gz` are deleted after | gpu group: CT clipped to **[-1024, 3071]** (`ct_clip.nii.gz`, deleted after), native, watchdog |
| trace | trace group: extract **tsba + ts** + VMTK on both | post group: extract **tsba + ts** + VMTK on both |

GPU lock `/sp/v5_va/.gpu_lock` as before. Before the post stage a case needs all of these:
- `arterial/.arterial_done`
- `arterial/runs/native/extracranial_vessels/segmentation.nii.gz`
- `arterial/{tsba,ts}/va_extract.json` and their VMTK outputs
- `totalseg_1p5/`

## 2. The post stage: one command per case or batch (Git Bash on the host)

```bash
V2=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/scaleup_v2_post.sh
TH=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice
# Song lane:
nohup bash $V2 song post2 AHMU1218111 [AHMU... ...] >> $TH/song/_logs_scaleup/v2_post.out 2>&1 &
# 172-series lane (V2_CPU_WANT keeps the lane's D3 priority: no post container starts while a fetch / tscpu stage waits):
V2_CPU_WANT=$TH/ia172/_logs_scaleup/.cpu_want_hi nohup bash $V2 ia172 post2 ia172_062 [ia172_... ...] \
  >> $TH/ia172/_logs_scaleup/v2_post.out 2>&1 &
# STEP 2, after post2 finishes, with the SAME environment (same V2_CPU_WANT for ia172):
FB=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/scaleup_ctseg_fallback.sh
bash $FB song AHMU1218111 [AHMU... ...] >> $TH/song/_logs_scaleup/v2_post.out 2>&1
```

What step 2 does (`scaleup_ctseg_fallback.sh`):
- **Check (~5 s per case).** For each VA, the VMTK centreline of the new wall must span at least 50 % of the label's
  s-range.
- **Act, only when a VA fails AND its pinch fix triggered.** That VA is re-cast with the pilot method (no pinch fix):
  - `ct_seg/pinch_skip_ct_seg4.json` lists the VA; `scaleup_ct_seg.py` block v2-c reads it.
  - The failed wall is kept as `ct_seg/ct_seg4_pinchfail`.
  - ctseg -> measure -> protrusion -> fig -> collect -> grade are redone.
  - The outcome goes to `ct_seg/pinch_fallback_ct_seg4.json`.
- **Tested on Song 138 in a private tag:** the R centreline went from 0.7 mm to 131.9 mm. The wall is measured but
  truncated at 131 of 206 mm (partial score).
- **A VA that fails without pinch events** is only reported (`NOPINCH`). It is not a pinch problem, and the tally marks
  that wall unmeasured.

The defaults are the lane names, so set nothing else:
- logs `<derived>/<src>/_logs_scaleup/<case>_v2_<stage>.log`, plus `timing.jsonl` with `"by": "scaleup_v2_post"`
- CPU lock `<derived>/<src>/_logs_scaleup/.cpu_lock`, the lane's own lock (owner line `v2post <pid> <epoch> <what>`).
  At most one heavy container per agent: 6 CPUs / 12 GB each, and each waits for 6 GB or more of MemAvailable in the WSL VM.
- wall `ct_seg/ct_seg4`, selection `qc/select.json`, gross `qc/gross.json`, figures `qc/.fig_v2`
- results `/sp/v5_va/results/scaleup_<src>.json`, grade `results/scaleup_<src>_grade.json/.txt`

`post2` runs these stages in order. Every stage is idempotent, so to resume after a cut-off you rerun the same command.
`V2_FORCE="<stages>"` redoes the named stages.

| stage | what runs | output | time (pilot) |
|---|---|---|---|
| qc | `pilot_proc_<src>_qc2.py` UNCHANGED on tsba and on ts (ia172: `PILOT_TRACE_DIR=ts`; song: `PILOT_EXT=ts`) | `qc/qc.json`, `qc/qc_ts.json`, `qc/qc_<case>.png` | 1-6 min (2 traces) |
| select | `scaleup_select.py` level 1: rank (passes as donor, whole-VA pass, is tsba); tsba wins ties | `qc/select.json` | 5 s |
| rescue | **fix 1**: `scaleup_v1rescue.py`. Only when a side is rescued: patched mask `arterial/runs/native_v1r` -> re-trace with the base set's own tool (tsba `pilot_art_extract.py`, ts `art_extract.py`) -> `arterial/<e>_v1r` -> VMTK -> the unchanged QC -> select level 2 | `qc/v1rescue.json`, `qc/v1rescue_<case>.png`, `qc/qc_<e>_v1r.json`, `qc/select.json` | 20 s; 8-20 min if rescued |
| gross | `scaleup_v2_gross.py` (pilot gross on the selected trace) | `qc/gross.json` | 10 s |
| ctseg | **fix 2**: `scaleup_ct_seg.py` (`VMR_OPEN_BRIGHT=1`, `PILOT_CT=shipped`, `PILOT_KEEP_CONNECTED=1`), seeded from the selected traces, + `vmr_vmtk.py` MISR | `ct_seg/ct_seg4/` (`seg_qc.json` "pinch_fix", "pipeline") | 1-4 min |
| measure | `pilot_proc_<src>_{landmarks,measure,edge}.py` UNCHANGED | `ct_seg/ct_seg4/{landmarks,measure,edge}.json` | 3-10 min (CPU contention) |
| protrusion | `scaleup_v2_protrusion.py` (pilot protrusion) | `ct_seg/ct_seg4/protrusion.json` | 20 s |
| fig | `scaleup_v2_fig.py` (pilot figures, selected traces) | `qc/*_detail_along_va.png`, `qc/*_wall3d_*.png` | 40 s |
| collect | `scaleup_v2_collect.py` (pilot record + v2 fields: `trace_set`, `v1_rescue_used`, `v1_rescue`, `wall.pinch_fix`) | `results/scaleup_<src>.json` (case record replaced) | 5 s |
| grade | `scaleup_v2_grade.py`, once after the case loop | `results/scaleup_<src>_grade.json/.txt` (+ `grade_v5` merged) | 10 s |

A case takes about 10-20 min, or 25-35 min with a rescue (measured on the pilot re-runs, sharing the CPUs with the lanes). If no side passes as a donor (`select.json` `any_donor` false),
ctseg and everything after it are skipped, but the case is still collected as a failing record.

**Do not run the lane drivers' own pilot-style grade stages** (`scaleup_song_run.sh` qc/gross/ctseg/measure/protrusion/
fig/collect, the "grade" group). If they ever ran on a case, redo it with
`V2_FORCE="qc select rescue gross ctseg measure protrusion fig" bash $V2 <src> post2 <case>`.

## 3. The two fixes: rules, so a lane can tell whether a fix acted on its case

**Fix 1, the V1 mask-break rescue** (`scaleup_v1rescue.py`; header holds the full rule).

A side qualifies when all three hold:
1. Its selected trace has V1 bridges, i.e. art_extract bridged gaps in Arterial's mask between the origin and the
   Erec C6 level.
2. Every one of those bridges runs through enhanced lumen: native HU median along the bridge >= 0.6 x the CCA median,
   the same bar as gate rule 5b.
3. Its gate failures are bridge failures only (subset of 1_origin_le_2mm, 3a_no_V1_bridge, 3d_total_le_25), or it
   passes only through the provisional V1 <= 3 mm limit.

The rescue itself:
- Grow the mask inside a capsule along the bridge span. Capsule radius = clip(1.6 r_VA + 0.5, 2.5, 4) mm. The grown
  voxels must have HU in [0.5, 1.5] x CCA, or already be mask.
- One 26-connected component must join the mask 0.5-3 mm below the span to the mask 0.5-3 mm above it.
- The added voxels need HU median >= 0.6 x CCA.
- The cross-section along the span needs r_eq >= max(0.5 mm, 0.35 r_VA).

A bridge through fat or soft tissue never qualifies. Neither does a gate failure other than a bridge failure. The
rescued trace replaces the level-1 trace only if it passes as a donor and the level-1 trace fails, or passes only
provisionally.

Record: `qc/v1rescue.json` and `select.json` `sides.<S>.v1_rescue` (used / why). The tally flags every rescued side.
- **By eye, for every rescued side:** open `qc/v1rescue_<case>.png`. The red added voxels must continue the VA lumen
  (cyan mask) across the break. If they are a vein, the SCA wall or a neighbouring artery, reject the side with a by-eye
  rejection (section 4). The tally's "V1 mask-break rescue" flag lists every rescued side.

**Fix 2, the C2-exit pinch fix** (`scaleup_ct_seg.py`, block "v2 pinch"; on by default, `SCALEUP_PINCH_FIX=0` turns it
off).
- After each ray cast, a station is flagged when its centre is not lumen: centre HU < the station's median T_loc, OR at
  least 50 % of its rays end at the 0.3 R start floor.
- A flagged station moves in its own normal plane to the deepest point of the lumen-bright region connected to the
  nearest lumen point, at most min(R + 1, 3) mm away. Then the vessel is cast again.
- If no station is flagged, the wall is bit-identical to the pilot's.
- Record: `ct_seg/ct_seg4/seg_qc.json` "pinch_fix" (events per vessel and pass: flagged s-runs, moved, max move) and
  `wall.pinch_fix` in the results.
- **Known limits:**
  - When stations stay flagged after the re-cast (`flagged_after_recast` > 0), the wall can still be truncated. Song
    138 R still stopped at 154.5 of 209 mm.
  - The extended wall can defeat VMTK. That case is handled by the fallback (step 2).
  - Per-vessel fallback: `seg_qc.json` `pinch_fix.skipped_vessels_fallback`.

## 4. Eligibility, coarseness, tally and stop rule

The pair rule is the build's rule, logged 03:35:
- **Eligible:** both sides pass the recalibrated QC gate AND both have a valid ct_seg4 wall (gates V1-V6, acquisition
  voxel).
  - The tally also requires the wall to have been measured (whole-VA area > 0).
  - An unmeasured wall shows as "wall not measured ... -> run scaleup_ctseg_fallback.sh".
  - A V1 wall that wraps round a neighbouring dense vessel fails gate V4 (genus >= 1). Example: 172_061 L after its
    rescue. That pair stays ineligible.
- **Review:** a side that passes only through the provisional limits (V1 bridge <= 3 mm, the dropped V2-band IQR term)
  is flagged "review: provisional". So is a side with status `donor_review` (rule 6 label agreement, or the bridged
  flag), and a rescued side.
- **By eye, for every flagged side:** check `qc/qc_<case>.png` and `qc/<case>_detail_along_va*.png` before the pair is
  used. The pilot did this for its rule-6 sides: all six 172-series sides by eye, and Song 008 / 094 / 100 / 113 at
  overview level.
- **Recording a by-eye verdict:** add an entry to `/sp/v5_va/results/scaleup_eye_rejections.json`:
  `{"rejections": [{"source": "song", "case": "AHMU1218xxx", "side": "R", "why": "...", "by": "<prefix>", "when": "..."}]}`.
  - Read-modify-write the file; both lanes share it.
  - The tally counts a listed side as failing QC.
  - For a flagged side that was looked at and kept, add an entry with `"verdict": "kept"`. It is recorded, and the pair's
    `eye_check_pending_sides` clears.

**Tally**, after batch 1 (18 per lane) and then after every 4 new cases in either lane. It is light, about 30 s:

```bash
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
docker run --rm --name <lane>_tally_$$ --cpus 2 --memory 4g -e TZ=EST5EDT,M3.2.0,M11.1.0 -v "D:\Arjun\neve:/repo" \
  -v "$SPW:/sp" -w /repo eve-training-fixed python3 /sp/v5_va/scaleup_tally.py | grep -A40 "^ELIGIBLE"
```

The tally writes `/sp/v5_va/results/scaleup_tally.json`. It never writes a lane or pilot results file; it grades into
`results/scaleup_tally_work/`. It reads:
- pilot rows, overridden by the pilot cases re-run under v2 (`results/scaleup_fixes_cases_<src>.json`, wall `ct_seg4v2`)
- both lanes' `results/scaleup_<src>.json`
- VMR 0248

What to read in the output:
- `counts.eligible_pairs` and `counts.eligible_all_bins_gt_dense_host_both_sides` (all-bins donor-stretch union > 0.3 mm
  over 0.027, the dense host RVA mesh, on both sides)
- `stop_rule.met` and `stop_rule.next`
- per pair: `score_mean_contact_free` (ranking key; min side reported), `R_L_all_bins`, `flags`

**STOP RULE:** eligible >= 30 AND >= 25 eligible pairs with all-bins donor-stretch union > 0.027 on BOTH sides.
Otherwise continue in selection order (`results/pilot_next_cases.json`) and re-tally every 4 cases, up to 25 new cases
per lane. If a lane case duplicates a pilot case, the tally ignores it and lists it under `inputs[].ignored_duplicates`.

## 5. What pipeline v2 changed on the pilot (evidence)

Details are in `notes/scaleup_fixes.md` and `results/scaleup_fixes_before_after.json`.
- **Rescue applies to:** 172_061 L, Song 141 R and Song 044 R (failing), plus 172_001 R and Song 109 R (provisional).
  - Used: 061 L, 141 R, 109 R, 001 R.
  - Not used: 044 R. Its mask was re-joined, but the side still fails rule 1: the junction is 3.25 mm from the TS SA,
    which is origin placement, not a mask break.
  - Correctly refused: 172_041 R/L (177 / 128 HU), 073 L (-25 HU), Song 106 R and Song 090 R/L.
- **Pinch fix, on the 30 pilot walls with pilot seeds** (`results/scaleup_fixes_ctseg_compare.json`):
  - 19 walls did not trigger. 16 of them are bit-identical.
  - The other 3 (Song 008 R, 032 R, 100 R) are sides whose other side triggered. They differ by 2-7 voxels at the
    vertebrobasilar confluence (R/L label swaps; `results/scaleup_fixes_ctseg_offside.json`).
  - 11 walls triggered. Song 094 R now runs 0 -> 241.75 mm (the pilot cut it at 122.75 mm); the other 10 have Dice
    0.975-0.999 against the pilot. None of the 11 lost gate validity.
- **Pilot under v2, measured** (`results/scaleup_fixes_before_after.json`):
  - 13 eligible pairs, against 12 for the pilot. Song 141 was added; nothing was lost.
  - Song 109 R is no longer provisional.
  - Song 094 R now gets an exact donor-stretch score.
  - 172_061 stays ineligible: its L wall fails gate V4 (genus 1, V1 wrapped round a dense vessel).
  - Re-measuring identical labels moves whole-VA union by up to 0.0005 (run-to-run noise of the measure chain).
- **Lane cases checked for fix-2 harm (08:55):**
  - Song 138 R: the fix defeated VMTK. That is the fallback above.
  - 172_067 L: genus 3 with or without the fix.
  - Song 020 L: the fix extends a wall truncated at 135.5 mm to 229.5 mm, but the extended wall has genus 3. That pair
    is ineligible either way, because its R fails QC.
  - Song 111 L and 120 R: genus without any pinch event.
- **Pilot cases not re-measured** (172-series 039, 058, 140, 141, 190; Song 044, 113; the wall-less 172-series 041, 073
  and Song 090): no fix touched them, so v2 = pilot there. Their 16 walls are bit-identical, and the tally keeps their pilot rows.
