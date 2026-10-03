# scaleup_fixes: the two pre-bulk fixes (V1 mask-break rescue, ct_seg4 C2-exit pinch), the tally, pipeline v2

Agent prefix `scaleup_fixes`. The first instance ran from 03:30 EDT 2026-10-02; the second took over at 06:16, after the
04:57 session limit. State file: `/sp/v5_va/state/scaleup_fixes.md`.
- No git, and no tracked repo file touched.
- No pilot file overwritten. The pilot re-runs write new names: `ct_seg/ct_seg4v2`, `qc/select_v2.json`,
  `qc/gross_v2.json`, `*_v2.png`, `results/scaleup_fixes_cases_<src>.json`.
- Licences: Song 2024 and the 172-series are CC BY 4.0. Everything derived through Arterial (masks, traces, the rescued
  masks, and the walls seeded from them) is non-commercial.

## Answer (final, 09:00)

**Tally** (`results/scaleup_tally.json`; pilot cases under pipeline v2 plus VMR 0248):
- 13 eligible pairs: 172-series 5, Song 7, VMR 1. The pilot baseline was 12.
- All 13 have an all-bins donor-stretch union > 0.027 on both sides; 8 do on contact-free bins.
- Stop rule NOT met: short by 17 eligible pairs and 12 coarse pairs. The lanes have just started batch 1 and have no
  eligible pair yet.

**What changed against the pilot:**
- Song 141 is a new pair, via the V1 rescue of its R side. It ranks last (contact-free score 0.0108, min side 0.0021),
  and its all-bins R score is 0.0331, just above 0.027.
- Song 109 R is no longer provisional (contact-free 0.0232).
- Song 094 R now has a full-length wall: 0-241.75 mm, against 0-122.75 mm in the pilot. Its donor-stretch score is now
  exact (all-bins 0.1542; contact-free 0.0606).
- No pair was lost.

**172_061 is still not a pair.** The rescue made its L side pass QC, but the L wall fails gate V4 (genus 1). The handle
sits in origin -> C6, where the wall wraps round a dense neighbouring vessel (bins 10-40 mm have union 0.74-0.81).
That matches the by-eye look: the gap is a diffuse ~300 HU zone next to a >650 HU vessel.

**Fix 2 problem found on a lane case, and fixed.** On Song 138 R the pinch fix extended the label from 131 to 154.5 mm,
but VMTK returned a 0.7 mm centreline, so nothing was measured even though the gates read "pass". What I added:
- a per-vessel fallback to the pilot method (`scaleup_ct_seg.py` v2-c + `scaleup_ctseg_fallback.sh`). Tested on 138 in
  a private tag: the centreline went from 0.7 to 131.9 mm.
- a tally rule: a wall must have been measured to count.

**Fix 2 safety on the lane cases' genus failures:**
- 067 L: genus 3 with and without the fix.
- 111 L and 120 R: no pinch events at all.
- 020 L: without the fix the wall is genus 0 but truncated at 135.5 of 279 mm. With the fix it reaches 229.5 mm with
  genus 3. 020 R fails QC either way.
- On the pilot, 0 of the 11 triggered walls lost gate validity.

The earlier progress record follows.

(06:45) Pipeline v2 is READY for the lanes (`notes/SCALEUP_PIPELINE.md`).
- **Tally baseline (pilot only, 04:05; kept as `results/scaleup_tally_baseline_pilot_0405.json`):** 12 eligible pairs
  (172-series 5, Song 6, VMR 0248). All 12 have an all-bins donor-stretch union > 0.027 on both sides; 8 also do on
  contact-free bins. Stop rule not met.
- **V1 rescue (Phase A, 04:17-05:45; logs `_scaleup_fixes_logs/pilot_v2_{song,ia172}_A.out`, `pilot_v2_061.out`):**
  - USED: 172_061 L (5.6 mm bridge at 310 HU = 0.76 x CCA; 2791 voxels added, r_eq min 1.56 mm; re-trace passes as donor).
  - USED: Song 141 R (4.6 mm at 587 HU; 881 voxels; donor).
  - USED: Song 109 R (2.0 mm provisional bridge; 349 voxels; now donor without a bridge).
  - USED: 172_001 R (2.1 mm provisional; 675 voxels; donor_review, rule 6 only, no longer provisional).
  - NOT USED: Song 044 R. The mask was re-joined (9.1 mm, 2947 voxels, 459 HU), but the side still fails rule 1: the
    VA-SA junction is 3.25 mm from the TS subclavian. That is origin placement, not a mask break.
  - REFUSED, correctly (bridge HU < 0.6 x CCA): 172_041 R/L (177 / 128 HU), 073 L (-25 HU), Song 106 R (11 / 92 HU),
    Song 090 R/L (26 / 62 HU).
- **Pinch fix, label check with PILOT seeds** (`results/scaleup_fixes_ctseg_compare.json`, 19 cases / 30 walls):
  - 11 walls triggered: 172 001 R, 015 R/L, 090 R, 118 L; Song 008 L, 032 L, 094 R/L, 100 L, 106 L.
  - Song 094 R: the truncation is fixed (label 0 -> 241.75 mm; the pilot stopped at 122.75 mm; Dice 0.736). The other 10
    have Dice 0.975-0.999.
  - 19 walls did not trigger. 16 are bit-identical. The other 3 (Song 008 R, 032 R, 100 R) differ by 2-7 voxels at the
    vertebrobasilar confluence, where labels moved between R and L because the other side's wall changed
    (`results/scaleup_fixes_ctseg_offside.json`).

## Method

**Fix 1: V1 mask-break rescue**
- Script `scaleup_v1rescue.py`; the rule is in its header and in SCALEUP_PIPELINE.md section 3.
- Applicability scan: `scaleup_scan_bridges.py`.
- After a rescue: re-trace with the base set's tool, VMTK, then the unchanged QC (`qc_<e>_v1r.json`).
- `scaleup_select.py` level 2 decides whether the rescued trace replaces the level-1 trace.
- Figures `qc/v1rescue_<case>.png`, plus by-eye looks `qc/v1rescue_look_*.png` (`scaleup_v1rescue_look.py`).

**Fix 2: C2-exit pinch**
- `scaleup_ct_seg.py` (one script, both sources; adapters `scaleup_ctseg_common_{ia172,song}.py` = the pilot adapters
  plus the selected trace set).
- A station is flagged when its centre HU < the median T_loc, or when >= 50 % of its rays sit at the 0.3 R floor.
  - The floor test is needed for 094 R, whose centre HU of 344-396 was not below T_loc (341-350).
  - Probe: `scaleup_pinch_probe.py`.
- A flagged station is moved to the deepest lumen-bright point (<= min(R + 1, 3) mm), then the vessel is re-cast.

**Pilot re-runs**
- `scaleup_fixes_ctseg_check.sh`: ct_seg4v2 for all 19 walled cases, pilot seeds.
- `scaleup_fixes_pilot_v2.sh`: scaleup_v2_post.sh with the re-run names. Phase A = qc select rescue on all 22 cases.
- `scaleup_fixes_wall2.sh`: gross through collect, run 06:21-08:20 on the 12 cases a fix changed:
  - the rescued cases 001, 061, Song 141 and 109, whose walls are re-seeded from select_v2.json;
  - the pinch-triggered cases 015, 090, 118 and Song 008, 032, 094, 100, 106.
- `scaleup_fixes_wall2_finish.sh`: stopped wall2 before group C (the 10 untouched cases; decision 07:10, so as not to
  slow the lanes) and ran the grade.
- Fix-2 fallback: `scaleup_ct_seg.py` block v2-c reads `ct_seg/pinch_skip_<tag>.json`; the driver is
  `scaleup_ctseg_fallback.sh`.
  - Test: `scaleup_fixes_fbtest.sh`, evidence in `scaleup_work/fbtest_138/`.
  - Safety diagnostics: `scaleup_fixes_diag_measure.sh`, `scaleup_fixes_labelcc.py`, `scaleup_fixes_genus.py`; records
    in `scaleup_work/diag_nopinch/`.
- By-eye context: `scaleup_fixes_rescue_ctx.py` writes `qc/v1rescue_ctx_<case>_<side>.png`.
- By-eye verdicts go to `results/scaleup_eye_rejections.json`. Kept: Song 141 R, Song 109 R, 172_001 R. 172_061 L is
  left pending.
- Before / after: `scaleup_fixes_before_after.py` -> `results/scaleup_fixes_before_after.json`.

**Tally**
- `scaleup_tally.py` -> `results/scaleup_tally.json`. Inputs: pilot rows, then v2 re-runs, then lane results, plus VMR
  0248 and the by-eye verdicts in `results/scaleup_eye_rejections.json`.
