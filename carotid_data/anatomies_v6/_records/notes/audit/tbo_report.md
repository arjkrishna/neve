# tbo_report: user report and v6 to-do list for the TopBrain orientation audit

Agent prefix `tbo_report` (state: `state/tbo_report.md`). It writes up the results of tbo_history, tbo_geom,
tbo_cause, va_narrow and verify_tbo. The numbers come only from their result files, with the verifier's corrections
applied. No new measurement was made, apart from re-running tbo_cause's `repro.one` for figure curves; that run
reproduces its numbers exactly.

## Answer so far (final)

- **Report:** `D:\Arjun\neve\reports\TopBrain piece orientation audit (v5).md`. It covers:
  - the documented fix and where it lives;
  - what the fix pins: roll only, while tilt and heading come from the anchor;
  - a code-path table;
  - counts by piece and by class (v5 / v4 / v3);
  - the user's four images explained;
  - the root-cause re-run and the candidate fixes;
  - the narrow-VA verdicts with the verifier's corrections;
  - "For v6", the verification changes, open points and files.
- **Figures, `reports/tbo_figs/`:**
  - `fig1_user_cases_before_after.png`: one anatomy from each user image, front and side views, shipped (orange)
    against re-posed (blue):
    - k_011_left mr_012_L upright pose, which fails with a 106° join;
    - w_003_left mr_010 anatomical heading;
    - m_006_left mr_025_L seam 13 mm back, which passes;
    - m_006_left mr_027_L anatomical pose.
  - `fig2_seam_elevation_and_tilt.png`: (a) seam elevation against rise for v5 siphons; (b) tilt against the
    elevation gap for 568 frame pieces.
  - `fig3_va_top_ends.png`: VA end separation and left–right order, native / v4 / v5.
- **v6 list:** `v5_va\v6\V6_ITEMS.md`, items (a1–a7) orientation, (b) the w_018b move, (c) narrow VA, (d) records,
  (e) subclavian junction, (f) colour later. Item 5 (stackdumps) is recorded as done.

## Checks made

- `tbo/report/figs/figs_check.txt` recomputes from tbo_geom's files:
  - left siphon: rise < 20 in 19, negative in 10, seam tangent < 10° in 35;
  - minimum right-siphon seam tangent 25.8°;
  - VA ends: separation median 26.6 (v5) / 34.5 (v4), crossed > 10 mm 47 / 67, worst −34.8; native minimum
    du −0.3.

  All match the agents' figures.
- The fig1 metrics (`figs_check.txt`) equal tbo_cause's `repro.txt`. Examples: k_011 left siphon −38.9 / 61.6 and
  anat join 106.0; mr_025_L atrim 31.0, tilt 4.3, join 13.5; mr_027_L ship du −21.4 → anat 0.3, separation 3.6.
- fig2b, from tbo_geom's tsv: |tilt − gap| median 1.1, p90 4.5, max 15.2. The verifier's own extraction gives 1.6 /
  5.5 / 11.7, and the report quotes the verifier's figures.

## Verifier corrections applied in the report

- w_003 rises are 56.0 / 51.8 (−12 / −19 are errors).
- The VA cut search moves for join and clearance.
- The fit's "start pinned" applies to the translation only.
- Join uncertainty for re-posed seams is 20–45°: mr_027_L LVA anat is 57–75, mr_025_L is 13–58.
- R_PITCH bend45 clearance is 0.39–0.56, so not a clear pass.
- The k_004 RECA has an anchor 12 mm back but still fails.
- The v3 fork rule refuses 26 / 125.
- w_025 RVA deepest 0.547 on the donor stretch; 4 VAs below 0.65 on the polyline (w_047_left mr_023 RVA 0.637); LVA
  0.567 and w_027 0.630.

## Not done / for the orchestrator

- Repo docs are not edited. CAROTID_V5.md's Follow-ups and narrow-VA paragraph need the audit results and the record
  correction. `reports/README.md` has no entry for the new report; it is already modified in the working tree by
  someone else, so it was left alone.
- The report is a repo .md file only, not published anywhere.

## Files

- `tbo/report/dump_curves.py`: imports `tbo/repro/repro.py` and replaces only its figure() to save the curves. Output
  is `tbo/report/curves/*.json`.
- `tbo/report/figs.py`: the three figures, in `tbo/report/figs/`, plus `figs_check.txt`.
- `tbo/report/peek_curves.py`, `tbo/report/dr.sh`: container runner, /repo read-only.

Status: complete
