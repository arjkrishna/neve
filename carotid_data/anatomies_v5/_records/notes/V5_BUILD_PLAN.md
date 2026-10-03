# v5 build plan (autonomous; written 2026-10-01 ~22:10 EDT)

The user authorised (2026-10-01, before leaving): after the 1 am limit reset, continue without them, use best
judgement at decision points, finish the project and produce the v5 anatomies; if quota remains, render mesh images
the way two v4 images were made. Follow memories: v5-lower-va-goal, checkpointing-agents (tightened protocol),
workflow-resume-after-limit, match-v3-wall-detail, ask-before-investigating (no side investigations), and the
standing rules: NO git commit/push (not asked), do not edit launch scripts, never modify carotid_data/anatomies_v4,
"user copy.md" and saved/v4_trial are not ours, two-machine rule applies to any later push.

Progress log: append one line per milestone to `v5_va/state/V5_BUILD_LOG.md` (time, step, outcome, decision).

## Step 0 -- finish the pilot (workflow wf_54a8a0f4-215, task wedc98wvw)
Resume it if the limit killed it (resume briefs / state files; see workflow-resume-after-limit). Read
`reports/Thin-slice cervical VA pilot (v5).md` and `v5_va/results/pilot_decide.json`.

## Step 1 -- donor pool (decision rules)
- Need: one donor PAIR per TopBrain patient whose VA pieces v4 uses (pilot_decide counts it; ~25).
- Eligible: QC-passing thin-slice pairs (Song 2024, 172-series, AortaSeg Dissec_01) and VMR 0248.
  NOT TaG-Net (no data licence -> internal only). SPL course-only, last resort.
- If eligible pairs < need + 5 margin and the pilot pass rate >= 20% of sides: process more open thin-slice cases
  (Song first, then 172-series; distinct patients; TotalSegmentator `--ml` + CPU split; same pipeline/QC as the pilot)
  until eligible >= need + 5 or the pool is exhausted. If pass rate < 10%: stop and use the fallback below.
- Fallback only for TopBrain patients left without a thin-slice pair: 1.5 mm TotalSegmentator traces (V3-loop check
  mandatory), baked as tube (`real_surface: false`), flagged in provenance.
- Selection: rank QC-passing PAIRS by coarseness (whole-VA and C6->C1 union > 0.3 mm on the CT-edge wall, plain-tube
  bin fraction), excluding relief from vein/carotid/bone contamination; rank within source and keep both sources
  represented; take the coarsest.

## Step 2 -- graft tool (scratch v5_va/graft_v5_va.py -> registry version; repo files untouched)
- Strategy B: host subclavian kept byte for byte; donor VA origin -> splice in V3; v4's TopBrain distal piece.
- Fixes from the feasibility report: gate the origin-seam turn <= 60 deg; course-plausibility gate (course RMS to the
  v4 corridor <= ~20 mm, tip distance, take-off band); host parent-tangent chord 5 mm; splice calibre rule; record
  take-off native vs placed (`takeoff_not_preserved` flag); keep every v4 gate (0.6 mm clearance incl. ramps/self,
  joins <= 60 deg, proper rotations, handedness follow rule).
- Assignment ONE-TO-ONE donor pair <-> TopBrain patient: dry-run every eligible pair against every anatomy of every
  TopBrain patient, then solve the assignment (Hungarian) maximising gate passes, then coarseness. A pair is used only
  with its one TopBrain patient. Mirrored "_L" anatomies take the donor's opposite side mirrored.
- Anatomy where no assigned donor passes: keep its v4 VA unchanged and flag it (223 stay one-to-one with v4).

## Step 3 -- bake and check (copies only: carotid_data/anatomies_v5/<name>/ + BUILD_v5.json)
- bake_v5.py = bake_meshes_v4 unchanged + run-time sections_for wrapper; real-surface donors as real sections,
  fallback donors as tube; 5 shards like v4.
- Gates per anatomy: one closed genus-0 two-manifold, 0 self-intersections, 60k / 23.5k clean decimation, every
  centreline point inside, right_route_detail identical to v4 (ratio 1.000), VA detail on the undecimated union
  (> v4 for every replaced VA), clearance. SOFA load test fixed (check the PASSED line; a schedule that enters the RVA).
- Split check: max overlap-free train/test split by shared donors must stay >= the v4 value (~170: 153/17).
- Verify with adversarial agents (checkpointed), as for v4.

## Step 4 -- records
CAROTID_V5.md at the repo root (new, uncommitted): what changed vs v4, donors and assignment, gates, wall numbers,
licences (Arterial-derived = non-commercial; Song / 172-series CC BY attribution; VMR acknowledgement; TotalSegmentator
CC BY), limitations. BUILD_v5.json mirrors BUILD_v4.json.

## Step 5 -- images (if quota remains)
monitoring/figure_carotid_anatomies_v5.py from the v4 script (same view, shared cube, two per image), adding the donor
lower VA in its own colour beside host subclavian / TopBrain VA; render a couple first
(saved/figs/carotid_v5/), then all 223 if quota allows.

## When the user returns
A short report: what was built, numbers, every judgement call made and why, what is left. Nothing committed.
