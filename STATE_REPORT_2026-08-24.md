# State report — before check-in

Scope: what is committed, what the latest design doc gets wrong, what happened that it never
records, and what is implemented vs still to build. **Nothing has been committed in producing
this report.**

---

## A. Headline

**`RL_IMPROV_18_P2_DESIGN.md` was last committed 2026-07-28 (`c5cf35e`). Every correction that
followed landed 07-29 onward.** The doc therefore states, as verified fact, six conclusions
that were subsequently retracted — including its own "open question that outranks everything
else", which has since been answered and dissolved.

It contains no mention of the geometric wall, the re-meshing defect, the clearance audit, the
calibration, the corrected results, or the checkpoint-selection finding.

---

## B. RETRACTED — statements in the doc now known false

| doc location | says | actually |
|---|---|---|
| §4.1 runs | both runs "Ended by OOM" | **IPC-deadlock watchdog** (`os._exit 42`); `grep OOM` on both `main.log` = 0 hits |
| §4.2 matrix | real patient **35.7%** (v1b and v1bp), H₀ 25.5% | measured on a **re-meshed reconstruction**, not the patient surface. True surface: v1b 52.0 / 63.3, v1bp **72.4 / 75.5** |
| §4.2 matrix | 50 generated: 57.1 / 55.1 | measured on anatomies of which **~2/3 are geometrically impassable**. Calibrated + gated: **84.7 / 83.7** |
| §4.3 #4 | "the siphon is an **absolute wall**: 0/30 for all three" | the arrest sat at **proj_s 153.4 mm — 57 mm BEFORE the siphon band begins**. On the true surface v1bp reaches **70.0%** siphon (ckpt514264) |
| §4.3 #5 | "the reward pair is a **null result**" | on the corrected mesh it is a **large win**: +20.4 pp overall, +63.3 pp siphon |
| §4.3 #6 | "the generator makes **EASIER** vessels than reality" | **inverted** — the generator makes *impassable* vessels (median clearance 1.26 mm vs patient 2.14 mm; 4 of 6 blocked) |
| §4.3 #9 | "recovery never emerged; soft share flat 2–6%" | detector-dependent; the canonical detector gives 22.6% (v1b) / 29.5% (v1bp). Its successor finding (r = −0.82, avoidance-not-escape) is itself **provisional** — computed on walled anatomies, never re-stratified |
| §6.2 item 7 | "Memory cap / replay trim — **OOM killed both**" | misdiagnosed; same watchdog cause as §4.1 |
| §6.4 item 12 | P2b gate: "worth doing once the real-anatomy number is worth inheriting — currently 35.7%" | that number is now **75.5%**; the gate is passed |
| §7 | "OPEN QUESTION THAT OUTRANKS EVERYTHING ELSE: why is the siphon 0/30?" | **answered** — a mesh artifact, not a siphon problem. Replace the section, don't amend it |

---

## C. NOT RECORDED ANYWHERE IN THE DOC

1. **Geometric-wall investigation → resolution.** Three controllers (v1b, v1bp, and the
   parameterless heuristic) arrest at 153.3–153.4 mm on 91% of real-patient failures; max
   solved path_len 156.2 mm identical across all three.
2. **The evaluator flaw.** `--real_patient_anatomy` reproduced the patient *centerlines* to
   zero floating-point error but re-meshed the *surface*. Fixed in `eb753d9` by pinning the
   original `.obj`; the wall disappeared.
3. **Clearance audit as a method** (`monitoring/mesh_clearance.py`) — surface-sampled, not
   nearest-vertex; the vertex version reported 1% erosion where the true figure is ~45%.
4. **Calibration:** `radius_scale = 1.6` reproduces patient clearance (median 2.14 vs 2.14;
   p05 1.13 vs 1.20, marginally tighter).
5. **New evaluator flags**, all default-off: `--require_passable`, `--passable_min_median_mm`,
   `--radius_scale`, `--stochastic_eval`.
6. **Corrected 2×2 results** and the **checkpoint-selection finding** — the eval-picked "best"
   checkpoints (757854, 514264) were early; true best by explore success are **3259127** and
   **2002292**, worth +11.3 and +3.1 pp.
7. **Stuck/recovery analysis** and its tooling (`extract_stuck`, `analyze_stuck`,
   `verify_stuck`, `report_stuck`, `report_single`) plus two 11-page reports.
8. **Stochastic-eval null** — strict superset, +1 episode; noise is not the lever.
9. **`MESH_GENERATOR_FIX_PLAN.md`** — parked; ablation written, **never run**.
10. **SPIE paper** and the 82-reference citation audit (12 gaps, 12 risks).

---

## D. CODE-STATE FINDINGS that contradict working assumptions

From the six-agent audit of the shipped `runner.yml` / `env_train.yml` / launchers:

1. **The algorithm is SAC**, not AWAC. Both launchers pass `--algo sac`; AWAC is gated behind
   `self.algo == 'awac'` and inert.
2. **The asymmetric actor-critic is OFF in v1b/v1bp.** `--privileged_actor` sets
   `privileged_obs_dim = 0`, making the policy's input slice a no-op; `runner.yml` records
   policy `n_observations: 125`, identical to the critics. **The shipped policy reads SOFA
   forces at training and test time.**
3. **No student was ever trained.** Both headline numbers are **teacher** numbers; no
   observation-only number exists anywhere in the repo.
4. **The planned-path ↔ force correspondence is asserted, not demonstrated**, and its dominant
   predictor (`gw_slack`) is *numerically identical* to privileged dim 23.
5. **UTD is 0.99**, not the 0.25 recorded in `PAPER_PLAN`; **throughput 10–11 steps/s**, not 47.
6. **Eval anatomy seeding is misdescribed**: the per-worker factory seed builds only the first
   tree; the *episode* seed drives every subsequent regeneration.

---

## E. UNCOMMITTED — 71 entries

**Should be committed (work product)**
- SPIE paper set — 7 `.tex` files + `SPIE_ABSTRACT/PURPOSE/METHODS/FIGURES.md`
- `ANATOMY_GENERATION_RATIONALE.md`
- `PAPER_PLAN_NEUROVASC_RL.md`, `HANDOFF_V3A_MACHINE2.md` — untracked despite being referenced
  by committed docs
- `launch_rcca_p2_teacher_v1b.sh` — **a launcher for a shipped run, untracked**, while its
  sibling `_v1bp.sh` is tracked
- `monitoring/report_single.py` (modified — table restructuring)
- Regenerated per-run report PDFs and PNGs

**Deleted by you — will honour unless told otherwise**
- `saved/p2a_deep_dive/STUCK_RECOVERY_REPORT.pdf` and all 13 `report_png/` pages (the combined
  v1b+v1bp report; the per-run reports are kept)

**Recommend NOT committing (bulk / regenerable)**
- `saved/stuck_v1b.jsonl` (2.0 MB), `stuck_v1bp.jsonl` (1.8 MB) — regenerable from
  `extract_stuck.py`; better as a `.gitignore` entry
- `eve_rl_deep_review_combined.pdf/.txt`, `saved/p2a_deep_dive/New folder/`,
  `saved/steplog_*.txt`, raw run directories

---

## F. IMPLEMENTED vs OUTSTANDING

**Implemented, verified, in use**
- `env_eval_factory` + per-worker eval anatomy streams; branch-coordinate anatomy hashing
- Standalone evaluator: Wilson CIs, depth split, seed reconciliation, parse audit, spawn
- `--real_patient_anatomy` (fixed — pins the original surface)
- `--require_passable`, `--passable_min_median_mm`, `--radius_scale`, `--stochastic_eval`
- Clearance audit; stuck/recovery extraction, analysis, verification, reporting

**Built, never launched**
- v1c crunchpass lane (`07aecf4`) — ⚠ premise now contested: grind beats soft in *both* runs
- v1d (v1c + reward pair); E3 stuck lane

**Parked mid-investigation**
- Mesh-generator fix — diagnosis measured, **ablation written and never run**
  (`monitoring/mesh_ablation.py`); mechanism hypothesis unconfirmed

**Never built**
- P2b student distillation — gate now passed (75.5%)
- A genuinely asymmetric run (`privileged_obs_dim > 0`) — believed running on machine 2,
  **unverified from here**
- The planned-path ↔ force correspondence probe

**Owed re-analysis (CPU only, data in hand)**
- Re-stratify r = −0.82 on audited anatomies — blocks a paper figure
- v1b calibrated-synthetic eval (~1.5 h; cancelled mid-run, leaves a hole in the results table)

**Infrastructure debt, unchanged**
- Trainer restart must restore state; probe re-arm; remaining invariant alarms; reproducible
  init dump. Item 7 ("memory cap") should be **re-scoped to IPC-deadlock robustness**, since
  OOM was a misdiagnosis.

---

## G. Proposed check-in plan

1. **Rewrite `RL_IMPROV_18_P2_DESIGN.md`** — §4 results, §4.3 facts, §6.2 item 7, §6.4 item 12
   and §7 need replacing rather than patching; add the missing material from C and the
   code-state findings from D.
2. **Commit the SPIE set** + rationale + plan/handoff docs.
3. **Commit `launch_rcca_p2_teacher_v1b.sh`** — it reproduces a run we cite.
4. **Commit `monitoring/report_single.py`** and the regenerated reports.
5. **Add a `.gitignore` entry** for `saved/stuck_*.jsonl` rather than committing 3.8 MB of
   regenerable intermediate.
6. Leave the deleted combined report deleted.
