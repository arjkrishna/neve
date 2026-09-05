# saved/stuck — extracted stall / recovery records

These are the INPUTS to every recovery-metric analysis in HANDOFF 14.1 (grind / soft /
hard / unrecovered, the cross-matrix cells, buckle-clearing adjudication, per-500-episode
bucketing). They are the product of running an extractor over per-step worker logs. The
worker logs themselves are gigabytes (run 2: 670 files, 4.5 GB), are gitignored, and will
not survive a machine change — these files are the durable record. Every file here was
matched back to its source eval by seed → (success, steps) agreement of 100%.

Two precedent files sit one level up: `saved/stuck_v1b.jsonl` and `saved/stuck_v1bp.jsonl`
are `extract_stuck.py` output over the July v1b / v1bp training runs.

## Two record formats

| extractor | one record per | fields | eval handling |
|---|---|---|---|
| `monitoring/extract_stuck.py <run_dir> <out>` | explore episode | `t pid pl steps succ reason gw_max cath_lead_frac d_tgt min_r ev{canon,sens,strict}` — events with kind grind/soft/hard/unrec and withdrawn length | EXCLUDES eval episodes using main.log `evaluation : <dur>s` windows |
| `monitoring/buckle_clear_extract_v1.py` (`run(log_dir, out)`) | episode | `seed tag pl succ reason proj[] gw[] cmd[] cs[] fold[] events[]` — full per-step series | none; eval episodes carry a small `seed`, explore episodes `seed: null` |

The three detector configs are in `extract_stuck.py` (`CONFIGS`); HANDOFF 14.1 quotes them.

## Files

| file | rows | format | source logs | backs |
|---|---|---|---|---|
| `tr_A.jsonl` | 98 | buckle | v1bp `checkpoint2002292` on HOST — `2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292/logs/20260729_085006` | host TEST 75.5% (HANDOFF 10.4) |
| `tr_A514.jsonl` | 98 | buckle | v1bp `checkpoint514264` on HOST — `.../eval_anatomies_checkpoint514264/logs/20260729_070938` | v1bp mid-run host baseline |
| `tr_H0.jsonl` | 98 | buckle | v1bp `checkpoint0` (pure heuristic) on HOST — `.../eval_anatomies_checkpoint0/logs/20260728_045004` | host heuristic 25.5% |
| `tr_ATB.jsonl` | 220 | buckle | v1bp `checkpoint2002292` on TopBrain ALL-22 — `.../eval_anatomies_checkpoint2002292/logs/20260828_053306` | 75.0% all-22 / 55.6% grafted |
| `trB_256370_host.jsonl` | 98 | buckle | TopBrain run 1 `checkpoint256370` on HOST — `2026-08-28_075919_rcca_topbrain_v1/checkpoints/eval_anatomies_checkpoint256370/logs/20260828_185008` | 44.9% |
| `trB_505230_host.jsonl` | 98 | buckle | run 1 `checkpoint505230` on HOST — `.../eval_anatomies_checkpoint505230/logs/20260828_171534` | 64.3% |
| `trB_own_eval1.jsonl` | 98 | buckle | run 1 in-run validation eval block 1 (= H0, before any update) from `diagnostics/logs_subprocesses`; 4-anatomy holdout cycled | 73.5% |
| `trB_own_eval2.jsonl`, `trB_own_eval3.jsonl` | 98 each | buckle | run 1 validation eval blocks 2 and 3 | 99.0 / 99.0 |
| `trB_own_explore.jsonl` | 5,902 | buckle | run 1 EXPLORE episodes (`seed: null`) | per-500-episode bucketing, heatup stall collapse |
| `tb_stuck.jsonl` | 5,902 | extract_stuck | run 1 explore, eval windows excluded | recovery rate between `ck256370` and `ck505230` |

Run 2 (`2026-08-29_235446_rcca_topbrain_v1`, finished 2026-09-02 at the 48,000-episode
limit, ~4.98 M explore steps): see the RUN 2 section at the bottom, added after extraction.

## Caveats that bit before

- **Clock offset.** In run 1, main.log timestamps and STEP `wall_time` were offset by
  ~2 h, so `extract_stuck.py`'s eval windows can miss. Check: output row count should equal
  the explore episode count, not explore + eval. `task4_report_topbrain_v1.py` recovers eval
  blocks by clustering seeded episodes on their own `wall_time` instead.
- **Per-worker `global_steps` is not a clock.** Order episodes by `t` / `wall_time`
  (HANDOFF 14.6). `task4_report_topbrain_v1.py` section 5 buckets by per-worker steps and
  should not be used for the time course.
- **Worker logs drop each worker's final episode**, so counts are ~16 short per run.
- **Always the shipped extractor.** A hand-rolled detector found 45 stalls where
  `extract_stuck.py` finds 1,059 on the same stream (it uses `abs(cmd_action[0])`).

## Regenerate / consume

```bash
python monitoring/extract_stuck.py <run_dir> <out.jsonl>
python -c "import sys; sys.path.insert(0,'monitoring'); import buckle_clear_extract_v1 as m; m.run('<log_dir>', '<out.jsonl>')"
python monitoring/analyze_stuck.py <a.jsonl> <b.jsonl> [canon|sens|strict]
python monitoring/buckle_clear_adjudicate_v4.py        # reads this directory (SP)
```

`buckle_clear_adjudicate_v4.py` used to point at a session temp directory; it now resolves
`SP` to this folder. Files stored as `.jsonl.gz` need `gunzip -k` first.
