# H0 Discrepancy: v1b baseline 36.7% (36/98) vs v1 pooled 46.9% (45–48/98)

**Date:** 2026-07-16 · **Runs:** `2026-07-16_235715_rcca_p2_teacher_v1` vs `2026-07-18_030448_rcca_p2_teacher_v1b` · **Sources:** episode-level comparison (jsonl + worker logs) and code audit (synchron.py, runner.py, DualDeviceNav_train.py, sacmodel.py)

## Verdict

**The −9 episode gap is an eval-protocol artifact — cross-episode environment-state carry-over putting the two launches in opposite "orbit phase" — amplified by unseeded launch RNG and SOFA noise. It is NOT a policy regression and NOT the synchron fix (c29af94).**

## Ranked causes

1. **PRIMARY — eval env is not fully reset between episodes.** `rot_inst` at EPISODE_START is nonzero, differs per episode, and depends on run history rather than seed (same seeds → different values across runs; v1's per-worker ep-start sequence repeats EXACTLY across all six eval windows). Each worker's eval env follows a deterministic *orbit*, so outcome is largely determined by episode POSITION in the worker's sequence, not by the seed/target: v1 pooled per-position (n=96) shows pos1/pos3 ~90% RVA, pos2/pos4 ~75% success; v1b (n=16) is in the opposite phase (pos1 0 RVA, pos2 16/16 RVA, pos6 finals ~14 RVA). Identical targets flip purely by position within the same v1b eval. The entire −9 sits in the 16 final-per-worker episodes: v1 13/16 success (stable across all six windows) vs v1b 1/16 (14 RVA wrong-branch stuck at inserted≈44). On the 82 log-matched episodes v1b is equal-or-better (success 35 vs 32; clean 51 vs 50).

2. **ENABLER — unseeded launch RNG sets the phase.** No `torch.manual_seed`/`np.random.seed` anywhere in the launch path; policy residual init (deployed as `clip(a_heur + tanh(mu))` every step) and heatup random-action history differ per launch, giving v1b a different pre-eval device state → opposite-phase orbit. Pre-existing launch lottery, unrelated to c29af94.

3. **AMPLIFIER — SOFA nondeterminism** on marginal seeds (13–18/98 mixed within-process, v1 windows 45.9–49.0%). But flip structure across runs is anti-correlated, not random: 61 flips, 52 on v1-*deterministic* seeds; v1b succeeded on 31 of v1's 32 failures. So symmetric coin-flip noise alone cannot explain the pattern — the position/orbit mechanism (1) does.

## Exonerated

- **(a) Target/anatomy generation:** all 98 (worker, seed, target) tuples byte-identical across runs, including seed→worker assignment and order. Reproducible.
- **Synchron change c29af94:** `_update_algo_state_dicts` has no call site before the baseline eval (`--pretrain_updates 0` skips `update()`); zero NET_SYNC log lines pre/during baseline; at update=0 trainer≡algo (deepcopy) so the pull is an identity op anyway.
- **(d) Code/config drift:** working tree clean in all code dirs; run configs differ only in `_id`/path fields; launcher diff is a 5-line name rename.
- **(e) Heatup as differential cause:** present with identical config in both runs; only matters as the history that seeds mechanism (1)/(2).

## Is v1b still a valid experiment?

**Yes, within-run.** v1 demonstrated the orbit repeats exactly across all six eval windows of a run, so the phase is stable *within* a launch. v1b's H0-control logic (judge learning by delta from its own 36.7% baseline) remains valid. The run is healthy — do not restart it over the baseline number.

**Cross-run/absolute comparisons are NOT permitted** until the reset bug is fixed. Both 36.7% and 46.9% largely measure orbit phase, not policy quality.

## Noise model correction

The ±1.8pp noise model was **wrong in scope**: it was estimated from within-process repeats (same launch, same orbit, ~±1.5pp / 13–18 mixed seeds) and does not transfer across launches. The honest **cross-launch** noise floor is ~±10 episodes (~±10pp) of structured phase noise at n=98 — and it is not Gaussian, it is bimodal in orbit phase. Within-run window-to-window noise stays ~±1.5–2pp.

## Required changes

1. **Fix eval reset:** hard-reset device/rotation state (or reconstruct the eval env) per episode/seed so outcome depends on seed, not position. This is the P0 fix.
2. **Seed the launch** (`torch.manual_seed` + `np.random.seed`) and snapshot `policy_0.pt` so baselines are comparable across launches.
3. **Fix eval logging:** each worker's final episode never flushes EPISODE_OUTCOME (98 STARTs vs 82 OUTCOMEs); rely on `diagnostics/csv/episode_summary.jsonl` (complete, 98/98) until fixed.
4. **Re-read v1's deterministic-probe conclusions:** its "85/98 deterministic seeds" and "6 identical evals" were *position*-determinism from the orbit, not seed-determinism — any inference built on that needs revisiting.
5. Until (1)–(2) land, average 2–3 baseline repeats and compare runs only on within-run deltas.

## Key evidence paths

- `saved/eve_paper/neurovascular/full/mesh_ben/<run>/diagnostics/csv/episode_summary.jsonl` (both runs, 98 records each)
- `eve_rl/eve_rl/agent/synchron.py` L617, L753, L915; `eve_rl/eve_rl/runner/runner.py` L834–875; `training _scripts/DualDeviceNav_train.py` L601–625; `eve_rl/eve_rl/model/sacmodel.py` L152–155
- Analysis scripts: scratchpad `compare_evals.py`, `analyze2.py`–`analyze4.py`
