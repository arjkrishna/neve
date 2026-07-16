# RL_IMPROV_18 — P2: Privileged-Actor Teacher, Residual on the Scripted Heuristic

Branch `rl_improv_18_p2` (from `rl_improv_17_rlpd` @ 7fc1efb). Roadmap: RL_PARADIGM_ROADMAP.md P2.
Trigger: P1 RLPD closed negative 2026-07-16 (evals 0.0/0.0; start-state deterministic
mean never left random-init across 226k updates at realized UTD 0.39 — from-scratch
SAC cannot bootstrap this task at our sim throughput; see saved/monitor_rcca_procedural.md).

## Idea

Stop bootstrapping from zero. The scripted `CenterlineFollowerHeuristic` already
navigates (P-control on heading + cross-track along the planned centerline, off-path
retract phase). P2a trains a **residual** on top of it with a **privileged actor**
(Scaffolder, arXiv:2405.14853: actor-side privileged information recovers most of the
teacher-student gap — critic-only left it on the table, which is what Gen-4 did).
P2b (later) distills the teacher to a deployable obs-only student via DAgger.

At init the residual mean ≈ 0 ⇒ behavior = pure heuristic ⇒ the run *starts* at
heuristic competence and RL only has to learn corrections — precisely the
stuck-state recoveries where the heuristic grinds.

## Mechanics (all default-off; legacy byte-identical)

| Piece | Where | What |
|---|---|---|
| Residual composition | `env5.BenchEnv5.step` | `a_total = clip(a_heur + residual_scale·a_policy)` in RAW units (worker de-normalizes before env.step; action shape (2,2)) |
| Heuristic ownership | `env5._heur_next_action` | Lazy `HeuristicActionFunction(self, noise_std=0, normalize_output=False)`; **once-per-state cache** (controller has phase counters — the obs component and step() composition share one call); invalidated in `_on_intervention_stepped` (post-SOFA, pre-observation) and marked stale before `super().reset()` |
| Heuristic intent obs | `env5.HeurActionObs` | 4-dim raw heuristic action, Normalize-wrapped, inserted **before** the privileged tail (deployable prefix — the student will see it too). Obs 121→125, policy prefix 97→101 |
| Privileged actor | `DualDeviceNav_train` → `privileged_obs_dim=0` | `GaussianPolicy`'s `[..., :n]` slice becomes a no-op → policy consumes the full obs incl. the 24-dim tail. No eve_rl change. Guard: requires `--aux_coef 0` (aux labels become inputs; rel→abs math would land out of range) |
| Heatup band | `--heatup_action_scale 0.3` | Heatup = heuristic + small residual noise → buffer seeds at ~heuristic quality with diversity. No cache/seed (old seed is obs- and action-space incompatible) |
| Baseline eval | runner.py hoist | `--eval_after_pretrain` now fires with `pretrain_updates 0` → measures the **pure heuristic** on held-out eval seeds before any learning: the run's null hypothesis |
| Stuck lane indices | train script | slack 89 unchanged; contact 103→107 under `--heur_action_obs` (handled) |

Kept from P1 (validated there): `--critic_layernorm` (Q stayed bounded 226k updates
with no BC anchor), `--no_entropy_backup`, sac alpha rails [−5, 0], target_entropy 1.0.
Restored from v2: PER + `--balanced_fraction 0.3` clean-lane success amplification
(P1's uniform symmetric sampler is off here).

## Why the critic stays full-width and the policy actions stay residual

- Q(s, a_res) is Markov-consistent: a_heur is a deterministic function of sim state,
  and the critic sees the privileged tail + heur_action dims.
- The buffer stores the policy's residual (normalized), exactly what the worker emitted —
  no change to storage, PER, caches, or diagnostics.

## Run design (launch_rcca_p2_teacher_v1.sh)

`--algo sac`, 16 procedural workers, buckle 0.5, relax truncations, UTD 1.0,
shm 30g (P1 OOM lesson), all deadlock guards. Decision gates:
1. Baseline eval = heuristic quality H₀. Everything after must beat H₀.
2. Explore success must START near H₀ (if ~0 → composition bug, kill).
3. Eval curve vs v2 (6.1/30.6/49.0/30.6) and vs H₀.
4. Q-divergence watch (LayerNorm should hold, as in P1).
5. Residual magnitude: small early (trusting heuristic), growing in stuck states.

## P2b — student distillation (next, after teacher validates)

DAgger: run the teacher, log (deployable-prefix obs → teacher action) pairs on-policy,
train an obs-only student (short obs history replaces ep_step per the obs audit);
student evaluated WITHOUT privileged tail. Optionally warm-start student = teacher's
prefix weights. Deferred until the teacher beats v2's eval3 ceiling.

## Deferred (needs explicit approval — reward changes are frozen)

- "reduce-slack" aux reward (roadmap P2 companion) — potential-based like buckle_reward.
- Stuck-restore curriculum from STUCK_CHECKPOINT_DIR (E4) — machine-2 harvest pending.
