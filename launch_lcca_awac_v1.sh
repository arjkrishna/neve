#!/bin/bash
# Plan v10 — RCCA AWAC training v3: few-state curriculum + entropy-floored AWAC.
#
# ONLY changes vs v2 (NO reward / observation / terminal / dynamics changes):
#   - Start states: ONLY the 5 good catheter-forward restore states
#     (saved/rcca_5good_checkpoints), instead of the 1028-state pool.
#   - Seed: heatup-until-10-threaded-RCCA (random, restore-at-fork from the 5
#     states) — NO heuristic, NO cache files. "threaded" (final_branch==target,
#     from info) is only a finite-time stop criterion; real success stays the
#     env's unchanged TargetReached.
#   - Entropy floor: --log_std_min -2 (std~0.135) — the durable fix for the
#     AWAC deterministic collapse both prior runs hit by update ~2.5k.
#   - Pretrain 1000 on the heatup seed buffer.
#   - Per-state restore logging (EPISODE_START | restore_ckpt=...) + eval
#     wrapped to start from the same 5 states (eval Quality = per-state
#     TargetReached rate).
# Env (env5 reward, LocalGuidance obs, TargetReached terminal) is byte-for-byte
# Plan v9. Target: >80% (deep-target TargetReached) from the 5 states.

set -e
export MSYS_NO_PATHCONV=1

docker rm lcca_awac_v1 2>/dev/null || true

docker run --name lcca_awac_v1 --gpus all --shm-size=24g --init -d \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\heuristic_only_run.py:/opt/eve_training/training_scripts/heuristic_only_run.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\heuristic_run_LCCA.py:/opt/eve_training/training_scripts/heuristic_run_LCCA.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\heuristic_run_LVA.py:/opt/eve_training/training_scripts/heuristic_run_LVA.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\heuristic_run_RCCA.py:/opt/eve_training/training_scripts/heuristic_run_RCCA.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\heuristic_run_RVA.py:/opt/eve_training/training_scripts/heuristic_run_RVA.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\collect_sofa_checkpoints.py:/opt/eve_training/training_scripts/collect_sofa_checkpoints.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\DualDeviceNav_train.py:/opt/eve_training/training_scripts/DualDeviceNav_train.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\env.py:/opt/eve_training/training_scripts/util/env.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\env2.py:/opt/eve_training/training_scripts/util/env2.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\env3.py:/opt/eve_training/training_scripts/util/env3.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\env4.py:/opt/eve_training/training_scripts/util/env4.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\env5.py:/opt/eve_training/training_scripts/util/env5.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\util.py:/opt/eve_training/training_scripts/util/util.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\agent.py:/opt/eve_training/training_scripts/util/agent.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\action_curriculum.py:/opt/eve_training/training_scripts/util/action_curriculum.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\heuristic_controller.py:/opt/eve_training/training_scripts/util/heuristic_controller.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\heuristic_policy.py:/opt/eve_training/training_scripts/util/heuristic_policy.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\heuristic_policy_rva.py:/opt/eve_training/training_scripts/util/heuristic_policy_rva.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\heuristic_policy_lcca.py:/opt/eve_training/training_scripts/util/heuristic_policy_lcca.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\heuristic_policy_rcca.py:/opt/eve_training/training_scripts/util/heuristic_policy_rcca.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\heuristic_policy_lva.py:/opt/eve_training/training_scripts/util/heuristic_policy_lva.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\checkpoint_restore.py:/opt/eve_training/training_scripts/util/checkpoint_restore.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\snapshot.py:/opt/eve_training/training_scripts/util/snapshot.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\util\diagnostics_logger.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/diagnostics_logger.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\util\probe_evaluator.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/probe_evaluator.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\util\experience_cache.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/experience_cache.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\algo\sac.py:/usr/local/lib/python3.8/dist-packages/eve_rl/algo/sac.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\network\gaussianpolicy.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/gaussianpolicy.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\agent\single.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/single.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\agent\singelagentprocess.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/singelagentprocess.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\agent\synchron.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/synchron.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\runner\runner.py:/usr/local/lib/python3.8/dist-packages/eve_rl/runner/runner.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\vanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillashared.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\vanillaepisode.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillaepisode.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\vanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillastep.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\pervanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillastep.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\pervanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillashared.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\replaybuffer.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/replaybuffer.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\util\polyline.py:/usr/local/lib/python3.8/dist-packages/eve/util/polyline.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\util\pathcontext.py:/usr/local/lib/python3.8/dist-packages/eve/util/pathcontext.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/util/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\reward\arclengthprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/arclengthprogress.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\reward\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/reward/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\reward\waypointprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/waypointprogress.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\observation\localguidance.py:/usr/local/lib/python3.8/dist-packages/eve/observation/localguidance.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\observation\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/observation/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\observation\centerlines2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/centerlines2d.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\observation\target2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/target2d.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\pathfinder\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\pathfinder\fixedpath.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/fixedpath.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\pathfinder\dijkstra2.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/dijkstra2.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\intervention\monoplanestatic.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/monoplanestatic.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\intervention\simulation\sofabeamadapter.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/simulation/sofabeamadapter.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\intervention\target\centerlinerandom.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/target/centerlinerandom.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_bench\eve_bench\dualdevicenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_bench\eve_bench\dualdevicenav.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\saved:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
    --env_version 5 \
    -n lcca_awac_v1 \
    --insertion_z 345 \
    --replay_mode step \
    --per \
    --algo awac \
    --balanced_fraction 0.3 \
    --grad_clip 1.0 \
    --hidden 256 256 \
    --embedder_layers 0 \
    --learning_rate 0.0003 \
    --log_std_min -2 \
    --update_per_explore_step 0.5 \
    --replay_buffer_size 11000000 \
    --heatup_cache_file /opt/eve_training/results/lcca_awac_seed_v1.npz \
    --pretrain_updates 10000 \
    --target_branches "Centerline curve - LCCA.mrk" \
    --snapshots centerlines \
    -nw 16 -d cuda:0
#
# LCCA ONLINE AWAC from z=345 (Plan v12 Stage 3 / Challenge-1). Mirrors the v3
# RCCA pipeline (the 86%-on-pid19116 stack) byte-for-byte on the frozen knobs:
#   AWAC lambda=3.0 (default), step PER, balanced_fraction=0.3, grad_clip=1.0,
#   log_std_min=-2, MLP 256x256, embedder_layers 0, lr=3e-4, pretrain 1000.
# DIFFERENCES (all data-side / start-side, no algo change):
#   * from z=345: NO --checkpoint_dir restore, NO --rl_start_mode sofa_restore.
#   * SEED = curated heatup load lcca_awac_seed_v1.npz via --heatup_cache_file
#     (NOT heuristic-from-restore). Built per D1+D2: all LCCA successes (1308) +
#     all ostium near-misses F0 (1405) + all 'other' wrong-branch (1575), with
#     F2(LVA)->3000 and F1(trunk)->1500 reservoir-thinned; is_clean=1012 with the
#     87 overshooters dropped from the clean lane (D1). ~8.7k eps / 1.07M trans.
#   * --replay_buffer_size 11e6 so the 1.07M-transition seed is <10% of capacity
#     -> the is_clean lane is not FIFO-evicted during the early/mid window.
#
# Rebuild the seed (re-run when the harvest has grown):  scratchpad/build_seed.py
# Monitor:  docker logs -f lcca_awac_v1   |   Stop:  docker stop lcca_awac_v1
