#!/bin/bash
# Plan v9 — RCCA AWAC training v2.
#
# Reward / signal changes vs rcca_awac_warmstart:
#   - Bug A fix: _committed_forks latch per (j_arc, +1) tuple — wrong-then-
#     correct AND correct-then-wrong episodes now both fire their full
#     sequence of fork rewards. (pathcontext.py)
#   - Bug B fix: clean_thread seeding filter = final_branch_idx ==
#     target_daughter_branch_idx; RVA-detour-then-RCCA episodes admitted.
#     (DualDeviceNav_train.py + env5.py info dict)
#   - Fork rewards: +1 (latched once per fork per episode) / -0.05
#     (repeat-fires on every wrong commit). Removes the freeze-at-fork
#     incentive of the old +-1.
#   - Step penalty: path-segment-conditioned — trunk (2) linear interp
#     -0.007 -> -0.002 by trunk arclength; (0) post-bif -0.007; (11)
#     bridge 0.0; target daughter (RCCA) 0.0; wrong daughter -0.007.
#     (env5.py inline reward)
#   - Depth reward inside RCCA: ArcLengthProgress doubles its
#     progress_factor when wire is in target daughter AND moving forward.
#     (arclengthprogress.py)
#   - SOFA-restore fuller save: 6 extra DOF DataFields (velocity, force,
#     externalForce, free_position, free_velocity, derivX) + restored
#     rotation_instrument + settle_steps default 50 -> 3.
#     (sofabeamadapter.py)
#
# Online RL setup (Plan v9 Change 8):
#   - --rl_start_mode sofa_restore: every online episode starts from a
#     random SOFA-restored "just before bif(11)" checkpoint.
#   - --checkpoint_dir saved/rcca_pre_bif11_checkpoints: the pool
#     captured during the heuristic regen (PRE_BIF11_CHECKPOINT_DIR=...
#     was set when generating rcca_heuristic_cache_v2.npz).
#   - Heuristic-cache-loaded episodes still come from full z=345 demos —
#     CheckpointRestoreWrapper bypasses heuristic_mode=True resets.
#   - --balanced_fraction 0.3: 30% of every batch drawn from
#     RCCA-final transitions (clean stream).
#   - --grad_clip 1.0: cheap insurance against gradient spikes.
#   - --restore_start_snapshots: dumps post-restore wire-pose PNGs into
#     diagnostics/restore_start_snapshots/ for visual verification.
#
# Caches required (regenerate before running this):
#   - saved/rcca_heuristic_cache_v2.npz  — 100 RCCA-final heuristic eps
#     under the new reward function, generated with PRE_BIF11_CHECKPOINT_DIR
#     set so each successful episode also writes its pre-bif(11) state.
#   - saved/rcca_heatup_cache_20_v2.npz  — heatup eps under new reward.
#   - saved/rcca_pre_bif11_checkpoints/  — pool of pre-bif(11) restore
#     points (one .npz per RCCA-final heuristic episode).

set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_awac_v2 2>/dev/null || true

docker run --name rcca_awac_v2 --gpus all --shm-size=24g --init -d \
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
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\pathfinder\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\pathfinder\fixedpath.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/fixedpath.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\pathfinder\dijkstra2.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/dijkstra2.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\intervention\monoplanestatic.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/monoplanestatic.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\intervention\simulation\sofabeamadapter.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/simulation/sofabeamadapter.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_bench\eve_bench\dualdevicenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_bench\eve_bench\dualdevicenav.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\intervention\target\centerlinerandom.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/target/centerlinerandom.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\saved:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
    --env_version 5 \
    -n rcca_awac_v2 \
    --insertion_z 345 \
    --replay_mode step \
    --per \
    --algo awac \
    --balanced_fraction 0.3 \
    --grad_clip 1.0 \
    --hidden 256 256 \
    --embedder_layers 0 \
    --learning_rate 0.0003 \
    --heuristic_seeding 100 \
    --min_success_rate 1.0 \
    --max_seeding_multiplier 6 \
    --target_branches "Centerline curve - RCCA.mrk" \
    --heuristic_factory "util.heuristic_policy_rcca:RCCAHeuristicActionFunctionFactory" \
    --seeding_success clean_thread \
    --snapshots centerlines \
    --restore_start_snapshots \
    --rl_start_mode sofa_restore \
    --checkpoint_dir /opt/eve_training/results/rcca_pre_bif11_checkpoints \
    --heuristic_cache_file /opt/eve_training/results/rcca_heuristic_cache_v2.npz \
    --heatup_cache_file /opt/eve_training/results/rcca_heatup_cache_20_v2.npz \
    --pretrain_updates 10000 \
    -nw 16 -d cuda:0
# NOTE: load-mode (reuses the v2 caches + the existing rcca_pre_bif11_checkpoints
# pool from the prior run — reward/seeding code unchanged, only the buffer-save
# was fixed). To REGENERATE caches+pool from scratch, swap the two
# --*_cache_file lines back to --save_heuristic_cache / --save_heatup_cache and
# set -e PRE_BIF11_CHECKPOINT_DIR=... on the docker run.
