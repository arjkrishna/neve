#!/bin/bash
# Plan v13 — multi-target random heatup harvest, OBS-V3 (PathLookahead3D).
#
# Same harvester as launch_rcca_z345_multitarget_heatup.sh with THREE changes:
#   1. obs-v3 (commit a49c1c9): LocalGuidance 30->48 (4x3D path waypoints +
#      bend_hat_3d + entry_dir_3d), total obs 78->96. Chunks written here are
#      96-dim and MUST NOT be mixed with the old 78-dim runs — hence the new
#      output folder heatup_z345_obsv3/.
#   2. EVE_OFF_BRANCH_GRACE_STEPS=150 (was 50): gives random exploration room
#      to COMPLETE off-path recoveries (retract ~1.3mm/step vs up to ~50mm
#      depth), so the harvest can contain retraction/recovery successes —
#      the demos AWAC needs to clone recovery behaviour (fable.md concern #1).
#      Also matches the grace the per-branch v3 training runs will use.
#   3. --base_seed 43 (fresh random stream vs run2's 42).
#
# Consumers: per-daughter seeds for launch_<k>_awac_v3 runs (all 4 daughters).
# Stop: docker stop -t 60 heatup_z345_obsv3   (SIGTERM flush)

set -e
export MSYS_NO_PATHCONV=1

docker rm heatup_z345_obsv3 2>/dev/null || true

docker run --name heatup_z345_obsv3 --gpus all --shm-size=24g --init -d \
  -e EVE_RL_EVAL_HARD_TIMEOUT_MIN=525600 \
  -e EVE_OFF_BRANCH_GRACE_STEPS=150 \
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
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\reward\targetreached.py:/usr/local/lib/python3.8/dist-packages/eve/reward/targetreached.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\terminal\targetreached.py:/usr/local/lib/python3.8/dist-packages/eve/terminal/targetreached.py" \
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
    -n heatup_z345_obsv3 \
    --insertion_z 345 \
    --multi_target_heatup \
    --heatup_only \
    --base_seed 43 \
    --save_heatup_cache /opt/eve_training/results/heatup_z345_obsv3/heatup.npz \
    --target_branches "Centerline curve - RCCA.mrk" \
    --snapshots centerlines \
    --snapshot_every 10 \
    --heatup_save_every 30 \
    --replay_mode step \
    --per \
    --algo awac \
    --balanced_fraction 0.3 \
    --grad_clip 1.0 \
    --hidden 256 256 \
    --embedder_layers 0 \
    --learning_rate 0.0003 \
    --log_std_min -2 \
    --pretrain_updates 0 \
    -nw 16 -d cuda:0
#
# Stage 1 SMOKE run (Phase 7) — short, validates the new code paths:
#   --heatup_episodes 100 is intentionally small so the docker container
#   completes in ~15–25 min with 16 workers. Once Phases 1–6 are
#   confirmed working end-to-end (Episodes saved with correct
#   target_branch_idx partitioning, R17-safe is_clean populated,
#   final_branch_short distribution sane), bump --heatup_episodes for
#   the real Stage 2 indefinite harvest.
#
# Manual stop / inspect (Decision #13 manual stop):
#   docker stop -t 60 heatup_z345_obsv3   # SIGTERM → workers finish current SOFA
#                                      # episode + flush final <500 batch, then
#                                      # exit. -t 60 gives the grace window so
#                                      # the in-flight batch is not SIGKILLed.
#   docker logs -f heatup_z345_obsv3      # live progress; EPISODE_OUTCOME lines
#                                      # carry every npz filter key.
#   ls saved/heatup_z345_multitarget/  # rolling Version A + Version B chunks
#   # Then ask Claude to glob+concatenate the chunks and report per-daughter
#   # clean / success / failure counts when deciding the AWAC data mix.
#
# When real harvest looks satisfactory, the per-target .npz files become
# the Stage 3 --heatup_cache_file inputs for the per-daughter AWAC runs.
