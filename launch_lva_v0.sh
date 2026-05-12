#!/bin/bash
# LVA-target heuristic v1 — Phase C only (no Phase A/B; default heuristic
# handles trunk-to-(18) bridge crossing). 50-episode smoke test.
set -e
export MSYS_NO_PATHCONV=1

docker rm env5_rl8_LVA_v0_50ep 2>/dev/null || true

docker run --name env5_rl8_LVA_v0_50ep --gpus all --shm-size=24g --init -d \
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
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\agent\single.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/single.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\agent\singelagentprocess.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/singelagentprocess.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\agent\synchron.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/synchron.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\runner\runner.py:/usr/local/lib/python3.8/dist-packages/eve_rl/runner/runner.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\vanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillashared.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\replaybuffer\vanillaepisode.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillaepisode.py" \
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
  python3 /opt/eve_training/training_scripts/heuristic_run_LVA.py \
    --env_version 5 \
    -n env5_rl8_LVA_v0_50ep \
    --insertion_z 345 \
    --episodes 50 \
    --snapshots mesh \
    -nw 8 -d cuda:0
