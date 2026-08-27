#!/bin/bash
# RL_IMPROV_16 (v3c) — fresh RCCA procedural-ostium harvest under the v3c
# REWARD PAIR (human-approved 2026-07-21):
#   --progress_tip_mode avg  : progress = delta of the average arc of BOTH
#       device tips (parked guidewire halves pay; the measured v3a reward
#       tie at the choke breaks toward deploying the wire)
#   --cath_slack_coef 0.5    : catheter-slack potential channel (prices
#       the coil/knot formation the tip-based terms cannot see)
# The old rcca_proc_heatup/seed.npz rewards are STALE under this pair and
# the per-device tip arcs are not recoverable from its stored obs — hence
# this fresh harvest. All four reward-version stamps (buckle, cath-slack,
# tip-mode, gw-weight) are embedded in the npz; the v3c training launcher
# fails fast on any mismatch. Everything below the reward flags is
# byte-matched to the v1/v3a harvest (same meshes, same relaxed-truncation
# MDP, same 480 episodes).
#
set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_proc_harvest_v3c 2>/dev/null || true

docker run --name rcca_proc_harvest_v3c --gpus all --shm-size=24g --init -d \
  -e EVE_RL_EVAL_HARD_TIMEOUT_MIN=525600 \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\DualDeviceNav_train.py:/opt/eve_training/training_scripts/DualDeviceNav_train.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\env.py:/opt/eve_training/training_scripts/util/env.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\env2.py:/opt/eve_training/training_scripts/util/env2.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\env3.py:/opt/eve_training/training_scripts/util/env3.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\env4.py:/opt/eve_training/training_scripts/util/env4.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\env5.py:/opt/eve_training/training_scripts/util/env5.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\util.py:/opt/eve_training/training_scripts/util/util.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\agent.py:/opt/eve_training/training_scripts/util/agent.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\action_curriculum.py:/opt/eve_training/training_scripts/util/action_curriculum.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\checkpoint_restore.py:/opt/eve_training/training_scripts/util/checkpoint_restore.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\buffer_filter.py:/opt/eve_training/training_scripts/util/buffer_filter.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\buckle_reward.py:/opt/eve_training/training_scripts/util/buckle_reward.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\training _scripts\util\snapshot.py:/opt/eve_training/training_scripts/util/snapshot.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\util\diagnostics_logger.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/diagnostics_logger.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\util\probe_evaluator.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/probe_evaluator.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\util\experience_cache.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/experience_cache.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\algo\sac.py:/usr/local/lib/python3.8/dist-packages/eve_rl/algo/sac.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\network\gaussianpolicy.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/gaussianpolicy.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\network\component\mlp.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/component/mlp.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\agent\single.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/single.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\agent\singelagentprocess.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/singelagentprocess.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\agent\synchron.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/synchron.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\runner\runner.py:/usr/local/lib/python3.8/dist-packages/eve_rl/runner/runner.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\replaybuffer\vanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillashared.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\replaybuffer\vanillaepisode.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillaepisode.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\replaybuffer\vanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillastep.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\replaybuffer\pervanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillastep.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\replaybuffer\pervanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillashared.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\replaybuffer\replaybuffer.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/replaybuffer.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_rl\eve_rl\replaybuffer\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\env.py:/usr/local/lib/python3.8/dist-packages/eve/env.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\util\polyline.py:/usr/local/lib/python3.8/dist-packages/eve/util/polyline.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\util\pathcontext.py:/usr/local/lib/python3.8/dist-packages/eve/util/pathcontext.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/util/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\reward\arclengthprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/arclengthprogress.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\reward\waypointprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/waypointprogress.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\reward\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/reward/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\observation\localguidance.py:/usr/local/lib/python3.8/dist-packages/eve/observation/localguidance.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\observation\meshinvariant.py:/usr/local/lib/python3.8/dist-packages/eve/observation/meshinvariant.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\observation\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/observation/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\observation\centerlines2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/centerlines2d.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\observation\target2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/target2d.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\pathfinder\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\pathfinder\fixedpath.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/fixedpath.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\pathfinder\dijkstra2.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/dijkstra2.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\monoplanestatic.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/monoplanestatic.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\simulation\sofabeamadapter.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/simulation/sofabeamadapter.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\target\centerlinerandom.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/target/centerlinerandom.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\vesseltree\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\vesseltree\rccavariedfrommesh.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/rccavariedfrommesh.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\vesseltree\rccaprocedural.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/rccaprocedural.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\vesseltree\aorticarcharteries\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve\eve\intervention\vesseltree\aorticarcharteries\carotidsiphon.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\__init__.py:/opt/eve_training/eve_bench/eve_bench/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\dualdevicenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\dualdevicenav.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\dualdevicenavrccavaried.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\dualdevicenavrccavaried.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\dualdevicenavprocedural.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\dualdevicenavprocedural.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\archvariety.py:/usr/local/lib/python3.8/dist-packages/eve_bench/archvariety.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\archvariety.py:/opt/eve_training/eve_bench/eve_bench/archvariety.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\basicwirenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/basicwirenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\eve_bench\eve_bench\basicwirenav.py:/opt/eve_training/eve_bench/eve_bench/basicwirenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\saved:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
    --env_version 5 \
    -n rcca_proc_harvest_v3c \
    --procedural_rcca \
    --procedural_seed 12345 \
    --procedural_change_every 10 \
    --relax_failure_truncations \
    --buckle_reward_coef 0.5 \
    --cath_slack_coef 0.5 \
    --progress_tip_mode avg \
    --avg_gw_weight 0.5 \
    --heatup_only \
    --heatup_episodes 480 \
    --save_heatup_cache /opt/eve_training/results/rcca_proc_heatup_v3c/seed.npz \
    --target_branches "Centerline curve - RCCA.mrk" \
    --base_seed 42 \
    --replay_mode step \
    --per \
    --algo awac \
    --balanced_fraction 0.3 \
    --grad_clip 1.0 \
    --hidden 256 256 \
    --embedder_layers 0 \
    --learning_rate 0.0003 \
    --log_std_min -2 \
    --log_std_max 0.0 \
    --pretrain_updates 0 \
    --snapshots centerlines \
    -nw 16 -d cuda:0
#
# --procedural_change_every 10 : each worker re-randomizes its siphon every 10
#   episodes DURING the harvest → over 480 episodes / 16 workers = 30 ep/worker
#   = ~3 meshes/worker (bump --heatup_episodes / lower change_every for more
#   mesh variety). --procedural_seed 12345 matches v1.sh so the harvest meshes
#   overlap the training meshes.
# Startup: 16 distinct "[Gen-4] varied-RCCA" seeds, a per-worker marching-cubes
# build every 10 eps. Runs 480 episodes then writes ONE seed.npz and exits.
# Then TRAIN:  bash launch_rcca_procedural_v3c.sh   (already loads this seed via
#   --heatup_cache_file; coefs match at 0.5 so the reward-version guard passes).
# Monitor: docker logs -f rcca_proc_harvest_v3c   |   Seed: saved/rcca_proc_heatup_v3c/seed.npz
