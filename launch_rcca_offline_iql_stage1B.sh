#!/bin/bash
# Plan v11 Stage 1B — Offline IQL bake-off variant.
#
# Replaces AWAC's behavior-cloning-on-buffer signal (which collapsed to
# 5.6% in Stage 1A because the buffer is failure-dominated) with
# Implicit Q-Learning's expectile-V + advantage-weighted regression
# pipeline:
#   tau = 0.7      — V tracks 70-th-percentile Q on the buffer
#   beta = 3.0     — AWR temperature (workflow synthesis recommendation)
#   awr_max = 5.0  — cap on exp(adv/beta), prevents demo-dominance blow-up
#   lr_policy=5e-5, lr_value=3e-4, lr_critic=3e-4
#   grad_clip = 0.3 (tightened from AWAC's 1.0)
#
# Safety guardrails (Stage 1B):
#   --step0_eval               — mandatory eval BEFORE the first
#                                 optimizer.step() to capture warm-start
#                                 baseline quality
#   --abort_quality_drop 10.0  — abort training if eval Quality drops
#                                 > 10pp below the step-0 baseline at
#                                 any subsequent eval
#
# Eval cadence is 2k updates (configurable via --eval_every_updates)
# instead of AWAC's 10k — earlier divergence detection.

set -e
export MSYS_NO_PATHCONV=1

# v2 + v3 replay buffer archives — produced by `runner.eval()` periodic
# replay-buffer saves at the end of each rcca_awac_v{2,3} training run.
V2_RUN_DIR=/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2026-05-23_051700_rcca_awac_v2
V3_RUN_DIR=/opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2026-05-25_031951_rcca_awac_v3
V2_BUFFER=${V2_RUN_DIR}/checkpoints/replay_buffer.npz
V3_BUFFER=${V3_RUN_DIR}/checkpoints/replay_buffer.npz

# Warm-start: v3 eval-#1 policy (the same checkpoint Task C anchors on).
# IQL loads ONLY the network weights from this .everl (q1/q2/target_q1/
# target_q2/policy). The V-network is initialized fresh — there is no
# pre-trained V to inherit from SAC.
WARM_START=${V3_RUN_DIR}/checkpoints/checkpoint250531.everl

docker rm rcca_offline_iql_1B 2>/dev/null || true

docker run --name rcca_offline_iql_1B --gpus all --shm-size=24g --init -d \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\DualDeviceNav_offline.py:/opt/eve_training/training_scripts/DualDeviceNav_offline.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\DualDeviceNav_train.py:/opt/eve_training/training_scripts/DualDeviceNav_train.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\training _scripts\util\buffer_filter.py:/opt/eve_training/training_scripts/util/buffer_filter.py" \
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
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\algo\iql.py:/usr/local/lib/python3.8/dist-packages/eve_rl/algo/iql.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\algo\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/algo/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\model\iqlmodel.py:/usr/local/lib/python3.8/dist-packages/eve_rl/model/iqlmodel.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\model\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/model/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\network\value.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/value.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_rl\eve_rl\network\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/__init__.py" \
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
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve\eve\intervention\target\centerlinerandom.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/target/centerlinerandom.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_bench\eve_bench\dualdevicenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\eve_bench\eve_bench\dualdevicenav.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_8\saved:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_offline.py \
    -n rcca_offline_iql_1B \
    --algo iql \
    --iql_tau 0.7 \
    --iql_beta 3.0 \
    --iql_awr_max 5.0 \
    --v2_buffer ${V2_BUFFER} \
    --v3_buffer ${V3_BUFFER} \
    --v3_action_abs_max 0.85 \
    --warm_start_checkpoint ${WARM_START} \
    --checkpoint_dir /opt/eve_training/results/rcca_5good_checkpoints \
    --canonical_eval \
    --drop_restore_states pid10145,pid20043 \
    --target_branches "Centerline curve - RCCA.mrk" \
    --snapshots centerlines \
    --insertion_z 345 \
    --rl_start_mode sofa_restore \
    --n_updates 100000 \
    --eval_every_updates 10000 \
    --lr_policy 5e-5 \
    --lr_critic 3e-4 \
    --lr_value 3e-4 \
    --batch_size 256 \
    --grad_clip 0.3 \
    --per_alpha 0.6 \
    --per_beta_start 1.0 \
    --demo_priority_bonus 0.0 \
    --balanced_fraction 0.5 \
    --priority_mode outcome \
    --max_failure_per_band 5000 \
    --drop_all_demos_unconditional \
    --replay_buffer_size 350000 \
    --hidden 256 256 \
    --embedder_layers 0 \
    --log_std_min -2 \
    --entropy_beta_per_dim 0,0,0,0 \
    --action_mean_penalty 0.0 \
    --warm_start_mode network_only \
    --baseline_quality 0.577 \
    --abort_quality_drop 10.0 \
    -nw 8 -d cuda:0
# NOTE: --step0_eval intentionally OMITTED. With --baseline_quality
# pinned to Task C's 0.577 anchor (v3 checkpoint250531.everl on the
# canonical 54 seeds, 30/52 strict), the abort guardrail has a free
# reference point and we save the ~50 min of redundant step-0 eval
# compute. Use --step0_eval only on new ALGOS where construction sanity
# is unverified; use --step0_smoke_eval_seeds N for a cheap (~5 min)
# warm-start integrity check instead.
# After it starts:
#   docker logs -f rcca_offline_iql_1B
# Expected:
#   - "Filtering V2 buffer ..." / "Filtering V3 buffer ..."
#   - "Warm-start: loading model weights from .../checkpoint250531.everl"
#   - "Stage 1B: running mandatory step-0 eval BEFORE first update ..."
#   - "Stage 1B step-0 baseline: quality=0.577 ... (abort_quality_drop=10.0pp)"
#   - "Offline-train: 100000 updates, eval every 2000"
#   - every 2k updates: a Quality line from runner.eval() on 54 canonical seeds
#
# DECISION GATE (Plan v11 Stage 1B → Stage 2):
#   Best canonical strict % over all evals must beat the Task C anchor by
#   ≥ +5pp, AND no eval may drop > 10pp below the step-0 baseline (the
#   abort guardrail enforces this — training halts on violation).
