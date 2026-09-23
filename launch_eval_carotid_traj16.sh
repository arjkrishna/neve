#!/bin/bash
# + rl_improv_16_resume/saved mounted read-only at /opt/eve_training/results16 (procedural checkpoints).
# CAROTID v3 replay with per-step trajectory RECORDING (ANIM_PIPELINE tier 2).
# = launch_eval_topbrain.sh + carotid_data mount + util/traj_record.py mount +
#   --topbrain_dir carotid anatomies_v3, --topbrain_only <the 16 validation
#   anatomies of the carotid v3 split>, --traj_record. Records land in
#   <ckpt-dir>/eval_anatomies_<ckpt>/traj_records/<run_tag>/*.npz; render with
#   python monitoring/traj/traj_render.py <that dir> <out dir>.
# Evaluating a MASKED checkpoint needs the same env vars as its training run:
#   EVE_RL_ACTOR_ZERO_OBS=... [EVE_RL_CRITIC_ZERO_OBS=...] ./launch_eval_carotid_traj.sh <ckpt>
# Replaying the in-run validation blocks by seed: SEED_LIST=validation ./launch_eval_carotid_traj.sh <ckpt>
# Overrides: N_WORKER, CHANGE_EVERY, SEED_BASE, SEED_LIST, SNAPSHOT_MODE, TOPBRAIN_ONLY, NAME, EXTRA_FLAGS.
# TopBrain variant of launch_eval_anatomies.sh — evaluate a checkpoint on the
# grafted 25-patient cohort instead of the procedural set.
#
# Adds four mounts (loader module, env in BOTH eve_bench locations, read-only
# anatomy data) and defaults EXTRA_FLAGS to --topbrain plus the three v1bp
# reward-pair flags absent from the inherited ARCH/ENV defaults. Omitting those
# three changes the observation width and the strict load fails loudly.
#
# Everything else is inherited unchanged, so a number here and one from
# launch_eval_anatomies.sh come from the same harness and the same seeds.
#
# USAGE
#   TOPBRAIN_ONLY="topcow_mr_004 topcow_mr_008 topcow_mr_017 topcow_mr_023" \
#   NAME=eval_tb_teacher bash launch_eval_topbrain.sh <ckpt-in-container> 98
# Standalone MULTI-ANATOMY evaluation of a trained checkpoint.
#
# WHY: the training loop's eval is SINGLE-ANATOMY. DualDeviceNav_train.py
# builds env_eval as DualDeviceNavRCCAVaried(episodes_between_change=10**9)
# and RCCAVariedFromMesh.reset() only regenerates when
# episode_nr % episodes_between_change == 0 — a per-episode seed merely
# re-seeds the RNG. Verified 2026-07-27 by hashing vessel_tree.branches:
# geometry byte-identical across episodes while only the target and the
# device start-rotation moved. So every eval number so far (incl. the
# 81.6% v1b crest) measures ONE vessel tree, not generalization.
#
# THIS SCRIPT: worker i gets its OWN anatomy stream via the new
# env_eval_factory (mirrors training's env_train_factory), regenerating
# every CHANGE_EVERY episodes. Snapshots ON for every episode (successes
# AND failures, bucketed by outcome). EPISODE_START now carries
# anatomy=<branch-hash>, so diversity is VERIFIED in the report, not assumed.
#
# USAGE:
#   bash launch_eval_anatomies.sh <checkpoint-path-inside-container> [n_episodes]
# e.g.
#   bash launch_eval_anatomies.sh \
#     /opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2026-07-18_030448_rcca_p2_teacher_v1b/checkpoints/checkpoint757854.everl 98
#
# Override anything via env: CHANGE_EVERY=2 N_WORKER=16 SEED_BASE=900000
# ARCH_FLAGS="--residual_heuristic --heur_action_obs --privileged_actor --critic_layernorm"
# ENV_FLAGS="--relax_failure_truncations --buckle_reward_coef 0.5"
# (defaults below match the v1b / v1bp P2-teacher family; a mismatch makes
#  the strict checkpoint load fail loudly, which is intended.)
#
# COST: every anatomy regeneration forces a full SOFA scene rebuild. 98
# episodes at CHANGE_EVERY=2 over 16 workers ~= 48 rebuilds ~= 30-45 min.
# CHANGE_EVERY=1 doubles the rebuilds (~2 h) for ~2x the anatomies.

set -e
export MSYS_NO_PATHCONV=1

CKPT="${1:?usage: launch_eval_anatomies.sh <checkpoint-in-container> [n_episodes]}"
N_EPISODES="${2:-98}"
CHANGE_EVERY="${CHANGE_EVERY:-2}"
N_WORKER="${N_WORKER:-16}"
SEED_BASE="${SEED_BASE:-900000}"
SNAPSHOT_MODE="${SNAPSHOT_MODE:-centerlines}"
ARCH_FLAGS="${ARCH_FLAGS:---residual_heuristic --heur_action_obs --privileged_actor --critic_layernorm}"
ENV_FLAGS="${ENV_FLAGS:---relax_failure_truncations --buckle_reward_coef 0.5}"
MAX_STEPS="${MAX_STEPS:-600}"
# EXTRA_FLAGS e.g. "--frozen_anatomy --anatomy_seed 12344" to reproduce the
# legacy single-anatomy protocol (procedural_seed-1 = the tree every prior
# eval in this program actually used).
TOPBRAIN_ONLY="${TOPBRAIN_ONLY:-case_m_030_right__topcow_mr_003_L case_m_030_right__topcow_mr_013_L case_m_030_right__topcow_mr_016 case_m_030_right__topcow_mr_016_L case_w_007_left__topcow_mr_013_L case_w_007_left__topcow_mr_016_L case_w_007_left__topcow_mr_017 case_w_007_left__topcow_mr_021_L case_w_013_right__topcow_mr_011 case_w_013_right__topcow_mr_013_L case_w_013_right__topcow_mr_021_L case_w_017_right__topcow_mr_011 case_w_025_right__topcow_mr_016_L case_w_025_right__topcow_mr_021_L case_w_033_left__topcow_mr_011_L case_w_033_left__topcow_mr_016}"
# SEED_LIST=validation replays the trainer's 98 EVAL_SEEDS (same seed -> anatomy/target as the in-run blocks);
# SEED_LIST=<comma list> replays those seeds; unset = seed_base band (held-out, not comparable by seed).
SEED_LIST="${SEED_LIST:-}"
[ "$SEED_LIST" = "validation" ] && SEED_LIST="1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
_TB="--topbrain --topbrain_dir /opt/eve_training/carotid_data/anatomies_v3"
[ -n "$SEED_LIST" ] && _TB="$_TB --seed_list $SEED_LIST"
[ -n "$TOPBRAIN_ONLY" ] && _TB="$_TB --topbrain_only $TOPBRAIN_ONLY"
EXTRA_FLAGS="${EXTRA_FLAGS:-$_TB --cath_slack_coef 0.5 --progress_tip_mode avg --avg_gw_weight 0.5 --traj_record}"
NAME="${NAME:-eval_carotid_traj16}"

docker rm "$NAME" 2>/dev/null || true

docker run --name "$NAME" --rm -i --gpus all --shm-size=16g \
  -v "D:\neve\.claude\worktrees\rl_improv_16_resume\saved:/opt/eve_training/results16:ro" \
  ${EVE_RL_ACTOR_ZERO_OBS:+-e EVE_RL_ACTOR_ZERO_OBS=$EVE_RL_ACTOR_ZERO_OBS} \
  ${EVE_RL_CRITIC_ZERO_OBS:+-e EVE_RL_CRITIC_ZERO_OBS=$EVE_RL_CRITIC_ZERO_OBS} \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\DualDeviceNav_train.py:/opt/eve_training/training_scripts/DualDeviceNav_train.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\env.py:/opt/eve_training/training_scripts/util/env.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\env2.py:/opt/eve_training/training_scripts/util/env2.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\env3.py:/opt/eve_training/training_scripts/util/env3.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\env4.py:/opt/eve_training/training_scripts/util/env4.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\env5.py:/opt/eve_training/training_scripts/util/env5.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\util.py:/opt/eve_training/training_scripts/util/util.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\agent.py:/opt/eve_training/training_scripts/util/agent.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\action_curriculum.py:/opt/eve_training/training_scripts/util/action_curriculum.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\checkpoint_restore.py:/opt/eve_training/training_scripts/util/checkpoint_restore.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\buffer_filter.py:/opt/eve_training/training_scripts/util/buffer_filter.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\buckle_reward.py:/opt/eve_training/training_scripts/util/buckle_reward.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\snapshot.py:/opt/eve_training/training_scripts/util/snapshot.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\traj_record.py:/opt/eve_training/training_scripts/util/traj_record.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\heuristic_policy.py:/opt/eve_training/training_scripts/util/heuristic_policy.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\util\heuristic_controller.py:/opt/eve_training/training_scripts/util/heuristic_controller.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\util\diagnostics_logger.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/diagnostics_logger.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\util\probe_evaluator.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/probe_evaluator.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\util\experience_cache.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/experience_cache.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\algo\sac.py:/usr/local/lib/python3.8/dist-packages/eve_rl/algo/sac.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\network\gaussianpolicy.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/gaussianpolicy.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\network\component\mlp.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/component/mlp.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\agent\single.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/single.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\agent\singelagentprocess.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/singelagentprocess.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\agent\synchron.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/synchron.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\runner\runner.py:/usr/local/lib/python3.8/dist-packages/eve_rl/runner/runner.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\replaybuffer\vanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillashared.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\replaybuffer\vanillaepisode.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillaepisode.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\replaybuffer\vanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillastep.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\replaybuffer\pervanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillastep.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\replaybuffer\pervanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillashared.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\replaybuffer\replaybuffer.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/replaybuffer.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_rl\eve_rl\replaybuffer\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\env.py:/usr/local/lib/python3.8/dist-packages/eve/env.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\util\polyline.py:/usr/local/lib/python3.8/dist-packages/eve/util/polyline.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\util\pathcontext.py:/usr/local/lib/python3.8/dist-packages/eve/util/pathcontext.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/util/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\reward\arclengthprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/arclengthprogress.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\reward\waypointprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/waypointprogress.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\reward\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/reward/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\observation\localguidance.py:/usr/local/lib/python3.8/dist-packages/eve/observation/localguidance.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\observation\meshinvariant.py:/usr/local/lib/python3.8/dist-packages/eve/observation/meshinvariant.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\observation\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/observation/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\observation\centerlines2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/centerlines2d.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\observation\target2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/target2d.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\pathfinder\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\pathfinder\fixedpath.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/fixedpath.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\pathfinder\dijkstra2.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/dijkstra2.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\monoplanestatic.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/monoplanestatic.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\simulation\sofabeamadapter.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/simulation/sofabeamadapter.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\target\centerlinerandom.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/target/centerlinerandom.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\vesseltree\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\vesseltree\rccavariedfrommesh.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/rccavariedfrommesh.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\vesseltree\rccaprocedural.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/rccaprocedural.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\vesseltree\aorticarcharteries\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\vesseltree\aorticarcharteries\carotidsiphon.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\__init__.py:/opt/eve_training/eve_bench/eve_bench/__init__.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenav.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenavrccavaried.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenavrccavaried.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenavprocedural.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenavprocedural.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\archvariety.py:/usr/local/lib/python3.8/dist-packages/eve_bench/archvariety.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\archvariety.py:/opt/eve_training/eve_bench/eve_bench/archvariety.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\basicwirenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/basicwirenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\basicwirenav.py:/opt/eve_training/eve_bench/eve_bench/basicwirenav.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\saved:/opt/eve_training/results" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\training _scripts\eval_anatomies.py:/opt/eve_training/training_scripts/eval_anatomies.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve\eve\intervention\vesseltree\topbrainanatomyset.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/topbrainanatomyset.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenavtopbrain.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavtopbrain.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\eve_bench\eve_bench\dualdevicenavtopbrain.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavtopbrain.py" \
  -v "D:\neve\.claude\worktrees\rl_improv_18_p2\carotid_data:/opt/eve_training/carotid_data:ro" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/eval_anatomies.py \
    --checkpoint "$CKPT" \
    --n_episodes "$N_EPISODES" \
    --change_every "$CHANGE_EVERY" \
    --n_worker "$N_WORKER" \
    --seed_base "$SEED_BASE" \
    --snapshot_mode "$SNAPSHOT_MODE" \
    --max_steps "$MAX_STEPS" \
    $ARCH_FLAGS $ENV_FLAGS $EXTRA_FLAGS
#
# Output lands next to the checkpoint:
#   <ckpt-dir>/eval_anatomies_<ckpt-name>/
#     logs/         per-worker STEP logs (all pids interleave one file — the
#                   analyzer keys episode state by pid=, do not parse
#                   sequentially)
#     snapshots/    <mode>/<target>/<outcome>/ep###_pid#####_step###_R±#.##.png
#     episodes.csv  per-episode: path_len, section, steps, success
#
# The console report prints overall success + Wilson CI, the distinct-anatomy
# count (with a loud warning if it collapses to 1), per-anatomy success
# spread, and the CCA / ICA-mid / siphon depth split at the 146/210 mm cuts.
