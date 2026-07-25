#!/bin/bash
# Gen-4 — RCCA P2 teacher v1d (RL_IMPROV_18, branch rl_improv_18_p2).
#
# v1d = v1c (crunchpass lane) + the MACHINE-2 v3c REWARD PAIR:
#   --progress_tip_mode avg --avg_gw_weight 0.5
#       Tip-average progress: pays progress_factor * delta(0.5*s_gw +
#       0.5*s_cath). A parked guidewire halves the pay; advancing the
#       TRAILING device is paid; the catheter-forward+gw-retract
#       telescoping gait nets ~0. Off-path the avg tracker FREEZES (the
#       machine-2 BLOCKER: rebaselining creates a farmable pump, ~+5/ep).
#   --cath_slack_coef 0.5
#       Catheter-slack potential (dead-band 15mm, cap 150mm, [-1,0]):
#       prices the coil the tip terms can't see. Machine-2 validated knot
#       detector (precision .97/recall .96 @ >50mm).
# WHY HERE: the SAME pathology verified on v1b — catheter leads the gw by
# >50mm on 69% of late steps, max cath insertion 820mm (vs the ~898 hard
# cap machine 2 measured); frontier-only pay + unpriced cath slack select
# catheter-shove. Machine-2 result: eval1 50%->62.2% (AWAC line).
# All ported per the handoff (saved/p2a_deep_dive/Handoff — the v3c
# REWARD PAIR*.md), 6/6 tests incl. the pump regression
# (tests/test_v3c_reward.py). Reward-version stamps + 4-field cache
# guards active (no cache used here — P2 heatups fresh).
#
# ATTRIBUTION CAVEAT (two new levers vs v1b): v1d bundles crunchpass +
# reward pair. Machine-2's evidence covers the pair alone (on AWAC);
# crunchpass alone is unproven. If v1d regresses, peel to v1c (lane only)
# or a pair-only variant before concluding anything about either lever.
#
# KNOWN RISK (handoff §7): both machine-2 pair runs peaked at eval1 then
# declined via an endogenous critic instability (q1 -> -90). WATCH
# q1_mean (losses CSV col 12): sustained descent past ~-2.5 = the critic
# is diverging and the peak is behind. (v1b's q1_mean dipped to -2.7 and
# RECOVERED under LayerNorm + entropy-free backup — this line may be more
# robust than machine 2's AWAC, but watch it.)
#
# Baseline note (CORRECTED from handoff §6 — their dip rides on a 10k
# PRETRAIN pathway this line does not have): H0 here is the pure
# reward-blind heuristic, so H0 SUCCESS must land in v1b's ~37-47% band
# — outside it = the port leaked into dynamics/obs, STOP. Only the H0
# REWARD number reads lower (cosmetic, new scale). The §6 early dip
# applies AFTER learning starts (explore/eval1: shove de-incentivized
# before the two-device gait replaces it). Judge at eval1+, vs v1b's
# ~61% explore plateau and 62% eval-avg.
# Monitor: docker logs -f rcca_p2_teacher_v1d | Stop: docker stop -t 60 rcca_p2_teacher_v1d
set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_p2_teacher_v1d 2>/dev/null || true

docker run --name rcca_p2_teacher_v1d --gpus all --shm-size=30g --init -d \
  -e EVE_RL_MODEL_QUEUE_TIMEOUT_S=900 \
  -e EVE_RL_TRAINER_RESULT_TIMEOUT_S=1800 \
  -e EVE_RL_WATCHDOG_STALL_S=2400 \
  -v "D:\Arjun\workspace\neve\training _scripts\DualDeviceNav_train.py:/opt/eve_training/training_scripts/DualDeviceNav_train.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\env.py:/opt/eve_training/training_scripts/util/env.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\env2.py:/opt/eve_training/training_scripts/util/env2.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\env3.py:/opt/eve_training/training_scripts/util/env3.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\env4.py:/opt/eve_training/training_scripts/util/env4.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\env5.py:/opt/eve_training/training_scripts/util/env5.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\util.py:/opt/eve_training/training_scripts/util/util.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\agent.py:/opt/eve_training/training_scripts/util/agent.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\action_curriculum.py:/opt/eve_training/training_scripts/util/action_curriculum.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\checkpoint_restore.py:/opt/eve_training/training_scripts/util/checkpoint_restore.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\buffer_filter.py:/opt/eve_training/training_scripts/util/buffer_filter.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\buckle_reward.py:/opt/eve_training/training_scripts/util/buckle_reward.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\snapshot.py:/opt/eve_training/training_scripts/util/snapshot.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\heuristic_policy.py:/opt/eve_training/training_scripts/util/heuristic_policy.py" \
  -v "D:\Arjun\workspace\neve\training _scripts\util\heuristic_controller.py:/opt/eve_training/training_scripts/util/heuristic_controller.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\util\diagnostics_logger.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/diagnostics_logger.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\util\probe_evaluator.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/probe_evaluator.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\util\experience_cache.py:/usr/local/lib/python3.8/dist-packages/eve_rl/util/experience_cache.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\algo\sac.py:/usr/local/lib/python3.8/dist-packages/eve_rl/algo/sac.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\network\gaussianpolicy.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/gaussianpolicy.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\network\component\mlp.py:/usr/local/lib/python3.8/dist-packages/eve_rl/network/component/mlp.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\agent\single.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/single.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\agent\singelagentprocess.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/singelagentprocess.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\agent\synchron.py:/usr/local/lib/python3.8/dist-packages/eve_rl/agent/synchron.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\runner\runner.py:/usr/local/lib/python3.8/dist-packages/eve_rl/runner/runner.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\replaybuffer\vanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillashared.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\replaybuffer\vanillaepisode.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillaepisode.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\replaybuffer\vanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/vanillastep.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\replaybuffer\pervanillastep.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillastep.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\replaybuffer\pervanillashared.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/pervanillashared.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\replaybuffer\replaybuffer.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/replaybuffer.py" \
  -v "D:\Arjun\workspace\neve\eve_rl\eve_rl\replaybuffer\__init__.py:/usr/local/lib/python3.8/dist-packages/eve_rl/replaybuffer/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\env.py:/usr/local/lib/python3.8/dist-packages/eve/env.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\util\polyline.py:/usr/local/lib/python3.8/dist-packages/eve/util/polyline.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\util\pathcontext.py:/usr/local/lib/python3.8/dist-packages/eve/util/pathcontext.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\util\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/util/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\reward\arclengthprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/arclengthprogress.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\reward\waypointprogress.py:/usr/local/lib/python3.8/dist-packages/eve/reward/waypointprogress.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\reward\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/reward/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\observation\localguidance.py:/usr/local/lib/python3.8/dist-packages/eve/observation/localguidance.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\observation\meshinvariant.py:/usr/local/lib/python3.8/dist-packages/eve/observation/meshinvariant.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\observation\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/observation/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\observation\centerlines2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/centerlines2d.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\observation\target2d.py:/usr/local/lib/python3.8/dist-packages/eve/observation/target2d.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\pathfinder\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\pathfinder\fixedpath.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/fixedpath.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\pathfinder\dijkstra2.py:/usr/local/lib/python3.8/dist-packages/eve/pathfinder/dijkstra2.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\monoplanestatic.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/monoplanestatic.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\simulation\sofabeamadapter.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/simulation/sofabeamadapter.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\target\centerlinerandom.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/target/centerlinerandom.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\rccavariedfrommesh.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/rccavariedfrommesh.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\rccaprocedural.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/rccaprocedural.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\aorticarcharteries\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\aorticarcharteries\carotidsiphon.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\__init__.py:/opt/eve_training/eve_bench/eve_bench/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenav.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenav.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenav.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavrccavaried.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavrccavaried.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavprocedural.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavprocedural.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\archvariety.py:/usr/local/lib/python3.8/dist-packages/eve_bench/archvariety.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\archvariety.py:/opt/eve_training/eve_bench/eve_bench/archvariety.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\basicwirenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/basicwirenav.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\basicwirenav.py:/opt/eve_training/eve_bench/eve_bench/basicwirenav.py" \
  -v "D:\Arjun\workspace\neve\saved:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
    --env_version 5 \
    -n rcca_p2_teacher_v1d \
    --procedural_rcca \
    --procedural_seed 12345 \
    --procedural_change_every 10 \
    --relax_failure_truncations \
    --buckle_reward_coef 0.5 \
    --aux_coef 0.0 \
    --residual_heuristic \
    --residual_scale 1.0 \
    --heur_action_obs \
    --privileged_actor \
    --heatup_action_scale 0.3 \
    --target_entropy 1.0 \
    --replay_mode step \
    --per \
    --balanced_fraction 0.2 \
    --crunchpass_fraction 0.2 \
    --crunchpass_engage_thresh 0.25 \
    --crunchpass_radius_thresh 0.175 \
    --cath_slack_coef 0.5 \
    --progress_tip_mode avg \
    --avg_gw_weight 0.5 \
    --algo sac \
    --critic_layernorm \
    --no_entropy_backup \
    --grad_clip 1.0 \
    --hidden 256 256 \
    --embedder_layers 0 \
    --learning_rate 0.0003 \
    --log_std_min -2 \
    --log_std_max 0.0 \
    --log_alpha_min -5.0 \
    --log_alpha_max 0.0 \
    --update_per_explore_step 1.0 \
    --replay_buffer_size 1500000 \
    --pretrain_updates 0 \
    --eval_after_pretrain \
    --target_branches "Centerline curve - RCCA.mrk" \
    --snapshots centerlines \
    -nw 16 -d cuda:0
#
# Monitor:  docker logs -f rcca_p2_teacher_v1d   |   Stop:  docker stop -t 60 rcca_p2_teacher_v1d
