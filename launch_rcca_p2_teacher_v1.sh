#!/bin/bash
# Gen-4 — RCCA P2 teacher v1 (RL_IMPROV_18, branch rl_improv_18_p2).
#
# WHY P2 (see RL_PARADIGM_ROADMAP.md; RLPD P1 closed negative 2026-07-16:
# evals 0.0/0.0 — from-scratch SAC cannot bootstrap this task at realized
# UTD 0.39; the start-state mean never left random-init while machinery
# stayed healthy). P2 removes the bootstrap problem instead of scaling it:
#
#   --residual_heuristic       env5.step composes a_total = clip(a_heur +
#                              scale * a_policy) in RAW units. The policy
#                              learns a RESIDUAL on the scripted
#                              CenterlineFollowerHeuristic — at init
#                              (mean~0) behavior = pure heuristic, so the
#                              run STARTS at heuristic competence.
#   --heur_action_obs          4-dim heuristic intent in the deployable
#                              prefix (obs 121 -> 125; policy 97 -> 101+24).
#   --privileged_actor         Scaffolder-style privileged ACTOR: policy
#                              consumes the FULL obs incl. the 24-dim
#                              privileged tail (slice becomes no-op).
#                              Teacher is NOT deployable — P2b distills a
#                              deployable obs-only student via DAgger.
#                              Requires --aux_coef 0 (labels are inputs).
#   --heatup_action_scale 0.3  heatup = heuristic + small residual noise:
#                              the buffer seeds at ~heuristic quality with
#                              action diversity (no cache; old seed is obs-
#                              and action-space incompatible).
#   --eval_after_pretrain      with pretrain 0 this is the PURE-HEURISTIC
#                              baseline eval on the held-out seeds — the
#                              run's reference point (runner hoist).
#   Kept from P1: --critic_layernorm + --no_entropy_backup (validated: Q
#   bounded for 226k updates with no BC anchor), sac alpha rails [-5, 0],
#   target_entropy 1.0. PER + balanced 0.3 clean-lane amplification is
#   BACK (no symmetric sampler here — success amplification helped v2).
#   shm 24g -> 30g (P1 hit an in-container OOM kill at ~87% of 23.5g).
#
# Decisive checks once running:
#   * BASELINE eval (before explore) = pure heuristic quality. Everything
#     after must beat it — the run's null hypothesis.
#   * explore success should START near the baseline (heatup logs show
#     heuristic+noise success immediately) — if it starts ~0, the residual
#     composition is broken, kill and debug.
#   * eval curve vs v2 (6.1/30.6/49.0/30.6) AND vs own baseline.
#   * Q-divergence watch as in P1 (LayerNorm should hold).
#   * mean|residual| growth in probe: small early (trusting heuristic),
#     growing where the heuristic fails (stuck states) = the design working.
# Monitor:  docker logs -f rcca_p2_teacher_v1  |  Stop: docker stop -t 60 rcca_p2_teacher_v1
set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_p2_teacher_v1 2>/dev/null || true

docker run --name rcca_p2_teacher_v1 --gpus all --shm-size=30g --init -d \
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
    -n rcca_p2_teacher_v1 \
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
    --balanced_fraction 0.3 \
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
    --replay_buffer_size 2000000 \
    --pretrain_updates 0 \
    --eval_after_pretrain \
    --target_branches "Centerline curve - RCCA.mrk" \
    --snapshots centerlines \
    -nw 16 -d cuda:0
#
# Monitor:  docker logs -f rcca_p2_teacher_v1   |   Stop:  docker stop -t 60 rcca_p2_teacher_v1
