#!/bin/bash
# Gen-4 — RCCA procedural v2b: the AWAC-STABILITY one-more-chance run.
# = launch_rcca_procedural_v2.sh + EXACTLY TWO knob changes, NO Tier-A Es.
# Full analysis + gates: AWAC_STABILITY_EVOLUTION.md.
#
# WHY: v2 climbed 6.1% -> 30.6% -> 49.0% then regressed to 30.6% at eval3.
# The late-run destabilizer was MEAN-SATURATION growth (entropy fell through
# target to -2.2, clamp_fraction 2%->13.8%) — the AWAC BC engine cloning its
# own increasingly-aggressive successes faster than the brakes could resist
# (alpha capped at 0.100, mean penalty 0.005 = too weak at eval3 scale).
# v2b strengthens ONLY the two brakes, via flags that already exist:
#   --action_mean_penalty 0.005 -> 0.02   (4x direct anti-saturation force)
#   --log_alpha_max      -2.3  -> -2.0    (alpha ceiling 0.100 -> 0.135;
#                                          still 2-3x below v1's 0.28-0.45
#                                          mean-crush zone; freeze probe
#                                          guards the over-brake direction)
# NO Tier-A flags (awac_adv_norm_tau / aux_label_znorm / stuck_fraction all
# absent -> those code paths are byte-identical to v2). Deadlock guards
# (trainer-result deadline + progress watchdog) are default-on + explicit.
#
# GATES (AWAC_STABILITY_EVOLUTION.md §4): entropy >= -0.5 sustained;
# clamp <= 10%; probe ratio >= 2x and not falling; eval2 >= 45%;
# TARGET: eval3 >= 40-45% at speed >= 3mm/s (v2: 30.6%).
# Peel rules: probe < 1.25x -> revert alpha ceiling; clamp > 12% sustained
# -> raise penalty to 0.03.
set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_procedural_v2b 2>/dev/null || true

docker run --name rcca_procedural_v2b --gpus all --shm-size=24g --init -d \
  -e EVE_CLEAN_RAIL_MAX=0.15 \
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
    -n rcca_procedural_v2b \
    --procedural_rcca \
    --procedural_seed 12345 \
    --procedural_change_every 10 \
    --relax_failure_truncations \
    --buckle_reward_coef 0.5 \
    --aux_coef 0.05 \
    --aux_labels "2,3,5,6" \
    --heatup_cache_file /opt/eve_training/results/rcca_proc_heatup/seed.npz \
    --target_entropy 1.0 \
    --replay_mode step \
    --per \
    --algo awac \
    --awac_lambda 1.0 \
    --balanced_fraction 0.3 \
    --grad_clip 1.0 \
    --hidden 256 256 \
    --embedder_layers 0 \
    --learning_rate 0.0003 \
    --log_std_min -2 \
    --log_std_max 0.0 \
    --log_alpha_min -5.0 \
    --log_alpha_max -2.0 \
    --action_mean_penalty 0.02 \
    --update_per_explore_step 0.5 \
    --replay_buffer_size 2000000 \
    --pretrain_updates 10000 \
    --eval_after_pretrain \
    --target_branches "Centerline curve - RCCA.mrk" \
    --snapshots centerlines \
    -nw 16 -d cuda:0
#
# Decisive checks once running (beyond the v1 list):
#   * alpha stays inside [0.0067, 0.100] (losses CSV col 7) — never 4.5e-5,
#     never > 0.1.
#   * awac_weight_mean DEVIATES from 1.00 with nonzero spread (weight_max
#     2-6) — discrimination restored. If still pinned at 1.00, lambda needs
#     to go lower.
#   * CLEAN_RAIL_FILTER lines appear occasionally in the buffer subprocess
#     log (a few % of successes rejected is healthy; >30% = policy railing).
#   * DETERMINISTIC probe (monitor): start-state mean|tanh(mu0)| stays
#     > 0.10 — the v1 freeze fell 0.255 -> 0.089. Eval speed >= 3 mm/s.
# Monitor:  docker logs -f rcca_procedural_v2b   |   Stop:  docker stop rcca_procedural_v2b
