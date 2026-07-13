#!/bin/bash
# Gen-4 — RCCA procedural v3a: v2 + the RL_IMPROV_16 Tier-A bundle
# (E1b + E2 + E3 + E4-prep). See RL_IMPROV_16_EXPERIMENTS.md.
#
# WHY v3a (v2 live-run investigation, saved/v2_micro_investigation/):
#   * E1b --awac_adv_norm_tau 2.0 — v2's AWAC weights spanned [0.72,1.25]
#     (adv std ~0.09 @ lambda=1.0) ~= uniform BC: the critic's correctly
#     signed retract-when-stuck preference could not transmit. Batch-
#     normalized advantages target weight p99/p1 ~12 (band 5-20x; watch
#     the new awac_weight_p99p1 CSV column).
#   * E2 --aux_labels "0,1,5,6" + --aux_label_znorm — v2's aux force
#     labels (rel 2/3) are dead at source (SOFA MechanicalObject.force is
#     a per-solve scratch buffer, zero at any post-step read; verified on
#     live-wire checkpoints) and the contact labels are scale-starved
#     (std ~1e-3 -> aux grad ~5e-8). Repoint to live channels (node
#     velocities rel 0/1 + contact rel 5/6) and z-score at loss time
#     (cache-safe). Watch the new aux_loss CSV column (~O(1) if working).
#   * E3 --stuck_fraction 0.15 — stuck states (slack>8.7mm / contact top
#     decile) are ~10%/1% of the buffer: gradient-starved. Third sampling
#     lane gives them ~15% of every batch (+30% clean lane, 55% general).
#   * E4-prep -e STUCK_CHECKPOINT_DIR — harvest a mesh-fingerprinted
#     stuck-state pool during the run (zero cost); screen it later with
#     launch_screen_stuck.sh for the v3b restore curriculum.
#
# GATES (see RL_IMPROV_16_EXPERIMENTS.md Tier A):
#   eval1 (~287k explore) >= v2's 30.6%; awac_weight_p99p1 in [5,20];
#   aux_loss O(1) and aux R^2 rising; stuck-lane hit-rate ~15%;
#   P(retract | slack-tail) probe rising (v2: flat/decaying).
#   Peel-off protocol: if a gate fails, relaunch minus the implicated
#   flag (all default-off; seed reuse makes relaunch ~25 min).
#
# PORTABILITY (machine 2): mounts below use the absolute repo path
# D:\Arjun\workspace
eve — adjust the prefix to the local checkout, or
# run:  sed -i 's|D:\Arjun\workspace\neve|<your path>|g' <this file>
# Also copy saved/rcca_proc_heatup/seed.npz (the 282k-transition seed) to
# <repo>/saved/rcca_proc_heatup/ — the run pretrains from it. Keep the
# host awake (the v1 deadlock trigger was a host suspend; the IPC guard
# only converts hangs into visible crashes).
#
# Unchanged from v2: alpha rails (-5,-2.3), action_mean_penalty 0.005,
# EVE_CLEAN_RAIL_MAX 0.15, IPC timeout guard, eval_after_pretrain
# baseline, procedural RCCA/RVA per-worker meshes, relax recovery,
# buckle 0.5, target_entropy 1.0, PER + balanced 0.3, seed + 10k
# pretrain, 16 workers, cuda:0.
set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_procedural_v3a 2>/dev/null || true

docker run --name rcca_procedural_v3a --gpus all --shm-size=24g --init -d \
  -e EVE_CLEAN_RAIL_MAX=0.15 \
  -e EVE_RL_MODEL_QUEUE_TIMEOUT_S=900 \
  -e EVE_RL_TRAINER_RESULT_TIMEOUT_S=1800 \
  -e EVE_RL_WATCHDOG_STALL_S=2400 \
  -e STUCK_CHECKPOINT_DIR=/opt/eve_training/results/rcca_v3_stuck \
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
    -n rcca_procedural_v3a \
    --procedural_rcca \
    --procedural_seed 12345 \
    --procedural_change_every 10 \
    --relax_failure_truncations \
    --buckle_reward_coef 0.5 \
    --aux_coef 0.05 \
    --aux_labels "0,1,5,6" \
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
    --log_alpha_max -2.3 \
    --action_mean_penalty 0.005 \
    --update_per_explore_step 0.5 \
    --replay_buffer_size 2000000 \
    --pretrain_updates 10000 \
    --eval_after_pretrain \
    --awac_adv_norm_tau 2.0 \
    --aux_label_znorm \
    --stuck_fraction 0.15 \
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
# Monitor:  docker logs -f rcca_procedural_v3a   |   Stop:  docker stop rcca_procedural_v3a
