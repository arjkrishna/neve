#!/bin/bash
# Gen-4 — RCCA procedural RLPD v1 (RL_IMPROV_17 P1, branch rl_improv_17_rlpd).
#
# WHY RLPD (paradigm pivot — see RL_PARADIGM_ROADMAP.md + AWAC_STABILITY_EVOLUTION.md):
# The AWAC penalty bracket is CLOSED: penalty 0.005 (v2) let the mean grow
# back into the stuck-grinder ceiling (49% -> 30.6% at eval3), penalty 0.02
# (v2b) froze the mean at pretrain (eval 0.0%). The constraint family
# (AWAC/BC-anchored) is the wrong class for our data mix per the
# three-regimes analysis. RLPD (Ball et al., ICML 2023) drops the BC/AWAC
# anchor ENTIRELY and instead stabilizes off-policy SAC on offline data
# with three mechanisms, all implemented default-off on this branch:
#
#   --critic_layernorm        LayerNorm after every hidden Linear in q1/q2
#                             (policy body legacy). The RLPD ablation's
#                             decisive stabilizer: bounds Q-extrapolation
#                             on OOD actions with NO behavior constraint —
#                             the policy is free to sharpen retract-when-
#                             stuck instead of BC-cloning the seed.
#   --no_entropy_backup       plain min-Q Bellman target (actor keeps its
#                             entropy term). RLPD: entropy-in-backup
#                             destabilizes sparse-ish reward tasks. AWAC
#                             already backed up entropy-free; this extends
#                             that to --algo sac.
#   --rlpd_offline_fraction 0.5  symmetric sampling: every batch = 50%
#                             uniform from the seed (is_demo) + 50% uniform
#                             from online data. REPLACES PER/balanced/stuck
#                             lanes (IS weights = 1). The seed cache is
#                             pushed with is_demo=True under this flag.
#   --algo sac                actor loss = standard SAC (entropy-regularized
#                             Q-max). No AWAC weights, no BC term.
#                             action_mean_penalty / entropy_beta / rail
#                             filter are AWAC-gated -> inert; NOT passed.
#   --pretrain_updates 0      RLPD does NO offline pretrain phase — offline
#                             SAC updates without a BC anchor are exactly
#                             the Q-divergence failure RLPD avoids. The
#                             seed enters purely through the 50/50 sampler.
#                             (--eval_after_pretrain dropped with it: a
#                             random-init baseline eval is 30 wasted min.)
#   --update_per_explore_step 1.0  UTD 1.0 (v2 was 0.5). Paper uses UTD 20
#                             with an E=10 critic ensemble; our 7.5Hz SOFA
#                             sim + single GPU can't feed that. DOCUMENTED
#                             DEVIATIONS: twin critics (no ensemble), UTD 1.
#   --log_alpha_min -5.0 --log_alpha_max 0.0  pure-SAC alpha rails: floor
#                             6.7e-3 (no decay-to-zero whipsaw), ceiling
#                             1.0 — SAC's actor NEEDS a working entropy
#                             term (unlike AWAC v2's 0.1 cap, which guarded
#                             a BC-anchored loss that no longer exists).
#
# Unchanged from v2: procedural RCCA/RVA (seed 12345, change every 10),
# relax recovery, buckle shaping 0.5, aux 0.05 on 2,3,5,6 (legacy labels —
# isolate the paradigm change; E2 repointing stays on machine 2's v3a),
# target_entropy 1.0, soft log_std (-2,0), grad_clip 1.0, 256x256, seed.npz
# (282k transitions, meta_buckle_coef 0.5), 16 workers, cuda:0, incremental
# buffer save + deadlock guards (trainer deadline 1800s, watchdog 2400s).
#
# NOTE checkpoint re-eval / probes: nets built WITHOUT --critic_layernorm
# cannot load these checkpoints (missing _norms keys). Pass the flag to any
# eval-only invocation. monitoring/probe_policy_v3.py builds the POLICY
# only (never LayerNormed) — unaffected. Its freeze-ratio baselines against
# the earliest snapshot = eval1 here (no pretrain checkpoint0 exists).
#
# Decisive checks once running:
#   * Q health: critic loss (losses CSV) stays O(1) and q-value columns do
#     NOT trend monotonically up — Q-divergence is THE known failure of
#     BC-free offline mixing; LayerNorm is the guard. Divergence = kill.
#   * alpha lives INSIDE [0.0067, 1.0] and moves — pure SAC must regulate
#     entropy toward target 1.0; pinned-at-floor for >50k updates = the
#     entropy term died (v1's precursor).
#   * explore success climbs from ~0 (no pretrain!) — RLPD's claim is fast
#     online recovery via the 50/50 seed mix. If explore success is still
#     ~0% by eval1, the recipe isn't transferring — reassess, don't tune.
#   * eval curve target: beat v2's 6.1 / 30.6 / 49.0 / 30.6, especially
#     PAST eval3 (the stuck-grinder ceiling this pivot is aimed at).
#   * buffer log prints symmetric-sampler activation once at start;
#     seed = 282,310 offline transitions (14% of 2M capacity).
#
# HOST SLEEP: keep the machine awake; the watchdog converts a hang into a
# visible exit(42), it does not survive one.

set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_rlpd_v1 2>/dev/null || true

docker run --name rcca_rlpd_v1 --gpus all --shm-size=24g --init -d \
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
    -n rcca_rlpd_v1 \
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
    --algo sac \
    --critic_layernorm \
    --no_entropy_backup \
    --rlpd_offline_fraction 0.5 \
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
    --target_branches "Centerline curve - RCCA.mrk" \
    --snapshots centerlines \
    -nw 16 -d cuda:0
#
# Monitor:  docker logs -f rcca_rlpd_v1   |   Stop:  docker stop -t 60 rcca_rlpd_v1
