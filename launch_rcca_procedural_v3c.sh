#!/bin/bash
# RL_IMPROV_16 (v3c) — rcca_procedural_v3c: v3a + the human-approved REWARD
# PAIR, nothing else. v3c vs v3a differs ONLY in reward => clean
# attribution.
#
#   --progress_tip_mode avg + --avg_gw_weight 0.5 : progress grades the
#       average arc of BOTH device tips. v3a forensic: frontier-only
#       progress made catheter-only telescoping a full-pay strategy and
#       left gw-push vs gw-retract reward-TIED at the divergence step
#       that separates every choke passage from failure.
#   --cath_slack_coef 0.5 : catheter-slack potential channel
#       (util/buckle_reward.py cath_slack_potential; deadband 15mm, cap
#       150mm). v3a measured 700+mm coils forming at net ~0 reward and
#       knotting at the RVA shelf; cath_slack>50mm = knot (prec 0.97).
#
# SEED: fresh harvest REQUIRED first —
#   bash launch_rcca_harvest_v3c.sh   (~3h; writes
#   saved/rcca_proc_heatup_v3c/seed.npz with the full reward-version
#   stamp; this launcher fails fast if the stamps mismatch).
#
# TRAINER LEVERS (implemented, default-OFF, reserved for RUN 2 so run 1
# isolates the reward effect — add to the flags below to enable):
#   --awac_mode_adv_norm            (per-contact-mode E1b adv norm)
#   --contact_mean_penalty 0.05     (contact-gated anti-rail)
#
# GATES (eval2-3): eval-failure median gw insertion > 20mm (v3a:
#   0.1-1.3); cath_slack in failures collapsing (v3a: ~740mm); >=1
#   never-solved seed flipped by eval3; Quality >= own pretrain baseline.
#   If gw deploys but no seed flips by eval4: scripted demos -> E7 ->
#   two-actor.
#
# Unchanged from v3a: E1b/E2/E3 flags, alpha rails, guard timeouts
# (5400/7200), procedural meshes, eval protocol (ostium starts, 98
# seeds). Keep this file LF-only (the autocrlf trap) and never rewrite
# paths with sed patterns containing backslash-n sequences.
set -e
export MSYS_NO_PATHCONV=1

# Review fix — fail fast when the v3c seed is absent: a missing
# --heatup_cache_file silently degrades to a random-heatup run.
SEED="saved/rcca_proc_heatup_v3c/seed.npz"
if [ ! -f "$SEED" ]; then
  echo "v3c seed missing at $SEED - run: bash launch_rcca_harvest_v3c.sh" >&2
  exit 1
fi

docker rm rcca_procedural_v3c 2>/dev/null || true

docker run --name rcca_procedural_v3c --gpus all --shm-size=24g --init -d \
  -e EVE_CLEAN_RAIL_MAX=0.15 \
  -e EVE_RL_MODEL_QUEUE_TIMEOUT_S=900 \
  -e EVE_RL_TRAINER_RESULT_TIMEOUT_S=5400 \
  -e EVE_RL_WATCHDOG_STALL_S=7200 \
  -e STUCK_CHECKPOINT_DIR=/opt/eve_training/results/rcca_v3c_stuck \
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
    -n rcca_procedural_v3c \
    --procedural_rcca \
    --procedural_seed 12345 \
    --procedural_change_every 10 \
    --relax_failure_truncations \
    --buckle_reward_coef 0.5 \
    --aux_coef 0.05 \
    --aux_labels "0,1,5,6" \
    --heatup_cache_file /opt/eve_training/results/rcca_proc_heatup_v3c/seed.npz \
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
    --cath_slack_coef 0.5 \
    --progress_tip_mode avg \
    --avg_gw_weight 0.5 \
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
# Monitor:  docker logs -f rcca_procedural_v3c   |   Stop:  docker stop rcca_procedural_v3c
