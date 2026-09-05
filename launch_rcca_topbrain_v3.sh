#!/bin/bash
# ============================================================================
# v1 TOPBRAIN = BYTE-IDENTICAL v1bp teacher with ONE variable changed:
#   WHERE THE ANATOMY COMES FROM.
#     v1bp:      --procedural_rcca --procedural_seed 12345
#                --procedural_change_every 10
#                (one shipped patient, RCCA perturbed sinusoidally per worker)
#     this run:  --topbrain family (22 REAL patient anatomies, fixed meshes)
# Algorithm, reward, observation, devices, network, replay, PER, entropy
# rails, worker count and every other flag are UNCHANGED and must stay so —
# any eval delta vs v1bp is then attributable to the anatomy source alone.
# Three mounts are ADDED (the loader module, the bench env, the mesh data);
# every v1bp mount is retained verbatim.
# ============================================================================
# WHY. v1bp's variation is synthetic: one patient's RCCA displaced by a
# procedural perturbation, so the policy sees one anatomy's statistics with
# noise on top. The siphon ceiling may be an artifact of that. TopBrain
# supplies 25 genuinely different patient right-ICA siphons, so "generalize
# across patients" becomes a measurable claim instead of an assumption.
#
# THE ANATOMY SET (build record: TOPBRAIN_PIPELINE.md).
#   Each anatomy = the SHIPPED host tree (aortic arch, brachiocephalic trunk,
#   cervical vessel) with a real patient right-ICA grafted on at 130 mm
#   arclength from the ostium (237.5 mm shipped RCCA - 106 mm median TopBrain
#   ICA). Frame-matched at the junction (tangent AND superior axis), radii
#   held exactly as measured with the host ramped to meet them, meshes BAKED
#   to vessel_architecture_collision.obj so an anatomy folder reproduces
#   byte-identically anywhere. Route length 201-263 mm (host 238).
#
#   25 built, 3 EXCLUDED, 22 retained:
#     topcow_mr_015  distal 22 mm pinches shut — 22 consecutive centerline
#                    points OUTSIDE the mesh, up to 7.19 mm beyond the wall,
#                    11% of its targets unreachable. Surface truncated ~20 mm
#                    short of the centerline. It still runs in SOFA without
#                    complaint, which is why the enclosure check was written.
#     topcow_mr_013  centerline-outside-mesh, borderline (5 points, 1.37 mm)
#     topcow_mr_014  centerline-outside-mesh, borderline (3 points, 2.11 mm)
#   Retained 22: 001 002 003 004 005 006 007 008 010 011 012 016 017 018
#                020 021 022 023 024 025 026 027
#
# TRAIN / HOLDOUT. 18 train, 4 held out:
#   HOLDOUT = topcow_mr_007 topcow_mr_008 topcow_mr_017 topcow_mr_022
#   The holdout patients are NEVER seen by any explore worker. Eval therefore
#   measures generalization to UNSEEN PATIENTS, not to unseen seeds of one
#   patient (which is all v1bp's held-out seed base-1 could measure). The
#   train-average vs holdout gap is the generalization number for the paper.
#   All 4 holdouts are clean anatomies: none needed an RVA repair, an anchor
#   trim, or carries a residual blockage — so the gap is not confounded by
#   geometry defects on the eval side.
#
# THE DECISIVE STARTUP CHECK — H0, and the kill criterion.
#   --pretrain_updates 0 with --eval_after_pretrain means the FIRST eval runs
#   before a single gradient step. With --residual_heuristic and a mean-~0
#   init, behavior at that moment IS the pure scripted
#   CenterlineFollowerHeuristic. That first eval number is therefore the
#   PURE-HEURISTIC (H0) baseline on the TopBrain holdout.
#     * On the procedural set that band was 37-47%.
#     * ON TOPBRAIN IT IS UNKNOWN. It has never been measured. Do NOT assume
#       37-47% transfers — these are different vessels with different
#       curvature and 201-263 mm routes.
#   OPERATOR: record the first eval success as this run's NULL HYPOTHESIS and
#   write it into the run log before anything else. Every later eval is
#   judged against THAT number, not against v1bp's.
#   KILL CRITERION: explore success must START near the measured H0 band —
#   heatup is heuristic + small residual noise, so it succeeds immediately or
#   the composition is broken. Explore success starting near ~0 while H0 is
#   materially above 0 means a_total = clip(a_heur + scale*a_policy) is not
#   composing on these meshes (wrong frame, wrong insertion, bad heuristic
#   port). KILL THE RUN AND DEBUG — do not let it train through it.
#
# KNOWN CAVEAT — a small success floor may exist on two train anatomies.
#   Under exact signed distance (TOPBRAIN_ANATOMY_AUDIT.md):
#     topcow_mr_024  2 blocked stations, a 7 mm run at s=194-201 mm
#     topcow_mr_027  1 blocked station, terminal station only
#   Targets sampled at those stations may be unreachable, so per-anatomy
#   success on 024/027 can be capped below 100% for reasons that are not the
#   policy's fault. Both are TRAIN-side only (neither is in the holdout), so
#   the eval/generalization number is clean; only the explore-success trace
#   carries the floor. 20/25 anatomies have zero blocked stations.
#
# CARRIED OVER FROM v1bp — do not touch (rationale in that file):
#   --residual_heuristic --residual_scale 1.0   residual on the scripted
#                              heuristic in RAW units; run starts at
#                              heuristic competence instead of at random.
#   --heur_action_obs          4-dim heuristic intent in the deployable
#                              prefix (obs 121 -> 125).
#   --privileged_actor         privileged ACTOR (full obs incl. the 24-dim
#                              privileged tail). Teacher is NOT deployable —
#                              P2b distills an obs-only student via DAgger.
#                              Requires --aux_coef 0 (labels are inputs).
#   --heatup_action_scale 0.3  buffer seeds at ~heuristic quality with action
#                              diversity (no cache reuse; obs-space differs).
#   --critic_layernorm --no_entropy_backup, sac alpha rails [-5, 0],
#   --target_entropy 1.0, PER + --balanced_fraction 0.3, replay 2M,
#   v3c reward pair --progress_tip_mode avg --avg_gw_weight 0.5
#   --cath_slack_coef 0.5.
#   WATCH (handoff §7): q1_mean sustained < -2.5 = critic diverging.
#   Startup must echo BOTH [v3c] lines AND the TopBrain anatomy roster.
#
# MOUNT NOTE. Per TOPBRAIN_PIPELINE.md Fix D5, TopBrainAnatomySet is NOT
# exported from eve/intervention/vesseltree/__init__.py — 10 existing
# launchers mount that __init__ without mounting the new module and would
# fail at import. The env imports the module path DIRECTLY. Do not "tidy"
# this by adding it to __init__.py.
#
#
# FLAG CONTRACT expected of DualDeviceNav_train.py (--topbrain family; the
# parallel branch TOPBRAIN_PIPELINE.md 'Not done' calls for). These map 1:1
# onto DualDeviceNavTopBrain's kwargs � if the trainer's names differ, fix
# THIS FILE, not the trainer:
#   --topbrain               switch (mirrors --procedural_rcca). Env v5 only.
#   --topbrain_dir DIR       -> anatomy_dir
#   --topbrain_seed N        -> seed, worker i uses base + i
#   --topbrain_change_every N-> episodes_between_change
#   --topbrain_exclude A B C -> exclude (train AND eval both drop these)
#   --topbrain_holdout A B C -> added to the TRAIN env's exclude, and passed
#                               as `only` to the EVAL env. This is what makes
#                               the eval number a patient-level generalization
#                               number.
# --topbrain and --procedural_rcca are mutually exclusive.
#
# Monitor:  docker logs -f rcca_topbrain_v3  |  Stop: docker stop -t 60 rcca_topbrain_v3
set -e
export MSYS_NO_PATHCONV=1

docker rm rcca_topbrain_v3 2>/dev/null || true

docker run --name rcca_topbrain_v3 --gpus all --shm-size=30g --init -d \
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
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\topbrainanatomyset.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/topbrainanatomyset.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\aorticarcharteries\__init__.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve\eve\intervention\vesseltree\aorticarcharteries\carotidsiphon.py:/usr/local/lib/python3.8/dist-packages/eve/intervention/vesseltree/aorticarcharteries/carotidsiphon.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\__init__.py:/opt/eve_training/eve_bench/eve_bench/__init__.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenav.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenav.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenav.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavrccavaried.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavrccavaried.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavrccavaried.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavprocedural.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavprocedural.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavprocedural.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavtopbrain.py:/usr/local/lib/python3.8/dist-packages/eve_bench/dualdevicenavtopbrain.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\dualdevicenavtopbrain.py:/opt/eve_training/eve_bench/eve_bench/dualdevicenavtopbrain.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\archvariety.py:/usr/local/lib/python3.8/dist-packages/eve_bench/archvariety.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\archvariety.py:/opt/eve_training/eve_bench/eve_bench/archvariety.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\basicwirenav.py:/usr/local/lib/python3.8/dist-packages/eve_bench/basicwirenav.py" \
  -v "D:\Arjun\workspace\neve\eve_bench\eve_bench\basicwirenav.py:/opt/eve_training/eve_bench/eve_bench/basicwirenav.py" \
  -v "D:\Arjun\workspace\neve\topbrain_data:/opt/eve_training/topbrain_data:ro" \
  -v "D:\Arjun\workspace\neve\saved:/opt/eve_training/results" \
  eve-training-fixed \
  python3 /opt/eve_training/training_scripts/DualDeviceNav_train.py \
    --env_version 5 \
    -n rcca_topbrain_v3 \
    --topbrain \
    --topbrain_dir /opt/eve_training/topbrain_data/anatomies_v3 \
    --topbrain_seed 12345 \
    --topbrain_change_every 10 \
    --topbrain_holdout topcow_mr_007 topcow_mr_007_L topcow_mr_008 topcow_mr_008_L topcow_mr_017 topcow_mr_017_L topcow_mr_022 topcow_mr_022_L \
    --topbrain_exclude __none__ \
    --topbrain_target_min_arclength 133.0 \
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
    --replay_buffer_size 2000000 \
    --pretrain_updates 0 \
    --eval_after_pretrain \
    --target_branches "Centerline curve - RCCA.mrk" \
    --snapshots centerlines \
    -nw 16 -d cuda:0
#
# Monitor:  docker logs -f rcca_topbrain_v3   |   Stop:  docker stop -t 60 rcca_topbrain_v3
