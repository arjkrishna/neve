import os
import logging
import argparse
import numpy as np
import torch.multiprocessing as mp
import torch
from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.env2 import BenchEnv2  # NEW: Centerline-aware environment with waypoint rewards
from util.env3 import BenchEnv3  # NEW: Tuned rewards for better balance
from util.env4 import BenchEnv4  # Arclength progress + local guidance
from util.env5 import BenchEnv5  # Optimized env4 with shared projection cache
from eve.observation import PrivilegedState  # Gen-4 asymmetric-critic tail


def _parse_aux_labels(spec: str):
    """Parse --aux_labels: comma-separated ints RELATIVE to the privileged
    tail (e.g. "6,21,20" = max contact proxy, slip, fold counter). "" -> None."""
    if not spec:
        return None
    return [int(tok) for tok in str(spec).split(",") if tok.strip() != ""]
from util.agent import BenchAgentSynchron
from eve_rl import Runner
from eve_bench import DualDeviceNav


RESULTS_FOLDER = os.getcwd() + "/results/eve_paper/neurovascular/full/mesh_ben"

# Target branches for DualDeviceNav (4 supra-aortic branches)
TARGET_BRANCHES = [
    "Centerline curve - LCCA.mrk",
    "Centerline curve - LVA.mrk",
    "Centerline curve - RCCA.mrk",
    "Centerline curve - RVA.mrk",
]


def build_episode_schedule(n_episodes, branches, base_seed=42, heuristic_mode=False):
    """Build a branch-balanced, reproducibly-shuffled episode schedule.

    Returns a list of (seed, options) tuples where branches are assigned
    round-robin so each branch gets exactly n_episodes // len(branches)
    episodes (plus remainder distributed across the first branches).

    Args:
        n_episodes: Total number of episodes to schedule.
        branches: List of target branch names.
        base_seed: Base seed for reproducible seed generation.
        heuristic_mode: If True, add heuristic_mode=True to options
            (enables env-side abort detectors during heuristic seeding).

    Returns:
        List of (seed, options) tuples, one per episode.
    """
    rng = np.random.default_rng(base_seed)
    schedule = []
    for i in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        branch = branches[i % len(branches)]
        options = {"target_branch": branch}
        if heuristic_mode:
            options["heuristic_mode"] = True
        schedule.append((ep_seed, options))
    # Shuffle to avoid workers getting only one branch each,
    # but deterministically so runs are reproducible.
    rng.shuffle(schedule)
    return schedule

EVAL_SEEDS = "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
EVAL_SEEDS = EVAL_SEEDS.split(",")
EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS]
HEATUP_STEPS = 2e4  # 20k steps
TRAINING_STEPS = 2e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 2.5e5


GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20


LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.DEBUG

# HEATUP_STEPS = 5e3
# TRAINING_STEPS = 1e7
# CONSECUTIVE_EXPLORE_EPISODES = 10
# EXPLORE_STEPS_BTW_EVAL = 7.5e3
# EVAL_SEEDS = list(range(20))
# RESULTS_FOLDER = os.getcwd() + "/results/test"
# BATCH_SIZE = 8
# UPDATE_PER_EXPLORE_STEP = 1 / 200


def main(args):
    """Trainer entry. Split out of `if __name__ == "__main__":` so
    other entry points (e.g., per-daughter wrapper scripts, future
    step-version-RL variants, eval-only invocations) can call this
    after building their own argparse Namespace."""

    # Plan v9 — in 'sofa_restore' mode the two flags are COMPLEMENTARY,
    # not exclusive: --insertion_z is the start for heuristic-seeding
    # episodes (which bypass restore) and the fallback when the pool is
    # still empty, while --checkpoint_dir is the restore source for
    # heatup/online episodes. Only enforce mutual exclusivity in the
    # legacy z345 mode.
    if (
        args.checkpoint_dir
        and args.insertion_z is not None
        and getattr(args, "rl_start_mode", "z345") != "sofa_restore"
    ):
        raise SystemExit(
            "--checkpoint_dir and --insertion_z are mutually exclusive "
            "outside --rl_start_mode sofa_restore: checkpoint restore "
            "replaces the entire wire state, while anchoring high in the "
            "trunk is a clean alternative. (In sofa_restore mode they are "
            "complementary — insertion_z drives heuristic seeding, "
            "checkpoint_dir drives online restore.)"
        )

    # Plan v9 Change 8 — validate --rl_start_mode against --checkpoint_dir.
    if getattr(args, "rl_start_mode", "z345") == "sofa_restore" and not args.checkpoint_dir:
        raise SystemExit(
            "--rl_start_mode sofa_restore requires --checkpoint_dir to "
            "point at the pool of pre-bif(11) checkpoints (built by "
            "Plan v9 Change 5b: heuristic run with PRE_BIF11_CHECKPOINT_DIR "
            "env var set)."
        )
    if args.checkpoint_dir and getattr(args, "rl_start_mode", "z345") == "z345":
        print(
            "[Plan v9] --checkpoint_dir is set but --rl_start_mode is 'z345'; "
            "online episodes WILL be SOFA-restored from the pool. To start "
            "at z=345 instead, remove --checkpoint_dir."
        )

    # Plan v5 — apply per-daughter overrides from CLI. Use a local name
    # (`target_branches`) instead of rebinding the module-level
    # TARGET_BRANCHES — Python would otherwise mark TARGET_BRANCHES
    # function-local for the entire main() body, raising UnboundLocalError
    # at the schedule-builder when --target_branches is omitted.
    if args.target_branches:
        target_branches = [b.strip() for b in args.target_branches.split(",") if b.strip()]
        print(f"[Plan v5] TARGET_BRANCHES overridden: {target_branches}")
    else:
        target_branches = TARGET_BRANCHES

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    stochastic_eval = args.stochastic_eval
    lr = args.learning_rate
    hidden_layers = args.hidden
    embedder_nodes = args.embedder_nodes
    embedder_layers = args.embedder_layers
    env_version = args.env_version
    worker_device = torch.device("cpu")

    # Plan v6 — replay-mode-conditional buffer/batch/update hyperparameters.
    # Local variables (NOT module-global rebind — that would make the names
    # function-local everywhere in main() and risk UnboundLocalError, the
    # same trap as the §11 target_branches fix). "episode" keeps the legacy
    # values for the multi-daughter LSTM setup; "step" uses canonical
    # step-SAC values (1e6-transition buffer, 256 batch, 1 update/step).
    if args.replay_mode == "step":
        replay_buffer_size = (
            int(args.replay_buffer_size)
            if getattr(args, "replay_buffer_size", 0) and args.replay_buffer_size > 0
            else 1_000_000
        )
        batch_size = 256
        update_per_explore_step = 1.0
        print(f"[Plan v6] replay_mode=step — buffer={replay_buffer_size} transitions, "
              "batch=256, update_per_explore_step=1.0")
    else:
        replay_buffer_size = REPLAY_BUFFER_SIZE
        batch_size = BATCH_SIZE
        update_per_explore_step = UPDATE_PER_EXPLORE_STEP

    # Plan v8 — --update_per_explore_step overrides the replay-mode
    # default. Resolved HERE, before the PER beta schedule and buffer
    # construction below, which need the effective update:explore ratio.
    if args.update_per_explore_step is not None:
        update_per_explore_step = args.update_per_explore_step
        print(f"[Plan v8] update_per_explore_step overridden = "
              f"{update_per_explore_step}")

    # Plan v7 — PER is an orthogonal on/off switch on step mode. It is
    # step-only: enabling it with episode mode is a configuration error.
    if args.per and args.replay_mode != "step":
        raise ValueError(
            "--per requires --replay_mode step (PER is a step-buffer feature)."
        )
    # beta anneals beta_start → 1.0 over the whole training run. beta
    # advances once per UPDATE (buffer.sample call), NOT per explore
    # step, and total updates = TRAINING_STEPS * update_per_explore_step
    # — so the denominator must be scaled by the effective update ratio
    # (RL_IMPROV_10 B2; a bare TRAINING_STEPS denominator never lets
    # beta reach 1.0 at update ratios < 1).
    per_beta_steps = float(TRAINING_STEPS) * update_per_explore_step
    if args.per:
        print(f"[Plan v7] PER enabled — alpha={args.per_alpha}, "
              f"beta_start={args.per_beta_start}, beta_steps={per_beta_steps:.0f}")

    # Plan v8 — stabilization-suite knobs. The PER-buffer knobs
    # (demo_priority_bonus / priority_mode / balanced_fraction) are
    # step-only and need PER (except balanced_fraction, which also works on
    # the uniform step buffer).
    if args.replay_mode != "step":
        if args.demo_priority_bonus > 0 or args.priority_mode != "td" \
                or args.balanced_fraction > 0:
            raise ValueError(
                "--demo_priority_bonus / --priority_mode / --balanced_fraction "
                "require --replay_mode step."
            )
    if (args.demo_priority_bonus > 0 or args.priority_mode != "td") and not args.per:
        raise ValueError(
            "--demo_priority_bonus and --priority_mode != td require --per."
        )
    if args.grad_clip > 0:
        print(f"[Plan v8] gradient clipping enabled — max-norm {args.grad_clip}")
    if args.algo == "awac":
        print(f"[Plan v8] algo=AWAC — awac_lambda={args.awac_lambda}")
    if args.balanced_fraction > 0:
        print(f"[Plan v8] balanced two-stream sampling — "
              f"clean fraction {args.balanced_fraction}")
    if args.priority_mode != "td" or args.demo_priority_bonus > 0:
        print(f"[Plan v8] PER priority_mode={args.priority_mode}, "
              f"demo_priority_bonus={args.demo_priority_bonus}")

    # Select environment class based on version
    if env_version == 5:
        EnvClass = BenchEnv5
        print("Using BenchEnv5 (optimized: shared projection cache + logging fixes)")
    elif env_version == 4:
        EnvClass = BenchEnv4
        print("Using BenchEnv4 (arclength progress reward + local guidance observation)")
    elif env_version == 3:
        EnvClass = BenchEnv3
        print("Using BenchEnv3 (TUNED waypoint rewards: 5mm spacing, 1.0 increment, -0.001 step penalty)")
    elif env_version == 2:
        EnvClass = BenchEnv2
        print("Using BenchEnv2 (waypoint rewards + centerline observations)")
    else:
        EnvClass = BenchEnv
        print("Using BenchEnv (original PathDelta reward)")

    custom_parameters = {
        "lr": lr,
        "hidden_layers": hidden_layers,
        "embedder_nodes": embedder_nodes,
        "embedder_layers": embedder_layers,
        "env_version": env_version,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": batch_size,
        "UPDATE_PER_EXPLORE_STEP": update_per_explore_step,
        "replay_mode": args.replay_mode,
        "per": args.per,
        "per_alpha": args.per_alpha,
        "per_beta_start": args.per_beta_start,
        "grad_clip": args.grad_clip,
        "algo": args.algo,
        "awac_lambda": args.awac_lambda,
        "demo_priority_bonus": args.demo_priority_bonus,
        "priority_mode": args.priority_mode,
        "balanced_fraction": args.balanced_fraction,
    }

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
        _config_folder_dup,
        diagnostics_folder,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=trial_name, create_diagnostics=True
    )

    # CRITICAL: Set STEP_LOG_DIR BEFORE agent is created, so worker processes inherit it
    # Workers are spawned in BenchAgentSynchron.__init__, which happens before Runner.__init__
    if diagnostics_folder is not None:
        logs_subprocesses = os.path.join(diagnostics_folder, "logs_subprocesses")
        os.makedirs(logs_subprocesses, exist_ok=True)
        os.environ["STEP_LOG_DIR"] = logs_subprocesses
        print(f"Set STEP_LOG_DIR={logs_subprocesses}")

    # Plan v5 — set SNAPSHOT_MODE / SNAPSHOT_DIR BEFORE the agent spawns
    # workers (same spawn-inheritance constraint as STEP_LOG_DIR above).
    # env5 renders an end-of-episode PNG bucketed by training phase
    # (seed / eval / explore). prune_training_snapshots.py applies the
    # keep-policy post-hoc: all seed + all eval + 10-best/10-worst per
    # 100 explore episodes.
    if args.snapshots and args.snapshots.lower() not in ("none", "off"):
        if diagnostics_folder is not None:
            snap_dir = os.path.join(diagnostics_folder, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            os.environ["SNAPSHOT_MODE"] = args.snapshots
            os.environ["SNAPSHOT_DIR"] = snap_dir
            # Plan v12 — sparse per-physical-episode snapshot rate for the
            # MultiTargetEnv5 driver (full snapshots always taken for
            # physically-reached-daughter episodes regardless of this rate).
            os.environ["SNAPSHOT_EVERY"] = str(getattr(args, "snapshot_every", 10))
            print(f"Set SNAPSHOT_MODE={args.snapshots} SNAPSHOT_DIR={snap_dir} "
                  f"SNAPSHOT_EVERY={os.environ['SNAPSHOT_EVERY']}")
        else:
            print("WARNING: --snapshots set but no diagnostics_folder; snapshots disabled")

    # Plan v9 Change 8b — restore-start snapshots written into the MAIN
    # snapshots dir as a sibling phase bucket (snapshots/restore_start/...)
    # alongside seed / eval / explore, so they browse together. Set env
    # vars BEFORE agent worker spawn (spawn-inheritance). Works even when
    # --snapshots is off (independent flag) — we point at the same base
    # snapshots dir and force the render mode.
    if getattr(args, "restore_start_snapshots", False):
        if diagnostics_folder is not None:
            rs_dir = os.path.join(diagnostics_folder, "snapshots")
            os.makedirs(rs_dir, exist_ok=True)
            os.environ["RESTORE_START_SNAPSHOT_DIR"] = rs_dir
            os.environ["RESTORE_START_SNAPSHOT_MODE"] = "centerlines"
            print(
                f"[Plan v9 Change 8b] Set RESTORE_START_SNAPSHOT_DIR={rs_dir} "
                "(restore-start snapshots -> snapshots/restore_start/, "
                "sibling to seed/eval/explore)"
            )
        else:
            print(
                "WARNING: --restore_start_snapshots set but no "
                "diagnostics_folder; restore-start snapshots disabled"
            )

    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Diagnostics configuration for SAC training monitoring
    diagnostics_config = {
        "enabled": True,
        "diagnostics_folder": diagnostics_folder,
        "log_losses_every_n_steps": 1,
        "log_probe_values_every_n_steps": 100,
        "log_batch_samples_every_n_steps": 100,  # Log batch samples every 100 gradient steps
        "n_batch_samples": 3,  # Number of transitions to sample from each batch
        "tensorboard_enabled": True,
        "probe_plot_every_n_steps": 10000,
        "policy_snapshot_every_n_steps": 10000,
        "flush_every_n_rows": 100,
    }

    # NOTE: DualDeviceNav() uses hardcoded defaults from eve_bench/eve_bench/dualdevicenav.py:
    #   - rotation_yzx_deg = [90, -90, 0]  (line 36)
    #   - fluoroscopy_rot_zx = [20, 5]     (line 82)
    # These match the preprocessing in the original DualDeviceNav data (model 0105)
    # Plan v5 — single-daughter scoping. When exactly one target branch is
    # configured (per-daughter training, e.g. RCCA-only), pass it as the
    # env's default_target_branch. The eve_rl runner resets the env with no
    # options for heatup / explore / eval; without this default those
    # phases would sample random targets across all 4 daughters. Heuristic
    # seeding still passes target_branch explicitly via its schedule.
    # Only BenchEnv5 supports the default_target_branch kwarg.
    default_tb = target_branches[0] if len(target_branches) == 1 else None
    env_kwargs = {}
    if args.env_version == 5 and default_tb is not None:
        env_kwargs["default_target_branch"] = default_tb
        print(f"[Plan v5] single-daughter scoping: default_target_branch={default_tb}")
    # Gen-4 recovery training — fold-stall / off-path detectors keep
    # counting (obs features + stuck-pool triggers) but no longer truncate
    # RL episodes; MaxSteps/VesselEnd/SimError end them. Applied to BOTH
    # env_train and env_eval (same MDP; eval failures now run to MaxSteps
    # — slower eval, consistent measurement). Heuristic-mode aborts are
    # unaffected either way.
    if args.env_version == 5 and getattr(args, "relax_failure_truncations", False):
        env_kwargs["relax_failure_truncations"] = True
        print("[Gen-4] relax_failure_truncations: fold/off-path detectors "
              "no longer truncate RL episodes (counters + stuck-pool only)")
    # Gen-4 privileged reward shaping — anti-buckle potential (gw slack +
    # SOFA contact proxy, delta form; util/buckle_reward.py). Applied to
    # train AND eval (same MDP; success metric unaffected — the term only
    # reshapes per-step reward, telescoping to phi_end - phi_start).
    _buckle_coef = float(getattr(args, "buckle_reward_coef", 0.0) or 0.0)
    if args.env_version == 5 and _buckle_coef != 0.0:
        env_kwargs["buckle_reward_coef"] = _buckle_coef
        print(f"[Gen-4] anti-buckle potential shaping ON: "
              f"coef={_buckle_coef} (slack + contact channels)")
    # Reward-version stamp for experience caches: every save (including the
    # worker-side rolling heatup flushes, which inherit this env var) embeds
    # the coef its rewards were scored under; the cache-load guards below
    # fail fast on a mismatch. Set BEFORE any worker process is created.
    os.environ["EVE_RL_BUCKLE_COEF"] = repr(_buckle_coef)

    # Gen-4 procedural anatomy — per-worker RCCA->siphon variation. Each
    # worker gets a distinctly-seeded DualDeviceNavProcedural that
    # re-randomizes the vessel every --procedural_change_every episodes.
    # env_train here is a representative instance (used for network sizing /
    # config only; the workers get factory(i)). Full-trunk from z=root, NO
    # restore (a regenerating mesh invalidates restore checkpoints), NO
    # insertion_z. Mutually exclusive with --checkpoint_dir.
    procedural_env_factory = None
    if getattr(args, "procedural_rcca", False):
        if args.env_version != 5:
            raise ValueError("--procedural_rcca requires --env_version 5.")
        if args.checkpoint_dir:
            # Gen-4 (#3) — procedural + restore is now mesh-SAFE. Stuck
            # checkpoints are tagged with the mesh they were captured on
            # (env5._save_stuck_checkpoint) and CheckpointRestoreWrapper
            # pins each worker's vessel tree to the checkpoint's exact mesh
            # (pin_next -> regenerate_to_fingerprint) BEFORE the SOFA restore,
            # so dof_positions land in the geometry they came from. The pool
            # MUST be a mesh-fingerprinted stuck harvest (from a prior
            # --procedural_rcca STUCK_CHECKPOINT_DIR run); untagged/fixed-mesh
            # checkpoints are correctly ineligible on a procedural tree and
            # the episode falls through to the ostium start. Under restore the
            # pin overrides the per-worker seed, so the checkpoint (not the
            # worker id) dictates the mesh and procedural_change_every is inert.
            print(
                "[Gen-4 #3] --procedural_rcca + --checkpoint_dir: mesh-matched "
                "recovery-curriculum restore ON — each worker's tree is pinned "
                "to the picked checkpoint's mesh. Pool must be a "
                "mesh-fingerprinted stuck harvest (else nothing is eligible "
                "and episodes start at the ostium)."
            )
        # Correct design: keep the LOADED arch fixed, vary ONLY the RCCA,
        # start FIXED at the RCCA ostium. DualDeviceNavRCCAVaried re-meshes
        # the loaded centerlines with a per-worker-perturbed RCCA (same
        # vessel-CS frame as DualDeviceNav -> obs-compatible / warm-startable;
        # the wire never navigates the arch, so re-meshing it is immaterial).
        from eve_bench import DualDeviceNavRCCAVaried
        _proc_base_seed = int(getattr(args, "procedural_seed", 12345))
        _proc_change = int(getattr(args, "procedural_change_every", 10))

        # Factory returns a FULL env (BenchEnv5 wrapping a distinctly-seeded
        # RCCA-varied intervention). Worker i -> seed base+i.
        def procedural_env_factory(worker_id):
            interv = DualDeviceNavRCCAVaried(
                seed=_proc_base_seed + worker_id,
                episodes_between_change=_proc_change,
            )
            env = BenchEnv5(
                intervention=interv, mode="train", visualisation=False,
                **env_kwargs
            )
            # Gen-4 (#3) — per-worker mesh-matched restore for the procedural
            # recovery curriculum. The wrapper must sit on the WORKER'S own
            # env (not the master template that line ~507 wraps for sizing),
            # so pin_next regenerates THIS worker's tree in-process. Distinct
            # rng_seed per worker so the per-episode checkpoint picks diverge.
            if args.checkpoint_dir:
                from util.checkpoint_restore import CheckpointRestoreWrapper
                env = CheckpointRestoreWrapper(
                    env,
                    checkpoint_dir=args.checkpoint_dir,
                    rng_seed=42 + worker_id,
                )
            return env
        # Representative instance for network sizing / config save only.
        intervention = DualDeviceNavRCCAVaried(
            seed=_proc_base_seed, episodes_between_change=_proc_change
        )
        print(
            f"[Gen-4] varied-RCCA (loaded arch fixed, start@ostium): "
            f"base_seed={_proc_base_seed}, change_every={_proc_change} eps, "
            f"per-worker seeds {_proc_base_seed}..{_proc_base_seed + n_worker - 1}"
        )
    else:
        intervention = DualDeviceNav(insertion_z=args.insertion_z)

    # Plan v12 Stage 1 — multi-target heatup harvester. When
    # --multi_target_heatup is set, swap BenchEnv5 (single-target) for
    # MultiTargetEnv5 (4 virtual envs sharing the SOFA backend). Stage 0
    # calibration probe and any non-Plan-v12 use of env_train remain
    # single-target. The flag is mutually exclusive with --checkpoint_dir
    # (Plan v12 trains from z=345, NOT from a restore pool) — enforced
    # below.
    if getattr(args, "multi_target_heatup", False):
        if args.checkpoint_dir:
            raise ValueError(
                "--multi_target_heatup is incompatible with --checkpoint_dir "
                "(Plan v12 trains from z=345, NOT from restore pool — see "
                "Plan v12 Decision #1 + 'Why 98 not 54' in Context). Drop "
                "--checkpoint_dir."
            )
        if args.env_version != 5:
            raise ValueError(
                "--multi_target_heatup requires --env_version 5 "
                "(MultiTargetEnv5 lives in env5.py)."
            )
        from util.env5 import MultiTargetEnv5
        # Determine primary target from --target_branches (first listed)
        # if any, defaulting to RCCA otherwise. Secondaries auto-fill the
        # remaining 3 daughters.
        primary_short = "RCCA"
        if default_tb:
            for tag in ("RCCA", "LCCA", "RVA", "LVA"):
                if tag in default_tb:
                    primary_short = tag
                    break
        env_train_unwrapped = MultiTargetEnv5(
            intervention=intervention,
            mode="train",
            visualisation=False,
            primary_target_short=primary_short,
            # Harvest transitions must be scored under the training reward.
            buckle_reward_coef=_buckle_coef,
        )
        print(
            f"[Plan v12] MultiTargetEnv5: primary={primary_short}, "
            f"secondaries={env_train_unwrapped.secondary_target_shorts}"
        )
    else:
        env_train_unwrapped = EnvClass(
            intervention=intervention, mode="train", visualisation=False,
            **env_kwargs
        )
    env_train = env_train_unwrapped  # May be wrapped below

    # Wrap with action curriculum if requested
    if args.curriculum:
        from util.action_curriculum import ActionCurriculumWrapper
        env_train = ActionCurriculumWrapper(
            env_train_unwrapped,
            stage1_steps=args.curriculum_stage1,
            stage2_steps=args.curriculum_stage2,
        )
        print(f"Action curriculum enabled: Stage1={args.curriculum_stage1}, Stage2={args.curriculum_stage2}")

    # Wrap with per-episode SOFA-state restore if requested. Goes outside
    # the curriculum wrapper so the curriculum sees the restored-state env.
    if args.checkpoint_dir:
        from util.checkpoint_restore import CheckpointRestoreWrapper
        env_train = CheckpointRestoreWrapper(
            env_train,
            checkpoint_dir=args.checkpoint_dir,
            rng_seed=42,
        )
        print(
            f"Checkpoint restore enabled: {len(env_train.checkpoint_files)} "
            f".npz files in {args.checkpoint_dir}"
        )

    if getattr(args, "procedural_rcca", False):
        # Held-out RCCA anatomy: a FIXED varied-RCCA vessel whose seed is
        # disjoint from every worker's (base-1) and which never
        # re-randomizes during a run. The train-worker-average vs held-out
        # gap is the generalization metric.
        from eve_bench import DualDeviceNavRCCAVaried
        intervention_eval = DualDeviceNavRCCAVaried(
            seed=int(getattr(args, "procedural_seed", 12345)) - 1,
            episodes_between_change=10 ** 9,
        )
    else:
        intervention_eval = DualDeviceNav(insertion_z=args.insertion_z)
    env_eval = EnvClass(
        intervention=intervention_eval, mode="eval", visualisation=False, **env_kwargs
    )
    env_eval_unwrapped = env_eval  # keep for config save (before any wrapper)
    # Plan v10 C4 — eval must start from the SAME restore states as training
    # (else eval measures the z=345 full task, a train/eval mismatch). Wrap
    # env_eval with the same restorer (distinct rng_seed so its per-episode
    # checkpoint picks differ from training's).
    if args.checkpoint_dir:
        from util.checkpoint_restore import CheckpointRestoreWrapper
        env_eval = CheckpointRestoreWrapper(
            env_eval,
            checkpoint_dir=args.checkpoint_dir,
            rng_seed=43,
        )
        print(
            f"[Plan v10] eval restore enabled: {len(env_eval.checkpoint_files)} "
            f".npz files in {args.checkpoint_dir} (eval starts from the 5 states)"
        )
    agent = BenchAgentSynchron(
        trainer_device,
        worker_device,
        lr,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        GAMMA,
        batch_size,
        REWARD_SCALING,
        replay_buffer_size,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        n_worker,
        stochastic_eval,
        False,
        diagnostics_config=diagnostics_config,
        replay_mode=args.replay_mode,
        per=args.per,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_steps=per_beta_steps,
        grad_clip=args.grad_clip,
        algo=args.algo,
        awac_lambda=args.awac_lambda,
        demo_priority_bonus=args.demo_priority_bonus,
        priority_mode=args.priority_mode,
        balanced_fraction=args.balanced_fraction,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        # RL_IMPROV_10 B3 — auto-alpha entropy setpoint; None -> SAC
        # default (-n_actions).
        target_entropy=getattr(args, "target_entropy", None),
        # Plan v11 anti-rail (AWAC). All-zero / empty -> None (disabled).
        entropy_beta_per_dim=(
            list(args.entropy_beta_per_dim)
            if getattr(args, "entropy_beta_per_dim", None)
            and any(b > 0 for b in args.entropy_beta_per_dim)
            else None
        ),
        action_mean_penalty=float(getattr(args, "action_mean_penalty", 0.0)),
        # RL_IMPROV_15 collapse forensics — log_alpha clamp rails. v1 froze
        # via alpha decay-to-floor (-10) -> whipsaw recovery -> entropy term
        # crushing the action mean (std was ceiling-pinned, so the mean was
        # the only entropy lever). Tighter rails keep the entropy term
        # alive-but-bounded. Defaults preserve legacy (-10, 2).
        log_alpha_min=float(getattr(args, "log_alpha_min", -10.0)),
        log_alpha_max=float(getattr(args, "log_alpha_max", 2.0)),
        # Gen-4 asymmetric critic — env5's ObsDict appends a privileged
        # tail (PrivilegedState, LAST key); the critics consume the full
        # flat obs while the policy is built (total - tail) wide and
        # slices internally. env v4 and earlier have no tail (dim 0).
        privileged_obs_dim=(
            PrivilegedState.N_DIMS if args.env_version == 5 else 0
        ),
        # Gen-4 aux heads — indices RELATIVE to the privileged tail; the
        # policy predicts those privileged values from its deployable
        # prefix (MSE, weight aux_coef). Empty/0.0 = off.
        aux_coef=float(getattr(args, "aux_coef", 0.0)),
        aux_label_rel_indices=_parse_aux_labels(
            getattr(args, "aux_labels", "")
        ),
        # Gen-4 procedural anatomy — per-worker env factory (None unless
        # --procedural_rcca). Worker i gets a distinctly-seeded vessel.
        env_train_factory=procedural_env_factory,
    )

    # Save config from unwrapped env (ActionCurriculumWrapper has no save_config)
    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train_unwrapped.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval_unwrapped.save_config(env_eval_config)
    infos = list(env_eval_unwrapped.info.info.keys())
    runner = Runner(
        agent=agent,
        # Heatup sampling bounds are PHYSICAL (mm/s, rad/s); the sampled
        # action is inverse-normalized against the live env action space
        # before storage (eve_rl single.py random_action). Symmetric
        # translation bounds (with the 2.4 symmetric action space) make
        # the random harvest retract as often as it advances — retraction
        # transitions were previously rare (bounds [-10,+30], mean +10).
        # Trade-off: a zero-mean walk penetrates less deeply; if harvest
        # depth suffers, re-bias low toward e.g. -15 rather than -30.
        heatup_action_low=[[-30.0, -1.5], [-30.0, -1.5]],
        heatup_action_high=[[30.0, 1.5], [30.0, 1.5]],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
        diagnostics_folder=diagnostics_folder,
        policy_snapshot_every_steps=10000,
    )
    runner_config = os.path.join(config_folder, "runner.yml")
    runner.save_config(runner_config)

    if args.eval_only_checkpoint:
        print(f"[eval-only] loading checkpoint: {args.eval_only_checkpoint}")
        eval_seeds = EVAL_SEEDS
        if args.drop_restore_states:
            if not args.checkpoint_dir:
                raise ValueError(
                    "--drop_restore_states requires --checkpoint_dir (the wrapper's "
                    "sorted file list defines the seed → state mapping)"
                )
            import glob as _glob
            sorted_files = sorted(_glob.glob(os.path.join(args.checkpoint_dir, "*.npz")))
            dropped_tokens = [t.strip() for t in args.drop_restore_states.split(",") if t.strip()]
            print(f"[canonical-eval] restore pool ({len(sorted_files)} files): "
                  f"{[os.path.basename(p) for p in sorted_files]}")
            print(f"[canonical-eval] dropping states whose filename contains any of: {dropped_tokens}")

            def _seed_picks_dropped(seed):
                idx = int(np.random.default_rng(seed).integers(0, len(sorted_files)))
                fname = os.path.basename(sorted_files[idx])
                return any(token in fname for token in dropped_tokens)

            kept = []
            dropped_summary = {}
            for seed in EVAL_SEEDS:
                if _seed_picks_dropped(seed):
                    idx = int(np.random.default_rng(seed).integers(0, len(sorted_files)))
                    fname = os.path.basename(sorted_files[idx])
                    dropped_summary[fname] = dropped_summary.get(fname, 0) + 1
                else:
                    kept.append(seed)
            eval_seeds = kept
            print(f"[canonical-eval] EVAL_SEEDS filtered: {len(EVAL_SEEDS)} → {len(eval_seeds)} "
                  f"(dropped {len(EVAL_SEEDS) - len(eval_seeds)})")
            for fname, count in dropped_summary.items():
                print(f"[canonical-eval]   dropped {count} seeds mapping to {fname}")
        agent.load_checkpoint(args.eval_only_checkpoint)
        runner._replay_save_interval = 999999
        quality, reward = runner.eval(seeds=eval_seeds)
        print(f"[eval-only] result: quality={quality} reward={reward}")
        agent.close()
        return

    # Optional: seed replay buffer with heuristic episodes
    # Collects episodes using a centerline-following heuristic and pushes
    # them into the agent's replay buffer BEFORE SAC training starts.
    # This avoids the early "collapse to masked no-op" failure mode.
    # Uses parallel workers for fast seeding (same infrastructure as explore).
    if args.heuristic_seeding > 0 and env_version in (4, 5):
        from eve_rl.replaybuffer import EpisodeReplay
        from eve_rl.util.experience_cache import save_episodes_npz, load_episodes_npz

        # Try to load from cache first
        if args.heuristic_cache_file and os.path.isfile(args.heuristic_cache_file):
            print(f"Loading heuristic cache from {args.heuristic_cache_file}...")
            episodes_tuples, _, cache_meta = load_episodes_npz(
                args.heuristic_cache_file
            )
            # Gen-3 obs change (guidance 30->39, flat 78->87) — a cache built
            # under an older observation layout loads silently here and only
            # crashes at the first update batch with an opaque matmul shape
            # error far from the cause. Fail fast at load time instead.
            if episodes_tuples:
                try:
                    # Gen-4 — compare against the CRITIC width (full flat obs incl. the
                    # privileged tail); the policy is now the SLICED width and would
                    # false-fail a valid cache.
                    _expected_obs = int(agent.algo.model.q1.n_observations)
                except Exception:
                    _expected_obs = None
                _cache_obs = int(np.asarray(episodes_tuples[0][0]).shape[-1])
                if _expected_obs and _cache_obs != _expected_obs:
                    raise ValueError(
                        f"Heuristic cache obs dim {_cache_obs} != network obs "
                        f"dim {_expected_obs} ({args.heuristic_cache_file}). "
                        "The cache was harvested under a different observation "
                        "layout — regenerate it with the current code."
                    )
            # Gen-4 — reward-version guard. Cached REWARDS are baked at
            # harvest; the obs-dim check above cannot catch a
            # buckle_reward_coef mismatch (identical layout, different
            # scoring). A silent mismatch mixes two reward MDPs in one
            # buffer, biasing the critic/advantages. Absent stamp = 0.0
            # (pre-buckle cache) — valid only for a coef=0 run.
            from eve_rl.util.experience_cache import cache_buckle_coef
            _cache_coef = cache_buckle_coef(args.heuristic_cache_file)
            if abs(_cache_coef - _buckle_coef) > 1e-9:
                raise ValueError(
                    f"Heuristic cache reward version mismatch: cache scored "
                    f"with buckle_reward_coef={_cache_coef}, this run uses "
                    f"{_buckle_coef} ({args.heuristic_cache_file}). Pass "
                    f"--buckle_reward_coef {_cache_coef} or re-harvest."
                )
            n_pushed = 0
            for i, ep_tuple in enumerate(episodes_tuples):
                flat_obs, actions, rewards, terminals = ep_tuple
                # Plan v8 — every heuristic-cache episode IS a demo (blanket
                # tag, robust even if the npz predates the metadata format).
                # reached_target_daughter / episode_return come from the
                # per-episode metadata when present, else default.
                m = cache_meta[i] if cache_meta is not None else {}
                replay_ep = EpisodeReplay(
                    flat_obs=list(flat_obs),
                    actions=list(actions),
                    rewards=list(rewards),
                    terminals=list(terminals),
                    is_demo=True,
                    episode_return=float(m.get("episode_return", 0.0)),
                    reached_target_daughter=bool(
                        m.get("reached_target_daughter", False)
                    ),
                )
                agent.replay_buffer.push(replay_ep)
                n_pushed += 1
            print(f"Heuristic cache loaded: {n_pushed} episodes pushed to replay buffer.")
        else:
            # Parallel heuristic seeding with minimum success rate guarantee
            import math
            import importlib

            N = args.heuristic_seeding
            min_successes = math.ceil(args.min_success_rate * N)
            max_total = N * args.max_seeding_multiplier

            # Plan v5 — allow per-daughter override of the heuristic factory.
            if args.heuristic_factory:
                mod_path, cls_name = args.heuristic_factory.rsplit(":", 1)
                factory_cls = getattr(importlib.import_module(mod_path), cls_name)
                print(f"[Plan v5] heuristic factory overridden: {args.heuristic_factory}")
            else:
                from util.heuristic_policy import HeuristicActionFunctionFactory
                factory_cls = HeuristicActionFunctionFactory

            factory = factory_cls(
                noise_std=0.0,
                normalize_output=True,
            )

            # Plan v5 — allow per-daughter seeding-success criterion.
            # "term"         = TargetReached terminal (default, all daughters)
            # "clean_thread" = received_correct AND NOT received_wrong at the
            #                  final daughter fork (per-daughter looser)
            _seeding_success_mode = args.seeding_success

            def _is_success(ep):
                """Check episode success via info dict per --seeding_success mode."""
                if not ep.infos:
                    return bool(ep.terminals[-1]) if ep.terminals else False
                info = ep.infos[-1]
                if _seeding_success_mode == "clean_thread":
                    # Plan v9 Change 2 — strict final-state check.
                    # Old filter used the ever-touched latch
                    # `reached_target_daughter` AND excluded any episode
                    # that *ever* wrong-committed — wrongly disqualifying
                    # legitimate RVA-detour-then-RCCA episodes.
                    # New definition: "clean thread" = at episode end the
                    # wire's current branch IS the target daughter
                    # (final_branch_idx == target_daughter_branch_idx).
                    # Detours along the way are fine; only the final
                    # location matters.
                    final_idx = info.get("final_branch_idx")
                    target_idx = info.get("target_daughter_branch_idx")
                    if final_idx is None or target_idx is None:
                        return False
                    return int(final_idx) == int(target_idx)
                # default: TargetReached terminal
                if "success" in info:
                    return bool(info["success"])
                return bool(ep.terminals[-1]) if ep.terminals else False

            all_episodes = []
            batch_num = 0
            seed_offset = 0
            # Dedicated RNG for failure sampling — derived from base seed
            # so the final selected set is fully reproducible.
            selection_rng = np.random.default_rng(42 + 999)

            while True:
                batch_num += 1
                if batch_num == 1:
                    batch_size = N
                else:
                    # Deficit-based: estimate how many more episodes needed
                    n_success_so_far = sum(1 for ep in all_episodes if _is_success(ep))
                    needed = min_successes - n_success_so_far
                    observed_rate = max(n_success_so_far / len(all_episodes), 0.05)
                    batch_size = math.ceil(needed / observed_rate * 1.5)
                    batch_size = min(batch_size, max_total - len(all_episodes))

                if batch_size <= 0:
                    break

                schedule = build_episode_schedule(
                    batch_size, target_branches, base_seed=42 + seed_offset,
                    # Plan v10 — fork-heuristic: when --heuristic_from_restore,
                    # do NOT set heuristic_mode, so the CheckpointRestoreWrapper
                    # is NOT bypassed → the heuristic runs from the restored
                    # fork state (restore-at-fork) instead of z=345. (Trade-off:
                    # env-side abort detectors are off; fine for short fork runs.)
                    heuristic_mode=(not args.heuristic_from_restore),
                )
                seed_offset += batch_size

                batch_episodes = agent.heuristic_seed(
                    episodes=batch_size,
                    heuristic_factory=factory,
                    episode_schedule=schedule,
                    push_to_buffer=False,
                )
                all_episodes.extend(batch_episodes)

                n_success = sum(1 for ep in all_episodes if _is_success(ep))
                batch_success = sum(1 for ep in batch_episodes if _is_success(ep))
                print(f"  Batch {batch_num}: {len(batch_episodes)} episodes, "
                      f"{batch_success} successes | "
                      f"Total: {len(all_episodes)} episodes, {n_success} successes "
                      f"({100*n_success/len(all_episodes):.1f}%)")

                if n_success >= min_successes:
                    break
                if len(all_episodes) >= max_total:
                    print(f"  WARNING: hit max seeding cap ({max_total} episodes) "
                          f"with only {n_success} successes")
                    break

            # Filter: keep all successes + enough failures for healthy mix
            successes = [ep for ep in all_episodes if _is_success(ep)]
            failures = [ep for ep in all_episodes if not _is_success(ep)]

            n_success = len(successes)
            # Ensure success ratio <= 70%: pad with failures if needed
            min_failures = math.ceil(n_success / 0.7) - n_success
            # Also fill up to at least N total
            n_failures_needed = max(min_failures, N - n_success)
            n_failures_needed = max(n_failures_needed, 0)

            if n_failures_needed > 0 and len(failures) > 0:
                n_sample = min(n_failures_needed, len(failures))
                sampled_failures = list(
                    selection_rng.choice(failures, size=n_sample, replace=False)
                )
            else:
                sampled_failures = []

            to_push = successes + sampled_failures
            for ep in to_push:
                # Plan v11 — tag live-seeded heuristic episodes as demos for
                # PER's demo_priority_bonus. Matches the cache-load path
                # (line ~493) and cache-save metadata (line ~659) which
                # already tag is_demo=True. Without this, V3-style runs
                # (--heuristic_seeding ... --heuristic_from_restore, no
                # --heuristic_cache_file) end up with 0 demo-tagged
                # transitions in the buffer (the regression observed
                # comparing V2 18.5% demo-tagged vs V3 0%).
                ep.is_demo = True
                agent.replay_buffer.push(ep)

            print(f"Heuristic seeding complete: pushing {n_success} successes + "
                  f"{len(sampled_failures)} failures = {len(to_push)} episodes "
                  f"({100*n_success/len(to_push):.1f}% success rate)")

            # Save cache — saves the final selected set, not all attempts.
            # Plan v8 — also persist per-episode quality metadata so a
            # cache-loaded run carries the signals the stabilization-suite
            # samplers need (clean-thread flag, return; all are demos).
            if args.save_heuristic_cache and to_push:
                episodes_to_save = []
                cache_metadata = []
                for ep in to_push:
                    episodes_to_save.append((
                        np.array(ep.flat_obs),
                        np.array(ep.actions),
                        np.array(ep.rewards),
                        np.array(ep.terminals),
                    ))
                    ep_info = ep.infos[-1] if getattr(ep, "infos", None) else {}
                    cache_metadata.append({
                        "episode_return": float(getattr(ep, "episode_reward", 0.0)),
                        "reached_target_daughter": bool(
                            ep_info.get("reached_target_daughter", False)
                        ),
                        "is_demo": True,
                    })
                os.makedirs(os.path.dirname(args.save_heuristic_cache) or ".", exist_ok=True)
                save_episodes_npz(
                    args.save_heuristic_cache, episodes_to_save,
                    metadata=cache_metadata,
                )
                print(f"Saved heuristic cache to {args.save_heuristic_cache}")

    # Gen-4 harvester — --seed_only: after heuristic seeding + cache save,
    # exit BEFORE heatup/training. The saved --save_heuristic_cache .npz is
    # the deliverable (a Gen-4 121-dim RCCA demo seed for a later training
    # run to load via --heuristic_cache_file). Combine with --procedural_rcca
    # to harvest on the per-worker VARIED anatomy (on-distribution demos), or
    # run on the fixed mesh for the reliably-threading tuned heuristic.
    if getattr(args, "seed_only", False):
        if not args.save_heuristic_cache:
            print("[seed-only] WARNING: --save_heuristic_cache not set; the "
                  "harvested seed will be lost on exit.")
        print("[seed-only] harvest + cache save complete; exiting before "
              "heatup/training.")
        agent.close()
        return

    # Load heatup cache if provided (skips heatup phase)
    heatup_steps_effective = HEATUP_STEPS
    if args.heatup_cache_file and os.path.isfile(args.heatup_cache_file):
        from eve_rl.replaybuffer import EpisodeReplay
        from eve_rl.util.experience_cache import load_episodes_npz

        print(f"Loading heatup cache from {args.heatup_cache_file}...")
        episodes_tuples, _, heatup_meta = load_episodes_npz(args.heatup_cache_file)
        # Same fail-fast obs-dim guard as the heuristic-cache load above: a
        # stale seed (e.g. the 78-dim lcca_awac_seed_v1.npz after the Gen-3
        # 87-dim obs change) must fail HERE, not at the first update batch.
        if episodes_tuples:
            try:
                # Gen-4 — compare against the CRITIC width (full flat obs
                # incl. the privileged tail); the policy is the SLICED width
                # and would false-fail a valid cache.
                _expected_obs = int(agent.algo.model.q1.n_observations)
            except Exception:
                _expected_obs = None
            _cache_obs = int(np.asarray(episodes_tuples[0][0]).shape[-1])
            if _expected_obs and _cache_obs != _expected_obs:
                raise ValueError(
                    f"Heatup cache obs dim {_cache_obs} != network obs dim "
                    f"{_expected_obs} ({args.heatup_cache_file}). The cache "
                    "was harvested under a different observation layout — "
                    "regenerate it with the current code."
                )
        # Gen-4 — reward-version guard (mirror of the heuristic-cache site):
        # cached rewards must be scored under THIS run's buckle_reward_coef.
        from eve_rl.util.experience_cache import cache_buckle_coef
        _cache_coef = cache_buckle_coef(args.heatup_cache_file)
        if abs(_cache_coef - _buckle_coef) > 1e-9:
            raise ValueError(
                f"Heatup cache reward version mismatch: cache scored with "
                f"buckle_reward_coef={_cache_coef}, this run uses "
                f"{_buckle_coef} ({args.heatup_cache_file}). Pass "
                f"--buckle_reward_coef {_cache_coef} or re-harvest."
            )
        n_pushed = 0
        for i, ep_tuple in enumerate(episodes_tuples):
            flat_obs, actions, rewards, terminals = ep_tuple
            # Heatup episodes are random-action — NOT demos. Carry quality
            # metadata when present (a few heatup episodes reach the target).
            m = heatup_meta[i] if heatup_meta is not None else {}
            replay_ep = EpisodeReplay(
                flat_obs=list(flat_obs),
                actions=list(actions),
                rewards=list(rewards),
                terminals=list(terminals),
                is_demo=False,
                episode_return=float(m.get("episode_return", 0.0)),
                reached_target_daughter=bool(
                    m.get("reached_target_daughter", False)
                ),
            )
            agent.replay_buffer.push(replay_ep)
            n_pushed += 1
        print(f"Heatup cache loaded: {n_pushed} episodes. Skipping heatup phase.")
        heatup_steps_effective = 0

    # Plan v12 Stage 2 — indefinite --heatup_only harvest: give the heatup loop
    # an effectively-infinite step budget so neither the step nor the episode
    # limit ends it. The run ends ONLY on `docker stop` (SIGTERM → the shared
    # heatup_stop Event → workers flush their final batch and exit). Gated to
    # --heatup_only with no fixed episode / success quota, so normal training
    # heatup is byte-for-byte unchanged.
    if (
        getattr(args, "heatup_only", False)
        and not args.heatup_episodes
        and not args.heatup_until_successes
    ):
        heatup_steps_effective = int(1e12)

    # Determine heatup cache save path (only if not loading from cache)
    heatup_cache_save = args.save_heatup_cache if not args.heatup_cache_file else None

    # Warm-start gate — pretraining only makes sense with a seed buffer to
    # pretrain on. Require BOTH the heuristic and heatup caches loaded;
    # otherwise zero it out (no seed data → nothing to warm-start from).
    pretrain_updates = args.pretrain_updates
    both_caches = (
        args.heuristic_cache_file and os.path.isfile(args.heuristic_cache_file)
        and args.heatup_cache_file and os.path.isfile(args.heatup_cache_file)
    )
    # Plan v9 — warm-start is ALSO valid when the seed buffer is generated
    # in THIS run (heuristic seeding + heatup both populate the buffer
    # before pretraining runs inside training_run). The original gate only
    # accepted caches loaded from disk; the single-run generate path
    # (heuristic_seeding > 0, with --save_*_cache) populates the buffer
    # just as well. heatup always runs unless a heatup cache is loaded, so
    # `heuristic_seeding > 0` is sufficient to guarantee a non-empty seed
    # buffer at pretrain time.
    seed_generated_this_run = args.heuristic_seeding > 0
    # Plan v10 — heatup also populates a non-empty seed buffer (especially
    # heatup-until-N), so it is a valid warm-start seed source on its own.
    heatup_runs = (args.heatup_until_successes > 0) or (heatup_steps_effective > 0)
    # Plan v12 — a loaded --heatup_cache_file (e.g. the curated LCCA seed) IS a
    # valid warm-start seed on its own: its episodes were pushed to the replay
    # buffer above (and heatup_steps_effective is then 0, so heatup_runs is
    # False). Without this the pretrain on a heatup-cache seed is silently
    # skipped.
    heatup_cache_loaded = bool(
        args.heatup_cache_file and os.path.isfile(args.heatup_cache_file)
    )
    seed_buffer_available = (
        both_caches or seed_generated_this_run or heatup_runs or heatup_cache_loaded
    )
    if pretrain_updates > 0 and not seed_buffer_available:
        print("[Plan v8/v9/v10] --pretrain_updates ignored — no seed buffer: "
              "need caches loaded, --heuristic_seeding > 0, or heatup to run.")
        pretrain_updates = 0
    elif pretrain_updates > 0:
        if both_caches:
            src = "loaded caches"
        elif seed_generated_this_run:
            src = "heuristic seeding this run"
        else:
            src = "heatup seed this run"
        print(f"[Plan v8/v9/v10] warm-start: {pretrain_updates} pretraining "
              f"updates on the seeded buffer ({src}) before exploration.")

    # Plan v12 Stage 2 — for the indefinite --heatup_only harvest, install a
    # SIGTERM/SIGINT handler so `docker stop` (SIGTERM to PID 1) sets the shared
    # heatup-stop Event. Each worker's heatup loop checks it, exits, and runs its
    # final-batch flush (try/finally in single.heatup) so the in-flight <N batch
    # is not lost. NB: use `docker stop -t 60` so workers can finish the current
    # SOFA episode within the grace window before SIGKILL.
    if getattr(args, "heatup_only", False):
        import signal as _signal

        def _heatup_stop_handler(signum, frame):
            try:
                _evt = getattr(getattr(runner, "agent", None), "_heatup_stop", None)
                if _evt is not None:
                    _evt.set()
            except Exception:
                pass

        try:
            _signal.signal(_signal.SIGTERM, _heatup_stop_handler)
            _signal.signal(_signal.SIGINT, _heatup_stop_handler)
        except Exception:
            pass

    try:
        reward, success = runner.training_run(
            heatup_steps_effective,
            TRAINING_STEPS,
            EXPLORE_STEPS_BTW_EVAL,
            CONSECUTIVE_EXPLORE_EPISODES,
            update_per_explore_step,
            eval_seeds=EVAL_SEEDS,
            heatup_cache_save_path=heatup_cache_save,
            pretrain_updates=pretrain_updates,
            heatup_until_successes=args.heatup_until_successes,
            heatup_episode_limit=args.heatup_episodes,
            # Plan v12 Stage 2 — when --heatup_only is set, runner.training_run
            # exits after heatup harvest completes and per-target .npz files
            # are saved. NO pretrain, NO AWAC. Used by the standalone heatup
            # harvester launcher.
            heatup_only=getattr(args, "heatup_only", False),
            # Rolling per-daughter chunk save (Version A + Version B) every N
            # SOFA episodes — ONLY for --heatup_only runs (single-target training
            # heatup must keep returning its episodes for seeding).
            heatup_save_every=(
                int(getattr(args, "heatup_save_every", 0) or 0)
                if getattr(args, "heatup_only", False) else 0
            ),
            # RL_IMPROV_15 — pretrain-only baseline eval + checkpoint before
            # any exploration (reference point for all later evals).
            eval_after_pretrain=getattr(args, "eval_after_pretrain", False),
        )
    finally:
        agent.close()


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="perform IJCARS23 training")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, wehre the NN update is performed. ",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Runs optuna run with stochastic eval function of SAC.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="test", help="Name of the training run"
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.00021989352630306626,
        help="Learning Rate of Optimizers",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[900, 900, 900, 900],
        help="Hidden Layers",
    )
    parser.add_argument(
        "-en",
        "--embedder_nodes",
        type=int,
        default=500,
        help="Number of nodes per layer in embedder",
    )
    parser.add_argument(
        "-el",
        "--embedder_layers",
        type=int,
        default=1,
        help="Number of layers in embedder",
    )
    parser.add_argument(
        "--env_version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Environment version: 1=original (PathDelta), 2=waypoint+centerlines, 3=tuned waypoint, 4=arclength progress+guidance, 5=optimized env4 with shared projection cache",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable action-space curriculum (Stage 1: gw-only, Stage 2: scaled catheter, Stage 3: full)",
    )
    parser.add_argument(
        "--curriculum_stage1",
        type=int,
        default=200_000,
        help="Steps in curriculum Stage 1 (guidewire-only)",
    )
    parser.add_argument(
        "--curriculum_stage2",
        type=int,
        default=500_000,
        help="Cumulative steps at which curriculum Stage 2 ends",
    )
    parser.add_argument(
        "--heuristic_seeding",
        type=int,
        default=0,
        help="Number of heuristic episodes to seed replay buffer (0=disabled, recommended: 500)",
    )
    parser.add_argument(
        "--seed_only",
        action="store_true",
        help=(
            "Gen-4 harvester mode: run heuristic seeding (+ --save_heuristic_"
            "cache) and EXIT before heatup/training. Produces a Gen-4 121-dim "
            "RCCA demo seed .npz for a later run to load via "
            "--heuristic_cache_file. Combine with --procedural_rcca to harvest "
            "on the per-worker varied anatomy."
        ),
    )
    parser.add_argument(
        "--min_success_rate",
        type=float,
        default=0.3,
        help="Minimum fraction of successful episodes in heuristic seeding (default: 0.3)",
    )
    parser.add_argument(
        "--max_seeding_multiplier",
        type=int,
        default=5,
        help="Max total episodes = heuristic_seeding * this (safety cap, default: 5)",
    )
    parser.add_argument(
        "--heuristic_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heuristic episodes from (skips generation)",
    )
    parser.add_argument(
        "--save_heuristic_cache",
        type=str,
        default=None,
        help="Path to save generated heuristic episodes to (for reuse)",
    )
    parser.add_argument(
        "--heatup_cache_file",
        type=str,
        default=None,
        help="Path to load pre-generated heatup episodes from (skips heatup)",
    )
    parser.add_argument(
        "--save_heatup_cache",
        type=str,
        default=None,
        help="Path to save heatup episodes to (for reuse)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Directory of SOFA-state .npz checkpoints (from "
            "collect_sofa_checkpoints.py). If set, every training episode "
            "resets to a random checkpoint's state instead of zero insertion."
        ),
    )
    parser.add_argument(
        "--eval_only_checkpoint",
        type=str,
        default=None,
        help=(
            "If set, skip heatup/pretrain/training and run runner.eval() once "
            "with the loaded checkpoint weights. Produces the standard run-dir "
            "log/snapshot structure via the normal Synchron+env5 path."
        ),
    )
    parser.add_argument(
        "--drop_restore_states",
        type=str,
        default=None,
        help=(
            "Plan v11 canonical eval — comma-separated substring tokens of "
            "restore-state .npz filenames to DROP from EVAL_SEEDS. For each "
            "EVAL_SEED, the wrapper picks a restore file via "
            "np.random.default_rng(seed).integers(0, len(files)); if the "
            "picked filename contains any dropped token, that seed is "
            "filtered out. Example: --drop_restore_states pid10145,pid20043 "
            "drops 44 seeds whose mapping lands on those states, leaving the "
            "canonical 54-seed eval. Requires --checkpoint_dir to be set "
            "(the wrapper's sorted file list is the mapping basis)."
        ),
    )
    # Plan v9 Change 8 — explicit RL-start mode flag. The underlying
    # restore mechanism is the existing CheckpointRestoreWrapper +
    # --checkpoint_dir; this flag exists for clarity in launch scripts
    # and to validate the two are configured consistently.
    parser.add_argument(
        "--rl_start_mode",
        type=str,
        default="z345",
        choices=["z345", "sofa_restore"],
        help=(
            "Plan v9: where online RL episodes start. 'z345' (default) "
            "preserves existing behaviour (start at the insertion point). "
            "'sofa_restore' starts each online episode from a SOFA-restored "
            "checkpoint randomly selected from --checkpoint_dir — used "
            "for restore-at-fork training (pool of pre-bif(11) states "
            "produced by Plan v9 Change 5b). Heuristic-mode episodes are "
            "exempt from restore regardless of this flag (see "
            "CheckpointRestoreWrapper.reset)."
        ),
    )
    parser.add_argument(
        "--insertion_z",
        type=float,
        default=None,
        help=(
            "If set, override the femoral insertion point. Wire is anchored "
            "at the trunk centerline point with vessel-CS z closest to this "
            "value. RL_IMPROV_8 §16 wire-history A/B test: try ~345 (about "
            "30-40 mm short of bif2). Mutually exclusive with --checkpoint_dir."
        ),
    )
    # Plan v5 — per-daughter training overrides (RCCA-only first cycle).
    parser.add_argument(
        "--target_branches",
        type=str,
        default=None,
        help=(
            "Comma-separated list of target branch names to override the "
            "default 4-daughter TARGET_BRANCHES. E.g., "
            "'Centerline curve - RCCA.mrk' for RCCA-only training."
        ),
    )
    parser.add_argument(
        "--heuristic_factory",
        type=str,
        default=None,
        help=(
            "Import path of a custom heuristic-factory class, "
            "'module:ClassName'. E.g., "
            "'util.heuristic_policy_rcca:RCCAHeuristicActionFunctionFactory'. "
            "Defaults to the generic HeuristicActionFunctionFactory."
        ),
    )
    parser.add_argument(
        "--seeding_success",
        type=str,
        default="term",
        choices=["term", "clean_thread"],
        help=(
            "Definition of 'success' for heuristic-seeding filter. "
            "'term' (default) = TargetReached terminal (info['success']). "
            "'clean_thread' = info['reached_target_daughter'] AND NOT "
            "info['received_wrong_daughter'] — wire committed onto the "
            "target daughter and never wrong-deflected."
        ),
    )
    parser.add_argument(
        "--replay_mode",
        type=str,
        default="episode",
        choices=["episode", "step"],
        help=(
            "Replay-buffer mode. 'episode' (default) = VanillaEpisodeShared, "
            "stores whole episodes, padded-sequence batches (for the LSTM "
            "embedder; capacity in episodes). 'step' = VanillaStepShared, "
            "stores individual transitions, random-transition batches "
            "(canonical step-SAC; capacity in transitions). Step mode also "
            "switches the buffer/batch/update hyperparameters to 1e6 / 256 / "
            "1.0. Pair with '--embedder_layers 0 --hidden 256 256' for a "
            "feedforward network (step-RL needs a non-recurrent policy)."
        ),
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=0,
        help=(
            "Override the step-mode replay-buffer capacity (transitions). "
            "0 (default) = the mode default (1e6 for step). Set large when "
            "seeding from a big heatup cache so the seed stays a small "
            "fraction of capacity and the is_clean lane is not evicted "
            "(Plan v12 LCCA: ~11e6 keeps the ~1.07M-transition seed <10%%)."
        ),
    )
    parser.add_argument(
        "--per",
        action="store_true",
        help=(
            "Plan v7 — enable Prioritized Experience Replay (proportional "
            "sum-tree sampling + IS-weight bias correction). Orthogonal "
            "on/off switch on step mode: requires --replay_mode step. When "
            "OFF (default), step mode uses uniform VanillaStepShared, so a "
            "with-/without-PER pair isolates PER's contribution exactly."
        ),
    )
    parser.add_argument(
        "--per_alpha",
        type=float,
        default=0.6,
        help="PER priority exponent (0 = uniform, 1 = full prioritization).",
    )
    parser.add_argument(
        "--per_beta_start",
        type=float,
        default=0.4,
        help=(
            "PER initial IS-correction exponent; annealed linearly to 1.0 "
            "over TRAINING_STEPS sample() calls."
        ),
    )
    # Plan v8 — step-RL stabilization-suite knobs. All default to off /
    # neutral, so omitting them reproduces current step-SAC behavior.
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help=(
            "Plan v8 — max global grad-norm for critic + policy "
            "(torch clip_grad_norm_). 0 = off. The prime divergence fix — "
            "the first step-RL run had grad_norm run unbounded to ~434k."
        ),
    )
    parser.add_argument(
        "--log_std_min",
        type=float,
        default=-2.0,
        help=(
            "Anti-rail — FLOOR of the GaussianPolicy log-std band. The band "
            "is SOFTLY enforced by a tanh rescale (raw pre-activation 0 maps "
            "to the band MIDPOINT), not a hard clamp. (-2, 0) is the "
            "validated operational band: floor std~0.135 keeps AWAC policy "
            "entropy near target and prevents deterministic collapse. NB the "
            "old default band (-20, +2) would initialize at log_std=-9 "
            "(std~1e-4, collapsed) under the tanh parameterization."
        ),
    )
    parser.add_argument(
        "--log_std_max",
        type=float,
        default=2.0,
        help=(
            "Anti-rail — CEILING of the GaussianPolicy log-std band, softly "
            "enforced by the same tanh rescale (raw 0 maps to the band "
            "midpoint). Default 2.0 keeps the (-2, 2) midpoint-0 band so a "
            "fresh policy initializes at std~1; pass 0.0 (as the LCCA/RCCA "
            "AWAC launchers do) for the validated (-2, 0) band that caps the "
            "lcca_awac_v1 failure mode where log_std ran to the ceiling on "
            "all 4 dims and actions saturated from noise."
        ),
    )
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=None,
        help=(
            "Override the SAC/AWAC auto-alpha entropy setpoint (default: "
            "-n_actions). For this 4-dim tanh-Gaussian with log_std band "
            "(-2, 0) the healthy operating entropy is ~+2.5, so the -4 "
            "default leaves the regulator inert until the policy is already "
            "railed; ~+1.0 keeps it engaged."
        ),
    )
    parser.add_argument(
        "--entropy_beta_per_dim",
        type=float,
        nargs="+",
        default=None,
        help=(
            "OPTIONAL per-action-dim entropy bonus for AWAC (a manual override "
            "on top of the now-default auto-tuned entropy). Adds -beta_i * "
            "mean(log_std_i) to the policy loss. Order [gw_trans, gw_rot, "
            "cath_trans, cath_rot]. DEFAULT OFF — the principled anti-rail is "
            "auto-tuned entropy (alpha -> target_entropy=-n_actions), so no "
            "hand-picked betas are needed; the heatup data cannot derive them "
            "anyway (actions are uniform-random). Use only for targeted tuning."
        ),
    )
    parser.add_argument(
        "--action_mean_penalty",
        type=float,
        default=0.0,
        help=(
            "OPTIONAL mean-margin penalty for AWAC (override on top of the "
            "default auto-tuned entropy, which already counters the mean rail "
            "via the tanh-Jacobian in log_pi). Adds amp * "
            "mean(|atanh(tanh(mean).clamp(+/-0.99))|) to the policy loss. "
            "DEFAULT 0.0 OFF. AWAC-only."
        ),
    )
    parser.add_argument(
        "--eval_after_pretrain",
        action="store_true",
        help=(
            "RL_IMPROV_15 — run one held-out eval (+ checkpoint) right "
            "after the warm-start pretrain, BEFORE any exploration. "
            "Establishes the pretrain-only baseline quality so the first "
            "online eval has a reference, and banks a clean pretrained "
            "checkpoint. Costs ~1 eval (~30 min) of wall-clock."
        ),
    )
    parser.add_argument(
        "--log_alpha_min",
        type=float,
        default=-10.0,
        help=(
            "RL_IMPROV_15 — floor of the log_alpha clamp (SAC/AWAC "
            "auto-alpha). The v1 collapse: alpha decayed to the -10 floor "
            "over 165k updates (entropy term vanished), entropy ground down "
            "to 0.14, then alpha whipsawed to 0.45 and — with log_std "
            "ceiling-pinned — crushed the action MEAN toward zero (the "
            "freeze). A higher floor (e.g. -5 -> alpha_min 0.0067) keeps "
            "the entropy term alive so entropy never craters and alpha "
            "never needs a violent correction. DEFAULT -10 (legacy)."
        ),
    )
    parser.add_argument(
        "--log_alpha_max",
        type=float,
        default=2.0,
        help=(
            "RL_IMPROV_15 — ceiling of the log_alpha clamp. Caps how hard "
            "the entropy term can push the policy; with std saturated at "
            "its log_std_max ceiling the ONLY entropy lever left is "
            "shrinking |mean|, so an uncapped alpha mean-crushes the "
            "policy (v1 froze at alpha~0.45; healthy learning was seen at "
            "alpha<=0.1). E.g. -2.3 -> alpha_max 0.100. DEFAULT 2.0 "
            "(legacy, alpha_max 7.4)."
        ),
    )
    parser.add_argument(
        "--relax_failure_truncations",
        action="store_true",
        help=(
            "Gen-4 recovery training — fold-stall / off-path detectors keep "
            "counting (obs features 20-21/44 + STUCK_CHECKPOINT_DIR pool "
            "triggers) but no longer truncate RL episodes; MaxSteps/VesselEnd/"
            "SimError end them. You cannot learn recovery (retract, unbuckle, "
            "re-approach) from states the env kills you for entering. "
            "Heuristic-mode demo aborts are unaffected. Failure episodes run "
            "2-3x longer — pair with a stuck-pool restore curriculum "
            "(STUCK_CHECKPOINT_DIR harvest, then --checkpoint_dir on the pool)."
        ),
    )
    parser.add_argument(
        "--buckle_reward_coef",
        type=float,
        default=0.0,
        help=(
            "Gen-4 privileged reward shaping — anti-buckle potential. Adds "
            "coef*(phi_t - phi_{t-1}) per step, phi in [-1,0] from gw slack "
            "(inserted_gw - proj.s, 5mm dead-band, 40mm cap) + SOFA contact "
            "proxy mean|pos - free_pos| (2mm cap), equal weights "
            "(util/buckle_reward.py). Potential/delta form: closed loops net "
            "exactly zero (not farmable); forming a buckle costs up to -coef, "
            "recovering earns it back; stuck-pool restored episodes that "
            "unbuckle net POSITIVE (the recovery incentive the audit found "
            "missing). Reward is env-computed so using the critic-only "
            "contact signal is legitimate (privileged reward). 1.0 = slack "
            "channel at parity with the 0.01/mm progress factor; 0.5 "
            "recommended first run. DEFAULT 0.0 OFF (frozen legacy reward)."
        ),
    )
    parser.add_argument(
        "--procedural_rcca",
        action="store_true",
        help=(
            "Gen-4 varied-RCCA anatomy — keep the LOADED DualDeviceNav arch "
            "FIXED and vary ONLY the RCCA->RICA->siphon per worker "
            "(DualDeviceNavRCCAVaried: the loaded RCCA centerline is "
            "perturbed distally, anchored at the real ostium; the whole tree "
            "is re-meshed, same vessel-CS frame as DualDeviceNav so obs "
            "match / a fixed-mesh policy can warm-start). The wire START is "
            "FIXED at the RCCA ostium (identical every worker) - this "
            "isolates the siphon-navigation problem (no arch/fork-commit). "
            "Worker i seeded --procedural_seed + i; eval = fixed held-out "
            "RCCA (seed base-1). Re-randomizes every "
            "--procedural_change_every episodes. Env v5 only. Invalidates "
            "fixed-mesh caches/checkpoints. May be combined with "
            "--checkpoint_dir for a mesh-matched recovery curriculum (#3): "
            "each worker's tree is pinned to the restored checkpoint's mesh, "
            "so the pool must be a mesh-fingerprinted stuck harvest."
        ),
    )
    parser.add_argument(
        "--procedural_seed", type=int, default=12345,
        help="Base RNG seed for --procedural_rcca (worker i uses base+i).",
    )
    parser.add_argument(
        "--procedural_change_every", type=int, default=10,
        help="Regenerate each worker's procedural vessel every N episodes "
             "(default 10). Each regen triggers a SOFA scene rebuild "
             "(~seconds), amortized over N minutes-long episodes.",
    )
    parser.add_argument(
        "--aux_coef",
        type=float,
        default=0.0,
        help=(
            "Gen-4 auxiliary privileged-label supervision weight. > 0 adds a "
            "policy head predicting privileged-tail values (see --aux_labels) "
            "from the deployable obs prefix, MSE-weighted into the policy "
            "loss — representation shaping toward inferring contact/buckle "
            "state. DEFAULT 0.0 OFF."
        ),
    )
    parser.add_argument(
        "--aux_labels",
        type=str,
        default="",
        help=(
            "Comma-separated indices RELATIVE to the privileged tail for "
            "--aux_coef supervision (PrivilegedState layout: e.g. '6,21,20' "
            "= max contact proxy, slip, fold counter). Empty = off."
        ),
    )
    parser.add_argument(
        "--heatup_until_successes",
        type=int,
        default=0,
        help=(
            "Plan v10 — run heatup (random, restore-at-fork) in chunks of "
            "HEATUP_STEPS until N episodes THREAD RCCA (info final_branch== "
            "target), safety cap ~2000 eps. 0 = off (fixed-step heatup). The "
            "whole heatup run (threaded + fails) becomes the seed; no cache, "
            "no heuristic. 'threaded' is only a stop criterion (no reward "
            "change); real success stays TargetReached. NOTE: random heatup "
            "can't thread RCCA even from the good states — use heuristic "
            "seeding (--heuristic_from_restore) instead."
        ),
    )
    parser.add_argument(
        "--heuristic_from_restore",
        action="store_true",
        help=(
            "Plan v10 — run heuristic SEEDING from the SOFA restore states "
            "(restore-at-fork) instead of z=345. Builds the seeding schedule "
            "WITHOUT heuristic_mode so CheckpointRestoreWrapper restores the "
            "fork state and the heuristic steers from there. In-distribution "
            "demos for restore-at-fork training."
        ),
    )
    parser.add_argument(
        "--heatup_episodes",
        type=int,
        default=0,
        help=(
            "Plan v10 — run exactly N heatup episodes (random, restore-at-fork) "
            "AFTER heuristic seeding, for the fail/exploration side of the "
            "seed. 0 = off (fixed-step heatup)."
        ),
    )
    parser.add_argument(
        "--update_per_explore_step",
        type=float,
        default=None,
        help=(
            "Plan v8 — override the update:explore-step ratio (else the "
            "replay-mode default: step=1.0, episode=1/20). Lower (0.25-0.5) "
            "= fewer gradient steps per explore step."
        ),
    )
    parser.add_argument(
        "--demo_priority_bonus",
        type=float,
        default=0.0,
        help=(
            "Plan v8 (DQfD) — additive PER priority bonus for heuristic-"
            "seeded (demo) transitions, re-applied on every priority update "
            "so demos keep a floor and are never starved. 0 = off. Step+PER."
        ),
    )
    parser.add_argument(
        "--priority_mode",
        type=str,
        default="td",
        choices=["td", "return", "outcome"],
        help=(
            "Plan v8 — PER priority source. 'td' (default) = |TD|-error. "
            "'return' = episode return (fixed at push). 'outcome' = high/low "
            "by reached_target_daughter (fixed at push; robust to the "
            "deferred +-1 reward bug). Step+PER only."
        ),
    )
    parser.add_argument(
        "--balanced_fraction",
        type=float,
        default=0.0,
        help=(
            "Plan v8 — fraction of each batch drawn from a 'clean' stream "
            "(transitions from episodes that reached the target daughter). "
            "0 = off. Step mode (PER or uniform)."
        ),
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "awac"],
        help=(
            "Plan v8 — RL algorithm. 'sac' (default) or 'awac' (Advantage-"
            "Weighted Actor-Critic — advantage-weighted policy update, more "
            "stable on demo-seeded buffers)."
        ),
    )
    parser.add_argument(
        "--awac_lambda",
        type=float,
        default=3.0,
        help=(
            "Plan v8 — AWAC advantage temperature (only used with --algo "
            "awac). exp(A/lambda); lambda~3-10 spreads advantages, lambda=1 "
            "saturates the weight clamp for any A>3."
        ),
    )
    parser.add_argument(
        "--pretrain_updates",
        type=int,
        default=0,
        help=(
            "Warm-start — number of gradient updates on the seeded buffer "
            "BEFORE exploration begins, so the first explore episodes run a "
            "demo-informed policy rather than a random net. 0 = off. Only "
            "applied when BOTH --heuristic_cache_file and --heatup_cache_file "
            "are loaded (otherwise there is no seed buffer to pretrain on)."
        ),
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default="none",
        choices=["none", "mesh", "centerlines"],
        help=(
            "End-of-episode snapshot rendering. 'mesh' / 'centerlines' set "
            "SNAPSHOT_MODE before worker spawn; env5 renders a PNG per "
            "episode bucketed by phase (seed/eval/explore). Use "
            "prune_training_snapshots.py to keep all seed+eval and only "
            "the 10-best/10-worst per 100 explore episodes."
        ),
    )
    # Plan v9 Change 8b — restore-start debug snapshots into a SEPARATE
    # folder, regardless of the main --snapshots flag. One image per
    # SOFA-restored online episode at reset() time, captured AFTER the
    # restore + obs/reward re-reset (so the rendered pose IS the
    # post-restore state).
    parser.add_argument(
        "--restore_start_snapshots",
        action="store_true",
        help=(
            "Plan v9 Change 8b — render+save the wire's post-SOFA-restore "
            "pose into a separate snapshot folder for visual "
            "verification that the restore-at-fork mode is landing the "
            "wire at the intended pose (just before bif(11)). One PNG per "
            "online episode. Sets env vars RESTORE_START_SNAPSHOT_DIR + "
            "RESTORE_START_SNAPSHOT_MODE for env5 to pick up. Works "
            "independently of --snapshots."
        ),
    )

    # ====================================================================
    # Plan v12 Stage 1 — Multi-target random heatup harvester CLI flags.
    # ====================================================================
    parser.add_argument(
        "--multi_target_heatup",
        action="store_true",
        help=(
            "Plan v12 Stage 1 — replace BenchEnv5 single-target env_train "
            "with MultiTargetEnv5 wrapping ONE intervention + 4 virtual "
            "envs (RCCA primary + LCCA/RVA/LVA secondaries). Each SOFA "
            "tick advances the shared SOFA backend; every virtual env "
            "computes its own (obs, reward, terminal) and emits a "
            "target-tagged Episode. runner.py heatup-save partitions on "
            "target_branch_idx into 4 per-target .npz files. Stage 3 "
            "AWAC trains per-daughter from these files."
        ),
    )
    parser.add_argument(
        "--heatup_only",
        action="store_true",
        help=(
            "Plan v12 Stage 2 — run ONLY the heatup loop, then exit. "
            "Standalone heatup harvest process: NO pretrain, NO AWAC "
            "training, NO online explore. The trainer enters the heatup "
            "loop (multi-target if --multi_target_heatup is also set) "
            "and runs until SIGTERM, the episode/step limit, or the "
            "(optional) per-target success quota is reached. Rolling "
            "save every --heatup_save_every SOFA episodes; SIGTERM does a "
            "final clean flush of all 4 per-target .npz files."
        ),
    )
    parser.add_argument(
        "--heatup_save_every",
        type=int,
        default=100,
        help=(
            "Plan v12 Stage 2 — rolling save cadence. Every N SOFA "
            "episodes, the runner overwrites the per-target .npz files "
            "with everything harvested so far. Allows the user to inspect "
            "snapshots + cumulative clean counts mid-run and stop when "
            "satisfied (Decision #13). Default 100."
        ),
    )
    parser.add_argument(
        "--snapshot_every",
        type=int,
        default=10,
        help=(
            "Plan v12 Stage 2 — sparse snapshot rate. Every Nth episode's "
            "centerline snapshot is rendered to bound disk; "
            "is_clean=True successes are rendered regardless (rare and "
            "high-value). Combined with --snapshots centerlines. Default 10."
        ),
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help=(
            "Plan v12 R16 — single seed that controls the per-worker + "
            "per-episode RNG separation. Heatup is deterministic-per-"
            "(base_seed, worker_id, episode_idx) AND distinct across "
            "workers, so same --base_seed reproduces the harvest while "
            "16 workers produce 16 disjoint random streams. Read by "
            "single.py heatup loop + singelagentprocess.py spawn-time "
            "reseed. Default 42."
        ),
    )

    args = parser.parse_args()

    main(args)
