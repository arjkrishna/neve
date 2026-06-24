"""Plan v11 Stage 1 — pure-offline RL trainer (AWAC / IQL / BC+AWAC / CQL).

Bake-off harness. Loads filtered V2+V3 transition archives, warm-starts
the policy from `checkpoint250531.everl` (the v3 86 %-on-pid19116 model),
runs `runner.offline_train()` against the static buffer, evaluates on
the canonical 54-seed pool every `eval_every_updates`, and saves a
checkpoint each eval.

NEVER does heatup / explore / online-sample collection. Reward, obs,
terminal, dynamics — all UNCHANGED from Plan v9. The only change is the
RL gradient signal during training and the buffer composition.

Required for the Stage 1A/B/C/D variants:
  --algo {awac,iql,bc_awac,cql}
  --awac_lambda <float>              (1A: 8.0)
  --bc_warmup_updates <int>          (1C: 5k/10k/20k)
  --iql_tau <float>                  (1B: 0.7 or 0.9)
  --iql_beta <float>                 (1B: 0.1/0.3/1.0)
  --cql_alpha <float>                (1D conditional)
  --warm_start_checkpoint <path>     (v3's checkpoint250531.everl)
  --v2_buffer / --v3_buffer <path>   (saved PER replay_buffer.npz files)
  --canonical_eval                   (apply Plan v11 Decisions #1 seed filter)
"""

import argparse
import logging
import os
import sys
import tempfile

import numpy as np
import torch
import torch.multiprocessing as mp

# Make sibling `util/` package importable regardless of CWD. Python
# normally puts the script dir on sys.path automatically, but the docker
# entrypoint doesn't always; adding _HERE (the dir containing util/)
# fixes "'util' is not a package".
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from util.env5 import BenchEnv5
from util.agent import BenchAgentSynchron, OfflineDummyEnv
from util.buffer_filter import (
    filter_v2, filter_v3, concat_filtered, make_export_dict,
)
from util.util import get_result_checkpoint_config_and_log_path
from util.checkpoint_restore import CheckpointRestoreWrapper
from eve_rl import Runner
from eve_bench import DualDeviceNav


# Canonical eval seeds — same list DualDeviceNav_train.py uses.
EVAL_SEEDS = (
    "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,"
    "47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,"
    "97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,"
    "130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,"
    "154,155,156,158,159,161,162,167,168,171,175"
)
EVAL_SEEDS = [int(s) for s in EVAL_SEEDS.split(",")]


def _filter_canonical_seeds(checkpoint_dir, dropped_tokens):
    """Plan v11 Decisions #1 — drop seeds that deterministically map to
    any of the named restore-state tokens (default pid10145 + pid20043,
    the small-gap states). Mirrors DualDeviceNav_train.py's
    --eval_only_checkpoint branch but accepts a CLI-parameterized
    token list."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(checkpoint_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No .npz restore files in {checkpoint_dir}"
        )
    tokens = tuple(t.strip() for t in dropped_tokens.split(",") if t.strip())
    def _picks_dropped(seed):
        idx = int(np.random.default_rng(seed).integers(0, len(files)))
        return any(t in os.path.basename(files[idx]) for t in tokens)
    return [s for s in EVAL_SEEDS if not _picks_dropped(s)]


def _parse_args():
    p = argparse.ArgumentParser(description="Plan v11 Stage 1 offline RL trainer")
    # Run identity + dirs
    p.add_argument("-n", "--name", required=True)
    p.add_argument("-d", "--device", default="cuda:0")
    p.add_argument("-nw", "--n_worker", type=int, default=8,
                   help="Synchron workers — used for eval parallelism only")
    # Buffer sources
    p.add_argument("--v2_buffer", default=None,
                   help="Path to V2 replay_buffer.npz (PER export). "
                        "Either --v2_buffer or --v3_buffer is required.")
    p.add_argument("--v3_buffer", default=None,
                   help="Path to V3 replay_buffer.npz; omit to use V2 only")
    p.add_argument("--v3_action_abs_max", type=float, default=0.85,
                   help="V3 saturation filter (risk audit #2)")
    p.add_argument("--max_failure_per_band", type=int, default=30_000)
    # Stage 1B buffer-composition knobs (user strategy: all clean +
    # small failure tail + priority_mode=outcome biases sampling toward
    # successes WITHOUT discarding failure-learning signal for the critic).
    p.add_argument(
        "--drop_all_demos_unconditional", action="store_true",
        help="Stage 1B — drop V2 is_demo transitions entirely (they were "
             "from a 0-1%%-success checkpoint).",
    )
    p.add_argument(
        "--require_episode_return_min", type=float, default=0.0,
        help="Stage 1B — episode_return floor for inclusion. 0 disables.",
    )
    # Stage 1B-v2 — diversity-weighted priority. Workflow w6ys0odcm
    # found 85/15 mode split (fast-push vs slow-pullback) in V2+V3
    # successes; priority_mode=outcome 10x boost amplifies the dominant
    # mode and loses pullback expertise. Boosting rare-bin transitions
    # at export time reshapes PER's initial sampling distribution.
    p.add_argument(
        "--diversity_alpha", type=float, default=0.0,
        help="Stage 1B-v2 diversity strength. 0.0=OFF (byte-identical). "
             "0.5=mild (recommended first try). 1.0=strong.",
    )
    p.add_argument(
        "--diversity_resolution", type=float, default=0.5,
        help="Action-grid cell width in normalized [-1, 1] units.",
    )
    p.add_argument(
        "--diversity_max_boost", type=float, default=4.0,
        help="Symmetric clamp on per-leaf multiplier [1/x, x].",
    )
    p.add_argument(
        "--diversity_pool", choices=["all", "clean_only"], default="all",
        help="Count bin occupancy over full keep-set or clean-only.",
    )
    p.add_argument(
        "--diversity_dims", default="0,2",
        help="Comma-separated action-dim indices to use for binning. "
             "Default '0,2' = cath_trans + gw_trans (mode-discriminating).",
    )
    p.add_argument(
        "--replay_buffer_size", type=int, default=350_000,
        help="PER buffer capacity. Must be >= filtered transition count.",
    )
    # Algo
    p.add_argument("--algo", choices=["awac", "iql", "bc_awac", "cql"],
                   default="awac")
    p.add_argument("--awac_lambda", type=float, default=8.0)
    p.add_argument("--iql_tau", type=float, default=0.7)
    p.add_argument("--iql_beta", type=float, default=3.0)
    p.add_argument(
        "--iql_awr_max", type=float, default=5.0,
        help="Stage 1B — cap on exp(adv/beta) AWR weight. Workflow "
             "synthesis recommends 5.0 (vs AWAC's 20.0).",
    )
    p.add_argument("--cql_alpha", type=float, default=1.0)
    p.add_argument("--bc_warmup_updates", type=int, default=0)
    # Stage 1B safety guardrails.
    p.add_argument(
        "--step0_eval", action="store_true",
        help="Stage 1B — run a FULL canonical eval BEFORE the first "
             "optimizer step. Largely redundant when --baseline_quality "
             "is supplied from a prior anchor run (e.g. Task C 0.577); "
             "kept for first-run sanity checks of new algos.",
    )
    p.add_argument(
        "--step0_smoke_eval_seeds", type=int, default=0,
        help="If > 0, run a quick N-seed smoke eval before training "
             "(instead of full 54-seed step-0 eval). Cheap (~5 min for "
             "5 seeds) sanity check that warm-start loaded correctly. "
             "0 disables.",
    )
    p.add_argument(
        "--baseline_quality", type=float, default=0.577,
        help="Reference Quality (fractional, 0..1) for the abort "
             "guardrail when --step0_eval is OFF. Default 0.577 is the "
             "Task C canonical anchor (30/52 strict on the v3 "
             "checkpoint250531.everl + canonical 54 seeds). Override "
             "if using a different warm-start checkpoint.",
    )
    p.add_argument(
        "--skip_inline_eval", action="store_true",
        help="Train-then-eval-later mode. Skip the SOFA eval at each "
             "eval_every_updates boundary; just save the checkpoint. "
             "Eval whichever checkpoints later via --eval_only_checkpoint "
             "passes (possibly in parallel docker containers). Sidesteps "
             "the Synchron eval-hang that blocks training for 30-90 min "
             "per eval. Implies --abort_quality_drop 0 (no abort possible "
             "without inline eval).",
    )
    p.add_argument(
        "--abort_quality_drop", type=float, default=0.0,
        help="Stage 1B — abort training if eval Quality drops by more "
             "than this many percentage points below the step-0 baseline "
             "at any subsequent eval. 0.0 disables (default).",
    )
    # Optimization
    p.add_argument("--n_updates", type=int, default=100_000)
    p.add_argument("--eval_every_updates", type=int, default=10_000)
    p.add_argument("--lr_policy", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument(
        "--lr_value", type=float, default=3e-4,
        help="Stage 1B IQL — learning rate for the V-network optimizer. "
             "Ignored for non-IQL algos.",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # PER
    p.add_argument("--per_alpha", type=float, default=0.6)
    p.add_argument("--per_beta_start", type=float, default=1.0,
                   help="Risk audit #6: fixed 1.0 in offline mode")
    p.add_argument("--demo_priority_bonus", type=float, default=0.0)
    p.add_argument(
        "--balanced_fraction", type=float, default=0.3,
        help="DELIBERATE offline-specific stabilization (Plan v8). "
             "Online default is 0.0; offline 0.3 skews 30%% of each "
             "batch toward clean (reached-target) transitions.",
    )
    p.add_argument(
        "--priority_mode", choices=["td", "return", "outcome"],
        default="td",
        help="PER priority computation mode. td=dynamic TD-error, "
             "return=fixed episode-return, outcome=high/low by success.",
    )
    # Network
    p.add_argument("--hidden", nargs="+", type=int, default=[256, 256])
    p.add_argument("--embedder_nodes", type=int, default=0)
    p.add_argument("--embedder_layers", type=int, default=0)
    p.add_argument("--log_std_min", type=float, default=-2.0)
    # Direction 1 scaffolds (no-op in Stage 1; reserved for Stage 2)
    p.add_argument("--entropy_beta_per_dim", default="0,0,0,0")
    p.add_argument("--action_mean_penalty", type=float, default=0.0)
    # Warm-start
    p.add_argument("--warm_start_checkpoint", default=None,
                   help="Path to a .everl checkpoint to load model state from")
    p.add_argument(
        "--warm_start_mode", choices=["network_only", "full"],
        default="network_only",
        help="network_only (Plan v11 spec) loads only network weights, "
             "resets Adam state + LR schedule. full loads everything "
             "(networks + optimizer + scheduler) — the previously buggy "
             "behavior, kept as opt-in for A vs B bake-off comparison.",
    )
    # FIX 1 (RL_IMPROV_8 KL-anchor iteration) — KL-to-warmstart penalty.
    # alpha_kl multiplies the closed-form Gaussian KL between the current
    # policy and a FROZEN reference policy loaded from a separate .everl
    # checkpoint (typically the same checkpoint as --warm_start_checkpoint
    # — the Task C anchor — so the policy is anchored to the warm-start it
    # was initialized from).
    p.add_argument(
        "--alpha_kl", type=float, default=0.0,
        help="FIX 1 — KL-to-warmstart anchor strength. 0.0 disables "
             "(back-compat). 0.1 = mild anchor recommended for first "
             "KL-anchor iteration. Only used when --algo iql.",
    )
    p.add_argument(
        "--kl_warmstart_checkpoint", default=None,
        help="FIX 1 — path to the .everl checkpoint whose policy "
             "network_state_dict becomes the frozen reference for the KL "
             "anchor. If --alpha_kl > 0 and this is omitted, falls back "
             "to --warm_start_checkpoint. The reference policy is loaded "
             "ONCE at agent construction; its weights are frozen "
             "(requires_grad=False) for the entire run.",
    )
    # TRUE-resume mode (Stage 1B-IQL resume support). When --resume is
    # set, we preserve the step counter from the loaded checkpoint so
    # offline_train continues from update_step=N rather than restarting
    # from 0. Requires --warm_start_checkpoint to be set; auto-promotes
    # --warm_start_mode to "full" so optimizer + scheduler state are
    # also preserved (otherwise the resumed step counter would be
    # inconsistent with a freshly-reset LR schedule).
    p.add_argument(
        "--resume", action="store_true",
        help="True continuation of a prior offline run. Preserves "
             "step_counter.update from the loaded checkpoint (so eval "
             "cadence resumes at the correct boundary, e.g. continuing "
             "from update 10k schedules the next eval at update 20k "
             "instead of overwriting checkpoint10000.everl). Requires "
             "--warm_start_checkpoint; auto-sets --warm_start_mode=full.",
    )
    # Eval-only mode — load a checkpoint, run runner.eval(seeds=...) once,
    # exit. Useful for re-evaluating a saved Stage 1A offline checkpoint
    # without re-running 100k updates.
    p.add_argument(
        "--eval_only_checkpoint", default=None,
        help="If set, load this .everl, run a single eval, exit.",
    )
    p.add_argument(
        "--drop_restore_states", default="pid10145,pid20043",
        help="Comma-separated tokens; seeds whose deterministic restore "
             "checkpoint filename contains any token are dropped. "
             "Default matches Plan v11 Decisions #1.",
    )
    # Eval
    p.add_argument("--checkpoint_dir", required=True,
                   help="Path to the 5-state restore pool (env_eval wrapper)")
    p.add_argument("--canonical_eval", action="store_true",
                   help="Apply Plan v11 Decisions #1 seed filter (54 seeds)")
    p.add_argument(
        "--insertion_z", type=float, default=345.0,
        help="DualDeviceNav initial wire z-position. Default 345 = trunk "
             "near bif(11), matching Task C / rcca_5good_checkpoints. "
             "Without this, DualDeviceNav initializes at FEMORAL access "
             "(z=35, [65,-5,35]), so the vessel tree + target sampling + "
             "path-reward setup are computed for the femoral start. The "
             "SOFA-restore wrapper overrides the wire DOF at episode "
             "reset, but base simulation state still mismatches.",
    )
    p.add_argument(
        "--rl_start_mode", choices=["z345", "sofa_restore"],
        default="sofa_restore",
        help="Parity flag with train.py. For offline this is informational "
             "(no explore episodes), but document what Task C used.",
    )
    # Plan v11 — single-daughter scoping. Without this, DualDeviceNav
    # samples targets across ALL 4 daughters (RCCA / RVA / LCCA / LVA),
    # producing target_branch=unknown in EPISODE_START logs and a
    # 4-daughter Quality that is NOT comparable to the Task C canonical
    # RCCA-only anchor.
    p.add_argument(
        "--target_branches",
        default="Centerline curve - RCCA.mrk",
        help="Comma-separated list of target branch names. Default: "
             "RCCA only (Stage 1 bake-off scope). Pass an empty string "
             "to fall back to all 4 daughters (NOT recommended).",
    )
    # Plan v5 — env5 renders end-of-episode PNGs bucketed by phase.
    # Stage 1 has no explore/heatup, only eval, so all snapshots land
    # under snapshots/eval/<target>/{success,wfs,wbt,max_steps}/. "off"
    # disables.
    p.add_argument(
        "--snapshots",
        default="centerlines",
        help="Snapshot rendering mode (centerlines / off / none).",
    )
    return p.parse_args()


def _load_buffer_into_agent(agent, args, buffer_capacity):
    """Filter V2 (+V3), concat, persist as an export-format npz, then
    load into the agent's PERVanillaStepShared via the existing
    load_buffer_from_file pathway."""
    log = logging.getLogger("offline.buffer")
    parts = []
    if not args.v2_buffer and not args.v3_buffer:
        raise ValueError("Must supply at least one of --v2_buffer or --v3_buffer")
    if args.v2_buffer:
        log.info("Filtering V2 buffer: %s", args.v2_buffer)
        parts.append(filter_v2(
            args.v2_buffer,
            max_per_failure_stratum=args.max_failure_per_band,
            drop_all_demos_unconditional=args.drop_all_demos_unconditional,
            require_episode_return_min=args.require_episode_return_min,
        ))
    else:
        log.info("V2 buffer SKIPPED (state-distribution test: V3-only)")
    if args.v3_buffer:
        log.info("Filtering V3 buffer: %s", args.v3_buffer)
        parts.append(filter_v3(
            args.v3_buffer,
            max_per_failure_stratum=args.max_failure_per_band,
            action_abs_max=args.v3_action_abs_max,
            drop_all_demos_unconditional=args.drop_all_demos_unconditional,
            require_episode_return_min=args.require_episode_return_min,
        ))
    merged = concat_filtered(parts)
    n = int(merged["actions"].shape[0])
    log.info("Merged buffer: %d transitions (capacity=%d)", n, buffer_capacity)
    if n > buffer_capacity:
        raise ValueError(
            f"Filtered buffer ({n}) exceeds replay capacity ({buffer_capacity}); "
            "increase --replay_buffer_size or tighten filter"
        )

    diversity_dims = [
        int(x) for x in str(args.diversity_dims).split(",") if x.strip()
    ]
    export = make_export_dict(
        merged, buffer_capacity, uniform_priority=1.0,
        diversity_alpha=args.diversity_alpha,
        diversity_resolution=args.diversity_resolution,
        diversity_max_boost=args.diversity_max_boost,
        diversity_pool=args.diversity_pool,
        diversity_dims=diversity_dims,
    )
    with tempfile.NamedTemporaryFile(
        suffix=".npz", delete=False, prefix="offline_buffer_"
    ) as tf:
        tmp_path = tf.name
    np.savez(tmp_path, **export)
    log.info("Wrote merged buffer to %s; loading into shared PER buffer...", tmp_path)
    n_loaded = agent.replay_buffer.load_buffer_from_file(tmp_path)
    log.info("Buffer subprocess reported %d transitions loaded", n_loaded)
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    # The shared-buffer subprocess returns -1 on load failure (see
    # pervanillashared.loop / load_buffer task). Catch it here loudly
    # rather than letting the trainer silently train on an empty buffer
    # — the first sample() call would otherwise hang or fail much later
    # with a misleading error.
    if n_loaded is None or int(n_loaded) <= 0:
        raise RuntimeError(
            f"Buffer subprocess failed to load {tmp_path}: n_loaded={n_loaded}. "
            "Check the buffer-subprocess log for the underlying exception "
            "(e.g. capacity mismatch, missing field, corrupted .npz)."
        )
    if int(n_loaded) != n:
        log.warning(
            "Buffer load size mismatch: requested %d transitions, "
            "subprocess loaded %d. Continuing.", n, int(n_loaded),
        )
    return int(n_loaded)


def main():
    args = _parse_args()
    log = logging.getLogger("offline")

    # Plan v11 Stage 1B — IQL is now wired through BenchAgentSynchron
    # (via eve_rl.algo.IQL + eve_rl.model.IQLModel). CQL is still a stub.
    if args.algo == "cql":
        raise NotImplementedError(
            "--algo cql is stub-only; lands with Stage 1D."
        )

    # Env construction. env_train is real (workers idle but constructed
    # via deepcopy), env_eval is real + CheckpointRestoreWrapper so eval
    # uses the canonical 5-state pool. Both UNCHANGED from Plan v9 except
    # for the default_target_branch scoping below.
    log.info("Constructing envs...")
    intervention = DualDeviceNav(insertion_z=args.insertion_z)

    # Plan v11 — derive target_branches list, then build env_kwargs
    # with default_target_branch when single-daughter (Stage 1 RCCA-only).
    target_branches = [
        b.strip() for b in (args.target_branches or "").split(",") if b.strip()
    ]
    env_kwargs = {}
    if len(target_branches) == 1:
        env_kwargs["default_target_branch"] = target_branches[0]
        log.info(
            "Single-daughter scoping: default_target_branch=%s",
            target_branches[0],
        )
    elif len(target_branches) == 0:
        log.warning(
            "No --target_branches; falling back to all 4 daughters "
            "(eval will NOT be comparable to RCCA-only Task C anchor)."
        )

    env_train = BenchEnv5(
        intervention=intervention, mode="train",
        visualisation=False, **env_kwargs,
    )
    env_eval_unwrapped = BenchEnv5(
        intervention=DualDeviceNav(insertion_z=args.insertion_z),
        mode="eval", visualisation=False, **env_kwargs,
    )
    env_eval = CheckpointRestoreWrapper(
        env_eval_unwrapped,
        checkpoint_dir=args.checkpoint_dir,
        rng_seed=43,
    )
    log.info("env_eval restore pool: %d files", len(env_eval.checkpoint_files))

    # Result dirs (reuses the same util the online trainer uses so all
    # downstream tooling — TB readers, analysis scripts — works.) The
    # function returns 6 values; train.py's variable names for positions
    # 3 and 5 (config_folder vs _config_folder_dup) are misleading per
    # util.util — position 3 is results_folder, position 5 is the actual
    # config_folder. We mirror train.py's order to avoid a second naming
    # divergence.
    results_root = os.path.join(
        os.getcwd(), "results/eve_paper/neurovascular/full/mesh_ben"
    )
    (
        results_file,
        checkpoint_folder,
        results_folder,
        log_path,
        config_folder,
        diagnostics_folder,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=results_root,
        name=args.name,
        create_diagnostics=True,
    )
    # STEP_LOG_DIR MUST be set + the directory MUST exist before agent
    # spawn — worker subprocesses inherit os.environ at spawn time and
    # try to open per-worker log files. Using setdefault was a parity
    # bug: it silently kept a stale value from a prior shell invocation
    # OR didn't create the directory (FileHandler then falls back to
    # stderr). Mirror DualDeviceNav_train.py's pattern exactly.
    logs_subprocesses = os.path.join(diagnostics_folder, "logs_subprocesses")
    os.makedirs(logs_subprocesses, exist_ok=True)
    os.environ["STEP_LOG_DIR"] = logs_subprocesses
    log.info("Set STEP_LOG_DIR=%s", logs_subprocesses)

    # Plan v5 — SNAPSHOT_MODE / SNAPSHOT_DIR MUST be set before the agent
    # spawns worker subprocesses (spawn-inheritance). env5 renders an
    # end-of-episode PNG bucketed by phase (eval / success / wfs / wbt /
    # max_steps). Without this, the eval container produces ZERO PNGs
    # and per-state strict-rate breakdowns are impossible (as in the
    # previous broken Stage 1A run).
    if args.snapshots and args.snapshots.lower() not in ("none", "off"):
        snap_dir = os.path.join(diagnostics_folder, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        os.environ["SNAPSHOT_MODE"] = args.snapshots
        os.environ["SNAPSHOT_DIR"] = snap_dir
        log.info(
            "Snapshots enabled: SNAPSHOT_MODE=%s SNAPSHOT_DIR=%s",
            args.snapshots, snap_dir,
        )

    # IMPORTANT: must use FileHandler (not StreamHandler) — eve_rl's
    # SingleAgentProcess inherits the root logger's handlers and passes
    # them to subprocesses via a serialization callback registry that
    # only knows FileHandler. A StreamHandler triggers
    # KeyError(<class logging.StreamHandler>) at worker spawn time. So
    # mirror DualDeviceNav_train.py's basicConfig pattern.
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # PER capacity: CLI-overridable (default 350k = comfortable margin
    # above the ~280k filtered-buffer target). Online step-mode SAC uses
    # 1M; keeping 350k is acceptable because the offline buffer is
    # FIXED (no growth from explore), but documenting via the CLI flag
    # lets bake-off variants experiment with capacity.
    buffer_capacity = int(args.replay_buffer_size)

    log.info("Constructing agent (algo=%s, awac_lambda=%.2f)...",
             args.algo, args.awac_lambda)
    entropy_beta = [float(x) for x in args.entropy_beta_per_dim.split(",")]
    if len(entropy_beta) not in (0, 4):
        raise ValueError(
            "--entropy_beta_per_dim must be 4 comma-separated floats"
        )

    # Diagnostics configuration for offline training. Drives the trainer
    # subprocess to instantiate eve_rl.util.DiagnosticsLogger, which writes
    # `losses_trainer.csv` (q1/q2 loss, policy loss, alpha, log_pi, grad
    # norms, lr, target_q, clamp_fraction, AWAC-related stats per gradient
    # update) and TensorBoard event files. Must be created BEFORE agent
    # construction because BenchAgentSynchron.__init__ spawns the trainer
    # subprocess that reads this config. STEP_LOG_DIR is already set
    # above (line ~247) so subprocess logs land alongside diagnostics.
    #
    # log_probe_values_every_n_steps is kept at 100 but will silently
    # NOP in offline mode: probe_states are populated by the online
    # heatup phase which we skip, so DiagnosticsLogger.log_probe_values
    # short-circuits when probe_states is None (see single.py guard
    # `if self.diagnostics.probe_states is not None`). Probe values
    # derived from random-action heatup would be stale w.r.t. an
    # offline-trained policy anyway, so this NOP is the correct
    # behavior for Stage 1B/1C bake-offs.
    diagnostics_config = {
        "enabled": True,
        "diagnostics_folder": diagnostics_folder,
        "log_losses_every_n_steps": 1,
        "log_probe_values_every_n_steps": 100,  # NOP when probe_states is None
        "log_batch_samples_every_n_steps": 100,
        "n_batch_samples": 3,
        "tensorboard_enabled": True,
        "probe_plot_every_n_steps": 10000,
        "policy_snapshot_every_n_steps": 10000,
        "flush_every_n_rows": 100,
    }

    # NOTE: BenchAgentSynchron takes a single `lr`; asymmetric LRs are a
    # Stage 2 follow-up (would split q/policy optimizers + schedulers).
    # For Stage 1A we use --lr_policy as the unified LR — the asymmetric
    # LR sweep is documented as deferred to the Stage 1B/1C variants.
    # Plan v11 Stage 1B — map the CLI algo onto the BenchAgentSynchron
    # `algo` flag. SAC/AWAC share the SAC algo class; IQL gets its own
    # class (see eve_rl.algo.IQL). "bc_awac" is still routed through AWAC.
    if args.algo == "iql":
        _agent_algo = "iql"
    elif args.algo in ("awac", "bc_awac"):
        _agent_algo = "awac"
    else:
        _agent_algo = "sac"

    # FIX 1 (RL_IMPROV_8 KL-anchor iteration) — load the frozen reference
    # policy state_dict for the KL anchor BEFORE constructing the agent.
    # The state_dict is forwarded into BenchAgentSynchron -> IQL where it
    # is materialized lazily as a deepcopy of the live policy with these
    # weights and requires_grad=False. Falls back to --warm_start_checkpoint
    # when --kl_warmstart_checkpoint is unset.
    kl_warmstart_state_dict = None
    if args.alpha_kl > 0.0 and args.algo == "iql":
        _kl_cp_path = args.kl_warmstart_checkpoint or args.warm_start_checkpoint
        if _kl_cp_path is None:
            raise ValueError(
                "--alpha_kl > 0 requires either --kl_warmstart_checkpoint "
                "or --warm_start_checkpoint to be set (the frozen reference)."
            )
        log.info("FIX 1 — loading KL warm-start reference policy from %s",
                 _kl_cp_path)
        _kl_cp = torch.load(_kl_cp_path, map_location="cpu")
        if "network_state_dicts" not in _kl_cp:
            raise KeyError(
                f"{_kl_cp_path}: no 'network_state_dicts' key — cannot "
                "extract policy state_dict for KL anchor."
            )
        _kl_nsd = _kl_cp["network_state_dicts"]
        if "policy" not in _kl_nsd:
            raise KeyError(
                f"{_kl_cp_path}: 'network_state_dicts' has no 'policy' "
                "entry — checkpoint is not from a Gaussian-policy algo."
            )
        kl_warmstart_state_dict = _kl_nsd["policy"]
        log.info(
            "FIX 1 — KL anchor active: alpha_kl=%.4f, reference policy "
            "state_dict loaded (%d tensors).",
            args.alpha_kl, len(kl_warmstart_state_dict),
        )
    elif args.alpha_kl > 0.0:
        log.warning(
            "--alpha_kl=%.4f ignored: KL anchor is only implemented for "
            "--algo iql (current algo=%s).", args.alpha_kl, args.algo,
        )

    agent = BenchAgentSynchron(
        trainer_device=torch.device(args.device),
        worker_device=torch.device(args.device),
        lr=args.lr_policy,
        lr_end_factor=1.0,                 # no LR decay in offline mode
        lr_linear_end_steps=args.n_updates,
        hidden_layers=args.hidden,
        embedder_nodes=args.embedder_nodes,
        embedder_layers=args.embedder_layers,
        gamma=args.gamma,
        batch_size=args.batch_size,
        reward_scaling=1.0,
        replay_buffer_size=buffer_capacity,
        env_train=env_train,
        env_eval=env_eval,
        consecutive_action_steps=1,
        n_worker=args.n_worker,
        stochastic_eval=False,
        replay_mode="step",
        per=True,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_steps=1.0,                # no annealing (β fixed at 1.0)
        grad_clip=args.grad_clip,
        algo=_agent_algo,
        awac_lambda=args.awac_lambda,
        demo_priority_bonus=args.demo_priority_bonus,
        priority_mode=args.priority_mode,
        balanced_fraction=args.balanced_fraction,
        log_std_min=args.log_std_min,
        entropy_beta_per_dim=entropy_beta,
        action_mean_penalty=args.action_mean_penalty,
        offline_mode=True,
        diagnostics_config=diagnostics_config,
        iql_tau=args.iql_tau,
        iql_beta=args.iql_beta,
        iql_awr_max=args.iql_awr_max,
        lr_value=args.lr_value,
        # FIX 1 — KL-to-warmstart anchor (no-op when alpha_kl == 0 or
        # kl_warmstart_state_dict is None).
        alpha_kl=args.alpha_kl,
        kl_warmstart_state_dict=kl_warmstart_state_dict,
    )

    # Warm-start: NETWORK-ONLY load (Plan v11 spec). The previous
    # agent.load_checkpoint() call loaded optimizer + scheduler state too,
    # which caused AWAC critic oscillation (+/-15 across 10k updates) and
    # 4/51 success because v3's Adam momentum buffers + scheduler step
    # counter (~270k updates) were completely stale w.r.t. the offline
    # PER buffer. The correct sequence per Plan v11 is:
    #   1. Load only network_state_dicts into the parent algo.
    #   2. Propagate networks to the trainer subprocess.
    #   3. Propagate networks to all worker subprocesses.
    #   4. Reset Adam state on q1 / q2 / policy optimizers in the trainer.
    #   5. Reset LR schedulers in the trainer (so the LR schedule starts
    #      from update_step 0, not from v3's 270k).
    # We DO NOT touch eve_rl; instead we exploit the fact that the parent
    # Synchron's self.algo retains its freshly-constructed (untrained)
    # optimizers/schedulers — sent to CPU at __init__ but never stepped —
    # so we can push *those* state_dicts to the trainer subprocess to
    # effectively reset its optimizer/scheduler state.
    # --resume guardrails: must have a checkpoint, must use full mode.
    # Auto-promote warm_start_mode to "full" so optimizer + scheduler
    # state are preserved alongside the step counter.
    if args.resume:
        if not args.warm_start_checkpoint:
            raise ValueError(
                "--resume requires --warm_start_checkpoint to be set."
            )
        if args.warm_start_mode != "full":
            log.info(
                "Resume mode: auto-promoting --warm_start_mode from %r to "
                "'full' (resume requires full state continuation).",
                args.warm_start_mode,
            )
            args.warm_start_mode = "full"

    if args.warm_start_checkpoint:
        log.info(
            "Warm-start (mode=%s, resume=%s): loading %s",
            args.warm_start_mode, args.resume, args.warm_start_checkpoint,
        )
        cp = torch.load(args.warm_start_checkpoint, map_location="cpu")
        if "network_state_dicts" not in cp:
            raise KeyError(
                f"{args.warm_start_checkpoint}: no 'network_state_dicts' "
                "key — not an Agent.save_checkpoint-format .everl file."
            )

        if args.warm_start_mode == "full":
            # B (comparison condition) — load networks + optimizer +
            # scheduler from the .everl file via the standard
            # agent.load_checkpoint API. This is the previously buggy
            # behavior (causes Adam momentum carry-over from v3 and
            # critic oscillation) kept here as the A vs B comparison
            # condition.
            agent.load_checkpoint(args.warm_start_checkpoint)
        else:
            # A — network_only (Plan v11 spec): networks load, Adam
            # state + LR scheduler RESET via the parent's fresh
            # (never-stepped) state_dicts.
            # (1) Parent algo: network only.
            agent.algo.load_state_dicts_network(cp["network_state_dicts"])
            # (2) Trainer subprocess: receive the loaded network weights.
            agent.trainer.load_state_dicts_network(
                agent.algo.state_dicts_network()
            )
            # (3) Worker subprocesses: receive the loaded network weights.
            agent._worker_load_state_dicts_network(
                agent.algo.state_dicts_network()
            )
            # (4) Reset Adam momentum in the trainer's optimizers by
            # pushing the parent's fresh (never-stepped) optimizer
            # state_dicts (parent.algo is constructed but never trained
            # — Synchron moves it to CPU and trains in the subprocess).
            agent.trainer.load_state_dicts_optimizer(
                agent.algo.state_dicts_optimizer()
            )
            # (5) Reset LR schedulers (last_epoch = 0) similarly.
            fresh_scheduler_state = agent.algo.state_dicts_scheduler()
            if fresh_scheduler_state:
                agent.trainer.load_state_dicts_scheduler(fresh_scheduler_state)

        if args.resume:
            # True-resume path: preserve step counters loaded from the
            # checkpoint so offline_train's update_start = step_counter.update
            # continues at the right boundary (e.g. resume from update 10k
            # schedules the next eval at update 20k rather than overwriting
            # checkpoint10000.everl). The scheduler's last_epoch is also
            # already at the checkpoint's training step from
            # load_state_dicts_scheduler() above, so the LR decay continues
            # coherently.
            log.info(
                "Resume mode: continuing from update_step=%d "
                "(exploration=%d, evaluation=%d, heatup=%d) — "
                "step counters PRESERVED from checkpoint.",
                int(agent.step_counter.update),
                int(agent.step_counter.exploration),
                int(agent.step_counter.evaluation),
                int(agent.step_counter.heatup),
            )
        else:
            # Reset step counters — this is a fresh offline run, not a
            # continuation of v3's (or any other prior run's) training.
            agent.step_counter.update = 0
            agent.step_counter.exploration = 0
            agent.step_counter.heatup = 0
            log.info(
                "Warm-start (mode=%s): loaded; step counters reset to 0.",
                args.warm_start_mode,
            )

    # Load the filtered buffer into the shared PER subprocess.
    _load_buffer_into_agent(agent, args, buffer_capacity)

    # Eval-seed plan.
    eval_seeds = (
        _filter_canonical_seeds(args.checkpoint_dir, args.drop_restore_states)
        if args.canonical_eval else EVAL_SEEDS
    )
    log.info("Eval: %d seeds (%s)", len(eval_seeds),
             "canonical" if args.canonical_eval else "all")

    # Expanded agent_parameter_for_result_file — mirror train.py's
    # ~17-key CSV schema so main_results.csv has compatible columns and
    # downstream analysis scripts can cross-reference offline / online
    # runs without reading the wrong columns (parity audit CRITICAL #3).
    custom_parameters = {
        "algo": args.algo,
        "awac_lambda": args.awac_lambda,
        "lr": args.lr_policy,
        "hidden_layers": args.hidden,
        "embedder_nodes": args.embedder_nodes,
        "embedder_layers": args.embedder_layers,
        "batch_size": args.batch_size,
        "replay_buffer_size": buffer_capacity,
        "n_updates": args.n_updates,
        "eval_every_updates": args.eval_every_updates,
        "per_alpha": args.per_alpha,
        "per_beta_start": args.per_beta_start,
        "grad_clip": args.grad_clip,
        "balanced_fraction": args.balanced_fraction,
        "demo_priority_bonus": args.demo_priority_bonus,
        "priority_mode": args.priority_mode,
        "log_std_min": args.log_std_min,
        "target_branches": args.target_branches,
    }

    # Runner.
    infos = list(env_eval_unwrapped.info.info.keys())
    runner = Runner(
        agent=agent,
        heatup_action_low=[[-10.0, -1.5], [-10.0, -1.5]],
        heatup_action_high=[[30.0, 1.5], [30.0, 1.5]],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
        diagnostics_folder=diagnostics_folder,
        policy_snapshot_every_steps=max(args.eval_every_updates, 10_000),
    )
    # Persist env + runner configs so post-hoc validation /
    # validate_experiment.py works (parity audit HIGH — env_train.yml,
    # env_eval.yml, runner.yml all missing previously). env_eval_unwrapped
    # is the bare BenchEnv5 (not the CheckpointRestoreWrapper instance)
    # so the saved config matches train.py's schema exactly.
    try:
        env_train.save_config(os.path.join(config_folder, "env_train.yml"))
        env_eval_unwrapped.save_config(os.path.join(config_folder, "env_eval.yml"))
        runner.save_config(os.path.join(config_folder, "runner.yml"))
        log.info("Saved env_train.yml / env_eval.yml / runner.yml to %s", config_folder)
    except Exception as e:
        log.warning("Config save failed (non-blocking): %s", e)

    # Eval-only mode: load checkpoint, run one eval, exit.
    if args.eval_only_checkpoint:
        log.info("[eval-only] loading %s", args.eval_only_checkpoint)
        agent.load_checkpoint(args.eval_only_checkpoint)
        runner._replay_save_interval = 10 ** 12
        quality, reward = runner.eval(seeds=eval_seeds)
        log.info("[eval-only] result: quality=%s reward=%s", quality, reward)
        agent.close()
        return

    # BC warmup (Stage 1C). Drives policy via pure log-prob on the buffer
    # for `bc_warmup_updates` steps before switching to AWAC. The cleanest
    # mechanism for now: temporarily flip the algo to "sac" with a very
    # high alpha so SAC's policy loss collapses to BC. Real BC support
    # lands when 1C is launched — for AWAC-only Stage 1A this is unused.
    if args.bc_warmup_updates > 0:
        raise NotImplementedError(
            "BC warmup branch lands with Stage 1C; not wired in Stage 1A."
        )

    # Plan v11 Stage 1B safety guardrails — baseline-quality reference +
    # quality-drop abort. The baseline can come from any of three sources
    # (most preferred first):
    #   1. --step0_eval ON — measured: full 54-seed eval at update 0
    #      (~50 min compute; mostly redundant if the warm-start
    #      checkpoint is the same one Task C already measured).
    #   2. --step0_smoke_eval_seeds N>0 — partial: N-seed quick check
    #      (~5 min for N=5) for sanity-check loaded weights.
    #   3. --baseline_quality FLOAT — hardcoded prior (default 0.577 =
    #      Task C anchor on v3 checkpoint250531.everl). Zero compute.
    step0_quality = None
    runner._is_offline_mode = True
    runner._replay_save_interval = 10 ** 12
    # Plan v11 — train-then-eval-later mode toggle. When set, offline_train
    # uses runner.save_only at each cadence boundary instead of self.eval.
    runner._skip_inline_eval = bool(args.skip_inline_eval)
    if runner._skip_inline_eval:
        log.info(
            "Stage 1 train-then-eval-later mode ON: cadence boundaries "
            "save .everl checkpoints only (no SOFA eval). Use "
            "--eval_only_checkpoint passes to score afterward."
        )
    if args.step0_eval:
        log.info(
            "Stage 1B: running FULL step-0 canonical eval "
            "(--step0_eval ON, ~50 min)..."
        )
        step0_quality, step0_reward = runner.eval(seeds=eval_seeds)
        log.info(
            "Stage 1B step-0 baseline: quality=%.3f reward=%.3f "
            "(abort_quality_drop=%.1fpp)",
            step0_quality, step0_reward, args.abort_quality_drop,
        )
    elif args.step0_smoke_eval_seeds > 0:
        smoke_seeds = eval_seeds[: int(args.step0_smoke_eval_seeds)]
        log.info(
            "Stage 1B: running %d-seed smoke eval (warm-start sanity)...",
            len(smoke_seeds),
        )
        step0_quality, step0_reward = runner.eval(seeds=smoke_seeds)
        log.info(
            "Stage 1B smoke-eval: quality=%.3f reward=%.3f on %d seeds",
            step0_quality, step0_reward, len(smoke_seeds),
        )
    else:
        # Use the hardcoded prior. This is the new default — saves
        # ~50 min of redundant compute when we already know the
        # warm-start checkpoint's canonical Quality.
        step0_quality = float(args.baseline_quality)
        log.info(
            "Stage 1B: using --baseline_quality=%.3f as the abort "
            "guardrail reference (no step-0 eval).",
            step0_quality,
        )

    log.info("Starting offline_train: %d updates, eval every %d.",
             args.n_updates, args.eval_every_updates)
    # Sanity log so the user can confirm true resume vs. fresh start at a
    # glance. offline_train uses update_start = self.step_counter.update
    # as the baseline; if --resume worked, this prints the loaded value
    # (e.g. 10000), otherwise 0.
    log.info(
        "Offline-train: starting at update=%d (resume=%s); will run to "
        "update=%d.",
        int(agent.step_counter.update),
        args.resume,
        int(agent.step_counter.update) + int(args.n_updates),
    )

    # If an abort guardrail is configured we wrap the eval method to
    # check the drop after each periodic eval inside offline_train.
    if args.abort_quality_drop > 0.0 and step0_quality is not None:
        _orig_eval = runner.eval
        _drop_thresh_pp = float(args.abort_quality_drop)
        _baseline = float(step0_quality)

        def _guarded_eval(*a, **kw):
            quality, reward = _orig_eval(*a, **kw)
            drop_pp = (_baseline - float(quality)) * 100.0
            if drop_pp > _drop_thresh_pp:
                log.error(
                    "ABORT (quality-drop guardrail): quality dropped "
                    "%.1fpp (%.3f -> %.3f) > threshold %.1fpp; halting "
                    "training to preserve checkpoint pool.",
                    drop_pp, _baseline, float(quality), _drop_thresh_pp,
                )
                raise RuntimeError(
                    f"Stage 1B abort: quality drop {drop_pp:.1f}pp "
                    f"exceeds threshold {_drop_thresh_pp:.1f}pp"
                )
            return quality, reward

        runner.eval = _guarded_eval

    try:
        runner.offline_train(
            n_updates=args.n_updates,
            eval_every_updates=args.eval_every_updates,
            eval_seeds=eval_seeds,
        )
    except RuntimeError as exc:
        log.error("Offline run aborted: %s", exc)

    log.info("Offline run complete: %s", args.name)
    agent.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
