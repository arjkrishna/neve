"""Phase C variant grid test (RL_IMPROV_8 Plan v3).

Tests every Phase C variant (C0..C8) against every SOFA checkpoint
captured at the moment a wire reached `rva_jn_arc` in a prior
v2cfg-500-inst run. Each (checkpoint, variant) pair becomes one
episode. Workers run them in parallel via heuristic_seed; the
single shared RVAHeuristicActionFunctionFactory dispatches to the
right variant at runtime via env._phase_c_variant (set in
env5.reset() from options['phase_c_variant']).

Wraps env in CheckpointRestoreWrapper (extended to honor
options['checkpoint_id']) so each episode starts from the captured
SOFA state at RVA-jn — skipping the upstream traversal entirely.

Usage (inside docker):
    python3 /opt/eve_training/training_scripts/run_phase_c_grid.py \\
        --env_version 5 -n phase_c_grid \\
        --checkpoint_dir /opt/eve_training/results/.../rva_checkpoints \\
        --variants C0,C1,C2,C3,C4,C5,C6,C7,C8 \\
        --episodes_per_variant 0 \\
        -nw 16 -d cuda:0

`--episodes_per_variant 0` (default) means one episode per checkpoint
per variant, i.e. N_checkpoints * len(variants) total.
"""
import argparse
import json
import os
import time
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp

from util.env5 import BenchEnv5
from util.heuristic_policy_rva import (
    RVAHeuristicActionFunctionFactory,
    PHASE_C_VARIANTS,
)
from util.checkpoint_restore import CheckpointRestoreWrapper
from util.agent import BenchAgentSynchron
from util.util import get_result_checkpoint_config_and_log_path
from eve_bench import DualDeviceNav


RESULTS_FOLDER = os.path.join(
    "/opt/eve_training/results",
    "eve_paper",
    "neurovascular",
    "full",
    "mesh_ben",
)

DAUGHTER_TAG = "RVA"
TARGET_BRANCH = f"Centerline curve - {DAUGHTER_TAG}.mrk"


def build_grid_schedule(
    n_checkpoints: int,
    variants: List[str],
    base_seed: int = 42,
    episodes_per_pair: int = 1,
):
    """Build factorial schedule = checkpoints × variants × episodes_per_pair.

    Each (cp_id, variant) gets a deterministic seed derived from
    (base_seed, cp_id, variant), so re-runs of the same grid are
    reproducible. The schedule is shuffled to balance load across workers
    (all variants would run on cp 0 first if not shuffled).
    """
    rng = np.random.default_rng(base_seed)
    schedule = []
    for cp_id in range(n_checkpoints):
        for variant in variants:
            for ep_in_pair in range(episodes_per_pair):
                # Deterministic seed: same (cp, variant) → same seed across runs
                ep_seed = int(
                    abs(hash((base_seed, cp_id, variant, ep_in_pair))) & 0x7fffffff
                )
                schedule.append((
                    ep_seed,
                    {
                        "target_branch": TARGET_BRANCH,
                        "heuristic_mode": True,
                        "checkpoint_id": cp_id,
                        "phase_c_variant": variant,
                    },
                ))
    rng.shuffle(schedule)
    return schedule


def main():
    parser = argparse.ArgumentParser(
        description="Phase C variant grid test (factorial: states × strategies)"
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True,
        help="Run name (used for results folder + log path).",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Directory of *.npz SOFA checkpoints captured at RVA-jn.",
    )
    parser.add_argument(
        "--variants", type=str,
        default="C0,C1,C2,C3,C4,C5,C6,C7,C8",
        help="Comma-separated variant IDs to test (default: all 9).",
    )
    parser.add_argument(
        "--episodes_per_variant", type=int, default=1,
        help="Episodes per (checkpoint, variant) pair (default 1).",
    )
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=16,
        help="Number of parallel SOFA workers.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda:0",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "--env_version", type=int, default=5, choices=[5],
    )
    parser.add_argument(
        "--insertion_z", type=float, default=345.0,
        help="Wire insertion anchor (must match the run that produced the "
             "checkpoints; default 345 matches v2cfg-500-inst).",
    )
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--snapshots", type=str, default="none",
        choices=["none", "mesh", "centerlines"],
    )
    parser.add_argument(
        "--max_checkpoints", type=int, default=0,
        help="If >0, limit number of checkpoints used (for quick smoke test).",
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    # Validate checkpoints
    if not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint dir {args.checkpoint_dir} does not exist"
        )
    import glob as _glob
    npz_files = sorted(_glob.glob(os.path.join(args.checkpoint_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(
            f"No *.npz files found in {args.checkpoint_dir}"
        )
    n_checkpoints = len(npz_files)
    if args.max_checkpoints > 0 and args.max_checkpoints < n_checkpoints:
        n_checkpoints = args.max_checkpoints
        print(f"[grid] Limiting to first {n_checkpoints} checkpoints "
              f"(of {len(npz_files)} available) for smoke test", flush=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in PHASE_C_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s) {unknown}. Available: {list(PHASE_C_VARIANTS)}"
        )

    print(f"[grid] checkpoints={n_checkpoints} variants={variants} "
          f"episodes_per_pair={args.episodes_per_variant}", flush=True)

    n_episodes = n_checkpoints * len(variants) * args.episodes_per_variant
    print(f"[grid] total episodes = {n_episodes}", flush=True)

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
        _config_folder_dup,
        diagnostics_folder,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=args.name, create_diagnostics=True
    )

    if diagnostics_folder is not None:
        logs_subprocesses = os.path.join(diagnostics_folder, "logs_subprocesses")
        os.makedirs(logs_subprocesses, exist_ok=True)
        os.environ["STEP_LOG_DIR"] = logs_subprocesses
        print(f"Set STEP_LOG_DIR={logs_subprocesses}", flush=True)

        if args.snapshots != "none":
            snapshots_dir = os.path.join(diagnostics_folder, "snapshots")
            os.makedirs(snapshots_dir, exist_ok=True)
            os.environ["SNAPSHOT_MODE"] = args.snapshots
            os.environ["SNAPSHOT_DIR"] = snapshots_dir
        else:
            os.environ.pop("SNAPSHOT_MODE", None)

    print(
        f"Run: {args.name} | checkpoints={n_checkpoints} | "
        f"variants={len(variants)} | total_ep={n_episodes} | "
        f"workers={args.n_worker} | insertion_z={args.insertion_z}",
        flush=True,
    )

    # Build env factories. CheckpointRestoreWrapper wraps BenchEnv5 so that
    # each episode reset injects the SOFA state from the chosen checkpoint
    # (selected via options['checkpoint_id'] per the §28 extension).
    intervention_train = DualDeviceNav(insertion_z=args.insertion_z)
    env_train = BenchEnv5(
        intervention=intervention_train, mode="train", visualisation=False
    )
    env_train = CheckpointRestoreWrapper(
        env_train, args.checkpoint_dir, rng_seed=args.base_seed
    )
    intervention_eval = DualDeviceNav(insertion_z=args.insertion_z)
    env_eval = BenchEnv5(
        intervention=intervention_eval, mode="eval", visualisation=False
    )
    env_eval = CheckpointRestoreWrapper(
        env_eval, args.checkpoint_dir, rng_seed=args.base_seed
    )

    agent = BenchAgentSynchron(
        trainer_device=torch.device(args.device),
        worker_device=torch.device("cpu"),
        lr=1e-4,
        lr_end_factor=0.15,
        lr_linear_end_steps=int(6e6),
        hidden_layers=[64, 64],
        embedder_nodes=64,
        embedder_layers=1,
        gamma=0.99,
        batch_size=32,
        reward_scaling=1,
        replay_buffer_size=100,
        env_train=env_train,
        env_eval=env_eval,
        consecutive_action_steps=1,
        n_worker=args.n_worker,
        stochastic_eval=False,
        ff_only=False,
        diagnostics_config=None,
    )

    factory = RVAHeuristicActionFunctionFactory(
        noise_std=0.0, normalize_output=True,
    )

    schedule = build_grid_schedule(
        n_checkpoints, variants, args.base_seed,
        episodes_per_pair=args.episodes_per_variant,
    )

    # Save manifest
    if diagnostics_folder is not None:
        manifest = {
            "base_seed": int(args.base_seed),
            "n_checkpoints": n_checkpoints,
            "variants": variants,
            "episodes_per_pair": args.episodes_per_variant,
            "n_episodes": len(schedule),
            "checkpoint_files": [os.path.basename(p) for p in npz_files[:n_checkpoints]],
            "schedule": [
                {
                    "schedule_idx": i,
                    "seed": int(seed),
                    "checkpoint_id": opts["checkpoint_id"],
                    "phase_c_variant": opts["phase_c_variant"],
                }
                for i, (seed, opts) in enumerate(schedule)
            ],
        }
        manifest_path = os.path.join(diagnostics_folder, "phase_c_grid_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {manifest_path}", flush=True)

    t0 = time.time()
    episodes = agent.heuristic_seed(
        episodes=len(schedule),
        heuristic_factory=factory,
        episode_schedule=schedule,
        push_to_buffer=False,
    )
    elapsed = time.time() - t0
    n_ep = len(episodes) if episodes is not None else 0

    n_success = 0
    if episodes is not None:
        for ep in episodes:
            try:
                if ep.infos and "success" in ep.infos[-1]:
                    if bool(ep.infos[-1]["success"]):
                        n_success += 1
            except Exception:
                pass

    print(
        f"\nDone in {elapsed:.1f}s | episodes_collected={n_ep} | "
        f"successes={n_success}",
        flush=True,
    )
    if diagnostics_folder is not None:
        print(
            f"Worker logs: "
            f"{os.path.join(diagnostics_folder, 'logs_subprocesses')}",
            flush=True,
        )

    agent.close()


if __name__ == "__main__":
    main()
