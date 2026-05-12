"""Phase C variants — paired-comparison run.

Builds a fresh pool of N seeds (default 500) and runs every seed with
EACH of the 9 Phase C variants. Total = N * len(variants) episodes.
With paired snapshots bucketed by seed (snapshot.py change),
``snapshots/RVA/seed_<SEED>/`` will contain one PNG per variant,
side-by-side, all on the same target.

Variant is dispatched at runtime via options['phase_c_variant'] →
env5._phase_c_variant → RVAHeuristicActionFunctionFactory's polymorphic
__call__.

Schedule is shuffled so all 16 workers stay busy with mixed
(seed, variant) tuples — variants are NOT batched per worker.

Usage (inside docker):
    python3 /opt/eve_training/training_scripts/run_phase_c_seeds.py \\
        --env_version 5 -n env5_rl8_phase_c_paired_500 \\
        --episodes 500 \\
        --variants C0,C1,C2,C3,C4,C5,C6,C7,C8 \\
        --insertion_z 345 \\
        --snapshots mesh \\
        -nw 16 -d cuda:0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from util.env5 import BenchEnv5
from util.heuristic_policy_rva import (
    RVAHeuristicActionFunctionFactory,
    PHASE_C_VARIANTS,
)
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


def build_paired_schedule(n_episodes, variants, base_seed=42):
    """Build schedule = N seeds × len(variants) variants, paired.

    Schedule is NOT shuffled — entries are grouped by seed so a given
    seed's 9 variants run consecutively (within the worker pool). This
    makes per-seed snapshot folders fill up sooner during the run, at
    the cost of correlated runtimes when a particular seed wedges all
    9 variants. Workers still process entries in parallel, just with
    seed-aligned batching across the schedule timeline.
    Returns the schedule list and the underlying seeds.
    """
    seed_rng = np.random.default_rng(base_seed)
    seeds = [int(seed_rng.integers(0, 2**31)) for _ in range(n_episodes)]

    schedule = []
    for seed in seeds:
        for variant in variants:
            schedule.append((
                seed,
                {
                    "target_branch": TARGET_BRANCH,
                    "heuristic_mode": True,
                    "phase_c_variant": variant,
                },
            ))
    return schedule, seeds


def main():
    parser = argparse.ArgumentParser(
        description="Phase C variants — paired comparison."
    )
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of distinct seeds (each run with every variant).",
    )
    parser.add_argument(
        "--variants", type=str,
        default="C0,C1,C2,C3,C4,C5,C6,C7,C8",
    )
    parser.add_argument("-nw", "--n_worker", type=int, default=16)
    parser.add_argument(
        "-d", "--device", type=str, default="cuda:0",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "--env_version", type=int, default=5, choices=[5],
    )
    parser.add_argument("--insertion_z", type=float, default=345.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--snapshots", type=str, default="none",
        choices=["none", "mesh", "centerlines"],
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in PHASE_C_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s) {unknown}. Available: {list(PHASE_C_VARIANTS)}"
        )

    n_total = args.episodes * len(variants)
    print(f"[paired] seeds={args.episodes} variants={variants} "
          f"total_ep={n_total}", flush=True)

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
            print(
                f"Set SNAPSHOT_MODE={args.snapshots} SNAPSHOT_DIR={snapshots_dir}",
                flush=True,
            )
        else:
            os.environ.pop("SNAPSHOT_MODE", None)

    print(
        f"Run: {args.name} | seeds={args.episodes} | "
        f"variants={len(variants)} | total_ep={n_total} | "
        f"workers={args.n_worker} | insertion_z={args.insertion_z}",
        flush=True,
    )

    intervention_train = DualDeviceNav(insertion_z=args.insertion_z)
    env_train = BenchEnv5(
        intervention=intervention_train, mode="train", visualisation=False
    )
    intervention_eval = DualDeviceNav(insertion_z=args.insertion_z)
    env_eval = BenchEnv5(
        intervention=intervention_eval, mode="eval", visualisation=False
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

    schedule, seeds = build_paired_schedule(
        args.episodes, variants, args.base_seed,
    )

    if diagnostics_folder is not None:
        manifest = {
            "base_seed": int(args.base_seed),
            "n_seeds": args.episodes,
            "variants": variants,
            "n_episodes": len(schedule),
            "seeds": seeds,
            "schedule": [
                {
                    "schedule_idx": i,
                    "seed": int(seed),
                    "phase_c_variant": opts["phase_c_variant"],
                }
                for i, (seed, opts) in enumerate(schedule)
            ],
        }
        manifest_path = os.path.join(diagnostics_folder, "phase_c_paired_manifest.json")
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
