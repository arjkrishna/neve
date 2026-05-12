"""Heuristic-only run targeting LVA daughter ONLY.

One of four per-daughter runners (LVA / LVA / RCCA / RVA) used to
prototype daughter-specific heuristic policies WITHOUT touching the
shared observation / reward code that the eventual RL agent consumes.

Differences from the generic ``heuristic_only_run.py``:
  - ``TARGET_BRANCHES`` is a single-element list (LVA only).
  - Custom policy / controller for this daughter is plugged in via the
    ``HEURISTIC_FACTORY`` block below — keep changes confined here.

Example (from inside the SOFA container):
    python3 /opt/eve_training/training_scripts/heuristic_run_LVA.py \\
        --env_version 5 -n env5_rl8_LVA_50ep \\
        --insertion_z 345 --episodes 50 -nw 16
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from util.env5 import BenchEnv5
from util.heuristic_policy import HeuristicActionFunctionFactory
from util.heuristic_policy_lva import LVAHeuristicActionFunctionFactory
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


# ----- Per-daughter target ---------------------------------------------------
DAUGHTER_TAG = "LVA"
TARGET_BRANCHES = [
    f"Centerline curve - {DAUGHTER_TAG}.mrk",
]


def _build_heuristic_factory(args):
    """LVA-specific heuristic factory.

    Phase C only: dynamic_tangent target at the (18)→LVA daughter
    junction. No Phase A / B — default centerline-follower handles
    the trunk → (18) bridge crossing naturally (wire's ascent up
    trunk(2) past trunk-top delivers it into branch (18)).
    """
    return LVAHeuristicActionFunctionFactory(
        noise_std=0.0,
        normalize_output=True,
    )


def build_episode_schedule(n_episodes, branches, base_seed=42):
    """Branch-balanced, deterministically-shuffled (seed, options) tuples.

    With a single-element ``branches`` list every episode gets the same
    ``target_branch``. ``heuristic_mode=True`` so env5's wrong-branch /
    fold detectors fire (matches training-time heuristic seeding).
    """
    rng = np.random.default_rng(base_seed)
    schedule = []
    for i in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        branch = branches[i % len(branches)]
        schedule.append(
            (ep_seed, {"target_branch": branch, "heuristic_mode": True})
        )
    rng.shuffle(schedule)
    return schedule


def main():
    parser = argparse.ArgumentParser(
        description=f"Heuristic-only parallel run, target={DAUGHTER_TAG} only."
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True,
        help="Run name (used for results folder + log path).",
    )
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=16,
        help="Number of parallel SOFA workers.",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help=f"Total {DAUGHTER_TAG}-target heuristic episodes across all workers.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda:0",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
        help="Trainer device (networks allocated but never trained).",
    )
    parser.add_argument(
        "--env_version", type=int, default=5, choices=[5],
        help="Env version (only env5 is supported here).",
    )
    parser.add_argument(
        "--insertion_z", type=float, default=None,
        help=(
            "If set, anchor the wire at the trunk centerline point with "
            "vessel-CS z closest to this value (RL_IMPROV_8 §16; try ~345)."
        ),
    )
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--snapshots",
        type=str,
        default="none",
        choices=["none", "mesh", "centerlines"],
        help=(
            "End-of-episode snapshot backend. PNGs land under "
            "<diagnostics>/snapshots/<target>/<reason>/."
        ),
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

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

    if args.snapshots != "none" and diagnostics_folder is not None:
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
        f"Run: {args.name} | daughter={DAUGHTER_TAG} | episodes={args.episodes} "
        f"| workers={args.n_worker} | insertion_z={args.insertion_z} | "
        f"snapshots={args.snapshots}",
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

    factory = _build_heuristic_factory(args)

    schedule = build_episode_schedule(
        args.episodes, TARGET_BRANCHES, args.base_seed
    )
    print(
        f"Schedule: {len(schedule)} episodes, all targeting {DAUGHTER_TAG}",
        flush=True,
    )

    t0 = time.time()
    episodes = agent.heuristic_seed(
        episodes=args.episodes,
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
            f"Worker logs: {os.path.join(diagnostics_folder, 'logs_subprocesses')}",
            flush=True,
        )

    agent.close()


if __name__ == "__main__":
    main()
