"""Heuristic-only run for the §16 high-insertion-point A/B test.

Runs N parallel heuristic episodes (no SAC training, no eval, no replay
buffer push) and exits. Logs land in
``<RESULTS_FOLDER>/<name>/diagnostics/logs_subprocesses/worker_<pid>.log``
so the existing ``analyze_run28_branches.py`` can compute the
(80, 55, 395) wedge rate without any changes.

Pass ``--insertion_z`` to anchor the wire high in the trunk
(RL_IMPROV_8 §16). Omit it to run from the femoral entry baseline.

Example (from inside the SOFA container):
    python3 /opt/eve_training/training_scripts/heuristic_only_run.py \\
        --env_version 5 -n env5_rl8_highinsert_50ep \\
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


TARGET_BRANCHES = [
    "Centerline curve - LCCA.mrk",
    "Centerline curve - LVA.mrk",
    "Centerline curve - RCCA.mrk",
    "Centerline curve - RVA.mrk",
]


def build_episode_schedule(n_episodes, branches, base_seed=42):
    """Branch-balanced, deterministically-shuffled (seed, options) tuples.

    Always sets ``heuristic_mode=True`` so env5's wrong-branch / fold
    detectors fire (matches training-time heuristic seeding).
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
        description=(
            "Heuristic-only parallel run (no SAC training). "
            "Use with --insertion_z for the §16 high-insertion A/B test."
        )
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
        help="Total heuristic episodes across all workers.",
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

    # Workers inherit STEP_LOG_DIR via os.environ at spawn time. Must be set
    # BEFORE BenchAgentSynchron creates worker processes.
    if diagnostics_folder is not None:
        logs_subprocesses = os.path.join(diagnostics_folder, "logs_subprocesses")
        os.makedirs(logs_subprocesses, exist_ok=True)
        os.environ["STEP_LOG_DIR"] = logs_subprocesses
        print(f"Set STEP_LOG_DIR={logs_subprocesses}", flush=True)

    print(
        f"Run: {args.name} | episodes={args.episodes} | workers={args.n_worker} | "
        f"insertion_z={args.insertion_z}",
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

    # BenchAgentSynchron requires a SAC model + replay buffer, but neither
    # is exercised: heuristic_seed(push_to_buffer=False) skips the push,
    # and we never call .update() / .explore(). Use the smallest viable
    # network config to keep memory low.
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

    factory = HeuristicActionFunctionFactory(
        noise_std=0.0,
        normalize_output=True,
    )

    schedule = build_episode_schedule(
        args.episodes, TARGET_BRANCHES, args.base_seed
    )
    branch_counts = {
        b.split(" - ")[-1].replace(".mrk", ""): sum(
            1 for _, o in schedule if o["target_branch"] == b
        )
        for b in TARGET_BRANCHES
    }
    print(
        f"Schedule: {len(schedule)} episodes -> per-branch={branch_counts}",
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
        print(f"Worker logs: {os.path.join(diagnostics_folder, 'logs_subprocesses')}", flush=True)

    agent.close()


if __name__ == "__main__":
    main()
