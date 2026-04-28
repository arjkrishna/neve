"""Collect SOFA-state checkpoints at ~380 mm guidewire insertion (parallel).

Reuses the existing BenchAgentSynchron worker pool (same plumbing as
training-time heuristic seeding) so SOFA episodes run across N workers in
parallel. Each worker drives the centerline-following heuristic and, on the
first step where guidewire insertion enters [370, 385] mm, dumps the live
SOFA controller + DOF state to .npz + .json. No replay buffer push.

Usage:
    python "training _scripts/collect_sofa_checkpoints.py" \
        --episodes 200 -nw 16 -d cuda:0 \
        --out /opt/eve_training/results/sofa_checkpoints
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from util.env5 import BenchEnv5
from util.heuristic_policy import HeuristicActionFunction
from util.agent import BenchAgentSynchron
from eve_bench import DualDeviceNav


TARGET_BRANCHES = [
    "Centerline curve - LCCA.mrk",
    "Centerline curve - LVA.mrk",
    "Centerline curve - RCCA.mrk",
    "Centerline curve - RVA.mrk",
]

CAPTURE_DEPTH_MIN_MM = 370.0
CAPTURE_DEPTH_MAX_MM = 385.0
WIRE_SHAPE_SCORE_THRESHOLD = 0.25


def build_episode_schedule(n_episodes, branches, base_seed=42):
    """Branch-balanced, deterministically-shuffled (seed, options) tuples.

    Mirrors DualDeviceNav_train.build_episode_schedule but always sets
    heuristic_mode=True so env5's wrong-branch / fold detectors fire — same
    behaviour as training-time heuristic seeding.
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


def compute_wire_shape_score(tracking3d, n_points=5):
    """Coefficient of variation of distances between the first n tracked nodes.

    Clean wire ≈ 0 (uniform spacing); folds/loops produce a spike.
    """
    pts = tracking3d[:n_points]
    if len(pts) < 2:
        return float("inf")
    consec = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mean = float(np.mean(consec))
    if mean < 1e-6:
        return float("inf")
    return float(np.std(consec) / mean)


# ---------------------------------------------------------------------------
# Per-worker action function: heuristic + SOFA-state save side-effect.
# Each worker process gets its own instance (created via factory.create(env)
# inside the worker), so file writes go straight to the shared output dir.
# ---------------------------------------------------------------------------

class CheckpointCollectingActionFunction(HeuristicActionFunction):
    """HeuristicActionFunction that snapshots SOFA state on the first step
    inside [capture_min_mm, capture_max_mm] of guidewire insertion (one
    capture per episode max)."""

    def __init__(
        self,
        env,
        output_dir,
        capture_min_mm=370.0,
        capture_max_mm=385.0,
        quality_threshold=0.25,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.output_dir = output_dir
        self.capture_min = capture_min_mm
        self.capture_max = capture_max_mm
        self.quality_threshold = quality_threshold
        self._captured_in_window = False
        self._ep_idx = 0
        self._step_idx = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def reset(self):
        super().reset()
        self._captured_in_window = False
        self._ep_idx += 1
        self._step_idx = 0

    def __call__(self, obs):
        action = super().__call__(obs)
        self._step_idx += 1
        try:
            base_env = getattr(self.env, "unwrapped", self.env)
            inserted = float(base_env.intervention.device_lengths_inserted[0])
            if (
                not self._captured_in_window
                and self.capture_min <= inserted <= self.capture_max
            ):
                self._save(base_env, inserted)
                self._captured_in_window = True
        except Exception as e:
            print(
                f"CHECKPOINT_CAPTURE_ERR pid={os.getpid()} err={e}",
                flush=True,
            )
        return action

    def _save(self, base_env, inserted_gw):
        sim = base_env.intervention.simulation
        ic = sim._instruments_combined
        state = {
            "xtip": np.array(ic.m_ircontroller.xtip.value, copy=True),
            "rotation_instrument": np.array(
                ic.m_ircontroller.rotationInstrument.value, copy=True
            ),
            "index_first_node": np.array(
                ic.m_ircontroller.indexFirstNode.value, copy=True
            ),
            "dof_positions": np.array(ic.DOFs.position.value, copy=True),
            "tracking3d": np.array(
                base_env.intervention.fluoroscopy.tracking3d, copy=True
            ),
        }

        score = compute_wire_shape_score(state["tracking3d"])
        on_correct_branch = True
        try:
            on_correct_branch = bool(base_env._path_context.is_on_correct_branch())
        except Exception:
            pass
        looks_clean = bool(
            score < self.quality_threshold and on_correct_branch
        )
        try:
            target_coords = list(
                np.array(
                    base_env.intervention.target.coordinates3d, copy=True
                ).flatten().astype(float)
            )
        except Exception:
            target_coords = None

        pid = os.getpid()
        base = f"ckpt_pid{pid}_ep{self._ep_idx:04d}_step{self._step_idx:04d}"
        npz_path = os.path.join(self.output_dir, base + ".npz")
        json_path = os.path.join(self.output_dir, base + ".json")

        np.savez(npz_path, **state)
        meta = {
            "pid": pid,
            "episode_idx": self._ep_idx,
            "step_idx": self._step_idx,
            "target_coordinates3d": target_coords,
            "inserted_lengths": [
                float(x) for x in base_env.intervention.device_lengths_inserted
            ],
            "is_on_correct_branch": on_correct_branch,
            "wire_shape_score": score,
            "looks_clean": looks_clean,
            "wall_time": time.time(),
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        msg = (
            f"CAPTURE pid={pid} ep={self._ep_idx} step={self._step_idx} "
            f"inserted_gw={inserted_gw:.2f} score={score:.3f} "
            f"on_branch={on_correct_branch} clean={looks_clean} -> {base}.npz"
        )
        print(msg, flush=True)


class CheckpointCollectingActionFunctionFactory:
    """Pickleable factory shipped to each worker; creates the action function
    in-process so it has direct access to the worker's env."""

    def __init__(
        self,
        output_dir,
        capture_min_mm=370.0,
        capture_max_mm=385.0,
        quality_threshold=0.25,
        noise_std=0.0,
        normalize_output=True,
    ):
        self.output_dir = output_dir
        self.capture_min_mm = capture_min_mm
        self.capture_max_mm = capture_max_mm
        self.quality_threshold = quality_threshold
        self.noise_std = noise_std
        self.normalize_output = normalize_output

    def create(self, env):
        return CheckpointCollectingActionFunction(
            env=env,
            output_dir=self.output_dir,
            capture_min_mm=self.capture_min_mm,
            capture_max_mm=self.capture_max_mm,
            quality_threshold=self.quality_threshold,
            noise_std=self.noise_std,
            normalize_output=self.normalize_output,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Parallel SOFA-state checkpoint collection at ~380 mm insertion."
    )
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=16,
        help="Number of parallel SOFA workers.",
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Total episodes across all workers (split round-robin).",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
        help="Trainer device. Networks are allocated but never trained.",
    )
    parser.add_argument(
        "--out", type=str, default="/opt/eve_training/results/sofa_checkpoints",
        help="Output dir for .npz + .json (must be on a shared mount).",
    )
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--noise_std", type=float, default=0.0,
        help=(
            "Heuristic action noise std (0 matches training-time seeding; "
            "the underlying CenterlineFollowerHeuristic still adds 10%% "
            "internal noise for trajectory diversity)."
        ),
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    os.makedirs(args.out, exist_ok=True)

    intervention_train = DualDeviceNav()
    env_train = BenchEnv5(
        intervention=intervention_train, mode="train", visualisation=False
    )
    intervention_eval = DualDeviceNav()
    env_eval = BenchEnv5(
        intervention=intervention_eval, mode="eval", visualisation=False
    )

    # BenchAgentSynchron requires a SAC model + replay buffer, but neither is
    # exercised: heuristic_seed(push_to_buffer=False) skips the buffer push,
    # and no .update() / .explore() is called. Use the smallest viable network
    # config to keep memory low.
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

    factory = CheckpointCollectingActionFunctionFactory(
        output_dir=args.out,
        capture_min_mm=CAPTURE_DEPTH_MIN_MM,
        capture_max_mm=CAPTURE_DEPTH_MAX_MM,
        quality_threshold=WIRE_SHAPE_SCORE_THRESHOLD,
        noise_std=args.noise_std,
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
    print(
        f"Workers: {args.n_worker} | Output: {args.out} | "
        f"Capture window: [{CAPTURE_DEPTH_MIN_MM}, {CAPTURE_DEPTH_MAX_MM}] mm",
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
    print(
        f"\nDone in {elapsed:.1f}s | episodes_collected={n_ep}",
        flush=True,
    )

    npz_files = sorted(f for f in os.listdir(args.out) if f.endswith(".npz"))
    print(f"Captures: {len(npz_files)} .npz files in {args.out}", flush=True)

    agent.close()


if __name__ == "__main__":
    main()
