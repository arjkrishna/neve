#!/usr/bin/env python
"""Eval-only: run a trained policy on the 98 RCCA eval seeds, SOFA-restoring
EVERY episode from a single fixed start state.

Plan v10 follow-up. Goal: take the eval-#1 policy (the deterministic policy that
scored 32.67% across the 5 restore states) and measure it when every one of the
98 eval seeds starts from the BEST restore state (pid19116, which scored ~70% in
the original mixed eval). This isolates the per-state ceiling of that exact
policy — no training, no reward/obs/terminal changes.

Mirrors visualize_policy.py's load path (AlgoPlayOnly.from_checkpoint +
get_env_from_checkpoint) but headless, wraps env_eval with
CheckpointRestoreWrapper pointed at a single-state dir, and loops the same 98
EVAL_SEEDS used by DualDeviceNav_train.py with deterministic eval actions.

Usage (inside docker):
  python3 /opt/eve_training/training_scripts/eval_policy_from_state.py \
    --checkpoint /opt/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2026-05-25_031951_rcca_awac_v3/checkpoints/checkpoint250531.everl \
    --restore_dir /opt/eve_training/results/rcca_best_state_pid19116 \
    --out /opt/eve_training/results/eval_best_state_pid19116.csv

Optional:
  --snapshot <policy_XXXXX.pt>   override policy weights (arch still from --checkpoint)
  --max_steps N                  override truncation (default: use checkpoint's eval env)
  --seeds "1,2,3"                override the 98 eval seeds
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

import eve  # noqa: F401  (registers config classes for from_config_dict)
import eve_rl

# Make `util.*` importable (matches how DualDeviceNav_train.py is launched).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util.checkpoint_restore import CheckpointRestoreWrapper  # noqa: E402

# Exact 98 eval seeds from DualDeviceNav_train.py (EVAL_SEEDS).
EVAL_SEEDS = [
    1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 27, 31, 34,
    35, 37, 39, 42, 43, 44, 47, 48, 50, 52, 55, 56, 58, 61, 62, 63, 68, 69, 70,
    71, 73, 79, 80, 81, 84, 89, 91, 92, 93, 95, 97, 102, 103, 108, 109, 110,
    115, 116, 117, 118, 120, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132,
    134, 136, 138, 139, 140, 141, 142, 143, 144, 147, 148, 149, 150, 151, 152,
    154, 155, 156, 158, 159, 161, 162, 167, 168, 171, 175,
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Path to .everl checkpoint (provides net architecture + weights)")
    p.add_argument("--restore_dir", required=True,
                   help="Dir containing the single start-state .npz to restore every episode")
    p.add_argument("--snapshot", default=None,
                   help="Optional policy_XXXXX.pt to override the policy weights")
    p.add_argument("--out", default=None, help="CSV output path for per-episode results")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override env truncation max_steps (default: keep checkpoint's)")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seed override (default: the 98 EVAL_SEEDS)")
    p.add_argument("--rng_seed", type=int, default=43,
                   help="CheckpointRestoreWrapper rng_seed (irrelevant for a single-state dir)")
    args = p.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else EVAL_SEEDS)
    out_path = args.out or os.path.splitext(args.checkpoint)[0] + "_eval_from_state.csv"

    print(f"Loading policy from checkpoint: {args.checkpoint}")
    algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(args.checkpoint)
    if args.snapshot:
        snap = torch.load(args.snapshot, map_location="cpu")
        algo.model.policy.load_state_dict(snap["policy"])
        print(f"  overrode policy weights from snapshot (explore_step={snap.get('explore_step', '?')})")

    print("Building eval env from checkpoint config ...")
    env = eve_rl.util.get_env_from_checkpoint(args.checkpoint, "eval")
    env.intervention.make_non_mp()  # single-process stepping
    if args.max_steps is not None:
        env.truncation.max_steps = args.max_steps

    env = CheckpointRestoreWrapper(env, checkpoint_dir=args.restore_dir, rng_seed=args.rng_seed)
    if not env.checkpoint_files:
        raise SystemExit(f"No .npz checkpoints found in {args.restore_dir}")
    restore_name = os.path.basename(env.checkpoint_files[0])
    print(f"Restore pool ({len(env.checkpoint_files)} file): {restore_name}")
    if len(env.checkpoint_files) != 1:
        print("  WARNING: restore_dir has >1 file; episodes will sample among them, "
              "not the single best state.")

    print(f"\nRunning {len(seeds)} eval seeds (deterministic actions), "
          f"all restored from {restore_name}\n" + "-" * 70)

    rows = []
    n_success = 0
    for i, seed in enumerate(seeds):
        algo.reset()
        obs, info = env.reset(seed=seed)
        obs_flat, _ = eve_rl.util.flatten_obs(obs)
        ep_r, steps = 0.0, 0
        term = trunc = False
        while True:
            action = algo.get_eval_action(obs_flat)
            obs, r, term, trunc, info = env.step(action)
            obs_flat, _ = eve_rl.util.flatten_obs(obs)
            ep_r += r
            steps += 1
            if term or trunc:
                break

        success = bool(info.get("success", False))
        succ_by_term = bool(term and not trunc)
        if success:
            n_success += 1
        rows.append({
            "seed": seed,
            "success": int(success),
            "success_by_term": int(succ_by_term),
            "reward": round(ep_r, 4),
            "steps": steps,
            "terminated": int(term),
            "truncated": int(trunc),
            "reached_target_daughter": int(bool(info.get("reached_target_daughter", False))),
            "final_branch_idx": info.get("final_branch_idx"),
            "target_daughter_branch_idx": info.get("target_daughter_branch_idx"),
            "restore_ckpt": restore_name,
        })
        print(f"[{i + 1:3d}/{len(seeds)}] seed={seed:4d}  "
              f"{'SUCCESS' if success else 'fail   '}  R={ep_r:8.3f}  steps={steps:4d}  "
              f"term={int(term)} trunc={int(trunc)}")

    pct = 100.0 * n_success / len(seeds)
    print("-" * 70)
    print(f"RESULT: {n_success}/{len(seeds)} success ({pct:.2f}%)  "
          f"from start state {restore_name}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Per-episode CSV written: {out_path}")

    try:
        algo.close()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
