"""Post-hoc analysis for the Phase C variant grid run (Plan v3).

Parses worker logs from a grid-test run and produces:
  1. Per-variant summary: success rate, mean reward, # reach-RVA across
     all checkpoints.
  2. Per-checkpoint summary: which variants succeeded vs failed.
  3. Heatmap matrix (checkpoints × variants → reward), saved as both
     a CSV and a printed ASCII visualization.
  4. Variant ranking: which variants robustly outperform the baseline.

Usage:
    python analyze_phase_c_grid.py <run_dir>

where ``<run_dir>`` is the timestamped run directory under
``saved/eve_paper/.../mesh_ben/`` produced by run_phase_c_grid.py.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


def parse_logs(run_dir):
    """Walk worker logs and extract per-episode results.

    Returns: list of dicts with keys:
      pid, ep, seed, phase_c_variant, checkpoint_file (from
      _restore_checkpoint_file in options), final_reward, end_step,
      reached_rva (bool), success (bool), abort_reason
    """
    log_dir = os.path.join(run_dir, "diagnostics", "logs_subprocesses")
    results = []
    start_re = re.compile(
        r"EPISODE_START \| ep=(\d+) \|.*?pid=(\d+).*?"
        r"target=\(([^)]+)\)"
        r".*?seed=(-?\d+)"
        r"(?:.*?phase_c_variant=(\w+))?"
    )
    end_re = re.compile(
        r"EPISODE_END \| ep=(\d+) \| steps=(\d+) \| "
        r"total_reward=([-0-9.]+) \|.*?pid=(\d+)"
        r"(?:.*?heur_abort=(\w+))?"
    )

    for log_path in sorted(glob.glob(os.path.join(log_dir, "worker_*.log"))):
        # Per-pid: track in-progress episode
        current = None
        in_episode_lines = []
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m_start = start_re.search(line)
                if m_start:
                    if current is not None:
                        # previous episode never got an EPISODE_END
                        results.append(current)
                    current = {
                        "ep": int(m_start.group(1)),
                        "pid": int(m_start.group(2)),
                        "target": m_start.group(3),
                        "seed": int(m_start.group(4)),
                        "phase_c_variant": m_start.group(5) or "?",
                        "reached_rva": False,
                        "abort_reason": "none",
                        "final_reward": None,
                        "end_step": None,
                    }
                    continue
                if current is None:
                    continue
                # Track RVA reach
                if "cur_branch=Centerline curve - RVA.mrk" in line:
                    current["reached_rva"] = True
                # Track abort
                m_abort = re.search(r"abort=(\w+)", line)
                if m_abort:
                    a = m_abort.group(1)
                    if a not in ("none",):
                        current["abort_reason"] = a
                m_end = end_re.search(line)
                if m_end:
                    current["end_step"] = int(m_end.group(2))
                    current["final_reward"] = float(m_end.group(3))
                    if m_end.group(5):
                        ab = m_end.group(5)
                        if ab not in ("none",):
                            current["abort_reason"] = ab
                    results.append(current)
                    current = None
        if current is not None:
            results.append(current)
    return results


def find_checkpoint_id(results, manifest):
    """Match each episode result to (checkpoint_id, phase_c_variant) by joining via seed.
    Manifest is source of truth for variant — log-parsed variant is overwritten.
    """
    seed_to_entry = {
        entry["seed"]: entry for entry in manifest["schedule"]
    }
    for r in results:
        entry = seed_to_entry.get(r["seed"])
        if entry is not None:
            r["checkpoint_id"] = entry["checkpoint_id"]
            r["phase_c_variant"] = entry["phase_c_variant"]
        else:
            r["checkpoint_id"] = -1
    return results


def per_variant_summary(results, variants):
    print("\n" + "=" * 78)
    print("Per-variant summary (across all checkpoints)")
    print("=" * 78)
    print(f"{'variant':>10} {'n':>5} {'success':>8} {'success%':>9} "
          f"{'reach RVA%':>11} {'mean R':>9} {'best R':>9}")
    print("-" * 78)
    rows = []
    for v in variants:
        sub = [r for r in results if r["phase_c_variant"] == v]
        if not sub:
            continue
        n = len(sub)
        rewards = [r["final_reward"] for r in sub if r["final_reward"] is not None]
        success = [r for r in sub if r["final_reward"] is not None
                   and r["final_reward"] > 0]
        n_succ = len(success)
        reach = sum(1 for r in sub if r["reached_rva"])
        mean_r = float(np.mean(rewards)) if rewards else float("nan")
        best_r = float(np.max(rewards)) if rewards else float("nan")
        print(f"{v:>10} {n:>5} {n_succ:>8} {100*n_succ/n:>8.1f}% "
              f"{100*reach/n:>10.1f}% {mean_r:>9.2f} {best_r:>9.2f}")
        rows.append((v, n, n_succ, mean_r, best_r))
    return rows


def per_checkpoint_heatmap(results, variants, n_checkpoints):
    """Build (n_checkpoints × n_variants) reward matrix; print ASCII heatmap."""
    R = np.full((n_checkpoints, len(variants)), float("nan"))
    cp_var_ix = {v: i for i, v in enumerate(variants)}
    for r in results:
        cp = r.get("checkpoint_id", -1)
        v = r.get("phase_c_variant", "?")
        if cp < 0 or v not in cp_var_ix:
            continue
        if cp >= n_checkpoints:
            continue
        if r["final_reward"] is not None:
            R[cp, cp_var_ix[v]] = r["final_reward"]
    return R


def write_csv(R, variants, out_path):
    with open(out_path, "w") as f:
        f.write("checkpoint_id," + ",".join(variants) + "\n")
        for cp in range(R.shape[0]):
            row = [str(cp)] + [
                f"{R[cp, j]:.3f}" if not np.isnan(R[cp, j]) else "nan"
                for j in range(R.shape[1])
            ]
            f.write(",".join(row) + "\n")
    print(f"Wrote heatmap CSV to {out_path}")


def per_checkpoint_winners(R, variants):
    print("\n" + "=" * 78)
    print("Per-checkpoint winners (variant with highest reward; ties broken by index)")
    print("=" * 78)
    winner_counts = defaultdict(int)
    for cp in range(R.shape[0]):
        row = R[cp, :]
        if np.all(np.isnan(row)):
            continue
        best_j = int(np.nanargmax(row))
        winner_counts[variants[best_j]] += 1
    for v, c in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"  {v}: best on {c} checkpoints")


def main(run_dir):
    diag = os.path.join(run_dir, "diagnostics")
    if not os.path.isdir(diag):
        print(f"ERR: {diag} does not exist")
        sys.exit(1)
    manifest_path = os.path.join(diag, "phase_c_grid_manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ERR: {manifest_path} missing — was this a grid run?")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    variants = manifest["variants"]
    n_checkpoints = manifest["n_checkpoints"]
    print(f"[analyze] manifest: {n_checkpoints} checkpoints × {len(variants)} variants")

    results = parse_logs(run_dir)
    print(f"[analyze] parsed {len(results)} episode results")
    results = find_checkpoint_id(results, manifest)

    per_variant_summary(results, variants)
    R = per_checkpoint_heatmap(results, variants, n_checkpoints)
    csv_path = os.path.join(diag, "phase_c_grid_rewards.csv")
    write_csv(R, variants, csv_path)
    per_checkpoint_winners(R, variants)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
