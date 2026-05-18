"""Keep-policy pruner for RCCA SAC training snapshots (RL_IMPROV_9, Plan v5).

env5 renders an end-of-episode PNG for every episode when SNAPSHOT_MODE is
set, bucketed by training phase:

    <snapshot_dir>/seed/<target>/<reason>/ep..._R<reward>_<reason>.png
    <snapshot_dir>/eval/<target>/<reason>/...
    <snapshot_dir>/explore/<target>/<reason>/...

Snapshotting every explore episode of a 20M-step run is ~66k PNGs. This
script enforces the keep-policy the user asked for:

    seed/    — keep ALL  (every heuristic episode fed into the buffer)
    eval/    — keep ALL  (every evaluation episode)
    explore/ — keep only the 10 best + 10 worst (by episode reward) per
               100 consecutive explore episodes

Idempotent / incremental design — no manifest needed:

    explore/        is a STAGING area. env5 writes new PNGs here.
    explore_kept/   holds finalized blocks (block_0000/, block_0001/, ...).

Each run: gather PNGs currently in explore/, sort by file mtime (= global
chronological order across all workers), form complete 100-blocks, move
each block's 20 keepers into explore_kept/block_<N>/ and delete the other
80 from explore/. The trailing partial block (< 100) is left in explore/
for the next run. Because processed episodes leave the staging area,
mtime-grouping stays stable across runs — safe to run periodically (cron)
or once post-hoc.

Usage:
    py prune_training_snapshots.py --snapshot_dir <path-to/snapshots>
    py prune_training_snapshots.py --snapshot_dir <...> --dry-run
"""
import argparse
import os
import re
import shutil
import glob


_REWARD_RE = re.compile(r"_R([+-][0-9]+\.[0-9]+)_")


def _parse_reward(path):
    """Extract the episode reward encoded in the PNG filename
    (``..._R<+/-><value>_<reason>.png``). Returns 0.0 if unparseable."""
    m = _REWARD_RE.search(os.path.basename(path))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def _next_block_index(kept_dir):
    """Lowest unused block_<N> index under explore_kept/."""
    if not os.path.isdir(kept_dir):
        return 0
    idx = 0
    for name in os.listdir(kept_dir):
        m = re.match(r"block_(\d+)$", name)
        if m:
            idx = max(idx, int(m.group(1)) + 1)
    return idx


def prune(snapshot_dir, block_size=100, keep_best=10, keep_worst=10,
          dry_run=False):
    explore_dir = os.path.join(snapshot_dir, "explore")
    kept_dir = os.path.join(snapshot_dir, "explore_kept")

    if not os.path.isdir(explore_dir):
        print(f"No explore/ subdir under {snapshot_dir} — nothing to prune.")
        return

    # Gather every PNG currently staged under explore/, chronological order.
    pngs = glob.glob(os.path.join(explore_dir, "**", "*.png"), recursive=True)
    pngs.sort(key=lambda p: os.path.getmtime(p))
    n_total = len(pngs)
    n_complete = n_total // block_size
    print(f"explore/ staged PNGs: {n_total}  →  {n_complete} complete "
          f"block(s) of {block_size}, {n_total % block_size} in the "
          f"trailing partial block (left in place).")

    if n_complete == 0:
        print("No complete block yet — nothing to finalize.")
        return

    block_idx = _next_block_index(kept_dir)
    n_moved = 0
    n_deleted = 0

    for b in range(n_complete):
        block = pngs[b * block_size:(b + 1) * block_size]
        ranked = sorted(block, key=_parse_reward)
        worst = ranked[:keep_worst]
        best = ranked[len(ranked) - keep_best:]
        # dedupe (tiny blocks could overlap; block_size=100 won't)
        keepers = []
        seen = set()
        for p in worst + best:
            if p not in seen:
                seen.add(p)
                keepers.append(p)
        losers = [p for p in block if p not in seen]

        dest = os.path.join(kept_dir, f"block_{block_idx:04d}")
        rewards = [_parse_reward(p) for p in block]
        print(f"  block_{block_idx:04d}: {len(block)} eps "
              f"(reward {min(rewards):+.2f}..{max(rewards):+.2f})  "
              f"→ keep {len(keepers)}, delete {len(losers)}")

        if not dry_run:
            os.makedirs(dest, exist_ok=True)
            for p in keepers:
                shutil.move(p, os.path.join(dest, os.path.basename(p)))
            for p in losers:
                os.remove(p)
        n_moved += len(keepers)
        n_deleted += len(losers)
        block_idx += 1

    verb = "(dry-run) would move" if dry_run else "moved"
    print(f"\n{verb} {n_moved} keepers to explore_kept/, "
          f"{'would delete' if dry_run else 'deleted'} {n_deleted} PNGs.")
    print("seed/ and eval/ are never touched — all kept.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot_dir", required=True,
                    help="Path to the run's diagnostics/snapshots directory "
                         "(contains seed/ eval/ explore/).")
    ap.add_argument("--block_size", type=int, default=100,
                    help="Explore episodes per block (default 100).")
    ap.add_argument("--keep_best", type=int, default=10,
                    help="Highest-reward episodes kept per block (default 10).")
    ap.add_argument("--keep_worst", type=int, default=10,
                    help="Lowest-reward episodes kept per block (default 10).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be moved/deleted without doing it.")
    args = ap.parse_args()
    prune(args.snapshot_dir, args.block_size, args.keep_best,
          args.keep_worst, args.dry_run)


if __name__ == "__main__":
    main()
