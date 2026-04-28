"""Curate SOFA-state checkpoints produced by collect_sofa_checkpoints.py.

Selection criteria (per branch):

  1. Include every capture whose ending episode was a success
     (heur_abort=none AND total_reward > 0). These are gold-standard.
  2. Fill remaining slots from the "good" pool:
        looks_clean=True  AND
        cum_reward_at_capture >= 0  AND
        wire_shape_score < quality_threshold
     Ranked by (step_idx ASC, wire_shape_score ASC, cum_reward_at_capture DESC).
     Shorter step_idx = fewer bif1 oscillations = less chance of hidden wire
     kinking outside the 5-point tracking window.
  3. If the good pool is exhausted before `per_branch` is hit, relax the
     cum_reward requirement (fall back to lowest step_idx).
  4. Per-branch clustering via k-means (k=4) on target_coordinates3d, with
     clusters labeled by quadrant of their centroid for run-to-run stability.

Usage:
    python "training _scripts/select_sofa_checkpoints.py" \
        --src saved/sofa_checkpoints \
        --out saved/sofa_checkpoints/selected \
        --per_branch 10
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


RE_EPISODE_START = re.compile(
    r"EPISODE_START \| ep=(?P<ep>\d+)"
)
RE_EPISODE_END = re.compile(
    r"EPISODE_END \| ep=(?P<ep>\d+) \| steps=(?P<steps>\d+).*?"
    r"total_reward=(?P<tr>-?\d+\.\d+)"
    r"(?:.*?heur_abort=(?P<abort>[a-z_]+))?"
)
RE_STEP_DEBUG = re.compile(
    r"STEP \| ep=(?P<ep>\d+) \| ep_step=(?P<st>\d+).*?cum_reward=(?P<cr>-?\d+\.\d+)"
)


def parse_worker_log(path: str) -> Dict[int, Dict]:
    """Return {ep_idx: {"end": {...}, "cum_by_step": {ep_step: cum_reward}}}."""
    episodes: Dict[int, Dict] = defaultdict(lambda: {"end": None, "cum_by_step": {}})
    with open(path, errors="ignore") as f:
        for line in f:
            m = RE_EPISODE_END.search(line)
            if m:
                ep = int(m["ep"])
                episodes[ep]["end"] = {
                    "steps": int(m["steps"]),
                    "total_reward": float(m["tr"]),
                    "heur_abort": m["abort"] or "unknown",
                }
                continue
            m = RE_STEP_DEBUG.search(line)
            if m:
                ep = int(m["ep"])
                episodes[ep]["cum_by_step"][int(m["st"])] = float(m["cr"])
    return dict(episodes)


def cum_reward_at(episode_data: Dict, step_idx: int) -> Optional[float]:
    """Return cum_reward just before action at step_idx was applied. Our
    collection code increments _step_idx BEFORE saving, so step_idx=K
    corresponds to the env state after (K-1) env.step()s. We look up the
    nearest ep_step <= step_idx - 1; falls back to an exact or nearest
    match if needed."""
    cbs = episode_data.get("cum_by_step", {})
    if not cbs:
        return None
    target = step_idx - 1
    if target in cbs:
        return cbs[target]
    # nearest lower
    lowers = [k for k in cbs if k <= target]
    if lowers:
        return cbs[max(lowers)]
    # nearest higher as last resort
    return cbs[min(cbs.keys())]


def kmeans_xyz(points: np.ndarray, k: int = 4, max_iter: int = 50, seed: int = 0) -> np.ndarray:
    """Plain k-means on 3D points. Returns cluster labels [0..k-1]."""
    rng = np.random.default_rng(seed)
    i0 = int(rng.integers(0, len(points)))
    centroids = [points[i0]]
    for _ in range(k - 1):
        d = np.min(
            np.linalg.norm(points[:, None, :] - np.stack(centroids)[None, :, :], axis=-1),
            axis=1,
        )
        centroids.append(points[int(np.argmax(d))])
    centroids = np.stack(centroids)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        d = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = np.argmin(d, axis=1)
        new_c = np.stack([
            points[new_labels == kk].mean(axis=0) if (new_labels == kk).any() else centroids[kk]
            for kk in range(k)
        ])
        if np.allclose(new_c, centroids, atol=1e-3):
            labels = new_labels
            break
        centroids = new_c
        labels = new_labels
    # Relabel by quadrant of centroid for stability across runs.
    mean_x = centroids[:, 0].mean()
    mean_y = centroids[:, 1].mean()

    def quad(c):
        sx = c[0] < mean_x
        sy = c[1] > mean_y
        return 0 if (sx and sy) else 1 if (sx and not sy) else 2 if (not sx and sy) else 3

    order = sorted(range(k), key=lambda kk: quad(centroids[kk]))
    relabel = {old: new for new, old in enumerate(order)}
    return np.array([relabel[l] for l in labels]), centroids[order]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="saved/sofa_checkpoints",
                   help="Dir with ckpt_*.npz + ckpt_*.json + worker_*.log")
    p.add_argument("--out", default="saved/sofa_checkpoints/selected",
                   help="Output dir for curated .npz/.json (wiped first)")
    p.add_argument("--per_branch", type=int, default=10)
    p.add_argument("--quality_threshold", type=float, default=0.25,
                   help="Upper bound on wire_shape_score for the 'good pool'")
    args = p.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)

    # ------ Load worker logs ------
    worker_logs: Dict[int, Dict[int, Dict]] = {}
    for lp in glob.glob(os.path.join(src, "worker_*.log")):
        pid = int(re.search(r"worker_(\d+)\.log", os.path.basename(lp)).group(1))
        worker_logs[pid] = parse_worker_log(lp)
    print(f"Loaded worker logs for {len(worker_logs)} pids")

    # ------ Load all captures ------
    records = []
    for jpath in sorted(glob.glob(os.path.join(src, "ckpt_*.json"))):
        base = os.path.basename(jpath)
        m = re.match(r"ckpt_pid(\d+)_ep(\d+)_step(\d+)\.json", base)
        if not m:
            continue
        pid, ep, step = int(m.group(1)), int(m.group(2)), int(m.group(3))
        meta = json.load(open(jpath))
        ep_log = worker_logs.get(pid, {}).get(ep, {})
        cum_at = cum_reward_at(ep_log, step)
        end = ep_log.get("end")
        is_success = bool(
            end and end.get("heur_abort") == "none" and end.get("total_reward", 0.0) > 0.0
        )
        tc = meta.get("target_coordinates3d")
        if tc is None:
            continue
        records.append({
            "json": jpath,
            "npz": jpath[:-5] + ".npz",
            "base": base[:-5],  # stem
            "pid": pid, "ep": ep, "step": step,
            "target": np.array(tc),
            "score": float(meta.get("wire_shape_score", float("inf"))),
            "clean": bool(meta.get("looks_clean", False)),
            "cum_reward_at_capture": cum_at,
            "episode_end": end,
            "is_success": is_success,
        })
    print(f"Loaded {len(records)} captures; {sum(1 for r in records if r['is_success'])} from successful episodes")

    if not records:
        print("No captures found.", file=sys.stderr)
        return 1

    # ------ Cluster ------
    targets = np.stack([r["target"] for r in records])
    labels, centroids = kmeans_xyz(targets, k=4)
    cluster_names = [f"cluster_{c}" for c in "ABCD"]
    print("\nCluster centroids (quadrant-sorted):")
    for kk in range(4):
        c = centroids[kk]
        n = int((labels == kk).sum())
        s = int(sum(1 for i, r in enumerate(records) if labels[i] == kk and r["is_success"]))
        print(f"  {cluster_names[kk]}  x={c[0]:+6.1f}  y={c[1]:+6.1f}  z={c[2]:+6.1f}   n={n:3d} (successes={s})")

    for i, r in enumerate(records):
        r["cluster"] = cluster_names[int(labels[i])]

    # ------ Per-cluster selection ------
    selections = []
    for cname in cluster_names:
        members = [r for r in records if r["cluster"] == cname]
        successes = [r for r in members if r["is_success"]]
        non_success_pool = [r for r in members if not r["is_success"]]

        # Primary pool: positive cum_reward at capture, clean, score under threshold
        primary = [
            r for r in non_success_pool
            if r["clean"]
            and r["score"] < args.quality_threshold
            and r["cum_reward_at_capture"] is not None
            and r["cum_reward_at_capture"] >= 0.0
        ]
        primary.sort(key=lambda r: (r["step"], r["score"], -(r["cum_reward_at_capture"] or 0.0)))

        kept = list(successes)
        want = max(0, args.per_branch - len(kept))
        kept.extend(primary[:want])

        # Fallback: if short, fill from remaining clean captures by step_idx
        if len(kept) < args.per_branch:
            seen_bases = {r["base"] for r in kept}
            fallback = [
                r for r in non_success_pool
                if r["clean"] and r["base"] not in seen_bases
            ]
            fallback.sort(key=lambda r: (r["step"], r["score"]))
            kept.extend(fallback[: args.per_branch - len(kept)])

        selections.append((cname, kept))
        cum_str = [
            f"{r['cum_reward_at_capture']:+.3f}" if r["cum_reward_at_capture"] is not None else "?"
            for r in kept
        ]
        steps_str = [str(r["step"]) for r in kept]
        print(f"\n{cname}: kept {len(kept)} (successes={len(successes)}, primary_pool={len(primary)})")
        print(f"  steps: {steps_str}")
        print(f"  cum_reward_at_capture: {cum_str}")

    # ------ Copy to out dir (wipe first) ------
    if os.path.isdir(out):
        for f in os.listdir(out):
            full = os.path.join(out, f)
            if os.path.isfile(full):
                os.remove(full)
    else:
        os.makedirs(out, exist_ok=True)

    n_copied = 0
    for cname, kept in selections:
        for r in kept:
            shutil.copy2(r["npz"], os.path.join(out, f"{cname}_{os.path.basename(r['npz'])}"))
            shutil.copy2(r["json"], os.path.join(out, f"{cname}_{os.path.basename(r['json'])}"))
            n_copied += 1

    print(f"\nTotal selected: {n_copied} captures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
