"""Track tip3d against ALL 25 centerlines (named + numbered) to see where the
wire actually ends up. Where the named-only confusion matrix shows 'none', this
reveals which sub-branch or bridge curve the wire wandered into.
"""
import re
import glob
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
from analyze_branches import load_curve, CENTERLINE_DIR, vessel_to_tracking3d

NAMED = ['LCCA', 'LVA', 'RCCA', 'RVA']
LOG_BASE = Path('saved/eve_paper/neurovascular/full/mesh_ben')


def load_all_centerlines():
    """Load all 25 centerlines (named + numbered) in tracking3d frame.

    Returns: dict {short_name: (poly_t3d, cum_arclength)}
    """
    out = {}
    for f in sorted(CENTERLINE_DIR.glob('*.json')):
        if 'curve - ' in f.name:
            name = f.stem.replace('.mrk', '').replace('Centerline curve - ', '')
        else:
            name = f.stem.replace('.mrk', '').replace('Centerline curve ', '')
        poly = vessel_to_tracking3d(load_curve(f))
        if len(poly) < 2:
            continue
        diffs = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(diffs)])
        out[name] = (poly, cum)
    return out


def deepest_branch_per_episode(tip_traj, all_centerlines, perp_thresh=8.0):
    """For each centerline, find max arclength penetration (where any tip
    came within perp_thresh mm of that centerline). Return a list of
    (centerline_name, max_arclength, perp_distance_at_that_point) sorted by
    max_arclength descending. Top entry = deepest reached centerline."""
    results = []
    for name, (poly, cum) in all_centerlines.items():
        dists = np.linalg.norm(tip_traj[:, None, :] - poly[None, :, :], axis=2)
        nearest_pt_idx = dists.argmin(axis=1)
        per_tip_min = dists[np.arange(len(tip_traj)), nearest_pt_idx]
        within = per_tip_min < perp_thresh
        if not within.any():
            continue
        valid_arcs = cum[nearest_pt_idx[within]]
        deepest_idx = int(valid_arcs.argmax())
        results.append((name, float(valid_arcs[deepest_idx]),
                        float(per_tip_min[within][deepest_idx])))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def classify_target(target_xyz, branch_polys):
    return min(branch_polys,
               key=lambda n: np.linalg.norm(branch_polys[n] - target_xyz, axis=1).min())


def parse_run(log_dir):
    rows = []
    for f in sorted(glob.glob(str(log_dir / 'diagnostics/logs_subprocesses/worker_*.log'))):
        cur = None
        for line in open(f):
            if 'EPISODE_START' in line:
                if cur and cur.get('tips') and 'target' in cur:
                    rows.append(cur)
                m = re.search(
                    r'EPISODE_START \| ep=(\d+).*target=\(([-\d.]+),([-\d.]+),([-\d.]+)\)',
                    line)
                if m:
                    cur = {'ep': int(m.group(1)), 'tips': [],
                           'target': np.array([float(m.group(2)), float(m.group(3)),
                                               float(m.group(4))])}
                continue
            if cur is None:
                continue
            m = re.search(r'STEP \|.*tip3d=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
            if m:
                cur['tips'].append(
                    [float(m.group(1)), float(m.group(2)), float(m.group(3))])
                continue
            m = re.search(r'EPISODE_END \|.*total_reward=([-\d.]+)', line)
            if m:
                cur['reward'] = float(m.group(1))
        if cur and cur.get('tips') and 'target' in cur:
            rows.append(cur)
    return rows


def main():
    centerlines = load_all_centerlines()
    print(f'Loaded {len(centerlines)} centerlines')
    # Named for target classification
    named_polys = {n: centerlines[n][0] for n in NAMED if n in centerlines}

    runs_to_analyze = ['ckpttest17', 'ckpttest23', 'ckpttest26']
    for run_name in runs_to_analyze:
        matches = list(LOG_BASE.glob(f'*{run_name}'))
        if not matches:
            continue
        run_dir = matches[0]
        episodes = parse_run(run_dir)
        if not episodes:
            continue

        # Bucket by target branch
        by_target = defaultdict(list)
        for ep in episodes:
            tgt = classify_target(ep['target'], named_polys)
            by_target[tgt].append(ep)

        print(f'\n=== {run_name} ({len(episodes)} eps) ===')
        for tgt in NAMED:
            eps = by_target[tgt]
            if not eps:
                continue
            # Where does the wire go for these target=tgt episodes?
            deepest_centerline_counts = Counter()
            no_branch = 0
            for ep in eps:
                tips = np.array(ep['tips'])
                if len(tips) == 0:
                    no_branch += 1
                    continue
                results = deepest_branch_per_episode(tips, centerlines)
                if not results:
                    no_branch += 1
                    continue
                # Take top 1
                top_branch = results[0][0]
                deepest_centerline_counts[top_branch] += 1
            print(f'  TARGET={tgt} ({len(eps)} eps) — wire went DEEPEST into:')
            top = deepest_centerline_counts.most_common(8)
            for name, count in top:
                tag = ''
                if name in NAMED:
                    tag = f'  [{name}=NAMED daughter]'
                elif name in ('0', '2', '3'):
                    tag = f'  [trunk/arch]'
                else:
                    tag = f'  [sub-branch]'
                print(f'    {name:<10} {count:>3}  ({100*count/len(eps):>3.0f}%){tag}')
            if no_branch:
                print(f'    (no branch reached: {no_branch})')


if __name__ == '__main__':
    main()
