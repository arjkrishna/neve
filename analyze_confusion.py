"""Confusion matrix per run: assigned target branch vs actually-entered branch.

A wire "entered" a daughter X if the tip came within `THRESH` mm of BOTH the
first centerline point AND the second centerline point of branch X at some
point during the episode. The first two points are sufficient to disambiguate
RCCA/RVA which share the entry point but diverge at the second.

Output: per-run confusion matrix.
"""
import re
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
from analyze_branches import load_curve, CENTERLINE_DIR, vessel_to_tracking3d

NAMED = ['LCCA', 'LVA', 'RCCA', 'RVA']
LOG_BASE = Path('saved/eve_paper/neurovascular/full/mesh_ben')

# A wire is considered to have "entered" a branch if its tip passed through
# the branch's "entry zone" — the first ~30mm of centerline beyond the
# junction. We look at the FAR end of the entry zone (point at ~25mm in)
# because anything that close means the wire is committed, not just brushing.


def load_branch_entry_zone(name, far_mm=25.0):
    """Return the centerline point ~far_mm into the branch (in tracking3d)."""
    poly_vcs = load_curve(CENTERLINE_DIR / f'Centerline curve - {name}.mrk.json')
    poly_t3d = vessel_to_tracking3d(poly_vcs)
    cum = 0.0
    for i in range(1, len(poly_t3d)):
        cum += np.linalg.norm(poly_t3d[i] - poly_t3d[i - 1])
        if cum >= far_mm:
            return poly_t3d[i]
    return poly_t3d[-1]


def classify_entered(tip_traj, branch_polys, perp_threshold=8.0, min_penetration=10.0):
    """For each branch, find the deepest arclength point on the centerline
    that any tip3d came within `perp_threshold` mm of. Wire 'entered' the
    branch whose maximum penetration is greatest (and exceeds
    `min_penetration` mm). 'none' if no branch is penetrated past the
    threshold."""
    if len(tip_traj) == 0:
        return 'none'
    best_name, best_pen = 'none', min_penetration
    for name, (poly, cum) in branch_polys.items():
        # For each tip, find nearest centerline point
        dists = np.linalg.norm(tip_traj[:, None, :] - poly[None, :, :], axis=2)
        nearest_idx = dists.argmin(axis=1)
        # Filter to tips that came within perp_threshold of the centerline
        within = dists[np.arange(len(tip_traj)), nearest_idx] < perp_threshold
        if not within.any():
            continue
        deepest_arclength = cum[nearest_idx[within]].max()
        if deepest_arclength > best_pen:
            best_pen = deepest_arclength
            best_name = name
    return best_name


def classify_target(target_xyz, branch_polys):
    return min(branch_polys, key=lambda n: np.linalg.norm(branch_polys[n] - target_xyz, axis=1).min())


def parse_run(log_dir, branch_polys_arc, branch_polys):
    """Return list of (target_branch, entered_branch, succ)."""
    rows = []
    for f in sorted(glob.glob(str(log_dir / 'diagnostics/logs_subprocesses/worker_*.log'))):
        cur = None
        for line in open(f):
            if 'EPISODE_START' in line:
                if cur and 'tips' in cur and 'target' in cur:
                    if len(cur['tips']) > 0:
                        traj = np.array(cur['tips'])
                        rows.append((classify_target(cur['target'], branch_polys),
                                     classify_entered(traj, branch_polys_arc),
                                     cur.get('reward', 0) > 0))
                m = re.search(r'EPISODE_START \| ep=(\d+).*target=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
                if m:
                    cur = {'tips': [], 'target': np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])}
                continue
            if cur is None:
                continue
            m = re.search(r'STEP \|.*tip3d=\(([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
            if m:
                cur['tips'].append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                continue
            m = re.search(r'EPISODE_END \|.*total_reward=([-\d.]+)', line)
            if m:
                cur['reward'] = float(m.group(1))
        if cur and 'tips' in cur and 'target' in cur and len(cur['tips']) > 0:
            traj = np.array(cur['tips'])
            rows.append((classify_target(cur['target'], branch_polys),
                         classify_entered(traj, branch_polys_arc),
                         cur.get('reward', 0) > 0))
    return rows


def print_confusion(name, rows):
    cm = defaultdict(lambda: defaultdict(int))
    succ = defaultdict(int)
    n_total = len(rows)
    n_succ = sum(1 for _, _, s in rows if s)
    for tgt, ent, s in rows:
        cm[tgt][ent] += 1
        if s:
            succ[tgt] += 1
    cols = NAMED + ['none']
    print(f'\n=== {name} ({n_total} eps, {n_succ} successes) ===')
    print(f'{"target":<6}', '  '.join(f'{c:>5}' for c in cols), '  succ')
    for tgt in NAMED:
        if sum(cm[tgt][c] for c in cols) == 0:
            continue
        row = '  '.join(f'{cm[tgt][c]:>5}' for c in cols)
        print(f'{tgt:<6}', row, f'  {succ[tgt]:>4}')


def main():
    branch_polys = {}      # for target classification (full polyline)
    branch_polys_arc = {}  # for entry classification (with arclength)
    for n in NAMED:
        poly_vcs = load_curve(CENTERLINE_DIR / f'Centerline curve - {n}.mrk.json')
        poly_t3d = vessel_to_tracking3d(poly_vcs)
        branch_polys[n] = poly_t3d
        # Compute cumulative arclength along this branch
        diffs = np.linalg.norm(np.diff(poly_t3d, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(diffs)])
        branch_polys_arc[n] = (poly_t3d, cum)

    runs = sorted(LOG_BASE.glob('*env5_rl7_ckpttest*'))
    runs = [r for r in runs if r.is_dir() and not str(r).endswith('.csv')]
    for run_dir in runs:
        run_name = run_dir.name.split('env5_rl7_')[1]
        log_files = list((run_dir / 'diagnostics/logs_subprocesses').glob('worker_*.log'))
        if not log_files:
            continue
        # Quick check — does it have any episode ends?
        has_data = False
        for f in log_files[:3]:
            for line in open(f):
                if 'EPISODE_END' in line:
                    has_data = True
                    break
            if has_data:
                break
        if not has_data:
            continue
        rows = parse_run(run_dir, branch_polys_arc, branch_polys)
        if rows:
            print_confusion(run_name, rows)


if __name__ == '__main__':
    main()
