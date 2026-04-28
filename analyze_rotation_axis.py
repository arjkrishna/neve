"""Deep analysis of wire geometry at bif2 to determine correct rotation axis.

Investigates:
  1. Aorta/arch 3D shape — where does the wire's travel direction point at 372mm?
  2. Direction from restore point (end of trunk) to each branch tip — which axis
     dominates? Does any branch require motion in a direction "behind" the wire
     (would need retraction)?
  3. For run 17 successes vs fold-stalls: what was the wire's device_dir
     orientation at bif2? Which cross-product component of (device_dir x
     path_tangent) best correlates with successful rotation direction?
  4. Log rotation effectiveness: when gw_rot was applied, did the tip actually
     move toward the correct branch in subsequent steps?
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CENTERLINE_DIR = Path("eve_bench/data/dualdevicenav/Centrelines_comb")
RUN17_DIR = Path(
    "saved/eve_paper/neurovascular/full/mesh_ben/2026-04-20_191712_env5_rl7_ckpttest17"
)
RESTORE_INSERTED = 372.0
IMAGE_ROT_ZX = (20.0, 5.0)


def _rot_matrix(rx_zx):
    rz = -rx_zx[0] * np.pi / 180
    rx = -rx_zx[1] * np.pi / 180
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    return Rz @ Rx


ROT = _rot_matrix(IMAGE_ROT_ZX)


def vessel_to_tracking3d(pts):
    return (ROT @ pts.T).T


def load_curve(path):
    with open(path) as f:
        d = json.load(f)
    pts = []
    for cp in d["markups"][0]["controlPoints"]:
        x, y, z = cp["position"]
        pts.append((y, -z, -x))
    return np.array(pts, dtype=float)


def arclength(pts):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


# ---- 1. Aortic arch shape ----
print("=" * 70)
print("SECTION 1: Aortic trunk 3D shape (vessel_cs -> tracking3d)")
print("=" * 70)
trunk_file = CENTERLINE_DIR / "Centerline curve (2).mrk.json"
trunk_vcs = load_curve(trunk_file)
# Flip so s=0 = insertion (femoral, low z in vessel_cs)
if trunk_vcs[0, 2] > trunk_vcs[-1, 2]:
    trunk_vcs = trunk_vcs[::-1]
trunk = vessel_to_tracking3d(trunk_vcs)
cl = arclength(trunk)
print(f"trunk length: {cl[-1]:.1f} mm, {len(trunk)} points")
print("Sampled points along trunk (tracking3d frame):")
for s in (0, 50, 100, 150, 200, 250, 300, 340, 370, cl[-1]):
    idx = np.argmin(np.abs(cl - s))
    p = trunk[idx]
    # Compute local tangent
    if idx < len(trunk) - 1:
        tan = trunk[idx + 1] - trunk[idx]
        tan /= np.linalg.norm(tan) + 1e-12
    else:
        tan = trunk[idx] - trunk[idx - 1]
        tan /= np.linalg.norm(tan) + 1e-12
    print(f"  s={cl[idx]:6.1f}  pt=({p[0]:+6.1f},{p[1]:+6.1f},{p[2]:+6.1f})  "
          f"tangent=({tan[0]:+.2f},{tan[1]:+.2f},{tan[2]:+.2f})")

# ---- 2. Restore-point direction & branch direction vectors ----
print()
print("=" * 70)
print("SECTION 2: Wire direction at 372mm vs vectors to each branch tip")
print("=" * 70)
# Restore point ≈ trunk position at arclength 372 mm (trunk is ~373 mm)
idx_restore = np.argmin(np.abs(cl - RESTORE_INSERTED))
restore_pt = trunk[idx_restore]
# Wire direction at restore = local tangent (approximate)
if idx_restore < len(trunk) - 1:
    wire_dir = trunk[idx_restore + 1] - trunk[idx_restore]
else:
    wire_dir = trunk[idx_restore] - trunk[idx_restore - 1]
wire_dir /= np.linalg.norm(wire_dir) + 1e-12
print(f"Restore point: ({restore_pt[0]:+6.1f},{restore_pt[1]:+6.1f},{restore_pt[2]:+6.1f})")
print(f"Wire tangent at restore: ({wire_dir[0]:+.2f},{wire_dir[1]:+.2f},{wire_dir[2]:+.2f})")
print()

branch_names = [
    "Centerline curve - LCCA",
    "Centerline curve - LVA",
    "Centerline curve - RCCA",
    "Centerline curve - RVA",
]
branches = {b: vessel_to_tracking3d(load_curve(CENTERLINE_DIR / (b + ".mrk.json")))
            for b in branch_names}
print("Per-branch: vector from restore point to branch mid-point")
print("(positive dot with wire_dir = FORWARD reachable; negative = BEHIND/needs retract)")
print()
for b, poly in branches.items():
    b_cl = arclength(poly)
    # Mid-branch point (reasonable target region)
    mid = poly[len(poly) // 2]
    tip = poly[-1] if np.linalg.norm(poly[-1] - restore_pt) > np.linalg.norm(poly[0] - restore_pt) else poly[0]
    v = mid - restore_pt
    d = np.linalg.norm(v)
    v_hat = v / (d + 1e-12)
    dot = np.dot(v_hat, wire_dir)
    print(f"  {b[-5:]}:")
    print(f"    mid-branch pt: ({mid[0]:+6.1f},{mid[1]:+6.1f},{mid[2]:+6.1f})")
    print(f"    vec from restore: ({v[0]:+6.1f},{v[1]:+6.1f},{v[2]:+6.1f}) |v|={d:.1f}")
    print(f"    dot(vec_hat, wire_dir)={dot:+.2f}  "
          f"({'FORWARD' if dot > 0.2 else 'BEHIND' if dot < -0.2 else 'SIDEWAYS'})")


# ---- 3. Wire device_dir at bif2 in success vs failure ----
print()
print("=" * 70)
print("SECTION 3: Wire direction at bif2 (run 17 episodes)")
print("=" * 70)
# Parse run 17 logs: at d_corr around 40 mm (approach zone), record tip3d
# and the two successive tip3d points to compute device_dir
# We only have tip3d (single point) per step, so device_dir ≈
# (tip3d[t] - tip3d[t-1]) / |...|. Good enough.
step_re = re.compile(
    r"STEP \| ep=(\d+) \| ep_step=(\d+).*"
    r"cmd_action=\[([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\].*"
    r"on_br=(\d+).*fold=(\d+)/\d+.*"
    r"d_corr=([-\d.]+).*"
    r"tip3d=\(([-\d.]+),([-\d.]+),([-\d.]+)\).*"
    r"rot_inst=\[([-\d.]+),([-\d.]+)\]"
)
ep_start_re = re.compile(
    r"EPISODE_START \| ep=(\d+).*target=\(([-\d.]+),([-\d.]+),([-\d.]+)\)"
)
ep_end_re = re.compile(
    r"EPISODE_END \| ep=(\d+) \| steps=(\d+) \| total_reward=([-\d.]+).*heur_abort=(\S+)"
)

# Build per-episode step sequences
episodes = []  # list of dicts
for lf in sorted((RUN17_DIR / "diagnostics/logs_subprocesses").glob("worker_*.log")):
    pid = lf.stem.split("_")[1]
    cur = None
    with open(lf) as f:
        for line in f:
            m = ep_start_re.search(line)
            if m:
                if cur is not None:
                    episodes.append(cur)
                cur = {
                    "pid": pid,
                    "ep": int(m.group(1)),
                    "target": (float(m.group(2)), float(m.group(3)), float(m.group(4))),
                    "steps": [],
                    "result": None,
                }
                continue
            m = step_re.search(line)
            if m and cur is not None:
                cur["steps"].append({
                    "ep_step": int(m.group(2)),
                    "trans_gw": float(m.group(3)),
                    "rot_gw": float(m.group(4)),
                    "on_br": int(m.group(7)),
                    "fold": int(m.group(8)),
                    "d_corr": float(m.group(9)),
                    "tip3d": (float(m.group(10)), float(m.group(11)), float(m.group(12))),
                    "rot_inst": float(m.group(13)),
                })
                continue
            m = ep_end_re.search(line)
            if m and cur is not None:
                cur["result"] = {
                    "steps": int(m.group(2)),
                    "reward": float(m.group(3)),
                    "abort": m.group(4),
                }
    if cur is not None:
        episodes.append(cur)

# Classify episodes by outcome
good = [e for e in episodes if e["result"] and e["result"]["reward"] > 0]
fold = [e for e in episodes if e["result"] and e["result"]["abort"] == "wire_fold_stall"]
print(f"parsed {len(episodes)} episodes  ({len(good)} success, {len(fold)} fold-stall)\n")


def analyze_approach(ep, window=(20, 60)):
    """Look at wire direction when d_corr is in [window[0], window[1]] mm."""
    relevant = [s for s in ep["steps"] if window[0] <= s["d_corr"] <= window[1]
                and s["on_br"] == 1]
    if len(relevant) < 3:
        return None
    # Compute wire_dir from tip3d differences (skip first)
    dirs = []
    for i in range(1, len(relevant)):
        p1 = np.array(relevant[i - 1]["tip3d"])
        p2 = np.array(relevant[i]["tip3d"])
        d = p2 - p1
        n = np.linalg.norm(d)
        if n > 0.3:  # real motion
            dirs.append(d / n)
    if not dirs:
        return None
    mean_dir = np.mean(dirs, axis=0)
    mean_dir /= np.linalg.norm(mean_dir) + 1e-12
    return mean_dir, len(relevant)


# Wire direction statistics
for label, pool in [("SUCCESS", good), ("FOLD_STALL", fold[:30])]:
    print(f"--- {label}: mean wire direction at d_corr in [20,60] mm ---")
    all_dirs = []
    for ep in pool:
        r = analyze_approach(ep)
        if r is None:
            continue
        d, n = r
        all_dirs.append(d)
    if all_dirs:
        arr = np.array(all_dirs)
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        print(f"  n={len(arr)} episodes contributed direction")
        print(f"  mean wire_dir: ({m[0]:+.2f}, {m[1]:+.2f}, {m[2]:+.2f})")
        print(f"  std:           ({s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f})")

# ---- 4. Does rotation help steer? Correlate gw_rot with subsequent d_corr change ----
print()
print("=" * 70)
print("SECTION 4: Does gw_rot correlate with d_corr improvement?")
print("=" * 70)

def analyze_rot_effectiveness(pool, label):
    # For each step where rot was commanded, look at d_corr change over next 5 steps
    pos_rot_events = []  # [(rot, delta_d_corr)]
    neg_rot_events = []
    translate_events = []
    for ep in pool:
        steps = ep["steps"]
        for i in range(len(steps) - 5):
            s = steps[i]
            future = steps[i + 5] if i + 5 < len(steps) else steps[-1]
            if s["on_br"] != 1 or future["on_br"] != 1:
                continue
            dd = future["d_corr"] - s["d_corr"]  # negative = improvement
            if s["rot_gw"] > 0.1:
                pos_rot_events.append((s["rot_gw"], dd))
            elif s["rot_gw"] < -0.1:
                neg_rot_events.append((s["rot_gw"], dd))
            elif abs(s["rot_gw"]) < 0.05 and s["trans_gw"] > 2:
                translate_events.append((s["trans_gw"], dd))
    print(f"  {label}:")
    for name, evs in [("pos_rot (>+0.1)", pos_rot_events),
                      ("neg_rot (<-0.1)", neg_rot_events),
                      ("trans_only", translate_events)]:
        if evs:
            dds = np.array([e[1] for e in evs])
            print(f"    {name}: n={len(evs):4d}  mean_delta_dcorr={dds.mean():+.2f} mm  "
                  f"(improved in {(dds < -0.5).sum():3d} cases, worsened in {(dds > 0.5).sum():3d})")

analyze_rot_effectiveness(good, "SUCCESS episodes")
analyze_rot_effectiveness(fold, "FOLD_STALL episodes")
