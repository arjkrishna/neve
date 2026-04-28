"""Analyze DualDeviceNav branch geometry and map run 17 episodes to branches.

Loads the 4 named target centerlines (LCCA, LVA, RCCA, RVA) plus the numbered
trunk centerlines, then:
  1. Picks a "main trunk" centerline by length
  2. For each target branch, finds where it diverges from the trunk — this
     gives the arclength along the trunk at which that branch takes off
  3. Reads run 17 EPISODE_START/END lines and assigns each episode's target
     to the nearest of the 4 branches
  4. Reports success/fold/timeout rate per branch plus reachability vs 372mm

Run with: python analyze_branches.py
"""
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CENTERLINE_DIR = Path("eve_bench/data/dualdevicenav/Centrelines_comb")
RUN17_DIR = Path(
    "saved/eve_paper/neurovascular/full/mesh_ben/2026-04-20_191712_env5_rl7_ckpttest17"
)
RESTORE_INSERTED = 372.0  # approx wire insertion length at restore (mm)

# DualDeviceNav uses image_rot_zx=[20, 5], image_center=[0,0,0]
IMAGE_ROT_ZX = (20.0, 5.0)


def _get_rot_matrix(image_rot_zx):
    # Same sign convention as eve/util/coordtransform.py
    rz = -image_rot_zx[0] * np.pi / 180
    rx = -image_rot_zx[1] * np.pi / 180
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    return Rz @ Rx


ROT = _get_rot_matrix(IMAGE_ROT_ZX)


def vessel_to_tracking3d(pts: np.ndarray) -> np.ndarray:
    """Apply DualDeviceNav vessel_cs -> tracking3d transform (image_center=0)."""
    return (ROT @ pts.T).T


def tracking3d_to_vessel(pts: np.ndarray) -> np.ndarray:
    return (ROT.T @ pts.T).T


def load_curve(path: Path) -> np.ndarray:
    """Load a Slicer centerline and apply eve_bench's (x,y,z)->(y,-z,-x) transform
    so the coordinates match what the sim / logs use."""
    with open(path) as f:
        d = json.load(f)
    pts = []
    for cp in d["markups"][0]["controlPoints"]:
        x, y, z = cp["position"]
        pts.append((y, -z, -x))
    return np.array(pts, dtype=float)


def arclength(pts: np.ndarray) -> np.ndarray:
    """Return cumulative arclength per point, starting at 0."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_point_to_polyline(p: np.ndarray, poly: np.ndarray, cumlen: np.ndarray):
    """Return (nearest arclength s, nearest index, min distance)."""
    best_d, best_s, best_i = float("inf"), 0.0, 0
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        denom = np.dot(ab, ab)
        if denom < 1e-12:
            continue
        t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
        q = a + t * ab
        d = np.linalg.norm(p - q)
        if d < best_d:
            best_d = d
            best_s = cumlen[i] + t * np.sqrt(denom)
            best_i = i
    return best_s, best_i, best_d


def find_takeoff(branch: np.ndarray, trunk: np.ndarray, trunk_cumlen: np.ndarray):
    """Find the point where a branch diverges from the trunk.

    We treat it as: the arclength along the branch at which the branch-point
    distance to the trunk first exceeds a threshold (2 mm).

    Returns the arclength along the TRUNK where the branch takes off.
    """
    prev_d = 0.0
    for i, p in enumerate(branch):
        s, _, d = project_point_to_polyline(p, trunk, trunk_cumlen)
        if i == 0:
            # The branch start itself should be ON the trunk
            entry_s = s
        if d > 2.0 and prev_d <= 2.0 and i > 0:
            # Branch just left the trunk
            return entry_s, d, i
        prev_d = d
    return entry_s, prev_d, len(branch)


def main():
    # ---- Load centerlines ----
    files = sorted(CENTERLINE_DIR.glob("*.json"))
    curves = {}
    for f in files:
        name = f.stem.replace(".mrk", "")
        curves[name] = load_curve(f)
    print(f"Loaded {len(curves)} centerlines")

    # ---- Identify trunk: the longest of the unnamed (numbered) curves usually
    # represents the aorta/trunk. Named curves are branches ----
    branch_names = [
        "Centerline curve - LCCA",
        "Centerline curve - LVA",
        "Centerline curve - RCCA",
        "Centerline curve - RVA",
    ]

    trunk_candidates = [
        (n, pts) for n, pts in curves.items() if "-" not in n
    ]
    # Pick the longest unnumbered / numbered curve as trunk
    trunk_candidates.sort(key=lambda t: arclength(t[1])[-1], reverse=True)
    print("\nTop 5 curves by arclength:")
    for n, pts in trunk_candidates[:5]:
        print(f"  {n}: n_pts={len(pts)}, len={arclength(pts)[-1]:.1f} mm,"
              f" z=[{pts[:, 2].min():.1f}..{pts[:, 2].max():.1f}]")

    # We need a trunk that includes the insertion point (near origin?) all the
    # way up to the supra-aortic branch region. Look for the longest one whose
    # bounding z-range is widest.
    trunk_name, trunk = trunk_candidates[0]
    trunk_cumlen = arclength(trunk)
    print(f"\nUsing trunk: {trunk_name}")
    print(f"  total length: {trunk_cumlen[-1]:.1f} mm")
    print(f"  z range: {trunk[:, 2].min():.1f} .. {trunk[:, 2].max():.1f}")

    # ---- Orient the trunk so s=0 is the insertion point (femoral) and
    # s=trunk_len is the aortic arch. The raw JSON could be stored either way. ----
    # Heuristic: the end of the trunk CLOSER to the named branches is the arch.
    branch_start_xyzs = []
    for b in branch_names:
        if b in curves:
            branch_start_xyzs.append(curves[b][0])
            branch_start_xyzs.append(curves[b][-1])
    if branch_start_xyzs:
        branch_cloud = np.array(branch_start_xyzs)
        d0 = np.linalg.norm(branch_cloud - trunk[0], axis=1).min()
        dN = np.linalg.norm(branch_cloud - trunk[-1], axis=1).min()
        if dN > d0:
            # trunk[0] is the arch end — flip so s=0 is insertion (femoral) instead.
            trunk = trunk[::-1]
            trunk_cumlen = arclength(trunk)
            print("\nFlipped trunk so s=0 is insertion point (femoral)")
    insertion_pt = trunk[0]
    print(f"\nInsertion point (trunk[0]): ({insertion_pt[0]:.1f}, {insertion_pt[1]:.1f}, {insertion_pt[2]:.1f})")
    print(f"Trunk end (trunk[-1] = arch): ({trunk[-1, 0]:.1f}, {trunk[-1, 1]:.1f}, {trunk[-1, 2]:.1f})")

    # ---- Branch take-off arclengths ----
    print("\n=== Branch take-off positions (along trunk) ===")
    takeoff_s = {}
    for b in branch_names:
        if b not in curves:
            print(f"  MISSING: {b}")
            continue
        br = curves[b]
        # The branch start point should be near the trunk; we just project
        # the branch's start onto the trunk to get its take-off arclength.
        s_start, _, d_start = project_point_to_polyline(br[0], trunk, trunk_cumlen)
        s_end, _, d_end = project_point_to_polyline(br[-1], trunk, trunk_cumlen)
        # Decide take-off end: whichever end is CLOSER to the trunk
        if d_start < d_end:
            s_takeoff = s_start
            tip_xyz = br[-1]
        else:
            s_takeoff = s_end
            tip_xyz = br[0]
        branch_len = arclength(br)[-1]
        # Also: distance from branch start (the one CLOSER to the trunk) to the
        # nearest trunk point — if this is large, there's a bridging curve between
        # trunk and branch.
        branch_start = br[0] if d_start < d_end else br[-1]
        bridge_gap = min(d_start, d_end)
        # Rough "path distance from insertion to branch start"
        # = trunk arclength to takeoff + bridge gap
        trunk_to_takeoff = s_takeoff
        print(f"  {b}:")
        print(f"     takeoff_s along trunk: {s_takeoff:.1f} mm")
        print(f"     bridge gap trunk->branch: {bridge_gap:.1f} mm")
        print(f"     branch start xyz: ({branch_start[0]:.1f}, {branch_start[1]:.1f}, {branch_start[2]:.1f})")
        print(f"     branch length:    {branch_len:.1f} mm")
        print(f"     branch tip xyz:   ({tip_xyz[0]:.1f}, {tip_xyz[1]:.1f}, {tip_xyz[2]:.1f})")
        # Approximate reachability = trunk_to_takeoff + bridge_gap
        path_to_branch_start = trunk_to_takeoff + bridge_gap
        print(f"     approx arclength from insertion to branch start: {path_to_branch_start:.1f} mm")
        takeoff_s[b] = (path_to_branch_start, tip_xyz, branch_start)

    # ---- Relative to 372mm restore ----
    print(f"\nRestore insertion: {RESTORE_INSERTED} mm")
    print("\n=== Reachability ===")
    for b, (s, tip, start) in takeoff_s.items():
        delta = s - RESTORE_INSERTED
        reach = "AFTER restore (forward-only reachable)" if delta > 0 else "BEFORE restore (needs retraction)"
        print(f"  {b}: branch start at ~{s:.1f} mm insertion, {delta:+.1f} mm from restore  [{reach}]")

    # ---- Map run 17 episodes to branches ----
    print("\n=== Run 17 episodes by branch ===")
    episodes = []  # (target_xyz, steps, reward, abort)
    log_files = sorted((RUN17_DIR / "diagnostics/logs_subprocesses").glob("worker_*.log"))
    for lf in log_files:
        ep_targets = {}  # episode -> target
        with open(lf) as f:
            for line in f:
                m = re.search(r"EPISODE_START \| ep=(\d+).*target=\(([-\d.]+),([-\d.]+),([-\d.]+)\)", line)
                if m:
                    ep, x, y, z = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
                    ep_targets[ep] = np.array([x, y, z])
                    continue
                m = re.search(r"EPISODE_END \| ep=(\d+) \| steps=(\d+) \| total_reward=([-\d.]+).*heur_abort=(\S+)", line)
                if m:
                    ep = int(m.group(1))
                    steps = int(m.group(2))
                    reward = float(m.group(3))
                    abort = m.group(4)
                    if ep in ep_targets:
                        episodes.append((ep_targets[ep], steps, reward, abort))

    print(f"Parsed {len(episodes)} episodes\n")

    # Classify targets by nearest point on the "DISTAL" portion of each branch
    # polyline. We exclude the first 20 mm of each branch (its start, which
    # overlaps with other branches and the trunk) so that the nearest-point
    # query actually distinguishes them. Convert branch vessel_cs -> tracking3d
    # so they match the logged target coordinate frame.
    def distal_polyline_tracking3d(poly):
        cl = arclength(poly)
        mask = cl > 20.0
        p = poly[mask] if mask.any() else poly
        return vessel_to_tracking3d(p)
    branch_polylines = {b: distal_polyline_tracking3d(curves[b]) for b in branch_names if b in curves}
    print("\nBranch tip in tracking3d frame:")
    for b, poly in branch_polylines.items():
        tip = poly[-1]
        print(f"  {b}: distal polyline tip = ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f})")
    stats = defaultdict(lambda: {"n": 0, "success": 0, "fold": 0, "timeout": 0, "restore_fail": 0})
    unclassified_z = []
    for target, steps, reward, abort in episodes:
        best_b, best_d = None, float("inf")
        for b, poly in branch_polylines.items():
            dists = np.linalg.norm(poly - target, axis=1)
            d = dists.min()
            if d < best_d:
                best_d = d
                best_b = b
        st = stats[best_b]
        st["n"] += 1
        if reward > 0:
            st["success"] += 1
        elif abort == "wire_fold_stall":
            st["fold"] += 1
        elif steps <= 3:
            st["restore_fail"] += 1
        elif steps >= 600:
            st["timeout"] += 1

    print(f"{'Branch':<35} {'n':>4} {'succ':>5} {'fold':>5} {'to':>5} {'restore':>8}")
    for b in branch_names:
        s = stats[b]
        pct = f"{100*s['success']/s['n']:.0f}%" if s["n"] else "-"
        print(f"{b:<35} {s['n']:>4} {s['success']:>5} {s['fold']:>5} {s['timeout']:>5} {s['restore_fail']:>8}  ({pct} success)")


if __name__ == "__main__":
    main()
