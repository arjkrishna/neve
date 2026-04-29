"""For the near-RVA-success episodes, trace tip3d through the centerline.

User asked: did pid=312 ep=3 and pid=654 ep=3 actually pass through the
RVA entry point AND the next point along RVA? They reached tip_max_z=572
(RVA endpoint) and have entries_gained=2.

This script:
1. Loads the first 5 points of every daughter centerline (in vessel-CS).
2. For the two near-success RVA episodes, transforms every logged STEP
   tip3d to vessel-CS and finds the closest tip-step to each daughter
   centerline point.
3. Reports the trajectory.
"""
import os, re, json, glob
import numpy as np

LOG_DIR = (
    "saved/eve_paper/neurovascular/full/mesh_ben/"
    "2026-04-29_013802_env5_rl8_highinsert_50ep/diagnostics/logs_subprocesses"
)
DATA_DIR = "eve_bench/data/dualdevicenav/Centrelines_comb"


def get_rot_matrix(rzx):
    rz = -rzx[0] * np.pi / 180
    rx = -rzx[1] * np.pi / 180
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    return Rz @ Rx


_ROT = get_rot_matrix((20, 5))


# Load full daughter centerlines (vessel-CS)
DAUGHTERS = {}
for d in ["LCCA", "LVA", "RCCA", "RVA"]:
    with open(os.path.join(DATA_DIR, f"Centerline curve - {d}.mrk.json")) as f:
        data = json.load(f)
    pts = []
    for m in data["markups"]:
        if m["type"] == "Curve":
            for cp in m["controlPoints"]:
                x, y, z = cp["position"]
                pts.append((y, -z, -x))
    DAUGHTERS[d] = np.array(pts)


print("=== First 5 control points of each daughter centerline (vessel-CS) ===")
for d, pts in DAUGHTERS.items():
    print(f"\n  {d} ({len(pts)} pts total):")
    for i, p in enumerate(pts[:5]):
        seg_len = float(np.linalg.norm(pts[i+1] - pts[i])) if i + 1 < len(pts) else 0.0
        print(f"    [{i}] vcs=({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f})  next-segment={seg_len:.2f} mm")


print("\n=== Distance between daughter pairs at their first point ===")
print("  LCCA[0] vs LVA[0]:  %.2f mm" % float(np.linalg.norm(DAUGHTERS['LCCA'][0] - DAUGHTERS['LVA'][0])))
print("  LCCA[0] vs RCCA[0]: %.2f mm" % float(np.linalg.norm(DAUGHTERS['LCCA'][0] - DAUGHTERS['RCCA'][0])))
print("  LCCA[0] vs RVA[0]:  %.2f mm" % float(np.linalg.norm(DAUGHTERS['LCCA'][0] - DAUGHTERS['RVA'][0])))
print("  LVA[0]  vs RCCA[0]: %.2f mm" % float(np.linalg.norm(DAUGHTERS['LVA'][0] - DAUGHTERS['RCCA'][0])))
print("  LVA[0]  vs RVA[0]:  %.2f mm" % float(np.linalg.norm(DAUGHTERS['LVA'][0] - DAUGHTERS['RVA'][0])))
print("  RCCA[0] vs RVA[0]:  %.2f mm" % float(np.linalg.norm(DAUGHTERS['RCCA'][0] - DAUGHTERS['RVA'][0])))


# Find RVA's near-success episodes — pid=312 ep=3 and pid=654 ep=3.
# Read every STEP line, extract ep_step, tip3d.
RE_EP_START = re.compile(r"EPISODE_START \| ep=(\d+) \| .* pid=(\d+)")
RE_STEP_TIP = re.compile(r"STEP \| ep=(\d+) \| ep_step=(\d+) \|.*?tip3d=\(([\d.\-]+),([\d.\-]+),([\d.\-]+)\)")
RE_STEP_DCORR = re.compile(r"STEP \| ep=(\d+) \| ep_step=(\d+) \|.*?d_corr_arc=([\d.infa]+).*?arc_past=([\d.\-]+).*?entries_passed=(\d+)")


def extract_episode(target_pid, target_ep):
    log = os.path.join(LOG_DIR, f"worker_{target_pid}.log")
    if not os.path.exists(log):
        return None
    cur_ep = None
    steps = []  # list of (ep_step, tip3d, d_corr_arc, arc_past, entries)
    pending = {}  # ep_step -> partial dict
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_EP_START.search(line)
            if m:
                cur_ep = int(m.group(1))
                continue
            if cur_ep != target_ep:
                continue
            m = RE_STEP_TIP.search(line)
            if m:
                ep_step = int(m.group(2))
                tip3d = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
                pending.setdefault(ep_step, {})["tip3d"] = tip3d
            m = RE_STEP_DCORR.search(line)
            if m:
                ep_step = int(m.group(2))
                pending.setdefault(ep_step, {})
                try:
                    pending[ep_step]["d_corr"] = float(m.group(3)) if m.group(3) != "inf" else float("inf")
                except ValueError:
                    pending[ep_step]["d_corr"] = float("inf")
                pending[ep_step]["arc_past"] = float(m.group(4))
                pending[ep_step]["entries"] = int(m.group(5))
    for ep_step, d in sorted(pending.items()):
        if "tip3d" in d:
            steps.append((ep_step, d["tip3d"], d.get("d_corr"), d.get("arc_past"), d.get("entries")))
    return steps


for pid, ep in [(312, 3), (654, 3)]:
    print(f"\n=== Episode pid={pid} ep={ep} (RVA target, near-success) ===")
    steps = extract_episode(pid, ep)
    if not steps:
        print(f"  (no log found for pid {pid})")
        continue
    print(f"  {len(steps)} INFO STEP lines logged for this episode")
    print(f"  ep_step range: {steps[0][0]} -> {steps[-1][0]}")

    # For each step: convert tip3d to vcs, find min-distance to RVA[0..4] points
    rva = DAUGHTERS["RVA"]
    rcca = DAUGHTERS["RCCA"]
    print(f"\n  Trajectory (tip3d -> vessel-CS, dist to first 5 RVA pts, dist to first 5 RCCA pts):")
    print(f"  {'step':>5} {'tip3d':>22} {'tip_vcs':>22} {'eg':>3} {'arc_past':>9} {'d_to_RVA[0..4]':>30} {'d_to_RCCA[0..4]':>30}")
    for i, (s, t3d, dc, ap, eg) in enumerate(steps):
        tip_vcs = _ROT.T @ np.asarray(t3d)
        d_rva = [float(np.linalg.norm(tip_vcs - rva[j])) for j in range(min(5, len(rva)))]
        d_rcca = [float(np.linalg.norm(tip_vcs - rcca[j])) for j in range(min(5, len(rcca)))]
        rva_str = " ".join("%5.1f" % d for d in d_rva)
        rcca_str = " ".join("%5.1f" % d for d in d_rcca)
        # Filter: only print steps near RVA junction OR every Nth step
        if min(d_rva) < 30 or i < 5 or i % max(1, len(steps) // 15) == 0 or i == len(steps) - 1:
            print(f"  {s:>5} ({t3d[0]:5.1f},{t3d[1]:5.1f},{t3d[2]:5.1f})  ({tip_vcs[0]:5.1f},{tip_vcs[1]:5.1f},{tip_vcs[2]:5.1f})  {eg:>3} {ap:>9.1f}  {rva_str}  {rcca_str}")

    # Min distance achieved overall to RVA[0] (entry) and RVA[1] (next-point)
    print(f"\n  Summary: min Euclidean distance achieved over the whole trajectory:")
    for j in range(min(5, len(rva))):
        all_d = [float(np.linalg.norm(_ROT.T @ np.asarray(s[1]) - rva[j])) for s in steps]
        print(f"    RVA[{j}] vcs=({rva[j][0]:.2f},{rva[j][1]:.2f},{rva[j][2]:.2f}): min={min(all_d):.2f} mm")
    for j in range(min(3, len(rcca))):
        all_d = [float(np.linalg.norm(_ROT.T @ np.asarray(s[1]) - rcca[j])) for s in steps]
        print(f"    RCCA[{j}] vcs=({rcca[j][0]:.2f},{rcca[j][1]:.2f},{rcca[j][2]:.2f}): min={min(all_d):.2f} mm")
