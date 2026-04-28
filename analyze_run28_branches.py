"""Where do RVA / RCCA episodes go? Run-28 diagnostic.

Maps each episode's target (in tracking3d frame) → named branch by
transforming to vessel-CS and matching against centerlines from env_train.yml.

Then groups episodes by TRUE target branch and reports wedge patterns.
"""

import re
import os
import glob
import numpy as np
import yaml
from collections import Counter, defaultdict


# ---- Coord transform helpers (from eve.util.coordtransform) ----
IMAGE_ROT_ZX = (20.0, 5.0)
IMAGE_CENTER = (0.0, 0.0, 0.0)


def _get_rot_matrix(image_rot_zx):
    rot_z = -image_rot_zx[0] * np.pi / 180
    rot_x = -image_rot_zx[1] * np.pi / 180
    Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0],
                   [np.sin(rot_z), np.cos(rot_z), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rot_x), -np.sin(rot_x)],
                   [0, np.sin(rot_x), np.cos(rot_x)]])
    return Rz @ Rx


_ROT = _get_rot_matrix(IMAGE_ROT_ZX)
_IC = np.array(IMAGE_CENTER)


def t3d_to_vcs(p):
    return _ROT.T @ (np.asarray(p, dtype=float) + _ROT @ _IC)


# ---- Load named centerlines ----
class _AnyTagLoader(yaml.SafeLoader):
    pass


def _ignore(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


yaml.add_multi_constructor("tag:yaml.org,2002:python/", _ignore, Loader=_AnyTagLoader)
yaml.add_multi_constructor("!", _ignore, Loader=_AnyTagLoader)


def _walk(node):
    if isinstance(node, dict):
        if "name" in node and "coordinates" in node:
            yield node["name"], node["coordinates"]
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for it in node:
            yield from _walk(it)


with open(
    "saved/eve_paper/neurovascular/full/mesh_ben/"
    "2026-04-27_173014_env5_rl7_ckpttest28/env_train.yml",
    "r",
) as f:
    _CFG = yaml.load(f, Loader=_AnyTagLoader)

NAMED_CL = {}
for name, coords in _walk(_CFG):
    if name and name.startswith("Centerline curve - "):
        short = name.split(" - ")[1].split(".")[0]
        NAMED_CL[short] = np.array([[float(c[0]), float(c[1]), float(c[2])] for c in coords])


def classify_target_t3d(target):
    tv = t3d_to_vcs(target)
    best, best_d = None, 1e18
    for n, pts in NAMED_CL.items():
        d2 = np.min(np.sum((pts - tv) ** 2, axis=1))
        if d2 < best_d:
            best_d, best = d2, n
    return best

LOG_DIR = (
    "saved/eve_paper/neurovascular/full/mesh_ben/"
    "2026-04-27_173014_env5_rl7_ckpttest28/diagnostics/logs_subprocesses"
)

RE_EP_START = re.compile(
    r"EPISODE_START \| ep=(\d+) \| .* pid=(\d+) \| target=\(([\d.\-]+),([\d.\-]+),([\d.\-]+)\)"
)
RE_EP_END = re.compile(
    r"EPISODE_END \| ep=(\d+) \| steps=(\d+) \| total_reward=([\-\d.]+).*?heur_abort=(\w+)"
)
RE_STEP = re.compile(
    r"STEP \| ep=(\d+) \| ep_step=(\d+) \|.*?"
    r"reward=([\-\d.]+) \| cum_reward=([\-\d.]+) \|.*?"
    r"on_br=(\d) \| off_br=(\d) \| fold=(\d+)/\d+ \| "
    r"d_corr_arc=([\d.infa]+) \| arc_past=([\d.\-]+) \| nearest_named=(\w+) \| "
    r"entries_passed=(\d+) \| tip3d=\(([\d.\-]+),([\d.\-]+),([\d.\-]+)\)"
)


def parse_worker(path):
    pid = None
    cur_ep = None
    cur_target = None
    cur_steps = []
    cur_end = None  # (true_steps, total_reward, heur_abort)
    out = []

    def flush():
        nonlocal cur_ep, cur_target, cur_steps, cur_end
        if cur_ep is not None and cur_steps:
            out.append({
                "pid": pid, "ep": cur_ep, "target": cur_target,
                "steps": cur_steps, "end": cur_end,
            })
        cur_ep, cur_target, cur_steps, cur_end = None, None, [], None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_EP_START.search(line)
            if m:
                flush()
                ep, p, x, y, z = m.groups()
                cur_ep = int(ep)
                pid = int(p)
                cur_target = (float(x), float(y), float(z))
                continue
            m = RE_EP_END.search(line)
            if m and cur_ep is not None and int(m.group(1)) == cur_ep:
                cur_end = (int(m.group(2)), float(m.group(3)), m.group(4))
                continue
            m = RE_STEP.search(line)
            if m and cur_ep is not None and int(m.group(1)) == cur_ep:
                cur_steps.append({
                    "ep_step": int(m.group(2)),
                    "cum": float(m.group(4)),
                    "on_br": int(m.group(5)),
                    "off_br": int(m.group(6)),
                    "fold": int(m.group(7)),
                    "arc_past": float(m.group(9)),
                    "nearest": m.group(10),
                    "entries": int(m.group(11)),
                    "tip3d": (float(m.group(12)), float(m.group(13)), float(m.group(14))),
                })
    flush()
    return out


def main():
    all_eps = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "worker_*.log"))):
        all_eps.extend(parse_worker(path))
    print(f"parsed {len(all_eps)} episodes")

    # CLASSIFY each episode by: target z-band + the "dominant nearest_named"
    # across last 100 steps. We'll then inspect what these labels reveal.
    rows = []
    for e in all_eps:
        steps = e["steps"]
        last = steps[-1]
        n = len(steps)
        last100 = steps[-min(100, n):]
        named_ct = Counter(s["nearest"] for s in last100)
        # exclude 'none' for branch inference
        named_only = Counter({k: v for k, v in named_ct.items() if k in {"LCCA", "LVA", "RCCA", "RVA"}})
        dominant = named_only.most_common(1)[0][0] if named_only else "none"
        max_z = max(s["tip3d"][2] for s in steps)
        max_entries = max(s["entries"] for s in steps)
        max_arc_past = max(s["arc_past"] for s in steps)
        off_br_pct = sum(s["off_br"] for s in steps) / n
        rows.append({
            "pid": e["pid"], "ep": e["ep"],
            "target": e["target"],
            "n": n, "R": last["cum"],
            "tip_final": last["tip3d"],
            "tip_max_z": max_z,
            "max_entries": max_entries,
            "max_arc_past": max_arc_past,
            "off_br_pct": off_br_pct,
            "dom_named_last100": dominant,
            "named_dist_last100": named_ct,
        })

    # CATEGORIZE outcome — use TRUE step count from EPISODE_END, not n_snapshots
    for r in rows:
        e = next((x for x in all_eps if x["pid"] == r["pid"] and x["ep"] == r["ep"]), None)
        true_n = e["end"][0] if (e and e["end"]) else r["n"]
        true_R = e["end"][1] if (e and e["end"]) else r["R"]
        abort = e["end"][2] if (e and e["end"]) else "?"
        r["true_n"] = true_n
        r["true_R"] = true_R
        r["abort"] = abort
        r["target_branch"] = classify_target_t3d(r["target"])
        if true_R > 1.0:
            r["outcome"] = "SUCCESS"
        elif true_n >= 580:
            r["outcome"] = "WEDGE600"
        elif true_n < 200:
            r["outcome"] = "FOLD"
        else:
            r["outcome"] = "MID"

    # PER-TRUE-BRANCH BREAKDOWN
    print(f"\n--- Per-target-branch outcomes ---")
    by_branch = defaultdict(list)
    for r in rows:
        by_branch[r["target_branch"]].append(r)
    for branch in ["LCCA", "LVA", "RCCA", "RVA"]:
        eps = by_branch.get(branch, [])
        outcomes = Counter(r["outcome"] for r in eps)
        successes = [r for r in eps if r["outcome"] == "SUCCESS"]
        wedges = [r for r in eps if r["outcome"] == "WEDGE600"]
        folds = [r for r in eps if r["outcome"] == "FOLD"]
        mid = [r for r in eps if r["outcome"] == "MID"]
        print(f"\n  {branch}: {len(eps)} eps "
              f"({len(successes)}S / {len(folds)}F / {len(mid)}M / {len(wedges)}W)")
        if wedges:
            avg_max_z = sum(r["tip_max_z"] for r in wedges) / len(wedges)
            avg_entries = sum(r["max_entries"] for r in wedges) / len(wedges)
            avg_off_pct = sum(r["off_br_pct"] for r in wedges) / len(wedges)
            print(f"    wedge stats: mean_max_z={avg_max_z:.0f} mean_max_entries={avg_entries:.2f} mean_off%={100*avg_off_pct:.0f}")
            named_ct = Counter()
            for r in wedges:
                named_ct.update(r["named_dist_last100"])
            tot = sum(named_ct.values())
            print(f"    last-100 nearest_named (across all wedges):")
            for n, c in named_ct.most_common():
                print(f"      {n}: {c} ({100*c/tot:.0f}%)")
            # Final tip locations — bin into 10mm boxes
            tip_buckets = Counter(
                (round(r["tip_final"][0] / 5) * 5,
                 round(r["tip_final"][1] / 5) * 5,
                 round(r["tip_final"][2] / 5) * 5)
                for r in wedges
            )
            print(f"    final tip clusters (5mm grid):")
            for box, c in tip_buckets.most_common(8):
                print(f"      {box}: {c}")
            # Print first 3 wedge episodes
            print(f"    sample wedge eps:")
            for r in wedges[:5]:
                print(f"      pid={r['pid']} ep={r['ep']} target_t3d=({r['target'][0]:.0f},{r['target'][1]:.0f},{r['target'][2]:.0f}) "
                      f"R={r['true_R']:.1f} max_z={r['tip_max_z']:.0f} entries={r['max_entries']} "
                      f"off={100*r['off_br_pct']:.0f}% final_tip=({r['tip_final'][0]:.0f},{r['tip_final'][1]:.0f},{r['tip_final'][2]:.0f}) "
                      f"named_l100={dict(r['named_dist_last100'].most_common())}")
        if successes:
            print(f"    SUCCESSES:")
            for r in successes:
                print(f"      pid={r['pid']} ep={r['ep']} target=({r['target'][0]:.0f},{r['target'][1]:.0f},{r['target'][2]:.0f}) R={r['true_R']:.2f}")
    return  # end of main

    # -- Group wedge eps by their dominant nearest_named (where wire settled)
    wedges = [r for r in rows if r["outcome"] == "WEDGE600"]
    print(f"\n--- {len(wedges)} WEDGE600 episodes, grouped by dominant-nearest in last 100 steps ---")
    by_dom = defaultdict(list)
    for r in wedges:
        by_dom[r["dom_named_last100"]].append(r)
    for dom in sorted(by_dom):
        eps = by_dom[dom]
        # Mean target z (broad branch hint)
        mz = sum(r["target"][2] for r in eps) / len(eps)
        my = sum(r["target"][1] for r in eps) / len(eps)
        mx = sum(r["target"][0] for r in eps) / len(eps)
        print(f"\n  dom={dom}: {len(eps)} wedges  mean_target=({mx:.1f},{my:.1f},{mz:.1f})")
        # Mean tip final position
        ftx = sum(r["tip_final"][0] for r in eps) / len(eps)
        fty = sum(r["tip_final"][1] for r in eps) / len(eps)
        ftz = sum(r["tip_final"][2] for r in eps) / len(eps)
        print(f"    mean_tip_final=({ftx:.1f},{fty:.1f},{ftz:.1f})")
        avg_max_z = sum(r["tip_max_z"] for r in eps) / len(eps)
        avg_entries = sum(r["max_entries"] for r in eps) / len(eps)
        avg_off = sum(r["off_br_pct"] for r in eps) / len(eps)
        print(f"    mean_tip_max_z={avg_max_z:.1f}  mean_max_entries={avg_entries:.2f}  mean_off_br={100*avg_off:.0f}%")
        # Sample 5
        for r in eps[:5]:
            print(
                f"      pid={r['pid']} ep={r['ep']} target=({r['target'][0]:.0f},{r['target'][1]:.0f},{r['target'][2]:.0f}) "
                f"R={r['R']:.1f} max_z={r['tip_max_z']:.0f} entries={r['max_entries']} off={100*r['off_br_pct']:.0f}% "
                f"final=({r['tip_final'][0]:.0f},{r['tip_final'][1]:.0f},{r['tip_final'][2]:.0f}) "
                f"named_l100={dict(r['named_dist_last100'].most_common())}"
            )

    # Cross-tab: for each target Z-band (proxy for branch), what does the wire commit to?
    # Use 4 z-bands seen in centerlines (after vessel_cs transform): need empirical.
    # Just use the SUCCESS targets as anchor points.
    if successes:
        anchors = [(r["target"], r["dom_named_last100"]) for r in successes]
        print(f"\n--- Cross-tab: target-anchor branch (nearest success) → wedge dominant_named ---")
        # For each wedge, find the closest success-target and label by its branch
        cross = defaultdict(lambda: Counter())
        for r in wedges:
            best_name = "?"
            best_d2 = 1e18
            for (tx, ty, tz), name in anchors:
                d2 = (r["target"][0] - tx) ** 2 + (r["target"][1] - ty) ** 2 + (r["target"][2] - tz) ** 2
                if d2 < best_d2:
                    best_d2, best_name = d2, name
            cross[best_name][r["dom_named_last100"]] += 1
        for src, dst_ct in cross.items():
            total = sum(dst_ct.values())
            print(f"  target_label={src} ({total} wedges):")
            for d, c in dst_ct.most_common():
                print(f"    settled→{d}: {c} ({100*c/total:.0f}%)")


if __name__ == "__main__":
    main()
