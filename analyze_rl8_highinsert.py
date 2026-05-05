"""Re-analysis of env5_rl8_highinsert_50ep using the run-28 methodology.

Mirrors analyze_run28_branches.py:
- SUCCESS:  total_reward > 1.0  (not term flag)
- WEDGE600: true steps >= 580
- FOLD:     true steps < 200
- MID:      otherwise
- entries_passed > entries_at_start := wire crossed at least one new junction
- nearest_named over last 100 INFO steps := which daughter the wire was in
"""
import os
import re
import glob
import json
import numpy as np
from collections import Counter

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


def t3d_to_vcs(p):
    return _ROT.T @ np.asarray(p, dtype=float)


NAMED_CL = {}
for d in ["LCCA", "LVA", "RCCA", "RVA"]:
    with open(os.path.join(DATA_DIR, "Centerline curve - %s.mrk.json" % d)) as f:
        data = json.load(f)
    pts = []
    for m in data["markups"]:
        if m["type"] == "Curve":
            for cp in m["controlPoints"]:
                x, y, z = cp["position"]
                pts.append((y, -z, -x))
    NAMED_CL[d] = np.array(pts)


def classify_target(t3d):
    tv = t3d_to_vcs(t3d)
    return min(NAMED_CL, key=lambda n: float(np.min(np.linalg.norm(NAMED_CL[n] - tv, axis=1))))


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
# RL_IMPROV_8: separate regex for new fields. Optional — old logs won't match.
RE_STEP_NEW = re.compile(
    r"on_path=(\d).*?d_corr_3d=([\d.infa]+) \| arc_past_d=([\d.\-]+).*?daughters_passed=(\d+)"
)


def parse_worker(path, pid):
    cur = {"ep": None, "target": None, "steps": [], "end": None}
    out = []

    def flush():
        if cur["ep"] is not None and cur["steps"]:
            out.append(
                {
                    "pid": pid,
                    "ep": cur["ep"],
                    "target": cur["target"],
                    "steps": list(cur["steps"]),
                    "end": cur["end"],
                }
            )
        cur["ep"] = None
        cur["target"] = None
        cur["steps"] = []
        cur["end"] = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_EP_START.search(line)
            if m:
                flush()
                cur["ep"] = int(m.group(1))
                cur["target"] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
                continue
            m = RE_EP_END.search(line)
            if m and cur["ep"] is not None and int(m.group(1)) == cur["ep"]:
                cur["end"] = (int(m.group(2)), float(m.group(3)), m.group(4))
                continue
            m = RE_STEP.search(line)
            if m and cur["ep"] is not None and int(m.group(1)) == cur["ep"]:
                cur["steps"].append(
                    {
                        "ep_step": int(m.group(2)),
                        "cum": float(m.group(4)),
                        "on_br": int(m.group(5)),
                        "off_br": int(m.group(6)),
                        "fold": int(m.group(7)),
                        "arc_past": float(m.group(9)),
                        "nearest": m.group(10),
                        "entries": int(m.group(11)),
                        "tip3d": (
                            float(m.group(12)),
                            float(m.group(13)),
                            float(m.group(14)),
                        ),
                    }
                )
    flush()
    return out


def main():
    all_eps = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "worker_*.log"))):
        pid = int(re.search(r"worker_(\d+)\.log", path).group(1))
        all_eps.extend(parse_worker(path, pid))
    print("Parsed %d episodes\n" % len(all_eps))

    rows = []
    for e in all_eps:
        if not e["steps"]:
            continue
        last = e["steps"][-1]
        n_snap = len(e["steps"])

        if e["end"]:
            true_n = e["end"][0]
            true_R = e["end"][1]
            abort = e["end"][2]
        else:
            true_n = last["ep_step"]
            true_R = last["cum"]
            abort = "?"

        last100 = e["steps"][-min(100, n_snap):]
        named_only = Counter(s["nearest"] for s in last100 if s["nearest"] in {"LCCA", "LVA", "RCCA", "RVA"})
        dom = named_only.most_common(1)[0][0] if named_only else "none"

        entries_first = e["steps"][0]["entries"]
        entries_max = max(s["entries"] for s in e["steps"])
        entries_gained = entries_max - entries_first

        if true_R > 1.0:
            outcome = "SUCCESS"
        elif true_n >= 580:
            outcome = "WEDGE"
        elif true_n < 200:
            outcome = "FOLD"
        else:
            outcome = "MID"

        rows.append(
            {
                "pid": e["pid"],
                "ep": e["ep"],
                "target": e["target"],
                "true_n": true_n,
                "true_R": true_R,
                "abort": abort,
                "tip_final": last["tip3d"],
                "tip_max_z": max(s["tip3d"][2] for s in e["steps"]),
                "entries_first": entries_first,
                "entries_max": entries_max,
                "entries_gained": entries_gained,
                "dom_named": dom,
                "target_branch": classify_target(e["target"]),
                "outcome": outcome,
            }
        )

    print("Schedule (target distribution): %s" % dict(Counter(r["target_branch"] for r in rows)))
    print("Outcomes: %s\n" % dict(Counter(r["outcome"] for r in rows)))

    print("%-6s %3s %4s %5s %3s %4s" % ("Target", "n", "SUCC", "WEDGE", "MID", "FOLD"))
    for br in ["LCCA", "LVA", "RCCA", "RVA"]:
        eps = [r for r in rows if r["target_branch"] == br]
        print("%-6s %3d %4d %5d %3d %4d" % (
            br,
            len(eps),
            sum(1 for r in eps if r["outcome"] == "SUCCESS"),
            sum(1 for r in eps if r["outcome"] == "WEDGE"),
            sum(1 for r in eps if r["outcome"] == "MID"),
            sum(1 for r in eps if r["outcome"] == "FOLD"),
        ))

    print("\n=== DAUGHTER ENTRY CHECK ===")
    print("entries_first (junctions pre-seen at episode start):")
    for k, v in sorted(Counter(r["entries_first"] for r in rows).items()):
        print("  entries_first=%d: %d episodes" % (k, v))

    gained = [r for r in rows if r["entries_gained"] >= 1]
    print("\nEpisodes that crossed >=1 NEW junction during the episode: %d / %d = %.0f%%" % (
        len(gained), len(rows), 100 * len(gained) / len(rows)
    ))

    print("\nentries_gained distribution:")
    for k, v in sorted(Counter(r["entries_gained"] for r in rows).items()):
        print("  +%d: %d" % (k, v))

    print("\n=== Dominant nearest_named (last 100 INFO steps) ===")
    print("Across all 50 episodes:")
    for k, v in Counter(r["dom_named"] for r in rows).most_common():
        print("  %s: %d" % (k, v))
    print("\nFor episodes with entries_gained >= 1 (committed past a junction):")
    for k, v in Counter(r["dom_named"] for r in gained).most_common():
        print("  %s: %d" % (k, v))

    print("\n=== ALL SUCCESSES ===")
    for r in [x for x in rows if x["outcome"] == "SUCCESS"]:
        print("  pid=%d ep=%d target=%s R=%.2f steps=%d entries=%d->%d (gained %d) dom_named=%s tip_final=(%.0f,%.0f,%.0f)" % (
            r["pid"], r["ep"], r["target_branch"], r["true_R"], r["true_n"],
            r["entries_first"], r["entries_max"], r["entries_gained"],
            r["dom_named"], r["tip_final"][0], r["tip_final"][1], r["tip_final"][2]
        ))

    print("\n=== AORTIC ARCH WEDGE (80,55,395)+-8 — the section 16 metric ===")
    non_lva_wedges = [r for r in rows if r["outcome"] == "WEDGE" and r["target_branch"] != "LVA"]
    arch = [
        r for r in non_lva_wedges
        if abs(r["tip_final"][0] - 80) < 8
        and abs(r["tip_final"][1] - 55) < 8
        and abs(r["tip_final"][2] - 395) < 8
    ]
    print("Non-LVA WEDGE600: %d  |  at (80,55,395)+-8: %d (%.0f%% of non-LVA wedges)" % (
        len(non_lva_wedges), len(arch),
        100 * len(arch) / max(1, len(non_lva_wedges))
    ))

    print("\nFinal tip3d clusters (5mm grid) for non-LVA wedges:")
    clusters = Counter(
        (round(r["tip_final"][0] / 5) * 5, round(r["tip_final"][1] / 5) * 5, round(r["tip_final"][2] / 5) * 5)
        for r in non_lva_wedges
    )
    for k, v in clusters.most_common(8):
        print("  %s: %d" % (k, v))

    print("\nFinal tip3d clusters (5mm grid) for LVA wedges:")
    lva_wedges = [r for r in rows if r["outcome"] == "WEDGE" and r["target_branch"] == "LVA"]
    clusters = Counter(
        (round(r["tip_final"][0] / 5) * 5, round(r["tip_final"][1] / 5) * 5, round(r["tip_final"][2] / 5) * 5)
        for r in lva_wedges
    )
    for k, v in clusters.most_common(5):
        print("  %s: %d" % (k, v))

    print("\n=== entries_gained CROSSTABS ===")
    print("Per target_branch x outcome (mean entries_gained):")
    print("%-6s %5s %7s %5s %7s %5s %7s %5s %7s" % (
        "Target", "S_n", "S_eg", "W_n", "W_eg", "M_n", "M_eg", "F_n", "F_eg"
    ))
    for br in ["LCCA", "LVA", "RCCA", "RVA"]:
        out_data = {}
        for oc in ["SUCCESS", "WEDGE", "MID", "FOLD"]:
            eps = [r for r in rows if r["target_branch"] == br and r["outcome"] == oc]
            mean_eg = sum(r["entries_gained"] for r in eps) / max(1, len(eps))
            out_data[oc] = (len(eps), mean_eg)
        print("%-6s %5d %7.2f %5d %7.2f %5d %7.2f %5d %7.2f" % (
            br,
            out_data["SUCCESS"][0], out_data["SUCCESS"][1],
            out_data["WEDGE"][0],   out_data["WEDGE"][1],
            out_data["MID"][0],     out_data["MID"][1],
            out_data["FOLD"][0],    out_data["FOLD"][1],
        ))

    print("\nFor ARCH-WEDGE episodes (final tip at (80,55,395)+-8) only:")
    arch_eps = [r for r in rows
                if r["outcome"] == "WEDGE" and r["target_branch"] != "LVA"
                and abs(r["tip_final"][0] - 80) < 8
                and abs(r["tip_final"][1] - 55) < 8
                and abs(r["tip_final"][2] - 395) < 8]
    print("  n=%d  entries_gained dist: %s" % (
        len(arch_eps),
        dict(Counter(r["entries_gained"] for r in arch_eps))
    ))
    for r in arch_eps:
        print("    pid=%d ep=%d target=%s entries=%d->%d (gained %d) tip_final=(%.0f,%.0f,%.0f) max_z=%.0f" % (
            r["pid"], r["ep"], r["target_branch"],
            r["entries_first"], r["entries_max"], r["entries_gained"],
            r["tip_final"][0], r["tip_final"][1], r["tip_final"][2], r["tip_max_z"]
        ))

    print("\nFor non-LVA WEDGE episodes that gained >=1 junction:")
    nonlva_w_gained = [r for r in rows
                       if r["outcome"] == "WEDGE" and r["target_branch"] != "LVA"
                       and r["entries_gained"] >= 1]
    print("  n=%d  final-tip clusters:" % len(nonlva_w_gained))
    for k, v in Counter(
        (round(r["tip_final"][0]/5)*5, round(r["tip_final"][1]/5)*5, round(r["tip_final"][2]/5)*5)
        for r in nonlva_w_gained
    ).most_common(8):
        print("    %s: %d" % (k, v))

    print("\n=== d_corr_arc — how close did the wire get to its target's bif2 ostium? ===")
    print("(d_corr_arc = arclength along planned path to NEXT correct entry; 0 = at the entry, lower = closer)")

    # Min d_corr_arc per episode — needs the raw step records
    ep_min_d = {}
    for e in all_eps:
        if not e["steps"]:
            continue
        finite_d = [s["arc_past"] for s in e["steps"] if s.get("arc_past") is not None]
        # arc_past is junction-past, NOT what we want — we want d_corr_arc
        # Re-parse with a different field
        # Use min of all step values reported as 'arc_past' here is wrong; rebuild from regex.
    # Simpler: re-parse logs and capture d_corr_arc directly.
    DCORR_RE = re.compile(
        r"STEP \| ep=(\d+) \|.*?d_corr_arc=([\d.infa]+)"
    )
    ep_min_d = {}  # (pid, ep) -> min finite d_corr_arc seen
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "worker_*.log"))):
        pid = int(re.search(r"worker_(\d+)\.log", path).group(1))
        cur_ep = None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = RE_EP_START.search(line)
                if m:
                    cur_ep = int(m.group(1))
                    continue
                m = DCORR_RE.search(line)
                if m and cur_ep is not None and int(m.group(1)) == cur_ep:
                    val_str = m.group(2)
                    if val_str == "inf":
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    key = (pid, cur_ep)
                    if key not in ep_min_d or val < ep_min_d[key]:
                        ep_min_d[key] = val

    # Attach to rows
    for r in rows:
        r["min_d_corr_arc"] = ep_min_d.get((r["pid"], r["ep"]))

    # Per-target summary
    print("\nPer target_branch x outcome — min d_corr_arc reached during episode:")
    print("%-6s %-8s %4s %10s %10s %10s" % (
        "Target", "Outcome", "n", "mean_min_d", "min_min_d", "max_min_d"
    ))
    for br in ["LCCA", "LVA", "RCCA", "RVA"]:
        for oc in ["SUCCESS", "WEDGE", "MID", "FOLD"]:
            eps = [r for r in rows if r["target_branch"] == br and r["outcome"] == oc and r["min_d_corr_arc"] is not None]
            if not eps:
                continue
            ds = [r["min_d_corr_arc"] for r in eps]
            print("%-6s %-8s %4d %10.2f %10.2f %10.2f" % (
                br, oc, len(eps),
                sum(ds) / len(ds), min(ds), max(ds)
            ))

    print("\nFor RCCA + RVA target episodes specifically (the ones the user asked about):")
    rrva = [r for r in rows if r["target_branch"] in {"RCCA", "RVA"}]
    print("  n=%d total" % len(rrva))
    thresholds = [1, 2, 5, 10, 20]
    for thr in thresholds:
        n_close = sum(1 for r in rrva if r["min_d_corr_arc"] is not None and r["min_d_corr_arc"] < thr)
        print("  episodes with min d_corr_arc < %d mm: %d" % (thr, n_close))

    print("\nDetail per RCCA/RVA episode (min d_corr_arc, entries_gained, outcome, final tip):")
    for r in sorted(rrva, key=lambda x: x["min_d_corr_arc"] if x["min_d_corr_arc"] is not None else 999):
        d = r["min_d_corr_arc"]
        d_str = ("%.2f" % d) if d is not None else "  N/A"
        print("  pid=%d ep=%d %s outcome=%-7s min_d_corr=%s mm  eg=%d  R=%6.2f  tip_final=(%.0f,%.0f,%.0f) max_z=%.0f" % (
            r["pid"], r["ep"], r["target_branch"], r["outcome"],
            d_str, r["entries_gained"], r["true_R"],
            r["tip_final"][0], r["tip_final"][1], r["tip_final"][2], r["tip_max_z"]
        ))

    print("\n=== CORRECTED: 3D Euclidean min-distance from tip to actual ostia ===")
    print("(Replaces d_corr_arc which only measures arclength-projection, not 3D distance)")

    # Junction points per target's required path (vessel-CS, from raw JSON first_pt)
    OSTIA = {
        "LCCA": np.array([23.2, 15.7, 384.7]),    # main trunk -> LCCA
        "LVA":  np.array([47.5, 34.5, 430.1]),    # internal LCCA-LVA junction
        "RCCA": np.array([-0.4, 24.1, 416.2]),    # bif2b
        "RVA":  np.array([-0.4, 24.1, 416.2]),    # bif2b (shared with RCCA)
    }

    # For each episode, compute min 3D Euclidean dist from tip-in-vessel-CS to the
    # *target-specific* ostium across all logged STEP tip3d values
    ep_min_eucl = {}  # (pid, ep) -> min euclidean dist (mm)
    for e in all_eps:
        if not e["steps"]:
            continue
        tgt_branch = classify_target(e["target"])
        ostium = OSTIA[tgt_branch]
        min_d = float("inf")
        for s in e["steps"]:
            tip_t3d = np.asarray(s["tip3d"], dtype=float)
            tip_vcs = _ROT.T @ tip_t3d
            d = float(np.linalg.norm(tip_vcs - ostium))
            if d < min_d:
                min_d = d
        ep_min_eucl[(e["pid"], e["ep"])] = min_d

    for r in rows:
        r["min_euclid_to_ostium"] = ep_min_eucl.get((r["pid"], r["ep"]))

    print("\nPer target_branch x outcome — min 3D Euclidean dist from tip to its ostium:")
    print("%-6s %-8s %4s %12s %10s %10s" % (
        "Target", "Outcome", "n", "mean_min_eu", "min_min_eu", "max_min_eu"
    ))
    for br in ["LCCA", "LVA", "RCCA", "RVA"]:
        for oc in ["SUCCESS", "WEDGE", "MID", "FOLD"]:
            eps = [r for r in rows if r["target_branch"] == br and r["outcome"] == oc and r["min_euclid_to_ostium"] is not None]
            if not eps:
                continue
            ds = [r["min_euclid_to_ostium"] for r in eps]
            print("%-6s %-8s %4d %12.2f %10.2f %10.2f" % (
                br, oc, len(eps),
                sum(ds) / len(ds), min(ds), max(ds),
            ))

    print("\nFor RCCA + RVA target episodes (the ones the user asked about):")
    rrva = [r for r in rows if r["target_branch"] in {"RCCA", "RVA"}]
    print("  n=%d" % len(rrva))
    for thr in [2, 5, 10, 20, 50]:
        n_close = sum(1 for r in rrva if r["min_euclid_to_ostium"] is not None and r["min_euclid_to_ostium"] < thr)
        print("  min Euclidean dist < %2d mm: %d" % (thr, n_close))

    print("\nDetail per RCCA/RVA episode (min Euclidean dist to ostium, both metrics):")
    print("%-22s %-7s %-6s %12s %12s %4s %7s" % (
        "pid/ep/branch", "outcome", "tip_z", "min_d_corr", "min_eucl_3d", "eg", "R"
    ))
    for r in sorted(rrva, key=lambda x: x["min_euclid_to_ostium"] if x["min_euclid_to_ostium"] is not None else 999):
        d_arc = ("%.2f" % r["min_d_corr_arc"]) if r["min_d_corr_arc"] is not None else "  N/A"
        d_eu  = ("%.1f" % r["min_euclid_to_ostium"]) if r["min_euclid_to_ostium"] is not None else "  N/A"
        print("%-22s %-7s %6.0f %10s mm %10s mm %4d %7.2f" % (
            "pid=%d ep=%d %s" % (r["pid"], r["ep"], r["target_branch"]),
            r["outcome"], r["tip_max_z"],
            d_arc, d_eu,
            r["entries_gained"], r["true_R"]
        ))

    print("\n=== Sanity-check the classifier on the four success targets ===")
    for r in [x for x in rows if x["outcome"] == "SUCCESS"]:
        tv = t3d_to_vcs(r["target"])
        print("  target_t3d=%s -> vcs=(%.1f,%.1f,%.1f)" % (r["target"], tv[0], tv[1], tv[2]))
        for n, pts in NAMED_CL.items():
            d = float(np.min(np.linalg.norm(pts - tv, axis=1)))
            print("    dist_to_%s_centerline: %.2f mm" % (n, d))


if __name__ == "__main__":
    main()
