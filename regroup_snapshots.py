"""One-shot: regroup existing snapshot PNGs by (target_branch, reason).

Reads worker_<pid>.log files to extract (pid, ep) -> target_t3d, classifies
each target by min-distance to the four named daughter centerlines (LCCA,
LVA, RCCA, RVA), and moves each PNG from ``snapshots/<reason>/`` to
``snapshots/<target_branch>/<reason>/``.

Idempotent: if PNGs are already in the new layout it skips them. Will not
delete the now-empty ``<reason>`` dirs; remove manually after sanity-check.

Usage:
    python regroup_snapshots.py <run_dir>

where ``<run_dir>`` is the timestamped run directory under
``saved/eve_paper/.../mesh_ben/``.
"""
import os
import re
import sys
import json
import glob
import shutil
import numpy as np


CENTERLINE_FILES_DIR = "eve_bench/data/dualdevicenav/Centrelines_comb"
DAUGHTER_TAGS = ("LCCA", "LVA", "RCCA", "RVA")


def _get_rot_matrix(rzx):
    rz = -rzx[0] * np.pi / 180
    rx = -rzx[1] * np.pi / 180
    Rz = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ]
    )
    return Rz @ Rx


_ROT = _get_rot_matrix((20, 5))


def t3d_to_vcs(p):
    return _ROT.T @ np.asarray(p, dtype=float)


def load_daughter_centerlines():
    """Load LCCA / LVA / RCCA / RVA centerlines from raw JSON, transformed
    to vessel-CS via the same (y, -z, -x) mapping used at runtime.
    """
    out = {}
    for tag in DAUGHTER_TAGS:
        fp = os.path.join(
            CENTERLINE_FILES_DIR,
            f"Centerline curve - {tag}.mrk.json",
        )
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = []
        for m in data["markups"]:
            if m["type"] == "Curve":
                for cp in m["controlPoints"]:
                    x, y, z = cp["position"]
                    pts.append((y, -z, -x))
        out[tag] = np.array(pts, dtype=float)
    return out


def classify_target(t3d, daughters):
    vcs = t3d_to_vcs(t3d)
    best_tag = "unknown"
    best_d = float("inf")
    for tag, coords in daughters.items():
        if coords.size == 0:
            continue
        d = float(np.min(np.linalg.norm(coords - vcs, axis=1)))
        if d < best_d:
            best_d = d
            best_tag = tag
    return best_tag


def parse_worker_logs(log_dir):
    """Return dict ``{(pid, ep): (t3d_x, t3d_y, t3d_z)}`` from
    EPISODE_START lines."""
    pat = re.compile(
        r"EPISODE_START \| ep=(\d+) \|.*?pid=(\d+) \| target=\(([^)]+)\)"
    )
    out = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "worker_*.log"))):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    ep = int(m.group(1))
                    pid = int(m.group(2))
                    t3d = tuple(float(x) for x in m.group(3).split(","))
                    out[(pid, ep)] = t3d
    return out


def main(run_dir):
    snapshots_dir = os.path.join(run_dir, "diagnostics", "snapshots")
    log_dir = os.path.join(run_dir, "diagnostics", "logs_subprocesses")
    if not os.path.isdir(snapshots_dir):
        print(f"ERR: {snapshots_dir} does not exist")
        sys.exit(1)
    if not os.path.isdir(log_dir):
        print(f"ERR: {log_dir} does not exist")
        sys.exit(1)

    print(f"Loading daughter centerlines from {CENTERLINE_FILES_DIR} ...")
    daughters = load_daughter_centerlines()
    for tag, c in daughters.items():
        print(f"  {tag}: {len(c)} pts, z=[{c[:,2].min():.1f}, {c[:,2].max():.1f}]")

    print(f"\nParsing worker logs in {log_dir} ...")
    targets = parse_worker_logs(log_dir)
    print(f"  found {len(targets)} (pid, ep) -> target_t3d entries")

    # The CURRENT layout is snapshots/<reason>/<file>.png.
    # The TARGET layout is snapshots/<target>/<reason>/<file>.png.
    # Move each PNG accordingly. Skip files already under a top-level
    # daughter dir (idempotent).
    fname_pat = re.compile(
        r"ep(\d+)_pid(\d+)_step(\d+)_(.+)\.png"
    )

    moved = 0
    skipped_already = 0
    skipped_no_target = 0
    skipped_unknown = 0

    # Iterate top-level dirs in snapshots_dir. Anything that's a daughter
    # tag is already-organised — skip. Otherwise it's a reason dir from
    # the old layout: descend.
    for reason in sorted(os.listdir(snapshots_dir)):
        reason_dir = os.path.join(snapshots_dir, reason)
        if not os.path.isdir(reason_dir):
            continue
        if reason in DAUGHTER_TAGS or reason == "unknown":
            # Already in the new layout; leave alone.
            continue

        for fname in sorted(os.listdir(reason_dir)):
            src = os.path.join(reason_dir, fname)
            if not os.path.isfile(src) or not fname.endswith(".png"):
                continue

            m = fname_pat.match(fname)
            if not m:
                print(f"  SKIP malformed filename: {fname}")
                continue
            ep = int(m.group(1))
            pid = int(m.group(2))

            t3d = targets.get((pid, ep))
            if t3d is None:
                skipped_no_target += 1
                continue

            tag = classify_target(t3d, daughters)
            if tag == "unknown":
                skipped_unknown += 1

            dst_dir = os.path.join(snapshots_dir, tag, reason)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            if os.path.exists(dst):
                skipped_already += 1
                continue
            shutil.move(src, dst)
            moved += 1

    print(
        f"\nDone. Moved {moved} files. "
        f"Skipped: already-existing={skipped_already}, "
        f"no-target-found={skipped_no_target}, "
        f"unknown-target={skipped_unknown}."
    )

    print("\nNew layout:")
    for tag in (*DAUGHTER_TAGS, "unknown"):
        tag_dir = os.path.join(snapshots_dir, tag)
        if not os.path.isdir(tag_dir):
            continue
        per_reason = {}
        for reason in sorted(os.listdir(tag_dir)):
            r_dir = os.path.join(tag_dir, reason)
            if os.path.isdir(r_dir):
                per_reason[reason] = len(os.listdir(r_dir))
        if per_reason:
            print(f"  {tag}: {per_reason}")

    print(
        "\nOld <reason>/ dirs still exist (now empty if all PNGs were"
        " classified). Remove them with:\n"
        f'  rmdir "{snapshots_dir}/wire_fold_stall"  '
        f'"{snapshots_dir}/max_steps"  '
        f'"{snapshots_dir}/wrong_branch_timeout"  (etc.)'
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
