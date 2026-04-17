#!/usr/bin/env python3
"""Export episode summary rows from `docker logs` into a CSV.

Parses BenchEnv5-style logs: the last `inserted=[gw_mm,cath_mm]` before each
`EPISODE_END` for a given (pid, ep) pair, plus optional `heur_abort`.

Usage:
  python docker_episode_log_to_csv.py --container env5_parallel_seed_r2
  python docker_episode_log_to_csv.py -c my_run -o results/my_episodes.csv
  python docker_episode_log_to_csv.py -c my_run --tail 50000   # last N lines only

Environment:
  DOCKER_EXPERIMENT_CONTAINER  default container name if --container omitted
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Last inserted lengths from a STEP line: inserted=[gw,cath] (mm)
_RE_INSERTED = re.compile(r"inserted=\[([\d.]+),([\d.]+)\]")
_RE_STEP_HEAD = re.compile(r"STEP: STEP \| ep=(\d+)")
_RE_STEP_PID = re.compile(r"pid=(\d+)")
_RE_EPISODE_END = re.compile(
    r"EPISODE_END \| ep=(\d+) \| steps=(\d+) \| total_reward=([-\d.]+)"
)
_RE_PID = re.compile(r"pid=(\d+)")
_RE_HEUR_ABORT = re.compile(r"heur_abort=(\S+)")


def _accumulate_until_pid(lines: List[str], start_i: int) -> Tuple[str, int]:
    """Join EPISODE_END block until `pid=` appears (may span 2 lines)."""
    parts: List[str] = []
    i = start_i
    while i < len(lines):
        parts.append(lines[i].strip())
        block = " ".join(parts)
        if "pid=" in block:
            return block, i + 1
        i += 1
    return " ".join(parts), i


def parse_bench_env_log(text: str) -> List[Dict[str, Any]]:
    """Parse docker log text into one dict per completed episode."""
    lines = text.splitlines()
    # Most recent inserted per (worker_pid, episode_index) seen in STEP logs
    last_inserted: Dict[Tuple[int, int], Tuple[float, float]] = {}
    rows: List[Dict[str, Any]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("STEP: STEP |"):
            block_parts = [line]
            i += 1
            while i < len(lines) and not (
                lines[i].startswith("STEP: ")
                or lines[i].startswith("TRACE_")
            ):
                block_parts.append(lines[i])
                i += 1
            block = "".join(block_parts)
            m_ep = _RE_STEP_HEAD.search(block)
            m_pid = _RE_STEP_PID.search(block)
            m_ins = _RE_INSERTED.search(block)
            if m_ep and m_pid and m_ins:
                ep = int(m_ep.group(1))
                pid = int(m_pid.group(1))
                last_inserted[(pid, ep)] = (
                    float(m_ins.group(1)),
                    float(m_ins.group(2)),
                )
            continue

        if "EPISODE_END" in line:
            block, next_i = _accumulate_until_pid(lines, i)
            i = next_i
            m_end = _RE_EPISODE_END.search(block)
            m_pid = _RE_PID.search(block)
            if not m_end or not m_pid:
                continue
            ep = int(m_end.group(1))
            steps = int(m_end.group(2))
            total_reward = float(m_end.group(3))
            pid = int(m_pid.group(1))
            m_h = _RE_HEUR_ABORT.search(block)
            heur_abort = m_h.group(1) if m_h else ""
            ins = last_inserted.get((pid, ep))
            if ins:
                ig, ic = ins[0], ins[1]
            else:
                ig, ic = "", ""
            rows.append(
                {
                    "pid": pid,
                    "ep_index": ep,
                    "steps": steps,
                    "total_reward": total_reward,
                    "inserted_gw_mm": ig,
                    "inserted_cath_mm": ic,
                    "heur_abort": heur_abort,
                }
            )
            continue

        i += 1

    rows.sort(key=lambda r: (r["pid"], r["ep_index"]))
    return rows


def fetch_docker_logs(
    container: str,
    *,
    docker_bin: str = "docker",
    tail_lines: Optional[int] = None,
) -> str:
    cmd = [docker_bin, "logs", container]
    if tail_lines is not None and tail_lines > 0:
        cmd = [docker_bin, "logs", "--tail", str(tail_lines), container]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(
            f"docker logs failed (exit {proc.returncode}) for container {container!r}"
        )
    return proc.stdout + (proc.stderr or "")


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "pid",
        "ep_index",
        "steps",
        "total_reward",
        "inserted_gw_mm",
        "inserted_cath_mm",
        "heur_abort",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def default_output_path(container: str) -> str:
    repo_results = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"{container}_episodes.csv",
    )
    return repo_results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write episode CSV from docker logs (BenchEnv5 step format)."
    )
    p.add_argument(
        "-c",
        "--container",
        default=os.environ.get("DOCKER_EXPERIMENT_CONTAINER", ""),
        help="Docker container name (or set DOCKER_EXPERIMENT_CONTAINER)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help="Output CSV path (default: training_scripts/results/<container>_episodes.csv)",
    )
    p.add_argument(
        "--docker-bin",
        default="docker",
        help="Docker executable (default: docker)",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=None,
        metavar="N",
        help="Only parse last N log lines (docker logs --tail N)",
    )
    p.add_argument(
        "--from-file",
        metavar="PATH",
        help="Read log from file instead of docker logs (for offline use)",
    )
    args = p.parse_args()

    if args.from_file:
        with open(args.from_file, encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        if not args.container:
            p.error("Pass --container NAME or set DOCKER_EXPERIMENT_CONTAINER")
        text = fetch_docker_logs(
            args.container, docker_bin=args.docker_bin, tail_lines=args.tail
        )

    rows = parse_bench_env_log(text)
    out = args.output or default_output_path(args.container or "experiment")
    write_csv(rows, out)

    print(
        f"Wrote {len(rows)} episodes to {out}",
        file=sys.stderr,
    )
    if rows:
        rewards = [float(r["total_reward"]) for r in rows]
        print(
            f"  total_reward min={min(rewards):.4f} max={max(rewards):.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
