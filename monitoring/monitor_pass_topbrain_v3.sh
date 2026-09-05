#!/usr/bin/env bash
# READ-ONLY health pass for the rcca_topbrain_v3 training run (the v3 mirror of the
# branch-18 machine's rcca_topbrain_v2; recipe: REPRO_topbrain_v3.md, launch record:
# HANDOFF 15.5). Makes NO changes to the run. Run from anywhere:
#     bash monitoring/monitor_pass_topbrain_v3.sh
# Override the container / run glob with CONTAINER= / RUN_GLOB= if reused for another run.
set -uo pipefail
C=${CONTAINER:-rcca_topbrain_v3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MB="$ROOT/saved/eve_paper/neurovascular/full/mesh_ben"
R=$(ls -d "$MB"/${RUN_GLOB:-2026-*_rcca_topbrain_v3} 2>/dev/null | tail -1)
BAD=0
echo "===== HEALTH PASS $(date '+%F %T')  container=$C  run=$(basename "${R:-none}") ====="

echo "[container]"
docker inspect "$C" --format '  Running={{.State.Running}} Restarts={{.RestartCount}} StartedAt={{.State.StartedAt}} ExitCode={{.State.ExitCode}}' 2>&1
docker inspect "$C" --format '{{.State.Running}}' 2>/dev/null | grep -q true || { echo "  !! CONTAINER NOT RUNNING"; BAD=1; }
docker stats --no-stream --format '  CPU {{.CPUPerc}}  MEM {{.MemUsage}}' "$C" 2>&1
mem=$(docker stats --no-stream --format '{{.MemUsage}}' "$C" 2>/dev/null | awk '{print $1}' | sed 's/GiB//')
[ -n "$mem" ] && awk -v m="$mem" 'BEGIN{ if (m+0 > 21.0) { print "  !! container RSS above 21 GiB of the 23.47 GiB VM"; exit 1 } }' || BAD=1

echo "[host]"
powershell -NoProfile -Command "\$o=Get-CimInstance Win32_OperatingSystem; '  RAM free {0:N1} / {1:N1} GiB' -f (\$o.FreePhysicalMemory/1MB),(\$o.TotalVisibleMemorySize/1MB); '  D: free {0:N0} GB' -f ((Get-PSDrive D).Free/1GB)" 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  GPU util,mem: /'
free_gb=$(powershell -NoProfile -Command "[math]::Floor((Get-PSDrive D).Free/1GB)" 2>/dev/null | tr -d '\r')
[ -n "$free_gb" ] && [ "$free_gb" -lt 20 ] && { echo "  !! under 20 GB free on D:"; BAD=1; }

if [ -n "${R:-}" ]; then
  echo "[workers] last-STEP age per worker; stalled = no STEP for 600 s"
  now=$(date +%s); n=0; stalled=0
  for f in "$R"/diagnostics/logs_subprocesses/worker_*.log; do
    [ -f "$f" ] || continue
    t=$(tail -c 6000 "$f" | grep -o 'wall_time=[0-9]*' | tail -1 | cut -d= -f2)
    [ -z "$t" ] && continue
    n=$((n+1)); age=$((now-t))
    if [ "$age" -gt 600 ]; then stalled=$((stalled+1)); echo "  STALLED $(basename "$f") age=${age}s"; fi
  done
  echo "  workers with STEP lines=$n  stalled=$stalled   (the eval worker is idle between evals and does not count as stalled)"
  # only flag if MOST workers are stalled: the eval worker is legitimately idle, and a
  # single busy SOFA episode can exceed 600 s
  [ "$n" -gt 0 ] && [ "$stalled" -ge 4 ] && { echo "  !! $stalled workers without a STEP in 10 min"; BAD=1; }

  echo "[throughput] env-steps/s over the last 10 min, all workers"
  python - "$R" <<'PY'
import glob, os, re, sys, time
R = sys.argv[1]; now = time.time(); lo = now - 600; ts = []
for p in glob.glob(os.path.join(R, "diagnostics/logs_subprocesses/worker_*.log")):
    with open(p, "rb") as fh:
        fh.seek(0, 2); sz = fh.tell(); fh.seek(max(0, sz - 400000)); tail = fh.read().decode("latin-1")
    for m in re.finditer(r" STEP \|.*?wall_time=([0-9.]+)", tail):
        t = float(m.group(1))
        if lo <= t <= now: ts.append(t)
n = len(ts)
print("  STEP lines in window: %d  ->  %.1f env-steps/s" % (n, n / 600.0))
if n == 0: print("  !! zero steps in the last 10 min")
PY

  echo "[main.log] last exploration / evaluation lines"
  grep -E "evaluation :|exploration: steps" "$R/main.log" 2>/dev/null | tail -2 | cut -c1-170 | sed 's/^/  /'
  echo "[eval table] $(basename "$R").csv — last 3 rows (episodes;steps;quality;success;...)"
  [ -f "$R.csv" ] && tail -3 "$R.csv" | cut -d';' -f1-8 | sed 's/^/  /' || echo "  (no eval rows yet)"

  echo "[checkpoints] on disk vs tracked in git"
  cd "$ROOT" || exit 1
  disk=$(ls "$R/checkpoints" 2>/dev/null | grep -v replay_incremental | sort -V | tr '\n' ' ')
  untracked=$(comm -13 <(git ls-files "$R/checkpoints" | xargs -n1 basename 2>/dev/null | sort) <(ls "$R/checkpoints" 2>/dev/null | grep -v replay_incremental | sort) | tr '\n' ' ')
  echo "  on disk: ${disk:-none}"
  echo "  untracked (force-add pending, models only — never replay_incremental): ${untracked:-none}"
fi

echo "[container log] failure signatures since the last 2 h (SOFA 'should never happen' noise excluded)"
docker logs --since 2h "$C" 2>&1 | grep -E "Traceback|WATCHDOG|CUDA error|CUDA out of memory|OutOfMemory|out of memory|Killed|RuntimeError|Setting time limit" | grep -v "should never happen" | tail -5 | cut -c1-170 | sed 's/^/  /'
tb=$(docker logs --since 2h "$C" 2>&1 | grep -c "^Traceback\|WATCHDOG: stall")
[ "${tb:-0}" -gt 0 ] && { echo "  !! $tb traceback/watchdog lines in the last 2 h"; BAD=1; }

echo "===== RESULT: $([ $BAD -eq 0 ] && echo HEALTHY || echo 'ATTENTION NEEDED (see !! lines)') ====="
exit $BAD
