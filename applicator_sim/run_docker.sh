#!/usr/bin/env bash
# Launch ONE applicator_sim container (Git Bash on the Windows host).
#   ./run_docker.sh <tag> <script.py> [script args...]
#   ./run_docker.sh G0 gate_tests.py
#   ./run_docker.sh R1_P2_S0 run_insertion.py --tag R1_P2_S0 --cfg '{"pose":"P2"}'
#   ./run_docker.sh F_final run_insertion.py --batch /app/runs_def/final.json
# Environment (host paths; the defaults are the original machine's):
#   APPSIM_DATA  read-only image/label folder      APPSIM_OUT  derived-data root      APPSIM_IMAGE  docker image
#   APPSIM_CPUS  docker --cpus limit (default 12 = the whole Docker VM; lower it when the host is shared)
# Guard: never touches rcca_carotid_v3; waits (poll 60 s, max 30 min) while any OTHER container of the image is
# running, so at most one short-lived container of ours runs at a time.
# Limits: --cpus $APPSIM_CPUS (12) --memory 8g, `timeout 570` inside the container (each invocation < 10 min wall).
# The data directory is mounted read-only. Log: $APPSIM_OUT/logs/<tag>.log
set -uo pipefail
TAG="$1"; shift
SCRIPT="$1"; shift
APP_DIR=$(cd "$(dirname "$0")" && pwd)
DATA="${APPSIM_DATA:-$HOME/Downloads/MRI_GYN}"
OUT="${APPSIM_OUT:-$HOME/Downloads/MRI_GYN_sim}"
IMAGE="${APPSIM_IMAGE:-eve-training-fixed}"
CPUS="${APPSIM_CPUS:-12}"
win() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
for i in $(seq 1 30); do
  # ignore the training container and any live viewer (appsim_gui_*): a viewer idles unless Animate is pressed
  OTHERS=$(docker ps --filter ancestor="$IMAGE" --format '{{.Names}}' | grep -v -E '^(rcca_carotid_v3|appsim_gui_.*)$' || true)
  [ -z "$OTHERS" ] && break
  echo "[run_docker] waiting for: $OTHERS ($i/30)"; sleep 60
done
if [ -n "${OTHERS:-}" ]; then echo "[run_docker] still busy, giving up"; exit 3; fi
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo unknown)
echo "[run_docker] $(date '+%F %T') start appsim_$TAG ($IMAGE $IMAGE_ID, cpus $CPUS): $SCRIPT $*" | tee "$LOGDIR/$TAG.log"
MSYS_NO_PATHCONV=1 docker run --rm --name "appsim_$TAG" --cpus "$CPUS" --memory 8g \
  -v "$(win "$APP_DIR"):/app" \
  -v "$(win "$DATA"):/data:ro" \
  -v "$(win "$OUT"):/out" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e APPSIM_IMAGE_ID="$IMAGE_ID" -e APPSIM_CPUS="$CPUS" \
  "$IMAGE" timeout 570 python3 "/app/$SCRIPT" "$@" 2>&1 \
  | grep --line-buffered -v -E '^(==========|== CUDA ==|CUDA Version|Container image Copyright|This container image|By pulling|https://developer.nvidia|A copy of this license|WARNING: The NVIDIA Driver|   Use the NVIDIA|   https://docs.nvidia)' \
  | tee -a "$LOGDIR/$TAG.log"
RC=${PIPESTATUS[0]}
echo "[run_docker] $(date '+%F %T') exit $RC" | tee -a "$LOGDIR/$TAG.log"
exit $RC
