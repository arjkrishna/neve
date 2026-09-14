#!/usr/bin/env bash
# Launch ONE applicator_sim container WITHOUT waiting for other containers (parallel variant of run_docker.sh).
#   ./run_docker_par.sh <tag> <script.py> [script args...]
# Use when the host docker is free and you want several runs at once (e.g. a parameter sweep). A SOFA run here is
# effectively single-threaded (SparseLDL / CG), so N concurrent containers need ~N cores: keep N <= 4 on this
# 12-core host and give each 3 CPUs, or the containers slow each other down.
#   for d in 2 3.5 5 7; do ./run_docker_par.sh V_lr$d hybrid/run_hybrid.py --tag V_lr$d --cfg "{\"lumen_r0_mm\":$d}" & done; wait
# Environment: APPSIM_DATA, APPSIM_OUT, APPSIM_IMAGE as run_docker.sh; APPSIM_CPUS (default 3 here, not 12);
# APPSIM_TIMEOUT (default 570 s, the in-container cap).
# Same guarantees as run_docker.sh otherwise: data mounted read-only, log tee'd to $APPSIM_OUT/logs/<tag>.log,
# container removed on exit. It does NOT wait for, and never touches, any other container.
set -uo pipefail
TAG="$1"; shift
SCRIPT="$1"; shift
APP_DIR=$(cd "$(dirname "$0")" && pwd)
DATA="${APPSIM_DATA:-$HOME/Downloads/MRI_GYN}"
OUT="${APPSIM_OUT:-$HOME/Downloads/MRI_GYN_sim}"
IMAGE="${APPSIM_IMAGE:-eve-training-fixed}"
CPUS="${APPSIM_CPUS:-3}"
TMO="${APPSIM_TIMEOUT:-570}"
win() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo unknown)
echo "[run_docker_par] $(date '+%F %T') start appsim_$TAG ($IMAGE $IMAGE_ID, cpus $CPUS, timeout ${TMO}s): $SCRIPT $*" | tee "$LOGDIR/$TAG.log"
MSYS_NO_PATHCONV=1 docker run --rm --name "appsim_$TAG" --cpus "$CPUS" --memory 8g \
  -v "$(win "$APP_DIR"):/app" \
  -v "$(win "$DATA"):/data:ro" \
  -v "$(win "$OUT"):/out" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e APPSIM_IMAGE_ID="$IMAGE_ID" -e APPSIM_CPUS="$CPUS" \
  "$IMAGE" timeout "$TMO" python3 "/app/$SCRIPT" "$@" 2>&1 \
  | grep --line-buffered -v -E '^(==========|== CUDA ==|CUDA Version|Container image Copyright|This container image|By pulling|https://developer.nvidia|A copy of this license|WARNING: The NVIDIA Driver|   Use the NVIDIA|   https://docs.nvidia)' \
  | tee -a "$LOGDIR/$TAG.log"
RC=${PIPESTATUS[0]}
echo "[run_docker_par] $(date '+%F %T') exit $RC" | tee -a "$LOGDIR/$TAG.log"
exit $RC
