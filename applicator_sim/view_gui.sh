#!/usr/bin/env bash
# Optional interactive view of one run in runSofa over X11 (Git Bash on Windows).
#   ./view_gui.sh <tag>        (uses $APPSIM_OUT/runs/<tag>/cfg.json)
# Needs an X server on the host (VcXsrv or X410) with access control OFF, listening on :0.
# runSofa is NOT on PATH in the image: /opt/sofa/build/install/bin/runSofa (measured).
# Never run this while another container of ours is running (one container at a time).
# Env as run_docker.sh: APPSIM_DATA, APPSIM_OUT, APPSIM_IMAGE, APPSIM_CPUS (default 12);
# APPSIM_DISPLAY = X display as seen from the container (default host.docker.internal:0;
# use <host LAN IP>:0 if host.docker.internal does not reach the X server).
set -euo pipefail
TAG="$1"
APP_DIR=$(cd "$(dirname "$0")" && pwd)
DATA="${APPSIM_DATA:-$HOME/Downloads/MRI_GYN}"
OUT="${APPSIM_OUT:-$HOME/Downloads/MRI_GYN_sim}"
IMAGE="${APPSIM_IMAGE:-eve-training-fixed}"
CPUS="${APPSIM_CPUS:-12}"
XDISPLAY="${APPSIM_DISPLAY:-host.docker.internal:0}"
win() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }
OTHERS=$(docker ps --filter ancestor="$IMAGE" --format '{{.Names}}' | grep -v '^rcca_carotid_v3$' || true)
if [ -n "$OTHERS" ]; then echo "another container is running: $OTHERS"; exit 3; fi
MSYS_NO_PATHCONV=1 docker run --rm --name "appsim_gui_$TAG" --cpus "$CPUS" --memory 8g \
  -e DISPLAY="$XDISPLAY" -e APPSIM_CFG="/out/runs/$TAG/cfg.json" -e PYTHONPATH=/app \
  -v "$(win "$APP_DIR"):/app" \
  -v "$(win "$DATA"):/data:ro" \
  -v "$(win "$OUT"):/out" \
  "$IMAGE" /opt/sofa/build/install/bin/runSofa -l SofaPython3 /app/scene.py
