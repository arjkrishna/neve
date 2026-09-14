#!/usr/bin/env bash
# Interactive view of one hybrid run in runSofa over X11 (Git Bash on Windows).  Copy of applicator_sim/view_gui.sh
# pointed at hybrid/scene_hybrid.py.
#   ./view_hybrid.sh <tag>            (uses $APPSIM_OUT/hybrid/runs/<tag>/cfg.json if it exists, else scene defaults)
# Needs an X server on the host (VcXsrv or X410) with access control OFF, listening on :0.
# runSofa is NOT on PATH in the image: /opt/sofa/build/install/bin/runSofa (measured).
# Env as run_docker.sh: APPSIM_DATA, APPSIM_OUT, APPSIM_IMAGE, APPSIM_CPUS (default 12);
# APPSIM_DISPLAY = X display as seen from the container (default host.docker.internal:0;
# use <host LAN IP>:0 if host.docker.internal does not reach the X server).
# Like run_docker.sh, other appsim_gui_* viewers and the training container are ignored (a viewer idles unless
# Animate is pressed); a running COMPUTE container (appsim_<tag>) still blocks, one at a time.
set -euo pipefail
TAG="${1:-HV}"
APP_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATA="${APPSIM_DATA:-$HOME/Downloads/MRI_GYN}"
OUT="${APPSIM_OUT:-$HOME/Downloads/MRI_GYN_sim}"
IMAGE="${APPSIM_IMAGE:-eve-training-fixed}"
CPUS="${APPSIM_CPUS:-12}"
XDISPLAY="${APPSIM_DISPLAY:-host.docker.internal:0}"
win() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }
OTHERS=$(docker ps --filter ancestor="$IMAGE" --format '{{.Names}}' | grep -v -E '^(rcca_[a-z0-9_]*|appsim_gui_.*)$' || true)
if [ -n "$OTHERS" ]; then echo "another container is running: $OTHERS"; exit 3; fi
CFG="/out/hybrid/runs/$TAG/cfg.json"
if [ ! -f "$OUT/hybrid/runs/$TAG/cfg.json" ]; then
  echo "[view_hybrid] no runs/$TAG/cfg.json -- starting with the scene defaults"; CFG=""
fi
MSYS_NO_PATHCONV=1 docker run --rm --name "appsim_gui_$TAG" --cpus "$CPUS" --memory 8g \
  -e DISPLAY="$XDISPLAY" -e APPSIM_CFG="$CFG" -e PYTHONPATH=/app:/app/hybrid -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(win "$APP_DIR"):/app" \
  -v "$(win "$DATA"):/data:ro" \
  -v "$(win "$OUT"):/out" \
  "$IMAGE" /opt/sofa/build/install/bin/runSofa -l SofaPython3 /app/hybrid/scene_hybrid.py
