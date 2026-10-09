#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\dr.sh (= /sp/v6/wall/dr.sh in the build containers)
# v6_wall: run a command in a container. /repo READ-ONLY; scratch at /sp; writable large outputs at /v6w
# (= D:\Arjun\neve\carotid_data\v6_work\wall, a separate non-nested mount).
#   bash dr.sh <image> <name|-> <cmd...>   name '-' = foreground --rm; else detached --rm (name must start v6wall_)
# Limits: --cpus 4 --memory 6g. Never used for containers this agent did not start.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IMG=$1; NAME=$2; shift 2
UOPT=(); [ "${U0:-0}" = 1 ] && UOPT=(-u 0)   # U0=1: run as root (neve-build-vmtk, as v5 ran it)
MNT=(-v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" -v "D:\Arjun\neve\carotid_data\v6_work\wall:/v6w"
     -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro")
if [ "$NAME" = "-" ]; then
  docker run --rm "${UOPT[@]}" --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 "${MNT[@]}" $IMG "$@"
else
  case "$NAME" in v6wall_*) ;; *) echo "refusing: container name must start v6wall_"; exit 2;; esac
  docker run -d --rm "${UOPT[@]}" --name $NAME --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 "${MNT[@]}" $IMG "$@"
fi
