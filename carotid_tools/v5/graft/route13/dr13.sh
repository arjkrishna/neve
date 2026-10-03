#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\dr13.sh (= /sp/v5_va/route13/dr13.sh in the build containers)
# v5_route13: run a command in a container. /repo READ-ONLY except carotid_data/va_sources/v5_reg/route13 (rw scratch).
#   bash dr13.sh <image> <name|-> <cmd...>    name '-' = foreground --rm; else detached --rm (name must start v5r13_)
# Limits: --cpus 4 --memory 6g.  Never used for containers this agent did not start.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IMG=$1; NAME=$2; shift 2
MNT=(-v "D:\Arjun\neve:/repo:ro" -v "D:\Arjun\neve\carotid_data\va_sources\v5_reg\route13:/repo/carotid_data/va_sources/v5_reg/route13"
     -w /repo -v "$SPW:/sp"
     -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro")
if [ "$NAME" = "-" ]; then
  docker run --rm --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 "${MNT[@]}" $IMG "$@"
else
  case "$NAME" in v5r13_*) ;; *) echo "refusing: container name must start v5r13_"; exit 2;; esac
  docker run -d --rm --name $NAME --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 "${MNT[@]}" $IMG "$@"
fi
