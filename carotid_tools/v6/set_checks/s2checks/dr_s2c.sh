#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\dr_s2c.sh (= /sp/v6/s2checks/dr_s2c.sh in the build containers)
# s2_checks: run a command in a container. /repo READ-ONLY; /sp = scratchpad (rw); /full = carotid_data/v6_work/full
# (READ-ONLY: the build's records, tree3, iso); /s2c = carotid_data/v6_work/s2checks (rw: this agent's outputs).
#   bash dr_s2c.sh <image> <name|-> <cmd...>   name '-' = foreground --rm; else detached --rm (name must start s2c_)
# Limits: --cpus 4 --memory 6g. Never used for containers this agent did not start.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IMG=$1; NAME=$2; shift 2
MNT=(-v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" -v "D:\Arjun\neve\carotid_data\v6_work\full:/full:ro"
     -v "D:\Arjun\neve\carotid_data\v6_work\s2checks:/s2c"
     -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro")
if [ "$NAME" = "-" ]; then
  docker run --rm --cpus ${DR_CPUS:-4} --memory ${DR_MEM:-6g} -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp ${DR_EXTRA:-} "${MNT[@]}" $IMG "$@"
else
  case "$NAME" in s2c_*) ;; *) echo "refusing: container name must start s2c_"; exit 2;; esac
  docker run -d --rm --name $NAME --cpus ${DR_CPUS:-4} --memory ${DR_MEM:-6g} -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp ${DR_EXTRA:-} "${MNT[@]}" $IMG "$@"
fi
