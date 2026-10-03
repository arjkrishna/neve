#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\dr.sh (= /sp/v5_va/checks/dr.sh in the build containers)
# v5_checks: run a command in a container with the v4/v5 mounts; /repo READ-ONLY (checks never write the repo).
#   bash dr.sh <image> <name|-> <cmd...>      name '-' = foreground --rm without a name
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IMG=$1; NAME=$2; shift 2
if [ "$NAME" = "-" ]; then
  docker run --rm -e PYTHONUNBUFFERED=1 -v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" \
    -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro" \
    $IMG "$@"
else
  docker run -d --rm --name $NAME ${DOCKER_EXTRA} -e PYTHONUNBUFFERED=1 -v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" \
    -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro" \
    $IMG "$@"
fi
