#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_run.sh (= /sp/v5_va/vmr_run.sh in the build containers)
# usage: vmr_run.sh <image> <script> [args...]   (Git Bash)
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SP="C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad"
IMG=$1; shift; SCR=$1; shift
PY=python3
docker run --rm -e PYTHONUNBUFFERED=1 $(env | grep "^VMR_" | sed "s/^/-e /") -v "D:\Arjun\neve:/repo" -w /repo -v "$SP:/sp" -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools" $IMG $PY /sp/v5_va/$SCR "$@" 2>&1 | grep -vE "CUDA|NVIDIA|nvidia|Container image|license|^$|====|Driver|Toolkit|By pulling|governed"
