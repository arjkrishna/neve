#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\dr_s3.sh (= /sp/v6/s3fix/dr_s3.sh in the build containers)
# s3_fix runner. /repo RO, /sp rw (scratchpad), /w = carotid_data/v6_work RO, /full = carotid_data/v6_work/full RO,
# /s3 = carotid_data/v6_work/s3fix rw.
#   bash dr_s3.sh <image> <name|-> <cmd...>  ('-' foreground --rm; else detached --rm, name must start s3f_). --cpus 4 --memory 6g.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IMG=$1; NAME=$2; shift 2
MNT=(-v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" -v "D:\Arjun\neve\carotid_data\v6_work:/w:ro"
     -v "D:\Arjun\neve\carotid_data\v6_work\full:/full:ro" -v "D:\Arjun\neve\carotid_data\v6_work\build:/bld:ro"
     -v "D:\Arjun\neve\carotid_data\v6_work\s3fix:/s3"
     -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro")
if [ "$NAME" = "-" ]; then
  docker run --rm --cpus ${DR_CPUS:-4} --memory ${DR_MEM:-6g} -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp ${DR_EXTRA:-} "${MNT[@]}" $IMG "$@"
else
  case "$NAME" in s3f_*) ;; *) echo "refusing: name must start s3f_"; exit 2;; esac
  docker run -d --rm --name $NAME --cpus ${DR_CPUS:-4} --memory ${DR_MEM:-6g} -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp ${DR_EXTRA:-} "${MNT[@]}" $IMG "$@"
fi
