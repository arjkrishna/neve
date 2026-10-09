#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records\dr_rec.sh (= /sp/v6/records/dr_rec.sh in the build containers)
# s2_records: run a command in a foreground --rm container. /repo READ-ONLY; /sp = scratchpad (rw);
# /full = carotid_data/v6_work/full (ro); /s2c = carotid_data/v6_work/s2checks (ro).
#   bash dr_rec.sh <image> <cmd...>      (container name s2r_<pid>; --cpus 4 --memory 6g)
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IMG=$1; shift 1
docker run --rm --name s2r_$$ --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" \
  -v "D:\Arjun\neve\carotid_data\v6_work\full:/full:ro" -v "D:\Arjun\neve\carotid_data\v6_work\s2checks:/s2c:ro" \
  $IMG "$@"
