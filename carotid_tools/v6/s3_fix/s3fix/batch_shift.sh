#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\batch_shift.sh (= /sp/v6/s3fix/batch_shift.sh in the build containers)
# idempotent shift screen: diag_shift on names in $1 (claims by mkdir) -> /s3/shift/<name>.json (per-shift resume inside)
mkdir -p /s3/shift/_claim
SH="0,0,0 0.225,0,0 0,0.225,0 0,0,0.225 0.225,0.225,0 0.225,0,0.225 0,0.225,0.225 0.225,0.225,0.225 0.15,0.15,0.15 0.3,0.3,0.3"
for n in $(tr -d '\r' < $1); do
  [ -f /s3/shift/_done_$n ] && continue
  mkdir /s3/shift/_claim/$n 2>/dev/null || continue
  python3 /sp/v6/s3fix/diag_shift.py $n /s3/shift/$n.json $SH >> /s3/shift/$n.log 2>&1 && touch /s3/shift/_done_$n
  echo "$(date +%T) $n rc $?" >> /s3/shift/batch_shift.log
done
