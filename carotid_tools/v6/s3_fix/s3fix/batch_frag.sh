#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\batch_frag.sh (= /sp/v6/s3fix/batch_frag.sh in the build containers)
# idempotent: diag_frag on names in $1 (claims by mkdir), outputs /s3/diag/frag_<name>.json
mkdir -p /s3/diag/_claim
for n in $(tr -d '\r' < $1); do
  [ -f /s3/diag/frag_$n.json ] && continue
  mkdir /s3/diag/_claim/$n 2>/dev/null || continue
  python3 /sp/v6/s3fix/diag_frag.py $n /s3/diag/frag_$n.json > /s3/diag/frag_$n.log 2>&1
  echo "$(date +%T) $n rc $?" >> /s3/diag/batch_frag.log
done
