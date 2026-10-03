#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\rerun_v1.sh (= /sp/v5_va/swap13/rerun_v1.sh in the build containers)
# v5_swap13: re-run rrdx.py --force on every finished output written by version 1 (no "rrdx_version" key), one at a
# time, so every record carries the tolerant match too. Idempotent (a re-run output has the key). bash rerun_v1.sh
W=/sp/v5_va/swap13
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for f in $W/rrdx/*.json; do
  grep -q rrdx_version $f && continue
  N=$(basename $f .json)
  echo "$(date '+%F %T') rerun $N" >> $W/logs/rerun_v1.log
  python3 $W/rrdx.py $N --force > $W/logs/one/rrdx2.$N.log 2>&1
  echo "$(date '+%F %T') end $N exit $?" >> $W/logs/rerun_v1.log
done
echo "$(date '+%F %T') RERUN DONE" >> $W/logs/rerun_v1.log
