#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\sofa_worker.sh (= /sp/v5_va/swap13/sofa_worker.sh in the build containers)
# v5_swap13 SOFA worker (eve-training-fixed, /repo read-only): checks/sofa_rva.py UNCHANGED (final run4 schedule:
# training loader + insertion, vessel-frame checks, closed-loop schedule into the RVA, up to 3 x 300 steps), items
# "v5:<name>" (anatomies_v5 -> swap13/sofa/v5) and "v4:<name>" (control -> swap13/sofa/v4). Skip if the output exists;
# one python process per item (sofa_rva.py hard-exits after its PASSED line). Then the training loader over all 223
# (checks/loader_v5.py with only its output path changed -> swap13/loader_v5.json).
#   bash sofa_worker.sh
W=/sp/v5_va/swap13; R=$W/sofa
mkdir -p $R/v5 $R/v4 $R/logs
export PYTHONDONTWRITEBYTECODE=1
cd $R
log() { echo "$(date '+%F %T') [s] $*" >> $R/logs/worker.log; }
log start
for IT in $(tr -d '\r' < $W/sofa_list.txt); do
  V=${IT%%:*}; N=${IT#*:}
  O=$R/$V/$N.json
  [ -f $O ] && continue
  log "run $V $N"
  if [ "$V" = "v5" ]; then AD=/repo/carotid_data/anatomies_v5; else AD=/repo/carotid_data/anatomies_v4; fi
  SOFA_OUT=$R/$V timeout 1800 python3 /sp/v5_va/checks/sofa_rva.py $N 300 $AD > $R/logs/$V.$N.log 2>&1
  log "run $V $N exit $? $(grep -h 'SOFA RVA\|RVA EXERCISED' $R/logs/$V.$N.log | tr '\n' ' ')"
  if [ "$V" = "v5" ] && [ ! -f $W/loader_v5.json ] && [ "$(ls $R/v5/*.json 2>/dev/null | wc -l)" -ge 13 ]; then
    log "loader start"; python3 $W/loader_v5.py > $W/logs/loader_v5.log 2>&1; log "loader exit $?"
  fi
done
if [ ! -f $W/loader_v5.json ]; then log "loader start"; python3 $W/loader_v5.py > $W/logs/loader_v5.log 2>&1; log "loader exit $?"; fi
log "WORKER DONE"
