#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\sofa_worker.sh (= /sp/v5_va/checks/sofa_worker.sh in the build containers)
# v5_checks SOFA worker (eve-training-fixed, /repo read-only). Items "v5:<name>" (anatomies_v5 -> sofa/run4/v5) and
# "v4:<name>" (control: same schedule on anatomies_v4 -> sofa/run4/v4). Claim = atomic mkdir under sofa/run4/claim;
# skip if the output exists; one python process per item (sofa_rva.py hard-exits after its PASSED line).
#   bash sofa_worker.sh <id>
W=$1; C=/sp/v5_va/checks; S=$C/sofa; R=$S/run4
mkdir -p $R/claim $R/v5 $R/v4 $R/logs
export PYTHONDONTWRITEBYTECODE=1
cd $S
log() { echo "$(date '+%F %T') [s$W] $*" >> $R/logs/worker_$W.log; }
log start
for IT in $(tr -d '\r' < $S/list.txt); do
  V=${IT%%:*}; N=${IT#*:}
  O=$R/$V/$N.json
  [ -f $O ] && continue
  mkdir $R/claim/$V.$N 2>/dev/null || continue
  log "run $V $N"
  if [ "$V" = "v5" ]; then AD=/repo/carotid_data/anatomies_v5; else AD=/repo/carotid_data/anatomies_v4; fi
  SOFA_OUT=$R/$V timeout 1800 python3 $C/sofa_rva.py $N 300 $AD > $R/logs/$V.$N.log 2>&1
  log "run $V $N exit $? $(grep -h 'SOFA RVA\|RVA EXERCISED' $R/logs/$V.$N.log | tr '\n' ' ')"
done
log "WORKER DONE"
