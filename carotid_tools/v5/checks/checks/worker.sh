#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\worker.sh (= /sp/v5_va/checks/worker.sh in the build containers)
# v5_checks worker (inside neve-build-meshlab, /repo read-only). Claims an anatomy by atomic mkdir of claim/<name>,
# runs chk_one.py in its own process (memory freed per anatomy); skips anatomies with out/<name>.json. Logs on disk.
#   bash worker.sh <id>
W=$1; C=/sp/v5_va/checks
mkdir -p $C/claim $C/out $C/logs/one
export PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date '+%F %T') [w$W] $*" >> $C/logs/worker_$W.log; }
log start
for N in $(tr -d '\r' < $C/names.txt); do
  [ -f $C/out/$N.json ] && continue
  mkdir $C/claim/$N 2>/dev/null || continue
  log "chk $N"
  python3 $C/chk_one.py $N >> $C/logs/one/$N.log 2>&1
  log "chk $N exit $?"
done
log "WORKER DONE"
