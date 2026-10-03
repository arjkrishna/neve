#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\worker.sh (= /sp/v5_va/bake_full/worker.sh in the build containers)
# v5_bake_full worker (inside neve-build-meshlab). Walks the ordered list (hardest first), claims an anatomy by an
# atomic mkdir of <BF>/_claim/<name>, then bake (bake_full_one.py) and measure (measure_full_one.py), each in its own
# python process (memory freed between anatomies), each skipped if its done-marker exists. Idempotent; logs on disk.
#   bash worker.sh <worker id>      env BF_DIR (default /repo/carotid_data/va_sources/v5_reg/bake_full), ORDER, A5_DIR
W=$1
BF=${BF_DIR:-/repo/carotid_data/va_sources/v5_reg/bake_full}
D=/sp/v5_va/bake_full
mkdir -p $BF/_claim $BF/_done $BF/_logs
log() { echo "$(date '+%F %T') [w$W] $*" >> $BF/_logs/worker_$W.log; }
log "start"
for N in $(tr -d '\r' < ${ORDER:-$D/order.txt}); do
  [ -f $BF/_done/$N.measure.json ] && continue
  [ -f $BF/_done/$N.bake_failed.json ] && continue
  mkdir $BF/_claim/$N 2>/dev/null || continue
  echo "w$W $(date '+%F %T')" > $BF/_claim/$N/by
  if [ ! -f $BF/_done/$N.bake.json ]; then
    log "bake $N"
    python3 $D/bake_full_one.py $N >> $BF/_logs/$N.bake.log 2>&1
    log "bake $N exit $?"
  fi
  if [ -f $BF/_done/$N.bake.json ] && [ ! -f $BF/_done/$N.measure.json ]; then
    log "measure $N"
    python3 $D/measure_full_one.py $N >> $BF/_logs/$N.measure.log 2>&1
    log "measure $N exit $?"
  fi
done
log "WORKER DONE"
