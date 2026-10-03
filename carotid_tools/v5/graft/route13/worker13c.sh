#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\worker13c.sh (= /sp/v5_va/route13/worker13c.sh in the build containers)
# v5_route13 worker with CLAIMS (several workers on one list): claim <V>_bf/_claim/<name> by atomic mkdir, then bake +
# measure (bake_full_one / measure_full_one via run_bake.py; done-markers <V>_bf/_done/<name>.{bake,measure}.json).
# A name already claimed, or done, is skipped. Waits for the graft record when TAG is set (moved variants).
#   VARIANT=moved_ext TAG=ext OWNERSHIP=standard bash worker13c.sh <list> <worker id>
R=/repo/carotid_data/va_sources/v5_reg/route13
export A5_DIR=$R/$VARIANT BF_DIR=$R/${VARIANT}_bf ISO_DIR=$R/${VARIANT}_iso
mkdir -p $BF_DIR/_done $BF_DIR/_logs $BF_DIR/_claim $ISO_DIR
log() { echo "$(date '+%F %T') [$VARIANT w$2] $1" >> $BF_DIR/_logs/worker.log; }
for N in $(tr -d '\r' < $1); do
  [ -f $BF_DIR/_done/$N.measure.json ] && continue
  if [ -n "$TAG" ]; then
    t=0; until [ -f /sp/v5_va/route13/graft/$TAG/$N.json ] || [ $t -ge 10800 ]; do sleep 30; t=$((t+30)); done
  fi
  [ -d $A5_DIR/$N/Centrelines_comb ] || { log "skip $N (no anatomy)" $2; continue; }
  mkdir $BF_DIR/_claim/$N 2>/dev/null || continue
  if [ ! -f $BF_DIR/_done/$N.bake.json ]; then
    log "bake $N ($OWNERSHIP)" $2
    python3 /sp/v5_va/route13/run_bake.py $N >> $BF_DIR/_logs/$N.bake.log 2>&1
    log "bake $N exit $?" $2
  fi
  if [ -f $BF_DIR/_done/$N.bake.json ] && [ ! -f $BF_DIR/_done/$N.measure.json ]; then
    log "measure $N" $2
    python3 /sp/v5_va/bake_full/measure_full_one.py $N >> $BF_DIR/_logs/$N.measure.log 2>&1
    log "measure $N exit $?" $2
  fi
done
log "WORKER DONE $1" $2
