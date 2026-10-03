#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\worker13.sh (= /sp/v5_va/route13/worker13.sh in the build containers)
# v5_route13 worker (inside neve-build-meshlab): bake + measure each name of a list in one VARIANT folder of
# /repo/carotid_data/va_sources/v5_reg/route13 (moved | prio | ...). Idempotent: done-markers <V>_bf/_done/<name>.{bake,measure}.json.
#   VARIANT=moved OWNERSHIP=standard bash worker13.sh <list file>
R=/repo/carotid_data/va_sources/v5_reg/route13
export A5_DIR=$R/$VARIANT BF_DIR=$R/${VARIANT}_bf ISO_DIR=$R/${VARIANT}_iso
mkdir -p $BF_DIR/_done $BF_DIR/_logs $ISO_DIR
log() { echo "$(date '+%F %T') [$VARIANT] $*" >> $BF_DIR/_logs/worker.log; }
for N in $(tr -d '\r' < $1); do
  [ -d $A5_DIR/$N/Centrelines_comb ] || { log "skip $N (no anatomy)"; continue; }
  if [ ! -f $BF_DIR/_done/$N.bake.json ]; then
    log "bake $N ($OWNERSHIP)"
    python3 /sp/v5_va/route13/run_bake.py $N >> $BF_DIR/_logs/$N.bake.log 2>&1
    log "bake $N exit $?"
  fi
  if [ -f $BF_DIR/_done/$N.bake.json ] && [ ! -f $BF_DIR/_done/$N.measure.json ]; then
    log "measure $N"
    python3 /sp/v5_va/bake_full/measure_full_one.py $N >> $BF_DIR/_logs/$N.measure.log 2>&1
    log "measure $N exit $?"
  fi
done
log "WORKER DONE $1"
