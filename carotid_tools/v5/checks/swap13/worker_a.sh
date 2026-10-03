#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\worker_a.sh (= /sp/v5_va/swap13/worker_a.sh in the build containers)
# v5_swap13 job worker (inside neve-build-meshlab; /repo read-only). Jobs in swap13/jobs_a.txt ("kind name"), claimed by
# atomic mkdir under swap13/claim/; a job is skipped when its output exists (idempotent). One python process per job.
#   bash worker_a.sh <id>
W=/sp/v5_va/swap13; ID=$1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p $W/claim $W/logs/one
log() { echo "$(date '+%F %T') [a$ID] $*" >> $W/logs/worker_a$ID.log; }
log start
while read -r K N; do
  K=$(echo $K | tr -d '\r'); N=$(echo $N | tr -d '\r'); [ -n "$K" ] || continue
  case $K in
    split)    O=$W/split.done ;;
    chk)      O=$W/chk_out/$N.json ;;
    vvw)      O=$W/vvw_out/$N.json ;;
    rrdx)     O=$W/rrdx/$N.json ;;
    lumen13)  O=$W/lumen13.json ;;
    valumen13) O=$W/va_lumen13.json ;;
    *) log "unknown job $K"; continue ;;
  esac
  [ -f $O ] && continue
  mkdir $W/claim/$K.$N 2>/dev/null || continue
  log "run $K $N"
  case $K in
    split)    python3 /sp/v5_va/v5_split.py /repo/carotid_data/anatomies_v5 --baseline /sp/v5_va/results/split_v4.json \
                --out /sp/v5_va/results/split_v5_final.json > $W/logs/split.log 2>&1; rc=$?; echo "EXIT $rc" >> $W/logs/split.log
              echo "exit $rc $(date '+%F %T')" > $O ;;
    chk)      python3 $W/chk13.py $N > $W/logs/one/chk.$N.log 2>&1; rc=$? ;;
    vvw)      python3 $W/vvw13.py $N > $W/logs/one/vvw.$N.log 2>&1; rc=$? ;;
    rrdx)     python3 $W/rrdx.py $N > $W/logs/one/rrdx.$N.log 2>&1; rc=$? ;;
    lumen13)  python3 $W/lumen13.py > $W/logs/lumen13.log 2>&1; rc=$? ;;
    valumen13) python3 $W/valumen13.py > $W/logs/valumen13.log 2>&1; rc=$? ;;
  esac
  log "end $K $N exit $rc"
done < $W/jobs_a.txt
log "WORKER DONE"
