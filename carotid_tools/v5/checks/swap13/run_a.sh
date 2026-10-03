#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\run_a.sh (= /sp/v5_va/swap13/run_a.sh in the build containers)
# v5_swap13 container A entry: NW workers over jobs_a.txt, then exit.   bash run_a.sh [NW=5]
NW=${1:-5}
for k in $(seq 0 $((NW-1))); do bash /sp/v5_va/swap13/worker_a.sh $k & sleep 3; done
wait
echo "$(date '+%F %T') ALL WORKERS DONE" >> /sp/v5_va/swap13/logs/run_a.log
