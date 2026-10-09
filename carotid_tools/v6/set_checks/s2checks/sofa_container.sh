#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\sofa_container.sh (= /sp/v6/s2checks/sofa_container.sh in the build containers)
# s2_checks: one container hosting slots <a> .. <b> of the SOFA worker (background processes; the container exits when all end).
for k in $(seq $1 $2); do bash /sp/v6/s2checks/sofa_worker.sh $k & sleep 20; done
wait
echo "container slots $1-$2 done $(date '+%F %T')" >> /s2c/sofa/logs/worker.log
