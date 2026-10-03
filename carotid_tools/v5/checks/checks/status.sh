#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\status.sh (= /sp/v5_va/checks/status.sh in the build containers)
# v5_checks: one status block (host).
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
C=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/checks
date '+%F %T'
echo "workers live: $(docker ps --format '{{.Names}}' | grep -c '^v5chk_w')  claims: $(ls $C/claim 2>/dev/null | wc -l)  done: $(ls $C/out/*.json 2>/dev/null | grep -vc error)  errors: $(ls $C/out/*.error.json 2>/dev/null | wc -l)"
echo "strict_pass false: $(grep -l '"strict_pass": false' $C/out/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
grep -h "WORKER DONE" $C/logs/worker_*.log 2>/dev/null | wc -l | sed 's/^/workers done: /'
docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.CPUPerc}}' 2>/dev/null | tr '\n' ';'; echo
