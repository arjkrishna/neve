#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\status.sh (= /sp/v5_va/bake_full/status.sh in the build containers)
# v5_bake_full: one status line set (host).   bash status.sh
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
BFH=/d/Arjun/neve/carotid_data/va_sources/v5_reg/bake_full
date '+%F %T'
echo "workers live: $(docker ps --format '{{.Names}}' | grep -c '^v5bake_w')  claims: $(ls $BFH/_claim 2>/dev/null | wc -l)"
echo "baked: $(ls $BFH/_done/*.bake.json 2>/dev/null | wc -l)  measured: $(ls $BFH/_done/*.measure.json 2>/dev/null | wc -l)  bake_failed: $(ls $BFH/_done/*.bake_failed.json 2>/dev/null | wc -l)"
echo "failed: $(ls $BFH/_done/*.bake_failed.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
echo "not all_pass: $(grep -L '"all_pass": true' $BFH/_done/*.measure.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
echo "flagged: $(grep -l '"flags": \[[^]]' $BFH/_done/*.bake.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
echo "rung2+: $(grep -L '"rung": "1 pinned' $BFH/_done/*.bake.json 2>/dev/null | wc -l)"
grep -h "WORKER DONE" $BFH/_logs/worker_*.log 2>/dev/null | wc -l | sed 's/^/workers done: /'
docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.CPUPerc}}' 2>/dev/null | grep v5bake_ | tr '\n' ';'; echo
