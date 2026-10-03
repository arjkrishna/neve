#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\checks\launch.sh (= /sp/v5_va/checks/launch.sh in the build containers)
# v5_checks launcher (host). Starts NW detached workers v5chk_w<k>; idempotent (a live worker is not restarted;
# stale claims without output are cleared when no worker is live).   bash launch.sh [NW=8]
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
C=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/checks
NW=${1:-8}
mkdir -p $C/claim $C/out $C/logs
if [ "$(docker ps --format '{{.Names}}' | grep -c '^v5chk_w')" -eq 0 ]; then
  for c in $C/claim/*/; do [ -d "$c" ] || continue; N=$(basename $c); [ -f $C/out/$N.json ] || { rm -rf "$c"; echo "$(date '+%F %T') cleared stale claim $N" >> $C/logs/launch.log; }; done
fi
for k in $(seq 0 $((NW-1))); do
  W=v5chk_w$k
  if docker ps --format '{{.Names}}' | grep -qx "$W"; then echo "live $W"; continue; fi
  DOCKER_EXTRA="--cpus 2 --memory 4g" bash $C/dr.sh neve-build-meshlab $W bash /sp/v5_va/checks/worker.sh $k > /dev/null \
    && echo "$(date '+%F %T') started $W" | tee -a $C/logs/launch.log || echo "START FAILED $W"
  sleep 2
done
