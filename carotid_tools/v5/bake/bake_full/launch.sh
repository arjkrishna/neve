#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\launch.sh (= /sp/v5_va/bake_full/launch.sh in the build containers)
# v5_bake_full launcher (host, Git Bash). Starts NW detached worker containers v5bake_w<k> (docker run -d --rm).
# If no v5bake_ worker is live, stale claims (claimed, no done-marker) are cleared first. Rerunnable (idempotent):
# a live worker is never started twice.      bash launch.sh [NW=8]
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
NW=${1:-8}
BFH=/d/Arjun/neve/carotid_data/va_sources/v5_reg/bake_full
mkdir -p $BFH/_claim $BFH/_done $BFH/_logs
if [ "$(docker ps --format '{{.Names}}' | grep -c '^v5bake_w')" -eq 0 ]; then
  for c in $BFH/_claim/*/; do
    [ -d "$c" ] || continue
    N=$(basename $c)
    if [ ! -f $BFH/_done/$N.measure.json ] && [ ! -f $BFH/_done/$N.bake_failed.json ]; then
      rm -rf "$c"; echo "$(date '+%F %T') cleared stale claim $N" >> $BFH/_logs/launch.log
    fi
  done
fi
for k in $(seq 0 $((NW-1))); do
  C=v5bake_w$k
  if docker ps --format '{{.Names}}' | grep -qx "$C"; then echo "live $C"; continue; fi
  docker run -d --rm --name $C --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 \
    -v "D:\Arjun\neve:/repo" -w /repo -v "$SPW:/sp" \
    -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools" \
    neve-build-meshlab bash /sp/v5_va/bake_full/worker.sh $k > /dev/null \
    && echo "$(date '+%F %T') started $C" | tee -a $BFH/_logs/launch.log || echo "START FAILED $C"
  sleep 3
done
