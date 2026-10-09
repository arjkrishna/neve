#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\sofa_worker.sh (= /sp/v6/s2checks/sofa_worker.sh in the build containers)
# s2_checks SOFA worker (eve-training-fixed; /repo read-only). Runs the FROZEN tools of TOOLCHAIN_s2build.md5 (d65b7209):
#   rcca -> /sp/v6/sofa/sofa_rcca_v6.py 1.0 (RCCA training route; soundness verdict + reach), seeds 777,778,779, 300 steps
#   d14  -> /sp/v6/va/sofa_va_entry.py 1.1 (D14 VA entry, device reset per seed), seeds 777,778,779, 300 steps
# Items "<kind>:<tag>:<name>" from /s2c/sofa/items.txt; tag -> tree: v6 anatomies_v6, v5 anatomies_v5, v4 anatomies_v4.
# Output /s2c/sofa/<kind>_<tag>/<name>.json (done = exists). Claim by atomic mkdir /s2c/sofa/claim/<kind>_<tag>.<name>
# (owner = slot). A relaunched slot first drops its OWN stale claims. Slot <k> only claims while k < cap (/s2c/sofa/cap),
# so lowering cap retires slots after their current item (no container is killed).
#   bash sofa_worker.sh <slot>      (one process per slot; a container may host several slots)
K=$1; R=/s2c/sofa; L=$R/items.txt
mkdir -p $R/claim $R/logs/items
export PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date '+%F %T') [s$K] $*" >> $R/logs/worker.log; }
if ! grep -v '^#' /sp/v6/build/TOOLCHAIN_s2build.md5 | md5sum -c --quiet; then log "REFUSED: toolchain != TOOLCHAIN_s2build.md5"; exit 2; fi
log "slot start (toolchain ok $(md5sum /sp/v6/build/TOOLCHAIN_s2build.md5 | cut -c1-8))"
for c in $R/claim/*/; do
  [ -f ${c}owner ] || continue
  b=$(basename $c); kt=${b%%.*}; n=${b#*.}
  if [ "$(cat ${c}owner)" = "$K" ] && [ ! -f $R/$kt/$n.json ]; then log "dropping own stale claim $b"; rm -rf $c; fi
done
for IT in $(tr -d '\r' < $L); do
  CAP=$(cat $R/cap 2>/dev/null || echo 0)
  [ "$K" -lt "$CAP" ] || { log "slot retired (cap $CAP)"; break; }
  KIND=${IT%%:*}; REST=${IT#*:}; T=${REST%%:*}; N=${REST#*:}
  case "$T" in v6) ROOT=/repo/carotid_data/anatomies_v6;; v5) ROOT=/repo/carotid_data/anatomies_v5;; v4) ROOT=/repo/carotid_data/anatomies_v4;; *) log "bad tag $IT"; continue;; esac
  case "$KIND" in rcca) TOOL=/sp/v6/sofa/sofa_rcca_v6.py;; d14) TOOL=/sp/v6/va/sofa_va_entry.py;; *) log "bad kind $IT"; continue;; esac
  O=$R/${KIND}_$T; mkdir -p $O
  [ -f $O/$N.json ] && continue
  mkdir $R/claim/${KIND}_$T.$N 2>/dev/null || continue
  echo $K > $R/claim/${KIND}_$T.$N/owner
  t0=$(date +%s)
  log "run $KIND $T $N"
  SOFA_OUT=$O timeout 3600 python3 $TOOL $N 300 $ROOT 777,778,779 > $R/logs/items/${KIND}_$T.$N.log 2>&1
  rc=$?
  log "end $KIND $T $N rc $rc $(( $(date +%s) - t0 )) s $(grep -h '^SOFA R\|^RCCA EXERCISED\|^D14' $R/logs/items/${KIND}_$T.$N.log | tr '\n' ' ' | cut -c1-260)"
  [ -f $O/$N.json ] || { log "NO OUTPUT $KIND $T $N (claim kept: see item log)"; }
done
log "slot done"
