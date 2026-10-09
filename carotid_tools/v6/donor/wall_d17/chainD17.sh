#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\chainD17.sh (= /sp/v6/wall_d17/chainD17.sh in the build containers)
# s2_harden_wall_bake D16 + D17 chain (host). Step 1 runs TWO containers (v5 rule, plateau rule) in parallel, the rest
# one at a time. Markers v6_work/wall/d17/chain/<step>.done; log chain.log; stop: touch chain/STOP. Idempotent.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
C=/d/Arjun/neve/carotid_data/v6_work/wall/d17/chain; mkdir -p $C
log() { echo "$(date '+%F %T') $*" >> $C/chain.log; }
step() {
  local name=$1 img=$2 u0=$3; shift 3
  [ -f $C/$name.done ] && return 0
  [ -f $C/STOP ] && { log "STOP"; exit 0; }
  log "$name start ($img)"
  DR_NAME=s2wb_d17_$name U0=$u0 DR_EXTRA="${XENV:-}" bash $SP/v6/bake/dr.sh $img - bash -c "$* > /v6w/d17/chain/$name.log 2>&1; echo rc \$? > /v6w/d17/chain/$name.exit"
  if grep -q "rc 0" $C/$name.exit 2>/dev/null; then touch $C/$name.done; log "$name done"; else log "$name FAILED"; return 1; fi
}
if [ ! -f $C/runs.done ]; then
  log "runs start (2 containers)"
  ( step run_v5 neve-build-stagea 0 "python3 /sp/v6/wall_d17/sub_batch_d17.py v5 all /v6w/d17/v5rule" ) &
  ( step run_pl neve-build-stagea 0 "python3 /sp/v6/wall_d17/sub_batch_d17.py plateau all /v6w/d17/plateau" ) &
  wait
  if [ -f $C/run_v5.done ] && [ -f $C/run_pl.done ]; then touch $C/runs.done; log "runs done"; else log "runs FAILED"; exit 1; fi
fi
XENV="-e LOCAL_SUB_PL=/v6w/d17/plateau -e LOCAL_SUB_V5R=/v6w/d17/v5rule -e LOCAL_SUB_OUT=/v6w/d17/local" step local neve-build-stagea 0 "python3 /sp/v6/wall_local/local_sub.py all" || exit 1
step vmtk_local neve-build-vmtk 1 "bash /sp/v6/wall/jfix/vmtk_sub3.sh /v6w/d17/local" || exit 1
step vmtk_v5rule neve-build-vmtk 1 "bash /sp/v6/wall/jfix/vmtk_sub3.sh /v6w/d17/v5rule"
step records neve-build-stagea 0 "python3 /sp/v6/wall_d17/junction_records_d17.py" || exit 1
step wd neve-build-stagea 0 "python3 /sp/v6/wall_local/wd_sub.py --all d17_local /v6w/d17/local && python3 /sp/v6/wall_local/wd_sub.py --all d17_v5rule /v6w/d17/v5rule"
log "chain D17 finished"
