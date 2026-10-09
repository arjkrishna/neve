#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\chainB.sh (= /sp/v6/wall_local/chainB.sh in the build containers)
# s2_harden_wall_bake D15 chain B (host; ONE container at a time): the local subclavian / junction walls.
# Needs chain A's seg step (zones.json). Markers v6_work/wall/local/chain/B_<step>.done; log chainB.log; stop: STOP_B.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
C=/d/Arjun/neve/carotid_data/v6_work/wall/local/chain; mkdir -p $C
log() { echo "$(date '+%F %T') $*" >> $C/chainB.log; }
step() {
  local name=B_$1 img=$2 u0=$3; shift 3
  [ -f $C/$name.done ] && return 0
  [ -f $C/STOP_B ] && { log "STOP"; exit 0; }
  log "$name start ($img)"
  U0=$u0 bash $SP/v6/bake/dr.sh $img - bash -c "$* > /v6w/local/chain/$name.log 2>&1; echo rc \$? > /v6w/local/chain/$name.exit"
  if grep -q "rc 0" $C/$name.exit 2>/dev/null; then touch $C/$name.done; log "$name done"; else log "$name FAILED"; return 1; fi
}
step sub neve-build-stagea 0 "python3 /sp/v6/wall_local/local_sub.py all" || exit 1
step vmtk_local neve-build-vmtk 1 "bash /sp/v6/wall/jfix/vmtk_sub3.sh /v6w/subclavian_local" || exit 1
step vmtk_v5rule neve-build-vmtk 1 "bash /sp/v6/wall/jfix/vmtk_sub3.sh /v6w/subclavian_v5rule"
step records neve-build-stagea 0 "python3 /sp/v6/wall_local/junction_records.py"
step wd neve-build-stagea 0 "python3 /sp/v6/wall_local/wd_sub.py --all subclavian_local /v6w/subclavian_local && python3 /sp/v6/wall_local/wd_sub.py --all subclavian_v5rule /v6w/subclavian_v5rule"
step report neve-build-stagea 0 "python3 /sp/v6/wall_local/wd_sub_local.py"
log "chain B finished"
