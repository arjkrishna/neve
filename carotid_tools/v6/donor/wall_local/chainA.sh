#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\chainA.sh (= /sp/v6/wall_local/chainA.sh in the build containers)
# s2_harden_wall_bake D15 chain A (host, Git Bash; ONE container at a time, names s2wb_A_<step>): the local VA walls.
# Idempotent: a step with marker v6_work/wall/local/chain/<step>.done is skipped. Log: v6_work/wall/local/chain/chainA.log
# Stop: create v6_work/wall/local/chain/STOP_A.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
C=/d/Arjun/neve/carotid_data/v6_work/wall/local/chain; mkdir -p $C
log() { echo "$(date '+%F %T') $*" >> $C/chainA.log; }
step() {  # name image u0 cmd
  local name=$1 img=$2 u0=$3; shift 3
  [ -f $C/$name.done ] && return 0
  [ -f $C/STOP_A ] && { log "STOP"; exit 0; }
  log "$name start ($img)"
  U0=$u0 bash $SP/v6/bake/dr.sh $img - bash -c "$* > /v6w/local/chain/$name.log 2>&1; echo rc \$? > /v6w/local/chain/$name.exit"
  if grep -q "rc 0" $C/$name.exit 2>/dev/null; then touch $C/$name.done; log "$name done"; else log "$name FAILED"; return 1; fi
}
step seg neve-build-stagea 0 "python3 /sp/v6/wall_local/local_seg.py all" || exit 1
step vmtk neve-build-vmtk 1 "bash /sp/v6/wall/vmtk_batch.sh /v6w/seg/local" || exit 1
step splice neve-build-stagea 0 "python3 /sp/v6/wall_local/splice_lines.py" || exit 1
step build eve-training-fixed 0 "python3 /sp/v6/wall_local/build_donors_local.py" || exit 1
step wd neve-build-stagea 0 "python3 /sp/v6/wall/wd_batch.py local" || exit 1
step gate_ct neve-build-stagea 0 "python3 /sp/v6/wall/gate_ct.py /v6w/seg/local local"
step scans neve-build-stagea 0 "mkdir -p /v6w/scan/local && V6_DONORS=/repo/carotid_data/v6_work/wall/donors_local V6_SCAN_OUT=/v6w/scan/local python3 /sp/v6/wall/floor_scan6.py && V6_DONORS=/repo/carotid_data/v6_work/wall/donors_local V6_SCAN_OUT=/v6w/scan/local python3 /sp/v6/wall/bone_scan6.py"
step wdlocal neve-build-stagea 0 "python3 /sp/v6/wall_local/wd_local.py"
log "chain A finished"
