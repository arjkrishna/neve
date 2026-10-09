#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\chain3.sh (= /sp/v6/wall/chain3.sh in the build containers)
# v6_wall chain3 (= chain2 with seg_half after sub_pilot; host, Git Bash, run detached): starts the post-seg steps as their inputs appear, never more than TWO
# v6wall_ containers at once (the agent's limit; the seg containers count). Idempotent: a step with a done marker
# /v6w/chain/<step>.done (host D:/Arjun/neve/carotid_data/v6_work/wall/chain/) is never rerun. Log: chain/chain.log.
#   bash chain.sh            (stop: create chain/STOP; check: cat chain/chain.log; docker ps | grep v6wall)
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
W=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v6/wall
H=/d/Arjun/neve/carotid_data/v6_work/wall
C=$H/chain; mkdir -p $C
log() { echo "$(date '+%F %T') $*" >> $C/chain.log; }
nrun() { docker ps --format '{{.Names}}' | grep -c '^v6wall_'; }
segdone() {   # $1 variant, $2 expected count: every unit has .seg_done and the batch container is gone
  local n; n=$(ls $H/seg/$1/*/.seg_done 2>/dev/null | wc -l); [ "$n" -ge "$2" ] && ! docker ps --format '{{.Names}}' | grep -q "^$3\$"; }
done_() { [ -f $C/$1.done ]; }
running() { docker ps --format '{{.Names}}' | grep -q "^v6wall_$1\$"; }
# step: name image deps(comma) command...
STEPS=(
 "vmtk_plateau|neve-build-vmtk|SEG_plateau|bash /sp/v6/wall/vmtk_batch.sh /v6w/seg/plateau"
 "post_B|neve-build-stagea|SEG_plateau|bash /sp/v6/wall/post_light.sh gate_plateau c3"
 "sub_gen|neve-build-stagea||python3 /sp/v6/wall/make_vmrseg6.py"
 "sub_pilot|neve-build-stagea|sub_gen|python3 /sp/v6/wall/sub_batch.py plateau pilot"
 "build_plateau|neve-build-stagea|vmtk_plateau|python3 /sp/v6/wall/build_donors6.py plateau"
 "sub_vmtk_pilot|neve-build-vmtk|sub_pilot|bash /sp/v6/wall/vmtk_sub.sh"
 "wd_v5|neve-build-stagea||python3 /sp/v6/wall/wd_batch.py v5,v5:ct:v5floorCT"
 "wd_plateau|neve-build-stagea|vmtk_plateau|python3 /sp/v6/wall/wd_batch.py plateau"
 "lumen|neve-build-stagea||bash /sp/v6/wall/post_light.sh lumen_v5 c4"
 "vmr_repro|neve-build-stagea|sub_gen|python3 /sp/v6/wall/seg_batch.py v5 /v6w/seg/v5repro vmr0248"
 "scan_plateau|neve-build-stagea|build_plateau|bash -c 'mkdir -p /v6w/scan/plateau && V6_DONORS=/repo/carotid_data/v6_work/wall/donors/plateau V6_SCAN_OUT=/v6w/scan/plateau python3 /sp/v6/wall/floor_scan6.py && V6_DONORS=/repo/carotid_data/v6_work/wall/donors/plateau V6_SCAN_OUT=/v6w/scan/plateau python3 /sp/v6/wall/bone_scan6.py'"
 "sub_rest|neve-build-stagea|sub_pilot|python3 /sp/v6/wall/sub_batch.py plateau rest"
 "sub_vmtk_rest|neve-build-vmtk|sub_rest,sub_vmtk_pilot|bash /sp/v6/wall/vmtk_sub.sh"
 "seg_half|neve-build-stagea|SEG_plateau,sub_pilot|python3 /sp/v6/wall/seg_batch.py half /v6w/seg/half all"
 "vmtk_half|neve-build-vmtk|seg_half|bash /sp/v6/wall/vmtk_batch.sh /v6w/seg/half"
 "post_A|neve-build-stagea|seg_half,vmr_repro|bash /sp/v6/wall/post_light.sh regress gate_half gate_plateau_g"
 "wd_half|neve-build-stagea|vmtk_half|python3 /sp/v6/wall/wd_batch.py half"
 "surf|neve-build-stagea|vmtk_plateau,vmtk_half|bash /sp/v6/wall/post_light.sh surf"
 "collect|neve-build-stagea|wd_v5,wd_plateau,wd_half,post_A,post_B,surf|bash -c 'python3 /sp/v6/wall/wd_collect.py v5,v5floorCT,plateau,half && python3 /sp/v6/wall/gate_summary.py'"
)
depok() {
  local d IFS=,
  for d in $1; do
    case $d in
      SEG_plateau) { segdone plateau 20 v6wall_segB && ! running segB2; } || return 1;;   # >= 20 units done, both plateau workers gone
      SEG_half) done_ seg_half || return 1;;
      "") ;;
      *) done_ $d || return 1;;
    esac
  done
  return 0
}
log "chain start"
while true; do
  [ -f $C/STOP ] && { log "STOP file: exit"; exit 0; }
  pending=0
  for s in "${STEPS[@]}"; do
    IFS='|' read -r name img deps cmd <<< "$s"
    done_ $name && continue
    pending=1
    if running $name; then continue; fi
    if [ -f $C/$name.started ] && ! running $name; then
      # container ended: done if its exit marker says rc 0
      if grep -q "rc 0" $C/$name.exit 2>/dev/null; then touch $C/$name.done; log "$name done"; else log "$name FAILED (see $C/$name.exit); not retried"; touch $C/$name.done; touch $C/$name.failed; fi
      continue
    fi
    depok "$deps" || continue
    [ "$(nrun)" -ge 2 ] && continue
    U=""; [ "$img" = neve-build-vmtk ] && U="-u 0"
    SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
    docker run -d --rm $U --name v6wall_$name --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
      -v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" -v "D:\Arjun\neve\carotid_data\v6_work\wall:/v6w" \
      -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro" \
      $img bash -c "$cmd > /v6w/chain/$name.log 2>&1; echo rc \$? > /v6w/chain/$name.exit" >/dev/null && touch $C/$name.started && log "$name started ($img)"
    sleep 3
  done
  [ $pending = 0 ] && { log "chain: all steps done"; exit 0; }
  sleep 30
done
