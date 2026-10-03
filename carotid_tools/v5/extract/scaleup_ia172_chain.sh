#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_chain.sh (= /sp/v5_va/scaleup_ia172_chain.sh in the build containers)
# scaleup_ia172_chain.sh <fetch|ts|tscpu|gpu|post|grade> <case> [case ...]   (scaleup_lane_ia172, 2026-10-02; Git Bash)
# Detached per-group loop (the pilot's GPU / TS / post loops; same design as scaleup_song_chain.sh): per case, in order,
# wait for the prerequisite of the group, then run scaleup_ia172_run.sh <group> <case> (idempotent).
#   fetch : no wait (the stage itself enforces the pre-fetch arch check)
#   ts    : (stage tsgpu) _logs_scaleup/fetch_done_<case>, and no auto flag unless I accepted the case by eye (visual_ok_<case>)
#   tscpu : totalseg/.ts_ml_done (or per-class masks)
#   gpu   : TS per-class masks vertebrae_C1 + skull (args / clip / Arterial)
#   post  : arterial/.arterial_done and the totalseg_1p5 masks (extract tsba + ts, VMTK)
#   grade : post_done_<case> and the GO marker _logs_scaleup/.grade_go (pipeline v2, scaleup_ia172_grade.sh)
# Skips a case with rejected_<case> / fetch_failed_<case>. Wait cap per case 8 h (then skipped, logged).
# Log: _logs_scaleup/chain_<group>.out (nohup). Done marker: _logs_scaleup/chain_<group>_done_<case>.
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
SRC_H=/d/Arjun/neve/carotid_data/va_sources/thin_slice/ia172
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172
LOGD=$DER_H/_logs_scaleup
mkdir -p "$LOGD"
group=$1; shift
echo "== $(date '+%F %T') chain $group start (pid $$): $*"
for c in "$@"; do
  [ -f "$LOGD/chain_${group}_done_$c" ] && { echo "$c: $group done before"; continue; }
  n=0
  while :; do
    { [ -f "$LOGD/rejected_$c" ] || [ -f "$LOGD/fetch_failed_$c" ]; } && { echo "$(date '+%F %T') $c: rejected / fetch failed: skip"; continue 2; }
    case $group in
      fetch) break;;
      ts) [ -f "$LOGD/fetch_done_$c" ] && { [ ! -f "$LOGD/fetch_flag_$c" ] || [ -f "$LOGD/visual_ok_$c" ]; } && break;;
      tscpu) { [ -f "$DER_H/$c/totalseg/.ts_ml_done" ] || [ -f "$DER_H/$c/totalseg/segmentations/vertebrae_C1.nii.gz" ]; } && break;;
      gpu) [ -f "$DER_H/$c/totalseg/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$DER_H/$c/totalseg/segmentations/skull.nii.gz" ] && break;;
      post) [ -f "$DER_H/$c/arterial/.arterial_done" ] && [ -f "$DER_H/$c/totalseg_1p5/segmentations/vertebrae_C1.nii.gz" ] && break;;
      grade) [ -f "$LOGD/post_done_$c" ] && [ -f "$LOGD/.grade_go" ] && break;;
      *) echo "unknown group $group"; exit 2;;
    esac
    n=$((n+1)); [ $n -gt 960 ] && { echo "$(date '+%F %T') $c: prerequisite of $group not met after 8 h: skipped"; continue 2; }
    sleep 30
  done
  if [ "$group" = "grade" ]; then
    bash $SP/v5_va/scaleup_ia172_grade.sh "$c"
  else
    st=$group; [ "$group" = ts ] && st=tsgpu
    bash $SP/v5_va/scaleup_ia172_run.sh $st "$c"
  fi
  touch "$LOGD/chain_${group}_done_$c"
done
echo "== $(date '+%F %T') chain $group finished"
