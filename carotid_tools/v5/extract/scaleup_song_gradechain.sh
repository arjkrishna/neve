#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_gradechain.sh (= /sp/v5_va/scaleup_song_gradechain.sh in the build containers)
# scaleup_song_gradechain.sh <case> [case ...]   (scaleup_lane_song, 2026-10-02 relaunch; Git Bash on the host)
# Detached grade loop (= scaleup_song_chain.sh grade, but calling the pipeline-v2 wrapper scaleup_song_grade.sh instead of
# the pilot grade group of scaleup_song_run.sh; a new file so the running chain script is never edited).
# Per case, in order: wait for _logs_scaleup/chain_trace_done_<case> (Arterial + extract tsba/ts + VMTK done) and the GO
# marker _logs_scaleup/.grade_go, then scaleup_song_grade.sh <case>. Rejected / fetch-failed cases are skipped. Wait cap
# per case 8 h. Log: _logs_scaleup/chain_grade2.out (nohup). Done-marker: _logs_scaleup/chain_grade2_done_<case>.
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOGD=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song/_logs_scaleup
echo "== $(date '+%F %T') chain grade2 start (pid $$): $*"
for c in "$@"; do
  [ -f "$LOGD/chain_grade2_done_$c" ] && { echo "$c: grade2 done before"; continue; }
  n=0
  while :; do
    { [ -f "$LOGD/rejected_$c" ] || [ -f "$LOGD/fetch_failed_$c" ]; } && { echo "$c: rejected / fetch failed: skip"; continue 2; }
    [ -f "$LOGD/chain_trace_done_$c" ] && [ -f "$LOGD/.grade_go" ] && break
    n=$((n+1)); [ $n -gt 960 ] && { echo "$(date '+%F %T') $c: prerequisite of grade2 not met after 8 h: skipped"; continue 2; }
    sleep 30
  done
  bash $SP/v5_va/scaleup_song_grade.sh "$c"
  [ -f "$LOGD/grade_done_$c" ] && touch "$LOGD/chain_grade2_done_$c"
done
echo "== $(date '+%F %T') chain grade2 finished"
