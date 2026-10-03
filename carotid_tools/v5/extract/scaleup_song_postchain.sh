#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_postchain.sh (= /sp/v5_va/scaleup_song_postchain.sh in the build containers)
# scaleup_song_postchain.sh <case> [case ...]   (scaleup_lane_song instance 3, 2026-10-02 09:55; Git Bash; detached via nohup)
# Pipeline-v2 STEP 2 (notes/SCALEUP_PIPELINE.md, UPDATE 08:45) for every Song lane case + the v2 grade of cases that the
# batch-1 grade chain (scaleup_song_gradechain.sh, list = batch 1) does not cover. Per case, in order:
#   - if the case is in batch1_cases.txt: wait for chain_grade2_done_<case> (the running grade2 chain grades it; no
#     second grader -> never graded twice), then scaleup_ctseg_fallback.sh song <case>
#   - else (extension cases): wait for chain_trace_done_<case> + .grade_go, then scaleup_song_grade.sh <case> (v2 post2),
#     touch chain_grade2_done_<case>, then the fallback
# Rejected / fetch-failed cases are skipped. Wait cap 8 h per case. Done-marker fallback_done_<case>.
# Log: _logs_scaleup/chain_postchain.out (nohup).
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOGD=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song/_logs_scaleup
echo "== $(date '+%F %T') postchain start (pid $$): $*"
for c in "$@"; do
  [ -f "$LOGD/fallback_done_$c" ] && { echo "$c: fallback done before"; continue; }
  inb1=0; grep -qw "$c" "$LOGD/batch1_cases.txt" && inb1=1
  n=0
  while :; do
    { [ -f "$LOGD/rejected_$c" ] || [ -f "$LOGD/fetch_failed_$c" ]; } && { echo "$c: rejected / fetch failed: skip"; continue 2; }
    if [ $inb1 -eq 1 ]; then [ -f "$LOGD/chain_grade2_done_$c" ] && break
    else [ -f "$LOGD/chain_trace_done_$c" ] && [ -f "$LOGD/.grade_go" ] && break; fi
    n=$((n+1)); [ $n -gt 960 ] && { echo "$(date '+%F %T') $c: prerequisite not met after 8 h: skipped"; continue 2; }
    sleep 30
  done
  if [ $inb1 -eq 0 ]; then
    echo "== $(date '+%F %T') $c grade (v2 post2)"
    bash $SP/v5_va/scaleup_song_grade.sh "$c"
    [ -f "$LOGD/grade_done_$c" ] && touch "$LOGD/chain_grade2_done_$c"
  fi
  [ -f "$LOGD/grade_done_$c" ] || { echo "$c: not graded -> no fallback"; continue; }
  echo "== $(date '+%F %T') $c fallback (step 2)"
  bash $SP/v5_va/scaleup_ctseg_fallback.sh song "$c"
  touch "$LOGD/fallback_done_$c"
done
echo "== $(date '+%F %T') postchain finished"
