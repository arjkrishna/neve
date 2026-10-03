#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_grade4.sh (= /sp/v5_va/scaleup_song_grade4.sh in the build containers)
# scaleup_song_grade4.sh <case> [case ...]  (scaleup_lane_song instance 4, 2026-10-02 13:15; Git Bash; one background job)
# Pipeline v2 exactly as notes/SCALEUP_PIPELINE.md sec 2, UNPATCHED driver, one case at a time:
#   scaleup_v2_post.sh song post2 <case>  -> grade_done_<case> (+ chain_grade2_done_<case>)
#   scaleup_ctseg_fallback.sh song <case> -> fallback_done_<case>  (log <case>_v2_fallback.log)
# A case is graded only after its by-eye fetch review: waits (cap 3 h) for _logs_scaleup/fetchrev_ok_<case>
# (touched by the lane after "accept"); rejected_<case> -> skipped. Idempotent (done markers). Log: v2_post.out.
SPU=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va
LOGD=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song/_logs_scaleup
echo "== $(date '+%F %T') grade4 start (pid $$): $*"
for c in "$@"; do
  [ -f "$LOGD/grade_done_$c" ] && [ -f "$LOGD/fallback_done_$c" ] && { echo "$c: graded before"; continue; }
  n=0
  until [ -f "$LOGD/fetchrev_ok_$c" ]; do
    [ -f "$LOGD/rejected_$c" ] && { echo "$(date '+%F %T') $c: rejected at fetch review: not graded"; continue 2; }
    n=$((n+1)); [ $n -gt 360 ] && { echo "$(date '+%F %T') $c: no fetch verdict after 3 h: skipped"; continue 2; }
    sleep 30
  done
  [ -f "$LOGD/chain_trace_done_$c" ] || { echo "$c: not traced: skipped"; continue; }
  if [ ! -f "$LOGD/grade_done_$c" ]; then
    echo "== $(date '+%F %T') $c post2"
    bash $SPU/scaleup_v2_post.sh song post2 "$c"
    rc=$?
    [ $rc -eq 0 ] && { echo "pipeline v2 (scaleup_v2_post.sh song post2, unpatched; grade4) $(date '+%F %T')" > "$LOGD/grade_done_$c"; touch "$LOGD/chain_grade2_done_$c"; }
    [ $rc -ne 0 ] && { echo "$(date '+%F %T') $c post2 rc $rc: no fallback"; continue; }
  fi
  echo "== $(date '+%F %T') $c step 2 fallback"
  bash $SPU/scaleup_ctseg_fallback.sh song "$c" 2>&1 | tee -a "$LOGD/${c}_v2_fallback.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && echo "scaleup_ctseg_fallback.sh $(date '+%F %T')" > "$LOGD/fallback_done_$c"
done
echo "== $(date '+%F %T') grade4 finished"
