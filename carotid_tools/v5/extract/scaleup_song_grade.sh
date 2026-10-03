#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_grade.sh (= /sp/v5_va/scaleup_song_grade.sh in the build containers)
# scaleup_song_grade.sh <case> [case ...]   v2 (scaleup_lane_song instance 3, 2026-10-02 10:15; v1 in scaleup_work/; Git Bash)
# The POST stage (QC onward) of a Song scale-up case = pipeline v2 (notes/SCALEUP_PIPELINE.md, scaleup_fixes; complete 09:00):
#   step 1  scaleup_v2_post.sh song post2 <case> (pilot qc2 on tsba + ts -> select -> V1 rescue -> gross -> ct_seg4 wall with
#           the C2-exit pinch fix -> landmarks / measure / edge -> protrusion -> figures -> collect -> grade) -> grade_done_<case>
#   step 2  scaleup_ctseg_fallback.sh song <case> (UPDATE 08:45)                                      -> fallback_done_<case>
# v2 changes RESOURCES only, not rules (same as the 172 lane's scaleup_ia172_grade.sh):
#   - the v2 driver runs from a per-call patched copy (scaleup_song_v2patch.sh): stages qc gross seg pr fig v1r as light 2-CPU
#     containers in two light slots (_logs_scaleup/.cpu_lock_v2light, _b) [11:05 v3: + ms = landmarks/measure/edge, measured ~1.2 cores]; VMTK and re-trace stay heavy
#     under the lane lock; collect / grade under _logs_scaleup/.results_lock (two chains cannot lose a record);
#   - per-case claim (mkdir grade_claim_<case>; a claim whose owner pid is dead is taken over) so two grade chains can share
#     one list; step 2 runs here, with the same patched driver.
# Never runs the pilot grade group (no case graded twice). Rejected cases skipped. Idempotent.
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOGD=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song/_logs_scaleup
export LIGHT_V2=${LIGHT_V2:-" qc gross seg ms pr fig v1r "}
for c in "$@"; do
  [ -f "$LOGD/rejected_$c" ] && { echo "$c: rejected at fetch review: not graded"; continue; }
  [ -f "$LOGD/grade_done_$c" ] && [ -f "$LOGD/fallback_done_$c" ] && { echo "$c: graded before ($(cat "$LOGD/grade_done_$c"))"; continue; }
  [ -f "$LOGD/.grade_go" ] || { echo "$c: no .grade_go (pipeline v2 not released): not graded"; continue; }
  if ! mkdir "$LOGD/grade_claim_$c" 2>/dev/null; then
    op=$(awk '{print $2}' "$LOGD/grade_claim_$c/owner" 2>/dev/null)
    if [ -n "$op" ] && ! kill -0 "$op" 2>/dev/null; then echo "$c: stale claim ($(cat "$LOGD/grade_claim_$c/owner")) taken over"
    else echo "$c: claimed by another grade chain ($(cat "$LOGD/grade_claim_$c/owner" 2>/dev/null)): skipped here"; continue; fi
  fi
  echo "pid $$ $(date '+%F %T')" > "$LOGD/grade_claim_$c/owner"
  PD=$LOGD/scaleup_v2_post_patched_$$.sh
  bash $SP/v5_va/scaleup_song_v2patch.sh "$PD" | tee -a "$LOGD/v2patch.log"
  if [ ! -f "$LOGD/grade_done_$c" ]; then
    V2_NAME=scaleup_song_v2 bash "$PD" song post2 "$c"
    [ $? -eq 0 ] && echo "pipeline v2 (scaleup_v2_post.sh song post2 via scaleup_song_v2patch) $(date '+%F %T')" > "$LOGD/grade_done_$c"
  fi
  if [ -f "$LOGD/grade_done_$c" ] && [ ! -f "$LOGD/fallback_done_$c" ]; then
    FBP=$LOGD/scaleup_ctseg_fallback_patched_$$.sh
    sed "s#bash \$SPU/scaleup_v2_post.sh #bash $PD #" $SP/v5_va/scaleup_ctseg_fallback.sh > "$FBP"
    grep -q "bash $PD " "$FBP" || cp $SP/v5_va/scaleup_ctseg_fallback.sh "$FBP"
    echo "== $(date '+%F %T') $c step 2 fallback check"
    V2_NAME=scaleup_song_v2 bash "$FBP" song "$c" 2>&1 | tee -a "$LOGD/${c}_v2_fallback.log"
    [ ${PIPESTATUS[0]} -eq 0 ] && echo "scaleup_ctseg_fallback.sh $(date '+%F %T')" > "$LOGD/fallback_done_$c"
    rm -f "$FBP"
  fi
  rm -f "$PD"; rm -rf "$LOGD/grade_claim_$c"
done
