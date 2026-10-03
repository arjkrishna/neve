#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_grade.sh (= /sp/v5_va/scaleup_ia172_grade.sh in the build containers)
# scaleup_ia172_grade.sh <case> [case ...]   (scaleup_lane_ia172, 2026-10-02; Git Bash)
# The POST stage (QC onward) of a 172-series scale-up case = pipeline v2 (notes/SCALEUP_PIPELINE.md, by scaleup_fixes;
# Status complete 09:00):
#   step 1  scaleup_v2_post.sh ia172 post2 <case>  (qc on tsba + ts -> select -> V1 rescue -> gross -> ct_seg4 wall with the
#           C2-exit pinch fix -> landmarks / measure / edge -> protrusion -> figures -> collect into results/scaleup_ia172.json
#           -> grade)                                                                   -> marker grade_done_<case>
#   step 2  scaleup_ctseg_fallback.sh ia172 <case> (UPDATE 08:45: VMTK-centreline check of each ct_seg4 VA; per-vessel
#           fallback to the pilot cast only when the check fails AND the pinch fix acted) -> marker fallback_done_<case>
# Run by scaleup_ia172_chain.sh grade (after _logs_scaleup/.grade_go) or by hand for cases graded before step 2 existed.
# V2_CPU_WANT keeps the lane's D3 priority. Idempotent.
# Resources (07:30 / 08:45, scaleup_ia172_v2patch.sh): the v2 driver runs from a per-call patched copy (single-core stages as
#   light 2-CPU containers, two light slots, results lock around collect / grade); the fallback script is used unchanged
#   except that it calls that patched copy instead of scaleup_v2_post.sh.
# Two grade chains share one case list through a per-case claim (mkdir _logs_scaleup/grade_claim_<case>): a claimed case is
#   skipped by the other chain. v1-v3 of this file: scaleup_work/scaleup_ia172_grade.sh.v*.
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOGD=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172/_logs_scaleup
for c in "$@"; do
  [ -f "$LOGD/grade_done_$c" ] && [ -f "$LOGD/fallback_done_$c" ] && { echo "$c: graded before ($(cat "$LOGD/grade_done_$c"); $(cat "$LOGD/fallback_done_$c"))"; continue; }
  [ -f "$LOGD/.grade_go" ] || { echo "$c: no .grade_go (pipeline v2 not released): not graded"; continue; }
  mkdir "$LOGD/grade_claim_$c" 2>/dev/null || { echo "$c: claimed by another grade chain ($(cat "$LOGD/grade_claim_$c/owner" 2>/dev/null)): skipped here"; continue; }
  echo "pid $$ $(date '+%F %T')" > "$LOGD/grade_claim_$c/owner"
  PD=$LOGD/scaleup_v2_post_patched_$$.sh
  bash $SP/v5_va/scaleup_ia172_v2patch.sh "$PD" | tee -a "$LOGD/v2patch.log"
  if [ ! -f "$LOGD/grade_done_$c" ]; then
    V2_CPU_WANT=$LOGD/.cpu_want_hi V2_NAME=scaleup_ia172_v2 bash "$PD" ia172 post2 "$c"
    [ $? -eq 0 ] && echo "pipeline v2 (scaleup_v2_post.sh post2 via v2patch) $(date '+%F %T')" > "$LOGD/grade_done_$c"
  fi
  if [ -f "$LOGD/grade_done_$c" ] && [ ! -f "$LOGD/fallback_done_$c" ]; then
    FBP=$LOGD/scaleup_ctseg_fallback_patched_$$.sh
    sed "s#bash \$SPU/scaleup_v2_post.sh #bash $PD #" $SP/v5_va/scaleup_ctseg_fallback.sh > "$FBP"
    grep -q "bash $PD " "$FBP" || cp $SP/v5_va/scaleup_ctseg_fallback.sh "$FBP"
    echo "== $(date '+%F %T') $c step 2 fallback check"
    V2_CPU_WANT=$LOGD/.cpu_want_hi V2_NAME=scaleup_ia172_v2 bash "$FBP" ia172 "$c" 2>&1 | tee -a "$LOGD/${c}_v2_fallback.log"
    [ ${PIPESTATUS[0]} -eq 0 ] && echo "scaleup_ctseg_fallback.sh $(date '+%F %T')" > "$LOGD/fallback_done_$c"
    rm -f "$FBP"
  fi
  rm -f "$PD"; rm -rf "$LOGD/grade_claim_$c"
done
