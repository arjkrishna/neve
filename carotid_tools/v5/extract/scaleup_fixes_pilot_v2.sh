#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_fixes_pilot_v2.sh (= /sp/v5_va/scaleup_fixes_pilot_v2.sh in the build containers)
# scaleup_fixes_pilot_v2.sh <source> <stage|group> <case> [...] -- (scaleup_fixes, 2026-10-02) pipeline v2 on PILOT cases without
# touching the pilot's files: scaleup_v2_post.sh with the pilot-re-run names (wall tag ct_seg4v2, qc/select_v2.json,
# qc/gross_v2.json, figures *_v2.png, results/scaleup_fixes_cases_<src>.json, logs <derived>/<src>/_logs_scaleup_fixes) and
# this agent's single heavy-container lock (derived/thin_slice/_scaleup_fixes_logs/.cpu_lock, shared with
# scaleup_fixes_ctseg_check.sh).
S=$1; shift
TH=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice
export V2_LOGD=$TH/$S/_logs_scaleup_fixes V2_CPU_LOCK=$TH/_scaleup_fixes_logs/.cpu_lock PILOT_SEG_TAG=ct_seg4v2 \
       SCALEUP_SELECT=select_v2.json SCALEUP_GROSS=gross_v2.json SCALEUP_FIG_SUFFIX=_v2 V2_NAME=scaleup_fixes_v2_$S \
       SCALEUP_RESULTS=/sp/v5_va/results/scaleup_fixes_cases_$S.json
bash /c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/scaleup_v2_post.sh "$S" "$@"
