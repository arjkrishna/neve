#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_fixes_wall2.sh (= /sp/v5_va/scaleup_fixes_wall2.sh in the build containers)
# scaleup_fixes_wall2.sh -- (scaleup_fixes, 2nd instance, 2026-10-02 06:30) pipeline v2 POST stages after "qc select rescue"
# on every PILOT case, with the pilot-re-run names (scaleup_fixes_pilot_v2.sh: wall ct_seg/ct_seg4v2, qc/select_v2.json,
# qc/gross_v2.json, *_v2.png, results/scaleup_fixes_cases_<src>.json). Order = cases a fix changed first:
#   A  rescued sides (selected trace changed -> the ct_seg4v2 of scaleup_fixes_ctseg_check.sh, seeded from the PILOT
#      selection, is re-made from select_v2.json): V2_FORCE=ctseg   ia172 001 061 | song 141 109
#   B  pinch-triggered walls (ct_seg4v2 from the check is already the v2 wall): ia172 015 090 118 | song 008 032 094 100 106
#   C  untouched cases (bit-identical walls; measured to confirm): ia172 039 058 140 141 190 041 073 | song 044 113 090
#   then grade (once per source).
# Idempotent (every stage of scaleup_v2_post.sh skips finished work; part A's force is dropped once its marker exists).
# Log: derived/thin_slice/_scaleup_fixes_logs/wall2.log. Agent lock: _scaleup_fixes_logs/.cpu_lock (one heavy container).
TH=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice
LOGD=$TH/_scaleup_fixes_logs
P=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/scaleup_fixes_pilot_v2.sh
ST="gross ctseg measure protrusion fig collect"
{
echo "== $(date '+%F %T') wall2 start (pid $$)"
for sc in ia172/ia172_001 ia172/ia172_061 song/AHMU1218141 song/AHMU1218109; do
  s=${sc%%/*}; c=${sc##*/}
  if [ ! -f $LOGD/.wall2_forced_$c ]; then
    V2_FORCE="ctseg fig" bash $P $s "$ST" $c && touch $LOGD/.wall2_forced_$c
  else
    bash $P $s "$ST" $c
  fi
done
bash $P ia172 "$ST" ia172_015 ia172_090 ia172_118
bash $P song "$ST" AHMU1218008 AHMU1218032 AHMU1218094 AHMU1218100 AHMU1218106
bash $P ia172 "$ST grade" ia172_039 ia172_058 ia172_140 ia172_141 ia172_190 ia172_041 ia172_073
bash $P song "$ST grade" AHMU1218044 AHMU1218113 AHMU1218090
touch $LOGD/.wall2_done
echo "== $(date '+%F %T') wall2 done"
} >> $LOGD/wall2.log 2>&1
