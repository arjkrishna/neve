#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_final2.sh (= /sp/v5_va/pilot_proc_ia172_final2.sh in the build containers)
# pilot_proc_ia172_final2.sh -- after pilot_proc_ia172_final.sh (fourth instance, 2026-10-02 01:20). Idempotent; log
# _logs_proc/final2.log. Steps:
#  1. wait until final.log ends with "final pass done"
#  2. 140 and 001 were measured (21:59 / 22:28) before pilot_proc_ia172_measure.py gained the "rec" landmark stretches
#     (their landmarks.json already has landmarks_rec): re-run measure only, the old file kept as measure_pre_rec.json
#  3. collect all 12, summary, and the new grade / ranking table (pilot_proc_ia172_grade.py)
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172
LOGD=$DER_H/_logs_proc
CLOCK=$LOGD/.cpu_lock
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'
ALL="ia172_058 ia172_061 ia172_140 ia172_141 ia172_001 ia172_039 ia172_041 ia172_073 ia172_190 ia172_015 ia172_118 ia172_090"
lock() { until mkdir "$CLOCK" 2>/dev/null; do sleep 0.5; done; echo "final2 $$ $(date +%s)" > "$CLOCK/owner"; }
unlock() { rm -rf "$CLOCK"; }
{
echo "== $(date '+%F %T') final2 waiting for final.log"
until tail -1 $LOGD/final.log 2>/dev/null | grep -q "final pass done"; do sleep 20; done
echo "== $(date '+%F %T') final2 start"
for c in ia172_140 ia172_001; do
  D=$DER_H/$c/ct_seg/ct_seg4
  grep -q 'rec:origin_to_C1' $D/measure.json 2>/dev/null && { echo "$c measure has rec"; continue; }
  [ -f $D/measure_pre_rec.json ] || cp $D/measure.json $D/measure_pre_rec.json
  lock; docker run --rm --name pilot_proc_ia172_ms2_$$ --cpus 8 --memory 14g -e PYTHONUNBUFFERED=1 -e PILOT_CASE=$c -e PILOT_SEG_TAGS=ct_seg4 \
    "${MNT[@]}" -w /repo eve-training-fixed python3 /sp/v5_va/pilot_proc_ia172_measure.py 2>&1 | grep -v "$FILT"; unlock
done
docker run --rm --name pilot_proc_ia172_cl2_$$ --cpus 2 --memory 4g "${MNT[@]}" -w /repo eve-training-fixed bash -c \
  "python3 /sp/v5_va/pilot_proc_ia172_collect.py $ALL | tail -14 && python3 /sp/v5_va/pilot_proc_ia172_summary.py > /dev/null && echo summary written && python3 /sp/v5_va/pilot_proc_ia172_grade.py > /dev/null && echo grade written" 2>&1 | grep -v "$FILT"
echo "== $(date '+%F %T') final2 done"
} 2>&1 | tee -a $LOGD/final2.log
