#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_watchdog.sh (= /sp/v5_va/pilot_proc_ia172_watchdog.sh in the build containers)
# pilot_proc_ia172_watchdog.sh -- stops an Arterial run of THIS agent (an ia172 case) whose centreline stage has written
# nothing to its arterial.log for $WD_MIN minutes (default 15). Reason: on ia172_061 Arterial's centreline stage
# (vtkvmtkPolyDataNetworkExtraction, after "can't reconstruct new profile") ran 22 min at 100 % of one core without
# output while holding the GPU lock (058 finished the stage in 120 s). Its segmentation stage has finished by then
# (segmentation.nii.gz + run_report.json written), so the driver moves on and art_extract traces the binary mask
# without Arterial labels (its documented no-label path; s0960 in the study). Writes runs/native/watchdog.json.
# Only containers whose arguments contain this agent's case path are touched. Exits when every case has
# arterial/.arterial_done. Log: _logs_proc/watchdog.log.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172
LOG=$DER_H/_logs_proc/watchdog.log
WD_MIN=${WD_MIN:-15}
CASES="ia172_058 ia172_061 ia172_140 ia172_141 ia172_001 ia172_039 ia172_041 ia172_073 ia172_190 ia172_015 ia172_118 ia172_090"
echo "$(date '+%F %T') watchdog start (limit $WD_MIN min)" >> $LOG
while true; do
  left=0
  for c in $CASES; do
    d=$DER_H/$c/arterial
    [ -f $d/.arterial_done ] && continue
    left=$((left + 1))
    lg=$d/runs/native/arterial.log
    [ -f $lg ] || continue
    grep -q "===== stage cl =====" $lg || continue
    grep -q "===== stage cl:" $lg && continue
    age=$(( $(date +%s) - $(stat -c %Y $lg) ))
    [ $age -lt $((WD_MIN * 60)) ] && continue
    for n in $(docker ps --format '{{.Names}}' | grep '^arterial_'); do
      if docker inspect -f '{{.Args}}' $n 2>/dev/null | grep -q "thin_slice/ia172/$c/"; then
        printf '{"stopped_by": "pilot_proc_ia172_watchdog.sh", "when": "%s", "container": "%s", "why": "Arterial centreline stage wrote nothing to arterial.log for %d s (limit %d min); segmentation stage finished; art_extract traces the binary mask without Arterial labels"}\n' \
          "$(date '+%F %T')" "$n" "$age" "$WD_MIN" > $d/runs/native/watchdog.json
        echo "$(date '+%F %T') $c: no log output for $age s in the centreline stage -> docker stop $n" >> $LOG
        docker stop $n >> $LOG 2>&1
      fi
    done
  done
  [ $left -eq 0 ] && { echo "$(date '+%F %T') watchdog: all cases done" >> $LOG; exit 0; }
  sleep 60
done
