#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_fixes_wall2_finish.sh (= /sp/v5_va/scaleup_fixes_wall2_finish.sh in the build containers)
# scaleup_fixes_wall2_finish.sh <wall2 msys pid> -- (scaleup_fixes, 2026-10-02 07:10) DECISION: group C of scaleup_fixes_wall2.sh
# (the 10 pilot cases no fix touched: 16 bit-identical walls + 3 wall-less cases) is NOT re-measured -- v2 = pilot there, the
# tally keeps their pilot rows, and 7 x ~10 min of 6-CPU containers would only slow the lanes. This script waits until group
# B is finished (Song 106 collect done), stops wall2 (its process tree and its own container, scaleup_fixes_v2_*), frees this
# agent's lock if a dead v2post owns it, then runs the v2 grade for both sources (results/scaleup_fixes_cases_<src>_grade.json)
# and touches _scaleup_fixes_logs/.wall2_done. Log: _scaleup_fixes_logs/wall2.log. Idempotent (a rerun after .wall2_done exits).
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
W=$1
LOGD=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/_scaleup_fixes_logs
P=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/scaleup_fixes_pilot_v2.sh
[ -f $LOGD/.wall2_done ] && exit 0
until grep -q "AHMU1218106 collect done" $LOGD/wall2.log; do
  sleep 3
done
{
echo "== $(date '+%F %T') wall2_finish: group B done -> stopping wall2 (pid $W) before group C (decision 07:10)"
desc() { ps -ef | awk -v p=$1 '$3==p {print $2}' | while read c; do echo $c; desc $c; done; }
PIDS="$W $(desc $W)"
kill $W 2>/dev/null
for p in $PIDS; do kill $p 2>/dev/null; done
sleep 2
for p in $PIDS; do kill -9 $p 2>/dev/null; done
for n in $(docker ps --format '{{.Names}}' | grep '^scaleup_fixes_v2_'); do echo "docker stop $n"; docker stop $n >/dev/null; done
L=$LOGD/.cpu_lock
if [ -f $L/owner ]; then o=$(awk '{print $1}' $L/owner); op=$(awk '{print $2}' $L/owner)
  if [ "$o" = v2post ] && ! kill -0 "$op" 2>/dev/null; then echo "removing my stale lock: $(cat $L/owner)"; rm -rf $L; fi
fi
echo "== $(date '+%F %T') wall2_finish: grade ia172 + song"
bash $P ia172 grade ia172_001
bash $P song grade AHMU1218141
touch $LOGD/.wall2_done
echo "== $(date '+%F %T') wall2 done (group C skipped by decision; pilot rows stand for those cases)"
} >> $LOGD/wall2.log 2>&1
