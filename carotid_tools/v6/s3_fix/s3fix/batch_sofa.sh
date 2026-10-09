#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\batch_sofa.sh (= /sp/v6/s3fix/batch_sofa.sh in the build containers)
# idempotent SOFA RCCA (sofa_rcca_v6 1.0, frozen manifest tool), seeds 777,778,779 x 300 steps, on names in $1 from the
# anatomy folder $2 -> /s3/sofa/$3/<name>.json (done = exists). Claims /s3/sofa/_claim_$3/<name>. eve-training-fixed.
L=$1; ROOT=$2; TAG=$3; O=/s3/sofa/$TAG; C=/s3/sofa/_claim_$TAG
mkdir -p $O $C /s3/sofa/logs
for N in $(tr -d '\r' < $L); do
  [ -f $O/$N.json ] && continue
  mkdir $C/$N 2>/dev/null || continue
  t0=$(date +%s)
  SOFA_OUT=$O timeout 3600 python3 /sp/v6/sofa/sofa_rcca_v6.py $N 300 $ROOT 777,778,779 < /dev/null > /s3/sofa/logs/$TAG.$N.log 2>&1
  echo "$(date '+%F %T') $N rc $? $(( $(date +%s) - t0 )) s $(grep -h '^SOFA R\|^RCCA EXERCISED' /s3/sofa/logs/$TAG.$N.log | tr '\n' ' ' | cut -c1-200)" >> /s3/sofa/logs/$TAG.batch.log
done
