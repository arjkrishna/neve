#!/bin/bash
# Re-run graft_v5_reg_sep 1.0 (route13/graft_v5_reg_sep.v2.py) DRY on the 13 route13 anatomies, with the pre-swap
# anatomies_v5 folders overlaid read-only, to check it reproduces the shipped ext records. Idempotent.
cd /repo
while read -r N T; do
  [ -z "$N" ] && continue
  [ -f /sp/v5_va/r10check/out/$N.json ] && continue
  echo "$(date '+%F %T') start $N $T" >> /sp/v5_va/r10check/run.log
  python3 /sp/v5_va/r10check/graft_v5_reg_sep_v10.py --only $N --target $T --extended \
     --dry-out /sp/v5_va/r10check/out/$N.json > /sp/v5_va/r10check/out/$N.log 2>&1; rc=$?
  echo "$(date '+%F %T') end $N exit $rc" >> /sp/v5_va/r10check/run.log
done < /sp/v5_va/r10check/jobs.txt
echo "$(date '+%F %T') DONE" >> /sp/v5_va/r10check/run.log
