#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\graft13.sh (= /sp/v5_va/route13/graft13.sh in the build containers)
# v5_route13: re-plan anatomies with graft_v5_reg_sep.py (inside eve-training-fixed), one per line of a job file:
#   <name> <target right|left|both> <tag> [--extended] [--tb-shift] [--write]
# DRY by default (record /sp/v5_va/route13/graft/<tag>/<name>.json); --write also writes route13/moved_<tag>/<name>.
# Idempotent: a job whose record exists is skipped.        bash graft13.sh <job file>
OUT=/sp/v5_va/route13/graft
while read -r N T TAG REST; do
  [ -z "$N" ] && continue
  mkdir -p $OUT/$TAG
  [ -f $OUT/$TAG/$N.json ] && continue
  W=""
  FL=""
  for x in $REST; do
    case $x in --write) W="--out /repo/carotid_data/va_sources/v5_reg/route13/moved_$TAG";; *) FL="$FL $x";; esac
  done
  echo "$(date '+%F %T') start $N $T $TAG $FL $W" >> $OUT/graft13.log
  python3 /sp/v5_va/graft_v5_reg_sep.py --only $N --target $T --dry-out $OUT/$TAG/$N.json $FL $W > $OUT/$TAG/$N.log 2>&1
  echo "$(date '+%F %T') end $N exit $?" >> $OUT/graft13.log
done < <(tr -d '\r' < $1)
echo "$(date '+%F %T') JOBS DONE $1" >> $OUT/graft13.log
