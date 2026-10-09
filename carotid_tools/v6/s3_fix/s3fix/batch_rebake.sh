#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\batch_rebake.sh (= /sp/v6/s3fix/batch_rebake.sh in the build containers)
# idempotent re-bake + checks: items "<name> <sx,sy,sz>" in $1, tag $2 (tree /s3/trees/<tag>, fresh copy of /full/tree3/<name>
# unless the bake marker exists), bake = $BAKE (default bake_v6_1p2_dev.py), checks = $CK parts $PARTS (default
# coverage,folds). Outputs: /s3/bake/<tag>/_done/<n>.bake.json, /s3/iso/<tag>/<n>.vtp, /s3/checks/<tag>/<n>.json.
# Claims: /s3/claim/<tag>/<n> (mkdir). Log /s3/logs/<tag>.batch.log.
L=$1; TAG=$2
BAKE=${BAKE:-/sp/v6/bake/bake_v6_1p2_dev.py}; CK=${CK:-/sp/v6/checks/checks_v6_1p6_dev.py}; PARTS=${PARTS:-coverage,folds}
T=/s3/trees/$TAG; BK=/s3/bake/$TAG; ISO=/s3/iso/$TAG; PIN=/s3/pin/$TAG; CO=/s3/checks/$TAG; CL=/s3/claim/$TAG; LG=/s3/logs
mkdir -p $T $BK/_done $ISO $PIN $CO $CL $LG/items
if [ -n "${MANIFEST:-}" ]; then
  if ! grep -v "^#" $MANIFEST | md5sum -c --quiet; then echo "$(date "+%F %T") REFUSED: toolchain != $MANIFEST" >> $LG/$TAG.batch.log; exit 2; fi
  echo "$(date "+%F %T") worker start, toolchain ok ($(md5sum $MANIFEST | cut -c1-8))" >> $LG/$TAG.batch.log
fi
tr -d '\r' < $L | while read N SH OPT; do
  [ -z "$N" ] && continue
  [ -f $CO/_done_$N ] && continue
  mkdir $CL/$N 2>/dev/null || continue
  if [ ! -f $BK/_done/$N.bake.json ]; then
    rm -rf $T/$N; cp -a /full/tree3/$N $T/$N < /dev/null
    t0=$(date +%s)
    python3 $BAKE $N --tree $T --bake $BK --iso $ISO --pin $PIN --lattice-shift $SH $( [ "$OPT" = bridge ] && echo --bridge-route-islands ) < /dev/null > $LG/items/$TAG.$N.bake.log 2>&1
    echo "$(date '+%F %T') bake $N shift $SH rc $? $(( $(date +%s) - t0 )) s $(tail -1 $LG/items/$TAG.$N.bake.log | cut -c1-200)" >> $LG/$TAG.batch.log
  fi
  if [ -f $BK/_done/$N.bake.json ]; then
    python3 $CK $N --tree $T --iso $ISO --out $CO --parts $PARTS < /dev/null > $LG/items/$TAG.$N.check.log 2>&1
    grep -q '^GATES' $LG/items/$TAG.$N.check.log && touch $CO/_done_$N
    echo "$(date '+%F %T') check $N $(grep '^GATES' $LG/items/$TAG.$N.check.log | cut -c1-300)" >> $LG/$TAG.batch.log
  fi
done
