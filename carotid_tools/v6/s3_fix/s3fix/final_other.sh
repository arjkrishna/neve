#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\final_other.sh (= /sp/v6/s3fix/final_other.sh in the build containers)
# s3_fix final round for the anatomies that are NOT re-baked (neve-build-meshlab; idempotent; claims by mkdir):
#   items "<name> fold"   -> /s3/trees/final/<n> = fresh copy of /full/tree3/<n>; fold_fix.py on it; union = /full/iso/<n>.vtp
#                            (unchanged: only the OBJ changes); FULL checks_v6 (every part).
#   items "<name> record" -> /s3/trees/final/<n> = fresh copy of /full/tree3/<n> (nothing changes but the stamped record);
#                            checks record = /full/checks4/<n>.json + checks_v6 parts coverage,folds (the other parts are
#                            carried: geometry and bake identical to the checked tree, as checks_v6 1.4's route_d22 round).
# The union is copied to /s3/iso/final/<n>.vtp so every final anatomy has its union beside the re-baked ones.
#   MANIFEST=<toolchain md5 file> bash final_other.sh <items>
L=$1; T=/s3/trees/final; ISO=/s3/iso/final; CO=/s3/checks/final; CL=/s3/claim/final_other; LG=/s3/logs
CK=/sp/v6/checks/checks_v6.py; FF=/sp/v6/s3fix/fold_fix.py
mkdir -p $T $ISO $CO $CL $LG/items /s3/fold
if [ -n "${MANIFEST:-}" ]; then
  if ! grep -v "^#" $MANIFEST | md5sum -c --quiet; then echo "$(date "+%F %T") REFUSED: toolchain != $MANIFEST" >> $LG/final_other.batch.log; exit 2; fi
  echo "$(date "+%F %T") worker start, toolchain ok ($(md5sum $MANIFEST | cut -c1-8))" >> $LG/final_other.batch.log
fi
tr -d '\r' < $L | while read N KIND; do
  [ -z "$N" ] && continue
  [ -f $CO/_done_$N ] && continue
  mkdir $CL/$N 2>/dev/null || continue
  rm -rf $T/$N; cp -a /full/tree3/$N $T/$N < /dev/null
  cp /full/iso/$N.vtp $ISO/$N.vtp.tmp && mv $ISO/$N.vtp.tmp $ISO/$N.vtp
  if [ "$KIND" = fold ]; then
    python3 $FF $T/$N /s3/fold/$N.json < /dev/null > $LG/items/final.$N.fold.log 2>&1
    echo "$(date '+%F %T') fold $N rc $? $(grep -h '"result"' /s3/fold/$N.json | head -1)" >> $LG/final_other.batch.log
    rm -f $CO/$N.json
    PARTS=item1,route,coverage,folds,lumen,va,orient,loader,geom
  else
    cp /full/checks4/$N.json $CO/$N.json
    PARTS=coverage,folds
  fi
  python3 $CK $N --tree $T --iso $ISO --out $CO --parts $PARTS < /dev/null > $LG/items/final.$N.check.log 2>&1
  grep -q '^GATES' $LG/items/final.$N.check.log && touch $CO/_done_$N
  echo "$(date '+%F %T') check $N ($KIND) $(grep '^GATES' $LG/items/final.$N.check.log | cut -c1-400)" >> $LG/final_other.batch.log
done
