#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\batch_cov.sh (= /sp/v6/s3fix/batch_cov.sh in the build containers)
# idempotent: checks_v6 (dev 1.6) parts coverage,folds on names in $1 against tree $2 / iso $3 -> out $4 (claims by mkdir)
NL=$1; TREE=$2; ISO=$3; OUT=$4; CK=${CK:-/sp/v6/checks/checks_v6_1p6_dev.py}
mkdir -p $OUT/_claim $OUT/logs
for n in $(tr -d '\r' < $NL); do
  [ -f $OUT/_done_$n ] && continue
  mkdir $OUT/_claim/$n 2>/dev/null || continue
  python3 $CK $n --tree $TREE --iso $ISO --out $OUT --parts coverage,folds > $OUT/logs/$n.log 2>&1 && grep -q '^GATES' $OUT/logs/$n.log && touch $OUT/_done_$n
  echo "$(date +%T) $n $(tail -1 $OUT/logs/$n.log | cut -c1-160)" >> $OUT/batch.log
done
