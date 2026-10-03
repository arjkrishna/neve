#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\worker13w.sh (= /sp/v5_va/route13/worker13w.sh in the build containers)
# v5_route13: like worker13.sh, but waits (up to 3 h per anatomy) for the graft record /sp/v5_va/route13/graft/<TAG>/<name>.json
# of the moved variant before baking route13/moved_<TAG>/<name>.   VARIANT=moved_ext TAG=ext OWNERSHIP=standard bash worker13w.sh <list>
for N in $(tr -d '\r' < $1); do
  t=0
  until [ -f /sp/v5_va/route13/graft/$TAG/$N.json ] || [ $t -ge 10800 ]; do sleep 30; t=$((t+30)); done
  echo $N > /tmp/one_$N.txt
  bash /sp/v5_va/route13/worker13.sh /tmp/one_$N.txt
done
