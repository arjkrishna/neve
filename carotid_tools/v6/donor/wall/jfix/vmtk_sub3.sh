#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\jfix\vmtk_sub3.sh (= /sp/v6/wall/jfix/vmtk_sub3.sh in the build containers)
# v6_wall (e): VMTK MISR lines of the j3 subclavian surfaces (vmr_vmtk.py unchanged), inside ONE neve-build-vmtk
#   bash vmtk_sub3.sh [root (default /v6w/subclavian_j3)]   skips done lines
V=${1:-/v6w/subclavian_j3}
for DC in $V/*/; do
  u=$(basename $DC)
  [ -f $DC/.seg_done ] || { echo "$u: no seg"; continue; }
  for v in rsub lsub; do
    if [ -f $DC/${v}_stageA.vtp ] && [ -f $DC/${v}_seeds.json ] && [ ! -f $DC/${v}_stageA_vmtk.json ]; then
      timeout 900 python3 /sp/v5_va/vmr_vmtk.py $DC/${v}_stageA.vtp $DC/${v}_seeds.json $DC/${v}_stageA_vmtk.json 2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing" | tail -2
      python3 -c "import json; d=json.load(open('$DC/${v}_stageA_vmtk.json')); print('$u $v', round(d['length_mm'],1), d['n'])" 2>/dev/null || echo "VMTK FAILED $u $v"
    fi
  done
done
echo VMTK_SUB3_DONE
