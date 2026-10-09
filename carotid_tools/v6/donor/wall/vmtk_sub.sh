#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\vmtk_sub.sh (= /sp/v6/wall/vmtk_sub.sh in the build containers)
# v6_wall (e): VMTK MISR lines of the requested subclavian surfaces (vmr_vmtk.py unchanged), inside ONE neve-build-vmtk
#   bash vmtk_sub.sh [unit ...]   (default: every /v6w/subclavian/<unit> with .seg_done; skips done lines)
V=/v6w/subclavian
UNITS="$@"; [ -z "$UNITS" ] && UNITS=$(ls $V | grep -v '^req$')
for u in $UNITS; do
  DC=$V/$u
  [ -f $DC/.seg_done ] || { echo "$u: no seg"; continue; }
  for v in rsub lsub; do
    if [ -f $DC/${v}_stageA.vtp ] && [ -f $DC/${v}_seeds.json ] && [ ! -f $DC/${v}_stageA_vmtk.json ]; then
      timeout 900 python3 /sp/v5_va/vmr_vmtk.py $DC/${v}_stageA.vtp $DC/${v}_seeds.json $DC/${v}_stageA_vmtk.json 2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing" | tail -2
      python3 -c "import json; d=json.load(open('$DC/${v}_stageA_vmtk.json')); print('$u $v', round(d['length_mm'],1), d['n'])" 2>/dev/null || echo "VMTK FAILED $u $v"
    fi
  done
done
echo VMTK_SUB_DONE
