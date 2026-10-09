#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\vmtk_batch.sh (= /sp/v6/wall/vmtk_batch.sh in the build containers)
# v6_wall: VMTK MISR centrelines of the new stage-A VA surfaces (vmr_vmtk.py unchanged, as scaleup_v2_post.sh stage ctseg)
#   bash vmtk_batch.sh <variant_dir> [unit ...]   (inside ONE neve-build-vmtk container; skips done units)
V=$1; shift
UNITS="$@"; [ -z "$UNITS" ] && UNITS=$(ls $V)
for u in $UNITS; do
  DC=$V/$u
  [ -f $DC/.seg_done ] || { echo "$u: no seg"; continue; }
  for v in rva lva; do
    if [ -f $DC/${v}_stageA.vtp ] && [ -f $DC/${v}_seeds.json ] && { [ ! -f $DC/${v}_stageA_vmtk.json ] || [ $DC/${v}_stageA.vtp -nt $DC/${v}_stageA_vmtk.json ]; }; then
      timeout 900 python3 /sp/v5_va/vmr_vmtk.py $DC/${v}_stageA.vtp $DC/${v}_seeds.json $DC/${v}_stageA_vmtk.json 2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing" | tail -2
      python3 -c "import json; d=json.load(open('$DC/${v}_stageA_vmtk.json')); print('$u $v', round(d['length_mm'],1), d['n'])" 2>/dev/null || echo "VMTK FAILED $u $v"
    fi
  done
done
echo VMTK_BATCH_DONE
