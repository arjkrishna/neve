#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\vmr_vmtk_all.sh (= /sp/v5_va/vmr_vmtk_all.sh in the build containers)
# all VMTK centerlines for the vmr agent, inside ONE neve-build-vmtk container
D=/repo/carotid_data/va_sources/derived/vmr0248
TAG=${1:-ct_seg2}
for v in rva lva; do
  for k in stageA iso; do
    python3 /sp/v5_va/vmr_vmtk.py $D/$TAG/${v}_$k.vtp $D/$TAG/${v}_seeds.json $D/$TAG/${v}_${k}_vmtk.json
  done
done
if [ "$2" == "lofted" ]; then
for side in right left; do
  python3 /sp/v5_va/vmr_vmtk.py $D/lofted/wall_${side}_vertebral_mm.vtp $D/lofted/${side}_seeds.json $D/lofted/${side}_vmtk.json --cap
done
fi
