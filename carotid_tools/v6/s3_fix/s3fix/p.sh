#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\p.sh (= /sp/v6/s3fix/p.sh in the build containers)
# foreground python3 in eve-training-fixed, filtering the NVIDIA banner
bash /c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v6/s3fix/dr_s3.sh ${IMG:-eve-training-fixed} - python3 "$@" 2>&1 | sed -e '/^=*$/d' -e '/== CUDA ==/d' -e '/CUDA Version/d' -e '/NVIDIA/d' -e '/governed by/d' -e '/By pulling/d' -e '/https:\/\/developer.nvidia/d' -e '/A copy of this license/d' -e '/Use the NVIDIA/d' -e '/docs.nvidia.com/d' -e '/Container image Copyright/d' -e '/^WARNING: The NVIDIA/d'
