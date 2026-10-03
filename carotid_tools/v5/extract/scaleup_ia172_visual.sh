#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_visual.sh (= /sp/v5_va/scaleup_ia172_visual.sh in the build containers)
# scaleup_ia172_visual.sh <case> <accept|reject> '<json object without verdict>'  -- record my by-eye verdict
# (scaleup_ia172_visual_set.py in neve-build-stagea) and the chain marker visual_ok_<case> / rejected_<case>.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
L=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172/_logs_scaleup
c=$1; v=$2; j=$3
j="${j%\}}, \"verdict\": \"$v\"}"
docker run --rm -v "D:\Arjun\neve:/repo" -v "$SPW:/sp" neve-build-stagea python3 /sp/v5_va/scaleup_ia172_visual_set.py "$c" "$j" 2>&1 | grep -v "CUDA\|NVIDIA\|^=\|^$\|icense\|Container\|governed\|pulling\|copy of\|Driver\|nvidia"
if [ "$v" = accept ]; then touch "$L/visual_ok_$c"; else echo "rejected by eye $(date '+%F %T'): $j" > "$L/rejected_$c"; fi
