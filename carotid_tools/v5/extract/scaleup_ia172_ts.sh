#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_ts.sh (= /sp/v5_va/scaleup_ia172_ts.sh in the build containers)
# scaleup_ia172_ts.sh <case> [<case> ...] = pilot_ia172_ts.sh (copied 2026-10-02, scaleup_lane_ia172) with TWO changes:
# --cpus 6 (the scale-up limit; was 8) and the container name scaleup_ia172_fast_<pid>. Otherwise unchanged:
#
# Git Bash on the host. Quick-look TotalSegmentator 2.18.0 "total --fast" (3 mm model, Dataset297) on the CPU, on the
# 1.5 mm RAS copy written by pilot_ia172_convert.py:
#   in : D:\Arjun\neve\carotid_data\va_sources\derived\thin_slice\ia172\<case>\ct_1p5.nii.gz
#   out: ...\<case>\ts_fast_1p5.nii.gz (multilabel, --ml) and ...\<case>\ts_fast.log
# These masks are ONLY for the fetch QC (CCA / subclavian / vein HU, arch and skull-base coverage, MIP z range).
# They are not the masks the study tools want (those are total / total_v3 on the CTA grid, tag_totalseg.md).
# No GPU (no lock needed). Private weights home on D: (..\ia172\_ts_home, config.json send_usage_stats=false written
# before the first run) so the shared totalseg_weights cache and its config.json are not touched.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172/_ts_home
mkdir -p "$H"
if [ ! -f "$H/config.json" ]; then
  printf '{\n    "totalseg_id": "totalseg_neve_local_pilot_ia172",\n    "send_usage_stats": false,\n    "prediction_counter": 0,\n    "statistics_disclaimer_shown": true\n}\n' > "$H/config.json"
fi
CASES="$*"
docker run --rm --name scaleup_ia172_fast_$$ --cpus 6 --memory 12g --shm-size 2g \
  -e TOTALSEG_HOME_DIR=/ts_home -e PYTHONUNBUFFERED=1 \
  -v "D:\\Arjun\\neve\\carotid_data\\va_sources\\derived\\thin_slice\\ia172\\_ts_home:/ts_home" \
  -v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" \
  neve-totalseg-gpu bash -c "
for c in $CASES; do
  D=/repo/carotid_data/va_sources/derived/thin_slice/ia172/\$c
  if [ -f \$D/ts_fast_1p5.nii.gz ]; then echo \"\$c: exists\"; continue; fi
  t=\$(date +%s)
  /opt/conda/bin/TotalSegmentator -i \$D/ct_1p5.nii.gz -o \$D/ts_fast_1p5.nii.gz --ml --fast --device cpu \
     --nr_thr_resamp 4 --nr_thr_saving 2 > \$D/ts_fast.log 2>&1
  echo \"\$c: rc \$? \$(( \$(date +%s) - t )) s\"
done
/opt/conda/bin/python -c \"
import json
from totalsegmentator.map_to_binary import class_map
json.dump(class_map['total'], open('/repo/carotid_data/va_sources/derived/thin_slice/ia172/_ts_total_labels.json', 'w'), indent=0)
\"
"
