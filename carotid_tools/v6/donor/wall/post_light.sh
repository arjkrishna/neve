#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\post_light.sh (= /sp/v6/wall/post_light.sh in the build containers)
# v6_wall: light analyses after the seg runs (inside ONE neve-build-stagea container). Each step logs to /v6w/post/<step>.log
mkdir -p /v6w/post
cd /sp/v6/wall
for step in "$@"; do
  case $step in
    regress)  python3 regress.py /v6w/seg/v5repro all ;;
    gate_half) python3 gate_ct.py /v6w/seg/half half ;;
    gate_plateau_g) python3 gate_ct.py /v6w/seg/plateau_g plateau_g ;;
    gate_plateau) python3 gate_ct.py /v6w/seg/plateau plateau ;;
    c3) python3 c3_inspect.py plateau ;;
    surf) python3 surf_stats.py half,plateau_g,plateau ;;
    lumen_v5) python3 va_lumen_poly.py --root /repo/carotid_data/anatomies_v5 --all --out /v6w/lumen/va_lumen_poly_v5.json --workers 3 ;;
    c4) python3 c4_check.py /v6w/lumen/va_lumen_poly_v5.json ;;
    *) echo "unknown step $step" ;;
  esac > /v6w/post/$step.log 2>&1
  echo "$step rc $? $(date '+%T')" >> /v6w/post/steps.log
done
echo ALL_DONE >> /v6w/post/steps.log
