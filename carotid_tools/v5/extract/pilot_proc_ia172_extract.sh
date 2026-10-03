#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_extract.sh (= /sp/v5_va/pilot_proc_ia172_extract.sh in the build containers)
# pilot_proc_ia172_extract.sh <case> <mode>   (inside neve-build-stagea) -- the VA traces of one 172-series pilot case.
# mode tsba : pilot_proc_ia172_art_extract.py (= art_extract + the basilar-seed fix; diff in its docstring),
#             --ts <der>/totalseg_1p5 (TS total_v3 masks, 1.5 mm RAS copy: tag_totalseg's recommendation, the same QC
#             within ~1 mm of the CTA-grid dir at ~1/13 of the memory) -> <der>/arterial/tsba   (PRIMARY)
# mode ts   : art_extract.py UNCHANGED, same --ts -> <der>/arterial/ts   (reference: what the study tool does)
D=/repo/carotid_data/va_sources/derived/thin_slice/ia172/$1
R=$D/arterial
case $2 in
  tsba) python3 /sp/v5_va/pilot_proc_ia172_art_extract.py $R/runs/native $R/tsba --ts $D/totalseg_1p5 --tag ${1}_tsba ;;
  ts)   python3 /sp/v5_va/art_extract.py $R/runs/native $R/ts --ts $D/totalseg_1p5 --tag ${1}_ts ;;
esac
