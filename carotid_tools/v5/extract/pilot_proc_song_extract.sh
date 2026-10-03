#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_extract.sh (= /sp/v5_va/pilot_proc_song_extract.sh in the build containers)
# pilot_proc_song_extract.sh <case> <mode>   (inside neve-build-stagea) -- the VA traces of one Song pilot case.
# = pilot_proc_ia172_extract.sh with the song derived dir and the shared pilot copy /sp/v5_va/pilot_art_extract.py
#   (art_extract + the basilar-seed fix; the diff is in its docstring: find_ba window top z_FM + 40 instead of + 55, and a
#   seed component whose lowest voxel lies > 20 mm above the window bottom is down-weighted x 0.1; midline = TS spinal cord).
# mode tsba : pilot_art_extract.py --ts <der>/totalseg_1p5 (TS total_v3, 1.5 mm RAS copy) -> <der>/arterial/tsba  (PRIMARY)
# mode ts   : art_extract.py UNCHANGED, same --ts -> <der>/arterial/ts   (reference: what the study tool does)
D=/repo/carotid_data/va_sources/derived/thin_slice/song/$1
R=$D/arterial
case $2 in
  tsba) python3 /sp/v5_va/pilot_art_extract.py $R/runs/native $R/tsba --ts $D/totalseg_1p5 --tag ${1}_tsba ;;
  ts)   python3 /sp/v5_va/art_extract.py $R/runs/native $R/ts --ts $D/totalseg_1p5 --tag ${1}_ts ;;
esac
