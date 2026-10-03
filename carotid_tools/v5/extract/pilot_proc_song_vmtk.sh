#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_vmtk.sh (= /sp/v5_va/pilot_proc_song_vmtk.sh in the build containers)
# pilot_proc_song_vmtk.sh = pilot_proc_ia172_vmtk.sh, byte copy apart from this line (pilot_proc_song). <extract_dir> [<extract_dir> ...] inside ONE neve-build-vmtk container
# = art_vmtk.sh (a copy of the AortaSeg pilot's pilot_vmtk.sh) with ONE change: the extraction dirs are given as full paths (art_vmtk.sh hard-codes
# derived/arterial/<tag>). VMTK MISR centerlines of the cut-out VA stage-A surfaces, art_vmtk_one.py UNCHANGED.
export PYTHONUNBUFFERED=1
for D in "$@"; do
  t=$(basename $D)
  for v in rva lva; do
    if [ -f $D/${v}_stageA.vtp ] && [ -f $D/${v}_seeds.json ]; then
      if [ ! -f $D/${v}_stageA_vmtk.json ] || [ $D/${v}_stageA.vtp -nt $D/${v}_stageA_vmtk.json ]; then
        side=R; [ $v == lva ] && side=L
        python3 -c "import json; d=json.load(open('$D/va_extract.json')); json.dump({'points': d['sides']['$side']['trace']['points']}, open('/tmp/${t}_${v}_trace.json','w'))"
        rm -f $D/${v}_stageA_vmtk.json
        timeout 900 python3 /sp/v5_va/art_vmtk_one.py $D/${v}_stageA.vtp $D/${v}_seeds.json /tmp/${t}_${v}_trace.json $D/${v}_stageA_vmtk.json 2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
        python3 -c "import json,sys; d=json.load(open('$D/${v}_stageA_vmtk.json')); print('$t $v VMTK', 'ok' if d['ok'] else 'NOT ok', round(d['length_mm'],1), d['note'])" 2>/dev/null || echo "VMTK FAILED $t $v (no JSON)"
      fi
    fi
  done
done
