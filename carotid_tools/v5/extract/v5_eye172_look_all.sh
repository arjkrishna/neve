#!/bin/sh
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_eye172_look_all.sh (= /sp/v5_va/v5_eye172_look_all.sh in the build containers)
# v5_eye172_look_all.sh -- idempotent: one look PNG per side (skips existing). Runs INSIDE neve-build-stagea.
set -u
while read src case side ext qcf; do
  [ -z "$src" ] && continue
  out=/sp/v5_va/results/v5_eye172/look_${case}_${side}.png
  if [ -f "$out" ]; then echo "skip $case $side"; continue; fi
  python3 /sp/v5_va/v5_eye172_look.py $src $case $side $ext $qcf || echo "FAIL $case $side"
done <<LIST
ia172 ia172_039 R tsba qc.json
ia172 ia172_039 L tsba qc.json
ia172 ia172_015 R tsba qc.json
ia172 ia172_190 R tsba qc.json
ia172 ia172_190 L tsba qc.json
ia172 ia172_141 R tsba qc.json
ia172 ia172_172 L tsba qc.json
song AHMU1218138 R tsba_v1r qc_tsba_v1r.json
song AHMU1218138 L tsba qc.json
LIST
echo ALLDONE
