#!/usr/bin/env bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_qc_all.sh (= /sp/v5_va/pilot_fetch_song_qc_all.sh in the build containers)
# pilot_fetch_song_qc_all.sh: run pilot_fetch_song_qc.py on each case, ONE container at a time, once its ct.nii.gz
# is fetched and CRC-verified (fetch.json crc_ok true). Cases whose side files already exist run with NO_SIDE=1
# (the side-file code has not changed since they were written; qc_pass1.json keeps their record). Host (Git Bash).
# usage: bash pilot_fetch_song_qc_all.sh CASE [CASE ...]
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IMG=/d/Arjun/neve/carotid_data/va_sources/thin_slice/song
DER=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song
LOG=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad/v5_va/results/pilot_fetch_song_qc.log
echo "== pilot_fetch_song_qc_all $(date) : $*" >> "$LOG"
for c in "$@"; do
  n=0
  until [ -f "$IMG/$c/fetch.json" ] && grep -q '"crc_ok": true' "$IMG/$c/fetch.json"; do
    sleep 20; n=$((n+1)); [ $n -gt 180 ] && { echo "$c: no verified ct.nii.gz after 60 min" >> "$LOG"; continue 2; }
  done
  NS=0; E=""
  if [ -f "$DER/$c/ct_bg-100.nii.gz" ] && [ -f "$DER/$c/ct_negzero-100.nii.gz" ]; then NS=1; E="-e NO_SIDE=1"; fi
  docker run --rm --name "pilot_fetch_song_qc_$c" --cpus 6 --memory 12g $E -e PYTHONUNBUFFERED=1 \
    -v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" neve-build-stagea python3 /sp/v5_va/pilot_fetch_song_qc.py "$c" 2>&1 \
    | grep -vE "CUDA|NVIDIA|nvidia|Container image|license|^$|====|Driver|Toolkit|By pulling|governed" \
    | sed "s/\$/ no_side=$NS/" >> "$LOG"
done
echo "qc_all done $*" >> "$LOG"
