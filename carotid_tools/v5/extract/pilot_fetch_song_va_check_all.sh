#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_fetch_song_va_check_all.sh (= /sp/v5_va/pilot_fetch_song_va_check_all.sh in the build containers)
# pilot_fetch_song_va_check_all.sh (pilot_fetch_song, instance 3, 2026-10-01 ~20:25 EDT)
# Idempotent batch: runs pilot_fetch_song_va_check2.py (column frame) once per case; skips a case whose qc_va_check.png AND .json exist
# (done-marker). Log: /sp/v5_va/results/pilot_fetch_song_va_check.log. Run inside neve-build-stagea, detached:
#   docker run -d --name pilot_fetch_song_vacheck -v D:\Arjun\neve:/repo -v <scratchpad>:/sp neve-build-stagea \
#     bash /sp/v5_va/pilot_fetch_song_va_check_all.sh [CASE ...]
LOG=/sp/v5_va/results/pilot_fetch_song_va_check.log
IMG=/repo/carotid_data/va_sources/thin_slice/song
CASES="$@"
[ -z "$CASES" ] && CASES="AHMU1218008 AHMU1218032 AHMU1218044 AHMU1218090 AHMU1218094 AHMU1218100 AHMU1218106 AHMU1218109 AHMU1218113 AHMU1218141"
echo "== va_check_all start $(date) : $CASES" >> $LOG
for c in $CASES; do
  if [ -s $IMG/$c/qc_va_check.png ] && [ -s $IMG/$c/qc_va_check.json ]; then echo "$c skip (done)" >> $LOG; continue; fi
  python3 /sp/v5_va/pilot_fetch_song_va_check2.py $c >> $LOG 2>&1 || echo "$c FAILED rc=$?" >> $LOG
done
echo "va_check_all done $(date)" >> $LOG
