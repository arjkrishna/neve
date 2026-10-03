#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_refqc.sh (= /sp/v5_va/scaleup_song_refqc.sh in the build containers)
# scaleup_song_refqc.sh <case> <arch_top_z> [aorta_hu]  (scaleup_lane_song, 2026-10-02; Git Bash on the host)
# (04:27) optional aorta_hu: then scaleup_song_fetch_qc.py (pilot qc + AORTA_HU fallback when the track found no limb) is used.
# Re-run the fetch QC crops of one case with a corrected arch top (the pilot's ARCH_TOP_Z by-eye override; here the
# value comes from the TS aorta top, scaleup_song_archtop.py, checked on the MIPs): pilot_fetch_song_qc.py with NO_SIDE=1
# (side files kept: they do not depend on the arch top) + ARCH_TOP_Z, then va_check2 and va_zoom (old PNG/JSON moved
# to *_autoarch.*), then picks + meta. Under the lane's CPU lock (_logs_scaleup/.cpu_lock). Log _logs_scaleup/<case>_refqc.log.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SRC_H=/d/Arjun/neve/carotid_data/va_sources/thin_slice/song
LOGD=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song/_logs_scaleup
CLOCK=$LOGD/.cpu_lock
c=$1; z=$2; ah=${3:-}; QCS=pilot_fetch_song_qc.py; [ -n "$ah" ] && QCS=scaleup_song_fetch_qc.py
MNT=(-v "D:\Arjun\neve:/repo" -v "$SPW:/sp")
{
echo "== $(date '+%F %T') refqc $c ARCH_TOP_Z=$z AORTA_HU=${ah:-none} ($QCS)"
until mkdir "$CLOCK" 2>/dev/null; do
  op=$(awk '{print $2}' "$CLOCK/owner" 2>/dev/null)
  if [ -n "$op" ] && [ "$op" -eq "$op" ] 2>/dev/null && ! kill -0 "$op" 2>/dev/null; then rm -rf "$CLOCK"; continue; fi
  sleep 10
done
echo "refqc $$ $(date +%s)" > "$CLOCK/owner"
d=$SRC_H/$c
for f in qc_va_check.png qc_va_check.json qc_va_zoom.png qc_axial.png mip_neck_coronal.png mip_neck_sagittal.png qc_pass1.json; do
  [ -f $d/$f ] && [ ! -f $d/${f%.*}_autoarch.${f##*.} ] && cp $d/$f $d/${f%.*}_autoarch.${f##*.}
done
docker run --rm --name scaleup_song_refqc_$$ --cpus 6 --memory 12g -e NO_SIDE=1 -e ARCH_TOP_Z=$z ${ah:+-e AORTA_HU=$ah} -e PYTHONUNBUFFERED=1 "${MNT[@]}" -w /repo neve-build-stagea \
  bash -c "python3 /sp/v5_va/$QCS $c && python3 /sp/v5_va/pilot_fetch_song_va_check2.py $c && rm -f /repo/carotid_data/va_sources/thin_slice/song/$c/qc_va_zoom.png && python3 /sp/v5_va/pilot_fetch_song_va_zoom.py $c && python3 /sp/v5_va/scaleup_song_fetch_picks.py $c > /dev/null && python3 /sp/v5_va/scaleup_song_fetch_meta.py $c" 2>&1 | grep -v "CUDA\|NVIDIA\|nvidia\|Container\|license\|governed\|^$\|====\|Driver\|Toolkit\|By pulling\|A copy"
rm -rf "$CLOCK"
echo "== $(date '+%F %T') refqc $c done"
} 2>&1 | tee -a $LOGD/${c}_refqc.log
