#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_chain.sh (= /sp/v5_va/scaleup_song_chain.sh in the build containers)
# scaleup_song_chain.sh <fetch|ts|art|trace|grade> <case> [case ...]   (scaleup_lane_song, 2026-10-02; Git Bash on the host)
# Detached per-group loop (modelled on the pilot's gpuchain / TS loop / post loop): per case, in order, wait for the
# prerequisite of the group, then run scaleup_song_run.sh <group> <case> (idempotent; a rerun skips finished stages).
#   fetch : no wait
#   ts    : qc_va_check.json exists (fqc + fvac finished: side files complete)
#   art   : TS per-class masks (totalseg/segmentations/vertebrae_C1 + skull) and the fetch done-marker (meta.json)
#   trace : arterial/.arterial_done and totalseg_1p5 masks
#   grade : arterial/tsba/va_extract.json + VMTK done and the GO marker _logs_scaleup/.grade_go (written when the post
#           pipeline is settled: v2 from notes/SCALEUP_PIPELINE.md once state/scaleup_fixes.md says complete)
# A case with rejected_<case> / fetch_failed_<case> is skipped. Wait cap per case: 8 h (then the case is skipped, logged).
# Log: _logs_scaleup/chain_<group>.out (nohup). Done-marker per group and case: _logs_scaleup/chain_<group>_done_<case>.
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
SRC_H=/d/Arjun/neve/carotid_data/va_sources/thin_slice/song
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song
LOGD=$DER_H/_logs_scaleup
mkdir -p "$LOGD"
group=$1; shift
echo "== $(date '+%F %T') chain $group start (pid $$): $*"
for c in "$@"; do
  [ -f "$LOGD/chain_${group}_done_$c" ] && { echo "$c: $group done before"; continue; }
  n=0
  while :; do
    { [ -f "$LOGD/rejected_$c" ] || [ -f "$LOGD/fetch_failed_$c" ]; } && { echo "$c: rejected / fetch failed: skip"; continue 2; }
    case $group in
      fetch) break;;
      ts) [ -f "$SRC_H/$c/qc_va_check.json" ] && break;;
      art) [ -f "$DER_H/$c/totalseg/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$DER_H/$c/totalseg/segmentations/skull.nii.gz" ] && [ -f "$LOGD/fetch_done_$c" ] && break;;
      trace) [ -f "$DER_H/$c/arterial/.arterial_done" ] && [ -f "$DER_H/$c/totalseg_1p5/segmentations/vertebrae_C1.nii.gz" ] && break;;
      grade) [ -f "$LOGD/chain_trace_done_$c" ] && [ -f "$LOGD/.grade_go" ] && break;;
      *) echo "unknown group $group"; exit 2;;
    esac
    n=$((n+1)); [ $n -gt 960 ] && { echo "$(date '+%F %T') $c: prerequisite of $group not met after 8 h: skipped"; continue 2; }
    sleep 30
  done
  bash $SP/v5_va/scaleup_song_run.sh $group "$c"
  touch "$LOGD/chain_${group}_done_$c"
done
echo "== $(date '+%F %T') chain $group finished"
