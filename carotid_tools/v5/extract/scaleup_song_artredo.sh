#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_artredo.sh (= /sp/v5_va/scaleup_song_artredo.sh in the build containers)
# scaleup_song_artredo.sh <case>   (scaleup_lane_song instance 3, 2026-10-02 09:58; Git Bash on the host; detached via nohup)
# Redo of a Song case whose Arterial run CRASHED on a NaN FOV centre (args fallback branch, TS trachea absent from the
# lowest 60 mm; first case: AHMU1218065, Arterial exit 1 at 07:19 in prepare_input, so no segmentation, no trace, and the
# 09:13 v2 grade ran on nothing: qc rc 1 "no tsba trace").
# Steps (each idempotent, markers in _logs_scaleup):
#   1. move the failed arterial/ to arterial_failed_nanfov/ (kept as the record), args with scaleup_song_args.py (pilot
#      args + the trachea-window diff), light container as the driver's args stage
#   2. scaleup_song_run.sh arterial <case>   (unchanged driver stage: GPU lock, watchdog, air1000 input deleted after)
#   3. scaleup_song_run.sh trace <case>      (extract tsba + ts, VMTK; CPU lock)
#   4. pipeline v2 grade: V2_FORCE = all per-case stages (the 09:13 run produced no qc/select, so this is the FIRST real
#      grade of the case, same v2 rules), via scaleup_song_grade.sh; then step 2 scaleup_ctseg_fallback.sh
# Done-marker _logs_scaleup/artredo_done_<case>.
c=$1
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song
LOGD=$DER_H/_logs_scaleup
d=$DER_H/$c
[ -f "$LOGD/artredo_done_$c" ] && { echo "$c: artredo done before"; exit 0; }
echo "== $(date '+%F %T') artredo $c start (pid $$)"
if [ ! -f "$d/arterial/args_redo.json" ]; then
  if [ -d "$d/arterial" ] && [ ! -f "$d/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ]; then
    rm -rf "$d/arterial_failed_nanfov"; mv "$d/arterial" "$d/arterial_failed_nanfov"
  fi
  mkdir -p "$d/arterial"
  docker run --rm --name "scaleup_song_argsredo_$$" --cpus 4 --memory 8g -e PYTHONUNBUFFERED=1 -e TS_DIRNAME=totalseg \
    -v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data" \
    -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools" -w /repo neve-build-stagea \
    python3 /sp/v5_va/scaleup_song_args.py "$c" 2>&1 | grep -v "CUDA\|NVIDIA\|license\|^==\|^$\|By pulling\|governed\|cloud-native\|Container"
  grep -q "nan" "$d/arterial/args_run.txt" && { echo "$c: args still NaN -> stop"; exit 1; }
  printf '{"case": "%s", "args_script": "scaleup_song_args.py", "why": "pilot args fallback gave --fov-center=nan,nan (TS trachea absent from the lowest 60 mm); failed run kept in arterial_failed_nanfov/", "when": "%s"}\n' "$c" "$(date '+%F %T')" > "$d/arterial/args_redo.json"
  echo "args: $(cat "$d/arterial/args_run.txt")"
fi
bash $SP/v5_va/scaleup_song_run.sh arterial "$c"
[ -f "$d/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ] || { echo "$c: Arterial wrote no segmentation -> stop"; exit 1; }
bash $SP/v5_va/scaleup_song_run.sh trace "$c"
touch "$LOGD/chain_trace_done_$c"
rm -f "$LOGD/grade_done_$c" "$LOGD/chain_grade2_done_$c"
V2_FORCE="qc select rescue gross ctseg measure protrusion fig" bash $SP/v5_va/scaleup_song_grade.sh "$c"
[ -f "$LOGD/grade_done_$c" ] && touch "$LOGD/chain_grade2_done_$c"
bash $SP/v5_va/scaleup_ctseg_fallback.sh song "$c"
touch "$LOGD/artredo_done_$c" "$LOGD/fallback_done_$c"
echo "== $(date '+%F %T') artredo $c finished"
