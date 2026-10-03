#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_run5.sh (= /sp/v5_va/pilot_proc_song_run5.sh in the build containers)
# pilot_proc_song_run5.sh = pilot_proc_song_run4.sh + (23:40) ctseg runs when qc/qc.json OR qc/qc_ts.json (the unchanged tool trace, see qc2 select.json) has a donor-passing side.
# pilot_proc_song_run4.sh = pilot_proc_song_run3.sh + (23:12) env passthroughs for side runs: PILOT_KEEP_CONNECTED (ctseg;
#   0 = plain ct_seg4 without ia172's diff 8) and PILOT_CT (measure stage: the edge script's CT = shipped | negzero | bg).
#   Used with PILOT_SEG_TAG / PILOT_SEG_TAGS for the ct_seg4_negzero (clipping comparison) and ct_seg4_nofix runs.
#   (run3.sh was executing when this was written.)
# pilot_proc_song_run3.sh = pilot_proc_song_run2.sh + (22:35) Arterial input variants for the Song clipping problem:
#   ART_IN=air1000 (ct_negzero_air1000.nii.gz, pilot_proc_song_rewindow.py) and ART_IN=tex (ct_tex.nii.gz,
#   pilot_proc_song_rewindow2.py), each made on demand (light container) before the GPU lock is taken; for every ART_IN
#   other than shipped an existing runs/native is moved to runs/native_prev_<time> first. The re-windowed file is the
#   ARTERIAL INPUT ONLY; qc / ct_seg / edge read the shipped ct.nii.gz. (run2.sh was executing when this was written.)
# pilot_proc_song_run2.sh = pilot_proc_song_run.sh + ONE change (21:25): an Arterial WATCHDOG. run_arterial.sh runs in the
#   background; after ART_TIMEOUT s (default 1200) its container arterial_native_<pid> is stopped IF the segmentation
#   (extracranial_vessels/segmentation.nii.gz) is already written (the 172-series run ia172_061 sat > 25 min in vmtk's
#   network extraction after "can't reconstruct new profile", holding the GPU lock), so art_extract can trace the binary
#   mask (its no-label path); without a segmentation it is stopped after 2 x ART_TIMEOUT. Recorded in timing.jsonl and
#   arterial/watchdog.json. (run.sh itself is left alone: it was executing when this was written.)
# pilot_proc_song_run.sh <stage|group> [case ...]   (Git Bash on the host; agent pilot_proc_song, 2026-10-01)
#
# The thin-slice VA donor pipeline on the 10 Song et al. 2024 pilot cases (figshare 26965450, CC BY 4.0).
#   images  D:\Arjun\neve\carotid_data\va_sources\thin_slice\song\<case>\ct.nii.gz  (float64 HU, LAS, native grid,
#           every voxel <= 50 HU stored as 0)
#   derived D:\Arjun\neve\carotid_data\va_sources\derived\thin_slice\song\<case>\
# = pilot_proc_ia172_run2.sh (the 172-series pilot agent's driver, read-only) with these changes:
#   * song paths / case list / container + lock names (pilot_proc_song_*);
#   * tsprep stage (new): the TotalSegmentator input = an exact int16 copy of $TS_INPUT (default negzero =
#     ct_negzero-100.nii.gz; see the notes for the comparison with the shipped image) + ct.nii.gz hard link to the SHIPPED CT;
#   * no clip stage: Arterial runs on the SHIPPED ct.nii.gz first (brief); ART_IN=bg runs it on ct_bg-100.nii.gz instead
#     (only for a case where the shipped run failed; the failed run is kept as runs/native_shipped_failed);
#   * ts stages honour TS_DIRNAME (default totalseg) for the input comparison.
# Every stage is idempotent (skips work whose output / done marker exists). Per case and stage: wall time ->
# _logs_proc/timing.jsonl, output -> _logs_proc/<case>_<stage>.log.
#
# stages: tsprep tsgpu tscpu args arterial extract vmtk qc gross ctseg measure fig collect
# groups: gpu = tsprep tsgpu tscpu args arterial ; post = extract vmtk qc gross ctseg measure fig collect
# GPU LOCK: mkdir <sp>/v5_va/.gpu_lock (+ owner), removed as soon as the container exits (trap on failure too).
# CPU LOCK (this agent only; at most one heavy CPU container at a time): _logs_proc/.cpu_lock
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOCK=$SP/v5_va/.gpu_lock
SRC_H=/d/Arjun/neve/carotid_data/va_sources/thin_slice/song
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song
SRC_C=/repo/carotid_data/va_sources/thin_slice/song
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/song
LOGD=$DER_H/_logs_proc
CLOCK=$LOGD/.cpu_lock
TSN=${TS_DIRNAME:-totalseg}
mkdir -p "$LOGD"
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'

stage=$1; shift
cases=("$@")
[ ${#cases[@]} -eq 0 ] && cases=(AHMU1218100 AHMU1218109 AHMU1218044 AHMU1218032 AHMU1218113 AHMU1218141 AHMU1218106 AHMU1218090 AHMU1218008 AHMU1218094)

gpu_lock() {   # $1 = what
  until mkdir "$LOCK" 2>/dev/null; do
    if [ -f "$LOCK/owner" ]; then   # stale: older than 60 min and no GPU container running
      t=$(awk '{print $NF}' "$LOCK/owner" 2>/dev/null); now=$(date +%s)
      if [ -n "$t" ] && [ "$t" -eq "$t" ] 2>/dev/null && [ $((now - t)) -gt 3600 ] && \
         ! docker ps --format '{{.Image}}' | grep -qE 'neve-arterial-gpu|neve-totalseg-gpu'; then
        echo "$(date '+%F %T') removing stale GPU lock: $(cat "$LOCK/owner")" >> "$LOGD/gpu_lock.log"; rm -rf "$LOCK"; continue
      fi
    fi
    sleep 30
  done
  echo "pilot_proc_song $1 $(date +%s)" > "$LOCK/owner"
  echo "$(date '+%F %T') acquired: $1 (run.sh pid $$)" >> "$LOGD/gpu_lock.log"
  trap 'rm -rf "$LOCK"; echo "$(date "+%F %T") released (trap)" >> "$LOGD/gpu_lock.log"' EXIT INT TERM
}
gpu_unlock() { rm -rf "$LOCK"; trap - EXIT INT TERM; echo "$(date '+%F %T') released" >> "$LOGD/gpu_lock.log"; }
cpu_lock() { until mkdir "$CLOCK" 2>/dev/null; do sleep 10; done; echo "$1 $$ $(date +%s)" > "$CLOCK/owner"; }
cpu_unlock() { rm -rf "$CLOCK"; }

timing() {  # case stage seconds rc note
  printf '{"case": "%s", "stage": "%s", "seconds": %s, "rc": %s, "note": "%s", "end": "%s", "by": "run.sh"}\n' "$1" "$2" "$3" "$4" "$5" "$(date '+%F %T')" >> "$LOGD/timing.jsonl"
}

light_run() {  # name image cpus mem cmd...   (light containers, no CPU lock)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  docker run --rm --name "pilot_proc_song_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 -e TS_DIRNAME=$TSN \
    "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  return ${PIPESTATUS[0]}
}

cpu_run() {  # name image cpus mem cmd...   (one heavy CPU container at a time for this agent)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  cpu_lock "$nm"
  docker run --rm --name "pilot_proc_song_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 -e TS_DIRNAME=$TSN \
    ${PILOT_CASE_ENV:+-e PILOT_CASE=$PILOT_CASE_ENV} "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  local rc=${PIPESTATUS[0]}
  cpu_unlock
  return $rc
}

stage_tsprep() {
  c=$1; d=$DER_H/$c/$TSN; mkdir -p "$d"
  [ -f "$d/segmentations/vertebrae_C1.nii.gz" ] || [ -f "$d/.ts_ml_done" ] || [ -f "$d/ct_ts_input.nii.gz" ] && { echo "tsprep: nothing to do"; return; }
  [ -f "$d/ct.nii.gz" ] || ln "$SRC_H/$c/ct.nii.gz" "$d/ct.nii.gz"
  t0=$(date +%s)
  light_run tsprep neve-build-stagea 2 8g python3 /sp/v5_va/pilot_proc_song_tsprep.py "$c" "${TS_INPUT:-negzero}"
  timing "$c" "tsprep_$TSN" $(( $(date +%s) - t0 )) $? "TS input ${TS_INPUT:-negzero} as int16"
}

stage_tsgpu() {
  c=$1; d=$DER_H/$c/$TSN
  [ -f "$d/segmentations/vertebrae_C1.nii.gz" ] && { echo "per-class masks exist"; return; }
  [ -f "$d/.ts_ml_done" ] && { echo "multilabel exists"; return; }
  [ -f "$d/ct_ts_input.nii.gz" ] || { echo "no TS input for $c"; return; }
  gpu_lock "ts $c $TSN"
  t0=$(date +%s)
  bash $SP/v5_va/pilot_proc_song_tsrun.sh "$DER_C/$c/$TSN/ct_ts_input.nii.gz" "$DER_C/$c/$TSN/segmentations_ml.nii.gz" "${c}_${TSN}_total_v3" --task total_v3 --ml
  rc=$?
  gpu_unlock
  timing "$c" "ts_total_v3_ml_$TSN" $(( $(date +%s) - t0 )) $rc "TotalSegmentator total_v3 on the GPU (CTA grid, --ml)"
  printf '{"task": "total_v3", "tool": "TotalSegmentator 2.18.0 (neve-totalseg-gpu)", "grid": "the CTA grid (nearest-neighbour write-back of the 1.5 mm prediction)", "writer": "--ml multilabel, split by pilot_proc_song_tssplit.py", "input": "%s (see ts_input.json)", "why_not_total": "tag_totalseg.md: total_v3 C1 Dice 0.98 vs 0.92, CCA 0.92-0.95 vs 0.85-0.89, skull 0.93 vs 0.81-0.83; the V3-loop gate fails a real loop on total masks; tag_calibration gates set on total_v3", "rc": %s}\n' "${TS_INPUT:-negzero}" $rc > "$d/task.json"
  [ $rc -eq 0 ] && [ -f "$d/segmentations_ml.nii.gz" ] && touch "$d/.ts_ml_done"
}

stage_tscpu() {
  c=$1; d=$DER_H/$c/$TSN
  if [ ! -f "$d/segmentations/vertebrae_C1.nii.gz" ]; then
    [ -f "$d/.ts_ml_done" ] || { echo "no multilabel for $c"; return; }
    t0=$(date +%s)
    cpu_run split neve-build-stagea 6 12g python3 /sp/v5_va/pilot_proc_song_tssplit.py "$c"
    timing "$c" "ts_split_$TSN" $(( $(date +%s) - t0 )) $? "per-class masks from the multilabel"
  fi
  if [ ! -f "$DER_H/$c/${TSN}_1p5/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$d/segmentations/vertebrae_C1.nii.gz" ]; then
    t0=$(date +%s)
    cpu_run 1p5 neve-build-stagea 6 10g python3 /sp/v5_va/tag_totalseg_make1p5.py "$DER_C/$c/$TSN" "$DER_C/$c/${TSN}_1p5" | tail -3
    timing "$c" "make1p5_$TSN" $(( $(date +%s) - t0 )) 0 "1.5 mm RAS copy (tag_totalseg_make1p5.py unchanged)"
  fi
  [ -f "$d/segmentations/vertebrae_C1.nii.gz" ] && rm -f "$d/ct_ts_input.nii.gz"
}

stage_args() {
  c=$1; d=$DER_H/$c
  [ -f "$d/arterial/args_run.txt" ] && { echo "args exist: $(cat "$d/arterial/args_run.txt")"; return; }
  [ -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$d/totalseg/segmentations/skull.nii.gz" ] || { echo "no TS masks for $c"; return; }
  t0=$(date +%s)
  light_run args neve-build-stagea 4 8g python3 /sp/v5_va/pilot_proc_song_args.py "$c"
  timing "$c" args $(( $(date +%s) - t0 )) $? "arterial_ts_args + two-limb check"
}

stage_arterial() {
  c=$1; d=$DER_H/$c; out=$DER_C/$c/arterial/runs/native
  [ -f "$d/arterial/.arterial_done" ] && { echo "arterial done"; return; }
  [ -f "$d/arterial/args_run.txt" ] || { echo "no args for $c"; return; }
  case "${ART_IN:-shipped}" in
    shipped) IN=$SRC_C/$c/ct.nii.gz;;
    bg) IN=$DER_C/$c/ct_bg-100.nii.gz
        [ -d "$d/arterial/runs/native" ] && [ ! -d "$d/arterial/runs/native_shipped_failed" ] && mv "$d/arterial/runs/native" "$d/arterial/runs/native_shipped_failed";;
    negzero) IN=$DER_C/$c/ct_negzero-100.nii.gz
        [ -d "$d/arterial/runs/native" ] && [ ! -d "$d/arterial/runs/native_shipped_failed" ] && mv "$d/arterial/runs/native" "$d/arterial/runs/native_shipped_failed";;
    air1000) IN=$DER_C/$c/ct_negzero_air1000.nii.gz
        [ -f "$d/ct_negzero_air1000.nii.gz" ] || light_run rw neve-build-stagea 2 8g python3 /sp/v5_va/pilot_proc_song_rewindow.py "$c"
        [ -d "$d/arterial/runs/native" ] && mv "$d/arterial/runs/native" "$d/arterial/runs/native_prev_$(date +%H%M%S)";;
    tex) IN=$DER_C/$c/ct_tex.nii.gz
        [ -f "$d/ct_tex.nii.gz" ] || light_run rw2 neve-build-stagea 2 14g python3 /sp/v5_va/pilot_proc_song_rewindow2.py "$c"
        [ -d "$d/arterial/runs/native" ] && mv "$d/arterial/runs/native" "$d/arterial/runs/native_prev_$(date +%H%M%S)";;
  esac
  a=$(cat "$d/arterial/args_run.txt")
  for attempt in 1 2; do
    [ -f "$d/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ] && [ -f "$d/arterial/runs/native/run_report.json" ] && break
    if [ $attempt -eq 2 ]; then
      grep -qi "aortic\|arch\|endpoint" "$d/arterial/runs/native/arterial.log" 2>/dev/null || { echo "attempt 1: no arch-check failure in the log, no retry"; break; }
      a=$(cat "$d/arterial/args_retry.txt" 2>/dev/null)
      [ -z "$a" ] && { echo "attempt 1 failed at the arch check; no corrected crop exists (no two-limb slice)"; break; }
      mv "$d/arterial/runs/native" "$d/arterial/runs/native_attempt1_failed"
    fi
    gpu_lock "arterial $c attempt $attempt input ${ART_IN:-shipped}"
    t0=$(date +%s)
    echo "[pilot_proc_song] run_arterial $IN -> $out  args: $a --no-tta"
    ARTERIAL_GPU=1 ARTERIAL_CPUS=12 ARTERIAL_MEM=${ARTERIAL_MEM:-20g} bash $SP/v5_va/run_arterial.sh "$IN" "$out" $a --no-tta &
    apid=$!
    wd=""
    while kill -0 $apid 2>/dev/null; do
      sleep 15
      el=$(( $(date +%s) - t0 ))
      seg="$d/arterial/runs/native/extracranial_vessels/segmentation.nii.gz"
      if { [ $el -gt ${ART_TIMEOUT:-1200} ] && [ -f "$seg" ]; } || [ $el -gt $(( 2 * ${ART_TIMEOUT:-1200} )) ]; then
        wd="stopped after $el s (segmentation written: $([ -f "$seg" ] && echo yes || echo no)); last log line: $(tail -1 "$d/arterial/runs/native/arterial.log" 2>/dev/null | tr -d '\"' | cut -c1-200)"
        echo "[watchdog] $wd"
        docker stop "arterial_native_$apid" >/dev/null 2>&1
        break
      fi
    done
    wait $apid
    rc=$?
    [ -n "$wd" ] && printf '{"case": "%s", "attempt": %s, "watchdog": "%s"}
' "$c" $attempt "$wd" >> "$d/arterial/watchdog.json"
    gpu_unlock
    timing "$c" "arterial_attempt$attempt" $(( $(date +%s) - t0 )) $rc "$a --no-tta (input ${ART_IN:-shipped})"
    rm -f "$d/arterial/runs/native/cta.nii.gz" "$d/arterial/runs/native_attempt1_failed/cta.nii.gz"
  done
  printf '{"arterial_input": "%s", "path": "%s"}\n' "${ART_IN:-shipped}" "$IN" > "$d/arterial/input_used.json"
  touch "$d/arterial/.arterial_done"
}

stage_extract() {
  c=$1
  [ -f "$DER_H/$c/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ] || { echo "no Arterial segmentation for $c"; return; }
  for m in ${EXTRACT_MODES:-tsba ts}; do
    [ -f "$DER_H/$c/arterial/$m/va_extract.json" ] && continue
    t0=$(date +%s)
    cpu_run ex neve-build-stagea 8 18g bash /sp/v5_va/pilot_proc_song_extract.sh "$c" $m
    timing "$c" "extract_$m" $(( $(date +%s) - t0 )) $? "art_extract ($m)"
  done
}

stage_vmtk() {
  c=$1; t0=$(date +%s)
  dirs=""; for m in ${EXTRACT_MODES:-tsba ts}; do [ -f "$DER_H/$c/arterial/$m/va_extract.json" ] && dirs="$dirs $DER_C/$c/arterial/$m"; done
  [ -z "$dirs" ] && return
  cpu_lock vmtk
  docker run --rm --name pilot_proc_song_vmtk_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash /sp/v5_va/pilot_proc_song_vmtk.sh \
    $dirs 2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
  cpu_unlock
  timing "$c" vmtk $(( $(date +%s) - t0 )) 0 "VMTK MISR on the cut-out VA surfaces"
}

stage_qc() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/arterial/tsba/va_extract.json" ] || return
  [ -f "$DER_H/$c/qc/qc.json" ] && [ -z "$QC_FORCE" ] && { echo "qc exists"; return; }
  cpu_run qc neve-build-stagea 6 12g python3 /sp/v5_va/pilot_proc_song_qc2.py "$c"
  timing "$c" qc $(( $(date +%s) - t0 )) $? "recalibrated Stage-2 gate + brief checks"
}

stage_gross() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  [ -f "$DER_H/$c/qc/gross.json" ] && [ -z "$GROSS_FORCE" ] && { echo "gross exists"; return; }
  PILOT_CASE_ENV=$c cpu_run gross neve-build-stagea  4 8g python3 /sp/v5_va/pilot_proc_song_gross.py
  timing "$c" gross $(( $(date +%s) - t0 )) $? "gross shape on the Arterial VMTK lines"
}

stage_ctseg() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  T=${PILOT_SEG_TAG:-ct_seg4}
  if [ ! -f "$DER_H/$c/ct_seg/$T/seg_qc.json" ]; then
    grep -q '"passes_as_v5_donor": true' "$DER_H/$c/qc/qc.json" "$DER_H/$c/qc/qc_ts.json" 2>/dev/null || { echo "no donor-passing side in $c (tsba or ts): no ct_seg"; return; }
    cpu_lock ctseg
    docker run --rm --name pilot_proc_song_seg_$$ --cpus 8 --memory 16g -e PYTHONUNBUFFERED=1 -e VMR_OPEN_BRIGHT=1 \
      -e PILOT_CASE="$c" -e PILOT_SEG_TAG="$T" -e PILOT_CT="${PILOT_CT:-shipped}" -e PILOT_KEEP_CONNECTED="${PILOT_KEEP_CONNECTED:-1}" ${TAG_SE_R:+-e TAG_SE_R=$TAG_SE_R} "${MNT[@]}" -w /repo \
      ${CTSEG_IMAGE:-neve-build-stagea} python3 /sp/v5_va/pilot_proc_song_ct_seg.py 2>&1 | grep -v "$FILT"
    rc=${PIPESTATUS[0]}
    cpu_unlock
    timing "$c" "ctseg_$T" $(( $(date +%s) - t0 )) $rc "ct_seg4 seeded from the VMTK trace"
  fi
  D=$DER_C/$c/ct_seg/$T
  if [ -d "$DER_H/$c/ct_seg/$T" ]; then
    t0=$(date +%s)
    cpu_lock cvmtk
    docker run --rm --name pilot_proc_song_cvmtk_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash -c \
      "for v in rva lva; do [ -f $D/\${v}_stageA.vtp ] && [ -f $D/\${v}_seeds.json ] && [ ! -f $D/\${v}_stageA_vmtk.json ] && python3 /sp/v5_va/vmr_vmtk.py $D/\${v}_stageA.vtp $D/\${v}_seeds.json $D/\${v}_stageA_vmtk.json; done; true" \
      2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
    cpu_unlock
    timing "$c" "ctseg_vmtk_$T" $(( $(date +%s) - t0 )) 0 "VMTK MISR on the ct_seg stage-A surfaces (vmr_vmtk.py unchanged)"
  fi
}

stage_measure() {
  c=$1; T=${PILOT_SEG_TAG:-ct_seg4}; D=$DER_H/$c/ct_seg/$T
  ls $D/*_stageA_vmtk.json >/dev/null 2>&1 || { echo "no ct_seg centrelines for $c"; return; }
  for scr in pilot_proc_song_landmarks.py pilot_proc_song_measure.py pilot_proc_song_edge.py; do
    case $scr in *landmarks*) done_f=$D/landmarks.json;; *measure*) done_f=$D/measure.json;; *edge*) done_f=$D/edge.json;; esac
    [ -f "$done_f" ] && [ -z "$MEASURE_FORCE" ] && continue
    img=eve-training-fixed; case $scr in *landmarks*|*edge*) img=neve-build-stagea;; esac
    t0=$(date +%s)
    cpu_lock ms
    docker run --rm --name pilot_proc_song_ms_$$ --cpus 8 --memory 14g -e PYTHONUNBUFFERED=1 -e PILOT_CASE="$c" \
      -e PILOT_SEG_TAGS="${PILOT_SEG_TAGS:-ct_seg4}" -e PILOT_CT="${PILOT_CT:-shipped}" "${MNT[@]}" -w /repo $img python3 /sp/v5_va/$scr 2>&1 | grep -v "$FILT"
    rc=${PIPESTATUS[0]}
    cpu_unlock
    timing "$c" "${scr%.py}" $(( $(date +%s) - t0 )) $rc "$scr"
  done
}

stage_fig() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  [ -f "$DER_H/$c/qc/qc_${c}.png" ] && [ -z "$FIG_FORCE" ] && { echo "fig exists"; return; }
  PILOT_CASE_ENV=$c cpu_run fig neve-build-stagea  6 10g python3 /sp/v5_va/pilot_proc_song_fig.py "$c"
  timing "$c" fig $(( $(date +%s) - t0 )) $? "figures"
}

stage_collect() {
  c=$1; t0=$(date +%s)
  docker run --rm --name pilot_proc_song_col_$$ --cpus 2 --memory 4g "${MNT[@]}" -w /repo eve-training-fixed \
    python3 /sp/v5_va/pilot_proc_song_collect.py "$c" 2>&1 | grep -v "$FILT" | tail -5
  timing "$c" collect $(( $(date +%s) - t0 )) ${PIPESTATUS[0]} "results JSON"
}

STAGES=$stage
case "$stage" in
  gpu) STAGES="tsprep tsgpu tscpu args arterial";;
  ts) STAGES="tsprep tsgpu tscpu";;
  post) STAGES="extract vmtk qc gross ctseg measure fig collect";;
  wall) STAGES="ctseg measure fig collect";;
esac
for c in "${cases[@]}"; do
  if [ "$stage" = "post" ]; then   # wait for this case's Arterial run
    n=0; until [ -f "$DER_H/$c/arterial/.arterial_done" ]; do sleep 30; n=$((n+1)); [ $n -gt 480 ] && break; done
  fi
  for s in $STAGES; do
    echo "== $(date '+%F %T') $c $s"
    T0=$(date +%s)
    stage_$s "$c" 2>&1 | tee -a "$LOGD/${c}_$s.log"
    echo "== $(date '+%F %T') $c $s done in $(( $(date +%s) - T0 )) s"
    timing "$c" "stage_$s" $(( $(date +%s) - T0 )) 0 "whole stage wall time"
  done
done
echo "== $(date '+%F %T') run.sh $stage finished"
