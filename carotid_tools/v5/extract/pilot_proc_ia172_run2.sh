#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_run2.sh (= /sp/v5_va/pilot_proc_ia172_run2.sh in the build containers)
# pilot_proc_ia172_run2.sh <stage|group> [case ...] = pilot_proc_ia172_run.sh + ONE change (20:50): the light args / clip
# containers run without the agent CPU lock (light_run), so the GPU loop is not starved while the post loop holds it.
# + (21:05) the ctseg container image is neve-build-stagea (nibabel; tag_ct_seg.py ran there), not eve-training-fixed.
# (Git Bash on the host; third instance of pilot_proc_ia172, 2026-10-01)
#
# The thin-slice VA donor pipeline on the Head-CTA 172-series pilot cases (Zenodo 15697196, CC BY 4.0):
#   images  D:\Arjun\neve\carotid_data\va_sources\thin_slice\ia172\<case>\ct.nii.gz   (raw HU, RAS, native grid)
#   derived D:\Arjun\neve\carotid_data\va_sources\derived\thin_slice\ia172\<case>\
# Based on pilot_proc_ia172.sh.next (the first instance's newer driver, never run); pilot_proc_ia172.sh itself is left
# alone (its "ts" loop was still executing when this was written). Every stage is idempotent: it skips work whose output
# (done marker) exists, so a rerun continues where a cut-off run stopped. Per case and stage the wall time goes to
# _logs_proc/timing.jsonl, the output to _logs_proc/<case>_<stage>.log.
#
# stages:
#   tsgpu    TotalSegmentator 2.18.0 total_v3 on the GPU (GPU LOCK), one multilabel file (--ml; pilot_proc_ia172_tsrun2.sh)
#            -> <der>/totalseg/segmentations_ml.nii.gz + marker totalseg/.ts_ml_done. Skipped when per-class masks exist.
#   tscpu    split into <der>/totalseg/segmentations/<class>.nii.gz (pilot_proc_ia172_tssplit.py) + the 1.5 mm RAS copy
#            <der>/totalseg_1p5 (tag_totalseg_make1p5.py UNCHANGED). Waits for .ts_ml_done when the split is needed.
#   args     arterial_ts_args.py UNCHANGED (imported, --resample 0 = native) + two-limb check (pilot_proc_ia172_args.py)
#   clip     HU clip [-1024, 3071] of the CT for Arterial (pilot_proc_ia172_clip.py; ZScore normalisation, see its docstring)
#   arterial run_arterial.sh UNCHANGED on ct_clip.nii.gz, GPU, --no-tta, core stages, native grid, TS skull, crop/FOV from
#            args; GPU LOCK; one corrected crop if the arch check fails; cta.nii.gz + ct_clip deleted; marker .arterial_done
#   extract  art_extract.py UNCHANGED -> arterial/ts ; pilot_proc_ia172_art_extract.py (basilar fix) -> arterial/tsba
#   vmtk     VMTK MISR of the cut-out stage-A surfaces (art_vmtk_one.py UNCHANGED)
#   qc       recalibrated Stage-2 gate (tag_calibration.apply_gate imported) + the brief's checks (pilot_proc_ia172_qc2.py)
#   gross    gross shape of every traced side on its Arterial VMTK line (pilot_proc_ia172_gross.py)
#   ctseg    ct_seg4 seeded from the VMTK trace (pilot_proc_ia172_ct_seg.py) + VMTK of its surfaces (vmr_vmtk.py UNCHANGED)
#   measure  landmarks + va_metrics UNCHANGED + edge FWHM (pilot_proc_ia172_{landmarks,measure,edge}.py)
#   fig      QC PNG + 3-D wall renders (pilot_proc_ia172_fig.py)
#   collect  per-case records -> /sp/v5_va/results/pilot_proc_ia172.json (pilot_proc_ia172_collect.py)
# groups: gpu = args clip arterial ; post = extract vmtk qc gross ctseg measure fig collect (waits for .arterial_done)
# GPU LOCK: mkdir <sp>/v5_va/.gpu_lock (+ owner), removed as soon as the container exits (trap on failure too).
# CPU LOCK (this agent only; its two loops never run two heavy CPU containers at once): _logs_proc/.cpu_lock
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOCK=$SP/v5_va/.gpu_lock
SRC_H=/d/Arjun/neve/carotid_data/va_sources/thin_slice/ia172
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172
SRC_C=/repo/carotid_data/va_sources/thin_slice/ia172
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/ia172
LOGD=$DER_H/_logs_proc
CLOCK=$LOGD/.cpu_lock
mkdir -p "$LOGD"
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'

stage=$1; shift
cases=("$@")
[ ${#cases[@]} -eq 0 ] && cases=(ia172_058 ia172_061 ia172_140 ia172_141 ia172_001 ia172_039 ia172_041 ia172_073 ia172_190 ia172_015 ia172_118 ia172_090)

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
  echo "pilot_proc_ia172 $1 $(date +%s)" > "$LOCK/owner"
  echo "$(date '+%F %T') acquired: $1 (run.sh pid $$)" >> "$LOGD/gpu_lock.log"
  trap 'rm -rf "$LOCK"; echo "$(date "+%F %T") released (trap)" >> "$LOGD/gpu_lock.log"' EXIT INT TERM
}
gpu_unlock() { rm -rf "$LOCK"; trap - EXIT INT TERM; echo "$(date '+%F %T') released" >> "$LOGD/gpu_lock.log"; }
cpu_lock() { until mkdir "$CLOCK" 2>/dev/null; do sleep 10; done; echo "$1 $$ $(date +%s)" > "$CLOCK/owner"; }
cpu_unlock() { rm -rf "$CLOCK"; }

timing() {  # case stage seconds rc note
  printf '{"case": "%s", "stage": "%s", "seconds": %s, "rc": %s, "note": "%s", "end": "%s", "by": "run.sh"}\n' "$1" "$2" "$3" "$4" "$5" "$(date '+%F %T')" >> "$LOGD/timing.jsonl"
}

light_run() {  # name image cpus mem cmd...   (LIGHT containers: args / clip, <= 4 CPUs, ~2-3 GB, ~1-2 min; no CPU lock, so the
               # GPU loop is not starved by the post loop's long heavy stages)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  docker run --rm --name "pilot_proc_ia172_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 \
    "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  return ${PIPESTATUS[0]}
}

cpu_run() {  # name image cpus mem cmd...   (one heavy CPU container at a time for this agent)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  cpu_lock "$nm"
  docker run --rm --name "pilot_proc_ia172_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 \
    ${PILOT_CASE_ENV:+-e PILOT_CASE=$PILOT_CASE_ENV} "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  local rc=${PIPESTATUS[0]}
  cpu_unlock
  return $rc
}

stage_tsgpu() {
  c=$1; d=$DER_H/$c; mkdir -p "$d/totalseg"
  [ -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ] && { echo "per-class masks exist"; return; }
  [ -f "$d/totalseg/.ts_ml_done" ] && { echo "multilabel exists"; return; }
  [ -f "$d/totalseg/ct.nii.gz" ] || ln "$SRC_H/$c/ct.nii.gz" "$d/totalseg/ct.nii.gz"
  gpu_lock "ts $c"
  t0=$(date +%s)
  bash $SP/v5_va/pilot_proc_ia172_tsrun2.sh "$SRC_C/$c/ct.nii.gz" "$DER_C/$c/totalseg/segmentations_ml.nii.gz" "${c}_total_v3" --task total_v3 --ml
  rc=$?
  gpu_unlock
  timing "$c" ts_total_v3_ml $(( $(date +%s) - t0 )) $rc "TotalSegmentator total_v3 on the GPU (CTA grid, --ml)"
  printf '{"task": "total_v3", "tool": "TotalSegmentator 2.18.0 (neve-totalseg-gpu)", "grid": "the CTA grid (nearest-neighbour write-back of the 1.5 mm prediction)", "writer": "--ml multilabel, split by pilot_proc_ia172_tssplit.py", "input": "raw HU (CTNormalization clips at fixed training percentiles; the [-1024, 3071] pre-clip is a no-op for it)", "why_not_total": "tag_totalseg.md: total_v3 C1 Dice 0.98 vs 0.92, CCA 0.92-0.95 vs 0.85-0.89, skull 0.93 vs 0.81-0.83; the V3-loop gate fails a real loop on total masks", "rc": %s}\n' $rc > "$d/totalseg/task.json"
  [ $rc -eq 0 ] && [ -f "$d/totalseg/segmentations_ml.nii.gz" ] && touch "$d/totalseg/.ts_ml_done"
}

stage_tscpu() {
  c=$1; d=$DER_H/$c
  if [ ! -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ]; then
    n=0; until [ -f "$d/totalseg/.ts_ml_done" ]; do sleep 30; n=$((n+1)); [ $n -gt 240 ] && { echo "no multilabel for $c after 2 h"; return; }; done
    t0=$(date +%s)
    cpu_run split neve-build-stagea 6 12g python3 /sp/v5_va/pilot_proc_ia172_tssplit.py "$c"
    timing "$c" ts_split $(( $(date +%s) - t0 )) $? "per-class masks from the multilabel"
  fi
  if [ ! -f "$d/totalseg_1p5/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ]; then
    t0=$(date +%s)
    cpu_run 1p5 neve-build-stagea 6 10g python3 /sp/v5_va/tag_totalseg_make1p5.py "$DER_C/$c/totalseg" "$DER_C/$c/totalseg_1p5" | tail -3
    timing "$c" make1p5 $(( $(date +%s) - t0 )) 0 "1.5 mm RAS copy (tag_totalseg_make1p5.py unchanged)"
  fi
}

stage_args() {
  c=$1; d=$DER_H/$c
  [ -f "$d/arterial/args_run.txt" ] && { echo "args exist: $(cat "$d/arterial/args_run.txt")"; return; }
  n=0; until [ -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$d/totalseg/segmentations/skull.nii.gz" ]; do
    sleep 30; n=$((n+1)); [ $n -gt 240 ] && { echo "no TS masks for $c"; return; }; done
  t0=$(date +%s)
  light_run args neve-build-stagea 4 8g python3 /sp/v5_va/pilot_proc_ia172_args.py "$c"
  timing "$c" args $(( $(date +%s) - t0 )) $? "arterial_ts_args + two-limb check"
}

stage_clip() {
  c=$1; d=$DER_H/$c
  { [ -f "$d/arterial/.arterial_done" ] || [ -f "$d/arterial/ct_clip.nii.gz" ]; } && return
  t0=$(date +%s)
  light_run clip neve-build-stagea 2 6g python3 /sp/v5_va/pilot_proc_ia172_clip.py "$c"
  timing "$c" clip $(( $(date +%s) - t0 )) $? "HU clip for Arterial"
}

stage_arterial() {
  c=$1; d=$DER_H/$c; out=$DER_C/$c/arterial/runs/native
  [ -f "$d/arterial/.arterial_done" ] && { echo "arterial done"; return; }
  [ -f "$d/arterial/args_run.txt" ] || { echo "no args for $c"; touch "$d/arterial/.arterial_done"; return; }
  [ -f "$d/arterial/ct_clip.nii.gz" ] || { echo "no clipped CT for $c"; return; }
  a=$(cat "$d/arterial/args_run.txt")
  for attempt in 1 2; do
    [ -f "$d/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ] && [ -f "$d/arterial/runs/native/run_report.json" ] && break
    if [ $attempt -eq 2 ]; then
      grep -qi "aortic\|arch\|endpoint" "$d/arterial/runs/native/arterial.log" 2>/dev/null || { echo "attempt 1: no arch-check failure in the log, no retry"; break; }
      a=$(cat "$d/arterial/args_retry.txt" 2>/dev/null)
      [ -z "$a" ] && { echo "attempt 1 failed at the arch check; no corrected crop exists (no two-limb slice)"; break; }
      mv "$d/arterial/runs/native" "$d/arterial/runs/native_attempt1_failed"
    fi
    gpu_lock "arterial $c attempt $attempt"
    t0=$(date +%s)
    echo "[pilot_proc_ia172] run_arterial $DER_C/$c/arterial/ct_clip.nii.gz -> $out  args: $a --no-tta"
    ARTERIAL_GPU=1 ARTERIAL_CPUS=12 ARTERIAL_MEM=${ARTERIAL_MEM:-20g} bash $SP/v5_va/run_arterial.sh "$DER_C/$c/arterial/ct_clip.nii.gz" "$out" $a --no-tta
    rc=$?
    gpu_unlock
    timing "$c" "arterial_attempt$attempt" $(( $(date +%s) - t0 )) $rc "$a --no-tta (input ct_clip)"
    rm -f "$d/arterial/runs/native/cta.nii.gz" "$d/arterial/runs/native_attempt1_failed/cta.nii.gz"
  done
  rm -f "$d/arterial/ct_clip.nii.gz"
  touch "$d/arterial/.arterial_done"
}

stage_extract() {
  c=$1
  [ -f "$DER_H/$c/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ] || { echo "no Arterial segmentation for $c"; return; }
  for m in ${EXTRACT_MODES:-tsba ts}; do
    [ -f "$DER_H/$c/arterial/$m/va_extract.json" ] && continue
    t0=$(date +%s)
    cpu_run ex neve-build-stagea 8 18g bash /sp/v5_va/pilot_proc_ia172_extract.sh "$c" $m
    timing "$c" "extract_$m" $(( $(date +%s) - t0 )) $? "art_extract ($m)"
  done
}

stage_vmtk() {
  c=$1; t0=$(date +%s)
  dirs=""; for m in ${EXTRACT_MODES:-tsba ts}; do [ -f "$DER_H/$c/arterial/$m/va_extract.json" ] && dirs="$dirs $DER_C/$c/arterial/$m"; done
  [ -z "$dirs" ] && return
  cpu_lock vmtk
  docker run --rm --name pilot_proc_ia172_vmtk_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash /sp/v5_va/pilot_proc_ia172_vmtk.sh \
    $dirs 2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
  cpu_unlock
  timing "$c" vmtk $(( $(date +%s) - t0 )) 0 "VMTK MISR on the cut-out VA surfaces"
}

stage_qc() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/arterial/tsba/va_extract.json" ] || return
  [ -f "$DER_H/$c/qc/qc.json" ] && [ -z "$QC_FORCE" ] && { echo "qc exists"; return; }
  cpu_run qc neve-build-stagea 6 12g python3 /sp/v5_va/pilot_proc_ia172_qc2.py "$c"
  timing "$c" qc $(( $(date +%s) - t0 )) $? "recalibrated Stage-2 gate + brief checks"
}

stage_gross() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  [ -f "$DER_H/$c/qc/gross.json" ] && [ -z "$GROSS_FORCE" ] && { echo "gross exists"; return; }
  PILOT_CASE_ENV=$c cpu_run gross neve-build-stagea  4 8g python3 /sp/v5_va/pilot_proc_ia172_gross.py
  timing "$c" gross $(( $(date +%s) - t0 )) $? "gross shape on the Arterial VMTK lines"
}

stage_ctseg() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  T=${PILOT_SEG_TAG:-ct_seg4}
  if [ ! -f "$DER_H/$c/ct_seg/$T/seg_qc.json" ]; then
    grep -q '"passes_as_v5_donor": true' "$DER_H/$c/qc/qc.json" || { echo "no donor-passing side in $c: no ct_seg"; return; }
    cpu_lock ctseg
    docker run --rm --name pilot_proc_ia172_seg_$$ --cpus 8 --memory 16g -e PYTHONUNBUFFERED=1 -e VMR_OPEN_BRIGHT=1 \
      -e PILOT_CASE="$c" -e PILOT_SEG_TAG="$T" ${TAG_SE_R:+-e TAG_SE_R=$TAG_SE_R} "${MNT[@]}" -w /repo \
      neve-build-stagea python3 /sp/v5_va/pilot_proc_ia172_ct_seg.py 2>&1 | grep -v "$FILT"
    rc=${PIPESTATUS[0]}
    cpu_unlock
    timing "$c" "ctseg_$T" $(( $(date +%s) - t0 )) $rc "ct_seg4 seeded from the VMTK trace"
  fi
  D=$DER_C/$c/ct_seg/$T
  if [ -d "$DER_H/$c/ct_seg/$T" ]; then
    t0=$(date +%s)
    cpu_lock cvmtk
    docker run --rm --name pilot_proc_ia172_cvmtk_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash -c \
      "for v in rva lva; do [ -f $D/\${v}_stageA.vtp ] && [ -f $D/\${v}_seeds.json ] && [ ! -f $D/\${v}_stageA_vmtk.json ] && python3 /sp/v5_va/vmr_vmtk.py $D/\${v}_stageA.vtp $D/\${v}_seeds.json $D/\${v}_stageA_vmtk.json; done; true" \
      2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
    cpu_unlock
    timing "$c" "ctseg_vmtk_$T" $(( $(date +%s) - t0 )) 0 "VMTK MISR on the ct_seg stage-A surfaces (vmr_vmtk.py unchanged)"
  fi
}

stage_measure() {
  c=$1; T=${PILOT_SEG_TAG:-ct_seg4}; D=$DER_H/$c/ct_seg/$T
  ls $D/*_stageA_vmtk.json >/dev/null 2>&1 || { echo "no ct_seg centrelines for $c"; return; }
  for scr in pilot_proc_ia172_landmarks.py pilot_proc_ia172_measure.py pilot_proc_ia172_edge.py; do
    case $scr in *landmarks*) done_f=$D/landmarks.json;; *measure*) done_f=$D/measure.json;; *edge*) done_f=$D/edge.json;; esac
    [ -f "$done_f" ] && [ -z "$MEASURE_FORCE" ] && continue
    img=eve-training-fixed; case $scr in *landmarks*|*edge*) img=neve-build-stagea;; esac
    t0=$(date +%s)
    cpu_lock ms
    docker run --rm --name pilot_proc_ia172_ms_$$ --cpus 8 --memory 14g -e PYTHONUNBUFFERED=1 -e PILOT_CASE="$c" \
      -e PILOT_SEG_TAGS="${PILOT_SEG_TAGS:-ct_seg4}" "${MNT[@]}" -w /repo $img python3 /sp/v5_va/$scr 2>&1 | grep -v "$FILT"
    rc=${PIPESTATUS[0]}
    cpu_unlock
    timing "$c" "${scr%.py}" $(( $(date +%s) - t0 )) $rc "$scr"
  done
}

stage_fig() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  [ -f "$DER_H/$c/qc/qc_${c}.png" ] && [ -z "$FIG_FORCE" ] && { echo "fig exists"; return; }
  PILOT_CASE_ENV=$c cpu_run fig neve-build-stagea  6 10g python3 /sp/v5_va/pilot_proc_ia172_fig.py "$c"
  timing "$c" fig $(( $(date +%s) - t0 )) $? "figures"
}

stage_collect() {
  c=$1; t0=$(date +%s)
  docker run --rm --name pilot_proc_ia172_col_$$ --cpus 2 --memory 4g "${MNT[@]}" -w /repo eve-training-fixed \
    python3 /sp/v5_va/pilot_proc_ia172_collect.py "$c" 2>&1 | grep -v "$FILT" | tail -5
  timing "$c" collect $(( $(date +%s) - t0 )) ${PIPESTATUS[0]} "results JSON"
}

STAGES=$stage
case "$stage" in
  gpu) STAGES="args clip arterial";;
  post) STAGES="extract vmtk qc gross ctseg measure fig collect";;
  wall) STAGES="ctseg measure fig collect";;
esac
for c in "${cases[@]}"; do
  if [ "$stage" = "post" ]; then   # wait for this case's Arterial run
    until [ -f "$DER_H/$c/arterial/.arterial_done" ]; do sleep 30; done
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
