#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_run.sh (= /sp/v5_va/scaleup_ia172_run.sh in the build containers)
# scaleup_ia172_run.sh <stage|group> <case> [case ...]   (scaleup_lane_ia172, 2026-10-02; Git Bash on the host)
#
# = pilot_proc_ia172_run2.sh (the pilot driver that ran the 172-series pilot) COPIED, with these diffs (nothing else):
#  D1 logs / timing / locks in derived/thin_slice/ia172/_logs_scaleup (the pilot's _logs_proc is the pilot record);
#     container names scaleup_ia172_<stage>_<pid>; GPU-lock owner "scaleup_lane_ia172 <what> <epoch>".
#  D2 heavy CPU containers capped at 6 CPUs / 12g (the scale-up rule; the pilot used up to 8 CPUs / 18g); before each
#     heavy container wait until the WSL VM has >= 6 GB MemAvailable (>= 8 GB before a GPU container).
#  D3 CPU-lock priority: stages that feed the GPU (fetch, tscpu) set a "want" marker; low-priority stages (post / grade)
#     do not take the lock while it exists (keeps the GPU busy while case n is in post).
#  D4 new stage "fetch" (the pilot fetch method, pilot_fetch_ia172.md): pre-fetch arch check result required
#     (results/scaleup_ia172_prearch.json pass, or a by-eye override _logs_scaleup/prearch_eye_pass_<case>) ->
#     pilot_ia172_fetch.py UNCHANGED (zip byte-span range read, CRC-checked; pilot_ia172_convert.py DICOM -> NIfTI) ->
#     quick-look TotalSegmentator total --fast (scaleup_ia172_ts.sh = pilot_ia172_ts.sh, 6 CPUs) -> scaleup_ia172_qc.py
#     (= pilot_ia172_qc.py; phase / coverage / MIP / meta.json) -> scaleup_ia172_archcut.py (= pilot_ia172_archcut.py,
#     post-fetch two-limb confirmation). Marker _logs_scaleup/fetch_done_<case>; auto flags -> fetch_flag_<case>.
#     The visual check (both VAs, arterial phase) is mine, by eye on mip_neck.png -> results/scaleup_ia172_visual.json
#     (scaleup_ia172_meta_visual.py); a rejected case gets _logs_scaleup/rejected_<case> and the chains skip it.
#  D5 TotalSegmentator GPU via scaleup_ia172_tsrun.sh (= pilot_proc_ia172_tsrun2.sh, container / log prefix scaleup_ia172_).
#  D6 the Arterial watchdog runs inside stage_arterial for this case only (pilot_proc_ia172_watchdog.sh logic, 15 min
#     without arterial.log output in the centreline stage -> docker stop of THIS case's container; watchdog.json).
#  D7 grade stages (qc gross ctseg measure fig collect) are NOT in this file: they run pipeline v2
#     (notes/SCALEUP_PIPELINE.md, scaleup_fixes) through scaleup_ia172_grade.sh once state/scaleup_fixes.md is complete.
#  D8 (instance 2, 06:30) the fetch download + convert (pilot_ia172_fetch.py) runs as a LIGHT container (2 CPUs / 8g, after a
#     6 GB RAM wait, no CPU lock): it is network-bound (2-25 min per case) and was holding the agent's one heavy CPU slot,
#     starving post (extract / VMTK). ts_fast and fetch_qc keep the heavy slot. Earlier cases ran D1-D7 only (same outputs).
#  D9 (instance 2, 06:35) make1p5 (tag_totalseg_make1p5.py, single-threaded: ~1 core, ~2 GB) runs as a light container
#     (2 CPUs / 8g, after a 6 GB RAM wait, no CPU lock, no PYTHONPATH -- same command and image as before).
#  D10 (instance 2, 07:35) post-chain VMTK (single-core: ~1 core / 150 MB in docker stats) runs light (--cpus 2) under the
#     lane light-slot lock .cpu_lock_v2light (shared with the v2 light stages, scaleup_ia172_v2patch.sh), not the heavy lock.
# Everything else (stage bodies tsgpu, tscpu, args, clip, arterial, extract, vmtk; the scripts they call) is the pilot's.
# Every stage is idempotent (done markers / outputs), so a rerun continues where a cut-off run stopped.
# groups: gpu = args clip arterial ; post = extract vmtk
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOCK=$SP/v5_va/.gpu_lock
SRC_H=/d/Arjun/neve/carotid_data/va_sources/thin_slice/ia172
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172
SRC_C=/repo/carotid_data/va_sources/thin_slice/ia172
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/ia172
LOGD=$DER_H/_logs_scaleup
CLOCK=$LOGD/.cpu_lock
WANT=$LOGD/.cpu_want_hi
mkdir -p "$LOGD"
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'

stage=$1; shift
cases=("$@")
[ ${#cases[@]} -eq 0 ] && { echo "usage: scaleup_ia172_run.sh <stage|group> <case> ..."; exit 2; }

mem_avail_gb() { wsl -d docker-desktop cat /proc/meminfo 2>/dev/null | awk '/MemAvailable/ {printf "%d", $2/1048576}'; }
wait_ram() {   # $1 = GB needed (D2)
  local need=$1 n=0 m
  while :; do
    m=$(mem_avail_gb); [ -z "$m" ] && m=99
    [ "$m" -ge "$need" ] && return 0
    [ $((n % 10)) -eq 0 ] && echo "$(date '+%F %T') waiting for RAM: $m GB available < $need GB" >> "$LOGD/ram_wait.log"
    n=$((n+1)); sleep 30
  done
}

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
  echo "scaleup_lane_ia172 $1 $(date +%s)" > "$LOCK/owner"
  echo "$(date '+%F %T') acquired: $1 (run.sh pid $$)" >> "$LOGD/gpu_lock.log"
  trap 'rm -rf "$LOCK"; echo "$(date "+%F %T") released (trap)" >> "$LOGD/gpu_lock.log"' EXIT INT TERM
}
gpu_unlock() { rm -rf "$LOCK"; trap - EXIT INT TERM; echo "$(date '+%F %T') released" >> "$LOGD/gpu_lock.log"; }
cpu_lock() {   # $1 = what ; PRIO=hi for GPU-feeding stages (D3)
  if [ "${PRIO:-lo}" = "hi" ]; then
    mkdir -p "$WANT"; touch "$WANT/$$"
    until mkdir "$CLOCK" 2>/dev/null; do sleep 10; done
    rm -f "$WANT/$$"
  else
    while :; do
      if [ -z "$(ls -A "$WANT" 2>/dev/null)" ] && mkdir "$CLOCK" 2>/dev/null; then break; fi
      sleep 15
    done
  fi
  echo "$1 $$ $(date +%s)" > "$CLOCK/owner"
}
cpu_unlock() { rm -rf "$CLOCK"; }
CLOCK2=$LOGD/.cpu_lock_v2light   # D10: the lane's light-slot lock, shared with scaleup_ia172_v2patch.sh (one single-core post/grade container at a time)
lock2() { until mkdir "$CLOCK2" 2>/dev/null; do
            if [ -f "$CLOCK2/owner" ]; then p=$(awk '{print $2}' "$CLOCK2/owner"); [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null && rm -rf "$CLOCK2" && continue; fi
            sleep 5; done; echo "v2light $$ $(date +%s) $1" > "$CLOCK2/owner"; }
unlock2() { rm -rf "$CLOCK2"; }

timing() {  # case stage seconds rc note
  printf '{"case": "%s", "stage": "%s", "seconds": %s, "rc": %s, "note": "%s", "end": "%s", "by": "scaleup_ia172_run.sh"}\n' "$1" "$2" "$3" "$4" "$5" "$(date '+%F %T')" >> "$LOGD/timing.jsonl"
}

light_run() {  # name image cpus mem cmd...   (LIGHT containers: args / clip / archcut; no CPU lock, as in the pilot)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  docker run --rm --name "scaleup_ia172_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/sp/va_pool/pylib "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  return ${PIPESTATUS[0]}
}

cpu_run() {  # name image cpus mem cmd...   (one heavy CPU container at a time for this agent; D2 caps)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  cpu_lock "$nm"
  wait_ram 6
  docker run --rm --name "scaleup_ia172_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 \
    ${PILOT_CASE_ENV:+-e PILOT_CASE=$PILOT_CASE_ENV} ${PYP:+-e PYTHONPATH=$PYP} "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  local rc=${PIPESTATUS[0]}
  cpu_unlock
  return $rc
}

stage_fetch() {   # D4
  c=$1; sid=${c#ia172_}; d=$DER_H/$c
  [ -f "$LOGD/fetch_done_$c" ] && { echo "fetch done"; return; }
  if ! grep -q "\"$sid\"" "$SP/v5_va/results/scaleup_ia172_prearch.json" 2>/dev/null; then
    light_run prearch neve-build-stagea 2 3g python3 /sp/v5_va/scaleup_ia172_prearch.py "$sid"
  fi
  pa=$(docker run --rm -v "$SPW:/sp" neve-build-stagea python3 -c "import json; print(json.load(open('/sp/v5_va/results/scaleup_ia172_prearch.json'))['$sid']['pass'])" 2>/dev/null | tail -1)
  if [ "$pa" != "True" ] && [ ! -f "$LOGD/prearch_eye_pass_$c" ]; then
    echo "$c: pre-fetch arch check FAILED (results/scaleup_ia172_prearch.json; PNG results/scaleup_ia172_prearch/$sid.png) -> skipped"
    echo "prearch fail $(date '+%F %T')" > "$LOGD/rejected_$c"; return
  fi
  t0=$(date +%s)
  wait_ram 6; light_run fetch neve-build-stagea 2 8g python3 /sp/v5_va/pilot_ia172_fetch.py "$sid"   # D8
  rc=$?; timing "$c" fetch_convert $(( $(date +%s) - t0 )) $rc "pilot_ia172_fetch.py (range read + convert)"
  [ -f "$d/convert.json" ] && [ -f "$SRC_H/$c/ct.nii.gz" ] || { echo "$c: fetch / convert failed"; echo "fetch failed $(date '+%F %T')" > "$LOGD/fetch_failed_$c"; return; }
  if [ ! -f "$d/ts_fast_1p5.nii.gz" ]; then
    t0=$(date +%s)
    PRIO=hi cpu_lock fast; wait_ram 6
    bash $SP/v5_va/scaleup_ia172_ts.sh "$c" 2>&1 | grep -v "$FILT"
    cpu_unlock
    timing "$c" ts_fast $(( $(date +%s) - t0 )) 0 "quick-look TotalSegmentator total --fast (CPU)"
  fi
  t0=$(date +%s)
  PRIO=hi cpu_run fqc neve-build-stagea 6 12g python3 /sp/v5_va/scaleup_ia172_qc.py "$c"
  timing "$c" fetch_qc $(( $(date +%s) - t0 )) $? "scaleup_ia172_qc.py (= pilot_ia172_qc.py)"
  light_run archcut neve-build-stagea 2 4g python3 /sp/v5_va/scaleup_ia172_archcut.py "$c"
  if [ -f "$SRC_H/$c/meta.json" ]; then
    fl=$(docker run --rm -v "D:\\Arjun\\neve:/repo" neve-build-stagea python3 -c "
import json; m=json.load(open('$SRC_C/$c/meta.json')); print('; '.join(m['qc_auto_flags'] + ([] if m['phase_auto']=='arterial' else ['phase '+m['phase_auto']])))" 2>/dev/null | tail -1)
    [ -n "$fl" ] && echo "$fl" > "$LOGD/fetch_flag_$c"
    touch "$LOGD/fetch_done_$c"
    echo "$c: fetched; auto flags: ${fl:-none}"
  else
    echo "$c: no meta.json"; echo "fetch qc failed $(date '+%F %T')" > "$LOGD/fetch_failed_$c"
  fi
}

stage_tsgpu() {
  c=$1; d=$DER_H/$c; mkdir -p "$d/totalseg"
  [ -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ] && { echo "per-class masks exist"; return; }
  [ -f "$d/totalseg/.ts_ml_done" ] && { echo "multilabel exists"; return; }
  [ -f "$d/totalseg/ct.nii.gz" ] || ln "$SRC_H/$c/ct.nii.gz" "$d/totalseg/ct.nii.gz"
  wait_ram 8
  gpu_lock "ts $c"
  t0=$(date +%s)
  bash $SP/v5_va/scaleup_ia172_tsrun.sh "$SRC_C/$c/ct.nii.gz" "$DER_C/$c/totalseg/segmentations_ml.nii.gz" "${c}_total_v3" --task total_v3 --ml
  rc=$?
  gpu_unlock
  timing "$c" ts_total_v3_ml $(( $(date +%s) - t0 )) $rc "TotalSegmentator total_v3 on the GPU (CTA grid, --ml)"
  printf '{"task": "total_v3", "tool": "TotalSegmentator 2.18.0 (neve-totalseg-gpu)", "grid": "the CTA grid (nearest-neighbour write-back of the 1.5 mm prediction)", "writer": "--ml multilabel, split by pilot_proc_ia172_tssplit.py", "input": "raw HU (CTNormalization clips at fixed training percentiles; the [-1024, 3071] pre-clip is a no-op for it)", "why_not_total": "tag_totalseg.md: total_v3 C1 Dice 0.98 vs 0.92, CCA 0.92-0.95 vs 0.85-0.89, skull 0.93 vs 0.81-0.83; the V3-loop gate fails a real loop on total masks", "rc": %s, "by": "scaleup_ia172_run.sh"}\n' $rc > "$d/totalseg/task.json"
  [ $rc -eq 0 ] && [ -f "$d/totalseg/segmentations_ml.nii.gz" ] && touch "$d/totalseg/.ts_ml_done"
}

stage_tscpu() {
  c=$1; d=$DER_H/$c
  if [ ! -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ]; then
    n=0; until [ -f "$d/totalseg/.ts_ml_done" ]; do sleep 30; n=$((n+1)); [ $n -gt 240 ] && { echo "no multilabel for $c after 2 h"; return; }; done
    t0=$(date +%s)
    PRIO=hi cpu_run split neve-build-stagea 6 12g python3 /sp/v5_va/pilot_proc_ia172_tssplit.py "$c"
    timing "$c" ts_split $(( $(date +%s) - t0 )) $? "per-class masks from the multilabel"
  fi
  if [ ! -f "$d/totalseg_1p5/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ]; then
    t0=$(date +%s)
    wait_ram 6; docker run --rm --name "scaleup_ia172_1p5_$$" --cpus 2 --memory 8g -e PYTHONUNBUFFERED=1 "${MNT[@]}" -w /repo neve-build-stagea python3 /sp/v5_va/tag_totalseg_make1p5.py "$DER_C/$c/totalseg" "$DER_C/$c/totalseg_1p5" 2>&1 | grep -v "$FILT" | tail -3   # D9
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

watchdog_case() {   # D6: $1 case, $2 pid of the run_arterial.sh call; WD_MIN minutes without log output in stage cl
  local c=$1 pid=$2 lg=$DER_H/$1/arterial/runs/native/arterial.log WD=${WD_MIN:-15} age n
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    [ -f "$lg" ] || continue
    grep -q "===== stage cl =====" "$lg" || continue
    grep -q "===== stage cl:" "$lg" && continue
    age=$(( $(date +%s) - $(stat -c %Y "$lg") ))
    [ $age -lt $((WD * 60)) ] && continue
    for n in $(docker ps --format '{{.Names}}' | grep '^arterial_'); do
      if docker inspect -f '{{.Args}}' $n 2>/dev/null | grep -q "thin_slice/ia172/$c/"; then
        printf '{"stopped_by": "scaleup_ia172_run.sh watchdog (= pilot_proc_ia172_watchdog.sh logic)", "when": "%s", "container": "%s", "why": "Arterial centreline stage wrote nothing to arterial.log for %d s (limit %d min); segmentation stage finished; art_extract traces the binary mask without Arterial labels"}\n' \
          "$(date '+%F %T')" "$n" "$age" "$WD" > $DER_H/$c/arterial/runs/native/watchdog.json
        echo "$(date '+%F %T') $c: no log output for $age s in the centreline stage -> docker stop $n" >> $LOGD/watchdog.log
        docker stop $n >> $LOGD/watchdog.log 2>&1
      fi
    done
  done
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
    wait_ram 8
    gpu_lock "arterial $c attempt $attempt"
    t0=$(date +%s)
    echo "[scaleup_ia172] run_arterial $DER_C/$c/arterial/ct_clip.nii.gz -> $out  args: $a --no-tta"
    ARTERIAL_GPU=1 ARTERIAL_CPUS=12 ARTERIAL_MEM=${ARTERIAL_MEM:-20g} bash $SP/v5_va/run_arterial.sh "$DER_C/$c/arterial/ct_clip.nii.gz" "$out" $a --no-tta &
    apid=$!
    watchdog_case "$c" $apid &
    wpid=$!
    wait $apid; rc=$?
    kill $wpid 2>/dev/null
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
    cpu_run ex neve-build-stagea 6 12g bash /sp/v5_va/pilot_proc_ia172_extract.sh "$c" $m
    timing "$c" "extract_$m" $(( $(date +%s) - t0 )) $? "art_extract ($m)"
  done
}

stage_vmtk() {
  c=$1; t0=$(date +%s)
  dirs=""; for m in ${EXTRACT_MODES:-tsba ts}; do [ -f "$DER_H/$c/arterial/$m/va_extract.json" ] && dirs="$dirs $DER_C/$c/arterial/$m"; done
  [ -z "$dirs" ] && return
  lock2 vmtk; wait_ram 6   # D10
  docker run --rm --name scaleup_ia172_vmtk_$$ -u 0 --cpus 2 --memory 8g "${MNT[@]}" neve-build-vmtk bash /sp/v5_va/pilot_proc_ia172_vmtk.sh \
    $dirs 2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
  unlock2
  timing "$c" vmtk $(( $(date +%s) - t0 )) 0 "VMTK MISR on the cut-out VA surfaces"
  touch "$LOGD/post_done_$c"
}

STAGES=$stage
case "$stage" in
  gpu) STAGES="args clip arterial";;
  post) STAGES="extract vmtk";;
esac
for c in "${cases[@]}"; do
  { [ -f "$LOGD/rejected_$c" ] || [ -f "$LOGD/fetch_failed_$c" ]; } && { echo "$c: rejected / fetch failed: skip"; continue; }
  for s in $STAGES; do
    echo "== $(date '+%F %T') $c $s"
    T0=$(date +%s)
    stage_$s "$c" 2>&1 | tee -a "$LOGD/${c}_$s.log"
    echo "== $(date '+%F %T') $c $s done in $(( $(date +%s) - T0 )) s"
    timing "$c" "stage_$s" $(( $(date +%s) - T0 )) 0 "whole stage wall time"
  done
done
echo "== $(date '+%F %T') scaleup_ia172_run.sh $stage finished"
