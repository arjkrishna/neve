#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_song_run.sh (= /sp/v5_va/scaleup_song_run.sh in the build containers)
# scaleup_song_run.sh <stage|group> <case> [case ...]   (Git Bash on the host; agent scaleup_lane_song, 2026-10-02)
#
# The v5 VA scale-up, Song lane. = pilot_proc_song_run5.sh (the Song pilot driver, read-only) with these changes:
#   * FETCH stages added (the pilot fetch agent's method, its scripts unchanged unless named scaleup_*):
#       fget   pilot_fetch_song_get.py      (RAR5 range read of the stored member, CRC32 check; index results/pilot_fetch_song_index.json)
#       fqc    pilot_fetch_song_qc.py       (qc_pass1.json, neck MIPs, qc_axial.png, side files ct_bg-100 / ct_negzero-100 / body_mask)
#              (ARCH_TOP_Z_<case> env = the by-eye arch-top override, as the pilot's ARCH_TOP_Z)
#       fvac   pilot_fetch_song_va_check2.py (column-frame VA check + side-file integrity check)
#       fzoom  pilot_fetch_song_va_zoom.py   (zoomed axial VA crops for the by-eye verdict)
#       fmeta  scaleup_song_fetch_picks.py + scaleup_song_fetch_meta.py (pilot picks / meta2 with scale-up result paths)
#   * logs / timing / CPU lock in derived/thin_slice/song/_logs_scaleup (pilot: _logs_proc); container names scaleup_song_*;
#     GPU-lock owner "scaleup_lane_song <what> <epoch>"; TS via scaleup_song_tsrun.sh (pilot tsrun, container prefix only).
#   * CPU/RAM rule of the scale-up: heavy CPU containers <= 6 CPUs / 12 GB (pilot: extract 8/18g, ctseg 8/16g, measure 8/14g,
#     vmtk 4/8g kept) and each heavy container waits until the WSL VM has >= 6 GB MemAvailable (GPU containers: >= 8 GB,
#     at most 30 min); a stale CPU lock (owner pid dead) is removed.
#   * ART_IN defaults to air1000 (the pilot's decision 22:56: every Song case); after Arterial the re-windowed copy
#     ct_negzero_air1000.nii.gz is deleted (regenerable with pilot_proc_song_rewindow.py; post reads the shipped CT) and, as
#     in the pilot, Arterial's cta.nii.gz copy.
#   * TS GPU: a tsrun exit 3 ("another GPU container is running") releases the lock and retries (<= 6 x, 60 s apart).
#   * post: + protrusion stage (pilot_proc_song_protrusion.py, the pilot's closing-pass step 1, per case); collect ->
#     scaleup_song_collect.py (results/scaleup_song.json). The plain ct_seg4_nofix side record and the clip comparison of
#     the pilot's closing pass are NOT run (pilot-only studies).
#   * POST_V2=1: the QC-onward stages use pipeline v2 (notes/SCALEUP_PIPELINE.md); set when the fixes agent is complete.
#     04:05 same-length edit: make1p5 + fzoom light; watchdog stops arterial_native__<pid
#   * a case with derived/.../_logs_scaleup/rejected_<case> (by-eye fetch rejection) is skipped by every stage.
# Every stage is idempotent (skips work whose output / done marker exists). Per case and stage: wall time ->
# _logs_scaleup/timing.jsonl, output -> _logs_scaleup/<case>_<stage>.log.
#
# stages: fget fqc fvac fzoom fmeta tsprep tsgpu tscpu args arterial extract vmtk qc gross ctseg measure protrusion fig collect
# groups: fetch = fget fqc fvac fzoom fmeta ; ts = tsprep tsgpu tscpu ; art = args arterial ; trace = extract vmtk ;
#         grade = qc gross ctseg measure protrusion fig collect ; post = trace + grade
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
LOCK=$SP/v5_va/.gpu_lock
SRC_H=/d/Arjun/neve/carotid_data/va_sources/thin_slice/song
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song
SRC_C=/repo/carotid_data/va_sources/thin_slice/song
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/song
LOGD=$DER_H/_logs_scaleup
CLOCK=$LOGD/.cpu_lock
TSN=${TS_DIRNAME:-totalseg}
mkdir -p "$LOGD"
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native\|WARNING: The NVIDIA\|Use the NVIDIA\|A copy of this license'

stage=$1; shift
cases=("$@")
[ ${#cases[@]} -eq 0 ] && { echo "usage: scaleup_song_run.sh <stage|group> <case> [case ...]"; exit 2; }

mem_avail_gb() { docker run --rm --entrypoint sh python:3 -c "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null; }
ram_wait() {   # $1 = GB needed, $2 = max waits of 30 s (default 120 = 60 min)
  local n=0 a
  while :; do
    a=$(mem_avail_gb)
    [ -n "$a" ] && [ "$a" -ge "$1" ] && return 0
    n=$((n+1))
    [ $n -eq 1 ] && echo "$(date '+%F %T') waiting for RAM: ${a:-?} GB available < $1 GB" | tee -a "$LOGD/ram_wait.log"
    [ $n -gt ${2:-120} ] && { echo "$(date '+%F %T') RAM wait gave up after $n tries (${a:-?} GB); starting anyway" | tee -a "$LOGD/ram_wait.log"; return 0; }
    sleep 30
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
  echo "scaleup_lane_song $1 $(date +%s)" > "$LOCK/owner"
  echo "$(date '+%F %T') acquired: $1 (run.sh pid $$)" >> "$LOGD/gpu_lock.log"
  trap 'rm -rf "$LOCK"; echo "$(date "+%F %T") released (trap)" >> "$LOGD/gpu_lock.log"' EXIT INT TERM
}
gpu_unlock() { rm -rf "$LOCK"; trap - EXIT INT TERM; echo "$(date '+%F %T') released" >> "$LOGD/gpu_lock.log"; }
cpu_lock() {
  until mkdir "$CLOCK" 2>/dev/null; do
    op=$(awk '{print $2}' "$CLOCK/owner" 2>/dev/null)
    if [ -n "$op" ] && [ "$op" -eq "$op" ] 2>/dev/null && ! kill -0 "$op" 2>/dev/null; then
      echo "$(date '+%F %T') removing stale CPU lock: $(cat "$CLOCK/owner")" >> "$LOGD/cpu_lock.log"; rm -rf "$CLOCK"; continue
    fi
    sleep 10
  done
  echo "$1 $$ $(date +%s)" > "$CLOCK/owner"
}
cpu_unlock() { rm -rf "$CLOCK"; }

timing() {  # case stage seconds rc note
  printf '{"case": "%s", "stage": "%s", "seconds": %s, "rc": %s, "note": "%s", "end": "%s", "by": "scaleup_song_run.sh"}\n' "$1" "$2" "$3" "$4" "$5" "$(date '+%F %T')" >> "$LOGD/timing.jsonl"
}

light_run() {  # name image cpus mem cmd...   (light containers, no CPU lock)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  docker run --rm --name "scaleup_song_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 -e TS_DIRNAME=$TSN \
    "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  return ${PIPESTATUS[0]}
}

cpu_run() {  # name image cpus mem cmd...   (one heavy CPU container at a time for this agent; <= 6 CPUs / 12g)
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  cpu_lock "$nm"
  ram_wait 6
  docker run --rm --name "scaleup_song_${nm}_$$" --cpus "$cpus" --memory "$mem" -e PYTHONUNBUFFERED=1 -e TS_DIRNAME=$TSN \
    ${PILOT_CASE_ENV:+-e PILOT_CASE=$PILOT_CASE_ENV} ${EXTRA_ENV:+-e $EXTRA_ENV} "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  local rc=${PIPESTATUS[0]}
  cpu_unlock
  return $rc
}

# ---------------- fetch (pilot_fetch_song method) ----------------
stage_fget() {
  c=$1
  grep -q '"crc_ok": true' "$SRC_H/$c/fetch.json" 2>/dev/null && { echo "fetched (crc ok)"; return; }
  t0=$(date +%s)
  light_run get neve-build-stagea 2 2g python3 /sp/v5_va/pilot_fetch_song_get.py "$c"
  rc=$?
  timing "$c" fget $(( $(date +%s) - t0 )) $rc "RAR5 range read (pilot_fetch_song_get.py)"
  grep -q '"crc_ok": true' "$SRC_H/$c/fetch.json" 2>/dev/null || { echo "FETCH FAILED $c"; touch "$LOGD/fetch_failed_$c"; }
}

stage_fqc() {
  c=$1
  [ -f "$SRC_H/$c/qc_pass1.json" ] && [ -f "$DER_H/$c/ct_negzero-100.nii.gz" ] && [ -z "$FQC_FORCE" ] && { echo "qc_pass1 exists"; return; }
  grep -q '"crc_ok": true' "$SRC_H/$c/fetch.json" 2>/dev/null || { echo "not fetched"; return; }
  ov=$(eval echo "\${ARCH_TOP_Z_$c:-}")
  t0=$(date +%s)
  EXTRA_ENV=${ov:+ARCH_TOP_Z=$ov} cpu_run fqc neve-build-stagea 6 12g python3 /sp/v5_va/pilot_fetch_song_qc.py "$c"
  timing "$c" fqc $(( $(date +%s) - t0 )) $? "qc_pass1 + MIPs + side files (pilot_fetch_song_qc.py)${ov:+ ARCH_TOP_Z=$ov}"
}

stage_fvac() {
  c=$1
  [ -s "$SRC_H/$c/qc_va_check.png" ] && [ -s "$SRC_H/$c/qc_va_check.json" ] && [ -z "$FQC_FORCE" ] && { echo "va_check exists"; return; }
  [ -f "$SRC_H/$c/qc_pass1.json" ] || return
  t0=$(date +%s)
  cpu_run fvac neve-build-stagea 4 10g python3 /sp/v5_va/pilot_fetch_song_va_check2.py "$c"
  timing "$c" fvac $(( $(date +%s) - t0 )) $? "column-frame VA check (pilot_fetch_song_va_check2.py)"
}

stage_fzoom() {
  c=$1
  [ -s "$SRC_H/$c/qc_va_zoom.png" ] && { echo "va_zoom exists"; return; }
  [ -f "$SRC_H/$c/qc_va_check.json" ] || return
  t0=$(date +%s)
light_run fzoom neve-build-stagea 2 6g python3 /sp/v5_va/pilot_fetch_song_va_zoom.py "$c"
  timing "$c" fzoom $(( $(date +%s) - t0 )) $? "zoomed VA crops (pilot_fetch_song_va_zoom.py)"
}

stage_fmeta() {
  c=$1
  [ -f "$SRC_H/$c/qc_va_check.json" ] || return
  [ -f "$SRC_H/$c/meta.json" ] && [ -z "$FMETA_FORCE" ] && { echo "meta exists"; return; }
  t0=$(date +%s)
  light_run meta neve-build-stagea 1 3g bash -c "python3 /sp/v5_va/scaleup_song_fetch_picks.py $c && python3 /sp/v5_va/scaleup_song_fetch_meta.py $c"
  timing "$c" fmeta $(( $(date +%s) - t0 )) $? "picks + meta.json (scaleup_song_fetch_picks/meta.py)"
  [ -f "$SRC_H/$c/meta.json" ] && touch "$LOGD/fetch_done_$c"
}

# ---------------- TotalSegmentator / Arterial (pilot run5.sh stages) ----------------
stage_tsprep() {
  c=$1; d=$DER_H/$c/$TSN; mkdir -p "$d"
  [ -f "$d/segmentations/vertebrae_C1.nii.gz" ] || [ -f "$d/.ts_ml_done" ] || [ -f "$d/ct_ts_input.nii.gz" ] && { echo "tsprep: nothing to do"; return; }
  [ -f "$DER_H/$c/ct_negzero-100.nii.gz" ] || { echo "no negzero side file"; return; }
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
  for try in 1 2 3 4 5 6; do
    ram_wait 8 60
    gpu_lock "ts $c $TSN"
    t0=$(date +%s)
    bash $SP/v5_va/scaleup_song_tsrun.sh "$DER_C/$c/$TSN/ct_ts_input.nii.gz" "$DER_C/$c/$TSN/segmentations_ml.nii.gz" "${c}_${TSN}_total_v3" --task total_v3 --ml
    rc=$?
    gpu_unlock
    timing "$c" "ts_total_v3_ml_$TSN" $(( $(date +%s) - t0 )) $rc "TotalSegmentator total_v3 on the GPU (CTA grid, --ml) try $try"
    [ $rc -ne 3 ] && break
    sleep 60
  done
  printf '{"task": "total_v3", "tool": "TotalSegmentator 2.18.0 (neve-totalseg-gpu)", "grid": "the CTA grid (nearest-neighbour write-back of the 1.5 mm prediction)", "writer": "--ml multilabel, split by pilot_proc_song_tssplit.py", "input": "%s (see ts_input.json)", "why_not_total": "tag_totalseg.md: total_v3 C1 Dice 0.98 vs 0.92, CCA 0.92-0.95 vs 0.85-0.89, skull 0.93 vs 0.81-0.83; the V3-loop gate fails a real loop on total masks; tag_calibration gates set on total_v3", "rc": %s, "by": "scaleup_lane_song"}\n' "${TS_INPUT:-negzero}" $rc > "$d/task.json"
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
   light_run 1p5 neve-build-stagea 2 6g python3 /sp/v5_va/tag_totalseg_make1p5.py "$DER_C/$c/$TSN" "$DER_C/$c/${TSN}_1p5" | tail -3
    timing "$c" "make1p5_$TSN" $(( $(date +%s) - t0 )) 0 "1.5 mm RAS copy (tag_totalseg_make1p5.py unchanged)"
  fi
  [ -f "$d/segmentations/vertebrae_C1.nii.gz" ] && rm -f "$d/ct_ts_input.nii.gz"
}

stage_args() {
  c=$1; d=$DER_H/$c
  [ -f "$d/arterial/args_run.txt" ] && { echo "args exist: $(cat "$d/arterial/args_run.txt")"; return; }
  [ -f "$d/totalseg/segmentations/vertebrae_C1.nii.gz" ] && [ -f "$d/totalseg/segmentations/skull.nii.gz" ] || { echo "no TS masks for $c"; return; }
  [ -f "$SRC_H/$c/meta.json" ] || { echo "no meta.json for $c"; return; }
  t0=$(date +%s)
  light_run args neve-build-stagea 4 8g python3 /sp/v5_va/pilot_proc_song_args.py "$c"
  timing "$c" args $(( $(date +%s) - t0 )) $? "arterial_ts_args + two-limb check"
}

stage_arterial() {
  c=$1; d=$DER_H/$c; out=$DER_C/$c/arterial/runs/native
  [ -f "$d/arterial/.arterial_done" ] && { echo "arterial done"; return; }
  [ -f "$d/arterial/args_run.txt" ] || { echo "no args for $c"; return; }
  AIN=${ART_IN:-air1000}
  case "$AIN" in
    shipped) IN=$SRC_C/$c/ct.nii.gz;;
    air1000) IN=$DER_C/$c/ct_negzero_air1000.nii.gz
        [ -f "$d/ct_negzero_air1000.nii.gz" ] || light_run rw neve-build-stagea 2 8g python3 /sp/v5_va/pilot_proc_song_rewindow.py "$c"
        [ -d "$d/arterial/runs/native" ] && [ ! -f "$d/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ] && mv "$d/arterial/runs/native" "$d/arterial/runs/native_prev_$(date +%H%M%S)";;
    *) echo "ART_IN $AIN not supported in the scale-up"; return;;
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
    ram_wait 8 60
    gpu_lock "arterial $c attempt $attempt input $AIN"
    t0=$(date +%s)
    echo "[scaleup_song] run_arterial $IN -> $out  args: $a --no-tta"
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
        docker stop "arterial_native__$apid" >/dev/null 2>&1
        break
      fi
    done
    wait $apid
    rc=$?
    [ -n "$wd" ] && printf '{"case": "%s", "attempt": %s, "watchdog": "%s"}\n' "$c" $attempt "$wd" >> "$d/arterial/watchdog.json"
    gpu_unlock
    timing "$c" "arterial_attempt$attempt" $(( $(date +%s) - t0 )) $rc "$a --no-tta (input $AIN)"
    rm -f "$d/arterial/runs/native/cta.nii.gz" "$d/arterial/runs/native_attempt1_failed/cta.nii.gz"
  done
  printf '{"arterial_input": "%s", "path": "%s", "by": "scaleup_lane_song", "input_copy_deleted_after_run": true}\n' "$AIN" "$IN" > "$d/arterial/input_used.json"
  touch "$d/arterial/.arterial_done"
  [ "$AIN" = "air1000" ] && rm -f "$d/ct_negzero_air1000.nii.gz"
}

# ---------------- post (pilot run5.sh stages; limits capped at 6 CPUs / 12g) ----------------
stage_extract() {
  c=$1
  [ -f "$DER_H/$c/arterial/runs/native/extracranial_vessels/segmentation.nii.gz" ] || { echo "no Arterial segmentation for $c"; return; }
  for m in ${EXTRACT_MODES:-tsba ts}; do
    [ -f "$DER_H/$c/arterial/$m/va_extract.json" ] && continue
    t0=$(date +%s)
    cpu_run ex neve-build-stagea 6 12g bash /sp/v5_va/pilot_proc_song_extract.sh "$c" $m
    timing "$c" "extract_$m" $(( $(date +%s) - t0 )) $? "art_extract ($m)"
  done
}

stage_vmtk() {
  c=$1; t0=$(date +%s)
  dirs=""; for m in ${EXTRACT_MODES:-tsba ts}; do [ -f "$DER_H/$c/arterial/$m/va_extract.json" ] && dirs="$dirs $DER_C/$c/arterial/$m"; done
  [ -z "$dirs" ] && return
  need=0; for m in ${EXTRACT_MODES:-tsba ts}; do for v in rva lva; do D=$DER_H/$c/arterial/$m
    [ -f $D/${v}_stageA.vtp ] && [ -f $D/${v}_seeds.json ] && { [ ! -f $D/${v}_stageA_vmtk.json ] || [ $D/${v}_stageA.vtp -nt $D/${v}_stageA_vmtk.json ]; } && need=1; done; done
  [ $need -eq 0 ] && { echo "vmtk done"; return; }
  cpu_lock vmtk
  ram_wait 6
  docker run --rm --name scaleup_song_vmtk_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash /sp/v5_va/pilot_proc_song_vmtk.sh \
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
  PILOT_CASE_ENV=$c cpu_run gross neve-build-stagea 4 8g python3 /sp/v5_va/pilot_proc_song_gross.py
  timing "$c" gross $(( $(date +%s) - t0 )) $? "gross shape on the Arterial VMTK lines"
}

stage_ctseg() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  T=${PILOT_SEG_TAG:-ct_seg4}
  if [ ! -f "$DER_H/$c/ct_seg/$T/seg_qc.json" ]; then
    grep -q '"passes_as_v5_donor": true' "$DER_H/$c/qc/qc.json" "$DER_H/$c/qc/qc_ts.json" 2>/dev/null || { echo "no donor-passing side in $c (tsba or ts): no ct_seg"; return; }
    cpu_lock ctseg
    ram_wait 6
    docker run --rm --name scaleup_song_seg_$$ --cpus 6 --memory 12g -e PYTHONUNBUFFERED=1 -e VMR_OPEN_BRIGHT=1 \
      -e PILOT_CASE="$c" -e PILOT_SEG_TAG="$T" -e PILOT_CT="${PILOT_CT:-shipped}" -e PILOT_KEEP_CONNECTED="${PILOT_KEEP_CONNECTED:-1}" ${TAG_SE_R:+-e TAG_SE_R=$TAG_SE_R} "${MNT[@]}" -w /repo \
      ${CTSEG_IMAGE:-neve-build-stagea} python3 /sp/v5_va/${CTSEG_SCRIPT:-pilot_proc_song_ct_seg.py} 2>&1 | grep -v "$FILT"
    rc=${PIPESTATUS[0]}
    cpu_unlock
    timing "$c" "ctseg_$T" $(( $(date +%s) - t0 )) $rc "ct_seg4 seeded from the VMTK trace (${CTSEG_SCRIPT:-pilot_proc_song_ct_seg.py})"
  fi
  D=$DER_C/$c/ct_seg/$T
  if [ -d "$DER_H/$c/ct_seg/$T" ]; then
    t0=$(date +%s)
    cpu_lock cvmtk
    docker run --rm --name scaleup_song_cvmtk_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash -c \
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
    ram_wait 6
    docker run --rm --name scaleup_song_ms_$$ --cpus 6 --memory 12g -e PYTHONUNBUFFERED=1 -e PILOT_CASE="$c" \
      -e PILOT_SEG_TAGS="${PILOT_SEG_TAGS:-ct_seg4}" -e PILOT_CT="${PILOT_CT:-shipped}" "${MNT[@]}" -w /repo $img python3 /sp/v5_va/$scr 2>&1 | grep -v "$FILT"
    rc=${PIPESTATUS[0]}
    cpu_unlock
    timing "$c" "${scr%.py}" $(( $(date +%s) - t0 )) $rc "$scr"
  done
}

stage_protrusion() {
  c=$1; D=$DER_H/$c/ct_seg/ct_seg4
  [ -f $D/labels_kji.npz ] || { echo "no ct_seg4 label"; return; }
  [ -f $D/protrusion.json ] && [ -z "$PR_FORCE" ] && { echo "protrusion exists"; return; }
  t0=$(date +%s)
  cpu_run pr neve-build-stagea 4 10g python3 /sp/v5_va/pilot_proc_song_protrusion.py "$c"
  timing "$c" protrusion $(( $(date +%s) - t0 )) $? "protrusion / contamination (pilot_proc_song_protrusion.py)"
}

stage_fig() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || return
  [ -f "$DER_H/$c/qc/qc_${c}.png" ] && [ -z "$FIG_FORCE" ] && { echo "fig exists"; return; }
  PILOT_CASE_ENV=$c cpu_run fig neve-build-stagea 6 10g python3 /sp/v5_va/pilot_proc_song_fig.py "$c"
  timing "$c" fig $(( $(date +%s) - t0 )) $? "figures"
}

stage_collect() {
  c=$1; t0=$(date +%s)
  [ -f "$DER_H/$c/qc/qc.json" ] || { echo "no qc for $c"; return; }
  until mkdir "$LOGD/.collect_lock" 2>/dev/null; do sleep 3; done
  docker run --rm --name scaleup_song_col_$$ --cpus 2 --memory 4g "${MNT[@]}" -w /repo eve-training-fixed \
    python3 /sp/v5_va/scaleup_song_collect.py "$c" 2>&1 | grep -v "$FILT" | tail -5
  rc=${PIPESTATUS[0]}
  rm -rf "$LOGD/.collect_lock"
  timing "$c" collect $(( $(date +%s) - t0 )) $rc "results JSON (scaleup_song_collect.py)"
  [ $rc -eq 0 ] && touch "$LOGD/collect_done_$c"
}

STAGES=$stage
case "$stage" in
  fetch) STAGES="fget fqc fvac fzoom fmeta";;
  ts) STAGES="tsprep tsgpu tscpu";;
  art) STAGES="args arterial";;
  gpu) STAGES="tsprep tsgpu tscpu args arterial";;
  trace) STAGES="extract vmtk";;
  grade) STAGES="qc gross ctseg measure protrusion fig collect";;
  post) STAGES="extract vmtk qc gross ctseg measure protrusion fig collect";;
esac
for c in "${cases[@]}"; do
  [ -f "$LOGD/rejected_$c" ] && { echo "== $c rejected at fetch review: skipped"; continue; }
  for s in $STAGES; do
    [ -f "$LOGD/rejected_$c" ] && break
    [ -f "$LOGD/fetch_failed_$c" ] && [ "$s" != "fget" ] && break
    echo "== $(date '+%F %T') $c $s"
    T0=$(date +%s)
    stage_$s "$c" 2>&1 | tee -a "$LOGD/${c}_$s.log"
    echo "== $(date '+%F %T') $c $s done in $(( $(date +%s) - T0 )) s"
    timing "$c" "stage_$s" $(( $(date +%s) - T0 )) 0 "whole stage wall time"
  done
done
echo "== $(date '+%F %T') scaleup_song_run.sh $stage finished"
