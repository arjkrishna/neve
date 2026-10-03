#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_v2_post.sh (= /sp/v5_va/scaleup_v2_post.sh in the build containers)
# scaleup_v2_post.sh <source> <stage|group> <case> [case ...]   (Git Bash; scaleup_fixes, 2026-10-02)
#
# PIPELINE v2, POST stage (QC onward) for one source (ia172 | song) = the pilot post stage (pilot_proc_ia172_run2.sh /
# pilot_proc_song_run5.sh + their closing passes) with the two pre-bulk fixes and the both-source trace selection:
#   qc         the UNCHANGED pilot gate pilot_proc_<src>_qc2.py on BOTH traces: tsba -> qc/qc.json, ts -> qc/qc_ts.json
#              (ia172: PILOT_TRACE_DIR=ts, new in v2; song: qc2 grades ts itself and writes the pilot select.json)
#   select     scaleup_select.py -> qc/<SCALEUP_SELECT>   (level 1 = the Song pilot rule over tsba / ts)
#   rescue     FIX 1 -- scaleup_v1rescue.py (qc/v1rescue.json; when a side is rescued: arterial/runs/native_v1r = the patched
#              Arterial mask) -> re-trace with the base trace set's own tool (tsba: pilot_art_extract.py, ts: art_extract.py)
#              into arterial/<e>_v1r -> VMTK (pilot_proc_<src>_vmtk.sh) -> the unchanged QC on it (qc/qc_<e>_v1r.json)
#              -> scaleup_select.py again (level 2: the rescued trace replaces level 1 when it passes and level 1 fails or is
#              only a provisional V1 <= 3 mm pass)
#   gross      scaleup_v2_gross.py (pilot gross on the selected trace) -> qc/<SCALEUP_GROSS>
#   ctseg      FIX 2 -- scaleup_ct_seg.py (pilot ct_seg4 + the C2-exit pinch fix, seeds from the selected traces)
#              + VMTK MISR of its stage-A surfaces (vmr_vmtk.py unchanged) -> ct_seg/<PILOT_SEG_TAG>
#   measure    pilot_proc_<src>_{landmarks,measure,edge}.py UNCHANGED (PILOT_SEG_TAGS = the tag)
#   protrusion scaleup_v2_protrusion.py (the pilot protrusion, imported, on the tag)
#   fig        scaleup_v2_fig.py (pilot figures on the selected traces; names + SCALEUP_FIG_SUFFIX)
#   collect    scaleup_v2_collect.py -> SCALEUP_RESULTS (one record per case and side, pilot fields + v2 fields)
#   grade      scaleup_v2_grade.py   -> <SCALEUP_RESULTS>_grade.json/.txt (+ grade_v5 merged into the results file)
# groups: post2 = qc select rescue gross ctseg measure protrusion fig collect grade   (grade: once, after the loop)
#         wall2 = ctseg measure protrusion fig collect grade
# Prerequisites per case (unchanged pilot stages, run by the lane before this): Arterial (arterial/.arterial_done),
#   extract tsba + ts (arterial/{tsba,ts}/va_extract.json), VMTK on both (pilot_proc_<src>_vmtk.sh), TS totalseg_1p5.
# Environment (defaults in brackets):
#   V2_LOGD [<derived>/<src>/_logs_scaleup]     per-case logs <case>_v2_<stage>.log + timing.jsonl ("by": "scaleup_v2_post")
#   V2_CPU_LOCK [$V2_LOGD/.cpu_lock] the caller's mkdir lock (1 heavy container); V2_CPU_WANT []: no lock while it has a file
#   SCALEUP_RESULTS [/sp/v5_va/results/scaleup_<src>.json]   PILOT_SEG_TAG [ct_seg4]   SCALEUP_SELECT [select.json]
#   SCALEUP_GROSS [gross.json]   SCALEUP_FIG_SUFFIX []   V2_NAME [scaleup_v2_<src>] (container name prefix)
#   V2_FORCE="qc select rescue gross ctseg measure protrusion fig" -> redo those stages
# Heavy containers: <= 6 CPUs, <= 12 GB; each waits until the WSL VM has >= 6 GB MemAvailable. Every stage is idempotent.
SRCN=$1; stage=$2; shift 2
XENV=()
case "$SRCN" in ia172|song) ;; *) echo "usage: scaleup_v2_post.sh <ia172|song> <stage|group> <case> [...]"; exit 2;; esac
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/$SRCN
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/$SRCN
LOGD=${V2_LOGD:-$DER_H/_logs_scaleup}; mkdir -p "$LOGD"
CLOCK=${V2_CPU_LOCK:-$LOGD/.cpu_lock}
TAG=${PILOT_SEG_TAG:-ct_seg4}
SELF=${SCALEUP_SELECT:-select.json}
GROSSF=${SCALEUP_GROSS:-gross.json}
RESF=${SCALEUP_RESULTS:-/sp/v5_va/results/scaleup_${SRCN}.json}
NM=${V2_NAME:-scaleup_v2_${SRCN}}
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
ENVS=(-e PYTHONUNBUFFERED=1 -e PILOT_SOURCE=$SRCN -e PILOT_SEG_TAG=$TAG -e PILOT_SEG_TAGS=$TAG -e SCALEUP_SELECT=$SELF
      -e SCALEUP_GROSS=$GROSSF -e SCALEUP_RESULTS=$RESF -e SCALEUP_FIG_SUFFIX=${SCALEUP_FIG_SUFFIX:-}
      -e SCALEUP_TIMING=${DER_C}/$(basename "$LOGD")/timing.jsonl)
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'

forced() { case " ${V2_FORCE:-} " in *" $1 "*) return 0;; esac; return 1; }
memwait() { for i in $(seq 1 120); do a=$(docker run --rm --memory 64m alpine awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo 2>/dev/null)
            [ -n "$a" ] && [ "$a" -ge 6 ] && return; sleep 30; done; }
lock() { until { [ -z "$V2_CPU_WANT" ] || [ -z "$(ls -A "$V2_CPU_WANT" 2>/dev/null)" ]; } && mkdir "$CLOCK" 2>/dev/null; do
           if [ -f "$CLOCK/owner" ]; then p=$(awk '{print $2}' "$CLOCK/owner"); [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null && \
             [ "$(awk '{print $1}' "$CLOCK/owner")" = "v2post" ] && rm -rf "$CLOCK" && continue; fi
           sleep 5; done; echo "v2post $$ $(date +%s) $1" > "$CLOCK/owner"; }
unlock() { rm -rf "$CLOCK"; }
timing() { printf '{"case": "%s", "stage": "%s", "seconds": %s, "rc": %s, "note": "%s", "end": "%s", "by": "scaleup_v2_post"}\n' \
             "$1" "$2" "$3" "$4" "$5" "$(date '+%F %T')" >> "$LOGD/timing.jsonl"; }
run() {   # heavy: name image cpus mem cmd...
  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4
  lock "$nm"; memwait
  docker run --rm --name "${NM}_${nm}_$$" --cpus "$cpus" --memory "$mem" "${ENVS[@]}" "${XENV[@]}" "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  local rc=${PIPESTATUS[0]}; unlock; return $rc
}
light() {   # light (json only): name image cmd...
  local nm=$1 img=$2; shift 2
  docker run --rm --name "${NM}_${nm}_$$" --cpus 1 --memory 3g "${ENVS[@]}" "${MNT[@]}" -w /repo "$img" "$@" 2>&1 | grep -v "$FILT"
  return ${PIPESTATUS[0]}
}
vmtk_dirs() {   # VMTK MISR of the cut-out surfaces in the given trace dirs (pilot vmtk.sh, unchanged)
  lock vmtk; memwait
  docker run --rm --name "${NM}_vmtk_$$" -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash /sp/v5_va/pilot_proc_${SRCN}_vmtk.sh "$@" \
    2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
  unlock
}
qc_one() {   # $1 case, $2 trace dir (tsba | ts | tsba_v1r | ts_v1r) -> qc/qc.json | qc/qc_<dir>.json
  local c=$1 e=$2 f
  [ "$e" = tsba ] && f=qc.json || f=qc_$e.json
  [ -f "$DER_H/$c/arterial/$e/va_extract.json" ] || { echo "no $e trace"; return 0; }
  [ -f "$DER_H/$c/qc/$f" ] && ! forced qc && { echo "$f exists"; return 0; }
  if [ "$SRCN" = ia172 ]; then
    XENV=(-e PILOT_TRACE_DIR=$e); run qc neve-build-stagea 6 12g python3 /sp/v5_va/pilot_proc_ia172_qc2.py "$c"; rc_=$?; XENV=(); return $rc_
  else
    XENV=(-e PILOT_EXT=$e); run qc neve-build-stagea 6 12g python3 /sp/v5_va/pilot_proc_song_qc2.py "$c"; rc_=$?; XENV=(); return $rc_
  fi
}

stage_qc() {
  c=$1
  [ -f "$DER_H/$c/arterial/tsba/va_extract.json" ] || { echo "no tsba trace for $c (extract first)"; return 1; }
  qc_one "$c" tsba
  qc_one "$c" ts
}
stage_select() {
  c=$1
  light sel neve-build-stagea python3 /sp/v5_va/scaleup_select.py "$SRCN" "$c"
}
stage_rescue() {
  c=$1; D=$DER_H/$c
  if [ ! -f "$D/qc/v1rescue.json" ] || forced rescue; then
    run v1r neve-build-stagea 4 10g python3 /sp/v5_va/scaleup_v1rescue.py "$SRCN" "$c"
  fi
  exts=$(docker run --rm --memory 256m "${MNT[@]}" python:3 python3 -c "
import json; d = json.load(open('$DER_C/$c/qc/v1rescue.json')); print(' '.join(sorted(set(d.get('rescued_sides', {}).values()))))" 2>/dev/null)
  [ -z "$exts" ] && { echo "$c: no side rescued"; return 0; }
  for e in $exts; do
    t=${e}_v1r
    if [ ! -f "$D/arterial/$t/va_extract.json" ] || forced rescue; then
      tool=/sp/v5_va/art_extract.py; [ "$e" = tsba ] && tool=/sp/v5_va/pilot_art_extract.py
      run ex neve-build-stagea 6 12g python3 $tool "$DER_C/$c/arterial/runs/native_v1r" "$DER_C/$c/arterial/$t" \
        --ts "$DER_C/$c/totalseg_1p5" --tag "${c}_$t"
      vmtk_dirs "$DER_C/$c/arterial/$t"
    fi
    qc_one "$c" "$t"
  done
  light sel neve-build-stagea python3 /sp/v5_va/scaleup_select.py "$SRCN" "$c"
}
stage_gross() {
  c=$1
  [ -f "$DER_H/$c/qc/$GROSSF" ] && ! forced gross && { echo "gross exists"; return 0; }
  XENV=(-e PILOT_CASE=$c); run gross neve-build-stagea 4 8g python3 /sp/v5_va/scaleup_v2_gross.py; rc_=$?; XENV=(); return $rc_
}
stage_ctseg() {
  c=$1; D=$DER_H/$c/ct_seg/$TAG
  if [ ! -f "$D/seg_qc.json" ] || forced ctseg; then
    grep -q '"any_donor": true' "$DER_H/$c/qc/$SELF" 2>/dev/null || { echo "$c: no donor-passing side in $SELF: no ct_seg"; return 0; }
    XENV=(-e PILOT_CASE=$c -e VMR_OPEN_BRIGHT=1 -e PILOT_CT=shipped -e PILOT_KEEP_CONNECTED=1)
    run seg neve-build-stagea 6 12g python3 /sp/v5_va/scaleup_ct_seg.py; XENV=()
  fi
  if [ -d "$D" ]; then
    DC=$DER_C/$c/ct_seg/$TAG
    lock cvmtk; memwait
    docker run --rm --name "${NM}_cvmtk_$$" -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash -c \
      "for v in rva lva; do [ -f $DC/\${v}_stageA.vtp ] && [ -f $DC/\${v}_seeds.json ] && { [ ! -f $DC/\${v}_stageA_vmtk.json ] || [ $DC/\${v}_stageA.vtp -nt $DC/\${v}_stageA_vmtk.json ]; } && python3 /sp/v5_va/vmr_vmtk.py $DC/\${v}_stageA.vtp $DC/\${v}_seeds.json $DC/\${v}_stageA_vmtk.json; done; true" \
      2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"
    unlock
  fi
}
stage_measure() {
  c=$1; D=$DER_H/$c/ct_seg/$TAG
  ls $D/*_stageA_vmtk.json >/dev/null 2>&1 || { echo "no ct_seg centrelines for $c"; return 0; }
  for scr in landmarks measure edge; do
    case $scr in landmarks) f=$D/landmarks.json; img=neve-build-stagea;; measure) f=$D/measure.json; img=eve-training-fixed;;
                 edge) f=$D/edge.json; img=neve-build-stagea;; esac
    [ -f "$f" ] && [ "$f" -nt "$D/seg_qc.json" ] && ! forced measure && continue
    XENV=(-e PILOT_CASE=$c -e PILOT_CT=shipped -e PILOT_POOL=4); run ms $img 6 12g python3 /sp/v5_va/pilot_proc_${SRCN}_${scr}.py; XENV=()
  done
}
stage_protrusion() {
  c=$1; D=$DER_H/$c/ct_seg/$TAG
  [ -f "$D/labels_kji.npz" ] || return 0
  [ -f "$D/protrusion.json" ] && [ "$D/protrusion.json" -nt "$D/seg_qc.json" ] && ! forced protrusion && return 0
  run pr neve-build-stagea 4 10g python3 /sp/v5_va/scaleup_v2_protrusion.py "$c"
}
stage_fig() {
  c=$1
  [ -f "$DER_H/$c/qc/.fig_v2${SCALEUP_FIG_SUFFIX:-}" ] && ! forced fig && { echo "fig exists"; return 0; }
  XENV=(-e PILOT_CASE=$c); run fig neve-build-stagea 6 10g python3 /sp/v5_va/scaleup_v2_fig.py "$c" && touch "$DER_H/$c/qc/.fig_v2${SCALEUP_FIG_SUFFIX:-}"; rc_=$?; XENV=(); return $rc_
}
stage_collect() {
  c=$1
  light col eve-training-fixed python3 /sp/v5_va/scaleup_v2_collect.py "$c" | tail -3
}
stage_grade() {
  light grd eve-training-fixed python3 /sp/v5_va/scaleup_v2_grade.py | grep "^COUNTS\|^#\|Error\|error" | head -40
}

STAGES=$stage
case "$stage" in
  post2) STAGES="qc select rescue gross ctseg measure protrusion fig collect grade";;
  wall2) STAGES="ctseg measure protrusion fig collect grade";;
esac
DOGRADE=0
for c in "$@"; do
  for s in $STAGES; do
    [ "$s" = grade ] && { DOGRADE=1; continue; }
    echo "== $(date '+%F %T') v2 $SRCN $c $s"
    T0=$(date +%s)
    stage_$s "$c" 2>&1 | tee -a "$LOGD/${c}_v2_$s.log"
    rc=${PIPESTATUS[0]}
    timing "$c" "v2_$s" $(( $(date +%s) - T0 )) $rc "scaleup_v2_post.sh stage $s (tag $TAG, select $SELF)"
    echo "== $(date '+%F %T') v2 $SRCN $c $s done rc $rc in $(( $(date +%s) - T0 )) s"
  done
done
if [ $DOGRADE = 1 ]; then
  echo "== $(date '+%F %T') v2 $SRCN grade ($RESF)"
  stage_grade 2>&1 | tee -a "$LOGD/v2_grade.log"
fi
echo "== $(date '+%F %T') scaleup_v2_post.sh $SRCN $stage finished"
