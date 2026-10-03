#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_v2patch.sh (= /sp/v5_va/scaleup_ia172_v2patch.sh in the build containers)
# scaleup_ia172_v2patch.sh <out>  (scaleup_lane_ia172 instance 2, 2026-10-02; v2 of this file 08:45, v1 in scaleup_work/)
# Writes <out> = the CURRENT scaleup_v2_post.sh (pipeline v2, owned by scaleup_fixes; not modified) with changes to its
# run() and light() helpers only (resources and file safety, not rules):
#  1. v2 stages measured single-core / I/O-bound in docker stats (qc ~1-12 % CPU while reading over the Windows bind
#     mount, ct_seg ~1.0 core, measure ~1.2; gross / v1rescue / protrusion / fig have no parallelism) run as LIGHT
#     containers: --cpus 2 (memory as v2 sets it, after v2's own 6 GB RAM wait) under the lane's light-slot locks
#     <cpu_lock>_v2light and <cpu_lock>_v2light_b (at most two at a time; slot 1 is shared with the post chain's VMTK),
#     so they overlap the lane's one heavy container. Every other v2 container (art_extract re-trace, VMTK) keeps v2's
#     heavy lock. LIGHT_V2 env overrides the light stage set (default " qc gross seg ms pr fig v1r ").
#  2. light() takes a results lock (<logd>/.results_lock) around collect / grade, which read-modify-write
#     results/scaleup_ia172.json, so two grade chains cannot lose a record.
# Commands, images, env and outputs are unchanged. Regenerated before every case, so later fixes to scaleup_v2_post.sh
# flow through. If the patch does not apply, <out> is the unpatched v2 driver.
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
SRC=$SP/v5_va/scaleup_v2_post.sh; out=$1
awk '
/^run\(\) \{/ && !donerun {
  print "CLOCK2=${CLOCK}_v2light   # scaleup_ia172_v2patch.sh"
  print "LIGHT_V2=${LIGHT_V2:-\" qc gross seg ms pr fig v1r \"}"
  print "lock2() { while :; do for s_ in \"$CLOCK2\" \"${CLOCK2}_b\"; do"
  print "            if mkdir \"$s_\" 2>/dev/null; then L2HELD=$s_; echo \"v2light $$ $(date +%s) $1\" > \"$s_/owner\"; return 0; fi"
  print "            if [ -f \"$s_/owner\" ]; then p=$(awk \x27{print $2}\x27 \"$s_/owner\"); [ -n \"$p\" ] && ! kill -0 \"$p\" 2>/dev/null && rm -rf \"$s_\"; fi"
  print "          done; sleep 5; done; }"
  print "unlock2() { rm -rf \"$L2HELD\"; }"
  print "RLOCK=$LOGD/.results_lock"
  print "rlock() { until mkdir \"$RLOCK\" 2>/dev/null; do if [ -f \"$RLOCK/owner\" ]; then p=$(awk \x27{print $1}\x27 \"$RLOCK/owner\"); [ -n \"$p\" ] && ! kill -0 \"$p\" 2>/dev/null && rm -rf \"$RLOCK\" && continue; fi; sleep 2; done; echo \"$$ $(date +%s)\" > \"$RLOCK/owner\"; }"
  print "runlock() { rm -rf \"$RLOCK\"; }"
  print "run() {   # heavy, or LIGHT (2 CPUs, lock2) for the single-core stages in LIGHT_V2 (scaleup_ia172_v2patch.sh)"
  print "  local nm=$1 img=$2 cpus=$3 mem=$4; shift 4"
  print "  local lt=0; case \"$LIGHT_V2\" in *\" $nm \"*) lt=1; cpus=2;; esac"
  print "  if [ $lt = 1 ]; then lock2 \"$nm\"; else lock \"$nm\"; fi; memwait"
  print "  docker run --rm --name \"${NM}_${nm}_$$\" --cpus \"$cpus\" --memory \"$mem\" \"${ENVS[@]}\" \"${XENV[@]}\" \"${MNT[@]}\" -w /repo \"$img\" \"$@\" 2>&1 | grep -v \"$FILT\""
  print "  local rc=${PIPESTATUS[0]}; if [ $lt = 1 ]; then unlock2; else unlock; fi; return $rc"
  print "}"
  skip=1; donerun=1; next
}
/^light\(\) \{/ && !donelight {
  print "light() {   # light (json only): name image cmd...; collect / grade under the results lock (scaleup_ia172_v2patch.sh)"
  print "  local nm=$1 img=$2; shift 2"
  print "  local rl=0; case \"$nm\" in col|grd) rl=1; rlock;; esac"
  print "  docker run --rm --name \"${NM}_${nm}_$$\" --cpus 1 --memory 3g \"${ENVS[@]}\" \"${MNT[@]}\" -w /repo \"$img\" \"$@\" 2>&1 | grep -v \"$FILT\""
  print "  local rc=${PIPESTATUS[0]}; [ $rl = 1 ] && runlock; return $rc"
  print "}"
  skip=1; donelight=1; next
}
skip && /^\}/ { skip=0; next }
skip { next }
{ print }' "$SRC" > "$out.tmp"
n_src=$(grep -c 'docker run --rm --name "${NM}_${nm}_$$"' "$SRC"); n_out=$(grep -c 'docker run --rm --name "${NM}_${nm}_$$"' "$out.tmp")
if grep -q "scaleup_ia172_v2patch" "$out.tmp" && grep -q "^rlock()" "$out.tmp" && bash -n "$out.tmp" 2>/dev/null && [ "$n_src" -eq "$n_out" ] && \
   grep -q 'docker run --rm --name "${NM}_${nm}_$$" --cpus 1 --memory 3g' "$SRC"; then
  mv -f "$out.tmp" "$out"; echo "v2patch: applied ($(date '+%F %T'), source $(stat -c %y "$SRC" | cut -c1-19))"
else
  cp "$SRC" "$out"; rm -f "$out.tmp"; echo "v2patch: NOT applied -> unpatched v2 driver"
fi
