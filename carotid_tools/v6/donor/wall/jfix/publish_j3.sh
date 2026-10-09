#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\jfix\publish_j3.sh (= /sp/v6/wall/jfix/publish_j3.sh in the build containers)
# v6_wall (e): publish the j3 subclavian outputs (sub label cleanup + junction bridge) over subclavian/<u>/ (host side).
# Only when the VA surfaces and labels_kji.npz are bit-identical to the first run and the sub VMTK lines exist.
# Previous files -> subclavian/<u>/old_pre_j3/. Idempotent (.junction_v3). Usage: bash publish_j3.sh [unit ...]
W=/d/Arjun/neve/carotid_data/v6_work/wall
UNITS="$@"; [ -z "$UNITS" ] && UNITS=$(ls $W/subclavian_j3 | grep -v '\.log$')
FILES="rjunction_stageA.vtp ljunction_stageA.vtp rsub_stageA.vtp lsub_stageA.vtp rsub_seeds.json lsub_seeds.json rsub_stageA_vmtk.json lsub_stageA_vmtk.json seg_qc.json junction_labels_kji.npz"
for u in $UNITS; do
  N=$W/subclavian_j3/$u; O=$W/subclavian/$u
  [ -f $N/.seg_done ] || { echo "$u: not done"; continue; }
  [ -f $O/.junction_v3 ] && { echo "$u: already published"; continue; }
  ok=1
  for f in rva_stageA.vtp lva_stageA.vtp labels_kji.npz; do
    [ -f $O/$f ] || continue
    a=$(md5sum < $O/$f | cut -c1-32); b=$(md5sum < $N/$f | cut -c1-32)
    [ "$a" = "$b" ] || { echo "$u: $f DIFFERS -- not published"; ok=0; }
  done
  for v in rsub lsub; do
    [ -f $N/${v}_stageA.vtp ] && [ ! -f $N/${v}_stageA_vmtk.json ] && { echo "$u: $v VMTK missing -- not published"; ok=0; }
  done
  [ $ok = 1 ] || continue
  mkdir -p $O/old_pre_j3
  for f in $FILES; do
    [ -f $O/$f ] && [ ! -f $O/old_pre_j3/$f ] && cp -p $O/$f $O/old_pre_j3/$f
  done
  for f in $FILES; do
    [ -f $N/$f ] && cp -p $N/$f $O/.$f.tmp && mv -f $O/.$f.tmp $O/$f
  done
  cp -p $N/seg.log $O/seg_j3.log
  echo "v3 (jfix/junction6.py clean_sub + junction bridge) published $(date '+%F %T') from subclavian_j3/$u" > $O/.junction_v3
  echo "$u: published"
done
