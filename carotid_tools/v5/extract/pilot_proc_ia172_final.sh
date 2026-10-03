#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_final.sh (= /sp/v5_va/pilot_proc_ia172_final.sh in the build containers)
# pilot_proc_ia172_final.sh -- the closing pass after the post loop (Git Bash on the host). Idempotent; log
# _logs_proc/final.log. Steps:
#  1. 058 wall with fix 8 (its first ct_seg4 runs predate diffs 7 / 8; the plain result is kept as ct_seg4_nofix)
#  2. plain ct_seg4 (PILOT_KEEP_CONNECTED=0) as ct_seg/ct_seg4_nofix + landmarks / measure for every case where fix 8
#     acted (seg_qc.json open_bright.<v>.keep_connected_fix8.runs_not_opened_s_mm non-empty): the record of the plain method
#  3. protrusion.json for every case with a wall
#  4. figures again for every case (FIG_FORCE: the QC panel axes were fixed after the first cases)
#  5. collect all 12 cases, then the summary tables
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/ia172
LOGD=$DER_H/_logs_proc
CLOCK=$LOGD/.cpu_lock
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'
ALL="ia172_058 ia172_061 ia172_140 ia172_141 ia172_001 ia172_039 ia172_041 ia172_073 ia172_190 ia172_015 ia172_118 ia172_090"
lock() { until mkdir "$CLOCK" 2>/dev/null; do sleep 0.2; done; echo "final $$ $(date +%s)" > "$CLOCK/owner"; }
unlock() { rm -rf "$CLOCK"; }
{
echo "== $(date '+%F %T') final pass"
# 1
[ -f $DER_H/ia172_058/ct_seg/ct_seg4/measure.json ] || FIG_FORCE=1 bash $SP/v5_va/pilot_proc_ia172_run2.sh wall ia172_058
# 2
FIX=$(docker run --rm -v "D:\\Arjun\\neve:/repo" neve-build-stagea python3 -c "
import json, os, glob
for p in sorted(glob.glob('$DER_C/ia172_*/ct_seg/ct_seg4/seg_qc.json')):
    q = json.load(open(p))
    if any(((v or {}).get('keep_connected_fix8') or {}).get('runs_not_opened_s_mm') for v in (q.get('open_bright') or {}).values()):
        print(p.split('/')[-4])
" 2>/dev/null | grep '^ia172_')
echo "fix 8 acted on: $FIX"
for c in $FIX; do
  [ "$c" = "ia172_058" ] && continue          # 058's plain run (with the seed repair) is ct_seg4_nofix already
  if [ ! -f $DER_H/$c/ct_seg/ct_seg4_nofix/seg_qc.json ]; then
    lock; docker run --rm --name pilot_proc_ia172_nf_$$ --cpus 8 --memory 16g -e PYTHONUNBUFFERED=1 -e VMR_OPEN_BRIGHT=1 -e PILOT_KEEP_CONNECTED=0 \
      -e PILOT_CASE=$c -e PILOT_SEG_TAG=ct_seg4_nofix "${MNT[@]}" -w /repo neve-build-stagea python3 /sp/v5_va/pilot_proc_ia172_ct_seg.py 2>&1 | grep -v "$FILT" | tail -5; unlock
    D=$DER_C/$c/ct_seg/ct_seg4_nofix
    lock; docker run --rm --name pilot_proc_ia172_nfv_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash -c \
      "for v in rva lva; do [ -f $D/\${v}_stageA.vtp ] && [ -f $D/\${v}_seeds.json ] && [ ! -f $D/\${v}_stageA_vmtk.json ] && python3 /sp/v5_va/vmr_vmtk.py $D/\${v}_stageA.vtp $D/\${v}_seeds.json $D/\${v}_stageA_vmtk.json; done; true" 2>&1 | tail -2; unlock
  fi
  for scr in pilot_proc_ia172_landmarks.py pilot_proc_ia172_measure.py; do
    img=eve-training-fixed; case $scr in *landmarks*) img=neve-build-stagea;; esac
    out=landmarks.json; case $scr in *measure*) out=measure.json;; esac
    [ -f $DER_H/$c/ct_seg/ct_seg4_nofix/$out ] && continue
    lock; docker run --rm --name pilot_proc_ia172_nfm_$$ --cpus 8 --memory 14g -e PYTHONUNBUFFERED=1 -e PILOT_CASE=$c -e PILOT_SEG_TAGS=ct_seg4_nofix \
      "${MNT[@]}" -w /repo $img python3 /sp/v5_va/$scr 2>&1 | grep -v "$FILT" | grep "k64\|done"; unlock
  done
done
# 3
W=""; for c in $ALL; do [ -f $DER_H/$c/ct_seg/ct_seg4/labels_kji.npz ] && W="$W $c"; done
lock; docker run --rm --name pilot_proc_ia172_pr_$$ --cpus 4 --memory 10g "${MNT[@]}" -w /repo neve-build-stagea python3 /sp/v5_va/pilot_proc_ia172_protrusion.py $W 2>&1 | grep -v "$FILT"; unlock
# 4
for c in $ALL; do
  [ -f $DER_H/$c/qc/qc.json ] || continue
  lock; docker run --rm --name pilot_proc_ia172_fg_$$ --cpus 6 --memory 10g -e PILOT_CASE=$c -e PYTHONUNBUFFERED=1 "${MNT[@]}" -w /repo neve-build-stagea \
    python3 /sp/v5_va/pilot_proc_ia172_fig.py $c 2>&1 | grep -v "$FILT" | grep -v "^wrote"; unlock
done
# 5
docker run --rm --name pilot_proc_ia172_cl_$$ --cpus 2 --memory 4g "${MNT[@]}" -w /repo eve-training-fixed bash -c \
  "python3 /sp/v5_va/pilot_proc_ia172_collect.py $ALL | tail -14 && python3 /sp/v5_va/pilot_proc_ia172_summary.py > /dev/null && echo summary written" 2>&1 | grep -v "$FILT"
echo "== $(date '+%F %T') final pass done"
} 2>&1 | tee -a $LOGD/final.log
