#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_song_final.sh (= /sp/v5_va/pilot_proc_song_final.sh in the build containers)
# pilot_proc_song_final.sh -- the Song closing pass after post loop 2 (2nd instance, 2026-10-02 01:35; Git Bash on the host;
# modelled on pilot_proc_ia172_final.sh). Idempotent (every step skips existing outputs); log _logs_proc/final.log;
# done-marker _logs_proc/.final_done. One heavy container at a time (the run5.sh CPU lock, _logs_proc/.cpu_lock).
#  1. protrusion.json (pilot_proc_song_protrusion.py) for every case with a ct_seg4 wall
#  2. clipped-vs-re-windowed edge comparison on 3 more cases (100 R done 23:15): ct_seg4 on ct_negzero-100
#     (PILOT_CT=negzero, tag ct_seg4_negzero) + VMTK + landmarks + measure + edge via run5.sh, then pilot_proc_song_clipcmp.py
#     (merges per case; never re-run on 100, whose dirs were renamed)
#  3. plain ct_seg4 (PILOT_KEEP_CONNECTED=0) as ct_seg/ct_seg4_nofix + VMTK + landmarks + measure where fix 8 acted
#     (032 094 106 113; seg_qc open_bright.<v>.keep_connected_fix8.runs_not_opened_s_mm non-empty): the plain-method record
#  4. figures again for every case (FIG_FORCE; 100's last figure predates the 23:32 fig.py)
#  5. collect all 10 cases, then pilot_proc_song_summary.py
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP=/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/song
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/song
LOGD=$DER_H/_logs_proc
CLOCK=$LOGD/.cpu_lock
MNT=(-v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" -v "D:\\Arjun\\neve\\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\\Arjun\\neve\\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'
ALL="AHMU1218100 AHMU1218109 AHMU1218044 AHMU1218032 AHMU1218113 AHMU1218141 AHMU1218106 AHMU1218090 AHMU1218008 AHMU1218094"
CLIP="${CLIP_CASES:-AHMU1218113 AHMU1218008 AHMU1218032}"
FIX="${FIX_CASES:-AHMU1218032 AHMU1218094 AHMU1218106 AHMU1218113}"
RUN5=$SP/v5_va/pilot_proc_song_run5.sh
lock() { until mkdir "$CLOCK" 2>/dev/null; do sleep 2; done; echo "final $$ $(date +%s)" > "$CLOCK/owner"; }
unlock() { rm -rf "$CLOCK"; }
{
echo "== $(date '+%F %T') final pass (pid $$)"
# 1
W=""; for c in $ALL; do [ -f $DER_H/$c/ct_seg/ct_seg4/labels_kji.npz ] && [ ! -f $DER_H/$c/ct_seg/ct_seg4/protrusion.json -o -n "$PR_FORCE" ] && W="$W $c"; done
echo "== $(date '+%F %T') 1 protrusion:$W"
if [ -n "$W" ]; then
  lock; docker run --rm --name pilot_proc_song_pr_$$ --cpus 4 --memory 10g "${MNT[@]}" -w /repo neve-build-stagea \
    python3 /sp/v5_va/pilot_proc_song_protrusion.py $W 2>&1 | grep -v "$FILT"; unlock
fi
# 2
for c in $CLIP; do
  echo "== $(date '+%F %T') 2 clip comparison $c"
  [ "$c" = "AHMU1218100" ] && continue
  PILOT_CT=negzero PILOT_SEG_TAG=ct_seg4_negzero PILOT_SEG_TAGS=ct_seg4_negzero bash $RUN5 ctseg $c 2>&1 | grep -v "^piece\|^prepass\|^seeds\|^components\|^structuring" | tail -8
  PILOT_CT=negzero PILOT_SEG_TAG=ct_seg4_negzero PILOT_SEG_TAGS=ct_seg4_negzero bash $RUN5 measure $c 2>&1 | grep -v "^piece" | tail -6
  if [ -f $DER_H/$c/ct_seg/ct_seg4_negzero/edge.json ] && [ ! -f $DER_H/$c/ct_seg/ct_seg4_negzero/.clipcmp_done ]; then
    docker run --rm --name pilot_proc_song_clipcmp_$$ --cpus 1 --memory 3g "${MNT[@]}" -w /repo eve-training-fixed \
      python3 /sp/v5_va/pilot_proc_song_clipcmp.py $c 2>&1 | grep -v "$FILT" | cut -c1-400 && touch $DER_H/$c/ct_seg/ct_seg4_negzero/.clipcmp_done
  fi
done
# 3
for c in $FIX; do
  echo "== $(date '+%F %T') 3 plain ct_seg4 (nofix) $c"
  T=ct_seg4_nofix; D=$DER_C/$c/ct_seg/$T
  if [ ! -f $DER_H/$c/ct_seg/$T/seg_qc.json ]; then
    lock; docker run --rm --name pilot_proc_song_nf_$$ --cpus 8 --memory 16g -e PYTHONUNBUFFERED=1 -e VMR_OPEN_BRIGHT=1 -e PILOT_KEEP_CONNECTED=0 \
      -e PILOT_CASE=$c -e PILOT_SEG_TAG=$T "${MNT[@]}" -w /repo neve-build-stagea python3 /sp/v5_va/pilot_proc_song_ct_seg.py 2>&1 | grep -v "$FILT" | grep "^open_bright\|^rva \|^lva \|^done\|rror"; unlock
  fi
  if [ -d $DER_H/$c/ct_seg/$T ]; then
    lock; docker run --rm --name pilot_proc_song_nfv_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash -c \
      "for v in rva lva; do [ -f $D/\${v}_stageA.vtp ] && [ -f $D/\${v}_seeds.json ] && [ ! -f $D/\${v}_stageA_vmtk.json ] && python3 /sp/v5_va/vmr_vmtk.py $D/\${v}_stageA.vtp $D/\${v}_seeds.json $D/\${v}_stageA_vmtk.json; done; true" 2>&1 | tail -2; unlock
    for scr in pilot_proc_song_landmarks.py pilot_proc_song_measure.py; do
      img=eve-training-fixed; out=measure.json
      case $scr in *landmarks*) img=neve-build-stagea; out=landmarks.json;; esac
      [ -f $DER_H/$c/ct_seg/$T/$out ] && continue
      lock; docker run --rm --name pilot_proc_song_nfm_$$ --cpus 8 --memory 14g -e PYTHONUNBUFFERED=1 -e PILOT_CASE=$c -e PILOT_SEG_TAGS=$T \
        "${MNT[@]}" -w /repo $img python3 /sp/v5_va/$scr 2>&1 | grep -v "$FILT" | grep "k64\|done\|rror" | cut -c1-300; unlock
    done
  fi
done
# 4
echo "== $(date '+%F %T') 4 figures"
for c in $ALL; do
  [ -f $DER_H/$c/qc/qc.json ] || continue
  [ -f $DER_H/$c/qc/.fig_final ] && continue
  lock; docker run --rm --name pilot_proc_song_fg_$$ --cpus 6 --memory 10g -e PILOT_CASE=$c -e PYTHONUNBUFFERED=1 "${MNT[@]}" -w /repo neve-build-stagea \
    python3 /sp/v5_va/pilot_proc_song_fig.py $c 2>&1 | grep -v "$FILT" | grep -v "^wrote" | tail -3 && touch $DER_H/$c/qc/.fig_final; unlock
done
# 5
echo "== $(date '+%F %T') 5 collect + summary"
docker run --rm --name pilot_proc_song_cl_$$ --cpus 2 --memory 4g "${MNT[@]}" -w /repo eve-training-fixed bash -c \
  "python3 /sp/v5_va/pilot_proc_song_collect.py $ALL | tail -12 && (test -f /sp/v5_va/pilot_proc_song_summary.py && python3 /sp/v5_va/pilot_proc_song_summary.py > /dev/null && echo summary written || echo no summary)" 2>&1 | grep -v "$FILT"
touch $LOGD/.final_done
echo "== $(date '+%F %T') final pass done"
} 2>&1 | tee -a $LOGD/final.log
