#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_proc_ia172_rescue.sh (= /sp/v5_va/pilot_proc_ia172_rescue.sh in the build containers)
# pilot_proc_ia172_rescue.sh <case> [...] -- RESCUE retrace of a side whose trace failed because Arterial's labels were
# wrong (e.g. ia172_001 L: no labelled LVA junction, a wrong LSA label -> multi-start from its highest point, junction
# 59.8 mm from the TS subclavian). Runs pilot_proc_ia172_art_extract.py (the same basilar-fix copy) on a VIEW of the
# Arterial run without labelled_centerlines.json, i.e. art_extract's documented no-label path: start from the TS
# subclavian mask, TS CCA masks impassable (no Arterial carotid labels) -> arterial/tsba_nl; VMTK (art_vmtk_one.py
# unchanged); QC with pilot_proc_ia172_qc2.py (PILOT_TRACE_DIR=tsba_nl -> qc/qc_tsba_nl.json). The primary verdict stays
# the tsba one; a rescue pass is reported separately. CPU lock with 0.2 s polling. Log _logs_proc/<case>_rescue.log.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
DER_H=/d/Arjun/neve/carotid_data/va_sources/derived/thin_slice/ia172
DER_C=/repo/carotid_data/va_sources/derived/thin_slice/ia172
LOGD=$DER_H/_logs_proc
CLOCK=$LOGD/.cpu_lock
MNT=(-v "D:\Arjun\neve:/repo" -v "$SPW:/sp" -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data"
     -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools")
FILT='^==\|^$\|CUDA Version\|NVIDIA Deep\|By pulling\|governed\|NGC-DL\|nvidia.com\|Driver was not\|Container Toolkit\|Container image\|NVIDIA CORPORATION\|GPU functionality\|NVIDIA Container\|cloud-native'
lock() { until mkdir "$CLOCK" 2>/dev/null; do sleep 0.2; done; echo "rescue $$ $(date +%s)" > "$CLOCK/owner"; }
unlock() { rm -rf "$CLOCK"; }
for c in "$@"; do
  {
  echo "== $(date '+%F %T') $c rescue (no-label retrace)"
  t0=$(date +%s)
  R=$DER_C/$c/arterial
  if [ ! -f $DER_H/$c/arterial/tsba_nl/va_extract.json ]; then
    lock; docker run --rm --name pilot_proc_ia172_rsx_$$ --cpus 8 --memory 18g -e PYTHONUNBUFFERED=1 "${MNT[@]}" -w /repo neve-build-stagea bash -c \
      "mkdir -p /tmp/case && for f in extracranial_vessels run_report.json prep.json skull_mask.nii.gz; do ln -s $R/runs/native/\$f /tmp/case/\$f; done && \
       python3 /sp/v5_va/pilot_proc_ia172_art_extract.py /tmp/case $R/tsba_nl --ts $DER_C/$c/totalseg_1p5 --tag ${c}_tsba_nl" 2>&1 | grep -v "$FILT"; unlock
  fi
  lock; docker run --rm --name pilot_proc_ia172_rsv_$$ -u 0 --cpus 4 --memory 8g "${MNT[@]}" neve-build-vmtk bash /sp/v5_va/pilot_proc_ia172_vmtk.sh $R/tsba_nl \
    2>&1 | grep -v "^Reading\|^Cleaning\|^Triangulating\|^Computing"; unlock
  lock; docker run --rm --name pilot_proc_ia172_rsq_$$ --cpus 6 --memory 12g -e PYTHONUNBUFFERED=1 -e PILOT_TRACE_DIR=tsba_nl "${MNT[@]}" -w /repo \
    neve-build-stagea python3 /sp/v5_va/pilot_proc_ia172_qc2.py $c 2>&1 | grep -v "$FILT"; unlock
  printf '{"case": "%s", "stage": "rescue_tsba_nl", "seconds": %s, "rc": 0, "note": "no-label retrace", "end": "%s"}\n' "$c" $(( $(date +%s) - t0 )) "$(date '+%F %T')" >> $LOGD/timing.jsonl
  echo "== $(date '+%F %T') $c rescue done in $(( $(date +%s) - t0 )) s"
  } 2>&1 | tee -a $LOGD/${c}_rescue.log
done
