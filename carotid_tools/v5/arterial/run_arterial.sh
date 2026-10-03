#!/usr/bin/env bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\run_arterial.sh (= /sp/v5_va/run_arterial.sh in the build containers)
# run_arterial.sh <input_ct.nii.gz> <outdir> [options passed to arterial_cpu_run.py]
#
# Runs the Arterial head-and-neck CTA pipeline (FLOWCAT-CV/arterial @3ff58ab, weights Zenodo
# 22694951) CPU-only in the neve-arterial docker image, then summarises the vertebral arteries.
# Call from Git Bash on the Windows host. Paths may be given as host paths
# (D:\Arjun\neve\..., D:/Arjun/neve/..., /d/Arjun/neve/...), as scratchpad paths, or as the
# container paths /repo/... and /sp/... ; they must live under D:\Arjun\neve or the scratchpad.
#
# Options (forwarded; see arterial_cpu_run.py --help):
#   --stages core|full|seg,cl,lm,vl,fe,ap   default core = segmentation, centerlines, labelling
#   --fast                 3d_lowres (~1.2 mm) on the whole volume; no head/neck split
#   --skull-mask <nii>     skull mask on the input grid (TotalSegmentator skull.nii.gz) for the
#                          head/neck split instead of the slow 0.5 mm craniofacial model
#   --slicer ts|laplacian|mask
#   --crop-mm z0:z1 / --crop-k k0:k1   S-axis slab of the input to keep
#   --resample-mm 0.5      resample the CT isotropically first (strongly advised for 1.5 mm CTs:
#                          masks/surfaces then come out on a 0.5 mm grid and thin VAs stay connected)
#   --fov-mm W [--fov-center x,y]   in-plane crop (Arterial's arch heuristics assume ~220 mm FOV)
#   --tta / --no-tta       nnU-Net mirroring TTA (default: off on CPU, ~5x cost, Dice 0.98 vs on;
#                          on with a GPU)
#   --threads N            torch threads (default $ARTERIAL_CPUS)
# Env: ARTERIAL_CPUS (default 12), ARTERIAL_MEM (default 20g), ARTERIAL_IMAGE (default neve-arterial =
#      py3.11 env "arterial"; neve-arterial:py310 also works -- the python is picked automatically),
#      ARTERIAL_GPU (default 1: --gpus all and the neve-arterial-gpu image, the RTX A5000 on this host;
#      ARTERIAL_GPU=0 runs CPU-only in neve-arterial)
#
# Outputs in <outdir>: cta.nii.gz (LAS copy actually processed), prep.json, run_report.json,
#   extracranial_vessels/... (Arterial outputs), va_summary.json, labelled_centerlines.json, arterial.log
#
# Licences: Arterial code PolyForm-NC-1.0.0, weights CC BY-NC 4.0, TotalSegmentator craniofacial
# model Apache-2.0. Research use only.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,34p' "$0"; exit 2
fi

export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"
export MSYS_NO_PATHCONV=1

SP_WIN='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
SP_UNIX='/c/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad'
REPO_WIN='D:\Arjun\neve'
MODELS_WIN='D:\Arjun\neve\carotid_data\va_sources\arterial\models'
GPU="${ARTERIAL_GPU:-1}"
if [ "$GPU" = "1" ]; then IMAGE="${ARTERIAL_IMAGE:-neve-arterial-gpu}"; GPUFLAG="--gpus all"; else IMAGE="${ARTERIAL_IMAGE:-neve-arterial}"; GPUFLAG=""; fi
CPUS="${ARTERIAL_CPUS:-12}"
MEM="${ARTERIAL_MEM:-20g}"

SP_FWD='C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-neve/afabf2e9-c644-45a6-95cd-d6e6d88be352/scratchpad'
to_ctr() {   # host or container path -> container path
    local p="${1//\\//}"
    case "$p" in
        /repo/*|/sp/*) printf '%s' "$p" ;;
        [Dd]:/[Aa]rjun/neve/*) printf '/repo/%s' "${p:14}" ;;     # strip "D:/Arjun/neve/"
        /[dD]/[Aa]rjun/neve/*) printf '/repo/%s' "${p:14}" ;;     # strip "/d/Arjun/neve/"
        [Cc]:/[Uu]sers/akrish41/*/scratchpad/*) printf '/sp/%s' "${p:${#SP_FWD}+1}" ;;
        /[cC]/[Uu]sers/akrish41/*/scratchpad/*) printf '/sp/%s' "${p:${#SP_UNIX}+1}" ;;
        *) echo "run_arterial.sh: path must be under D:\\Arjun\\neve or the scratchpad: $1" >&2; return 1 ;;
    esac
}
to_host() {  # container path -> Git Bash host path
    case "$1" in
        /repo/*) printf '/d/Arjun/neve/%s' "${1:6}" ;;
        /sp/*) printf '%s/%s' "$SP_UNIX" "${1:4}" ;;
    esac
}

IN_C="$(to_ctr "$1")"; OUT_C="$(to_ctr "$2")"; shift 2
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skull-mask) ARGS+=("$1" "$(to_ctr "$2")"); shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
OUT_H="$(to_host "$OUT_C")"
mkdir -p "$OUT_H"
[ -f "$(to_host "$IN_C")" ] || { echo "input not found: $(to_host "$IN_C")" >&2; exit 2; }

NAME="arterial_$(basename "$OUT_C" | tr -c 'A-Za-z0-9_.-' '_')_$$"
echo "[run_arterial] $IN_C -> $OUT_C  (image $IMAGE, gpu $GPU, cpus $CPUS, mem $MEM, container $NAME)"
WRAP=/sp/v5_va/arterial_cpu_run.py
SUMM=/sp/v5_va/arterial_va_summary.py

set +e
docker run --rm --name "$NAME" $GPUFLAG --cpus "$CPUS" --memory "$MEM" --shm-size 2g \
    -v "$REPO_WIN:/repo" -v "$SP_WIN:/sp" -v "$MODELS_WIN:/models:ro" \
    -e ARTERIAL_MODELS_DIR=/models -e OMP_NUM_THREADS="$CPUS" -e MKL_NUM_THREADS="$CPUS" \
    -e ARTERIAL_THREADS="$CPUS" -e nnUNet_def_n_proc="$CPUS" -e PYTHONUNBUFFERED=1 \
    "$IMAGE" bash -c "if [ -d /opt/conda/envs/arterial ]; then PY='micromamba run -n arterial python'; \
        else PY=/opt/conda/bin/python; fi; \
        \$PY $WRAP --input '$IN_C' --case '$OUT_C' ${ARGS[*]:-}; rc=\$?; \
        if [ -f '$OUT_C/extracranial_vessels/segments_graph_pred.pickle' ]; then \
            \$PY $SUMM '$OUT_C' || rc=3; fi; echo \"[run_arterial] container exit \$rc\"; exit \$rc" \
    2>&1 | grep --line-buffered -vE "Container image|By pulling|governed" | tee "$OUT_H/arterial.log"
RC=${PIPESTATUS[0]}
set -e
echo "[run_arterial] exit $RC; report: $OUT_H/run_report.json"
exit "$RC"
