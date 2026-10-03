#!/bin/bash
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\tag_totalseg_run.sh (= /sp/v5_va/tag_totalseg_run.sh in the build containers)
# tag_totalseg_run.sh <input_ct> <out_seg_dir> <name> [extra TotalSegmentator args]
#
# Git Bash on the host. Runs TotalSegmentator 2.18.0 (image neve-totalseg-gpu, Dockerfile in
# /sp/v5_va/docker_totalseg/) on the RTX A5000, task "total" unless --task is given in the extra args.
#   <input_ct>, <out_seg_dir>: CONTAINER paths (/repo/... = D:\Arjun\neve, /sp/... = scratchpad).
#   Writes one <class>.nii.gz per class into <out_seg_dir>, on the input grid, and a log
#   <out_seg_dir>/../tag_totalseg_<name>.log (stdout+stderr, unfiltered: CUDA errors stay visible).
# Weights are cached in D:\Arjun\neve\carotid_data\va_sources\totalseg_weights (= TOTALSEG_HOME_DIR /ts_home);
# its config.json has send_usage_stats=false (TotalSegmentator would otherwise post usage statistics).
# Refuses to start while any other container holds a GPU (machine rule: one GPU container at a time).
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
IN=$1; OUT=$2; NAME=$3; shift 3
TASKARG="--task total"
case " $* " in *" --task "*) TASKARG="";; esac
for c in $(docker ps -q); do
  if [ "$(docker inspect -f '{{json .HostConfig.DeviceRequests}}' "$c")" != "null" ]; then
    echo "[tag_totalseg_run] another GPU container is running: $(docker inspect -f '{{.Name}} {{.Config.Image}}' "$c") -- not starting"
    exit 3
  fi
done
HOSTLOG_DIR=$(dirname "$OUT")
LOG_C="$HOSTLOG_DIR/tag_totalseg_$NAME.log"
docker run --rm --name "tag_totalseg_$NAME" --gpus all --cpus 8 --memory 16g --shm-size 2g \
  -e TOTALSEG_HOME_DIR=/ts_home -e PYTHONUNBUFFERED=1 \
  -v "D:\\Arjun\\neve\\carotid_data\\va_sources\\totalseg_weights:/ts_home" \
  -v "D:\\Arjun\\neve:/repo" -v "$SPW:/sp" \
  neve-totalseg-gpu bash -c "mkdir -p '$OUT' && /opt/conda/bin/python - $IN $OUT $TASKARG $* > '$LOG_C' 2>&1 <<'EOF'
import resource, subprocess, sys, time, json
import torch
from importlib.metadata import version
inp, out, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'TotalSegmentator', version('TotalSegmentator'),
      'nnunetv2', version('nnunetv2'), flush=True)
cmd = ['/opt/conda/bin/TotalSegmentator', '-i', inp, '-o', out, '--device', 'gpu'] + extra
print('cmd', ' '.join(cmd), flush=True)
t = time.time()
rc = subprocess.call(cmd)
print(json.dumps({'rc': rc, 'seconds': round(time.time() - t, 1),
                  'max_child_rss_gb': round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)}), flush=True)
sys.exit(rc)
EOF
rc=\$?; tail -5 '$LOG_C'; exit \$rc"
