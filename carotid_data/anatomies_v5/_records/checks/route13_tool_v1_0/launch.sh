#!/bin/bash
# The backup ROOT (13 pre-swap folders) replaces anatomies_v5 in this container; per-folder nested mounts came up empty.
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; export MSYS_NO_PATHCONV=1
SPW='C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad'
docker run -d --rm --name v5r10_check --cpus 4 --memory 6g -e PYTHONUNBUFFERED=1 \
  -v "D:\Arjun\neve:/repo:ro" -w /repo -v "$SPW:/sp" \
  -v "$SPW\v5_va\backup_pre_route13:/repo/carotid_data/anatomies_v5:ro" \
  -v "D:\Arjun\neve\carotid_data:/opt/eve_training/carotid_data:ro" \
  -v "$SPW\v5_va\backup_pre_route13:/opt/eve_training/carotid_data/anatomies_v5:ro" \
  -v "D:\Arjun\neve\topbrain_tools:/opt/eve_training/topbrain_tools:ro" \
  eve-training-fixed bash /sp/v5_va/r10check/inner.sh
