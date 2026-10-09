#!/bin/bash
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\final_set.sh (= /sp/v6/s3fix/final_set.sh in the build containers)
# s3_fix: the set-level checks on the FINAL anatomies_v6 (after copy_v6_s3.sh): the split (v5_split.py --baseline
# split_v4.json: exit 0, none of the 45 quantities lowered) and the training loader over the whole folder (loader_set.py,
# s2_checks' tool). eve-training-fixed; /repo read-only. Outputs /s3/set/.
mkdir -p /s3/set
python3 /sp/v5_va/v5_split.py /repo/carotid_data/anatomies_v6 --baseline /sp/v5_va/results/split_v4.json --out /s3/set/split_s3.json \
  < /dev/null > /s3/set/split_s3.log 2>&1
echo "$(date '+%F %T') split rc $?" >> /s3/set/set.log
python3 /sp/v6/s2checks/loader_set.py /repo/carotid_data/anatomies_v6 /s3/set/loader_s3.json < /dev/null > /s3/set/loader_s3.log 2>&1
echo "$(date '+%F %T') loader rc $?" >> /s3/set/set.log
