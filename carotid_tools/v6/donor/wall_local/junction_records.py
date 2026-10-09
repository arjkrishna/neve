#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_local\junction_records.py (= /sp/v6/wall_local/junction_records.py in the build containers)
"""junction_records.py (s2_harden_wall_bake, D15 for item e): the VA tool's junction records for the LOCAL junctions =
v6_va's sc_post3.py (imported unchanged) with its two folders pointed at the local walls:
  WALL /repo/carotid_data/v6_work/wall/subclavian_local (surfaces, junction_labels_kji.npz, seg_qc.json whose
       v6.junction_bridge.<x>junction is the LOCAL mask's bottleneck re-measure, marker .junction_v3);
  OUT  /v6w/donors_local/_junction_records/<donor>/<R|L>/junction.json   (graft_v6_va: V6VA_SEG=/repo/carotid_data/
       v6_work/wall/donors_local/_junction_records)
    python3 junction_records.py [donor ...]        (neve-build-stagea)"""
import sys
sys.path.insert(0, "/sp/v6/va")
import sc_post3 as S3  # noqa: E402
S3.WALL = "/repo/carotid_data/v6_work/wall/subclavian_local"
S3.OUT = "/v6w/donors_local/_junction_records"
if __name__ == "__main__":
    import os
    os.makedirs(S3.OUT, exist_ok=True)
    S3.main()
