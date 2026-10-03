# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_ia172_visual_set.py (= /sp/v5_va/scaleup_ia172_visual_set.py in the build containers)
"""scaleup_ia172_visual_set: merge my by-eye reading of one scale-up case's mip_neck.png into
/sp/v5_va/results/scaleup_ia172_visual.json (same fields as results/pilot_ia172_visual.json), then into the case's
meta.json / qc.json (scaleup_ia172_meta_visual.py).  usage: python3 scaleup_ia172_visual_set.py <case> '<json object>'"""
import json, os, subprocess, sys
P = "/sp/v5_va/results/scaleup_ia172_visual.json"
d = json.load(open(P)) if os.path.exists(P) else {
    "_how": "My reading of each scale-up case's mip_neck.png (coronal whole-volume MIP, coronal AP-slab and VA-corridor "
            "MIPs, left/right sagittal slab MIPs, axial 1 mm MIPs at C2/C4/C6/C7), 2026-10-02, scaleup_lane_ia172; same "
            "criteria as the pilot: 'VAs visible' = both VA lumina enhanced in the C4 and C6 transverse foramina on the "
            "axial panels, and the course seen on the corridor/sagittal MIPs where bone allows."}
d[sys.argv[1]] = json.loads(sys.argv[2])
json.dump(d, open(P, "w"), indent=1)
subprocess.call([sys.executable, "/sp/v5_va/scaleup_ia172_meta_visual.py", sys.argv[1]])
