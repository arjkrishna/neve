# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\sum_frag.py (= /sp/v6/s3fix/sum_frag.py in the build containers)
import json, glob, os
for f in sorted(glob.glob('/s3/diag/frag_case_*.json')):
    d = json.load(open(f))
    print("==", d['name'][5:], "sup", d['suppressed'], "comps", d['comps'])
    for r in d['dropped']:
        if r['rcca_tube_dist_centroid'] > 6 or r['volume_mm3'] < 0.3: continue
        print("  ", r['cells'], r['volume_mm3'], 'rccaT', r['rcca_tube_dist_centroid'], r['nearest_other'], r['other_tube_dist_centroid'], 's', r['rcca_s_range'], r['rcca_tube_dist_pts'], 'onL', r.get('on_lower_R_surface_frac'), 'onS', r.get('on_siphon_R_surface_frac'), [ (g['section'], g.get('why')) for g in r.get('gap',[])])
