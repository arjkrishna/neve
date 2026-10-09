# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\sum_shift.py (= /sp/v6/s3fix/sum_shift.py in the build containers)
import json, glob
for f in sorted(glob.glob('/s3/shift/case_*.json')):
    d = json.load(open(f))
    print("==", d['name'][5:])
    for r in d['shifts']:
        fr = [(x['cells'], x['volume_mm3'], x['s'], x.get('lower_R'), x.get('siphon_R')) for x in r['route_fragments'] if x['volume_mm3'] >= 0.3]
        print("  ", r['shift'], "g", r['genus'], "comps", r['comps'], "vol", r['route_fragment_volume_mm3'], fr)
