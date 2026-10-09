# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\peek_final.py (= /sp/v6/s3fix/peek_final.py in the build containers)
import json, glob, os
for f in sorted(glob.glob('/s3/checks/final/case_*.json')):
    d = json.load(open(f)); n = os.path.basename(f)[5:-5]
    g = d.get('gates') or {}
    bad = [k for k, v in g.items() if not v]
    d22 = (d.get('route') or {}).get('d22') or {}
    ref = {p: ((d22.get(p) or {}).get('d22_reference'), round((d22.get(p) or {}).get('value') or 0, 4), round((d22.get(p) or {}).get('value_v4') or 0, 4), (d22.get(p) or {}).get('d26_flag')) for p in ('fork', 'siphon')}
    cv = d.get('coverage') or {}
    print('%-34s pass %s fails %s d22 %s cov %s errs %s' % (n, d.get('pass'), bad, ref, {p: (cv.get(p) or {}).get('deficit_max') for p in ('fork','siphon')}, [k for k in d if k.endswith('_error')]))
