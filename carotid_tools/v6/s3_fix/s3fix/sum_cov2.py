# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\sum_cov2.py (= /sp/v6/s3fix/sum_cov2.py in the build containers)
import json, glob, sys, os
for D in sys.argv[1:]:
    for f in sorted(glob.glob(D + '/case_*.json')):
        d = json.load(open(f)); c = d.get('coverage') or {}
        out = []
        for p in ('fork', 'siphon'):
            out.append('%s v4 %.5f v5 %.5f lost %s/%s' % (p, (c['v4'].get(p) or {}).get('deficit') or 0, (c['v5'].get(p) or {}).get('deficit') or 0, (c['v4'].get(p) or {}).get('lost_mm2'), (c['v5'].get(p) or {}).get('lost_mm2')))
        print(os.path.basename(D), os.path.basename(f)[5:-5], ' | '.join(out), 'folds', {k: v['fold_pairs'] for k, v in (d.get('folds') or {}).items()}, 'pass', d.get('pass'))
