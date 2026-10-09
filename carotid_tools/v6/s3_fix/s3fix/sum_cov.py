# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\sum_cov.py (= /sp/v6/s3fix/sum_cov.py in the build containers)
import json, glob, sys, os
D = sys.argv[1] if len(sys.argv) > 1 else '/s3/cov_v6_0'
rows = []
errs = []
folds = []
for f in sorted(glob.glob(D + '/case_*.json')):
    d = json.load(open(f))
    n = os.path.basename(f)[5:-5]
    c = d.get('coverage')
    if not c:
        errs.append((n, [k for k in d if k.endswith('_error')])); continue
    for ref in ('v4', 'v5'):
        if 'error' in c[ref]: errs.append((n, ref, c[ref]['error'][:100]))
    for p in ('fork', 'siphon'):
        a = c['v4'].get(p) or {}; b = c['v5'].get(p) or {}
        rows.append((n, p, a.get('deficit'), b.get('deficit'), a.get('lost_mm2'), b.get('lost_mm2'), (a.get('lost') or {}).get('s_runs_v6'), c[p]['ok'], c[p]['flag']))
    fo = d.get('folds') or {}
    if (fo.get('obj') or {}).get('fold_pairs') or (fo.get('full') or {}).get('fold_pairs'):
        folds.append((n, fo['obj']['fold_pairs'], fo['full']['fold_pairs'], fo['obj']['pairs'][:2], fo['full']['pairs'][:2]))
print('anatomies', len(rows) // 2, 'errors', errs[:5])
mx = lambda r: max(x for x in (r[2], r[3]) if x is not None) if (r[2] is not None or r[3] is not None) else 0
for thr in (0.01, 0.005, 0.001, 0.0001):
    print('deficit >= %.4f: %d' % (thr, sum(1 for r in rows if mx(r) >= thr)))
for r in sorted(rows, key=lambda r: -mx(r)):
    if mx(r) < 0.001: break
    print('%-36s %-6s v4 %.4f v5 %.4f lost %s/%s runs %s ok %s flag %s' % (r[0], r[1], r[2] or 0, r[3] or 0, r[4], r[5], r[6], r[7], r[8]))
print('folds', folds)
