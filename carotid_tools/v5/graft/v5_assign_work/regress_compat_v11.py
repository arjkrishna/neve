# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_assign_work\regress_compat_v11.py (= /sp/v5_va/v5_assign_work/regress_compat_v11.py in the build containers)
"""graft_reg: compare graft_v5_reg --compat (VMR unit, native hand) with the prototype's final dry run
graft/dry_all_takeoffrec.json, leaf by leaf on the shared per-side keys."""
import json, os, sys
import numpy as np
proto = json.load(open('/sp/v5_va/graft/dry_all_takeoffrec.json'))['rows']
D = '/repo/carotid_data/va_sources/v5_reg/dry_v5_compat/compat_vmr0248'
keys = ('pose_deg', 'pose_axis', 'splice_move_mm', 'tb_start_trim_mm', 'splice_on_donor_mm', 'cut_at_mm', 'ostium_at_mm',
        'stub_mm', 'join_kink_deg', 'takeoff_10mm_deg', 'takeoff_5mm_deg', 'donor_takeoff_10mm_deg_native',
        'course_rms_to_corridor_mm', 'tip_moved_vs_v4_mm', 'tip_to_donor_confluence_mm', 'total_mm',
        'parent_tangent_mismatch_deg', 'va_10mm_elevation_deg')
ndiff, nleaf, nmiss, worst = 0, 0, 0, []
for n, a in proto.items():
    f = os.path.join(D, n + '.json')
    if not os.path.isfile(f):
        nmiss += 1; continue
    b = json.load(open(f))
    if 'error' in b:
        print('ERR', n, b['error']); ndiff += 1; continue
    for side in ('right', 'left'):
        x, y = a['sides'][side], b['sides'][side]
        for k in keys:
            nleaf += 1
            u, v = x.get(k), y.get(k)
            if isinstance(u, float) or isinstance(v, float):
                if u is None or v is None or abs(u - v) > 1e-6:
                    ndiff += 1; worst.append((n, side, k, u, v))
            elif u != v:
                ndiff += 1; worst.append((n, side, k, u, v))
        for k in ('clear', 'exit_mm', 'who'):
            nleaf += 1
            u, v = x['final'].get(k), y['final'].get(k)
            if (isinstance(u, float) and abs(u - v) > 1e-6) or (not isinstance(u, float) and u != v):
                ndiff += 1; worst.append((n, side, 'final.' + k, u, v))
print('compared %d anatomies (%d missing), %d leaves, %d differences' % (len(proto) - nmiss, nmiss, nleaf, ndiff))
for w in worst[:20]:
    print('  ', w)
json.dump({'n_anatomies': len(proto) - nmiss, 'missing': nmiss, 'leaves': nleaf, 'differences': ndiff, 'first': worst[:50]},
          open('/sp/v5_va/v5_assign_work/regress_compat_v11.json', 'w'), indent=1, default=str)
