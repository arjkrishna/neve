# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\peek_span.py (= /sp/v6/s3fix/peek_span.py in the build containers)
import json
for n in [l.strip() for l in open('/sp/v6/s3fix/names8.txt')] + ['case_w_012_left__topcow_mr_001','case_w_012_left__topcow_mr_013_L','case_w_017_right__topcow_mr_011','case_w_035_left__topcow_mr_015_L','case_w_025_right__topcow_mr_021_L']:
    r6 = json.load(open('/repo/carotid_data/anatomies_v6/%s/mesh_v6.json' % n))
    r5 = json.load(open('/repo/carotid_data/anatomies_v5/%s/mesh_v5.json' % n))
    p6 = json.load(open('/repo/carotid_data/anatomies_v6/%s/provenance.json' % n))
    p5 = json.load(open('/repo/carotid_data/anatomies_v5/%s/provenance.json' % n))
    def sp(r, k):
        v = r['spans'].get(k); return [round(v[1],2), round(v[2],2), round(v[2]-v[1],2)] if isinstance(v, list) else v
    x6, x5 = p6['xform']['lower'], p5['xform']['lower']
    print(n[5:], 'L6', sp(r6,'lower_R'), 'L5', sp(r5,'lower_R'), 'S6', sp(r6,'siphon_R'), 'S5', sp(r5,'siphon_R'),
          'cut/cca/ica6', p6['host_cut_mm'], round(p6['cca_mm'],2), round(p6['ica_mm'],2), x6.get('ica_real_mm'), 'blend6', x6.get('blend_mm'),
          '| cut/cca/ica5', p5['host_cut_mm'], round(p5['cca_mm'],2), round(p5['ica_mm'],2), x5.get('ica_real_mm'), 'blend5', x5.get('blend_mm'), 'mir', x6.get('mirror'))
