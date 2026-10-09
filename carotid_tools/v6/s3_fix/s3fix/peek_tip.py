# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s3fix\peek_tip.py (= /sp/v6/s3fix/peek_tip.py in the build containers)
import json
for n in ['case_k_011_left__topcow_mr_012_L','case_w_002_right__topcow_mr_013_L','case_w_012_left__topcow_mr_003_L','case_w_008_right__topcow_mr_003','case_w_052_left__topcow_mr_026','case_w_027_left__topcow_mr_022','case_w_051_right__topcow_mr_008_L','case_w_012_left__topcow_mr_001','case_w_012_left__topcow_mr_013_L','case_w_017_right__topcow_mr_011']:
    p6 = json.load(open('/repo/carotid_data/anatomies_v6/%s/provenance.json' % n))
    p5 = json.load(open('/repo/carotid_data/anatomies_v5/%s/provenance.json' % n))
    vc = p6.get('v6_carotid') or {}
    e = vc.get('eca_R') or vc.get('eca_right') or {}
    keys = list(vc.keys())
    x6 = p6['xform'].get('ecaext_R'); x5 = p5['xform'].get('ecaext_R')
    print(n[5:], 'tip6', None if not x6 else (x6.get('join_at_mm'), x6.get('surface','')[-30:]), 'tip5', None if not x5 else (x5.get('join_at_mm'),), 'd22base', p6['v6_checks'].get('d22_baseline'))
    for k in keys:
        if 'eca' in k.lower():
            print('   ', k, json.dumps(vc[k])[:600])
