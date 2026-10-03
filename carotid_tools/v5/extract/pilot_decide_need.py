# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\pilot_decide_need.py (= /sp/v5_va/pilot_decide_need.py in the build containers)
"""pilot_decide step 1: exact donor need from anatomies_v4 provenance (read-only)."""
import json,glob,os,collections
root='/repo/carotid_data/anatomies_v4'
b=json.load(open(root+'/BUILD_v4.json'))
for k in ['what_v4_adds','handedness']:
    print(k,':',b[k])
print('graft:',json.dumps(b['graft'])[:1500])
print('vertebral keys in BUILD:',[k for k in b if 'vert' in k.lower() or 'va' in k.lower()])
rows=[]
for p in sorted(glob.glob(root+'/*/provenance.json')):
    a=os.path.basename(os.path.dirname(p)); d=json.load(open(p))
    v=d.get('vertebrals',{})
    rows.append(dict(anat=a,patient=d.get('patient'),siphon=d.get('siphon'),hand=d.get('handedness'),
        R=v.get('right',{}).get('patient_side'),L=v.get('left',{}).get('patient_side'),
        Rlen=v.get('right',{}).get('donor_len_mm'),Llen=v.get('left',{}).get('donor_len_mm'),
        Rtube=v.get('right',{}).get('tube'), keysR=sorted(v.get('right',{}).keys())))
print(len(rows),'anatomies'); print('example keys',rows[0]['keysR'])
pat=collections.Counter(r['patient'] for r in rows)
print('distinct patients',len(pat)); print(sorted(pat.items()))
hc=collections.Counter((r['hand'],r['R'],r['L']) for r in rows); print(hc)
# per patient native/mirrored
pp=collections.defaultdict(lambda: collections.Counter())
for r in rows: pp[r['patient']][r['hand']]+=1
for k in sorted(pp): print(k, dict(pp[k]))
# missing VA?
miss=[r['anat'] for r in rows if r['R'] is None or r['L'] is None]
print('anatomies missing a vertebral entry',miss)
print('siphon vs patient mismatch',[r['anat'] for r in rows if r['siphon'].replace('_L','')!=r['patient']])
json.dump(dict(rows=rows),open('/sp/v5_va/results/pilot_decide_need_rows.json','w'),indent=0)
