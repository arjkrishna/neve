import csv, json, os
B="D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/"
def load(csvp, jsonp):
    rows={}
    with open(B+csvp) as f:
        for r in csv.DictReader(f):
            rows[int(r['seed'])]={'anat':r['anatomy'],'pl':float(r['path_len_mm']),'sec':r['section'],
                                  'csv_succ':int(r['success']),'steps':int(r['steps']),'gh':r['geometry_hash']}
    with open(B+jsonp) as f:
        for line in f:
            d=json.loads(line); s=d['seed']
            if s in rows:
                rows[s]['jsucc']=1 if d['success'] else 0
                rows[s]['grader']=1 if d.get('grader_success') else 0
                rows[s]['jsteps']=d['steps']
            else: print("MISSING seed in csv",s)
    return rows
T=load("eval_anatomies_checkpoint2002292/episodes.csv","eval_anatomies_checkpoint2002292/episodes_official_20260828_053306.jsonl")
H=load("eval_anatomies_checkpoint0/episodes.csv","eval_anatomies_checkpoint0/episodes_official_20260828_062606.jsonl")
print("nT",len(T),"nH",len(H))
print("seed sets equal:",set(T)==set(H))
mism_anat=sum(1 for s in T if T[s]['anat']!=H[s]['anat'])
mism_pl=sum(1 for s in T if abs(T[s]['pl']-H[s]['pl'])>0.05)
print("anat mismatch",mism_anat,"pathlen mismatch",mism_pl)
dT=sum(1 for s in T if T[s]['csv_succ']!=T[s]['jsucc']); dH=sum(1 for s in H if H[s]['csv_succ']!=H[s]['jsucc'])
print("csv-vs-jsonl success disagreements: teacher",dT,"heur",dH)
print("teacher total succ jsonl",sum(T[s]['jsucc'] for s in T),"heur",sum(H[s]['jsucc'] for s in H))
GR=166.91
gr=[s for s in T if T[s]['pl']>GR]
print("grafted n",len(gr),"teacher succ",sum(T[s]['jsucc'] for s in gr),"heur succ",sum(H[s]['jsucc'] for s in gr))
import collections
per=collections.defaultdict(lambda:[0,0,0])
for s in gr:
    a=T[s]['anat']; per[a][0]+=1; per[a][1]+=T[s]['jsucc']; per[a][2]+=H[s]['jsucc']
for a in sorted(per): print(a,per[a])
out=[]
for s in sorted(T):
    out.append({'seed':s,'anat':T[s]['anat'],'pl':T[s]['pl'],'sec':T[s]['sec'],
                'T':T[s]['jsucc'],'H':H[s]['jsucc'],'Tsteps':T[s]['jsteps'],'Hsteps':H[s]['jsteps'],
                'gh':T[s]['gh']})
json.dump(out,open("D:/Arjun/workspace/neve/monitoring/arj_t3_merged.json","w"))
print("wrote merged")
