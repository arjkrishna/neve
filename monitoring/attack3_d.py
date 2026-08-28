import os,csv,json,pickle
from collections import Counter,defaultdict
V1BP = r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
SP=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad\a3.pkl"
D=pickle.load(open(SP,"rb")); vb=D["vb"]
pm=defaultdict(Counter)
for e in vb: pm[e["pid"]][e["mesh"]]+=1
for p in sorted(pm,key=int): print("pid",p,dict(pm[p]))
print()
# seed -> mesh
rows=list(csv.DictReader(open(os.path.join(V1BP,"episodes.csv"))))
sm={int(r["seed"]):r for r in rows}
print("seed range:",min(sm),max(sm),"count",len(sm))
missing=[s for s in range(min(sm),max(sm)+1) if s not in sm]
print("gaps in seed range:",missing)
for m in sorted(set(r["anatomy"] for r in rows)):
    ss=sorted(int(r["seed"]) for r in rows if r["anatomy"]==m)
    print(m,"seeds:",ss)
