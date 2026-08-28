import random, csv, os, math
from collections import defaultdict
V1BP=r"d:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
rows=list(csv.DictReader(open(os.path.join(V1BP,"episodes.csv"))))
by=defaultdict(list)
for r in rows: by[r["anatomy"]].append(int(r["success"]))
keys=list(by)
random.seed(0); B=20000
pooled=[];unw=[]
for _ in range(B):
    pick=[random.choice(keys) for _ in keys]
    s=[x for k in pick for x in by[k]]
    pooled.append(sum(s)/len(s))
    unw.append(sum(sum(by[k])/len(by[k]) for k in pick)/len(pick))
def ci(v):
    v=sorted(v); return v[int(.025*len(v))],v[int(.975*len(v))]
print("cluster bootstrap over the 4 anatomies (B=20000):")
print("  pooled rate      point 0.9184  95%% CI %.3f-%.3f"%ci(pooled))
print("  unweighted mean  point 0.9011  95%% CI %.3f-%.3f"%ci(unw))
# episode-level Wilson for reference
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)
print("  reported Wilson (assumes 98 independent draws): %.3f-%.3f"%wilson(90,98))
