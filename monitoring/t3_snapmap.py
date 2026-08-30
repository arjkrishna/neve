import pickle, os, re
from collections import defaultdict
D=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl","rb")); eps=D["eps"]
SNAP=r"D:/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_075919_rcca_topbrain_v1/diagnostics/snapshots/eval/RCCA"
rec={}
for out in ("success","max_steps"):
    for fn in os.listdir(os.path.join(SNAP,out)):
        m=re.match(r"ep(\d+)_pid(\d+)_step(\d+)_R([-+0-9.]+)_",fn)
        if not m: print("UNPARSED",fn); continue
        ep,pid,stp,R=int(m.group(1)),m.group(2),int(m.group(3)),float(m.group(4))
        rec[(pid,ep)]=(out,stp,R,fn)
print("snapshots:",len(rec))
# map log episodes
byblock=defaultdict(list)
matched=0
for k,e in eps.items():
    key=(e["pid"],int(e["ep"]))
    r=rec.get(key)
    if r: matched+=1; byblock[e["block"]].append((e,r))
    else: byblock[e["block"]].append((e,None))
print("matched:",matched,"of",len(eps))
for b in (1,2,3):
    L=byblock[b]; s=sum(1 for e,r in L if r and r[0]=="success"); m=sum(1 for e,r in L if r and r[0]=="max_steps"); u=sum(1 for e,r in L if r is None)
    print(f"eval{b}: n={len(L)} success={s} max_steps={m} unmatched={u}  ->success_rate={s/max(1,s+m):.4f}")
    # step agreement
    bad=[(e['w'],e['pid'],e['ep'],len(e['steps']),r[1]) for e,r in L if r and abs(len(e['steps'])-r[1])>2]
    print("   step-count mismatches(>2):",len(bad), bad[:5])
pickle.dump({k:v for k,v in rec.items()}, open(r"D:/Arjun/workspace/neve/monitoring/_t3_snap.pkl","wb"))
# pick extremes for eval2/eval3 and eval1
for b in (1,2):
    L=[(e,r) for e,r in byblock[b] if r and r[0]=="success"]
    L.sort(key=lambda x:x[1][1])
    print(f"\neval{b} shortest successes:", [(x[1][3], x[0]['mesh']) for x in L[:3]])
    print(f"eval{b} longest successes:", [(x[1][3], x[0]['mesh']) for x in L[-3:]])
