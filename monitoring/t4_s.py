import pickle, numpy as np, collections, math, random
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
by=collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
# DerSimonian-Laird on log-odds (with 0.5 continuity)
ys=[];ws=[]
for a in sorted(by):
    g=by[a]; k=sum(r["succ"] for r in g); n=len(g)
    kk,nn=k+0.5,n-k+0.5
    y=math.log(kk/nn); v=1/kk+1/nn
    ys.append(y); ws.append(1/v)
ys=np.array(ys); ws=np.array(ws)
Q=float(np.sum(ws*(ys-np.sum(ws*ys)/np.sum(ws))**2)); df=len(ys)-1
C=np.sum(ws)-np.sum(ws**2)/np.sum(ws)
tau2=max(0.0,(Q-df)/C)
print("random-effects (DL) on per-anatomy log-odds: Q=%.2f df=%d tau2=%.4f (tau=%.3f logit) I2=%.1f%%"%(
  Q,df,tau2,tau2**.5,100*max(0,(Q-df)/Q)))
print("  -> anatomy SD on the logit scale %.2f; median within-anat sampling SD %.2f"%(tau2**.5,np.median(1/np.sqrt(ws))))
print("  -> reliability of a single anatomy's observed rate = tau2/(tau2+v) med %.2f"%np.median(tau2/(tau2+1/ws)))
# implied rate spread from tau alone
p0=69/124
for z in (-1,1):
    print("   +/-1 anatomy SD => rate %.3f"%(1/(1+math.exp(-(math.log(p0/(1-p0))+z*tau2**.5)))))
print()
print("=== observed vs binomial-expected spread ===")
obs=np.array([np.mean([r["succ"] for r in by[a]]) for a in sorted(by)])
nn=np.array([len(by[a]) for a in sorted(by)])
print(" observed sd of 22 rates %.3f ; expected sd under pooled binomial %.3f"%(
  obs.std(ddof=1), math.sqrt(np.mean(p0*(1-p0)/nn))))
# simulate
rng=random.Random(1); sds=[]
for _ in range(20000):
    sim=np.array([sum(1 for _ in range(n) if rng.random()<p0)/n for n in nn])
    sds.append(sim.std(ddof=1))
sds=np.array(sds)
print(" simulated null sd: med %.3f p95 %.3f ; p(sim>=obs)=%.4f"%(np.median(sds),np.percentile(sds,95),float((sds>=obs.std(ddof=1)).mean())))
print()
print("=== discordance between controllers on identical (anatomy,target,seed) ===")
disc=sum(1 for r in gt if r["succ"]!=mh[r["seed"]]["succ"])
print(" %d/%d = %.1f%% of grafted episodes flip outcome between teacher and heuristic"%(disc,len(gt),100*disc/len(gt)))
