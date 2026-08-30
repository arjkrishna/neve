import pickle, numpy as np, collections, math
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
GEO=pickle.load(open("_t4_geom.pkl","rb"))
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
def logit_fit(X,y,iters=200):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-X@b)); W=np.clip(p*(1-p),1e-9,None); z=X@b+(y-p)/W
        b=np.linalg.solve(X.T@(X*W[:,None])+1e-6*np.eye(X.shape[1]), X.T@(W*z))
    return b
u=lambda r:(r["tgt_s"]-183.3)/10.0
y=[r["succ"] for r in gt]
b=logit_fit([[1.0,u(r),u(r)**2] for r in gt],y)
pred={}
for r in gt: pred[r["seed"]]=1/(1+math.exp(-(b[0]+b[1]*u(r)+b[2]*u(r)**2)))
by=collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
print("anat      n  obs%   depth-predicted%   resid(obs-pred, eps)   Hobs%")
res=[]
for a in sorted(by):
    g=by[a]; o=np.mean([r["succ"] for r in g]); p=np.mean([pred[r["seed"]] for r in g])
    res.append((a,len(g),o,p))
    print(f"{a} {len(g):3d} {100*o:5.1f}      {100*p:5.1f}        {len(g)*(o-p):+6.2f}          {100*np.mean([mh[r['seed']]['succ'] for r in g]):5.1f}")
oo=np.array([t[2] for t in res]); pp=np.array([t[3] for t in res]); nn=np.array([t[1] for t in res])
print("\nvar of observed per-anat rate %.4f ; var of depth-predicted %.4f ; ratio %.2f"%(oo.var(),pp.var(),pp.var()/oo.var()))
def spear(x,yy):
    def rk(v):
        o2=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v); i=0
        while i<len(o2):
            j=i
            while j+1<len(o2) and v[o2[j+1]]==v[o2[i]]: j+=1
            for m in range(i,j+1): r[o2[m]]=(i+j)/2+1
            i=j+1
        return np.array(r,float)
    a,bb=rk(x),rk(yy)
    return float(np.corrcoef(a,bb)[0,1])
print("Spearman(obs rate, depth-pred rate) over 22 anat = %+.3f"%spear(list(oo),list(pp)))
print("Spearman(teacher rate, heuristic rate) over 22 anat = %+.3f"%spear(list(oo),
      [np.mean([mh[r['seed']]['succ'] for r in by[t[0]]]) for t in res]))
print()
print("=== fold onset geometry (fix nan) ===")
def key(a): return "topcow_"+a.replace("mr_mr","mr_")
on=[(r,r["s_at_fold20"]) for r in gt if r["s_at_fold20"] is not None]
v=[]
for r,s in on:
    g=GEO[key(r["anat"])]
    v.append(float(np.interp(s,g["q"],np.nan_to_num(g["Rc5"],nan=np.nanmedian(g["Rc5"])))))
print(" n=%d ; Rc5 at fold onset med %.1f ; anatomy median Rc5(20-200) ~20 ; onset s<=130 in %d/%d"%(
  len(on),np.nanmedian(v),sum(1 for _,s in on if s<=130),len(on)))
print(" onset stations:",sorted(round(s,1) for _,s in on))
print()
print("=== mr_025 removed ===")
g2=[r for r in gt if r["anat"]!="mr_mr025"]
print(" teacher %d/%d = %.1f%% ; heuristic %d/%d = %.1f%%"%(
  sum(r['succ'] for r in g2),len(g2),100*np.mean([r['succ'] for r in g2]),
  sum(mh[r['seed']]['succ'] for r in g2),len(g2),100*np.mean([mh[r['seed']]['succ'] for r in g2])))
print()
print("=== heuristic's own failure anatomy ===")
hf=[mh[r["seed"]] for r in gt if not mh[r["seed"]]["succ"]]
print(" n=%d ; %d arrest at s<10 (never cannulated); their tgt_s med %.1f"%(
  len(hf),sum(1 for r in hf if r['max_s']<10),np.median([r['tgt_s'] for r in hf if r['max_s']<10])))
print(" heuristic fold_max med %d ; slack_max med %.1f ; pull_frac med %.2f"%(
  np.median([r['fold_max'] for r in hf]),np.median([r['slack_max'] for r in hf]),np.median([r['pull_frac'] for r in hf])))
