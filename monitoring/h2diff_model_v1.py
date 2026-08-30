import json, math, collections
import numpy as np
rng=np.random.default_rng(11)
M="monitoring/"
T=json.load(open(M+"h2diff_table.json")); ROWS=json.load(open(M+"h2diff_eprows.json"))
name_map={a:"topcowmr%03d"%int(a.split("_")[-1]) for a in T if a!="HOST"}
inv={v:k for k,v in name_map.items()}
def gammp(a,x):
    if x<=0: return 0.0
    if x<a+1:
        ap=a; s=1.0/a; d=s
        for _ in range(500):
            ap+=1; d*=x/ap; s+=d
            if abs(d)<abs(s)*1e-14: break
        return s*math.exp(-x+a*math.log(x)-math.lgamma(a))
    b=x+1-a; c=1e300; d=1.0/b; h=d
    for i in range(1,500):
        an=-i*(i-a); b+=2
        d=an*d+b; d=1e-300 if abs(d)<1e-300 else d
        c=b+an/c; c=1e-300 if abs(c)<1e-300 else c
        d=1.0/d; de=d*c; h*=de
        if abs(de-1)<1e-14: break
    return 1.0-math.exp(-x+a*math.log(x)-math.lgamma(a))*h
def chi2sf(x,df): return 1.0-gammp(df/2.0,x/2.0) if x>0 else 1.0
def normsf(z): return 0.5*math.erfc(abs(z)/math.sqrt(2))*2
def logit(X,y,ridge=1e-6,it=200):
    b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p)+1e-9
        H=X.T@(X*W[:,None])+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        d=np.linalg.solve(H,g); b+=d
        if np.max(np.abs(d))<1e-10: break
    p=np.clip(1/(1+np.exp(-X@b)),1e-12,1-1e-12)
    ll=float((y*np.log(p)+(1-y)*np.log(1-p)).sum())
    return b,ll,p
def clusterse(X,y,b,groups):
    p=1/(1+np.exp(-X@b)); W=p*(1-p)+1e-9
    A=np.linalg.pinv(X.T@(X*W[:,None]))
    Bm=np.zeros((X.shape[1],X.shape[1]))
    for g in set(groups):
        m=np.array([gg==g for gg in groups])
        u=(X[m]*((y[m]-p[m])[:,None])).sum(0)
        Bm+=np.outer(u,u)
    V=A@Bm@A
    return np.sqrt(np.diag(V))
anat=[r["anat"] for r in ROWS]
y=np.array([r["succ"] for r in ROWS],float); yh=np.array([r["hsucc"] for r in ROWS],float)
def z(v): v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-12)
# ---------- composite difficulty (anatomy level, pre-specified weights) ----------
A=sorted(set(anat)); G={a:T[inv[a]] for a in A}
def zz(k,sign=1): 
    v=np.array([G[a][k] for a in A],float); return sign*(v-v.mean())/(v.std()+1e-12)
curv=(zz("Rc_p25",-1)+zz("n_Rc_lt8",1)+zz("bend_max",1))/3
turn=(zz("turn_cum",1)+zz("turn_per_mm",1)+zz("tort_graft",1))/3
cal =(zz("clr_p05",-1)+zz("clr_p25",-1)+zz("clr_n_lt_cath",1))/3
ext = zz("graft_len",1)
DIFF=0.40*curv+0.30*turn+0.20*cal+0.10*ext
Dmap=dict(zip(A,DIFF))
pc_in=np.stack([zz(k,s) for k,s in [("Rc_p25",-1),("n_Rc_lt8",1),("bend_max",1),("turn_cum",1),
        ("turn_per_mm",1),("tort_graft",1),("clr_p05",-1),("clr_p25",-1),("graft_len",1),("n_infl",1),("tors_cum",1)]],1)
U,S,Vt=np.linalg.svd(pc_in-pc_in.mean(0),full_matrices=False)
PC1=U[:,0]*S[0]; PC1=PC1*np.sign(np.corrcoef(PC1,DIFF)[0,1]); PCmap=dict(zip(A,(PC1-PC1.mean())/PC1.std()))
print("=== composite difficulty (higher = harder), 22 anatomies ===")
by=collections.defaultdict(lambda:[0,0,0])
for r in ROWS: b=by[r["anat"]]; b[0]+=1; b[1]+=r["succ"]; b[2]+=r["hsucc"]
print("%-12s %6s %6s %6s %6s %6s %6s %6s %5s %5s"%("anat","DIFF","PC1","curv","turn","cal","ext","n","pol","heu"))
for a in sorted(A,key=lambda a:-Dmap[a]):
    i=A.index(a); n,kp,kh=by[a]
    print("%-12s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6d %5.2f %5.2f"%(a,Dmap[a],PCmap[a],curv[i],turn[i],cal[i],ext[i],n,kp/n,kh/n))
json.dump({"DIFF":{a:float(Dmap[a]) for a in A},"PC1":{a:float(PCmap[a]) for a in A},
           "curv":dict(zip(A,map(float,curv))),"turn":dict(zip(A,map(float,turn))),
           "cal":dict(zip(A,map(float,cal))),"ext":dict(zip(A,map(float,ext)))},open(M+"h2diff_diff.json","w"),indent=1)
# ---------- episode level models ----------
d_anat=np.array([Dmap[a] for a in anat]); pc_anat=np.array([PCmap[a] for a in anat])
S_T=z([r["s_tgt"] for r in ROWS]); TURN=z([r["turn_cum"] for r in ROWS]); DG=z([r["dgraft"] for r in ROWS])
RCM=z([r["Rc_min"] for r in ROWS]); CLR=z([r["clr_min"] for r in ROWS]); TORT=z([r["tort"] for r in ROWS])
one=np.ones(len(y))
def fit(names,cols,yv,tag,groups=anat):
    X=np.stack([one]+cols,1); b,ll,p=logit(X,yv); se=clusterse(X,yv,b,groups)
    print("  %-34s ll=%9.3f  "%(tag,ll)+" ".join("%s=%+.3f(%.3f,z=%+.2f,p=%.3f)"%(n,b[i+1],se[i+1],b[i+1]/se[i+1],normsf(b[i+1]/se[i+1])) for i,n in enumerate(names)))
    return ll,b,se
print("\n=== EPISODE-LEVEL LOGISTIC (n=124 grafted), cluster-robust SE by anatomy ===")
for yv,lab in [(y,"POLICY"),(yh,"HEURISTIC")]:
    print(" -- %s --"%lab)
    ll0=logit(np.stack([one],1),yv)[1]; print("  %-34s ll=%9.3f"%("null",ll0))
    fit(["s_tgt"],[S_T],yv,"depth only")
    fit(["turn_to_tgt"],[TURN],yv,"route turning only")
    fit(["DIFFcomposite"],[d_anat],yv,"anatomy composite difficulty")
    fit(["PC1"],[pc_anat],yv,"anatomy PC1")
    fit(["s_tgt","DIFF"],[S_T,d_anat],yv,"depth + composite")
    fit(["s_tgt","turn"],[S_T,TURN],yv,"depth + route turning")
    # anatomy factor LRT vs depth-only
    A_=sorted(set(anat)); D=np.stack([[1.0 if a==aa else 0.0 for aa in A_[1:]] for a in anat])
    llD=logit(np.stack([one,S_T],1),yv)[1]
    llF=logit(np.concatenate([np.stack([one,S_T],1),D],1),yv)[1]
    lr=2*(llF-llD); df=D.shape[1]
    print("  anatomy-factor LRT | depth : G2=%.2f df=%d p=%.4f"%(lr,df,chi2sf(lr,df)))
# ---------- paired: policy minus heuristic ----------
print("\n=== POLICY vs MATCHED HEURISTIC on the same 124 grafted episodes ===")
b=int(((y==1)&(yh==0)).sum()); c=int(((y==0)&(yh==1)).sum())
mcp=0.0
for k in range(0,b+c+1):
    pr=math.comb(b+c,k)*0.5**(b+c)
    if abs(k-(b+c)/2)>=abs(b-(b+c)/2)-1e-9: mcp+=pr
print("  policy %d/124, heuristic %d/124 ; discordant b(pol only)=%d c(heur only)=%d  McNemar exact p=%.4f"%(y.sum(),yh.sum(),b,c,mcp))
dif=y-yh
for nm,v in [("DIFF",d_anat),("PC1",pc_anat),("s_tgt",S_T),("turn",TURN),("Rc_min",RCM),("clr_min",CLR)]:
    # cluster-permutation: flip policy/heuristic label within anatomy clusters
    r=np.corrcoef(v,dif)[0,1]
    B=20000; cnt=0
    gl=list(set(anat)); idxg={g:np.array([a==g for a in anat]) for g in gl}
    for _ in range(B):
        d2=dif.copy()
        for g in gl:
            if rng.random()<0.5: d2[idxg[g]]*=-1
        if abs(np.corrcoef(v,d2)[0,1])>=abs(r)-1e-12: cnt+=1
    print("  corr(%s, policy-heuristic per-episode diff) = %+.3f   cluster-sign-flip p=%.4f"%(nm,r,(cnt+1)/(B+1)))
