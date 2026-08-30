"""TASK3 episode-level modelling. Reads merged episodes + geometry profiles. Read-only."""
import json, math, collections
import numpy as np
from scipy import stats, optimize

EPS=json.load(open("/opt/mon/arj_t3_merged.json"))
G=json.load(open("/opt/mon/arj_t3_geom.json"))
OFF=33.31; SEAM=133.6; GRAFT_PL=166.91
for a in G:
    for k in ("g","d","r","k1","k3","cum"): G[a][k]=np.asarray(G[a][k],float)

def at(a,s,key):
    p=G[a]; return float(np.interp(s,p["g"],p[key]))
def win(a,s0,s1,key,fn):
    p=G[a]; m=(p["g"]>=s0)&(p["g"]<=s1)
    if m.sum()<2:
        m=(p["g"]>=min(s0,s1)-0.5)&(p["g"]<=max(s0,s1)+0.5)
    return float(fn(p[key][m]))

rows=[]
for e in EPS:
    if e["pl"]<=GRAFT_PL: continue
    a=e["anat"]; s=e["pl"]-OFF; L=G[a]["L"]
    s=min(s,L)
    r=dict(seed=e["seed"],anat=a,T=e["T"],H=e["H"],pl=e["pl"],s=s,L=L,
           dg=s-SEAM, frac=s/L, s_end=L-s,
           r_t=at(a,s,"r"), d_t=at(a,s,"d"), k1_t=at(a,s,"k1"), k3_t=at(a,s,"k3"),
           cum_t=at(a,s,"cum")-at(a,SEAM,"cum"),
           d_w=win(a,s-5,s+5,"d",np.min), k3_w=win(a,s-5,s+5,"k3",np.max),
           r_w=win(a,s-5,s+5,"r",np.min),
           d_sp=win(a,SEAM,s,"d",np.min), k3_sp=win(a,SEAM,s,"k3",np.max),
           k1_sp=win(a,SEAM,s,"k1",np.max), r_sp=win(a,SEAM,s,"r",np.min),
           Tsteps=e["Tsteps"], Hsteps=e["Hsteps"])
    r["Rc_t"]=1.0/max(r["k3_t"],1e-6); r["Rc_sp"]=1.0/max(r["k3_sp"],1e-6)
    rows.append(r)
print("n grafted episodes:",len(rows))
ANAT=sorted({r["anat"] for r in rows})
print("n anatomies:",len(ANAT))

# ---------- 1. HETEROGENEITY ----------
def het(rows,key,label,drop=()):
    rr=[r for r in rows if r["anat"] not in drop]
    by=collections.defaultdict(lambda:[0,0])
    for r in rr: by[r["anat"]][0]+=1; by[r["anat"]][1]+=r[key]
    ks=sorted(by); n=sum(by[a][0] for a in ks); y=sum(by[a][1] for a in ks); p=y/n
    G2=0.0; X2=0.0
    for a in ks:
        ni,yi=by[a]; ei=ni*p
        for obs,exp in ((yi,ei),(ni-yi,ni-ei)):
            if obs>0: G2+=2*obs*math.log(obs/exp)
            X2+=(obs-exp)**2/exp
    df=len(ks)-1
    pG=stats.chi2.sf(G2,df); pX=stats.chi2.sf(X2,df)
    # parametric bootstrap null for G2
    rng=np.random.default_rng(7); ns=np.array([by[a][0] for a in ks]); B=200000
    sim=rng.binomial(ns[None,:],p,size=(B,len(ks))).astype(float)
    exp=ns[None,:]*p
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=np.where(sim>0,2*sim*np.log(np.where(sim>0,sim/exp,1)),0.0)
        f=ns[None,:]-sim; ef=ns[None,:]-exp
        t2=np.where(f>0,2*f*np.log(np.where(f>0,f/ef,1)),0.0)
    Gs=(t1+t2).sum(1); pboot=float(((Gs>=G2).sum()+1)/(B+1))
    print("[HET] %-28s k=%2d n=%3d y=%3d p=%.4f  G2=%6.3f X2=%6.3f df=%d  p_G=%.4f p_X=%.4f p_boot=%.5f  disp(X2/df)=%.3f"%(
        label,len(ks),n,y,p,G2,X2,df,pG,pX,pboot,X2/df))
    return by,G2,X2,df,pboot

byT,_,_,_,_=het(rows,"T","TEACHER all 22")
byH,_,_,_,_=het(rows,"H","HEURISTIC all 22")
het(rows,"T","TEACHER drop mr_025","topcowmr025")
het(rows,"H","HEURISTIC drop mr_025","topcowmr025")
het(rows,"T","TEACHER drop 025+024",("topcowmr025","topcowmr024"))
het(rows,"H","HEURISTIC drop 025+024",("topcowmr025","topcowmr024"))

# Wilson CIs
print("\n[WILSON] anatomy  n  T_y  T_rate  T_CI95        H_y  H_rate  H_CI95        depth s med/min/max")
z=1.959964
def wil(y,n):
    if n==0: return (float('nan'),)*2
    ph=y/n; den=1+z*z/n; c=(ph+z*z/(2*n))/den
    h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return max(0,c-h),min(1,c+h)
for a in ANAT:
    rr=[r for r in rows if r["anat"]==a]; n=len(rr)
    ty=sum(r["T"] for r in rr); hy=sum(r["H"] for r in rr)
    ss=sorted(r["s"] for r in rr)
    lo,hi=wil(ty,n); lo2,hi2=wil(hy,n)
    print("%-12s %2d  %2d  %.3f  [%.3f,%.3f]   %2d  %.3f  [%.3f,%.3f]   %.1f/%.1f/%.1f  L=%.1f"%(
        a,n,ty,ty/n,lo,hi,hy,hy/n,lo2,hi2,np.median(ss),ss[0],ss[-1],rr[0]["L"]))

# ---------- 2. DEPTH ENTANGLEMENT ----------
print("\n[DEPTH] per-anatomy depth-into-graft (s-133.6) distribution")
dgm={}
for a in ANAT:
    dd=sorted(r["dg"] for r in rows if r["anat"]==a); dgm[a]=float(np.mean(dd))
    print("%-12s n=%2d  dg mean=%6.1f min=%6.1f max=%6.1f  frac mean=%.3f"%(
        a,len(dd),np.mean(dd),dd[0],dd[-1],np.mean([r["frac"] for r in rows if r["anat"]==a])))
dgall=np.array([r["dg"] for r in rows]); anid=np.array([ANAT.index(r["anat"]) for r in rows])
gm=np.array([dgm[r["anat"]] for r in rows])
ssb=((gm-dgall.mean())**2).sum(); sst=((dgall-dgall.mean())**2).sum()
print("depth between-anatomy R2 (eta^2) = %.4f  (one-way ANOVA on dg)"%(ssb/sst))
Fs=(ssb/(len(ANAT)-1))/(((dgall-gm)**2).sum()/(len(dgall)-len(ANAT)))
print("  F=%.3f df=(%d,%d) p=%.4g"%(Fs,len(ANAT)-1,len(dgall)-len(ANAT),stats.f.sf(Fs,len(ANAT)-1,len(dgall)-len(ANAT))))
json.dump(rows,open("/opt/mon/arj_t3_rows.json","w"))
print("\nWROTE rows")
