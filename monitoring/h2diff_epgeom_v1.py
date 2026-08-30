import json, math, collections
import numpy as np
M="monitoring/"
PROF=json.load(open(M+"h2diff_prof.json"))
EPS=json.load(open(M+"h2diff_alleps.json"))
OFF=33.31; SEAM=133.6
name_map={"topcowmr%03d"%int(k.split("_")[-1]):k for k in PROF if k!="HOST"}
def menger(p):
    a,b,c=p[:-2],p[1:-1],p[2:]
    ab=np.linalg.norm(b-a,axis=1); bc=np.linalg.norm(c-b,axis=1); ca=np.linalg.norm(a-c,axis=1)
    ar=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1); den=ab*bc*ca
    return np.where(den>1e-12,4*ar/np.maximum(den,1e-12),0.0)
def tang(p):
    d=np.diff(p,axis=0); n=np.linalg.norm(d,axis=1,keepdims=True); return d/np.maximum(n,1e-12)
def win_geom(key,s_end):
    P=PROF[key]; s=np.array(P["s"]); xyz=np.array(P["xyz"]); clr=np.array(P["clr"]); rd=np.array(P["rdec"])
    m=s<=s_end+1e-6
    if m.sum()<3:
        m=s<=max(s_end,float(s[2]))+1e-6
    s_=s[m]; x=xyz[m]; c=clr[m]; r=rd[m]
    d=s_[-1]-s_[0]
    o={"dgraft":d,"tort":d/max(np.linalg.norm(x[-1]-x[0]),1e-9),
       "clr_min":float(c.min()),"clr_p05":float(np.percentile(c,5)),"clr_med":float(np.median(c)),
       "r_min":float(r.min()),"r_med":float(np.median(r))}
    d=max(d,1e-6)
    n8=max(int(round(d/8.0))+1,4)
    t=np.linspace(s_[0],s_[-1],n8)
    q=np.stack([np.interp(t,s_,x[:,i]) for i in range(3)],1)
    k=menger(q); Rc=1.0/np.maximum(k,1e-9)
    T=tang(q); bd=np.degrees(np.arccos(np.clip((T[:-1]*T[1:]).sum(1),-1,1)))
    o.update(Rc_min=float(Rc.min()),Rc_p25=float(np.percentile(Rc,25)),
             bend_max=float(bd.max()),turn_cum=float(bd.sum()),turn_per_mm=float(bd.sum()/max(d,1e-9)),
             n_Rc_lt8=int((Rc<8).sum()))
    b=np.cross(T[:-1],T[1:]); nb=np.linalg.norm(b,axis=1,keepdims=True); bu=b/np.maximum(nb,1e-12)
    o["n_infl"]=int(((bu[:-1]*bu[1:]).sum(1)<0).sum())
    return o
rows=[]
for seed,v in EPS.items():
    p=v["P"]; h=v["H"]
    if p["plen"]<=166.91: continue
    key=name_map[p["anat"]]; s_t=p["plen"]-OFF
    g=win_geom(key,s_t)
    if g is None:
        print("SHORT WINDOW",seed,p["anat"],p["plen"],s_t); continue
    rows.append(dict(seed=int(seed),anat=p["anat"],plen=p["plen"],s_tgt=s_t,
                     succ=p["succ"],steps=p["steps"],hsucc=h["succ"],hsteps=h["steps"],**g))
json.dump(rows,open(M+"h2diff_eprows.json","w"),indent=1)
print("grafted episodes with geometry:",len(rows))
by=collections.defaultdict(list)
for r in rows: by[r["anat"]].append(r)
print("%-12s %3s %5s %7s %7s %7s %7s %7s %7s"%("anat","n","rate","s_min","s_max","dg_med","turn_md","Rcmin_md","clrmin"))
for a in sorted(by):
    R=by[a]; st=[r["s_tgt"] for r in R]
    print("%-12s %3d %5.2f %7.1f %7.1f %7.1f %7.1f %7.2f %7.3f"%(a,len(R),np.mean([r["succ"] for r in R]),
        min(st),max(st),np.median([r["dgraft"] for r in R]),np.median([r["turn_cum"] for r in R]),
        np.median([r["Rc_min"] for r in R]),min(r["clr_min"] for r in R)))
