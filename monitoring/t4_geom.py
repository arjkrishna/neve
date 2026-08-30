import json, math, os, glob, pickle
import numpy as np
ROOT="D:/Arjun/workspace/neve/topbrain_data/anatomies"
def load_cl(path):
    d=json.load(open(path)); m=d["markups"][0]
    P=np.array([c["position"] for c in m["controlPoints"]],float); r=None
    for meas in m.get("measurements",[]):
        if meas.get("name")=="Radius" and meas.get("controlPointValues"):
            r=np.array(meas["controlPointValues"],float)
    return P,r
def arclen(P):
    d=np.linalg.norm(np.diff(P,axis=0),axis=1); return np.concatenate([[0.0],np.cumsum(d)])
def resample(P,ds=0.25):
    s=arclen(P); q=np.arange(0,s[-1],ds)
    return q, np.stack([np.interp(q,s,P[:,i]) for i in range(3)],axis=1), s
def curv(R,q,win):
    k=int(round(win/(q[1]-q[0]))); Rc=np.full(len(q),np.nan)
    for i in range(k,len(q)-k):
        a,b,c=R[i-k],R[i],R[i+k]; v1=a-b; v2=c-b
        n1=np.linalg.norm(v1);n2=np.linalg.norm(v2);n3=np.linalg.norm(a-c)
        if min(n1,n2,n3)<1e-9: continue
        A=0.5*np.linalg.norm(np.cross(v1,v2))
        if A<1e-12: Rc[i]=1e6; continue
        Rc[i]=(n1*n2*n3)/(4*A)
    return Rc
def bend(R,q,base):
    k=int(round(base/(q[1]-q[0]))); a=np.full(len(q),np.nan)
    for i in range(k,len(q)-k):
        t1=R[i]-R[i-k]; t2=R[i+k]-R[i]
        n1=np.linalg.norm(t1);n2=np.linalg.norm(t2)
        if min(n1,n2)<1e-9: continue
        a[i]=math.degrees(math.acos(np.clip(np.dot(t1,t2)/(n1*n2),-1,1)))
    return a
out={}
for d in sorted(glob.glob(os.path.join(ROOT,"topcow_mr_*"))):
    nm=os.path.basename(d); p=os.path.join(d,"Centrelines_comb","Centerline curve - RCCA.mrk.json")
    if not os.path.exists(p): continue
    P,r=load_cl(p); q,R,s=resample(P)
    rq=np.interp(q,s,r) if r is not None else None
    out[nm]=dict(q=q,Rc2=curv(R,q,2.0),Rc5=curv(R,q,5.0),b5=bend(R,q,5.0),b10=bend(R,q,10.0),
                 r=rq,L=float(s[-1]),P=R)
    print(nm,"L=%.2f"%s[-1],"npts",len(P))
pickle.dump(out,open("_t4_geom.pkl","wb"))
