import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atk4_geom import load_cl, arclen, resample

def load_obj(p):
    V=[]; F=[]
    for line in open(p, errors="replace"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx=[int(t.split("/")[0]) for t in line.split()[1:]]
            idx=[i-1 if i>0 else len(V)+i for i in idx]
            for j in range(1,len(idx)-1):
                F.append([idx[0],idx[j],idx[j+1]])
    return np.array(V,float), np.array(F,int)

def ray_hits(V,F,origins,dirs,chunk=400):
    """Min positive hit distance for each ray. Moller-Trumbore, vectorised over tris."""
    A=V[F[:,0]]; B=V[F[:,1]]; C=V[F[:,2]]
    e1=B-A; e2=C-A
    out=np.full(len(origins), np.inf)
    for i0 in range(0,len(origins),chunk):
        O=origins[i0:i0+chunk]; D=dirs[i0:i0+chunk]
        # (nr,1,3) x (1,nt,3)
        pv=np.cross(D[:,None,:], e2[None,:,:])
        det=np.einsum('td,rtd->rt', e1, pv)
        ok=np.abs(det)>1e-10
        inv=np.where(ok,1.0/np.where(ok,det,1.0),0.0)
        tv=O[:,None,:]-A[None,:,:]
        u=np.einsum('rtd,rtd->rt', tv, pv)*inv
        qv=np.cross(tv, e1[None,:,:])
        v=np.einsum('rd,rtd->rt', D, qv)*inv
        t=np.einsum('td,rtd->rt', e2, qv)*inv
        good=ok&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>1e-6)
        t=np.where(good,t,np.inf)
        out[i0:i0+chunk]=t.min(axis=1)
    return out

def frame(t):
    t=t/np.linalg.norm(t)
    a=np.array([0,0,1.0]) if abs(t[2])<0.9 else np.array([1.0,0,0])
    n1=np.cross(t,a); n1/=np.linalg.norm(n1)
    n2=np.cross(t,n1)
    return n1,n2

def clearance_profile(cl_json, obj, ds=1.0, ndir=48):
    P,r=load_cl(cl_json); s=arclen(P)
    q,R=resample(P,ds)
    V,F=load_obj(obj)
    T=np.gradient(R,axis=0)
    O=[];D=[]
    ang=np.arange(ndir)*2*np.pi/ndir
    for i in range(len(q)):
        n1,n2=frame(T[i])
        for a_ in ang:
            O.append(R[i]); D.append(np.cos(a_)*n1+np.sin(a_)*n2)
    O=np.array(O); D=np.array(D)
    D/= np.linalg.norm(D,axis=1)[:,None]
    hit=ray_hits(V,F,O,D)
    hit=hit.reshape(len(q),ndir)
    inside_frac=np.mean(np.isfinite(hit),axis=1)
    hmin=np.nanmin(np.where(np.isfinite(hit),hit,np.nan),axis=1)
    hmed=np.nanmedian(np.where(np.isfinite(hit),hit,np.nan),axis=1)
    rq=np.interp(q,s,r)
    return q,hmin,hmed,rq,inside_frac,(V,F)
