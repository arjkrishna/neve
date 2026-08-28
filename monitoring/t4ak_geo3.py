import os, json
import numpy as np
import vtk

ROOT="/opt/eve_training/results_topbrain/anatomies"
MESH="vessel_architecture_collision.obj"; SUB="Centrelines_comb"
FP2NAME={"topcowmr004":"topcow_mr_004","topcowmr008":"topcow_mr_008",
         "topcowmr017":"topcow_mr_017","topcowmr023":"topcow_mr_023"}
OFF=33.314

def load_branch(p):
    d=json.load(open(p)); P=[];R=[]
    for m in d["markups"]:
        if m["type"]!="Curve": continue
        for cp in m["controlPoints"]:
            x,y,z=[float(v) for v in cp["position"]]; P.append((y,-z,-x))
        for me in m.get("measurements",[]):
            if me["name"]=="Radius": R.extend(me["controlPointValues"])
    return np.array(P,float), np.array(R,float)

def arclen(P):
    return np.concatenate([[0.],np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))])

def pt_at(P,s,q):
    return np.stack([np.interp(q,s,P[:,i]) for i in range(3)],axis=-1)

def imp_of(objpath):
    r=vtk.vtkOBJReader(); r.SetFileName(objpath); r.Update()
    imp=vtk.vtkImplicitPolyDataDistance(); imp.SetInput(r.GetOutput()); return imp

def proj_poly(pts, poly, s_poly):
    A=poly[:-1]; B=poly[1:]; AB=B-A; L2=(AB*AB).sum(1); L2[L2==0]=1e-12
    Ln=np.sqrt(L2); D=np.empty(len(pts)); S=np.empty(len(pts)); C=np.empty((len(pts),3))
    for k,p in enumerate(pts):
        t=np.clip(((p-A)*AB).sum(1)/L2,0,1)
        Pj=A+t[:,None]*AB
        d=np.linalg.norm(Pj-p,axis=1)
        j=int(np.argmin(d)); D[k]=d[j]; S[k]=s_poly[j]+t[j]*Ln[j]; C[k]=Pj[j]
    return D,S,C

IN=json.load(open("/opt/mon/t4ak_input.json"))
G={}
for fp,name in FP2NAME.items():
    root=os.path.join(ROOT,name)
    Pr,Rr=load_branch(os.path.join(root,SUB,"Centerline curve - RCCA.mrk.json"))
    Pv,Rv=load_branch(os.path.join(root,SUB,"Centerline curve - RVA.mrk.json"))
    G[fp]=dict(Pr=Pr,Rr=Rr,sr=arclen(Pr),Pv=Pv,Rv=Rv,sv=arclen(Pv),imp=imp_of(os.path.join(root,MESH)))

Xs=[];Xg=[]
for r in IN["h0"]:
    g=G[r["mesh"]]; Xs.append(r["tgt"]); Xg.append(pt_at(g["Pr"],g["sr"],r["path_len"]-OFF))
Xs=np.array(Xs); Xg=np.array(Xg); cs=Xs.mean(0); cg=Xg.mean(0)
H=(Xs-cs).T@(Xg-cg); U,S,Vt=np.linalg.svd(H); dd=np.sign(np.linalg.det(Vt.T@U.T))
R=Vt.T@np.diag([1,1,dd])@U.T; t=cg-R@cs
print("[kabsch] residual max %.4f mm"%np.linalg.norm((Xs@R.T+t)-Xg,axis=1).max())

def seg_min_sd(imp,a,b,n=201):
    lam=np.linspace(0,1,n)
    P=a[None,:]*(1-lam[:,None])+b[None,:]*lam[:,None]
    v=np.array([imp.EvaluateFunction(p) for p in P])
    return float(v.min()), int((v<0).sum())

print("")
print("=== DECISIVE LUMEN-SEPARATION TEST (segment tip -> nearest RCCA centerline point; sd<0 means wall crossed) ===")
print("mesh  ep/pid  when      tip_dR  rR   sV    | minSD_on_seg  nOutside/201 | verdict")
OUT=[]
for r in IN["h0"]:
    if r["succ"]: continue
    g=G[r["mesh"]]
    T=np.asarray(r["tip"])@R.T+t
    dR,sR,CR=proj_poly(T,g["Pr"],g["sr"]); dV,sV,_=proj_poly(T,g["Pv"],g["sv"])
    rR=np.interp(sR,g["sr"],g["Rr"]); rV=np.interp(sV,g["sv"],g["Rv"])
    inR=dR<rR; inV=dV<rV
    cand=np.where(inV&(~inR))[0]
    picks=[("final",len(T)-1)]
    if len(cand): picks.append(("maxRVA",int(cand[np.argmax(sV[cand])])))
    rec=dict(mesh=r["mesh"],ep=r["ep"],pid=r["pid"],path_len=r["path_len"])
    for lab,i in picks:
        mn,no=seg_min_sd(g["imp"],T[i],CR[i])
        verdict="SEPARATE LUMEN (wall between)" if mn<0 else "same lumen / shared trunk"
        rec[lab]=dict(i=i,dR=float(dR[i]),rR=float(rR[i]),sV=float(sV[i]),minsd=mn,nout=no,sep=bool(mn<0))
        print("%s %2d/%-4d %-8s %6.2f %5.2f %6.2f | %11.3f %8d      | %s"
              %(r["mesh"][-5:],r["ep"],r["pid"],lab,dR[i],rR[i],sV[i],mn,no,verdict))
    # RVA excursion timing
    exc=inV&(~inR)
    first=int(np.argmax(exc)) if exc.any() else -1
    lastx=int(len(exc)-1-np.argmax(exc[::-1])) if exc.any() else -1
    rec["exc_first"]=first; rec["exc_last"]=lastx; rec["exc_n"]=int(exc.sum()); rec["n"]=len(T)
    rec["outR_n"]=int((~inR).sum())
    OUT.append(rec)

print("")
print("=== RVA-EXCURSION TIMING (step index within the 600-step episode) ===")
print("mesh  ep/pid  plen   nsteps  outRCCA_steps  RVA_excursion_steps  first  last  steps_after_last")
for rec in OUT:
    print("%s %2d/%-4d %6.1f %6d %13d %20d %6d %5d %16d"
          %(rec["mesh"][-5:],rec["ep"],rec["pid"],rec["path_len"],rec["n"],rec["outR_n"],rec["exc_n"],
            rec["exc_first"],rec["exc_last"],rec["n"]-1-rec["exc_last"]))
json.dump(OUT,open("/opt/mon/t4ak_geo3.json","w"))
print("WROTE /opt/mon/t4ak_geo3.json")
