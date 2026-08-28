import os, json, math
import numpy as np
import vtk

ROOT="/opt/eve_training/results_topbrain/anatomies"
MESH="vessel_architecture_collision.obj"; SUB="Centrelines_comb"
FP2NAME={"topcowmr004":"topcow_mr_004","topcowmr008":"topcow_mr_008",
         "topcowmr017":"topcow_mr_017","topcowmr023":"topcow_mr_023"}
OFF=33.314
M=np.array([[0.,1.,0.],[0.,0.,-1.],[-1.,0.,0.]])

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

def resample(P,s,step=0.25):
    q=np.arange(0.,s[-1]+1e-9,step)
    return q, np.stack([np.interp(q,s,P[:,i]) for i in range(3)],axis=1)

def pt_at(P,s,q):
    return np.stack([np.interp(q,s,P[:,i]) for i in range(3)],axis=-1)

def implicit(objpath):
    r=vtk.vtkOBJReader(); r.SetFileName(objpath); r.Update(); pd=r.GetOutput()
    pdt=pd
    imp=vtk.vtkImplicitPolyDataDistance(); imp.SetInput(pdt)
    fe=vtk.vtkFeatureEdges(); fe.SetInputData(pdt); fe.BoundaryEdgesOn(); fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff(); fe.ManifoldEdgesOff(); fe.Update()
    return imp, pdt.GetNumberOfPoints(), fe.GetOutput().GetNumberOfCells()

def runs_below(s,c,thr):
    bad=c<thr; out=[]; i=0; n=len(bad)
    while i<n:
        if bad[i]:
            j=i
            while j+1<n and bad[j+1]: j+=1
            out.append(dict(s0=float(s[i]),s1=float(s[j]),min_clear=float(c[i:j+1].min()),terminal=bool(j==n-1)))
            i=j+1
        else: i+=1
    return out

def proj_poly(pts, poly, s_poly):
    A=poly[:-1]; B=poly[1:]; AB=B-A; L2=(AB*AB).sum(1); L2[L2==0]=1e-12
    Ln=np.sqrt(L2); D=np.empty(len(pts)); S=np.empty(len(pts))
    for k,p in enumerate(pts):
        t=np.clip(((p-A)*AB).sum(1)/L2,0,1)
        d=np.linalg.norm(A+t[:,None]*AB-p,axis=1)
        j=int(np.argmin(d)); D[k]=d[j]; S[k]=s_poly[j]+t[j]*Ln[j]
    return D,S

IN=json.load(open("/opt/mon/t4ak_input.json"))
G={}
for fp,name in FP2NAME.items():
    root=os.path.join(ROOT,name)
    Pr,Rr=load_branch(os.path.join(root,SUB,"Centerline curve - RCCA.mrk.json"))
    Pv,Rv=load_branch(os.path.join(root,SUB,"Centerline curve - RVA.mrk.json"))
    P11,_=load_branch(os.path.join(root,SUB,"Centerline curve (11).mrk.json"))
    sr=arclen(Pr); sv=arclen(Pv); s11=arclen(P11)
    imp,nv,nb=implicit(os.path.join(root,MESH))
    qs,Q=resample(Pr,sr,0.25)
    clear=np.array([imp.EvaluateFunction(p) for p in Q])
    sdc=np.array([imp.EvaluateFunction(p) for p in Pr])
    nout=int((sdc<0).sum())
    last_in=float(sr[int(np.max(np.where(sdc>0)[0]))]) if (sdc>0).any() else -1.
    G[fp]=dict(name=name,Pr=Pr,Rr=Rr,sr=sr,Pv=Pv,Rv=Rv,sv=sv,imp=imp,
               L=float(sr[-1]),qs=qs,clear=clear,nout=nout,last_in=last_in,
               nverts=nv,nopen=nb,bridge=float(s11[-1]-s11[2]),
               runs=dict((t,runs_below(qs,clear,th)) for t,th in (("wire",0.18),("sofa",0.30),("cath",0.35))))

print("="*110)
for fp,g in G.items():
    print("[geom] %s L=%.2f verts=%d openEdges=%d nOUT=%d lastIn=%.2f minClear=%.3f bridgeTail=%.3f"
          %(fp,g["L"],g["nverts"],g["nopen"],g["nout"],g["last_in"],g["clear"].min(),g["bridge"]))
    for t in ("wire","sofa","cath"):
        print("       %-5s %s"%(t,[(round(r["s0"],2),round(r["s1"],2),round(r["min_clear"],3),"TERM" if r["terminal"] else "MID") for r in g["runs"][t]]))

Xs=[];Xg=[]
for r in IN["h0"]:
    g=G[r["mesh"]]; s=r["path_len"]-OFF
    Xs.append(r["tgt"]); Xg.append(pt_at(g["Pr"],g["sr"],s))
Xs=np.array(Xs); Xg=np.array(Xg)
cs=Xs.mean(0); cg=Xg.mean(0)
H=(Xs-cs).T@(Xg-cg); U,S,Vt=np.linalg.svd(H); dd=np.sign(np.linalg.det(Vt.T@U.T))
R=Vt.T@np.diag([1,1,dd])@U.T; t=cg-R@cs
res=np.linalg.norm((Xs@R.T+t)-Xg,axis=1)
print("[sanity] clearance vs radius (median |clear-r|):")
for fp,g in G.items():
    rr=np.interp(g["qs"],g["sr"],g["Rr"])
    print("   %s  med|c-r|=%.3f  med_c=%.3f med_r=%.3f  min_c=%.3f at s=%.2f"%(fp,np.median(np.abs(g["clear"]-rr)),np.median(g["clear"]),np.median(rr),g["clear"].min(),g["qs"][int(np.argmin(g["clear"]))]))
print("[kabsch] n=%d residual med=%.4f p95=%.4f max=%.4f mm"%(len(res),np.median(res),np.percentile(res,95),res.max()))
def to_geo(p): return np.asarray(p)@R.T+t

print("")
print("[reach] per-anatomy clearance blocking")
for fp,g in G.items():
    mid=[r for r in g["runs"]["wire"] if not r["terminal"]]
    midc=[r for r in g["runs"]["cath"] if not r["terminal"]]
    print("   %s L=%.2f (path_len %.2f) wireMID=%s cathMID=%s cathTERM=%s"
          %(fp,g["L"],g["L"]+OFF,[(round(r["s0"],2),round(r["s1"],2)) for r in mid],
            [(round(r["s0"],2),round(r["s1"],2)) for r in midc],
            [(round(r["s0"],2),round(r["s1"],2),round(r["min_clear"],3)) for r in g["runs"]["cath"] if r["terminal"]]))

print("")
print("=== H0 23 FAILURES: geometric decomposition ===")
print("mesh  ep/pid  plen   tgt_s | sRmax sRfin | dRfin rRfin inR | sVfin dVfin rVfin inV | outR%  inV%  maxSVinV | move100 prange100 | mode")
FR=[]
for r in IN["h0"]:
    if r["succ"]: continue
    g=G[r["mesh"]]
    T=to_geo(np.array(r["tip"]))
    dR,sR=proj_poly(T,g["Pr"],g["sr"]); dV,sV=proj_poly(T,g["Pv"],g["sv"])
    rR=np.interp(sR,g["sr"],g["Rr"]); rV=np.interp(sV,g["sv"],g["Rv"])
    inR=dR<rR; inV=dV<rV
    sd=np.array([g["imp"].EvaluateFunction(p) for p in T])  # positive inside
    projs=np.array(r["projs"])
    L100=slice(max(0,len(T)-100),len(T))
    tipmove=float(np.linalg.norm(np.diff(T[L100],axis=0),axis=1).sum())
    prange=float(projs[L100].max()-projs[L100].min())
    dproj=float(projs[-1]-projs[L100].start if False else projs[-1]-projs[max(0,len(projs)-100)])
    if projs[-1]>=projs.max()-0.3 and dproj>1.0: mode="advancing_at_cap"
    elif prange>3.0: mode="oscillating"
    elif (~inR)[L100].mean()>0.5: mode="off_path_arrested"
    else: mode="arrested_on_path"
    rec=dict(mesh=r["mesh"],ep=r["ep"],pid=r["pid"],seed=r["seed"],path_len=r["path_len"],
             tgt_s=r["path_len"]-OFF,fb=r["fb"],n=r["n"],
             s_rcca_max=float(projs.max()-OFF), s_rcca_fin=float(projs[-1]-OFF),
             sR_fin=float(sR[-1]),dR_fin=float(dR[-1]),rR_fin=float(rR[-1]),inR_fin=bool(inR[-1]),
             sV_fin=float(sV[-1]),dV_fin=float(dV[-1]),rV_fin=float(rV[-1]),inV_fin=bool(inV[-1]),
             frac_out_R=float((~inR).mean()),frac_in_V=float((inV&(~inR)).mean()),
             maxSV_inV=float(sV[inV&(~inR)].max()) if (inV&(~inR)).any() else 0.0,
             maxSR_inR=float(sR[inR].max()) if inR.any() else -1.0,
             sd_min=float(sd.min()),n_tip_outside=int((sd<0).sum()),tipmove100=tipmove,prange100=prange,dproj100=dproj,mode=mode)
    FR.append(rec)
    print("%s %2d/%-4d %6.1f %6.1f | %5.1f %5.1f | %5.2f %5.2f %5s | %5.2f %5.2f %5.2f %5s | %5.1f %5.1f %6.2f | %7.1f %8.2f | %s"
          %(rec["mesh"][-5:],rec["ep"],rec["pid"],rec["path_len"],rec["tgt_s"],rec["s_rcca_max"],rec["s_rcca_fin"],
            rec["dR_fin"],rec["rR_fin"],rec["inR_fin"],rec["sV_fin"],rec["dV_fin"],rec["rV_fin"],rec["inV_fin"],
            100*rec["frac_out_R"],100*rec["frac_in_V"],rec["maxSV_inV"],rec["tipmove100"],rec["prange100"],rec["mode"]))

out=dict(fails=FR,
         geo=dict((fp,dict(name=g["name"],L=g["L"],nout=g["nout"],last_in=g["last_in"],
                      minclear=float(g["clear"].min()),nopen=g["nopen"],bridge=g["bridge"],
                      runs=g["runs"],qs=g["qs"].tolist(),clear=g["clear"].tolist(),
                      sr=g["sr"].tolist(),Rr=g["Rr"].tolist())) for fp,g in G.items()),
         R=R.tolist(),t=t.tolist(),kabsch_res=[float(np.median(res)),float(res.max())])
json.dump(out,open("/opt/mon/t4ak_geo2.json","w"))
print("WROTE /opt/mon/t4ak_geo2.json")
