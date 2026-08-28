import os, json
BODY = r'''
import sys, os, json, math
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np, pyvista as pv, vtk
from eve_bench.dualdevicenav import load_branches

ROOT="/opt/eve_training/results_topbrain/anatomies"
OFF=33.314
THR={"wire_0.18":0.18,"sofa_0.30":0.30,"cath_0.35":0.35}
TERM_MM=8.0
MESH2NAME={"topcowmr004":"topcow_mr_004","topcowmr008":"topcow_mr_008",
           "topcowmr017":"topcow_mr_017","topcowmr023":"topcow_mr_023"}

def arclen(c): return np.concatenate(([0.0],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))))
def signed(mesh,pts):
    imp=vtk.vtkImplicitPolyDataDistance(); imp.SetInput(mesh)
    return np.array([imp.EvaluateFunction(p) for p in pts])
def enclosed(mesh,pts):
    ps=pv.PolyData(np.asarray(pts,float))
    sel=vtk.vtkSelectEnclosedPoints(); sel.SetInputData(ps); sel.SetSurfaceData(mesh)
    sel.SetTolerance(1e-6); sel.CheckSurfaceOff(); sel.Update()
    o=sel.GetOutput().GetPointData().GetArray("SelectedPoints")
    return np.array([o.GetTuple1(i) for i in range(len(pts))])>0.5
def densify(C,step=0.25):
    S=arclen(C); g=np.arange(0.0,S[-1],step); g=np.append(g,S[-1])
    P=np.stack([np.interp(g,S,C[:,i]) for i in range(3)],1); return P,g
def runs(mask,g,d_eff):
    r=[];i=0
    while i<len(mask):
        if mask[i]:
            j=i
            while j+1<len(mask) and mask[j+1]: j+=1
            r.append(dict(s0=float(g[i]),s1=float(g[j]),length=float(g[j]-g[i]),
                          min_d=float(d_eff[i:j+1].min()))); i=j+1
        else: i+=1
    return r
def proj_poly(P,S,pts):
    A=P[:-1]; B=P[1:]; AB=B-A; L2=(AB*AB).sum(1); L2[L2==0]=1e-12
    seglen=np.linalg.norm(AB,axis=1)
    out_d=[]; out_s=[]
    for q in np.asarray(pts,float):
        t=((q-A)*AB).sum(1)/L2; t=np.clip(t,0,1)
        proj=A+t[:,None]*AB; d=np.linalg.norm(proj-q,axis=1)
        k=int(d.argmin())
        out_d.append(float(d[k])); out_s.append(float(S[k]+t[k]*seglen[k]))
    return np.array(out_d), np.array(out_s)
def interp_at(C,S,s):
    return np.stack([np.interp(s,S,C[:,i]) for i in range(3)],-1)
def kabsch(Plog,Qgeo):
    """rigid R,t mapping log frame -> geometry frame (allow reflection off)."""
    pc=Plog.mean(0); qc=Qgeo.mean(0)
    H=(Plog-pc).T@(Qgeo-qc)
    U,Sv,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,d]); R=Vt.T@D@U.T
    t=qc-R@pc
    return R,t

D=json.loads(DATA)
fails=D["fails"]; allep=D["allep"]

# ---------- load geometry ----------
GEO={}; CL={}; RVA={}
for fp,name in MESH2NAME.items():
    d0=os.path.join(ROOT,name)
    mesh=pv.read(os.path.join(d0,"vessel_architecture_collision.obj")).triangulate().clean()
    brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    rc=next(b for b in brs if "RCCA" in str(b.name).upper())
    rv=next(b for b in brs if "RVA" in str(b.name).upper())
    br11=next(b for b in brs if "(11)" in str(b.name))
    C=np.asarray(rc.coordinates,float); S=arclen(C); L=float(S[-1])
    V=np.asarray(rv.coordinates,float); SV=arclen(V)
    B=np.asarray(br11.coordinates,float)
    CL[fp]=(C,S); RVA[fp]=(V,SV)
    P,g=densify(C); sd=signed(mesh,P)
    if (sd<0).mean()<0.5: sd=-sd
    enc=enclosed(mesh,P); d=np.abs(sd); d_eff=np.where(enc,d,0.0)
    rec=dict(name=name,fp=fp,L=L,nstat=len(C),min_d=float(d_eff.min()),
             nout=int((~enc).sum()), agree=float(100.0*((sd<0)==enc).mean()),
             rcca_first=C[0].tolist(), rcca_last=C[-1].tolist(),
             br11_first=B[0].tolist(), br11_last=B[-1].tolist(), br11_n=len(B))
    for k,t in THR.items():
        rr=runs(d_eff<t,g,d_eff)
        for r in rr: r["terminal"]=(r["s1"]>=L-TERM_MM)
        mid=[r for r in rr if not r["terminal"]]
        rec[k]=dict(n_runs=len(rr),n_mid=len(mid),
                    s_block=(min(r["s0"] for r in mid) if mid else None),
                    s_block_any=(min(r["s0"] for r in rr) if rr else None), runs=rr)
    ins=np.where(enc)[0]; rec["s_last_inside"]=float(g[ins[-1]]) if len(ins) else None
    # RVA divergence from RCCA
    dv,sv=proj_poly(C,S,V)
    shared=dv<0.5
    idx=np.where(~shared)[0]
    i0=int(idx[0]) if len(idx) else len(V)-1
    rec["rva_fork_sv"]=float(SV[max(i0-1,0)]); rec["rva_fork_s_rcca"]=float(sv[max(i0-1,0)])
    rec["rva_L"]=float(SV[-1]); rec["rva_shared_n"]=int(shared.sum()); rec["rva_n"]=len(V)
    GEO[fp]=rec
    rr=rec["sofa_0.30"]
    print("[geom] %-12s L=%.1f nstat=%d nOUT=%d agree=%.1f min_d=%.3f | sofa nruns=%d nMID=%d s_block=%s | s_last_inside=%.1f | RVAfork s_rcca=%.2f sv=%.2f shared=%d/%d"%(
        fp,L,rec["nstat"],rec["nout"],rec["agree"],rec["min_d"],rr["n_runs"],rr["n_mid"],
        str(rr["s_block"]),rec["s_last_inside"],rec["rva_fork_s_rcca"],rec["rva_fork_sv"],
        rec["rva_shared_n"],rec["rva_n"]), flush=True)
    for k in THR:
        print("      %-10s"%k,[(round(r["s0"],1),round(r["s1"],1),round(r["min_d"],3),"TERM" if r["terminal"] else "MID") for r in rec[k]["runs"]][:14], flush=True)

# ---------- fit log-frame -> geometry-frame rigid transform ----------
# correspondence: each episode target (log frame) <-> RCCA station at s = path_len - OFF
Plog=[]; Qgeo=[]
for e in allep:
    C,S=CL[e["mesh"]]
    s=e["path_len"]-OFF
    if s<0 or s>S[-1]: continue
    Plog.append(e["tgt"]); Qgeo.append(interp_at(C,S,s))
Plog=np.array(Plog,float); Qgeo=np.array(Qgeo,float)
R,t=kabsch(Plog,Qgeo)
res=np.linalg.norm((R@Plog.T).T+t-Qgeo,axis=1)
print("\n[frame] Kabsch on %d target correspondences: residual med=%.3f mm p95=%.3f max=%.3f"%(
    len(Plog),float(np.median(res)),float(np.percentile(res,95)),float(res.max())))
print("[frame] R=\n",np.round(R,4),"\n[frame] t=",np.round(t,3), flush=True)
def L2G(pts): return (R@np.asarray(pts,float).T).T+t

# ---------- planned-path polyline in geometry frame: bridge(11) tail + RCCA ----------
PATH={}
for fp in MESH2NAME:
    C,S=CL[fp]
    d0=os.path.join(ROOT,MESH2NAME[fp])
    brs=load_branches(os.path.join(d0,"Centrelines_comb"))
    B=np.asarray(next(b for b in brs if "(11)" in str(b.name)).coordinates,float)
    SB=arclen(B)
    # insertion sits start_point_offset=2 points into (11); planned proj_s=0 there
    ins_i=2
    Bp=B[ins_i:]
    full=np.vstack([Bp,C[1:]]) if np.linalg.norm(Bp[-1]-C[0])<1e-6 else np.vstack([Bp,C])
    SF=arclen(full)
    PATH[fp]=(full,SF, float(SF[len(Bp)-1]))   # arclength of the RCCA ostium along planned path
    print("[path] %s bridge_tail_len=%.2f  ostium_s_on_path=%.2f (OFF=%.3f)  total=%.1f"%(
        fp,float(SB[-1]-SB[ins_i]),PATH[fp][2],OFF,float(SF[-1])), flush=True)

# ---------- verify with successful trajectories? (only failures embedded) ----------
print("\n=== 23 H0 FAILURES: trajectory decomposition (geometry frame) ===")
hdr=("%-12s%4s%5s%9s%8s | %8s%8s%8s | %8s%8s | %7s%7s%7s | %6s%6s"%(
  "mesh","ep","pid","path_len","tgt_s","maxProjS","finProjS","maxSrcca","dPath_md","dPath_mx",
  "RVApen","nRVA","physV","onp1","nstep"))
print(hdr)
FA=[]
for f in fails:
    fp=f["mesh"]; C,S=CL[fp]; V,SV=RVA[fp]; full,SF,ost=PATH[fp]
    T=L2G(f["tips"])
    dP,sP=proj_poly(full,SF,T)          # distance to planned path, arclen along planned path
    dR,sR=proj_poly(C,S,T)              # distance to RCCA
    dV,sV=proj_poly(V,SV,T)             # distance to RVA
    forkSV=GEO[fp]["rva_fork_sv"]
    pen=np.where(dV<dP, sV-forkSV, -1.0)   # mm into RVA beyond fork, when closer to RVA than to path
    physV=sum(1 for p in f["phys"] if p=="RVA")
    onp1=sum(f["on_path"])
    r=dict(mesh=fp,ep=f["ep"],pid=f["pid"],seed=f["seed"],path_len=f["path_len"],
        tgt_s=f["path_len"]-OFF,
        max_projs=float(sP.max()), fin_projs=float(sP[-1]),
        max_s_rcca=float(sP.max()-ost), fin_s_rcca=float(sP[-1]-ost),
        dpath_med=float(np.median(dP)), dpath_max=float(dP.max()),
        dpath_end_med=float(np.median(dP[-50:])),
        rva_pen_max=float(pen.max()), rva_pen_end=float(np.median(pen[-50:])),
        n_closer_rva=int((dV<dP).sum()), n_pen_gt2=int((pen>2.0).sum()),
        n_pen_gt5=int((pen>5.0).sum()), physV=physV, onp1=onp1, n=len(T),
        final_branch=f["final_branch"], log_max_proj=max(f["proj_s"]),
        log_fin_proj=f["proj_s"][-1])
    FA.append(r)
    print("%-12s%4d%5d%9.1f%8.1f | %8.1f%8.1f%8.1f | %8.2f%8.2f | %7.1f%7d%7d | %6d%6d"%(
      fp,r["ep"],r["pid"],r["path_len"],r["tgt_s"],r["max_projs"],r["fin_projs"],r["max_s_rcca"],
      r["dpath_med"],r["dpath_max"],r["rva_pen_max"],r["n_closer_rva"],r["physV"],r["onp1"],r["n"]), flush=True)

print("\n[check] log proj_s vs geometric proj_s (failures): ")
for r in FA[:6]:
    print("   %s ep%d pid%d  log_max=%.1f  geo_max=%.1f  log_fin=%.1f geo_fin=%.1f"%(
        r["mesh"],r["ep"],r["pid"],r["log_max_proj"],r["max_projs"],r["log_fin_proj"],r["fin_projs"]))

print("\n@@@JSON@@@")
print(json.dumps(dict(geo=GEO, fails=FA,
    frame=dict(R=R.tolist(), t=t.tolist(), res_med=float(np.median(res)), res_max=float(res.max())),
    ost={k:PATH[k][2] for k in PATH})))
'''

data = open(os.path.join(os.path.dirname(__file__), "h0_fail_slim.json")).read()
out = os.path.join(os.path.dirname(__file__), "h0_task4_geo.py")
with open(out, "w") as fh:
    fh.write("DATA = r" + chr(39)*3 + data + chr(39)*3 + "\n")
    fh.write(BODY)
print("wrote", out, os.path.getsize(out)/1e6, "MB")
