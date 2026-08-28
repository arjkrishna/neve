import os, re, json, math
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = "/opt/eve_training/results_topbrain/anatomies"
MESH = "vessel_architecture_collision.obj"
SUB  = "Centrelines_comb"
FP2NAME = {"topcowmr004":"topcow_mr_004","topcowmr008":"topcow_mr_008",
           "topcowmr017":"topcow_mr_017","topcowmr023":"topcow_mr_023"}
OFF = 33.314

def load_branch(path):
    d = json.load(open(path))
    P=[]; R=[]
    for m in d["markups"]:
        if m["type"]!="Curve": continue
        for cp in m["controlPoints"]:
            x,y,z = [float(v) for v in cp["position"]]
            P.append((y,-z,-x))
        for meas in m.get("measurements",[]):
            if meas["name"]=="Radius":
                R.extend(meas["controlPointValues"])
    return np.array(P,dtype=np.float64), np.array(R,dtype=np.float64)

def arclen(P):
    d = np.linalg.norm(np.diff(P,axis=0),axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def resample(P, s, step=0.25):
    out_s = np.arange(0.0, s[-1]+1e-9, step)
    out = np.empty((len(out_s),3))
    for i in range(3):
        out[:,i] = np.interp(out_s, s, P[:,i])
    return out_s, out

def interp_scalar(s, v, q):
    return np.interp(q, s, v)

def implicit(objpath):
    r = vtk.vtkOBJReader(); r.SetFileName(objpath); r.Update()
    pd = r.GetOutput()
    imp = vtk.vtkImplicitPolyDataDistance(); imp.SetInput(pd)
    # boundary edges
    fe = vtk.vtkFeatureEdges(); fe.SetInputData(pd)
    fe.BoundaryEdgesOn(); fe.FeatureEdgesOff(); fe.NonManifoldEdgesOff(); fe.ManifoldEdgesOff()
    fe.Update()
    return imp, pd.GetNumberOfPoints(), pd.GetNumberOfCells(), fe.GetOutput().GetNumberOfCells()

def runs_below(s, clear, thr):
    bad = clear < thr
    out=[]; i=0; n=len(bad)
    while i<n:
        if bad[i]:
            j=i
            while j+1<n and bad[j+1]: j+=1
            out.append(dict(s0=float(s[i]), s1=float(s[j]),
                            min_clear=float(clear[i:j+1].min()),
                            terminal=bool(j==n-1)))
            i=j+1
        else: i+=1
    return out

def point_seg_dist(pts, poly):
    # pts (N,3), poly (M,3): returns min dist and arclength position on poly
    A = poly[:-1]; B = poly[1:]
    AB = B-A; L2 = (AB*AB).sum(1); L2[L2==0]=1e-12
    sA = arclen(poly)[:-1]
    D = np.empty(len(pts)); S = np.empty(len(pts))
    for k,p in enumerate(pts):
        t = ((p-A)*AB).sum(1)/L2
        t = np.clip(t,0,1)
        proj = A + t[:,None]*AB
        d = np.linalg.norm(proj-p,axis=1)
        j = int(np.argmin(d))
        D[k]=d[j]; S[k]= sA[j] + t[j]*math.sqrt(L2[j])
    return D,S

GEO={}
print("="*100)
for fp,name in FP2NAME.items():
    root=os.path.join(ROOT,name)
    Prcca,Rrcca = load_branch(os.path.join(root,SUB,"Centerline curve - RCCA.mrk.json"))
    Prva ,Rrva  = load_branch(os.path.join(root,SUB,"Centerline curve - RVA.mrk.json"))
    P11 ,R11   = load_branch(os.path.join(root,SUB,"Centerline curve (11).mrk.json"))
    s_r = arclen(Prcca); s_v = arclen(Prva); s_11=arclen(P11)
    imp,npts,ncell,nbound = implicit(os.path.join(root,MESH))
    qs, Q = resample(Prcca, s_r, 0.25)
    sd = np.array([imp.EvaluateFunction(p) for p in Q])   # negative inside
    clear = -sd
    # control-point level too
    sdc = np.array([imp.EvaluateFunction(p) for p in Prcca])
    nout = int((sdc>0).sum())
    last_in = float(s_r[np.max(np.where(sdc<0)[0])]) if (sdc<0).any() else -1.0
    rr = interp_scalar(s_r, Rrcca, qs) if len(Rrcca)==len(Prcca) else None
    res={}
    for tag,thr in (("wire",0.18),("sofa",0.30),("cath",0.35)):
        res[tag]=runs_below(qs, clear, thr)
    # RVA / RCCA separation
    Dv,Sv = point_seg_dist(Prva, Prcca)
    sep_idx=None
    if len(Rrva)==len(Prva) and len(Rrcca)==len(Prcca):
        r_at = interp_scalar(s_r, Rrcca, Sv)
        # first RVA node whose distance to RCCA centerline exceeds the RCCA local radius
        w = np.where(Dv > r_at)[0]
        sep_idx = int(w[0]) if len(w) else None
    GEO[fp]=dict(name=name, L=float(s_r[-1]), n=len(Prcca), npts=npts, ncell=ncell, nbound=nbound,
                 min_clear=float(clear.min()), nout=nout, s_last_inside=last_in,
                 runs=res, rL=float(s_v[-1]), n_rva=len(Prva),
                 s_r=s_r.tolist(), R_rcca=Rrcca.tolist(),
                 s_v=s_v.tolist(), R_rva=Rrva.tolist(),
                 rva_d_to_rcca=Dv.tolist(), rva_s_on_rcca=Sv.tolist(),
                 sep_s_v = (float(s_v[sep_idx]) if sep_idx is not None else None),
                 qs=qs.tolist(), clear=clear.tolist(),
                 bridge_tail=float(s_11[-1]-s_11[2]),
                 Prcca=Prcca.tolist(), Prva=Prva.tolist())
    print("[%s] %s L=%.2f n=%d verts=%d cells=%d openEdges=%d minClear=%.3f nOUT=%d lastIn=%.2f rvaL=%.2f sepAt_sv=%s bridgeTail=%.3f"
          % (fp,name,s_r[-1],len(Prcca),npts,ncell,nbound,clear.min(),nout,last_in,s_v[-1],
             ("%.2f"%GEO[fp]["sep_s_v"]) if GEO[fp]["sep_s_v"] is not None else "NA", GEO[fp]["bridge_tail"]))
    for tag in ("wire","sofa","cath"):
        print("    %-5s %s"%(tag, [(round(r["s0"],2),round(r["s1"],2),round(r["min_clear"],3),"TERM" if r["terminal"] else "MID") for r in res[tag]]))
    # radius / separation table near the fork
    print("    fork profile (s_v, dist_to_RCCA_centerline, r_RVA, r_RCCA_at_proj):")
    for i in range(0, min(40,len(Prva))):
        if s_v[i] > 40: break
        rv = Rrva[i] if len(Rrva)==len(Prva) else float("nan")
        rc = float(np.interp(Sv[i], s_r, Rrcca)) if len(Rrcca)==len(Prcca) else float("nan")
        if i%2==0:
            print("      s_v=%6.2f  d=%6.3f  r_rva=%5.2f  r_rcca=%5.2f  s_on_rcca=%6.2f" % (s_v[i],Dv[i],rv,rc,Sv[i]))


with open("/opt/mon/t4ak_geo.json","w") as f: json.dump(GEO,f)
print("WROTE /opt/mon/t4ak_geo.json")
