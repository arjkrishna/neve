"""ATTACK 1: derive path_len<->RCCA-arclength mapping and decompose 98 eps by graft seam."""
import os, re, json, glob, math, sys
from collections import Counter, defaultdict
import numpy as np

ROOT = r"d:/Arjun/workspace/neve"
RUNDIR = os.path.join(ROOT, "saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
STAMP = sys.argv[1] if len(sys.argv) > 1 else "20260828_045651"
LOGDIR = os.path.join(RUNDIR, "logs", STAMP)
ANAT = os.path.join(ROOT, "topbrain_data/anatomies")
HOST = os.path.join(ROOT, "eve_bench/data/dualdevicenav/Centrelines_comb")

def rotmat(zx=(20.0, 5.0)):
    rz = -zx[0]*np.pi/180; rx = -zx[1]*np.pi/180
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    return Rz@Rx
M = rotmat()
def t2v(P):
    return (M.T @ np.atleast_2d(P).T).T

def load_curve(path):
    d = json.load(open(path)); pts=[]; radii=[]
    for m in d["markups"]:
        if m["type"] != "Curve": continue
        for cp in m["controlPoints"]:
            x,y,z = cp["position"]; pts.append((y,-z,-x))
        for meas in m.get("measurements", []):
            if meas["name"] == "Radius": radii.extend(meas["controlPointValues"])
    return np.array(pts,float), np.array(radii,float)

def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])

def project(pts, cum, P):
    A=pts[:-1]; B=pts[1:]; AB=B-A; L2=(AB*AB).sum(1); segL=np.sqrt(L2); c=cum[:-1]
    P=np.atleast_2d(P); S=np.empty(len(P)); D=np.empty(len(P))
    for i,p in enumerate(P):
        t=np.clip(((p-A)*AB).sum(1)/np.maximum(L2,1e-12),0,1)
        dd=np.linalg.norm(A+t[:,None]*AB-p,axis=1); k=int(np.argmin(dd))
        S[i]=c[k]+t[k]*segL[k]; D[i]=dd[k]
    return S,D

fp2dir = {re.sub(r"[^A-Za-z0-9]","",d): d for d in sorted(os.listdir(ANAT))}
G={}
for fp,d in fp2dir.items():
    p=os.path.join(ANAT,d,"Centrelines_comb")
    rc,rr = load_curve(os.path.join(p,"Centerline curve - RCCA.mrk.json"))
    b11,_ = load_curve(os.path.join(p,"Centerline curve (11).mrk.json"))
    G[fp]=dict(rcca=rc, rr=rr, s=arclen(rc), b11=b11)
hrc,hrr = load_curve(os.path.join(HOST,"Centerline curve - RCCA.mrk.json"))
hs = arclen(hrc)

print("="*78); print("A. GEOMETRY: where does each holdout RCCA depart from the HOST RCCA?")
print("="*78)
print("host RCCA: n=%d  total length %.2f mm" % (len(hrc), hs[-1]))
HOLD=["topcowmr004","topcowmr008","topcowmr017","topcowmr023"]
seam={}
for fp in sorted(G):
    rc=G[fp]["rcca"]; s=G[fp]["s"]
    n=min(len(rc),len(hrc))
    d=np.linalg.norm(rc[:n]-hrc[:n],axis=1)
    i001=int(np.argmax(d>0.01)) if (d>0.01).any() else n-1
    i1  =int(np.argmax(d>1.0))  if (d>1.0).any()  else n-1
    nr=min(len(G[fp]["rr"]),len(hrr))
    dr=np.abs(G[fp]["rr"][:nr]-hrr[:nr])
    ir=int(np.argmax(dr>0.01)) if (dr>0.01).any() else nr-1
    seam[fp]=dict(s1=float(s[i1]), s001=float(s[i001]), sr=float(s[ir]), tot=float(s[-1]))
    tag = "*HOLDOUT*" if fp in HOLD else ""
    print("  %-12s n=%3d len %7.2f | dep>0.01mm s=%7.2f | dep>1.00mm s=%7.2f | radii differ s=%7.2f %s"
          % (fp, len(rc), s[-1], s[i001], s[i1], s[ir], tag))

print()
print("="*78); print("B. LOGS")
print("="*78)
ES = re.compile(r"EPISODE_START \| ep=(\d+) \|.*?pid=(\d+) \| target=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\).*?seed=(\d+).*?mesh_fp=(\S+)")
ST = re.compile(r"STEP \| ep=(\d+) \| ep_step=(\d+) \|.*?pid=(\d+) \|.*?cur_branch=(.*?) \| local_r=(\S+) \| tol=\S+ \| nearest_named=(\S+).*?tip3d=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\).*?d_tgt=(\S+) \| xt_true=(\S+) \| proj_s=(\S+) \| path_len=(\S+)")
EO = re.compile(r"EPISODE_OUTCOME \| ep=(\d+) \|.*?reason=(\S+) \| is_clean=(\d) \| grader_success=(\d).*?pid=(\d+)")

eps={}; steps=defaultdict(list)
for f in sorted(glob.glob(os.path.join(LOGDIR,"worker_*.log"))):
    for line in open(f, errors="replace"):
        m=ES.search(line)
        if m:
            k=(int(m.group(2)),int(m.group(1)))
            eps[k]=dict(pid=k[0],ep=k[1],target=np.array([float(m.group(3)),float(m.group(4)),float(m.group(5))]),
                        seed=int(m.group(6)),fp=m.group(7)); continue
        m=ST.search(line)
        if m:
            k=(int(m.group(3)),int(m.group(1)))
            steps[k].append((int(m.group(2)),m.group(4).strip(),m.group(6),
                             float(m.group(7)),float(m.group(8)),float(m.group(9)),
                             float(m.group(10)),float(m.group(11)),float(m.group(12)),float(m.group(13))))
            continue
        m=EO.search(line)
        if m:
            k=(int(m.group(5)),int(m.group(1)))
            if k in eps: eps[k]["reason"]=m.group(2); eps[k]["grader"]=int(m.group(4))
print("parsed episodes=%d  reasons=%s  grader_success=%d"
      % (len(eps), dict(Counter(e.get("reason") for e in eps.values())),
         sum(e.get("grader",0) for e in eps.values())))

print()
print("="*78); print("C. EMPIRICAL OFFSET  proj_s - s_RCCA")
print("="*78)
rows=[]
for k,e in eps.items():
    fp=e["fp"]; C=G[fp]
    S=[s for s in steps[k] if s[1].startswith("Centerline curve - RCCA")]
    if not S: continue
    P=t2v(np.array([[s[3],s[4],s[5]] for s in S]))
    sr,dp = project(C["rcca"], C["s"], P)
    for s,a,b in zip(S,sr,dp): rows.append((fp,s[8],a,b))
fps=np.array([r[0] for r in rows]); R=np.array([[r[1],r[2],r[3]] for r in rows])
for thr in (1.0,2.0,3.0):
    g=R[:,2]<thr; o=R[g,0]-R[g,1]
    print("  perp<%.0fmm: n=%5d/%d  median %.3f  mean %.3f  sd %.3f  IQR[%.3f,%.3f]"
          % (thr,g.sum(),len(R),np.median(o),o.mean(),o.std(),np.percentile(o,25),np.percentile(o,75)))
g=(R[:,2]<2.0)&(R[:,1]>20)&(R[:,1]<180)
print("  band perp<2mm & 20<s_rcca<180:")
OFFS={}
for fp in HOLD:
    m=g&(fps==fp); o=R[m,0]-R[m,1]
    OFFS[fp]=float(np.median(o))
    print("    %-12s n=%5d  median %.4f  sd %.4f  p5 %.4f p95 %.4f"
          % (fp,m.sum(),np.median(o),o.std(),np.percentile(o,5),np.percentile(o,95)))
o=R[g,0]-R[g,1]; OFF_ALL=float(np.median(o))
print("    POOLED       n=%5d  median %.4f  sd %.4f  p1 %.4f p99 %.4f"
      % (g.sum(),OFF_ALL,o.std(),np.percentile(o,1),np.percentile(o,99)))

print()
print("  VALIDATION: episode path_len vs (target s_RCCA + per-anatomy offset)")
for k,e in eps.items():
    fp=e["fp"]; C=G[fp]
    ts,td = project(C["rcca"],C["s"], t2v(e["target"]))
    e["tgt_s"]=float(ts[0]); e["tgt_d"]=float(td[0])
    e["path_len"]=steps[k][0][9]
errs=np.array([e["path_len"]-(e["tgt_s"]+OFFS[e["fp"]]) for e in eps.values()])
print("    residual median %.3f  sd %.3f  max|.| %.3f ; max target-off-centerline %.4f mm"
      % (np.median(errs),errs.std(),np.abs(errs).max(),max(e["tgt_d"] for e in eps.values())))

import pickle
pickle.dump(dict(eps=eps,steps=dict(steps),OFFS=OFFS,OFF_ALL=OFF_ALL,seam=seam,
                 G={f:{k2:v2 for k2,v2 in v.items()} for f,v in G.items()}),
            open(os.path.join(ROOT,"monitoring","_attack1_%s.pkl"%STAMP),"wb"))
print("\nsaved pickle")
