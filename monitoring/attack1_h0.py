"""ATTACK 1 stage 4: same seam decomposition for the H0 heuristic run (training-loop eval)."""
import os, re, glob, json, math
from collections import defaultdict, Counter
import numpy as np

ROOT = r"d:/Arjun/workspace/neve"
H = os.path.join(ROOT,"saved/eve_paper/neurovascular/full/mesh_ben/2026-08-28_034549_rcca_topbrain_smoke")
ANAT = os.path.join(ROOT,"topbrain_data/anatomies")
HOLD = {"topcowmr004","topcowmr008","topcowmr017","topcowmr023"}
OFF = 33.314; SC = 133.6; SL = 103.5

def load(p):
    d=json.load(open(p)); pts=[]
    for m in d["markups"]:
        if m["type"]!="Curve": continue
        for cp in m["controlPoints"]:
            x,y,z=cp["position"]; pts.append((y,-z,-x))
    return np.array(pts,float)
def arc(p): return np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])
def rot(zx=(20.,5.)):
    rz=-zx[0]*np.pi/180; rx=-zx[1]*np.pi/180
    Rz=np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    return Rz@Rx
M=rot()
G={}
for fp in HOLD:
    d="topcow_mr_%s"%fp[-3:]
    c=load(os.path.join(ANAT,d,"Centrelines_comb","Centerline curve - RCCA.mrk.json"))
    G[fp]=(c,arc(c))
def proj1(p,poly,cum):
    A=poly[:-1];B=poly[1:];AB=B-A;L2=(AB*AB).sum(1)
    t=np.clip(((p-A)*AB).sum(1)/np.maximum(L2,1e-12),0,1)
    dd=np.linalg.norm(A+t[:,None]*AB-p,axis=1);k=int(np.argmin(dd))
    return cum[k]+t[k]*float(np.sqrt(L2[k])),float(dd[k])

# outcomes from snapshot filenames
out={}
for kind in ("success","max_steps"):
    for f in os.listdir(os.path.join(H,"diagnostics/snapshots/eval/RCCA",kind)):
        m=re.match(r"ep(\d+)_pid(\d+)_step(\d+)_R([-+0-9.]+)_",f)
        out[(int(m.group(2)),int(m.group(1)))]=(1 if kind=="success" else 0, int(m.group(3)))
print("eval snapshots: %d (success %d)"%(len(out),sum(v[0] for v in out.values())))

ES=re.compile(r"^(?P<ts>\S+ \S+) - EPISODE_START \| ep=(?P<ep>\d+) \|.*?pid=(?P<pid>\d+) \| target=\((?P<x>[-0-9.]+),(?P<y>[-0-9.]+),(?P<z>[-0-9.]+)\).*?mesh_fp=(?P<fp>\S+)")
PJ=re.compile(r"pid=(\d+) \|.*?proj_s=([-0-9.]+) \| path_len=([-0-9.]+)")
eps={}; mx=defaultdict(float); cur={}
for f in sorted(glob.glob(os.path.join(H,"diagnostics/logs_subprocesses/worker_*.log"))):
    for line in open(f,errors="replace"):
        m=ES.match(line)
        if m:
            ts=m.group("ts")
            if not ("2026-08-28 03:51:29" <= ts <= "2026-08-28 04:15:14"):
                cur[int(m.group("pid"))]=None
                continue
            k=(int(m.group("pid")),int(m.group("ep"))); cur[k[0]]=k
            eps[k]=dict(fp=m.group("fp"),target=np.array([float(m.group("x")),float(m.group("y")),float(m.group("z"))]))
            continue
        if " STEP | " in line:
            m=PJ.search(line)
            if m:
                k=cur.get(int(m.group(1)))
                if k is not None and k in eps:
                    mx[k]=max(mx[k],float(m.group(2))); eps[k]["path_len"]=float(m.group(3))
ev={k:v for k,v in eps.items() if v["fp"] in HOLD}
print("EPISODE_STARTs total %d ; on holdout anatomies %d ; anatomy counts %s"
      %(len(eps),len(ev),dict(Counter(v["fp"] for v in ev.values()))))
matched={k:v for k,v in ev.items() if k in out}
print("joined to a snapshot outcome: %d"%len(matched))

rows=[]
for k,v in matched.items():
    c,cum=G[v["fp"]]
    s,d=proj1((M.T@v["target"].T).T,c,cum)
    rows.append(dict(fp=v["fp"],tgt_s=s,tgt_d=d,path_len=v.get("path_len",float('nan')),
                     maxp=mx[k],succ=out[k][0],steps=out[k][1]))
print("target off-centerline max %.3f mm ; path_len-(tgt_s+off) median %.3f max|.| %.3f"
      %(max(r["tgt_d"] for r in rows),
        float(np.median([r["path_len"]-(r["tgt_s"]+OFF) for r in rows])),
        float(np.abs([r["path_len"]-(r["tgt_s"]+OFF) for r in rows]).max())))

def wilson(k,n,z=1.959964):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
def L(tag,ks):
    n=len(ks);k=sum(r["succ"] for r in ks);lo,hi=wilson(k,n)
    return "%-50s %3d/%-3d = %5.1f%%  [%4.1f,%5.1f]"%(tag,k,n,100*k/n if n else float('nan'),lo,hi)
print()
print(L("H0 ALL",rows))
print(L("H0 target <= seam 133.6 (host-identical course)",[r for r in rows if r["tgt_s"]<=SC]))
print(L("H0 target >  seam 133.6 (grafted, OOD)",[r for r in rows if r["tgt_s"]>SC]))
print(L("H0 target <= 103.5 (pre lumen seam)",[r for r in rows if r["tgt_s"]<=SL]))
print(L("H0 103.5 < target <= 133.6 (widened band)",[r for r in rows if SL<r["tgt_s"]<=SC]))
print()
for fp in sorted(HOLD):
    kk=[r for r in rows if r["fp"]==fp]
    print(L("  H0 %s"%fp,kk), " | OOD "+L("",[r for r in kk if r["tgt_s"]>SC]).strip())
print()
print("H0 target s_RCCA distribution: min %.1f med %.1f max %.1f"
      %(min(r["tgt_s"] for r in rows),float(np.median([r["tgt_s"] for r in rows])),max(r["tgt_s"] for r in rows)))
json.dump(rows,open(os.path.join(ROOT,"monitoring","_attack1_h0rows.json"),"w"),default=float)
