"""ATTACK 1 addendum: the HOST run is a DIFFERENT task; band-matched comparison."""
import os, re, glob, json, math
from collections import defaultdict
import numpy as np
ROOT=r"d:/Arjun/workspace/neve"
RUNDIR=os.path.join(ROOT,"saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
HOSTCL=os.path.join(ROOT,"eve_bench/data/dualdevicenav/Centrelines_comb")
V=json.load(open(os.path.join(ROOT,"monitoring","_attack1_rows_20260828_045651.json")))
H0=json.load(open(os.path.join(ROOT,"monitoring","_attack1_h0rows.json")))
SC=133.6; SL=103.5

def load(p):
    d=json.load(open(p)); pts=[]; rr=[]
    for m in d["markups"]:
        if m["type"]!="Curve": continue
        for cp in m["controlPoints"]:
            x,y,z=cp["position"]; pts.append((y,-z,-x))
        for me in m.get("measurements",[]):
            if me["name"]=="Radius": rr.extend(me["controlPointValues"])
    return np.array(pts,float),np.array(rr,float)
def arc(p): return np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])
def rot(zx=(20.,5.)):
    rz=-zx[0]*np.pi/180; rx=-zx[1]*np.pi/180
    Rz=np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    return Rz@Rx
M=rot()
hrc,hrr=load(os.path.join(HOSTCL,"Centerline curve - RCCA.mrk.json")); hs=arc(hrc)
def proj1(p,poly,cum):
    A=poly[:-1];B=poly[1:];AB=B-A;L2=(AB*AB).sum(1)
    t=np.clip(((p-A)*AB).sum(1)/np.maximum(L2,1e-12),0,1)
    dd=np.linalg.norm(A+t[:,None]*AB-p,axis=1);k=int(np.argmin(dd))
    return cum[k]+t[k]*float(np.sqrt(L2[k])),float(dd[k])

ES=re.compile(r"EPISODE_START \| ep=(\d+) \|.*?pid=(\d+) \| target=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\).*?seed=(\d+)")
PJ=re.compile(r"pid=(\d+) \|.*?proj_s=([-0-9.]+) \| path_len=([-0-9.]+)")
FS=re.compile(r"ep_step=1 \|.*?pid=(\d+) \|")
eps={}; mx=defaultdict(float); cur={}
for f in sorted(glob.glob(os.path.join(RUNDIR,"logs","20260826_180252","worker_*.log"))):
    for line in open(f,errors="replace"):
        m=ES.search(line)
        if m:
            k=(int(m.group(2)),int(m.group(1))); cur[k[0]]=k
            eps[k]=dict(seed=int(m.group(6)),target=np.array([float(m.group(3)),float(m.group(4)),float(m.group(5))]))
            continue
        if " STEP | " in line:
            m=PJ.search(line)
            if m:
                k=cur.get(int(m.group(1)))
                if k: mx[k]=max(mx[k],float(m.group(2))); eps[k]["path_len"]=float(m.group(3))
hj=[json.loads(l) for l in open(os.path.join(RUNDIR,"episodes_official_20260826_180252.jsonl"))]
hsucc={x["seed"]:int(x["success"]) for x in hj}
Hr=[]
for k,e in eps.items():
    s,d=proj1((M.T@e["target"].T).T,hrc,hs)
    Hr.append(dict(seed=e["seed"],tgt_s=s,tgt_d=d,path_len=e["path_len"],maxp=mx[k],
                   succ=hsucc.get(e["seed"],None)))
Hr=[r for r in Hr if r["succ"] is not None]
ins=np.array([r["tgt_s"]-r["path_len"] for r in Hr])
print("HOST RUN 20260826_180252")
print("  n=%d  successes=%d (%.1f%%)"%(len(Hr),sum(r['succ'] for r in Hr),100*np.mean([r['succ'] for r in Hr])))
print("  insertion arclength implied by (tgt_s - path_len): median %.3f mm  sd %.3f  min %.3f max %.3f"
      %(np.median(ins),ins.std(),ins.min(),ins.max()))
print("  -> the HOST run inserts INSIDE the RCCA at s=%.1f mm; the cohort run inserts in branch (11), "
      "%.2f mm of route BEFORE the RCCA ostium."%(np.median(ins),33.314))
print("  host target s_RCCA: min %.1f  median %.1f  max %.1f"
      %(min(r['tgt_s'] for r in Hr),np.median([r['tgt_s'] for r in Hr]),max(r['tgt_s'] for r in Hr)))
print("  cohort target s_RCCA: min %.1f  median %.1f  max %.1f"
      %(min(r['tgt_s'] for r in V),np.median([r['tgt_s'] for r in V]),max(r['tgt_s'] for r in V)))

def wil(k,n,z=1.959964):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
def P(tag,ks):
    n=len(ks); k=sum(r["succ"] for r in ks); lo,hi=wil(k,n)
    print("%-50s %3d/%-3d = %5.1f%%  [%4.1f,%5.1f]"%(tag,k,n,100*k/n if n else float('nan'),lo,hi))

print()
print("BAND-MATCHED (target RCCA arclength bands, all three runs)")
bands=[(0,103.5,"pre-lumen-seam, host-identical"),
       (103.5,133.6,"band the graft WIDENED (host r 1.4-1.6, graft r 2.3-2.9)"),
       (133.6,180,"early graft"),(180,240,"deep graft / siphon")]
for lo,hi,nm in bands:
    print("  s_RCCA (%5.1f,%5.1f]  %s"%(lo,hi,nm))
    P("      v1bp cohort",[r for r in V if lo<r["tgt_s"]<=hi])
    P("      H0   cohort",[r for r in H0 if lo<r["tgt_s"]<=hi])
    P("      v1bp HOST  ",[r for r in Hr if lo<r["tgt_s"]<=hi])
print()
print("HOST radii vs cohort radii along the shared course:")
for s in (60,80,100,110,120,130,133):
    print("   s=%3d mm  host r=%.2f mm"%(s,np.interp(s,hs,hrr)))
