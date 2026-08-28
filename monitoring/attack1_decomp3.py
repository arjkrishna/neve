"""ATTACK 1 stage 3: strict route-side OOD classification + host-run comparison."""
import os, re, csv, math, json, pickle, glob, sys
from collections import defaultdict
import numpy as np

ROOT = r"d:/Arjun/workspace/neve"
RUNDIR = os.path.join(ROOT, "saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
D = pickle.load(open(os.path.join(ROOT,"monitoring","_attack1_20260828_045651.pkl"),"rb"))
eps=D["eps"]; steps=D["steps"]; OFF=D["OFF_ALL"]; G=D["G"]
rows=json.load(open(os.path.join(ROOT,"monitoring","_attack1_rows_20260828_045651.json")))
by=(lambda r:(r["pid"],r["ep"]))
SC=133.6; SL=103.5; SCP=SC+OFF; SLP=SL+OFF

def wilson(k,n,z=1.959964):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
def L(tag,ks):
    n=len(ks); k=sum(r["succ"] for r in ks); lo,hi=wilson(k,n)
    return "%-52s %3d/%-3d = %5.1f%%  [%4.1f,%5.1f]"%(tag,k,n,100*k/n if n else float('nan'),lo,hi)

print("="*96)
print("STRICT ROUTE-SIDE: did the WIRE (not just the target) ever enter grafted course?")
print("="*96)
for r in rows: r["past_mm"]=r["maxp"]-SCP
never=[r for r in rows if r["past_mm"]<=0]
ent=[r for r in rows if r["past_mm"]>0]
print(L("(a) wire NEVER passed s_RCCA 133.6 (all-shared route)",never))
print(L("(b) wire entered grafted course",ent))
print()
for thr in (0,1,2,5,10,20):
    kk=[r for r in rows if r["past_mm"]>thr]
    print(L("  wire went >%2d mm past the course seam"%thr,kk))
print()
print("Among the 43 episodes whose TARGET is proximal to the seam, wire overshoot past seam:")
pre=[r for r in rows if r["tgt_s"]<=SC]
pm=np.array([r["past_mm"] for r in pre])
print("  max-past-seam mm: median %.2f  p90 %.2f  max %.2f ; %d of %d exceeded the seam at all"
      %(np.median(pm),np.percentile(pm,90),pm.max(),int((pm>0).sum()),len(pre)))
print("  of those that did exceed it, by how much: %s"
      %np.round(np.sort(pm[pm>0]),2).tolist())

print()
print("="*96)
print("DECOMPOSITION OF THE 90 SUCCESSES")
print("="*96)
succ=[r for r in rows if r["succ"]]
c1=[r for r in succ if r["tgt_s"]<=SC and r["past_mm"]<=0]
c2=[r for r in succ if r["tgt_s"]<=SC and r["past_mm"]>0]
c3=[r for r in succ if r["tgt_s"]>SC]
print("  %2d successes: target AND whole route inside host-identical course"%len(c1))
print("  %2d successes: target proximal to seam but wire briefly crossed it (<=%.1f mm)"
      %(len(c2), max([r["past_mm"] for r in c2]) if c2 else 0))
print("  %2d successes: target in the grafted siphon (genuinely OOD)"%len(c3))
print("  ---- %d total"%(len(c1)+len(c2)+len(c3)))

print()
print("="*96)
print("SAME DECOMPOSITION APPLIED TO THE HOST REAL-PATIENT RUN (20260826_180252)")
print("="*96)
HOSTCL=os.path.join(ROOT,"eve_bench/data/dualdevicenav/Centrelines_comb")
def load(p):
    d=json.load(open(p)); pts=[]
    for m in d["markups"]:
        if m["type"]!="Curve": continue
        for cp in m["controlPoints"]:
            x,y,z=cp["position"]; pts.append((y,-z,-x))
    return np.array(pts,float)
def arc(p): return np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])
hrc=load(os.path.join(HOSTCL,"Centerline curve - RCCA.mrk.json")); hs=arc(hrc)
def rot(zx=(20.,5.)):
    rz=-zx[0]*np.pi/180; rx=-zx[1]*np.pi/180
    Rz=np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    return Rz@Rx
M=rot()
def proj1(p,poly,cum):
    A=poly[:-1];B=poly[1:];AB=B-A;L2=(AB*AB).sum(1)
    t=np.clip(((p-A)*AB).sum(1)/np.maximum(L2,1e-12),0,1)
    dd=np.linalg.norm(A+t[:,None]*AB-p,axis=1);k=int(np.argmin(dd))
    return cum[k]+t[k]*np.sqrt(L2[k]),dd[k]
ES=re.compile(r"EPISODE_START \| ep=(\d+) \|.*?pid=(\d+) \| target=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\).*?seed=(\d+)")
PJ=re.compile(r"pid=(\d+) \|.*?proj_s=([-0-9.]+) \| path_len=([-0-9.]+)")
hlog=os.path.join(RUNDIR,"logs","20260826_180252")
hs_eps={}; hmax=defaultdict(float)
cur={}
for f in sorted(glob.glob(os.path.join(hlog,"worker_*.log"))):
    for line in open(f,errors="replace"):
        m=ES.search(line)
        if m:
            k=(int(m.group(2)),int(m.group(1)))
            cur[int(m.group(2))]=k
            hs_eps[k]=dict(seed=int(m.group(6)),
                           target=np.array([float(m.group(3)),float(m.group(4)),float(m.group(5))]))
            continue
        if " STEP | " in line:
            m=PJ.search(line)
            if m:
                k=cur.get(int(m.group(1)))
                if k: hmax[k]=max(hmax[k],float(m.group(2))); hs_eps[k]["path_len"]=float(m.group(3))
hsec={}
import csv as _csv
hcsv=os.path.join(RUNDIR,"episodes.csv")
hj=[json.loads(l) for l in open(os.path.join(RUNDIR,"episodes_official_20260826_180252.jsonl"))]
hsucc={x["seed"]:int(x["success"]) for x in hj}
print("host run: %d episodes in jsonl, %d successes (%.1f%%)"%(len(hj),sum(hsucc.values()),100*sum(hsucc.values())/len(hj)))
hrows=[]
for k,e in hs_eps.items():
    tv=(M.T@e["target"].T).T
    s,d=proj1(tv,hrc,hs)
    if e["seed"] not in hsucc: continue
    hrows.append(dict(seed=e["seed"],tgt_s=float(s),tgt_d=float(d),
                      path_len=e.get("path_len",float('nan')),maxp=hmax[k],succ=hsucc[e["seed"]]))
print("  matched %d ; target off-centerline max %.3f mm ; offset path_len-tgt_s median %.3f"
      %(len(hrows),max(r["tgt_d"] for r in hrows),
        float(np.median([r["path_len"]-r["tgt_s"] for r in hrows]))))
print(L("  HOST all",hrows))
print(L("  HOST target s_RCCA <= 133.6 (same proximal band)",[r for r in hrows if r["tgt_s"]<=SC]))
print(L("  HOST target s_RCCA >  133.6",[r for r in hrows if r["tgt_s"]>SC]))
print(L("  HOST target s_RCCA <= 103.5 (pre-lumen-seam band)",[r for r in hrows if r["tgt_s"]<=SL]))
print(L("  HOST target 103.5 < s <= 133.6 (band the graft WIDENED)",
        [r for r in hrows if SL<r["tgt_s"]<=SC]))
b=[r for r in rows if SL<r["tgt_s"]<=SC]
print(L("  COHORT same band 103.5 < s <= 133.6",b))
print()
print("host target s_RCCA range %.1f..%.1f (n=%d)"%(min(r["tgt_s"] for r in hrows),
      max(r["tgt_s"] for r in hrows),len(hrows)))
