"""ATTACK 1 stage 2: per-episode seam decomposition + Wilson CIs."""
import os, re, csv, math, pickle, sys
from collections import defaultdict, Counter
import numpy as np

ROOT = r"d:/Arjun/workspace/neve"
RUNDIR = os.path.join(ROOT, "saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292")
STAMP = sys.argv[1] if len(sys.argv)>1 else "20260828_045651"
D = pickle.load(open(os.path.join(ROOT,"monitoring","_attack1_%s.pkl"%STAMP),"rb"))
eps=D["eps"]; steps=D["steps"]; OFFS=D["OFFS"]; OFF=D["OFF_ALL"]

# seams, in RCCA arclength (mm) and in proj_s / path_len (mm)
SEAM_COURSE_S = 133.6      # first >1 mm course departure from host (measured 133.50-133.75 for the 4)
SEAM_LUMEN_S  = 103.5      # first >0.25 mm declared-radius departure (pipeline smoothstep)
SEAM_COURSE_P = SEAM_COURSE_S + OFF
SEAM_LUMEN_P  = SEAM_LUMEN_S  + OFF
print("offset proj_s - s_RCCA = %.3f mm (sd 0.042)" % OFF)
print("course seam  s_RCCA %.1f mm  ->  proj_s/path_len %.2f mm" % (SEAM_COURSE_S, SEAM_COURSE_P))
print("lumen  seam  s_RCCA %.1f mm  ->  proj_s/path_len %.2f mm" % (SEAM_LUMEN_S,  SEAM_LUMEN_P))

# join to episodes.csv on seed (section labels)
sec={}
with open(os.path.join(RUNDIR,"episodes.csv")) as f:
    for r in csv.DictReader(f):
        sec[int(r["seed"])] = (r["section"], float(r["path_len_mm"]), int(r["success"]), r["anatomy"])
print("episodes.csv rows=%d ; log episodes=%d ; seeds matched=%d"
      % (len(sec), len(eps), sum(1 for e in eps.values() if e["seed"] in sec)))

rows=[]
for k,e in sorted(eps.items()):
    S=steps[k]
    ps=np.array([s[8] for s in S])
    maxp=float(ps.max()); term=float(ps[-1])
    sc = sec.get(e["seed"])
    assert sc is not None, e["seed"]
    succ = sc[2]   # AUTHORITATIVE: episodes.csv / official jsonl (90/98)
    rows.append(dict(pid=k[0],ep=k[1],fp=e["fp"],seed=e["seed"],succ=succ,
                     reason=e.get("reason","?"),tgt_s=e["tgt_s"],path_len=e["path_len"],
                     maxp=maxp,term=term,nsteps=len(S),section=sc[0],csv_len=sc[1],csv_succ=sc[2]))
assert len(rows)==98

def wilson(k,n,z=1.959964):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
def line(tag,ks):
    n=len(ks); k=sum(r["succ"] for r in ks)
    lo,hi=wilson(k,n)
    return "%-46s %3d/%-3d = %5.1f%%  [%4.1f, %5.1f]"%(tag,k,n,100*k/n if n else float("nan"),lo,hi)

print("\n"+"="*90)
print("1. SECTION vs SEAM  (section label from episodes.csv; path_len is target's, from insertion)")
print("="*90)
for s in ["CCA","ICA-mid","siphon","?"]:
    ks=[r for r in rows if r["section"]==s]
    if not ks: continue
    pl=np.array([r["path_len"] for r in ks]); ts=np.array([r["tgt_s"] for r in ks])
    print("%-9s n=%2d  path_len %6.1f..%6.1f  target s_RCCA %6.1f..%6.1f  | targets past COURSE seam: %d  past LUMEN seam: %d"
          %(s,len(ks),pl.min(),pl.max(),ts.min(),ts.max(),
            int((ts>SEAM_COURSE_S).sum()),int((ts>SEAM_LUMEN_S).sum())))
    print("   "+line("   success",ks))

print("\n"+"="*90)
print("2. TARGET-SIDE CLASSIFICATION: does the TARGET sit in host-identical geometry?")
print("="*90)
a=[r for r in rows if r["tgt_s"]<=SEAM_COURSE_S]; b=[r for r in rows if r["tgt_s"]>SEAM_COURSE_S]
print(line("(a) target in HOST-IDENTICAL course (s<=133.6)",a))
print(line("(b) target in GRAFTED course      (s> 133.6)",b))
print()
a2=[r for r in rows if r["tgt_s"]<=SEAM_LUMEN_S]; b2=[r for r in rows if r["tgt_s"]>SEAM_LUMEN_S]
print(line("(a) target in host-identical LUMEN (s<=103.5)",a2))
print(line("(b) target past LUMEN divergence   (s> 103.5)",b2))

print("\n"+"="*90)
print("3. ROUTE-SIDE: how much of the ACTUALLY TRAVERSED route lay past the seam?")
print("="*90)
for r in rows:
    r["shared_mm"]=min(r["maxp"],SEAM_COURSE_P)-OFF if r["maxp"]>OFF else 0.0
    r["graft_mm"]=max(0.0,r["maxp"]-SEAM_COURSE_P)
    r["frac_graft"]=r["graft_mm"]/max(1e-9,r["maxp"]-OFF)
sh=np.array([r["shared_mm"] for r in rows]); gf=np.array([r["graft_mm"] for r in rows])
print("max proj_s reached: min %.1f  median %.1f  max %.1f"%(min(r["maxp"] for r in rows),
      np.median([r["maxp"] for r in rows]),max(r["maxp"] for r in rows)))
print("RCCA arclength actually reached (max proj_s - offset): median %.1f mm"%np.median([r["maxp"]-OFF for r in rows]))
print("mm of RCCA route inside host-identical course: median %.1f  (of %.1f median reached)"
      %(np.median(sh),np.median([r["maxp"]-OFF for r in rows])))
print("mm of RCCA route inside GRAFTED course:        median %.1f  mean %.1f  max %.1f"%(np.median(gf),gf.mean(),gf.max()))
print("episodes whose wire NEVER passed the course seam: %d/98"%sum(1 for r in rows if r["maxp"]<=SEAM_COURSE_P))
print("episodes whose wire NEVER passed the lumen seam : %d/98"%sum(1 for r in rows if r["maxp"]<=SEAM_LUMEN_P))
tot_shared=sum(min(r["maxp"],SEAM_COURSE_P)-OFF for r in rows)
tot_graft=sum(max(0.0,r["maxp"]-SEAM_COURSE_P) for r in rows)
print("COHORT TOTAL traversed RCCA arclength: %.0f mm shared + %.0f mm grafted = %.1f%% grafted"
      %(tot_shared,tot_graft,100*tot_graft/(tot_shared+tot_graft)))

print("\n"+"="*90)
print("4. BOTTOM LINE (course seam 133.6 mm)")
print("="*90)
ood=[r for r in rows if r["tgt_s"]>SEAM_COURSE_S]
ind=[r for r in rows if r["tgt_s"]<=SEAM_COURSE_S]
print(line("ALL",rows))
print(line("IN-DISTRIBUTION targets (host's own trunk)",ind))
print(line("OUT-OF-DISTRIBUTION targets (grafted siphon)",ood))
print()
for fp in sorted(set(r["fp"] for r in rows)):
    kk=[r for r in rows if r["fp"]==fp]
    print(line("  %s  ALL"%fp,kk))
    print(line("  %s  OOD only"%fp,[r for r in kk if r["tgt_s"]>SEAM_COURSE_S]))

print("\n"+"="*90)
print("5. FAILURES: where did the 8 failures arrest?")
print("="*90)
for r in sorted(rows,key=lambda r:-r["succ"]):
    if r["succ"]: continue
    print("  %-12s seed=%d section=%-8s tgt_s=%6.1f path_len=%6.1f maxproj=%6.1f "
          "(=s_RCCA %6.1f) steps=%d reason=%s"
          %(r["fp"],r["seed"],r["section"],r["tgt_s"],r["path_len"],r["maxp"],r["maxp"]-OFF,r["nsteps"],r["reason"]))

print("\n"+"="*90)
print("6. TARGET s_RCCA HISTOGRAM (10 mm bins) with success")
print("="*90)
bins=np.arange(30,270,10)
for lo in bins:
    kk=[r for r in rows if lo<=r["tgt_s"]<lo+10]
    if not kk: continue
    k=sum(r["succ"] for r in kk)
    mark=" <== course seam" if lo<=SEAM_COURSE_S<lo+10 else (" <== lumen seam" if lo<=SEAM_LUMEN_S<lo+10 else "")
    print("  s_RCCA [%3d,%3d)  n=%2d  succ=%2d  %5.1f%%%s"%(lo,lo+10,len(kk),k,100*k/len(kk),mark))

import json
json.dump(rows,open(os.path.join(ROOT,"monitoring","_attack1_rows_%s.json"%STAMP),"w"),indent=0,default=float)
