import pickle, numpy as np, collections
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
GEO=pickle.load(open("_t4_geom.pkl","rb"))
def key(a): return "topcow_"+a.replace("mr_mr","mr_")
def at(a,s,f):
    g=GEO[key(a)]; return float(np.interp(s,g["q"],g[f]))
def pct(a,s):
    g=GEO[key(a)]; m=(g["q"]>=20)&(g["q"]<=min(200,g["L"]-6))
    return 100*float(np.nanmean(np.array(g["Rc5"])[m]<at(a,s,"Rc5")))
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]; fai=[r for r in gt if not r["succ"]]
INV=[r for r in fai if r["max_s"]<=130]; DIS=[r for r in fai if r["max_s"]>130]
print("PROXIMAL-INVARIANT arrests (s<=130, geometry identical across all 22): %d/%d = %.1f%%"%(len(INV),len(fai),100*len(INV)/len(fai)))
print("DISTAL arrests (s>130, anatomy-specific graft):                        %d/%d = %.1f%%"%(len(DIS),len(fai),100*len(DIS)/len(fai)))
print()
print("Rc5 percentile at arrest (0=sharpest in own anatomy):")
for g,l in ((INV,"proximal-invariant"),(DIS,"distal")):
    v=[pct(r["anat"],r["max_s"]) for r in g]
    print("  %-20s n=%2d med %5.1f  <=20th pct: %d/%d"%(l,len(g),np.median(v),sum(1 for x in v if x<=20),len(v)))
print()
print("per anatomy: failures split proximal-invariant / distal (grafted)")
by=collections.defaultdict(lambda:[0,0,0])
for r in gt:
    by[r["anat"]][0]+=1
    if not r["succ"]:
        by[r["anat"]][1 if r["max_s"]<=130 else 2]+=1
print("anat      n  fail_prox  fail_distal   rate")
for a in sorted(by):
    n,p,d=by[a]; print(f"{a} {n:3d}  {p:6d}  {d:8d}    {100*(n-p-d)/n:5.1f}")
print()
print("=== HEURISTIC comparison on the SAME seeds ===")
print("teacher fails / heuristic succeeds: n=%d"%sum(1 for r in fai if mh[r['seed']]['succ']))
print("teacher fails / heuristic fails   : n=%d"%sum(1 for r in fai if not mh[r['seed']]['succ']))
tw=[r for r in fai if mh[r["seed"]]["succ"]]
print("\nWhere heuristic wins (n=%d): teacher arrest med %.1f ; heuristic reached target."%(len(tw),np.median([r['max_s'] for r in tw])))
print("  teacher fold_max med %d (>=20 in %d/%d) ; heuristic fold_max med %d (>=20 in %d/%d)"%(
  np.median([r['fold_max'] for r in tw]),sum(1 for r in tw if r['fold_max']>=20),len(tw),
  np.median([mh[r['seed']]['fold_max'] for r in tw]),sum(1 for r in tw if mh[r['seed']]['fold_max']>=20),len(tw)))
print("  teacher slack_max med %.1f ; heuristic slack_max med %.1f"%(
  np.median([r['slack_max'] for r in tw]),np.median([mh[r['seed']]['slack_max'] for r in tw])))
print("  duty cycle push/pull/hold  teacher %.2f/%.2f/%.2f ; heuristic %.2f/%.2f/%.2f"%(
  np.median([r['push_frac'] for r in tw]),np.median([r['pull_frac'] for r in tw]),np.median([r['hold_frac'] for r in tw]),
  np.median([mh[r['seed']]['push_frac'] for r in tw]),np.median([mh[r['seed']]['pull_frac'] for r in tw]),np.median([mh[r['seed']]['hold_frac'] for r in tw])))
print("  mean |d_ins| per step      teacher %.2f ; heuristic %.2f"%(
  np.median([r['allv'] for r in tw]),np.median([mh[r['seed']]['allv'] for r in tw])))
print("  steps                      teacher %d ; heuristic %d"%(
  np.median([r['n'] for r in tw]),np.median([mh[r['seed']]['n'] for r in tw])))
print("  net inserted gw (mm)       teacher %.0f ; heuristic %.0f ; abs travel T %.0f H %.0f"%(
  np.median([r['net_ins'] for r in tw]),np.median([mh[r['seed']]['net_ins'] for r in tw]),
  np.median([r['abs_ins'] for r in tw]),np.median([mh[r['seed']]['abs_ins'] for r in tw])))
print("\nAll grafted, teacher vs heuristic global instruments (median):")
for lab,f in (("fold_max","fold_max"),("slack_max","slack_max"),("push_frac","push_frac"),
              ("pull_frac","pull_frac"),("hold_frac","hold_frac"),("allv","allv"),("n","n"),("abs_ins","abs_ins")):
    print("  %-10s T %8.2f   H %8.2f"%(lab,np.median([r[f] for r in gt]),np.median([mh[r['seed']][f] for r in gt])))
