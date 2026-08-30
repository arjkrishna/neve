import pickle, numpy as np, collections, math, json
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
GEO=pickle.load(open("_t4_geom.pkl","rb"))
AGG={r["anatomy"]:r for r in json.load(open("attack2_geom.json"))["rows"]}
SB=json.load(open("attack2_siphonband.json"))
def key(a): return "topcow_"+a.replace("mr_mr","mr_")
mh={r["seed"]:r for r in H}
gt=[r for r in T if r["grafted"]]
print("heuristic pull_frac: max %.4f over 220; teacher pull_frac med %.2f"%(
   max(r["pull_frac"] for r in H), np.median([r["pull_frac"] for r in gt])))
suc=[r for r in gt if r["succ"]]; fai=[r for r in gt if not r["succ"]]
print("teacher grafted successes: push %.2f pull %.2f abs_ins %.0f net %.0f fold_max med %d"%(
  np.median([r['push_frac'] for r in suc]),np.median([r['pull_frac'] for r in suc]),
  np.median([r['abs_ins'] for r in suc]),np.median([r['net_ins'] for r in suc]),np.median([r['fold_max'] for r in suc])))
print("teacher grafted failures : push %.2f pull %.2f abs_ins %.0f net %.0f fold_max med %d"%(
  np.median([r['push_frac'] for r in fai]),np.median([r['pull_frac'] for r in fai]),
  np.median([r['abs_ins'] for r in fai]),np.median([r['net_ins'] for r in fai]),np.median([r['fold_max'] for r in fai])))
print()
print("=== is fold>=20 predicted by geometry? episode-level ===")
print("fold onset station vs local Rc5:")
on=[(r,r["s_at_fold20"]) for r in gt if r["s_at_fold20"] is not None]
def at(a,s,f):
    g=GEO[key(a)]; return float(np.interp(s,g["q"],g[f]))
v=[at(r["anat"],s,"Rc5") for r,s in on]
print("  n=%d  Rc5 at onset med %.1f ; onset s<=130: %d/%d (anatomy-invariant zone)"%(
  len(on),np.median(v),sum(1 for _,s in on if s<=130),len(on)))
print("  onset hist 10mm:",sorted(collections.Counter(int(s//10)*10 for _,s in on).items()))
print()
print("=== per-anatomy fold rate vs anatomy geometry ===")
by=collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
rows=[]
for a in sorted(by):
    g=by[a]; n=len(g); k=sum(1 for r in g if r["fold_max"]>=20)
    nm="topcow_"+a.replace("mr_mr","mr_"); sh=nm
    ag=AGG[nm]; sb=SB[nm]
    rows.append((a,n,k,k/n,sum(r["succ"] for r in g)/n, ag["rcca_len"], sb["Rcmin"], sb["bendmax"],
                 sb["tort"], sb["clrmin"], ag["r_med_graft"], ag["r_min_graft"]))
print("anat      n f>=20 foldrate rate    L    Rcmin bendmax tort  clrmin rmed rmin")
for t in rows:
    print(f"{t[0]} {t[1]:3d} {t[2]:4d} {t[3]:8.2f} {t[4]:5.2f} {t[5]:6.1f} {t[6]:6.2f} {t[7]:6.1f} {t[8]:5.2f} {t[9]:6.2f} {t[10]:4.2f} {t[11]:4.2f}")
def spear(x,y):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        i=0
        while i<len(o):
            j=i
            while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
            for m in range(i,j+1): r[o[m]]=(i+j)/2+1
            i=j+1
        return r
    a,b=rk(x),rk(y); n=len(x)
    ma,mb=np.mean(a),np.mean(b)
    return float(np.sum((np.array(a)-ma)*(np.array(b)-mb))/math.sqrt(np.sum((np.array(a)-ma)**2)*np.sum((np.array(b)-mb)**2)))
lab=["L","Rcmin","bendmax","tort","clrmin","rmed_graft","rmin_graft"]
print("\nSpearman(n=22) of per-anatomy FOLD RATE vs geometry (and of SUCCESS RATE vs geometry):")
for j,l in enumerate(lab):
    x=[t[5+j] for t in rows]
    print("  %-11s foldrate rho=%+.3f   succrate rho=%+.3f"%(l,spear(x,[t[3] for t in rows]),spear(x,[t[4] for t in rows])))
print("  foldrate vs succrate rho=%+.3f"%spear([t[3] for t in rows],[t[4] for t in rows]))
# n per anatomy vs rate (sanity)
print("  n_episodes vs succrate rho=%+.3f"%spear([t[1] for t in rows],[t[4] for t in rows]))
