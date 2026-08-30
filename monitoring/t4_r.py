import pickle, numpy as np, collections, math
GEO=pickle.load(open("_t4_geom.pkl","rb"))
G2=pickle.load(open("_t2_geo.pkl","rb"))["G"]
D=pickle.load(open("_t4_rows2.pkl","rb")); T,H=D["T"],D["H"]
mh={r["seed"]:r for r in H}
g=GEO["topcow_mr_001"]; q=g["q"]
m=(q>=95)&(q<=133)
Rc=np.array(g["Rc5"])[m]; qq=q[m]; B=np.array(g["b10"])[m]
i=int(np.nanargmin(Rc))
print("shared proximal segment (identical in all 22), 95-133 mm:")
print("  Rc5 minimum %.2f mm at s=%.1f ; bend10 there %.1f deg"%(Rc[i],qq[i],B[i]))
j=int(np.nanargmax(B)); print("  bend10 maximum %.1f deg at s=%.1f"%(B[j],qq[j]))
mm=(q>=2)&(q<=40); Rc2=np.array(g["Rc5"])[mm]; q2=q[mm]; B2=np.array(g["b10"])[mm]
k=int(np.nanargmin(Rc2)); print("  ostium zone 2-40: Rc5 min %.2f at s=%.1f, bend10 %.1f (bend10 max %.1f at s=%.1f)"%(
   Rc2[k],q2[k],B2[k],np.nanmax(B2),q2[int(np.nanargmax(B2))]))
# declared radius profile in shared zone across cohort
ks=[k for k in sorted(GEO) if k.split("_")[-1] in "001 002 003 004 005 006 007 008 010 011 012 016 017 018 020 021 022 023 024 025 026 027".split()]
print("  declared radius at s=117.5 across 22: %.2f..%.2f (min-radius station in 100-133 for each anat)"%(
   min(np.interp(117.5,GEO[k]["q"],GEO[k]["r"]) for k in ks),
   max(np.interp(117.5,GEO[k]["q"],GEO[k]["r"]) for k in ks)))
print()
gt=[r for r in T if r["grafted"]]
# does an anatomy's radius at 110-130 predict its fold rate?
by=collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
def key(a): return "topcow_"+a.replace("mr_mr","mr_")
print("anat     foldrate  rate  r_decl@117.5  r_decl@125  clear@117.5")
X=[];Y=[];Z=[]
for a in sorted(by):
    gg=by[a]; fr=np.mean([r["fold_max"]>=20 for r in gg]); sr=np.mean([r["succ"] for r in gg])
    r1=float(np.interp(117.5,GEO[key(a)]["q"],GEO[key(a)]["r"]))
    r2=float(np.interp(125,GEO[key(a)]["q"],GEO[key(a)]["r"]))
    v=G2["topcow"+a.replace("mr_mr","mr")]; c1=float(np.interp(117.5,np.array(v["g"]),np.array(v["d"])))
    print(f"{a} {fr:8.2f} {sr:5.2f}  {r1:11.2f} {r2:10.2f} {c1:11.2f}")
    X.append(r1);Y.append(fr);Z.append(sr)
def spear(x,yy):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0.]*len(v); i=0
        while i<len(o):
            j=i
            while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
            for mq in range(i,j+1): r[o[mq]]=(i+j)/2+1
            i=j+1
        return np.array(r)
    return float(np.corrcoef(rk(x),rk(yy))[0,1])
print("Spearman r_decl@117.5 vs foldrate %+.3f ; vs succrate %+.3f (n=22)"%(spear(X,Y),spear(X,Z)))
print()
# per-anatomy teacher-vs-heuristic discordance heterogeneity
print("=== 'hard for THIS policy' test: paired discordance per anatomy ===")
print("anat      b(T only)  c(H only)  concord_succ  concord_fail")
tb=tc=0
for a in sorted(by):
    gg=by[a]
    b=sum(1 for r in gg if r["succ"] and not mh[r["seed"]]["succ"])
    c=sum(1 for r in gg if not r["succ"] and mh[r["seed"]]["succ"])
    ss=sum(1 for r in gg if r["succ"] and mh[r["seed"]]["succ"])
    ff=sum(1 for r in gg if not r["succ"] and not mh[r["seed"]]["succ"])
    tb+=b; tc+=c
    print(f"{a} {b:9d} {c:10d} {ss:13d} {ff:13d}")
print("TOTAL     %d  %d"%(tb,tc))
# heterogeneity of b/(b+c)
ks2=[(sum(1 for r in by[a] if r["succ"] and not mh[r["seed"]]["succ"]),
      sum(1 for r in by[a] if r["succ"]!=mh[r["seed"]]["succ"])) for a in sorted(by)]
ks2=[t for t in ks2 if t[1]>0]
p=sum(t[0] for t in ks2)/sum(t[1] for t in ks2)
chi=sum((k-n*p)**2/(n*p*(1-p)) for k,n in ks2)
df=len(ks2)-1
def chi2sf(x,k):
    a=k/2.0; xx=x/2.0
    if xx<=0: return 1.0
    if xx<a+1:
        ap=a; s=1.0/a; d=s
        for _ in range(20000):
            ap+=1; d*=xx/ap; s+=d
            if abs(d)<abs(s)*1e-15: break
        return 1.0-s*math.exp(-xx+a*math.log(xx)-math.lgamma(a))
    bq=xx+1-a; c=1e300; d=1/bq; h=d
    for i in range(1,20000):
        an=-i*(i-a); bq+=2
        d=an*d+bq; d=1e-300 if abs(d)<1e-300 else d
        c=bq+an/c; c=1e-300 if abs(c)<1e-300 else c
        d=1/d; de=d*c; h*=de
        if abs(de-1)<1e-15: break
    return h*math.exp(-xx+a*math.log(xx)-math.lgamma(a))
print("discordant pairs n=%d across %d anatomies; pooled b/(b+c)=%.3f; heterogeneity chi2=%.2f df=%d p=%.3f"%(
  sum(t[1] for t in ks2),len(ks2),p,chi,df,chi2sf(chi,df)))
