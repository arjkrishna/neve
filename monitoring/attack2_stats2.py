import json,itertools,os,glob
import numpy as np
rng=np.random.default_rng(0)
G=json.load(open("monitoring/attack2_geom_m4.json"))
HOLD=["topcow_mr_004","topcow_mr_008","topcow_mr_017","topcow_mr_023"]
rows={x["a"]:x for x in G["rows"]}; names=sorted(rows)
def z(v): v=np.array(v,float); return (v-v.mean())/v.std(ddof=0)
Z=lambda k,f=lambda x:x: dict(zip(names,z([f(rows[n][k]) for n in names])))
z1=Z("Rcmin",lambda x:1/x); z2=Z("kp95"); z3=Z("bendmax"); z4=Z("turn"); z5=Z("tort")
z6=Z("rmin",lambda x:-x); z7=Z("flr"); z8=Z("clrmin",lambda x:-x); z9=Z("gL")
D={n:(z1[n]+z2[n]+z3[n]+z4[n]+z5[n])/5+(z6[n]+z7[n])/2+z8[n]+0.5*z9[n] for n in names}
COMB=list(itertools.combinations(names,4))
def test(v,name):
    obs=np.mean([v[n] for n in HOLD])
    a=np.array([np.mean([v[n] for n in c]) for c in COMB])
    p=(a>=obs).mean(); rk=sorted(names,key=lambda n:-v[n])
    print("  %-26s holdout=%8.3f cohort=%8.3f  p=%.5f (%d/7315)  benign-ranks=%s"%(
      name,obs,np.mean(list(v.values())),p,int(round(p*7315)),sorted(rk.index(n)+1 for n in HOLD)))
print("=== SELECTION-BIAS: exact permutation over all C(22,4)=7315 subsets (p = P[a random 4 is this benign or more]) ===")
test({n:rows[n]["Rcmin"] for n in names},"Rc_min graft (mm)")
test({n:-rows[n]["bendmax"] for n in names},"-max bend 8mm (deg)")
test({n:-rows[n]["turn"] for n in names},"-cumulative turning")
test({n:rows[n]["clrmin"] for n in names},"min clearance (mm)")
test({n:-rows[n]["flr"] for n in names},"-frac stations r<=2.0")
test({n:-D[n] for n in names},"composite benign-ness -D")
print()
obs={"topcow_mr_008":(26,26),"topcow_mr_017":(28,29),"topcow_mr_004":(24,27),"topcow_mr_023":(12,16)}
Dh=np.array([D[n] for n in HOLD]); k=np.array([obs[n][0] for n in HOLD],float); N=np.array([obs[n][1] for n in HOLD],float)
Do=np.array([D[n] for n in names if n not in HOLD])
print("=== EXTRAPOLATION (anatomy-level logistic on composite D) ===")
print("  D: holdout mean=%.2f (%s)  other18 mean=%.2f  gap=%.2f z"%(Dh.mean(),
  ", ".join("%s=%.2f"%(h[-3:],D[h]) for h in HOLD),Do.mean(),Do.mean()-Dh.mean()))
def fit(Dv,kv,Nv,ridge,it=60):
    X=np.column_stack([np.ones(len(Dv)),Dv]); b=np.zeros(2)
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=Nv*np.maximum(p*(1-p),1e-9)
        b=b+np.linalg.solve(X.T@(X*W[:,None])+ridge*np.eye(2), X.T@(kv-Nv*p)-ridge*b)
    return b
for ridge in (0.25,0.5,1.0,2.0):
    b=fit(Dh,k,N,ridge); pp=1/(1+np.exp(-(b[0]+b[1]*Do)))
    print("  ridge=%.2f slope=%+.3f  other18=%.3f  all22=%.3f"%(ridge,b[1],pp.mean(),(pp.sum()+(k/N).sum())/22))
bs=[]
for _ in range(3000):
    idx=rng.integers(0,4,4)
    kk=rng.binomial(N[idx].astype(int),np.clip(k[idx]/N[idx],1e-6,1-1e-6)).astype(float)
    b=fit(Dh[idx],kk,N[idx],0.5,40); pp=1/(1+np.exp(-(b[0]+b[1]*Do)))
    bs.append((pp.sum()+(kk/N[idx]).sum())/22)
bs=np.array(bs)
print("  bootstrap all-22: median=%.3f  80%%CI [%.3f,%.3f]  95%%CI [%.3f,%.3f]"%(
  np.median(bs),np.percentile(bs,10),np.percentile(bs,90),np.percentile(bs,2.5),np.percentile(bs,97.5)))
print()
exec(open("monitoring/attack2_holdout_geometry.py").read().split("def analyse")[0].replace("main()",""))
mix=[]
for n in names:
    p,r=read_curve(os.path.join(ANAT,n,"Centrelines_comb",RCCA_FILE)); s=arclength(p); L=s[-1]
    st=s[(s>=40)&(s<=L-8)]
    mix.append([(st<112.5).mean(),((st>=112.5)&(st<176.5)).mean(),(st>=176.5).mean()])
mix=np.array(mix); ih=[names.index(h) for h in HOLD]
print("=== SECTION MIX (targets uniform over stations s in [40,L-8]; cuts s=112.5/176.5 == path_len 146/210) ===")
print("  holdout4  CCA/ICAmid/siphon = %.3f/%.3f/%.3f"%tuple(mix[ih].mean(0)))
print("  other18                     = %.3f/%.3f/%.3f"%tuple(np.delete(mix,ih,0).mean(0)))
print("  all22                       = %.3f/%.3f/%.3f"%tuple(mix.mean(0)))
print("  observed holdout mix (98 eps) = %.3f/%.3f/%.3f"%(31/98,41/98,26/98))
base=np.array([1.00,0.902,0.846]); w=mix.mean(0)
print("=== SCENARIO BRACKET (per-section degradation applied to the other 18) ===")
for lab,d in (("A no degradation",[0,0,0]),("B mild  -2 mid / -5 siphon",[0,-.02,-.05]),
              ("C moder -5 mid / -15 siphon",[0,-.05,-.15]),("D severe -12 mid / -30 siphon",[0,-.12,-.30]),
              ("E 023-like: 1/4 of siphons at 0%",[0,-.05,-.21])):
    print("   %-34s all22 = %.3f"%(lab,float(w@(base+np.array(d)))))
