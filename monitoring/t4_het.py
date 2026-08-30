import pickle, numpy as np, collections, math
import math as _m
class _S:
    @staticmethod
    def chi2sf(x,k):
        # regularized upper incomplete gamma Q(k/2, x/2)
        a=k/2.0; xx=x/2.0
        if xx<=0: return 1.0
        if xx < a+1:
            # series for P
            ap=a; s=1.0/a; d=s
            for _ in range(10000):
                ap+=1; d*=xx/ap; s+=d
                if abs(d)<abs(s)*1e-15: break
            return 1.0-s*_m.exp(-xx+a*_m.log(xx)-_m.lgamma(a))
        else:
            b=xx+1-a; c=1e300; d=1/b; h=d
            for i in range(1,10000):
                an=-i*(i-a); b+=2
                d=an*d+b; d=1e-300 if abs(d)<1e-300 else d
                c=b+an/c; c=1e-300 if abs(c)<1e-300 else c
                d=1/d; de=d*c; h*=de
                if abs(de-1)<1e-15: break
            return h*_m.exp(-xx+a*_m.log(xx)-_m.lgamma(a))
    @staticmethod
    def binom_p2(k,n):
        from math import comb
        if n==0: return 1.0
        pk=sum(comb(n,i) for i in range(n+1) if comb(n,i)*0.5**n <= comb(n,k)*0.5**n+1e-12)
        return min(1.0, pk*0.5**n)
stats=None
D = pickle.load(open("_t4_rows.pkl","rb")); T,H = D["T"], D["H"]
mh = {r["seed"]:r for r in H}
gt = [r for r in T if r["grafted"]]
by = collections.defaultdict(list)
for r in gt: by[r["anat"]].append(r)
anats = sorted(by)

def het(counts, label):
    n = np.array([c[1] for c in counts], float); k = np.array([c[0] for c in counts], float)
    p = k.sum()/n.sum()
    # Pearson chi2
    exp = n*p
    chi2 = np.sum((k-exp)**2/(n*p*(1-p)))
    # LR
    G = 0.0
    for ki,ni in zip(k,n):
        for c,e in ((ki, ni*p),(ni-ki, ni*(1-p))):
            if c>0: G += 2*c*math.log(c/e)
    df = len(n)-1
    print(f"{label}: pooled p={p:.4f} N={int(n.sum())} k={int(k.sum())} groups={len(n)}")
    print(f"  Pearson chi2={chi2:.2f} df={df} p={_S.chi2sf(chi2,df):.4f}  dispersion={chi2/df:.3f}")
    print(f"  LR G2     ={G:.2f} df={df} p={_S.chi2sf(G,df):.4f}")
    return p

# teacher, all 22
cnt = [(sum(r["succ"] for r in by[a]), len(by[a])) for a in anats]
p0=het(cnt, "TEACHER grafted, all 22")
# excluding mr_025
a2=[a for a in anats if a!="mr_mr025"]
het([(sum(r["succ"] for r in by[a]), len(by[a])) for a in a2], "TEACHER grafted, excl mr_025")
# heuristic
het([(sum(mh[r["seed"]]["succ"] for r in by[a]), len(by[a])) for a in anats], "HEURISTIC grafted, all 22")
het([(sum(mh[r["seed"]]["succ"] for r in by[a]), len(by[a])) for a in a2], "HEURISTIC grafted, excl mr_025")

# Wilson intervals
def wilson(k,n,z=1.96):
    if n==0: return (0,1)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))
print("\nanat   k/n   rate   Wilson95        H k/n   Hrate")
for a in anats:
    rs=by[a]; n=len(rs); k=sum(r["succ"] for r in rs); hk=sum(mh[r["seed"]]["succ"] for r in rs)
    lo,hi=wilson(k,n)
    flag = "*" if (lo>p0 or hi<p0) else " "
    print(f"{a} {k:2d}/{n:2d} {100*k/n:6.1f}  [{100*lo:5.1f},{100*hi:5.1f}]{flag}  {hk:2d}/{n:2d} {100*hk/n:6.1f}")

# McNemar teacher vs heuristic on paired grafted episodes
b=sum(1 for r in gt if r["succ"] and not mh[r["seed"]]["succ"])
c=sum(1 for r in gt if not r["succ"] and mh[r["seed"]]["succ"])
print(f"\nPaired grafted: teacher-only-succ b={b}, heur-only-succ c={c}, "
      f"McNemar exact p={_S.binom_p2(min(b,c),b+c):.4f}")
bothf=sum(1 for r in gt if not r["succ"] and not mh[r["seed"]]["succ"])
boths=sum(1 for r in gt if r["succ"] and mh[r["seed"]]["succ"])
print(f"  both succ={boths} both fail={bothf}")
