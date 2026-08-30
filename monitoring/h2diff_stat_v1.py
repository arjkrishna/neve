import json, math, random, collections
random.seed(12345)
def gammp(a,x):
    if x<0 or a<=0: return float('nan')
    if x<a+1:
        ap=a; s=1.0/a; d=s
        for _ in range(500):
            ap+=1; d*=x/ap; s+=d
            if abs(d)<abs(s)*1e-14: break
        return s*math.exp(-x+a*math.log(x)-math.lgamma(a))
    b=x+1-a; c=1e300; d=1.0/b; h=d
    for i in range(1,500):
        an=-i*(i-a); b+=2
        d=an*d+b; d=1e-300 if abs(d)<1e-300 else d
        c=b+an/c; c=1e-300 if abs(c)<1e-300 else c
        d=1.0/d; de=d*c; h*=de
        if abs(de-1)<1e-14: break
    return 1.0-math.exp(-x+a*math.log(x)-math.lgamma(a))*h
def chi2_sf(x,df): return 1.0-gammp(df/2.0,x/2.0) if x>0 else 1.0
def wilson(k,n,z=1.96):
    if n==0: return (0,1)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))

G=json.load(open("monitoring/h2diff_grafted.json"))  # anat -> [n, pol, heur]
def hetero(rows,label,col):
    N=sum(r[0] for r in rows.values()); K=sum(r[col] for r in rows.values())
    p0=K/N; X2=0.0; G2=0.0
    for a,r in rows.items():
        n,k=r[0],r[col]
        e1=n*p0; e0=n*(1-p0)
        X2+=(k-e1)**2/e1+((n-k)-e0)**2/e0
        if k>0: G2+=2*k*math.log(k/e1)
        if n-k>0: G2+=2*(n-k)*math.log((n-k)/e0)
    df=len(rows)-1
    print("%s: N=%d K=%d p0=%.4f  X2=%.3f df=%d p=%.4f  G2=%.3f p=%.4f  dispersion X2/df=%.3f"
          %(label,N,K,p0,X2,df,chi2_sf(X2,df),G2,chi2_sf(G2,df),X2/df))
    return X2,N,K,p0
def permtest(rows,col,X2obs,B=200000):
    sizes=[r[0] for r in rows.values()]; N=sum(sizes); K=sum(r[col] for r in rows.values())
    p0=K/N; outcomes=[1]*K+[0]*(N-K); cnt=0
    exp1=[n*p0 for n in sizes]; exp0=[n*(1-p0) for n in sizes]
    for _ in range(B):
        random.shuffle(outcomes); i=0; X2=0.0
        for j,n in enumerate(sizes):
            k=sum(outcomes[i:i+n]); i+=n
            X2+=(k-exp1[j])**2/exp1[j]+((n-k)-exp0[j])**2/exp0[j]
        if X2>=X2obs-1e-9: cnt+=1
    return (cnt+1)/(B+1)

print("=== ALL 22 ANATOMIES, grafted targets ===")
Xp,_,_,_=hetero(G,"POLICY  ",1); Xh,_,_,_=hetero(G,"HEURISTIC",2)
print(" perm p POLICY =%.5f   perm p HEUR =%.5f"%(permtest(G,1,Xp,50000),permtest(G,2,Xh,50000)))
G21={a:v for a,v in G.items() if a!="topcowmr025"}
print("\n=== EXCLUDING mr_025 (21 anatomies) ===")
Xp2,_,_,_=hetero(G21,"POLICY  ",1); Xh2,_,_,_=hetero(G21,"HEURISTIC",2)
print(" perm p POLICY =%.5f   perm p HEUR =%.5f"%(permtest(G21,1,Xp2,50000),permtest(G21,2,Xh2,50000)))
G20={a:v for a,v in G21.items() if a!="topcowmr024"}
print("\n=== EXCLUDING mr_025 + mr_024 (20 anatomies) ===")
Xp3,_,_,_=hetero(G20,"POLICY  ",1); hetero(G20,"HEURISTIC",2)
print(" perm p POLICY =%.5f"%permtest(G20,1,Xp3,50000))
print("\n=== per-anatomy Wilson 95%% CI (policy, grafted) ===")
for a in sorted(G,key=lambda a:-G[a][1]/G[a][0]):
    n,k,h=G[a]; lo,hi=wilson(k,n)
    print("%-14s %d/%-2d = %5.1f%%  Wilson[%.3f,%.3f] width=%.3f   heur %d/%d"%(a,k,n,100*k/n,lo,hi,hi-lo,h,n))
