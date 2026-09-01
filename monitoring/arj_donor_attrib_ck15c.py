import json,os,math
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json")))
LOW,SIP=dn["low"],dn["sip"]; names=sorted(LOW); N=len(names)
def hsf(k,K,n,N):
    tot=math.comb(N,n); return sum(math.comb(K,i)*math.comb(N-K,n-i) for i in range(k,min(K,n)+1))/tot
def groups(f):
    g=defaultdict(list)
    for nm in names: g[f[nm]].append(nm)
    return g
GL=groups(LOW); GS=groups(SIP)
defects=[d for d in sorted(D) if 3<=sum(D[d].values())<=200]
rows=[]
for d in defects:
    K=sum(D[d].values())
    for fac,grp in (("LOWER",GL),("SIPHON",GS)):
        M=len(grp); res=[]
        for dn_,mem in grp.items():
            n=len(mem); k=sum(D[d][x] for x in mem)
            p=hsf(k,K,n,N) if k>0 else 1.0
            res.append((p,dn_,k,n))
        res.sort()
        # BH within (defect,factor)
        bh=[]
        for i,(p,dnm,k,n) in enumerate(res,1):
            bh.append((p,dnm,k,n,min(1.0,p*M),p*M/i))
        # monotone BH q
        q=1.0; qs=[]
        for i in range(len(bh)-1,-1,-1):
            q=min(q,bh[i][5]); qs.append(q)
        qs=qs[::-1]
        for (p,dnm,k,n,pb,_),qq in zip(bh,qs):
            rows.append(dict(defect=d,K=K,factor=fac,donor=dnm,k=k,n=n,rate=k/n,base=K/N,p=p,p_bonf=min(1.0,pb),q=min(1.0,qq),M=M))
json.dump(rows,open(os.path.join(SCR,"ck15_perdonor.json"),"w"))
print("BONFERRONI-SURVIVING PER-DONOR ENRICHMENTS (p_bonf < 0.05, correction = #donors in that family)")
print("%-26s %4s %-7s %-22s %2s/%1s  base   p         p_bonf   q" % ("defect","K","factor","donor","k","n"))
sig=[r for r in rows if r["p_bonf"]<0.05]
sig.sort(key=lambda r:(r["defect"],r["p"]))
for r in sig:
    print("%-26s %4d %-7s %-22s %2d/%d  %.3f  %.3e %.3e %.3e"%(r["defect"],r["K"],r["factor"],r["donor"],r["k"],r["n"],r["base"],r["p"],r["p_bonf"],r["q"]))
print("\ncount surviving:",len(sig),"of",len(rows),"donor-defect tests")
# global-corrected: bonferroni over ALL tests
allM=len(rows)
print("\nSURVIVING GLOBAL BONFERRONI over all %d donor x defect tests (alpha=%.2e):"%(allM,0.05/allM))
for r in sorted([r for r in rows if r["p"]*allM<0.05],key=lambda r:r["p"]):
    print("  %-26s %-7s %-22s %2d/%d p=%.3e p_all=%.3e"%(r["defect"],r["factor"],r["donor"],r["k"],r["n"],r["p"],r["p"]*allM))
