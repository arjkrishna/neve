import json,os,math
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
rows=json.load(open(os.path.join(SCR,"ck15_perdonor.json")))
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json"))); LOW,SIP=dn["low"],dn["sip"]
names=sorted(LOW)
# donors implicated in >=1 defect at BH q<0.05
by=defaultdict(list)
for r in rows:
    if r["q"]<0.05 and r["k"]>0: by[(r["factor"],r["donor"])].append((r["defect"],r["k"],r["n"],r["p"],r["q"]))
print("DONORS IMPLICATED AT BH q<0.05 (within defect x factor family), sorted by #defect families")
for (fac,d),v in sorted(by.items(),key=lambda kv:-len(kv[1])):
    print("%-7s %-22s %d families: %s"%(fac,d,len(v),"; ".join("%s %d/%d q=%.1e"%(a,b,c,e) for a,b,c,dd,e in sorted(v))))
# how many composites would be lost by donor exclusion vs per-composite exclusion, for the union defect
UNION=["RCCA_MIDVESSEL_030","ECA_TIP_FUSED_ICA","RCCA_RVA_FUSED","NONMANIFOLD_ge3"]
bad=set(nm for nm in names if any(D[u][nm] for u in UNION))
print("\nUNION of (RCCA mid-vessel@0.30, ECA-tip fusion, RCCA-RVA fusion, nonmanifold>=3): %d/216 composites"%len(bad))
sipbad=defaultdict(int); lowbad=defaultdict(int); sipn=defaultdict(int); lown=defaultdict(int)
for nm in names:
    sipn[SIP[nm]]+=1; lown[LOW[nm]]+=1
    if nm in bad: sipbad[SIP[nm]]+=1; lowbad[LOW[nm]]+=1
for lbl,b,n in (("SIPHON",sipbad,sipn),("LOWER",lowbad,lown)):
    fully=[k for k in n if b[k]==n[k]]
    clean=[k for k in n if b[k]==0]
    print("  %s: %d donors, %d fully-bad, %d fully-clean; rate range %.2f-%.2f"%(lbl,len(n),len(fully),len(clean),
        min(b[k]/n[k] for k in n),max(b[k]/n[k] for k in n)))
