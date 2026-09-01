import json,os,math
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json"))); LOW,SIP=dn["low"],dn["sip"]
names=sorted(LOW); N=216
def tab(d,f):
    g=defaultdict(lambda:[0,0])
    for nm in names:
        g[f[nm]][1]+=1; g[f[nm]][0]+=D[d][nm]
    return sorted(g.items(),key=lambda kv:(-kv[1][0]/kv[1][1],-kv[1][1]))
for d in ["RCCA_MIDVESSEL_035","RCCA_MIDVESSEL_018","RCCA_NEGCLEAR","RCCA_RVA_FUSED","NONMANIFOLD_ge3","RVA_STUB_MISSING","RECA_MID_035","ECA_TIP_FUSED_ICA","AUDIT22_lt090"]:
    K=sum(D[d].values())
    print("\n### %s  K=%d (base %.3f)"%(d,K,K/N))
    for fac,f in (("SIPHON",SIP),("LOWER",LOW)):
        t=tab(d,f)
        full=[x for x in t if x[1][0]==x[1][1] and x[1][1]>=2]
        zero=[x for x in t if x[1][0]==0]
        print("  %-6s donors: %d total; all-defective(n>=2): %d; zero-defect: %d ; top: %s"%(
          fac,len(t),len(full),len(zero),
          ", ".join("%s %d/%d"%(k,v[0],v[1]) for k,v in t[:6])))
# exclusion arithmetic: greedy donor removal for the union of "hard" defects
HARD={"RCCA_MIDVESSEL_035":"RCCA mid-vessel blockage @0.35",
      "RCCA_MIDVESSEL_030":"RCCA mid-vessel blockage @0.30"}
for key in HARD:
    bad=set(nm for nm in names if D[key][nm])
    print("\n### greedy exclusion for %s (%d bad)"%(key,len(bad)))
    for fac,f in (("SIPHON",SIP),("LOWER",LOW)):
        rem=set(names); b=set(bad); dropped=[]
        for step in range(8):
            g=defaultdict(lambda:[0,0])
            for nm in rem: g[f[nm]][1]+=1; g[f[nm]][0]+= (nm in b)
            best=max(g.items(),key=lambda kv:(kv[1][0], kv[1][0]/kv[1][1]))
            if best[1][0]==0: break
            dropped.append((best[0],best[1][0],best[1][1]))
            rem={nm for nm in rem if f[nm]!=best[0]}; b={nm for nm in b if nm in rem}
            print("   drop %-16s removes %2d bad / %d composites -> remaining %3d composites, %2d still bad"%(best[0],dropped[-1][1],dropped[-1][2],len(rem),len(b)))
# per-composite residual after dropping the 5 worst siphons
worst=["topcow_mr_001","topcow_mr_010_L","topcow_mr_023_L","topcow_mr_016_L","topcow_mr_013_L"]
rem=[nm for nm in names if SIP[nm] not in worst]
for key in ["RCCA_MIDVESSEL_035","RCCA_MIDVESSEL_030","RCCA_MIDVESSEL_018","RCCA_NEGCLEAR"]:
    print("drop 5 siphons (%d->%d composites): %s bad %d -> %d"%(N,len(rem),key,sum(D[key].values()),sum(D[key][nm] for nm in rem)))
