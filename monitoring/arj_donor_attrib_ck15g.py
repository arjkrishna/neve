import json,os
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json"))); LOW,SIP=dn["low"],dn["sip"]
names=sorted(LOW)
SIPDROP={"topcow_mr_001","topcow_mr_010_L","topcow_mr_023_L","topcow_mr_016_L","topcow_mr_013_L"}
LOWDROP={"case_w_050_left","case_m_024_left","case_w_047_left","case_w_034_right"}
keep=[n for n in names if SIP[n] not in SIPDROP and LOW[n] not in LOWDROP]
print("after dropping 5 siphon + 4 lower donors: %d composites keep (%d dropped)"%(len(keep),216-len(keep)))
FAM={"RCCA mid-vessel @0.30":"RCCA_MIDVESSEL_030","RCCA mid-vessel @0.35":"RCCA_MIDVESSEL_035",
     "RCCA neg clearance":"RCCA_NEGCLEAR","ECA-tip fused":"ECA_TIP_FUSED_ICA",
     "RCCA-RVA fused":"RCCA_RVA_FUSED","nonmanifold>=3":"NONMANIFOLD_ge3",
     "RECA guidewire mid-arrest":"RECA_MID_018","RECA catheter wedge<10mm":"RECA_WEDGE_035_lt10mm",
     "RVA stub missing":"RVA_STUB_MISSING"}
for lbl,k in FAM.items():
    a=sum(D[k].values()); b=sum(D[k][n] for n in keep)
    print("  %-26s %3d/216 -> %3d/%d  (residual rate %.3f vs %.3f)"%(lbl,a,b,len(keep),b/len(keep),a/216))
res=[n for n in keep if D["RCCA_MIDVESSEL_030"][n]]
print("\nresidual per-composite exclusions for RCCA mid-vessel@0.30 after donor drop (%d):"%len(res))
bys=defaultdict(list)
for n in res: bys[SIP[n]].append(n)
for s,v in sorted(bys.items(),key=lambda kv:-len(kv[1])): print("   %-18s %d: %s"%(s,len(v),", ".join(x.split("__")[0] for x in v)))
