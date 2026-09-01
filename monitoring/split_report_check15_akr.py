exec(open(r"D:\Arjun\workspace\neve\monitoring\split_frontier_check15_akr.py").read().split("ALLL=set")[0])
import json, statistics as stt
ALLL=set(range(NL)); ALLP=set(range(NP))
prov={}
for d in sorted(os.listdir(ROOT)):
    prov[d]=json.load(open(os.path.join(ROOT,d,"provenance.json"),encoding="utf-8"))

# stage 1: carve TEST
bb=None
for sd in range(10):
    b,Ls,Ps=anneal_sub(ALLL,ALLP,(0,43),300000,sd,60.0)
    if b[3][1]<43: continue
    if bb is None or b[3][0]>bb[0][3][0]: bb=(b,Ls,Ps)
b,Ls,Ps=bb
testL={Ls[i] for i in range(len(Ls)) if b[1][i]==1}
testP={Ps[j] for j in range(len(Ps)) if b[2][j]==1}
poolL=ALLL-testL; poolP=ALLP-testP
# stage 2: carve VAL out of the pool
cc=None
for sd in range(10):
    b2,Ls2,Ps2=anneal_sub(poolL,poolP,(0,22),300000,sd,60.0)
    if b2[3][1]<22: continue
    if cc is None or b2[3][0]>cc[0][3][0]: cc=(b2,Ls2,Ps2)
b2,Ls2,Ps2=cc
valL={Ls2[i] for i in range(len(Ls2)) if b2[2-1][i]==1} if False else {Ls2[i] for i in range(len(Ls2)) if b2[1][i]==1}
valP={Ps2[j] for j in range(len(Ps2)) if b2[2][j]==1}
trL=poolL-valL; trP=poolP-valP

def grp(l,p):
    li,pi=LI[l],PI[p]
    for nm,(A,B) in (("test",(testL,testP)),("val",(valL,valP)),("train",(trL,trP))):
        if li in A and pi in B: return nm
    return "DROP"
assign={}
for name,l,s,pt in rows: assign[name]=grp(l,pt)
from collections import Counter
cnt=Counter(assign.values())
print("FINAL 3-WAY donor-level split:", dict(cnt))
inv={"train":(trL,trP),"val":(valL,valP),"test":(testL,testP)}
for g,(A,B) in inv.items():
    print("  %-5s : %2d lower donors, %2d patients (%2d siphons), %3d composites"
          %(g,len(A),len(B),sum(1 for r in rows if PI[r[3]] in B),cnt[g]))
print("  DROP (bridging composites): %d (%.1f%%)"%(cnt["DROP"],100*cnt["DROP"]/216))
# leak verification
tr_l={r[1] for r in rows if assign[r[0]]=="train"}; tr_p={r[3] for r in rows if assign[r[0]]=="train"}
tr_s={r[2] for r in rows if assign[r[0]]=="train"}
bad=[r[0] for r in rows if assign[r[0]]=="test" and (r[1] in tr_l or r[3] in tr_p or r[2] in tr_s)]
print("  LEAK CHECK test-vs-train donor overlap:",len(bad))
va_l={r[1] for r in rows if assign[r[0]]=="val"}; va_p={r[3] for r in rows if assign[r[0]]=="val"}
bad2=[r[0] for r in rows if assign[r[0]]=="test" and (r[1] in va_l or r[3] in va_p)]
print("  LEAK CHECK test-vs-val donor overlap:",len(bad2))

# difficulty distributions per group
keys=["mismatch_mm","clearance_mm","max_kink","seam1_kink","route_min_r_raw","route_floored_frac","eca_floored_frac","total_mm"]
print("\nDIFFICULTY PROXIES (median [min,max]) by group:")
hdr="  %-18s"%"var"+"".join("%-26s"%g for g in ("train","val","test","DROP"))
print(hdr)
for k in keys:
    line="  %-18s"%k
    for g in ("train","val","test","DROP"):
        v=[prov[n][k] for n in assign if assign[n]==g and k in prov[n]]
        line+="%-26s"%("%.2f [%.2f,%.2f]"%(stt.median(v),min(v),max(v)) if v else "-")
    print(line)
# side balance
print("\nside balance:", {g:Counter(("left" if r[1].endswith("_left") else "right") for r in rows if assign[r[0]]==g) for g in ("train","val","test","DROP")})
json.dump(assign,open(r"D:\Arjun\workspace\neve\monitoring\_final_split_assign.json","w"),indent=0)
print("\nTEST composites:")
for r in rows:
    if assign[r[0]]=="test": print("   ",r[0])
print("\nVAL composites:")
for r in rows:
    if assign[r[0]]=="val": print("   ",r[0])
