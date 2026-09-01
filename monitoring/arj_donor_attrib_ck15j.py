import json,os,random,sys
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json"))); LOW,SIP=dn["low"],dn["sip"]
names=sorted(LOW); random.seed(7)
idx={n:i for i,n in enumerate(names)}
def grp(f):
    g=defaultdict(list)
    for n in names: g[f[n]].append(idx[n])
    return list(g.values())
GL=grp(LOW); GS=grp(SIP)
NP=20000; NPS=10000
KEY=["RCCA_MIDVESSEL_035","RCCA_MIDVESSEL_030","RCCA_MIDVESSEL_018","RCCA_NEGCLEAR","RCCA_DEEP40_035",
     "RECA_MID_035","RECA_WEDGE_035_lt10mm","RECA_TIP_OUTSIDE","ECA_TIP_FUSED_ICA","RCCA_RVA_FUSED",
     "NONMANIFOLD_ge3","RVA_STUB_MISSING","MID_FRAGMENT","AUDIT22_lt090","OPENBOUND_main_ge3"]
def conc(lab,G): return sum(k*(k-1) for k in (sum(lab[i] for i in g) for g in G))
def pg(base,G,obs,nperm):
    a=list(base); h=0; s2=0.0; s1=0.0
    for _ in range(nperm):
        random.shuffle(a); v=conc(a,G); s1+=v; s2+=v*v
        if v>=obs: h+=1
    mu=s1/nperm; sd=max(1e-9,(s2/nperm-mu*mu)**0.5)
    return (h+1)/(nperm+1),mu,(obs-mu)/sd
def ps(base,blocks,G,obs,nperm):
    h=0; s1=0.0; s2=0.0
    for _ in range(nperm):
        cur=list(base)
        for b in blocks:
            vs=[base[i] for i in b]; random.shuffle(vs)
            for i,v in zip(b,vs): cur[i]=v
        v=conc(cur,G); s1+=v; s2+=v*v
        if v>=obs: h+=1
    mu=s1/nperm; sd=max(1e-9,(s2/nperm-mu*mu)**0.5)
    return (h+1)/(nperm+1),mu,(obs-mu)/sd
BL_low=grp(LOW); BL_sip=grp(SIP)
print("defect                       K | LOWER  obs/exp z p      | SIPHON obs/exp z p      | S|L p   L|S p")
for d in KEY:
    base=[D[d][n] for n in names]; K=sum(base)
    ol=conc(base,GL); os_=conc(base,GS)
    pl,mul,zl=pg(base,GL,ol,NP); psi,mus,zs=pg(base,GS,os_,NP)
    q1,_,_=ps(base,BL_low,GS,os_,NPS)   # siphon effect within lower strata
    q2,_,_=ps(base,BL_sip,GL,ol,NPS)    # lower effect within siphon strata
    print("%-28s %3d | %4d/%6.1f z=%5.2f p=%.5f | %4d/%6.1f z=%5.2f p=%.5f | %.4f  %.4f"%(d,K,ol,mul,zl,pl,os_,mus,zs,psi,q1,q2))
    sys.stdout.flush()
