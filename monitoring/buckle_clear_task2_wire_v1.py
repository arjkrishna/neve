"""TASK 2 supplement 2: where wire travel is burned; distal-vs-proximal clearing; station detail."""
import json,sys,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
S="C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
def lab(e):
    if not (e["fold_load"]>=4 or e["slack_load"]>=10.0): return "COSMETIC"
    return "CLEARING" if (e["fold_close"]==0 and e["adv25"]>=15.0 and not e["restall_same"]) else "FUTILE"
def load(p): return [json.loads(l) for l in open(S+p)]
A=load("tr_A.jsonl"); A5=load("tr_A514.jsonl"); TB=load("tr_ATB.jsonl")
SL=[("HOST ck2002292",A),("HOST ck514264",A5),("TB GRAFTED",[e for e in TB if e["pl"]>166.91]),
    ("TB SHARED",[e for e in TB if e["pl"]<=166.91])]
print("="*130); print("WIRE TRAVEL ACCOUNTING: total |d gw| split into inside-stall-window vs free navigation")
for t,eps in SL:
    tot=0.;ins=0.;insk=collections.Counter();ntot=0.
    for ep in eps:
        g=ep["gw"]; n=len(g)
        if n<2: continue
        d=[abs(g[i+1]-g[i]) for i in range(n-1)]
        tot+=sum(d); ntot+=max(g)-g[0]
        mask=[0]*(n-1)
        for x in ep["events"]:
            a=max(0,x["first"]-1); b=(x["close"]-1) if x["close"]>0 else n-1
            for j in range(a,min(b,n-1)): 
                if not mask[j]: mask[j]=1; insk[x["k"]]+=d[j]
        ins+=sum(d[j] for j in range(n-1) if mask[j])
    print(" %-15s total gw travel=%9.0f mm  net gain=%8.0f mm  eff=%.3f | inside stall windows=%8.0f mm (%4.1f%%)  [grind %.0f soft %.0f hard %.0f unrec %.0f]"%(
        t,tot,ntot,ntot/tot,ins,100.*ins/tot,insk["grind"],insk["soft"],insk["hard"],insk["unrec"]))
print()
print("="*130); print("PROXIMAL (p0<100mm) vs DISTAL (p0>=100mm) clearing")
eA=features(S+"tr_A.jsonl"); eA5=features(S+"tr_A514.jsonl"); eTB=features(S+"tr_ATB.jsonl")
EV=[("HOST ck2002292",eA),("HOST ck514264",eA5),("TB GRAFTED",[e for e in eTB if e["pl"]>166.91])]
for t,ev in EV:
    for nm,f in [("PROXIMAL p0<100mm",lambda e:e["p0"]<100),("DISTAL p0>=100mm",lambda e:e["p0"]>=100)]:
        g=[e for e in ev if f(e)]
        if not g: print("  %-15s %-20s n=0"%(t,nm)); continue
        L=collections.Counter(lab(e) for e in g)
        bp=sum(1 for e in g if e["fold_load"]>=4 or e["slack_load"]>=10.0)
        print("  %-15s %-20s s+h n=%3d  buckle-present %2d (%4.1f%%)  CLEARING %2d (%4.1f%% of s+h, %4.1f%% of buckle-present)  FUT %2d COSM %2d"%(
            t,nm,len(g),bp,100.*bp/len(g),L["CLEARING"],100.*L["CLEARING"]/len(g),100.*L["CLEARING"]/max(1,bp),L["FUTILE"],L["COSMETIC"]))
print()
print("="*130); print("ARREST STATION HISTOGRAM (p0, 1mm resolution, top stations) + clearing at each")
for t,ev in EV:
    c=collections.Counter(round(e["p0"]) for e in ev)
    print(" %s  (n=%d s+h)"%(t,len(ev)))
    for st,n in c.most_common(6):
        g=[e for e in ev if round(e["p0"])==st]
        cl=sum(1 for e in g if lab(e)=="CLEARING"); bp=sum(1 for e in g if e["fold_load"]>=4 or e["slack_load"]>=10.0)
        print("   p0=%4d mm  n=%3d (%4.1f%% of s+h)  buckle-present %2d  CLEARING %2d (%4.1f%%)  P(ep succ)=%.3f"%(
            st,n,100.*n/len(ev),bp,cl,100.*cl/n,sum(1 for e in g if e["succ"])/len(g)))
print()
print("="*130); print("CONDITIONAL EFFICACY -- given a soft/hard event is ATTEMPTED, what fraction is genuine?")
print("%-15s %6s %6s %8s %8s %10s %10s"%("slice","s+h","buckP","buckP%","CLEAR","CLR/s+h%","CLR/buckP%"))
for t,ev in EV:
    bp=[e for e in ev if e["fold_load"]>=4 or e["slack_load"]>=10.0]
    cl=[e for e in ev if lab(e)=="CLEARING"]
    print("%-15s %6d %6d %7.1f%% %8d %9.1f%% %9.1f%%"%(t,len(ev),len(bp),100.*len(bp)/len(ev),len(cl),100.*len(cl)/len(ev),100.*len(cl)/max(1,len(bp))))
