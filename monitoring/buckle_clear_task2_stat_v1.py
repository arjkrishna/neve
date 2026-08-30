"""TASK 2 supplement 3: Fisher exact for CLEARING vs FUTILE per slice + deep-retraction band."""
import json,sys,math,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
S="C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
def lab(e):
    if not (e["fold_load"]>=4 or e["slack_load"]>=10.0): return "COSMETIC"
    return "CLEARING" if (e["fold_close"]==0 and e["adv25"]>=15.0 and not e["restall_same"]) else "FUTILE"
C=math.comb
def fisher(a,b,c,d):
    n=a+b+c+d; tot=0.
    def p(a_):
        b_=a+b-a_; c_=a+c-a_; d_=d-(a_-a)
        if min(a_,b_,c_,d_)<0: return 0.
        return C(a+b,a_)*C(c+d,c_)/C(n,a+c)
    obs=p(a)
    for a_ in range(0,min(a+b,a+c)+1):
        if a_>=a: tot+=p(a_)
    return tot
eA=features(S+"tr_A.jsonl"); eA5=features(S+"tr_A514.jsonl"); eTB=features(S+"tr_ATB.jsonl")
EV=[("HOST ck2002292",eA),("HOST ck514264",eA5),
    ("TB all-22",eTB),("TB GRAFTED",[e for e in eTB if e["pl"]>166.91])]
print("="*120); print("FISHER one-sided, P(ep succ | CLEARING) > P(ep succ | FUTILE)")
for t,ev in EV:
    cl=[e for e in ev if lab(e)=="CLEARING"]; fu=[e for e in ev if lab(e)=="FUTILE"]
    a=sum(1 for e in cl if e["succ"]); b=len(cl)-a; c=sum(1 for e in fu if e["succ"]); d=len(fu)-c
    print("  %-15s CLEAR %d/%d=%.3f  FUTILE %d/%d=%.3f  diff=%+.3f  Fisher p=%.4f"%(
        t,a,len(cl),a/max(1,len(cl)),c,len(fu),c/max(1,len(fu)),a/max(1,len(cl))-c/max(1,len(fu)),fisher(a,b,c,d)))
print()
print("="*120); print("CLEARING vs COSMETIC+FUTILE combined (event level)")
for t,ev in EV:
    cl=[e for e in ev if lab(e)=="CLEARING"]; ot=[e for e in ev if lab(e)!="CLEARING"]
    a=sum(1 for e in cl if e["succ"]); c=sum(1 for e in ot if e["succ"])
    print("  %-15s CLEAR %d/%d=%.3f  other %d/%d=%.3f  diff=%+.3f  Fisher p=%.4f"%(
        t,a,len(cl),a/max(1,len(cl)),c,len(ot),c/max(1,len(ot)),a/max(1,len(cl))-c/max(1,len(ot)),fisher(a,len(cl)-a,c,len(ot)-c)))
print()
print("="*120); print("RETRACTION-DEPTH LADDER on buckle-present events (is depth buying clearance?)")
BANDS=[(1.0,4.0),(4.0,8.0),(8.0,20.0),(20.0,50.0),(50.0,1e9)]
for t,ev in EV:
    bp=[e for e in ev if e["fold_load"]>=4 or e["slack_load"]>=10.0]
    print(" %s (buckle-present n=%d)"%(t,len(bp)))
    for lo,hi in BANDS:
        g=[e for e in bp if lo<=e["retract"]<hi]
        if not g: continue
        print("   retract %5.0f-%-5s n=%2d | med retract=%7.2f med slack_load=%6.2f med slack_rel=%6.2f  rel/retract=%5.3f | med adv25=%6.2f  CLEARING %d (%4.0f%%)  P(ep succ)=%.3f"%(
            lo,("inf" if hi>1e8 else "%.0f"%hi),len(g),q([e["retract"] for e in g],50),q([e["slack_load"] for e in g],50),
            q([e["slack_rel"] for e in g],50),q([e["slack_rel"]/max(e["retract"],1e-9) for e in g],50),
            q([e["adv25"] for e in g],50),sum(1 for e in g if lab(e)=="CLEARING"),
            100.*sum(1 for e in g if lab(e)=="CLEARING")/len(g),sum(1 for e in g if e["succ"])/len(g)))
print()
print("="*120); print("EVERY CLEARING EVENT (host ck2002292 and TB GRAFTED)")
for t,ev in [("HOST ck2002292",eA),("TB GRAFTED",[e for e in eTB if e["pl"]>166.91])]:
    print(" %s"%t)
    for e in sorted([x for x in ev if lab(x)=="CLEARING"],key=lambda x:x["p0"]):
        print("   seed=%-8s pl=%6.1f k=%-4s p0=%6.1f frac=%.3f retract=%7.2f fold_load=%2d slack_load=%6.2f slack_rel=%6.2f adv25=%6.2f adv_end=%7.2f succ=%s"%(
            e["seed"],e["pl"],e["k"],e["p0"],e["p0"]/e["pl"],e["retract"],e["fold_load"],e["slack_load"],e["slack_rel"],e["adv25"],e["adv_end"],e["succ"]))
