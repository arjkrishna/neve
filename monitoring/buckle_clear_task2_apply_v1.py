"""TASK 2 -- apply the VALIDATED buckle-clearing definition (verbatim) to family A slices."""
import json,sys,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q,med
S="C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
F_DEF,S_DEF,A_DEF=4,10.0,15.0
def lab(e):
    if not (e["fold_load"]>=F_DEF or e["slack_load"]>=S_DEF): return "COSMETIC"
    return "CLEARING" if (e["fold_close"]==0 and e["adv25"]>=A_DEF and not e["restall_same"]) else "FUTILE"
def load(p): return [json.loads(l) for l in open(S+p)]

def slice_report(tag,eps,evs):
    n_ep=len(eps); steps=sum(len(e["proj"]) for e in eps); succ=sum(1 for e in eps if e["succ"])
    allev=sum(len(e["events"]) for e in eps)
    print("="*140)
    print("%s"%tag)
    print("  episodes=%d  success=%d (%.1f%%)  steps=%d  all canon stalls=%d  soft+hard=%d"%(n_ep,succ,100.*succ/n_ep,steps,allev,len(evs)))
    if not evs:
        print("  NO SOFT/HARD EVENTS -- nothing to classify."); return None
    c={}
    for e in evs: c.setdefault(lab(e),[]).append(e)
    for k in ("CLEARING","FUTILE","COSMETIC"):
        g=c.get(k,[])
        if not g: print("   %-8s   0 ( 0.0%% of s+h, 0.0%% of all stalls)"%k); continue
        s=sum(1 for e in g if e["succ"])
        print("   %-8s %3d (%5.1f%% of s+h, %5.1f%% of all stalls)  P(ep succ)=%.3f (%d/%d)  rate=%.3f/1k %.3f/ep  med adv25=%6.2f  med adv_end=%7.2f  med retract=%6.2f  med slack_rel=%6.2f  med fold_load=%4.1f  hard%%=%.0f"%(
            k,len(g),100.*len(g)/len(evs),100.*len(g)/allev,s/len(g),s,len(g),
            1000.*len(g)/steps,len(g)/n_ep,
            q([e["adv25"] for e in g],50),q([e["adv_end"] for e in g],50),q([e["retract"] for e in g],50),
            q([e["slack_rel"] for e in g],50),q([float(e["fold_load"]) for e in g],50),
            100.*sum(1 for e in g if e["k"]=="hard")/len(g)))
    cl=c.get("CLEARING",[]); fu=c.get("FUTILE",[])
    df=[e for e in cl if not e["succ"]]
    print("   DEFERRED %d/%d CLEARING in a FAILED episode (%.1f%% of CLEARING, %.1f%% of s+h)"%(
        len(df),len(cl),100.*len(df)/max(1,len(cl)),100.*len(df)/len(evs)))
    if cl and fu:
        a=sum(1 for e in cl if e["succ"])/len(cl); b=sum(1 for e in fu if e["succ"])/len(fu)
        print("   >> event-level sep P(succ|CLEARING)=%.3f vs P(succ|FUTILE)=%.3f diff=%+.3f"%(a,b,a-b))
    # episode level: >=1 CLEARING vs only FUTILE/COSMETIC
    bys={}
    for e in evs: bys.setdefault(e["seed"],[]).append(e)
    sm={ep["seed"]:ep["succ"] for ep in eps}
    hc=[s for s,g in bys.items() if any(lab(x)=="CLEARING" for x in g)]
    nc=[s for s,g in bys.items() if not any(lab(x)=="CLEARING" for x in g)]
    def r(ss): return (sum(1 for s in ss if sm[s]),len(ss))
    a1,a2=r(hc); b1,b2=r(nc)
    print("   EPISODE-LEVEL  >=1 CLEARING: n=%d succ=%.3f (%d/%d)   |   only FUTILE/COSMETIC: n=%d succ=%.3f (%d/%d)   diff=%+.3f"%(
        a2,a1/max(1,a2),a1,a2,b2,b1/max(1,b2),b1,b2,(a1/max(1,a2))-(b1/max(1,b2))))
    epsno=n_ep-len(bys)
    nos=sum(1 for ep in eps if ep["seed"] not in bys and ep["succ"])
    print("   EPISODE-LEVEL  no soft/hard event at all: n=%d succ=%.3f"%(epsno,nos/max(1,epsno)))
    if cl:
        print("   CLEARING slack reduction (slack_rel, mm): p25=%.2f p50=%.2f p75=%.2f | slack_load p50=%.2f | slack_resid p50=%.2f"%(
            q([e["slack_rel"] for e in cl],25),q([e["slack_rel"] for e in cl],50),q([e["slack_rel"] for e in cl],75),
            q([e["slack_load"] for e in cl],50),q([e["slack_resid"] for e in cl],50)))
        print("   CLEARING post advance (mm): adv10 p50=%.2f | adv25 p25=%.2f p50=%.2f p75=%.2f max=%.2f | adv50 p50=%.2f | adv_end p50=%.2f"%(
            q([e["adv10"] for e in cl],50),q([e["adv25"] for e in cl],25),q([e["adv25"] for e in cl],50),
            q([e["adv25"] for e in cl],75),max(e["adv25"] for e in cl),q([e["adv50"] for e in cl],50),q([e["adv_end"] for e in cl],50)))
        print("   CLEARING location p0 (mm): "+" ".join("%.1f"%e["p0"] for e in sorted(cl,key=lambda x:x["p0"])))
        print("   CLEARING location frac of path_len p0/pl: "+" ".join("%.3f"%(e["p0"]/e["pl"]) for e in sorted(cl,key=lambda x:x["p0"]/x["pl"])))
        print("   CLEARING p0 quartiles: p25=%.1f p50=%.1f p75=%.1f  | frac p25=%.3f p50=%.3f p75=%.3f"%(
            q([e["p0"] for e in cl],25),q([e["p0"] for e in cl],50),q([e["p0"] for e in cl],75),
            q([e["p0"]/e["pl"] for e in cl],25),q([e["p0"]/e["pl"] for e in cl],50),q([e["p0"]/e["pl"] for e in cl],75)))
        print("   ALL s+h p0 quartiles for contrast: p25=%.1f p50=%.1f p75=%.1f | frac p50=%.3f"%(
            q([e["p0"] for e in evs],25),q([e["p0"] for e in evs],50),q([e["p0"] for e in evs],75),q([e["p0"]/e["pl"] for e in evs],50)))
        for kk in ("soft","hard"):
            g=[e for e in cl if e["k"]==kk]
            if not g: print("   CLEARING x %-4s : n=0"%kk); continue
            print("   CLEARING x %-4s : n=%d succ=%.3f  med retract=%6.2f  med slack_rel=%6.2f  med slack_load=%6.2f  med adv25=%6.2f  med adv_end=%7.2f  med fold_load=%4.1f  med rel/retract=%.3f"%(
                kk,len(g),sum(1 for e in g if e["succ"])/len(g),q([e["retract"] for e in g],50),q([e["slack_rel"] for e in g],50),
                q([e["slack_load"] for e in g],50),q([e["adv25"] for e in g],50),q([e["adv_end"] for e in g],50),
                q([float(e["fold_load"]) for e in g],50),q([e["slack_rel"]/max(e["retract"],1e-9) for e in g],50)))
    for kk in ("soft","hard"):
        g=[e for e in evs if e["k"]==kk]
        if not g: continue
        cg=collections.Counter(lab(e) for e in g)
        print("   %-4s n=%d -> CLEARING %d (%.1f%%) FUTILE %d COSMETIC %d ; med retract=%.2f"%(
            kk,len(g),cg["CLEARING"],100.*cg["CLEARING"]/len(g),cg["FUTILE"],cg["COSMETIC"],q([e["retract"] for e in g],50)))
    return c

RUNS=[]
A=load("tr_A.jsonl");      RUNS.append(("A ckpt2002292 HOST (own anatomy, 74/98=75.5%)",A,features(S+"tr_A.jsonl")))
A5=load("tr_A514.jsonl");  RUNS.append(("A ckpt514264 HOST (own anatomy, 71/98=72.4%)",A5,features(S+"tr_A514.jsonl")))
TB=load("tr_ATB.jsonl");   evTB=features(S+"tr_ATB.jsonl")
RUNS.append(("A ckpt2002292 TOPBRAIN all-22 (foreign, 165/220=75.0%)",TB,evTB))
RUNS.append(("A ckpt2002292 TOPBRAIN SHARED (pl<=166.91)",[e for e in TB if e["pl"]<=166.91],[e for e in evTB if e["pl"]<=166.91]))
RUNS.append(("A ckpt2002292 TOPBRAIN GRAFTED (pl>166.91)",[e for e in TB if e["pl"]>166.91],[e for e in evTB if e["pl"]>166.91]))
for t,eps,ev in RUNS: slice_report(t,eps,ev); print()
