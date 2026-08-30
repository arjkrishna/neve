"""Episode-level context, sensitivity sweep, and the COIL CELL for family B."""
import json,sys,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
def load(path):
    eps=[json.loads(l) for l in open(path)]
    ev=features(path)
    # features() preserves file order and per-episode event order -> index by soft/hard counts
    idx=[]; 
    for i,ep in enumerate(eps):
        for e in ep["events"]:
            if e["k"] in ("soft","hard"): idx.append(i)
    assert len(idx)==len(ev),(len(idx),len(ev))
    for e,i in zip(ev,idx): e["epi"]=i
    return eps,ev
def sens(ev,tag):
    def row(nm,**kw):
        c={}
        for e in ev: c.setdefault(lab(e,**kw),[]).append(e)
        o=[]
        for k in ("CLEARING","FUTILE","COSMETIC"):
            g=c.get(k,[]); sc=(sum(1 for x in g if x["succ"])/len(g)) if g else float('nan')
            o.append("%-5s%4d/%5.1f%%/%.3f"%(k[:4],len(g),100.*len(g)/len(ev),sc))
        cl,fu=c.get("CLEARING",[]),c.get("FUTILE",[])
        d=(sum(1 for x in cl if x["succ"])/len(cl)-sum(1 for x in fu if x["succ"])/len(fu)) if (cl and fu) else float('nan')
        print("  %-22s %s  sep=%+.3f"%(nm," | ".join(o),d))
    print("-- SENSITIVITY %s (n=%d) --"%(tag,len(ev)))
    for v in (2,3,4,5,6,8): row("F=%d"%v,F=v)
    for v in (6.,8.,10.,12.,15.,1e9): row("S=%s"%("off" if v>1e8 else "%g"%v),S=v)
    for v in (5.,10.,14.,15.,20.,25.,30.,40.): row("A=%g"%v,A=v)
    row("no REDUCED",use_reduced=False); row("no restall",use_restall=False)
    row("no REDUCED,no restall",use_reduced=False,use_restall=False)
for path in sys.argv[1:]:
    name=path.split("trB_")[-1].replace(".jsonl","")
    eps,ev=load(path)
    print("="*140); print("### %s   eps=%d  run-wide succ=%.3f"%(name,len(eps),sum(1 for e in eps if e["succ"])/len(eps)))
    byep=collections.defaultdict(list)
    for e in ev: byep[e["epi"]].append(e)
    with_ev=set(byep); no_ev=[i for i in range(len(eps)) if i not in with_ev]
    print("  episodes with >=1 soft/hard: n=%d succ=%.3f | with none: n=%d succ=%.3f"%(
        len(with_ev),sum(1 for i in with_ev if eps[i]["succ"])/max(1,len(with_ev)),
        len(no_ev),sum(1 for i in no_ev if eps[i]["succ"])/max(1,len(no_ev))))
    bp={i for i,g in byep.items() if any(e["fold_load"]>=4 or e["slack_load"]>=10 for e in g)}
    nb=[i for i in with_ev if i not in bp]
    print("  episodes with >=1 buckle-present: n=%d succ=%.3f | event-carrying no buckle: n=%d succ=%.3f"%(
        len(bp),sum(1 for i in bp if eps[i]["succ"])/max(1,len(bp)),len(nb),sum(1 for i in nb if eps[i]["succ"])/max(1,len(nb))))
    cnt=collections.Counter()
    for i,g in byep.items():
        ls=[lab(e) for e in g]
        top="CLEARING" if "CLEARING" in ls else ("FUTILE" if "FUTILE" in ls else "COSMETIC")
        cnt[(top,eps[i]["succ"])]+=1
    for t in ("CLEARING","FUTILE","COSMETIC"):
        a,b=cnt[(t,True)],cnt[(t,False)]
        print("  episode best soft/hard event = %-8s n=%3d succ=%.3f (%d/%d)"%(t,a+b,a/max(1,a+b),a,a+b))
    # ---- COIL CELL ----
    mcs=[max(e["cs"]) if e["cs"] else 0.0 for e in eps]
    mgs=[max(g-p for g,p in zip(e["gw"],e["proj"])) for e in eps]
    coil=[i for i in range(len(eps)) if mcs[i]>50.0 or mgs[i]>100.0]
    print("  COIL CELL: n=%d (%.1f%%) of %d episodes  [max cath_slack>50mm OR max slack_gw>100mm]"%(len(coil),100.*len(coil)/len(eps),len(eps)))
    print("     cath_slack per-episode max: p50=%.1f p90=%.1f max=%.1f | slack_gw per-episode max: p50=%.1f p90=%.1f max=%.1f"%(
        q(mcs,50),q(mcs,90),max(mcs),q(mgs,50),q(mgs,90),max(mgs)))
    if coil:
        cs_ok=sum(1 for i in coil if eps[i]["succ"])
        print("     coiled episodes graded SUCCESS: %d/%d = %.3f | non-coiled: %d/%d = %.3f"%(
            cs_ok,len(coil),cs_ok/len(coil),sum(1 for i in range(len(eps)) if i not in set(coil) and eps[i]["succ"]),
            len(eps)-len(coil),sum(1 for i in range(len(eps)) if i not in set(coil) and eps[i]["succ"])/max(1,len(eps)-len(coil))))
        cev=[e for e in ev if e["epi"] in set(coil)]
        c2=collections.Counter(lab(e) for e in cev)
        print("     soft/hard events inside coiled episodes: n=%d  CLEARING=%d FUTILE=%d COSMETIC=%d"%(len(cev),c2["CLEARING"],c2["FUTILE"],c2["COSMETIC"]))
        for k in ("CLEARING","FUTILE","COSMETIC"):
            g=[e for e in cev if lab(e)==k]
            if g: print("       %-8s n=%3d P(ep succ)=%.3f med adv25=%7.2f med retract=%7.2f med fold_load=%5.1f med slack_load=%7.2f med slack_rel=%8.2f med slack_resid=%8.2f"%(
                k,len(g),sum(1 for e in g if e["succ"])/len(g),q([e["adv25"] for e in g],50),q([e["retract"] for e in g],50),
                q([float(e["fold_load"]) for e in g],50),q([e["slack_load"] for e in g],50),q([e["slack_rel"] for e in g],50),q([e["slack_resid"] for e in g],50)))
        ncoil=[i for i in coil if i not in byep]
        print("     coiled episodes with NO soft/hard event at all: %d (succ %.3f)"%(len(ncoil),sum(1 for i in ncoil if eps[i]["succ"])/max(1,len(ncoil))))
        ccl={e["epi"] for e in cev if lab(e)=="CLEARING"}
        print("     coiled episodes containing >=1 CLEARING: %d/%d = %.3f, of which succeed %d"%(
            len(ccl),len(coil),len(ccl)/len(coil),sum(1 for i in ccl if eps[i]["succ"])))
        # slack reduction achieved inside coiled episodes
        red=[]
        for i in coil:
            s=[g-p for g,p in zip(eps[i]["gw"],eps[i]["proj"])]
            red.append(max(s)-s[-1])
        print("     slack_gw reduction from episode peak to final step, coiled eps: p50=%.1f p90=%.1f max=%.1f mm"%(q(red,50),q(red,90),max(red)))
        csr=[]
        for i in coil:
            s=eps[i]["cs"]
            if s: csr.append(max(s)-s[-1])
        if csr: print("     cath_slack reduction peak->final, coiled eps: p50=%.1f p90=%.1f max=%.1f mm"%(q(csr,50),q(csr,90),max(csr)))
    sens(ev,name)
