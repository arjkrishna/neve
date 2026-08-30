"""COIL CELL deep dive for family B: does a coiled episode ever get un-coiled and cleared?"""
import json,sys,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
for path in sys.argv[1:]:
    name=path.split("trB_")[-1].replace(".jsonl","")
    eps=[json.loads(l) for l in open(path)]
    ev=features(path); idx=[]
    for i,ep in enumerate(eps):
        for e in ep["events"]:
            if e["k"] in ("soft","hard"): idx.append(i)
    for e,i in zip(ev,idx): e["epi"]=i
    S=[[g-p for g,p in zip(e["gw"],e["proj"])] for e in eps]
    mcs=[max(e["cs"]) if e["cs"] else 0.0 for e in eps]; mgs=[max(s) for s in S]
    coil=[i for i in range(len(eps)) if mcs[i]>50.0 or mgs[i]>100.0]
    cset=set(coil)
    print("="*130); print("### %s  coiled=%d/%d (%.1f%%)"%(name,len(coil),len(eps),100.*len(coil)/len(eps)))
    # split contributions
    a=sum(1 for i in coil if mcs[i]>50.0); b=sum(1 for i in coil if mgs[i]>100.0)
    print("  by criterion: cath_slack>50mm %d ; slack_gw>100mm %d ; both %d"%(a,b,sum(1 for i in coil if mcs[i]>50 and mgs[i]>100)))
    # UN-COIL: does slack ever come back down below threshold after the peak?
    unc=0; unc_s=0
    for i in coil:
        s=S[i]; cs=eps[i]["cs"]; j=max(range(len(s)),key=lambda k:s[k])
        jc=max(range(len(cs)),key=lambda k:cs[k]) if cs else 0
        after_gw=min(s[j:]) if j<len(s) else s[-1]
        after_cs=min(cs[jc:]) if cs and jc<len(cs) else 0.0
        ok=(after_gw<=100.0 and after_cs<=50.0)
        unc+=ok; unc_s+=(ok and eps[i]["succ"])
    print("  UN-COILED (slack falls back below BOTH 100mm gw and 50mm cath after its peak): %d/%d = %.3f, of which succeed %d"%(unc,len(coil),unc/len(coil),unc_s))
    print("  coiled episodes graded SUCCESS: %d/%d = %.3f"%(sum(1 for i in coil if eps[i]["succ"]),len(coil),sum(1 for i in coil if eps[i]["succ"])/len(coil)))
    cev=[e for e in ev if e["epi"] in cset]
    ccl={e["epi"] for e in cev if lab(e)=="CLEARING"}
    print("  CLEARING events inside coiled eps: %d ; coiled eps with >=1 CLEARING: %d/%d = %.3f (succeed %d/%d)"%(
        sum(1 for e in cev if lab(e)=="CLEARING"),len(ccl),len(coil),len(ccl)/len(coil),sum(1 for i in ccl if eps[i]["succ"]),len(ccl)))
    rest=[i for i in coil if i not in ccl]
    print("  coiled eps with NO CLEARING: %d succ=%.3f"%(len(rest),sum(1 for i in rest if eps[i]["succ"])/max(1,len(rest))))
    # per-CLEARING-event detail inside coil
    print("  per-event detail, coiled episodes (all soft/hard):")
    for e in sorted(cev,key=lambda x:(lab(x),-x["adv25"])):
        i=e["epi"]; s=S[i]
        print("    ep#%-4d seed=%-8s %-4s %-8s retract=%7.2f fold_load=%3d fold_close=%2d slack_load=%8.2f slack_rel=%8.2f adv25=%7.2f p0=%6.1f restall=%d epMaxGw=%7.1f epMaxCath=%7.1f succ=%d"%(
            i,e["seed"],e["k"],lab(e),e["retract"],e["fold_load"],e["fold_close"],e["slack_load"],e["slack_rel"],e["adv25"],e["p0"],e["restall_same"],mgs[i],mcs[i],e["succ"]))
    # arrest-station concentration
    p0s=collections.Counter(round(e["p0"],1) for ep in eps for e in ep["events"])
    tot=sum(p0s.values())
    print("  arrest-station concentration (all canon stalls, n=%d): top5 %s"%(tot,[(k,v,"%.1f%%"%(100.*v/tot)) for k,v in p0s.most_common(5)]))
    lab_at=collections.Counter()
    for e in ev:
        lab_at[(round(e["p0"],1)==141.2,lab(e))]+=1
    print("  soft/hard events AT p0=141.2: CLEARING=%d FUTILE=%d COSMETIC=%d | elsewhere: CLEARING=%d FUTILE=%d COSMETIC=%d"%(
        lab_at[(True,"CLEARING")],lab_at[(True,"FUTILE")],lab_at[(True,"COSMETIC")],
        lab_at[(False,"CLEARING")],lab_at[(False,"FUTILE")],lab_at[(False,"COSMETIC")]))
