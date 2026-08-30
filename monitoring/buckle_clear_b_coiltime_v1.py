"""Was the coil ALREADY present when a CLEARING event fired, and did the event reduce it?"""
import json,sys
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
for path in sys.argv[1:]:
    name=path.split("trB_")[-1].replace(".jsonl","")
    eps=[json.loads(l) for l in open(path)]
    ev=features(path); idx=[]
    for i,ep in enumerate(eps):
        for e in ep["events"]:
            if e["k"] in ("soft","hard"): idx.append((i,e))
    for e,(i,raw) in zip(ev,idx): e["epi"]=i; e["raw"]=raw
    print("="*140); print("### %s"%name)
    print("  %-6s %-8s %-4s | slackGw@a  cath@a | peak_before_a | slackGw@b  cath@b | slack drop a->b   cath drop a->b | epMaxGw epMaxCath | adv25  succ"%("ep","label","kind"))
    for e in sorted(ev,key=lambda x:(x["epi"])):
        i=e["epi"]; ep=eps[i]; n=len(ep["proj"])
        S=[g-p for g,p in zip(ep["gw"],ep["proj"])]; CS=ep["cs"]
        a=max(0,min(e["raw"]["first"]-1,n-1)); b=(e["raw"]["close"]-1) if e["raw"]["close"]>0 else n-1
        b=max(a,min(b,n-1))
        if max(S)<=100.0 and (not CS or max(CS)<=50.0): continue
        print("  %-6d %-8s %-4s | %9.1f %7.1f | %13.1f | %9.1f %7.1f | %15.1f %16.1f | %7.1f %9.1f | %6.1f %d"%(
            i,lab(e),e["k"],S[a],CS[a] if CS else 0.0,max(S[:a+1]),S[b],CS[b] if CS else 0.0,
            S[a]-S[b],(CS[a]-CS[b]) if CS else 0.0,max(S),max(CS) if CS else 0.0,e["adv25"],e["succ"]))
    # aggregate: among coiled episodes, of events firing while ALREADY coiled (slack_gw@a>100 or cath@a>50)
    tot=cl=0
    for e in ev:
        i=e["epi"]; ep=eps[i]; n=len(ep["proj"])
        S=[g-p for g,p in zip(ep["gw"],ep["proj"])]; CS=ep["cs"]
        a=max(0,min(e["raw"]["first"]-1,n-1))
        if S[a]>100.0 or (CS and CS[a]>50.0):
            tot+=1; cl+= (lab(e)=="CLEARING")
    print("  events firing while the wire is ALREADY coiled (slack_gw>100 or cath_slack>50 at step a): n=%d, CLEARING=%d (%.3f)"%(tot,cl,cl/max(1,tot)))
