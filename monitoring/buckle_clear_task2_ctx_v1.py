"""TASK 2 supplement: exposure-fair class rates, p0 band concentration, overshoot, wire efficiency."""
import json,sys,collections
sys.path.insert(0,"d:/Arjun/workspace/neve/monitoring")
from buckle_clear_classify_v1 import features,q
S="C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
def lab(e):
    if not (e["fold_load"]>=4 or e["slack_load"]>=10.0): return "COSMETIC"
    return "CLEARING" if (e["fold_close"]==0 and e["adv25"]>=15.0 and not e["restall_same"]) else "FUTILE"
def load(p): return [json.loads(l) for l in open(S+p)]
A=load("tr_A.jsonl"); A5=load("tr_A514.jsonl"); TB=load("tr_ATB.jsonl")
eA=features(S+"tr_A.jsonl"); eA5=features(S+"tr_A514.jsonl"); eTB=features(S+"tr_ATB.jsonl")
SL=[("HOST ck2002292",A,eA),("HOST ck514264",A5,eA5),
    ("TB all-22",TB,eTB),
    ("TB SHARED",[e for e in TB if e["pl"]<=166.91],[e for e in eTB if e["pl"]<=166.91]),
    ("TB GRAFTED",[e for e in TB if e["pl"]>166.91],[e for e in eTB if e["pl"]>166.91])]

print("="*132); print("EXPOSURE-FAIR RATES PER 1000 STEPS (all canon classes + buckle labels)")
print("%-16s %7s %7s %8s | %6s %6s %6s %6s %6s | %6s %6s %6s"%("slice","eps","steps","st/ep","grind","soft","hard","unrec","s+h","CLEAR","FUTIL","COSM"))
for t,eps,ev in SL:
    st=sum(len(e["proj"]) for e in eps); c=collections.Counter()
    for e in eps:
        for x in e["events"]: c[x["k"]]+=1
    L=collections.Counter(lab(e) for e in ev)
    f=lambda n: 1000.*n/st
    print("%-16s %7d %7d %8.1f | %6.2f %6.2f %6.2f %6.2f %6.2f | %6.3f %6.3f %6.3f"%(
      t,len(eps),st,st/len(eps),f(c["grind"]),f(c["soft"]),f(c["hard"]),f(c["unrec"]),f(len(ev)),
      f(L["CLEARING"]),f(L["FUTILE"]),f(L["COSMETIC"])))
print()
print("PER EPISODE")
print("%-16s | %6s %6s %6s %6s %6s | %6s %6s %6s"%("slice","grind","soft","hard","unrec","s+h","CLEAR","FUTIL","COSM"))
for t,eps,ev in SL:
    n=len(eps); c=collections.Counter()
    for e in eps:
        for x in e["events"]: c[x["k"]]+=1
    L=collections.Counter(lab(e) for e in ev); f=lambda k: k/n
    print("%-16s | %6.3f %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f"%(
      t,f(c["grind"]),f(c["soft"]),f(c["hard"]),f(c["unrec"]),f(len(ev)),f(L["CLEARING"]),f(L["FUTILE"]),f(L["COSMETIC"])))

print(); print("="*132); print("p0 BAND CONCENTRATION -- CLEARING rate per band (band = 20mm bins of arrest arclength p0)")
for t,eps,ev in SL:
    if not ev: continue
    b=collections.defaultdict(lambda:[0,0])
    for e in ev:
        k=int(e["p0"]//20)*20; b[k][0]+=1; b[k][1]+= (lab(e)=="CLEARING")
    print(" %s"%t)
    for k in sorted(b):
        n,cl=b[k]; bar="#"*cl
        print("   p0 %3d-%3d mm : s+h n=%3d  CLEARING %2d (%5.1f%%) %s"%(k,k+19,n,cl,100.*cl/n,bar))

print(); print("="*132); print("OVERSHOOT: retraction depth vs clearance bought (buckle-present events only)")
def corr(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    sx=(sum((a-mx)**2 for a in x)/n)**.5; sy=(sum((a-my)**2 for a in y)/n)**.5
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/(n*sx*sy) if sx*sy>0 else float('nan')
for t,eps,ev in SL:
    bp=[e for e in ev if e["fold_load"]>=4 or e["slack_load"]>=10.0]
    if len(bp)<4: print(" %-16s buckle-present n=%d -- too few"%(t,len(bp))); continue
    print(" %-16s buckle-present n=%d  corr(retract,slack_rel)=%+.3f  corr(retract,adv25)=%+.3f  corr(slack_rel,adv25)=%+.3f"%(
        t,len(bp),corr([e["retract"] for e in bp],[e["slack_rel"] for e in bp]),
        corr([e["retract"] for e in bp],[e["adv25"] for e in bp]),
        corr([e["slack_rel"] for e in bp],[e["adv25"] for e in bp])))
    for lo,hi,nm in [(1.0,8.0,"soft 1-8mm"),(8.0,20.0,"hard 8-20"),(20.0,1e9,"hard >20")]:
        g=[e for e in bp if lo<=e["retract"]<hi]
        if not g: continue
        print("    %-11s n=%2d  med retract=%7.2f  med slack_rel=%6.2f  eff rel/retract=%.3f  med slack_load=%6.2f  med adv25=%6.2f  CLEARING=%d (%.0f%%)  succ=%.3f"%(
            nm,len(g),q([e["retract"] for e in g],50),q([e["slack_rel"] for e in g],50),
            q([e["slack_rel"]/max(e["retract"],1e-9) for e in g],50),q([e["slack_load"] for e in g],50),
            q([e["adv25"] for e in g],50),sum(1 for e in g if lab(e)=="CLEARING"),
            100.*sum(1 for e in g if lab(e)=="CLEARING")/len(g),sum(1 for e in g if e["succ"])/len(g)))

print(); print("="*132); print("WIRE-TRAVEL EFFICIENCY (recomputed): net inserted_gw gain / total |delta inserted_gw|")
for t,eps,ev in SL:
    effs=[]
    for e in eps:
        g=e["gw"]
        if len(g)<2: continue
        tot=sum(abs(g[i+1]-g[i]) for i in range(len(g)-1))
        if tot<=0: continue
        effs.append((max(g)-g[0])/tot)
    print(" %-16s n=%3d  median efficiency=%.3f  mean=%.3f  p25=%.3f p75=%.3f"%(
        t,len(effs),q(effs,50),sum(effs)/len(effs),q(effs,25),q(effs,75)))
    for w in (True,False):
        sub=[]
        for e in eps:
            g=e["gw"]
            if len(g)<2: continue
            tot=sum(abs(g[i+1]-g[i]) for i in range(len(g)-1))
            if tot<=0: continue
            if e["succ"]==w: sub.append((max(g)-g[0])/tot)
        if sub: print("      %-7s n=%3d median=%.3f"%("success" if w else "fail",len(sub),q(sub,50)))
