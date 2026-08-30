"""Adjudication addendum: buckle exposure, coil-vs-attempt, conversion ratio, episode-level."""
import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
SP=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
CELLS=[("A ckpt2002292","tr_A.jsonl"),("A ckpt514264","tr_A514.jsonl"),
       ("B ck505230","trB_505230_host.jsonl"),("B ck256370","trB_256370_host.jsonl"),
       ("heuristic ckpt0","tr_H0.jsonl")]
print("%-16s %7s %7s %8s %8s | %9s %9s | %8s"%("cell","bkl/1k","s+h/1k","stall/1k","unrec/1k","med rel/ret","p50 slack_rel","CLR/bkl"))
D={}
for tag,f in CELLS:
    p=os.path.join(SP,f); ev=features(p)
    for e in ev: e["lab"]=lab(e)
    eps=[json.loads(l) for l in open(p)]
    nst=sum(len(e["proj"]) for e in eps); ne=len(eps)
    allev=[x for e in eps for x in e["events"]]
    unrec=sum(1 for x in allev if x["k"]=="unrec")
    B=[e for e in ev if e["lab"]!="COSMETIC"]; C=[e for e in ev if e["lab"]=="CLEARING"]
    rr=[e["slack_rel"]/e["retract"] for e in C if e["retract"]>0]
    print("%-16s %7.3f %7.3f %8.3f %8.3f | %9s %9s | %7.1f%%"%(tag,
      1000.0*len(B)/nst,1000.0*len(ev)/nst,1000.0*len(allev)/nst,1000.0*unrec/nst,
      ("%.3f"%q(rr,50)) if rr else "n/a",("%.2f"%q([e["slack_rel"] for e in C],50)) if C else "n/a",
      100.0*len(C)/max(1,len(B))))
    D[tag]=(eps,ev)
print()
print("COILED EPISODES (max slack_gw>100mm): does the policy even ATTEMPT a retraction?")
print("%-16s %6s | %6s %6s %8s | %6s %6s %8s"%("cell","coiled","w/ s+h","w/ CLR","succ","no s+h","","succ"))
for tag,_ in CELLS:
    eps,ev=D[tag]
    bys={}
    for e in ev: bys.setdefault(e["seed"],[]).append(e)
    co=[e for e in eps if max(e["gw"][j]-e["proj"][j] for j in range(len(e["proj"])))>100.0]
    wsh=[e for e in co if bys.get(e["seed"])]
    wc=[e for e in co if any(x["lab"]=="CLEARING" for x in bys.get(e["seed"],[]))]
    no=[e for e in co if not bys.get(e["seed"])]
    fm=lambda g:(sum(1 for e in g if e["succ"])/len(g)) if g else float('nan')
    print("%-16s %6d | %6d %6d %8.3f | %6d %6s %8.3f"%(tag,len(co),len(wsh),len(wc),fm(wsh),len(no),"",fm(no)))
print()
print("EPISODE-LEVEL: P(success) by best label in the episode")
print("%-16s | %-18s %-18s %-18s %-18s"%("cell",">=1 CLEARING","only FUT/COS","no soft/hard ev","run-wide"))
for tag,_ in CELLS:
    eps,ev=D[tag]
    bys={}
    for e in ev: bys.setdefault(e["seed"],[]).append(e)
    g1=[e for e in eps if any(x["lab"]=="CLEARING" for x in bys.get(e["seed"],[]))]
    g2=[e for e in eps if bys.get(e["seed"]) and not any(x["lab"]=="CLEARING" for x in bys.get(e["seed"],[]))]
    g3=[e for e in eps if not bys.get(e["seed"])]
    fm=lambda g:"%.3f (%d/%d)"%(sum(1 for e in g if e["succ"])/len(g),sum(1 for e in g if e["succ"]),len(g)) if g else "n/a"
    print("%-16s | %-18s %-18s %-18s %-18s"%(tag,fm(g1),fm(g2),fm(g3),fm(eps)))
