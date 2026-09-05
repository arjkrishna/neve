import os
"""HOME-vs-FOREIGN, buckle-load-normalised. TopBrain column: A(foreign) vs B(home) vs heuristic."""
import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from buckle_clear_classify_v1 import features,q
from buckle_clear_final_v1 import lab
SP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","saved","stuck")  # stall extracts live in-repo (see saved/stuck/README.md)
CELLS=[("HOST  A ckpt2002292","tr_A.jsonl",None),("HOST  A ckpt514264","tr_A514.jsonl",None),
       ("HOST  B ck505230","trB_505230_host.jsonl",None),("HOST  B ck256370","trB_256370_host.jsonl",None),
       ("HOST  heuristic ck0","tr_H0.jsonl",None),
       ("TB    A ckpt2002292 all22","tr_ATB.jsonl",None),
       ("TB    A ckpt2002292 GRAFTED","tr_ATB.jsonl","graft"),
       ("TB    B own eval2","trB_own_eval2.jsonl",None),
       ("TB    B own eval3","trB_own_eval3.jsonl",None),
       ("TB    B own explore","trB_own_explore.jsonl",None),
       ("TB    heuristic own eval1","trB_own_eval1.jsonl",None)]
print("%-28s %6s %6s %7s %7s | %7s %7s %7s | %9s %9s"%("cell","eps","succ","steps","fold>=4","s+h","bkl","CLR","CLR/1k","CLR/1k-BL"))
for tag,f,filt in CELLS:
    p=os.path.join(SP,f)
    eps=[json.loads(l) for l in open(p)]
    if filt=="graft": eps=[e for e in eps if e["pl"]>166.91]
    keep=set(id(e) for e in eps); seeds=None
    if filt=="graft": seeds=set((e["seed"],round(e["pl"],3)) for e in eps)
    nst=sum(len(e["proj"]) for e in eps)
    bl=sum(sum(1 for x in e["fold"] if x>=4) for e in eps)
    ns=sum(1 for e in eps if e["succ"])
    ev=features(p)
    if filt=="graft":
        ok=set(e["seed"] for e in eps); ev=[x for x in ev if x["seed"] in ok]
    for x in ev: x["lab"]=lab(x)
    B=[x for x in ev if x["lab"]!="COSMETIC"]; C=[x for x in ev if x["lab"]=="CLEARING"]
    print("%-28s %6d %6d %7d %6.2f%% | %7d %7d %7d | %9.3f %9.3f"%(tag,len(eps),ns,nst,100.0*bl/nst,
      len(ev),len(B),len(C),1000.0*len(C)/nst,(1000.0*len(C)/bl) if bl else float('nan')))
