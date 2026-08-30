"""Attempt-INDEPENDENT buckle exposure: fold occupancy over all steps, incl. unrec stalls."""
import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from buckle_clear_classify_v1 import q
SP=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
CELLS=[("A ckpt2002292","tr_A.jsonl"),("A ckpt514264","tr_A514.jsonl"),
       ("B ck505230","trB_505230_host.jsonl"),("B ck256370","trB_256370_host.jsonl"),
       ("heuristic ckpt0","tr_H0.jsonl")]
print("%-16s %8s %8s %8s %8s | %8s %8s | %10s %10s"%("cell","steps","%fold>=4","%fold>=10","%fold>=20",
      "eps f>=4","eps f>=20","unrec ev","unrec f>=4"))
for tag,f in CELLS:
    eps=[json.loads(l) for l in open(os.path.join(SP,f))]
    nst=n4=n10=n20=0; e4=e20=0; un=0; un4=0
    for ep in eps:
        fo=ep["fold"]; nst+=len(fo)
        n4+=sum(1 for x in fo if x>=4); n10+=sum(1 for x in fo if x>=10); n20+=sum(1 for x in fo if x>=20)
        m=max(fo) if fo else 0
        e4+=int(m>=4); e20+=int(m>=20)
        for ev in ep["events"]:
            if ev["k"]!="unrec": continue
            a=max(0,min(ev["first"]-1,len(fo)-1)); b=len(fo)-1
            un+=1; un4+=int(max(fo[a:b+1] or [0])>=4)
    print("%-16s %8d %7.2f%% %7.2f%% %7.2f%% | %7d %8d | %10d %10d"%(tag,nst,
      100.0*n4/nst,100.0*n10/nst,100.0*n20/nst,e4,e20,un,un4))
