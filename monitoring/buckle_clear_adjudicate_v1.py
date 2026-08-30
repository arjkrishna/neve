"""Five-way HOST adjudication: verbatim definition, soft/hard split, station-match, coil."""
import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from buckle_clear_classify_v1 import features,q,med
from buckle_clear_final_v1 import lab

SP=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
CELLS=[("A ckpt2002292","tr_A.jsonl"),("A ckpt514264","tr_A514.jsonl"),
       ("B ck505230","trB_505230_host.jsonl"),("B ck256370","trB_256370_host.jsonl"),
       ("heuristic ckpt0","tr_H0.jsonl")]

def epstats(path):
    ne=ns=nst=0; coil=0; coil_succ=0
    for line in open(path):
        ep=json.loads(line); ne+=1; ns+=int(bool(ep["succ"])); nst+=len(ep["proj"])
        mx=max(ep["gw"][j]-ep["proj"][j] for j in range(len(ep["proj"])))
        if mx>100.0: coil+=1; coil_succ+=int(bool(ep["succ"]))
    return ne,ns,nst,coil,coil_succ

def pr(v,p): return q(v,p) if v else float('nan')

print("="*150)
print("FIVE-WAY HOST COLUMN (98 identical seeds, same pinned real-patient surface). Definition applied VERBATIM.")
print("%-16s %5s %5s %6s | %4s %4s %4s %4s %4s | %7s %7s | %6s %6s | %6s %6s"%(
  "cell","eps","succ","steps","s+h","bkl","CLR","FUT","COS","CLR/1k","CLR/ep","%s+h","%bkl","P(s|C)","P(s|F)"))
ALL={}
for tag,f in CELLS:
    p=os.path.join(SP,f); ev=features(p); ne,ns,nst,coil,coilsucc=epstats(p)
    for e in ev: e["lab"]=lab(e)
    ALL[tag]=(ev,ne,ns,nst,coil,coilsucc)
    C=[e for e in ev if e["lab"]=="CLEARING"]; F=[e for e in ev if e["lab"]=="FUTILE"]
    O=[e for e in ev if e["lab"]=="COSMETIC"]; B=C+F
    pc=(sum(1 for e in C if e["succ"])/len(C)) if C else float('nan')
    pf=(sum(1 for e in F if e["succ"])/len(F)) if F else float('nan')
    print("%-16s %5d %5d %6d | %4d %4d %4d %4d %4d | %7.3f %7.3f | %5.1f%% %5.1f%% | %6.3f %6.3f"%(
      tag,ne,ns,nst,len(ev),len(B),len(C),len(F),len(O),1000.0*len(C)/nst,len(C)/ne,
      100.0*len(C)/max(1,len(ev)),100.0*len(C)/max(1,len(B)),pc,pf))

print()
print("="*150)
print("SOFT vs HARD, per cell: clearing rate WITHIN each class, and within buckle-present of each class")
print("%-16s | %-38s | %-38s"%("cell","SOFT (retract 1-8mm)","HARD (retract >8mm)"))
print("%-16s | %4s %4s %4s %7s %7s %7s | %4s %4s %4s %7s %7s %7s"%(
  "","n","bkl","CLR","%ofn","%ofbkl","P(succ)","n","bkl","CLR","%ofn","%ofbkl","P(succ)"))
for tag,_ in CELLS:
    ev=ALL[tag][0]; row=[tag]
    cells=[]
    for k in ("soft","hard"):
        g=[e for e in ev if e["k"]==k]; b=[e for e in g if e["lab"]!="COSMETIC"]; c=[e for e in g if e["lab"]=="CLEARING"]
        ps=(sum(1 for e in c if e["succ"])/len(c)) if c else float('nan')
        cells.append("%4d %4d %4d %6.1f%% %6.1f%% %7.3f"%(len(g),len(b),len(c),
          100.0*len(c)/max(1,len(g)),100.0*len(c)/max(1,len(b)),ps))
    print("%-16s | %s | %s"%(tag,cells[0],cells[1]))

print()
print("POOLED HOST (all five cells): soft vs hard")
for k in ("soft","hard"):
    g=[e for tag,_ in CELLS for e in ALL[tag][0] if e["k"]==k]
    b=[e for e in g if e["lab"]!="COSMETIC"]; c=[e for e in g if e["lab"]=="CLEARING"]
    f=[e for e in g if e["lab"]=="FUTILE"]; o=[e for e in g if e["lab"]=="COSMETIC"]
    print("  %-5s n=%3d  CLEARING %3d (%4.1f%% of n, %4.1f%% of buckle-present)  FUTILE %3d (%4.1f%%)  COSMETIC %3d (%4.1f%%)  P(succ|CLR)=%.3f"%(
      k,len(g),len(c),100.0*len(c)/len(g),100.0*len(c)/max(1,len(b)),len(f),100.0*len(f)/len(g),
      len(o),100.0*len(o)/len(g),(sum(1 for e in c if e["succ"])/len(c)) if c else float('nan')))

print()
print("="*150)
print("CLEARING EFFICACY (medians over CLEARING events): slack released, load, retraction, advance")
print("%-16s %4s | %8s %8s %8s | %8s %8s %8s %8s"%("cell","n","slack_rel","slack_load","retract","fold_load","adv25","adv50","adv_end"))
for tag,_ in CELLS:
    C=[e for e in ALL[tag][0] if e["lab"]=="CLEARING"]
    if not C: print("%-16s %4d | %8s"%(tag,0,"n/a")); continue
    print("%-16s %4d | %8.2f %8.2f %8.2f | %8.1f %8.2f %8.2f %8.2f"%(tag,len(C),
      pr([e["slack_rel"] for e in C],50),pr([e["slack_load"] for e in C],50),pr([e["retract"] for e in C],50),
      pr([float(e["fold_load"]) for e in C],50),pr([e["adv25"] for e in C],50),pr([e["adv50"] for e in C],50),
      pr([e["adv_end"] for e in C],50)))

print()
print("="*150)
print("STATION-MATCHED: arrest station p0 in [139.5,143.0] mm  (the shared 141.2 siphon station)")
print("%-16s %4s %4s %4s %4s %4s %7s %7s | rest-of-path %4s %4s %7s"%(
  "cell","s+h","bkl","CLR","FUT","COS","%CLRs+h","P(s|C)","s+h","CLR","%CLR"))
for tag,_ in CELLS:
    ev=ALL[tag][0]
    g=[e for e in ev if 139.5<=e["p0"]<=143.0]; o=[e for e in ev if not(139.5<=e["p0"]<=143.0)]
    b=[e for e in g if e["lab"]!="COSMETIC"]; c=[e for e in g if e["lab"]=="CLEARING"]
    f=[e for e in g if e["lab"]=="FUTILE"]; cs=[e for e in g if e["lab"]=="COSMETIC"]
    oc=[e for e in o if e["lab"]=="CLEARING"]
    ps=(sum(1 for e in c if e["succ"])/len(c)) if c else float('nan')
    print("%-16s %4d %4d %4d %4d %4d %6.1f%% %7.3f | %16d %4d %6.1f%%"%(
      tag,len(g),len(b),len(c),len(f),len(cs),100.0*len(c)/max(1,len(g)),ps,len(o),len(oc),100.0*len(oc)/max(1,len(o))))

print()
print("="*150)
print("COIL INCIDENCE, gw-only criterion (max slack_gw > 100 mm) -- the only cross-build-comparable measure")
print("%-16s %5s %6s %7s %8s %10s"%("cell","eps","coiled","coil%","coil_succ","noncoil_succ"))
for tag,_ in CELLS:
    ev,ne,ns,nst,coil,coilsucc=ALL[tag]
    print("%-16s %5d %6d %6.1f%% %8d %10.3f"%(tag,ne,coil,100.0*coil/ne,coilsucc,(ns-coilsucc)/max(1,ne-coil)))
