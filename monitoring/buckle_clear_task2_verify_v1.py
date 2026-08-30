"""Verify canon reproduction on the four Task-2 run slices."""
import json,sys,collections
S="C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
def load(p):
    return [json.loads(l) for l in open(S+p)]
def canon(tag,eps):
    n=len(eps); succ=sum(1 for e in eps if e["succ"]); steps=sum(len(e["proj"]) for e in eps)
    c=collections.Counter()
    for e in eps:
        for ev in e["events"]: c[ev["k"]]+=1
    tot=sum(c.values()); res=tot-c["unrec"]
    mx=max([ev["r"] for e in eps for ev in e["events"]] or [0])
    print("%-34s eps=%3d succ=%3d (%5.1f%%) steps=%6d  events=%3d = %5.2f/1k  g/s/h/u = %d/%d/%d/%d = %.0f/%.0f/%.0f/%.0f%%  resolved %d/%d=%.1f%%  maxretract=%.2f"%(
      tag,n,succ,100.*succ/n,steps,tot,1000.*tot/steps,c["grind"],c["soft"],c["hard"],c["unrec"],
      100.*c["grind"]/tot,100.*c["soft"]/tot,100.*c["hard"]/tot,100.*c["unrec"]/tot,res,tot,100.*res/tot,mx))
A=load("tr_A.jsonl"); A514=load("tr_A514.jsonl"); ATB=load("tr_ATB.jsonl")
canon("A ckpt2002292 HOST (exp 74/98,6.07,32/43/15/10,90.4)",A)
canon("A ckpt514264 HOST (exp 71/98,3.71,20/14/35/31,69.4)",A514)
canon("A ckpt2002292 TOPBRAIN all22 (exp165/220,3.39,53/13/10/24,77.0)",ATB)
pls=sorted(set(round(e["pl"],2) for e in ATB))
print("\nTopBrain distinct path_len values (n=%d): %s"%(len(pls),pls))
SH=[e for e in ATB if e["pl"]<=166.91]; GR=[e for e in ATB if e["pl"]>166.91]
canon("  TB SHARED  (pl<=166.91)",SH)
canon("  TB GRAFTED (pl> 166.91)",GR)
print("\nHOST path_len distinct: %s"%sorted(set(round(e['pl'],2) for e in A)))
