"""Reproduce the canon taxonomy ledger for each family-B cell from the trace dumps."""
import json,sys,collections
def q(v,p):
    if not v: return float('nan')
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f
for path in sys.argv[1:]:
    eps=[json.loads(l) for l in open(path)]
    steps=sum(len(e["proj"]) for e in eps); succ=sum(1 for e in eps if e["succ"])
    c=collections.Counter(); rs=[]
    for e in eps:
        for v in e["events"]:
            c[v["k"]]+=1
            if v["k"]!="unrec": rs.append(v["r"])
    tot=sum(c.values()); res=tot-c["unrec"]
    mix=[100.*c[k]/tot if tot else 0 for k in ("grind","soft","hard","unrec")]
    print("%-14s eps=%-5d succ=%-4d (%5.1f%%) steps=%-7d events=%-5d %5.2f/1k  g/s/h/u=%d/%d/%d/%d = %.0f/%.0f/%.0f/%.0f%%  resolved=%.1f%%  retract med=%.2f p90=%.2f max=%.2f"%(
        path.split("_")[-1].replace(".jsonl",""),len(eps),succ,100.*succ/len(eps),steps,tot,1000.*tot/max(1,steps),
        c["grind"],c["soft"],c["hard"],c["unrec"],mix[0],mix[1],mix[2],mix[3],100.*res/max(1,tot),
        q(rs,50),q(rs,90),max(rs) if rs else float('nan')))
