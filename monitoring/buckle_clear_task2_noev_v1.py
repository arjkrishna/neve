"""What do the no-soft/hard episodes look like on host vs grafted?"""
import json,sys,collections
S="C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad/"
def load(p): return [json.loads(l) for l in open(S+p)]
A=load("tr_A.jsonl"); TB=load("tr_ATB.jsonl")
for t,eps in [("HOST ck2002292",A),("TB GRAFTED",[e for e in TB if e["pl"]>166.91]),
              ("TB SHARED",[e for e in TB if e["pl"]<=166.91])]:
    buck=collections.defaultdict(lambda:[0,0])
    for ep in eps:
        ks=set(x["k"] for x in ep["events"])
        if "soft" in ks or "hard" in ks: g="has soft/hard"
        elif "unrec" in ks: g="unrec only (never retracted out)"
        elif "grind" in ks: g="grind only"
        else: g="no stall at all"
        buck[g][0]+=1; buck[g][1]+=ep["succ"]
    print(" %-15s"%t)
    for g in ("has soft/hard","unrec only (never retracted out)","grind only","no stall at all"):
        n,s=buck[g]
        if n: print("    %-34s n=%3d succ=%.3f (%d/%d)  med steps=%.0f"%(g,n,s/n,s,n,
            sorted(len(e["proj"]) for e in eps if (
              ("soft" in set(x["k"] for x in e["events"]) or "hard" in set(x["k"] for x in e["events"])) if g=="has soft/hard"
              else ("unrec" in set(x["k"] for x in e["events"]) and not({"soft","hard"}&set(x["k"] for x in e["events"]))) if g.startswith("unrec")
              else ("grind" in set(x["k"] for x in e["events"]) and not({"soft","hard","unrec"}&set(x["k"] for x in e["events"]))) if g=="grind only"
              else not e["events"]))[n//2]))
    # failure reasons
    rs=collections.Counter(ep["reason"] for ep in eps if not ep["succ"])
    print("    fail reasons: %s"%dict(rs))
