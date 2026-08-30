import json, sys
ev=[json.loads(l) for l in open(sys.argv[1])]
sh=[e for e in ev if e["k"] in ("soft","hard")]
def q(v,p):
    v=sorted(v); i=(len(v)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(v)-1); f=i-lo
    return v[lo]*(1-f)+v[hi]*f
def desc(n,v):
    if not v: print("%-16s n=0"%n); return
    print("%-16s n=%-4d min=%8.2f p10=%7.2f p25=%7.2f med=%7.2f p75=%7.2f p90=%7.2f max=%8.2f"%(n,len(v),min(v),q(v,10),q(v,25),q(v,50),q(v,75),q(v,90),max(v)))
print("### SOFT+HARD n=%d"%len(sh))
for f in ["slack_base","slack_rise","gw_fed","slack_release","slack_resid","fold_ge5","fold_ge10","fold_ge20"]:
    desc(f,[e[f] for e in sh if e.get(f) is not None])
def hist(name,vals,edges):
    print("\n%s histogram (n=%d)"%(name,len(vals)))
    for i in range(len(edges)-1):
        c=sum(1 for v in vals if edges[i]<=v<edges[i+1])
        print("  [%7.1f,%7.1f) %4d %s"%(edges[i],edges[i+1],c,"#"*c))
    c=sum(1 for v in vals if v>=edges[-1]); print("  [%7.1f,   inf) %4d %s"%(edges[-1],c,"#"*c))
hist("adv_25",[e["adv_25"] for e in sh],[0,1,2,3,4,5,7.5,10,15,20,30,40,50,60])
hist("adv_end",[e["adv_end"] for e in sh],[0,2,4,6,8,10,15,20,30,40,60,80,100])
hist("slack_rise",[e["slack_rise"] for e in sh],[0,1,2,3,4,5,6,8,10,15,20,30])
hist("fold_max",[float(e["fold_max"]) for e in sh],[0,1,2,3,4,5,6,8,10,15,20])
hist("slack_release",[e["slack_release"] for e in sh],[-5,0,2,4,6,8,10,15,20,30,50])
