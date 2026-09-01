import json,os,re
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
dn=json.load(open(os.path.join(SCR,"ck15_donors.json"))); LOW,SIP=dn["low"],dn["sip"]
runs={}
for line in open(os.path.join(SCR,"check6a.txt")):
    if not line.startswith("ROW,"): continue
    f=line.strip().split(","); nm=f[1]; L=float(f[2])
    for seg in f[6:]:
        p=seg.split("|")
        if p[0]!="cath": continue
        rr=[]
        if p[3]!="-":
            for x in p[3].split("+"):
                m=re.match(r"([\d.]+)@([\d.]+)-([\d.]+)#(\d+)",x); rr.append((float(m.group(2)),int(m.group(4))))
        runs[nm]={"L":L,"runs":rr,"termonly":(int(p[4]) if p[4]!="-" else 1)}
# for each siphon donor, distance-from-terminus of the most-proximal MID run
g=defaultdict(list)
for nm,r in runs.items():
    if r["termonly"]==1 or not r["runs"]: continue
    mid=[r["L"]-s for s,_ in r["runs"] if (r["L"]-s)>8.0]
    if mid: g[SIP[nm]].append((nm,max(mid)))
print("Within-siphon-donor consistency of the deepest mid-vessel blockage position (mm from RCCA terminus):")
print("%-18s %2s  %-28s spread"%("siphon","n","values"))
sp=[]
for s,v in sorted(g.items(),key=lambda kv:-len(kv[1])):
    vals=sorted(x[1] for x in v)
    if len(vals)>=3:
        spread=vals[-1]-vals[0]; sp.append(spread)
        print("%-18s %2d  %-28s %.2f"%(s,len(vals),", ".join("%.1f"%x for x in vals[:6]),spread))
print("median within-donor spread over donors with n>=3 mid-vessel composites: %.2f mm"%sorted(sp)[len(sp)//2])
gl=defaultdict(list)
for nm,r in runs.items():
    if r["termonly"]==1 or not r["runs"]: continue
    mid=[r["L"]-s for s,_ in r["runs"] if (r["L"]-s)>8.0]
    if mid: gl[LOW[nm]].append(max(mid))
spl=[max(v)-min(v) for v in gl.values() if len(v)>=3]
print("same, grouped by LOWER donor (n>=3): median spread %.2f mm over %d donors"%(sorted(spl)[len(spl)//2],len(spl)))
