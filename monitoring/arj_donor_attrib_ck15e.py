import json,os,re,math
from collections import defaultdict
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
AN=r"D:\Arjun\workspace\neve\carotid_data\anatomies"
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json"))); LOW,SIP=dn["low"],dn["sip"]
names=sorted(LOW)
P={n:json.load(open(os.path.join(AN,n,"provenance.json"))) for n in names}
# parse check6a runs again
runs={}
for line in open(os.path.join(SCR,"check6a.txt")):
    if not line.startswith("ROW,"): continue
    f=line.strip().split(","); nm=f[1]; L=float(f[2])
    rec={"L":L}
    for seg in f[6:]:
        p=seg.split("|"); rr=[]
        if p[3]!="-":
            for x in p[3].split("+"):
                m=re.match(r"([\d.]+)@([\d.]+)-([\d.]+)#(\d+)",x)
                rr.append((float(m.group(2)),float(m.group(3)),int(m.group(4))))
        rec[p[0]]={"runs":rr,"termonly":(int(p[4]) if p[4]!="-" else 1)}
    runs[nm]=rec
seg_count=defaultdict(int); tot=0; segsum=defaultdict(int)
detail=[]
for nm in names:
    pr=P[nm]; s1=pr["host_cut_mm"]; s2=s1+pr["cca_mm"]+pr["ica_mm"]
    L=runs[nm]["L"]
    r=runs[nm]["cath"]
    if not (r["runs"] and r["termonly"]==0): continue
    for st,en,ns in r["runs"]:
        if (L-st)<=8.0: seg="TERMINAL(<=8mm)"
        elif st<s1: seg="HOST"
        elif st<s2: seg="LOWER(bifurc donor)"
        else: seg="SIPHON(topcow donor)"
        seg_count[seg]+=1; segsum[seg]+=ns; tot+=1
    detail.append((nm,s1,s2,L))
print("Where the 0.35 mm mid-vessel blocked RUNS sit, by source segment (composites with mid-vessel blockage only):")
for k,v in sorted(seg_count.items(),key=lambda kv:-kv[1]):
    print("   %-22s runs=%3d (%4.1f%%)  blocked stations=%4d"%(k,v,100*v/tot,segsum[k]))
print("   total runs",tot,"over",len(detail),"composites")
print("   seam2 (lower->siphon) arclength: min %.1f med %.1f max %.1f mm"%tuple(
   sorted(d[2] for d in detail)[i] for i in (0,len(detail)//2,-1)))
# partial vs deterministic donors
print("\nDonor-determinism: for each defect, how many donors are ALL-defective / PARTIAL / clean (donors with n>=3)")
print("%-26s %4s | %-24s | %-24s"%("defect","K","LOWER all/part/clean","SIPHON all/part/clean"))
for d in sorted(D):
    K=sum(D[d].values())
    if K<5 or K>150: continue
    line=[]
    for f in (LOW,SIP):
        g=defaultdict(lambda:[0,0])
        for nm in names: g[f[nm]][1]+=1; g[f[nm]][0]+=D[d][nm]
        a=sum(1 for k,v in g.items() if v[1]>=3 and v[0]==v[1])
        c=sum(1 for k,v in g.items() if v[1]>=3 and v[0]==0)
        pt=sum(1 for k,v in g.items() if v[1]>=3 and 0<v[0]<v[1])
        line.append("%2d / %2d / %2d"%(a,pt,c))
    print("%-26s %4d | %-24s | %-24s"%(d,K,line[0],line[1]))
