import json,os,math
SCR=r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-workspace-neve\81b186b6-3a3f-4f63-8491-2172316ef81f\scratchpad"
D=json.load(open(os.path.join(SCR,"ck15_defects.json")))
dn=json.load(open(os.path.join(SCR,"ck15_donors.json"))); LOW,SIP=dn["low"],dn["sip"]
names=sorted(LOW)
def fisher2(a,b,c,d):  # two-sided
    n=a+b+c+d; r1=a+b; c1=a+c
    def pmf(x): return math.comb(r1,x)*math.comb(n-r1,c1-x)/math.comb(n,c1)
    p0=pmf(a); lo=max(0,c1-(n-r1)); hi=min(r1,c1)
    return sum(pmf(x) for x in range(lo,hi+1) if pmf(x)<=p0*1.0000001)
left=[n for n in names if LOW[n].endswith("_left")]; right=[n for n in names if not LOW[n].endswith("_left")]
print("left-lower composites %d, right-lower %d"%(len(left),len(right)))
print("%-26s %-14s %-14s %s"%("defect","left rate","right rate","Fisher 2-sided p (composite-level, NOT donor-corrected)"))
for d in sorted(D):
    if d=="LOWER_LEFT_UNMIRRORED": continue
    K=sum(D[d].values())
    if K<8: continue
    a=sum(D[d][n] for n in left); c=sum(D[d][n] for n in right)
    p=fisher2(a,len(left)-a,c,len(right)-c)
    star="  <<<" if p<0.05/24 else ""
    print("%-26s %3d/%3d=%.3f  %3d/%3d=%.3f  p=%.4f%s"%(d,a,len(left),a/len(left),c,len(right),c/len(right),p,star))
