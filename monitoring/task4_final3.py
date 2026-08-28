import json,os,math,collections
MON=r"D:/Arjun/workspace/neve/monitoring"
SP=r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad"
OFF=33.314; SEAM=166.91
geo=json.load(open(os.path.join(MON,"t4ak_geo2.json")))["geo"]
h0=json.load(open(os.path.join(MON,"t4ak_input.json")))["h0"]
teach=json.load(open(os.path.join(SP,"teach_20260828_045651.json")))
deep=collections.defaultdict(float)
for rows in (h0,teach):
    for r in rows:
        if r["succ"]: deep[r["mesh"]]=max(deep[r["mesh"]],r["path_len"])
def ex_demo(r):
    g=geo[r["mesh"]]; ts=r["path_len"]-OFF
    return (ts > g["L"]-4.0) and (r["path_len"] > deep[r["mesh"]])
def fisher(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    p=lambda x: math.comb(r1,x)*math.comb(n-r1,c1-x)/math.comb(n,c1)
    lo=max(0,c1-(n-r1)); hi=min(r1,c1); p0=p(a)
    return min(1.0,sum(p(x) for x in range(lo,hi+1) if p(x)<=p0*(1+1e-9)))
print("### PREFERRED CORRECTION: exclude only targets inside the distal 4 mm of the RCCA centerline")
print("### that no run has demonstrated reachable (guidewire 0.18 never blocked on these 4 anatomies)")
for tag,rows in (("H0",h0),("TEACHER",teach)):
    ex=[r for r in rows if ex_demo(r)]
    keep=[r for r in rows if not ex_demo(r)]
    print("  %-8s excluded %d: %s"%(tag,len(ex),[(r["mesh"][-5:],r["path_len"],"succ" if r["succ"] else "FAIL") for r in ex]))
    print("           all depths  raw %d/%d=%.1f%%  ->  corrected %d/%d=%.1f%%"
          %(sum(r["succ"] for r in rows),len(rows),100*sum(r["succ"] for r in rows)/len(rows),
            sum(r["succ"] for r in keep),len(keep),100*sum(r["succ"] for r in keep)/len(keep)))
H=[r for r in h0 if r["path_len"]>SEAM and not ex_demo(r)]
T=[r for r in teach if r["path_len"]>SEAM and not ex_demo(r)]
hs=sum(r["succ"] for r in H); ts=sum(r["succ"] for r in T)
print("  past-seam: teacher %d/%d=%.1f%%  H0 %d/%d=%.1f%%  diff=%+.1f pp  Fisher p=%.4f"
      %(ts,len(T),100*ts/len(T),hs,len(H),100*hs/len(H),100*ts/len(T)-100*hs/len(H),fisher(ts,len(T)-ts,hs,len(H)-hs)))
# anatomy-standardised past-seam under this correction
SH=collections.defaultdict(lambda:[0,0]); ST=collections.defaultdict(lambda:[0,0])
for r in H: SH[r["mesh"]][1]+=1; SH[r["mesh"]][0]+=r["succ"]
for r in T: ST[r["mesh"]][1]+=1; ST[r["mesh"]][0]+=r["succ"]
cells=sorted(set(SH)&set(ST)); w=sum(SH[k][1]+ST[k][1] for k in cells)
hh=sum((SH[k][1]+ST[k][1])*SH[k][0]/SH[k][1] for k in cells)
tt=sum((SH[k][1]+ST[k][1])*ST[k][0]/ST[k][1] for k in cells)
print("  past-seam anatomy-standardised: teacher=%.1f%%  H0=%.1f%%  diff=%+.1f pp"%(100*tt/w,100*hh/w,100*(tt-hh)/w))
# H0 arrest depth vs target depth
print()
print("### H0: is the arrest anywhere near the target? (23 failures)")
GE=json.load(open(os.path.join(MON,"t4ak_geo2.json")))["fails"]
short=[(f["tgt_s"]-f["s_rcca_max"]) for f in GE]
print("  shortfall (target s_rcca - deepest s_rcca reached), mm: min=%.1f med=%.1f max=%.1f"%(min(short),sorted(short)[len(short)//2],max(short)))
print("  failures arresting >30 mm short of target: %d/23 ; >100 mm short: %d/23"%(sum(1 for x in short if x>30),sum(1 for x in short if x>100)))
