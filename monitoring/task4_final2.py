import json, os, math, collections
MON=r"D:/Arjun/workspace/neve/monitoring"
SP=r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad"
OFF=33.314; SEAM=166.91
GE=json.load(open(os.path.join(MON,"t4ak_geo2.json"))); geo=GE["geo"]
IN=json.load(open(os.path.join(MON,"t4ak_input.json"))); h0=IN["h0"]
teach=json.load(open(os.path.join(SP,"teach_20260828_045651.json")))

print("### TEACHER 98-EPISODE RUN: the 8 failures on the same 4 anatomies")
print("  mesh  ep/pid  path_len  tgt_s   maxProj_s  s_rcca_max  fb  nsteps")
for r in sorted(teach,key=lambda r:(r["mesh"],r["path_len"])):
    if r["succ"]: continue
    print("  %s %3d/%-4s %7.1f %7.1f %9.1f %10.1f  %-5s %5d"
          %(r["mesh"][-5:],r["ep"],r["pid"],r["path_len"],r["path_len"]-OFF,r["maxps"],r["maxps"]-OFF,str(r["fb"]),r["n"]))
nfail_ost=sum(1 for r in teach if not r["succ"] and r["maxps"]-OFF < 10)
print("  teacher failures arrested within 10 mm of the RCCA ostium: %d of %d"%(nfail_ost,sum(1 for r in teach if not r["succ"])))

def flags(fp,ts):
    g=geo[fp]
    wm=[x for x in g["runs"]["wire"] if not x["terminal"]]
    cm=[x for x in g["runs"]["cath"]]
    return dict(wire=bool(wm and ts>wm[0]["s0"]),
                cath=bool(cm and ts>cm[0]["s0"]),
                trim4=bool(ts>g["L"]-4.0))
def ex_none(r): return False
def ex_cath(r): return flags(r["mesh"],r["path_len"]-OFF)["cath"]
def ex_ct(r):
    f=flags(r["mesh"],r["path_len"]-OFF); return f["cath"] or f["trim4"]

def fisher(a,b,c,d):
    n=a+b+c+d; r1=a+b; c1=a+c
    p=lambda x: math.comb(r1,x)*math.comb(n-r1,c1-x)/math.comb(n,c1)
    lo=max(0,c1-(n-r1)); hi=min(r1,c1); p0=p(a)
    return min(1.0,sum(p(x) for x in range(lo,hi+1) if p(x)<=p0*(1+1e-9)))

def band(pl):
    return "167-200" if pl<=200 else ("200-240" if pl<=240 else ">240")

print()
print("### STANDARDISATION, past seam only (%.2f mm)"%SEAM)
for exname,ex in (("no exclusion",ex_none),("catheter-0.35",ex_cath),("cath OR distal-4mm",ex_ct)):
    H=[r for r in h0 if r["path_len"]>SEAM and not ex(r)]
    T=[r for r in teach if r["path_len"]>SEAM and not ex(r)]
    hs=sum(r["succ"] for r in H); ts=sum(r["succ"] for r in T)
    print(" --- exclusion=%s ; crude: teacher %d/%d=%.1f%%  H0 %d/%d=%.1f%%  p=%.4f"
          %(exname,ts,len(T),100*ts/len(T),hs,len(H),100*hs/len(H),fisher(ts,len(T)-ts,hs,len(H)-hs)))
    for keyname,keyf in (("anatomy",lambda r:r["mesh"]),("depth band",lambda r:band(r["path_len"])),
                         ("anatomy x band",lambda r:(r["mesh"],band(r["path_len"])))):
        SH=collections.defaultdict(lambda:[0,0]); ST=collections.defaultdict(lambda:[0,0])
        for r in H: k=keyf(r); SH[k][1]+=1; SH[k][0]+=r["succ"]
        for r in T: k=keyf(r); ST[k][1]+=1; ST[k][0]+=r["succ"]
        cells=[k for k in set(SH)|set(ST) if SH.get(k,[0,0])[1] and ST.get(k,[0,0])[1]]
        drop=[k for k in set(SH)|set(ST) if k not in cells]
        w=sum(SH[k][1]+ST[k][1] for k in cells)
        hh=sum((SH[k][1]+ST[k][1])*SH[k][0]/SH[k][1] for k in cells)
        tt=sum((SH[k][1]+ST[k][1])*ST[k][0]/ST[k][1] for k in cells)
        # Mantel-Haenszel odds ratio + CMH chi2
        num=den=0.0; E=V=Osum=0.0
        for k in cells:
            a=ST[k][0]; b=ST[k][1]-a; c=SH[k][0]; d=SH[k][1]-c; n=a+b+c+d
            num+=a*d/n; den+=b*c/n
            r1=a+b; r2=c+d; c1=a+c; c2=b+d
            Osum+=a; E+=r1*c1/n
            if n>1: V+=r1*r2*c1*c2/(n*n*(n-1))
        mh=(num/den) if den>0 else float("inf")
        chi=((abs(Osum-E)-0.5)**2)/V if V>0 else float("nan")
        # chi2 p, 1 df
        pv=math.erfc(math.sqrt(chi/2)) if chi==chi else float("nan")
        print("     standardise by %-15s cells=%d (dropped %d, n_dropped_eps=%d)  ->  teacher=%.1f%%  H0=%.1f%%  diff=%+.1f pp | MH_OR=%.2f  CMH p=%.4f"
              %(keyname,len(cells),len(drop),sum(SH.get(k,[0,0])[1]+ST.get(k,[0,0])[1] for k in drop),
                100*tt/w,100*hh/w,100*(tt-hh)/w,mh,pv))

print()
print("### ALL-DEPTH standardisation (anatomy mix only, all 98+98)")
for exname,ex in (("no exclusion",ex_none),("catheter-0.35",ex_cath)):
    H=[r for r in h0 if not ex(r)]; T=[r for r in teach if not ex(r)]
    SH=collections.defaultdict(lambda:[0,0]); ST=collections.defaultdict(lambda:[0,0])
    for r in H: SH[r["mesh"]][1]+=1; SH[r["mesh"]][0]+=r["succ"]
    for r in T: ST[r["mesh"]][1]+=1; ST[r["mesh"]][0]+=r["succ"]
    cells=sorted(set(SH)&set(ST)); w=sum(SH[k][1]+ST[k][1] for k in cells)
    hh=sum((SH[k][1]+ST[k][1])*SH[k][0]/SH[k][1] for k in cells)
    tt=sum((SH[k][1]+ST[k][1])*ST[k][0]/ST[k][1] for k in cells)
    hs=sum(r["succ"] for r in H); ts=sum(r["succ"] for r in T)
    print("  %-14s crude teacher %d/%d=%.1f%% H0 %d/%d=%.1f%% | anatomy-standardised teacher=%.1f%% H0=%.1f%% diff=%+.1f pp"
          %(exname,ts,len(T),100*ts/len(T),hs,len(H),100*hs/len(H),100*tt/w,100*hh/w,100*(tt-hh)/w))
