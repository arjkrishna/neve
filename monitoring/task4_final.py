import json, pickle, os, math, collections, statistics
SP=r"C:/Users/akrish41/AppData/Local/Temp/claude/d--Arjun-workspace-neve/81b186b6-3a3f-4f63-8491-2172316ef81f/scratchpad"
MON=r"D:/Arjun/workspace/neve/monitoring"
OFF=33.314; SEAM=166.91
GE=json.load(open(os.path.join(MON,"t4ak_geo2.json")))
IN=json.load(open(os.path.join(MON,"t4ak_input.json")))
h0=IN["h0"]; teach=IN["teach"]
FR={(f["pid"],f["ep"]):f for f in GE["fails"]}
geo=GE["geo"]

print("### A. GEOMETRY OF THE 4 HOLDOUT ANATOMIES (exact signed distance, 0.25 mm stations)")
for fp,g in geo.items():
    print("  %s  L_rcca=%.2f mm (path_len %.2f)  nCLpts_outside_surface=%d  s_last_inside=%.2f  min_clear=%.3f  openBoundaryEdges=%d"
          %(fp,g["L"],g["L"]+OFF,g["nout"],g["last_in"],g["minclear"],g["nopen"]))
    for t,thr in (("wire",0.18),("sofa",0.30),("cath",0.35)):
        rs=g["runs"][t]
        print("      thr=%.2f  runs=%s"%(thr,[(round(r["s0"],2),round(r["s1"],2),round(r["min_clear"],3),"TERM" if r["terminal"] else "MID") for r in rs]))

# ---------- failure mode refinement ----------
D=pickle.load(open(os.path.join(SP,"h0.pkl"),"rb")); eps,steps,snap=D["eps"],D["steps"],D["snap"]
print()
print("### B. H0 FAILURE MODES (last 150 steps)")
print("  mesh  ep/pid  plen  tgt_s | arrest_s_rcca | net_dproj tv_proj  net_dins | tipball | inRCCA_fin dR/rR | inRVA_fin sV maxSV | MODE")
MODE={}
for r in h0:
    if r["succ"]: continue
    k=(r["pid"],r["ep"]); f=FR[k]
    pj=r["projs"]; ins=[a[1] for a in r["ins"]]
    W=pj[-150:]; I=ins[-150:]
    net=W[-1]-W[0]; tv=sum(abs(W[i+1]-W[i]) for i in range(len(W)-1)); dins=I[-1]-I[0]
    if f["frac_out_R"]>0.90 and f["maxSV_inV"]>6.35:
        mode="wrong_branch_RVA_subclavian"
    elif net>2.0 and tv>0 and net/tv>0.25:
        mode="still_advancing_at_cap"
    elif tv>15.0:
        mode="oscillating_on_path"
    else:
        mode="arrested_on_path"
    MODE[k]=mode
    print("  %s %2d/%-4d %6.1f %6.1f | %8.1f | %8.1f %7.1f %8.1f | %6.2f | %5s %.2f/%.2f | %5s %6.2f %6.2f | %s"
          %(r["mesh"][-5:],r["ep"],r["pid"],r["path_len"],r["path_len"]-OFF,f["s_rcca_max"],
            net,tv,dins,f["tipmove100"],f["inR_fin"],f["dR_fin"],f["rR_fin"],f["inV_fin"],f["sV_fin"],f["maxSV_inV"],mode))
print("  MODE COUNTS:",collections.Counter(MODE.values()))
print("  final_branch label vs true excursion:")
tab=collections.Counter()
for r in h0:
    if r["succ"]: continue
    k=(r["pid"],r["ep"]); f=FR[k]
    true_rva = f["frac_out_R"]>0.90 and f["maxSV_inV"]>6.35
    tab[(r["fb"], "entered_RVA" if true_rva else "stayed_RCCA")]+=1
for kk,v in sorted(tab.items(),key=lambda x:str(x)): print("     final_branch=%s  %s : %d"%(kk[0],kk[1],v))
print("  tip inside mesh check: max n_steps_with_tip_outside_surface =",max(FR[k]["n_tip_outside"] for k in FR),
      " min signed dist (mm, +inside) =",round(min(FR[k]["sd_min"] for k in FR),3))

# ---------- reachability ----------
def wire_block_s(fp):
    m=[r for r in geo[fp]["runs"]["wire"] if not r["terminal"]]
    return m[0]["s0"] if m else None
def cath_block_s(fp):
    m=[r for r in geo[fp]["runs"]["cath"]]
    return m[0]["s0"] if m else None

def flags(fp, tgt_s):
    g=geo[fp]
    return dict(
      wire = (wire_block_s(fp) is not None and tgt_s>wire_block_s(fp)),
      cath = (cath_block_s(fp) is not None and tgt_s>cath_block_s(fp)),
      endcap4 = (tgt_s > g["L"]-4.0),
      outside = (tgt_s > g["last_in"]))

print()
print("### C. REACHABILITY EXCLUSION APPLIED")
for tag,rows in (("H0",h0),("TEACHER98",teach)):
    c=collections.Counter(); ex=collections.defaultdict(list)
    for r in rows:
        fp=r["mesh"]; ts=r["path_len"]-OFF; fl=flags(fp,ts)
        for k,v in fl.items():
            if v: c[k]+=1; ex[k].append((fp,r["path_len"],r["succ"]))
    print("  %-10s n=%d  excluded by: wire0.18=%d  cath0.35=%d  distal4mm=%d  outside_surface=%d"
          %(tag,len(rows),c["wire"],c["cath"],c["endcap4"],c["outside"]))
    for k in ("cath","endcap4"):
        if ex[k]: print("       %s -> %s"%(k,sorted(ex[k])))

print()
print("### deepest DEMONSTRATED success per anatomy (pooled H0 + teacher98), path_len and s_rcca")
deep={}
for rows in (h0,teach):
    for r in rows:
        if r["succ"]:
            deep[r["mesh"]]=max(deep.get(r["mesh"],0),r["path_len"])
for fp in sorted(geo): print("   %s  deepest_success_path_len=%.1f (s_rcca %.1f)  L=%.1f  gap_to_terminus=%.1f mm"
                            %(fp,deep[fp],deep[fp]-OFF,geo[fp]["L"],geo[fp]["L"]-(deep[fp]-OFF)))

# ---------- rates ----------
def rate(rows,pred=lambda r:True):
    s=[r for r in rows if pred(r)]
    return sum(x["succ"] for x in s), len(s)

def show(tag,rows,excl):
    keep=[r for r in rows if not excl(r)]
    a,b=rate(rows); c,d=rate(keep)
    print("  %-10s raw %d/%d = %.1f%%   corrected %d/%d = %.1f%%"%(tag,a,b,100*a/b,c,d,100*c/d if d else 0))
    return keep

def excl_none(r): return False
def excl_cath(r): return flags(r["mesh"],r["path_len"]-OFF)["cath"]
def excl_cath_or_trim(r):
    f=flags(r["mesh"],r["path_len"]-OFF); return f["cath"] or f["endcap4"]

print()
print("### D. CORRECTED RATES, ALL DEPTHS, 4 HOLDOUT ANATOMIES")
for name,ex in (("wire-0.18 (hard block)",excl_none),("catheter-0.35",excl_cath),("cath OR distal-4mm",excl_cath_or_trim)):
    print(" exclusion rule: %s"%name)
    show("H0",h0,ex); show("TEACHER",teach,ex)

# ---------- past-seam comparison ----------
def fisher(a,b,c,d):
    # 2x2 [[a,b],[c,d]] two-sided
    n=a+b+c+d; r1=a+b; c1=a+c
    def p(x):
        return math.comb(r1,x)*math.comb(n-r1,c1-x)/math.comb(n,c1)
    lo=max(0,c1-(n-r1)); hi=min(r1,c1)
    p0=p(a); tot=0.0
    for x in range(lo,hi+1):
        px=p(x)
        if px<=p0*(1+1e-9): tot+=px
    return min(1.0,tot)

print()
print("### E. PAST-SEAM (path_len > %.2f) COMPARISON, 4 HOLDOUT ANATOMIES"%SEAM)
for name,ex in (("uncorrected",excl_none),("catheter-0.35 exclusion",excl_cath),("cath OR distal-4mm",excl_cath_or_trim)):
    H=[r for r in h0 if r["path_len"]>SEAM and not ex(r)]
    T=[r for r in teach if r["path_len"]>SEAM and not ex(r)]
    hs=sum(r["succ"] for r in H); ts=sum(r["succ"] for r in T)
    p=fisher(ts,len(T)-ts,hs,len(H)-hs)
    print("  %-26s teacher %d/%d=%.1f%%  H0 %d/%d=%.1f%%  diff=%+.1f pp  Fisher p=%.4f"
          %(name,ts,len(T),100*ts/len(T),hs,len(H),100*hs/len(H),100*ts/len(T)-100*hs/len(H),p))

# ---------- standardisation ----------
print()
print("### F. STANDARDISED TO A COMMON ANATOMY x DEPTH MIX (past seam, cath-0.35 exclusion)")
def band(pl):
    if pl<=200: return "167-200"
    if pl<=240: return "200-240"
    return ">240"
def strat(rows,ex):
    d=collections.defaultdict(lambda:[0,0])
    for r in rows:
        if r["path_len"]<=SEAM or ex(r): continue
        k=(r["mesh"],band(r["path_len"]))
        d[k][1]+=1; d[k][0]+=r["succ"]
    return d
for exname,ex in (("cath-0.35",excl_cath),("none",excl_none)):
    SH=strat(h0,ex); ST=strat(teach,ex)
    cells=sorted(set(SH)|set(ST))
    wsum=0.0; hh=0.0; tt=0.0; both=0
    print("  exclusion=%s"%exname)
    print("     cell                       H0        TEACHER    weight")
    for c in cells:
        h=SH.get(c,[0,0]); t=ST.get(c,[0,0])
        w=h[1]+t[1]
        mark="" if (h[1] and t[1]) else "   (one-sided, dropped from standardised)"
        print("     %-12s %-9s %2d/%-3d    %2d/%-3d    %3d%s"%(c[0][-5:],c[1],h[0],h[1],t[0],t[1],w,mark))
        if h[1] and t[1]:
            wsum+=w; hh+=w*h[0]/h[1]; tt+=w*t[0]/t[1]; both+=1
    if wsum:
        print("     -> common-support cells=%d, total weight=%d"%(both,wsum))
        print("     -> STANDARDISED  teacher=%.1f%%   H0=%.1f%%   diff=%+.1f pp"%(100*tt/wsum,100*hh/wsum,100*(tt-hh)/wsum))

print()
print("### G. MIX DIFFERENCE (raw, all depths)")
for tag,rows in (("H0",h0),("TEACHER",teach)):
    c=collections.Counter(r["mesh"] for r in rows)
    prox=sum(1 for r in rows if r["path_len"]<=SEAM)
    print("  %-8s n=%d  %s  proximal_to_seam=%d (%.1f%%)"%(tag,len(rows),dict(sorted(c.items())),prox,100*prox/len(rows)))
print()
print("### H. PER-ANATOMY RATES (all depths)")
print("   anat      H0            TEACHER")
for fp in sorted(geo):
    h=[r for r in h0 if r["mesh"]==fp]; t=[r for r in teach if r["mesh"]==fp]
    print("   %s  %2d/%-3d %5.1f%%   %2d/%-3d %5.1f%%"%(fp[-5:],sum(r["succ"] for r in h),len(h),100*sum(r["succ"] for r in h)/len(h),
                                                       sum(r["succ"] for r in t),len(t),100*sum(r["succ"] for r in t)/len(t)))
