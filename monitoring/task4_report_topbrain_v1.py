import json, sys, statistics
from collections import Counter, defaultdict

P = sys.argv[1]
R = [json.loads(l) for l in open(P)]
S_OFF = 33.31   # s_RCCA = proj_s - 33.31
SEAM = 133.6    # s_RCCA of graft seam  (proj_s 166.9)

# main.log timestamps and STEP wall_time are offset ~2h in this run, so the
# main.log eval windows do not line up. Eval blocks are instead recovered by
# clustering the seeded (eval) episodes on their own wall_time: the three eval
# blocks are separated by ~4h of explore. Yields 98/98/98, matching the
# CSV success rates 72/98=0.735 and 97/98=0.9898 exactly.
_ev = sorted([r for r in R if r["stream"] == "eval"], key=lambda r: r["t"])
_blk = 0; _prev = None
for _r in _ev:
    if _prev is not None and _r["t"] - _prev > 1800:
        _blk += 1
    _prev = _r["t"]
    _r["evblk"] = _blk + 1

def grp(r):
    if r["stream"] == "eval":
        return "eval%d" % r["evblk"]
    return "explore"

# RECONCILIATION OF THE OUTCOME-LINE SHORTFALL (the known data defect).
# Eval blocks 1/2/3 have 98 EPISODE_STARTs each but only 86/89/82 EPISODE_OUTCOME
# lines. Closed successes are 60/88/81. The top-level CSV reports 0.735 / 0.9898 /
# 0.9898 == 72/98, 97/98, 97/98. 60+12=72, 88+9=97, 81+16=97 -- i.e. EVERY eval
# episode missing an outcome line was a SUCCESS, in all three blocks. (The last
# episode a worker finishes in an eval block terminates successfully and the block
# ends before its OUTCOME line is emitted.) So unclosed eval episodes are scored
# as successes; this reproduces the CSV exactly.
for _r in R:
    _r["succ"] = bool(_r["succ"]) or (_r["stream"] == "eval" and not _r["closed"])
    _r["scored"] = _r["closed"] or _r["stream"] == "eval"

_tr = {}
for _r in R:
    g = grp(_r)
    a, b = _tr.get(g, (1e18, -1e18))
    _tr[g] = (min(a, _r["t"]), max(b, _r["t"]))
import time as _t
print("block wall-clock ranges (UTC):")
for g in sorted(_tr):
    a, b = _tr[g]
    print("   %-8s %s .. %s" % (g, _t.strftime("%H:%M:%S", _t.gmtime(a)), _t.strftime("%H:%M:%S", _t.gmtime(b))))

print("=== 0. LEDGER / DATA DEFECT ===")
c = Counter(grp(r) for r in R)
print("episodes emitted by group:", dict(c))
noout = Counter(grp(r) for r in R if not r["closed"])
print("episodes WITHOUT EPISODE_OUTCOME:", dict(noout))
for g in ["eval1","eval2","eval3","explore"]:
    sub = [r for r in R if grp(r) == g]
    if not sub: continue
    cl = [r for r in sub if r["closed"]]
    print("%-8s n=%4d closed=%4d  succ(reason)=%5.1f%%  mean_steps=%6.1f  median=%5.0f" % (
        g, len(sub), len(cl), 100.0*sum(1 for r in cl if r["succ"])/max(1,len(cl)),
        statistics.mean([r["steps"] for r in sub]), statistics.median([r["steps"] for r in sub])))
    print("         reasons:", dict(Counter(r["reason"] for r in sub).most_common()))

def stats(sub, cfg, key="ev"):
    n = len(sub)
    evs = [e for r in sub for e in r[key][cfg]]
    nstall = len(evs)
    epi_with = sum(1 for r in sub if r[key][cfg])
    kc = Counter(e["k"] for e in evs)
    resolved = nstall - kc["unrec"]
    # episodes that stalled AND recovered at least once
    rec_eps = [r for r in sub if any(e["k"] != "unrec" for e in r[key][cfg])]
    rec_eps_cl = [r for r in rec_eps if r["scored"]]
    unrec_eps = [r for r in sub if r[key][cfg] and all(e["k"] == "unrec" for e in r[key][cfg])]
    unrec_eps_cl = [r for r in unrec_eps if r["scored"]]
    nostall_cl = [r for r in sub if not r[key][cfg] and r["scored"]]
    return dict(n=n, nstall=nstall, per_ep=nstall/max(1,n), epi_with=epi_with,
                epi_frac=epi_with/max(1,n), kc=kc, resolved=resolved,
                res_frac=resolved/max(1,nstall),
                rec_succ=(sum(1 for r in rec_eps_cl if r["succ"])/len(rec_eps_cl) if rec_eps_cl else None),
                n_rec_eps=len(rec_eps_cl),
                unrec_succ=(sum(1 for r in unrec_eps_cl if r["succ"])/len(unrec_eps_cl) if unrec_eps_cl else None),
                n_unrec_eps=len(unrec_eps_cl),
                nostall_succ=(sum(1 for r in nostall_cl if r["succ"])/len(nostall_cl) if nostall_cl else None),
                n_nostall=len(nostall_cl))

GROUPS = [("eval1", lambda r: grp(r)=="eval1"),
          ("eval2", lambda r: grp(r)=="eval2"),
          ("eval3", lambda r: grp(r)=="eval3"),
          ("eval2+3", lambda r: grp(r) in ("eval2","eval3")),
          ("explore", lambda r: r["stream"]=="explore")]

for KEY, KNAME in [("ev","PROJ_S-STALL (canonical extract_stuck.py)"),
                   ("evx","EXECUTED-ADVANCE delta_ins<eps (task-spec variant)")]:
    print("\n\n########## DETECTOR FAMILY: %s ##########" % KNAME)
    for cfg in ["canon","sens","strict"]:
        print("\n--- cfg=%s ---" % cfg)
        print("%-8s %5s %7s %8s %9s | %6s %6s %6s %6s | %7s | %8s %8s %8s" % (
            "group","n","stalls","st/ep","%ep>=1", "grind","soft","hard","unrec",
            "%resolv","succ|rec","succ|unrec","succ|nostall"))
        for gname, f in GROUPS:
            sub = [r for r in R if f(r)]
            if not sub: continue
            d = stats(sub, cfg, KEY)
            tot = max(1, d["nstall"])
            def pc(x): return "%d(%.0f%%)" % (d["kc"][x], 100.0*d["kc"][x]/tot)
            def fmt(v,n): return "%5.1f%%(%d)" % (100*v, n) if v is not None else "   -  "
            print("%-8s %5d %7d %8.3f %8.1f%% | %10s %10s %10s %10s | %6.1f%% | %s %s %s" % (
                gname, d["n"], d["nstall"], d["per_ep"], 100*d["epi_frac"],
                pc("grind"), pc("soft"), pc("hard"), pc("unrec"),
                100*d["res_frac"], fmt(d["rec_succ"],d["n_rec_eps"]),
                fmt(d["unrec_succ"],d["n_unrec_eps"]), fmt(d["nostall_succ"],d["n_nostall"])))

print("\n\n=== SENSITIVITY: stuck_steps sweep (other canon params fixed) ===")
for KEY, KNAME in [("ev","proj_s-stall"), ("evx","delta_ins-stall")]:
    print("\n-- %s --" % KNAME)
    print("%-8s %6s | %s" % ("group","n", "  ".join("ss=%s:stalls/ep(%%ep)" % s for s in [4,6,8,12])))
    for gname, f in GROUPS:
        sub = [r for r in R if f(r)]
        if not sub: continue
        cells = []
        for cfg in ["ss4","ss6","ss8","canon"]:
            d = stats(sub, cfg, KEY)
            cells.append("%6d %5.2f %5.1f%%" % (d["nstall"], d["per_ep"], 100*d["epi_frac"]))
        print("%-8s %6d | %s" % (gname, len(sub), "  ".join(cells)))

print("\n(cells = total stalls, stalls/episode, %% episodes with >=1 stall)")
print("\nsteps available to trigger: mean episode length")
for gname, f in GROUPS:
    sub = [r for r in R if f(r)]
    if not sub: continue
    st = [r["steps"] for r in sub]
    print("  %-8s mean %6.1f  median %5.0f  p10 %4.0f  frac_eps_with_<12_steps %.3f  <4 %.3f" % (
        gname, statistics.mean(st), statistics.median(st), sorted(st)[len(st)//10],
        sum(1 for x in st if x < 12)/len(st), sum(1 for x in st if x < 4)/len(st)))

print("\n\n=== 4. SPATIAL DISTRIBUTION OF STALL ONSETS (s_RCCA = proj_s - 33.31) ===")
BINS = [(-1e9,0),(0,20),(20,40),(40,60),(60,80),(80,100),(100,120),(120,133.6),
        (133.6,140),(140,150),(150,166.7),(166.7,180),(180,200),(200,1e9)]
def binlab(a,b):
    if a < -1e8: return "  <0 (pre-RCCA)"
    if b > 1e8: return " >200"
    return "%5.0f-%-5.0f" % (a,b)
for KEY,KNAME in [("ev","proj_s-stall"),("evx","delta_ins-stall")]:
    for cfg in ["canon","ss6"]:
        print("\n-- %s cfg=%s -- onset s_RCCA histogram" % (KNAME,cfg))
        cols = {}
        for gname, f in GROUPS:
            if gname == "eval2+3": continue
            sub = [r for r in R if f(r)]
            xs = [e["s0"]-S_OFF for r in sub for e in r[KEY][cfg]]
            cols[gname] = xs
        hdr = [g for g in cols if cols[g] or True]
        print("%-16s %s" % ("s_RCCA bin", "  ".join("%12s" % g for g in hdr)))
        for a,b in BINS:
            row = []
            for g in hdr:
                xs = cols[g]; n = sum(1 for x in xs if a <= x < b)
                row.append("%5d %5.1f%%" % (n, 100.0*n/max(1,len(xs))))
            print("%-16s %s" % (binlab(a,b), "  ".join("%12s" % c for c in row)))
        print("%-16s %s" % ("TOTAL", "  ".join("%12d" % len(cols[g]) for g in hdr)))
        for g in hdr:
            xs = cols[g]
            if xs:
                print("   %s: median %.1f  p25 %.1f  p75 %.1f  frac in [126,141] seam+-7.5: %.3f  frac in [167,200]: %.3f" % (
                    g, statistics.median(xs), sorted(xs)[len(xs)//4], sorted(xs)[3*len(xs)//4],
                    sum(1 for x in xs if 126 <= x <= 141)/len(xs),
                    sum(1 for x in xs if 167 <= x <= 200)/len(xs)))

print("\n\n=== EXPOSURE CONTROL: how much episode-time is spent past the seam ===")
for gname,f in GROUPS:
    sub=[r for r in R if f(r)]
    if not sub: continue
    mp=[r["maxp"]-S_OFF for r in sub]
    print("  %-8s max s_RCCA reached: median %6.1f  frac reaching >=133.6: %.3f  >=167: %.3f" % (
        gname, statistics.median(mp), sum(1 for x in mp if x>=133.6)/len(mp),
        sum(1 for x in mp if x>=167)/len(mp)))

print("\n\n=== 5. EXPLORE OVER TRAINING TIME (bucketed by per-worker global_steps) ===")
ex = [r for r in R if r["stream"]=="explore" and r["gs"]>=0]
ex.sort(key=lambda r: r["gs"])
NB=8
sz=len(ex)//NB
print("%-22s %6s %8s | %s" % ("bucket (worker gsteps)","n","meanstep","  ".join("%s" % c for c in ["canon st/ep","%ep","ss6 st/ep","%ep","unrec%","succ%"])))
for i in range(NB):
    sub = ex[i*sz:(i+1)*sz] if i<NB-1 else ex[i*sz:]
    d = stats(sub,"canon","ev"); d6 = stats(sub,"ss6","ev")
    cl=[r for r in sub if r["closed"]]
    print("%-22s %6d %8.1f | %10.3f %6.1f%% %10.3f %6.1f%% %6.1f%% %6.1f%%" % (
        "%d-%d"%(sub[0]["gs"],sub[-1]["gs"]), len(sub), statistics.mean([r["steps"] for r in sub]),
        d["per_ep"], 100*d["epi_frac"], d6["per_ep"], 100*d6["epi_frac"],
        100*d["kc"]["unrec"]/max(1,d["nstall"]),
        100*sum(1 for r in cl if r["succ"])/max(1,len(cl))))

print("\n=== retraction depth distribution of recoveries (canon, proj_s) ===")
for gname,f in GROUPS:
    sub=[r for r in R if f(r)]
    rs=[e["r"] for r in sub for e in r["ev"]["canon"] if e["k"]!="unrec"]
    ds=[e["dur"] for r in sub for e in r["ev"]["canon"] if e["k"]!="unrec"]
    if rs:
        print("  %-8s n=%4d retract mm: median %.2f p90 %.2f max %.2f | stall duration steps: median %.0f p90 %.0f" % (
            gname,len(rs),statistics.median(rs),sorted(rs)[int(.9*len(rs))],max(rs),
            statistics.median(ds),sorted(ds)[int(.9*len(ds))]))

print("\n\n=== 6. IS THE EVAL-2/3 'NO STALL' RESULT DETECTOR-LIMITED? ===")
print("maxrun = longest low-advance-while-pushing run reached in the episode")
print("(counter freezes at stuck_steps once a stall fires, so >=12 means a stall fired)")
for gname,f in GROUPS:
    sub=[r for r in R if f(r)]
    if not sub: continue
    for tag,k in [("proj_s","maxrun"),("delta_ins","maxrunx")]:
        v=sorted(r[k] for r in sub)
        n=len(v)
        q=lambda p: v[min(n-1,int(p*n))]
        print("  %-8s %-9s median %2d  p75 %2d  p90 %2d  p95 %2d  p99 %2d  max %3d | %%eps maxrun>=4 %5.1f >=6 %5.1f >=8 %5.1f >=12 %5.1f" % (
            gname,tag,q(.5),q(.75),q(.9),q(.95),q(.99),v[-1],
            100*sum(1 for x in v if x>=4)/n,100*sum(1 for x in v if x>=6)/n,
            100*sum(1 for x in v if x>=8)/n,100*sum(1 for x in v if x>=12)/n))

print("\ndwell: steps spent in the distal bands (opportunity for a stall to be seen)")
for gname,f in GROUPS:
    sub=[r for r in R if f(r)]
    if not sub: continue
    ns=[r["n_seam"] for r in sub]; n7=[r["n_167"] for r in sub]
    print("  %-8s steps past seam(s_RCCA>=133.6): mean %6.1f median %5.0f | steps in 167-200: mean %6.1f median %5.0f | frac eps with >=12 steps in 167-200: %.3f  >=4: %.3f" % (
        gname, statistics.mean(ns), statistics.median(ns), statistics.mean(n7), statistics.median(n7),
        sum(1 for x in n7 if x>=12)/len(n7), sum(1 for x in n7 if x>=4)/len(n7)))

print("\nstall RATE PER 1000 STEPS (exposure-normalised; removes the episode-length confound)")
for KEY,KN in [("ev","proj_s"),("evx","delta_ins")]:
    print("  -- %s --" % KN)
    for gname,f in GROUPS:
        sub=[r for r in R if f(r)]
        if not sub: continue
        tot=sum(r["steps"] for r in sub)
        cells=[]
        for cfg in ["ss4","ss6","ss8","canon"]:
            k=sum(len(r[KEY][cfg]) for r in sub)
            cells.append("%6.2f"%(1000.0*k/tot))
        print("    %-8s steps=%8d  ss4 %s  ss6 %s  ss8 %s  ss12 %s" % (gname,tot,*cells))

print("\n\n=== 7. LEDGER SNAPSHOT (logs still being appended by the live container) ===")
for g in ["eval1","eval2","eval3","explore"]:
    sub=[r for r in R if grp(r)==g]
    cl=[r for r in sub if r["closed"]]
    print("  %-8s starts=%5d outcomes=%5d shortfall=%4d (%.1f%%)  succ/starts=%5.1f%%  succ/outcomes=%5.1f%%" % (
        g,len(sub),len(cl),len(sub)-len(cl),100.0*(len(sub)-len(cl))/len(sub),
        100.0*sum(1 for r in sub if r["succ"])/len(sub),
        100.0*sum(1 for r in cl if r["succ"])/max(1,len(cl))))
    if cl:
        import statistics as _s
        print("           mean steps: counted-STEP-lines %.1f (all starts) | EPISODE_OUTCOME steps= field %.1f (closed only)" % (
            _s.mean([r["steps"] for r in sub]), _s.mean([r["osteps"] for r in cl if r["osteps"]>=0])))

print("\n\n=== 8. WHAT THE >200 AND 0-20 CLUSTERS ACTUALLY ARE ===")
print("(remaining = path_len - proj_s at stall onset; st0 = ep_step at onset)")
for KEY,KN in [("ev","proj_s"),("evx","delta_ins")]:
    for cfg in ["canon","ss4"]:
        print("\n-- %s cfg=%s --" % (KN,cfg))
        for gname,f in GROUPS:
            if gname=="eval2+3": continue
            sub=[r for r in R if f(r)]
            evs=[(e,r) for r in sub for e in r[KEY][cfg]]
            if not evs: continue
            rem=[r["pl"]-e["s0"] for e,r in evs if r["pl"]]
            st0=[e["st0"] for e,r in evs]
            early=sum(1 for x in st0 if x<=20)
            near=sum(1 for x in rem if x<=15)
            print("  %-8s n=%5d | onset ep_step<=20: %4d (%.1f%%) | remaining-to-path-end<=15mm: %4d (%.1f%%) | remaining median %6.1f p25 %6.1f" % (
                gname,len(evs),early,100.0*early/len(evs),near,100.0*near/max(1,len(rem)),
                statistics.median(rem) if rem else -1, sorted(rem)[len(rem)//4] if rem else -1))

print("\n\n=== 9. STALLS EXCLUDING THE EPISODE-START GUIDEWIRE-LAG ARTIFACT (onset ep_step>20) ===")
def stats2(sub,cfg,key,minst0=20):
    evs=[e for r in sub for e in r[key][cfg] if e["st0"]>minst0]
    kc=Counter(e["k"] for e in evs)
    epi=sum(1 for r in sub if any(e["st0"]>minst0 for e in r[key][cfg]))
    tot=sum(r["steps"] for r in sub)
    return len(evs),len(evs)/len(sub),100.0*epi/len(sub),kc,1000.0*len(evs)/tot
for KEY,KN in [("ev","proj_s"),("evx","delta_ins")]:
    print("\n-- %s --" % KN)
    print("%-8s %s" % ("group","   ".join("ss=%-2s n  /ep  %%ep  /1kstep"%s for s in [4,6,8,12])))
    for gname,f in GROUPS:
        sub=[r for r in R if f(r)]
        cells=[]
        for cfg in ["ss4","ss6","ss8","canon"]:
            n,pe,pep,kc,per1k=stats2(sub,cfg,KEY)
            cells.append("%5d %4.2f %5.1f%% %5.2f"%(n,pe,pep,per1k))
        print("%-8s %s" % (gname,"  ".join(cells)))
    print("  recovery mix (canon, onset ep_step>20):")
    for gname,f in GROUPS:
        sub=[r for r in R if f(r)]
        n,pe,pep,kc,per1k=stats2(sub,"canon",KEY)
        if n==0:
            print("    %-8s none"%gname); continue
        print("    %-8s n=%4d grind %d(%.0f%%) soft %d(%.0f%%) hard %d(%.0f%%) unrec %d(%.0f%%)"%(
            gname,n,kc["grind"],100.*kc["grind"]/n,kc["soft"],100.*kc["soft"]/n,
            kc["hard"],100.*kc["hard"]/n,kc["unrec"],100.*kc["unrec"]/n))

print("\n\n=== 10. EXPLORE OVER TRAINING TIME, EXPOSURE-NORMALISED ===")
ex=[r for r in R if r["stream"]=="explore" and r["gs"]>=0]
ex.sort(key=lambda r:r["gs"])
NB=8; sz=len(ex)//NB
print("%-16s %5s %8s %8s | %s" % ("worker gsteps","n","meanstep","stepspast","canon/1k  ss6/1k  ss4/1k  canon(st0>20)/1k  %ep>=1  unrec%  succ%"))
for i in range(NB):
    sub=ex[i*sz:(i+1)*sz] if i<NB-1 else ex[i*sz:]
    tot=sum(r["steps"] for r in sub)
    d=stats(sub,"canon","ev")
    k=lambda c: 1000.0*sum(len(r["ev"][c]) for r in sub)/tot
    n2,_,pep2,kc2,per1k2=stats2(sub,"canon","ev")
    cl=[r for r in sub if r["closed"]]
    print("%-16s %5d %8.1f %8.1f | %7.2f %7.2f %7.2f %14.2f %8.1f%% %6.1f%% %6.1f%%" % (
        "%d-%d"%(sub[0]["gs"],sub[-1]["gs"]),len(sub),tot/len(sub),
        statistics.mean([r["n_seam"] for r in sub]),
        k("canon"),k("ss6"),k("ss4"),per1k2,100*d["epi_frac"],
        100*d["kc"]["unrec"]/max(1,d["nstall"]),
        100*sum(1 for r in cl if r["succ"])/max(1,len(cl))))
