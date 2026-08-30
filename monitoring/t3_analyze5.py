import pickle, statistics as st
from collections import defaultdict
D=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_eval.pkl","rb")); eps=D["eps"]
SN=pickle.load(open(r"D:/Arjun/workspace/neve/monitoring/_t3_snap.pkl","rb"))
def pct(a,p):
    a=sorted(a)
    if not a: return float('nan')
    i=(len(a)-1)*p/100.0; lo=int(i); hi=min(lo+1,len(a)-1); f=i-lo
    return a[lo]*(1-f)+a[hi]*f
blocks=defaultdict(list)
for k,e in eps.items():
    r=SN.get((e["pid"],int(e["ep"]))); e["res"]=r[0] if r else None
    blocks[e["block"]].append(e)
print("="*72)
print("STATION-MATCHED |xt_true| (successes only), binned by normalized arclength proj_s/path_len")
hdr=f"{'bin':>10s}"+"".join(f"{'eval'+str(b):>26s}" for b in (1,2,3))
print(hdr)
bins=[(i/10,(i+1)/10) for i in range(10)]
data={b:defaultdict(lambda:([],0,0)) for b in (1,2,3)}
for b in (1,2,3):
    for e in blocks[b]:
        if e["res"]!="success": continue
        for s in e["steps"]:
            if s["ps"]!=s["ps"] or s["pl"]!=s["pl"] or s["pl"]<=0: continue
            f=s["ps"]/s["pl"]
            if f<0 or f>1.0: f=min(max(f,0),0.999)
            k=min(int(f*10),9)
            x=abs(s["xt"]); lr=s["lr"]
            if x!=x or lr!=lr or lr<=0: continue
            xs,tot,ex=data[b][k]; xs.append(x); data[b][k]=(xs,tot+1,ex+(1 if x>lr else 0))
for k in range(10):
    row=f"{bins[k][0]:.1f}-{bins[k][1]:.1f}".rjust(10)
    for b in (1,2,3):
        xs,tot,ex=data[b][k]
        row+=f"  med={pct(xs,50):5.2f} p95={pct(xs,95):5.2f} f>r={ex/max(1,tot):.3f}"
    print(row)
print()
print("median local_r per bin (eval2 successes):", [round(0,2)])
lr_b={b:defaultdict(list) for b in (1,2,3)}
for b in (1,2,3):
    for e in blocks[b]:
        if e["res"]!="success": continue
        for s in e["steps"]:
            if s["ps"]!=s["ps"] or s["pl"]<=0 or s["lr"]!=s["lr"]: continue
            lr_b[b][min(int(s["ps"]/s["pl"]*10),9)].append(s["lr"])
print("  bin   local_r med: e1 / e2 / e3")
for k in range(10):
    print(f"  {k/10:.1f}   {pct(lr_b[1][k],50):5.2f} / {pct(lr_b[2][k],50):5.2f} / {pct(lr_b[3][k],50):5.2f}")

print(); print("="*72); print("LOOP / BUCKLE INCIDENCE among SUCCESSES")
for b in (1,2,3):
    E=[e for e in blocks[b] if e["res"]=="success"]
    nf=0; ns=0; nb=0
    for e in E:
        mf=0; ms=-99; mb=0
        for s in e["steps"]:
            try: mf=max(mf,int(s["fold"].split("/")[0]))
            except: pass
            if s["slack"]==s["slack"]: ms=max(ms,s["slack"])
            if s["buck"]==s["buck"]: mb=max(mb,abs(s["buck"]))
        if mf>0: nf+=1
        if ms>20: ns+=1
        if mb>0.3: nb+=1
    print(f"eval{b} successes n={len(E)}: any fold>0 {nf} ({nf/len(E):.3f}) | max cath_slack>20mm {ns} ({ns/len(E):.3f}) | max|buckle_phi|>0.3 {nb} ({nb/len(E):.3f})")
