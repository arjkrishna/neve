import pickle,statistics as st
from collections import defaultdict
D=pickle.load(open("D:/Arjun/workspace/neve/monitoring/_t2_parsed.pkl","rb"))
eps=D["eps"]; by_seed=D["by_seed"]
OFF=33.314
recs=[]
for k,v in eps.items():
    m=by_seed[v["seed"]]
    S=v["steps"]
    ps=[s["proj_s"] for s in S]
    mx=max(ps); imx=ps.index(mx)
    recs.append(dict(seed=v["seed"],mesh=v["mesh"],anat=m["anat"],succ=m["success"],
        pl=m["path_len"],sec=m["section"],nst=m["steps"],logn=len(S),
        max_proj=mx, max_s=mx-OFF, tgt_s=m["path_len"]-OFF,
        final_proj=ps[-1], final_s=ps[-1]-OFF, imax=imx, S=S))
pickle.dump(recs,open("D:/Arjun/workspace/neve/monitoring/_t2_recs.pkl","wb"))
F=[r for r in recs if not r["succ"]]
print("failures",len(F))
# sanity: mesh_fp vs csv anatomy
bad=[r for r in recs if r["mesh"].replace("topcowmr","").lstrip("0")!=r["anat"].replace("topcowmr","").lstrip("0")]
print("mesh/anat mismatch",len(bad))
print()
print("=== ALL FAILURES: deepest s_RCCA reached vs target s_RCCA ===")
print(f"{'anat':<12}{'n_ep':>5}{'nfail':>6}  modal_arrest_s   spread(IQR)   min   max   med_gap_to_tgt")
by_an=defaultdict(list)
for r in recs: by_an[r["mesh"]].append(r)
def modal(vals,bw=2.0):
    # densest bw-mm window
    if not vals: return None,0
    best=(None,-1)
    for c in vals:
        n=sum(1 for x in vals if c-bw/2<=x<=c+bw/2)
        if n>best[1]: best=(c,n)
    lo=best[0]-bw/2; hi=best[0]+bw/2
    inw=[x for x in vals if lo<=x<=hi]
    return st.median(inw),best[1]
for a in sorted(by_an):
    rs=by_an[a]; fs=[r for r in rs if not r["succ"]]
    if not fs:
        print(f"{a:<12}{len(rs):>5}{0:>6}   -")
        continue
    d=sorted(r["max_s"] for r in fs)
    md,cnt=modal(d)
    q1=d[len(d)//4]; q3=d[(3*len(d))//4]
    gaps=sorted(r["tgt_s"]-r["max_s"] for r in fs)
    print(f"{a:<12}{len(rs):>5}{len(fs):>6}   {md:7.1f} (n={cnt}/{len(fs)})  IQR {q1:6.1f}-{q3:6.1f}  {d[0]:6.1f} {d[-1]:6.1f}   {st.median(gaps):6.1f}")
