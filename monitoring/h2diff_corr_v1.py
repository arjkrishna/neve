import json, math, random, collections
import numpy as np
random.seed(7); rng=np.random.default_rng(7)
M="monitoring/"
T=json.load(open(M+"h2diff_table.json"))
ROWS=json.load(open(M+"h2diff_eprows.json"))
name_map={a:"topcowmr%03d"%int(a.split("_")[-1]) for a in T if a!="HOST"}
MEAS=["rcca_len","graft_len","tort_graft","tort_w40max","Rc_min","Rc_p05","Rc_p25","Rc_med",
      "n_Rc_lt5","n_Rc_lt8","n_Rc_lt12","bend_max","bend_p90","turn_cum","turn_per_mm","turn_net","turn_eff",
      "n_infl","tors_cum","tors_mean","planarity","frac_top24","r_min","r_p05","r_med","r_term",
      "clr_min","clr_min_nonterm","clr_p05","clr_p25","clr_med","clr_n_lt_cath"]
by=collections.defaultdict(lambda:[0,0,0])
for r in ROWS:
    b=by[r["anat"]]; b[0]+=1; b[1]+=r["succ"]; b[2]+=r["hsucc"]
anats=sorted(by)
n=np.array([by[a][0] for a in anats],float)
kp=np.array([by[a][1] for a in anats],float); kh=np.array([by[a][2] for a in anats],float)
X={m:np.array([T[[k for k in T if name_map.get(k)==a][0]][m] for a in anats],float) for m in MEAS}
def wcorr(x,y,w):
    mx=np.average(x,weights=w); my=np.average(y,weights=w)
    cov=np.average((x-mx)*(y-my),weights=w)
    return cov/math.sqrt(np.average((x-mx)**2,weights=w)*np.average((y-my)**2,weights=w)+1e-30)
def run(kvec,label,B=20000,excl=None):
    idx=[i for i,a in enumerate(anats) if excl is None or a not in excl]
    nn=n[idx]; kk=kvec[idx]; rate=kk/nn
    N=int(nn.sum()); K=int(kk.sum())
    obs={m:wcorr(X[m][idx],rate,nn) for m in MEAS}
    # permutation: reshuffle episode outcomes across anatomies keeping group sizes
    pool=np.array([1]*K+[0]*(N-K))
    cuts=np.cumsum(nn).astype(int)[:-1]
    cnt={m:0 for m in MEAS}; maxdist=np.zeros(B)
    for b in range(B):
        rng.shuffle(pool)
        kb=np.array([s.sum() for s in np.split(pool,cuts)],float)
        rb=kb/nn
        mx=0.0
        for m in MEAS:
            r=abs(wcorr(X[m][idx],rb,nn))
            if r>=abs(obs[m])-1e-12: cnt[m]+=1
            mx=max(mx,r)
        maxdist[b]=mx
    print("\n=== %s  (n_anat=%d, N=%d, K=%d, base=%.3f) : weighted corr vs per-anatomy rate ==="%(label,len(idx),N,K,K/N))
    print("%-16s %7s %8s %8s"%("measure","w_r","perm_p","FWERp"))
    res=[]
    for m in sorted(MEAS,key=lambda m:-abs(obs[m])):
        p=(cnt[m]+1)/(B+1); fw=(np.sum(maxdist>=abs(obs[m]))+1)/(B+1)
        res.append((m,obs[m],p,fw))
        print("%-16s %7.3f %8.4f %8.4f"%(m,obs[m],p,fw))
    sig=[r for r in res if r[2]<0.05]
    print("  measures tested K=%d ; expected #p<0.05 by chance = %.1f ; observed = %d ; min FWER p = %.4f"
          %(len(MEAS),0.05*len(MEAS),len(sig),min(r[3] for r in res)))
    return res
run(kp,"POLICY ckpt2002292, all 22")
run(kp,"POLICY, excl mr_025",excl={"topcowmr025"})
run(kh,"HEURISTIC ckpt0, all 22")
