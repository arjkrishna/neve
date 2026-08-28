import re,glob,os,json,numpy as np
G=json.load(open(r"D:\Arjun\workspace\neve\monitoring\refute_ax\geo.json"))
MINR,MAXR,K,MINTOL=2.0,12.0,1.5,2.0
def tol_of(r): return np.maximum(MINTOL, K*np.clip(r,MINR,MAXR))
def prof(s,r):
    s=np.asarray(s); r=np.asarray(r); return s,r
hs,hr=prof(G["host"]["s"],G["host"]["r"])
coh={k:prof(v["s"],v["r"]) for k,v in G["cohort"].items()}
def lookup(s_q,s,r):
    i=np.clip(np.searchsorted(s,s_q),0,len(s)-1)
    i2=np.clip(i-1,0,len(s)-1)
    pick=np.where(np.abs(s[i]-s_q)<np.abs(s[i2]-s_q),i,i2)
    return r[pick]
# --- empirical cross-track from real policy rollouts ---
runs=[r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints\eval_anatomies_checkpoint2002292\logs\20260826_180252",
      r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints\eval_anatomies_checkpoint2002292\logs\20260729_085006"]
pat=re.compile(r"cur_branch=(.*?) \| local_r=(\S+) \| tol=(\S+).*?grader=(\S+).*?xt_true=(\S+) \| proj_s=(\S+) \| path_len=(\S+)")
BR=[];LR=[];TL=[];XT=[];PS=[]
for run in runs:
    for f in glob.glob(os.path.join(run,"*.log")):
        for line in open(f,errors="ignore"):
            m=pat.search(line)
            if not m: continue
            br,lr,tol,gr,xt,ps,pl=m.groups()
            if gr!="RCCA": continue
            try: LR.append(float(lr));TL.append(float(tol));XT.append(float(xt));PS.append(float(ps));BR.append("RCCA" in br)
            except ValueError: pass
LR=np.array(LR);TL=np.array(TL);XT=np.array(XT);PS=np.array(PS);BR=np.array(BR)
INS=33.47
s_r=PS-INS
m=BR&(s_r>0)&(s_r<200)
print(f"steps used (cur_branch RCCA, 0<s_rcca<200): {m.sum()} of {len(XT)}")
xt=XT[m]; sr=s_r[m]
print("empirical xt on RCCA: med %.3f p90 %.3f p95 %.3f"%(np.median(xt),np.percentile(xt,90),np.percentile(xt,95)))
print("realised tol in those rollouts: med %.3f"%np.median(TL[m]))
print("realised trigger rate P(xt>tol_realised) = %.4f"%float((XT[m]>TL[m]).mean()))
print()
th=tol_of(lookup(sr,hs,hr))
print("=== counterfactual: SAME cross-track sample, tolerance swapped ===")
print("HOST tol at matched arclength: med %.3f  frac==3.0 %.3f"%(np.median(th),float((th<=3.0+1e-9).mean())))
rate_h=float((xt>th).mean())
pen_h=float(np.maximum(0,xt-th).mean())
print("HOST: retract-trigger rate P(xt>tol) = %.4f ; mean lateral excess = %.4f mm"%(rate_h,pen_h))
rows=[]
for name,(s,r) in sorted(coh.items()):
    tc=tol_of(lookup(sr,s,r))
    rate=float((xt>tc).mean()); pen=float(np.maximum(0,xt-tc).mean())
    o49h=np.clip(xt/th/2,0,1); o49c=np.clip(xt/tc/2,0,1)
    rows.append((name,np.median(tc),rate,rate-rate_h,pen,pen-pen_h,float(np.median(o49c-o49h))))
print("\nanat            tol_med  trigrate  d_trig   latexc   d_lat   d_obs49_med")
for n,t,ra,dr,pe,dp,do in rows:
    print(f"{n:15s} {t:6.3f}  {ra:7.4f} {dr:+7.4f} {pe:7.4f} {dp:+7.4f} {do:+8.4f}")
tr=np.array([r[2] for r in rows]); dl=np.array([r[5] for r in rows]); do=np.array([r[6] for r in rows])
print(f"\nCOHORT median trigger rate {np.median(tr):.4f}  vs host {rate_h:.4f}  -> RELATIVE {np.median(tr)/rate_h:.3f}x")
print(f"COHORT median lateral-excess delta {np.median(dl):+.4f} mm/step ; obs49 median delta {np.median(do):+.4f}")
# zone restriction: graft only
for lo,hi,lab in [(0,103.4,"Z1 s<103.4"),(103.4,136,"Z2 103.4-136"),(136,200,"Z3 >=136")]:
    z=(sr>=lo)&(sr<hi)
    if z.sum()<50: print(lab,"n too small",z.sum()); continue
    thz=th[z]; xz=xt[z]
    rz=float((xz>thz).mean())
    rc=[float((xz>tol_of(lookup(sr[z],s,r))).mean()) for _,(s,r) in sorted(coh.items())]
    print(f"{lab}: n={int(z.sum())} host trig {rz:.4f} | cohort trig med {np.median(rc):.4f} range {min(rc):.4f}-{max(rc):.4f}")
np.savez(r"D:\Arjun\workspace\neve\monitoring\refute_ax\xt.npz",xt=xt,sr=sr)
