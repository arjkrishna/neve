import re,glob,os,numpy as np
runs=[r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints\eval_anatomies_checkpoint2002292\logs\20260826_180252"]
pat=re.compile(r"cmd_action=\[([^\]]*)\].*?cur_branch=(.*?) \| local_r=(\S+) \| tol=(\S+).*?grader=(\S+).*?xt_true=(\S+) \| proj_s=(\S+)")
A=[];XT=[];TL=[];BR=[];PS=[]
for run in runs:
    for f in glob.glob(os.path.join(run,"*.log")):
        for line in open(f,errors="ignore"):
            m=pat.search(line)
            if not m: continue
            ca,br,lr,tol,gr,xt,ps=m.groups()
            if gr!="RCCA": continue
            try:
                a=[float(v) for v in ca.split(",")]
                A.append(a[0]); XT.append(float(xt)); TL.append(float(tol)); BR.append("RCCA" in br); PS.append(float(ps))
            except ValueError: pass
A=np.array(A);XT=np.array(XT);TL=np.array(TL);BR=np.array(BR);PS=np.array(PS)
off=XT>TL
print("n=",len(A)," frac xt>tol=",float(off.mean()))
for lab,mask in [("ALL",np.ones(len(A),bool)),("cur_branch=RCCA",BR)]:
    a1=A[mask&off]; a0=A[mask&~off]
    print(f"\n{lab}: n_off={len(a1)} n_on={len(a0)}")
    print(f"  gw_trans | xt>tol : mean {a1.mean():+.3f} med {np.median(a1):+.3f} frac<0 {float((a1<0).mean()):.3f} p05 {np.percentile(a1,5):+.3f}")
    print(f"  gw_trans | xt<=tol: mean {a0.mean():+.3f} med {np.median(a0):+.3f} frac<0 {float((a0<0).mean()):.3f} p05 {np.percentile(a0,5):+.3f}")
    print(f"  DELTA mean {a1.mean()-a0.mean():+.3f} mm/step ; Cohen d {(a1.mean()-a0.mean())/np.sqrt((a1.var()+a0.var())/2):+.3f}")
