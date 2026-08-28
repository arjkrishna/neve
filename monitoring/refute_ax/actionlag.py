import re,glob,os,numpy as np
from collections import defaultdict
runs=[r"D:\Arjun\workspace\neve\saved\eve_paper\neurovascular\full\mesh_ben\2026-07-25_022443_rcca_p2_teacher_v1bp\checkpoints\eval_anatomies_checkpoint2002292\logs\20260826_180252"]
pat=re.compile(r"STEP \| ep=(\d+) \| ep_step=(\d+).*?pid=(\d+) \| cmd_action=\[([^\]]*)\].*?cur_branch=(.*?) \| local_r=(\S+) \| tol=(\S+).*?grader=(\S+).*?xt_true=(\S+) \| proj_s=(\S+)")
seq=defaultdict(list)
for run in runs:
    for f in glob.glob(os.path.join(run,"*.log")):
        for line in open(f,errors="ignore"):
            m=pat.search(line)
            if not m: continue
            ep,st,pid,ca,br,lr,tol,gr,xt,ps=m.groups()
            if gr!="RCCA": continue
            try:
                a=[float(v) for v in ca.split(",")]
                seq[(pid,int(ep))].append((int(st),a[0],float(xt),float(tol),float(ps),"RCCA" in br))
            except ValueError: pass
nxt_off=[];nxt_on=[]
for k,v in seq.items():
    v.sort()
    for i in range(len(v)-1):
        st,a,xt,tol,ps,isr=v[i]
        st2,a2,_,_,_,_=v[i+1]
        if st2!=st+1: continue
        (nxt_off if xt>tol else nxt_on).append(a2)
o=np.array(nxt_off); n=np.array(nxt_on)
print("LAGGED: action at t+1 conditioned on (xt>tol) at t")
print(f"  n_off={len(o)} n_on={len(n)}")
print(f"  a_gw[t+1] | off@t : mean {o.mean():+.3f} med {np.median(o):+.3f} frac<0 {float((o<0).mean()):.3f}")
print(f"  a_gw[t+1] | on @t : mean {n.mean():+.3f} med {np.median(n):+.3f} frac<0 {float((n<0).mean()):.3f}")
print(f"  DELTA {o.mean()-n.mean():+.3f} mm  Cohen d {(o.mean()-n.mean())/np.sqrt((o.var()+n.var())/2):+.3f}")
