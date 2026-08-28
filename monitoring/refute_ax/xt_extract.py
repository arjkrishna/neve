import re, glob, os, sys, numpy as np, json
base="/d/Arjun/workspace/neve/saved/eve_paper/neurovascular/full/mesh_ben"
runs=sys.argv[1:]
pat=re.compile(r"STEP \| ep=(\d+) \| ep_step=(\d+).*?pid=(\d+).*?on_path=(\d).*?cur_branch=(.*?) \| local_r=(\S+) \| tol=(\S+).*?grader=(\S+).*?xt_true=(\S+) \| proj_s=(\S+) \| path_len=(\S+)")
rows=[]
for run in runs:
    for f in glob.glob(os.path.join(run,"*.log")):
        for line in open(f, errors="ignore"):
            m=pat.search(line)
            if not m: continue
            ep,st,pid,onp,br,lr,tol,gr,xt,ps,pl=m.groups()
            if gr!="RCCA": continue
            try:
                rows.append((int(pid),int(ep),int(st),int(onp),br.strip(),float(lr),float(tol),float(xt),float(ps),float(pl)))
            except ValueError: continue
print("rows",len(rows))
if not rows: sys.exit()
import numpy as np
lr=np.array([r[5] for r in rows]); tol=np.array([r[6] for r in rows])
xt=np.array([r[7] for r in rows]); ps=np.array([r[8] for r in rows]); pl=np.array([r[9] for r in rows])
onp=np.array([r[3] for r in rows]); br=np.array([r[4] for r in rows])
isr=np.array(["RCCA" in b for b in br])
def q(a,name):
    if len(a)==0: print(name,"EMPTY"); return
    print(f"{name}: n={len(a)} min={a.min():.3f} p05={np.percentile(a,5):.3f} p50={np.percentile(a,50):.3f} p95={np.percentile(a,95):.3f} p99={np.percentile(a,99):.3f} max={a.max():.3f}")
print("=== ALL RCCA-grader steps")
q(xt,"xt_true"); q(tol,"tol"); q(lr,"local_r")
print("frac xt>tol (deadband fires):", float((xt>tol).mean()))
print("frac local_r at floor 2.0:", float((np.abs(lr-2.0)<1e-6).mean()))
print("frac on_path=0:", float((onp==0).mean()))
print("=== steps whose CURRENT BRANCH is RCCA")
q(xt[isr],"xt_true|RCCA"); q(tol[isr],"tol|RCCA"); q(lr[isr],"local_r|RCCA")
print("frac xt>tol on RCCA:", float((xt[isr]>tol[isr]).mean()))
print("frac local_r==2.0 on RCCA:", float((np.abs(lr[isr]-2.0)<1e-6).mean()))
print("=== RCCA branch AND on_path=1")
m=isr&(onp==1)
q(xt[m],"xt|RCCA,onpath"); q(tol[m],"tol|RCCA,onpath")
print("frac xt>tol:", float((xt[m]>tol[m]).mean()))
o49=np.clip(xt/tol/2.0,0,1)
q(o49,"obs49 all"); q(o49[m],"obs49 RCCA-onpath")
print("frac obs49==0:",float((o49==0).mean()), " frac obs49==1:",float((o49==1).mean()))
# deep zone: proj_s > 130 (graft-ish) on RCCA
deep=isr&(ps>130)
print("=== deep (proj_s>130) on RCCA, n=",int(deep.sum()))
if deep.sum():
    q(xt[deep],"xt deep"); q(tol[deep],"tol deep"); q(lr[deep],"local_r deep")
    print("frac xt>tol deep:", float((xt[deep]>tol[deep]).mean()))
    print("frac local_r==2.0 deep:", float((np.abs(lr[deep]-2.0)<1e-6).mean()))
np.savez("/d/Arjun/workspace/neve/monitoring/refute_ax/xt.npz",lr=lr,tol=tol,xt=xt,ps=ps,pl=pl,onp=onp,isr=isr)
