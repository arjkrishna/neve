import glob,os
import numpy as np
exec(open("monitoring/attack2_holdout_geometry.py").read().split("def analyse")[0].replace("main()",""))
EXCL={"topcow_mr_013","topcow_mr_014","topcow_mr_015"}
HOLD={"topcow_mr_004","topcow_mr_008","topcow_mr_017","topcow_mr_023"}
def verts(path):
    v=[]
    for ln in open(path,errors="ignore"):
        if ln.startswith("v "):
            a=ln.split(); v.append((float(a[1]),float(a[2]),float(a[3])))
    return np.array(v)
print("nearest-mesh-vertex distance minus declared MISR radius, graft region s>=133 (negative = surface TIGHTER than declared)")
print("%-15s %6s %8s %8s %8s"%("anat","nvert","mean","p05","p95"))
out={}
for d in sorted(glob.glob(os.path.join(ANAT,"topcow_mr_*"))):
    n=os.path.basename(d)
    if n in EXCL: continue
    V=verts(os.path.join(d,"vessel_architecture_collision.obj"))
    p,r=read_curve(os.path.join(d,"Centrelines_comb",RCCA_FILE)); s=arclength(p); g=s>=133.0
    q,qr=p[g][::3],r[g][::3]
    dd=np.linalg.norm(q[:,None,:]-V[None,:,:],axis=2).min(axis=1)-qr
    out[n]=dd
    m="*" if n in HOLD else " "
    print("%s%-14s %6d %8.3f %8.3f %8.3f"%(m,n,len(V),dd.mean(),np.percentile(dd,5),np.percentile(dd,95)))
allm=np.array([v.mean() for v in out.values()])
print("cohort mean of means = %+.3f mm  sd across anatomies = %.3f mm  range %+.3f..%+.3f"%(
  allm.mean(),allm.std(ddof=1),allm.min(),allm.max()))
h=[out[n].mean() for n in out if n in HOLD]; o=[out[n].mean() for n in out if n not in HOLD]
print("holdout4 mean %+.3f   other18 mean %+.3f   difference %+.3f mm"%(np.mean(h),np.mean(o),np.mean(h)-np.mean(o)))
