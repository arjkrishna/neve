"""Where does each anatomy's RCCA course depart from the HOST's? arclength-parameterised."""
import os, re, json
import numpy as np

ROOT = r"d:/Arjun/workspace/neve"
ANAT = os.path.join(ROOT, "topbrain_data/anatomies")
HOST = os.path.join(ROOT, "eve_bench/data/dualdevicenav/Centrelines_comb")

def load(p):
    d=json.load(open(p)); pts=[]; rr=[]
    for m in d["markups"]:
        if m["type"]!="Curve": continue
        for cp in m["controlPoints"]:
            x,y,z=cp["position"]; pts.append((y,-z,-x))
        for me in m.get("measurements",[]):
            if me["name"]=="Radius": rr.extend(me["controlPointValues"])
    return np.array(pts,float), np.array(rr,float)

def arc(p): return np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])

def resample(p, s, grid):
    return np.stack([np.interp(grid, s, p[:,i]) for i in range(3)],axis=1)

def pt2poly(P, poly):
    A=poly[:-1]; B=poly[1:]; AB=B-A; L2=(AB*AB).sum(1)
    out=np.empty(len(P))
    for i,p in enumerate(P):
        t=np.clip(((p-A)*AB).sum(1)/np.maximum(L2,1e-12),0,1)
        out[i]=np.linalg.norm(A+t[:,None]*AB-p,axis=1).min()
    return out

h,hr = load(os.path.join(HOST,"Centerline curve - RCCA.mrk.json")); hs=arc(h)
grid = np.arange(0, 200.0, 0.25)
HOLD={"topcowmr004","topcowmr008","topcowmr017","topcowmr023"}
print("host RCCA n=%d len=%.2f  (first inter-point step %.2f mm - ostium is coarsely sampled)"
      % (len(h), hs[-1], np.linalg.norm(h[1]-h[0])))
print()
print("%-13s %8s %8s %8s %8s %8s %8s" % ("anatomy","len","d>0.05","d>0.25","d>1.00","d>2.00","radii>0.01"))
rowfmt="%-13s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %s"
res={}
for d in sorted(os.listdir(ANAT)):
    fp=re.sub(r"[^A-Za-z0-9]","",d)
    p=os.path.join(ANAT,d,"Centrelines_comb","Centerline curve - RCCA.mrk.json")
    if not os.path.exists(p): continue
    a,ar_=load(p); s=arc(a)
    gg=grid[grid<=min(s[-1],hs[-1])]
    Pa=resample(a,s,gg)
    dd=pt2poly(Pa,h)
    def firstover(thr):
        m=dd>thr
        return float(gg[np.argmax(m)]) if m.any() else float("nan")
    # radii: interp both onto grid by arclength
    rha=np.interp(gg,hs,hr); raa=np.interp(gg,s,ar_)
    dr=np.abs(rha-raa); mr=dr>0.01
    sr=float(gg[np.argmax(mr)]) if mr.any() else float("nan")
    res[fp]=dict(dep05=firstover(0.05),dep25=firstover(0.25),dep1=firstover(1.0),
                 dep2=firstover(2.0),radii=sr,tot=float(s[-1]),
                 rms_pre=float(np.sqrt((dd[gg<130]**2).mean())))
    print(rowfmt%(fp,s[-1],res[fp]["dep05"],res[fp]["dep25"],res[fp]["dep1"],
                  res[fp]["dep2"],sr,"*HOLDOUT*" if fp in HOLD else ""))
print()
print("rms course deviation over s<130 mm (mm):")
for fp in sorted(res):
    print("  %-13s %.5f %s"%(fp,res[fp]["rms_pre"],"*HOLDOUT*" if fp in HOLD else ""))
