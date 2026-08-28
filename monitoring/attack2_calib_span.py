import glob, json, os
import numpy as np
exec(open("monitoring/attack2_holdout_geometry.py").read().split("def analyse")[0].replace("main()",""))

EXCL={"topcow_mr_013","topcow_mr_014","topcow_mr_015"}
def load(folder):
    p,r = read_curve(os.path.join(folder, RCCA_FILE)); return p,r

def menger_span(p,m):
    a=p[:-2*m]; b=p[m:-m]; c=p[2*m:]
    ab=np.linalg.norm(b-a,axis=1); bc=np.linalg.norm(c-b,axis=1); ca=np.linalg.norm(a-c,axis=1)
    area=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1)
    den=ab*bc*ca
    return np.where(den>1e-12, 4*area/np.maximum(den,1e-12), 0.0)

def bend_span(p,m):
    """angle between chord tangents separated by m stations, at each point"""
    t=p[m:]-p[:-m]
    t=t/np.linalg.norm(t,axis=1,keepdims=True)
    return np.degrees(np.arccos(np.clip((t[:-m]*t[m:]).sum(axis=1),-1,1)))

cases={}
for d in sorted(glob.glob(os.path.join(ANAT,"topcow_mr_*"))):
    n=os.path.basename(d)
    if n in EXCL: continue
    cases[n]=load(os.path.join(d,"Centrelines_comb"))
cases["HOST"]=load(HOST)

print("Rc_min calibration  (target: cohort median 3.09, HOST 2.49, 0/22 tighter than host)")
for m in range(1,9):
    vals={}
    for n,(p,r) in cases.items():
        s=arclength(p); g=s>=130.0
        k=menger_span(p,m); sk=s[m:-m]
        kk=k[sk>=130.0]
        vals[n]=1.0/kk.max() if kk.size and kk.max()>0 else np.inf
    coh=[v for n,v in vals.items() if n!="HOST"]
    n_tighter=sum(1 for v in coh if v<vals["HOST"])
    print("  m=%d (%.0fmm span) median=%.2f  HOST=%.2f  tighter=%d/%d  min=%.2f"%(
        m,2*m,np.median(coh),vals["HOST"],n_tighter,len(coh),min(coh)))

print()
print("max bend calibration (target: cohort max 74.6, HOST 91.3, 0/22 sharper)")
for m in range(1,13):
    vals={}
    for n,(p,r) in cases.items():
        s=arclength(p)
        b=bend_span(p,m); sb=s[m:len(p)-m]
        bb=b[sb[:len(b)]>=130.0]
        vals[n]=bb.max() if bb.size else np.nan
    coh=[v for n,v in vals.items() if n!="HOST"]
    print("  m=%d  cohort max=%.1f median=%.1f  HOST=%.1f  sharper=%d/%d"%(
        m,max(coh),np.median(coh),vals["HOST"],sum(1 for v in coh if v>vals["HOST"]),len(coh)))
