import glob, json, os
import numpy as np
exec(open("monitoring/attack2_holdout_geometry.py").read().split("def analyse")[0].replace("main()",""))
EXCL={"topcow_mr_013","topcow_mr_014","topcow_mr_015"}
HOLD={"topcow_mr_004","topcow_mr_008","topcow_mr_017","topcow_mr_023"}
M=4          # 8 mm span, calibrated to reproduce prior session's Rc/bend numbers
GS=133.0     # graft region per measured course departure

def menger_span(p,m):
    a=p[:-2*m]; b=p[m:-m]; c=p[2*m:]
    ab=np.linalg.norm(b-a,axis=1); bc=np.linalg.norm(c-b,axis=1); ca=np.linalg.norm(a-c,axis=1)
    area=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1); den=ab*bc*ca
    return np.where(den>1e-12,4*area/np.maximum(den,1e-12),0.0)
def bend_span(p,m):
    t=p[m:]-p[:-m]; t=t/np.linalg.norm(t,axis=1,keepdims=True)
    return np.degrees(np.arccos(np.clip((t[:-m]*t[m:]).sum(axis=1),-1,1)))
def kink(p):
    d=np.diff(p,axis=0); d=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-9)
    return np.degrees(np.arccos(np.clip((d[:-1]*d[1:]).sum(axis=1),-1,1)))

def one(folder,tag):
    bl=[]
    for f in sorted(glob.glob(os.path.join(folder,"*.mrk.json"))):
        n=os.path.basename(f)[:-9]; p,r=read_curve(f)
        if r is None: continue
        bl.append((n,p,r))
    _,p,r=[b for b in bl if b[0].endswith("- RCCA")][0]
    s=arclength(p); L=float(s[-1]); g=s>=GS
    k=menger_span(p,M); sk=s[M:len(p)-M]; kg=k[sk>=GS]
    b=bend_span(p,M); sb=s[M:len(p)-M]; bg=b[sb[:len(b)]>=GS]
    kk=kink(p); kkg=kk[s[1:-1]>=GS]
    d={"a":tag,"L":L,"gL":L-float(s[g][0]),
       "chord":float(np.linalg.norm(p[-1]-p[g][0])),
       "Rcmin":float(1/kg.max()) if kg.size else np.nan,
       "Rc_p05":float(1/np.percentile(kg,95)),
       "kp95":float(np.percentile(kg,95)),"kp05":float(np.percentile(kg,5)),
       "bendmax":float(bg.max()),"turn":float(kkg.sum()),
       "flr":float((r[s>=130.0]<=2.0).mean()),
       "rmin":float(r[g].min()),"rmed":float(np.median(r[g])),"rterm":float(r[-1])}
    d["tort"]=d["gL"]/d["chord"]
    oth=[(n,bp,br) for n,bp,br in bl if not n.endswith("- RCCA") and not (" - " not in n and bp[:,2].min()>500.0)]
    q,qr=p[g],r[g]; gaps=np.full(len(q),np.inf)
    for n,bp,br in oth:
        dd=np.linalg.norm(q[:,None,:]-bp[None,:,:],axis=2)
        gaps=np.minimum(gaps,(dd-qr[:,None]-br[None,:]).min(axis=1))
    d["clrmin"]=float(gaps.min()); d["clrp05"]=float(np.percentile(gaps,5))
    d["n035"]=int((gaps<0.35).sum()); d["n10"]=int((gaps<1.0).sum()); d["n20"]=int((gaps<2.0).sum())
    return d

rows=[one(os.path.join(d,"Centrelines_comb"),os.path.basename(d))
      for d in sorted(glob.glob(os.path.join(ANAT,"topcow_mr_*")))]
rows=[x for x in rows if x["a"] not in EXCL]
host=one(HOST,"HOST")
json.dump({"rows":rows,"host":host},open("monitoring/attack2_geom_m4.json","w"),indent=1)

def z(vals):
    v=np.array(vals,float); return (v-v.mean())/v.std(ddof=0)
# difficulty: higher = harder
comp = {}
feats = {
 "1/Rcmin": [1.0/x["Rcmin"] for x in rows],
 "kp95":    [x["kp95"] for x in rows],
 "bendmax": [x["bendmax"] for x in rows],
 "turn":    [x["turn"] for x in rows],
 "tort":    [x["tort"] for x in rows],
 "gL":      [x["gL"] for x in rows],
 "-clrmin": [-x["clrmin"] for x in rows],
 "-rmin":   [-x["rmin"] for x in rows],
 "flr":     [x["flr"] for x in rows],
}
Z={k:z(v) for k,v in feats.items()}
# core kinematic difficulty (curvature family) + narrowness + clearance, equal weight per family
curv = (Z["1/Rcmin"]+Z["kp95"]+Z["bendmax"]+Z["turn"]+Z["tort"])/5.0
narrow = (Z["-rmin"]+Z["flr"])/2.0
clear = Z["-clrmin"]
length = Z["gL"]
D = curv + narrow + clear + 0.5*length
for i,x in enumerate(rows):
    x["curv"]=float(curv[i]); x["narrow"]=float(narrow[i]); x["clear"]=float(clear[i]); x["D"]=float(D[i])

order=sorted(rows,key=lambda x:-x["D"])
print("%-8s %5s %5s %6s %6s %6s %5s %5s %5s %5s %5s %5s %6s %6s %6s %6s"%(
 "anat","L","gL","Rcmin","Rcp05","kp95","bend","turn","tort","clrmn","clrp5","flr%","rmin","curvZ","narrZ","D"))
for rk,x in enumerate(order,1):
    m="*" if x["a"] in HOLD else " "
    print("%2d%s%-6s %5.0f %5.0f %6.2f %6.2f %6.3f %5.1f %5.0f %5.2f %5.2f %5.2f %6.1f %6.2f %6.2f %6.2f %6.2f"%(
      rk,m,x["a"].replace("topcow_mr_","m"),x["L"],x["gL"],x["Rcmin"],x["Rc_p05"],x["kp95"],
      x["bendmax"],x["turn"],x["tort"],x["clrmin"],x["clrp05"],100*x["flr"],x["rmin"],
      x["curv"],x["narrow"],x["D"]))
h=host
print("  %-6s %5.0f %5.0f %6.2f %6.2f %6.3f %5.1f %5.0f %5.2f %5.2f %5.2f %6.1f %6.2f"%(
 "HOST",h["L"],h["gL"],h["Rcmin"],h["Rc_p05"],h["kp95"],h["bendmax"],h["turn"],h["tort"],
 h["clrmin"],h["clrp05"],100*h["flr"],h["rmin"]))
print("\nn035 counts (clearance<0.35mm):", {x["a"][-6:]:x["n035"] for x in rows})
print("n<1.0mm:", {x["a"][-6:]:x["n10"] for x in rows if x["n10"]})
holdranks=[(x["a"],i+1) for i,x in enumerate(order) if x["a"] in HOLD]
print("\nHOLDOUT ranks (1=hardest of 22):",holdranks)
print("mean holdout rank %.2f vs null 11.5"%np.mean([r for _,r in holdranks]))
