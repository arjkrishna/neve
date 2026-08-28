import glob,json,os
import numpy as np
exec(open("monitoring/attack2_holdout_geometry.py").read().split("def analyse")[0].replace("main()",""))
EXCL={"topcow_mr_013","topcow_mr_014","topcow_mr_015"}
HOLD={"topcow_mr_004","topcow_mr_008","topcow_mr_017","topcow_mr_023"}
M=4; OFF=33.5; CUT1,CUT2=146.0,210.0
S1,S2=CUT1-OFF,CUT2-OFF     # 112.5 , 176.5 in RCCA arclength
def menger(p,m):
    a=p[:-2*m];b=p[m:-m];c=p[2*m:]
    ab=np.linalg.norm(b-a,axis=1);bc=np.linalg.norm(c-b,axis=1);ca=np.linalg.norm(a-c,axis=1)
    ar=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1);den=ab*bc*ca
    return np.where(den>1e-12,4*ar/np.maximum(den,1e-12),0.0)
def bend(p,m):
    t=p[m:]-p[:-m];t=t/np.linalg.norm(t,axis=1,keepdims=True)
    return np.degrees(np.arccos(np.clip((t[:-m]*t[m:]).sum(axis=1),-1,1)))
def kink(p):
    d=np.diff(p,axis=0);d=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-9)
    return np.degrees(np.arccos(np.clip((d[:-1]*d[1:]).sum(axis=1),-1,1)))
def band(folder,tag,lo,hi=np.inf):
    bl=[]
    for f in sorted(glob.glob(os.path.join(folder,"*.mrk.json"))):
        n=os.path.basename(f)[:-9];p,r=read_curve(f)
        if r is None: continue
        bl.append((n,p,r))
    _,p,r=[b for b in bl if b[0].endswith("- RCCA")][0]
    s=arclength(p);L=float(s[-1]);g=(s>=lo)&(s<=hi)
    if g.sum()<3: return None
    k=menger(p,M);sk=s[M:len(p)-M];m1=(sk>=lo)&(sk<=hi);kg=k[m1]
    b=bend(p,M);m2=(sk[:len(b)]>=lo)&(sk[:len(b)]<=hi);bg=b[m2]
    kk=kink(p);m3=(s[1:-1]>=lo)&(s[1:-1]<=hi);kg2=kk[m3]
    oth=[(n,bp,br) for n,bp,br in bl if not n.endswith("- RCCA") and not (" - " not in n and bp[:,2].min()>500.)]
    q,qr=p[g],r[g];gaps=np.full(len(q),np.inf)
    for n,bp,br in oth:
        dd=np.linalg.norm(q[:,None,:]-bp[None,:,:],axis=2)
        gaps=np.minimum(gaps,(dd-qr[:,None]-br[None,:]).min(axis=1))
    return {"a":tag,"L":L,"blen":float(s[g][-1]-s[g][0]),
        "Rcmin":float(1/kg.max()) if kg.size and kg.max()>0 else np.inf,
        "kp95":float(np.percentile(kg,95)),"bendmax":float(bg.max()) if bg.size else np.nan,
        "turn":float(kg2.sum()),"turn_per_mm":float(kg2.sum()/max(s[g][-1]-s[g][0],1e-6)),
        "rmin":float(r[g].min()),"rmed":float(np.median(r[g])),
        "flr":float((r[g]<=2.0).mean()),"clrmin":float(gaps.min()),
        "tort":float((s[g][-1]-s[g][0])/max(np.linalg.norm(p[g][-1]-p[g][0]),1e-9))}
res={}
for d in sorted(glob.glob(os.path.join(ANAT,"topcow_mr_*"))):
    n=os.path.basename(d)
    if n in EXCL: continue
    res[n]=band(os.path.join(d,"Centrelines_comb"),n,S2)
res["HOST"]=band(HOST,"HOST",S2)
print("SIPHON BAND ONLY: RCCA arclength >= %.1f mm  (== planned path_len >= 210 mm, offset %.1f)"%(S2,OFF))
print("%-8s %6s %6s %6s %6s %6s %7s %6s %6s %6s %6s"%("anat","bandmm","Rcmin","kp95","bend","turn","trn/mm","rmin","flr%","clrmn","tort"))
rows=sorted([v for k,v in res.items() if k!="HOST"],key=lambda x:x["Rcmin"])
for x in rows:
    m="*" if x["a"] in HOLD else " "
    print("%s%-7s %6.1f %6.2f %6.3f %6.1f %6.0f %7.2f %6.2f %6.1f %6.2f %6.3f"%(m,x["a"].replace("topcow_mr_","m"),
      x["blen"],x["Rcmin"],x["kp95"],x["bendmax"],x["turn"],x["turn_per_mm"],x["rmin"],100*x["flr"],x["clrmin"],x["tort"]))
x=res["HOST"];print("H%-7s %6.1f %6.2f %6.3f %6.1f %6.0f %7.2f %6.2f %6.1f %6.2f %6.3f"%("HOST",
  x["blen"],x["Rcmin"],x["kp95"],x["bendmax"],x["turn"],x["turn_per_mm"],x["rmin"],100*x["flr"],x["clrmin"],x["tort"]))
json.dump({k:v for k,v in res.items()},open("monitoring/attack2_siphonband.json","w"),indent=1)
