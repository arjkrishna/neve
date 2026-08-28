import csv,glob,json,os
import numpy as np
exec(open("monitoring/attack2_holdout_geometry.py").read().split("def analyse")[0].replace("main()",""))
M=4;OFF=33.5
EXCL={"topcow_mr_013","topcow_mr_014","topcow_mr_015"}
HOLD=["topcow_mr_004","topcow_mr_008","topcow_mr_017","topcow_mr_023"]
def menger(p,m):
    a=p[:-2*m];b=p[m:-m];c=p[2*m:]
    ab=np.linalg.norm(b-a,axis=1);bc=np.linalg.norm(c-b,axis=1);ca=np.linalg.norm(a-c,axis=1)
    ar=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1);den=ab*bc*ca
    return np.where(den>1e-12,4*ar/np.maximum(den,1e-12),0.)
def geom(tag):
    p,r=read_curve(os.path.join(ANAT,tag,"Centrelines_comb",RCCA_FILE))
    s=arclength(p);k=menger(p,M);sk=s[M:len(p)-M]
    Rc=np.interp(s,sk,1.0/np.maximum(k,1e-9))
    return s,r,Rc
G={os.path.basename(d):geom(os.path.basename(d)) for d in sorted(glob.glob(os.path.join(ANAT,"topcow_mr_*")))
   if os.path.basename(d) not in EXCL}
short={"topcowmr%s"%t[-3:]:t for t in G}
R="saved/eve_paper/neurovascular/full/mesh_ben/2026-07-25_022443_rcca_p2_teacher_v1bp/checkpoints/eval_anatomies_checkpoint2002292"
rows=list(csv.DictReader(open(os.path.join(R,"episodes.csv"))))
def feats(tag,st):
    s,r,Rc=G[tag]
    m=(s>=133.0)&(s<=st)
    return dict(st=st,
      minRc=float(Rc[m].min()) if m.sum() else 30.0,
      minr =float(r[m].min())  if m.sum() else 4.0,
      graft_mm=float(max(0.0,st-133.0)),
      turn=float(np.degrees(np.trapezoid(1.0/np.maximum(Rc[m],1e-9),s[m]))) if m.sum()>1 else 0.0)
data=[]
for x in rows:
    tag=short[x["anatomy"]]; st=float(x["path_len_mm"])-OFF
    f=feats(tag,st); f["y"]=int(x["success"]); f["a"]=tag; data.append(f)
print("FAILURES vs SUCCESSES on candidate features")
for k in ("st","graft_mm","minRc","minr","turn"):
    fv=[d[k] for d in data if not d["y"]]; sv=[d[k] for d in data if d["y"]]
    print("  %-9s fail med=%8.2f range %8.2f..%8.2f | succ med=%8.2f range %8.2f..%8.2f"%(
      k,np.median(fv),min(fv),max(fv),np.median(sv),min(sv),max(sv)))
# logistic on (graft_mm, minRc) via simple IRLS
X=np.column_stack([np.ones(len(data)),[d["graft_mm"] for d in data],[1.0/d["minRc"] for d in data]])
y=np.array([d["y"] for d in data],float)
b=np.zeros(3)
for _ in range(200):
    p=1/(1+np.exp(-X@b)); W=np.maximum(p*(1-p),1e-6)
    b=b+np.linalg.solve(X.T@(X*W[:,None])+1e-6*np.eye(3),X.T@(y-p))
print("\nlogit(p_success) = %.3f + %.4f*graft_mm + %.3f*(1/minRc)"%(b[0],b[1],b[2]))
p=1/(1+np.exp(-X@b))
print("in-sample: mean pred %.3f vs observed %.3f ; AUC-ish sep: fail mean %.3f succ mean %.3f"%(
  p.mean(),y.mean(),p[y==0].mean(),p[y==1].mean()))
json.dump({"b":b.tolist()},open("monitoring/attack2_logit.json","w"))
# --- Monte Carlo target distribution over all 22 ---
print("\nMONTE-CARLO all-22 prediction (targets uniform over RCCA stations s in [40, L-8])")
res={}
for tag,(s,r,Rc) in G.items():
    L=s[-1]; sts=s[(s>=40.0)&(s<=L-8.0)]
    ff=[feats(tag,float(t)) for t in sts]
    Xa=np.column_stack([np.ones(len(ff)),[d["graft_mm"] for d in ff],[1.0/d["minRc"] for d in ff]])
    pa=1/(1+np.exp(-Xa@b)); res[tag]=(float(pa.mean()),len(sts))
for tag in sorted(res,key=lambda t:-res[t][0]):
    m="*" if tag in HOLD else " "
    print("  %s%-14s pred=%.3f  (n_stations=%d)"%(m,tag,res[tag][0],res[tag][1]))
ph=np.mean([res[t][0] for t in HOLD]); pa=np.mean([res[t][0] for t in res])
po=np.mean([res[t][0] for t in res if t not in HOLD])
print("\n  mean pred holdout4 = %.3f   other18 = %.3f   all22 = %.3f"%(ph,po,pa))
print("  observed holdout4 = 0.918 -> calibrated all22 = 0.918 * %.3f/%.3f = %.3f"%(pa,ph,0.918*pa/ph))
print("  calibrated other18 = 0.918 * %.3f/%.3f = %.3f"%(po,ph,0.918*po/ph))
