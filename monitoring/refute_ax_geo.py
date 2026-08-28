import numpy as np, json
MINR,MAXR,K,MINTOL=2.0,12.0,1.5,2.0
res={}
from eve_bench.dualdevicenav import DualDeviceNav
hv = DualDeviceNav().vessel_tree
def box(vt):
    cs=vt.coordinate_space
    return [ [round(float(x),6) for x in cs.low], [round(float(x),6) for x in cs.high] ]
def rcca_prof(vt):
    for b in vt.branches:
        if "RCCA" in b.name:
            c=np.asarray(b.coordinates,dtype=np.float64); r=np.asarray(b.radii,dtype=np.float64)
            s=np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))])
            return s,r
    return None,None
res["host_box"]=box(hv)
hs,hr=rcca_prof(hv)
res["host"]={"s":hs.tolist(),"r":hr.tolist()}
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
iv=DualDeviceNavTopBrain(anatomy_dir="/opt/eve_training/results_topbrain/anatomies",seed=42,
    episodes_between_change=1,exclude=["topcow_mr_013","topcow_mr_014","topcow_mr_015"])
vt=iv.vessel_tree
res["cohort"]={}; res["cohort_box"]={}
for n in list(vt.anatomy_names):
    vt.regenerate_to_fingerprint(vt.mesh_fingerprint if False else n.replace("_",""))
    got=vt.current_anatomy
    s,r=rcca_prof(vt)
    res["cohort"][got]={"s":s.tolist(),"r":r.tolist()}
    res["cohort_box"][got]=box(vt)
json.dump(res,open("/opt/eve_training/results_topbrain/../monitoring_out_geo.json","w")) if False else None
json.dump(res,open("/tmp/o.json","w"))
import shutil; shutil.copy("/tmp/o.json","/out/geo.json")
print("host box",res["host_box"])
bs=set(json.dumps(v) for v in res["cohort_box"].values())
print("n distinct cohort boxes",len(bs))
for k,v in list(res["cohort_box"].items())[:3]: print(k,v)
print("n cohort",len(res["cohort"]))
