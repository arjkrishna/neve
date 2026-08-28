import sys
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
EX = ["topcow_mr_013","topcow_mr_014","topcow_mr_015"]
iv = DualDeviceNavTopBrain(anatomy_dir="/opt/eve_training/results_topbrain/anatomies",
                           seed=42, episodes_between_change=1, exclude=EX)
vt = iv.vessel_tree
print("loader OK :", type(vt).__name__)
print("anatomies :", len(getattr(vt,'anatomy_names',[]) or getattr(vt,'_names',[])))
names = getattr(vt,'anatomy_names',None) or getattr(vt,'_names',None)
print("retained  :", names)
print("mesh_path :", vt.mesh_path)
print("branches  :", [str(b.name) for b in vt.branches][:6], "...")
print("insertion :", vt.insertion.position, vt.insertion.direction)
print("fingerprint:", getattr(vt,'mesh_fingerprint', getattr(vt,'fingerprint','n/a')))
