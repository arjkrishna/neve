"""H. SOFA load + step, one anatomy at a time (only=[name]). Reports whether the
mesh loads into the SOFA collision model, whether reset() succeeds, and whether
N forward steps advance the guidewire tip."""
import sys, os, time, traceback, json
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
import numpy as np
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
AD="/opt/eve_training/results_topbrain/anatomies"
names=sys.argv[1].split(",")
NSTEP=int(sys.argv[2]) if len(sys.argv)>2 else 40
print(f"{'anatomy':>15} {'load_s':>7} {'reset_s':>8} {'step_s':>7} {'nstep':>5} {'tip_adv_mm':>10} {'ntracking':>9} {'status':>28}")
out={}
for nm in names:
    t0=time.time(); st="ok"; adv=float('nan'); ntr=-1; tl=tr=ts=float('nan')
    try:
        iv=DualDeviceNavTopBrain(anatomy_dir=AD,seed=7,episodes_between_change=10**9,only=[nm])
        tl=time.time()-t0
        t1=time.time(); iv.reset(episode_number=0, seed=7); tr=time.time()-t1
        p0=np.asarray(iv.fluoroscopy.tracking3d,float)
        ntr=len(p0); tip0=p0[0].copy()
        t2=time.time()
        for k in range(NSTEP):
            iv.step(action=np.array([[25.0,0.0],[0.0,0.0]],dtype=np.float32))
        ts=(time.time()-t2)/NSTEP
        p1=np.asarray(iv.fluoroscopy.tracking3d,float)
        adv=float(np.linalg.norm(p1[0]-tip0))
        if adv<1.0: st="STEPPED BUT TIP DID NOT MOVE"
        iv.close()
    except Exception as e:
        st="FAIL: "+repr(e)[:60]; traceback.print_exc()
    print(f"{nm:>15} {tl:7.2f} {tr:8.2f} {ts:7.3f} {NSTEP:5d} {adv:10.2f} {ntr:9d} {st:>28}")
    out[nm]=dict(load=tl,reset=tr,step=ts,adv=adv,ntrack=ntr,status=st)
    sys.stdout.flush()
json.dump(out,open("/opt/eve_training/results_topbrain/_audit_sofa_%s.json"%names[0][-3:],"w"),indent=1)
