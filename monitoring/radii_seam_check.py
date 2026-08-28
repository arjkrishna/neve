"""Do the cohort's DECLARED radii diverge from the host before the course seam,
and in which direction? The graft pipeline overwrote the host's own radii over
~105-130 mm with a smoothstep ramp (TOPBRAIN_PIPELINE.md, 'Radius repair')."""
import glob, os, sys
import numpy as np
sys.path.insert(0,"/opt/eve_training/eve"); sys.path.insert(0,"/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

def prof(cl):
    b=[x for x in load_branches(cl) if "RCCA" in str(x.name).upper()][0]
    c=np.asarray(b.coordinates,float); r=np.asarray(b.radii,float)
    s=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))))
    return s,r

hs,hr = prof("/opt/eve_training/eve_bench/data/dualdevicenav/Centrelines_comb")
HOLD=["topcow_mr_004","topcow_mr_008","topcow_mr_017","topcow_mr_023"]
grid=np.array([90,100,105,110,115,120,125,130,133,140])
print(f"{'s (mm)':>8} {'HOST r':>8} | " + " ".join(f"{h[-6:]:>8}" for h in HOLD))
print("-"*64)
co={h:prof(f"/opt/eve_training/results_topbrain/anatomies/{h}/Centrelines_comb") for h in HOLD}
for g in grid:
    row=[f"{np.interp(g,hs,hr):8.2f}"]
    for h in HOLD:
        s,r=co[h]; row.append(f"{np.interp(g,s,r):8.2f}")
    print(f"{g:8.0f} {row[0]} | " + " ".join(row[1:]))
# where do radii first diverge?
for h in HOLD:
    s,r=co[h]
    gg=np.arange(0,160,0.25)
    d=np.abs(np.interp(gg,s,r)-np.interp(gg,hs,hr))
    i=np.argmax(d>0.10); j=np.argmax(d>0.25)
    print(f"  {h}: |dr|>0.10mm at s={gg[i]:.2f}  |dr|>0.25mm at s={gg[j]:.2f}")
