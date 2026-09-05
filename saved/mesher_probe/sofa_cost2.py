"""SOFA step cost between 3.7 k and 20 k triangles: where is the knee?"""
import os, shutil, sys, time, numpy as np, pyvista as pv
sys.path.insert(0, "/opt/eve_training/eve_bench")
P = "/opt/eve_training/saved/mesher_probe"; ROOT = P + "/sofa_cost"
SRC = "/opt/eve_training/topbrain_data/anatomies_v2_test/topcow_mr_001"
full = pv.read(SRC + "/collision_full.vtp")
variants = [("baked_3k7", None)]
for n in (6000, 9000, 12000, 16000):
    tag = "sdfq_%dk" % (n // 1000)
    d = os.path.join(ROOT, tag); os.makedirs(d, exist_ok=True)
    if not os.path.exists(d + "/Centrelines_comb"):
        shutil.copytree(SRC + "/Centrelines_comb", d + "/Centrelines_comb")
    m = full.decimate(1 - n / full.n_cells, volume_preservation=True)
    pv.save_meshio(d + "/vessel_architecture_collision.obj", m)
    variants.append((tag, m.n_cells))
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain
adv = np.array([[12.0, 0.0], [12.0, 0.0]], dtype=np.float32)
grind = np.array([[3.0, 1.5], [3.0, -1.5]], dtype=np.float32)
def t(iv, a, n):
    t0 = time.time()
    for _ in range(n): iv.step(a)
    return (time.time() - t0) / n * 1000
print("%-12s %8s %12s %12s" % ("mesh", "tris", "advance_ms", "contact_ms"))
for tag, n in variants:
    iv = DualDeviceNavTopBrain(anatomy_dir=ROOT, seed=0, episodes_between_change=1, only=[tag])
    rows = []
    for seed in (3, 4):
        iv.reset(episode_number=0, seed=seed); t(iv, adv, 5)
        rows.append((t(iv, adv, 40), t(iv, grind, 60)))
    tris = n if n else pv.read(ROOT + "/" + tag + "/vessel_architecture_collision.obj").n_cells
    print("%-12s %8d %12.1f %12.1f" % (tag, tris, np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])), flush=True)
    iv.close()
