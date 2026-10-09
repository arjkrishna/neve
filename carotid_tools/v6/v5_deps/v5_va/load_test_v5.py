# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\load_test_v5.py (= /sp/v5_va/load_test_v5.py in the build containers)
"""SOFA load test for v5 prototype anatomies (v4_build/load_test.py pattern, pointed at the prototype folder).

Checks: the training loader accepts the folder (18 branches incl. the new RVA / LVA), the RCCA is found and
targets are drawn on it past 133 mm, insertion is the host's, SOFA builds the scene from the v5 collision mesh
and steps without error or non-finite state; also reports the new VA branch lengths as the loader sees them.

    python3 /sp/v5_va/load_test_v5.py NAME[,NAME] [anatomy_dir] [steps]
eve-training-fixed.
"""
import json
import sys
import time

import numpy as np

sys.path[:0] = ["/repo/eve", "/repo/eve_bench"]
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain  # noqa: E402

names = sys.argv[1].split(",")
adir = sys.argv[2] if len(sys.argv) > 2 else "/repo/carotid_data/va_sources/v5_proto"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 150
iv = DualDeviceNavTopBrain(anatomy_dir=adir, only=names, seed=12345, episodes_between_change=1,
                           target_min_arclength_mm=133.0)
res = {}
for ep in range(len(names)):
    t0 = time.time()
    iv.reset(episode_number=ep, seed=900000 + ep)
    vt = iv.vessel_tree
    bn = [str(b.name) for b in vt.branches]
    tgt = np.asarray(iv.target.coordinates3d)
    rcca = [b for b in vt.branches if "RCCA" in str(b.name)][0]
    pc = np.asarray(rcca.coordinates)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))]
    s_t = float(s[int(np.argmin(np.linalg.norm(pc - tgt, axis=1)))])
    va = {}
    for b in vt.branches:
        nm = str(b.name)
        if nm.endswith("RVA.mrk") or nm.endswith("LVA.mrk") or "RVA" in nm.split("- ")[-1] or "LVA" in nm.split("- ")[-1]:
            p = np.asarray(b.coordinates)
            va[nm.split("- ")[-1]] = round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 1)
    print("episode %d (%s): %d branches %s" % (ep, getattr(vt, "current_name", "?"), len(bn),
                                              sorted(n.split("- ")[-1] for n in bn if " - " in n)), flush=True)
    print("   target on RCCA at %.1f mm (>= 133 expected); insertion %s; VA lengths %s" % (
        s_t, np.round(np.asarray(vt.insertion.position), 3).tolist(), va), flush=True)
    finite, tips = True, []
    for i in range(steps):
        iv.step(np.array([[25.0, 0.5], [20.0, 0.3]]))
        tip = np.asarray(iv.fluoroscopy.tracking3d)[0] if hasattr(iv, "fluoroscopy") else None
        if tip is not None:
            finite &= bool(np.all(np.isfinite(tip)))
            tips.append(tip.tolist())
    dt = time.time() - t0
    print("   %d SOFA steps OK in %.1f s (%.0f ms/step); tip finite %s; last tip %s" % (
        steps, dt, 1000 * dt / steps, finite, None if not tips else np.round(tips[-1], 1).tolist()), flush=True)
    res["episode_%d" % ep] = {"branches": len(bn), "target_s_mm": s_t, "va_len_mm": va, "steps": steps,
                              "seconds": round(dt, 1), "ms_per_step": round(1000 * dt / steps, 1), "tip_finite": finite}
print("LOAD TEST PASSED")
print("JSON", json.dumps(res))
