"""CHECK 9 -- SOFA loadability on the 216-anatomy three-source carotid set.
Loads, resets (cold + warm) and steps each anatomy with a scripted forward
insertion. NO policy. One anatomy at a time via only=[name].
argv[1] = comma-separated anatomy names, argv[2] = n forward steps (default 60)."""
import sys, os, time, json, traceback
sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
import numpy as np
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain

AD = "/opt/eve_training/carotid/anatomies"
names = sys.argv[1].split(",")
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ACT = np.array([[25.0, 0.0], [0.0, 0.0]], dtype=np.float32)   # 25 mm/s fwd, device1 only

out = {}
for nm in names:
    rec = dict(load=None, reset_cold=None, reset_warm=None, step_mean=None, step_max=None,
               ntrack=None, adv=None, tip_path=None, ins=None, maxabs=None,
               nonfinite=0, sim_error=None, maxjump=None, status="ok")
    try:
        t0 = time.time()
        iv = DualDeviceNavTopBrain(anatomy_dir=AD, seed=7, episodes_between_change=10**9,
                                   only=[nm])
        rec["load"] = time.time() - t0

        t1 = time.time(); iv.reset(episode_number=0, seed=7); rec["reset_cold"] = time.time() - t1
        t1 = time.time(); iv.reset(episode_number=1, seed=7); rec["reset_warm"] = time.time() - t1

        p0 = np.asarray(iv.fluoroscopy.tracking3d, float)
        rec["ntrack"] = int(len(p0)); tip0 = p0[0].copy()
        prev = tip0.copy(); path = 0.0; dts = []; jumps = []
        for k in range(NSTEP):
            ts = time.time(); iv.step(action=ACT); dts.append(time.time() - ts)
            p = np.asarray(iv.fluoroscopy.tracking3d, float)
            if not np.isfinite(p).all(): rec["nonfinite"] += 1
            else:
                d = float(np.linalg.norm(p[0] - prev)); jumps.append(d)
                path += d; prev = p[0].copy()
        p1 = np.asarray(iv.fluoroscopy.tracking3d, float)
        rec["step_mean"] = float(np.mean(dts)); rec["step_max"] = float(np.max(dts))
        rec["adv"] = float(np.linalg.norm(p1[0] - tip0))
        rec["tip_path"] = path
        rec["maxabs"] = float(np.abs(p1[np.isfinite(p1).all(1)]).max())
        rec["ins"] = [float(x) for x in iv.device_lengths_inserted]
        rec["maxjump"] = float(max(jumps)) if jumps else None
        rec["sim_error"] = bool(getattr(iv.simulation, "simulation_error", False))
        if rec["nonfinite"]: rec["status"] = "NONFINITE TRACKING x%d" % rec["nonfinite"]
        elif rec["sim_error"]: rec["status"] = "SIM ERROR (NaN reset by SOFA)"
        elif rec["adv"] < 1.0: rec["status"] = "STEPPED BUT TIP DID NOT MOVE"
        elif rec["maxabs"] > 1e4: rec["status"] = "DIVERGED (coords blew up)"
        iv.close()
    except Exception as e:
        rec["status"] = "FAIL: " + repr(e)[:90]; traceback.print_exc()
    out[nm] = rec
    print("ROW " + json.dumps({nm: rec})); sys.stdout.flush()
print("ALLDONE " + json.dumps(out))
