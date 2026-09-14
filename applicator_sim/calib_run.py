"""Stage-3 calibration runner (CONTAINER, SofaPython3 v22.12, py3.8).  Units mm, mN, s.

The device-tolerance parameters of the calibration are now handled by scene.build_scene itself (config keys
sph_r_scale [-] and sph_dz_mm [mm]; defaults 1.0 / 0.0 = the MEASURED sphere pack), so this file is only an alias of
run_insertion.py kept for the calibration batch files.  (The stage-3 version monkey-patched scene.build_scene; with
the finisher scene that would apply the perturbation twice.  The stage-3 runs themselves were made with the version
hashed in calibration/best.json.)

    python3 /app/calib_run.py --batch /out/runs/batch_C3_lhs_0.json        # same CLI as run_insertion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_insertion  # noqa: E402

if __name__ == "__main__":
    run_insertion.main()
    # SofaPython3 v22.12 segfaults at interpreter teardown once a Python ForceField (ties.py) has existed; every
    # output is closed by run_insertion, so leave without the finaliser (same as run_insertion.py).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
