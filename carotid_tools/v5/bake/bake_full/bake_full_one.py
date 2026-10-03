#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\bake_full_one.py (= /sp/v5_va/bake_full/bake_full_one.py in the build containers)
"""v5_bake_full: bake ONE anatomy of carotid_data/anatomies_v5 in place with /sp/v5_va/bake_v5b.py (bake_meshes_v4
unchanged + bake_v5 sections + the user handle rule, D1-D8) -- exactly what `bake_v5b.py --only <name>` does
(B.bake_anatomy -> record_provenance), plus the bookkeeping the full bake needs:

  * the done-marker <BF>/_done/<name>.bake.json (written last; the marker, not mesh_v5.json, says "baked"): wall
    time, peak RSS (VmHWM) and the container's memory.peak, every bake_one call (far clamp as v4 / flipped), rung,
    flags, VA tubed mm / %, and the error on failure (then <name>.bake_failed.json instead, so the worker moves on);
  * a rerun without the marker first removes this anatomy's own partial outputs (mesh_v5.json, collision_full.vtp,
    the OBJ, the iso dump), so a bake cut off half way is redone from scratch, never half-kept.

Outputs in the anatomy folder: vessel_architecture_collision.obj, collision_full.vtp, mesh_v5.json, provenance.json
(+ "bake_v5"). The undecimated union goes to carotid_data/va_sources/v5_reg/iso_v5/<name>.vtp (outside anatomies_v5).
    python3 bake_full_one.py <name>        (neve-build-meshlab)
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, "/sp/v5_va")
import bake_v5b as V  # noqa: E402  (installs bake_one_v5b / bake_anatomy_v5b into bake_meshes_v4)

B = V.B
A5 = os.environ.get("A5_DIR", "/repo/carotid_data/anatomies_v5")
BF = os.environ.get("BF_DIR", "/repo/carotid_data/va_sources/v5_reg/bake_full")
ISO = os.environ.get("ISO_DIR", "/repo/carotid_data/va_sources/v5_reg/iso_v5")


def vm_hwm_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


def cg_peak():
    try:
        return int(open("/sys/fs/cgroup/memory.peak").read().strip())
    except Exception:                                                    # noqa: BLE001
        return -1


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    name = sys.argv[1]
    root = os.path.join(A5, name)
    os.makedirs(os.path.join(BF, "_done"), exist_ok=True)
    marker = os.path.join(BF, "_done", name + ".bake.json")
    failed = os.path.join(BF, "_done", name + ".bake_failed.json")
    if os.path.exists(marker):
        print("done already:", marker, flush=True)
        return 0
    # a bake cut off half way: remove this anatomy's own partial outputs (never anything else)
    for f in (V.REPORT_V5, B.FULL_NAME, B.MESH_NAME):
        p = os.path.join(root, f)
        if os.path.exists(p):
            print("removing partial output", p, flush=True)
            os.remove(p)
    for p in (os.path.join(ISO, name + ".vtp"), os.path.join(ISO, "_flip", name + ".vtp")):
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(failed):
        os.remove(failed)
    T0 = time.time()
    calls = []
    orig = B.bake_one

    def bake_one_logged(root_, spacing, full_tris, obj_tris, verbose, dump_iso=None, far_clamp=False,
                        far_clamp_why=None):
        rec = {"call": len(calls), "far_clamp": bool(far_clamp), "why": far_clamp_why,
               "started_s": round(time.time() - T0, 1)}
        calls.append(rec)
        t = time.time()
        try:
            rep = orig(root_, spacing, full_tris, obj_tris, verbose, dump_iso, far_clamp, far_clamp_why)
            rec.update(ok=True, passed=bool(rep["pass"]), rung=rep["ladder_v5b"]["rung"],
                       flags=rep["ladder_v5b"]["flags"], attempts=len(rep["attempts"]))
            return rep
        except Exception as e:                                           # noqa: BLE001
            rec.update(ok=False, error="%s: %s" % (type(e).__name__, e))
            raise
        finally:
            rec["seconds"] = round(time.time() - t, 1)
            rec["vmhwm_gb_after"] = round(vm_hwm_kb() / 1048576.0, 3)
            print("   [full] bake_one call %d far_clamp=%s: %s in %.0f s, VmHWM %.2f GB" % (
                rec["call"], far_clamp, "ok" if rec.get("ok") else rec.get("error"), rec["seconds"],
                rec["vmhwm_gb_after"]), flush=True)

    B.bake_one = bake_one_logged
    out = {"name": name, "script": "/sp/v5_va/bake_v5b.py via bake_full_one.py",
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rep, err = None, None
    print("== %s" % name, flush=True)
    try:
        rep = B.bake_anatomy(root, B.SPACING_MM, 60000, 23500, True, ISO)
        V.record_provenance(root, rep)
    except Exception as e:                                               # noqa: BLE001
        traceback.print_exc()
        err = "%s: %s" % (type(e).__name__, e)
    out["seconds"] = round(time.time() - T0, 1)
    out["peak_rss_gb"] = round(vm_hwm_kb() / 1048576.0, 3)
    out["container_memory_peak_gb"] = round(cg_peak() / 1073741824.0, 3)
    out["calls"] = calls
    out["error"] = err
    if rep is None:
        out["passed"] = False
        atomic_json(failed, out)
        print("FULL %s: FAILED %s (%.0f s, peak RSS %.2f GB)" % (name, err, out["seconds"], out["peak_rss_gb"]),
              flush=True)
        return 1
    lad = rep["ladder_v5b"]
    out.update(passed=bool(rep["pass"]), rung=lad["rung"], flags=lad["flags"], union=rep["union"],
               margin_mm=rep["margin_mm"], va_margin_mm=lad["va_margin_mm"], far_clamp=rep["far_clamp"],
               sections_tube_only=rep["sections_tube_only"], suppressed_stretches_mm=rep["suppressed_stretches_mm"],
               va_tubed=lad["va_tubed"], attempts=len(rep["attempts"]), bake_one_seconds=rep["seconds"],
               obj_tris=rep["obj"]["tris"], obj_tris_budget=rep["obj_tris_budget"], full_tris=rep["full"]["tris"],
               routes=rep["routes"], centerline_pts_outside=rep["centerline_pts_outside"],
               right_route_detail=rep["right_route_detail"], decimation=rep["decimation"],
               iso=os.path.join(ISO, name + ".vtp"))
    atomic_json(marker, out)
    print("FULL %s: pass %s, rung %s, flags %s, tubed %s, %.0f s, peak RSS %.2f GB" % (
        name, rep["pass"], lad["rung"], lad["flags"],
        {k: (v["tubed_mm"], v["tubed_frac"]) for k, v in lad["va_tubed"].items()}, out["seconds"],
        out["peak_rss_gb"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
