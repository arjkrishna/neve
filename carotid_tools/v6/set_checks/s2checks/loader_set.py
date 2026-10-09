#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\loader_set.py (= /sp/v6/s2checks/loader_set.py in the build containers)
"""loader_set.py 1.0 (s2_checks, 2026-10-07): the TRAINING LOADER on the WHOLE folder, as training calls it.

DualDeviceNavTopBrain(anatomy_dir=<tree>) (repo eve / eve_bench first) loads every anatomy of the folder in one
TopBrainAnatomySet; then every anatomy is selected through the set's own fingerprint path (regenerate_to_fingerprint,
as a checkpoint restore does) and checked: 18 branches with the same branch names as the first anatomy, coordinates
finite, radii > 0, mesh_path = that anatomy's collision OBJ (exists), the shared insertion (derived once from the first
anatomy) inside that anatomy's OBJ (vtk enclosed-points), RCCA (the target branch) longer than the target's minimum
arclength (133 mm) + 5 mm, fingerprints unique; the training schedule (_index_for) covers every anatomy in one pass of
n generations. The per-anatomy loader_one runs (checks_v6 part loader, and load_v6_all on the final folder) and the SOFA
runs (one scene load per anatomy) complement this.
    python3 loader_set.py <tree> <out.json>        (eve-training-fixed or neve-build-meshlab; read-only)
"""
import json
import os
import sys
import time

sys.path[:0] = ["/repo/eve", "/repo/eve_bench"]
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from eve_bench.dualdevicenavtopbrain import DualDeviceNavTopBrain  # noqa: E402

VERSION = "loader_set 1.0 (s2_checks, 2026-10-07)"


def arcl(p):
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def main():
    tree, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    dirs = sorted(d for d in os.listdir(tree) if os.path.isdir(os.path.join(tree, d, "Centrelines_comb")))
    iv = DualDeviceNavTopBrain(anatomy_dir=tree, seed=4242, episodes_between_change=1, target_min_arclength_mm=133.0)
    vt = iv.vessel_tree
    res = {"version": VERSION, "tree": tree, "folders": len(dirs), "anatomy_count": vt.anatomy_count,
           "load_s": round(time.time() - t0, 1)}
    names = list(vt.anatomy_names)
    fps = list(vt._fps)
    res["names_equal_folders"] = names == dirs
    res["fingerprints_unique"] = len(set(fps)) == len(fps)
    n = vt.anatomy_count
    cover = sorted(vt._index_for(g) for g in range(n))
    res["schedule_covers_all_in_one_pass"] = cover == list(range(n))
    ins = np.asarray(vt.insertion.position, float)
    res["insertion"] = ins.round(4).tolist()
    ref_names = None
    per, bad = {}, []
    for k, (nm, fp) in enumerate(zip(names, fps)):
        vt.regenerate_to_fingerprint(fp)
        ok, why = True, []
        if vt.current_anatomy != nm:
            ok = False
            why.append("selected %s" % vt.current_anatomy)
        br = list(vt.branches)
        bn = sorted(str(b.name) for b in br)
        if ref_names is None:
            ref_names = bn
        if len(br) != 18:
            ok = False
            why.append("branches %d" % len(br))
        if bn != ref_names:
            ok = False
            why.append("branch names differ")
        fin = all(np.all(np.isfinite(np.asarray(b.coordinates, float))) for b in br)
        rpos = all(np.all(np.asarray(b.radii, float) > 0) for b in br)
        if not fin:
            ok = False
            why.append("non-finite coordinates")
        if not rpos:
            ok = False
            why.append("radius <= 0")
        obj = os.path.join(tree, nm, "vessel_architecture_collision.obj")
        mp_ok = os.path.normpath(str(vt.mesh_path)) == os.path.normpath(obj) and os.path.exists(obj)
        if not mp_ok:
            ok = False
            why.append("mesh_path %s" % vt.mesh_path)
        mesh = pv.read(obj).triangulate().clean()
        sel = pv.PolyData(ins[None, :]).select_enclosed_points(mesh, tolerance=0.0, check_surface=False)
        ins_in = bool(np.asarray(sel["SelectedPoints"], bool)[0])
        if not ins_in:
            ok = False
            why.append("insertion outside the OBJ")
        rc = [b for b in br if str(b.name).endswith("RCCA.mrk")]
        L = arcl(np.asarray(rc[0].coordinates, float)) if rc else 0.0
        if L < 133.0 + 5.0:
            ok = False
            why.append("RCCA %.1f mm < 138" % L)
        per[nm] = {"ok": ok, "why": why, "branches": len(br), "rcca_len_mm": round(L, 1), "insertion_inside": ins_in,
                   "mesh_faces": int(mesh.n_cells)}
        if not ok:
            bad.append(nm)
    res.update({"checked": len(per), "ok": sum(1 for v in per.values() if v["ok"]), "bad": bad,
                "branch_names": ref_names, "rcca_len_mm": {"min": min(v["rcca_len_mm"] for v in per.values()),
                                                            "max": max(v["rcca_len_mm"] for v in per.values())},
                "seconds": round(time.time() - t0, 1), "anatomies": per})
    res["pass"] = bool(res["anatomy_count"] == len(dirs) and res["names_equal_folders"] and res["fingerprints_unique"]
                       and res["schedule_covers_all_in_one_pass"] and not bad and res["checked"] == len(dirs))
    json.dump(res, open(out + ".tmp", "w"), indent=0)
    os.replace(out + ".tmp", out)
    print("LOADER_SET %s" % json.dumps({k: res[k] for k in ("tree", "folders", "anatomy_count", "checked", "ok", "bad", "pass",
                                                           "names_equal_folders", "fingerprints_unique",
                                                           "schedule_covers_all_in_one_pass", "seconds")}))
    os._exit(0)


if __name__ == "__main__":
    main()
