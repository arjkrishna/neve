#!/usr/bin/env python3
"""Re-cut every v2 collision .obj at a different triangle budget.

bake_meshes_v2.py keeps the 60 k-triangle SDF surface as collision_full.vtp
precisely so the budget is a one-line decision, not a re-bake: SOFA step cost
measured flat from 12 k to 38 k triangles (+55 % over the 3.7 k v1 meshes),
+25 % at 9 k, +5 % at 6 k; lumen deficit 0.12 mm at 20 k, 0.23 at 9 k,
0.33 at 6 k. Quadric, volume-preserving.

    python3 topbrain_tools/recut_obj.py --anatomies <dir> --tris 9000
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", required=True)
    ap.add_argument("--tris", type=int, required=True)
    a = ap.parse_args()
    import pyvista as pv
    from sdf_mesher import decimate_to, mesh_stats
    n = 0
    for f in sorted(glob.glob(os.path.join(a.anatomies, "*", "collision_full.vtp"))):
        root = os.path.dirname(f)
        m = decimate_to(pv.read(f), a.tris)
        pv.save_meshio(os.path.join(root, "vessel_architecture_collision.obj"), m)
        rp = os.path.join(root, "mesh_v2.json")
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        rep["obj"] = dict(rep.get("obj", {}), **mesh_stats(m), recut_to=a.tris)
        json.dump(rep, open(rp, "w", encoding="utf-8"), indent=1)
        n += 1
        print("%-44s %d tris" % (os.path.basename(root)[:44], m.n_cells), flush=True)
    print("re-cut %d meshes at %d triangles" % (n, a.tris))
    return 0


if __name__ == "__main__":
    sys.exit(main())
