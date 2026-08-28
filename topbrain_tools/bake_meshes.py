#!/usr/bin/env python3
"""Bake each grafted anatomy's collision mesh into its own folder.

These anatomies are FIXED, one per patient, so their meshes belong on disk
next to the centerlines exactly as the shipped patient's does. Generating them
at load time instead would mean paying seconds of marching-cubes per anatomy
switch, and would make the geometry depend on the scikit-image and pyvista
versions of whichever machine happened to run it. Baked once, the anatomy
folder is a self-contained artifact that can be copied to another machine and
produces byte-identical geometry there.

Layout written, matching eve_bench/data/dualdevicenav:

    <anatomy>/Centrelines_comb/*.mrk.json      centerlines + radii
    <anatomy>/vessel_architecture_collision.obj  <- this script

    python topbrain_tools/bake_meshes.py --anatomies topbrain_data/anatomies
"""
import argparse
import glob
import os
import sys

MESH_NAME = "vessel_architecture_collision.obj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomies", default="topbrain_data/anatomies")
    ap.add_argument("--decimate", type=float, default=0.99)
    ap.add_argument("--force", action="store_true",
                    help="re-bake even if the .obj is already there")
    a = ap.parse_args()

    sys.path.insert(0, "/opt/eve_training/eve_bench")
    from eve_bench.dualdevicenav import load_branches
    from eve.intervention.vesseltree.util.meshing import generate_mesh

    folders = sorted(glob.glob(os.path.join(a.anatomies, "*", "Centrelines_comb")))
    if not folders:
        print("no anatomies under %s" % a.anatomies)
        return 1
    print("baking %d meshes (decimate %.3f)\n" % (len(folders), a.decimate))

    print("%-16s %10s %10s" % ("anatomy", "size_kb", "status"))
    for f in folders:
        root = os.path.dirname(f)
        name = os.path.basename(root)
        out = os.path.join(root, MESH_NAME)
        if os.path.exists(out) and not a.force:
            print("%-16s %10.0f %10s" % (name, os.path.getsize(out) / 1024, "kept"))
            continue
        branches = load_branches(f)
        generate_mesh(branches, out, a.decimate)
        print("%-16s %10.0f %10s" % (name, os.path.getsize(out) / 1024, "baked"))

    total = sum(os.path.getsize(os.path.join(os.path.dirname(f), MESH_NAME))
                for f in folders
                if os.path.exists(os.path.join(os.path.dirname(f), MESH_NAME)))
    print("\n%.1f MB of meshes across %d anatomies" % (total / 1e6, len(folders)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
