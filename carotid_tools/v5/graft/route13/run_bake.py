#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\route13\run_bake.py (= /sp/v5_va/route13/run_bake.py in the build containers)
"""v5_route13: bake ONE scratch anatomy with bake_full/bake_full_one.py (unchanged), the ownership chosen by env
OWNERSHIP: "standard" -> bake_v5b as the full bake; "prio" -> bake_v5c (route-priority R1-R3) installed first.
A5_DIR / BF_DIR / ISO_DIR (route13 scratch) are read by bake_full_one.py itself.   python3 run_bake.py <name>"""
import os
import runpy
import sys

sys.path.insert(0, "/sp/v5_va")
if os.environ.get("OWNERSHIP", "standard") == "prio":
    import bake_v5c  # noqa: F401  (patches bake_v5b / bake_meshes_v4 at run time)
runpy.run_path("/sp/v5_va/bake_full/bake_full_one.py", run_name="__main__")
