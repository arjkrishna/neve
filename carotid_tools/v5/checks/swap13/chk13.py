#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\chk13.py (= /sp/v5_va/swap13/chk13.py in the build containers)
"""v5_swap13: checks/chk_one.py UNCHANGED (strict mesh audit = item 1, route + VA detail k24 = item 2) on one swapped
anatomy; the only diff is the output folder (/sp/v5_va/swap13/chk_out instead of checks/out, which keeps the
pre-swap records).    python3 chk13.py <name>   (neve-build-meshlab, /repo read-only)"""
import sys
sys.path.insert(0, "/sp/v5_va/checks")
import chk_one as C  # noqa: E402
C.OUT = "/sp/v5_va/swap13/chk_out"
sys.exit(C.main())
