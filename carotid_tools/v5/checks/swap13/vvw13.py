#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\swap13\vvw13.py (= /sp/v5_va/swap13/vvw13.py in the build containers)
"""v5_swap13: vvw/measure.py UNCHANGED (verify_v5_wall's own right_route_detail + VA wall detail, k24 AND exact
ownership, seam counted 0 as the study's convention, on the undecimated union) on one anatomy; only diff: output
folder /sp/v5_va/swap13/vvw_out (vvw/out keeps the pre-swap records).   python3 vvw13.py <name>"""
import sys
sys.path.insert(0, "/sp/v5_va/vvw")
import measure as M  # noqa: E402
M.OUT = "/sp/v5_va/swap13/vvw_out"
sys.exit(M.main())
