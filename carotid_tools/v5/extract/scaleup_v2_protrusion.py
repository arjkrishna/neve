#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\scaleup_v2_protrusion.py (= /sp/v5_va/scaleup_v2_protrusion.py in the build containers)
"""scaleup_v2_protrusion.py <case> [...] -- (scaleup_fixes, 2026-10-02) PIPELINE v2 protrusion: the source's UNCHANGED pilot
protrusion (pilot_proc_<PILOT_SOURCE>_protrusion.one, imported) on the wall tag PILOT_SEG_TAG (default ct_seg4). The pilot's
ia172 script has the tag only as a function argument; the Song script reads PILOT_SEG_TAG itself. neve-build-stagea."""
import importlib
import os
import sys

sys.path.insert(0, "/sp/v5_va")
PSRC = os.environ["PILOT_SOURCE"]
TAG = os.environ.get("PILOT_SEG_TAG", "ct_seg4")
M = importlib.import_module("pilot_proc_%s_protrusion" % PSRC)
for c in sys.argv[1:]:
    try:
        M.one(c, tag=TAG)
    except Exception as e:  # noqa: BLE001
        print(c, "failed:", repr(e))
