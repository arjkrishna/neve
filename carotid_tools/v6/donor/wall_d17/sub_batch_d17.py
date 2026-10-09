#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall_d17\sub_batch_d17.py (= /sp/v6/wall_d17/sub_batch_d17.py in the build containers)
"""sub_batch_d17.py (s2_harden_wall_bake, D17): the subclavian request runs for D17 = stage 1's jfix/sub_batch2.py
(same per-donor env from wall/donors.json, ctseg6.py / vmrseg6.py, V6_JUNC_BRIDGE=1, V6_SUB_CLEAN=1 default) with the
D17 requests (/v6w/d17/req: the subclavian to the end of its segmentation). Claim dirs (atomic mkdir) so two
containers can share a rule list; idempotent (.seg_done).
    python3 sub_batch_d17.py <rule v5|plateau> <unit,...|all|rev> <out_root>"""
import json
import os
import subprocess
import sys
import time

rule, which, ROOT = sys.argv[1:4]
D = json.load(open("/sp/v6/wall/donors.json"))
REQ = "/v6w/d17/req"
units = sorted(f[:-5] for f in os.listdir(REQ) if f.endswith(".json") and not f.startswith("_"))
if which == "rev":
    units = units[::-1]
elif which != "all":
    units = which.split(",")
os.makedirs(os.path.join(ROOT, "_claim"), exist_ok=True)
fails = []
for u in units:
    out = os.path.join(ROOT, u)
    if os.path.exists(os.path.join(out, ".seg_done")):
        continue
    try:
        os.mkdir(os.path.join(ROOT, "_claim", u))
    except FileExistsError:
        continue
    os.makedirs(out, exist_ok=True)
    if u == "vmr0248":
        script, env_u = "/sp/v6/wall/vmrseg6.py", {"VMR_OPEN_BRIGHT": "1"}
    else:
        script, env_u = "/sp/v6/wall/ctseg6.py", D[u]["env"]
    env = dict(os.environ, **env_u, V6_OUT=out, V6_RULE=rule, V6_SUB_REQ="%s/%s.json" % (REQ, u), V6_JUNC_BRIDGE="1")
    t0 = time.time()
    with open(os.path.join(out, "seg.log"), "w") as lf:
        rc = subprocess.call(["python3", script], env=env, stdout=lf, stderr=subprocess.STDOUT)
    print("%s rule %s rc %d %.0f s" % (u, rule, rc, time.time() - t0), flush=True)
    if rc == 0:
        open(os.path.join(out, ".seg_done"), "w").write("%s %s\n" % (rule, time.strftime("%F %T")))
    else:
        fails.append(u)
        os.rmdir(os.path.join(ROOT, "_claim", u))
print("SUB_BATCH_D17_DONE rule %s fails %s" % (rule, fails), flush=True)
