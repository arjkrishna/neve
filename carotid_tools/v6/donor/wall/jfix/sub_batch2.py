#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\jfix\sub_batch2.py (= /sp/v6/wall/jfix/sub_batch2.py in the build containers)
"""v6_wall (e) junction fix re-run: as sub_batch.py (same requests, same rule, same per-donor env) but with the junction
bridge (jfix/junction6.py) into /v6w/subclavian_j2/<donor_id> (idempotent, .seg_done). publish_j2.py later copies the
junction surfaces over /v6w/subclavian/<donor_id>/ after checking the subclavian surfaces are bit-identical.
    python3 sub_batch2.py <rule> <unit,...|all|rev> [out_root, default /v6w/subclavian_j2]"""
import json, os, subprocess, sys, time
rule, which = sys.argv[1:3]
ROOT = sys.argv[3] if len(sys.argv) > 3 else "/v6w/subclavian_j2"
D = json.load(open("/sp/v6/wall/donors.json"))
REQ = "/v6w/subclavian/req"
units = sorted(f[:-5] for f in os.listdir(REQ) if f.endswith(".json"))
if which == "rev":
    units = units[::-1]
elif which != "all":
    units = which.split(",")
for u in units:
    out = os.path.join(ROOT, u)
    if os.path.exists(os.path.join(out, ".seg_done")) or os.path.exists(os.path.join(out, ".running")):
        print("skip", u, flush=True)
        continue
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, ".running"), "w").write(time.strftime("%F %T"))
    if u == "vmr0248":
        script, env_u = "/sp/v6/wall/vmrseg6.py", {"VMR_OPEN_BRIGHT": "1"}
    else:
        script, env_u = "/sp/v6/wall/ctseg6.py", D[u]["env"]
    env = dict(os.environ, **env_u, V6_OUT=out, V6_RULE=rule, V6_SUB_REQ="%s/%s.json" % (REQ, u), V6_JUNC_BRIDGE="1")
    t0 = time.time()
    with open(os.path.join(out, "seg.log"), "w") as lf:
        rc = subprocess.call(["python3", script], env=env, stdout=lf, stderr=subprocess.STDOUT)
    print("%s rule %s rc %d %.0f s" % (u, rule, rc, time.time() - t0), flush=True)
    os.remove(os.path.join(out, ".running"))
    if rc == 0:
        open(os.path.join(out, ".seg_done"), "w").write("%s %s\n" % (rule, time.strftime("%F %T")))
