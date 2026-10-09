#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\sub_batch.py (= /sp/v6/wall/sub_batch.py in the build containers)
"""v6_wall (e): run the subclavian requests (wall/subclavian_requests.json, from v6_va) per donor unit with the chosen
rule: ctseg6.py (vmrseg6.py for VMR 0248) with V6_SUB_REQ = that donor's entries, V6_OUT = /v6w/subclavian/<donor_id>.
Idempotent (<out>/.seg_done); per-donor request files in /v6w/subclavian/req/<donor_id>.json.
    python3 sub_batch.py <rule> <unit,...|pilot|rest|all>"""
import json, os, subprocess, sys, time
rule, which = sys.argv[1:3]
R = json.load(open("/sp/v6/wall/subclavian_requests.json"))
D = json.load(open("/sp/v6/wall/donors.json"))
PILOT = ["ia172_058", "ia172_069", "ia172_071", "ia172_136", "ia172_159", "ia172_190", "ia172_192", "song_AHMU1218008",
         "song_AHMU1218100", "song_AHMU1218109", "song_AHMU1218113", "song_AHMU1218141", "vmr0248"]
by = {}
for e in R["entries"]:
    by.setdefault(e["donor_id"], []).append(e)
os.makedirs("/v6w/subclavian/req", exist_ok=True)
for d, ee in by.items():
    json.dump(ee, open("/v6w/subclavian/req/%s.json" % d, "w"))
units = {"pilot": PILOT, "rest": [u for u in by if u not in PILOT], "all": PILOT + [u for u in by if u not in PILOT]}.get(
    which, which.split(","))
for u in units:
    if u not in by:
        print("no request for", u, flush=True)
        continue
    out = "/v6w/subclavian/%s" % u
    if os.path.exists(os.path.join(out, ".seg_done")):
        print("skip", u, flush=True)
        continue
    os.makedirs(out, exist_ok=True)
    if u == "vmr0248":
        script, env_u = "/sp/v6/wall/vmrseg6.py", {"VMR_OPEN_BRIGHT": "1"}
    else:
        script, env_u = "/sp/v6/wall/ctseg6.py", D[u]["env"]
    env = dict(os.environ, **env_u, V6_OUT=out, V6_RULE=rule, V6_SUB_REQ="/v6w/subclavian/req/%s.json" % u)
    t0 = time.time()
    with open(os.path.join(out, "seg.log"), "w") as lf:
        rc = subprocess.call(["python3", script], env=env, stdout=lf, stderr=subprocess.STDOUT)
    print("%s rule %s rc %d %.0f s" % (u, rule, rc, time.time() - t0), flush=True)
    if rc == 0:
        open(os.path.join(out, ".seg_done"), "w").write("%s %s\n" % (rule, time.strftime("%F %T")))
