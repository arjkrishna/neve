#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\seg_batch.py (= /sp/v6/wall/seg_batch.py in the build containers)
"""v6_wall: run ctseg6.py (or vmrseg6.py for VMR 0248) for a list of donor units under one rule (inside ONE
neve-build-stagea container), each in its own subprocess with the env of wall/donors.json; VMTK runs separately in
neve-build-vmtk (vmtk_batch.sh). Idempotent: <out>/<unit>/.seg_done marks a finished unit; logs <out>/<unit>/seg.log.
    python3 seg_batch.py <rule> <variant_dir> <unit,unit,...|all|all+vmr> [extra KEY=VAL ...]
"""
import json, os, subprocess, sys, time
rule, vdir, units = sys.argv[1:4]
extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
D = json.load(open("/sp/v6/wall/donors.json"))
if units in ("all", "all+vmr"):
    todo = [u for u in D if u != "vmr0248"] + (["vmr0248"] if units == "all+vmr" else [])
else:
    todo = units.split(",")
for u in todo:
    cfg = D[u]
    out = os.path.join(vdir, u)
    if os.path.exists(os.path.join(out, ".seg_done")):
        print("skip", u, flush=True)
        continue
    os.makedirs(out, exist_ok=True)
    if u == "vmr0248":
        script, env_u = "/sp/v6/wall/vmrseg6.py", {"VMR_OPEN_BRIGHT": "1"}
    else:
        script, env_u = "/sp/v6/wall/ctseg6.py", cfg["env"]
    env = dict(os.environ, **env_u, V6_OUT=out, V6_RULE=rule, **extra)
    t0 = time.time()
    with open(os.path.join(out, "seg.log"), "w") as lf:
        rc = subprocess.call(["python3", script], env=env, stdout=lf, stderr=subprocess.STDOUT)
    print("%s rule %s rc %d %.0f s" % (u, rule, rc, time.time() - t0), flush=True)
    if rc == 0:
        open(os.path.join(out, ".seg_done"), "w").write("%s %s\n" % (rule, time.strftime("%F %T")))
