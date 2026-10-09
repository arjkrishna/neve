#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\wall\wd_batch.py (= /sp/v6/wall/wd_batch.py in the build containers)
"""v6_wall: run wd_proxy.py over every donor side for the given jobs, 3 at a time (one neve-build-meshlab container).
A job is "<variant>[:floor[:tag]]", e.g. v5, plateau, half, v5:ct:v5floorCT. Existing outputs are skipped (idempotent).
    python3 wd_batch.py <job,job,...> [unit,...]   -> logs /v6w/wd/batch.log"""
import os, subprocess, sys
from multiprocessing import Pool
R5 = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
jobs = sys.argv[1].split(",")
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
todo = []
for j in jobs:
    parts = j.split(":")
    var, floor, tag = parts[0], (parts[1] if len(parts) > 1 else "1.0"), (parts[2] if len(parts) > 2 else parts[0])
    for unit in sorted(os.listdir(R5)):
        if only and unit not in only:
            continue
        for sd in ("R", "L"):
            if not os.path.exists(os.path.join(R5, unit, "%s.json" % sd)):
                continue
            if var == "subclavian":
                if not os.path.exists("/v6w/subclavian/%s/%s_stageA_vmtk.json" % (unit, "rsub" if sd == "R" else "lsub")):
                    continue
            elif var != "v5" and not os.path.exists("/v6w/seg/%s/%s/%s_stageA_vmtk.json" % (var, unit, "rva" if sd == "R" else "lva")):
                continue
            if floor == "ct" and unit == "vmr0248":
                continue
            if os.path.exists("/v6w/wd/%s/%s_%s.json" % (tag, unit, sd)):
                continue
            todo.append((var, unit, sd, floor, tag))


def run(a):
    var, unit, sd, floor, tag = a
    r = subprocess.run(["python3", "/sp/v6/wall/wd_proxy.py", var, unit, sd, "--floor", floor, "--tag", tag],
                       capture_output=True, text=True)
    return "%s %s %s rc %d %s" % (tag, unit, sd, r.returncode, (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0
                                   else r.stderr.strip()[-300:])


os.makedirs("/v6w/wd", exist_ok=True)
print(len(todo), "jobs", flush=True)
with Pool(3) as p, open("/v6w/wd/batch.log", "a") as lf:
    for line in p.imap_unordered(run, todo):
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
