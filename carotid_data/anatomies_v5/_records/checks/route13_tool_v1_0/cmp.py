#!/usr/bin/env python3
"""Compare the 1.0 re-run records (r10check/out) with the records the route13 ext run wrote (route13/graft/ext).

Exact JSON equality first; otherwise list the differing paths (floats compared exactly, then within 1e-9)."""
import json
import os
import sys

NEW = "/sp/v5_va/r10check/out"
OLD = "/sp/v5_va/route13/graft/ext"


def diff(a, b, path="", out=None, tol=0.0):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in a else 'old'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > tol:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]!r} vs {str(b)[:80]!r}")
    return out


rows, bad = [], 0
for f in sorted(os.listdir(OLD)):
    if not f.endswith(".json"):
        continue
    pn = os.path.join(NEW, f)
    if not os.path.exists(pn):
        rows.append(f"{f[:-5]}\tMISSING")
        bad += 1
        continue
    a, b = json.load(open(pn)), json.load(open(os.path.join(OLD, f)))
    if a == b:
        rows.append(f"{f[:-5]}\tIDENTICAL")
        continue
    d0 = diff(a, b, tol=1e-9)
    if not d0:
        rows.append(f"{f[:-5]}\tEQUAL_WITHIN_1e-9")
        continue
    bad += 1
    rows.append(f"{f[:-5]}\tDIFFERS ({len(d0)} paths)\n    " + "\n    ".join(d0[:12]))
print("\n".join(rows))
print(f"\n{len(rows) - bad} of {len(rows)} reproduce")
sys.exit(1 if bad else 0)
