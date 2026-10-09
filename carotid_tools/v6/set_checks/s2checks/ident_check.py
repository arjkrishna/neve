#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\s2checks\ident_check.py (= /sp/v6/s2checks/ident_check.py in the build containers)
"""ident_check.py 1.0 (s2_checks, 2026-10-07): is carotid_data/anatomies_v6 the tree the per-anatomy checks measured?

The build's checks (checks_v6 1.3/1.4/1.5 records in /full/checks4) ran on /full/tree3 + the undecimated unions /full/iso.
For every anatomy: every file of tree3/<n> must exist in anatomies_v6/<n> with the same sha256, except provenance.json,
which must equal tree3's plus exactly ONE new key (v6_checks); anatomies_v6 may add only the licence files
(LICENSE_v6_donors.txt; LICENSE_VMR_0248.txt is inherited); the union /full/iso/<n>.vtp exists; the checks4 record is
complete (done marker) and passes. Then the per-anatomy check results apply to anatomies_v6 file for file.
    python3 ident_check.py <out.json>
"""
import hashlib
import json
import os
import sys

A6, T3, ISO, CK = "/repo/carotid_data/anatomies_v6", "/full/tree3", "/full/iso", "/full/checks4"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files(root):
    out = set()
    for d, _, fs in os.walk(root):
        for f in fs:
            out.add(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return out


def main():
    names6 = sorted(n for n in os.listdir(A6) if n.startswith("case_") and os.path.isdir(os.path.join(A6, n)))
    names3 = sorted(n for n in os.listdir(T3) if n.startswith("case_") and os.path.isdir(os.path.join(T3, n)))
    res = {"version": "ident_check 1.0 (s2_checks)", "n_v6": len(names6), "n_tree3": len(names3),
           "names_equal": names6 == names3, "per": {}, "bad": []}
    nfiles = 0
    for n in names6:
        f6, f3 = files(os.path.join(A6, n)), files(os.path.join(T3, n))
        extra = sorted(f6 - f3)
        missing = sorted(f3 - f6)
        diff = []
        for f in sorted(f3 & f6):
            if f == "provenance.json":
                continue
            nfiles += 1
            if sha(os.path.join(A6, n, f)) != sha(os.path.join(T3, n, f)):
                diff.append(f)
        p6 = json.load(open(os.path.join(A6, n, "provenance.json"), encoding="utf-8"))
        p3 = json.load(open(os.path.join(T3, n, "provenance.json"), encoding="utf-8"))
        new_keys = sorted(set(p6) - set(p3))
        gone = sorted(set(p3) - set(p6))
        changed = sorted(k for k in set(p3) & set(p6) if p3[k] != p6[k])
        prov_ok = new_keys == ["v6_checks"] and not gone and not changed
        extra_ok = all(e.startswith("LICENSE_") for e in extra)
        lic_removed_ok = all(m.startswith("LICENSE_v5") for m in missing)
        iso_ok = os.path.isfile(os.path.join(ISO, n + ".vtp"))
        done = os.path.isfile(os.path.join(CK, "_done", n + ".json"))
        rec = json.load(open(os.path.join(CK, n + ".json")))
        ck_pass = bool(rec.get("pass"))
        ok = not diff and prov_ok and extra_ok and lic_removed_ok and iso_ok and done and ck_pass
        res["per"][n] = {"ok": ok, "diff": diff, "extra": extra, "missing": missing, "prov_new_keys": new_keys,
                         "prov_changed": changed, "prov_gone": gone, "iso": iso_ok, "checks4_done": done, "checks4_pass": ck_pass,
                         "v6_checks_pass": (p6.get("v6_checks") or {}).get("pass")}
        if not ok:
            res["bad"].append(n)
    res["files_compared"] = nfiles
    res["ok"] = res["names_equal"] and not res["bad"] and len(names6) == 223
    json.dump(res, open(sys.argv[1], "w"), indent=0)
    print("IDENT %s" % json.dumps({k: res[k] for k in ("n_v6", "n_tree3", "names_equal", "files_compared", "bad", "ok")}))


if __name__ == "__main__":
    main()
