#!/usr/bin/env python3
# [carotid_tools/v5 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\bake_full\keep_v4_va.py (= /sp/v5_va/bake_full/keep_v4_va.py in the build containers)
"""v5_bake_full: the LAST rung of the bake failure rule -- keep v4's VA on one side of one anatomy (flagged), when
the donor VA cannot bake even as a tube (a centreline-level fusion: the tube-only union has a handle).

Re-assembles the side exactly as graft_v5_assign writes a side whose donor fails (the 15 one-sided anatomies):
  * Centrelines_comb/"Centerline curve - RVA|LVA.mrk.json"  <- v4's file (byte copy)
  * provenance xform["va_R"|"va_L"]                         <- v4's entry; xform["vaC_R"|"vaC_L"] removed
  * provenance v5_va[side] = {"replaced": false, "why": ..., "pass": false, "v4_va_kept": true,
                              "donor_rejected_at_bake": <the donor record that was there>}
  * v5_va.summary (v4_va_kept / replaced / donor), v5_va.assignment (served, flags) updated; licences and
    LICENSE_v5_donors.txt kept (the other side still uses the donor unit; if no side uses it any more the licence
    entry is kept and noted, never deleted).
The originals are saved first to <BF>/_reassembled/<name>/ (outside anatomies_v5). Any bake outputs of the anatomy
(mesh_v5.json, collision_full.vtp, the OBJ) and its bake markers are removed so it is re-baked from scratch.
    python3 keep_v4_va.py <name> <right|left> "<why>"       (neve-build-meshlab)
"""
import json
import os
import shutil
import sys
import time

A5, A4 = "/repo/carotid_data/anatomies_v5", "/repo/carotid_data/anatomies_v4"
BF = "/repo/carotid_data/va_sources/v5_reg/bake_full"


def main():
    name, side, why = sys.argv[1], sys.argv[2], sys.argv[3]
    tag, tk, ck = {"right": ("RVA", "va_R", "vaC_R"), "left": ("LVA", "va_L", "vaC_L")}[side]
    r5, r4 = os.path.join(A5, name), os.path.join(A4, name)
    cl = "Centerline curve - %s.mrk.json" % tag
    keep = os.path.join(BF, "_reassembled", name)
    os.makedirs(keep, exist_ok=True)
    for f in ("provenance.json", os.path.join("Centrelines_comb", cl)):
        dst = os.path.join(keep, os.path.basename(f) + ".before")
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(r5, f), dst)
    prov = json.load(open(os.path.join(r5, "provenance.json"), encoding="utf-8"))
    p4 = json.load(open(os.path.join(r4, "provenance.json"), encoding="utf-8"))
    v = prov["v5_va"]
    old = v[side]
    if old.get("v4_va_kept"):
        print("already v4 on", side)
        return 0
    shutil.copyfile(os.path.join(r4, "Centrelines_comb", cl), os.path.join(r5, "Centrelines_comb", cl))
    prov["xform"][tk] = p4["xform"][tk]
    prov["xform"].pop(ck, None)
    v[side] = {"replaced": False, "why": why, "pass": False, "v4_va_kept": True,
               "kept_by": "v5_bake_full keep_v4_va.py %s (bake failure rule, last rung)" % time.strftime("%Y-%m-%d %H:%M"),
               "donor_rejected_at_bake": old}
    s = v.get("summary") or {}
    s.setdefault("v4_va_kept", {})[side] = True
    s.setdefault("replaced", {})[side] = False
    s.setdefault("donor", {})
    s["donor_rejected_at_bake"] = {side: s["donor"].get(side)}
    s["donor"][side] = None
    a = v.get("assignment") or {}
    a.setdefault("served", {})[side] = False
    a.setdefault("flags", []).append("%s: v4 VA kept -- %s" % (side, why))
    used = [x for x in ("right", "left") if not v[x].get("v4_va_kept")]
    if not used:
        v["licences_note"] = "no side uses the donor unit any more (both v4 VAs kept); licence entry kept for the record"
    tmp = os.path.join(r5, "provenance.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    os.replace(tmp, os.path.join(r5, "provenance.json"))
    for f in ("mesh_v5.json", "collision_full.vtp", "vessel_architecture_collision.obj"):
        p = os.path.join(r5, f)
        if os.path.exists(p):
            os.remove(p)
    for suf in (".bake.json", ".bake_failed.json", ".measure.json"):
        p = os.path.join(BF, "_done", name + suf)
        if os.path.exists(p):
            os.replace(p, os.path.join(keep, os.path.basename(p) + ".before"))
    print("%s: %s now v4's VA (centreline + xform %s); %s removed; originals in %s" % (name, side, tk, ck, keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
