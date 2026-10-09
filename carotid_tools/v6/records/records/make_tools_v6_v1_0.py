#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\records\make_tools_v6_v1_0.py (= /sp/v6/records/make_tools_v6_v1_0.py in the build containers)
"""s2_records: stage carotid_tools/v6 -- the toolchain the v6 build ran, copied from the session scratchpad (the v5 pattern:
one header line per code file, FILES.tsv with sha256, restore_layout.sh, MANIFEST.md written separately).

Selection:
  build     every /sp file of the build's toolchain manifests (build/TOOLCHAIN_s2i2.md5 bedf172b, TOOLCHAIN_s2build_*.md5
            a80644e0 / 80b29bc3 / 11875b8d / 88ab481d, the final TOOLCHAIN_s2build.md5 d65b7209, full/TOOLCHAIN_s2build_supplement.md5);
            an entry whose current file no longer has the manifest md5 is matched to the frozen copy that has it
  v5_deps   the /sp/v5_va and /sp/v4_build files those manifests list
  donor     the stage-1 / stage-2 tools that made the donor-side inputs the build reads (D15 local walls, item-e junction
            walls, D16 / D17 records and paths, D10 native offsets, C4 spec)
  gates     heading evidence (per-patient Zenodo frames) and the gate API test
  item_b    the pilot route13 tools whose diagnosis the build's route13 stage reads (w_018b_left mr_002)
  set_checks the s2_checks scripts (split, loader, SOFA, orientation, identity, coverage diagnosis)
  records   this stage's scripts (BUILD_v6.json generator and checker, _records copy, this script)
then the closure: every scratchpad module a selected Python file imports (its own folder or a /sp folder it puts on
sys.path) and every scratchpad .py / .sh a selected shell script names.

    python3 make_tools_v6.py --out /sp/v6/records/tools_v6     (scratchpad mounted at /sp, repo at /repo)
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import re
import shutil

VERSION = "make_tools_v6 1.0 (s2_records, 2026-10-07)"
ap = argparse.ArgumentParser()
ap.add_argument("--sp", default="/sp")
ap.add_argument("--out", default="/sp/v6/records/tools_v6")
a = ap.parse_args()
SP = a.sp
HOSTSP = r"C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad"
HEADER = "# [carotid_tools/v6 copy] source: %s\\%s (= /sp/%s in the build containers)"
MANIFESTS = ["v6/build/TOOLCHAIN_s2i2.md5", "v6/build/TOOLCHAIN_s2build_a80644e0.md5", "v6/build/TOOLCHAIN_s2build_80b29bc3.md5",
             "v6/build/TOOLCHAIN_s2build_11875b8d.md5", "v6/build/TOOLCHAIN_s2build_88ab481d.md5", "v6/build/TOOLCHAIN_s2build.md5",
             "v6/full/TOOLCHAIN_s2build_supplement.md5"]
MAN_TAG = {m: os.path.basename(m).replace(".md5", "") for m in MANIFESTS}


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sha256b(b):
    return hashlib.sha256(b).hexdigest()


sel = collections.OrderedDict()      # rel path (under SP) -> stage
why = {}
man_rows = []                        # manifest, container path, md5, resolved rel path / repo path, status


def add(rel, stage, reason):
    rel = rel.replace("\\", "/")
    if not os.path.isfile(os.path.join(SP, rel)):
        raise SystemExit("missing: %s (%s)" % (rel, reason))
    if rel not in sel:
        sel[rel] = stage
        why[rel] = reason


def stage_of(rel):
    if rel.startswith(("v5_va/", "v4_build/")):
        return "v5_deps"
    return "build"


# ---- 1. manifests
for m in MANIFESTS:
    add(m, "build", "toolchain manifest")
    for line in open(os.path.join(SP, m)):
        mm = re.match(r"^([0-9a-f]{32}) \*(\S+)$", line.strip())
        if not mm:
            continue
        md5, cpath = mm.groups()
        if cpath.startswith("/sp/"):
            rel = cpath[4:]
            cur = os.path.join(SP, rel)
            if os.path.isfile(cur) and md5f(cur) == md5:
                add(rel, stage_of(rel), "listed in %s" % MAN_TAG[m])
                man_rows.append([MAN_TAG[m], cpath, md5, rel, "current file"])
                continue
            d = os.path.dirname(cur)
            hit = [f for f in sorted(os.listdir(d), key=lambda f: ("dev" in f, f))      # prefer the frozen _v* name to a _dev copy
                   if os.path.isfile(os.path.join(d, f)) and md5f(os.path.join(d, f)) == md5]
            if hit:
                r2 = os.path.join(os.path.dirname(rel), hit[0])
                add(r2, stage_of(r2), "frozen copy of %s as listed in %s" % (rel, MAN_TAG[m]))
                man_rows.append([MAN_TAG[m], cpath, md5, r2, "frozen copy (the file was later versioned)"])
            else:
                man_rows.append([MAN_TAG[m], cpath, md5, None, "NOT FOUND"])
        else:
            repo = cpath.replace("/opt/eve_training/topbrain_tools/", "/repo/topbrain_tools/")
            ok = os.path.isfile(repo) and md5f(repo) == md5
            man_rows.append([MAN_TAG[m], cpath, md5, repo.replace("/repo/", "repo:"), "repo module, md5 matches" if ok else "repo module, MD5 DIFFERS"])

# ---- 2. explicit upstream / set / records selections
DONOR = ["v6/wall/" + f for f in ("ctseg6.py", "make_vmrseg6.py", "vmrseg6.py", "make_donors.py", "donors.json", "seg_batch.py",
                                  "vmtk_batch.sh", "vmtk_fix.py", "vmtk_fix2.py", "build_donors6.py", "c4_check.py", "sub_batch.py", "vmtk_sub.sh",
                                  "gate_ct.py", "wd_proxy.py", "wd_batch.py", "wd_collect.py", "surf_stats.py", "floor_scan6.py",
                                  "bone_scan6.py", "gate_summary.py", "chain2.sh", "chain3.sh", "post_light.sh", "dr.sh")]
DONOR += ["v6/wall/jfix/" + f for f in ("junction6.py", "sub_batch2.py", "vmtk_sub3.sh", "publish_j3.sh", "sub_genus.py", "wd_sub_collect.py")]
DONOR += ["v6/wall_local/" + f for f in ("local_seg.py", "vmtk_local_fix.py", "splice_lines.py", "build_donors_local.py", "local_sub.py",
                                         "local_sub_v1_0.py", "junction_records.py", "wd_local.py", "wd_sub.py", "wd_sub_local.py",
                                         "d15_report.py", "chainA.sh", "chainB.sh")]
DONOR += ["v6/wall_d17/" + f for f in ("paths_d17.py", "make_req_d17.py", "sub_batch_d17.py", "junction_records_d17.py", "wd_sub_d17.py",
                                       "d17_report.py", "publish_d17.py", "chainD17.sh", "sub_ct_gate.py", "genus_cmp.py")]
DONOR += ["v6/va/" + f for f in ("subclavian_paths.py", "sc_post3.py", "paths_d17.py")]
GATES = ["v6/heading/" + f for f in ("heading_evidence.py", "zen_frames.py", "host_tree.py", "c6check.py", "zenodo_frames.json")]
GATES += ["v6/gates/test_api.py"]
ITEMB = ["v6/pilot_build/route_diag_v6.py"]     # writes pilot/route13/diag/<name>.rcca_v4.npz, read by build_v6's route13 stage
SETC = ["v6/s2checks/" + f for f in ("collect_s2c.py", "orient_set.py", "loader_set.py", "ident_check.py", "diag_cov.py", "bif_angles.py",
                                     "make_items.py", "sofa_worker.sh", "sofa_container.sh", "dr_s2c.sh")]
RECS = ["v6/records/" + f for f in ("make_tools_v6.py", "build_v6_record.py", "check_build_record.py", "copy_records.sh", "dr_rec.sh",
                                    "finalize_tools.sh", "imports_check.py", "restore_layout.sh", "MANIFEST_head.md")]
BUILDX = ["v6/full/load_v6_all.sh", "v6/full/load_v6_summary.py",          # the load of the final folder (not in a manifest)
          "v6/build/stage2_dry2_compose.py", "v6/build/repilot2_summary.py"]  # the DRY / re-pilot summaries in _records/results
for lst, st in ((BUILDX, "build"), (DONOR, "donor"), (GATES, "gates"), (ITEMB, "item_b"), (SETC, "set_checks"), (RECS, "records")):
    for r in lst:
        add(r, st, "selected (%s)" % st)

# ---- 3. closure over scratchpad imports / script references
SPDIRS = re.compile(r"""["'](/sp/[A-Za-z0-9_./-]+)["']""")
SHREF = re.compile(r"""/sp/((?:v6|v5_va|v4_build)/[A-Za-z0-9_./-]+\.(?:py|sh))""")
added = []


def py_imports(src):
    try:
        t = ast.parse(src)
    except SyntaxError:
        return set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", src, re.M))
    out = set()
    for node in ast.walk(t):
        if isinstance(node, ast.Import):
            out.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("import_module",) and node.args and isinstance(node.args[0], ast.Constant):
            out.add(str(node.args[0].value).split(".")[0])
    return out


changed = True
while changed:
    changed = False
    for rel, st in list(sel.items()):
        p = os.path.join(SP, rel)
        src = open(p, encoding="utf-8", errors="replace").read()
        if rel.endswith(".py"):
            dirs = {os.path.dirname(p)}
            for s in SPDIRS.findall(src):
                q = os.path.join(SP, s[4:])
                if os.path.isdir(q):
                    dirs.add(q)
                elif os.path.isdir(os.path.dirname(q)):
                    dirs.add(os.path.dirname(q))
            for mod in py_imports(src):
                for d in sorted(dirs):
                    f = os.path.join(d, mod + ".py")
                    if os.path.isfile(f):
                        r2 = os.path.relpath(f, SP).replace("\\", "/")
                        if r2 not in sel:
                            add(r2, sel[rel] if stage_of(r2) == "build" else stage_of(r2), "imported by %s" % rel)
                            added.append([r2, rel])
                            changed = True
                        break
        elif rel.endswith(".sh"):
            for r2 in SHREF.findall(src):
                if os.path.isfile(os.path.join(SP, r2)) and r2 not in sel:
                    add(r2, sel[rel] if stage_of(r2) == "build" else stage_of(r2), "named by %s" % rel)
                    added.append([r2, rel])
                    changed = True

# ---- 4. write the staged copy
if os.path.isdir(a.out):
    shutil.rmtree(a.out)
rows = []
for rel, st in sorted(sel.items(), key=lambda kv: (kv[1], kv[0])):
    raw = open(os.path.join(SP, rel), "rb").read()
    sub = rel[3:] if rel.startswith("v6/") else rel          # v6/carotid/x.py -> carotid/x.py ; v5_va/... kept
    cp = "%s/%s" % (st, sub)
    code = rel.endswith((".py", ".sh")) or os.path.basename(rel) == "Dockerfile"
    out = raw
    if code:
        nl = b"\r\n" if b"\r\n" in raw[:2000] else b"\n"
        hdr = (HEADER % (HOSTSP, rel.replace("/", "\\"), rel)).encode() + nl
        lines = raw.split(nl)
        k = 0
        if lines and lines[0].startswith(b"#!"):
            k = 1
        if len(lines) > k and re.match(rb"^#.*coding[:=]", lines[k] or b""):
            k += 1
        out = nl.join(lines[:k]) + (nl if k else b"") + hdr + nl.join(lines[k:])
        assert out.replace(hdr, b"", 1) == raw, rel
    dst = os.path.join(a.out, cp)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(out)
    rows.append([st, rel, cp, "yes" if code else "no", sha256b(raw), sha256b(out), len(raw), hashlib.md5(raw).hexdigest(), why[rel]])
with open(os.path.join(a.out, "FILES.tsv"), "w") as f:
    f.write("stage\toriginal_path\tcopy_path\theader_added\tsha256_original\tsha256_copy\tbytes_original\tmd5_original\twhy\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(a.out, "TOOLCHAINS.tsv"), "w") as f:
    f.write("manifest\tentry\tmd5\tresolved\tstatus\n")
    for r in man_rows:
        f.write("\t".join(str(x) for x in r) + "\n")
summ = {"version": VERSION, "files": len(rows), "by_stage": dict(collections.Counter(r[0] for r in rows)),
        "closure_added": added, "manifest_rows": len(man_rows),
        "manifest_status": dict(collections.Counter(r[4] for r in man_rows)),
        "not_found": [r for r in man_rows if r[4] == "NOT FOUND"], "repo_md5_differs": [r for r in man_rows if "DIFFERS" in r[4]]}
with open(os.path.join(os.path.dirname(a.out), "make_tools_v6.json"), "w") as f:
    json.dump(summ, f, indent=1)
print(json.dumps(summ, indent=1)[:6000])
