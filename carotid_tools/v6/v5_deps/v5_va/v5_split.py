#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v5_va\v5_split.py (= /sp/v5_va/v5_split.py in the build containers)
"""v5_split.py -- donor-overlap train/test split checker for the carotid anatomy sets (v3, v4, v5).

    python3 v5_split.py <anatomies_dir> [--out result.json] [--baseline split_v4.json]
                        [--defs v3,v4side,strict] [--test-sizes 16,17] [--no-div]
                        [--extra-donor-field v5_va.donor_id ...] [--extra-donor-regex REGEX]
                        [--no-auto-va] [--donor-alias aliases.json] [--time-limit 300]

Every anatomy is a composite of donors. Two anatomies overlap when they share a donor. A split
is overlap-free when no donor appears in both a train and a test anatomy; anatomies that would
break this are dropped ("excluded"). The script builds the anatomy-donor graph from each
<anatomy>/provenance.json and reports, for each donor definition:

  roster_v3   The carotid v3 launcher split (4eef6bb, GUIDE_stitched_datasets.md 7.2): holdout =
              the block {route lower in 6 holdout lowers} x {TopBrain patient in 6 holdout patients}
              (16 anatomies); every anatomy sharing ANY donor with the block is excluded; the rest is
              train. On v3 / v4 with the v3 donor definition this gives 153 train / 16 test / 54
              excluded -- the user's "~170 (153 / 17)".  The block is fixed by name; new donors
              (e.g. v5 VA donors) can only remove train anatomies, never add test ones.
  max_k       The true maximum: max (train + test) over all donor partitions with exactly k test
              anatomies (MILP, HiGHS via scipy.optimize.milp). Optimal unless status says otherwise.
  max_k_div   The same with the roster's breadth: the test set must contain >= 6 TopBrain patients
              and >= 6 Zenodo donors (each with >= 1 test anatomy).

Donor definitions (--defs):
  v3      route lower (Zenodo case-side, e.g. case_w_007_left) + TopBrain patient   [v3 launcher's]
          (+ extra/V: donors, so a v5 set can be compared with the 153 / 16 baseline)
  v4side  + the left-fork lower (case-side) + every TopBrain/extra donor found in the provenance
  strict  v4side with Zenodo donors at PATIENT level (case_w_007_left == case_w_007_right) and
          case_w_018b merged into case_w_018 (GUIDE: "probably a second scan")      [default guard]

TopBrain donors are taken from every topcow_mr_NNN id in the anatomy's xform sources (siphons,
VA pieces, ECA tips) plus "patient"; mirroring (_L) is the same patient. Extra donors (v5 VAs):
  * --extra-donor-field PATH (dotted, '*' wildcard; str or list values), any number of times;
  * auto (default on): xform sections whose kind is neither zenodo nor topbrain: their donor_id
    (graft_v5_reg.py's vaC_R / vaC_L), else the id in their surface / centerline path
    (va_sources/v5_reg/donors/<id>/, or va_sources/derived/: vmr0248, thin_slice/<src>/<case> ->
    <case> or <src>_<case>, arterial/ts_<s> -> ts_<s>, tagnet/...; as graft_v5_reg_donor.unit_id);
    else the prototype's free-text v5_va.donor;
  * --extra-donor-regex: first capture group normalises a value to a donor PATIENT id;
  * --donor-alias JSON {"id": "canonical id"} merges ids known to be one patient (e.g. a series
    present in two open datasets).
Extra donors are prefixed "V:". One-to-one VA donors (each V: donor used by one TopBrain patient)
are nested in that patient's node and provably leave every quantity unchanged; the report lists
any V: donor that spans more than one TopBrain patient. A TaG-Net donor is an error (no licence).

--baseline: compare against a previous result (e.g. results/split_v4.json); exit code 3 if any
quantity (train, test or kept) is lower than the baseline's for the same definition.
v5_assign extension (2026-10-02): the V: donor nesting is printed per definition, and exit code 4 if any V: donor
spans more than one TopBrain patient (one_to_one_nested false) even when no quantity drops (exit 2 / 3 take precedence).
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------------------------
# The carotid v3 launcher roster (4eef6bb; GUIDE_stitched_datasets.md section 7.2, "checked
# against env_train.yml"): 16 holdout = the full 6x6 block, 54 cross-terms excluded, 153 train.
V3_HOLDOUT_LOWERS = ["case_m_030_right", "case_w_007_left", "case_w_013_right",
                     "case_w_017_right", "case_w_025_right", "case_w_033_left"]
V3_HOLDOUT_PATIENTS = ["topcow_mr_003", "topcow_mr_011", "topcow_mr_013",
                       "topcow_mr_016", "topcow_mr_017", "topcow_mr_021"]

ZEN_RE = re.compile(r"(case_[kmw]_\d{3}b?)_(left|right)")
TB_RE = re.compile(r"(topcow_mr_\d{3})")
DEFS_ALL = ("v3", "v4side", "strict")


# ---------------------------------------------------------------------------------------------
def load_set(adir):
    prov = collections.OrderedDict()
    for p in sorted(glob.glob(os.path.join(adir, "*", "provenance.json"))):
        prov[os.path.basename(os.path.dirname(p))] = json.load(open(p))
    build = None
    for b in sorted(glob.glob(os.path.join(adir, "BUILD*.json"))):
        build = (os.path.basename(b), json.load(open(b)))
    return prov, build


def _walk(obj, parts):
    """dotted path with '*' wildcard -> list of leaf values"""
    if not parts:
        return [obj]
    head, rest = parts[0], parts[1:]
    out = []
    if head == "*":
        it = obj.values() if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
        for v in it:
            out += _walk(v, rest)
    elif isinstance(obj, dict) and head in obj:
        out += _walk(obj[head], rest)
    return out


def _flat_str(vals):
    out = []
    for v in vals:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out += [x for x in v if isinstance(x, str)]
    return out


def donor_id_from_path(path):
    """va_sources/derived/... path -> donor patient id, or None"""
    s = str(path).replace("\\", "/")
    m = re.search(r"va_sources/v5_reg/donors/([^/]+)/", s)
    if m:
        return m.group(1)                                # registry unit id (graft_v5_reg_donor)
    m = re.search(r"va_sources/derived/(.+)$", s)
    if not m:
        m = re.search(r"va_sources/(tagnet|vmr0248|spl_atlas|thin_slice|totalseg_v3)/(.+)$", s)
        if not m:
            return None
        parts = [m.group(1)] + m.group(2).split("/")
    else:
        parts = m.group(1).split("/")
    head = parts[0]
    if head.startswith("vmr"):
        return head                                      # vmr0248
    # ids follow graft_v5_reg_donor.unit_id so a path-only provenance names a donor as the registry does
    if head == "thin_slice" and len(parts) >= 3:
        src, case = parts[1], parts[2]                     # thin_slice/<source>/<case>
        return case if case.startswith(src) else "%s_%s" % (src, case)
    if head == "arterial" and len(parts) >= 2:
        run = parts[1]
        if run == "runs" and len(parts) >= 3:
            run = parts[2]
        return run if run.startswith("ts_") else "arterial/%s" % run
    if len(parts) >= 2:
        return "%s/%s" % (head, parts[1])
    return head


def norm_donor_string(s):
    """free-text donor field (e.g. 'vmr0248 ct_seg4 stage-A (VMTK MISR)') -> id"""
    s = s.strip()
    p = donor_id_from_path(s)
    if p:
        return p
    m = re.match(r"(vmr\s*\d{4})", s, re.I)
    if m:
        return m.group(1).replace(" ", "").lower()
    return s.split()[0] if s else s


class DonorExtractor:
    def __init__(self, extra_fields=(), extra_regex=None, auto_va=True, alias=None):
        self.extra_fields = list(extra_fields or [])
        self.extra_regex = re.compile(extra_regex) if extra_regex else None
        self.auto_va = auto_va
        self.alias = alias or {}
        self.checks = collections.Counter()
        self.notes = []
        self.extra_sources = collections.Counter()
        self.patient_key = {}      # registry donor_id -> patient_key (de-duplication key), when readable

    def finalize(self, raws):
        """merge registry donor ids that share a patient_key (one patient in two datasets)"""
        groups = collections.defaultdict(set)
        for i, k in self.patient_key.items():
            if k:
                groups[k].add(i)
        merge = {}
        for k, ids in groups.items():
            if len(ids) > 1:
                c = sorted(ids)[0]
                for i in ids:
                    merge[i] = c
                self.notes.append("donor ids %s share patient_key %s -> merged" % (sorted(ids), k))
                self.checks["registry donor ids merged by patient_key"] += 1
        if merge:
            for r in raws:
                r["extra"] = {merge.get(e, e) for e in r["extra"]}
        return raws

    def _canon(self, x):
        return self.alias.get(x, x)

    def _norm_extra(self, v):
        if self.extra_regex:
            m = self.extra_regex.search(v)
            if m:
                v = m.group(1) if m.groups() else m.group(0)
        return self._canon(v)

    def raw(self, name, d):
        """-> dict(route_lower, left_lower, zen_paths, tb, extra)"""
        xf = d.get("xform") or {}
        left = d.get("left") or {}
        route_lower = d.get("lower")
        left_lower = left.get("lower")
        zen_from_xf = set()
        tb = set()
        extra = set()
        if d.get("patient"):
            tb.add(d["patient"])
        if d.get("siphon"):
            m = TB_RE.search(d["siphon"])
            if m:
                tb.add(m.group(1))
        for sec, v in xf.items():
            if not isinstance(v, dict):
                continue
            kind = str(v.get("kind", ""))
            paths = [str(v.get(f)) for f in ("surface", "centerline_src") if v.get(f)]
            if kind == "zenodo":
                for p in paths:
                    m = ZEN_RE.search(p)
                    if m:
                        zen_from_xf.add(m.group(0))
            elif kind == "topbrain":
                for p in paths:
                    m = TB_RE.search(p)
                    if m:
                        tb.add(m.group(1))
            elif self.auto_va:
                if isinstance(v.get("donor_id"), str):
                    # v5 registry convention (graft_v5_reg.py): xform vaC_* carries donor_id (+ donor_record)
                    i = v["donor_id"]
                    self.extra_sources["xform.%s.donor_id" % sec] += 1
                    rec = v.get("donor_record")
                    if rec and i not in self.patient_key:
                        for cand in (rec, rec.replace("/repo/carotid_data", "/opt/eve_training/carotid_data")):
                            try:
                                self.patient_key[i] = json.load(open(cand)).get("patient_key")
                                break
                            except Exception:
                                continue
                    extra.add(self._canon(i))
                    continue
                ids = set(filter(None, (donor_id_from_path(p) for p in paths)))
                if ids:
                    for i in ids:
                        extra.add(self._canon(i))
                        self.extra_sources["xform.%s(path)" % sec] += 1
                else:
                    # unknown kind with no recognisable path: report, never silently drop
                    self.checks["xform section with unknown donor kind: %s" % sec] += 1
        for sec in ("eca_extension",):
            for side, v in (d.get(sec) or {}).items():
                if isinstance(v, dict) and v.get("patient"):
                    m = TB_RE.search(str(v["patient"]))
                    if m:
                        tb.add(m.group(1))
        if self.auto_va and not extra and isinstance(d.get("v5_va"), dict):
            # prototype convention: free-text v5_va.donor (only when no xform section named a donor)
            for side in ("right", "left"):
                sv = d["v5_va"].get(side)
                if isinstance(sv, dict) and not sv.get("v4_va_kept"):
                    for k in ("donor_id", "donor_patient"):
                        if isinstance(sv.get(k), str):
                            extra.add(self._canon(norm_donor_string(sv[k])))
                            self.extra_sources["v5_va.%s.%s" % (side, k)] += 1
                            break
            if not extra:
                for k in ("donor_id", "donor_patient", "donor"):
                    if isinstance(d["v5_va"].get(k), str):
                        extra.add(self._canon(norm_donor_string(d["v5_va"][k])))
                        self.extra_sources["v5_va.%s" % k] += 1
                        break
        for f in self.extra_fields:
            vals = _flat_str(_walk(d, f.split(".")))
            for v in vals:
                extra.add(self._norm_extra(v))
                self.extra_sources["field %s" % f] += 1
        # consistency checks
        zen_named = {x for x in (route_lower, left_lower) if x}
        if zen_from_xf and zen_from_xf != zen_named:
            self.checks["zenodo xform paths != lower/left.lower"] += 1
            self.notes.append("%s: zenodo xform %s vs named %s" % (name, sorted(zen_from_xf), sorted(zen_named)))
        if d.get("patient") and tb != {d["patient"]}:
            self.checks["TopBrain ids in xform != patient"] += 1
            self.notes.append("%s: TopBrain ids %s vs patient %s" % (name, sorted(tb), d["patient"]))
        for e in extra:
            if "tagnet" in e.lower():
                self.checks["TaG-Net donor present (NEVER a shipped donor)"] += 1
                self.notes.append("%s: TaG-Net donor %s" % (name, e))
        return dict(route_lower=route_lower, left_lower=left_lower,
                    zen=zen_named | zen_from_xf, tb=tb, extra=extra)

    @staticmethod
    def zen_node(case_side, level):
        if level == "side":
            return "Z:" + case_side
        m = ZEN_RE.search(case_side)
        pid = m.group(1) if m else re.sub(r"_(left|right)$", "", case_side)
        pid = pid.replace("case_w_018b", "case_w_018")
        return "Z:" + pid

    def donors(self, r, defn):
        S = set()
        if defn == "v3":
            S.add(self.zen_node(r["route_lower"], "side"))
            S |= {"T:" + t for t in r["tb"]}
            S |= {"V:" + e for e in r["extra"]}     # v5 VA donors count in every definition
            return S
        level = "side" if defn == "v4side" else "patient"
        for z in r["zen"]:
            S.add(self.zen_node(z, level))
        S |= {"T:" + t for t in r["tb"]}
        S |= {"V:" + e for e in r["extra"]}
        return S


# ---------------------------------------------------------------------------------------------
def components(names, dsets):
    parent = {}

    def f(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for n, s in zip(names, dsets):
        a = "A:" + n
        f(a)
        for d in s:
            ra, rd = f(a), f(d)
            if ra != rd:
                parent[ra] = rd
    comp = collections.defaultdict(list)
    for n in names:
        comp[f("A:" + n)].append(n)
    sizes = sorted((len(v) for v in comp.values()), reverse=True)
    return sizes


def roster_split(names, dsets, raws, holdout_lowers, holdout_patients):
    hl, hp = set(holdout_lowers), set(holdout_patients)
    test = [i for i, r in enumerate(raws) if r["route_lower"] in hl and (r["tb"] & hp)]
    H = set().union(*[dsets[i] for i in test]) if test else set()
    train = [i for i in range(len(names)) if i not in set(test) and not (dsets[i] & H)]
    excl = len(names) - len(test) - len(train)
    # plain (v3-style) roster check: anatomies that the v3 rule would train on but now overlap
    v3_train = [i for i, r in enumerate(raws)
                if r["route_lower"] not in hl and not (r["tb"] & hp)]
    leaked = sorted(names[i] for i in v3_train if dsets[i] & H)
    return dict(test=len(test), train=len(train), kept=len(test) + len(train), excluded=excl,
                test_anatomies=[names[i] for i in test],
                test_donors=sorted(H),
                v3_rule_train=len(v3_train),
                v3_rule_train_overlapping_test=len(leaked),
                v3_rule_train_overlapping_test_names=leaked)


def max_split(dsets, k, min_groups=None, time_limit=300.0):
    """max (#test + #train) s.t. #test == k; donors partitioned test/train; anatomy test iff all its
    donors test, train iff all train, else dropped. min_groups: [(prefix, m)] -> at least m donors
    with that prefix carry >= 1 test anatomy. Tie-break: more train."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    allD = sorted(set().union(*dsets))
    di = {d: i for i, d in enumerate(allD)}
    nA, nD = len(dsets), len(allD)
    groups = []
    for pre, m in (min_groups or []):
        groups.append(([d for d in allD if d.startswith(pre)], m))
    nU = sum(len(g) for g, _ in groups)
    nv = nD + 2 * nA + nU
    T0, R0, U0 = nD, nD + nA, nD + 2 * nA
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, l, h):
        nonlocal r
        for c, v in entries:
            rows.append(r); cols.append(c); vals.append(v)
        lo.append(l); hi.append(h); r += 1

    d2a = collections.defaultdict(list)
    for a, s in enumerate(dsets):
        for d in s:
            add([(T0 + a, 1.0), (di[d], -1.0)], -np.inf, 0.0)      # t_a <= x_d
            add([(R0 + a, 1.0), (di[d], 1.0)], -np.inf, 1.0)       # r_a <= 1 - x_d
            d2a[d].append(a)
    add([(T0 + a, 1.0) for a in range(nA)], k, k)                   # sum t = k
    u = U0
    for g, m in groups:
        us = []
        for d in g:                                                  # u_d <= sum_{a ni d} t_a
            add([(u, 1.0)] + [(T0 + a, -1.0) for a in d2a[d]], -np.inf, 0.0)
            us.append(u); u += 1
        add([(x, 1.0) for x in us], m, np.inf)
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    c = np.zeros(nv)
    c[T0:R0] = -1.0
    c[R0:U0] = -1.0 - 1e-3
    t0 = time.time()
    sol = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": float(time_limit)})
    sec = round(time.time() - t0, 1)
    if sol.x is None:
        return dict(k=k, status=int(sol.status), message=str(sol.message), sec=sec)
    x = np.round(sol.x)
    test = [a for a in range(nA) if x[T0 + a] > 0.5]
    train = [a for a in range(nA) if x[R0 + a] > 0.5]
    test_side = [allD[i] for i in range(nD) if x[i] > 0.5]
    used_test = set().union(*[dsets[a] for a in test]) if test else set()
    out = dict(k=k, test=len(test), train=len(train), kept=len(test) + len(train),
               dropped=nA - len(test) - len(train), status=int(sol.status),
               optimal=bool(sol.status == 0), message=str(sol.message), sec=sec,
               mip_gap=getattr(sol, "mip_gap", None),
               test_idx=test, train_idx=train,
               test_donors=sorted(used_test),
               n_test_tb=sum(1 for d in used_test if d.startswith("T:")),
               n_test_z=sum(1 for d in used_test if d.startswith("Z:")))
    # independent check of the returned split
    tr_d = set().union(*[dsets[a] for a in train]) if train else set()
    out["verified_disjoint"] = not (tr_d & used_test)
    return out


# ---------------------------------------------------------------------------------------------
def analyse(prov, defs=DEFS_ALL, test_sizes=(16, 17), div=True, extractor=None,
            time_limit=300.0, verbose=True, solve=True):
    ex = extractor or DonorExtractor()
    names = list(prov)
    raws = ex.finalize([ex.raw(n, prov[n]) for n in names])
    res = collections.OrderedDict()
    res["n_anatomies"] = len(names)
    res["checks"] = dict(ex.checks)
    res["check_notes"] = ex.notes[:50]
    res["extra_donor_sources"] = dict(ex.extra_sources)
    tb_of = [sorted(r["tb"]) for r in raws]
    res["topbrain_patients"] = len(set(t for r in raws for t in r["tb"]))
    res["handedness"] = dict(collections.Counter(str(prov[n].get("handedness")) for n in names))
    res["left_lower_fallbacks"] = sum(1 for n in names if (prov[n].get("left") or {}).get("lower_fallback"))
    res["defs"] = collections.OrderedDict()
    for defn in defs:
        dsets = [ex.donors(r, defn) for r in raws]
        allD = set().union(*dsets)
        kinds = collections.Counter(d[:2] for d in allD)
        deg = collections.Counter(d for s in dsets for d in s)
        R = collections.OrderedDict()
        R["donors"] = {k.rstrip(":"): v for k, v in sorted(kinds.items())}
        R["donors_per_anatomy"] = dict(collections.Counter(len(s) for s in dsets))
        R["anatomies_per_donor"] = {k.rstrip(":"): dict(min=min(deg[d] for d in allD if d.startswith(k)),
                                                       median=float(np.median([deg[d] for d in allD if d.startswith(k)])),
                                                       max=max(deg[d] for d in allD if d.startswith(k)))
                                    for k in sorted(kinds)}
        comp = components(names, dsets)
        R["components"] = dict(n=len(comp), sizes=comp[:10])
        # extra (V:) donors: nesting in TopBrain patients
        vd = sorted(d for d in allD if d.startswith("V:"))
        if vd:
            span = {}
            for d in vd:
                pts = sorted(set(t for s, tb in zip(dsets, tb_of) if d in s for t in tb))
                span[d] = pts
            multi = {d: p for d, p in span.items() if len(p) > 1}
            per_tb = collections.defaultdict(set)
            for d, p in span.items():
                for t in p:
                    per_tb[t].add(d)
            R["extra_donors"] = dict(
                n=len(vd), n_spanning_more_than_one_topbrain_patient=len(multi),
                spanning=multi,
                topbrain_patients_with_extra_donor=len(per_tb),
                topbrain_patients_with_several_extra_donors=sorted(t for t, s in per_tb.items() if len(s) > 1),
                one_to_one_nested=(len(multi) == 0),
                note=("every V: donor lies inside one TopBrain patient's anatomies -> its node is nested in "
                      "that patient's node and cannot change any quantity below" if not multi else
                      "V: donors shared across TopBrain patients couple those patients; quantities can drop"))
        R["roster_v3"] = roster_split(names, dsets, raws, V3_HOLDOUT_LOWERS, V3_HOLDOUT_PATIENTS)
        if solve:
            R["max"] = collections.OrderedDict()
            for k in test_sizes:
                m = max_split(dsets, k, time_limit=time_limit)
                m["test_anatomies"] = [names[a] for a in m.pop("test_idx", [])]
                m.pop("train_idx", None)
                R["max"]["k%d" % k] = m
                if verbose:
                    print("  %-6s max k=%d: test %s train %s kept %s dropped %s (%s, %ss)" % (
                        defn, k, m.get("test"), m.get("train"), m.get("kept"), m.get("dropped"),
                        "optimal" if m.get("optimal") else m.get("message"), m.get("sec")), flush=True)
                if div:
                    m = max_split(dsets, k, min_groups=[("T:", 6), ("Z:", 6)], time_limit=time_limit)
                    m["test_anatomies"] = [names[a] for a in m.pop("test_idx", [])]
                    m.pop("train_idx", None)
                    R["max"]["k%d_div6x6" % k] = m
                    if verbose:
                        print("  %-6s max k=%d div6x6: test %s train %s kept %s dropped %s (%s, %ss)" % (
                            defn, k, m.get("test"), m.get("train"), m.get("kept"), m.get("dropped"),
                            "optimal" if m.get("optimal") else m.get("message"), m.get("sec")), flush=True)
        if verbose:
            ro = R["roster_v3"]
            print("  %-6s roster_v3: test %d train %d excluded %d (v3-rule train %d, of which overlap test %d); "
                  "components %s; donors %s" % (defn, ro["test"], ro["train"], ro["excluded"], ro["v3_rule_train"],
                                                ro["v3_rule_train_overlapping_test"], R["components"]["sizes"][:3],
                                                R["donors"]), flush=True)
        res["defs"][defn] = R
    return res


def compare(res, base):
    """-> list of (def, quantity, base, now) where now < base"""
    lowered, rows = [], []
    for defn, R in res["defs"].items():
        B = (base.get("defs") or {}).get(defn)
        if not B:
            continue
        pairs = [("roster_v3", R["roster_v3"], B["roster_v3"])]
        for key, m in (R.get("max") or {}).items():
            if key in (B.get("max") or {}):
                pairs.append(("max." + key, m, B["max"][key]))
        for q, now, was in pairs:
            for f in ("train", "test", "kept"):
                if now.get(f) is None or was.get(f) is None:
                    continue
                rows.append((defn, q, f, was[f], now[f]))
                if now[f] < was[f]:
                    lowered.append((defn, q, f, was[f], now[f]))
    return lowered, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anatomies_dir")
    ap.add_argument("--out")
    ap.add_argument("--baseline")
    ap.add_argument("--defs", default=",".join(DEFS_ALL))
    ap.add_argument("--test-sizes", default="16,17")
    ap.add_argument("--no-div", action="store_true")
    ap.add_argument("--no-solve", action="store_true", help="roster/components/nesting only (no MILP)")
    ap.add_argument("--extra-donor-field", action="append", default=[])
    ap.add_argument("--extra-donor-regex")
    ap.add_argument("--no-auto-va", action="store_true")
    ap.add_argument("--donor-alias")
    ap.add_argument("--time-limit", type=float, default=300.0)
    a = ap.parse_args(argv)
    prov, build = load_set(a.anatomies_dir)
    if not prov:
        sys.exit("no */provenance.json under %s" % a.anatomies_dir)
    alias = json.load(open(a.donor_alias)) if a.donor_alias else None
    ex = DonorExtractor(a.extra_donor_field, a.extra_donor_regex, not a.no_auto_va, alias)
    print("v5_split: %s, %d anatomies%s" % (a.anatomies_dir, len(prov),
                                            (", build record %s" % build[0]) if build else ""), flush=True)
    res = analyse(prov, defs=[d for d in a.defs.split(",") if d],
                  test_sizes=[int(x) for x in a.test_sizes.split(",") if x], div=not a.no_div,
                  extractor=ex, time_limit=a.time_limit, solve=not a.no_solve)
    res["anatomies_dir"] = a.anatomies_dir
    res["build_record"] = build[0] if build else None
    if build and isinstance(build[1], dict):
        g = build[1].get("graft") or {}
        res["build_record_graft"] = {k: g[k] for k in ("left_fork_own_other_side", "left_fork_substitute",
                                                       "mirrored", "native") if k in g}
    res["args"] = vars(a)
    rc = 0
    if res["checks"]:
        print("CHECKS:", res["checks"], flush=True)
        if any("TaG-Net" in k for k in res["checks"]):
            rc = 2
    # v5_assign extension (2026-10-02; verify_graft_reg item 12): the nesting of the extra (V:) donors is printed, and a
    # V: donor spanning more than one TopBrain patient is an error (exit 4) even when it lowers no quantity
    nest_bad = []
    for defn, R in (res.get("defs") or {}).items():
        ed = R.get("extra_donors")
        if ed is None:
            print("  %-6s extra (V:) donors: none" % defn, flush=True)
            continue
        print("  %-6s extra (V:) donors %d in %d TopBrain patients; spanning > 1 patient: %d; one_to_one_nested %s" % (
            defn, ed["n"], ed["topbrain_patients_with_extra_donor"], ed["n_spanning_more_than_one_topbrain_patient"],
            ed["one_to_one_nested"]), flush=True)
        if not ed["one_to_one_nested"]:
            nest_bad.append(defn)
    if nest_bad:
        print("NOT one-to-one nested: %s" % nest_bad, flush=True)
        rc = rc or 4
    if a.baseline:
        base = json.load(open(a.baseline))
        lowered, rows = compare(res, base)
        res["baseline"] = dict(path=a.baseline, compared=len(rows), lowered=lowered,
                               rows=[dict(defn=r[0], quantity=r[1], field=r[2], baseline=r[3], now=r[4]) for r in rows])
        for r in rows:
            print("  %-6s %-16s %-5s baseline %4d  now %4d%s" % (r + ("  LOWER" if r[4] < r[3] else "",)), flush=True)
        if lowered:
            print("LOWERED vs baseline: %d quantities" % len(lowered), flush=True)
            rc = rc or 3
        else:
            print("not lowered vs baseline (%d quantities compared)" % len(rows), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print("wrote", a.out, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
