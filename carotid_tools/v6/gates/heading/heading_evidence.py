#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\heading\heading_evidence.py (= /sp/v6/heading/heading_evidence.py in the build containers)
"""v6_plan task 3: settle the heading maps with evidence (read-only on /repo).

A. TopBrain (25 patients, raw centrelines): is the source frame RAS (+x patient right, +y anterior, +z superior)?
   A1 rICA lies at +x of lICA; A2 each ICA lies at +y (anterior) of the same-side VA; A3 ICA / VA rise in +z;
   A4 the ECA piece lies anterior (+y) / lateral of the ICA at the same z (calibrates the fork landmark below);
   A5 VA ends converge (basilar) near the patient midline.
B. Host branch frame (eve_bench dualdevicenav, Centrelines_comb, json -> branch (y, -z, -x)): +x left, +y posterior,
   +z superior? B1 RCCA at -x of LCCA; B2 VAs at +y of the CCAs; B3 RCCA shares its proximal course with the RVA
   (brachiocephalic trunk -> right subclavian), the LCCA does not share with the LVA (labels are anatomical, not
   mirrored); B4 chirality (x cross y = z for a proper map).
C. Zenodo lower forks (carotid_bifurcation_database, scanner frame): per patient with both sides
   C1 left-vs-right bifurcation x (patient left-right axis); C2 superior z; C3 ECA anterior of ICA (A4 landmark) in a
   10-20 mm window above the bifurcation; C4 ECA medial/lateral; C5 the CCA's anterior-posterior lean.
   -> the scanner frame (LPS / RAS / mirrored) and whether its y sign is settled.
D. v5 placed forks (446 seam-1 placements): yaw of R_lower against the anatomical map settled in C, the placed
   ECA-anterior landmark, contralateral fork use, the bend an anatomical fork pose would need at seam 1.
Writes /sp/v6/heading/heading_evidence.json and .txt.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/repo/carotid_tools")
REPO = "/repo"
TB = REPO + "/topbrain_data"
ZEN = REPO + "/carotid_data/v2/carotid_bifurcation_database"
HOST = REPO + "/eve_bench/data/dualdevicenav/Centrelines_comb"
V5 = REPO + "/carotid_data/anatomies_v5"
OUT = "/sp/v6/heading"
L = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def arcl(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def window(p, s0, s1):
    s = arcl(p)
    k = (s >= s0) & (s <= s1)
    return p[k] if k.sum() >= 2 else None


def at_s(p, st):
    s = arcl(p)
    return np.array([np.interp(st, s, p[:, i]) for i in range(3)])


def twist_z(M):
    qw = 0.5 * np.sqrt(max(1e-12, 1 + M[0, 0] + M[1, 1] + M[2, 2]))
    qz = (M[1, 0] - M[0, 1]) / (4 * qw)
    y = float(np.degrees(2 * np.arctan2(qz, qw)))
    return (y + 180) % 360 - 180


def j2b(p):
    p = np.asarray(p, float)
    return np.stack([p[:, 1], -p[:, 2], -p[:, 0]], 1)


def read_mrk(path):
    d = json.load(open(path, encoding="utf-8"))
    return j2b([c["position"] for c in d["markups"][0]["controlPoints"]])


def tbload(sub, pid, suf):
    f = "%s/%s/%s_%s.json" % (TB, sub, pid, suf)
    return np.asarray(json.load(open(f))["points"], float) if os.path.exists(f) else None


def frac(a, n):
    return "%d/%d" % (a, n)


# ------------------------------------------------------------------ A. TopBrain
def part_a():
    log("== A. TopBrain raw centrelines (claimed RAS: +x patient right, +y anterior, +z superior)")
    pids = sorted({os.path.basename(f)[:13] for f in glob.glob(TB + "/centerlines*/topcow_mr_*.json")})
    res = {}
    a1 = a2 = a2n = a3 = a3n = 0
    a4 = {"ant": 0, "lat": 0, "n": 0, "dy": [], "dlat": []}
    a5 = []
    for pid in pids:
        ri, li = tbload("centerlines", pid, "ica"), tbload("centerlines_left", pid, "ica")
        rv, lv = tbload("centerlines_rva", pid, "va"), tbload("centerlines_lva", pid, "va")
        re_, le = tbload("centerlines_reca", pid, "eca"), tbload("centerlines_leca", pid, "eca")
        r = {}
        if ri is not None and li is not None:
            r["rICA_minus_lICA_x"] = float(ri[:, 0].mean() - li[:, 0].mean())
            a1 += r["rICA_minus_lICA_x"] > 0
        for ica, va, tag in ((ri, rv, "R"), (li, lv, "L")):
            if ica is not None and va is not None:
                d = float(ica[:, 1].mean() - va[:, 1].mean())
                r["ICA_minus_VA_y_" + tag] = d
                a2n += 1
                a2 += d > 0
            for v, nm in ((ica, "ICA"), (va, "VA")):
                if v is not None:
                    a3n += 1
                    a3 += (v[-1, 2] - v[0, 2]) > 0
        # A4: ECA vs ICA at the same z (first 10 mm of the ECA piece)
        for ica, eca, sg, tag in ((ri, re_, 1.0, "R"), (li, le, -1.0, "L")):
            if ica is None or eca is None:
                continue
            w = window(eca, 0.0, 10.0)
            if w is None:
                continue
            zc = w[:, 2].mean()
            k = int(np.argmin(np.abs(ica[:, 2] - zc)))
            if abs(ica[k, 2] - zc) > 3.0:
                continue
            dy = float(w[:, 1].mean() - ica[k, 1])
            dl = float(sg * (w[:, 0].mean() - ica[k, 0]))    # + = ECA lateral of the ICA (RAS: right = +x)
            r["ECA_minus_ICA_y_" + tag] = dy
            r["ECA_lateral_of_ICA_" + tag] = dl
            a4["n"] += 1
            a4["ant"] += dy > 0
            a4["lat"] += dl > 0
            a4["dy"].append(dy)
            a4["dlat"].append(dl)
        if rv is not None and li is not None and lv is not None and ri is not None:
            xm = (ri[-1, 0] + li[-1, 0]) / 2
            a5.append((float(np.linalg.norm(rv[-1] - lv[-1])), float(rv[-1, 0] - lv[-1, 0])))
        res[pid] = r
    n_pair = sum(1 for r in res.values() if "rICA_minus_lICA_x" in r)
    log("  A1 rICA at +x of lICA (mean x):", frac(a1, n_pair), "-> +x = patient right")
    log("  A2 ICA at +y of the same-side VA (ICA anterior of VA):", frac(a2, a2n), "-> +y = anterior")
    log("  A3 ICA / VA rise in +z (end above start):", frac(a3, a3n), "-> +z = superior")
    log("  A4 ECA piece vs ICA at the same z: ECA at +y (anterior)", frac(a4["ant"], a4["n"]),
        "median dy %.1f mm; ECA lateral of ICA %s, median %.1f mm" % (
            np.median(a4["dy"]) if a4["dy"] else np.nan, frac(a4["lat"], a4["n"]),
            np.median(a4["dlat"]) if a4["dlat"] else np.nan))
    if a5:
        a5 = np.array(a5)
        log("  A5 VA ends: separation median %.1f mm, rVA end at +x of lVA end (right of it) in %s" % (
            np.median(a5[:, 0]), frac(int((a5[:, 1] > -0.5).sum()), len(a5))))
    log("  chirality: (+x right, +y anterior, +z superior) = RAS, right-handed -> TopBrain sources are not mirrored")
    return {"per_patient": res, "A1": [a1, n_pair], "A2": [a2, a2n], "A3": [a3, a3n],
            "A4": {"ant": a4["ant"], "lat": a4["lat"], "n": a4["n"],
                   "dy_median": float(np.median(a4["dy"])) if a4["dy"] else None,
                   "dlat_median": float(np.median(a4["dlat"])) if a4["dlat"] else None}}


# ------------------------------------------------------------------ B. host
def common_prefix_mm(a, b, tol=1.0):
    d = np.array([np.min(np.linalg.norm(b - p, axis=1)) for p in a])
    far = np.nonzero(d > tol)[0]
    k = int(far[0]) if len(far) else len(a)
    return float(arcl(a[:max(k, 1)])[-1]), k


def part_b():
    log("== B. host branch frame (claimed LPS-like: +x patient left, +y posterior, +z superior)")
    H = {b: read_mrk("%s/Centerline curve - %s.mrk.json" % (HOST, b)) for b in ("RCCA", "LCCA", "RVA", "LVA")}
    out = {}
    for z in (440, 470, 500):
        row = {}
        for b, p in H.items():
            i = int(np.argmin(np.abs(p[:, 2] - z)))
            row[b] = p[i].round(1).tolist()
        out["z%d" % z] = row
        log("  z %d: " % z + " | ".join("%s x %.1f y %.1f" % (b, v[0], v[1]) for b, v in row.items()))
    b1 = all(out[k]["RCCA"][0] < out[k]["LCCA"][0] for k in out if k.startswith("z"))
    b2 = all(min(out[k]["RVA"][1], out[k]["LVA"][1]) > max(out[k]["RCCA"][1], out[k]["LCCA"][1])
             for k in out if k.startswith("z"))
    log("  B1 RCCA at -x of LCCA at every level:", b1, "-> +x = patient left (if labels are anatomical)")
    log("  B2 both VAs at +y of both CCAs at every level:", b2, "-> +y = posterior")
    rise = {b: float(p[-1, 2] - p[0, 2]) for b, p in H.items()}
    log("  B3a every route rises (end z - start z):", {k: round(v, 1) for k, v in rise.items()})
    # B3: the arch tree. The host curves are segments; find what each named route starts on.
    seg = {}
    for f in sorted(glob.glob(HOST + "/Centerline curve (*).mrk.json")):
        seg[os.path.basename(f)[16:-9]] = read_mrk(f)
    def starts_on(p0):
        hits = []
        for k, q in seg.items():
            i = int(np.argmin(np.linalg.norm(q - p0, axis=1)))
            if np.linalg.norm(q[i] - p0) < 0.5:
                hits.append((k, "end" if i == len(q) - 1 else ("start" if i == 0 else "mid")))
        return hits
    st = {b: starts_on(H[b][0]) for b in H}
    log("  B3 route starts: " + "; ".join("%s at %s on %s" % (b, H[b][0].round(1).tolist(), st[b]) for b in H))
    shared = bool(np.linalg.norm(H["RCCA"][0] - H["RVA"][0]) < 0.5)
    trunk = [k for k, w in st["RCCA"] if w == "end"]
    tk = seg[trunk[0]] if trunk else None
    log("  B3 RCCA and RVA start at the same point (a shared trunk = brachiocephalic): %s; trunk segment %s runs "
        "%s -> %s (%.1f mm, toward %s x)" % (shared, trunk, None if tk is None else tk[0].round(1).tolist(),
                                            None if tk is None else tk[-1].round(1).tolist(),
                                            0 if tk is None else arcl(tk)[-1],
                                            "-" if tk is not None and tk[-1, 0] < tk[0, 0] else "+"))
    # subclavians to the arms: the segment leaving the RVA start region toward the arm and the one past the LVA start
    arms = sorted(((float(q[-1, 0]), k) for k, q in seg.items() if abs(q[-1, 2] - 380) < 40 and arcl(q)[-1] > 80))
    log("  B3 long segments ending near z 340-420 (subclavians to the arms), end x:", [(k, round(x, 1)) for x, k in arms])
    # B5: the aortic arch segment that the brachiocephalic / LCCA start on, start -> end direction
    arch = [k for k, w in st["LCCA"] if w == "mid"]
    if arch:
        q = seg[arch[0]]
        d = q[-1] - q[0]
        log("  B5 arch segment %s from %s to %s: dx %+.1f (toward +x), dy %+.1f (toward +y); the arch runs from the "
            "ascending aorta (anterior, right) to the descending aorta (posterior, left)" % (
                arch[0], q[0].round(1).tolist(), q[-1].round(1).tolist(), d[0], d[1]))
        out["B5_arch_d"] = d.round(2).tolist()
    log("  B4 chirality: (+x left, +y posterior, +z superior) = LPS, right-handed (x cross y = z) -> proper")
    out.update({"B1": b1, "B2": b2, "rise": rise, "B3_rcca_rva_shared_start": shared, "route_starts": st})
    return out


# ------------------------------------------------------------------ C. Zenodo forks
def load_fork(path):
    from analyze_bifurcations import read_centerlines, split_tree
    paths = read_centerlines(path)
    cca, dau = split_tree(paths)
    if cca is None or len(dau) < 2:
        return None

    def cal(r):
        return float(np.median(r[:max(len(r) // 2, 1)])) if r is not None and len(r) else 0.0
    dau = sorted(dau, key=lambda d: -cal(d[1]))
    return cca[0], dau[0][0], dau[1][0]     # cca, ica, eca points


def fork_landmarks(cca, ica, eca, s0=10.0, s1=20.0):
    """ECA - ICA offset (mean over a [s0, s1] mm arclength window above the bifurcation), CCA lean, superior."""
    wi, we = window(ica, s0, s1), window(eca, s0, s1)
    if wi is None or we is None:
        return None
    d = we.mean(0) - wi.mean(0)
    c = cca[-1] - (cca[0] if arcl(cca)[-1] < 25 else at_s(cca, arcl(cca)[-1] - 25.0))
    sup = ica[-1] - cca[0]
    si = arcl(ica)[-1]
    idist = ica[-1] - at_s(ica, max(si - 10.0, 0.0))
    return {"eca_minus_ica": d.tolist(), "cca_dir": unit(c).tolist(), "sup": unit(sup).tolist(),
            "bif": cca[-1].tolist(), "ica_tip_minus_bif": (ica[-1] - cca[-1]).tolist(),
            "ica_distal_dir": unit(idist).tolist(), "ica_mm": float(si),
            "eca_tip_minus_bif": (eca[-1] - cca[-1]).tolist()}


RZ180 = np.diag([-1.0, -1.0, 1.0])


def anat_summary(vals, side_of, label):
    """vals: list of (name, side, v_anat LPS). Report anterior (-y) and medial (+x right fork, -x left fork)."""
    a = np.array([-v[1] for _, s, v in vals])
    m = np.array([(v[0] if s == "right" else -v[0]) for _, s, v in vals])
    log("    %-34s n %3d  anterior %3d (median %+.2f)  medial %3d (median %+.2f)" % (
        label, len(vals), int((a > 0).sum()), np.median(a), int((m > 0).sum()), np.median(m)))
    return {"n": len(vals), "anterior": int((a > 0).sum()), "ant_median": float(np.median(a)),
            "medial": int((m > 0).sum()), "med_median": float(np.median(m))}


def part_c():
    log("== C. Zenodo lower forks (carotid_bifurcation_database, scanner frame)")
    rows = {}
    for f in sorted(glob.glob(ZEN + "/case_*/case_*_lumen_centerlines.vtp")):
        name = os.path.basename(f).replace("_lumen_centerlines.vtp", "")
        try:
            fk = load_fork(f)
        except Exception as e:  # noqa
            rows[name] = {"error": repr(e)[:120]}
            continue
        if fk is None:
            rows[name] = {"error": "no split"}
            continue
        lm = {w: fork_landmarks(*fk, s0=a, s1=b) for w, (a, b) in {"w5_15": (5, 15), "w10_20": (10, 20)}.items()}
        rows[name] = {"side": "left" if name.endswith("_left") else "right",
                      "patient": name.rsplit("_", 1)[0], "lm": lm}
    pats = {}
    for n, r in rows.items():
        if "error" in r:
            continue
        pats.setdefault(r["patient"], {})[r["side"]] = r
    # C1: left-right axis per patient
    dx = []
    for p, sides in pats.items():
        if "left" in sides and "right" in sides:
            dx.append(sides["left"]["lm"]["w10_20"]["bif"][0] - sides["right"]["lm"]["w10_20"]["bif"][0]
                      if sides["left"]["lm"]["w10_20"] and sides["right"]["lm"]["w10_20"] else np.nan)
    dx = np.array([v for v in dx if np.isfinite(v)])
    log("  forks loaded %d (errors %d); patients with both sides %d" % (
        sum(1 for r in rows.values() if "error" not in r), sum(1 for r in rows.values() if "error" in r), len(dx)))
    log("  C1 left-fork bifurcation x minus right-fork x: positive in %s (median %.1f mm, min %.1f, max %.1f)"
        % (frac(int((dx > 0).sum()), len(dx)), np.median(dx), dx.min(), dx.max()))
    sup = np.array([r["lm"]["w10_20"]["sup"] for r in rows.values() if "error" not in r and r["lm"]["w10_20"]])
    log("  C2 CCA-inlet -> ICA-tip direction z > 0 in %s (median z %.2f)" % (frac(int((sup[:, 2] > 0).sum()), len(sup)),
                                                                         np.median(sup[:, 2])))
    out = {"C1_dx": dx.tolist(), "rows": rows}
    for w in ("w5_15", "w10_20"):
        dys, dls, n = [], [], 0
        for r in rows.values():
            if "error" in r or not r["lm"][w]:
                continue
            d = np.asarray(r["lm"][w]["eca_minus_ica"])
            sg = -1.0 if r["side"] == "right" else 1.0   # lateral sign if +x = patient left: right fork lateral = -x
            dys.append(d[1])
            dls.append(sg * d[0])
        dys, dls = np.array(dys), np.array(dls)
        log("  C3 [%s mm] ECA - ICA y: negative in %s, positive in %s (median %.2f mm; |dy| > 2 mm: neg %d / pos %d)"
            % (w, frac(int((dys < 0).sum()), len(dys)), frac(int((dys > 0).sum()), len(dys)), np.median(dys),
               int((dys < -2).sum()), int((dys > 2).sum())))
        log("  C4 [%s mm] ECA lateral of ICA (lateral = -x on a right fork, +x on a left, IF +x = left): %s (median %.2f mm)"
            % (w, frac(int((dls > 0).sum()), len(dls)), np.median(dls)))
        out["C3_" + w] = {"neg": int((dys < 0).sum()), "pos": int((dys > 0).sum()), "n": int(len(dys)),
                          "median": float(np.median(dys)), "neg_gt2": int((dys < -2).sum()), "pos_gt2": int((dys > 2).sum())}
        out["C4_" + w] = {"lat": int((dls > 0).sum()), "n": int(len(dls)), "median": float(np.median(dls))}
    # per patient: do the two sides agree on the ECA y sign?
    agree = tot = 0
    for p, sides in pats.items():
        if "left" in sides and "right" in sides and sides["left"]["lm"]["w10_20"] and sides["right"]["lm"]["w10_20"]:
            a = np.sign(sides["left"]["lm"]["w10_20"]["eca_minus_ica"][1])
            b = np.sign(sides["right"]["lm"]["w10_20"]["eca_minus_ica"][1])
            tot += 1
            agree += a == b
    log("  C3b both sides of one patient agree on the ECA y sign: %s" % frac(agree, tot))
    cd = np.array([r["lm"]["w10_20"]["cca_dir"] for r in rows.values() if "error" not in r and r["lm"]["w10_20"]])
    log("  C5 CCA distal 25 mm direction: y component median %.2f (neg %d / pos %d), z median %.2f" % (
        np.median(cd[:, 1]), int((cd[:, 1] < 0).sum()), int((cd[:, 1] > 0).sum()), np.median(cd[:, 2])))
    out["C3b_agree"] = [agree, tot]
    out["C5_cca_y"] = {"median": float(np.median(cd[:, 1])), "neg": int((cd[:, 1] < 0).sum()), "pos": int((cd[:, 1] > 0).sum())}

    # ---- C6: per-patient frames. xdir +1: +x = patient left (LPS-like); -1: +x = patient right (RAS-like).
    log("  C6 per-patient frame (the scanner frames are NOT one frame):")
    xdir, how = {}, {}
    for p, sides in pats.items():
        if "left" in sides and "right" in sides and sides["left"]["lm"]["w10_20"] and sides["right"]["lm"]["w10_20"]:
            xdir[p] = 1 if sides["left"]["lm"]["w10_20"]["bif"][0] > sides["right"]["lm"]["w10_20"]["bif"][0] else -1
            how[p] = "bilateral"
    # unilateral: the scanner origin sits near the patient midline -> sign of the bifurcation x and the side
    ok = n_ = 0
    for p, sides in pats.items():
        for sd, r in sides.items():
            if not r["lm"]["w10_20"]:
                continue
            g = (1 if r["lm"]["w10_20"]["bif"][0] > 0 else -1) * (1 if sd == "left" else -1)
            if p in xdir and how[p] == "bilateral":
                n_ += 1
                ok += g == xdir[p]
            elif p not in xdir:
                xdir[p] = g
                how[p] = "origin_sign"
    cnt = {"bilateral_lps": sum(1 for p in xdir if how[p] == "bilateral" and xdir[p] == 1),
           "bilateral_ras": sum(1 for p in xdir if how[p] == "bilateral" and xdir[p] == -1),
           "origin_lps": sum(1 for p in xdir if how[p] == "origin_sign" and xdir[p] == 1),
           "origin_ras": sum(1 for p in xdir if how[p] == "origin_sign" and xdir[p] == -1)}
    log("    bilateral patients: +x = left (LPS-like) %d, +x = right (RAS-like) %d" % (cnt["bilateral_lps"], cnt["bilateral_ras"]))
    log("    rule for one-sided patients (sign of the bifurcation x relative to the scanner origin) agrees with the "
        "bilateral answer on %s forks; applied to %d one-sided patients (LPS-like %d, RAS-like %d)" % (
            frac(ok, n_), cnt["origin_lps"] + cnt["origin_ras"], cnt["origin_lps"], cnt["origin_ras"]))
    by_grp = {}
    for p, v in xdir.items():
        g = p.split("_")[1]
        by_grp.setdefault(g, [0, 0])[0 if v == 1 else 1] += 1
    log("    by group (LPS-like, RAS-like):", by_grp)
    # does the CCA y-lean flip with xdir (proper frames 180 deg apart about z) or not (mirror)?
    flip = same = 0
    for r in rows.values():
        if "error" in r or not r["lm"]["w10_20"] or how.get(r["patient"]) != "bilateral":
            continue
        cy = r["lm"]["w10_20"]["cca_dir"][1]
        flip += (cy * xdir[r["patient"]]) < 0
        same += (cy * xdir[r["patient"]]) > 0
    log("    CCA distal y-lean sign opposite to xdir (frames differ by a 180 deg turn about z, x and y flip together): "
        "%s; same sign (an x-only mirror) %d" % (frac(flip, flip + same), same))
    # anatomical coordinates (LPS), assuming each scanner frame is proper: A = I (LPS-like) or Rz(180) (RAS-like)
    A = {p: (np.eye(3) if v == 1 else RZ180) for p, v in xdir.items()}
    log("    landmarks in anatomical LPS coordinates, assuming proper frames (anterior = -y; medial = toward the midline):")
    out["C6"] = {"xdir": xdir, "how": how, "counts": cnt, "rule_agree": [ok, n_], "by_group": by_grp,
                 "cca_lean_flip": [flip, flip + same]}
    for key, lab in (("cca_dir", "CCA distal 25 mm, going up"), ("eca_minus_ica", "ECA - ICA, 10-20 mm window"),
                     ("ica_tip_minus_bif", "ICA tip - bifurcation"), ("ica_distal_dir", "ICA distal 10 mm direction"),
                     ("eca_tip_minus_bif", "ECA tip - bifurcation")):
        vals = [(n, r["side"], A[r["patient"]] @ np.asarray(r["lm"]["w10_20"][key]))
                for n, r in rows.items() if "error" not in r and r["lm"]["w10_20"] and r["patient"] in A]
        out["C6"][key] = anat_summary(vals, None, lab)
    vals = [(n, r["side"], A[r["patient"]] @ np.asarray(r["lm"]["w5_15"]["eca_minus_ica"]))
            for n, r in rows.items() if "error" not in r and r["lm"]["w5_15"] and r["patient"] in A]
    out["C6"]["eca_minus_ica_w5_15"] = anat_summary(vals, None, "ECA - ICA, 5-15 mm window")
    # the same landmarks on TopBrain (RAS -> LPS by Rz(180)): the lowest 10 mm of each ICA, going up
    tv = []
    for f in sorted(glob.glob(TB + "/centerlines/*_ica.json")) + sorted(glob.glob(TB + "/centerlines_left/*_ica.json")):
        p = np.asarray(json.load(open(f))["points"], float)
        sd = "right" if "/centerlines/" in f else "left"
        tv.append((f, sd, RZ180 @ unit(at_s(p, 10.0) - p[0])))
    out["C6"]["topbrain_ica_lowest_10mm"] = anat_summary(tv, None, "TopBrain ICA lowest 10 mm (known RAS)")
    out["frames"] = {p: ("LPS" if v == 1 else "RAS") for p, v in xdir.items()}
    return out


def host_cca_lean():
    H = {b: read_mrk("%s/Centerline curve - %s.mrk.json" % (HOST, b)) for b in ("RCCA", "LCCA")}
    res = {}
    for b, p in H.items():
        k = (p[:, 2] > 420) & (p[:, 2] < 470)
        q = p[k]
        res[b] = unit(q[-1] - q[0]).round(3).tolist() if len(q) > 2 else None
    return res


# ------------------------------------------------------------------ D. v5 placed forks
def part_d(frames):
    log("== D. v5 placed forks (seam 1), against the per-patient anatomical map (I for LPS-like, Rz(180) for RAS-like)")
    rows = []
    for d in sorted(glob.glob(V5 + "/case_*")):
        n = os.path.basename(d)
        prov = json.load(open(d + "/provenance.json", encoding="utf-8"))
        for key, br, eca_br, side, lname in (("lower", "RCCA", "RECA", "right", prov.get("lower")),
                                             ("lower_L", "LCCA", "LECA", "left", (prov.get("left") or {}).get("lower"))):
            x = prov["xform"].get(key)
            if not x:
                continue
            R = np.asarray(x["R"], float)
            fr = frames.get((lname or "").rsplit("_", 1)[0])
            A_zen = np.eye(3) if fr == "LPS" else RZ180
            M = np.asarray(A_zen, float).T @ R
            yaw = twist_z(M)
            tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            cur = read_mrk("%s/Centrelines_comb/Centerline curve - %s.mrk.json" % (d, br))
            eca = read_mrk("%s/Centrelines_comb/Centerline curve - %s.mrk.json" % (d, eca_br))
            anchor = np.asarray(x["anchor"], float)
            i0 = int(np.argmin(np.linalg.norm(cur - anchor, axis=1)))
            t_out = at_s(cur[i0:], 5.0) - cur[i0]
            t_src = R.T @ t_out
            bend = ang(t_out, np.asarray(A_zen, float) @ t_src)
            # bifurcation on the route = nearest route point to the ECA's first point
            ib = int(np.argmin(np.linalg.norm(cur - eca[0], axis=1)))
            ica = cur[ib:]
            wi, we = window(ica, 10, 20), window(eca, 10, 20)
            d_eca = (we.mean(0) - wi.mean(0)) if (wi is not None and we is not None) else None
            fork_side = "left" if (lname or "").endswith("_left") else ("right" if (lname or "").endswith("_right") else "?")
            # the alternative for a contralateral fork: the map turned 180 deg about z (medial-lateral kept, A-P lost)
            yaw_alt = twist_z((RZ180 @ A_zen).T @ R)
            bend_alt = ang(t_out, RZ180 @ A_zen @ t_src)
            # RICA / LICA seam (siphon anchor) distance from the host midline, own side positive
            sk = "siphon" if key == "lower" else "siphon_L"
            u_seam = None
            if sk in prov["xform"]:
                ax = prov["xform"][sk]["anchor"][0]
                u_seam = float((14.8 - ax) if side == "right" else (ax - 14.8))
            rows.append({"anatomy": n, "key": key, "route_side": side, "lower": lname, "fork_side": fork_side,
                         "frame": fr, "contralateral": fork_side != side, "yaw": yaw, "tilt": tilt,
                         "bend_for_anat": bend, "yaw_alt": yaw_alt, "bend_alt": bend_alt, "u_seam2": u_seam,
                         "eca_minus_ica": None if d_eca is None else d_eca.round(2).tolist(),
                         "bif_x": float(cur[ib, 0])})
    out = {"rows": rows}
    for key in ("lower", "lower_L"):
        R_ = [r for r in rows if r["key"] == key]
        y = np.abs([r["yaw"] for r in R_])
        b = np.array([r["bend_for_anat"] for r in R_])
        con = sum(r["contralateral"] for r in R_)
        ant = [r for r in R_ if r["eca_minus_ica"] is not None]
        ant_n = sum(1 for r in ant if r["eca_minus_ica"][1] < 0)      # branch: anterior = -y
        log("  %-8s n %d (contralateral fork %d): |yaw vs map| median %.1f, >45 %d, >90 %d; tilt median %.1f; "
            "bend to the host tangent if posed by the map: median %.1f, <=30 %d, <=45 %d; placed ECA anterior of ICA %s"
            % (key, len(R_), con, np.median(y), int((y > 45).sum()), int((y > 90).sum()),
               np.median([r["tilt"] for r in R_]), np.median(b), int((b <= 30).sum()), int((b <= 45).sum()),
               frac(ant_n, len(ant))))
        sub = {}
        for c in (False, True):
            S = [r for r in R_ if r["contralateral"] == c]
            if S:
                yy = np.abs([r["yaw"] for r in S])
                ya = np.abs([r["yaw_alt"] for r in S])
                bb = np.array([r["bend_for_anat"] for r in S])
                ba = np.array([r["bend_alt"] for r in S])
                a2 = sum(1 for r in S if r["eca_minus_ica"] is not None and r["eca_minus_ica"][1] < 0)
                log("      %s: n %d, |yaw| median %.1f, >45 %d, >90 %d; bend for the map <=45 %d (median %.1f); "
                    "map+180: |yaw| median %.1f, bend <=45 %d; placed ECA anterior %d" % (
                        "contralateral" if c else "same-side", len(S), np.median(yy), int((yy > 45).sum()),
                        int((yy > 90).sum()), int((bb <= 45).sum()), np.median(bb), np.median(ya),
                        int((ba <= 45).sum()), a2))
                sub["contralateral" if c else "same_side"] = {
                    "n": len(S), "yaw_abs_median": float(np.median(yy)), "yaw_gt45": int((yy > 45).sum()),
                    "yaw_gt90": int((yy > 90).sum()), "bend_le45": int((bb <= 45).sum()), "bend_le30": int((bb <= 30).sum()),
                    "bend_median": float(np.median(bb)), "alt_yaw_abs_median": float(np.median(ya)),
                    "alt_bend_le45": int((ba <= 45).sum()), "eca_anterior": a2}
        # C6: seam-2 near the midline vs fork heading
        us = [(r["u_seam2"], abs(r["yaw"])) for r in R_ if r["u_seam2"] is not None]
        near = [a for u, a in us if u < 10]
        far = [a for u, a in us if u >= 10]
        log("      siphon seam < 10 mm from the midline: %d (|yaw| median %.1f) vs >= 10 mm: %d (|yaw| median %.1f)" % (
            len(near), np.median(near) if near else np.nan, len(far), np.median(far) if far else np.nan))
        out[key] = {"n": len(R_), "contralateral": con, "yaw_abs_median": float(np.median(y)), "yaw_gt45": int((y > 45).sum()),
                    "yaw_gt90": int((y > 90).sum()), "bend_median": float(np.median(b)), "bend_le30": int((b <= 30).sum()),
                    "bend_le45": int((b <= 45).sum()), "eca_anterior": [ant_n, len(ant)], "by_side_use": sub,
                    "seam2_near_midline": [len(near), float(np.median(near)) if near else None]}
    # same Zenodo patient on both sides of one anatomy
    pairs = {}
    for r in rows:
        pairs.setdefault(r["anatomy"], {})[r["key"]] = r
    dd = []
    for n, p in pairs.items():
        if "lower" in p and "lower_L" in p and p["lower"]["lower"] and p["lower_L"]["lower"] \
                and p["lower"]["lower"].rsplit("_", 1)[0] == p["lower_L"]["lower"].rsplit("_", 1)[0]:
            dd.append(abs((p["lower"]["yaw"] - p["lower_L"]["yaw"] + 180) % 360 - 180))
    dd = np.array(dd)
    log("  same Zenodo patient on both sides: n %d, |yaw_R - yaw_L| median %.1f, >45 %d, >90 %d" % (
        len(dd), np.median(dd), int((dd > 45).sum()), int((dd > 90).sum())))
    out["same_patient_dyaw"] = {"n": int(len(dd)), "median": float(np.median(dd)), "gt45": int((dd > 45).sum()),
                                "gt90": int((dd > 90).sum())}
    return out


def main():
    res = {"A": part_a(), "B": part_b(), "host_cca_lean_420_470": host_cca_lean()}
    log("  host CCA direction (z 420-470, branch frame):", res["host_cca_lean_420_470"])
    res["C"] = part_c()
    res["D"] = part_d(res["C"]["frames"])
    json.dump(res["C"]["frames"], open(OUT + "/zenodo_frames.json", "w"), indent=0, sort_keys=True)
    json.dump(res, open(OUT + "/heading_evidence.json", "w"), indent=1, default=float)
    open(OUT + "/heading_evidence.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
