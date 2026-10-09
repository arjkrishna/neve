#!/usr/bin/env python3
# [carotid_tools/v6 copy] source: C:\Users\akrish41\AppData\Local\Temp\claude\d--Arjun-neve\afabf2e9-c644-45a6-95cd-d6e6d88be352\scratchpad\v6\va\subclavian_paths.py (= /sp/v6/va/subclavian_paths.py in the build containers)
"""v6_va step 1a: the donor's OWN subclavian path (origin -> VA ostium -> ~40 mm past it) for every donor side.

For each donor unit in donors_v5 (27: 25 primary + 2 reserves) and each built side, in the donor's own frame (RAS mm,
unmirrored, the frame of the side record):
  * path      Arterial labelled centreline (runs/*/labelled_centerlines.json): the chain of xSA-class segments from the
              segment nearest the VA ostium foot O_v (record) back to the subclavian ORIGIN (the end node that touches a
              BT / AA / CCA segment, i.e. the brachiocephalic split on the right, the arch on the left) and forward
              (largest-radius continuation) to >= 40 mm past O_v. Radius = Arterial's radius_mm.
              Fallback (no xSA chain within 6 mm of O_v, or no origin found): the TotalSegmentator subclavian mask
              (totalseg/segmentations, CTA grid) -- origin = mask voxel nearest the brachiocephalic_trunk (R) / aorta (L)
              mask, path = the record's subclavian (+-40 mm) joined to it. VMR 0248: the expert SimVascular subclavian
              path (sources.load_vmr0248), origin = its first point.
  * s_ostium  arclength of O_v's foot on the path (origin = 0); want [s_origin + 10, s_ostium + 20] (V6_ITEMS e).
  * ct_seg    what the v5 ct_seg run (the side's surface folder) already holds for this subclavian: the rsub / lsub
              station path (+-30 mm about the VA origin on ONE Arterial segment) projected on the path -> covered range.
  * fov       does the donor CT cover the wanted stretch (z range of the CT vs the path).
Writes /work/subclavian/paths/<donor_id>_<side>.json and /work/subclavian/paths/_summary.json.
Runs in neve-build-stagea (numpy, scipy, nibabel). Read-only on /repo.
"""
import glob
import json
import os
import sys

import numpy as np

DONORS = "/repo/carotid_data/va_sources/v5_reg/donors_v5"
DER = "/repo/carotid_data/va_sources/derived"
SRC = "/repo/carotid_data/va_sources/thin_slice"
OUT = "/work/subclavian/paths"
WANT_PROX_MM = 10.0     # seam this far after the subclavian origin
WANT_DIST_MM = 20.0     # stretch this far past the VA ostium
KEEP_PAST_MM = 45.0     # path kept this far past the ostium (room for the distal seam search)
BIG = {"R": ("BT", "RCCA", "AA", "LCCA"), "L": ("AA", "LCCA", "BT")}


def arclen(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def foot_on(sp, q):
    sp = np.asarray(sp, float)
    s = arclen(sp)
    k = int(np.argmin(np.linalg.norm(sp - q, axis=1)))
    best = (np.inf, sp[k], s[k])
    for a, b in ((max(k - 1, 0), k), (k, min(k + 1, len(sp) - 1))):
        if a == b:
            continue
        ab = sp[b] - sp[a]
        t = float(np.clip((q - sp[a]) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
        f = sp[a] + t * ab
        d = float(np.linalg.norm(f - q))
        if d < best[0]:
            best = (d, f, s[a] + t * (s[b] - s[a]))
    return best[1], best[0], float(best[2])


def resample(p, r, step=0.5):
    s = arclen(p)
    q = np.arange(0.0, s[-1] + 1e-9, step)
    P = np.stack([np.interp(q, s, p[:, i]) for i in range(3)], 1)
    R = np.interp(q, s, r) if r is not None else None
    return P, R


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(type(x))
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(o, f, indent=1, default=conv)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------------------------- Arterial chain
def arterial_chain(case_dir, side, O_v):
    """(points origin -> distal, radii, info) or (None, None, why)."""
    lcs = sorted(glob.glob(os.path.join(case_dir, "arterial", "runs", "*", "labelled_centerlines.json")))
    lcs = sorted(lcs, key=lambda f: (0 if "/native/" in f else 1, f))
    why = []
    for lcp in lcs:
        segs = jload(lcp)["segments"]
        cls = side + "SA"
        S = []
        for i, sg in enumerate(segs):
            P = np.asarray(sg["points_ras_mm"], float)
            if len(P) < 2:
                continue
            S.append({"i": i, "cls": sg["class"], "a": sg["nodes"][0], "b": sg["nodes"][1], "P": P,
                      "r": np.asarray(sg["radius_mm"], float)})
        sa = [s for s in S if s["cls"] == cls]
        if not sa:
            why.append("%s: no %s segment" % (lcp, cls))
            continue
        d = [float(np.linalg.norm(s["P"] - O_v, axis=1).min()) for s in sa]
        k = int(np.argmin(d))
        if d[k] > 6.0:
            why.append("%s: nearest %s segment %.1f mm from O_v" % (lcp, cls, d[k]))
            continue
        seg0 = sa[k]
        node_segs = {}
        for s in S:
            node_segs.setdefault(s["a"], []).append(s)
            node_segs.setdefault(s["b"], []).append(s)

        def walk(node, came, pick, allow_other=False):
            """Follow cls segments from `node` (not `came`); return list of (seg, reversed?) and the terminal node.
            allow_other: also wide 'other' segments (median r >= 2.5 mm; Arterial left the medial SCA unlabelled)."""
            out, used = [], {came["i"]}
            cur = node
            for _ in range(50):
                nxt = [s for s in node_segs.get(cur, []) if (s["cls"] == cls or (allow_other and s["cls"] == "other"
                       and float(np.median(s["r"])) >= 2.5)) and s["i"] not in used]
                if not nxt:
                    break
                s = pick(nxt)
                used.add(s["i"])
                rev = s["b"] == cur
                out.append((s, rev))
                cur = s["a"] if rev else s["b"]
            return out, cur

        def touches_big(node):
            return sorted({s["cls"] for s in node_segs.get(node, []) if s["cls"] in BIG[side]})

        cands = []
        for end_node, other in ((seg0["a"], seg0["b"]), (seg0["b"], seg0["a"])):
            prox, term = walk(end_node, seg0, lambda xs: max(xs, key=lambda s: float(np.median(s["r"]))))
            cands.append((end_node, other, prox, term, touches_big(term)))
        if not any(c[4] for c in cands):
            cands = []
            for end_node, other in ((seg0["a"], seg0["b"]), (seg0["b"], seg0["a"])):
                prox, term = walk(end_node, seg0, lambda xs: max(xs, key=lambda s: float(np.median(s["r"]))), True)
                cands.append((end_node, other, prox, term, touches_big(term)))
            why.append("%s: medial walk through wide 'other' segments" % lcp)
        # the proximal direction = the walk that ends on a big vessel; tie -> the one ending nearer the midline
        good = [c for c in cands if c[4]]
        if len(good) == 1:
            pc = good[0]
        elif len(good) == 2:
            # both ends touch big vessels (e.g. a short chain): take the one whose terminal is more medial
            def term_x(c):
                seg = c[2][-1][0] if c[2] else seg0
                node = c[3]
                P = seg["P"]
                pt = P[0] if seg["a"] == node else P[-1]
                return abs(pt[0])
            pc = min(good, key=term_x)
        else:
            # no parent contact (FOV / labels): the walk whose terminal is the more MEDIAL end (RAS: right side medial
            # = smaller x, left side = larger x); flagged
            def term_pt(c):
                seg = c[2][-1][0] if c[2] else seg0
                P = seg["P"]
                return P[0] if seg["a"] == c[3] else P[-1]
            sg = 1.0 if side == "R" else -1.0
            pc = min(cands, key=lambda c: sg * term_pt(c)[0])
            why.append("%s: no origin node touching %s -> medial chain end used" % (lcp, BIG[side]))
            pc = (pc[0], pc[1], pc[2], pc[3], ["none (medial chain end; no parent contact)"])
        end_node, other_node, prox, term, big = pc
        # assemble origin -> seg0 -> distal
        pts, rad = [], []
        for s, rev in reversed(prox):
            # this segment walked away from seg0 (rev means walked b->a); assemble origin-first = opposite direction
            P, r = (s["P"], s["r"]) if rev else (s["P"][::-1], s["r"][::-1])
            pts.append(P); rad.append(r)
        P0, r0 = (seg0["P"], seg0["r"]) if seg0["a"] == end_node else (seg0["P"][::-1], seg0["r"][::-1])
        pts.append(P0); rad.append(r0)
        dist, dterm = walk(other_node, seg0, lambda xs: max(xs, key=lambda s: float(np.median(s["r"])) * min(s["P"].shape[0], 40)))
        for s, rev in dist:
            P, r = (s["P"][::-1], s["r"][::-1]) if rev else (s["P"], s["r"])
            pts.append(P); rad.append(r)
        P = np.vstack(pts)
        r = np.concatenate(rad)
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
        P, r = P[keep], r[keep]
        info = {"src": lcp, "class": cls, "origin_node": int(term), "origin_touches": big,
                "segments": [int(s["i"]) for s, _ in reversed(prox)] + [int(seg0["i"])] + [int(s["i"]) for s, _ in dist],
                "seg0_dist_to_Ov_mm": d[k]}
        return P, r, info
    return None, None, "; ".join(why) or "no labelled_centerlines.json"


# ---------------------------------------------------------------------------------------------- TS fallback
def ts_origin(case_dir, side):
    import nibabel as nib
    seg = os.path.join(case_dir, "totalseg", "segmentations")
    nm = "subclavian_artery_right" if side == "R" else "subclavian_artery_left"
    ref = "brachiocephalic_trunk" if side == "R" else "aorta"
    out = {}
    try:
        a = nib.load(os.path.join(seg, nm + ".nii.gz"))
        b = nib.load(os.path.join(seg, ref + ".nii.gz"))
    except FileNotFoundError as e:
        return None, "TS mask missing: %s" % e
    ma = np.asanyarray(a.dataobj) > 0
    mb = np.asanyarray(b.dataobj) > 0
    if not ma.any() or not mb.any():
        return None, "TS mask empty (%s %d, %s %d)" % (nm, int(ma.sum()), ref, int(mb.sum()))
    from scipy import ndimage
    dil = ndimage.binary_dilation(mb, iterations=2)
    touch = np.argwhere(ma & dil)
    if not len(touch):
        return None, "TS %s does not touch %s" % (nm, ref)
    W = (a.affine @ np.c_[touch, np.ones(len(touch))].T)[:3].T
    out = W.mean(0)
    return out, "TS %s voxels touching %s (dilated 2), centroid of %d" % (nm, ref, len(touch))


def ct_fov(source, case):
    if source == "vmr0248":
        return {"z_mm": [-311.5, 0.0], "note": "VMR p12full.vti bounds (inventory.json)"}
    import nibabel as nib
    f = os.path.join(SRC, source, case, "ct.nii.gz")
    if not os.path.isfile(f):
        return None
    im = nib.load(f)
    A = im.affine
    sh = np.array(im.shape[:3])
    corners = np.array([[i, j, k, 1] for i in (0, sh[0] - 1) for j in (0, sh[1] - 1) for k in (0, sh[2] - 1)], float)
    W = (A @ corners.T)[:3].T
    return {"lo": W.min(0).round(2).tolist(), "hi": W.max(0).round(2).tolist(), "file": f}


# ---------------------------------------------------------------------------------------------- per side
def one(donor_id, side):
    rec = jload(os.path.join(DONORS, donor_id, "%s.json" % side))
    src, case = rec["source"], rec["case_id"]
    Pva = np.asarray(rec["centerline"]["points"], float)
    O_v = np.asarray(rec["ostium"]["O_v"], float)
    seg_dir = os.path.dirname(rec["surface"]["src"]) if rec.get("surface") else None
    out = {"donor_id": donor_id, "side": side, "source": src, "case": case, "frame": "donor RAS mm (unmirrored)",
           "O_v": O_v, "va_start": Pva[0], "seg_dir_v5": seg_dir}
    if src == "vmr0248":
        sys.path.insert(0, "/sp/v5_va")
        from sources import load_vmr0248
        v = load_vmr0248(with_ct=False, with_model=False, with_walls=False)
        nm = ("right" if side == "R" else "left") + "_subclavian"
        P = np.asarray(v["vessels"][nm]["points"], float)
        r = np.asarray(v["vessels"][nm]["radius_mm"], float)
        if np.linalg.norm(P[-1] - O_v) < np.linalg.norm(P[0] - O_v) and False:
            P, r = P[::-1], r[::-1]
        info = {"src": "sources.load_vmr0248 vessels %s (expert SimVascular path); origin = its first point" % nm}
        origin_how = "expert path start"
        case_dir = os.path.join(DER, "vmr0248")
    else:
        case_dir = os.path.join(DER, "thin_slice", src, case)
        P, r, info = arterial_chain(case_dir, side, O_v)
        origin_how = "Arterial chain end node touching %s" % (info.get("origin_touches") if isinstance(info, dict) else None)
        if P is None:
            out["arterial_fail"] = info
            # fallback: record subclavian (+-40 mm, medial -> lateral) + TS origin
            sub = np.asarray(rec["subclavian"]["points"], float)
            sr = np.asarray(rec["subclavian"]["radii"], float)
            og, how = ts_origin(case_dir, side)
            if og is None:
                out["status"] = "no_path"
                out["why"] = "Arterial: %s; TS: %s" % (info, how)
                return out
            # prepend a straight run from the TS origin to the record path start when the origin lies proximal
            if np.linalg.norm(og - sub[0]) > 1.0:
                n = max(int(np.ceil(np.linalg.norm(og - sub[0]) / 0.5)), 1)
                pre = og + np.linspace(0, 1, n + 1)[:-1, None] * (sub[0] - og)
                P = np.vstack([pre, sub])
                r = np.concatenate([np.full(len(pre), sr[0]), sr])
            else:
                P, r = sub, sr
            info = {"src": "record subclavian (+-40 mm) + TS origin", "ts": how}
            origin_how = "TS mask contact"
    P, r = resample(P, r, 0.5)
    foot, d_ax, s_o = foot_on(P, O_v)
    s = arclen(P)
    keep = s <= s_o + KEEP_PAST_MM
    P, r, s = P[keep], r[keep], s[keep]
    out.update({"status": "ok", "path_info": info, "origin_how": origin_how,
                "path": P.round(3), "radius_path": r.round(3), "s_ostium_mm": s_o, "Ov_to_path_mm": d_ax,
                "origin": P[0].round(3), "path_len_mm": float(s[-1]),
                "past_ostium_mm": float(s[-1] - s_o)})
    # TS origin as a cross-check (thin-slice only)
    if src != "vmr0248":
        og, how = ts_origin(case_dir, side)
        if og is not None:
            fo, dd, so = foot_on(P, og)
            out["ts_origin"] = {"point": og.round(2), "dist_to_path_mm": dd, "s_on_path_mm": so, "how": how}
    # want
    lo, hi = WANT_PROX_MM, s_o + WANT_DIST_MM
    out["want_mm"] = [lo, hi]
    out["origin_to_ostium_mm"] = s_o
    # what the v5 ct_seg already holds for this subclavian
    if seg_dir and os.path.isfile(os.path.join(seg_dir, "stations_rays.json")):
        st = jload(os.path.join(seg_dir, "stations_rays.json"))
        key = "rsub" if side == "R" else "lsub"
        if key in st:
            Q = np.asarray(st[key]["P"], float)
            s_on = np.array([foot_on(P, q)[2] for q in (Q[0], Q[-1])])
            dq = np.array([foot_on(P, q)[1] for q in Q[:: max(1, len(Q) // 20)]])
            out["ct_seg_v5_sub"] = {"stations": len(Q), "len_mm": float(st[key]["s"][-1]),
                                    "covers_s_mm": [float(s_on.min()), float(s_on.max())],
                                    "off_path_mm_med_max": [float(np.median(dq)), float(dq.max())],
                                    "covers_want": bool(s_on.min() <= lo + 1.0 and s_on.max() >= hi - 1.0)}
        else:
            out["ct_seg_v5_sub"] = {"stations": 0}
    try:
        out["fov"] = ct_fov(src, case)
        if out["fov"] and "lo" in out["fov"]:
            lo3, hi3 = np.asarray(out["fov"]["lo"]), np.asarray(out["fov"]["hi"])
            inside = np.all((P >= lo3 - 1e-6) & (P <= hi3 + 1e-6), axis=1)
            m = (s >= lo) & (s <= hi)
            out["fov"]["want_inside_frac"] = float(inside[m].mean()) if m.any() else None
            out["fov"]["origin_inside"] = bool(inside[0])
        elif out["fov"]:
            zlo, zhi = out["fov"]["z_mm"]
            m = (s >= lo) & (s <= hi)
            inside = (P[:, 2] >= zlo) & (P[:, 2] <= zhi)
            out["fov"]["want_inside_frac"] = float(inside[m].mean()) if m.any() else None
            out["fov"]["origin_inside"] = bool(inside[0])
    except Exception as e:  # noqa: BLE001
        out["fov"] = {"error": str(e)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    units = sorted(d for d in os.listdir(DONORS) if os.path.isfile(os.path.join(DONORS, d, "record.json")))
    only = os.environ.get("ONLY")
    only = set(only.split(",")) if only else None
    summ = {}
    for u in units:
        ur = jload(os.path.join(DONORS, u, "record.json"))
        for side in ("R", "L"):
            sd = (ur.get("sides") or {}).get(side) or {}
            if only and "%s_%s" % (u, side) not in only:
                continue
            if not sd.get("built"):
                summ["%s_%s" % (u, side)] = {"status": "side not built"}
                continue
            try:
                o = one(u, side)
            except Exception as e:  # noqa: BLE001
                import traceback
                o = {"donor_id": u, "side": side, "status": "error", "why": "%s: %s" % (type(e).__name__, e),
                     "trace": traceback.format_exc()[-800:]}
            o["usable_v5"] = sd.get("usable")
            o["eligible_class"] = ur.get("eligible_class")
            jdump(o, os.path.join(OUT, "%s_%s.json" % (u, side)))
            row = {k: o.get(k) for k in ("status", "origin_how", "origin_to_ostium_mm", "past_ostium_mm", "Ov_to_path_mm",
                                         "eligible_class", "usable_v5", "why")}
            row["ct_seg_v5_sub"] = o.get("ct_seg_v5_sub")
            row["fov"] = {k: (o.get("fov") or {}).get(k) for k in ("want_inside_frac", "origin_inside")}
            row["ts_origin_s"] = (o.get("ts_origin") or {}).get("s_on_path_mm")
            row["ts_origin_dist"] = (o.get("ts_origin") or {}).get("dist_to_path_mm")
            summ["%s_%s" % (u, side)] = row
            print("%-22s %s %-6s orig->ost %6s past %6s  v5sub %s  fov %s  ts_s %s" % (
                u, side, o.get("status"), ("%.1f" % o["origin_to_ostium_mm"]) if o.get("origin_to_ostium_mm") is not None else "-",
                ("%.1f" % o["past_ostium_mm"]) if o.get("past_ostium_mm") is not None else "-",
                (o.get("ct_seg_v5_sub") or {}).get("covers_s_mm"), row["fov"],
                ("%.1f/%.1f" % (row["ts_origin_s"], row["ts_origin_dist"])) if row["ts_origin_s"] is not None else "-"), flush=True)
    if not only:
        jdump(summ, os.path.join(OUT, "_summary.json"))


if __name__ == "__main__":
    main()
