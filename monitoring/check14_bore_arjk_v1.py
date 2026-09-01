"""CHECK 14 -- declared radius (branch.radii) vs exact measured bore, 216 three-source set.

HANDOFF 11.3: exact vtkImplicitPolyDataDistance, centerline densified to <=0.25 mm.
HANDOFF 11.2: sign is NOT assumed -- a known-good control (topcow_mr_001, 49-set) runs
through identical code in the same process, and the host runs too when reachable.
HANDOFF 11.1: cohort .obj files are frame-consistent with their own centerlines;
only the host needs the FromMesh accessor.

Segments along the RCCA route (composed arclength from the ostium, s=0):
    host    [0, host_cut)          shipped host arch; radii ramped over the last
                                   blend_mm before host_cut (graft_three.ramp_radius)
    lower   [host_cut, 130)        donor CCA + ICA, floored at ROUTE_MIN_R=1.60
    siphon  [130, total]           TopBrain siphon, radii untouched
host_cut and blend_mm come from each anatomy's provenance.json; the siphon seam is
pinned at SIPHON_SEAM_MM = 130.0 in carotid_tools/graft_three.py.
"""
import glob, json, os, sys
import numpy as np, pyvista as pv, vtk

sys.path.insert(0, "/opt/eve_training/eve"); sys.path.insert(0, "/opt/eve_training/eve_bench")
from eve_bench.dualdevicenav import load_branches

STEP = 0.25
SEAM = 130.0


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def densify(p, r, step=STEP):
    s = arclen(p)
    n = int(np.ceil(s[-1] / step)) + 1
    t = np.linspace(0.0, s[-1], max(n, 2))
    q = np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)
    rr = np.interp(t, s, r)
    return q, rr, t


def probe(mesh_path, cl_dir, wants):
    m = pv.read(mesh_path).triangulate().clean()
    f = vtk.vtkImplicitPolyDataDistance(); f.SetInput(m)
    brs = {str(b.name): b for b in load_branches(cl_dir)}
    out = {}
    for want in wants:
        hit = [k for k in brs if want in k.upper()]
        if not hit:
            out[want] = None; continue
        b = brs[hit[0]]
        p = np.asarray(b.coordinates, float)
        r = np.asarray(b.radii, float) if getattr(b, "radii", None) is not None else None
        if r is None or len(r) != len(p):
            out[want] = None; continue
        q, rr, t = densify(p, r)
        sd = np.array([f.EvaluateFunction(pt) for pt in q])  # POSITIVE = inside, verified
        out[want] = dict(s=t, clear=sd, decl=rr, n_native=len(p))
    return out


def q(a):
    a = np.asarray(a, float)
    return (np.percentile(a, 5), np.percentile(a, 25), np.median(a),
            np.percentile(a, 75), np.percentile(a, 95))


def fq(a, w=7, d=3):
    return "  ".join(("%*.*f" % (w, d, v)) for v in q(a))


print("=== CONTROLS (sign + reproduction of the published 49-set / host figures) ===")
CTRL_DIR = "/opt/eve_training/results_topbrain/anatomies/topcow_mr_001"
c = probe(os.path.join(CTRL_DIR, "vessel_architecture_collision.obj"),
          os.path.join(CTRL_DIR, "Centrelines_comb"), ["RCCA"])["RCCA"]
d_ctrl = c["clear"] - c["decl"]
print("49-set control topcow_mr_001 RCCA: n_dense=%d  clearance med %.3f  min %.3f  outside %d"
      % (len(c["s"]), np.median(c["clear"]), c["clear"].min(), (c["clear"] < 0).sum()))
print("   SIGN: median clearance %s 0 -> %s"
      % (">" if np.median(c["clear"]) > 0 else "<",
         "positive-inside CONFIRMED" if np.median(c["clear"]) > 0 else "INVERTED, ABORT"))
print("   median(clearance - declared r) = %+.3f mm   (HANDOFF 11.5 reports -0.75 on the 49-set)"
      % np.median(d_ctrl))

try:
    from eve_bench.dualdevicenav import DualDeviceNav
    dv = DualDeviceNav()
    hm = dv.vessel_tree.mesh_path
    hbr = {str(b.name): b for b in dv.vessel_tree.branches}
    hk = [k for k in hbr if "RCCA" in k.upper()]
    mh = pv.read(hm).triangulate().clean()
    fh = vtk.vtkImplicitPolyDataDistance(); fh.SetInput(mh)
    hb = hbr[hk[0]]
    hp = np.asarray(hb.coordinates, float); hr = np.asarray(hb.radii, float)
    hq, hrr, ht = densify(hp, hr)
    hsd = np.array([fh.EvaluateFunction(pt) for pt in hq])
    print("HOST control (FromMesh accessor) RCCA: clearance med %.3f  min %.3f  outside %d"
          % (np.median(hsd), hsd.min(), (hsd < 0).sum()))
    print("   median(clearance - declared r) = %+.3f mm   (HANDOFF 11.5 reports +0.38 on the host)"
          % np.median(hsd - hrr))
except Exception as e:
    print("HOST control unavailable: %s: %s" % (type(e).__name__, e))
print()

rows = []
LIMIT = int(os.environ.get("C14_LIMIT", "0"))
dirs = sorted(glob.glob("/opt/eve_training/carotid/anatomies/*"))
if LIMIT:
    dirs = dirs[:LIMIT]
for d in dirs:
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    try:
        prov = json.load(open(os.path.join(d, "provenance.json")))
        res = probe(os.path.join(d, "vessel_architecture_collision.obj"),
                    os.path.join(d, "Centrelines_comb"), ["RCCA", "RECA"])
        rows.append((name, prov, res))
    except Exception as e:
        print("  FAIL %s: %s: %s" % (name, type(e).__name__, e))

print("loaded %d anatomies" % len(rows))
out = {}
for name, prov, res in rows:
    rec = {"host_cut": prov["host_cut_mm"], "blend": prov["blend_mm"],
           "total": prov["total_mm"], "eca_floor": prov["eca_floored_frac"],
           "route_floor": prov["route_floored_frac"], "min_r_raw": prov["route_min_r_raw"]}
    a = res.get("RCCA")
    if a is not None:
        s, dl = a["s"], a["clear"] - a["decl"]
        hc = prov["host_cut_mm"]; bl = prov["blend_mm"]
        seg = {
            "all":    dl,
            "host":   dl[s < hc],
            "flare":  dl[s < 10.0],
            "hostmid": dl[(s >= 10.0) & (s < hc - bl)],
            "ramp":   dl[(s >= hc - bl) & (s < hc)],
            "lower":  dl[(s >= hc) & (s < SEAM)],
            "cca":    dl[(s >= hc) & (s < hc + prov["cca_mm"])],
            "ica":    dl[(s >= hc + prov["cca_mm"]) & (s < SEAM)],
            "siphon": dl[s >= SEAM],
        }
        rec["rcca"] = {k: (float(np.median(v)) if len(v) else None) for k, v in seg.items()}
        rec["rcca_n"] = {k: int(len(v)) for k, v in seg.items()}
        rec["rcca_min"] = {k: (float(np.min(v)) if len(v) else None) for k, v in seg.items()}
        rec["rcca_clear_med"] = float(np.median(a["clear"]))
        rec["rcca_out"] = int((a["clear"] < 0).sum())
        rec["rcca_declmed"] = float(np.median(a["decl"]))
        rec["rcca_frac_opt"] = float(np.mean(dl < 0))
        # absolute calibre per segment, so the ratio can be read too
        rec["rcca_cl"] = {k: (float(np.median(a["clear"][mask])) if mask.sum() else None)
                          for k, mask in [("hostmid", (s >= 10.0) & (s < hc - bl)),
                                          ("lower", (s >= hc) & (s < SEAM)),
                                          ("siphon", s >= SEAM)]}
        rec["rcca_dr"] = {k: (float(np.median(a["decl"][mask])) if mask.sum() else None)
                          for k, mask in [("hostmid", (s >= 10.0) & (s < hc - bl)),
                                          ("lower", (s >= hc) & (s < SEAM)),
                                          ("siphon", s >= SEAM)]}
        # device fit: catheter radius 0.35 + SOFA contactDistance 0.30
        rec["rcca_tight"] = float(np.mean(a["clear"] < 0.65))
        rec["rcca_clear_min"] = float(a["clear"].min())
    b = res.get("RECA")
    if b is not None:
        dlb = b["clear"] - b["decl"]
        rec["reca"] = float(np.median(dlb))
        rec["reca_min"] = float(np.min(dlb))
        rec["reca_clear_med"] = float(np.median(b["clear"]))
        rec["reca_declmed"] = float(np.median(b["decl"]))
        rec["reca_frac_opt"] = float(np.mean(dlb < 0))
    out[name] = rec

if os.path.isdir("/out"):
    json.dump(out, open("/out/check14_bore_arjk.json", "w"))

print()
print("=== PER-ANATOMY median(exact clearance - declared radius), mm ===")
print("%-28s %8s %8s %8s %8s %8s" % ("scope", "p5", "p25", "median", "p75", "p95"))
for lab, key in [("RCCA whole route", "all"), ("RCCA host segment", "host"),
                 ("  ostium flare s<10", "flare"), ("  host mid (flare..ramp)", "hostmid"),
                 ("  radius ramp window", "ramp"), ("RCCA lower (donor)", "lower"),
                 ("  lower CCA", "cca"), ("  lower ICA", "ica"),
                 ("RCCA siphon (s>=130)", "siphon")]:
    v = [r["rcca"][key] for r in out.values() if r.get("rcca") and r["rcca"][key] is not None]
    print("%-28s %s   (n=%d)" % (lab, fq(v, 8), len(v)))
v = [r["reca"] for r in out.values() if "reca" in r]
print("%-28s %s   (n=%d)" % ("RECA (ECA fork)", fq(v, 8), len(v)))

print()
print("=== PAIRED WITHIN-ROUTE CONTRAST: does the discrepancy differ along the route? ===")


def paired(k1, k2, lab):
    d = [r["rcca"][k1] - r["rcca"][k2] for r in out.values()
         if r.get("rcca") and r["rcca"][k1] is not None and r["rcca"][k2] is not None]
    d = np.array(d)
    print("%-34s %s  | n=%d  same-sign-as-median %d (%.0f%%)  |  >0: %d  <0: %d"
          % (lab, fq(d, 8), len(d),
             int((np.sign(d) == np.sign(np.median(d))).sum()),
             100 * np.mean(np.sign(d) == np.sign(np.median(d))),
             int((d > 0).sum()), int((d < 0).sum())))
    return d


print("%-34s %8s %8s %8s %8s %8s" % ("contrast (per anatomy)", "p5", "p25", "median", "p75", "p95"))
d_sl = paired("siphon", "lower", "siphon - lower")
paired("siphon", "hostmid", "siphon - host mid")
paired("lower", "hostmid", "lower  - host mid")
paired("ica", "cca", "lower ICA - lower CCA")

print()
print("sign disagreement WITHIN a route (segment medians straddle zero):")
n_str = 0; examples = []
for n, r in out.items():
    if not r.get("rcca"):
        continue
    vals = [r["rcca"][k] for k in ("hostmid", "lower", "siphon") if r["rcca"][k] is not None]
    if len(vals) == 3 and (min(vals) < 0 < max(vals)):
        n_str += 1; examples.append((n, vals, max(vals) - min(vals)))
print("  %d / %d routes have >=1 segment pessimistic and >=1 optimistic" % (n_str, len(out)))
examples.sort(key=lambda x: -x[2])
for n, vals, sp in examples[:6]:
    print("    %-46s hostmid %+.2f  lower %+.2f  siphon %+.2f  (span %.2f)"
          % (n, vals[0], vals[1], vals[2], sp))

print()
print("within-route SPAN of segment medians (max-min over hostmid/lower/siphon), mm:")
spans = np.array([max(v) - min(v) for v in
                  ([[r["rcca"][k] for k in ("hostmid", "lower", "siphon")] for r in out.values()
                    if r.get("rcca") and all(r["rcca"][k] is not None
                                             for k in ("hostmid", "lower", "siphon"))])])
print("   %s   (n=%d)" % (fq(spans, 8), len(spans)))

print()
print("=== OPTIMISM FRACTION (share of densified points where declared r > exact bore) ===")
for lab, key in [("RCCA route", "rcca_frac_opt"), ("RECA fork", "reca_frac_opt")]:
    v = [r[key] for r in out.values() if key in r]
    print("%-14s %s   (n=%d)" % (lab, fq(v, 8), len(v)))

print()
print("=== OUTLIERS ===")
srt = sorted([(r["rcca"]["all"], n) for n, r in out.items() if r.get("rcca")])
print("most OPTIMISTIC whole-route (declared r most overstates bore):")
for v, n in srt[:6]:
    print("   %-46s %+.3f mm   clear_med %.2f  decl_med %.2f"
          % (n, v, out[n]["rcca_clear_med"], out[n]["rcca_declmed"]))
print("most PESSIMISTIC whole-route:")
for v, n in srt[-4:][::-1]:
    print("   %-46s %+.3f mm   clear_med %.2f  decl_med %.2f"
          % (n, v, out[n]["rcca_clear_med"], out[n]["rcca_declmed"]))
srt2 = sorted([(r["rcca"]["siphon"] - r["rcca"]["lower"], n) for n, r in out.items()
               if r.get("rcca") and r["rcca"]["siphon"] is not None
               and r["rcca"]["lower"] is not None])
print("largest siphon-vs-lower DISAGREEMENT (siphon more optimistic):")
for v, n in srt2[:5]:
    print("   %-46s %+.3f mm  (lower %+.3f  siphon %+.3f)"
          % (n, v, out[n]["rcca"]["lower"], out[n]["rcca"]["siphon"]))
print("largest siphon-vs-lower DISAGREEMENT (lower more optimistic):")
for v, n in srt2[-5:][::-1]:
    print("   %-46s %+.3f mm  (lower %+.3f  siphon %+.3f)"
          % (n, v, out[n]["rcca"]["lower"], out[n]["rcca"]["siphon"]))
srt3 = sorted([(r["reca"], n) for n, r in out.items() if "reca" in r])
print("RECA most optimistic:")
for v, n in srt3[:5]:
    print("   %-46s %+.3f mm   clear_med %.2f  decl_med %.2f"
          % (n, v, out[n]["reca_clear_med"], out[n]["reca_declmed"]))

print()
print("=== worst-case (minimum, not median) declared-vs-bore error per segment, mm ===")
print("%-24s %8s %8s %8s %8s %8s" % ("segment", "p5", "p25", "median", "p75", "p95"))
for lab, key in [("RCCA all", "all"), ("host mid", "hostmid"), ("ramp", "ramp"),
                 ("lower", "lower"), ("siphon", "siphon")]:
    v = [r["rcca_min"][key] for r in out.values()
         if r.get("rcca_min") and r["rcca_min"][key] is not None]
    print("%-24s %s   (n=%d)" % (lab, fq(v, 8), len(v)))
v = [r["reca_min"] for r in out.values() if "reca_min" in r]
print("%-24s %s   (n=%d)" % ("RECA", fq(v, 8), len(v)))

print()
print("=== does the floor explain it? corr(route_floored_frac, lower median delta) ===")
xs = np.array([r["route_floor"] for r in out.values() if r.get("rcca")])
ys = np.array([r["rcca"]["lower"] for r in out.values() if r.get("rcca")])
print("   route_floored_frac: median %.3f  max %.3f   pearson r = %.3f"
      % (np.median(xs), xs.max(), np.corrcoef(xs, ys)[0, 1]))
xe = np.array([r["eca_floor"] for r in out.values() if "reca" in r])
ye = np.array([r["reca"] for r in out.values() if "reca" in r])
print("   eca_floored_frac  : median %.3f  max %.3f   pearson r = %.3f (vs RECA delta)"
      % (np.median(xe), xe.max(), np.corrcoef(xe, ye)[0, 1]))

print()
print("=== ABSOLUTE CALIBRE per segment (median over anatomies of the per-anatomy median) ===")
print("%-10s %10s %10s %10s" % ("segment", "exact bore", "declared r", "bore/decl"))
for k in ("hostmid", "lower", "siphon"):
    cl = np.array([r["rcca_cl"][k] for r in out.values()
                   if r.get("rcca_cl") and r["rcca_cl"][k] is not None])
    dr = np.array([r["rcca_dr"][k] for r in out.values()
                   if r.get("rcca_dr") and r["rcca_dr"][k] is not None])
    print("%-10s %10.3f %10.3f %10.3f" % (k, np.median(cl), np.median(dr),
                                          np.median(cl / dr)))
rc = np.array([r["reca_clear_med"] for r in out.values() if "reca_clear_med" in r])
rd = np.array([r["reca_declmed"] for r in out.values() if "reca_declmed" in r])
print("%-10s %10.3f %10.3f %10.3f" % ("RECA", np.median(rc), np.median(rd),
                                      np.median(rc / rd)))

print()
print("=== DEVICE FIT: fraction of route below catheter r 0.35 + contactDistance 0.30 ===")
t = np.array([r["rcca_tight"] for r in out.values() if "rcca_tight" in r])
mn = np.array([r["rcca_clear_min"] for r in out.values() if "rcca_clear_min" in r])
print("   frac(clearance < 0.65 mm): %s" % fq(t, 8))
print("   min clearance on route   : %s" % fq(mn, 8))
print("   anatomies with any point < 0.65 mm: %d / %d" % (int((t > 0).sum()), len(t)))

print()
print("=== VARIANCE: is the siphon-vs-lower gap a donor property? ===")


def donor_split(name):
    lo, si = name.split("__")
    return lo, si


for key, lab in [("siphon", "siphon segment delta"), ("lower", "lower segment delta")]:
    vals = {n: r["rcca"][key] for n, r in out.items()
            if r.get("rcca") and r["rcca"][key] is not None}
    by_lo, by_si = {}, {}
    for n, v in vals.items():
        lo, si = donor_split(n)
        by_lo.setdefault(lo, []).append(v)
        by_si.setdefault(si, []).append(v)
    allv = np.array(list(vals.values()))
    ml = np.array([np.mean(v) for v in by_lo.values()])
    ms = np.array([np.mean(v) for v in by_si.values()])
    print("%-22s total sd %.3f | between-LOWER-donor sd %.3f (%d donors) | "
          "between-SIPHON-donor sd %.3f (%d donors)"
          % (lab, allv.std(), ml.std(), len(ml), ms.std(), len(ms)))

print()
print("worst SIPHON donors by median delta (across the anatomies using them):")
bysi = {}
for n, r in out.items():
    if r.get("rcca") and r["rcca"]["siphon"] is not None:
        bysi.setdefault(donor_split(n)[1], []).append(r["rcca"]["siphon"])
ranked = sorted(((np.median(v), k, len(v)) for k, v in bysi.items()))
for v, k, c in ranked[:5]:
    print("   %-22s %+.3f mm  (n=%d)" % (k, v, c))
print("best SIPHON donors:")
for v, k, c in ranked[-3:][::-1]:
    print("   %-22s %+.3f mm  (n=%d)" % (k, v, c))
bylo = {}
for n, r in out.items():
    if r.get("rcca") and r["rcca"]["lower"] is not None:
        bylo.setdefault(donor_split(n)[0], []).append(r["rcca"]["lower"])
rl = sorted(((np.median(v), k, len(v)) for k, v in bylo.items()))
print("worst LOWER donors by median lower-segment delta:")
for v, k, c in rl[:5]:
    print("   %-22s %+.3f mm  (n=%d)" % (k, v, c))
print("best LOWER donors:")
for v, k, c in rl[-3:][::-1]:
    print("   %-22s %+.3f mm  (n=%d)" % (k, v, c))
